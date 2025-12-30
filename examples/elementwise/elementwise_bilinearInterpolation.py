import argparse

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

src0 = torch.arange(1, 513, dtype=torch.float16).reshape(1, -1).npu()
src0offset_int = torch.arange(0, 1024, 32, dtype=torch.int64).reshape(1, -1).npu()
src0offset = src0offset_int.to(dtype=torch.uint32)
src1 = torch.arange(2, 18, dtype=torch.float16).reshape(1, -1).npu()

hRepeat = 2
mask1 = 128
repeatMode = False
dstBlkStride = 1
vROffset = 128
vRepeat = 2
mask0 = 0

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def bilinear_interpolation(mask, h_repeat, repeat_mode,
                           dst_blk_stride, v_r_offset, v_repeat):
    m_num = 1
    n_num = 1

    VEC_NUM = 1

    @T.prim_func
    def main(
            src0: T.Tensor((src0.shape[0], src0.shape[1]), "float16"),
            src0_offset: T.Tensor((src0offset.shape[0], src0offset.shape[1]), "uint32"),
            src1: T.Tensor((src1.shape[0], src1.shape[1]), "float16"),
            dst: T.Tensor((src0.shape[0], src0.shape[1] // 2), "float16"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            src0_ub = T.alloc_ub((src0.shape[0] // VEC_NUM, src0.shape[1]), "float16")
            src0_offset_ub = T.alloc_ub((src0offset.shape[0] // VEC_NUM, src0offset.shape[1]), "uint32")
            src1_ub = T.alloc_ub((src1.shape[0] // VEC_NUM, src1.shape[1]), "float16")
            dst_ub = T.alloc_ub((src0.shape[0] // VEC_NUM, src0.shape[1] // 2), "float16")
            shared_tmp_buffer_ub = T.alloc_ub((src0.shape[0], src0.shape[1]), "uint8")

            T.copy(src0[0, 0], src0_ub)
            T.copy(src0_offset[0, 0], src0_offset_ub)
            T.copy(src1[0, 0], src1_ub)

            T.tile.bilinear_interpolation(dst_ub, src0_ub, src0_offset_ub, src1_ub, mask, h_repeat,
                                        repeat_mode, dst_blk_stride, v_r_offset, v_repeat, shared_tmp_buffer_ub)

            T.copy(dst_ub, dst[0, 0])

    return main


func = bilinear_interpolation(mask1, hRepeat, repeatMode, dstBlkStride, vROffset, vRepeat)

torch.manual_seed(0)

torch.npu.synchronize()
print("init successful!")

c = func(src0, src0offset, src1)

# cal_ref_c

def fun_ref(a, b, c, hRepeat, vRepeat, repeatMode, vROffset):
    a = a.flatten()
    b = b.flatten()
    c = c.flatten()
    re = []
    
    if repeatMode:
        for k in range(vRepeat):
            s = torch.zeros(128, dtype=torch.float16).npu()  #初始化累加器
            r = torch.zeros(128 * hRepeat, dtype=torch.float16).npu()
            for i in range(hRepeat):
                for j in range(8):
                    idx = b[k * 8 * hRepeat + i * 8 + j].to(torch.int64) // 32
                    r[i * 128 + j * 16 : i * 128 + (j+1) * 16] = a[idx * 16 : (idx + 1) * 16] * c[k * 8 * hRepeat + i * 8 + j]
                s += r[i * 128 : (i + 1) * 128]
            re.append(s)
    else:
        for k in range(vRepeat):
            s = torch.zeros(128, dtype=torch.float16).npu()
            r = torch.zeros(128 * hRepeat, dtype=torch.float16).npu()
            for i in range(hRepeat):
                for j in range(8):
                    idx = b[k * 8 * hRepeat + i * 8 + j].to(torch.int64) // 32
                    r[i * 128 + j * 16 : i * 128 + (j + 1) * 16] = a[idx * 16 : (idx + 1) * 16] * c[k * hRepeat + i]
                s += r[i * 128 : (i + 1) * 128]
            re.append(s)
    return torch.cat(re, dim=0).flatten()

if vROffset > 128:
    outsize = vRepeat * vROffset
else:
    outsize = vRepeat * 128

out = fun_ref(src0, src0offset, src1, hRepeat, vRepeat, repeatMode, vROffset)

out_real = torch.zeros(outsize, dtype=torch.float16).npu()
if mask0 == 0:
    for i in range(vRepeat):
        n = mask1 // 16
        l = mask1 % 16
        for j in range(n):
            out_real[i * vROffset + j * 16 : i * vROffset + (j + 1) * 16] = out[i * 128 + j * 16 : i * 128 + (j + 1) * 16]
        out_real[i * vROffset + n * 16 : i * vROffset + n * 16 + l] = out[i * 128 + n * 16 : i * 128 + n * 16 + l]

ref_c = out_real[:vRepeat * 128].unsqueeze(0)
                    
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")