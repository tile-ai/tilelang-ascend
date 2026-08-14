"""Test fp32 reciprocal sigmoid: 1/(1+exp(-x)) = Muls+Exp+Adds+Reciprocal = 4 V ops."""

import torch
import tilelang
from tilelang import language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def sigmoid_recip_fp32(M, N, block_M, block_N, dtype="float"):
    """fp32 reciprocal sigmoid: Muls(-1)→Exp→Adds(1)→Reciprocal = 4 V ops."""
    m_num = (M + block_M - 1) // block_M
    n_num = (N + block_N - 1) // block_N
    VEC_NUM = 2
    rpv = block_M // VEC_NUM

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a = T.alloc_shared((rpv, block_N), dtype)
            b = T.alloc_shared((rpv, block_N), dtype)
            T.copy(A[bx * block_M + vid * rpv, by * block_N], a)
            T.tile.mul(a, a, -1.0)
            T.tile.exp(a, a)
            T.tile.add(a, a, 1.0)
            T.tile.reciprocal(b, a)
            T.copy(b, B[bx * block_M + vid * rpv, by * block_N])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def sigmoid_recip_fp16(M, N, block_M, block_N, dtype="float16"):
    """fp16 via fp32: cast→Muls→Exp→Adds→Reciprocal→cast = 6 V ops."""
    m_num = (M + block_M - 1) // block_M
    n_num = (N + block_N - 1) // block_N
    VEC_NUM = 2
    rpv = block_M // VEC_NUM
    elem_num = rpv * block_N
    ACC = "float32"
    CAST_UP = "CAST_NONE"
    CAST_DOWN = "CAST_RINT"

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            tmp_in = T.alloc_shared((rpv, block_N), dtype)
            a_ub = T.alloc_shared((rpv, block_N), ACC)
            tmp_out = T.alloc_shared((rpv, block_N), dtype)
            T.copy(A[bx * block_M + vid * rpv, by * block_N], tmp_in)
            T.tile.cast(a_ub, tmp_in, CAST_UP, elem_num)
            T.tile.mul(a_ub, a_ub, -1.0)
            T.tile.exp(a_ub, a_ub)
            T.tile.add(a_ub, a_ub, 1.0)
            T.tile.reciprocal(a_ub, a_ub)
            T.tile.cast(tmp_out, a_ub, CAST_DOWN, elem_num)
            T.copy(tmp_out, B[bx * block_M + vid * rpv, by * block_N])

    return main


torch.manual_seed(0)

# Test fp32
fn32 = sigmoid_recip_fp32(8192, 8192, 128, 128, dtype="float")
x32 = torch.randn(8192, 8192, dtype=torch.float32).npu()
y32 = fn32(x32)
ref32 = torch.sigmoid(x32)
y32c, r32c = y32.cpu().float(), ref32.cpu().float()
m32 = torch.isfinite(r32c)
err32 = (y32c[m32] - r32c[m32]).abs()
rel32 = err32 / (r32c[m32].abs() + 1e-7)
print(
    f"fp32 reciprocal: MERE={rel32.mean().item():.8f} MARE={rel32.max().item():.8f} -> {'PASS' if rel32.mean().item() < 2**-13 and rel32.max().item() < 10 * 2**-13 else 'FAIL'}",
    flush=True,
)

# Test fp32 with extreme values
x32b = torch.empty(8192, 8192, dtype=torch.float32).uniform_(-100, 100).npu()
y32b = fn32(x32b)
ref32b = torch.sigmoid(x32b)
y32bc, r32bc = y32b.cpu().float(), ref32b.cpu().float()
m32b = torch.isfinite(r32bc)
err32b = (y32bc[m32b] - r32bc[m32b]).abs()
rel32b = err32b / (r32bc[m32b].abs() + 1e-7)
print(
    f"fp32 [-100,100] reciprocal: MERE={rel32b.mean().item():.8f} MARE={rel32b.max().item():.8f} -> {'PASS' if rel32b.mean().item() < 2**-13 and rel32b.max().item() < 10 * 2**-13 else 'FAIL'}",
    flush=True,
)

# Test fp16 via fp32
fn16 = sigmoid_recip_fp16(8192, 8192, 128, 128, dtype="float16")
x16 = torch.randn(8192, 8192, dtype=torch.float16).npu()
y16 = fn16(x16)
ref16 = torch.sigmoid(x16)
y16c, r16c = y16.cpu().float(), ref16.cpu().float()
m16 = torch.isfinite(r16c)
err16 = (y16c[m16] - r16c[m16]).abs()
rel16 = err16 / (r16c[m16].abs() + 1e-7)
print(
    f"fp16 via fp32 reciprocal: MERE={rel16.mean().item():.8f} MARE={rel16.max().item():.8f} -> {'PASS' if rel16.mean().item() < 2**-10 and rel16.max().item() < 10 * 2**-10 else 'FAIL'}",
    flush=True,
)

print("done", flush=True)
