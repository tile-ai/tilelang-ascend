import tilelang
from tilelang import DataType, language as T
import torch

torch.set_default_device('npu')
torch.manual_seed(0)

tilelang.disable_cache()

stages = 2
B, S, H, D = 1, 128, 1, 512

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}

@tilelang.jit(out_idx=[3], workspace_idx=[4,5,6], pass_configs=pass_configs)
def flash_attention_fwd(
    heads,
    dim,
):
    block_M, block_N = 32, 64

    batch = B
    seq_len = S

    dtype = "float16"
    accum_dtype = "float"

    sm_scale = (1.0 / dim)**0.5

    shape = [batch, heads, seq_len, dim]

    block_num = seq_len // block_M * heads * batch

    @T.prim_func
    def main(
            Q: T.Tensor(shape, dtype),  # type: ignore
            K: T.Tensor(shape, dtype),  # type: ignore
            V: T.Tensor(shape, dtype),  # type: ignore
            Output: T.Tensor(shape, dtype),  # type: ignore
            workspace_1: T.Tensor([block_num, block_M, block_N], accum_dtype),
            workspace_2: T.Tensor([block_num, block_M, block_N], dtype),
            workspace_3: T.Tensor([block_num, block_M, dim], accum_dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len // block_M)
            by = cid // (seq_len // block_M) % heads
            bz = cid // (seq_len // block_M) // heads % batch

            q_l1 = T.alloc_shared([block_M, dim], dtype)
            k_l1 = T.alloc_shared([block_N, dim], dtype)
            v_l1 = T.alloc_shared([block_N, dim], dtype)

            acc_s_l1 = T.alloc_shared([block_M, block_N], dtype)

            acc_s_l0c = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_o_l0c = T.alloc_fragment([block_M, dim], accum_dtype)

            acc_o = T.alloc_shared([block_M // 2, dim], accum_dtype)
            sumexp = T.alloc_shared([block_M // 2], accum_dtype)
            m_i = T.alloc_shared([block_M // 2], accum_dtype)

            acc_s_ub = T.alloc_shared([block_M // 2, block_N], accum_dtype)
            m_i_prev = T.alloc_shared([block_M // 2], accum_dtype)
            acc_s_ub_ = T.alloc_shared([block_M // 2, block_N], accum_dtype)
            tmp_ub = T.alloc_shared([3 * DataType(accum_dtype).bits // 8 * block_M // 2 * block_N],
                                "uint8")
            sumexp_i_ub = T.alloc_shared([block_M // 2], accum_dtype)
            acc_s_half = T.alloc_shared([block_M // 2, block_N], dtype)
            acc_o_ub = T.alloc_shared([block_M // 2, dim], accum_dtype)
            acc_o_half = T.alloc_shared([block_M // 2, dim], dtype)

            T.tile.fill(acc_o, 0.0)
            T.tile.fill(sumexp, 0.0)
            T.tile.fill(m_i, -2**30)
            T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], q_l1)

            for k in T.Pipelined(T.ceildiv(seq_len, block_N), num_stages=2):
                T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], k_l1)
                T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                T.copy(acc_s_l0c, workspace_1[cid, :, :])

                T.tile.fill(acc_s_ub, 0.0)
                T.copy(m_i, m_i_prev)
                T.copy(
                    workspace_1[cid, vid * block_M // 2:vid * block_M // 2 + block_M // 2, :],
                    acc_s_ub_)
                T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                T.tile.reduce_max(m_i, acc_s_ub, tmp_ub, dim=-1)
                T.tile.max(m_i, m_i, m_i_prev)
                T.tile.sub(m_i_prev, m_i_prev, m_i)
                T.tile.exp(m_i_prev, m_i_prev)
                for h_i in range(block_M // 2):
                    T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])
                T.tile.exp(acc_s_ub, acc_s_ub)
                T.tile.reduce_sum(sumexp_i_ub, acc_s_ub, tmp_ub, dim=-1)
                T.tile.mul(sumexp, sumexp, m_i_prev)
                T.tile.add(sumexp, sumexp, sumexp_i_ub)
                T.copy(acc_s_ub, acc_s_half)
                T.copy(
                    acc_s_half,
                    workspace_2[cid, vid * block_M // 2:vid * block_M // 2 + block_M // 2, :])
                
                T.copy(workspace_2[cid, :, :], acc_s_l1)
                T.copy(V[bz, by, k * block_N:(k + 1) * block_N, :], v_l1)
                T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                T.copy(acc_o_l0c, workspace_3[cid, :, :])

                for h_i in range(block_M // 2):
                    T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], m_i_prev[h_i])
                T.copy(
                    workspace_3[cid, vid * block_M // 2:vid * block_M // 2 + block_M // 2, :],
                    acc_o_ub)
                T.tile.add(acc_o, acc_o, acc_o_ub)

            for h_i in range(block_M // 2):
                T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])

            T.copy(acc_o, acc_o_half)
            T.copy(
                acc_o_half, Output[bz, by, bx * block_M + vid * block_M // 2:bx * block_M +
                                    vid * block_M // 2 + block_M // 2, :])

    return main


func = flash_attention_fwd(
    heads=H,
    dim=D,
)


def ref_flash_attn(q, k, v):
    q = q.float()
    k = k.float()
    v = v.float()

    acc = torch.einsum("bhsd,bhkd->bhsk", q, k) * (1.0 / q.shape[-1])**0.5
    acc = acc.softmax(dim=-1)
    o = torch.einsum("bhsk,bhkd->bhsd", acc, v)
    return o.to(torch.float16)

q = torch.randn((B, H, S, D), dtype=torch.float16)
k = torch.randn((B, H, S, D), dtype=torch.float16)
v = torch.randn((B, H, S, D), dtype=torch.float16)

torch.npu.synchronize()
print("init successful!")

output = func(q, k, v)
ref_output = ref_flash_attn(q, k, v)
torch.npu.synchronize()

torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)

from tilelang.profiler import do_bench

tilelang_time = do_bench(lambda: func(q, k, v))
print(tilelang_time)

print("Test Passed!")
