import tilelang
from tilelang import DataType, language as T
import torch

torch.set_default_device('npu')
torch.manual_seed(0)

tilelang.disable_cache()


@tilelang.jit(out_idx=[3],)
def flash_attention_fwd(
    heads,
    dim,
):
    block_M, block_N = 64, 64

    batch = 1
    seq_len = 128

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

            q_l1 = T.alloc_L1([block_M, dim], dtype)
            k_l1 = T.alloc_L1([block_N, dim], dtype)
            v_l1 = T.alloc_L1([block_N, dim], dtype)

            acc_s_l1 = T.alloc_L1([block_M, block_N], dtype)

            acc_s_l0c = T.alloc_L0C([block_M, block_N], accum_dtype)
            acc_o_l0c = T.alloc_L0C([block_M, dim], accum_dtype)

            acc_o = T.alloc_ub([block_M // 2, dim], accum_dtype)
            sumexp = T.alloc_ub([block_M // 2], accum_dtype)
            m_i = T.alloc_ub([block_M // 2], accum_dtype)

            acc_s_ub = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            m_i_prev = T.alloc_ub([block_M // 2], accum_dtype)
            acc_s_ub_ = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            tmp_ub = T.alloc_ub([3 * DataType(accum_dtype).bits // 8 * block_M // 2 * block_N],
                                "uint8")
            sumexp_i_ub = T.alloc_ub([block_M // 2], accum_dtype)
            acc_s_half = T.alloc_ub([block_M // 2, block_N], dtype)
            acc_o_ub = T.alloc_ub([block_M // 2, dim], accum_dtype)
            acc_o_half = T.alloc_ub([block_M // 2, dim], dtype)

            T.annotate_address({
                # L1 address
                q_l1: 0,
                k_l1: block_M * dim * DataType(dtype).bits // 8,
                acc_s_l1: block_M * dim * DataType(dtype).bits // 8,
                v_l1: block_M * (block_N + dim) * DataType(dtype).bits // 8,

                # L0C address
                acc_s_l0c: 0,
                acc_o_l0c: 0,

                ## ub address
                acc_o: 0,
                sumexp: 65536,
                m_i: 65664,
                acc_s_ub: 66048,
                m_i_prev: 74240,
                acc_s_ub_: 74368,
                tmp_ub: 74368,
                sumexp_i_ub: 98944,
                acc_s_half: 98944,
                acc_o_ub: 98944,
                acc_o_half: 98944
            })

            with T.Scope("C"):
                T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], q_l1)
                T.barrier_all()
                for k in T.serial(T.ceildiv(seq_len, block_N)):
                    T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], k_l1)
                    T.barrier_all()

                    T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                    T.barrier_all()

                    T.copy(acc_s_l0c, workspace_1[cid, :, :])
                    T.barrier_all()
                    T.set_cross_flag("FIX", 0)

                    T.wait_cross_flag(1)
                    T.barrier_all()

                    T.copy(workspace_2[cid, :, :], acc_s_l1)
                    T.copy(V[bz, by, k * block_N:(k + 1) * block_N, :], v_l1)
                    T.barrier_all()

                    T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                    T.barrier_all()

                    T.copy(acc_o_l0c, workspace_3[cid, :, :])
                    T.barrier_all()

                    T.set_cross_flag("FIX", 2)
                    T.wait_cross_flag(3)

            with T.Scope("V"):

                T.fill(acc_o, 0.0)
                T.fill(sumexp, 0.0)
                T.fill(m_i, -2**30)
                T.barrier_all()

                for _k in T.serial(T.ceildiv(seq_len, block_N)):
                    T.fill(acc_s_ub, 0.0)
                    T.barrier_all()

                    T.copy(m_i, m_i_prev)
                    T.barrier_all()

                    T.wait_cross_flag(0)
                    T.copy(
                        workspace_1[cid, vid * block_M // 2:vid * block_M // 2 + block_M // 2, :],
                        acc_s_ub_)
                    T.barrier_all()

                    T.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                    T.barrier_all()

                    T.mul(acc_s_ub, acc_s_ub, sm_scale)
                    T.barrier_all()

                    T.reduce_max(m_i, acc_s_ub, tmp_ub, dim=-1)
                    T.barrier_all()

                    T.max(m_i, m_i, m_i_prev)
                    T.barrier_all()

                    T.sub(m_i_prev, m_i_prev, m_i)
                    T.barrier_all()

                    T.exp(m_i_prev, m_i_prev)
                    T.barrier_all()

                    for h_i in range(block_M // 2):
                        T.barrier_all()
                        T.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])  # -
                        T.barrier_all()

                    T.exp(acc_s_ub, acc_s_ub)
                    T.barrier_all()

                    T.reduce_sum(sumexp_i_ub, acc_s_ub, tmp_ub, dim=-1)
                    T.barrier_all()

                    T.mul(sumexp, sumexp, m_i_prev)  # check
                    T.barrier_all()

                    T.add(sumexp, sumexp, sumexp_i_ub)
                    T.barrier_all()

                    for h_i in range(block_M // 2):
                        T.barrier_all()
                        T.mul(acc_o[h_i, :], acc_o[h_i, :], m_i_prev[h_i])
                        T.barrier_all()

                    T.copy(acc_s_ub, acc_s_half)
                    T.barrier_all()

                    T.copy(
                        acc_s_half,
                        workspace_2[cid, vid * block_M // 2:vid * block_M // 2 + block_M // 2, :])
                    T.barrier_all()

                    T.set_cross_flag("MTE3", 1)

                    T.wait_cross_flag(2)
                    T.barrier_all()

                    T.copy(
                        workspace_3[cid, vid * block_M // 2:vid * block_M // 2 + block_M // 2, :],
                        acc_o_ub)
                    T.barrier_all()

                    T.add(acc_o, acc_o, acc_o_ub)
                    T.barrier_all()

                    T.set_cross_flag("V", 3)
                    T.barrier_all()

                for h_i in range(block_M // 2):
                    T.barrier_all()
                    T.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])
                    T.barrier_all()

                T.copy(acc_o, acc_o_half)
                T.barrier_all()
                T.copy(
                    acc_o_half, Output[bz, by, bx * block_M + vid * block_M // 2:bx * block_M +
                                       vid * block_M // 2 + block_M // 2, :])

    return main


func = flash_attention_fwd(
    heads=1,
    dim=512,
)


def ref_flash_attn(q, k, v):
    q = q.float()
    k = k.float()
    v = v.float()

    acc = torch.einsum("bhsd,bhkd->bhsk", q, k) * (1.0 / q.shape[-1])**0.5
    acc = acc.softmax(dim=-1)
    o = torch.einsum("bhsk,bhkd->bhsd", acc, v)
    return o.to(torch.float16)


B, S, H, D = 1, 128, 1, 512

q = torch.randn((B, H, S, D), dtype=torch.float16)
k = torch.randn((B, H, S, D), dtype=torch.float16)
v = torch.randn((B, H, S, D), dtype=torch.float16)

block_num = S // 64 * H * B

workspace_1 = torch.zeros((block_num, 64, 64), dtype=torch.float)
workspace_2 = torch.zeros((block_num, 64, 64), dtype=torch.float16)
workspace_3 = torch.zeros((block_num, 64, 512), dtype=torch.float)

torch.npu.synchronize()
print("init successful!")

output = func(q, k, v, workspace_1, workspace_2, workspace_3)
ref_output = ref_flash_attn(q, k, v)
torch.npu.synchronize()

torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)

print("Test Passed!")
