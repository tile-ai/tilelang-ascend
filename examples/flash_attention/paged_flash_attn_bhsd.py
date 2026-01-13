import argparse
import tilelang as tl
from tilelang import DataType, language as T
import torch

@tl.jit(
    out_idx=[4],
    workspace_idx=[5, 6, 7],
    pass_configs = {
        tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True
    }
)
def paged_flash_attention_fwd(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    cache_blocks: int,  # block count in K/V cache
    table_blocks: int,  # block count in block_table
    block_M: int = 64,
    block_N: int = 64,  # block_size for one block
):
    DTYPE = "float16"
    ACCUM_DTYPE = "float"
    TEMP_DTYPE = "uint8"
    INDICES_DTYPE = "int32"
    VEC_NUM = 2
    CAST_MODE = "CAST_NONE"

    def bytes_of(dtype: str) -> int:
        return DataType(dtype).bits // 8

    DTYPE_BYTES = bytes_of(DTYPE)
    ACCUM_DTYPE_BYTES = bytes_of(ACCUM_DTYPE)

    sm_scale = (1.0 / dim)**0.5

    q_shape = [batch, heads, seq_len, dim]
    kv_cache_shape = [cache_blocks, block_N, heads, dim]

    block_M_2 = T.ceildiv(block_M, VEC_NUM)
    block_num = T.ceildiv(seq_len, block_M) * heads * batch
    tmp_ub_size = 3 * ACCUM_DTYPE_BYTES * block_M_2 * block_N

    # address annotation constants
    addr_k_l1 = block_M * dim * DTYPE_BYTES
    addr_v_l1 = addr_k_l1 + block_N * dim * DTYPE_BYTES
    addr_sumexp = block_M_2 * dim * ACCUM_DTYPE_BYTES
    addr_m_i = addr_sumexp + block_M_2 * ACCUM_DTYPE_BYTES
    addr_acc_s_ub = addr_m_i + block_M_2 * ACCUM_DTYPE_BYTES
    addr_m_i_prev = addr_acc_s_ub + block_M_2 * block_N * ACCUM_DTYPE_BYTES
    addr_acc_s_ub_ = addr_m_i_prev + block_M_2 * ACCUM_DTYPE_BYTES
    addr_sumexp_i_ub = addr_acc_s_ub_ + tmp_ub_size * bytes_of(TEMP_DTYPE)

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, DTYPE),             # type: ignore
        KCache: T.Tensor(kv_cache_shape, DTYPE), # type: ignore
        VCache: T.Tensor(kv_cache_shape, DTYPE), # type: ignore
        block_table: T.Tensor([batch, table_blocks], INDICES_DTYPE),        # type: ignore
        Output: T.Tensor(q_shape, DTYPE),        # type: ignore
        workspace_1: T.Tensor([block_num, block_M, block_N], ACCUM_DTYPE),  # type: ignore
        workspace_2: T.Tensor([block_num, block_M, block_N], DTYPE),        # type: ignore
        workspace_3: T.Tensor([block_num, block_M, dim], ACCUM_DTYPE),      # type: ignore
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid % T.ceildiv(seq_len, block_M)
            by = cid // T.ceildiv(seq_len, block_M) % heads
            bz = cid // T.ceildiv(seq_len, block_M) // heads % batch

            q_l1 = T.alloc_L1([block_M, dim], DTYPE)
            k_l1 = T.alloc_L1([block_N, dim], DTYPE)
            v_l1 = T.alloc_L1([block_N, dim], DTYPE)

            acc_s_l1 = T.alloc_L1([block_M, block_N], DTYPE)

            acc_s_l0c = T.alloc_L0C([block_M, block_N], ACCUM_DTYPE)
            acc_o_l0c = T.alloc_L0C([block_M, dim], ACCUM_DTYPE)

            acc_o = T.alloc_ub([block_M_2, dim], ACCUM_DTYPE)
            sumexp = T.alloc_ub([block_M_2], ACCUM_DTYPE)
            m_i = T.alloc_ub([block_M_2], ACCUM_DTYPE)

            acc_s_ub = T.alloc_ub([block_M_2, block_N], ACCUM_DTYPE)
            m_i_prev = T.alloc_ub([block_M_2], ACCUM_DTYPE)
            acc_s_ub_ = T.alloc_ub([block_M_2, block_N], ACCUM_DTYPE)
            tmp_ub = T.alloc_ub([tmp_ub_size], TEMP_DTYPE)
            sumexp_i_ub = T.alloc_ub([block_M_2], ACCUM_DTYPE)
            acc_s_half = T.alloc_ub([block_M_2, block_N], DTYPE)
            acc_o_ub = T.alloc_ub([block_M_2, dim], ACCUM_DTYPE)
            acc_o_half = T.alloc_ub([block_M_2, dim], DTYPE)

            T.annotate_address({
                # L1 address
                q_l1: 0,
                k_l1: addr_k_l1,
                acc_s_l1: addr_k_l1,
                v_l1: addr_v_l1,

                # L0C address
                acc_s_l0c: 0,
                acc_o_l0c: 0,

                ## ub address
                acc_o: 0,
                sumexp: addr_sumexp,
                m_i: addr_m_i,
                acc_s_ub: addr_acc_s_ub,
                m_i_prev: addr_m_i_prev,
                acc_s_ub_: addr_acc_s_ub_,
                tmp_ub: addr_acc_s_ub_,
                sumexp_i_ub: addr_sumexp_i_ub,
                acc_s_half: addr_sumexp_i_ub,
                acc_o_ub: addr_sumexp_i_ub,
                acc_o_half: addr_sumexp_i_ub
            })

            # with T.Scope("C"):
            T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], q_l1)
            for k in T.serial(T.ceildiv(seq_len, block_N)):
                bt = block_table[bz, k]
                T.copy(KCache[bt, :,  by, :], k_l1)
                T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                T.copy(acc_s_l0c, workspace_1[cid, :, :])

                T.copy(workspace_2[cid, :, :], acc_s_l1)

                T.copy(VCache[bt, :,  by, :], v_l1)
                T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                T.copy(acc_o_l0c, workspace_3[cid, :, :])

            # with T.Scope("V"):
            T.tile.fill(acc_o, 0.0)
            T.tile.fill(sumexp, 0.0)
            T.tile.fill(m_i, -T.infinity(ACCUM_DTYPE))
            # softmax
            for _k in T.serial(T.ceildiv(seq_len, block_N)):
                T.tile.fill(acc_s_ub, 0.0)
                T.copy(m_i, m_i_prev)
                T.copy(workspace_1[cid, vid * block_M_2:(vid + 1) * block_M_2, :], acc_s_ub_)
                T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                T.tile.reduce_max(m_i, acc_s_ub, tmp_ub, dim=-1)
                T.tile.max(m_i, m_i, m_i_prev)
                T.tile.sub(m_i_prev, m_i_prev, m_i)
                T.tile.exp(m_i_prev, m_i_prev)

                for h_i in T.serial(block_M_2):
                    T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])

                T.tile.exp(acc_s_ub, acc_s_ub)
                T.tile.reduce_sum(sumexp_i_ub, acc_s_ub, tmp_ub, dim=-1)
                T.tile.mul(sumexp, sumexp, m_i_prev)  # check
                T.tile.add(sumexp, sumexp, sumexp_i_ub)

                for h_i in T.serial(block_M_2):
                    T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], m_i_prev[h_i])

                T.tile.cast_tl(acc_s_half, acc_s_ub, CAST_MODE, block_M_2 * block_N)
                T.copy(acc_s_half, workspace_2[cid, vid * block_M_2:(vid + 1) * block_M_2, :])
                T.copy(workspace_3[cid, vid * block_M_2:(vid + 1) * block_M_2, :], acc_o_ub)
                T.tile.add(acc_o, acc_o, acc_o_ub)

            for h_i in T.serial(block_M_2):
                T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])

            T.tile.cast_tl(acc_o_half, acc_o, CAST_MODE, block_M_2 * dim)
            T.copy(acc_o_half, Output[bz, by, bx * block_M + vid * block_M_2:bx * block_M + (vid + 1) * block_M_2, :])

    return main

def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b

def gen_cache(data: torch.Tensor, block_table: torch.LongTensor, block_size: int):
    batch, heads, seq_len, dim = data.shape

    seq_blocks = ceildiv(seq_len, block_size)
    cache_blocks = batch * seq_blocks

    data_cache = torch.zeros((cache_blocks, block_size, heads, dim), dtype=data.dtype, device=data.device)

    for b in range(batch):
        for si in range(seq_blocks):
            start_idx = si * block_size
            end_idx = min(start_idx + block_size, seq_len)
            length = end_idx - start_idx
            block_idx = b * seq_blocks + si
            data_cache[block_idx, :length, :, :] = data[b, :, start_idx:end_idx, :].permute(1, 0, 2)
            block_table[b, si] = block_idx
    return data_cache


def ref_program(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, block_table: torch.LongTensor):
    q = q.to(torch.float32)              # [batch, heads, seq_len, dim]
    k_cache = k_cache.to(torch.float32)  # [n_blocks, block_size, heads, dim]
    v_cache = v_cache.to(torch.float32)  # [n_blocks, block_size, heads, dim]

    batch, heads, seq_len, dim = q.shape
    sm_scale = (1.0 / dim)**0.5

    o = torch.zeros_like(q, dtype=torch.float32)

    for b in range(batch):
        blocks = block_table[b]    # [table_blocks]
        k_seq = k_cache[blocks].reshape(-1, heads, dim)[:seq_len]    # [seq_len, heads, dim]
        v_seq = v_cache[blocks].reshape(-1, heads, dim)[:seq_len]    # [seq_len, heads, dim]
        for s in range(seq_len):
            q_vec = q[b, :, s, :]  # [heads, dim]
            scores = torch.einsum("hd,shd->hs", q_vec, k_seq) * sm_scale  # [heads, seq_len]
            attn = torch.softmax(scores, dim=-1)
            o[b, :, s, :] = torch.einsum("hs,shd->hd", attn, v_seq)  # [heads, dim]
    return o.to(torch.float16)

def check_case(batch: int, heads: int, seq_len: int, dim: int, block_size: int = 64):
    q = torch.randn((batch, heads, seq_len, dim), dtype=torch.float16)
    k = torch.randn((batch, heads, seq_len, dim), dtype=torch.float16)
    v = torch.randn((batch, heads, seq_len, dim), dtype=torch.float16)

    table_blocks = ceildiv(seq_len, block_size)
    block_table = torch.zeros((batch, table_blocks), dtype=torch.int32)
    k_cache = gen_cache(k, block_table, block_size)
    v_cache = gen_cache(v, block_table, block_size)
    cache_blocks = k_cache.shape[0]

    kernel = paged_flash_attention_fwd(batch, heads, seq_len, dim, cache_blocks, table_blocks, block_N=block_size)
    output = kernel(q, k_cache, v_cache, block_table)

    ref_output = ref_program(q, k_cache, v_cache, block_table)

    torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)

def main(custom_args=None):
    parser = argparse.ArgumentParser(description="Paged Flash Attention Example", add_help=False)
    parser.add_argument("-b", "--batch", type=int, default=1, help="Batch Size")
    parser.add_argument("-h", "--heads", type=int, default=1, help="Number of Heads")
    parser.add_argument("-s", "--seq_len", type=int, default=128, help="Sequence Length")
    parser.add_argument("-d", "--hidden_dim", type=int, default=512, help="Hidden Dimension")
    args, remains = parser.parse_known_args(custom_args)
    if remains:
        print(f"[{parser.description}]", "Unknown args:", remains)
    batch, heads, seq_len, dim = args.batch, args.heads, args.seq_len, args.hidden_dim

    tl.cache.clear_cache()
    torch.set_default_device('npu')
    torch.manual_seed(0)

    check_case(batch, heads, seq_len, dim)

    print("Paged Flash Attention example passed!")
    print("Kernel Output Match!")

if __name__ == "__main__":
    main()
