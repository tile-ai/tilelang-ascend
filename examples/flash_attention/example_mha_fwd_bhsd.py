import tilelang
import tilelang.language as T
import torch
import argparse

torch.set_default_device("npu")
torch.manual_seed(0)

tilelang.disable_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)
def flash_attention_fwd(batch, heads, seq_len, dim):
    block_M, block_N = 64, 64

    dtype = "float16"
    accum_dtype = "float"

    sm_scale = (1.0 / dim) ** 0.5

    shape = [batch, heads, seq_len, dim]

    m_num = T.ceildiv(seq_len, block_M)
    block_num = m_num * heads * batch

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
        workspace_1: T.Tensor([block_num, block_M, block_N], accum_dtype),
        workspace_2: T.Tensor([block_num, block_M, block_N], dtype),
        workspace_3: T.Tensor([block_num, block_M, dim], accum_dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid % m_num
            by = cid // m_num % heads
            bz = cid // m_num // heads % batch

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
            sumexp_i_ub = T.alloc_ub([block_M // 2], accum_dtype)
            acc_s_half = T.alloc_ub([block_M // 2, block_N], dtype)
            acc_o_ub = T.alloc_ub([block_M // 2, dim], accum_dtype)
            acc_o_half = T.alloc_ub([block_M // 2, dim], dtype)

            T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], q_l1)
            for k in T.serial(T.ceildiv(seq_len, block_N)):
                T.copy(K[bz, by, k * block_N : (k + 1) * block_N, :], k_l1)
                T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                T.copy(acc_s_l0c, workspace_1[cid, :, :])

                T.copy(workspace_2[cid, :, :], acc_s_l1)

                T.copy(V[bz, by, k * block_N : (k + 1) * block_N, :], v_l1)
                T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                T.copy(acc_o_l0c, workspace_3[cid, :, :])

            T.tile.fill(acc_o, 0.0)
            T.tile.fill(sumexp, 0.0)
            T.tile.fill(m_i, -(2**30))
            for _k in T.serial(T.ceildiv(seq_len, block_N)):
                T.tile.fill(acc_s_ub, 0.0)
                T.copy(m_i, m_i_prev)
                T.copy(workspace_1[cid, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :], acc_s_ub_)
                T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                T.reduce_max(acc_s_ub, m_i, dim=-1)
                T.tile.max(m_i, m_i, m_i_prev)
                T.tile.sub(m_i_prev, m_i_prev, m_i)
                T.tile.exp(m_i_prev, m_i_prev)

                for h_i in range(block_M // 2):
                    T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])

                T.tile.exp(acc_s_ub, acc_s_ub)
                T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                T.tile.mul(sumexp, sumexp, m_i_prev)
                T.tile.add(sumexp, sumexp, sumexp_i_ub)

                for h_i in range(block_M // 2):
                    T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], m_i_prev[h_i])

                T.copy(acc_s_ub, acc_s_half)
                T.copy(acc_s_half, workspace_2[cid, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :])
                T.copy(workspace_3[cid, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :], acc_o_ub)
                T.tile.add(acc_o, acc_o, acc_o_ub)

            for h_i in range(block_M // 2):
                T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])

            T.copy(acc_o, acc_o_half)
            T.copy(acc_o_half, Output[bz, by, bx * block_M + vid * block_M // 2 : bx * block_M + vid * block_M // 2 + block_M // 2, :])

    return main


def ref_flash_attn(q, k, v):
    q = q.float()
    k = k.float()
    v = v.float()

    acc = torch.einsum("bhsd,bhkd->bhsk", q, k) * (1.0 / q.shape[-1]) ** 0.5
    acc = acc.softmax(dim=-1)
    o = torch.einsum("bhsk,bhkd->bhsd", acc, v)
    return o.to(torch.float16)


def run_single_test(batch, heads, seq_len, dim, name=""):
    torch.manual_seed(0)

    func = flash_attention_fwd(batch=batch, heads=heads, seq_len=seq_len, dim=dim)

    q = torch.randn((batch, heads, seq_len, dim), dtype=torch.float16)
    k = torch.randn((batch, heads, seq_len, dim), dtype=torch.float16)
    v = torch.randn((batch, heads, seq_len, dim), dtype=torch.float16)

    torch.npu.synchronize()
    output = func(q, k, v)
    ref_output = ref_flash_attn(q, k, v)
    torch.npu.synchronize()

    try:
        torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)
        print(f"[{name}] B={batch}, H={heads}, S={seq_len}, D={dim} - Passed")
        return True
    except AssertionError as e:
        print(f"[{name}] B={batch}, H={heads}, S={seq_len}, D={dim} - Failed")
        print(f"  Error: {str(e)[:100]}")
        return False


def run_all_tests():
    test_configs = [
        {"batch": 1, "heads": 1, "seq_len": 128, "dim": 512, "name": "Config 1"},
        {"batch": 1, "heads": 1, "seq_len": 256, "dim": 64, "name": "Config 2"},
        {"batch": 2, "heads": 1, "seq_len": 128, "dim": 512, "name": "Config 3"},
        {"batch": 1, "heads": 8, "seq_len": 128, "dim": 512, "name": "Config 4"},
        {"batch": 1, "heads": 1, "seq_len": 128, "dim": 256, "name": "Config 5"},
    ]

    print("=" * 60)
    print("Flash Attention Forward Test")
    print("=" * 60)

    passed = 0
    total = len(test_configs)

    for i, cfg in enumerate(test_configs, 1):
        print(f"\n[Test {i}/{total}] {cfg['name']}")
        if run_single_test(cfg["batch"], cfg["heads"], cfg["seq_len"], cfg["dim"], cfg["name"]):
            passed += 1

    print("\n" + "=" * 60)
    print(f"Test Summary: {passed}/{total} passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flash Attention Forward Test")
    parser.add_argument("--batch", type=int, default=None, help="batch size")
    parser.add_argument("--heads", type=int, default=None, help="heads")
    parser.add_argument("--seq_len", type=int, default=None, help="sequence length")
    parser.add_argument("--dim", type=int, default=None, help="dim")
    parser.add_argument("--all", action="store_true", help="run all test configs")
    args = parser.parse_args()

    if args.all:
        success = run_all_tests()
        if success:
            print("Test Passed!")
    elif args.batch and args.heads and args.seq_len and args.dim:
        success = run_single_test(args.batch, args.heads, args.seq_len, args.dim, "Custom")
        if success:
            print("Test Passed!")
    else:
        success = run_all_tests()
        if success:
            print("Test Passed!")
