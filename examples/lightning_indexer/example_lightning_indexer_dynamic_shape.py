import torch
from collections import Counter

torch.manual_seed(2)
import tilelang
import tilelang.language as T

tilelang.disable_cache()


@tilelang.jit(out_idx=[-1])  # for jit
def indexer(N2,
            G,
            D,
            TOP_K,
            VECTOR_BASEN,
            VECTOR_BASEG,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            input_dtype="float16",
            calc_dtype="float"):

    B = T.symbolic("B")
    S1 = T.symbolic("S1")
    S2 = T.symbolic("S2")

    @T.prim_func
    def main(Query: T.Tensor((B, S1, N2, G * D), input_dtype), KEY: T.Tensor(
        (B, S2, N2, D), input_dtype), QK_RES: T.Tensor((B, N2, S1, G, S2), calc_dtype),
             WEIGHTS: T.Tensor((B, S1, N2, G), calc_dtype), OUT: T.Tensor((B, N2, S1, TOP_K),
                                                                          "int")):

        total_process_num = N2 * S1
        each_core_process_num = total_process_num // 2
        with T.Kernel(B * N2, is_npu=True) as (cid, vid):
            n2_id = cid % N2

            with T.Scope("C"):
                Q_L1 = T.alloc_L1((BLOCK_M, BLOCK_K), input_dtype)
                K_L1 = T.alloc_L1((BLOCK_N, BLOCK_K), input_dtype)

                C_L0 = T.alloc_L0C((BLOCK_M, BLOCK_N), calc_dtype)

                T.annotate_address({
                    # L1 address
                    Q_L1: 0,
                    K_L1: 32768,

                    # L0C address
                    C_L0: 0,
                })
                T.barrier_all()
                for n2 in T.serial(N2):
                    for g in T.serial(G):
                        for m in T.serial(S1 // BLOCK_M):
                            for n in T.serial(S2 // BLOCK_N):
                                T.barrier_all()
                                T.copy(Query[cid, m * BLOCK_M: (m + 1) * BLOCK_M, n2, g * D: (g + 1) * D], Q_L1)
                                T.barrier_all()
                                T.copy(KEY[cid, n * BLOCK_N: (n + 1) * BLOCK_N, n2, 0: D], K_L1)
                                T.barrier_all()
                                T.gemm_v0(Q_L1, K_L1, C_L0, transpose_B=True, init=True)
                                T.barrier_all()
                                T.copy(
                                    C_L0,
                                    QK_RES[cid, n2, m * BLOCK_M: (m + 1) * BLOCK_M, g, n * BLOCK_N: (n + 1) * BLOCK_N], # [B, N2, S1, G, S2]
                                    enable_relu=True)
                                T.barrier_all()
                T.set_cross_flag("FIX", 0)

            with T.Scope("V"):
                mm_res_ub = T.alloc_ub((VECTOR_BASEG, VECTOR_BASEN), calc_dtype)
                mm_res_ub_flat = T.alloc_ub((VECTOR_BASEG * VECTOR_BASEN), calc_dtype)
                mm_res_ub_uint8 = T.alloc_ub((VECTOR_BASEG, VECTOR_BASEN), "uint8")
                weight_ub = T.alloc_ub(VECTOR_BASEG, calc_dtype)
                weight_brcb_ub = T.alloc_ub((VECTOR_BASEG, 8), calc_dtype)
                reduce_tmp_ub = T.alloc_ub((VECTOR_BASEG, VECTOR_BASEN), calc_dtype)
                reduce_g_ub = T.alloc_ub(VECTOR_BASEN, calc_dtype)
                sort_indice_tmp_ub = T.alloc_ub(VECTOR_BASEN, "int")
                sort_indice_tmp_ub_uint = T.alloc_ub(VECTOR_BASEN, "uint")
                topk_indices_tmp_ub = T.alloc_ub(VECTOR_BASEN, "int")
                topk_indices_tmp_ub_uint = T.alloc_ub(VECTOR_BASEN, "uint")
                topk_global_ub1 = T.alloc_ub([TOP_K // VECTOR_BASEN, VECTOR_BASEN * 2], calc_dtype)
                topk_global_ub1_flat = T.alloc_ub(TOP_K, "int")
                topk_global_ub1_uint = T.alloc_ub([TOP_K // VECTOR_BASEN, VECTOR_BASEN * 2], "uint")
                topk_global_ub2 = T.alloc_ub(TOP_K * 2, calc_dtype)

                T.annotate_address({
                    # ub address
                    mm_res_ub: 0,
                    mm_res_ub_flat: 0,
                    mm_res_ub_uint8: 0,
                    weight_ub: 65536,
                    weight_brcb_ub: 65664,
                    reduce_tmp_ub: 66688,
                    reduce_g_ub: 132224,
                    sort_indice_tmp_ub: 134272,
                    sort_indice_tmp_ub_uint: 134272,
                    topk_indices_tmp_ub: 136320,
                    topk_indices_tmp_ub_uint: 136320,
                    topk_global_ub1: 138368,
                    topk_global_ub1_uint: 138368,
                    topk_global_ub1_flat: 138368,
                    topk_global_ub2: 154752
                })

                s1_start_idx = vid * each_core_process_num
                s1_end_idx = s1_start_idx + each_core_process_num

                T.wait_cross_flag(0)
                T.arith_progression(topk_indices_tmp_ub, 0, 1, VECTOR_BASEN)
                for s1_id in T.serial(s1_start_idx, s1_end_idx):
                    T.barrier_all()
                    T.init_sort_buf(topk_global_ub2, TOP_K * 2, 0)
                    for s2_id in T.serial(S2 // VECTOR_BASEN):
                        T.barrier_all()
                        T.fill(reduce_tmp_ub, 0)
                        T.fill(reduce_g_ub, 0)
                        T.barrier_all()

                        for g_id in T.serial(G // VECTOR_BASEG):
                            T.barrier_all()
                            T.copy(QK_RES[cid, n2_id, s1_id, g_id * VECTOR_BASEG: (g_id + 1) * VECTOR_BASEG, s2_id * VECTOR_BASEN: (s2_id + 1) * VECTOR_BASEN], mm_res_ub)
                            T.barrier_all()
                            T.copy(WEIGHTS[cid, s1_id, n2_id, g_id * VECTOR_BASEG: (g_id + 1) * VECTOR_BASEG], weight_ub)
                            T.barrier_all()
                            for i in range(VECTOR_BASEG):
                                T.barrier_all()
                                T.mul(mm_res_ub[i, :], mm_res_ub[i, :], weight_ub[i])
                                T.barrier_all()
                            T.barrier_all()
                            T.add(reduce_tmp_ub, mm_res_ub, reduce_tmp_ub)
                            T.barrier_all()
                        # topK
                        merge_sort_times = TOP_K // VECTOR_BASEN
                        T.barrier_all()
                        T.reduce_sum(reduce_g_ub, reduce_tmp_ub, mm_res_ub_uint8, 0)
                        T.barrier_all()
                        T.add(sort_indice_tmp_ub, topk_indices_tmp_ub,
                              T.int32(s2_id * VECTOR_BASEN))
                        T.barrier_all()
                        T.sort(topk_global_ub1[(s2_id % merge_sort_times), :], reduce_g_ub,
                               sort_indice_tmp_ub_uint, mm_res_ub, VECTOR_BASEN // 32)
                        T.barrier_all()
                        if s2_id % merge_sort_times == merge_sort_times - 1:
                            if s2_id == merge_sort_times - 1:
                                T.merge_sort(topk_global_ub2, topk_global_ub1, VECTOR_BASEN,
                                             merge_sort_times, 0)
                            else:
                                T.merge_sort(mm_res_ub, topk_global_ub1, VECTOR_BASEN,
                                             merge_sort_times, 1)
                                T.barrier_all()
                                T.topk(topk_global_ub2, topk_global_ub1, mm_res_ub,
                                       VECTOR_BASEN * merge_sort_times)
                        T.barrier_all()
                    T.barrier_all()
                    T.gather_mask(topk_global_ub1, topk_global_ub2, TOP_K)
                    T.barrier_all()
                    T.copy(topk_global_ub1_flat, OUT[cid, n2_id, s1_id, 0:TOP_K])
                    T.barrier_all()

    return main


N2 = 1
G = 64
D = 128
TOP_K = 2048


def index_golden(q, k, weights):
    score_1 = torch.einsum("bsmgd, btmd->bmsgt", q, k)
    score_1 = score_1.relu()
    score = score_1.permute(0, 2, 1, 3, 4)
    mul_res = score * weights
    reduce_res = torch.sum(mul_res, dim=3)
    golden_out = torch.topk(reduce_res, TOP_K, dim=3, largest=True, sorted=True)
    return score_1.float(), golden_out.indices.to(torch.int32).permute(0, 2, 1, 3)


def count_mismatches_last_dim(tensor1, tensor2):
    assert tensor1.shape[-1] == tensor2.shape[
        -1], "the last dimension of two tensors must be the same"
    last_dim = tensor1.shape[-1]
    tensor1_flat = tensor1.view(-1, last_dim)
    tensor2_flat = tensor2.view(-1, last_dim)

    total_mismatches = 0

    for i in range(tensor1_flat.shape[0]):
        row1 = tensor1_flat[i].tolist()
        row2 = tensor2_flat[i].tolist()

        counter1 = Counter(row1)
        counter2 = Counter(row2)

        diff = (counter1 - counter2) + (counter2 - counter1)
        total_mismatches += sum(diff.values())

    return total_mismatches


def compare_tensors(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        print("error: two tensors have different shapes")
        print(f"tensor1 shape: {tensor1.shape}")
        print(f"tensor2 shape: {tensor2.shape}")
        return

    diff_mask = tensor1 != tensor2

    if not torch.any(diff_mask):
        print("two tensors are completely the same")
        return

    diff_indices = torch.nonzero(diff_mask)

    print(f"found {len(diff_indices)} different elements:")
    print("index\t\ttensor1 value\t\ttensor2 value")
    print("-" * 40)

    for idx in diff_indices:
        idx_str = str(tuple(idx.tolist()))

        val1 = tensor1[tuple(idx)]
        val2 = tensor2[tuple(idx)]

        print(f"{idx_str}\t{val1.item()}\t\t{val2.item()}")


def test_indexer():
    B = 2
    S1 = 1024
    S2 = 8192
    func = indexer(N2, G, D, TOP_K, 512, 32, 128, 128, 128)
    print(f"{func.get_kernel_source()}")

    q = torch.randn(B, S1, N2, G, D).half()
    k = torch.randn(B, S2, N2, D).half()
    weights = torch.randn(B, S1, N2, G, 1).float()

    qk_res_workspace = torch.zeros(B, N2, S1, G, S2).float()
    qk_res_workspace_, golden_out = index_golden(q, k, weights)

    q_npu = q.view(B, S1, N2, -1).npu()
    k_npu = k.npu()
    weights_npu = weights.npu()
    qk_res_workspace_npu = qk_res_workspace.npu()
    torch.npu.synchronize()
    npu_out = func(q_npu, k_npu, qk_res_workspace_npu, weights_npu).to(torch.int32)
    torch.npu.synchronize()
    print(npu_out.cpu().shape)
    print(f"npu out: {npu_out.cpu()}")
    print(golden_out.cpu().shape)
    print(f"golden out: {golden_out.cpu()}")

    total_mismatches = count_mismatches_last_dim(golden_out.cpu(), npu_out.cpu())
    print(
        f"mismatch number: {total_mismatches}, accuracy: {1 - total_mismatches / (B * S1 * N2 * TOP_K)}"
    )


if __name__ == "__main__":
    test_indexer()
