import math
import tilelang
import tilelang.language as T
import torch
import torch_npu

from moe_token_utils import is_fp32_dtype, pad_last_dim

PASS_CONFIGS_EXPERT = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}


def _build_fused_permute_kernel(
    num_tokens,
    topK,
    hidden_size,
    E,
    padded_E,
    out_len,
    num_experts,
    actual_cores,
    chunk_size,
    tokens_per_core,
    TILE_H,
    n_htiles,
    dtype,
    idx_dtype,
):
    ws_total = actual_cores * num_experts
    HALF_H = TILE_H // 2
    total_iters = tokens_per_core * n_htiles

    @tilelang.jit(out_idx=[2, 3], workspace_idx=[4], pass_configs=PASS_CONFIGS_EXPERT)
    def _build(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_E,
        out_len,
        num_experts,
        actual_cores,
        chunk_size,
        tokens_per_core,
        ws_total,
        TILE_H,
        n_htiles,
        HALF_H,
        total_iters,
        dtype,
        idx_dtype,
    ):
        stages = 2

        @T.macro
        def init_flag():
            T.set_flag("mte3", "mte2", 0)
            T.set_flag("mte3", "mte2", 1)

        @T.macro
        def clear_flag():
            T.wait_flag("mte3", "mte2", 0)
            T.wait_flag("mte3", "mte2", 1)

        @T.prim_func
        def moe_token_permute(
            tokens_gm: T.Tensor([num_tokens, hidden_size], dtype),
            indices_gm: T.Tensor([1, padded_E], idx_dtype),
            perm_out_gm: T.Tensor([out_len, hidden_size], dtype),
            sio_out_gm: T.Tensor([1, padded_E], idx_dtype),
            workspace_gm: T.Tensor([1, ws_total], idx_dtype),
        ):
            with T.Kernel(actual_cores, is_npu=True) as (cid, vid):
                idx_ub = T.alloc_ub([1, chunk_size], idx_dtype)
                hist_ub = T.alloc_ub([1, num_experts], idx_dtype)
                ws_ub = T.alloc_ub([1, ws_total], idx_dtype)
                offsets_ub = T.alloc_ub([1, num_experts], idx_dtype)
                counters_ub = T.alloc_ub([1, num_experts], idx_dtype)
                sio_chunk_ub = T.alloc_ub([1, chunk_size], idx_dtype)
                acc_ub = T.alloc_ub([1], idx_dtype)
                cpre_ub = T.alloc_ub([1], idx_dtype)
                running_ub = T.alloc_ub([1], idx_dtype)
                wp_ub = T.alloc_ub([1], idx_dtype)
                row_buf = T.alloc_ub([stages, HALF_H], dtype)

                my_start = cid * chunk_size

                with T.Scope("C"):
                    T.sync_all()

                with T.Scope("V"):
                    for e in T.Pipelined(num_experts):
                        hist_ub[0, e] = 0

                    T.copy(indices_gm[0, my_start], idx_ub)
                    T.set_flag("mte2", "v", 2)
                    T.wait_flag("mte2", "v", 2)

                    for i in T.Pipelined(chunk_size):
                        if my_start + i < E:
                            expert = idx_ub[0, i]
                            hist_ub[0, expert] = hist_ub[0, expert] + 1

                    T.set_flag("v", "mte3", 2)
                    T.wait_flag("v", "mte3", 2)

                    T.copy(hist_ub, workspace_gm[0, cid * num_experts])

                    T.set_flag("mte3", "mte2", 2)
                    T.wait_flag("mte3", "mte2", 2)
                    T.sync_all()

                    T.copy(workspace_gm[0, 0], ws_ub)
                    T.set_flag("mte2", "v", 3)
                    T.wait_flag("mte2", "v", 3)

                    running_ub[0] = 0
                    for e in T.Pipelined(num_experts):
                        acc_ub[0] = 0
                        cpre_ub[0] = 0
                        for c in T.Pipelined(actual_cores):
                            acc_ub[0] = acc_ub[0] + ws_ub[0, c * num_experts + e]
                            if c < cid:
                                cpre_ub[0] = cpre_ub[0] + ws_ub[0, c * num_experts + e]
                        offsets_ub[0, e] = running_ub[0] + cpre_ub[0]
                        counters_ub[0, e] = 0
                        running_ub[0] = running_ub[0] + acc_ub[0]

                    for i in T.Pipelined(chunk_size):
                        if my_start + i < E:
                            expert = idx_ub[0, i]
                            wp_ub[0] = offsets_ub[0, expert] + counters_ub[0, expert]
                            counters_ub[0, expert] = counters_ub[0, expert] + 1
                            sio_chunk_ub[0, i] = wp_ub[0]

                    T.set_flag("v", "mte3", 3)
                    T.wait_flag("v", "mte3", 3)

                    init_flag()

                    if total_iters > 0:
                        pro_src = cid * tokens_per_core
                        pro_h_off = 0 + vid * HALF_H
                        T.wait_flag("mte3", "mte2", 0)
                        if pro_src < num_tokens:
                            T.copy(tokens_gm[pro_src, pro_h_off], row_buf[0, :])
                        T.set_flag("mte2", "v", 0)
                        T.set_flag("mte2", "mte3", 10)

                    for i in T.serial(total_iters):
                        cur = i % stages
                        nxt = (i + 1) % stages

                        cur_t = i // n_htiles
                        cur_ht = i % n_htiles
                        cur_src = cid * tokens_per_core + cur_t
                        cur_h_off = cur_ht * TILE_H + vid * HALF_H
                        cur_base = cur_t * topK

                        next_i = i + 1
                        next_t = next_i // n_htiles
                        next_ht = next_i % n_htiles
                        next_src = cid * tokens_per_core + next_t
                        next_h_off = next_ht * TILE_H + vid * HALF_H

                        has_next = next_i < total_iters

                        if has_next:
                            T.wait_flag("mte3", "mte2", nxt)
                            if next_src < num_tokens:
                                T.copy(tokens_gm[next_src, next_h_off], row_buf[nxt, :])
                            T.set_flag("mte2", "v", nxt)
                            T.set_flag("mte2", "mte3", nxt + 10)

                        T.wait_flag("mte2", "v", cur)
                        T.wait_flag("mte2", "mte3", cur + 10)

                        if cur_src < num_tokens:
                            for k in T.serial(topK):
                                wp_ub[0] = sio_chunk_ub[0, cur_base + k]
                                if wp_ub[0] < out_len:
                                    T.copy(
                                        row_buf[cur, :],
                                        perm_out_gm[wp_ub[0], cur_h_off],
                                    )

                        T.set_flag("v", "mte3", cur)
                        T.wait_flag("v", "mte3", cur)
                        T.set_flag("mte3", "mte2", cur)

                    clear_flag()

                    T.set_flag("v", "mte3", 2)
                    T.wait_flag("v", "mte3", 2)

                    T.copy(sio_chunk_ub, sio_out_gm[0, my_start])

                    T.set_flag("mte3", "mte2", 2)
                    T.wait_flag("mte3", "mte2", 2)

        return moe_token_permute

    return _build(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_E,
        out_len,
        num_experts,
        actual_cores,
        chunk_size,
        tokens_per_core,
        ws_total,
        TILE_H,
        n_htiles,
        HALF_H,
        total_iters,
        dtype,
        idx_dtype,
    )


def _compile_fused(
    num_tokens,
    topK,
    hidden_size,
    E,
    out_len,
    num_experts,
    NUM_CORES=24,
    TILE_H=None,
    dtype="float16",
    idx_dtype="int32",
):
    actual_cores = min(NUM_CORES, max(1, num_tokens))
    tokens_per_core = math.ceil(num_tokens / actual_cores)
    chunk_size = tokens_per_core * topK
    padded_E = actual_cores * chunk_size
    TILE_H = hidden_size if TILE_H is None else TILE_H
    n_htiles = hidden_size // TILE_H
    fused_func = _build_fused_permute_kernel(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_E,
        out_len,
        num_experts,
        actual_cores,
        chunk_size,
        tokens_per_core,
        TILE_H,
        n_htiles,
        dtype,
        idx_dtype,
    )
    return fused_func, padded_E


class MoeTokenPermute:
    def __init__(
        self,
        num_tokens,
        topK,
        hidden_size,
        num_experts=64,
        num_out_tokens=0,
        NUM_CORES=24,
        TILE_H=None,
        dtype="float16",
    ):
        self.num_tokens = num_tokens
        self.topK = topK
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.E = num_tokens * topK
        self._out_len = num_out_tokens if num_out_tokens > 0 else self.E

        min_compile_h = 64 if is_fp32_dtype(dtype) else 32
        self._compile_hidden_size = max(hidden_size, min_compile_h)
        compile_tile_h = TILE_H if TILE_H is None else max(TILE_H, min_compile_h)
        self._fused_func, self._padded_E = _compile_fused(
            num_tokens,
            topK,
            self._compile_hidden_size,
            self.E,
            self._out_len,
            num_experts,
            NUM_CORES=NUM_CORES,
            TILE_H=compile_tile_h,
            dtype=dtype,
        )

    def __call__(self, tokens, indices):
        device = tokens.device
        E = self.E
        tokens_in = pad_last_dim(tokens, self._compile_hidden_size)
        indices_padded = torch.zeros(self._padded_E, dtype=torch.int32, device=device)
        indices_padded[:E] = indices
        perm_out, sio_padded = self._fused_func(tokens_in, indices_padded.unsqueeze(0))
        sio = sio_padded.squeeze(0)[:E]
        if self._compile_hidden_size != self.hidden_size:
            perm_out = perm_out[:, : self.hidden_size].contiguous()
        return perm_out, sio


def test_permute_parameterized(pt_dtype, tl_dtype_str):
    print(f"\n{'=' * 60}")
    print(f"Testing MoeTokenPermute, dtype: {tl_dtype_str.upper()}")
    print(f"{'=' * 60}")

    torch.manual_seed(42)

    num_tokens = 16
    hidden_size = 8
    topk = 4
    num_experts = 4

    all_passed = True

    print(">>> Test case 1: Standard Forward test")

    tokens = torch.randn(num_tokens, hidden_size, dtype=pt_dtype, device="npu")
    indices = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device="npu")

    npu_permuted, npu_sorted_idx = torch_npu.npu_moe_token_permute(tokens, indices)

    tl_op = MoeTokenPermute(
        num_tokens=num_tokens,
        topK=topk,
        hidden_size=hidden_size,
        num_experts=num_experts,
        dtype=tl_dtype_str,
    )
    tl_permuted, tl_sorted_idx = tl_op(tokens, indices.view(-1))

    try:
        torch.testing.assert_close(tl_permuted, npu_permuted)
        torch.testing.assert_close(tl_sorted_idx, npu_sorted_idx)
        print(f"    [PASS] {tl_dtype_str.upper()} Standard Forward precision test passed!")
    except AssertionError as e:
        print(
            f"    [FAILED] {tl_dtype_str.upper()} Standard Forward precision test failed!\n",
            e,
        )
        all_passed = False

    print("\n>>> Test case 2: Clip test with truncation")
    num_out_tokens = 10

    tokens_clip = torch.randn(num_tokens, hidden_size, dtype=pt_dtype, device="npu")
    indices_clip = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device="npu")

    npu_permuted_clip, npu_sorted_idx_clip = torch_npu.npu_moe_token_permute(tokens_clip, indices_clip, num_out_tokens=num_out_tokens)

    tl_op_clip = MoeTokenPermute(
        num_tokens=num_tokens,
        topK=topk,
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_out_tokens=num_out_tokens,
        dtype=tl_dtype_str,
    )
    tl_permuted_clip, tl_sorted_idx_clip = tl_op_clip(tokens_clip, indices_clip.view(-1))

    try:
        torch.testing.assert_close(tl_permuted_clip, npu_permuted_clip)
        torch.testing.assert_close(tl_sorted_idx_clip, npu_sorted_idx_clip)
        print(f"    [PASS] {tl_dtype_str.upper()} Clip truncation precision test passed!")
    except AssertionError as e:
        print(
            f"    [FAILED] {tl_dtype_str.upper()} Clip truncation precision test failed!\n",
            e,
        )
        all_passed = False

    return all_passed


def test_permute():
    dtypes_to_test = [
        (torch.float16, "float16"),
        (torch.bfloat16, "bfloat16"),
        (torch.float32, "float32"),
    ]

    overall_passed = True
    for pt_type, tl_type_str in dtypes_to_test:
        passed = test_permute_parameterized(pt_dtype=pt_type, tl_dtype_str=tl_type_str)
        if not passed:
            overall_passed = False

    print(f"\n{'=' * 60}")
    if overall_passed:
        print("Test passed!")
    else:
        print("Test failed! The precision is not correct!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    test_permute()
