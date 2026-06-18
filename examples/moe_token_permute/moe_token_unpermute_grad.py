import math
import torch
import torch_npu

try:
    from .moe_token_utils import auto_tile_h, auto_tile_t, is_fp32_dtype, pad_first_dim, pad_last_dim
except ImportError:
    from moe_token_utils import auto_tile_h, auto_tile_t, is_fp32_dtype, pad_first_dim, pad_last_dim

try:
    import tilelang
    import tilelang.language as T

    HAS_TILELANG = True

    PASS_CONFIGS = {
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    }

    PASS_CONFIGS_EXPERT = {
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    }
except ImportError:
    HAS_TILELANG = False
    T = None
    PASS_CONFIGS = {}

CAST_LOW2HIGH = "CAST_NONE"
CAST_HIGH2LOW = "CAST_RINT"


def _auto_launch_cores_for_probs(num_tokens: int, hidden_size: int, num_cores: int) -> int:
    if num_tokens <= 64 and hidden_size <= 256:
        return 1
    if num_tokens <= 256 and hidden_size <= 512:
        return int(min(num_cores, max(1, num_tokens), 4))
    return int(min(num_cores, max(1, num_tokens)))


def _build_scatter_kernel_no_probs(
    E: int,
    hidden_size: int,
    padded_E: int,
    n_etiles: int,
    tiles_per_core: int,
    actual_cores: int,
    n_htiles: int,
    TILE_E: int,
    TILE_H: int,
    dtype: str,
    idx_dtype: str,
):
    HALF_H = TILE_H // 2
    if TILE_E >= 8:
        B = 8
    elif TILE_E >= 4:
        B = 4
    else:
        B = 2

    B = min(B, TILE_E)

    @tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS)
    def _build(
        E,
        hidden_size,
        padded_E,
        n_etiles,
        tiles_per_core,
        actual_cores,
        n_htiles,
        TILE_E,
        TILE_H,
        dtype,
        idx_dtype,
        HALF_H,
        B,
    ):
        @T.prim_func
        def moe_token_unpermute_grad(
            out_grad_gm: T.Tensor([E, hidden_size], dtype),
            sorted_idx_gm: T.Tensor([1, padded_E], idx_dtype),
            perm_grad_gm: T.Tensor([E, hidden_size], dtype),
        ):
            with T.Kernel(actual_cores, is_npu=True) as (cid, vid):
                idx_ub = T.alloc_ub([1, TILE_E], idx_dtype)
                row_buf = T.alloc_ub([B, HALF_H], dtype)

                with T.Scope("V"):
                    for t_local in T.serial(tiles_per_core):
                        tt = cid * tiles_per_core + t_local
                        if tt < n_etiles:
                            e_base = tt * TILE_E
                            T.copy(sorted_idx_gm[0, e_base], idx_ub)
                            T.barrier_all()

                            n_groups = TILE_E // B
                            for gi in T.serial(n_groups):
                                ei = gi * B
                                for ht in T.serial(n_htiles):
                                    h_off = ht * TILE_H + vid * HALF_H
                                    for lane in T.serial(B):
                                        row = e_base + ei + lane
                                        if row < E:
                                            T.copy(
                                                out_grad_gm[row, h_off],
                                                row_buf[lane, :],
                                            )
                                    T.barrier_all()
                                    for lane in T.serial(B):
                                        row = e_base + ei + lane
                                        if row < E:
                                            dst = idx_ub[0, ei + lane]
                                            T.copy(
                                                row_buf[lane, :],
                                                perm_grad_gm[dst, h_off],
                                            )
                                    T.barrier_all()

                            remainder = TILE_E % B
                            for r in T.serial(remainder):
                                ei = n_groups * B + r
                                row = e_base + ei
                                if row < E:
                                    dst = idx_ub[0, ei]
                                    for ht in T.serial(n_htiles):
                                        h_off = ht * TILE_H + vid * HALF_H
                                        T.copy(out_grad_gm[row, h_off], row_buf[0, :])
                                        T.barrier_all()
                                        T.copy(row_buf[0, :], perm_grad_gm[dst, h_off])
                                        T.barrier_all()

        return moe_token_unpermute_grad

    return _build(
        E,
        hidden_size,
        padded_E,
        n_etiles,
        tiles_per_core,
        actual_cores,
        n_htiles,
        TILE_E,
        TILE_H,
        dtype,
        idx_dtype,
        HALF_H,
        B,
    )


def _build_grad_kernel_with_probs(
    num_tokens: int,
    topK: int,
    hidden_size: int,
    E: int,
    padded_tokens: int,
    padded_E: int,
    actual_cores: int,
    n_htiles: int,
    TILE_H: int,
    dtype: str,
    idx_dtype: str,
    acc_dtype: str,
):
    if dtype == acc_dtype:
        return _build_grad_kernel_with_probs_f32(
            num_tokens,
            topK,
            hidden_size,
            E,
            padded_tokens,
            padded_E,
            actual_cores,
            n_htiles,
            TILE_H,
            dtype,
            idx_dtype,
        )

    tokens_per_core = int(math.ceil(num_tokens / actual_cores))
    HALF_H = TILE_H // 2

    B = min(8, topK)
    n_groups = topK // B
    remainder = topK % B

    @tilelang.jit(out_idx=[4, 5], pass_configs=PASS_CONFIGS)
    def _build_fast(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_tokens,
        padded_E,
        actual_cores,
        n_htiles,
        TILE_H,
        dtype,
        idx_dtype,
        acc_dtype,
        tokens_per_core,
        HALF_H,
        B,
        n_groups,
        remainder,
    ):
        perm_buf_shape = [B, HALF_H]

        @T.macro
        def emit_lane(
            k_idx,
            lane_idx,
            h_off,
            idx_ub,
            probs_f32,
            grad_f32,
            perm_buf,
            perm_tmp,
            perm_f32,
            mul_buf,
            out_tmp,
            reduce_dst,
            pg_acc,
            perm_grad_gm,
        ):
            dst = idx_ub[0, k_idx]
            prob = probs_f32[0, k_idx]

            T.wait_flag("mte3", "v", 2)

            T.tile.mul(mul_buf, grad_f32, prob)
            T.tile.cast(out_tmp, mul_buf, CAST_HIGH2LOW, HALF_H)

            T.set_flag("v", "mte3", 2)
            T.wait_flag("v", "mte3", 2)
            T.copy(out_tmp, perm_grad_gm[dst, h_off])
            T.set_flag("mte3", "v", 2)

            T.copy(perm_buf[lane_idx, :], perm_tmp)
            T.tile.cast(perm_f32, perm_tmp, CAST_LOW2HIGH, HALF_H)
            T.tile.mul(mul_buf, grad_f32, perm_f32)
            T.reduce_sum(mul_buf, reduce_dst, dim=-1)
            pg_acc[0, k_idx] = pg_acc[0, k_idx] + reduce_dst[0, 0]

        @T.prim_func
        def moe_token_unpermute_grad(
            perm_tokens_gm: T.Tensor([E, hidden_size], dtype),
            out_grad_gm: T.Tensor([num_tokens, hidden_size], dtype),
            sorted_idx_gm: T.Tensor([1, padded_E], idx_dtype),
            probs_gm: T.Tensor([padded_tokens, topK], dtype),
            perm_grad_gm: T.Tensor([E, hidden_size], dtype),
            probs_grad_gm: T.Tensor([2, padded_tokens, topK], acc_dtype),
        ):
            with T.Kernel(actual_cores, is_npu=True) as (cid, vid):
                idx_ub = T.alloc_ub([1, topK], idx_dtype)
                probs_ub = T.alloc_ub([1, topK], dtype)
                probs_f32 = T.alloc_ub([1, topK], acc_dtype)
                grad_buf = T.alloc_ub([1, HALF_H], dtype)
                grad_f32 = T.alloc_ub([1, HALF_H], acc_dtype)
                perm_buf = T.alloc_ub(perm_buf_shape, dtype)
                perm_tmp = T.alloc_ub([1, HALF_H], dtype)
                perm_f32 = T.alloc_ub([1, HALF_H], acc_dtype)
                mul_buf = T.alloc_ub([1, HALF_H], acc_dtype)
                out_tmp = T.alloc_ub([1, HALF_H], dtype)
                reduce_dst = T.alloc_ub([1, 1], acc_dtype)
                pg_acc = T.alloc_ub([1, topK], acc_dtype)

                with T.Scope("V"):
                    for ti in T.serial(tokens_per_core):
                        i = cid * tokens_per_core + ti
                        if i < num_tokens:
                            T.copy(sorted_idx_gm[0, i * topK], idx_ub)
                            T.copy(probs_gm[i, 0], probs_ub)
                            T.set_flag("mte2", "v", 4)
                            T.wait_flag("mte2", "v", 4)
                            T.tile.cast(probs_f32, probs_ub, CAST_LOW2HIGH, topK)
                            T.tile.fill(pg_acc, 0.0)

                            T.set_flag("mte3", "v", 2)

                            for ht in T.serial(n_htiles):
                                h_off = ht * TILE_H + vid * HALF_H
                                T.copy(out_grad_gm[i, h_off], grad_buf)
                                T.set_flag("mte2", "v", 5)
                                T.wait_flag("mte2", "v", 5)
                                T.tile.cast(grad_f32, grad_buf, CAST_LOW2HIGH, HALF_H)

                                for g in T.serial(n_groups):
                                    k_base = g * B
                                    for lane in T.serial(B):
                                        dst = idx_ub[0, k_base + lane]
                                        T.copy(
                                            perm_tokens_gm[dst, h_off],
                                            perm_buf[lane, :],
                                        )
                                    T.set_flag("mte2", "v", 3)
                                    T.wait_flag("mte2", "v", 3)

                                    for lane in T.serial(B):
                                        emit_lane(
                                            k_base + lane,
                                            lane,
                                            h_off,
                                            idx_ub,
                                            probs_f32,
                                            grad_f32,
                                            perm_buf,
                                            perm_tmp,
                                            perm_f32,
                                            mul_buf,
                                            out_tmp,
                                            reduce_dst,
                                            pg_acc,
                                            perm_grad_gm,
                                        )

                                if remainder > 0:
                                    r_base = n_groups * B
                                    for r in T.serial(remainder):
                                        rk = r_base + r
                                        dst = idx_ub[0, rk]
                                        T.copy(perm_tokens_gm[dst, h_off], perm_buf[0, :])
                                        T.set_flag("mte2", "v", 3)
                                        T.wait_flag("mte2", "v", 3)
                                        emit_lane(
                                            rk,
                                            0,
                                            h_off,
                                            idx_ub,
                                            probs_f32,
                                            grad_f32,
                                            perm_buf,
                                            perm_tmp,
                                            perm_f32,
                                            mul_buf,
                                            out_tmp,
                                            reduce_dst,
                                            pg_acc,
                                            perm_grad_gm,
                                        )

                            T.wait_flag("mte3", "v", 2)

                            T.set_flag("v", "mte3", 4)
                            T.wait_flag("v", "mte3", 4)
                            T.copy(pg_acc, probs_grad_gm[vid, i, 0])
                            T.set_flag("mte3", "v", 4)
                            T.wait_flag("mte3", "v", 4)

        return moe_token_unpermute_grad

    return _build_fast(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_tokens,
        padded_E,
        actual_cores,
        n_htiles,
        TILE_H,
        dtype,
        idx_dtype,
        acc_dtype,
        tokens_per_core,
        HALF_H,
        B,
        n_groups,
        remainder,
    )


def _build_grad_kernel_with_probs_f32(
    num_tokens: int,
    topK: int,
    hidden_size: int,
    E: int,
    padded_tokens: int,
    padded_E: int,
    actual_cores: int,
    n_htiles: int,
    TILE_H: int,
    dtype: str,
    idx_dtype: str,
):
    tokens_per_core = int(math.ceil(num_tokens / actual_cores))

    @tilelang.jit(out_idx=[4, 5], pass_configs=PASS_CONFIGS)
    def _build(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_tokens,
        padded_E,
        actual_cores,
        n_htiles,
        TILE_H,
        dtype,
        idx_dtype,
        tokens_per_core,
    ):
        @T.prim_func
        def moe_token_unpermute_grad(
            perm_tokens_gm: T.Tensor([E, hidden_size], dtype),
            out_grad_gm: T.Tensor([num_tokens, hidden_size], dtype),
            sorted_idx_gm: T.Tensor([1, padded_E], idx_dtype),
            probs_gm: T.Tensor([padded_tokens, topK], dtype),
            perm_grad_gm: T.Tensor([E, hidden_size], dtype),
            probs_grad_gm: T.Tensor([padded_tokens, topK], dtype),
        ):
            with T.Kernel(actual_cores, is_npu=True) as (cid, vid):
                idx_ub = T.alloc_shared([1, topK], idx_dtype)
                probs_ub = T.alloc_shared([1, topK], dtype)
                grad_buf = T.alloc_shared([1, TILE_H], dtype)
                perm_buf = T.alloc_shared([1, TILE_H], dtype)
                mul_buf = T.alloc_shared([1, TILE_H], dtype)
                reduce_dst = T.alloc_shared([1, 1], dtype)
                pg_acc = T.alloc_shared([1, topK], dtype)

                for ti in T.serial(tokens_per_core):
                    i = cid * tokens_per_core + ti
                    if i < num_tokens:
                        T.copy(sorted_idx_gm[0, i * topK], idx_ub)
                        T.copy(probs_gm[i, 0], probs_ub)

                        T.tile.fill(pg_acc, 0.0)

                        for k in T.serial(topK):
                            dst_idx = idx_ub[0, k]
                            prob_val = probs_ub[0, k]

                            for ht in T.serial(n_htiles):
                                h_off = ht * TILE_H

                                T.copy(out_grad_gm[i, h_off], grad_buf)
                                T.copy(perm_tokens_gm[dst_idx, h_off], perm_buf)

                                T.copy(grad_buf, mul_buf)
                                T.tile.mul(mul_buf, mul_buf, prob_val)
                                T.copy(mul_buf, perm_grad_gm[dst_idx, h_off])

                                T.tile.mul(mul_buf, grad_buf, perm_buf)
                                T.reduce_sum(mul_buf, reduce_dst, dim=-1)
                                pg_acc[0, k] = pg_acc[0, k] + reduce_dst[0, 0]

                        T.copy(pg_acc, probs_grad_gm[i, 0])

        return moe_token_unpermute_grad

    return _build(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_tokens,
        padded_E,
        actual_cores,
        n_htiles,
        TILE_H,
        dtype,
        idx_dtype,
        tokens_per_core,
    )


def _compile_grad(
    num_tokens: int,
    topK: int,
    hidden_size: int,
    has_probs: bool = True,
    NUM_CORES: int = 24,
    TILE_T: int = None,
    TILE_H: int = None,
    dtype: str = "float16",
    idx_dtype: str = "int32",
    acc_dtype: str = "float32",
):
    if TILE_H is None:
        TILE_H = auto_tile_h(hidden_size)
    if TILE_T is None:
        total = num_tokens if has_probs else int(num_tokens * topK)
        TILE_T = auto_tile_t(total, NUM_CORES)

    min_tile_h = 64 if is_fp32_dtype(dtype) else 8
    if min_tile_h > TILE_H:
        TILE_H = min(hidden_size, min_tile_h)
    assert hidden_size % TILE_H == 0, f"hidden_size ({hidden_size}) must be a multiple of TILE_H ({TILE_H})!"
    assert HAS_TILELANG
    assert topK <= 512

    E = int(num_tokens * topK)
    n_htiles = int(hidden_size // TILE_H)

    if has_probs:
        padded_tokens = int(math.ceil(num_tokens / TILE_T) * TILE_T)
        padded_E = int(padded_tokens * topK)
        actual_cores = _auto_launch_cores_for_probs(num_tokens, hidden_size, NUM_CORES)

        compiled = _build_grad_kernel_with_probs(
            num_tokens,
            topK,
            hidden_size,
            E,
            padded_tokens,
            padded_E,
            actual_cores,
            n_htiles,
            TILE_H,
            dtype,
            idx_dtype,
            acc_dtype,
        )
        return compiled, padded_tokens, padded_E, actual_cores

    TILE_E = TILE_T
    padded_E = int(math.ceil(E / TILE_E) * TILE_E)
    n_etiles = int(padded_E // TILE_E)
    actual_cores = int(min(NUM_CORES, max(1, n_etiles)))
    tiles_per_core = int(math.ceil(n_etiles / actual_cores))

    compiled = _build_scatter_kernel_no_probs(
        E,
        hidden_size,
        padded_E,
        n_etiles,
        tiles_per_core,
        actual_cores,
        n_htiles,
        TILE_E,
        TILE_H,
        dtype,
        idx_dtype,
    )
    return compiled, 0, padded_E, actual_cores


class MoeTokenUnpermuteGrad:
    def __init__(
        self,
        num_tokens: int,
        topK: int,
        hidden_size: int,
        has_probs: bool = True,
        NUM_CORES: int = 24,
        TILE_T: int = None,
        TILE_H: int = None,
        dtype: str = "float16",
    ):
        assert topK <= 512
        assert dtype in ("float16", "bfloat16", "float32", "float")

        self.num_tokens = num_tokens
        self.topK = topK
        self.hidden_size = hidden_size
        self.has_probs = has_probs
        self.dtype = dtype
        self.E = num_tokens * topK
        min_compile_h = 64 if is_fp32_dtype(dtype) else 32
        self._compile_hidden_size = max(hidden_size, min_compile_h)
        compile_tile_h = TILE_H if TILE_H is None else max(TILE_H, min_compile_h)

        self._kernel, self._padded_tokens, self._padded_E, self._actual_cores = _compile_grad(
            num_tokens,
            topK,
            self._compile_hidden_size,
            has_probs=has_probs,
            NUM_CORES=NUM_CORES,
            TILE_T=TILE_T,
            TILE_H=compile_tile_h,
            dtype=dtype,
        )

    def __call__(self, permuted_tokens, unpermuted_tokens_grad, sorted_indices, probs=None):
        if self._kernel is None:
            raise RuntimeError("tilelang not installed")

        indices_padded_2d = pad_first_dim(sorted_indices, self._padded_E).unsqueeze(0)
        permuted_tokens_in = pad_last_dim(permuted_tokens, self._compile_hidden_size)
        unperm_grad_in = pad_last_dim(unpermuted_tokens_grad, self._compile_hidden_size)

        if self.has_probs:
            assert probs is not None, "has_probs=True but probs is not provided"
            probs_padded = pad_first_dim(probs, self._padded_tokens)
            perm_grad, probs_grad_raw = self._kernel(permuted_tokens_in, unperm_grad_in, indices_padded_2d, probs_padded)
            perm_grad = perm_grad[:, : self.hidden_size].contiguous()

            probs_grad = probs_grad_raw[0] + probs_grad_raw[1] if probs_grad_raw.dim() == 3 else probs_grad_raw
            probs_grad = probs_grad[: self.num_tokens, : self.topK]
            if probs_grad.dtype != probs.dtype:
                probs_grad = probs_grad.to(probs.dtype)
            return perm_grad, probs_grad

        perm_grad = self._kernel(unperm_grad_in, indices_padded_2d)
        return perm_grad[:, : self.hidden_size].contiguous()

    def __repr__(self):
        return f"MoeTokenUnpermuteGrad(T={self.num_tokens}, K={self.topK}, H={self.hidden_size}, probs={self.has_probs}, cores={self._actual_cores})"


def test_unpermute_grad_parameterized(pt_dtype, tl_dtype_str):
    print(f"\n{'=' * 65}")
    print(f"Testing MoeTokenUnpermuteGrad, dtype: {tl_dtype_str.upper()}")
    print(f"{'=' * 65}")

    torch.manual_seed(42)
    all_passed = True

    num_tokens = 8
    hidden_size = 4
    topk = 2

    E = num_tokens * topk

    permuted_tokens = torch.randn(
        E,
        hidden_size,
        dtype=pt_dtype,
        device="npu",
        requires_grad=True,
    )
    sorted_indices = torch.randperm(
        E,
        dtype=torch.int32,
        device="npu",
    )
    probs = torch.randn(
        num_tokens,
        topk,
        dtype=pt_dtype,
        device="npu",
        requires_grad=True,
    )

    print(f"\n{'-' * 65}")
    print("  Part 1: has_probs=True")
    print(f"{'-' * 65}")

    tokens_fwd = torch_npu.npu_moe_token_unpermute(
        permuted_tokens,
        sorted_indices,
        probs,
    )
    grad_tokens = torch.ones_like(tokens_fwd, dtype=pt_dtype, device="npu")
    tokens_fwd.backward(grad_tokens)

    ref_permuted_tokens_grad = permuted_tokens.grad.clone()
    ref_probs_grad = probs.grad.clone()

    tl_op = MoeTokenUnpermuteGrad(
        num_tokens=num_tokens,
        topK=topk,
        hidden_size=hidden_size,
        has_probs=True,
        NUM_CORES=24,
        TILE_H=4,
        dtype=tl_dtype_str,
    )

    tl_permuted_tokens_grad, tl_probs_grad = tl_op(
        permuted_tokens.detach(),
        grad_tokens,
        sorted_indices,
        probs=probs.detach(),
    )

    print(f"\n>>> Verifying permuted_tokens_grad (shape: [{E}, {hidden_size}])")
    print(f"    ref shape: {ref_permuted_tokens_grad.shape}, tl shape: {tl_permuted_tokens_grad.shape}")

    try:
        torch.testing.assert_close(tl_permuted_tokens_grad, ref_permuted_tokens_grad)
        print(f"    [PASS] {tl_dtype_str.upper()} permuted_tokens_grad precision test passed!")
    except Exception as e:
        print(f"    [FAILED] {tl_dtype_str.upper()} permuted_tokens_grad precision test failed!")
        max_diff = (tl_permuted_tokens_grad - ref_permuted_tokens_grad).abs().max().item()
        print(f"    Max absolute error: {max_diff}")
        print(e)
        all_passed = False

    print(f"\n>>> Verifying probs_grad (shape: [{num_tokens}, {topk}])")
    print(f"    ref shape: {ref_probs_grad.shape}, tl shape: {tl_probs_grad.shape}")

    try:
        torch.testing.assert_close(tl_probs_grad, ref_probs_grad)
        print(f"    [PASS] {tl_dtype_str.upper()} probs_grad precision test passed!")
    except Exception as e:
        print(f"    [FAILED] {tl_dtype_str.upper()} probs_grad precision test failed!")
        max_diff = (tl_probs_grad - ref_probs_grad).abs().max().item()
        print(f"    Max absolute error: {max_diff}")
        print(e)
        all_passed = False

    print(f"\n{'-' * 65}")
    print("  Part 2: has_probs=False")
    print(f"{'-' * 65}")

    permuted_tokens_np = torch.randn(
        E,
        hidden_size,
        dtype=pt_dtype,
        device="npu",
        requires_grad=True,
    )

    tokens_fwd_np = torch_npu.npu_moe_token_unpermute(
        permuted_tokens_np,
        sorted_indices,
    )
    grad_tokens_np = torch.ones_like(tokens_fwd_np, dtype=pt_dtype, device="npu")
    tokens_fwd_np.backward(grad_tokens_np)

    ref_permuted_tokens_grad_np = permuted_tokens_np.grad.clone()

    tl_op_np = MoeTokenUnpermuteGrad(
        num_tokens=num_tokens,
        topK=topk,
        hidden_size=hidden_size,
        has_probs=False,
        NUM_CORES=24,
        TILE_H=4,
        dtype=tl_dtype_str,
    )

    tl_permuted_tokens_grad_np = tl_op_np(
        permuted_tokens_np.detach(),
        grad_tokens_np,
        sorted_indices,
    )

    print(f"\n>>> Verifying permuted_tokens_grad (shape: ref {ref_permuted_tokens_grad_np.shape}, tl {tl_permuted_tokens_grad_np.shape})")

    try:
        torch.testing.assert_close(tl_permuted_tokens_grad_np, ref_permuted_tokens_grad_np)
        print(f"    [PASS] {tl_dtype_str.upper()} no-probs permuted_tokens_grad precision test passed!")
    except Exception as e:
        print(f"    [FAILED] {tl_dtype_str.upper()} no-probs permuted_tokens_grad precision test failed!")
        max_diff = (tl_permuted_tokens_grad_np - ref_permuted_tokens_grad_np).abs().max().item()
        print(f"    Max absolute error: {max_diff}")
        print(f"    ref:\n{ref_permuted_tokens_grad_np}")
        print(f"    tl:\n{tl_permuted_tokens_grad_np}")
        print(e)
        all_passed = False

    return all_passed


def test_unpermute_grad():
    dtypes_to_test = [
        (torch.float16, "float16"),
        (torch.bfloat16, "bfloat16"),
        (torch.float32, "float32"),
    ]

    overall_passed = True
    for pt_type, tl_type_str in dtypes_to_test:
        passed = test_unpermute_grad_parameterized(pt_dtype=pt_type, tl_dtype_str=tl_type_str)
        if not passed:
            overall_passed = False

    print(f"\n{'=' * 65}")
    if overall_passed:
        print("Test passed!")
    else:
        print("Test failed! The precision is not correct!")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    test_unpermute_grad()
