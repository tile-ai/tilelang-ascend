import math
import torch
import torch_npu

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
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    }
except ImportError:
    HAS_TILELANG = False
    T = None
    PASS_CONFIGS = {}

CAST_LOW2HIGH = "CAST_NONE"
CAST_HIGH2LOW = "CAST_RINT"


def is_fp32_dtype(dtype: str) -> bool:
    return dtype in ("float32", "float")


def auto_tile_h(hidden_size: int, dtype: str) -> int:
    dtype_scale = 2 if is_fp32_dtype(dtype) else 1
    max_tile_h = 4096 // dtype_scale
    for candidate in [hidden_size, max_tile_h, 2048 // dtype_scale, 1024, 512, 256]:
        if candidate > 0 and hidden_size % candidate == 0:
            return candidate
    return 256


def auto_tile_t(total: int, num_cores: int, large_candidates):
    if total < num_cores:
        for candidate in [64, 32, 16, 8, 4, 2, 1]:
            if candidate <= total and total % candidate == 0:
                return candidate
        return max(1, total)
    for candidate in large_candidates:
        if total // candidate >= num_cores:
            return candidate
    return max(1, total // num_cores)


def auto_launch_cores(
    work_items: int,
    hidden_size: int,
    num_cores: int,
    small_work: int,
    small_hidden: int,
    mid_work: int,
    mid_hidden: int,
    mid_cap: int = 4,
) -> int:
    if work_items <= small_work and hidden_size <= small_hidden:
        return 1
    if work_items <= mid_work and hidden_size <= mid_hidden:
        return int(min(num_cores, max(1, work_items), mid_cap))
    return int(min(num_cores, max(1, work_items)))


def pad_first_dim(tensor: torch.Tensor, target_rows: int) -> torch.Tensor:
    if tensor.shape[0] >= target_rows:
        return tensor
    out = torch.zeros(
        (target_rows, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device
    )
    out[: tensor.shape[0]] = tensor
    return out


def pad_last_dim(tensor: torch.Tensor, target_cols: int) -> torch.Tensor:
    if tensor.shape[-1] >= target_cols:
        return tensor
    out = torch.zeros(
        (*tensor.shape[:-1], target_cols), dtype=tensor.dtype, device=tensor.device
    )
    out[..., : tensor.shape[-1]] = tensor
    return out


def _build_gather_kernel_with_probs(
    num_tokens: int,
    topK: int,
    hidden_size: int,
    E: int,
    padded_tokens: int,
    padded_E: int,
    n_ttiles: int,
    tiles_per_core: int,
    actual_cores: int,
    n_htiles: int,
    TILE_T: int,
    TILE_H: int,
    dtype: str,
    idx_dtype: str,
    acc_dtype: str,
):
    if dtype == acc_dtype:
        return _build_gather_kernel_with_probs_f32(
            num_tokens,
            topK,
            hidden_size,
            E,
            padded_tokens,
            padded_E,
            n_ttiles,
            tiles_per_core,
            actual_cores,
            n_htiles,
            TILE_T,
            TILE_H,
            dtype,
            idx_dtype,
        )

    tokens_per_core = int(math.ceil(num_tokens / actual_cores))
    HALF_H = TILE_H // 2

    BATCH_T = min(tokens_per_core, max(1, 4096 // (topK * 10)))
    while BATCH_T > 1 and tokens_per_core % BATCH_T != 0:
        BATCH_T -= 1
    n_batches = int(math.ceil(tokens_per_core / BATCH_T))

    LANES_PER_ITER = min(8, topK)
    n_iters = topK // LANES_PER_ITER
    rem = topK % LANES_PER_ITER

    @tilelang.jit(out_idx=[3], pass_configs=PASS_CONFIGS_EXPERT)
    def _build(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_tokens,
        padded_E,
        n_ttiles,
        tiles_per_core,
        actual_cores,
        n_htiles,
        TILE_T,
        TILE_H,
        dtype,
        idx_dtype,
        acc_dtype,
        tokens_per_core,
        HALF_H,
        BATCH_T,
        n_batches,
        LANES_PER_ITER,
        n_iters,
        rem,
    ):
        @T.macro
        def cast_axpy(slot, prob, row_buf, row_tmp, row_f32, acc_buf):
            T.copy(row_buf[slot, :], row_tmp)
            T.tile.cast(row_f32, row_tmp, CAST_LOW2HIGH, HALF_H)
            T.tile.axpy(acc_buf, row_f32, prob)

        @T.prim_func
        def moe_token_unpermute(
            perm_tokens_gm: T.Tensor([E, hidden_size], dtype),
            sorted_idx_gm: T.Tensor([1, padded_E], idx_dtype),
            probs_gm: T.Tensor([1, padded_tokens * topK], dtype),
            out_gm: T.Tensor([num_tokens, hidden_size], dtype),
        ):
            with T.Kernel(actual_cores, is_npu=True) as (cid, vid):
                idx_ub = T.alloc_ub([1, BATCH_T * topK], idx_dtype)
                probs_ub = T.alloc_ub([1, BATCH_T * topK], dtype)
                probs_f32 = T.alloc_ub([1, BATCH_T * topK], acc_dtype)
                row_buf = T.alloc_ub([LANES_PER_ITER, HALF_H], dtype)
                row_tmp = T.alloc_ub([1, HALF_H], dtype)
                row_f32 = T.alloc_ub([1, HALF_H], acc_dtype)
                acc_buf = T.alloc_ub([1, HALF_H], acc_dtype)
                out_buf = T.alloc_ub([1, HALF_H], dtype)

                with T.Scope("V"):
                    for batch_id in T.serial(n_batches):
                        batch_base = cid * tokens_per_core + batch_id * BATCH_T

                        T.copy(sorted_idx_gm[0, batch_base * topK], idx_ub)
                        T.copy(probs_gm[0, batch_base * topK], probs_ub)
                        T.barrier_all()
                        T.tile.cast(probs_f32, probs_ub, CAST_LOW2HIGH, BATCH_T * topK)

                        for ti in T.serial(BATCH_T):
                            i = batch_base + ti
                            if i < num_tokens:
                                for ht in T.serial(n_htiles):
                                    h_off = ht * TILE_H + vid * HALF_H
                                    tk_off = ti * topK

                                    T.tile.fill(acc_buf, 0.0)

                                    for jj in T.serial(n_iters):
                                        base = jj * LANES_PER_ITER
                                        for lane in T.serial(LANES_PER_ITER):
                                            src = idx_ub[0, tk_off + base + lane]
                                            T.copy(
                                                perm_tokens_gm[src, h_off],
                                                row_buf[lane, :],
                                            )
                                        T.barrier_all()
                                        for lane in T.serial(LANES_PER_ITER):
                                            prob = probs_f32[0, tk_off + base + lane]
                                            cast_axpy(
                                                lane,
                                                prob,
                                                row_buf,
                                                row_tmp,
                                                row_f32,
                                                acc_buf,
                                            )

                                    if rem > 0:
                                        base = n_iters * LANES_PER_ITER
                                        for lane in T.serial(rem):
                                            src = idx_ub[0, tk_off + base + lane]
                                            T.copy(
                                                perm_tokens_gm[src, h_off],
                                                row_buf[lane, :],
                                            )
                                        T.barrier_all()
                                        for lane in T.serial(rem):
                                            prob = probs_f32[0, tk_off + base + lane]
                                            cast_axpy(
                                                lane,
                                                prob,
                                                row_buf,
                                                row_tmp,
                                                row_f32,
                                                acc_buf,
                                            )

                                    T.barrier_all()
                                    T.tile.cast(out_buf, acc_buf, CAST_HIGH2LOW, HALF_H)
                                    T.pipe_barrier("v")
                                    T.copy(out_buf, out_gm[i, h_off])
                                    T.pipe_barrier("mte3")

        return moe_token_unpermute

    return _build(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_tokens,
        padded_E,
        n_ttiles,
        tiles_per_core,
        actual_cores,
        n_htiles,
        TILE_T,
        TILE_H,
        dtype,
        idx_dtype,
        acc_dtype,
        tokens_per_core,
        HALF_H,
        BATCH_T,
        n_batches,
        LANES_PER_ITER,
        n_iters,
        rem,
    )


def _build_gather_kernel_with_probs_f32(
    num_tokens: int,
    topK: int,
    hidden_size: int,
    E: int,
    padded_tokens: int,
    padded_E: int,
    n_ttiles: int,
    tiles_per_core: int,
    actual_cores: int,
    n_htiles: int,
    TILE_T: int,
    TILE_H: int,
    dtype: str,
    idx_dtype: str,
):
    tokens_per_core = int(math.ceil(num_tokens / actual_cores))
    BATCH_T = min(tokens_per_core, max(1, 4096 // (topK * 12)))
    while BATCH_T > 1 and tokens_per_core % BATCH_T != 0:
        BATCH_T -= 1
    n_batches = int(math.ceil(tokens_per_core / BATCH_T))

    @tilelang.jit(out_idx=[3], pass_configs=PASS_CONFIGS)
    def _build(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_tokens,
        padded_E,
        n_ttiles,
        tiles_per_core,
        actual_cores,
        n_htiles,
        TILE_T,
        TILE_H,
        dtype,
        idx_dtype,
        tokens_per_core,
        BATCH_T,
        n_batches,
    ):
        @T.prim_func
        def moe_token_unpermute(
            perm_tokens_gm: T.Tensor([E, hidden_size], dtype),
            sorted_idx_gm: T.Tensor([1, padded_E], idx_dtype),
            probs_gm: T.Tensor([1, padded_tokens * topK], dtype),
            out_gm: T.Tensor([num_tokens, hidden_size], dtype),
        ):
            with T.Kernel(actual_cores, is_npu=True) as (cid, vid):
                idx_ub = T.alloc_shared([1, BATCH_T * topK], idx_dtype)
                probs_ub = T.alloc_shared([1, BATCH_T * topK], dtype)
                row_buf = T.alloc_shared([3, TILE_H], dtype)
                acc_buf = T.alloc_shared([1, TILE_H], dtype)

                for batch_id in T.serial(n_batches):
                    batch_base = cid * tokens_per_core + batch_id * BATCH_T

                    T.copy(sorted_idx_gm[0, batch_base * topK], idx_ub)
                    T.copy(probs_gm[0, batch_base * topK], probs_ub)

                    for ti in T.serial(BATCH_T):
                        i = batch_base + ti
                        if i < num_tokens:
                            for ht in T.serial(n_htiles):
                                h_off = ht * TILE_H
                                tk_off = ti * topK

                                src_row_0 = idx_ub[0, tk_off]
                                prob_val_0 = probs_ub[0, tk_off]
                                T.copy(perm_tokens_gm[src_row_0, h_off], row_buf[0, :])
                                T.copy(row_buf[0, :], acc_buf)
                                T.tile.mul(acc_buf, acc_buf, prob_val_0)

                                n_triples = (topK - 1) // 3
                                remainder = (topK - 1) % 3

                                for j3 in T.serial(n_triples):
                                    j = j3 * 3
                                    for lane in T.serial(3):
                                        off = j + lane + 1
                                        src = idx_ub[0, tk_off + off]
                                        T.copy(
                                            perm_tokens_gm[src, h_off],
                                            row_buf[lane, :],
                                        )
                                    for lane in T.serial(3):
                                        off = j + lane + 1
                                        prob = probs_ub[0, tk_off + off]
                                        T.tile.axpy(acc_buf, row_buf[lane, :], prob)

                                for r in T.serial(remainder):
                                    off = n_triples * 3 + r + 1
                                    src = idx_ub[0, tk_off + off]
                                    prob = probs_ub[0, tk_off + off]
                                    T.copy(perm_tokens_gm[src, h_off], row_buf[r, :])
                                    T.tile.axpy(acc_buf, row_buf[r, :], prob)

                                T.copy(acc_buf, out_gm[i, h_off])

        return moe_token_unpermute

    return _build(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_tokens,
        padded_E,
        n_ttiles,
        tiles_per_core,
        actual_cores,
        n_htiles,
        TILE_T,
        TILE_H,
        dtype,
        idx_dtype,
        tokens_per_core,
        BATCH_T,
        n_batches,
    )


def _build_gather_kernel_no_probs(
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

    @tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS_EXPERT)
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
    ):
        @T.prim_func
        def moe_token_unpermute(
            perm_tokens_gm: T.Tensor([E, hidden_size], dtype),
            sorted_idx_gm: T.Tensor([1, padded_E], idx_dtype),
            out_gm: T.Tensor([E, hidden_size], dtype),
        ):
            with T.Kernel(actual_cores, is_npu=True) as (cid, vid):
                idx_ub = T.alloc_ub([1, TILE_E], idx_dtype)
                row_buf = T.alloc_ub([4, HALF_H], dtype)

                with T.Scope("V"):
                    for t_local in T.serial(tiles_per_core):
                        tt = cid * tiles_per_core + t_local
                        if tt < n_etiles:
                            e_base = tt * TILE_E
                            T.copy(sorted_idx_gm[0, e_base], idx_ub)
                            T.barrier_all()

                            n_quads = TILE_E // 4
                            for ei4 in T.serial(n_quads):
                                ei = ei4 * 4
                                for ht in T.serial(n_htiles):
                                    h_off = ht * TILE_H + vid * HALF_H
                                    for lane in T.serial(4):
                                        row = e_base + ei + lane
                                        if row < E:
                                            src = idx_ub[0, ei + lane]
                                            T.copy(
                                                perm_tokens_gm[src, h_off],
                                                row_buf[lane, :],
                                            )
                                    T.barrier_all()
                                    for lane in T.serial(4):
                                        row = e_base + ei + lane
                                        if row < E:
                                            T.copy(row_buf[lane, :], out_gm[row, h_off])
                                    T.pipe_barrier("mte3")

                            remainder = TILE_E % 4
                            for r in T.serial(remainder):
                                ei = n_quads * 4 + r
                                row = e_base + ei
                                if row < E:
                                    src = idx_ub[0, ei]
                                    for ht in T.serial(n_htiles):
                                        h_off = ht * TILE_H + vid * HALF_H
                                        T.copy(
                                            perm_tokens_gm[src, h_off], row_buf[0, :]
                                        )
                                        T.barrier_all()
                                        T.copy(row_buf[0, :], out_gm[row, h_off])
                                        T.pipe_barrier("mte3")

        return moe_token_unpermute

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
    )


def _compile_gather(
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
        TILE_H = auto_tile_h(hidden_size, dtype)
    if TILE_T is None:
        total = num_tokens if has_probs else int(num_tokens * topK)
        large_candidates = (
            [64, 32, 16, 8, 4, 2, 1] if has_probs else [128, 64, 32, 16, 8, 4, 2, 1]
        )
        TILE_T = auto_tile_t(total, NUM_CORES, large_candidates)

    min_tile_h = 64 if is_fp32_dtype(dtype) else 8
    if min_tile_h > TILE_H:
        TILE_H = min(hidden_size, min_tile_h)
    assert hidden_size % TILE_H == 0, (
        f"hidden_size ({hidden_size}) must be a multiple of TILE_H ({TILE_H})!"
    )
    assert HAS_TILELANG, "tilelang is required"
    assert topK <= 512, "topK ≤ 512 (Atlas A2/A3)"

    E = int(num_tokens * topK)
    n_htiles = int(hidden_size // TILE_H)

    if has_probs:
        padded_tokens = int(math.ceil(num_tokens / TILE_T) * TILE_T)
        padded_E = int(padded_tokens * topK)
        n_ttiles = int(padded_tokens // TILE_T)
        actual_cores = auto_launch_cores(
            n_ttiles,
            hidden_size,
            NUM_CORES,
            small_work=64,
            small_hidden=256,
            mid_work=256,
            mid_hidden=512,
            mid_cap=4,
        )
        tiles_per_core = int(math.ceil(n_ttiles / actual_cores))

        tokens_per_core = int(math.ceil(num_tokens / actual_cores))
        required_span = actual_cores * tokens_per_core
        if required_span > padded_tokens:
            padded_tokens = int(math.ceil(required_span / TILE_T) * TILE_T)
            padded_E = int(padded_tokens * topK)
            n_ttiles = int(padded_tokens // TILE_T)
            tiles_per_core = int(math.ceil(n_ttiles / actual_cores))

        compiled = _build_gather_kernel_with_probs(
            num_tokens,
            topK,
            hidden_size,
            E,
            padded_tokens,
            padded_E,
            n_ttiles,
            tiles_per_core,
            actual_cores,
            n_htiles,
            TILE_T,
            TILE_H,
            dtype,
            idx_dtype,
            acc_dtype,
        )
        return compiled, padded_tokens, padded_E, actual_cores

    TILE_E = TILE_T
    padded_E = int(math.ceil(E / TILE_E) * TILE_E)
    n_etiles = int(padded_E // TILE_E)

    actual_cores = auto_launch_cores(
        n_etiles,
        hidden_size,
        NUM_CORES,
        small_work=256,
        small_hidden=256,
        mid_work=1024,
        mid_hidden=512,
        mid_cap=4,
    )
    tiles_per_core = int(math.ceil(n_etiles / actual_cores))

    compiled = _build_gather_kernel_no_probs(
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


class MoeTokenUnpermute:
    def __init__(
        self,
        num_tokens: int,
        topK: int,
        hidden_size: int,
        has_probs: bool = True,
        padded_mode: bool = False,
        NUM_CORES: int = 24,
        TILE_T: int = None,
        TILE_H: int = None,
        dtype: str = "float16",
    ):
        if padded_mode:
            raise NotImplementedError("paddedMode=True not supported.")
        assert topK <= 512
        assert dtype in ("float16", "bfloat16", "float32", "float"), (
            f"dtype must be float16/bfloat16/float32, got {dtype}"
        )

        self.num_tokens = num_tokens
        self.topK = topK
        self.hidden_size = hidden_size
        self.has_probs = has_probs
        self.dtype = dtype
        self.E = num_tokens * topK
        min_compile_h = 64 if is_fp32_dtype(dtype) else 32
        self._compile_hidden_size = max(hidden_size, min_compile_h)
        compile_tile_h = TILE_H if TILE_H is None else max(TILE_H, min_compile_h)

        self._kernel, self._padded_tokens, self._padded_E, self._actual_cores = (
            _compile_gather(
                num_tokens,
                topK,
                self._compile_hidden_size,
                has_probs=has_probs,
                NUM_CORES=NUM_CORES,
                TILE_T=TILE_T,
                TILE_H=compile_tile_h,
                dtype=dtype,
            )
        )

    def __call__(self, permuted_tokens, sorted_indices, probs=None):
        if self._kernel is None:
            raise RuntimeError("tilelang not installed")

        indices_padded_2d = pad_first_dim(sorted_indices, self._padded_E).unsqueeze(0)
        permuted_tokens_in = pad_last_dim(permuted_tokens, self._compile_hidden_size)

        if self.has_probs:
            assert probs is not None, "has_probs=True but probs is not provided"
            probs_padded = pad_first_dim(probs, self._padded_tokens)

            probs_flat = probs_padded.contiguous().reshape(1, -1)
            out = self._kernel(permuted_tokens_in, indices_padded_2d, probs_flat)
            return out[:, : self.hidden_size].contiguous()

        out = self._kernel(permuted_tokens_in, indices_padded_2d)
        return out[:, : self.hidden_size].contiguous()

    def __repr__(self):
        return f"MoeTokenUnpermute(T={self.num_tokens}, K={self.topK}, H={self.hidden_size}, probs={self.has_probs}, cores={self._actual_cores})"


def test_unpermute_parameterized(pt_dtype, tl_dtype_str):
    print(f"\n{'=' * 65}")
    print(f"Testing MoeTokenUnpermute, dtype: {tl_dtype_str.upper()}")
    print(f"{'=' * 65}")

    torch.manual_seed(42)
    all_passed = True

    print(
        ">>> Test case 1: Forward alignment test without probs (N=16, H=8, K=4) -> output [64, 8]"
    )

    num_tokens = 16
    hidden_size = 8
    topk = 4

    permuted_tokens = torch.randn(
        num_tokens * topk, hidden_size, dtype=pt_dtype, device="npu"
    )
    sorted_indices = torch.randperm(num_tokens * topk, dtype=torch.int32, device="npu")

    npu_tokens = torch_npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices)

    tl_op = MoeTokenUnpermute(
        num_tokens=num_tokens,
        topK=topk,
        hidden_size=hidden_size,
        has_probs=False,
        TILE_T=16,
        TILE_H=8,
        dtype=tl_dtype_str,
    )
    tl_tokens = tl_op(permuted_tokens, sorted_indices)

    print(
        f"    npu_tokens shape: {npu_tokens.shape}, tl_tokens shape: {tl_tokens.shape}"
    )

    try:
        torch.testing.assert_close(tl_tokens, npu_tokens)
        print(
            f"    [PASS] {tl_dtype_str.upper()} Forward without probs precision test passed!"
        )
    except AssertionError as e:
        print(
            f"    [FAILED] {tl_dtype_str.upper()} Forward without probs precision test failed!"
        )
        max_diff = (tl_tokens - npu_tokens).abs().max().item()
        print(f"    Max absolute error: {max_diff}")
        print(e)
        all_passed = False

    print(
        "\n>>> Test case 2: Weighted forward alignment test with probs (N=8, H=4, K=2) -> output [8, 4]"
    )

    torch.manual_seed(42)

    num_tokens = 8
    hidden_size = 4
    topk = 2

    permuted_tokens_2 = torch.randn(
        num_tokens * topk, hidden_size, dtype=pt_dtype, device="npu"
    )
    sorted_indices_2 = torch.randperm(
        num_tokens * topk, dtype=torch.int32, device="npu"
    )
    probs_2 = torch.randn(num_tokens, topk, dtype=pt_dtype, device="npu")

    npu_tokens_2 = torch_npu.npu_moe_token_unpermute(
        permuted_tokens_2, sorted_indices_2, probs_2
    )

    tl_op_2 = MoeTokenUnpermute(
        num_tokens=num_tokens,
        topK=topk,
        hidden_size=hidden_size,
        has_probs=True,
        TILE_T=8,
        TILE_H=4,
        dtype=tl_dtype_str,
    )
    tl_tokens_2 = tl_op_2(permuted_tokens_2, sorted_indices_2, probs_2)

    print(
        f"    npu_tokens shape: {npu_tokens_2.shape}, tl_tokens shape: {tl_tokens_2.shape}"
    )

    try:
        torch.testing.assert_close(tl_tokens_2, npu_tokens_2)
        print(
            f"    [PASS] {tl_dtype_str.upper()} Weighted with-probs precision test passed!"
        )
    except AssertionError as e:
        print(
            f"    [FAILED] {tl_dtype_str.upper()} Weighted with-probs precision test failed!"
        )
        max_diff = (tl_tokens_2 - npu_tokens_2).abs().max().item()
        print(f"    Max absolute error: {max_diff}")
        print(e)
        all_passed = False

    print(
        "\n>>> Test case 3: permute -> unpermute round-trip consistency test (N=16, H=8, K=4)"
    )

    torch.manual_seed(42)

    num_tokens = 16
    hidden_size = 8
    topk = 4

    tokens_3 = torch.randn(num_tokens, hidden_size, dtype=pt_dtype, device="npu")
    indices_3 = torch.randint(0, 4, (num_tokens, topk), dtype=torch.int32, device="npu")

    npu_permuted_3, npu_sorted_idx_3 = torch_npu.npu_moe_token_permute(
        tokens_3, indices_3
    )
    npu_reconstruct_3 = torch_npu.npu_moe_token_unpermute(
        npu_permuted_3, npu_sorted_idx_3
    )

    tl_op_3 = MoeTokenUnpermute(
        num_tokens=num_tokens,
        topK=topk,
        hidden_size=hidden_size,
        has_probs=False,
        TILE_T=16,
        TILE_H=8,
        dtype=tl_dtype_str,
    )
    tl_reconstruct_3 = tl_op_3(npu_permuted_3, npu_sorted_idx_3)

    print(
        f"    npu_reconstruct shape: {npu_reconstruct_3.shape}, tl_reconstruct shape: {tl_reconstruct_3.shape}"
    )

    try:
        torch.testing.assert_close(tl_reconstruct_3, npu_reconstruct_3)
        print(
            f"    [PASS] {tl_dtype_str.upper()} permute -> unpermute round-trip consistency test passed!"
        )
    except AssertionError as e:
        print(
            f"    [FAILED] {tl_dtype_str.upper()} permute -> unpermute round-trip consistency test failed!"
        )
        max_diff = (tl_reconstruct_3 - npu_reconstruct_3).abs().max().item()
        print(f"    Max absolute error: {max_diff}")
        print(e)
        all_passed = False

    return all_passed


def test_unpermute():
    dtypes_to_test = [
        (torch.float16, "float16"),
        (torch.bfloat16, "bfloat16"),
        (torch.float32, "float32"),
    ]

    overall_passed = True
    for pt_type, tl_type_str in dtypes_to_test:
        passed = test_unpermute_parameterized(
            pt_dtype=pt_type, tl_dtype_str=tl_type_str
        )
        if not passed:
            overall_passed = False

    print(f"\n{'=' * 65}")
    if overall_passed:
        print("Test passed!")
    else:
        print("Test failed! The precision is not correct!")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    test_unpermute()
