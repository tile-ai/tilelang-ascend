import math
import torch

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
    out = torch.zeros((target_rows, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
    out[: tensor.shape[0]] = tensor
    return out


def pad_last_dim(tensor: torch.Tensor, target_cols: int) -> torch.Tensor:
    if tensor.shape[-1] >= target_cols:
        return tensor
    out = torch.zeros((*tensor.shape[:-1], target_cols), dtype=tensor.dtype, device=tensor.device)
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
    ):
        @T.macro
        def cast_axpy(slot, prob, row_buf, row_tmp, row_f32, acc_buf):
            T.copy(row_buf[slot, :, :], row_tmp)
            T.tile.cast(row_f32, row_tmp, CAST_LOW2HIGH, HALF_H)
            T.tile.axpy(acc_buf, row_f32, prob)

        @T.prim_func
        def moe_token_unpermute(
            perm_tokens_gm: T.Tensor([E, hidden_size], dtype),
            sorted_idx_gm: T.Tensor([1, padded_E], idx_dtype),
            probs_gm: T.Tensor([padded_tokens, topK], dtype),
            out_gm: T.Tensor([num_tokens, hidden_size], dtype),
        ):
            with T.Kernel(actual_cores, is_npu=True) as (cid, vid):
                idx_ub = T.alloc_ub([1, BATCH_T * topK], idx_dtype)
                probs_ub = T.alloc_ub([1, BATCH_T * topK], dtype)
                probs_f32 = T.alloc_ub([1, BATCH_T * topK], acc_dtype)
                row_buf = T.alloc_ub([8, 1, HALF_H], dtype)
                row_tmp = T.alloc_ub([1, HALF_H], dtype)
                row_f32 = T.alloc_ub([1, HALF_H], acc_dtype)
                acc_buf = T.alloc_ub([1, HALF_H], acc_dtype)
                out_buf = T.alloc_ub([1, HALF_H], dtype)

                with T.Scope("V"):
                    for batch_id in T.serial(n_batches):
                        batch_base = cid * tokens_per_core + batch_id * BATCH_T

                        T.copy(sorted_idx_gm[0, batch_base * topK], idx_ub)
                        T.copy(probs_gm[batch_base, 0], probs_ub)
                        T.barrier_all()
                        T.tile.cast(probs_f32, probs_ub, CAST_LOW2HIGH, BATCH_T * topK)

                        for ti in T.serial(BATCH_T):
                            i = batch_base + ti
                            if i < num_tokens:
                                for ht in T.serial(n_htiles):
                                    h_off = ht * TILE_H + vid * HALF_H
                                    tk_off = ti * topK

                                    if topK == 8:
                                        T.tile.fill(acc_buf, 0.0)
                                        for lane in T.serial(8):
                                            src = idx_ub[0, tk_off + lane]
                                            T.copy(perm_tokens_gm[src, h_off], row_buf[lane, :, :])
                                        T.barrier_all()
                                        for lane in T.serial(8):
                                            prob = probs_f32[0, tk_off + lane]
                                            cast_axpy(lane, prob, row_buf, row_tmp, row_f32, acc_buf)
                                    else:
                                        src_row_0 = idx_ub[0, tk_off]
                                        prob_val_0 = probs_f32[0, tk_off]
                                        T.copy(perm_tokens_gm[src_row_0, h_off], row_buf[0, :, :])
                                        T.barrier_all()
                                        T.copy(row_buf[0, :, :], row_tmp)
                                        T.tile.cast(acc_buf, row_tmp, CAST_LOW2HIGH, HALF_H)
                                        T.tile.mul(acc_buf, acc_buf, prob_val_0)

                                        n_quads = (topK - 1) // 4
                                        remainder = (topK - 1) % 4

                                        for j4 in T.serial(n_quads):
                                            j = j4 * 4
                                            for lane in T.serial(4):
                                                src = idx_ub[0, tk_off + j + lane + 1]
                                                T.copy(perm_tokens_gm[src, h_off], row_buf[lane, :, :])
                                            T.barrier_all()
                                            for lane in T.serial(4):
                                                prob = probs_f32[0, tk_off + j + lane + 1]
                                                cast_axpy(lane, prob, row_buf, row_tmp, row_f32, acc_buf)

                                        for r in T.serial(remainder):
                                            off = n_quads * 4 + r + 1
                                            src = idx_ub[0, tk_off + off]
                                            T.copy(perm_tokens_gm[src, h_off], row_buf[r, :, :])
                                        T.barrier_all()
                                        for r in T.serial(remainder):
                                            off = n_quads * 4 + r + 1
                                            prob = probs_f32[0, tk_off + off]
                                            cast_axpy(r, prob, row_buf, row_tmp, row_f32, acc_buf)

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
            probs_gm: T.Tensor([padded_tokens, topK], dtype),
            out_gm: T.Tensor([num_tokens, hidden_size], dtype),
        ):
            with T.Kernel(actual_cores, is_npu=True) as (cid, vid):
                idx_ub = T.alloc_shared([1, BATCH_T * topK], idx_dtype)
                probs_ub = T.alloc_shared([1, BATCH_T * topK], dtype)
                row_buf = T.alloc_shared([3, 1, TILE_H], dtype)
                acc_buf = T.alloc_shared([1, TILE_H], dtype)
                T.alloc_shared([1, TILE_H], dtype)

                for batch_id in T.serial(n_batches):
                    batch_base = cid * tokens_per_core + batch_id * BATCH_T

                    T.copy(sorted_idx_gm[0, batch_base * topK], idx_ub)
                    T.copy(probs_gm[batch_base, 0], probs_ub)

                    for ti in T.serial(BATCH_T):
                        i = batch_base + ti
                        if i < num_tokens:
                            for ht in T.serial(n_htiles):
                                h_off = ht * TILE_H
                                tk_off = ti * topK

                                src_row_0 = idx_ub[0, tk_off]
                                prob_val_0 = probs_ub[0, tk_off]
                                T.copy(perm_tokens_gm[src_row_0, h_off], row_buf[0, :, :])
                                T.copy(row_buf[0, :, :], acc_buf)
                                T.tile.mul(acc_buf, acc_buf, prob_val_0)

                                n_triples = (topK - 1) // 3
                                remainder = (topK - 1) % 3

                                for j3 in T.serial(n_triples):
                                    j = j3 * 3
                                    for lane in T.serial(3):
                                        off = j + lane + 1
                                        src = idx_ub[0, tk_off + off]
                                        T.copy(perm_tokens_gm[src, h_off], row_buf[lane, :, :])
                                    for lane in T.serial(3):
                                        off = j + lane + 1
                                        prob = probs_ub[0, tk_off + off]
                                        T.tile.axpy(acc_buf, row_buf[lane, :, :], prob)

                                for r in T.serial(remainder):
                                    off = n_triples * 3 + r + 1
                                    src = idx_ub[0, tk_off + off]
                                    prob = probs_ub[0, tk_off + off]
                                    T.copy(perm_tokens_gm[src, h_off], row_buf[r, :, :])
                                    T.tile.axpy(acc_buf, row_buf[r, :, :], prob)

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
                row_buf = T.alloc_ub([4, 1, HALF_H], dtype)

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
                                            T.copy(perm_tokens_gm[src, h_off], row_buf[lane, :, :])
                                    T.barrier_all()
                                    for lane in T.serial(4):
                                        row = e_base + ei + lane
                                        if row < E:
                                            T.copy(row_buf[lane, :, :], out_gm[row, h_off])
                                    T.pipe_barrier("mte3")

                            remainder = TILE_E % 4
                            for r in T.serial(remainder):
                                ei = n_quads * 4 + r
                                row = e_base + ei
                                if row < E:
                                    src = idx_ub[0, ei]
                                    for ht in T.serial(n_htiles):
                                        h_off = ht * TILE_H + vid * HALF_H
                                        T.copy(perm_tokens_gm[src, h_off], row_buf[0, :, :])
                                        T.barrier_all()
                                        T.copy(row_buf[0, :, :], out_gm[row, h_off])
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
        large_candidates = [64, 32, 16, 8, 4, 2, 1] if has_probs else [128, 64, 32, 16, 8, 4, 2, 1]
        TILE_T = auto_tile_t(total, NUM_CORES, large_candidates)

    min_tile_h = 64 if is_fp32_dtype(dtype) else 8
    if min_tile_h > TILE_H:
        TILE_H = min(hidden_size, min_tile_h)
    assert hidden_size % TILE_H == 0, f"hidden_size ({hidden_size}) 必须是 TILE_H ({TILE_H}) 的整数倍！"
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
        assert dtype in ("float16", "bfloat16", "float32", "float"), f"dtype must be float16/bfloat16/float32, got {dtype}"

        self.num_tokens = num_tokens
        self.topK = topK
        self.hidden_size = hidden_size
        self.has_probs = has_probs
        self.dtype = dtype
        self.E = num_tokens * topK
        min_compile_h = 64 if is_fp32_dtype(dtype) else 32
        self._compile_hidden_size = max(hidden_size, min_compile_h)
        compile_tile_h = TILE_H if TILE_H is None else max(TILE_H, min_compile_h)

        self._kernel, self._padded_tokens, self._padded_E, self._actual_cores = _compile_gather(
            num_tokens,
            topK,
            self._compile_hidden_size,
            has_probs=has_probs,
            NUM_CORES=NUM_CORES,
            TILE_T=TILE_T,
            TILE_H=compile_tile_h,
            dtype=dtype,
        )

    def __call__(self, permuted_tokens, sorted_indices, probs=None):
        if self._kernel is None:
            raise RuntimeError("tilelang not installed")

        indices_padded_2d = pad_first_dim(sorted_indices, self._padded_E).unsqueeze(0)
        permuted_tokens_in = pad_last_dim(permuted_tokens, self._compile_hidden_size)

        if self.has_probs:
            assert probs is not None, "has_probs=True 但未传入 probs"
            probs_padded = pad_first_dim(probs, self._padded_tokens)
            out = self._kernel(permuted_tokens_in, indices_padded_2d, probs_padded)
            return out[:, : self.hidden_size].contiguous()

        out = self._kernel(permuted_tokens_in, indices_padded_2d)
        return out[:, : self.hidden_size].contiguous()

    def __repr__(self):
        return f"MoeTokenUnpermute(T={self.num_tokens}, K={self.topK}, H={self.hidden_size}, probs={self.has_probs}, cores={self._actual_cores})"
