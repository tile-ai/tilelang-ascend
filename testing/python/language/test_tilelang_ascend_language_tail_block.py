import pytest
import tilelang
import tilelang.language as T
import torch

"""
Tail-block (尾块) guard suite.

Feature under test
------------------
"尾块处理": the framework automatically handles the partial last tile when a
tensor dimension is NOT a multiple of the tile/block size. The frontend simply
allocates full-size ``block_M x block_N`` tiles, drives the grid/loops with
``T.ceildiv``, and indexes with ``bx * block_M`` -- it never special-cases the
edge. CUBE / VECTOR / CV-fusion operators are all covered, and the frontend is
"无需感知" (does not need to be aware of) the tail.

Mechanism (src/op/ascend.cc :: compute_valid_extent)
----------------------------------------------------
Every GM<->on-chip ``T.copy`` is clamped at lowering time::

    valid = Select(shape - off >= extent, extent,          # full block
                   Select(shape - off > 0,  shape - off,   # tail block
                          0))                               # fully OOB

where ``shape`` is the GM tensor's real dim and ``off`` is the tile offset
(e.g. ``bx * block_M``). The clamp is emitted for these copy directions:

    CUBE   : gm2l1 (load A/B)   + l0c2gm (store C)   -> M / N / K tails
    VECTOR : gm2ub (load)       + ub2gm  (store)     -> M / N tails
    CV     : C-scope uses the cube path, V-scope the vector path

pad_value vs real_shape (the subtle VECTOR case)
------------------------------------------------
On ``gm2ub`` loads the UB area outside ``validRow x validCol`` *can* be filled
with ``pad_value`` (``T.copy(..., pad_value=...)``; default 0 -- ascend.cc:58 /
copy.py:277), but this is backend-dependent and NOT reliable as a correctness
mechanism: the PTO backend emits ``PadValue::Null`` for sliced loads
(codegen_ascend_pto.cc), so the tail region stays *garbage*. Impact:

    * element-wise (add/abs/...) : the tail is computed but NOT stored back
      (ub2gm re-clamps the store) -> pad_value is irrelevant, default 0 is fine.
    * CUBE gemm K-tail           : the L1 tail is implicitly 0, and 0 * B = 0,
      so the matmul stays correct.
    * reduce                     : native fallback reductions must be told their
      logical valid extent via ``real_shape=[rows, cols]`` (reduce_ascend.py).
      With ``TL_ASCEND_TAIL_MASK``, the guarded axis-0 float32 contract instead
      carries the runtime valid rectangle into backend-native reduction code.
      Relying on a -inf pad produces inf/nan on NPU (verified) because PTO does
      not pad sliced loads. ``test_reduce_max_tail`` guards the real_shape
      fallback; Group 2d guards the valid-region rewrite.

NOTE: these cases execute on real NPU hardware (``.npu()``); they cannot run in a
CPU-only environment. Risk levels are annotated per group so unsupported
(target, dtype) combos can be dropped after an NPU run, per the established
workflow (cf. #683 / #700 tail-block iterations).
"""

# CUBE: mirrors examples/gemm/example_gemm_tail_block_developer.py
CUBE_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# VECTOR: the elementwise suite keeps its established configuration.
VEC_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# Tail reduction is a pure-vector kernel. Reuse the established vector pass
# configuration so PTO mixed-target compilation scopes vector intrinsics to
# the vector branch.
TAIL_REDUCE_PASS_CONFIGS = {
    **VEC_PASS_CONFIGS,
    tilelang.PassConfigKey.TL_ASCEND_TAIL_MASK: True,
}


def _vec_configs(tail_mask):
    """Vector pass configs, optionally enabling the opt-in tail-block scheme.

    Every vector tail case is run twice (see the ``tail_mask`` parametrize):
      * ``tail_mask=False`` -> the existing path (manual pad_value / ub2gm
        re-clamp / real_shape), i.e. the behaviour these tests already guarded.
      * ``tail_mask=True``  -> additionally exercises AscendTailMaskPropagation,
        so the same kernel + reference also covers the new valid-region rewrite
        (unary/binary/scalar). The result must match either way.
    """
    return {**VEC_PASS_CONFIGS, tilelang.PassConfigKey.TL_ASCEND_TAIL_MASK: tail_mask}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Clear tilelang cache before the session."""
    tilelang.cache.clear_cache()
    yield


def _torch_dtype(dtype):
    return {"float": torch.float32, "float16": torch.float16}[dtype]


# =============================================================================
# Group 1 - CUBE (gemm) tail block      [risk: low]
# M / N / K all non-divisible. Guards gm2l1 (load A/B) + l0c2gm (store C) clamp.
# Structure copied verbatim from example_gemm_tail_block_developer.py.
# =============================================================================
def cube_matmul_tail(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),  # type: ignore
        B: T.Tensor((K, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)  # gm2l1: M & K tail
                    T.copy(B[k * K_L1, by * block_N], B_L1)  # gm2l1: K & N tail
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

                T.copy(C_L0, C[bx * block_M, by * block_N])  # l0c2gm: M & N tail

    return main


def run_test_cube_matmul_tail(M, N, K, block_M, block_N, K_L1, target):
    torch.manual_seed(0)
    func = cube_matmul_tail(M, N, K, block_M, block_N, K_L1)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=CUBE_PASS_CONFIGS, target=target)

    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()

    torch.npu.synchronize()
    c = func(a, b)

    ref_c = a @ b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


# (M, N, K, block_M, block_N, K_L1) - every dim deliberately non-divisible.
cube_tail_configs = [
    (32 * 3 + 30, 32 * 2 + 16, 32 * 4 + 31, 32, 32, 32),  # (126, 80, 159)
    (64 * 8 + 45, 64 * 8, 64 * 8 + 27, 64, 64, 64),  # (557, 512, 539) - N exact
    (128 * 4, 128 * 4 + 99, 128 * 4, 128, 128, 128),  # (512, 611, 512) - only N tail
    (1024 + 118, 1024 + 206, 1024 + 55, 128, 256, 64),  # (1142, 1230, 1079)
]


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("M,N,K,block_M,block_N,K_L1", cube_tail_configs)
def test_cube_matmul_tail(M, N, K, block_M, block_N, K_L1, target):
    run_test_cube_matmul_tail(M, N, K, block_M, block_N, K_L1, target=target)


# =============================================================================
# Group 2a - VECTOR element-wise tail   [risk: low]
# M / N non-divisible. Guards gm2ub (load) + ub2gm (store) clamp. Full-block
# layout (no vid split) to isolate the tail mechanism. pad_value irrelevant here
# (the padded UB region is never stored back).
# =============================================================================
def vec_add_tail(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            c_ub = T.alloc_ub((block_M, block_N), dtype)

            T.copy(A[bx * block_M, by * block_N], a_ub)  # gm2ub: M & N tail
            T.copy(B[bx * block_M, by * block_N], b_ub)
            T.tile.add(c_ub, a_ub, b_ub)
            T.copy(c_ub, C[bx * block_M, by * block_N])  # ub2gm: M & N tail

    return main


def run_test_vec_add_tail(M, N, block_M, block_N, dtype, target, tail_mask):
    torch.manual_seed(0)
    func = vec_add_tail(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=_vec_configs(tail_mask), target=target)

    td = _torch_dtype(dtype)
    a = torch.randn(M, N, dtype=td).npu()
    b = torch.randn(M, N, dtype=td).npu()

    torch.npu.synchronize()
    c = func(a, b)

    ref_c = a + b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


# =============================================================================
# Group 2b - VECTOR single-input tail   [risk: low]
# =============================================================================
def vec_abs_tail(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)

            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.tile.abs(b_ub, a_ub)
            T.copy(b_ub, B[bx * block_M, by * block_N])

    return main


def run_test_vec_abs_tail(M, N, block_M, block_N, dtype, target, tail_mask):
    torch.manual_seed(0)
    func = vec_abs_tail(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=_vec_configs(tail_mask), target=target)

    td = _torch_dtype(dtype)
    a = torch.randn(M, N, dtype=td).npu()

    torch.npu.synchronize()
    b = func(a)

    ref_b = torch.abs(a)
    torch.testing.assert_close(b, ref_b, rtol=1e-2, atol=1e-2)


# (M, N, block_M, block_N) - both dims non-divisible.
# Tiles are kept small enough that 3x full-block UB buffers stay within the
# Unified Buffer: this group uses NO VEC_NUM split (one AIV handles the whole
# block_M), so the footprint is block_M*block_N*sizeof(dtype)*3. 64x128 fp32 x3
# = 96KB is comfortably under budget. The earlier 128x128 (192KB) / 128x256
# (384KB) fp32 full-block configs over-allocated UB and segfaulted the AscendC
# compiler in OptimizeForTarget -- keep tiles <= 64x128 here.
vec_tail_configs = [
    (32 * 2 + 13, 32 * 3 + 7, 32, 32),  # (77, 103)  - 32x32  x3 fp32 = 12KB
    (64 * 2 + 2, 64 + 36, 64, 64),  # (130, 100) - 64x64  x3 fp32 = 48KB
    (64 * 3 + 8, 128 + 22, 64, 128),  # (200, 150) - 64x128 x3 fp32 = 96KB
]


@pytest.mark.parametrize("tail_mask", [False, True])
@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("M,N,block_M,block_N", vec_tail_configs)
def test_vec_add_tail(M, N, block_M, block_N, dtype, target, tail_mask):
    run_test_vec_add_tail(M, N, block_M, block_N, dtype, target=target, tail_mask=tail_mask)


@pytest.mark.parametrize("tail_mask", [False, True])
@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("M,N,block_M,block_N", vec_tail_configs)
def test_vec_abs_tail(M, N, block_M, block_N, dtype, target, tail_mask):
    run_test_vec_abs_tail(M, N, block_M, block_N, dtype, target=target, tail_mask=tail_mask)


# =============================================================================
# Group 2c - VECTOR reduce over a sliced/tail UB tile   [risk: medium]
# The tail along the *reduced* dimension is handled by real_shape, NOT pad_value.
# A physically (rows_phys, cols) tile holds only rows_valid (< rows_phys) rows of
# real data; real_shape=[rows_valid, cols] tells the reduce its logical valid
# extent so the [rows_valid, rows_phys) tail rows are never touched. pad_value is
# the wrong tool here -- the PTO backend emits PadValue::Null for sliced gm2ub
# loads (codegen_ascend_pto.cc), leaving the tail region as garbage, so a
# full-tile reduce that relied on a -inf pad produced inf/nan on every backend.
# Mirrors examples/reduce/example_col_reduce_max_slice_buffer.py (known-good on pto).
# =============================================================================
def reduce_max_tail(rows_valid, rows_phys, cols, dtype="float"):
    @T.prim_func
    def main(
        Input: T.Tensor((rows_phys, cols), dtype),  # type: ignore
        Output: T.Tensor((1, cols), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            in_ub = T.alloc_ub((rows_phys, cols), dtype)
            out_ub = T.alloc_ub((1, cols), dtype)

            if vid == 0:
                T.copy(Input, in_ub)
                # Reduce dim=0 over only the first rows_valid rows; the
                # [rows_valid, rows_phys) tail rows are excluded via real_shape.
                T.reduce_max(in_ub, out_ub, dim=0, real_shape=[rows_valid, cols])
                T.copy(out_ub, Output)

    return main


def run_test_reduce_max_tail(rows_valid, rows_phys, cols, dtype, target, tail_mask):
    torch.manual_seed(0)
    func = reduce_max_tail(rows_valid, rows_phys, cols, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=_vec_configs(tail_mask), target=target)

    td = _torch_dtype(dtype)
    a = torch.randn(rows_phys, cols, dtype=td).npu()

    torch.npu.synchronize()
    out = func(a)

    # Only the first rows_valid rows are logically valid.
    ref = torch.max(a[:rows_valid, :], dim=0, keepdim=True).values
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


# (rows_valid, rows_phys, cols): rows_valid < rows_phys is the row tail that
# real_shape must exclude from the dim=0 reduce.
reduce_tail_configs = [
    (3, 5, 8),  # mirrors example_col_reduce_max_slice_buffer.py exactly
    (30, 32, 64),  # 32-row tile, 30 valid (tail 2)
    (100, 128, 96),  # 128-row tile, 100 valid (tail 28)
]


# With tail-mask enabled, this explicit real_shape does not match the physical
# tile and therefore stays on the native real_shape path on both backends.
@pytest.mark.parametrize("tail_mask", [False, True])
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("rows_valid,rows_phys,cols", reduce_tail_configs)
def test_reduce_max_tail(rows_valid, rows_phys, cols, dtype, target, tail_mask):
    run_test_reduce_max_tail(rows_valid, rows_phys, cols, dtype, target=target, tail_mask=tail_mask)


# =============================================================================
# Group 2d - AscendC/PTO axis-0 tail reductions   [risk: high]
# Covers sum/max/min with clear=true. Each tile writes one partial
# reduction, so different blocks never race on the output.
# =============================================================================
def reduce_axis0_tail(M, N, block_M, block_N, kind, dim=0, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    reduce_fn = {
        "sum": T.reduce_sum,
        "max": T.reduce_max,
        "min": T.reduce_min,
    }[kind]

    @T.prim_func
    def main(
        Input: T.Tensor((M, N), dtype),  # type: ignore
        Output: T.Tensor((m_num, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            in_ub = T.alloc_ub((block_M, block_N), dtype)
            out_ub = T.alloc_ub((1, block_N), dtype)

            T.copy(Input[bx * block_M, by * block_N], in_ub)
            reduce_fn(in_ub, out_ub, dim=dim, clear=True)
            T.copy(out_ub, Output[bx, by * block_N])

    return main


reduce_axis0_configs = [
    (34, 128, 32, 32),  # row tail only
    (32, 130, 32, 32),  # column tail only
    (34, 130, 32, 32),  # row and column tails
    (33, 129, 32, 32),  # one valid row and one valid column in the last tile
    (7, 13, 32, 32),  # tensor smaller than one physical tile
    (32, 128, 32, 32),  # exact full tiles
    (65, 130, 64, 32),  # larger physical row with a one-row tail
]

REDUCE_TAIL_TARGETS = ("ascendc", "pto")
REDUCE_TAIL_KINDS = ("sum", "max", "min")
REDUCE_TAIL_AXIS0_DIMS = (0, -2)


@pytest.mark.parametrize("kind", REDUCE_TAIL_KINDS)
@pytest.mark.parametrize("dim", REDUCE_TAIL_AXIS0_DIMS)
@pytest.mark.parametrize("target", REDUCE_TAIL_TARGETS)
@pytest.mark.parametrize("M,N,block_M,block_N", reduce_axis0_configs)
def test_reduce_axis0_tail(M, N, block_M, block_N, target, kind, dim):
    func = reduce_axis0_tail(M, N, block_M, block_N, kind, dim=dim)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=TAIL_REDUCE_PASS_CONFIGS, target=target)

    torch.manual_seed(0)
    a = torch.randn(M, N, dtype=torch.float32).npu()
    out = func(a)

    m_num = (M + block_M - 1) // block_M
    ref = torch.empty((m_num, N), dtype=torch.float32, device=a.device)
    for bx in range(m_num):
        tile = a[bx * block_M : min((bx + 1) * block_M, M), :]
        reduced = getattr(tile, kind)(dim=0)
        ref[bx, :] = reduced if kind == "sum" else reduced.values
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(("kind", "sign"), [("max", -1.0), ("min", 1.0)])
@pytest.mark.parametrize("target", REDUCE_TAIL_TARGETS)
def test_reduce_axis0_tail_does_not_consume_zero_padding(target, kind, sign):
    """Guard max/min against treating the zero-filled physical tail as data."""
    M, N, block_M, block_N = 3, 8, 4, 8
    func = reduce_axis0_tail(M, N, block_M, block_N, kind)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=TAIL_REDUCE_PASS_CONFIGS, target=target)

    base = torch.arange(1, M * N + 1, dtype=torch.float32).reshape(M, N)
    a = (base * sign).npu()
    out = func(a)[0]
    reduced = getattr(a, kind)(dim=0)
    ref = reduced.values
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


# NaN propagation and signed-zero tie-breaking are backend-instruction
# semantics rather than part of the shared valid-region contract. Keep this
# exact-bit regression scoped to the AscendC helper until PTO documents and
# validates an equivalent guarantee on hardware.
@pytest.mark.parametrize("kind", ["sum", "max", "min"])
def test_reduce_axis0_special_values_ascendc(kind):
    M, N, block_M, block_N = 3, 8, 4, 8
    func = reduce_axis0_tail(M, N, block_M, block_N, kind)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=TAIL_REDUCE_PASS_CONFIGS, target="ascendc")

    a = torch.tensor(
        [
            [0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1.0, -1.0, 3.0],
            [-0.0, 0.0, 2.0, -2.0, 4.0, float("nan"), -3.0, 3.0],
            [0.0, -0.0, -5.0, 5.0, -4.0, 2.0, float("nan"), -3.0],
        ],
        dtype=torch.float32,
    ).npu()
    out = func(a)[0]
    reduced = getattr(a, kind)(dim=0)
    ref = reduced if kind == "sum" else reduced.values
    torch.testing.assert_close(out, ref, rtol=0, atol=0, equal_nan=True)

    finite_zero = (ref == 0) & ~torch.isnan(ref)
    torch.testing.assert_close(torch.signbit(out[finite_zero]), torch.signbit(ref[finite_zero]))


# =============================================================================
# Group 3 - CV fusion (matmul + add) tail   [risk: medium]
# Mirrors examples/simple_fusion/matmul_add.py, but the grid uses T.ceildiv with
# non-divisible M/N. C-scope (cube) tails ride gm2l1/l0c2gm; V-scope (vector,
# dual-AIV vid split) tails ride gm2ub/ub2gm. The same clamp formula covers the
# `bx*block_M + vid*block_M//VEC_NUM` per-vid offset. Manual cross-core sync, so
# no auto pass_configs (faithful to the example's plain @jit).
# =============================================================================
def cv_matmul_add_tail(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),  # type: ignore
        B: T.Tensor((K, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
        D: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_K, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            d_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, block_K)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * block_K], A_L1)  # gm2l1: M & K tail
                    T.copy(B[k * block_K, by * block_N], B_L1)  # gm2l1: K & N tail

                    T.barrier_all()
                    if k == 0:
                        T.gemm_v0(A_L1, B_L1, C_L0, init=True)
                    else:
                        T.gemm_v0(A_L1, B_L1, C_L0)
                    T.barrier_all()

                T.copy(C_L0, C[bx * block_M, by * block_N])  # l0c2gm: M & N tail
                T.set_cross_flag("FIX", 0)

            with T.Scope("V"):
                T.wait_cross_flag(0)

                T.copy(C[bx * block_M + vid * block_M // VEC_NUM, by * block_N], c_ub)  # gm2ub tail
                T.copy(D[bx * block_M + vid * block_M // VEC_NUM, by * block_N], d_ub)

                T.barrier_all()
                T.tile.add(c_ub, c_ub, d_ub)
                T.barrier_all()

                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])  # ub2gm tail

    return main


def run_test_cv_matmul_add_tail(M, N, K, block_M, block_N, block_K, target):
    torch.manual_seed(0)
    func = cv_matmul_add_tail(M, N, K, block_M, block_N, block_K)
    # out_idx=[-2] -> C (A@B written by cube, then += D by vector). Faithful to
    # examples/simple_fusion/matmul_add.py: plain compile, manual sync, no auto
    # pass_configs.
    func = tilelang.compile(func, out_idx=[-2], target=target)

    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()
    d = torch.randn(M, N).half().npu()

    torch.npu.synchronize()
    c = func(a, b, d)

    ref_c = a @ b + d
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


# (M, N, K, block_M, block_N, block_K) - M/N/K non-divisible.
cv_tail_configs = [
    (128 + 30, 256 + 16, 64 + 8, 128, 256, 64),  # (158, 272, 72)
    (256 + 33, 256 + 40, 128 + 5, 128, 256, 64),  # (289, 296, 133)
]


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("M,N,K,block_M,block_N,block_K", cv_tail_configs)
def test_cv_matmul_add_tail(M, N, K, block_M, block_N, block_K, target):
    run_test_cv_matmul_add_tail(M, N, K, block_M, block_N, block_K, target=target)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
