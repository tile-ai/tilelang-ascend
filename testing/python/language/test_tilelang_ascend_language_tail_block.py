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
    * reduce                     : the tail WOULD corrupt the result, so the
      reduce must be told its logical valid extent via ``real_shape=[rows, cols]``
      (reduce_ascend.py) and never reads the tail at all. Relying on a -inf
      pad instead produces inf/nan on NPU (verified) because PTO does not pad
      sliced loads. ``test_reduce_max_tail`` guards the ``real_shape`` path.

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

# VECTOR: mirrors the elementwise suite's config (CV combine is harmless for a
# pure-vector kernel and matches the existing passing tests).
VEC_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


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


def run_test_vec_add_tail(M, N, block_M, block_N, dtype, target):
    torch.manual_seed(0)
    func = vec_add_tail(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)

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


def run_test_vec_abs_tail(M, N, block_M, block_N, dtype, target):
    torch.manual_seed(0)
    func = vec_abs_tail(M, N, block_M, block_N, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)

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


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("M,N,block_M,block_N", vec_tail_configs)
def test_vec_add_tail(M, N, block_M, block_N, dtype, target):
    run_test_vec_add_tail(M, N, block_M, block_N, dtype, target=target)


@pytest.mark.parametrize("dtype", ["float", "float16"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("M,N,block_M,block_N", vec_tail_configs)
def test_vec_abs_tail(M, N, block_M, block_N, dtype, target):
    run_test_vec_abs_tail(M, N, block_M, block_N, dtype, target=target)


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


def run_test_reduce_max_tail(rows_valid, rows_phys, cols, dtype, target):
    torch.manual_seed(0)
    func = reduce_max_tail(rows_valid, rows_phys, cols, dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_PASS_CONFIGS, target=target)

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


@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("rows_valid,rows_phys,cols", reduce_tail_configs)
def test_reduce_max_tail(rows_valid, rows_phys, cols, dtype, target):
    run_test_reduce_max_tail(rows_valid, rows_phys, cols, dtype, target=target)


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
