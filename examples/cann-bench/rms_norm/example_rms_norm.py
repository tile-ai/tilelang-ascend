"""RMSNorm example — multi-case optimized version with cann-bench 20 cases.

Two-pass RMS normalization with dual-kernel strategy (Hybrid simple + Expert pipeline)
for arbitrary shape, dtype, and epsilon. Targets cann-bench multi-case evaluation.

Algorithm:
    y = x / sqrt(mean(x^2) + eps) * gamma

Reference: torch.nn.functional.rms_norm (PyTorch >= 2.4)

Key optimizations:
- Dual-kernel: n_num=1 uses Hybrid (AUTO_SYNC=True), n_num>1 uses Expert
  (AUTO_SYNC=False + double buffer + three-way flag pipeline)
- Adaptive tiling via UB budget formula
- Newton iteration for rsqrt precision (~1e-3 -> ~1e-6)
- output_ub separation keeps a_ub MTE2-write-only (enables double buffer)
"""

import argparse
import sys

import torch

import tilelang
from tilelang import language as T


# ========== JIT Configuration ==========
EXPERT_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}

HYBRID_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

CAST_NONE = "CAST_NONE"
CAST_RINT = "CAST_RINT"

_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

# cann-bench precision thresholds (MERE = mean relative error, MARE = max relative error)
_CANNBENCH_THRESHOLDS = {
    "float16": 2**-10,
    "bfloat16": 2**-7,
    "float32": 2**-13,
}


# ========== Kernel: Simple (n_num=1, Hybrid mode) ==========
@tilelang.jit(out_idx=[2], pass_configs=HYBRID_CONFIGS)
def _rms_norm_simple(S, D, block_M, block_N, eps=1e-6, dtype="float16"):
    """Hybrid mode RMS Norm for n_num=1 (single tile covers D)."""
    VEC_NUM = 2
    ROWS = block_M // VEC_NUM
    m_num = T.ceildiv(S, block_M)
    need_cast = dtype not in ("float", "float32")
    acc_dtype = "float32" if need_cast else dtype

    @T.prim_func
    def tilelang_rms_norm(
        A: T.Tensor((S, D), dtype),  # type: ignore
        G: T.Tensor((D,), dtype),  # type: ignore
        B: T.Tensor((S, D), dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_start = cid * block_M + vid * ROWS

            a_ub = T.alloc_ub([ROWS, block_N], dtype)
            a_cal = T.alloc_ub([ROWS, block_N], acc_dtype)
            output_ub = T.alloc_ub([ROWS, block_N], dtype)
            sum_sq_acc = T.alloc_ub([ROWS, block_N], acc_dtype)
            sum_sq_row = T.alloc_ub([ROWS, 1], acc_dtype)
            inv_rms_ub = T.alloc_ub([ROWS, 1], acc_dtype)
            inv_rms_tile = T.alloc_ub([ROWS, block_N], acc_dtype)
            gamma_ub = T.alloc_ub([block_N], dtype)
            gamma_cal = T.alloc_ub([block_N if need_cast else 1], acc_dtype)
            gamma_tile = T.alloc_ub([ROWS, block_N], acc_dtype)
            newton_ub = T.alloc_ub([ROWS, 1], acc_dtype)

            # Pass 1: accumulate x^2
            T.tile.fill(sum_sq_acc, 0.0)
            T.copy(A[row_start : row_start + ROWS, 0:block_N], a_ub)
            if need_cast:
                T.tile.cast(a_cal, a_ub, CAST_NONE, ROWS * block_N)
                T.tile.mul(a_cal, a_cal, a_cal)
                T.tile.add(sum_sq_acc, sum_sq_acc, a_cal)
            else:
                T.tile.mul(a_cal, a_ub, a_ub)
                T.tile.add(sum_sq_acc, sum_sq_acc, a_cal)

            # reduce + inv_rms (Newton iteration for precision)
            T.reduce_sum(sum_sq_acc, sum_sq_row, dim=-1)
            inv_n = T.cast(1.0 / D, acc_dtype)
            eps_val = T.cast(eps, acc_dtype)
            T.tile.mul(sum_sq_row, sum_sq_row, inv_n)
            T.tile.add(sum_sq_row, sum_sq_row, eps_val)
            T.tile.rsqrt(inv_rms_ub, sum_sq_row)
            T.tile.mul(newton_ub, inv_rms_ub, inv_rms_ub)
            T.tile.mul(newton_ub, newton_ub, sum_sq_row)
            half_neg = T.cast(-0.5, acc_dtype)
            T.tile.mul(newton_ub, newton_ub, half_neg)
            three_half = T.cast(1.5, acc_dtype)
            T.tile.add(newton_ub, newton_ub, three_half)
            T.tile.mul(inv_rms_ub, inv_rms_ub, newton_ub)
            T.tile.broadcast(inv_rms_tile, inv_rms_ub)

            # Pass 2: normalize + gamma + write back (n_num=1: reuse a_ub)
            if need_cast:
                T.tile.cast(a_cal, a_ub, CAST_NONE, ROWS * block_N)
                T.tile.mul(a_cal, a_cal, inv_rms_tile)
            else:
                T.tile.mul(a_cal, a_ub, inv_rms_tile)

            T.copy(G[0:block_N], gamma_ub)
            if need_cast:
                T.tile.cast(gamma_cal, gamma_ub, CAST_NONE, block_N)
                T.tile.broadcast(gamma_tile, gamma_cal)
            else:
                T.tile.broadcast(gamma_tile, gamma_ub)
            T.tile.mul(a_cal, a_cal, gamma_tile)

            if need_cast:
                T.tile.cast(output_ub, a_cal, CAST_RINT, ROWS * block_N)
                T.copy(output_ub, B[row_start : row_start + ROWS, 0:block_N])
            else:
                T.copy(a_cal, B[row_start : row_start + ROWS, 0:block_N])

    return tilelang_rms_norm


# ========== Kernel: Pipeline (n_num>1, Expert mode with double buffer) ==========
@tilelang.jit(out_idx=[2], pass_configs=EXPERT_CONFIGS)
def _rms_norm_pipeline(S, D, block_M, block_N, eps=1e-6, dtype="float16"):
    """Expert mode RMS Norm for n_num>1 with double buffer pipeline.

    Pass 2 uses three-way flag pipeline (mte3->mte2, mte2->v, v->mte3).
    Pass 1 uses barrier_all (no MTE3 in Pass 1, v->mte2 flag unavailable).
    """
    VEC_NUM = 2
    ROWS = block_M // VEC_NUM
    m_num = T.ceildiv(S, block_M)
    n_num = T.ceildiv(D, block_N)
    need_cast = dtype not in ("float", "float32")
    acc_dtype = "float32" if need_cast else dtype

    @T.prim_func
    def tilelang_rms_norm(
        A: T.Tensor((S, D), dtype),  # type: ignore
        G: T.Tensor((D,), dtype),  # type: ignore
        B: T.Tensor((S, D), dtype),  # type: ignore
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_start = cid * block_M + vid * ROWS

            a_ub_db = T.alloc_ub([2, ROWS, block_N], dtype)
            gamma_ub_db = T.alloc_ub([2, block_N], dtype)
            a_cal = T.alloc_ub([ROWS, block_N], acc_dtype)
            sum_sq_acc = T.alloc_ub([ROWS, block_N], acc_dtype)
            sum_sq_row = T.alloc_ub([ROWS, 1], acc_dtype)
            inv_rms_ub = T.alloc_ub([ROWS, 1], acc_dtype)
            inv_rms_tile = T.alloc_ub([ROWS, block_N], acc_dtype)
            gamma_cal = T.alloc_ub([block_N if need_cast else 1], acc_dtype)
            gamma_tile = T.alloc_ub([ROWS, block_N], acc_dtype)
            newton_ub = T.alloc_ub([ROWS, 1], acc_dtype)

            with T.Scope("V"):
                # Pass 1: accumulate x^2 (single buffer, barrier_all per iteration)
                T.tile.fill(sum_sq_acc, 0.0)
                for by in T.serial(n_num):
                    col_off = by * block_N
                    T.copy(
                        A[row_start : row_start + ROWS, col_off : col_off + block_N],
                        a_ub_db[0, :, :],
                    )
                    T.barrier_all()
                    if need_cast:
                        T.tile.cast(a_cal, a_ub_db[0, :, :], CAST_NONE, ROWS * block_N)
                        T.tile.mul(a_cal, a_cal, a_cal)
                        T.tile.add(sum_sq_acc, sum_sq_acc, a_cal)
                    else:
                        T.tile.mul(a_cal, a_ub_db[0, :, :], a_ub_db[0, :, :])
                        T.tile.add(sum_sq_acc, sum_sq_acc, a_cal)

                # reduce + inv_rms (Newton iteration)
                T.reduce_sum(sum_sq_acc, sum_sq_row, dim=-1)
                inv_n = T.cast(1.0 / D, acc_dtype)
                eps_val = T.cast(eps, acc_dtype)
                T.tile.mul(sum_sq_row, sum_sq_row, inv_n)
                T.tile.add(sum_sq_row, sum_sq_row, eps_val)
                T.tile.rsqrt(inv_rms_ub, sum_sq_row)
                T.tile.mul(newton_ub, inv_rms_ub, inv_rms_ub)
                T.tile.mul(newton_ub, newton_ub, sum_sq_row)
                half_neg = T.cast(-0.5, acc_dtype)
                T.tile.mul(newton_ub, newton_ub, half_neg)
                three_half = T.cast(1.5, acc_dtype)
                T.tile.add(newton_ub, newton_ub, three_half)
                T.tile.mul(inv_rms_ub, inv_rms_ub, newton_ub)
                T.tile.broadcast(inv_rms_tile, inv_rms_ub)

                T.barrier_all()

                # Pass 2: double buffer pipeline (three-way flag)
                T.set_flag("mte3", "mte2", 0)
                T.set_flag("mte3", "mte2", 1)

                # Prefetch tile 0
                T.wait_flag("mte3", "mte2", 0)
                T.copy(A[row_start : row_start + ROWS, 0:block_N], a_ub_db[0, :, :])
                T.copy(G[0:block_N], gamma_ub_db[0, :])
                T.set_flag("mte2", "v", 0)

                # Main body
                for tile in T.serial(0, n_num - 1):
                    cur = tile % 2
                    nxt = (tile + 1) % 2
                    col_cur = tile * block_N
                    col_nxt = (tile + 1) * block_N

                    T.wait_flag("mte3", "mte2", nxt)
                    T.copy(
                        A[row_start : row_start + ROWS, col_nxt : col_nxt + block_N],
                        a_ub_db[nxt, :, :],
                    )
                    T.copy(G[col_nxt : col_nxt + block_N], gamma_ub_db[nxt, :])
                    T.set_flag("mte2", "v", nxt)

                    T.wait_flag("mte2", "v", cur)
                    if need_cast:
                        T.tile.cast(a_cal, a_ub_db[cur, :, :], CAST_NONE, ROWS * block_N)
                        T.tile.mul(a_cal, a_cal, inv_rms_tile)
                        T.tile.cast(gamma_cal, gamma_ub_db[cur, :], CAST_NONE, block_N)
                        T.tile.broadcast(gamma_tile, gamma_cal)
                        T.tile.mul(a_cal, a_cal, gamma_tile)
                        T.tile.cast(a_ub_db[cur, :, :], a_cal, CAST_RINT, ROWS * block_N)
                    else:
                        T.tile.mul(a_ub_db[cur, :, :], a_ub_db[cur, :, :], inv_rms_tile)
                        T.tile.broadcast(gamma_tile, gamma_ub_db[cur, :])
                        T.tile.mul(a_ub_db[cur, :, :], a_ub_db[cur, :, :], gamma_tile)
                    T.set_flag("v", "mte3", cur)

                    T.wait_flag("v", "mte3", cur)
                    T.copy(
                        a_ub_db[cur, :, :],
                        B[row_start : row_start + ROWS, col_cur : col_cur + block_N],
                    )
                    T.set_flag("mte3", "mte2", cur)

                # Epilogue
                last_tile = n_num - 1
                last_stage = last_tile % 2
                col_last = last_tile * block_N

                T.wait_flag("mte2", "v", last_stage)
                if need_cast:
                    T.tile.cast(a_cal, a_ub_db[last_stage, :, :], CAST_NONE, ROWS * block_N)
                    T.tile.mul(a_cal, a_cal, inv_rms_tile)
                    T.tile.cast(gamma_cal, gamma_ub_db[last_stage, :], CAST_NONE, block_N)
                    T.tile.broadcast(gamma_tile, gamma_cal)
                    T.tile.mul(a_cal, a_cal, gamma_tile)
                    T.tile.cast(a_ub_db[last_stage, :, :], a_cal, CAST_RINT, ROWS * block_N)
                else:
                    T.tile.mul(a_ub_db[last_stage, :, :], a_ub_db[last_stage, :, :], inv_rms_tile)
                    T.tile.broadcast(gamma_tile, gamma_ub_db[last_stage, :])
                    T.tile.mul(a_ub_db[last_stage, :, :], a_ub_db[last_stage, :, :], gamma_tile)
                T.set_flag("v", "mte3", last_stage)

                T.wait_flag("v", "mte3", last_stage)
                T.copy(
                    a_ub_db[last_stage, :, :],
                    B[row_start : row_start + ROWS, col_last : col_last + block_N],
                )
                T.set_flag("mte3", "mte2", last_stage)

                T.wait_flag("mte3", "mte2", 0)
                T.wait_flag("mte3", "mte2", 1)

    return tilelang_rms_norm


# ========== Golden Reference ==========
def golden_rms_norm(x, gamma, epsilon=1e-6):
    """Golden reference using torch.nn.functional.rms_norm (CPU fp32)."""
    return torch.nn.functional.rms_norm(x, normalized_shape=gamma.shape, weight=gamma, eps=epsilon)


# ========== Tiling Selection ==========
_BN_CANDIDATES = [2048, 1024, 768, 512, 384, 256, 192, 128, 64, 32, 16]
_BN_POW2 = [1024, 512, 256, 128, 64, 32, 16]


def _find_best_block_n(D, max_bn):
    for bn in _BN_CANDIDATES:
        if bn <= max_bn and bn <= D and D % bn == 0:
            return bn
    for bn in _BN_POW2:
        if bn <= max_bn:
            return bn
    return 16


def _select_tiling(S, D, dtype_str):
    """Select block_M, block_N adaptively based on S, D, dtype."""
    VEC_NUM = 2
    CORE_NUM = 24
    UB_BUDGET = 170 * 1024
    is_low_prec = dtype_str in ("float16", "bfloat16")
    per_unit = 18 if is_low_prec else 16
    D_aligned = max(((D + 15) // 16) * 16, 16)

    best_score = float("inf")
    best_bm, best_bn = 64, 128

    for block_M in (1024, 512, 256, 128, 64, 32, 16):
        if block_M % VEC_NUM != 0:
            continue
        rows = block_M // VEC_NUM
        max_bn = UB_BUDGET // (rows * per_unit)
        max_bn = (max_bn // 16) * 16
        max_bn = max(max_bn, 16)

        if D_aligned <= max_bn:
            block_N = D_aligned
        else:
            block_N = _find_best_block_n(D, max_bn)

        n_num = (D + block_N - 1) // block_N
        m_num = (S + block_M - 1) // block_M

        if m_num < CORE_NUM:
            m_penalty = (CORE_NUM - m_num) * 1000
        else:
            m_penalty = max(0, m_num - 2 * CORE_NUM) * 1
        score = n_num * 100000 + m_penalty

        if score < best_score:
            best_score = score
            best_bm, best_bn = block_M, block_N

    n_num_check = (D + best_bn - 1) // best_bn
    if n_num_check > 1:
        per_unit_pipeline = 20
        rows = best_bm // VEC_NUM
        max_bn = UB_BUDGET // (rows * per_unit_pipeline)
        max_bn = (max_bn // 16) * 16
        max_bn = max(max_bn, 16)
        if D_aligned > max_bn:
            best_bn = _find_best_block_n(D, max_bn)

    return best_bm, best_bn


# ========== Host Wrapper ==========
def rms_norm_host(x, gamma, eps=1e-6):
    """Host-side wrapper: flatten (..., D) -> (S, D), call kernel, reshape back."""
    original_shape = x.shape
    D = original_shape[-1]
    S = 1
    for d in original_shape[:-1]:
        S *= d
    x_2d = x.reshape(S, D)
    dtype_str = str(x.dtype).replace("torch.", "")
    block_M, block_N = _select_tiling(S, D, dtype_str)
    n_num = (D + block_N - 1) // block_N
    if n_num > 1:
        func = _rms_norm_pipeline(S, D, block_M=block_M, block_N=block_N, eps=eps, dtype=dtype_str)
    else:
        func = _rms_norm_simple(S, D, block_M=block_M, block_N=block_N, eps=eps, dtype=dtype_str)
    y_2d = func(x_2d, gamma)
    return y_2d.reshape(original_shape)


# ========== Precision Standards ==========
def get_precision(dtype):
    """Normalization-class precision standards (L0 threshold)."""
    precision_map = {
        "float16": (1e-3, 1e-3),
        "float32": (1e-4, 1e-4),
        "bfloat16": (1e-2, 5e-3),
    }
    return precision_map[dtype]


# ========== cann-bench Precision Checker ==========
def _mere_mare(actual, golden):
    """Compute MERE (mean relative error) and MARE (max relative error).

    Formula (cann-bench standard):
        relative_error = abs(actual - golden) / (abs(golden) + 1e-7)
    """
    diff = (actual.float() - golden.float()).abs()
    denom = golden.float().abs() + 1e-7
    rel_err = diff / denom
    # Handle NaN/Inf: positions where both are NaN/Inf are considered matching
    both_nan = torch.isnan(actual.float()) & torch.isnan(golden.float())
    both_inf = torch.isinf(actual.float()) & torch.isinf(golden.float())
    rel_err = torch.where(both_nan | both_inf, torch.zeros_like(rel_err), rel_err)
    mere = rel_err.mean().item()
    mare = rel_err.max().item()
    return mere, mare


def check_precision(actual, golden, dtype_str, label=""):
    """Check precision against cann-bench thresholds."""
    mere, mare = _mere_mare(actual, golden)
    threshold = _CANNBENCH_THRESHOLDS[dtype_str]
    mare_threshold = 10 * threshold
    passed = mere < threshold and mare < mare_threshold
    status = "PASS" if passed else "FAIL"
    tag = f"[{'PRECISION_PASS' if passed else 'PRECISION_FAIL'}]"
    print(f"{tag} {label} dtype={dtype_str} MERE={mere:.6e} MARE={mare:.6e} threshold={threshold:.6e} -> {status}")
    return passed


# ========== Input Generation ==========
def _gen_input(shape, dtype, value_range, seed_offset=0):
    """Generate input tensor with specified value range."""
    torch.manual_seed(42 + seed_offset)
    torch_dtype = _DTYPE_MAP[dtype] if isinstance(dtype, str) else dtype
    dtype_str = dtype if isinstance(dtype, str) else str(dtype).replace("torch.", "")

    if value_range == "inf":
        x = torch.randn(shape, dtype=torch.float32).uniform_(-1, 1)
        x[0, 0] = float("inf")
        x[1, 0] = float("-inf")
    elif value_range == "nan":
        x = torch.randn(shape, dtype=torch.float32).uniform_(-1, 1)
        x[0, 0] = float("nan")
    elif value_range == "zero":
        x = torch.zeros(shape, dtype=torch.float32)
    else:
        lo, hi = value_range
        x = torch.empty(shape, dtype=torch.float32).uniform_(lo, hi)

    return x.to(torch_dtype).npu(), dtype_str


def _gen_gamma(D, dtype_str):
    """Generate gamma (ones, as per cann-bench standard golden)."""
    torch_dtype = _DTYPE_MAP[dtype_str]
    return torch.ones(D, dtype=torch_dtype).npu()


# ========== Test Runner ==========
def _run_case(name, shape, dtype, eps, value_range, level="cann-bench"):
    """Run a single test case: generate input -> kernel -> golden -> check."""
    dtype_str = dtype if isinstance(dtype, str) else str(dtype).replace("torch.", "")
    D = shape[-1]

    x, dtype_str = _gen_input(shape, dtype_str, value_range, seed_offset=hash(name) % 1000)
    gamma = _gen_gamma(D, dtype_str)

    # Kernel
    y = rms_norm_host(x, gamma, eps)

    # Golden (CPU fp32, cast back to original dtype)
    ref = golden_rms_norm(x.cpu().float(), gamma.cpu().float(), eps).to(_DTYPE_MAP[dtype_str])

    # Precision check
    if level in ("l0", "l1"):
        # Use torch.testing.assert_close for L0/L1
        atol, rtol = get_precision(dtype_str)
        max_diff = (y.cpu().float() - ref.float()).abs().max().item()
        try:
            torch.testing.assert_close(y.cpu(), ref, atol=atol, rtol=rtol)
            print(f"[PRECISION_PASS] {level} {name} shape={shape} dtype={dtype_str} max_diff={max_diff:.6e}")
            return True
        except Exception as e:
            print(f"[PRECISION_FAIL] {level} {name} shape={shape} dtype={dtype_str} max_diff={max_diff:.6e}: {e}")
            return False
    else:
        # Use cann-bench MERE/MARE for cann-bench cases
        return check_precision(y.cpu(), ref, dtype_str, label=f"{level} {name} shape={shape}")


def _run_boundary(level, name, fn):
    """L2/Boundary single case: PASS -> [BOUNDARY_PASS], FAIL -> [BOUNDARY_WARN]."""
    try:
        fn()
        print(f"[BOUNDARY_PASS] {level} {name}")
    except Exception as e:
        print(f"[BOUNDARY_WARN] {level} {name}: {e}")


# ========== L0 Tests ==========
def test_rms_norm_l0():
    """L0 threshold tests: 4 cases covering 3 dtypes + aligned/unaligned shapes."""
    configs = [
        ("l0-1", [32, 128, 768], "float16", 1e-6, (-1, 1)),
        ("l0-2", [32, 128, 1024], "float32", 1e-6, (-2, 2)),
        ("l0-3", [32, 128, 2048], "bfloat16", 1e-6, (-3, 3)),
        ("l0-4", [63, 67, 1023], "float16", 1e-6, (-1, 1)),
    ]
    ok = True
    for name, shape, dtype, eps, vrange in configs:
        ok &= _run_case(name, shape, dtype, eps, vrange, level="l0")
    return ok


# ========== L1 Functional Tests ==========
def test_rms_norm_l1():
    """L1 functional tests: dtype x shape x gamma combinations."""
    configs = [
        ("l1-1", [128, 256], "float16", 1e-6, (-1, 1)),
        ("l1-2", [128, 256], "float32", 1e-6, (-2, 2)),
        ("l1-3", [128, 256], "bfloat16", 1e-6, (-3, 3)),
        ("l1-4", [256, 512], "float16", 1e-6, (-1, 1)),
        ("l1-5", [100, 100], "float16", 1e-6, (-1, 1)),
        ("l1-6", [65, 129], "float16", 1e-6, (-1, 1)),
        ("l1-7", [4, 32, 128], "float16", 1e-6, (-1, 1)),
        ("l1-8", [2, 4, 8, 256], "float32", 1e-6, (-2, 2)),
    ]
    ok = True
    for name, shape, dtype, eps, vrange in configs:
        ok &= _run_case(name, shape, dtype, eps, vrange, level="l1")
    return ok


# ========== L2 Exception Tests ==========
def test_rms_norm_l2():
    """L2 exception tests: unsupported dtype, gamma mismatch, large S, D=1."""

    def test_unsupported_dtype():
        x = torch.ones(64, 128, dtype=torch.int32).npu()
        gamma = torch.ones(128, dtype=torch.int32).npu()
        rms_norm_host(x, gamma, 1e-6)

    def test_gamma_mismatch():
        x = torch.ones(64, 128, dtype=torch.float16).npu()
        gamma = torch.ones(64, dtype=torch.float16).npu()
        rms_norm_host(x, gamma, 1e-6)

    def test_large_s():
        x = torch.randn(100003, 128, dtype=torch.float16).npu()
        gamma = torch.ones(128, dtype=torch.float16).npu()
        y = rms_norm_host(x, gamma, 1e-6)
        ref = golden_rms_norm(x.cpu().float(), gamma.cpu().float(), 1e-6).to(torch.float16)
        torch.testing.assert_close(y.cpu(), ref, atol=1e-3, rtol=1e-3)

    def test_d_equals_1():
        x = torch.randn(64, 1, dtype=torch.float16).npu()
        gamma = torch.ones(1, dtype=torch.float16).npu()
        y = rms_norm_host(x, gamma, 1e-6)
        ref = golden_rms_norm(x.cpu().float(), gamma.cpu().float(), 1e-6).to(torch.float16)
        torch.testing.assert_close(y.cpu(), ref, atol=1e-3, rtol=1e-3)

    _run_boundary("l2", "unsupported_dtype_int32", test_unsupported_dtype)
    _run_boundary("l2", "gamma_dim_mismatch", test_gamma_mismatch)
    _run_boundary("l2", "large_s_100003", test_large_s)
    _run_boundary("l2", "d_equals_1", test_d_equals_1)


# ========== Boundary Tests ==========
def test_rms_norm_boundary():
    """Boundary tests: INF, NAN, all-zeros, extreme values."""

    def test_inf_input():
        x = torch.randn(64, 128, dtype=torch.float16).npu()
        x[0, 0] = float("inf")
        gamma = torch.ones(128, dtype=torch.float16).npu()
        y = rms_norm_host(x, gamma, 1e-6)
        assert torch.isinf(y.cpu()).any() or torch.isnan(y.cpu()).any()

    def test_nan_input():
        x = torch.randn(64, 128, dtype=torch.float16).npu()
        x[0, 0] = float("nan")
        gamma = torch.ones(128, dtype=torch.float16).npu()
        y = rms_norm_host(x, gamma, 1e-6)
        assert torch.isnan(y.cpu()).any()

    def test_all_zeros():
        x = torch.zeros(64, 128, dtype=torch.float16).npu()
        gamma = torch.ones(128, dtype=torch.float16).npu()
        y = rms_norm_host(x, gamma, 1e-6)
        ref = golden_rms_norm(x.cpu().float(), gamma.cpu().float(), 1e-6).to(torch.float16)
        torch.testing.assert_close(y.cpu(), ref, atol=1e-3, rtol=1e-3)

    def test_extreme_values():
        x = torch.empty(64, 128, dtype=torch.float32).uniform_(-65504, 65504).to(torch.float16).npu()
        gamma = torch.ones(128, dtype=torch.float16).npu()
        y = rms_norm_host(x, gamma, 1e-6)
        ref = golden_rms_norm(x.cpu().float(), gamma.cpu().float(), 1e-6).to(torch.float16)
        torch.testing.assert_close(y.cpu(), ref, atol=1e-2, rtol=1e-2)

    _run_boundary("boundary", "inf_input", test_inf_input)
    _run_boundary("boundary", "nan_input", test_nan_input)
    _run_boundary("boundary", "all_zeros", test_all_zeros)
    _run_boundary("boundary", "extreme_values_fp16max", test_extreme_values)


# ========== cann-bench 20 Cases ==========
def test_rms_norm_cann_bench():
    """cann-bench level2/rms_norm official 20 cases.

    Source: cann-bench/tasks/level2/rms_norm/cases.yaml
    Precision: MERE < threshold, MARE < 10*threshold (cann-bench standard)
    """
    configs = [
        # (name, shape, dtype, eps, value_range)
        ("cann-bench-1", [32, 128, 768], "float16", 1e-6, (-1, 1)),
        ("cann-bench-2", [32, 128, 1024], "float32", 1e-6, (-2, 2)),
        ("cann-bench-3", [32, 128, 2048], "bfloat16", 1e-6, (-3, 3)),
        ("cann-bench-4", [16, 256, 4096], "float16", 1e-6, (-10, 10)),
        ("cann-bench-5", [8, 512, 8192], "float32", 1e-6, (-100, 100)),
        ("cann-bench-6", [4, 1023, 4097], "bfloat16", 1e-5, (-5, 5)),
        ("cann-bench-7", [63, 67, 1023], "float16", 1e-8, (-0.1, 0.1)),
        ("cann-bench-8", [16, 511, 2049], "float32", 1e-4, (-1, 1)),
        ("cann-bench-9", [8, 1021, 4099], "bfloat16", 1e-12, (-0.5, 0.5)),
        ("cann-bench-10", [33, 127, 769], "float16", 1e-6, (-1, 2)),
        ("cann-bench-11", [31, 129, 2049], "float32", 1e-6, (-50, 100)),
        ("cann-bench-12", [17, 255, 4097], "bfloat16", 1e-6, (-3, 6)),
        ("cann-bench-13", [7, 1009, 1021], "float16", 1e-7, (-1, 1)),
        ("cann-bench-14", [11, 367, 373], "float32", 1e-5, (-10, 10)),
        ("cann-bench-15", [1000003, 2], "bfloat16", 1e-6, "inf"),
        ("cann-bench-16", [11, 13, 17, 67], "float16", 1e-8, "nan"),
        ("cann-bench-17", [3, 7, 11, 4096], "float32", 1e-4, "zero"),
        ("cann-bench-18", [2, 511, 8192], "bfloat16", 1e-6, (-0.2, 0.2)),
        ("cann-bench-19", [4, 255, 4096], "float16", 1e-3, (-65504, 65504)),
        ("cann-bench-20", [2, 3, 17, 1024, 128], "float32", 1e-6, (-20, 40)),
    ]
    ok = True
    for name, shape, dtype, eps, vrange in configs:
        ok &= _run_case(name, shape, dtype, eps, vrange, level="cann-bench")
    return ok


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description="RMSNorm example with cann-bench 20 cases")
    parser.add_argument(
        "--level",
        default="l0",
        choices=["l0", "l1", "l2", "boundary", "cann-bench", "all"],
        help="Test level to run (default: l0)",
    )
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(0)

    blocking_ok = True
    if args.level in ("l0", "all"):
        blocking_ok &= test_rms_norm_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_rms_norm_l1()
    if args.level in ("l2", "all"):
        test_rms_norm_l2()
    if args.level in ("boundary", "all"):
        test_rms_norm_boundary()
    if args.level in ("cann-bench", "all"):
        blocking_ok &= test_rms_norm_cann_bench()

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
