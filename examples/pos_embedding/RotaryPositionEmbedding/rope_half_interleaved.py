"""RoPE (Half + Interleaved) unified implementation for Ascend NPU.

Forward-only rotary position embedding with NPU-internal mask generation.
Supports Half (GPT-NeoX/LLaMA) and Interleaved (GPT-J) layouts via a
compile-time ``layout`` parameter that selects the code path inside
``@T.prim_func``.

Layouts:
  - interleaved: rotate([x0, x1, x2, x3, ...]) = [x1, x0, x3, x2, ...]
                 sin_mask = [-1, +1, -1, +1, ...]
  - half:        rotate([x1, x2]) = [x2, x1]  (copy-swap)
                 sin_mask = [-1, ..., -1, +1, ..., +1]

Unified formula:  out = x * cos + rotate(x) * (sin * sin_mask)
"""

import argparse
import sys

import tilelang
import tilelang.language as T
import torch

# ========== Pass Configs (Developer mode, auto-sync) ==========
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

NUM_CORES = 48
device = torch.device("npu")

torch_dtype_map = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

tilelang_dtype_map = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}


# ========== Kernel ==========
@tilelang.jit(pass_configs=pass_configs)
def rope_kernel(M, block_M, num_blocks, total_chunks, sc_rows, hidden_size, rope_dim, head_num, layout, dtype="float16"):
    """RoPE forward kernel (in-place on ``x``).

    Parameters are Python-level (JIT compile-time constants). ``layout`` is a
    Python string that selects the code path during ``@T.prim_func`` parsing.
    """
    VEC_NUM = 2
    dim_start = hidden_size - rope_dim
    row_per_vec = block_M // VEC_NUM
    half = rope_dim // 2
    ACC_DTYPE = "float32"
    MASK_DTYPE = "uint32"
    need_cast = dtype != "float32"

    chunks_per_block = (total_chunks + num_blocks - 1) // num_blocks

    x_elem_count = row_per_vec * rope_dim
    sc_elem_count = rope_dim

    @T.prim_func
    def kernel(
        x: T.Tensor([M, hidden_size], dtype),  # type: ignore
        sin: T.Tensor([sc_rows, rope_dim], dtype),  # type: ignore
        cos: T.Tensor([sc_rows, rope_dim], dtype),  # type: ignore
    ):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            # ===== Buffer allocation (all unconditional; MEMORY_PLANNING reclaims
            #       unused mask buffers for the half layout) =====
            x_half_ub = T.alloc_shared([row_per_vec, rope_dim], dtype)
            x_ub = T.alloc_shared([row_per_vec, rope_dim], ACC_DTYPE)
            sin_ub = T.alloc_shared([1, rope_dim], ACC_DTYPE)
            sin_half_ub = T.alloc_shared([1, rope_dim], dtype)
            cos_ub = T.alloc_shared([1, rope_dim], ACC_DTYPE)
            cos_half_ub = T.alloc_shared([1, rope_dim], dtype)
            sin_block_ub = T.alloc_shared([row_per_vec, rope_dim], ACC_DTYPE)
            cos_block_ub = T.alloc_shared([row_per_vec, rope_dim], ACC_DTYPE)
            x_rotate_ub = T.alloc_shared([row_per_vec, rope_dim], ACC_DTYPE)
            out_ub = T.alloc_shared([row_per_vec, rope_dim], ACC_DTYPE)
            # Mask buffers (used by interleaved gather; unused for half)
            mask_ub = T.alloc_shared([row_per_vec, rope_dim], MASK_DTYPE)
            idx_ub = T.alloc_shared([row_per_vec, rope_dim], "int32")
            tmp_ub_i16 = T.alloc_shared([row_per_vec, rope_dim], "int16")
            ones_mask_ub = T.alloc_shared([row_per_vec, rope_dim], "int16")
            mask_ub_i16 = T.alloc_shared([row_per_vec, rope_dim], "int16")
            mask_ub_f32 = T.alloc_shared([row_per_vec, rope_dim], "float32")
            mask_ub_i32 = T.alloc_shared([row_per_vec, rope_dim], "int32")
            # sin_mask (1D UB, needs scalar element access)
            sin_mask_ub = T.alloc_ub(rope_dim, ACC_DTYPE)

            # ===== NPU-internal gather-mask generation (Interleaved only) =====
            if layout == "interleaved":
                T.tile.createvecindex(idx_ub, 0)  # [0,1,2,...]
                T.copy(idx_ub, tmp_ub_i16)
                T.tile.fill(ones_mask_ub, 1)
                T.tile.bitwise_xor(mask_ub_i16, tmp_ub_i16, ones_mask_ub)  # [1,0,3,2,...]
                T.copy(mask_ub_i16, mask_ub_f32)
                T.copy(mask_ub_f32, mask_ub_i32)
                T.tile.mul(mask_ub_i32, mask_ub_i32, 4)  # byte offset (fp32=4B)
                T.reinterpretcast(mask_ub, mask_ub_i32, "uint32_t")

            # ===== NPU-internal sin_mask generation =====
            T.tile.fill(sin_mask_ub, -1.0)
            if layout == "interleaved":
                for i in T.serial(0, half):
                    sin_mask_ub[2 * i + 1] = 1.0  # [-1,+1,-1,+1,...]
            else:  # half
                for i in T.serial(0, half):
                    sin_mask_ub[half + i] = 1.0  # [-1,...,-1,+1,...,+1]

            # ===== Chunk loop =====
            for chunk in T.serial(0, chunks_per_block):
                chunk_idx = cid * chunks_per_block + chunk
                if chunk_idx < total_chunks:
                    row_x = chunk_idx * block_M + vid * row_per_vec
                    row_sin_cos = (row_x // head_num) % sc_rows

                    # --- Load x (fast path vs tail-guard path) ---
                    if row_x + row_per_vec <= M:
                        if dim_start == 0:
                            T.copy(x[row_x : row_x + row_per_vec, :], x_half_ub)
                        else:
                            for i in T.serial(0, row_per_vec):
                                T.copy(x[row_x + i, dim_start:], x_half_ub[i, :])
                    else:
                        for i in T.serial(0, row_per_vec):
                            if row_x + i < M:
                                if dim_start == 0:
                                    T.copy(x[row_x + i, :], x_half_ub[i, :])
                                else:
                                    T.copy(x[row_x + i, dim_start:], x_half_ub[i, :])

                    if need_cast:
                        T.tile.cast(x_ub, x_half_ub, "CAST_NONE", x_elem_count)
                    else:
                        T.copy(x_half_ub, x_ub)

                    # --- Load sin / cos ---
                    T.copy(sin[row_sin_cos, :], sin_half_ub[0, :])
                    if need_cast:
                        T.tile.cast(sin_ub, sin_half_ub, "CAST_NONE", sc_elem_count)
                    else:
                        T.copy(sin_half_ub, sin_ub)
                    T.copy(cos[row_sin_cos, :], cos_half_ub[0, :])
                    if need_cast:
                        T.tile.cast(cos_ub, cos_half_ub, "CAST_NONE", sc_elem_count)
                    else:
                        T.copy(cos_half_ub, cos_ub)

                    # --- Apply sin_mask (in-place) ---
                    T.tile.mul(sin_ub[0, :], sin_ub[0, :], sin_mask_ub)

                    # --- Broadcast sin/cos to [row_per_vec, rope_dim] ---
                    T.tile.broadcast(sin_block_ub, sin_ub)
                    T.tile.broadcast(cos_block_ub, cos_ub)

                    # --- Rotate x (layout branch) ---
                    if layout == "interleaved":
                        T.tile.gather(x_rotate_ub, x_ub, mask_ub, 0)
                    else:  # half — copy-swap
                        for i in T.serial(0, row_per_vec):
                            T.copy(x_ub[i, half:], x_rotate_ub[i, :half])
                            T.copy(x_ub[i, :half], x_rotate_ub[i, half:])

                    # --- out = x * cos + x_rotate * sin_signed ---
                    T.tile.mul(out_ub, x_ub, cos_block_ub)
                    T.tile.mul(x_rotate_ub, x_rotate_ub, sin_block_ub)
                    T.tile.add(out_ub, out_ub, x_rotate_ub)

                    # --- Downcast and write back ---
                    if need_cast:
                        T.tile.cast(x_half_ub, out_ub, "CAST_RINT", x_elem_count)
                    else:
                        T.copy(out_ub, x_half_ub)

                    if row_x + row_per_vec <= M:
                        if dim_start == 0:
                            T.copy(x_half_ub, x[row_x : row_x + row_per_vec, :])
                        else:
                            for i in T.serial(0, row_per_vec):
                                T.copy(x_half_ub[i, :], x[row_x + i, dim_start:])
                    else:
                        for i in T.serial(0, row_per_vec):
                            if row_x + i < M:
                                if dim_start == 0:
                                    T.copy(x_half_ub[i, :], x[row_x + i, :])
                                else:
                                    T.copy(x_half_ub[i, :], x[row_x + i, dim_start:])

    return kernel


# ========== Wrapper ==========
def select_block_M(head_num, rope_dim, layout):
    """Select block_M satisfying head_num alignment + UB budget.

    Constraints:
      - head_num % (block_M // 2) == 0  (sin/cos broadcast correctness)
      - block_M * rope_dim * factor <= 192KB  (UB capacity)

    Factor models total allocation (not peak-after-reuse).  Empirically,
    AscendMemoryPlanning's LinearScanAllocator does not aggressively overlap
    non-overlapping lifetimes (e.g. mask-generation temporaries remain
    resident during chunk processing), so the effective constraint is close
    to the sum of all alloc_shared/alloc_ub buffers.

    Verified: block_M=64, rope_dim=256, interleaved (total ~356KB) crashes
    at compile time; block_M=32 (total ~178KB) passes.  Half layout omits
    7 mask buffers (~68KB), so factor=18.
    """
    UB_LIMIT = 196608  # 192KB
    factor = 22 if layout == "interleaved" else 18
    for bm in [64, 32, 16, 8, 4, 2]:
        rpv = bm // 2
        if head_num % rpv == 0 and bm * rope_dim * factor <= UB_LIMIT:
            return bm
    return 2


def tilelang_rope(x, sin, cos, layout="interleaved"):
    """Apply RoPE in-place to ``x``'s last ``rope_dim`` dimensions.

    Args:
        x:   [BS, N, D] (TND) or [B, S, N, D] (BSND)
        sin: [BS, 1, RD] (TND) or [1, S, 1, RD] (BSND) — raw sin (no sign mask)
        cos: same shape as sin — raw cos
        layout: "interleaved" or "half"

    Returns:
        x (modified in-place, view-restored to original shape)

    Host-side operations are metadata-only views (``.view()``); no
    ``.clone()`` / ``.contiguous()`` / ``torch.cat`` / aclnn calls.
    """
    rope_dim = sin.shape[-1]
    if rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be even, got {rope_dim}")

    org_shape = x.shape

    if x.dim() == 3:
        # TND: x=[BS, N, D], sin/cos=[BS, 1, RD]
        bs, head_num, hidden_size = x.shape
        x = x.view(-1, hidden_size)
        sin = sin.view(-1, rope_dim)
        cos = cos.view(-1, rope_dim)
        sc_rows = bs
    elif x.dim() == 4:
        # BSND: x=[B, S, N, D], sin/cos=[1, S, 1, RD]
        b, s, head_num, hidden_size = x.shape
        x = x.view(-1, hidden_size)
        sin = sin.view(-1, rope_dim)
        cos = cos.view(-1, rope_dim)
        sc_rows = s
    else:
        raise NotImplementedError(f"x.dim()={x.dim()} not supported, expected 3 (TND) or 4 (BSND)")

    if rope_dim > hidden_size:
        raise ValueError(f"rope_dim ({rope_dim}) must be <= hidden_size ({hidden_size})")

    M = x.shape[0]
    dtype_str = tilelang_dtype_map[x.dtype]

    block_M = select_block_M(head_num, rope_dim, layout)

    m_num_full = M // block_M
    tail_rows = M % block_M
    has_tail = 1 if tail_rows > 0 else 0
    total_chunks = m_num_full + has_tail
    num_blocks = min(total_chunks, NUM_CORES)

    kernel = rope_kernel(M, block_M, num_blocks, total_chunks, sc_rows, hidden_size, rope_dim, head_num, layout, dtype=dtype_str)
    kernel(x, sin, cos)

    return x.view(org_shape)


# ========== Reference (Golden) ==========
def torch_rope_ref(x, sin, cos, layout="interleaved"):
    """PyTorch reference for RoPE (Half + Interleaved).

    CPU fp32 intermediate computation, then downcast to ``x.dtype``.
    ``sin`` is raw (no sign mask); the sign is absorbed into ``x_rotated``.
    """
    rope_dim = sin.shape[-1]
    dim_start = x.shape[-1] - rope_dim

    x_part = x[..., dim_start:].to(torch.float32)
    sin_f = sin.to(torch.float32)
    cos_f = cos.to(torch.float32)

    if layout == "interleaved":
        # rotate([x0, x1, x2, x3, ...]) = [-x1, x0, -x3, x2, ...]
        x_reshaped = x_part.reshape(*x_part.shape[:-1], -1, 2)
        x_rotated = torch.stack([-x_reshaped[..., 1], x_reshaped[..., 0]], dim=-1).flatten(-2)
    else:  # half
        # rotate([x1, x2]) = [-x2, x1]
        half = rope_dim // 2
        x_rotated = torch.cat([-x_part[..., half:], x_part[..., :half]], dim=-1)

    rope_out = (x_part * cos_f + x_rotated * sin_f).to(x.dtype)

    out = x.clone()
    out[..., dim_start:] = rope_out
    return out


# ========== Precision Standard (mixed tolerance) ==========
def get_precision(dtype):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio)."""
    fp_table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),  # atol 6.10e-5, rtol 1.95e-3
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),  # atol 9.77e-4, rtol 1.56e-2
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
    }
    if dtype in {"int8", "int16", "int32", "int64", "uint8"}:
        return (0.0, 0.0, 0.0, 1.0)
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    """Mixed-tolerance dual-gate check: returns (passed, matched_ratio, max_abs_error)."""
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a = actual.detach().cpu()
    g = golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:  # integer exact match
        mism = (a != g).sum().item()
        total = max(a.numel(), 1)
        return mism == 0, 1.0 - mism / total, (0.0 if mism == 0 else float("inf"))
    a = a.float()
    g = g.float()
    m = torch.isfinite(g)
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    return (ratio >= required_ratio and max_abs <= max_abs_limit), ratio, max_abs


# ========== Test helpers ==========
def _make_input(shape, dtype_str, layout_kind):
    """Generate test input on CPU. Returns (x_cpu, sin_cpu, cos_cpu)."""
    torch_dtype = torch_dtype_map[dtype_str]
    if layout_kind == "tnd":
        bs, h, hs, rd = shape
        x_cpu = torch.randn(bs, h, hs, dtype=torch_dtype)
        sin_cpu = torch.randn(bs, 1, rd, dtype=torch_dtype)
        cos_cpu = torch.randn(bs, 1, rd, dtype=torch_dtype)
    elif layout_kind == "bsnd":
        b, s, h, hs, rd = shape
        x_cpu = torch.randn(b, s, h, hs, dtype=torch_dtype)
        sin_cpu = torch.randn(1, s, 1, rd, dtype=torch_dtype)
        cos_cpu = torch.randn(1, s, 1, rd, dtype=torch_dtype)
    else:
        raise ValueError(f"Unknown layout_kind: {layout_kind}")
    return x_cpu, sin_cpu, cos_cpu


def run_case(shape, layout, dtype_str, layout_kind, level="l0", inputs=None, tag="PRECISION", label=None):
    """Run a single test case. Returns (passed, ratio, max_abs).

    inputs:  optional (x, sin, cos) CPU tuple; if None, generated via _make_input
    tag:     "PRECISION" (blocking) or "BOUNDARY" (non-blocking)
    label:   optional print label (default: derived from shape/layout/kind)
    """
    if inputs is None:
        x_cpu, sin_cpu, cos_cpu = _make_input(shape, dtype_str, layout_kind)
    else:
        x_cpu, sin_cpu, cos_cpu = inputs

    out_ref = torch_rope_ref(x_cpu.clone(), sin_cpu, cos_cpu, layout)

    x_npu = x_cpu.to(device)
    sin_npu = sin_cpu.to(device)
    cos_npu = cos_cpu.to(device)

    tilelang_rope(x_npu, sin_npu, cos_npu, layout)

    out_npu = x_npu.to("cpu")

    passed, ratio, max_abs = check_precision(out_npu, out_ref, dtype_str)

    status = "PASS" if passed else ("WARN" if tag == "BOUNDARY" else "FAIL")
    desc = label if label else f"shape={shape} layout={layout} dtype={dtype_str} kind={layout_kind}"
    print(f"[{tag}_{status}] {level} {desc} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
    return passed, ratio, max_abs


def _run_cases(cases, level):
    """Run a list of (shape, layout, dtype, kind) cases. Returns all-passed."""
    ok = True
    for shape, layout, dtype_str, kind in cases:
        try:
            passed, _, _ = run_case(shape, layout, dtype_str, kind, level=level)
            ok &= passed
        except Exception as e:
            print(f"[PRECISION_FAIL] {level} shape={shape} layout={layout} dtype={dtype_str} kind={kind}: {e}")
            ok = False
    return ok


def _run_negative(name, make_inputs, expected_exc, layout="interleaved"):
    """Run a negative test: expect tilelang_rope to raise expected_exc."""
    try:
        x, sin, cos = make_inputs()
        tilelang_rope(x, sin, cos, layout)
        print(f"[BOUNDARY_WARN] l2 {name}: expected exception but none raised")
    except expected_exc as e:
        print(f"[BOUNDARY_PASS] l2 {name}: rejected ({type(e).__name__})")
    except Exception as e:
        print(f"[BOUNDARY_WARN] l2 {name}: unexpected {type(e).__name__}: {e}")


# ========== L0 Tests (gate, regular shapes, block-aligned) ==========
L0_CASES = [
    ([16, 64, 512, 256], "interleaved", "float16", "tnd"),
    ([16, 64, 512, 256], "half", "float16", "tnd"),
    ([16, 64, 512, 256], "interleaved", "bfloat16", "tnd"),
    ([16, 64, 512, 256], "half", "bfloat16", "tnd"),
    ([4, 4, 64, 512, 256], "interleaved", "float16", "bsnd"),
    ([4, 4, 64, 512, 256], "half", "float16", "bsnd"),
]


def test_rope_l0():
    """L0 gate tests: 2 layout x 2 dtype x 2 layout_kind = 6 core cases."""
    return _run_cases(L0_CASES, "l0")


# ========== L1 Tests (functional, irregular / tail / varied params) ==========
L1_CASES = [
    ([16, 32, 512, 128], "half", "float16", "tnd"),
    ([16, 8, 256, 64], "interleaved", "float16", "tnd"),
    ([1, 1, 128, 64], "half", "float16", "tnd"),
    ([16, 64, 256, 256], "interleaved", "bfloat16", "tnd"),
    ([128, 64, 512, 256], "half", "float16", "tnd"),
    ([4, 32, 8, 512, 256], "interleaved", "float16", "bsnd"),
    ([5, 32, 512, 256], "interleaved", "float16", "tnd"),
    ([33, 8, 256, 128], "half", "float16", "tnd"),
    ([8, 1, 128, 64], "interleaved", "bfloat16", "tnd"),
    ([2, 16, 32, 256, 256], "half", "bfloat16", "bsnd"),
    ([2, 128, 64, 512, 256], "interleaved", "float16", "bsnd"),
    ([3, 4, 128, 64], "half", "bfloat16", "tnd"),
]


def test_rope_l1():
    """L1 functional tests: varied head_num, rope_dim, full/partial rope, tail rows."""
    return _run_cases(L1_CASES, "l1")


# ========== L2 Tests (negative: invalid inputs should be rejected) ==========
L2_CASES = [
    (
        "odd_rope_dim",
        lambda: (
            torch.randn(4, 8, 128, dtype=torch.float16, device=device),
            torch.randn(4, 1, 63, dtype=torch.float16, device=device),
            torch.randn(4, 1, 63, dtype=torch.float16, device=device),
        ),
        ValueError,
    ),
    (
        "bad_dim_1d",
        lambda: (
            torch.randn(128, dtype=torch.float16, device=device),
            torch.randn(1, 64, dtype=torch.float16, device=device),
            torch.randn(1, 64, dtype=torch.float16, device=device),
        ),
        NotImplementedError,
    ),
    (
        "bad_dim_5d",
        lambda: (
            torch.randn(2, 4, 8, 16, 64, dtype=torch.float16, device=device),
            torch.randn(2, 4, 1, 32, dtype=torch.float16, device=device),
            torch.randn(2, 4, 1, 32, dtype=torch.float16, device=device),
        ),
        NotImplementedError,
    ),
    (
        "rope_dim_too_large",
        lambda: (
            torch.randn(4, 8, 64, dtype=torch.float16, device=device),
            torch.randn(4, 1, 128, dtype=torch.float16, device=device),
            torch.randn(4, 1, 128, dtype=torch.float16, device=device),
        ),
        (ValueError, IndexError, RuntimeError),
    ),
]


def test_rope_l2():
    """L2 negative tests: invalid inputs must raise exceptions (non-blocking)."""
    for name, make_inputs, expected_exc in L2_CASES:
        _run_negative(name, make_inputs, expected_exc)


# ========== Boundary Tests (legal special/extreme values, non-blocking) ==========
BOUNDARY_CASES = [
    (
        "large_values",
        lambda: (
            torch.randn(16, 64, 512, dtype=torch.float16) * 1000,
            torch.randn(16, 1, 256, dtype=torch.float16),
            torch.randn(16, 1, 256, dtype=torch.float16),
        ),
        "interleaved",
        "float16",
    ),
    (
        "zero_values",
        lambda: (
            torch.zeros(16, 64, 512, dtype=torch.float16),
            torch.randn(16, 1, 256, dtype=torch.float16),
            torch.randn(16, 1, 256, dtype=torch.float16),
        ),
        "half",
        "float16",
    ),
    (
        "full_rope_large_batch",
        lambda: (
            torch.randn(128, 64, 256, dtype=torch.float16),
            torch.randn(128, 1, 256, dtype=torch.float16),
            torch.randn(128, 1, 256, dtype=torch.float16),
        ),
        "interleaved",
        "float16",
    ),
    (
        "min_rope_dim",
        lambda: (
            torch.randn(8, 16, 128, dtype=torch.bfloat16),
            torch.randn(8, 1, 64, dtype=torch.bfloat16),
            torch.randn(8, 1, 64, dtype=torch.bfloat16),
        ),
        "half",
        "bfloat16",
    ),
    (
        "single_row",
        lambda: (
            torch.randn(1, 1, 128, dtype=torch.float16),
            torch.randn(1, 1, 64, dtype=torch.float16),
            torch.randn(1, 1, 64, dtype=torch.float16),
        ),
        "interleaved",
        "float16",
    ),
    (
        "bsnd_tail",
        lambda: (
            torch.randn(3, 33, 8, 256, dtype=torch.float16),
            torch.randn(1, 33, 1, 128, dtype=torch.float16),
            torch.randn(1, 33, 1, 128, dtype=torch.float16),
        ),
        "half",
        "float16",
    ),
]


def test_rope_boundary():
    """Boundary tests: extreme values, full rope, minimal rope_dim, large batch."""
    torch.manual_seed(123)
    for name, make_inputs, layout, dtype_str in BOUNDARY_CASES:
        try:
            run_case(None, layout, dtype_str, None, level="boundary", inputs=make_inputs(), tag="BOUNDARY", label=name)
        except Exception as e:
            print(f"[BOUNDARY_WARN] boundary {name}: {type(e).__name__}: {e}")


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description="RoPE (Half + Interleaved) on Ascend NPU")
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all"], help="Test level to run (default: l0)")
    parser.add_argument("--shape", type=int, nargs="+", help="Single-case shape: 4 ints (TND: BS H HS RD) or 5 ints (BSND: B S H HS RD)")
    parser.add_argument("--layout", default="interleaved", choices=["interleaved", "half"], help="RoPE layout (default: interleaved)")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"], help="Data type (default: float16)")
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(42)

    # Single-case mode
    if args.shape:
        if len(args.shape) == 4:
            kind = "tnd"
        elif len(args.shape) == 5:
            kind = "bsnd"
        else:
            print(f"Error: --shape needs 4 (TND) or 5 (BSND) values, got {len(args.shape)}")
            sys.exit(1)
        try:
            passed, ratio, max_abs = run_case(args.shape, args.layout, args.dtype, kind, level="single")
            if passed:
                print("Test Passed!")
                sys.exit(0)
            sys.exit(1)
        except Exception as e:
            print(f"[PRECISION_FAIL] single shape={args.shape} layout={args.layout} dtype={args.dtype}: {e}")
            sys.exit(1)

    # Level mode
    blocking_ok = True
    if args.level in ("l0", "all"):
        print("=" * 60)
        print("L0 Gate Tests")
        print("=" * 60)
        blocking_ok &= test_rope_l0()
    if args.level in ("l1", "all"):
        print("=" * 60)
        print("L1 Functional Tests")
        print("=" * 60)
        blocking_ok &= test_rope_l1()
    if args.level in ("l2", "all"):
        print("=" * 60)
        print("L2 Negative Tests")
        print("=" * 60)
        test_rope_l2()
    if args.level in ("boundary", "all"):
        print("=" * 60)
        print("Boundary Tests")
        print("=" * 60)
        test_rope_boundary()

    print("=" * 60)
    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    else:
        print("Test FAILED (L0/L1 precision errors)")
        sys.exit(1)


if __name__ == "__main__":
    main()
