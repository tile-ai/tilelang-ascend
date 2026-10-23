# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms of the CANN Open Software License Agreement Version 2.0 (the
# "License"). You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
# KIND, EITHER EXPRESS OR IMPLIED.
# ----------------------------------------------------------------------------

"""RMSNorm TileLang kernel — dual-kernel strategy (Hybrid simple + Expert pipeline).

Two-pass RMS normalization for arbitrary shape, dtype, and epsilon.

Algorithm:
    y = x / sqrt(mean(x^2) + eps) * gamma

Reference: torch.nn.functional.rms_norm (golden_rms_norm in test_rms_norm.py)

Key optimizations:
- Dual-kernel: n_num=1 uses Hybrid (AUTO_SYNC=True), n_num>1 uses Expert
  (AUTO_SYNC=False + double buffer + three-way flag pipeline)
- Adaptive tiling via UB budget formula
- Newton iteration for rsqrt precision (~1e-3 -> ~1e-6)
- output_ub separation keeps a_ub MTE2-write-only (enables double buffer)

Interface: rms_norm_host(x, gamma, eps) -> y
"""

import torch  # noqa: F401  (used by the __main__ smoke test)

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
    if not isinstance(x, torch.Tensor) or not isinstance(gamma, torch.Tensor):
        raise TypeError(f"rms_norm_host: x/gamma must be torch tensors, got {type(x).__name__}/{type(gamma).__name__}")
    if x.dtype not in (torch.float16, torch.float32, torch.bfloat16) or gamma.dtype != x.dtype:
        raise TypeError(f"rms_norm_host: x/gamma must be float16/float32/bfloat16 of the same dtype, got x={x.dtype}, gamma={gamma.dtype}")
    if gamma.dim() != 1 or gamma.shape[0] != x.shape[-1]:
        raise ValueError(f"rms_norm_host: gamma must be [D] with D=x.shape[-1]={x.shape[-1]}, got shape {tuple(gamma.shape)}")
    if not x.is_contiguous() or not gamma.is_contiguous():
        raise ValueError("rms_norm_host: x/gamma must be contiguous tensors")

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


# ----------------------------------------------------------------------------
# CI smoke test (examples/bench_test.sh runs this file directly and requires
# "Kernel Output Match!" or "TEST PASSED!" in the output, same as the
# lstm/gru/depthwise_conv_2d/nms/group_norm smoke tests).  The full layered
# suite over the 20 cann-bench cases lives in test_rms_norm.py.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import torch.nn.functional as F

    torch.manual_seed(0)

    # one case per kernel path and dtype: Hybrid simple / Expert pipeline / 3D unaligned
    for shape, dtype, eps in (
        ((32, 128, 768), torch.float16, 1e-6),  # n_num=1 -> Hybrid simple kernel
        ((8, 512, 8192), torch.float32, 1e-6),  # n_num>1 -> Expert pipeline kernel
        ((63, 67, 1023), torch.bfloat16, 1e-8),  # 3D unaligned
    ):
        x = (torch.rand(*shape) * 2 - 1).to(dtype)
        gamma = torch.ones(shape[-1], dtype=dtype)
        y = rms_norm_host(x.npu(), gamma.npu(), eps).cpu()
        ref = F.rms_norm(x.float(), [shape[-1]], weight=gamma.float(), eps=eps).to(dtype)
        atol, rtol = {
            torch.float16: (1e-3, 1e-3),
            torch.float32: (1e-4, 1e-4),
            torch.bfloat16: (1e-2, 5e-3),
        }[dtype]
        torch.testing.assert_close(y, ref, atol=atol, rtol=rtol)
        diff = (y.float() - ref.float()).abs().max().item()
        print(f"[SMOKE] {tuple(shape)} {str(dtype).split('.')[-1]} max_abs={diff:.3e}")
    print("Kernel Output Match!")
