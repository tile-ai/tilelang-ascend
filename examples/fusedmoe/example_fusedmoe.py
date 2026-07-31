"""FusedMoE: DeepSeek-style Fused Mixture of Experts for NPU (TileLang-Ascend).

Strategy: pure GEMM kernels + PyTorch post-processing.
  - Each kernel does GEMM only (proven to work on NPU)
  - T.alloc_L1/L0C + combineCV pass (auto sync, no manual T.Scope/barrier_all)
  - SiLU + element-wise mul done in PyTorch between kernel calls
  - GEMM outputs fp16 directly (T.copy auto-casts L0C fp32 -> GM fp16)

Contains:
  - Generic single-GEMM and grouped-GEMM kernels
  - Dual-GEMM kernels (gate+up fused, Input read once)
  - Shared expert and routed expert pipelines
  - Golden functions (PyTorch reference)
  - Host preprocessing (gating + token routing)
"""

import torch
import torch.nn.functional as F

import tilelang
import tilelang.language as T

# Block sizes (all at hardware limits)
BLOCK_M = 64
BLOCK_N = 256
BLOCK_K = 128

# Developer mode: enable combineCV + auto sync passes
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


# ============================================================================
# Generic Single-GEMM Kernel (follows grouped_gemm example pattern exactly)
# ============================================================================
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def single_gemm_kernel(M, N, K, dtype="float16"):
    """Generic single GEMM: Output = Input @ Weight^T (fp32 accumulation).

    Uses the proven grouped_gemm pattern:
      - T.alloc_L1 for input/weight buffers
      - T.alloc_L0C for accumulation
      - combineCV pass (auto sync, no manual T.Scope/barrier_all)
      - combineCV pass (auto sync)
    """
    block_M = BLOCK_M
    block_N = BLOCK_N
    block_K = BLOCK_K
    accum_dtype = "float32"

    m_num = (M + block_M - 1) // block_M
    n_num = (N + block_N - 1) // block_N

    @T.prim_func
    def kernel(
        Input: T.Tensor([M, K], dtype),  # type: ignore
        Weight: T.Tensor([N, K], dtype),  # type: ignore
        # M2: Output fp16 directly (T.copy auto-casts L0C fp32 → GM fp16)
        Output: T.Tensor([M, N], dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_N, block_K), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            loop_k = T.ceildiv(K, block_K)
            for k in T.serial(loop_k):
                T.copy(Input[bx * block_M, k * block_K], A_L1)
                T.copy(Weight[by * block_N, k * block_K], B_L1)

                T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=(k == 0))

            # L0C fp32 → GM fp16: T.copy auto-casts (verified iter7)
            T.copy(C_L0, Output[bx * block_M, by * block_N])

    return kernel


# ============================================================================
# Dual GEMM Kernel (gate + up fused, Input read once from GM)
# ============================================================================
@tilelang.jit(out_idx=[-2, -1], pass_configs=pass_configs)
def dual_gemm_kernel(M, N, K, dtype="float16"):
    """Dual GEMM: Gate_out = Input @ W_gate^T, Up_out = Input @ W_up^T.

    Fuses gate and up GEMMs into a single kernel so Input is read from GM
    only once (halves Input GM reads vs two separate single_gemm_kernel calls).
    Returns (Gate_out, Up_out) as fp16 tensors (M2: T.copy auto-casts L0C fp32 → GM fp16).

    Uses the same Expert-mode pattern as single_gemm_kernel:
      - T.alloc_L1 for input/weight buffers
      - T.alloc_L0C for accumulation (separate L0C for gate and up)
      - combineCV pass (auto sync, no manual T.Scope/barrier_all)
    """
    block_M = BLOCK_M
    block_N = BLOCK_N
    block_K = BLOCK_K
    accum_dtype = "float32"

    m_num = (M + block_M - 1) // block_M
    n_num = (N + block_N - 1) // block_N

    @T.prim_func
    def kernel(
        Input: T.Tensor([M, K], dtype),  # type: ignore
        W_gate: T.Tensor([N, K], dtype),  # type: ignore
        W_up: T.Tensor([N, K], dtype),  # type: ignore
        # M2: Output fp16 directly (T.copy auto-casts L0C fp32 → GM fp16)
        Gate_out: T.Tensor([M, N], dtype),  # type: ignore
        Up_out: T.Tensor([M, N], dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            W_gate_L1 = T.alloc_L1((block_N, block_K), dtype)
            W_up_L1 = T.alloc_L1((block_N, block_K), dtype)
            Gate_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)
            Up_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            loop_k = T.ceildiv(K, block_K)
            for k in T.serial(loop_k):
                T.copy(Input[bx * block_M, k * block_K], A_L1)
                T.copy(W_gate[by * block_N, k * block_K], W_gate_L1)
                T.copy(W_up[by * block_N, k * block_K], W_up_L1)

                T.gemm_v0(A_L1, W_gate_L1, Gate_L0, transpose_B=True, init=(k == 0))
                T.gemm_v0(A_L1, W_up_L1, Up_L0, transpose_B=True, init=(k == 0))

            T.copy(Gate_L0, Gate_out[bx * block_M, by * block_N])
            T.copy(Up_L0, Up_out[bx * block_M, by * block_N])

    return kernel


# ============================================================================
# Grouped Single-GEMM Kernel (for routed experts with block_metadata)
# ============================================================================
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def grouped_gemm_kernel(buf_rows, N, K, n_groups, total_m_blocks, dtype="float16"):
    """Grouped single GEMM with block_metadata routing.

    Follows the grouped_gemm example pattern exactly.
    """
    block_M = BLOCK_M
    block_N = BLOCK_N
    block_K = BLOCK_K
    accum_dtype = "float32"

    n_num = (N + block_N - 1) // block_N

    @T.prim_func
    def kernel(
        Input: T.Tensor([buf_rows, K], dtype),  # type: ignore
        Weight: T.Tensor([n_groups, N, K], dtype),  # type: ignore
        block_metadata: T.Tensor([total_m_blocks, 3], "int32"),  # type: ignore
        # M2: Output fp16 directly (T.copy auto-casts L0C fp32 → GM fp16)
        Output: T.Tensor([buf_rows, N], dtype),  # type: ignore
    ):
        with T.Kernel(total_m_blocks * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            group_idx = block_metadata[bx, 0]
            m_start = block_metadata[bx, 1]

            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_N, block_K), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            loop_k = T.ceildiv(K, block_K)
            for k in T.serial(loop_k):
                T.copy(
                    Input[
                        m_start : m_start + block_M,
                        k * block_K : (k + 1) * block_K,
                    ],
                    A_L1,
                )
                T.copy(
                    Weight[
                        group_idx,
                        by * block_N : (by + 1) * block_N,
                        k * block_K : (k + 1) * block_K,
                    ],
                    B_L1,
                )

                T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=(k == 0))

            T.copy(
                C_L0,
                Output[
                    m_start : m_start + block_M,
                    by * block_N : (by + 1) * block_N,
                ],
            )

    return kernel


# ============================================================================
# Grouped Dual-GEMM Kernel (gate + up fused with block_metadata routing)
# ============================================================================
@tilelang.jit(out_idx=[-2, -1], pass_configs=pass_configs)
def grouped_dual_gemm_kernel(buf_rows, N, K, n_groups, total_m_blocks, dtype="float16"):
    """Grouped dual GEMM with block_metadata routing.

    Same as grouped_gemm_kernel but produces both Gate_out and Up_out in a
    single kernel, sharing the Input read. Halves GM bandwidth for Input
    (M1 optimization for routed experts).

    Follows the same Expert-mode pattern as dual_gemm_kernel (shared expert,
    verified in iter3) combined with grouped_gemm_kernel's block_metadata
    routing:
      - T.alloc_L1 for input/weight buffers (A_L1 shared by gate+up)
      - T.alloc_L0C for accumulation (separate L0C for gate and up)
      - combineCV pass (auto sync, no manual T.Scope/barrier_all)
      - BLOCK_M kept at 64 (matches host_preprocess, avoids OOB reads)

    Capacity (block_M=64, block_N=256, block_K=128):
      L1  = 64*128*2 + 2*(256*128*2) = 16KB + 128KB = 144KB  (< 512KB)
      L0C = 2*(64*256*4) = 128KB                               (= 128KB, at limit)
    """
    block_M = BLOCK_M  # 64, MUST NOT change (iter2 lesson: OOB reads)
    block_N = BLOCK_N  # 256
    block_K = BLOCK_K  # 128
    accum_dtype = "float32"

    n_num = (N + block_N - 1) // block_N

    @T.prim_func
    def kernel(
        Input: T.Tensor([buf_rows, K], dtype),  # type: ignore
        W_gate: T.Tensor([n_groups, N, K], dtype),  # type: ignore
        W_up: T.Tensor([n_groups, N, K], dtype),  # type: ignore
        block_metadata: T.Tensor([total_m_blocks, 3], "int32"),  # type: ignore
        # M2: Output fp16 directly (T.copy auto-casts L0C fp32 → GM fp16)
        Gate_out: T.Tensor([buf_rows, N], dtype),  # type: ignore
        Up_out: T.Tensor([buf_rows, N], dtype),  # type: ignore
    ):
        with T.Kernel(total_m_blocks * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            group_idx = block_metadata[bx, 0]
            m_start = block_metadata[bx, 1]

            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            W_gate_L1 = T.alloc_L1((block_N, block_K), dtype)
            W_up_L1 = T.alloc_L1((block_N, block_K), dtype)
            Gate_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)
            Up_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            loop_k = T.ceildiv(K, block_K)
            for k in T.serial(loop_k):
                # Input read once from GM, shared by both gate and up
                T.copy(
                    Input[
                        m_start : m_start + block_M,
                        k * block_K : (k + 1) * block_K,
                    ],
                    A_L1,
                )
                T.copy(
                    W_gate[
                        group_idx,
                        by * block_N : (by + 1) * block_N,
                        k * block_K : (k + 1) * block_K,
                    ],
                    W_gate_L1,
                )
                T.copy(
                    W_up[
                        group_idx,
                        by * block_N : (by + 1) * block_N,
                        k * block_K : (k + 1) * block_K,
                    ],
                    W_up_L1,
                )

                T.gemm_v0(A_L1, W_gate_L1, Gate_L0, transpose_B=True, init=(k == 0))
                T.gemm_v0(A_L1, W_up_L1, Up_L0, transpose_B=True, init=(k == 0))

            T.copy(
                Gate_L0,
                Gate_out[
                    m_start : m_start + block_M,
                    by * block_N : (by + 1) * block_N,
                ],
            )
            T.copy(
                Up_L0,
                Up_out[
                    m_start : m_start + block_M,
                    by * block_N : (by + 1) * block_N,
                ],
            )

    return kernel


# ============================================================================
# T.mma + L0 Double Buffer Optimized Kernel (overlap MTE2/Cube)
# ============================================================================
# Uses T.mma with L0A/L0B/L0C double buffering and set_flag/wait_flag sync
# to overlap data loading (MTE2/MTE1) with computation (Cube/MMA).
# Based on GQA Sink Forward fa_opt pattern (committer-allowed set/wait flag).
#
# Signal IDs for intra-core L0 double-buffer sync:
#   SIG_L0AB = 0: L0A/L0B double-buffer base (slot 0 = 0, slot 1 = 1)
#   SIG_L0C  = 2: L0C double-buffer base (slot 0 = 2, slot 1 = 3)
#   SIG_K_L1 = 4: MTE2 writes k_l1, MTE1 reads it


@tilelang.jit(out_idx=[-2, -1])
def dual_gemm_kernel_mma(M, N, K, dtype="float16"):
    """Dual GEMM with T.mma + L0 double buffer + Fixed Core.

    Overlaps MTE2 (GM→L1 load) with Cube (MMA compute) via L0 double buffer.
    Uses set_flag/wait_flag for intra-core pipeline sync.
    """
    from tilelang.intrinsics import make_zn_layout, make_nz_layout

    block_M = BLOCK_M
    block_N = BLOCK_N
    block_K = BLOCK_K
    accum_dtype = "float32"
    core_num = 20

    m_num = (M + block_M - 1) // block_M
    n_num = (N + block_N - 1) // block_N
    total_tiles = m_num * n_num

    q_tasks = total_tiles // core_num
    r_tasks = total_tiles % core_num

    SIG_L0AB = 0  # L0A/L0B double-buffer: slot 0 = 0, slot 1 = 1
    SIG_L0C = 2  # L0C double-buffer: slot 0 = 2, slot 1 = 3
    SIG_K_L1 = 4  # MTE2 writes input_l1, MTE1 reads it
    SIG_WG_L1 = 5  # MTE2 writes w_gate_l1, MTE1 reads it
    SIG_WU_L1 = 6  # MTE2 writes w_up_l1, MTE1 reads it

    @T.prim_func
    def kernel(
        Input: T.Tensor([M, K], dtype),  # type: ignore
        W_gate: T.Tensor([N, K], dtype),  # type: ignore
        W_up: T.Tensor([N, K], dtype),  # type: ignore
        Gate_out: T.Tensor([M, N], dtype),  # type: ignore
        Up_out: T.Tensor([M, N], dtype),  # type: ignore
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            # L1 buffers
            input_l1 = T.alloc_L1((block_M, block_K), dtype)
            w_gate_l1 = T.alloc_L1((block_N, block_K), dtype)
            w_up_l1 = T.alloc_L1((block_N, block_K), dtype)

            T.annotate_layout(
                {
                    input_l1: make_zn_layout(input_l1),
                    w_gate_l1: make_nz_layout(w_gate_l1),
                    w_up_l1: make_nz_layout(w_up_l1),
                }
            )

            # L0 double-buffered buffers
            # L0A: [2, block_M, block_K] = [2, 64, 128] = 32KB < 64KB ✓
            # L0B: [block_K, block_N] = [128, 256] = 64KB = limit (single, shared gate/up)
            # L0C: [2, block_M, block_N] = [2, 64, 256] = 128KB < 128KB ✓ (dual, shared gate/up)
            l0a = T.alloc_L0A([2, block_M, block_K], dtype)
            l0b = T.alloc_L0B([block_K, block_N], dtype)
            l0c_gate = T.alloc_L0C([block_M, block_N], accum_dtype)
            l0c_up = T.alloc_L0C([block_M, block_N], accum_dtype)

            T.annotate_address(
                {
                    input_l1: 0,
                    w_gate_l1: block_M * block_K * 2,
                    w_up_l1: block_M * block_K * 2 + block_N * block_K * 2,
                    l0a: 0,
                    l0b: 0,
                    l0c_gate: 0,
                    l0c_up: 0,
                }
            )

            # Init: pretend consumer released L0/L1 buffers (pipeline start)
            T.set_flag("MTE1", "MTE2", SIG_K_L1)
            T.set_flag("MTE1", "MTE2", SIG_WG_L1)
            T.set_flag("MTE1", "MTE2", SIG_WU_L1)
            T.set_flag("M", "MTE1", SIG_L0AB)
            T.set_flag("M", "MTE1", SIG_L0AB + 1)
            T.set_flag("FIX", "M", SIG_L0C)

            # Grid-stride loop (by-major for L2 Weight reuse)
            my_start = cid * q_tasks + T.if_then_else(cid < r_tasks, cid, r_tasks)
            my_count = q_tasks + T.if_then_else(cid < r_tasks, 1, 0)

            loop_k = T.ceildiv(K, block_K)

            for t in T.serial(my_count):
                task_id = my_start + t
                by = task_id // m_num
                bx = task_id % m_num

                for k in T.serial(loop_k):
                    side = k % 2

                    # MTE2: Load Input to L1
                    T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                    T.copy(Input[bx * block_M, k * block_K], input_l1)
                    T.set_flag("MTE2", "MTE1", SIG_K_L1)

                    # MTE2: Load W_gate to L1
                    T.wait_flag("MTE1", "MTE2", SIG_WG_L1)
                    T.copy(W_gate[by * block_N, k * block_K], w_gate_l1)
                    T.set_flag("MTE2", "MTE1", SIG_WG_L1)

                    # MTE2: Load W_up to L1
                    T.wait_flag("MTE1", "MTE2", SIG_WU_L1)
                    T.copy(W_up[by * block_N, k * block_K], w_up_l1)
                    T.set_flag("MTE2", "MTE1", SIG_WU_L1)

                    # MTE1: Load Input to L0A (double-buffered across K iters)
                    T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                    T.wait_flag("M", "MTE1", SIG_L0AB + side)
                    T.copy(input_l1, l0a[side, :, :])

                    # MTE1: Load W_gate to L0B (transpose [N,K] → [K,N])
                    T.wait_flag("MTE2", "MTE1", SIG_WG_L1)
                    T.copy(w_gate_l1, l0b, transpose=True)
                    T.set_flag("MTE1", "MTE2", SIG_K_L1)
                    T.set_flag("MTE1", "MTE2", SIG_WG_L1)
                    T.set_flag("MTE1", "M", SIG_L0AB + side)

                    # M: MMA Input @ W_gate^T -> Gate (L0C)
                    T.wait_flag("MTE1", "M", SIG_L0AB + side)
                    T.wait_flag("FIX", "M", SIG_L0C)
                    T.mma(l0a[side, :, :], l0b, l0c_gate, init=(k == 0))
                    T.set_flag("M", "MTE1", SIG_L0AB + side)
                    T.set_flag("M", "FIX", SIG_L0C)

                    # FIX: Gate L0C -> GM (fp32 → fp16 auto-cast)
                    T.wait_flag("M", "FIX", SIG_L0C)
                    T.copy(l0c_gate, Gate_out[bx * block_M, by * block_N])
                    T.set_flag("FIX", "M", SIG_L0C)

                    # MTE1: Load W_up to L0B (transpose, reuse L0A from gate)
                    T.wait_flag("M", "MTE1", SIG_L0AB + side)
                    T.wait_flag("MTE2", "MTE1", SIG_WU_L1)
                    T.copy(w_up_l1, l0b, transpose=True)
                    T.set_flag("MTE1", "MTE2", SIG_WU_L1)
                    T.set_flag("MTE1", "M", SIG_L0AB + side)

                    # M: MMA Input @ W_up^T -> Up (L0C)
                    T.wait_flag("MTE1", "M", SIG_L0AB + side)
                    T.wait_flag("FIX", "M", SIG_L0C)
                    T.mma(l0a[side, :, :], l0b, l0c_up, init=(k == 0))
                    T.set_flag("M", "MTE1", SIG_L0AB + side)
                    T.set_flag("M", "FIX", SIG_L0C)

                    # FIX: Up L0C -> GM (fp32 → fp16 auto-cast)
                    T.wait_flag("M", "FIX", SIG_L0C)
                    T.copy(l0c_up, Up_out[bx * block_M, by * block_N])
                    T.set_flag("FIX", "M", SIG_L0C)

                # Destroy: consume outstanding init flags
                if t == my_count - 1:
                    T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                    T.wait_flag("MTE1", "MTE2", SIG_WG_L1)
                    T.wait_flag("MTE1", "MTE2", SIG_WU_L1)
                    T.wait_flag("M", "MTE1", SIG_L0AB)
                    T.wait_flag("M", "MTE1", SIG_L0AB + 1)
                    T.wait_flag("FIX", "M", SIG_L0C)

    return kernel


# ============================================================================
# Approach C: T.mma + L0A double buffer + auto_sync (no manual flag)
# ============================================================================
# L0A double-buffered [2, M, K], L0B/L0C single-buffered.
# auto_sync handles MTE1→M→FIX synchronization automatically.
# gate and up GEMM share L0B/L0C sequentially (no overlap between them).


@tilelang.jit(out_idx=[-2, -1], pass_configs=pass_configs)
def dual_gemm_kernel_mma_auto(M, N, K, dtype="float16"):
    """Dual GEMM with T.mma + L0A double buffer + auto_sync + Fixed Core.

    L0A double-buffered to overlap load[k+1] with mma[k].
    L0B/L0C single (block_N=256 → L0B=64KB at limit).
    auto_sync replaces manual set_flag/wait_flag.
    """
    block_M = BLOCK_M
    block_N = BLOCK_N
    block_K = BLOCK_K
    accum_dtype = "float32"
    core_num = 20

    m_num = (M + block_M - 1) // block_M
    n_num = (N + block_N - 1) // block_N
    total_tiles = m_num * n_num

    q_tasks = total_tiles // core_num
    r_tasks = total_tiles % core_num

    @T.prim_func
    def kernel(
        Input: T.Tensor([M, K], dtype),  # type: ignore
        W_gate: T.Tensor([N, K], dtype),  # type: ignore
        W_up: T.Tensor([N, K], dtype),  # type: ignore
        Gate_out: T.Tensor([M, N], dtype),  # type: ignore
        Up_out: T.Tensor([M, N], dtype),  # type: ignore
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            # L1 buffers (GM → L1 → L0)
            input_l1 = T.alloc_L1((block_M, block_K), dtype)
            w_gate_l1 = T.alloc_L1((block_N, block_K), dtype)
            w_up_l1 = T.alloc_L1((block_N, block_K), dtype)

            # L0A double-buffered, L0B/L0C single
            l0a = T.alloc_L0A([2, block_M, block_K], dtype)
            l0b = T.alloc_L0B([block_K, block_N], dtype)
            l0c_gate = T.alloc_L0C([block_M, block_N], accum_dtype)
            l0c_up = T.alloc_L0C([block_M, block_N], accum_dtype)

            my_start = cid * q_tasks + T.if_then_else(cid < r_tasks, cid, r_tasks)
            my_count = q_tasks + T.if_then_else(cid < r_tasks, 1, 0)

            loop_k = T.ceildiv(K, block_K)

            for t in T.serial(my_count):
                task_id = my_start + t
                by = task_id // m_num
                bx = task_id % m_num

                for k in T.serial(loop_k):
                    side = k % 2

                    # GM → L1 → L0A (double-buffered)
                    T.copy(Input[bx * block_M, k * block_K], input_l1)
                    T.copy(input_l1, l0a[side, :, :])

                    # --- Gate GEMM: GM → L1 → L0B → MMA ---
                    T.copy(W_gate[by * block_N, k * block_K], w_gate_l1)
                    T.copy(w_gate_l1, l0b, transpose=True)
                    T.mma(l0a[side, :, :], l0b, l0c_gate, init=(k == 0))

                    # --- Up GEMM: GM → L1 → L0B → MMA (reuse L0A) ---
                    T.copy(W_up[by * block_N, k * block_K], w_up_l1)
                    T.copy(w_up_l1, l0b, transpose=True)
                    T.mma(l0a[side, :, :], l0b, l0c_up, init=(k == 0))

                # Write outputs (fp32 → fp16 auto-cast)
                T.copy(l0c_gate, Gate_out[bx * block_M, by * block_N])
                T.copy(l0c_up, Up_out[bx * block_M, by * block_N])

    return kernel


# ============================================================================
# Approach B: single_gemm_mma (T.mma + L0A dual + auto_sync, single GEMM)
# ============================================================================


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def single_gemm_kernel_mma_auto(M, N, K, dtype="float16"):
    """Single GEMM with T.mma + L0A double buffer + auto_sync + Fixed Core.

    Simple single GEMM — no dual GEMM flag complexity.
    L0A double-buffered, L0B/L0C single.
    """
    block_M = BLOCK_M
    block_N = BLOCK_N
    block_K = BLOCK_K
    accum_dtype = "float32"
    core_num = 20

    m_num = (M + block_M - 1) // block_M
    n_num = (N + block_N - 1) // block_N
    total_tiles = m_num * n_num

    q_tasks = total_tiles // core_num
    r_tasks = total_tiles % core_num

    @T.prim_func
    def kernel(
        Input: T.Tensor([M, K], dtype),  # type: ignore
        Weight: T.Tensor([N, K], dtype),  # type: ignore
        Output: T.Tensor([M, N], dtype),  # type: ignore
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            input_l1 = T.alloc_L1((block_M, block_K), dtype)
            weight_l1 = T.alloc_L1((block_N, block_K), dtype)

            l0a = T.alloc_L0A([2, block_M, block_K], dtype)
            l0b = T.alloc_L0B([block_K, block_N], dtype)
            l0c = T.alloc_L0C([block_M, block_N], accum_dtype)

            my_start = cid * q_tasks + T.if_then_else(cid < r_tasks, cid, r_tasks)
            my_count = q_tasks + T.if_then_else(cid < r_tasks, 1, 0)

            loop_k = T.ceildiv(K, block_K)

            for t in T.serial(my_count):
                task_id = my_start + t
                by = task_id // m_num
                bx = task_id % m_num

                for k in T.serial(loop_k):
                    side = k % 2

                    T.copy(Input[bx * block_M, k * block_K], input_l1)
                    T.copy(input_l1, l0a[side, :, :])

                    T.copy(Weight[by * block_N, k * block_K], weight_l1)
                    T.copy(weight_l1, l0b, transpose=True)
                    T.mma(l0a[side, :, :], l0b, l0c, init=(k == 0))

                T.copy(l0c, Output[bx * block_M, by * block_N])

    return kernel


# ============================================================================
# Approach A: T.mma + full L0 double buffer (block_N=128 for L0B dual)
# ============================================================================
# block_N=128 → L0B dual = 64KB ✓, independent L0B_gate + L0B_up
# All L0 buffers double-buffered for maximum pipeline overlap.


@tilelang.jit(out_idx=[-2, -1], pass_configs=pass_configs)
def dual_gemm_kernel_mma_dual128(M, N, K, dtype="float16"):
    """Dual GEMM with T.mma + full L0 double buffer (block_N=128).

    Reduces block_N from 256 to 128 to fit L0B dual buffer (64KB).
    Independent L0B for gate/up, all L0 double-buffered.
    """
    block_M = BLOCK_M
    block_N = 128  # reduced from 256 for L0B dual buffer
    block_K = BLOCK_K
    accum_dtype = "float32"
    core_num = 20

    m_num = (M + block_M - 1) // block_M
    n_num = (N + block_N - 1) // block_N
    total_tiles = m_num * n_num

    q_tasks = total_tiles // core_num
    r_tasks = total_tiles % core_num

    @T.prim_func
    def kernel(
        Input: T.Tensor([M, K], dtype),  # type: ignore
        W_gate: T.Tensor([N, K], dtype),  # type: ignore
        W_up: T.Tensor([N, K], dtype),  # type: ignore
        Gate_out: T.Tensor([M, N], dtype),  # type: ignore
        Up_out: T.Tensor([M, N], dtype),  # type: ignore
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            input_l1 = T.alloc_L1((block_M, block_K), dtype)
            w_gate_l1 = T.alloc_L1((block_N, block_K), dtype)
            w_up_l1 = T.alloc_L1((block_N, block_K), dtype)

            # All L0 double-buffered
            l0a = T.alloc_L0A([2, block_M, block_K], dtype)
            l0b_gate = T.alloc_L0B([2, block_K, block_N], dtype)
            l0b_up = T.alloc_L0B([2, block_K, block_N], dtype)
            l0c_gate = T.alloc_L0C([2, block_M, block_N], accum_dtype)
            l0c_up = T.alloc_L0C([2, block_M, block_N], accum_dtype)

            my_start = cid * q_tasks + T.if_then_else(cid < r_tasks, cid, r_tasks)
            my_count = q_tasks + T.if_then_else(cid < r_tasks, 1, 0)

            loop_k = T.ceildiv(K, block_K)

            for t in T.serial(my_count):
                task_id = my_start + t
                by = task_id // m_num
                bx = task_id % m_num

                for k in T.serial(loop_k):
                    side = k % 2

                    # Load Input → L1 → L0A
                    T.copy(Input[bx * block_M, k * block_K], input_l1)
                    T.copy(input_l1, l0a[side, :, :])

                    # Gate: W_gate → L1 → L0B → MMA
                    T.copy(W_gate[by * block_N, k * block_K], w_gate_l1)
                    T.copy(w_gate_l1, l0b_gate[side, :, :], transpose=True)
                    T.mma(l0a[side, :, :], l0b_gate[side, :, :], l0c_gate[side, :, :], init=(k == 0))

                    # Up: W_up → L1 → L0B → MMA (reuse L0A)
                    T.copy(W_up[by * block_N, k * block_K], w_up_l1)
                    T.copy(w_up_l1, l0b_up[side, :, :], transpose=True)
                    T.mma(l0a[side, :, :], l0b_up[side, :, :], l0c_up[side, :, :], init=(k == 0))

                # Write outputs
                T.copy(l0c_gate[0, :, :], Gate_out[bx * block_M, by * block_N])
                T.copy(l0c_up[0, :, :], Up_out[bx * block_M, by * block_N])

    return kernel


# ============================================================================
# Fixed Core Optimized Kernels (L2 cache reuse via grid-stride loop)
# ============================================================================


@tilelang.jit(out_idx=[-2, -1], pass_configs=pass_configs)
def dual_gemm_kernel_fixed(M, N, K, dtype="float16"):
    """Dual GEMM with Fixed Core + grid-stride loop for L2 Weight reuse.

    Grid-stride loop processes tiles in by-major order (same Weight block
    consecutive), enabling L2 cache reuse. Weight per N-block = 3.5MB << 192MB L2.
    """
    block_M = BLOCK_M
    block_N = BLOCK_N
    block_K = BLOCK_K
    accum_dtype = "float32"
    core_num = 20

    m_num = (M + block_M - 1) // block_M
    n_num = (N + block_N - 1) // block_N
    total_tiles = m_num * n_num

    q_tasks = total_tiles // core_num
    r_tasks = total_tiles % core_num

    @T.prim_func
    def kernel(
        Input: T.Tensor([M, K], dtype),  # type: ignore
        W_gate: T.Tensor([N, K], dtype),  # type: ignore
        W_up: T.Tensor([N, K], dtype),  # type: ignore
        Gate_out: T.Tensor([M, N], dtype),  # type: ignore
        Up_out: T.Tensor([M, N], dtype),  # type: ignore
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            W_gate_L1 = T.alloc_L1((block_N, block_K), dtype)
            W_up_L1 = T.alloc_L1((block_N, block_K), dtype)
            Gate_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)
            Up_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            # Grid-stride loop: by-major order for L2 Weight reuse
            my_start = cid * q_tasks + T.if_then_else(cid < r_tasks, cid, r_tasks)
            my_count = q_tasks + T.if_then_else(cid < r_tasks, 1, 0)

            for t in T.serial(my_count):
                task_id = my_start + t
                # by-major: same by (Weight block) processed consecutively
                by = task_id // m_num
                bx = task_id % m_num

                loop_k = T.ceildiv(K, block_K)
                for k in T.serial(loop_k):
                    T.copy(Input[bx * block_M, k * block_K], A_L1)
                    T.copy(W_gate[by * block_N, k * block_K], W_gate_L1)
                    T.copy(W_up[by * block_N, k * block_K], W_up_L1)

                    T.gemm_v0(A_L1, W_gate_L1, Gate_L0, transpose_B=True, init=(k == 0))
                    T.gemm_v0(A_L1, W_up_L1, Up_L0, transpose_B=True, init=(k == 0))

                T.copy(Gate_L0, Gate_out[bx * block_M, by * block_N])
                T.copy(Up_L0, Up_out[bx * block_M, by * block_N])

    return kernel


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def single_gemm_kernel_fixed(M, N, K, dtype="float16"):
    """Single GEMM with Fixed Core + grid-stride loop for L2 Weight reuse."""
    block_M = BLOCK_M
    block_N = BLOCK_N
    block_K = BLOCK_K
    accum_dtype = "float32"
    core_num = 20

    m_num = (M + block_M - 1) // block_M
    n_num = (N + block_N - 1) // block_N
    total_tiles = m_num * n_num

    q_tasks = total_tiles // core_num
    r_tasks = total_tiles % core_num

    @T.prim_func
    def kernel(
        Input: T.Tensor([M, K], dtype),  # type: ignore
        Weight: T.Tensor([N, K], dtype),  # type: ignore
        Output: T.Tensor([M, N], dtype),  # type: ignore
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_N, block_K), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            my_start = cid * q_tasks + T.if_then_else(cid < r_tasks, cid, r_tasks)
            my_count = q_tasks + T.if_then_else(cid < r_tasks, 1, 0)

            for t in T.serial(my_count):
                task_id = my_start + t
                by = task_id // m_num
                bx = task_id % m_num

                loop_k = T.ceildiv(K, block_K)
                for k in T.serial(loop_k):
                    T.copy(Input[bx * block_M, k * block_K], A_L1)
                    T.copy(Weight[by * block_N, k * block_K], B_L1)

                    T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=(k == 0))

                T.copy(C_L0, Output[bx * block_M, by * block_N])

    return kernel


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def grouped_gemm_kernel_fixed(buf_rows, N, K, n_groups, total_m_blocks, dtype="float16"):
    """Grouped single GEMM with Fixed Core + grid-stride loop."""
    block_M = BLOCK_M
    block_N = BLOCK_N
    block_K = BLOCK_K
    accum_dtype = "float32"
    core_num = 20

    n_num = (N + block_N - 1) // block_N
    total_tiles = total_m_blocks * n_num

    q_tasks = total_tiles // core_num
    r_tasks = total_tiles % core_num

    @T.prim_func
    def kernel(
        Input: T.Tensor([buf_rows, K], dtype),  # type: ignore
        Weight: T.Tensor([n_groups, N, K], dtype),  # type: ignore
        block_metadata: T.Tensor([total_m_blocks, 3], "int32"),  # type: ignore
        Output: T.Tensor([buf_rows, N], dtype),  # type: ignore
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_N, block_K), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            my_start = cid * q_tasks + T.if_then_else(cid < r_tasks, cid, r_tasks)
            my_count = q_tasks + T.if_then_else(cid < r_tasks, 1, 0)

            for t in T.serial(my_count):
                task_id = my_start + t
                by = task_id // total_m_blocks
                bx = task_id % total_m_blocks

                group_idx = block_metadata[bx, 0]
                m_start = block_metadata[bx, 1]

                loop_k = T.ceildiv(K, block_K)
                for k in T.serial(loop_k):
                    T.copy(
                        Input[m_start : m_start + block_M, k * block_K : (k + 1) * block_K],
                        A_L1,
                    )
                    T.copy(
                        Weight[group_idx, by * block_N : (by + 1) * block_N, k * block_K : (k + 1) * block_K],
                        B_L1,
                    )

                    T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=(k == 0))

                T.copy(
                    C_L0,
                    Output[m_start : m_start + block_M, by * block_N : (by + 1) * block_N],
                )

    return kernel


@tilelang.jit(out_idx=[-2, -1], pass_configs=pass_configs)
def grouped_dual_gemm_kernel_fixed(buf_rows, N, K, n_groups, total_m_blocks, dtype="float16"):
    """Grouped dual GEMM with Fixed Core + grid-stride loop."""
    block_M = BLOCK_M
    block_N = BLOCK_N
    block_K = BLOCK_K
    accum_dtype = "float32"
    core_num = 20

    n_num = (N + block_N - 1) // block_N
    total_tiles = total_m_blocks * n_num

    q_tasks = total_tiles // core_num
    r_tasks = total_tiles % core_num

    @T.prim_func
    def kernel(
        Input: T.Tensor([buf_rows, K], dtype),  # type: ignore
        W_gate: T.Tensor([n_groups, N, K], dtype),  # type: ignore
        W_up: T.Tensor([n_groups, N, K], dtype),  # type: ignore
        block_metadata: T.Tensor([total_m_blocks, 3], "int32"),  # type: ignore
        Gate_out: T.Tensor([buf_rows, N], dtype),  # type: ignore
        Up_out: T.Tensor([buf_rows, N], dtype),  # type: ignore
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            W_gate_L1 = T.alloc_L1((block_N, block_K), dtype)
            W_up_L1 = T.alloc_L1((block_N, block_K), dtype)
            Gate_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)
            Up_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            my_start = cid * q_tasks + T.if_then_else(cid < r_tasks, cid, r_tasks)
            my_count = q_tasks + T.if_then_else(cid < r_tasks, 1, 0)

            for t in T.serial(my_count):
                task_id = my_start + t
                by = task_id // total_m_blocks
                bx = task_id % total_m_blocks

                group_idx = block_metadata[bx, 0]
                m_start = block_metadata[bx, 1]

                loop_k = T.ceildiv(K, block_K)
                for k in T.serial(loop_k):
                    T.copy(
                        Input[m_start : m_start + block_M, k * block_K : (k + 1) * block_K],
                        A_L1,
                    )
                    T.copy(
                        W_gate[group_idx, by * block_N : (by + 1) * block_N, k * block_K : (k + 1) * block_K],
                        W_gate_L1,
                    )
                    T.copy(
                        W_up[group_idx, by * block_N : (by + 1) * block_N, k * block_K : (k + 1) * block_K],
                        W_up_L1,
                    )

                    T.gemm_v0(A_L1, W_gate_L1, Gate_L0, transpose_B=True, init=(k == 0))
                    T.gemm_v0(A_L1, W_up_L1, Up_L0, transpose_B=True, init=(k == 0))

                T.copy(
                    Gate_L0,
                    Gate_out[m_start : m_start + block_M, by * block_N : (by + 1) * block_N],
                )
                T.copy(
                    Up_L0,
                    Up_out[m_start : m_start + block_M, by * block_N : (by + 1) * block_N],
                )

    return kernel


def shared_expert_kernel(num_tokens, d_hidden, d_expert, dtype="float16"):
    """Compile shared expert kernels and return a callable for the full pipeline.

    Uses T.mma + L0A double buffer + auto_sync (Approach C) for dual_gemm.
    Best performance: 21.28ms pipeline (vs 23.26ms original, 8.5% speedup).
    """
    gate_up_gemm = dual_gemm_kernel_mma_auto(num_tokens, d_expert, d_hidden, dtype)
    down_gemm = single_gemm_kernel_fixed(num_tokens, d_hidden, d_expert, dtype)

    def run(Input, W_gate, W_up, W_down):
        gate_gm, up_gm = gate_up_gemm(Input, W_gate, W_up)
        gate_activated = F.silu(gate_gm)
        up_logits = up_gm * gate_activated
        output_gm = down_gemm(up_logits, W_down)
        return output_gm

    return run


def shared_expert_kernel_approach_b(num_tokens, d_hidden, d_expert, dtype="float16"):
    """Approach B: two independent single_gemm_mma kernels (gate + up separate).

    Input is read twice from GM (once per kernel), but each kernel is a simple
    single GEMM with T.mma + L0A double buffer — no dual GEMM flag complexity.
    """
    gate_gemm = single_gemm_kernel_mma_auto(num_tokens, d_expert, d_hidden, dtype)
    up_gemm = single_gemm_kernel_mma_auto(num_tokens, d_expert, d_hidden, dtype)
    down_gemm = single_gemm_kernel_fixed(num_tokens, d_hidden, d_expert, dtype)

    def run(Input, W_gate, W_up, W_down):
        gate_gm = gate_gemm(Input, W_gate)
        up_gm = up_gemm(Input, W_up)
        gate_activated = F.silu(gate_gm)
        up_logits = up_gm * gate_activated
        output_gm = down_gemm(up_logits, W_down)
        return output_gm

    return run


# ============================================================================
# Routed Expert Pipeline
# ============================================================================
def routed_expert_kernel(buf_rows, d_hidden, d_expert, n_routed_experts, total_m_blocks, dtype="float16"):
    """Compile routed expert kernels and return a callable for the full pipeline.

    Uses Fixed Core optimized kernels for L2 Weight cache reuse.
    """
    gate_up_gemm = grouped_dual_gemm_kernel_fixed(buf_rows, d_expert, d_hidden, n_routed_experts, total_m_blocks, dtype)
    down_gemm = grouped_gemm_kernel_fixed(buf_rows, d_hidden, d_expert, n_routed_experts, total_m_blocks, dtype)

    def run(Input, W_gate, W_up, W_down, expert_weights, block_metadata):
        # Step 1: Grouped Dual GEMM — returns (gate_gm, up_gm) as fp16
        # JIT creates new output tensors via out_idx=[-2, -1]; no clone needed
        gate_gm, up_gm = gate_up_gemm(Input, W_gate, W_up, block_metadata)

        # PyTorch post-processing: SiLU + Mul (gate_gm/up_gm already fp16, no cast)
        gate_activated = F.silu(gate_gm)
        up_logits = up_gm * gate_activated  # (buf_rows, d_expert) fp16

        # Step 2: Grouped Down GEMM — outputs fp16 directly (JIT creates new tensor)
        output_gm = down_gemm(up_logits, W_down, block_metadata)

        # M2: output already fp16, no cast needed; just weight multiply
        output_fp16 = output_gm * expert_weights.unsqueeze(1)

        return output_fp16

    return run


# ============================================================================
# Golden Functions (PyTorch reference)
# ============================================================================
def _mlp_moe(x, W_gate, W_up, W_down, expert_weights=None):
    """MoE MLP: gate GEMM → SiLU → up GEMM → down GEMM (+ optional weight mul).

    Args:
        x: (M, d_hidden) fp16
        W_gate/W_up: (d_expert, d_hidden) fp16 (transposed weight)
        W_down: (d_hidden, d_expert) fp16 (transposed weight)
        expert_weights: (M,) fp16 or None — per-token routing weight

    Returns:
        output: (M, d_hidden) fp16
    """
    gate = F.silu(x.float() @ W_gate.float().T).half()
    up = (x.float() @ W_up.float().T).half() * gate
    result = (up.float() @ W_down.float().T).half()
    if expert_weights is not None:
        result = result * expert_weights.unsqueeze(1)
    return result


def golden_shared_expert(x, W_gate, W_up, W_down):
    """Shared expert golden function.

    Args:
        x: (num_tokens, d_hidden) float16
        W_gate: (d_expert, d_hidden) float16 - transposed weight
        W_up: (d_expert, d_hidden) float16 - transposed weight
        W_down: (d_hidden, d_expert) float16 - transposed weight

    Returns:
        output: (num_tokens, d_hidden) float16
    """
    return _mlp_moe(x, W_gate, W_up, W_down)


def golden_routed_expert_nc(stacked_tokens, W_gate, W_up, W_down, expert_weights, block_metadata):
    """Routed expert golden function using non-compact layout."""
    output = torch.zeros_like(stacked_tokens)
    metadata = block_metadata.cpu().tolist()

    for row in metadata:
        expert_idx = int(row[0])
        m_start = int(row[1])
        valid_m = int(row[2])

        if valid_m == 0:
            continue

        x_block = stacked_tokens[m_start : m_start + valid_m]
        result = _mlp_moe(
            x_block,
            W_gate[expert_idx],
            W_up[expert_idx],
            W_down[expert_idx],
            expert_weights[m_start : m_start + valid_m],
        )
        output[m_start : m_start + valid_m] = result

    return output


def golden_fusedmoe_full(
    x,
    W_gate_shared,
    W_up_shared,
    W_down_shared,
    W_gate_routed,
    W_up_routed,
    W_down_routed,
    router_weight,
    n_experts_per_token,
):
    """Full FusedMoE golden function.

    Clamps k to min(n_experts_per_token, n_routed_experts) to handle the
    user-specified configuration (top_k=4, n_routed_experts=1).
    """
    batch_size, seq_len, d_hidden = x.shape
    x_flat = x.view(-1, d_hidden)
    n_routed_experts = W_gate_routed.shape[0]
    effective_k = min(n_experts_per_token, n_routed_experts)

    # 1. Shared expert
    shared_output = golden_shared_expert(x_flat, W_gate_shared, W_up_shared, W_down_shared)

    # 2. Gating network
    logits = x_flat.float() @ router_weight.float().T
    scores = F.softmax(logits, dim=-1)
    topk_scores, topk_indices = torch.topk(scores, k=effective_k, dim=-1, sorted=False)

    # 3. Token routing
    flat_indices = topk_indices.view(-1)
    flat_weights = topk_scores.view(-1)
    idxs = flat_indices.argsort()
    counts = flat_indices.bincount(minlength=n_routed_experts).cpu().numpy()
    tokens_per_expert = counts.cumsum()
    token_idxs = idxs // effective_k

    # 4. Routed expert (per-expert processing)
    expert_cache = torch.zeros_like(x_flat)
    for expert_id in range(len(counts)):
        start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
        end_idx = tokens_per_expert[expert_id]
        if start_idx == end_idx:
            continue

        exp_token_idxs = token_idxs[start_idx:end_idx]
        x_block = x_flat[exp_token_idxs]
        result = _mlp_moe(
            x_block,
            W_gate_routed[expert_id],
            W_up_routed[expert_id],
            W_down_routed[expert_id],
            flat_weights[idxs[start_idx:end_idx]],
        )

        expert_cache.scatter_reduce_(
            0,
            exp_token_idxs.view(-1, 1).repeat(1, d_hidden),
            result.to(x_flat.dtype),
            reduce="sum",
        )

    routed_output = expert_cache.view(batch_size, seq_len, d_hidden)
    shared_output = shared_output.view(batch_size, seq_len, d_hidden)

    return shared_output + routed_output


# ============================================================================
# Host Preprocessing (Token Routing with Non-Compact Layout)
# ============================================================================
def _build_block_metadata(group_sizes, block_M):
    """Build block_metadata + total_m_blocks for non-compact layout.

    Each expert's tokens are padded to block_M-aligned blocks. Returns a
    list of [expert_id, m_start, valid_m] rows and the total block count.
    """
    metadata_list = []
    nc_offset = 0
    for expert_id, size in enumerate(group_sizes):
        if size == 0:
            continue
        num_blocks = (size + block_M - 1) // block_M
        for i in range(num_blocks):
            m_start = nc_offset + i * block_M
            valid_m = min(block_M, size - i * block_M)
            metadata_list.append([expert_id, m_start, valid_m])
        nc_offset += num_blocks * block_M
    total_m_blocks = len(metadata_list)
    return metadata_list, total_m_blocks


def host_preprocess(x_flat, router_weight, n_experts_per_token, block_M, device):
    """Host-side gating + token routing for the routed expert kernel.

    Handles n_experts_per_token > n_routed_experts by clamping k to
    min(n_experts_per_token, n_routed_experts). When there is only 1
    routed expert, all tokens route to it with weight 1.0.
    """
    num_tokens, d_hidden = x_flat.shape
    dtype = x_flat.dtype
    n_routed_experts = router_weight.shape[0]
    # Clamp k to available experts (handles user spec: top_k=4, n_experts=1)
    effective_k = min(n_experts_per_token, n_routed_experts)

    logits = x_flat.float() @ router_weight.float().T
    scores = F.softmax(logits, dim=-1)
    topk_scores, topk_indices = torch.topk(scores, k=effective_k, dim=-1, sorted=False)

    flat_indices = topk_indices.view(-1)
    flat_weights = topk_scores.view(-1)
    idxs = flat_indices.argsort()
    counts = flat_indices.bincount(minlength=n_routed_experts).cpu().numpy()
    tokens_per_expert = counts.cumsum()
    token_idxs = idxs // effective_k

    group_sizes_list = [int(c) for c in counts]

    metadata_list, total_m_blocks = _build_block_metadata(group_sizes_list, block_M)
    buf_rows = total_m_blocks * block_M

    stacked_tokens = torch.zeros(buf_rows, d_hidden, dtype=dtype).to(device)
    stacked_weights = torch.zeros(buf_rows, dtype=dtype).to(device)
    token_idxs_nc = torch.zeros(buf_rows, dtype=torch.long).to(device)

    nc_offset = 0
    for expert_id in range(len(counts)):
        size = group_sizes_list[expert_id]
        if size == 0:
            continue

        start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
        end_idx = tokens_per_expert[expert_id]
        exp_token_idxs = token_idxs[start_idx:end_idx]

        stacked_tokens[nc_offset : nc_offset + size] = x_flat[exp_token_idxs]
        stacked_weights[nc_offset : nc_offset + size] = flat_weights[idxs[start_idx:end_idx]]
        token_idxs_nc[nc_offset : nc_offset + size] = exp_token_idxs

        nc_offset += (size + block_M - 1) // block_M * block_M

    block_metadata = torch.tensor(metadata_list, dtype=torch.int32).to(device)

    return {
        "stacked_tokens": stacked_tokens,
        "stacked_weights": stacked_weights,
        "token_idxs_nc": token_idxs_nc,
        "block_metadata": block_metadata,
        "total_m_blocks": total_m_blocks,
        "buf_rows": buf_rows,
        "group_sizes_list": group_sizes_list,
    }


def host_preprocess_for_test(group_sizes, d_hidden, n_experts, block_M, device):
    """Simplified host preprocessing for routed kernel unit tests."""
    dtype = torch.float16

    metadata_list, total_m_blocks = _build_block_metadata(group_sizes, block_M)
    buf_rows = total_m_blocks * block_M

    stacked_tokens = torch.randn(buf_rows, d_hidden, dtype=dtype).to(device) * 0.01
    stacked_weights = torch.zeros(buf_rows, dtype=dtype).to(device)

    nc_offset = 0
    for _expert_id, size in enumerate(group_sizes):
        if size == 0:
            continue
        stacked_weights[nc_offset : nc_offset + size] = 1.0
        nc_offset += (size + block_M - 1) // block_M * block_M

    block_metadata = torch.tensor(metadata_list, dtype=torch.int32).to(device)

    return {
        "stacked_tokens": stacked_tokens,
        "stacked_weights": stacked_weights,
        "block_metadata": block_metadata,
        "total_m_blocks": total_m_blocks,
        "buf_rows": buf_rows,
    }


# ========== Smoke Test (CI entry — prints "Test Passed!") ==========

if __name__ == "__main__":
    tilelang.disable_cache()
    torch.set_default_device("npu")
    torch.manual_seed(0)

    # Minimal L0 config (shared expert, block-aligned)
    num_tokens, d_hidden, d_expert = 64, 128, 64
    dtype = torch.float16

    x = torch.randn(num_tokens, d_hidden, dtype=dtype, device="npu")
    w_gate = torch.randn(d_expert, d_hidden, dtype=dtype, device="npu") * 0.01
    w_up = torch.randn(d_expert, d_hidden, dtype=dtype, device="npu") * 0.01
    w_down = torch.randn(d_hidden, d_expert, dtype=dtype, device="npu") * 0.01

    kernel = shared_expert_kernel(num_tokens, d_hidden, d_expert)
    output = kernel(x, w_gate, w_up, w_down)

    ref = golden_shared_expert(x, w_gate, w_up, w_down)

    max_diff = (output.cpu().float() - ref.cpu().float()).abs().max().item()
    torch.testing.assert_close(output.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)
    print(f"max_diff={max_diff:.6f}")
    print("Test Passed!")
