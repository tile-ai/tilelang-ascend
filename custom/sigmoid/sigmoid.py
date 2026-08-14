"""Sigmoid activation kernel: y = 1 / (1 + exp(-x)).

Element-wise activation using Developer mode with T.alloc_shared (auto-mapped to UB)
and T.tile.sigmoid buffer-level SIMD primitive.

Uses T.tile.sigmoid (one-step primitive) instead of the 5-step decomposition
(fill/sub/exp/add/reciprocal) because the latter's T.tile.exp and T.tile.reciprocal
internally compute in float16 regardless of buffer dtype, causing precision failures
for float32. T.tile.sigmoid correctly preserves the buffer dtype throughout.

Reference: examples/activation/sigmoidv2.py (T.tile.sigmoid usage).
Design: custom/sigmoid/DESIGN.md §3.2 alternative path.

Perf tuning iter1:
- [#1] Closed TL_ASCEND_AUTO_CV_COMBINE for this pure Vector op. The auto-CV-combine
  pass was emitting KERNEL_TYPE_MIX_AIC_1_2 with all compute inside
  `if ASCEND_IS_AIV { ... }`, leaving the AIC core idle but still paying its
  launch + L0A/L0B/L1/L0C buffer init cost.
- [#3] Fixed Core mode: launch min(block_num, CORE_NUM) cores instead of block_num,
  each core processes ceildiv(block_num, launch_cores) tiles via T.serial.
  This eliminates per-block launch overhead (was 512 launches for (1024,8192)
  fp16, now 24). Buffers are hoisted out of the loop and reused across tiles
  (AUTO_SYNC ensures each tile's MTE2/V/MTE3 completes before the next).
  Reference: examples/linear_attention_and_rnn/linear_attention_causal.py
  (T.Kernel(core_num) + T.serial(ceildiv(B*H, core_num)) + if pid < B*H pattern).
  See .agents/skills/tilelang-perf-optimization/references/performance-antipatterns.md
  "launch core 数需要重点关注" 关注项 A.
"""

import tilelang
from tilelang import language as T

# ========== Operator implementation ==========
# AUTO_CV_COMBINE intentionally OFF: sigmoid is pure Vector (element-wise),
# the pass was adding an idle AIC core. AUTO_SYNC + MEMORY_PLANNING kept on
# (Developer mode auto-sync + UB memory planning).
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# Ascend A2/A3 physical AI Core count. Fixed Core mode launches at most this
# many blocks; each core processes ceildiv(block_num, launch_cores) tiles.
CORE_NUM = 24


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def sigmoid(M, N, block_M, block_N, dtype="float16"):
    """Sigmoid kernel: y = 1 / (1 + exp(-x)).

    Args:
        M, N: tensor shape (rows, cols)
        block_M, block_N: tile size per block
        dtype: "float16" or "float32"

    Returns:
        prim_func mapping A (M, N) -> B (M, N)
    """
    # Host-side Python int computation (JIT compile-time constants)
    m_num = (M + block_M - 1) // block_M
    n_num = (N + block_N - 1) // block_N
    block_num = m_num * n_num
    # Fixed Core: launch min(block_num, CORE_NUM) cores; each handles ceildiv of work.
    # Pad block_num up to a multiple of launch_cores so no tail-block guard is needed
    # (T.copy handles out-of-range slices via dynamic shape slicing, see DESIGN.md §5.4).
    launch_cores = min(block_num, CORE_NUM)
    single_core_load = (block_num + launch_cores - 1) // launch_cores

    VEC_NUM = 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):  # type: ignore
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            # Striped work distribution: core `cid` handles tiles
            # cid, cid+launch_cores, cid+2*launch_cores, ...
            for block_idx in T.serial(single_core_load):
                logical_cid = block_idx * launch_cores + cid
                bx = logical_cid // n_num
                by = logical_cid % n_num

                a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
                b_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)

                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.tile.sigmoid(b_ub, a_ub)
                T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
