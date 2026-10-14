"""SituGlu: SiTU gated linear unit activation.

Merges the input x (2D view (M, N), N even) into gate/up halves along the last
axis and computes:

    situ_a = beta * tanh(gate / beta) * sigmoid(gate)
    up'    = linear_beta * tanh(up / linear_beta)   (linear_beta > 0, else up)
    y      = situ_a * up'

Output shape is (M, N // 2). tanh is composed via the numerically stable
identity tanh(t) = 2 * sigmoid(2t) - 1 (no T.tile.tanh primitive).
Reference: CANN ops-nn activation/situ_glu (aclnnSituGlu).
See benchmark.md / benchmark_zh.md for architecture and measured performance.

Kernel design (per-block flag-pipelined double buffering,
examples/elementwise/elementwise_add_pipeline.py pattern): each grid block
owns a (block_M, block_N) output tile and walks it in vec_proc row chunks
(sub_M rows each, split across the two vector sub-blocks); the next chunk's
MTE2 load overlaps the current chunk's vector compute and the previous chunk's
MTE3 store. Synchronization uses manual hardware-event flags, so no
pass_configs are enabled (auto-sync passes serialize the loop and interfere
with the flags). Tail blocks are handled by the frontend auto-clamp of
full-tile T.copy (no host-side padding).

M / N accept either None (symbolic, one compile serves any shape) or concrete
ints (static specialization: shorter dispatch and constant grid arithmetic).
The mixed-dtype and fp32 paths are two self-contained prim_funcs (buffer
allocations are not shared across if-branches in the script parser).
"""

import tilelang
from tilelang import language as T

VEC_NUM = 2
STAGES = 2


@tilelang.jit(out_idx=[1])
def situ_glu(block_M, block_N, beta, linear_beta, activate_left, dtype="float", sub_M=32, M=None, N=None):
    """SituGlu kernel (flag-pipelined double buffering).

    Args:
        block_M: rows per grid block (multiple of sub_M).
        block_N: column tile size over the halved output space.
        beta: SiTU gate scaling coefficient (non-zero).
        linear_beta: bounded-tanh coefficient for the up path; <= 0 bypasses it.
        activate_left: True -> gate is the first half, up the second half.
        dtype: "float"/"float32", "float16" or "bfloat16" (compute in fp32).
        sub_M: rows per pipelined chunk (multiple of VEC_NUM=2); the block
            pipeline depth is block_M // sub_M.
        M / N: None -> symbolic dims (one compile for any shape); concrete
            ints -> static specialization (constant grid, lighter dispatch).
    """
    if M is None:
        M = T.symbolic("M")
    if N is None:
        N = T.symbolic("N")
    assert block_M % sub_M == 0, "block_M must be a multiple of sub_M"
    assert sub_M % VEC_NUM == 0, "sub_M must be a multiple of VEC_NUM"

    need_cast = dtype not in ("float", "float32")
    acc_dtype = "float32"
    out_cast_mode = "CAST_RINT" if dtype == "bfloat16" else "CAST_NONE"

    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N // 2, block_N)
    vec_proc = block_M // sub_M
    rows_per_vec = sub_M // VEC_NUM
    elem_num = rows_per_vec * block_N
    gate_off = 0 if activate_left else N // 2
    up_off = N // 2 if activate_left else 0

    if need_cast:

        @T.prim_func
        def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N // 2), dtype)):
            with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
                bx = cid // n_num
                by = cid % n_num
                n0 = by * block_N

                # staged raw I/O for MTE/V overlap; fp32 compute buffers are
                # single (vector ops serialize on the V pipe anyway)
                raw_g = T.alloc_ub((STAGES, rows_per_vec, block_N), dtype)
                raw_u = T.alloc_ub((STAGES, rows_per_vec, block_N), dtype)
                raw_y = T.alloc_ub((STAGES, rows_per_vec, block_N), dtype)
                gate_f = T.alloc_ub((rows_per_vec, block_N), acc_dtype)
                up_f = T.alloc_ub((rows_per_vec, block_N), acc_dtype)
                y_f = T.alloc_ub((rows_per_vec, block_N), acc_dtype)
                t_ub = T.alloc_ub((rows_per_vec, block_N), acc_dtype)
                s_ub = T.alloc_ub((rows_per_vec, block_N), acc_dtype)

                with T.Scope("V"):
                    T.set_flag("mte3", "mte2", 0)
                    T.set_flag("mte3", "mte2", 1)

                    row0 = bx * block_M + vid * rows_per_vec
                    T.wait_flag("mte3", "mte2", 0)
                    T.copy(A[row0, gate_off + n0], raw_g[0, :, :])
                    T.copy(A[row0, up_off + n0], raw_u[0, :, :])
                    T.set_flag("mte2", "v", 0)

                    for mm in T.serial(vec_proc):
                        cur = mm % STAGES
                        nxt = (mm + 1) % STAGES
                        row = bx * block_M + mm * sub_M + vid * rows_per_vec

                        if mm < vec_proc - 1:
                            T.wait_flag("mte3", "mte2", nxt)
                            T.copy(A[row + sub_M, gate_off + n0], raw_g[nxt, :, :])
                            T.copy(A[row + sub_M, up_off + n0], raw_u[nxt, :, :])
                            T.set_flag("mte2", "v", nxt)

                        T.wait_flag("mte2", "v", cur)
                        T.tile.cast(gate_f, raw_g[cur, :, :], "CAST_NONE", elem_num)
                        T.tile.cast(up_f, raw_u[cur, :, :], "CAST_NONE", elem_num)

                        # gate path: situ_a = beta * tanh(gate / beta) * sigmoid(gate)
                        T.tile.mul(t_ub, gate_f, 2.0 / beta)
                        T.tile.sigmoid(t_ub, t_ub)
                        T.tile.mul(t_ub, t_ub, 2.0)
                        T.tile.sub(t_ub, t_ub, 1.0)
                        T.tile.sigmoid(s_ub, gate_f)
                        T.tile.mul(t_ub, t_ub, s_ub)
                        if beta != 1.0:
                            T.tile.mul(t_ub, t_ub, beta)

                        # up path
                        if linear_beta > 0:
                            T.tile.mul(s_ub, up_f, 2.0 / linear_beta)
                            T.tile.sigmoid(s_ub, s_ub)
                            T.tile.mul(s_ub, s_ub, 2.0)
                            T.tile.sub(s_ub, s_ub, 1.0)
                            T.tile.mul(s_ub, s_ub, linear_beta)
                            T.tile.mul(y_f, t_ub, s_ub)
                        else:
                            T.tile.mul(y_f, t_ub, up_f)

                        T.tile.cast(raw_y[cur, :, :], y_f, out_cast_mode, elem_num)
                        T.set_flag("v", "mte3", cur)
                        T.wait_flag("v", "mte3", cur)
                        T.copy(raw_y[cur, :, :], B[row, n0])
                        T.set_flag("mte3", "mte2", cur)

                    T.wait_flag("mte3", "mte2", 0)
                    T.wait_flag("mte3", "mte2", 1)
    else:

        @T.prim_func
        def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N // 2), dtype)):
            with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
                bx = cid // n_num
                by = cid % n_num
                n0 = by * block_N

                gate_io = T.alloc_ub((STAGES, rows_per_vec, block_N), acc_dtype)
                up_io = T.alloc_ub((STAGES, rows_per_vec, block_N), acc_dtype)
                y_io = T.alloc_ub((STAGES, rows_per_vec, block_N), acc_dtype)
                t_ub = T.alloc_ub((rows_per_vec, block_N), acc_dtype)
                s_ub = T.alloc_ub((rows_per_vec, block_N), acc_dtype)

                with T.Scope("V"):
                    T.set_flag("mte3", "mte2", 0)
                    T.set_flag("mte3", "mte2", 1)

                    row0 = bx * block_M + vid * rows_per_vec
                    T.wait_flag("mte3", "mte2", 0)
                    T.copy(A[row0, gate_off + n0], gate_io[0, :, :])
                    T.copy(A[row0, up_off + n0], up_io[0, :, :])
                    T.set_flag("mte2", "v", 0)

                    for mm in T.serial(vec_proc):
                        cur = mm % STAGES
                        nxt = (mm + 1) % STAGES
                        row = bx * block_M + mm * sub_M + vid * rows_per_vec

                        if mm < vec_proc - 1:
                            T.wait_flag("mte3", "mte2", nxt)
                            T.copy(A[row + sub_M, gate_off + n0], gate_io[nxt, :, :])
                            T.copy(A[row + sub_M, up_off + n0], up_io[nxt, :, :])
                            T.set_flag("mte2", "v", nxt)

                        T.wait_flag("mte2", "v", cur)
                        # gate path: situ_a = beta * tanh(gate / beta) * sigmoid(gate)
                        T.tile.mul(t_ub, gate_io[cur, :, :], 2.0 / beta)
                        T.tile.sigmoid(t_ub, t_ub)
                        T.tile.mul(t_ub, t_ub, 2.0)
                        T.tile.sub(t_ub, t_ub, 1.0)
                        T.tile.sigmoid(s_ub, gate_io[cur, :, :])
                        T.tile.mul(t_ub, t_ub, s_ub)
                        if beta != 1.0:
                            T.tile.mul(t_ub, t_ub, beta)

                        # up path
                        if linear_beta > 0:
                            T.tile.mul(s_ub, up_io[cur, :, :], 2.0 / linear_beta)
                            T.tile.sigmoid(s_ub, s_ub)
                            T.tile.mul(s_ub, s_ub, 2.0)
                            T.tile.sub(s_ub, s_ub, 1.0)
                            T.tile.mul(s_ub, s_ub, linear_beta)
                            T.tile.mul(y_io[cur, :, :], t_ub, s_ub)
                        else:
                            T.tile.mul(y_io[cur, :, :], t_ub, up_io[cur, :, :])

                        T.set_flag("v", "mte3", cur)
                        T.wait_flag("v", "mte3", cur)
                        T.copy(y_io[cur, :, :], B[row, n0])
                        T.set_flag("mte3", "mte2", cur)

                    T.wait_flag("mte3", "mte2", 0)
                    T.wait_flag("mte3", "mte2", 1)

    return main
