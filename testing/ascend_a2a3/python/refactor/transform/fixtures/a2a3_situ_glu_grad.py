# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

"""Complete TA 0deb8b93 SiTU factory; only the private T import is adapted.

The original JIT pass configurations are applied by the regression context.
"""

from tilelang.ascend.a2a3 import language as T


def situ_glu_grad_kernel(
    rows,
    w_in,
    h_in,
    block_m,
    block_n,
    dtype="bfloat16",
    beta=4.0,
    linear_beta=25.0,
    activate_left=True,
    tiles_per_block=8,
):
    """Build the SituGluGrad kernel.

    Kernel sees flattened rows: ``x`` -> [rows, cut * inner], ``grad_y`` ->
    [rows, h]. Gate data lives at column offset ``gate_off`` and up data at
    ``up_off``; the two offsets swap with ``activate_left``. The host adapter
    guarantees ``rows % block_m == 0``, ``h_in % block_n == 0`` and
    ``n_num % tiles_per_block == 0``.

    Each AI Core drives two Vector cores that split the row block between them;
    every block then walks a strip of ``tiles_per_block`` column tiles in a
    two-stage software pipeline (``T.Pipelined``), so the next tile's MTE2 load
    overlaps the current tile's vector math and the previous tile's MTE3 store.

    All math is computed in fp32 for low-precision dtypes; results are cast
    back with CAST_RINT. Compile-time specialization on (dtype, beta,
    linear_beta > 0, activate_left) keeps the linear_beta <= 0 branch free.
    """
    has_linear = linear_beta > 0.0
    assert rows % block_m == 0 and h_in % block_n == 0, "host adapter must pad to block multiples"
    vec_num = 2
    assert block_m % vec_num == 0
    rows_per_vec = block_m // vec_num
    m_num = rows // block_m
    n_num = h_in // block_n
    assert n_num % tiles_per_block == 0
    n_groups = n_num // tiles_per_block
    num_blocks = m_num * n_groups
    use_fp32 = dtype not in ("float16", "bfloat16")
    cal_dtype = "float32"
    gate_off = 0 if activate_left else h_in
    up_off = h_in if activate_left else 0

    @T.prim_func
    def main(
        X: T.Tensor((rows, w_in), dtype),
        dY: T.Tensor((rows, h_in), dtype),
        dX: T.Tensor((rows, w_in), dtype),
    ):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            bx = cid // n_groups
            bg = cid % n_groups
            # Each AI Core drives two Vector cores; split the row block between them.
            row0 = bx * block_m + vid * rows_per_vec

            one_ub = T.alloc_ub((rows_per_vec, block_n), cal_dtype)
            g_ub = T.alloc_ub((rows_per_vec, block_n), cal_dtype)
            u_ub = T.alloc_ub((rows_per_vec, block_n), cal_dtype)
            dy_ub = T.alloc_ub((rows_per_vec, block_n), cal_dtype)
            t_ub = T.alloc_ub((rows_per_vec, block_n), cal_dtype)
            s_ub = T.alloc_ub((rows_per_vec, block_n), cal_dtype)
            sa_ub = T.alloc_ub((rows_per_vec, block_n), cal_dtype)
            upp_ub = T.alloc_ub((rows_per_vec, block_n), cal_dtype)
            tu_ub = T.alloc_ub((rows_per_vec, block_n), cal_dtype)
            dtu_ub = T.alloc_ub((rows_per_vec, block_n), cal_dtype)
            tmp1_ub = T.alloc_ub((rows_per_vec, block_n), cal_dtype)
            tmp2_ub = T.alloc_ub((rows_per_vec, block_n), cal_dtype)
            gu_ub = T.alloc_ub((rows_per_vec, block_n), cal_dtype)
            # Raw staging buffers for the low-precision path; allocated
            # unconditionally so the pipelined loop body can reference them
            # (the fp32 path simply leaves them untouched).
            raw_g = T.alloc_ub((rows_per_vec, block_n), dtype)
            raw_u = T.alloc_ub((rows_per_vec, block_n), dtype)
            raw_dy = T.alloc_ub((rows_per_vec, block_n), dtype)
            raw_out = T.alloc_ub((rows_per_vec, block_n), dtype)

            T.tile.fill(one_ub, 1.0)

            for t in T.Pipelined(tiles_per_block, num_stages=2):
                col0 = (bg * tiles_per_block + t) * block_n

                if use_fp32:
                    T.copy(X[row0 : row0 + rows_per_vec, gate_off + col0 : gate_off + col0 + block_n], g_ub)
                    T.copy(X[row0 : row0 + rows_per_vec, up_off + col0 : up_off + col0 + block_n], u_ub)
                    T.copy(dY[row0 : row0 + rows_per_vec, col0 : col0 + block_n], dy_ub)
                else:
                    T.copy(X[row0 : row0 + rows_per_vec, gate_off + col0 : gate_off + col0 + block_n], raw_g)
                    T.copy(X[row0 : row0 + rows_per_vec, up_off + col0 : up_off + col0 + block_n], raw_u)
                    T.copy(dY[row0 : row0 + rows_per_vec, col0 : col0 + block_n], raw_dy)
                    T.tile.cast(g_ub, raw_g, "CAST_NONE", rows_per_vec * block_n)
                    T.tile.cast(u_ub, raw_u, "CAST_NONE", rows_per_vec * block_n)
                    T.tile.cast(dy_ub, raw_dy, "CAST_NONE", rows_per_vec * block_n)

                # t = tanh(g / beta) = 2 * sigmoid(2g / beta) - 1
                T.tile.mul(t_ub, g_ub, 2.0 / beta)
                T.tile.sigmoid(t_ub, t_ub)
                T.tile.add(t_ub, t_ub, t_ub)
                T.tile.add(t_ub, t_ub, -1.0)

                # s = sigmoid(g)
                T.tile.sigmoid(s_ub, g_ub)

                # situ_a = beta * t * s
                T.tile.mul(sa_ub, t_ub, s_ub)
                T.tile.mul(sa_ub, sa_ub, beta)

                # up' and d(up')/d(up), then grad_situ_a = grad_y * up' (reuse upp_ub)
                if has_linear:
                    # tu = tanh(u / linear_beta) = 2 * sigmoid(2u / linear_beta) - 1
                    T.tile.mul(tu_ub, u_ub, 2.0 / linear_beta)
                    T.tile.sigmoid(tu_ub, tu_ub)
                    T.tile.add(tu_ub, tu_ub, tu_ub)
                    T.tile.add(tu_ub, tu_ub, -1.0)
                    T.tile.mul(upp_ub, tu_ub, linear_beta)
                    T.tile.mul(dtu_ub, tu_ub, tu_ub)
                    T.tile.sub(dtu_ub, one_ub, dtu_ub)
                    T.tile.mul(upp_ub, dy_ub, upp_ub)
                else:
                    # up' = up and d(up')/d(up) = 1, so grad_situ_a = grad_y * u
                    # directly; the identity dtu folds into grad_up below.
                    T.tile.mul(upp_ub, dy_ub, u_ub)

                # grad_gate = grad_situ_a * (s * (1 - t^2) + situ_a * (1 - s))
                # (beta * t * s * (1 - s) == situ_a * (1 - s), so the bracket
                # costs one multiply by s plus one multiply by situ_a)
                T.tile.mul(tmp1_ub, t_ub, t_ub)
                T.tile.sub(tmp1_ub, one_ub, tmp1_ub)  # 1 - t^2
                T.tile.sub(tmp2_ub, one_ub, s_ub)  # 1 - s
                T.tile.mul(tmp1_ub, tmp1_ub, s_ub)  # s * (1 - t^2)
                T.tile.mul(tmp2_ub, tmp2_ub, sa_ub)  # situ_a * (1 - s)
                T.tile.add(tmp1_ub, tmp1_ub, tmp2_ub)
                T.tile.mul(upp_ub, upp_ub, tmp1_ub)  # grad_gate

                # grad_up = grad_y * situ_a * dtu   (sa_ub is consumed here).
                # For the identity branch (dtu == 1) the multiply is implicit.
                if has_linear:
                    T.tile.mul(sa_ub, sa_ub, dtu_ub)
                T.tile.mul(gu_ub, sa_ub, dy_ub)

                # Store: grad_gate -> gate half, grad_up -> up half.
                if use_fp32:
                    T.copy(upp_ub, dX[row0 : row0 + rows_per_vec, gate_off + col0 : gate_off + col0 + block_n])
                    T.copy(gu_ub, dX[row0 : row0 + rows_per_vec, up_off + col0 : up_off + col0 + block_n])
                else:
                    T.tile.cast(raw_out, upp_ub, "CAST_RINT", rows_per_vec * block_n)
                    T.copy(raw_out, dX[row0 : row0 + rows_per_vec, gate_off + col0 : gate_off + col0 + block_n])
                    T.tile.cast(raw_out, gu_ub, "CAST_RINT", rows_per_vec * block_n)
                    T.copy(raw_out, dX[row0 : row0 + rows_per_vec, up_off + col0 : up_off + col0 + block_n])

    return main
