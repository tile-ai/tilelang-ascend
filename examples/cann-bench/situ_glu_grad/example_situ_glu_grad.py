"""SiTU-GLU backward (SituGluGrad) operator for Ascend NPU using TileLang.

SiTU-GLU is the gated routed activation of Kimi K3's Stable LatentMoE:

    situ_a = beta * tanh(gate / beta) * sigmoid(gate)
    up'    = linear_beta * tanh(up / linear_beta)   (linear_beta > 0), else up
    y      = situ_a * up'

This file implements the backward operator ``situ_glu_grad`` (grad_x given
``grad_y`` and the forward input ``x``), matching the official CANN contract
(aclnnSituGluGrad / cann_ops_nn.situ_glu_grad):

    grad_gate = (grad_y * up') * sigmoid(gate)
                * ((1 - tanh^2(gate / beta)) + beta * tanh(gate / beta) * (1 - sigmoid(gate)))
    grad_up   = (grad_y * situ_a) * (1 - tanh^2(up / linear_beta))   (linear_beta > 0)
    grad_up   = (grad_y * situ_a)                                   (linear_beta <= 0)

Layout: ``x`` is split along ``dim`` into two equal halves (gate / up, with
``activate_left`` selecting which half is gate). The host adapter flattens the
tensor to ``[rows, cut * inner]``; because the two halves stay contiguous per
row, the kernel only ever needs a column split. Non-final ``dim`` works without
any stride handling. Shapes whose row count or half width is not a multiple of
the block size are zero-padded on the host to block multiples and trimmed back
(aligned shapes, including the Kimi K3 shape, pay nothing extra); all kernel
copies are therefore full static block copies.

Mode: Hybrid (Developer pass_configs + Ascend ``T.tile`` primitives and
``T.alloc_ub`` buffers).
"""

import tilelang
import tilelang.language as T
import torch

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_KERNEL_CACHE = {}

_SITU_GLU_GRAD_DTYPES = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}

# tiles_per_block candidates, tried in order; the first one that divides the
# column-tile count and still keeps enough AIV tasks in flight is used
# (a strip of 1 disables pipelining for awkward shapes).
_TILES_PER_BLOCK_CANDIDATES = (12, 8, 6, 4, 3, 2, 1)


# ========== Golden reference ==========
def golden_situ_glu_grad(x, grad_y, dim=-1, beta=4.0, linear_beta=25.0, activate_left=True):
    """PyTorch fp64 reference (cann-bench golden semantics)."""
    xd = x.double()
    gd = grad_y.double()
    d = dim if dim >= 0 else dim + x.dim()
    half = x.shape[d] // 2
    gate = xd.narrow(d, 0, half) if activate_left else xd.narrow(d, half, half)
    up = xd.narrow(d, half, half) if activate_left else xd.narrow(d, 0, half)

    t = torch.tanh(gate / beta)
    s = torch.sigmoid(gate)
    situ_a = beta * t * s
    if linear_beta > 0:
        tu = torch.tanh(up / linear_beta)
        up_prime = linear_beta * tu
        d_up = 1.0 - tu * tu
    else:
        up_prime = up
        d_up = torch.ones_like(up)

    grad_gate = (gd * up_prime) * s * ((1.0 - t * t) + beta * t * (1.0 - s))
    grad_up = (gd * situ_a) * d_up
    grad_x = torch.cat([grad_gate, grad_up], dim=d) if activate_left else torch.cat([grad_up, grad_gate], dim=d)
    return grad_x


# ========== Kernel ==========
@tilelang.jit(out_idx=[2], pass_configs=_PASS_CONFIGS)
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

                # up' and d(up')/d(up)
                if has_linear:
                    # tu = tanh(u / linear_beta) = 2 * sigmoid(2u / linear_beta) - 1
                    T.tile.mul(tu_ub, u_ub, 2.0 / linear_beta)
                    T.tile.sigmoid(tu_ub, tu_ub)
                    T.tile.add(tu_ub, tu_ub, tu_ub)
                    T.tile.add(tu_ub, tu_ub, -1.0)
                    T.tile.mul(upp_ub, tu_ub, linear_beta)
                    T.tile.mul(dtu_ub, tu_ub, tu_ub)
                    T.tile.sub(dtu_ub, one_ub, dtu_ub)
                else:
                    T.copy(u_ub, upp_ub)
                    T.copy(one_ub, dtu_ub)

                # grad_situ_a = grad_y * up'   (reuse upp_ub)
                T.tile.mul(upp_ub, dy_ub, upp_ub)

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

                # grad_up = grad_y * situ_a * dtu   (sa_ub is consumed here)
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


# ========== Host wrapper ==========
def _validate(x, grad_y, dim, beta):
    if x.dim() < 1:
        raise ValueError("x rank should be greater than 0")
    if dim < -x.dim() or dim > x.dim() - 1:
        raise ValueError(f"dim should be in [{-x.dim()}, {x.dim() - 1}], but got {dim}")
    actual_dim = dim if dim >= 0 else dim + x.dim()
    if x.shape[actual_dim] % 2 != 0:
        raise ValueError(f"x size at dim {actual_dim} should be even, but got {x.shape[actual_dim]}")
    if beta == 0.0:
        raise ValueError("beta must be non-zero (official op leaves beta==0 undefined; division by zero)")
    if x.dtype not in _SITU_GLU_GRAD_DTYPES:
        raise TypeError(f"x must be float32/float16/bfloat16, got {x.dtype}")
    if grad_y.dtype != x.dtype:
        raise TypeError(f"grad_y dtype {grad_y.dtype} must match x dtype {x.dtype}")
    expected = list(x.shape)
    expected[actual_dim] //= 2
    if list(grad_y.shape) != expected:
        raise ValueError(f"grad_y must have shape {tuple(expected)}, got {tuple(grad_y.shape)}")
    return actual_dim


def situ_glu_grad(
    x,
    grad_y,
    dim=-1,
    beta=4.0,
    linear_beta=25.0,
    activate_left=True,
    block_m=32,
    block_n=128,
    tiles_per_block=None,
):
    """SiTU-GLU backward on NPU tensors.

    Args:
        x: Forward input, float32/float16/bfloat16, dim axis even.
        grad_y: Upstream gradient with the dim axis halved, same dtype as x.
        dim: Split axis, in [-x.dim(), x.dim()-1], default -1.
        beta: SiTU gate soft-cap scale (Kimi K3 uses 4.0), must be non-zero.
        linear_beta: Up-branch soft-cap scale (Kimi K3 uses 25.0); <= 0 makes
            the up branch the identity.
        activate_left: True puts gate in the first half, False in the second.
        block_m / block_n: Kernel tile sizes; shapes not divisible by them are
            zero-padded on the host and the padding is sliced away.
        tiles_per_block: Column tiles per software-pipeline strip. ``None``
            picks one that divides the column-tile count.

    Returns:
        grad_x, same shape and dtype as x.
    """
    actual_dim = _validate(x, grad_y, dim, beta)
    if x.numel() == 0:
        return torch.empty_like(x)

    shape = x.shape
    pre = 1
    for s in shape[:actual_dim]:
        pre *= s
    cut = shape[actual_dim]
    inner = 1
    for s in shape[actual_dim + 1 :]:
        inner *= s

    rows = pre
    w_in = cut * inner
    h_in = w_in // 2

    x2 = x.reshape(rows, w_in)
    gy2 = grad_y.reshape(rows, h_in)

    rows_pad = ((rows + block_m - 1) // block_m) * block_m
    h_pad = ((h_in + block_n - 1) // block_n) * block_n

    if rows_pad != rows or h_pad != h_in:
        # Pad each half to block multiples; the padding is sliced away below.
        # Zero padding is safe: gate == 0 / up == 0 / grad_y == 0 produce finite
        # gradients and the values never reach the output.
        padded = x2.new_zeros((rows_pad, 2, h_pad))
        padded[:rows, 0, :h_in] = x2[:, :h_in]
        padded[:rows, 1, :h_in] = x2[:, h_in:]
        x2 = padded.reshape(rows_pad, 2 * h_pad)
        gy_pad = gy2.new_zeros((rows_pad, h_pad))
        gy_pad[:rows, :h_in] = gy2
        gy2 = gy_pad

    n_num = h_pad // block_n
    if tiles_per_block is None:
        # Prefer long pipeline strips (fewer waves, less fill/drain), but keep
        # at least ~32 AIV tasks in flight for small row counts.
        tiles_per_block = 1
        for c in _TILES_PER_BLOCK_CANDIDATES:
            if n_num % c == 0 and 2 * (rows_pad // block_m) * (n_num // c) >= 32:
                tiles_per_block = c
                break
    else:
        assert n_num % tiles_per_block == 0, "tiles_per_block must divide the column-tile count"

    key = (
        rows_pad,
        2 * h_pad,
        h_pad,
        block_m,
        block_n,
        tiles_per_block,
        x.dtype,
        beta,
        linear_beta,
        activate_left,
    )
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = situ_glu_grad_kernel(
            rows_pad,
            2 * h_pad,
            h_pad,
            block_m,
            block_n,
            dtype=_SITU_GLU_GRAD_DTYPES[x.dtype],
            beta=beta,
            linear_beta=linear_beta,
            activate_left=activate_left,
            tiles_per_block=tiles_per_block,
        )
    out = _KERNEL_CACHE[key](x2, gy2)
    if rows_pad != rows or h_pad != h_in:
        out = out.reshape(rows_pad, 2, h_pad)[:rows, :, :h_in].reshape(rows, w_in)
    return out.reshape(shape)


if __name__ == "__main__":
    torch.manual_seed(0)

    # Kimi K3 routed-activation shape: [M, 6144] -> grad_y [M, 3072].
    x = torch.randn(512, 6144, dtype=torch.bfloat16).npu()
    grad_y = torch.randn(512, 3072, dtype=torch.bfloat16).npu()

    grad_x = situ_glu_grad(x, grad_y, dim=-1, beta=4.0, linear_beta=25.0, activate_left=True)

    ref = golden_situ_glu_grad(x.cpu(), grad_y.cpu(), dim=-1, beta=4.0, linear_beta=25.0, activate_left=True)
    torch.testing.assert_close(grad_x.cpu(), ref.to(torch.bfloat16), rtol=1e-2, atol=1e-2)
    print("Kernel Output Match!")
