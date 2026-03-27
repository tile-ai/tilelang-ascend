import torch
import tilelang
import tilelang.language as T
import argparse

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

device = torch.device("npu")


def _rope_kernel_base(block_M, hidden_size, rope_dim, dtype="float16"):
    """Common setup: returns constants used by both fwd and bwd kernels."""
    VEC_NUM = 2
    dim_start = hidden_size - rope_dim
    row_per_vec = block_M // VEC_NUM
    need_cast = dtype not in ("float", "float32")
    return VEC_NUM, dim_start, row_per_vec, need_cast


@tilelang.jit(pass_configs=pass_configs)
def rope_fwd_kernel(M, block_M, num_blocks, sc_rows, hidden_size, rope_dim, head_num, dtype="float16"):
    """Forward: out = x * cos + rotate(x) * sin
    rotate([x0,x1,x2,x3]) = [-x1, x0, -x3, x2]
    """
    _, dim_start, row_per_vec, need_cast = _rope_kernel_base(
        block_M, hidden_size, rope_dim, dtype)
    ACC_DTYPE = "float32"
    MASK_DTYPE = "uint32"
    x_elem_count = row_per_vec * rope_dim
    sc_elem_count = rope_dim
    chunks_per_block = (M // block_M + num_blocks - 1) // num_blocks

    @T.prim_func
    def kernel(
        x: T.Tensor([M, hidden_size], dtype),  # type: ignore
        sin: T.Tensor([sc_rows, rope_dim], dtype),  # type: ignore
        cos: T.Tensor([sc_rows, rope_dim], dtype),  # type: ignore
        mask: T.Tensor([row_per_vec, rope_dim], MASK_DTYPE),  # type: ignore
    ):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            x_half_ub = T.alloc_shared([row_per_vec, rope_dim], dtype)
            x_ub = T.alloc_shared([row_per_vec, rope_dim], ACC_DTYPE)
            sin_ub = T.alloc_shared([1, rope_dim], ACC_DTYPE)
            sin_half_ub = T.alloc_shared([1, rope_dim], dtype)
            cos_ub = T.alloc_shared([1, rope_dim], ACC_DTYPE)
            cos_half_ub = T.alloc_shared([1, rope_dim], dtype)
            mask_ub = T.alloc_shared([row_per_vec, rope_dim], MASK_DTYPE)
            out_ub = T.alloc_shared([row_per_vec, rope_dim], ACC_DTYPE)

            T.copy(mask, mask_ub)

            for chunk in T.serial(0, chunks_per_block):
                chunk_idx = cid * chunks_per_block + chunk
                if chunk_idx < M // block_M:
                    row_x = chunk_idx * block_M + vid * row_per_vec
                    row_sin_cos = (row_x // head_num) % sc_rows

                    if dim_start == 0:
                        T.copy(x[row_x: row_x + row_per_vec, :], x_half_ub)
                    else:
                        for i in T.serial(0, row_per_vec):
                            T.copy(x[row_x + i, dim_start:], x_half_ub[i, :])
                    if need_cast:
                        T.tile.cast(x_ub, x_half_ub, "CAST_NONE", x_elem_count)
                    else:
                        T.copy(x_half_ub, x_ub)

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

                    T.tile.gather(out_ub, x_ub, mask_ub, 0)

                    for i in T.serial(0, row_per_vec):
                        T.tile.mul(out_ub[i, :], out_ub[i, :], sin_ub[0, :])
                        T.tile.mul(x_ub[i, :], x_ub[i, :], cos_ub[0, :])
                        T.tile.add(x_ub[i, :], x_ub[i, :], out_ub[i, :])

                    if need_cast:
                        T.tile.cast(x_half_ub, x_ub, "CAST_RINT", x_elem_count)
                    else:
                        T.copy(x_ub, x_half_ub)
                    if dim_start == 0:
                        T.copy(x_half_ub, x[row_x: row_x + row_per_vec, :])
                    else:
                        for i in T.serial(0, row_per_vec):
                            T.copy(x_half_ub[i, :], x[row_x + i, dim_start:])

    return kernel


@tilelang.jit(pass_configs=pass_configs)
def rope_bwd_kernel(M, block_M, num_blocks, sc_rows, hidden_size, rope_dim, head_num, dtype="float16"):
    """Backward: dx = cos*dy + swap(sin_masked * dy)

    sin_masked uses the same [-1,1,-1,1,...] sign mask as forward.

    Derivation (per-element sin/cos, pair indices 2k, 2k+1):
      fwd: out[2k]   = cos[2k]*x[2k]   + (-sin[2k])*x[2k+1]
           out[2k+1] = cos[2k+1]*x[2k+1] + sin[2k+1]*x[2k]
      bwd: dx[2k]   = cos[2k]*dy[2k]   + sin[2k+1]*dy[2k+1]
           dx[2k+1] = -sin[2k]*dy[2k]   + cos[2k+1]*dy[2k+1]
    Which is: dx = cos*dy + swap(sign_mask * sin * dy)
    """
    _, dim_start, row_per_vec, need_cast = _rope_kernel_base(
        block_M, hidden_size, rope_dim, dtype)
    ACC_DTYPE = "float32"
    MASK_DTYPE = "uint32"
    x_elem_count = row_per_vec * rope_dim
    sc_elem_count = rope_dim
    chunks_per_block = (M // block_M + num_blocks - 1) // num_blocks

    @T.prim_func
    def kernel(
        x: T.Tensor([M, hidden_size], dtype),  # type: ignore
        sin: T.Tensor([sc_rows, rope_dim], dtype),  # type: ignore
        cos: T.Tensor([sc_rows, rope_dim], dtype),  # type: ignore
        mask: T.Tensor([row_per_vec, rope_dim], MASK_DTYPE),  # type: ignore
    ):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            x_half_ub = T.alloc_shared([row_per_vec, rope_dim], dtype)
            x_ub = T.alloc_shared([row_per_vec, rope_dim], ACC_DTYPE)
            sin_ub = T.alloc_shared([1, rope_dim], ACC_DTYPE)
            sin_half_ub = T.alloc_shared([1, rope_dim], dtype)
            cos_ub = T.alloc_shared([1, rope_dim], ACC_DTYPE)
            cos_half_ub = T.alloc_shared([1, rope_dim], dtype)
            mask_ub = T.alloc_shared([row_per_vec, rope_dim], MASK_DTYPE)
            dy_sin_ub = T.alloc_shared([row_per_vec, rope_dim], ACC_DTYPE)
            out_ub = T.alloc_shared([row_per_vec, rope_dim], ACC_DTYPE)

            T.copy(mask, mask_ub)

            for chunk in T.serial(0, chunks_per_block):
                chunk_idx = cid * chunks_per_block + chunk
                if chunk_idx < M // block_M:
                    row_x = chunk_idx * block_M + vid * row_per_vec
                    row_sin_cos = (row_x // head_num) % sc_rows

                    if dim_start == 0:
                        T.copy(x[row_x: row_x + row_per_vec, :], x_half_ub)
                    else:
                        for i in T.serial(0, row_per_vec):
                            T.copy(x[row_x + i, dim_start:], x_half_ub[i, :])
                    if need_cast:
                        T.tile.cast(x_ub, x_half_ub, "CAST_NONE", x_elem_count)
                    else:
                        T.copy(x_half_ub, x_ub)

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

                    for i in T.serial(0, row_per_vec):
                        T.tile.mul(dy_sin_ub[i, :], x_ub[i, :], sin_ub[0, :])

                    T.tile.gather(out_ub, dy_sin_ub, mask_ub, 0)

                    for i in T.serial(0, row_per_vec):
                        T.tile.mul(x_ub[i, :], x_ub[i, :], cos_ub[0, :])
                        T.tile.add(out_ub[i, :], out_ub[i, :], x_ub[i, :])

                    if need_cast:
                        T.tile.cast(x_half_ub, out_ub, "CAST_RINT", x_elem_count)
                    else:
                        T.copy(out_ub, x_half_ub)
                    if dim_start == 0:
                        T.copy(x_half_ub, x[row_x: row_x + row_per_vec, :])
                    else:
                        for i in T.serial(0, row_per_vec):
                            T.copy(x_half_ub[i, :], x[row_x + i, dim_start:])

    return kernel


torch_dtype_map = {
    "float16": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
}

tilelang_dtype_map = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float: "float",
}


def _apply_rope_in_place(x, sin, cos, kernel_fn):
    rope_dim = sin.shape[-1]
    org_shape = x.shape
    dtype_str = tilelang_dtype_map[x.dtype]

    if x.dim() == 3:
        # x: [BS, N, D], sin/cos: [BS, 1, RD] -> squeeze to [BS, RD]
        bs, head_num, hidden_size = x.shape
        x = x.view(-1, hidden_size)
        sin = sin.view(-1, rope_dim)
        cos = cos.view(-1, rope_dim)
        sc_rows = bs
    elif x.dim() == 4:
        # x: [B, S, N, D], sin/cos: [1, S, 1, RD] -> squeeze to [S, RD]
        b, s, head_num, hidden_size = x.shape
        x = x.view(-1, hidden_size)
        sin = sin.view(-1, rope_dim)
        cos = cos.view(-1, rope_dim)
        sc_rows = s
    else:
        raise NotImplementedError(f"x_shape={x.shape} not supported")

    total_rows = x.shape[0]
    max_block = 32 if dtype_str == "float" else 64
    block_M = max_block if total_rows % max_block == 0 else 32
    row_per_vec = block_M // 2

    NUM_CORES = 48
    m_num = total_rows // block_M
    num_blocks = min(m_num, NUM_CORES)

    idx = torch.arange(rope_dim * row_per_vec, dtype=torch.int64, device="cpu")
    mask = torch.empty(rope_dim * row_per_vec, dtype=torch.uint32, device="cpu")
    mask[0::2] = idx[1::2].to(torch.uint32)
    mask[1::2] = idx[0::2].to(torch.uint32)
    mask = (mask * 4).to(x.device)

    sin_mask = torch.ones(rope_dim, dtype=sin.dtype, device=x.device)
    sin_mask[0::2] = -1
    sin = sin * sin_mask

    kernel = kernel_fn(total_rows, block_M, num_blocks, sc_rows, hidden_size, rope_dim, head_num, dtype=dtype_str)
    kernel(x, sin, cos, mask)

    return x.view(org_shape)


class _RoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sin, cos):
        x_out = x.clone()
        _apply_rope_in_place(x_out, sin, cos, rope_fwd_kernel)
        ctx.save_for_backward(sin, cos)
        return x_out

    @staticmethod
    def backward(ctx, grad_output):
        sin, cos = ctx.saved_tensors
        dx = grad_output.clone()
        _apply_rope_in_place(dx, sin, cos, rope_bwd_kernel)
        return dx, None, None


def tilelang_rope(x, sin, cos):
    return _RoPE.apply(x, sin, cos)


# --- Reference ---

def torch_rope_partial(x, sin, cos):
    """Apply RoPE to the last rope_dim dimensions, keep the rest unchanged."""
    rope_dim = sin.shape[-1]
    dim_start = x.shape[-1] - rope_dim

    if dim_start == 0 and x.dim() == 3:
        import torch_npu
        return torch_npu.npu_rotary_mul(x, cos, sin, rotary_mode='interleave')

    x_part = x[..., dim_start:].to(torch.float32)
    sin_f = sin.to(torch.float32)
    cos_f = cos.to(torch.float32)

    # broadcast sin/cos to match x_part shape
    # TND x: [BS, N, D], sin/cos: [BS, 1, RD]  -> already broadcastable
    # BSND x: [B, S, N, D], sin/cos: [1, S, 1, RD] -> already broadcastable

    # interleaved rotate: [-x1, x0, -x3, x2, ...]
    x_reshaped = x_part.reshape(*x_part.shape[:-1], -1, 2)
    x_rotated = torch.stack([-x_reshaped[..., 1], x_reshaped[..., 0]], dim=-1).flatten(-2)

    rope_out = (x_part * cos_f + x_rotated * sin_f).to(x.dtype)

    out = x.clone()
    out[..., dim_start:] = rope_out
    return out


# --- Main ---

def check_case_tnd(batch_size, head_num, hidden_size, rope_dim, dtype_str="float16"):
    """Test TND input: x=[BS, N, D], sin/cos=[BS, 1, RD]"""
    torch_dtype = torch_dtype_map[dtype_str]

    x = torch.randn(
        (batch_size, head_num, hidden_size), device=device, dtype=torch_dtype
    ).requires_grad_(True)
    sin = torch.randn((batch_size, 1, rope_dim), device=device, dtype=torch_dtype)
    cos = torch.randn((batch_size, 1, rope_dim), device=device, dtype=torch_dtype)

    x_ref = x.clone().detach().requires_grad_(True)
    out_ref = torch_rope_partial(x_ref, sin, cos)
    dout = torch.randn_like(out_ref)
    out_ref.backward(dout)
    dx_ref = x_ref.grad.clone()

    out_tl = tilelang_rope(x, sin, cos)
    out_tl.backward(dout)
    dx_tl = x.grad.clone()

    fwd_ok = torch.allclose(out_tl, out_ref, rtol=1e-3, atol=1e-3)
    bwd_ok = torch.allclose(dx_tl, dx_ref, rtol=1e-3, atol=1e-3)

    tag = f"[tnd {dtype_str}]"
    if fwd_ok and bwd_ok:
        print(f"{tag} Forward and Backward Match!")
    else:
        if not fwd_ok:
            print(f"{tag} Forward Mismatch! max diff: {(out_tl - out_ref).abs().max().item()}")
        if not bwd_ok:
            print(f"{tag} Backward Mismatch! max diff: {(dx_tl - dx_ref).abs().max().item()}")


def check_case_bsnd(batch, seq_len, head_num, hidden_size, rope_dim, dtype_str="float16"):
    """Test BSND input: x=[B, S, N, D], sin/cos=[1, S, 1, RD]"""
    torch_dtype = torch_dtype_map[dtype_str]

    x = torch.randn(
        (batch, seq_len, head_num, hidden_size), device=device, dtype=torch_dtype
    ).requires_grad_(True)
    sin = torch.randn((1, seq_len, 1, rope_dim), device=device, dtype=torch_dtype)
    cos = torch.randn((1, seq_len, 1, rope_dim), device=device, dtype=torch_dtype)

    x_ref = x.clone().detach().requires_grad_(True)
    out_ref = torch_rope_partial(x_ref, sin, cos)
    dout = torch.randn_like(out_ref)
    out_ref.backward(dout)
    dx_ref = x_ref.grad.clone()

    out_tl = tilelang_rope(x, sin, cos)
    out_tl.backward(dout)
    dx_tl = x.grad.clone()

    fwd_ok = torch.allclose(out_tl, out_ref, rtol=1e-3, atol=1e-3)
    bwd_ok = torch.allclose(dx_tl, dx_ref, rtol=1e-3, atol=1e-3)

    tag = f"[bsnd {dtype_str}]"
    if fwd_ok and bwd_ok:
        print(f"{tag} Forward and Backward Match!")
    else:
        if not fwd_ok:
            print(f"{tag} Forward Mismatch! max diff: {(out_tl - out_ref).abs().max().item()}")
        if not bwd_ok:
            print(f"{tag} Backward Mismatch! max diff: {(dx_tl - dx_ref).abs().max().item()}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shape", type=int, nargs=4,
        metavar=("BS", "H", "HS", "RD"),
        default=[16, 64, 512, 256],
        help="batch_size head_num hidden_size rope_dim",
    )
    return p.parse_args()


if __name__ == "__main__":
    torch.manual_seed(42)
    tilelang.disable_cache()

    args = parse_args()
    batch_size, head_num, hidden_size, rope_dim = args.shape

    # Test TND: x=[BS, N, D], sin/cos=[BS, 1, RD]
    for dtype_str in ["float16", "bfloat16", "float"]:
        check_case_tnd(batch_size, head_num, hidden_size, rope_dim, dtype_str)

    # Test BSND: x=[B, S, N, D], sin/cos=[1, S, 1, RD]
    B, S = 4, batch_size // 4 if batch_size >= 4 else batch_size
    for dtype_str in ["float16", "bfloat16", "float"]:
        check_case_bsnd(B, S, head_num, hidden_size, rope_dim, dtype_str)
