"""PTO backend test for RoPE kernel.

Re-JITs the RoPE kernel with target="pto" (Expert mode: T.Scope + alloc_ub)
and verifies precision against the CANN golden (torch_npu.npu_rotary_mul).

PTO backend requires T.Scope("V") + alloc_ub (Expert mode) because the CCE
compiler (-xcce) needs [aicore] attribute on all code using PTO intrinsics.

Usage:
    python pto_rope.py --shape 16 64 512 256 --layout half --dtype float16
"""

import argparse
import os
import sys

import tilelang
import tilelang.language as T
import torch

sys.path.insert(0, ".")
from rope_half_interleaved import cann_rope_ref, check_precision, select_block_M  # noqa: E402

tilelang.cache.clear_cache()
torch.manual_seed(42)

parser = argparse.ArgumentParser(description="RoPE PTO backend test")
parser.add_argument("--shape", type=int, nargs=4, default=[16, 64, 512, 256], metavar=("BS", "H", "HS", "RD"))
parser.add_argument("--layout", default="half", choices=["half", "interleaved"])
parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
parser.add_argument("--platform", default="auto", help="Hardware platform (auto/A2/A3/A5)")
args = parser.parse_args()

if args.platform != "auto":
    os.environ["TL_PLATFORM"] = args.platform

bs, head_num, hidden_size, rope_dim = args.shape
dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
torch_dtype = dtype_map[args.dtype]


@tilelang.jit(target="pto")
def rope_kernel_pto(M, block_M, num_blocks, total_chunks, sc_rows, hidden_size, rope_dim, head_num, layout, dtype="float16"):
    VEC_NUM = 2
    dim_start = hidden_size - rope_dim
    row_per_vec = block_M // VEC_NUM
    half = rope_dim // 2
    ACC_DTYPE = "float32"
    # MASK_DTYPE = "uint32"
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
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid), T.Scope("V"):
            x_half_ub = T.alloc_ub([row_per_vec, rope_dim], dtype)
            x_ub = T.alloc_ub([row_per_vec, rope_dim], ACC_DTYPE)
            sin_ub = T.alloc_ub([1, rope_dim], ACC_DTYPE)
            sin_half_ub = T.alloc_ub([1, rope_dim], dtype)
            cos_ub = T.alloc_ub([1, rope_dim], ACC_DTYPE)
            cos_half_ub = T.alloc_ub([1, rope_dim], dtype)
            sin_block_ub = T.alloc_ub([row_per_vec, rope_dim], ACC_DTYPE)
            cos_block_ub = T.alloc_ub([row_per_vec, rope_dim], ACC_DTYPE)
            x_rotate_ub = T.alloc_ub([row_per_vec, rope_dim], ACC_DTYPE)
            out_ub = T.alloc_ub([row_per_vec, rope_dim], ACC_DTYPE)
            out_half_ub = T.alloc_ub([row_per_vec, rope_dim], dtype)
            idx_ub = T.alloc_ub([row_per_vec, rope_dim], "int32")
            ones_ub = T.alloc_ub([row_per_vec, rope_dim], "int16")
            mask_ub_i16 = T.alloc_ub([row_per_vec, rope_dim], "int16")
            mask_ub_f32 = T.alloc_ub([row_per_vec, rope_dim], "float32")
            mask_ub_i32 = T.alloc_ub([row_per_vec, rope_dim], "int32")
            sin_mask_ub = T.alloc_ub([1, rope_dim], ACC_DTYPE)

            # Gather offset generation (interleaved only)
            if layout == "interleaved":
                T.tile.createvecindex(idx_ub, 0)
                T.copy(idx_ub, mask_ub_i16)
                T.tile.fill(ones_ub, 1)
                T.tile.bitwise_xor(mask_ub_i16, mask_ub_i16, ones_ub)
                T.copy(mask_ub_i16, mask_ub_f32)
                T.copy(mask_ub_f32, mask_ub_i32)
                T.tile.mul(mask_ub_i32, mask_ub_i32, 4)
                T.barrier_all()
                # Use mask_ub_i32 directly as gather offset (PTO accepts int32)

            # sin_mask generation
            T.tile.fill(sin_mask_ub, -1.0)
            T.barrier_all()
            if layout == "interleaved":
                for i in T.serial(0, half):
                    sin_mask_ub[0, 2 * i + 1] = 1.0
            else:
                for i in T.serial(0, half):
                    sin_mask_ub[0, half + i] = 1.0

            # Chunk loop
            for chunk in T.serial(0, chunks_per_block):
                chunk_idx = cid * chunks_per_block + chunk
                if chunk_idx < total_chunks:
                    row_x = chunk_idx * block_M + vid * row_per_vec
                    row_sin_cos = (row_x // head_num) % sc_rows

                    # Load x
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

                    T.barrier_all()

                    if need_cast:
                        T.tile.cast(x_ub, x_half_ub, "CAST_NONE", x_elem_count)
                    else:
                        T.copy(x_half_ub, x_ub)

                    # Load sin/cos
                    T.copy(sin[row_sin_cos, :], sin_half_ub[0, :])
                    T.copy(cos[row_sin_cos, :], cos_half_ub[0, :])
                    T.barrier_all()
                    if need_cast:
                        T.tile.cast(sin_ub, sin_half_ub, "CAST_NONE", sc_elem_count)
                        T.tile.cast(cos_ub, cos_half_ub, "CAST_NONE", sc_elem_count)
                    else:
                        T.copy(sin_half_ub, sin_ub)
                        T.copy(cos_half_ub, cos_ub)

                    T.barrier_all()

                    # Apply sin_mask + broadcast
                    T.tile.mul(sin_ub[0, :], sin_ub[0, :], sin_mask_ub[0, :])
                    T.tile.broadcast(sin_block_ub, sin_ub)
                    T.tile.broadcast(cos_block_ub, cos_ub)

                    # Rotate x
                    if layout == "interleaved":
                        T.tile.gather(x_rotate_ub, x_ub, mask_ub_i32, 0)
                    else:
                        for i in T.serial(0, row_per_vec):
                            T.copy(x_ub[i, half:], x_rotate_ub[i, :half])
                            T.copy(x_ub[i, :half], x_rotate_ub[i, half:])

                    # out = x * cos + x_rotate * sin_signed
                    T.tile.mul(out_ub, x_ub, cos_block_ub)
                    T.tile.mul(x_rotate_ub, x_rotate_ub, sin_block_ub)
                    T.tile.add(out_ub, out_ub, x_rotate_ub)

                    # Downcast
                    if need_cast:
                        T.tile.cast(out_half_ub, out_ub, "CAST_RINT", x_elem_count)
                    else:
                        T.copy(out_ub, out_half_ub)

                    T.barrier_all()

                    # Write back
                    if row_x + row_per_vec <= M:
                        if dim_start == 0:
                            T.copy(out_half_ub, x[row_x : row_x + row_per_vec, :])
                        else:
                            for i in T.serial(0, row_per_vec):
                                T.copy(out_half_ub[i, :], x[row_x + i, dim_start:])
                    else:
                        for i in T.serial(0, row_per_vec):
                            if row_x + i < M:
                                if dim_start == 0:
                                    T.copy(out_half_ub[i, :], x[row_x + i, :])
                                else:
                                    T.copy(out_half_ub[i, :], x[row_x + i, dim_start:])

    return kernel


# Compute kernel params
M = bs * head_num
block_M = select_block_M(head_num, rope_dim, args.layout)
m_num_full = M // block_M
tail_rows = M % block_M
has_tail = 1 if tail_rows > 0 else 0
total_chunks = m_num_full + has_tail
num_blocks = min(total_chunks, 48)

# Inputs
x = torch.randn(bs, head_num, hidden_size, dtype=torch_dtype, device="npu")
sin = torch.randn(bs, 1, rope_dim, dtype=torch_dtype, device="npu")
cos = torch.randn(bs, 1, rope_dim, dtype=torch_dtype, device="npu")

# Golden
out_ref = cann_rope_ref(x.cpu().clone(), sin.cpu(), cos.cpu(), args.layout, args.dtype)

# Reshape for kernel
x_2d = x.view(-1, hidden_size).contiguous()
sin_2d = sin.view(-1, rope_dim).contiguous()
cos_2d = cos.view(-1, rope_dim).contiguous()

# Compile and run PTO kernel
print(f"Compiling PTO kernel (M={M}, block_M={block_M}, num_blocks={num_blocks})...")
kernel = rope_kernel_pto(M, block_M, num_blocks, total_chunks, bs, hidden_size, rope_dim, head_num, args.layout, dtype=args.dtype)

kernel(x_2d, sin_2d, cos_2d)
torch.npu.synchronize()

# Compare
out_npu = x_2d.view(bs, head_num, hidden_size).cpu()
passed, ratio, max_abs = check_precision(out_npu, out_ref, args.dtype)
tag = "PASS" if passed else "FAIL"
print(f"[PTO_{tag}] shape={args.shape} layout={args.layout} dtype={args.dtype} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")

if passed:
    print("PTO Kernel Output Match!")
else:
    print("PTO Kernel Output MISMATCH!")
    sys.exit(1)
