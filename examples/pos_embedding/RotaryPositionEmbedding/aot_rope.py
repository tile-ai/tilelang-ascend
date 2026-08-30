"""AOT compilation and test for RoPE kernel.

Lowers the RoPE @T.prim_func to C++ source via tilelang.engine.lower,
then compiles to a shared library using LibraryGenerator. Optionally
runs the compiled kernel via ctypes and verifies precision against
the CANN golden (torch_npu.npu_rotary_mul).

Usage:
    # Compile only
    python aot_rope.py --shape 16 64 512 256 --layout half --dtype float16

    # Compile + test
    python aot_rope.py --shape 16 64 512 256 --layout half --dtype float16 --test
"""

import argparse
import ctypes
import shutil
import sys

import tilelang
import torch
import tvm
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.utils.target import determine_platform

sys.path.insert(0, ".")
from rope_half_interleaved import cann_rope_ref, check_precision, rope_kernel, select_block_M  # noqa: E402
from rope_half_interleaved import pass_configs as rope_pass_configs  # noqa: E402

NUM_CORES = 48


def compile_rope(shape, layout, dtype_str, target, platform, output):
    """Lower RoPE kernel to C++ source and compile to .so."""
    bs, head_num, hidden_size, rope_dim = shape

    block_M = select_block_M(head_num, rope_dim, layout)
    M = bs * head_num
    sc_rows = bs
    m_num_full = M // block_M
    tail_rows = M % block_M
    has_tail = 1 if tail_rows > 0 else 0
    total_chunks = m_num_full + has_tail
    num_blocks = min(total_chunks, NUM_CORES)

    prim_func = rope_kernel.__wrapped__(
        M, block_M, num_blocks, total_chunks, sc_rows, hidden_size, rope_dim, head_num, layout, dtype=dtype_str
    )

    pass_ctx_map = {k.value: v for k, v in rope_pass_configs.items()}
    with tvm.transform.PassContext(opt_level=3, config=pass_ctx_map):
        resolved_platform = determine_platform(platform)
        artifact = tilelang.engine.lower(prim_func, target=target, platform=resolved_platform)

    lib_generator = LibraryGenerator(target=target, platform=resolved_platform)
    lib_generator.update_lib_code(artifact.kernel_source)
    lib_generator.compile_lib()
    shutil.copy(lib_generator.get_lib_path(), output)

    print(f"Built {output} (target={target}, platform={resolved_platform})")
    print(f"  shape={shape} layout={layout} dtype={dtype_str}")
    print(f"  M={M} block_M={block_M} num_blocks={num_blocks} total_chunks={total_chunks}")
    return output


def test_rope(shape, layout, dtype_str, lib_path):
    """Load AOT-compiled .so via ctypes and verify precision."""
    torch.manual_seed(42)
    bs, head_num, hidden_size, rope_dim = shape
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[dtype_str]

    x = torch.randn(bs, head_num, hidden_size, dtype=torch_dtype, device="npu")
    sin = torch.randn(bs, 1, rope_dim, dtype=torch_dtype, device="npu")
    cos = torch.randn(bs, 1, rope_dim, dtype=torch_dtype, device="npu")

    out_ref = cann_rope_ref(x.cpu().clone(), sin.cpu(), cos.cpu(), layout, dtype_str)

    x_2d = x.view(-1, hidden_size).contiguous()
    sin_2d = sin.view(-1, rope_dim).contiguous()
    cos_2d = cos.view(-1, rope_dim).contiguous()

    lib = ctypes.CDLL(lib_path)
    stream = torch.npu.current_stream()._as_parameter_

    lib.call(
        ctypes.c_void_p(x_2d.data_ptr()),
        ctypes.c_void_p(sin_2d.data_ptr()),
        ctypes.c_void_p(cos_2d.data_ptr()),
        stream,
    )
    torch.npu.synchronize()

    out_npu = x_2d.view(bs, head_num, hidden_size).cpu()
    passed, ratio, max_abs = check_precision(out_npu, out_ref, dtype_str)
    tag = "PASS" if passed else "FAIL"
    print(f"[AOT_{tag}] shape={shape} layout={layout} dtype={dtype_str} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")

    if passed:
        print("AOT Kernel Output Match!")
    else:
        print("AOT Kernel Output MISMATCH!")
        sys.exit(1)


def main():
    tilelang.cache.clear_cache()

    parser = argparse.ArgumentParser(description="RoPE AOT Compilation and Test")
    parser.add_argument("--shape", type=int, nargs=4, default=[16, 64, 512, 256], metavar=("BS", "H", "HS", "RD"))
    parser.add_argument("--layout", default="half", choices=["half", "interleaved"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--target", default="ascendc", choices=["ascendc", "pto"])
    parser.add_argument("--platform", default="auto")
    parser.add_argument("-o", "--output", default="./rope_lib.so")
    parser.add_argument("--test", action="store_true", help="Run precision test after compilation")
    args = parser.parse_args()

    compile_rope(args.shape, args.layout, args.dtype, args.target, args.platform, args.output)
    print("Test Passed!")

    if args.test:
        test_rope(args.shape, args.layout, args.dtype, args.output)


if __name__ == "__main__":
    main()
