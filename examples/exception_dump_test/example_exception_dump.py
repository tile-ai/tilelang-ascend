import argparse
import ctypes
import glob
import os
import time

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="TileLang Exception Dump Example")
parser.add_argument("--m", type=int, default=128, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=128, help="Matrix N dimension")
parser.add_argument(
    "--dump-dir",
    type=str,
    default="/tmp",
    help="Directory for exception dump log files",
)
args = parser.parse_args()

M = args.m
N = args.n

os.environ["TILELANG_EXCEPTION_DUMP_DIR"] = args.dump_dir
os.makedirs(args.dump_dir, exist_ok=True)


@tilelang.jit(
    out_idx=[2],
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    },
)
def vec_add(M, N, dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((M, N), dtype)
            b_ub = T.alloc_ub((M, N), dtype)
            c_ub = T.alloc_ub((M, N), dtype)
            T.copy(A[:, :], a_ub)
            T.copy(B[:, :], b_ub)
            for i, j in T.Parallel(M, N):
                c_ub[i, j] = a_ub[i, j] + b_ub[i, j]
            T.copy(c_ub[:, :], C[:, :])

    return main


def trigger_hw_exception(kernel, x, y):
    """Trigger an AI Core hardware exception by launching the kernel with a
    null output pointer.

    This forces the NPU to attempt writing to address 0x0 — an unmapped
    address — which reliably triggers an MTE (Memory Transfer Engine) error
    and invokes the exception dump callback.

    Normal Python/torch calls always produce valid device pointers, so
    out-of-bounds access from a valid base typically stays within device
    memory and does not trigger a hardware exception.  Passing a null
    pointer via ctypes is the simplest reliable way to get an unmapped
    address.
    """
    so_path = kernel.adapter.libpath
    lib = ctypes.CDLL(so_path)
    call_fn = lib.call
    call_fn.restype = None
    call_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]

    z_null = ctypes.c_void_p(0)
    exc_stream = torch.npu.Stream()
    print(f"  Launching on stream {exc_stream.npu_stream}")
    call_fn(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(y.data_ptr()),
        z_null,
        ctypes.c_void_p(exc_stream.npu_stream),
    )
    try:
        exc_stream.synchronize()
        print("  Stream sync completed without error (unexpected)")
    except Exception as e:
        print(f"  Stream sync raised (expected): {type(e).__name__}")


print("Compiling kernel...")
func = vec_add(M, N)

a = torch.randn(M, N, dtype=torch.float16).npu()
b = torch.randn(M, N, dtype=torch.float16).npu()

print("Normal execution...")
c = func(a, b)
ref_c = a + b
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")

print("\n--- Triggering AI Core exception (null output pointer) ---")
trigger_hw_exception(func, a, b)

time.sleep(1)

dump_files = sorted(
    glob.glob(os.path.join(args.dump_dir, "tilelang_exception_dump_*.log"))
)
# Also check /tmp as fallback
tmp_files = sorted(glob.glob("/tmp/tilelang_exception_dump_*.log"))
all_files = sorted(set(dump_files + tmp_files))

if all_files:
    print(f"\n--- Exception dump file generated: {all_files[-1]} ---")
    with open(all_files[-1], "r") as f:
        print(f.read())
else:
    print(f"\n--- No exception dump file found ---")
    print(f"  Checked: {args.dump_dir} and /tmp")

print("--- Exception test done ---")
