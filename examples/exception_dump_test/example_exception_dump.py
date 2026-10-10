import argparse
import os

parser = argparse.ArgumentParser(description="TileLang Exception Dump Example")
parser.add_argument("--m", type=int, default=128, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=128, help="Matrix N dimension")
parser.add_argument(
    "--dump-dir",
    type=str,
    default="/tmp/tilelang_exc_dump",
    help="Directory for CANN exception dump files (ASCEND_DUMP_PATH)",
)
args = parser.parse_args()

# CANN reads ASCEND_DUMP_PATH / ASCEND_DUMP_SCENE during ACL initialization,
# which happens at the first NPU call (or torch_npu import). These must be
# set before importing torch / tilelang so they take effect.
os.environ["ASCEND_DUMP_PATH"] = args.dump_dir
os.environ["ASCEND_DUMP_SCENE"] = "aic_err_brief_dump"
os.makedirs(args.dump_dir, exist_ok=True)

import ctypes

import tilelang
import tilelang.language as T
import torch
from tilelang.tools.ascend_exception_dump_bin import parse_exception_dump


def _check_precision(actual, golden):
    atol, rtol, cap = {
        torch.float16: (2**-14, 2**-9, 0.1),
        torch.bfloat16: (2**-10, 2**-6, 1.0),
        torch.float32: (2**-16, 2**-10, 0.01),
    }.get(actual.dtype, (2**-14, 2**-9, 0.1))
    sa = torch.isnan(actual) | torch.isinf(actual)
    sg = torch.isnan(golden) | torch.isinf(golden)
    if not torch.equal(sa, sg) or (sa.any() and not torch.equal(actual[sa], golden[sg])):
        raise AssertionError("NaN/Inf mismatch")
    v = ~sg
    if v.any():
        d = (actual[v] - golden[v]).abs()
        p = d <= atol + rtol * golden[v].abs()
        if p.float().mean().item() < 0.99 or d.max().item() > cap:
            raise AssertionError("precision mismatch")


tilelang.cache.clear_cache()

M = args.m
N = args.n


@tilelang.jit(
    out_idx=[2],
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tilelang.PassConfigKey.TL_ASCEND_EXCEPTION_DUMP: True,
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


def launch_kernel_with_hw_exception_triggered(kernel, x, y):
    """Trigger an AI Core hardware exception by launching the kernel with a
    null output pointer.

    This forces the NPU to attempt writing to address 0x0 — an unmapped
    address — which reliably triggers an MTE (Memory Transfer Engine) error
    and invokes the exception dump callback.

    Raises
    ------
    RuntimeError
        Propagated from stream.synchronize() when the hardware exception fires.
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
    exc_stream.synchronize()
    print("  Stream sync completed without error (unexpected)")


print("Compiling kernel...")
func = vec_add(M, N)

a = torch.randn(M, N, dtype=torch.float16).npu()
b = torch.randn(M, N, dtype=torch.float16).npu()

print("Normal execution...")
c = func(a, b)
ref_c = a + b
_check_precision(c, ref_c)
print("Kernel Output Match!")

print("\n--- Triggering AI Core exception (null output pointer) ---")
try:
    launch_kernel_with_hw_exception_triggered(func, a, b)
except Exception as e:
    print(f"  Exception caught: {type(e).__name__}")
    tensors = parse_exception_dump(args.dump_dir, kernel_name="main_kernel")
    print(f"  Dump file parsed, {len(tensors)} tensor(s) recovered:")
    for t in tensors:
        data = t["data"].reshape(M, N)
        print(f"    {t['type']}[{t['index']}] dtype={t['dtype']}, shape={data.shape}, min={data.min():.4f}, max={data.max():.4f}")

print("--- Exception test done ---")
