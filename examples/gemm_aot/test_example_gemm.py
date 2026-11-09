import torch
import argparse
import ctypes


def _check_precision(actual, golden):
    if not actual.is_floating_point():
        torch.testing.assert_close(actual, golden, rtol=0, atol=0)
        return
    atol, rtol, cap = {
        torch.float16: (2**-14, 2**-9, 1e-1),
        torch.bfloat16: (2**-10, 2**-6, 1.0),
        torch.float32: (2**-16, 2**-10, 1e-2),
    }.get(actual.dtype, (2**-14, 2**-9, 1e-1))
    sa = torch.isnan(actual) | torch.isinf(actual)
    sg = torch.isnan(golden) | torch.isinf(golden)
    if not torch.equal(sa, sg) or (sa.any() and not torch.equal(actual[sa], golden[sg])):
        raise AssertionError("NaN/Inf mismatch")
    valid = ~sg
    if valid.any():
        d = (actual[valid] - golden[valid]).abs()
        p = d <= atol + rtol * golden[valid].abs()
        if p.float().mean().item() < 0.99 or d.max().item() > cap:
            raise AssertionError("precision mismatch")


torch.manual_seed(42)

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=8192, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=8192, help="Matrix K dimension")
args = parser.parse_args()

M = args.m
N = args.n
K = args.k


a = torch.randn(M, K).half().npu()
b = torch.randn(K, N).half().npu()
c = torch.empty(M, N).half().npu()
print("init successful!")

lib_path = "./kernel_lib.so"
lib = ctypes.CDLL(lib_path)

stream = torch.npu.current_stream()._as_parameter_


def tl_gemm():
    return lib.call(ctypes.c_void_p(a.data_ptr()), ctypes.c_void_p(b.data_ptr()), ctypes.c_void_p(c.data_ptr()), stream)


tl_gemm()
torch.npu.synchronize()


ref_c = a @ b

_check_precision(c, ref_c)
print("Kernel Output Match!")
