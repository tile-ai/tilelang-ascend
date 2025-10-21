import torch
import argparse
import ctypes
from functools import partial
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
    return lib.call(
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(c.data_ptr()),
        stream)

tl_gemm()
torch.npu.synchronize()


ref_c = a @ b

torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")

