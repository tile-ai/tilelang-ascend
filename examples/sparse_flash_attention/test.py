import ctypes
from tilelang import DataType, language as T
import torch

torch.set_default_device('npu')
torch.manual_seed(0)

lib_path = "./kernel_lib.so"
lib = ctypes.CDLL(lib_path)

stream = torch.npu.current_stream()._as_parameter_

core_num = 20

block_num = 516
block_size = 128

B, S, SKV, H, HKV, DQK, DV, topk = 1, 1024, 32768, 128, 1, 576, 512, 2048
dtype = torch.bfloat16


q = torch.randn((B, S, H, DQK), dtype=dtype)
kv = torch.randn((block_num, block_size, 1, DQK), dtype=dtype)
indices = torch.full((S, HKV, topk), SKV, dtype=torch.int32)

for t in range(S):
    for h in range(HKV):
        i_i = torch.randperm(max(1, t))[:topk]
        indices[t, h, :len(i_i)] = i_i
torch.npu.synchronize()
# output = torch.empty((B, S, H, DV), dtype=dtype)
workspace_1 = torch.zeros((core_num, 64, 512), dtype=dtype)
workspace_2 = torch.zeros((core_num, 64, 64), dtype=dtype)
workspace_3 = torch.zeros((core_num, 64, 64), dtype=torch.float)
workspace_4 = torch.zeros((core_num, 64, 64), dtype=dtype)
workspace_5 = torch.zeros((core_num, 64, 512), dtype=torch.float)

block_table = torch.zeros((B, SKV // block_size), dtype=torch.int32)


output = torch.empty((B, S, H, DV), dtype=dtype)

actual_q_len = torch.tensor([S] * B, dtype=torch.int32)
actual_kv_len = torch.tensor([SKV] * B, dtype=torch.int32)


torch.npu.synchronize()
print("init successful!")

def tl_gemm():
    return lib.call(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(kv.data_ptr()),
        ctypes.c_void_p(indices.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(actual_q_len.data_ptr()),
        ctypes.c_void_p(actual_kv_len.data_ptr()),
        ctypes.c_void_p(block_table.data_ptr()),
        ctypes.c_void_p(workspace_1.data_ptr()),
        ctypes.c_void_p(workspace_2.data_ptr()),
        ctypes.c_void_p(workspace_3.data_ptr()),
        ctypes.c_void_p(workspace_4.data_ptr()),
        ctypes.c_void_p(workspace_5.data_ptr()),
        S, SKV // block_size, stream)

tl_gemm()
torch.npu.synchronize()

print(output)

print(torch.isnan(output).sum())
