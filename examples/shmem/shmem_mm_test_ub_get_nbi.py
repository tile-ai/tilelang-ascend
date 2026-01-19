import argparse

import tilelang
import tilelang.language as T
import torch
import shmem as aclshmem_module
import multiprocessing as mp
from multiprocessing import Barrier
import sys
import os

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=1, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=352, help="Matrix N dimension")
parser.add_argument("--num_processes", type=int, default=8, help="number of processes")
args = parser.parse_args()

M = args.m
N = args.n
g_ash_size = 1024 * 1024 * 1024
G_IP_PORT = "tcp://100.102.180.145:8666"

num_processes = args.num_processes

@tilelang.jit()
def shmem_ub_get_nbi(M, N, nelems, newPe, dtype="int8"):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub_tensor = T.alloc_ub((1, nelems), dtype)
            with T.Scope("V"):
                if vid == 0:
                    # Copy from the newPe GM to the local UB
                    T.shmem_ub_get_nbi(ub_tensor, A, nelems, newPe)
                    T.set_flag("mte2", "mte3", 0x7)
                    T.wait_flag("mte2", "mte3", 0x7)
                    T.copy(ub_tensor, B)
                    T.pipe_barrier("mte3")
    return main

def worker(rank, barrier):
    print(f"Rank {rank}: Setting device")
    torch.npu.set_device(rank)
    ret = aclshmem_module.set_conf_store_tls(False, "")
    if ret != 0:
        raise ValueError("[ERROR] set_conf_store_tls failed")
    # Create initialization attribute object
    attributes = aclshmem_module.InitAttr()
    npu_num = 8
    attributes.my_rank = rank
    attributes.n_ranks = npu_num
    attributes.local_mem_size = g_ash_size
    attributes.ip_port = G_IP_PORT
    attributes.option_attr.data_op_engine_type = aclshmem_module.OpEngineType.MTE
    # Initialize aclshmem
    ret = aclshmem_module.aclshmem_init(attributes)
    if ret == 0:
        print(f"Rank {rank}: Initialization successful")
        torch.manual_seed(0)
        # Create shared memory tensor
        tensor = aclshmem_module.aclshmem_create_tensor([M, 2*N], dtype=torch.int8, device_id=rank)
        a = tensor[0:1, 0:N].fill_(2)
        b = tensor[0:1, N:2*N].fill_(0)
        torch.npu.synchronize()
        nelems = M * N
        # Get data from a new card to this PE; here, it's set as the previous rank.
        newPe = (rank + npu_num - 1) % npu_num

        func = shmem_ub_get_nbi(M, N, nelems, newPe)
        func(a, b)
        barrier.wait()
        print("b after=", b)
        if torch.equal(a, b):
            print("Test passed!")
        else:
            print("[ERROR] Kernel Output Not Match!")
        aclshmem_module.aclshmem_free_tensor(tensor)

    else:
        print(f"Rank {rank}: Initialization failed with code {ret}")    
    # Clean
    aclshmem_module.aclshmem_finialize()
    print(f"Rank {rank}: Finalized")

# [Program Start Location]
print(f"Number of processes: {num_processes}")
# Synchronize a specified number of processes
barrier = Barrier(num_processes)
processes = []

for rank in range(num_processes):
    p = mp.Process(target=worker, args=(rank, barrier))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

print("Kernel Output Match!")