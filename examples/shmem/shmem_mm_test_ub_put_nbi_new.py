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
parser.add_argument("--num_processes", type=int, default=2, help="number of processes")
args = parser.parse_args()

M = args.m
N = args.n
g_ash_size = 1024 * 1024 * 1024
g_malloc_size = 8 * 1024 * 1024
G_IP_PORT = "tcp://100.102.180.145:8666"

num_processes = args.num_processes

@tilelang.jit(out_idx=[-1])
def shmem_ub_put_nbi(M, N, nelems, newPe, dtype="int8"):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub_tensor = T.alloc_ub((1, nelems), dtype)
            with T.Scope("V"):
                if vid == 0:
                    T.copy(A, ub_tensor)
                    # T.set_flag("mte2", "mte3", 0x7)
                    # T.wait_flag("mte2", "mte3", 0x7)
                    T.barrier_all()
                    T.shmem_ub_put_nbi_new(ub_tensor, B, nelems, newPe)
                    # T.pipe_barrier("mte3")
                    T.barrier_all()
    return main

def worker(rank, barrier):
    print(f"Rank {rank}: Setting device")
    torch.npu.set_device(rank)
    # 1. test set tls info 是否需要？
    ret = aclshmem_module.set_conf_store_tls(False, "")
    if ret != 0:
        raise ValueError("[ERROR] set_conf_store_tls failed")
    # 创建初始化属性对象
    attributes = aclshmem_module.InitAttr()
    npu_num = 2
    attributes.my_rank = rank
    attributes.n_ranks = npu_num
    attributes.local_mem_size = g_ash_size
    attributes.ip_port = G_IP_PORT
    attributes.option_attr.data_op_engine_type = aclshmem_module.OpEngineType.MTE
    # 初始化aclshmem
    ret = aclshmem_module.aclshmem_init(attributes)
    if ret == 0:
        print(f"Rank {rank}: Initialization successful")
        # 分配内存
        torch.manual_seed(0)
    
        tensor = aclshmem_module.aclshmem_create_tensor([M, 2*N], dtype=torch.int8, device_id=rank)
        a = tensor[0:1, 0:N].fill_(2)
        b = tensor[0:1, N:2*N].fill_(0)
        torch.npu.synchronize()
        nelems = M * N
        # 将本卡数据put到另一张卡上，这里设置为下一张卡
        newPe = (rank + 1) % npu_num

        func = shmem_ub_put_nbi(M, N, nelems, newPe)
        b = func(a)
        barrier.wait()
        # print("b after=", b)
        if torch.equal(a, b):
            print("Kernel Output Match!")
        else:
            print("[ERROR] Kernel Output Not Match!")
        aclshmem_module.aclshmem_free_tensor(tensor)

    else:
        print(f"Rank {rank}: Initialization failed with code {ret}")    
    # 清理
    aclshmem_module.aclshmem_finialize()
    print(f"Rank {rank}: Finalized")

# 程序起始位置
print(f"Number of processes: {num_processes}")

barrier = Barrier(num_processes)  # 同步指定数量的进程
processes = []

for rank in range(num_processes):
    p = mp.Process(target=worker, args=(rank, barrier))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

print("All processes completed")