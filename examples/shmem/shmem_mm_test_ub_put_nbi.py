import argparse

import tilelang
import tilelang.language as T
import torch
import aclshmem as aclshmem_module
import multiprocessing as mp
from multiprocessing import Barrier
import sys
import os

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=1, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=352, help="Matrix N dimension")
parser.add_argument("--rank_table_path", type=str, default="", help="Matrix N dimension")
parser.add_argument("--num_processes", type=int, default=8, help="Matrix N dimension")
args = parser.parse_args()

M = args.m
N = args.n
M = 1
N = 352
rank_table_path = args.rank_table_path
num_processes = args.num_processes

@tilelang.jit()
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
                    T.set_flag("mte2", "mte3", 0x7)
                    T.wait_flag("mte2", "mte3", 0x7)
                    T.shmem_ub_put_nbi(ub_tensor, B, nelems, newPe)
                    T.pipe_barrier("mte3")
    return main

def worker(rank, barrier, rank_table_path):
    print(f"Rank {rank}: Setting device")
    torch.npu.set_device(rank)
    # 创建初始化属性对象
    attr = aclshmem_module.AclShmemInitAttr()
    npu_num = 8
    attr.rank_size = npu_num
    attr.rank = rank
    
    # 设置rank_table_path - 现在可以直接赋值字符串
    attr.rank_table_path = rank_table_path
    
    # 设置root_info - 现在可以直接赋值字符串
    attr.root_info = "initialization_data"
    
    # 初始化aclshmem
    print(f"Rank {rank}: Initializing aclshmem...")
    result = aclshmem_module.aclshmem_init_attr(1, attr)  # flags=1
    
    if result == 0:
        print(f"Rank {rank}: Initialization successful")
        # 分配内存
        torch.manual_seed(0)
    
        mem_type=aclshmem_module.MemType.DEVICE_SIDE
        tensor = aclshmem_module.aclshmem_create_tensor([M, 2*N], dtype=torch.int8, 
                                        mem_type=mem_type, device_id=rank)
        a = tensor[0:1, 0:N].fill_(2)
        b = tensor[0:1, N:2*N].fill_(0)
        torch.npu.synchronize()
        nelems = M * N
        # 将本卡数据put到另一张卡上，这里设置为下一张卡
        newPe = (rank + 1) % npu_num

        func = shmem_ub_put_nbi(M, N, nelems, newPe)
        func(a, b)
        barrier.wait()
        print("b after=", b)
        if torch.equal(a, b):
            print("Kernel Output Match!")
        else:
            print("[ERROR] Kernel Output Not Match!")
        aclshmem_module.aclshmem_free_tensor(tensor, mem_type=mem_type)

    else:
        print(f"Rank {rank}: Initialization failed with code {result}")
    
    # 清理
    aclshmem_module.aclshmem_finalize()
    print(f"Rank {rank}: Finalized")

# 程序起始位置
if not os.path.exists(rank_table_path):
    print(f"Error: Rank table file not found: {rank_table_path}")
    print("Please provide a valid rank table file path")
    sys.exit(1)
    
print(f"Using rank table: {rank_table_path}")
print(f"Number of processes: {num_processes}")

barrier = Barrier(num_processes)  # 同步指定数量的进程
processes = []

for rank in range(num_processes):
    p = mp.Process(target=worker, args=(rank, barrier, rank_table_path))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

print("All processes completed")