import argparse
import tilelang
from tilelang import language as T
import torch

@tilelang.jit(out_idx=[1])
def broadcast_pipeline(M, N, block_M, sub_M, dtype="float"):
    m_num = M // block_M  
    VEC_NUM = 2
    stages = 2 
    

    sub_block_M = sub_M // VEC_NUM
    
    @T.prim_func
    def main(
        A: T.Tensor([1, N], dtype),   
        B: T.Tensor([M, N], dtype), 
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((stages, 1, N), dtype)
            b_ub = T.alloc_ub((stages, sub_block_M, N), dtype)
            tmp = T.alloc_ub((2 * block_M, N), "uint8")
            
            row_base = cid * block_M
            
            with T.Scope("V"):
                T.barrier_all()
                
                T.copy(A[0, :], a_ub[0, 0, :])
                T.barrier_all()
                
                for stage in T.serial(block_M // sub_M):
                    cur = stage % stages
                    nxt = (stage + 1) % stages
                    
                    if stage < (block_M // sub_M) - 1:
                        T.barrier_all()
                        T.copy(A[0, :], a_ub[nxt, 0, :])
                        T.barrier_all()
                    
                    T.barrier_all()
                    
                    cur_row_start = row_base + stage * sub_M
                    
                    T.tile.broadcast(
                        b_ub[cur, :, :],      
                        a_ub[cur, 0, :],    
                        tmp 
                    )
                    
                    T.barrier_all()
                    
                    T.copy(
                        b_ub[cur, :, :],
                        B[cur_row_start + vid * sub_block_M:cur_row_start + (vid + 1) * sub_block_M, :]
                    )
                    
                    T.barrier_all()
                    
    return main

if __name__ == "__main__":
    tilelang.cache.clear_cache()
    
    parser = argparse.ArgumentParser(description="Broadcast Pipeline NPU Kernel Compilation")
    parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
    parser.add_argument("--n", type=int, default=256, help="Matrix N dimension")
    parser.add_argument("--block-m", type=int, default=128, help="Block size in M dimension")
    parser.add_argument("--sub-m", type=int, default=64, help="Sub-block size for pipeline")
    args = parser.parse_args()
    
    M = args.m
    N = args.n
    block_M = args.block_m
    sub_M = args.sub_m
    
    print(f"Configuration: M={M}, N={N}, block_M={block_M}, sub_M={sub_M}")
    
    func = broadcast_pipeline(M, N, block_M, sub_M)
    
    torch.manual_seed(0)
    
    a = torch.randn(1, N).npu()
    
    torch.npu.synchronize()
    print("init successful!")
    
    c = func(a)
    
    ref_c = a.expand(M, N)
    
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Kernel Output Match!")