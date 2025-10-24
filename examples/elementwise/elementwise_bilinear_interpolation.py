import argparse

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
args = parser.parse_args()

M = args.m
N = args.n


@tilelang.jit(out_idx=[-1])
def bilinear_interpolation(M, N, block_M, block_N, dtype="float16"):
    M, N = src0.shape
    BLOCK_SIZE = 8
    OFFSET_GROUP_SIZE = 8  # 每组8个偏移值
    
    print(f"输入: src0={src0.shape}, src0Offset={src0Offset.shape}, src1={src1.shape}")
    print(f"参数: hRepeat={hRepeat}, vRepeat={vRepeat}, repeatMode={repeatMode}")
    
    # 初始化输出 - 确保与输入同形状
    dst = torch.zeros(M, N, dtype=torch.float16, device=src0.device)
    
    # 关键：重新解释src0Offset的形状
    # src0Offset是(M, N)，但实际存储的是偏移值组
    # 每组8个偏移值，所以总组数 = (M * N) // 8
    total_offset_groups = (M * N) // OFFSET_GROUP_SIZE
    
    # 将src0Offset重塑为 [total_offset_groups, 8]
    offset_reshaped = src0Offset.reshape(-1, OFFSET_GROUP_SIZE)
    print(f"Offset重塑为: {offset_reshaped.shape}")
    
    # 垂直迭代
    for v in range(vRepeat):
        # 计算当前垂直迭代的dst起始行
        # vROffset=128 可能表示128字节，假设每个元素2字节(float16)，则64个元素
        dst_v_start = v * (vROffset // 2)  # 除以2因为float16是2字节
        
        # 水平迭代
        for h in range(hRepeat):
            # 处理每个偏移值组
            for group_idx in range(min(total_offset_groups, M * N // (BLOCK_SIZE * BLOCK_SIZE))):
                # 读取8个偏移值
                offset_vals = offset_reshaped[group_idx, :].long()  # [8]
                
                # 计算当前组对应的输出块位置
                # 假设按行优先排列
                block_i = (group_idx * hRepeat + h) % (M // BLOCK_SIZE)
                block_j = group_idx % (N // BLOCK_SIZE)
                
                # 根据repeatMode处理src1
                if not repeatMode:
                    # 从src1取一个值 - 使用当前块位置
                    src1_val = src1[block_i * BLOCK_SIZE, block_j * BLOCK_SIZE]
                else:
                    # 从src1取8个值 - 使用当前水平迭代的8个连续值
                    src1_start = h * BLOCK_SIZE
                    src1_vals = src1[block_i * BLOCK_SIZE, src1_start:src1_start+BLOCK_SIZE]
                
                # 处理8个DataBlock
                block_result = torch.zeros(BLOCK_SIZE, dtype=torch.float16, device=src0.device)
                
                for data_block_idx in range(BLOCK_SIZE):
                    # 获取当前DataBlock的起始行
                    src0_start_row = offset_vals[data_block_idx].item() % M
                    
                    # 读取DataBlock的8个连续元素
                    src0_col_start = block_j * BLOCK_SIZE
                    if src0_col_start + BLOCK_SIZE <= N:
                        src0_block = src0[src0_start_row, src0_col_start:src0_col_start+BLOCK_SIZE]
                    else:
                        # 处理边界情况
                        valid_cols = N - src0_col_start
                        src0_block = torch.zeros(BLOCK_SIZE, dtype=torch.float16, device=src0.device)
                        src0_block[:valid_cols] = src0[src0_start_row, src0_col_start:N]
                    
                    # 乘操作
                    if not repeatMode:
                        multiplied = src0_block * src1_val
                    else:
                        multiplied = src0_block * src1_vals[data_block_idx]
                    
                    # 累加到当前块结果
                    block_result += multiplied
                
                # 计算dst写入位置
                dst_row = dst_v_start + block_i * BLOCK_SIZE
                dst_col = block_j * BLOCK_SIZE
                
                # 确保不越界
                if dst_row < M and dst_col + BLOCK_SIZE <= N:
                    # 累加到输出
                    dst[dst_row, dst_col:dst_col+BLOCK_SIZE] += block_result
    
    return dst


func = bilinear_interpolation(M, N, 128, 256)

torch.manual_seed(0)

a = torch.randn(M, N).npu().to(torch.float16)
b = torch.randn(M, N).npu().to(torch.float16)
offset = torch.ones(M, N).npu().to(torch.uint32)

torch.npu.synchronize()
print("init successful!")

c = func(a, b, offset)

def cann_bilinear_ground_truth(src0, src0Offset, src1, mask=128, hRepeat=2, repeatMode=False, 
                                  dstBlkStride=1, vROffset=128, vRepeat=2):
    M, N = src0.shape
    BLOCK_SIZE = 8  # 每个DataBlock包含8个元素
    
    print(f"输入形状: src0={src0.shape}, src0Offset={src0Offset.shape}, src1={src1.shape}")
    print(f"参数: hRepeat={hRepeat}, vRepeat={vRepeat}, repeatMode={repeatMode}, vROffset={vROffset}")
    
    # 初始化输出张量 - 形状与输入相同
    dst = torch.zeros(M, N, dtype=torch.float16, device=src0.device)
    
    # 计算offset张量的有效形状 (H_offset, W_offset)
    # 假设offset张量被解释为(H_offset, W_offset, 8)的形状，每个位置存储8个偏移值
    H_offset = src0Offset.shape[0]
    W_offset = src0Offset.shape[1] // 8  # 每个位置8个偏移值
    
    print(f"Offset张量解释为: {H_offset} x {W_offset} 个位置, 每个位置8个偏移值")
    
    # 垂直迭代
    for v_iter in range(vRepeat):
        # 计算当前垂直迭代的dst起始行
        dst_v_start = v_iter * (vROffset // BLOCK_SIZE)
        
        # 水平迭代
        for h_iter in range(hRepeat):
            # 遍历offset张量的每个位置
            for i in range(H_offset):
                for j in range(W_offset):
                    # 从src0Offset读取8个偏移值
                    # offset张量的布局: [H_offset, W_offset * 8]
                    offset_start = j * 8
                    offset_vals = src0Offset[i, offset_start:offset_start+8].long()  # [8]个偏移值
                    
                    # 根据repeatMode处理src1
                    if not repeatMode:
                        # repeatMode=false: 从src1中取一个值
                        # 使用当前offset位置对应的src1值
                        src1_val = src1[i, j]  # 标量值
                        src1_data = src1_val
                    else:
                        # repeatMode=true: 从src1中取8个值
                        # 从当前水平迭代位置取8个连续值
                        src1_start = h_iter * BLOCK_SIZE
                        src1_data = src1[i, src1_start:src1_start+BLOCK_SIZE]  # [8]个值
                    
                    # 处理src0的8个DataBlock
                    current_result = torch.zeros(BLOCK_SIZE, dtype=torch.float16, device=src0.device)
                    
                    for block_idx in range(BLOCK_SIZE):
                        # 获取当前DataBlock的起始地址
                        src0_block_start = offset_vals[block_idx].item()
                        
                        # 从src0读取当前DataBlock的8个元素
                        # 假设DataBlock是连续的8个元素
                        src0_block = src0[src0_block_start, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE]  # [8]
                        
                        # 乘操作
                        if not repeatMode:
                            # 与src1的单个值相乘
                            multiplied = src0_block * src1_data
                        else:
                            # 与src1的对应值相乘
                            multiplied = src0_block * src1_data[block_idx]
                        
                        # 累加到当前结果 (按文档说明是累加)
                        current_result += multiplied
                    
                    # 计算dst写入位置
                    dst_h_pos = h_iter * W_offset * BLOCK_SIZE + j * BLOCK_SIZE
                    dst_v_pos = dst_v_start + i * BLOCK_SIZE
                    
                    # 确保不越界
                    if dst_v_pos < M and dst_h_pos + BLOCK_SIZE <= N:
                        # 与之前的结果累加 (按文档说明)
                        dst[dst_v_pos, dst_h_pos:dst_h_pos+BLOCK_SIZE] += current_result
    
    return dst

ref_c = cann_bilinear_ground_truth(a, offset, b)

torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")
