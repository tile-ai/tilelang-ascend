# Copyright (c) Huawei Technologies Co., Ltd. 2025.
# Import necessary libraries
import torch
import argparse

# Import tilelang modules for NPU kernel development
import tilelang
from tilelang import language as T
from tilelang import tvm

# Clear any cached kernels to ensure fresh compilation
tilelang.cache.clear_cache()

# Set the NPU device to use device ID 0
torch.npu.set_device(0)

# Create argument parser for configuring the sparse attention kernel
parser = argparse.ArgumentParser(description="NPU Kernel Compilation")

# Define command line arguments for kernel parameters
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for input data")
parser.add_argument("--seq_len", type=int, default=4096, help="Sequence length for query")
parser.add_argument("--seq_len_kv", type=int, default=32768, help="Sequence length for key-value")
parser.add_argument("--heads", type=int, default=128, help="Number of attention heads")
parser.add_argument("--dim", type=int, default=512, help="Dimension of query and key")
parser.add_argument("--tail_dim", type=int, default=64, help="Additional dimension for value")
parser.add_argument("--top_k", type=int, default=2048, help="Number of top-k elements to select")
parser.add_argument("--block_i", type=int, default=64, help="Block size for I dimension")
parser.add_argument("--block_k", type=int, default=64, help="Block size for K dimension")
parser.add_argument("--kv_group", type=int, default=1, help="KV group factor for grouped attention")
parser.add_argument("--num_kernels", type=int, default=24, help="Number of parallel kernels to launch")

def sparse_attention_mla(
        batch,
        seq_len,
        seq_len_kv,
        heads,
        dim,
        tail_dim,
        top_k,
        num_kernels,
        kv_group=1,
        sm_scale=None,
        is_causal=True,
        block_I=64,
        block_K=64,
):
    # Validate input parameters
    assert block_I % 2 == 0, "Block I size must be even"
    assert dim == tilelang.math.next_power_of_2(
        dim
    ), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(
        tail_dim
    ), f"haven't check padding correctness yet, dim={tail_dim}"
    assert is_causal == True, "non-casual is not supported"
    assert (
            top_k % block_I == 0
    ), "otherwise will load some index=0 thus causing wrong kv to be loaded"
    
    # Set softmax scale if not provided
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5
    else:
        sm_scale = sm_scale

    # The maximum pipline depth of the 910B/910C chip is 15
    FFTS_FLAG_THRESHOLD = 15

    # Calculate total number of logical kernels
    num_logic_kernels = batch * seq_len

    # Calculate number of KV heads
    head_kv = heads // kv_group

    # Define data types
    indices_dtype = "int32"
    dtype = "float16"
    accum_dtype = "float"

    # Calculate padded head dimension
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != head_kv:
        assert (
                kv_group == 1
        ), "here we solve the H padding automically, other wise you should handle Q copy and Output copy with your mask \
           (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automically)"

    # Calculate half block sizes for vector operations
    block_I_half = block_I // 2

    # Determine replication factor for heads
    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    # Set block size for head dimension
    block_H = padded_H if REPLICATE_H == 1 else 64
    block_H_half = block_H // 2

    # Calculate shared block size for I and K dimensions
    share_block_IK = max(block_I, block_K)

    # Calculate full dimension (dim + tail_dim)
    full_dim = dim + tail_dim
    
    # Define tensor shapes
    shape_q = [batch * seq_len * heads, full_dim]
    shape_kv = [batch * seq_len_kv * kv_group, full_dim]
    shape_out = [batch * seq_len * heads, dim]
    shape_idx = [batch * seq_len * kv_group * top_k]
    shape_kv_sparse_work = [num_kernels * top_k, full_dim]
    shape_s_work = [num_kernels * block_H, block_I]
    shape_o_work = [num_kernels * block_H, dim]

    # Precompute frequently used expressions
    heads_mul_seq_len = heads * seq_len
    top_k_mul_kv_group = top_k * kv_group
    top_k_mul_kv_group_mul_seq_len = top_k_mul_kv_group * seq_len
    kv_group_mul_seq_len_kv = kv_group * seq_len_kv

    # Define the main sparse attention kernel using TileLang
    @T.prim_func
    def main(
            Q: T.Tensor(shape_q, dtype),  # type: ignore
            KV: T.Tensor(shape_kv, dtype),  # type: ignore
            Indices: T.Tensor(shape_idx, indices_dtype),  # type: ignore
            Output: T.Tensor(shape_out, dtype),  # type: ignore
            workspace_0: T.Tensor(shape_kv_sparse_work, dtype),
            workspace_1: T.Tensor(shape_s_work, dtype),
            workspace_2: T.Tensor(shape_o_work, dtype),
    ):
        # Launch NPU kernel with specified number of parallel kernels
        with T.Kernel(num_kernels, is_npu=True) as (kernel_id, subid):
            acc_s_scale = sm_scale

            # Cube computation section (matrix operations)
            with T.Scope("Cube"):
                # Allocate L1 buffers for cube operations
                l1_q = T.alloc_L1([block_H, block_K], dtype)
                l1_p = T.alloc_L1([block_H, block_I], dtype)
                l1_kv_sparse = T.alloc_L1([block_I, block_K], dtype)

                # Allocate L0 buffer for accumulation
                l0_c = T.alloc_L0C([block_H, share_block_IK], accum_dtype)

                # Process tasks in serial across logical kernels
                for task_id in T.serial(T.ceildiv(num_logic_kernels, num_kernels)):
                    logic_kernel_id = task_id * num_kernels
                    logic_kernel_id = logic_kernel_id + kernel_id
                    if logic_kernel_id < num_logic_kernels:
                        # Calculate batch and sequence indices
                        batch_id = logic_kernel_id // seq_len
                        seq_id = logic_kernel_id % seq_len

                        # Process blocks along head dimension
                        for block_h_id in T.serial(T.ceildiv(heads, block_H)):
                            block_h_offset = block_h_id * block_H

                            # Process blocks along I dimension (top-k)
                            for block_i_id in T.serial(T.ceildiv(top_k, block_I)):
                                block_i_offset = block_i_id * block_I

                                # Wait for Vector computation section synchronization
                                with T.rs("PIPE_MTE2"):
                                    T.sync_block_wait(0)

                                # Calculate synchronization counter for cube operations
                                stride = T.ceildiv(heads, block_H)
                                sync_count_cube = task_id * stride
                                sync_count_cube = sync_count_cube + block_h_id
                                stride = T.ceildiv(top_k, block_I)
                                sync_count_cube = sync_count_cube * stride
                                sync_count_cube = sync_count_cube + block_i_id
                                sync_count_cube = sync_count_cube % FFTS_FLAG_THRESHOLD
                                temp = FFTS_FLAG_THRESHOLD - 1
                                if sync_count_cube == temp:
                                    with T.rs("PIPE_MTE3"):
                                        T.sync_block_set(1)

                                # Process blocks along K dimension
                                for block_k_id in T.serial(T.ceildiv(full_dim, block_K)):
                                    block_k_offset = block_k_id * block_K
                                    tail_size_k = full_dim - block_k_offset
                                    tail_size_k = T.min(tail_size_k, block_K)

                                    # Load sparse KV data to L1
                                    offset = kernel_id * top_k
                                    offset = offset + block_i_offset
                                    T.npuir_load_nd2nz(workspace_0[offset, block_k_offset], l1_kv_sparse,
                                                       size=[block_I, tail_size_k])

                                    # Load query data to L1
                                    offset = batch_id * seq_len
                                    offset = offset + seq_id
                                    offset = offset * heads
                                    offset = offset + block_h_offset

                                    T.npuir_load_nd2nz(Q[offset, block_k_offset], l1_q,
                                                       size=[block_H, tail_size_k])

                                    # Perform matrix multiplication (Q @ K^T)
                                    if block_k_id == 0:
                                        T.npuir_dot(l1_q, l1_kv_sparse, l0_c, initC=True, b_transpose=True,
                                                    size=[block_H, tail_size_k, block_I])
                                    else:
                                        T.npuir_dot(l1_q, l1_kv_sparse, l0_c, initC=False, b_transpose=True,
                                                    size=[block_H, tail_size_k, block_I])

                                # Store intermediate results and synchronize
                                with T.rs("PIPE_FIX"):
                                    offset = kernel_id * block_H
                                    T.npuir_store_fixpipe(l0_c, workspace_1[offset, 0], size=[block_H, block_I],
                                                          enable_nz2nd=True)
                                    T.sync_block_set(0)

                                # Load intermediate results for softmax
                                with T.rs("PIPE_MTE2"):
                                    T.sync_block_wait(0)
                                    offset = kernel_id * block_H
                                    T.npuir_load_nd2nz(workspace_1[offset, 0], l1_p, size=[block_H, block_I])

                                # Process blocks for output computation
                                for block_k_id in T.serial(T.ceildiv(dim, block_K)):
                                    block_k_offset = block_k_id * block_K
                                    tail_size_k = dim - block_k_offset
                                    tail_size_k = T.min(tail_size_k, block_K)

                                    # Load sparse KV data for output computation
                                    offset_1 = kernel_id * top_k
                                    offset = offset_1 + block_i_offset
                                    T.npuir_load_nd2nz(workspace_0[offset, block_k_offset], l1_kv_sparse,
                                                       size=[block_I, tail_size_k])

                                    # Perform matrix multiplication (P @ V)
                                    T.npuir_dot(l1_p, l1_kv_sparse, l0_c, initC=True, size=[block_H, block_I, tail_size_k])
                                    offset = kernel_id * block_H
                                    T.npuir_store_fixpipe(l0_c, workspace_2[offset, block_k_offset],
                                                          size=[block_H, tail_size_k], enable_nz2nd=True)

                                # Synchronize after output computation
                                with T.rs("PIPE_FIX"):
                                    T.sync_block_set(0)

            # Vector computation section (softmax and normalization)
            with T.Scope("Vector"):
                # Allocate unified buffers for vector operations
                ub_kv_sparse = T.alloc_ub([block_I_half, full_dim], dtype)
                ub_indices = T.alloc_ub([block_I_half], indices_dtype)
                ub_acc_o = T.alloc_ub([block_H_half, dim], accum_dtype)
                ub_acc_o_new = T.alloc_ub([block_H_half, dim], accum_dtype)
                ub_cross_kernel_16 = T.alloc_ub([block_H_half, share_block_IK], dtype)
                ub_cross_kernel_32 = T.alloc_ub([block_H_half, share_block_IK], accum_dtype)

                # Allocate buffers for softmax variables
                ub_var_logsum = T.alloc_ub([block_H_half, 1], accum_dtype)
                ub_var_scores_max = T.alloc_ub([block_H_half, 1], accum_dtype)
                ub_var_scores_max_prev = T.alloc_ub([block_H_half, 1], accum_dtype)
                ub_var_scores_scale = T.alloc_ub([block_H_half, 1], accum_dtype)
                ub_var_scores_sum = T.alloc_ub([block_H_half, 1], accum_dtype)
                
                # Process tasks in serial across logical kernels
                for task_id in T.serial(T.ceildiv(num_logic_kernels, num_kernels)):
                    logic_kernel_id = task_id * num_kernels
                    logic_kernel_id = logic_kernel_id + kernel_id
                    if logic_kernel_id < num_logic_kernels:
                        batch_id = logic_kernel_id // seq_len
                        seq_id = logic_kernel_id % seq_len

                        # Process blocks along head dimension
                        for block_h_id in T.serial(T.ceildiv(heads, block_H)):
                            block_h_offset = block_h_id * block_H
                            block_h_offset_sub = subid * block_H_half
                            block_h_offset = block_h_offset + block_h_offset_sub

                            # Initialize softmax variables
                            value_zero = 0
                            value_min = -T.infinity("float32")
                            T.npuir_brc(value_zero, ub_var_logsum)
                            T.npuir_brc(value_zero, ub_acc_o)
                            T.npuir_brc(value_zero, ub_var_scores_scale)
                            T.npuir_brc(value_min, ub_var_scores_max)

                            # Process blocks along I dimension (top-k)
                            for block_i_id in T.serial(T.ceildiv(top_k, block_I)):
                                block_i_offset = block_i_id * block_I
                                block_i_offset_sub = subid * block_I_half
                                block_i_offset = block_i_offset + block_i_offset_sub

                                # Load indices and gather sparse KV data (only for first head block)
                                if block_h_id == 0:
                                    offset_2 = seq_id * top_k_mul_kv_group
                                    offset_1 = batch_id * top_k_mul_kv_group_mul_seq_len
                                    offset = offset_1 + offset_2
                                    offset = offset + block_i_offset
                                    T.copy(Indices[offset], ub_indices, size=[block_I_half])
                                    for idx_id in T.serial(block_I_half):
                                        current_index = ub_indices[idx_id]
                                        offset = batch_id * kv_group_mul_seq_len_kv
                                        offset = offset + current_index
                                        T.copy(KV[offset, 0], ub_kv_sparse[idx_id, 0], size=[1, full_dim])

                                    # Store gathered KV data to workspace
                                    offset = kernel_id * top_k
                                    offset = offset + block_i_offset
                                    T.copy(ub_kv_sparse, workspace_0[offset, 0], size=[block_I_half, full_dim])

                                # Synchronize after KV gathering
                                with T.rs("PIPE_MTE3"):
                                    T.sync_block_set(0)

                                # Calculate synchronization counter for vector operations
                                stride = T.ceildiv(heads, block_H)
                                sync_count_vec = task_id * stride
                                sync_count_vec = sync_count_vec + block_h_id
                                stride = T.ceildiv(top_k, block_I)
                                sync_count_vec = sync_count_vec * stride
                                sync_count_vec = sync_count_vec + block_i_id
                                sync_count_vec = sync_count_vec % FFTS_FLAG_THRESHOLD
                                temp = FFTS_FLAG_THRESHOLD - 1
                                if sync_count_vec == temp:
                                    with T.rs("PIPE_MTE3"):
                                        T.sync_block_wait(1)

                                # Save previous max scores for numerical stability
                                T.copy(ub_var_scores_max, ub_var_scores_max_prev)

                                # Load attention scores from workspace
                                offset = kernel_id * block_H
                                offset_sub = subid * block_H_half
                                offset = offset + offset_sub
                                with T.rs("PIPE_MTE2"):
                                    T.sync_block_wait(0)
                                    T.copy(workspace_1[offset, 0], ub_cross_kernel_16, size=[block_H_half, block_I])

                                # Apply softmax scaling and compute max
                                T.npuir_cast(ub_cross_kernel_16, ub_cross_kernel_32, round_mode="rint", size=[block_H_half, block_I])
                                T.npuir_mul(ub_cross_kernel_32, acc_s_scale, ub_cross_kernel_32)
                                T.npuir_reduce(ub_cross_kernel_32, ub_var_scores_max, dims=[1], reduce_mode="max")
                                
                                # Update max scores and compute scaling factors
                                if block_i_id != 0:
                                    T.npuir_max(ub_var_scores_max_prev, ub_var_scores_max, ub_var_scores_max)
                                    T.npuir_sub(ub_var_scores_max_prev, ub_var_scores_max, ub_var_scores_scale)
                                    T.npuir_exp(ub_var_scores_scale, ub_var_scores_scale)
                                
                                # Apply softmax stabilization and compute exponentials
                                T.npuir_sub(ub_cross_kernel_32, ub_var_scores_max, ub_cross_kernel_32)
                                T.npuir_exp(ub_cross_kernel_32, ub_cross_kernel_32)
                                T.npuir_cast(ub_cross_kernel_32, ub_cross_kernel_16, round_mode="rint", size=[block_H_half, block_I])

                                # Store softmax results and synchronize
                                with T.rs("PIPE_MTE3"):
                                    T.copy(ub_cross_kernel_16, workspace_1[offset, 0], size=[block_H_half, block_I])
                                    T.sync_block_set(0)

                                # Compute sum of exponentials for softmax denominator
                                T.npuir_reduce(ub_cross_kernel_32, ub_var_scores_sum, dims=[1], reduce_mode="sum")

                                # Update logsum and accumulate output
                                T.npuir_mul(ub_var_logsum, ub_var_scores_scale, ub_var_logsum)
                                T.npuir_add(ub_var_logsum, ub_var_scores_sum, ub_var_logsum)
                                T.npuir_mul(ub_acc_o, ub_var_scores_scale, ub_acc_o)

                                # Load and accumulate output values
                                with T.rs("PIPE_MTE2"):
                                    T.sync_block_wait(0)
                                    for block_k_id in T.serial(T.ceildiv(dim, block_K)):
                                        block_k_offset = block_k_id * block_K
                                        tail_size_k = dim - block_k_offset
                                        tail_size_k = T.min(tail_size_k, block_K)

                                        T.copy(workspace_2[offset, block_k_offset], ub_cross_kernel_16,
                                               size=[block_H_half, tail_size_k])
                                        T.npuir_cast(ub_cross_kernel_16, ub_cross_kernel_32, round_mode="rint")
                                        T.copy(ub_cross_kernel_32, ub_acc_o_new[0, block_k_offset],
                                               size=[block_H_half, tail_size_k])

                                T.npuir_add(ub_acc_o, ub_acc_o_new, ub_acc_o)

                            # Normalize output by softmax denominator
                            T.npuir_div(ub_acc_o, ub_var_logsum, ub_acc_o)
                            
                            # Store final output results
                            for block_k_id in T.serial(T.ceildiv(dim, block_K)):
                                block_k_offset = block_k_id * block_K
                                tail_size_k = dim - block_k_offset
                                tail_size_k = T.min(tail_size_k, block_K)

                                T.npuir_cast(ub_acc_o[0, block_k_offset], ub_cross_kernel_16, round_mode="rint",
                                             size=[block_H_half, tail_size_k])

                                offset_2 = seq_id * heads
                                offset_1 = batch_id * heads_mul_seq_len
                                offset = offset_1 + offset_2
                                offset = offset + block_h_offset
                                T.copy(ub_cross_kernel_16, Output[offset, block_k_offset], size=[block_H_half, tail_size_k])

    return main


def run_generate(args):
    """Generate and compile the sparse attention kernel"""
    # Create the sparse attention function with given arguments
    func = sparse_attention_mla(batch=args.batch_size,
                                seq_len=args.seq_len,
                                seq_len_kv=args.seq_len_kv,
                                heads=args.heads,
                                dim=args.dim,
                                tail_dim=args.tail_dim,
                                top_k=args.top_k,
                                num_kernels=args.num_kernels,
                                kv_group=args.kv_group,
                                block_I=args.block_i,
                                block_K=args.block_k,
                                )

    # Lower the function to NPU kernel
    kernel = tilelang.engine.lower(func)
    print(kernel)

    # Export to .mlir file
    file_name = './mlir_files/sparse_attn_mla_npuir_ver2.mlir'
    with open(file_name, 'w') as f:
        f.write(kernel)


def generate_tensor(shape, dtype, clear=False):
    """Generate tensor with specified shape and data type"""
    if clear:
        return torch.zeros(shape, dtype=eval("torch." + dtype))
    if dtype in ("float32", "float16", "bfloat16"):
        return torch.randn(size=shape, dtype=eval("torch." + dtype))
    if dtype in ("int32", "int64", "int16"):
        return torch.randint(low=0, high=2000, size=shape, dtype=eval("torch." + dtype))
    if dtype == "int8":
        return torch.randint(low=0, high=127, size=shape, dtype=eval("torch." + dtype))
    if dtype == "bool":
        return torch.randint(low=0, high=2, size=shape).bool()
    raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))


def gather_from_kv(KV, indices):
    """Gather key-value pairs using indices for reference implementation"""
    b, s1, g, k = indices.shape
    batch_idx = torch.arange(b, device=KV.device).view(b, 1, 1).expand(-1, s1, k)
    indices_flat = indices.squeeze(2).long()
    out = KV[batch_idx, indices_flat, 0, :].squeeze(dim=3)

    return out


def run_test(args):
    """Test the compiled sparse attention kernel against reference implementation"""
    # Create the sparse attention function
    func = sparse_attention_mla(batch=args.batch_size,
                                seq_len=args.seq_len,
                                seq_len_kv=args.seq_len_kv,
                                heads=args.heads,
                                dim=args.dim,
                                tail_dim=args.tail_dim,
                                top_k=args.top_k,
                                num_kernels=args.num_kernels,
                                kv_group=args.kv_group,
                                block_I=args.block_i,
                                block_K=args.block_k,
                                )

    # Compile the function to NPU kernel
    compiled_kernel = tilelang.compile(func, target='npuir')

    # Set random seed for reproducibility
    torch.manual_seed(88888888)

    # Define data types
    dtype = "float16"
    accum_dtype = "float32"
    indices_dtype = "int32"

    # Calculate kernel parameters
    head_kv = args.heads // args.kv_group
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)

    if head_kv > 64:
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    block_H = padded_H if REPLICATE_H == 1 else 64

    # Calculate dimensions and shapes
    full_dim = args.dim + args.tail_dim
    shape_q = [args.batch_size, args.seq_len, args.heads, full_dim]
    shape_kv = [args.batch_size, args.seq_len_kv, args.kv_group, full_dim]
    shape_out = [args.batch_size, args.seq_len, args.heads, args.dim]
    shape_idx = [args.batch_size, args.seq_len, args.kv_group, args.top_k]
    shape_kv_sparse_work = [args.num_kernels, args.top_k, full_dim]
    shape_s_work = [args.num_kernels, block_H, args.block_i]
    shape_o_work = [args.num_kernels, block_H, args.dim]

    # Generate test tensors
    q = generate_tensor(shape_q, dtype).npu()
    kv = generate_tensor(shape_kv, dtype).npu()
    indices = generate_tensor(shape_idx, indices_dtype).npu()
    o = generate_tensor(shape_out, dtype, clear=True).npu()
    w0 = generate_tensor(shape_kv_sparse_work, dtype, clear=True).npu()
    w1 = generate_tensor(shape_s_work, dtype, clear=True).npu()
    w2 = generate_tensor(shape_o_work, dtype, clear=True).npu()

    # Compute reference output using standard PyTorch operations
    ref_kv_sparse = gather_from_kv(kv, indices)
    scale = (1.0 / (args.dim + args.tail_dim)) ** 0.5
    ref_o = torch.nn.functional.softmax((q @ ref_kv_sparse.transpose(-2, -1)).to(torch.float32) * scale, dim=-1).to(
        torch.float16) @ ref_kv_sparse[:, :, :, :args.dim]

    # Execute the compiled kernel
    compiled_kernel(q, kv, indices, o, w0, w1, w2)

    # Print and compare results
    torch.set_printoptions(sci_mode=False)
    print("Actual Result:")
    print(o)
    print("Expected Result:")
    print(ref_o)
    
    # Verify correctness
    torch.testing.assert_close(o, ref_o, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    # Parse command line arguments and run tests
    main_args = parser.parse_args()
    run_test(main_args)
