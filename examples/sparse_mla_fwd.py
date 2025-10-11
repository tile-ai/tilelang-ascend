import torch
torch.npu.set_device(0)
import argparse

import tilelang
from tilelang import language as T
from tilelang import tvm

# Clear tilelang cache
tilelang.cache.clear_cache()

# Argument parser for kernel configuration parameters
parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--batch_size", type=int, default=1, help="")
parser.add_argument("--seq_len", type=int, default=4096, help="")
parser.add_argument("--seq_len_kv", type=int, default=4096, help="")
parser.add_argument("--heads", type=int, default=128, help="")
parser.add_argument("--dim", type=int, default=512, help="")
parser.add_argument("--tail_dim", type=int, default=64, help="")
parser.add_argument("--top_k", type=int, default=2048, help="")
parser.add_argument("--block_i", type=int, default=64, help="")
parser.add_argument("--block_k", type=int, default=64, help="")
parser.add_argument("--kv_group", type=int, default=1, help="")
parser.add_argument("--num_kernels", type=int, default=24, help="")
parser.add_argument("--sm_scale", type=float, help="")


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
    assert block_I % 2 == 0, ""
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

    # Fast thread synchronization threshold
    FFTS_FLAG_THRESHOLD = 15

    # Calculate number of logical kernels
    num_logic_kernels = batch * seq_len
    head_kv = heads // kv_group

    # Data type definitions
    indices_dtype = "int32"
    dtype = "float16"
    accum_dtype = "float"

    # Calculate padded head dimension
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != head_kv:
        assert (
                kv_group == 1
        ), "here we solve the H padding automically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automically)"

    # Calculate half block sizes
    block_I_half = block_I // 2

    # Determine head replication factor
    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    # Set block sizes for head dimension
    block_H = padded_H if REPLICATE_H == 1 else 64
    block_H_half = block_H // 2

    # Shared block size for I and K dimensions
    share_block_IK = max(block_I, block_K)

    # Calculate full dimension (query + tail)
    full_dim = dim + tail_dim
    
    # Define tensor shapes
    shape_q = [batch * seq_len * heads, full_dim]
    shape_kv = [batch * seq_len_kv * kv_group, full_dim]
    shape_out = [batch * seq_len * heads, dim]
    shape_idx = [batch * seq_len * kv_group * top_k]
    shape_kv_sparse_work = [num_kernels * top_k, full_dim]

    shape_s_work = [num_kernels * block_H, block_I]
    shape_o_work = [num_kernels * block_H, dim]

    # Precompute constants for indexing
    heads_mul_seq_len = heads * seq_len
    top_k_mul_kv_group = top_k * kv_group
    top_k_mul_kv_group_mul_seq_len = top_k_mul_kv_group * seq_len
    kv_group_mul_seq_len_kv = kv_group * seq_len_kv

    @T.prim_func
    def main(
            Q: T.Tensor(shape_q, dtype),
            KV: T.Tensor(shape_kv, dtype),
            Indices: T.Tensor(shape_idx, indices_dtype),
            Output: T.Tensor(shape_out, dtype),
            workspace_kv: T.Tensor(shape_kv_sparse_work, dtype),
            workspace_1: T.Tensor(shape_s_work, dtype),
            workspace_2: T.Tensor(shape_o_work, dtype),
    ):
        # Kernel definition with NPU support
        with T.Kernel(num_kernels, is_npu=True) as (kernel_id, subid):
            acc_s_scale = sm_scale
            
            # Cube computation scope for matrix operations
            with T.Scope("Cube"):
                # Local memory allocations for cube operations
                l1_q = T.alloc_L1([block_H, block_K], dtype)
                l1_p = T.alloc_L1([block_H, block_I], dtype)
                l1_kv_sparse = T.alloc_L1([block_I, block_K], dtype)

                l0_c = T.alloc_L0C([block_H, share_block_IK], accum_dtype)

                # Process logical kernels in parallel
                for task_id in T.serial(T.ceildiv(num_logic_kernels, num_kernels)):
                    logic_kernel_id = task_id * num_kernels
                    logic_kernel_id = logic_kernel_id + kernel_id
                    if logic_kernel_id < num_logic_kernels:
                        batch_id = logic_kernel_id // seq_len
                        seq_id = logic_kernel_id % seq_len

                        # Process head blocks
                        for block_h_id in T.serial(T.ceildiv(heads, block_H)):
                            block_h_offset = block_h_id * block_H

                            # Process top-k blocks
                            for block_i_id in T.serial(T.ceildiv(top_k, block_I)):
                                block_i_offset = block_i_id * block_I

                                with T.rs("PIPE_MTE2"):
                                    T.sync_block_wait(0)

                                # Calculate synchronization count for cube operations
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
                                        T.sync_block_set(0)

                                # Process K dimension blocks
                                for block_k_id in T.serial(T.ceildiv(full_dim, block_K)):
                                    block_k_offset = block_k_id * block_K
                                    tail_size_k = full_dim - block_k_offset
                                    tail_size_k = T.min(tail_size_k, block_K)

                                    # Load sparse KV values to local memory
                                    offset = kernel_id * top_k
                                    offset = offset + block_i_offset
                                    T.npuir_load_nd2nz(workspace_kv[offset, block_k_offset], l1_kv_sparse,
                                                       size=[block_I, tail_size_k])

                                    # Load query values to local memory
                                    offset = batch_id * seq_len
                                    offset = offset + seq_id
                                    offset = offset * heads
                                    offset = offset + block_h_offset

                                    T.npuir_load_nd2nz(Q[offset, block_k_offset], l1_q,
                                                       size=[block_H, tail_size_k])

                                    # Matrix multiplication: Q * K^T
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

                                # Load intermediate results for next computation
                                with T.rs("PIPE_MTE2"):
                                    T.sync_block_wait(0)
                                    offset = kernel_id * block_H
                                    T.npuir_load_nd2nz(workspace_1[offset, 0], l1_p, size=[block_H, block_I])

                                # Matrix multiplication: P * V
                                for block_k_id in T.serial(T.ceildiv(dim, block_K)):
                                    block_k_offset = block_k_id * block_K
                                    tail_size_k = dim - block_k_offset
                                    tail_size_k = T.min(tail_size_k, block_K)

                                    offset_1 = kernel_id * top_k
                                    offset = offset_1 + block_i_offset
                                    T.npuir_load_nd2nz(workspace_kv[offset, block_k_offset], l1_kv_sparse,
                                                       size=[block_I, tail_size_k])

                                    T.npuir_dot(l1_p, l1_kv_sparse, l0_c, initC=True,
                                                size=[block_H, block_I, tail_size_k])
                                    offset = kernel_id * block_H
                                    T.npuir_store_fixpipe(l0_c, workspace_2[offset, block_k_offset],
                                                              size=[block_H, tail_size_k], enable_nz2nd=True)

                                with T.rs("PIPE_FIX"):
                                    T.sync_block_set(0)

            # Vector computation scope for softmax and normalization
            with T.Scope("Vector"):
                # Unified buffer allocations for vector operations
                ub_kv_sparse = T.alloc_ub([block_I_half, full_dim], dtype)
                ub_indices = T.alloc_ub([block_I_half], indices_dtype)
                ub_acc_o = T.alloc_ub([block_H_half, dim], accum_dtype)
                ub_acc_o_new = T.alloc_ub([block_H_half, dim], accum_dtype)
                ub_cross_kernel_16 = T.alloc_ub([block_H_half, share_block_IK], dtype)
                ub_cross_kernel_32 = T.alloc_ub([block_H_half, share_block_IK], accum_dtype)

                # Softmax-related buffers
                ub_var_logsum = T.alloc_ub([block_H_half, 1], accum_dtype)
                ub_var_scores_max = T.alloc_ub([block_H_half, 1], accum_dtype)
                ub_var_scores_max_prev = T.alloc_ub([block_H_half, 1], accum_dtype)
                ub_var_scores_scale = T.alloc_ub([block_H_half, 1], accum_dtype)
                ub_var_scores_sum = T.alloc_ub([block_H_half, 1], accum_dtype)
                ub_var_valid_mask = T.alloc_ub([1, block_I], accum_dtype)

                # Constants for numerical stability
                value_minimum = -T.infinity("float32")
                value_eps = 0.0005
                value_zero = 0

                # Process logical kernels in parallel
                for task_id in T.serial(T.ceildiv(num_logic_kernels, num_kernels)):
                    logic_kernel_id = task_id * num_kernels
                    logic_kernel_id = logic_kernel_id + kernel_id
                    if logic_kernel_id < num_logic_kernels:
                        batch_id = logic_kernel_id // seq_len
                        seq_id = logic_kernel_id % seq_len

                        # Calculate actual top-k for current sequence position
                        available_top_k = T.max(seq_id, 1)
                        real_top_k = T.min(available_top_k, top_k)

                        # Create valid mask for incomplete blocks
                        valid_mod = real_top_k % block_I
                        T.npuir_brc(value_zero, ub_var_valid_mask)
                        for idx in T.serial(valid_mod):
                            tmp1 = 0.1
                            tmp2 = 1.1
                            tmp3 = tmp2 - tmp1
                            ub_var_valid_mask[0, idx] = tmp3

                        # Process head blocks
                        for block_h_id in T.serial(T.ceildiv(heads, block_H)):
                            block_h_offset = block_h_id * block_H
                            block_h_offset_sub = subid * block_H_half
                            block_h_offset = block_h_offset + block_h_offset_sub

                            # Initialize accumulation buffers
                            T.npuir_brc(value_zero, ub_var_logsum)
                            T.npuir_brc(value_zero, ub_acc_o)
                            T.npuir_brc(value_zero, ub_var_scores_scale)
                            T.npuir_brc(value_minimum, ub_var_scores_max)

                            # Process top-k blocks for softmax computation
                            for block_i_id in T.serial(T.ceildiv(top_k, block_I)):
                                block_i_offset = block_i_id * block_I
                                tail_size_i = real_top_k - block_i_offset
                                tail_size_i = T.min(tail_size_i, block_I)
                                tail_size_i_half = tail_size_i + 1
                                tail_size_i_half = tail_size_i_half // 2

                                block_i_offset_sub = subid * tail_size_i_half
                                block_i_offset = block_i_offset + block_i_offset_sub

                                tail_size_i_mod2 = tail_size_i % 2
                                incomplete_block_i = tail_size_i_mod2 * subid
                                tail_size_i_half = tail_size_i_half - incomplete_block_i

                                # Gather sparse KV values using indices
                                if block_h_id == 0:
                                    T.npuir_brc(value_zero, ub_kv_sparse)
                                    if tail_size_i_half > 0:
                                        offset_2 = seq_id * top_k_mul_kv_group
                                        offset_1 = batch_id * top_k_mul_kv_group_mul_seq_len
                                        offset = offset_1 + offset_2
                                        offset = offset + block_i_offset
                                        T.copy(Indices[offset], ub_indices, size=[block_I_half])
                                        for idx_id in T.serial(block_I_half):
                                            current_index = ub_indices[idx_id]
                                            offset = batch_id * kv_group_mul_seq_len_kv
                                            offset = offset + current_index
                                            if current_index < seq_len_kv:
                                                T.copy(KV[offset, 0], ub_kv_sparse[idx_id, 0], size=[1, full_dim])

                                    # Store gathered KV values to workspace
                                    offset = kernel_id * top_k
                                    offset = offset + block_i_offset
                                    T.copy(ub_kv_sparse, workspace_kv[offset, 0], size=[block_I_half, full_dim])

                                with T.rs("PIPE_MTE3"):
                                    T.sync_block_set(0)

                                # Calculate synchronization count for vector operations
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
                                        T.sync_block_wait(0)

                                T.copy(ub_var_scores_max, ub_var_scores_max_prev)

                                offset = kernel_id * block_H
                                offset_sub = subid * block_H_half
                                offset = offset + offset_sub
                                with T.rs("PIPE_MTE2"):
                                    T.sync_block_wait(0)

                                # Softmax computation
                                if tail_size_i > 0:
                                    # Load attention scores
                                    T.copy(workspace_1[offset, 0], ub_cross_kernel_16, size=[block_H_half, block_I])
                                    # Cast to accumulation dtype
                                    T.npuir_cast(ub_cross_kernel_16, ub_cross_kernel_32, round_mode="rint", size=[block_H_half, block_I])
                                    # Apply softmax scale
                                    T.npuir_mul(ub_cross_kernel_32, acc_s_scale, ub_cross_kernel_32)
                                    # Compute max for numerical stability
                                    T.npuir_reduce(ub_cross_kernel_32, ub_var_scores_max, dims=[1], reduce_mode="max")
                                    # Update max and compute scale factor
                                    if block_i_id != 0:
                                        T.npuir_max(ub_var_scores_max_prev, ub_var_scores_max, ub_var_scores_max)
                                        T.npuir_sub(ub_var_scores_max_prev, ub_var_scores_max, ub_var_scores_scale)
                                        T.npuir_exp(ub_var_scores_scale, ub_var_scores_scale)
                                    # Subtract max and compute exp
                                    T.npuir_sub(ub_cross_kernel_32, ub_var_scores_max, ub_cross_kernel_32)
                                    T.npuir_exp(ub_cross_kernel_32, ub_cross_kernel_32)
                                    # Apply valid mask
                                    if tail_size_i < block_I:
                                        T.npuir_mul(ub_cross_kernel_32, ub_var_valid_mask, ub_cross_kernel_32)
                                    # Cast back to original dtype
                                    T.npuir_cast(ub_cross_kernel_32, ub_cross_kernel_16, round_mode="rint", size=[block_H_half, block_I])
                                else:
                                    T.npuir_brc(value_zero, ub_cross_kernel_16)

                                # Store softmax results and synchronize
                                with T.rs("PIPE_MTE3"):
                                    T.copy(ub_cross_kernel_16, workspace_1[offset, 0], size=[block_H_half, block_I])
                                    T.sync_block_set(0)

                                # Accumulate softmax statistics
                                if tail_size_i > 0:
                                    T.npuir_reduce(ub_cross_kernel_32, ub_var_scores_sum, dims=[1], reduce_mode="sum")

                                    T.npuir_mul(ub_var_logsum, ub_var_scores_scale, ub_var_logsum)
                                    T.npuir_add(ub_var_logsum, ub_var_scores_sum, ub_var_logsum)
                                    T.npuir_mul(ub_acc_o, ub_var_scores_scale, ub_acc_o)

                                # Accumulate output values
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
                            T.npuir_brc(value_eps, ub_var_scores_sum)
                            T.npuir_max(ub_var_logsum, ub_var_scores_sum, ub_var_logsum)
                            T.npuir_div(ub_acc_o, ub_var_logsum, ub_acc_o)
                            
                            # Write final output
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
                                T.copy(ub_cross_kernel_16, Output[offset, block_k_offset],
                                       size=[block_H_half, tail_size_k])

    return main

def generate_tensor(shape, dtype, clear=False):
    """Generate tensor with specified shape and data type"""
    if clear:
        return torch.zeros(shape, dtype=eval("torch." + dtype))
    if dtype in ("float32", "float16", "bfloat16"):
        return torch.randn(size=shape, dtype=eval("torch." + dtype))
    if dtype in ("int32", "int64", "int16"):
        return torch.randint(low=0, high=10000, size=shape, dtype=eval("torch." + dtype))
    if dtype == "int8":
        return torch.randint(low=0, high=127, size=shape, dtype=eval("torch." + dtype))
    if dtype == "bool":
        return torch.randint(low=0, high=2, size=shape).bool()
    raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))


def gather_from_kv(KV, indices):
    """Gather key-value pairs using indices"""
    b, s1, g, k = indices.shape
    batch_idx = torch.arange(b, device=KV.device).view(b, 1, 1).expand(-1, s1, k)
    indices_flat = indices.squeeze(2).long()
    out = KV[batch_idx, indices_flat, 0, :].squeeze(dim=3)

    return out


def ref_sparse_attention_fwd_interface(q, kv, indices, args, is_casual=True):
    """Reference implementation of sparse attention for verification"""
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape

    dim = args.dim
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    g_index = g
    h_index = h // g
    # Create causal mask
    compressed_casual_mask = torch.arange(0, sq, dtype=torch.int32).npu().view(
        -1, 1
    ) >= torch.arange(1 - 1, sk * 1, 1, dtype=torch.int32).npu().view(1, -1)

    # Create sparse attention mask using indices
    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(
        3, indices.long(), 1
    )
    mask = mask[..., :-1]
    if is_casual:
        mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
        mask[:, :, : 1 - 1, 0] = True
    mask = mask.view(b, g_index, 1, sq, sk)

    # Compute attention scores
    q = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q ** -0.5 if args.sm_scale is None else args.sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk)
    p = p.view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim_v)
    return o.to(torch.float16)


def run_test(args):
    """Run test with provided arguments"""
    # Compile sparse attention kernel
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

    compiled_kernel = tilelang.compile(func, target='npuir')

    # Set random seed for reproducibility
    torch.manual_seed(88888888)

    dtype = "float16"

    # Calculate block configuration parameters
    head_kv = args.heads // args.kv_group
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)

    if head_kv > 64:
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    block_H = padded_H if REPLICATE_H == 1 else 64

    # Define tensor shapes
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

    # Create sparse indices (top-k from previous positions)
    indices = torch.full(shape_idx, args.seq_len_kv, dtype=torch.int32).npu()
    for b in range(args.batch_size):
        for t in range(args.seq_len):
            for h in range(args.kv_group):
                i_i = torch.randperm(max(1, t))[:args.top_k]
                indices[b, t, h, : len(i_i)] = i_i

    # Initialize output and workspace tensors
    o = generate_tensor(shape_out, dtype, clear=True).npu()
    ref_o = generate_tensor(shape_out, dtype, clear=True).npu()
    w0_kv = generate_tensor(shape_kv_sparse_work, dtype, clear=True).npu()
    w1 = generate_tensor(shape_s_work, dtype, clear=True).npu()
    w2 = generate_tensor(shape_o_work, dtype, clear=True).npu()

    # Compute reference output using CPU implementation
    ref_o = ref_sparse_attention_fwd_interface(q.to(dtype=torch.float32), kv.to(dtype=torch.float32), indices, args)
    # Run compiled kernel on NPU
    compiled_kernel(q, kv, indices, o, w0_kv, w1, w2)

    # Compare results and validate
    torch.set_printoptions(sci_mode=False)
    print("Actual Result:")
    print(o)
    print(ref_o)
    torch.testing.assert_close(o, ref_o, rtol=5e-3, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main_args = parser.parse_args()
    run_test(main_args)
