# Copyright (c) Huawei Technologies Co., Ltd. 2026.
import torch
import tilelang
import tilelang.language as T

from typing import Optional


@tilelang.jit(target="npuir")
def sparse_attn_kernel(
        batch,
        seq_len,
        seq_len_kv,
        heads,
        dim,
        tail_dim,
        top_k,
        num_kernels,
        sm_scale=None,
        block_I=64,
        block_K=64,
):
    # Validate input parameters
    assert block_I % 2 == 0, "Block I size must be even"
    assert dim == tilelang.math.next_power_of_2(
        dim
    ), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim==0 or tail_dim == tilelang.math.next_power_of_2(
        tail_dim
    ), f"haven't check padding correctness yet, dim={tail_dim}"

    # Set softmax scale if not provided
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5
    else:
        sm_scale = sm_scale


    # Calculate total number of logical kernels
    num_logic_kernels = batch * seq_len

    # Define data types
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"

    # Set block size for head dimension
    heads_half = heads // 2

    # Calculate half block sizes for vector operations
    block_I_half = block_I // 2

    # Calculate shared block size for I and K dimensions
    share_block_IK = max(block_I, block_K)

    # Calculate full dimension (dim + tail_dim)
    full_dim = dim + tail_dim
    num_block_i = T.ceildiv(top_k, block_I)

    # Define tensor shapes
    shape_q = [batch, seq_len, heads, full_dim]
    shape_kv = [batch, seq_len_kv, full_dim]
    shape_out = [batch, seq_len, heads, dim]
    shape_idx = [batch, seq_len, top_k]
    shape_kv_sparse_work = [num_kernels, top_k, full_dim]
    shape_s_work = [num_kernels, num_block_i, heads, block_I]
    shape_o_work = [num_kernels, num_block_i, heads, dim]
    shape_u_work = [num_kernels, num_block_i, heads, 1]

    # Define the main sparse attention kernel using TileLang
    @T.prim_func
    def SparseAttnExp(
            Q: T.Tensor(shape_q, dtype),
            KV: T.Tensor(shape_kv, dtype),
            Output: T.Tensor(shape_out, dtype),
            attn_sink: T.Tensor((heads), accum_dtype),
            Indices: T.Tensor(shape_idx, indices_dtype),
            workspace_kv: T.Tensor(shape_kv_sparse_work, dtype),
            workspace_s: T.Tensor(shape_s_work, accum_dtype),
            workspace_p: T.Tensor(shape_s_work, dtype),
            workspace_o: T.Tensor(shape_o_work, accum_dtype),
            workspace_u: T.Tensor(shape_u_work, accum_dtype),
    ):
        # Launch NPU kernel with specified number of parallel kernels
        with T.Kernel(num_kernels, is_npu=True) as (kernel_id, subid):
            acc_s_scale = sm_scale

            # Cube computation section (matrix operations)
            with T.Scope("Cube"):
                # Allocate L1 buffers for cube operations
                l1_q = T.alloc_L1([heads, full_dim], dtype)
                l1_p = T.alloc_L1([heads, block_I], dtype)
                l1_kv_sparse = T.alloc_L1([block_I, full_dim], dtype)

                # Allocate L0 buffer for accumulation
                l0_c = T.alloc_L0C([heads, dim], accum_dtype)

                # Process tasks in serial across logical kernels
                for task_id in T.serial(T.ceildiv(num_logic_kernels, num_kernels)):
                    logic_kernel_id = task_id * num_kernels
                    logic_kernel_id = logic_kernel_id + kernel_id
                    if logic_kernel_id < num_logic_kernels:
                        # Calculate batch and sequence indices
                        batch_id = logic_kernel_id // seq_len
                        seq_id = logic_kernel_id % seq_len

                        # Load query data to L1
                        T.npuir_load_nd2nz(Q[batch_id, seq_id, 0, 0], l1_q, size=[heads, full_dim])

                        # Process blocks along I dimension (top-k)
                        for block_i_id in T.serial(T.ceildiv(top_k, block_I)):
                            block_i_offset = block_i_id * block_I

                            # Wait for Vector computation section synchronization
                            with T.rs("PIPE_MTE2"):
                                T.sync_block_wait(block_i_id)

                            # Load sparse KV data to L1
                            T.npuir_load_nd2nz(workspace_kv[kernel_id, block_i_offset, 0],
                                               l1_kv_sparse, size=[block_I, full_dim])

                            # Perform matrix multiplication (Q @ K^T)
                            T.npuir_dot(l1_q, l1_kv_sparse, l0_c, initC=True, b_transpose=True,
                                        size=[heads, full_dim, block_I])

                            # Store intermediate results and synchronize
                            with T.rs("PIPE_FIX"):
                                T.npuir_store_fixpipe(l0_c, workspace_s[kernel_id, block_i_id, 0, 0], size=[heads, block_I],
                                                      enable_nz2nd=True)
                                T.sync_block_set(block_i_id)

                        for block_i_id in T.serial(T.ceildiv(top_k, block_I)):
                            block_i_offset = block_i_id * block_I
                            # Load intermediate results for softmax
                            with T.rs("PIPE_MTE2"):
                                T.sync_block_wait(block_i_id)
                                T.npuir_load_nd2nz(workspace_p[kernel_id, block_i_id, 0, 0], l1_p, size=[heads, block_I])

                            # Perform matrix multiplication (P @ V)
                            T.npuir_load_nd2nz(workspace_kv[kernel_id, block_i_offset, 0],
                                               l1_kv_sparse, size=[block_I, full_dim])

                            T.npuir_dot(l1_p, l1_kv_sparse, l0_c, initC=True,
                                        size=[heads, block_I, dim])

                            T.npuir_store_fixpipe(l0_c, workspace_o[kernel_id, block_i_id, 0, 0],
                                                  size=[heads, dim], enable_nz2nd=True)

                            # Synchronize after output computation
                            with T.rs("PIPE_FIX"):
                                T.sync_block_set(block_i_id)

            # Vector computation section (softmax and normalization)
            with T.Scope("Vector"):
                value_zero = 0
                value_ignore_index = -1
                value_min = -T.infinity("float32")

                # Allocate unified buffers for vector operations
                ub_kv_sparse = T.alloc_ub([block_I_half, full_dim], dtype)
                ub_indices = T.alloc_ub([1, block_I], indices_dtype)
                ub_acc_o = T.alloc_ub([heads_half, dim], accum_dtype)
                ub_acc_o_new = T.alloc_ub([heads_half, dim], accum_dtype)
                ub_cross_kernel_16 = T.alloc_ub([heads_half, share_block_IK], dtype)
                ub_cross_kernel_32 = T.alloc_ub([heads_half, share_block_IK], accum_dtype)

                # Allocate buffers for softmax variables
                ub_var_logsum = T.alloc_ub([heads_half, 1], accum_dtype)
                ub_var_scores_max = T.alloc_ub([heads_half, 1], accum_dtype)
                ub_var_scores_max_prev = T.alloc_ub([heads_half, 1], accum_dtype)
                ub_var_scores_scale = T.alloc_ub([heads_half, 1], accum_dtype)
                ub_var_scores_sum = T.alloc_ub([heads_half, 1], accum_dtype)
                ub_var_valid_indices = T.alloc_ub([1, block_I], "bool")
                ub_var_valid_mask = T.alloc_ub([1, block_I], "bool")
                ub_var_valid_indices_32 = T.alloc_ub([1, block_I], accum_dtype)

                ub_arrange_mask = T.alloc_ub([1, block_I], "int16")
                T.npuir_arange(ub_arrange_mask, [0, 1], 0)

                # Process tasks in serial across logical kernels
                for task_id in T.serial(T.ceildiv(num_logic_kernels, num_kernels)):
                    logic_kernel_id = task_id * num_kernels
                    logic_kernel_id = logic_kernel_id + kernel_id
                    if logic_kernel_id < num_logic_kernels:
                        batch_id = logic_kernel_id // seq_len
                        seq_id = logic_kernel_id % seq_len

                        # Initialize softmax variables
                        T.npuir_brc(value_zero, ub_var_logsum)
                        T.npuir_brc(value_zero, ub_acc_o)
                        T.npuir_brc(value_zero, ub_var_scores_scale)
                        T.npuir_brc(value_min, ub_var_scores_max)

                        # Process blocks along I dimension (top-k)
                        for block_i_id in T.serial(T.ceildiv(top_k, block_I)):
                            tail_size_i = T.min(top_k - block_i_id * block_I, block_I)
                            tail_size_i_half = (tail_size_i + 1) // 2

                            block_i_offset = block_i_id * block_I
                            block_i_sub_offset = subid * tail_size_i_half
                            tail_size_i_half = tail_size_i_half - (tail_size_i % 2) * subid

                            # Load indices and gather sparse KV data (only for first head block)
                            T.copy(Indices[batch_id, seq_id, block_i_offset], ub_indices, size=[1, block_I])
                            if tail_size_i_half > 0:
                                T.npuir_brc(value_zero, ub_kv_sparse)
                                for idx_id in T.serial(tail_size_i_half):
                                    current_index = ub_indices[0, block_i_sub_offset + idx_id]
                                    if current_index >= 0 and current_index < seq_len_kv:
                                        T.copy(KV[batch_id, current_index, 0], ub_kv_sparse[idx_id, 0],
                                               size=[1, full_dim])

                                # Store gathered KV data to workspace
                                T.copy(ub_kv_sparse, workspace_kv[kernel_id, block_i_offset + block_i_sub_offset, 0],
                                       size=[tail_size_i_half, full_dim])

                            # Synchronize after KV gathering
                            with T.rs("PIPE_MTE3"):
                                T.sync_block_set(block_i_id)

                        for block_i_id in T.serial(T.ceildiv(top_k, block_I)):
                            # Save previous max scores for numerical stability
                            T.copy(ub_var_scores_max, ub_var_scores_max_prev)
                            block_i_offset = block_i_id * block_I
                            T.copy(Indices[batch_id, seq_id, block_i_offset], ub_indices, size=[1, block_I])

                            # Load attention scores from workspace
                            with T.rs("PIPE_MTE2"):
                                T.sync_block_wait(block_i_id)
                                T.copy(workspace_s[kernel_id, block_i_id, subid * heads_half, 0],
                                       ub_cross_kernel_32, size=[heads_half, block_I])

                            # Apply softmax scaling and compute max
                            T.npuir_mul(ub_cross_kernel_32, acc_s_scale, ub_cross_kernel_32)
                            T.npuir_reduce(ub_cross_kernel_32, ub_var_scores_max, dims=[1], reduce_mode="max")

                            # Update max scores and compute scaling factors
                            if block_i_id != 0:
                                T.npuir_max(ub_var_scores_max_prev, ub_var_scores_max, ub_var_scores_max)
                                T.npuir_sub(ub_var_scores_max_prev, ub_var_scores_max, ub_var_scores_scale)
                                T.npuir_exp(ub_var_scores_scale, ub_var_scores_scale)

                            T.copy(ub_var_scores_scale, workspace_u[kernel_id, block_i_id, subid * heads_half, 0],
                                   size=[heads_half, 1])
                            # Apply softmax stabilization and compute exponentials
                            T.npuir_sub(ub_cross_kernel_32, ub_var_scores_max, ub_cross_kernel_32)
                            T.npuir_exp(ub_cross_kernel_32, ub_cross_kernel_32)

                            # Create valid mask for incomplete blocks
                            T.npuir_cmp(ub_indices, value_ignore_index, ub_var_valid_indices, "ne")
                            tail_size_i = T.min(top_k - block_i_id * block_I, block_I)
                            if tail_size_i < block_I:
                                tail_size_i = tail_size_i
                                T.npuir_cmp(ub_arrange_mask, tail_size_i, ub_var_valid_mask, "lt")
                                T.npuir_and(ub_var_valid_indices, ub_var_valid_mask, ub_var_valid_indices)

                            # Apply valid mask
                            T.npuir_cast(ub_var_valid_indices, ub_var_valid_indices_32)
                            T.npuir_mul(ub_cross_kernel_32, ub_var_valid_indices_32, ub_cross_kernel_32)
                            T.npuir_cast(ub_cross_kernel_32, ub_cross_kernel_16, round_mode="rint",
                                         size=[heads_half, block_I])

                            # Store softmax results and synchronize
                            with T.rs("PIPE_MTE3"):
                                T.copy(ub_cross_kernel_16, workspace_p[kernel_id, block_i_id, subid * heads_half, 0],
                                       size=[heads_half, block_I])
                                T.sync_block_set(block_i_id)

                            # Compute sum of exponential for softmax denominator
                            T.npuir_reduce(ub_cross_kernel_32, ub_var_scores_sum, dims=[1], reduce_mode="sum")

                            # Update logsum and accumulate output
                            T.npuir_mul(ub_var_logsum, ub_var_scores_scale, ub_var_logsum)
                            T.npuir_add(ub_var_logsum, ub_var_scores_sum, ub_var_logsum)

                        for block_i_id in T.serial(T.ceildiv(top_k, block_I)):
                            T.copy(workspace_u[kernel_id, block_i_id, subid * heads_half, 0],
                                   ub_var_scores_scale,
                                   size=[heads_half, 1])
                            T.npuir_mul(ub_acc_o, ub_var_scores_scale, ub_acc_o)

                            # Load and accumulate output values
                            with T.rs("PIPE_MTE2"):
                                T.sync_block_wait(block_i_id)
                                T.copy(workspace_o[kernel_id, block_i_id, subid * heads_half, 0], ub_acc_o_new,
                                       size=[heads_half, dim])

                            T.npuir_add(ub_acc_o, ub_acc_o_new, ub_acc_o)

                        # Normalize output by softmax denominator
                        T.copy(attn_sink[subid * heads_half:(subid + 1) * heads_half], ub_var_scores_max_prev[:, 0])
                        T.npuir_sub(ub_var_scores_max_prev, ub_var_scores_max, ub_var_scores_max_prev)
                        T.npuir_exp(ub_var_scores_max_prev, ub_var_scores_max_prev)
                        T.npuir_add(ub_var_logsum, ub_var_scores_max_prev, ub_var_logsum)
                        T.npuir_div(ub_acc_o, ub_var_logsum, ub_acc_o)

                        # Store final output results
                        for block_k_id in T.serial(T.ceildiv(dim, block_K)):
                            block_k_offset = block_k_id * block_K
                            tail_size_k = dim - block_k_offset
                            tail_size_k = T.min(tail_size_k, block_K)

                            T.npuir_cast(ub_acc_o[0, block_k_offset], ub_cross_kernel_16, round_mode="rint",
                                         size=[heads_half, tail_size_k])

                            T.copy(ub_cross_kernel_16, Output[batch_id, seq_id, subid * heads_half, block_k_offset],
                                   size=[heads_half, tail_size_k])

    return SparseAttnExp


def sparse_attn(
        q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor,
        softmax_scale: Optional[float] = None
):
    num_kernels = 24
    block_I = 64
    block_K = 64
    batch, seq_len, heads, dim = q.size()
    _, seq_len_kv, _ = kv.size()
    _, _, top_k = topk_idxs.size()
    num_block_i = torch.ceil(torch.tensor(top_k / block_I)).int().item()
    kernel = sparse_attn_kernel(batch, seq_len, seq_len_kv, heads, dim, tail_dim=0, top_k=top_k,
                                num_kernels=num_kernels, sm_scale=softmax_scale, block_I=block_I, block_K=block_K)
    output = torch.empty((batch, seq_len, heads, dim), dtype=q.dtype).npu()
    w_kv = torch.empty((num_kernels, top_k, dim), dtype=q.dtype).npu()
    w_s = torch.empty((num_kernels, num_block_i, heads, block_I), dtype=attn_sink.dtype).npu()
    w_p = torch.empty((num_kernels, num_block_i, heads, block_I), dtype=q.dtype).npu()
    w_o = torch.empty((num_kernels, num_block_i, heads, dim), dtype=attn_sink.dtype).npu()
    w_u = torch.empty((num_kernels, num_block_i, heads, 1), dtype=attn_sink.dtype).npu()
    kernel(q, kv, output, attn_sink, topk_idxs, w_kv, w_s, w_p, w_o, w_u)
    return output


def gather_from_kv(KV, indices):
    """Gather key-value pairs using indices for reference implementation"""
    b, s1, k = indices.shape
    batch_idx = torch.arange(b, device=KV.device).view(b, 1, 1).expand(-1, s1, k)
    indices_flat = indices.long()
    out = KV[batch_idx, indices_flat, :].squeeze(dim=2)
    return out


def softmax_with_sink(x: torch.Tensor, attn_sink: torch.Tensor, head_dim, dim=-1):
    max_vals = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_vals)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)

    sink_view_shape = [1] * x.dim()
    sink_view_shape[head_dim if head_dim > 0 else head_dim % x.dim()] = x.shape[head_dim]

    sink_term = torch.exp(attn_sink.view(sink_view_shape) - max_vals)
    adjusted_sum = sum_exp + sink_term

    return exp_x / adjusted_sum


def sparse_attn_torch(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor,
                      softmax_scale: Optional[float] = None):
    """Reference Sparse Attention kernel implemented in PyTorch"""
    kv_sparse = gather_from_kv(kv, topk_idxs)
    mask_acc_s = torch.where((topk_idxs == -1).unsqueeze(-2), -torch.inf, 0.)
    mask_acc_s = mask_acc_s.to(device=q.device, dtype=torch.float32)
    ref_output = softmax_with_sink(
        ((q @ kv_sparse.transpose(-2, -1)).to(torch.float32) + mask_acc_s) * softmax_scale, attn_sink, head_dim=-2,
        dim=-1).to(torch.bfloat16) @ kv_sparse

    return ref_output


def rand_sparse_attn_input(batch_size, num_heads, seq_len, seq_len_kv, top_k, dim, seed=88888888, causal=True):
    """Generate legalized random inputs for Sparse Attention"""
    torch.manual_seed(seed)

    # Generate inputs
    q = torch.randn((batch_size, seq_len, num_heads, dim), dtype=torch.bfloat16).npu()
    kv = torch.randn((batch_size, seq_len_kv, dim), dtype=torch.bfloat16).npu()
    attn_sink = torch.randn((num_heads,), dtype=torch.float32).npu()
    top_k_indices = torch.randint(low=0, high=seq_len_kv, size=(batch_size, seq_len, top_k), dtype=torch.int32).npu()

    if causal:
        # Apply causal mask on top_k_indices
        max_len = max(seq_len, top_k)
        causal_mask = torch.tril(torch.ones(max_len, max_len)).to(top_k_indices.device)
        causal_mask = causal_mask[:seq_len, :top_k]
        causal_mask = causal_mask.unsqueeze(dim=0).bool()
        top_k_indices = torch.where(causal_mask, top_k_indices, -1)

    scale = (1.0 / dim) ** 0.5

    return {
        'q': q,
        'kv': kv,
        'attn_sink': attn_sink,
        'topk_idxs': top_k_indices,
        'softmax_scale': scale,
    }


def generate_and_save_data(case_id, **kwargs):
    inputs = rand_sparse_attn_input(**kwargs)
    outputs = sparse_attn_torch(**inputs)
    torch.save({'inputs': inputs, 'outputs': outputs}, f"case_{case_id}.pt")


def generate_data():
    generate_and_save_data(
        case_id=0,
        batch_size=1,
        num_heads=32,
        seq_len=4096,
        seq_len_kv=4096,
        top_k=128,
        dim=512,
        # causal=False,
    )


def run_test(verify_result=True):
    data = torch.load("case_0.pt", map_location=torch.device('npu'))
    output = sparse_attn(**data['inputs'])
    if verify_result:
        print("torch:", data['outputs'])
        print("output:", output)
        torch.testing.assert_close(data['outputs'], output, rtol=1e-2, atol=1e-2)
        print('\033[92mAll check passed.\033[0m')
    else:
        print(output)


if __name__ == "__main__":
    # Generate data and run tests
    generate_data()
    run_test(verify_result=True)
    # run_test(verify_result=False)