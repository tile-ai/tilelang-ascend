# Copyright (c) Huawei Technologies Co., Ltd. 2025.

# Import necessary libraries
import os
import glob
import torch

# Import tilelang modules for NPU kernel development
import tilelang
import tilelang.language as T

from typing import Optional

# Clear any cached kernels to ensure fresh compilation
tilelang.cache.clear_cache()


@tilelang.jit(target="npuir")
def sparse_attn_kernel(
        batch,
        heads,
        dim,
        tail_dim,
        top_k,
        num_kernels,
        sm_scale=None,
        block=64,
):
    # Validate input parameters
    assert block % 2 == 0, "Block I size must be even"
    assert dim == tilelang.math.next_power_of_2(
        dim
    ), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == 0 or tail_dim == tilelang.math.next_power_of_2(
        tail_dim
    ), f"haven't check padding correctness yet, dim={tail_dim}"

    seq_len = T.symbolic("seqLen")
    seq_len_kv = T.symbolic("seqLenKV")

    # Set softmax scale if not provided
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5
    else:
        sm_scale = sm_scale

    # Calculate total number of logical kernels
    num_logic_kernels = batch * seq_len

    # Define data types
    indices_dtype = "int32"
    dtype = "float16"
    accum_dtype = "float"

    # Set block size for head dimension
    heads_half = heads // 2

    # Calculate half block sizes for vector operations
    block_half = block // 2

    # Calculate full dimension (dim + tail_dim)
    full_dim = dim + tail_dim

    @T.macro
    def CubeQKMatmul(
            Q: T.Tensor([batch, seq_len, heads, full_dim], dtype),
            workspace_kv: T.Tensor([num_kernels, top_k, full_dim], dtype),
            workspace_s: T.Tensor([num_kernels, 1, heads, block], accum_dtype),
            l1_q, l1_kv_sparse, l0_c,
            num_top_k_blocks, kernel_id, task_id,
    ):
        local_id = task_id // num_top_k_blocks
        block_i_id = task_id % num_top_k_blocks

        # Calculate batch and sequence indices
        logic_kernel_id = local_id * num_kernels + kernel_id
        batch_id = logic_kernel_id // seq_len
        seq_id = logic_kernel_id % seq_len

        if block_i_id == 0:
            # Load query data to L1
            T.load_nd2nz(Q[batch_id, seq_id, 0, 0], l1_q, size=[heads, full_dim])

        block_i_offset = block_i_id * block

        # Wait for Vector computation section synchronization
        with T.rs("PIPE_MTE2"):
            T.sync_block_wait(0)

        # Load sparse KV data to L1
        T.load_nd2nz(workspace_kv[kernel_id, block_i_offset, 0],
                           l1_kv_sparse, size=[block, full_dim])

        # Perform matrix multiplication (Q @ K^T)
        T.gemm(l1_q, l1_kv_sparse, l0_c, initC=True, b_transpose=True,
                    size=[heads, full_dim, block])

        # Store intermediate results and synchronize
        with T.rs("PIPE_FIX"):
            T.store_fixpipe(l0_c, workspace_s[kernel_id, 0, 0, 0], size=[heads, block],
                                  enable_nz2nd=True)
            T.sync_block_set(0)

    @T.macro
    def CubePVMatmul(
            workspace_p: T.Tensor([num_kernels, 1, heads, block], dtype),
            workspace_o: T.Tensor([num_kernels, 1, heads, dim], accum_dtype),
            l1_kv_sparse, l0_c,
            kernel_id, task_id,
    ):
        l1_p = T.alloc_L1([heads, block], dtype)

        # Load intermediate results for softmax
        with T.rs("PIPE_MTE2"):
            T.sync_block_wait(0)
            T.load_nd2nz(workspace_p[kernel_id, 0, 0, 0], l1_p, size=[heads, block])

        # Perform matrix multiplication (P @ V)
        T.gemm(l1_p, l1_kv_sparse, l0_c, initC=True,
                    size=[heads, block, dim])

        T.store_fixpipe(l0_c, workspace_o[kernel_id, 0, 0, 0],
                              size=[heads, dim], enable_nz2nd=True)

        # Synchronize after output computation
        with T.rs("PIPE_FIX"):
            T.sync_block_set(0)

    @T.macro
    def VectorGatherSparseKV(
            KV: T.Tensor([batch, seq_len_kv, full_dim], dtype),
            Indices: T.Tensor([batch, seq_len, top_k], indices_dtype),
            workspace_kv: T.Tensor([num_kernels, top_k, full_dim], dtype),
            ub_acc_o, ub_var_logsum, ub_indices,
            num_top_k_blocks, kernel_id, subid, task_id,
    ):
        value_zero = 0

        local_id = task_id // num_top_k_blocks
        block_i_id = task_id % num_top_k_blocks

        logic_kernel_id = local_id * num_kernels + kernel_id
        batch_id = logic_kernel_id // seq_len
        seq_id = logic_kernel_id % seq_len

        ub_kv_sparse = T.alloc_ub([block_half, full_dim], dtype)

        if block_i_id == 0:
            # Initialize softmax variables
            T.vbrc(value_zero, ub_var_logsum)
            T.vbrc(value_zero, ub_acc_o)

        tail_size_i = T.min(top_k - block_i_id * block, block)
        tail_size_i_half = (tail_size_i + 1) // 2

        block_i_offset = block_i_id * block
        block_i_sub_offset = subid * tail_size_i_half
        tail_size_i_half = tail_size_i_half - (tail_size_i % 2) * subid

        # Load indices and gather sparse KV data (only for first head block)
        T.copy(Indices[batch_id, seq_id, block_i_offset], ub_indices, size=[1, block])
        if tail_size_i_half > 0:
            T.vbrc(value_zero, ub_kv_sparse)
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
            T.sync_block_set(0)

    @T.macro
    def VectorScoresExp(
            workspace_s: T.Tensor([num_kernels, 1, heads, block], accum_dtype),
            workspace_p: T.Tensor([num_kernels, 1, heads, block], dtype),
            ub_var_scores_max, ub_var_scores_scale, ub_indices, ub_arrange_mask, ub_var_logsum,
            num_top_k_blocks, kernel_id, subid, task_id,
    ):
        acc_s_scale = sm_scale
        value_zero = 0
        value_ignore_index = -1

        block_i_id = task_id % num_top_k_blocks
        tail_size_i = T.min(top_k - block_i_id * block, block)

        ub_var_scores_max_prev = T.alloc_ub([heads_half, 1], accum_dtype)
        ub_var_scores_sum = T.alloc_ub([heads_half, 1], accum_dtype)
        ub_cross_kernel_16 = T.alloc_ub([heads_half, block], dtype)
        ub_cross_kernel_32 = T.alloc_ub([heads_half, block], accum_dtype)
        ub_var_valid_indices = T.alloc_ub([1, block], "bool")
        ub_var_valid_mask = T.alloc_ub([1, block], "bool")
        ub_var_valid_indices_32 = T.alloc_ub([1, block], accum_dtype)

        # Save previous max scores for numerical stability
        T.copy(ub_var_scores_max, ub_var_scores_max_prev)

        # Load attention scores from workspace
        with T.rs("PIPE_MTE2"):
            T.sync_block_wait(0)
            T.copy(workspace_s[kernel_id, 0, subid * heads_half, 0],
                   ub_cross_kernel_32, size=[heads_half, block])

        # Apply softmax scaling and compute max
        T.vmul(ub_cross_kernel_32, acc_s_scale, ub_cross_kernel_32)
        T.reduce(ub_cross_kernel_32, ub_var_scores_max, dims=[1], reduce_mode="max")

        # Update max scores and compute scaling factors
        if block_i_id != 0:
            T.vmax(ub_var_scores_max_prev, ub_var_scores_max, ub_var_scores_max)
            T.vsub(ub_var_scores_max_prev, ub_var_scores_max, ub_var_scores_scale)
            T.vexp(ub_var_scores_scale, ub_var_scores_scale)
        else:
            T.vbrc(value_zero, ub_var_scores_scale)

        # Apply softmax stabilization and compute exponentials
        T.vsub(ub_cross_kernel_32, ub_var_scores_max, ub_cross_kernel_32)
        T.vexp(ub_cross_kernel_32, ub_cross_kernel_32)

        # Create valid mask for incomplete blocks
        T.vcmp(ub_indices, value_ignore_index, ub_var_valid_indices, "ne")
        if tail_size_i < block:
            tail_size_i = tail_size_i
            T.vcmp(ub_arrange_mask, tail_size_i, ub_var_valid_mask, "lt")
            T.vand(ub_var_valid_indices, ub_var_valid_mask, ub_var_valid_indices)

        # Apply valid mask
        T.vcast(ub_var_valid_indices, ub_var_valid_indices_32)
        T.vmul(ub_cross_kernel_32, ub_var_valid_indices_32, ub_cross_kernel_32)
        T.vcast(ub_cross_kernel_32, ub_cross_kernel_16, round_mode="rint",
                     size=[heads_half, block])

        # Store softmax results and synchronize
        with T.rs("PIPE_MTE3"):
            T.copy(ub_cross_kernel_16, workspace_p[kernel_id, 0, subid * heads_half, 0],
                   size=[heads_half, block])
            T.sync_block_set(0)

        # Compute sum of exponential for softmax denominator
        T.reduce(ub_cross_kernel_32, ub_var_scores_sum, dims=[1], reduce_mode="sum")

        # Update logsum and accumulate output
        T.vmul(ub_var_logsum, ub_var_scores_scale, ub_var_logsum)
        T.vadd(ub_var_logsum, ub_var_scores_sum, ub_var_logsum)

    @T.macro
    def VectorAccAndNormOutput(
            Output: T.Tensor([batch, seq_len, heads, dim], dtype),
            workspace_o: T.Tensor([num_kernels, 1, heads, dim], accum_dtype),
            ub_acc_o, ub_var_scores_scale, ub_var_logsum,
            num_top_k_blocks, kernel_id, subid, task_id,
    ):
        eps = 1e-8

        local_id = task_id // num_top_k_blocks
        block_i_id = task_id % num_top_k_blocks

        logic_kernel_id = local_id * num_kernels + kernel_id
        batch_id = logic_kernel_id // seq_len
        seq_id = logic_kernel_id % seq_len

        ub_cross_kernel_16 = T.alloc_ub([heads_half, block], dtype)
        ub_acc_o_new = T.alloc_ub([heads_half, dim], accum_dtype)

        T.vmul(ub_acc_o, ub_var_scores_scale, ub_acc_o)

        # Load and accumulate output values
        with T.rs("PIPE_MTE2"):
            T.sync_block_wait(0)
            T.copy(workspace_o[kernel_id, 0, subid * heads_half, 0], ub_acc_o_new,
                   size=[heads_half, dim])

        T.vadd(ub_acc_o, ub_acc_o_new, ub_acc_o)

        if block_i_id == num_top_k_blocks - 1:
            # Normalize output by softmax denominator
            T.vadd(ub_var_logsum, eps, ub_var_logsum)
            T.vdiv(ub_acc_o, ub_var_logsum, ub_acc_o)

            # Cast & store final output results
            for block_k_id in T.serial(T.ceildiv(dim, block)):
                block_k_offset = block_k_id * block
                tail_size_k = dim - block_k_offset
                tail_size_k = T.min(tail_size_k, block)

                T.vcast(ub_acc_o[0, block_k_offset], ub_cross_kernel_16, round_mode="rint",
                             size=[heads_half, tail_size_k])

                T.copy(ub_cross_kernel_16, Output[batch_id, seq_id, subid * heads_half, block_k_offset],
                       size=[heads_half, tail_size_k])

    # Define the main sparse attention kernel using TileLang
    @T.prim_func
    def SparseAttnExp(
            Q: T.Tensor([batch, seq_len, heads, full_dim], dtype),  # type: ignore
            KV: T.Tensor([batch, seq_len_kv, full_dim], dtype),  # type: ignore
            Output: T.Tensor([batch, seq_len, heads, dim], dtype),  # type: ignore
            Indices: T.Tensor([batch, seq_len, top_k], indices_dtype),  # type: ignore
            workspace_kv: T.Tensor([num_kernels, top_k, full_dim], dtype),
            workspace_s: T.Tensor([num_kernels, 1, heads, block], accum_dtype),
            workspace_p: T.Tensor([num_kernels, 1, heads, block], dtype),
            workspace_o: T.Tensor([num_kernels, 1, heads, dim], accum_dtype),
    ):
        # Launch NPU kernel with specified number of parallel kernels
        with T.Kernel(num_kernels, is_npu=True) as (kernel_id, subid):
            # Cube computation section (matrix operations)
            with T.Scope("Cube"):
                # Allocate L1 buffers for cube operations
                l1_q = T.alloc_L1([heads, full_dim], dtype)
                l1_kv_sparse = T.alloc_L1([block, full_dim], dtype)

                # Allocate L0 buffer for accumulation
                l0_c = T.alloc_L0C([heads, dim], accum_dtype)

                num_local_logic_kernels = T.ceildiv(num_logic_kernels - kernel_id, num_kernels)
                num_top_k_blocks = T.ceildiv(top_k, block)

                # Add some operations to stop TVM from moving alloc into the inner loop
                T.reshape(l1_q, l1_q)

                for task_id in T.serial(num_local_logic_kernels * num_top_k_blocks):
                    CubeQKMatmul(Q, workspace_kv, workspace_s,
                                 l1_q, l1_kv_sparse, l0_c,
                                 num_top_k_blocks, kernel_id, task_id)

                    CubePVMatmul(workspace_p, workspace_o,
                                 l1_kv_sparse, l0_c,
                                 kernel_id, task_id)

            # Vector computation section (softmax and normalization)
            with T.Scope("Vector"):
                # Allocate unified buffers for vector operations
                ub_indices = T.alloc_ub([1, block], indices_dtype)
                ub_acc_o = T.alloc_ub([heads_half, dim], accum_dtype)

                # Allocate buffers for softmax variables
                ub_var_logsum = T.alloc_ub([heads_half, 1], accum_dtype)
                ub_var_scores_max = T.alloc_ub([heads_half, 1], accum_dtype)
                ub_var_scores_scale = T.alloc_ub([heads_half, 1], accum_dtype)

                ub_arrange_mask = T.alloc_ub([1, block], "int16")
                T.arange(ub_arrange_mask, [0, 1], 0)

                num_local_logic_kernels = T.ceildiv(num_logic_kernels - kernel_id, num_kernels)
                num_top_k_blocks = T.ceildiv(top_k, block)

                # Add some operations to stop TVM from moving alloc into the inner loop
                T.reshape(ub_var_logsum, ub_var_logsum)
                T.reshape(ub_acc_o, ub_acc_o)
                T.reshape(ub_var_scores_max, ub_var_scores_max)

                for task_id in T.serial(num_local_logic_kernels * num_top_k_blocks):
                    VectorGatherSparseKV(
                        KV, Indices, workspace_kv,
                        ub_acc_o, ub_var_logsum, ub_indices,
                        num_top_k_blocks, kernel_id, subid, task_id
                    )

                    VectorScoresExp(
                        workspace_s, workspace_p,
                        ub_var_scores_max, ub_var_scores_scale, ub_indices, ub_arrange_mask, ub_var_logsum,
                        num_top_k_blocks, kernel_id, subid, task_id,
                    )

                    VectorAccAndNormOutput(
                        Output, workspace_o,
                        ub_acc_o, ub_var_scores_scale, ub_var_logsum,
                        num_top_k_blocks, kernel_id, subid, task_id,
                    )

    return SparseAttnExp


def sparse_attn(
        q: torch.Tensor, kv: torch.Tensor, topk_idxs: torch.Tensor,
        softmax_scale: Optional[float] = None
):
    num_kernels = 24
    block = 64
    batch, seq_len, heads, dim = q.size()
    _, _, top_k = topk_idxs.size()
    new_static_param_dict = {
        "batch": batch,
        "heads": heads,
        "dim": dim,
        "top_k": top_k,
    }
    if not hasattr(sparse_attn, 'kernel_cache') or new_static_param_dict != sparse_attn.static_param_dict:
        sparse_attn.kernel = sparse_attn_kernel(batch, heads, dim, tail_dim=0, top_k=top_k,
                                                num_kernels=num_kernels, sm_scale=softmax_scale, block=block, )
        sparse_attn.static_param_dict = new_static_param_dict
    output = torch.empty((batch, seq_len, heads, dim), dtype=q.dtype, device=q.device)
    w_kv = torch.empty((num_kernels, top_k, dim), dtype=q.dtype, device=q.device)
    w_s = torch.empty((num_kernels, 1, heads, block), dtype=torch.float32, device=q.device)
    w_p = torch.empty((num_kernels, 1, heads, block), dtype=q.dtype, device=q.device)
    w_o = torch.empty((num_kernels, 1, heads, dim), dtype=torch.float32, device=q.device)
    sparse_attn.kernel(q, kv, output, topk_idxs, w_kv, w_s, w_p, w_o)
    return output


def gather_from_kv(KV, indices):
    """Gather key-value pairs using indices for reference implementation"""
    b, s1, k = indices.shape
    batch_idx = torch.arange(b, device=KV.device).view(b, 1, 1).expand(-1, s1, k)
    indices_flat = indices.long()
    out = KV[batch_idx, indices_flat, :].squeeze(dim=2)
    return out


def sparse_attn_torch(q: torch.Tensor, kv: torch.Tensor, topk_idxs: torch.Tensor,
                      softmax_scale: Optional[float] = None):
    """Reference Sparse Attention kernel implemented in PyTorch"""
    kv_sparse = gather_from_kv(kv, topk_idxs)
    mask_acc_s = torch.where((topk_idxs == -1).unsqueeze(-2), -torch.inf, 0.)
    mask_acc_s = mask_acc_s.to(device=q.device, dtype=torch.float32)
    ref_output = torch.nn.functional.softmax(
        ((q @ kv_sparse.transpose(-2, -1)).to(torch.float32) + mask_acc_s) * softmax_scale,
        dim=-1).to(torch.float16) @ kv_sparse

    return ref_output


def rand_sparse_attn_input(batch_size, num_heads, seq_len, seq_len_kv, top_k, dim, seed=88888888, causal=True):
    """Generate legalized random inputs for Sparse Attention"""
    torch.manual_seed(seed)

    # Generate inputs
    q = torch.randn((batch_size, seq_len, num_heads, dim), dtype=torch.float16).npu()
    kv = torch.randn((batch_size, seq_len_kv, dim), dtype=torch.float16).npu()
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
    )

    generate_and_save_data(
        case_id=1,
        batch_size=1,
        num_heads=32,
        seq_len=2777,
        seq_len_kv=4096,
        top_k=128,
        dim=512,
    )

    generate_and_save_data(
        case_id=2,
        batch_size=1,
        num_heads=32,
        seq_len=4096,
        seq_len_kv=3973,
        top_k=128,
        dim=512,
    )


def run_test(verify_acc=True):
    """
    Traverse all case_*.pt files in the current directory for testing.
    The verify_acc parameter determines whether to perform accuracy check.
    """
    pattern = os.path.join(os.getcwd(), "case_*.pt")
    file_paths = sorted(glob.glob(pattern))
    if not file_paths:
        raise FileNotFoundError("No case_*.pt files found in current directory.")

    errors = []
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        try:
            data = torch.load(file_path, map_location=torch.device('npu'))
            output = sparse_attn(**data['inputs'])

            if verify_acc:
                torch.testing.assert_close(data['outputs'], output, rtol=1e-2, atol=1e-2)
                print(f'{filename}: \033[92mPassed.\033[0m')
            else:
                assert output is not None
                print(f'{filename}: \033[92mFinished.\033[0m')
        except Exception as e:
            errors.append(f"{filename}: {str(e)}")
            print(f'{filename}: \033[91mFailed: {e}\033[0m')

    if errors:
        error_msg = "\n".join(errors)
        raise AssertionError(f"Some cases failed:\n{error_msg}")
    else:
        print('\033[92mAll checks passed.\033[0m')


if __name__ == "__main__":
    # Specifies which NPU device to use
    torch.npu.set_device(0)

    # Generate data and run tests
    generate_data()
    run_test(verify_acc=True)
