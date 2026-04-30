# Copyright (c) Huawei Technologies Co., Ltd. 2026.
import os
import torch
import tilelang
import tilelang.language as T

from typing import Optional

FP16 = "float16"
FP32 = "float32"
INT32 = "int32"


@tilelang.jit(out_idx=[2], target="npuir")
def sparse_attn_kernel(block_top_k_vec, block_top_k_cube, block_heads, num_heads, dim, max_top_k=640, scale=None,
                       dtype=FP16, accum_dtype=FP32, indices_dtype=INT32):
    if scale is None:
        scale = (1.0 / dim) ** 0.5
    else:
        scale = scale
    batch_size = T.symbolic("batchSize")
    seq_len = T.symbolic("seqLen")
    seq_len_kv = T.symbolic("seqLenKV")
    top_k = T.symbolic("topK")
    top_k_reserved = T.symbolic("topKReserved")

    @T.prim_func
    def sparseAttn(
            Q: T.Tensor((batch_size, seq_len, num_heads, dim), dtype),
            KV: T.Tensor((batch_size, seq_len_kv, dim), dtype),
            Output: T.Tensor((batch_size, seq_len, num_heads, dim), dtype),
            AttnSink: T.Tensor((num_heads, 1), accum_dtype),
            TopKIndices: T.Tensor((batch_size, seq_len, top_k), indices_dtype),
            SparseKVBuffer: T.Tensor((batch_size, seq_len, top_k_reserved, dim), dtype),
            ValidMaskBuffer: T.Tensor((batch_size, seq_len, top_k_reserved), accum_dtype),
            WorkspaceScore: T.Tensor((batch_size, seq_len, block_heads, top_k_reserved), accum_dtype),
    ):
        with T.Kernel(batch_size * seq_len, is_npu=True) as (cid, _):
            by = cid // seq_len
            bx = cid % seq_len
            value_zero = 0

            q_shared = T.alloc_shared((block_heads, dim), dtype)
            kv_gather = T.alloc_shared((block_top_k_vec, dim), dtype)
            kv_shared = T.alloc_shared((block_top_k_cube, dim), dtype)
            o_shared = T.alloc_shared((block_heads, dim), dtype)
            acc_s_cast = T.alloc_shared((block_heads, max_top_k), dtype)
            attn_sink_shared = T.alloc_shared((block_heads, 1), accum_dtype)
            
            idxs = T.alloc_fragment((block_top_k_vec,), indices_dtype)
            acc_s_block = T.alloc_fragment((block_heads, block_top_k_cube), accum_dtype)
            acc_s = T.alloc_fragment((block_heads, max_top_k), accum_dtype)
            acc_o = T.alloc_fragment((block_heads, dim), accum_dtype)
            scores_max = T.alloc_fragment((block_heads, 1), accum_dtype)
            scores_sum = T.alloc_fragment((block_heads, 1), accum_dtype)
            valid_mask = T.alloc_fragment((1, max_top_k), accum_dtype)
            valid_mask_block = T.alloc_fragment((block_top_k_vec,), accum_dtype)

            for n in T.Pipelined(T.ceildiv(num_heads, block_heads), num_stages=2):
                T.vbrc(value_zero, acc_o)
                T.copy(Q[by, bx, n * block_heads, 0], q_shared)
                if n == 0:
                    for k in T.Pipelined(T.ceildiv(top_k_reserved, block_top_k_vec), num_stages=2):
                        real_block_top_k = T.min(top_k - k * block_top_k_vec, block_top_k_vec)
                        real_block_top_k = T.max(real_block_top_k, 0)
                        T.copy(TopKIndices[by, bx, k * block_top_k_vec], idxs)
                        T.vbrc(value_zero, valid_mask_block)
                        T.vbrc(value_zero, kv_gather)
                        for i in T.serial(real_block_top_k):
                            cur_idx = idxs[i]
                            if cur_idx != -1:
                                valid_mask_block[i] = 1.
                                T.copy(KV[by, cur_idx, 0], kv_gather[i, 0], size=[1, dim])
                        T.copy(valid_mask_block, ValidMaskBuffer[by, bx, k * block_top_k_vec])
                        T.copy(kv_gather, SparseKVBuffer[by, bx, k * block_top_k_vec, 0])

                for k in T.Pipelined(T.ceildiv(top_k, block_top_k_cube), num_stages=2):
                    T.copy(SparseKVBuffer[by, bx, k * block_top_k_cube, 0], kv_shared)
                    T.gemm(q_shared, kv_shared, acc_s_block, initC=True, b_transpose=True)
                    T.copy(acc_s_block, WorkspaceScore[by, bx, 0, k * block_top_k_cube])

                T.copy(WorkspaceScore[by, bx, 0, 0], acc_s, size=[block_heads, top_k_reserved])
                for i, j in T.Parallel(block_heads, max_top_k):
                    acc_s[i, j] *= scale
                T.reduce_max(acc_s, scores_max, dim=1, size=[block_heads, top_k])
                for i, j in T.Parallel(block_heads, max_top_k):
                    acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i, 0])
                T.vbrc(value_zero, valid_mask)
                T.copy(ValidMaskBuffer[by, bx, 0], valid_mask[0, 0], size=[1, top_k_reserved])
                for i, j in T.Parallel(block_heads, max_top_k):
                    acc_s[i, j] *= valid_mask[0, j]
                T.reduce_sum(acc_s, scores_sum, dim=1, size=[block_heads, top_k])
                T.copy(AttnSink[n * block_heads, 0], attn_sink_shared)
                for i in T.Parallel(block_heads):
                    scores_sum[i, 0] += T.exp(attn_sink_shared[i, 0] - scores_max[i, 0])
                for i, j in T.Parallel(block_heads, max_top_k):
                    acc_s[i, j] /= scores_sum[i, 0]
                T.copy(acc_s, acc_s_cast)

                for k in T.Pipelined(T.ceildiv(top_k, block_top_k_cube), num_stages=2):
                    T.copy(SparseKVBuffer[by, bx, k * block_top_k_cube, 0], kv_shared)
                    T.gemm(acc_s_cast[0, k * block_top_k_cube], kv_shared, acc_o, initC=False,
                           size=[block_heads, block_top_k_cube, dim])
                T.copy(acc_o, o_shared)
                T.copy(o_shared, Output[by, bx, n * block_heads, 0])

    return sparseAttn


def next_divisible_number(num, divisor):
    return num + divisor - (num % divisor) if num % divisor != 0 else num


def sparse_attn(
        q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor,
        softmax_scale: Optional[float] = None
):
    block_vec = 32
    block_cube = 128
    block_heads = 16
    max_top_k = 256
    batch_size, seq_len, num_heads, dim = q.size()
    top_k = topk_idxs.shape[-1]
    if not hasattr(sparse_attn, 'kernel') or sparse_attn.num_heads != num_heads or sparse_attn.dim != dim:
        os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
        bytes_workspace = block_cube * dim * 2 + block_heads * block_cube * 4 * 2 + block_heads * dim * 4
        os.environ['TILELANG_ASCEND_WORKSPACE_SIZE'] = str(bytes_workspace * 16)
        sparse_attn.kernel = sparse_attn_kernel(block_vec, block_cube, block_heads, num_heads, dim, max_top_k, softmax_scale)
        sparse_attn.num_heads = num_heads
        sparse_attn.dim = dim
    output = torch.empty((batch_size, seq_len, num_heads, dim), dtype=q.dtype, device=q.device)
    sparse_kv_buffer = torch.empty((batch_size, seq_len, next_divisible_number(top_k, block_cube), dim),
                                   dtype=q.dtype, device=q.device)
    valid_mask_buffer = torch.empty((batch_size, seq_len, next_divisible_number(top_k, block_cube)),
                                    dtype=attn_sink.dtype, device=attn_sink.device)
    workspace_score = torch.empty((batch_size, seq_len, block_heads, next_divisible_number(top_k, block_cube)),
                                  dtype=attn_sink.dtype, device=attn_sink.device)
    output = sparse_attn.kernel(q, kv.contiguous(), attn_sink, topk_idxs, sparse_kv_buffer, valid_mask_buffer, workspace_score)
    return output


def gather_from_kv(KV, indices):
    """Gather key-value pairs using indices for reference implementation"""
    b, s1, k = indices.shape
    batch_idx = torch.arange(b, device=KV.device).view(b, 1, 1).expand(-1, s1, k)
    indices_flat = indices.long()
    out = KV[batch_idx, indices_flat, :].squeeze(dim=2)

    mask = (indices != -1).float().unsqueeze(-1)
    out = out * mask

    return out


def softmax_with_sink(x: torch.Tensor, attn_sink: torch.Tensor, head_dim, dim=-1):
    max_vals = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_vals)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)

    sink_view_shape = [1] * x.dim()
    sink_view_shape[head_dim if head_dim > 0 else head_dim % x.dim()] = x.shape[head_dim]

    # attention sink
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
        dim=-1).to(torch.float16) @ kv_sparse

    return ref_output


def rand_sparse_attn_input(batch_size, num_heads, seq_len, seq_len_kv, top_k, dim, seed=88888888, causal=True):
    """Generate legalized random inputs for Sparse Attention"""
    torch.manual_seed(seed)

    # Generate inputs
    q = torch.randn((batch_size, seq_len, num_heads, dim), dtype=torch.float16).npu()
    kv = torch.randn((batch_size, seq_len_kv, dim), dtype=torch.float16).npu()
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
    )


def run_test():
    data = torch.load("case_0.pt", map_location=torch.device('npu'))
    output = sparse_attn(**data['inputs'])
    print(output)
    torch.testing.assert_close(data['outputs'], output, rtol=1e-2, atol=1e-2)
    print('\033[92mAll check passed.\033[0m')


if __name__ == "__main__":
    generate_data()
    run_test()