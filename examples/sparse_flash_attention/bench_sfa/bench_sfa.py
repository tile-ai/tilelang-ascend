# https://gitcode.com/cann/cann-recipes-infer/blob/master/ops/ascendc/torch_ops_extension/custom_ops/converter/npu_sparse_flash_attention.py

import torch
import torch_npu
import custom_ops

DEVICE = "npu"

def test_op(T, B, KV_S, Q_N, KV_N, D, D_rope,
            sparse_size, scale_value,
            sparse_block_size, sparse_mode, block_size, act_kv_s, tl_ops: list):
    assert sparse_size <= KV_S
    assert KV_N == 1
    assert sparse_mode == 0 or 3
    assert sparse_block_size == 1
    assert (B * KV_S) % block_size == 0
    assert D == 512
    assert D_rope == 0 or 64

    qkv_dtype = torch.bfloat16
    #sparse_size = KV_S
    query = torch.empty((T, Q_N, D), dtype=qkv_dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    key   = torch.empty((B * KV_S // block_size, block_size, KV_N, D), dtype=qkv_dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    value = key.clone()

    rand_vals = torch.rand(T, KV_N, act_kv_s, device=DEVICE)
    _, indices = torch.topk(rand_vals, sparse_size, dim=-1)
    sparse_indices = indices.to(torch.int32)
    print(sparse_indices)
    actual_seq_lengths_query = torch.tensor([1]).reshape(B).to(torch.int32).to(DEVICE)
    actual_seq_lengths_kv = torch.tensor([act_kv_s] * B, dtype=torch.int32, device=DEVICE)
    print(actual_seq_lengths_kv)
    block_table = torch.tensor([range(B * KV_S // block_size)], dtype=torch.int32, device=DEVICE).reshape(B, -1)

    if D_rope == 0:
        query_rope = None
        key_rope = None
    else:
        query_rope = torch.empty((T, Q_N, D_rope), dtype=qkv_dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
        key_rope = torch.empty((B * KV_S // block_size, block_size, KV_N, D_rope), dtype=qkv_dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()

    print("q.shape=",query.shape)
    print("k.shape=",key.shape)
    print("v.shape=",value.shape)
    print("sparse_indices=",sparse_indices.shape)
    print("act_seq_query=",actual_seq_lengths_query)
    print("act_seq_kv=", actual_seq_lengths_kv)

    first_out = None
    for tl_op in tl_ops:
        out = tl_op(
            query = query,
            key = key,
            value = value,
            sparse_indices = sparse_indices,
            scale_value = scale_value,
            sparse_block_size = sparse_block_size,
            actual_seq_lengths_query = actual_seq_lengths_query,
            actual_seq_lengths_kv = actual_seq_lengths_kv,
            query_rope = query_rope,
            key_rope = key_rope,
            layout_query = 'TND',
            layout_kv= 'PA_BSND',
            sparse_mode = sparse_mode,
            block_table= block_table,
        )
        if first_out is None:
            first_out = out
            print(f"(first op) {tl_op.__name__} {out=}")
        else:
            out = out.to(first_out.dtype)
        torch.testing.assert_close(out, first_out, rtol=1e-2, atol=1e-2, equal_nan=True)
    print(f"(last op) {tl_op.__name__} {out=}")
    print("[PASSED]")

if __name__ == "__main__":
    from sparse_flash_attn_pa import init_test
    from sparse_flash_attn_pa import sparse_attn_tilelang as sparse_flash_attn_pa
    init_test()

    tl_ops = [torch_npu.npu_sparse_flash_attention, sparse_flash_attn_pa]
    test_op(T = 1, B = 1, KV_S = 2560, Q_N = 128, KV_N = 1, D = 512, D_rope = 64,
        sparse_size = 2048, scale_value = 0.5, sparse_block_size = 1, sparse_mode = 0,
        block_size = 128, act_kv_s = 2560, tl_ops = tl_ops)
    test_op(T = 1, B = 1, KV_S = 6400, Q_N = 128, KV_N = 1, D = 512, D_rope = 64,
        sparse_size = 2048, scale_value = 0.5, sparse_block_size = 1, sparse_mode = 0,
        block_size = 128, act_kv_s = 2560, tl_ops = tl_ops)
    test_op(T = 1, B = 1, KV_S = 48000, Q_N = 128, KV_N = 1, D = 512, D_rope = 64,
        sparse_size = 2048, scale_value = 0.5, sparse_block_size = 1, sparse_mode = 0,
        block_size = 128, act_kv_s = 2560, tl_ops = tl_ops)
