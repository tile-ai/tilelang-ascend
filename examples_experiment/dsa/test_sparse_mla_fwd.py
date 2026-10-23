# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Correctness and performance test for the sparse_mla_fwd Ascend kernel."""

import torch
import torch_npu  # noqa: F401  (registers the NPU device)

from sparse_mla_fwd import sparse_mla_fwd_interface


def calculate_tensor_similarity(x, y, name="tensor"):
    """
    Calculate similarity between two tensors using a normalized dot product metric.

    Unlike torch.testing.assert_close which uses absolute/relative tolerance based on
    element-wise differences, this function computes a global similarity score:
        sim = 2 * <x, y> / (||x||^2 + ||y||^2)

    This metric is scale-invariant and measures the cosine-like similarity normalized
    by the magnitude of both tensors. It returns 1 for identical tensors and values
    closer to 0 for dissimilar ones. This is particularly useful for comparing tensors
    with varying magnitudes where relative errors matter more than absolute differences.

    Args:
        x: First tensor to compare
        y: Second tensor to compare
        name: Name of the tensor for logging purposes

    Returns:
        Similarity score in range [0, 1] where 1 means identical
    """
    x, y = x.data.double(), y.data.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        print(f"\033[33mWARNING: {name} all zero\033[0m")
        return 1
    sim = 2 * (x * y).sum() / denominator
    return sim


def assert_tensors_similar(x, y, eps=1e-8, name="tensor", raise_assert=True):
    """
    Assert that two tensors are similar using a global similarity metric.

    Key differences from torch.testing.assert_close:
    - torch.testing.assert_close: Uses element-wise comparison with rtol/atol, checking
      that |x - y| <= atol + rtol * |y| for each element. It's sensitive to outliers
      and requires all elements to satisfy the tolerance.
    - assert_tensors_similar: Uses a single global similarity score (1 - sim) where sim is the
      normalized dot product. It's more robust to outliers and focuses on overall
      tensor similarity rather than element-wise precision. This is better suited for
      comparing large tensors where a few outlier elements shouldn't fail the test.

    Args:
        x: First tensor to compare
        y: Second tensor to compare
        eps: Maximum allowed difference (1 - similarity), default 1e-8
        name: Name of the tensor for error messages
        raise_assert: Whether to raise assertion error on failure
    """
    sim = calculate_tensor_similarity(x, y, name)
    diff = 1.0 - sim
    if not (0 <= diff <= eps):
        print(f"\033[31mERROR: {name} similarity check failed, diff={diff:.2e} (threshold={eps:.2e})\033[0m")
        if raise_assert:
            assert False  # noqa: B011


def ref_sparse_mla_fwd_interface(Q, KV, Indices, offsets, sm_scale=None, is_casual=True):
    Q = Q.float()
    KV = KV.float()
    all_o = []
    for i in range(offsets.shape[0] - 1):
        q = Q[None, offsets[i] : offsets[i + 1]]
        kv = KV[None, offsets[i] : offsets[i + 1]]
        indices = Indices[None, offsets[i] : offsets[i + 1]].clone()

        indices = indices.transpose(1, 2)
        b, sq, h, dim_q = q.shape
        b, sk, g, _ = kv.shape

        assert kv.shape[-1] == 576, "you should assign dim otherwise"
        dim = 512
        k = kv
        v = kv[..., :dim]

        b, _, _, dim_v = v.shape
        g_index = g
        h_index = h // g
        compressed_casual_mask = torch.arange(0, sq, dtype=torch.int32, device=Q.device).view(-1, 1) >= torch.arange(
            1 - 1, sk * 1, 1, dtype=torch.int32, device=Q.device
        ).view(1, -1)

        indices[indices > sk] = sk
        mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
        mask = mask[..., :-1]
        mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
        mask[:, :, : 1 - 1, 0] = True
        mask = mask.view(b, g_index, 1, sq, sk)

        q = q.view(b, sq, g, -1, dim_q)
        score = torch.einsum("bmghd,bngd->bghmn", q, k)
        sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
        score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
        p = score.softmax(dim=-1)
        p = p.view(b, g_index, h_index, -1, sq, sk)
        p = p.view(b, g, -1, sq, sk)
        o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
        o = o.reshape(b, sq, h, dim_v)
        all_o.append(o.squeeze(0))
    o = torch.cat(all_o, dim=0)
    return o.to(torch.bfloat16)


def test_sparse_mla_fwd(
    B=1,
    S=4096,
    H=128,
    HKV=1,
    DQK=576,
    DV=512,
    topk=1024,  # upstream __main__ case; 2048 exceeds the first varlen segment (2047)
    dtype=torch.bfloat16,
    check_correctness=True,
    block_I=64,
    num_stages=2,
    threads=256,
):
    torch.random.manual_seed(0)
    q = torch.randn((S, H, DQK), dtype=dtype, device="npu").requires_grad_(True)
    kv = torch.randn((S, HKV, DQK), dtype=dtype, device="npu").requires_grad_(True)
    offsets = torch.tensor([0, S // 2 - 1, S], dtype=torch.int32, device="npu")

    indices = torch.full((S, HKV, topk), S, dtype=torch.int32, device="npu")
    for i in range(offsets.shape[0] - 1):
        seq_len = (offsets[i + 1] - offsets[i]).item()
        assert seq_len >= topk
        for t in range(seq_len):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[offsets[i] + t, h, : len(i_i)] = i_i

    tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices, offsets, block_I=block_I, num_stages=num_stages, threads=threads)

    if check_correctness:
        # otherwise may cause out of memory
        ref_out = ref_sparse_mla_fwd_interface(q, kv, indices, offsets)
        assert_tensors_similar(tl_out, ref_out, eps=1e-2, name="out")
        print("assert_tensors_similar passed")

    def fn():
        return sparse_mla_fwd_interface(q, kv, indices, offsets, block_I=block_I, num_stages=num_stages, threads=threads)

    from tilelang.profiler import do_bench

    ms = do_bench(
        fn,
        rep=100,
        warmup=250,
    )
    print(f"Average time: {ms:.3f} ms")
    print("fwd io bandwidth = ", (B * S * DQK * topk * 2) / (ms * 1e-3) / 1e12)
    print("fwd tflops = ", (B * S * (DQK + DV) * topk * 2 * H) / (ms * 1e-3) / 1e12)


if __name__ == "__main__":
    test_sparse_mla_fwd(
        B=1,
        S=4096,
        H=128,
        HKV=1,
        DQK=576,
        DV=512,
        topk=1024,
        dtype=torch.bfloat16,
        check_correctness=True,
        block_I=64,
        num_stages=2,
        threads=256,
    )
