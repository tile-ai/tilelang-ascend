"""Test for the sparse_mla_bwd operator (DSA sparse MLA backward, Ascend NPU).

Follows the upstream tile-ai/tilelang
``examples/dsa_sparse_finetune/sparse_mla_bwd.py`` test: same case, shapes,
dtypes, input construction (random causal top-k indices with out-of-range
fills) and autograd golden, with two Ascend adaptations: tensors live on the
NPU, and the forward outputs (o, lse) fed to the backward are computed with
the exact fp32 PyTorch reference of the forward (the forward kernel ships as
a separate example).
"""

import torch
import torch_npu  # noqa: F401  (registers the NPU device)

from sparse_mla_bwd import prepare_token_indices, sparse_mla_bwd


def calculate_tensor_similarity(x, y, name="tensor"):
    """Normalized dot-product similarity (1 = identical)."""
    x, y = x.data.double(), y.data.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        print(f"\033[33mWARNING: {name} all zero\033[0m")
        return 1
    return 2 * (x * y).sum() / denominator


def assert_tensors_similar(x, y, eps=1e-8, name="tensor"):
    """Assert a global similarity score 1 - sim <= eps (upstream metric)."""
    sim = calculate_tensor_similarity(x, y, name)
    diff = 1.0 - sim
    if not (0 <= diff <= eps):
        print(f"\033[31mERROR: {name} similarity check failed, diff={diff:.2e} (threshold={eps:.2e})\033[0m")
        assert False  # noqa: B011


def ref_sparse_mla_fwd_interface(Q, KV, Indices, offsets, sm_scale=None, is_casual=True):
    """fp32 PyTorch reference of the sparse MLA forward (upstream reference)."""
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


def ref_sparse_mla_fwd_lse(Q, KV, Indices, offsets, sm_scale=None):
    """fp32 logsumexp over the selected causal index slots ([S, H] float32).

    Matches the forward kernel's lse semantics: only the topk selected
    positions (with multiplicity) enter the normalizer; invalid slots are
    masked to -inf.
    """
    Q = Q.float()
    KV = KV.float()
    S, H, DDT = Q.shape
    G = KV.shape[1]
    head_kv = H // G
    if sm_scale is None:
        sm_scale = DDT**-0.5
    lse = torch.zeros(S, H, dtype=torch.float32, device=Q.device)
    off = offsets.tolist()
    chunk = 256
    for i in range(len(off) - 1):
        bos, eos = int(off[i]), int(off[i + 1])
        for g in range(G):
            for t0 in range(bos, eos, chunk):
                t1 = min(t0 + chunk, eos)
                s_i = torch.arange(t0 - bos, t1 - bos, device=Q.device)  # [n]
                idx = Indices[t0:t1, g].long()  # [n, topk]
                valid = (idx >= 0) & (idx <= s_i[:, None])  # [n, topk]
                rows = (idx + bos).clamp(0, S - 1)  # [n, topk]
                k_g = KV[rows, g, :]  # [n, topk, DDT] fp32
                q_h = Q[t0:t1, g * head_kv : (g + 1) * head_kv, :]  # [n, hk, DDT]
                score = torch.einsum("nhd,njd->nhj", q_h, k_g) * sm_scale
                score = score.masked_fill(~valid[:, None, :], float("-inf"))
                lse[t0:t1, g * head_kv : (g + 1) * head_kv] = torch.logsumexp(score, dim=-1)
    return lse


def ref_sparse_mla_bwd_interface(q, kv, do, indices, offsets, sm_scale=None, is_casual=True):
    """Autograd golden: backward through the fp32 forward reference."""
    q = q.detach().clone()
    kv = kv.detach().clone()
    q.requires_grad = True
    kv.requires_grad = True
    o = ref_sparse_mla_fwd_interface(q, kv, indices, offsets, sm_scale, is_casual)
    o.backward(do)
    return q.grad, kv.grad


def test_sparse_mla_bwd(
    B=1,
    S=2048,
    H=64,
    HKV=1,
    DQKV=576,
    DV=512,
    topk=512,
    dtype=torch.bfloat16,
    check_correctness=True,
    target="ascendc",
):
    # Prepare data
    q = torch.randn((S, H, DQKV), dtype=dtype, device="npu").requires_grad_(True)
    kv = torch.randn((S, HKV, DQKV), dtype=dtype, device="npu").requires_grad_(True)
    do = torch.randn((S, H, DV), dtype=dtype, device="npu")
    offsets = torch.tensor([0, S], dtype=torch.int32, device="npu")

    indices = torch.full((S, HKV, topk), S, dtype=torch.int32, device="npu")
    for i in range(offsets.shape[0] - 1):
        seq_len = (offsets[i + 1] - offsets[i]).item()
        assert seq_len >= topk
        for t in range(seq_len):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[offsets[i] + t, h, : len(i_i)] = i_i

    # Forward outputs (o, lse) that the backward consumes
    o = ref_sparse_mla_fwd_interface(q, kv, indices, offsets)
    lse = ref_sparse_mla_fwd_lse(q, kv, indices, offsets)

    tl_dq, tl_dkv = sparse_mla_bwd(q, kv, o, do, indices, lse, offsets, target=target)
    ref_dq, ref_dkv = ref_sparse_mla_bwd_interface(q, kv, do, indices, offsets)

    if check_correctness:
        assert_tensors_similar(tl_dq, ref_dq, eps=1e-4, name="dq")
        assert_tensors_similar(tl_dkv, ref_dkv, eps=1e-4, name="dkv")
        print("assert_tensors_similar passed")

    per_token_flop = 2 * sum(
        [
            H * DV * topk,
            H * DQKV * topk,
            H * DQKV * topk,
            H * DQKV * topk,
            H * DV * topk,
        ]
    )
    from tilelang.profiler import do_bench

    def fn():
        return sparse_mla_bwd(q, kv, o, do, indices, lse, offsets, target=target)

    ms = do_bench(fn, rep=100, warmup=250)
    print(f"Average time: {ms:.3f} ms")
    print("bwd io bandwidth = ", (B * S * max(DQKV * 2, DQKV + DV) * topk * 2) / (ms * 1e-3) / 1e12)
    print("bwd tflops = ", per_token_flop * S / (ms * 1e-3) / 1e12)


def test_prepare_token_indices_inplace_mutation():
    """Regression test: the prepare_token_indices single-entry cache must
    invalidate on in-place offsets mutation, not on tensor identity alone.

    offsets [0, 256, 512] updated in place to [0, 128, 512]: token 128 must
    move from (batch=0, position=128) to (batch=1, position=0)."""
    offsets = torch.tensor([0, 256, 512], dtype=torch.int32, device="npu")
    ti = prepare_token_indices(offsets)
    assert ti[128].tolist() == [0, 128]

    offsets[1] = 128  # in-place update of the SAME tensor object
    ti = prepare_token_indices(offsets)
    assert ti[127].tolist() == [0, 127]
    assert ti[128].tolist() == [1, 0]
    assert ti[129].tolist() == [1, 1]
    assert ti[511].tolist() == [1, 383]

    # a distinct tensor object with equal content must stay correct too
    offsets2 = torch.tensor([0, 128, 512], dtype=torch.int32, device="npu")
    assert prepare_token_indices(offsets2)[128].tolist() == [1, 0]
    print("prepare_token_indices in-place mutation: cache invalidation passed")


def test_sparse_mla_bwd_inplace_offsets_mutation(
    S=512,
    H=16,
    HKV=1,
    DQKV=576,
    DV=512,
    topk=128,
    dtype=torch.bfloat16,
    target="ascendc",
):
    """Regression test: consecutive sparse_mla_bwd calls with the same
    offsets tensor object whose content is updated in place between the
    calls — the second call's gradients must match the reference for the
    NEW segmentation.

    With identity-only cache keying the second call reused the stale
    (batch, position) mapping against the freshly read offsets, mixing
    segment starts and causal boundaries across segments (wrong dQ/dKV)."""
    q = torch.randn((S, H, DQKV), dtype=dtype, device="npu").requires_grad_(True)
    kv = torch.randn((S, HKV, DQKV), dtype=dtype, device="npu").requires_grad_(True)
    do = torch.randn((S, H, DV), dtype=dtype, device="npu")
    offsets = torch.tensor([0, 256, 512], dtype=torch.int32, device="npu")

    def build_inputs(offsets):
        indices = torch.full((S, HKV, topk), S, dtype=torch.int32, device="npu")
        for i in range(offsets.shape[0] - 1):
            seq_len = (offsets[i + 1] - offsets[i]).item()
            assert seq_len >= topk
            for t in range(seq_len):
                for h in range(HKV):
                    i_i = torch.randperm(max(1, t))[:topk]
                    indices[offsets[i] + t, h, : len(i_i)] = i_i
        o = ref_sparse_mla_fwd_interface(q, kv, indices, offsets)
        lse = ref_sparse_mla_fwd_lse(q, kv, indices, offsets)
        return indices, o, lse

    # call 1: offsets [0, 256, 512] (also warms the token_indices cache)
    indices, o, lse = build_inputs(offsets)
    tl_dq, tl_dkv = sparse_mla_bwd(q, kv, o, do, indices, lse, offsets, target=target)
    ref_dq, ref_dkv = ref_sparse_mla_bwd_interface(q, kv, do, indices, offsets)
    assert_tensors_similar(tl_dq, ref_dq, eps=1e-4, name="dq (before mutation)")
    assert_tensors_similar(tl_dkv, ref_dkv, eps=1e-4, name="dkv (before mutation)")

    # in-place update of the SAME tensor object: [0, 256, 512] -> [0, 128, 512]
    offsets[1] = 128
    indices, o, lse = build_inputs(offsets)
    tl_dq, tl_dkv = sparse_mla_bwd(q, kv, o, do, indices, lse, offsets, target=target)
    ref_dq, ref_dkv = ref_sparse_mla_bwd_interface(q, kv, do, indices, offsets)
    assert_tensors_similar(tl_dq, ref_dq, eps=1e-4, name="dq (after mutation)")
    assert_tensors_similar(tl_dkv, ref_dkv, eps=1e-4, name="dkv (after mutation)")
    print("sparse_mla_bwd in-place offsets mutation passed")


if __name__ == "__main__":
    test_prepare_token_indices_inplace_mutation()
    test_sparse_mla_bwd_inplace_offsets_mutation()
    test_sparse_mla_bwd(B=1, S=2048, H=64, HKV=1, DQKV=576, DV=512, topk=512, dtype=torch.bfloat16, check_correctness=True)
