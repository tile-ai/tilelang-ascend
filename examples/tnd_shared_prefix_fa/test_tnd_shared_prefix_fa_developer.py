"""TND Shared-Prefix FlashAttention (Developer mode) layered tests: L0 + main(--level)."""

import argparse
import math
import os
import sys

import tilelang
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tnd_shared_prefix_fa_developer import tnd_shared_prefix_fa_developer  # noqa: E402


# ========== block_metadata construction (design.md §4.6) ==========
def build_block_metadata(
    shared_prefix_len,
    private_q_lens,
    block_M,
    device,
):
    metadata_list = []

    if shared_prefix_len > 0:
        for i in range(math.ceil(shared_prefix_len / block_M)):
            q_start = i * block_M
            q_valid = min(block_M, shared_prefix_len - q_start)
            metadata_list.append([q_start, q_valid, 0, 0])

    priv_offset = 0
    for _b, priv_len in enumerate(private_q_lens):
        if priv_len == 0:
            continue
        q_packed_offset = shared_prefix_len + priv_offset
        for i in range(math.ceil(priv_len / block_M)):
            q_start = q_packed_offset + i * block_M
            q_valid = min(block_M, priv_len - i * block_M)
            metadata_list.append([q_start, q_valid, priv_offset, priv_len])
        priv_offset += priv_len

    return torch.tensor(metadata_list, dtype=torch.int32, device=device)


# ========== Golden reference (CPU computation) ==========
def ref_tnd_shared_prefix_fa(
    Q,
    K_shared,
    V_shared,
    K_private,
    V_private,
    shared_prefix_len,
    private_q_lens,
    q_head,
    kv_head,
    head_dim,
    sm_scale=None,
    causal_mask=False,
):
    sm_scale = (1.0 / head_dim) ** 0.5 if sm_scale is None else sm_scale
    group_size = q_head // kv_head
    total_q = Q.shape[0]
    dtype = Q.dtype
    Q = Q.float()
    K_shared = K_shared.float()
    V_shared = V_shared.float()
    K_private = K_private.float()
    V_private = V_private.float()

    O = torch.zeros((total_q, q_head, head_dim), dtype=torch.float32)

    if shared_prefix_len > 0:
        q_seg = Q[:shared_prefix_len]
        for h_q in range(q_head):
            h_kv = h_q // group_size
            q = q_seg[:, h_q, :]
            k = K_shared[:, h_kv, :]
            v = V_shared[:, h_kv, :]
            scores = torch.matmul(q, k.T) * sm_scale
            if causal_mask:
                mask = torch.triu(
                    torch.ones(shared_prefix_len, shared_prefix_len), diagonal=1
                ).bool()
                scores = scores.masked_fill(mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            O[:shared_prefix_len, h_q, :] = torch.matmul(attn, v)

    priv_offset = 0
    for _b, priv_len in enumerate(private_q_lens):
        if priv_len == 0:
            continue
        q_start = shared_prefix_len + priv_offset
        q_seg = Q[q_start : q_start + priv_len]
        k_priv = K_private[priv_offset : priv_offset + priv_len]
        v_priv = V_private[priv_offset : priv_offset + priv_len]
        for h_q in range(q_head):
            h_kv = h_q // group_size
            q = q_seg[:, h_q, :]
            if shared_prefix_len > 0:
                k = torch.cat([K_shared[:, h_kv, :], k_priv[:, h_kv, :]], dim=0)
                v = torch.cat([V_shared[:, h_kv, :], v_priv[:, h_kv, :]], dim=0)
            else:
                k = k_priv[:, h_kv, :]
                v = v_priv[:, h_kv, :]
            scores = torch.matmul(q, k.T) * sm_scale
            if causal_mask:
                total_kv = shared_prefix_len + priv_len
                q_pos = torch.arange(q_start, q_start + priv_len).unsqueeze(1)
                kv_pos = torch.arange(total_kv).unsqueeze(0)
                mask = kv_pos > q_pos
                scores = scores.masked_fill(mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            O[q_start : q_start + priv_len, h_q, :] = torch.matmul(attn, v)
        priv_offset += priv_len

    return O.to(dtype)


# ========== Precision standard ==========
def get_precision(dtype):
    fp_table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
    }
    if dtype in {"int8", "int16", "int32", "int64", "uint8"}:
        return (0.0, 0.0, 0.0, 1.0)
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a, g = actual.detach().cpu(), golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:
        mism = (a != g).sum().item()
        return mism == 0, 1.0 - mism / max(a.numel(), 1), (0.0 if mism == 0 else float("inf"))
    a, g = a.float(), g.float()
    m = torch.isfinite(g)
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    return (ratio >= required_ratio and max_abs <= max_abs_limit), ratio, max_abs


# ========== Test runner ==========
def run_test_case(
    batch,
    q_head,
    kv_head,
    head_dim,
    shared_prefix_len,
    private_q_lens,
    block_M,
    block_N,
    dtype_str,
    tag,
    level="L0",
    causal_mask=False,
    threads=2,
):
    try:
        total_q = shared_prefix_len + sum(private_q_lens)
        total_private_kv = sum(private_q_lens)
        max_private_kv_len = max(private_q_lens) if private_q_lens else 0
        total_q_blocks = math.ceil(shared_prefix_len / block_M) + sum(
            math.ceil(l / block_M) for l in private_q_lens
        )

        torch.manual_seed(0)
        npu_dtype = getattr(torch, dtype_str)

        Q_cpu = torch.randn(total_q, q_head, head_dim, dtype=npu_dtype)
        K_shared_cpu = (
            torch.randn(shared_prefix_len, kv_head, head_dim, dtype=npu_dtype)
            if shared_prefix_len > 0
            else torch.zeros(0, kv_head, head_dim, dtype=npu_dtype)
        )
        V_shared_cpu = (
            torch.randn(shared_prefix_len, kv_head, head_dim, dtype=npu_dtype)
            if shared_prefix_len > 0
            else torch.zeros(0, kv_head, head_dim, dtype=npu_dtype)
        )
        K_private_cpu = torch.randn(total_private_kv, kv_head, head_dim, dtype=npu_dtype)
        V_private_cpu = torch.randn(total_private_kv, kv_head, head_dim, dtype=npu_dtype)

        block_metadata = build_block_metadata(shared_prefix_len, private_q_lens, block_M, "cpu")

        sm_scale = (1.0 / head_dim) ** 0.5

        kernel = tnd_shared_prefix_fa_developer(
            q_head=q_head,
            kv_head=kv_head,
            head_dim=head_dim,
            shared_prefix_len=shared_prefix_len,
            max_private_kv_len=max_private_kv_len,
            total_q=total_q,
            total_private_kv=total_private_kv,
            total_q_blocks=total_q_blocks,
            block_M=block_M,
            block_N=block_N,
            sm_scale=sm_scale,
            dtype_str=dtype_str,
            causal_mask=causal_mask,
            threads=threads,
        )

        Q = Q_cpu.npu()
        KS = K_shared_cpu.npu()
        VS = V_shared_cpu.npu()
        KP = K_private_cpu.npu()
        VP = V_private_cpu.npu()
        bm = block_metadata.npu()

        output = kernel(Q, KS, VS, KP, VP, bm)
        torch.npu.synchronize()

        output = output.permute(1, 0, 2)

        ref = ref_tnd_shared_prefix_fa(
            Q_cpu,
            K_shared_cpu,
            V_shared_cpu,
            K_private_cpu,
            V_private_cpu,
            shared_prefix_len,
            private_q_lens,
            q_head,
            kv_head,
            head_dim,
            sm_scale=sm_scale,
            causal_mask=causal_mask,
        )

        passed, ratio, max_abs = check_precision(output, ref, dtype_str)

        status = "PASS" if passed else "FAIL"
        warn_tag = "PRECISION" if level in ("L0", "L1") else "BOUNDARY"
        print(
            f"[{warn_tag}_{status}] {tag} "
            f"batch={batch} q_head={q_head} kv_head={kv_head} "
            f"sp_len={shared_prefix_len} priv_lens={private_q_lens} "
            f"dtype={dtype_str} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}"
        )
        return passed
    except Exception as e:
        print(f"[PRECISION_FAIL] {tag}: {e}")
        import traceback

        traceback.print_exc()
        return False


# ========== L0 tests (design.md §9.2) ==========
def test_l0():
    ok = True

    # l0_business: business typical scenario (non-aligned, sp=24, seq=150)
    ok &= run_test_case(
        batch=10,
        q_head=14,
        kv_head=2,
        head_dim=64,
        shared_prefix_len=24,
        private_q_lens=[150] * 10,
        block_M=128,
        block_N=64,
        dtype_str="float16",
        tag="l0_business",
        level="L0",
    )

    # l0_p99: business p99 scenario (seq=218)
    ok &= run_test_case(
        batch=10,
        q_head=14,
        kv_head=2,
        head_dim=64,
        shared_prefix_len=24,
        private_q_lens=[218] * 10,
        block_M=128,
        block_N=64,
        dtype_str="float16",
        tag="l0_p99",
        level="L0",
    )

    # l0_aligned: block-aligned case for baseline verification
    ok &= run_test_case(
        batch=2,
        q_head=4,
        kv_head=2,
        head_dim=64,
        shared_prefix_len=128,
        private_q_lens=[128, 128],
        block_M=128,
        block_N=64,
        dtype_str="float16",
        tag="l0_aligned",
        level="L0",
    )

    # l0_causal: causal mask enabled, block-aligned
    ok &= run_test_case(
        batch=2,
        q_head=4,
        kv_head=2,
        head_dim=64,
        shared_prefix_len=64,
        private_q_lens=[128, 128],
        block_M=128,
        block_N=64,
        dtype_str="float16",
        tag="l0_causal",
        level="L0",
        causal_mask=True,
        threads=1,
    )

    # l0_bf16: bfloat16 dtype
    ok &= run_test_case(
        batch=10,
        q_head=14,
        kv_head=2,
        head_dim=64,
        shared_prefix_len=24,
        private_q_lens=[150] * 10,
        block_M=128,
        block_N=64,
        dtype_str="bfloat16",
        tag="l0_bf16",
        level="L0",
    )

    return ok


# ========== L1/L2/Boundary: stubs ==========
def test_l1():
    print("[L1] not expanded yet - run tilelang-op-test-design (scenario B)")
    return True


def test_l2():
    print("[L2] not expanded yet - run tilelang-op-test-design (scenario B)")


def test_boundary():
    print("[BOUNDARY] not expanded yet - run tilelang-op-test-design (scenario B)")


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(0)

    blocking_ok = True
    if args.level in ("l0", "all"):
        blocking_ok &= test_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_l1()
    if args.level in ("l2", "all"):
        test_l2()
    if args.level in ("boundary", "all"):
        test_boundary()

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
