import torch
import torch_npu  # noqa: F401

from torch_tl_ascend.op_source.flash_attn_bhsd import flash_attention_fwd


def _check_precision(actual, golden):
    actual, golden = actual.detach().cpu().float(), golden.detach().cpu().float()
    if actual.shape != golden.shape:
        return False, 0.0, float("inf")
    error = (actual - golden).abs()
    finite = torch.isfinite(golden)
    if not finite.any():
        return True, 1.0, 0.0
    error = torch.where(torch.isfinite(error), error, torch.full_like(error, float("inf")))
    ratio, maximum = (error[finite] <= 2**-14 + 2**-9 * golden[finite].abs()).float().mean().item(), error[finite].max().item()
    return ratio >= 0.99 and maximum <= 0.1, ratio, maximum


def ref_flash_attention(q, k, v):
    q = q.float()
    k = k.float()
    v = v.float()

    acc = torch.einsum("bhsd,bhkd->bhsk", q, k) * (1.0 / q.shape[-1]) ** 0.5
    acc = acc.softmax(dim=-1)
    o = torch.einsum("bhsk,bhkd->bhsd", acc, v)
    return o.to(torch.float16)


if __name__ == "__main__":
    B, S, H, D = 4, 4096, 16, 128

    torch.set_default_device("npu")
    torch.manual_seed(0)

    q = torch.randn((B, H, S, D), dtype=torch.float16)
    k = torch.randn((B, H, S, D), dtype=torch.float16)
    v = torch.randn((B, H, S, D), dtype=torch.float16)

    torch.npu.synchronize()
    print("init successful!")

    kernel = flash_attention_fwd(B, S, H, D)
    output = kernel(q, k, v)
    ref_output = ref_flash_attention(q, k, v)
    torch.npu.synchronize()

    passed, ratio, max_abs = _check_precision(output, ref_output)
    assert passed, f"matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"

    print("Test Passed!")
