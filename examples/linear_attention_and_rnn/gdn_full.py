import tilelang
import torch


def _check_precision(a, b):
    a, b = a.detach().cpu(), b.detach().cpu()
    p = {
        "torch.float16": (2**-14, 2**-9, 0.1),
        "torch.bfloat16": (2**-10, 2**-6, 1.0),
        "torch.float32": (2**-16, 2**-10, 0.01),
        "hifloat32": (2**-16, 2**-10, 0.01),
        "float8_e4m3": (2**-4, 2**-2, 1.0),
        "float8_e5m2": (2**-3, 2**-1, 0.1),
    }.get(
        "float8_e4m3" if "float8_e4m3" in str(b.dtype) else "float8_e5m2" if "float8_e5m2" in str(b.dtype) else str(b.dtype),
        (2**-14, 2**-9, 0.1),
    )
    if not (a.is_floating_point() or b.is_floating_point()):
        if not torch.equal(a, b):
            raise AssertionError("integer mismatch")
        return
    if not (torch.equal(torch.isnan(a), torch.isnan(b)) and torch.equal(torch.isinf(a), torch.isinf(b))):
        raise AssertionError("NaN/Inf structure mismatch")
    v = torch.isfinite(b)
    if v.any():
        d = (a.float() - b.float()).abs()[v]
        d = torch.where(torch.isfinite(d), d, torch.full_like(d, float("inf")))
        t = p[0] + p[1] * b.float().abs()[v]
        if (d <= t).float().mean().item() < 0.99 or d.max().item() > p[2]:
            raise AssertionError("precision mismatch")


import torch.nn.functional as F

from gdn.gdn_chunk_cumsum import chunk_cumsum, ref_chunk_cumsum
from gdn.gdn_chunk_h import chunk_h, ref_chunk_h
from gdn.gdn_chunk_o import chunk_o, ref_chunk_o
from gdn.gdn_chunk_scaled_dot_kkt import kkt, ref_kkt
from gdn.gdn_solve_tril import solve_tril, ref_solve_tril
from gdn.gdn_wy_fast import wy_fast, ref_wy_fast


def ref_seq_gdn(q, k, v, g, beta):
    g = torch.exp(g)
    q = q.float()
    k = k.float()
    v = v.float()
    beta = beta.float()
    Batch, H, L, DK = q.shape
    DV = v.shape[-1]
    S = torch.zeros((Batch, H, DV, DK)).npu().to(torch.float)
    o = torch.empty((Batch, H, L, DV)).npu().to(torch.float)
    I = torch.eye(DK).npu().to(torch.float).view(1, 1, DK, DK)
    for i in range(0, L):
        q_i = q[:, :, i, :]
        k_i = k[:, :, i, :]
        v_i = v[:, :, i, :]
        beta_i = beta[:, :, i].view(Batch, H, 1, 1)
        g_i = g[:, :, i].view(Batch, H, 1, 1)
        kkt = k_i.unsqueeze(-1) * k_i.unsqueeze(-2)
        vkt = v_i.unsqueeze(-1) * k_i.unsqueeze(-2)
        A_i = g_i * (I - beta_i * kkt)
        term_1 = torch.matmul(S, A_i)
        term_2 = beta_i * vkt
        S = term_1 + term_2
        o[:, :, i, :] = torch.einsum("bhpq,bhq->bhp", S, q_i)
    return o.to(torch.float16)


def ref_chunk_gdn(q, k, v, g, beta, C):
    g = ref_chunk_cumsum(g, C)
    a = ref_kkt(k, beta, g, C)
    a = ref_solve_tril(a)
    w, u = ref_wy_fast(k, v, beta, g, a, C)
    s, nv, fs = ref_chunk_h(k, w, u, g, C)
    o = ref_chunk_o(q, k, nv, s, g, C)
    return o


def kernel_chunk_gdn(q, k, v, g, beta, C, BK, BV):
    g = chunk_cumsum(g, C)
    a = kkt(k, beta, g, C, BK)
    a = solve_tril(a)
    w, u = wy_fast(k, v, beta, g, a, C, BK, BV)
    s, nv, fs = chunk_h(k, w, u, g, C, BK, BV)
    o = chunk_o(q, k, nv, s, g, C, BK, BV)
    return o


tilelang.cache.clear_cache()
torch.manual_seed(0)
torch.set_printoptions(threshold=float("inf"), sci_mode=True)

test_configs = [
    (2, 32, 1024, 64, 64, 32, 32, 32),
]

for Batch, H, L, DK, DV, C, BK, BV in test_configs:
    print(f"Testing GDN with Batch={Batch}, H={H}, L={L}, DK={DK}, DV={DV}, C={C}, BK={BK}, BV={BV}")
    q = torch.randn((Batch, H, L, DK)).npu().to(torch.float16)
    k = torch.randn((Batch, H, L, DK)).npu().to(torch.float16)
    v = torch.randn((Batch, H, L, DV)).npu().to(torch.float16)
    q, k = F.normalize(q, dim=-1, p=2), F.normalize(k, dim=-1, p=2)
    g = torch.randn((Batch, H, L)).npu().to(torch.float)
    g = F.logsigmoid(g)
    beta = torch.rand((Batch, H, L)).npu().to(torch.float16)
    ref_o = ref_seq_gdn(q, k, v, g, beta)
    chunk_o_ref = ref_chunk_gdn(q, k, v, g, beta, C)
    ker_o = kernel_chunk_gdn(q, k, v, g, beta, C, BK, BV)
    _check_precision(chunk_o_ref.cpu(), ref_o.cpu())
    _check_precision(ker_o.cpu(), chunk_o_ref.cpu())
    print("Test passed!")

print("Kernel Output Match!")
