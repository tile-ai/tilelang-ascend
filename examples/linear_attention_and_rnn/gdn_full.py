import tilelang
import torch
import torch.nn.functional as F

from gdn.gdn_chunk_cumsum import chunk_cumsum, ref_chunk_cumsum
from gdn.gdn_chunk_h import chunk_h, ref_chunk_h
from gdn.gdn_chunk_o import chunk_o, ref_chunk_o
from gdn.gdn_chunk_scaled_dot_kkt import kkt, ref_kkt
from gdn.gdn_wy_fast import wy_fast, ref_wy_fast

def solve_tri(a):
	B, H, L, C = a.shape
	chunk_num = (L + C - 1) // C
	I = torch.eye(C).npu().to(torch.float).view(1, 1, C, C)
	o = torch.zeros((B, H, L, C)).npu().to(torch.float)
	a = a.float()
	for i in range(chunk_num):
		a_c = a[:, :, i * C : (i + 1) * C, :]
		o_c = torch.linalg.solve_triangular(a_c + I, I, upper = False, left = True)
		o[:, :, i * C : (i + 1) * C, :] = o_c
	return o.to(torch.float16)

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
	a = solve_tri(a)
	w, u = ref_wy_fast(k, v, beta, g, a, C)
	s, nv, fs = ref_chunk_h(k, w, u, g, C)
	o = ref_chunk_o(q, k, nv, s, g, C)
	return o

def kernel_chunk_gdn(q, k, v, g, beta, C, BK, BV):
	g = chunk_cumsum(g, C)
	a = kkt(k, beta, g, C, BK)
	a = solve_tri(a)
	w, u = wy_fast(k, v, beta, g, a, C, BK, BV)
	s, nv, fs = chunk_h(k, w, u, g, C, BK, BV)
	o = chunk_o(q, k, nv, s, g, C, BK, BV)
	return o

tilelang.cache.clear_cache()
torch.manual_seed(0)
torch.set_printoptions(threshold = float('inf'), sci_mode = True)

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
	ref_chunk_o = ref_chunk_gdn(q, k, v, g, beta, C)
	ker_o = kernel_chunk_gdn(q, k, v, g, beta, C, BK, BV)
	torch.testing.assert_close(ref_chunk_o.cpu(), ref_o.cpu(), rtol=1e-3, atol=1e-3)
	torch.testing.assert_close(ker_o.cpu(), ref_chunk_o.cpu(), rtol=1e-5, atol=1e-5)
	print("Test passed!")

print("Kernel Output Match!")
