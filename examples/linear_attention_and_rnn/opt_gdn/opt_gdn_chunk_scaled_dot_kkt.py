import tilelang
from tilelang import language as T
import torch

'''
Functionality:
A = strictLower(diag(Beta) * (Gamma \odot K * K^T))
where
Gamma_{i,j} = exp(g_i - g_j)
'''

pass_configs = {
	tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}

@tilelang.jit(out_idx=[-1], workspace_idx=[-2], pass_configs=pass_configs)
def kkt_ker(B, H, L, DK, C, BK = None, dtype="float16", accum_dtype="float"):
	if BK == None:
		BK = DK
	chunk_num = T.ceildiv(L, C)
	bk_num = T.ceildiv(DK, BK)
	VEC_NUM = 2

	@T.prim_func
	def main(
			K: T.Tensor([B, H, L, DK], dtype),
			Beta: T.Tensor([B, H, L], dtype),
			G: T.Tensor([B, H, L], accum_dtype),
			Msk: T.Tensor([C, C], accum_dtype),
			workspace: T.Tensor([B, H, L, C], dtype),
			A: T.Tensor([B, H, L, C], dtype),
	):
		with T.Kernel(B * H * chunk_num, is_npu=True) as (cid, vid):
			bx = cid % chunk_num
			by = (cid // chunk_num) % H
			bz = (cid // chunk_num) // H

			beta_ub_half = T.alloc_ub([C // VEC_NUM,], dtype)
			a_ub_half = T.alloc_ub([C // VEC_NUM, C], dtype)
			a_ub = T.alloc_ub([C // VEC_NUM, C], accum_dtype)
			msk_ub = T.alloc_ub([C // VEC_NUM, C], accum_dtype)
			coeff_ub = T.alloc_ub([C // VEC_NUM, C], accum_dtype)
			beta_ub = T.alloc_ub([C // VEC_NUM,], accum_dtype)
			g_ub = T.alloc_ub([C,], accum_dtype)
			g_v_ub = T.alloc_ub([C // VEC_NUM,], accum_dtype)

			k_l1 = T.alloc_L1([C, BK], dtype)
			a_l0 = T.alloc_L0C([C, C], accum_dtype)

			with T.Scope("C"):
				for i in T.serial(bk_num):
					T.copy(K[bz, by, bx * C, i * BK], k_l1)
					T.gemm_v0(k_l1, k_l1, a_l0, transpose_B = True, init = (i == 0))
				T.copy(a_l0, workspace[bz, by, bx * C, 0])
				T.set_cross_flag("FIX", 0)

			with T.Scope("V"):
				T.copy(G[bz, by, bx * C], g_ub)
				T.copy(Beta[bz, by, bx * C + vid * C // VEC_NUM], beta_ub_half)
				T.set_flag("mte2", "v", 0)
				T.wait_flag("mte2", "v", 0)
				T.copy(beta_ub_half, beta_ub)
				T.copy(g_ub[vid * C // VEC_NUM : (vid + 1) * C // VEC_NUM], g_v_ub)
				T.tile.fill(a_ub, 0.0)
				T.tile.ln(beta_ub, beta_ub)
				T.tile.add(g_v_ub, g_v_ub, beta_ub)
				for i in range((C // VEC_NUM) // 4):
					tmp0 = g_v_ub[i * 4]
					tmp1 = g_v_ub[i * 4 + 1]
					tmp2 = g_v_ub[i * 4 + 2]
					tmp3 = g_v_ub[i * 4 + 3]
					T.tile.sub(coeff_ub[i * 4, :], g_ub, tmp0)
					T.tile.sub(coeff_ub[i * 4 + 1, :], g_ub, tmp1)
					T.tile.sub(coeff_ub[i * 4 + 2, :], g_ub, tmp2)
					T.tile.sub(coeff_ub[i * 4 + 3, :], g_ub, tmp3)
				T.tile.sub(coeff_ub, a_ub, coeff_ub)
				T.tile.exp(coeff_ub, coeff_ub)
				T.copy(Msk[vid * C // VEC_NUM, 0], msk_ub)

				T.wait_cross_flag(0)
				T.copy(workspace[bz, by, bx * C + vid * C // VEC_NUM, 0], a_ub_half)
				T.set_flag("mte2", "v", 0)
				T.wait_flag("mte2", "v", 0)
				T.copy(a_ub_half, a_ub)
				T.tile.mul(a_ub, a_ub, coeff_ub)
				T.tile.mul(a_ub, a_ub, msk_ub)
				T.copy(a_ub, a_ub_half)
				T.set_flag("v", "mte3", 0)
				T.wait_flag("v", "mte3", 0)
				T.copy(a_ub_half, A[bz, by, bx * C + vid * C // VEC_NUM, 0])

	return main

def kkt(k, beta, g, C):
	B, H, L, DK = k.shape
	msk = torch.tril(torch.ones((C, C)), diagonal = -1).npu().to(torch.float)
	ker = kkt_ker(B, H, L, DK, C)
	a = ker(k, beta, g, msk)
	return a

def ref_kkt(k, beta, g, C):
	B, H, L, DK = k.shape
	chunk_num = (L + C - 1) // C
	a = torch.zeros((B, H, L, C)).npu().to(torch.float)
	beta = beta.float()

	for i in range(chunk_num):
		k_c = k[:, :, i * C : (i + 1) * C, :]
		beta_c = beta[:, :, i * C : (i + 1) * C]
		g_c = g[:, :, i * C : (i + 1) * C]
		kkt = torch.einsum("bhid,bhjd->bhij", k_c, k_c).float()
		gamma = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)
		gamma = torch.exp(gamma)
		a_c = (kkt * beta_c.unsqueeze(-1) * gamma).tril(-1)
		a[:, :, i * C : (i + 1) * C, :] = a_c
	
	return a.to(torch.float16)

if __name__ == "__main__":
	tilelang.cache.clear_cache()
	torch.manual_seed(0)
	torch.set_printoptions(threshold = float('inf'), sci_mode = True)

	test_configs = [
		(1, 1, 64, 64, 64),
	]

	for B, H, L, DK, C in test_configs:
		print(f"Testing KKT with B={B}, H={H}, L={L}, DK={DK}, C={C}")
		k = torch.randn((B, H, L, DK)).npu().to(torch.float16)
		beta = torch.rand((B, H, L)).npu().to(torch.float16)
		g = torch.randn((B, H, L)).npu().to(torch.float)
		a = kkt(k, beta, g, C)
		ref_a = ref_kkt(k, beta, g, C)
		torch.testing.assert_close(a.cpu(), ref_a.cpu(), rtol=1e-5, atol=1e-5)
		print("Test passed!")

	print("Kernel Output Match!")
