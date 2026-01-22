import tilelang
from tilelang import language as T
import torch

'''
Functionality:
U = A * diag(Beta) * V
W = A * diag(exp(g) * Beta) * K
'''

pass_configs = {
	tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}

@tilelang.jit(out_idx=[-2, -1], workspace_idx=[-4, -3], pass_configs=pass_configs)
def wy_fast_ker(B, H, L, DK, DV, C, BK = None, BV = None, dtype="float16", accum_dtype="float"):
	# BK, BV are deprecated
	if BK == None:
		BK = DK
	if BV == None:
		BV = DV
	chunk_num = T.ceildiv(L, C)
	bk_num = T.ceildiv(DK, BK)
	bv_num = T.ceildiv(DV, BV)
	VEC_NUM = 2

	@T.prim_func
	def main(
			K: T.Tensor([B, H, L, DK], dtype),
			V: T.Tensor([B, H, L, DV], dtype),
			Beta: T.Tensor([B, H, L], dtype),
			G: T.Tensor([B, H, L], accum_dtype),
			A: T.Tensor([B, H, L, C], dtype),
			workspace_a1: T.Tensor([B, H, L, C], dtype),
			workspace_a2: T.Tensor([B, H, L, C], dtype),
			W: T.Tensor([B, H, L, DK], dtype),
			U: T.Tensor([B, H, L, DV], dtype),
	):
		with T.Kernel(B * H * chunk_num, is_npu=True) as (cid, vid):
			bx = cid % chunk_num
			by = (cid // chunk_num) % H
			bz = (cid // chunk_num) // H

			a1_ub = T.alloc_ub([C // VEC_NUM, C], accum_dtype)
			a2_ub = T.alloc_ub([C // VEC_NUM, C], accum_dtype)
			beta_r_ub = T.alloc_ub([1, C], accum_dtype)
			beta_2d_ub = T.alloc_ub([C // VEC_NUM, C], accum_dtype)
			g_r_ub = T.alloc_ub([1, C], accum_dtype)
			g_2d_ub = T.alloc_ub([C // VEC_NUM, C], accum_dtype)
			beta_ub = T.alloc_ub([C,], accum_dtype)
			g_ub = T.alloc_ub([C,], accum_dtype)
			a1_ub_half = T.alloc_ub([C // VEC_NUM, C], dtype)
			a2_ub_half = T.alloc_ub([C // VEC_NUM, C], dtype)
			beta_ub_half = T.alloc_ub([C,], dtype)
			tmp_ub = T.alloc_ub([3 * C * C // VEC_NUM], "uint8")

			k_l1 = T.alloc_L1([C, BK], dtype)
			v_l1 = T.alloc_L1([C, BV], dtype)
			a1_l1 = T.alloc_L1([C, C], dtype)
			a2_l1 = T.alloc_L1([C, C], dtype)
			w_l0 = T.alloc_L0C([C, BK], accum_dtype)
			u_l0 = T.alloc_L0C([C, BV], accum_dtype)

			with T.Scope("V"):
				# First calculate A1 = A * diag(exp(g) * Beta), A2 = A * diag(Beta)
				T.copy(Beta[bz, by, bx * C], beta_ub_half)
				T.set_flag("mte2", "v", 0)
				T.wait_flag("mte2", "v", 0)
				T.copy(A[bz, by, bx * C + vid * C // VEC_NUM, 0], a1_ub_half)
				T.copy(beta_ub_half, beta_ub)
				T.copy(beta_ub, beta_r_ub[0, :])
				T.tile.broadcast(beta_2d_ub, beta_r_ub, tmp_ub)
				T.set_flag("mte2", "v", 0)
				T.wait_flag("mte2", "v", 0)
				T.copy(a1_ub_half, a1_ub)
				T.tile.mul(a2_ub, a1_ub, beta_2d_ub) # A2 = A * diag(Beta)
				T.copy(a2_ub, a2_ub_half)
				T.set_flag("v", "mte3", 0)
				T.wait_flag("v", "mte3", 0)
				T.copy(a2_ub_half, workspace_a2[bz, by, bx * C + vid * C // VEC_NUM, 0])
				T.set_cross_flag("MTE3", 2)

				T.copy(G[bz, by, bx * C], g_ub)
				T.set_flag("mte2", "v", 0)
				T.wait_flag("mte2", "v", 0)
				T.tile.exp(g_ub, g_ub)
				T.tile.mul(g_ub, g_ub, beta_ub) # g_ub now stores exp(g) * Beta
				T.copy(g_ub, g_r_ub[0, :])
				T.tile.broadcast(g_2d_ub, g_r_ub, tmp_ub)
				T.tile.mul(a1_ub, a1_ub, g_2d_ub) # A1 = A * diag(exp(g) * Beta)
				T.copy(a1_ub, a1_ub_half)
				T.set_flag("v", "mte3", 0)
				T.wait_flag("v", "mte3", 0)
				T.copy(a1_ub_half, workspace_a1[bz, by, bx * C + vid * C // VEC_NUM, 0])
				T.set_cross_flag("MTE3", 1)
			
			with T.Scope("C"):
				T.copy(K[bz, by, bx * C, 0], k_l1)
				T.copy(V[bz, by, bx * C, 0], v_l1)

				# Then calculate U = A2 * V, W = A1 * K
				T.wait_cross_flag(2)
				T.copy(workspace_a2[bz, by, bx * C, 0], a2_l1)
				T.gemm_v0(a2_l1, v_l1, u_l0, init = True)
				T.copy(u_l0, U[bz, by, bx * C, 0])

				T.wait_cross_flag(1)
				T.copy(workspace_a1[bz, by, bx * C, 0], a1_l1)
				T.gemm_v0(a1_l1, k_l1, w_l0, init = True)
				T.copy(w_l0, W[bz, by, bx * C, 0])

	return main

def wy_fast(k, v, beta, g, a, C):
	B, H, L, DK = k.shape
	DV = v.shape[-1]
	ker = wy_fast_ker(B, H, L, DK, DV, C)
	w, u = ker(k, v, beta, g, a)
	return w, u

def ref_wy_fast(k, v, beta, g, a, C):
	B, H, L, DK = k.shape
	DV = v.shape[-1]
	chunk_num = (L + C - 1) // C
	w = torch.zeros((B, H, L, DK)).npu().to(torch.float16)
	u = torch.zeros((B, H, L, DV)).npu().to(torch.float16)
	g = torch.exp(g)
	beta = beta.float()

	for i in range(chunk_num):
		a_c = a[:, :, i * C : (i + 1) * C, :].to(torch.float)
		k_c = k[:, :, i * C : (i + 1) * C, :]
		v_c = v[:, :, i * C : (i + 1) * C, :]
		beta_c = beta[:, :, i * C : (i + 1) * C]
		g_c = g[:, :, i * C : (i + 1) * C]
		g_c = g_c * beta_c
		a2_c = torch.einsum("bhlc,bhc->bhlc", a_c, beta_c).to(torch.float16)
		a1_c = torch.einsum("bhlc,bhc->bhlc", a_c, g_c).to(torch.float16)
		w[:, :, i * C : (i + 1) * C, :] = torch.matmul(a1_c, k_c)
		u[:, :, i * C : (i + 1) * C, :] = torch.matmul(a2_c, v_c)

	return w, u

if __name__ == "__main__":
	tilelang.cache.clear_cache()
	torch.manual_seed(0)
	torch.set_printoptions(threshold = float('inf'), sci_mode = True)

	test_configs = [
		(2, 16, 16384, 128, 128, 128),
	]

	for B, H, L, DK, DV, C in test_configs:
		print(f"Testing WY-fast with B={B}, H={H}, L={L}, DK={DK}, DV={DV}, C={C}")
		k = torch.randn((B, H, L, DK)).npu().to(torch.float16)
		v = torch.randn((B, H, L, DV)).npu().to(torch.float16)
		beta = torch.rand((B, H, L)).npu().to(torch.float16)
		g = torch.randn((B, H, L)).npu().to(torch.float)
		a = torch.randn((B, H, L, C)).npu().to(torch.float16)
		w, u = wy_fast(k, v, beta, g, a, C)
		ref_w, ref_u = ref_wy_fast(k, v, beta, g, a, C)
		torch.testing.assert_close(w.cpu(), ref_w.cpu(), rtol=1e-5, atol=1e-5)
		torch.testing.assert_close(u.cpu(), ref_u.cpu(), rtol=1e-5, atol=1e-5)
		print("Test passed!")

	print("Kernel Output Match!")
