import tilelang
from tilelang import language as T
import torch

'''
Functionality:
U = A * diag(Beta) * V
W = A * diag(exp(g) * Beta) * K
'''

pass_configs = {
	tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True
}

@tilelang.jit(out_idx=[-2, -1], workspace_idx=[-4, -3], pass_configs=pass_configs)
def wy_fast_ker(B, H, L, DK, DV, C, BK, BV, dtype="float16", accum_dtype="float"):
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
			workspace_k: T.Tensor([B, H, L, DK], dtype),
			workspace_v: T.Tensor([B, H, L, DV], dtype),
			W: T.Tensor([B, H, L, DK], dtype),
			U: T.Tensor([B, H, L, DV], dtype),
	):
		with T.Kernel(B * H * chunk_num, is_npu=True) as (cid, vid):
			bx = cid % chunk_num
			by = (cid // chunk_num) % H
			bz = (cid // chunk_num) // H

			k_ub = T.alloc_ub([C // VEC_NUM, DK], accum_dtype)
			v_ub = T.alloc_ub([C // VEC_NUM, DV], accum_dtype)
			beta_ub = T.alloc_ub([C // VEC_NUM,], accum_dtype)
			g_ub = T.alloc_ub([C // VEC_NUM,], accum_dtype)
			k_ub_half = T.alloc_ub([C // VEC_NUM, DK], dtype)
			v_ub_half = T.alloc_ub([C // VEC_NUM, DV], dtype)
			beta_ub_half = T.alloc_ub([C // VEC_NUM,], dtype)

			k_l1 = T.alloc_L1([C, BK], dtype)
			v_l1 = T.alloc_L1([C, BV], dtype)
			a_l1 = T.alloc_L1([C, C], dtype)
			w_l0 = T.alloc_L0C([C, BK], accum_dtype)
			u_l0 = T.alloc_L0C([C, BV], accum_dtype)

			with T.Scope("V"):
				T.copy(K[bz, by, bx * C + vid * C // VEC_NUM, 0], k_ub_half)
				T.copy(V[bz, by, bx * C + vid * C // VEC_NUM, 0], v_ub_half)
				T.copy(Beta[bz, by, bx * C + vid * C // VEC_NUM], beta_ub_half)
				T.copy(G[bz, by, bx * C + vid * C // VEC_NUM], g_ub)
				T.copy(k_ub_half, k_ub)
				T.copy(v_ub_half, v_ub)
				T.copy(beta_ub_half, beta_ub)

				for i in T.Parallel(C // VEC_NUM):
					g_ub[i] = T.exp(g_ub[i])
				for (i, j) in T.Parallel(C // VEC_NUM, DK):
					k_ub[i, j] = k_ub[i, j] * beta_ub[i]
				for (i, j) in T.Parallel(C // VEC_NUM, DV):
					v_ub[i, j] = v_ub[i, j] * beta_ub[i]
				for (i, j) in T.Parallel(C // VEC_NUM, DK):
					k_ub[i, j] = k_ub[i, j] * g_ub[i]
				
				T.copy(k_ub, k_ub_half)
				T.copy(v_ub, v_ub_half)
				T.copy(k_ub_half, workspace_k[bz, by, bx * C + vid * C // VEC_NUM, 0])
				T.copy(v_ub_half, workspace_v[bz, by, bx * C + vid * C // VEC_NUM, 0])
				T.set_cross_flag("MTE3", 1)

			with T.Scope("C"):
				T.copy(A[bz, by, bx * C, 0], a_l1)
				T.wait_cross_flag(1)

				for i in T.serial(bk_num):
					T.copy(workspace_k[bz, by, bx * C, i * BK], k_l1)
					T.gemm_v0(a_l1, k_l1, w_l0, init = True)
					T.copy(w_l0, W[bz, by, bx * C, i * BK])
				
				for i in T.serial(bv_num):
					T.copy(workspace_v[bz, by, bx * C, i * BV], v_l1)
					T.gemm_v0(a_l1, v_l1, u_l0, init = True)
					T.copy(u_l0, U[bz, by, bx * C, i * BV])
	
	return main

def wy_fast(k, v, beta, g, a, C, BK, BV):
	B, H, L, DK = k.shape
	DV = v.shape[-1]
	ker = wy_fast_ker(B, H, L, DK, DV, C, BK, BV)
	w, u = ker(k, v, beta, g, a)
	return w, u

def ref_wy_fast(k, v, beta, g, a, C):
	B, H, L, DK = k.shape
	DV = v.shape[-1]
	chunk_num = (L + C - 1) // C
	w = torch.zeros((B, H, L, DK)).npu().to(torch.float16)
	u = torch.zeros((B, H, L, DV)).npu().to(torch.float16)
	g = torch.exp(g)
	k = k.float()
	v = v.float()
	beta = beta.float()

	kg = torch.einsum("bhld,bhl->bhld", k, beta)
	kg = torch.einsum("bhld,bhl->bhld", kg, g)
	vg = torch.einsum("bhld,bhl->bhld", v, beta)

	for i in range(chunk_num):
		a_c = a[:, :, i * C : (i + 1) * C, :].to(torch.float16)
		k_c = kg[:, :, i * C : (i + 1) * C, :].to(torch.float16)
		v_c = vg[:, :, i * C : (i + 1) * C, :].to(torch.float16)
		w[:, :, i * C : (i + 1) * C, :] = torch.matmul(a_c, k_c)
		u[:, :, i * C : (i + 1) * C, :] = torch.matmul(a_c, v_c)
	
	return w, u


if __name__ == "__main__":
	tilelang.cache.clear_cache()
	torch.manual_seed(0)
	torch.set_printoptions(threshold = float('inf'), sci_mode = True)

	test_configs = [
		(2, 32, 256, 64, 64, 32, 32, 32),
	]

	for B, H, L, DK, DV, C, BK, BV in test_configs:
		print(f"Testing WY-fast with B={B}, H={H}, L={L}, DK={DK}, DV={DV}, C={C}, BK={BK}, BV={BV}")
		k = torch.randn((B, H, L, DK)).npu().to(torch.float16)
		v = torch.randn((B, H, L, DV)).npu().to(torch.float16)
		beta = torch.randn((B, H, L)).npu().to(torch.float16)
		g = torch.randn((B, H, L)).npu().to(torch.float)
		a = torch.randn((B, H, L, C)).npu().to(torch.float16)
		w, u = wy_fast(k, v, beta, g, a, C, BK, BV)
		ref_w, ref_u = ref_wy_fast(k, v, beta, g, a, C)
		torch.testing.assert_close(w.cpu(), ref_w.cpu(), rtol=1e-5, atol=1e-5)
		torch.testing.assert_close(u.cpu(), ref_u.cpu(), rtol=1e-5, atol=1e-5)
		print("Test passed!")
	
	print("Kernel Output Match!")
