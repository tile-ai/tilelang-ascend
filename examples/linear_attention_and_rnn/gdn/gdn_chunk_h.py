import tilelang
from tilelang import language as T
import torch
import torch.nn.functional as F

'''
Functionality:
Calculate the chunk-by-chunk hidden state
'''

pass_configs = {
	tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True
}

@tilelang.jit(out_idx=[-2, -1], workspace_idx=[-7, -6, -4], pass_configs=pass_configs)
def chunk_h_ker(B, H, L, DK, DV, C, BK, BV, dtype="float16", accum_dtype="float"):
	chunk_num = T.ceildiv(L, C)
	bv_num = T.ceildiv(DV, BV)
	VEC_NUM = 2

	@T.prim_func
	def main(
			K: T.Tensor([B, H, L, DK], dtype),
			W: T.Tensor([B, H, L, DK], dtype),
			U: T.Tensor([B, H, L, DV], dtype),
			G: T.Tensor([B, H, L], accum_dtype),
			workspace_1: T.Tensor([B * H * bv_num, C, BV], dtype),
			workspace_2: T.Tensor([B * H * bv_num, C, DK], dtype),
			workspace_3: T.Tensor([B * H * bv_num, DK, BV], dtype), # need to be manually set to 0
			workspace_4: T.Tensor([B * H * bv_num, DK, BV], dtype),
			S: T.Tensor([B, H, chunk_num, DK, DV], dtype), # need to be manually set to 0
			V: T.Tensor([B, H, L, DV], dtype),
			FS: T.Tensor([B, H, DK, DV], dtype),
	):
		with T.Kernel(B * H * bv_num, is_npu=True) as (cid, vid):
			bx = cid % bv_num
			by = (cid // bv_num) % H
			bz = (cid // bv_num) // H

			s_l1 = T.alloc_L1([DK, BV], dtype)
			w_l1 = T.alloc_L1([C, DK], dtype)
			k_l1 = T.alloc_L1([C, DK], dtype)
			v_l1 = T.alloc_L1([C, BV], dtype)
			ws_l0 = T.alloc_L0C([C, BV], accum_dtype)
			kv_l0 = T.alloc_L0C([DK, BV], accum_dtype)
			
			zero_ub = T.alloc_ub([C // VEC_NUM,], accum_dtype)
			g_ub = T.alloc_ub([C,], accum_dtype)
			gv_ub = T.alloc_ub([C // VEC_NUM,], accum_dtype)
			coeff_ub = T.alloc_ub([C // VEC_NUM,], accum_dtype)
			k_ub = T.alloc_ub([C // VEC_NUM, DK], accum_dtype)
			s_ub = T.alloc_ub([DK // VEC_NUM, BV], accum_dtype)
			kv_ub = T.alloc_ub([DK // VEC_NUM, BV], accum_dtype)
			u_ub = T.alloc_ub([C // VEC_NUM, BV], accum_dtype)
			ws_ub = T.alloc_ub([C // VEC_NUM, BV], accum_dtype)
			k_ub_half = T.alloc_ub([C // VEC_NUM, DK], dtype)
			s_ub_half = T.alloc_ub([DK // VEC_NUM, BV], dtype)
			u_ub_half = T.alloc_ub([C // VEC_NUM, BV], dtype)
			kv_ub_half = T.alloc_ub([DK // VEC_NUM, BV], dtype)
			ws_ub_half = T.alloc_ub([C // VEC_NUM, BV], dtype)

			with T.Scope("C"):
				for i in T.serial(chunk_num):
					T.copy(workspace_3[cid, 0, 0], s_l1)
					T.copy(W[bz, by, i * C, 0], w_l1)
					T.gemm_v0(w_l1, s_l1, ws_l0, init = True)
					T.copy(ws_l0, workspace_1[cid, 0, 0])
					T.set_cross_flag("FIX", 0)
					
					T.wait_cross_flag(1)
					T.copy(workspace_2[cid, 0, 0], k_l1)
					T.copy(V[bz, by, i * C, bx * BV], v_l1)
					T.gemm_v0(k_l1, v_l1, kv_l0, transpose_A = True, init = True)
					T.copy(kv_l0, workspace_4[cid, 0, 0])
					T.set_cross_flag("FIX", 2)

					T.wait_cross_flag(3)

			with T.Scope("V"):
				T.tile.fill(zero_ub, 0.0)
				T.tile.fill(s_ub, 0.0)
				for i in T.serial(chunk_num):
					T.copy(U[bz, by, i * C + vid * C // VEC_NUM, bx * BV], u_ub_half)
					T.copy(u_ub_half, u_ub)
					T.copy(K[bz, by, i * C + vid * C // VEC_NUM, 0], k_ub_half)
					T.copy(k_ub_half, k_ub)
					T.copy(G[bz, by, i * C], g_ub)
					T.copy(G[bz, by, i * C + vid * C // VEC_NUM], gv_ub)
					for i in T.Parallel(C // VEC_NUM):
						coeff_ub[i] = gv_ub[i] - g_ub[C - 1]
					for i in T.Parallel(C // VEC_NUM):
						coeff_ub[i] = zero_ub[i] - coeff_ub[i]
					for i in T.Parallel(C // VEC_NUM):
						coeff_ub[i] = T.exp(coeff_ub[i])
					for i in T.Parallel(C):
						g_ub[i] = T.exp(g_ub[i])
					for (i, j) in T.Parallel(C // VEC_NUM, DK):
						k_ub[i, j] = k_ub[i, j] * coeff_ub[i]
					
					T.wait_cross_flag(0)
					T.copy(workspace_1[cid, vid * C // VEC_NUM, 0], ws_ub_half)
					T.copy(ws_ub_half, ws_ub)
					for (i, j) in T.Parallel(C // VEC_NUM, BV):
						u_ub[i, j] = u_ub[i, j] - ws_ub[i, j]
					T.copy(u_ub, u_ub_half)
					T.copy(u_ub_half, V[bz, by, i * C + vid * C // VEC_NUM, bx * BV])
					T.copy(k_ub, k_ub_half)
					T.copy(k_ub_half, workspace_2[cid, vid * C // VEC_NUM, 0])
					T.set_cross_flag("MTE3", 1)

					for (i, j) in T.Parallel(DK // VEC_NUM, BV):
						s_ub[i, j] = s_ub[i, j] * g_ub[C - 1]
					
					T.wait_cross_flag(2)
					T.copy(workspace_4[cid, vid * DK // VEC_NUM, 0], kv_ub_half)
					T.copy(kv_ub_half, kv_ub)
					for (i, j) in T.Parallel(DK // VEC_NUM, BV):
						s_ub[i, j] = s_ub[i, j] + kv_ub[i, j]
					T.copy(s_ub, s_ub_half)
					if i < chunk_num - 1:
						T.copy(s_ub_half, workspace_3[cid, vid * DK // VEC_NUM, 0])
						T.copy(s_ub_half, S[bz, by, i + 1, vid * DK // VEC_NUM, bx * BV])
					T.set_cross_flag("MTE3", 3)
				
				T.copy(s_ub_half, FS[bz, by, vid * DK // VEC_NUM, bx * BV])
	
	return main

def chunk_h(k, w, u, g, C, BK, BV):
	B, H, L, DK = k.shape
	DV = u.shape[-1]
	bv_num = (DV + BV - 1) // BV
	workspace_3 = torch.zeros((B * H * bv_num, DK ,BV)).npu().to(torch.float16)
	s = torch.zeros((B, H, (L + C - 1) // C, DK, DV)).npu().to(torch.float16)
	ker = chunk_h_ker(B, H, L, DK, DV, C, BK, BV)
	new_v, final_s = ker(k, w, u, g, workspace_3, s)
	return s, new_v, final_s

def ref_chunk_h(k, w, u, g, C):
	B, H, L, DK = k.shape
	DV = u.shape[-1]
	chunk_num = (L + C - 1) // C
	s = torch.zeros((B, H, chunk_num, DK, DV)).npu().to(torch.float)
	new_v = torch.zeros((B, H, L, DV)).npu().to(torch.float)
	k = k.float()
	u = u.float()

	for i in range(chunk_num):
		las_s = s[:, :, i, :, :]
		k_c = k[:, :, i * C : (i + 1) * C, :]
		w_c = w[:, :, i * C : (i + 1) * C, :]
		u_c = u[:, :, i * C : (i + 1) * C, :]
		g_c = g[:, :, i * C : (i + 1) * C]
		ws = torch.matmul(w_c, las_s.to(torch.float16)).float()
		new_v_c = u_c - ws
		new_v[:, :, i * C : (i + 1) * C, :] = new_v_c
		g_last = g[:, :, (i + 1) * C - 1].view(B, H, 1, 1)
		coeff_k = g_last - g_c.view(B, H, C, 1)
		g_last = torch.exp(g_last)
		coeff_k = torch.exp(coeff_k)
		k_c = (k_c * coeff_k).transpose(-2, -1)
		las_s = las_s * g_last
		kv = torch.matmul(k_c.to(torch.float16), new_v_c.to(torch.float16)).float()
		s_c = las_s + kv
		if i < chunk_num - 1:
			s[:, :, i + 1, :, :] = s_c
	
	return s.to(torch.float16), new_v.to(torch.float16), s_c.to(torch.float16)

def ref_chunk_cumsum(g, C):
	B, H, L = g.shape
	chunk_num = (L + C - 1) // C
	g = g.view(B, H, chunk_num, C)
	g_sum = torch.cumsum(g, dim = -1)
	g_sum = g_sum.view(B, H, L)
	return g_sum


if __name__ == "__main__":
	tilelang.cache.clear_cache()
	torch.manual_seed(0)
	torch.set_printoptions(threshold = float('inf'), sci_mode = True)

	test_configs = [
		(2, 32, 256, 64, 64, 32, 32, 32),
	]

	for B, H, L, DK, DV, C, BK, BV in test_configs:
		print(f"Testing Hidden State with B={B}, H={H}, L={L}, DK={DK}, DV={DV}, C={C}, BK={BK}, BV={BV}")
		k = torch.randn((B, H, L, DK)).npu().to(torch.float16)
		w = torch.randn((B, H, L, DK)).npu().to(torch.float16)
		u = torch.randn((B, H, L, DV)).npu().to(torch.float16)
		g = torch.randn((B, H, L)).npu().to(torch.float)
		g = F.logsigmoid(g)
		k, w = F.normalize(k, dim=-1, p=2), F.normalize(w, dim=-1, p=2)
		g = ref_chunk_cumsum(g, C)
		s, new_v, final_s = chunk_h(k, w, u, g, C, BK, BV)
		ref_s, ref_new_v, ref_final_s = ref_chunk_h(k, w, u, g, C)
		torch.testing.assert_close(s.cpu(), ref_s.cpu(), rtol=1e-5, atol=1e-5)
		torch.testing.assert_close(new_v.cpu(), ref_new_v.cpu(), rtol=1e-5, atol=1e-5)
		torch.testing.assert_close(final_s.cpu(), ref_final_s.cpu(), rtol=1e-5, atol=1e-5)
		print("Test passed!")
	
	print("Kernel Output Match!")
