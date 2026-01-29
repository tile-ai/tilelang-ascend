import tilelang
from tilelang import language as T
import torch
import torch.nn.functional as F

'''
Functionality:
Calculate the chunk-by-chunk hidden state
(Refer to README.md for formula. In this file, we transpose S by default)
'''

pass_configs = {
	tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
	tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}

@tilelang.jit(out_idx=[-2, -1], workspace_idx=[-7, -6, -4], pass_configs=pass_configs)
def chunk_h_ker(B, H, L, DK, DV, C, BK = None, BV = None, dtype="float16", accum_dtype="float"):
	if BK == None:
		BK = DK
	if BV == None:
		BV = DV
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
			g_v_ub = T.alloc_ub([C // VEC_NUM,], accum_dtype)
			coeff_ub = T.alloc_ub([C // VEC_NUM,], accum_dtype)
			k_ub = T.alloc_ub([C // VEC_NUM, DK], accum_dtype)
			s_ub = T.alloc_ub([DK // VEC_NUM, BV], accum_dtype)
			kv_ub = T.alloc_ub([DK // VEC_NUM, BV], accum_dtype)
			u_ub = T.alloc_ub([C // VEC_NUM, BV], accum_dtype)
			ws_ub = T.alloc_ub([C // VEC_NUM, BV], accum_dtype)
			k_ub_half = T.alloc_ub([C // VEC_NUM, DK], dtype)
			s_ub_half = T.alloc_ub([DK // VEC_NUM, BV], dtype)
			u_ub_half = T.alloc_ub([C // VEC_NUM, BV], dtype)

			with T.Scope("C"):
				for i in T.serial(chunk_num): # Calculate hidden state S chunk by chunk
					T.copy(workspace_3[cid, 0, 0], s_l1) # Previous S
					T.copy(W[bz, by, i * C, 0], w_l1)
					T.gemm_v0(w_l1, s_l1, ws_l0, init = True)
					T.copy(ws_l0, workspace_1[cid, 0, 0]) # W * S
					T.set_cross_flag("FIX", 0)

					T.wait_cross_flag(1)
					T.copy(workspace_2[cid, 0, 0], k_l1) # \tilde K
					T.copy(V[bz, by, i * C, bx * BV], v_l1) # New_V = U - W * S
					T.gemm_v0(k_l1, v_l1, kv_l0, transpose_A = True, init = True)
					T.copy(kv_l0, workspace_4[cid, 0, 0]) # \tilde K * New_V
					T.set_cross_flag("FIX", 2)

					T.wait_cross_flag(3)

			with T.Scope("V"):
				T.tile.fill(zero_ub, 0.0)
				T.tile.fill(s_ub, 0.0)
				T.copy(K[bz, by, vid * C // VEC_NUM, 0], k_ub_half) # Preload K and g for the first chunk
				T.copy(G[bz, by, 0], g_ub) # The g value of the whole chunk
				T.set_flag("mte2", "v", 0)
				T.wait_flag("mte2", "v", 0)
				for i in T.serial(chunk_num): # Calculate hidden state S chunk by chunk
					T.copy(U[bz, by, i * C + vid * C // VEC_NUM, bx * BV], u_ub_half)
					T.copy(k_ub_half, k_ub)
					T.copy(g_ub[vid * C // VEC_NUM : (vid + 1) * C // VEC_NUM], g_v_ub) # The g value of current vector core
					tmp = g_ub[C - 1]
					for i in T.Parallel(C // VEC_NUM):
						coeff_ub[i] = g_v_ub[i] - tmp
					for i in T.Parallel(C // VEC_NUM):
						coeff_ub[i] = zero_ub[i] - coeff_ub[i]
					for i in T.Parallel(C // VEC_NUM):
						coeff_ub[i] = T.exp(coeff_ub[i])
					# coeff_ub now stores exp(g_last - g_i)
					
					for i in T.Parallel(C):
						g_ub[i] = T.exp(g_ub[i])
					T.set_flag("mte2", "v", 0)
					T.wait_flag("mte2", "v", 0)
					T.copy(u_ub_half, u_ub)

					#\tilde K = K * exp(g_last - g_i)
					for i in range((C // VEC_NUM) // 4):
						tmp0 = coeff_ub[i * 4]
						tmp1 = coeff_ub[i * 4 + 1]
						tmp2 = coeff_ub[i * 4 + 2]
						tmp3 = coeff_ub[i * 4 + 3]
						T.tile.mul(k_ub[i * 4, :], k_ub[i * 4, :], tmp0)
						T.tile.mul(k_ub[i * 4 + 1, :], k_ub[i * 4 + 1, :], tmp1)
						T.tile.mul(k_ub[i * 4 + 2, :], k_ub[i * 4 + 2, :], tmp2)
						T.tile.mul(k_ub[i * 4 + 3, :], k_ub[i * 4 + 3, :], tmp3)

					T.wait_cross_flag(0)
					T.copy(workspace_1[cid, vid * C // VEC_NUM, 0], u_ub_half)
					T.set_flag("mte2", "v", 0)
					T.wait_flag("mte2", "v", 0)
					T.copy(u_ub_half, ws_ub)
					for (i, j) in T.Parallel(C // VEC_NUM, BV):
						u_ub[i, j] = u_ub[i, j] - ws_ub[i, j] # New_V = U - W * S
					T.copy(u_ub, u_ub_half)
					T.copy(k_ub, k_ub_half)
					T.set_flag("v", "mte3", 0)
					T.wait_flag("v", "mte3", 0)
					T.copy(u_ub_half, V[bz, by, i * C + vid * C // VEC_NUM, bx * BV])
					T.copy(k_ub_half, workspace_2[cid, vid * C // VEC_NUM, 0])
					T.set_cross_flag("MTE3", 1)

					tmp = g_ub[C - 1]
					T.tile.mul(s_ub, s_ub, tmp)
					# s_ub now stores S * exp(g_last)

					T.set_flag("v", "mte2", 0)
					T.wait_flag("v", "mte2", 0)
					if i < chunk_num - 1:
						T.copy(K[bz, by, (i + 1) * C + vid * C // VEC_NUM, 0], k_ub_half) # Preload K and g for the next chunk
						T.copy(G[bz, by, (i + 1) * C], g_ub) # The g value of the whole chunk
					
					T.wait_cross_flag(2)
					T.copy(workspace_4[cid, vid * DK // VEC_NUM, 0], s_ub_half)
					T.set_flag("mte2", "v", 0)
					T.wait_flag("mte2", "v", 0)
					T.copy(s_ub_half, kv_ub)
					T.barrier_all()
					for (i, j) in T.Parallel(DK // VEC_NUM, BV):
						s_ub[i, j] = s_ub[i, j] + kv_ub[i, j] # S_next = S * exp(g_last) + \tilde K * New_V
					T.copy(s_ub, s_ub_half)
					if i < chunk_num - 1:
						T.set_flag("v", "mte3", 0)
						T.wait_flag("v", "mte3", 0)
						T.copy(s_ub_half, workspace_3[cid, vid * DK // VEC_NUM, 0])
						T.copy(s_ub_half, S[bz, by, i + 1, vid * DK // VEC_NUM, bx * BV]) # Store state S at the end of this chunk
					T.set_cross_flag("MTE3", 3)
				
				T.set_flag("v", "mte3", 0)
				T.wait_flag("v", "mte3", 0)
				T.copy(s_ub_half, FS[bz, by, vid * DK // VEC_NUM, bx * BV]) # Final state, will not be used to calculate output, just for verification

	return main

def chunk_h(k, w, u, g, C):
	B, H, L, DK = k.shape
	DV = u.shape[-1]
	BV = DV
	bv_num = (DV + BV - 1) // BV
	workspace_3 = torch.zeros((B * H * bv_num, DK, BV)).npu().to(torch.float16)
	s = torch.zeros((B, H, (L + C - 1) // C, DK, DV)).npu().to(torch.float16)
	ker = chunk_h_ker(B, H, L, DK, DV, C)
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
		(2, 16, 16384, 128, 128, 128),
	]

	for B, H, L, DK, DV, C in test_configs:
		print(f"Testing Hidden State with B={B}, H={H}, L={L}, DK={DK}, DV={DV}, C={C}")
		k = torch.randn((B, H, L, DK)).npu().to(torch.float16)
		w = torch.randn((B, H, L, DK)).npu().to(torch.float16)
		u = torch.randn((B, H, L, DV)).npu().to(torch.float16)
		g = torch.randn((B, H, L)).npu().to(torch.float)
		g = F.logsigmoid(g)
		k, w = F.normalize(k, dim=-1, p=2), F.normalize(w, dim=-1, p=2)
		g = ref_chunk_cumsum(g, C)
		s, new_v, final_s = chunk_h(k, w, u, g, C)
		ref_s, ref_new_v, ref_final_s = ref_chunk_h(k, w, u, g, C)
		torch.testing.assert_close(s.cpu(), ref_s.cpu(), rtol=1e-5, atol=1e-5)
		torch.testing.assert_close(new_v.cpu(), ref_new_v.cpu(), rtol=1e-5, atol=1e-5)
		torch.testing.assert_close(final_s.cpu(), ref_final_s.cpu(), rtol=1e-5, atol=1e-5)
		print("Test passed!")

	print("Kernel Output Match!")
