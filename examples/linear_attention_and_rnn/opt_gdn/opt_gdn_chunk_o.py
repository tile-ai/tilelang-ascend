import tilelang
from tilelang import language as T
import torch
import torch.nn.functional as F

'''
Functionality:
Calculate output, given chunk-by-chunk hidden state
(Refer to README.md for formula. In this file, we transpose S by default)
'''

pass_configs = {
	tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
	tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}

@tilelang.jit(out_idx=[-1], workspace_idx=[-4, -3, -2], pass_configs=pass_configs)
def chunk_o_ker(B, H, L, DK, DV, C, BK = None, BV = None, dtype="float16", accum_dtype="float"):
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
			Q: T.Tensor([B, H, L, DK], dtype),
			K: T.Tensor([B, H, L, DK], dtype),
			V: T.Tensor([B, H, L, DV], dtype),
			S: T.Tensor([B, H, chunk_num, DK, DV], dtype),
			G: T.Tensor([B, H, L], accum_dtype),
			Msk: T.Tensor([C, C], accum_dtype),
			workspace_1: T.Tensor([B * H * chunk_num, C, C], dtype),
			workspace_2: T.Tensor([B * H * chunk_num, C, DV], dtype),
			workspace_3: T.Tensor([B * H * chunk_num, C, C], dtype),
			O: T.Tensor([B, H, L, DV], dtype),
	):
		with T.Kernel(B * H * chunk_num, is_npu=True) as (cid, vid):
			bx = cid % chunk_num
			by = (cid // chunk_num) % H
			bz = (cid // chunk_num) // H

			q_l1 = T.alloc_L1([C, BK], dtype)
			k_l1 = T.alloc_L1([C, BK], dtype)
			v_l1 = T.alloc_L1([C, BV], dtype)
			s_l1 = T.alloc_L1([BK, DV], dtype)
			qk_l1 = T.alloc_L1([C, C], dtype)
			qk_l0 = T.alloc_L0C([C, C], accum_dtype)
			qs_l0 = T.alloc_L0C([C, DV], accum_dtype)
			qkv_l0 = T.alloc_L0C([C, BV], accum_dtype)

			qk_ub_half = T.alloc_ub([C // VEC_NUM, C], dtype)
			qs_ub_half = T.alloc_ub([C // VEC_NUM, DV], dtype)
			o_ub_half = T.alloc_ub([C // VEC_NUM, DV], dtype)
			qk_ub = T.alloc_ub([C // VEC_NUM, C], accum_dtype)
			msk_ub = T.alloc_ub([C // VEC_NUM, C], accum_dtype)
			qs_ub = T.alloc_ub([C // VEC_NUM, DV], accum_dtype)
			o_ub = T.alloc_ub([C // VEC_NUM, DV], accum_dtype)
			coeff_ub = T.alloc_ub([C // VEC_NUM, C], accum_dtype)
			g_ub = T.alloc_ub([C,], accum_dtype)
			g_v_ub = T.alloc_ub([C // VEC_NUM,], accum_dtype)

			with T.Scope("C"):
				for i in T.serial(bk_num):
					T.copy(Q[bz, by, bx * C, i * BK], q_l1)
					T.copy(K[bz, by, bx * C, i * BK], k_l1)
					T.gemm_v0(q_l1, k_l1, qk_l0, transpose_B = True, init = (i == 0)) # Q * K^T
				for i in T.serial(bk_num):
					T.copy(Q[bz, by, bx * C, i * BK], q_l1)
					T.copy(S[bz, by, bx, i * BK, 0], s_l1)
					T.gemm_v0(q_l1, s_l1, qs_l0, init = (i == 0)) # Q * S
				T.copy(qk_l0, workspace_1[cid, 0, 0])
				T.copy(qs_l0, workspace_2[cid, 0, 0])
				T.set_cross_flag("FIX", 0)

				T.wait_cross_flag(1)
				T.copy(workspace_3[cid, 0, 0], qk_l1) # Gamma \odot Mask \odot (Q * K^T)
				for i in T.serial(bv_num):
					T.copy(V[bz, by, bx * C, i * BV], v_l1)
					T.gemm_v0(qk_l1, v_l1, qkv_l0, init = True)
					T.copy(qkv_l0, workspace_2[cid, 0, i * BV]) # Term 2 of the formula (intra-chunk)
				T.set_cross_flag("FIX", 2)
			
			with T.Scope("V"):
				T.copy(G[bz, by, bx * C], g_ub) # The g value of the whole chunk
				T.copy(Msk[vid * C // VEC_NUM, 0], msk_ub)
				T.set_flag("mte2", "v", 0)
				T.wait_flag("mte2", "v", 0)
				T.tile.fill(qk_ub, 0.0) # reuse qk_ub as zero buffer temporarily
				T.copy(g_ub[vid * C // VEC_NUM : (vid + 1) * C // VEC_NUM], g_v_ub) # The g value of current vector core
				for i in range((C // VEC_NUM) // 4):
					tmp0 = g_v_ub[i * 4]
					tmp1 = g_v_ub[i * 4 + 1]
					tmp2 = g_v_ub[i * 4 + 2]
					tmp3 = g_v_ub[i * 4 + 3]
					T.tile.sub(coeff_ub[i * 4, :], g_ub, tmp0)
					T.tile.sub(coeff_ub[i * 4 + 1, :], g_ub, tmp1)
					T.tile.sub(coeff_ub[i * 4 + 2, :], g_ub, tmp2)
					T.tile.sub(coeff_ub[i * 4 + 3, :], g_ub, tmp3)
				T.tile.sub(coeff_ub, qk_ub, coeff_ub)
				T.tile.mul(coeff_ub, coeff_ub, msk_ub) # This doesn't effect the result theoretically (because we apply the causal mask again later), but avoids overflow in exp in the next line
				T.tile.exp(coeff_ub, coeff_ub)
				# coeff_ub_{i, j} now stores exp((g_i - g_j) * Mask_{i, j})

				T.tile.exp(g_v_ub, g_v_ub)

				T.wait_cross_flag(0)
				T.copy(workspace_1[cid, vid * C // VEC_NUM, 0], qk_ub_half)
				T.set_flag("mte2", "v", 0)
				T.wait_flag("mte2", "v", 0)
				T.copy(qk_ub_half, qk_ub)
				T.set_flag("v", "mte2", 0)
				T.wait_flag("v", "mte2", 0)
				T.copy(workspace_2[cid, vid * C // VEC_NUM, 0], qs_ub_half)
				T.tile.mul(qk_ub, qk_ub, coeff_ub) # Apply the coeff
				T.tile.mul(qk_ub, qk_ub, msk_ub) # Apply the causal mask
				T.copy(qk_ub, qk_ub_half)
				T.set_flag("v", "mte3", 0)
				T.wait_flag("v", "mte3", 0)
				T.copy(qk_ub_half, workspace_3[cid, vid * C // VEC_NUM, 0]) # Gamma \odot Mask \odot (Q * K^T)
				T.set_cross_flag("MTE3", 1)

				T.set_flag("mte2", "v", 0)
				T.wait_flag("mte2", "v", 0)
				T.copy(qs_ub_half, qs_ub) # Q * S
				for i in range((C // VEC_NUM) // 4):
					tmp0 = g_v_ub[i * 4]
					tmp1 = g_v_ub[i * 4 + 1]
					tmp2 = g_v_ub[i * 4 + 2]
					tmp3 = g_v_ub[i * 4 + 3]
					T.tile.mul(qs_ub[i * 4, :], qs_ub[i * 4, :], tmp0)
					T.tile.mul(qs_ub[i * 4 + 1, :], qs_ub[i * 4 + 1, :], tmp1)
					T.tile.mul(qs_ub[i * 4 + 2, :], qs_ub[i * 4 + 2, :], tmp2)
					T.tile.mul(qs_ub[i * 4 + 3, :], qs_ub[i * 4 + 3, :], tmp3)
				# qs_ub now stores diag(exp(g)) * Q * S, i.e. Term 1 of the formula (inter-chunk)
				
				T.wait_cross_flag(2)
				T.copy(workspace_2[cid, vid * C // VEC_NUM, 0], o_ub_half)
				T.set_flag("mte2", "v", 0)
				T.wait_flag("mte2", "v", 0)
				T.copy(o_ub_half, o_ub)
				for (i, j) in T.Parallel(C // VEC_NUM, DV):
					o_ub[i, j] = qs_ub[i, j] + o_ub[i, j] # O = Term 1 + Term 2
				T.copy(o_ub, o_ub_half)
				T.set_flag("v", "mte3", 0)
				T.wait_flag("v", "mte3", 0)
				T.copy(o_ub_half, O[bz, by, bx * C + vid * C // VEC_NUM, 0])

	return main

def chunk_o(q, k, v, s, g, C):
	B, H, L, DK = k.shape
	DV = v.shape[-1]
	msk = torch.tril(torch.ones((C, C)), diagonal = 0).npu().to(torch.float)
	ker = chunk_o_ker(B, H, L, DK, DV, C)
	o = ker(q, k, v, s, g, msk)
	return o

def ref_chunk_o(q, k, v, s, g, C):
	B, H, L, DK = k.shape
	DV = v.shape[-1]
	chunk_num = (L + C - 1) // C
	o = torch.zeros((B, H, L, DV)).npu().to(torch.float)
	M = torch.tril(torch.ones((C, C))).npu().to(torch.float)

	for i in range(chunk_num):
		q_c = q[:, :, i * C : (i + 1) * C, :]
		k_c = k[:, :, i * C : (i + 1) * C, :].transpose(-2, -1)
		v_c = v[:, :, i * C : (i + 1) * C, :]
		s_c = s[:, :, i, :, :]
		g_c = g[:, :, i * C : (i + 1) * C]
		gamma = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)
		g_c = torch.exp(g_c)
		gamma = torch.exp(gamma)
		term1 = torch.matmul(q_c, s_c).float()
		term1 = g_c.unsqueeze(-1) * term1
		qkt = torch.matmul(q_c, k_c).float()
		qkt = (qkt * gamma * M.view(1, 1, C, C)).to(torch.float16)
		term2 = torch.matmul(qkt, v_c).float()
		o_t = term1 + term2
		o[:, :, i * C : (i + 1) * C, :] = o_t

	return o.to(torch.float16)


if __name__ == "__main__":
	tilelang.cache.clear_cache()
	torch.manual_seed(0)
	torch.set_printoptions(threshold = float('inf'), sci_mode = True)

	test_configs = [
		(2, 16, 16384, 128, 128, 128),
	]

	for B, H, L, DK, DV, C in test_configs:
		print(f"Testing Output with B={B}, H={H}, L={L}, DK={DK}, DV={DV}, C={C}")
		q = torch.randn((B, H, L, DK)).npu().to(torch.float16)
		k = torch.randn((B, H, L, DK)).npu().to(torch.float16)
		v = torch.randn((B, H, L, DV)).npu().to(torch.float16)
		s = torch.randn((B, H, (L + C - 1) // C, DK, DV)).npu().to(torch.float16)
		g = torch.randn((B, H, L)).npu().to(torch.float)
		q, k = F.normalize(q, dim=-1, p=2), F.normalize(k, dim=-1, p=2)
		o = chunk_o(q, k, v, s, g, C)
		ref_o = ref_chunk_o(q, k, v, s, g, C)
		torch.testing.assert_close(o.cpu(), ref_o.cpu(), rtol=1e-5, atol=1e-5)
		print("Test passed!")

	print("Kernel Output Match!")
