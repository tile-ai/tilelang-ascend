import tilelang
from tilelang import DataType, language as T
import torch
import torch.nn.functional as F

'''
Functionality:
O = (I + A)^{-1}
'''

pass_configs = {
	tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def solve_tril_ker(B, H, L, C, dtype="float16", accum_dtype="float"):
	chunk_num = T.ceildiv(L, C)
	VEC_NUM = 2

	@T.prim_func
	def main(
			A: T.Tensor([B, H, L, C], dtype),
			I: T.Tensor([C, C], accum_dtype),
			O: T.Tensor([B, H, L, C], dtype),
	):
		with T.Kernel(B * (H // VEC_NUM) * chunk_num, is_npu=True) as (cid, vid):
			bx = cid % chunk_num
			by = (cid // chunk_num) % (H // VEC_NUM) * 2 + vid
			bz = (cid // chunk_num) // (H // VEC_NUM)

			o_ub = T.alloc_ub([C, C], accum_dtype)
			o_row_ub = T.alloc_ub([C, 1], accum_dtype)
			o_row_2d_ub = T.alloc_ub([C, C], accum_dtype)
			i_ub = T.alloc_ub([C, C], accum_dtype)
			mul_ub = T.alloc_ub([C, C], accum_dtype)
			red_ub = T.alloc_ub([C,], accum_dtype)
			o_ub_half = T.alloc_ub([C, C], dtype)
			tmp_ub = T.alloc_ub([3 * C * C], "uint8")

			with T.Scope("V"):
				T.copy(A[bz, by, bx * C, 0], o_ub_half)
				T.copy(I[0, 0], i_ub)
				T.set_flag("mte2", "v", 0)
				T.wait_flag("mte2", "v", 0)
				T.tile.fill(mul_ub, 0.0)
				T.copy(o_ub_half, o_ub)
				for i in range(2, C):
					T.copy(o_ub[i, :], o_row_ub)
					T.tile.broadcast(o_row_2d_ub, o_row_ub, tmp_ub)
					T.tile.fill(red_ub, 0.0)
					T.tile.mul(mul_ub, o_ub, o_row_2d_ub)
					T.tile.reduce_sum(red_ub, mul_ub, tmp_ub, dim = 0)
					T.tile.sub(o_ub[i, :], o_ub[i, :], red_ub)
				T.tile.sub(o_ub, i_ub, o_ub)
				T.copy(o_ub, o_ub_half)
				T.set_flag("v", "mte3", 0)
				T.wait_flag("v", "mte3", 0)
				T.copy(o_ub_half, O[bz, by, bx * C, 0])

	return main

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def solve_tril_64_ker(B, H, L, dtype="float16", accum_dtype="float"):
	C = 64
	F = 32
	chunk_num = T.ceildiv(L, C)
	VEC_NUM = 2

	@T.prim_func
	def main(
			A: T.Tensor([B, H, L, C], dtype),
			I: T.Tensor([C, C], accum_dtype),
			O: T.Tensor([B, H, L, C], dtype),
	):
		with T.Kernel(B * H * chunk_num, is_npu=True) as (cid, vid):
			bx = cid % chunk_num
			by = (cid // chunk_num) % H
			bz = (cid // chunk_num) // H

			o_ub = T.alloc_ub([F, F], accum_dtype)
			o_row_ub = T.alloc_ub([F, 1], accum_dtype)
			o_row_2d_ub = T.alloc_ub([F, F], accum_dtype)
			i_ub = T.alloc_ub([F, F], accum_dtype)
			mul_ub = T.alloc_ub([F, F], accum_dtype)
			red_ub = T.alloc_ub([F,], accum_dtype)
			o_ub_half = T.alloc_ub([F, F], dtype)
			zero_ub_half = T.alloc_ub([F, F], dtype)
			tmp_ub = T.alloc_ub([3 * F * F], "uint8")

			o11_l1 = T.alloc_L1([F, F], dtype)
			o22_l1 = T.alloc_L1([F, F], dtype)
			a21_l1 = T.alloc_L1([F, F], dtype)
			mult_l1 = T.alloc_L1([F, F], dtype)
			mult_l0 = T.alloc_L0C([F, F], accum_dtype)
			final_l0 = T.alloc_L0C([F, F], accum_dtype)

			with T.Scope("V"):
				T.copy(A[bz, by, bx * C + vid * F, vid * F], o_ub_half)
				T.copy(I[0, 0], i_ub)
				T.set_flag("mte2", "v", 0)
				T.wait_flag("mte2", "v", 0)
				T.tile.fill(mul_ub, 0.0)
				T.copy(o_ub_half, o_ub)
				for i in range(2, F):
					T.copy(o_ub[i, :], o_row_ub)
					T.tile.broadcast(o_row_2d_ub, o_row_ub, tmp_ub)
					T.tile.fill(red_ub, 0.0)
					T.tile.mul(mul_ub, o_ub, o_row_2d_ub)
					T.tile.reduce_sum(red_ub, mul_ub, tmp_ub, dim = 0)
					T.tile.sub(o_ub[i, :], o_ub[i, :], red_ub)
				T.tile.sub(o_ub, i_ub, o_ub)
				T.copy(o_ub, o_ub_half)
				T.set_flag("v", "mte3", 0)
				T.wait_flag("v", "mte3", 0)
				T.copy(o_ub_half, O[bz, by, bx * C + vid * F, vid * F])
				T.set_cross_flag("MTE3", 0)
				T.tile.fill(zero_ub_half, 0.0)
				if vid == 0:
					T.copy(zero_ub_half, O[bz, by, bx * C, F])
				else:
					T.wait_cross_flag(1)
					T.copy(O[bz, by, bx * C + F, 0], o_ub_half)
					T.set_flag("mte2", "v", 0)
					T.wait_flag("mte2", "v", 0)
					T.tile.sub(o_ub_half, zero_ub_half, o_ub_half)
					T.set_flag("v", "mte3", 0)
					T.wait_flag("v", "mte3", 0)
					T.copy(o_ub_half, O[bz, by, bx * C + F, 0])

			with T.Scope("C"):
				T.copy(A[bz, by, bx * C + F, 0], a21_l1)
				T.wait_cross_flag(0)
				T.copy(O[bz, by, bx * C, 0], o11_l1)
				T.copy(O[bz, by, bx * C + F, F], o22_l1)
				T.gemm_v0(o22_l1, a21_l1, mult_l0, init = True)
				T.copy(mult_l0, O[bz, by, bx * C + F, 0])
				T.set_flag("fix", "mte2", 0)
				T.wait_flag("fix", "mte2", 0)
				T.copy(O[bz, by, bx * C + F, 0], mult_l1)
				T.gemm_v0(mult_l1, o11_l1, final_l0, init = True)
				T.copy(final_l0, O[bz, by, bx * C + F, 0])
				T.set_cross_flag("FIX", 1)

	return main

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def solve_tril_128_ker(B, H, L, dtype="float16", accum_dtype="float"):
	C = 128
	F = 32
	chunk_num = T.ceildiv(L, C)
	VEC_NUM = 2

	@T.prim_func
	def main(
			A: T.Tensor([B, H, L, C], dtype),
			I: T.Tensor([C, C], accum_dtype),
			O: T.Tensor([B, H, L, C], dtype),
	):
		with T.Kernel(B * H * chunk_num, is_npu=True) as (cid, vid):
			bx = cid % chunk_num
			by = (cid // chunk_num) % H
			bz = (cid // chunk_num) // H

			o_ub = T.alloc_ub([F, F], accum_dtype)
			o_row_ub = T.alloc_ub([F, 1], accum_dtype)
			o_row_2d_ub = T.alloc_ub([F, F], accum_dtype)
			i_ub = T.alloc_ub([F, F], accum_dtype)
			mul_ub = T.alloc_ub([F, F], accum_dtype)
			red_ub = T.alloc_ub([F,], accum_dtype)
			o_ub_half = T.alloc_ub([F, F], dtype)
			a_ub_half = T.alloc_ub([2, F, F], dtype)
			zero_ub_half = T.alloc_ub([F, F], dtype)
			fat_o_ub_half = T.alloc_ub([F, 2 * F], dtype)
			fat_zero_ub_half = T.alloc_ub([F, 2 * F], dtype)
			tmp_ub = T.alloc_ub([3 * DataType(accum_dtype).bits // 8 * F * F // VEC_NUM], "uint8")

			o11_s_l1 = T.alloc_L1([F, F], dtype)
			o22_s_l1 = T.alloc_L1([F, F], dtype)
			a21_s_l1 = T.alloc_L1([F, F], dtype)
			mult_s_l1 = T.alloc_L1([F, F], dtype)
			mult_s_l0 = T.alloc_L0C([F, F], accum_dtype)
			final_s_l0 = T.alloc_L0C([F, F], accum_dtype)
			o11_l_l1 = T.alloc_L1([2 * F, 2 * F], dtype)
			o22_l_l1 = T.alloc_L1([2 * F, 2 * F], dtype)
			a21_l_l1 = T.alloc_L1([2 * F, 2 * F], dtype)
			mult_l_l1 = T.alloc_L1([2 * F, 2 * F], dtype)
			mult_l_l0 = T.alloc_L0C([2 * F, 2 * F], accum_dtype)
			final_l_l0 = T.alloc_L0C([2 * F, 2 * F], accum_dtype)

			with T.Scope("V"):
				T.copy(A[bz, by, bx * C + (vid * 2 + 1) * F, (vid * 2) * F], o_ub_half)
				T.copy(A[bz, by, bx * C + (vid + 2) * F, 0], fat_o_ub_half)
				T.tile.fill(zero_ub_half, 0.0)
				T.tile.fill(fat_zero_ub_half, 0.0)
				T.set_flag("v", "mte3", 0)
				T.wait_flag("v", "mte3", 0)
				T.copy(zero_ub_half, O[bz, by, bx * C + (vid * 2) * F, (vid * 2 + 1) * F])
				T.copy(fat_zero_ub_half, O[bz, by, bx * C + vid * F, 2 * F])
				T.set_flag("mte2", "v", 0)
				T.wait_flag("mte2", "v", 0)
				T.tile.sub(o_ub_half, zero_ub_half, o_ub_half)
				T.tile.sub(fat_o_ub_half, fat_zero_ub_half, fat_o_ub_half)
				T.set_flag("v", "mte3", 0)
				T.wait_flag("v", "mte3", 0)
				T.copy(o_ub_half, O[bz, by, bx * C + (vid * 2 + 1) * F, (vid * 2) * F])
				T.copy(fat_o_ub_half, O[bz, by, bx * C + (vid + 2) * F, 0])
				T.set_cross_flag("MTE3", 2)

				for ii in range(2):
					T.copy(A[bz, by, bx * C + (ii * 2 + vid) * F, (ii * 2 + vid) * F], a_ub_half[ii, :, :])
					T.tile.fill(mul_ub, 0.0)
					T.set_flag("mte2", "v", 0)
					T.wait_flag("mte2", "v", 0)
					T.copy(I[0, 0], i_ub)
					T.copy(a_ub_half[ii, :, :], o_ub)
					for i in range(2, F):
						T.copy(o_ub[i, :], o_row_ub)
						T.tile.broadcast(o_row_2d_ub, o_row_ub, tmp_ub)
						T.tile.fill(red_ub, 0.0)
						T.tile.mul(mul_ub, o_ub, o_row_2d_ub)
						T.tile.reduce_sum(red_ub, mul_ub, tmp_ub, dim = 0)
						T.tile.sub(o_ub[i, :], o_ub[i, :], red_ub)
					T.set_flag("mte2", "v", 0)
					T.wait_flag("mte2", "v", 0)
					T.tile.sub(o_ub, i_ub, o_ub)
					T.copy(o_ub, a_ub_half[ii, :, :])
					T.set_flag("v", "mte3", 0)
					T.wait_flag("v", "mte3", 0)
					T.copy(a_ub_half[ii, :, :], O[bz, by, bx * C + (ii * 2 + vid) * F, (ii * 2 + vid) * F])
					T.set_cross_flag("MTE3", ii)

			with T.Scope("C"):
				T.wait_cross_flag(2)
				T.copy(O[bz, by, bx * C + 2 * F, 0], a21_l_l1)
				T.copy(O[bz, by, bx * C + F, 0], a21_s_l1)
				for ii in range(2):
					T.wait_cross_flag(ii)
					T.copy(O[bz, by, bx * C + (ii * 2) * F, (ii * 2) * F], o11_s_l1)
					T.gemm_v0(a21_s_l1, o11_s_l1, mult_s_l0, init = True)
					T.copy(mult_s_l0, O[bz, by, bx * C + (ii * 2 + 1) * F, (ii * 2) * F])
					T.copy(O[bz, by, bx * C + (ii * 2 + 1) * F, (ii * 2 + 1) * F], o22_s_l1)
					T.set_flag("fix", "mte2", 0)
					T.wait_flag("fix", "mte2", 0)
					T.copy(O[bz, by, bx * C + (ii * 2 + 1) * F, (ii * 2) * F], mult_s_l1)
					T.gemm_v0(o22_s_l1, mult_s_l1, final_s_l0, init = True)
					T.copy(final_s_l0, O[bz, by, bx * C + (ii * 2 + 1) * F, (ii * 2) * F])
					if ii == 0:
						T.copy(O[bz, by, bx * C + 3 * F, 2 * F], a21_s_l1)
				
				T.copy(O[bz, by, bx * C, 0], o11_l_l1)
				T.set_flag("fix", "mte2", 0)
				T.wait_flag("fix", "mte2", 0)
				T.gemm_v0(a21_l_l1, o11_l_l1, mult_l_l0, init = True)
				T.copy(mult_l_l0, O[bz, by, bx * C + 2 * F, 0])
				T.copy(O[bz, by, bx * C + 2 * F, 2 * F], o22_l_l1)
				T.set_flag("fix", "mte2", 0)
				T.wait_flag("fix", "mte2", 0)
				T.copy(O[bz, by, bx * C + 2 * F, 0], mult_l_l1)
				T.gemm_v0(o22_l_l1, mult_l_l1, final_l_l0, init = True)
				T.copy(final_l_l0, O[bz, by, bx * C + 2 * F, 0])

	return main

def solve_tril(a):
	B, H, L, C = a.shape
	idt = torch.eye(C).npu().to(torch.float)
	if C == 32:
		ker = solve_tril_ker(B, H, L, C)
	elif C == 64:
		ker = solve_tril_64_ker(B, H, L)
	elif C == 128:
		ker = solve_tril_128_ker(B, H, L)
	b = ker(a, idt)
	return b

def solve_triangular(a):
	B, H, C = a.shape[0 : -1]
	idt = torch.eye(C).npu().to(torch.float).view(1, 1, C, C)
	for i in range(C):
		mul = torch.zeros((B, H, C, C)).npu().to(torch.float)
		for j in range(C):
			mul[:, :, j] = a[:, :, j] * a[:, :, i, j].unsqueeze(-1)
		mul = mul.sum(axis = -2)
		a[:, :, i] -= mul
	a = idt - a
	return a

def ref_solve_tril(a):
	B, H, L, C = a.shape
	chunk_num = (L + C - 1) // C
	o = torch.zeros((B, H, L, C)).npu().to(torch.float)
	a = a.float()
	for i in range(chunk_num):
		a_c = a[:, :, i * C : (i + 1) * C, :]
		o_c = solve_triangular(a_c)
		o[:, :, i * C : (i + 1) * C, :] = o_c
	return o.to(torch.float16)

def ref_chunk_cumsum(g, C):
	B, H, L = g.shape
	chunk_num = (L + C - 1) // C
	g = g.view(B, H, chunk_num, C)
	g_sum = torch.cumsum(g, dim = -1)
	g_sum = g_sum.view(B, H, L)
	return g_sum

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
		(1, 1, 128, 128, 128),
	]

	for B, H, L, DK, C in test_configs:
		print(f"Testing Solve Tril with B={B}, H={H}, L={L}, C={C}")
		chunk_num = (L + C - 1) // C
		k = torch.randn((B, H, L, DK)).npu().to(torch.float16)
		beta = torch.rand((B, H, L)).npu().to(torch.float16)
		g = torch.randn((B, H, L)).npu().to(torch.float)
		k = F.normalize(k, dim=-1, p=2)
		g = F.logsigmoid(g)
		g = ref_chunk_cumsum(g, C)
		a = ref_kkt(k, beta, g, C)

		o = solve_tril(a)
		ref_o = ref_solve_tril(a)
		torch.testing.assert_close(o.cpu(), ref_o.cpu(), rtol=1e-3, atol=1e-3)
		print("Test passed!")

	print("Kernel Output Match!")
