import tilelang
from tilelang import language as T
import torch

'''
Functionality:
Chunkwisely calculate the prefix sum
'''

pass_configs = {
	tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def cumsum_ker(B, H, L, C, accum_dtype="float"):
	chunk_num = T.ceildiv(L, C)
	VEC_NUM = 2

	@T.prim_func
	def main(
			G: T.Tensor([B, H, L], accum_dtype),
			S: T.Tensor([B, H, L], accum_dtype),
	):
		with T.Kernel(B * (H // VEC_NUM) * chunk_num, is_npu=True) as (cid, vid):
			bx = cid % chunk_num
			by = (cid // chunk_num) % (H // VEC_NUM) * 2 + vid
			bz = (cid // chunk_num) // (H // VEC_NUM)

			g_ub = T.alloc_ub([C,], accum_dtype)
			s_ub = T.alloc_ub([C,], accum_dtype)

			with T.Scope("V"):
				T.tile.fill(s_ub, 0.0)
				T.copy(G[bz, by, bx * C], g_ub)
				for i in range(C):
					if i > 0:
						s_ub[i] = s_ub[i - 1]
					tmp = s_ub[i] + g_ub[i]
					s_ub[i] = tmp
				T.copy(s_ub, S[bz, by, bx * C])
	
	return main

def chunk_cumsum(g, C):
	B, H, L = g.shape
	ker = cumsum_ker(B, H, L, C)
	g_sum = ker(g)
	return g_sum

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
		(2, 32, 256, 32),
	]

	for B, H, L, C in test_configs:
		print(f"Testing cumsum with B={B}, H={H}, L={L}, C={C}")
		g = torch.randn((B, H, L)).npu().to(torch.float)
		g_sum = chunk_cumsum(g, C)
		ref_g_sum = ref_chunk_cumsum(g, C)
		torch.testing.assert_close(g_sum.cpu(), ref_g_sum.cpu(), rtol=1e-5, atol=1e-5)
		print("Test passed!")
	
	print("Kernel Output Match!")
