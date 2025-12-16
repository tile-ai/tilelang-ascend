import tilelang
from tilelang import language as T
import torch

'''
Functionality:
O = ((Q * K^T) \odot M) * V, where M is the causal mask
'''

tilelang.cache.clear_cache()

pass_configs = {
	tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def linear_attention_ker(B, H, L, D, C, dtype="float16", accum_dtype="float"):
	shape = [B, H, L, D]
	chunk_num = T.ceildiv(L, C)
	VEC_NUM = 2

	@T.prim_func
	def main(
			Q: T.Tensor(shape, dtype),
			K: T.Tensor(shape, dtype),
			V: T.Tensor(shape, dtype),
			workspace_1: T.Tensor([B, H, C, C], dtype),
			workspace_2: T.Tensor([B, H, D, D], dtype),
			O: T.Tensor(shape, dtype),
	):
		with T.Kernel(B * H, is_npu=True) as (cid, vid):
			by = cid % H
			bz = cid // H

			q_l1 = T.alloc_L1([C, D], dtype)
			k_l1 = T.alloc_L1([C, D], dtype)
			v_l1 = T.alloc_L1([C, D], dtype)
			h_l1 = T.alloc_L1([D, D], dtype)
			acc_l1 = T.alloc_L1([C, C], dtype)
			h_l0 = T.alloc_L0C([D, D], accum_dtype)
			acc_l0 = T.alloc_L0C([C, C], accum_dtype)
			o_l0 = T.alloc_L0C([C, D], accum_dtype)

			hsum_ub = T.alloc_ub([D // VEC_NUM, D], dtype)
			h_ub = T.alloc_ub([D // VEC_NUM, D], dtype)
			acc_ub = T.alloc_ub([C // VEC_NUM, C], dtype)
			zero_ub = T.alloc_ub([C // VEC_NUM, C], dtype)

			with T.Scope("C"):
				for i in T.serial(chunk_num):
					T.copy(Q[bz, by, i * C, 0], q_l1)
					T.copy(K[bz, by, i * C, 0], k_l1)
					T.copy(V[bz, by, i * C, 0], v_l1)
					T.copy(workspace_2[bz, by, 0, 0], h_l1)
					T.gemm_v0(q_l1, k_l1, acc_l0, transpose_B = True, init = True)
					T.copy(acc_l0, workspace_1[bz, by, 0, 0])
					T.gemm_v0(k_l1, v_l1, h_l0, transpose_A = True, init = True)
					T.copy(h_l0, workspace_2[bz, by, 0, 0])
					T.set_cross_flag("FIX", 0)

					T.wait_cross_flag(1)
					T.copy(workspace_1[bz, by, 0, 0], acc_l1)
					T.gemm_v0(acc_l1, v_l1, o_l0, init = True)
					T.gemm_v0(q_l1, h_l1, o_l0, init = False)
					T.copy(o_l0, O[bz, by, i * C, 0])
			
			with T.Scope("V"):
				T.tile.fill(hsum_ub, 0.0)
				T.tile.fill(zero_ub, 0.0)
				for i in T.serial(chunk_num):
					T.wait_cross_flag(0)
					T.copy(workspace_1[bz, by, vid * C // VEC_NUM, 0], acc_ub)
					T.copy(workspace_2[bz, by, vid * D // VEC_NUM, 0], h_ub)
					for j in range(C // VEC_NUM):
						for k in range(C):
							if (j + vid * C // VEC_NUM) < k:
								acc_ub[j, k] = zero_ub[j, k]
					T.tile.add(hsum_ub, hsum_ub, h_ub)
					T.copy(acc_ub, workspace_1[bz, by, vid * C // VEC_NUM, 0])
					T.copy(hsum_ub, workspace_2[bz, by, vid * D // VEC_NUM, 0])
					T.set_cross_flag("MTE3", 1)
	
	return main

def linear_attention(q, k, v, C):
	B, H, L, D = q.shape
	ker = linear_attention_ker(B, H, L, D, C)
	workspace_1 = torch.zeros([B, H, C, C]).npu().to(torch.float16)
	workspace_2 = torch.zeros([B, H, D, D]).npu().to(torch.float16)
	o = ker(q, k, v, workspace_1, workspace_2)
	return o

def ref_linear_attention(q, k, v):
	B, H, L, D = q.shape
	q = q.float()
	k = k.float()
	v = v.float()
	h = torch.zeros([B, H, D, D]).npu().to(torch.float)
	o = torch.zeros([B, H, L, D]).npu().to(torch.float)
	for i in range(L):
		q_i = q[:, :, i, :]
		k_i = k[:, :, i, :]
		v_i = v[:, :, i, :]
		dh = torch.einsum("bhi,bhj->bhij", k_i, v_i)
		h = h + dh
		o_i = torch.einsum("bhi,bhij->bhj", q_i, h)
		o[:, :, i, :] = o_i
	return o.to(torch.float16)

torch.manual_seed(0)
torch.set_printoptions(threshold = float('inf'), sci_mode = False)

test_configs = [
	(2, 2, 512, 128, 64),
]

for B, H, L, D, C in test_configs:
	print(f"Testing linear attention with B={B}, H={H}, L={L}, D={D}, C={C}")
	q = torch.randn([B, H, L, D]).npu().to(torch.float16)
	k = torch.randn([B, H, L, D]).npu().to(torch.float16)
	v = torch.randn([B, H, L, D]).npu().to(torch.float16)
	q = q / (q.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
	k = k / (k.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
	o = linear_attention(q, k, v, C)
	ref_o = ref_linear_attention(q, k, v)
	torch.testing.assert_close(o.cpu(), ref_o.cpu(), rtol=1e-2, atol=1e-2)
	print("Test passed!")

print("Kernel Output Match!")
