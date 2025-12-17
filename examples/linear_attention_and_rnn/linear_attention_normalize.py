import tilelang
from tilelang import DataType, language as T
import torch

'''
Functionality:
O = (Q * K^T * V) / (Q * K^T * 1)
'''

tilelang.cache.clear_cache()

pass_configs = {
	tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True
}

# Calculate K^T*V and 1^T*K
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def linear_attention_ker1(B, H, L, D, block_L, block_D, dtype="float16", accum_dtype="float"):
	shape = [B, H, L, D]
	lb_num = T.ceildiv(L, block_L)
	db_num = T.ceildiv(D, block_D)
	VEC_NUM = 2

	@T.prim_func
	def main(
			K: T.Tensor(shape, dtype),
			V: T.Tensor(shape, dtype),
			Acc: T.Tensor([B, H, D + 1, D], dtype), # [0, D-1] for K^T*V, D for 1^T*K
	):
		with T.Kernel(B * H * db_num, is_npu=True) as (cid, vid):
			bx = cid % db_num
			by = (cid // db_num) % H
			bz = (cid // db_num) // H

			k_l1 = T.alloc_L1([block_L, block_D], dtype)
			v_l1 = T.alloc_L1([block_L, D], dtype)
			acc_l0 = T.alloc_L0C([block_D, D], accum_dtype)

			k_half_ub = T.alloc_ub([block_L, block_D // VEC_NUM], dtype)
			k_ub = T.alloc_ub([block_L, block_D // VEC_NUM], accum_dtype)
			sumk_ub = T.alloc_ub([block_D // VEC_NUM,], accum_dtype)
			acc_half_ub = T.alloc_ub([block_D // VEC_NUM,], dtype)
			acc_ub = T.alloc_ub([block_D // VEC_NUM,], accum_dtype)
			tmp_ub = T.alloc_ub([3 * DataType(accum_dtype).bits // 8 * block_L * block_D // VEC_NUM], "uint8")

			with T.Scope("C"):
				for i in T.serial(lb_num):
					T.copy(K[bz, by, i * block_L, bx * block_D], k_l1)
					T.copy(V[bz, by, i * block_L, 0], v_l1)
					T.gemm_v0(k_l1, v_l1, acc_l0, transpose_A = True, init = (i == 0))
				
				T.copy(acc_l0, Acc[bz, by, bx * block_D, 0])
			
			with T.Scope("V"):
				for i in T.serial(lb_num):
					T.copy(K[bz, by, i * block_L, bx * block_D + vid * block_D // VEC_NUM], k_half_ub)
					T.copy(k_half_ub, k_ub)
					T.tile.reduce_sum(sumk_ub, k_ub, tmp_ub, dim = 0)
					T.tile.add(acc_ub, acc_ub, sumk_ub)
				
				T.copy(acc_ub, acc_half_ub)
				T.copy(acc_half_ub, Acc[bz, by, D, bx * block_D + vid * block_D // VEC_NUM])

	return main

# Calculate QK^TV / QK^T*1
@tilelang.jit(out_idx=[-1], workspace_idx=[-2], pass_configs=pass_configs)
def linear_attention_ker2(B, H, L, D, block_L, block_D, dtype="float16", accum_dtype="float"):
	shape = [B, H, L, D]
	lb_num = T.ceildiv(L, block_L)
	db_num = T.ceildiv(D, block_D)
	VEC_NUM = 2

	@T.prim_func
	def main(
			Q: T.Tensor(shape, dtype),
			Acc: T.Tensor([B, H, D + 1, D], dtype), # [0, D-1] for K^T*V, D for 1^T*K
			workspace: T.Tensor([B * H * lb_num, block_L, D], accum_dtype),
			O: T.Tensor(shape, dtype),
	):
		with T.Kernel(B * H * lb_num, is_npu=True) as (cid, vid):
			bx = cid % lb_num
			by = (cid // lb_num) % H
			bz = (cid // lb_num) // H

			q_l1 = T.alloc_L1([block_L, block_D], dtype)
			acc_l1 = T.alloc_L1([block_D, D], dtype)
			o_l0 = T.alloc_L0C([block_L, D], accum_dtype)

			q_half_ub = T.alloc_ub([block_L // VEC_NUM, D], dtype)
			q_ub = T.alloc_ub([block_L // VEC_NUM, D], accum_dtype)
			qkv_ub = T.alloc_ub([block_L // VEC_NUM, D], accum_dtype)
			acc_half_ub = T.alloc_ub([D,], dtype)
			acc_ub = T.alloc_ub([D,], accum_dtype)
			denom_ub = T.alloc_ub([block_L // VEC_NUM,], accum_dtype)
			o_half_ub = T.alloc_ub([block_L // VEC_NUM, D], dtype)
			o_ub = T.alloc_ub([block_L // VEC_NUM, D], accum_dtype)
			tmp_ub = T.alloc_ub([3 * DataType(accum_dtype).bits // 8 * block_L * D // VEC_NUM], "uint8")

			with T.Scope("C"):
				for i in T.serial(db_num):
					T.copy(Q[bz, by, bx * block_L, i * block_D], q_l1)
					T.copy(Acc[bz, by, i * block_D, 0], acc_l1)
					T.gemm_v0(q_l1, acc_l1, o_l0, init = (i == 0))
				
				T.copy(o_l0, workspace[cid, 0, 0])
				T.set_cross_flag("FIX", 0)
			
			with T.Scope("V"):
				T.copy(Q[bz, by, bx* block_L + vid * block_L // VEC_NUM, 0], q_half_ub)
				T.copy(Acc[bz, by, D, 0], acc_half_ub)
				T.copy(q_half_ub, q_ub)
				T.copy(acc_half_ub, acc_ub)

				for h_i in range(block_L // VEC_NUM):
					T.tile.mul(q_ub[h_i, :], q_ub[h_i, :], acc_ub)
				
				T.tile.reduce_sum(denom_ub, q_ub, tmp_ub, dim = -1)
				T.wait_cross_flag(0)
				T.copy(workspace[cid, vid * block_L // VEC_NUM, 0], qkv_ub)

				for h_i in range(block_L // VEC_NUM):
					T.tile.div(o_ub[h_i, :], qkv_ub[h_i, :], denom_ub[h_i])
				
				T.copy(o_ub, o_half_ub)
				T.copy(o_half_ub, O[bz, by, bx * block_L + vid * block_L // VEC_NUM, 0])
	
	return main

def linear_attention(q, k, v, block_L, block_D):
	B, H, L, D = q.shape
	ker1 = linear_attention_ker1(B, H, L, D, block_L, block_D)
	ker2 = linear_attention_ker2(B, H, L, D, block_L, block_D)
	acc = ker1(k, v)
	o = ker2(q, acc)
	return o

def ref_linear_attention(q, k, v):
	q = q.float()
	k = k.float()
	v = v.float()

	s = torch.einsum("bhlk,bhlv->bhkv", k, v)
	o = torch.einsum("bhld,bhdv->bhlv", q, s)

	z = k.sum(axis = 2)
	denom = torch.einsum("bhld,bhd->bhl", q, z)
	o = o / denom.unsqueeze(-1)
	return o.to(torch.float16)

torch.manual_seed(0)
torch.set_printoptions(threshold = float('inf'), sci_mode = False)

test_configs = [
	(2, 2, 512, 128, 64, 64),
]

for B, H, L, D, block_L, block_D in test_configs:
	print(f"Testing linear attention with B={B}, H={H}, L={L}, D={D}, block_L={block_L}, block_D={block_D}")
	q = torch.randn([B, H, L, D]).npu().to(torch.float16)
	k = torch.randn([B, H, L, D]).npu().to(torch.float16)
	v = torch.randn([B, H, L, D]).npu().to(torch.float16)
	q = q.abs()
	k = k.abs()
	o = linear_attention(q, k, v, block_L, block_D)
	ref_o = ref_linear_attention(q, k, v)
	torch.testing.assert_close(o.cpu(), ref_o.cpu(), rtol=1e-3, atol=1e-3)
	print("Test passed!")

print("Kernel Output Match!")
