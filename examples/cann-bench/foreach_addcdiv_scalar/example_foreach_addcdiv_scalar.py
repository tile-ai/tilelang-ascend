"""ForeachAddcdivScalar example: fused elementwise with MTE2/V/MTE3 pipeline.

Formula: y = x1 + (x2 / x3) * scalar

This example demonstrates two advanced kernel patterns on Ascend NPU:

1. Pipeline kernel (finite scalar): A three-stage MTE2 -> V -> MTE3 software
   pipeline with set_flag/wait_flag synchronization and double-buffered UB
   (stages=2). The scalar is baked in as a compile-time constant to enable
   T.tile.axpy fusion (mul + add in one SIMD instruction).

2. Barrier kernel (inf/nan scalar): A fallback path that loads the scalar
   from a small GM tensor into scalar_ub at runtime, working around the
   CUDART_INF/CUDART_NAN undeclared-identifier codegen errors that occur
   when inf/nan is used as a compile-time axpy constant.

Both kernels:
  - Expert mode: T.alloc_ub, T.Scope("V"), T.tile.div/axpy
  - FP16/BF16 -> FP32 compute -> cast back (cast_or_copy pattern)
  - pad_value handles non-aligned shapes (x1/x2 pad=0.0, x3 pad=1.0)
  - VEC_NUM=2 row split across two Vector cores
"""

import argparse

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="ForeachAddcdivScalar NPU Kernel")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
parser.add_argument("--dtype", type=str, default="float", choices=["float", "float16", "bfloat16"])
args = parser.parse_args()

M = args.m
N = args.n
DTYPE = args.dtype

CAST_MODE_LOW2HIGH = "CAST_NONE"
CAST_MODE_HIGH2LOW = "CAST_RINT"
_SCALAR_PAD = 8

_PIPELINE_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_BARRIER_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}


@tilelang.jit(out_idx=[3], pass_configs=_PIPELINE_CONFIGS)
def addcdiv_pipeline_fp32(M, N, block_M, block_N, sub_M, scalar, dtype="float"):
    """Pipeline kernel for finite scalar, fp32 path (no cal buffers).

    Computes directly on UB — no cast needed. Only 3 input UB buffers
    (double-buffered), giving more UB budget for larger block_N.
    """
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_M = sub_M // VEC_NUM
    vec_proc = block_M // sub_M
    stages = 2

    @T.prim_func
    def main(
        X1: T.Tensor((M, N), dtype),  # type: ignore
        X2: T.Tensor((M, N), dtype),  # type: ignore
        X3: T.Tensor((M, N), dtype),  # type: ignore
        Y: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            x1_ub = T.alloc_ub((stages, sub_block_M, block_N), dtype)
            x2_ub = T.alloc_ub((stages, sub_block_M, block_N), dtype)
            x3_ub = T.alloc_ub((stages, sub_block_M, block_N), dtype)

            with T.Scope("V"):
                col_start = by * block_N
                row_base = bx * block_M + vid * sub_block_M

                T.set_flag("mte3", "mte2", 0)
                T.set_flag("mte3", "mte2", 1)

                T.wait_flag("mte3", "mte2", 0)
                T.copy(X1[row_base, col_start], x1_ub[0, :, :], pad_value=0.0)
                T.copy(X2[row_base, col_start], x2_ub[0, :, :], pad_value=0.0)
                T.copy(X3[row_base, col_start], x3_ub[0, :, :], pad_value=1.0)
                T.set_flag("mte2", "v", 0)

                for mm in T.serial(vec_proc):
                    cur = mm % stages
                    nxt = (mm + 1) % stages

                    if mm < vec_proc - 1:
                        T.wait_flag("mte3", "mte2", nxt)
                        row_nxt = row_base + (mm + 1) * sub_M
                        T.copy(X1[row_nxt, col_start], x1_ub[nxt, :, :], pad_value=0.0)
                        T.copy(X2[row_nxt, col_start], x2_ub[nxt, :, :], pad_value=0.0)
                        T.copy(X3[row_nxt, col_start], x3_ub[nxt, :, :], pad_value=1.0)
                        T.set_flag("mte2", "v", nxt)

                    T.wait_flag("mte2", "v", cur)
                    T.tile.div(x2_ub[cur, :, :], x2_ub[cur, :, :], x3_ub[cur, :, :])
                    T.tile.axpy(x1_ub[cur, :, :], x2_ub[cur, :, :], scalar)
                    T.set_flag("v", "mte3", cur)

                    T.wait_flag("v", "mte3", cur)
                    row_cur = row_base + mm * sub_M
                    T.copy(x1_ub[cur, :, :], Y[row_cur, col_start])
                    T.set_flag("mte3", "mte2", cur)

                T.wait_flag("mte3", "mte2", 0)
                T.wait_flag("mte3", "mte2", 1)

    return main


@tilelang.jit(out_idx=[3], pass_configs=_PIPELINE_CONFIGS)
def addcdiv_pipeline_lowprec(M, N, block_M, block_N, sub_M, scalar, dtype="float16"):
    """Pipeline kernel for finite scalar, fp16/bf16 path (with fp32 cal buffers).

    Casts to fp32 for compute precision, then casts back. Uses 6 UB buffers
    (3 input + 3 cal), all double-buffered for input.
    """
    cal_dtype = "float32"

    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_M = sub_M // VEC_NUM
    vec_proc = block_M // sub_M
    stages = 2
    cnt = sub_block_M * block_N

    @T.prim_func
    def main(
        X1: T.Tensor((M, N), dtype),  # type: ignore
        X2: T.Tensor((M, N), dtype),  # type: ignore
        X3: T.Tensor((M, N), dtype),  # type: ignore
        Y: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            x1_ub = T.alloc_ub((stages, sub_block_M, block_N), dtype)
            x2_ub = T.alloc_ub((stages, sub_block_M, block_N), dtype)
            x3_ub = T.alloc_ub((stages, sub_block_M, block_N), dtype)
            x1_cal = T.alloc_ub((sub_block_M, block_N), cal_dtype)
            x2_cal = T.alloc_ub((sub_block_M, block_N), cal_dtype)
            x3_cal = T.alloc_ub((sub_block_M, block_N), cal_dtype)

            with T.Scope("V"):
                col_start = by * block_N
                row_base = bx * block_M + vid * sub_block_M

                T.set_flag("mte3", "mte2", 0)
                T.set_flag("mte3", "mte2", 1)

                T.wait_flag("mte3", "mte2", 0)
                T.copy(X1[row_base, col_start], x1_ub[0, :, :], pad_value=0.0)
                T.copy(X2[row_base, col_start], x2_ub[0, :, :], pad_value=0.0)
                T.copy(X3[row_base, col_start], x3_ub[0, :, :], pad_value=1.0)
                T.set_flag("mte2", "v", 0)

                for mm in T.serial(vec_proc):
                    cur = mm % stages
                    nxt = (mm + 1) % stages

                    if mm < vec_proc - 1:
                        T.wait_flag("mte3", "mte2", nxt)
                        row_nxt = row_base + (mm + 1) * sub_M
                        T.copy(X1[row_nxt, col_start], x1_ub[nxt, :, :], pad_value=0.0)
                        T.copy(X2[row_nxt, col_start], x2_ub[nxt, :, :], pad_value=0.0)
                        T.copy(X3[row_nxt, col_start], x3_ub[nxt, :, :], pad_value=1.0)
                        T.set_flag("mte2", "v", nxt)

                    T.wait_flag("mte2", "v", cur)
                    T.tile.cast(x1_cal, x1_ub[cur, :, :], CAST_MODE_LOW2HIGH, cnt)
                    T.tile.cast(x2_cal, x2_ub[cur, :, :], CAST_MODE_LOW2HIGH, cnt)
                    T.tile.cast(x3_cal, x3_ub[cur, :, :], CAST_MODE_LOW2HIGH, cnt)
                    T.tile.div(x2_cal, x2_cal, x3_cal)
                    T.tile.axpy(x1_cal, x2_cal, scalar)
                    T.tile.cast(x1_ub[cur, :, :], x1_cal, CAST_MODE_HIGH2LOW, cnt)
                    T.set_flag("v", "mte3", cur)

                    T.wait_flag("v", "mte3", cur)
                    row_cur = row_base + mm * sub_M
                    T.copy(x1_ub[cur, :, :], Y[row_cur, col_start])
                    T.set_flag("mte3", "mte2", cur)

                T.wait_flag("mte3", "mte2", 0)
                T.wait_flag("mte3", "mte2", 1)

    return main


@tilelang.jit(out_idx=[4], pass_configs=_BARRIER_CONFIGS)
def addcdiv_barrier(M, N, block_M, block_N, dtype="float"):
    """Barrier kernel for inf/nan scalar. Uses scalar_ub with barrier_all."""
    use_fp32_compute = dtype in ["bfloat16", "float16"]
    cal_dtype = "float32" if use_fp32_compute else dtype

    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    def cast_or_copy(dst, src, mode, count):
        if use_fp32_compute:
            return T.tile.cast(dst, src, mode, count)
        else:
            return T.copy(src, dst)

    @T.prim_func
    def main(
        X1: T.Tensor((M, N), dtype),  # type: ignore
        X2: T.Tensor((M, N), dtype),  # type: ignore
        X3: T.Tensor((M, N), dtype),  # type: ignore
        S: T.Tensor((_SCALAR_PAD,), cal_dtype),  # type: ignore
        Y: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            x1_ub = T.alloc_ub((sub_block_M, block_N), dtype)
            x2_ub = T.alloc_ub((sub_block_M, block_N), dtype)
            x3_ub = T.alloc_ub((sub_block_M, block_N), dtype)
            x1_cal = T.alloc_ub((sub_block_M, block_N), cal_dtype)
            x2_cal = T.alloc_ub((sub_block_M, block_N), cal_dtype)
            x3_cal = T.alloc_ub((sub_block_M, block_N), cal_dtype)
            scalar_ub = T.alloc_ub((_SCALAR_PAD,), cal_dtype)

            with T.Scope("V"):
                row_start = bx * block_M + vid * sub_block_M
                col_start = by * block_N

                T.copy(X1[row_start, col_start], x1_ub, pad_value=0.0)
                T.copy(X2[row_start, col_start], x2_ub, pad_value=0.0)
                T.copy(X3[row_start, col_start], x3_ub, pad_value=1.0)
                T.copy(S, scalar_ub)

                T.barrier_all()

                if use_fp32_compute:
                    cast_or_copy(x1_cal, x1_ub, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                    cast_or_copy(x2_cal, x2_ub, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                    cast_or_copy(x3_cal, x3_ub, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                    T.tile.div(x2_cal, x2_cal, x3_cal)
                    T.tile.axpy(x1_cal, x2_cal, scalar_ub[0])
                    cast_or_copy(x1_ub, x1_cal, CAST_MODE_HIGH2LOW, sub_block_M * block_N)
                else:
                    T.tile.div(x2_ub, x2_ub, x3_ub)
                    T.tile.axpy(x1_ub, x2_ub, scalar_ub[0])

                T.barrier_all()

                T.copy(x1_ub, Y[row_start, col_start])

    return main


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
torch.manual_seed(0)
torch_dtype = {"float": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[DTYPE]

x1 = torch.randn(M, N, dtype=torch.float32).to(torch_dtype).npu()
x2 = torch.randn(M, N, dtype=torch.float32).to(torch_dtype).npu()
x3 = (torch.rand(M, N, dtype=torch.float32) * 1.5 + 0.5).to(torch_dtype).npu()

torch.npu.synchronize()
print(f"init successful! M={M}, N={N}, dtype={DTYPE}")

# Test 1: finite scalar (pipeline kernel)
scalar = 1.0
print(f"\n[Test 1] Pipeline kernel, scalar={scalar}")
if DTYPE == "float":
    kernel = addcdiv_pipeline_fp32(M, N, 128, 256, 32, scalar, dtype=DTYPE)
else:
    kernel = addcdiv_pipeline_lowprec(M, N, 128, 256, 32, scalar, dtype=DTYPE)
y = kernel(x1, x2, x3)
torch.npu.synchronize()

ref = (x1.float() + (x2.float() / x3.float()) * scalar).to(torch_dtype)
torch.testing.assert_close(y, ref, rtol=1e-2, atol=1e-2)
print("  Kernel Output Match!")

# Test 2: inf scalar (barrier kernel)
scalar_inf = float("inf")
print(f"\n[Test 2] Barrier kernel, scalar={scalar_inf}")
# Barrier kernel allocates 7 UB buffers (3 input + 3 cal + scalar), so
# block_M must be clamped to fit within 192 KB UB limit.
barrier_block_M = 128
barrier_block_N = 256
VEC_NUM = 2
_sub = barrier_block_M // VEC_NUM
_ub_total = 3 * _sub * barrier_block_N * 4 + 3 * _sub * barrier_block_N * 4 + 8 * 4
while _ub_total > 192 * 1024 and barrier_block_M >= 4:
    barrier_block_M //= 2
    _sub = barrier_block_M // VEC_NUM
    _ub_total = 3 * _sub * barrier_block_N * 4 + 3 * _sub * barrier_block_N * 4 + 8 * 4
kernel_barrier = addcdiv_barrier(M, N, barrier_block_M, barrier_block_N, dtype=DTYPE)
s_tensor = torch.zeros(_SCALAR_PAD, dtype=torch.float32)
s_tensor[0] = scalar_inf
s_tensor = s_tensor.npu()
y_inf = kernel_barrier(x1, x2, x3, s_tensor)
torch.npu.synchronize()

ref_inf = (x1.float() + (x2.float() / x3.float()) * scalar_inf).to(torch_dtype)
assert torch.equal(y_inf, ref_inf), "inf scalar mismatch"
print("  Kernel Output Match!")

# Test 3: nan scalar (barrier kernel)
scalar_nan = float("nan")
print(f"\n[Test 3] Barrier kernel, scalar={scalar_nan}")
s_tensor_nan = torch.zeros(_SCALAR_PAD, dtype=torch.float32)
s_tensor_nan[0] = scalar_nan
s_tensor_nan = s_tensor_nan.npu()
y_nan = kernel_barrier(x1, x2, x3, s_tensor_nan)
torch.npu.synchronize()

ref_nan = (x1.float() + (x2.float() / x3.float()) * scalar_nan).to(torch_dtype)
assert torch.isnan(y_nan).all() and torch.isnan(ref_nan).all(), "nan scalar mismatch"
print("  Kernel Output Match!")

print("\nAll tests passed!")
