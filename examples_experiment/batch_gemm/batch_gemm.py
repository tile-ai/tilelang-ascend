import argparse

import tilelang
import tilelang.language as T
import torch


def _check_precision(actual, golden, dtype):
    table = {
        "float16": (2**-14, 2**-9, 0.1),
        "bfloat16": (2**-10, 2**-6, 1.0),
        "float32": (2**-16, 2**-10, 0.01),
        "hifloat32": (2**-16, 2**-10, 0.01),
        "float8_e4m3": (2**-4, 2**-2, 1.0),
        "float8_e5m2": (2**-3, 2**-1, 0.1),
    }
    actual, golden = actual.detach().cpu(), golden.detach().cpu()
    if actual.shape != golden.shape:
        return False, 0.0, float("inf")
    name = str(dtype).replace("torch.", "")
    if name.startswith("float8_e4m3"):
        name = "float8_e4m3"
    if name.startswith("float8_e5m2"):
        name = "float8_e5m2"
    if name in {"int8", "int16", "int32", "int64", "uint8"}:
        mismatches = (actual != golden).sum().item()
        total = max(actual.numel(), 1)
        return mismatches == 0, 1.0 - mismatches / total, 0.0 if mismatches == 0 else float("inf")
    atol, rtol, limit = table.get(name, table["float16"])
    actual, golden = actual.float(), golden.float()
    special = ~torch.isfinite(golden)
    if special.any() and (
        not torch.equal(torch.isnan(actual[special]), torch.isnan(golden[special]))
        or not torch.equal(torch.isinf(actual[special]), torch.isinf(golden[special]))
        or not torch.equal(actual[special][torch.isinf(golden[special])], golden[special][torch.isinf(golden[special])])
    ):
        return False, 0.0, float("inf")
    finite = torch.isfinite(golden)
    if not finite.any():
        return True, 1.0, 0.0
    error = (actual[finite] - golden[finite]).abs()
    error = torch.where(torch.isfinite(error), error, torch.full_like(error, float("inf")))
    ratio, max_abs = (error <= atol + rtol * golden[finite].abs()).float().mean().item(), error.max().item()
    return ratio >= 0.99 and max_abs <= limit, ratio, max_abs


tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Batch Kernel Compilation")
parser.add_argument("--b", type=int, default=8, help="Batch size")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=1024, help="Matrix K dimension")
args = parser.parse_args()

B = args.b
M = args.m
N = args.n
K = args.k

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def batch_matmul(B, M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A_mat: T.Tensor((B, M, K), dtype),
        B_mat: T.Tensor((B, K, N), dtype),
        C_mat: T.Tensor((B, M, N), dtype),
    ):
        total = B * m_num * n_num
        with T.Kernel(total, is_npu=True) as (cid, _):
            bid = cid // (m_num * n_num)
            rem = cid % (m_num * n_num)
            bx = rem // n_num
            by = rem % n_num

            A_L1 = T.alloc_shared((block_M, K_L1), dtype)
            B_L1 = T.alloc_shared((K_L1, block_N), dtype)

            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            loop_k = T.ceildiv(K, K_L1)
            for k in T.serial(loop_k):
                T.copy(A_mat[bid, bx * block_M, k * K_L1], A_L1)
                T.copy(B_mat[bid, k * K_L1, by * block_N], B_L1)

                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

            T.copy(C_L0, C_mat[bid, bx * block_M, by * block_N])

    return main


if __name__ == "__main__":
    func = batch_matmul(B, M, N, K, 128, 256, 64)

    torch.manual_seed(0)

    a = torch.randn(B, M, K).half().npu()
    b = torch.randn(B, K, N).half().npu()

    print("init successful!")

    c = func(a, b)

    ref_c = torch.matmul(a, b)

    passed, ratio, max_abs = _check_precision(c, ref_c, c.dtype)
    assert passed, f"dtype={c.dtype}, matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"
    print("Batch Kernel Output Match!")
