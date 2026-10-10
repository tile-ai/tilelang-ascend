import argparse
import tilelang
from tilelang import language as T
import torch


def _check_precision(actual, golden, dtype):
    limits = {"float16": (2**-14, 2**-9, 0.1), "bfloat16": (2**-10, 2**-6, 1.0), "float32": (2**-16, 2**-10, 0.01)}
    actual_cpu, golden_cpu = actual.detach().cpu(), golden.detach().cpu()
    if actual_cpu.shape != golden_cpu.shape:
        return False, 0.0, float("inf")
    name = str(dtype).replace("torch.", "")
    if name in {"int8", "int16", "int32", "int64", "uint8"}:
        mismatches = (actual_cpu != golden_cpu).sum().item()
        return mismatches == 0, 1.0 - mismatches / max(actual_cpu.numel(), 1), 0.0 if mismatches == 0 else float("inf")
    atol, rtol, maximum = limits.get(name, limits["float16"])
    actual_cpu, golden_cpu = actual_cpu.float(), golden_cpu.float()
    special = ~torch.isfinite(golden_cpu)
    if special.any() and (
        not torch.equal(torch.isnan(actual_cpu[special]), torch.isnan(golden_cpu[special]))
        or not torch.equal(torch.isinf(actual_cpu[special]), torch.isinf(golden_cpu[special]))
        or not torch.equal(actual_cpu[special][torch.isinf(golden_cpu[special])], golden_cpu[special][torch.isinf(golden_cpu[special])])
    ):
        return False, 0.0, float("inf")
    finite = torch.isfinite(golden_cpu)
    if not finite.any():
        return True, 1.0, 0.0
    error = (actual_cpu[finite] - golden_cpu[finite]).abs()
    error = torch.where(torch.isfinite(error), error, torch.full_like(error, float("inf")))
    ratio, max_abs = (error <= atol + rtol * golden_cpu[finite].abs()).float().mean().item(), error.max().item()
    return ratio >= 0.99 and max_abs <= maximum, ratio, max_abs


@tilelang.jit(out_idx=[1])
def broadcast_pipeline(M, N, block_M, sub_M, dtype="float"):
    m_num = M // block_M
    VEC_NUM = 2
    stages = 2

    sub_block_M = sub_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor([1, N], dtype),
        B: T.Tensor([M, N], dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((stages, 1, N), dtype)
            b_ub = T.alloc_ub((stages, sub_block_M, N), dtype)

            row_base = cid * block_M

            with T.Scope("V"):
                T.barrier_all()

                T.copy(A[0, :], a_ub[0, 0, :])
                T.barrier_all()

                for stage in T.serial(block_M // sub_M):
                    cur = stage % stages
                    nxt = (stage + 1) % stages

                    if stage < (block_M // sub_M) - 1:
                        T.barrier_all()
                        T.copy(A[0, :], a_ub[nxt, 0, :])
                        T.barrier_all()

                    T.barrier_all()

                    cur_row_start = row_base + stage * sub_M

                    T.tile.broadcast(b_ub[cur, :, :], a_ub[cur, 0, :])

                    T.barrier_all()

                    T.copy(b_ub[cur, :, :], B[cur_row_start + vid * sub_block_M : cur_row_start + (vid + 1) * sub_block_M, :])

                    T.barrier_all()

    return main


if __name__ == "__main__":
    tilelang.cache.clear_cache()

    parser = argparse.ArgumentParser(description="Broadcast Pipeline NPU Kernel Compilation")
    parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
    parser.add_argument("--n", type=int, default=256, help="Matrix N dimension")
    parser.add_argument("--block-m", type=int, default=128, help="Block size in M dimension")
    parser.add_argument("--sub-m", type=int, default=64, help="Sub-block size for pipeline")
    args = parser.parse_args()

    M = args.m
    N = args.n
    block_M = args.block_m
    sub_M = args.sub_m

    print(f"Configuration: M={M}, N={N}, block_M={block_M}, sub_M={sub_M}")

    func = broadcast_pipeline(M, N, block_M, sub_M)

    torch.manual_seed(0)

    a = torch.randn(1, N).npu()

    torch.npu.synchronize()
    print("init successful!")

    c = func(a)

    ref_c = a.expand(M, N)

    passed, ratio, max_abs = _check_precision(c, ref_c, c.dtype)
    assert passed, f"dtype={c.dtype}, matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"
    print("Kernel Output Match!")
