import argparse
import shutil

import tilelang
import tilelang.language as T
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.utils.target import determine_platform

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel AOT Compilation")
parser.add_argument("--m", type=int, default=8192, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=8192, help="Matrix K dimension")
parser.add_argument(
    "--target",
    type=str,
    default="ascendc",
    choices=["ascendc", "pto"],
    help="Codegen/compile backend",
)
parser.add_argument(
    "--platform",
    type=str,
    default="auto",
    help="Hardware platform (auto/A2/A3/A5); auto-detected by default",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="./kernel_lib.so",
    help="Output shared library path",
)
args = parser.parse_args()

M = args.m
N = args.n
K = args.k


def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):

        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)

            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M:(bx + 1) * block_M, k * K_L1:(k + 1) * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)

                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

                    T.barrier_all()

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


func = matmul(M, N, K, 128, 256, 64)

# Generate the device source with the framework, then compile it into a shared
# library using the same LibraryGenerator that the JIT path uses. This keeps the
# AOT build in sync with the framework's compiler flags across toolkit releases,
# instead of duplicating them in a standalone build script.
platform = determine_platform(args.platform)
artifact = tilelang.engine.lower(func, target=args.target, platform=platform)

lib_generator = LibraryGenerator(target=args.target, platform=platform)
lib_generator.update_lib_code(artifact.kernel_source)
lib_generator.compile_lib()
shutil.copy(lib_generator.get_lib_path(), args.output)

print(f"Built {args.output} (target={args.target}, platform={platform})")
