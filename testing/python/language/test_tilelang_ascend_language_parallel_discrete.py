import pytest
import tilelang
import tilelang.language as T
import torch


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.cache.clear_cache()
    yield


def _make_idx(M: int, block_M: int, vec_num: int, mode: str = "reverse") -> torch.Tensor:
    """
    Per-row IDX, matches the "discrete index" pattern: A[idx[i], j]
    """
    block_rows = block_M // vec_num
    if mode == "reverse":
        idx_cpu = torch.arange(block_rows - 1, -1, -1, dtype=torch.int32)
    elif mode == "mod_k":
        K = 8
        idx_cpu = (torch.arange(block_rows, dtype=torch.int32) % K).to(torch.int32)
    elif mode == "affine":
        idx_cpu = ((torch.arange(block_rows, dtype=torch.int32) * 3 + 1) % block_rows).to(torch.int32)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return idx_cpu.repeat(M // block_M * vec_num).npu()


def _ref_discrete_gather_2d(A, IDX, block_M: int, vec_num: int):
    M, N = A.shape
    block_rows = block_M // vec_num
    out = torch.empty((M, N), device=A.device, dtype=A.dtype)
    for base in range(0, M, block_rows):
        a_block = A[base : base + block_rows, :]
        idx_block = IDX[base : base + block_rows].to(torch.long)
        gather_index = idx_block[:, None].expand(block_rows, N)
        out[base : base + block_rows, :] = a_block.gather(dim=0, index=gather_index)
    return out


# 1) 2 matrices: C[i, j] = A[idx[i], j] * B[i, j]
@tilelang.jit(out_idx=[-1])
def discrete_case_mat_mat(M, N, block_M=128, block_N=128, dtype="float"):
    vec_num = 2
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        IDX: T.Tensor((M,), "int32"),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a = T.alloc_ub((block_M // vec_num, block_N), dtype)
            b = T.alloc_ub((block_M // vec_num, block_N), dtype)
            idx = T.alloc_ub((block_M // vec_num,), "int32")
            c = T.alloc_ub((block_M // vec_num, block_N), dtype)

            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * (block_M // vec_num), by * block_N], a)
                T.copy(B[bx * block_M + vid * (block_M // vec_num), by * block_N], b)
                T.copy(IDX[bx * block_M + vid * (block_M // vec_num)], idx)

                T.barrier_all()
                for i, j in T.Parallel(block_M // vec_num, block_N):
                    c[i, j] = a[idx[i], j] * b[i, j]
                T.barrier_all()

                T.copy(c, C[bx * block_M + vid * (block_M // vec_num), by * block_N])

    return main


def ref_discrete_case_mat_mat(A, B, IDX, C, block_M=128, vec_num=2):
    return _ref_discrete_gather_2d(A, IDX, block_M, vec_num) * B


@pytest.mark.parametrize("idx_mode", ["reverse", "mod_k", "affine"])
def test_parallel_discrete_mat_mat(idx_mode):
    torch.manual_seed(0)
    M, N = 1024, 1024
    block_M, block_N = 128, 128
    vec_num = 2

    func = discrete_case_mat_mat(M, N, block_M, block_N)
    A = torch.randn((M, N), device="npu", dtype=torch.float32)
    B = torch.randn((M, N), device="npu", dtype=A.dtype)
    IDX = _make_idx(M, block_M, vec_num, mode=idx_mode)
    C = torch.empty((M, N), device="npu", dtype=A.dtype)

    torch.npu.synchronize()
    out = func(A, B, IDX, C)
    ref = ref_discrete_case_mat_mat(A, B, IDX, C, block_M, vec_num)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


# 2) 1 matrix + row vector: C[i, j] = A[idx[i], j] + B_row[i]
@tilelang.jit(out_idx=[-1])
def discrete_case_mat_row(M, N, block_M=128, block_N=128, dtype="float"):
    vec_num = 2
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M,), dtype),
        IDX: T.Tensor((M,), "int32"),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a = T.alloc_ub((block_M // vec_num, block_N), dtype)
            b = T.alloc_ub((block_M // vec_num,), dtype)
            idx = T.alloc_ub((block_M // vec_num,), "int32")
            c = T.alloc_ub((block_M // vec_num, block_N), dtype)

            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * (block_M // vec_num), by * block_N], a)
                T.copy(B[bx * block_M + vid * (block_M // vec_num)], b)
                T.copy(IDX[bx * block_M + vid * (block_M // vec_num)], idx)

                T.barrier_all()
                for i, j in T.Parallel(block_M // vec_num, block_N):
                    c[i, j] = a[idx[i], j] + b[i]
                T.barrier_all()

                T.copy(c, C[bx * block_M + vid * (block_M // vec_num), by * block_N])

    return main


def ref_discrete_case_mat_row(A, B, IDX, C, block_M=128, vec_num=2):
    return _ref_discrete_gather_2d(A, IDX, block_M, vec_num) + B.unsqueeze(1)


@pytest.mark.parametrize("idx_mode", ["reverse", "mod_k", "affine"])
def test_parallel_discrete_mat_row(idx_mode):
    torch.manual_seed(0)
    M, N = 1024, 1024
    block_M, block_N = 128, 128
    vec_num = 2

    func = discrete_case_mat_row(M, N, block_M, block_N)
    A = torch.randn((M, N), device="npu", dtype=torch.float32)
    B = torch.randn((M,), device="npu", dtype=A.dtype)
    IDX = _make_idx(M, block_M, vec_num, mode=idx_mode)
    C = torch.empty((M, N), device="npu", dtype=A.dtype)

    torch.npu.synchronize()
    out = func(A, B, IDX, C)
    ref = ref_discrete_case_mat_row(A, B, IDX, C, block_M, vec_num)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


# 3) 1 matrix + col vector: C[i, j] = A[idx[i], j] + B_col[j]
@tilelang.jit(out_idx=[-1])
def discrete_case_mat_col(M, N, block_M=128, block_N=128, dtype="float"):
    vec_num = 2
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((N,), dtype),
        IDX: T.Tensor((M,), "int32"),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a = T.alloc_ub((block_M // vec_num, block_N), dtype)
            b = T.alloc_ub((block_N,), dtype)
            idx = T.alloc_ub((block_M // vec_num,), "int32")
            c = T.alloc_ub((block_M // vec_num, block_N), dtype)

            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * (block_M // vec_num), by * block_N], a)
                T.copy(B[by * block_N], b)
                T.copy(IDX[bx * block_M + vid * (block_M // vec_num)], idx)

                T.barrier_all()
                for i, j in T.Parallel(block_M // vec_num, block_N):
                    c[i, j] = a[idx[i], j] + b[j]
                T.barrier_all()

                T.copy(c, C[bx * block_M + vid * (block_M // vec_num), by * block_N])

    return main


def ref_discrete_case_mat_col(A, B, IDX, C, block_M=128, vec_num=2):
    return _ref_discrete_gather_2d(A, IDX, block_M, vec_num) + B.unsqueeze(0)


@pytest.mark.parametrize("idx_mode", ["reverse", "mod_k", "affine"])
def test_parallel_discrete_mat_col(idx_mode):
    torch.manual_seed(0)
    M, N = 1024, 1024
    block_M, block_N = 128, 128
    vec_num = 2

    func = discrete_case_mat_col(M, N, block_M, block_N)
    A = torch.randn((M, N), device="npu", dtype=torch.float32)
    B = torch.randn((N,), device="npu", dtype=A.dtype)
    IDX = _make_idx(M, block_M, vec_num, mode=idx_mode)
    C = torch.empty((M, N), device="npu", dtype=A.dtype)

    torch.npu.synchronize()
    out = func(A, B, IDX, C)
    ref = ref_discrete_case_mat_col(A, B, IDX, C, block_M, vec_num)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])