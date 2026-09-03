import pytest
import tilelang
import tilelang.language as T
import torch

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# =============================================================================
# Regression: W4A4 (int4 x int4 -> int32) gemm_v0 and fp16->int4 quantization
# (issue #438)
# -----------------------------------------------------------------------------
# int4b_t is a nibble-packed type: two elements share one byte (low nibble =
# even element). AscendC addresses int4 tensors in element units (offset / 2
# bytes), and the s4 cube path (Nd2Nz -> load_cbuf_to_ca_s4 /
# load_cbuf_to_cb_transpose_s4 -> mad_s4) expects fractal geometries with 64
# elements per 32B C0 block. The C++ helpers previously derived every fractal
# constant from ``sizeof(int4b_t) == 1`` (32 elements per C0), so all copy
# strides and the UB/GM byte math were 2x off and the matmul produced garbage.
#
# GM int4 tensors follow the packed convention: row r of a (M, N) int4 tensor
# starts at byte ``r * N / 2``. NOTE: ``torch.int4`` is an UNPACKED dtype (one
# byte per element in storage), so a torch int4 tensor of shape (M, N) is 2x
# larger than the packed kernel view -- kernels only touch the first M*N/2
# bytes. The helpers below build packed inputs by writing raw bytes into the
# tensor storage.
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.disable_cache()
    yield


def _compile(program, target="ascendc"):
    return tilelang.compile(program, pass_configs=PASS_CONFIGS, target=target)


def _pack_int4(t):
    """Pack an int tensor row-major into packed int4 bytes (low nibble first)."""
    m, n = t.shape
    assert n % 2 == 0
    b = t.reshape(m, n // 2, 2)
    return ((b[:, :, 0] & 0xF) | ((b[:, :, 1] & 0xF) << 4)).to(torch.uint8).flatten()


def _make_int4_npu(t):
    """Create a torch int4 NPU tensor holding ``t`` in the packed convention."""
    out = torch.empty(t.shape, dtype=torch.int4, device="npu")
    packed = _pack_int4(t).npu()
    out.view(torch.uint8).flatten()[: packed.numel()].copy_(packed)
    return out


def _fp16_to_int4_quant_kernel(M, N, block_M, block_N):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "int4"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M, block_N), "float16")
            q_ub = T.alloc_ub((block_M, block_N), "int4")

            with T.Scope("V"):
                T.copy(A[bx * block_M, by * block_N], a_ub)
                T.tile.cast(q_ub, a_ub, mode="CAST_RINT", count=block_M * block_N)
                T.copy(q_ub, B[bx * block_M, by * block_N])

    return main


def _gemm_v0_int4_kernel(M, N, K, block_M, block_N, K_L1):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "int4"),
        B: T.Tensor((K, N), "int4"),
        C: T.Tensor((M, N), "int32"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, K_L1), "int4")
            B_L1 = T.alloc_L1((K_L1, block_N), "int4")
            C_L0 = T.alloc_L0C((block_M, block_N), "int32")

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.barrier_all()
                    T.copy(B[k * K_L1, by * block_N], B_L1)
                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                    T.barrier_all()

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


requires_npu = pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="W4A4 correctness requires an Ascend NPU runtime",
)

requires_torch_int4 = pytest.mark.skipif(
    not hasattr(torch, "int4"),
    reason="packed-input helpers require torch.int4 (torch >= 2.3)",
)


@requires_npu
def test_fp16_to_int4_quant_packed():
    M, N = 128, 256
    torch.manual_seed(0)
    a = torch.randint(-8, 8, (M, N), dtype=torch.float16, device="npu")
    a_q = torch.empty(M, N, dtype=torch.int4, device="npu")

    kernel = _compile(_fp16_to_int4_quant_kernel(M, N, 128, 256))
    kernel(a, a_q)
    torch.npu.synchronize()

    expected = _pack_int4(a.cpu().to(torch.int32))
    raw = a_q.view(torch.uint8).flatten()[: expected.numel()].cpu()
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)


@requires_npu
@requires_torch_int4
@pytest.mark.parametrize(
    "M, N, K, block_M, block_N, K_L1",
    [
        (128, 256, 64, 128, 256, 64),  # single K tile, single N tile
        (128, 256, 1024, 128, 256, 64),  # K loop over multiple tiles
        (256, 512, 512, 128, 256, 64),  # multiple M/N blocks
        (64, 64, 64, 64, 64, 64),  # minimal single-fractal shapes
        (128, 256, 128, 128, 256, 128),  # K_L1 = 128: two s4 K blocks per L0B load
        (128, 256, 1024, 128, 256, 128),  # K loop with two-block L0B loads
        (128, 1024, 256, 128, 256, 64),  # N needs L0B N-tiling inside gemm_v0
    ],
)
def test_gemm_v0_int4(M, N, K, block_M, block_N, K_L1):
    torch.manual_seed(0)
    a = torch.randint(-8, 8, (M, K), dtype=torch.int16)
    b = torch.randint(-8, 8, (K, N), dtype=torch.int16)
    ref_c = a.to(torch.int32) @ b.to(torch.int32)

    a_q = _make_int4_npu(a)
    b_q = _make_int4_npu(b)
    torch.npu.synchronize()

    kernel = _compile(_gemm_v0_int4_kernel(M, N, K, block_M, block_N, K_L1))
    c = torch.zeros(M, N, dtype=torch.int32, device="npu")
    kernel(a_q, b_q, c)
    torch.npu.synchronize()

    # int4 -> int32 matmul is exact; assert a tight tolerance to catch any
    # layout/packing regression (the original bug mismatched ~99.6% of elements).
    torch.testing.assert_close(c.cpu(), ref_c, rtol=0, atol=0)


@requires_npu
@requires_torch_int4
def test_w4a4_quant_then_matmul():
    # End-to-end chain from issue #438: quantize both operands on NPU with the
    # tilelang quant kernel, then run the W4A4 matmul on the quantized tensors.
    M, N, K = 256, 256, 512
    block_M, block_N, block_K = 128, 256, 64

    torch.manual_seed(0)
    a = torch.randint(-8, 8, (M, K), dtype=torch.float16, device="npu")
    b = torch.randint(-8, 8, (K, N), dtype=torch.float16, device="npu")

    quant_a = _compile(_fp16_to_int4_quant_kernel(M, K, block_M, block_K))
    quant_b = _compile(_fp16_to_int4_quant_kernel(K, N, block_K, block_N))
    matmul = _compile(_gemm_v0_int4_kernel(M, N, K, block_M, block_N, block_K))

    a_q = torch.empty(M, K, dtype=torch.int4, device="npu")
    b_q = torch.empty(K, N, dtype=torch.int4, device="npu")
    c = torch.zeros(M, N, dtype=torch.int32, device="npu")

    quant_a(a, a_q)
    torch.npu.synchronize()
    quant_b(b, b_q)
    torch.npu.synchronize()
    matmul(a_q, b_q, c)
    torch.npu.synchronize()

    ref_c = a.cpu().to(torch.int32) @ b.cpu().to(torch.int32)
    torch.testing.assert_close(c.cpu(), ref_c, rtol=0, atol=0)


@requires_npu
def test_int4_pto_target_clear_error():
    # The pto backend cannot support int4 yet: pto-isa lacks the s4
    # instructions (mad_s4 / vconv_f162s4), see cann/pto-isa#115. The failure
    # must be an actionable message, not a bare "Unsupported datatype".
    with pytest.raises(Exception, match="int4 is not supported by the pto target"):
        _compile(_fp16_to_int4_quant_kernel(64, 64, 64, 64), target="pto")


if __name__ == "__main__":
    pytest.main(__file__)
