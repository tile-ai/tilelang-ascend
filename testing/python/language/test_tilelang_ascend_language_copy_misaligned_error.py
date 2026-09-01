import pytest
import tilelang
import tilelang.language as T
import torch

"""
Copy address-alignment compile-time error (codegen hardening).

Feature under test
------------------
AscendC's DataCopyPad requires the **LocalTensor (UB) start address** to be
32-Byte aligned (the GlobalMemory side has no alignment constraint).  A UB copy
destination / source whose column offset is a *compile-time constant* that is
NOT a 32-Byte multiple (in the buffer's dtype) is a latent correctness bug: the
DMA engine silently operates on a misaligned UB base.

This hardening makes the violation explicit at compile time: when a copy's UB
base offset is a compile-time constant whose byte value is not a 32-Byte
multiple, the codegen raises a clear error naming the buffer, the offset, and
the alignment requirement.  Runtime (non-constant) offsets are let through
(they may be aligned at run time; rejecting them would be a conservative
false-positive).

Cases
-----
  * misaligned constant UB dst offset (fp32 offset 1 -> 4 B): must FAIL to
    compile.
  * aligned constant UB dst offset (fp32 offset 8 -> 32 B): compiles and runs.

NOTE: the misaligned case asserts a compile-time failure; it never reaches the
device.
"""

VEC_MIN_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.cache.clear_cache()
    yield


def _ub_dst_offset_kernel(dst_off, N=8, dtype="float"):
    """Copy X[0:N] into win[dst_off : dst_off+N] (UB), then store the window
    back.  ``dst_off`` is a compile-time constant column offset into the UB
    buffer."""

    @T.prim_func
    def main(X: T.Tensor((N,), dtype), Y: T.Tensor((2 * N,), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            win = T.alloc_ub((2 * N,), dtype)
            T.copy(X[0:N], win[dst_off : dst_off + N])
            T.copy(win, Y[0 : 2 * N])

    return main


def test_misaligned_const_ub_offset_compile_error():
    """A compile-time-constant UB dst offset of 1 fp32 element (4 B, not a 32 B
    multiple) must raise a compile-time error mentioning alignment."""
    func = _ub_dst_offset_kernel(dst_off=1, N=8, dtype="float")
    with pytest.raises(Exception) as exc_info:
        tilelang.compile(func, out_idx=[-1], pass_configs=VEC_MIN_CONFIGS, target="ascendc")
    msg = str(exc_info.value).lower()
    # The error must mention alignment / 32-byte so the author knows the cause.
    assert ("align" in msg) or ("32" in msg), f"error message should mention alignment, got: {exc_info.value}"


def test_aligned_const_ub_offset_ok():
    """A compile-time-constant UB dst offset of 8 fp32 elements (32 B, aligned)
    compiles and runs correctly."""
    N = 8
    func = _ub_dst_offset_kernel(dst_off=8, N=N, dtype="float")
    func = tilelang.compile(func, out_idx=[-1], pass_configs=VEC_MIN_CONFIGS, target="ascendc")
    torch.manual_seed(0)
    x = torch.randn(N, dtype=torch.float32).npu()
    torch.npu.synchronize()
    y = func(x).cpu()
    x = x.cpu()
    # The window [8:16) holds X; the rest is whatever the UB held (we only check
    # the copied region).
    torch.testing.assert_close(y[8:16], x, rtol=0, atol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
