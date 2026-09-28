"""Regression for coalesced 32-byte-row GM-to-L1 copies."""

import pytest
import torch

import tilelang
import tilelang.language as T

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


def _copy_band_through_identity_mma(rows=64, cols=16):
    @tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS, target="ascendc")
    def kernel():
        @T.prim_func
        def main(
            source: T.Tensor((rows, cols), "float16"),
            identity: T.Tensor((cols, cols), "float16"),
            output: T.Tensor((rows, cols), "float32"),
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                source_l1 = T.alloc_L1((rows, cols), "float16")
                identity_l1 = T.alloc_L1((cols, cols), "float16")
                source_l0 = T.alloc_L0A((rows, cols), "float16")
                identity_l0 = T.alloc_L0B((cols, cols), "float16")
                output_l0 = T.alloc_L0C((rows, cols), "float32")
                with T.Scope("C"):
                    T.copy(source, source_l1)
                    T.copy(identity, identity_l1)
                    T.copy(source_l1, source_l0)
                    T.copy(identity_l1, identity_l0)
                    T.mma(source_l0, identity_l0, output_l0, init=True)
                    T.copy(output_l0, output)

        return main

    return kernel()


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="coalesced GM-to-L1 correctness requires an Ascend NPU runtime",
)
def test_coalesced_gm_to_l1_band_is_bit_complete():
    torch.manual_seed(0)
    rows, cols = 64, 16
    source = torch.randn(rows, cols, dtype=torch.float16)
    identity = torch.eye(cols, dtype=torch.float16)
    output = _copy_band_through_identity_mma(rows, cols)(source.npu(), identity.npu()).cpu()
    torch.testing.assert_close(output, source.float(), rtol=1e-3, atol=1e-3)
