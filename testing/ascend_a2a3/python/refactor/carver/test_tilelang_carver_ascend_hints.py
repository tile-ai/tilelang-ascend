"""Ascend Carver search works from an offline chip profile without an NPU."""

import pytest

from tilelang import carver
from tvm.target import Target
from tilelang.carver.arch import get_arch
import tilelang.carver.arch.ascend as ascend_arch


def test_ascend_target_dispatch(monkeypatch):
    monkeypatch.setattr(ascend_arch, "_TORCH_NPU_AVAILABLE", False)
    for target in ("ascend", Target({"kind": "llvm", "keys": ["ascend"]})):
        arch = get_arch(target)
        assert isinstance(arch, ascend_arch.Ascend)
        assert arch.platform == "ascend"
        assert arch.cube_shape == [16, 16, 16]


@pytest.mark.parametrize("chip_name", ["Ascend910A", "Ascend910B"])
def test_ascend_cube_recommendations(monkeypatch, chip_name):
    monkeypatch.setattr(ascend_arch, "_TORCH_NPU_AVAILABLE", False)
    arch = ascend_arch.Ascend(chip_name=chip_name)
    hints = (
        carver.MatmulTemplate(M=1024, N=1024, K=1024, in_dtype="float16", out_dtype="float16", accum_dtype="float16")
        .with_arch(arch)
        .recommend_hints(topk=20)
    )
    assert 0 < len(hints) <= 20
    for hint in hints:
        block_m, block_n = hint.block
        reduce_k = hint.rstep[0]
        assert hint.use_tc
        assert hint.pipeline_stage == 2
        assert hint.thread == [1, 1]
        assert reduce_k % arch.cube_dim == 0
        assert 1024 % block_m == 1024 % block_n == 1024 % reduce_k == 0
        # rstep describes the L1 reduction tile, not the inner L0 tile.
        assert (block_m + block_n) * reduce_k * 2 * hint.pipeline_stage <= arch.l1_cap
        assert block_m * block_n * 2 <= arch.l0c_cap


def test_ascend_vector_fallback(monkeypatch):
    monkeypatch.setattr(ascend_arch, "_TORCH_NPU_AVAILABLE", False)
    arch = ascend_arch.Ascend(chip_name="Ascend910B")
    hints = carver.ElementwiseTemplate(shape=[128, 256], dtype="float16").with_arch(arch).recommend_hints(topk=4)
    assert hints
    assert all(not hint.use_tc and hint.thread == [1, 1] for hint in hints)
