# Copyright (c) Huawei Technologies Co., Ltd. 2025.
"""Minimal test to verify cardless codegen_lower infrastructure (Phase 1.3)."""
import tilelang.language as T

from testcommon import codegen_lower


def _simple_add_kernel():
    """Minimal PrimFunc: copy + add for codegen-only test."""

    @T.prim_func
    def main(
        A: T.Tensor((256, 256), "float16"),
        B: T.Tensor((256, 256), "float16"),
        C: T.Tensor((256, 256), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a = T.alloc_shared((256, 256), "float16")
            b = T.alloc_shared((256, 256), "float16")
            T.copy(A, a)
            T.npuir_add(a, a, b)
            T.copy(b, C)

    return main


def test_codegen_lower_returns_non_empty_mlir():
    """Verify codegen_lower runs without NPU and returns non-empty MLIR."""
    func = _simple_add_kernel()
    mlir = codegen_lower(func, mode="Developer")
    assert isinstance(mlir, str)
    assert len(mlir) > 0
    assert "module" in mlir or "func" in mlir or "tl." in mlir


if __name__ == "__main__":
    test_codegen_lower_returns_non_empty_mlir()
    print("codegen_lower infrastructure OK")
