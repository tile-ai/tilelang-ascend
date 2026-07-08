import pytest
from unittest.mock import patch

import tilelang
import tilelang.language as T

"""
Regression tests for issue #1301: PTO codegen crashes with
"Find undefined Variable batch" when a buffer's shape is a composite
symbolic expression (e.g. ``batch + 1``) and that buffer appears before
a buffer whose shape is the bare symbolic variable.

Root cause: ``CodeGenTileLangAscendPto::AddFunction`` used a top-level
``as<VarNode>()`` check to collect shape variables, which only recognises a
shape that *is* a variable, not one that *contains* a variable inside an
arithmetic expression (AddNode/MulNode/...).  The fix recursively visits the
shape expression tree so every VarNode is registered before any shape is
printed.
"""


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.cache.clear_cache()


def _composite_shape_first():
    """``[batch + 1]`` buffer precedes the ``[batch]`` buffer.

    This is the exact ordering that triggered the original crash: ``batch``
    is never registered by a preceding bare-variable buffer, so PTO codegen
    cannot resolve it when printing the ``batch + 1`` shape."""
    batch = T.symbolic("batch")

    @T.prim_func
    def main(
        B: T.Tensor([batch + 1], "int32"),
        A: T.Tensor([batch, 16], "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            s = T.alloc_var("int32", init=0)
            for _i in T.serial(batch):
                s = s + 1

    return main


def _multi_var_composite_shape():
    """Shape expression contains two distinct symbolic vars (``batch + seq``).

    Exercises recursive collection of more than one VarNode from a single
    composite shape expression."""
    batch = T.symbolic("batch")
    seq = T.symbolic("seq")

    @T.prim_func
    def main(
        B: T.Tensor([batch + seq], "int32"),
        A: T.Tensor([batch, seq], "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            s = T.alloc_var("int32", init=0)
            for _i in T.serial(batch):
                s = s + 1

    return main


def _nested_composite_shape():
    """Shape expression is a nested arithmetic (``batch * 2 + 1``).

    Verifies the visitor descends through MulNode -> AddNode -> VarNode."""
    batch = T.symbolic("batch")

    @T.prim_func
    def main(
        B: T.Tensor([batch * 2 + 1], "int32"),
        A: T.Tensor([batch, 16], "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            s = T.alloc_var("int32", init=0)
            for _i in T.serial(batch):
                s = s + 1

    return main


def _compile_and_get_source(prim_func, target):
    with (
        patch("tilelang.jit.adapter.libgen.LibraryGenerator.compile_lib") as mock_compile,
        patch(
            "tilelang.jit.adapter.libgen.LibraryGenerator.load_lib",
            return_value=None,
        ),
    ):
        mock_compile.return_value = None
        compiled = tilelang.compile(prim_func, target=target)
    return compiled.get_kernel_source()


def _assert_shape_var_in_signature(code, *var_names):
    """Assert that each symbolic var is emitted as a kernel parameter.

    Both backends emit shape variables as ``int64_t <name>`` in the kernel
    signature, so we check the first line containing ``main_kernel``."""
    sig_lines = [l for l in code.splitlines() if "main_kernel" in l and "(" in l]
    assert sig_lines, f"kernel signature not found in generated code:\n{code}"
    sig = sig_lines[0]
    for name in var_names:
        assert f"int64_t {name}" in sig, f"symbolic var '{name}' should be a kernel parameter, signature:\n{sig}"


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_composite_shape_first_codegen(target):
    """Issue #1301 regression: ``[batch + 1]`` before ``[batch]`` must not
    crash codegen, and ``batch`` must be emitted as a kernel parameter."""
    code = _compile_and_get_source(_composite_shape_first(), target)
    _assert_shape_var_in_signature(code, "batch")


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_multi_var_composite_shape_codegen(target):
    """A composite shape (``batch + seq``) must register both vars."""
    code = _compile_and_get_source(_multi_var_composite_shape(), target)
    _assert_shape_var_in_signature(code, "batch", "seq")


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_nested_composite_shape_codegen(target):
    """A nested arithmetic shape (``batch * 2 + 1``) must still register
    ``batch``."""
    code = _compile_and_get_source(_nested_composite_shape(), target)
    _assert_shape_var_in_signature(code, "batch")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
