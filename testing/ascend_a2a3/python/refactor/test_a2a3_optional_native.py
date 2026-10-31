"""A2/A3 dialect imports with native support disabled, without fake type keys."""

import pytest
from tilelang import tvm
import tilelang.ascend.a2a3.language as T
from tilelang.ascend.a2a3 import _native
from tilelang.ascend.a2a3.layout import AscendLayout, Layout, ceil_div, round_up, layout_str
from tilelang.ascend.a2a3.language.warpgroup import ScopeFrame


def test_dialect_pure_helpers_do_not_require_native_constructors():
    assert T.Layout is Layout
    assert ceil_div(17, 16) == 2 and round_up(17, 16) == 32
    assert [tag.value for tag in AscendLayout] == [0, 1, 2, 3, 4]
    assert callable(T.copy) and callable(T.tile.add) and callable(T.Kernel)


@pytest.mark.parametrize(
    "operation,name",
    [
        (lambda: T.Scope("C"), "tl.A2A3Scope"),
        (lambda: T.Kernel(1, is_npu=True), "tl.A2A3KernelLaunch"),
        (lambda: Layout((2,), lambda i: [i]), "tl.ascend.a2a3.WrapLayout"),
    ],
)
def test_missing_constructor_has_an_actionable_build_diagnostic(monkeypatch, operation, name):
    original = _native.ffi.get_global_func
    monkeypatch.setattr(
        _native.ffi, "get_global_func", lambda key, allow_missing=False: None if key == name else original(key, allow_missing=allow_missing)
    )
    with pytest.raises(RuntimeError, match="USE_ASCEND=ON"):
        operation()


def test_missing_constructor_does_not_register_an_object_key():
    key = "tl.A2A3OptionalNativeUnregisteredTest"

    class Placeholder:
        pass

    assert _native.register_optional_object(key, "tl.A2A3MissingConstructorTest")(Placeholder) is Placeholder
    import tvm_ffi

    assert tvm_ffi.core._object_type_key_to_index(key) is None


def test_available_constructor_does_not_hide_an_invalid_native_type_key():
    if tvm.ffi.get_global_func("tl.A2A3Scope", allow_missing=True) is None:
        pytest.skip("ON registration contract requires native A2/A3")

    class InvalidType:
        pass

    with pytest.raises(ValueError, match="Cannot find object type index"):
        _native.register_optional_object("tl.A2A3InvalidNativeTypeTest", "tl.A2A3Scope")(InvalidType)


def test_real_on_scope_layout_and_launch_bind_the_original_classes():
    if tvm.ffi.get_global_func("tl.A2A3Scope", allow_missing=True) is None:
        pytest.skip("requires native A2/A3 constructors")
    assert isinstance(T.Scope("C"), ScopeFrame)
    layout = Layout((2,), lambda i: [i])
    assert isinstance(layout, Layout) and layout_str(layout) == "layout::RowMajor"

    @T.prim_func
    def kernel(A: T.Tensor((2,), "int32")):
        with T.Kernel(1, is_npu=True) as (cid, vid), T.Scope("V"):
            A[cid + vid] = 1

    assert kernel.attrs["global_symbol"] == "kernel"


def test_real_off_import_keeps_native_type_keys_absent():
    if tvm.ffi.get_global_func("tl.A2A3Scope", allow_missing=True) is not None:
        pytest.skip("OFF-only absence contract")
    import tvm_ffi

    assert tvm_ffi.core._object_type_key_to_index("tl.A2A3ScopeFrame") is None
    assert tvm_ffi.core._object_type_key_to_index("tl.A2A3Layout") is None
    for operation in [lambda: T.Scope("V"), lambda: T.Kernel(1, is_npu=True), lambda: Layout((2,), lambda i: [i])]:
        with pytest.raises(RuntimeError, match="USE_ASCEND=ON"):
            operation()
