import pytest

import tilelang
import tilelang.language as T


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Clear tilelang cache before tests"""
    tilelang.disable_cache()


def test_src_code_basic_injection():
    """Test that T._src_code() injects a single line of code into generated source."""

    @T.prim_func
    def main(
        A: T.Tensor((256,), "float16"),
        B: T.Tensor((256,), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((128,), dtype="float16")
            T.copy(A[cid * 128], a_ub)
            T._src_code("// Injected by _src_code test")
            T.copy(a_ub, B[cid * 128])

    mod = tilelang.lower(main, target="ascendc")
    source = mod.kernel_source

    assert "// Injected by _src_code test" in source, f"Source code did not contain injected string.\nSource:\n{source}"


def test_src_code_multiline():
    """Test that T._src_code() handles multi-line source code."""

    @T.prim_func
    def main(
        A: T.Tensor((256,), "float16"),
        B: T.Tensor((256,), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((128,), dtype="float16")
            T.copy(A[cid * 128], a_ub)
            T._src_code("int var1 = 0;\nint var2 = 1;\n// end of block")
            T.copy(a_ub, B[cid * 128])

    mod = tilelang.lower(main, target="ascendc")
    source = mod.kernel_source

    assert "int var1 = 0;" in source
    assert "int var2 = 1;" in source
    assert "// end of block" in source


def test_src_code_empty_string():
    """Test that T._src_code() with empty string produces valid output."""

    @T.prim_func
    def main(
        A: T.Tensor((256,), "float16"),
        B: T.Tensor((256,), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((128,), dtype="float16")
            T.copy(A[cid * 128], a_ub)
            T._src_code("")
            T.copy(a_ub, B[cid * 128])

    mod = tilelang.lower(main, target="ascendc")
    source = mod.kernel_source

    assert "main_kernel" in source.lower() or "main" in source.lower()


def test_src_code_multiple_injections():
    """Test that multiple T._src_code() calls all appear in source."""

    @T.prim_func
    def main(
        A: T.Tensor((256,), "float16"),
        B: T.Tensor((256,), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((128,), dtype="float16")
            T._src_code("// MARKER_A")
            T.copy(A[cid * 128], a_ub)
            T._src_code("// MARKER_B")
            T.copy(a_ub, B[cid * 128])
            T._src_code("// MARKER_C")

    mod = tilelang.lower(main, target="ascendc")
    source = mod.kernel_source

    assert "// MARKER_A" in source
    assert "// MARKER_B" in source
    assert "// MARKER_C" in source
    idx_a = source.index("// MARKER_A")
    idx_b = source.index("// MARKER_B")
    idx_c = source.index("// MARKER_C")
    assert idx_a < idx_b < idx_c, f"Expected MARKER order A<B<C, got A={idx_a} B={idx_b} C={idx_c}"


def test_src_code_pto_backend():
    """Test that T._src_code() works with PTO backend."""

    @T.prim_func
    def main(
        A: T.Tensor((256,), "float16"),
        B: T.Tensor((256,), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((128,), dtype="float16")
            T.copy(A[cid * 128], a_ub)
            T._src_code("// PTO_injected_marker_12345")
            T.copy(a_ub, B[cid * 128])

    mod = tilelang.lower(main, target="pto")
    source = mod.kernel_source

    assert "// PTO_injected_marker_12345" in source, f"PTO source did not contain injected string.\nSource:\n{source}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
