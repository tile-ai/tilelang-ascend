"""Independent local typed-view regressions, with explicit synchronization."""

import numpy as np
import pytest
import tilelang
import tilelang.language as T

pytestmark = [pytest.mark.op("view"), pytest.mark.mode("Expert")]
CASES = (
    "cast",
    "cast_small",
    "bits_to_half",
    "bits_to_float",
    "same_dtype",
    "scalar_access",
    "offset",
    "oversize",
    "strided",
    "same_reshape",
    "same_offset",
    "same_oversize",
    "same_strided",
)
ERRORS = {
    "same_oversize": "exceeds allocation capacity",
    "same_strided": "Noncontiguous typed local views are not supported",
    "oversize": "exceeds allocation capacity",
    "strided": "Noncontiguous typed local views are not supported",
}


@T.macro
def wait_for(src, dst):
    with T.rs(src):
        T.set_flag(dst, 0)
    with T.rs(dst):
        T.wait_flag(src, 0)


def alias_kernel(case):
    if case in ("same_reshape", "same_offset", "same_oversize", "same_strided"):
        shape = (
            (16, 32)
            if case == "same_reshape"
            else ((8, 65) if case == "same_oversize" else (4, 64))
        )
        offset = 64 if case == "same_offset" else 0
        strides = (128, 1) if case == "same_strided" else ()

        @T.prim_func
        def same_view(A: T.Tensor((8, 64), "float32"), B: T.Tensor(shape, "float32")):
            with T.Kernel(1, is_npu=True):
                a = T.alloc_ub((8, 64), "float32")
                b = T.Buffer(
                    shape,
                    "float32",
                    data=a.data,
                    elem_offset=offset,
                    strides=strides,
                    scope="shared",
                )
                T.copy(A, a)
                if case == "same_offset":
                    wait_for("PIPE_MTE2", "PIPE_S")
                    b[0, 0] = b[0, 1]
                    wait_for("PIPE_S", "PIPE_MTE3")
                else:
                    wait_for("PIPE_MTE2", "PIPE_MTE3")
                T.copy(b, B)
                wait_for("PIPE_MTE3", "PIPE_MTE2")

        return same_view
    rows, cols = (64, 128) if case == "cast" else (8, 64)
    src_dtype = "float16" if case == "bits_to_float" else "float32"
    dst_dtype = "float32" if case in ("bits_to_float", "same_dtype") else "float16"
    src_cols = cols * 2 if case == "bits_to_float" else cols
    dst_cols = cols * 2 if case == "bits_to_half" else cols
    if case == "oversize":
        dst_cols = cols * 2 + 1
    if case in ("offset", "strided"):
        offset = 16 if case == "offset" else 0
        strides = (cols, 1) if case == "offset" else (cols * 2, 1)

        @T.prim_func
        def special(
            A: T.Tensor((rows, cols), "float32"), B: T.Tensor((rows, cols), "float16")
        ):
            with T.Kernel(1, is_npu=True):
                a = T.alloc_ub((rows, cols), "float32")
                b = T.Buffer(
                    (rows, cols),
                    "float16",
                    data=a.data,
                    strides=strides,
                    elem_offset=offset,
                    scope="shared",
                )
                T.copy(A, a)
                wait_for("PIPE_MTE2", "PIPE_V")
                wait_for("PIPE_V", "PIPE_MTE3")
                T.copy(b, B)
                wait_for("PIPE_MTE3", "PIPE_V")

        return special

    @T.prim_func
    def main(
        A: T.Tensor((rows, src_cols), src_dtype),
        B: T.Tensor((rows, dst_cols), dst_dtype),
    ):
        with T.Kernel(1, is_npu=True):
            a = T.alloc_ub((rows, src_cols), src_dtype)
            b = T.view(a, (rows, dst_cols), dtype=dst_dtype)
            T.copy(A, a)
            if case == "scalar_access":
                wait_for("PIPE_MTE2", "PIPE_S")
                a[0, 0] = a[0, 1]
                wait_for("PIPE_S", "PIPE_MTE3")
            else:
                wait_for("PIPE_MTE2", "PIPE_V")
                if case == "cast" or case == "cast_small":
                    T.vcast(a, b, round_mode="round")
                wait_for("PIPE_V", "PIPE_MTE3")
            T.copy(b, B)
            if case == "scalar_access":
                wait_for("PIPE_MTE3", "PIPE_S")
            else:
                wait_for("PIPE_MTE3", "PIPE_V")

    return main


def inputs_and_reference(case):
    rows, cols = (64, 128) if case == "cast" else (8, 64)
    a = np.linspace(-30, 30, rows * cols, dtype=np.float32).reshape(rows, cols)
    if case == "same_reshape":
        return a, a.reshape(16, 32).copy()
    if case == "same_offset":
        ref = a.ravel()[64:320].reshape(4, 64).copy()
        ref[0, 0] = ref[0, 1]
        return a, ref
    if case == "bits_to_float":
        return a.view("float16").copy(), a.copy()
    if case == "bits_to_half":
        return a, a.view("float16").copy()
    if case == "same_dtype":
        return a, a.copy()
    if case == "scalar_access":
        changed = a.copy()
        changed[0, 0] = changed[0, 1]
        ref = changed.view("float16").ravel()[: rows * cols].reshape(rows, cols)
        return a, ref.copy()
    if case == "offset":
        ref = a.view("float16").ravel()[16 : 16 + rows * cols].reshape(rows, cols)
        return a, ref.copy()
    return a, a.astype("float16")


def _check(case):
    import torch
    import torch_npu  # noqa: F401

    flags = {
        "npuir.enable_auto_multi_buffer": False,
        "npuir.disable_hivm_auto_inject_sync": True,
        "tl.enable_plan_and_update_buffer_allocation": False,
    }
    if case in ERRORS:
        with pytest.raises(Exception, match=ERRORS[case]):
            tilelang.compile(alias_kernel(case), target="npuir", pass_configs=flags)
        return
    fn = tilelang.compile(alias_kernel(case), target="npuir", pass_configs=flags)
    a, expected = inputs_and_reference(case)
    da = torch.from_numpy(a).npu()
    db = torch.empty(
        expected.shape, dtype=getattr(torch, str(expected.dtype)), device="npu"
    )
    fn(da, db)
    torch.npu.synchronize()
    actual = db.cpu().numpy()
    # Reinterpret cases can contain NaN payloads: compare bytes, not float equality.
    assert actual.tobytes() == expected.tobytes()


def test_cast():
    _check("cast")


def test_cast_small():
    _check("cast_small")


def test_bits_to_half():
    _check("bits_to_half")


def test_bits_to_float():
    _check("bits_to_float")


def test_same_dtype():
    _check("same_dtype")


def test_scalar_access():
    _check("scalar_access")


def test_offset():
    _check("offset")


def test_oversize():
    _check("oversize")


def test_strided():
    _check("strided")


@pytest.mark.parametrize(
    "case", ["same_reshape", "same_offset", "same_oversize", "same_strided"]
)
def test_same_dtype_view_metadata(case):
    _check(case)


def grouped_reduce_probe():
    @T.prim_func
    def main(A: T.Tensor((8, 1024), "float32"), B: T.Tensor((1024, 1), "float32")):
        with T.Kernel(1, is_npu=True):
            a = T.alloc_ub((8, 1024), "float32")
            grouped = T.view(a, (1024, 8), dtype="float32")
            b = T.alloc_ub((1024, 1), "float32")
            T.copy(A, a)
            wait_for("PIPE_MTE2", "PIPE_V")
            T.reduce_max(grouped, b, dim=1)
            wait_for("PIPE_V", "PIPE_MTE3")
            T.copy(b, B)
            wait_for("PIPE_MTE3", "PIPE_V")

    return main


def test_grouped_block_max():
    import torch
    import torch_npu  # noqa: F401

    flags = {
        "npuir.enable_auto_multi_buffer": False,
        "npuir.disable_hivm_auto_inject_sync": True,
        "tl.enable_plan_and_update_buffer_allocation": False,
    }
    fn = tilelang.compile(grouped_reduce_probe(), target="npuir", pass_configs=flags)
    a = np.random.default_rng(20261027).normal(size=(8, 1024)).astype("float32")
    da = torch.from_numpy(a).npu()
    db = torch.full((1024, 1), float("nan"), device="npu", dtype=torch.float32)
    fn(da, db)
    torch.npu.synchronize()
    np.testing.assert_array_equal(
        db.cpu().numpy(), a.reshape(1024, 8).max(axis=1, keepdims=True)
    )
