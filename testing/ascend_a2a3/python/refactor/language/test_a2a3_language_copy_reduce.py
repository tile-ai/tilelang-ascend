"""Original TA copy/reduce forms, pointer contracts and top-level primitives."""

import pytest

from tilelang import tvm
import tilelang.ascend.a2a3.language as T
from tilelang.ascend.a2a3.language import ascend, copy_op, customize


def _buffer(name, shape=(8, 64), dtype="float32", scope="shared.ub", **kwargs):
    return tvm.tirx.decl_buffer(shape, dtype, name=name, scope=scope, **kwargs)


@pytest.mark.parametrize("form", ["buffer", "load", "region"])
@pytest.mark.parametrize("options", [{}, {"unit_flag": 3}, {"real_k": 32}, {"real_n": 16}])
def test_copy_regions_and_positional_runtime_options(form, options):
    src, dst = _buffer("src"), _buffer("dst")
    source = src if form == "buffer" else tvm.tirx.BufferLoad(src, [0, 0]) if form == "load" else src[:, :]
    call = T.copy(source, dst, enable_relu=True, pad_value=-1.0, **options)
    assert call.op.name == "tl.ascend_copy"
    assert len(call.args) == (9 if "real_n" in options else 8 if "real_k" in options else 7 if options else 6)
    assert [arg.op.name for arg in call.args[:2]] == ["tl.region", "tl.region"]
    assert int(call.args[2]) == 1
    assert float(call.args[4]) == -1.0
    if "unit_flag" in options:
        assert int(call.args[6]) == 3
    if "real_k" in options:
        assert [int(arg) for arg in call.args[6:]] == [0, 32]
    if "real_n" in options:
        assert [int(arg) for arg in call.args[6:]] == [0, 0, 16]


def test_copy_transpose_tmp_and_symbolic_pad():
    src = _buffer("src", (8, 64))
    dst = _buffer("dst", (64, 8))
    tmp = _buffer("tmp")
    pad = tvm.tirx.Var("pad", "float32")
    call = T.copy(src, dst, transpose=True, pad_value=pad, tmp=tmp)
    assert int(call.args[3]) == 1
    assert call.args[4].same_as(pad)
    assert call.args[5].op.name == "tl.region"


def test_copy_cross_cv_shapes_accept_one_ratio_dimension():
    src = _buffer("src", (8, 64), scope="shared.ub")
    dst = _buffer("dst", (16, 64), scope="shared.l1")
    assert T.copy(src, dst).op.name == "tl.ascend_copy"
    with pytest.raises(ValueError, match="More than one dimension differs"):
        T.copy(src, _buffer("bad", (16, 128), scope="shared.l1"))
    with pytest.raises(ValueError, match="not 2x ratio"):
        T.copy(src, _buffer("bad", (24, 64), scope="shared.l1"))


def test_ta_ramp_region_conversion_keeps_spanned_extent():
    buf = _buffer("src", (64,))
    ramp = tvm.tirx.Ramp(4, 2, 8)
    load = tvm.tirx.BufferLoad(buf, [ramp])
    region = T.tile._buffer_load_to_buffer_region(load)
    # TA represents the full span, including the gaps in a non-unit ramp.
    assert int(region.region[0].min) == 4
    assert int(region.region[0].extent) == 15
    assert region.buffer.same_as(buf)


@pytest.mark.parametrize("dtype", ["uint8", "float16", "float32", "int32"])
def test_tmp_arena_uses_bytes_for_alignment(dtype):
    arena = _buffer("arena", (128,), dtype)
    size = tvm.DataType(dtype).itemsize
    start = 32 // size
    ptr = ascend._get_tmp_arena_access_ptr("test", arena[start : start + 16])
    assert ptr.op.name == "tirx.tvm_access_ptr"
    assert [int(arg) for arg in ptr.args[2:]] == [start, 16, 3]
    with pytest.raises(ValueError, match="32-byte aligned"):
        ascend._get_tmp_arena_access_ptr("test", arena[1:16])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"shape": (8, 8)}, "one-dimensional"),
        ({"scope": "local"}, "shared.ub"),
        ({"strides": [2]}, "unit stride"),
        ({"shape": (tvm.tirx.Var("n", "int32"),)}, "extent must be static"),
    ],
)
def test_tmp_arena_original_rejections(kwargs, message):
    kwargs = {"shape": (128,), **kwargs}
    with pytest.raises(ValueError, match=message):
        ascend._get_tmp_arena_access_ptr("test", _buffer("bad", **kwargs))


@pytest.mark.parametrize("name", ["reduce_sum", "reduce_min", "reduce_max"])
@pytest.mark.parametrize("dim", [0, 1, -1, -2])
def test_reduce_axis_and_clear_compatibility(name, dim):
    src = _buffer("src")
    dst = _buffer("dst", (64,) if dim in (0, -2) else (8,))
    call = getattr(T, name)(src, dst, dim, False, (8, 64))
    assert call.op.name == "tl.ascend_reduce"
    assert int(call.args[3]) == 0
    assert call.args[0].value.startswith(name + "<float, 8, 64, ")
    with pytest.raises(TypeError, match="multiple values for clear"):
        getattr(T, name)(src, dst, dim, False, clear=True)


def test_reduce_keeps_physical_row_stride_for_slices():
    src = _buffer("src", (8, 128))
    dst = _buffer("dst", (8,))
    call = T.reduce_sum(src[:, 16:80], dst)
    assert call.args[0].value == "reduce_sum<float, 8, 64, -1>"
    assert int(call.args[-1]) == 128


def test_reduce_dynamic_shape_and_explicit_workspace():
    n = tvm.tirx.Var("n", "int32")
    src = _buffer("src", (8, n))
    dst = _buffer("dst", (8,))
    arena = _buffer("arena", (1024,), "uint8")
    call = T.reduce_sum(src, dst, tmp=arena)
    assert len(call.args) == 5
    assert call.args[3].args[1].same_as(arena.data)


def test_reduce_invalid_shapes_and_axis_remain_errors():
    src = _buffer("src")
    with pytest.raises(ValueError, match="output shape"):
        T.reduce_sum(src, _buffer("bad", (7,)))
    with pytest.raises(ValueError, match="only supports axis"):
        T.reduce_sum(src, _buffer("dst", (8,)), dim=2)
    with pytest.raises(TypeError, match="dim must be an integer"):
        T.reduce_sum(src, _buffer("dst", (8,)), dim=True)


@pytest.mark.parametrize("variant", ["default", "runtime", "transposed", "leading_ones", "region"])
def test_gemm_and_mma_keep_original_optional_operands(variant):
    a = _buffer("a", (8, 64), "float16", "shared.l1")
    b = _buffer("b", (64, 64), "float16", "shared.l1")
    c = _buffer("c", (8, 64), "float32", "wmma.accumulator")
    options = {}
    if variant == "runtime":
        options = {"n_actual": tvm.tirx.Var("actual_n", "int32")}
    elif variant == "transposed":
        options = {"transpose_B": True}
    elif variant == "leading_ones":
        a = _buffer("a", (1, 8, 64), "float16", "shared.l1")
        b = _buffer("b", (1, 64, 64), "float16", "shared.l1")
        c = _buffer("c", (1, 8, 64), "float32", "wmma.accumulator")
    elif variant == "region":
        a, b, c = a[:, :], b[:, :], c[:, :]
    assert T.gemm_v0(a, b, c, init=True, **options).op.name == "tl.ascend_gemm_v0"
    mma = T.mma(a, b, c, init=True, unit_flag=3, k_actual=32)
    assert mma.op.name == "tl.ascend_mma"
    assert [int(arg) for arg in mma.args[-3:]] == [32, 64, 3]


def test_top_level_ascend_primitives():
    src, dst = _buffer("src"), _buffer("dst")
    calls = [
        T.set_cross_flag("MTE3", 1),
        T.wait_cross_flag(1),
        T.sync_all(),
        T.shmem_put_nbi(dst, src, 512, 0),
        T.shmem_get_nbi(dst, src, 512, 0),
        T.shmem_ub_put_nbi(src, dst, 512, 0),
        T.shmem_ub_get_nbi(dst, src, 512, 0),
        T.printf("ptr=%p\n", src),
        T.dump_tensor(src, 1, 512, (8, 64)),
        T.reinterpretcast(dst, src, "uint32_t"),
        T.set_deq_scale(1.0),
        T._src_code("// body", 1),
        T.loop_break(),
    ]
    assert all(isinstance(call, tvm.tirx.Call) for call in calls)
    assert calls[7].args[0].value == r"ptr=0x%x\n"
    assert int(calls[8].args[3]) == 2


@pytest.mark.parametrize("width", [None, 8])
def test_generic_copy_helper(width):
    src, dst = _buffer("src"), _buffer("dst")
    call = copy_op.copy(src, dst, width)
    assert call.op.name == "tl.tileop.copy"
    assert len(call.args) == 2
    if width is not None:
        assert int(call.annotations["coalesced_width"]) == width


def test_im2col_helper():
    src, dst = _buffer("src"), _buffer("dst")
    with pytest.raises(NotImplementedError, match="CUDA Hopper-only.*tilelang.cuda.language.im2col"):
        copy_op.c2d_im2col(src, dst, 1, 1, 3, 1, 1, 0)


def test_cross_cv_experiment_helpers():
    src, dst = _buffer("src"), _buffer("dst")
    assert copy_op.copy_cv_experiment(src, dst).op.name == "tl.ascend_copy_cv_experiment"
    for view in (dst, tvm.tirx.BufferLoad(dst, [1, 2]), dst[:, :]):
        assert copy_op.copy_vc_experiment(src, view, _buffer("tmp")).op.name == "tl.ascend_copy_vc_experiment"


def test_custom_views_and_scalar_clamp():
    src = _buffer("src")
    reshaped = customize.reshape(src, [4, 128])
    viewed = customize.view(src, shape=[4, 128])
    assert reshaped.data.same_as(src.data) and viewed.data.same_as(src.data)
    value = tvm.tirx.Var("value", "float32")
    assert isinstance(customize.clamp(value, -1.0, 1.0), tvm.tirx.Min)


def test_original_custom_external_calls():
    src, dst = _buffer("src"), _buffer("dst")
    a, b = tvm.tirx.BufferLoad(src, [0, 0]), tvm.tirx.BufferLoad(dst, [0, 0])
    calls = [T.atomic_add(a, 1.0), T.atomic_addx2(a, b), T.atomic_addx4(a, b), T.dp4a(a, b, a)]
    assert [call.args[0].value for call in calls] == ["AtomicAdd", "AtomicAddx2", "AtomicAddx4", "DP4A"]
