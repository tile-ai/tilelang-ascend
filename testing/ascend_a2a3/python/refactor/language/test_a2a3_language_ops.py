"""Hardware-independent TA language contracts on the eager frontend."""

import warnings

import pytest

from tilelang import tvm
import tilelang.ascend.a2a3.language as T
from tilelang.ascend.a2a3.language import copy_op, customize


def _buffer(name, shape=(8, 64), dtype="float32", scope="shared.ub"):
    return tvm.tirx.decl_buffer(shape, dtype, name=name, scope=scope)


def _buffers(dtype="float32"):
    return {
        "dst": _buffer("dst", dtype=dtype),
        "src": _buffer("src", dtype=dtype),
        "other": _buffer("other", dtype=dtype),
        "flat": _buffer("flat", (512,), dtype),
        "wide": _buffer("wide", (2048,), dtype),
        "rows": _buffer("rows", (8,), dtype),
        "packed_rows": _buffer("packed_rows", (8, 8), dtype),
        "columns": _buffer("columns", (64,), dtype),
        "mask": _buffer("mask", (512,), "uint8"),
        "offsets": _buffer("offsets", (512,), "uint32"),
        "arena": _buffer("arena", (8192,), "uint8"),
        "gm": _buffer("gm", scope="global", dtype=dtype),
        "l1": _buffer("l1", scope="shared.l1", dtype=dtype),
        "l0a": _buffer("l0a", scope="wmma.matrix_a", dtype=dtype),
        "l0b": _buffer("l0b", (64, 64), dtype, "wmma.matrix_b"),
        "l0c": _buffer("l0c", scope="wmma.accumulator", dtype=dtype),
    }


_UNARY = ["exp", "ln", "abs", "reciprocal", "sqrt", "rsqrt", "relu", "bitwise_not"]
_BINARY = ["add", "sub", "mul", "div", "max", "min", "bitwise_and", "bitwise_or"]
_WORKSPACE = [
    "sigmoid",
    "sin",
    "cos",
    "round",
    "pow",
    "bitwise_xor",
    "clamp_max",
    "clamp_min",
    "clamp",
    "sort",
    "merge_sort",
    "topk",
    "gather_mask",
    "select",
    "gather",
    "broadcast",
    "reduce_sum_experiment",
    "reduce_sum_mask_experiment",
]
_SPECIAL = [
    "atomic_add",
    "fill",
    "clear",
    "arith_progression",
    "gatherb",
    "init_sort_buf",
    "brcb",
    "brcb_experiment",
    "leaky_relu",
    "axpy",
    "mul_add_dst",
    "bitwise_lshift",
    "bitwise_rshift",
    "bilinear_interpolation",
    "wholereducemax",
    "wholereducemin",
    "wholereducesum",
    "sort32",
    "createvecindex",
    "create_vec_index",
    "transpose",
    "block_reduce_max",
    "block_reduce_min",
    "block_reduce_sum",
    "compare",
    "cast",
    "row_expand_mul",
    "row_expand_mul_experiment",
    "row_expand_sub_experiment",
    "row_expand_div_experiment",
    "exp_experiment",
    "sub_experiment",
    "abs_experiment",
    "mins_experiment",
    "gathermask_experiment",
    "fill_experiment",
    "sum_experiment",
    "datacachecleanandinvalid_experiment",
]


def _tile_call(name, b, explicit_tmp=False):
    dst, src, other = b["dst"], b["src"], b["other"]
    fn = getattr(T.tile, name)
    if name in _UNARY:
        return fn(dst, src)
    if name in _BINARY:
        return fn(dst, src, other)
    kw = {"tmp": b["arena"]} if explicit_tmp else {}
    if name in ("sigmoid", "sin", "cos"):
        return fn(dst, src, **kw)
    if name in ("pow", "bitwise_xor"):
        return fn(dst, src, other, **kw)
    if name in ("clamp_min", "clamp_max"):
        return fn(dst, src, 1.0, 512, **kw)
    if name == "clamp":
        return fn(dst, src, -1.0, 1.0, 512, **kw)
    if name == "round":
        return fn(dst, src, 512, **kw)
    if name == "sort":
        return fn(b["wide"], b["flat"], 512, **kw)
    if name == "merge_sort":
        return fn(b["wide"], b["flat"], b["flat"], **kw)
    if name == "topk":
        return fn(b["wide"], b["flat"], 16, 512, **kw)
    if name == "gather_mask":
        return fn(dst, src, "P0101", **kw)
    if name == "select":
        return fn(dst, b["mask"], src, 1.0, "VSEL_TENSOR_SCALAR_MODE", **kw)
    if name == "gather":
        return fn(dst, src, b["offsets"], 0, **kw)
    if name == "broadcast":
        return fn(dst, b["rows"], axis=1, **kw)
    if name == "reduce_sum_experiment":
        return fn(dst, src, 512, **kw)
    if name == "reduce_sum_mask_experiment":
        return fn(dst, src, 64, 1, 1, **kw)
    args = {
        "atomic_add": (b["gm"], src),
        "fill": (dst, 1.0),
        "clear": (dst,),
        "arith_progression": (dst, 0, 1, 512),
        "gatherb": (dst, src, b["offsets"], 1, 1, 8),
        "init_sort_buf": (dst, 512, 0),
        "brcb": (dst, src, 1, 1, 8),
        "brcb_experiment": (dst, src, 1, 1, 8),
        "leaky_relu": (dst, src, 0.1),
        "axpy": (dst, src, 0.1),
        "mul_add_dst": (dst, src, other),
        "bitwise_lshift": (dst, src, 1),
        "bitwise_rshift": (dst, src, 1),
        "bilinear_interpolation": (dst, src, b["offsets"], other, 64, 1, False, 1, 0, 1),
        "wholereducemax": (dst, src, 64, 1, 1, 1, 8),
        "wholereducemin": (dst, src, 64, 1, 1, 1, 8),
        "wholereducesum": (dst, src, 64, 1, 1, 1, 8),
        "sort32": (dst, src, other),
        "createvecindex": (dst, 0),
        "create_vec_index": (dst, 0),
        "transpose": (dst, src),
        "block_reduce_max": (dst, src, 1, 64, 1, 1, 8),
        "block_reduce_min": (dst, src, 1, 64, 1, 1, 8),
        "block_reduce_sum": (dst, src, 1, 64, 1, 1, 8),
        "compare": (b["mask"], src, other, "LT"),
        "cast": (dst, src, "CAST_NONE", 512),
        "row_expand_mul": (dst, src, b["rows"]),
        "row_expand_mul_experiment": (dst, src, b["packed_rows"]),
        "row_expand_sub_experiment": (dst, src, b["packed_rows"]),
        "row_expand_div_experiment": (dst, src, b["packed_rows"]),
        "exp_experiment": (dst, src),
        "sub_experiment": (dst, src, other, 512),
        "abs_experiment": (dst, src, 512),
        "mins_experiment": (dst, src, 1.0, 512),
        "gathermask_experiment": (dst, src, b["offsets"], False, 64, [1, 1, 8, 8], 0),
        "fill_experiment": (dst, 1.0, [64, 0], 1, 1, 8),
        "sum_experiment": (dst, src, [1, 8, 8, 1]),
        "datacachecleanandinvalid_experiment": (dst, "SINGLE_CACHE_LINE", "CACHELINE_OUT"),
    }
    return fn(*args[name])


@pytest.mark.parametrize("name", _UNARY + _BINARY + _WORKSPACE + _SPECIAL)
def test_public_tile_api_builds_registered_call(name):
    def emit():
        buffers = {key: T.alloc_fragment(buffer.shape, str(buffer.dtype), scope=buffer.scope()) for key, buffer in _buffers().items()}
        return _tile_call(name, buffers)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        call = _tile_call(name, _buffers())

        @T.prim_func
        def main():
            with T.Kernel(1, is_npu=True):
                emit()

    assert isinstance(call, tvm.tirx.Call)
    assert call.dtype == "handle"
    assert str(call.op.name).startswith(("tl.ascend_", "tirx.call_extern"))
    traced = []
    tvm.tirx.stmt_functor.post_order_visit(main.body, lambda n: traced.append(n) if isinstance(n, tvm.tirx.Call) else None)
    assert any(node.op.same_as(call.op) for node in traced)


@pytest.mark.parametrize("name", _WORKSPACE)
def test_explicit_tmp_preserves_call_and_adds_arena(name):
    b = _buffers()
    implicit = _tile_call(name, b)
    explicit = _tile_call(name, b, explicit_tmp=True)
    assert implicit.op.same_as(explicit.op)
    assert len(explicit.args) == len(implicit.args) + 1
    arena_ptrs = [
        arg
        for arg in explicit.args
        if isinstance(arg, tvm.tirx.Call) and arg.op.name == "tirx.tvm_access_ptr" and arg.args[1].same_as(b["arena"].data)
    ]
    assert len(arena_ptrs) == 1
    assert int(arena_ptrs[0].args[-1]) == 3


@pytest.mark.parametrize("name", _BINARY)
@pytest.mark.parametrize("kind", ["scalar", "load", "region"])
def test_dynamic_binary_dispatch(name, kind):
    b = _buffers()
    value = 2.0 if kind == "scalar" else tvm.tirx.BufferLoad(b["other"], [0, 0]) if kind == "load" else b["other"][:, :]
    suffix = "s" if kind in ("scalar", "load") else ""
    if name in ("bitwise_and", "bitwise_or") and suffix:
        # TA advertises the scalar shape, but has no bitwise_ands/ors Op.
        with pytest.raises(AttributeError, match=f"Operator tl.ascend_{name}s is not registered"):
            getattr(T.tile, name)(b["dst"], b["src"], value)
        return
    call = getattr(T.tile, name)(b["dst"], b["src"], value)
    assert call.op.name == f"tl.ascend_{name}{suffix}"


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "int32", "int16", "uint8"])
def test_dtype_templates_and_eager_macro(dtype):
    @T.macro
    def emit(dst, src):
        T.tile.add(dst, src, 1.0)
        T.tile.fill(dst, 0)

    @T.prim_func
    def main(A: T.Tensor((8, 64), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            dst = T.alloc_ub((8, 64), dtype)
            with T.Scope("V"):
                T.copy(A, dst)
                emit(dst, dst)
                T.copy(dst, A)

    calls = []
    tvm.tirx.stmt_functor.post_order_visit(main.body, lambda n: calls.append(n) if isinstance(n, tvm.tirx.Call) else None)
    assert sum(call.op.name == "tl.ascend_copy" for call in calls) == 2
    assert any(call.op.name == "tl.ascend_adds" for call in calls)


def test_sync_uses_shared_two_argument_abi():
    assert [arg.value for arg in T.set_flag("mte2", "mte1", 3).args] == ["MTE2_MTE1", 3]
    assert [arg.value for arg in T.wait_flag("mte2", "mte1", 3).args] == ["MTE2_MTE1", 3]
    assert T.pipe_barrier("mte3").args[0].value == "PIPE_MTE3"
    assert T.pipe_barrier("PIPE_V").args[0].value == "PIPE_V"
    assert T.barrier_all().args[0].value == "PIPE_ALL"


def test_exports_and_dialects_are_independent():
    import tilelang.ascend.language as A5
    import tilelang.cuda.language as CUDA

    assert T.copy is copy_op.npu_copy_v2
    assert T.mma is customize.npu_gemm
    assert T._src_code("// body").op.name == "tl.ascend_src_code"
    assert T.copy is not A5.copy and T.copy is not CUDA.copy
    assert T.set_flag is not A5.ascend_set_flag
    assert len(A5.ascend_set_flag("MTE2_MTE1", 0).args) == 2


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("matmul", (256, 256, 128, 128, 128, 64)),
        ("matmul_add", (256, 256, 128, 128, 128, 64)),
        ("indexer", (2, 1, 32, 512, 4096, 64, 1024, 512, 4, 128, 256, 64)),
        ("indexer_dynamic", (1, 32, 64, 1024, 512, 4, 128, 256, 64)),
    ],
)
def test_real_ta_kernel_bodies_construct_ir(name, args):
    from testing.ascend_a2a3.python.refactor.language import _a2a3_source_kernels

    func = getattr(_a2a3_source_kernels, name)(*args)
    assert isinstance(func, tvm.tirx.PrimFunc)
    calls = []
    tvm.tirx.stmt_functor.post_order_visit(func.body, lambda n: calls.append(n) if isinstance(n, tvm.tirx.Call) else None)
    assert any(call.op.name == "tl.ascend_copy" for call in calls)
    assert any(call.op.name == "tl.ascend_gemm_v0" for call in calls)
    if name.startswith("indexer"):
        assert any(call.op.name == "tl.ascend_topk" for call in calls)
        assert any(call.op.name == "tl.ascend_reduce" for call in calls)


@pytest.mark.parametrize("ways", [2, 3, 4])
def test_merge_sort_all_source_counts(ways):
    b = _buffers()
    call = T.tile.merge_sort(b["wide"], *[b["flat"]] * ways)
    assert int(call.args[1]) == ways
    assert len(call.args) == 3 + ways * 2


@pytest.mark.parametrize("mode", ["VSEL_CMPMASK_SPR", "VSEL_TENSOR_SCALAR_MODE", "VSEL_TENSOR_TENSOR_MODE"])
def test_select_original_operand_modes(mode):
    b = _buffers()
    src1 = 1.0 if mode == "VSEL_TENSOR_SCALAR_MODE" else tvm.tirx.BufferLoad(b["src"], [0, 0]) if mode == "VSEL_CMPMASK_SPR" else b["other"]
    call = T.tile.select(b["dst"], b["mask"], b["src"], src1, mode)
    assert call.op.name == "tl.ascend_select"
    assert int(call.args[3]) == {"VSEL_CMPMASK_SPR": 0, "VSEL_TENSOR_SCALAR_MODE": 1, "VSEL_TENSOR_TENSOR_MODE": 2}[mode]


def test_gather_custom_mask_and_row_expand_tmp():
    b = _buffers()
    call = T.tile.gather_mask(b["dst"], b["src"], b["offsets"])
    assert call.args[3].args[1].same_as(b["offsets"].data)
    for name in ["row_expand_mul_experiment", "row_expand_sub_experiment", "row_expand_div_experiment"]:
        call = getattr(T.tile, name)(b["dst"], b["src"], b["rows"], tmp=_buffer("row_tmp", (8, 8)))
        assert call.op.name == "tl.ascend_" + name


@pytest.mark.parametrize("axis", [0, 1, None])
def test_broadcast_axis_inference(axis):
    b = _buffers()
    src = b["columns"] if axis in (0, None) else b["rows"]
    call = T.tile.broadcast(b["dst"], src, axis=axis)
    assert call.op.name == "tl.ascend_broadcast"
    assert call.args[0].value == f"Broadcast<float, 2, {1 if axis == 1 else 0}, false>"


@pytest.mark.parametrize(
    "format_string,string_argument",
    [
        ("line1\nline2\n", None),
        ("one\ttwo\n", None),
        (r"C:\tmp\file\n", None),
        ("s=%s\n", "hello\tworld\\tail"),
        ('quote="\n', None),
        ("é🙂\n", None),
    ],
)
def test_printf_preserves_ta_preescaped_strings(format_string, string_argument):
    arguments = [] if string_argument is None else [string_argument]
    call = T.printf(format_string, *arguments)
    assert call.op.name == "tl.ascend_printf"
    assert call.args[0].value == format_string.encode("unicode_escape").decode("utf-8")
    if string_argument is not None:
        assert call.args[1].value == string_argument.encode("unicode_escape").decode("utf-8")


def test_printf_preserves_pointer_format_and_buffer_access():
    buffer = _buffer("printf_pointer", scope="global")
    call = T.printf("p=%p\n", buffer)
    assert call.args[0].value == r"p=0x%x\n"
    assert call.args[1].op.name == "tirx.tvm_access_ptr"
    assert call.args[1].args[1].same_as(buffer.data)


@pytest.mark.parametrize("dialect,prefix", [("ascendc", "AscendC::PRINTF"), ("pto", "cce::printf")])
@pytest.mark.parametrize(
    "format_string,c_literal",
    [("one\ttwo\\tail\n", '"one\\ttwo\\\\tail\\n"'), ('quote="\n', '"quote="\\n"'), ("é🙂\n", '"\\xe9\\U0001f642\\n"')],
)
def test_printf_full_pipeline_preserves_ta_string_literals(dialect, prefix, format_string, c_literal):
    import tilelang

    @T.prim_func
    def main(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub = T.alloc_ub((32,), "int32")
            with T.Scope("V"):
                T.printf(format_string)
                T.copy(A, ub)
                T.copy(ub, B)

    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": [dialect, "ascend"]})
    source = tilelang.lower(main, target=target).kernel_source
    assert prefix + "(" + c_literal + ");" in source
