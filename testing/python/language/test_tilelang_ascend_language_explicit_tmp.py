"""Regression tests for compiler-managed and explicit temporary arenas."""

import inspect
import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.language import ascend_tile
from tvm import tir


PTO_UNSUPPORTED_WORKSPACE_OPS = {
    "tl.ascend_bilinear_interpolation",
    "tl.ascend_cos",
    "tl.ascend_reducesum_experiment",
    "tl.ascend_reducesum_mask_experiment",
    "tl.ascend_sin",
}


def _target(model: str):
    return tvm.target.Target({"kind": "llvm", "model": model})


def _inject(program, model: str, repeat: int = 1):
    mod = tvm.IRModule.from_expr(program)
    inject = tilelang.transform.InjectTmpBuffer(_target(model))
    for _ in range(repeat):
        mod = inject(mod)
    return mod["main"]


def _collect_calls(func, op_name: str):
    calls = []
    expected_op = tir.op.Op.get(op_name)

    def visitor(node):
        if isinstance(node, tir.Call) and node.op.same_as(expected_op):
            calls.append(node)

    tir.stmt_functor.post_order_visit(func.body, visitor)
    return calls


def _allocated_buffer_names(func):
    names = []

    def visitor(node):
        if isinstance(node, tir.Block):
            names.extend(buffer.name for buffer in node.alloc_buffers)

    tir.stmt_functor.post_order_visit(func.body, visitor)
    return names


def _ub_buffer(name: str, shape, dtype: str = "float32"):
    return tir.decl_buffer(shape, dtype, name=name, scope="shared.ub")


def _workspace_api_calls(explicit: bool, arena_dtype: str = "uint8"):
    arena_elements = 4096 // tvm.DataType(arena_dtype).itemsize()
    arena = _ub_buffer("arena", (arena_elements,), arena_dtype)
    src = _ub_buffer("src", (64,))
    dst = _ub_buffer("dst", (64,))
    wide_dst = _ub_buffer("wide_dst", (128,))
    reduce_src = _ub_buffer("reduce_src", (8, 64))
    reduce_dst = _ub_buffer("reduce_dst", (8,))
    mask = _ub_buffer("mask", (64,), "uint8")
    offsets = _ub_buffer("offsets", (64,), "uint32")
    tmp_arg = {"tmp": arena} if explicit else {}

    with pytest.warns(DeprecationWarning):
        bilinear = ascend_tile.bilinear_interpolation(
            dst,
            src,
            offsets,
            src,
            64,
            1,
            False,
            1,
            0,
            1,
            **tmp_arg,
        )

    calls_and_slots = [
        (T.reduce_sum(reduce_src, reduce_dst, **tmp_arg), 3),
        (T.reduce_max(reduce_src, reduce_dst, **tmp_arg), 3),
        (T.reduce_min(reduce_src, reduce_dst, **tmp_arg), 3),
        (ascend_tile.sort(wide_dst, src, 64, **tmp_arg), 3),
        (ascend_tile.merge_sort(wide_dst, src, src, **tmp_arg), 3),
        (ascend_tile.topk(wide_dst, src, 16, 64, **tmp_arg), 3),
        (ascend_tile.gather_mask(dst, src, "P0101", **tmp_arg), 4),
        (ascend_tile.gather_mask(dst, src, offsets, **tmp_arg), 4),
        (
            ascend_tile.select(
                dst,
                mask,
                src,
                1.0,
                "VSEL_TENSOR_SCALAR_MODE",
                **tmp_arg,
            ),
            3,
        ),
        (ascend_tile.sigmoid(dst, src, **tmp_arg), 2),
        (bilinear, 10),
        (ascend_tile.gather(dst, src, offsets, 0, **tmp_arg), 5),
        (ascend_tile.sin(dst, src, **tmp_arg), 2),
        (ascend_tile.cos(dst, src, **tmp_arg), 2),
        (ascend_tile.pow(dst, src, src, **tmp_arg), 3),
        (ascend_tile.bitwise_xor(dst, src, src, **tmp_arg), 3),
        (ascend_tile.clamp_max(dst, src, 1.0, 64, **tmp_arg), 3),
        (ascend_tile.clamp_min(dst, src, -1.0, 64, **tmp_arg), 3),
        (ascend_tile.clamp(dst, src, -1.0, 1.0, 64, **tmp_arg), 3),
        (ascend_tile.round(dst, src, 64, **tmp_arg), 2),
        (ascend_tile.broadcast(reduce_src, reduce_dst, axis=1, **tmp_arg), 3),
        (ascend_tile.reduce_sum_experiment(dst, src, 64, **tmp_arg), 2),
        (
            ascend_tile.reduce_sum_mask_experiment(
                dst,
                src,
                64,
                1,
                1,
                **tmp_arg,
            ),
            2,
        ),
    ]
    buffers = [
        arena,
        src,
        dst,
        wide_dst,
        reduce_src,
        reduce_dst,
        mask,
        offsets,
    ]
    return calls_and_slots, buffers, arena


def _program_from_calls(calls_and_slots, buffers, include_arena: bool):
    calls = [tir.Evaluate(call) for call, _ in calls_and_slots]
    body = calls[0] if len(calls) == 1 else tir.SeqStmt(calls)
    alloc_buffers = buffers if include_arena else buffers[1:]
    block = tir.Block(
        [],
        [],
        [],
        "tilelang_root",
        body,
        alloc_buffers=alloc_buffers,
    )
    return tir.PrimFunc([], tir.BlockRealize([], True, block))


def _lower_single_workspace(call, slot: int, buffers, model: str = "ascendc"):
    block = tir.Block(
        [],
        [],
        [],
        "tilelang_root",
        tir.Evaluate(call),
        alloc_buffers=buffers,
    )
    program = tir.PrimFunc([], tir.BlockRealize([], True, block))
    func = _inject(program, model)
    lowered_call = _collect_calls(func, call.op.name)[0]
    workspace = lowered_call.args[slot] if slot < len(lowered_call.args) else None
    if not (isinstance(workspace, tir.Call) and workspace.op.name == "tir.tvm_access_ptr" and workspace.args[1].name == "tmp_ub"):
        workspace = None
    return func, lowered_call, workspace


def _workspace_bytes(workspace):
    if workspace is None:
        return 0
    return int(workspace.args[3]) * tvm.DataType(workspace.args[0].dtype).itemsize()


def test_every_workspace_api_exposes_the_public_tmp_contract():
    apis = [
        T.reduce_sum,
        T.reduce_max,
        T.reduce_min,
        ascend_tile.sort,
        ascend_tile.merge_sort,
        ascend_tile.topk,
        ascend_tile.gather_mask,
        ascend_tile.select,
        ascend_tile.sigmoid,
        ascend_tile.bilinear_interpolation,
        ascend_tile.gather,
        ascend_tile.sin,
        ascend_tile.cos,
        ascend_tile.pow,
        ascend_tile.bitwise_xor,
        ascend_tile.clamp_max,
        ascend_tile.clamp_min,
        ascend_tile.clamp,
        ascend_tile.round,
        ascend_tile.broadcast,
        ascend_tile.reduce_sum_experiment,
        ascend_tile.reduce_sum_mask_experiment,
    ]

    for api in apis:
        parameter = inspect.signature(api).parameters["tmp"]
        assert parameter.kind == inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None

    arena_dtype = "uint8"
    calls_and_slots, _, arena = _workspace_api_calls(explicit=True, arena_dtype=arena_dtype)

    for call, slot in calls_and_slots:
        workspace = call.args[slot]
        assert isinstance(workspace, tir.Call)
        assert workspace.op.name == "tir.tvm_access_ptr"
        assert workspace.args[1].same_as(arena.data)
        assert workspace.args[0].dtype == arena_dtype


def test_explicit_tmp_arena_preserves_byte_geometry():
    reduce_src = tir.decl_buffer((4, 8), "float32")
    reduce_dst = tir.decl_buffer((4,), "float32")
    broadcast_src = tir.decl_buffer((4, 1), "float32")

    for dtype, region_min, region_extent in [
        ("uint8", 32, 64),
        ("float16", 16, 32),
        ("float32", 8, 16),
        ("int16", 16, 32),
        ("uint32", 8, 16),
    ]:
        arena = tir.decl_buffer((128,), dtype, scope="shared.ub")
        region = tir.BufferRegion(
            arena,
            [tvm.ir.Range.from_min_extent(region_min, region_extent)],
        )
        workspace = T.reduce_sum(reduce_src, reduce_dst, tmp=region).args[3]
        assert workspace.args[0].dtype == dtype
        assert int(workspace.args[2]) == region_min
        assert int(workspace.args[3]) == region_extent
        assert int(workspace.args[3]) * tvm.DataType(dtype).itemsize() == 64

    empty_buffer = tir.decl_buffer((0,), "uint8", scope="shared.ub")
    region_buffer = tir.decl_buffer((256,), "uint8", scope="shared.ub")
    empty_region = tir.BufferRegion(
        region_buffer,
        [tvm.ir.Range.from_min_extent(32, 0)],
    )
    for tmp in [empty_buffer, empty_region]:
        calls = [
            T.reduce_sum(reduce_src, reduce_dst, tmp=tmp),
            T.tile.broadcast(reduce_src, broadcast_src, axis=1, tmp=tmp),
        ]
        for call in calls:
            workspace = call.args[3]
            assert workspace.op.name == "tir.tvm_access_ptr"
            assert int(workspace.args[3]) == 0


def test_explicit_tmp_arena_validation():
    input_buffer = tir.decl_buffer((4, 8), "float32")
    output_buffer = tir.decl_buffer((4,), "float32")
    dynamic = tir.Var("dynamic", "int32")
    base = tir.decl_buffer((256,), "uint8", scope="shared.ub")
    invalid_cases = [
        (object(), TypeError, "reduce_sum tmp must be"),
        (
            tir.decl_buffer((256,), "float16x4", scope="shared.ub"),
            ValueError,
            "fixed-width scalar dtype",
        ),
        (tir.decl_buffer((256,), "uint8"), ValueError, "scope shared.ub"),
        (tir.decl_buffer((16, 16), "uint8", scope="shared.ub"), ValueError, "one-dimensional"),
        (
            tir.decl_buffer((dynamic,), "uint8", scope="shared.ub"),
            ValueError,
            "backing-buffer extent must be static",
        ),
        (
            tir.decl_buffer((-1,), "uint8", scope="shared.ub"),
            ValueError,
            "backing-buffer extent must be non-negative",
        ),
        (
            tir.decl_buffer((256,), "uint8", strides=(2,), scope="shared.ub"),
            ValueError,
            "unit stride",
        ),
        (
            tir.decl_buffer((256,), "uint8", elem_offset=dynamic, scope="shared.ub"),
            ValueError,
            "starting offset must be static",
        ),
        (
            tir.decl_buffer((256,), "uint8", elem_offset=1, scope="shared.ub"),
            ValueError,
            "32-byte aligned",
        ),
        (
            tir.decl_buffer((256,), "uint8", elem_offset=-32, scope="shared.ub"),
            ValueError,
            "starting address must be non-negative",
        ),
        (
            tir.BufferRegion(base, [tvm.ir.Range.from_min_extent(dynamic, 32)]),
            ValueError,
            "offset must be static",
        ),
        (
            tir.BufferRegion(base, [tvm.ir.Range.from_min_extent(32, dynamic)]),
            ValueError,
            "extent must be static",
        ),
        (
            tir.BufferRegion(base, [tvm.ir.Range.from_min_extent(32, -1)]),
            ValueError,
            "extent must be non-negative",
        ),
        (
            tir.BufferRegion(base, [tvm.ir.Range.from_min_extent(240, 32)]),
            ValueError,
            "exceeds backing extent",
        ),
        (
            tir.BufferRegion(base, [tvm.ir.Range.from_min_extent(1, 32)]),
            ValueError,
            "32-byte aligned",
        ),
    ]

    for tmp, error_type, match in invalid_cases:
        with pytest.raises(error_type, match=match):
            T.reduce_sum(input_buffer, output_buffer, tmp=tmp)


@pytest.mark.parametrize("model", ["ascendc", "pto"])
@pytest.mark.parametrize(
    ("explicit", "arena_dtype"),
    [(False, "uint8"), (True, "uint8")],
)
def test_every_workspace_api_has_the_target_specific_lowered_layout(model, explicit, arena_dtype):
    calls_and_slots, buffers, arena = _workspace_api_calls(explicit, arena_dtype=arena_dtype)
    if model == "pto":
        calls_and_slots = [(call, slot) for call, slot in calls_and_slots if call.op.name not in PTO_UNSUPPORTED_WORKSPACE_OPS]
    program = _program_from_calls(calls_and_slots, buffers, include_arena=explicit)
    func = _inject(program, model)

    slots = {call.op.name: slot for call, slot in calls_and_slots if isinstance(call.op, tvm.ir.Op)}
    ascendc_workspace_ops = {
        "tl.ascend_reduce",
        "tl.ascend_sort",
        "tl.ascend_topk",
        "tl.ascend_sigmoid",
        "tl.ascend_bilinear_interpolation",
        "tl.ascend_sin",
        "tl.ascend_cos",
        "tl.ascend_pow",
        "tl.ascend_bitwise_xor",
        "tl.ascend_broadcast",
        "tl.ascend_reducesum_experiment",
        "tl.ascend_reducesum_mask_experiment",
    }
    pto_workspace_ops = {
        "tl.ascend_reduce",
        "tl.ascend_bitwise_xor",
        "tl.ascend_merge_sort",
        "tl.ascend_select",
        "tl.ascend_sort",
        "tl.ascend_topk",
        "tl.ascend_gather",
    }
    workspace_ops = ascendc_workspace_ops if model == "ascendc" else pto_workspace_ops

    lowered_calls = []

    def visitor(node):
        if isinstance(node, tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name in slots:
            lowered_calls.append(node)

    tir.stmt_functor.post_order_visit(func.body, visitor)
    assert len(lowered_calls) == len(calls_and_slots)

    for call in lowered_calls:
        slot = slots[call.op.name]
        workspace = call.args[slot] if slot < len(call.args) else None
        expected_name = arena.name if explicit else "tmp_ub"
        has_workspace = (
            isinstance(workspace, tir.Call) and workspace.op.name == "tir.tvm_access_ptr" and workspace.args[1].name == expected_name
        )
        expected_workspace = call.op.name in workspace_ops
        if model == "pto" and call.op.name == "tl.ascend_gather_mask":
            expected_workspace = isinstance(call.args[3], tir.Call)
        assert has_workspace == expected_workspace
        if has_workspace:
            assert workspace.args[1].name == expected_name
            assert int(workspace.args[4]) == 2

    allocation_names = _allocated_buffer_names(func)
    if explicit:
        assert "tmp_ub" not in allocation_names
    else:
        assert allocation_names.count("tmp_ub") == 1


@pytest.mark.parametrize("op_name", sorted(PTO_UNSUPPORTED_WORKSPACE_OPS))
def test_pto_unsupported_workspace_api_fails_before_codegen(op_name):
    calls_and_slots, buffers, _ = _workspace_api_calls(explicit=False)
    call_and_slot = next(item for item in calls_and_slots if item[0].op.name == op_name)
    program = _program_from_calls([call_and_slot], buffers, include_arena=False)

    with pytest.raises(tvm.error.InternalError, match=rf"{op_name} is not supported by the PTO backend"):
        _inject(program, "pto")


def _reduce_program(
    arena_bytes: int | None,
    clear: bool = True,
    real_shape=None,
    dim: int = -1,
    op: str = "sum",
    shape: tuple[int, int] | None = None,
    dtype: str = "float32",
):
    rows, cols = shape or ((4, 8) if real_shape is not None else (8, 64))
    output_size = rows if dim == -1 else cols
    reduce_fn = {
        "sum": T.reduce_sum,
        "max": T.reduce_max,
        "min": T.reduce_min,
    }[op]

    if arena_bytes is None:

        @T.prim_func
        def main(
            A: T.Tensor((rows, cols), dtype),  # type: ignore
            B: T.Tensor((output_size,), dtype),  # type: ignore
        ):
            with T.Kernel(1, is_npu=True) as (_, vid):
                a_ub = T.alloc_ub((rows, cols), dtype)
                b_ub = T.alloc_ub((output_size,), dtype)
                if vid == 0:
                    T.copy(A, a_ub)
                    reduce_fn(
                        a_ub,
                        b_ub,
                        dim=dim,
                        clear=clear,
                        real_shape=real_shape,
                    )
                    T.copy(b_ub, B)

    else:

        @T.prim_func
        def main(
            A: T.Tensor((rows, cols), dtype),  # type: ignore
            B: T.Tensor((output_size,), dtype),  # type: ignore
        ):
            with T.Kernel(1, is_npu=True) as (_, vid):
                a_ub = T.alloc_ub((rows, cols), dtype)
                b_ub = T.alloc_ub((output_size,), dtype)
                arena_ub = T.alloc_ub((arena_bytes,), "uint8")
                if vid == 0:
                    T.copy(A, a_ub)
                    reduce_fn(
                        a_ub,
                        b_ub,
                        dim=dim,
                        clear=clear,
                        real_shape=real_shape,
                        tmp=arena_ub,
                    )
                    T.copy(b_ub, B)

    return main


def _row_reduce_region_program():
    @T.prim_func
    def main(
        A: T.Tensor((8, 64), "float32"),  # type: ignore
        B: T.Tensor((8,), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (_, vid):
            a_ub = T.alloc_ub((8, 64), "float32")
            b_ub = T.alloc_ub((8,), "float32")
            arena_ub = T.alloc_ub((320,), "uint8")
            if vid == 0:
                T.copy(A, a_ub)
                T.reduce_sum(a_ub, b_ub, clear=False, tmp=arena_ub[32:320])
                T.copy(b_ub, B)

    return main


def _sort_program(explicit: bool, use_region: bool = False, arena_dtype: str = "uint8"):
    arena_elements = 2080 // tvm.DataType(arena_dtype).itemsize()
    region_start = 32 // tvm.DataType(arena_dtype).itemsize()
    if explicit:

        @T.prim_func
        def main():
            with T.Kernel(1, is_npu=True) as (_, vid):
                src_ub = T.alloc_ub((64,), "float32")
                dst_ub = T.alloc_ub((128,), "float32")
                arena_ub = T.alloc_ub((arena_elements,), arena_dtype)
                if vid == 0:
                    if use_region:
                        T.tile.sort(
                            dst_ub,
                            src_ub,
                            64,
                            tmp=arena_ub[region_start:arena_elements],
                        )
                    else:
                        T.tile.sort(dst_ub, src_ub, 64, tmp=arena_ub)

    else:

        @T.prim_func
        def main():
            with T.Kernel(1, is_npu=True) as (_, vid):
                src_ub = T.alloc_ub((64,), "float32")
                dst_ub = T.alloc_ub((128,), "float32")
                if vid == 0:
                    T.tile.sort(dst_ub, src_ub, 64)

    return main


def _explicit_merge_sort_program():
    @T.prim_func
    def main():
        with T.Kernel(1, is_npu=True) as (_, vid):
            src0_ub = T.alloc_ub((64,), "float32")
            src1_ub = T.alloc_ub((64,), "float32")
            dst_ub = T.alloc_ub((128,), "float32")
            arena_ub = T.alloc_ub((544,), "uint8")
            if vid == 0:
                T.tile.merge_sort(
                    dst_ub,
                    src0_ub,
                    src1_ub,
                    tmp=arena_ub[32:544],
                )

    return main


def test_ascendc_merge_sort_codegen_has_no_workspace_argument():
    source = tilelang.lower(
        _explicit_merge_sort_program(),
        target="ascendc",
    ).kernel_source
    merge_line = next(line for line in source.splitlines() if "MergeSort<float>" in line)

    assert "arena_ub" not in merge_line
    assert "dst_ub[0], src0_ub[0], src1_ub[0]" in merge_line


def _run_pipeline_planning(program, model: str):
    target = _target(model)
    mod = tvm.IRModule.from_expr(program)
    mod = tilelang.transform.InjectTmpBuffer(target)(mod)
    mod = tir.transform.BindTarget(target)(mod)
    return tilelang.transform.PipelinePlanning()(mod)["main"]


def test_optional_workspace_metadata_uses_the_lowered_call_layout():
    def ub(name: str, extent: int):
        return tir.decl_buffer((extent,), "float32", name=name, scope="shared.ub")

    pow_dst, pow_src0, pow_src1 = ub("pow_dst", 64), ub("pow_src0", 64), ub("pow_src1", 64)
    i = tir.Var("i", "int32")
    pow_loop = tir.For(
        i,
        0,
        2,
        tir.ForKind.SERIAL,
        tir.SeqStmt(
            [
                tir.Evaluate(ascend_tile.pow(pow_dst, pow_src0, pow_src1)),
                tir.Evaluate(ascend_tile.pow(pow_dst, pow_src0, pow_src1)),
            ]
        ),
        annotations={"num_stages": 2},
    )
    pow_root = tir.Block(
        [],
        [],
        [],
        "tilelang_root",
        pow_loop,
        alloc_buffers=[pow_dst, pow_src0, pow_src1],
    )
    pow_func = tir.PrimFunc([], tir.BlockRealize([], True, pow_root))
    planned_pow = _run_pipeline_planning(pow_func, "pto")
    assert "software_pipeline_order" in planned_pow.body.block.body.annotations

    source = tir.decl_buffer((64,), "float32", name="source", scope="global")
    src0, src1, dst = ub("src0", 64), ub("src1", 64), ub("dst", 128)
    copy = tir.call_extern(
        "handle",
        "copy_gm_to_ub",
        source.access_ptr("r"),
        src0.access_ptr("w"),
    )
    merge_loop = tir.For(
        i,
        0,
        2,
        tir.ForKind.SERIAL,
        tir.SeqStmt(
            [
                tir.Evaluate(copy),
                tir.Evaluate(ascend_tile.merge_sort(dst, src0, src1)),
            ]
        ),
        annotations={"num_stages": 2},
    )
    merge_root = tir.Block(
        [],
        [],
        [],
        "tilelang_root",
        merge_loop,
        alloc_buffers=[src0, src1, dst],
    )
    merge_func = tir.PrimFunc(
        [source.data],
        tir.BlockRealize([], True, merge_root),
        buffer_map={source.data: source},
    )
    planned_merge = _run_pipeline_planning(merge_func, "ascendc")
    assert list(planned_merge.body.block.body.annotations["software_pipeline_order"]) == [0, 1]
    assert list(planned_merge.body.block.body.annotations["software_pipeline_stage"]) == [0, 1]


def test_pto_select_metadata_tracks_shifted_src_and_destination():
    def ub(name: str, dtype: str = "float32"):
        return tir.decl_buffer((64,), dtype, name=name, scope="shared.ub")

    output = tir.decl_buffer((64,), "float32", name="output", scope="global")
    mask, src0, src1, dst = ub("mask", "uint8"), ub("src0"), ub("src1"), ub("dst")
    body = tir.SeqStmt(
        [
            tir.Evaluate(ascend_tile.add(src1, src0, src0)),
            tir.Evaluate(ascend_tile.select(dst, mask, src0, src1, "VSEL_TENSOR_TENSOR_MODE")),
            tir.Evaluate(
                tir.call_extern(
                    "handle",
                    "copy_ub_to_gm",
                    dst.access_ptr("r"),
                    output.access_ptr("w"),
                )
            ),
        ]
    )
    root = tir.Block(
        [],
        [],
        [],
        "tilelang_root",
        body,
        alloc_buffers=[mask, src0, src1, dst],
    )
    func = tir.PrimFunc(
        [output.data],
        tir.BlockRealize([], True, root),
        buffer_map={output.data: output},
    )
    target = _target("pto")
    mod = tvm.IRModule.from_expr(func)
    mod = tilelang.transform.InjectTmpBuffer(target)(mod)
    mod = tir.transform.BindTarget(target)(mod)
    with tvm.transform.PassContext(config={"tl.ascend_auto_sync": True}):
        mod = tilelang.transform.AscendSyncInsert(target, "A2")(mod)
        mod = tilelang.transform.AscendSyncInsertVS(target, "A2")(mod)

    lowered = mod["main"]
    assert _collect_calls(lowered, "tl.ascend_auto_barrier")
    assert _collect_calls(lowered, "tl.ascend_auto_set_flag")
    assert _collect_calls(lowered, "tl.ascend_auto_wait_flag")


def test_mixed_typed_implicit_calls_share_one_byte_arena():
    @T.prim_func
    def main():
        with T.Kernel(1, is_npu=True) as (_, vid):
            sort_src = T.alloc_ub((64,), "float32")
            sort_dst = T.alloc_ub((128,), "float32")
            explicit_sort_dst = T.alloc_ub((128,), "float32")
            xor_src0 = T.alloc_ub((64,), "int16")
            xor_src1 = T.alloc_ub((64,), "int16")
            xor_dst = T.alloc_ub((64,), "int16")
            arena_ub = T.alloc_ub((4096,), "uint8")
            if vid == 0:
                T.tile.sort(sort_dst, sort_src, 64)
                T.tile.sort(explicit_sort_dst, sort_src, 64, tmp=arena_ub)
                T.tile.bitwise_xor(xor_dst, xor_src0, xor_src1)

    func = _inject(main, "pto")
    sort_calls = _collect_calls(func, "tl.ascend_sort")
    implicit_sort_tmp = next(call.args[3] for call in sort_calls if call.args[3].args[1].name == "tmp_ub")
    explicit_sort_tmp = next(call.args[3] for call in sort_calls if call.args[3].args[1].name == "arena_ub")
    xor_tmp = _collect_calls(func, "tl.ascend_bitwise_xor")[0].args[3]
    allocations = _allocated_buffer_names(func)

    assert allocations.count("tmp_ub") == 1
    assert not any(name.startswith("tmp_ub_") for name in allocations)
    assert implicit_sort_tmp.args[1].same_as(xor_tmp.args[1])
    assert implicit_sort_tmp.args[0].dtype == "float32"
    assert xor_tmp.args[0].dtype == "int16"
    assert int(implicit_sort_tmp.args[3]) * 4 == 1024
    assert explicit_sort_tmp.args[1].name == "arena_ub"


def test_implicit_workspace_names_do_not_collide_with_explicit_arenas():
    @T.prim_func
    def main():
        with T.Kernel(1, is_npu=True) as (_, vid):
            reduce_src = T.alloc_ub((8, 64), "float32")
            reduce_dst = T.alloc_ub((8,), "float32")
            sort_src = T.alloc_ub((64,), "float32")
            sort_dst0 = T.alloc_ub((128,), "float32")
            sort_dst1 = T.alloc_ub((128,), "float32")
            tmp_ub = T.alloc_ub((1024,), "uint8")
            tmp_ub_reduce_out = T.alloc_ub((1024,), "uint8")
            if vid == 0:
                T.tile.sort(sort_dst0, sort_src, 64, tmp=tmp_ub)
                T.tile.sort(sort_dst1, sort_src, 64, tmp=tmp_ub_reduce_out)
                T.reduce_sum(reduce_src, reduce_dst, clear=False)

    source = tilelang.lower(main, target="pto").kernel_source

    assert "tmp_ub_1" in source
    assert "tmp_ub_reduce_out_1" in source


def test_pto_zero_workspace_apis_elide_an_explicit_empty_arena():
    @T.prim_func
    def main():
        with T.Kernel(1, is_npu=True) as (_, vid):
            src_ub = T.alloc_ub((64,), "float32")
            dst_ub = T.alloc_ub((64,), "float32")
            arena_ub = T.alloc_ub((0,), "uint8")
            if vid == 0:
                T.tile.sigmoid(dst_ub, src_ub, tmp=arena_ub)
                T.tile.pow(dst_ub, src_ub, src_ub, tmp=arena_ub)
                T.tile.clamp_max(dst_ub, src_ub, 1.0, 64, tmp=arena_ub)
                T.tile.clamp_min(dst_ub, src_ub, -1.0, 64, tmp=arena_ub)
                T.tile.clamp(dst_ub, src_ub, -1.0, 1.0, 64, tmp=arena_ub)
                T.tile.round(dst_ub, src_ub, 64, tmp=arena_ub)
                T.tile.gather_mask(dst_ub, src_ub, "P0101", tmp=arena_ub)

    func = _inject(main, "pto", repeat=2)
    op_names = {
        "tl.ascend_sigmoid",
        "tl.ascend_pow",
        "tl.ascend_clamp_max",
        "tl.ascend_clamp_min",
        "tl.ascend_clamp",
        "tl.ascend_round",
        "tl.ascend_gather_mask",
    }

    def references_arena(node):
        return isinstance(node, tir.Call) and node.op.name == "tir.tvm_access_ptr" and node.args[1].name == "arena_ub"

    for op_name in op_names:
        call = _collect_calls(func, op_name)[0]
        assert not any(references_arena(arg) for arg in call.args)
    assert "tmp_ub" not in _allocated_buffer_names(func)

    source = tilelang.lower(main, target="pto").kernel_source
    assert "arena_ub" not in source
    assert "tmp_ub" not in source
    assert "TSIGMOID" in source
    assert "tl::ascend_pto::pow" in source
    assert "TGATHER<" in source


def test_typed_workspace_codegen_preserves_dtype_and_byte_address():
    program = _sort_program(explicit=True, use_region=True, arena_dtype="int16")
    expected = {
        "ascendc": (512, "arena_ub.ReinterpretCast<float>()[8]"),
        "pto": (256, "+ ((8) * 4)"),
    }
    for model, (expected_extent, expected_source) in expected.items():
        func = _inject(program, model)
        workspace = _collect_calls(func, "tl.ascend_sort")[0].args[3]
        assert "tmp_ub" not in _allocated_buffer_names(func), model
        assert workspace.args[1].name == "arena_ub", model
        assert workspace.args[0].dtype == "float32", model
        assert int(workspace.args[2]) == 8, model
        assert int(workspace.args[3]) == expected_extent, model

        source = tilelang.lower(program, target=model).kernel_source
        assert expected_source in source, model
        assert "tmp_ub" not in source, model


def test_pto_explicit_reduce_uses_byte_correct_row_and_column_views():
    @T.prim_func
    def main():
        with T.Kernel(1, is_npu=True) as (_, vid):
            src_ub = T.alloc_ub((8, 64), "float32")
            dst_ub = T.alloc_ub((8,), "float32")
            arena_ub = T.alloc_ub((80,), "float32")
            if vid == 0:
                T.reduce_sum(
                    src_ub,
                    dst_ub,
                    clear=False,
                    tmp=arena_ub[8:80],
                )

    func = _inject(main, "pto")
    call = _collect_calls(func, "tl.ascend_reduce")[0]
    main_view, output_view = call.args[3], call.args[4]

    assert main_view.args[0].dtype == "uint8"
    assert output_view.args[0].dtype == "uint8"
    assert int(main_view.args[2]) == 32
    assert int(main_view.args[3]) == 256
    assert int(output_view.args[2]) == 288
    assert int(output_view.args[3]) == 32

    column_func = _inject(_reduce_program(256, clear=False, dim=0), "pto")
    column_call = _collect_calls(column_func, "tl.ascend_reduce")[0]
    column_output_view = column_call.args[3]
    assert column_output_view.args[1].name == "arena_ub"
    assert int(column_output_view.args[2]) == 0
    assert int(column_output_view.args[3]) == 256
    assert not bool(column_call.args[4])

    row_source = tilelang.lower(_row_reduce_region_program(), target="pto").kernel_source
    assert "+ (32 / 4) * 4" in row_source
    assert "+ (288 / 4) * 4" in row_source
    assert "TROWSUM(" in row_source
    assert "TADD(" in row_source

    column_source = tilelang.lower(
        _reduce_program(256, clear=False, dim=0),
        target="pto",
    ).kernel_source
    assert "arena_ub_temp_" not in column_source
    assert "TCOLSUM(" in column_source
    assert "TADD(" in column_source


def test_explicit_arena_capacity_contract():
    pto_func = _inject(_reduce_program(1, clear=False), "pto")
    pto_call = _collect_calls(pto_func, "tl.ascend_reduce")[0]
    assert pto_call.args[3].args[1].name == "arena_ub"
    assert int(pto_call.args[3].args[3]) == 256
    assert int(pto_call.args[4].args[2]) == 256
    assert int(pto_call.args[4].args[3]) == 32

    ascendc_func = _inject(_reduce_program(1), "ascendc")
    ascendc_call = _collect_calls(ascendc_func, "tl.ascend_reduce")[0]
    assert ascendc_call.args[3].args[1].name == "arena_ub"
    assert int(ascendc_call.args[3].args[3]) == 1

    for model in ["ascendc", "pto"]:
        with pytest.raises(tvm.error.TVMError, match=r"is empty.*non-empty workspace"):
            _inject(_reduce_program(0), model)


def test_implicit_pto_reduce_allocations_follow_row_and_column_layouts():
    func = _inject(_reduce_program(None, clear=False), "pto")
    names = _allocated_buffer_names(func)
    call = _collect_calls(func, "tl.ascend_reduce")[0]

    assert "tmp_ub" in names
    assert "tmp_ub_reduce_out" in names
    assert call.args[3].args[1].name == "tmp_ub"
    assert call.args[4].args[1].name == "tmp_ub_reduce_out"

    column_func = _inject(_reduce_program(None, clear=False, dim=0), "pto")
    column_names = _allocated_buffer_names(column_func)
    column_call = _collect_calls(column_func, "tl.ascend_reduce")[0]
    assert "tmp_ub" not in column_names
    assert "tmp_ub_reduce_out" in column_names
    assert column_call.args[3].args[1].name == "tmp_ub_reduce_out"
    assert not bool(column_call.args[4])


def test_physical_row_remains_after_clear_false_and_two_tmp_views():
    arena_bytes = 160
    func = _inject(
        _reduce_program(
            arena_bytes,
            clear=False,
            real_shape=[4, 4],
        ),
        "pto",
    )
    call = _collect_calls(func, "tl.ascend_reduce")[0]

    assert call.args[3].op.name == "tir.tvm_access_ptr"
    assert call.args[4].op.name == "tir.tvm_access_ptr"
    assert not bool(call.args[5])
    assert int(call.args[6]) == 8


def test_inject_tmp_buffer_is_idempotent_for_all_workspace_layouts():
    cases = [
        ("ascendc-implicit", _sort_program(explicit=False), "ascendc"),
        ("ascendc-explicit", _sort_program(explicit=True), "ascendc"),
        ("pto-two-view", _reduce_program(None, clear=False), "pto"),
        ("pto-output-only", _reduce_program(256, clear=False, dim=0), "pto"),
    ]
    for case_id, program, model in cases:
        once = _inject(program, model)
        twice = _inject(program, model, repeat=2)
        assert tvm.ir.structural_equal(once, twice, map_free_vars=True), case_id


def test_ascendc_broadcast_codegen_preserves_explicit_tmp_region_offset():
    @T.prim_func
    def main(
        A: T.Tensor((8, 1), "float32"),  # type: ignore
        B: T.Tensor((8, 64), "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (_, vid):
            src_ub = T.alloc_ub((8, 1), "float32")
            dst_ub = T.alloc_ub((8, 64), "float32")
            arena_ub = T.alloc_ub((544,), "uint8")
            if vid == 0:
                T.copy(A, src_ub)
                T.tile.broadcast(
                    dst_ub,
                    src_ub,
                    axis=1,
                    tmp=arena_ub[32:544],
                )
                T.copy(dst_ub, B)

    source = tilelang.lower(main, target="ascendc").kernel_source

    assert "arena_ub[32]" in source


def test_reduce_zero_workspace_paths_elide_explicit_and_implicit_tmp():
    func = _inject(_reduce_program(0, dim=0), "pto")
    call = _collect_calls(func, "tl.ascend_reduce")[0]
    assert len(call.args) == 4
    assert bool(call.args[3])

    for case_id, kwargs in [
        ("narrow-row", {"real_shape": [4, 4]}),
        ("half-sum", {"dtype": "float16"}),
    ]:
        for arena_bytes in [0, None]:
            reduced = _inject(_reduce_program(arena_bytes, **kwargs), "ascendc")
            reduce_call = _collect_calls(reduced, "tl.ascend_reduce")[0]
            assert not any(isinstance(arg, tir.Call) and arg.op.name == "tir.tvm_access_ptr" for arg in reduce_call.args[3:]), (
                case_id,
                arena_bytes,
            )
            assert "tmp_ub" not in _allocated_buffer_names(reduced), (case_id, arena_bytes)


@pytest.mark.parametrize(
    ("op", "shape", "dim", "expected_bytes"),
    [
        ("sum", (8, 64), -1, 32),
        ("sum", (8, 256), -1, 4096),
        ("max", (8, 32), -1, 256),
        ("max", (8, 64), -1, 2048),
        ("sum", (8, 64), 0, 1024),
    ],
)
def test_ascendc_implicit_reduce_uses_transitional_heuristic(
    op,
    shape,
    dim,
    expected_bytes,
):
    func = _inject(_reduce_program(None, op=op, shape=shape, dim=dim), "ascendc")
    call = _collect_calls(func, "tl.ascend_reduce")[0]

    assert call.args[3].args[1].name == "tmp_ub"
    assert int(call.args[3].args[3]) == expected_bytes


@pytest.mark.parametrize(
    ("op_name", "dtype", "count", "expected_dtype", "expected_bytes"),
    [
        ("sort", "float16", 64, "float16", 1024),
        ("sort", "float32", 64, "float32", 512),
        ("topk", "float16", 64, "float16", 1280),
        ("topk", "float32", 64, "float32", 1024),
        ("sin", "float16", 64, "uint8", 512),
        ("sin", "float16", 512, "uint8", 2048),
        ("cos", "float32", 32, "uint8", 384),
        ("cos", "float32", 256, "uint8", 2048),
        ("pow", "float16", 64, "uint8", 1152),
        ("pow", "float32", 32, "uint8", 768),
        ("pow", "float32", 256, "uint8", 2048),
        ("pow", "int32", 32, "uint8", 768),
        ("pow", "int32", 256, "uint8", 2048),
        ("xor", "int16", 16, "uint8", 64),
        ("xor", "int16", 64, "uint8", 128),
        ("round", "float16", 64, "uint8", 256),
        ("round", "float16", 256, "uint8", 512),
        ("sigmoid", "float32", 64, "uint8", 256),
        ("reduce_sum_experiment", "float16", 64, "float16", 128),
        ("reduce_sum_mask_experiment", "float32", 64, "float32", 256),
    ],
)
def test_ascendc_dav_2201_implicit_workspace_policy(
    op_name,
    dtype,
    count,
    expected_dtype,
    expected_bytes,
):
    src = _ub_buffer("src", (count,), dtype)
    dst_extent = count * 2 if op_name in {"sort", "topk"} else count
    dst = _ub_buffer("dst", (dst_extent,), dtype)
    buffers = [src, dst]

    if op_name == "sort":
        call, slot = ascend_tile.sort(dst, src, count), 3
    elif op_name == "topk":
        call, slot = ascend_tile.topk(dst, src, 16, count), 3
    elif op_name == "sin":
        call, slot = ascend_tile.sin(dst, src), 2
    elif op_name == "cos":
        call, slot = ascend_tile.cos(dst, src), 2
    elif op_name == "pow":
        call, slot = ascend_tile.pow(dst, src, src), 3
    elif op_name == "xor":
        call, slot = ascend_tile.bitwise_xor(dst, src, src), 3
    elif op_name == "round":
        call, slot = ascend_tile.round(dst, src, count), 2
    elif op_name == "sigmoid":
        call, slot = ascend_tile.sigmoid(dst, src), 2
    elif op_name == "reduce_sum_experiment":
        call, slot = ascend_tile.reduce_sum_experiment(dst, src, count), 2
    else:
        call, slot = ascend_tile.reduce_sum_mask_experiment(dst, src, count, 1, 1), 2

    func, _, workspace = _lower_single_workspace(call, slot, buffers)

    assert workspace is not None
    assert workspace.args[0].dtype == expected_dtype
    assert _workspace_bytes(workspace) == expected_bytes
    assert _allocated_buffer_names(func).count("tmp_ub") == 1


def test_ascendc_dav_2201_bilinear_workspace_policy():
    src0 = _ub_buffer("src0", (32,), "float32")
    src1 = _ub_buffer("src1", (64,), "float32")
    dst = _ub_buffer("dst", (64,), "float32")
    offsets = _ub_buffer("offsets", (64,), "uint32")
    with pytest.warns(DeprecationWarning):
        call = ascend_tile.bilinear_interpolation(
            dst,
            src0,
            offsets,
            src1,
            64,
            1,
            False,
            1,
            0,
            1,
        )

    _, _, workspace = _lower_single_workspace(call, 10, [src0, src1, dst, offsets])

    assert workspace.args[0].dtype == "uint8"
    assert _workspace_bytes(workspace) == (32 + 64) * 32


@pytest.mark.parametrize(
    ("dtype", "src_shape", "dst_shape", "axis", "expected_bytes"),
    [
        ("float16", (4, 8), (4, 8), 1, 0),
        ("float32", (1,), (64,), 0, 0),
        ("float16", (1, 64), (8, 64), 0, 32),
        ("float16", (8, 1), (8, 16), 1, 512),
        ("float16", (8, 1), (8, 17), 1, 1536),
        ("float32", (8, 1), (8, 8), 1, 256),
        ("float32", (8, 1), (8, 9), 1, 768),
        ("uint8", (4, 8), (4, 8), 1, 128),
        ("uint8", (1,), (64,), 0, 160),
        ("uint8", (1, 64), (8, 64), 0, 1184),
        ("uint8", (8, 1), (8, 16), 1, 800),
        ("uint8", (8, 1), (8, 17), 1, 1856),
    ],
)
def test_ascendc_dav_2201_broadcast_workspace_policy(
    dtype,
    src_shape,
    dst_shape,
    axis,
    expected_bytes,
):
    src = _ub_buffer("src", src_shape, dtype)
    dst = _ub_buffer("dst", dst_shape, dtype)
    call = ascend_tile.broadcast(dst, src, axis=axis)

    func, lowered_call, workspace = _lower_single_workspace(call, 3, [src, dst])

    assert _workspace_bytes(workspace) == expected_bytes
    assert (workspace is not None) == (expected_bytes > 0)
    assert ("tmp_ub" in _allocated_buffer_names(func)) == (expected_bytes > 0)
    if workspace is None:
        assert int(lowered_call.args[3]) == axis + 1
    else:
        assert workspace.args[0].dtype == "uint8"


def test_ascendc_zero_workspace_codegen_uses_basic_intrinsics():
    @T.prim_func
    def main():
        with T.Kernel(1, is_npu=True) as (_, vid):
            src_ub = T.alloc_ub((64,), "float32")
            dst_ub = T.alloc_ub((64,), "float32")
            equal_ub = T.alloc_ub((64,), "float32")
            if vid == 0:
                T.tile.clamp(dst_ub, src_ub, -1.0, 1.0, 64)
                T.tile.round(dst_ub, src_ub, 64)
                T.tile.broadcast(equal_ub, src_ub, axis=0)

    source = tilelang.lower(main, target="ascendc").kernel_source

    assert "AscendC::Maxs" in source
    assert "AscendC::Mins" in source
    assert "CAST_RINT" in source
    assert "tl::ascend::Broadcast" in source
    assert "tmp_ub" not in source
    assert "PopStackBuffer" not in source


def test_ascendc_experimental_reduce_codegen_uses_source_dtype_workspace():
    @T.prim_func
    def main():
        with T.Kernel(1, is_npu=True) as (_, vid):
            src_ub = T.alloc_ub((64,), "float32")
            dst_ub = T.alloc_ub((64,), "float32")
            if vid == 0:
                T.tile.reduce_sum_experiment(dst_ub, src_ub, 64)

    source = tilelang.lower(main, target="ascendc").kernel_source
    reduce_line = next(line for line in source.splitlines() if "AscendC::ReduceSum(" in line)

    assert "tmp_ub.ReinterpretCast<float>()" in reduce_line
    assert "ReinterpretCast<uint8_t>()" not in reduce_line


def test_pto_select_and_gather_codegen_preserve_workspace_region_offsets():
    @T.prim_func
    def main():
        with T.Kernel(1, is_npu=True) as (_, vid):
            src_ub = T.alloc_ub((64,), "float32")
            dst_ub = T.alloc_ub((64,), "float32")
            mask_ub = T.alloc_ub((64,), "uint8")
            indices_ub = T.alloc_ub((64,), "uint32")
            select_arena = T.alloc_ub((288,), "uint8")
            gather_arena = T.alloc_ub((144,), "uint32")
            if vid == 0:
                T.tile.select(
                    dst_ub,
                    mask_ub,
                    src_ub,
                    1.0,
                    "VSEL_TENSOR_SCALAR_MODE",
                    tmp=select_arena[32:288],
                )
                T.tile.gather(
                    dst_ub,
                    src_ub,
                    indices_ub,
                    0,
                    tmp=gather_arena[8:72],
                )
                T.tile.gather_mask(
                    dst_ub,
                    src_ub,
                    indices_ub,
                    tmp=gather_arena[72:136],
                )

    source = tilelang.lower(main, target="pto").kernel_source
    assign_line = next(line for line in source.splitlines() if "TASSIGN(select_arena_temp_" in line)

    assert "+ 32 * 1" in assign_line
    assert "TSELS(" in source
    assert "+ 8 * 4" in source
    assert "+ 72 * 4" in source
    assert source.count("TGATHER(") == 2


def test_pto_half_sort_codegen_scales_typed_workspace_offset():
    @T.prim_func
    def main():
        with T.Kernel(1, is_npu=True) as (_, vid):
            src_ub = T.alloc_ub((64,), "float16")
            sort_dst = T.alloc_ub((128,), "float16")
            sort_arena = T.alloc_ub((2080,), "uint8")
            if vid == 0:
                T.tile.sort(sort_dst, src_ub, 64, tmp=sort_arena[32:2080])

    func = _inject(main, "pto")
    workspace = _collect_calls(func, "tl.ascend_sort")[0].args[3]
    assert workspace.args[0].dtype == "float16"
    assert int(workspace.args[2]) == 16
    assert int(workspace.args[3]) == 1024

    source = tilelang.lower(main, target="pto").kernel_source
    assert "+ ((16) * 2)" in source
