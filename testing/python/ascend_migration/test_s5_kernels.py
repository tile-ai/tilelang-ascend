"""S5 fixtures built from S4's aligned copy/fill/add Vector operations."""

from tilelang import language as T
from tilelang import tvm


def calibration_pass_configs():
    """Explicit synchronization belongs only to this legacy measurement harness."""
    return {
        "tl.ascend_pipeline_planner": "legacy",
        "tl.ascend_auto_sync": False,
        "tl.ascend_auto_sync_vs": False,
        "tl.ascend_memory_planning": True,
        "tl.ascend_auto_cross_core_sync": False,
        "tl.ascend_auto_cv_combine": False,
    }


def make_calibration_kernel(operation="add", pattern="dependent", dtype="float32", tile=64):
    """Dynamic-count legacy microbenchmark; A/B/O must be disjoint allocations."""
    if operation not in ("gmub", "ubgm", "fill", "add"):
        raise ValueError("Unsupported calibration operation")
    if pattern not in ("dependent", "independent"):
        raise ValueError("Unsupported calibration pattern")
    if dtype not in ("float16", "float32") or tile not in (32, 64):
        raise ValueError("Calibration supports f16/f32 with tile32/64")
    independent = pattern == "independent"
    row = "i % 128" if independent else "0"
    declarations = "        a = T.alloc_ub((128, tile), dtype)\n"
    setup = ""
    final = ""
    if operation == "add":
        declarations += "        b = T.alloc_ub((128, tile), dtype)\n"
        setup = "            T.copy(A, a)\n            T.copy(B, b)\n"
        if independent:
            declarations += "        c = T.alloc_ub((128, tile), dtype)\n"
            setup += "            T.tile.fill(c, 0.0)\n"
            measured = "T.tile.add(c[i % 128, :], a[i % 128, :], b[i % 128, :])"
            final = "            T.copy(c, O)\n"
        else:
            measured = "T.tile.add(a[0, :], a[0, :], b[0, :])"
            final = "            T.copy(a, O)\n"
        pipe = "V"
    elif operation == "fill":
        setup = "            T.copy(A, a)\n" if not independent else "            T.tile.fill(a, 0.0)\n"
        measured = f"T.tile.fill(a[{row}, :], 1.0)"
        final = "            T.copy(a, O)\n"
        pipe = "V"
    elif operation == "gmub":
        setup = "            T.tile.fill(a, 0.0)\n"
        measured = f"T.copy(A[{row}, :], a[{row}, :])"
        final = "            T.copy(a, O)\n"
        pipe = "MTE2"
    else:
        setup = "            T.copy(A, a)\n"
        measured = f"T.copy(a[{row}, :], O[{row}, :])"
        pipe = "MTE3"
    inner_barrier = (
        f'                if (i + 1) % 128 == 0 and i + 1 < n:\n                    T.pipe_barrier("{pipe}")\n'
        if independent
        else f'                T.pipe_barrier("{pipe}")\n'
    )
    source = f"""@T.prim_func
def main(A: T.Tensor((128, tile), dtype), B: T.Tensor((128, tile), dtype),
         O: T.Tensor((128, tile), dtype), Count: T.Tensor((n,), "int32")):
    T.func_attr({{"tir.noalias": True}})
    with T.Kernel(1, threads=1, is_npu=True):
{declarations}        with T.Scope("V"):
{setup}            T.barrier_all()
            for i in T.serial(n):
                {measured}
{inner_barrier}            T.barrier_all()
{final}            T.barrier_all()
"""
    return tvm.script.from_source(
        source,
        extra_vars={"T": T, "dtype": dtype, "tile": tile, "n": T.symbolic("s5_count", "int32")},
    )


def _check_vector_arguments(dtype, tile, num_stages):
    if dtype not in ("float16", "float32") or type(tile) is not int or tile not in (32, 64):
        raise ValueError("S5 Vector fixtures support f16/f32 with tile32/64")
    if type(num_stages) is not int or not 0 <= num_stages <= 3:
        raise ValueError("S5 fixtures support requested num_stages=0/1/2/3")


def _body(calls, indentation, manual_stages=None):
    lines = []
    for index, call in enumerate(calls):
        if manual_stages is None:
            lines.append(" " * indentation + call)
        else:
            lines.extend(
                [
                    " " * indentation + f"with T.Stage({manual_stages[index]}):",
                    " " * (indentation + 4) + "with T.Task():",
                    " " * (indentation + 8) + call,
                ]
            )
    return "\n".join(lines) + "\n"


def make_pipeline_kernel(
    trips=8,
    num_stages=2,
    dtype="float32",
    tile=64,
    tail=0,
    *,
    manual_stages=None,
    enable_offset=False,
    legacy_zero_baseline=False,
):
    """Aligned add with a finite anchor and untouched suffix; A/B/O are disjoint.

    Requested stages are not a promise about actual physical versions. The
    caller must inspect the injector output and final physical allocation.
    """
    _check_vector_arguments(dtype, tile, num_stages)
    if type(trips) is not int or trips < 0:
        raise ValueError("trips must be a nonnegative static integer")
    if type(tail) is not int or tail not in (0, 32) or tail and tile != 64:
        raise ValueError("tail must be zero or a separate complete32 tile after tile64")
    if type(enable_offset) is not bool:
        raise ValueError("enable_offset must be bool")
    if type(legacy_zero_baseline) is not bool or legacy_zero_baseline and (trips or num_stages):
        raise ValueError("legacy_zero_baseline is restricted to trip0/num_stages0")
    if manual_stages is not None:
        if (
            not isinstance(manual_stages, (tuple, list))
            or len(manual_stages) != 4
            or any(type(stage) is not int or stage < 0 for stage in manual_stages)
        ):
            raise ValueError("manual_stages requires four nonnegative static stages")
        if any(manual_stages) and not enable_offset:
            raise ValueError("Nonzero manual stages require enable_offset=True")
    total = (trips + 2) * tile + tail
    declarations = ""
    tail_body = ""
    if tail:
        declarations = """        tail_a = T.alloc_ub((32,), dtype)
        tail_b = T.alloc_ub((32,), dtype)
        tail_c = T.alloc_ub((32,), dtype)
"""
        start = (trips + 1) * tile
        tail_body = _body(
            [
                f"T.copy(A[{start}:{start + tail}], tail_a)",
                f"T.copy(B[{start}:{start + tail}], tail_b)",
                "T.tile.add(tail_c, tail_a, tail_b)",
                f"T.copy(tail_c, O[{start}:{start + tail}])",
            ],
            12,
        )
    loop = _body(
        [
            "T.copy(A[(block + 1) * tile:(block + 2) * tile], a)",
            "T.copy(B[(block + 1) * tile:(block + 2) * tile], b)",
            "T.tile.add(c, a, b)",
            "T.copy(c, O[(block + 1) * tile:(block + 2) * tile])",
        ],
        16,
        manual_stages,
    )
    annotations = ', annotations={"enable_offset": True}' if enable_offset else ""
    anchor_declaration = "        anchor = T.alloc_ub((tile,), dtype)\n"
    anchor_body = "            T.tile.fill(anchor, 7.0)\n            T.copy(anchor, O[0:tile])\n"
    if legacy_zero_baseline:
        # Legacy codegen retains unused zero-loop allocations without addresses.
        # This explicitly serial baseline gives those same buffers real work;
        # the constraint fixture keeps its original, private zero-trip candidate.
        anchor_declaration = ""
        anchor_body = (
            "            T.tile.fill(a, 3.0)\n            T.tile.fill(b, 4.0)\n"
            "            T.tile.add(c, a, b)\n            T.copy(c, O[0:tile])\n"
        )
    source = f"""@T.prim_func
def main(A: T.Tensor((total,), dtype), B: T.Tensor((total,), dtype),
         O: T.Tensor((total,), dtype)):
    T.func_attr({{"tir.noalias": True}})
    with T.Kernel(1, threads=1, is_npu=True):
{anchor_declaration}        a = T.alloc_ub((tile,), dtype)
        b = T.alloc_ub((tile,), dtype)
        c = T.alloc_ub((tile,), dtype)
{declarations}        with T.Scope("V"):
{anchor_body}            for block in T.Pipelined(trips, num_stages=num_stages{annotations}):
{loop}{tail_body}"""
    return tvm.script.from_source(
        source,
        extra_vars={
            "T": T,
            "dtype": dtype,
            "tile": tile,
            "total": total,
            "trips": trips,
            "num_stages": num_stages,
        },
    )


def persistent_layout(domain, cores, group_size):
    """Allocate canary columns for all padded and final-wave overshoot coordinates."""
    if (
        not isinstance(domain, (tuple, list))
        or len(domain) != 2
        or any(type(extent) is not int or extent < 1 for extent in domain)
        or type(cores) is not int
        or cores < 1
        or type(group_size) is not int
        or group_size < 1
    ):
        raise ValueError("Persistent fixture requires a positive static 2D domain/cores/group")
    rows, columns = domain
    padded = rows * ((columns + group_size - 1) // group_size) * group_size
    waves = (padded + cores - 1) // cores
    scheduled_slots = waves * cores
    stride = rows * group_size
    physical_columns = ((scheduled_slots + stride - 1) // stride) * group_size + group_size
    return {
        "rows": rows,
        "columns": columns,
        "physical_columns": physical_columns,
        "padded_slots": padded,
        "waves": waves,
    }


def make_persistent_pipeline_kernel(domain=(3, 10), cores=8, group_size=4, num_stages=2, dtype="float32", tile=64):
    """Vector add over logical tiles with finite padded inputs and NaN output canaries."""
    _check_vector_arguments(dtype, tile, num_stages)
    layout = persistent_layout(domain, cores, group_size)
    shape = (layout["rows"], layout["physical_columns"], tile)
    source = """@T.prim_func
def main(A: T.Tensor(shape, dtype), B: T.Tensor(shape, dtype), O: T.Tensor(shape, dtype)):
    T.func_attr({"tir.noalias": True})
    with T.Kernel(cores, threads=1, is_npu=True) as cid:
        a = T.alloc_ub((tile,), dtype)
        b = T.alloc_ub((tile,), dtype)
        c = T.alloc_ub((tile,), dtype)
        with T.Scope("V"):
            for row, column in T.Persistent(domain, cores, cid, group_size, num_stages):
                T.copy(A[row, column, :], a)
                T.copy(B[row, column, :], b)
                T.tile.add(c, a, b)
                T.copy(c, O[row, column, :])
"""
    return tvm.script.from_source(
        source,
        extra_vars={
            "T": T,
            "shape": shape,
            "dtype": dtype,
            "tile": tile,
            "domain": list(domain),
            "cores": cores,
            "group_size": group_size,
            "num_stages": num_stages,
        },
    )
