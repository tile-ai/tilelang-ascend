"""S6 reuse fixtures derived from the existing scoped Vector copy/add kernels."""

from tilelang import language as T
from tilelang import tvm


def make_reuse_kernel(kind="readonly", trips=5, dtype="float32", tile=64, tiles=3, *, num_stages=1):
    """Use initialized scratch across N1 epochs; caller supplies disjoint GM tensors.

    The first output is the initialized state, the following outputs expose
    every logical update, and the penultimate output observes loop exit. The
    final output row remains a canary. No uninitialized UB data are read.
    """
    if kind not in ("readonly", "carry", "views", "dtype_views", "consecutive", "nested"):
        raise ValueError("Unsupported S6 reuse fixture")
    if dtype not in ("float16", "float32") or type(tile) is not int or tile not in (32, 64):
        raise ValueError("S6 fixtures support f16/f32 and aligned32/64 tiles")
    if type(trips) is not int or trips < 0 or type(tiles) is not int or tiles < 1:
        raise ValueError("trips must be nonnegative and tiles positive static integers")
    if type(num_stages) is not int or num_stages not in (0, 1):
        raise ValueError("S6 initialized scratch uses bound1 or the serial legacy baseline0")
    if kind == "views" and tile != 64:
        raise ValueError("The two offset views use a complete64-element owner")
    if kind == "dtype_views" and (dtype != "float32" or tile != 32):
        raise ValueError("The dtype-alias fixture uses one f32[32]/f16[64] 128-byte owner")
    steps = trips * (2 if kind in ("consecutive", "nested") else 1)
    input_elements = 32 if kind == "views" else tile
    input_steps = max(steps, 1)
    declarations = "        u = T.alloc_ub((tile,), dtype)\n"
    if kind == "dtype_views":
        declarations += '        v = T.decl_buffer((64,), "float16", data=u.data, scope="shared.ub")\n'
        initialize = "                T.copy(Scratch[t, :], u)\n"
    elif kind == "views":
        declarations += (
            '        v = T.decl_buffer((48,), dtype, data=u.data, elem_offset=8, scope="shared.ub")\n'
            '        w = T.decl_buffer((32,), dtype, data=u.data, elem_offset=16, scope="shared.ub")\n'
        )
        initialize = "                T.copy(Scratch[t, :], u)\n"
    else:
        declarations += "        a = T.alloc_ub((tile,), dtype)\n"
        # Both allocations have real, initialized work even for a zero-trip loop.
        initialize = "                T.copy(Scratch[t, :], a)\n                T.tile.add(u, a, a)\n"

    def operations(index, indentation):
        prefix = " " * indentation
        if kind == "dtype_views":
            lines = [
                f"T.copy(A[t, {index}, :], u)",
                "T.tile.add(v, v, v)",
                f"T.copy(u, O[t, ({index}) + 1, :])",
            ]
        elif kind == "views":
            lines = [
                f"T.copy(A[t, {index}, :], v[8:40])",
                "T.tile.add(w, w, w)",
                f"T.copy(u, O[t, ({index}) + 1, :])",
            ]
        else:
            destination = "a" if kind == "readonly" else "u"
            lines = [
                f"T.copy(A[t, {index}, :], a)",
                f"T.tile.add({destination}, a, u)",
                f"T.copy({destination}, O[t, ({index}) + 1, :])",
            ]
        return "".join(prefix + line + "\n" for line in lines)

    if kind == "consecutive":
        loop = (
            "                for i in T.Pipelined(trips, num_stages=num_stages):\n"
            + operations("i", 20)
            + "                for j in T.Pipelined(trips, num_stages=num_stages):\n"
            + operations("trips + j", 20)
        )
    elif kind == "nested":
        loop = (
            "                for epoch in T.serial(2):\n"
            "                    for i in T.Pipelined(trips, num_stages=num_stages):\n" + operations("epoch * trips + i", 24)
        )
    else:
        loop = "                for i in T.Pipelined(trips, num_stages=num_stages):\n" + operations("i", 20)
    source = f"""@T.prim_func
def main(A: T.Tensor((tiles, input_steps, input_elements), dtype),
         Scratch: T.Tensor((tiles, tile), dtype),
         O: T.Tensor((tiles, steps + 3, tile), dtype)):
    T.func_attr({{"tir.noalias": True}})
    with T.Kernel(1, threads=1, is_npu=True):
{declarations}        with T.Scope("V"):
            for t in T.serial(tiles):
{initialize}                T.copy(u, O[t, 0, :])
{loop}                T.copy(u, O[t, steps + 1, :])
"""
    return tvm.script.from_source(
        source,
        extra_vars={
            "T": T,
            "tiles": tiles,
            "tile": tile,
            "dtype": dtype,
            "trips": trips,
            "steps": steps,
            "input_steps": input_steps,
            "input_elements": input_elements,
            "num_stages": num_stages,
        },
    )


def make_dtype_version_kernel(trips=5, num_stages=3, *, manual_stages=True, developer=False):
    """Private f32[32]/f16[64] owner with an independent, initialized anchor.

    The manual lifetime spans three stages; actual N and physical byte pitch
    must be inspected after planning and injection, rather than inferred here.
    """
    if type(trips) is not int or trips < 1 or type(num_stages) is not int or num_stages not in (0, 3):
        raise ValueError("The private alias probe uses positive trips and bound3 or serial baseline0")
    if type(manual_stages) is not bool or type(developer) is not bool:
        raise ValueError("manual_stages and developer must be bool")
    if manual_stages and num_stages != 3:
        raise ValueError("The manual private lifetime requires requested bound3")
    allocation = "alloc_shared" if developer else "alloc_ub"
    scope = "shared" if developer else "shared.ub"
    calls = ["T.copy(A[i, :], u)", "T.tile.add(v, v, v)", "T.copy(u, O[i + 1, :])"]
    body = ""
    for stage, call in zip((0, 3, 3), calls):
        body += (
            f"                with T.Stage({stage}):\n                    with T.Task():\n                        {call}\n"
            if manual_stages
            else f"                {call}\n"
        )
    annotation = ', annotations={"enable_offset": True}' if manual_stages else ""
    source = f"""@T.prim_func
def main(A: T.Tensor((trips, 32), "float32"), Scratch: T.Tensor((32,), "float32"),
         O: T.Tensor((trips + 2, 32), "float32")):
    T.func_attr({{"tir.noalias": True}})
    with T.Kernel(1, threads=1, is_npu=True):
        anchor = T.{allocation}((32,), "float32")
        u = T.{allocation}((32,), "float32")
        v = T.decl_buffer((64,), "float16", data=u.data, scope="{scope}")
        with T.Scope("V"):
            T.copy(Scratch, anchor)
            T.copy(anchor, O[0, :])
            for i in T.Pipelined(trips, num_stages=num_stages{annotation}):
{body}
"""
    return tvm.script.from_source(source, extra_vars={"T": T, "trips": trips, "num_stages": num_stages})
