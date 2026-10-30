"""S7 compiler fixtures derived from S5 Vector and the existing copy_cv GEMM tests."""

from tilelang import language as T
from tilelang import tvm


def make_vector_sync_kernel(trips=257, versions=2, cores=8, dtype="float32", tile=64):
    """Keep S5 copy/add/copy, with disjoint per-core input and an untouched suffix."""
    if type(trips) is not int or trips < 0:
        raise ValueError("trips must be a nonnegative static integer")
    if type(versions) is not int or versions not in (1, 2, 3):
        raise ValueError("versions is the requested static bound1/2/3, not an actual-N assertion")
    if type(cores) is not int or cores not in (1, 8):
        raise ValueError("S7 Vector fixtures use one or eight physical workers")
    if dtype not in ("float16", "float32") or type(tile) is not int or tile not in (32, 64):
        raise ValueError("S7 Vector geometry uses the measured aligned f16/f32 tile32/64 scope")
    statements = (
        "T.copy(A[cid, i + 1, :], a)",
        "T.copy(B[cid, i + 1, :], b)",
        "T.tile.add(c, a, b)",
        "T.copy(c, O[cid, i + 1, :])",
    )
    body = ""
    for stage, statement in zip((0, 0, versions, versions), statements):
        body += (
            f"                with T.Stage({stage}):\n                    with T.Task():\n                        {statement}\n"
            if versions > 1
            else f"                {statement}\n"
        )
    annotation = ', annotations={"enable_offset": True}' if versions > 1 else ""
    source = f"""@T.prim_func
def main(A: T.Tensor((cores, trips + 2, tile), dtype),
         B: T.Tensor((cores, trips + 2, tile), dtype),
         O: T.Tensor((cores, trips + 2, tile), dtype)):
    T.func_attr({{"tir.noalias": True}})
    with T.Kernel(cores, threads=1, is_npu=True) as cid:
        anchor = T.alloc_ub((tile,), dtype)
        a = T.alloc_ub((tile,), dtype)
        b = T.alloc_ub((tile,), dtype)
        c = T.alloc_ub((tile,), dtype)
        with T.Scope("V"):
            T.tile.fill(anchor, 7.0)
            T.copy(anchor, O[cid, 0, :])
            for i in T.Pipelined(trips, num_stages=versions{annotation}):
{body}"""
    return tvm.script.from_source(
        source,
        extra_vars={
            "T": T,
            "trips": trips,
            "versions": versions,
            "cores": cores,
            "dtype": dtype,
            "tile": tile,
        },
    )


def make_cv_handoff_kernel(direction="v2c", trips=17, cores=4, *, manual_legacy=False, lowercase_workspace=False):
    """Use explicit GM workspace and the copy_cv tests' existing Cube GEMM structure.

    Every core owns one complete128x128 physical GM slot. Both Vector halves
    participate in each round. Caller supplies distinct A/B/workspace/O buffers;
    workspace has16 untouched trailing rows, and O has one untouched matrix.
    The fixed serial source introduces no joint C/V scheduling or extra tile API.
    """
    if direction not in ("v2c", "c2v"):
        raise ValueError("S7 handoff direction must be v2c or c2v")
    if type(trips) is not int or trips < 0 or type(cores) is not int or cores not in (1, 4):
        raise ValueError("S7 handoff uses static nonnegative rounds and one/four core groups")
    if type(manual_legacy) is not bool or type(lowercase_workspace) is not bool:
        raise ValueError("manual_legacy and lowercase_workspace must be bool")
    workspace_dtype = "float16" if direction == "v2c" else "float32"
    if direction == "v2c":
        scopes = """        with T.Scope("V"):
            for vr in T.serial(trips):
                T.copy(A[cid, vr, vid * 64:(vid + 1) * 64, :], u)
                T.copy(u, Workspace[cid, vid * 64:(vid + 1) * 64, :])
        with T.Scope("C"):
            for cr in T.serial(trips):
                T.copy(Workspace[cid, 0:128, :], a_l1)
                T.copy(B[cid, cr, :, :], b_l1)
                T.gemm_v0(a_l1, b_l1, c_l0, init=True)
                T.copy(c_l0, O[cid, cr, :, :])
"""
        if manual_legacy:
            scopes = scopes.replace(
                "                T.copy(Workspace[cid, 0:128, :], a_l1)",
                "                T.wait_cross_flag(0)\n"
                "                T.copy(Workspace[cid, 0:128, :], a_l1)\n"
                '                T.set_cross_flag("MTE2", 1)',
            ).replace(
                "                T.copy(u, Workspace[cid, vid * 64:(vid + 1) * 64, :])",
                "                T.copy(u, Workspace[cid, vid * 64:(vid + 1) * 64, :])\n"
                '                T.set_cross_flag("MTE3", 0)\n'
                "                T.wait_cross_flag(1)",
            )
    else:
        scopes = """        with T.Scope("C"):
            for cr in T.serial(trips):
                T.copy(A[cid, cr, :, :], a_l1)
                T.copy(B[cid, cr, :, :], b_l1)
                T.gemm_v0(a_l1, b_l1, c_l0, init=True)
                T.copy(c_l0, Workspace[cid, 0:128, :])
        with T.Scope("V"):
            for vr in T.serial(trips):
                T.copy(Workspace[cid, vid * 64:(vid + 1) * 64, :], u)
                T.copy(u, O[cid, vr, vid * 64:(vid + 1) * 64, :])
"""
        if manual_legacy:
            scopes = scopes.replace(
                "                T.copy(c_l0, Workspace[cid, 0:128, :])",
                "                T.copy(c_l0, Workspace[cid, 0:128, :])\n"
                '                T.set_cross_flag("FIX", 0)\n'
                "                T.wait_cross_flag(1)",
            ).replace(
                "                T.copy(Workspace[cid, vid * 64:(vid + 1) * 64, :], u)",
                "                T.wait_cross_flag(0)\n"
                "                T.copy(Workspace[cid, vid * 64:(vid + 1) * 64, :], u)\n"
                '                T.set_cross_flag("MTE2", 1)',
            )
    source = f"""@T.prim_func
def main(A: T.Tensor((cores, trips, 128, 128), "float16"),
         B: T.Tensor((cores, trips, 128, 128), "float16"),
         Workspace: T.Tensor((cores, 144, 128), workspace_dtype),
         O: T.Tensor((cores, trips + 1, 128, 128), "float32")):
    T.func_attr({{"tir.noalias": True}})
    with T.Kernel(cores, is_npu=True) as (cid, vid):
        u = T.alloc_ub((64, 128), workspace_dtype)
        a_l1 = T.alloc_L1((128, 128), "float16")
        b_l1 = T.alloc_L1((128, 128), "float16")
        c_l0 = T.alloc_L0C((128, 128), "float32")
{scopes}"""
    if lowercase_workspace:
        source = source.replace("Workspace", "workspace")
    return tvm.script.from_source(
        source,
        extra_vars={"T": T, "trips": trips, "cores": cores, "workspace_dtype": workspace_dtype},
    )
