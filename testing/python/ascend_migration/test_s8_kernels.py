"""S8 compiler fixtures derived from the existing GEMM, row-reduce and S7 C/V tests.

Every GM argument is supplied by the caller with disjoint storage.  The mixed
legacy reference adds the already validated balanced READY/FREE protocol; auto
receives the same compute and actor order without handwritten synchronization.
"""

from tilelang import language as T
from tilelang import tvm


def _domain(trips, cores):
    if type(trips) is not int or trips < 0:
        raise ValueError("S8 fixtures require a nonnegative static trip count")
    if type(cores) is not int or cores not in (1, 4):
        raise ValueError("S8 fixtures use one or four core groups")


def make_vector_kernel(trips=3, cores=1):
    """Small finite Copy/Add/Copy probe for backend connectivity, without pipeline ABI."""
    _domain(trips, cores)
    source = """@T.prim_func
def main(A: T.Tensor((cores, input_trips, 128), "float32"),
         B: T.Tensor((cores, input_trips, 128), "float32"),
         O: T.Tensor((cores, trips + 1, 128), "float32")):
    T.func_attr({"tir.noalias": True})
    with T.Kernel(cores, is_npu=True) as (cid, vid):
        a = T.alloc_ub((64,), "float32")
        b = T.alloc_ub((64,), "float32")
        c = T.alloc_ub((64,), "float32")
        with T.Scope("V"):
            for iteration in T.serial(trips):
                T.copy(A[cid, iteration, vid * 64:(vid + 1) * 64], a)
                T.copy(B[cid, iteration, vid * 64:(vid + 1) * 64], b)
                T.tile.add(c, a, b)
                T.copy(c, O[cid, iteration, vid * 64:(vid + 1) * 64])
"""
    return tvm.script.from_source(source, extra_vars={"T": T, "trips": trips, "cores": cores, "input_trips": max(trips, 1)})


def make_gemm_kernel(trips=3, cores=1, *, m=128, n=128, k=128, k_l0=128, noalias=True):
    """Independent f16 GEMMs with f32 L0C/output; no accumulator carry."""
    _domain(trips, cores)
    if any(type(value) is not int or value <= 0 for value in (m, n, k, k_l0)):
        raise ValueError("GEMM tile dimensions must be positive static integers")
    if type(noalias) is not bool:
        raise ValueError("noalias must be bool")
    source = f"""@T.prim_func
def main(A: T.Tensor((cores, input_trips, m, k), "float16"),
         B: T.Tensor((cores, input_trips, k, n), "float16"),
         O: T.Tensor((cores, trips + 1, m, n), "float32")):
    {"T.func_attr({'tir.noalias': True})" if noalias else ""}
    with T.Kernel(cores, is_npu=True) as (cid, vid):
        a_l1 = T.alloc_L1((m, k), "float16")
        b_l1 = T.alloc_L1((k, n), "float16")
        c_l0 = T.alloc_L0C((m, n), "float32")
        with T.Scope("C"):
            for iteration in T.serial(trips):
                T.copy(A[cid, iteration, :, :], a_l1)
                T.copy(B[cid, iteration, :, :], b_l1)
                T.gemm_v0(a_l1, b_l1, c_l0, init=True, kL0Size=k_l0)
                T.copy(c_l0, O[cid, iteration, :, :])
"""
    return tvm.script.from_source(
        source,
        extra_vars={
            "T": T,
            "trips": trips,
            "cores": cores,
            "input_trips": max(trips, 1),
            "m": m,
            "n": n,
            "k": k,
            "k_l0": k_l0,
        },
    )


def make_reduce_kernel(trips=3, cores=1, *, rows=8, columns=64, noalias=True):
    """Each Vector actor reduces eight rows to eight f32 values (32 bytes)."""
    _domain(trips, cores)
    if type(rows) is not int or rows <= 0 or type(columns) is not int or columns <= 0:
        raise ValueError("Reduction rows/columns must be positive static integers")
    if type(noalias) is not bool:
        raise ValueError("noalias must be bool")
    source = f"""@T.prim_func
def main(A: T.Tensor((cores, input_trips, 2 * rows, columns), "float32"),
         O: T.Tensor((cores, trips + 1, 2 * rows), "float32")):
    {"T.func_attr({'tir.noalias': True})" if noalias else ""}
    with T.Kernel(cores, is_npu=True) as (cid, vid):
        u = T.alloc_ub((rows, columns), "float32")
        total = T.alloc_ub((rows,), "float32")
        with T.Scope("V"):
            for iteration in T.serial(trips):
                T.copy(A[cid, iteration, vid * rows:(vid + 1) * rows, :], u)
                T.reduce_sum(u, total, dim=-1, clear=True)
                T.copy(total, O[cid, iteration, vid * rows:(vid + 1) * rows])
"""
    return tvm.script.from_source(
        source,
        extra_vars={
            "T": T,
            "trips": trips,
            "cores": cores,
            "rows": rows,
            "columns": columns,
            "input_trips": max(trips, 1),
        },
    )


def make_mixed_kernel(trips=3, cores=1, *, operation="reduce", manual_legacy=False, noalias=True):
    """GEMM -> caller f32 GM slot -> Vector, retaining the final workspace value."""
    _domain(trips, cores)
    if operation not in ("add", "reduce"):
        raise ValueError("Mixed operation must be add or reduce")
    if type(manual_legacy) is not bool or type(noalias) is not bool:
        raise ValueError("manual_legacy and noalias must be bool")
    extra_parameter = '         D: T.Tensor((cores, input_trips, 128, 128), "float32"),\n' if operation == "add" else ""
    output_shape = "(cores, trips + 1, 128, 128)" if operation == "add" else "(cores, trips + 1, 128)"
    allocation = '        d = T.alloc_ub((64, 128), "float32")' if operation == "add" else '        total = T.alloc_ub((64,), "float32")'
    compute = (
        "                T.copy(D[cid, vr, vid * 64:(vid + 1) * 64, :], d)\n"
        "                T.tile.add(u, u, d)\n"
        "                T.copy(u, O[cid, vr, vid * 64:(vid + 1) * 64, :])"
        if operation == "add"
        else "                T.reduce_sum(u, total, dim=-1, clear=True)\n"
        "                T.copy(total, O[cid, vr, vid * 64:(vid + 1) * 64])"
    )
    producer_events = '                T.set_cross_flag("FIX", 0)\n                T.wait_cross_flag(1)\n' if manual_legacy else ""
    consumer_wait = "                T.wait_cross_flag(0)\n" if manual_legacy else ""
    consumer_release = '                T.set_cross_flag("MTE2", 1)\n' if manual_legacy else ""
    source = f"""@T.prim_func
def main(A: T.Tensor((cores, input_trips, 128, 128), "float16"),
         B: T.Tensor((cores, input_trips, 128, 128), "float16"),
{extra_parameter}         Workspace: T.Tensor((cores, 144, 128), "float32"),
         O: T.Tensor({output_shape}, "float32")):
    {"T.func_attr({'tir.noalias': True})" if noalias else ""}
    with T.Kernel(cores, is_npu=True) as (cid, vid):
        a_l1 = T.alloc_L1((128, 128), "float16")
        b_l1 = T.alloc_L1((128, 128), "float16")
        c_l0 = T.alloc_L0C((128, 128), "float32")
        u = T.alloc_ub((64, 128), "float32")
{allocation}
        with T.Scope("C"):
            for cr in T.serial(trips):
                T.copy(A[cid, cr, :, :], a_l1)
                T.copy(B[cid, cr, :, :], b_l1)
                T.gemm_v0(a_l1, b_l1, c_l0, init=True)
                T.copy(c_l0, Workspace[cid, 0:128, :])
{producer_events}        with T.Scope("V"):
            for vr in T.serial(trips):
{consumer_wait}                T.copy(Workspace[cid, vid * 64:(vid + 1) * 64, :], u)
{consumer_release}{compute}
"""
    return tvm.script.from_source(source, extra_vars={"T": T, "trips": trips, "cores": cores, "input_trips": max(trips, 1)})
