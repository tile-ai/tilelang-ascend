"""Shared S4 copy/add/copy fixture, derived from the existing S3 Vector fixture.

The tail uses its own complete, aligned short tile. It does not require partial
or dynamic DMA support, and allocation identity does not imply physical reuse.
"""

from tilelang import language as T
from tilelang import tvm


def vector_pass_configs(strategy="constraint"):
    return {
        "tl.ascend_pipeline_planner": strategy,
        "tl.ascend_auto_sync": True,
        "tl.ascend_memory_planning": True,
        "tl.ascend_auto_cross_core_sync": False,
        "tl.ascend_auto_sync_vs": False,
    }


def make_vector_kernel(total=192, dtype="float32", inplace=False, noalias=True):
    """Build static Vector add; callers must supply disjoint A/B/O with noalias=True.

    The device runner meets this contract with three independent allocations.
    Optional inplace changes only UB storage, and noalias=False is for rejection tests.
    """
    if total not in (160, 192) or isinstance(total, bool):
        raise ValueError("S4 fixture supports full192 or aligned-tail160")
    if dtype not in ("float16", "float32"):
        raise ValueError("S4 fixture supports float16/float32")
    if not isinstance(inplace, bool):
        raise ValueError("inplace must be bool")
    if not isinstance(noalias, bool):
        raise ValueError("noalias must be bool")
    trips, tail = divmod(total, 64)
    destination = "a" if inplace else "c"
    main_allocation = "" if inplace else "            c = T.alloc_ub((64,), dtype)\n"
    tail_allocation = ""
    tail_body = ""
    if tail:
        tail_destination = "tail_a" if inplace else "tail_c"
        tail_allocation = "            tail_a = T.alloc_ub((32,), dtype)\n            tail_b = T.alloc_ub((32,), dtype)\n"
        if not inplace:
            tail_allocation += "            tail_c = T.alloc_ub((32,), dtype)\n"
        tail_body = f"""                for last in T.Pipelined(1, num_stages=1):
                    T.copy(A[128 + last * 32:160 + last * 32], tail_a)
                    T.copy(B[128 + last * 32:160 + last * 32], tail_b)
                    T.tile.add({tail_destination}, tail_a, tail_b)
                    T.copy({tail_destination}, O[128 + last * 32:160 + last * 32])
"""
    source = f"""@T.prim_func
def main(A: T.Tensor((total,), dtype), B: T.Tensor((total,), dtype),
         O: T.Tensor((total,), dtype)):
        T.func_attr({{"tir.noalias": {noalias!r}}})
        with T.Kernel(1, threads=1, is_npu=True):
            a = T.alloc_ub((64,), dtype)
            b = T.alloc_ub((64,), dtype)
{main_allocation}{tail_allocation}            with T.Scope("V"):
                for block in T.Pipelined(trips, num_stages=1):
                    T.copy(A[block * 64:(block + 1) * 64], a)
                    T.copy(B[block * 64:(block + 1) * 64], b)
                    T.tile.add({destination}, a, b)
                    T.copy({destination}, O[block * 64:(block + 1) * 64])
{tail_body}"""
    return tvm.script.from_source(source, extra_vars={"T": T, "total": total, "dtype": dtype, "trips": trips})
