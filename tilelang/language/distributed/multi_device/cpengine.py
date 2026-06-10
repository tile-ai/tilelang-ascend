"""The language interface for tl programs."""

from tvm import tir


def cpengine_cpasync(*args):
    return tir.call_intrin("int32", tir.op.Op.get("tl.CpengineCpAsync"), *args)
