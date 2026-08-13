# # """Direct debug script for reduce_abssum/reduce_absmax/cumsum on Ascend.

# # This is intentionally not a pytest-style file.  It is meant to be run directly
# # while developing/fixing the issue.

# # Examples:

#     python testing/python/language/test_tilelang_ascend_language_reduce_abssum_absmax_cumsum_issue.py
#     python testing/python/language/test_tilelang_ascend_language_reduce_abssum_absmax_cumsum_issue.py reduce_abssum ascendc
#     python testing/python/language/test_tilelang_ascend_language_reduce_abssum_absmax_cumsum_issue.py --runtime reduce_absmax ascendc
#     python testing/python/language/test_tilelang_ascend_language_reduce_abssum_absmax_cumsum_issue.py --runtime reduce_abssum ascendc
#     python testing/python/language/test_tilelang_ascend_language_reduce_abssum_absmax_cumsum_issue.py --runtime cumsum ascendc


# # What it prints:

# # * which Python function T.reduce_sum/reduce_abssum/reduce_absmax/cumsum binds to
# # * the PrimFunc produced by @T.prim_func
# # * each important lower pass, especially the pass that fails
# # * generated kernel_source if lowering reaches codegen
# # """

from __future__ import annotations

import os
import sys
import inspect
import traceback
from pathlib import Path
from typing import Callable


def _bootstrap_repo_paths() -> Path:
    """Prefer this checkout's Python code and build/libtvm when run directly."""

    repo = Path(__file__).resolve().parents[3]
    tvm_python = repo / "3rdparty" / "tvm" / "python"
    tvm_build = repo / "build" / "tvm"

    os.environ.setdefault("TVM_LIBRARY_PATH", str(tvm_build))
    for path in (repo, tvm_python):
        path_str = str(path)
        while path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)

    return repo


REPO_ROOT = _bootstrap_repo_paths()

import tilelang  # noqa: E402
import tilelang.transform  # noqa: E402
from tilelang import language as T  # noqa: E402
from tilelang.engine.lower import device_codegen, extrac_params  # noqa: E402
from tilelang.engine.phase import allow_vectorize  # noqa: E402
from tilelang.utils.target import check_npu_availability, determine_platform  # noqa: E402
import tvm  # noqa: E402
from tvm import tir  # noqa: E402

ALL_TARGETS = ["ascendc", "pto"]
REDUCE_OPS = {"reduce_abssum", "reduce_absmax"}
CONTROL_REDUCE_OPS = {"reduce_sum", "reduce_max", "reduce_min"}
ALL_OPS = ["reduce_sum", "reduce_max", "reduce_min", "reduce_abssum", "reduce_absmax", "cumsum"]


RUNTIME_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def line(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def subline(title: str) -> None:
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)


def source_of(obj) -> str:
    try:
        return inspect.getsourcefile(obj) or "<unknown>"
    except TypeError:
        return "<unknown>"


def print_api_binding() -> None:
    line("API binding: which Python implementation does T.* use?")
    for name in ALL_OPS:
        obj = getattr(T, name)
        print(f"T.{name}")
        print(f"  object : {obj}")
        print(f"  module : {getattr(obj, '__module__', '<unknown>')}")
        print(f"  file   : {source_of(obj)}")
        try:
            src = inspect.getsource(obj).strip().splitlines()
            print("  source :")
            for src_line in src[:8]:
                print(f"    {src_line}")
            if len(src) > 8:
                print("    ...")
        except (OSError, TypeError):
            print("  source : <not available>")


def make_reduce_kernel(op_name: str):
    op = getattr(T, op_name)

    @T.prim_func
    def main(
        src: T.Tensor([8, 64], "float"),  # type: ignore
        out: T.Tensor([8], "float"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (_, vid):
            src_ub = T.alloc_ub([8, 64], "float")
            out_ub = T.alloc_ub([8], "float")

            if vid == 0:
                T.copy(src, src_ub)
                op(src_ub, out_ub, dim=-1)
                T.copy(out_ub, out)

    return main


def make_cumsum_kernel(dim: int = -1, reverse: bool = False):
    @T.prim_func
    def main(
        src: T.Tensor([8, 64], "float"),  # type: ignore
        out: T.Tensor([8, 64], "float"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (_, vid):
            src_ub = T.alloc_ub([8, 64], "float")
            out_ub = T.alloc_ub([8, 64], "float")

            if vid == 0:
                T.copy(src, src_ub)
                T.cumsum(src_ub, out_ub, dim=dim, reverse=reverse)
                T.copy(out_ub, out)

    return main


def make_program(op_name: str):
    if op_name == "cumsum":
        return make_cumsum_kernel(dim=-1, reverse=False)
    if op_name in REDUCE_OPS or op_name in CONTROL_REDUCE_OPS:
        return make_reduce_kernel(op_name)
    raise ValueError(f"unknown op: {op_name}")


def print_prim_func(func: tir.PrimFunc) -> None:
    subline("PrimFunc generated by @T.prim_func")
    print("type:", type(func))
    print("attrs:", func.attrs)
    print("params:", list(func.params))
    print("\nscript:")
    print(func.script())


def print_module(mod: tvm.IRModule, title: str) -> None:
    subline(title)
    try:
        print(mod.script())
    except Exception:  # pylint: disable=broad-except
        print(mod)


def run_pass(mod: tvm.IRModule, name: str, pass_func: Callable[[tvm.IRModule], tvm.IRModule]):
    print(f"\n>>> PASS START: {name}")
    try:
        new_mod = pass_func(mod)
    except Exception as err:  # pylint: disable=broad-except
        print(f"<<< PASS FAILED: {name}")
        print(f"exception type: {type(err).__name__}")
        print(f"exception msg : {err}")
        print("\nPython traceback:")
        traceback.print_exc(file=sys.stdout)
        raise
    print(f"<<< PASS OK: {name}")
    return new_mod


def phase1_passes(target: tvm.target.Target):
    return [
        ("InjectTmpBuffer", lambda m: tilelang.transform.InjectTmpBuffer(target)(m)),
        ("AscendInferBufferScope", lambda m: tilelang.transform.AscendInferBufferScope()(m)),
        ("AscendVidReduction", lambda m: tilelang.transform.AscendVidReduction()(m)),
        ("BufferShapeCollector", lambda m: tilelang.transform.BufferShapeCollector()(m)),
        ("tir.BindTarget", lambda m: tir.transform.BindTarget(target)(m)),
        ("HostProcesser", lambda m: tilelang.transform.HostProcesser()(m)),
        ("tir.Simplify before LowerTileOp", lambda m: tir.transform.Simplify()(m)),
        ("AscendLowerParallelToVector", lambda m: tilelang.transform.AscendLowerParallelToVector()(m)),
        ("LayoutInference", lambda m: tilelang.transform.LayoutInference()(m)),
        ("CollectBufferShapes", lambda m: tilelang.transform.CollectBufferShapes()(m)),
        ("LowerTileOp", lambda m: tilelang.transform.LowerTileOp()(m)),
        (
            "AscendTailMaskPropagation",
            lambda m: tilelang.transform.AscendTailMaskPropagation(rewrite_reduce=target.model in {"ascendc", "pto", "auto"})(m),
        ),
        ("AscendWorkspaceReduction", lambda m: tilelang.transform.AscendWorkspaceReduction()(m)),
        ("LegalizeVectorizedLoop", lambda m: tilelang.transform.LegalizeVectorizedLoop()(m)),
        ("LegalizeSafeMemoryAccess", lambda m: tilelang.transform.LegalizeSafeMemoryAccess()(m)),
        ("tir.Simplify after legalize", lambda m: tir.transform.Simplify()(m)),
    ]


def phase2_passes(target: tvm.target.Target, platform: str):
    pass_ctx = tilelang.transform.get_pass_context()
    return [
        ("tir.PlanAndUpdateBufferAllocationLocation", lambda m: tir.transform.PlanAndUpdateBufferAllocationLocation()(m)),
        ("CrossCorePipeline", lambda m: tilelang.transform.CrossCorePipeline()(m)),
        ("CombineCV", lambda m: tilelang.transform.CombineCV()(m)),
        ("PipelinePlanning", lambda m: tilelang.transform.PipelinePlanning()(m)),
        ("InjectSoftwarePipeline", lambda m: tilelang.transform.InjectSoftwarePipeline()(m)),
        ("AscendLowerOpaqueBlock", lambda m: tilelang.transform.AscendLowerOpaqueBlock()(m)),
        ("tir.NarrowDataType(32)", lambda m: tir.transform.NarrowDataType(32)(m)),
        ("ConfigIndexBitwidth", lambda m: tilelang.transform.ConfigIndexBitwidth()(m)),
        ("Flatten2DBuffer", lambda m: tilelang.transform.Flatten2DBuffer()(m)),
        ("FlattenBuffer", lambda m: tilelang.transform.FlattenBuffer()(m)),
        ("tir.Simplify before VectorizeLoop", lambda m: tir.transform.Simplify()(m)),
        (
            "VectorizeLoop",
            lambda m: tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(m),
        ),
        (
            "AscendStorageRewrite",
            lambda m: tilelang.transform.AscendStorageRewrite(is_npu=check_npu_availability())(m),
        ),
        ("tir.UnrollLoop", lambda m: tir.transform.UnrollLoop()(m)),
        ("tir.RenormalizeSplitPattern", lambda m: tir.transform.RenormalizeSplitPattern()(m)),
        ("tir.Simplify after unroll", lambda m: tir.transform.Simplify()(m)),
        ("tir.RemoveNoOp", lambda m: tir.transform.RemoveNoOp()(m)),
        ("tir.RewriteUnsafeSelect", lambda m: tir.transform.RewriteUnsafeSelect()(m)),
        ("tir.HoistIfThenElse", lambda m: tir.transform.HoistIfThenElse()(m)),
        ("AscendMemoryPlanning", lambda m: tilelang.transform.AscendMemoryPlanning()(m)),
        ("AscendSyncInsert", lambda m: tilelang.transform.AscendSyncInsert(target, platform)(m)),
        ("AscendSyncInsertVS", lambda m: tilelang.transform.AscendSyncInsertVS(target, platform)(m)),
    ]


def lower_step_by_step(func: tir.PrimFunc, target_name: str) -> None:
    target = tvm.target.Target({"kind": "llvm", "model": target_name})
    platform = determine_platform("auto")
    params = extrac_params(func)
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})

    print(f"target  : {target}")
    print(f"platform: {platform}")
    print("params  :")
    for param in params:
        print(f"  {param}")

    print_module(mod, "Initial IRModule before lowering")

    line("Phase 1: LowerAndLegalize, expanded pass by pass")
    for name, pass_func in phase1_passes(target):
        if name == "LowerTileOp":
            print_module(mod, "IRModule immediately before LowerTileOp")
        mod = run_pass(mod, name, pass_func)
        if name == "LowerTileOp":
            print_module(mod, "IRModule immediately after LowerTileOp")

    line("Phase 2: OptimizeForTarget, expanded pass by pass")
    for name, pass_func in phase2_passes(target, platform):
        mod = run_pass(mod, name, pass_func)

    print_module(mod, "IRModule before device_codegen")

    line("Device codegen")
    try:
        codegen_mod = device_codegen(mod, target, platform)
    except Exception as err:  # pylint: disable=broad-except
        print("device_codegen FAILED")
        print(f"exception type: {type(err).__name__}")
        print(f"exception msg : {err}")
        traceback.print_exc(file=sys.stdout)
        raise

    source = codegen_mod.get_source()
    print("device_codegen OK")
    print("\nGenerated kernel_source:")
    print(source)


def run_case(op_name: str, target_name: str) -> bool:
    line(f"CASE op={op_name}, target={target_name}")
    tilelang.cache.clear_cache()
    func = make_program(op_name)
    print_prim_func(func)

    try:
        lower_step_by_step(func, target_name)
    except Exception:
        print(f"\nCASE RESULT: FAILED op={op_name}, target={target_name}")
        return False

    print(f"\nCASE RESULT: OK op={op_name}, target={target_name}")
    return True


def compile_runtime_kernel(op_name: str, target_name: str):
    """Compile one direct NPU runtime kernel for correctness testing."""

    func = make_program(op_name)
    return tilelang.compile(
        func,
        out_idx=[-1],
        pass_configs=RUNTIME_PASS_CONFIGS,
        target=target_name,
    )


def reference_result(op_name: str, src):
    if op_name == "reduce_sum":
        return src.sum(dim=-1)
    if op_name == "reduce_max":
        return src.max(dim=-1).values
    if op_name == "reduce_min":
        return src.min(dim=-1).values
    if op_name == "reduce_abssum":
        return src.abs().sum(dim=-1)
    if op_name == "reduce_absmax":
        return src.abs().amax(dim=-1)
    if op_name == "cumsum":
        return src.cumsum(dim=-1)
    raise ValueError(f"unknown op: {op_name}")


def run_runtime_case(op_name: str, target_name: str) -> bool:
    """Compile and run the kernel, then compare with PyTorch on NPU."""

    line(f"RUNTIME CASE op={op_name}, target={target_name}")
    if target_name != "ascendc":
        print("runtime test is intended for ascendc first; pto codegen can still be checked with the normal mode")
        return False

    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        print(f"torch import failed: {err}")
        return False

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        print("torch.npu is not available in this environment")
        return False

    tilelang.cache.clear_cache()
    kernel = compile_runtime_kernel(op_name, target_name)
    print(f"{kernel.get_kernel_source()}")
    torch.manual_seed(0)
    src = torch.randn(8, 64, dtype=torch.float32).npu()
    # Add deterministic edge values so abs reductions check negative and zero data.
    src[0, 0] = -7.0
    src[1, 3] = 0.0
    torch.npu.synchronize()

    got = kernel(src)
    torch.npu.synchronize()
    expected = reference_result(op_name, src)
    print(f"expected: {expected}\n")
    torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)
    print("runtime output matched PyTorch reference")
    print("got shape:", tuple(got.shape))
    print("got sample:", got.flatten()[:8].detach().cpu())
    print(f"RUNTIME RESULT: OK op={op_name}, target={target_name}")
    return True


def test_reduce_abssum_runtime():
    assert run_runtime_case("reduce_abssum", "ascendc")


def test_reduce_absmax_runtime():
    assert run_runtime_case("reduce_absmax", "ascendc")


def test_cumsum_runtime():
    assert run_runtime_case("cumsum", "ascendc")


def parse_args() -> tuple[bool, list[str], list[str]]:
    args = sys.argv[1:]
    runtime = False
    if "--runtime" in args:
        runtime = True
        args = [arg for arg in args if arg != "--runtime"]

    if not args:
        return runtime, ["reduce_abssum", "reduce_absmax", "cumsum"], ALL_TARGETS
    if len(args) != 2:
        print("Usage:")
        print(f"  python {Path(__file__).name}")
        print(f"  python {Path(__file__).name} --runtime reduce_abssum ascendc")
        print(f"  python {Path(__file__).name} reduce_abssum ascendc")
        print(f"  python {Path(__file__).name} reduce_absmax pto")
        print(f"  python {Path(__file__).name} cumsum ascendc")
        raise SystemExit(2)

    op_name, target_name = args
    if op_name not in ALL_OPS:
        raise SystemExit(f"unknown op {op_name}, expected one of {ALL_OPS}")
    if target_name not in ALL_TARGETS:
        raise SystemExit(f"unknown target {target_name}, expected one of {ALL_TARGETS}")
    return runtime, [op_name], [target_name]


def main() -> int:
    print("repo root:", REPO_ROOT)
    print("tvm python:", tvm.__file__)
    print("tilelang language:", T.__file__)
    print("TVM_LIBRARY_PATH:", os.environ.get("TVM_LIBRARY_PATH"))

    print_api_binding()

    runtime, op_names, target_names = parse_args()
    failed = 0
    for op_name in op_names:
        for target_name in target_names:
            ok = run_runtime_case(op_name, target_name) if runtime else run_case(op_name, target_name)
            failed += 0 if ok else 1
    return failed


if __name__ == "__main__":
    raise SystemExit(main())


# # 复现原issue的代码
# import tilelang
# from tilelang import language as T


# op_name, target = "reduce_abssum", "ascendc"
# op_name, target = "reduce_absmax", "ascendc"
# op_name, target = "cumsum", "ascendc"


# def make_reduce_kernel():
#     op = getattr(T, op_name)

#     @T.prim_func
#     def main(
#         src: T.Tensor([8, 64], "float"),
#         out: T.Tensor([8], "float"),
#     ):
#         with T.Kernel(1, is_npu=True) as (_, vid):
#             src_ub = T.alloc_ub([8, 64], "float")
#             out_ub = T.alloc_ub([8], "float")

#             if vid == 0:
#                 T.copy(src, src_ub)
#                 op(src_ub, out_ub, dim=-1)
#                 T.copy(out_ub, out)

#     return main


# def make_cumsum_kernel():
#     @T.prim_func
#     def main(
#         src: T.Tensor([8, 64], "float"),
#         out: T.Tensor([8, 64], "float"),
#     ):
#         with T.Kernel(1, is_npu=True) as (_, vid):
#             src_ub = T.alloc_ub([8, 64], "float")
#             out_ub = T.alloc_ub([8, 64], "float")

#             if vid == 0:
#                 T.copy(src, src_ub)
#                 T.cumsum(src_ub, out_ub, dim=-1)
#                 T.copy(out_ub, out)

#     return main


# program = make_cumsum_kernel() if op_name == "cumsum" else make_reduce_kernel()
# print(tilelang.lower(program, target=target).kernel_source)
