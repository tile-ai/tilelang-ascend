"""Complete original factories consume the restored TA loop and swizzle APIs."""

import ast
from functools import lru_cache
import hashlib
import importlib.util
import inspect
from pathlib import Path

import pytest

import tilelang
from tilelang import tvm
import tilelang.ascend.a2a3.language as T
from tilelang.ascend.a2a3 import compiler
from tilelang.language.annotations import use_swizzle as shared_swizzle

TARGET = {"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]}


@lru_cache(maxsize=2)
def _factory(name):
    path = Path(__file__).parent / "fixtures" / f"a2a3_{name}.py"
    spec = importlib.util.spec_from_file_location(f"a2a3_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.matmul if name == "swizzle" else module.flash_attention_fwd


def _lower(name, interval=1):
    factory = _factory(name)
    args = (8192, 1024, 8192, 128, 256, 64, 256, 2, 2) if name == "swizzle" else (4, 4096, 16, 16, 128)
    kwargs = {} if name == "swizzle" else {"cross_interval": interval}
    func = factory._get_tir_with_target(tvm.target.Target(TARGET), *args, **kwargs)
    observed = []

    @tvm.instrument.pass_instrument
    class CrossCoreInputs:
        def run_before_pass(self, mod, info):
            if info.name != "tl.A2A3CrossCorePipeline":
                return
            for function in mod.functions.values():
                tvm.tirx.stmt_functor.post_order_visit(
                    function.body,
                    lambda node: (
                        observed.append(int(node.annotations["tl_cross_interval"]))
                        if isinstance(node, tvm.tirx.For) and "num_stages" in node.annotations and "tl_cross_interval" in node.annotations
                        else None
                    ),
                )

    with tvm.transform.PassContext(config=factory.pass_configs or {}, instruments=[CrossCoreInputs()]):
        artifact = tilelang.lower(func, target=TARGET)
    if name == "cross_interval":
        assert observed == [interval]
    return func, artifact


@pytest.mark.parametrize(
    "name,expected",
    [
        ("swizzle", "368e4a01b9256174852c10ee06dfa11311254456f724532cdf33f34f0f6d8a3a"),
        ("cross_interval", "0bb3f83c031cf043330868a58ec2305ab92e19eb7f9009028c22c5dfec4cf089"),
    ],
)
def test_full_original_factory_decorator_ast(name, expected):
    path = Path(__file__).parent / "fixtures" / f"a2a3_{name}.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef))
    assert hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() == expected


@pytest.mark.parametrize("off,direction,in_loop", [(1, 0, False), (3, 0, False), (2, 1, True)])
def test_complete_coordinate_transform_call(off, direction, in_loop):
    cid = tvm.tirx.Var("cid", "int32")
    args = (cid, 8192, 1024, 8192, 128, 256, off, direction, in_loop)
    call = T.use_swizzle(*args)
    assert call.op.name == "tl.ascend_use_swizzle" and call.dtype == "int32"
    assert call.args[0].value == f"thread_block_swizzle<8192, 1024, 8192, 128, 256, {off}, {direction}>"
    assert call.args[1].same_as(cid) and int(call.args[2]) == 256
    tvm.ir.assert_structural_equal(call, T.npu_use_swizzle(*args))
    # in_loop is a reserved TA argument, not a new scheduling mode.
    tvm.ir.assert_structural_equal(call, T.use_swizzle(*args[:-1], not in_loop))


@pytest.mark.parametrize("m,n", [(1025, 1024), (1024, 1025)])
def test_source_divisibility_assertion(m, n):
    with pytest.raises(AssertionError, match="m and n must be divisible"):
        T.use_swizzle(tvm.tirx.Var("cid", "int32"), m, n, 512, 128, 128)


def test_backend_coordinate_transform_does_not_replace_shared_annotation():
    assert list(inspect.signature(T.use_swizzle).parameters) == ["cid", "m", "n", "k", "block_m", "block_n", "off", "dir", "in_loop"]
    assert list(inspect.signature(shared_swizzle).parameters) == ["panel_size", "order", "enable"]


@pytest.mark.parametrize("name,interval", [("swizzle", 1), ("cross_interval", 1), ("cross_interval", 2)])
def test_complete_original_factory_native_consumers_and_bisheng(name, interval):
    func, artifact = _lower(name, interval)
    if name == "swizzle":
        assert "thread_block_swizzle<8192, 1024, 8192, 128, 256, 3, 0>" in artifact.kernel_source
        assert len(artifact.params) == 3
    else:
        loops = []
        tvm.tirx.stmt_functor.post_order_visit(
            func.body, lambda node: loops.append(node) if isinstance(node, tvm.tirx.For) and "num_stages" in node.annotations else None
        )
        assert len(loops) == 1 and int(loops[0].annotations["tl_cross_interval"]) == interval
        assert len(artifact.params) == 7
        assert "CrossCoreSetFlag" in artifact.kernel_source
    try:
        config = compiler.resolve_compile_config(tvm.target.Target(TARGET), _factory(name).pass_configs)
    except FileNotFoundError as error:
        pytest.skip(str(error))
    library = compiler.LibraryGenerator(config)
    library.update_lib_code(artifact.kernel_source)
    library.compile_lib(timeout=120)
    assert Path(library.get_lib_path()).is_file()
