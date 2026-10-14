import pytest
import torch
import tvm
from tvm import tir

import tilelang
import tilelang.language as T


N = 256
ROWS = 4
COLS = 64
EPS = 1.0e-5
SCALE = 1.25
BIAS = 0.125

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

UNARY_OPS = {
    "exp": lambda value: T.exp(value),
    "log": lambda value: T.log(value),
    "sqrt": lambda value: T.sqrt(value),
    "rsqrt": lambda value: T.rsqrt(value),
    "abs": lambda value: T.abs(value),
}

UNARY_SOURCE_MARKERS = {
    "ascendc": {
        "exp": "AscendC::Exp",
        "log": "AscendC::Ln",
        "sqrt": "AscendC::Sqrt",
        "rsqrt": "AscendC::Rsqrt",
        "abs": "AscendC::Abs",
    },
    "pto": {
        "exp": "TEXP(",
        "log": "TLOG(",
        "sqrt": "TSQRT(",
        "rsqrt": "TRSQRT(",
        "abs": "TABS(",
    },
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.disable_cache()
    yield


def _compound_unary_1d(dtype, op_name):
    op = UNARY_OPS[op_name]

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _vid):
            a_ub = T.alloc_ub((N,), dtype)
            b_ub = T.alloc_ub((N,), dtype)
            T.copy(A, a_ub)
            for i in T.Parallel(N):
                b_ub[i] = op(a_ub[i] + T.cast(EPS, dtype))
            T.copy(b_ub, B)

    return main


def _compound_rsqrt_2d(dtype):
    @T.prim_func
    def main(
        A: T.Tensor((ROWS, COLS), dtype),
        B: T.Tensor((ROWS, COLS), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _vid):
            a_ub = T.alloc_ub((ROWS, COLS), dtype)
            b_ub = T.alloc_ub((ROWS, COLS), dtype)
            T.copy(A, a_ub)
            for i, j in T.Parallel(ROWS, COLS):
                b_ub[i, j] = T.rsqrt(a_ub[i, j] + T.cast(EPS, dtype))
            T.copy(b_ub, B)

    return main


def _normalization_expression(dtype):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _vid):
            a_ub = T.alloc_ub((N,), dtype)
            b_ub = T.alloc_ub((N,), dtype)
            T.copy(A, a_ub)
            for i in T.Parallel(N):
                b_ub[i] = T.rsqrt(a_ub[i] + T.cast(EPS, dtype)) * T.cast(SCALE, dtype) + T.cast(BIAS, dtype)
            T.copy(b_ub, B)

    return main


def _lower_through_parallel_to_vector(program):
    target = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    with tvm.transform.PassContext(opt_level=3, config=PASS_CONFIGS):
        mod = tvm.IRModule({program.attrs["global_symbol"]: program})
        mod = tilelang.transform.InjectTmpBuffer(target)(mod)
        mod = tilelang.transform.AscendInferBufferScope()(mod)
        mod = tilelang.transform.AscendVidReduction()(mod)
        mod = tilelang.transform.BufferShapeCollector()(mod)
        mod = tir.transform.BindTarget(target)(mod)
        mod = tilelang.transform.HostProcesser()(mod)
        mod = tir.transform.Simplify()(mod)
        return tilelang.transform.AscendLowerParallelToVector()(mod)


def _lower_source(program, target):
    with tvm.transform.PassContext(opt_level=3, config=PASS_CONFIGS):
        return tilelang.lower(program, target=target).kernel_source


def _assert_ordered(source, *markers):
    positions = [source.index(marker) for marker in markers]
    assert positions == sorted(positions), source


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_compound_rsqrt_materializes_operand_in_vector_ir(dtype):
    lowered = _lower_through_parallel_to_vector(_compound_unary_1d(dtype, "rsqrt"))
    ir_text = str(lowered)

    assert "T.rsqrt" not in ir_text
    assert "_tmp_" in ir_text
    assert 'scope="shared.ub"' in ir_text
    _assert_ordered(ir_text, "T.ascend_adds", "T.ascend_rsqrt")


@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("op_name", UNARY_OPS)
def test_compound_unary_codegen_uses_mapped_vector_ops(target, op_name):
    source = _lower_source(_compound_unary_1d("float32", op_name), target)
    add_marker = "AscendC::Adds" if target == "ascendc" else "TADDS("

    _assert_ordered(source, add_marker, UNARY_SOURCE_MARKERS[target][op_name])


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_compound_rsqrt_2d_preserves_vector_plan(target):
    source = _lower_source(_compound_rsqrt_2d("float32"), target)
    add_marker = "AscendC::Adds" if target == "ascendc" else "TADDS("

    _assert_ordered(source, add_marker, UNARY_SOURCE_MARKERS[target]["rsqrt"])


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_compound_rsqrt_inside_outer_binary_tree(target):
    source = _lower_source(_normalization_expression("float32"), target)
    if target == "ascendc":
        add_marker = "AscendC::Adds"
        rsqrt_marker = "AscendC::Rsqrt"
        mul_marker = "AscendC::Muls"
    else:
        add_marker = "TADDS("
        rsqrt_marker = "TRSQRT("
        mul_marker = "TMULS("

    positions = [
        source.index(add_marker),
        source.index(rsqrt_marker),
        source.index(mul_marker),
        source.rindex(add_marker),
    ]
    assert len(set(positions)) == len(positions), source
    assert positions == sorted(positions), source


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="compound unary correctness requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_compound_rsqrt_runtime_matches_torch(dtype):
    torch_dtype = getattr(torch, dtype)
    kernel = tilelang.compile(
        _compound_unary_1d(dtype, "rsqrt"),
        target="ascendc",
        pass_configs=PASS_CONFIGS,
    )
    a = torch.linspace(0.25, 4.0, N, dtype=torch_dtype, device="npu")
    b = torch.empty_like(a)
    expected = torch.rsqrt(a + torch.tensor(EPS, dtype=torch_dtype, device="npu"))

    for _ in range(3):
        kernel(a, b)
        torch.npu.synchronize()
        torch.testing.assert_close(b, expected, rtol=5.0e-3, atol=5.0e-4)


@pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="normalization expression correctness requires an Ascend NPU runtime",
)
def test_compound_rsqrt_inside_outer_binary_runtime_matches_torch():
    kernel = tilelang.compile(
        _normalization_expression("float32"),
        target="ascendc",
        pass_configs=PASS_CONFIGS,
    )
    a = torch.linspace(0.25, 4.0, N, dtype=torch.float32, device="npu")
    b = torch.empty_like(a)
    expected = torch.rsqrt(a + EPS) * SCALE + BIAS

    kernel(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(b, expected, rtol=5.0e-3, atol=5.0e-4)
