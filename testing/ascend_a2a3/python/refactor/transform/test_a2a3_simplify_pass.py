"""Original TA Simplify eligibility and real allocation-order regressions."""

import importlib.util
from pathlib import Path
import re

import pytest
import tilelang
from tilelang import tvm

tirx = tvm.tirx
ROOT = Path(__file__).parents[5]


def _pass():
    factory = tvm.ffi.get_global_func("tl.transform.A2A3Simplify", allow_missing=True)
    if factory is None:
        pytest.skip("A2/A3 Simplify compatibility component is not built")
    return factory()


@pytest.fixture(autouse=True)
def _require_native_component_before_backend_ir():
    _pass()


def _run(body, params=(), attrs=None):
    func = tirx.PrimFunc(params, body).with_attr({"global_symbol": "main", **(attrs or {})})
    return _pass()(tvm.IRModule.from_expr(func))["main"]


def _binds(func):
    found = []
    tirx.stmt_functor.post_order_visit(func.body, lambda n: found.append(n) if isinstance(n, tirx.Bind) else None)
    return found


def test_original_integer_inline_removes_residue_and_keeps_effectful_consumer():
    x, n = tirx.Var("x", "int32"), tirx.Var("n", "int32")
    body = tirx.SeqStmt([tirx.Bind(n, x + 1), tirx.Evaluate(tirx.call_extern("int32", "capture", n))])
    after = _run(body, [x])
    assert not _binds(after)
    calls = []
    tirx.stmt_functor.post_order_visit(after.body, lambda n: calls.append(n) if isinstance(n, tirx.Call) else None)
    assert len(calls) == 1 and str(calls[0].args[0]) == '"capture"'
    tvm.ir.assert_structural_equal(calls[0].args[1], x + 1)


@pytest.mark.parametrize("dtype,value", [("float32", 3.0), ("int32", 3)])
def test_original_constant_number_binding_is_eligible(dtype, value):
    n = tirx.Var("n", dtype)
    after = _run(tirx.SeqStmt([tirx.Bind(n, tirx.const(value, dtype)), tirx.Evaluate(0)]))
    assert not _binds(after)


@pytest.mark.parametrize("dtype", ["float32", "uint32", "bool"])
def test_pre_simplify_noneligible_expression_is_not_reclassified_after_folding(dtype):
    if dtype == "bool":
        value = tirx.And(tirx.const(True), tirx.const(False))
    else:
        value = tirx.Add(tirx.const(1, dtype), tirx.const(2, dtype))
    n = tirx.Var("n", dtype)
    after = _run(tirx.SeqStmt([tirx.Bind(n, value), tirx.Evaluate(0)]))
    binds = _binds(after)
    assert len(binds) == 1 and binds[0].var.same_as(n)
    assert isinstance(binds[0].value, (tirx.IntImm, tirx.FloatImm))


def test_unused_ordinary_read_and_opaque_binding_are_not_noop_cleanup():
    buffer = tirx.decl_buffer((1,), "int32", name="buffer")
    read, effect = tirx.Var("read", "int32"), tirx.Var("effect", "int32")
    body = tirx.SeqStmt(
        [tirx.Bind(read, tirx.BufferLoad(buffer, [0])), tirx.Bind(effect, tirx.call_extern("int32", "effect")), tirx.Evaluate(0)]
    )
    after = _run(body, [buffer.data])
    assert {b.var for b in _binds(after)} == {read, effect}


@pytest.mark.parametrize("kind", ["function", "buffer", "loop", "call"])
def test_scalar_identity_in_metadata_or_descriptor_remains_bound(kind):
    x, n = tirx.Var("x", "int32"), tirx.Var("n", "int32")
    attrs = {}
    if kind == "function":
        attrs = {"scalar_metadata": {"carrier": n}}
        tail = tirx.Evaluate(0)
    elif kind == "buffer":
        tail = tirx.SeqStmt([tirx.AllocBuffer(tirx.decl_buffer((n,), "float32", name="scratch")), tirx.Evaluate(0)])
    elif kind == "loop":
        i = tirx.Var("i", "int32")
        tail = tirx.For(i, 0, 2, tirx.ForKind.SERIAL, tirx.Evaluate(tirx.call_extern("int32", "effect")), annotations={"carrier": n})
    else:
        tail = tirx.Evaluate(tirx.Call("int32", "tirx.call_extern", [tirx.StringImm("effect")], annotations={"carrier": n}))
    after = _run(tirx.SeqStmt([tirx.Bind(n, x + 1), tail]), [x], attrs)
    assert any(b.var.same_as(n) for b in _binds(after))


def test_ancestor_binding_implicitly_used_by_nested_raw_code_is_retained():
    x, n, i = tirx.Var("x", "int32"), tirx.Var("n", "int32"), tirx.Var("i", "int32")
    raw = tirx.Evaluate(tirx.call_intrin("handle", tvm.ir.Op.get("tl.ascend_src_code"), "int raw_consumer = n;"))
    after = _run(tirx.SeqStmt([tirx.Bind(n, x + 1), tirx.For(i, 0, 2, tirx.ForKind.SERIAL, raw)]), [x])
    assert any(b.var.same_as(n) for b in _binds(after))


def test_real_module_context_configuration_and_global_information_are_preserved():
    x, n = tirx.Var("x", "int32"), tirx.Var("n", "int32")
    f = tirx.PrimFunc([x], tirx.SeqStmt([tirx.Bind(n, x + 1), tirx.Evaluate(0)])).with_attr("global_symbol", "first")
    g = tirx.PrimFunc([], tirx.Evaluate(tirx.call_extern("int32", "effect"))).with_attr("global_symbol", "second")
    mod = tvm.IRModule({"first": f, "second": g}, attrs={"module_token": "keep"})
    info = tvm.ir.DummyGlobalInfo()
    mod.update_global_info("test_info", [info])
    first, second = mod.get_global_var("first"), mod.get_global_var("second")
    with tvm.transform.PassContext(config={"tirx.Simplify": {"transitively_prove_inequalities": True}}):
        after = _pass()(mod)
    assert len(after.functions) == 2 and str(after.attrs["module_token"]) == "keep"
    assert after.get_global_var("first").same_as(first) and after.get_global_var("second").same_as(second)
    # GlobalInfo.same_as delegates through recursive Python equality; compare
    # the actual object handles to retain the stronger identity contract.
    assert after.global_infos["test_info"][0].__chandle__() == info.__chandle__()
    assert not _binds(after["first"])
    tvm.ir.assert_structural_equal(after["second"].body, g.body)


def _load(relative):
    path = ROOT / "examples/ascend_a2a3" / relative
    spec = importlib.util.spec_from_file_location("_simplify_original_factory", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("case", ["gqa_small", "h32_default"])
def test_original_complete_factories_restore_ta_storage_order_and_views(case):
    _pass()
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    if case == "gqa_small":
        module = _load("gqa_fwd_varlen/gqa_fwd_varlen.py")
        factory = module.flashattn
        func = factory._get_tir_with_target(
            target, 1, 2, 4, 128, 128, 128, False, block_M=128, block_N=128, num_stages=8, cross_interval=1, apply_mask=False
        )
        expected_order = [
            "q_l1",
            "k_l1",
            "l0a",
            "l0b",
            "l0c",
            "v_l1",
            "p_l1",
            "acc_o",
            "sumexp",
            "neg_sm",
            "io_buf",
            "work_ub",
            "tmp_ub",
            "buf_2d",
            "acc_s_half",
            "sumexp_is",
            "r_factors",
        ]
        expected_views = {
            "io_buf": ("half", 16384, 33536),
            "work_ub": ("float", 8192, 66304),
            "tmp_ub": ("uint8_t", 4672, 99072),
            "buf_2d": ("float", 8192, 103744),
            "acc_s_half": ("half", 8192, 136512),
        }
    else:
        module = _load("flash_attention/fa_opt/auto_pipeline/h32_d512.py")
        factory = module.flash_attention_fwd
        func = factory._get_tir_with_target(target, 4, 4096, 32, 32, 512)
        expected_order = [
            "q_l1",
            "k_l1",
            "acc_s_l0c",
            "acc_s_l1",
            "v_l1",
            "acc_o_l0c",
            "acc_o",
            "sumexp",
            "m_i",
            "m_i_prev",
            "acc_s_ub",
            "acc_s_ub_",
            "tmp_ub",
            "m_i_2d",
            "sumexp_i_ub",
            "acc_s_half",
            "acc_o_ub",
            "acc_o_half",
        ]
        expected_views = {"acc_o_half": ("half", 16384, 65792)}
    with tvm.transform.PassContext(config=factory.pass_configs):
        artifact = tilelang.lower(func, target=target)
    # Exact old source contracts at the actual changed allocation/descriptor
    # boundary. These are complete original factories, not synthetic kernels.
    views = re.findall(r"auto (\w+) = \w+\.GetWithOffset<([^>]+)>\((\d+), (\d+)\);", artifact.kernel_source)
    assert [v[0] for v in views] == expected_order
    actual = {name: (dtype, int(count), int(offset)) for name, dtype, count, offset in views}
    for name, value in expected_views.items():
        assert actual[name] == value
