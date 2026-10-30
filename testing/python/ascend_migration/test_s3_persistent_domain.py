"""Execute actual Persistent index/guard IR to check exact-once tile coverage.

The scalar C marker is an IR probe only. Device regressions use the existing
GEMM kernel, rather than concurrent scalar GM stores from multiple cores.
"""

from collections import Counter
import itertools

import pytest

from tilelang import language as T
from tilelang import tvm


def make_domain_probe(domain, wave_size, group_size=8):
    names = [f"i{index}" for index in range(len(domain))]
    indices = ", ".join(names)
    source = f"""@T.prim_func
def main(O: T.Tensor(shape, "int32")):
    with T.Kernel(workers, is_npu=True) as (cid, vid):
        for {indices} in T.Persistent(shape, workers, cid, grouping):
            with T.Scope("C"):
                O[{indices}] = 1
"""
    return tvm.script.from_source(
        source,
        extra_vars={"T": T, "shape": tuple(domain), "workers": wave_size, "grouping": group_size},
    )


class _BreakWave(Exception):
    pass


def _integer(expression, bindings):
    if isinstance(expression, (int, bool)):
        return int(expression)
    bound = tvm.tir.stmt_functor.substitute(expression, bindings)
    value = tvm.arith.Analyzer().simplify(bound)
    if not isinstance(value, tvm.tir.IntImm):
        raise AssertionError(f"Unresolved expression in domain IR: {value}")
    return int(value)


def trace_domain(function, bindings=None):
    """Interpret native-generated loops, predicates and output indices, not a replica formula."""
    bindings = {variable: tvm.tir.IntImm(variable.dtype, value) for variable, value in (bindings or {}).items()}
    # Native Persistent currently emits pure coordinate Lets outside the wave
    # For. The same standard simplifier used by lowering substitutes those Lets
    # into the loop before concrete execution; evaluating raw Lets first would
    # incorrectly demand a value for the not-yet-bound wave variable.
    module = tvm.tir.transform.Simplify()(tvm.IRModule.from_expr(function))
    stores = []

    def execute(statement, environment, owner=None):
        if isinstance(statement, tvm.tir.SeqStmt):
            for child in statement.seq:
                execute(child, environment, owner)
        elif isinstance(statement, tvm.tir.For):
            start = _integer(statement.min, environment)
            extent = _integer(statement.extent, environment)
            if extent < 0:
                raise AssertionError("Generated loop has a negative extent")
            for value in range(start, start + extent):
                nested = dict(environment)
                nested[statement.loop_var] = tvm.tir.IntImm(statement.loop_var.dtype, value)
                try:
                    execute(statement.body, nested, owner)
                except _BreakWave:
                    break
        elif isinstance(statement, tvm.tir.IfThenElse):
            selected = statement.then_case if _integer(statement.condition, environment) else statement.else_case
            if selected is not None:
                execute(selected, environment, owner)
        elif isinstance(statement, tvm.tir.LetStmt):
            nested = dict(environment)
            nested[statement.var] = tvm.tir.IntImm(statement.var.dtype, _integer(statement.value, environment))
            execute(statement.body, nested, owner)
        elif isinstance(statement, tvm.tir.AttrStmt):
            if statement.attr_key == "thread_extent":
                tag = statement.node.thread_tag
                extent = _integer(statement.value, environment)
                if tag not in ("blockIdx.x", "blockIdx.y"):
                    raise AssertionError(f"Unexpected probe launch axis: {tag}")
                # A C-owned marker executes once per Cube core. blockIdx.y is
                # the two Vector subcores; it must not duplicate the C trace.
                values = range(extent) if tag == "blockIdx.x" else (0,)
                for value in values:
                    nested = dict(environment)
                    nested[statement.node.var] = tvm.tir.IntImm(statement.node.var.dtype, value)
                    execute(statement.body, nested, owner)
            elif statement.attr_key == "resource_scope":
                scope = _integer(statement.value, environment)
                if scope != 0:
                    raise AssertionError("Domain probe must contain only a C-owned marker")
                execute(statement.body, environment, "C")
            else:
                execute(statement.body, environment, owner)
        elif isinstance(statement, tvm.tir.BlockRealize):
            if _integer(statement.predicate, environment):
                nested = dict(environment)
                for variable, value in zip(statement.block.iter_vars, statement.iter_values):
                    nested[variable.var] = tvm.tir.IntImm(variable.var.dtype, _integer(value, environment))
                execute(statement.block, nested, owner)
        elif isinstance(statement, tvm.tir.Block):
            if statement.init is not None:
                execute(statement.init, environment, owner)
            execute(statement.body, environment, owner)
        elif isinstance(statement, tvm.tir.AssertStmt):
            if not _integer(statement.condition, environment):
                raise AssertionError(f"Generated domain assertion failed: {statement.message}")
            execute(statement.body, environment, owner)
        elif isinstance(statement, tvm.tir.BufferStore):
            if owner != "C" or _integer(statement.value, environment) != 1:
                raise AssertionError("Unexpected side effect in the C domain probe")
            stores.append(tuple(_integer(index, environment) for index in statement.indices))
        elif isinstance(statement, tvm.tir.Evaluate):
            value = statement.value
            if isinstance(value, tvm.tir.Call) and getattr(value.op, "name", None) == "tl.loop_break":
                raise _BreakWave
            if not isinstance(value, tvm.tir.IntImm) or int(value) != 0:
                raise AssertionError(f"Unsupported probe evaluation: {value}")
        else:
            raise AssertionError(f"Unsupported statement in domain probe: {type(statement).__name__}")

    execute(module["main"].body, bindings)
    return Counter(stores)


def assert_exact_domain(function, domain, bindings=None):
    concrete = [
        _integer(
            value,
            {variable: tvm.tir.IntImm(variable.dtype, bound) for variable, bound in (bindings or {}).items()},
        )
        for value in domain
    ]
    if any(value <= 0 for value in concrete):
        raise AssertionError("Only positive symbolic-domain bindings satisfy the API contract")
    expected = Counter({point: 1 for point in itertools.product(*(range(value) for value in concrete))})
    observed = trace_domain(function, bindings)
    assert observed == expected, {
        "missing": list((expected - observed).items())[:8],
        "duplicate_or_invalid": list((observed - expected).items())[:8],
    }


@pytest.mark.parametrize(
    "domain,workers,group",
    [
        ([1], 8, 8),
        ([4], 8, 2),
        ([8], 8, 8),
        ([9], 4, 8),
        ([2, 2], 8, 8),
        ([2, 4], 8, 2),
        ([3, 3], 4, 3),
        ([3, 10], 8, 4),
        ([1, 5], 8, 2),
        ([2, 3, 5], 8, 2),
        ([2, 3], 1, 8),
        ([2, 4, 8], 8, 4),
    ],
)
def test_generated_static_domain_is_exact_once(domain, workers, group):
    assert_exact_domain(make_domain_probe(domain, workers, group), domain)


@pytest.mark.parametrize("rows,cols", [(1, 1), (1, 3), (2, 8), (3, 10)])
def test_generated_positive_symbolic_2d_domain_is_exact_once(rows, cols):
    r, c = T.symbolic("rows", "int32"), T.symbolic("cols", "int32")
    function = make_domain_probe([r, c], 8, 4)
    assert_exact_domain(function, [r, c], {r: rows, c: cols})


@pytest.mark.parametrize("batch,rows", [(1, 1), (2, 3)])
def test_generated_positive_symbolic_3d_domain_is_exact_once(batch, rows):
    b, r = T.symbolic("batch", "int32"), T.symbolic("rows", "int32")
    function = make_domain_probe([b, r, 5], 4, 2)
    assert_exact_domain(function, [b, r, 5], {b: batch, r: rows})


@pytest.mark.parametrize("mutation", ["duplicate", "out_of_range"])
def test_domain_checker_rejects_corrupted_generated_ir(mutation):
    function = make_domain_probe([2, 2], 8, 2)

    def corrupt(node):
        if mutation == "duplicate":
            return tvm.tir.SeqStmt([node, node])
        indices = list(node.indices)
        indices[0] = tvm.tir.IntImm(indices[0].dtype, 2)
        return tvm.tir.BufferStore(node.buffer, node.value, indices)

    body = tvm.tir.stmt_functor.ir_transform(function.body, None, corrupt, ["tir.BufferStore"])
    with pytest.raises(AssertionError, match="duplicate_or_invalid"):
        assert_exact_domain(function.with_body(body), [2, 2])
