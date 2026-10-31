# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""CPU tests for the migrated C++ prover, independent of Python's z3 module.

An OFF build runs only its unavailable-service diagnostic; semantic cases are
explicitly skipped. These tests exercise the internal PackedFunc interface and
do not compile or execute an Ascend kernel.
"""

import re
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from tilelang import tvm

tir = tvm.tir
Service = Callable[..., object]


def _available() -> bool:
    return bool(tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")())


@pytest.fixture
def create_prover() -> Callable[[], Service]:
    if not _available():
        pytest.skip("requires a build with USE_ASCEND_SYNC_Z3=ON")
    return tvm.get_global_func("tl.ascend_sync.CreateZ3ProofService")


@pytest.fixture
def prover(create_prover: Callable[[], Service]) -> Service:
    service = create_prover()
    service("set_rlimit", 1_000_000)
    service("set_timeout_ms", 1_000)
    return service


def _prove(service: Service, predicate: tir.PrimExpr) -> bool:
    return bool(service("can_prove", predicate))


def _bind_range(service: Service, var: tir.Var, minimum: int, extent: int) -> None:
    service("bind_range", var, tvm.ir.Range.from_min_extent(minimum, extent), False)


@contextmanager
def _constraint(service: Service, predicate: tir.PrimExpr, *, assume: bool = False) -> Iterator[None]:
    cleanup = service("enter_constraint", predicate, assume)
    assert callable(cleanup), "valid constraints must return a PackedFunc cleanup"
    try:
        yield
    finally:
        cleanup()


def test_unavailable_service_reports_build_option() -> None:
    if _available():
        pytest.skip("unavailable-service diagnostic requires USE_ASCEND_SYNC_Z3=OFF")
    create = tvm.get_global_func("tl.ascend_sync.CreateZ3ProofService")
    with pytest.raises(tvm.error.TVMError, match="USE_ASCEND_SYNC_Z3"):
        create()


def test_range_binding_proves_bounds_but_not_a_counterexample(prover: Service) -> None:
    x = tir.Var("x", "int32")
    _bind_range(prover, x, -4, 12)
    assert _prove(prover, x >= -4)
    assert _prove(prover, x < 8)
    assert not _prove(prover, x >= 0)
    assert not _prove(prover, x < 0)


def test_empty_range_does_not_make_all_queries_vacuously_true(prover: Service) -> None:
    x = tir.Var("x", "int32")
    _bind_range(prover, x, 0, 0)
    assert not _prove(prover, x >= 0)
    assert not _prove(prover, x < 0)


def test_expression_binding_and_explicit_override(prover: Service) -> None:
    x = tir.Var("x", "int32")
    y = tir.Var("y", "int32")
    _bind_range(prover, x, 0, 16)
    prover("bind_expr", y, x * 3 + 2, False)
    assert _prove(prover, y >= 2)
    assert _prove(prover, y < 48)
    prover("bind_expr", y, x + 100, True)
    assert _prove(prover, y >= 100)
    assert not _prove(prover, y < 48)


def test_distinct_variables_with_the_same_name_are_not_aliased(prover: Service) -> None:
    left = tir.Var("duplicate", "int32")
    right = tir.Var("duplicate", "int32")
    _bind_range(prover, left, 0, 8)
    _bind_range(prover, right, 16, 8)
    assert _prove(prover, left < right)
    assert not _prove(prover, left == right)


@pytest.mark.parametrize("materialize_first", [False, True], ids=["lazy", "eager"])
def test_nested_constraints_restore_outer_and_base_state(prover: Service, materialize_first: bool) -> None:
    x = tir.Var("x", "int32")
    _bind_range(prover, x, 0, 32)
    if materialize_first:
        assert _prove(prover, x >= 0)
    with _constraint(prover, x >= 4):
        assert _prove(prover, x >= 4)
        with _constraint(prover, x < 8):
            assert _prove(prover, x < 8)
            assert _prove(prover, x >= 4)
        assert not _prove(prover, x < 8)
        assert _prove(prover, x >= 4)
    assert not _prove(prover, x >= 4)
    assert _prove(prover, x < 32)


@pytest.mark.parametrize("materialize_first", [False, True], ids=["lazy", "eager"])
def test_exited_unqueried_scope_is_not_replayed(prover: Service, materialize_first: bool) -> None:
    x = tir.Var("x", "int32")
    _bind_range(prover, x, 0, 16)
    if materialize_first:
        assert _prove(prover, x < 16)
    with _constraint(prover, x >= 12):
        pass
    assert not _prove(prover, x >= 12)
    assert _prove(prover, x < 16)


@pytest.mark.parametrize("bind_kind", ["bind_range", "bind_expr"])
@pytest.mark.parametrize("materialization", ["before", "inside", "after"])
def test_scoped_bindings_do_not_escape_their_constraint_scope(prover: Service, bind_kind: str, materialization: str) -> None:
    local = tir.Var("local", "int32")
    if materialization == "before":
        prover("smtlib", None)
    with _constraint(prover, tir.const(True, "bool")):
        if bind_kind == "bind_range":
            _bind_range(prover, local, 7, 1)
        else:
            prover("bind_expr", local, tir.IntImm("int32", 7), False)
        if materialization != "after":
            assert _prove(prover, local == 7)
    assert not _prove(prover, local == 7)


@pytest.mark.parametrize("bits", [8, 16, 32, 64])
@pytest.mark.parametrize("operation", ["add", "subtract", "multiply"])
def test_unsigned_arithmetic_preserves_wraparound(
    prover: Service,
    bits: int,
    operation: str,
) -> None:
    dtype = f"uint{bits}"
    value = tir.Var("unsigned_value", dtype)

    def constant(number: int) -> tir.IntImm:
        return tir.IntImm(dtype, number)

    if operation == "add":
        # Every literal fits int64, including the uint64 case.
        bound = (1 << (bits - 1)) - 1
        expression = (value + value) + constant(2)
    elif operation == "subtract":
        bound = 0
        expression = value - constant(1)
    else:
        bound = 1 << (bits // 2)
        expression = value * value
    prover("bind_expr", value, constant(bound), False)
    if operation == "subtract":
        assert _prove(prover, expression > constant(0))
        assert not _prove(prover, expression == constant(0))
    else:
        assert _prove(prover, expression == constant(0))
        assert not _prove(prover, expression > constant(0))


@pytest.mark.parametrize("numerator", [-17, -7, 7, 17])
@pytest.mark.parametrize("denominator", [-3, 3])
def test_signed_floor_division_and_modulo_match_integer_oracle(prover: Service, numerator: int, denominator: int) -> None:
    x = tir.Var("x", "int32")
    prover("bind_expr", x, tir.IntImm("int32", numerator), False)
    quotient, remainder = divmod(numerator, denominator)
    assert _prove(prover, tir.floordiv(x, denominator) == quotient)
    assert _prove(prover, tir.floormod(x, denominator) == remainder)
    assert not _prove(prover, tir.floormod(x, denominator) == remainder + 1)


@pytest.mark.parametrize(
    "numerator,denominator",
    [
        (-17, 3),
        (-17, -3),
        (17, -3),
        (17, 3),
        (-1, 2),
        (-1, -2),
        (-6, 3),
        (-6, -3),
        (0, 3),
        (0, -3),
        (-(1 << 31), 1),
        (-(1 << 31), 3),
        (-(1 << 31), -(1 << 31)),
        ((1 << 31) - 1, -3),
    ],
)
def test_signed_truncating_division_and_modulo_match_integer_oracle(
    prover: Service,
    numerator: int,
    denominator: int,
) -> None:
    value = tir.Var("signed_value", "int32")
    prover("bind_expr", value, tir.IntImm("int32", numerator), False)
    quotient = abs(numerator) // abs(denominator)
    if (numerator < 0) != (denominator < 0):
        quotient = -quotient
    remainder = numerator - quotient * denominator
    assert _prove(prover, tir.Div(value, denominator) == quotient)
    assert _prove(prover, tir.Mod(value, denominator) == remainder)
    assert not _prove(prover, tir.Div(value, denominator) == quotient + 1)
    assert not _prove(prover, tir.Mod(value, denominator) == remainder + 1)


@pytest.mark.parametrize("operation", [tir.Div, tir.Mod])
@pytest.mark.parametrize("numerator,denominator", [(0, 0), (7, 0), (-(1 << 31), -1)])
def test_undefined_truncating_division_does_not_prove_a_value(
    prover: Service, operation: Callable, numerator: int, denominator: int
) -> None:
    value = tir.Var("signed_value", "int32")
    prover("bind_expr", value, tir.IntImm("int32", numerator), False)
    expression = operation(value, denominator)
    assert not _prove(prover, expression == 0)
    assert not _prove(prover, expression > 0)


@pytest.mark.parametrize("divisor", [3, 16, 127])
def test_symbolic_divmod_reconstructs_negative_and_positive_indices(prover: Service, divisor: int) -> None:
    x = tir.Var("index", "int32")
    _bind_range(prover, x, -4096, 8192)
    quotient = tir.floordiv(x, divisor)
    remainder = tir.floormod(x, divisor)
    assert _prove(prover, quotient * divisor + remainder == x)
    assert _prove(prover, remainder >= 0)
    assert _prove(prover, remainder < divisor)


def test_tiled_address_bounds_require_both_ranges_and_guard(prover: Service) -> None:
    tile = tir.Var("tile", "int32")
    lane = tir.Var("lane", "int32")
    _bind_range(prover, tile, 0, 128)
    _bind_range(prover, lane, 0, 32)
    address = tile * 32 + lane
    with _constraint(prover, address < 1003):
        assert _prove(prover, tile <= 31)
        assert _prove(prover, address >= 0)
    assert not _prove(prover, tile <= 31)


def test_widening_cast_preserves_negative_value(prover: Service) -> None:
    x = tir.Var("x", "int32")
    _bind_range(prover, x, -17, 8)
    widened = tir.Cast("int64", x)
    assert _prove(prover, widened < 0)
    assert _prove(prover, widened >= -17)


def test_narrowing_cast_is_rejected_instead_of_proved_unsoundly(prover: Service) -> None:
    x = tir.Var("x", "int32")
    _bind_range(prover, x, 128, 128)
    narrowed = tir.Cast("int8", x)
    with pytest.raises(tvm.error.TVMError, match="(?i)(narrow|unsupported|not supported)"):
        _prove(prover, narrowed >= 0)


@pytest.mark.parametrize("materialize_first", [False, True], ids=["lazy", "eager"])
@pytest.mark.parametrize("operation", ["enter_constraint", "bind_expr"])
@pytest.mark.parametrize("invalid_kind", ["shift", "narrowing"])
def test_invalid_input_does_not_poison_scope_or_binding_journal(
    prover: Service, materialize_first: bool, operation: str, invalid_kind: str
) -> None:
    x = tir.Var("valid_index", "int32")
    y = tir.Var("binding_target", "int32")
    amount = tir.Var("invalid_amount", "int32")
    _bind_range(prover, x, 0, 16)
    prover("bind_expr", amount, tir.IntImm("int32", -1), False)
    if materialize_first:
        assert _prove(prover, x >= 0)

    invalid = tir.shift_right(x, amount) if invalid_kind == "shift" else tir.Cast("int8", x)
    with pytest.raises(tvm.error.TVMError, match="(?i)(shift|narrow|not support)"):
        if operation == "enter_constraint":
            prover(operation, invalid >= 0, False)
        else:
            prover(operation, y, invalid, False)

    assert _prove(prover, x >= 0)
    assert _prove(prover, x < 16)
    assert not _prove(prover, x == 99)
    prover("bind_expr", y, x + 2, False)
    assert _prove(prover, y >= 2)
    assert _prove(prover, y < 18)
    with _constraint(prover, x < 8):
        assert _prove(prover, x < 8)
    assert not _prove(prover, x < 8)


def test_bitwise_mask_and_shift_on_bounded_nonnegative_indices(prover: Service) -> None:
    # Mixed integer/bit-vector reasoning needs about 7 s on the current host;
    # keep a bounded budget without weakening either equivalence assertion.
    prover("set_timeout_ms", 15_000)
    x = tir.Var("x", "int32")
    _bind_range(prover, x, 0, 256)
    assert _prove(prover, tir.bitwise_and(x, 15) == tir.floormod(x, 16))
    assert _prove(prover, tir.shift_right(x, 4) == tir.floordiv(x, 16))


def test_unsigned_bitwise_high_bit_is_preserved_or_explicitly_rejected(prover: Service) -> None:
    value = tir.Var("unsigned_value", "uint32")
    high_bit = tir.IntImm("uint32", 1 << 31)
    zero = tir.IntImm("uint32", 0)
    prover("bind_expr", value, high_bit, False)
    masked = tir.bitwise_and(value, high_bit)
    try:
        negative_proved = _prove(prover, masked < zero)
    except tvm.error.TVMError as error:
        assert re.search("(?i)(unsigned|unsupported|not support)", str(error))
    else:
        assert not negative_proved, "an unsigned bitwise result cannot be negative"
        assert _prove(prover, masked == high_bit)
    assert _prove(prover, value == high_bit)
    assert not _prove(prover, value == zero)


def test_invalid_symbolic_shift_is_rejected_without_poisoning_solver(prover: Service) -> None:
    x = tir.Var("x", "int32")
    amount = tir.Var("amount", "int32")
    _bind_range(prover, x, 0, 16)
    prover("bind_expr", amount, tir.IntImm("int32", -1), False)
    error = None
    try:
        _prove(prover, tir.shift_right(x, amount) >= 0)
    except tvm.error.TVMError as caught:
        error = caught
    assert not _prove(prover, x == 99), "an invalid shift must not make the context unsatisfiable"
    assert error is not None, "an invalid symbolic shift must produce a diagnostic"
    assert re.search("(?i)(shift|unsupported|not support)", str(error))


@pytest.mark.parametrize("materialize_first", [False, True], ids=["lazy", "eager"])
def test_assumed_mutable_read_expires_with_its_scope(prover: Service, materialize_first: bool) -> None:
    buffer = tir.decl_buffer((1,), "int32", name="mutable_value")
    value = tir.BufferLoad(buffer, [tir.IntImm("int32", 0)])
    if materialize_first:
        prover("smtlib", None)
    with _constraint(prover, value >= 0, assume=True):
        assert _prove(prover, value >= 0)
    assert not _prove(prover, value >= 0)
    with _constraint(prover, value < 0, assume=True):
        assert _prove(prover, value < 0)
        assert not _prove(prover, value >= 0)
    assert not _prove(prover, value < 0)


def test_unfrozen_mutable_read_is_not_a_persistent_assumption(prover: Service) -> None:
    buffer = tir.decl_buffer((1,), "int32", name="mutable_value")
    value = tir.BufferLoad(buffer, [tir.IntImm("int32", 0)])
    with _constraint(prover, value >= 0, assume=False):
        assert not _prove(prover, value >= 0)


def test_two_services_do_not_share_bindings(create_prover: Callable[[], Service]) -> None:
    x = tir.Var("shared_python_var", "int32")
    left, right = create_prover(), create_prover()
    left("bind_expr", x, tir.IntImm("int32", 3), False)
    right("bind_expr", x, tir.IntImm("int32", 9), False)
    for _ in range(3):
        assert _prove(left, x == 3)
        assert not _prove(left, x == 9)
        assert _prove(right, x == 9)
        assert not _prove(right, x == 3)


def test_parallel_independent_services_have_isolated_contexts(
    create_prover: Callable[[], Service],
) -> None:
    def check(value: int) -> bool:
        service = create_prover()
        x = tir.Var("same_name_in_each_worker", "int32")
        _bind_range(service, x, value * 16, 16)
        return _prove(service, x >= value * 16) and not _prove(service, x < value * 16)

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert all(executor.map(check, range(12)))


def test_low_resource_limit_cannot_turn_unknown_into_a_proof(prover: Service) -> None:
    x = tir.Var("x", "int32")
    _bind_range(prover, x, -4096, 8192)
    predicate = tir.floormod(x * 17 + 3, 7) < 7
    prover("set_rlimit", 1)
    assert not _prove(prover, predicate)
    prover("set_rlimit", 1_000_000)
    assert _prove(prover, predicate)


@pytest.mark.parametrize("command", ["set_rlimit", "set_timeout_ms"])
def test_negative_solver_limits_are_rejected(prover: Service, command: str) -> None:
    with pytest.raises(tvm.error.TVMError, match="(?i)(nonnegative|negative|range|unsigned)"):
        prover(command, -1)


def test_cleanup_still_restores_state_after_an_ffi_exception(prover: Service) -> None:
    x = tir.Var("x", "int32")
    _bind_range(prover, x, 0, 16)
    with pytest.raises(tvm.error.TVMError), _constraint(prover, x >= 12):
        assert _prove(prover, x >= 12)
        prover("set_rlimit", -1)
    assert not _prove(prover, x >= 12)
    assert _prove(prover, x < 16)


def test_debug_smtlib_query_does_not_leave_its_negation_asserted(prover: Service) -> None:
    x = tir.Var("x", "int32")
    _bind_range(prover, x, 0, 32)
    context = str(prover("smtlib", None))
    query = str(prover("smtlib", x == 11))
    assert "(assert" in context
    assert "(check-sat)" in query
    assert "Trying to prove" in query
    assert not _prove(prover, x == 11)
    assert _prove(prover, x < 32)
    assert str(prover("stats")).strip()
