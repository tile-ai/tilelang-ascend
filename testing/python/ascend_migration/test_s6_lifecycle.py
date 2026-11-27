"""Source readiness uses complete byte definitions on each actual activation."""

import pytest

import tilelang
from tilelang import transform, tvm

from test_s2_task_ir import _add, _buffer, _dma, _fill, _function

tir = tvm.tir


def _analyze(function):
    before = tvm.ir.save_json(function)
    result = transform.AnalyzeAscendLifetimes(function, platform="A3")
    assert tvm.ir.save_json(function) == before
    assert int(result["schema_version"]) == 1
    return result


def _buffers(size=64):
    return _buffer("value", shape=(size,)), _buffer("output", "global", shape=(size,))


def test_full_initialization_defines_the_read_without_modifying_input():
    value, output = _buffers()
    result = _analyze(_function([value, output], [_fill(value), _dma(value, output)]))
    case = result["cases"][0]
    assert len(case["reads"]) == 1
    read = case["reads"][0]
    assert [int(number) for number in read["consumer_key"]] == [1]
    assert [int(number) for number in read["segments"][0]["producer_key"]] == [0]
    assert int(read["begin"]) == 0 and int(read["end"]) == 256


def test_allocating_storage_does_not_initialize_its_contents():
    value, output = _buffers()
    with pytest.raises(tvm.TVMError, match="no complete reaching definition"):
        _analyze(_function([value, output], _dma(value, output)))


def test_internal_global_allocation_is_not_a_caller_entry_definition():
    value, _ = _buffers()
    scratch = _buffer("internal_workspace", "global", shape=(64,))
    allocation = tir.Allocate(scratch.data, "float32", [64], tir.const(True, "bool"), _dma(scratch, value))
    with pytest.raises(tvm.TVMError, match="no complete reaching definition"):
        _analyze(_function([value], allocation))


def test_inplace_operation_reads_before_applying_its_own_write():
    value, _ = _buffers()
    with pytest.raises(tvm.TVMError, match="no complete reaching definition"):
        _analyze(_function([value], _add(value, value, value)))


@pytest.mark.parametrize("hole", [False, True])
def test_partial_write_union_requires_complete_read_coverage(hole):
    value, output = _buffers(128)
    function = _function(
        [value, output],
        [
            _fill(value, count=56 if hole else 64),
            _fill(value, offset=64),
            _dma(value, output, count=128),
        ],
    )
    if hole:
        with pytest.raises(tvm.TVMError, match="coverage hole"):
            _analyze(function)
    else:
        read = _analyze(function)["cases"][0]["reads"][0]
        assert [(int(part["begin"]), int(part["end"])) for part in read["segments"]] == [
            (0, 256),
            (256, 512),
        ]


@pytest.mark.parametrize("kind", ["matching", "exhaustive", "missing"])
def test_boolean_predicates_are_checked_for_every_path(kind):
    value, output = _buffers()
    predicate = tir.Var("predicate", "bool")
    producer = tir.IfThenElse(predicate, _fill(value), _fill(value) if kind == "exhaustive" else None)
    consumer = _dma(value, output)
    if kind == "matching":
        consumer = tir.IfThenElse(predicate, consumer, None)
    function = _function([value, output], [producer, consumer], scalars=[predicate])
    if kind == "missing":
        with pytest.raises(tvm.TVMError, match="no complete reaching definition"):
            _analyze(function)
    else:
        result = _analyze(function)
        assert len(result["cases"]) == 2
        assert sorted(len(case["reads"]) for case in result["cases"]) == ([0, 1] if kind == "matching" else [1, 1])


def test_boolean_parameters_cannot_become_unbounded_data_or_addresses():
    value, _ = _buffers()
    predicate = tir.Var("predicate", "bool")
    call = _fill(value).value
    arguments = list(call.args)
    arguments[2] = tir.Cast("float32", predicate)
    function = _function([value], tir.Evaluate(tir.Call(call.dtype, call.op, arguments)), scalars=[predicate])
    with pytest.raises(tvm.TVMError, match="only in predicates"):
        _analyze(function)


@pytest.mark.parametrize("trip", [0, 1, 3])
@pytest.mark.parametrize("initialize", [False, True])
def test_zero_trip_writes_do_not_create_a_definition(trip, initialize):
    value, output = _buffers()
    iteration = tir.Var("iteration", "int32")
    statements = [_fill(value)] if initialize else []
    statements += [
        tir.For(iteration, 0, trip, tir.ForKind.SERIAL, _fill(value)),
        _dma(value, output),
    ]
    function = _function([value, output], statements)
    if not initialize and trip == 0:
        with pytest.raises(tvm.TVMError, match="no complete reaching definition"):
            _analyze(function)
    else:
        assert len(_analyze(function)["cases"][0]["reads"]) == 1


def test_initialized_n1_carry_keeps_the_previous_iteration_writer():
    value, output = _buffers()
    iteration = tir.Var("iteration", "int32")
    body = tir.For(iteration, 0, 3, tir.ForKind.SERIAL, _add(value, value, value))
    function = _function([value, output], [_fill(value), body, _dma(value, output)])
    reads = _analyze(function)["cases"][0]["reads"]
    assert len(reads) == 7
    assert [int(number) for number in reads[2]["segments"][0]["producer_key"]] == [1, 0, 0]
    assert [int(number) for number in reads[-1]["segments"][0]["producer_key"]] == [1, 0, 2]


@pytest.mark.parametrize("every_iteration", [False, True])
def test_loop_body_allocation_gets_a_fresh_activation(every_iteration):
    value, output = _buffers()
    iteration = tir.Var("iteration", "int32")
    producer = _fill(value)
    if not every_iteration:
        producer = tir.IfThenElse(iteration == 0, producer, None)
    block = tir.Block([], [], [], "fresh", tir.SeqStmt([producer, _dma(value, output)]), alloc_buffers=[value])
    loop = tir.For(iteration, 0, 2, tir.ForKind.SERIAL, tir.BlockRealize([], True, block))
    function = _function([output], loop)
    if every_iteration:
        case = _analyze(function)["cases"][0]
        assert len(case["activations"]) == 2
        keys = [tuple(int(number) for number in item["key"]) for item in case["activations"]]
        assert keys[0] != keys[1]
    else:
        with pytest.raises(tvm.TVMError, match="no complete reaching definition"):
            _analyze(function)


def test_outer_activation_survives_nested_and_consecutive_loops():
    value, output = _buffers()
    outer, inner, following = [tir.Var("iteration", "int32") for _ in range(3)]
    empty = tir.For(inner, 0, 0, tir.ForKind.SERIAL, _fill(value))
    nested = tir.For(outer, 0, 2, tir.ForKind.SERIAL, tir.SeqStmt([empty, _add(value, value, value)]))
    later = tir.For(following, 0, 2, tir.ForKind.SERIAL, _dma(value, output))
    case = _analyze(_function([value, output], [_fill(value), nested, later]))["cases"][0]
    assert len(case["activations"]) == 1
    keys = [tuple(int(number) for number in key) for key in case["task_keys"]]
    assert len(keys) == len(set(keys)) == 5
    assert [int(number) for number in case["reads"][-1]["segments"][0]["producer_key"]] == [
        2,
        0,
        1,
    ]


def test_boolean_case_product_is_bounded_before_execution():
    value, _ = _buffers()
    predicates = [tir.Var(f"predicate_{index}", "bool") for index in range(13)]
    with pytest.raises(tvm.TVMError, match="case limit exceeded"):
        _analyze(_function([value], _fill(value), scalars=predicates))


@pytest.mark.parametrize("kind", ["readonly", "carry", "consecutive", "nested"])
def test_real_shared_kernel_lowers_through_source_and_final_lifetime_proofs(kind):
    from test_s5_versions_ir import configs
    from test_s6_kernels import make_reuse_kernel

    with tvm.transform.PassContext(config=configs()):
        artifact = tilelang.lower(make_reuse_kernel(kind=kind, trips=2, tiles=2), target="ascendc", platform="A3")
    assert "void main" in artifact.kernel_source or "auto main" in artifact.kernel_source


def _integer_offset_loop(minimum, *, overflow=False):
    value, output = _buffers()
    iteration = tir.Var("large_iteration", "int64")
    offset = iteration * 64 if overflow else (iteration - minimum) * 64
    body = tir.For(
        iteration,
        tir.IntImm("int64", minimum),
        tir.IntImm("int64", 1),
        tir.ForKind.SERIAL,
        tir.SeqStmt([_fill(value, offset=offset), _dma(value, output)]),
        annotations={"num_stages": 1},
    )
    return _function([value, output], body, noalias=True)


@pytest.mark.parametrize("entry", ["analysis", "planner"])
def test_original_singleton_integer_overflow_cannot_fold_into_an_inbounds_zero(entry):
    function = _integer_offset_loop(2**62, overflow=True)
    with pytest.raises(tvm.TVMError, match="overflow|exceeds signed int64"):
        if entry == "analysis":
            _analyze(function)
        else:
            from test_s4_kernels import vector_pass_configs

            function = function.with_attr("npu_platform", "A3").with_attr("target", tvm.target.Target("c -keys=ascend -model=ascendc"))
            with tvm.transform.PassContext(config=vector_pass_configs()):
                transform.PipelinePlanning()(tvm.IRModule.from_expr(function))


@pytest.mark.parametrize("minimum", [0, 3])
def test_checked_original_offsets_keep_small_and_nonzero_origin_cases(minimum):
    case = _analyze(_integer_offset_loop(minimum))["cases"][0]
    assert int(case["writes"][0]["begin"]) == 0
    assert int(case["writes"][0]["end"]) == 256


def test_a_multirepeat_add_cannot_use_shifted_overlapping_input_and_output():
    value, _ = _buffers(160)
    call = _add(value, value, value, count=128).value
    arguments = list(call.args)
    arguments[0] = value.access_ptr("w", offset=32)
    shifted = tir.Evaluate(tir.Call(call.dtype, call.op, arguments))
    with pytest.raises(tvm.TVMError, match="same-task.*overlap"):
        _analyze(_function([value], [_fill(value, count=160), shifted]))


def test_multirepeat_disjoint_views_keep_the_union_definition():
    value, output = _buffers(256)
    call = _add(value, value, value, count=128).value
    arguments = list(call.args)
    arguments[0] = value.access_ptr("w", offset=128)
    disjoint = tir.Evaluate(tir.Call(call.dtype, call.op, arguments))
    case = _analyze(_function([value, output], [_fill(value, count=128), disjoint, _dma(value, output, count=256)]))["cases"][0]
    assert len(case["reads"][-1]["segments"]) == 2
