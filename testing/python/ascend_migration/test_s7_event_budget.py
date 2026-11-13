"""Independent interval oracle for the A3 allocator; no compiler or device invocation."""

import copy

import pytest

import tilelang
from tilelang import tvm


def request(begin=0, end=8, *, instance=0, src="MTE2", dst="V", **extra):
    return {
        "core": 1,
        "instance": instance,
        "src": src,
        "dst": dst,
        "begin": tvm.tir.IntImm("int64", begin),
        "end": tvm.tir.IntImm("int64", end),
        **extra,
    }


def domain(row):
    return (int(row["core"]), int(row.get("instance", 0)), str(row["src"]), str(row["dst"]))


def allocate(rows, platform="A3"):
    payload = tvm.runtime.convert(rows)
    before = tvm.ir.save_json(payload)
    result = tilelang.transform.AllocateAscendSyncEvents(payload, platform)
    assert tvm.ir.save_json(payload) == before
    assert int(result["schema_version"]) == 1
    assert len(result["allocations"]) == len(rows)
    assignments = {int(item["request"]): int(item["id"]) for item in result["allocations"]}
    assert set(assignments) == set(range(len(rows)))
    reserved = {}
    for row in rows:
        reserved.setdefault(domain(row), set()).update(int(x) for x in row.get("reserved_ids", []))
    for index, row in enumerate(rows):
        slot = assignments[index]
        assert 0 <= slot < 8 and slot not in reserved[domain(row)]
        if "fixed_id" in row:
            assert slot == int(row["fixed_id"])
        for other in range(index):
            previous = rows[other]
            if domain(row) == domain(previous) and max(int(row["begin"]), int(previous["begin"])) < min(
                int(row["end"]), int(previous["end"])
            ):
                assert slot != assignments[other]
    return result


@pytest.mark.parametrize("count", [7, 8])
def test_actual_simultaneous_peak_before_and_at_bank_capacity(count):
    result = allocate([request() for _ in range(count)])
    record = result["domains"][0]
    assert int(record["capacity"]) == 8 and int(record["peak_required"]) == count
    assert set(map(int, record["available_ids"])) == set(range(8))


def test_ninth_simultaneous_interval_cannot_wrap_an_active_id():
    with pytest.raises(tvm.TVMError, match="[Ee]vent|budget|capacity|exhaust"):
        allocate([request() for _ in range(9)])


def test_half_open_boundaries_reuse_slots_without_accumulating_total_request_count():
    rows = [request(index, index + 1) for index in range(40)]
    result = allocate(rows)
    assert int(result["domains"][0]["peak_required"]) == 1


def test_directed_pipe_pair_and_physical_instance_are_independent_domains():
    rows = (
        [request(instance=0) for _ in range(8)] + [request(instance=1) for _ in range(8)] + [request(src="V", dst="MTE2") for _ in range(8)]
    )
    result = allocate(rows)
    assert len(result["domains"]) == 3
    assert all(int(row["peak_required"]) == 8 for row in result["domains"])


def test_reserved_ids_are_a_domain_wide_union_and_fixed_id_is_preserved():
    rows = [request(0, 2, reserved_ids=[0, 2]), request(2, 4, reserved_ids=[4], fixed_id=7)]
    result = allocate(rows)
    record = result["domains"][0]
    assert set(map(int, record["reserved_ids"])) == {0, 2, 4}
    assert set(map(int, record["available_ids"])) == {1, 3, 5, 6, 7}


@pytest.mark.parametrize(
    "rows",
    [
        [request(fixed_id=3), request(fixed_id=3)],
        [request(fixed_id=3, reserved_ids=[3])],
        [request(reserved_ids=list(range(8)))],
    ],
)
def test_explicit_conflicts_and_exhausted_reservations_are_rejected(rows):
    with pytest.raises(tvm.TVMError):
        allocate(rows)


def test_disjoint_explicit_lifetimes_can_reuse_the_same_id():
    allocate([request(0, 1, fixed_id=5), request(1, 2, fixed_id=5)])


@pytest.mark.parametrize(
    "field,value",
    [
        ("core", 0),
        ("core", 2),
        ("core", True),
        ("instance", -1),
        ("instance", True),
        ("src", "S"),
        ("src", "MTE1"),
        ("dst", "MTE2"),
        ("begin", -1),
        ("begin", 8),
        ("end", 0),
        ("begin", tvm.tir.Var("unknown", "int64")),
        ("end", tvm.tir.FloatImm("float32", 8)),
        ("fixed_id", -1),
        ("fixed_id", 8),
        ("fixed_id", True),
        ("fixed_id", tvm.tir.IntImm("uint32", 1)),
        ("reserved_ids", [8]),
        ("reserved_ids", [True]),
    ],
)
def test_unknown_invalid_or_wrong_core_requests_are_rejected(field, value):
    row = request()
    row[field] = value
    with pytest.raises(tvm.TVMError):
        allocate([row])


@pytest.mark.parametrize("platform", ["A5", "unknown"])
def test_allocator_platform_gate_does_not_relabel_other_architectures(platform):
    with pytest.raises(tvm.TVMError):
        allocate([request()], platform)


def test_empty_request_list_has_no_events_and_missing_fields_still_reject():
    result = allocate([])
    assert not result["allocations"] and not result["domains"]
    row = copy.copy(request())
    del row["src"]
    with pytest.raises(tvm.TVMError):
        allocate([row])


def test_fixed_ahead_of_flexible_requests_is_reserved_before_greedy_allocation():
    # Request order must not make the flexible allocation steal a later fixed ID.
    rows = [request(), request(fixed_id=0)]
    allocate(rows)
