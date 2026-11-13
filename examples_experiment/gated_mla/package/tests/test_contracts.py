import ast
from functools import lru_cache
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gmla._contracts import (
    INT32_MAX,
    checked_elements,
    validate_attention_spans,
    validate_chain_spans,
    validate_projection_spans,
    validate_split_spans,
    validate_tail_spans,
    validated_host_scale,
)
from gmla._packed import DEFAULT_SCALE, attention_route, checked_scale
from gmla._runtime import DEFAULT_SCALE as RUNTIME_DEFAULT_SCALE, validate_scale


ROOT = Path(__file__).resolve().parents[1]


def extracted(path, name, namespace=None, class_name=None):
    source = ROOT / path
    tree = ast.parse(source.read_text())
    nodes = (
        tree.body
        if class_name is None
        else next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    )
    function = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    scope = {"lru_cache": lru_cache, "math": math, **(namespace or {})}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
    return scope[name]


def validators():
    scope = {"validated_host_scale": validated_host_scale, "DEFAULT_SCALE": RUNTIME_DEFAULT_SCALE}
    return [
        validate_scale,
        checked_scale,
        extracted("gmla/kernels/prefill.py", "_validated_scale", scope),
        extracted("gmla/kernels/zero.py", "validate_scale", scope),
    ]


def test_prefill_mask_buffers_fit_each_declared_tile_dtype_and_pipeline_depth():
    mode = extracted("gmla/kernels/prefill.py", "probability_precision_mode")
    plan = extracted("gmla/kernels/prefill.py", "buffer_plan", {"probability_precision_mode": mode})
    for block_m in (16, 32):
        for block_n in (64, 128):
            for dtype in ("float16", "bfloat16"):
                # pipeline_stages 2 is the cross-wave plan; its slot dimension has to
                # fit the same budget and stay disjoint and 32-byte aligned.
                for residual, stages in ((False, 1), (False, 2), (True, 1), (True, 2)):
                    result = plan(block_m, block_n, dtype=dtype, has_ql_residual=residual, pipeline_stages=stages)
                    half_m = block_m // 2
                    ub = result["UB"]["buffers"]
                    # Packed uint8 mask rows are physically32-byte aligned,
                    # even when a logical row needs only8 or16 bytes.
                    assert ub["block_mask"]["bytes"] >= half_m * 32
                    assert ub["block_select_scratch"]["bytes"] >= half_m * block_n * 4
                    for scope, limit in (("UB", 192 * 1024), ("L1", 512 * 1024), ("L0C", 128 * 1024)):
                        regions = sorted(result[scope]["buffers"].values(), key=lambda row: row["offset"])
                        assert all(row["offset"] % 32 == 0 for row in regions)
                        assert all(left["end"] <= right["offset"] for left, right in zip(regions, regions[1:]))
                        # Reserve the observed worst automatic reduction arena;
                        # generated-source/hardware checks remain separate.
                        scratch = 4096 if scope == "UB" else 0
                        assert result[scope]["total_bytes"] + scratch <= limit


@pytest.mark.parametrize("value", [1e300, 1e-300, 2.0**-150])
def test_scale_rejects_nonrepresentable_float32_in_all_routes(value):
    for validate in validators():
        with pytest.raises(ValueError, match="float32"):
            validate(value)


@pytest.mark.parametrize("value", [float(torch.finfo(torch.float32).max), 2.0**-149, 1e-45, DEFAULT_SCALE, DEFAULT_SCALE * 0.75])
def test_scale_preserves_valid_original_host_value(value):
    for validate in validators():
        assert validate(value) == value


def test_default_scale_and_c3_route_unchanged():
    for validate, expected in zip(validators(), [RUNTIME_DEFAULT_SCALE, DEFAULT_SCALE, RUNTIME_DEFAULT_SCALE, RUNTIME_DEFAULT_SCALE]):
        assert validate(None) == expected
    # Each public path retains its own exact historical default: 1/sqrt(192)
    # and 192**-0.5 differ by one host ULP. Do not canonicalize either here.
    assert validate_scale(None) == RUNTIME_DEFAULT_SCALE
    assert attention_route(1, None) == "c3_decode"
    assert attention_route(1, checked_scale(None)) == "c3_decode"
    assert attention_route(1, checked_scale(DEFAULT_SCALE * 0.75)) == "c4_prefill"


def test_full_element_span_int32_boundary():
    assert checked_elements("edge", INT32_MAX) == INT32_MAX
    with pytest.raises(ValueError, match="element span"):
        checked_elements("overflow", INT32_MAX + 1)
    validate_projection_spans(1, INT32_MAX // 512, 1)
    with pytest.raises(ValueError, match="QL/Z"):
        validate_projection_spans(1, INT32_MAX // 512 + 1, 1)
    validate_projection_spans(1, 1, INT32_MAX // (256 * 512), packed=True)
    validate_projection_spans(1, 1, INT32_MAX // (256 * 512) + 1, packed=False)
    with pytest.raises(ValueError, match="packed weight"):
        validate_projection_spans(1, 1, INT32_MAX // (256 * 512) + 1, packed=True)


@pytest.mark.parametrize("decode", [False, True])
def test_cache_span_guard_rejects_actual_negative_stride_case(decode):
    validate_attention_spans(1, 1, 1, INT32_MAX // 512, decode=decode)
    with pytest.raises(ValueError, match="cache C"):
        validate_attention_spans(1, 1, 1, 4194368, decode=decode)


def test_split_partials_and_actual_tail_spans():
    last_batch = INT32_MAX // (6 * 32 * 512)
    validate_split_spans(last_batch, 1)
    with pytest.raises(ValueError, match="partial Z"):
        validate_split_spans(last_batch + 1, 1)
    tail_batch = INT32_MAX // (32 * 512)
    validate_tail_spans(tail_batch, 1, 32)
    with pytest.raises(ValueError, match="padded output"):
        validate_tail_spans(tail_batch + 1, 1, 32)
    # Source fits: the padded cache-tail buffer is the actual limiting allocation.
    with pytest.raises(ValueError, match="padded output"):
        validate_attention_spans(INT32_MAX // (64 * 512) + 1, 1, 1, 1)


def test_empty_routes_preserve_their_actual_domains():
    validate_chain_spans(1, 0, 1, INT32_MAX)
    qcap = INT32_MAX // 128
    validate_chain_spans(1, qcap, 1, 0)
    with pytest.raises(ValueError, match="empty-cache output"):
        validate_chain_spans(1, qcap + 1, 1, 0)
    # The same QL span would be invalid on a nonempty attention route.
    with pytest.raises(ValueError, match="QL/Z"):
        validate_chain_spans(1, qcap, 1, 1)


def test_append_raw_source_span_is_checked_only_when_used():
    tq = INT32_MAX // 576 + 1
    validate_chain_spans(1, tq, 1, 1, packed=True)
    with pytest.raises(ValueError, match="rawKV"):
        validate_chain_spans(1, tq, 1, 1, packed=True, append_sources=True)


class ShapeTensor:
    requires_grad = False

    def __init__(self, shape, dtype, device):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.ndim = len(shape)

    def is_contiguous(self):
        return True


def fail_if_called(*args, **kwargs):
    raise AssertionError("view/provider/append reached before shape rejection")


def test_separate_rejects_unsafe_span_before_first_provider_or_view():
    device = SimpleNamespace(type="npu", index=0)
    fake_torch = SimpleNamespace(Tensor=ShapeTensor, is_grad_enabled=lambda: False, float16="fp16", bfloat16="bf16", int32="i32")
    validate = extracted(
        "gmla/_runtime.py",
        "validate_forward_inputs",
        {
            "torch": fake_torch,
            "validate_scale": validate_scale,
            "DEFAULT_SCALE": DEFAULT_SCALE,
            "validate_chain_spans": validate_chain_spans,
            "_view": fail_if_called,
        },
    )
    bundle = SimpleNamespace(
        layout_module=SimpleNamespace(require_dense_base=lambda *a, **k: None),
        projections=SimpleNamespace(q_absorb=fail_if_called),
        block_m=32,
        block_n=64,
        max_blocks=20,
    )
    dims = [(1, 1, 1, 128), (1, 1, 1, 64), (1, 4194368, 512), (1, 4194368, 64), (1, 128, 512), (1, 128, 512), (1, 1, 1, 128)]
    tensors = [ShapeTensor(shape, "fp16", device) for shape in dims]
    with pytest.raises(ValueError, match="cache C"):
        validate(*tensors, ShapeTensor((1, 8), "i32", device), layout_module=bundle.layout_module)


def test_packed_rejects_before_view_append_or_commit():
    from gmla._packed import PackedModelAdapter

    owner = SimpleNamespace(
        B=1, capacity=17, dtype=torch.float16, device=torch.device("cpu"), lengths=(0,), generation=0, launches=1, append=fail_if_called
    )
    core = PackedModelAdapter.__new__(PackedModelAdapter)
    core.owner, core.heads = owner, 16384
    core.parts = SimpleNamespace(
        layout=SimpleNamespace(require_dense_base=lambda *a, **k: None, guarded_view=fail_if_called), block_m=32, block_n=64, max_blocks=20
    )
    core.check_context = lambda: None
    core._check_cache_formats = lambda: None
    query = ShapeTensor((1, 17, 16384, 192), owner.dtype, owner.device)
    weight = ShapeTensor((16384, 256, 512), owner.dtype, owner.device)
    gate = ShapeTensor((1, 17, 16384, 128), owner.dtype, owner.device)
    before = owner.lengths, owner.generation, owner.launches
    with torch.no_grad(), pytest.raises(ValueError, match="packed weight"):
        core.append_and_forward(query, weight, None, None, gate, [17])
    assert (owner.lengths, owner.generation, owner.launches) == before


@pytest.mark.parametrize(
    "filename,name,kwargs,guard_name,guard",
    [
        (
            "decode.py",
            "build_cube_attention",
            {"batch": 1, "heads": 1, "cache_capacity": 4194368},
            "validate_attention_spans",
            validate_attention_spans,
        ),
        (
            "prefill.py",
            "build_cube_attention",
            {"batch": 1, "heads": 1, "query_capacity": 1, "cache_capacity": 4194368},
            "validate_attention_spans",
            validate_attention_spans,
        ),
    ],
)
def test_exposed_attention_factories_reject_before_tir_or_compilation(filename, name, kwargs, guard_name, guard):
    factory = extracted(
        "gmla/kernels/" + filename,
        name,
        {
            guard_name: guard,
            "_validated_scale": validate_scale,
            "buffer_plan": fail_if_called,
        },
    )
    with pytest.raises(ValueError, match="cache C"):
        factory(**kwargs)


@pytest.mark.parametrize("filename", ["vector.py", "packed.py", "query_residual.py"])
def test_vector_plans_reject_actual_negative_output_stride_case(filename):
    plan = extracted("gmla/kernels/" + filename, "projection_plan", {"validate_projection_spans": validate_projection_spans})
    with pytest.raises(ValueError, match="QL/Z"):
        plan(1, 4194305, 1, 8)


def test_direct_packed_residual_factory_checks_packed_weight_span():
    factory = extracted(
        "gmla/kernels/query_residual.py",
        "q_absorb_packed_residual_kernel",
        {
            "_target": lambda target: None,
            "DTYPES": {torch.float16: "float16", torch.bfloat16: "bfloat16"},
            "validate_projection_spans": validate_projection_spans,
            "projection_plan": fail_if_called,
        },
    )
    with pytest.raises(ValueError, match="packed weight"):
        factory(1, 1, 16384)
