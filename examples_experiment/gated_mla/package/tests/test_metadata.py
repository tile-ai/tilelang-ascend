import pytest
import torch
import ast
from pathlib import Path
from types import SimpleNamespace

from gmla import build_packed_adapter, build_runtime, create_cache, metadata_for_reference, prepare_metadata, validate_runtime_metadata_cpu
from gmla._cache_state import append_transition, read_transition, reject_storage_aliases
from gmla._packed import require_inference


@pytest.mark.parametrize("length", [0, 1, 17, 127, 128, 129, 4096, 16384])
def test_lengths_are_capacities_not_a_128_limit(length):
    meta = prepare_metadata([[length, length, 0]], batch=1, query_capacity=length, cache_capacity=length)
    assert meta.device.type == "cpu" and meta.dtype == torch.int32 and meta.shape == (1, 8)
    validate_runtime_metadata_cpu(meta, B=1, Tq=length, Tk=length)
    assert metadata_for_reference(meta)["q_lens"] == (length,)


@pytest.mark.parametrize("rows", [[[True, 17, 0]], [[18, 17, 0]], [[1, 17, 17]], [[0, 0, -1]], [[1, 17, 0, 1, 0, 0, 0, 0]]])
def test_invalid_metadata_rejected(rows):
    with pytest.raises((TypeError, ValueError)):
        prepare_metadata(rows, batch=1, query_capacity=17, cache_capacity=17)


def test_device_metadata_is_never_read_as_cpu_lengths():
    with pytest.raises(TypeError):
        validate_runtime_metadata_cpu(torch.empty((1, 8), device="meta", dtype=torch.int32), B=1, Tq=1, Tk=17)
    with pytest.raises(TypeError):
        prepare_metadata(torch.zeros((1, 3), dtype=torch.int32), batch=1, query_capacity=1, cache_capacity=17)


def test_owner_transition_commit_and_read_protocol():
    meta, lengths = append_transition((0, 0), (17, 0), Qcap=17, capacity=65)
    assert lengths == (17, 0) and meta.tolist()[0][:3] == [17, 17, 0]
    second, lengths = append_transition(lengths, (9, 17), Qcap=17, capacity=65)
    assert lengths == (26, 17) and second.tolist()[0][:3] == [9, 26, 17]
    assert read_transition(lengths, (1, 0), (25, 17), Qcap=1, capacity=65).tolist()[1][:3] == [0, 17, 17]
    with pytest.raises(ValueError):
        append_transition(lengths, (40, 0), Qcap=40, capacity=65)
    with pytest.raises(ValueError):
        append_transition(lengths, (1, 0), Qcap=1, capacity=65, metadata_cpu=meta)
    cache = torch.empty(2, 65, 512)
    with pytest.raises(ValueError):
        reject_storage_aliases(cache, cache.view_as(cache))


def test_default_backend_and_unsupported_alias():
    assert build_runtime().target == "pto"
    assert build_runtime(target="ascendc").target == "ascendc"
    with pytest.raises(ValueError):
        build_runtime(target="auto")


def test_inference_and_cpu_rejection_before_kernel_import():
    runtime = build_runtime()
    x = torch.empty((1, 1, 1, 128), requires_grad=True)
    with pytest.raises(RuntimeError, match="inference-only"):
        runtime.forward(x, x, x, x, x, x, x, x)
    with torch.no_grad(), pytest.raises(ValueError, match="NPU"):
        runtime.forward(x, x, x, x, x, x, x, x)


def test_packed_inference_guard_runs_before_owner_or_npu_access():
    with torch.enable_grad(), pytest.raises(RuntimeError, match="no_grad"):
        build_packed_adapter(object(), heads=96)
    with torch.enable_grad(), pytest.raises(RuntimeError, match="no_grad"):
        create_cache(1, 17)


def _cache_owner_class_without_kernel_import(guard=None):
    # Execute the exact packaged class AST without importing TileLang. The
    # negative paths must reject before they need an owner/NPU/kernel object.
    path = Path(__file__).resolve().parents[1] / "gmla" / "_cache_owner.py"
    owner = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "CacheOwner")
    scope = {"torch": torch, "require_inference": require_inference, "require_dense_base": guard}
    exec(compile(ast.Module(body=[owner], type_ignores=[]), str(path), "exec"), scope)
    return scope["CacheOwner"]


@pytest.mark.parametrize("method", ["append", "read_existing"])
def test_direct_owner_inference_guard_precedes_owner_or_device_access(method):
    owner_class = _cache_owner_class_without_kernel_import()
    with torch.enable_grad(), pytest.raises(RuntimeError, match="no_grad"):
        if method == "append":
            owner_class.append(object(), None, None, None)
        else:
            owner_class.read_existing(object(), None, None, Qcap=1)


def test_owner_tensor_validation_calls_physical_guard():
    class PhysicalFormatRejected(ValueError):
        pass

    calls = []

    def reject(value, **kwargs):
        calls.append((value, kwargs))
        raise PhysicalFormatRejected("physical format")

    owner_class = _cache_owner_class_without_kernel_import(reject)
    owner = SimpleNamespace(dtype=torch.float16, device=torch.device("cpu"))
    value = torch.empty((1, 17, 512), dtype=torch.float16)
    with pytest.raises(PhysicalFormatRejected):
        owner_class._tensor(owner, "Cnorm", value, (1, 17, 512))
    assert len(calls) == 1
    assert calls[0][1] == {"name": "Cnorm", "shape": (1, 17, 512), "dtype": torch.float16, "device": torch.device("cpu")}
