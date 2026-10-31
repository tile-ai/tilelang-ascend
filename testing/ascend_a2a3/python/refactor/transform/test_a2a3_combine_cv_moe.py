"""Full original MoE factories that exposed dead UB bindings in the C clone."""

import importlib.util
from pathlib import Path

import pytest
import tilelang
from tilelang import tvm


def _factory():
    path = Path(__file__).parent / "fixtures" / "a2a3_moe_backward.py"
    spec = importlib.util.spec_from_file_location("_a2a3_moe_backward", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_topk_sum_and_topk_group_idx_backward_kernel


@pytest.mark.parametrize("groups,experts,topk,sumk", [(12, 6, 2, 1), (4, 64, 4, 2), (8, 9, 2, 2), (8, 32, 4, 1), (16, 16, 2, 2)])
def test_original_moe_backward_full_pipeline(groups, experts, topk, sumk):
    func = _factory()(groups, experts, topk, sumk)
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    with tvm.transform.PassContext(opt_level=3, config={"tl.ascend_auto_cv_combine": True}):
        artifact = tilelang.lower(func, target=target)
    final = next(iter(artifact.device_mod.functions.values()))
    assert len(final.params) == 4
    assert len(final.buffer_map) == 4
    assert "TopK" in artifact.kernel_source
    assert "grad_scores" in artifact.kernel_source
    # Full lowering includes the unchanged resource verifier.  Retained vector
    # reads/stores and four user buffers ensure this is not an empty kernel.
    nodes = []
    tvm.tirx.stmt_functor.post_order_visit(final.body, nodes.append)
    assert any(isinstance(node, tvm.tirx.BufferLoad) and node.buffer.scope() == "shared.ub" for node in nodes)
