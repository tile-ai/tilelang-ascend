"""Original software-pipelined SiTU factory with an empty C counterpart."""

import importlib.util
from pathlib import Path

import pytest
import tilelang
from tilelang import tvm


@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
def test_original_situ_factory_plans_injects_and_emits(dtype):
    path = Path(__file__).parent / "fixtures" / "a2a3_situ_glu_grad.py"
    spec = importlib.util.spec_from_file_location("_a2a3_situ_glu_grad", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = module.situ_glu_grad_kernel(256, 6144, 3072, 32, 128, dtype=dtype, tiles_per_block=12)
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    seen = {}

    @tvm.instrument.pass_instrument
    class Trace:
        def run_after_pass(self, mod, info):
            if info.name == "tl.A2A3PipelinePlanning":
                loops = []
                for f in mod.functions.values():
                    tvm.tirx.stmt_functor.post_order_visit(f.body, lambda n: loops.append(n) if isinstance(n, tvm.tirx.For) else None)
                seen["planned_loops"] = sum("software_pipeline_stage" in loop.annotations for loop in loops)
            if info.name == "tl.A2A3InjectSoftwarePipeline":
                seen["injected"] = True

    with tvm.transform.PassContext(
        opt_level=3,
        config={"tl.ascend_auto_sync": True, "tl.ascend_memory_planning": True},
        instruments=[Trace()],
    ):
        artifact = tilelang.lower(func, target=target)
    assert seen["planned_loops"] >= 1
    assert seen["injected"]
    assert len(artifact.params) == 3
    assert "copy_gm_to_ub" in artifact.kernel_source
    assert "Sigmoid" in artifact.kernel_source
