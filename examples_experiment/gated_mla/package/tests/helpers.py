from functools import lru_cache

import torch

from gmla import DEFAULT_SCALE, build_runtime, prepare_metadata
from gmla import reference

from . import fixture_decode, fixtures


@lru_cache(maxsize=24)
def cpu_inputs(B, H, Tq, Tk, dtype, exact_decode=False):
    spec = dict(B=B, H=H, Tq=Tq, Tk=Tk)
    if exact_decode:
        return fixture_decode.make_inputs(spec, 0, dtype, "head_specific")
    return fixtures.make_inputs(spec, dtype, 0, "head_specific")


def runtime_for(target):
    return build_runtime() if target is None else build_runtime(target=target)


def metadata(rows, inputs):
    return prepare_metadata(
        rows, batch=inputs["q_content"].shape[0], query_capacity=inputs["q_content"].shape[1], cache_capacity=inputs["latent_kv"].shape[1]
    )


def refs(inputs, cpu_meta, scale=DEFAULT_SCALE):
    return fixtures.references(inputs, cpu_meta, reference, scale)


def verify_result(output, intermediate, audit, expected, dtype, target):
    assert output.dtype == dtype and tuple(output.shape) == tuple(expected["expanded"].shape)
    reference.assert_fp16_close(output, expected["expanded"])
    reference.assert_fp16_close(output, expected["absorbed"])
    high, low = intermediate["q_latent_hi"], intermediate["q_latent_lo"]
    if high is not None:
        logical = high.cpu().double() + (low.cpu().double() if low is not None else 0)
        reference.assert_fp16_close(logical, expected["q_latent"])
        reference.assert_fp16_close(intermediate["latent_out"], expected["latent_out"])
    assert all(str(kernel.target) == (target or "pto") for kernel in audit["kernels"].values())


def execute_case(inputs, rows, target, *, repeat=3, scale=DEFAULT_SCALE):
    runtime = runtime_for(target)
    cpu_meta = metadata(rows, inputs)
    expected = refs(inputs, cpu_meta, scale)
    device = {name: value.npu() for name, value in inputs.items()}
    device_meta = cpu_meta.npu()
    first = None
    with torch.no_grad():
        for _ in range(repeat):
            output, intermediate, audit = runtime.forward(**device, metadata=device_meta, scale=scale, return_intermediates=True)
            torch.npu.synchronize()
            verify_result(output, intermediate, audit, expected, inputs["q_content"].dtype, target)
            if first is not None:
                torch.testing.assert_close(output.cpu(), first, rtol=0, atol=0)
            first = output.cpu()
    fixtures.verify_read_only(device, inputs, device_meta, cpu_meta)
    return runtime, device, device_meta, output, intermediate, audit
