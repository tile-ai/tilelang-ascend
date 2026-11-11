"""Frozen original protocol definitions; see protocol_manifest.json."""

import math
import torch

SCALE = 1 / math.sqrt(192)
PAGE = 128


def qualify(actual, expected, dtype, golden):
    record = {"qualified": False}
    try:
        if tuple(actual.shape) != (1, 1, 96, 128) or actual.dtype != dtype:
            raise AssertionError(f"Unexpected output shape/dtype: {actual.shape}, {actual.dtype}")
        values = actual.detach().cpu().double()
        if not torch.isfinite(values).all() or not torch.isfinite(expected).all():
            raise AssertionError("Nonfinite output/reference")
        error = (values - expected).abs()
        failed = error > golden.FP16_ATOL + golden.FP16_RTOL * expected.abs()
        amplitude = float(expected.abs().max())
        record.update(
            max_abs=float(error.max()),
            max_normalized=float(error.max()) / max(amplitude, 1e-9),
            elementwise_failures=int(failed.sum()),
            failed_indices=failed.nonzero().tolist()[:32],
        )
        record["qualified"] = not bool(failed.any()) and record["max_normalized"] <= golden.FP16_MAX_NORMALIZED_ERROR
        if not record["qualified"]:
            record["reason"] = "output exceeded unchanged original seven-input FP64 budget"
    except Exception as error:
        record.update(error_type=type(error).__name__, reason=str(error))
    return record


class FiaComposition:
    """Official latent attention plus FP32 torch projections/gate, all per-call.

    Static preparation is limited to identity page indices and host metadata.
    Weight/input casts, query projection, QL rounding, latent cast, value
    projection, sigmoid, multiplication and final cast remain inside __call__.
    """

    def __init__(self, api, golden, inputs, host_meta, block_table):
        self.api, self.golden, self.inputs = api, golden, inputs
        self.host_meta, self.block_table = host_meta, block_table
        self.visible_lengths = [min(host_meta["kv_lens"][0], host_meta["query_start"][0] + 1)]

    def __call__(self):
        x = self.inputs
        spec = self.golden.validate_inputs(**x, **self.host_meta, require_npu=True, allowed_dtypes=(torch.float16, torch.bfloat16))
        if (spec.B, spec.H, spec.Tq, spec.Tk, spec.D, spec.E, spec.R, spec.V) != (1, 96, 1, 4096, 128, 64, 512, 128):
            raise ValueError("This benchmark only supports its declared fixed full-chain domain")
        ql = torch.bmm(x["q_content"].view(96, 1, 128).float(), x["w_k"].float()).to(x["q_content"].dtype)
        cache = x["latent_kv"].view(4096 // PAGE, 1, PAGE, 512)
        auxiliary = x["key_aux"].view(4096 // PAGE, 1, PAGE, 64)
        latent, _ = self.api(
            ql.view(1, 96, 1, 512),
            cache,
            cache,
            query_rope=x["q_aux"].view(1, 96, 1, 64),
            key_rope=auxiliary,
            block_table=self.block_table,
            block_size=PAGE,
            actual_seq_kvlen=self.visible_lengths,
            num_query_heads=96,
            num_key_value_heads=1,
            input_layout="BNSD",
            softmax_scale=spec.scale,
            sparse_mode=0,
        )
        values = torch.bmm(x["w_v"].float(), latent.view(96, 512, 1).float()).view(1, 1, 96, 128)
        return (values * torch.sigmoid(x["gate_logits"].float())).to(x["q_content"].dtype)


def cpu_fixture(fixture, golden, dtype, seed, gate_case, visible):
    shape = fixture.CASES["model_4k"]
    if (shape["B"], shape["H"], shape["Tk"]) != (1, 96, 4096):
        raise ValueError("Unexpected fixture domain")
    inputs = fixture.make_inputs(shape, seed, dtype, gate_case)
    metadata = {"q_lens": (1,), "kv_lens": (4096,), "query_start": (visible - 1,)}
    expected = golden.expanded_reference(**inputs, **metadata)
    absorbed = golden.absorbed_reference(**inputs, **metadata)
    torch.testing.assert_close(expected, absorbed, rtol=2e-12, atol=2e-12)
    return inputs, metadata, expected


def hardware_error(error):
    return any(x in str(error) for x in ("507015", "aicore execution is abnormal", "timeout or trap"))
