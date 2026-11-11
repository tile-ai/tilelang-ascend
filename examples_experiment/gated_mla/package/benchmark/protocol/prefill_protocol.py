"""Frozen original protocol definitions; see protocol_manifest.json."""

import torch

BUDGET = {"rtol": 0.01, "atol": 0.002, "max_normalized": 0.005}


def metrics(actual, reference):
    a, r = actual.detach().cpu().double(), reference.detach().cpu().double()
    if a.shape != r.shape or not torch.isfinite(a).all() or not torch.isfinite(r).all():
        raise AssertionError("shape or finite-output contract failed")
    difference = (a - r).abs()
    max_abs = float(difference.max())
    normalized = max_abs / max(float(r.abs().max()), 1e-9)
    failures = int(torch.count_nonzero(difference > BUDGET["atol"] + BUDGET["rtol"] * r.abs()))
    return {
        "max_abs": max_abs,
        "max_normalized": normalized,
        "elementwise_out_of_budget": failures,
        "passed": failures == 0 and normalized <= BUDGET["max_normalized"],
    }


def graph_error_policy(error):
    message = str(error).lower()
    fatal_markers = (
        "aicore",
        "aicpu",
        "device-side assert",
        "device lost",
        "device is lost",
        "device hung",
        "task exception",
        "timeout or trap",
        "watchdog",
        "stream synchronize failed",
    )
    found = [marker for marker in fatal_markers if marker in message]
    if found:
        return {"continue": False, "category": "hardware_fatal_marker", "markers": found}
    graph_context = any(word in message for word in ("graph", "capture"))
    if graph_context and (
        isinstance(error, AttributeError) or any(word in message for word in ("not supported", "unsupported", "not implemented"))
    ):
        return {"continue": True, "category": "graph_compatibility_unavailable", "markers": []}
    # Unknown failures are not assumed to leave the NPU safe for more calls.
    return {"continue": False, "category": "unclassified_graph_error_abort", "markers": []}


class OfficialFullChains:
    """Explicit NPU compositions. Projection/cast/packing/gate costs stay inside calls."""

    def __init__(self, inputs, mask, block_table, api, scale):
        self.x, self.mask, self.block_table, self.api, self.scale = inputs, mask, block_table, api, scale
        self.common = dict(
            actual_seq_qlen=[128],
            actual_seq_kvlen=[128],
            num_query_heads=96,
            input_layout="TND",
            softmax_scale=scale,
            sparse_mode=3,
            return_softmax_lse=False,
        )

    def mla_paged(self):
        x = self.x
        dtype = x["q_content"].dtype
        ql = torch.einsum("bthd,hdr->bthr", x["q_content"].float(), x["w_k"].float()).to(dtype)
        cache = x["latent_kv"].view(1, 1, 128, 512)
        auxiliary = x["key_aux"].view(1, 1, 128, 64)
        latent, _ = self.api(
            ql.view(128, 96, 512).contiguous(),
            cache,
            cache,
            query_rope=x["q_aux"].view(128, 96, 64),
            key_rope=auxiliary,
            block_table=self.block_table,
            block_size=128,
            atten_mask=self.mask,
            num_key_value_heads=1,
            **self.common,
        )
        value = torch.einsum("thr,hvr->thv", latent.float(), x["w_v"].float())
        return (value * torch.sigmoid(x["gate_logits"][0].float())).to(dtype).contiguous().unsqueeze(0)

    def expanded_standard(self):
        x = self.x
        dtype = x["q_content"].dtype
        c = x["latent_kv"].float()
        keys = torch.einsum("bkr,hdr->bkhd", c, x["w_k"].float()).to(dtype)
        values = torch.einsum("bkr,hvr->bkhv", c, x["w_v"].float()).to(dtype)
        query = torch.cat((x["q_content"][0], x["q_aux"][0]), dim=-1)
        key = torch.cat((keys[0], x["key_aux"][0].unsqueeze(1).expand(-1, 96, -1)), dim=-1)
        value = values[0].contiguous()  # required FIA TND packing, timed explicitly
        output, _ = self.api(query, key, value, atten_mask=self.mask, num_key_value_heads=96, **self.common)
        return (output.float() * torch.sigmoid(x["gate_logits"][0].float())).to(dtype).contiguous().unsqueeze(0)
