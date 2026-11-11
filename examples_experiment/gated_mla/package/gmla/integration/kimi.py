"""Optional frozen Kimi layer bridge; external projections and W_O are explicit."""

import torch
from .._packed import require_inference, attention_route, CacheCommittedForwardError
from .official_layer_fixture import prepare_external_projections, validate_frozen_layer


class FrozenKimiLayerAdapter:
    """One frozen layer's external projections/core/W_O composition, inference only."""

    def __init__(self, layer, packed_core):
        validate_frozen_layer(layer)
        if layer.num_heads != packed_core.heads:
            raise ValueError("official layer and packed core head counts differ")
        self.layer, self.core = layer, packed_core

    def forward_new_tokens(self, hidden_states, q_lens, *, return_intermediates=False):
        require_inference()
        self.core.check_context()
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
            raise ValueError("hidden_states must be [B,Tq,hidden_size]")
        if hidden_states.shape[0] != self.core.owner.B or hidden_states.shape[2] != self.layer.hidden_size:
            raise ValueError("hidden_states batch/hidden size differs from the bound layer/cache")
        Tq = int(hidden_states.shape[1])
        attention_route(Tq, self.layer.scaling)
        if not isinstance(q_lens, (tuple, list)) or len(q_lens) != self.core.owner.B or any(type(x) is not int for x in q_lens):
            raise TypeError("q_lens must be CPU-owned Python integers")
        if any(count < 0 or count > Tq or old + count > self.core.owner.capacity for old, count in zip(self.core.owner.lengths, q_lens)):
            raise ValueError("query lengths exceed the physical query/cache capacity")
        self.core.parts.layout.require_dense_base(
            hidden_states, name="hidden_states", dtype=self.core.owner.dtype, device=self.core.owner.device
        )
        self.core.parts.layout.require_dense_base(
            self.layer.kv_b_proj.weight, name="original kv_b.weight", dtype=self.core.owner.dtype, device=self.core.owner.device
        )
        prepared = prepare_external_projections(self.layer, hidden_states, guard=self.core.parts.layout)
        gated, stages, audit = self.core.append_and_forward(*prepared, q_lens, scale=self.layer.scaling, return_intermediates=True)
        try:
            before_wo = self.core.parts.layout.guarded_view(
                gated, (gated.shape[0], gated.shape[1], self.layer.num_heads * 128), name="W_O input"
            )
            output = self.layer.o_proj(before_wo)
            self.core.synchronize()
        except Exception as error:
            raise CacheCommittedForwardError(self.core.owner) from error
        if return_intermediates:
            return output, {"before_wo": before_wo, "prepared": prepared, **stages}, audit
        return output
