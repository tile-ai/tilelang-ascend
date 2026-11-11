"""Seven-input inference API for the isolated selected-kernel package."""

import hashlib
from pathlib import Path

from ._runtime import DEFAULT_SCALE as DEFAULT_SCALE, validate_forward_inputs
from ._contracts import public_prefill_block_n


def validate_target(target):
    if target not in ("pto", "ascendc"):
        raise ValueError("target must be 'pto' or 'ascendc'")
    return target


class GatedMLARuntime:
    """Fixed query/value providers and decode/prefill routes, with lazy imports.

    Compilation and NPU execution start on forward/factory use, not construction.
    CPU owner validates metadata contents before explicitly transferring them.
    """

    def __init__(self, *, target="pto"):
        self._target = validate_target(target)

    @property
    def target(self):
        return self._target

    @property
    def kernel_bundle(self):
        """Diagnostic/precompile access to selected factories; never launches here."""
        from ._selection import selected_bundle

        return selected_bundle(self.target)

    def forward(self, q_content, q_aux, latent_kv, key_aux, w_k, w_v, gate_logits, metadata, *, scale=None, return_intermediates=False):
        tensors = (q_content, q_aux, latent_kv, key_aux, w_k, w_v, gate_logits, metadata)
        contract = validate_forward_inputs(*tensors, scale=scale, block_n_selector=public_prefill_block_n)
        if public_prefill_block_n(*contract) == 128:
            from ._selection import selected_prefill128_bundle

            bundle = selected_prefill128_bundle(self.target)
        else:
            bundle = self.kernel_bundle
        return bundle._forward_validated(*tensors, contract, return_intermediates=return_intermediates)


def build_runtime(*, target="pto"):
    return GatedMLARuntime(target=target)


def kernel_report(audit):
    """Read generated sources outside timed calls; preserve actual selected targets."""
    return {
        name: {"target": str(kernel.target), "source_sha256": hashlib.sha256(kernel.get_kernel_source().encode()).hexdigest()}
        for name, kernel in audit["kernels"].items()
    }


def package_sources():
    """Metadata-only source inventory; does not load TileLang or touch an NPU."""
    root = Path(__file__).resolve().parent
    return {
        str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.rglob("*.py"))
    }
