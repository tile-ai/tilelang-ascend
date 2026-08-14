"""cann_bench package: SwiGLU operator for CANN Bench."""

__version__ = "1.0.0"

import tilelang

# Disable tilelang disk cache: cann-bench runs each case in an independent
# subprocess. Symbolic M/N kernels compiled once must not rely on cross-process
# disk cache state (stale / concurrent access). In-process _kernel_cache handles
# reuse within a single subprocess.
tilelang.disable_cache()

from .swi_glu import swi_glu  # noqa: E402

__all__ = ["swi_glu"]
