"""cann_bench package: Mish operator for CANN Bench."""

__version__ = "1.0.0"

import tilelang

# Clear tilelang disk cache at import: cann-bench runs each case in an
# independent subprocess, so stale cross-process disk cache state must be
# flushed. In-process _kernel_cache handles reuse within a single subprocess.
tilelang.cache.clear_cache()

from .mish import mish  # noqa: E402

__all__ = ["mish"]
