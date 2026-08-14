"""cann_bench package: ForeachNorm operator for CANN Bench."""

__version__ = "1.0.0"

import tilelang

# Clear tilelang disk cache at import: cann-bench runs each case in an
# independent subprocess, so stale cross-process disk cache state must be
# flushed. In-process _kernel_cache handles reuse within a single subprocess.
tilelang.cache.clear_cache()

from .foreach_norm import foreach_norm  # noqa: E402

__all__ = ["foreach_norm"]
