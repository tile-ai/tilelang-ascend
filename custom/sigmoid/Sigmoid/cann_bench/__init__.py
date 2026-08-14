"""cann_bench package: Sigmoid operator for CANN Bench."""

__version__ = "1.0.0"

import tilelang

# Clear tilelang disk cache at import: cann-bench runs each case in an
# independent subprocess, so stale cross-process disk cache state must be
# flushed. In-process _kernel_cache handles reuse within a single subprocess.
#
# NOTE: tilelang.disable_cache() is NOT used here because it triggers a
# tilelang compiler stall when combined with T.tile.sigmoid (works fine with
# T.tile.silu as in SwiGLU, but T.tile.sigmoid has a compiler bug under
# disabled-cache mode). clear_cache() achieves the same stale-cache avoidance
# without triggering the stall.
tilelang.cache.clear_cache()

from .sigmoid import sigmoid  # noqa: E402

__all__ = ["sigmoid"]
