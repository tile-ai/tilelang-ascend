"""Both A2/A3 source variants use the shared Cython execution backend."""

from tilelang.backend.execution_backend import ExecutionBackendSpec

EXECUTION_BACKENDS = (ExecutionBackendSpec("cython"),)

# Per-request attributes are stamped on a derived PrimFunc before cache lookup.
# They are data only; the target registry does not import the compiler eagerly.
WORKSPACE_IDX_ATTR = "tilelang.a2a3.workspace_idx"
COMPILER_CONFIG_ATTR = "tilelang.a2a3.compiler_config"
