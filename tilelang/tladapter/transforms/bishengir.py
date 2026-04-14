# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""BishengIR dialect transformation passes."""

from tilelang.tladapter.utils import pass_fn

adapt_triton_kernel = pass_fn("adapt-triton-kernel")
bind_workspace_arg = pass_fn("hivm-bind-workspace-arg", "func.func")
infer_workspace_size_func = pass_fn("hivm-insert-infer-workspace-size-func", "func.func")
graph_sync_solver = pass_fn("hivm-graph-sync-solver", "func.func")
lower_memref_ext = pass_fn("lower-memref-ext")
