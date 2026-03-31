# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""TileLangIR dialect transformation passes."""

from tilelang.tladapter.utils import pass_fn

insert_workspace = pass_fn("tilelangir-insert-workspace", "func.func")
mark_multibuffer = pass_fn("tilelangir-mark-multibuffer", "func.func")
cv_split = pass_fn("tilelangir-cv-split", "func.func")
infer_mem_scope = pass_fn("tilelangir-infer-mem-scope", "func.func")
plan_workspace_memory = pass_fn("tilelangir-plan-workspace-memory", "func.func")
vectorize = pass_fn("tilelangir-vectorize")
enable_multi_buffer = pass_fn("tilelangir-enable-multi-buffer")
insert_cv_sync = pass_fn("tilelangir-insert-cv-sync")
split_mix_kernel = pass_fn("tilelangir-split-mix-kernel")
specialize_cube = pass_fn("tilelangir-specialize-cube", "func.func")
