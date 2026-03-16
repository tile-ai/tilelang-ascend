# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
TileLangIR conversion: lowering / dialect-conversion passes.

Each pass accepts str and Module. Use pass(**opts) for pipeline with options.
"""

from tilelang.tladapter.utils import pass_fn

# ── pipelines (kept for reference / backward compat) ───────────────────
convert_to_hivm_pipeline = pass_fn("convert-to-hivm-pipeline")
optimize_hivm_pipeline = pass_fn("optimize-hivm-pipeline")

# ── individual conversion passes (decomposed from convert-to-hivm-pipeline)
hfusion_to_hivm = pass_fn("convert-hfusion-to-hivm")
triton_global_kernel_args_to_hivm_op = pass_fn("triton-global-kernel-args-to-hivm-op")
tensor_to_hivm = pass_fn("convert-tensor-to-hivm")
to_hivm_op = pass_fn("convert-to-hivm-op")
