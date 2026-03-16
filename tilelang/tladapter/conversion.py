# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
TileLangIR conversion: lowering passes (dialect conversion, etc.).

Each pass accepts str and Module. Use pass(**opts) for pipeline with options.
Add lowering passes: name = pass_fn("pass-name").
"""

from tilelang.tladapter.utils import pass_fn

# HIVM pipelines (migrated from bishengir-compile PassPipeline.cpp)
convert_to_hivm_pipeline = pass_fn("convert-to-hivm-pipeline")
optimize_hivm_pipeline = pass_fn("optimize-hivm-pipeline")
