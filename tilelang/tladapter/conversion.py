# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
TileLangIR conversion: lowering passes (dialect conversion, etc.).

Each pass accepts str and Module. Use pass(**opts) for pipeline with options.
Add lowering passes: name = pass_fn("pass-name").
"""

from tilelang.tladapter.utils import pass_fn

# Conversion passes - add lowering passes here
