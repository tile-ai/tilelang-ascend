# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""TileLangIR dialect transformation passes."""

from tilelang.tladapter.utils import pass_fn

cv_split = pass_fn("tilelangir-cv-split")
vectorize = pass_fn("tilelangir-vectorize")
