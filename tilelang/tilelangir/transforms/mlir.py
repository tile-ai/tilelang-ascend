# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""MLIR builtin/built-in transformation passes."""

from tilelang.tilelangir.utils import pass_fn

canonicalize = pass_fn("canonicalize")
cse = pass_fn("cse")
sccp = pass_fn("sccp")
