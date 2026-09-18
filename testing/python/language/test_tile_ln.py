"""T.tile.ln test suite.

Registered against the shared unary-op framework in testing/python/base/.
Developer mode via DEFAULT_PASS_CONFIGS (auto sync + memory planning).
"""

import torch

import tilelang.language as T

from base import UnaryOpSpec, register_unary_op_tests

ln_spec = UnaryOpSpec(
    name="ln",
    tile_op=T.tile.ln,
    golden=torch.log,
    supported_dtypes=["float16", "float32"],
    low_priority_dtypes=["float32"],
)

register_unary_op_tests(ln_spec)
