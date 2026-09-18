"""T.tile.exp test suite.

Registered against the shared unary-op framework in testing/python/base/.
Developer mode via DEFAULT_PASS_CONFIGS (auto sync + memory planning).
"""

import torch

import tilelang.language as T

from base import UnaryOpSpec, register_unary_op_tests

exp_spec = UnaryOpSpec(
    name="exp",
    tile_op=T.tile.exp,
    golden=torch.exp,
    supported_dtypes=["float16", "float32"],
    low_priority_dtypes=["float32"],
)

register_unary_op_tests(exp_spec)
