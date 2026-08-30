"""T.tile.min test suite."""

import torch

import tilelang.language as T

from base import BinaryOpSpec, register_binary_op_tests
from base.binary_op import (
    make_1d_kernel,
    make_buffload_kernel,
    make_buffer_mismatch_kernel,
    make_multidim_buffload_kernel,
    make_region_mismatch_kernel,
    make_row_slice_kernel,
    make_src0_mismatch_kernel,
)

tile_op = T.tile.min

min_spec = BinaryOpSpec(
    name="min",
    tile_op=tile_op,
    golden=torch.minimum,
    supported_dtypes=["float16", "float32", "int16", "int32"],
    low_priority_dtypes=["float16", "int16", "int32"],
    kernel_1d=make_1d_kernel(tile_op),
    kernel_buffload=make_buffload_kernel(tile_op),
    kernel_row_slice=make_row_slice_kernel(tile_op),
    kernel_multidim_buffload=make_multidim_buffload_kernel(tile_op),
    mismatch_kernels=(
        lambda: make_src0_mismatch_kernel(tile_op),
        lambda: make_region_mismatch_kernel(tile_op),
        lambda: make_buffer_mismatch_kernel(tile_op),
    ),
)

register_binary_op_tests(min_spec)
