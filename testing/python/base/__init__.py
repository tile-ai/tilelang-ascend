"""
Base test infrastructure for TileLang-Ascend API tests.

Usage:
    from base import BinaryOpSpec, register_binary_op_tests
    from base import UnaryOpSpec, register_unary_op_tests
    from base import TOLERANCE, DTYPE_MAP, assert_close_npu, make_test_data
"""

from base.common import (
    TOLERANCE as TOLERANCE,
    DTYPE_MAP as DTYPE_MAP,
    DEFAULT_PASS_CONFIGS as DEFAULT_PASS_CONFIGS,
    assert_close_npu as assert_close_npu,
    make_test_data as make_test_data,
    skip_if_missing as skip_if_missing,
)

from base.binary_op import (
    BinaryOpSpec as BinaryOpSpec,
    BinaryOpTestClasses as BinaryOpTestClasses,
    register_binary_op_tests as register_binary_op_tests,
    make_binary_kernel as make_binary_kernel,
    make_1d_kernel as make_1d_kernel,
    make_scalar_kernel as make_scalar_kernel,
    make_buffload_kernel as make_buffload_kernel,
    make_row_slice_kernel as make_row_slice_kernel,
    make_inplace_kernel as make_inplace_kernel,
    make_src0_mismatch_kernel as make_src0_mismatch_kernel,
    make_region_mismatch_kernel as make_region_mismatch_kernel,
    make_buffer_mismatch_kernel as make_buffer_mismatch_kernel,
    run_binary_op as run_binary_op,
)

from base.unary_op import (
    UnaryOpSpec as UnaryOpSpec,
    UnaryOpTestClasses as UnaryOpTestClasses,
    register_unary_op_tests as register_unary_op_tests,
    make_unary_kernel as make_unary_kernel,
    run_unary_op as run_unary_op,
)
