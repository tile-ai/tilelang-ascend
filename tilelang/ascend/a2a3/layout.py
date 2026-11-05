# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

# Source: TA 0deb8b93 tilelang/intrinsics/ascend_layout.py and the Layout
# constructor metadata from tilelang/layout/layout.py.

from tvm import arith, DataType
import tvm_ffi
from tilelang.layout import Layout as _SharedLayout
from enum import Enum
from ._native import register_optional_object, require_native


@register_optional_object("tl.A2A3Layout", "tl.ascend.a2a3.WrapLayout")
class Layout(_SharedLayout):
    """TA coordinate map with its explicit template tag (default RowMajor)."""

    def __init__(self, shape, forward_fn, layout_tag=0):
        constructor = require_native("tl.ascend.a2a3.WrapLayout", "Layout")
        base = _SharedLayout(shape, forward_fn)
        self.__init_handle_by_constructor__(constructor, base, int(layout_tag))


def wrap_layout(layout, layout_tag):
    """Explicitly bridge a shared Layout with a known TA template tag."""
    tag = layout_tag.value if isinstance(layout_tag, AscendLayout) else int(layout_tag)
    return require_native("tl.ascend.a2a3.WrapLayout", "wrap_layout")(layout, tag)


def layout_str(layout):
    return require_native("tl.ascend.a2a3.LayoutStr", "layout_str")(layout)


def physical_allocation_shape(layout, dtype):
    return require_native("tl.ascend.a2a3.PhysicalAllocationShape", "physical_allocation_shape")(layout, DataType(dtype))


class AscendLayout(Enum):
    kRowMajor = 0
    kColMajor = 1
    kzN = 2
    kzZ = 3
    knZ = 4


BYTE_PER_C0 = 32
C0_NUM_PER_FRACTAL = 16
BYTE_PER_FRACTAL = BYTE_PER_C0 * C0_NUM_PER_FRACTAL


def ceil_div(a, b):
    """Ceiling division of a by b."""
    return (a + b - 1) // b


def round_up(a, b):
    """Round up a to the nearest multiple of b."""
    return ceil_div(a, b) * b


@tvm_ffi.register_global_func("tl.ascend.a2a3.make_zn_layout")
def make_zn_layout(buf):
    ana = arith.Analyzer()
    dtype = buf.dtype
    shape = buf.shape

    ELE_NUM_PER_C0 = BYTE_PER_C0 // DataType(dtype).bits * 8
    ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL // DataType(dtype).bits * 8

    def transform_func(*args):
        i, j = args[-2:]
        """
        zn:
            Layout:
            MakeShape(MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv<Catlass::C0_NUM_PER_FRACTAL>(rows)),
                          MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv<ELE_NUM_PER_C0>(cols))),
            MakeStride(MakeStride(Int<ELE_NUM_PER_C0>{}, Int<ELE_NUM_PER_FRACTAL>{}),
                           MakeStride(Int<1>{}, (int64_t)RoundUp<Catlass::C0_NUM_PER_FRACTAL>(rows) * ELE_NUM_PER_C0)));

            Coord compute:
                const uint32_t rowsInFractal = get<0, 0>(shape);
                const uint32_t colsInFractal = get<1, 0>(shape);
                const int64_t strideRowsByFractal = get<0, 1>(stride);
                const int64_t strideColsByFractal = get<1, 1>(stride);
                return row / rowsInFractal * strideRowsByFractal + col / colsInFractal * strideColsByFractal
                    + (row % rowsInFractal) * get<0, 0>(stride) + (col % colsInFractal) * get<1, 0>(stride);
        """
        new_shape = [[C0_NUM_PER_FRACTAL, ceil_div(shape[-2], C0_NUM_PER_FRACTAL)], [ELE_NUM_PER_C0, ceil_div(shape[-1], ELE_NUM_PER_C0)]]
        new_stride = [[ELE_NUM_PER_C0, ELE_NUM_PER_FRACTAL], [1, round_up(shape[-2], C0_NUM_PER_FRACTAL) * ELE_NUM_PER_C0]]  # srcM
        rowInFractal = new_shape[0][0]
        colInFractal = new_shape[1][0]
        strideRowsByFractal = new_stride[0][1]
        strideColsByFractal = new_stride[1][1]

        return [
            *args[:-2],
            ana.simplify(
                i // rowInFractal * strideRowsByFractal
                + j // colInFractal * strideColsByFractal
                + (i % rowInFractal) * new_stride[0][0]
                + (j % colInFractal) * new_stride[1][0]
            ),
        ]

    return Layout(shape, transform_func, layout_tag=AscendLayout.kzN.value)


def make_col_major_layout(buf):
    def transform_func(*args):
        i, j = args[-2:]

        return [*args[:-2], j, i]

    return Layout(buf.shape, transform_func, layout_tag=AscendLayout.kColMajor.value)


def make_nz_layout(buf):
    """
    return MakeLayout(
        MakeShape(MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv<ELE_NUM_PER_C0>(rows)),
                  MakeShape(Int<Catlass::C0_NUM_PER_FRCATLASSAL>{}, CeilDiv<Catlass::C0_NUM_PER_FRCATLASSAL>(cols))),
        MakeStride(MakeStride(Int<1>{}, (int64_t)RoundUp<Catlass::C0_NUM_PER_FRCATLASSAL>(cols) * ELE_NUM_PER_C0),
                   MakeStride(Int<ELE_NUM_PER_C0>{}, Int<ELE_NUM_PER_FRCATLASSAL>{})));

            Coord compute:
            const uint32_t rowsInFractal = get<0, 0>(shape);
            const uint32_t colsInFractal = get<1, 0>(shape);
            const int64_t strideRowsByFractal = get<0, 1>(stride);
            const int64_t strideColsByFractal = get<1, 1>(stride);
            return row / rowsInFractal * strideRowsByFractal + col / colsInFractal * strideColsByFractal
                + (row % rowsInFractal) * get<0, 0>(stride) + (col % colsInFractal) * get<1, 0>(stride);
    """

    ana = arith.Analyzer()
    dtype = buf.dtype
    shape = buf.shape

    ELE_NUM_PER_C0 = BYTE_PER_C0 // DataType(dtype).bits * 8
    ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL // DataType(dtype).bits * 8
    C0_NUM_PER_FRACTAL = BYTE_PER_FRACTAL // BYTE_PER_C0

    def transform_func(*args):
        i, j = args[-2:]

        new_shape = [[ELE_NUM_PER_C0, ceil_div(shape[-2], ELE_NUM_PER_C0)], [C0_NUM_PER_FRACTAL, ceil_div(shape[-1], C0_NUM_PER_FRACTAL)]]
        new_stride = [[1, round_up(shape[-1], C0_NUM_PER_FRACTAL) * ELE_NUM_PER_C0], [ELE_NUM_PER_C0, ELE_NUM_PER_FRACTAL]]

        rowInFractal = new_shape[0][0]
        colInFractal = new_shape[1][0]
        strideRowsByFractal = new_stride[0][1]
        strideColsByFractal = new_stride[1][1]

        return [
            *args[:-2],
            ana.simplify(
                i // rowInFractal * strideRowsByFractal
                + j // colInFractal * strideColsByFractal
                + (i % rowInFractal) * new_stride[0][0]
                + (j % colInFractal) * new_stride[1][0]
            ),
        ]

    return Layout(shape, transform_func, layout_tag=AscendLayout.knZ.value)
