// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file src/transform/common/ascend_tail_mask.h
 * \brief Shared data model for the AscendC vector tail-block scheme.
 *
 * Background: on the AscendC backend a GM->UB copy may load a *tail* block that
 * is smaller than the physical UB tile.  After removing the front-end
 * `pad_value`, the unused UB region is left untouched (garbage), so any op that
 * mixes the gap into a valid lane (cross-lane reductions in particular) must be
 * told the real valid extent.  AscendTailMaskPropagation tracks, per UB buffer,
 * the logical valid rectangle and the physical tile pitch, and rewrites the
 * affected vector ops to tail-aware variants.
 *
 * This header only carries the data model; it is intentionally light so it can
 * be shared between the pass and (future) consumers without pulling in heavy
 * dependencies.
 */
#ifndef TVM_TL_TRANSFORM_COMMON_ASCEND_TAIL_MASK_H_
#define TVM_TL_TRANSFORM_COMMON_ASCEND_TAIL_MASK_H_

#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace tl {

using namespace tir;

/*! \brief What kind of valid-region a UB buffer currently carries. */
enum class TailMaskKind {
  /*! \brief Whole physical tile is valid (statically full -> never rewritten).
   */
  kFull = 0,
  /*! \brief A regular 2D rectangle [valid_row, valid_col] inside the tile. */
  kTail = 1,
  /*! \brief Packed comparison mask (Batch 2: compare/select). */
  kPackedCmp = 2,
};

/*!
 * \brief Logical valid region plus physical pitch of a UB tile.
 *
 *  - `valid_row` / `valid_col` describe the logically-valid rectangle. They may
 *    be dynamic (a `Select(...)` over the block index), which is exactly why
 * the rewrite produces a *runtime* tail helper instead of a compile-time shape.
 *  - `physical_row` / `physical_col` are the allocated UB tile dims;
 * `physical_col` is the row pitch used to derive repeat strides.
 *  - `storage_col` is reserved for packed comparison masks (Batch 2).
 */
struct TailMaskInfo {
  TailMaskKind kind = TailMaskKind::kFull;
  PrimExpr valid_row;
  PrimExpr valid_col;
  PrimExpr physical_row;
  PrimExpr physical_col;
  PrimExpr storage_col;

  bool is_tail() const { return kind == TailMaskKind::kTail; }
};

/*! \brief A fully-valid mask for a tile of the given physical dims. */
inline TailMaskInfo MakeFullMask(PrimExpr physical_row, PrimExpr physical_col) {
  TailMaskInfo m;
  m.kind = TailMaskKind::kFull;
  m.valid_row = physical_row;
  m.valid_col = physical_col;
  m.physical_row = physical_row;
  m.physical_col = physical_col;
  m.storage_col = physical_col;
  return m;
}

/*!
 * \brief Decide whether a (valid_row,valid_col) vs (physical_row,physical_col)
 *        pair is statically a full tile. When both extents are constants equal
 *        to the physical dims the buffer is full and must NOT be rewritten, so
 *        non-tail kernels keep generating identical code.
 */
inline bool IsStaticallyFull(const PrimExpr &valid_row,
                             const PrimExpr &valid_col,
                             const PrimExpr &physical_row,
                             const PrimExpr &physical_col,
                             arith::Analyzer *analyzer) {
  return analyzer->CanProveEqual(valid_row, physical_row) &&
         analyzer->CanProveEqual(valid_col, physical_col);
}

/*!
 * \brief Build a tail mask from a copy's valid/physical extents. If statically
 *        full, returns a kFull mask so downstream ops are left untouched.
 */
inline TailMaskInfo MakeCopyMask(PrimExpr valid_row, PrimExpr valid_col,
                                 PrimExpr physical_row, PrimExpr physical_col,
                                 arith::Analyzer *analyzer) {
  if (IsStaticallyFull(valid_row, valid_col, physical_row, physical_col,
                       analyzer)) {
    return MakeFullMask(physical_row, physical_col);
  }
  TailMaskInfo m;
  m.kind = TailMaskKind::kTail;
  m.valid_row = valid_row;
  m.valid_col = valid_col;
  m.physical_row = physical_row;
  m.physical_col = physical_col;
  m.storage_col = physical_col;
  return m;
}

/*!
 * \brief Intersection of two operand masks (used for binary ops). The result is
 *        tail iff either input is tail; valid extents are the element-wise min,
 *        and the physical pitch is taken from the first tail operand.
 *
 *  NB: the repeat-plan assumes both operands (and the dst) share the same
 *  physical pitch. When they differ the tail helper falls back to a per-row
 *  loop, so a mismatch is a perf concern, not a correctness one.
 */
inline TailMaskInfo IntersectMasks(const TailMaskInfo &a, const TailMaskInfo &b,
                                   arith::Analyzer *analyzer) {
  if (!a.is_tail() && !b.is_tail()) {
    return a.kind == TailMaskKind::kFull ? a : b;
  }
  const TailMaskInfo &base = a.is_tail() ? a : b;
  TailMaskInfo m = base;
  m.kind = TailMaskKind::kTail;
  if (a.valid_row.defined() && b.valid_row.defined()) {
    m.valid_row = analyzer->CanProveEqual(a.valid_row, b.valid_row)
                      ? a.valid_row
                      : Min(a.valid_row, b.valid_row);
  }
  if (a.valid_col.defined() && b.valid_col.defined()) {
    m.valid_col = analyzer->CanProveEqual(a.valid_col, b.valid_col)
                      ? a.valid_col
                      : Min(a.valid_col, b.valid_col);
  }
  return m;
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_ASCEND_TAIL_MASK_H_
