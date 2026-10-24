// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/op/ascend.cc
 *
 * Define ascend-related operators.
 */

#include "ascend.h"

#include <algorithm>
#include <atomic>
#include <sstream>
#include <tuple>
#include <vector>

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "builtin.h"

namespace tvm {
namespace tl {

using namespace tir;

#define TIR_DEFINE_TL_BUILTIN(OpName)                                          \
  const Op &OpName() {                                                         \
    static const Op &op = Op::Get("tl." #OpName);                              \
    return op;                                                                 \
  }                                                                            \
  TVM_REGISTER_OP("tl." #OpName)                                               \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", #OpName)

AscendCopy::AscendCopy(Array<PrimExpr> args, BufferMap vmap) : args_(args) {
  Array<Range> rgs[2];
  Buffer bf[2];
  Array<PrimExpr> ets[2];
  transposeL1 = 0;
  for (int i = 0; i < 2; i++) {
    auto expr = args[i];
    auto call = expr.as<CallNode>();
    ICHECK(call);
    auto region = RegionOp(call->args, vmap);
    ets[i] = region.GetExtents();
    rgs[i] = region.GetRanges();
    bf[i] = region.GetBuffer();
  }
  if (args.size() >= 3) {
    enRelu = args[2].as<Bool>().value();
  }
  if (args.size() >= 4) {
    transposeL1 = args[3].as<Bool>().value();
  }
  if (args.size() >= 5) {
    padValue = args[4];
  } else {
    padValue = Integer(0);
  }
  // Handle tmp parameter: check if it's a valid region (not IntImm(0) marker)
  if (args.size() >= 6) {
    auto tmp_expr = args[5];
    // Check if tmp is a Call (valid region) or IntImm (none marker)
    if (auto *tmp_call = tmp_expr.as<CallNode>()) {
      auto tmp_region = RegionOp(tmp_call->args, vmap);
      this->tmp = tmp_region.GetBuffer();
      this->tmp_range = tmp_region.GetRanges();
      this->tmp_extents = tmp_region.GetExtents();
    }
    // If tmp_expr is IntImm(0), leave tmp as undefined (default)
  }
  // Optional trailing runtime arguments for the kernel-driven cube path. All
  // default to 0, which reproduces the previous behaviour exactly: unitFlag 0
  // is a standalone fixpipe, and realK/realN 0 mean "take the extent from the
  // destination L0 buffer".
  unitFlag = Integer(0);
  realK = Integer(0);
  realN = Integer(0);
  if (args.size() >= 7) {
    unitFlag = args[6];
  }
  if (args.size() >= 8) {
    realK = args[7];
  }
  if (args.size() >= 9) {
    realN = args[8];
  }
  std::tie(this->src, this->dst) = std::tie(bf[0], bf[1]);
  std::tie(this->src_range, this->dst_range) = std::tie(rgs[0], rgs[1]);
  std::tie(this->src_extents, this->dst_extents) = std::tie(ets[0], ets[1]);
}

AscendAtomicAdd::AscendAtomicAdd(Array<PrimExpr> args, BufferMap vmap)
    : args_(args) {
  ICHECK_EQ(args.size(), 2U) << "tl.ascend_atomic_add expects dst and src";

  auto dst_call = args[0].as<CallNode>();
  auto src_call = args[1].as<CallNode>();
  ICHECK(dst_call) << "tl.ascend_atomic_add dst must be a tl.region call";
  ICHECK(src_call) << "tl.ascend_atomic_add src must be a tl.region call";

  auto dst_region = RegionOp(dst_call->args, vmap);
  auto src_region = RegionOp(src_call->args, vmap);
  ICHECK(dst_region.GetAccessMask() & 2)
      << "tl.ascend_atomic_add dst region must be writable";
  ICHECK(src_region.GetAccessMask() & 1)
      << "tl.ascend_atomic_add src region must be readable";

  this->dst = dst_region.GetBuffer();
  this->src = src_region.GetBuffer();
  this->dst_range = dst_region.GetRanges();
  this->src_range = src_region.GetRanges();
  this->dst_extents = dst_region.GetExtents();
  this->src_extents = src_region.GetExtents();
}

// A GM<->UB DMA moves data as `rows` bursts of `cols` elements each. Two
// hardware constraints govern the UB side (verified empirically on 910B /
// CANN 9.0.0):
//   * every burst's payload must be at least 8-byte granular: a multi-burst
//     copy of 2-byte payloads (issue #929's strided one-element-per-row load
//     into a flat 1D UB buffer) scatters bursts to alignment boundaries past
//     the buffer and silently corrupts neighbouring UB allocations, while
//     8-byte payloads work (tail_block's packed-compare mask store relies on
//     it);
//   * the per-row pitch gap is described in 32-byte units (the helper passes
//     the UB-side inter-burst gap as (buf_inner - cols) * elem / 32), so a
//     gap that is not a 32-byte multiple truncates and packs rows at the
//     wrong pitch; the same happens whenever the template column count is
//     not the local tile's physical innermost dim (a column slice into a
//     wider tile).
// A copy is therefore NOT representable as a single DMA when any of these
// holds and more than one burst is involved. Single-burst copies (rows
// provably <= 1) keep the DMA: the pad path handles any tail width. Copies
// whose column count is not a compile-time constant also keep the DMA (the
// runtime width may be aligned; status quo).
static bool NeedsScalarGMCopyFallback(const PrimExpr &rows,
                                      const PrimExpr &cols, int elem_bytes,
                                      const PrimExpr &tmpl_n,
                                      const PrimExpr &buf_inner) {
  if (const auto *rows_imm = rows.as<IntImmNode>()) {
    if (rows_imm->value <= 1) {
      return false;
    }
  }
  const auto *cols_imm = cols.as<IntImmNode>();
  if (!cols_imm || cols_imm->value < 1) {
    return false;
  }
  int64_t payload = static_cast<int64_t>(cols_imm->value) * elem_bytes;
  if (payload % 8 != 0) {
    return true;
  }
  const auto *tmpl_imm = tmpl_n.as<IntImmNode>();
  const auto *inner_imm = buf_inner.as<IntImmNode>();
  if (!tmpl_imm || !inner_imm || tmpl_imm->value != inner_imm->value) {
    return true;
  }
  int64_t gap = (inner_imm->value - cols_imm->value) * elem_bytes;
  return gap % 32 != 0;
}

// Elementwise fallback for GM->UB loads that a single 2D DMA cannot express
// (see NeedsScalarGMCopyFallback). Mirrors the DMA semantics:
//   * when (valid_row != tmpl_m || valid_col != tmpl_n): prefill the local
//     tile's first tmpl_m * tmpl_n elements (from the region base, linearly in
//     buffer memory) with padValue -- the scalar equivalent of the helper's
//     Duplicate prefill;
//   * then copy the valid_row x valid_col rect element by element between
//     the region's declared row-major layouts: the GM side indexes its active
//     row/col dims, the local side is unraveled over its region extents.
//     This is exactly the layout the DMA cannot express.
// The loops stay scalar (GetValue/SetValue codegen on both backends): the
// triggering shapes have < 32 bytes of contiguous run, so there is nothing
// for a vector instruction to chew on anyway. Scalar GM *loads* are safe;
// only scalar GM stores are unreliable (see MakeStagedUBToGMCopy).
static Stmt
MakeScalarGMCopyLoop(const Buffer &gm_buf, const Array<Range> &gm_range,
                     const std::vector<int> &gm_active, const Buffer &local_buf,
                     const Array<Range> &local_range,
                     const Array<PrimExpr> &local_extents,
                     const PrimExpr &valid_row, const PrimExpr &valid_col,
                     const PrimExpr &tmpl_m, const PrimExpr &tmpl_n,
                     const PrimExpr &pad_value, arith::Analyzer *analyzer) {
  int gm_row_idx = gm_active[gm_active.size() - 2];
  int gm_col_idx = gm_active.back();
  int gm_ndim = static_cast<int>(gm_range.size());
  int loc_ndim = static_cast<int>(local_range.size());

  // Row-major strides over the LOCAL region extents (element linear position
  // within the region -> per-dim index).
  std::vector<PrimExpr> loc_strides(loc_ndim);
  PrimExpr acc = Integer(1);
  for (int d = loc_ndim - 1; d >= 0; --d) {
    loc_strides[d] = acc;
    acc = analyzer->Simplify(acc * local_extents[d]);
  }

  static std::atomic<int> counter{0};
  std::string suffix = std::to_string(counter.fetch_add(1));
  Var iv("copy_gm_row_" + suffix, DataType::Int(32));
  Var jv("copy_gm_col_" + suffix, DataType::Int(32));

  auto gm_indices = [&](const PrimExpr &i, const PrimExpr &j) {
    Array<PrimExpr> idx;
    for (int d = 0; d < gm_ndim; ++d) {
      PrimExpr e = gm_range[d]->min;
      if (d == gm_row_idx) {
        e = e + i;
      } else if (d == gm_col_idx) {
        e = e + j;
      }
      idx.push_back(analyzer->Simplify(e));
    }
    return idx;
  };
  auto local_indices = [&](const PrimExpr &linear) {
    Array<PrimExpr> idx;
    for (int d = 0; d < loc_ndim; ++d) {
      PrimExpr e = local_range[d]->min;
      if (!is_one(local_extents[d])) {
        e = e + truncmod(floordiv(linear, loc_strides[d]), local_extents[d]);
      }
      idx.push_back(analyzer->Simplify(e));
    }
    return idx;
  };

  Stmt copy_loop;
  {
    // Row pitch of the LOCAL region's declared layout: for a multi-dim tile
    // it is the region's inner extent -- NOT the runtime-clamped valid_col
    // (a column-tail clamp shrinks only how many elements each row
    // contributes, not where the tile's rows live). A 1-D flat region has no
    // row pitch: its rows pack back-to-back, so the advance is valid_col
    // (exactly the DMA's own stream advance: payload + truncated gap).
    PrimExpr local_pitch = loc_ndim >= 2
                               ? analyzer->Simplify(local_extents[loc_ndim - 1])
                               : valid_col;
    PrimExpr linear = iv * local_pitch + jv;
    PrimExpr value = BufferLoad(gm_buf, gm_indices(iv, jv));
    if (value->dtype != local_buf->dtype) {
      value = Cast(local_buf->dtype, value);
    }
    copy_loop = BufferStore(local_buf, value, local_indices(linear));
    copy_loop = For(jv, IntImm(DataType::Int(32), 0), valid_col,
                    ForKind::kSerial, copy_loop);
    copy_loop = For(iv, IntImm(DataType::Int(32), 0), valid_row,
                    ForKind::kSerial, copy_loop);
  }

  // Pad prefill, mirroring the helper's Duplicate: fill the local tile's
  // first tmpl_m * tmpl_n elements linearly from the region base with
  // padValue whenever the valid rect does not cover the whole tile.
  PrimExpr need_pad = analyzer->Simplify(valid_row != tmpl_m);
  bool pad_always = false, pad_never = false;
  if (const auto *b = need_pad.as<IntImmNode>()) {
    if (need_pad.dtype().is_bool()) {
      pad_always = b->value != 0;
      pad_never = b->value == 0;
    }
  }
  if (!pad_never) {
    const auto *col_imm = valid_col.as<IntImmNode>();
    const auto *tmpl_n_imm = tmpl_n.as<IntImmNode>();
    if (col_imm && tmpl_n_imm && col_imm->value != tmpl_n_imm->value) {
      pad_always = true;
    }
  }

  if (pad_never) {
    return copy_loop;
  }

  // Linear fill from the region base, unraveling over the local buffer's
  // shape (Duplicate fills the tensor view's memory linearly).
  std::vector<PrimExpr> buf_strides(local_buf->shape.size());
  acc = Integer(1);
  for (int d = static_cast<int>(local_buf->shape.size()) - 1; d >= 0; --d) {
    buf_strides[d] = acc;
    acc = analyzer->Simplify(acc * local_buf->shape[d]);
  }
  Array<PrimExpr> base_idx;
  for (int d = 0; d < loc_ndim; ++d) {
    base_idx.push_back(local_range[d]->min);
  }
  PrimExpr base = local_buf.OffsetOf(base_idx).back();

  Var kv("copy_gm_pad_" + suffix, DataType::Int(32));
  Array<PrimExpr> pad_idx;
  for (size_t d = 0; d < local_buf->shape.size(); ++d) {
    PrimExpr e;
    if (d + 1 == local_buf->shape.size()) {
      e = truncmod(base + kv, local_buf->shape[d]);
    } else {
      e = truncmod(floordiv(base + kv, buf_strides[d]), local_buf->shape[d]);
    }
    pad_idx.push_back(analyzer->Simplify(e));
  }
  Stmt pad_loop = BufferStore(local_buf, pad_value, pad_idx);
  pad_loop =
      For(kv, IntImm(DataType::Int(32), 0), analyzer->Simplify(tmpl_m * tmpl_n),
          ForKind::kSerial, pad_loop);

  if (pad_always) {
    return SeqStmt::Flatten(pad_loop, copy_loop);
  }
  return SeqStmt::Flatten(IfThenElse(need_pad, pad_loop), copy_loop);
}

// Store-direction fallback for UB->GM copies a single 2D DMA cannot express
// (sub-32-byte burst payloads / inexpressible UB-side pitch, issue #929).
// Scalar GM stores are NOT a viable fallback on this uArch: the S-pipe store
// path goes through the write-back cache, which drops most partial-line
// (2-byte) stores before they ever reach GM (the #1304 failure mode; verified
// empirically -- DataCacheCleanAndInvalid does not recover them). Instead,
// stage the strided elements into a 32-byte-row-aligned UB scratch tile
// (one row of `pad_n` elements per copied row, element j at column j) via
// scalar UB->UB moves, then emit ONE single-burst MTE3 DMA per row: a
// single burst has no UB-side pitch requirement, and its source (the staging
// row start) is 32-byte aligned by construction.
static Stmt MakeStagedUBToGMCopy(
    const LowerArgs &T, const Buffer &gm_buf, const Array<Range> &gm_range,
    const std::vector<int> &gm_active, const Buffer &local_buf,
    const Array<Range> &local_range, const Array<PrimExpr> &local_extents,
    const PrimExpr &valid_row, const PrimExpr &valid_col,
    const std::string &dtype_str, arith::Analyzer *analyzer) {
  int gm_row_idx = gm_active[gm_active.size() - 2];
  int gm_col_idx = gm_active.back();
  int loc_ndim = static_cast<int>(local_range.size());

  // GM-side element strides of the row / col dims (row-major buffer layout).
  PrimExpr row_stride = Integer(1);
  for (size_t k = gm_row_idx + 1; k < gm_buf->shape.size(); ++k) {
    row_stride = row_stride * gm_buf->shape[k];
  }
  PrimExpr col_stride = Integer(1);
  for (size_t k = gm_col_idx + 1; k < gm_buf->shape.size(); ++k) {
    col_stride = col_stride * gm_buf->shape[k];
  }
  bool cols_contiguous = is_one(analyzer->Simplify(col_stride));

  // Region upper bounds (the DMA runtime bounds valid_row/valid_col clamp
  // within them). The staging tile is sized from these.
  const auto *row_extent_imm =
      analyzer->Simplify(gm_range[gm_row_idx]->extent).as<IntImmNode>();
  const auto *col_extent_imm =
      analyzer->Simplify(gm_range[gm_col_idx]->extent).as<IntImmNode>();
  ICHECK(row_extent_imm && row_extent_imm->value >= 1)
      << "copy_ub_to_gm fallback: the strided store's row count must be a "
         "compile-time constant, but got "
      << gm_range[gm_row_idx]->extent;
  ICHECK(col_extent_imm && col_extent_imm->value >= 1)
      << "copy_ub_to_gm fallback: the strided store's column count must be a "
         "compile-time constant, but got "
      << gm_range[gm_col_idx]->extent;
  int64_t row_extent = row_extent_imm->value;
  int64_t col_extent = col_extent_imm->value;

  // Elements per staging row: at least the column count, rounded up so every
  // row start is 32-byte aligned (all supported element sizes divide 32).
  int elem_bytes = gm_buf->dtype.bytes();
  int64_t align_elems = 32 / elem_bytes;
  if (align_elems < 1) {
    align_elems = 1;
  }
  int64_t pad_n = std::max(col_extent, align_elems);
  pad_n = (pad_n + align_elems - 1) / align_elems * align_elems;

  // Staging scratch: one flat UB workspace, row i at element offset i*pad_n.
  Buffer stage = T.AddWorkspace(static_cast<int>(row_extent * pad_n),
                                local_buf->dtype, "shared.ub");

  // Local-side region unravel (row-major over the region extents).
  std::vector<PrimExpr> loc_strides(loc_ndim);
  PrimExpr acc = Integer(1);
  for (int d = loc_ndim - 1; d >= 0; --d) {
    loc_strides[d] = acc;
    acc = analyzer->Simplify(acc * local_extents[d]);
  }
  auto local_indices = [&](const PrimExpr &linear) {
    Array<PrimExpr> idx;
    for (int d = 0; d < loc_ndim; ++d) {
      PrimExpr e = local_range[d]->min;
      if (!is_one(local_extents[d])) {
        e = e + truncmod(floordiv(linear, loc_strides[d]), local_extents[d]);
      }
      idx.push_back(analyzer->Simplify(e));
    }
    return idx;
  };

  // GM base flat offset of the region.
  Array<PrimExpr> gm_base_idx;
  for (size_t d = 0; d < gm_range.size(); ++d) {
    gm_base_idx.push_back(gm_range[d]->min);
  }
  PrimExpr gm_base = gm_buf.OffsetOf(gm_base_idx).back();

  static std::atomic<int> counter{0};
  std::string suffix = std::to_string(counter.fetch_add(1));
  Var iv("copy_stage_row_" + suffix, DataType::Int(32));
  Var jv("copy_stage_col_" + suffix, DataType::Int(32));

  // 1) Scalar UB->UB staging fill: element (i, j) lands at stage[i*pad_n + j].
  //    The source read walks the local region's declared layout: a multi-dim
  //    tile's row pitch is the region's inner extent -- not the runtime-
  //    clamped valid_col (a column-tail clamp shrinks only how many elements
  //    each row contributes, not where the tile's rows live) -- while a 1-D
  //    flat region packs its rows back-to-back at valid_col (the DMA's own
  //    stream advance).
  Stmt fill;
  {
    PrimExpr local_pitch = loc_ndim >= 2
                               ? analyzer->Simplify(local_extents[loc_ndim - 1])
                               : valid_col;
    PrimExpr linear = iv * local_pitch + jv;
    fill = BufferStore(stage, BufferLoad(local_buf, local_indices(linear)),
                       {iv * Integer(static_cast<int>(pad_n)) + jv});
    fill = For(jv, IntImm(DataType::Int(32), 0), valid_col, ForKind::kSerial,
               fill);
    fill = For(iv, IntImm(DataType::Int(32), 0), valid_row, ForKind::kSerial,
               fill);
  }

  // 2) Per-row (or per-element when the GM columns are not contiguous)
  // single-burst MTE3 calls.
  auto make_dma_call = [&](const PrimExpr &gm_off, const PrimExpr &stage_off,
                           const PrimExpr &ncol) {
    Array<PrimExpr> args;
    args.push_back(StringImm("tl::ascend::copy_ub_to_gm<" + dtype_str + ", " +
                             std::to_string(pad_n) + ">"));
    args.push_back(stage.access_ptr(1, stage->dtype, 1, stage_off,
                                    Integer(static_cast<int>(pad_n))));
    args.push_back(gm_buf.access_ptr(2, gm_buf->dtype, 1, gm_off, ncol));
    args.push_back(row_stride); // realDstN: unused by a single burst
    args.push_back(Integer(1)); // maskShapeM: one row per call
    args.push_back(ncol);       // maskShapeN
    return Evaluate(Call(DataType::Handle(), builtin::call_extern(), args));
  };

  Stmt dma;
  if (cols_contiguous) {
    // One burst of valid_col elements per row, straight from the staging row.
    // gm_base already carries every fixed dim's min (the col min included).
    PrimExpr gm_off = gm_base + iv * row_stride;
    dma =
        make_dma_call(gm_off, iv * Integer(static_cast<int>(pad_n)), valid_col);
    dma =
        For(iv, IntImm(DataType::Int(32), 0), valid_row, ForKind::kSerial, dma);
  } else {
    // Columns are strided in GM as well: fall back to one element per DMA.
    PrimExpr gm_off = gm_base + iv * row_stride + jv * col_stride;
    dma = make_dma_call(
        gm_off, (iv * valid_col + jv) * Integer(static_cast<int>(pad_n)),
        Integer(1));
    dma =
        For(jv, IntImm(DataType::Int(32), 0), valid_col, ForKind::kSerial, dma);
    dma =
        For(iv, IntImm(DataType::Int(32), 0), valid_row, ForKind::kSerial, dma);
  }

  return SeqStmt::Flatten(fill, dma);
}

Stmt AscendCopy::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  auto get_dtype = [](const Buffer &buf) -> std::string {
    auto dtype = buf->dtype;
    if (dtype.is_float16()) {
      return "half";
    } else if (dtype.is_float() && dtype.bits() == 32) {
      return "float";
    } else if (dtype.is_int() && dtype.bits() == 4) {
      return "AscendC::int4b_t";
    } else if (dtype.is_int() && dtype.bits() == 8) {
      return "int8_t";
    } else if (dtype.is_int() && dtype.bits() == 16) {
      return "int16_t";
    } else if (dtype.is_int() && dtype.bits() == 32) {
      return "int";
    } else if (dtype.is_int() && dtype.bits() == 64) {
      return "int64_t";
    } else if (dtype.is_uint() && dtype.bits() == 8) {
      return "uint8_t";
    } else if (dtype.is_uint() && dtype.bits() == 16) {
      return "uint16_t";
    } else if (dtype.is_uint() && dtype.bits() == 32) {
      return "uint32_t";
    } else if (dtype.is_uint() && dtype.bits() == 64) {
      return "uint64_t";
    } else if (dtype.is_bfloat16()) {
      return "bfloat16_t";
    } else if (dtype.is_float8_e4m3fn()) {
      return "float8_e4m3_t";
    } else if (dtype.is_float8_e5m2()) {
      return "float8_e5m2_t";
    }
    LOG(FATAL) << "Unsupported data type: " << dtype;
    return "";
  };

  auto compute_strideN = [](const Buffer &buf,
                            const Array<PrimExpr> &extents) -> PrimExpr {
    // The GM copy is emitted as a 2D DMA: blockCount rows (the second-to-last
    // active dim, i.e. the highest dim below the last whose extent != 1) by
    // blockLen columns (the last dim), with one uniform stride between
    // consecutive rows. That stride is the product of every buffer dim below
    // the row dim, so extent-1 dims sitting under the row dim must be folded
    // into it (e.g. Query[cid, m:m+BM, n2, g*D:(g+1)*D] folds the scalar n2
    // dim into the stride). But when every leading extent is 1 -- a row slice
    // like C2[row, 0:N] -- the row dim is the second-to-last dim itself
    // (blockCount = 1) and the stride is just the last buffer dim: folding the
    // leading dims in would return the whole buffer size as the stride and
    // corrupt the DMA (issue #1263).
    PrimExpr strideN = buf->shape[buf->shape.size() - 1];
    PrimExpr folded = Integer(1);
    bool has_row_dim = false;
    for (int i = static_cast<int>(extents.size()) - 2; i >= 0; --i) {
      auto *extent = extents[i].as<IntImmNode>();
      if (!extent || extent->value != 1) {
        has_row_dim = true;
        break;
      }
      folded = folded * buf->shape[i];
    }
    if (has_row_dim) {
      strideN = strideN * folded;
    }
    return strideN;
  };

  auto compute_blocklen = [](const Buffer &buf,
                             const Array<PrimExpr> &extents) -> PrimExpr {
    PrimExpr res = buf->shape[buf->shape.size() - 2];
    auto ext_size = extents.size();
    if (ext_size > 1 && extents[ext_size - 2]->IsInstance<IntImmNode>() &&
        res->IsInstance<IntImmNode>()) {
      auto extent =
          static_cast<int>(extents[ext_size - 2].as<IntImmNode>()->value);
      auto shape = static_cast<int>(res.as<IntImmNode>()->value);
      res = shape < extent ? res : extents[ext_size - 2];
    }

    return res;
  };

  auto build_indices = [](const Array<Range> &range) -> Array<PrimExpr> {
    Array<PrimExpr> indices;
    for (size_t i = 0; i < range.size(); i++) {
      indices.push_back(range[i]->min);
    }
    return indices;
  };

  struct CopyConfig {
    bool needs_strideN = false;
    bool l0c2gm = false;
    bool gm2l1 = false;
    bool l12l0 = false;
    bool print_gm_layout = false;
    bool print_src_layout = false;
    bool print_dst_layout = false;
    bool print_ub = false;
    bool l0_dst_split = false;
    bool virtual_channel = false;
    bool gm2ub = false;
    bool ub2gm = false;
    bool ub2ub = false;
  } config;

  std::stringstream ss;
  ss << "tl::ascend::";
  PrimExpr strideN;
  // Local-tile template dims of the GM<->UB copy call (dstN/dstM for a load,
  // srcN/srcM for a store), recorded for the sub-32B fallback check below.
  PrimExpr local_tmpl_n;
  PrimExpr local_tmpl_m = Integer(1);

  if (src.scope() == "global" && dst.scope() == "shared.l1") {
    ss << "copy_gm_to_l1";
    config.gm2l1 = true;
  } else if (src.scope() == "shared.l1" && dst.scope() == "wmma.matrix_a") {
    ss << "copy_l1_to_l0a";
    // config.print_src_layout = true;
    config.l12l0 = true;
    config.l0_dst_split = true;
  } else if (src.scope() == "shared.l1" && dst.scope() == "wmma.matrix_b") {
    ss << "copy_l1_to_l0b";
    // config.print_src_layout = true;
    config.l12l0 = true;
    config.l0_dst_split = true;
  } else if (src.scope() == "wmma.accumulator" && dst.scope() == "global") {
    ss << "copy_l0c_to_gm";
    config.l0c2gm = true;
    config.print_gm_layout = true;
  } else if (src.scope() == "shared.ub" || dst.scope() == "shared.ub") {
    config.print_ub = true;

    if (src.scope() == "global") {
      config.gm2ub = true;
      strideN = compute_strideN(src, src_extents);
      config.needs_strideN = true;

      ss << "copy_gm_to_ub<";
      ss << get_dtype(src) << ", ";
      // The column dim is a COMPILE-TIME template arg. A runtime inner-extent
      // slice (dynamic width, e.g. a softmax's actual window tw_a < buffer
      // width) would put a non-const expr in the template -> invalid C++ ("use
      // of undeclared identifier"). Use the buffer's compile-time shape for the
      // template; the runtime width is already carried by the maskShapeN
      // function arg (validCol_dst). A const extent stays byte-identical (==
      // shape for a full copy, or the const slice width), so every existing
      // caller is unchanged -- only the previously-uncompilable runtime-slice
      // case changes.
      PrimExpr gm2ub_tmpl_n = dst_extents[dst->shape.size() - 1];
      if (!gm2ub_tmpl_n->IsInstance<IntImmNode>()) {
        gm2ub_tmpl_n = dst->shape[dst->shape.size() - 1];
        // The shape fallback must itself be a compile-time constant; a buffer
        // declared with a dynamic inner dim would still emit a non-const
        // template arg (the invalid-C++ case above), so fail early and clearly.
        ICHECK(gm2ub_tmpl_n->IsInstance<IntImmNode>())
            << "copy_gm_to_ub: the inner (column) dimension of the destination "
               "buffer shape must be a compile-time constant, but got "
            << gm2ub_tmpl_n;
      }
      ss << gm2ub_tmpl_n;
      if (dst->shape.size() > 1) {
        local_tmpl_m = compute_blocklen(dst, dst_extents);
        ss << ", " << local_tmpl_m;
      }
      local_tmpl_n = gm2ub_tmpl_n;
      ss << ">";
    } else if (dst.scope() == "global") {
      config.ub2gm = true;
      strideN = compute_strideN(dst, dst_extents);
      config.needs_strideN = true;

      ss << "copy_ub_to_gm<";
      ss << get_dtype(dst) << ", ";
      // The template column dim must be a compile-time constant. Prefer the
      // SOURCE BUFFER's physical inner dim: the helper derives the UB-side
      // inter-burst stride from it, so a const narrower region (a column
      // slice into a wider tile) would otherwise pack rows at the region
      // width instead of the tile's real pitch (issue #929). This also
      // matches what the PTO backend already does for column slices ("use
      // buffer's full shape for UB tile ... fixes column-slice DMA stride
      // mismatch") and what the dynamic-extent path below has always done.
      // A full-tile copy keeps an identical template.
      PrimExpr ub2gm_tmpl_n = src->shape[src->shape.size() - 1];
      if (!ub2gm_tmpl_n->IsInstance<IntImmNode>()) {
        // Dynamic buffer shape: fall back to the region's inner extent (the
        // previous behaviour) -- it must itself be compile-time.
        ub2gm_tmpl_n = src_extents[src->shape.size() - 1];
        ICHECK(ub2gm_tmpl_n->IsInstance<IntImmNode>())
            << "copy_ub_to_gm: the inner (column) dimension of the source "
               "buffer shape must be a compile-time constant, but got "
            << ub2gm_tmpl_n;
      }
      ss << ub2gm_tmpl_n;
      if (src->shape.size() > 1) {
        local_tmpl_m = compute_blocklen(src, src_extents);
        ss << ", " << local_tmpl_m;
      }
      local_tmpl_n = ub2gm_tmpl_n;
      ss << ">";
    } else if (dst.scope() == "shared.l1") {
      config.virtual_channel = true;
      ss << "copy_ub_to_l1<";
      ss << get_dtype(dst) << ", ";
      ss << src_extents[src->shape.size() - 1];
      if (src->shape.size() > 1) {
        ss << ", " << compute_blocklen(src, src_extents);
      }
      ss << ">";
    } else if (src.scope() == "wmma.accumulator") {
      config.virtual_channel = true;
      ss << "copy_l0c_to_ub<";
      ss << get_dtype(src) << ", " << get_dtype(dst) << ", ";
      ss << "layout::RowMajor, "; // In "ub -> gm -> l1" path, gm is
                                  // always row major
      ss << src->shape[src->shape.size() - 2] << ", "
         << src->shape[src->shape.size() - 1];
      ss << ", " << enRelu << ">";
    } else {
      PrimExpr len = 1;
      for (auto &shape : dst_extents) {
        len *= shape;
      }
      ss << "copy_ub_to_ub<" << get_dtype(dst) << ", " << get_dtype(src) << ", "
         << len << ">";
      config.ub2ub = true;
    }
  } else {
    LOG(FATAL) << "Unsupported scope: src = " << src.scope()
               << ", dst = " << dst.scope();
  }

  if (!config.print_ub) {
    ss << "<" << get_dtype(src) << ", ";

    if (config.l0c2gm) {
      ss << get_dtype(dst) << ", ";
    }

    if (config.print_gm_layout) {
      ss << (T.layout_map.count(src) ? T.layout_map[src]->AscendLayoutStr()
                                     : "layout::RowMajor")
         << ", ";
    }

    if (config.print_src_layout) {
      ICHECK(T.layout_map.count(src))
          << "Layout map does not contain source buffer: " << src->name;
      ss << T.layout_map[src]->AscendLayoutStr() << ", ";
    } else if (config.print_dst_layout) {
      ICHECK(T.layout_map.count(dst))
          << "Layout map does not contain destination buffer: " << dst->name;
      ss << T.layout_map[dst]->AscendLayoutStr() << ", ";
    }

    int src_ndim = src->shape.size(), dst_ndim = dst->shape.size();

    if (dst.scope() == "global") {
      ss << src->shape[src_ndim - 2] << ", " << src->shape[src_ndim - 1];
      if (config.l0c2gm) {
        ss << ", " << enRelu;
      }
      ss << ">";
    } else if (src.scope() == "global") {
      ss << dst->shape[dst_ndim - 2] << ", " << dst->shape[dst_ndim - 1] << ">";
    } else {
      ss << src->shape[src_ndim - 2] << ", " << src->shape[src_ndim - 1];
      if (config.l12l0) {
        transposeL1 == 0 ? ss << ", false" << ">" : ss << ", true" << ">";
      }
      // ss << src->shape[src_ndim - 2] << ", " << src->shape[src_ndim - 1] <<
      // ", "
      //    << dst->shape[dst_ndim - 2] << ", " << dst->shape[dst_ndim - 1] <<
      //    ">";
    }
  }

  auto src_indices = build_indices(src_range);
  auto dst_indices = build_indices(dst_range);

  auto src_new_indices = T.layout_map.count(src)
                             ? T.layout_map[src]->Forward(src_indices)
                             : src_indices;
  auto dst_new_indices = T.layout_map.count(dst)
                             ? T.layout_map[dst]->Forward(dst_indices)
                             : dst_indices;

  auto src_new_buffer = T.buffer_remap.count(src) ? T.buffer_remap[src] : src;
  auto dst_new_buffer = T.buffer_remap.count(dst) ? T.buffer_remap[dst] : dst;

  PrimExpr src_len = 1;
  for (auto &shape : src_extents) {
    src_len *= shape;
  }

  PrimExpr dst_len = 1;
  for (auto &shape : dst_extents) {
    dst_len *= shape;
  }

  auto src_ptr = src_new_buffer.access_ptr(
      1, src_new_buffer->dtype, 1,
      src_new_buffer.OffsetOf(src_new_indices).back(), src_len);
  auto dst_ptr = dst_new_buffer.access_ptr(
      2, dst_new_buffer->dtype, 1,
      dst_new_buffer.OffsetOf(dst_new_indices).back(), dst_len);

  PrimExpr tmp_ptr;
  if (tmp.defined()) {
    auto tmp_new_indices =
        T.layout_map.count(tmp)
            ? T.layout_map[tmp]->Forward(build_indices(tmp_range))
            : build_indices(tmp_range);
    auto tmp_new_buffer = T.buffer_remap.count(tmp) ? T.buffer_remap[tmp] : tmp;
    PrimExpr tmp_len = 1;
    for (auto &shape : tmp_extents) {
      tmp_len *= shape;
    }
    tmp_ptr = tmp_new_buffer.access_ptr(
        3, tmp_new_buffer->dtype, 1,
        tmp_new_buffer.OffsetOf(tmp_new_indices).back(), tmp_len);
  }

  auto compute_valid_extent = [](PrimExpr min_val, PrimExpr extent,
                                 PrimExpr shape) -> PrimExpr {
    PrimExpr remaining = shape - min_val;
    if (remaining.dtype().lanes() > 1) {
      return extent;
    }
    return Select(remaining >= extent, extent,
                  Select(remaining > 0, remaining, 0));
  };

  auto find_active_dim_indices =
      [](const Array<PrimExpr> &extents) -> std::vector<int> {
    std::vector<int> active_indices;
    int size = static_cast<int>(extents.size());

    // Traverse from 0 to size-1, find dimensions where extent != 1
    for (int i = 0; i < size; ++i) {
      if (auto *int_imm = extents[i].as<IntImmNode>()) {
        if (int_imm->value != 1) {
          active_indices.push_back(i);
        }
      } else {
        active_indices.push_back(i);
      }
    }

    // The last dimension must always be included in the result
    if (size >= 1 &&
        (active_indices.empty() || active_indices.back() != size - 1)) {
      active_indices.push_back(size - 1);
    }

    // Special: If extents.size() >= 2 and active_indices.size() == 1,
    // insert size-2 at the second-to-last position
    if (size >= 2 && active_indices.size() == 1) {
      active_indices.insert(active_indices.begin(), size - 2);
    }

    return active_indices;
  };

  PrimExpr validRow_src, validCol_src, validRow_dst, validCol_dst;

  // src: compute validRow and validCol using active dimension indices
  std::vector<int> src_active = find_active_dim_indices(src_extents);
  if (src_active.size() >= 2) {
    int row_idx = src_active[src_active.size() - 2];
    int col_idx = src_active.back();
    validRow_src =
        compute_valid_extent(src_range[row_idx]->min,
                             src_range[row_idx]->extent, src->shape[row_idx]);
    validCol_src =
        compute_valid_extent(src_range[col_idx]->min,
                             src_range[col_idx]->extent, src->shape[col_idx]);
  } else if (src_active.size() == 1) {
    int col_idx = src_active[0];
    validRow_src = Integer(1);
    validCol_src =
        compute_valid_extent(src_range[col_idx]->min,
                             src_range[col_idx]->extent, src->shape[col_idx]);
  } else {
    validRow_src = 0;
    validCol_src = 0;
  }

  // dst: compute validRow and validCol using active dimension indices
  std::vector<int> dst_active = find_active_dim_indices(dst_extents);
  if (dst_active.size() >= 2) {
    int row_idx = dst_active[dst_active.size() - 2];
    int col_idx = dst_active.back();
    validRow_dst =
        compute_valid_extent(dst_range[row_idx]->min,
                             dst_range[row_idx]->extent, dst->shape[row_idx]);
    validCol_dst =
        compute_valid_extent(dst_range[col_idx]->min,
                             dst_range[col_idx]->extent, dst->shape[col_idx]);
  } else if (dst_active.size() == 1) {
    int col_idx = dst_active[0];
    validRow_dst = Integer(1);
    validCol_dst =
        compute_valid_extent(dst_range[col_idx]->min,
                             dst_range[col_idx]->extent, dst->shape[col_idx]);
  } else {
    validRow_dst = 0;
    validCol_dst = 0;
  }

  // Issue #929: a GM<->UB copy whose DMA would use multiple bursts with a
  // sub-32-byte payload (e.g. a strided one-element-per-row load into a flat
  // 1D UB buffer) or a non 32-byte-expressible UB-side row pitch cannot be
  // represented by the MTE engine -- bursts scatter to 32-byte boundaries and
  // run past the buffer, silently corrupting neighbouring UB allocations.
  // Lower such copies to a correct fallback: an elementwise scalar loop for
  // loads (see NeedsScalarGMCopyFallback / MakeScalarGMCopyLoop), or a
  // 32-byte-aligned UB staging tile plus per-row single-burst DMAs for stores
  // (MakeStagedUBToGMCopy -- scalar GM stores are unreliable on this uArch,
  // their write-back cache drops partial-line stores). Only the provably-
  // broken shapes change; every aligned full-tile / single-burst copy keeps
  // its DMA. The PTO backend keeps its own lowering for these shapes: its
  // TLOAD model expresses sub-32B runs natively (copy_cv's ub_to_l1
  // sub-32B-row case passes on PTO), so the gate below is AscendC-only.
  std::string target_model;
  if (T.target->attrs.count("model")) {
    target_model = Downcast<String>(T.target->attrs["model"]);
  }
  if ((config.gm2ub || config.ub2gm) && target_model != "pto") {
    const bool is_load = config.gm2ub;
    const Buffer &gm_buf = is_load ? src : dst;
    const Buffer &local_buf = is_load ? dst : src;
    const PrimExpr &vrow = is_load ? validRow_src : validRow_dst;
    const PrimExpr &vcol = is_load ? validCol_src : validCol_dst;
    const std::vector<int> &gm_active = is_load ? src_active : dst_active;
    const Array<Range> &gm_rg = is_load ? src_range : dst_range;
    int elem_bytes = gm_buf->dtype.bytes();
    // A single-active-dim region is a single-burst DMA (row count 1): the pad
    // path handles any width, so it can never trigger the fallback. The valid
    // extents are runtime clamps (Select nodes around runtime bases); for the
    // compile-time shape check use the region's raw extents, which are the
    // in-bounds values the DMA actually moves.
    bool needs_fallback = false;
    if (gm_active.size() >= 2) {
      int gm_row_idx = gm_active[gm_active.size() - 2];
      int gm_col_idx = gm_active.back();
      PrimExpr row_extent = analyzer->Simplify(gm_rg[gm_row_idx]->extent);
      PrimExpr col_extent = analyzer->Simplify(gm_rg[gm_col_idx]->extent);
      needs_fallback = NeedsScalarGMCopyFallback(
          row_extent, col_extent, elem_bytes, local_tmpl_n,
          local_buf->shape[local_buf->shape.size() - 1]);
    }
    if (needs_fallback) {
      if (is_load) {
        PrimExpr pad_val = padValue;
        if (pad_val->dtype != local_buf->dtype) {
          pad_val = Cast(local_buf->dtype, pad_val);
        }
        return MakeScalarGMCopyLoop(
            gm_buf, gm_rg, gm_active, local_buf, dst_range, dst_extents, vrow,
            vcol, local_tmpl_m, local_tmpl_n, pad_val, analyzer);
      }
      return MakeStagedUBToGMCopy(T, gm_buf, gm_rg, gm_active, local_buf,
                                  src_range, src_extents, vrow, vcol,
                                  get_dtype(dst), analyzer);
    }
  }

  Array<PrimExpr> new_args;
  new_args.push_back(StringImm(ss.str()));
  new_args.push_back(src_ptr);
  new_args.push_back(dst_ptr);

  if (config.needs_strideN) {
    new_args.push_back(strideN);
  }

  if (config.l0_dst_split) {
    int dst_dim = dst->shape.size();
    PrimExpr dm = dst->shape[dst_dim - 2];
    PrimExpr dn = dst->shape[dst_dim - 1];
    // realK overrides the L0 fractal's K extent so it matches the following
    // mma's runtime K. K is the trailing dim for matrix_a ([M, K]) and the
    // leading one for matrix_b ([K, N]). realK == 0 keeps dst->shape, which is
    // byte-identical for every existing L1->L0 copy.
    const auto *rk_imm = realK.as<IntImmNode>();
    if (!(rk_imm && rk_imm->value == 0)) {
      if (dst.scope() == "wmma.matrix_a") {
        dn = realK;
      } else if (dst.scope() == "wmma.matrix_b") {
        dm = realK;
      }
    }
    // realN overrides matrix_b's N extent, the axis realK does not cover.
    const auto *rn_imm = realN.as<IntImmNode>();
    if (!(rn_imm && rn_imm->value == 0) && dst.scope() == "wmma.matrix_b") {
      dn = realN;
    }
    new_args.push_back(dm);
    new_args.push_back(dn);
  }

  if (config.l0c2gm) {
    new_args.push_back(compute_strideN(dst, dst_extents));
    new_args.push_back(validRow_dst);
    new_args.push_back(validCol_dst);
    new_args.push_back(src->shape[src->shape.size() - 2]);
    new_args.push_back(src->shape[src->shape.size() - 1]);
    new_args.push_back(Bool(enRelu)); // Add enable_relu parameter
    // unitFlag goes last, not in call order: this list is shared with the PTO
    // codegen, which reads several of these by position (the pad enum and the
    // relu flag among them). The ascendc codegen picks it up off the end and
    // emits it in the position the C++ helper expects.
    new_args.push_back(unitFlag);
  }

  if (config.gm2l1) {
    new_args.push_back(compute_strideN(src, src_extents));
    new_args.push_back(validRow_src);
    new_args.push_back(validCol_src);
    new_args.push_back(dst->shape[dst->shape.size() - 2]);
    new_args.push_back(dst->shape[dst->shape.size() - 1]);
  }

  if (config.gm2ub) {
    new_args.push_back(validRow_src);
    new_args.push_back(validCol_src);
    PrimExpr pad_val = padValue;
    if (pad_val->dtype != dst->dtype) {
      pad_val = Cast(dst->dtype, pad_val);
    }
    new_args.push_back(pad_val);
    // Physical UB tile dims (row pitch). These trail pad_val and are consumed
    // by AscendTailMaskPropagation to model the tail rect; CopyCodegen prints
    // only the first 4 extra args (strideN, validRow, validCol, pad_val).
    if (dst->shape.size() > 1) {
      new_args.push_back(dst->shape[dst->shape.size() - 2]);
    }
    new_args.push_back(dst->shape[dst->shape.size() - 1]);
  }

  if (config.ub2gm) {
    new_args.push_back(validRow_dst);
    new_args.push_back(validCol_dst);
    if (src->shape.size() > 1) {
      new_args.push_back(src->shape[src->shape.size() - 2]);
    }
    new_args.push_back(src->shape[src->shape.size() - 1]);
  }

  if (config.virtual_channel) {
    new_args.push_back(src_extents[src_extents.size() - 1]); // src_N
    if (src_extents.size() >= 2) {
      new_args.push_back(src_extents[src_extents.size() - 2]); // src_M
    } else {
      new_args.push_back(IntImm(DataType::Int(32), 0)); // 1D: no M dim
    }
    new_args.push_back(dst_extents[dst_extents.size() - 2]); // dst_M
    new_args.push_back(dst_extents[dst_extents.size() - 1]); // dst_N
    // Add tmp buffer for UB->L1 copy on A5 (for ND->Nz conversion)
    if (tmp.defined()) {
      new_args.push_back(tmp_ptr);
      new_args.push_back(tmp_extents[tmp_extents.size() - 2]); // tmp_M
      new_args.push_back(tmp_extents[tmp_extents.size() - 1]); // tmp_N
    }
  }

  if (config.ub2ub) {
    PrimExpr src_tile_rows = src_extents.size() >= 2
                                 ? src_extents[src_extents.size() - 2]
                                 : Integer(1);
    PrimExpr src_tile_cols = src_extents[src_extents.size() - 1];
    PrimExpr src_buf_cols = src->shape[src->shape.size() - 1];
    PrimExpr dst_tile_rows = dst_extents.size() >= 2
                                 ? dst_extents[dst_extents.size() - 2]
                                 : Integer(1);
    PrimExpr dst_tile_cols = dst_extents[dst_extents.size() - 1];
    PrimExpr dst_buf_cols = dst->shape[dst->shape.size() - 1];
    new_args.push_back(src_tile_rows);
    new_args.push_back(src_tile_cols);
    new_args.push_back(src_buf_cols);
    new_args.push_back(dst_tile_rows);
    new_args.push_back(dst_tile_cols);
    new_args.push_back(dst_buf_cols);
  }
  // if (config.l12l0) {
  //   ICHECK(src->shape.size() == dst->shape.size());
  //   bool is_extract = false;
  //   auto dst_shape_size = dst->shape.size();
  //   auto src_shape_size = src->shape.size();
  //   for (int i = dst_shape_size - 1; i >= dst_shape_size - 2; i--) {
  //     if (i == -1) {
  //         break;
  //     }
  //     // std::cout << "dst_shape_size -2 = " << dst_shape_size - 2  << ", i =
  //     " << i << ", dst->shape: " << dst->shape[i]
  //     //   << "src->shape: " << src->shape[i] << "\n";
  //     if (src->shape[i].as<IntImmNode>()->value !=
  //     dst->shape[i].as<IntImmNode>()->value) {
  //         is_extract = true;
  //     }
  //     new_args.push_back(src_indices[i]);
  //   }
  //   new_args.push_back(Bool(is_extract));
  //   auto dst_var = dst_ptr.as<CallNode>()->args[1];
  //   std::cout << "newargs in copy to l1: " << new_args << "\n";
  //   std::cout << "src_indices: " << src_indices << "\n";
  // }

  auto new_call = Call(DataType::Handle(), builtin::call_extern(), new_args);
  return Evaluate(new_call);
}

Stmt AscendAtomicAdd::Lower(const LowerArgs &T,
                            arith::Analyzer *analyzer) const {
  auto get_dtype = [](const Buffer &buf) -> std::string {
    auto dtype = buf->dtype;
    if (dtype.is_float16()) {
      return "half";
    } else if (dtype.is_float() && dtype.bits() == 32) {
      return "float";
    } else if (dtype.is_int() && dtype.bits() == 8) {
      return "int8_t";
    } else if (dtype.is_int() && dtype.bits() == 16) {
      return "int16_t";
    } else if (dtype.is_int() && dtype.bits() == 32) {
      return "int";
    } else if (dtype.is_int() && dtype.bits() == 64) {
      return "int64_t";
    } else if (dtype.is_uint() && dtype.bits() == 8) {
      return "uint8_t";
    } else if (dtype.is_uint() && dtype.bits() == 16) {
      return "uint16_t";
    } else if (dtype.is_uint() && dtype.bits() == 32) {
      return "uint32_t";
    } else if (dtype.is_uint() && dtype.bits() == 64) {
      return "uint64_t";
    } else if (dtype.is_bfloat16()) {
      return "bfloat16_t";
    } else if (dtype.is_float8_e4m3fn()) {
      return "float8_e4m3_t";
    } else if (dtype.is_float8_e5m2()) {
      return "float8_e5m2_t";
    }
    LOG(FATAL) << "Unsupported data type for tl.ascend_atomic_add: " << dtype;
    return "";
  };

  auto compute_strideN = [](const Buffer &buf,
                            const Array<PrimExpr> &extents) -> PrimExpr {
    // The GM copy is emitted as a 2D DMA: blockCount rows (the second-to-last
    // active dim, i.e. the highest dim below the last whose extent != 1) by
    // blockLen columns (the last dim), with one uniform stride between
    // consecutive rows. That stride is the product of every buffer dim below
    // the row dim, so extent-1 dims sitting under the row dim must be folded
    // into it (e.g. Query[cid, m:m+BM, n2, g*D:(g+1)*D] folds the scalar n2
    // dim into the stride). But when every leading extent is 1 -- a row slice
    // like C2[row, 0:N] -- the row dim is the second-to-last dim itself
    // (blockCount = 1) and the stride is just the last buffer dim: folding the
    // leading dims in would return the whole buffer size as the stride and
    // corrupt the DMA (issue #1263).
    PrimExpr strideN = buf->shape[buf->shape.size() - 1];
    PrimExpr folded = Integer(1);
    bool has_row_dim = false;
    for (int i = static_cast<int>(extents.size()) - 2; i >= 0; --i) {
      auto *extent = extents[i].as<IntImmNode>();
      if (!extent || extent->value != 1) {
        has_row_dim = true;
        break;
      }
      folded = folded * buf->shape[i];
    }
    if (has_row_dim) {
      strideN = strideN * folded;
    }
    return strideN;
  };

  auto compute_blocklen = [](const Buffer &buf,
                             const Array<PrimExpr> &extents) -> PrimExpr {
    PrimExpr res = buf->shape[buf->shape.size() - 2];
    auto ext_size = extents.size();
    if (ext_size > 1 && extents[ext_size - 2]->IsInstance<IntImmNode>() &&
        res->IsInstance<IntImmNode>()) {
      auto extent =
          static_cast<int>(extents[ext_size - 2].as<IntImmNode>()->value);
      auto shape = static_cast<int>(res.as<IntImmNode>()->value);
      res = shape < extent ? res : extents[ext_size - 2];
    }
    return res;
  };

  auto build_indices = [](const Array<Range> &range) -> Array<PrimExpr> {
    Array<PrimExpr> indices;
    for (size_t i = 0; i < range.size(); i++) {
      indices.push_back(range[i]->min);
    }
    return indices;
  };

  auto compute_valid_extent = [](PrimExpr min_val, PrimExpr extent,
                                 PrimExpr shape) -> PrimExpr {
    PrimExpr remaining = shape - min_val;
    if (remaining.dtype().lanes() > 1) {
      return extent;
    }
    return Select(remaining >= extent, extent,
                  Select(remaining > 0, remaining, 0));
  };

  auto find_active_dim_indices =
      [](const Array<PrimExpr> &extents) -> std::vector<int> {
    std::vector<int> active_indices;
    int size = static_cast<int>(extents.size());
    for (int i = 0; i < size; ++i) {
      if (auto *int_imm = extents[i].as<IntImmNode>()) {
        if (int_imm->value != 1) {
          active_indices.push_back(i);
        }
      } else {
        active_indices.push_back(i);
      }
    }
    if (size >= 1 &&
        (active_indices.empty() || active_indices.back() != size - 1)) {
      active_indices.push_back(size - 1);
    }
    if (size >= 2 && active_indices.size() == 1) {
      active_indices.insert(active_indices.begin(), size - 2);
    }
    return active_indices;
  };

  ICHECK(dst.scope() == "global")
      << "tl.ascend_atomic_add V1 requires global dst, got " << dst.scope();
  ICHECK(src.scope() == "shared.ub" || src.scope() == "wmma.accumulator")
      << "tl.ascend_atomic_add V1 requires UB/shared or L0C/wmma.accumulator "
         "src, got "
      << src.scope();
  ICHECK(src->dtype == dst->dtype)
      << "tl.ascend_atomic_add requires src and dst dtype to match, got src "
      << src->dtype << " and dst " << dst->dtype;
  ICHECK_EQ(src_extents.size(), src->shape.size())
      << "tl.ascend_atomic_add source region rank must match source buffer "
         "rank";
  ICHECK_EQ(dst_extents.size(), dst->shape.size())
      << "tl.ascend_atomic_add destination region rank must match "
         "destination buffer rank";

  std::stringstream ss;
  // The inner (column) dim is a template arg, so a runtime inner-extent --
  // e.g. atomic_add(A[rows, 0:n], src_ub[:, 0:n]) with a dynamic n -- would
  // put a non-const expr in the template and emit invalid C++. Prefer the
  // SOURCE BUFFER's physical inner dim (the helper derives the UB-side
  // inter-burst stride from it -- see the copy_ub_to_gm template note above /
  // issue #929); fall back to the region's inner extent when the buffer shape
  // is dynamic. The runtime column count is carried by validCol_dst below.
  PrimExpr atomic_tmpl_n = src->shape[src->shape.size() - 1];
  if (!atomic_tmpl_n->IsInstance<IntImmNode>()) {
    atomic_tmpl_n = src_extents[src->shape.size() - 1];
    ICHECK(atomic_tmpl_n->IsInstance<IntImmNode>())
        << "tl.ascend_atomic_add: the inner (column) dimension of the source "
           "buffer shape must be a compile-time constant, but got "
        << atomic_tmpl_n;
  }
  if (src.scope() == "shared.ub") {
    ss << "tl::ascend::atomic_add_ub_to_gm<";
    ss << get_dtype(dst) << ", " << atomic_tmpl_n;
    if (src->shape.size() > 1) {
      ss << ", " << compute_blocklen(src, src_extents);
    }
    ss << ">";
  } else if (src.scope() == "wmma.accumulator") {
    ss << "tl::ascend::atomic_add_l0c_to_gm<";
    ss << get_dtype(src) << ", " << get_dtype(dst) << ", "
       << (T.layout_map.count(src) ? T.layout_map[src]->AscendLayoutStr()
                                   : "layout::RowMajor");
    if (src->shape.size() > 1) {
      ss << ", " << compute_blocklen(src, src_extents) << ", " << atomic_tmpl_n;
    } else {
      ss << ", 1, " << atomic_tmpl_n;
    }
    ss << ">";
  }

  auto src_indices = build_indices(src_range);
  auto dst_indices = build_indices(dst_range);
  auto src_new_indices = T.layout_map.count(src)
                             ? T.layout_map[src]->Forward(src_indices)
                             : src_indices;
  auto dst_new_indices = T.layout_map.count(dst)
                             ? T.layout_map[dst]->Forward(dst_indices)
                             : dst_indices;

  auto src_new_buffer = T.buffer_remap.count(src) ? T.buffer_remap[src] : src;
  auto dst_new_buffer = T.buffer_remap.count(dst) ? T.buffer_remap[dst] : dst;

  // Scalarize indices: extract base from Ramp expressions to avoid vector lane
  // mismatches when Ramp lanes differ across dimensions (e.g. int32x32 vs
  // int32x64). The Ramp lanes already contribute to the access extent (src_len
  // / dst_len).
  auto scalarize = [](const PrimExpr &idx) -> PrimExpr {
    if (const auto *ramp = idx.as<RampNode>()) {
      return ramp->base;
    }
    return idx;
  };
  auto src_scalar_indices = src_new_indices.Map(scalarize);
  auto dst_scalar_indices = dst_new_indices.Map(scalarize);

  PrimExpr src_len = 1;
  for (auto &shape : src_extents) {
    src_len *= shape;
  }
  // dst_len must also account for lanes in Ramp indices
  PrimExpr dst_len = 1;
  for (size_t i = 0; i < dst_extents.size(); ++i) {
    PrimExpr dim_len = dst_extents[i];
    if (const auto *ramp = dst_new_indices[i].as<RampNode>()) {
      dim_len = dim_len * ramp->lanes;
    }
    dst_len = dst_len * dim_len;
  }

  auto src_offset = src_new_buffer.OffsetOf(src_scalar_indices).back();
  auto dst_offset = dst_new_buffer.OffsetOf(dst_scalar_indices).back();

  auto src_ptr = src_new_buffer.access_ptr(1, src_new_buffer->dtype, 1,
                                           src_offset, src_len);
  auto dst_ptr = dst_new_buffer.access_ptr(2, dst_new_buffer->dtype, 1,
                                           dst_offset, dst_len);

  // Compute effective extents accounting for lanes in Ramp indices,
  // so that find_active_dim_indices correctly identifies the 2D access pattern.
  Array<PrimExpr> effective_dst_extents;
  for (size_t i = 0; i < dst_extents.size(); ++i) {
    PrimExpr eff = dst_extents[i];
    if (const auto *ramp = dst_new_indices[i].as<RampNode>()) {
      // When a dimension was vectorized from extent N to 1 vector with N lanes,
      // the true element count is the Ramp lanes.  When the extent already
      // reflects the total element count (extent == Ramp lanes), keep it as-is.
      if (is_one(dst_extents[i])) {
        eff = ramp->lanes;
      }
    }
    effective_dst_extents.push_back(eff);
  }

  // Identify row/col Ramp dimensions for stride and boundary computation.
  // When vectorized, Ramps with different lane counts (e.g. 32, 64) mark the
  // true 2D access pattern, bypassing scrambled extents.
  auto get_ramp_lanes = [](const PrimExpr &idx) -> int {
    if (const auto *ramp = idx.as<RampNode>()) {
      if (auto *imm = ramp->lanes.as<IntImmNode>()) {
        return static_cast<int>(imm->value);
      }
    }
    return 0;
  };
  int row_ramp_idx = -1, col_ramp_idx = -1;
  for (size_t i = 0; i < dst_new_indices.size(); ++i) {
    int lanes = get_ramp_lanes(dst_new_indices[i]);
    if (lanes > 1) {
      if (row_ramp_idx < 0 ||
          lanes < get_ramp_lanes(dst_new_indices[row_ramp_idx])) {
        col_ramp_idx = row_ramp_idx >= 0 ? row_ramp_idx : col_ramp_idx;
        row_ramp_idx = static_cast<int>(i);
      } else {
        col_ramp_idx = static_cast<int>(i);
      }
    }
  }

  PrimExpr validRow_dst, validCol_dst;
  if (row_ramp_idx >= 0 && col_ramp_idx >= 0) {
    // Use Ramp-identified dimensions for boundary checks
    PrimExpr row_min = dst_scalar_indices[row_ramp_idx];
    PrimExpr col_min = dst_scalar_indices[col_ramp_idx];
    PrimExpr row_extent = effective_dst_extents[row_ramp_idx];
    PrimExpr col_extent = effective_dst_extents[col_ramp_idx];
    validRow_dst =
        compute_valid_extent(row_min, row_extent, dst->shape[row_ramp_idx]);
    validCol_dst =
        compute_valid_extent(col_min, col_extent, dst->shape[col_ramp_idx]);
  } else {
    std::vector<int> dst_active =
        find_active_dim_indices(effective_dst_extents);
    if (dst_active.size() >= 2) {
      int row_idx = dst_active[dst_active.size() - 2];
      int col_idx = dst_active.back();
      PrimExpr row_min = dst_scalar_indices[row_idx];
      PrimExpr col_min = dst_scalar_indices[col_idx];
      PrimExpr row_extent = effective_dst_extents[row_idx];
      PrimExpr col_extent = effective_dst_extents[col_idx];
      validRow_dst =
          compute_valid_extent(row_min, row_extent, dst->shape[row_idx]);
      validCol_dst =
          compute_valid_extent(col_min, col_extent, dst->shape[col_idx]);
    } else if (dst_active.size() == 1) {
      int col_idx = dst_active[0];
      PrimExpr col_min = dst_scalar_indices[col_idx];
      PrimExpr col_extent = effective_dst_extents[col_idx];
      validRow_dst = Integer(1);
      validCol_dst =
          compute_valid_extent(col_min, col_extent, dst->shape[col_idx]);
    } else {
      validRow_dst = Integer(0);
      validCol_dst = Integer(0);
    }
  }

  // Compute strideN: when Ramp dimensions are identified, use buffer shape
  // to compute the inter-row stride. Otherwise fall back to extent-based logic.
  PrimExpr strideN;
  if (row_ramp_idx >= 0 && col_ramp_idx >= 0) {
    strideN = Integer(1);
    for (int i = row_ramp_idx + 1; i < static_cast<int>(dst->shape.size());
         ++i) {
      strideN = strideN * dst->shape[i];
    }
  } else {
    strideN = compute_strideN(dst, dst_extents);
  }

  // Issue #929 (atomic variant): the atomic UB->GM DMA inherits the same
  // 32-byte UB-side burst alignment contract as the plain copy. A multi-burst
  // atomic-add whose burst payload is not a 32-byte multiple (or whose UB tile
  // pitch is not 32-byte expressible) silently corrupts memory, and unlike the
  // copy there is no correct fallback: the scalar atomic API only exists on
  // 5102/3510-class parts, and single-burst decomposition still needs
  // 32-byte-aligned UB reads. Fail loudly instead of emitting the broken DMA.
  // (PTO keeps its own lowering -- see the copy path note above.)
  std::string atomic_target_model;
  if (T.target->attrs.count("model")) {
    atomic_target_model = Downcast<String>(T.target->attrs["model"]);
  }
  if (src.scope() == "shared.ub" && atomic_target_model != "pto") {
    // See the copy path above: use the region's raw extents (the runtime
    // clamps are Select nodes) for the compile-time shape check.
    PrimExpr row_extent, col_extent;
    if (row_ramp_idx >= 0 && col_ramp_idx >= 0) {
      row_extent = effective_dst_extents[row_ramp_idx];
      col_extent = effective_dst_extents[col_ramp_idx];
    } else {
      std::vector<int> at_dst_active =
          find_active_dim_indices(effective_dst_extents);
      if (at_dst_active.size() >= 2) {
        row_extent =
            effective_dst_extents[at_dst_active[at_dst_active.size() - 2]];
        col_extent = effective_dst_extents[at_dst_active.back()];
      }
    }
    if (row_extent.defined() && col_extent.defined() &&
        NeedsScalarGMCopyFallback(analyzer->Simplify(row_extent),
                                  analyzer->Simplify(col_extent),
                                  dst->dtype.bytes(), atomic_tmpl_n,
                                  src->shape[src->shape.size() - 1])) {
      LOG(FATAL)
          << "tl.tile.atomic_add: a strided UB->GM scatter whose contiguous "
             "run per row is smaller than 32 bytes (or whose UB tile row "
             "pitch is not a 32-byte multiple) cannot be expressed as an MTE "
             "DMA and has no correct scalar fallback on this target; it "
             "previously corrupted memory silently. Got a "
          << col_extent << "-column add per row with " << row_extent
          << " rows from a UB tile of inner dim "
          << src->shape[src->shape.size() - 1]
          << ". Restructure the access (e.g. stage through a UB tile whose "
             "rows are 32-byte multiples) or add per-row.";
    }
  }

  Array<PrimExpr> new_args;
  new_args.push_back(StringImm(ss.str()));
  new_args.push_back(src_ptr);
  new_args.push_back(dst_ptr);
  new_args.push_back(strideN);
  new_args.push_back(validRow_dst);
  new_args.push_back(validCol_dst);

  auto new_call = Call(DataType::Handle(), builtin::call_extern(), new_args);
  return Evaluate(new_call);
}

LayoutMap AscendCopy::InferLayout(const LayoutInferArgs &T, InferLevel level) {
  LayoutMap results;
  // TODO: add logic to infer layout for AscendCopy
  return results;
}

LayoutMap AscendAtomicAdd::InferLayout(const LayoutInferArgs &T,
                                       InferLevel level) {
  LayoutMap results;
  return results;
}

TIR_REGISTER_TL_OP(AscendCopy, ascend_copy)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(AscendAtomicAdd, ascend_atomic_add)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

const Op &ascend_atomic_add() { return AscendAtomicAdd::Get(); }

TIR_DEFINE_TL_BUILTIN(ascend_add)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_sub)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_mul)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_div)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_max)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_min)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_bitwise_and)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_bitwise_or)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_adds)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_subs)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_muls)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_divs)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_maxs)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_mins)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_compare)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_compare_scalar)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_exp)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_ln).set_num_inputs(3).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_abs)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_reciprocal)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_sqrt)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_rsqrt)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_relu)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_bitwise_not)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_select)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_leaky_relu)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_axpy)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_mul_add_dst)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_bitwise_lshift)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_bitwise_rshift)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_sin)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_cos)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_transpose)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_createvecindex)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_fill)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_arith_progression)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_sort)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_merge_sort)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_topk)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_shmem_put_nbi)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_shmem_get_nbi)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_shmem_ub_put_nbi)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_shmem_ub_get_nbi)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_gather_mask)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_gatherb)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_init_sort_buf)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_sort32)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_gather)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_reduce)
    // tag, dst, src, optional tmp, clear, optional physical row
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_block_reduce_max)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_block_reduce_min)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_block_reduce_sum)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_cast)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_set_deq_scale)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_pow)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_bitwise_xor)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_broadcast)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_row_expand_mul)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_wait_cross_flag)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_set_cross_flag)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_wait_flag)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_set_flag)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_pipe_barrier)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_free_pipe)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_sync_all)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_gemm_v0)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_gemm_v1)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_printf)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_dump_tensor)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_src_code)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_bilinear_interpolation)
    .set_num_inputs(11)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_wholereducemax)
    .set_num_inputs(8)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_wholereducemin)
    .set_num_inputs(8)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_wholereducesum)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_auto_barrier)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_auto_set_flag)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_auto_wait_flag)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_auto_set_cross_flag)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_auto_wait_cross_flag)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_use_swizzle)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// -1 = variadic: the 6 required args (name, A, B, C, init, K) plus the optional
// trailing n_actual / unitFlag pair.
TIR_DEFINE_TL_BUILTIN(ascend_mma)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_sigmoid)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_silu)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_clamp_max)
    // The public call has one optional workspace operand, removed by
    // InjectTmpBuffer for the managed AscendC path.
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_clamp_min)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_clamp)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_round)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_reinterpretcast)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_sub_experiment)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_abs_experiment)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_mins_experiment)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_reducesum_experiment)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_reducesum_mask_experiment)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_gather_mask_experiment)
    .set_num_inputs(11)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_fill_experiment)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_sum_experiment)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_datacachecleanandinvalid_experiment)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_brcb_experiment)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_row_expand_mul_experiment)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_row_expand_sub_experiment)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_row_expand_div_experiment)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_exp_experiment)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_copy_cv_experiment)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_copy_vc_experiment)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// Internal tail-aware ops (see ascend.h). They carry an AscendC op tag string,
// buffer pointers, and the runtime tail rectangle.
TIR_DEFINE_TL_BUILTIN(ascend_tail_unary)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_tail_binary)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_tail_scalar)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_tail_reduce)
    .set_num_inputs(8)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

#define TL_ASCEND_SEMANTIC_OP(...)
#define TL_ASCEND_PHYSICAL(selected, selector, dtype_domain, operands,         \
                           payload, emitter, intrinsic)                        \
  TIR_DEFINE_TL_BUILTIN(selected)                                              \
      .set_num_inputs(-1)                                                      \
      .set_attr<TCallEffectKind>("TCallEffectKind",                            \
                                 Integer(CallEffectKind::kOpaque));
#define TL_ASCEND_HELPER(selected, selector, contract)                         \
  TIR_DEFINE_TL_BUILTIN(selected)                                              \
      .set_num_inputs(-1)                                                      \
      .set_attr<TCallEffectKind>("TCallEffectKind",                            \
                                 Integer(CallEffectKind::kOpaque));
#include "ascend_vector_mask_ops.inc"
#undef TL_ASCEND_HELPER
#undef TL_ASCEND_PHYSICAL
#undef TL_ASCEND_SEMANTIC_OP

TIR_DEFINE_TL_BUILTIN(ascend_set_mask_mode)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_set_mask_payload)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_tail_compare)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_tail_compare_scalar)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_tail_select)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_tail_broadcast)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
} // namespace tl
} // namespace tvm
