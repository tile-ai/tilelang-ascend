// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file src/transform/ascend_tail_mask_propagation.cc
 * \brief Propagate UB tail valid-regions and rewrite vector ops to tail-aware
 *        variants for the AscendC and PTO backends.
 *
 * After LowerTileOp, a GM->UB copy is lowered to
 *   call_extern("tl::ascend::copy_gm_to_ub<...>", src_ptr, dst_ptr,
 *               strideN, validRow, validCol, [physRow], physCol)
 * and the element-wise / reduce ops are plain tl.ascend_* calls whose `count`
 * argument spans the whole physical tile.
 *
 * This pass tracks, per UB data Var, the logical valid rectangle that was
 * loaded, propagates it through the UB data flow, and when an op touches a tail
 * buffer rewrites it to the internal tl.ascend_tail_* op (carrying the runtime
 * valid_row/valid_col/physical_col) so the codegen emits a tl::ascend::tail_*
 * helper that computes only over the valid region.
 *
 * Rewrites: unary / binary / scalar(immediate), compare / select / broadcast,
 * plus a conservative allow-list of 2D reduce contracts. Cast and UB-to-UB
 * copy only propagate the rectangle.
 */

#include "arith/ir_mutator_with_analyzer.h"
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "../op/ascend.h"
#include "common/ascend_tail_mask.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

/*! \brief Return the buffer data Var behind an access_ptr (or a bare Var). */
const VarNode *GetPtrVar(const PrimExpr &e) {
  if (const auto *call = e.as<CallNode>()) {
    if (call->op.same_as(builtin::tvm_access_ptr()) && call->args.size() >= 2) {
      return call->args[1].as<VarNode>();
    }
  }
  return e.as<VarNode>();
}

/*! \brief Map a plain binary tl op to its AscendC op tag, or "" if not one. */
std::string BinaryTag(const CallNode *call) {
  if (call->op.same_as(ascend_add()))
    return "Add";
  if (call->op.same_as(ascend_sub()))
    return "Sub";
  if (call->op.same_as(ascend_mul()))
    return "Mul";
  if (call->op.same_as(ascend_div()))
    return "Div";
  if (call->op.same_as(ascend_max()))
    return "Max";
  if (call->op.same_as(ascend_min()))
    return "Min";
  // bitwise And/Or are type-restricted (int16 only); keep them on the
  // full-tile path (still correct, per-lane).
  return "";
}

/*! \brief Map a plain unary tl op to its AscendC op tag, or "" if not one. */
std::string UnaryTag(const CallNode *call) {
  if (call->op.same_as(ascend_exp()))
    return "Exp";
  if (call->op.same_as(ascend_ln()))
    return "Ln";
  if (call->op.same_as(ascend_abs()))
    return "Abs";
  if (call->op.same_as(ascend_reciprocal()))
    return "Reciprocal";
  if (call->op.same_as(ascend_sqrt()))
    return "Sqrt";
  if (call->op.same_as(ascend_rsqrt()))
    return "Rsqrt";
  if (call->op.same_as(ascend_relu()))
    return "Relu";
  // bitwise Not is int-only; keep it on the full-tile path.
  return "";
}

/*! \brief Map a plain scalar tl op (immediate form) to its AscendC op tag. */
std::string ScalarTag(const CallNode *call) {
  if (call->op.same_as(ascend_adds()))
    return "Adds";
  if (call->op.same_as(ascend_muls()))
    return "Muls";
  if (call->op.same_as(ascend_maxs()))
    return "Maxs";
  if (call->op.same_as(ascend_mins()))
    return "Mins";
  return "";
}

/*! \brief Extract the external helper name from a call_extern's first arg. */
std::string ExternName(const CallNode *call) {
  if (!call->op.same_as(builtin::call_extern()))
    return "";
  if (call->args.empty())
    return "";
  if (const auto *s = call->args[0].as<StringImmNode>())
    return s->value;
  return "";
}

} // namespace

class AscendTailMaskPropagator : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f, bool rewrite_reduce) {
    arith::Analyzer analyzer;
    AscendTailMaskPropagator m(&analyzer, rewrite_reduce);
    m.CollectOutputHints(f->body);
    f.CopyOnWrite()->body = m.VisitStmt(f->body);
    return f;
  }

  AscendTailMaskPropagator(arith::Analyzer *analyzer, bool rewrite_reduce)
      : arith::IRMutatorWithAnalyzer(analyzer),
        rewrite_reduce_(rewrite_reduce) {}

private:
  // Per UB data Var -> current valid region. Absent => full (untracked).
  std::unordered_map<const VarNode *, TailMaskInfo> state_;
  // Direct UB->GM sinks provide the target valid rectangle needed when a
  // broadcast expands a dimension that is already full in its source.
  std::unordered_map<const VarNode *, TailMaskInfo> output_hints_;
  // Whether reduce ops may be rewritten to tail_reduce. The caller enables
  // this only for backends with a dedicated tail-reduce lowering.
  bool rewrite_reduce_ = true;

  // --- loop-variable scope tracking ---------------------------------------
  // A valid_row/valid_col expression seeded inside a loop may reference the
  // loop variable (e.g. copy_gm_to_ub in `for by` sets valid_col = f(by)).  If
  // a downstream op (e.g. reduce) runs *outside* that loop, emitting the
  // expression verbatim produces an undeclared-identifier compile error.  We
  // track all loop vars ever entered (all_loop_vars_) and those currently in
  // scope (active_loop_vars_); an expression referencing a loop var that is no
  // longer in scope is "out of scope" and the rewrite must bail.
  std::unordered_set<const VarNode *> all_loop_vars_;
  std::unordered_set<const VarNode *> active_loop_vars_;

  bool HasOutOfScopeLoopVar(const PrimExpr &e) const {
    if (!e.defined())
      return false;
    bool found = false;
    PostOrderVisit(e, [&](const ObjectRef &n) {
      if (const auto *v = n.as<VarNode>()) {
        if (all_loop_vars_.count(v) && !active_loop_vars_.count(v))
          found = true;
      }
    });
    return found;
  }

  // --- conservative guards -------------------------------------------------
  // Only float-like dtypes have validated tail helpers; int/uint stay on the
  // full-tile path to avoid unsupported AscendC intrinsic instantiations.
  static bool SupportedTailDtype(DataType dt) {
    return dt.is_float() || dt.is_bfloat16();
  }
  static bool SupportedCmpSelDtype(DataType dt) {
    return dt == DataType::Float(16) || dt == DataType::Float(32);
  }
  static bool SupportedPackedDtype(DataType dt) {
    return dt == DataType::UInt(8);
  }
  // Element dtype behind an access_ptr (mirrors GetAccessPtrDtype in codegen).
  static DataType PtrDtype(const PrimExpr &e) {
    const auto *ap = e.as<CallNode>();
    if (ap == nullptr || ap->args.empty())
      return DataType::Handle();
    if (const auto *c = ap->args[0].as<CallNode>())
      return c->dtype;
    return DataType::Handle();
  }
  // Element count (extent) behind an access_ptr.
  static PrimExpr PtrExtent(const PrimExpr &e) {
    if (const auto *ap = e.as<CallNode>())
      if (ap->op.same_as(builtin::tvm_access_ptr()) && ap->args.size() >= 4)
        return ap->args[3];
    return PrimExpr();
  }
  // The 2D tail model only holds when the op's element count equals the
  // physical tile (physical_row * physical_col). For 3D / mismatched tiles the
  // rewrite would compute the wrong region, so we bail to the full-tile path.
  bool CleanTail(const PrimExpr &count, const TailMaskInfo &m) {
    return m.is_tail() && m.physical_row.defined() &&
           m.physical_col.defined() && count.defined() &&
           analyzer_->CanProveEqual(count, m.physical_row * m.physical_col);
  }

  bool SamePhysicalShape(const TailMaskInfo &a, const TailMaskInfo &b) {
    return a.physical_row.defined() && a.physical_col.defined() &&
           b.physical_row.defined() && b.physical_col.defined() &&
           analyzer_->CanProveEqual(a.physical_row, b.physical_row) &&
           analyzer_->CanProveEqual(a.physical_col, b.physical_col);
  }

  bool CompareWidthSupported(const TailMaskInfo &m, DataType dtype) {
    if (!m.physical_col.defined() || !dtype.bytes())
      return false;
    PrimExpr eight = IntImm(DataType::Int(32), 8);
    PrimExpr vector_lanes = IntImm(DataType::Int(32), 256 / dtype.bytes());
    return analyzer_->CanProve(m.physical_col <= vector_lanes) &&
           analyzer_->CanProve(indexmod(m.physical_col, eight) == 0);
  }

  PrimExpr PackedStorageCol(const PrimExpr &ptr, const TailMaskInfo &data) {
    PrimExpr extent = PtrExtent(ptr);
    if (!extent.defined() || !data.physical_row.defined())
      return PrimExpr();
    return analyzer_->Simplify(indexdiv(extent, data.physical_row));
  }

  TailMaskInfo IntersectDataMask(const TailMaskInfo &packed,
                                 const TailMaskInfo &data) {
    TailMaskInfo out = packed;
    out.kind = TailMaskKind::kTail;
    out.storage_col = out.physical_col;
    if (!data.valid_row.defined() || !data.valid_col.defined())
      return out;
    out.valid_row = analyzer_->CanProveEqual(out.valid_row, data.valid_row)
                        ? out.valid_row
                        : Min(out.valid_row, data.valid_row);
    out.valid_col = analyzer_->CanProveEqual(out.valid_col, data.valid_col)
                        ? out.valid_col
                        : Min(out.valid_col, data.valid_col);
    return out;
  }

  void CollectOutputHints(const Stmt &body) {
    PostOrderVisit(body, [&](const ObjectRef &node) {
      const auto *call = node.as<CallNode>();
      if (call == nullptr ||
          ExternName(call).find("copy_ub_to_gm") == std::string::npos ||
          call->args.size() < 8)
        return;
      const VarNode *src_v = GetPtrVar(call->args[1]);
      if (src_v == nullptr)
        return;
      output_hints_[src_v] =
          MakeCopyMask(call->args[4], call->args[5], call->args[6],
                       call->args[7], analyzer_);
    });
  }

  // Detect a broadcast-scalar tail mask: valid_col is a compile-time constant 1
  // while physical_col > 1. This happens when a 1D scalar (e.g. a per-channel
  // scale[bn]) is copied into a multi-element UB tile whose downstream op
  // broadcasts it across the full physical width. The valid_col=1 is correct
  // for the *copy* (only one scalar element exists), but rewriting the
  // downstream op to a tail helper with valid_col=1 would compute only one
  // element instead of the full broadcast -- so bail and let the full-tile path
  // (which reads the pad-filled gap) handle it.
  bool IsBroadcastScalarMask(const TailMaskInfo &m) const {
    if (!m.physical_col.defined())
      return false;
    auto *pcol = m.physical_col.as<IntImmNode>();
    if (pcol == nullptr || pcol->value <= 1)
      return false;
    // valid_col may be a Min(...) expr; use the analyzer to prove it equals 1.
    if (!m.valid_col.defined())
      return false;
    if (analyzer_->CanProveEqual(m.valid_col, 1))
      return true;
    return false;
  }

  TailMaskInfo GetMask(const VarNode *v) const {
    if (v == nullptr)
      return TailMaskInfo{};
    auto it = state_.find(v);
    return it == state_.end() ? TailMaskInfo{} : it->second;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call == nullptr)
      return arith::IRMutatorWithAnalyzer::VisitStmt_(op);

    // --- GM->UB copy: seed the destination's valid region. -----------------
    std::string ext = ExternName(call);
    if (ext.find("copy_gm_to_ub") != std::string::npos) {
      HandleGmToUbCopy(call);
      return GetRef<Stmt>(op);
    }
    // --- UB->UB copy: inherit the source's valid region. -------------------
    if (ext.find("copy_ub_to_ub") != std::string::npos) {
      if (call->args.size() >= 3) {
        const VarNode *src_v = GetPtrVar(call->args[1]);
        const VarNode *dst_v = GetPtrVar(call->args[2]);
        if (dst_v != nullptr)
          state_[dst_v] = GetMask(src_v);
      }
      return GetRef<Stmt>(op);
    }
    // copy_ub_to_gm and other copies are sinks: nothing to propagate.
    if (ext.find("copy_") != std::string::npos)
      return GetRef<Stmt>(op);

    // --- Vector ops. -------------------------------------------------------
    if (Stmt rewritten = TryRewriteVectorOp(call); rewritten.defined())
      return rewritten;

    return GetRef<Stmt>(op);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    all_loop_vars_.insert(op->loop_var.get());
    active_loop_vars_.insert(op->loop_var.get());
    Stmt s = arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    active_loop_vars_.erase(op->loop_var.get());
    return s;
  }

  void HandleGmToUbCopy(const CallNode *call) {
    // args: name(0) src_ptr(1) dst_ptr(2) strideN(3) validRow(4) validCol(5)
    //       pad_val(6) [physRow(7) physCol(8)]   (physRow omitted for 1D tiles;
    //       pad_val is always present in the hybrid scheme)
    if (call->args.size() < 7)
      return;
    const VarNode *dst_v = GetPtrVar(call->args[2]);
    if (dst_v == nullptr)
      return;
    PrimExpr valid_row = call->args[4];
    PrimExpr valid_col = call->args[5];
    PrimExpr phys_row, phys_col;
    if (call->args.size() >= 9) {
      phys_row = call->args[7];
      phys_col = call->args[8];
    } else if (call->args.size() == 8) {
      phys_row = IntImm(DataType::Int(32), 1);
      phys_col = call->args[7];
    } else {
      return;
    }
    state_[dst_v] =
        MakeCopyMask(valid_row, valid_col, phys_row, phys_col, analyzer_);
  }

  // Returns a rewritten Stmt, or an undefined Stmt to keep the original.
  Stmt TryRewriteVectorOp(const CallNode *call) {
    // Binary: dst(0) src0(1) src1(2) count(3)
    if (std::string tag = BinaryTag(call); !tag.empty())
      return RewriteBinary(call, tag);
    // Unary: dst(0) src(1) count(2)
    if (std::string tag = UnaryTag(call); !tag.empty())
      return RewriteUnary(call, tag);
    // Scalar (immediate): dst(0) src(1) scalar(2) count(3)
    if (std::string tag = ScalarTag(call); !tag.empty())
      return RewriteScalar(call, tag);
    // Reduce: name(0) out(1) src(2) tmp(3) clear(4)
    if (call->op.same_as(ascend_reduce()))
      return RewriteReduce(call);
    // Compare: tensor/tensor or immediate scalar -> packed uint8 mask.
    if (call->op.same_as(ascend_compare()))
      return RewriteCompare(call, false);
    if (call->op.same_as(ascend_compare_scalar()))
      return RewriteCompare(call, true);
    // Select accepts only a packed mask produced by a tracked compare.
    if (call->op.same_as(ascend_select()))
      return RewriteSelect(call);
    // Cast: dst(0) src(1) roundmode(2) count(3) -- propagate only.
    if (call->op.same_as(ascend_cast())) {
      PropagateUnaryShape(call->args[0], call->args[1]);
      return Stmt();
    }
    // Broadcast: name(0) dst(1) src(2) tmp(3) dim(4) dstShape... srcShape...
    if (call->op.same_as(ascend_broadcast())) {
      return RewriteBroadcast(call);
    }
    return Stmt();
  }

  // dst inherits src's rectangle (used for cast / unrewritten unary shapes).
  void PropagateUnaryShape(const PrimExpr &dst_ptr, const PrimExpr &src_ptr) {
    const VarNode *dst_v = GetPtrVar(dst_ptr);
    if (dst_v != nullptr)
      state_[dst_v] = GetMask(GetPtrVar(src_ptr));
  }

  Stmt RewriteUnary(const CallNode *call, const std::string &tag) {
    if (call->args.size() < 3)
      return Stmt();
    const VarNode *dst_v = GetPtrVar(call->args[0]);
    TailMaskInfo in = GetMask(GetPtrVar(call->args[1]));
    bool ok = CleanTail(call->args[2], in) &&
              SupportedTailDtype(PtrDtype(call->args[0])) &&
              !HasOutOfScopeLoopVar(in.valid_row) &&
              !HasOutOfScopeLoopVar(in.valid_col) && !IsBroadcastScalarMask(in);
    if (dst_v != nullptr)
      state_[dst_v] = ok ? in : TailMaskInfo{};
    if (!ok)
      return Stmt();
    Array<PrimExpr> a = {StringImm(tag), call->args[0], call->args[1],
                         in.valid_row,   in.valid_col,  in.physical_col};
    return Evaluate(Call(DataType::Handle(), ascend_tail_unary(), a));
  }

  Stmt RewriteBinary(const CallNode *call, const std::string &tag) {
    if (call->args.size() < 4)
      return Stmt();
    const VarNode *dst_v = GetPtrVar(call->args[0]);
    TailMaskInfo lhs = GetMask(GetPtrVar(call->args[1]));
    TailMaskInfo rhs = GetMask(GetPtrVar(call->args[2]));
    TailMaskInfo out = IntersectMasks(lhs, rhs, analyzer_);
    bool ok = CleanTail(call->args[3], out) &&
              SupportedTailDtype(PtrDtype(call->args[0])) &&
              !HasOutOfScopeLoopVar(out.valid_row) &&
              !HasOutOfScopeLoopVar(out.valid_col) &&
              !IsBroadcastScalarMask(out);
    if (dst_v != nullptr)
      state_[dst_v] = ok ? out : TailMaskInfo{};
    if (!ok)
      return Stmt();
    Array<PrimExpr> a = {StringImm(tag),  call->args[0], call->args[1],
                         call->args[2],   out.valid_row, out.valid_col,
                         out.physical_col};
    return Evaluate(Call(DataType::Handle(), ascend_tail_binary(), a));
  }

  Stmt RewriteScalar(const CallNode *call, const std::string &tag) {
    if (call->args.size() < 4)
      return Stmt();
    // Only the immediate-scalar form (args[2] is a scalar expr, not a pointer)
    // is rewritten; the "load scalar from buffer" form keeps the full path.
    if (GetPtrVar(call->args[2]) != nullptr)
      return Stmt();
    const VarNode *dst_v = GetPtrVar(call->args[0]);
    TailMaskInfo in = GetMask(GetPtrVar(call->args[1]));
    bool ok = CleanTail(call->args[3], in) &&
              SupportedTailDtype(PtrDtype(call->args[0])) &&
              !HasOutOfScopeLoopVar(in.valid_row) &&
              !HasOutOfScopeLoopVar(in.valid_col) && !IsBroadcastScalarMask(in);
    if (dst_v != nullptr)
      state_[dst_v] = ok ? in : TailMaskInfo{};
    if (!ok)
      return Stmt();
    Array<PrimExpr> a = {StringImm(tag), call->args[0], call->args[1],
                         call->args[2],  in.valid_row,  in.valid_col,
                         in.physical_col};
    return Evaluate(Call(DataType::Handle(), ascend_tail_scalar(), a));
  }

  Stmt RewriteCompare(const CallNode *call, bool scalar) {
    const VarNode *dst_v =
        call->args.empty() ? nullptr : GetPtrVar(call->args[0]);
    if (call->args.size() != 5) {
      // Unsupported forms (notably BufferLoad scalar compare) overwrite dst
      // through the native path. Drop packed provenance left by an earlier
      // compare on the same UB buffer before a later select can consume it.
      if (dst_v != nullptr)
        state_[dst_v] = TailMaskInfo{};
      return Stmt();
    }
    TailMaskInfo lhs = GetMask(GetPtrVar(call->args[1]));
    TailMaskInfo data = lhs;
    if (!scalar) {
      TailMaskInfo rhs = GetMask(GetPtrVar(call->args[2]));
      data = IntersectMasks(lhs, rhs, analyzer_);
      if (!SamePhysicalShape(lhs, rhs)) {
        if (dst_v != nullptr)
          state_[dst_v] = TailMaskInfo{};
        return Stmt();
      }
    } else if (GetPtrVar(call->args[2]) != nullptr) {
      if (dst_v != nullptr)
        state_[dst_v] = TailMaskInfo{};
      return Stmt();
    }
    if (!data.valid_row.defined() || !data.valid_col.defined() ||
        !data.physical_row.defined() || !data.physical_col.defined()) {
      if (dst_v != nullptr)
        state_[dst_v] = TailMaskInfo{};
      return Stmt();
    }

    DataType src_dtype = PtrDtype(call->args[1]);
    PrimExpr storage_col = PackedStorageCol(call->args[0], data);
    PrimExpr packed_min = indexdiv(data.physical_col + 7, 8);
    PrimExpr dst_extent = PtrExtent(call->args[0]);
    bool ok =
        CleanTail(call->args[4], data) && SupportedCmpSelDtype(src_dtype) &&
        PtrDtype(call->args[scalar ? 1 : 2]) == src_dtype &&
        SupportedPackedDtype(PtrDtype(call->args[0])) &&
        CompareWidthSupported(data, src_dtype) && storage_col.defined() &&
        dst_extent.defined() &&
        analyzer_->CanProveEqual(dst_extent, data.physical_row * storage_col) &&
        analyzer_->CanProve(storage_col >= packed_min) &&
        call->args[3].as<StringImmNode>() != nullptr &&
        !HasOutOfScopeLoopVar(data.valid_row) &&
        !HasOutOfScopeLoopVar(data.valid_col) && !IsBroadcastScalarMask(data);
    if (dst_v != nullptr)
      state_[dst_v] =
          ok ? MakePackedCmpMask(data, storage_col) : TailMaskInfo{};
    if (!ok)
      return Stmt();

    Array<PrimExpr> a = {call->args[0], call->args[1], call->args[2]};
    a.push_back(call->args[3]);
    a.push_back(data.valid_row);
    a.push_back(data.valid_col);
    a.push_back(data.physical_row);
    a.push_back(data.physical_col);
    a.push_back(storage_col);
    return Evaluate(
        Call(DataType::Handle(),
             scalar ? ascend_tail_compare_scalar() : ascend_tail_compare(), a));
  }

  Stmt RewriteSelect(const CallNode *call) {
    const VarNode *dst_v =
        call->args.empty() ? nullptr : GetPtrVar(call->args[0]);
    if (call->args.size() < 7) {
      if (dst_v != nullptr)
        state_[dst_v] = TailMaskInfo{};
      return Stmt();
    }
    TailMaskInfo packed = GetMask(GetPtrVar(call->args[1]));
    int type_idx = call->args[3].as<IntImmNode>() != nullptr ? 3 : 4;
    if (type_idx >= static_cast<int>(call->args.size())) {
      if (dst_v != nullptr)
        state_[dst_v] = TailMaskInfo{};
      return Stmt();
    }
    const auto *type_imm = call->args[type_idx].as<IntImmNode>();
    if (!packed.is_packed_cmp() || type_imm == nullptr ||
        (type_imm->value != 1 && type_imm->value != 2)) {
      if (dst_v != nullptr)
        state_[dst_v] = TailMaskInfo{};
      return Stmt();
    }

    int src1_idx = type_idx + 1;
    int mode_idx = type_idx + 2;
    int size_idx = type_idx + 3;
    if (size_idx >= static_cast<int>(call->args.size())) {
      if (dst_v != nullptr)
        state_[dst_v] = TailMaskInfo{};
      return Stmt();
    }
    TailMaskInfo out =
        IntersectDataMask(packed, GetMask(GetPtrVar(call->args[2])));
    if (type_imm->value == 2) {
      TailMaskInfo src1 = GetMask(GetPtrVar(call->args[src1_idx]));
      if (!SamePhysicalShape(out, src1)) {
        if (dst_v != nullptr)
          state_[dst_v] = TailMaskInfo{};
        return Stmt();
      }
      out = IntersectDataMask(out, src1);
    }

    DataType data_dtype = PtrDtype(call->args[0]);
    const auto *mode_imm = call->args[mode_idx].as<StringImmNode>();
    bool supported_mode = mode_imm != nullptr &&
                          ((type_imm->value == 1 &&
                            mode_imm->value == "VSEL_TENSOR_SCALAR_MODE") ||
                           (type_imm->value == 2 &&
                            mode_imm->value == "VSEL_TENSOR_TENSOR_MODE"));
    bool ok =
        CleanTail(call->args[size_idx], out) &&
        SupportedCmpSelDtype(data_dtype) &&
        PtrDtype(call->args[2]) == data_dtype &&
        (type_imm->value == 1 ||
         PtrDtype(call->args[src1_idx]) == data_dtype) &&
        (type_imm->value == 2 || GetPtrVar(call->args[src1_idx]) == nullptr) &&
        SupportedPackedDtype(PtrDtype(call->args[1])) &&
        CompareWidthSupported(out, data_dtype) && supported_mode &&
        !HasOutOfScopeLoopVar(out.valid_row) &&
        !HasOutOfScopeLoopVar(out.valid_col);
    if (dst_v != nullptr)
      state_[dst_v] = ok ? out : TailMaskInfo{};
    if (!ok)
      return Stmt();

    // Normalize the backend-specific tmp insertion. AscendC does not consume
    // a select tmp, so its mask pointer is a harmless placeholder at arg 4.
    PrimExpr tmp = type_idx == 4 ? call->args[3] : call->args[1];
    Array<PrimExpr> a = {StringImm(type_imm->value == 1 ? "Scalar" : "Tensor"),
                         call->args[0],
                         call->args[1],
                         call->args[2],
                         tmp,
                         call->args[type_idx],
                         call->args[src1_idx],
                         call->args[mode_idx],
                         out.valid_row,
                         out.valid_col,
                         out.physical_row,
                         out.physical_col,
                         packed.storage_col};
    return Evaluate(Call(DataType::Handle(), ascend_tail_select(), a));
  }

  Stmt RewriteReduce(const CallNode *call) {
    // name(0) out(1) src(2) [tmp(3)] clear(3/4)
    if (call->args.size() < 4)
      return Stmt();
    const auto *name = call->args[0].as<StringImmNode>();
    if (name == nullptr)
      return Stmt();
    std::string reduce_tag = name->value; // e.g. reduce_sum<...>
    std::string kind = reduce_tag.substr(0, reduce_tag.find('<')); // reduce_sum
    int raw_dim = ParseReduceDim(reduce_tag);
    const VarNode *out_v = GetPtrVar(call->args[1]);
    TailMaskInfo in = GetMask(GetPtrVar(call->args[2]));

    // Reduce is rewritten to a valid-region tail_reduce (which needs no pad)
    // only for a clean 2D float tile on a backend that supports the internal
    // op. Unsupported contracts stay on the native reduce over the pad-filled
    // tile.
    // The valid_row/valid_col must also not reference loop vars that have
    // already gone out of scope (e.g. a copy seeded inside `for by` whose
    // valid_col = f(by), but the reduce runs outside the loop).
    // Enable the contracts whose output layout is explicit and validated:
    // float32 sum/max/min, clear=true, reducing rows (axis 0/-2) of a 2D tile.
    // The scalar last-axis fallback is not device-reliable for a full 32-row
    // tile, so keep that contract on the established native path.
    // Accumulating reductions and lower-precision accumulation keep using the
    // established full-tile + pad path until their backend semantics are
    // validated independently.
    DataType src_dtype = PtrDtype(call->args[2]);
    PrimExpr out_extent = PtrExtent(call->args[1]);
    bool supported_kind =
        kind == "reduce_sum" || kind == "reduce_max" || kind == "reduce_min";
    bool supported_dim = raw_dim == 0 || raw_dim == -2;
    const bool has_tmp =
        call->args.size() == 5 && GetPtrVar(call->args[3]) != nullptr;
    const size_t clear_index = has_tmp ? 4 : 3;
    PrimExpr expected_out_extent = in.physical_col;
    bool supported_contract =
        call->args.size() == clear_index + 1 && supported_kind &&
        supported_dim && is_one(call->args[clear_index]) &&
        src_dtype == DataType::Float(32) &&
        PtrDtype(call->args[1]) == src_dtype && out_extent.defined() &&
        expected_out_extent.defined() &&
        analyzer_->CanProveEqual(out_extent, expected_out_extent) &&
        ReduceShapeMatchesPhysical(reduce_tag, in);
    bool ok = rewrite_reduce_ && supported_contract &&
              CleanTail(PtrExtent(call->args[2]), in) &&
              SupportedTailDtype(PtrDtype(call->args[2])) &&
              !HasOutOfScopeLoopVar(in.valid_row) &&
              !HasOutOfScopeLoopVar(in.valid_col) && !IsBroadcastScalarMask(in);

    // Output rectangle for downstream propagation (only when rewriting).
    TailMaskInfo out;
    if (ok) {
      PrimExpr one = IntImm(DataType::Int(32), 1);
      out = MakeCopyMask(one, in.valid_col, one, in.physical_col, analyzer_);
    }
    if (out_v != nullptr)
      state_[out_v] = out;

    if (!ok)
      return Stmt();

    // The validated tail helper uses basic vector instructions and consumes no
    // workspace, irrespective of the native reduce layout.
    Array<PrimExpr> a = {StringImm(kind), call->args[1], call->args[2]};
    a.push_back(IntImm(DataType::Int(32), 0));
    a.push_back(in.valid_row);
    a.push_back(in.valid_col);
    a.push_back(in.physical_col);
    a.push_back(call->args[clear_index]);
    return Evaluate(Call(DataType::Handle(), ascend_tail_reduce(), a));
  }

  Stmt RewriteBroadcast(const CallNode *call) {
    // name(0) dst(1) src(2) [tmp(3)] dim(3/4) dstShape... srcShape...
    if (call->args.size() < 2)
      return Stmt();
    const VarNode *dst_v = GetPtrVar(call->args[1]);
    if (dst_v == nullptr)
      return Stmt();
    if (call->args.size() < 4) {
      state_[dst_v] = TailMaskInfo{};
      return Stmt();
    }
    TailMaskInfo in = GetMask(GetPtrVar(call->args[2]));
    const bool has_tmp = GetPtrVar(call->args[3]) != nullptr;
    const size_t dim_index = has_tmp ? 4 : 3;
    if (call->args.size() <= dim_index) {
      state_[dst_v] = TailMaskInfo{};
      return Stmt();
    }
    const auto *dim_imm = call->args[dim_index].as<IntImmNode>();
    if (!in.valid_row.defined() || !in.valid_col.defined() ||
        dim_imm == nullptr || dim_imm->value != 2) {
      state_[dst_v] = TailMaskInfo{};
      return Stmt();
    }
    const size_t shape_index = dim_index + 1;
    if (call->args.size() < shape_index + 4) {
      state_[dst_v] = TailMaskInfo{};
      return Stmt();
    }
    PrimExpr dst_rows = call->args[shape_index];
    PrimExpr dst_cols = call->args[shape_index + 1];
    PrimExpr src_rows = call->args[shape_index + 2];
    PrimExpr src_cols = call->args[shape_index + 3];
    if (!analyzer_->CanProveEqual(in.physical_row, src_rows) ||
        !analyzer_->CanProveEqual(in.physical_col, src_cols)) {
      state_[dst_v] = TailMaskInfo{};
      return Stmt();
    }
    PrimExpr dst_extent = PtrExtent(call->args[1]);
    PrimExpr src_extent = PtrExtent(call->args[2]);
    if (!dst_extent.defined() || !src_extent.defined() ||
        !analyzer_->CanProveEqual(dst_extent, dst_rows * dst_cols) ||
        !analyzer_->CanProveEqual(src_extent, src_rows * src_cols)) {
      state_[dst_v] = TailMaskInfo{};
      return Stmt();
    }
    // Same-shape broadcast is a shape-preserving native copy. It needs no
    // tail helper, but its valid rectangle remains valid for downstream ops.
    if (analyzer_->CanProveEqual(dst_rows, src_rows) &&
        analyzer_->CanProveEqual(dst_cols, src_cols)) {
      state_[dst_v] = in;
      return Stmt();
    }
    // [1, 1] makes both axes look like the broadcast axis. The current tail
    // ABI does not carry the explicit frontend axis, so retain the native op
    // rather than silently choosing axis 1 for an axis-0 broadcast.
    if (is_one(src_rows) && is_one(src_cols)) {
      state_[dst_v] = TailMaskInfo{};
      return Stmt();
    }
    TailMaskInfo out = MakeFullMask(dst_rows, dst_cols);
    auto hint_it = output_hints_.find(dst_v);
    bool has_hint = hint_it != output_hints_.end() &&
                    SamePhysicalShape(hint_it->second, out);
    if (has_hint)
      out = hint_it->second;
    if (is_one(src_cols)) {
      // [M,1] -> [M,N]: row tail carries, all columns become valid.
      out.valid_row =
          has_hint ? Min(out.valid_row, in.valid_row) : in.valid_row;
      if (!has_hint)
        out.valid_col = dst_cols;
    } else if (is_one(src_rows)) {
      // [1,N] -> [M,N]: column tail carries, all rows become valid.
      if (!has_hint)
        out.valid_row = dst_rows;
      out.valid_col =
          has_hint ? Min(out.valid_col, in.valid_col) : in.valid_col;
    } else {
      state_[dst_v] = TailMaskInfo{};
      return Stmt();
    }
    out.kind = IsStaticallyFull(out.valid_row, out.valid_col, dst_rows,
                                dst_cols, analyzer_)
                   ? TailMaskKind::kFull
                   : TailMaskKind::kTail;
    out.physical_row = dst_rows;
    out.physical_col = dst_cols;
    out.storage_col = dst_cols;
    bool ok = out.is_tail() && SupportedCmpSelDtype(PtrDtype(call->args[1])) &&
              PtrDtype(call->args[2]) == PtrDtype(call->args[1]) &&
              !HasOutOfScopeLoopVar(out.valid_row) &&
              !HasOutOfScopeLoopVar(out.valid_col) &&
              !HasOutOfScopeLoopVar(in.valid_row) &&
              !HasOutOfScopeLoopVar(in.valid_col);
    state_[dst_v] = ok ? out : TailMaskInfo{};
    if (!ok)
      return Stmt();
    Array<PrimExpr> a(call->args.begin(), call->args.end());
    a.push_back(out.valid_row);
    a.push_back(out.valid_col);
    a.push_back(in.valid_row);
    a.push_back(in.valid_col);
    return Evaluate(Call(DataType::Handle(), ascend_tail_broadcast(), a));
  }

  static int ParseReduceDim(const std::string &tag) {
    // tag like "reduce_sum<float, 16, 32, -1>"; dim is the fourth template
    // field. Parse it from the known row/column fields instead of assuming it
    // remains the final field if the tag gains more metadata later.
    size_t lt = tag.find('<');
    size_t gt = tag.rfind('>');
    if (lt == std::string::npos || gt == std::string::npos || gt <= lt)
      return 2;
    std::string inner = tag.substr(lt + 1, gt - lt - 1);
    size_t dtype_end = inner.find(',');
    if (dtype_end == std::string::npos)
      return 2;
    size_t row_end = inner.find(',', dtype_end + 1);
    if (row_end == std::string::npos)
      return 2;
    size_t col_end = inner.find(',', row_end + 1);
    if (col_end == std::string::npos)
      return 2;
    size_t dim_end = inner.find(',', col_end + 1);
    std::string dim_str = inner.substr(col_end + 1, dim_end - (col_end + 1));
    try {
      size_t parsed = 0;
      int dim = std::stoi(dim_str, &parsed);
      return dim_str.find_first_not_of(" \t", parsed) == std::string::npos ? dim
                                                                           : 2;
    } catch (...) {
      return 2;
    }
  }

  bool ReduceShapeMatchesPhysical(const std::string &tag,
                                  const TailMaskInfo &in) {
    if (!in.physical_row.defined() || !in.physical_col.defined())
      return false;
    size_t lt = tag.find('<');
    size_t gt = tag.rfind('>');
    if (lt == std::string::npos || gt == std::string::npos || gt <= lt)
      return false;
    std::string inner = tag.substr(lt + 1, gt - lt - 1);
    size_t dtype_end = inner.find(',');
    if (dtype_end == std::string::npos)
      return false;
    size_t row_end = inner.find(',', dtype_end + 1);
    if (row_end == std::string::npos)
      return false;
    size_t col_end = inner.find(',', row_end + 1);
    if (col_end == std::string::npos)
      return false;
    try {
      auto parse_int = [](const std::string &token, int *value) {
        size_t parsed = 0;
        *value = std::stoi(token, &parsed);
        return token.find_first_not_of(" \t", parsed) == std::string::npos;
      };
      int row = 0;
      int col = 0;
      if (!parse_int(inner.substr(dtype_end + 1, row_end - dtype_end - 1),
                     &row) ||
          !parse_int(inner.substr(row_end + 1, col_end - row_end - 1), &col))
        return false;
      return row > 0 && col > 0 &&
             analyzer_->CanProveEqual(in.physical_row, row) &&
             analyzer_->CanProveEqual(in.physical_col, col);
    } catch (...) {
      // Dynamic or otherwise unparseable explicit real_shape stays on the
      // established native reduce path.
      return false;
    }
  }
};

// Opt-in switch for the tail-block valid-region scheme. Must be registered so
// PassContext accepts it as a config option (default off).
static constexpr const char *kAscendTailMask = "tl.ascend_tail_mask";
TVM_REGISTER_PASS_CONFIG_OPTION(kAscendTailMask, Bool);

namespace transform {

using namespace tir::transform;

tvm::transform::Pass AscendTailMaskPropagation(bool rewrite_reduce) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    // Opt-in: the tail-block scheme is off unless TL_ASCEND_TAIL_MASK is set,
    // so non-tail kernels are left untouched.
    bool ascend_tail_mask =
        ctx->GetConfig<Bool>(kAscendTailMask, Bool(false)).value();
    if (!ascend_tail_mask) {
      return f;
    }
    return AscendTailMaskPropagator::Substitute(std::move(f), rewrite_reduce);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendTailMaskPropagation", {});
}

TVM_REGISTER_GLOBAL("tl.transform.AscendTailMaskPropagation")
    .set_body_typed(AscendTailMaskPropagation);
} // namespace transform

} // namespace tl
} // namespace tvm
