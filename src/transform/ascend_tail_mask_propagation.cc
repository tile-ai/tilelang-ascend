// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file src/transform/ascend_tail_mask_propagation.cc
 * \brief Propagate UB tail valid-regions and rewrite vector ops to tail-aware
 *        variants for the AscendC backend.
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
 * Batch 1 rewrites: unary / binary / scalar(immediate) / reduce.
 * Batch 1 propagates-but-does-not-rewrite: cast / broadcast / copy_ub_to_ub
 * (they are per-lane or shape-only, so the full-tile path stays numerically
 * correct; the mask still flows through to a downstream reduce).
 */

#include "arith/ir_mutator_with_analyzer.h"
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <string>
#include <unordered_map>

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
    f.CopyOnWrite()->body = m.VisitStmt(f->body);
    return f;
  }

  AscendTailMaskPropagator(arith::Analyzer *analyzer, bool rewrite_reduce)
      : arith::IRMutatorWithAnalyzer(analyzer),
        rewrite_reduce_(rewrite_reduce) {}

private:
  // Per UB data Var -> current valid region. Absent => full (untracked).
  std::unordered_map<const VarNode *, TailMaskInfo> state_;
  // Whether reduce ops may be rewritten to tail_reduce. Disabled for the PTO
  // backend, whose reduce codegen handles valid shapes natively.
  bool rewrite_reduce_ = true;

  // --- conservative guards -------------------------------------------------
  // Only float-like dtypes have validated tail helpers; int/uint stay on the
  // full-tile path to avoid unsupported AscendC intrinsic instantiations.
  static bool SupportedTailDtype(DataType dt) {
    return dt.is_float() || dt.is_bfloat16();
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
    // Cast: dst(0) src(1) roundmode(2) count(3) -- propagate only.
    if (call->op.same_as(ascend_cast())) {
      PropagateUnaryShape(call->args[0], call->args[1]);
      return Stmt();
    }
    // Broadcast: name(0) dst(1) src(2) tmp(3) dim(4) dstShape... srcShape...
    if (call->op.same_as(ascend_broadcast())) {
      PropagateBroadcast(call);
      return Stmt();
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
              SupportedTailDtype(PtrDtype(call->args[0]));
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
              SupportedTailDtype(PtrDtype(call->args[0]));
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
              SupportedTailDtype(PtrDtype(call->args[0]));
    if (dst_v != nullptr)
      state_[dst_v] = ok ? in : TailMaskInfo{};
    if (!ok)
      return Stmt();
    Array<PrimExpr> a = {StringImm(tag), call->args[0], call->args[1],
                         call->args[2],  in.valid_row,  in.valid_col,
                         in.physical_col};
    return Evaluate(Call(DataType::Handle(), ascend_tail_scalar(), a));
  }

  Stmt RewriteReduce(const CallNode *call) {
    // name(0) out(1) src(2) tmp(3) clear(4)
    if (call->args.size() < 5)
      return Stmt();
    const auto *name = call->args[0].as<StringImmNode>();
    if (name == nullptr)
      return Stmt();
    std::string reduce_tag = name->value; // e.g. reduce_sum<...>
    std::string kind = reduce_tag.substr(0, reduce_tag.find('<')); // reduce_sum
    int raw_dim = ParseReduceDim(reduce_tag);
    // Normalize to 0 (reduce rows) or -1 (reduce last axis). For 2D tiles
    // axis 0/-2 reduce rows and axis 1/-1 reduce columns.
    int dim = (raw_dim == 0 || raw_dim == -2) ? 0 : -1;

    const VarNode *out_v = GetPtrVar(call->args[1]);
    TailMaskInfo in = GetMask(GetPtrVar(call->args[2]));

    // Reduce is rewritten to a valid-region tail_reduce (which needs no pad)
    // only on the AscendC backend, for a clean 2D float tile. On PTO, or for
    // 3D / int tiles, it stays the native reduce over the pad-filled tile.
    bool ok = rewrite_reduce_ && CleanTail(PtrExtent(call->args[2]), in) &&
              SupportedTailDtype(PtrDtype(call->args[2]));

    // Output rectangle for downstream propagation (only when rewriting).
    TailMaskInfo out;
    if (ok) {
      out.kind = TailMaskKind::kTail;
      if (dim == 0) {
        out.valid_row = IntImm(DataType::Int(32), 1);
        out.valid_col = in.valid_col;
        out.physical_row = IntImm(DataType::Int(32), 1);
        out.physical_col = in.physical_col;
      } else { // dim == -1 (reduce last axis) -> column vector
        out.valid_row = in.valid_row;
        out.valid_col = IntImm(DataType::Int(32), 1);
        out.physical_row = in.physical_row;
        out.physical_col = IntImm(DataType::Int(32), 1);
      }
      out.storage_col = out.physical_col;
    }
    if (out_v != nullptr)
      state_[out_v] = out;

    if (!ok)
      return Stmt();

    // tail_reduce: kind(0) out(1) src(2) tmp(3) dim(4)
    //              valid_row(5) valid_col(6) phys_col(7) clear(8)
    Array<PrimExpr> a = {StringImm(kind),
                         call->args[1],
                         call->args[2],
                         call->args[3],
                         IntImm(DataType::Int(32), dim),
                         in.valid_row,
                         in.valid_col,
                         in.physical_col,
                         call->args[4]};
    return Evaluate(Call(DataType::Handle(), ascend_tail_reduce(), a));
  }

  void PropagateBroadcast(const CallNode *call) {
    // name(0) dst(1) src(2) tmp(3) dim(4) dstShape[dim] srcShape[dim]
    if (call->args.size() < 5)
      return;
    const VarNode *dst_v = GetPtrVar(call->args[1]);
    if (dst_v == nullptr)
      return;
    TailMaskInfo in = GetMask(GetPtrVar(call->args[2]));
    const auto *dim_imm = call->args[4].as<IntImmNode>();
    if (!in.is_tail() || dim_imm == nullptr || dim_imm->value != 2) {
      state_[dst_v] = in; // 1D / untracked: pass through
      return;
    }
    // 2D broadcast. dstShape = args[5],args[6]; srcShape = args[7],args[8].
    if (call->args.size() < 9) {
      state_[dst_v] = in;
      return;
    }
    PrimExpr dst_rows = call->args[5];
    PrimExpr dst_cols = call->args[6];
    PrimExpr src_rows = call->args[7];
    PrimExpr src_cols = call->args[8];
    TailMaskInfo out;
    out.kind = TailMaskKind::kTail;
    out.physical_row = dst_rows;
    out.physical_col = dst_cols;
    out.storage_col = dst_cols;
    if (is_one(src_cols)) {
      // [M,1] -> [M,N]: row tail carries, all columns become valid.
      out.valid_row = in.valid_row;
      out.valid_col = dst_cols;
    } else if (is_one(src_rows)) {
      // [1,N] -> [M,N]: column tail carries, all rows become valid.
      out.valid_row = dst_rows;
      out.valid_col = in.valid_col;
    } else {
      out = in;
    }
    state_[dst_v] = out;
  }

  static int ParseReduceDim(const std::string &tag) {
    // tag like "reduce_sum<float, 16, 32, -1>"; dim is the last template field.
    size_t lt = tag.find('<');
    size_t gt = tag.rfind('>');
    if (lt == std::string::npos || gt == std::string::npos || gt <= lt)
      return -1;
    std::string inner = tag.substr(lt + 1, gt - lt - 1);
    size_t comma = inner.rfind(',');
    std::string dim_str =
        comma == std::string::npos ? inner : inner.substr(comma + 1);
    try {
      return std::stoi(dim_str);
    } catch (...) {
      return -1;
    }
  }
};

namespace transform {

using namespace tir::transform;

tvm::transform::Pass AscendTailMaskPropagation(bool rewrite_reduce) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return AscendTailMaskPropagator::Substitute(std::move(f), rewrite_reduce);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendTailMaskPropagation", {});
}

TVM_REGISTER_GLOBAL("tl.transform.AscendTailMaskPropagation")
    .set_body_typed(AscendTailMaskPropagation);
} // namespace transform

} // namespace tl
} // namespace tvm
