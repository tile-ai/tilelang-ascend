// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file ascend_scalar_store_to_dma.cc
 * \brief Rewrite contiguous scalar GM store loops into UB staging + one DMA
 * burst (issue #1304).
 *
 * A scalar GM store (BufferStore lowering to S-pipe SetValue) goes through a
 * write-back cache that MTE3 DMA bypasses. In a multi-block kernel whose row
 * pitch is not a cache-line multiple, adjacent rows owned by different cores
 * share cache lines, and one core's dirty scalar-store line can evict over
 * another core's freshly DMA'd bytes. dcci/MTE3_S only act on the local
 * core's cache, so the cross-core race cannot be fixed by synchronization.
 * DMA-vs-DMA writes, however, go through the same L2 path with byte-granular
 * merging and are coherent. Staging the scalar values in a UB buffer and
 * emitting one ascend_copy burst therefore removes the hazard entirely.
 *
 * Pattern matched (at the entry of LowerAndLegalize, before buffer-scope
 * inference and copy lowering):
 *
 *   for tw in serial(extent):            // constant extent
 *     let ow = <expr>                     // optional let chain
 *     Y[..., ow] = value                  // GM buffer, innermost index
 *                                         // affine in tw with stride 1,
 *                                         // other indices tw-invariant
 *
 * Rewrite:
 *
 *   allocate staging: shared.ub[extent] {
 *     for tw in serial(extent):
 *       let ow = <expr>
 *       staging[tw - min] = value
 *     ascend_copy(region(staging, [0], read, extent),
 *                 region(Y, [..., base], write, 1, ..., 1, extent))
 *   }
 */

#include <tvm/arith/analyzer.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "../op/op.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

namespace {

static constexpr const char *kScalarStoreToDma =
    "tl.ascend_scalar_store_to_dma";
TVM_REGISTER_PASS_CONFIG_OPTION(kScalarStoreToDma, Bool);

class AllocatedVarCollector : public StmtVisitor {
public:
  std::unordered_set<const VarNode *> allocated;
  void VisitStmt_(const AllocateNode *op) final {
    allocated.insert(op->buffer_var.get());
    StmtVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const BlockNode *op) final {
    // T.alloc_buffer declarations live in the Block's alloc_buffers field,
    // not in Allocate statements.
    for (const auto &buf : op->alloc_buffers) {
      allocated.insert(buf->data.get());
    }
    StmtVisitor::VisitStmt_(op);
  }
};

class ScalarStoreToDmaRewriter : public StmtMutator {
public:
  explicit ScalarStoreToDmaRewriter(
      const std::unordered_set<const VarNode *> &allocated_vars)
      : allocated_vars_(allocated_vars) {}

  Stmt VisitStmt_(const ForNode *op) final {
    // Rewrite inner loops first.
    Stmt stmt = StmtMutator::VisitStmt_(op);
    const ForNode *loop = stmt.as<ForNode>();
    if (loop == nullptr) {
      return stmt;
    }
    Stmt rewritten = TryRewriteLoop(loop);
    return rewritten.defined() ? rewritten : stmt;
  }

private:
  // Vars allocated inside the function body (on-chip buffers). A store whose
  // buffer data var is NOT in this set targets a GM parameter buffer.
  const std::unordered_set<const VarNode *> &allocated_vars_;
  int staging_counter_ = 0;

  struct MatchedStore {
    const BufferStoreNode *store;
    // Let chain preserved verbatim in the rewritten loop body.
    std::vector<std::pair<Var, PrimExpr>> lets;
    // Store indices with the let values substituted; the last index is
    // <loop_var> + inner_base (stride 1).
    Array<PrimExpr> substituted_indices;
    PrimExpr inner_base;
  };

  static bool ExprUses(const PrimExpr &expr, const VarNode *var) {
    return UsesVar(expr, [var](const VarNode *v) { return v == var; });
  }

  // Matches idx == loop_var + base (base free of loop_var), stride 1 only.
  static std::optional<PrimExpr> MatchUnitStrideIndex(const PrimExpr &idx,
                                                      const VarNode *loop_var) {
    if (idx->IsInstance<VarNode>()) {
      if (idx.as<VarNode>() == loop_var) {
        return PrimExpr(Integer(0));
      }
      return std::nullopt;
    }
    if (const auto *add = idx.as<AddNode>()) {
      const auto *lhs = add->a.as<VarNode>();
      const auto *rhs = add->b.as<VarNode>();
      if (lhs == loop_var && !ExprUses(add->b, loop_var)) {
        return std::optional<PrimExpr>(add->b);
      }
      if (rhs == loop_var && !ExprUses(add->a, loop_var)) {
        return std::optional<PrimExpr>(add->a);
      }
    }
    return std::nullopt;
  }

  // Walks [Let*] BufferStore and validates the pattern.
  bool MatchLoop(const ForNode *loop, MatchedStore *out) {
    if (loop->kind != ForKind::kSerial) {
      return false;
    }
    const auto *extent = loop->extent.as<IntImmNode>();
    if (extent == nullptr || extent->value <= 0) {
      return false;
    }

    // Peel the let chain.
    std::vector<std::pair<Var, PrimExpr>> lets;
    Stmt body = loop->body;
    while (const auto *let = body.as<LetStmtNode>()) {
      lets.emplace_back(let->var, let->value);
      body = let->body;
    }
    const auto *store = body.as<BufferStoreNode>();
    if (store == nullptr) {
      return false;
    }
    // Only GM parameter buffers (not allocated on-chip) take the rewrite;
    // on-chip stores (UB/L1) have no cross-core write-back-cache hazard.
    if (allocated_vars_.count(store->buffer->data.get()) > 0) {
      return false;
    }
    if (store->indices.empty()) {
      return false;
    }

    // Substitute the let chain into the indices.
    Map<Var, PrimExpr> subst;
    for (const auto &kv : lets) {
      PrimExpr value = kv.second;
      if (!subst.empty()) {
        value = Substitute(value, subst);
      }
      subst.Set(kv.first, value);
    }
    Array<PrimExpr> indices;
    for (const auto &idx : store->indices) {
      indices.push_back(subst.empty() ? idx : Substitute(idx, subst));
    }

    const VarNode *loop_var = loop->loop_var.get();
    // Innermost (last) index must be loop_var + base with stride 1.
    std::optional<PrimExpr> base =
        MatchUnitStrideIndex(indices.back(), loop_var);
    if (!base.has_value()) {
      return false;
    }
    // All other indices must be loop-var free.
    for (size_t i = 0; i + 1 < indices.size(); ++i) {
      if (ExprUses(indices[i], loop_var)) {
        return false;
      }
    }

    out->store = store;
    out->lets = std::move(lets);
    out->substituted_indices = std::move(indices);
    out->inner_base = *base;
    return true;
  }

  Stmt TryRewriteLoop(const ForNode *loop) {
    MatchedStore match;
    if (!MatchLoop(loop, &match)) {
      return Stmt();
    }

    const Buffer &gm_buffer = match.store->buffer;
    const DataType dtype = gm_buffer->dtype;
    const PrimExpr extent = loop->extent;

    // Fresh staging buffer in shared.ub.
    const std::string name =
        "scalar_dma_staging_" + std::to_string(staging_counter_++);
    Var data_var(name, PointerType(PrimType(dtype), "shared.ub"));
    Buffer staging(data_var, dtype, {extent}, {}, PrimExpr(), name, -1, 0,
                   BufferType::kDefault);

    // Rewritten loop body: same let chain, store redirected to the staging
    // buffer at [loop_var - min].
    PrimExpr staging_idx = loop->loop_var;
    if (!is_zero(loop->min)) {
      staging_idx = Sub(loop->loop_var, loop->min);
    }
    Stmt new_store = BufferStore(staging, match.store->value, {staging_idx});
    Stmt new_body = new_store;
    for (auto it = match.lets.rbegin(); it != match.lets.rend(); ++it) {
      new_body = LetStmt(it->first, it->second, new_body);
    }
    For new_loop(loop->loop_var, loop->min, loop->extent, loop->kind, new_body,
                 loop->thread_binding, loop->annotations);

    // ascend_copy(staging -> gm_buffer region).
    Array<PrimExpr> dst_indices = match.substituted_indices;
    dst_indices.Set(dst_indices.size() - 1, match.inner_base);
    Array<PrimExpr> dst_extents;
    for (size_t i = 0; i + 1 < gm_buffer->shape.size(); ++i) {
      dst_extents.push_back(Integer(1));
    }
    dst_extents.push_back(extent);

    auto make_region = [](const Buffer &buffer, const Array<PrimExpr> &indices,
                          int64_t mask, const Array<PrimExpr> &extents) {
      Array<PrimExpr> args = {BufferLoad(buffer, indices),
                              IntImm(DataType::Int(32), mask)};
      for (const auto &e : extents) {
        args.push_back(e);
      }
      return Call(DataType::Handle(), Op::Get("tl.region"), args);
    };

    Array<PrimExpr> copy_args = {
        make_region(staging, {Integer(0)}, 1, {extent}),
        make_region(gm_buffer, dst_indices, 2, dst_extents),
        Bool(false),                  // enable_relu
        Bool(false),                  // transpose
        IntImm(DataType::Int(32), 0), // pad_value
        IntImm(DataType::Int(32), 0), // tmp
    };
    Stmt copy_stmt = Evaluate(
        Call(DataType::Handle(), Op::Get("tl.ascend_copy"), copy_args));

    return Allocate(data_var, dtype, {extent}, const_true(),
                    SeqStmt({new_loop, copy_stmt}));
  }
};

} // namespace

tvm::transform::Pass AscendScalarStoreToDma() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    bool enabled = ctx->GetConfig<Bool>(kScalarStoreToDma, Bool(false)).value();
    if (!enabled) {
      return f;
    }
    // GM buffers are those NOT allocated inside the function body: function
    // parameter buffers live in global memory. This is scope-annotation
    // independent, so the pass can run before AscendInferBufferScope.
    AllocatedVarCollector collector;
    collector(f->body);
    ScalarStoreToDmaRewriter rewriter(collector.allocated);
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = rewriter(fptr->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendScalarStoreToDma", {});
}

TVM_REGISTER_GLOBAL("tl.transform.AscendScalarStoreToDma")
    .set_body_typed(AscendScalarStoreToDma);

} // namespace tl
} // namespace tvm
