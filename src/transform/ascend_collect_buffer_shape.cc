// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file ascend_collect_buffer_shape.cc
 * \brief This file contains two TIR passes for collecting buffer shapes to
 * guide PTO codegen.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "common/attr.h"
#include "common/operation_config.h"

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief A visitor to collect buffer information (shapes, scopes, and data type
 * bits).
 */
class BufferShapeCollector : public StmtExprVisitor {
public:
  explicit BufferShapeCollector(const PrimFunc &func) {
    // gm
    for (const auto &[name, buffer] : func->buffer_map) {
      ProcessAllocation(buffer->data, buffer->shape, buffer->dtype);
    }

    this->VisitStmt(func->body);
  }

  const Map<Var, Array<PrimExpr>> &GetBufferShapes() const {
    return buffer_shapes_;
  }
  const Map<Var, String> &GetBufferScopes() const { return buffer_scopes_; }
  const Map<Var, Integer> &GetBufferBits() const { return buffer_bits_; }

private:
  void VisitStmt_(const BlockNode *op) override {
    // alloc_buffer
    for (const Buffer &buffer : op->alloc_buffers) {
      ProcessAllocation(buffer->data, buffer->shape, buffer->dtype);
    }

    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocateNode *op) override {
    ProcessAllocation(op->buffer_var, op->extents, op->dtype);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocateConstNode *op) override {
    ProcessAllocation(op->buffer_var, op->extents, op->dtype);
    StmtExprVisitor::VisitStmt_(op);
  }

  /*!
   * \brief Helper to register shape, scope, and bit width for a buffer
   * variable.
   * \note This is the single point of data collection.
   */
  void ProcessAllocation(const Var &buffer_var, const Array<PrimExpr> &extents,
                         const DataType &dtype) {
    String scope = "global";
    if (auto ptr_type = buffer_var->type_annotation.as<PointerTypeNode>()) {
      scope = ptr_type->storage_scope;
    }
    buffer_shapes_.Set(buffer_var, extents);
    buffer_scopes_.Set(buffer_var, scope);
    buffer_bits_.Set(buffer_var, Integer(dtype.bits()));
  }

  Map<Var, Array<PrimExpr>> buffer_shapes_;
  Map<Var, String> buffer_scopes_;
  Map<Var, Integer> buffer_bits_;
};

PrimExpr GetInnerDim(const Var &buffer_var,
                     const Array<PrimExpr> &current_shape,
                     const Map<Var, Array<PrimExpr>> &initial_shapes) {
  if (initial_shapes.count(buffer_var)) {
    const Array<PrimExpr> &initial_shape = initial_shapes.at(buffer_var);
    return initial_shape.empty() ? Integer(1) : initial_shape.back();
  } else {
    return current_shape.empty() ? Integer(1) : current_shape.back();
  }
}

Array<PrimExpr> AlignInnerDim(PrimExpr outer_dim, PrimExpr inner_dim,
                              const std::string &scope, int bits,
                              arith::Analyzer *analyzer) {
  PrimExpr M = outer_dim;
  PrimExpr valid_M = outer_dim;
  PrimExpr valid_N = inner_dim;
  PrimExpr N = valid_N;

  // Apply alignment only for the configured scope and when necessary
  if (kScopeForAlignment.count(scope)) {
    int alignment_bits = kScopeForAlignment.at(scope);
    PrimExpr total_inner_bits = analyzer->Simplify(valid_N * bits);
    if (!analyzer->CanProve(truncmod(total_inner_bits, alignment_bits) == 0)) {
      PrimExpr aligned_total_bits = analyzer->Simplify(
          truncdiv(total_inner_bits + alignment_bits - 1, alignment_bits) *
          alignment_bits);
      N = analyzer->Simplify(truncdiv(aligned_total_bits, bits));
    }
  }

  return {M, N, valid_M, valid_N};
}

// Collect buffer shape before the lower_tile_op pass to avoid the buffer shape
// in the layout_map from being flattened to one dimension
tvm::transform::Pass CreateInitialPass() {
  auto pass_func = [=](PrimFunc f, IRModule m,
                       tvm::transform::PassContext ctx) {
    BufferShapeCollector collector(f);

    Map<String, ObjectRef> new_attrs;
    if (f->attrs.defined()) {
      new_attrs = f->attrs->dict;
    }
    new_attrs.Set(kInitialBufferShapes, collector.GetBufferShapes());
    return WithAttrs(f, std::move(new_attrs));
  };
  return tvm::tir::transform::CreatePrimFuncPass(pass_func, 0,
                                                 "tl.BufferShapeCollector", {});
}

// 1. Collect buffer shape again (pipeline might introduce extra dimensions)
// 2. Flatten the dimensions to 2D to meet PTO’s dimensionality requirements
// 3. Alignment
tvm::transform::Pass CreateFlatten2DPass() {
  auto pass_func = [=](PrimFunc f, IRModule m,
                       tvm::transform::PassContext ctx) {
    // 1. Collect current buffer info
    BufferShapeCollector collector(f);

    // 2. Retrieve initial shapes collected by InitialPass
    Map<Var, Array<PrimExpr>> initial_shapes =
        f->GetAttr<Map<Var, Array<PrimExpr>>>(kInitialBufferShapes)
            .value_or({});

    // 3. Merge and compress shapes into the final 2D format
    Map<Var, Array<PrimExpr>> logical_2d_shapes;
    arith::Analyzer analyzer;
    for (const auto &[buffer_var, shape] : collector.GetBufferShapes()) {
      String scope = collector.GetBufferScopes().at(buffer_var);

      if (!kScopesToFlatten.count(scope)) {
        logical_2d_shapes.Set(buffer_var, shape);
        continue;
      }

      PrimExpr total_elements = Integer(1);
      for (const auto &ext : shape) {
        total_elements = analyzer.Simplify(total_elements * ext);
      }

      // The outer dimension is calculated to preserve the total number of
      // elements. This formula correctly handles all cases:
      // - 1D [m] -> [1, m]
      // - 2D [n, m] -> [n, m]
      // - ND [d1, d2, ..., m] -> [d1*d2*..., m]
      PrimExpr inner_dim = GetInnerDim(buffer_var, shape, initial_shapes);

      PrimExpr outer_dim =
          analyzer.Simplify(truncdiv(total_elements, inner_dim));

      // handle DN/ND
      bool is_inner_dim_one = analyzer.CanProve(inner_dim == 1);
      if (scope == "shared" && is_inner_dim_one) {
        logical_2d_shapes.Set(buffer_var, {inner_dim, outer_dim});
      } else {
        logical_2d_shapes.Set(buffer_var, {outer_dim, inner_dim});
      }
    }

    // 4. Alignment
    Map<Var, Array<PrimExpr>> final_layouts;
    for (const auto &[buffer_var, logical_shape] : logical_2d_shapes) {
      String scope = collector.GetBufferScopes().at(buffer_var);
      int bits = collector.GetBufferBits().at(buffer_var)->value;

      if (!kScopesToFlatten.count(scope)) {
        final_layouts.Set(buffer_var, logical_shape);
        continue;
      }

      Array<PrimExpr> aligned_shape = AlignInnerDim(
          logical_shape[0], logical_shape[1], scope, bits, &analyzer);

      final_layouts.Set(buffer_var, aligned_shape);
    }

    // 5. Attach the final logic shapes and remove the intermediate attribute.
    Map<String, ObjectRef> final_dict;
    if (f->attrs.defined()) {
      for (const auto &kv : f->attrs->dict) {
        if (kv.first != kInitialBufferShapes) {
          final_dict.Set(kv.first, kv.second);
        }
      }
    }

    final_dict.Set(kLogicBufferShapes, final_layouts);

    return WithAttrs(f, final_dict);
  };
  return tvm::tir::transform::CreatePrimFuncPass(pass_func, 0,
                                                 "tl.Flatten2DBuffer", {});
}

TVM_REGISTER_GLOBAL("tl.transform.BufferShapeCollector")
    .set_body_typed(CreateInitialPass);

TVM_REGISTER_GLOBAL("tl.transform.Flatten2DBuffer")
    .set_body_typed(CreateFlatten2DPass);
} // namespace tl
} // namespace tvm