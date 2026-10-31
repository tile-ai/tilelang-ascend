/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file flatten_buffer.cc
 * Source: TileLang-Ascend 0deb8b93, src/transform/flatten_buffer.cc.
 */

#include "arith/ir_mutator_with_analyzer.h"
#include "tirx/ir/data_type_rewriter.h"
#include "tirx/transform/ir_utils.h"
#include <tvm/arith/iter_affine_map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "op/builtin.h"

namespace tvm {
namespace tl {
namespace a2a3 {

using namespace tirx;
using namespace ffi;

/*!
 * \brief Transform multi-dimension BufferLoad/BufferStore into device-supported
 * dimension for the TIR not contains opaque block.
 */
class BufferFlattener : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Flatten(PrimFunc func) {
    arith::Analyzer ana;
    auto pass = BufferFlattener(&ana);
    if (auto init_map =
            func->attrs.GetAttr<Map<Var, PrimExpr>>(tl::attr::kLocalVarInit)) {
      pass.local_var_init_map_ = init_map.value();
    }
    auto writer = func.CopyOnWrite();
    pass.MarkBufferMapShapes(func);
    writer->body = pass.VisitStmt(func->body);
    // The buffers in func->buffer_map are deliberately left
    // unflattened, as they are used for validation of user-provided
    // arguments.  The flattened buffers used in the updated
    // function body alias the argument buffers.
    return func;
  }

private:
  using IRMutatorWithAnalyzer::VisitExpr;
  using IRMutatorWithAnalyzer::VisitExpr_;
  using IRMutatorWithAnalyzer::VisitStmt;
  using IRMutatorWithAnalyzer::VisitStmt_;

  class Int64Promoter : public tirx::IndexDataTypeRewriter {
  public:
    using Parent = IndexDataTypeRewriter;

    PrimExpr VisitExpr_(const VarNode *op) final {
      if (op->dtype.is_int() && op->dtype.bits() < 64) {
        return cast(DataType::Int(64), GetRef<Var>(op));
      }
      return GetRef<PrimExpr>(op);
    }

    PrimExpr VisitExpr_(const IntImmNode *op) final {
      if (op->dtype.is_int() && op->dtype.bits() < 64) {
        return IntImm(DataType::Int(64), op->value);
      }
      return GetRef<PrimExpr>(op);
    }

    PrimExpr VisitExpr_(const CastNode *op) final {
      if (op->dtype.is_int() && op->dtype.bits() < 64) {
        return cast(DataType::Int(64), op->value);
      }
      return GetRef<PrimExpr>(op);
    }

    Stmt VisitStmt_(const BufferStoreNode *op) final {
      // Force indices to be int64
      auto node = Downcast<BufferStore>(Parent::VisitStmt_(op));
      return std::move(node);
    }

    PrimExpr VisitExpr_(const BufferLoadNode *op) final {
      auto node = Downcast<BufferLoad>(Parent::VisitExpr_(op));
      return std::move(node);
    }
  };

  explicit BufferFlattener(arith::Analyzer *ana) : IRMutatorWithAnalyzer(ana) {}

  Stmt VisitStmt_(const SBlockNode *op) final {
    ICHECK_EQ(op->match_buffers.size(), 0)
        << "Unexpected MatchBufferRegion found during "
           "tir.transform.FlattenBuffer.  "
        << "All MatchBufferRegion should be removed in "
           "tir.transform.LowerMatchBuffer.";

    SBlock block = GetRef<SBlock>(op);

    Array<Buffer> alloc_buffers = op->alloc_buffers;
    alloc_buffers.MutateByApply(
        [this](Buffer buf) { return GetFlattenedBuffer(buf); });
    if (!alloc_buffers.same_as(op->alloc_buffers)) {
      block.CopyOnWrite()->alloc_buffers = alloc_buffers;
    }

    Array<BufferRegion> reads = op->reads;
    reads.MutateByApply(
        [this](BufferRegion region) { return MutateBufferRegion(region); });
    if (!reads.same_as(op->reads)) {
      block.CopyOnWrite()->reads = reads;
    }

    Array<BufferRegion> writes = op->writes;
    writes.MutateByApply(
        [this](BufferRegion region) { return MutateBufferRegion(region); });
    if (!writes.same_as(op->writes)) {
      block.CopyOnWrite()->writes = writes;
    }

    return StmtExprMutator::VisitStmt_(block.get());
  }

  // TIRX makes allocation and declaration flat statements.  Preserve TA's
  // immediate Allocate/DeclBuffer matching before visiting either statement.
  Stmt VisitStmt_(const SeqStmtNode *op) final {
    for (size_t i = 0; i + 1 < op->seq.size(); ++i) {
      const auto *alloc = op->seq[i].as<AllocBufferNode>();
      if (!alloc)
        continue;
      Stmt next = op->seq[i + 1];
      while (const auto *seq = next.as<SeqStmtNode>()) {
        if (seq->seq.empty())
          break;
        next = seq->seq[0];
      }
      if (const auto *decl = next.as<DeclBufferNode>()) {
        allocation_views_.emplace(GetRef<AllocBuffer>(alloc), decl->buffer);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocBufferNode *op) final {
    const Buffer &allocation = op->buffer;
    Array<PrimExpr> new_extents;
    auto view = allocation_views_.find(GetRef<AllocBuffer>(op));
    if (allocation->shape.size() == 1) {
      new_extents = allocation->shape;
    } else if (view != allocation_views_.end()) {
      const Buffer &buffer = view->second;
      bool matching_buffer = allocation->data.same_as(buffer->data) &&
                             allocation->dtype == buffer->dtype &&
                             allocation->shape.size() == buffer->shape.size();
      ExprDeepEqual expr_equal;
      for (size_t i = 0; matching_buffer && i < allocation->shape.size(); ++i) {
        matching_buffer = expr_equal(allocation->shape[i], buffer->shape[i]);
      }
      if (matching_buffer) {
        new_extents = GetFlattenedBuffer(buffer)->shape;
      } else {
        ICHECK(buffer->axis_separators.empty())
            << "DeclBuffer node doesn't match Allocate extents, but also "
               "shouldn't be flattened to 1-d physical memory";
      }
    } else {
      // AllocBuffer itself declares its logical view in TIRX.
      new_extents = GetFlattenedBuffer(allocation)->shape;
    }
    if (new_extents.empty()) {
      PrimExpr flat_extent = 1;
      for (const auto &dim : allocation->shape)
        flat_extent *= dim;
      new_extents = {flat_extent};
    }
    AllocBuffer alloc = Downcast<AllocBuffer>(StmtExprMutator::VisitStmt_(op));
    Buffer flat_buffer = GetFlattenedBuffer(allocation);
    flat_buffer.CopyOnWrite()->shape = new_extents;
    alloc.CopyOnWrite()->buffer = flat_buffer;
    if (!local_var_init_map_.empty()) {
      auto init_it = local_var_init_map_.find(allocation->data);
      if (init_it != local_var_init_map_.end()) {
        alloc.CopyOnWrite()->annotations.Set(tl::attr::kLocalVarInit,
                                             (*init_it).second);
      }
    }
    return alloc;
  }

  Stmt VisitStmt_(const DeclBufferNode *op) final {
    // TA strips the declaration wrapper; TIRX retains the declaration because
    // it carries lexical Buffer identity for the remaining flat statements.
    return DeclBuffer(GetFlattenedBuffer(op->buffer));
  }

  Buffer GetFlattenedBuffer(Buffer buf) {
    auto it = buffer_remap_.find(buf);
    if (it != buffer_remap_.end()) {
      return it->second;
    }
    auto flattened = buf.GetFlattenedBuffer();
    auto writer = flattened.CopyOnWrite();

    // TODO(Lunderberg): Move the handling of boolean into a
    // dedicated pass.
    if (flattened->dtype == DataType::Bool()) {
      writer->dtype = DataType::Int(8);
    }
    // canonicalize shape
    for (size_t i = 0; i < flattened->shape.size(); ++i) {
      writer->shape.Set(i, analyzer_->canonical_simplify(flattened->shape[i]));
    }

    buffer_remap_[buf] = flattened;
    return flattened;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    bool store_returns_bool = (op->value.dtype() == DataType::Bool());
    store = VisitBufferAccess(store);

    // Handle casts from the value's dtype to the dtype of the
    // backing array.
    // TODO(Lunderberg): Move the handling of boolean into a
    // dedicated pass.
    if (store_returns_bool) {
      ICHECK_EQ(store->buffer->dtype, DataType::Int(8))
          << "Expected int8 backing array for boolean tensor";
      auto writer = store.CopyOnWrite();
      writer->value = tvm::cast(DataType::Int(8), store->value);
      return std::move(store);
    }
    return std::move(store);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    bool load_returns_bool = (op->dtype == DataType::Bool());
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    load = VisitBufferAccess(load);
    // Handle casts from dtype of the backing array to value's dtype.
    // TODO(Lunderberg): Move the handling of boolean into a
    // dedicated pass.
    if (load_returns_bool && !under_address_of) {
      ICHECK_EQ(load->buffer->dtype, DataType::Int(8))
          << "Expected int8 backing array for boolean tensor";
      load.CopyOnWrite()->dtype = DataType::Int(8);
      return tvm::cast(DataType::Bool(), load);
    } else {
      return std::move(load);
    }
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::address_of())) {
      under_address_of = true;
      auto result = StmtExprMutator::VisitExpr_(op);
      under_address_of = false;
      return result;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Array<PrimExpr> GetSimplifiedElemOffset(const Buffer &buffer,
                                          const Array<PrimExpr> &indices) {
    auto flattened_indices = buffer->ElemOffset(indices);
    Array<PrimExpr> safe_indices;
    for (auto index : flattened_indices) {
      auto int_bound = analyzer_->const_int_bound(index);
      DataType dtype = index->dtype;
      if (dtype.is_int() && dtype.bits() < 64) {
        int64_t max_value = int_bound->max_value;
        int64_t min_value = int_bound->min_value;
        const int64_t type_max = (1LL << (dtype.bits() - 1));
        const int64_t type_min = -(1LL << (dtype.bits() - 1));

        if (max_value >= (type_max - 1) || min_value < type_min) {
          Int64Promoter promoter;
          for (auto &index : flattened_indices) {
            safe_indices.push_back(promoter(index));
          }
        } else {
          safe_indices.push_back(index);
        }
      } else {
        safe_indices.push_back(index);
      }
    }
    return this->IterMapSimplifyWithContext(safe_indices, false);
  }

  template <typename Node> Node VisitBufferAccess(Node node) {
    ICHECK(node->buffer.defined());
    auto flattened_indices =
        GetSimplifiedElemOffset(node->buffer, node->indices);
    Buffer flattened_buffer = GetFlattenedBuffer(node->buffer);

    auto writer = node.CopyOnWrite();
    writer->buffer = flattened_buffer;
    writer->indices = flattened_indices;
    return node;
  }

  BufferRegion MutateBufferRegion(BufferRegion region) {
    Buffer orig_buf = region->buffer;
    Buffer flattened_buf = GetFlattenedBuffer(orig_buf);
    if (flattened_buf.same_as(orig_buf)) {
      return region;
    }

    Array<PrimExpr> min_values;
    Array<PrimExpr> max_values;
    for (const auto &range : region->region) {
      min_values.push_back(range->min);
      max_values.push_back(range->min + range->extent - 1);
    }

    Array<PrimExpr> flattened_min =
        GetSimplifiedElemOffset(orig_buf, min_values);
    Array<PrimExpr> flattened_max =
        GetSimplifiedElemOffset(orig_buf, max_values);

    Array<Range> flattened_ranges;
    ICHECK_EQ(flattened_min.size(), flattened_max.size());
    for (size_t i = 0; i < flattened_min.size(); i++) {
      flattened_ranges.push_back(Range(flattened_min[i], flattened_max[i] + 1));
    }

    return BufferRegion(flattened_buf, flattened_ranges);
  }

  /*! \brief Whether the current buffer is under address_of */
  bool under_address_of = false;
  std::unordered_map<AllocBuffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      allocation_views_;
  /*! \brief Map of buffers being remapped. */
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      buffer_remap_;

  /*! \brief The updated external buffer map. */
  Map<Var, Buffer> updated_extern_buffer_map_;

  /*! \brief Local var initializers preserved from block annotations. */
  Map<Var, PrimExpr> local_var_init_map_;
};

PrimFunc FlattenBufferRewriter(PrimFunc f) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!f->GetAttr<Bool>("from_legacy_te_schedule", Bool(false)).value()) {
    return BufferFlattener::Flatten(f);
  } else {
    return f;
  }
}

using namespace tirx::transform;
tvm::transform::Pass FlattenBuffer() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return FlattenBufferRewriter(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.A2A3FlattenBuffer", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def("tl.transform.A2A3FlattenBuffer",
                                        FlattenBuffer);
}

} // namespace a2a3
} // namespace tl
} // namespace tvm
