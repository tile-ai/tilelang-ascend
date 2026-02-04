// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file ascend_vid_reduction.cc
 * \brief host specialized for Ascend npu
 */

#include "arith/ir_mutator_with_analyzer.h"
#include "tir/analysis/var_use_def_analysis.h"
#include "tir/transforms/ir_utils.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include "../op/builtin.h"
#include "./common/collector.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

static constexpr const char *ascendVidReduction = "tl.ascend_vid_reduction";

TVM_REGISTER_PASS_CONFIG_OPTION(ascendVidReduction, Bool);

class AscendVidReduction : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f, PassContext ctx) {
    arith::Analyzer analyzer;
    AscendVidReduction substituter(&analyzer);
    PrimFuncNode *fptr = f.CopyOnWrite();
    // // TODO:访问buffer是否这样实现？
    // tir::PostOrderVisit(f->body, [&](const ObjectRef& obj) {
    //     if (const auto* realize = obj.as<tir::BlockRealizeNode>()) {
    //         for (auto buf : realize->block->alloc_buffers) {
    //             ModifyBufferShape(buf);
    //         }
    //     }
    // });
    // // TODO:影响及作用？
    bool ascend_vid_reduction = ctx->GetConfig<Bool>(ascendVidReduction, Bool(false)).value();
    if (!ascend_vid_reduction) {
      return f;
    }
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map_;

  bool IsUbBuffer(const Buffer& buffer) const {
    // TODO:buffer不是指指针为啥->data取内容
    if (buffer->data->type_annotation.defined()) {
      if (const auto* ptr_type = buffer->data->type_annotation.as<PointerTypeNode>()) {
        return ptr_type->storage_scope == "shared";
      }
    }
    return false;
  }

  Buffer ModifyBufferShape(const Buffer& buffer) {
    if (buffer->shape.empty()) {
      return buffer;
    }
    ObjectPtr<BufferNode> new_buffer = make_object<BufferNode>(*buffer.get());

    new_buffer->shape = ModifyExtents(buffer->shape);
    return Buffer(new_buffer);

  }

  Array<PrimExpr> ModifyExtents(const Array<PrimExpr>& extents) {
    if (extents.empty()) {
      return extents;
    }
    Array<PrimExpr> new_extents;
    arith::Analyzer analyzer;

    for (size_t i=0; i < extents.size(); i++) {
      if (i == 0) {
        PrimExpr first_extent = analyzer.Simplify(extents[i]);
        if (const IntImmNode* int_imm = first_extent.as<IntImmNode>()) {
          int64_t new_value = int_imm->value / 2;
          if (new_value < 1) {
            new_value = 1;
          }
          new_extents.push_back(IntImm(first_extent.dtype(), new_value));
          // TODO:非小于1不需要处理？
        } else if (const CastNode* cast_node = first_extent.as<CastNode>()) {
          if (const IntImmNode* int_imm = cast_node->value.as<IntImmNode>()) {
            int64_t new_value = int_imm->value / 2;
            if (new_value < 1) {
              new_value = 1;
            }
            new_extents.push_back(IntImm(cast_node->dtype, new_value));
          } else {
            new_extents.push_back(floordiv(first_extent, 2));
          }
        } else {
          new_extents.push_back(floordiv(first_extent, 2));
        }
      } else {
        new_extents.push_back(extents[i]);
      }
    }
    return new_extents;
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto it = buffer_map_.find(op->buffer);
    if (it != buffer_map_.end()) {
      return BufferLoad(it->second, op->indices);
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto it = buffer_map_.find(op->buffer);
    if (it != buffer_map_.end()) {
      return BufferStore(it->second, op->value, op->indices);
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BlockNode* op) override {
    if (op->alloc_buffers.defined()) {
      for (const auto& buffer : op->alloc_buffers) {
        if (IsUbBuffer(buffer)) {
          Buffer new_buffer = ModifyBufferShape(buffer);
          buffer_map_[buffer] = new_buffer;
        }
      }
    }
    Stmt new_body = this->VisitStmt(op->body);

    Array<Buffer> new_alloc_buffers;
    if (op->alloc_buffers.defined()) {
      for (const auto& buffer : op->alloc_buffers) {
        auto it = buffer_map_.find(buffer);
        if (it != buffer_map_.end()) {
          new_alloc_buffers.push_back(it->second);
        } else {
          new_alloc_buffers.push_back(buffer);
        }
      }
    }
    ObjectPtr<BlockNode> new_block = make_object<BlockNode>(*op);
    new_block->body = new_body;
    new_block->alloc_buffers = new_alloc_buffers;
    return Stmt(new_block);
  }
};



tvm::transform::Pass AscendVidReduction() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto new_func = AscendVidReduction::Substitute(std::move(f), ctx);
    return new_func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendVidReduction", {});
}

// regist host path
TVM_REGISTER_GLOBAL("tl.transform.AscendVidReduction")
    .set_body_typed(AscendVidReduction);

}
}