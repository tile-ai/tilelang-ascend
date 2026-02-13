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

class AscendVidReduction : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f, PassContext ctx) {
    arith::Analyzer analyzer;
    AscendVidReduction substituter(&analyzer);
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  Var vid_;

  int threads_cnt_ = 1;

  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> origin_to_new_buffer_;

  bool IsUbBuffer(const Buffer& buffer) const {
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

    for (size_t i=0; i < extents.size(); i++) {
      if (i == 0) {
        PrimExpr first_extent = analyzer_->Simplify(extents[i]);
        if (const IntImmNode* int_imm = first_extent.as<IntImmNode>()) {
          int64_t new_value = int_imm->value / 2;
          if (new_value < 1) {
            new_value = 1;
          }
          new_extents.push_back(IntImm(first_extent.dtype(), new_value));
        } else if (const CastNode* cast_node = first_extent.as<CastNode>()) {
          if (const IntImmNode* int_imm = cast_node->value.as<IntImmNode>()) {
            int64_t new_value = int_imm->value / 2;
            if (new_value < 1) {
              new_value = 1;
            }
            new_extents.push_back(IntImm(cast_node->dtype, new_value));
          } else {
            new_extents.push_back(indexdiv(first_extent, 2));
          }
        } else {
          new_extents.push_back(indexdiv(first_extent, 2));
        }
      } else {
        new_extents.push_back(VisitExpr(extents[i]));
      }
    }
    return new_extents;
  }

  Stmt VisitStmt_(const BlockNode* op) override {
    if (op->name_hint == "root") {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    if (threads_cnt_ != 2) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    if (op->alloc_buffers.defined()) {
      for (const auto& buffer : op->alloc_buffers) {
        if (IsUbBuffer(buffer)) {
          Buffer new_buffer = ModifyBufferShape(buffer);
          origin_to_new_buffer_[buffer] = new_buffer;
        }
      }
    }

    Stmt new_body = this->VisitStmt(op->body);
    Array<Buffer> new_alloc_buffers;
    if (op->alloc_buffers.defined()) {
      for (const auto& buffer : op->alloc_buffers) {
        auto it = origin_to_new_buffer_.find(buffer);
        if (it != origin_to_new_buffer_.end()) {
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

  BufferLoad ExtractBufferLoadFromRegion(const Call& region_call) const {
    ICHECK(region_call->args.size() >= 1) << "[Error]<ascend_vid_reduction.cc>: tl.region must have at least 1 arg (BufferLoad)";
    const BufferLoadNode* load_node = region_call->args[0].as<BufferLoadNode>();
    ICHECK(load_node != nullptr) << "[Error]<ascend_vid_reduction.cc>: BufferLoadNode is nullptr";
    return GetRef<BufferLoad>(load_node);
  }

  BufferLoad ModifyBufferLoadIndices(const BufferLoad& load, size_t ub_dims, const Buffer& ub_buf) {
    Array<PrimExpr> new_indices;
    // 1. Recursively process original indices (sub-expressions included)
    for (const PrimExpr& idx : load->indices) {
        new_indices.push_back(VisitExpr(idx));
    }

    // 2. Add vid to the (size - dims)-th dimension
    ICHECK(new_indices.size() >= ub_dims) << "[Error]<ascend_vid_reduction.cc>: ub dims may not be more than gm dims!";
    int target_dim = new_indices.size() - ub_dims;
    Buffer modified_ub_buf = origin_to_new_buffer_[ub_buf];
    new_indices.Set(
        target_dim, 
        new_indices[target_dim] + vid_ * modified_ub_buf->shape[0]// Insert vid (vector core ID)
    );

    // 3. Refactor BufferLoad
    return BufferLoad(load->buffer, new_indices);
  }

  bool ExtentIsEqualOne(const PrimExpr& extent) {
    PrimExpr simplified_extent = analyzer_->Simplify(extent);
    ICHECK(simplified_extent.defined()) 
      << "[Error]<ascend_vid_reduction.cc>: Fail to simplify the extent."
      << "Extent: " << extent;
    const IntImmNode* imm = simplified_extent.as<IntImmNode>();
    ICHECK(imm != nullptr) 
      << "[Error]<ascend_vid_reduction.cc>: Extent is not IntImm after simplify."
      << "Extent: " << extent
      << " Simplified extent: " << simplified_extent;
    return imm->value == 1;
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    // Check if the number of vector cores (v-cores) is 2 
    if (threads_cnt_ != 2) {
      return IRMutatorWithAnalyzer::VisitExpr_(op);
    }

    // Filter out the tl.ascend_copy operator
    const OpNode* call_op = op->op.as<OpNode>();
    if (!call_op) {
      std::cerr << "[info]<callnode>: call_op is nullptr\n";
      return IRMutatorWithAnalyzer::VisitExpr_(op);
    }
    std::string op_name = call_op->name;
    if (op_name != "tl.ascend_copy") {
      return IRMutatorWithAnalyzer::VisitExpr_(op);
    }
    Call ascend_copy = GetRef<Call>(op);

    // Extract the two tl.region CallNodes for src and dst
    Call src_region = Downcast<Call>(ascend_copy->args[0]);
    Call dst_region = Downcast<Call>(ascend_copy->args[1]);
    ICHECK(Downcast<Op>(src_region->op)->name == "tl.region") << "args[0] must be tl.region";
    ICHECK(Downcast<Op>(dst_region->op)->name == "tl.region") << "args[1] must be tl.region";

    // Extract BufferLoad and Buffer from the two regions
    BufferLoad src_load = ExtractBufferLoadFromRegion(src_region);
    BufferLoad dst_load = ExtractBufferLoadFromRegion(dst_region);
    Buffer src_buf = src_load->buffer;
    Buffer dst_buf = dst_load->buffer;

    // Check if there is exactly one UB Buffer
    bool src_is_ub = IsUbBuffer(src_buf);
    bool dst_is_ub = IsUbBuffer(dst_buf);
    bool only_one_ub = (src_is_ub && !dst_is_ub) || (!src_is_ub && dst_is_ub);
    if (!only_one_ub) {
      return IRMutatorWithAnalyzer::VisitExpr_(op);
    }
    // Locate the non-UB region(GM region) and modify the indices of its BufferLoad
    Call target_region = src_is_ub ? dst_region : src_region; // target region is gm
    BufferLoad target_load = src_is_ub ? dst_load : src_load; 
    Buffer ub_buf = src_is_ub ? src_buf : dst_buf;
    int ub_dims = ub_buf->shape.size();
    BufferLoad modified_load = ModifyBufferLoadIndices(target_load, ub_dims, ub_buf);

    // Refactor GM Region
    Array<PrimExpr> new_region_args = target_region->args;
    new_region_args.Set(0, VisitExpr(modified_load)); 
    for (size_t i = 1; i < new_region_args.size(); ++i) {
      if (i != new_region_args.size() - ub_dims) {
        new_region_args.Set(i, VisitExpr(new_region_args[i]));
      } else {
        if (ExtentIsEqualOne(new_region_args[i])) {
          new_region_args.Set(i, VisitExpr(new_region_args[i]));
        } else {
          new_region_args.Set(i, VisitExpr(indexdiv(new_region_args[i], threads_cnt_)));
        }
      }
    }
    Call modified_region = Call(target_region->dtype, target_region->op, new_region_args, target_region->span);

    // Refactor UB Region
    Call ub_region = src_is_ub ? src_region : dst_region;
    Array<PrimExpr> ub_region_args = ub_region->args;
    size_t target_idx = ub_region_args.size() - ub_dims;
    for (size_t i = 0; i < ub_region_args.size(); i++) {
      if (i != target_idx) {
        ub_region_args.Set(i, VisitExpr(ub_region_args[i]));
      } else {
        if (ExtentIsEqualOne(ub_region_args[i])) {
          ub_region_args.Set(i, VisitExpr(ub_region_args[i]));
        } else {
          ub_region_args.Set(i, VisitExpr(indexdiv(ub_region_args[i], threads_cnt_)));
        }
      }
    }

    Call modified_ub_region = Call(ub_region->dtype, ub_region->op, ub_region_args, ub_region->span);

    // Refactor tl.ascend_copy
    Array<PrimExpr> new_copy_args = ascend_copy->args;
    if (src_is_ub) {
      new_copy_args.Set(0, VisitExpr(modified_ub_region)); // replace ub region
      new_copy_args.Set(1, VisitExpr(modified_region));  // replace gm region
    } else {
      new_copy_args.Set(0, VisitExpr(modified_region));  // replace gm region
      new_copy_args.Set(1, VisitExpr(modified_ub_region)); // replace ub region
    }
    
    for (size_t i = 2; i < new_copy_args.size(); ++i) {
      new_copy_args.Set(i, VisitExpr(new_copy_args[i]));
    }

    return Call(ascend_copy->dtype, ascend_copy->op, new_copy_args, ascend_copy->span);
  }
 
  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto it = origin_to_new_buffer_.find(op->buffer);
    if (it != origin_to_new_buffer_.end()) {
      return BufferLoad(it->second, op->indices);
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto it = origin_to_new_buffer_.find(op->buffer);
    if (it != origin_to_new_buffer_.end()) {
      return BufferStore(it->second, op->value, op->indices);
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tvm::tir::attr::thread_extent) {
      if (const IterVarNode* iter_var_node = op->node.as<IterVarNode>()) {
        IterVar iter_var = GetRef<IterVar>(iter_var_node);
        if (iter_var->thread_tag == "threadIdx.x") {
          vid_ = iter_var->var;
          threads_cnt_ = Downcast<IntImm>(op->value)->value;
        }
      }
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
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

} // tl namespace
} // tvm namespace