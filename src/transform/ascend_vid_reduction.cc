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

// static constexpr const char *ascendVidReduction = "tl.ascend_vid_reduction";

// TVM_REGISTER_PASS_CONFIG_OPTION(ascendVidReduction, Bool);

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
    // bool ascend_vid_reduction = ctx->GetConfig<Bool>(ascendVidReduction, Bool(false)).value();
    // if (!ascend_vid_reduction) {
    //   return f;
    // }
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  Var vid_;

  int threads_cnt_ = 1;

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

    for (size_t i=0; i < extents.size(); i++) {
      if (i == 0) {
        PrimExpr first_extent = analyzer_.Simplify(extents[i]);
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

  // 辅助函数：从tl.region CallNode中提取BufferLoad
  BufferLoad ExtractBufferLoadFromRegion(const Call& region_call) const {
    ICHECK(region_call->args.size() >= 1) << "tl.region must have at least 1 arg (BufferLoad)";
    const BufferLoadNode* load_node = region_call->args[0].as<BufferLoadNode>();
    ICHECK(load_node != nullptr) << "tl.region args[0] must be BufferLoad";
    return GetRef<BufferLoad>(load_node);
  }

  // 辅助函数：修改BufferLoad的indices（核心逻辑，可自定义修改规则）
  BufferLoad ModifyBufferLoadIndices(const BufferLoad& load, size_t ub_dims, const Buffer& ub_buf) {
    Array<PrimExpr> new_indices;
    // 1. 递归处理原始indices（保证子表达式被处理）
    for (const PrimExpr& idx : load->indices) {
        new_indices.push_back(VisitExpr(idx));
    }

    // std::cout << "[info]<ModifyBufferLoadIndices>: indices: " << new_indices[target_dim] << std::endl;

    // 2. 自定义修改：第 size - dims 维度添加vid
    ICHECK(new_indices.size() >= ub_dims) << "[Error]<erase_vid>: ub dims must not be more than gm dims!";
    int target_dim = new_indices.size() - ub_dims;
    Buffer modified_ub_buf = buffer_map_[ub_buf];
    new_indices.Set(
        target_dim, 
        new_indices[target_dim] + vid_ * modified_ub_buf->shape[0]// 核心修改：加vid
    );
    
    // std::cout << "[info]<ModifyBufferLo adIndices>: modified indices: " << new_indices[target_dim] << std::endl;



    // 3. 重构BufferLoad
    return BufferLoad(load->buffer, new_indices);
  }


  PrimExpr VisitExpr_(const CallNode* op) final {
    // Step 0：判断v核数是否为2
    // std::cout << "[info]<callnode>: threads_cnt_ = " << threads_cnt_ << '\n';
    if (threads_cnt_ != 2) {
      return IRMutatorWithAnalyzer::VisitExpr_(op);
    }

    // Step 1：过滤出tl.ascend_copy算子
    const OpNode* call_op = op->op.as<OpNode>();
    if (!call_op) {
      std::cerr << "[info]<callnode>: call_op is nullptr\n";
      return IRMutatorWithAnalyzer::VisitExpr_(op);
    }
    std::string op_name = call_op->name;
    // std::cout << "[info]<callnode>: op_name is " << op_name << '\n';
    if (op_name != "tl.ascend_copy") {
      return IRMutatorWithAnalyzer::VisitExpr_(op);
    }
    Call ascend_copy = GetRef<Call>(op);

    // Step 2：提取src/dst两个tl.region CallNode
    Call src_region = Downcast<Call>(ascend_copy->args[0]);
    Call dst_region = Downcast<Call>(ascend_copy->args[1]);
    ICHECK(Downcast<Op>(src_region->op)->name == "tl.region") << "args[0] must be tl.region";
    ICHECK(Downcast<Op>(dst_region->op)->name == "tl.region") << "args[1] must be tl.region";

    // Step 3：提取两个region的BufferLoad和Buffer
    BufferLoad src_load = ExtractBufferLoadFromRegion(src_region);
    BufferLoad dst_load = ExtractBufferLoadFromRegion(dst_region);
    Buffer src_buf = src_load->buffer;
    Buffer dst_buf = dst_load->buffer;

    // Step 4：判断是否「有且仅有一个UB Buffer」
    bool src_is_ub = IsUbBuffer(src_buf);
    bool dst_is_ub = IsUbBuffer(dst_buf);
    bool only_one_ub = (src_is_ub && !dst_is_ub) || (!src_is_ub && dst_is_ub);
    if (!only_one_ub) {
      // 不满足条件我什么都不处理 直接返回
      return IRMutatorWithAnalyzer::VisitExpr_(op);
    }
    // Step 5：定位非UB的region，修改其BufferLoad的indices
    Call target_region = src_is_ub ? dst_region : src_region; // target region is gm
    BufferLoad target_load = src_is_ub ? dst_load : src_load; 
    Buffer ub_buf = src_is_ub ? src_buf : dst_buf;
    int ub_dims = ub_buf->shape.size();
    // std::cout << "[info]<callnode>: ub_dims is " << ub_dims << std::endl;
    BufferLoad modified_load = ModifyBufferLoadIndices(target_load, ub_dims, ub_buf);

    // Step 7：重构目标region（替换修改后的BufferLoad）
    Array<PrimExpr> new_region_args = target_region->args;
    new_region_args.Set(0, modified_load);  // 替换args[0]为新的BufferLoad
    // 递归处理region的其他参数（access_type/extents） 其中extents 对应维度要减半
    for (size_t i = 1; i < new_region_args.size(); ++i) {
      if (i != new_region_args.size() - ub_dims) {
        new_region_args.Set(i, VisitExpr(new_region_args[i]));
      } else {
        new_region_args.Set(i, indexdiv(new_region_args[i], threads_cnt_));
      }
    }
    Call modified_region = Call(target_region->dtype, target_region->op, new_region_args, target_region->span);

    // Step 8: 修改ub region的extents
    Call ub_region = src_is_ub ? src_region : dst_region;

    Array<PrimExpr> ub_region_args = ub_region->args;

    size_t idx = ub_region_args.size() - ub_dims;
    ub_region_args.Set(idx, indexdiv(ub_region_args[idx], threads_cnt_)); // extent的第一维 减半

    Call modified_ub_region = Call(ub_region->dtype, ub_region->op, ub_region_args, ub_region->span);

    // Step 9：重构tl.ascend_copy（替换修改后的region）替换时要进行递归处理
    Array<PrimExpr> new_copy_args = ascend_copy->args;
    if (src_is_ub) {
      new_copy_args.Set(0, VisitExpr(modified_ub_region)); // 替换 ub region
      new_copy_args.Set(1, VisitExpr(modified_region));  // 替换 gm region
    } else {
      new_copy_args.Set(0, VisitExpr(modified_region));  // 替换 gm region
      new_copy_args.Set(1, VisitExpr(modified_ub_region)); // 替换 ub region
    }
    
    // 递归处理ascend_copy的其他参数（如第三个参数bool值）
    for (size_t i = 2; i < new_copy_args.size(); ++i) {
      new_copy_args.Set(i, VisitExpr(new_copy_args[i]));
    }

    // Step 9：返回修改后的tl.ascend_copy CallNode
    return Call(ascend_copy->dtype, ascend_copy->op, new_copy_args, ascend_copy->span);
  }
 
  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    // std::cout << "[info]<bufferloadnode>: op->buffer : " << op->buffer->name << '\n';
    // std::cout << "[info]<bufferloadnode>: op->indices \n";
    // for (auto const &i : op->indices) {
    //   std::cout << "  " << i << '\n';
    // }  
    auto it = buffer_map_.find(op->buffer);
    if (it != buffer_map_.end()) {
      // std::cout << "[info]<bufferloadnode>: modefied op->buffer " << op->buffer->name << '\n';
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

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tvm::tir::attr::thread_extent) {
      if (const IterVarNode* iter_var_node = op->node.as<IterVarNode>()) {
        IterVar iter_var = GetRef<IterVar>(iter_var_node);
        // std::cout << "[info]<cc>: iter_var->thread_tag = " << iter_var->thread_tag << std::endl;
        if (iter_var->thread_tag == "threadIdx.x") {
          vid_ = iter_var->var;
          threads_cnt_ = Downcast<IntImm>(op->value)->value;
          // std::cout << "[info]<pass.cc>: visit attrs vid_: " << vid_ << " threads_cnt_: " << threads_cnt_ << std::endl;
        }
      }
    }
    // copy(A_ub[cid * 1024 + Var(vid) * sizeof(A_ub)], )
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