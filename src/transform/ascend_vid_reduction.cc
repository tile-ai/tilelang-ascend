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

#include <set>
#include <string>

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

  // The vector number takes the value 2 when cid:vid = 1:2 in vid elimination
  // mode
  int threads_cnt_ = 1;

  // Store the original map and the modified map
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      origin_to_new_buffer_;

  // Store GM buffer offset info: GM buffer -> UB buffer
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      gm_buffer_offset_info_;

  // Determine if it is a ub buffer
  bool IsUbBuffer(const Buffer &buffer) const {
    if (buffer->data->type_annotation.defined()) {
      if (const auto *ptr_type =
              buffer->data->type_annotation.as<PointerTypeNode>()) {
        return ptr_type->storage_scope == "shared";
      }
    }
    return false;
  }

  // Modify buffer shape
  Buffer ModifyBufferShape(const Buffer &buffer) {
    if (buffer->shape.empty()) {
      return buffer;
    }
    ObjectPtr<BufferNode> new_buffer = make_object<BufferNode>(*buffer.get());

    new_buffer->shape = ModifyExtents(buffer->shape);
    return Buffer(new_buffer);
  }

  Array<PrimExpr> ModifyExtents(const Array<PrimExpr> &extents) {
    if (extents.empty()) {
      return extents;
    }
    Array<PrimExpr> new_extents;

    for (size_t i = 0; i < extents.size(); i++) {
      if (i == 0) {
        // First dimension divided by 2
        PrimExpr first_extent = analyzer_->Simplify(extents[i]);
        if (const IntImmNode *int_imm = first_extent.as<IntImmNode>()) {
          // Handle constants
          int64_t new_value = int_imm->value / 2;
          if (new_value < 1) {
            new_value = 1;
          }
          new_extents.push_back(IntImm(first_extent.dtype(), new_value));
        } else {
          // Complex expression processing
          new_extents.push_back(indexdiv(first_extent, 2));
        }
      } else {
        // Other dimensions remain unchanged
        new_extents.push_back(VisitExpr(extents[i]));
      }
    }
    return new_extents;
  }

  Stmt VisitStmt_(const BlockNode *op) override {
    if (op->name_hint == "root") {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    // Non-vid reduce mode and vid number is 2, no custom processing required
    if (threads_cnt_ != 2) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    if (op->alloc_buffers.defined()) {
      for (const auto &buffer : op->alloc_buffers) {
        if (IsUbBuffer(buffer)) {
          Buffer new_buffer = ModifyBufferShape(buffer);
          origin_to_new_buffer_[buffer] = new_buffer;
        }
      }
    }

    Stmt new_body = this->VisitStmt(op->body);
    Array<Buffer> new_alloc_buffers;
    if (op->alloc_buffers.defined()) {
      for (const auto &buffer : op->alloc_buffers) {
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

    // Modify reads/writes BufferRegion for GM buffers
    Array<BufferRegion> new_reads = ModifyBufferRegions(op->reads);
    Array<BufferRegion> new_writes = ModifyBufferRegions(op->writes);
    if (!new_reads.same_as(op->reads)) {
      new_block->reads = new_reads;
    }
    if (!new_writes.same_as(op->writes)) {
      new_block->writes = new_writes;
    }

    return Stmt(new_block);
  }

  // Modify BufferRegions for GM buffers that have offset info
  Array<BufferRegion> ModifyBufferRegions(const Array<BufferRegion> &regions) {
    Array<BufferRegion> new_regions = regions;
    bool modified = false;

    for (size_t i = 0; i < regions.size(); ++i) {
      const BufferRegion &region = regions[i];
      auto it = gm_buffer_offset_info_.find(region->buffer);
      if (it != gm_buffer_offset_info_.end()) {
        Buffer ub_buf = it->second;
        int ub_dims = ub_buf->shape.size();

        // Get modified UB buffer shape, skip if not found
        auto ub_it = origin_to_new_buffer_.find(ub_buf);
        if (ub_it == origin_to_new_buffer_.end()) {
          continue;
        }
        Buffer modified_ub_buf = ub_it->second;

        // Calculate target dimension (same logic as ascend_copy)
        int gm_dims = region->region.size();
        int target_dim = gm_dims - ub_dims;

        // Modify the target dimension's Range min
        Array<Range> new_ranges = region->region;
        const Range &old_range = new_ranges[target_dim];
        PrimExpr new_min = old_range->min + vid_ * modified_ub_buf->shape[0];
        new_ranges.Set(target_dim,
                       Range::FromMinExtent(new_min, old_range->extent));

        new_regions.Set(i, BufferRegion(region->buffer, new_ranges));
        modified = true;
      }
    }

    return modified ? new_regions : regions;
  }

  BufferLoad ModifyBufferLoadIndices(const BufferLoad &load, size_t ub_dims,
                                     const Buffer &ub_buf) {
    // Check if ub_buf is in origin_to_new_buffer_, return original if not found
    auto ub_it = origin_to_new_buffer_.find(ub_buf);
    if (ub_it == origin_to_new_buffer_.end()) {
      return load;
    }
    Buffer modified_ub_buf = ub_it->second;

    Array<PrimExpr> new_indices;
    // 1. Recursively process original indices (sub-expressions included)
    for (const PrimExpr &idx : load->indices) {
      new_indices.push_back(VisitExpr(idx));
    }

    // 2. Add vid to the (size - dims)-th dimension
    ICHECK(new_indices.size() >= ub_dims)
        << "[Error]<ascend_vid_reduction.cc>: ub dims may not be more than gm "
           "dims!";
    int target_dim = new_indices.size() - ub_dims;
    new_indices.Set(
        target_dim,
        new_indices[target_dim] +
            vid_ * modified_ub_buf->shape[0] // Insert vid (vector core ID)
    );

    // 3. Refactor BufferLoad
    return BufferLoad(load->buffer, new_indices);
  }

  bool ExtentIsEqualOne(const PrimExpr &extent) {
    PrimExpr simplified_extent = analyzer_->Simplify(extent);
    ICHECK(simplified_extent.defined())
        << "[Error]<ascend_vid_reduction.cc>: Fail to simplify the extent."
        << "Extent: " << extent;
    const IntImmNode *imm = simplified_extent.as<IntImmNode>();
    ICHECK(imm != nullptr) << "[Error]<ascend_vid_reduction.cc>: Extent is not "
                              "IntImm after simplify."
                           << "Extent: " << extent
                           << " Simplified extent: " << simplified_extent;
    return imm->value == 1;
  }

  // Extract buffer from access_ptr call argument
  Buffer ExtractBufferFromArg(const PrimExpr &arg) const {
    if (const CallNode *call = arg.as<CallNode>()) {
      if (call->op.same_as(builtin::tvm_access_ptr())) {
        // access_ptr args: [access_mask, buffer_data, offset, extent, ...]
        // buffer_data is usually at index 1
        if (call->args.size() >= 2) {
          if (const VarNode *var = call->args[1].as<VarNode>()) {
            // Find buffer by data var in origin_to_new_buffer_
            for (const auto &pair : origin_to_new_buffer_) {
              if (pair.first->data.get() == var) {
                return pair.first;
              }
            }
          }
        }
      }
    }
    return Buffer();
  }

  // Check if any buffer in the call args is in origin_to_new_buffer_
  bool HasModifiedBuffer(const CallNode *op) const {
    for (const PrimExpr &arg : op->args) {
      Buffer buf = ExtractBufferFromArg(arg);
      if (buf.defined()) {
        auto it = origin_to_new_buffer_.find(buf);
        if (it != origin_to_new_buffer_.end()) {
          return true;
        }
      }
    }
    return false;
  }

  // Check if the operation is a tile op that needs size modification
  // Only include operations whose last argument is size calculated from buffer
  // shape and has size_0 == size_1 assertion
  bool IsTileOp(const std::string &op_name) const {
    static const std::set<std::string> tile_ops = {
        // binary_op: size = math.prod(dst_extent), assert size_0 == size_1
        "tl.ascend_add", "tl.ascend_adds", "tl.ascend_mul", "tl.ascend_muls",
        "tl.ascend_sub", "tl.ascend_subs", "tl.ascend_div", "tl.ascend_divs",
        "tl.ascend_max", "tl.ascend_maxs", "tl.ascend_min", "tl.ascend_mins",
        "tl.ascend_bitwise_and", "tl.ascend_bitwise_or",
        // unary_op: size = math.prod(dst_extent), assert size_0 == size_1
        "tl.ascend_exp", "tl.ascend_ln", "tl.ascend_abs",
        "tl.ascend_reciprocal", "tl.ascend_sqrt", "tl.ascend_rsqrt",
        "tl.ascend_relu", "tl.ascend_bitwise_not",
        // scalar_op: size = math.prod(src0_extent), assert size_0 == size_2
        "tl.ascend_leaky_relu", "tl.ascend_axpy",
        // bitwise_shift: size = math.prod(src0_extent), assert size_0 == size_2
        "tl.ascend_bitwise_lshift", "tl.ascend_bitwise_rshift",
        // compare: dst_size = math.prod(src0_extent)
        "tl.ascend_compare", "tl.ascend_compare_scalar",
        // sin/cos: size = math.prod(src_extent), assert size_0 == size_2
        "tl.ascend_sin", "tl.ascend_cos", "tl.ascend_fill"};
    return tile_ops.find(op_name) != tile_ops.end();
  }

  // Modify the last argument (size) of tile operations
  PrimExpr ModifyTileOpSize(const CallNode *op) {
    Call tile_call = GetRef<Call>(op);
    Array<PrimExpr> new_args = tile_call->args;

    // Only modify size if the op involves modified buffer
    if (!HasModifiedBuffer(op)) {
      return IRMutatorWithAnalyzer::VisitExpr_(op);
    }

    size_t last_idx = new_args.size() - 1;
    PrimExpr last_arg = new_args[last_idx];

    PrimExpr simplified = analyzer_->Simplify(last_arg);
    if (const IntImmNode *int_imm = simplified.as<IntImmNode>()) {
      int64_t new_value = int_imm->value / threads_cnt_;
      if (new_value < 1)
        new_value = 1;
      new_args.Set(last_idx, IntImm(last_arg.dtype(), new_value));
    } else {
      new_args.Set(last_idx, VisitExpr(indexdiv(last_arg, threads_cnt_)));
    }

    for (size_t i = 0; i < last_idx; ++i) {
      new_args.Set(i, VisitExpr(new_args[i]));
    }

    return Call(tile_call->dtype, tile_call->op, new_args, tile_call->span);
  }

  BufferLoad ExtractBufferLoadFromRegion(const Call &region_call) const {
    ICHECK(region_call->args.size() >= 1)
        << "[Error]<ascend_vid_reduction.cc>: tl.region must have at least 1 "
           "arg (BufferLoad)";
    const BufferLoadNode *load_node = region_call->args[0].as<BufferLoadNode>();
    ICHECK(load_node != nullptr)
        << "[Error]<ascend_vid_reduction.cc>: BufferLoadNode is nullptr";
    return GetRef<BufferLoad>(load_node);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (threads_cnt_ != 2) {
      return IRMutatorWithAnalyzer::VisitExpr_(op);
    }

    // Handle tvm_access_ptr: modify extent if buffer is in
    // origin_to_new_buffer_
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_EQ(op->args.size(), 5U);
      const VarNode *buffer_var = op->args[1].as<VarNode>();

      // Check if this buffer is in origin_to_new_buffer_
      Buffer matched_buffer;
      for (const auto &pair : origin_to_new_buffer_) {
        if (pair.first->data.get() == buffer_var) {
          matched_buffer = pair.first;
          break;
        }
      }

      if (matched_buffer.defined()) {
        // Modify extent (args[3]) by dividing by threads_cnt_
        PrimExpr extent = op->args[3];
        PrimExpr simplified = analyzer_->Simplify(extent);
        PrimExpr new_extent;
        if (const IntImmNode *int_imm = simplified.as<IntImmNode>()) {
          int64_t new_value = int_imm->value / threads_cnt_;
          if (new_value < 1)
            new_value = 1;
          new_extent = IntImm(extent.dtype(), new_value);
        } else {
          new_extent = VisitExpr(indexdiv(extent, threads_cnt_));
        }

        Array<PrimExpr> new_args = op->args;
        new_args.Set(3, new_extent);
        return Call(op->dtype, op->op, new_args, op->span);
      }
      return IRMutatorWithAnalyzer::VisitExpr_(op);
    }

    const OpNode *call_op = op->op.as<OpNode>();
    if (!call_op) {
      return IRMutatorWithAnalyzer::VisitExpr_(op);
    }
    std::string op_name = call_op->name;

    if (IsTileOp(op_name)) {
      return ModifyTileOpSize(op);
    }

    if (op_name != "tl.ascend_copy") {
      return IRMutatorWithAnalyzer::VisitExpr_(op);
    }
    Call ascend_copy = GetRef<Call>(op);

    // Extract the two tl.region CallNodes for src and dst
    Call src_region = Downcast<Call>(ascend_copy->args[0]);
    Call dst_region = Downcast<Call>(ascend_copy->args[1]);
    ICHECK(Downcast<Op>(src_region->op)->name == "tl.region")
        << "args[0] must be tl.region";
    ICHECK(Downcast<Op>(dst_region->op)->name == "tl.region")
        << "args[1] must be tl.region";

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
    // Locate the non-UB region(GM region) and modify the indices of its
    // BufferLoad
    Call target_region =
        src_is_ub ? dst_region : src_region; // target region is gm
    BufferLoad target_load = src_is_ub ? dst_load : src_load;
    Buffer ub_buf = src_is_ub ? src_buf : dst_buf;
    Buffer gm_buf = src_is_ub ? dst_buf : src_buf;
    int ub_dims = ub_buf->shape.size();
    BufferLoad modified_load =
        ModifyBufferLoadIndices(target_load, ub_dims, ub_buf);

    // Record GM buffer offset info for later use in BlockNode reads/writes
    gm_buffer_offset_info_[gm_buf] = ub_buf;

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
          new_region_args.Set(
              i, VisitExpr(indexdiv(new_region_args[i], threads_cnt_)));
        }
      }
    }
    Call modified_region = Call(target_region->dtype, target_region->op,
                                new_region_args, target_region->span);

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
          ub_region_args.Set(
              i, VisitExpr(indexdiv(ub_region_args[i], threads_cnt_)));
        }
      }
    }

    Call modified_ub_region =
        Call(ub_region->dtype, ub_region->op, ub_region_args, ub_region->span);

    // Refactor tl.ascend_copy
    Array<PrimExpr> new_copy_args = ascend_copy->args;
    if (src_is_ub) {
      new_copy_args.Set(0, VisitExpr(modified_ub_region)); // replace ub region
      new_copy_args.Set(1, VisitExpr(modified_region));    // replace gm region
    } else {
      new_copy_args.Set(0, VisitExpr(modified_region));    // replace gm region
      new_copy_args.Set(1, VisitExpr(modified_ub_region)); // replace ub region
    }

    for (size_t i = 2; i < new_copy_args.size(); ++i) {
      new_copy_args.Set(i, VisitExpr(new_copy_args[i]));
    }

    return Call(ascend_copy->dtype, ascend_copy->op, new_copy_args,
                ascend_copy->span);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto it = origin_to_new_buffer_.find(op->buffer);
    if (it != origin_to_new_buffer_.end()) {
      return BufferLoad(it->second, op->indices);
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto it = origin_to_new_buffer_.find(op->buffer);
    if (it != origin_to_new_buffer_.end()) {
      return BufferStore(it->second, op->value, op->indices);
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tvm::tir::attr::thread_extent) {
      if (const IterVarNode *iter_var_node = op->node.as<IterVarNode>()) {
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

} // namespace tl
} // namespace tvm

