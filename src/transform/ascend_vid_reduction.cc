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

#include <sstream>
#include <string>
#include <unordered_set>

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

    // Apply transformation (buffers_skip_vid_reduction_ collected internally)
    fptr->body = substituter.VisitStmt(f->body);

    // Store skip buffer names as PrimFunc attrs for downstream passes
    // Only set attrs if there are buffers to skip (avoid empty attrs)
    if (!substituter.buffers_skip_vid_reduction_.empty()) {
      Array<String> skip_buffer_names;
      for (const Buffer &buf : substituter.buffers_skip_vid_reduction_) {
        skip_buffer_names.push_back(buf->name);
      }

      Map<String, ObjectRef> attrs_map;
      if (fptr->attrs.defined()) {
        attrs_map = fptr->attrs->dict;
      }
      attrs_map.Set("buffers_skip_vid_reduction", skip_buffer_names);
      fptr->attrs = DictAttrs(attrs_map);
    }

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

  // Collect all UB buffers in current Block
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> ub_buffers_;

  // UB buffers that should skip vid reduction (when GM indices contain UB
  // BufferLoad)
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>
      buffers_skip_vid_reduction_;

  // Collect current loop variables and their extents
  std::vector<std::pair<Var, PrimExpr>> current_loops_;

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

  // Check if indices directly contain BufferLoad from UB buffers
  // BufferLoad appears directly in indices, not nested in expressions
  bool IndicesContainUbBufferLoad(
      const Array<PrimExpr> &indices,
      const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> &ub_set) {
    for (const PrimExpr &idx : indices) {
      if (const auto *load = idx.as<BufferLoadNode>()) {
        if (ub_set.count(load->buffer) > 0) {
          return true;
        }
      }
    }
    return false;
  }

  // Find loop variable dimension in GM indices (exact match only)
  int FindLoopVarDimInGmIndices(const Array<PrimExpr> &indices,
                                const Var &loop_var) {
    for (size_t i = 0; i < indices.size(); i++) {
      if (const VarNode *v = indices[i].as<VarNode>()) {
        if (v == loop_var.get()) {
          return static_cast<int>(i);
        }
      }
    }
    return -1;
  }

  // Check if GM dimension needs vid offset (loop_extent * threads_cnt <=
  // gm_dim_size)
  bool GmDimNeedsVidOffset(const PrimExpr &gm_dim_size,
                           const PrimExpr &loop_extent) {
    PrimExpr total_extent = loop_extent * threads_cnt_;
    PrimExpr simplified_total = analyzer_->Simplify(total_extent);
    PrimExpr simplified_gm = analyzer_->Simplify(gm_dim_size);

    return analyzer_->CanProve(simplified_total <= simplified_gm);
  }

  class UbBufferCollector : public StmtExprVisitor {
  public:
    std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> ub_buffers;
    std::function<bool(const Buffer &)> is_ub_checker;

    UbBufferCollector(std::function<bool(const Buffer &)> checker)
        : is_ub_checker(checker) {}

    void VisitStmt_(const BlockNode *op) final {
      if (op->alloc_buffers.defined()) {
        for (const Buffer &buffer : op->alloc_buffers) {
          if (is_ub_checker(buffer)) {
            ub_buffers.insert(buffer);
          }
        }
      }
      StmtExprVisitor::VisitStmt_(op);
    }
  };

  class AscendCopyAnalyzer : public StmtExprVisitor {
  public:
    const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> &ub_buffers;
    std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>
        &buffers_skip_vid_reduction;
    std::function<bool(const Buffer &)> is_ub_checker;
    std::function<bool(
        const Array<PrimExpr> &,
        const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> &)>
        indices_contain_ub_checker;

    AscendCopyAnalyzer(
        const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> &ubs,
        std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> &skip_set,
        std::function<bool(const Buffer &)> checker,
        std::function<bool(
            const Array<PrimExpr> &,
            const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> &)>
            idx_checker)
        : ub_buffers(ubs), buffers_skip_vid_reduction(skip_set),
          is_ub_checker(checker), indices_contain_ub_checker(idx_checker) {}

    BufferLoad ExtractBufferLoadFromRegion(const Call &region_call) const {
      ICHECK(region_call->args.size() >= 1)
          << "[Error]<ascend_vid_reduction.cc>: tl.region must have at least 1 "
             "arg (BufferLoad)";
      const BufferLoadNode *load_node =
          region_call->args[0].as<BufferLoadNode>();
      ICHECK(load_node != nullptr)
          << "[Error]<ascend_vid_reduction.cc>: BufferLoadNode is nullptr";
      return GetRef<BufferLoad>(load_node);
    }

    void VisitExpr_(const CallNode *op) final {
      const OpNode *call_op = op->op.as<OpNode>();
      if (!call_op || call_op->name != "tl.ascend_copy") {
        StmtExprVisitor::VisitExpr_(op);
        return;
      }

      Call src_region = Downcast<Call>(op->args[0]);
      Call dst_region = Downcast<Call>(op->args[1]);

      // Extract BufferLoad and Buffer from the two regions
      BufferLoad src_load = ExtractBufferLoadFromRegion(src_region);
      BufferLoad dst_load = ExtractBufferLoadFromRegion(dst_region);
      Buffer src_buf = src_load->buffer;
      Buffer dst_buf = dst_load->buffer;

      bool src_is_ub = is_ub_checker(src_buf);
      bool dst_is_ub = is_ub_checker(dst_buf);

      if (src_is_ub && !dst_is_ub) {
        // UB -> target: src is UB
        // Check if target indices contain UB buffer load
        if (indices_contain_ub_checker(dst_load->indices, ub_buffers)) {
          buffers_skip_vid_reduction.insert(src_buf);
        }
      } else if (!src_is_ub && dst_is_ub) {
        // src -> UB: dst is UB
        // Check if src indices contain UB buffer load
        if (indices_contain_ub_checker(src_load->indices, ub_buffers)) {
          buffers_skip_vid_reduction.insert(dst_buf);
        }
      }

      StmtExprVisitor::VisitExpr_(op);
    }
  };

  bool NeedsVidReduction(const Buffer &buffer) const {
    return IsUbBuffer(buffer) && buffers_skip_vid_reduction_.count(buffer) == 0;
  }

  void AnalyzeBlockBuffers(const Array<Buffer> &alloc_buffers,
                           const Stmt &body) {
    // First, collect UB buffers from current Block's alloc_buffers
    for (const Buffer &buffer : alloc_buffers) {
      if (IsUbBuffer(buffer)) {
        ub_buffers_.insert(buffer);
      }
    }

    // Also collect UB buffers from nested Blocks in body
    UbBufferCollector collector(
        [this](const Buffer &b) { return IsUbBuffer(b); });
    collector(body);
    for (const Buffer &buf : collector.ub_buffers) {
      ub_buffers_.insert(buf);
    }

    // Analyze ascend_copy to find buffers that should skip vid reduction
    AscendCopyAnalyzer analyzer(
        ub_buffers_, buffers_skip_vid_reduction_,
        [this](const Buffer &b) { return IsUbBuffer(b); },
        [this](const Array<PrimExpr> &indices,
               const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>
                   &ub_set) {
          return IndicesContainUbBufferLoad(indices, ub_set);
        });
    analyzer(body);
  }

  Stmt VisitStmt_(const BlockNode *op) override {
    if (op->name_hint == "root") {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    // Non-vid reduce mode and vid number is 2, no custom processing required
    if (threads_cnt_ != 2) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    AnalyzeBlockBuffers(op->alloc_buffers, op->body);

    if (op->alloc_buffers.defined()) {
      for (const Buffer &buffer : op->alloc_buffers) {
        if (NeedsVidReduction(buffer)) {
          Buffer new_buffer = ModifyBufferShape(buffer);
          origin_to_new_buffer_[buffer] = new_buffer;
        }
      }
    }

    Stmt new_body = this->VisitStmt(op->body);
    Array<Buffer> new_alloc_buffers;
    if (op->alloc_buffers.defined()) {
      for (const Buffer &buffer : op->alloc_buffers) {
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
    Buffer buf = load->buffer;
    // If it's not a ub buffer, it could be L1/L0C. If it's not a GM buffer, no
    // action is needed.
    if (buf.scope() != "global") {
      return load;
    }
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
  // Only check origin_to_new_buffer_ keys (original buffers)
  // Note: origin_to_new_buffer_ keys and values share the same data (Var),
  // so we can match by buffer.data
  Buffer ExtractBufferFromArg(const PrimExpr &arg) const {
    if (const CallNode *call = arg.as<CallNode>()) {
      if (call->op.same_as(builtin::tvm_access_ptr())) {
        // access_ptr args: [access_mask, buffer_data, offset, extent, ...]
        // buffer_data is usually at index 1
        if (call->args.size() >= 2) {
          if (const VarNode *var = call->args[1].as<VarNode>()) {
            // Find buffer by data var in origin_to_new_buffer_ keys (original
            // buffers)
            for (const auto &pair : origin_to_new_buffer_) {
              if (pair.first->data.get() == var) {
                return pair.first; // Return original buffer
              }
            }
          }
        }
      }
    }
    return Buffer();
  }

  // Check if tvm_access_ptr has offset == 0
  bool AccessPtrOffsetIsZero(const PrimExpr &arg) const {
    if (const CallNode *call = arg.as<CallNode>()) {
      if (call->op.same_as(builtin::tvm_access_ptr())) {
        if (call->args.size() >= 3) {
          PrimExpr offset = call->args[2];
          PrimExpr simplified = analyzer_->Simplify(offset);
          if (const IntImmNode *offset_imm = simplified.as<IntImmNode>()) {
            return offset_imm->value == 0;
          }
        }
      }
    }
    return false;
  }

  // Check if any buffer in the call args is in origin_to_new_buffer_
  // AND has offset == 0 (for tvm_access_ptr)
  bool HasModifiedBuffer(const CallNode *op) const {
    for (const PrimExpr &arg : op->args) {
      Buffer buf = ExtractBufferFromArg(arg);
      if (buf.defined()) {
        auto it = origin_to_new_buffer_.find(buf);
        if (it != origin_to_new_buffer_.end()) {
          // Check if this arg is tvm_access_ptr with offset == 0
          return AccessPtrOffsetIsZero(arg);
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

  // Modify tl.ascend_reduce: if buffer is vid-reduced, divide M dimension by 2
  PrimExpr ModifyAscendReduce(const CallNode *op) {
    Call ascend_reduce = GetRef<Call>(op);

    // args[0]: string like "reduce_sum<float, 64, 64, -1>"
    // args[1]: out_ptr (tvm_access_ptr)
    // args[2]: buffer_ptr (tvm_access_ptr)
    // args[3]: tmp_ptr (tvm_access_ptr, optional)

    // Extract buffer from args[2] (tvm_access_ptr)
    Buffer buffer = ExtractBufferFromArg(ascend_reduce->args[2]);

    if (!buffer.defined()) {
      return IRMutatorWithAnalyzer::VisitExpr_(op);
    }

    // Check if buffer is vid-reduced (only check origin_to_new_buffer_ keys)
    bool buffer_is_vid_reduced = origin_to_new_buffer_.count(buffer) > 0;
    if (!buffer_is_vid_reduced) {
      return IRMutatorWithAnalyzer::VisitExpr_(op);
    }

    // Parse string parameter: "reduce_sum<float, 64, 64, -1>"
    if (!op->args[0].as<StringImmNode>()) {
      return IRMutatorWithAnalyzer::VisitExpr_(op);
    }
    StringImm param_str = Downcast<StringImm>(op->args[0]);
    std::string param = param_str->value;

    // Find <...> part
    size_t start = param.find('<');
    size_t end = param.find('>');
    if (start == std::string::npos || end == std::string::npos) {
      return IRMutatorWithAnalyzer::VisitExpr_(op);
    }

    std::string content = param.substr(start + 1, end - start - 1);

    // Split by comma: dtype, M, N, dim
    std::vector<std::string> parts;
    std::stringstream ss(content);
    std::string part;
    while (std::getline(ss, part, ',')) {
      parts.push_back(part);
    }

    if (parts.size() < 4) {
      return IRMutatorWithAnalyzer::VisitExpr_(op);
    }

    // parts[0]: dtype (e.g., "float")
    // parts[1]: M (e.g., "64")
    // parts[2]: N (e.g., "64")
    // parts[3]: dim (e.g., "-1")

    std::string dtype = parts[0];
    std::string M_str = parts[1];
    std::string N = parts[2];
    std::string dim = parts[3];

    // Trim leading/trailing spaces
    auto trim = [](std::string s) -> std::string {
      size_t start = s.find_first_not_of(" \t");
      if (start == std::string::npos)
        return std::string("");
      size_t end = s.find_last_not_of(" \t");
      return s.substr(start, end - start + 1);
    };
    dtype = trim(dtype);
    M_str = trim(M_str);
    N = trim(N);
    dim = trim(dim);

    // Try to parse M as integer
    long long M;
    try {
      M = std::stoll(M_str);
    } catch (const std::exception &e) {
      // M is not an integer (e.g., symbolic variable like "v_block")
      // Skip modification for symbolic M
      return IRMutatorWithAnalyzer::VisitExpr_(op);
    }

    // Divide M by 2
    long long new_M = M / threads_cnt_;
    if (new_M < 1)
      new_M = 1;

    // Rebuild string: "reduce_sum<float, 32, 64, -1>"
    std::string new_param = param.substr(0, start + 1) + dtype + ", " +
                            std::to_string(new_M) + ", " + N + ", " + dim + ">";

    // Build new Call
    Array<PrimExpr> new_args = ascend_reduce->args;
    new_args.Set(0, StringImm(new_param));
    for (size_t i = 1; i < new_args.size(); ++i) {
      new_args.Set(i, VisitExpr(new_args[i]));
    }

    return Call(ascend_reduce->dtype, ascend_reduce->op, new_args,
                ascend_reduce->span);
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
        // Check offset (args[2]): only modify extent when offset == 0
        PrimExpr offset = op->args[2];
        PrimExpr simplified_offset = analyzer_->Simplify(offset);
        bool offset_is_zero = false;
        if (const IntImmNode *offset_imm = simplified_offset.as<IntImmNode>()) {
          offset_is_zero = (offset_imm->value == 0);
        }

        if (!offset_is_zero) {
          // offset != 0: accessing different positions in loop, no need to
          // modify extent
          return IRMutatorWithAnalyzer::VisitExpr_(op);
        }

        // offset == 0: modify extent (args[3]) by dividing by threads_cnt_
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

    // Handle tl.ascend_reduce: modify M dimension if buffer is vid-reduced
    if (op_name == "tl.ascend_reduce") {
      return ModifyAscendReduce(op);
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
    bool src_was_vid_reduced = origin_to_new_buffer_.count(src_buf) > 0;
    bool dst_was_vid_reduced = origin_to_new_buffer_.count(dst_buf) > 0;

    // Handle UB -> UB case where both buffers are vid-reduced
    if (src_is_ub && dst_is_ub && src_was_vid_reduced && dst_was_vid_reduced) {
      int src_dims = src_buf->shape.size();
      int dst_dims = dst_buf->shape.size();

      // Refactor src UB Region
      Array<PrimExpr> src_region_args = src_region->args;
      size_t src_target_idx = src_region_args.size() - src_dims;
      for (size_t i = 0; i < src_region_args.size(); i++) {
        if (i == src_target_idx) {
          if (ExtentIsEqualOne(src_region_args[i])) {
            src_region_args.Set(i, VisitExpr(src_region_args[i]));
          } else {
            src_region_args.Set(
                i, VisitExpr(indexdiv(src_region_args[i], threads_cnt_)));
          }
        } else {
          src_region_args.Set(i, VisitExpr(src_region_args[i]));
        }
      }
      Call modified_src_region = Call(src_region->dtype, src_region->op,
                                      src_region_args, src_region->span);

      // Refactor dst UB Region
      Array<PrimExpr> dst_region_args = dst_region->args;
      size_t dst_target_idx = dst_region_args.size() - dst_dims;
      for (size_t i = 0; i < dst_region_args.size(); i++) {
        if (i == dst_target_idx) {
          if (ExtentIsEqualOne(dst_region_args[i])) {
            dst_region_args.Set(i, VisitExpr(dst_region_args[i]));
          } else {
            dst_region_args.Set(
                i, VisitExpr(indexdiv(dst_region_args[i], threads_cnt_)));
          }
        } else {
          dst_region_args.Set(i, VisitExpr(dst_region_args[i]));
        }
      }
      Call modified_dst_region = Call(dst_region->dtype, dst_region->op,
                                      dst_region_args, dst_region->span);

      // Refactor tl.ascend_copy
      Array<PrimExpr> new_copy_args = ascend_copy->args;
      new_copy_args.Set(0, modified_src_region);
      new_copy_args.Set(1, modified_dst_region);
      for (size_t i = 2; i < new_copy_args.size(); ++i) {
        new_copy_args.Set(i, VisitExpr(new_copy_args[i]));
      }

      return Call(ascend_copy->dtype, ascend_copy->op, new_copy_args,
                  ascend_copy->span);
    }

    bool only_one_ub = (src_is_ub && !dst_is_ub) || (!src_is_ub && dst_is_ub);
    if (!only_one_ub) {
      return IRMutatorWithAnalyzer::VisitExpr_(op);
    }
    // Locate the non-UB region and modify the indices of its
    // BufferLoad
    Buffer ub_buf = src_is_ub ? src_buf : dst_buf;
    Buffer gm_buf = src_is_ub ? dst_buf : src_buf;
    BufferLoad gm_load = src_is_ub ? dst_load : src_load;

    bool ub_was_vid_reduced = origin_to_new_buffer_.count(ub_buf) > 0;

    bool gm_indices_contain_ub_load =
        IndicesContainUbBufferLoad(gm_load->indices, ub_buffers_);

    if (gm_indices_contain_ub_load) {
      return IRMutatorWithAnalyzer::VisitExpr_(op);
    }

    // Handle case where UB was not vid-reduced
    // Check if GM indices contain loop variable and needs vid offset
    if (!ub_was_vid_reduced) {
      if (gm_buf.scope() != "global") {
        return IRMutatorWithAnalyzer::VisitExpr_(op);
      }
      Array<PrimExpr> new_gm_indices = gm_load->indices;
      bool indices_modified = false;

      for (const auto &loop_info : current_loops_) {
        Var loop_var = loop_info.first;
        PrimExpr loop_extent = loop_info.second;

        int dim = FindLoopVarDimInGmIndices(gm_load->indices, loop_var);
        if (dim >= 0 && dim < static_cast<int>(gm_buf->shape.size())) {
          // Check if loop_extent * threads_cnt <= gm_dim_size
          if (GmDimNeedsVidOffset(gm_buf->shape[dim], loop_extent)) {
            PrimExpr offset = vid_ * loop_extent;
            new_gm_indices.Set(dim, new_gm_indices[dim] + offset);
            indices_modified = true;
          }
        }
      }

      if (indices_modified) {
        BufferLoad modified_gm_load = BufferLoad(gm_buf, new_gm_indices);
        gm_buffer_offset_info_[gm_buf] = ub_buf;

        // Refactor GM Region
        Call gm_region = src_is_ub ? dst_region : src_region;
        Array<PrimExpr> gm_region_args = gm_region->args;
        gm_region_args.Set(0, VisitExpr(modified_gm_load));
        for (size_t i = 1; i < gm_region_args.size(); ++i) {
          gm_region_args.Set(i, VisitExpr(gm_region_args[i]));
        }
        Call modified_gm_region = Call(gm_region->dtype, gm_region->op,
                                       gm_region_args, gm_region->span);

        // Refactor UB Region (no changes needed since UB was not vid-reduced)
        Call ub_region = src_is_ub ? src_region : dst_region;
        Array<PrimExpr> ub_region_args = ub_region->args;
        for (size_t i = 0; i < ub_region_args.size(); i++) {
          ub_region_args.Set(i, VisitExpr(ub_region_args[i]));
        }
        Call modified_ub_region = Call(ub_region->dtype, ub_region->op,
                                       ub_region_args, ub_region->span);

        // Refactor tl.ascend_copy
        Array<PrimExpr> new_copy_args = ascend_copy->args;
        if (src_is_ub) {
          new_copy_args.Set(0, modified_ub_region);
          new_copy_args.Set(1, modified_gm_region);
        } else {
          new_copy_args.Set(0, modified_gm_region);
          new_copy_args.Set(1, modified_ub_region);
        }
        for (size_t i = 2; i < new_copy_args.size(); ++i) {
          new_copy_args.Set(i, VisitExpr(new_copy_args[i]));
        }

        return Call(ascend_copy->dtype, ascend_copy->op, new_copy_args,
                    ascend_copy->span);
      }

      return IRMutatorWithAnalyzer::VisitExpr_(op);
    }

    // Divide the extent at target_idx by threads_cnt_ when it is not 1.
    // All other args are just visited.
    //   target_idx = static_cast<size_t>(-1) skip extent reduction entirely
    //   arg0_override (optional) replaces args[0] (typically a modified
    //     BufferLoad) before visiting.
    auto reduce_region_extents = [&](const Call &region, size_t target_idx,
                                     PrimExpr arg0_override =
                                         PrimExpr()) -> Call {
      Array<PrimExpr> args = region->args;
      for (size_t i = 0; i < args.size(); i++) {
        if (i == 0 && arg0_override.defined()) {
          args.Set(i, VisitExpr(arg0_override));
        } else if (i == target_idx && !ExtentIsEqualOne(args[i])) {
          args.Set(i, VisitExpr(indexdiv(args[i], threads_cnt_)));
        } else {
          args.Set(i, VisitExpr(args[i]));
        }
      }
      return Call(region->dtype, region->op, args, region->span);
    };

    int ub_dims = ub_buf->shape.size();
    BufferLoad modified_load =
        ModifyBufferLoadIndices(gm_load, ub_dims, ub_buf);

    // Record GM buffer offset info for later use in BlockNode reads/writes
    gm_buffer_offset_info_[gm_buf] = ub_buf;

    // If it's not a UB buffer, it could be L1/L0C.
    // If it's not a GM buffer, no action is needed.
    Call target_region = src_is_ub ? dst_region : src_region;
    size_t target_idx = (gm_buf.scope() == "global")
                            ? (target_region->args.size() - ub_dims)
                            : static_cast<size_t>(-1);
    Call modified_region =
        reduce_region_extents(target_region, target_idx, modified_load);

    // Refactor UB Region
    Call ub_region = src_is_ub ? src_region : dst_region;
    size_t ub_target_idx = ub_region->args.size() - ub_dims;
    Call modified_ub_region = reduce_region_extents(ub_region, ub_target_idx);

    // Refactor tl.ascend_copy
    Array<PrimExpr> new_copy_args = ascend_copy->args;
    if (src_is_ub) {
      new_copy_args.Set(0, VisitExpr(modified_ub_region)); // replace ub region
      new_copy_args.Set(1, VisitExpr(modified_region)); // replace target region
    } else {
      new_copy_args.Set(0, VisitExpr(modified_region));    // replace src region
      new_copy_args.Set(1, VisitExpr(modified_ub_region)); // replace ub region
    }

    for (size_t i = 2; i < new_copy_args.size(); ++i) {
      new_copy_args.Set(i, VisitExpr(new_copy_args[i]));
    }

    if (new_copy_args.size() > 5) {
      auto tmp_expr = ascend_copy->args[5];
      if (auto *tmp_call = tmp_expr.as<CallNode>();
          tmp_call && Downcast<Op>(tmp_call->op)->name == "tl.region") {
        Call tmp_region = Downcast<Call>(tmp_expr);
        BufferLoad tmp_load = ExtractBufferLoadFromRegion(tmp_region);
        if (IsUbBuffer(tmp_load->buffer) &&
            origin_to_new_buffer_.count(tmp_load->buffer) > 0) {
          int tmp_dims = tmp_load->buffer->shape.size();
          size_t tmp_target_idx = tmp_region->args.size() - tmp_dims;
          Call reduced_tmp = reduce_region_extents(tmp_region, tmp_target_idx);
          new_copy_args.Set(5, VisitExpr(reduced_tmp));
        }
      }
    }

    return Call(ascend_copy->dtype, ascend_copy->op, new_copy_args,
                ascend_copy->span);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto it = origin_to_new_buffer_.find(op->buffer);
    Array<PrimExpr> new_indices;
    for (const PrimExpr &idx : op->indices) {
      new_indices.push_back(VisitExpr(idx));
    }
    if (it != origin_to_new_buffer_.end()) {
      return BufferLoad(it->second, new_indices);
    }
    return BufferLoad(op->buffer, new_indices);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto it = origin_to_new_buffer_.find(op->buffer);
    PrimExpr new_value = VisitExpr(op->value);
    Array<PrimExpr> new_indices;
    for (const PrimExpr &idx : op->indices) {
      new_indices.push_back(VisitExpr(idx));
    }
    if (it != origin_to_new_buffer_.end()) {
      return BufferStore(it->second, new_value, new_indices);
    }
    return BufferStore(op->buffer, new_value, new_indices);
  }

  // Check if loop variable is used in vid-reduced UB buffer's first dimension
  // Analyze ORIGINAL body, check ORIGINAL buffers (keys in
  // origin_to_new_buffer_) Two cases:
  // 1. Loop var directly used as first index in BufferLoad/BufferStore
  // 2. Loop var used in tvm_access_ptr offset where offset/extent = loop_var
  bool LoopVarUsedInVidReducedUbFirstDim(const Var &loop_var,
                                         const Stmt &body) {
    class LoopVarAnalyzer : public StmtExprVisitor {
    public:
      Var target_var;
      arith::Analyzer *analyzer;
      const std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
          &origin_to_new_buffer;
      bool found = false;

      LoopVarAnalyzer(const Var &var, arith::Analyzer *a,
                      const std::unordered_map<Buffer, Buffer, ObjectPtrHash,
                                               ObjectPtrEqual> &buffers)
          : target_var(var), analyzer(a), origin_to_new_buffer(buffers) {}

      void VisitExpr_(const BufferLoadNode *op) final {
        if (origin_to_new_buffer.count(op->buffer) > 0) {
          if (!op->indices.empty()) {
            const PrimExpr &first_idx = op->indices[0];
            if (const VarNode *v = first_idx.as<VarNode>()) {
              if (v == target_var.get()) {
                found = true;
                return;
              }
            }
          }
        }
        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitStmt_(const BufferStoreNode *op) final {
        if (origin_to_new_buffer.count(op->buffer) > 0) {
          if (!op->indices.empty()) {
            const PrimExpr &first_idx = op->indices[0];
            if (const VarNode *v = first_idx.as<VarNode>()) {
              if (v == target_var.get()) {
                found = true;
                return;
              }
            }
          }
        }
        StmtExprVisitor::VisitStmt_(op);
      }

      void VisitExpr_(const CallNode *op) final {
        // Handle tvm_access_ptr: check if offset/extent equals loop_var
        if (!op->op.same_as(builtin::tvm_access_ptr()) || op->args.size() < 4) {
          StmtExprVisitor::VisitExpr_(op);
          return;
        }

        // Extract buffer from args[1] (buffer_data)
        const VarNode *buffer_var = op->args[1].as<VarNode>();
        if (!buffer_var) {
          StmtExprVisitor::VisitExpr_(op);
          return;
        }

        // Find buffer in origin_to_new_buffer
        Buffer buffer;
        for (const auto &pair : origin_to_new_buffer) {
          if (pair.first->data.get() == buffer_var) {
            buffer = pair.first;
            break;
          }
        }

        // buffer.defined() implies buffer is vid-reduced UB (checked
        // during BlockNode processing)
        if (!buffer.defined()) {
          StmtExprVisitor::VisitExpr_(op);
          return;
        }

        // Get offset and extent from access_ptr
        PrimExpr offset = op->args[2];
        PrimExpr extent = op->args[3];

        PrimExpr simplified_extent = analyzer->Simplify(extent);
        const IntImmNode *extent_imm = simplified_extent.as<IntImmNode>();
        if (!extent_imm || extent_imm->value == 0) {
          StmtExprVisitor::VisitExpr_(op);
          return;
        }

        // Check if offset / extent simplifies to loop_var
        PrimExpr division = analyzer->Simplify(indexdiv(offset, extent));
        const VarNode *v = division.as<VarNode>();
        if (v && v == target_var.get()) {
          found = true;
          return;
        }

        StmtExprVisitor::VisitExpr_(op);
      }
    };

    LoopVarAnalyzer analyzer(loop_var, analyzer_, origin_to_new_buffer_);
    analyzer(body);
    return analyzer.found;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    if (threads_cnt_ == 2) {
      // Step 1: Analyze whether vid reduction is needed (analyze original body)
      bool need_reduce =
          LoopVarUsedInVidReducedUbFirstDim(op->loop_var, op->body);

      // Step 2: Calculate the correct extent
      PrimExpr effective_extent = op->extent;
      if (need_reduce) {
        PrimExpr simplified = analyzer_->Simplify(op->extent);
        if (const IntImmNode *int_imm = simplified.as<IntImmNode>()) {
          int64_t new_value = int_imm->value / threads_cnt_;
          if (new_value < 1)
            new_value = 1;
          effective_extent = IntImm(op->extent.dtype(), new_value);
        } else {
          effective_extent = indexdiv(op->extent, threads_cnt_);
        }

        // Step 3: Only add vid-reduced loops to current_loops_
        current_loops_.push_back({op->loop_var, effective_extent});
      }

      // Step 4: Process the loop body
      Stmt new_body = VisitStmt(op->body);

      // Step 5: Pop on exit (if previously pushed)
      if (need_reduce) {
        current_loops_.pop_back();
      }

      // Step 6: Return the new ForNode
      return For(op->loop_var, op->min, effective_extent, op->kind, new_body,
                 op->thread_binding, op->annotations);
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