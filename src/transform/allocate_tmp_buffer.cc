// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include <tvm/ir/op.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "tir/transforms/ir_utils.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../op/ascend.h"
#include "common/operation_config.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

static constexpr const char *kAscendInjectTmpBuffer =
    "tl.ascend_inject_tmp_buffer";

TVM_REGISTER_PASS_CONFIG_OPTION(kAscendInjectTmpBuffer, Bool);

namespace {

bool IsConstFalse(const PrimExpr &expr) {
  return expr.defined() && expr.dtype().is_bool() && is_zero(expr);
}

int64_t AlignReduceOutputCols(int64_t valid_col, int64_t dtype_bytes) {
  const int64_t aligned_bytes = ((valid_col * dtype_bytes + 31) / 32) * 32;
  return aligned_bytes / dtype_bytes;
}

int64_t GetPtoRowReduceTmpCols(int64_t valid_col, int64_t dtype_bytes) {
  constexpr int64_t kVectorRepeatBytes = 256;
  const int64_t elem_per_repeat = kVectorRepeatBytes / dtype_bytes;
  const int64_t tmp_col = valid_col <= elem_per_repeat
                              ? 1
                              : std::max(valid_col / 2, elem_per_repeat);
  return AlignReduceOutputCols(tmp_col, dtype_bytes);
}

} // namespace

class CallNodeCollector : public ExprVisitor, public StmtVisitor {
public:
  static std::vector<Call> Collect(PrimFunc f, Target target) {
    CallNodeCollector collector;
    std::string target_ = Downcast<String>(target.get()->attrs["model"]);
    if ("pto" == target_) {
      collector.tmp_arg_ops_ = pto_tmp_arg_ops;
    } else if ("ascendc" == target_ || "auto" == target_) {
      collector.tmp_arg_ops_ = ascendc_tmp_arg_ops;
    }
    return collector.Find(f->body);
  }

private:
  std::vector<Call> Find(const Stmt &stmt) {
    calls_.clear();
    VisitStmt(stmt);
    return calls_;
  }

  void VisitExpr_(const CallNode *op) override {
    if (const auto *op_node = op->op.as<OpNode>()) {
      // Here we only focus on CallNodes that require a tmp parameter.
      if (tmp_arg_ops_.count(op_node) > 0) {
        calls_.push_back(GetRef<Call>(op));
      }
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr(const PrimExpr &expr) override {
    ExprVisitor::VisitExpr(expr);
  }

  std::vector<Call> calls_;
  std::unordered_map<const tvm::OpNode *, int64_t> tmp_arg_ops_;
};

class CallNodeModifier : public StmtExprMutator {
public:
  static Stmt Modify(PrimFunc f, Target target, Buffer &tmp_buffer,
                     Array<Buffer> &tmp_buffers,
                     Buffer &reduce_out_tmp_buffer) {
    CallNodeModifier modifier;
    modifier.target_ = Downcast<String>(target.get()->attrs["model"]);
    if ("pto" == modifier.target_) {
      modifier.tmp_arg_ops_ = pto_tmp_arg_ops;
    } else if ("ascendc" == modifier.target_ || "auto" == modifier.target_) {
      modifier.tmp_arg_ops_ = ascendc_tmp_arg_ops;
    }
    modifier.tmp_buf_ = tmp_buffer;
    modifier.tmp_bufs_ = tmp_buffers;
    modifier.reduce_out_tmp_buf_ = reduce_out_tmp_buffer;
    return modifier.AddTmpArg(f->body);
  }

private:
  Stmt AddTmpArg(const Stmt &stmt) { return VisitStmt(stmt); }

  PrimExpr VisitExpr_(const CallNode *op) override {
    if (const auto *op_node = op->op.as<OpNode>()) {
      if (tmp_arg_ops_.count(op_node) > 0) {
        int64_t tmp_buffer_param_offset = tmp_arg_ops_.at(op_node);
        if (NeedReduceOutputTmp(op)) {
          return CallNodeAddReduceOutputTmp(op, tmp_buffer_param_offset, 1);
        }
        if (op->op.same_as(tl::ascend_sigmoid()) ||
            op->op.same_as(tl::ascend_pow()) ||
            op->op.same_as(tl::ascend_bitwise_xor()) ||
            op->op.same_as(tl::ascend_merge_sort()) ||
            ("pto" == target_ && (op->op.same_as(tl::ascend_sort()) ||
                                  op->op.same_as(tl::ascend_topk())))) {
          return CallNodeAddTmp(op, tmp_buffer_param_offset, 2);
        } else {
          return CallNodeAddTmp(op, tmp_buffer_param_offset, 1);
        }
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Call CallNodeAddTmp(const CallNode *op, int64_t tmp_buffer_param_offset,
                      int64_t rw_mask) {
    PrimExpr access_ptr = this->AddTmpArgs_(op, rw_mask);
    Array<PrimExpr> new_args =
        this->InsertExprAt_(op->args, tmp_buffer_param_offset, access_ptr);
    return Call(op->dtype, op->op, new_args, Span());
  }

  Call CallNodeAddReduceOutputTmp(const CallNode *op,
                                  int64_t tmp_buffer_param_offset,
                                  int64_t rw_mask) {
    Array<PrimExpr> new_args = this->InsertExprAt_(
        op->args, tmp_buffer_param_offset, this->AddTmpArgs_(op, rw_mask));
    new_args = this->InsertExprAt_(
        new_args, tmp_buffer_param_offset + 1,
        this->MakeAccessPtrFromBuffer_(reduce_out_tmp_buf_, rw_mask));
    return Call(op->dtype, op->op, new_args, Span());
  }

  // Insert an expression at the specified position.
  Array<PrimExpr> InsertExprAt_(const Array<PrimExpr> &arr, size_t pos,
                                const PrimExpr &expr) {
    Array<PrimExpr> new_arr;

    for (size_t i = 0; i < pos && i < arr.size(); ++i) {
      new_arr.push_back(arr[i]);
    }

    new_arr.push_back(expr);

    for (size_t i = pos; i < arr.size(); ++i) {
      new_arr.push_back(arr[i]);
    }

    return new_arr;
  }

  PrimExpr AddTmpArgs_(const CallNode *op, int64_t rw_mask) {
    Buffer tmp_buffer;
    if (("ascendc" == target_ || "auto" == target_) &&
        (op->op.same_as(tl::ascend_sort()) ||
         op->op.same_as(tl::ascend_topk())) &&
        tmp_bufs_.size() > 0) {
      const CallNode *src_access_ptr = Downcast<Call>(op->args[1]).get();
      DataType dtype = src_access_ptr->args[0].as<CallNode>()->dtype;
      if (dtype == DataType::UInt(8)) {
        tmp_buffer = tmp_buf_;
      } else {
        for (const Buffer &sort_topk_tmp_buffer : tmp_bufs_) {
          if (sort_topk_tmp_buffer.get()->dtype == dtype) {
            tmp_buffer = sort_topk_tmp_buffer;
            break;
          }
        }
      }
    } else if ("pto" == target_ &&
               (op->op.same_as(tl::ascend_bitwise_xor()) ||
                op->op.same_as(tl::ascend_sort()) ||
                op->op.same_as(tl::ascend_topk())) &&
               tmp_bufs_.size() > 0) {
      const CallNode *src_access_ptr = Downcast<Call>(op->args[1]).get();
      DataType dtype = src_access_ptr->args[0].as<CallNode>()->dtype;
      if (dtype == DataType::UInt(8)) {
        tmp_buffer = tmp_buf_;
      } else {
        for (const Buffer &xor_tmp_buffer : tmp_bufs_) {
          if (xor_tmp_buffer.get()->dtype == dtype) {
            tmp_buffer = xor_tmp_buffer;
            break;
          }
        }
      }
    } else if ("pto" == target_ && op->op.same_as(tl::ascend_merge_sort()) &&
               tmp_bufs_.size() > 0) {
      const CallNode *dst_access_ptr = Downcast<Call>(op->args[2]).get();
      DataType dtype = dst_access_ptr->args[0].as<CallNode>()->dtype;
      if (dtype == DataType::UInt(8)) {
        tmp_buffer = tmp_buf_;
      } else {
        for (const Buffer &merge_sort_tmp_buffer : tmp_bufs_) {
          if (merge_sort_tmp_buffer.get()->dtype == dtype) {
            tmp_buffer = merge_sort_tmp_buffer;
            break;
          }
        }
      }
    } else if ("pto" == target_ && op->op.same_as(tl::ascend_gather_mask()) &&
               tmp_bufs_.size() > 0 && op->args[3].as<CallNode>()) {
      // gather_mask args[3] is the src1 pattern: either a Call (Buffer
      // pattern) or a StringImm ("P1010" etc.). Only the Buffer-pattern
      // form needs a dtype-keyed tmp; the string-pattern form falls through
      // to the generic uint8 tmp_buf_ below.
      const CallNode *src_access_ptr = Downcast<Call>(op->args[3]).get();
      DataType dtype = src_access_ptr->args[0].as<CallNode>()->dtype;
      if (dtype == DataType::UInt(8)) {
        tmp_buffer = tmp_buf_;
      } else {
        for (const Buffer &xor_tmp_buffer : tmp_bufs_) {
          if (xor_tmp_buffer.get()->dtype == dtype) {
            tmp_buffer = xor_tmp_buffer;
            break;
          }
        }
      }
    } else if ("pto" == target_ && op->op.same_as(tl::ascend_gather()) &&
               tmp_bufs_.size() > 0) {
      // TGATHER requires tmp buffer dtype == indices dtype (static_assert
      // in TGather.hpp). The indices are at op->args[2]; pick the matching
      // dtype buffer from tmp_bufs_.
      const CallNode *idx_access_ptr = Downcast<Call>(op->args[2]).get();
      DataType dtype = idx_access_ptr->args[0].as<CallNode>()->dtype;
      for (const Buffer &gather_tmp_buffer : tmp_bufs_) {
        if (gather_tmp_buffer.get()->dtype == dtype) {
          tmp_buffer = gather_tmp_buffer;
          break;
        }
      }
    } else {
      tmp_buffer = tmp_buf_;
    }

    return MakeAccessPtrFromBuffer_(tmp_buffer, rw_mask);
  }

  PrimExpr MakeAccessPtrFromBuffer_(const Buffer &tmp_buffer, int64_t rw_mask) {
    ICHECK(tmp_buffer.defined()) << "Expected tmp buffer to be defined.";

    int64_t shape_size = 0;
    for (size_t j = 0; j < tmp_buffer.get()->shape.size(); j++) {
      if (shape_size == 0) {
        shape_size = tmp_buffer.get()->shape[j].as<IntImmNode>()->value;
      } else {
        shape_size *= tmp_buffer.get()->shape[j].as<IntImmNode>()->value;
      }
    }
    // Directly construct a CallNode for tvm_access_ptr
    Array<PrimExpr> args;
    args.push_back(TypeAnnotation(tmp_buffer.get()->dtype));
    args.push_back(tmp_buffer->data);
    args.push_back(Integer(0));
    args.push_back(Integer(shape_size));
    args.push_back(Integer(rw_mask));
    return Call(DataType::Handle(), builtin::tvm_access_ptr(), args);
  }

  bool NeedReduceOutputTmp(const CallNode *op) const {
    return "pto" == target_ && reduce_out_tmp_buf_.defined() &&
           op->op.same_as(tl::ascend_reduce()) && op->args.size() >= 4 &&
           IsConstFalse(op->args[op->args.size() - 1]);
  }

  Buffer tmp_buf_;
  Array<Buffer> tmp_bufs_;
  Buffer reduce_out_tmp_buf_;
  std::string target_;
  std::unordered_map<const tvm::OpNode *, int64_t> tmp_arg_ops_;
};

class TmpBufferInjector : public StmtExprMutator {
public:
  static PrimFunc TmpBufferInject(PrimFunc f, Target target,
                                  bool inject_enabled) {
    TmpBufferInjector injector;
    injector.target_ = Downcast<String>(target.get()->attrs["model"]);
    injector.inject_enabled_ = inject_enabled;
    PrimFuncNode *fptr = f.CopyOnWrite();
    injector.calls_ = CallNodeCollector::Collect(f, target);
    Stmt new_body = injector.inject(f->body);
    fptr->body = new_body;
    new_body = CallNodeModifier::Modify(f, target, injector.tmp_buf_,
                                        injector.tmp_bufs_,
                                        injector.reduce_out_tmp_buf_);
    fptr->body = new_body;

    if (inject_enabled && injector.tmp_buf_.defined()) {
      auto fn_attr = fptr->attrs.CopyOnWrite();
      Array<Var> tmp_buffer_vars;
      tmp_buffer_vars.push_back(injector.tmp_buf_->data);
      for (const auto &buf : injector.tmp_bufs_) {
        tmp_buffer_vars.push_back(buf->data);
      }
      if (injector.reduce_out_tmp_buf_.defined()) {
        tmp_buffer_vars.push_back(injector.reduce_out_tmp_buf_->data);
      }
      fn_attr->dict.Set("tmp_buffer_vars", tmp_buffer_vars);
    }

    return f;
  }

private:
  Stmt inject(const Stmt &stmt) { return VisitStmt(stmt); }

  Stmt VisitStmt_(const BlockRealizeNode *node) override {
    if (node->block->name_hint == "tilelang_root") {
      Block block = Downcast<Block>(node->block);
      BlockNode *op = block.CopyOnWrite();
      // Insert a tmp buffer into the alloc_buffers of the Block
      Array<Buffer> new_alloc_buffers = op->alloc_buffers;

      if (inject_enabled_) {
        tmp_buf_ = createTmpBuffer_(op->alloc_buffers);
        if (tmp_buf_.defined()) {
          new_alloc_buffers.push_back(tmp_buf_);
        }

        if ("pto" == target_) {
          reduce_out_tmp_buf_ =
              createPTOClearReduceOutputTmpBuffer_(op->alloc_buffers);
          if (reduce_out_tmp_buf_.defined()) {
            new_alloc_buffers.push_back(reduce_out_tmp_buf_);
          }
          tmp_bufs_ = createPTOXORAndMergeSortAndGatherMaskTmpBuffer_(
              op->alloc_buffers);
          for (const Buffer &tmp_buffer : tmp_bufs_) {
            new_alloc_buffers.push_back(tmp_buffer);
          }
        } else if ("ascendc" == target_ || "auto" == target_) {
          tmp_bufs_ = createASCTopKAndSortTmpBuffer_(op->alloc_buffers);
          for (const Buffer &tmp_buffer : tmp_bufs_) {
            new_alloc_buffers.push_back(tmp_buffer);
          }
        }
      } else {
        CollectUserProvidedTmpBuffers_(op->alloc_buffers);
      }

      // return new Block
      Block new_block = Block(
          op->iter_vars, op->reads, op->writes, op->name_hint, op->body,
          op->init, new_alloc_buffers, op->match_buffers, op->annotations);
      return BlockRealize(node->iter_values, node->predicate, new_block);
    }
    return StmtExprMutator::VisitStmt_(node);
  }

  Buffer createTmpBuffer_(Array<Buffer> alloc_buffers) {
    Array<PrimExpr> shape;
    if ("pto" == target_) {
      shape = GetPTOTmpBufferSize_(alloc_buffers);
    } else if ("ascendc" == target_ || "auto" == target_) {
      shape = GetAscendCTmpBufferSize_(alloc_buffers);
    }

    if (shape.size() > 0) {
      Var tmp_buf(buffer_name_,
                  PointerType(PrimType(DataType::UInt(8)), "shared"));
      Buffer buffer = Buffer(tmp_buf, DataType::UInt(8), shape, {}, PrimExpr(),
                             buffer_name_, -1, 0, BufferType::kDefault);

      return buffer;
    } else {
      return Buffer();
    }
  }

  Buffer createPTOClearReduceOutputTmpBuffer_(Array<Buffer> alloc_buffers) {
    int64_t shape_size = 0;
    for (size_t i = 0; i < calls_.size(); i++) {
      const CallNode *call = calls_[i].get();
      if (!call->op.same_as(tl::ascend_reduce()) || call->args.size() < 4 ||
          !IsConstFalse(call->args[call->args.size() - 1])) {
        continue;
      }

      const CallNode *dst_access_ptr = Downcast<Call>(call->args[1]).get();
      std::string dst_buffer_name =
          dst_access_ptr->args[1].as<VarNode>()->name_hint;
      const BufferNode *dst_buffer_node =
          GetBufferNodeByName_(alloc_buffers, dst_buffer_name);
      ICHECK(dst_buffer_node) << "Buffer not found for " << dst_buffer_name;

      int64_t col = 1;
      if (dst_buffer_node->shape.size() == 1) {
        col = dst_buffer_node->shape[0].as<IntImmNode>()->value;
      } else if (dst_buffer_node->shape.size() == 2 &&
                 dst_buffer_node->shape[0].as<IntImmNode>()->value == 0) {
        col = dst_buffer_node->shape[1].as<IntImmNode>()->value;
      } else if (dst_buffer_node->shape.size() == 2 &&
                 dst_buffer_node->shape[1].as<IntImmNode>()->value == 0) {
        col = dst_buffer_node->shape[0].as<IntImmNode>()->value;
      } else {
        col = dst_buffer_node->shape[1].as<IntImmNode>()->value;
      }

      const int64_t extent = Downcast<IntImm>(dst_access_ptr->args[3])->value;
      const int64_t valid_row = std::max<int64_t>((extent + col - 1) / col, 1);
      const int64_t valid_col = extent > col ? col : extent;
      const int64_t padded_col =
          AlignReduceOutputCols(valid_col, dst_buffer_node->dtype.bytes());
      const int64_t tmp_shape_size =
          valid_row * padded_col * dst_buffer_node->dtype.bytes();
      shape_size = std::max(shape_size, tmp_shape_size);
    }

    if (shape_size == 0) {
      return Buffer();
    }

    const std::string buffer_name = buffer_name_ + "_reduce_out";
    Var tmp_buf(buffer_name,
                PointerType(PrimType(DataType::UInt(8)), "shared"));
    return Buffer(tmp_buf, DataType::UInt(8),
                  {IntImm(DataType::Int(32), shape_size)}, {}, PrimExpr(),
                  buffer_name, -1, 0, BufferType::kDefault);
  }

  Array<Buffer> createASCTopKAndSortTmpBuffer_(Array<Buffer> alloc_buffers) {
    std::unordered_map<DataType, Array<PrimExpr>> shapes;
    for (size_t i = 0; i < calls_.size(); i++) {
      const CallNode *call = calls_[i].get();
      if (call->op.same_as(tl::ascend_sort())) {
        const CallNode *src_access_ptr = Downcast<Call>(call->args[2]).get();
        std::string src_buffer_name =
            src_access_ptr->args[1].as<VarNode>()->name_hint;
        const BufferNode *src_buffer_node =
            GetBufferNodeByName_(alloc_buffers, src_buffer_name);
        DataType dtype = src_buffer_node->dtype;
        if (dtype != DataType::UInt(8)) {
          if (shapes.count(dtype) > 0) {
            int64_t shape_size = 0;
            for (size_t k = 0; k < shapes.at(dtype).size(); k++) {
              if (shape_size == 0) {
                shape_size = shapes.at(dtype)[k].as<IntImmNode>()->value;
              } else {
                shape_size *= shapes.at(dtype)[k].as<IntImmNode>()->value;
              }
            }
            int64_t tmp_shape_size =
                Downcast<IntImm>(src_access_ptr->args[3])->value * 8;
            if (tmp_shape_size > shape_size) {
              Array<PrimExpr> tmp_shape;
              tmp_shape.push_back(IntImm(DataType::Int(32), tmp_shape_size));
              shapes[dtype] = tmp_shape;
            }
          } else {
            Array<PrimExpr> tmp_shape;
            int64_t tmp_shape_size =
                Downcast<IntImm>(src_access_ptr->args[3])->value * 8;
            tmp_shape.push_back(IntImm(DataType::Int(32), tmp_shape_size));
            shapes[dtype] = tmp_shape;
          }
        }
      } else if (call->op.same_as(tl::ascend_topk())) {
        const CallNode *src_access_ptr = Downcast<Call>(call->args[2]).get();
        std::string src_buffer_name =
            src_access_ptr->args[1].as<VarNode>()->name_hint;
        const BufferNode *src_buffer_node =
            GetBufferNodeByName_(alloc_buffers, src_buffer_name);
        DataType dtype = src_buffer_node->dtype;
        if (dtype != DataType::UInt(8)) {
          if (shapes.count(dtype) > 0) {
            int64_t shape_size = 0;
            for (size_t k = 0; k < shapes.at(dtype).size(); k++) {
              if (shape_size == 0) {
                shape_size = shapes.at(dtype)[k].as<IntImmNode>()->value;
              } else {
                shape_size *= shapes.at(dtype)[k].as<IntImmNode>()->value;
              }
            }
            int64_t tmp_shape_size =
                Downcast<IntImm>(src_access_ptr->args[3])->value * 4;
            if (tmp_shape_size > shape_size) {
              Array<PrimExpr> tmp_shape;
              tmp_shape.push_back(IntImm(DataType::Int(32), tmp_shape_size));
              shapes[dtype] = tmp_shape;
            }
          } else {
            Array<PrimExpr> tmp_shape;
            int64_t tmp_shape_size =
                Downcast<IntImm>(src_access_ptr->args[3])->value * 4;
            tmp_shape.push_back(IntImm(DataType::Int(32), tmp_shape_size));
            shapes[dtype] = tmp_shape;
          }
        }
      }
    }
    // Create a tmp_buffer of the required type
    Array<Buffer> buffers;
    int64_t i = 1;
    for (const auto &kv : shapes) {
      const DataType &key = kv.first;
      const Array<PrimExpr> &value = kv.second;
      std::string buffer_name = buffer_name_ + "_" + std::to_string(i);
      Var tmp_buf(buffer_name, PointerType(PrimType(key), "shared"));
      Buffer buffer = Buffer(tmp_buf, key, value, {}, PrimExpr(), buffer_name,
                             -1, 0, BufferType::kDefault);
      buffers.push_back(buffer);
      i++;
    }
    return buffers;
  }

  Array<Buffer>
  createPTOXORAndMergeSortAndGatherMaskTmpBuffer_(Array<Buffer> alloc_buffers) {
    std::unordered_map<DataType, Array<PrimExpr>> shapes;
    // Iterate over the stored CallNodes, find the corresponding xor and
    // merge_sort and allocate tmp_ub for them (requires tmp_ub of the
    // corresponding datatype)
    for (size_t i = 0; i < calls_.size(); i++) {
      const CallNode *call = calls_[i].get();
      if (call->op.same_as(tl::ascend_bitwise_xor())) {
        const CallNode *src_access_ptr = Downcast<Call>(call->args[1]).get();
        std::string src_buffer_name =
            src_access_ptr->args[1].as<VarNode>()->name_hint;
        const BufferNode *src_buffer_node;
        src_buffer_node = GetBufferNodeByName_(alloc_buffers, src_buffer_name);
        DataType dtype = src_buffer_node->dtype;
        if (dtype != DataType::UInt(8)) {
          if (shapes.count(dtype) > 0) {
            int64_t shape_size = 0;
            for (size_t k = 0; k < shapes.at(dtype).size(); k++) {
              if (shape_size == 0) {
                shape_size = shapes.at(dtype)[k].as<IntImmNode>()->value;
              } else {
                shape_size *= shapes.at(dtype)[k].as<IntImmNode>()->value;
              }
            }
            int64_t tmp_shape_size =
                Downcast<IntImm>(src_access_ptr->args[3])->value;
            if (tmp_shape_size > shape_size) {
              Array<PrimExpr> tmp_shape;
              for (size_t k = 0; k < src_buffer_node->shape.size(); k++) {
                tmp_shape.push_back(
                    IntImm(DataType::Int(32),
                           src_buffer_node->shape[k].as<IntImmNode>()->value));
              }
              shapes[dtype] = tmp_shape;
            }
          } else {
            Array<PrimExpr> tmp_shape;
            for (size_t k = 0; k < src_buffer_node->shape.size(); k++) {
              tmp_shape.push_back(
                  IntImm(DataType::Int(32),
                         src_buffer_node->shape[k].as<IntImmNode>()->value));
            }
            shapes[dtype] = tmp_shape;
          }
        }
      } else if (call->op.same_as(tl::ascend_merge_sort())) {
        const CallNode *dst_access_ptr = Downcast<Call>(call->args[2]).get();
        std::string dst_buffer_name =
            dst_access_ptr->args[1].as<VarNode>()->name_hint;
        const BufferNode *dst_buffer_node;
        dst_buffer_node = GetBufferNodeByName_(alloc_buffers, dst_buffer_name);
        DataType dtype = dst_buffer_node->dtype;
        if (dtype != DataType::UInt(8)) {
          if (shapes.count(dtype) > 0) {
            int64_t shape_size = 0;
            for (size_t k = 0; k < shapes.at(dtype).size(); k++) {
              if (shape_size == 0) {
                shape_size = shapes.at(dtype)[k].as<IntImmNode>()->value;
              } else {
                shape_size *= shapes.at(dtype)[k].as<IntImmNode>()->value;
              }
            }
            int64_t tmp_shape_size =
                Downcast<IntImm>(dst_access_ptr->args[3])->value;
            if (tmp_shape_size > shape_size) {
              Array<PrimExpr> tmp_shape;
              tmp_shape.push_back(IntImm(DataType::Int(32), tmp_shape_size));
              shapes[dtype] = tmp_shape;
            }
          } else {
            Array<PrimExpr> tmp_shape;
            tmp_shape.push_back(
                IntImm(DataType::Int(32),
                       Downcast<IntImm>(dst_access_ptr->args[3])->value));
            shapes[dtype] = tmp_shape;
          }
        }
      } else if (call->op.same_as(tl::ascend_sort()) ||
                 call->op.same_as(tl::ascend_topk())) {
        const CallNode *src_access_ptr = Downcast<Call>(call->args[2]).get();
        std::string src_buffer_name =
            src_access_ptr->args[1].as<VarNode>()->name_hint;
        const BufferNode *src_buffer_node =
            GetBufferNodeByName_(alloc_buffers, src_buffer_name);
        DataType dtype = src_buffer_node->dtype;
        if (dtype != DataType::UInt(8)) {
          // sort:  bufA (2*alignedCount) + bufC (2*alignedCount) for float
          //        (dst doubles as bufB). Half: also a float cast scratch.
          // topk:  must fit bufA + bufB + bufC = 6*alignedCount float since
          //        user dst (size 2*K) is too small to host bufB ping-pong.
          // Both paths share this allocation; use the larger (topk) sizing
          // when topk is present so the same tmp pool serves both ops.
          bool is_topk = call->op.same_as(tl::ascend_topk());
          int64_t multiplier;
          if (dtype.bytes() == 2) {
            multiplier = 16; // half: reserve enough for cast-to-float pool
          } else {
            multiplier = is_topk ? 6 : 4;
          }

          // For dynamic-shape topk, use max_actual_num (args[7]) if available.
          // args layout after Python frontend change:
          // [4] K, [5] repeatTimes, [6] actual_num (runtime), [7]
          // max_actual_num (compile-time)
          int64_t aligned_count = 0;
          if (is_topk && call->args.size() >= 8) {
            // New API: use max_actual_num for buffer sizing
            int64_t max_actual_num = Downcast<IntImm>(call->args[7])->value;
            aligned_count = ((max_actual_num + 31) / 32) * 32;
          } else {
            // Legacy API or sort: use src_access_ptr extent
            aligned_count = Downcast<IntImm>(src_access_ptr->args[3])->value;
          }

          int64_t tmp_shape_size = aligned_count * multiplier;
          if (shapes.count(dtype) > 0) {
            int64_t shape_size = 0;
            for (size_t k = 0; k < shapes.at(dtype).size(); k++) {
              if (shape_size == 0) {
                shape_size = shapes.at(dtype)[k].as<IntImmNode>()->value;
              } else {
                shape_size *= shapes.at(dtype)[k].as<IntImmNode>()->value;
              }
            }
            if (tmp_shape_size > shape_size) {
              Array<PrimExpr> tmp_shape;
              tmp_shape.push_back(IntImm(DataType::Int(32), tmp_shape_size));
              shapes[dtype] = tmp_shape;
            }
          } else {
            Array<PrimExpr> tmp_shape;
            tmp_shape.push_back(IntImm(DataType::Int(32), tmp_shape_size));
            shapes[dtype] = tmp_shape;
          }
        }
      } else if (call->op.same_as(tl::ascend_gather_mask())) {
        if (call->args[3].as<CallNode>()) {
          const CallNode *dst_access_ptr = Downcast<Call>(call->args[1]).get();
          std::string dst_buffer_name =
              dst_access_ptr->args[1].as<VarNode>()->name_hint;
          const BufferNode *src0_buffer_node =
              GetBufferNodeByName_(alloc_buffers, dst_buffer_name);
          const CallNode *src1_pattern_access_ptr =
              Downcast<Call>(call->args[3]).get();
          std::string src1_pattern_buffer_name =
              src1_pattern_access_ptr->args[1].as<VarNode>()->name_hint;
          const BufferNode *src1_pattern_buffer;
          src1_pattern_buffer =
              GetBufferNodeByName_(alloc_buffers, src1_pattern_buffer_name);
          DataType dtype = src1_pattern_buffer->dtype;
          if (dtype != DataType::UInt(8)) {
            if (shapes.count(dtype) > 0) {
              int64_t shape_size = 0;
              for (size_t k = 0; k < shapes.at(dtype).size(); k++) {
                if (shape_size == 0) {
                  shape_size = shapes.at(dtype)[k].as<IntImmNode>()->value;
                } else {
                  shape_size *= shapes.at(dtype)[k].as<IntImmNode>()->value;
                }
              }
              int64_t tmp_shape_size =
                  Downcast<IntImm>(dst_access_ptr->args[3])->value;
              if (tmp_shape_size > shape_size) {
                Array<PrimExpr> tmp_shape;
                tmp_shape.push_back(IntImm(DataType::Int(32), tmp_shape_size));
                shapes[dtype] = tmp_shape;
              }
            } else {
              Array<PrimExpr> tmp_shape;
              tmp_shape.push_back(
                  IntImm(DataType::Int(32),
                         Downcast<IntImm>(dst_access_ptr->args[3])->value));
              shapes[dtype] = tmp_shape;
            }
          }
        }
      } else if (call->op.same_as(tl::ascend_gather())) {
        // TGATHER requires tmp dtype == indices dtype (TGather.hpp
        // static_assert). Allocate a dtype-keyed tmp in tmp_bufs_. PTO
        // doesn't actually consume this tmp (see operation_config.h
        // comment), so 8 elements is enough placeholder.
        const CallNode *idx_access_ptr = Downcast<Call>(call->args[2]).get();
        std::string idx_buffer_name =
            idx_access_ptr->args[1].as<VarNode>()->name_hint;
        const BufferNode *idx_buffer_node =
            GetBufferNodeByName_(alloc_buffers, idx_buffer_name);
        DataType dtype = idx_buffer_node->dtype;
        if (shapes.count(dtype) == 0) {
          Array<PrimExpr> tmp_shape;
          tmp_shape.push_back(IntImm(DataType::Int(32), 8));
          shapes[dtype] = tmp_shape;
        }
        // If something else (e.g. xor) already requested this dtype with a
        // larger size, keep that — we only need 8 elements minimum.
      }
    }
    // Create an xor_tmp_buffer of the required type
    Array<Buffer> buffers;
    int64_t i = 1;
    for (const auto &kv : shapes) {
      const DataType &key = kv.first;
      const Array<PrimExpr> &value = kv.second;
      std::string buffer_name = buffer_name_ + "_" + std::to_string(i);
      Var tmp_buf(buffer_name, PointerType(PrimType(key), "shared"));
      Buffer buffer = Buffer(tmp_buf, key, value, {}, PrimExpr(), buffer_name,
                             -1, 0, BufferType::kDefault);
      buffers.push_back(buffer);
      i++;
    }
    return buffers;
  }

  Array<PrimExpr> GetAscendCTmpBufferSize_(Array<Buffer> alloc_buffers) {
    Array<PrimExpr> shape;
    int64_t shape_size = 0;
    for (size_t i = 0; i < calls_.size(); i++) {
      const CallNode *call = calls_[i].get();
      if (call->op.same_as(tl::ascend_sin()) ||
          call->op.same_as(tl::ascend_cos()) ||
          call->op.same_as(tl::ascend_pow())) {
        const CallNode *src_access_ptr = Downcast<Call>(call->args[1]).get();
        std::string src_buffer_name =
            src_access_ptr->args[1].as<VarNode>()->name_hint;
        const BufferNode *src_buffer_node =
            GetBufferNodeByName_(alloc_buffers, src_buffer_name);
        int64_t src_buffer_node_shape = 0;
        for (size_t j = 0; j < src_buffer_node->shape.size(); j++) {
          if (src_buffer_node_shape == 0) {
            src_buffer_node_shape =
                src_buffer_node->shape[j].as<IntImmNode>()->value;
          } else {
            src_buffer_node_shape *=
                src_buffer_node->shape[j].as<IntImmNode>()->value;
          }
        }
        int64_t tmp_shape_size =
            src_buffer_node_shape * src_buffer_node->dtype.bytes() * 2;
        if (tmp_shape_size > shape_size) {
          shape = {
              IntImm(DataType::Int(32), tmp_shape_size),
          };
          shape_size = tmp_shape_size;
        }
      } else if (call->op.same_as(tl::ascend_clamp()) ||
                 call->op.same_as(tl::ascend_clamp_max()) ||
                 call->op.same_as(tl::ascend_clamp_min())) {
        const CallNode *src_access_ptr = Downcast<Call>(call->args[2]).get();
        std::string src_buffer_name =
            src_access_ptr->args[1].as<VarNode>()->name_hint;
        const BufferNode *src_buffer_node =
            GetBufferNodeByName_(alloc_buffers, src_buffer_name);
        int64_t tmp_shape_size =
            Downcast<IntImm>(src_access_ptr->args[3])->value *
            src_buffer_node->dtype.bytes();
        if (tmp_shape_size > shape_size) {
          shape = {
              IntImm(DataType::Int(32), tmp_shape_size),
          };
          shape_size = tmp_shape_size;
        }
      } else if (call->op.same_as(tl::ascend_reduce())) {
        const CallNode *src_access_ptr = Downcast<Call>(call->args[2]).get();
        std::string src_buffer_name =
            src_access_ptr->args[1].as<VarNode>()->name_hint;
        const BufferNode *src_buffer_node =
            GetBufferNodeByName_(alloc_buffers, src_buffer_name);
        int64_t tmp_shape_size =
            Downcast<IntImm>(src_access_ptr->args[3])->value *
            src_buffer_node->dtype.bytes();
        if (tmp_shape_size > shape_size) {
          shape = {
              IntImm(DataType::Int(32), tmp_shape_size),
          };
          shape_size = tmp_shape_size;
        }
      } else if (call->op.same_as(tl::ascend_sort())) {
        const CallNode *src_access_ptr = Downcast<Call>(call->args[2]).get();
        std::string src_buffer_name =
            src_access_ptr->args[1].as<VarNode>()->name_hint;
        const BufferNode *src_buffer_node =
            GetBufferNodeByName_(alloc_buffers, src_buffer_name);
        int64_t tmp_shape_size =
            Downcast<IntImm>(src_access_ptr->args[3])->value *
            src_buffer_node->dtype.bytes() * 8;
        if (tmp_shape_size > shape_size) {
          shape = {
              IntImm(DataType::Int(32), tmp_shape_size),
          };
          shape_size = tmp_shape_size;
        }
      } else if (call->op.same_as(tl::ascend_topk())) {
        const CallNode *src_access_ptr = Downcast<Call>(call->args[2]).get();
        std::string src_buffer_name =
            src_access_ptr->args[1].as<VarNode>()->name_hint;
        const BufferNode *src_buffer_node =
            GetBufferNodeByName_(alloc_buffers, src_buffer_name);
        int64_t tmp_shape_size =
            Downcast<IntImm>(src_access_ptr->args[3])->value * 4;
        if (tmp_shape_size > shape_size) {
          shape = {
              IntImm(DataType::Int(32), tmp_shape_size),
          };
          shape_size = tmp_shape_size;
        }
      } else if (call->op.same_as(tl::ascend_sigmoid()) ||
                 call->op.same_as(tl::ascend_bilinear_interpolation()) ||
                 call->op.same_as(tl::ascend_bitwise_xor()) ||
                 call->op.same_as(tl::ascend_reducesum_experiment()) ||
                 call->op.same_as(tl::ascend_reducesum_mask_experiment())) {
        const CallNode *src_access_ptr = Downcast<Call>(call->args[1]).get();
        std::string src_buffer_name =
            src_access_ptr->args[1].as<VarNode>()->name_hint;
        const BufferNode *src_buffer_node =
            GetBufferNodeByName_(alloc_buffers, src_buffer_name);
        int64_t tmp_shape_size =
            Downcast<IntImm>(src_access_ptr->args[3])->value *
            src_buffer_node->dtype.bytes();
        if (tmp_shape_size > shape_size) {
          shape = {
              IntImm(DataType::Int(32), tmp_shape_size),
          };
          shape_size = tmp_shape_size;
        }
      } else if (call->op.same_as(tl::ascend_broadcast())) {
        const CallNode *dst_access_ptr = Downcast<Call>(call->args[1]).get();
        std::string dst_buffer_name =
            dst_access_ptr->args[1].as<VarNode>()->name_hint;
        const BufferNode *dst_buffer_node =
            GetBufferNodeByName_(alloc_buffers, dst_buffer_name);
        int64_t tmp_shape_size =
            Downcast<IntImm>(dst_access_ptr->args[3])->value *
            dst_buffer_node->dtype.bytes() / 4;
        if (tmp_shape_size > shape_size) {
          shape = {
              IntImm(DataType::Int(32), tmp_shape_size),
          };
          shape_size = tmp_shape_size;
        }
      } else if (call->op.same_as(tl::ascend_round())) {
        const CallNode *dst_access_ptr = Downcast<Call>(call->args[1]).get();
        std::string dst_buffer_name =
            dst_access_ptr->args[1].as<VarNode>()->name_hint;
        const BufferNode *dst_buffer_node =
            GetBufferNodeByName_(alloc_buffers, dst_buffer_name);
        int64_t tmp_size = 256;
        int64_t tmp_shape_size = std::max(
            tmp_size, Downcast<IntImm>(dst_access_ptr->args[3])->value *
                          dst_buffer_node->dtype.bytes());
        if (tmp_shape_size > shape_size) {
          shape = {
              IntImm(DataType::Int(32), tmp_shape_size),
          };
          shape_size = tmp_shape_size;
        }
      } else if (call->op.same_as(tl::ascend_merge_sort())) {
        const CallNode *dst_access_ptr = Downcast<Call>(call->args[2]).get();
        std::string dst_buffer_name =
            dst_access_ptr->args[1].as<VarNode>()->name_hint;
        const BufferNode *dst_buffer_node =
            GetBufferNodeByName_(alloc_buffers, dst_buffer_name);
        int64_t tmp_shape_size =
            Downcast<IntImm>(dst_access_ptr->args[3])->value *
            dst_buffer_node->dtype.bytes();
        if (tmp_shape_size > shape_size) {
          shape = {
              IntImm(DataType::Int(32), tmp_shape_size),
          };
          shape_size = tmp_shape_size;
        }
      }
    }
    return shape;
  }

  Array<PrimExpr> GetPTOTmpBufferSize_(Array<Buffer> alloc_buffers) {
    Array<PrimExpr> shape;
    int64_t shape_size = 0;
    for (size_t i = 0; i < calls_.size(); i++) {
      const CallNode *call = calls_[i].get();
      // pto uses formula calculate tmp_buffer size
      if (call->op.same_as(tl::ascend_reduce())) {
        const CallNode *src_access_ptr = Downcast<Call>(call->args[2]).get();
        std::string op_name = Downcast<StringImm>(call->args[0])->value;
        auto template_params = ExtractTemplateParamsForSliceBuffer(op_name);
        int param4_int = std::get<2>(template_params);
        std::string mode = "";
        if (param4_int == -1) {
          mode = "row";
        } else if (param4_int == 0) {
          mode = "col";
        } else {
          ICHECK(false)
              << "Only row-wise or column-wise reduce operations are "
                 "supported. Row direction is denoted by -1, and column "
                 "direction by 0.";
        }

        if (op_name.find("reduce_sum") != std::string::npos) {
          op_name = (mode == "row") ? "TROWSUM" : "TCOLSUM";
        } else if (op_name.find("reduce_max") != std::string::npos) {
          op_name = (mode == "row") ? "TROWMAX" : "TCOLMAX";
        } else if (op_name.find("reduce_min") != std::string::npos) {
          op_name = (mode == "row") ? "TROWMIN" : "TCOLMIN";
        } else {
          ICHECK(false) << "not support reduce type: " << op_name;
        }
        // get src_buffer valid_row and valid_col
        std::string src_buffer_name =
            src_access_ptr->args[1].as<VarNode>()->name_hint;
        const BufferNode *src_buffer_node;
        int64_t valid_row;
        int64_t valid_col;
        src_buffer_node = GetBufferNodeByName_(alloc_buffers, src_buffer_name);

        if (src_buffer_node->shape.size() == 1) {
          valid_row = 1;
          valid_col = src_buffer_node->shape[0].as<IntImmNode>()->value;
        } else if (src_buffer_node->shape.size() == 2) {
          valid_row = src_buffer_node->shape[0].as<IntImmNode>()->value;
          valid_col = src_buffer_node->shape[1].as<IntImmNode>()->value;
        } else if (src_buffer_node->shape.size() == 3) {
          valid_row = src_buffer_node->shape[1].as<IntImmNode>()->value;
          valid_col = src_buffer_node->shape[2].as<IntImmNode>()->value;
        }
        if (op_name == "TROWMAX" || op_name == "TROWMIN" ||
            op_name == "TROWSUM") {
          const int64_t dtype_bytes = src_buffer_node->dtype.bytes();
          const int64_t tmp_col =
              GetPtoRowReduceTmpCols(valid_col, dtype_bytes);
          const int64_t tmp_shape_size = valid_row * tmp_col * dtype_bytes;
          if (tmp_shape_size > shape_size) {
            Array<PrimExpr> tmp_shape = {
                IntImm(DataType::Int(32), tmp_shape_size),
            };
            shape = tmp_shape;
            shape_size = tmp_shape_size;
          }
        } else if (op_name == "TCOLSUM") {
          int64_t tmp_shape_size = valid_row * valid_col / 2;
          if (tmp_shape_size > shape_size) {
            Array<PrimExpr> tmp_shape = {
                IntImm(DataType::Int(32), valid_row),
                IntImm(DataType::Int(32), valid_col / 2),
            };
            shape = tmp_shape;
            shape_size = tmp_shape_size;
          }
        } else {
          if (shape.size() == 0) {
            shape = {IntImm(DataType::Int(32), 1)};
          }
        }
      } else if (call->op.same_as(tl::ascend_bitwise_xor())) {
        const CallNode *src_access_ptr = Downcast<Call>(call->args[1]).get();
        std::string src_buffer_name =
            src_access_ptr->args[1].as<VarNode>()->name_hint;
        const BufferNode *src_buffer_node;
        src_buffer_node = GetBufferNodeByName_(alloc_buffers, src_buffer_name);
        DataType dtype = src_buffer_node->dtype;
        // If it's uint8 type, use a shared tmp_buffer; otherwise, handle
        // separately and store tmp_buffer for other dtypes individually.
        if (dtype == DataType::UInt(8)) {
          int64_t tmp_shape_size =
              Downcast<IntImm>(src_access_ptr->args[3])->value *
              src_buffer_node->dtype.bytes();
          if (tmp_shape_size > shape_size) {
            shape = {
                IntImm(DataType::Int(32), tmp_shape_size),
            };
            shape_size = tmp_shape_size;
          }
        }
      } else if (call->op.same_as(tl::ascend_sigmoid()) ||
                 call->op.same_as(tl::ascend_pow()) ||
                 call->op.same_as(tl::ascend_round()) ||
                 call->op.same_as(tl::ascend_broadcast())) {
        const CallNode *src_access_ptr = Downcast<Call>(call->args[1]).get();
        if (shape_size == 0) {
          std::string src_buffer_name =
              src_access_ptr->args[1].as<VarNode>()->name_hint;
          const BufferNode *src_buffer_node =
              GetBufferNodeByName_(alloc_buffers, src_buffer_name);
          int64_t tmp_shape_size = 1;
          Array<PrimExpr> tmp_shape;
          shape = {
              IntImm(DataType::Int(32), tmp_shape_size),
          };
          shape_size = tmp_shape_size;
        }
      } else if (call->op.same_as(tl::ascend_merge_sort())) {
        const CallNode *dst_access_ptr = Downcast<Call>(call->args[2]).get();
        std::string dst_buffer_name =
            dst_access_ptr->args[1].as<VarNode>()->name_hint;
        const BufferNode *dst_buffer_node =
            GetBufferNodeByName_(alloc_buffers, dst_buffer_name);
        DataType dtype = dst_buffer_node->dtype;
        if (dtype == DataType::UInt(8)) {
          int64_t tmp_shape_size =
              Downcast<IntImm>(dst_access_ptr->args[3])->value * 4;
          Array<PrimExpr> tmp_shape;
          shape = {
              IntImm(DataType::Int(32), tmp_shape_size),
          };
          shape_size = tmp_shape_size;
        }
      } else if (call->op.same_as(tl::ascend_clamp()) ||
                 call->op.same_as(tl::ascend_clamp_max()) ||
                 call->op.same_as(tl::ascend_clamp_min())) {
        const CallNode *src_access_ptr = Downcast<Call>(call->args[2]).get();
        if (shape_size == 0) {
          std::string src_buffer_name =
              src_access_ptr->args[1].as<VarNode>()->name_hint;
          const BufferNode *src_buffer_node =
              GetBufferNodeByName_(alloc_buffers, src_buffer_name);
          int64_t tmp_shape_size =
              Downcast<IntImm>(src_access_ptr->args[3])->value *
              src_buffer_node->dtype.bytes();
          Array<PrimExpr> tmp_shape;
          shape = {
              IntImm(DataType::Int(32), tmp_shape_size),
          };
          shape_size = tmp_shape_size;
        }
      } else if (call->op.same_as(tl::ascend_select())) {
        const CallNode *src_access_ptr = Downcast<Call>(call->args[0]).get();
        std::string src_buffer_name =
            src_access_ptr->args[1].as<VarNode>()->name_hint;
        const BufferNode *src_buffer_node =
            GetBufferNodeByName_(alloc_buffers, src_buffer_name);
        int64_t tmp_shape_size =
            Downcast<IntImm>(src_access_ptr->args[3])->value *
            src_buffer_node->dtype.bytes();
        Array<PrimExpr> tmp_shape;
        if (tmp_shape_size > shape_size) {
          shape = {
              IntImm(DataType::Int(32), tmp_shape_size),
          };
          shape_size = tmp_shape_size;
        }
      } else if (call->op.same_as(tl::ascend_gather_mask())) {
        if (call->args[3].as<CallNode>()) {
          const CallNode *src0_access_ptr = Downcast<Call>(call->args[1]).get();
          const CallNode *src1_pattern_access_ptr =
              Downcast<Call>(call->args[3]).get();
          std::string src1_pattern_buffer_name =
              src1_pattern_access_ptr->args[1].as<VarNode>()->name_hint;
          const BufferNode *src1_pattern_buffer;
          src1_pattern_buffer =
              GetBufferNodeByName_(alloc_buffers, src1_pattern_buffer_name);
          DataType dtype = src1_pattern_buffer->dtype;
          if (dtype == DataType::UInt(8)) {
            std::string src0_buffer_name =
                src0_access_ptr->args[1].as<VarNode>()->name_hint;
            const BufferNode *src0_buffer_node =
                GetBufferNodeByName_(alloc_buffers, src0_buffer_name);
            int64_t tmp_shape_size =
                Downcast<IntImm>(src0_access_ptr->args[3])->value *
                src0_buffer_node->dtype.bytes();
            Array<PrimExpr> tmp_shape;
            if (tmp_shape_size > shape_size) {
              shape = {
                  IntImm(DataType::Int(32), tmp_shape_size),
              };
              shape_size = tmp_shape_size;
            }
          }
        } else {
          if (shape_size == 0) {
            shape = {
                IntImm(DataType::Int(32), 1),
            };
            shape_size = 1;
          } else {
            shape = {
                IntImm(DataType::Int(32), shape_size),
            };
          }
        }
      }
    }
    return shape;
  }

  const BufferNode *GetBufferNodeByName_(Array<Buffer> alloc_buffers,
                                         std::string src_buffer_name) {
    const BufferNode *buffer = nullptr;
    for (size_t j = 0; j < alloc_buffers.size(); j++) {
      if (alloc_buffers[j].get()->name == src_buffer_name) {
        buffer = alloc_buffers[j].get();
        break;
      }
    }
    return buffer;
  }

  std::tuple<int, int, int, bool>
  ExtractTemplateParamsForSliceBuffer(const std::string &op_name) {
    int second_param = 0;
    int third_param = 0;
    int forth_param = 0;
    size_t left = op_name.find('<');
    size_t right = op_name.find('>');

    if (left == std::string::npos || right == std::string::npos ||
        left >= right) {
      return std::make_tuple(second_param, third_param, forth_param, false);
    }

    std::string params_str = op_name.substr(left + 1, right - left - 1);
    std::vector<std::string> params;
    size_t start = 0;
    size_t comma = 0;
    while ((comma = params_str.find(',', start)) != std::string::npos) {
      std::string param = params_str.substr(start, comma - start);
      param.erase(0, param.find_first_not_of(" \t"));
      param.erase(param.find_last_not_of(" \t") + 1);
      params.push_back(param);
      start = comma + 1;
    }

    std::string last_param = params_str.substr(start);
    last_param.erase(0, last_param.find_first_not_of(" \t"));
    last_param.erase(last_param.find_last_not_of(" \t") + 1);
    params.push_back(last_param);

    if (params.size() >= 4) {
      try {
        second_param = std::stoi(params[1]);
        third_param = std::stoi(params[2]);
        forth_param = std::stoi(params[3]);
        return std::make_tuple(second_param, third_param, forth_param, true);
      } catch (const std::exception &e) {
        return std::make_tuple(second_param, third_param, forth_param, false);
      }
    } else {
      ICHECK(false) << "reduce params less than 4.";
    }
    return std::make_tuple(second_param, third_param, forth_param, false);
  }

  struct BufferInfo {
    Array<PrimExpr> shape;
    DataType dtype;
  };

  std::string target_;
  std::vector<Call> calls_;
  const std::string buffer_name_ = "tmp_ub";
  Buffer tmp_buf_;
  Array<Buffer> tmp_bufs_;
  Buffer reduce_out_tmp_buf_;
  bool inject_enabled_ = true;

  Buffer FindUserProvidedBuffer_(const Array<Buffer> &alloc_buffers,
                                 const std::string &name) {
    for (const auto &buf : alloc_buffers) {
      if (buf->name == name) {
        return buf;
      }
    }
    return Buffer();
  }

  int64_t BufferSizeInBytes_(const Buffer &buf) {
    int64_t size = 1;
    for (const auto &dim : buf->shape) {
      const auto *imm = dim.as<IntImmNode>();
      ICHECK(imm) << "Buffer extent must be integer constant for tmp buffer "
                  << buf->name;
      size *= imm->value;
    }
    return size * buf->dtype.bytes();
  }

  void CollectUserProvidedTmpBuffers_(const Array<Buffer> &alloc_buffers) {
    tmp_buf_ = FindUserProvidedBuffer_(alloc_buffers, buffer_name_);

    bool needs_tmp = !calls_.empty();
    if (needs_tmp && !tmp_buf_.defined()) {
      LOG(FATAL) << "InjectTmpBuffer pass is disabled "
                 << "(TL_ASCEND_INJECT_TMP_BUFFER=False) but the kernel "
                 << "contains operations (reduce/broadcast/sigmoid/etc.) "
                 << "that require a temporary buffer. Please manually "
                 << "allocate a UB buffer named '" << buffer_name_ << "' "
                 << "with uint8 dtype and sufficient size, e.g.: "
                 << "tmp_ub = T.alloc_ub((size,), 'uint8')";
    }

    if (tmp_buf_.defined()) {
      std::string scope = GetPtrStorageScope(tmp_buf_->data);
      if (scope != "shared") {
        LOG(FATAL) << "User-provided tmp buffer '" << buffer_name_
                   << "' must be in 'shared' (UB) scope, but got '" << scope
                   << "'. Use T.alloc_ub() to allocate a UB buffer.";
      }
      if (tmp_buf_->dtype != DataType::UInt(8)) {
        LOG(FATAL) << "User-provided tmp buffer '" << buffer_name_
                   << "' must be uint8 dtype, but got " << tmp_buf_->dtype
                   << ".";
      }
      ValidateTmpBufferSize_(alloc_buffers);
    }

    for (int i = 1;; i++) {
      std::string name = buffer_name_ + "_" + std::to_string(i);
      Buffer buf = FindUserProvidedBuffer_(alloc_buffers, name);
      if (buf.defined()) {
        tmp_bufs_.push_back(buf);
      } else {
        break;
      }
    }

    if ("pto" == target_) {
      reduce_out_tmp_buf_ =
          FindUserProvidedBuffer_(alloc_buffers, buffer_name_ + "_reduce_out");
    }

    bool needs_dtype_tmp = false;
    for (const auto &call : calls_) {
      if (call->op.same_as(tl::ascend_sort()) ||
          call->op.same_as(tl::ascend_topk()) ||
          call->op.same_as(tl::ascend_bitwise_xor()) ||
          call->op.same_as(tl::ascend_merge_sort()) ||
          call->op.same_as(tl::ascend_gather()) ||
          call->op.same_as(tl::ascend_gather_mask())) {
        needs_dtype_tmp = true;
        break;
      }
    }
    if (needs_dtype_tmp && tmp_bufs_.empty()) {
      LOG(FATAL) << "InjectTmpBuffer pass is disabled but sort/topk/xor/"
                 << "merge_sort operations require dtype-specific tmp "
                 << "buffers. Please also allocate buffers named '"
                 << buffer_name_ << "_1', '" << buffer_name_
                 << "_2', etc. with appropriate dtypes.";
    }
  }

  void ValidateTmpBufferSize_(const Array<Buffer> &alloc_buffers) {
    Array<PrimExpr> required_shape;
    if ("pto" == target_) {
      required_shape = GetPTOTmpBufferSize_(alloc_buffers);
    } else if ("ascendc" == target_ || "auto" == target_) {
      required_shape = GetAscendCTmpBufferSize_(alloc_buffers);
    }

    if (required_shape.empty()) {
      return;
    }

    int64_t required_bytes = 1;
    for (const auto &dim : required_shape) {
      const auto *imm = dim.as<IntImmNode>();
      if (!imm) {
        return;
      }
      required_bytes *= imm->value;
    }

    int64_t user_bytes = BufferSizeInBytes_(tmp_buf_);
    if (user_bytes < required_bytes) {
      LOG(FATAL) << "User-provided tmp buffer '" << buffer_name_
                 << "' is too small: " << user_bytes << " bytes, but at "
                 << "least " << required_bytes << " bytes are required.";
    }
  }
};

tvm::transform::Pass InjectTmpBuffer(Target target) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    bool inject_enabled =
        ctx->GetConfig<Bool>(kAscendInjectTmpBuffer, Bool(true)).value();
    return TmpBufferInjector::TmpBufferInject(std::move(f), target,
                                              inject_enabled);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectTmpBuffer", {});
}

TVM_REGISTER_GLOBAL("tl.transform.InjectTmpBuffer")
    .set_body_typed(InjectTmpBuffer);

} // namespace tl
} // namespace tvm
