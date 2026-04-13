// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include <tvm/ir/op.h>

#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

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
                     Array<Buffer> &tmp_buffers) {
    CallNodeModifier modifier;
    modifier.target_ = Downcast<String>(target.get()->attrs["model"]);
    if ("pto" == modifier.target_) {
      modifier.tmp_arg_ops_ = pto_tmp_arg_ops;
    } else if ("ascendc" == modifier.target_ || "auto" == modifier.target_) {
      modifier.tmp_arg_ops_ = ascendc_tmp_arg_ops;
    }
    modifier.tmp_buf_ = tmp_buffer;
    modifier.tmp_bufs_ = tmp_buffers;
    return modifier.AddTmpArg(f->body);
  }

private:
  Stmt AddTmpArg(const Stmt &stmt) { return VisitStmt(stmt); }

  PrimExpr VisitExpr_(const CallNode *op) override {
    if (const auto *op_node = op->op.as<OpNode>()) {
      if (tmp_arg_ops_.count(op_node) > 0) {
        int64_t tmp_buffer_param_offset = tmp_arg_ops_.at(op_node);
        if (op->op.same_as(tl::ascend_sigmoid()) ||
            op->op.same_as(tl::ascend_pow()) ||
            op->op.same_as(tl::ascend_bitwise_xor()) ||
            op->op.same_as(tl::ascend_merge_sort())) {
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
    } else if ("pto" == target_ && op->op.same_as(tl::ascend_bitwise_xor()) &&
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
    } else {
      tmp_buffer = tmp_buf_;
    }

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

  Buffer tmp_buf_;
  Array<Buffer> tmp_bufs_;
  std::string target_;
  std::unordered_map<const tvm::OpNode *, int64_t> tmp_arg_ops_;
};

class TmpBufferInjector : public StmtExprMutator {
public:
  static PrimFunc TmpBufferInject(PrimFunc f, Target target) {
    TmpBufferInjector injector;
    injector.target_ = Downcast<String>(target.get()->attrs["model"]);
    PrimFuncNode *fptr = f.CopyOnWrite();
    injector.calls_ = CallNodeCollector::Collect(f, target);
    Stmt new_body = injector.inject(f->body);
    fptr->body = new_body;
    new_body = CallNodeModifier::Modify(f, target, injector.tmp_buf_,
                                        injector.tmp_bufs_);
    fptr->body = new_body;
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

      tmp_buf_ = createTmpBuffer_(op->alloc_buffers);
      if (tmp_buf_.defined()) {
        new_alloc_buffers.push_back(tmp_buf_);
      }

      if ("pto" == target_) {
        tmp_bufs_ = createPTOXORAndMergeSortTmpBuffer_(op->alloc_buffers);
        for (const Buffer &tmp_buffer : tmp_bufs_) {
          new_alloc_buffers.push_back(tmp_buffer);
        }
      } else if ("ascendc" == target_ || "auto" == target_) {
        tmp_bufs_ = createASCTopKAndSortTmpBuffer_(op->alloc_buffers);
        for (const Buffer &tmp_buffer : tmp_bufs_) {
          new_alloc_buffers.push_back(tmp_buffer);
        }
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
  createPTOXORAndMergeSortTmpBuffer_(Array<Buffer> alloc_buffers) {
    std::unordered_map<DataType, Array<PrimExpr>> shapes;
    // Iterate over the stored CallNodes, find the corresponding xor and merge_sort
    // and allocate tmp_ub for them (requires tmp_ub of the corresponding datatype)
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
                Downcast<IntImm>(dst_access_ptr->args[3])->value * 4;
            if (tmp_shape_size > shape_size) {
              Array<PrimExpr> tmp_shape;
              for (size_t k = 0; k < dst_buffer_node->shape.size(); k++) {
                tmp_shape.push_back(
                    IntImm(DataType::Int(32),
                           dst_buffer_node->shape[k].as<IntImmNode>()->value));
              }
              shapes[dtype] = tmp_shape;
            }
          } else {
            Array<PrimExpr> tmp_shape;
            for (size_t k = 0; k < dst_buffer_node->shape.size(); k++) {
              tmp_shape.push_back(
                  IntImm(DataType::Int(32),
                         dst_buffer_node->shape[k].as<IntImmNode>()->value));
            }
            shapes[dtype] = tmp_shape;
          }
        }
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
        }
      } else if (call->op.same_as(tl::ascend_reduce())) {
        const CallNode *src_access_ptr = Downcast<Call>(call->args[2]).get();
        std::string src_buffer_name =
            src_access_ptr->args[1].as<VarNode>()->name_hint;
        const BufferNode *src_buffer_node =
            GetBufferNodeByName_(alloc_buffers, src_buffer_name);
        int64_t tmp_shape_size =
            Downcast<IntImm>(src_access_ptr->args[3])->value *
            src_buffer_node->dtype.bytes() / 2;
        if (tmp_shape_size > shape_size) {
          shape = {
              IntImm(DataType::Int(32), tmp_shape_size),
          };
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
        if (op_name == "TROWMAX" || op_name == "TROWMIN") {
          int64_t tmp_shape_size = valid_row;
          if (valid_row > shape_size) {
            Array<PrimExpr> tmp_shape = {IntImm(DataType::Int(32), valid_row)};
            shape = tmp_shape;
            shape_size = tmp_shape_size;
          }
        } else if (op_name == "TROWSUM" || op_name == "TCOLSUM") {
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
          int64_t tmp_shape_size =
              Downcast<IntImm>(src_access_ptr->args[3])->value *
              src_buffer_node->dtype.bytes() / 2;
          Array<PrimExpr> tmp_shape;
          shape = {
              IntImm(DataType::Int(32), tmp_shape_size),
          };
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
};

tvm::transform::Pass InjectTmpBuffer(Target target) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return TmpBufferInjector::TmpBufferInject(std::move(f), target);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectTmpBuffer", {});
}

TVM_REGISTER_GLOBAL("tl.transform.InjectTmpBuffer")
    .set_body_typed(InjectTmpBuffer);

} // namespace tl
} // namespace tvm