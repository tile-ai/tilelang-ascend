// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file simple_buffer_shape_collector.cc
 * \brief Simple buffer shape collection pass
 */

#include <iostream>
#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <stack>

#include "arith/ir_mutator_with_analyzer.h"
#include "tir/analysis/var_use_def_analysis.h"
#include "tir/transforms/ir_utils.h"

#include <tvm/tir/analysis.h>
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

// 简单的buffer shape收集器
class SimpleBufferShapeCollector : public StmtVisitor {
 public:
  static PrimFunc Collect(PrimFunc f) {
    SimpleBufferShapeCollector collector(f);
    
    PrimFuncNode* fptr = f.CopyOnWrite();
    auto fn_attr = fptr->attrs.CopyOnWrite();
    fn_attr->dict.Set("buffer_shapess", collector.shape_map_);
    
    return f;
  }

 private:
  explicit SimpleBufferShapeCollector(const PrimFunc& func) {
    // 1. 首先从buffer_map收集
    const PrimFuncNode* fptr = func.get();
    for (const auto& kv : fptr->buffer_map) {
      Var buffer_var = kv.second->data;
      shape_map_.Set(buffer_var, kv.second->shape);
    }
    
    // 2. 然后遍历函数体收集Allocate节点
    this->VisitStmt(func->body);
  }
  
  void VisitStmt_(const AllocateNode* op) override {
    Array<PrimExpr> shape;
    for (const auto& dim : op->extents) {
      shape.push_back(dim);
    }
    shape_map_.Set(op->buffer_var, shape);
    StmtVisitor::VisitStmt_(op);
  }
  
  void VisitStmt_(const AllocateConstNode* op) override {
    Array<PrimExpr> shape;
    for (const auto& dim : op->extents) {
      shape.push_back(dim);
    }
    shape_map_.Set(op->buffer_var, shape);
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BlockNode *op) final {
    for (const Buffer& buffer : op->alloc_buffers) {
      shape_map_.Set(buffer->data, buffer->shape);
    }
    StmtVisitor::VisitStmt_(op);
  }
  
  Map<Var, Array<PrimExpr>> shape_map_;
};

// Pass函数实现
tvm::transform::Pass CollectBufferShapes() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    return SimpleBufferShapeCollector::Collect(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.CollectBufferShapes", {});
}


// TVM注册
TVM_REGISTER_GLOBAL("tl.transform.CollectBufferShapes")
    .set_body_typed(CollectBufferShapes);

}  // namespace tl
}  // namespace tvm