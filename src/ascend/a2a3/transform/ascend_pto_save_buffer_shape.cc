// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file simple_buffer_shape_collector.cc
 * \brief Simple buffer shape collection pass
 */

#include <utility>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {
namespace a2a3 {

using namespace tirx;
using namespace ffi;
using namespace tirx::transform;

// 简单的buffer shape收集器
class SimpleBufferShapeCollector : public StmtVisitor {
public:
  static PrimFunc Collect(PrimFunc f) {
    SimpleBufferShapeCollector collector(f);

    return WithAttr(std::move(f), "buffer_shapess", collector.shape_map_);
  }

private:
  explicit SimpleBufferShapeCollector(const PrimFunc &func) {
    // 1. 首先从buffer_map收集
    const PrimFuncNode *fptr = func.get();
    for (const auto &kv : fptr->buffer_map) {
      Var buffer_var = kv.second->data;
      shape_map_.Set(buffer_var, kv.second->shape);
    }

    // 2. 然后遍历函数体收集Allocate节点
    this->VisitStmt(func->body);
  }

  // Allocate and AllocateConst shape metadata is carried by the
  // explicit TIRX allocation's Buffer; there is no nested allocation body.
  void VisitStmt_(const AllocBufferNode *op) override {
    shape_map_.Set(op->buffer->data, op->buffer->shape);
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const SBlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers) {
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
  return CreatePrimFuncPass(pass_func, 0, "tl.A2A3CollectBufferShapes", {});
}

// TVM注册
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.A2A3CollectBufferShapes",
                        CollectBufferShapes);
}

} // namespace a2a3
} // namespace tl
} // namespace tvm
