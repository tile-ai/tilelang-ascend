// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#ifndef TVM_TL_ASCEND_A2A3_PARTITION_THREAD_H_
#define TVM_TL_ASCEND_A2A3_PARTITION_THREAD_H_

#include "launch_policy.h"

#include <tvm/ir/cast.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm::tl::a2a3 {

// TA ir.cc emitted developer vid as threadIdx.x, but default vid as
// blockIdx.y. Only the former was consumed by LayoutInference/LowerTileOp.
// The migrated producer unifies both to cthread and retains the launch marker.
inline bool IsPartitionThread(const tirx::AttrStmtNode *attr) {
  auto iv = tvm::Downcast<tirx::IterVar>(attr->node);
  if (iv->thread_tag != "cthread") {
    return false;
  }
  class KernelModeFinder : public tirx::StmtVisitor {
  public:
    bool found{false};
    ffi::Map<ffi::String, ffi::Any> annotations;

  private:
    void VisitStmt_(const tirx::SBlockNode *block) final {
      if (found)
        return;
      if (block->annotations.count("tilelang.is_npu_kernel_frame_dev_mode")) {
        found = true;
        annotations = block->annotations;
      } else if (block->annotations.count("tilelang.is_npu_kernel_frame")) {
        found = true;
        annotations = block->annotations;
      } else {
        StmtVisitor::VisitStmt_(block);
      }
    }
  } finder;
  finder(attr->body);
  return UsesVidThreadBinding(finder.annotations);
}

} // namespace tvm::tl::a2a3
#endif // TVM_TL_ASCEND_A2A3_PARTITION_THREAD_H_
