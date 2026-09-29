// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#ifndef TVM_TL_TRANSFORM_COMMON_ASCEND_VID_REDUCTION_H_
#define TVM_TL_TRANSFORM_COMMON_ASCEND_VID_REDUCTION_H_

#include <tvm/tir/function.h>

#include <unordered_map>

namespace tvm {
namespace tl {

struct AscendVidReductionInfo {
  int source_divisor{1};
  // Describes whether the allocation's first dimension will be divided.
  // The actual pass clamps static dimensions to one, and only divides
  // access_ptr extents when their offset is zero.
  int workspace_divisor{1};
};

using AscendVidReductionPlan =
    std::unordered_map<const tir::CallNode *, AscendVidReductionInfo>;

// Analyze an already scope-inferred function without changing its IR. Call
// identities remain valid until its body is rewritten.
AscendVidReductionPlan AnalyzeAscendVidReduction(const tir::PrimFunc &func);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_ASCEND_VID_REDUCTION_H_
