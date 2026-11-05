// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#ifndef TVM_TL_ASCEND_A2A3_TRANSFORM_COMMON_ASCEND_VID_REDUCTION_H_
#define TVM_TL_ASCEND_A2A3_TRANSFORM_COMMON_ASCEND_VID_REDUCTION_H_

#include <tvm/tirx/function.h>

#include <unordered_map>

namespace tvm {
namespace tl {
namespace a2a3 {

struct AscendVidReductionInfo {
  int source_divisor{1};
  // Describes whether the allocation's first dimension will be divided.
  // The actual pass clamps static dimensions to one, and only divides
  // access_ptr extents when their offset is zero.
  int workspace_divisor{1};
};

using AscendVidReductionPlan =
    std::unordered_map<tirx::Call, AscendVidReductionInfo, ffi::ObjectPtrHash,
                       ffi::ObjectPtrEqual>;

// Analyze an already scope-inferred function without changing its IR. Call
// identities remain valid until its body is rewritten.
AscendVidReductionPlan AnalyzeAscendVidReduction(const tirx::PrimFunc &func);

} // namespace a2a3
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_A2A3_TRANSFORM_COMMON_ASCEND_VID_REDUCTION_H_
