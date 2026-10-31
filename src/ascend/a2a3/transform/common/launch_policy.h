// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#ifndef TVM_TL_ASCEND_A2A3_TRANSFORM_COMMON_LAUNCH_POLICY_H_
#define TVM_TL_ASCEND_A2A3_TRANSFORM_COMMON_LAUNCH_POLICY_H_

#include <tvm/ffi/container/map.h>

#include "ascend/a2a3/transform/common/attr.h"

namespace tvm::tl::a2a3 {

// TA's default launch used blockIdx.y for manually handled vid, while the
// developer form used threadIdx.x for automatic partitioning. Both become
// cthread in TIRX, so retain the original producer's distinction. Unmarked
// direct IR has the former threadIdx.x consumption contract.
inline bool
UsesVidThreadBinding(const ffi::Map<ffi::String, ffi::Any> &annotations) {
  return !annotations.count(tilelang_is_npu_kernel_frame);
}

} // namespace tvm::tl::a2a3

#endif // TVM_TL_ASCEND_A2A3_TRANSFORM_COMMON_LAUNCH_POLICY_H_
