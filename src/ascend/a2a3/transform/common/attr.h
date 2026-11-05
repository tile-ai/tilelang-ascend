// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file attr.h
 * \brief Check attributes of the IR
 */

#ifndef TVM_TL_ASCEND_A2A3_TRANSFORM_COMMON_ATTR_H_
#define TVM_TL_ASCEND_A2A3_TRANSFORM_COMMON_ATTR_H_

namespace tvm {
namespace tl {
namespace a2a3 {

constexpr const char *MainBlockName = "tilelang_root";

constexpr const char *tilelang_is_cpu_kernel_frame =
    "tilelang.is_cpu_kernel_frame";

constexpr const char *tilelang_is_npu_kernel_frame =
    "tilelang.is_npu_kernel_frame";

constexpr const char *tilelang_is_npu_kernel_frame_dev_mode =
    "tilelang.is_npu_kernel_frame_dev_mode";

constexpr const char *cv_1_1 = "cv_1_1";

constexpr const char *cv_1_2 = "cv_1_2";

constexpr const char *kInitialBufferShapes = "initial_buffer_shapes";

constexpr const char *kLogicBufferShapes = "logic_buffer_shapes";

} // namespace a2a3
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_A2A3_TRANSFORM_COMMON_ATTR_H_
