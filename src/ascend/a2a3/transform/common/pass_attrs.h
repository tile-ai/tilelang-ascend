// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#ifndef TVM_TL_ASCEND_A2A3_TRANSFORM_COMMON_PASS_ATTRS_H_
#define TVM_TL_ASCEND_A2A3_TRANSFORM_COMMON_PASS_ATTRS_H_

namespace tvm::tl::a2a3 {

// TA 0deb8b93 src/op/builtin.h. This differs from mainline's safe_value_map.
inline constexpr const char *kPaddingMap = "padding_map";

} // namespace tvm::tl::a2a3

#endif // TVM_TL_ASCEND_A2A3_TRANSFORM_COMMON_PASS_ATTRS_H_
