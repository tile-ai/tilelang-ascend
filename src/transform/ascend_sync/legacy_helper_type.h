// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

#pragma once

#include <string>

namespace tvm::tl::ascend_sync {

// AscendCopy::Lower and the GEMM emitter spell the signed 32-bit element
// type as `int`. Normalize that ABI spelling before applying each helper's
// existing supported-type contract; this does not admit additional dtypes.
inline std::string CanonicalHelperElementType(const std::string &type) {
  return type == "int" ? "int32_t" : type;
}

} // namespace tvm::tl::ascend_sync
