// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#ifndef TVM_TL_ASCEND_A2A3_LAYOUT_ASCEND_LAYOUT_H_
#define TVM_TL_ASCEND_A2A3_LAYOUT_ASCEND_LAYOUT_H_

#include <string>

#include "layout/layout.h"

namespace tvm::tl::a2a3 {

// Numeric tags and emitted template names retain the TA Layout ABI.
enum class AscendLayoutKind {
  kRowMajor = 0,
  kColMajor = 1,
  kzN = 2,
  kzZ = 3,
  knZ = 4,
};

/// Attach an explicit TA layout tag without changing its coordinate map.
Layout WrapAscendLayout(const Layout &layout, AscendLayoutKind kind);

/// Return the TA template spelling. Untagged external layouts require a bridge.
std::string AscendLayoutStr(const Layout &layout);

/// TA allocation geometry, including whole fractals for static matrix tails.
ffi::Array<PrimExpr> PhysicalAllocationShape(const Layout &layout,
                                             DataType dtype);

} // namespace tvm::tl::a2a3

#endif // TVM_TL_ASCEND_A2A3_LAYOUT_ASCEND_LAYOUT_H_
