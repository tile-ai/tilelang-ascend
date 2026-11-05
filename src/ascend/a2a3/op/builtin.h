// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#ifndef TVM_TL_ASCEND_A2A3_OP_BUILTIN_H_
#define TVM_TL_ASCEND_A2A3_OP_BUILTIN_H_

#include "ascend/op/builtin.h"
#include "op/builtin.h"

namespace tvm {
namespace tl {
namespace a2a3 {

TVM_DLL const Op &a2a3_const_data();
TVM_DLL const Op &ascend_scalar_op();
TVM_DLL const Op &ascend_unary_op();
TVM_DLL const Op &ascend_binary_op();
TVM_DLL const Op &ascend_binary_ops();

} // namespace a2a3
} // namespace tl
} // namespace tvm
#endif // TVM_TL_ASCEND_A2A3_OP_BUILTIN_H_
