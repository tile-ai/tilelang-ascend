// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#ifndef TVM_TL_ASCEND_A2A3_CODEGEN_UTILS_H_
#define TVM_TL_ASCEND_A2A3_CODEGEN_UTILS_H_
#include <tvm/runtime/data_type.h>
namespace tvm::tl::a2a3 {
int TVMDataTypeToACL(const DataType &dtype);
inline constexpr const char *kAscendExceptionDump = "tl.ascend_exception_dump";
} // namespace tvm::tl::a2a3
#endif // TVM_TL_ASCEND_A2A3_CODEGEN_UTILS_H_
