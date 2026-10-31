// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#include "utils.h"
#include <tvm/ir/transform.h>
namespace tvm::tl::a2a3 {
TVM_REGISTER_PASS_CONFIG_OPTION(kAscendExceptionDump, Bool);
int TVMDataTypeToACL(const DataType &dtype) {
  if (dtype.is_float()) {
    switch (dtype.bits()) {
    case 16:
      return 1;
    case 32:
      return 0;
    case 64:
      return 11;
    default:
      return -1;
    }
  } else if (dtype.is_bfloat16()) {
    return 27;
  } else if (dtype.is_int()) {
    switch (dtype.bits()) {
    case 4:
      return 29;
    case 8:
      return 2;
    case 16:
      return 6;
    case 32:
      return 3;
    case 64:
      return 9;
    default:
      return -1;
    }
  } else if (dtype.is_uint()) {
    switch (dtype.bits()) {
    case 1:
      return 30;
    case 8:
      return 4;
    case 16:
      return 7;
    case 32:
      return 8;
    case 64:
      return 10;
    default:
      return -1;
    }
  } else if (dtype.is_bool()) {
    return 12;
  }
  return -1;
}

} // namespace tvm::tl::a2a3
