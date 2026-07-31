// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/target/utils.cc
 * \brief helper functions for target attributes.
 */

#include "utils.h"

#include <tvm/ir/transform.h>

namespace tvm {
namespace tl {

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

bool TargetIsCuda(Target target) {
  return target->GetTargetDeviceType() == kDLCUDA;
}
bool TargetIsRocm(Target target) {
  return target->GetTargetDeviceType() == kDLROCM;
}

int GetArchInt(Target target) {
  auto s = target->GetAttr<String>("arch");
  ICHECK(s.defined());
  const char *arch_str = s.value().c_str();
  ICHECK_EQ(arch_str[0], 's');
  ICHECK_EQ(arch_str[1], 'm');
  ICHECK_EQ(arch_str[2], '_');
  return atoi(&arch_str[3]);
}

bool TargetIsVolta(Target target) {
  if (!TargetIsCuda(target))
    return false;
  int arch = GetArchInt(target);
  return arch >= 70 && arch < 75;
}

bool TargetIsTuring(Target target) {
  if (!TargetIsCuda(target))
    return false;
  int arch = GetArchInt(target);
  return arch >= 75 && arch < 80;
}

bool TargetIsAmpere(Target target) {
  if (!TargetIsCuda(target))
    return false;
  int arch = GetArchInt(target);
  return arch >= 80 && arch < 90;
}

bool TargetIsHopper(Target target) {
  if (!TargetIsCuda(target))
    return false;
  int arch = GetArchInt(target);
  return arch >= 90;
}

bool TargetIsCDNA(Target target) {
  if (!TargetIsRocm(target))
    return false;
  if (target->attrs.count("mcpu")) {
    std::string mcpu = Downcast<String>(target->attrs.at("mcpu"));
    // if mcpu start with "gfx9", it is CDNA
    return mcpu.find("gfx9") == 0;
  }
  return false;
}

bool TargetHasAsyncCopy(Target target) {
  if (TargetIsCuda(target)) {
    int arch = GetArchInt(target);
    return arch >= 80;
  } else if (TargetIsCDNA(target)) {
    if (target->attrs.count("mcpu")) {
      std::string mcpu = Downcast<String>(target->attrs.at("mcpu"));
      if (mcpu.rfind("gfx9", 0) == 0) {
        int gfx_version = std::stoi(mcpu.substr(3, 2));
        return gfx_version >= 94;
      }
      return false;
    } else {
      return false;
    }
  }

  return false;
}
bool TargetHasLdmatrix(Target target) {
  if (!TargetIsCuda(target))
    return false;
  int arch = GetArchInt(target);
  return arch >= 75;
}

bool TargetHasStmatrix(Target target) {
  if (!TargetIsCuda(target))
    return false;
  int arch = GetArchInt(target);
  return arch >= 90;
}

} // namespace tl
} // namespace tvm
