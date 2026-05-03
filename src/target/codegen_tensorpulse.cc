// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "codegen_tensorpulse.h"

namespace tvm {
namespace codegen {

std::string CodeGenTileLangTensorPulse::Finish() {
  std::ostringstream code;
  code << CodeGenC::Finish();
  return code.str();
}

void CodeGenTileLangTensorPulse::PrintFuncPrefix(std::ostream &os) {
  os << "extern \"C\" ";
}

void CodeGenTileLangTensorPulse::PrintStorageScope(const std::string &scope,
                                                   std::ostream &os) {
  // TensorPulse memory model is 2-level: external address space + UR.
  if (scope == "global") {
    os << " ";
  } else if (scope == "shared" || scope == "ur") {
    os << " ";  // UR (User Register); placeholder qualifier for now.
  } else {
    LOG(FATAL) << "TensorPulse: unsupported storage scope: " << scope;
  }
}

void CodeGenTileLangTensorPulse::PrintType(DataType t, std::ostream &os) {
  // FP8 / FP4 specialisations will be added later; defer to base for now.
  CodeGenC::PrintType(t, os);
}

}  // namespace codegen
}  // namespace tvm
