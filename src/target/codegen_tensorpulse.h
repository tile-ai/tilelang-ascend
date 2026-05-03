// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file target/codegen_tensorpulse.h
 * \brief Code-generator skeleton for the TensorPulse (AIACC) backend.
 *
 * V1.0 minimal skeleton: enough to register `target.build.tilelang_tensorpulse`
 * and pass cmake build. Per-op visitors will be filled in incrementally.
 */
#ifndef TVM_TL_TARGET_CODEGEN_TENSORPULSE_H_
#define TVM_TL_TARGET_CODEGEN_TENSORPULSE_H_

#include <tvm/target/codegen.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <string>

#include "target/source/codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenTileLangTensorPulse final : public CodeGenC {
 public:
  CodeGenTileLangTensorPulse() = default;

  std::string Finish();

  void PrintFuncPrefix(std::ostream &os) final;
  void PrintStorageScope(const std::string &scope, std::ostream &os) final;
  void PrintType(DataType t, std::ostream &os) final;

 private:
  bool IsScopePartOfType() const final { return false; }
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TL_TARGET_CODEGEN_TENSORPULSE_H_
