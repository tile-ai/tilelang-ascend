// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#ifndef TVM_TL_ASCEND_A2A3_CODEGEN_CONSTANT_H_
#define TVM_TL_ASCEND_A2A3_CODEGEN_CONSTANT_H_

#include <tvm/tirx/expr.h>

#include <functional>
#include <ostream>
#include <string>
#include <unordered_map>

namespace tvm::codegen::a2a3 {

// Literal declaration serialization belongs to the A2/A3 backend. Each dialect
// retains its own binding/Var bookkeeping and rejects unsupported pointer uses.
class ConstantDeclarations {
public:
  std::string
  Emit(const tirx::Call &call, std::ostream &decl_stream,
       const std::function<void(DataType, std::ostream &)> &print_type,
       const std::function<std::string(const std::string &)> &reserve_name);

private:
  std::unordered_map<std::string, std::string> declarations_;
  std::unordered_map<std::string, std::string> names_;
};

} // namespace tvm::codegen::a2a3
#endif // TVM_TL_ASCEND_A2A3_CODEGEN_CONSTANT_H_
