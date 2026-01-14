// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "codegen_ascend_pto.h"

namespace tvm {
namespace codegen {

runtime::Module BuildTileLangAscendPto(IRModule mod, Target target, std::string plantform) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenTileLangAscendPto cg(plantform);
  cg.Init(output_ssa);

  Array<String> function_names;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangAscendPto: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(gvar, f);
    function_names.push_back(cg.GetFunctionName(gvar));
  }

  std::string code = cg.Finish();

  return CSourceModuleCreate(code, "c", function_names);
}

TVM_REGISTER_GLOBAL("target.build.tilelang_ascend_pto")
    .set_body_typed(BuildTileLangAscendPto);

} // namespace codegen
} // namespace tvm

