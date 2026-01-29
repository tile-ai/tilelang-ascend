// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "codegen_ptoas.h"

namespace tvm {
namespace codegen {

runtime::Module BuildTileLangPTOAS(IRModule mod, Target target, std::string platform) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenTileLangPTOAS cg(platform);
  cg.Init(output_ssa);

  Array<String> function_names;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangPTOAS: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(gvar, f);
    function_names.push_back(cg.GetFunctionName(gvar));
  }

  std::string code = cg.Finish();

  return CSourceModuleCreate(code, "c", function_names);
}

TVM_REGISTER_GLOBAL("target.build.tilelang_ptoas")
    .set_body_typed(BuildTileLangPTOAS);

} // namespace codegen
} // namespace tvm

