// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "codegen_tensorpulse.h"

namespace tvm {
namespace codegen {

runtime::Module BuildTileLangTensorPulse(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangTensorPulse cg;
  cg.Init(output_ssa);

  Array<String> function_names;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangTensorPulse: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(gvar, f);
    function_names.push_back(cg.GetFunctionName(gvar));
  }

  std::string code = cg.Finish();
  return CSourceModuleCreate(code, "c", function_names);
}

TVM_REGISTER_GLOBAL("target.build.tilelang_tensorpulse")
    .set_body_typed(BuildTileLangTensorPulse);

}  // namespace codegen
}  // namespace tvm
