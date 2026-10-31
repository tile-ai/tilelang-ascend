// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "../op/builtin.h"
#include "codegen_ascend.h"
#include "support/check.h"
#include <tvm/tirx/stmt_functor.h>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>

namespace tvm::codegen::a2a3 {

ffi::Module BuildTileLangAscend(IRModule mod, Target target,
                                std::string platform) {
  ICHECK(target.defined()) << "A2/A3 AscendC codegen requires a defined target";
  auto arch = target->GetAttr<ffi::String>("arch");
  ICHECK(target->kind->name == "ascend" && arch.has_value() &&
         arch.value() == "dav-2201")
      << "A2/A3 AscendC codegen requires ascend arch=dav-2201";
  ICHECK(platform == "A2" || platform == "A3")
      << "A2/A3 AscendC codegen requires a resolved A2/A3 platform for "
         "Reduce2D "
         "repeat-zero capabilities; set TL_PLATFORM=A2/A3 or resolve a known "
         "NPU model";
  if (mod->functions.size() != 1) {
    for (const auto &entry : mod->functions) {
      if (const auto *function = entry.second.as<tirx::PrimFuncNode>()) {
        tirx::PostOrderVisit(function->body, [](const ObjectRef &node) {
          if (const auto *call = node.as<tirx::CallNode>()) {
            ICHECK(!call->op.same_as(tl::a2a3::a2a3_const_data()))
                << "A2/A3 literal constants require a single exported kernel; "
                   "multi-function direct codegen is unsupported";
          }
        });
      }
    }
  }
  CodeGenTileLangAscend cg(std::move(platform));
  cg.SetEmitLineDirectives(
      tvm::transform::PassContext::Current()
          ->GetConfig<Bool>("tl.emit_line_directives", Bool(false))
          .value());
  cg.Init(false);

  ffi::Array<ffi::String> function_names;
  for (const auto &kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tirx::PrimFuncNode>())
        << "CodeGenTileLangAscend: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<tirx::PrimFunc>(kv.second);
    cg.AddFunction(gvar, f);
    function_names.push_back(cg.GetFunctionName(gvar));
  }
  return CSourceModuleCreate(cg.Finish(), "c", function_names);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def("target.build.tilelang_ascend_a2a3",
                                        BuildTileLangAscend);
}

} // namespace tvm::codegen::a2a3
