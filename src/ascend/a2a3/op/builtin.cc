// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "builtin.h"

#include <unordered_map>
#include <unordered_set>

#include "op/builtin_registry.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/script/builder/frame.h>

namespace tvm {
namespace tl {
namespace a2a3 {

using namespace tirx;

// Literal metadata is opaque to DCE but has no hardware resource/mask effect.
TIR_DEFINE_TL_BUILTIN(a2a3_const_data)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// ascend_pto
TIR_DEFINE_TL_BUILTIN(ascend_scalar_op)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_unary_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_binary_op)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_binary_ops)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

} // namespace a2a3
} // namespace tl
} // namespace tvm

// This is a builder-only repair for the handle Var created by the eager
// frontend from an opaque const-data expression. Like the builder Namer, it
// preserves identity while constructing IR; it cannot rewrite a finished func.

namespace tvm::tl::a2a3 {
namespace {
tirx::Var PrepareConstBufferData(tirx::Var var) {
  using tvm::script::ir_builder::IRBuilder;
  namespace builder = tvm::script::ir_builder::tirx;
  ICHECK(IRBuilder::IsInScope())
      << "A2/A3 constant pointer typing is only valid in a live IR builder";
  auto current = IRBuilder::Current();
  ICHECK(current->FindFrame<builder::PrimFuncFrame>().defined())
      << "A2/A3 constant pointer typing requires a live PrimFunc builder";
  std::unordered_map<tirx::Var, tirx::Call, ffi::ObjectPtrHash,
                     ffi::ObjectPtrEqual>
      origins;
  std::unordered_set<tirx::Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      definitions;
  std::unordered_set<tirx::Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      ambiguous;
  std::unordered_map<tirx::Var, tirx::Var, ffi::ObjectPtrHash,
                     ffi::ObjectPtrEqual>
      aliases;
  for (const auto &frame : current->frames) {
    if (const auto *tir_frame = frame.as<builder::TIRFrameNode>()) {
      // Only direct statements of live lexical frames are in scope. Do not
      // discover declarations hidden in already-closed branch/loop frames.
      for (const tirx::Stmt &stmt : tir_frame->stmts) {
        if (const auto *binding = stmt.as<tirx::BindNode>()) {
          if (!definitions.insert(binding->var).second) {
            ambiguous.insert(binding->var);
          }
          if (const auto *call = binding->value.as<tirx::CallNode>();
              call && call->op.same_as(a2a3_const_data())) {
            origins.emplace(binding->var, ffi::GetRef<tirx::Call>(call));
          } else if (const auto *alias = binding->value.as<tirx::VarNode>()) {
            auto found = origins.find(ffi::GetRef<tirx::Var>(alias));
            if (found != origins.end()) {
              origins.emplace(binding->var, found->second);
              aliases.emplace(binding->var, ffi::GetRef<tirx::Var>(alias));
            }
          }
        }
      }
    }
  }
  auto found = origins.find(var);
  if (found == origins.end())
    return var; // Preserve ordinary decl_buffer.
  std::unordered_set<tirx::Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> chain;
  for (tirx::Var current_var = var;;) {
    ICHECK(!ambiguous.count(current_var) && chain.insert(current_var).second)
        << "A2/A3 constant pointer source has an ambiguous Var redefinition";
    auto parent = aliases.find(current_var);
    if (parent == aliases.end())
      break;
    current_var = parent->second;
  }
  ICHECK(var.dtype().is_handle()) << "A2/A3 constant pointer must be a handle";
  ICHECK(!found->second->args.empty());
  const auto *dtype_string = found->second->args[0].as<tirx::StringImmNode>();
  ICHECK(dtype_string) << "A2/A3 constant data needs a literal dtype";
  DataType dtype(ffi::StringToDLDataType(dtype_string->value));
  if (const auto *pointer = var->type_annotation.as<PointerTypeNode>()) {
    const auto *element = pointer->element_type.as<PrimTypeNode>();
    ICHECK(element && element->dtype == dtype)
        << "A2/A3 constant pointer has a conflicting element type";
    return var;
  }
  const auto *primitive = var->type_annotation.as<PrimTypeNode>();
  ICHECK(primitive && primitive->dtype.is_handle())
      << "A2/A3 constant pointer has a conflicting type annotation";
  const_cast<tirx::VarNode *>(var.get())->type_annotation =
      PointerType(PrimType(dtype));
  return var;
}
} // namespace
TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def("tl.a2a3.PrepareConstBufferData",
                                        PrepareConstBufferData);
}
} // namespace tvm::tl::a2a3
