// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#pragma once

#include "../../op/builtin.h"
#include "ascend_vector_mask.h"
#include "operation_config.h"
#include "tir/transforms/ir_utils.h"
#include <algorithm>
#include <cctype>
#include <tvm/tir/utils.h>

namespace tvm {
namespace tl {
using namespace tir;

enum class AscendResource {
  kNone,
  kCommon,
  kExplicit,
  kCube,
  kVector,
};

inline AscendResource ResourceForStorageScope(const std::string &scope) {
  if (scope == "shared.ub") {
    return AscendResource::kVector;
  }
  if (scope == "shared.l1" || scope == "wmma.matrix_a" ||
      scope == "wmma.matrix_b" || scope == "wmma.accumulator") {
    return AscendResource::kCube;
  }
  return AscendResource::kNone;
}

inline AscendResource ResourceForBufferStore(const BufferStoreNode *store) {
  // A GM address does not identify which core should write it. Require an
  // explicit owner rather than duplicating the side effect or choosing Cube.
  // Per-core scalar assignments remain shared with their control flow.
  if (store->buffer.scope() == "global") {
    return AscendResource::kExplicit;
  }
  return ResourceForStorageScope(store->buffer.scope());
}

inline std::string NormalizePipeName(std::string pipe) {
  std::transform(pipe.begin(), pipe.end(), pipe.begin(),
                 [](unsigned char ch) { return std::toupper(ch); });
  if (pipe.rfind("PIPE_", 0) == 0) {
    pipe.erase(0, 5);
  }
  return pipe;
}

inline AscendResource ResourceForPipe(const std::string &pipe) {
  std::string normalized = NormalizePipeName(pipe);
  if (normalized == "V") {
    return AscendResource::kVector;
  }
  if (normalized == "M" || normalized == "MTE1" || normalized == "FIX") {
    return AscendResource::kCube;
  }
  return AscendResource::kExplicit;
}

inline AscendResource ResourceForPipeArgument(const CallNode *call,
                                              size_t index) {
  if (call->args.size() <= index) {
    return AscendResource::kExplicit;
  }
  const auto *pipe = call->args[index].as<StringImmNode>();
  return pipe == nullptr ? AscendResource::kExplicit
                         : ResourceForPipe(pipe->value);
}

inline AscendResource MergeResources(AscendResource lhs, AscendResource rhs,
                                     const std::string &operation) {
  if (rhs == AscendResource::kNone || rhs == AscendResource::kCommon) {
    return lhs;
  }
  if (lhs == AscendResource::kNone || lhs == AscendResource::kCommon) {
    return rhs;
  }
  ICHECK(lhs == rhs || lhs == AscendResource::kExplicit ||
         rhs == AscendResource::kExplicit)
      << "Ascend operation mixes Cube and Vector local buffers: " << operation;
  return lhs == AscendResource::kExplicit ? rhs : lhs;
}

inline std::string StorageScopeForAccessPtr(const PrimExpr &expr) {
  const auto *access = expr.as<CallNode>();
  if (access == nullptr || !access->op.same_as(builtin::tvm_access_ptr()) ||
      access->args.size() < 2) {
    return "";
  }
  const auto *var = access->args[1].as<VarNode>();
  if (var == nullptr) {
    return "";
  }
  return tir::GetPtrStorageScope(GetRef<Var>(var));
}

inline AscendResource ResourceForAccessPtr(const PrimExpr &expr) {
  return ResourceForStorageScope(StorageScopeForAccessPtr(expr));
}

inline AscendResource ResourceForAccessPtrs(const Array<PrimExpr> &args,
                                            size_t begin,
                                            const std::string &operation) {
  AscendResource result = AscendResource::kNone;
  for (size_t i = begin; i < args.size(); ++i) {
    result = MergeResources(result, ResourceForAccessPtr(args[i]), operation);
  }
  return result;
}

inline std::string NormalizeFunctionName(std::string name) {
  size_t template_pos = name.find('<');
  if (template_pos != std::string::npos) {
    name.resize(template_pos);
  }
  size_t namespace_pos = name.rfind("tl::ascend::");
  if (namespace_pos != std::string::npos) {
    name = name.substr(namespace_pos + 12);
  }
  return name;
}

inline AscendResource
ResourceForConfiguredOperation(const std::string &name,
                               const OperationConfig &config,
                               const Array<PrimExpr> &args, size_t begin) {
  if (!config.copy_scopes.empty()) {
    ICHECK(config.copy_resource_scope == 0 || config.copy_resource_scope == 1);
    AscendResource owner = config.copy_resource_scope == 0
                               ? AscendResource::kCube
                               : AscendResource::kVector;
    ICHECK_GE(config.copy_scopes.size(), 2U);
    ICHECK_GE(args.size(), begin + config.copy_scopes.size());
    for (size_t i = 0; i < config.copy_scopes.size(); ++i) {
      ICHECK_EQ(StorageScopeForAccessPtr(args[begin + i]),
                config.copy_scopes[i])
          << "Invalid copy operand storage scope: " << name << " operand " << i;
    }
    for (size_t i = begin + config.copy_scopes.size(); i < args.size(); ++i) {
      MergeResources(owner, ResourceForAccessPtr(args[i]), name);
    }
    return owner;
  }
  AscendResource operands = ResourceForAccessPtrs(args, begin, name);
  if (config.default_pipeline == "PIPE_V") {
    return MergeResources(AscendResource::kVector, operands, name);
  }
  if (config.default_pipeline == "PIPE_M" ||
      config.default_pipeline == "PIPE_MTE1" ||
      config.default_pipeline == "PIPE_FIX") {
    return MergeResources(AscendResource::kCube, operands, name);
  }
  return operands == AscendResource::kNone ? AscendResource::kExplicit
                                           : operands;
}

inline AscendResource ResourceForPipePair(const std::string &src,
                                          const std::string &dst,
                                          const std::string &operation) {
  AscendResource src_resource = ResourceForPipe(src);
  AscendResource dst_resource = ResourceForPipe(dst);
  return MergeResources(src_resource, dst_resource, operation);
}

inline AscendResource ResourceForEventType(const std::string &event_type,
                                           const std::string &operation) {
  size_t separator = event_type.find('_');
  if (separator == std::string::npos) {
    return AscendResource::kExplicit;
  }
  return ResourceForPipePair(event_type.substr(0, separator),
                             event_type.substr(separator + 1), operation);
}

inline AscendResource ResourceForCall(const CallNode *call,
                                      std::string *operation) {
  Call call_ref = GetRef<Call>(call);
  const auto *op = call->op.as<OpNode>();
  *operation = op == nullptr ? "Ascend call" : op->name;

  if (call->op.same_as(ascend_printf()) ||
      call->op.same_as(ascend_sync_all()) ||
      call->op.same_as(ascend_use_swizzle()) ||
      call->op.same_as(loop_break())) {
    return AscendResource::kCommon;
  }
  if (call->op.same_as(ascend_datacachecleanandinvalid_experiment())) {
    ICHECK_EQ(call->args.size(), 2U);
    std::string scope = StorageScopeForAccessPtr(call->args[1]);
    ICHECK(scope == "global" || scope == "shared.ub")
        << "DataCacheCleanAndInvalid requires a GM or UB operand";
    // A GM address does not identify whose scalar cache needs maintenance.
    // Its owner must come from explicit scope or sufficient surrounding work.
    return scope == "shared.ub" ? AscendResource::kVector
                                : AscendResource::kExplicit;
  }
  if (call->op.same_as(ascend_src_code())) {
    return AscendResource::kExplicit;
  }
  if (call->op.same_as(ascend_free_pipe())) {
    if (call->args.empty()) {
      return AscendResource::kExplicit;
    }
    const auto *name = call->args[0].as<StringImmNode>();
    if (name != nullptr && name->value == "free_pipe_C") {
      return AscendResource::kCube;
    }
    if (name != nullptr && name->value == "free_pipe_V") {
      return AscendResource::kVector;
    }
    return AscendResource::kExplicit;
  }
  if (call->op.same_as(ascend_set_deq_scale()) ||
      IsVectorMaskSetter(call_ref) || IsSelectedVectorTerminal(call_ref)) {
    return AscendResource::kVector;
  }
  if (call->op.same_as(ascend_dump_tensor())) {
    AscendResource resource = call->args.empty()
                                  ? AscendResource::kNone
                                  : ResourceForAccessPtr(call->args[0]);
    return resource == AscendResource::kNone ? AscendResource::kCommon
                                             : resource;
  }
  if (call->op.same_as(ascend_set_flag()) ||
      call->op.same_as(ascend_wait_flag())) {
    if (call->args.size() < 2) {
      return AscendResource::kExplicit;
    }
    const auto *src = call->args[0].as<StringImmNode>();
    const auto *dst = call->args[1].as<StringImmNode>();
    return src == nullptr || dst == nullptr
               ? AscendResource::kExplicit
               : ResourceForPipePair(src->value, dst->value, *operation);
  }
  if (call->op.same_as(ascend_pipe_barrier()) ||
      call->op.same_as(ascend_set_cross_flag()) ||
      call->op.same_as(ascend_auto_barrier())) {
    return ResourceForPipeArgument(call, 0);
  }
  if (call->op.same_as(ascend_auto_set_flag()) ||
      call->op.same_as(ascend_auto_wait_flag())) {
    if (call->args.empty()) {
      return AscendResource::kExplicit;
    }
    const auto *event_type = call->args[0].as<StringImmNode>();
    return event_type == nullptr
               ? AscendResource::kExplicit
               : ResourceForEventType(event_type->value, *operation);
  }
  if (call->op.same_as(ascend_wait_cross_flag()) ||
      call->op.same_as(ascend_auto_set_cross_flag()) ||
      call->op.same_as(ascend_auto_wait_cross_flag())) {
    return ResourceForPipeArgument(call, 1);
  }

  if (call->op.same_as(builtin::call_extern())) {
    if (call->args.empty()) {
      return AscendResource::kExplicit;
    }
    const auto *callee = call->args[0].as<StringImmNode>();
    if (callee == nullptr) {
      return AscendResource::kExplicit;
    }
    *operation = callee->value;
    std::string name = NormalizeFunctionName(callee->value);
    if (name == "free_pipe_C") {
      return AscendResource::kCube;
    }
    if (name == "free_pipe_V") {
      return AscendResource::kVector;
    }
    auto it = GetOperationConfig().find(name);
    if (it != GetOperationConfig().end()) {
      return ResourceForConfiguredOperation(name, it->second, call->args, 1);
    }
    AscendResource resource = ResourceForAccessPtrs(call->args, 1, *operation);
    return resource == AscendResource::kNone ? AscendResource::kExplicit
                                             : resource;
  }

  if (op == nullptr) {
    return AscendResource::kNone;
  }
  auto it = GetOperationConfig().find(op->name);
  if (it != GetOperationConfig().end()) {
    return ResourceForConfiguredOperation(op->name, it->second, call->args, 0);
  }
  if (std::string(op->name).rfind("tl.ascend_", 0) == 0) {
    AscendResource resource = ResourceForAccessPtrs(call->args, 0, *operation);
    return resource == AscendResource::kNone ? AscendResource::kExplicit
                                             : resource;
  }
  return AscendResource::kNone;
}

} // namespace tl
} // namespace tvm
