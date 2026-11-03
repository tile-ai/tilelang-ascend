// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

// Private Cython allocator promises. Merely naming a workspace is not proof
// of allocation; public JIT/cache reconstruction checks the actual recipe.

#pragma once

#include <tvm/runtime/logging.h>
#include <tvm/tir/function.h>

#include <limits>
#include <unordered_set>

namespace tvm::tl::ascend_sync {

inline constexpr const char *kAutoGMFreshContract =
    "tl.ascend_auto_gm_fresh_contract";
inline constexpr const char *kAutoGMFreshContractVersion = "cython-auto-gm-v1";
inline constexpr const char *kCythonFreshContractVersion = "cython-auto-gm-v2";
inline constexpr const char *kCythonResultIndices =
    "tl.ascend_jit_result_indices";
inline constexpr const char *kCythonWorkspaceIndices =
    "tl.ascend_jit_workspace_indices";
inline constexpr const char *kCythonInputSignature =
    "tl.ascend_jit_input_signature";
inline constexpr const char *kCythonFinalSignature =
    "tl.ascend_jit_final_signature";

namespace runtime_contract_detail {

inline void CheckSignature(const tir::PrimFunc &function, const char *name,
                           bool prefix) {
  auto raw = function->GetAttr<ObjectRef>(name);
  ICHECK(raw.defined() && raw.value().as<ArrayNode>())
      << "Cython fresh allocator requires its exact signature provenance: "
      << name;
  const auto signature = Downcast<Array<ObjectRef>>(raw.value());
  ICHECK(prefix ? signature.size() <= function->params.size()
                : signature.size() == function->params.size())
      << "Cython fresh allocator signature changed after binding";
  for (size_t index = 0; index < signature.size(); ++index)
    ICHECK(signature[index].as<tir::VarNode>() &&
           signature[index].same_as(function->params[index]))
        << "Cython fresh allocator parameters were inserted, removed or "
           "reordered after binding";
}

struct Contract {
  Array<tir::Var> storages;
  Array<tir::Var> automatic_storages;
  bool explicit_v2{false};
};

inline Contract Read(const tir::PrimFunc &function, bool require_final) {
  auto contract = function->GetAttr<ObjectRef>(kAutoGMFreshContract);
  if (!contract.defined())
    return {};
  const auto *version = contract.value().as<tir::StringImmNode>();
  ICHECK(version && (version->value == kAutoGMFreshContractVersion ||
                     version->value == kCythonFreshContractVersion))
      << "Unsupported automatic GM fresh-allocation contract";
  const bool v2 = version->value == kCythonFreshContractVersion;
  if (v2) {
    CheckSignature(function, kCythonInputSignature, true);
    if (require_final ||
        function->GetAttr<ObjectRef>(kCythonFinalSignature).defined())
      CheckSignature(function, kCythonFinalSignature, false);
  }
  Contract result;
  std::unordered_set<int64_t> seen_auto_indices;
  std::unordered_set<int64_t> seen_indices;
  auto add_storage = [&](int64_t index, bool static_workspace) {
    ICHECK(index >= 0 && static_cast<uint64_t>(index) < function->params.size())
        << "Cython fresh allocation index is outside the final signature";
    const tir::Var &parameter = function->params[index];
    auto buffer = function->buffer_map.Get(parameter);
    ICHECK(
        parameter.dtype().is_handle() && buffer.defined() &&
        (buffer.value().scope().empty() || buffer.value().scope() == "global"))
        << "Automatic GM fresh-allocation contract requires GM buffer "
           "parameters";
    if (static_workspace)
      for (const PrimExpr &extent : buffer.value()->shape) {
        const auto *constant = extent.as<IntImmNode>();
        ICHECK(constant && constant->value > 0)
            << "Automatic GM fresh-allocation contract requires static "
               "positive workspace shapes";
      }
    if (!seen_indices.insert(index).second)
      return; // Membership, including overlapping explicit roles, allocates
              // once.
    const tir::Var &storage = buffer.value()->data;
    for (const auto &[other_parameter, other_buffer] : function->buffer_map)
      ICHECK(other_parameter.same_as(parameter) ||
             !other_buffer->data.same_as(storage))
          << "Automatic GM fresh-allocation contract shares one data Var "
             "between parameter slots";
    result.storages.push_back(storage);
    if (static_workspace)
      result.automatic_storages.push_back(storage);
  };
  auto indices = function->GetAttr<ObjectRef>("auto_gm_indices");
  ICHECK(indices.defined() && indices.value().as<ArrayNode>())
      << "Automatic GM fresh-allocation contract requires auto_gm_indices";
  for (const ObjectRef &entry : Downcast<Array<ObjectRef>>(indices.value())) {
    const auto *index = entry.as<IntImmNode>();
    ICHECK(index && index->dtype.is_scalar() && !index->dtype.is_bool() &&
           index->value >= 0 && seen_auto_indices.insert(index->value).second)
        << "Automatic GM fresh-allocation contract has invalid or duplicate "
           "parameter indices";
    add_storage(index->value, true);
  }
  if (v2) {
    ICHECK_LE(function->params.size(),
              static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
    const int64_t count = static_cast<int64_t>(function->params.size());
    for (const char *name : {kCythonResultIndices, kCythonWorkspaceIndices}) {
      auto raw = function->GetAttr<ObjectRef>(name);
      ICHECK(raw.defined() && raw.value().as<ArrayNode>())
          << "Cython fresh allocator requires immutable raw role indices";
      for (const ObjectRef &entry : Downcast<Array<ObjectRef>>(raw.value())) {
        const auto *index = entry.as<IntImmNode>();
        ICHECK(index && index->dtype == DataType::Int(64))
            << "Cython fresh allocator has a malformed raw role index";
        const int64_t normalized =
            index->value < 0 ? index->value + count : index->value;
        add_storage(normalized, false);
        result.explicit_v2 = true;
      }
    }
  }
  return result;
}

} // namespace runtime_contract_detail

inline tir::PrimFunc
BindTrustedCythonFreshFinalSignature(const tir::PrimFunc &function) {
  auto contract = function->GetAttr<ObjectRef>(kAutoGMFreshContract);
  const auto *version =
      contract.defined() ? contract.value().as<tir::StringImmNode>() : nullptr;
  if (version == nullptr || version->value != kCythonFreshContractVersion)
    return function;
  runtime_contract_detail::Read(function, false);
  if (function->GetAttr<ObjectRef>(kCythonFinalSignature).defined())
    return function;
  return WithAttr(function, String(kCythonFinalSignature), function->params);
}

inline Array<tir::Var>
ReadTrustedFreshGMStorages(const tir::PrimFunc &function) {
  return runtime_contract_detail::Read(function, true).storages;
}

// Preserve the original local-analysis fact set while still validating every
// v2 role and signature. Choosing fewer facts never weakens those checks.
inline Array<tir::Var>
ReadTrustedAutoGMStorages(const tir::PrimFunc &function) {
  return runtime_contract_detail::Read(function, true).automatic_storages;
}

inline bool HasTrustedFreshGMV2Contract(const tir::PrimFunc &function) {
  return runtime_contract_detail::Read(function, true).explicit_v2;
}

} // namespace tvm::tl::ascend_sync
