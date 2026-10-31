// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

// The Cython JIT allocator publishes this private contract before lowering.
// auto_gm_indices alone does not establish a fresh-allocation fact.

#pragma once

#include <tvm/runtime/logging.h>
#include <tvm/tir/function.h>

#include <unordered_set>

namespace tvm::tl::ascend_sync {

inline constexpr const char *kAutoGMFreshContract =
    "tl.ascend_auto_gm_fresh_contract";
inline constexpr const char *kAutoGMFreshContractVersion = "cython-auto-gm-v1";

inline Array<tir::Var>
ReadTrustedFreshGMStorages(const tir::PrimFunc &function) {
  auto contract = function->GetAttr<ObjectRef>(kAutoGMFreshContract);
  if (!contract.defined())
    return {};
  const auto *version = contract.value().as<tir::StringImmNode>();
  ICHECK(version != nullptr && version->value == kAutoGMFreshContractVersion)
      << "Unsupported automatic GM fresh-allocation contract";
  auto indices = function->GetAttr<ObjectRef>("auto_gm_indices");
  ICHECK(indices.defined() && indices.value().as<ArrayNode>())
      << "Automatic GM fresh-allocation contract requires auto_gm_indices";
  Array<tir::Var> result;
  std::unordered_set<int64_t> seen_indices;
  std::unordered_set<tir::Var, ObjectPtrHash, ObjectPtrEqual> seen_storage;
  for (const ObjectRef &entry : Downcast<Array<ObjectRef>>(indices.value())) {
    const auto *index = entry.as<IntImmNode>();
    ICHECK(index != nullptr && index->dtype.is_scalar() &&
           !index->dtype.is_bool() && index->value >= 0 &&
           static_cast<uint64_t>(index->value) < function->params.size() &&
           seen_indices.insert(index->value).second)
        << "Automatic GM fresh-allocation contract has invalid or duplicate "
           "parameter indices";
    const tir::Var &parameter = function->params[index->value];
    auto buffer = function->buffer_map.Get(parameter);
    ICHECK(
        parameter.dtype().is_handle() && buffer.defined() &&
        (buffer.value().scope().empty() || buffer.value().scope() == "global"))
        << "Automatic GM fresh-allocation contract requires GM buffer "
           "parameters";
    const tir::Var &storage = buffer.value()->data;
    ICHECK(seen_storage.insert(storage).second)
        << "Automatic GM fresh-allocation contract has ambiguous storage "
           "identities";
    for (const auto &[other_parameter, other_buffer] : function->buffer_map) {
      ICHECK(other_parameter.same_as(parameter) ||
             !other_buffer->data.same_as(storage))
          << "Automatic GM fresh-allocation contract shares one data Var "
             "between parameter slots";
    }
    for (const PrimExpr &extent : buffer.value()->shape) {
      const auto *constant = extent.as<IntImmNode>();
      ICHECK(constant != nullptr && constant->value > 0)
          << "Automatic GM fresh-allocation contract requires static positive "
             "workspace shapes";
    }
    result.push_back(storage);
  }
  return result;
}

} // namespace tvm::tl::ascend_sync
