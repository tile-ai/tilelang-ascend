/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

// Ported from tile-ai/tilelang 1c0f402e166a7d0a055c991d2f82d2175a7917b7.
// Source: src/ascend/transform/buffer_version.h. Legacy adaptation: version
// counts are pass-local integers; TIR attribute serialization is
// importer-owned.

#pragma once

#include <tvm/runtime/logging.h>
#include <tvm/tir/expr.h>

#include <optional>
#include <unordered_map>

namespace tvm::tl::ascend_sync {

// Legacy TVM Map requires ObjectRef values. Keep the upstream integer-valued
// model local to analysis, with its small Get/Set API preserved for consumers.
class BufferVersionMap
    : public std::unordered_map<tir::Var, int, ObjectPtrHash, ObjectPtrEqual> {
public:
  using Base = std::unordered_map<tir::Var, int, ObjectPtrHash, ObjectPtrEqual>;
  using Base::Base;

  std::optional<int> Get(const tir::Var &storage) const {
    auto it = find(storage);
    return it == end() ? std::nullopt : std::optional<int>(it->second);
  }
  void Set(const tir::Var &storage, int versions) {
    ICHECK_GT(versions, 0) << "Buffer version counts must be positive";
    insert_or_assign(storage, versions);
  }
};

} // namespace tvm::tl::ascend_sync
