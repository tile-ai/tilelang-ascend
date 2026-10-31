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

#include "memory_info.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>

#include "support/check.h"

namespace tvm::tl::a2a3 {

void MemoryInfoNode::RegisterReflection() {
  ffi::reflection::ObjectDef<MemoryInfoNode>()
      .def_ro("unit_bits", &MemoryInfoNode::unit_bits)
      .def_ro("max_num_bits", &MemoryInfoNode::max_num_bits)
      .def_ro("max_simd_bits", &MemoryInfoNode::max_simd_bits)
      .def_ro("head_address", &MemoryInfoNode::head_address);
}

MemoryInfo GetMemoryInfo(const std::string &scope) {
  std::string name = "tvm.info.mem." + scope;
  auto callback = ffi::Function::GetGlobal(name);
  if (!callback) {
    LOG(WARNING) << "MemoryInfo for scope = " << scope << " is undefined";
    return {};
  }
  ffi::Any value = callback.value()();
  if (value == nullptr) {
    return {}; // Original nullable ObjectRef callback result.
  }
  auto info = value.try_cast<MemoryInfo>();
  ICHECK(info.has_value()) << name
                           << " must return tl.ascend.a2a3.MemoryInfo "
                              "created by MakeMemoryInfo, got "
                           << value.GetTypeKey();
  return info.value();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  MemoryInfoNode::RegisterReflection();
  ffi::reflection::GlobalDef().def(
      "tl.ascend.a2a3.MakeMemoryInfo",
      [](int64_t unit_bits, int64_t max_num_bits, int64_t max_simd_bits,
         ffi::Optional<PrimExpr> head_address) {
        auto node = ffi::make_object<MemoryInfoNode>();
        node->unit_bits = unit_bits;
        node->max_num_bits = max_num_bits;
        node->max_simd_bits = max_simd_bits;
        node->head_address = head_address.value_or(PrimExpr());
        return MemoryInfo(node);
      });
}

} // namespace tvm::tl::a2a3
