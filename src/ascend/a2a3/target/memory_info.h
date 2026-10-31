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

#ifndef TVM_TL_ASCEND_A2A3_TARGET_MEMORY_INFO_H_
#define TVM_TL_ASCEND_A2A3_TARGET_MEMORY_INFO_H_

#include <string>
#include <tvm/ir/expr.h>

namespace tvm::tl::a2a3 {

// Source: TA 0deb8b93 3rdparty/tvm/include/tvm/target/target_info.h.
// The shared TVM MemoryInfo type was removed; keep its four fields here.
class MemoryInfoNode : public ffi::Object {
public:
  int64_t unit_bits{0};
  int64_t max_num_bits{0};
  int64_t max_simd_bits{0};
  PrimExpr head_address;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.ascend.a2a3.MemoryInfo", MemoryInfoNode,
                                    ffi::Object);
};

class MemoryInfo : public ffi::ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(MemoryInfo, ffi::ObjectRef,
                                             MemoryInfoNode);
};

MemoryInfo GetMemoryInfo(const std::string &scope);

} // namespace tvm::tl::a2a3
#endif // TVM_TL_ASCEND_A2A3_TARGET_MEMORY_INFO_H_
