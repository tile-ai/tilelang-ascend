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

#include "target_policy.h"

#include <tvm/runtime/logging.h>

#include "../common/operation_config.h"

namespace tvm::tl::ascend_sync {

void SyncTargetPolicy::Validate() const {
  ICHECK(platform == "A2" || platform == "A3" || platform == "A5")
      << "Unsupported Ascend synchronization platform: " << platform;
  ICHECK_GT(intra_core_flag_limit, 0);
  ICHECK_LE(intra_core_flag_limit, 8)
      << "The legacy Ascend event ABI exposes at most eight intra-core slots";
}

bool SyncTargetPolicy::EnabledForPipes(const std::string &producer,
                                       const std::string &consumer) const {
  if (ordinary_sync)
    return true;
  const auto vs_pipe = [](const std::string &pipe) {
    return pipe == "V" || pipe == "S" || pipe == "MTE2" || pipe == "MTE3";
  };
  const bool vector_self = producer == "V" && consumer == "V";
  const bool scalar_handoff =
      producer != consumer && (producer == "S" || consumer == "S");
  return vector_scalar_sync && (vector_self || scalar_handoff) &&
         vs_pipe(producer) && vs_pipe(consumer);
}

bool SyncTargetPolicy::HasImplicitSamePipeOrder(const std::string &pipe) const {
  // Only scalar completion order is established for this legacy policy.
  // Vector and Cube issue order must not silently become completion proofs.
  return pipe == "S";
}

bool SyncTargetPolicy::UsesFlag(const std::string &producer,
                                const std::string &consumer) const {
  return producer != consumer || (producer == "V" && platform == "A5");
}

void SyncTargetPolicy::RequireEvent(const std::string &event) const {
  const auto &mapping = GetEventMapping();
  for (const auto &entry : mapping) {
    if (entry.second == event)
      return;
  }
  LOG(FATAL) << "Ascend synchronization event " << event
             << " is unsupported by the legacy operation catalog on "
             << platform;
}

} // namespace tvm::tl::ascend_sync
