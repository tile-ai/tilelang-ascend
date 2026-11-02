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
// Source: src/ascend/transform/insert_sync.cc. The legacy adapter owns TIR
// import/export; this API owns upstream dependency and synchronization
// planning.

#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "../target_policy.h"
#include "ir_structure.h"
#include "multi_buffer.h"

namespace tvm::tl::ascend_sync {

std::unique_ptr<IRExtraInfo> MakeSyncExtraInfo();

// Describe an audited atomic helper whose incoming and outgoing synchronization
// use different hardware pipes. This does not change its canonical task pipe
// or certify completion at return. Both pipes must be concrete single pipes;
// unset metadata preserves the ordinary task's canonical single-pipe behavior.
// Call before RunSyncEngine, on a task carrying MakeSyncExtraInfo().
void SetSyncBoundaryPipes(TaskNode *task, ResourcePipe acquire,
                          ResourcePipe completion);

// Mark an adapter's explicit For(0, Select(condition, 1, 0)) representation
// of an optional statement. Its before/after physical boundaries still exist
// when its payload does not execute. Only these audited controls may use the
// non-versioned forward-carry strengthening in the synchronization engine.
void SetSyncOptionalControl(ControlNode *control);

// Grant only after the legacy adapter verifies the complete static Cube SIP
// skeleton, true L1 B2 lineage, clocks, helper footprints and clear semantics.
// This marks the real steady For, never a public attribute or synthetic epoch.
void SetSyncStaticCubePipelineControl(ControlNode *control);

struct SyncPlacement {
  IRStructure *anchor{nullptr};
  bool at_beginning{false};
  tir::Stmt statement;
  CoreMask core{kCoreUnassigned};
};

// Original nodes return nullopt. Generated tasks retain their precise anchor
// so a legacy exporter need not infer scope from physical sibling adjacency.
std::optional<SyncPlacement> GetSyncPlacement(IRStructure *node);

// All original nodes must carry MakeSyncExtraInfo(). Calculations, buffer
// addresses and physical versioning stay unchanged. Generated synchronization
// becomes adjacent new TaskNodes; original node identities remain stable.
void RunSyncEngine(std::vector<std::shared_ptr<IRStructure>> &roots,
                   const MultiBufferPlan &multibuffer, BufferVersionMap manual,
                   const tir::Stmt &original_body,
                   const SyncTargetPolicy &policy);

} // namespace tvm::tl::ascend_sync
