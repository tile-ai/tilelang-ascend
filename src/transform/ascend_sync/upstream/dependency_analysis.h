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
// Source: src/ascend/transform/auto_schedule/dependency_analysis.h.
// Adaptation: legacy TIR/containers, ProofAnalyzer, optional physical oracle.

#pragma once

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/optional.h>

#include <functional>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "buffer_version.h"
#include "ir_structure.h"

namespace tvm::tl::ascend_sync {

using DependencyTaskPair = std::pair<TaskNode *, TaskNode *>;

enum class DependencyKind {
  kData,
  // Disjoint owners of one automatically versioned storage. Pair orientation
  // is stable; the eventual physical order selects the directed protocols.
  kOwnerExclusion,
};

// Preserve each access's logical identity even if the physical oracle finds
// aliases between different data Vars. Owner/epoch lookup must use each side's
// own storage and control path, never an invented common version ring.
struct DependencyAccessPair {
  TaskNode *producer;
  BufferRegion producer_region;
  TaskNode *consumer;
  BufferRegion consumer_region;
};

struct DepInfo {
  IRStructure *prod_node;
  IRStructure *cons_node;
  // Shared logical storage, if any. Undefined for distinct-storage aliases or
  // explicit conflicts; those edges cannot inherit one side's buffer versions.
  Optional<Var> storage;
  std::vector<DependencyTaskPair> task_pairs;
  // 0: same iteration; -1: unresolved carry, resolved by the final owner/ring;
  // >=1: physical iteration distance for an already manual-versioned buffer.
  int distance;
  DependencyKind kind{DependencyKind::kData};
  std::vector<DependencyAccessPair> access_pairs;
};

// Cache scope remains one analysis phase, with unchanged tree, effects, oracle
// and address facts. Reusing a cache after any of those change is invalid.
using DependencyCache = std::map<ControlNode *, std::vector<DepInfo>>;

// Each ObjectRef entry is an Array<ObjectRef> containing operands,
// IntImm(cross_code) and a constant boolean polarity.
using ConflictHintList = Array<ObjectRef>;

struct PhysicalConflictQuery {
  const ConstrSet &producer_context;
  const BufferRegion &producer_region;
  const ConstrSet &consumer_context;
  const BufferRegion &consumer_region;
  ControlNode *loop;
  int iteration_offset;
  size_t num_storage_owners;
  Optional<PrimExpr> producer_pitch;
  Optional<PrimExpr> consumer_pitch;
};

// A result answers the physical conflict question, including its contexts and
// iteration relation. nullopt means unknown and therefore stays conflicting.
// An oracle may explicitly reuse RegionsMayConflict without an oracle once it
// has established that the logical regions describe the physical footprints.
using ConflictOracle =
    std::function<std::optional<bool>(const PhysicalConflictQuery &)>;

// Preserve upstream dependency direction, stage order, manual ring-distance
// search, owner exclusion and nested-loop coverage. The physical oracle is
// threaded through all recursive analyses and must not rename logical Vars.
// Without an oracle this has the original logical-storage semantics and must
// NOT be used on MemoryPlanning outputs with unaccounted address reuse.
std::vector<DepInfo>
AnalyzeDependencies(std::vector<IRStructure *> nodes,
                    ControlNode *loop = nullptr,
                    const BufferVersionMap &manual_buffer_versions = {},
                    const MultiBufferOwnerMap &multi_buffer_owners = {},
                    DependencyCache *dependency_cache = nullptr,
                    const ConflictHintList &root_conflicts = {},
                    const ConflictOracle &physical_conflict_oracle = {});

// offset=0 compares one iteration. Positive offsets fix consumer=producer+d;
// negative offsets rename the storage-local iteration nest and search carries.
// Unknown disjointness stays conflicting. Logical shape/pitch proofs are used
// only when the physical oracle has not already answered the complete query.
bool RegionsMayConflict(const ConstrSet &a_ctx, const BufferRegion &a_region,
                        const ConstrSet &b_ctx, const BufferRegion &b_region,
                        ControlNode *loop, int offset,
                        size_t num_storage_owners = 0,
                        const ConflictHintList &root_conflicts = {},
                        Optional<PrimExpr> a_pitch = NullOpt,
                        Optional<PrimExpr> b_pitch = NullOpt,
                        const ConflictOracle &physical_conflict_oracle = {});

} // namespace tvm::tl::ascend_sync
