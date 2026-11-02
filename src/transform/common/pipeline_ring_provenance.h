// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

#pragma once

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/map.h>
#include <tvm/tir/function.h>

namespace tvm::tl {

// A logical ring id is the original injector allocation Var. It stays stable
// when StorageRewrite merges allocations or changes a pointer's element type.
// Only kPhysicalRingRoot changes; bit units preserve offsets across dtype
// rewriting. Records are emitted only for statically known physical layouts.
inline constexpr const char *kPhysicalRingProvenance =
    "tl.ascend_physical_ring_provenance";
inline constexpr const char *kPhysicalRingClock =
    "tl.ascend_physical_ring_clock";
inline constexpr const char *kPhysicalPipelineOwners =
    "tl.ascend_physical_pipeline_owners";
inline constexpr const char *kPhysicalPipelineExtent =
    "tl.ascend_physical_pipeline_extent";
inline constexpr const char *kPhysicalPipelineSegment =
    "tl.ascend_physical_pipeline_segment";
inline constexpr const char *kPhysicalPipelineStage =
    "tl.ascend_physical_pipeline_stage";

// Owner records contain only stable identities/static scheduling facts.
// Dynamic extent and clock expressions live in AttrStmt.value so Substitute
// and SSA conversion update them together with the materialized operations.
using PhysicalPipelineOwners = Map<tir::Var, Array<PrimExpr>>;
enum PhysicalPipelineSegment : int {
  kPipelinePrologue,
  kPipelineSteady,
  kPipelineEpilogue
};
inline bool IsPhysicalPipelineMetadata(const String &key) {
  return key == kPhysicalRingClock || key == kPhysicalPipelineExtent ||
         key == kPhysicalPipelineSegment || key == kPhysicalPipelineStage;
}

enum PhysicalRingField : size_t {
  kPhysicalRingSchema = 0,
  kPhysicalRingVersions = 1,
  kPhysicalRingRoot = 2,
  kPhysicalRingBaseBits = 3,
  kPhysicalRingStrideBits = 4,
  kPhysicalRingElementBits = 5,
  kPhysicalRingOriginalMin = 6,
  kPhysicalRingFieldCount = 7,
};

using PhysicalRingProvenance = Map<tir::Var, Array<PrimExpr>>;

inline void ValidatePhysicalRingRecord(const Array<PrimExpr> &record) {
  ICHECK_EQ(record.size(), kPhysicalRingFieldCount)
      << "Malformed physical ring provenance record";
  const auto *schema = record[kPhysicalRingSchema].as<IntImmNode>();
  const auto *versions = record[kPhysicalRingVersions].as<IntImmNode>();
  ICHECK(schema != nullptr && schema->value == 1)
      << "Unsupported physical ring provenance schema";
  ICHECK(versions != nullptr && versions->value > 1)
      << "Physical ring provenance requires more than one slot";
  ICHECK(record[kPhysicalRingRoot].as<tir::VarNode>() != nullptr)
      << "Physical ring provenance is missing its allocation root";
}

// Clock shells use Array<Var>{logical_id}, rather than a bare Var node. Legacy
// TVM Substitute rewrites bare AttrStmt Var nodes during physical pointer
// substitution. The opaque identity container keeps the logical id stable,
// while the shell's PrimExpr value follows stage skewing and loop unrolling.
inline tir::Var PhysicalRingClockId(const ObjectRef &node) {
  const Array<tir::Var> ids = Downcast<Array<tir::Var>>(node);
  ICHECK_EQ(ids.size(), 1U) << "Malformed physical ring clock identity";
  return ids[0];
}

} // namespace tvm::tl
