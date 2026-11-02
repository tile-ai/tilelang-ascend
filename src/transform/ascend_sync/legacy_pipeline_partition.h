// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

#pragma once

#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>

#include <optional>
#include <vector>

#include "upstream/constraints.h"

namespace tvm::tl::ascend_sync {

struct LegacyPipelineCase {
  PrimExpr condition;
  tir::PrimFunc function;
  bool zero_work{false};
  tir::Var steady_loop_var;
  PrimExpr steady_extent;

  // Internal admission capability, constructed only by the verified
  // partitioner. A public attribute cannot enable arbitrary ring loops.
  bool AllowsSteadyLoop(const tir::For &loop, const ConstrSet &context) const;
};

struct LegacyPipelinePartition {
  tir::PrimFunc original;
  tir::Var owner;
  PrimExpr extent;
  int lead{0};
  std::vector<LegacyPipelineCase> cases;
};

// Require a genuinely empty executable body, including no mutable reads or
// opaque effects in wrappers. This also handles static zero-trip pipelines
// whose unused provenance survives removal of their original allocation.
bool IsLegacyPipelineNoOp(const tir::PrimFunc &function);

// Only one top-level, injector-certified SIP family is admitted. Static
// families keep the established path. Dynamic cases retain shared allocation
// shells, maps and the true version table while removing only verified bounds
// guards and proven-empty steady loops.
std::optional<LegacyPipelinePartition>
PartitionLegacyDynamicPipeline(const tir::PrimFunc &function);

// Extract synchronized envelope bodies and put each complete protocol inside
// its actual runtime case. The surrounding thread/Allocate/Let shells remain
// single instances. SSA conversion is limited to local bindings in the body.
tir::PrimFunc
MergeLegacyPipelineCases(const LegacyPipelinePartition &partition,
                         const std::vector<tir::PrimFunc> &synchronized);

} // namespace tvm::tl::ascend_sync
