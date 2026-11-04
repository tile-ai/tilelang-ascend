// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

#pragma once

#include <tvm/tir/function.h>

namespace tvm::tl::ascend_sync {

struct LegacyManualEventResult {
  tir::PrimFunc function;
  bool normalized{false};
  // -1: unannotated pipes shared by both owners; 0: Cube; 1: Vector.
  int required_owner{-1};
  // A candidate only: mixed execution also requires the complete epoch proof.
  bool cross_core{false};
};

// Prove that explicit intra-core SetFlag/WaitFlag calls are a forward token
// protocol under sequential execution, then remove those calls. This is only
// a candidate normalization: the caller must require ordinary automatic sync,
// successful complete memory-effect import, and either one execution owner or
// a complete cross-core epoch proof. Cross-core calls are never removed, and
// automatically generated hard flags cannot be normalized.
// Flat literal protocols are left to the existing validator/reservation path.
// Repeated pure SSA conditions outside loops share a Boolean choice; every
// combination is checked, within a shared finite protocol execution budget.
LegacyManualEventResult
NormalizeLegacyManualEvents(const tir::PrimFunc &function);

} // namespace tvm::tl::ascend_sync
