// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

// Legacy-adapter code: no upstream synchronization algorithm is duplicated.
// Physical byte views remain local to the conflict oracle; logical TaskNode
// regions and their data Vars are never replaced by a common allocation Var.

#pragma once

#include <tvm/runtime/container/map.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>

#include <memory>

#include "upstream/dependency_analysis.h"

namespace tvm::tl::ascend_sync {

using PhysicalRingClockMap = Map<tir::Var, PrimExpr>;

class PhysicalStorageResolver {
public:
  // The one-argument form reads address_map/size_map from function attributes.
  explicit PhysicalStorageResolver(tir::PrimFunc function);
  PhysicalStorageResolver(tir::PrimFunc function,
                          Map<tir::Var, PrimExpr> address_map,
                          Map<tir::Var, PrimExpr> size_map);

  // Keep pointer offsets/extents in their declared element units. This is a
  // logical access view, not proof that a DMA only touches that interval.
  tir::BufferRegion AccessPtrRegion(const tir::CallNode *access_ptr) const;
  // A materialized ring requires its injector-published logical clock. The
  // returned view retains the stable logical ring id and covers one COMPLETE
  // physical stride, including padding, after validating the actual pointer.
  tir::BufferRegion AccessPtrRegion(const tir::CallNode *access_ptr,
                                    const PhysicalRingClockMap &clocks,
                                    const ConstrSet &context) const;
  BufferVersionMap ManualRingVersions() const;
  bool IsPhysicalRingStorage(const tir::Var &data) const;
  void ValidateRingClock(const tir::Var &logical_id, const PrimExpr &clock,
                         const ConstrSet &context) const;
  tir::Buffer LookupBuffer(const tir::Var &data) const;

  // Analysis-only invocation view: a fresh single-version logical identity
  // aliases the original complete physical allocation. It never enters
  // emitted TIR or function maps, and never changes the original ring B.
  tir::BufferRegion InvocationProjection(const tir::BufferRegion &region);

  // The closure owns its analysis state and can outlive the resolver. Every
  // oracle call compares byte regions under the original logical contexts and
  // iteration relation through the upstream RegionsMayConflict implementation.
  ConflictOracle MakeOracle() const;

private:
  struct State;
  std::shared_ptr<State> state_;
};

} // namespace tvm::tl::ascend_sync
