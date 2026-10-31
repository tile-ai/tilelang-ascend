// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

#pragma once

#include <tvm/target/target.h>
#include <tvm/tir/function.h>

#include <memory>
#include <optional>
#include <vector>

#include "legacy_persistent_normalize.h"
#include "physical_storage.h"
#include "target_policy.h"

namespace tvm::tl::ascend_sync {

// Pass-local capability, created only after the closed single-owner audit.
// Its source and occurrence are the exact current normalized AST, not public
// attributes. Inner analysis keeps the real B2 epoch; the outer task uses
// analysis-only complete-invocation views that retain physical alias links.
class LegacyPersistentPipelineImport {
public:
  bool MatchesSource(const tir::PrimFunc &function) const;
  const tir::Stmt &ScopedRoot() const { return scoped_root_; }
  const ConstrSet &RootContext() const { return root_context_; }
  size_t RootOccurrence() const { return root_occurrence_; }
  const tir::Stmt &ClosedOwner() const { return closed_owner_; }
  const tir::Stmt &ClosedReplacement() const { return closed_replacement_; }
  size_t ClosedOccurrence() const { return closed_occurrence_; }
  const std::vector<tir::BufferRegion> &Reads() const { return reads_; }
  const std::vector<tir::BufferRegion> &Writes() const { return writes_; }
  const std::vector<tir::Var> &ReadVars() const { return read_vars_; }
  bool AllowsOuterLoop(const tir::For &loop) const;
  const tir::Stmt &Reservations() const { return reservations_; }
  const std::shared_ptr<PhysicalStorageResolver> &Storage() const {
    return storage_;
  }

private:
  LegacyPersistentPipelineImport() = default;
  friend std::optional<tir::PrimFunc>
  SynchronizeLegacyPersistentPipeline(const LegacyPersistentNormalization &,
                                      const Target &, const String &, bool,
                                      const SyncTargetPolicy &);

  tir::PrimFunc source_;
  tir::For outer_loop_;
  tir::Stmt scoped_root_, closed_owner_, closed_replacement_;
  ConstrSet root_context_;
  size_t root_occurrence_{0}, closed_occurrence_{0};
  std::vector<tir::BufferRegion> reads_, writes_;
  std::vector<tir::Var> read_vars_;
  tir::Stmt reservations_;
  std::shared_ptr<PhysicalStorageResolver> storage_;
};

// Return nullopt for an input without the combination. A nested owner that
// resembles this feature but fails its private contract is diagnosed. This
// executes the actual upstream engine for both the inner B2 graph and the
// outer S->S invocation graph, then returns one function with unchanged
// allocations, version/provenance tables and planned addresses.
std::optional<tir::PrimFunc> SynchronizeLegacyPersistentPipeline(
    const LegacyPersistentNormalization &normalization, const Target &target,
    const String &platform, bool ordinary_sync, const SyncTargetPolicy &policy);

} // namespace tvm::tl::ascend_sync
