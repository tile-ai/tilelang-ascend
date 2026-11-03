// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

// Admission and fence effects for existing legacy mode-2 C->V publication.
// This module never creates, moves, or renumbers a cross-core instruction.

#pragma once

#include <tvm/target/target.h>
#include <tvm/tir/function.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "upstream/dependency_analysis.h"

namespace tvm::tl::ascend_sync {

enum class LegacyCrossFlagKind { kSet, kWait };

struct LegacyCrossFlag {
  LegacyCrossFlagKind kind;
  int id;
  // Only Set supplies a mode; Wait uses the unchanged legacy codegen ABI.
  int mode;
  std::string pipe;
};

// Supports the four actual legacy op ABIs. Malformed/nonliteral cross calls
// are diagnosed, rather than being mistaken for an unrelated operation.
std::optional<LegacyCrossFlag>
DecodeLegacyCrossFlag(const tir::Stmt &statement);

struct LegacyOwnedTask {
  std::shared_ptr<TaskNode> task;
  // Mutable Let definitions use their original LetStmt here, even though the
  // scalar definition Task's stmt is Evaluate(let->value).
  tir::Stmt original_statement;
  CoreMask owner;
  // Exact pre-order Stmt occurrence, including structural wrappers. This is
  // the existing legacy export occurrence, not a data-Var/name-based key.
  size_t order;
};

struct LegacyOwnedControl {
  std::shared_ptr<ControlNode> control;
  tir::Stmt original_statement;
  CoreMask owner;
  size_t order;
};

struct LegacyCrossShadowEffect {
  std::shared_ptr<TaskNode> task;
  uint16_t pipe_mask;
  std::vector<BufferRegion> reads;
  std::vector<BufferRegion> writes;
};

struct LegacyOwnedFlagReservation {
  CoreMask owner;
  bool cross_core;
  // Empty for cross-core channels; local channels retain their hard-event.
  std::string event;
  int id;
};

struct LegacyCrossCorePlan {
  // These are analysis-only fence effects, separate from real GM effects.
  // C Set reads on FIX; V Wait writes on S. Trusted fresh-workspace mode also
  // completes C input reads before publication and delays V output writes.
  // The original calls remain intact.
  std::vector<LegacyCrossShadowEffect> shadows;
  Array<tir::Var> shared_gm_storages;
  std::vector<LegacyOwnedFlagReservation> reservations;
};

// Pass-local certificate for the complete old flat proof. Successful analysis
// owns its plan; callers neither recreate the proof nor attach partial effects.
// This grants only compiler-added workspace facts, never full recipe authority.
class LegacyFlatEligibility {
public:
  bool IsEligible() const { return plan_.has_value(); }
  const std::string &Rejection() const { return rejection_; }
  LegacyCrossCorePlan TakePlan();

private:
  std::optional<LegacyCrossCorePlan> plan_;
  std::string rejection_;

  friend LegacyFlatEligibility AnalyzeLegacyFlatEligibility(
      const tir::PrimFunc &, const Target &, const std::string &,
      const std::vector<LegacyOwnedTask> &, const ConflictOracle &,
      const std::vector<LegacyOwnedControl> &, bool);
};

// Mixed-resource contract: A2/A3 AscendC, B1, flat V and at most one static
// positive C-only prefix For. C control trees stay in their local graph;
// only their real leaves participate in cross-resource preflight.
// explicit noalias (or a trusted Cython fresh-workspace contract with one
// workgroup and one audited disjoint V output), C1/V2, and one literal mode-2
// FIX C->V publication.
// C1/V2 is either the legacy blockIdx.y extent2 or Developer threadIdx.x
// extent2 with StringImm npu_cv_ratio="cv_1_2". Ambiguous topologies are
// rejected. Every real cross-resource RAW/WAR/WAW is checked with the physical
// oracle. The caller must enable ordinary synchronization for both owner-local
// runs. No input Task or original statement is changed by this function.
// Expected profile/proof misses return a rejection, without catching errors or
// mutating tasks. The oracle must use GMFreshnessScope::kCompilerAdded. Generic
// mixed GEMM invocations require the closed proof even if their real GM effects
// otherwise fit the old profile. Malformed shared contracts remain errors.
LegacyFlatEligibility
AnalyzeLegacyFlatEligibility(const tir::PrimFunc &function,
                             const Target &target, const std::string &platform,
                             const std::vector<LegacyOwnedTask> &tasks,
                             const ConflictOracle &compiler_added_oracle,
                             const std::vector<LegacyOwnedControl> &controls,
                             bool requires_generic_composite);

// Single existing C->V publication over owner-pure static For trees. This
// profile audits actual DMA rectangles across workgroups and AIV executions;
// unknown aliases/geometry reject. The optional private pool identity belongs
// only to composite helper analysis and remains in all owner-local graphs.
// flat_rejection is diagnostic-only context appended after the actual closed
// failure; it cannot change proof conditions or trigger another planner.
LegacyCrossCorePlan
PlanLegacyClosedCrossCore(const tir::PrimFunc &function, const Target &target,
                          const std::string &platform,
                          const std::vector<LegacyOwnedTask> &tasks,
                          const ConflictOracle &physical_conflict_oracle,
                          const std::vector<LegacyOwnedControl> &controls,
                          Optional<tir::Var> private_composite_pool = NullOpt,
                          const std::string &flat_rejection = {});

} // namespace tvm::tl::ascend_sync
