// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

#pragma once

#include <tvm/target/target.h>
#include <tvm/tir/function.h>

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>

#include "physical_storage.h"

namespace tvm::tl::ascend_sync {

struct LegacyTailCopyEffect {
  tir::BufferRegion source_read;
  tir::BufferRegion destination_write;
};

// Audits the existing GM-to-UB helpers without rewriting their invocations.
// Both runtime paths have an MTE2 acquire/completion boundary: on the padded
// path MTE2_V#0 places Duplicate after acquire, and V_MTE2#0 places its writes
// before DMA/completion. Explicit events must not use those helper-owned IDs.
// The bounded PTO profile instead executes TLOAD before TFILLPAD_INPLACE;
// MTE2_V#0 orders the fill after the DMA, and V_MTE2#0 feeds completion back
// to MTE2. It additionally reserves V_MTE3#0, rather than MTE3_V#0.
class LegacyTailCopyPlan {
public:
  LegacyTailCopyPlan(const tir::PrimFunc &function, const Target &target,
                     const String &platform);

  // A provably full/default-mask call returns nullopt, retaining the existing
  // full-copy path, including its target support. Otherwise validate the
  // complete runtime full/tail envelope in this call's real proof context.
  // The returned regions preserve logical data identities. The destination
  // includes the whole Duplicate tile; GM reads include the true row stride.
  std::optional<LegacyTailCopyEffect>
  AuditCall(const tir::CallNode *call, size_t argument_offset,
            const ConstrSet &context, const PhysicalStorageResolver &storage);

  const std::map<std::string, std::set<int>> &ReservedVectorIntraIds() const;
  bool HasTailCopies() const;

  // Run after auditing every call and before allocating any event. The
  // constructor has collected all original explicit events, so an event that
  // lexically precedes the first padded copy is checked as well.
  void ValidateManualEvents() const;

private:
  struct State;
  std::shared_ptr<State> state_;
};

} // namespace tvm::tl::ascend_sync
