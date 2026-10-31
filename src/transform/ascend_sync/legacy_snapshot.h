// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

// Runtime snapshots for the legacy synchronization adapter. Unlike the
// proof-only FreshenMutableReads, these bindings really evaluate the original
// memory reads exactly once at their original statement boundary.

#pragma once

#include <tvm/tir/function.h>

#include <unordered_set>

namespace tvm::tl::ascend_sync {

struct LegacySnapshotResult {
  tir::PrimFunc function;
  // These definitions read mutable memory. Import them as scalar read tasks
  // with a write to the bound Var, without asserting var == mutable value in
  // later proof contexts. The value itself remains in the runtime LetStmt.
  std::unordered_set<tir::Var, ObjectPtrHash, ObjectPtrEqual> mutable_bindings;
};

// Existing LetStmt definitions keep their original evaluation and lexical
// scope. If/else captures its condition once and becomes two sequential Ifs
// with complementary conditions and independent insertion anchors. Mutable If
// conditions and configured hardware scalar operands also gain a LetStmt at
// their original evaluation point. Opaque expressions, non-scalar snapshots
// and mutable For bounds remain unsupported.
LegacySnapshotResult NormalizeLegacySnapshots(const tir::PrimFunc &function);

} // namespace tvm::tl::ascend_sync
