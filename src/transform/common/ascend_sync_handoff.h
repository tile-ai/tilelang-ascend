// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#pragma once

#include <tvm/tir/function.h>

namespace tvm {
namespace tl {
constexpr const char *kAscendHandoffSource = "tl.ascend_handoff_source";
constexpr const char *kAscendHandoffTaskId = "tl.ascend_handoff_task_id";
constexpr const char *kAscendHandoffIteration = "tl.ascend_handoff_iteration";
constexpr const char *kAscendHandoffPlan = "tl.ascend_handoff_plan";
// The stable CANN900 A3 API exposes IDs 0..10. The hardware counter namespace
// is larger; SDK system synchronization and the current ABI use IDs 11..15.
constexpr int kAscendA3HandoffUserEvents = 11;
constexpr int kAscendA3CrossCounterDepth = 15;
bool HasAscendSyncHandoff(const tir::PrimFunc &function);
} // namespace tl
} // namespace tvm
