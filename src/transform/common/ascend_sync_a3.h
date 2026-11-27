// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#pragma once

#include <tvm/target/target.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {
constexpr const char *kAscendSyncStrategy = "tl.ascend_sync_strategy";
constexpr const char *kAscendLocalSyncPlan = "tl.ascend_local_sync_plan";
// CANN9.1 kernel_macros.h: __NPU_ARCH__=2201 has QUE_MAX_EVENT=8;
// kernel_tquesync_impl.h checks SetFlag/WaitFlag IDs against that limit.
constexpr int kAscendA3LocalEventCount = 8;
Map<String, ObjectRef>
AllocateAscendSyncEvents(Array<Map<String, ObjectRef>> intervals,
                         String platform);
} // namespace tl
} // namespace tvm
