// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#pragma once

#include <tvm/target/target.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {
using AutoRecord = Map<String, ObjectRef>;

constexpr const char *kAscendAutoSchedule = "tl.ascend_auto_schedule";
constexpr const char *kAscendAutoScheduleTaskId =
    "tl.ascend_auto_schedule_task_id";
constexpr const char *kAscendAutoScheduleBufferRemap =
    "tl.ascend_auto_schedule_buffer_remap";
constexpr const char *kAscendAutoSchedulePrepared =
    "tl.ascend_auto_schedule_prepared";
constexpr const char *kAscendAutoScheduleSyncPlan =
    "tl.ascend_auto_schedule_sync_plan";
constexpr int64_t kAscendAutoScheduleTaskLimit = 256;
constexpr int64_t kAscendAutoScheduleControlExpansionLimit = 65536;

// Read-only finite source analysis. Each task has id/group/actor (0=C, 1/2=V),
// epoch/loop_coords/key, statement/call, operation and
// reads/writes/must_defines/ scratch. Effects contain data/scope/dtype and
// signed-int64 byte begin/end. Storages contain
// data/buffer/scope/origin/capacity_bytes/allocation_site_id.
AutoRecord AnalyzeAscendAutoSource(tir::PrimFunc function);
AutoRecord AuditAscendAutoCall(tir::Call call, int actor);
int64_t AscendAutoMemoryBudget(const String &scope);
void CheckAscendAutoScheduleContext(Target target, String platform);

// Independently rebuild source definitions and cross-actor FIFO bindings.
// Returns read_producers, logical_deps and pin_n1; the planner must not infer
// initialization merely from an allocated storage or from a may-write hull.
AutoRecord BuildAscendAutoSourceContract(AutoRecord source);

// Whole-kernel auto backend only. The public Python route calls these in order;
// no entry point invokes the legacy planner, injector or sync passes.
tir::transform::Pass AscendPrepareAutoSchedule(Target target);
tir::transform::Pass AscendAutoSchedule(Target target, String platform);
tir::transform::Pass AscendMaterializeSchedule(Target target, String platform);
tir::transform::Pass AscendInsertScheduledSync(Target target, String platform);
tir::transform::Pass AscendVerifyAutoSchedule(Target target, String platform);

} // namespace tl
} // namespace tvm
