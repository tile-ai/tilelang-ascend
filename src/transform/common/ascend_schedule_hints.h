// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#pragma once

#include <tvm/runtime/container/map.h>
#include <tvm/tir/function.h>

namespace tvm {
namespace tl {

constexpr const char *kAscendTaskHint = "tl.ascend_task";
constexpr const char *kAscendStageHint = "tl.ascend_stage";
constexpr const char *kAscendPerCoreTaskHint = "tl.ascend_per_core_task";
constexpr const char *kAscendScheduleHintsPending =
    "tl.ascend_schedule_hints_pending";

bool HasAscendScheduleHints(const tir::Stmt &body);
void ValidateAscendScheduleHintStructure(const tir::PrimFunc &function);
void ValidateAscendTaskHintPayload(const tir::AttrStmtNode *hint);
void ValidateAscendStageHintPayload(const tir::AttrStmtNode *hint);

// The same effect validator serves explicit analysis and single-call
// hint consumption. A probe retains declaration and lexical scope context.
Map<String, ObjectRef> AnalyzeAscendSchedule(tir::PrimFunc function,
                                             String platform);

} // namespace tl
} // namespace tvm
