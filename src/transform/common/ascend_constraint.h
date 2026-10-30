// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#pragma once

#include <tvm/ir/transform.h>
#include <tvm/tir/function.h>

namespace tvm {
namespace tl {

constexpr const char *kAscendPipelinePlanner = "tl.ascend_pipeline_planner";
constexpr const char *kAscendConstraintTaskId = "tl.ascend_constraint_task_id";
constexpr const char *kAscendConstraintIteration =
    "tl.ascend_constraint_iteration";
constexpr const char *kAscendConstraintPlan = "tl.ascend_constraint_plan";
constexpr const char *kAscendPipelineCostModel =
    "tl.ascend_pipeline_cost_model";
constexpr const char *kAscendPipelineTaskHint = "tl.ascend_pipeline_task_hint";
constexpr const char *kAscendPipelineLoopId = "tl.ascend_pipeline_loop_id";
constexpr const char *kAscendPipelineVersions = "tl.ascend_pipeline_versions";
constexpr const char *kAscendPipelineBufferRemap =
    "tl.ascend_pipeline_buffer_remap";
constexpr const char *kAscendPipelineDegenerateCandidates =
    "tl.ascend_pipeline_degenerate_candidates";
constexpr const char *kAscendPersistentGuard = "tl.ascend_persistent_guard";
// Shared with the existing allocator; this is its compiler budget, not a new
// independently maintained scheduling capacity table.
constexpr int64_t kAscendA2A3UBBudget = 196352;

tir::PrimFunc PlanAscendConstraintPipeline(tir::PrimFunc function,
                                           tvm::transform::PassContext context);
Map<String, ObjectRef> AnalyzeAscendScheduleEffects(tir::PrimFunc function,
                                                    String platform);

} // namespace tl
} // namespace tvm
