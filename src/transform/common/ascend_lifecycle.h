// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#pragma once

#include <tvm/tir/function.h>

namespace tvm {
namespace tl {

tir::PrimFunc MarkAscendScheduleSource(tir::PrimFunc source,
                                       Map<String, ObjectRef> graph);
Map<String, ObjectRef>
BuildAscendLifetimeContract(tir::PrimFunc source,
                            Map<String, ObjectRef> canonical_graph,
                            Map<String, ObjectRef> canonical_catalog);
Map<String, ObjectRef> AnalyzeAscendLifetimes(tir::PrimFunc function,
                                              String platform);
Map<String, ObjectRef>
RebuildAscendLifetimeContract(tir::PrimFunc marked_source);
int64_t EvaluateAscendLifetimeInteger(PrimExpr expression,
                                      Map<tir::Var, PrimExpr> bindings);
void ValidateAscendLifetimeCallIndices(tir::Call call,
                                       Map<tir::Var, PrimExpr> bindings);

} // namespace tl
} // namespace tvm
