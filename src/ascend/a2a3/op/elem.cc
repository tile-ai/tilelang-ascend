// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/op/elem.cc
 *
 * Define elment-wise operators.
 */

#include "elem.h"

#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

#include "../transform/common/loop_fusion_utils.h"
#include "../transform/loop_partition.h"
#include "../transform/loop_vectorize.h"
#include "builtin.h"
#include "generic.h"

namespace tvm {
namespace tl {
namespace a2a3 {

using namespace tirx;
using namespace ffi;

Copy::Copy(Array<PrimExpr> args, BufferMap vmap) : args_(args) {
  ICHECK_GE(args.size(), 2U);
  ICHECK_LE(args.size(), 3U);
  Array<Range> rgs[2];
  Buffer bf[2];
  for (int i = 0; i < 2; i++) {
    auto expr = args[i];
    auto call = expr.as<CallNode>();
    ICHECK(call);
    auto region = RegionOp(call->args, vmap);
    rgs[i] = region.GetRanges();
    bf[i] = region.GetBuffer();
  }
  std::tie(this->src, this->dst) = std::tie(bf[0], bf[1]);
  std::tie(this->src_range, this->dst_range) = std::tie(rgs[0], rgs[1]);
  if (args.size() >= 3) {
    coalesced_width = Downcast<IntImm>(args[2]);
  }
}

Array<IterVar> Copy::MakeIterVars() const {
  Array<IterVar> loop_vars;
  size_t idx = 0;
  for (size_t i = 0; i < src_range.size(); i++) {
    if (is_one(src_range[i]->extent))
      continue;
    Var var = Var(std::string{char('i' + idx)}, src_range[i]->extent->dtype);
    idx++;
    loop_vars.push_back(
        {Range(0, src_range[i]->extent), var, IterVarType::kDataPar});
  }
  return loop_vars;
}

// ivs: itervars returned by MakeIterVars()
// src_dst: 0 for src_indices, 1 for dst_indices
Array<PrimExpr> Copy::MakeIndices(const Array<IterVar> &ivs,
                                  int src_dst) const {
  Array<PrimExpr> indices;
  Array<Range> ranges = src_dst == 0 ? src_range : dst_range;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (is_one(ranges[i]->extent))
      indices.push_back(ranges[i]->min);
    else {
      indices.push_back(ranges[i]->min + ivs[idx]->var);
      idx++;
    }
  }
  ICHECK(idx == ivs.size())
      << "idx = " << idx << ", ivs.size() = " << ivs.size()
      << "src name = " << src->name << ", dst name = " << dst->name;
  return indices;
}

PrimExpr Copy::MakePredicate(arith::Analyzer *analyzer,
                             const Array<IterVar> &ivs, Array<PrimExpr> extents,
                             int src_dst) const {
  Array<Range> ranges = src_dst == 0 ? src_range : dst_range;
  Array<PrimExpr> cond_list;
  ICHECK(extents.size() == ranges.size()) << extents << " " << ranges;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (is_one(ranges[i]->extent))
      continue;
    PrimExpr cond = ranges[i]->min + ivs[idx]->var < extents[i];
    if (!analyzer->CanProve(cond, arith::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
    cond = ranges[i]->min + ivs[idx]->var >= 0;
    if (!analyzer->CanProve(cond, arith::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
    idx++;
  }
  if (cond_list.empty())
    return {};
  else {
    PrimExpr cond = cond_list[0];
    for (size_t i = 1; i < cond_list.size(); i++)
      cond = And(cond, cond_list[i]);
    return cond;
  }
}

For Copy::MakeSIMTLoop(arith::Analyzer *analyzer) const {
  Array<IterVar> loop_vars = MakeIterVars();
  bool is_scalar = loop_vars.size() == 0;
  if (is_scalar) {
    return For(Var("i"), 0, 1, ForKind::kSerial,
               BufferStore(dst, BufferLoad(src, {0}), {0}));
  }

  for (const auto &iv : loop_vars)
    analyzer->Bind(iv->var, iv->dom);

  ICHECK(loop_vars.size() <= src_range.size())
      << "loop_vars.size() = " << loop_vars.size()
      << ", src_range.size() = " << src_range.size() << ", src = " << src->name
      << ", dst = " << dst->name;

  ICHECK(loop_vars.size() <= dst_range.size())
      << "loop_vars.size() = " << loop_vars.size()
      << ", dst_range.size() = " << dst_range.size() << ", src = " << src->name
      << ", dst = " << dst->name;

  Array<PrimExpr> src_indices = MakeIndices(loop_vars, 0);
  Array<PrimExpr> dst_indices = MakeIndices(loop_vars, 1);

  PrimExpr src_predicate = MakePredicate(analyzer, loop_vars, src->shape, 0);
  PrimExpr dst_predicate = MakePredicate(analyzer, loop_vars, dst->shape, 1);

  PrimExpr value = BufferLoad(src, src_indices);
  if (src->dtype != dst->dtype)
    value = Cast(dst->dtype, value);
  if (src_predicate.defined())
    value = if_then_else(src_predicate, value, make_zero(dst->dtype));

  Stmt body = BufferStore(dst, value, dst_indices);
  if (dst_predicate.defined())
    body = IfThenElse(dst_predicate, body);
  for (int i = loop_vars.size() - 1; i >= 0; i--) {
    Map<String, ObjectRef> annotations = {};
    if (coalesced_width.defined()) {
      annotations.Set("coalesced_width", coalesced_width);
    }
    body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent,
               ForKind::kParallel, body, std::nullopt, annotations);
  }
  return Downcast<For>(body);
}

Stmt Copy::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  Target target = T.target;
  // TA represented Ascend as llvm -keys=ascend (kDLCPU). Preserve that
  // exact lowering branch after migration to the native Ascend target kind.
  bool is_cpu_target = target->GetTargetDeviceType() == kDLCPU ||
                       IsAscendCompatibilityTarget(target);
  Stmt ldsm_stmt = LowerLDSMCopy(T, analyzer);
  if (ldsm_stmt.defined())
    return ldsm_stmt;

  Stmt bulk_copy_stmt = LowerBulkCopy(T, analyzer);
  if (bulk_copy_stmt.defined())
    return bulk_copy_stmt;
  auto simt_loop = MakeSIMTLoop(analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));

  For vectorized_thread_loop;
  auto par_op = std::make_unique<ParallelOp>(fused_loop);

  if (is_cpu_target) {
    vectorized_thread_loop = VectorizeLoop(fused_loop);
  } else {
    std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                      InferLevel::kFree};
    for (auto level : levels) {
      par_op->InferLayout(
          {T.target, T.thread_bounds, T.layout_map, T.buffer_remap}, level);
    }
    auto loop_layout = par_op->GetLoopLayout();
    auto thread_var = T.thread_var;
    auto thread_loop =
        PartitionLoop(par_op->GetRoot(), T.thread_var, analyzer, loop_layout);
    vectorized_thread_loop = VectorizeLoop(thread_loop);
  }

  if (par_op->GetPredicate(T.thread_var).defined()) {
    return IfThenElse(par_op->GetPredicate(T.thread_var).value(),
                      vectorized_thread_loop);
  }

  return vectorized_thread_loop;
}

// TA's LDSM/STSM and Hopper bulk-copy gates are always false for an
// Ascend target. Keep the original fall-through without GPU implementations.
Stmt Copy::LowerLDSMCopy(const LowerArgs &, arith::Analyzer *) const {
  return {};
}
Stmt Copy::LowerBulkCopy(const LowerArgs &, arith::Analyzer *) const {
  return {};
}

LayoutMap Copy::InferLayout(const LayoutInferArgs &T, InferLevel level) {
  // Use parallel op to infer the layout
  if (par_op_ == nullptr) {
    arith::Analyzer analyzer;
    par_op_ = std::make_unique<ParallelOp>(MakeSIMTLoop(&analyzer));
  }
  if (T.layout_map.count(src) && T.layout_map.count(dst)) {
    // Only compare fragment layout
    if (src.scope() == "local.fragment" && dst.scope() == "local.fragment") {
      const FragmentNode *src_layout =
          T.layout_map[src].as<Fragment>().value().get();
      const FragmentNode *dst_layout =
          T.layout_map[dst].as<Fragment>().value().get();
      if (src_layout && dst_layout) {
        ICHECK(src_layout->IsEqual(dst_layout, true))
            << "Get different layout for " << src << " and " << dst
            << "\nLHS = " << src_layout->DebugOutput()
            << "\nRHS = " << dst_layout->DebugOutput()
            << "\nYou may need to use a shared memory to transform the layout";
      }
    }
  }
  return par_op_->InferLayout(T, level);
}

Fill::Fill(Array<PrimExpr> args, BufferMap vmap) {
  ICHECK_EQ(args.size(), 2U);

  if (args[0]->IsInstance<BufferLoadNode>()) {
    auto buffer_load = Downcast<BufferLoad>(args[0]);
    for (const auto &index : buffer_load->indices) {
      if (const auto *ramp = index.as<RampNode>()) {
        ICHECK(ramp->stride.as<IntImmNode>()->value == 1)
            << "Only stride 1 ramps are supported";
        const auto *lanes = ramp->lanes.as<IntImmNode>();
        ICHECK(lanes)
            << "Scalable vectors not supported in BufferRegion conversion";
        region.push_back(Range::FromMinExtent(ramp->base, ramp->lanes));
      } else {
        region.push_back(Range::FromMinExtent(index, 1));
      }
    }
    dst = buffer_load->buffer;
  } else {
    dst = vmap[GetVarFromAccessPtr(args[0])];
    for (int i = 0; i < dst->shape.size(); i++) {
      region.push_back(Range(0, dst->shape[i]));
    }
  }

  if (args[1]->dtype != dst->dtype) {
    value = Cast(dst->dtype, args[1]);
  } else {
    value = args[1];
  }

  ICHECK(region.size() == dst->shape.size())
      << "region size = " << region.size() << " != " << dst->shape.size();
  for (int i = 0; i < region.size(); i++) {
    // bound check if region is static
    if (region[i]->min.as<IntImm>()) {
      int64_t min = Downcast<IntImm>(region[i]->min)->value;
      ICHECK_GE(min, 0) << "region[" << i << "] = " << min << " < 0";
    }
    if (region[i]->extent.as<IntImm>()) {
      int64_t extent = Downcast<IntImm>(region[i]->extent)->value;
      ICHECK_LE(extent, Downcast<IntImm>(dst->shape[i])->value)
          << "region[" << i << "] = " << extent << " > " << dst->shape[i];
    }
  }
}

For Fill::MakeSIMTLoop(arith::Analyzer *analyzer) const {
  int ndim = dst->shape.size();
  Array<IterVar> loop_vars;
  Array<PrimExpr> dst_indices;
  for (int i = 0; i < ndim; i++) {
    Var var = Var(std::string{char('i' + i)}, region[i]->extent->dtype);
    loop_vars.push_back({region[i], var, IterVarType::kDataPar});
    dst_indices.push_back(var);
  }
  Stmt body = BufferStore(dst, value, dst_indices);
  for (int i = ndim - 1; i >= 0; i--) {
    body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent,
               ForKind::kParallel, body);
  }
  return Downcast<For>(body);
}

Stmt Fill::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {

  if (dst.scope() == "local.fragment") {
    auto par_op = std::make_unique<ParallelOp>(MakeSIMTLoop(analyzer));
    par_op->InferLayout({T.target, T.thread_bounds, T.layout_map},
                        InferLevel::kFree);
    par_op->InferLayout({T.target, T.thread_bounds, T.layout_map},
                        InferLevel::kFree);
    auto thread_loop = PartitionLoop(par_op->GetRoot(), T.thread_var, analyzer,
                                     par_op->GetLoopLayout());
    auto vectorized_thread_loop = VectorizeLoop(thread_loop);
    if (par_op->GetPredicate(T.thread_var).defined()) {
      return IfThenElse(par_op->GetPredicate(T.thread_var).value(),
                        vectorized_thread_loop);
    }
    return vectorized_thread_loop;
  } else if (dst.scope() == "local") {
    auto init_loop = MakeSIMTLoop(analyzer);
    auto vectorized_thread_loop = VectorizeLoop(init_loop);
    return vectorized_thread_loop;
  } else if (dst.scope() == "shared.dyn" || dst.scope() == "shared") {
    auto par_op = std::make_unique<ParallelOp>(MakeSIMTLoop(analyzer));
    par_op->InferLayout({T.target, T.thread_bounds, T.layout_map},
                        InferLevel::kFree);
    auto thread_loop = PartitionLoop(par_op->GetRoot(), T.thread_var, analyzer,
                                     par_op->GetLoopLayout());
    auto vectorized_thread_loop = VectorizeLoop(thread_loop);
    return vectorized_thread_loop;
  } else {
    LOG(FATAL) << "Unsupported scope " << dst.scope();
  }
  return {};
}

} // namespace a2a3
} // namespace tl
} // namespace tvm
