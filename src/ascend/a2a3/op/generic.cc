// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "generic.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>

#include "elem.h"
#include "gemm.h"
#include "reduce.h"

namespace tvm::tl::a2a3 {

bool IsAscendCompatibilityTarget(const Target &target) {
  auto arch = target->GetAttr<String>("arch");
  return target->kind->name == "ascend" && arch.has_value() &&
         arch.value() == "dav-2201";
}

// TA src/op/op.cc: the legacy region min is the exact load index, including
// Ramp nodes. Do not silently scalarize the region's index representation.
RegionOp::RegionOp(Array<PrimExpr> args, BufferMap) {
  ICHECK_GE(args.size(), 2U) << "A2/A3 tl.region requires a buffer and mask";
  const size_t ndim = args.size() - 2;
  const auto *load = args[0].as<BufferLoadNode>();
  ICHECK(load);
  ICHECK_EQ(load->indices.size(), ndim);
  buffer_ = load->buffer;
  const auto *mask = as_const_int(args[1]);
  ICHECK(mask) << "A2/A3 tl.region requires a constant access mask";
  access_mask_ = static_cast<int>(*mask);
  for (size_t i = 0; i < ndim; i++) {
    ranges_.push_back(Range::FromMinExtent(load->indices[i], args[2 + i]));
    extents_.push_back(args[2 + i]);
  }
}

bool RegionOp::IsFullRegion() const {
  for (size_t i = 0; i < ranges_.size(); i++) {
    if (!is_zero(ranges_[i]->min) ||
        !StructuralEqual()(ranges_[i]->extent, buffer_->shape[i])) {
      return false;
    }
  }
  return true;
}

Var GetVarFromAccessPtr(const PrimExpr &expr) {
  const auto *call = expr.as<CallNode>();
  ICHECK(call);
  ICHECK(call->op.same_as(builtin::tvm_access_ptr()));
  const auto *var = call->args[1].as<VarNode>();
  ICHECK(var);
  return GetRef<Var>(var);
}

namespace {

bool IsRegion(const PrimExpr &expr) {
  if (const auto *call = expr.as<CallNode>()) {
    if (auto op = call->op.as<Op>()) {
      return op.value()->name == "tl.region";
    }
  }
  return false;
}

PrimExpr FullRegionAccess(const PrimExpr &expr, const BufferMap &vmap) {
  if (!IsRegion(expr)) {
    return expr;
  }
  const auto *call = expr.as<CallNode>();
  RegionOp region(call->args, vmap);
  ICHECK(region.IsFullRegion())
      << "A2/A3 generic reduce expects full buffers, as in the TA API";
  return region.GetBuffer().access_ptr(region.GetAccessMask());
}

Array<PrimExpr> CopyArgs(const Call &call) {
  Array<PrimExpr> args = call->args;
  auto width = call->annotations.find("coalesced_width");
  if (width != call->annotations.end()) {
    ICHECK_EQ(args.size(), 2U)
        << "A2/A3 copy coalesced_width cannot be both positional and annotated";
    args.push_back(Downcast<PrimExpr>((*width).second));
  }
  return args;
}

Array<PrimExpr> FillArgs(const Call &call, const BufferMap &vmap) {
  Array<PrimExpr> args = call->args;
  ICHECK_EQ(args.size(), 2U);
  if (IsRegion(args[0])) {
    RegionOp region(args[0].as<CallNode>()->args, vmap);
    Array<PrimExpr> indices;
    for (const Range &range : region.GetRanges()) {
      indices.push_back(is_one(range->extent)
                            ? range->min
                            : Ramp(range->min, 1, range->extent));
    }
    args.Set(0, BufferLoad(region.GetBuffer(), indices));
  }
  return args;
}

Array<PrimExpr> ReduceArgs(const Call &call, const BufferMap &vmap) {
  Array<PrimExpr> args = call->args;
  ICHECK_EQ(args.size(), 5U);
  args.Set(0, FullRegionAccess(args[0], vmap));
  args.Set(1, FullRegionAccess(args[1], vmap));
  return args;
}

Array<PrimExpr> GemmArgs(const Call &call, const BufferMap &vmap) {
  Array<PrimExpr> args = call->args;
  if (args.size() == 13U) {
    // The shared dense protocol adds mbar/C coordinates; the original TA
    // generic operation has no supported Ascend lowering for those controls.
    Array<PrimExpr> original;
    for (size_t i = 0; i < 10; ++i)
      original.push_back(args[i]);
    for (size_t i = 0; i < 3; ++i) {
      if (IsRegion(original[i])) {
        RegionOp region(original[i].as<CallNode>()->args, vmap);
        original.Set(i, region.GetBuffer().access_ptr(region.GetAccessMask()));
      }
    }
    auto pack = call->annotations.find("k_pack");
    auto wait = call->annotations.find("wg_wait");
    original.push_back(pack == call->annotations.end()
                           ? PrimExpr(1)
                           : Downcast<PrimExpr>((*pack).second));
    original.push_back(wait == call->annotations.end()
                           ? PrimExpr(0)
                           : Downcast<PrimExpr>((*wait).second));
    args = original;
  }
  return args;
}

} // namespace

const Op &Copy::Get() {
  static const Op &op = Op::Get("tl.tileop.copy");
  return op;
}

const Op &Fill::Get() {
  static const Op &op = Op::Get("tl.tileop.fill");
  return op;
}

const Op &Gemm::Get() {
  static const Op &op = Op::Get("tl.tileop.gemm");
  return op;
}

const Op &ReduceOp::Get() {
  static const Op &op = Op::Get("tl.tileop.reduce");
  return op;
}

const Op &CumSumOp::Get() {
  static const Op &op = Op::Get("tl.tileop.cumsum");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  for (const char *name : {"tl.copy", "tl.tileop.copy"}) {
    RegisterOperatorParser(name, [](const Call &call, const BufferMap &vmap) {
      return std::make_unique<Copy>(CopyArgs(call), vmap);
    });
  }
  for (const char *name : {"tl.fill", "tl.tileop.fill"}) {
    RegisterOperatorParser(name, [](const Call &call, const BufferMap &vmap) {
      return std::make_unique<Fill>(FillArgs(call, vmap), vmap);
    });
  }
  for (const char *name : {"tl.reduce", "tl.tileop.reduce"}) {
    RegisterOperatorParser(name, [](const Call &call, const BufferMap &vmap) {
      return std::make_unique<ReduceOp>(ReduceArgs(call, vmap), vmap);
    });
  }
  for (const char *name : {"tl.cumsum", "tl.tileop.cumsum"}) {
    RegisterOperatorParser(name, [](const Call &call, const BufferMap &vmap) {
      Array<PrimExpr> args = call->args;
      ICHECK_EQ(args.size(), 4U);
      args.Set(0, FullRegionAccess(args[0], vmap));
      args.Set(1, FullRegionAccess(args[1], vmap));
      return std::make_unique<CumSumOp>(args, vmap);
    });
  }
  for (const char *name : {"tl.gemm", "tl.tileop.gemm"}) {
    RegisterOperatorParser(name, [](const Call &call, const BufferMap &vmap) {
      return std::make_unique<Gemm>(GemmArgs(call, vmap), vmap);
    });
  }
}

} // namespace tvm::tl::a2a3
