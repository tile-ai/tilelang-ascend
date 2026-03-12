// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file legalize_npuir_bf16.cc
 * \brief Legalize BF16 tl.npuir_add into fp32 compute plus casts.
 */

#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <string>
#include <utility>
#include <vector>

#include "../op/op.h"
#include "arith/ir_mutator_with_analyzer.h"

namespace tvm {
namespace tl {

using namespace tir;
using arith::IRMutatorWithAnalyzer;

class LegalizeNpuirBF16Mutator : public IRMutatorWithAnalyzer {
 public:
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    LegalizeNpuirBF16Mutator mutator(&analyzer);
    PrimFuncNode* fptr = f.CopyOnWrite();
    fptr->body = mutator.VisitStmt(f->body);
    return f;
  }

 private:
  using IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  struct PreparedRegion {
    PrimExpr region;
    Array<Stmt> prefix;
  };

  bool IsBF16(const Buffer& buffer) const {
    return buffer.defined() && buffer->dtype == DataType::BFloat(16);
  }

  bool IsNpuirAdd(const CallNode* call) const {
    return call != nullptr && call->op.same_as(Op::Get("tl.npuir_add"));
  }

  Buffer CreateTempBuffer(const Array<Range>& region, DataType dtype, const String& scope,
                          const std::string& prefix) {
    Array<PrimExpr> shape;
    for (const Range& r : region) {
      shape.push_back(r->extent);
    }

    std::string name = prefix + "_" + std::to_string(temp_buffer_id_++);
    Buffer buf(Var(name, PointerType(PrimType(dtype), scope)), dtype, shape, {}, PrimExpr(0), name, 0,
               0, kDefault);
    ICHECK(!block_temp_buffers_.empty())
        << "BF16 legalization expects to create temporaries inside a Block.";
    block_temp_buffers_.back().push_back(buf);
    return buf;
  }

  PrimExpr CreateRegion(const Buffer& buffer, const Array<Range>& region, int access_mask) const {
    Array<PrimExpr> indices;
    Array<PrimExpr> args;

    for (const Range& r : region) {
      indices.push_back(make_zero(r->min.dtype()));
    }

    args.push_back(BufferLoad(buffer, indices));
    args.push_back(make_const(DataType::Int(32), access_mask));
    for (const Range& r : region) {
      args.push_back(r->extent);
    }
    return Call(DataType::Handle(), Op::Get("tl.region"), args);
  }

  Stmt CreateCastStmt(const PrimExpr& src_region, const PrimExpr& dst_region) const {
    Array<PrimExpr> args{src_region, dst_region, StringImm("rint")};
    return Evaluate(Call(DataType::Void(), Op::Get("tl.npuir_cast"), args));
  }

  PreparedRegion PrepareSourceRegion(const RegionOp& src, const String& scope) {
    PreparedRegion prepared{CreateRegion(src.GetBuffer(), src.GetRanges(), /*access_mask=*/1), {}};
    if (!IsBF16(src.GetBuffer())) {
      return prepared;
    }

    Buffer temp = CreateTempBuffer(src.GetRanges(), DataType::Float(32), scope, "bf16_src_f32");
    PrimExpr temp_dst_region = CreateRegion(temp, src.GetRanges(), /*access_mask=*/2);
    prepared.prefix.push_back(CreateCastStmt(prepared.region, temp_dst_region));
    prepared.region = CreateRegion(temp, src.GetRanges(), /*access_mask=*/1);
    return prepared;
  }

  PreparedRegion PrepareDestinationRegion(const RegionOp& dst) {
    PreparedRegion prepared{CreateRegion(dst.GetBuffer(), dst.GetRanges(), /*access_mask=*/2), {}};
    if (!IsBF16(dst.GetBuffer())) {
      return prepared;
    }

    Buffer temp = CreateTempBuffer(dst.GetRanges(), DataType::Float(32), dst.GetBuffer().scope(),
                                   "bf16_dst_f32");
    PrimExpr temp_src_region = CreateRegion(temp, dst.GetRanges(), /*access_mask=*/1);
    prepared.prefix.push_back(CreateCastStmt(temp_src_region, prepared.region));
    prepared.region = CreateRegion(temp, dst.GetRanges(), /*access_mask=*/2);
    return prepared;
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    block_temp_buffers_.push_back({});

    Block new_block = Downcast<Block>(IRMutatorWithAnalyzer::VisitStmt_(op));

    Array<Buffer> new_allocs = new_block->alloc_buffers;
    for (const Buffer& buf : block_temp_buffers_.back()) {
      new_allocs.push_back(buf);
    }
    block_temp_buffers_.pop_back();

    if (new_allocs.same_as(new_block->alloc_buffers)) {
      return new_block;
    }

    new_block.CopyOnWrite()->alloc_buffers = std::move(new_allocs);
    return new_block;
  }

  Stmt VisitStmt_(const EvaluateNode* op) final {
    PrimExpr new_value = this->VisitExpr(op->value);
    const CallNode* call = new_value.as<CallNode>();
    if (!IsNpuirAdd(call) || block_temp_buffers_.empty()) {
      if (new_value.same_as(op->value)) {
        return GetRef<Stmt>(op);
      }
      return Evaluate(new_value);
    }

    const CallNode* src0_call = call->args[0].as<CallNode>();
    const CallNode* src1_call = call->args[1].as<CallNode>();
    const CallNode* dst_call = call->args[2].as<CallNode>();
    if (src0_call == nullptr || src1_call == nullptr || dst_call == nullptr ||
        !src0_call->op.same_as(Op::Get("tl.region")) || !src1_call->op.same_as(Op::Get("tl.region")) ||
        !dst_call->op.same_as(Op::Get("tl.region"))) {
      return Evaluate(new_value);
    }

    RegionOp src0_region(src0_call->args, BufferMap{});
    RegionOp src1_region(src1_call->args, BufferMap{});
    RegionOp dst_region(dst_call->args, BufferMap{});
    if (!IsBF16(src0_region.GetBuffer()) && !IsBF16(src1_region.GetBuffer()) &&
        !IsBF16(dst_region.GetBuffer())) {
      return Evaluate(new_value);
    }

    String compute_scope = dst_region.GetBuffer().scope();
    PreparedRegion src0 = PrepareSourceRegion(src0_region, compute_scope);
    PreparedRegion src1 = PrepareSourceRegion(src1_region, compute_scope);
    PreparedRegion dst = PrepareDestinationRegion(dst_region);

    Array<Stmt> seq;
    for (const Stmt& stmt : src0.prefix) {
      seq.push_back(stmt);
    }
    for (const Stmt& stmt : src1.prefix) {
      seq.push_back(stmt);
    }
    Array<PrimExpr> add_args{src0.region, src1.region, dst.region};
    seq.push_back(Evaluate(Call(DataType::Void(), Op::Get("tl.npuir_add"), add_args)));
    for (const Stmt& stmt : dst.prefix) {
      seq.push_back(stmt);
    }
    return SeqStmt::Flatten(seq);
  }

  int temp_buffer_id_{0};
  std::vector<std::vector<Buffer>> block_temp_buffers_;
};

namespace transform {

tvm::transform::Pass LegalizeNpuirBF16() {
  auto pass_func = [=](PrimFunc f, IRModule m, tvm::transform::PassContext ctx) {
    return LegalizeNpuirBF16Mutator::Substitute(std::move(f));
  };
  return tir::transform::CreatePrimFuncPass(pass_func, 0, "tl.LegalizeNpuirBF16", {});
}

TVM_REGISTER_GLOBAL("tl.transform.LegalizeNpuirBF16").set_body_typed(LegalizeNpuirBF16);

}  // namespace transform
}  // namespace tl
}  // namespace tvm
