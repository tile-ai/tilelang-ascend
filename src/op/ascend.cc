// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/op/ascend.cc
 *
 * Define ascend-related operators.
 */

#include "ascend.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "builtin.h"
#include "tvm/ir/expr.h"

namespace tvm {
namespace tl {

using namespace tir;

AscendCopy::AscendCopy(Array<PrimExpr> args, BufferMap vmap) : args_(args) {
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
}

Stmt AscendCopy::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  auto get_dtype = [](const Buffer &buf) -> std::string {
    auto dtype = buf->dtype;
    if (dtype.is_float16()) {
      return "half";
    } else if (dtype.is_float() && dtype.bits() == 32) {
      return "float";
    }
    LOG(FATAL) << "Unsupported data type: " << dtype;
    return "";
  };

  std::stringstream ss;
  ss << "tl::ascend::";
  bool flag = false;
  bool print_dst_layout = false;
  bool print_src_layout = false;
  bool print_gm_layout = false;
  if (src.scope() == "global" && dst.scope() == "shared.dyn") {
    ss << "copy_gm_to_l1";
    print_dst_layout = true;
    print_gm_layout = true;
  } else if (src.scope() == "shared.dyn" && dst.scope() == "wmma.matrix_a") {
    ss << "copy_l1_to_l0a";
    print_src_layout = true;
  } else if (src.scope() == "shared.dyn" && dst.scope() == "wmma.matrix_b") {
    ss << "copy_l1_to_l0b";
    print_src_layout = true;
  } else if (src.scope() == "wmma.accumulator" && dst.scope() == "global") {
    ss << "copy_l0c_to_gm";
    flag = true;
    print_gm_layout = true;
  } else if (src.scope() == "global" && dst.scope() == "shared") {
    ss << "copy_gm_to_ub";
  } else if (src.scope() == "shared" && dst.scope() == "global") {
    ss << "copy_ub_to_gm";
  } else {
    LOG(FATAL) << "Unsupported scope: src = " << src.scope()
               << ", dst = " << dst.scope();
  }

  ss << "<" << get_dtype(src) << ", ";

  if (flag) {
    ss << get_dtype(dst) << ", ";
  }
  if (print_gm_layout) {
    if (T.layout_map.count(src))
      ss << T.layout_map[src]->AscendLayoutStr() << ", ";
    else
      ss << "layout::RowMajor, ";
  }

  if (print_src_layout) {
    ICHECK(T.layout_map.count(src))
        << "Layout map does not contain source buffer: " << src->name;
    ss << T.layout_map[src]->AscendLayoutStr() << ", ";
  } else if (print_dst_layout) {
    ICHECK(T.layout_map.count(dst))
        << "Layout map does not contain destination buffer: " << dst->name;
    ss << T.layout_map[dst]->AscendLayoutStr() << ", ";
  }
  int src_ndim = src->shape.size(), dst_ndim = dst->shape.size();
  ss << src->shape[src_ndim - 2] << ", " << src->shape[src_ndim - 1] << ", "
     << dst->shape[dst_ndim - 2] << ", " << dst->shape[dst_ndim - 1] << ">";

  Array<PrimExpr> src_indices, dst_indices;

  for (size_t i = 0; i < src_range.size(); i++) {
    src_indices.push_back(src_range[i]->min);
  }

  for (size_t i = 0; i < dst_range.size(); i++) {
    dst_indices.push_back(dst_range[i]->min);
  }

  auto src_new_indices = T.layout_map.count(src)
                             ? T.layout_map[src]->Forward(src_indices)
                             : src_indices;
  auto dst_new_indices = T.layout_map.count(dst)
                             ? T.layout_map[dst]->Forward(dst_indices)
                             : dst_indices;

  auto src_new_buffer = T.buffer_remap.count(src) ? T.buffer_remap[src] : src;
  auto dst_new_buffer = T.buffer_remap.count(dst) ? T.buffer_remap[dst] : dst;
  auto src_ptr = src_new_buffer.access_ptr(
      1, DataType::Handle(), 1,
      src_new_buffer.OffsetOf(src_new_indices).back());
  auto dst_ptr = dst_new_buffer.access_ptr(
      2, DataType::Handle(), 1,
      dst_new_buffer.OffsetOf(dst_new_indices).back());
  Array<PrimExpr> new_args;
  new_args.push_back(StringImm(ss.str()));

  new_args.push_back(src_ptr);
  new_args.push_back(dst_ptr);

  auto new_call = Call(DataType::Handle(), builtin::call_extern(), new_args);
  return Evaluate(new_call);
}

LayoutMap AscendCopy::InferLayout(const LayoutInferArgs &T, InferLevel level) {
  LayoutMap results;
  // TODO: add logic to infer layout for AscendCopy
  return results;
}

#define NPUIR_BINARY_OP_CTOR(OPNAME, opname)                                   \
  Npuir##OPNAME::Npuir##OPNAME(Array<PrimExpr> args, BufferMap vmap) {         \
    Array<Range> rgs[3];                                                       \
    Buffer bf[3];                                                              \
    for (int i = 0; i < 3; i++) {                                              \
      auto expr = args[i];                                                     \
      auto call = expr.as<CallNode>();                                         \
      ICHECK(call);                                                            \
      auto region = RegionOp(call->args, vmap);                                \
      rgs[i] = region.GetRanges();                                             \
      bf[i] = region.GetBuffer();                                              \
    }                                                                          \
    std::tie(this->src0, this->src1, this->dst) =                              \
        std::tie(bf[0], bf[1], bf[2]);                                         \
    std::tie(this->src0_range, this->src1_range, this->dst_range) =            \
        std::tie(rgs[0], rgs[1], rgs[2]);                                      \
  }                                                                            \
  TIR_REGISTER_TL_OP(Npuir##OPNAME, npuir_##opname)                            \
      .set_num_inputs(3)                                                       \
      .set_attr<TCallEffectKind>("TCallEffectKind",                            \
                                 Integer(CallEffectKind::kOpaque));

NPUIR_BINARY_OP_CTOR(Add, add)
NPUIR_BINARY_OP_CTOR(Sub, sub)
NPUIR_BINARY_OP_CTOR(Mul, mul)
NPUIR_BINARY_OP_CTOR(Div, div)
NPUIR_BINARY_OP_CTOR(Max, max)
NPUIR_BINARY_OP_CTOR(Min, min)

#define NPUIR_UNARY_OP_CTOR(OPNAME, opname)                                    \
  Npuir##OPNAME::Npuir##OPNAME(Array<PrimExpr> args, BufferMap vmap) {         \
    Array<Range> rgs[2];                                                       \
    Buffer bf[2];                                                              \
    for (int i = 0; i < 2; i++) {                                              \
      auto expr = args[i];                                                     \
      auto call = expr.as<CallNode>();                                         \
      ICHECK(call);                                                            \
      auto region = RegionOp(call->args, vmap);                                \
      rgs[i] = region.GetRanges();                                             \
      bf[i] = region.GetBuffer();                                              \
    }                                                                          \
    std::tie(this->src, this->dst) = std::tie(bf[0], bf[1]);                   \
    std::tie(this->src_range, this->dst_range) = std::tie(rgs[0], rgs[1]);     \
    op_name = "hivm.hir.v" #opname;                                            \
  }                                                                            \
  TIR_REGISTER_TL_OP(Npuir##OPNAME, npuir_##opname)                            \
      .set_num_inputs(2)                                                       \
      .set_attr<TCallEffectKind>("TCallEffectKind",                            \
                                 Integer(CallEffectKind::kOpaque));

NPUIR_UNARY_OP_CTOR(Exp, exp)
NPUIR_UNARY_OP_CTOR(Ln, ln)
NPUIR_UNARY_OP_CTOR(Relu, relu)

NpuirBrc::NpuirBrc(Array<PrimExpr> args, BufferMap vmap) {
  in = args[0], out = args[1];
  Array<Range> rgs[2];
  Buffer bf[2];
  for (int i = 0; i < 2; i++) {
    auto expr = args[i];
    auto call = expr.as<CallNode>();
    if (!call && i == 0) {
      continue;
    } else {
      ICHECK(call);
    }
    auto region = RegionOp(call->args, vmap);
    rgs[i] = region.GetRanges();
    bf[i] = region.GetBuffer();
  }
  std::tie(this->src, this->dst) = std::tie(bf[0], bf[1]);
  std::tie(this->src_range, this->dst_range) = std::tie(rgs[0], rgs[1]);
}
TIR_REGISTER_TL_OP(NpuirBrc, npuir_brc)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

NpuirNd2nz::NpuirNd2nz(Array<PrimExpr> args, BufferMap vmap) {
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

  dst_continuous = args[2].as<Bool>().value();
}

NpuirFixpipe::NpuirFixpipe(Array<PrimExpr> args, BufferMap vmap) {
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

  enable_nz2nd = args[2].as<Bool>().value();
  channel_split = args[3].as<Bool>().value();
  pre_relu_mode = args[4].as<IntImm>().value()->value;
}

NpuirDot::NpuirDot(Array<PrimExpr> args, BufferMap vmap) {
  Array<Range> rgs[3];
  Buffer bf[3];
  for (int i = 0; i < 3; i++) {
    auto expr = args[i];
    auto call = expr.as<CallNode>();
    ICHECK(call);
    auto region = RegionOp(call->args, vmap);
    rgs[i] = region.GetRanges();
    bf[i] = region.GetBuffer();
  }
  std::tie(this->src0, this->src1, this->dst) = std::tie(bf[0], bf[1], bf[2]);
  std::tie(this->src0_range, this->src1_range, this->dst_range) =
      std::tie(rgs[0], rgs[1], rgs[2]);
  initC = args[3];
  a_transpose = args[4].as<Bool>().value();
  b_transpose = args[5].as<Bool>().value();
}

NpuirPipeBarrier::NpuirPipeBarrier(Array<PrimExpr> args, BufferMap vmap) {
  pipe_type = args[0].as<StringImm>().value()->value;
}

NpuirSetFlag::NpuirSetFlag(Array<PrimExpr> args, BufferMap vmap) {
  pipe1 = args[0].as<StringImm>().value()->value;
  pipe2 = args[1].as<StringImm>().value()->value;
  event_id = args[2];
}

NpuirWaitFlag::NpuirWaitFlag(Array<PrimExpr> args, BufferMap vmap) {
  pipe1 = args[0].as<StringImm>().value()->value;
  pipe2 = args[1].as<StringImm>().value()->value;
  event_id = args[2];
}

NpuirSyncBlock::NpuirSyncBlock(Array<PrimExpr> args, BufferMap vmap) {
  mode = static_cast<SyncBlockMode>(args[0].as<IntImm>().value()->value);
  pipe_type = args[1].as<StringImm>().value()->value;
  flag_id = args[2];
}

NpuirSyncBlockSet::NpuirSyncBlockSet(Array<PrimExpr> args, BufferMap vmap) {
  mode = static_cast<SyncBlockMode>(args[0].as<IntImm>().value()->value);
  pipe_type = args[1].as<StringImm>().value()->value;
  flag_id = args[2];
}

NpuirSyncBlockWait::NpuirSyncBlockWait(Array<PrimExpr> args, BufferMap vmap) {
  pipe_type = args[0].as<StringImm>().value()->value;
  flag_id = args[1];
}

NpuirCast::NpuirCast(Array<PrimExpr> args, BufferMap vmap) {
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
  round_mode = args[2].as<StringImmNode>()->value;
}

NpuirReduce::NpuirReduce(Array<PrimExpr> args, BufferMap vmap) {
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

  std::string str_reduce_dims = args[2].as<StringImmNode>()->value;
  std::stringstream ss(str_reduce_dims);
  std::string dim;
  while (std::getline(ss, dim, ',')) {
    reduce_dims.push_back(std::stoi(dim));
  }

  reduce_mode = args[3].as<StringImmNode>()->value;
}

NpuirSelect::NpuirSelect(Array<PrimExpr> args, BufferMap vmap) {
  Array<Range> rgs[4];
  Buffer bf[4];
  for (int i = 0; i < 4; i++) {
    auto expr = args[i];
    auto call = expr.as<CallNode>();
    ICHECK(call);
    auto region = RegionOp(call->args, vmap);
    rgs[i] = region.GetRanges();
    bf[i] = region.GetBuffer();
  }
  std::tie(this->cond, this->src0, this->src1, this->dst) =
      std::tie(bf[0], bf[1], bf[2], bf[3]);
  std::tie(this->cond_range, this->src0_range, this->src1_range,
           this->dst_range) = std::tie(rgs[0], rgs[1], rgs[2], rgs[3]);
}

NpuirCmp::NpuirCmp(Array<PrimExpr> args, BufferMap vmap) {
  Array<Range> rgs[4];
  Buffer bf[4];
  for (int i = 0; i < 3; i++) {
    auto expr = args[i];
    auto call = expr.as<CallNode>();
    ICHECK(call);
    auto region = RegionOp(call->args, vmap);
    rgs[i] = region.GetRanges();
    bf[i] = region.GetBuffer();
  }
  std::tie(this->src0, this->src1, this->dst) = std::tie(bf[0], bf[1], bf[2]);
  std::tie(this->src0_range, this->src1_range, this->dst_range) =
      std::tie(rgs[0], rgs[1], rgs[2]);
  cmp_mod = args[3].as<StringImm>().value()->value;
}

TIR_REGISTER_TL_OP(AscendCopy, ascend_copy)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_REGISTER_TL_OP(NpuirDot, npuir_dot)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(NpuirNd2nz, npuir_load_nd2nz)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(NpuirFixpipe, npuir_store_fixpipe)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(NpuirSyncBlockSet, npuir_sync_block_set)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(NpuirSyncBlockWait, npuir_sync_block_wait)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(NpuirSyncBlock, npuir_sync_block)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(NpuirPipeBarrier, npuir_pipe_barrier)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(NpuirSetFlag, npuir_set_flag)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(NpuirWaitFlag, npuir_wait_flag)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(NpuirCast, npuir_cast)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(NpuirReduce, npuir_reduce)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(NpuirSelect, npuir_select)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(NpuirCmp, npuir_cmp)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

} // namespace tl
} // namespace tvm