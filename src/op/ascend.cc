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
NPUIR_BINARY_OP_CTOR(Or, or)
NPUIR_BINARY_OP_CTOR(And, and)
NPUIR_BINARY_OP_CTOR(Xor, xor)
NPUIR_BINARY_OP_CTOR(Pow, pow)
NPUIR_BINARY_OP_CTOR(Shl, shl)

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
NPUIR_UNARY_OP_CTOR(Sqrt, sqrt)
NPUIR_UNARY_OP_CTOR(Rsqrt, rsqrt)
NPUIR_UNARY_OP_CTOR(Abs, abs)
NPUIR_UNARY_OP_CTOR(Rec, rec)
NPUIR_UNARY_OP_CTOR(Not, not )

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

NpuirNz2nd::NpuirNz2nd(Array<PrimExpr> args, BufferMap vmap) {
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
  int src_rank = static_cast<int>(src->shape.size());
  while (std::getline(ss, dim, ',')) {
    int axis = std::stoi(dim);
    if (axis < 0) {
      axis += src_rank;
    }
    reduce_dims.push_back(axis);
  }

  reduce_mode = args[3].as<StringImmNode>()->value;
}

NpuirCumsum::NpuirCumsum(Array<PrimExpr> args, BufferMap vmap) {
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

  std::string str_cum_dims = args[2].as<StringImmNode>()->value;
  std::stringstream ss(str_cum_dims);
  std::string dim;
  while (std::getline(ss, dim, ',')) {
    cum_dims.push_back(std::stoi(dim));
  }
  reverse = args[3].as<Bool>().value();
}

NpuirAtomicAdd::NpuirAtomicAdd(Array<PrimExpr> args, BufferMap vmap) {
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

NpuirShr::NpuirShr(Array<PrimExpr> args, BufferMap vmap) {
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
  round = args[3].as<Bool>().value();
}

NpuirDevicePrintVar::NpuirDevicePrintVar(Array<PrimExpr> args, BufferMap vmap) {
  src = args[0];
  prefix = args[1].as<StringImmNode>()->value;
  hex = args[2].as<Bool>().value();
}

NpuirDevicePrintBuf::NpuirDevicePrintBuf(Array<PrimExpr> args, BufferMap vmap) {
  Array<Range> rg;
  Buffer bf;
  auto expr = args[0];
  auto call = expr.as<CallNode>();
  ICHECK(call);
  auto region = RegionOp(call->args, vmap);
  rg = region.GetRanges();
  bf = region.GetBuffer();
  this->src = bf;
  this->src_range = rg;

  prefix = args[1].as<StringImmNode>()->value;
  hex = args[2].as<Bool>().value();
}

#define NPUIR_GEN_BUF(arg)                                                     \
  Array<Range> rg;                                                             \
  Buffer bf;                                                                   \
  auto expr = arg;                                                             \
  auto call = expr.as<CallNode>();                                             \
  ICHECK(call);                                                                \
  auto region = RegionOp(call->args, vmap);                                    \
  rg = region.GetRanges();                                                     \
  bf = region.GetBuffer();

#define NPUIR_SRC_DST_BUF                                                      \
  Array<Range> rgs[2];                                                         \
  Buffer bf[2];                                                                \
  for (int i = 0; i < 2; i++) {                                                \
    auto expr = args[i];                                                       \
    auto call = expr.as<CallNode>();                                           \
    ICHECK(call);                                                              \
    auto region = RegionOp(call->args, vmap);                                  \
    rgs[i] = region.GetRanges();                                               \
    bf[i] = region.GetBuffer();                                                \
  }                                                                            \
  std::tie(this->src, this->dst) = std::tie(bf[0], bf[1]);                     \
  std::tie(this->src_range, this->dst_range) = std::tie(rgs[0], rgs[1]);

#define NPUIR_LIST_PARAM(list_param, arg_pos)                                  \
  std::string str_##list_param = args[arg_pos].as<StringImmNode>()->value;     \
  std::stringstream ss_##list_param(str_##list_param);                                      \
  std::string num_##list_param;                                                             \
  while (std::getline(ss_##list_param, num_##list_param, ',')) {                                         \
    list_param.push_back(std::stoi(num_##list_param));                                      \
  }

NpuirGather::NpuirGather(Array<PrimExpr> args, BufferMap vmap) {
  NPUIR_SRC_DST_BUF

  Array<Range> range;
  Buffer buffer;
  auto expr = args[2];
  auto call = expr.as<CallNode>();
  ICHECK(call);
  auto region = RegionOp(call->args, vmap);
  range = region.GetRanges();
  buffer = region.GetBuffer();
  this->indices = buffer;
  this->indices_range = range;
}

NpuirArange::NpuirArange(Array<PrimExpr> args, BufferMap vmap) {
  NPUIR_GEN_BUF(args[0])
  this->dst = bf;
  this->dst_range = rg;

  NPUIR_LIST_PARAM(strides, 1)

  this->offset = args[2].as<IntImm>().value()->value;
}

NpuirConcat::NpuirConcat(Array<PrimExpr> args, BufferMap vmap) {
  this->dim = args[0].as<IntImm>().value()->value;

  NPUIR_GEN_BUF(args[1])
  this->dst = bf;
  this->dst_range = rg;

  size_t n_srcs = args.size() - 2;
  for (size_t i = 0; i < n_srcs; i++) {
    NPUIR_GEN_BUF(args[2 + i])
    this->srcs.push_back(bf);
    this->srcs_range.push_back(rg);
  }
}

NpuirPad::NpuirPad(Array<PrimExpr> args, BufferMap vmap) {
  NPUIR_SRC_DST_BUF

  this->pad_value = args[2];

  this->pad_dim = args[3].as<IntImm>().value()->value;

  NPUIR_LIST_PARAM(s_low, 4)
  NPUIR_LIST_PARAM(s_high, 5)

  int num_dynamic_low = args[6].as<IntImm>().value()->value;
  int num_dynamic = args.size() - 7;
  for (int i = 0; i < num_dynamic; i++) {
    if (i < num_dynamic_low) {
      this->low.push_back(args[7 + i]);
    } else {
      this->high.push_back(args[7 + i]);
    }
  }
}

NpuirFlip::NpuirFlip(Array<PrimExpr> args, BufferMap vmap){NPUIR_SRC_DST_BUF}

NpuirBitcast::NpuirBitcast(Array<PrimExpr> args, BufferMap vmap) {
  NPUIR_GEN_BUF(args[0])
  this->src = bf;
  this->src_range = rg;
  this->dtype = args[1].as<StringImmNode>()->value;
  ;
}

NpuirTranspose::NpuirTranspose(Array<PrimExpr> args, BufferMap vmap){
    NPUIR_SRC_DST_BUF NPUIR_LIST_PARAM(permutation, 2)}

NpuirInterleave::NpuirInterleave(Array<PrimExpr> args, BufferMap vmap) {
  this->channel_nums = args[0].as<IntImm>().value()->value;

  NPUIR_GEN_BUF(args[1])
  this->dst = bf;
  this->dst_range = rg;

  size_t n_srcs = args.size() - 2;
  for (size_t i = 0; i < n_srcs; i++) {
    NPUIR_GEN_BUF(args[2 + i])
    this->srcs.push_back(bf);
    this->srcs_range.push_back(rg);
  }
}

NpuirDeinterleave::NpuirDeinterleave(Array<PrimExpr> args, BufferMap vmap) {
  this->channel_nums = args[0].as<IntImm>().value()->value;
  this->index_mode = args[1].as<StringImmNode>()->value;

  NPUIR_GEN_BUF(args[2])
  this->src = bf;
  this->src_range = rg;

  size_t n_dsts = args.size() - 3;
  for (size_t i = 0; i < n_dsts; i++) {
    NPUIR_GEN_BUF(args[3 + i])
    this->dsts.push_back(bf);
    this->dsts_range.push_back(rg);
  }
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

TIR_REGISTER_TL_OP(NpuirNz2nd, npuir_store_nz2nd)
    .set_num_inputs(2)
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

TIR_REGISTER_TL_OP(NpuirCumsum, npuir_cumsum)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(NpuirAtomicAdd, npuir_atomic_add)
    .set_num_inputs(2)
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

TIR_REGISTER_TL_OP(NpuirShr, npuir_shr)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(NpuirDevicePrintVar, npuir_debug_print_var)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(NpuirDevicePrintBuf, npuir_debug_print_buffer_value)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(NpuirGather, npuir_gather)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(NpuirTranspose, npuir_transpose)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(NpuirInterleave, npuir_interleave)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(NpuirDeinterleave, npuir_deinterleave)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(NpuirArange, npuir_arange)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(NpuirConcat, npuir_concat)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(NpuirPad, npuir_pad)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(NpuirFlip, npuir_flip)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(NpuirBitcast, npuir_bitcast)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

} // namespace tl
} // namespace tvm



