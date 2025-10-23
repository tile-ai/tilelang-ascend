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

namespace tvm {
namespace tl {

using namespace tir;

AscendCopy::AscendCopy(Array<PrimExpr> args, BufferMap vmap) : args_(args) {
  Array<Range> rgs[2];
  Buffer bf[2];
  Array<PrimExpr> ets[2];
  for (int i = 0; i < 2; i++) {
    auto expr = args[i];
    auto call = expr.as<CallNode>();
    ICHECK(call);
    auto region = RegionOp(call->args, vmap);
    ets[i] = region.GetExtents();
    rgs[i] = region.GetRanges();
    bf[i] = region.GetBuffer();
  }
  if (args.size() >= 2) {
    enRelu = args[2].as<Bool>().value();
  }
  std::tie(this->src, this->dst) = std::tie(bf[0], bf[1]);
  std::tie(this->src_range, this->dst_range) = std::tie(rgs[0], rgs[1]);
  std::tie(this->src_extents, this->dst_extents) = std::tie(ets[0], ets[1]);
}

Stmt AscendCopy::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  auto get_dtype = [](const Buffer &buf) -> std::string {
    auto dtype = buf->dtype;
    if (dtype.is_float16()) {
      return "half";
    } else if (dtype.is_float() && dtype.bits() == 32) {
      return "float";
    } else if (dtype.is_int()) {
      return "int";
    } else if (dtype.is_uint() && dtype.bits() == 8) {
      return "uint8_t";
    } else if (dtype.is_uint() && dtype.bits() == 16) {
      return "uint16_t";
    } else if (dtype.is_uint() && dtype.bits() == 32) {
      return "uint32_t";
    } else if (dtype.is_bfloat16()) {
      return "bfloat16_t";
    }
    LOG(FATAL) << "Unsupported data type: " << dtype;
    return "";
  };

  auto compute_strideN = [](const Buffer &buf, const Array<PrimExpr> &extents) -> PrimExpr {
    PrimExpr strideN = buf->shape[buf->shape.size() - 1];
    if (extents.size() > 1) {
      for (int i = extents.size() - 2; i >= 0; --i) {
        auto extend = static_cast<int>(extents[i].as<IntImmNode>()->value);
        if (extend != 1) {
          break;
        }
        strideN = strideN * buf->shape[i];
      }
    }
    return strideN;
  };

  auto build_indices = [](const Array<Range> &range) -> Array<PrimExpr> {
    Array<PrimExpr> indices;
    for (size_t i = 0; i < range.size(); i++) {
      indices.push_back(range[i]->min);
    }
    return indices;
  };

  struct CopyConfig {
    bool needs_strideN = false;
    bool l0c2gm = false;
    bool gm2l1 = false;
    bool print_gm_layout = false;
    bool print_src_layout = false;
    bool print_dst_layout = false;
    bool print_ub = false;
  } config;

  std::stringstream ss;
  ss << "tl::ascend::";
  PrimExpr strideN;

  if (src.scope() == "global" && dst.scope() == "shared.dyn") {
    ss << "copy_gm_to_l1";
    config.gm2l1 = true;
  } else if (src.scope() == "shared.dyn" && dst.scope() == "wmma.matrix_a") {
    ss << "copy_l1_to_l0a";
    config.print_src_layout = true;
  } else if (src.scope() == "shared.dyn" && dst.scope() == "wmma.matrix_b") {
    ss << "copy_l1_to_l0b";
    config.print_src_layout = true;
  } else if (src.scope() == "wmma.accumulator" && dst.scope() == "global") {
    ss << "copy_l0c_to_gm";
    config.l0c2gm = true;
    config.print_gm_layout = true;
  } else if (src.scope() == "shared" || dst.scope() == "shared") {
    config.print_ub = true;

    if (src.scope() == "global") {
      strideN = compute_strideN(src, src_extents);
      config.needs_strideN = true;

      ss << "copy_gm_to_ub<";
      ss << get_dtype(src) << ", ";
      ss << dst->shape[dst->shape.size() - 1];
      if (dst->shape.size() > 1) {
        ss << ", " << dst->shape[dst->shape.size() - 2];
      }
      ss << ">";
    } else if (dst.scope() == "global") {
      strideN = compute_strideN(dst, dst_extents);
      config.needs_strideN = true;

      ss << "copy_ub_to_gm<";
      ss << get_dtype(dst) << ", ";
      ss << src->shape[src->shape.size() - 1];
      if (src->shape.size() > 1) {
        ss << ", " << src->shape[src->shape.size() - 2];
      }
      ss << ">";
    } else {
      PrimExpr len = 1;
      for (auto &shape : src->shape) {
        len *= shape;
      }
      ss << "copy_ub_to_ub<" << get_dtype(dst) << ", "
         << get_dtype(src) << ", " << len << ">";
    }
  } else {
    LOG(FATAL) << "Unsupported scope: src = " << src.scope()
               << ", dst = " << dst.scope();
  }

  if (!config.print_ub) {
    ss << "<" << get_dtype(src) << ", ";

    if (config.l0c2gm) {
      ss << get_dtype(dst) << ", ";
    }

    if (config.print_gm_layout) {
      ss << (T.layout_map.count(src) ? T.layout_map[src]->AscendLayoutStr()
                                     : "layout::RowMajor") << ", ";
    }

    if (config.print_src_layout) {
      ICHECK(T.layout_map.count(src))
          << "Layout map does not contain source buffer: " << src->name;
      ss << T.layout_map[src]->AscendLayoutStr() << ", ";
    } else if (config.print_dst_layout) {
      ICHECK(T.layout_map.count(dst))
          << "Layout map does not contain destination buffer: " << dst->name;
      ss << T.layout_map[dst]->AscendLayoutStr() << ", ";
    }

    int src_ndim = src->shape.size(), dst_ndim = dst->shape.size();

    if (dst.scope() == "global") {
      ss << src->shape[src_ndim - 2] << ", " << src->shape[src_ndim - 1] << ">";
    } else if (src.scope() == "global") {
      ss << dst->shape[dst_ndim - 2] << ", " << dst->shape[dst_ndim - 1] << ">";
    } else {
      ss << src->shape[src_ndim - 2] << ", " << src->shape[src_ndim - 1] << ", "
         << dst->shape[dst_ndim - 2] << ", " << dst->shape[dst_ndim - 1] << ">";
    }
  }

  auto src_indices = build_indices(src_range);
  auto dst_indices = build_indices(dst_range);

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

  if (config.needs_strideN) {
    new_args.push_back(strideN);
  }

  if (config.l0c2gm) {
    new_args.push_back(compute_strideN(dst, dst_extents));
    new_args.push_back(enRelu);
  }

  if (config.gm2l1) {
    new_args.push_back(compute_strideN(src, src_extents));
  }

  auto new_call = Call(DataType::Handle(), builtin::call_extern(), new_args);
  return Evaluate(new_call);
}



LayoutMap AscendCopy::InferLayout(const LayoutInferArgs &T, InferLevel level) {
  LayoutMap results;
  // TODO: add logic to infer layout for AscendCopy
  return results;
}

TIR_REGISTER_TL_OP(AscendCopy, ascend_copy)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
} // namespace tl
} // namespace tvm