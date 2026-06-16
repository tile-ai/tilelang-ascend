// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/op/ascend.cc
 *
 * Define ascend-related operators.
 */

#include "ascend.h"

#include <sstream>
#include <tuple>
#include <vector>

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "builtin.h"

namespace tvm {
namespace tl {

using namespace tir;

#define TIR_DEFINE_TL_BUILTIN(OpName)                                          \
  const Op &OpName() {                                                         \
    static const Op &op = Op::Get("tl." #OpName);                              \
    return op;                                                                 \
  }                                                                            \
  TVM_REGISTER_OP("tl." #OpName)                                               \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", #OpName)

AscendCopy::AscendCopy(Array<PrimExpr> args, BufferMap vmap) : args_(args) {
  Array<Range> rgs[2];
  Buffer bf[2];
  Array<PrimExpr> ets[2];
  transposeL1 = 0;
  for (int i = 0; i < 2; i++) {
    auto expr = args[i];
    auto call = expr.as<CallNode>();
    ICHECK(call);
    auto region = RegionOp(call->args, vmap);
    ets[i] = region.GetExtents();
    rgs[i] = region.GetRanges();
    bf[i] = region.GetBuffer();
  }
  if (args.size() >= 3) {
    enRelu = args[2].as<Bool>().value();
  }
  if (args.size() >= 4) {
    transposeL1 = args[3].as<Bool>().value();
  }
  if (args.size() >= 5) {
    padValue = args[4];
  } else {
    padValue = Integer(0);
  }
  std::tie(this->src, this->dst) = std::tie(bf[0], bf[1]);
  std::tie(this->src_range, this->dst_range) = std::tie(rgs[0], rgs[1]);
  std::tie(this->src_extents, this->dst_extents) = std::tie(ets[0], ets[1]);
}

AscendAtomicAdd::AscendAtomicAdd(Array<PrimExpr> args, BufferMap vmap)
    : args_(args) {
  ICHECK_EQ(args.size(), 2U) << "tl.ascend_atomic_add expects dst and src";

  auto dst_call = args[0].as<CallNode>();
  auto src_call = args[1].as<CallNode>();
  ICHECK(dst_call) << "tl.ascend_atomic_add dst must be a tl.region call";
  ICHECK(src_call) << "tl.ascend_atomic_add src must be a tl.region call";

  auto dst_region = RegionOp(dst_call->args, vmap);
  auto src_region = RegionOp(src_call->args, vmap);
  ICHECK(dst_region.GetAccessMask() & 2)
      << "tl.ascend_atomic_add dst region must be writable";
  ICHECK(src_region.GetAccessMask() & 1)
      << "tl.ascend_atomic_add src region must be readable";

  this->dst = dst_region.GetBuffer();
  this->src = src_region.GetBuffer();
  this->dst_range = dst_region.GetRanges();
  this->src_range = src_region.GetRanges();
  this->dst_extents = dst_region.GetExtents();
  this->src_extents = src_region.GetExtents();
}

Stmt AscendCopy::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  auto get_dtype = [](const Buffer &buf) -> std::string {
    auto dtype = buf->dtype;
    if (dtype.is_float16()) {
      return "half";
    } else if (dtype.is_float() && dtype.bits() == 32) {
      return "float";
    } else if (dtype.is_int() && dtype.bits() == 4) {
      return "AscendC::int4b_t";
    } else if (dtype.is_int() && dtype.bits() == 8) {
      return "int8_t";
    } else if (dtype.is_int() && dtype.bits() == 16) {
      return "int16_t";
    } else if (dtype.is_int() && dtype.bits() == 32) {
      return "int";
    } else if (dtype.is_int() && dtype.bits() == 64) {
      return "int64_t";
    } else if (dtype.is_uint() && dtype.bits() == 8) {
      return "uint8_t";
    } else if (dtype.is_uint() && dtype.bits() == 16) {
      return "uint16_t";
    } else if (dtype.is_uint() && dtype.bits() == 32) {
      return "uint32_t";
    } else if (dtype.is_uint() && dtype.bits() == 64) {
      return "uint64_t";
    } else if (dtype.is_bfloat16()) {
      return "bfloat16_t";
    }
    LOG(FATAL) << "Unsupported data type: " << dtype;
    return "";
  };

  auto compute_strideN = [](const Buffer &buf,
                            const Array<PrimExpr> &extents) -> PrimExpr {
    PrimExpr strideN = buf->shape[buf->shape.size() - 1];
    if (extents.size() > 1) {
      for (int i = extents.size() - 2; i >= 0; --i) {
        auto *extent = extents[i].as<IntImmNode>();
        if (!extent || extent->value != 1) {
          break;
        }
        strideN = strideN * buf->shape[i];
      }
    }
    return strideN;
  };

  auto compute_blocklen = [](const Buffer &buf,
                             const Array<PrimExpr> &extents) -> PrimExpr {
    PrimExpr res = buf->shape[buf->shape.size() - 2];
    auto ext_size = extents.size();
    if (ext_size > 1 && extents[ext_size - 2]->IsInstance<IntImmNode>() &&
        res->IsInstance<IntImmNode>()) {
      auto extent =
          static_cast<int>(extents[ext_size - 2].as<IntImmNode>()->value);
      auto shape = static_cast<int>(res.as<IntImmNode>()->value);
      res = shape < extent ? res : extents[ext_size - 2];
    }

    return res;
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
    bool l12l0 = false;
    bool print_gm_layout = false;
    bool print_src_layout = false;
    bool print_dst_layout = false;
    bool print_ub = false;
    bool l0_dst_split = false;
    bool virtual_channel = false;
    bool gm2ub = false;
    bool ub2gm = false;
    bool ub2ub = false;
  } config;

  std::stringstream ss;
  ss << "tl::ascend::";
  PrimExpr strideN;

  if (src.scope() == "global" && dst.scope() == "shared.dyn") {
    ss << "copy_gm_to_l1";
    config.gm2l1 = true;
  } else if (src.scope() == "shared.dyn" && dst.scope() == "wmma.matrix_a") {
    ss << "copy_l1_to_l0a";
    // config.print_src_layout = true;
    config.l12l0 = true;
    config.l0_dst_split = true;
  } else if (src.scope() == "shared.dyn" && dst.scope() == "wmma.matrix_b") {
    ss << "copy_l1_to_l0b";
    // config.print_src_layout = true;
    config.l12l0 = true;
    config.l0_dst_split = true;
  } else if (src.scope() == "wmma.accumulator" && dst.scope() == "global") {
    ss << "copy_l0c_to_gm";
    config.l0c2gm = true;
    config.print_gm_layout = true;
  } else if (src.scope() == "shared" || dst.scope() == "shared") {
    config.print_ub = true;

    if (src.scope() == "global") {
      config.gm2ub = true;
      strideN = compute_strideN(src, src_extents);
      config.needs_strideN = true;

      ss << "copy_gm_to_ub<";
      ss << get_dtype(src) << ", ";
      ss << dst_extents[dst->shape.size() - 1];
      // ss << dst->shape[dst->shape.size() - 1];
      if (dst->shape.size() > 1) {
        ss << ", " << compute_blocklen(dst, dst_extents);
      }
      ss << ">";
    } else if (dst.scope() == "global") {
      config.ub2gm = true;
      strideN = compute_strideN(dst, dst_extents);
      config.needs_strideN = true;

      ss << "copy_ub_to_gm<";
      ss << get_dtype(dst) << ", ";
      ss << src_extents[src->shape.size() - 1];
      // ss << src->shape[src->shape.size() - 1];
      if (src->shape.size() > 1) {
        ss << ", " << compute_blocklen(src, src_extents);
      }
      ss << ">";
    } else if (dst.scope() == "shared.dyn") {
      config.virtual_channel = true;
      ss << "copy_ub_to_l1<"; // real channel is "ub -> gm -> l1"
      ss << get_dtype(dst) << ", ";
      ss << src->shape[src->shape.size() - 1];
      if (src->shape.size() > 1) {
        ss << ", " << src->shape[src->shape.size() - 2];
      }
      ss << ">";
    } else if (src.scope() == "wmma.accumulator") {
      config.virtual_channel = true;
      ss << "copy_l0c_to_ub<";
      ss << get_dtype(src) << ", " << get_dtype(dst) << ", ";
      ss << "layout::RowMajor, "; // real channel is "ub -> gm -> l1", so gm is
                                  // always row major
      ss << src->shape[src->shape.size() - 2] << ", "
         << src->shape[src->shape.size() - 1];
      ss << ", " << enRelu << ">";
    } else {
      PrimExpr len = 1;
      for (auto &shape : dst_extents) {
        len *= shape;
      }
      ss << "copy_ub_to_ub<" << get_dtype(dst) << ", " << get_dtype(src) << ", "
         << len << ">";
      config.ub2ub = true;
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
                                     : "layout::RowMajor")
         << ", ";
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
      ss << src->shape[src_ndim - 2] << ", " << src->shape[src_ndim - 1];
      if (config.l0c2gm) {
        ss << ", " << enRelu;
      }
      ss << ">";
    } else if (src.scope() == "global") {
      ss << dst->shape[dst_ndim - 2] << ", " << dst->shape[dst_ndim - 1] << ">";
    } else {
      ss << src->shape[src_ndim - 2] << ", " << src->shape[src_ndim - 1];
      if (config.l12l0) {
        transposeL1 == 0 ? ss << ", false" << ">" : ss << ", true" << ">";
      }
      // ss << src->shape[src_ndim - 2] << ", " << src->shape[src_ndim - 1] <<
      // ", "
      //    << dst->shape[dst_ndim - 2] << ", " << dst->shape[dst_ndim - 1] <<
      //    ">";
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

  PrimExpr src_len = 1;
  for (auto &shape : src_extents) {
    src_len *= shape;
  }

  PrimExpr dst_len = 1;
  for (auto &shape : dst_extents) {
    dst_len *= shape;
  }

  auto src_ptr = src_new_buffer.access_ptr(
      1, src_new_buffer->dtype, 1,
      src_new_buffer.OffsetOf(src_new_indices).back(), src_len);
  auto dst_ptr = dst_new_buffer.access_ptr(
      2, dst_new_buffer->dtype, 1,
      dst_new_buffer.OffsetOf(dst_new_indices).back(), dst_len);

  auto compute_valid_extent = [](PrimExpr min_val, PrimExpr extent,
                                 PrimExpr shape) -> PrimExpr {
    PrimExpr remaining = shape - min_val;
    if (remaining.dtype().lanes() > 1) {
      return extent;
    }
    return Select(remaining >= extent, extent,
                  Select(remaining > 0, remaining, 0));
  };

  auto find_active_dim_indices =
      [](const Array<PrimExpr> &extents) -> std::vector<int> {
    std::vector<int> active_indices;
    int size = static_cast<int>(extents.size());

    // Traverse from 0 to size-1, find dimensions where extent != 1
    for (int i = 0; i < size; ++i) {
      if (auto *int_imm = extents[i].as<IntImmNode>()) {
        if (int_imm->value != 1) {
          active_indices.push_back(i);
        }
      } else {
        active_indices.push_back(i);
      }
    }

    // The last dimension must always be included in the result
    if (size >= 1 &&
        (active_indices.empty() || active_indices.back() != size - 1)) {
      active_indices.push_back(size - 1);
    }

    // Special: If extents.size() >= 2 and active_indices.size() == 1,
    // insert size-2 at the second-to-last position
    if (size >= 2 && active_indices.size() == 1) {
      active_indices.insert(active_indices.begin(), size - 2);
    }

    return active_indices;
  };

  PrimExpr validRow_src, validCol_src, validRow_dst, validCol_dst;

  // src: compute validRow and validCol using active dimension indices
  std::vector<int> src_active = find_active_dim_indices(src_extents);
  if (src_active.size() >= 2) {
    int row_idx = src_active[src_active.size() - 2];
    int col_idx = src_active.back();
    validRow_src =
        compute_valid_extent(src_range[row_idx]->min,
                             src_range[row_idx]->extent, src->shape[row_idx]);
    validCol_src =
        compute_valid_extent(src_range[col_idx]->min,
                             src_range[col_idx]->extent, src->shape[col_idx]);
  } else if (src_active.size() == 1) {
    int col_idx = src_active[0];
    validRow_src = Integer(1);
    validCol_src =
        compute_valid_extent(src_range[col_idx]->min,
                             src_range[col_idx]->extent, src->shape[col_idx]);
  } else {
    validRow_src = 0;
    validCol_src = 0;
  }

  // dst: compute validRow and validCol using active dimension indices
  std::vector<int> dst_active = find_active_dim_indices(dst_extents);
  if (dst_active.size() >= 2) {
    int row_idx = dst_active[dst_active.size() - 2];
    int col_idx = dst_active.back();
    validRow_dst =
        compute_valid_extent(dst_range[row_idx]->min,
                             dst_range[row_idx]->extent, dst->shape[row_idx]);
    validCol_dst =
        compute_valid_extent(dst_range[col_idx]->min,
                             dst_range[col_idx]->extent, dst->shape[col_idx]);
  } else if (dst_active.size() == 1) {
    int col_idx = dst_active[0];
    validRow_dst = Integer(1);
    validCol_dst =
        compute_valid_extent(dst_range[col_idx]->min,
                             dst_range[col_idx]->extent, dst->shape[col_idx]);
  } else {
    validRow_dst = 0;
    validCol_dst = 0;
  }

  Array<PrimExpr> new_args;
  new_args.push_back(StringImm(ss.str()));
  new_args.push_back(src_ptr);
  new_args.push_back(dst_ptr);

  if (config.needs_strideN) {
    new_args.push_back(strideN);
  }

  if (config.l0_dst_split) {
    int dst_dim = dst->shape.size();
    new_args.push_back(dst->shape[dst_dim - 2]);
    new_args.push_back(dst->shape[dst_dim - 1]);
  }

  if (config.l0c2gm) {
    new_args.push_back(compute_strideN(dst, dst_extents));
    new_args.push_back(validRow_dst);
    new_args.push_back(validCol_dst);
    new_args.push_back(src->shape[src->shape.size() - 2]);
    new_args.push_back(src->shape[src->shape.size() - 1]);
    new_args.push_back(Bool(enRelu)); // Add enable_relu parameter
  }

  if (config.gm2l1) {
    new_args.push_back(compute_strideN(src, src_extents));
    new_args.push_back(validRow_src);
    new_args.push_back(validCol_src);
    new_args.push_back(dst->shape[dst->shape.size() - 2]);
    new_args.push_back(dst->shape[dst->shape.size() - 1]);
  }

  if (config.gm2ub) {
    new_args.push_back(validRow_src);
    new_args.push_back(validCol_src);
    PrimExpr pad_val = padValue;
    if (pad_val->dtype != dst->dtype) {
      pad_val = Cast(dst->dtype, pad_val);
    }
    new_args.push_back(pad_val);
    if (dst->shape.size() > 1) {
      new_args.push_back(dst->shape[dst->shape.size() - 2]);
    }
    new_args.push_back(dst->shape[dst->shape.size() - 1]);
  }

  if (config.ub2gm) {
    new_args.push_back(validRow_dst);
    new_args.push_back(validCol_dst);
    if (src->shape.size() > 1) {
      new_args.push_back(src->shape[src->shape.size() - 2]);
    }
    new_args.push_back(src->shape[src->shape.size() - 1]);
  }

  if (config.virtual_channel) {
    new_args.push_back(
        src->shape[src->shape.size() -
                   1]); // ub/l0c -> gm need realdstN which is equal to srcN in
                        // virtural channel scenario
  }

  if (config.ub2ub) {
    PrimExpr src_tile_rows = src_extents.size() >= 2
                                 ? src_extents[src_extents.size() - 2]
                                 : Integer(1);
    PrimExpr src_tile_cols = src_extents[src_extents.size() - 1];
    PrimExpr src_buf_cols = src->shape[src->shape.size() - 1];
    PrimExpr dst_tile_rows = dst_extents.size() >= 2
                                 ? dst_extents[dst_extents.size() - 2]
                                 : Integer(1);
    PrimExpr dst_tile_cols = dst_extents[dst_extents.size() - 1];
    PrimExpr dst_buf_cols = dst->shape[dst->shape.size() - 1];
    new_args.push_back(src_tile_rows);
    new_args.push_back(src_tile_cols);
    new_args.push_back(src_buf_cols);
    new_args.push_back(dst_tile_rows);
    new_args.push_back(dst_tile_cols);
    new_args.push_back(dst_buf_cols);
  }
  // if (config.l12l0) {
  //   ICHECK(src->shape.size() == dst->shape.size());
  //   bool is_extract = false;
  //   auto dst_shape_size = dst->shape.size();
  //   auto src_shape_size = src->shape.size();
  //   for (int i = dst_shape_size - 1; i >= dst_shape_size - 2; i--) {
  //     if (i == -1) {
  //         break;
  //     }
  //     // std::cout << "dst_shape_size -2 = " << dst_shape_size - 2  << ", i =
  //     " << i << ", dst->shape: " << dst->shape[i]
  //     //   << "src->shape: " << src->shape[i] << "\n";
  //     if (src->shape[i].as<IntImmNode>()->value !=
  //     dst->shape[i].as<IntImmNode>()->value) {
  //         is_extract = true;
  //     }
  //     new_args.push_back(src_indices[i]);
  //   }
  //   new_args.push_back(Bool(is_extract));
  //   auto dst_var = dst_ptr.as<CallNode>()->args[1];
  //   std::cout << "newargs in copy to l1: " << new_args << "\n";
  //   std::cout << "src_indices: " << src_indices << "\n";
  // }

  auto new_call = Call(DataType::Handle(), builtin::call_extern(), new_args);
  return Evaluate(new_call);
}

Stmt AscendAtomicAdd::Lower(const LowerArgs &T,
                            arith::Analyzer *analyzer) const {
  auto get_dtype = [](const Buffer &buf) -> std::string {
    auto dtype = buf->dtype;
    if (dtype.is_float16()) {
      return "half";
    } else if (dtype.is_float() && dtype.bits() == 32) {
      return "float";
    } else if (dtype.is_int() && dtype.bits() == 8) {
      return "int8_t";
    } else if (dtype.is_int() && dtype.bits() == 16) {
      return "int16_t";
    } else if (dtype.is_int() && dtype.bits() == 32) {
      return "int";
    } else if (dtype.is_int() && dtype.bits() == 64) {
      return "int64_t";
    } else if (dtype.is_uint() && dtype.bits() == 8) {
      return "uint8_t";
    } else if (dtype.is_uint() && dtype.bits() == 16) {
      return "uint16_t";
    } else if (dtype.is_uint() && dtype.bits() == 32) {
      return "uint32_t";
    } else if (dtype.is_uint() && dtype.bits() == 64) {
      return "uint64_t";
    } else if (dtype.is_bfloat16()) {
      return "bfloat16_t";
    }
    LOG(FATAL) << "Unsupported data type for tl.ascend_atomic_add: " << dtype;
    return "";
  };

  auto compute_strideN = [](const Buffer &buf,
                            const Array<PrimExpr> &extents) -> PrimExpr {
    PrimExpr strideN = buf->shape[buf->shape.size() - 1];
    if (extents.size() > 1) {
      for (int i = extents.size() - 2; i >= 0; --i) {
        auto *extent = extents[i].as<IntImmNode>();
        if (!extent || extent->value != 1) {
          break;
        }
        strideN = strideN * buf->shape[i];
      }
    }
    return strideN;
  };

  auto compute_blocklen = [](const Buffer &buf,
                             const Array<PrimExpr> &extents) -> PrimExpr {
    PrimExpr res = buf->shape[buf->shape.size() - 2];
    auto ext_size = extents.size();
    if (ext_size > 1 && extents[ext_size - 2]->IsInstance<IntImmNode>() &&
        res->IsInstance<IntImmNode>()) {
      auto extent =
          static_cast<int>(extents[ext_size - 2].as<IntImmNode>()->value);
      auto shape = static_cast<int>(res.as<IntImmNode>()->value);
      res = shape < extent ? res : extents[ext_size - 2];
    }
    return res;
  };

  auto build_indices = [](const Array<Range> &range) -> Array<PrimExpr> {
    Array<PrimExpr> indices;
    for (size_t i = 0; i < range.size(); i++) {
      indices.push_back(range[i]->min);
    }
    return indices;
  };

  auto compute_valid_extent = [](PrimExpr min_val, PrimExpr extent,
                                 PrimExpr shape) -> PrimExpr {
    PrimExpr remaining = shape - min_val;
    if (remaining.dtype().lanes() > 1) {
      return extent;
    }
    return Select(remaining >= extent, extent,
                  Select(remaining > 0, remaining, 0));
  };

  auto find_active_dim_indices =
      [](const Array<PrimExpr> &extents) -> std::vector<int> {
    std::vector<int> active_indices;
    int size = static_cast<int>(extents.size());
    for (int i = 0; i < size; ++i) {
      if (auto *int_imm = extents[i].as<IntImmNode>()) {
        if (int_imm->value != 1) {
          active_indices.push_back(i);
        }
      } else {
        active_indices.push_back(i);
      }
    }
    if (size >= 1 &&
        (active_indices.empty() || active_indices.back() != size - 1)) {
      active_indices.push_back(size - 1);
    }
    if (size >= 2 && active_indices.size() == 1) {
      active_indices.insert(active_indices.begin(), size - 2);
    }
    return active_indices;
  };

  ICHECK(dst.scope() == "global")
      << "tl.ascend_atomic_add V1 requires global dst, got " << dst.scope();
  ICHECK(src.scope() == "shared" || src.scope() == "wmma.accumulator")
      << "tl.ascend_atomic_add V1 requires UB/shared or L0C/wmma.accumulator "
         "src, got "
      << src.scope();
  ICHECK(src->dtype == dst->dtype)
      << "tl.ascend_atomic_add requires src and dst dtype to match, got src "
      << src->dtype << " and dst " << dst->dtype;
  ICHECK_EQ(src_extents.size(), src->shape.size())
      << "tl.ascend_atomic_add source region rank must match source buffer "
         "rank";
  ICHECK_EQ(dst_extents.size(), dst->shape.size())
      << "tl.ascend_atomic_add destination region rank must match "
         "destination buffer rank";

  std::stringstream ss;
  if (src.scope() == "shared") {
    ss << "tl::ascend::atomic_add_ub_to_gm<";
    ss << get_dtype(dst) << ", " << src_extents[src->shape.size() - 1];
    if (src->shape.size() > 1) {
      ss << ", " << compute_blocklen(src, src_extents);
    }
    ss << ">";
  } else if (src.scope() == "wmma.accumulator") {
    ss << "tl::ascend::atomic_add_l0c_to_gm<";
    ss << get_dtype(src) << ", " << get_dtype(dst) << ", "
       << (T.layout_map.count(src) ? T.layout_map[src]->AscendLayoutStr()
                                   : "layout::RowMajor");
    if (src->shape.size() > 1) {
      ss << ", " << compute_blocklen(src, src_extents) << ", "
         << src_extents[src->shape.size() - 1];
    } else {
      ss << ", 1, " << src_extents[src->shape.size() - 1];
    }
    ss << ">";
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

  // Scalarize indices: extract base from Ramp expressions to avoid vector lane
  // mismatches when Ramp lanes differ across dimensions (e.g. int32x32 vs
  // int32x64). The Ramp lanes already contribute to the access extent (src_len
  // / dst_len).
  auto scalarize = [](const PrimExpr &idx) -> PrimExpr {
    if (const auto *ramp = idx.as<RampNode>()) {
      return ramp->base;
    }
    return idx;
  };
  auto src_scalar_indices = src_new_indices.Map(scalarize);
  auto dst_scalar_indices = dst_new_indices.Map(scalarize);

  PrimExpr src_len = 1;
  for (auto &shape : src_extents) {
    src_len *= shape;
  }
  // dst_len must also account for lanes in Ramp indices
  PrimExpr dst_len = 1;
  for (size_t i = 0; i < dst_extents.size(); ++i) {
    PrimExpr dim_len = dst_extents[i];
    if (const auto *ramp = dst_new_indices[i].as<RampNode>()) {
      dim_len = dim_len * ramp->lanes;
    }
    dst_len = dst_len * dim_len;
  }

  auto src_offset = src_new_buffer.OffsetOf(src_scalar_indices).back();
  auto dst_offset = dst_new_buffer.OffsetOf(dst_scalar_indices).back();

  auto src_ptr = src_new_buffer.access_ptr(1, src_new_buffer->dtype, 1,
                                           src_offset, src_len);
  auto dst_ptr = dst_new_buffer.access_ptr(2, dst_new_buffer->dtype, 1,
                                           dst_offset, dst_len);

  // Compute effective extents accounting for lanes in Ramp indices,
  // so that find_active_dim_indices correctly identifies the 2D access pattern.
  Array<PrimExpr> effective_dst_extents;
  for (size_t i = 0; i < dst_extents.size(); ++i) {
    PrimExpr eff = dst_extents[i];
    if (const auto *ramp = dst_new_indices[i].as<RampNode>()) {
      // When a dimension was vectorized from extent N to 1 vector with N lanes,
      // the true element count is the Ramp lanes.  When the extent already
      // reflects the total element count (extent == Ramp lanes), keep it as-is.
      if (is_one(dst_extents[i])) {
        eff = ramp->lanes;
      }
    }
    effective_dst_extents.push_back(eff);
  }

  // Identify row/col Ramp dimensions for stride and boundary computation.
  // When vectorized, Ramps with different lane counts (e.g. 32, 64) mark the
  // true 2D access pattern, bypassing scrambled extents.
  auto get_ramp_lanes = [](const PrimExpr &idx) -> int {
    if (const auto *ramp = idx.as<RampNode>()) {
      if (auto *imm = ramp->lanes.as<IntImmNode>()) {
        return static_cast<int>(imm->value);
      }
    }
    return 0;
  };
  int row_ramp_idx = -1, col_ramp_idx = -1;
  for (size_t i = 0; i < dst_new_indices.size(); ++i) {
    int lanes = get_ramp_lanes(dst_new_indices[i]);
    if (lanes > 1) {
      if (row_ramp_idx < 0 ||
          lanes < get_ramp_lanes(dst_new_indices[row_ramp_idx])) {
        col_ramp_idx = row_ramp_idx >= 0 ? row_ramp_idx : col_ramp_idx;
        row_ramp_idx = static_cast<int>(i);
      } else {
        col_ramp_idx = static_cast<int>(i);
      }
    }
  }

  PrimExpr validRow_dst, validCol_dst;
  if (row_ramp_idx >= 0 && col_ramp_idx >= 0) {
    // Use Ramp-identified dimensions for boundary checks
    PrimExpr row_min = dst_scalar_indices[row_ramp_idx];
    PrimExpr col_min = dst_scalar_indices[col_ramp_idx];
    PrimExpr row_extent = effective_dst_extents[row_ramp_idx];
    PrimExpr col_extent = effective_dst_extents[col_ramp_idx];
    validRow_dst =
        compute_valid_extent(row_min, row_extent, dst->shape[row_ramp_idx]);
    validCol_dst =
        compute_valid_extent(col_min, col_extent, dst->shape[col_ramp_idx]);
  } else {
    std::vector<int> dst_active =
        find_active_dim_indices(effective_dst_extents);
    if (dst_active.size() >= 2) {
      int row_idx = dst_active[dst_active.size() - 2];
      int col_idx = dst_active.back();
      PrimExpr row_min = dst_scalar_indices[row_idx];
      PrimExpr col_min = dst_scalar_indices[col_idx];
      PrimExpr row_extent = effective_dst_extents[row_idx];
      PrimExpr col_extent = effective_dst_extents[col_idx];
      validRow_dst =
          compute_valid_extent(row_min, row_extent, dst->shape[row_idx]);
      validCol_dst =
          compute_valid_extent(col_min, col_extent, dst->shape[col_idx]);
    } else if (dst_active.size() == 1) {
      int col_idx = dst_active[0];
      PrimExpr col_min = dst_scalar_indices[col_idx];
      PrimExpr col_extent = effective_dst_extents[col_idx];
      validRow_dst = Integer(1);
      validCol_dst =
          compute_valid_extent(col_min, col_extent, dst->shape[col_idx]);
    } else {
      validRow_dst = Integer(0);
      validCol_dst = Integer(0);
    }
  }

  // Compute strideN: when Ramp dimensions are identified, use buffer shape
  // to compute the inter-row stride. Otherwise fall back to extent-based logic.
  PrimExpr strideN;
  if (row_ramp_idx >= 0 && col_ramp_idx >= 0) {
    strideN = Integer(1);
    for (int i = row_ramp_idx + 1; i < static_cast<int>(dst->shape.size());
         ++i) {
      strideN = strideN * dst->shape[i];
    }
  } else {
    strideN = compute_strideN(dst, dst_extents);
  }

  Array<PrimExpr> new_args;
  new_args.push_back(StringImm(ss.str()));
  new_args.push_back(src_ptr);
  new_args.push_back(dst_ptr);
  new_args.push_back(strideN);
  new_args.push_back(validRow_dst);
  new_args.push_back(validCol_dst);

  auto new_call = Call(DataType::Handle(), builtin::call_extern(), new_args);
  return Evaluate(new_call);
}

LayoutMap AscendCopy::InferLayout(const LayoutInferArgs &T, InferLevel level) {
  LayoutMap results;
  // TODO: add logic to infer layout for AscendCopy
  return results;
}

LayoutMap AscendAtomicAdd::InferLayout(const LayoutInferArgs &T,
                                       InferLevel level) {
  LayoutMap results;
  return results;
}

TIR_REGISTER_TL_OP(AscendCopy, ascend_copy)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(AscendAtomicAdd, ascend_atomic_add)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

const Op &ascend_atomic_add() { return AscendAtomicAdd::Get(); }

TIR_DEFINE_TL_BUILTIN(ascend_add)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_sub)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_mul)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_div)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_max)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_min)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_bitwise_and)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_bitwise_or)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_adds)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_subs)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_muls)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_divs)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_maxs)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_mins)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_compare)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_compare_scalar)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_exp)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_ln).set_num_inputs(3).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_abs)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_reciprocal)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_sqrt)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_rsqrt)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_relu)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_bitwise_not)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_select)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_leaky_relu)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_axpy)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_mul_add_dst)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_bitwise_lshift)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_bitwise_rshift)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_sin)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_cos)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_transpose)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_createvecindex)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_fill)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_arith_progression)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_sort)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_merge_sort)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_topk)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_shmem_put_nbi)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_shmem_get_nbi)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_shmem_ub_put_nbi)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_shmem_ub_get_nbi)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_gather_mask)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_gatherb)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_init_sort_buf)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_sort32)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_gather)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_reduce)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_block_reduce_max)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_block_reduce_min)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_block_reduce_sum)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_cast)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_set_deq_scale)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_pow)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_bitwise_xor)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_broadcast)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_row_expand_mul)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_wait_cross_flag)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_set_cross_flag)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_wait_flag)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_set_flag)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_pipe_barrier)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_sync_all)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_gemm_v0)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_gemm_v1)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_printf)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_dump_tensor)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_bilinear_interpolation)
    .set_num_inputs(11)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_wholereducemax)
    .set_num_inputs(8)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_wholereducemin)
    .set_num_inputs(8)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_wholereducesum)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_auto_barrier)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_auto_set_flag)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_auto_wait_flag)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_auto_set_cross_flag)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_auto_wait_cross_flag)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_use_swizzle)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_mma)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_sigmoid)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_silu)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_clamp_max)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_clamp_min)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_clamp)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_round)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_reinterpretcast)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_sub_experiment)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_abs_experiment)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_mins_experiment)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_reducesum_experiment)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_reducesum_mask_experiment)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_gather_mask_experiment)
    .set_num_inputs(11)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_fill_experiment)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_sum_experiment)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_datacachecleanandinvalid_experiment)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_brcb)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
} // namespace tl
} // namespace tvm
