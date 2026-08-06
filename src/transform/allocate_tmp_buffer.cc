// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include <tvm/ir/name_supply.h>
#include <tvm/ir/op.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../op/ascend.h"
#include "common/operation_config.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

namespace {

bool IsConstFalse(const PrimExpr &expr) {
  return expr.defined() && expr.dtype().is_bool() && is_zero(expr);
}

bool IsAccessPtrCall(const PrimExpr &expr) {
  const auto *call = expr.as<CallNode>();
  return call && call->op.same_as(builtin::tvm_access_ptr());
}

bool IsExplicitWorkspace(const PrimExpr &expr) { return IsAccessPtrCall(expr); }

struct ReduceCallLayout {
  size_t clear_index;
  size_t tmp_count;
  bool clear;
  int64_t physical_row;
};

struct WorkspaceSpec {
  bool requires_workspace;
  DataType view_dtype;
  // Bytes in the primary view. This may be zero when a PTO column reduction
  // needs only its separate clear=False output view.
  int64_t primary_bytes;
  int64_t access_mask;
};

ReduceCallLayout ParseReduceCallLayout(const CallNode *op) {
  ICHECK(op->op.same_as(tl::ascend_reduce()));
  ICHECK_GE(op->args.size(), 4U) << "Malformed Ascend reduce call.";

  size_t clear_index = op->args.size() - 1;
  int64_t physical_row = 0;
  if (!op->args[clear_index].dtype().is_bool() &&
      op->args[clear_index].as<IntImmNode>()) {
    physical_row = Downcast<IntImm>(op->args[clear_index])->value;
    ICHECK_GT(clear_index, 0U) << "Malformed Ascend reduce call.";
    clear_index--;
  }
  ICHECK(op->args[clear_index].dtype().is_bool())
      << "Ascend reduce clear argument must precede the optional physical row.";
  ICHECK_GE(clear_index, 3U) << "Malformed Ascend reduce call.";

  const size_t tmp_count = clear_index - 3;
  ICHECK_LE(tmp_count, 2U)
      << "Ascend reduce accepts at most a main and an output tmp view.";
  for (size_t i = 0; i < tmp_count; ++i) {
    ICHECK(IsAccessPtrCall(op->args[3 + i]))
        << "Ascend reduce optional tmp operands must be access_ptr views.";
  }
  return {clear_index, tmp_count, !IsConstFalse(op->args[clear_index]),
          physical_row};
}

bool HasWorkspaceOperand(const CallNode *op, int64_t tmp_pos) {
  if (op->op.same_as(tl::ascend_reduce())) {
    return ParseReduceCallLayout(op).tmp_count > 0;
  }

  if (op->op.same_as(tl::ascend_merge_sort())) {
    ICHECK_GE(op->args.size(), 2U) << "Malformed merge_sort call.";
    const auto *num_ways = op->args[1].as<IntImmNode>();
    ICHECK(num_ways) << "merge_sort num_ways must be static.";
    ICHECK_GE(num_ways->value, 2);
    ICHECK_LE(num_ways->value, 4);
    const size_t implicit_arg_count = 3 + 2 * num_ways->value;
    ICHECK(op->args.size() == implicit_arg_count ||
           op->args.size() == implicit_arg_count + 1)
        << "Malformed merge_sort call with " << op->args.size()
        << " arguments for " << num_ways->value << " inputs.";
    return op->args.size() == implicit_arg_count + 1;
  }

  return tmp_pos < static_cast<int64_t>(op->args.size()) &&
         IsExplicitWorkspace(op->args[tmp_pos]);
}

const BufferNode *FindBufferByDataVar(const Array<Buffer> &alloc_buffers,
                                      const VarNode *data_var) {
  for (const Buffer &buffer : alloc_buffers) {
    if (buffer->data.get() == data_var) {
      return buffer.get();
    }
  }
  return nullptr;
}

const CallNode *AsAccessPtr(const PrimExpr &expr) {
  const auto *access_ptr = expr.as<CallNode>();
  ICHECK(access_ptr && access_ptr->op.same_as(builtin::tvm_access_ptr()))
      << "Expected tvm_access_ptr.";
  return access_ptr;
}

int64_t GetAccessPtrOffset(const PrimExpr &expr) {
  return Downcast<IntImm>(AsAccessPtr(expr)->args[2])->value;
}

int64_t GetAccessPtrExtent(const PrimExpr &expr) {
  return Downcast<IntImm>(AsAccessPtr(expr)->args[3])->value;
}

DataType GetAccessPtrDtype(const PrimExpr &expr) {
  const CallNode *access_ptr = AsAccessPtr(expr);
  const auto *type_annotation = access_ptr->args[0].as<CallNode>();
  ICHECK(type_annotation) << "Expected access-ptr type annotation.";
  return type_annotation->dtype;
}

int64_t GetAccessPtrBytes(const PrimExpr &expr) {
  return GetAccessPtrExtent(expr) * GetAccessPtrDtype(expr).bytes();
}

int64_t GetAccessPtrByteOffset(const PrimExpr &expr) {
  return GetAccessPtrOffset(expr) * GetAccessPtrDtype(expr).bytes();
}

std::string Trim(const std::string &value) {
  const size_t first = value.find_first_not_of(" \t");
  if (first == std::string::npos) {
    return "";
  }
  const size_t last = value.find_last_not_of(" \t");
  return value.substr(first, last - first + 1);
}

int64_t ParseStaticInt(const std::string &value, const std::string &op_name) {
  size_t parsed = 0;
  int64_t result = 0;
  try {
    result = std::stoll(value, &parsed);
  } catch (const std::exception &) {
    ICHECK(false) << "Expected static reduce extents in " << op_name;
  }
  ICHECK_EQ(parsed, value.size())
      << "Expected static reduce extents in " << op_name;
  return result;
}

enum class ReduceKind { kSum, kMax, kMin };

struct ReduceTemplateInfo {
  ReduceKind kind;
  std::string dtype;
  int64_t rows;
  int64_t cols;
  int direction;
};

ReduceTemplateInfo ParseReduceTemplateInfo(const CallNode *call) {
  const std::string op_name = Downcast<StringImm>(call->args[0])->value;
  const size_t left = op_name.find('<');
  const size_t right = op_name.rfind('>');
  ICHECK(left != std::string::npos && right != std::string::npos &&
         left < right)
      << "Failed to parse reduce template " << op_name;

  std::vector<std::string> params;
  size_t begin = left + 1;
  while (begin < right) {
    const size_t comma = op_name.find(',', begin);
    const size_t end =
        comma == std::string::npos || comma > right ? right : comma;
    params.push_back(Trim(op_name.substr(begin, end - begin)));
    begin = end + 1;
  }
  ICHECK_EQ(params.size(), 4U) << "Failed to parse reduce template " << op_name;

  ReduceKind kind;
  if (op_name.find("reduce_sum") != std::string::npos) {
    kind = ReduceKind::kSum;
  } else if (op_name.find("reduce_max") != std::string::npos) {
    kind = ReduceKind::kMax;
  } else {
    ICHECK(op_name.find("reduce_min") != std::string::npos)
        << "Unsupported reduce operation " << op_name;
    kind = ReduceKind::kMin;
  }

  const int64_t rows = ParseStaticInt(params[1], op_name);
  const int64_t cols = ParseStaticInt(params[2], op_name);
  const int64_t direction = ParseStaticInt(params[3], op_name);
  ICHECK_GT(rows, 0);
  ICHECK_GT(cols, 0);
  ICHECK(direction == -1 || direction == 0)
      << "Only row and column reductions are supported.";
  return {kind, params[0], rows, cols, static_cast<int>(direction)};
}

int64_t FloorPowerOfTwo(int64_t value) {
  ICHECK_GT(value, 0);
  int64_t result = 1;
  while (result <= value / 2) {
    result *= 2;
  }
  return result;
}

int64_t AlignUp(int64_t value, int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

struct BroadcastTemplateInfo {
  DataType dtype;
  int64_t dim;
  int64_t axis;
  std::vector<int64_t> dst_shape;
  std::vector<int64_t> src_shape;
};

BroadcastTemplateInfo ParseBroadcastTemplateInfo(const CallNode *call) {
  ICHECK(call->op.same_as(tl::ascend_broadcast()));
  const bool has_workspace = HasWorkspaceOperand(call, 3);
  const size_t dim_index = has_workspace ? 4 : 3;
  ICHECK_GT(call->args.size(), dim_index);
  const int64_t dim = Downcast<IntImm>(call->args[dim_index])->value;
  ICHECK(dim == 1 || dim == 2)
      << "Broadcast only supports one or two dimensions.";
  ICHECK_EQ(call->args.size(), dim_index + 1 + 2 * dim)
      << "Malformed Ascend broadcast call.";

  std::vector<int64_t> dst_shape;
  std::vector<int64_t> src_shape;
  dst_shape.reserve(dim);
  src_shape.reserve(dim);
  for (int64_t i = 0; i < dim; ++i) {
    dst_shape.push_back(Downcast<IntImm>(call->args[dim_index + 1 + i])->value);
    src_shape.push_back(
        Downcast<IntImm>(call->args[dim_index + 1 + dim + i])->value);
  }

  const std::string op_name = Downcast<StringImm>(call->args[0])->value;
  const size_t left = op_name.find('<');
  const size_t right = op_name.rfind('>');
  ICHECK(left != std::string::npos && right != std::string::npos &&
         left < right)
      << "Failed to parse broadcast template " << op_name;
  std::vector<std::string> params;
  size_t begin = left + 1;
  while (begin < right) {
    const size_t comma = op_name.find(',', begin);
    const size_t end =
        comma == std::string::npos || comma > right ? right : comma;
    params.push_back(Trim(op_name.substr(begin, end - begin)));
    begin = end + 1;
  }
  ICHECK_GE(params.size(), 3U)
      << "Failed to parse broadcast template " << op_name;
  const int64_t axis = ParseStaticInt(params[2], op_name);
  ICHECK(axis == 0 || axis == 1);
  return {GetAccessPtrDtype(call->args[2]), dim, axis, dst_shape, src_shape};
}

int64_t ShapeElements(const std::vector<int64_t> &shape) {
  int64_t elements = 1;
  for (int64_t extent : shape) {
    ICHECK_GE(extent, 0);
    elements *= extent;
  }
  return elements;
}

int64_t EstimateAscendCBroadcastWorkspaceBytes(const CallNode *call) {
  const BroadcastTemplateInfo info = ParseBroadcastTemplateInfo(call);
  const int64_t dtype_bytes = info.dtype.bytes();
  ICHECK(dtype_bytes == 1 || dtype_bytes == 2 || dtype_bytes == 4)
      << "dav-2201 Broadcast workspace policy only supports b8/b16/b32, got "
      << info.dtype;

  const bool equal_shape = info.dst_shape == info.src_shape;
  const int64_t src_elements = ShapeElements(info.src_shape);
  const int64_t dst_elements = ShapeElements(info.dst_shape);
  const bool scalar_source = src_elements == 1;

  auto inner_workspace_elements = [&](int64_t element_bytes) {
    if (equal_shape || scalar_source) {
      return int64_t{0};
    }
    const int64_t elements_per_block = 32 / element_bytes;
    if (info.axis == 0) {
      return elements_per_block;
    }
    ICHECK_EQ(info.dim, 2);
    int64_t elements = elements_per_block * elements_per_block;
    const int64_t inner = info.dst_shape[1];
    if ((inner * element_bytes) % 32 != 0) {
      elements += elements_per_block * AlignUp(inner, elements_per_block);
    }
    return elements;
  };

  if (dtype_bytes == 1) {
    // The dav-2201 b8 implementation casts both tensors to half and then uses
    // the half broadcast staging area. All quantities below are half elements.
    const int64_t half_src = AlignUp(src_elements, 16);
    const int64_t half_dst = AlignUp(dst_elements, 16);
    const int64_t half_inner = inner_workspace_elements(/*element_bytes=*/2);
    return 2 * (half_src + half_dst + half_inner);
  }

  return inner_workspace_elements(dtype_bytes) * dtype_bytes;
}

int64_t AlignReduceOutputCols(int64_t valid_col, int64_t dtype_bytes) {
  const int64_t aligned_bytes = ((valid_col * dtype_bytes + 31) / 32) * 32;
  return aligned_bytes / dtype_bytes;
}

int64_t GetPtoRowReduceTmpCols(int64_t valid_col, int64_t dtype_bytes) {
  constexpr int64_t kVectorRepeatBytes = 256;
  const int64_t elem_per_repeat = kVectorRepeatBytes / dtype_bytes;
  const int64_t tmp_col = valid_col <= elem_per_repeat
                              ? 1
                              : std::max(valid_col / 2, elem_per_repeat);
  return AlignReduceOutputCols(tmp_col, dtype_bytes);
}

int64_t EstimatePTOReduceWorkspaceBytes(const CallNode *call,
                                        const Array<Buffer> &alloc_buffers) {
  const ReduceTemplateInfo info = ParseReduceTemplateInfo(call);
  if (info.direction == 0) {
    return 0;
  }

  const CallNode *src_access_ptr = AsAccessPtr(call->args[2]);
  const auto *src_var = src_access_ptr->args[1].as<VarNode>();
  ICHECK(src_var) << "Expected reduce source data variable.";
  const BufferNode *src_buffer = FindBufferByDataVar(alloc_buffers, src_var);
  ICHECK(src_buffer) << "Buffer not found for " << src_var->name_hint;

  const int64_t dtype_bytes = src_buffer->dtype.bytes();
  const int64_t tmp_col = GetPtoRowReduceTmpCols(info.cols, dtype_bytes);
  return info.rows * tmp_col * dtype_bytes;
}

bool AscendCReduceUsesTmp(const CallNode *call) {
  const ReduceCallLayout layout = ParseReduceCallLayout(call);
  const ReduceTemplateInfo info = ParseReduceTemplateInfo(call);
  if (layout.physical_row > 0) {
    return false;
  }
  return !(info.kind == ReduceKind::kSum && info.dtype == "half" &&
           layout.clear);
}

int64_t
EstimateAscendCReduceWorkspaceBytes(const CallNode *call,
                                    const Array<Buffer> &alloc_buffers) {
  if (!AscendCReduceUsesTmp(call)) {
    return 0;
  }

  const ReduceTemplateInfo info = ParseReduceTemplateInfo(call);
  const CallNode *src_access_ptr = AsAccessPtr(call->args[2]);
  const auto *src_var = src_access_ptr->args[1].as<VarNode>();
  ICHECK(src_var) << "Expected reduce source data variable.";
  const BufferNode *src_buffer = FindBufferByDataVar(alloc_buffers, src_var);
  ICHECK(src_buffer) << "Buffer not found for " << src_var->name_hint;
  const int64_t dtype_bytes = src_buffer->dtype.bytes();

  // Transitional allocation heuristic for the CANN Advanced Reduce APIs. It
  // mirrors the public GetReduce{Sum,Max,Min}MaxMinTmpSize results for the
  // static AR/RA cases used here without making their version-specific sizing
  // contract part of TileLang's public explicit-tmp API. This goes away when
  // the AscendC backend switches to its TileLang-owned reduce helpers.
  int64_t bytes = 0;
  if (info.direction == 0) {
    const int64_t padded_row_bytes = AlignUp(info.cols * dtype_bytes, 32);
    const int64_t power = FloorPowerOfTwo(info.rows);
    const int64_t active_rows =
        info.rows == 1 ? 1 : (info.rows == power ? power / 2 : power);
    bytes = active_rows * padded_row_bytes;
  } else if (info.kind == ReduceKind::kSum) {
    constexpr int64_t kVectorRepeatBytes = 256;
    const int64_t elements_per_repeat = kVectorRepeatBytes / dtype_bytes;
    if (info.cols > elements_per_repeat) {
      const int64_t power = FloorPowerOfTwo(info.cols);
      const int64_t per_row_elements =
          info.rows == 1 || info.cols != power ? power : power / 2;
      bytes = info.rows * per_row_elements * dtype_bytes;
    }
  } else {
    constexpr int64_t kDataBlockBytes = 32;
    constexpr int64_t kVectorRepeatBytes = 256;
    const int64_t elements_per_block = kDataBlockBytes / dtype_bytes;
    const int64_t elements_per_repeat = kVectorRepeatBytes / dtype_bytes;
    if (info.cols > elements_per_block) {
      bytes =
          info.rows * (info.cols < elements_per_repeat ? kDataBlockBytes
                                                       : kVectorRepeatBytes);
    }
  }

  // The current wrapper still has a sharedTmpBuffer parameter even when the
  // selected CANN branch reports zero bytes. Keep one aligned block
  // for implicit calls; explicit arenas are deliberately not size-checked.
  return std::max<int64_t>(bytes, 32);
}

int64_t EstimateReduceOutputWorkspaceBytes(const CallNode *call,
                                           const Array<Buffer> &alloc_buffers) {
  const CallNode *dst_access_ptr = AsAccessPtr(call->args[1]);
  const auto *dst_var = dst_access_ptr->args[1].as<VarNode>();
  ICHECK(dst_var) << "Expected reduce destination data variable.";
  const BufferNode *dst_buffer = FindBufferByDataVar(alloc_buffers, dst_var);
  ICHECK(dst_buffer) << "Buffer not found for " << dst_var->name_hint;

  int64_t col = 1;
  if (dst_buffer->shape.size() == 1) {
    col = Downcast<IntImm>(dst_buffer->shape[0])->value;
  } else if (dst_buffer->shape.size() == 2 &&
             Downcast<IntImm>(dst_buffer->shape[0])->value == 0) {
    col = Downcast<IntImm>(dst_buffer->shape[1])->value;
  } else if (dst_buffer->shape.size() == 2 &&
             Downcast<IntImm>(dst_buffer->shape[1])->value == 0) {
    col = Downcast<IntImm>(dst_buffer->shape[0])->value;
  } else {
    ICHECK_GE(dst_buffer->shape.size(), 2U);
    col = Downcast<IntImm>(dst_buffer->shape[1])->value;
  }

  const int64_t extent = GetAccessPtrExtent(call->args[1]);
  const int64_t valid_row = std::max<int64_t>((extent + col - 1) / col, 1);
  const int64_t valid_col = extent > col ? col : extent;
  const int64_t padded_col =
      AlignReduceOutputCols(valid_col, dst_buffer->dtype.bytes());
  return valid_row * padded_col * dst_buffer->dtype.bytes();
}

int64_t EstimatePTOWorkspaceBytes(const CallNode *call,
                                  const Array<Buffer> &alloc_buffers) {
  if (call->op.same_as(tl::ascend_reduce())) {
    return EstimatePTOReduceWorkspaceBytes(call, alloc_buffers);
  }
  if (call->op.same_as(tl::ascend_bitwise_xor())) {
    return GetAccessPtrBytes(call->args[1]);
  }
  if (call->op.same_as(tl::ascend_merge_sort())) {
    const DataType dtype = GetAccessPtrDtype(call->args[2]);
    const int64_t extent = GetAccessPtrExtent(call->args[2]);
    return extent * (dtype == DataType::UInt(8) ? 4 : dtype.bytes());
  }
  if (call->op.same_as(tl::ascend_select())) {
    return GetAccessPtrBytes(call->args[0]);
  }
  if (call->op.same_as(tl::ascend_gather_mask())) {
    ICHECK(IsAccessPtrCall(call->args[3]));
    return GetAccessPtrBytes(call->args[3]);
  }
  if (call->op.same_as(tl::ascend_sort()) ||
      call->op.same_as(tl::ascend_topk())) {
    const bool is_topk = call->op.same_as(tl::ascend_topk());
    const DataType dtype = GetAccessPtrDtype(call->args[2]);
    int64_t aligned_count = GetAccessPtrExtent(call->args[2]);
    if (is_topk) {
      const size_t max_actual_index = HasWorkspaceOperand(call, 3) ? 7 : 6;
      const int64_t max_actual_num =
          Downcast<IntImm>(call->args[max_actual_index])->value;
      aligned_count = AlignUp(max_actual_num, 32);
    }
    const int64_t multiplier = dtype.bytes() == 2 ? 16 : (is_topk ? 6 : 4);
    return aligned_count * multiplier * dtype.bytes();
  }
  if (call->op.same_as(tl::ascend_gather())) {
    return GetAccessPtrBytes(call->args[2]);
  }
  ICHECK(false) << "Missing PTO workspace heuristic for "
                << call->op.as<OpNode>()->name;
  return 0;
}

// A public tmp arena has no input-content contract. Every consuming call may
// clobber it, so dependence analysis must treat the view as a write.
constexpr int64_t kWorkspaceWriteAccess = 2;

WorkspaceSpec NoWorkspace() {
  return WorkspaceSpec{false, DataType::UInt(8), 0, kWorkspaceWriteAccess};
}

WorkspaceSpec RequireWorkspace(DataType dtype, int64_t bytes) {
  ICHECK_GE(bytes, 0);
  return WorkspaceSpec{true, dtype, bytes, kWorkspaceWriteAccess};
}

WorkspaceSpec GetPTOWorkspaceSpec(const CallNode *call,
                                  const Array<Buffer> &alloc_buffers) {
  const DataType byte_dtype = DataType::UInt(8);
  if (call->op.same_as(tl::ascend_reduce())) {
    const ReduceTemplateInfo info = ParseReduceTemplateInfo(call);
    const ReduceCallLayout layout = ParseReduceCallLayout(call);
    if (info.direction != -1 && layout.clear) {
      return NoWorkspace();
    }
    return RequireWorkspace(
        byte_dtype, EstimatePTOReduceWorkspaceBytes(call, alloc_buffers));
  }
  if (call->op.same_as(tl::ascend_bitwise_xor())) {
    return RequireWorkspace(GetAccessPtrDtype(call->args[1]),
                            EstimatePTOWorkspaceBytes(call, alloc_buffers));
  }
  if (call->op.same_as(tl::ascend_sort()) ||
      call->op.same_as(tl::ascend_topk()) ||
      call->op.same_as(tl::ascend_merge_sort())) {
    return RequireWorkspace(GetAccessPtrDtype(call->args[2]),
                            EstimatePTOWorkspaceBytes(call, alloc_buffers));
  }
  if (call->op.same_as(tl::ascend_select())) {
    return RequireWorkspace(byte_dtype,
                            EstimatePTOWorkspaceBytes(call, alloc_buffers));
  }
  if (call->op.same_as(tl::ascend_gather_mask())) {
    if (!IsAccessPtrCall(call->args[3])) {
      return NoWorkspace();
    }
    return RequireWorkspace(GetAccessPtrDtype(call->args[3]),
                            EstimatePTOWorkspaceBytes(call, alloc_buffers));
  }
  if (call->op.same_as(tl::ascend_gather())) {
    return RequireWorkspace(GetAccessPtrDtype(call->args[2]),
                            EstimatePTOWorkspaceBytes(call, alloc_buffers));
  }
  if (call->op.same_as(tl::ascend_clamp()) ||
      call->op.same_as(tl::ascend_clamp_max()) ||
      call->op.same_as(tl::ascend_clamp_min()) ||
      call->op.same_as(tl::ascend_sigmoid()) ||
      call->op.same_as(tl::ascend_pow()) ||
      call->op.same_as(tl::ascend_round()) ||
      call->op.same_as(tl::ascend_broadcast())) {
    return NoWorkspace();
  }
  ICHECK(false) << "Missing PTO workspace policy for "
                << call->op.as<OpNode>()->name;
  return NoWorkspace();
}

WorkspaceSpec GetAscendCWorkspaceSpec(const CallNode *call,
                                      const Array<Buffer> &alloc_buffers) {
  const DataType byte_dtype = DataType::UInt(8);
  if (call->op.same_as(tl::ascend_reduce())) {
    if (!AscendCReduceUsesTmp(call)) {
      return NoWorkspace();
    }
    return RequireWorkspace(
        byte_dtype, EstimateAscendCReduceWorkspaceBytes(call, alloc_buffers));
  }
  if (call->op.same_as(tl::ascend_broadcast())) {
    const int64_t bytes = EstimateAscendCBroadcastWorkspaceBytes(call);
    return bytes == 0 ? NoWorkspace() : RequireWorkspace(byte_dtype, bytes);
  }
  if (call->op.same_as(tl::ascend_sort()) ||
      call->op.same_as(tl::ascend_topk())) {
    const DataType dtype = GetAccessPtrDtype(call->args[2]);
    const bool is_half = dtype.is_float() && dtype.bits() == 16;
    const bool is_float = dtype.is_float() && dtype.bits() == 32;
    ICHECK(is_half || is_float)
        << "dav-2201 Sort/TopK only supports float16/float32, got " << dtype;
    const bool has_workspace = HasWorkspaceOperand(call, 3);
    const size_t repeat_index = call->op.same_as(tl::ascend_sort())
                                    ? (has_workspace ? 4 : 3)
                                    : (has_workspace ? 5 : 4);
    const int64_t repeat_times =
        Downcast<IntImm>(call->args[repeat_index])->value;
    const int64_t count = repeat_times * 32;
    const int64_t factor = call->op.same_as(tl::ascend_sort())
                               ? (is_half ? 8 : 2)
                               : (is_half ? 10 : 4);
    return RequireWorkspace(dtype, factor * count * dtype.bytes());
  }
  if (call->op.same_as(tl::ascend_bilinear_interpolation())) {
    const int64_t bytes = (GetAccessPtrExtent(call->args[1]) +
                           GetAccessPtrExtent(call->args[3])) *
                          32;
    return RequireWorkspace(byte_dtype, bytes);
  }
  if (call->op.same_as(tl::ascend_sin()) ||
      call->op.same_as(tl::ascend_cos())) {
    const DataType dtype = GetAccessPtrDtype(call->args[1]);
    const int64_t src_bytes = GetAccessPtrBytes(call->args[1]);
    ICHECK(dtype.is_float() && (dtype.bits() == 16 || dtype.bits() == 32));
    const int64_t minimum = dtype.bits() == 16 ? 512 : 384;
    return RequireWorkspace(byte_dtype, std::max(2 * src_bytes, minimum));
  }
  if (call->op.same_as(tl::ascend_pow())) {
    const DataType dtype = GetAccessPtrDtype(call->args[1]);
    const int64_t src_bytes = GetAccessPtrBytes(call->args[1]);
    const bool is_half = dtype.is_float() && dtype.bits() == 16;
    const bool is_float = dtype.is_float() && dtype.bits() == 32;
    const bool is_int32 = dtype == DataType::Int(32);
    ICHECK(is_half || is_float || is_int32)
        << "dav-2201 Pow only supports float16/float32/int32, got " << dtype;
    const int64_t minimum = is_half ? 1152 : 768;
    return RequireWorkspace(byte_dtype, std::max(2 * src_bytes, minimum));
  }
  if (call->op.same_as(tl::ascend_bitwise_xor())) {
    return RequireWorkspace(
        byte_dtype, std::max<int64_t>(GetAccessPtrBytes(call->args[1]), 64));
  }
  if (call->op.same_as(tl::ascend_clamp()) ||
      call->op.same_as(tl::ascend_clamp_max()) ||
      call->op.same_as(tl::ascend_clamp_min())) {
    return NoWorkspace();
  }
  if (call->op.same_as(tl::ascend_round())) {
    const DataType dtype = GetAccessPtrDtype(call->args[1]);
    if (dtype.is_float() && dtype.bits() == 32) {
      return NoWorkspace();
    }
    ICHECK(dtype.is_float() && dtype.bits() == 16)
        << "dav-2201 Round only supports float16/float32, got " << dtype;
    return RequireWorkspace(
        byte_dtype, std::max<int64_t>(GetAccessPtrBytes(call->args[1]), 256));
  }
  if (call->op.same_as(tl::ascend_sigmoid())) {
    return RequireWorkspace(byte_dtype, GetAccessPtrBytes(call->args[1]));
  }
  if (call->op.same_as(tl::ascend_reducesum_experiment()) ||
      call->op.same_as(tl::ascend_reducesum_mask_experiment())) {
    return RequireWorkspace(GetAccessPtrDtype(call->args[1]),
                            GetAccessPtrBytes(call->args[1]));
  }
  if (call->op.same_as(tl::ascend_merge_sort()) ||
      call->op.same_as(tl::ascend_select()) ||
      call->op.same_as(tl::ascend_gather_mask()) ||
      call->op.same_as(tl::ascend_gather())) {
    return NoWorkspace();
  }
  ICHECK(false) << "Missing AscendC workspace policy for "
                << call->op.as<OpNode>()->name;
  return NoWorkspace();
}

WorkspaceSpec GetWorkspaceSpec(const CallNode *call,
                               const Array<Buffer> &alloc_buffers,
                               const std::string &target) {
  const auto *op_node = call->op.as<OpNode>();
  ICHECK(op_node);
  const auto config_it = GetWorkspaceOpConfigs().find(op_node);
  ICHECK(config_it != GetWorkspaceOpConfigs().end())
      << "Missing workspace operation configuration for " << op_node->name;

  if (target == "pto") {
    ICHECK(config_it->second.pto_supported)
        << op_node->name << " is not supported by the PTO backend";
    return GetPTOWorkspaceSpec(call, alloc_buffers);
  }
  ICHECK(target == "ascendc" || target == "auto")
      << "Unsupported workspace target model " << target;
  ICHECK(config_it->second.ascendc_supported)
      << op_node->name << " is not supported by the AscendC backend";
  return GetAscendCWorkspaceSpec(call, alloc_buffers);
}

} // namespace

class CallNodeCollector : public ExprVisitor, public StmtVisitor {
public:
  static std::vector<Call> Collect(PrimFunc f, Target target,
                                   const Array<Buffer> &alloc_buffers) {
    CallNodeCollector collector;
    collector.target_ = Downcast<String>(target.get()->attrs["model"]);
    collector.alloc_buffers_ = alloc_buffers;
    return collector.Find(f->body);
  }

private:
  std::vector<Call> Find(const Stmt &stmt) {
    calls_.clear();
    VisitStmt(stmt);
    return calls_;
  }

  void VisitExpr_(const CallNode *op) override {
    if (const auto *op_node = op->op.as<OpNode>()) {
      // Here we only focus on CallNodes that require a tmp parameter.
      const auto config_it = GetWorkspaceOpConfigs().find(op_node);
      if (config_it != GetWorkspaceOpConfigs().end()) {
        const int64_t tmp_pos = config_it->second.tmp_arg_index;
        const WorkspaceSpec spec =
            GetWorkspaceSpec(op, alloc_buffers_, target_);
        if (!HasWorkspaceOperand(op, tmp_pos) && spec.requires_workspace) {
          calls_.push_back(GetRef<Call>(op));
        }
      }
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr(const PrimExpr &expr) override {
    ExprVisitor::VisitExpr(expr);
  }

  std::vector<Call> calls_;
  std::string target_;
  Array<Buffer> alloc_buffers_;
};

class CallNodeModifier : public StmtExprMutator {
public:
  static Stmt Modify(PrimFunc f, Target target, Buffer &tmp_buffer,
                     Buffer &reduce_out_tmp_buffer,
                     const Array<Buffer> &alloc_buffers) {
    CallNodeModifier modifier;
    modifier.target_ = Downcast<String>(target.get()->attrs["model"]);
    modifier.tmp_buf_ = tmp_buffer;
    modifier.reduce_out_tmp_buf_ = reduce_out_tmp_buffer;
    modifier.alloc_buffers_ = alloc_buffers;
    return modifier.AddTmpArg(f->body);
  }

private:
  Stmt AddTmpArg(const Stmt &stmt) { return VisitStmt(stmt); }

  PrimExpr VisitExpr_(const CallNode *op) override {
    if (const auto *op_node = op->op.as<OpNode>()) {
      const auto config_it = GetWorkspaceOpConfigs().find(op_node);
      if (config_it != GetWorkspaceOpConfigs().end()) {
        const int64_t tmp_buffer_param_offset = config_it->second.tmp_arg_index;
        const bool has_workspace =
            HasWorkspaceOperand(op, tmp_buffer_param_offset);
        const WorkspaceSpec spec =
            GetWorkspaceSpec(op, alloc_buffers_, target_);
        if (!spec.requires_workspace) {
          return has_workspace
                     ? CallWithoutWorkspaceArgs(op, tmp_buffer_param_offset)
                     : StmtExprMutator::VisitExpr_(op);
        }
        if (has_workspace) {
          return HandleExistingTmp(op, tmp_buffer_param_offset, spec);
        }
        if (NeedReduceOutputTmp(op)) {
          return CallNodeAddReduceOutputTmp(op, tmp_buffer_param_offset, spec);
        }
        return CallNodeAddTmp(op, tmp_buffer_param_offset, spec);
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  std::string GetOpName(const CallNode *op) const {
    if (op->args.size() > 0) {
      if (const auto *name = op->args[0].as<StringImmNode>()) {
        return name->value;
      }
    }
    return op->op.as<OpNode>()->name;
  }

  PrimExpr MakeAccessPtrView(const PrimExpr &arena,
                             int64_t relative_offset_bytes,
                             int64_t extent_bytes, DataType dtype,
                             int64_t access_mask) const {
    const CallNode *access_ptr = AsAccessPtr(arena);
    const DataType arena_dtype = GetAccessPtrDtype(arena);
    const int64_t byte_offset =
        GetAccessPtrOffset(arena) * arena_dtype.bytes() + relative_offset_bytes;
    ICHECK_GE(byte_offset, 0) << "tmp arena byte offset must be non-negative";
    ICHECK_GE(extent_bytes, 0) << "tmp arena extent must be non-negative";
    ICHECK_EQ(byte_offset % dtype.bytes(), 0)
        << "tmp arena byte offset " << byte_offset
        << " cannot be represented as " << dtype;
    // A trailing partial element cannot be represented by the typed view and
    // remains outside the backend-visible extent.
    Array<PrimExpr> args{TypeAnnotation(dtype), access_ptr->args[1],
                         Integer(byte_offset / dtype.bytes()),
                         Integer(extent_bytes / dtype.bytes()),
                         Integer(access_mask)};
    return Call(DataType::Handle(), builtin::tvm_access_ptr(), args);
  }

  PrimExpr RetypeWorkspace(const PrimExpr &workspace,
                           const WorkspaceSpec &spec) const {
    return MakeAccessPtrView(workspace, 0, GetAccessPtrBytes(workspace),
                             spec.view_dtype, spec.access_mask);
  }

  PrimExpr ReplaceWorkspace(const CallNode *op, size_t workspace_index,
                            const PrimExpr &workspace) const {
    Array<PrimExpr> new_args = op->args;
    new_args.Set(workspace_index, workspace);
    return Call(op->dtype, op->op, new_args, op->span);
  }

  PrimExpr HandleExistingTmp(const CallNode *op,
                             int64_t tmp_buffer_param_offset,
                             const WorkspaceSpec &spec) {
    if (GetAccessPtrBytes(op->args[tmp_buffer_param_offset]) == 0) {
      ICHECK(false) << GetOpName(op) << " explicit tmp arena for target "
                    << target_ << " is empty, but this backend path requires "
                    << "a non-empty workspace.";
    }

    if (!op->op.same_as(tl::ascend_reduce())) {
      if (target_ == "pto") {
        return ReplaceWorkspace(
            op, tmp_buffer_param_offset,
            MakeAccessPtrView(op->args[tmp_buffer_param_offset], 0,
                              spec.primary_bytes, spec.view_dtype,
                              spec.access_mask));
      }
      return ReplaceWorkspace(
          op, tmp_buffer_param_offset,
          RetypeWorkspace(op->args[tmp_buffer_param_offset], spec));
    }

    // AscendC still lowers through transitional CANN helpers. Their heuristic
    // only sizes implicit allocations; it must not reject a user-provided
    // non-empty arena. PTO reduce(clear=False), in contrast, has a
    // TileLang-owned layout whose main/output views are derived here.
    if (target_ != "pto") {
      return ReplaceWorkspace(
          op, tmp_buffer_param_offset,
          RetypeWorkspace(op->args[tmp_buffer_param_offset], spec));
    }

    const int64_t main_bytes = spec.primary_bytes;

    const ReduceCallLayout layout = ParseReduceCallLayout(op);
    ICHECK_GE(layout.tmp_count, 1U);
    if (layout.clear) {
      ICHECK_EQ(layout.tmp_count, 1U)
          << "Only PTO reduce(clear=False) accepts two tmp views.";
      const PrimExpr &arena = op->args[tmp_buffer_param_offset];
      return ReplaceWorkspace(op, tmp_buffer_param_offset,
                              MakeAccessPtrView(arena, 0, main_bytes,
                                                spec.view_dtype,
                                                spec.access_mask));
    }

    const int64_t output_bytes =
        EstimateReduceOutputWorkspaceBytes(op, alloc_buffers_);
    if (main_bytes == 0) {
      ICHECK_EQ(layout.tmp_count, 1U)
          << "PTO column reduce(clear=False) expects one output tmp view.";
      const PrimExpr &arena = op->args[tmp_buffer_param_offset];

      Array<PrimExpr> new_args = op->args;
      new_args.Set(tmp_buffer_param_offset,
                   MakeAccessPtrView(arena, 0, output_bytes, spec.view_dtype,
                                     spec.access_mask));
      return Call(op->dtype, op->op, new_args, op->span);
    }

    if (layout.tmp_count == 2) {
      const PrimExpr &main_view = op->args[tmp_buffer_param_offset];
      const PrimExpr &output_view = op->args[tmp_buffer_param_offset + 1];
      if (AsAccessPtr(main_view)->args[1].same_as(
              AsAccessPtr(output_view)->args[1])) {
        ICHECK_GE(GetAccessPtrByteOffset(output_view),
                  GetAccessPtrByteOffset(main_view) +
                      GetAccessPtrBytes(main_view))
            << "PTO reduce tmp views must not overlap.";
      }
      return StmtExprMutator::VisitExpr_(op);
    }

    ICHECK_EQ(layout.tmp_count, 1U);
    const int64_t output_offset = AlignUp(main_bytes, 32);
    const PrimExpr &arena = op->args[tmp_buffer_param_offset];

    Array<PrimExpr> new_args;
    for (int64_t i = 0; i < tmp_buffer_param_offset; ++i) {
      new_args.push_back(op->args[i]);
    }
    new_args.push_back(MakeAccessPtrView(arena, 0, main_bytes, spec.view_dtype,
                                         spec.access_mask));
    new_args.push_back(MakeAccessPtrView(arena, output_offset, output_bytes,
                                         spec.view_dtype, spec.access_mask));
    for (size_t i = tmp_buffer_param_offset + 1; i < op->args.size(); ++i) {
      new_args.push_back(op->args[i]);
    }
    return Call(op->dtype, op->op, new_args, op->span);
  }

  Call CallNodeAddTmp(const CallNode *op, int64_t tmp_buffer_param_offset,
                      const WorkspaceSpec &spec) {
    PrimExpr access_ptr = this->AddTmpArgs_(op, spec);
    Array<PrimExpr> new_args =
        this->InsertExprAt_(op->args, tmp_buffer_param_offset, access_ptr);
    return Call(op->dtype, op->op, new_args, Span());
  }

  Call CallNodeAddReduceOutputTmp(const CallNode *op,
                                  int64_t tmp_buffer_param_offset,
                                  const WorkspaceSpec &spec) {
    Array<PrimExpr> new_args = op->args;
    if (spec.primary_bytes > 0) {
      new_args = this->InsertExprAt_(new_args, tmp_buffer_param_offset,
                                     this->AddTmpArgs_(op, spec));
      tmp_buffer_param_offset++;
    }
    new_args = this->InsertExprAt_(
        new_args, tmp_buffer_param_offset,
        this->MakeAccessPtrFromBuffer_(reduce_out_tmp_buf_, spec.access_mask));
    return Call(op->dtype, op->op, new_args, Span());
  }

  Call CallWithoutWorkspaceArgs(const CallNode *op,
                                size_t workspace_index) const {
    size_t workspace_count = 1;
    if (op->op.same_as(tl::ascend_reduce())) {
      workspace_count = ParseReduceCallLayout(op).tmp_count;
    }
    Array<PrimExpr> new_args;
    for (size_t i = 0; i < op->args.size(); ++i) {
      if (i < workspace_index || i >= workspace_index + workspace_count) {
        new_args.push_back(op->args[i]);
      }
    }
    return Call(op->dtype, op->op, new_args, op->span);
  }

  // Insert an expression at the specified position.
  Array<PrimExpr> InsertExprAt_(const Array<PrimExpr> &arr, size_t pos,
                                const PrimExpr &expr) {
    Array<PrimExpr> new_arr;

    for (size_t i = 0; i < pos && i < arr.size(); ++i) {
      new_arr.push_back(arr[i]);
    }

    new_arr.push_back(expr);

    for (size_t i = pos; i < arr.size(); ++i) {
      new_arr.push_back(arr[i]);
    }

    return new_arr;
  }

  PrimExpr AddTmpArgs_(const CallNode *op, const WorkspaceSpec &spec) {
    ICHECK_GT(spec.primary_bytes, 0)
        << "Expected a non-empty workspace view for " << GetOpName(op);
    const PrimExpr arena = MakeAccessPtrFromBuffer_(tmp_buf_, spec.access_mask);
    return MakeAccessPtrView(arena, 0, spec.primary_bytes, spec.view_dtype,
                             spec.access_mask);
  }

  PrimExpr MakeAccessPtrFromBuffer_(const Buffer &tmp_buffer, int64_t rw_mask) {
    ICHECK(tmp_buffer.defined()) << "Expected tmp buffer to be defined.";

    int64_t shape_size = 0;
    for (size_t j = 0; j < tmp_buffer.get()->shape.size(); j++) {
      if (shape_size == 0) {
        shape_size = tmp_buffer.get()->shape[j].as<IntImmNode>()->value;
      } else {
        shape_size *= tmp_buffer.get()->shape[j].as<IntImmNode>()->value;
      }
    }
    // Directly construct a CallNode for tvm_access_ptr
    Array<PrimExpr> args;
    args.push_back(TypeAnnotation(tmp_buffer.get()->dtype));
    args.push_back(tmp_buffer->data);
    args.push_back(Integer(0));
    args.push_back(Integer(shape_size));
    args.push_back(Integer(rw_mask));
    return Call(DataType::Handle(), builtin::tvm_access_ptr(), args);
  }

  bool NeedReduceOutputTmp(const CallNode *op) const {
    if (target_ != "pto" || !reduce_out_tmp_buf_.defined() ||
        !op->op.same_as(tl::ascend_reduce())) {
      return false;
    }
    const ReduceCallLayout layout = ParseReduceCallLayout(op);
    return layout.tmp_count == 0 && !layout.clear;
  }

  Buffer tmp_buf_;
  Buffer reduce_out_tmp_buf_;
  Array<Buffer> alloc_buffers_;
  std::string target_;
};

class RootAllocBufferFinder : public StmtVisitor {
public:
  static Array<Buffer> Find(const Stmt &stmt) {
    RootAllocBufferFinder finder;
    finder.VisitStmt(stmt);
    return finder.alloc_buffers_;
  }

private:
  void VisitStmt_(const BlockRealizeNode *node) override {
    if (node->block->name_hint == "tilelang_root") {
      ICHECK(!found_) << "Expected exactly one tilelang_root block.";
      alloc_buffers_ = node->block->alloc_buffers;
      found_ = true;
    }
    StmtVisitor::VisitStmt_(node);
  }

  bool found_{false};
  Array<Buffer> alloc_buffers_;
};

class TmpBufferInjector : public StmtExprMutator {
public:
  static PrimFunc TmpBufferInject(PrimFunc f, Target target) {
    TmpBufferInjector injector;
    injector.target_ = Downcast<String>(target.get()->attrs["model"]);
    injector.alloc_buffers_ = RootAllocBufferFinder::Find(f->body);
    PrimFuncNode *fptr = f.CopyOnWrite();
    injector.calls_ =
        CallNodeCollector::Collect(f, target, injector.alloc_buffers_);
    Stmt new_body = injector.inject(f->body);
    fptr->body = new_body;
    new_body = CallNodeModifier::Modify(f, target, injector.tmp_buf_,
                                        injector.reduce_out_tmp_buf_,
                                        injector.alloc_buffers_);
    fptr->body = new_body;
    return f;
  }

private:
  Stmt inject(const Stmt &stmt) { return VisitStmt(stmt); }

  Stmt VisitStmt_(const BlockRealizeNode *node) override {
    if (node->block->name_hint == "tilelang_root") {
      Block block = Downcast<Block>(node->block);
      BlockNode *op = block.CopyOnWrite();
      // Insert a tmp buffer into the alloc_buffers of the Block
      Array<Buffer> new_alloc_buffers = op->alloc_buffers;
      alloc_buffers_ = op->alloc_buffers;

      NameSupply name_supply("");
      for (const Buffer &buffer : op->alloc_buffers) {
        name_supply->ReserveName(buffer->name, /*add_prefix=*/false);
        name_supply->ReserveName(buffer->data->name_hint,
                                 /*add_prefix=*/false);
      }

      tmp_buf_ = createTmpBuffer_(op->alloc_buffers, name_supply);
      if (tmp_buf_.defined()) {
        new_alloc_buffers.push_back(tmp_buf_);
      }

      if ("pto" == target_) {
        reduce_out_tmp_buf_ = createPTOClearReduceOutputTmpBuffer_(
            op->alloc_buffers, name_supply);
        if (reduce_out_tmp_buf_.defined()) {
          new_alloc_buffers.push_back(reduce_out_tmp_buf_);
        }
      }

      // return new Block
      Block new_block = Block(
          op->iter_vars, op->reads, op->writes, op->name_hint, op->body,
          op->init, new_alloc_buffers, op->match_buffers, op->annotations);
      return BlockRealize(node->iter_values, node->predicate, new_block);
    }
    return StmtExprMutator::VisitStmt_(node);
  }

  Buffer createTmpBuffer_(Array<Buffer> alloc_buffers,
                          NameSupply &name_supply) {
    Array<PrimExpr> shape = GetTmpBufferSize_(alloc_buffers);

    if (shape.size() > 0) {
      const std::string buffer_name =
          name_supply->FreshName(buffer_name_, /*add_prefix=*/false);
      Var tmp_buf(buffer_name,
                  PointerType(PrimType(DataType::UInt(8)), "shared.ub"));
      Buffer buffer = Buffer(tmp_buf, DataType::UInt(8), shape, {}, PrimExpr(),
                             buffer_name, -1, 0, BufferType::kDefault);

      return buffer;
    } else {
      return Buffer();
    }
  }

  Buffer createPTOClearReduceOutputTmpBuffer_(Array<Buffer> alloc_buffers,
                                              NameSupply &name_supply) {
    int64_t shape_size = 0;
    for (size_t i = 0; i < calls_.size(); i++) {
      const CallNode *call = calls_[i].get();
      if (!call->op.same_as(tl::ascend_reduce())) {
        continue;
      }
      const ReduceCallLayout layout = ParseReduceCallLayout(call);
      if (!layout.clear) {
        shape_size =
            std::max(shape_size,
                     EstimateReduceOutputWorkspaceBytes(call, alloc_buffers));
      }
    }

    if (shape_size == 0) {
      return Buffer();
    }

    const std::string buffer_name = name_supply->FreshName(
        buffer_name_ + "_reduce_out", /*add_prefix=*/false);
    Var tmp_buf(buffer_name,
                PointerType(PrimType(DataType::UInt(8)), "shared.ub"));
    return Buffer(tmp_buf, DataType::UInt(8),
                  {IntImm(DataType::Int(32), shape_size)}, {}, PrimExpr(),
                  buffer_name, -1, 0, BufferType::kDefault);
  }

  Array<PrimExpr> GetTmpBufferSize_(Array<Buffer> alloc_buffers) {
    int64_t shape_size = 0;
    for (const Call &call : calls_) {
      const WorkspaceSpec spec =
          GetWorkspaceSpec(call.get(), alloc_buffers, target_);
      ICHECK(spec.requires_workspace);
      shape_size = std::max(shape_size, spec.primary_bytes);
    }
    return shape_size == 0
               ? Array<PrimExpr>{}
               : Array<PrimExpr>{IntImm(DataType::Int(32), shape_size)};
  }

  std::string target_;
  std::vector<Call> calls_;
  const std::string buffer_name_ = "tmp_ub";
  Buffer tmp_buf_;
  Buffer reduce_out_tmp_buf_;
  Array<Buffer> alloc_buffers_;
};

tvm::transform::Pass InjectTmpBuffer(Target target) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return TmpBufferInjector::TmpBufferInject(std::move(f), target);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectTmpBuffer", {});
}

TVM_REGISTER_GLOBAL("tl.transform.InjectTmpBuffer")
    .set_body_typed(InjectTmpBuffer);

} // namespace tl
} // namespace tvm
