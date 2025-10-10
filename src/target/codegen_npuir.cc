// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file target/codegen.cc
 */

#include "codegen_npuir.h"
#include <cassert>
#include <elf.h>
#include <memory>
#include <ostream>
#include <sstream>
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "../op/builtin.h"
#include "../op/ascend.h"

#include "arith/pattern_match.h"
#include "tvm/ir/expr.h"
#include "tvm/runtime/container/array.h"
#include "tvm/runtime/data_type.h"
#include "tvm/tir/buffer.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/stmt.h"

namespace tvm {
namespace codegen {

static std::map<NPU_CORETYPE, std::string> NPU_CORETYPE_STR{
    {NPU_CORETYPE::AIC, "AIC"},
    {NPU_CORETYPE::AIV, "AIV"},
    {NPU_CORETYPE::MIX, "MIX"}};

static std::map<NPU_CORETYPE, std::string> NPU_CORETYPE_str{
    {NPU_CORETYPE::AIC, "aic"},
    {NPU_CORETYPE::AIV, "aiv"},
    {NPU_CORETYPE::MIX, "mix"}};

static std::map<NPU_CORETYPE, std::string> coretype_syncblock_map{
    {NPU_CORETYPE::AIC, "CUBE"}, {NPU_CORETYPE::AIV, "VECTOR"}};

static std::map<int, std::string> fixpipe_pre_relu_mode{
    {0, "NO_RELU"}, {1, "NORMAL_RELU"}, {2, "LEAKY_RELU"}, {3, "P_RELU"}};

static std::map<tl::SyncBlockMode, std::string> SyncBlockMode_str{
    {tl::SyncBlockMode::INTER_BLOCK, "INTER_BLOCK_SYNCHRONIZATION"},
    {tl::SyncBlockMode::INTER_SUBBLOCK, "INTER_SUBBLOCK_SYNCHRONIZATION"},
    {tl::SyncBlockMode::INTRA_BLOCK, "INTRA_BLOCK_SYNCHRONIZATION"}};

constexpr uint8_t FLAG_ID_BITS = 64;

template <typename T>
inline void PrintBinary(const T *op, const char *opstr, std::ostream &os,
                        CodeGenC *CG) {
  auto PrintOp = [op, &os, CG](auto Operand) {
    std::ostringstream tmpos;
    if (Operand.template as<tvm::tir::IntImmNode>() ||
        Operand.template as<tvm::tir::VarNode>()) {
      CG->PrintExpr(Operand, tmpos << "%");
    } else if (auto *float_imm =
                   Operand.template as<tvm::tir::FloatImmNode>()) {
      tmpos << "Invalid float scalar operation"
            << "\n";
    } else {
      // TODO: codegen expr as a seperate instruction in mlir
      tmpos << "<<<expr:%" << Operand << ">>>";
    }
    return tmpos.str();
  };
  if (op->dtype.lanes() == 1) {
    // left op
    os << "arith." << opstr << " ";
    os << PrintOp(op->a);
    os << ", ";
    // right op
    os << PrintOp(op->b);
    os << " : ";
    CG->PrintType(op->a->dtype, os);
  } else {
    os << "<<<invalid-op-dtype-lanes-not-one: %" << opstr << ">>>\n";
  }
}

String GetAddressSpace(String address_space) {
  if (address_space == "global")
    return "gm";
  else if (address_space == "shared")
    return "ub";
  else if (address_space == "shared.dyn")
    return "cbuf";
  else if (address_space == "wmma.accumulator")
    return "cc";
  return "unknown";
}

bool IsEqual(Array<PrimExpr> a, Array<PrimExpr> b) {
  if (a.size() != b.size())
    return false;
  for (int i = 0; i < a.size(); i++) {
    if (!(a[i].same_as(b[i])))
      return false;
  }
  return true;
}

bool AllZero(Array<PrimExpr> a) {
  for (PrimExpr pe : a) {
    if (!is_zero(pe))
      return false;
  }
  return true;
}

std::vector<unsigned long> GetStrideFromShape(Array<tvm::PrimExpr> shape) {
  std::vector<unsigned long> strides;
  unsigned long total_size = 1;
  std::vector<int> shape_int;
  for (PrimExpr s : shape) {
    if (auto s_int = as_const_int(s)) {
      total_size *= *s_int;
      shape_int.push_back(*s_int);
    }
  }
  for (int i = 0; i < shape.size(); i++) {
    total_size /= shape_int[i];
    strides.push_back(total_size);
  }
  return strides;
}

// get string formate of buffer stride
// If the buffer stride is empty, indicating buffer is contiguous
String GetBufferStrides(Buffer buffer) {
  Array<PrimExpr> shape = buffer->shape;
  std::vector<unsigned long> strides;
  int dim = buffer->shape.size();
  if (buffer->strides.empty()) {
    strides = GetStrideFromShape(shape);
  } else {
    for (PrimExpr stride : buffer->strides) {
      if (auto stride_int = as_const_int(stride)) {
        strides.push_back(*stride_int);
      }
    }
  }
  String res = "[";
  for (int i = 0; i < dim; i++) {
    if (i > 0)
      res = res + ", ";
    res = res + std::to_string(strides[i]);
  }
  res = res + "]";
  return res;
}

static std::vector<int> getBroadcastDim(Array<PrimExpr> &buffer_shape0,
                                        Array<PrimExpr> &buffer_shape1) {
  assert(buffer_shape0.size() == buffer_shape1.size());
  std::vector<int> dims;
  for (int i = 0; i < buffer_shape0.size(); i++) {
    if (*as_const_int(buffer_shape0[i]) == 1 &&
        *as_const_int(buffer_shape1[i]) != 1) {
      dims.emplace_back(i);
    }
    if (*as_const_int(buffer_shape0[i]) != 1 &&
        *as_const_int(buffer_shape1[i]) == 1) {
      dims.emplace_back(i);
    }
    assert(*as_const_int(buffer_shape0[i]) == *as_const_int(buffer_shape1[i]));
  }
  return dims;
}

static std::string broadcastAttrCodegen(Array<PrimExpr> &buffer_shape0,
                                        Array<PrimExpr> &buffer_shape1) {
  if (buffer_shape0.empty() || buffer_shape1.empty()) {
    return "";
  }
  std::vector<int> broadcastDims;
  if (buffer_shape0.size() && buffer_shape1.size()) {
    broadcastDims = getBroadcastDim(buffer_shape0, buffer_shape1);
  }
  std::ostringstream temp;
  if (broadcastDims.size()) {
    temp << " = [";
    for (auto dim : broadcastDims) {
      temp << dim;
      if (dim != broadcastDims.back()) {
        temp << ", ";
      }
    }
    temp << "]";
  }
  return temp.str();
}

template <typename T>
std::string CodeGenTileLangNPUIR::ScalarConvertType(T *imm,
                                                    DataType targetDtype) {
  auto castNode = std::make_unique<tir::Cast>(targetDtype, *imm);
  std::string castId = SSAGetID(PrintExpr(*castNode), targetDtype);
  return castId;
}

CodeGenTileLangNPUIR::CodeGenTileLangNPUIR() {
  //  restrict_keyword_ = "GM_ADDR";
}

void CodeGenTileLangNPUIR::PrintFuncPrefix(std::ostream &os) {
  //  os << "CATLASS_GLOBAL\n";
}

std::string CodeGenTileLangNPUIR::Finish() {
  // decl_stream << "#include \"common.h\"\n";
  // decl_stream << "#include \"acl/acl.h\"\n";
  // decl_stream << "using namespace Catlass;\n";
  // decl_stream << "\n";
  std::ostringstream code;
  code << decl_stream.str();
  code << stream.str();
  return code.str();
}

void CodeGenTileLangNPUIR::VisitStmt_(const tir::ForNode *op) {
  auto flush = false;
  if (flush_out_) {
    flush = true;
    flush_out_ = false;
  }
  // TODO: Do we need add unroll attribute?
  // if (op->kind == tir::ForKind::kUnrolled) {
  //   PrintIndent();
  //   stream << "#pragma unroll\n";
  // }
  std::string upperBoundId =
      SSAGetID(PrintExpr(arith::Analyzer().Simplify(op->extent + op->min)),
               op->extent->dtype);
  assert(op->extent.dtype().is_int() || op->extent.dtype().is_uint());
  assert(op->min.dtype() == op->extent.dtype());
  std::string vid =
      SSAGetID(AllocVarID(op->loop_var.get()), op->loop_var->dtype);
  std::string lowerBoundId = SSAGetID(PrintExpr(op->min), op->min->dtype);
  std::string extentId = SSAGetID(PrintExpr(op->extent), op->extent->dtype);
  auto stepNode = std::make_unique<IntImm>(op->min.dtype(), 1);
  auto stepId = SSAGetID(PrintExpr(*stepNode), stepNode->dtype());
  PrintIndent();
  stream << "scf.for %" << vid << " = %" << lowerBoundId << " to %"
         << upperBoundId << " step %" << stepId << " : ";
  PrintType(op->min.dtype(), stream);
  stream << " {\n";
  int for_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
}

void CodeGenTileLangNPUIR::VisitStmt_(const tir::IfThenElseNode *op) {
  std::string cond = SSAGetID(PrintExpr(op->condition), op->condition->dtype);
  PrintIndent();
  stream << "scf.if %" << cond << " {\n";
  int then_scope = BeginScope();
  PrintStmt(op->then_case);
  this->EndScope(then_scope);
  if (op->else_case) {
    PrintIndent();
    stream << "} else {\n";
    int else_scope = BeginScope();
    PrintStmt(op->else_case.value());
    this->EndScope(else_scope);
  }
  PrintIndent();
  stream << "}\n";
}

void CodeGenTileLangNPUIR::PrintSSAAssign(const std::string &target,
                                          const std::string &src, DataType t) {
  stream << "%" << target << " = " << src << "\n";
}

void CodeGenTileLangNPUIR::PrintShape(Array<PrimExpr> shape,
                                      std::string delimiter, std::ostream &os) {
  for (size_t i = 0; i < shape.size(); i++) {
    if (i != 0)
      os << delimiter;
    os << shape[i];
  }
}

void CodeGenTileLangNPUIR::PrintType(DataType t,
                                     std::ostream &os) { // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    // ICHECK(t.is_scalar()) << "do not yet support vector types";
    // os << "void*";
    return;
  }

  if (t.is_void()) {
    //    os << "void";
    return;
  }

  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
    case 16:
      enable_fp16_ = true;
      if (t.is_scalar()) {
        os << "f16";
      } else {
        fail = true;
      }
      break;
    case 32:
      os << "f32";
      break;
    case 64:
      os << "f64";
      break;
    default:
      fail = true;
      break;
    }
    if (!fail && (t.is_scalar() || t.bits() == 16))
      return;
  } else if (t.is_bfloat16()) {
    enable_bf16_ = true;
    if (t.is_scalar()) {
      os << "bf16";
    } else {
      fail = true;
    }
    if (!fail)
      return;
  } else if (t == DataType::Bool()) {
    os << "i1";
    return;
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << "u";
    }
    switch (t.bits()) {
    case 1: {
      if (t.is_scalar()) {
        os << "i1";
        return;
      } else {
        LOG(FATAL) << "Cannot convert type " << t;
      }
    }
    case 4: {
      if (t.is_scalar()) {
        os << "i4";
        return;
      } else {
        LOG(FATAL) << "Cannot convert type " << t;
      }
    }
    case 8: {
      if (t.is_scalar()) {
        os << "i8";
        return;
      } else {
        LOG(FATAL) << "Cannot convert type " << t;
      }
    }
    case 16: {
      if (t.is_scalar()) {
        os << "i16";
      } else {
        fail = true;
      }
      if (!fail) {
        return;
      }
      break;
    }
    case 32: {
      if (t.is_scalar()) {
        os << "i32";
      } else {
        fail = true;
      }
      if (!fail) {
        return;
      }
      break;
    }
    case 64: {
      if (t.is_scalar()) {
        os << "i64";
      }
      return;
    }
    default:
      fail = true;
      break;
    }
    if (!fail) {
      return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t;
}

void CodeGenTileLangNPUIR::PrintStorageScope(const std::string &scope,
                                             std::ostream &os) { // NOLINT(*)
}

void CodeGenTileLangNPUIR::VisitExpr_(const FloorDivNode *op,
                                      std::ostream &os) {
  // FIXME: The floor div in python is not the same as arith.divsi in negative
  // scenarios.
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    PrintBinary(op, "divsi", os, this);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "divf", os, this);
  }
}

void CodeGenTileLangNPUIR::VisitExpr_(const FloorModNode *op,
                                      std::ostream &os) {
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    PrintBinary(op, "remsi", os, this);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "remf", os, this);
  }
}

void CodeGenTileLangNPUIR::VisitExpr_(const LTNode *op, std::ostream &os) {
  if (op->a->dtype.is_int()) {
    PrintBinary(op, "cmpi slt,", os, this);
  } else if (op->a->dtype.is_uint()) {
    PrintBinary(op, "cmpi ult,", os, this);
  } else {
    PrintBinary(op, "cmpf olt,", os, this);
  }
}

void CodeGenTileLangNPUIR::VisitExpr_(const NENode *op, std::ostream &os) {
  if (op->a->dtype.is_int() || op->a->dtype.is_uint()) {
    PrintBinary(op, "cmpi ne,", os, this);
  } else {
    PrintBinary(op, "cmpf one,", os, this);
  }
}

void CodeGenTileLangNPUIR::VisitExpr_(const EQNode *op, std::ostream &os) {
  if (op->a->dtype.is_int() || op->a->dtype.is_uint()) {
    PrintBinary(op, "cmpi eq,", os, this);
  } else {
    PrintBinary(op, "cmpf oeq,", os, this);
  }
}

void CodeGenTileLangNPUIR::VisitExpr_(const LENode *op, std::ostream &os) {
  if (op->a->dtype.is_int()) {
    PrintBinary(op, "cmpi sle,", os, this);
  } else if (op->a->dtype.is_uint()) {
    PrintBinary(op, "cmpi ule,", os, this);
  } else {
    PrintBinary(op, "cmpf ole,", os, this);
  }
}

void CodeGenTileLangNPUIR::VisitExpr_(const GENode *op, std::ostream &os) {
  if (op->a->dtype.is_int()) {
    PrintBinary(op, "cmpi sge,", os, this);
  } else if (op->a->dtype.is_uint()) {
    PrintBinary(op, "cmpi uge,", os, this);
  } else {
    PrintBinary(op, "cmpf oge,", os, this);
  }
}

void CodeGenTileLangNPUIR::VisitExpr_(const GTNode *op, std::ostream &os) {
  if (op->a->dtype.is_int()) {
    PrintBinary(op, "cmpi sgt,", os, this);
  } else if (op->a->dtype.is_uint()) {
    PrintBinary(op, "cmpi ugt,", os, this);
  } else {
    PrintBinary(op, "cmpf ogt,", os, this);
  }
}

void CodeGenTileLangNPUIR::VisitExpr_(const CastNode *op, std::ostream &os) {
  bool srcIsFloat =
      op->value->dtype.is_float() || op->value->dtype.is_bfloat16();
  bool srcIsInt = op->value->dtype.is_int();
  bool srcIsUInt = op->value->dtype.is_uint();
  bool targetIsFloat = op->dtype.is_float() || op->dtype.is_bfloat16();
  bool targetIsInt = op->dtype.is_int();
  bool targetIsUInt = op->dtype.is_uint();
  auto val = PrintExpr(op->value);
  if (srcIsFloat && targetIsInt) {
    os << "arith.fptosi \%" << val << " : ";
  } else if (srcIsFloat && targetIsUInt) {
    os << "arith.fptoui \%" << val << " : ";
  } else if (srcIsInt && targetIsFloat) {
    os << "arith.sitofp \%" << val << " : ";
  } else if (srcIsUInt && targetIsFloat) {
    os << "arith.uitofp \%" << val << " : ";
  } else if (targetIsInt) {
    if (op->dtype.bits() > op->value->dtype.bits()) {
      os << "arith.extsi \%" << val << " : ";
    } else {
      os << "arith.trunci \%" << val << " : ";
    }
  } else if (targetIsUInt) {
    if (op->dtype.bits() > op->value->dtype.bits()) {
      os << "arith.extui \%" << val << " : ";
    } else {
      os << "arith.trunci \%" << val << " : ";
    }
  } else if (targetIsFloat) {
    if (op->dtype.bits() > op->value->dtype.bits()) {
      os << "arith.extf \%" << val << " : ";
    } else {
      os << "arith.truncf \%" << val << " : ";
    }
  }
  PrintType(op->value->dtype, os);
  os << " to ";
  PrintType(op->dtype, os);
}

Array<String> CodeGenTileLangNPUIR::GenConvertIndex(Array<PrimExpr> exprs) {
  Array<String> cast_array;
  for (PrimExpr curr_expr : exprs) {
    if (auto curr_expr_int = curr_expr.as<IntImmNode>()) {
      cast_array.push_back(std::to_string(curr_expr_int->value));
    } else {
      DataType indice_type = curr_expr->dtype;
      std::ostringstream temp;
      std::string var_name = PrintExpr(curr_expr);
      if (!curr_expr.as<VarNode>()) {
        var_name = SSAGetID(var_name, indice_type);
      }
      temp << "arith.index_cast \%" << var_name << " : ";
      PrintType(indice_type, temp);
      temp << " to index";
      String cast_indice_name = "\%" + SSAGetID(temp.str(), indice_type);
      cast_array.push_back(cast_indice_name);
    }
  }
  return cast_array;
}

unsigned long ComputeOffset(Memref *src_buffer, Array<PrimExpr> op_offset) {
  if (src_buffer->var_offset)
    return -1;
  if (src_buffer->stride_int.size() != src_buffer->dim)
    return -1;
  unsigned long offset = src_buffer->offset;
  for (int i = 0; i < src_buffer->dim; i++) {
    const int64_t *op_off = as_const_int(op_offset[i]);
    if (op_off == nullptr)
      return -1;
    offset += (*op_off) * src_buffer->stride_int[i];
  }
  return offset;
}

String CodeGenTileLangNPUIR::GenSubviewFromRegion(const CallNode *region_node) {
  tvm::tl::RegionOp regionop(region_node->args, this->vmap);
  return GenSubviewFromRegion(regionop.GetBuffer(), regionop.GetRanges());
}

String CodeGenTileLangNPUIR::GenSubviewFromRegion(Buffer buffer_data,
                                                  Array<Range> range) {
  std::ostringstream temp;
  DataType buffer_type = buffer_data->dtype;
  String buffer_name = buffer_data->name;
  Array<PrimExpr> buffer_shape = buffer_data->shape;
  int dim = buffer_shape.size();
  Array<PrimExpr> region_shape, region_indeces;
  for (Range r : range) {
    region_shape.push_back(r.get()->extent);
    region_indeces.push_back(r.get()->min);
  }
  String buffer_name_val = "";
  if (auto memrefInfo = dynamic_cast<Memref *>(type_info[buffer_name])) {
    if (memrefInfo->is_arg) {
      buffer_name_val = buffer_name + "_Recast";
    } else {
      buffer_name_val = buffer_name;
    }
  } else {
    LOG(FATAL) << buffer_name << " should be a memref";
  }
  String new_buffer_name = buffer_name_val;
  String src_data_info = GetMemrefInfo(buffer_name_val);
  if (!(IsEqual(buffer_shape, region_shape) && AllZero(region_indeces))) {
    // get Indice name from BufferLoadNode, add type cast to index if needed.
    Array<String> cast_offset_array = GenConvertIndex(region_indeces);
    Array<String> cast_shape_array = GenConvertIndex(region_shape);
    // add new memref obj
    if (!dynamic_cast<Memref *>(type_info[buffer_name_val])) {
      LOG(FATAL) << buffer_name_val << " should be a memref";
    }
    unsigned long offset = ComputeOffset(
        dynamic_cast<Memref *>(type_info[buffer_name_val]), region_indeces);
    // gen subview
    new_buffer_name = buffer_name_val + "_subview";
    auto tempMemref = new Memref(
        new_buffer_name, region_shape, buffer_type,
        dynamic_cast<Memref *>(type_info[buffer_name_val])->address_space,
        offset == -1,
        dynamic_cast<Memref *>(type_info[buffer_name_val])->stride, offset);
    String dst_data_info = GetMemrefInfo(tempMemref);
    temp << "memref.subview \%" + buffer_name_val;
    // gen offset
    temp << "[";
    for (int i = 0; i < dim; i++) {
      if (i > 0) {
        temp << ", ";
      }
      temp << cast_offset_array[i];
    }
    temp << "]";
    // gen shape
    temp << "[";
    for (int i = 0; i < dim; i++) {
      if (i > 0) {
        temp << ", ";
      }
      temp << cast_shape_array[i];
    }
    temp << "]";
    // gen stride
    // Assuming the stride remains unchanged in region, set all 1.
    temp << "[";
    for (int i = 0; i < dim; i++) {
      if (i > 0) {
        temp << ", ";
      }
      temp << "1";
    }
    temp << "]";
    temp << " : ";
    temp << src_data_info;
    temp << " to ";
    temp << dst_data_info;
    delete tempMemref;
    new_buffer_name = SSAGetID(temp.str(), buffer_type);
    this->type_info[new_buffer_name] = new Memref(
        new_buffer_name, region_shape, buffer_type,
        dynamic_cast<Memref *>(type_info[buffer_name_val])->address_space,
        offset == -1,
        dynamic_cast<Memref *>(type_info[buffer_name_val])->stride, offset);
    GetMemrefInfo(new_buffer_name);
  }
  return new_buffer_name;
}

/// Generate hivm.hir.load or hivm.hir.store for tl.ascend_copy.
/// before:
///   T.ascend_copy(T.region(A[bx, by], 1, 128, 256), T.region(A_VEC[0, 0],
///   2, 128, 256))
/// after:
///   memref.reinterpret_cast; memref.subview; memref.subview;
///   hivm.hir.store/load
void CodeGenTileLangNPUIR::AscendCopyCodegen(const CallNode *op,
                                             std::ostream &os) {
  tvm::tl::AscendCopy npuirop(op->args, this->vmap);
  // gen memref.subview
  String src_data_name = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  String dst_data_name = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);

  // gen hivm.ir.load / hivm.ir.store
  this->PrintIndent();
  if (!dynamic_cast<Memref *>(type_info[src_data_name])) {
    LOG(FATAL) << src_data_name << " should be a memref";
  }
  if (!dynamic_cast<Memref *>(type_info[dst_data_name])) {
    LOG(FATAL) << dst_data_name << " should be a memref";
  }
  if (dynamic_cast<Memref *>(type_info[src_data_name])->address_space == "gm") {
    this->stream << "hivm.hir.load";
  } else if (dynamic_cast<Memref *>(type_info[dst_data_name])->address_space ==
             "gm") {
    this->stream << "hivm.hir.store";
  } else if (dynamic_cast<Memref *>(type_info[src_data_name])->address_space ==
                 "ub" &&
             dynamic_cast<Memref *>(type_info[dst_data_name])->address_space ==
                 "ub") {
    this->stream << "hivm.hir.copy";
  }
  this->stream << " ins("
               << "\%" << src_data_name << " : " << GetMemrefInfo(src_data_name)
               << ")";
  this->stream << " outs("
               << "\%" << dst_data_name << " : " << GetMemrefInfo(dst_data_name)
               << ")";
  // TODO: error: custom op 'init_out_buffer' is unknown
  // this->stream << " left_padding_num = %c0 : index init_out_buffer =
  // false";
  this->stream << "\n";
}

template <typename T>
void CodeGenTileLangNPUIR::UnaryVecOpCodegen(const CallNode *op,
                                             std::ostream &os) {
  // Generate hivm.hir.vexp for tl.npuir_exp.
  // before:
  //   T.npuir_exp(T.region(A_VEC[0], 1, 1024), T.region(B_VEC[0], 2, 1024))
  // after:
  //   hivm.hir.vexp ins(%9 : memref<1024xf32, #hivm.address_space<ub>>)
  //       outs(%16 : memref<1024xf32, #hivm.address_space<ub>>)
  T npuirop(op->args, this->vmap);
  std::string in_data_name = "", out_data_name = "";
  Array<PrimExpr> buffer_shape0 = npuirop.src->shape,
                  buffer_shape1 = npuirop.dst->shape;
  in_data_name = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  out_data_name = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  this->PrintIndent();
  this->stream << npuirop.op_name << " ins(\%" << in_data_name << " : "
               << this->type_info[in_data_name]->printType() << ") outs(\%"
               << out_data_name << " : "
               << this->type_info[out_data_name]->printType() << ")";
  auto dims = broadcastAttrCodegen(buffer_shape0, buffer_shape1);
  if (dims != "") {
    this->stream << " broadcast" << dims;
  }
  this->stream << "\n";
}

void CodeGenTileLangNPUIR::VselectCodegen(const CallNode *op,
                                          std::ostream &os) {
  /// Generate hivm.hir.vsel for tl.npuir_select.
  /// before:
  ///   T.npuir_select(Cond_VEC, A_VEC, B_VEC, C_VEC)
  /// after:
  ///   hivm.hir.vsel ins(%v__9, %A_VEC, %B_VEC : memref<32x64xi1, strided<[64,
  ///   1], offset:0>, #hivm.address_space<ub>>, memref<32x64xf16, strided<[64,
  ///   1], offset:0>, #hivm.address_space<ub>>, memref<32x64xf16, strided<[64,
  ///   1], offset:0>, #hivm.address_space<ub>>) outs(%C_VEC : memref<32x64xf16,
  ///   strided<[64, 1], offset:0>, #hivm.address_space<ub>>)
  tvm::tl::NpuirSelect npuirop(op->args, this->vmap);
  // gen memref.subview
  String cond_data_name =
      GenSubviewFromRegion(npuirop.cond, npuirop.cond_range);
  String src0_data_name =
      GenSubviewFromRegion(npuirop.src0, npuirop.src0_range);
  String src1_data_name =
      GenSubviewFromRegion(npuirop.src1, npuirop.src1_range);
  String dst_data_name = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);

  // gen hivm.hir.nd2nz
  this->PrintIndent();
  this->stream << "hivm.hir.vsel";
  this->stream << " ins(%" << cond_data_name << ", %" << src0_data_name << ", %"
               << src1_data_name << " : ";
  this->stream << GetMemrefInfo(cond_data_name) << ", "
               << GetMemrefInfo(src0_data_name) << ", "
               << GetMemrefInfo(src1_data_name) << ")";
  this->stream << " outs(%" << dst_data_name << " : "
               << GetMemrefInfo(dst_data_name) << ")";
  this->stream << "\n";
}

void CodeGenTileLangNPUIR::VcmpCodegen(const CallNode *op, std::ostream &os) {
  /// Generate hivm.hir.vcmp for tl.npuir_cmp.
  /// before:
  ///   T.npuir_cmp(A_VEC, B_VEC, Cond_VEC, "ge")
  /// after:
  ///   hivm.hir.vcmp ins(%A_VEC, %B_VEC : memref<32x64xf16, strided<[64, 1],
  ///   offset:0>, #hivm.address_space<ub>>, memref<32x64xf16, strided<[64, 1],
  ///   offset:0>, #hivm.address_space<ub>>) outs(%Cond_VEC : memref<32x64xi1,
  ///   strided<[64, 1], offset:0>, #hivm.address_space<ub>>) compare_mode =
  ///   <ge>
  BinaryVecOpCodegen(op, "cmp", os);
  this->stream.seekp(-1, std::ios_base::end);
  this->stream << " compare_mode = <"
               << op->args[3].as<StringImm>().value()->value << ">\n";
}

void CodeGenTileLangNPUIR::VbrcCodegen(const CallNode *op, std::ostream &os) {
  // Generate hivm.hir.vbrc for tl.npuir_vbrc.
  tvm::tl::NpuirBrc npuirop(op->args, this->vmap);
  std::string in_data_name = "", out_data_name = "";
  DataType outDtype = npuirop.dst->dtype;
  Array<PrimExpr> buffer_shape0, buffer_shape1;
  if (auto intImm = npuirop.in.as<IntImm>()) {
    auto immObj = intImm.value();
    in_data_name = PrintExpr(immObj);
    if (intImm.value()->dtype != outDtype) {
      in_data_name = ScalarConvertType(&immObj, outDtype);
    }
    std::ostringstream temp;
    PrintType(outDtype, temp);
    this->type_info[in_data_name] = new Scalar(in_data_name, temp.str());
  } else if (auto floatImm = npuirop.in.as<FloatImm>()) {
    auto immObj = floatImm.value();
    in_data_name = PrintExpr(immObj);
    if (floatImm.value()->dtype != outDtype) {
      in_data_name = ScalarConvertType(&immObj, outDtype);
    }
    std::ostringstream temp;
    PrintType(outDtype, temp);
    this->type_info[in_data_name] = new Scalar(in_data_name, temp.str());
  } else {
    buffer_shape0 = npuirop.src->shape;
    in_data_name = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  }
  buffer_shape1 = npuirop.dst->shape;
  out_data_name = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  this->PrintIndent();
  this->stream << "hivm.hir.vbrc ins(\%" << in_data_name << " : "
               << this->type_info[in_data_name]->printType() << ") outs(\%"
               << out_data_name << " : "
               << this->type_info[out_data_name]->printType() << ")";
  auto dims = broadcastAttrCodegen(buffer_shape0, buffer_shape1);
  if (dims != "") {
    this->stream << " broadcast_dims" << dims;
  }
  this->stream << "\n";
}

void CodeGenTileLangNPUIR::VcastCodegen(const CallNode *op, std::ostream &os) {
  // Generate hivm.hir.vcast for tl.npuir_cast.
  tvm::tl::NpuirCast npuirop(op->args, this->vmap);
  std::string in_data_name = "", out_data_name = "";
  Array<PrimExpr> buffer_shape0 = npuirop.src->shape,
                  buffer_shape1 = npuirop.dst->shape;
  in_data_name = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  out_data_name = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  std::string round_mode = op->args[2].as<StringImmNode>()->value;
  this->PrintIndent();
  this->stream << "hivm.hir.vcast ins(\%" << in_data_name << " : "
               << this->type_info[in_data_name]->printType() << ") outs(\%"
               << out_data_name << " : "
               << this->type_info[out_data_name]->printType() << ")";
  this->stream << " round_mode = <" << round_mode << ">";
  auto dims = broadcastAttrCodegen(buffer_shape0, buffer_shape1);
  if (dims != "") {
    this->stream << " broadcast" << dims;
  }
  this->stream << "\n";
}

void CodeGenTileLangNPUIR::VreduceCodegen(const CallNode *op,
                                          std::ostream &os) {
  // Generate hivm.hir.vreduce for tl.npuir_reduce.
  tvm::tl::NpuirReduce npuirop(op->args, this->vmap);
  std::string in_data_name = "", out_data_name = "";
  Array<PrimExpr> buffer_shape0 = npuirop.src->shape,
                  buffer_shape1 = npuirop.dst->shape;
  in_data_name = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  out_data_name = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  std::vector<int> reduceDims;
  reduceDims = npuirop.reduce_dims;
  std::string reduce_mode = npuirop.reduce_mode;
  this->PrintIndent();
  this->stream << "hivm.hir.vreduce <" << reduce_mode << "> ins(\%"
               << in_data_name << " : "
               << this->type_info[in_data_name]->printType() << ") outs(\%"
               << out_data_name << " : "
               << this->type_info[out_data_name]->printType() << ")";
  this->stream << " reduce_dims = [";
  for (auto dim : reduceDims) {
    this->stream << dim;
    if (dim != reduceDims.back()) {
      this->stream << ", ";
    }
  }
  this->stream << "]";
  this->stream << "\n";
}

void CodeGenTileLangNPUIR::Nd2NzCodegen(const CallNode *op, std::ostream &os) {
  // Generate hivm.hir.nd2nz for tl.npuir_load_nd2nz.
  tvm::tl::NpuirNd2nz npuirop(op->args, this->vmap);
  // gen memref.subview
  String src_data_name = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  String dst_data_name = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  bool dst_continuous = npuirop.dst_continuous;

  // gen hivm.hir.nd2nz
  this->PrintIndent();
  this->stream << "hivm.hir.nd2nz";
  if (dst_continuous)
    this->stream << " {dst_continuous}";
  this->stream << " ins(%" << src_data_name << " : "
               << GetMemrefInfo(src_data_name) << ")";
  this->stream << " outs(%" << dst_data_name << " : "
               << GetMemrefInfo(dst_data_name) << ")";
  this->stream << "\n";
}

void CodeGenTileLangNPUIR::FixpipeCodegen(const CallNode *op,
                                          std::ostream &os) {
  // Generate hivm.hir.fixpipe for tl.npuir_store_fixpipe.
  tvm::tl::NpuirFixpipe npuirop(op->args, this->vmap);
  // gen memref.subview
  String src_data_name = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  String dst_data_name = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  bool enable_nz2nd = npuirop.enable_nz2nd;
  bool channel_split = npuirop.channel_split;
  int pre_relu_mode = npuirop.pre_relu_mode;

  // gen hivm.hir.fixpipe
  this->PrintIndent();

  auto src_dtype = npuirop.src->dtype;
  auto dst_dtype = npuirop.dst->dtype;
  std::string pre_quant_attr = "";
  if (src_dtype != dst_dtype) {
    if (src_dtype == DataType::Float(32) && dst_dtype == DataType::Float(16)) {
      pre_quant_attr = "F322F16";
    } else if (src_dtype == DataType::Float(32) &&
               dst_dtype == DataType::BFloat(16)) {
      pre_quant_attr = "F322BF16";
    } else if (src_dtype == DataType::Int(32) &&
               dst_dtype == DataType::Int(8)) {
      pre_quant_attr = "S322I8";
    } else {
      assert(false && "Unexpected pre-quant mode. Should not reach here.");
    }
  }

  this->stream << "hivm.hir.fixpipe {";
  this->stream << "channel_split = " << (channel_split ? "true" : "false");
  this->stream << (enable_nz2nd ? ", enable_nz2nd" : "");
  this->stream << ", pre_relu = #hivm.fixpipe_pre_relu_mode<"
               << fixpipe_pre_relu_mode[pre_relu_mode] << ">";
  if (pre_quant_attr != "") {
    this->stream << ", pre_quant = #hivm.fixpipe_pre_quant_mode<"
                 << pre_quant_attr << ">";
  }
  this->stream << "}";
  this->stream << " ins(%" << src_data_name << " : "
               << GetMemrefInfo(src_data_name) << ")";
  this->stream << " outs(%" << dst_data_name << " : "
               << GetMemrefInfo(dst_data_name) << ")";
  this->stream << "\n";
}

void CodeGenTileLangNPUIR::BarrierCodegen(const CallNode *op,
                                          std::ostream &os) {
  tvm::tl::NpuirPipeBarrier npuirop(op->args, this->vmap);
  this->PrintIndent();
  this->stream << "hivm.hir.pipe_barrier[<";
  this->stream << npuirop.pipe_type;
  this->stream << ">]\n";
}

std::string CodeGenTileLangNPUIR::PrintID(PrimExpr id) {
  auto raw_type = id.dtype();
  auto id_name = SSAGetID(PrintExpr(id), raw_type);
  assert(raw_type.is_int() || raw_type.is_uint());
  if (raw_type.bits() < FLAG_ID_BITS) {
    std::ostringstream temp;
    temp << "arith.";
    if (raw_type.is_int()) {
      temp << "extsi %";
    } else {
      temp << "extui %";
    }
    temp << id_name << " : ";
    PrintType(raw_type, temp);
    temp << " to i64\n";
    id_name = SSAGetID(temp.str(), DataType::Int(FLAG_ID_BITS));
  }
  return "%" + id_name;
}

template <typename T>
void CodeGenTileLangNPUIR::PipeFlagCodegen(const T &sync_op, std::ostream &os) {
  std::string event_id;
  if (auto *int_imm = sync_op.event_id.template as<tvm::tir::IntImmNode>()) {
    event_id = "<EVENT_ID" + std::to_string(int_imm->value) + ">";
  } else {
    event_id = PrintID(sync_op.event_id);
  }

  this->PrintIndent();
  this->stream << "hivm.hir.";
  this->stream << T::op;
  this->stream << "_flag[<" << sync_op.pipe1 << ">, <" << sync_op.pipe2 << ">, "
               << event_id << "]\n";
}

template <typename T>
void CodeGenTileLangNPUIR::SyncBlockWaitCodegen(const T &sync_op,
                                                std::ostream &os) {
  std::string flag_id;
  if (auto *int_imm = sync_op.flag_id.template as<tvm::tir::IntImmNode>()) {
    flag_id = std::to_string(int_imm->value);
  } else {
    flag_id = PrintID(sync_op.flag_id);
  }
  this->PrintIndent();
  this->stream << "hivm.hir.sync_block_wait[<";
  this->stream << coretype_syncblock_map[current_coretype];
  this->stream << ">, <PIPE_S>, <" << sync_op.pipe_type
               << ">] flag = " << flag_id << "\n";
}

template <typename T>
void CodeGenTileLangNPUIR::SyncBlockSetCodegen(const T &sync_op,
                                               std::ostream &os) {
  std::string flag_id;
  if (auto *int_imm = sync_op.flag_id.template as<tvm::tir::IntImmNode>()) {
    flag_id = std::to_string(int_imm->value);
  } else {
    flag_id = PrintID(sync_op.flag_id);
  }
  this->PrintIndent();
  this->stream << "hivm.hir.sync_block_set[<";
  this->stream << coretype_syncblock_map[current_coretype] << ">, <";
  this->stream << sync_op.pipe_type;
  this->stream << ">, <PIPE_S>] flag = " << flag_id;
  this->stream << " syn_instr_mode = <" << SyncBlockMode_str[sync_op.mode];
  this->stream << ">\n";
}

void CodeGenTileLangNPUIR::DotCodegen(const CallNode *op, std::ostream &os) {
  // Generate hivm.hir.mmadL1 for tl.npuir_dot.
  // before:
  //   T.npuir_dot(T.region(A_BUF[0, 0], 1, 128, 1024),
  //               T.region(B_BUF[0, 0], 1, 1024, 256),
  //               T.region(C_BUF[0, 0], 3, 128, 256), T.bool(True))
  // after:
  // hivm.hir.mmadL1 ins(%alloc_8,  %alloc_5,  %true,  %c128,  %c64,  %c64 :
  //                     memref<128x64xf16,  #hivm.address_space<cbuf>>,
  //                     memref<64x64xf16,  #hivm.address_space<cbuf>>,
  //                     i1,  index,  index,  index)
  //                 outs(%alloc_9 : memref<128x64xf32,
  //                      #hivm.address_space<cc>>)
  tvm::tl::NpuirDot npuirop(op->args, this->vmap);
  Buffer a_buffer = npuirop.src0;
  Buffer b_buffer = npuirop.src1;
  Buffer c_buffer = npuirop.dst;
  Array<PrimExpr> a_region_shape, b_region_shape;
  for (int i = 0; i < npuirop.src0_range.size(); i++) {
    a_region_shape.push_back(npuirop.src0_range[i].get()->extent);
    b_region_shape.push_back(npuirop.src1_range[i].get()->extent);
  }
  auto init_c_name = SSAGetID(PrintExpr(npuirop.initC), op->args[3]->dtype);

  auto GetRealName = [this](const PrimExpr &extent) {
    std::ostringstream temp;
    auto real_name = SSAGetID(PrintExpr(extent), extent.dtype());
    temp << "arith.index_cast \%" << real_name << " : ";
    PrintType(extent.dtype(), temp);
    temp << " to index";
    real_name = SSAGetID(temp.str(), extent.dtype());
    return real_name;
  };
  auto real_m_name = GetRealName(a_region_shape[0]);
  auto real_k_name = GetRealName(b_region_shape[0]);
  auto real_n_name = GetRealName(b_region_shape[1]);

  this->PrintIndent();
  this->stream << "hivm.hir.mmadL1";
  if (npuirop.a_transpose || npuirop.b_transpose) {
    this->stream << " {";
    this->stream << (npuirop.a_transpose ? "a_transpose" : "");
    this->stream << (npuirop.a_transpose && npuirop.b_transpose ? ", " : "");
    this->stream << (npuirop.b_transpose ? "b_transpose" : "");
    this->stream << "}";
  }
  this->stream << " ins(%" << type_info[a_buffer->name]->var_id;
  this->stream << ", %" << type_info[b_buffer->name]->var_id;
  this->stream << ", %" << init_c_name;
  this->stream << ", %" << real_m_name;
  this->stream << ", %" << real_k_name;
  this->stream << ", %" << real_n_name;
  this->stream << " : " << GetMemrefInfo(a_buffer->name);
  this->stream << ", " << GetMemrefInfo(b_buffer->name);
  this->stream << ", i1, index, index, index";
  this->stream << ")";
  this->stream << " outs(%" << type_info[c_buffer->name]->var_id;
  this->stream << " : " << GetMemrefInfo(c_buffer->name);
  this->stream << ")\n";
}

void CodeGenTileLangNPUIR::BinaryVecOpCodegen(const CallNode *op,
                                              std::string opName,
                                              std::ostream &os) {
  // Generate hivm.hir.vadd for tl.npuir_add.
  // before:
  //   T.npuir_add(T.region(A_VEC[0], 1, 1024), T.region(B_VEC[0], 1, 1024),
  //   T.region(C_VEC[0], 2, 1024))
  // after:
  //   hivm.hir.vadd ins(%9, %15 : memref<1024xf32,
  //   #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>)
  //       outs(%16 : memref<1024xf32, #hivm.address_space<ub>>)
  std::string left_data_name = "", right_data_name = "";
  Array<PrimExpr> buffer_shape0, buffer_shape1;
  auto processImm = [&](std::string &data_name, int arg_id,
                        Array<PrimExpr> &buffer_shape) {
    if (auto intImm = op->args[arg_id].as<IntImm>()) {
      auto immObj = intImm.value();
      data_name = PrintExpr(immObj);
      const CallNode *region_node = op->args[1 - arg_id].as<CallNode>();
      const BufferLoadNode *buffer_load_node =
          region_node->args[0].as<BufferLoadNode>();
      if (intImm.value()->dtype != buffer_load_node->buffer->dtype) {
        data_name = ScalarConvertType(&immObj, buffer_load_node->buffer->dtype);
      }
      std::ostringstream temp;
      PrintType(buffer_load_node->buffer->dtype, temp);
      this->type_info[data_name] = new Scalar(data_name, temp.str());
    } else if (auto floatImm = op->args[arg_id].as<FloatImm>()) {
      auto immObj = floatImm.value();
      data_name = PrintExpr(immObj);
      const CallNode *region_node = op->args[1 - arg_id].as<CallNode>();
      const BufferLoadNode *buffer_load_node =
          region_node->args[0].as<BufferLoadNode>();
      if (floatImm.value()->dtype != buffer_load_node->buffer->dtype) {
        data_name = ScalarConvertType(&immObj, buffer_load_node->buffer->dtype);
      }
      std::ostringstream temp;
      PrintType(buffer_load_node->buffer->dtype, temp);
      this->type_info[data_name] = new Scalar(data_name, temp.str());
    } else {
      const CallNode *region_node = op->args[arg_id].as<CallNode>();
      buffer_shape = region_node->args[0].as<BufferLoadNode>()->buffer->shape;
      data_name = GenSubviewFromRegion(region_node);
    }
  };
  processImm(left_data_name, 0, buffer_shape0);
  processImm(right_data_name, 1, buffer_shape1);
  const CallNode *out_region_node = op->args[2].as<CallNode>();
  String out_data_name = "", out_addr_space = "";
  out_data_name = GenSubviewFromRegion(out_region_node);
  this->PrintIndent();
  this->stream << "hivm.hir.v" << opName;
  this->stream << " ins("
               << "\%" << left_data_name << ", "
               << "\%" << right_data_name;
  this->stream << " : ";
  this->stream << this->type_info[left_data_name]->printType() << ", "
               << this->type_info[right_data_name]->printType() << ")";
  this->stream << " outs("
               << "\%" << out_data_name << " : "
               << this->type_info[out_data_name]->printType() << ")";
  auto dims = broadcastAttrCodegen(buffer_shape0, buffer_shape1);
  if (dims != "") {
    this->stream << " broadcast" << dims;
  }
  this->stream << "\n";
}

std::string CodeGenTileLangNPUIR::ConstIntCodegen(const IntImmNode *op) {
  std::ostringstream temp;
  temp << "arith.constant ";
  if (op->dtype.is_bool()) {
    temp << (op->value == 1 ? "true" : "false");
  } else {
    temp << op->value << " : ";
    PrintType(op->dtype, temp);
  }
  return temp.str();
}

std::string CodeGenTileLangNPUIR::ConstFloatCodegen(const FloatImmNode *op) {
  std::ostringstream temp;
  if (op->value == -std::numeric_limits<float>::infinity()) {
    temp << "arith.constant 0xFF800000 : ";
  } else if (op->value == std::numeric_limits<float>::infinity()) {
    temp << "arith.constant 0x7F800000 : ";
  } else {
    temp << "arith.constant " << op->value << " : ";
  }
  PrintType(op->dtype, temp);
  return temp.str();
}

void CodeGenTileLangNPUIR::VisitExpr_(const CallNode *op, std::ostream &os) {
  if (op->op.same_as(Op::Get("tl.npuir_pipe_barrier"))) {
    BarrierCodegen(op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_set_flag"))) {
    tvm::tl::NpuirSetFlag sync_op(op->args, this->vmap);
    PipeFlagCodegen(sync_op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_wait_flag"))) {
    tvm::tl::NpuirWaitFlag sync_op(op->args, this->vmap);
    PipeFlagCodegen(sync_op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_sync_block"))) {
    tvm::tl::NpuirSyncBlock sync_op(op->args, this->vmap);
    SyncBlockSetCodegen(sync_op, os);
    SyncBlockWaitCodegen(sync_op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_sync_block_set"))) {
    tvm::tl::NpuirSyncBlockSet sync_op(op->args, this->vmap);
    SyncBlockSetCodegen(sync_op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_sync_block_wait"))) {
    tvm::tl::NpuirSyncBlockWait sync_op(op->args, this->vmap);
    SyncBlockWaitCodegen(sync_op, os);
  } else if (op->op.same_as(Op::Get("tl.ascend_copy"))) {
    AscendCopyCodegen(op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_add"))) {
    BinaryVecOpCodegen(op, "add", os);
  } else if (op->op.same_as(Op::Get("tl.npuir_exp"))) {
    UnaryVecOpCodegen<tvm::tl::NpuirExp>(op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_ln"))) {
    UnaryVecOpCodegen<tvm::tl::NpuirLn>(op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_relu"))) {
    UnaryVecOpCodegen<tvm::tl::NpuirRelu>(op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_select"))) {
    VselectCodegen(op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_cmp"))) {
    VcmpCodegen(op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_load_nd2nz"))) {
    Nd2NzCodegen(op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_store_fixpipe"))) {
    FixpipeCodegen(op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_dot"))) {
    DotCodegen(op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_div"))) {
    BinaryVecOpCodegen(op, "div", os);
  } else if (op->op.same_as(Op::Get("tl.npuir_mul"))) {
    BinaryVecOpCodegen(op, "mul", os);
  } else if (op->op.same_as(Op::Get("tl.npuir_sub"))) {
    BinaryVecOpCodegen(op, "sub", os);
  } else if (op->op.same_as(Op::Get("tl.npuir_max"))) {
    BinaryVecOpCodegen(op, "max", os);
  } else if (op->op.same_as(Op::Get("tl.npuir_min"))) {
    BinaryVecOpCodegen(op, "min", os);
  } else if (op->op.same_as(Op::Get("tl.npuir_brc"))) {
    VbrcCodegen(op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_cast"))) {
    VcastCodegen(op, os);
  } else if (op->op.same_as(Op::Get("tl.npuir_reduce"))) {
    VreduceCodegen(op, os);
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}
void CodeGenTileLangNPUIR::VisitStmt_(const LetStmtNode *op) {
  std::string value = "";
  if (const IntImmNode *int_imm = op->value.as<IntImmNode>()) {
    value = ConstIntCodegen(int_imm);
  } else if (const FloatImmNode *float_imm = op->value.as<FloatImmNode>()) {
    value = ConstFloatCodegen(float_imm);
  } else {
    value = PrintExpr(op->value);
  }
  PrintIndent();
  this->stream << '%' << AllocVarID(op->var.get()) << " = " << value << "\n";
  PrintStmt(op->body);
}

void CodeGenTileLangNPUIR::VisitStmt_(const AttrStmtNode *op) {
  if (op->attr_key == "thread_extent") {
    IterVar iv = Downcast<IterVar>(op->node);
    if (iv->thread_tag == "blockIdx.x" && iv->var->name_hint != "_") {
      this->block_id_ = AllocVarID(iv->var.get());
      this->PrintIndent();
      this->stream << "%" << this->block_id_
                   << "_i64 = hivm.hir.get_block_idx -> i64\n";
      this->PrintIndent();
      this->stream << "%" << this->block_id_ << " = arith.trunci %"
                   << this->block_id_ << "_i64 : i64 to i32\n";
      this->core_num_ = op->value.as<IntImmNode>()->value;
    } else if (iv->thread_tag == "blockIdx.y" && iv->var->name_hint != "_") {
      auto vec_id_ = AllocVarID(iv->var.get());
      this->PrintIndent();
      this->stream << "%" << vec_id_
                   << "_i64 = hivm.hir.get_sub_block_idx -> i64\n";
      this->PrintIndent();
      this->stream << "%" << vec_id_ << " = arith.trunci %" << vec_id_
                   << "_i64 : i64 to i32\n";
    }
    this->VisitStmt(op->body);
    return;
  } else if (op->attr_key == "resource_scope") {
    auto resource_id = Downcast<IntImm>(op->value)->value;
    auto resource_name = resource_id == 0 ? "AIC" : "AIV";

    if (NPU_CORETYPE_STR[this->current_coretype] == resource_name) {
      this->VisitStmt(op->body);
    }
    // else do nothing but return.
    return;
  }
  CodeGenC::VisitStmt_(op);
}

/// Generate memref.alloc for TIR AllocateNode like T.decl_buffer.
/// before:
///      A_VEC = T.decl_buffer((128, 256), "float16", scope="shared")
/// after:
///      %A_VEC = memref.alloc() : memref<128x256xf16,
///      #hivm.address_space<ub>>
void CodeGenTileLangNPUIR::VisitStmt_(const AllocateNode *op) {
  ICHECK(!is_zero(op->condition));
  std::string scope = GetPtrStorageScope(op->buffer_var);
  std::map<std::string, NPU_CORETYPE> scope_coretype_map{
      {"shared", NPU_CORETYPE::AIV},
      {"shared.dyn", NPU_CORETYPE::AIC},
      {"wmma.accumulator", NPU_CORETYPE::AIC}};
  if (scope_coretype_map[scope] == this->current_coretype) {
    std::string vid = AllocVarID(op->buffer_var.get());
    String address_space = GetAddressSpace(scope);
    // add new memref obj
    this->type_info[vid] =
        new Memref(vid, op->extents, op->dtype, address_space, false);
    this->PrintIndent();
    stream << "%" << vid << " = "
           << "memref.alloc() : " << GetMemrefInfo(vid) << "\n";
  }
  this->VisitStmt(op->body);
}

void CodeGenTileLangNPUIR::VisitExpr_(const MinNode *op, std::ostream &os) {
  if (op->dtype.is_int()) {
    PrintBinary(op, "minsi", os, this);
  } else if (op->dtype.is_uint()) {
    PrintBinary(op, "minui", os, this);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "minnumf", os, this);
  }
}

void CodeGenTileLangNPUIR::VisitExpr_(const MaxNode *op, std::ostream &os) {
  if (op->dtype.is_int()) {
    PrintBinary(op, "maxsi", os, this);
  } else if (op->dtype.is_uint()) {
    PrintBinary(op, "maxui", os, this);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "maxnumf", os, this);
  }
}

void CodeGenTileLangNPUIR::VisitExpr_(const AddNode *op, std::ostream &os) {
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    PrintBinary(op, "addi", os, this);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "addf", os, this);
  }
}

void CodeGenTileLangNPUIR::VisitExpr_(const SubNode *op, std::ostream &os) {
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    PrintBinary(op, "subi", os, this);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "subf", os, this);
  }
}

void CodeGenTileLangNPUIR::VisitExpr_(const FloatImmNode *op,
                                      std::ostream &os) { // NOLINT(*)
  std::string temp = ConstFloatCodegen(op);
  os << SSAGetID(temp, op->dtype);
}

void CodeGenTileLangNPUIR::VisitExpr_(const IntImmNode *op, std::ostream &os) {
  std::string temp = ConstIntCodegen(op);
  os << SSAGetID(temp, op->dtype);
}

void CodeGenTileLangNPUIR::VisitExpr_(const MulNode *op, std::ostream &os) {
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    PrintBinary(op, "muli", os, this);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "mulf", os, this);
  }
}

void CodeGenTileLangNPUIR::VisitExpr_(const AndNode *op, std::ostream &os) {
  assert(op->a.dtype().is_int() || op->a.dtype().is_uint());
  assert(op->b.dtype().is_int() || op->b.dtype().is_uint());
  PrintBinary(op, "andi", os, this);
}

void CodeGenTileLangNPUIR::VisitExpr_(const OrNode *op, std::ostream &os) {
  assert(op->a.dtype().is_int() || op->a.dtype().is_uint());
  assert(op->b.dtype().is_int() || op->b.dtype().is_uint());
  PrintBinary(op, "ori", os, this);
}

void CodeGenTileLangNPUIR::VisitExpr_(const DivNode *op, std::ostream &os) {
  PrintBinary(op, "<<<divf>>>", os, this);
}

void CodeGenTileLangNPUIR::VisitExpr_(const SelectNode *op, std::ostream &os) {
  auto condition = PrintExpr(op->condition);
  auto true_value = PrintExpr(op->true_value);
  auto false_value = PrintExpr(op->false_value);

  os << "(" << condition << " ? "
     << "" << true_value << " : " << false_value << ")";
}

void PrintHostFunc(const PrimFunc &f, const std::string &name, std::ostream &os,
                   int core) {
  os << "extern \"C\" void call(";
  std::vector<std::string> arg_names;
  for (size_t i = 0; i < f->params.size(); ++i) {
    auto v = f->params[i];
    if (i != 0) {
      os << ", ";
    }
    arg_names.push_back(v->name_hint);
    os << "uint8_t* " << v->name_hint;
  }
  os << ", aclrtStream stream) {\n  ";

  os << name << "<<<" << core << ", nullptr, stream>>>(";
  for (auto &arg_name : arg_names) {
    os << arg_name;
    if (arg_name != arg_names.back()) {
      os << ", ";
    }
  }
  os << ");\n";
  os << "}\n";
}

std::string CodeGenTileLangNPUIR::GetBufferRef(const BufferNode* buffer,
                                               Array<PrimExpr> indices) {
  std::vector<std::string> idx_names;
  for (auto index : indices) {
    auto index_str = SSAGetID(PrintExpr(index), index.dtype());
    std::ostringstream temp;
    temp << "arith.index_cast %" << index_str << " : ";
    PrintType(index.dtype(), temp);
    temp << " to index";
    idx_names.push_back(SSAGetID(temp.str(), index.dtype()));
  }

  // FIXME: Some Vars maybe have specific types when they are from Allocate
  // operator, in this case, the corresponding Buffer dtype and the Var data
  // type may be different.
  const VarNode* buffer_var = buffer->data.get();
  std::string vid = GetVarID(buffer_var);
  if (auto memrefInfo = dynamic_cast<Memref *>(type_info[vid])) {
    if (memrefInfo->is_arg) {
      vid = vid + "_Recast";
    }
  } else {
    LOG(FATAL) << vid << " should be a memref";
  }

  std::ostringstream temp;
  temp << "%" << vid << "[";
  bool first = true;
  for (auto name : idx_names) {
    if (first) {
      first = false;
    } else {
      temp << ", ";
    }
    temp << "%" << name;
  }
  temp <<  "] : ";
  temp << GetMemrefInfo(vid);
  return temp.str();
}

void CodeGenTileLangNPUIR::VisitExpr_(const BufferLoadNode* op, std::ostream& os) {
  if (op->dtype.lanes() != 1) {
    LOG(FATAL) << "lanes not one";
  }
  auto buffer = op->buffer.get();
  if (op->dtype != buffer->dtype) {
    LOG(FATAL) << "The load type and buffer element type do not match";
  }

  auto buffer_ref =  GetBufferRef(buffer, op->indices);
  os << "memref.load " << buffer_ref;
}

void CodeGenTileLangNPUIR::VisitStmt_(const BufferStoreNode* op) {
  if (op->value.dtype().lanes() != 1) {
    LOG(FATAL) << "lanes not one";
  }
  auto buffer = op->buffer.get();
  if (op->value.dtype() != buffer->dtype) {
    LOG(FATAL) << "The load type and buffer element type do not match";
  }

  auto buffer_ref = GetBufferRef(buffer, op->indices);
  std::string value = this->PrintExpr(op->value);
  this->PrintIndent();
  stream << "memref.store %";
  stream << value << ", " << buffer_ref << "\n";
}

void CodeGenTileLangNPUIR::GenRecastFromArg(Buffer curr_buffer, String arg_name,
                                            String &recast_inst) {
  // reinterpret_cast memref from 1D to xD
  std::ostringstream res;
  String target_strides = GetBufferStrides(curr_buffer);
  String cast_name = arg_name + "_Recast";
  // add new memref obj
  this->type_info[cast_name] = new Memref(cast_name, curr_buffer);
  res << "\%" << cast_name << " = ";
  res << "memref.reinterpret_cast \%";
  res << arg_name;
  res << " to offset: [0], sizes: [";
  PrintShape(curr_buffer->shape, ", ", res);
  res << "], strides: ";
  res << target_strides;
  res << " : ";
  res << GetMemrefInfo(arg_name);
  res << " to ";
  res << GetMemrefInfo(cast_name);
  res << "\n";
  recast_inst = res.str();
}

void CodeGenTileLangNPUIR::AddFunctionForCoreType(const GlobalVar &gvar,
                                                  const PrimFunc &f) {
  // If the function has already been forward-declared, this is a
  // no-op.
  CodeGenC::DeclareFunction(gvar, f);
  // clear previous generated state.
  this->InitFuncState(f);

  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);
  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";

  int func_scope = this->BeginScope();
  this->PrintIndent();
  auto func_name = static_cast<std::string>(global_symbol.value());
  if (this->func_coretype == NPU_CORETYPE::MIX) {
    func_name = func_name + "_mix_" + NPU_CORETYPE_str[this->current_coretype];
  }

  this->stream << "func.func @" << func_name << "(";

  std::vector<String> recast_need_insert;
  // add args before memref
  stream << "\%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, "
            "\%arg1: memref<?xi8>, \%arg2: memref<?xi8>, ";
  this->type_info.clear();
  for (size_t i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];
    std::string vid = AllocVarID(v.get());

    if (i != 0)
      stream << ", ";

    if (v.dtype().is_handle()) {
      this->vmap = f->buffer_map;
      auto real_v = f->buffer_map[v]->data;
      String arg_name = AllocVarID(real_v.get());
      // add new memref obj
      Memref *buffer = new Memref(arg_name, f->buffer_map[v], true);
      this->type_info[arg_name] = buffer;
      stream << "%" << arg_name << ": " << GetMemrefInfo(arg_name);
      String recast_inst = "";
      GenRecastFromArg(f->buffer_map[v], arg_name, recast_inst);
      recast_need_insert.push_back(recast_inst);

      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
        }
      }
    } else {
      stream << "%" << vid << ": ";
      CodeGenC::PrintType(GetType(v), stream);
    }
  }
  // add args after memref
  stream << ", \%arg13: i32, \%arg14: i32, \%arg15: i32, \%arg16: i32, "
            "\%arg17: i32, \%arg18: i32";
  stream << ")\n";
  this->PrintIndent();
  stream
      << "attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : "
         "i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, "
         "hivm.func_core_type = #hivm.func_core_type<"
      << NPU_CORETYPE_STR[this->current_coretype] << ">";
  if (this->func_coretype == NPU_CORETYPE::MIX) {
    stream << ", hivm.part_of_mix, mix_mode = \"mix\"";
  }
  stream << "} {\n";

  int func_body_scope = this->BeginScope();
  this->PrintIndent();
  stream << "hivm.hir.set_ffts_base_addr \%arg0\n";
  for (String recast_inst : recast_need_insert) {
    this->PrintIndent();
    stream << recast_inst;
  }
  this->PrintStmt(f->body);
  this->EndScope(func_body_scope);
  this->PrintIndent();
  this->stream << "return\n";
  this->PrintIndent();
  this->stream << "}\n";
  this->EndScope(func_scope);
}

/// Infer function core type: aic, aiv, mix
class InferFuncCoreType : public StmtExprVisitor {
  std::map<std::string, NPU_CORETYPE> scope_coretype_map{
      {"shared", NPU_CORETYPE::AIV},
      {"shared.dyn", NPU_CORETYPE::AIC},
      {"wmma.accumulator", NPU_CORETYPE::AIC},
      {"wmma.matrix_a", NPU_CORETYPE::AIC},
      {"wmma.matrix_b", NPU_CORETYPE::AIC}};

public:
  void VisitStmt(const Stmt &stmt) override {
    StmtExprVisitor::VisitStmt(stmt);
  }
  void VisitStmt_(const AttrStmtNode *op) final {
    // It is mixkernel iff there exists T.rs.
    if (op->attr_key == "resource_scope") {
      func_coretype = NPU_CORETYPE::MIX;
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const CallNode *op) final {
    // It is cube kernel if there exists T.npuir_dot.
    if (op->op.same_as(Op::Get("tl.npuir_dot"))) {
      if (func_coretype != NPU_CORETYPE::MIX) {
        func_coretype = NPU_CORETYPE::AIC;
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const AllocateNode *op) final {
    // It is cube kernel if there exists buffer with shared.dyn/wmma.xxx address
    // space
    std::string scope = GetPtrStorageScope(op->buffer_var);
    if (func_coretype != NPU_CORETYPE::MIX) {
      if (scope_coretype_map.count(scope) != 0) {
        func_coretype = scope_coretype_map[scope];
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  NPU_CORETYPE func_coretype{NPU_CORETYPE::AIV};
};

void CodeGenTileLangNPUIR::AddFunction(const GlobalVar &gvar,
                                       const PrimFunc &f) {
  InferFuncCoreType infer;
  infer.VisitStmt(f->body);

  this->func_coretype = infer.func_coretype; // NPU_CORETYPE::MIX;

  this->stream
      << "module attributes {hivm.module_core_type = #hivm.module_core_type<"
      << NPU_CORETYPE_STR[this->func_coretype] << ">} {\n";

  if (this->func_coretype == NPU_CORETYPE::MIX ||
      this->func_coretype == NPU_CORETYPE::AIC) {
    this->current_coretype = NPU_CORETYPE::AIC;
    AddFunctionForCoreType(gvar, f);
  }

  if (this->func_coretype == NPU_CORETYPE::MIX ||
      this->func_coretype == NPU_CORETYPE::AIV) {
    this->current_coretype = NPU_CORETYPE::AIV;
    AddFunctionForCoreType(gvar, f);
  }

  this->PrintIndent();
  this->stream << "}\n\n";

  //  PrintHostFunc(f, func_name, stream, this->core_num_);
}

String CodeGenTileLangNPUIR::GetMemrefInfo(String name) {
  if (this->type_info.count(name) == 0)
    LOG(FATAL) << "Can not find memref ssa object: " << name;
  if (!dynamic_cast<Memref *>(type_info[name])) {
    LOG(FATAL) << name << " should be a memref";
  }
  Memref *MemrefObj = dynamic_cast<Memref *>(this->type_info[name]);
  return GetMemrefInfo(MemrefObj);
}

String CodeGenTileLangNPUIR::GetMemrefInfo(Memref *memrefObj) {
  if (memrefObj->type_str != "")
    return memrefObj->type_str;
  std::ostringstream memref_type;
  memref_type << "memref<";
  // dump shape
  if (memrefObj->is_arg) {
    memref_type << "?x";
  } else {
    for (PrimExpr s : memrefObj->shape) {
      if (auto s_int = as_const_int(s)) {
        memref_type << std::to_string(*s_int) + "x";
      } else {
        // not support ssa var in memref size
        memref_type << "?x";
      }
    }
  }
  PrintType(memrefObj->dtype, memref_type);
  // dump strides and offset
  if (!memrefObj->is_arg) {
    memref_type << ", strided<[";
    for (int i = 0; i < memrefObj->dim; i++) {
      if (i > 0)
        memref_type << ", ";
      if (memrefObj->stride.size() > 0) {
        if (auto s_int = as_const_int(memrefObj->stride[i])) {
          memref_type << std::to_string(*s_int);
        } else {
          // not support ssa var in memref size
          memref_type << "?";
        }
      } else {
        memref_type << memrefObj->stride_int[i];
      }
    }
    memref_type << "], offset:";
    if (memrefObj->var_offset)
      memref_type << "?";
    else
      memref_type << memrefObj->offset;
    memref_type << ">";
  }
  memref_type << ", #hivm.address_space<" << memrefObj->address_space << ">>";
  memrefObj->type_str = memref_type.str();
  return memrefObj->type_str;
}

void Memref::GetIntStride() {
  if (stride.empty()) {
    stride_int = GetStrideFromShape(shape);
    for (unsigned long s : stride_int) {
      stride.push_back(IntImm(DataType::Int(64), s));
    }
  } else {
    for (PrimExpr s : stride) {
      if (auto s_int = as_const_int(s))
        stride_int.push_back(*s_int);
    }
  }
}

Memref::Memref(String name, Array<PrimExpr> shape_in, DataType dtype_in,
               String addr_space_in, bool var_offset_in,
               Array<PrimExpr> stride_in, int offset_in, bool is_arg_in) {
  var_id = name;
  shape = shape_in;
  stride = stride_in;
  dtype = dtype_in;
  offset = offset_in;
  is_arg = is_arg_in;
  address_space = addr_space_in;
  var_offset = var_offset_in;
  dim = shape_in.size();
  GetIntStride();
}

Memref::Memref(String name, Buffer buffer, bool is_arg_in) {
  var_id = name;
  shape = buffer->shape;
  stride = buffer->strides;
  dtype = buffer->dtype;
  offset = 0;
  is_arg = is_arg_in;
  String scope = GetPtrStorageScope(buffer->data);
  address_space = GetAddressSpace(scope);
  var_offset = false;
  dim = shape.size();
  GetIntStride();
}

} // namespace codegen

} // namespace tvm
