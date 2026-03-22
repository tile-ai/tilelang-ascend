// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file target/codegen_ascend_pto.cc
 */

#include "codegen_ascend_pto.h"
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/container/string.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../op/ascend.h"
#include "../op/builtin.h"

#include "arith/pattern_match.h"

#define DEC_STR_TO_HEX_STR(dec_str)                                            \
  ([](const std::string &s) {                                                  \
    std::stringstream ss;                                                      \
    ss << std::showbase << std::hex << std::uppercase << std::stoi(s);         \
    return ss.str();                                                           \
  }(dec_str))

namespace tvm {
namespace codegen {
const std::string kAscendPtoScope = "tl::ascend_pto::";

using ShapeInfo = CodeGenTileLangAscendPto::ShapeInfo;

static std::string getType(const DataType &dtype) {
  if (dtype.is_float16()) {
    return "half";
  } else if (dtype.is_float()) {
    return "float";
  } else if (dtype.is_int() && dtype.bits() == 4) {
    return "int4b_t";
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
}

static DataType GetAccessPtrDtypePto(const CallNode *access_ptr) {
  if (!access_ptr) {
    LOG(FATAL) << "access_ptr is nullptr";
  }
  if (access_ptr->args.empty()) {
    LOG(FATAL) << "access_ptr has no arguments";
  }
  auto type_arg = access_ptr->args[0];
  if (auto *call = type_arg.as<CallNode>()) {
    return call->dtype;
  } else if (auto *str = type_arg.as<StringImmNode>()) {
    return DataType(runtime::String2DLDataType(str->value));
  } else {
    LOG(FATAL) << "Unexpected type for access_ptr first argument: " << type_arg->GetTypeKey();
    return DataType();
  }
}

int32_t GetTypeLen(std::string type) {
  int32_t typeSize = 1;
  if (type == "float") {
    typeSize = 4;
  } else if (type == "bfloat16_t") {
    typeSize = 2;
  } else if (type == "half") {
    typeSize = 2;
  } else if (type == "int8_t" || type == "uint8_t") {
    typeSize = 1;
  } else if (type == "int16_t" || type == "uint16_t") {
    typeSize = 2;
  } else if (type == "int" || type == "uint32_t") {
    typeSize = 4;
  } else {
    ICHECK(false) << "Unsupported datatype";
  }
  return typeSize;
}

std::string GetTypeLenString(std::string type) {
  std::string typeSize = "1";
  if (type == "float") {
    typeSize = "4";
  } else if (type == "bfloat16_t") {
    typeSize = "2";
  } else if (type == "half") {
    typeSize = "2";
  } else if (type == "int8_t" || type == "uint8_t") {
    typeSize = "1";
  } else if (type == "int16_t" || type == "uint16_t") {
    typeSize = "2";
  } else if (type == "int" || type == "uint32_t") {
    typeSize = "4";
  } else {
    ICHECK(false) << "Unsupported datatype";
  }
  return typeSize;
}

std::string
CodeGenTileLangAscendPto::GetTempVarName(const std::string &temp_name) {
  return temp_name + "_" + "temp" + "_" +
         std::to_string(counters_[temp_name]++);
}

void CodeGenTileLangAscendPto::CreateUbVariableND(const std::string &temp_name,
                                                  const ShapeInfo &shape_info) {
  int32_t slice_row;
  int32_t slice_col;
  if ((GetTypeLen(shape_info.type) * shape_info.slice_col % 32) != 0 &&
      (GetTypeLen(shape_info.type) * shape_info.slice_row % 32) == 0) {
    slice_row = shape_info.slice_col;
    slice_col = shape_info.slice_row;
  } else if ((GetTypeLen(shape_info.type) * shape_info.slice_col % 32) == 0) {
    slice_row = shape_info.slice_row;
    slice_col = shape_info.slice_col;
  } else {
    ICHECK(false) << "The sliced column does not support ND typing";
  }

  this->PrintIndent();
  this->stream << kAscendPtoScope << "TileUbDataND<" << shape_info.type << ", "
               << slice_row << ", " << slice_col << ", " << slice_row << ", "
               << slice_col << "> " << temp_name << ";\n";

  this->PrintIndent();
  this->stream << "TASSIGN(" << temp_name << ", " << shape_info.first_addr
               << " + " << shape_info.offset << " * "
               << GetTypeLen(shape_info.type) << ");\n";
}

void CodeGenTileLangAscendPto::CreateUbVariableDN(const std::string &temp_name,
                                                  const ShapeInfo &shape_info) {
  this->PrintIndent();
  this->stream << kAscendPtoScope << "TileUbDataDN<" << shape_info.type << ", "
               << shape_info.slice_col << ", " << shape_info.slice_row << ", "
               << shape_info.slice_col << ", " << shape_info.slice_row << "> "
               << temp_name << ";\n";

  this->PrintIndent();
  this->stream << "TASSIGN(" << temp_name << ", " << shape_info.first_addr
               << " + " << shape_info.offset << " * "
               << GetTypeLen(shape_info.type) << ");\n";
}

ShapeInfo CodeGenTileLangAscendPto::GetSliceInfo(const CallNode *op) {
  auto shape = buffer_shapess_[GetRef<tir::Var>(op->args[1].as<VarNode>())];
  int32_t row;
  int32_t col;
  if (shape.size() == 1) {
    row = 1;
    col = shape[0].as<IntImmNode>()->value;
  } else if (shape.size() == 2 && shape[0].as<IntImmNode>()->value == 0) {
    row = 1;
    col = shape[1].as<IntImmNode>()->value;
  } else if (shape.size() == 2 && shape[1].as<IntImmNode>()->value == 0) {
    row = 1;
    col = shape[0].as<IntImmNode>()->value;
  } else {
    row = shape[0].as<IntImmNode>()->value;
    col = shape[1].as<IntImmNode>()->value;
  }
  int32_t extent = op->args[3].as<IntImmNode>()->value;
  int32_t slice_row = (extent / col) > 1 ? (extent / col) : 1;
  int32_t slice_col = extent > col ? col : extent;
  auto src_addr = ub_data_map_[PrintExpr(op->args[1])][3];
  auto offset = PrintExpr(op->args[2]);
  auto type = ub_data_map_[PrintExpr(op->args[1])][0];
  bool is_slice;
  if (shape.size() == 1) {
    is_slice = extent != col;
  } else {
    is_slice = extent != row * col;
  }
  auto ub_name = var_idmap_[op->args[1].as<VarNode>()];
  return ShapeInfo{row,      col,    slice_row, slice_col, extent,
                   src_addr, offset, type,      ub_name,   is_slice};
}

CodeGenTileLangAscendPto::CodeGenTileLangAscendPto(std::string platform) {
  // restrict_keyword_ = "__gm__ uint8_t *";
  platform_ = platform;
}

void CodeGenTileLangAscendPto::PrintFuncPrefix(std::ostream &os) {
  // os << "extern \"C\" CATLASS_GLOBAL\n";
}

std::string CodeGenTileLangAscendPto::Finish() {
  decl_stream << "#include \"tl_templates/pto/common.h\"\n";
  decl_stream << "#include <pto/pto-inst.hpp>\n";
  decl_stream << "#include \"acl/acl.h\"\n";
  decl_stream << "#include <runtime/rt_ffts.h>\n";
  decl_stream << "using namespace pto;\n";
  decl_stream << "\n";
  std::ostringstream code;
  code << decl_stream.str();
  code << stream.str();
  return code.str();
}

void CodeGenTileLangAscendPto::VisitStmt_(const tir::ForNode *op) {
  auto flush = false;
  if (flush_out_) {
    flush = true;
    flush_out_ = false;
  }
  if (op->kind == tir::ForKind::kUnrolled) {
    PrintIndent();
    stream << "#pragma unroll\n";
  }
  std::string extent =
      PrintExpr(arith::Analyzer().Simplify(op->extent + op->min));
  std::string vid = AllocVarID(op->loop_var.get());
  std::string start = PrintExpr(op->min);
  for_num_map_[vid] = extent;
  stream << "\n  for (";
  PrintType(op->loop_var.dtype(), stream);
  stream << ' ' << vid << " = " << start << "; " << vid << " < " << extent
         << "; ++" << vid << ") {\n";
  int for_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
  if (flush) {
    while (!inst_.empty()) {
      PrintIndent();
      stream << inst_.back();
      inst_.pop_back();
    }
  }
}

void CodeGenTileLangAscendPto::PrintType(DataType t,
                                         std::ostream &os) { // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK(t.is_scalar()) << "do not yet support vector types";
    os << "void*";
    return;
  }

  if (t.is_void()) {
    os << "void";
    return;
  }

  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
    case 16:
      enable_fp16_ = true;
      if (t.is_scalar()) {
        os << "half";
      } else if (lanes <= 8) {
        // Emit CUDA code to access fp16 vector elements.
        //
        // half4 is stored as uint2
        //
        // h4.x is emitted as *(half2*)(&(u2.x)).x
        // h4.y is emitted as *(half2*)(&(u2.x)).y
        // h4.z is emitted as *(half2*)(&(u2.y)).x
        // h4.w is emitted as *(half2*)(&(u2.y)).y
        //
        ICHECK_EQ(lanes % 2, 0) << "only support even lane for half type";
        os << "uint" << lanes / 2;
      } else {
        fail = true;
      }
      break;
    case 32:
      if (lanes <= 4) {
        os << "float";
      } else if (lanes <= 8) {
        // Emit CUDA code to access fp32 vector elements for 4 < lanes <= 8.
        //
        // float8 is stored as ulonglong4
        //
        // f8.v1 is emitted as *(float2*)(&(ul4.x)).x
        // f8.v2 is emitted as *(float2*)(&(ul4.x)).y
        //
        ICHECK_EQ(lanes % 2, 0)
            << "only support even lane for float type with lanes > 4";
        os << "ulonglong" << lanes / 2;
      } else {
        fail = true;
      }
      break;
    case 64:
      os << "double";
      break;
    default:
      fail = true;
      break;
    }
    if (!fail && (t.is_scalar() || t.bits() == 16))
      return;
    if (!fail && (lanes > 4 && lanes <= 8 && t.bits() == 32))
      return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  } else if (t.is_bfloat16()) {
    enable_bf16_ = true;
    if (t.is_scalar()) {
      os << "bfloat16_t";
    } else if (lanes <= 8) {
      ICHECK_EQ(lanes % 2, 0) << "only support even lane for half type";
      os << "uint" << lanes / 2;
    } else {
      fail = true;
    }
    if (!fail)
      return;
  } else if (t.is_float8()) {
    // enable_fp8_ = true;
    // os << GetFP8Type(t);
    return;
  } else if (t == DataType::Bool()) {
    os << "bool";
    return;
  } else if (t.is_vector_bool()) {
    // CUDA does not support bool vectors.
    // Use ushort vectors to represent instead.
    int n = t.lanes();
    if (n <= 4) {
      os << "ushort" << n;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << "u";
    }
    switch (t.bits()) {
    case 1: {
      if (t.is_scalar()) {
        os << "int";
        return;
      } else if (t.lanes() == 8) {
        os << "int8_t";
        return;
      } else if (t.lanes() == 16) {
        os << "int16_t";
        return;
      } else if (t.lanes() == 32) {
        os << "int";
        return;
      } else {
        LOG(FATAL) << "Cannot convert type " << t << " to CUDA type!";
      }
    }
    case 4: {
      if (t.is_scalar()) {
        os << "int";
        return;
      } else if (t.lanes() == 4) {
        os << "int16_t";
        return;
      } else if (t.lanes() == 8) {
        // directly 8 4-bit int in integer.
        os << "int";
        return;
      } else if (t.lanes() == 16) {
        os << "int2";
        return;
      } else if (t.lanes() == 32) {
        os << "int4";
        return;
      } else if (t.lanes() == 64) {
        os << "int8";
        return;
      } else {
        LOG(FATAL) << "Cannot convert type " << t << " to CUDA type!";
      }
    }
    case 8: {
      if (t.lanes() == 4) {
        // directly 4 8 bit int in integer.
        enable_int8_ = true;

        // We use int for int8x4 instead of char4 because using char4 is
        // likely to produce extra instructions to pack four int8 elements
        // into 32-bit data.
        os << "int";
        return;
      } else if (t.lanes() == 8) {
        enable_int8_ = true;
        os << "int2";
        return;
      } else if (t.lanes() == 16) {
        enable_int8_ = true;
        os << "int4";
        return;
      } else if (!t.is_uint() && t.is_scalar()) {
        os << "signed char";
        break;
      } else {
        os << "char";
        break;
      }
    }
    case 16: {
      if (t.is_scalar()) {
        os << "short";
      } else if (t.lanes() <= 4) {
        os << "short" << lanes;
      } else if (t.lanes() <= 8) {
        // Emit CUDA code to access int16 vector elements.
        //
        // short4 is stored as int2
        //
        // s4.x is emitted as *(short2*)(&(i2.x)).x
        // s4.y is emitted as *(short2*)(&(i2.x)).y
        // s4.z is emitted as *(short2*)(&(i2.y)).x
        // s4.w is emitted as *(short2*)(&(i2.y)).y
        //
        ICHECK_EQ(t.lanes() % 2, 0)
            << "only support even lane for shorT type with lanes > 4";
        os << "int" << t.lanes() / 2;
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
        os << "int32_t";
      } else if (t.lanes() <= 4) {
        os << "int" << t.lanes();
      } else if (t.lanes() <= 8) {
        // Emit CUDA code to access int32 vector elements for 4 < lanes <= 8.
        //
        // int8 is stored as longlong4
        //
        // i8.v1 is emitted as *(int2*)(&(l4.x)).x
        // i8.v2 is emitted as *(int2*)(&(l4.x)).y
        //
        ICHECK_EQ(lanes % 2, 0)
            << "only support even lane for int32 type with lanes > 4";
        os << "longlong" << lanes / 2;
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
        os << "int64_t";
      } else if (t.lanes() == 2) {
        os << "longlong2";
      } else if (t.lanes() == 3) {
        os << "longlong3";
      } else if (t.lanes() == 4) {
        os << "longlong4";
      }
      return;
    }
    default:
      fail = true;
      break;
    }
    if (!fail && lanes == 1) {
      return;
    }
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to CUDA type";
}

void CodeGenTileLangAscendPto::PrintStorageScope(
    const std::string &scope,
    std::ostream &os) { // NOLINT(*)
}

void CodeGenTileLangAscendPto::VisitExpr_(const FloorDivNode *op,
                                          std::ostream &os) {
  os << "(";
  PrintExpr(op->a, os);
  os << " / ";
  PrintExpr(op->b, os);
  os << ")";
}

void CodeGenTileLangAscendPto::VisitExpr_(const FloorModNode *op,
                                          std::ostream &os) {
  os << "(";
  PrintExpr(op->a, os);
  os << " % ";
  PrintExpr(op->b, os);
  os << ")";
}

void CodeGenTileLangAscendPto::VisitExpr_(const BufferLoadNode *op,
                                          std::ostream &os) {
  auto var_name = var_idmap_[op->buffer->data.get()];
  std::string scope = op->buffer.scope();
  if (scope == "" || scope == "global") {
    os << "*(" << var_name << "_handle + " << PrintExpr(op->indices.back())
       << ")";
  } else if (scope == "local.var") {
    os << var_name;
  } else {
    os << var_name << ".GetValue(" << PrintExpr(op->indices.back()) << ")";
  }
}

void CodeGenTileLangAscendPto::VisitStmt_(const BufferStoreNode *op) {
  auto var_name = var_idmap_[op->buffer->data.get()];
  this->PrintIndent();
  std::string scope = op->buffer.scope();

  if (scope == "" || scope == "global") {
    this->stream << "*(" << var_name << "_handle + "
                 << PrintExpr(op->indices.back())
                 << ") = " << PrintExpr(op->value) << ";\n";
  } else if (scope == "local.var") {
    this->stream << var_name << " = " << PrintExpr(op->value) << ";\n";
  } else {
    this->stream << var_name << ".SetValue(" << PrintExpr(op->indices.back())
                 << ", " << PrintExpr(op->value) << ");\n";
  }
}

std::map<std::string, std::string>
extractTemplateParams(const std::string &input) {
  std::map<std::string, std::string> result;
  size_t start = input.find('<');
  size_t end = input.rfind('>');

  if (start == std::string::npos || end == std::string::npos || start >= end) {
    return result;
  }
  std::string inner = input.substr(start + 1, end - start - 1);
  std::vector<std::string> params;
  std::stringstream ss(inner);
  std::string param;
  while (std::getline(ss, param, ',')) {
    param.erase(0, param.find_first_not_of(" \t"));
    param.erase(param.find_last_not_of(" \t") + 1);
    params.push_back(param);
  }
  std::vector<std::string> paramNames = {
      "data_type_input", "data_type_output", "M", "N", "K",
      "transpose_A",     "transpose_B"};
  for (size_t i = 0; i < params.size() && i < paramNames.size(); ++i) {
    result[paramNames[i]] = params[i];
  }
  for (size_t i = paramNames.size(); i < params.size(); ++i) {
    result["extra_param_" + std::to_string(i - paramNames.size() + 1)] =
        params[i];
  }
  return result;
}

std::map<std::string, std::string>
extractTemplateParams1(const std::string &input) {
  std::map<std::string, std::string> result;
  size_t start = input.find('<');
  size_t end = input.rfind('>');

  if (start == std::string::npos || end == std::string::npos || start >= end) {
    return result;
  }
  std::string inner = input.substr(start + 1, end - start - 1);
  std::vector<std::string> params;
  std::stringstream ss(inner);
  std::string param;
  while (std::getline(ss, param, ',')) {
    param.erase(0, param.find_first_not_of(" \t"));
    param.erase(param.find_last_not_of(" \t") + 1);
    params.push_back(param);
  }
  std::vector<std::string> paramNames = {
      "data_type_input", "data_type_output", "L1_BLOCK_M", "L1_BLOCK_N",
      "L1_BLOCK_K",      "BLOCK_M",          "BLOCK_N",    "L1_BLOCK_K",
      "transpose_A",     "transpose_B"};
  for (size_t i = 0; i < params.size() && i < paramNames.size(); ++i) {
    result[paramNames[i]] = params[i];
  }
  for (size_t i = paramNames.size(); i < params.size(); ++i) {
    result["extra_param_" + std::to_string(i - paramNames.size() + 1)] =
        params[i];
  }
  return result;
}

std::vector<std::string> extractShapeFromTemplate(const std::string &input) {
  std::vector<std::string> numbers;
  size_t start = input.find('<');
  if (start == std::string::npos) {
    return numbers;
  }
  size_t end = input.find('>', start);
  if (end == std::string::npos) {
    return numbers;
  }
  std::string templatePart = input.substr(start + 1, end - start - 1);
  templatePart.erase(std::remove(templatePart.begin(), templatePart.end(), ' '),
                     templatePart.end());
  std::vector<std::string> parts;
  std::stringstream ss(templatePart);
  std::string token;
  while (std::getline(ss, token, ',')) {
    parts.push_back(token);
  }
  for (size_t i = 1; i < parts.size(); ++i) {
    bool isNumber = !parts[i].empty() &&
                    std::all_of(parts[i].begin(), parts[i].end(), ::isdigit);
    if (isNumber) {
      numbers.push_back(parts[i]);
    }
  }
  return numbers;
}

int GetValidShape(int shape, std::string &dtype) {
  int dtype_len = GetTypeLen(dtype);
  int shape_mod = shape * GetTypeLen(dtype) % 32;
  if (shape_mod == 0) {
    return shape;
  }
  return shape + (32 - shape_mod) / dtype_len;
}

void CodeGenTileLangAscendPto::VisitExpr_(const CallNode *op,
                                          std::ostream &os) {
  if (op->op.same_as(builtin::call_extern())) {
    CallExternCodegen(op);
  } else if (op->op.same_as(tl::loop_break())) {
    this->PrintIndent();
    this->stream << "break;\n";
  } else if (op->op.same_as(tl::ascend_gemm_v0())) {
    GemmV0Codegen(op);
  } else if (op->op.same_as(tl::ascend_gemm_v1())) {
    GemmV1Codegen(op);
  } else if (op->op.same_as(tl::ascend_fill())) {
    FillCodegen(op);
  } else if (op->op.same_as(tl::ascend_exp())) {
    UnaryVecOpCodegen(op, "TEXP");
  } else if (op->op.same_as(tl::ascend_ln())) {
    UnaryVecOpCodegen(op, "TLOG");
  } else if (op->op.same_as(tl::ascend_abs())) {
    UnaryVecOpCodegen(op, "TABS");
  } else if (op->op.same_as(tl::ascend_reciprocal())) {
    UnaryVecOpCodegen(op, "TRECIP");
  } else if (op->op.same_as(tl::ascend_sqrt())) {
    UnaryVecOpCodegen(op, "TSQRT");
  } else if (op->op.same_as(tl::ascend_rsqrt())) {
    UnaryVecOpCodegen(op, "TRSQRT");
  } else if (op->op.same_as(tl::ascend_relu())) {
    UnaryVecOpCodegen(op, "TRELU");
  } else if (op->op.same_as(tl::ascend_bitwise_not())) {
    UnaryVecOpCodegen(op, "TNOT");
  } else if (op->op.same_as(tl::ascend_leaky_relu())) {
    ScalarOpCodegen(op, "TLRELU");
  } else if (op->op.same_as(tl::ascend_axpy())) {
    AxpyCodegen(op);
  } else if (op->op.same_as(tl::ascend_reduce())) {
    ReduceOpCodegen(op);
  } else if (op->op.same_as(tl::ascend_add())) {
    BinaryVecOpCodegen(op, "TADD");
  } else if (op->op.same_as(tl::ascend_sub())) {
    BinaryVecOpCodegen(op, "TSUB");
  } else if (op->op.same_as(tl::ascend_mul())) {
    BinaryVecOpCodegen(op, "TMUL");
  } else if (op->op.same_as(tl::ascend_div())) {
    BinaryVecOpCodegen(op, "TDIV");
  } else if (op->op.same_as(tl::ascend_max())) {
    BinaryVecOpCodegen(op, "TMAX");
  } else if (op->op.same_as(tl::ascend_min())) {
    BinaryVecOpCodegen(op, "TMIN");
  } else if (op->op.same_as(tl::ascend_bitwise_and())) {
    BinaryVecOpCodegen(op, "tand");
  } else if (op->op.same_as(tl::ascend_bitwise_or())) {
    BinaryVecOpCodegen(op, "tor");
  } else if (op->op.same_as(tl::ascend_adds())) {
    BinaryVecOpsCodegen(op, "TADDS");
  } else if (op->op.same_as(tl::ascend_subs())) {
    BinaryVecOpsCodegen(op, "TSUBS");
  } else if (op->op.same_as(tl::ascend_muls())) {
    BinaryVecOpsCodegen(op, "TMULS");
  } else if (op->op.same_as(tl::ascend_divs())) {
    BinaryVecOpsCodegen(op, "TDIVS");
  } else if (op->op.same_as(tl::ascend_maxs())) {
    BinaryVecOpsCodegen(op, "TMAXS");
  } else if (op->op.same_as(tl::ascend_mins())) {
    BinaryVecOpsCodegen(op, "TMINS");
  } else if (op->op.same_as(tl::ascend_sync_all())) {
    SyncAllCodegen(op);
  } else if (op->op.same_as(tl::ascend_pipe_barrier())) {
    PipeBarrierCodegen(op);
  } else if (op->op.same_as(tl::ascend_set_flag())) {
    SetAndWaitFlagCodegen(op, "set_flag");
  } else if (op->op.same_as(tl::ascend_wait_flag())) {
    SetAndWaitFlagCodegen(op, "wait_flag");
  } else if (op->op.same_as(tl::ascend_set_cross_flag())) {
    SetCrossFlagCodegen(op);
  } else if (op->op.same_as(tl::ascend_wait_cross_flag())) {
    WaitCrossFlagCodegen(op);
  } else if (op->op.same_as(tl::ascend_auto_set_flag())) {
    AutoFlagOpCodegen(op, "set_flag");
  } else if (op->op.same_as(tl::ascend_auto_wait_flag())) {
    AutoFlagOpCodegen(op, "wait_flag");
  } else if (op->op.same_as(tl::ascend_auto_set_cross_flag())) {
    AutoSetCrossFlagCodegen(op);
  } else if (op->op.same_as(tl::ascend_auto_wait_cross_flag())) {
    WaitCrossFlagCodegen(op);
  } else if (op->op.same_as(tl::ascend_auto_barrier())) {
    AutoBarrierCodegen(op);
  } else if (op->op.same_as(tl::ascend_clamp_max())) {
    BinaryVecClampMaxMinOpsCodegen(op, "TMINS");
  } else if (op->op.same_as(tl::ascend_clamp_min())) {
    BinaryVecClampMaxMinOpsCodegen(op, "TMAXS");
  } else if (op->op.same_as(tl::ascend_clamp())) {
    BinaryVecClampOpsCodegen(op, "TCLAMP");
  } else if (op->op.same_as(tl::ascend_sigmoid())) {
    SigmoidCodegen(op, "TSIGMOID");
  } else if (op->op.same_as(tl::ascend_gather_mask())) {
    GatherMaskCodegen(op, "TGATHER");
  } else if (op->op.same_as(tl::ascend_round())) {
    CastCodegen(op, "RoundMode::CAST_ROUND");
  } else if (op->op.same_as(tl::ascend_cast())) {
    std::string cast_type = op->args[2].as<StringImmNode>()->value;
    if (cast_type == "CAST_NONE") {
      CastCodegen(op, "RoundMode::CAST_NONE");
    } else if (cast_type == "CAST_RINT") {
      CastCodegen(op, "RoundMode::CAST_RINT");
    } else if (cast_type == "CAST_FLOOR") {
      CastCodegen(op, "RoundMode::CAST_FLOOR");
    } else if (cast_type == "CAST_CEIL") {
      CastCodegen(op, "RoundMode::CAST_CEIL");
    } else if (cast_type == "CAST_ROUND") {
      CastCodegen(op, "RoundMode::CAST_ROUND");
    } else if (cast_type == "CAST_TRUNC") {
      CastCodegen(op, "RoundMode::CAST_TRUNC");
    } else if (cast_type == "CAST_ODD") {
      CastCodegen(op, "RoundMode::CAST_ODD");
    }
  } else if (op->op.same_as(tl::ascend_createvecindex())) {
    CreateVecIndexCodegen(op, "TCI");
  } else if (op->op.same_as(tl::ascend_gatherb())) {
    GatherbCodegen(op, "TGATHERB");
  } else if (op->op.same_as(tl::ascend_pow())) {
    PowCodegen(op);
  } else if (op->op.same_as(tl::ascend_sort32())) {
    Sort32Codegen(op, "TSORT32");
  } else if (op->op.same_as(tl::ascend_transpose())) {
    TransposeCodegen(op, "TTRANS");
  } else if (op->op.same_as(tl::ascend_bitwise_xor())) {
    XorCodegen(op, "TXOR");
  } else if (op->op.same_as(tl::ascend_compare())) {
    CompareCodegen(op, "TCMP");
  } else if (op->op.same_as(tl::ascend_compare_scalar())) {
    CompareScalarCodegen(op, "TCMPS");
  } else if (op->op.same_as(tl::ascend_bitwise_lshift())) {
    TshCodegen(op, "TSHLS");
  } else if (op->op.same_as(tl::ascend_bitwise_rshift())) {
    TshCodegen(op, "TSHRS");
  } else if (op->op.same_as(tl::ascend_arith_progression())) {
    ArithProgressionCodegen(op, "TCI");
  } else if (op->op.same_as(tl::ascend_broadcast())) {
    BroadcastOpCodegen(op);
  } else if (op->op.same_as(tl::ascend_select())) {
    SelectCodegen(op);
  } else if (op->op.same_as(tl::ascend_dump_tensor())) {
    DumpTensorCodegen(op, "TPRINT");
  } else if (op->op.same_as(tl::ascend_printf())) {
    PrintfOpCodegen(op, "cce::printf");
  } else if (op->op.same_as(tl::ascend_set_deq_scale())) {
    SetDeqScaleCodegen(op);
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

std::string CodeGenTileLangAscendPto::PrintBufferOffset(const CallNode *op) {
  auto _var = op->args[1].as<VarNode>();
  std::string _var_name = var_idmap_[_var];
  return _var_name;
}

std::vector<std::string>
CodeGenTileLangAscendPto::GetGlobalTensorShapes(const CallNode *op,
                                                std::string tensor_addr) {
  auto srcN = PrintExpr(op->args[3]);
  int count = 0;
  auto srcN_tmp = 0;
  if (srcN[0] >= '1' && srcN[0] <= '9') {
    srcN_tmp = std::stoi(srcN);
    for (int i =
             global_tensor_template[String(tensor_addr)].shape_list.size() - 1;
         i >= 0; i--) {
      count++;
      srcN_tmp =
          srcN_tmp /
          std::stoi(global_tensor_template[String(tensor_addr)].shape_list[i]);
      if (srcN_tmp == 1) {
        break;
      }
    }
  }
  std::vector<std::string> global_tensor_shapes;
  auto size = global_tensor_template[String(tensor_addr)].shape_list.size();
  for (int i = 0; i < size; i++) {
    global_tensor_shapes.push_back(
        global_tensor_template[String(tensor_addr)].shape_list[i]);
  }
  if (count == 2) {
    global_tensor_shapes.pop_back();
    global_tensor_shapes.pop_back();
    global_tensor_shapes.push_back(srcN);
  } else if (count == 3) {
    global_tensor_shapes.pop_back();
    global_tensor_shapes.pop_back();
    global_tensor_shapes.pop_back();
    global_tensor_shapes.push_back(srcN);
  }
  return global_tensor_shapes;
}

void CodeGenTileLangAscendPto::CallExternCodegen(const CallNode *op) {
  std::string op_name = Downcast<StringImm>(op->args[0])->value;
  if (op_name.find("tl::ascend::copy") != std::string::npos) {
    auto src_var = op->args[1].as<CallNode>()->args[1].as<VarNode>();
    auto dst_var = op->args[2].as<CallNode>()->args[1].as<VarNode>();

    auto src_var_id = var_idmap_[src_var];
    auto dst_var_id = var_idmap_[dst_var];
    if (src_var_id == "") {
      src_var_id = src_var->name_hint;
    }
    if (dst_var_id == "") {
      dst_var_id = dst_var->name_hint;
    }

    auto src_offset = PrintExpr(op->args[1].as<CallNode>()->args[2]);
    auto dst_offset = PrintExpr(op->args[2].as<CallNode>()->args[2]);

    auto src_shape = PrintExpr(op->args[1].as<CallNode>()->args[3]);
    auto dst_shape = PrintExpr(op->args[2].as<CallNode>()->args[3]);

    auto src_type = GetAccessPtrDtypePto(op->args[1].as<CallNode>());
    auto dst_type = GetAccessPtrDtypePto(op->args[2].as<CallNode>());

    static const std::unordered_map<std::string, int> kCopyOpExtraArgs = {
        {"copy_l0c_to_gm", 1}, {"copy_gm_to_l1", 1}, {"copy_l1_to_l0a", 3},
        {"copy_l1_to_l0b", 3}, {"copy_gm_to_ub", 1}, {"copy_ub_to_gm", 1},
        {"copy_ub_to_ub", 0}};

    std::unordered_map<std::string, std::string> ptoCopyMap = {
        {"copy_l0c_to_gm", "TSTORE"},   {"copy_gm_to_l1", "TLOAD"},
        {"copy_l1_to_l0a", "TEXTRACT"}, {"copy_l1_to_l0b", "TEXTRACT"},
        {"copy_gm_to_ub", "TLOAD"},     {"copy_ub_to_gm", "TSTORE"},
        {"copy_ub_to_ub", "TCVT"}};

    bool found = false;
    int extra_args = 0;
    std::string real_name = "";

    for (const auto &pair : kCopyOpExtraArgs) {
      if (op_name.find(pair.first) != std::string::npos) {
        real_name = pair.first;
        found = true;
        extra_args = pair.second;
        break;
      }
    }

    if (found) {
      auto api_name = ptoCopyMap[real_name];
      PrimExpr row_index;
      PrimExpr col_index;
      if (api_name == "TCVT") {
        api_name = src_type == dst_type ? "TMOV" : "TCVT";
      } else if (api_name == "TEXTRACT") {
        if (op->args.size() >= 3 && op->args[5].as<IntImmNode>()->value != 0) {
          row_index = op->args[4];
          col_index = op->args[3];
        } else {
          api_name = "TCVT";
        }
      }
      if (api_name == "TEXTRACT") {
        this->PrintIndent();
        this->stream << api_name << "(" << dst_var_id << ", " << src_var_id
                     << ", " << row_index << ", " << col_index << ");\n";
      } else if (api_name == "TCVT") {
        ShapeInfo src_shape_info = GetSliceInfo(op->args[1].as<CallNode>());
        ShapeInfo dst_shape_info = GetSliceInfo(op->args[2].as<CallNode>());
        if (src_shape_info.is_slice || dst_shape_info.is_slice) {
          std::string src_temp_name = GetTempVarName(src_shape_info.ub_name);
          std::string dst_temp_name = GetTempVarName(dst_shape_info.ub_name);
          CreateUbVariableND(src_temp_name, src_shape_info);
          CreateUbVariableND(dst_temp_name, dst_shape_info);
          this->PrintIndent();
          this->stream << api_name << "(" << dst_temp_name << ", "
                       << src_temp_name << ", " << "pto::RoundMode::CAST_NONE"
                       << ");\n";
        } else {
          this->PrintIndent();
          this->stream << api_name << "(" << dst_var_id << ", " << src_var_id
                       << ", pto::RoundMode::CAST_NONE" << ");\n";
        }
      } else if (api_name == "TMOV") {
        std::vector<std::string> var_names;
        for (int i = 1; i < op->args.size(); i++) {
          auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
          var_names.push_back(var_name);
        }
        ShapeInfo src_shape_info = GetSliceInfo(op->args[1].as<CallNode>());
        ShapeInfo dst_shape_info = GetSliceInfo(op->args[2].as<CallNode>());
        if (src_shape_info.is_slice || dst_shape_info.is_slice) {
          std::string src_temp_name = GetTempVarName(src_shape_info.ub_name);
          std::string dst_temp_name = GetTempVarName(dst_shape_info.ub_name);
          CreateUbVariableND(src_temp_name, src_shape_info);
          CreateUbVariableND(dst_temp_name, dst_shape_info);
          this->PrintIndent();
          this->stream << api_name << "(" << dst_temp_name << ", "
                       << src_temp_name << ");\n";
        } else {
          this->PrintIndent();
          this->stream << api_name << "(";
          if (prefetch_n_stages_map_[var_names[1]].first > 0) {
            PrimExpr element_count_src0;
            auto shape_src0 = buffer_shapess_[GetRef<tir::Var>(
                op->args[1].as<CallNode>()->args[1].as<VarNode>())];
            if (shape_src0.size() == 3) {
              element_count_src0 = shape_src0[1] * shape_src0[2];
            } else if (shape_src0.size() == 2) {
              element_count_src0 = shape_src0[0] * shape_src0[1];
            } else if (shape_src0.size() == 1) {
              element_count_src0 = shape_src0[0];
            } else {
              ICHECK(false)
                  << "An error occurred. Please check prefetch_n_stages_map_, "
                     "buffer_shapes_, and buffer_versions_.";
            }
            auto buffer_k_src0 =
                op->args[2].as<CallNode>()->args[2] / element_count_src0;
            tvm::arith::Analyzer analyzer;
            PrimExpr simplified_buffer_k_src0 =
                analyzer.Simplify(buffer_k_src0);
            this->stream << var_names[1] << "["
                         << PrintExpr(simplified_buffer_k_src0) << "], ";
          } else {
            this->stream << var_names[1] << ", ";
          }
          if (prefetch_n_stages_map_[var_names[0]].first > 0) {
            PrimExpr element_count_dst;
            auto shape_dst = buffer_shapess_[GetRef<tir::Var>(
                op->args[0].as<CallNode>()->args[1].as<VarNode>())];
            if (shape_dst.size() == 3) {
              element_count_dst = shape_dst[1] * shape_dst[2];
            } else if (shape_dst.size() == 2) {
              element_count_dst = shape_dst[0] * shape_dst[1];
            } else if (shape_dst.size() == 1) {
              element_count_dst = shape_dst[0];
            } else {
              ICHECK(false)
                  << "An error occurred. Please check prefetch_n_stages_map_, "
                     "buffer_shapes_, and buffer_versions_.";
            }
            auto buffer_k_dst =
                op->args[1].as<CallNode>()->args[2] / element_count_dst;
            tvm::arith::Analyzer analyzer;
            PrimExpr simplified_buffer_k_dst = analyzer.Simplify(buffer_k_dst);
            this->stream << var_names[0] << "["
                         << (PrintExpr(simplified_buffer_k_dst)) << "]);\n";
          } else {
            this->stream << var_names[0] << ");\n";
          }
        }
      }
      if (api_name == "TLOAD") {
        ICHECK((copy_base_addr_map_.find(String(src_var_id)) !=
                copy_base_addr_map_.end()));
        std::vector<std::string> l_valid_shapes = l_data_map_[dst_var_id];
        std::vector<std::string> ub_valid_shapes = ub_data_map_[dst_var_id];
        std::vector<std::string> dynamic_names;
        std::string tensor_addr = copy_base_addr_map_[String(src_var_id)];
        std::string tensor_template =
            "<" + global_tensor_template[String(tensor_addr)].dtype;
        std::string shape_template = "", stride_template = "",
                    valid_template = "";
        size_t shape_len = 2;
        size_t op_arg_len = op->args.size();
        size_t shape_size = 5;
        // Dynamic Shape and Static Shape

        // generate shape
        std::vector shape_tile = extractShapeFromTemplate(op_name);
        std::vector<std::string> shape_nums(shape_len);
        bool is_chunking = false;

        if (shape_tile[0] != PrintExpr(op->args[op_arg_len - 1]) &&
            op_name.find("copy_gm_to_ub") != std::string::npos) {
          ub_valid_shapes[2] = shape_tile[0];
          is_chunking = true;
        }
        shape_nums[1] = PrintExpr(op->args[op_arg_len - 1]);
        if (op_arg_len == 5) {
          shape_nums[0] = "1";
        } else if (shape_tile[1] != PrintExpr(op->args[op_arg_len - 2]) &&
                   op_name.find("copy_gm_to_ub") != std::string::npos) { //
          is_chunking = true;
          ub_valid_shapes[1] = shape_tile[1];
          shape_nums[0] = PrintExpr(op->args[op_arg_len - 2]);
        } else {
          shape_nums[0] = PrintExpr(op->args[op_arg_len - 2]);
        }
        for (size_t i = 0; i < shape_size; i++) {
          if (i < shape_size - shape_len) {
            shape_template += "1";
          } else {
            if (is_chunking) {
              shape_template += ub_valid_shapes[i + shape_len - shape_size + 1];
            } else {
              shape_template += shape_nums[i + shape_len - shape_size];
            }
          }
          if (i < shape_size - 1) {
            shape_template += ", ";
          }
        }
        std::vector<std::string> global_tensor_shapes =
            GetGlobalTensorShapes(op, tensor_addr);
        size_t len = global_tensor_shapes.size();
        for (size_t i = 0; i < 4; i++) {
          if (len > 3 - i) {
            std::string tensor_template = global_tensor_shapes[len + i - 4];
            if (tensor_template[0] < '1' || tensor_template[0] > '9') {
              stride_template += "-1, ";
              dynamic_names.push_back(tensor_template);
            } else {
              std::string tmp_shape = "";
              for (size_t j = 0; j < 4 - i; j++) {
                tmp_shape += global_tensor_shapes[len - j - 1];
                if (j < 3 - i)
                  tmp_shape += " * ";
              }
              stride_template = stride_template + tmp_shape + ", ";
            }
          } else {
            stride_template += "1, ";
          }
        }
        stride_template += "1";
        // get gm2l1 shape

        bool is_dynamic =
            global_tensor_template[String(tensor_addr)].shape_type == "dynamic";
        std::string src_var = "";
        if (op_name.find("copy_gm_to_l1") != std::string::npos) {
          src_var = "copy_gm_to_l1";
          if (is_dynamic) {
            src_var = src_var + "_dynamic";
          }
          tensor_template = tensor_template + ", " + l_valid_shapes[0] + ", ";
          valid_template = l_valid_shapes[1] + ", " + l_valid_shapes[2];
        } else if (op_name.find("copy_gm_to_ub") != std::string::npos) {
          src_var = "copy_gm_to_ub";
          if (is_dynamic) {
            src_var = src_var + "_dynamic";
          }
          tensor_template = tensor_template + ", " + ub_valid_shapes[0] + ", ";
          valid_template = shape_nums[0] + ", " + shape_nums[1] + ", " +
                           ub_valid_shapes[1] + ", " + ub_valid_shapes[2];
        }
        tensor_template = tensor_template + shape_template + ", " +
                          stride_template + ", " + valid_template + ">";
        this->PrintIndent();
        this->stream << kAscendPtoScope << src_var << tensor_template << "("
                     << tensor_addr << " + " << src_offset;
        if (is_dynamic) {
          std::string shape = "pto::Shape<" + shape_template + ">()";
          this->stream << ", " << "pto::Shape<" << shape_template << ">"
                       << "(), " << "pto::Stride<" << stride_template << ">"
                       << "(";
          for (size_t i = 0; i < dynamic_names.size(); i++) {
            this->stream << dynamic_names[i];
            if (i != dynamic_names.size() - 1) {
              this->stream << ", ";
            }
          }
          this->stream << ")";
        }
        // for the pipeline scenario of l1
        if (api_name == "TLOAD" && prefetch_n_stages_map_.count(dst_var_id) &&
            prefetch_n_stages_map_[dst_var_id].first > 0 &&
            op_name.find("copy_gm_to_l1") != std::string::npos) {
          PrimExpr element_count;
          auto shape = buffer_shapess_[GetRef<tir::Var>(dst_var)];
          if (shape.size() == 3) {
            element_count = shape[1] * shape[2];
          } else if (shape.size() == 2) {
            element_count = shape[0] * shape[1];
          } else {
            ICHECK(false)
                << "An error occurred. Please check prefetch_n_stages_map_, "
                   "buffer_shapes_, and buffer_versions_.";
          }
          auto buffer_k = op->args[2].as<CallNode>()->args[2] / element_count;
          tvm::arith::Analyzer analyzer;
          PrimExpr simplified_k = analyzer.Simplify(buffer_k);
          this->stream << ", " << dst_var_id << "[" << simplified_k << "]"
                       << ");\n";

          prefetch_n_stages_map_[dst_var_id].second++;
        } else {
          if (op_name.find("copy_gm_to_ub") != std::string::npos) {
            int32_t type_len =
                GetTypeLen(global_tensor_template[String(tensor_addr)].dtype);
            this->stream << ", " << ub_valid_shapes[3] << ", " << dst_offset
                         << "," << type_len << ");\n";
          } else {
            this->stream << ", " << dst_var_id << ");\n";
          }
        }

      } else if (api_name == "TSTORE") {
        ICHECK((copy_base_addr_map_.find(String(dst_var_id)) !=
                copy_base_addr_map_.end()));
        std::vector<std::string> l_valid_shapes = l_data_map_[src_var_id];
        std::vector<std::string> ub_valid_shapes = ub_data_map_[src_var_id];
        std::vector<std::string> dynamic_names;
        std::string tensor_addr = copy_base_addr_map_[String(dst_var_id)];
        std::string tensor_template =
            "<" + global_tensor_template[String(tensor_addr)].dtype;
        std::string shape_template = "", stride_template = "",
                    valid_template = "";
        size_t len =
            global_tensor_template[String(tensor_addr)].shape_list.size();
        size_t shape_len = 2;
        size_t op_arg_len = op->args.size();
        size_t shape_size = 5;
        // Dynamic Shape and Static Shape

        // generate shape
        std::vector shape_tile = extractShapeFromTemplate(op_name);
        std::vector<std::string> shape_nums(shape_len);
        bool is_chunking = false;

        if (shape_tile[0] != PrintExpr(op->args[op_arg_len - 1]) &&
            op_name.find("copy_ub_to_gm") != std::string::npos) {
          ub_valid_shapes[2] = shape_tile[0];
          is_chunking = true;
        }
        shape_nums[1] = PrintExpr(op->args[op_arg_len - 1]);
        if (op_arg_len == 5) {
          shape_nums[0] = "1";
        } else if (shape_tile[1] != PrintExpr(op->args[op_arg_len - 2]) &&
                   op_name.find("copy_ub_to_gm") != std::string::npos) { //
          is_chunking = true;
          ub_valid_shapes[1] = shape_tile[1];
          shape_nums[0] = PrintExpr(op->args[op_arg_len - 2]);
        } else {
          shape_nums[0] = PrintExpr(op->args[op_arg_len - 2]);
        }
        for (size_t i = 0; i < shape_size; i++) {
          if (i < shape_size - shape_len) {
            shape_template += "1";
          } else {
            if (is_chunking) {
              shape_template += ub_valid_shapes[i + shape_len - shape_size + 1];
            } else {
              shape_template += shape_nums[i + shape_len - shape_size];
            }
          }
          if (i < shape_size - 1) {
            shape_template += ", ";
          }
        }

        for (size_t i = 0; i < 4; i++) {
          if (len > 3 - i) {
            std::string tensor_template =
                global_tensor_template[String(tensor_addr)]
                    .shape_list[len + i - 4];
            if (tensor_template[0] < '1' || tensor_template[0] > '9') {
              stride_template += "-1, ";
              dynamic_names.push_back(tensor_template);
            } else {
              std::string tmp_shape = "";
              for (size_t j = 0; j < 4 - i; j++) {
                tmp_shape += global_tensor_template[String(tensor_addr)]
                                 .shape_list[len - j - 1];
                if (j < 3 - i)
                  tmp_shape += " * ";
              }
              stride_template = stride_template + tmp_shape + ", ";
            }
          } else {
            stride_template += "1, ";
          }
        }
        stride_template += "1";

        // get gm2l1 shape
        bool is_dynamic =
            global_tensor_template[String(tensor_addr)].shape_type == "dynamic";
        std::string src_var = "";
        if (op_name.find("copy_l0c_to_gm") != std::string::npos) {
          src_var = "copy_l0c_to_gm";
          if (is_dynamic) {
            src_var = src_var + "_dynamic";
          }
          tensor_template = tensor_template + ", " + l_valid_shapes[0] + ", ";
          valid_template = l_valid_shapes[1] + ", " + l_valid_shapes[2];
        } else if (op_name.find("copy_ub_to_gm") != std::string::npos) {
          src_var = "copy_ub_to_gm";
          if (is_dynamic) {
            src_var = src_var + "_dynamic";
          }
          tensor_template = tensor_template + ", " + ub_valid_shapes[0] + ", ";
          valid_template = "";
        }
        // tensor_template = tensor_template + shape_template + ", " +
        // stride_template + ", " + valid_template;
        if (op_name.find("copy_ub_to_gm") != std::string::npos) {
          std::string shape_num_dtype =
              global_tensor_template[String(tensor_addr)].dtype;
          int num1 = std::stoi(shape_nums[1]);
          // when shape_nums[1] * sizeof(dtype) < 32, do this
          if (shape_num_dtype == "int" && num1 < 8) {
            shape_nums[1] = "8";
          } else if (shape_num_dtype == "float" && num1 < 8) {
            shape_nums[1] = "8";
          } else if (shape_num_dtype == "half" && num1 < 16) {
            shape_nums[1] = "16";
          }
          tensor_template = tensor_template + shape_template + ", " +
                            stride_template + ", " + shape_nums[0] + ", " +
                            shape_nums[1] + ", " + ub_valid_shapes[1] + ", " +
                            ub_valid_shapes[2] + ">";
        } else {
          tensor_template = tensor_template + shape_template + ", " +
                            stride_template + ", " + valid_template + ">";
        }
        this->PrintIndent();
        this->stream << kAscendPtoScope << src_var << tensor_template << "("
                     << tensor_addr << " + " << dst_offset;
        if (is_dynamic) {
          this->stream << ", " << "pto::Shape<" << shape_template << ">"
                       << "(), " << "pto::Stride<" << stride_template << ">"
                       << "(";
          for (size_t i = 0; i < dynamic_names.size(); i++) {
            this->stream << dynamic_names[i];
            if (i != dynamic_names.size() - 1) {
              this->stream << ", ";
            }
          }
          this->stream << ")";
        }
        if (op_name.find("copy_ub_to_gm") != std::string::npos) {
          this->stream << ", " << ub_valid_shapes[3] << ", " << src_offset
                       << ");\n";
        } else {
          this->stream << ", " << src_var_id << ");\n";
        }
      }
    } else {
      this->PrintIndent();
      this->stream << "not implemented yet\n";
    }
  }
}

void CodeGenTileLangAscendPto::GemmV0Codegen(const CallNode *op) {
  std::string op_name = Downcast<StringImm>(op->args[0])->value;
  this->PrintIndent();
  auto a_var = op->args[1].as<CallNode>()->args[1].as<VarNode>();
  auto b_var = op->args[2].as<CallNode>()->args[1].as<VarNode>();
  auto c_var = op->args[3].as<CallNode>()->args[1].as<VarNode>();

  auto a_offset = PrintExpr(op->args[1].as<CallNode>()->args[2]);
  auto b_offset = PrintExpr(op->args[2].as<CallNode>()->args[2]);
  auto c_offset = PrintExpr(op->args[3].as<CallNode>()->args[2]);

  auto a_name = var_idmap_[a_var];
  auto b_name = var_idmap_[b_var];
  auto c_name = var_idmap_[c_var];

  std::map<std::string, std::string> params = extractTemplateParams(op_name);
  uint32_t K = std::stoi(params["K"]);
  uint32_t kL0Size = 128; // L0切片大小，适配64K内存限制
  uint32_t kL0split = (K + kL0Size - 1) / kL0Size; // 切片数量
  uint32_t kL0Tail = K - (kL0split - 1) * kL0Size; // 最后一块大小

  auto a_shape = buffer_shapess_[GetRef<tir::Var>(a_var)];
  auto b_shape = buffer_shapess_[GetRef<tir::Var>(b_var)];
  auto c_shape = buffer_shapess_[GetRef<tir::Var>(c_var)];

  bool a_buffer_in_pipeline = false;
  if (buffer_versions_.count(GetRef<tir::Var>(a_var))) {
    if (!is_one(buffer_versions_[GetRef<tir::Var>(a_var)])) {
      a_buffer_in_pipeline = true;
    }
  } else if (a_shape.size() == 3) {
    a_buffer_in_pipeline = true;
  }

  bool b_buffer_in_pipeline = false;
  if (buffer_versions_.count(GetRef<tir::Var>(b_var))) {
    if (!is_one(buffer_versions_[GetRef<tir::Var>(b_var)])) {
      b_buffer_in_pipeline = true;
    }
  } else if (b_shape.size() == 3) {
    b_buffer_in_pipeline = true;
  }

  bool c_buffer_in_pipeline = false;
  if (buffer_versions_.count(GetRef<tir::Var>(c_var))) {
    if (!is_one(buffer_versions_[GetRef<tir::Var>(c_var)])) {
      c_buffer_in_pipeline = true;
    }
  } else if (c_shape.size() == 3) {
    c_buffer_in_pipeline = true;
  }

  // std::map<std::string, std::string> params = extractTemplateParams(op_name);
  std::string data_type_input = params["data_type_input"];
  this->stream << kAscendPtoScope << "gemm_v0" << "<"
               << params["data_type_input"] << ", "
               << params["data_type_output"] << ", "
               << GetValidShape(std::stoi(params["M"]), data_type_input) << ", "
               << GetValidShape(std::stoi(params["N"]), data_type_input) << ", "
               << GetValidShape(std::stoi(params["K"]), data_type_input) << ", "
               << params["M"] << ", " << params["N"] << ", " << params["K"]
               << ", " << kL0Tail << ", " << params["transpose_A"] << ", "
               << params["transpose_B"] << ">" << "(";

  if (a_buffer_in_pipeline) {
    auto a_k = op->args[2].as<CallNode>()->args[2] /
               op->args[2].as<CallNode>()->args[3];
    tvm::arith::Analyzer analyzer;
    PrimExpr simplified_a_k = analyzer.Simplify(a_k);
    auto buffer_name = a_name + "[" + PrintExpr(simplified_a_k) + "]";

    this->stream << buffer_name << ", ";
  } else {
    this->stream << a_name << ", ";
  }

  if (b_buffer_in_pipeline) {
    auto b_k = op->args[2].as<CallNode>()->args[2] /
               op->args[2].as<CallNode>()->args[3];
    tvm::arith::Analyzer analyzer;
    PrimExpr simplified_b_k = analyzer.Simplify(b_k);
    auto buffer_name = b_name + "[" + PrintExpr(simplified_b_k) + "]";

    this->stream << buffer_name << ", ";
  } else {
    this->stream << b_name << ", ";
  }

  if (c_buffer_in_pipeline) {
    auto c_k = op->args[2].as<CallNode>()->args[2] /
               op->args[2].as<CallNode>()->args[3];
    tvm::arith::Analyzer analyzer;
    PrimExpr simplified_c_k = analyzer.Simplify(c_k);
    auto buffer_name = c_name + "[" + PrintExpr(simplified_c_k) + "]";

    this->stream << buffer_name << ", ";
  } else {
    this->stream << c_name << ", ";
  }

  this->stream << PrintExpr(op->args[4]) << ");\n";
}

void CodeGenTileLangAscendPto::GemmV1Codegen(const CallNode *op) {
  std::string op_name = Downcast<StringImm>(op->args[0])->value;
  this->PrintIndent();
  auto a_var = op->args[1].as<CallNode>()->args[1].as<VarNode>();
  auto b_var = op->args[2].as<CallNode>()->args[1].as<VarNode>();
  auto c_var = op->args[3].as<CallNode>()->args[1].as<VarNode>();

  auto a_offset = PrintExpr(op->args[1].as<CallNode>()->args[2]);
  auto b_offset = PrintExpr(op->args[2].as<CallNode>()->args[2]);
  auto c_offset = PrintExpr(op->args[3].as<CallNode>()->args[2]);

  auto a_name = var_idmap_[a_var];
  auto b_name = var_idmap_[b_var];
  auto c_name = var_idmap_[c_var];

  std::map<std::string, std::string> params = extractTemplateParams1(op_name);
  if (prefetch_n_stages_map_[a_name].first > 0) {
    auto a_k = op->args[1].as<CallNode>()->args[2] /
               op->args[1].as<CallNode>()->args[3];
    tvm::arith::Analyzer analyzer;
    PrimExpr simplified_a_k = analyzer.Simplify(a_k);
    auto b_k = op->args[2].as<CallNode>()->args[2] /
               op->args[2].as<CallNode>()->args[3];
    PrimExpr simplified_b_k = analyzer.Simplify(b_k);
    std::string data_type_input = params["data_type_input"];
    this->stream
        << kAscendPtoScope << "gemm_v1" << "<" << params["data_type_input"]
        << ", " << params["data_type_output"] << ", "
        << GetValidShape(std::stoi(params["L1_BLOCK_M"]), data_type_input)
        << ", "
        << GetValidShape(std::stoi(params["L1_BLOCK_N"]), data_type_input)
        << ", "
        << GetValidShape(std::stoi(params["L1_BLOCK_K"]), data_type_input)
        << ", " << GetValidShape(std::stoi(params["BLOCK_M"]), data_type_input)
        << ", " << GetValidShape(std::stoi(params["BLOCK_N"]), data_type_input)
        << ", "
        << GetValidShape(std::stoi(params["L1_BLOCK_K"]), data_type_input)
        << ", " << params["L1_BLOCK_M"] << ", " << params["L1_BLOCK_N"] << ", "
        << params["L1_BLOCK_K"] << ", " << params["BLOCK_M"] << ", "
        << params["BLOCK_N"] << ", " << params["L1_BLOCK_K"] << ", "
        << params["transpose_A"] << ", " << params["transpose_B"] << ">"
        << "(" << a_name << "[" << simplified_a_k << "], " << b_name << "["
        << simplified_a_k << "], " << c_name << ", " << PrintExpr(op->args[4])
        << ");\n";
  } else {
    std::string data_type_input = params["data_type_input"];
    this->stream
        << kAscendPtoScope << "gemm_v1" << "<" << params["data_type_input"]
        << ", " << params["data_type_output"] << ", "
        << GetValidShape(std::stoi(params["L1_BLOCK_M"]), data_type_input)
        << ", "
        << GetValidShape(std::stoi(params["L1_BLOCK_N"]), data_type_input)
        << ", "
        << GetValidShape(std::stoi(params["L1_BLOCK_K"]), data_type_input)
        << ", " << GetValidShape(std::stoi(params["BLOCK_M"]), data_type_input)
        << ", " << GetValidShape(std::stoi(params["BLOCK_N"]), data_type_input)
        << ", "
        << GetValidShape(std::stoi(params["L1_BLOCK_K"]), data_type_input)
        << ", " << params["L1_BLOCK_M"] << ", " << params["L1_BLOCK_N"] << ", "
        << params["L1_BLOCK_K"] << ", " << params["BLOCK_M"] << ", "
        << params["BLOCK_N"] << ", " << params["L1_BLOCK_K"] << ", "
        << params["transpose_A"] << ", " << params["transpose_B"] << ">"
        << "(" << a_name << ", " << b_name << ", " << c_name << ", "
        << PrintExpr(op->args[4]) << ");\n";
  }
}

void CodeGenTileLangAscendPto::SyncAllCodegen(const CallNode *op) {
  LOG(FATAL) << "Unsupport SyncAll in pto backend.";
}

void CodeGenTileLangAscendPto::PipeBarrierCodegen(const CallNode *op) {
  std::string pipe = Downcast<StringImm>(op->args[0])->value;
  this->PrintIndent();
  this->stream << "pipe_barrier(PIPE_" << pipe << ");\n";
}

void CodeGenTileLangAscendPto::SetAndWaitFlagCodegen(
    const CallNode *op, const std::string &op_name) {
  std::string src = Downcast<StringImm>(op->args[0])->value;
  std::string dst = Downcast<StringImm>(op->args[1])->value;
  std::string event_id = PrintExpr(op->args[2]);
  this->PrintIndent();
  this->stream << kAscendPtoScope << op_name << "_pipeline<PIPE_" << src << ", "
               << "PIPE_" << dst << "> (" << event_id << ");\n";
}

void CodeGenTileLangAscendPto::HandleA5Flag(const std::string &op,
                                            const std::string &pipe, int flag) {
  if (this->current_resource_scope_ == "CUBE") {
    this->PrintIndent();
    this->stream << op << "(" << "PIPE_" << pipe << ", " << flag << ");\n";
    this->PrintIndent();
    this->stream << op << "(" << "PIPE_" << pipe << ", " << flag + 16 << ");\n";
  } else if (this->current_resource_scope_ == "VEC") {
    this->PrintIndent();
    this->stream << op << "(" << "PIPE_" << pipe << ", " << flag << ");\n";
  } else {
    LOG(WARNING) << op << " called outside of known scope (CUBE/VEC)!";
  }
}

void CodeGenTileLangAscendPto::SetCrossFlagCodegen(const CallNode *op) {
  std::string pipe = Downcast<StringImm>(op->args[0])->value;
  std::string flag = PrintExpr(op->args[1]);
  std::string mode = PrintExpr(op->args[2]);

  if (this->platform_ == "A5") {
    if (this->current_resource_scope_ == "CUBE") {
      this->PrintIndent();
      this->stream << kAscendPtoScope << "set_intra_block_cube<PIPE_" << pipe
                   << ">(" << flag << ");\n";
    } else if (this->current_resource_scope_ == "VEC") {
      this->PrintIndent();
      this->stream << kAscendPtoScope << "set_intra_block_vec<PIPE_" << pipe
                   << ">(" << flag << ");\n";
    } else {
      LOG(WARNING)
          << "set_cross_flag called outside of known scope (CUBE/VEC)!";
    }
  } else {
    this->PrintIndent();
    this->stream << kAscendPtoScope << "set_cross_flag<PIPE_" << pipe << ">("
                 << flag << ", " << mode << ");\n";
  }
}

void CodeGenTileLangAscendPto::AutoSetCrossFlagCodegen(const CallNode *op) {
  auto pipe = op->args[1].as<StringImmNode>()->value;
  auto flag = op->args[2].as<IntImmNode>()->value;
  if (this->platform_ == "A5") {
    HandleA5Flag("set_intra_block", pipe, flag);
  } else {
    auto mode = op->args[0].as<IntImmNode>()->value;
    int config = 1 | (mode << 4) | (flag << 8);
    this->PrintIndent();
    this->stream << "ffts_cross_core_sync" << "(" << "PIPE_" << pipe << ", "
                 << config << ");\n";
  }
}

void CodeGenTileLangAscendPto::WaitCrossFlagCodegen(const CallNode *op) {
  std::string pipe = Downcast<StringImm>(op->args[1])->value;
  std::string flag = PrintExpr(op->args[0]);

  if (this->platform_ == "A5") {
    if (pipe.empty()) {
      if (this->current_resource_scope_ == "CUBE") {
        pipe = "MTE1";
      } else if (this->current_resource_scope_ == "VEC") {
        pipe = "V";
      } else {
        LOG(WARNING) << "Cannot infer default pipe for wait_intra_block in "
                        "unknown scope";
      }
    }
  } else {
    if (!pipe.empty()) {
      LOG(FATAL) << "Pipe argument for wait_cross_flag is only supported on A5 "
                    "architecture.";
    }
  }

  if (this->platform_ == "A5") {
    if (this->current_resource_scope_ == "CUBE") {
      this->PrintIndent();
      this->stream << kAscendPtoScope << "wait_intra_block_cube<PIPE_" << pipe
                   << ">(" << flag << ");\n";
    } else if (this->current_resource_scope_ == "VEC") {
      this->PrintIndent();
      this->stream << kAscendPtoScope << "wait_intra_block_vec<PIPE_" << pipe
                   << ">(" << flag << ");\n";
    } else {
      LOG(WARNING)
          << "wait_cross_flag called outside of known scope (CUBE/VEC)!";
    }
  } else {
    this->PrintIndent();
    this->stream << kAscendPtoScope << "wait_cross_flag(" << flag << ");\n";
  }
}

void CodeGenTileLangAscendPto::FillCodegen(const CallNode *op) {
  this->PrintIndent();
  this->stream << "set_flag(PIPE_V, PIPE_S, EVENT_ID0);\n";
  this->PrintIndent();
  this->stream << "wait_flag(PIPE_V, PIPE_S, EVENT_ID0);\n";
  this->PrintIndent();
  this->stream << "TEXPANDS" << "("
               << PrintBufferOffset(op->args[1].as<CallNode>()) << ", "
               << PrintExpr(op->args[2]) << ");\n";
}

void CodeGenTileLangAscendPto::CreateVecIndexCodegen(const CallNode *op,
                                                    const std::string &op_name) {
  this->PrintIndent();
  std::string dst_name = PrintExpr(op->args[1].as<CallNode>()->args[1]);
  std::string dst_offset = PrintExpr(op->args[1].as<CallNode>()->args[2]);
  std::string first_value = PrintExpr(op->args[2]);
  std::vector<std::string> ub_data = ub_data_map_[dst_name];
  int32_t len = GetTypeLen(ub_data[0]);
  this->stream << kAscendPtoScope << "tci" << "<" << ub_data[0] << ", "
               << ub_data[1] << ", " << ub_data[2] << ">" << "(" << ub_data[3]
               << ", " << dst_offset << ", " << len << ", " << first_value
               << ");\n";
}

void CodeGenTileLangAscendPto::GatherbCodegen(const CallNode *op,
                                              const std::string &op_name) {
  this->PrintIndent();
  std::string dst_name = PrintExpr(op->args[1].as<CallNode>()->args[1]);
  std::string src_name = PrintExpr(op->args[2].as<CallNode>()->args[1]);
  std::string idx_name = PrintExpr(op->args[3].as<CallNode>()->args[1]);
  this->stream << op_name << "(" << dst_name << ", " << src_name << ", "
               << idx_name << ");\n";
}

void CodeGenTileLangAscendPto::GatherMaskCodegen(const CallNode *op,
                                                 const std::string &op_name) {
  if (op->args[3].as<CallNode>()) {
    this->PrintIndent();
    std::string dst_name = PrintExpr(op->args[1].as<CallNode>()->args[1]);
    std::string src_name = PrintExpr(op->args[2].as<CallNode>()->args[1]);
    std::string idx_name = PrintExpr(op->args[3].as<CallNode>()->args[1]);
    this->stream << op_name << "(" << dst_name << ", " << src_name << ", "
                 << idx_name << ");\n";
  } else {
    std::string src1Pattern = Downcast<StringImm>(op->args[3])->value;
    this->PrintIndent();
    std::string dst_name = PrintExpr(op->args[1].as<CallNode>()->args[1]);
    std::string src_name = PrintExpr(op->args[2].as<CallNode>()->args[1]);
    std::string DstT_type = ub_data_map_[dst_name][0];
    std::string DstT_row = ub_data_map_[dst_name][1];
    std::string DstT_col = ub_data_map_[dst_name][2];
    std::string DstT_rowValid = ub_data_map_[dst_name][5];
    std::string DstT_colValid = ub_data_map_[dst_name][6];
    std::string SrcT_type = ub_data_map_[src_name][0];
    std::string SrcT_row = ub_data_map_[src_name][1];
    std::string SrcT_col = ub_data_map_[src_name][2];
    std::string SrcT_rowValid = ub_data_map_[dst_name][5];
    std::string SrcT_colValid = ub_data_map_[dst_name][6];

    this->stream << op_name << "<" << kAscendPtoScope << "TileUbDataND<"
                 << DstT_type << ", " << DstT_row << ", " << DstT_col << ", "
                 << DstT_rowValid << ", " << DstT_colValid << ">, "
                 << kAscendPtoScope << "TileUbDataND<" << SrcT_type << ", "
                 << SrcT_row << ", " << SrcT_col << ", " << SrcT_rowValid
                 << ", " << SrcT_colValid << ">, "
                 << "MaskPattern::" << src1Pattern << ">(" << dst_name << ", "
                 << src_name << ");\n";
  }
}

void CodeGenTileLangAscendPto::PowCodegen(const CallNode *op) {
  this->PrintIndent();
  std::string dst_name = PrintExpr(op->args[0].as<CallNode>()->args[1]);
  std::string src0_name = PrintExpr(op->args[1].as<CallNode>()->args[1]);
  std::string src1_name = PrintExpr(op->args[2].as<CallNode>()->args[1]);
  std::string tmp_name = PrintExpr(op->args[3].as<CallNode>()->args[1]);
  std::vector<std::string> ub_data = ub_data_map_[dst_name];
  std::vector<std::string> tmp_data = ub_data_map_[tmp_name];
  this->stream << kAscendPtoScope << "pow" << "<" << ub_data[0] << ", " << ub_data[1] << ", " << ub_data[2] << ", " << tmp_data[1] << ">"
   << "(" << dst_name << ", " << src0_name << ", " << src1_name << ", " << tmp_name << ");\n";
}

void CodeGenTileLangAscendPto::Sort32Codegen(const CallNode *op,
                                             const std::string &op_name) {
  this->PrintIndent();
  std::string dst_name = PrintExpr(op->args[0].as<CallNode>()->args[1]);
  std::string src_name = PrintExpr(op->args[1].as<CallNode>()->args[1]);
  std::string idx_name = PrintExpr(op->args[2].as<CallNode>()->args[1]);
  this->stream << op_name << "(" << dst_name << ", " << src_name << ", "
               << idx_name << ");\n";
}

void CodeGenTileLangAscendPto::TransposeCodegen(const CallNode *op,
                                                const std::string &op_name) {
  this->PrintIndent();
  std::string dst_name = PrintExpr(op->args[0].as<CallNode>()->args[1]);
  std::string src_name = PrintExpr(op->args[1].as<CallNode>()->args[1]);
  DataType dtype = GetAccessPtrDtypePto(op->args[1].as<CallNode>());
  std::string type = getType(dtype);
  
  this->stream << "{\n";
  this->PrintIndent();
  this->stream << "  tl::ascend_pto::TileUbDataND<" << type << ", 16, 16, 16, 16> tmp_ub;\n";
  this->PrintIndent();
  this->stream << "  pto::TASSIGN(tmp_ub, 2048);\n";
  this->PrintIndent();
  this->stream << "  tl::ascend_pto::transpose<" << type << ", 16, 16>(" << dst_name << ", " << src_name << ", tmp_ub);\n";
  this->PrintIndent();
  this->stream << "}\n";
}

void CodeGenTileLangAscendPto::XorCodegen(const CallNode *op,
                                          const std::string &op_name) {
  this->PrintIndent();
  std::string dst_name = PrintExpr(op->args[0].as<CallNode>()->args[1]);
  std::string src0_name = PrintExpr(op->args[1].as<CallNode>()->args[1]);
  std::string src1_name = PrintExpr(op->args[2].as<CallNode>()->args[1]);
  std::string tmp_name = PrintExpr(op->args[3].as<CallNode>()->args[1]);
  this->stream << op_name << "(" << dst_name << ", " << src0_name << ", "
               << src1_name << ", " << tmp_name << ");\n";
}

void CodeGenTileLangAscendPto::CompareCodegen(const CallNode *op,
                                              const std::string &op_name) {
  this->PrintIndent();
  std::string dst_name = PrintExpr(op->args[0].as<CallNode>()->args[1]);
  std::string src0_name = PrintExpr(op->args[1].as<CallNode>()->args[1]);
  std::string src1_name = PrintExpr(op->args[2].as<CallNode>()->args[1]);
  std::string mode = Downcast<StringImm>(op->args[3])->value;
  
  this->stream << kAscendPtoScope << "compare(" << dst_name << ", " << src0_name << ", " <<
  src1_name << ", " << "CmpMode::" << mode << ");\n";
}

void CodeGenTileLangAscendPto::CompareScalarCodegen(
    const CallNode *op, const std::string &op_name) {
  this->PrintIndent();
  std::string dst_name = PrintExpr(op->args[0].as<CallNode>()->args[1]);
  std::string src0_name = PrintExpr(op->args[1].as<CallNode>()->args[1]);
  
  DataType src_dtype = GetAccessPtrDtypePto(op->args[1].as<CallNode>());
  DataType scalar_dtype = op->args[2].dtype();
  std::string src1_name = PrintExpr(op->args[2]);
  if (scalar_dtype != src_dtype) {
    std::string target_type = getType(src_dtype);
    src1_name = target_type + "(" + src1_name + ")";
  }
  
  std::string mode = Downcast<StringImm>(op->args[3])->value;
  
  this->stream << kAscendPtoScope << "compare_scalar(" << dst_name << ", " << src0_name << ", " <<
  src1_name << ", " << "CmpMode::" << mode << ");\n";
}

void CodeGenTileLangAscendPto::TshCodegen(const CallNode *op,
                                                       const std::string &op_name) {
  this->PrintIndent();
  std::string dst_name = PrintExpr(op->args[0].as<CallNode>()->args[1]);
  std::string src0_name = PrintExpr(op->args[1].as<CallNode>()->args[1]);
  
  DataType src_dtype = GetAccessPtrDtypePto(op->args[1].as<CallNode>());
  DataType scalar_dtype = op->args[2].dtype();
  std::string src1_name = PrintExpr(op->args[2]);
  if (scalar_dtype != src_dtype) {
    std::string target_type = getType(src_dtype);
    src1_name = target_type + "(" + src1_name + ")";
  }
  
  this->stream << op_name << "(" << dst_name << ", " << src0_name << ", " << src1_name << ");\n";
}

void CodeGenTileLangAscendPto::ArithProgressionCodegen(
    const CallNode *op, const std::string &op_name) {
  this->PrintIndent();
  std::string buffer_name = PrintExpr(op->args[1].as<CallNode>()->args[1]);
  std::string template_str = Downcast<StringImm>(op->args[0])->value;
  size_t start = template_str.find('<');
  size_t end = template_str.find('>');
  std::string dtype = template_str.substr(start + 1, end - start - 1);
  std::string first_value = PrintExpr(op->args[2]);
  std::string diff_value = PrintExpr(op->args[3]);
  int descending = 0;
  if (const auto *diff_int = op->args[3].as<IntImmNode>()) {
    if (diff_int->value < 0) {
      descending = 1;
    }
  }
  this->stream << "TCI<decltype(" << buffer_name << "), " << dtype
               << ", /*descending=*/" << descending << ">(" << buffer_name
               << ", " << first_value << ");\n";
}

void CodeGenTileLangAscendPto::PrintfOpCodegen(const CallNode *op,
                                               const std::string &op_name) {
  this->PrintIndent();
  this->stream << op_name << "(";
  for (size_t i = 0; i < op->args.size(); ++i) {
    if (i > 0) {
      this->stream << ", ";
    }
    this->stream << PrintExpr(op->args[i]);
  }
  this->stream << ");\n";
}

void CodeGenTileLangAscendPto::DumpTensorCodegen(const CallNode *op,
                                                 const std::string &op_name) {
  this->PrintIndent();
  this->stream << "TPRINT" << "(";
  this->stream << PrintBufferOffset(op->args[0].as<CallNode>());
  this->stream << ");\n";
}

void CodeGenTileLangAscendPto::SetDeqScaleCodegen(const CallNode *op) {
  this->PrintIndent();
  this->stream << "set_deqscale(static_cast<half>(";
  this->stream << PrintExpr(op->args[0]);
  this->stream << "));\n";
}

void CodeGenTileLangAscendPto::BinaryVecOpCodegen(const CallNode *op,
                                                  const std::string &op_name) {

  ShapeInfo src0_shape_info = GetSliceInfo(op->args[1].as<CallNode>());
  ShapeInfo src1_shape_info = GetSliceInfo(op->args[2].as<CallNode>());
  ShapeInfo dst_shape_info = GetSliceInfo(op->args[0].as<CallNode>());

  std::string src0_offset = src0_shape_info.offset;
  std::string src1_offset = src1_shape_info.offset;
  std::string dst_offset = dst_shape_info.offset;

  int32_t src0_extent = src0_shape_info.extent;
  int32_t src1_extent = src1_shape_info.extent;
  int32_t dst_extent = dst_shape_info.extent;

  std::string ns_prefix = "";
  if (op_name == "tand" || op_name == "tor") {
    ns_prefix = kAscendPtoScope;
  }

  std::vector<std::string> var_names;
  for (int i = 0; i < op->args.size() - 1; i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }
  if (!(src0_shape_info.is_slice || src1_shape_info.is_slice ||
        dst_shape_info.is_slice)) {
    this->PrintIndent();
    this->stream << ns_prefix << op_name << "(";

    if (prefetch_n_stages_map_[var_names[0]].first > 0) {
      PrimExpr element_count_dst;
      auto shape_dst = buffer_shapess_[GetRef<tir::Var>(
          op->args[0].as<CallNode>()->args[1].as<VarNode>())];
      if (shape_dst.size() == 3) {
        element_count_dst = shape_dst[1] * shape_dst[2];
      } else if (shape_dst.size() == 2) {
        element_count_dst = shape_dst[0] * shape_dst[1];
      } else if (shape_dst.size() == 1) {
        element_count_dst = shape_dst[0];
      } else {
        ICHECK(false)
            << "An error occurred. Please check prefetch_n_stages_map_, "
               "buffer_shapes_, and buffer_versions_.";
      }
      auto buffer_k_dst =
          op->args[0].as<CallNode>()->args[2] / element_count_dst;
      tvm::arith::Analyzer analyzer;
      PrimExpr simplified_buffer_k_dst = analyzer.Simplify(buffer_k_dst);
      this->stream << var_names[0] << "["
                   << (PrintExpr(simplified_buffer_k_dst)) << "], ";
    } else {
      this->stream << var_names[0] << ", ";
    }

    if (prefetch_n_stages_map_[var_names[1]].first > 0) {
      PrimExpr element_count_src0;
      auto shape_src0 = buffer_shapess_[GetRef<tir::Var>(
          op->args[1].as<CallNode>()->args[1].as<VarNode>())];
      if (shape_src0.size() == 3) {
        element_count_src0 = shape_src0[1] * shape_src0[2];
      } else if (shape_src0.size() == 2) {
        element_count_src0 = shape_src0[0] * shape_src0[1];
      } else if (shape_src0.size() == 1) {
        element_count_src0 = shape_src0[0];
      } else {
        ICHECK(false)
            << "An error occurred. Please check prefetch_n_stages_map_, "
               "buffer_shapes_, and buffer_versions_.";
      }
      auto buffer_k_src0 =
          op->args[1].as<CallNode>()->args[2] / element_count_src0;
      tvm::arith::Analyzer analyzer;
      PrimExpr simplified_buffer_k_src0 = analyzer.Simplify(buffer_k_src0);
      this->stream << var_names[1] << "[" << PrintExpr(simplified_buffer_k_src0)
                   << "], ";
    } else {
      this->stream << var_names[1] << ", ";
    }

    if (prefetch_n_stages_map_[var_names[2]].first > 0) {
      PrimExpr element_count_src1;
      auto shape_src1 = buffer_shapess_[GetRef<tir::Var>(
          op->args[2].as<CallNode>()->args[1].as<VarNode>())];
      std::cout << "shape_src1: " << shape_src1 << std::endl;
      if (shape_src1.size() == 3) {
        element_count_src1 = shape_src1[1] * shape_src1[2];
      } else if (shape_src1.size() == 2) {
        element_count_src1 = shape_src1[0] * shape_src1[1];
      } else if (shape_src1.size() == 1) {
        element_count_src1 = shape_src1[0];
      } else {
        ICHECK(false)
            << "An error occurred. Please check prefetch_n_stages_map_, "
               "buffer_shapes_, and buffer_versions_.";
      }
      auto buffer_k_src1 =
          op->args[2].as<CallNode>()->args[2] / element_count_src1;
      tvm::arith::Analyzer analyzer;
      PrimExpr simplified_buffer_k_src1 = analyzer.Simplify(buffer_k_src1);
      this->stream << var_names[2] << "[" << PrintExpr(simplified_buffer_k_src1)
                   << "]);\n";
    } else {
      this->stream << var_names[2] << ");\n";
    }

  } else if (src0_shape_info.is_slice || src1_shape_info.is_slice ||
             dst_shape_info.is_slice) {
    std::string src0_temp_name = GetTempVarName(src0_shape_info.ub_name);
    std::string src1_temp_name = GetTempVarName(src1_shape_info.ub_name);
    std::string dst_temp_name = GetTempVarName(dst_shape_info.ub_name);
    CreateUbVariableND(src0_temp_name, src0_shape_info);
    CreateUbVariableND(src1_temp_name, src1_shape_info);
    CreateUbVariableND(dst_temp_name, dst_shape_info);
    this->PrintIndent();
    this->stream << ns_prefix << op_name << "(" << dst_temp_name << ", " << src0_temp_name
                 << ", " << src1_temp_name << ");\n";
  } else if (prefetch_n_stages_map_[var_names[1]].first == 0) {
    this->PrintIndent();
    this->stream << ns_prefix << op_name << "(";
    for (int i = 0; i < var_names.size(); i++) {
      this->stream << var_names[i];
      if (i != var_names.size() - 1) {
        this->stream << ", ";
      }
    }
    this->stream << ");\n";
  } else {
    ICHECK(false) << "BinaryVecOpCodegen Failed";
  }
}

std::string extractBroadCastAxis(const std::string &input) {
  std::string axis;
  size_t start = input.find('<');
  if (start == std::string::npos) {
    return axis;
  }
  size_t end = input.find('>', start);
  if (end == std::string::npos) {
    return axis;
  }
  std::string templatePart = input.substr(start + 1, end - start - 1);
  templatePart.erase(std::remove(templatePart.begin(), templatePart.end(), ' '),
                     templatePart.end());
  std::vector<std::string> parts;
  std::stringstream ss(templatePart);
  std::string token;
  while (std::getline(ss, token, ',')) {
    parts.push_back(token);
  }
  return parts[2];
}

void CodeGenTileLangAscendPto::BroadcastOpCodegen(const CallNode *op) {
  std::string template_args = PrintExpr(op->args[0]);
  std::string src_name = PrintExpr(op->args[2].as<CallNode>()->args[1]);
  std::string dst_name = PrintExpr(op->args[1].as<CallNode>()->args[1]);

  auto src_element_count = op->args[2].as<CallNode>()->args[3];
  auto dst_element_count = op->args[1].as<CallNode>()->args[3];

  bool is_slice = false;
  auto src_shape = buffer_shapess_[GetRef<tir::Var>(
      op->args[2].as<CallNode>()->args[1].as<VarNode>())];
  auto dst_shape = buffer_shapess_[GetRef<tir::Var>(
      op->args[1].as<CallNode>()->args[1].as<VarNode>())];
  ShapeInfo src_shape_info = GetSliceInfo(op->args[2].as<CallNode>());
  ShapeInfo dst_shape_info = GetSliceInfo(op->args[1].as<CallNode>());

  auto check_equal = [&](const PrimExpr &a, const PrimExpr &b) -> bool {
    tvm::arith::Analyzer analyzer;
    PrimExpr simplified = analyzer.Simplify(a == b);
    if (auto imm = simplified.as<tir::IntImmNode>())
      return imm->value != 0;
    return false;
  };

  bool src_buffer_in_pipeline = buffer_versions_.count(
      GetRef<tir::Var>(op->args[2].as<CallNode>()->args[1].as<VarNode>()));

  // Check whether slicing is required.
  if (src_shape.size() == 3 && src_buffer_in_pipeline) {
    tvm::arith::Analyzer analyzer;
    PrimExpr cond0 = (src_shape[1] != 1);
    PrimExpr cond1 = (src_shape[2] != 1);
    PrimExpr both_cond = analyzer.Simplify(cond0 && cond1);
    auto imm = both_cond.as<tir::IntImmNode>();
    if (imm->value != 0) {
      ICHECK(false) << "check src_shape in broadcast.";
    }
    is_slice = !check_equal(src_element_count, src_shape[1] * src_shape[2]);
  } else if (src_shape.size() == 2) {
    tvm::arith::Analyzer analyzer;
    PrimExpr cond0 = (src_shape[0] != 1);
    PrimExpr cond1 = (src_shape[1] != 1);
    PrimExpr both_cond = analyzer.Simplify(cond0 && cond1);
    auto imm = both_cond.as<tir::IntImmNode>();
    if (imm->value != 0) {
      ICHECK(false) << "check src_shape in broadcast.";
    }
    is_slice = !check_equal(src_element_count, src_shape[0] * src_shape[1]);
  } else if (src_shape.size() == 1) {
    is_slice = !check_equal(src_element_count, src_shape[0]);
  } else {
    ICHECK(false) << "check src_shape in broadcast.";
  }

  std::vector<std::string> ub_data_vector = ub_data_map_[src_name];
  std::string ub_data_type = ub_data_vector[0];
  std::string row = ub_data_vector[2];
  std::string col = ub_data_vector[1];
  std::string address = ub_data_vector[3];

  std::string axis = extractBroadCastAxis(template_args);
  if (axis == "1") {
    // Handle pipeline logic
    PrimExpr simplified_src_k;
    int8_t typeSize = GetTypeLen(ub_data_type);
    if (src_buffer_in_pipeline) {
      auto src_k = op->args[2].as<CallNode>()->args[2] /
                   op->args[2].as<CallNode>()->args[3];
      tvm::arith::Analyzer analyzer;
      simplified_src_k = analyzer.Simplify(src_k);
      // src_name = src_name + "[" + PrintExpr(simplified_src_k) + "]";
    }

    // To match the PTO interface parameters, allocate a temporary tile in DN
    // format.
    this->PrintIndent();
    this->stream << kAscendPtoScope << "TileUbDataDN <" << ub_data_type << ", "
                 << row << ", " << col << ", " << row << ", " << col << "> "
                 << src_name << "_DN_ROWEXPAND;\n";
    this->PrintIndent();
    if (src_buffer_in_pipeline) {
      this->stream << "TASSIGN(" << src_name << "_DN_ROWEXPAND, " << address
                   << " + " << PrintExpr(simplified_src_k) << " * "
                   << static_cast<int>(typeSize) << " * " << col << " * " << row
                   << ");\n";
    } else {
      this->stream << "TASSIGN(" << src_name << "_DN_ROWEXPAND, " << address
                   << ");\n";
    }

    // Handle slice logic
    this->PrintIndent();
    if (!is_slice) {
      this->stream << "TROWEXPAND" << "(" << dst_name << ", " << src_name
                   << "_DN_ROWEXPAND" << ");\n";
    } else {
      std::string src_temp_name = GetTempVarName(src_shape_info.ub_name);
      std::string dst_temp_name = GetTempVarName(dst_shape_info.ub_name);
      CreateUbVariableDN(src_temp_name, src_shape_info);
      CreateUbVariableND(dst_temp_name, dst_shape_info);
      this->PrintIndent();
      this->stream << "TROWEXPAND" << "(" << dst_temp_name << ", "
                   << src_temp_name << ");\n";
    }

    this->PrintIndent();
    if (src_buffer_in_pipeline) {
      this->stream << "TRESHAPE(" << src_name << "["
                   << PrintExpr(simplified_src_k) << "], " << src_name
                   << "_DN_ROWEXPAND);\n";
    } else {
      this->stream << "TRESHAPE(" << src_name << ", " << src_name
                   << "_DN_ROWEXPAND);\n";
    }
  } else {
    // Handle pipeline logic
    if (src_buffer_in_pipeline) {
      auto src_k = op->args[2].as<CallNode>()->args[2] /
                   op->args[2].as<CallNode>()->args[3];
      tvm::arith::Analyzer analyzer;
      PrimExpr simplified_src_k = analyzer.Simplify(src_k);
      src_name = src_name + "[" + PrintExpr(simplified_src_k) + "]";
    }

    this->PrintIndent();
    this->stream << "TCOLEXPAND" << "(" << dst_name << ", " << src_name
                 << ");\n";
  }
}

std::string getValueOrProcess(const std::map<std::string, std::string> &myMap,
                              const std::string &key) {
  auto it = myMap.find(key);
  if (it != myMap.end()) {
    return it->second;
  } else {
    std::string bestMatchValue = "";
    size_t bestMatchLength = 0;
    for (const auto &pair : myMap) {
      size_t pos = key.find(pair.first);
      if (pos != std::string::npos) {
        if (pair.first.length() > bestMatchLength) {
          bestMatchLength = pair.first.length();
          bestMatchValue = pair.second;
        }
      }
    }
    return bestMatchValue;
  }
}

bool IsComplexExpression(const PrimExpr &expr) {
  if (expr.as<tir::AddNode>()) {
    return true;
  }
  if (expr.as<tir::SubNode>()) {
    return true;
  }
  if (expr.as<tir::MulNode>()) {
    return true;
  }
  if (expr.as<tir::DivNode>()) {
    return true;
  }

  if (expr.as<tir::ModNode>() || expr.as<tir::FloorDivNode>() ||
      expr.as<tir::FloorModNode>() || expr.as<tir::MaxNode>() ||
      expr.as<tir::MinNode>()) {
    return true;
  }
  return false;
}

void CodeGenTileLangAscendPto::BinaryVecOpsCodegen(const CallNode *op,
                                                   const std::string &op_name) {
  std::vector<std::string> var_names;
  std::string operation;
  std::string final_scalar;
  std::string raw_index = PrintExpr(op->args[op->args.size() - 2]);
  if (op_name == "TSUBS") {
    operation = "TADDS";
    final_scalar = "-" + raw_index;
  } else if (op_name == "TDIVS") {
    operation = "TMULS";
    final_scalar = "1.0f / " + raw_index;
  } else {
    operation = op_name;
    final_scalar = raw_index;
  }
  for (int i = 0; i < op->args.size() - 2; i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }
  auto selected_elements_buffer = op->args[2].as<CallNode>();
  bool is_call = selected_elements_buffer != nullptr;
  if (is_call) {
    std::string index = PrintExpr(op->args[op->args.size() - 2]);

    auto selected_elements_buffer_shape = buffer_shapess_[GetRef<tir::Var>(
        selected_elements_buffer->args[1].as<tir::VarNode>())];
    bool selected_elements_buffer_in_pipeline = false;
    if (buffer_versions_.count(GetRef<tir::Var>(
            selected_elements_buffer->args[1].as<tir::VarNode>()))) {
      if (!is_one(buffer_versions_[GetRef<tir::Var>(
              selected_elements_buffer->args[1].as<tir::VarNode>())])) {
        selected_elements_buffer_in_pipeline = true;
      }
    } else if (selected_elements_buffer_shape.size() == 3) {
      selected_elements_buffer_in_pipeline = true;
    }

    std::string scalar_expr;
    if (!selected_elements_buffer_in_pipeline) {
      scalar_expr = PrintBufferOffset(selected_elements_buffer) +
                    ".GetValue(" + index + ")";
    } else {
      std::string index_pipeline = PrintExpr(op->args[op->args.size() - 2] -
                                             selected_elements_buffer->args[2]);
      auto a_k =
          selected_elements_buffer->args[2] / selected_elements_buffer->args[3];
      tvm::arith::Analyzer analyzer;
      PrimExpr simplified_buffer_k = analyzer.Simplify(a_k);
      scalar_expr = PrintBufferOffset(selected_elements_buffer) +
                    "[" + PrintExpr(simplified_buffer_k) + "]" +
                    ".GetValue(" + index_pipeline + ")";
    }
    std::string scalar_name =
        GetTempVarName(PrintBufferOffset(selected_elements_buffer) + "_scalar");
    this->PrintIndent();
    this->stream << "set_flag(PIPE_V, PIPE_S, EVENT_ID0);\n";
    this->PrintIndent();
    this->stream << "wait_flag(PIPE_V, PIPE_S, EVENT_ID0);\n";
    this->PrintIndent();
    this->stream << "auto " << scalar_name << " = " << scalar_expr << ";\n";

    ShapeInfo src_shape_info = GetSliceInfo(op->args[1].as<CallNode>());
    ShapeInfo dst_shape_info = GetSliceInfo(op->args[0].as<CallNode>());

    std::string loop_num = getValueOrProcess(for_num_map_, index);

    std::string final_op_name = operation;
    std::string applied_scalar;
    if (op_name == "TSUBS") {
      DataType dtype0 = GetAccessPtrDtypePto(op->args[0].as<CallNode>());
      DataType scalar_dtype = GetAccessPtrDtypePto(selected_elements_buffer);
      std::string scalar_value = scalar_name;
      if (scalar_dtype != dtype0 && dtype0.is_float16()) {
        scalar_value = "float(" + scalar_name + ")";
      }
      applied_scalar = "-" + scalar_value;
    } else if (op_name == "TDIVS") {
      DataType dtype0 = GetAccessPtrDtypePto(op->args[0].as<CallNode>());
      DataType scalar_dtype = GetAccessPtrDtypePto(selected_elements_buffer);
      std::string scalar_value = scalar_name;
      if (scalar_dtype != dtype0 && dtype0.is_float16()) {
        scalar_value = "float(" + scalar_name + ")";
      }
      applied_scalar = "1.0f / " + scalar_value;
    } else {
      applied_scalar = scalar_name;
    }
    if (loop_num >= "0" && loop_num <= "9") {
      std::string src_temp_name = GetTempVarName(src_shape_info.ub_name);
      std::string dst_temp_name = GetTempVarName(dst_shape_info.ub_name);
      CreateUbVariableND(src_temp_name, src_shape_info);
      CreateUbVariableND(dst_temp_name, dst_shape_info);
      this->PrintIndent();
      this->stream << final_op_name << "(" << dst_temp_name << ", "
                   << src_temp_name << ", " << applied_scalar << ");\n";
    } else {
      this->PrintIndent();
      this->stream << operation << "(" << var_names[0] << ", " << var_names[1]
                   << ", " << applied_scalar << ");\n";
    }
  } else {
    this->PrintIndent();
    this->stream << operation << "(";
    for (size_t i = 0; i < var_names.size(); ++i) {
      this->stream << var_names[i] << ", ";
    }
    this->stream << final_scalar << ");\n";
  }
}

void CodeGenTileLangAscendPto::UnaryVecOpCodegen(const CallNode *op,
                                                 const std::string &op_name) {
  std::vector<std::string> var_names;
  for (int i = 0; i < op->args.size() - 1; i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }

  ShapeInfo src_shape_info = GetSliceInfo(op->args[1].as<CallNode>());
  ShapeInfo dst_shape_info = GetSliceInfo(op->args[0].as<CallNode>());

  bool is_src_slice =
      src_shape_info.extent != src_shape_info.row * src_shape_info.col;
  bool is_dst_slice =
      dst_shape_info.extent != dst_shape_info.row * dst_shape_info.col;

  if (src_shape_info.is_slice || dst_shape_info.is_slice) {
    std::string src_temp_name = GetTempVarName(src_shape_info.ub_name);
    std::string dst_temp_name = GetTempVarName(dst_shape_info.ub_name);
    CreateUbVariableND(src_temp_name, src_shape_info);
    CreateUbVariableND(dst_temp_name, dst_shape_info);
    this->PrintIndent();
    this->stream << op_name << "(" << dst_temp_name << ", " << src_temp_name
                 << ");\n";
  } else {
    this->PrintIndent();
    this->stream << op_name << "(";

    if (prefetch_n_stages_map_[var_names[0]].first > 0) {
      PrimExpr element_count_dst;
      auto shape_dst = buffer_shapess_[GetRef<tir::Var>(
          op->args[0].as<CallNode>()->args[1].as<VarNode>())];
      if (shape_dst.size() == 3) {
        element_count_dst = shape_dst[1] * shape_dst[2];
      } else if (shape_dst.size() == 2) {
        element_count_dst = shape_dst[0] * shape_dst[1];
      } else if (shape_dst.size() == 1) {
        element_count_dst = shape_dst[0];
      } else {
        ICHECK(false)
            << "An error occurred. Please check prefetch_n_stages_map_, "
               "buffer_shapes_, and buffer_versions_.";
      }
      auto buffer_k_dst =
          op->args[0].as<CallNode>()->args[2] / element_count_dst;
      tvm::arith::Analyzer analyzer;
      PrimExpr simplified_buffer_k_dst = analyzer.Simplify(buffer_k_dst);
      this->stream << var_names[0] << "["
                   << (PrintExpr(simplified_buffer_k_dst)) << "], ";
    } else {
      this->stream << var_names[0] << ", ";
    }

    if (prefetch_n_stages_map_[var_names[1]].first > 0) {
      PrimExpr element_count_src0;
      auto shape_src0 = buffer_shapess_[GetRef<tir::Var>(
          op->args[1].as<CallNode>()->args[1].as<VarNode>())];
      if (shape_src0.size() == 3) {
        element_count_src0 = shape_src0[1] * shape_src0[2];
      } else if (shape_src0.size() == 2) {
        element_count_src0 = shape_src0[0] * shape_src0[1];
      } else if (shape_src0.size() == 1) {
        element_count_src0 = shape_src0[0];
      } else {
        ICHECK(false)
            << "An error occurred. Please check prefetch_n_stages_map_, "
               "buffer_shapes_, and buffer_versions_.";
      }
      auto buffer_k_src0 =
          op->args[1].as<CallNode>()->args[2] / element_count_src0;
      tvm::arith::Analyzer analyzer;
      PrimExpr simplified_buffer_k_src0 = analyzer.Simplify(buffer_k_src0);
      this->stream << var_names[1] << "[" << PrintExpr(simplified_buffer_k_src0)
                   << "]);\n";
    } else {
      this->stream << var_names[1] << ");\n";
    }
  }
}

void CodeGenTileLangAscendPto::ScalarOpCodegen(const CallNode *op,
                                               const std::string &op_name) {
  this->PrintIndent();
  this->stream << op_name << "("
               << PrintBufferOffset(op->args[0].as<CallNode>()) << ", "
               << PrintBufferOffset(op->args[1].as<CallNode>()) << ", "
               << PrintExpr(op->args[2]) << ");\n";
}

void CodeGenTileLangAscendPto::AxpyCodegen(const CallNode *op) {

  std::string dst_name = PrintBufferOffset(op->args[0].as<CallNode>());
  std::string src0_name = PrintBufferOffset(op->args[1].as<CallNode>());

  DataType dtype0 = GetAccessPtrDtypePto(op->args[0].as<CallNode>());
  DataType scalar_dtype = op->args[2].dtype();
  std::string scalar = PrintExpr(op->args[2]);
  if (scalar_dtype != dtype0) {
    if (dtype0.is_float16()) {
      scalar = "float(" + scalar + ")";
    } else {
      std::string target_type = getType(dtype0);
      scalar = target_type + "(" + scalar + ")";
    }
  }

  std::vector<std::string> ub_data = ub_data_map_[dst_name];
  this->PrintIndent();
  this->stream << kAscendPtoScope << "axpy" << "<" << ub_data[0] << ", "
               << ub_data[1] << ", " << ub_data[2] << ">"
               << "(" << dst_name << ", " << src0_name << ", " << scalar
               << ");\n";
}

void CodeGenTileLangAscendPto::BinaryVecClampMaxMinOpsCodegen(
    const CallNode *op, const std::string &op_name) {
  std::vector<std::string> var_names;
  std::string operation = op_name;
  for (int i = 1; i < op->args.size() - 3; i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }
  if (op->args[4].as<CallNode>()) {
    auto var_name = PrintBufferOffset(op->args[4].as<CallNode>());
    std::string ub_name_dst = var_names[1];
    std::string ub_name_src = var_names[2];
    this->PrintIndent();
    std::string index = PrintExpr(op->args[op->args.size() - 2]);
    std::string scalar_name = var_name + "_scalar";
    this->stream << "auto " << scalar_name << "= " << var_name << ".GetValue("
                 << index << ");\n";
    this->stream << operation << "(";
    this->stream << ub_name_dst << ", " << ub_name_src << ", " << scalar_name;
  } else {
    this->PrintIndent();
    this->stream << operation << "(";
    std::string scalar = PrintExpr(op->args[op->args.size() - 2]);
    var_names.push_back(scalar);
    for (int i = 0; i < var_names.size(); i++) {
      this->stream << var_names[i];
      if (i != var_names.size() - 1) {
        this->stream << ", ";
      }
    }
  }
  this->stream << ");\n";
}

void CodeGenTileLangAscendPto::BinaryVecClampOpsCodegen(
    const CallNode *op, const std::string &op_name) {
  std::vector<std::string> var_names;
  std::string operation = op_name;
  std::string scalar_min = PrintExpr(op->args[op->args.size() - 3]);
  std::string scalar_max = PrintExpr(op->args[op->args.size() - 2]);
  for (int i = 1; i < op->args.size() - 4; i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }
  this->PrintIndent();

  this->stream << "set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);\n";
  this->stream << "wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);\n";
  
  // clamp_min: achieve with TMAXS
  this->stream << "TMAXS" << "(";
  
  for (int i = 0; i < var_names.size(); i++) {
    this->stream << var_names[i];
    if (i != var_names.size() - 1) {
      this->stream << ", ";
    }
  }
  this->stream << ", " << scalar_min << ");\n";
  
  this->stream << "set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);\n";
  this->stream << "wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);\n";
  // clamp_max: achieve with TMINS
  this->stream << "set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);\n";
  this->stream << "wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);\n";
  this->stream << "TMINS" << "(";
  
  for (int i = 0; i < var_names.size(); i++) {
    if (i != 1) {
      this->stream << var_names[i];
      if (i != var_names.size() - 1) {
        this->stream << ", ";
      }
    } else {
      this->stream << var_names[0];
      if (i != var_names.size() - 1) {
        this->stream << ", ";
      }
    }
  }
  this->stream << ", " << scalar_max << ");\n";
  this->stream << "set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);\n";
  this->stream << "wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);\n";
}

void CodeGenTileLangAscendPto::SigmoidCodegen(const CallNode *op,
                                              const std::string &op_name) {
  std::vector<std::string> var_names;
  for (int i = 0; i < op->args.size() - 2; i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }
  std::string dst_name = PrintExpr(op->args[0].as<CallNode>()->args[1]);
  std::vector<std::string> ub_data = ub_data_map_[dst_name];
  this->PrintIndent();
  this->stream << kAscendPtoScope << op_name << "<" << ub_data[0] << ", "
               << ub_data[1] << ", " << ub_data[2] << ">" << "(";
  for (int i = 0; i < var_names.size(); i++) {
    this->stream << var_names[i];
    if (i != var_names.size() - 1) {
      this->stream << ", ";
    }
  }
  this->stream << ", " << PrintExpr(op->args[op->args.size() - 1]) << ");\n";
}

void CodeGenTileLangAscendPto::CastCodegen(const CallNode *op,
                                           const std::string &op_type) {
  std::vector<std::string> var_names;
  for (int i = 0; i < op->args.size() - 2; i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }

  ShapeInfo src_shape_info = GetSliceInfo(op->args[0].as<CallNode>());
  ShapeInfo dst_shape_info = GetSliceInfo(op->args[1].as<CallNode>());
  if (src_shape_info.is_slice || dst_shape_info.is_slice) {
    std::string src_temp_name = GetTempVarName(src_shape_info.ub_name);
    std::string dst_temp_name = GetTempVarName(dst_shape_info.ub_name);
    CreateUbVariableND(src_temp_name, src_shape_info);
    CreateUbVariableND(dst_temp_name, dst_shape_info);
    this->PrintIndent();
    this->stream << "TCVT" << "(" << dst_temp_name << ", " << src_temp_name
                 << ", " << "pto::RoundMode::CAST_NONE" << ");\n";
  } else {
    this->stream << "TCVT" << "(";
    var_names.push_back(op_type);
    for (int i = 0; i < var_names.size(); i++) {
      this->stream << var_names[i];
      if (i != var_names.size() - 1) {
        this->stream << ", ";
      }
    }
    this->stream << ");\n";
  }
}

std::tuple<int, int, int, bool>
ExtractTemplateParamsForSliceBuffer(const std::string &op_name) {
  int second_param = 0;
  int third_param = 0;
  int forth_param = 0;
  size_t left = op_name.find('<');
  size_t right = op_name.find('>');

  if (left == std::string::npos || right == std::string::npos ||
      left >= right) {
    return std::make_tuple(second_param, third_param, forth_param, false);
  }

  std::string params_str = op_name.substr(left + 1, right - left - 1);
  std::vector<std::string> params;
  size_t start = 0;
  size_t comma = 0;
  while ((comma = params_str.find(',', start)) != std::string::npos) {
    std::string param = params_str.substr(start, comma - start);
    param.erase(0, param.find_first_not_of(" \t"));
    param.erase(param.find_last_not_of(" \t") + 1);
    params.push_back(param);
    start = comma + 1;
  }

  std::string last_param = params_str.substr(start);
  last_param.erase(0, last_param.find_first_not_of(" \t"));
  last_param.erase(last_param.find_last_not_of(" \t") + 1);
  params.push_back(last_param);

  if (params.size() >= 4) {
    try {
      second_param = std::stoi(params[1]);
      third_param = std::stoi(params[2]);
      forth_param = std::stoi(params[3]);
      return std::make_tuple(second_param, third_param, forth_param, true);
    } catch (const std::exception &e) {
      return std::make_tuple(second_param, third_param, forth_param, false);
    }
  } else {
    ICHECK(false) << "reduce params less than 4.";
  }
  return std::make_tuple(second_param, third_param, forth_param, false);
}

void CodeGenTileLangAscendPto::ReduceOpCodegen(const CallNode *op) {
  std::string op_name = Downcast<StringImm>(op->args[0])->value;

  // Determine whether the reduce operation needs to be sliced.
  auto template_params = ExtractTemplateParamsForSliceBuffer(op_name);
  int param2_int = std::get<0>(template_params);
  int param3_int = std::get<1>(template_params);
  int param4_int = std::get<2>(template_params);
  std::string param2 = std::to_string(param2_int);
  std::string param3 = std::to_string(param3_int);
  std::string mode = "";
  if (param4_int == -1) {
    mode = "row";
  } else if (param4_int == 0) {
    mode = "col";
  } else {
    ICHECK(false)
        << "Only row-wise or column-wise reduce operations are supported. Row "
           "direction is denoted by -1, and column direction by 0.";
  }

  bool success = std::get<3>(template_params);
  if (!success) {
    ICHECK(false) << "ExtractTemplateParams failed";
  }

  if (op_name.find("reduce_sum") != std::string::npos) {
    op_name = (mode == "row") ? "TROWSUM" : "TCOLSUM";
  } else if (op_name.find("reduce_max") != std::string::npos) {
    op_name = (mode == "row") ? "TROWMAX" : "TCOLMAX";
  } else if (op_name.find("reduce_min") != std::string::npos) {
    op_name = (mode == "row") ? "TROWMIN" : "TCOLMIN";
  } else {
    ICHECK(false) << "not support reduce type: " << op_name;
  }

  std::vector<std::string> var_names;
  for (int i = 1; i < op->args.size(); i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }
  std::string ub_name = var_names[0];
  std::string ub_name_dn = ub_name + "_DN_" + std::to_string(reduce_num);
  std::vector<std::string> ub_data_vector = ub_data_map_[ub_name];
  std::string ub_data_type = ub_data_vector[0];
  std::string row = ub_data_vector[2];
  std::string col = ub_data_vector[1];
  std::string ffts = ub_data_vector[3];

  std::string ub_name_src = var_names[1];
  std::vector<std::string> ub_data_vector_src = ub_data_map_[ub_name_src];
  std::string ub_data_type_src = ub_data_vector_src[0];
  std::string row_src = ub_data_vector_src[1];
  std::string col_src = ub_data_vector_src[2];
  std::string ffts_src = ub_data_vector_src[3];

  std::string ub_name_tmp = var_names[2];
  std::vector<std::string> ub_data_vector_tmp = ub_data_map_[ub_name_tmp];
  std::string ub_data_type_tmp = ub_data_vector_tmp[0];
  std::string row_tmp = ub_data_vector_tmp[1];
  std::string col_tmp = ub_data_vector_tmp[2];
  std::string ffts_tmp = ub_data_vector_tmp[3];

  // Determine whether to request the TileUbData with DN arrangement.
  ICHECK(ub_data_vector.size() == 7)
      << "TileUbData needs 7 elements (type, row, col, ffts, applied DN or "
         "not, validrow, validcol), got "
      << ub_data_vector.size() << ".";
  if (mode == "row") {
    if (ub_data_vector[4] ==
        "Unapplied for tileUbDataDN") { // If not applied yet, prioritize
                                        // applying for it.
      this->PrintIndent();
      this->stream << kAscendPtoScope << "TileUbDataDN <" << ub_data_type
                   << ", " << row << ", " << col << ", " << row_src << ", "
                   << col << "> " << ub_name_dn << ";\n";
      this->PrintIndent();
      this->stream << "TASSIGN(" << ub_name_dn << ", " << ffts << ");\n";
      if (param2 == row_src && param3 == col_src) {
        this->PrintIndent();
        this->stream << op_name << "(";
        for (int i = 0; i < var_names.size(); i++) {
          this->stream << var_names[i];
          if (i == 0) {
            this->stream << "_DN_" << reduce_num;
          }
          if (i != var_names.size() - 1) {
            this->stream << ", ";
          }
        }
        this->stream << ");\n";
      } else {
        if (op_name == "TROWMAX") {
          this->PrintIndent();
          this->stream << kAscendPtoScope << "TROWMAX_with_slice_buffer <"
                       << ub_data_type_src << ", " << ub_data_type << ", "
                       << ub_data_type_tmp << ", " << row_src << ", " << col_src
                       << ", " << param2 << ", " << param3 << ", " << row
                       << ", " << row_tmp << ", " << col_tmp << "> ("
                       << ffts_src << ", " << ffts << ", " << ub_name_dn << ", "
                       << ub_name_tmp << ");\n";
        } else if (op_name == "TROWSUM") {
          this->PrintIndent();
          this->stream << kAscendPtoScope << "TROWSUM_with_slice_buffer <"
                       << ub_data_type_src << ", " << ub_data_type << ", "
                       << ub_data_type_tmp << ", " << row_src << ", " << col_src
                       << ", " << param2 << ", " << param3 << ", " << row
                       << ", " << row_tmp << ", " << col_tmp << "> ("
                       << ffts_src << ", " << ffts << ", " << ub_name_dn << ", "
                       << ub_name_tmp << ");\n";
        } else if (op_name == "TROWMIN") {
          this->PrintIndent();
          this->stream << kAscendPtoScope << "TROWMAX_with_slice_buffer <"
                       << ub_data_type_src << ", " << ub_data_type << ", "
                       << ub_data_type_tmp << ", " << row_src << ", " << col_src
                       << ", " << param2 << ", " << param3 << ", " << row
                       << ", " << row_tmp << ", " << col_tmp << "> ("
                       << ffts_src << ", " << ffts << ", " << ub_name_dn << ", "
                       << ub_name_tmp << ");\n";
        }
      }
      this->PrintIndent();
      this->stream << "TRESHAPE(" << var_names[0] << ", " << var_names[0]
                   << "_DN_" << reduce_num << ");\n";
      ub_data_vector[4] = "Applied for tileUbDataDN";
    } else if (ub_data_vector[4] ==
               "Applied for tileUbDataDN") { // If already applied, leverage the
                                             // existing application.
      this->PrintIndent();
      this->stream << op_name << "(";
      for (int i = 0; i < var_names.size(); i++) {
        this->stream << var_names[i];
        if (i == 0) {
          this->stream << "_DN_" << reduce_num;
        }
        if (i != var_names.size() - 1) {
          this->stream << ", ";
        }
      }
      this->stream << ");\n";
      this->PrintIndent();
      this->stream << "TRESHAPE(" << var_names[0] << ", " << var_names[0]
                   << "_DN_" << reduce_num << ");\n";
    } else {
      ICHECK(false) << "Error route in ReduceOpCodegen";
    }
  } else {
    this->PrintIndent();
    if (param2 == row_src && param3 == col_src) {
      if (op_name == "TCOLSUM") {
        int tmp_rows = (std::stoi(row_src) + 1) / 2;
        std::string tmp_cols = col_src;
        this->stream << "{\n";
        this->PrintIndent();
        this->stream << "  " << kAscendPtoScope << "TileUbDataND<" << ub_data_type_src
                     << ", " << tmp_rows << ", " << tmp_cols << "> " << ub_name_tmp << "_cast;\n";
        this->PrintIndent();
        this->stream << "  TASSIGN(" << ub_name_tmp << "_cast, " << ffts_tmp << ");\n";
        this->PrintIndent();
        this->stream << "  " << op_name << "(";
        for (int i = 0; i < var_names.size() - 1; i++) {
          this->stream << var_names[i];
          if (i != var_names.size() - 2) {
            this->stream << ", ";
          }
        }
        this->stream << ", " << ub_name_tmp << "_cast, true);\n";
        this->PrintIndent();
        this->stream << "}\n";
      } else {
        this->stream << op_name << "(";
        for (int i = 0; i < var_names.size() - 1; i++) {
          this->stream << var_names[i];
          if (i != var_names.size() - 2) {
            this->stream << ", ";
          }
        }
        this->stream << ");\n";
      }
    } else {
      if (op_name == "TCOLSUM") {
        int tmp_rows = (std::stoi(param2) + 1) / 2;
        std::string tmp_cols = param3;
        this->stream << kAscendPtoScope << op_name << "_with_slice_buffer <"
                     << ub_data_type_src << ", " << ub_data_type << ", "
                     << row_src << ", " << col_src
                     << ", " << param2 << ", " << param3 << ", " << row << ", "
                     << tmp_rows << ", " << tmp_cols << "> (" << ffts_src << ", "
                     << ffts << ", " << ub_name << ", " << ffts_tmp << ");\n";
      } else {
        this->stream << kAscendPtoScope << op_name << "_with_slice_buffer <"
                     << ub_data_type_src << ", " << ub_data_type << ", "
                     << ub_data_type_tmp << ", " << row_src << ", " << col_src
                     << ", " << param2 << ", " << param3 << ", " << row << ", "
                     << row_tmp << ", " << col_tmp << "> (" << ffts_src << ", "
                     << ffts << ", " << ub_name << ", " << ub_name_tmp << ");\n";
      }
    }
  }
  reduce_num++;
}

void CodeGenTileLangAscendPto::VisitStmt_(const AttrStmtNode *op) {
  if (op->attr_key == "threadblock_swizzle_pattern") {
    this->PrintIndent();
    const StringImmNode *pattern = op->value.as<StringImmNode>();
    ICHECK(pattern);
    this->stream << this->block_id_ << " = " << pattern->value << "("
                 << this->block_id_ << ");\n";
    this->VisitStmt(op->body);
    return;
  } else if (op->attr_key == "thread_extent") {
    IterVar iv = Downcast<IterVar>(op->node);
    if (iv->thread_tag == "blockIdx.x" && iv->var->name_hint != "_") {
      this->block_id_ = AllocVarID(iv->var.get());
      this->PrintIndent();
      auto current_block_id = this->block_id_;
      if (this->use_swizzle_) {
        current_block_id = current_block_id + "_";
      }
      this->stream << "auto " << current_block_id << " = get_block_idx();\n";
      this->PrintIndent();
      stream << "set_ffts_base_addr(ffts_Addr);\n\n";

      this->core_num_ = PrintExpr(op->value);
    } else if (iv->thread_tag == "blockIdx.y" && iv->var->name_hint != "_") {
      this->vec_id_ = AllocVarID(iv->var.get());
      this->PrintIndent();
      auto current_vec_id = this->vec_id_;
      this->stream << "auto " << current_vec_id << " = get_subblockid();\n";
    }
    this->VisitStmt(op->body);
    return;
  } else if (op->attr_key == "resource_scope") { // other core
    auto resource_id = Downcast<IntImm>(op->value)->value;
    auto resource_name = resource_id == 0 ? "CUBE" : "VEC";
    std::string arch_name = (this->platform_ == "A5") ? "C310" : "C220";

    stream << "#if defined(__DAV_" << arch_name << "_" << resource_name
           << "__)\n";
    if (resource_name == "VEC") {
      this->PrintIndent();
      stream << "  set_mask_norm();\n";
      this->PrintIndent();
      stream << "  set_vector_mask(-1, -1);\n";
    }

    std::string old_scope = this->current_resource_scope_;
    this->current_resource_scope_ = resource_name;

    int func_scope = this->BeginScope();
    this->VisitStmt(op->body);
    this->EndScope(func_scope);
    stream << "#endif\n";

    this->current_resource_scope_ = old_scope;
    return;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangAscendPto::UbShapeInputCheck(const AllocateNode *op) {
  auto shape = buffer_shapess_[op->buffer_var];
  if (shape.size() > 3 || shape.size() == 0) {
    ICHECK(false) << "Unsupported ubsize which is expected to be 1, 2 or 3";
  }
}

bool CodeGenTileLangAscendPto::ValidLayoutEnabled(const AllocateNode *op) {
  auto shape = buffer_shapess_[op->buffer_var];
  bool valid = false;
  std::string type = getType(op->dtype);
  int8_t typeSize = GetTypeLen(type);
  bool has_buffer_version = buffer_versions_.count(
      op->buffer_var); // Check if multiple buffers are enabled/configured in
                       // the T.pipeline.
  if (!has_buffer_version) {
    if (shape.size() == 3) {
      if (tvm::tir::is_zero(tvm::truncmod(shape[2] * typeSize, 32))) {
        valid = false;
      } else {
        valid = true;
      }
    } else if (shape.size() == 2) {
      if (tvm::tir::is_zero(tvm::truncmod(shape[1] * typeSize, 32))) {
        valid = false;
      } else {
        valid = true;
      }
    } else if (shape.size() == 1) {
      if (tvm::tir::is_zero(tvm::truncmod(shape[0] * typeSize, 32))) {
        valid = false;
      } else {
        valid = true;
      }
    } else {
      ICHECK(false) << "ValidLayoutEnabled Error";
    }
  } else {
    if (shape.size() == 2) {
      if (tvm::tir::is_zero(tvm::truncmod(shape[0] * typeSize, 32))) {
        valid = false;
      } else {
        valid = true;
      }
    } else if (shape.size() == 3) {
      if (tvm::tir::is_zero(tvm::truncmod(shape[2] * typeSize, 32))) {
        valid = false;
      } else {
        valid = true;
      }
    } else if (shape.size() == 1) {
      if (tvm::tir::is_zero(tvm::truncmod(shape[0] * typeSize, 32))) {
        valid = false;
      } else {
        valid = true;
      }
    } else {
      ICHECK(false) << "ValidLayoutEnabled Error";
    }
  }
  return valid;
}

void CodeGenTileLangAscendPto::VisitStmt_(const AllocateNode *op) {
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get()); // var_name
  std::string scope = GetPtrStorageScope(op->buffer_var);
  std::string type = getType(op->dtype);
  const VarNode *buffer = op->buffer_var.as<VarNode>();

  /// Allocate PTO Tile Memory Address
  auto print_buffer = [&](const std::string &pos) {
    if (!buffer_shapess_.count(op->buffer_var)) {
      ICHECK(false) << "Buffer_shape not found.";
    }
    auto shape = buffer_shapess_[op->buffer_var];
    std::vector<std::string> ub_data(7);
    std::vector<std::string> l_data(3);
    ub_data[0] = type;
    l_data[0] = type;
    if (pos == kAscendPtoScope + "TileUbData") {
      UbShapeInputCheck(op);
    }
    PrimExpr target_expr;
    bool found_by_name = false;
    std::string target_var_name = op->buffer_var->name_hint;

    for (const auto &pair : address_map_) {
      Var var_key = pair.first;
      if (var_key->name_hint == target_var_name) {
        target_expr = pair.second;
        found_by_name = true;
        break;
      }
    }
    this->PrintIndent();

    // Recording the count of buffer allocations in the pipeline.
    bool buffer_in_pipeline = false;
    if (buffer_versions_.count(op->buffer_var)) {
      if (!is_one(buffer_versions_[op->buffer_var])) {
        int8_t bufferNum =
            Downcast<Integer>(buffer_versions_[op->buffer_var])->value;
        prefetch_n_stages_map_[vid] = std::pair<int, int>{bufferNum, 0};
        buffer_in_pipeline = true;
      }
    } else if (shape.size() == 3) {
      auto bufferNum = shape[0].as<IntImmNode>()->value;
      prefetch_n_stages_map_[vid] =
          std::pair<int, int>{static_cast<int>(bufferNum), 0};
      buffer_in_pipeline = true;
    }

    // Allocate buffer
    if (found_by_name) {
      if (pos == kAscendPtoScope + "TileUbData") {
        // Log Unified Buffer (UB) information.
        if (shape.size() == 1) {
          ub_data[1] = "1";
          ub_data[2] = PrintExpr(shape[0]);
          ub_data[4] = "Unapplied for tileUbDataDN";
          ub_data[5] = "1";
          ub_data[6] = PrintExpr(shape[0]);
        } else if (shape.size() == 2) {
          if (shape[1].as<IntImmNode>()->value != 1) {
            if (!buffer_in_pipeline) {
              ub_data[1] = PrintExpr(shape[0]);
              ub_data[2] = PrintExpr(shape[1]);
              ub_data[4] = "Unapplied for tileUbDataDN";
              ub_data[5] = PrintExpr(shape[0]);
              ub_data[6] = PrintExpr(shape[1]);
            } else {
              ub_data[1] = "1";
              ub_data[2] = PrintExpr(shape[1]);
              ub_data[4] = "Unapplied for tileUbDataDN";
              ub_data[5] = "1";
              ub_data[6] = PrintExpr(shape[1]);
            }
          } else {
            ub_data[1] = "1";
            ub_data[2] = PrintExpr(shape[0]);
            ub_data[4] = "Unapplied for tileUbDataDN";
            ub_data[5] = "1";
            ub_data[6] = PrintExpr(shape[0]);
            shape.pop_back();
          }
        } else if (shape.size() == 3) {
          if (buffer_in_pipeline) {
            if (shape[2].as<IntImmNode>()->value != 1) {
              ub_data[1] = PrintExpr(shape[1]);
              ub_data[2] = PrintExpr(shape[2]);
              ub_data[4] = "Unapplied for tileUbDataDN";
              ub_data[5] = PrintExpr(shape[1]);
              ub_data[6] = PrintExpr(shape[2]);
            } else {
              ub_data[1] = "1";
              ub_data[2] = PrintExpr(shape[1]);
              ub_data[4] = "Unapplied for tileUbDataDN";
              ub_data[5] = "1";
              ub_data[6] = PrintExpr(shape[1]);
            }
          } else {
            ICHECK(false) << "Check for cases where the shape size is 3 but is "
                             "not part of the pipeline.";
          }
        }

        // Check if padding is needed at the time of request.
        auto valid = ValidLayoutEnabled(op);
        if (valid) {
          int8_t typeSize = GetTypeLen(type);
          int8_t NDBlockSize = 32 / typeSize;
          int shape2_val = std::stoi(ub_data[2]);
          PrimExpr shape2_expr = IntImm(DataType::Int(32), shape2_val);
          PrimExpr padded_shape2 =
              tvm::floordiv(shape2_expr + NDBlockSize - 1, NDBlockSize) *
              NDBlockSize;
          ub_data[2] = PrintExpr(padded_shape2);
        }

        if (buffer_in_pipeline) {
          int8_t typeSize = GetTypeLen(type);
          // Recording the count of buffer allocations in the pipeline.
          auto bufferNum = prefetch_n_stages_map_[vid].first;
          // prefetch_n_stages_map_[vid] = std::pair<int, int>{bufferNum, 0};

          // Output the valid shape.
          stream << pos << "ND<" << type << ", " << ub_data[1] << ", "
                 << ub_data[2] << ", " << ub_data[5] << ", " << ub_data[6];
          stream << "> " << vid << "[" << bufferNum << "];\n";

          // Batch allocate addresses.
          for (size_t i = 0; i < bufferNum; i++) {
            this->PrintIndent();
            stream << "TASSIGN(" << vid << "[" << i << "], "
                   << PrintExpr(target_expr) << " + " << i << " * "
                   << static_cast<int>(typeSize) << " * " << ub_data[1] << " * "
                   << ub_data[2] << ");\n";
            // if (ub_data[3].empty()) {
            //     ub_data[3] = PrintExpr(address_offset_[String(pos)]);
            // }
            // ub_data_map_[vid] = ub_data;
            // address_offset_.Set(String(pos),
            //                     PrimExpr(int(op->ConstantAllocationSize() *
            //                                  op->dtype.bytes() / 2)) +
            //                         address_offset_[String(pos)]);
          }
        } else {
          stream << pos << "ND<" << type << ", " << ub_data[1] << ", "
                 << ub_data[2] << ", " << ub_data[5] << ", " << ub_data[6];
          stream << "> " << vid << ";\n";
          this->PrintIndent();
          stream << "TASSIGN(" << vid << ", " << PrintExpr(target_expr)
                 << ");\n";
        }
        ub_data[3] = PrintExpr(target_expr);
        ub_data_map_[vid] = ub_data;

      } else {
        if (!buffer_in_pipeline) {
          int dtype_bytes = op->dtype.bytes();
          std::vector<PrimExpr> valid_shapes;
          valid_shapes.reserve(2);
          stream << pos << "<" << type;
          int shape_value = shape[0].as<tvm::tir::IntImmNode>()->value;
          if (shape_value * dtype_bytes % 32 == 0) {
            valid_shapes.push_back(shape[0]);
          } else {
            valid_shapes.push_back(tvm::IntImm(
                shape[0].dtype(),
                shape_value +
                    (32 - shape_value * dtype_bytes % 32) / dtype_bytes));
          }
          valid_shapes.push_back(shape[1]);
          for (size_t i = 0; i < valid_shapes.size(); i++) {
            l_data[i + 1] = PrintExpr(valid_shapes[i]);
            stream << ", " << valid_shapes[i];
          }
          for (size_t i = 0; i < shape.size(); i++) {
            stream << ", " << shape[i];
          }
          stream << "> " << vid << ";\n";
          this->PrintIndent();
          stream << "TASSIGN(" << vid << ", " << PrintExpr(target_expr)
                 << ");\n";
        } else {
          auto bufferNum = prefetch_n_stages_map_[vid].first;
          // prefetch_n_stages_map_[vid] = std::pair<int, int>{bufferNum, 0};
          int dtype_bytes = op->dtype.bytes();
          std::vector<PrimExpr> valid_shapes;
          valid_shapes.reserve(shape.size() - 1);
          stream << pos << "<" << type;
          int shape_value = shape[0].as<tvm::tir::IntImmNode>()->value;
          if (shape_value * dtype_bytes % 32 == 0) {
            valid_shapes.push_back(shape[0]);
          } else {
            valid_shapes.push_back(tvm::IntImm(
                shape[0].dtype(),
                shape_value +
                    (32 - shape_value * dtype_bytes % 32) / dtype_bytes));
          }
          valid_shapes.push_back(shape[1]);
          for (size_t i = 0; i < valid_shapes.size(); i++) {
            l_data[i + 1] = PrintExpr(valid_shapes[i]);
            stream << ", " << valid_shapes[i];
          }
          for (size_t i = 0; i < shape.size(); i++) {
            stream << ", " << shape[i];
          }
          stream << "> " << vid << "[" << bufferNum << "];\n";
          for (size_t j = 0; j < bufferNum; j++) {
            this->PrintIndent();
            stream << "TASSIGN(" << vid << "[" << j << "], "
                   << PrintExpr(target_expr) << " + " << j << " * "
                   << dtype_bytes << " * " << valid_shapes[0] << " * "
                   << valid_shapes[1] << ");\n";
          }
        }
      }
      l_data_map_[vid] = l_data;
    } else {
      if (address_offset_.find(String(pos)) == address_offset_.end()) {
        address_offset_.Set(String(pos), 0);
      }
      if (pos == kAscendPtoScope + "TileUbData") {

        // Log Unified Buffer (UB) information.
        if (shape.size() == 1) {
          ub_data[1] = "1";
          ub_data[2] = PrintExpr(shape[0]);
          ub_data[4] = "Unapplied for tileUbDataDN";
          ub_data[5] = "1";
          ub_data[6] = PrintExpr(shape[0]);
          // when ub_data[6] * sizeof(dtype) < 32, do this
          int num1 = std::stoi(ub_data[6]);
          if (ub_data[0] == "int" && num1 < 8) {
            ub_data[6] = "8";
          } else if (ub_data[0] == "float" && num1 < 8) {
            ub_data[6] = "8";
          } else if (ub_data[0] == "half" && num1 < 16) {
            ub_data[6] = "16";
          }
        } else if (shape.size() == 2) {
          if (shape[1].as<IntImmNode>()->value != 1) {
            if (!buffer_in_pipeline) {
              ub_data[1] = PrintExpr(shape[0]);
              ub_data[2] = PrintExpr(shape[1]);
              ub_data[4] = "Unapplied for tileUbDataDN";
              ub_data[5] = PrintExpr(shape[0]);
              ub_data[6] = PrintExpr(shape[1]);
            } else {
              ub_data[1] = "1";
              ub_data[2] = PrintExpr(shape[1]);
              ub_data[4] = "Unapplied for tileUbDataDN";
              ub_data[5] = "1";
              ub_data[6] = PrintExpr(shape[1]);
            }
          } else {
            ub_data[1] = "1";
            ub_data[2] = PrintExpr(shape[0]);
            ub_data[4] = "Unapplied for tileUbDataDN";
            ub_data[5] = "1";
            ub_data[6] = PrintExpr(shape[0]);
            shape.pop_back();
          }
        } else if (shape.size() == 3) {
          ub_data[1] = PrintExpr(shape[1]);
          ub_data[2] = PrintExpr(shape[2]);
          ub_data[4] = "Unapplied for tileUbDataDN";
          ub_data[5] = PrintExpr(shape[1]);
          ub_data[6] = PrintExpr(shape[2]);
        }

        // Check if padding is needed at the time of request.
        auto valid = ValidLayoutEnabled(op);
        if (valid) {
          int8_t typeSize = GetTypeLen(type);
          int8_t NDBlockSize = 32 / typeSize;
          int shape2_val = std::stoi(ub_data[2]);
          PrimExpr shape2_expr = IntImm(DataType::Int(32), shape2_val);
          PrimExpr padded_shape2 =
              tvm::floordiv(shape2_expr + NDBlockSize - 1, NDBlockSize) *
              NDBlockSize;
          ub_data[2] = PrintExpr(padded_shape2);
        }

        if (buffer_in_pipeline) {
          // Recording the count of buffer allocations in the pipeline.
          auto bufferNum = prefetch_n_stages_map_[vid].first;
          // prefetch_n_stages_map_[vid] = std::pair<int, int>{bufferNum, 0};

          // Output the valid shape.
          stream << pos << "ND<" << type << ", " << ub_data[1] << ", "
                 << ub_data[2] << ", " << ub_data[5] << ", " << ub_data[6];
          stream << "> " << vid << "[" << bufferNum << "];\n";

          // Batch allocate addresses.
          for (size_t i = 0; i < bufferNum; i++) {
            this->PrintIndent();
            stream << "TASSIGN(" << vid << "[" << i << "], "
                   << PrintExpr(address_offset_[String(pos)]) << ");\n";
            if (ub_data[3].empty()) {
              ub_data[3] = PrintExpr(address_offset_[String(pos)]);
            }
            ub_data_map_[vid] = ub_data;
            address_offset_.Set(String(pos),
                                PrimExpr(int(op->ConstantAllocationSize() *
                                             op->dtype.bytes() / 2)) +
                                    address_offset_[String(pos)]);
          }
        } else {
          stream << pos << "ND<" << type << ", " << ub_data[1] << ", "
                 << ub_data[2] << ", " << ub_data[5] << ", " << ub_data[6];
          stream << "> " << vid << ";\n";
          this->PrintIndent();
          stream << "TASSIGN(" << vid << ", "
                 << PrintExpr(address_offset_[String(pos)]) << ");\n";
          ub_data[3] = PrintExpr(address_offset_[String(pos)]);
          ub_data_map_[vid] = ub_data;
          address_offset_.Set(
              String(pos),
              PrimExpr(int(op->ConstantAllocationSize() * op->dtype.bytes())) +
                  address_offset_[String(pos)]);
        }

      } else {
        if (!buffer_in_pipeline) {
          int dtype_bytes = op->dtype.bytes();
          std::vector<PrimExpr> valid_shapes;
          valid_shapes.reserve(2);
          stream << pos << "<" << type;
          int shape_value = shape[0].as<tvm::tir::IntImmNode>()->value;
          if (shape_value * dtype_bytes % 32 == 0) {
            valid_shapes.push_back(shape[0]);
          } else {
            valid_shapes.push_back(tvm::IntImm(
                shape[0].dtype(),
                shape_value +
                    (32 - shape_value * dtype_bytes % 32) / dtype_bytes));
          }
          valid_shapes.push_back(shape[1]);
          for (size_t i = 0; i < valid_shapes.size(); i++) {
            l_data[i + 1] = PrintExpr(valid_shapes[i]);
            stream << ", " << valid_shapes[i];
          }
          for (size_t i = 0; i < shape.size(); i++) {
            stream << ", " << shape[i];
          }
          stream << "> " << vid << ";\n";
          this->PrintIndent();
          stream << "TASSIGN(" << vid << ", "
                 << PrintExpr(address_offset_[String(pos)]) << ");\n";
          address_offset_.Set(
              String(pos),
              PrimExpr(int(op->ConstantAllocationSize() * op->dtype.bytes())) +
                  address_offset_[String(pos)]);
        } else {
          auto bufferNum = prefetch_n_stages_map_[vid].first;
          // prefetch_n_stages_map_[vid] = std::pair<int, int>{bufferNum, 0};
          int dtype_bytes = op->dtype.bytes();
          std::vector<PrimExpr> valid_shapes;
          valid_shapes.reserve(shape.size() - 1);
          stream << pos << "<" << type;
          int shape_value = shape[0].as<tvm::tir::IntImmNode>()->value;
          if (shape_value * dtype_bytes % 32 == 0) {
            valid_shapes.push_back(shape[0]);
          } else {
            valid_shapes.push_back(tvm::IntImm(
                shape[0].dtype(),
                shape_value +
                    (32 - shape_value * dtype_bytes % 32) / dtype_bytes));
          }
          valid_shapes.push_back(shape[1]);
          for (size_t i = 0; i < valid_shapes.size(); i++) {
            l_data[i + 1] = PrintExpr(valid_shapes[i]);
            stream << ", " << valid_shapes[i];
          }
          for (size_t i = 0; i < shape.size(); i++) {
            stream << ", " << shape[i];
          }
          stream << "> " << vid << "[" << bufferNum << "];\n";
          for (size_t j = 0; j < bufferNum; j++) {
            this->PrintIndent();
            stream << "TASSIGN(" << vid << "[" << j << "], "
                   << PrintExpr(address_offset_[String(pos)]) << ");\n";
            address_offset_.Set(String(pos),
                                PrimExpr(int(op->ConstantAllocationSize() *
                                             op->dtype.bytes())) +
                                    address_offset_[String(pos)]);
          }
        }
        l_data_map_[vid] = l_data;
      }
    }
  };

  if (scope == "wmma.matrix_a") {
    print_buffer("TileLeft");
  } else if (scope == "wmma.matrix_b") {
    print_buffer("TileRight");
  } else if (scope == "wmma.accumulator") {
    print_buffer("TileAcc");
  } else if (scope == "shared.dyn") {
    print_buffer(kAscendPtoScope + "TileMatL1");
  } else if (scope == "shared") {
    print_buffer(kAscendPtoScope + "TileUbData");
  } else if (scope == "local.var") {
    PrimExpr init = tir::make_const(op->dtype, 0);
    std::string init_type = type;
    auto init_it = op->annotations.find(tl::attr::kLocalVarInit);
    if (init_it != op->annotations.end()) {
      PrimExpr user_init = Downcast<PrimExpr>((*init_it).second);
      if (user_init.dtype().is_bool()) {
        init_type = "bool";
      } else if (!user_init.dtype().is_void() &&
                 user_init.dtype() != op->dtype) {
        user_init = tir::Cast(op->dtype, user_init);
        init_type = getType(user_init.dtype());
      }
      init = user_init;
    }
    this->PrintIndent();
    stream << init_type + " " << vid << " = " << PrintExpr(init) << ";\n";
  }
  this->PrintStmt(op->body);
}

inline void PrintConst(const FloatImmNode *op, std::ostream &os,
                       CodeGenTileLangAscendPto *p) { // NOLINT(*)
  // Type code is kBFloat
  if (op->dtype.is_bfloat16()) {
    os << "bfloat16_t";
    os << '(' << std::scientific << op->value << 'f' << ')';
    return;
  }
  // Type code is kFloat8_e5m2 or kE4M4Float
  if (op->dtype.is_float8() || op->dtype.is_float4()) {
    p->PrintType(op->dtype, os);
    os << '(' << std::scientific << op->value << 'f' << ')';
    return;
  }
  // Type code is kFloat
  switch (op->dtype.bits()) {
  case 64:
  case 32: {
    std::ostringstream temp;
    if (std::isinf(op->value)) {
      if (op->value < 0) {
        temp << "-";
      }
      temp << ((op->dtype.bits() == 32) ? "CUDART_INF_F" : "CUDART_INF");
      p->need_math_constants_h_ = true;
    } else if (std::isnan(op->value)) {
      temp << ((op->dtype.bits() == 32) ? "CUDART_NAN_F" : "CUDART_NAN");
      p->need_math_constants_h_ = true;
    } else {
      temp << std::scientific << op->value;
      if (op->dtype.bits() == 32)
        temp << 'f';
    }
    p->MarkConst(temp.str());
    os << temp.str();
    break;
  }
  case 16: {
    os << "half_t" << '(';
    FloatImm const_f32 = FloatImm(DataType::Float(32), op->value);
    PrintConst(const_f32.get(), os, p);
    os << ')';
    break;
  }
  default:
    LOG(FATAL) << "Bad bit-width for float: " << op->dtype << "\n";
  }
}

void CodeGenTileLangAscendPto::VisitExpr_(const FloatImmNode *op,
                                          std::ostream &os) { // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenTileLangAscendPto::PreFunctionBody(const PrimFunc &f) {
  int func_scope = this->BeginScope();
  // this->PrintIndent();

  ICHECK(this->para_.size() % 3 == 0)
      << "CodeGenTileLangAscendPto: parameters should be in pairs of (var, "
         "handle, dtype, shape0, shape1)";

  for (size_t i = 0; i < this->para_.size(); i += 3) {
    copy_base_addr_map_.Set(String(this->para_[i + 1]), String(this->para_[i]));
  }

  this->EndScope(func_scope);
}

void CodeGenTileLangAscendPto::VisitExpr_(const SelectNode *op,
                                          std::ostream &os) {
  auto condition = PrintExpr(op->condition);
  auto true_value = PrintExpr(op->true_value);
  auto false_value = PrintExpr(op->false_value);

  os << "(" << condition << " ? "
     << "" << true_value << " : " << false_value << ")";
}

static void ProcessHostInput(std::ostream &os,
                             std::vector<std::string> &arg_names,
                             std::vector<const tir::VarNode *> &shape_vars,
                             bool add_args = true) {
  for (auto shape_var : shape_vars) {
    os << ", "
       << "int64_t " << shape_var->name_hint;
    if (add_args) {
      arg_names.push_back(shape_var->name_hint);
    }
  }
}

void CodeGenTileLangAscendPto::CallTilingInput(
    std::ostream &os, std::string func_name,
    std::vector<std::string> &tiling_args,
    std::vector<const tir::VarNode *> &shape_vars) {}

void CodeGenTileLangAscendPto::ProcessTilingInput(
    std::ostream &os, std::string func_name,
    std::vector<std::string> &tiling_args,
    std::vector<const tir::VarNode *> &shape_vars) {}

void CodeGenTileLangAscendPto::PrintHostFunc(
    const PrimFunc &f, const std::string &name, std::ostringstream &os,
    std::string &core, std::vector<const tir::VarNode *> &shape_vars) {
  std::vector<std::string> tiling_args;
  std::string tiling_func_name = name;
  // ProcessTilingInput(os, tiling_func_name, tiling_args, shape_vars);

  // launch kernel
  os << "extern \"C\" __global__ AICORE void launch_kernel(";
  std::vector<std::string> arg_names;
  for (size_t i = 0; i < f->params.size(); ++i) { // params
    auto v = f->params[i];
    if (i != 0) {
      os << ", ";
    }
    arg_names.push_back(v->name_hint);
    if (v.dtype() == DataType::Handle()) {
      os << "__gm__ uint8_t *" << v->name_hint;
    } else {
      os << getType(v.dtype()) << " " << v->name_hint;
    }
  }
  ProcessHostInput(os, arg_names, shape_vars);
  int func_scope = this->BeginScope();
  os << ", uint64_t fftsAddr)\n{\n  ";
  this->PrintIndent();
  // template function
  os << name << "(";
  for (size_t i = 0; i < f->params.size(); ++i) { // params
    auto v = f->params[i];
    if (i != 0) {
      os << ",\n     ";
    }
    if (v.dtype() == DataType::Handle()) {
      os << "reinterpret_cast<__gm__ "
         << global_tensor_template[String(v->name_hint)].dtype << " *>("
         << v->name_hint << ")";
    } else {
      os << v->name_hint;
    }
  }
  for (auto shape_var : shape_vars) {
    os << ", " << shape_var->name_hint;
  }
  os << ",\n     reinterpret_cast<uint64_t>(fftsAddr));\n}\n\n";

  // call kernel
  os << "extern \"C\" void call(";
  for (size_t i = 0; i < f->params.size(); ++i) { // params
    auto v = f->params[i];
    if (i != 0) {
      os << ", ";
    }
    if (v.dtype() == DataType::Handle()) {
      os << "uint8_t *" << v->name_hint;
    } else {
      os << getType(v.dtype()) << " " << v->name_hint;
    }
  }
  ProcessHostInput(os, arg_names, shape_vars, false);
  os << ", void *stream)\n{\n  ";
  os << "  uint32_t fftsLen{0};\n  ";
  os << "  uint64_t fftsAddr{0};\n  ";
  os << "  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);\n";
  this->PrintIndent();
  os << "  launch_kernel" << "<<<" << core << ", nullptr, stream>>>(";

  for (auto &arg_name : arg_names) {
    os << arg_name;
    if (arg_name != arg_names.back()) {
      os << ", ";
    }
  }
  if (!tiling_args.empty()) {
    os << ", ";
  }
  for (auto &tiling_data : tiling_args) {
    os << tiling_data;
    if (tiling_data != tiling_args.back()) {
      os << ", ";
    }
  }
  os << ", fftsAddr);\n}\n";
  this->EndScope(func_scope);
  std::string content = os.str();
}

void CodeGenTileLangAscendPto::AddFunction(const GlobalVar &gvar,
                                           const PrimFunc &f) {
  CodeGenC::DeclareFunction(gvar, f);
  // clear previous generated state.
  this->InitFuncState(f);

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);

  address_map_ = f->GetAttr<Map<Var, PrimExpr>>("address_map")
                     .value_or(Map<Var, PrimExpr>());
  use_swizzle_ = f->GetAttr<Bool>("use_swizzle").value_or(Bool(false));
  // tiling_map_ = f->GetAttr<Map<Var,
  // PrimExpr>>("tiling_map").value_or(Map<Var, PrimExpr>());
  buffer_shapess_ = f->GetAttr<Map<Var, Array<PrimExpr>>>("buffer_shapess")
                        .value_or(Map<Var, Array<PrimExpr>>());
  buffer_versions_ = f->GetAttr<Map<Var, PrimExpr>>("buffer_versions")
                         .value_or(Map<Var, PrimExpr>());
  var_sequence_ = f->GetAttr<Array<Var>>("var_sequence").value_or(Array<Var>());
  ICHECK(global_symbol.defined())
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";
  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);

  this->PrintFuncPrefix(stream);
  this->stream << "AICORE ";
  CodeGenC::PrintType(f->ret_type, stream);

  auto func_name = static_cast<std::string>(global_symbol.value()) + "_kernel";
  this->stream << " " << func_name << "(";
  std::vector<const tir::VarNode *> shape_vars;

  for (size_t i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];
    std::string vid = AllocVarID(v.get());
    if (f->buffer_map.find(v) != f->buffer_map.end()) {
      tir::Buffer buffer = f->buffer_map[v];
      for (size_t j = 0; j < buffer->shape.size(); j++) {
        auto shape_var = buffer->shape[j].as<VarNode>();
        if ((std::find(shape_vars.begin(), shape_vars.end(), shape_var) ==
             shape_vars.end()) &&
            shape_var != 0) {
          (void)AllocVarID(shape_var);
          shape_vars.push_back(shape_var);
        }
      }
    }

    if (i != 0)
      stream << ", ";
    if (v.dtype().is_handle()) {
      auto real_v = f->buffer_map[v]->data;
      this->para_.push_back(vid);
      // vid = AllocVarID(real_v.get());
      this->para_.push_back(AllocVarID(real_v.get()));
      this->para_.push_back(getType(f->buffer_map[v]->dtype));
      Array<String> copy_tmp_shape = {};
      String shape_type = "static";
      for (size_t i = 0; i < f->buffer_map[v]->shape.size(); i++) {
        std::string shape_info = PrintExpr(f->buffer_map[v]->shape[i]);
        copy_tmp_shape.push_back(shape_info);
        if (shape_info[0] < '1' || shape_info[0] > '9')
          shape_type = "dynamic";
      }
      global_tensor gt = {shape_type, String(getType(f->buffer_map[v]->dtype)),
                          copy_tmp_shape};
      global_tensor_template[String(vid)] = gt;

      PrintRestrict(v, stream);

      auto it = alloc_storage_scope_.find(v.get());
      if (it != alloc_storage_scope_.end()) {
        PrintStorageScope(it->second, stream);
      }

      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
        }
      }
    } else {
      CodeGenC::PrintType(GetType(v), stream);
      stream << " " << vid;
    }
    if (v.dtype() == DataType::Handle()) {
      stream << "__gm__ " << getType(f->buffer_map[v]->dtype) << " *" << vid;
    }
  }
  size_t index = 0;
  if (shape_vars.size() != 0 && f->params.size() != 0) {
    stream << ", ";
  }
  for (auto shape_var : shape_vars) {
    stream << "int64_t" << " " << GetVarID(shape_var);
    if (index != shape_vars.size() - 1) {
      stream << ", ";
    }
    index++;
  }

  stream << ", uint64_t ffts_Addr) {\n";
  this->PreFunctionBody(f);
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";

  PrintHostFunc(f, func_name, stream, this->core_num_, shape_vars);
  std::string content = stream.str();
}

void CodeGenTileLangAscendPto::AutoBarrierCodegen(const CallNode *op) {
  this->PrintIndent();
  std::string pipeline = "PIPE_ALL";
  if (op->args.size() >= 1) {
    if (auto pipeline_imm = op->args[0].as<StringImmNode>()) {
      pipeline = pipeline_imm->value;
    }
  }
  this->stream << "pipe_barrier(" << pipeline << ");\n";
}

void CodeGenTileLangAscendPto::AutoFlagOpCodegen(const CallNode *op,
                                                 std::string op_name) {
  this->PrintIndent();

  std::string event_type;
  if (auto pipeline_imm = op->args[0].as<StringImmNode>()) {
    event_type = pipeline_imm->value;
  } else {
    LOG(FATAL) << "Expected StringImm for event_type";
    return;
  }

  size_t pos = event_type.find('_');

  if (pos == 0 || pos == event_type.length() - 1) {
    LOG(FATAL) << "Invalid event_type format: " << event_type;
    return;
  }
  std::string src = event_type.substr(0, pos);
  std::string dst = event_type.substr(pos + 1);

  auto event_id = PrintExpr(op->args[1]);
  this->stream << op_name << "(PIPE_" << src << ", " << "PIPE_" << dst << ", "
               << "EVENT_ID" << event_id << ");\n";
}

void CodeGenTileLangAscendPto::SelectCodegen(const CallNode *op) {
  ShapeInfo src0_shape_info = GetSliceInfo(op->args[2].as<CallNode>());
  ShapeInfo dst_shape_info = GetSliceInfo(op->args[0].as<CallNode>());

  std::string mask_name = PrintBufferOffset(op->args[1].as<CallNode>());
  std::string src1_name;
  std::string op_name;

  DataType src0_dtype = GetAccessPtrDtypePto(op->args[2].as<CallNode>());
  std::string src0_type = getType(src0_dtype);

  int src1_type = std::stoi(PrintExpr(op->args[3]));
  if (src1_type == 2) {
    src1_name = PrintBufferOffset(op->args[4].as<CallNode>());
    op_name = "TSEL";
  } else if (src1_type == 1) {
    std::string scalar_value = PrintExpr(op->args[4]);
    std::string temp_name = GetTempVarName("select_scalar_tmp");
    this->PrintIndent();
    this->stream << kAscendPtoScope << "TileUbDataND<" << src0_type << ", 1, " 
                 << dst_shape_info.slice_col << ", 1, " << dst_shape_info.slice_col 
                 << "> " << temp_name << ";\n";
    this->PrintIndent();
    this->stream << "TASSIGN(" << temp_name << ", " << dst_shape_info.first_addr 
                 << " + " << dst_shape_info.offset << " * " << GetTypeLen(src0_type) << ");\n";
    this->PrintIndent();
    this->stream << kAscendPtoScope << "fill_scalar(" << temp_name << ", static_cast<" << src0_type << ">(" 
                 << scalar_value << "));\n";
    src1_name = temp_name;
    op_name = "TSEL";
  } else {
    LOG(FATAL) << "CodeGenAscendPto: Select currently only supports "
                  "src1_type=2 (Tensor-Tensor mode). "
               << "Got type=" << src1_type;
  }
  if (src0_shape_info.is_slice || dst_shape_info.is_slice) {
    std::string src0_temp_name = GetTempVarName(src0_shape_info.ub_name);
    std::string dst_temp_name = GetTempVarName(dst_shape_info.ub_name);
    CreateUbVariableND(src0_temp_name, src0_shape_info);
    CreateUbVariableND(dst_temp_name, dst_shape_info);
    this->PrintIndent();
    this->stream << op_name << "(" << dst_temp_name << ", " << mask_name << ", "
                 << src0_temp_name << ", " << src1_name << ");\n";
  } else {
    this->PrintIndent();
    this->stream << op_name << "(" << dst_shape_info.ub_name << ", "
                 << mask_name << ", " << src0_shape_info.ub_name << ", "
                 << src1_name << ");\n";
  }
}

} // namespace codegen
} // namespace tvm
