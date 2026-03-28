// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file target/codegen_ascend_pto.cc
 */

#include "codegen_ascend_pto.h"
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/registry.h>
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
#include "../transform/common/attr.h"

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

using BufferInfo = CodeGenTileLangAscendPto::BufferInfo;

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
    LOG(FATAL) << "Unexpected type for access_ptr first argument: "
               << type_arg->GetTypeKey();
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

int GetValidShape(int shape, const std::string &dtype) {
  int dtype_len = GetTypeLen(dtype);
  int shape_mod = shape * GetTypeLen(dtype) % 32;
  if (shape_mod == 0) {
    return shape;
  }
  return shape + (32 - shape_mod) / dtype_len;
}

std::string CodeGenTileLangAscendPto::GetVarId(const Var &var) const {
  auto it = var_idmap_.find(var.get());
  return (it != var_idmap_.end() && !it->second.empty())
             ? it->second
             : std::string(var->name_hint);
}

BufferInfo CodeGenTileLangAscendPto::GetBufferInfo(const PrimExpr &arg) const {
  auto *access_ptr = arg.as<CallNode>();
  ICHECK(access_ptr)
      << "Argument is not a CallNode representing a buffer access.";

  BufferInfo info;
  info.access_ptr = access_ptr;
  info.var = Downcast<Var>(access_ptr->args[1]);
  info.id = GetVarId(info.var);
  info.offset = access_ptr->args[2];
  info.dtype = access_ptr->args[0].as<CallNode>()->dtype;
  ICHECK(buffer_shapess_.count(info.var))
      << "Buffer shape not found for: " << info.var->name_hint;
  info.shape = buffer_shapess_.at(info.var);
  return info;
}

std::string
CodeGenTileLangAscendPto::GetTempVarName(const std::string &temp_name) {
  return temp_name + "_" + "temp" + "_" +
         std::to_string(counters_[temp_name]++);
}

void CodeGenTileLangAscendPto::CreateUbVariableND(const std::string &temp_name,
                                                  const ShapeInfo &shape_info) {
  this->PrintIndent();
  this->stream << kAscendPtoScope << "TileUbDataND<" << shape_info.type << ", "
               << shape_info.slice_row << ", " << shape_info.slice_col << ", "
               << shape_info.slice_valid_row << ", "
               << shape_info.slice_valid_col << "> " << temp_name << ";\n";

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
               << shape_info.slice_valid_col << ", "
               << shape_info.slice_valid_row << "> " << temp_name << ";\n";

  this->PrintIndent();
  this->stream << "TASSIGN(" << temp_name << ", " << shape_info.first_addr
               << " + " << shape_info.offset << " * "
               << GetTypeLen(shape_info.type) << ");\n";
}

void CodeGenTileLangAscendPto::CreateCubeVariable(
    const std::string &temp_name, const ShapeInfo &shape_info,
    const std::string &tile_name) {
  int32_t slice_row = shape_info.slice_row;
  int32_t slice_col = shape_info.slice_col;

  this->PrintIndent();
  this->stream << tile_name << "<" << shape_info.type << ", " << slice_row
               << ", " << slice_col << ", " << slice_row << ", " << slice_col
               << "> " << temp_name << ";\n";

  this->PrintIndent();
  this->stream << "TASSIGN(" << temp_name << ", " << shape_info.first_addr
               << " + " << shape_info.offset << " * "
               << GetTypeLen(shape_info.type) << ");\n";
}

ShapeInfo CodeGenTileLangAscendPto::GetSliceInfo(const CallNode *op) {
  ICHECK(op);
  ICHECK(op->op.same_as(builtin::tvm_access_ptr()));

  Var buffer_var = Downcast<Var>(op->args[1]);

  ICHECK(buffer_shapess_.count(buffer_var))
      << "Buffer shape not found: " << buffer_var->name_hint;
  auto shape = buffer_shapess_.at(buffer_var);

  int32_t row = 1;
  int32_t col = 1;
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
    ICHECK(shape[0]->IsInstance<IntImmNode>()) << "Shape[0] is not IntImm!";
    ICHECK(shape[1]->IsInstance<IntImmNode>()) << "Shape[1] is not IntImm!";
    row = shape[0].as<IntImmNode>()->value;
    col = shape[1].as<IntImmNode>()->value;
  }

  int32_t extent = op->args[3].as<IntImmNode>()->value;
  int32_t slice_valid_row = (extent / col) > 1 ? (extent / col) : 1;
  int32_t slice_valid_col = extent > col ? col : extent;

  ICHECK(buffer_address_map_.count(buffer_var))
      << "Buffer address not found: " << buffer_var->name_hint;
  auto src_addr = buffer_address_map_.at(buffer_var);
  auto offset = PrintExpr(op->args[2]);

  auto type = getType(op->args[0].dtype());

  bool is_slice;
  if (shape.size() == 1) {
    is_slice = extent != col;
  } else {
    is_slice = extent != row * col;
  }

  int32_t slice_row = slice_valid_row;
  int32_t slice_col = GetValidShape(slice_valid_col, type);

  auto ub_name = var_idmap_[op->args[1].as<VarNode>()];
  return ShapeInfo{
      row,    col,      slice_row, slice_col, slice_valid_row, slice_valid_col,
      extent, src_addr, offset,    type,      ub_name,         is_slice};
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

void CodeGenTileLangAscendPto::VisitExpr_(const CallNode *op,
                                          std::ostream &os) {
  if (op->op.same_as(builtin::call_extern())) {
    CallExternCodegen(op);
  } else if (op->op.same_as(tl::loop_break())) {
    this->PrintIndent();
    this->stream << "break;\n";
  } else if (op->op.same_as(tl::ascend_gemm_v0())) {
    GemmV0Codegen(op);
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
  } else if (op->op.same_as(tl::ascend_mma())) {
    MmaCodegen(op);
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

std::string CodeGenTileLangAscendPto::PrintBufferOffset(const CallNode *op) {
  auto _var = op->args[1].as<VarNode>();
  std::string _var_name = var_idmap_[_var];
  return _var_name;
}

// merge shape's lower dimensions based on srcN
Array<PrimExpr> MergeShapeBySrcN(const Array<PrimExpr> &shape,
                                 const PrimExpr &srcN,
                                 tvm::arith::Analyzer *analyzer) {
  Array<PrimExpr> merged_shape;
  int count = 0;
  const auto *srcN_imm = analyzer->Simplify(srcN).as<IntImmNode>();
  if (srcN_imm && !shape.empty()) {
    int64_t srcN_val = srcN_imm->value;
    int64_t tmp_val = srcN_val;
    // Divide from the lowest dimension, calculate how many dimensions srcN
    // covers
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      const auto *dim_imm = analyzer->Simplify(shape[i]).as<IntImmNode>();
      if (dim_imm && tmp_val > 1) {
        count++;
        tmp_val /= dim_imm->value;
        if (tmp_val == 1) {
          break; // Perfectly divided, stop merging
        }
      } else {
        break; // Encountered dynamic dimension or cannot divide evenly, stop
               // merging
      }
    }
    // If successfully merged more than 1 dimension and perfectly divided
    // (tmp_val == 1)
    if (count > 1 && tmp_val == 1) {
      for (size_t i = 0; i < shape.size() - count; ++i) {
        merged_shape.push_back(shape[i]);
      }
      // Use the merged continuous dimension as the new lowest dimension
      merged_shape.push_back(srcN);
    } else {
      // Cannot merge or no need to merge, keep original state
      merged_shape = shape;
    }
  } else {
    // srcN is dynamic expression or shape is empty, keep original state
    merged_shape = shape;
  }
  return merged_shape;
}

Array<PrimExpr> ComputeStrides(const Array<PrimExpr> &shape, PrimExpr srcN) {
  constexpr int kMaxDims = 5;
  tvm::arith::Analyzer analyzer;
  int count = 0;

  // =====================================================================
  // 1. Dimension Merging Logic (Shape Flattening)
  // If srcN spans multiple inner dimensions, merge them logically
  // =====================================================================
  Array<PrimExpr> merged_shape = MergeShapeBySrcN(shape, srcN, &analyzer);

  // =====================================================================
  // 2. Standard Stride Computation (based on merged_shape)
  // =====================================================================
  int ndim = static_cast<int>(merged_shape.size());
  // Ensure at least 5 outputs for safe truncation later
  int out_dims = std::max(kMaxDims, ndim + 1);
  std::vector<PrimExpr> strides_vec(out_dims, Integer(1));

  PrimExpr current_stride = Integer(1);
  int stride_idx = out_dims - 1;

  // Calculate stride from the last dimension, write to strides_vec end
  for (int i = ndim - 1; i >= 0; --i, stride_idx--) {
    strides_vec[stride_idx] = current_stride;
    current_stride = analyzer.Simplify(current_stride * merged_shape[i]);
  }
  strides_vec[stride_idx] = current_stride;

  // Convert to TVM Array and return
  Array<PrimExpr> strides;
  for (const auto &s : strides_vec) {
    strides.push_back(s);
  }
  return strides;
}

std::tuple<bool, std::string, std::string>
FormatStrides(const Array<PrimExpr> &shape, const Array<PrimExpr> &strides) {
  bool is_dynamic = false;
  std::stringstream stride_ss;

  // =====================================================================
  // 1) Generate stride string: take last 5 strides (contiguous memory layout)
  // =====================================================================
  size_t total_strides = strides.size();
  size_t start_idx = total_strides > 5 ? total_strides - 5 : 0;

  for (size_t i = start_idx; i < total_strides; ++i) {
    if (const auto *int_imm = strides[i].as<IntImmNode>()) {
      stride_ss << int_imm->value;
    } else {
      stride_ss << "-1"; // Has PrimExpr variable, set to -1
      is_dynamic = true;
    }
    if (i + 1 < total_strides) {
      stride_ss << ", ";
    }
  }

  // =====================================================================
  // 2) Generate bracket parameter string: traverse shape to decide
  // =====================================================================
  std::stringstream ctor_args_ss;
  bool first = true;

  // Core insight: bottom 5 strides only depend on last 4 dimensions of shape!
  // (e.g., for 7D tensor, first 3 dimensions don't participate in stride
  // computation) Therefore, only dynamic variables in these last dimensions
  // need template parameters
  int shape_size = static_cast<int>(shape.size());
  int shape_start = std::max(0, shape_size - 4);

  for (int i = shape_start; i < shape_size; ++i) {
    if (!shape[i]->IsInstance<IntImmNode>()) {
      if (!first) {
        ctor_args_ss << ", ";
      }
      ctor_args_ss << shape[i];
      first = false;
      is_dynamic =
          true; // Mark as dynamic if any dynamic shapes in relevant range
    }
  }

  return {is_dynamic, stride_ss.str(), ctor_args_ss.str()};
}

std::string CodeGenTileLangAscendPto::GetPadEnum(const PrimExpr value) {
  std::string value_str = PrintExpr(value);

  std::string pad_value_enum = "pto::PadValue::Null";
  if (value_str.find("-CUDART_INF") != std::string::npos ||
      value_str.find("-inf") != std::string::npos ||
      value_str.find("-INFINITY") != std::string::npos ||
      value_str == "-std::numeric_limits<float>::infinity()") {
    pad_value_enum = "pto::PadValue::Min";
  } else if (value_str.find("CUDART_INF") != std::string::npos ||
             value_str.find("+inf") != std::string::npos ||
             value_str.find("INFINITY") != std::string::npos ||
             value_str == "std::numeric_limits<float>::infinity()") {
    pad_value_enum = "pto::PadValue::Max";
  } else if (value_str == "0" || value_str == "0.0" || value_str == "0.0f" ||
             value_str.find("0.000000e+00") != std::string::npos ||
             value_str.find("0e+00") != std::string::npos) {
    pad_value_enum = "pto::PadValue::Zero";
  }

  return pad_value_enum;
}

void CodeGenTileLangAscendPto::GMCopyCall(const CallNode *call,
                                          std::string op_name) {
  static const std::unordered_map<std::string, bool> kIsGmToLocalOp = {
      {"copy_gm_to_ub", true},
      {"copy_ub_to_gm", false},
      {"copy_gm_to_l1", true},
      {"copy_l0c_to_gm", false}};

  ICHECK(kIsGmToLocalOp.count(op_name))
      << "Unsupported GM copy op: " << op_name;
  bool is_load = kIsGmToLocalOp.at(op_name);

  BufferInfo src_info = GetBufferInfo(call->args[1]);
  BufferInfo dst_info = GetBufferInfo(call->args[2]);

  const auto &gm_info = is_load ? src_info : dst_info;
  const auto &local_info = is_load ? dst_info : src_info;

  ShapeInfo slice_info = GetSliceInfo(local_info.access_ptr);
  int32_t shape4 =
      slice_info.is_slice ? slice_info.slice_valid_row : slice_info.row;
  int32_t shape5 =
      slice_info.is_slice ? slice_info.slice_valid_col : slice_info.col;
  std::string shape_tmpl =
      "1, 1, 1, " + std::to_string(shape4) + ", " + std::to_string(shape5);

  auto strides = ComputeStrides(gm_info.shape, call->args[3]);
  auto [is_dynamic, stride_tmpl, stride_param] =
      FormatStrides(gm_info.shape, strides);
  if (is_dynamic) {
    op_name += "_dynamic";
  }

  this->PrintIndent();
  stream << kAscendPtoScope << op_name << "<" << getType(gm_info.dtype) << ", "
         << getType(local_info.dtype) << ", " << shape_tmpl << ", "
         << stride_tmpl << ", ";
  if (op_name.rfind("copy_gm_to_ub", 0) == 0) {
    stream << slice_info.slice_valid_row << ", " << slice_info.slice_valid_col
           << ", ";
    stream << GetPadEnum(call->args[6]);
  } else if (op_name.rfind("copy_ub_to_gm", 0) == 0) {
    stream << slice_info.slice_valid_row << ", " << slice_info.slice_valid_col;
  } else {
    stream << slice_info.slice_valid_row << ", " << slice_info.slice_valid_col;
  }
  stream << ">(";

  // gm addr
  stream << copy_base_addr_map_.at(gm_info.id) << " + "
         << PrintExpr(gm_info.offset);

  if (is_dynamic) {
    stream << ", pto::Shape<" << shape_tmpl << ">()"
           << ", pto::Stride<" << stride_tmpl << ">(" << stride_param << ")";
  }

  stream << ", " << PrintExpr(buffer_address_map_.at(local_info.var)) << ", "
         << PrintExpr(local_info.offset) << ", " << PrintExpr(call->args[4])
         << ", " << PrintExpr(call->args[5]) << ");\n";
}

void CodeGenTileLangAscendPto::CopyUBToUBCodegen(const CallNode *call) {
  BufferInfo src_info = GetBufferInfo(call->args[1]);
  BufferInfo dst_info = GetBufferInfo(call->args[2]);

  bool is_cast = src_info.dtype != dst_info.dtype;
  std::string api_name = is_cast ? "TCVT" : "TMOV";

  ShapeInfo src_shape_info = GetSliceInfo(src_info.access_ptr);
  ShapeInfo dst_shape_info = GetSliceInfo(dst_info.access_ptr);

  std::string src_name = src_info.id;
  std::string dst_name = dst_info.id;
  if (src_shape_info.is_slice) {
    src_name = GetTempVarName(src_shape_info.ub_name);
    CreateUbVariableND(src_name, src_shape_info);
  }

  if (dst_shape_info.is_slice) {
    dst_name = GetTempVarName(dst_shape_info.ub_name);
    CreateUbVariableND(dst_name, dst_shape_info);
  }

  this->PrintIndent();
  this->stream << api_name << "(" << dst_name << ", " << src_name;

  if (is_cast) {
    this->stream << ", pto::RoundMode::CAST_NONE";
  }
  this->stream << ");\n";
}

void CodeGenTileLangAscendPto::CopyL1ToL0Codegen(const CallNode *call,
                                                 bool is_a) {
  BufferInfo src_info = GetBufferInfo(call->args[1]);
  BufferInfo dst_info = GetBufferInfo(call->args[2]);

  std::string api_name = is_a ? "copy_l1_to_l0a" : "copy_l1_to_l0b";
  std::string tile_name = is_a ? "TileMatL0A" : "TileMatL0B";

  ShapeInfo src_shape_info = GetSliceInfo(src_info.access_ptr);
  ShapeInfo dst_shape_info = GetSliceInfo(dst_info.access_ptr);

  PrimExpr index_row = floordiv(src_info.offset, src_info.shape[1]);
  PrimExpr index_col = floormod(src_info.offset, src_info.shape[1]);

  this->PrintIndent();
  this->stream << kAscendPtoScope << api_name << "<" << src_shape_info.type
               << ", " << dst_shape_info.slice_row << ", "
               << dst_shape_info.slice_col << ", " << src_shape_info.slice_row
               << ", " << src_shape_info.slice_col << "> ";
  if (src_shape_info.is_slice || dst_shape_info.is_slice) {
    std::string src_temp_name = GetTempVarName(src_shape_info.ub_name);
    std::string dst_temp_name = GetTempVarName(dst_shape_info.ub_name);
    CreateCubeVariable(src_temp_name, src_shape_info,
                       kAscendPtoScope + "TileMatL1");
    CreateCubeVariable(dst_temp_name, dst_shape_info,
                       kAscendPtoScope + "TileMatL0B");
    this->stream << "(" << dst_temp_name << ", " << src_temp_name
                 << PrintExpr(index_row) << ", " << PrintExpr(index_col)
                 << ");\n";
  } else {
    this->stream << "(" << dst_info.id << ", " << src_info.id << ", "
                 << PrintExpr(index_row) << ", " << PrintExpr(index_col)
                 << ");\n";
  }
}

void CodeGenTileLangAscendPto::CallExternCodegen(const CallNode *op) {
  std::string op_name = Downcast<StringImm>(op->args[0])->value;

  if (op_name.find("tl::ascend::copy_gm_to_ub") != std::string::npos) {
    GMCopyCall(op, "copy_gm_to_ub");
  } else if (op_name.find("tl::ascend::copy_ub_to_gm") != std::string::npos) {
    GMCopyCall(op, "copy_ub_to_gm");
  } else if (op_name.find("tl::ascend::copy_gm_to_l1") != std::string::npos) {
    GMCopyCall(op, "copy_gm_to_l1");
  } else if (op_name.find("tl::ascend::copy_l0c_to_gm") != std::string::npos) {
    GMCopyCall(op, "copy_l0c_to_gm");
  } else if (op_name.find("tl::ascend::copy_ub_to_ub") != std::string::npos) {
    CopyUBToUBCodegen(op);
  } else if (op_name.find("tl::ascend::copy_l1_to_l0a") != std::string::npos) {
    CopyL1ToL0Codegen(op, true);
  } else if (op_name.find("tl::ascend::copy_l1_to_l0b") != std::string::npos) {
    CopyL1ToL0Codegen(op, false);
  }
}

void CodeGenTileLangAscendPto::GemmV0Codegen(const CallNode *op) {
  std::string template_args = Downcast<StringImm>(op->args[0])->value;

  ShapeInfo a_info = GetSliceInfo(op->args[1].as<CallNode>());
  ShapeInfo b_info = GetSliceInfo(op->args[2].as<CallNode>());
  ShapeInfo c_info = GetSliceInfo(op->args[3].as<CallNode>());

  std::map<std::string, std::string> params =
      extractTemplateParams(template_args);
  uint32_t K = std::stoi(params["K"]);
  uint32_t kL0Size = 128; // L0 slice size, adapted for 64K memory limit
  uint32_t kL0split = (K + kL0Size - 1) / kL0Size; // slice count
  uint32_t kL0Tail = K - (kL0split - 1) * kL0Size; // last block size

  std::string a_name = a_info.ub_name;
  std::string b_name = b_info.ub_name;
  std::string c_name = c_info.ub_name;
  if (a_info.is_slice) {
    a_name = GetTempVarName(a_name);
    CreateCubeVariable(a_name, a_info, kAscendPtoScope + "TileMatL1");
  }

  if (b_info.is_slice) {
    b_name = GetTempVarName(b_name);
    CreateCubeVariable(b_name, b_info, kAscendPtoScope + "TileMatL1");
  }

  if (c_info.is_slice) {
    c_name = GetTempVarName(c_name);
    CreateCubeVariable(c_name, c_info, "pto::TileAcc");
  }

  this->PrintIndent();
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
  this->stream << a_name << ", " << b_name << ", " << c_name << ", "
               << PrintExpr(op->args[4]) << ");\n";
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

void CodeGenTileLangAscendPto::CreateVecIndexCodegen(
    const CallNode *op, const std::string &op_name) {
  BufferInfo dst_info = GetBufferInfo(op->args[1]);
  ShapeInfo dst_slice_info = GetSliceInfo(op->args[1].as<CallNode>());
  std::string first_value = PrintExpr(op->args[2]);

  const auto &M = dst_info.shape[0];
  const auto &N = dst_info.shape[1];

  this->PrintIndent();
  this->stream << kAscendPtoScope << "tci" << "<" << getType(dst_info.dtype)
               << ", " << PrintExpr(M) << ", " << PrintExpr(N) << ">"
               << "(" << PrintExpr(dst_slice_info.first_addr) << ", "
               << dst_slice_info.offset << ", "
               << GetTypeLen(dst_slice_info.type) << ", " << first_value
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
  BufferInfo dst_info = GetBufferInfo(op->args[1]);
  BufferInfo src_info = GetBufferInfo(op->args[2]);

  if (op->args[3].as<CallNode>()) {
    this->PrintIndent();
    std::string idx_name = PrintExpr(op->args[3].as<CallNode>()->args[1]);
    this->stream << op_name << "(" << dst_info.id << ", " << src_info.id << ", "
                 << idx_name << ");\n";
  } else {
    std::string src1Pattern = Downcast<StringImm>(op->args[3])->value;
    this->PrintIndent();
    this->stream << op_name << "<" << kAscendPtoScope << "TileUbDataND<"
                 << getType(dst_info.dtype) << ", " << dst_info.shape[0] << ", "
                 << dst_info.shape[1] << ", " << dst_info.shape[2] << ", "
                 << dst_info.shape[3] << ">, " << kAscendPtoScope
                 << "TileUbDataND<" << getType(src_info.dtype) << ", "
                 << src_info.shape[0] << ", " << src_info.shape[1] << ", "
                 << src_info.shape[2] << ", " << src_info.shape[3] << ">, "
                 << "MaskPattern::" << src1Pattern << ">(" << dst_info.id
                 << ", " << src_info.id << ");\n";
  }
}

void CodeGenTileLangAscendPto::PowCodegen(const CallNode *op) {
  ShapeInfo src0_shape_info = GetSliceInfo(op->args[1].as<CallNode>());
  ShapeInfo src1_shape_info = GetSliceInfo(op->args[2].as<CallNode>());
  ShapeInfo dst_shape_info = GetSliceInfo(op->args[0].as<CallNode>());
  ShapeInfo temp_shape_info = GetSliceInfo(op->args[3].as<CallNode>());

  if (src0_shape_info.is_slice || src1_shape_info.is_slice || dst_shape_info.is_slice) {
    auto src0_temp_name = GetTempVarName(src0_shape_info.ub_name);
    auto src1_temp_name = GetTempVarName(src1_shape_info.ub_name);
    auto dst_temp_name = GetTempVarName(dst_shape_info.ub_name);
    CreateUbVariableND(src0_temp_name, src0_shape_info);
    CreateUbVariableND(src1_temp_name, src1_shape_info);
    CreateUbVariableND(dst_temp_name, dst_shape_info);
    this->PrintIndent();
    this->stream << kAscendPtoScope << "pow" << "<" << dst_shape_info.type << ", "
               << dst_shape_info.slice_row << ", " << dst_shape_info.slice_col << "," << temp_shape_info.row << ">" 
               << "(" << dst_temp_name << ", " << src0_temp_name << ", " << src1_temp_name << ", " << temp_shape_info.ub_name
               << ");\n";
  } else {
  this->PrintIndent();
  this->stream << kAscendPtoScope << "pow" << "<" << dst_shape_info.type << ", "
               << dst_shape_info.row << ", " << dst_shape_info.col << ", " << temp_shape_info.row << ">"
               << "(" << dst_shape_info.ub_name << ", " << src0_shape_info.ub_name << ", " << src1_shape_info.ub_name
               << ", " << temp_shape_info.ub_name << ");\n";
  }
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
  this->stream << "  tl::ascend_pto::TileUbDataND<" << type
               << ", 16, 16, 16, 16> tmp_ub;\n";
  this->PrintIndent();
  this->stream << "  pto::TASSIGN(tmp_ub, 2048);\n";
  this->PrintIndent();
  this->stream << "  tl::ascend_pto::transpose<" << type << ", 16, 16>("
               << dst_name << ", " << src_name << ", tmp_ub);\n";
  this->PrintIndent();
  this->stream << "}\n";
}

void CodeGenTileLangAscendPto::XorCodegen(const CallNode *op,
                                          const std::string &op_name) {
  ShapeInfo src0_shape_info = GetSliceInfo(op->args[1].as<CallNode>());
  ShapeInfo src1_shape_info = GetSliceInfo(op->args[2].as<CallNode>());
  ShapeInfo dst_shape_info = GetSliceInfo(op->args[0].as<CallNode>());
  auto tmp_name = PrintExpr(op->args[3].as<CallNode>()->args[1]);
  
  if (src0_shape_info.is_slice || src1_shape_info.is_slice || dst_shape_info.is_slice) {
    auto src0_temp_name = GetTempVarName(src0_shape_info.ub_name);
    auto src1_temp_name = GetTempVarName(src1_shape_info.ub_name);
    auto dst_temp_name = GetTempVarName(dst_shape_info.ub_name);
    CreateUbVariableND(src0_temp_name, src0_shape_info);
    CreateUbVariableND(src1_temp_name, src1_shape_info);
    CreateUbVariableND(dst_temp_name, dst_shape_info);
    this->PrintIndent();
    this->stream << op_name << "(" << dst_temp_name << ", " << src0_temp_name << ", "
               << src1_temp_name << ", " << tmp_name << ");\n";
  } else {
  this->PrintIndent();
  this->stream << op_name << "(" << dst_shape_info.ub_name << ", " << src0_shape_info.ub_name << ", "
               << src1_shape_info.ub_name << ", " << tmp_name << ");\n";
  }
}

void CodeGenTileLangAscendPto::CompareCodegen(const CallNode *op,
                                              const std::string &op_name) {
  ShapeInfo src0_shape_info = GetSliceInfo(op->args[1].as<CallNode>());
  ShapeInfo src1_shape_info = GetSliceInfo(op->args[2].as<CallNode>());
  ShapeInfo dst_shape_info = GetSliceInfo(op->args[0].as<CallNode>());
  auto mode = Downcast<StringImm>(op->args[3])->value;
  if (src0_shape_info.is_slice || src1_shape_info.is_slice || dst_shape_info.is_slice) {
    auto src0_temp_name = GetTempVarName(src0_shape_info.ub_name);
    auto src1_temp_name = GetTempVarName(src1_shape_info.ub_name);
    auto dst_temp_name = GetTempVarName(dst_shape_info.ub_name);
    CreateUbVariableND(src0_temp_name, src0_shape_info);
    CreateUbVariableND(src1_temp_name, src1_shape_info);
    CreateUbVariableND(dst_temp_name, dst_shape_info);
    this->PrintIndent();
    this->stream << kAscendPtoScope << "compare(" << dst_temp_name << ", " << src0_temp_name << ", " << src1_temp_name
                 << ", " << "CmpMode::" << mode << ");\n";
  } else {
  this->PrintIndent();
  this->stream << kAscendPtoScope << "compare(" << dst_shape_info.ub_name << ", " << src0_shape_info.ub_name
               << ", " << src1_shape_info.ub_name << ", " << "CmpMode::" << mode << ");\n";
  }
}

void CodeGenTileLangAscendPto::CompareScalarCodegen(
    const CallNode *op, const std::string &op_name) {
  ShapeInfo src0_shape_info = GetSliceInfo(op->args[1].as<CallNode>());
  ShapeInfo dst_shape_info = GetSliceInfo(op->args[0].as<CallNode>());
  auto src1_name = PrintExpr(op->args[2]);
  auto mode = Downcast<StringImm>(op->args[3])->value;

  std::string dst_name = PrintExpr(op->args[0].as<CallNode>()->args[1]);
  std::string src0_name = PrintExpr(op->args[1].as<CallNode>()->args[1]);

  DataType src_dtype = GetAccessPtrDtypePto(op->args[1].as<CallNode>());
  DataType scalar_dtype = op->args[2].dtype();
  if (scalar_dtype != src_dtype) {
    std::string target_type = getType(src_dtype);
    src1_name = target_type + "(" + src1_name + ")";
  }
  if (src0_shape_info.is_slice || dst_shape_info.is_slice) {
    auto src0_temp_name = GetTempVarName(src0_shape_info.ub_name);
    auto dst_temp_name = GetTempVarName(dst_shape_info.ub_name);
    CreateUbVariableND(src0_temp_name, src0_shape_info);
    CreateUbVariableND(dst_temp_name, dst_shape_info);
    this->PrintIndent();
    this->stream << kAscendPtoScope << "compare_scalar(" << dst_temp_name << ", " << src0_temp_name << ", " << src1_name
                 << ", " << "CmpMode::" << mode << ");\n";
  } else {
  this->PrintIndent();
  this->stream << kAscendPtoScope << "compare_scalar(" << dst_shape_info.ub_name << ", "
               << src0_shape_info.ub_name << ", " << src1_name << ", " << "CmpMode::" << mode
               << ");\n";
  }
}

void CodeGenTileLangAscendPto::TshCodegen(const CallNode *op,
                                          const std::string &op_name) {
  this->PrintIndent();
  ShapeInfo src0_shape_info = GetSliceInfo(op->args[1].as<CallNode>());
  ShapeInfo dst_shape_info = GetSliceInfo(op->args[0].as<CallNode>());
  auto src1_name = PrintExpr(op->args[2]);

  DataType src_dtype = GetAccessPtrDtypePto(op->args[1].as<CallNode>());
  DataType scalar_dtype = op->args[2].dtype();

  if (scalar_dtype != src_dtype) {
    std::string target_type = getType(src_dtype);
    src1_name = target_type + "(" + src1_name + ")";
  }
  if (src0_shape_info.is_slice || dst_shape_info.is_slice) {
    auto src_temp_name = GetTempVarName(src0_shape_info.ub_name);
    auto dst_temp_name = GetTempVarName(dst_shape_info.ub_name);
    CreateUbVariableND(src_temp_name, src0_shape_info);
    CreateUbVariableND(dst_temp_name, dst_shape_info);
    this->PrintIndent();
    this->stream << op_name << "(" << dst_temp_name << ", " << src_temp_name << ", " << src1_name
                 << ");\n";
  } else {
    this->stream << op_name << "(" << dst_shape_info.ub_name << ", " << src0_shape_info.ub_name << ", "
               << src1_name << ");\n";
  }
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
  if (src0_shape_info.is_slice || src1_shape_info.is_slice ||
      dst_shape_info.is_slice) {
    std::string src0_temp_name = GetTempVarName(src0_shape_info.ub_name);
    std::string src1_temp_name = GetTempVarName(src1_shape_info.ub_name);
    std::string dst_temp_name = GetTempVarName(dst_shape_info.ub_name);
    CreateUbVariableND(src0_temp_name, src0_shape_info);
    CreateUbVariableND(src1_temp_name, src1_shape_info);
    CreateUbVariableND(dst_temp_name, dst_shape_info);
    this->PrintIndent();
    this->stream << ns_prefix << op_name << "(" << dst_temp_name << ", "
                 << src0_temp_name << ", " << src1_temp_name << ");\n";
  } else {
    this->PrintIndent();
    this->stream << ns_prefix << op_name << "(";
    for (int i = 0; i < var_names.size(); i++) {
      this->stream << var_names[i];
      if (i != var_names.size() - 1) {
        this->stream << ", ";
      }
    }
    this->stream << ");\n";
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

void CodeGenTileLangAscendPto::CodegenRowBroadcast(const ShapeInfo &dst,
                                                   const ShapeInfo &src) {
  std::string dst_name = dst.ub_name;
  std::string src_name = src.ub_name;

  // src: ND -> DN
  src_name = GetTempVarName(src.ub_name);
  CreateUbVariableDN(src_name, src);

  if (dst.is_slice) {
    dst_name = GetTempVarName(dst.ub_name);
    CreateUbVariableND(dst_name, dst);
  }

  this->PrintIndent();
  this->stream << "TROWEXPAND" << "(" << dst_name << ", " << src_name << ");\n";
}

void CodeGenTileLangAscendPto::CodegenColBroadcast(const ShapeInfo &dst,
                                                   const ShapeInfo &src) {
  std::string dst_name = dst.ub_name;
  std::string src_name = src.ub_name;

  if (dst.is_slice) {
    dst_name = GetTempVarName(dst.ub_name);
    CreateUbVariableND(dst_name, dst);
  }

  if (src.is_slice) {
    src_name = GetTempVarName(src.ub_name);
    CreateUbVariableND(src_name, src);
  }

  this->PrintIndent();
  this->stream << "TCOLEXPAND"
               << "(" << dst_name << ", " << src_name << ");\n";
}

void CodeGenTileLangAscendPto::BroadcastOpCodegen(const CallNode *op) {
  std::string template_args = PrintExpr(op->args[0]);

  ShapeInfo dst_shape_info = GetSliceInfo(op->args[1].as<CallNode>());
  ShapeInfo src_shape_info = GetSliceInfo(op->args[2].as<CallNode>());

  // Parse axis from template args
  std::string axis = extractBroadCastAxis(template_args);

  if (axis == "1") {
    CodegenRowBroadcast(dst_shape_info, src_shape_info);
  } else {
    CodegenColBroadcast(dst_shape_info, src_shape_info);
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
  for (int i = 0; i < (int)op->args.size() - 2; i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }

  DataType dtype0 = GetAccessPtrDtypePto(op->args[0].as<CallNode>());
  bool is_half = dtype0.is_float16();
  bool is_subs = (op_name == "TSUBS");
  bool is_divs = (op_name == "TDIVS");
  std::string operation =
      (is_subs || is_divs) ? (is_subs ? "TADDS" : "TMULS") : op_name;
  std::string index = PrintExpr(op->args[op->args.size() - 2]);

  auto apply_scalar_for_half = [&](const std::string &expr) -> std::string {
    if (is_subs) {
      return is_half ? "half(-(float)" + expr + ")" : "-" + expr;
    } else if (is_divs) {
      return is_half ? "half(1.0f / (float)" + expr + ")" : "1.0f / " + expr;
    }
    return expr;
  };

  auto buffer = op->args[2].as<CallNode>();

  if (!buffer) {
    std::string scalar = apply_scalar_for_half(index);
    this->PrintIndent();
    this->stream << operation << "(";
    for (const auto &name : var_names) {
      this->stream << name << ", ";
    }
    this->stream << scalar << ");\n";
    return;
  }

  std::string buf_offset = PrintBufferOffset(buffer);
  std::string scalar_name = GetTempVarName(buf_offset + "_scalar");

  this->PrintIndent();
  this->stream << "set_flag(PIPE_V, PIPE_S, EVENT_ID0);\n";
  this->PrintIndent();
  this->stream << "wait_flag(PIPE_V, PIPE_S, EVENT_ID0);\n";
  this->PrintIndent();
  this->stream << "auto " << scalar_name << " = " << buf_offset << ".GetValue("
               << index << ");\n";

  std::string applied_scalar = apply_scalar_for_half(scalar_name);

  std::string loop_num = getValueOrProcess(for_num_map_, index);
  if (loop_num >= "0" && loop_num <= "9") {
    ShapeInfo src_info = GetSliceInfo(op->args[1].as<CallNode>());
    ShapeInfo dst_info = GetSliceInfo(op->args[0].as<CallNode>());
    std::string src_name = GetTempVarName(src_info.ub_name);
    std::string dst_name = GetTempVarName(dst_info.ub_name);
    CreateUbVariableND(src_name, src_info);
    CreateUbVariableND(dst_name, dst_info);
    this->PrintIndent();
    this->stream << operation << "(" << dst_name << ", " << src_name << ", "
                 << applied_scalar << ");\n";
  } else {
    this->PrintIndent();
    this->stream << operation << "(";
    for (const auto &name : var_names) {
      this->stream << name << ", ";
    }
    this->stream << applied_scalar << ");\n";
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
    this->stream << op_name << "(" << var_names[0] << ", " << var_names[1]
                 << ");\n";
  }
}

void CodeGenTileLangAscendPto::ScalarOpCodegen(const CallNode *op,
                                               const std::string &op_name) {
  ShapeInfo src_shape_info = GetSliceInfo(op->args[1].as<CallNode>());
  ShapeInfo dst_shape_info = GetSliceInfo(op->args[0].as<CallNode>());

  if (src_shape_info.is_slice || dst_shape_info.is_slice) {
    auto src_temp_name = GetTempVarName(src_shape_info.ub_name);
    auto dst_temp_name = GetTempVarName(dst_shape_info.ub_name);
    CreateUbVariableND(src_temp_name, src_shape_info);
    CreateUbVariableND(dst_temp_name, dst_shape_info);
    this->PrintIndent();
    this->stream << op_name << "(" << dst_temp_name << ", " << src_temp_name << "," << PrintExpr(op->args[2]) << ");\n";
  } else {
    this->PrintIndent();
    this->stream << op_name << "("
               << dst_shape_info.ub_name << ", "
               << src_shape_info.ub_name << ", "
               << PrintExpr(op->args[2]) << ");\n";
  }
}

void CodeGenTileLangAscendPto::AxpyCodegen(const CallNode *op) {
  ShapeInfo src_shape_info = GetSliceInfo(op->args[1].as<CallNode>());
  ShapeInfo dst_shape_info = GetSliceInfo(op->args[0].as<CallNode>());
  auto scalar = PrintExpr(op->args[2]);

  DataType dtype0 = GetAccessPtrDtypePto(op->args[0].as<CallNode>());
  DataType scalar_dtype = op->args[2].dtype();
  if (scalar_dtype != dtype0) {
    if (dtype0.is_float16()) {
      scalar = "float(" + scalar + ")";
    } else {
      scalar = getType(dst_info.dtype) + "(" + scalar + ")";
    }
  }

  if (src_shape_info.is_slice || dst_shape_info.is_slice) {
    auto src_temp_name = GetTempVarName(src_shape_info.ub_name);
    auto dst_temp_name = GetTempVarName(dst_shape_info.ub_name);
    CreateUbVariableND(src_temp_name, src_shape_info);
    CreateUbVariableND(dst_temp_name, dst_shape_info);
    this->PrintIndent();
    this->stream << kAscendPtoScope << "axpy" << "<" << src_shape_info.type << ", " << src_shape_info.slice_row << 
      ", " << src_shape_info.slice_col << ">" << "(" << dst_temp_name << ", " << src_temp_name << ", " << scalar << ");\n";
  } else {
    this->PrintIndent();
    this->stream << kAscendPtoScope << "axpy" << "<" << dst_shape_info.type << ", "
                << dst_shape_info.row << ", " << dst_shape_info.col << ">"
                << "(" << dst_shape_info.ub_name << ", " << src_shape_info.ub_name << ", " << scalar
                << ");\n";
  }
}

void CodeGenTileLangAscendPto::BinaryVecClampMaxMinOpsCodegen(
    const CallNode *op, const std::string &op_name) {
  std::vector<std::string> var_names;
  std::string operation = op_name;
  for (int i = 1; i < op->args.size() - 3; i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }
  ShapeInfo src_shape_info = GetSliceInfo(op->args[2].as<CallNode>());
  ShapeInfo dst_shape_info = GetSliceInfo(op->args[1].as<CallNode>());

  if (op->args[4].as<CallNode>()) {
    this->PrintIndent();
    auto var_name = PrintBufferOffset(op->args[4].as<CallNode>());
    std::string index = PrintExpr(op->args[op->args.size() - 2]);
    std::string scalar_name = var_name + "_scalar";
    this->stream << "auto " << scalar_name << "= " << var_name << ".GetValue("
                 << index << ");\n";
    if (src_shape_info.is_slice || dst_shape_info.is_slice) {
      auto src_temp_name = GetTempVarName(src_shape_info.ub_name);
      auto dst_temp_name = GetTempVarName(dst_shape_info.ub_name);
      CreateUbVariableND(src_temp_name, src_shape_info);
      CreateUbVariableND(dst_temp_name, dst_shape_info);
      this->PrintIndent();
      this->stream << operation << "(" << dst_temp_name << ", " << src_temp_name << ", " << scalar_name << ");\n";
    } else {
      this->PrintIndent();
      this->stream << operation << "(" << dst_shape_info.ub_name << ", " << src_shape_info.ub_name << ", " << scalar_name << ");\n";
    }    
  } else {
    auto scalar = PrintExpr(op->args[op->args.size() - 2]);
    if (src_shape_info.is_slice || dst_shape_info.is_slice) {
      auto src_temp_name = GetTempVarName(src_shape_info.ub_name);
      auto dst_temp_name = GetTempVarName(dst_shape_info.ub_name);
      CreateUbVariableND(src_temp_name, src_shape_info);
      CreateUbVariableND(dst_temp_name, dst_shape_info);
      this->PrintIndent();
      this->stream << operation << "(" << dst_temp_name << ", " << src_temp_name << ", " << scalar << ");\n";
    } else {
      this->PrintIndent();
      this->stream << operation << "(" << dst_shape_info.ub_name << ", " << src_shape_info.ub_name << ", " << scalar << ");\n";
    }
  }
}

void CodeGenTileLangAscendPto::BinaryVecClampOpsCodegen(
    const CallNode *op, const std::string &op_name) {
  // Extract shape information
  ShapeInfo src_shape_info = GetSliceInfo(op->args[2].as<CallNode>());
  ShapeInfo dst_shape_info = GetSliceInfo(op->args[1].as<CallNode>());
  
  // Get scalar bounds (last two arguments)
  auto scalar_min = PrintExpr(op->args[op->args.size() - 3]);
  auto scalar_max = PrintExpr(op->args[op->args.size() - 2]);
  
  // Collect variable names (skip first arg and last 3 args: scalar_min, scalar_max, and one more)
  std::vector<std::string> var_names;
  var_names.reserve(op->args.size() - 5);  // Pre-allocate memory
  for (size_t i = 1; i < op->args.size() - 4; ++i) {
    var_names.push_back(PrintBufferOffset(op->args[i].as<CallNode>()));
  }
  
  this->PrintIndent();
  
  if (src_shape_info.is_slice || dst_shape_info.is_slice) {
    // Handle slice case with temporary variables
    std::string src_temp_name = GetTempVarName(src_shape_info.ub_name);
    std::string dst_temp_name = GetTempVarName(dst_shape_info.ub_name);
    
    CreateUbVariableND(src_temp_name, src_shape_info);
    CreateUbVariableND(dst_temp_name, dst_shape_info);
    
    this->stream << "pipe_barrier(PIPE_ALL);\n";
    this->PrintIndent();
    this->stream << "TMAXS(" << dst_temp_name << ", " << src_temp_name 
                 << ", " << scalar_min << ");\n";
    this->stream << "pipe_barrier(PIPE_ALL);\n";
    this->PrintIndent();
    this->stream << "TMINS(" << dst_temp_name << ", " << dst_temp_name 
                 << ", " << scalar_max << ");\n";
    this->stream << "pipe_barrier(PIPE_ALL);\n";
  } else {
    // Handle non-slice case - clamp_min with TMAXS
    this->stream << "TMAXS(";
    for (size_t i = 0; i < var_names.size(); ++i) {
      this->stream << var_names[i];
      if (i != var_names.size() - 1) {
        this->stream << ", ";
      }
    }
    this->stream << ", " << scalar_min << ");\n";
    
    // clamp_max with TMINS - fixed the bug where index 1 was incorrectly handled
    this->stream << "TMINS(";
    for (size_t i = 0; i < var_names.size(); ++i) {
      this->stream << var_names[i];
      if (i != var_names.size() - 1) {
        this->stream << ", ";
      }
    }
    this->stream << ", " << scalar_max << ");\n";
  }
}

void CodeGenTileLangAscendPto::SigmoidCodegen(const CallNode *op, const std::string& op_name) {
  std::vector<std::string> var_names;
  for (int i = 0; i < op->args.size() - 2; i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }
  ShapeInfo src_shape_info = GetSliceInfo(op->args[1].as<CallNode>());
  ShapeInfo dst_shape_info = GetSliceInfo(op->args[0].as<CallNode>());

  if (src_shape_info.is_slice || dst_shape_info.is_slice) {
    auto src_temp_name = GetTempVarName(src_shape_info.ub_name);
    auto dst_temp_name = GetTempVarName(dst_shape_info.ub_name);
    CreateUbVariableND(src_temp_name, src_shape_info);
    CreateUbVariableND(dst_temp_name, dst_shape_info);
    this->PrintIndent();
    this->stream << kAscendPtoScope << op_name << "<" << src_shape_info.type << ", " << src_shape_info.slice_row 
    << ", " << src_shape_info.slice_col << ">" << "(" << dst_temp_name << ", " << src_temp_name << ");\n";
  } else {
    this->PrintIndent();
    this->stream << kAscendPtoScope << op_name << "<" << dst_shape_info.type << ", " 
    << dst_shape_info.row << ", " << dst_shape_info.col << ">" << "(";
    for (int i = 0; i < var_names.size(); i++) {
      this->stream << var_names[i];
      if (i != var_names.size() - 1) {
        this->stream << ", ";
      }
    }
    this->stream << ");\n";
  }
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

CodeGenTileLangAscendPto::ReduceOpInfo
CodeGenTileLangAscendPto::ParseReduceOpInfo(const std::string &op_name) {
  auto [slice_row, slice_col, direction_val, success] =
      ExtractTemplateParamsForSliceBuffer(op_name);

  ICHECK(success) << "ExtractTemplateParams failed";
  ICHECK(direction_val == -1 || direction_val == 0)
      << "Only row-wise (-1) or column-wise (0) reduce supported";

  ReduceOpInfo info;
  info.buffer_slice_row = slice_row;
  info.buffer_slice_col = slice_col;
  info.direction =
      (direction_val == -1) ? ReduceDirection::ROW : ReduceDirection::COL;

  if (op_name.find("reduce_sum") != std::string::npos) {
    info.kind = ReduceKind::SUM;
  } else if (op_name.find("reduce_max") != std::string::npos) {
    info.kind = ReduceKind::MAX;
  } else if (op_name.find("reduce_min") != std::string::npos) {
    info.kind = ReduceKind::MIN;
  } else {
    ICHECK(false) << "Unsupported reduce type: " << op_name;
  }

  return info;
}

std::string
CodeGenTileLangAscendPto::GetReduceOpName(ReduceKind kind,
                                          ReduceDirection direction) {
  static const std::unordered_map<
      ReduceKind, std::unordered_map<ReduceDirection, std::string>>
      kOpNames = {
          {ReduceKind::SUM,
           {{ReduceDirection::ROW, "TROWSUM"},
            {ReduceDirection::COL, "TCOLSUM"}}},
          {ReduceKind::MAX,
           {{ReduceDirection::ROW, "TROWMAX"},
            {ReduceDirection::COL, "TCOLMAX"}}},
          {ReduceKind::MIN,
           {{ReduceDirection::ROW, "TROWMIN"},
            {ReduceDirection::COL, "TCOLMIN"}}},
      };
  return kOpNames.at(kind).at(direction);
}

void CodeGenTileLangAscendPto::CodegenRowReduce(const ReduceOpInfo &op_info,
                                                const ShapeInfo &dst,
                                                const ShapeInfo &src,
                                                const ShapeInfo &tmp) {
  std::string op_name = GetReduceOpName(op_info.kind, ReduceDirection::ROW);
  std::string dst_name = dst.ub_name;
  std::string src_name = src.ub_name;

  // dst: ND -> DN
  dst_name = GetTempVarName(dst.ub_name);
  CreateUbVariableDN(dst_name, dst);

  if (src.is_slice) {
    src_name = GetTempVarName(src.ub_name);
    CreateUbVariableND(src_name, src);
  }

  this->PrintIndent();
  this->stream << op_name << "(" << dst_name << ", " << src_name << ", "
               << tmp.ub_name << ");\n";
}

void CodeGenTileLangAscendPto::CodegenColReduce(const ReduceOpInfo &op_info,
                                                const ShapeInfo &dst,
                                                const ShapeInfo &src,
                                                const ShapeInfo &tmp) {

  std::string op_name = GetReduceOpName(op_info.kind, ReduceDirection::COL);

  std::string dst_name = dst.ub_name;
  std::string src_name = src.ub_name;

  if (dst.is_slice) {
    dst_name = GetTempVarName(dst.ub_name);
    CreateUbVariableND(dst_name, dst);
  }

  if (src.is_slice) {
    src_name = GetTempVarName(src.ub_name);
    CreateUbVariableND(src_name, src);
  }

  // TCOLSUM: src.dtyp == dst.dtyp == tmp.dtype
  std::string temp_name = tmp.ub_name;
  if (op_info.kind == ReduceKind::SUM) {
    ICHECK(dst.type == src.type)
        << "Reduce_sum input dtype must be consistent with the output "
           "dtype.";
    if (dst.type != tmp.type) {
      temp_name = GetTempVarName(temp_name);

      int tmp_col =
          tmp.row * tmp.col * GetTypeLen(tmp.type) / GetTypeLen(dst.type);
      tmp_col = GetValidShape(tmp_col, dst.type);
      ShapeInfo tmp_cast =
          ShapeInfo{1,          tmp_col,  1,           tmp_col,
                    1,          tmp_col,  tmp.extent,  tmp.first_addr,
                    tmp.offset, dst.type, tmp.ub_name, false};
      CreateUbVariableND(temp_name, tmp_cast);
    }
  }

  this->PrintIndent();
  this->stream << op_name << "(" << dst_name << ", " << src_name;
  // TCOLSUM needs tmp
  if (op_info.kind == ReduceKind::SUM) {
    this->stream << ", " << temp_name << ", false";
  }

  this->stream << ");\n";
}

void CodeGenTileLangAscendPto::ReduceOpCodegen(const CallNode *op) {
  std::string op_name_str = Downcast<StringImm>(op->args[0])->value;

  ReduceOpInfo op_info = ParseReduceOpInfo(op_name_str);
  ShapeInfo dst = GetSliceInfo(op->args[1].as<CallNode>());
  ShapeInfo src = GetSliceInfo(op->args[2].as<CallNode>());
  bool is_slice = src.slice_valid_row != op_info.buffer_slice_row ||
                  src.slice_valid_col != op_info.buffer_slice_col;
  // reduce offer real_shape
  if (is_slice) {
    src.slice_valid_row = op_info.buffer_slice_row;
    src.slice_valid_col = op_info.buffer_slice_col;
    src.is_slice = true;
    src.offset = "0";
  }

  ShapeInfo tmp = GetSliceInfo(op->args[3].as<CallNode>());

  if (op_info.direction == ReduceDirection::ROW) {
    if (is_slice) {
      dst.slice_valid_col = op_info.buffer_slice_row;
    }
    CodegenRowReduce(op_info, dst, src, tmp);
  } else {
    if (is_slice) {
      dst.slice_valid_col = op_info.buffer_slice_col;
    }
    CodegenColReduce(op_info, dst, src, tmp);
  }
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

const std::unordered_map<std::string, std::string> scope_to_tile = {
    {"wmma.matrix_a", kAscendPtoScope + "TileMatL0A"},
    {"wmma.matrix_b", kAscendPtoScope + "TileMatL0B"},
    {"wmma.accumulator", "TileAcc"},
    {"shared.dyn", kAscendPtoScope + "TileMatL1"},
    {"shared", kAscendPtoScope + "TileUbDataND"},
};

void CodeGenTileLangAscendPto::AllocateLocalVar(const AllocateNode *op,
                                                std::string &vid,
                                                std::string dtype) {
  PrimExpr init = tir::make_const(op->dtype, 0);

  auto init_it = op->annotations.find(tl::attr::kLocalVarInit);
  if (init_it != op->annotations.end()) {
    PrimExpr user_init = Downcast<PrimExpr>((*init_it).second);
    if (user_init.dtype().is_bool()) {
      dtype = "bool";
    } else if (!user_init.dtype().is_void() && user_init.dtype() != op->dtype) {
      user_init = tir::Cast(op->dtype, user_init);
      dtype = getType(user_init.dtype());
    }
    init = user_init;
  }
  this->PrintIndent();
  stream << dtype + " " << vid << " = " << PrintExpr(init) << ";\n";
}

void CodeGenTileLangAscendPto::VisitStmt_(const AllocateNode *op) {
  ICHECK(!is_zero(op->condition)) << "Allocation condition must not be zero.";

  // 1. Extract basic allocation info
  std::string vid = AllocVarID(op->buffer_var.get()); // var_name
  std::string type = getType(op->dtype);
  std::string scope = GetPtrStorageScope(op->buffer_var);

  // 2. Determine the corresponding PTO Tile class name
  // handle T.var
  if (scope == "local.var") {
    AllocateLocalVar(op, vid, type);

    this->PrintStmt(op->body);
    return;
  }

  ICHECK(scope_to_tile.count(scope))
      << "Unsupported storage scope for PTO allocation: " << scope
      << ", variable: " << op->buffer_var->name_hint;
  const std::string &op_name = scope_to_tile.at(scope);

  // 3. Retrieve and validate the 4D physical layout [M, N, Valid_M, Valid_N]
  ICHECK(buffer_shapess_.count(op->buffer_var))
      << "Buffer shape not found for variable: " << op->buffer_var->name_hint;
  const auto &shape = buffer_shapess_.at(op->buffer_var);

  ICHECK(shape.size() == 4)
      << "Expected a 4D shape [M, N, Valid_M, Valid_N] for PTO, but got "
      << shape.size() << "D for " << op->buffer_var->name_hint;
  const auto &M = shape[0];
  const auto &N = shape[1];
  const auto &valid_M = shape[2];
  const auto &valid_N = shape[3];

  // Print the Tile object declaration
  this->PrintIndent();
  stream << op_name << "<" << type << ", " << M << ", " << N << ", " << valid_M
         << ", " << valid_N << "> " << vid << ";\n";

  // address_map, use name_hint as key
  Map<String, PrimExpr> address_map_name_hint;
  for (const auto &[var, address] : address_map_) {
    address_map_name_hint.Set(var->name_hint, address);
  }

  // 4. Resolve the target physical memory address
  PrimExpr target_address;
  if (address_map_name_hint.count(op->buffer_var->name_hint)) {
    target_address = address_map_name_hint.at(op->buffer_var->name_hint);
  } else {
    PrimExpr current_offset =
        address_offset_.Get(String(scope)).value_or(Integer(0));
    target_address = current_offset;

    int64_t alloc_bytes = op->ConstantAllocationSize() * op->dtype.bytes();
    address_offset_.Set(String(scope), current_offset + Integer(alloc_bytes));
  }
  buffer_address_map_.Set(op->buffer_var, target_address);

  // Print the address assignment (TASSIGN)
  this->PrintIndent();
  stream << "TASSIGN(" << vid << ", " << PrintExpr(target_address) << ");\n";

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
  buffer_shapess_ =
      f->GetAttr<Map<Var, Array<PrimExpr>>>(tvm::tl::kLogicBufferShapes)
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

  std::string src0_type = src0_shape_info.type;

  int src1_type = std::stoi(PrintExpr(op->args[3]));
  if (src1_type == 2) {
    src1_name = PrintBufferOffset(op->args[4].as<CallNode>());
    op_name = "TSEL";
  } else if (src1_type == 1) {
    src1_name = PrintExpr(op->args[4]);
    op_name = "TSELS";
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

void CodeGenTileLangAscendPto::MmaCodegen(const CallNode *op) {
  auto k = PrintExpr(op->args[5]);

  // mma<..., M, N> -> mma<..., M, N, K>
  std::string s = Downcast<StringImm>(op->args[0])->value;
  auto pos = s.rfind('>');
  if (pos != std::string::npos) {
    s.insert(pos, ", " + k);
  }

  std::string op_name = kAscendPtoScope + s;

  auto a_var = op->args[1].as<CallNode>()->args[1].as<VarNode>();
  auto b_var = op->args[2].as<CallNode>()->args[1].as<VarNode>();
  auto c_var = op->args[3].as<CallNode>()->args[1].as<VarNode>();

  auto a_name = var_idmap_[a_var];
  auto b_name = var_idmap_[b_var];
  auto c_name = var_idmap_[c_var];

  this->PrintIndent();
  this->stream << op_name << "(" << a_name << ", " << b_name << ", " << c_name
               << ", " << PrintExpr(op->args[4]) << ");\n";
}

} // namespace codegen
} // namespace tvm
