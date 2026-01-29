// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file target/codegen_ascend_pto.cc
 */

#include "codegen_ascend_pto.h"
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>

#include <cmath>
#include <string>
#include <utility>
#include <vector>
#include <sstream>
#include <iomanip>

#include "../op/ascend.h"
#include "../op/builtin.h"

#include "arith/pattern_match.h"

#define DEC_STR_TO_HEX_STR(dec_str) \
  ([](const std::string& s){ std::stringstream ss; \
  ss << std::showbase << std::hex << std::uppercase << std::stoi(s); \
  return ss.str(); }(dec_str))

namespace tvm {
namespace codegen {
  const std::string kAscendPtoScope = "tl::ascend_pto::";

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

int8_t GetTypeLen(std::string type) {
  int8_t typeSize = 1;
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

CodeGenTileLangAscendPto::CodeGenTileLangAscendPto(std::string platform) {
  // restrict_keyword_ = "__gm__ uint8_t *";
  platform_ = platform;
}

void CodeGenTileLangAscendPto::PrintFuncPrefix(std::ostream &os) {
  //os << "extern \"C\" CATLASS_GLOBAL\n";
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
        os << "half_t";
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

void CodeGenTileLangAscendPto::PrintStorageScope(const std::string &scope,
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
    os << "*(" << var_name << "_handle + " << PrintExpr(op->indices.back()) << ")";
  } else {
    os << var_name << ".GetValue("
                << PrintExpr(op->indices.back()) << ")";
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
  } else {
    this->stream << var_name << ".SetValue(" << PrintExpr(op->indices.back())
                 << ", " << PrintExpr(op->value) << ");\n";
  }
}

std::map<std::string, std::string> extractTemplateParams(const std::string& input) {
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
        "data_type_input",
        "data_type_output",
        "M",
        "N",
        "K",
        "transpose_A",
        "transpose_B"
    };
    for (size_t i = 0; i < params.size() && i < paramNames.size(); ++i) {
        result[paramNames[i]] = params[i];
    }
    for (size_t i = paramNames.size(); i < params.size(); ++i) {
        result["extra_param_" + std::to_string(i - paramNames.size() + 1)] = params[i];
    }
    return result;
}

std::vector<std::string> extractShapeFromTemplate(const std::string& input) {
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

int GetValidShape(int shape, std::string& dtype) {
  int dtype_len = GetTypeLen(dtype);
  int shape_mod = shape  * GetTypeLen(dtype) % 32;
  if (shape_mod == 0) {
      return shape;
  }
  return shape + (32 - shape_mod) / dtype_len;
}

void CodeGenTileLangAscendPto::VisitExpr_(const CallNode *op, std::ostream &os) {
  if (op->op.same_as(builtin::call_extern())) {
    CallExternCodegen(op);
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
    ICHECK(false) << "axpy not support in pto. use muls and add instead";
    ScalarOpCodegen(op, "T");
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
    BinaryVecOpCodegen(op, "TAND");
  } else if (op->op.same_as(tl::ascend_bitwise_or())) {
    BinaryVecOpCodegen(op, "TOR");
  } else if (op->op.same_as(tl::ascend_adds())) {
    BinaryVecOpsCodegen(op, "TADDS");
  } else if (op->op.same_as(tl::ascend_subs())) {
    BinaryVecOpsCodegen(op, "TSUBS");
  } else if (op->op.same_as(tl::ascend_muls())) {
    BinaryVecOpsCodegen(op, "TMULS");
  } else if (op->op.same_as(tl::ascend_divs())) {
    BinaryVecOpsCodegen(op, "TDIVS");
  }
  // else if (op->op.same_as(tl::ascend_maxs())) {
  //   BinaryVecOpsCodegen(op, "TMAXS");
  // } else if (op->op.same_as(tl::ascend_mins())) {
  //   BinaryVecOpsCodegen(op, "TMINS");
  // }
    else if (op->op.same_as(tl::ascend_pipe_barrier())) {
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
  }
}

std::string CodeGenTileLangAscendPto::PrintBufferOffset(const CallNode *op) {
    auto _var = op->args[1].as<VarNode>();
    std::string _var_name = var_idmap_[_var];
    return _var_name;
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

    auto src_type = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;
    auto dst_type = op->args[2].as<CallNode>()->args[0].as<CallNode>()->dtype;

    static const std::unordered_map<std::string, int> kCopyOpExtraArgs = {
      {"copy_l0c_to_gm", 1},
      {"copy_gm_to_l1", 1},
      {"copy_l1_to_l0a", 3},
      {"copy_l1_to_l0b", 3},
      {"copy_gm_to_ub", 1},
      {"copy_ub_to_gm", 1},
      {"copy_ub_to_ub", 0}
    };

    std::unordered_map<std::string, std::string>
      ptoCopyMap = {
      {"copy_l0c_to_gm", "TSTORE"},
      {"copy_gm_to_l1", "TLOAD"},
      {"copy_l1_to_l0a", "TEXTRACT"},
      {"copy_l1_to_l0b", "TEXTRACT"},
      {"copy_gm_to_ub", "TLOAD"},
      {"copy_ub_to_gm", "TSTORE"},
      {"copy_ub_to_ub", "TCVT"}
    };

    bool found = false;
    int extra_args = 0;
    std::string real_name = "";

    for (const auto& pair : kCopyOpExtraArgs) {
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
        api_name = src_type == dst_type ? "TMOV":"TCVT";
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
        this->stream << api_name << "(" << dst_var_id << ", "
        << src_var_id << ", " << row_index << ", " << col_index << ");\n";
      } else if (api_name == "TCVT") {
        this->PrintIndent();
        this->stream << api_name << "(" << dst_var_id << ", "
        << src_var_id << ", pto::RoundMode::CAST_NONE" << ");\n";
      } else if (api_name == "TMOV") {
        this->PrintIndent();
        std::vector src_ub_data = ub_data_map_[src_var_id];
        std::vector dst_ub_data = ub_data_map_[dst_var_id];
        int32_t len = GetTypeLen(src_ub_data[0]);
        std::vector shapes = extractShapeFromTemplate(op_name);
        if (shapes.size() == 1 && (src_offset != dst_offset || src_shape != dst_shape || src_offset != "0" || dst_offset != "0")) {
          this->stream << kAscendPtoScope << "mov_tile<" << src_ub_data[0] << ", " << std::stoi(shapes[0]) << ">("
          << src_ub_data[3] << ", " << dst_ub_data[3]
          << ", " << src_offset << ", " << dst_offset << ", " << len << ");\n";
        } else {
          this->stream << api_name << "(" << dst_var_id << ", " << src_var_id << ");\n";
        }
      }
      if (api_name == "TLOAD") {
        ICHECK((copy_base_addr_map_.find(String(src_var_id)) != copy_base_addr_map_.end()));
        std::vector<std::string> l_valid_shapes = l_data_map_[dst_var_id];
        std::vector<std::string> ub_valid_shapes = ub_data_map_[dst_var_id];
        std::vector<std::string> dynamic_names;
        std::string tensor_addr = copy_base_addr_map_[String(src_var_id)];
        std::string tensor_template = "<" + global_tensor_template[String(tensor_addr)].dtype;
        std::string shape_template = "", stride_template = "", valid_template = "";
        size_t len = global_tensor_template[String(tensor_addr)].shape_list.size();
        size_t shape_len = 2;
        size_t op_arg_len = op->args.size();
        size_t shape_size = 5;
        // Dynamic Shape and Static Shape

        // generate shape
        std::vector<std::string> shape_nums(shape_len);
        shape_nums[1] =  PrintExpr(op->args[op_arg_len - 1]);
        if (op_arg_len == 5) {
            shape_nums[0] =  "1";
        } else {
            shape_nums[0] =  PrintExpr(op->args[op_arg_len - 2]);
        }
        for (size_t i = 0; i < shape_size; i++) {
            if (i < shape_size - shape_len) {
              shape_template += "1";
            } else {
              shape_template += shape_nums[i + shape_len - shape_size];
            }
            if (i < shape_size - 1) {
              shape_template += ", ";
            }
        }
        // shape_template += ">";
        // generate stride
        for (size_t i = 0; i < 4; i++) {
          if (len > 3 - i) {
            std::string tensor_template = global_tensor_template[String(tensor_addr)].shape_list[len + i - 4];
            if (tensor_template[0] < '1' || tensor_template[0] > '9')  {
              stride_template += "-1, ";
              dynamic_names.push_back(tensor_template);
            }
            else {
              std::string tmp_shape = "";
              for(size_t j = 0; j < 4 - i; j++) {
                tmp_shape += global_tensor_template[String(tensor_addr)].shape_list[len - j - 1];
                if (j < 3 - i) tmp_shape += " * ";
              }
              stride_template = stride_template +  tmp_shape + ", ";
            }
          } else {
            stride_template += "1, ";
          }
        }
        stride_template += "1";
        // get gm2l1 shape

        bool is_dynamic = global_tensor_template[String(tensor_addr)].shape_type=="dynamic";
        std::string src_var = "";
        if(op_name.find("copy_gm_to_l1") != std::string::npos) {
          src_var = "copy_gm_to_l1";
          if (is_dynamic) {
            src_var = src_var + "_dynamic";
          }
          tensor_template = tensor_template + ", " + l_valid_shapes[0] + ", ";
          valid_template = l_valid_shapes[1] + ", " + l_valid_shapes[2];
        } else if(op_name.find("copy_gm_to_ub") != std::string::npos) {
          src_var = "copy_gm_to_ub";
          if (is_dynamic) {
            src_var = src_var + "_dynamic";
          }
          tensor_template = tensor_template + ", " + ub_valid_shapes[0] + ", ";
          valid_template = ub_valid_shapes[1] + ", " + ub_valid_shapes[2];
        }
        tensor_template = tensor_template + shape_template + ", " + stride_template + ", " + valid_template + ">";
        this->PrintIndent();
        this->stream << kAscendPtoScope << src_var << tensor_template << "(" << tensor_addr << " + " << src_offset;
        if(is_dynamic) {
          std::string shape = "pto::Shape<"  + shape_template + ">()";
          this->stream << ", " << "pto::Shape<"  << shape_template << ">" << "(), " << "pto::Stride<" << stride_template << ">" << "(";
          for(size_t i = 0; i < dynamic_names.size(); i++) {
            this->stream << dynamic_names[i];
            if (i != dynamic_names.size() - 1) {
              this->stream << ", ";
            }
          }
          this->stream << ")";
        }
        if (api_name == "TLOAD" && prefetch_n_stages_map_.count(dst_var_id) && prefetch_n_stages_map_[dst_var_id].first > 0) {

		  auto buffer_k = op->args[2].as<CallNode>()->args[2] / op->args[2].as<CallNode>()->args[3];
          tvm::arith::Analyzer analyzer;
          PrimExpr simplified_k = analyzer.Simplify(buffer_k);
		  this->stream << ", " << dst_var_id << "[" << simplified_k << "]" << ");\n";

          prefetch_n_stages_map_[dst_var_id].second++;
        } else {
          this->stream << ", " << dst_var_id << ");\n";
        }

      } else if (api_name == "TSTORE") {
        ICHECK((copy_base_addr_map_.find(String(dst_var_id)) != copy_base_addr_map_.end()));
        std::vector<std::string> l_valid_shapes = l_data_map_[src_var_id];
        std::vector<std::string> ub_valid_shapes = ub_data_map_[src_var_id];
        std::vector<std::string> dynamic_names;
        std::string tensor_addr = copy_base_addr_map_[String(dst_var_id)];
        std::string tensor_template = "<" + global_tensor_template[String(tensor_addr)].dtype;
        std::string shape_template = "", stride_template = "", valid_template = "";
        size_t len = global_tensor_template[String(tensor_addr)].shape_list.size();
        size_t shape_len = 2;
        size_t op_arg_len = op->args.size();
        size_t shape_size = 5;
        // Dynamic Shape and Static Shape

        // generate shape
        std::vector shape_tile = extractShapeFromTemplate(op_name);
        std::vector<std::string> shape_nums(shape_len);
        bool is_chunking = false;

        if (shape_tile[0] != PrintExpr(op->args[op_arg_len - 1]) && op_name.find("copy_ub_to_gm") != std::string::npos) {
          ub_valid_shapes[2] = shape_tile[0];
          is_chunking = true;
        }
        shape_nums[1] =  PrintExpr(op->args[op_arg_len - 1]);
        if (op_arg_len == 5) {
          shape_nums[0] =  "1";
        } else if(shape_tile[1] != PrintExpr(op->args[op_arg_len - 2]) && op_name.find("copy_ub_to_gm") != std::string::npos) { //
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
            std::string tensor_template = global_tensor_template[String(tensor_addr)].shape_list[len + i - 4];
            if (tensor_template[0] < '1' || tensor_template[0] > '9')  {
              stride_template += "-1, ";
              dynamic_names.push_back(tensor_template);
            }
            else {
              std::string tmp_shape = "";
              for(size_t j = 0; j < 4 - i; j++) {
                tmp_shape += global_tensor_template[String(tensor_addr)].shape_list[len - j - 1];
                if (j < 3 - i) tmp_shape += " * ";
              }
              stride_template = stride_template +  tmp_shape + ", ";
            }
          } else {
            stride_template += "1, ";
          }
        }
        stride_template += "1";

        // get gm2l1 shape
        bool is_dynamic = global_tensor_template[String(tensor_addr)].shape_type=="dynamic";
        std::string src_var = "";
        if(op_name.find("copy_l0c_to_gm") != std::string::npos) {
          src_var = "copy_l0c_to_gm";
          if (is_dynamic) {
            src_var = src_var + "_dynamic";
          }
          tensor_template = tensor_template + ", " + l_valid_shapes[0] + ", ";
          valid_template = l_valid_shapes[1] + ", " + l_valid_shapes[2];
        } else if(op_name.find("copy_ub_to_gm") != std::string::npos) {
          src_var = "copy_ub_to_gm";
          if (is_dynamic) {
            src_var = src_var + "_dynamic";
          }
          tensor_template = tensor_template + ", " + ub_valid_shapes[0] + ", ";
          valid_template = "";
        }
        // tensor_template = tensor_template + shape_template + ", " + stride_template + ", " + valid_template;
        if (op_name.find("copy_ub_to_gm") != std::string::npos) {
          tensor_template = tensor_template + shape_template + ", " + stride_template  + ", " +
          shape_nums[0] + ", " + shape_nums[1] + ", " + ub_valid_shapes[1] + ", " + ub_valid_shapes[2] + ">";
        } else {
          tensor_template = tensor_template + shape_template + ", " + stride_template + ", " + valid_template + ">";
        }
        this->PrintIndent();
        this->stream << kAscendPtoScope << src_var << tensor_template << "(" << tensor_addr << " + " << dst_offset;
        if(is_dynamic) {
          this->stream << ", " << "pto::Shape<"  << shape_template << ">" << "(), " << "pto::Stride<" << stride_template << ">" << "(";
          for(size_t i = 0; i < dynamic_names.size(); i++) {
             this->stream << dynamic_names[i];
            if (i != dynamic_names.size() - 1) {
              this->stream << ", ";
            }
          }
          this->stream << ")";
        }
        if (op_name.find("copy_ub_to_gm") != std::string::npos) {
          int32_t type_len = GetTypeLen(global_tensor_template[String(tensor_addr)].dtype);
          this->stream << ", " << ub_valid_shapes[3] << ", " << src_offset << "," << type_len << ");\n";
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

  if (prefetch_n_stages_map_[a_name].first > 0) {
    auto a_k = op->args[1].as<CallNode>()->args[2] / op->args[1].as<CallNode>()->args[3];
    tvm::arith::Analyzer analyzer;
    PrimExpr simplified_a_k = analyzer.Simplify(a_k);
    auto b_k = op->args[2].as<CallNode>()->args[2] / op->args[2].as<CallNode>()->args[3];
    PrimExpr simplified_b_k = analyzer.Simplify(b_k);

    std::map<std::string, std::string> params = extractTemplateParams(op_name);
    std::string data_type_input = params["data_type_input"];
    this->stream << kAscendPtoScope << "gemm_v0" << "<" <<  params["data_type_input"] << ", " << params["data_type_output"] << ", "
    << GetValidShape(std::stoi(params["M"]), data_type_input) << ", "
    << GetValidShape(std::stoi(params["N"]), data_type_input) << ", "
    << GetValidShape(std::stoi(params["K"]), data_type_input) << ", "
    << params["M"] << ", " << params["N"] << ", " << params["K"] << ", "
    << params["transpose_A"] << ", " << params["transpose_B"] << ">"
    << "(" << a_name << "[" << simplified_a_k << "], " << b_name << "[" << simplified_a_k << "], " << c_name << ", " << PrintExpr(op->args[4]) << ");\n";
  } else {
    std::map<std::string, std::string> params = extractTemplateParams(op_name);
    std::string data_type_input = params["data_type_input"];
    this->stream << kAscendPtoScope << "gemm_v0" << "<" <<  params["data_type_input"] << ", " << params["data_type_output"] << ", "
    << GetValidShape(std::stoi(params["M"]), data_type_input) << ", "
    << GetValidShape(std::stoi(params["N"]), data_type_input) << ", "
    << GetValidShape(std::stoi(params["K"]), data_type_input) << ", "
    << params["M"] << ", " << params["N"] << ", " << params["K"] << ", "
    << params["transpose_A"] << ", " << params["transpose_B"] << ">"
    << "(" << a_name << ", " << b_name << ", " << c_name << ", " << PrintExpr(op->args[4]) << ");\n";
  }

}

void CodeGenTileLangAscendPto::PipeBarrierCodegen(const CallNode *op) {
  std::string pipe = Downcast<StringImm>(op->args[0])->value;
  this->PrintIndent();
  this->stream << "pipe_barrier(PIPE_" << pipe << ");\n";
}

void CodeGenTileLangAscendPto::SetAndWaitFlagCodegen(const CallNode *op, const std::string &op_name) {
  std::string src = Downcast<StringImm>(op->args[0])->value;
  std::string dst = Downcast<StringImm>(op->args[1])->value;
  std::string event_id = PrintExpr(op->args[2]);
  this->PrintIndent();
  this->stream << op_name << "(PIPE_" << src << ", " << "PIPE_" <<
  dst << ", " << "EVENT_ID" << event_id << ");\n";
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
  int flag = std::stoi(PrintExpr(op->args[1]));

  if (this->platform_ == "A5") {
    HandleA5Flag("set_intra_block", pipe, flag);
  } else {
    int mode = 2;
    int config = 1 | (mode << 4) | (flag << 8);
    this->PrintIndent();
    this->stream << "ffts_cross_core_sync" << "(" << "PIPE_" << pipe << ", "
                 << config << ");\n";
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
  auto flag = op->args[0].as<IntImmNode>()->value;
  auto pipe = op->args[1].as<StringImmNode>()->value;

  if (this->platform_ == "A5") {
    if (pipe.empty()) {
      if (this->current_resource_scope_ == "CUBE") {
        pipe = "MTE1";
      } else if (this->current_resource_scope_ == "VEC") {
        pipe = "V";
      } else {
        LOG(WARNING) << "Cannot infer default pipe for wait_intra_block in unknown scope";
      }
    }
  } else {
    if (!pipe.empty()) {
      LOG(FATAL) << "Pipe argument for wait_cross_flag is only supported on A5 architecture.";
    }
  }

  if (this->platform_ == "A5") {
    HandleA5Flag("wait_intra_block", pipe, flag);
  } else {
    this->PrintIndent();
    this->stream << "wait_flag_dev" << "(" << flag << ");\n";
  }
}

void CodeGenTileLangAscendPto::FillCodegen(const CallNode *op) {
  this->PrintIndent();
  this->stream << "TEXPANDS" << "(" << PrintBufferOffset(op->args[1].as<CallNode>()) << ", "
              << PrintExpr(op->args[2]) << ");\n";
}

void CodeGenTileLangAscendPto::CreateVecIndexCodegen(const CallNode *op,
                                                  const std::string &op_name) {
  this->PrintIndent();
  std::string dst_name = PrintExpr(op->args[0].as<CallNode>()->args[1]);
  std::string dst_offset = PrintExpr(op->args[0].as<CallNode>()->args[2]);
  std::string first_value = PrintExpr(op->args[1]);
  std::vector<std::string> ub_data = ub_data_map_[dst_name];
  int32_t len = GetTypeLen(ub_data[0]);
  this->stream << kAscendPtoScope << "tci" << "<" << ub_data[0] << ", " << ub_data[1] << ", " 
  << ub_data[2] << ">" << "(" << ub_data[3] << ", " << dst_offset << ", " << len << ", " << first_value << ");\n";
}

void CodeGenTileLangAscendPto::GatherbCodegen(const CallNode *op,
                                                  const std::string &op_name) {
  this->PrintIndent();
  std::string dst_name = PrintExpr(op->args[1].as<CallNode>()->args[1]);
  std::string src_name = PrintExpr(op->args[2].as<CallNode>()->args[1]);
  std::string idx_name = PrintExpr(op->args[3].as<CallNode>()->args[1]);
  this->stream << op_name << "(" << dst_name << ", " << src_name << ", " << idx_name << ");\n";
}

void CodeGenTileLangAscendPto::PowCodegen(const CallNode *op) {
  this->PrintIndent();
  std::string dst_name = PrintExpr(op->args[0].as<CallNode>()->args[1]);
  std::string src0_name = PrintExpr(op->args[1].as<CallNode>()->args[1]);
  std::string src1_name = PrintExpr(op->args[2].as<CallNode>()->args[1]);
  std::vector<std::string> ub_data = ub_data_map_[dst_name];
  this->stream << kAscendPtoScope << "pow" << "<" << ub_data[0] << ", " << ub_data[1] << ", " << ub_data[2] << ">"
   << "(" << dst_name << ", " << src0_name << ", " << src1_name << ");\n";
}

void CodeGenTileLangAscendPto::Sort32Codegen(const CallNode *op, const std::string &op_name) {
  this->PrintIndent();
  std::string dst_name = PrintExpr(op->args[0].as<CallNode>()->args[1]);
  std::string src_name = PrintExpr(op->args[1].as<CallNode>()->args[1]);
  std::string idx_name = PrintExpr(op->args[2].as<CallNode>()->args[1]);
  this->stream << op_name << "(" << dst_name << ", " << src_name << ", " << idx_name << ");\n";
}

void CodeGenTileLangAscendPto::TransposeCodegen(const CallNode *op, const std::string &op_name) {
  this->PrintIndent();
  std::string dst_name = PrintExpr(op->args[0].as<CallNode>()->args[1]);
  std::string src_name = PrintExpr(op->args[1].as<CallNode>()->args[1]);
  this->stream << op_name << "(" << dst_name << ", " << src_name << ", " << src_name << ");\n";
}

void CodeGenTileLangAscendPto::XorCodegen(const CallNode *op, const std::string &op_name) {
  this->PrintIndent();
  std::string dst_name = PrintExpr(op->args[0].as<CallNode>()->args[1]);
  std::string src0_name = PrintExpr(op->args[1].as<CallNode>()->args[1]);
  std::string src1_name = PrintExpr(op->args[2].as<CallNode>()->args[1]);
  this->stream << op_name << "(" << dst_name << ", " << src0_name << ", " << src1_name << ", " << src1_name << ");\n";
}

void CodeGenTileLangAscendPto::CompareCodegen(const CallNode *op, const std::string &op_name) {
  this->PrintIndent();
  std::string dst_name = PrintExpr(op->args[0].as<CallNode>()->args[1]);
  std::string src0_name = PrintExpr(op->args[1].as<CallNode>()->args[1]);
  std::string src1_name = PrintExpr(op->args[2].as<CallNode>()->args[1]);
  std::string mode = Downcast<StringImm>(op->args[3])->value;
  this->stream << op_name << "(" << dst_name << ", " << src0_name << ", " << 
  src1_name << ", " << "CmpMode::" << mode << ");\n";
}

void CodeGenTileLangAscendPto::CompareScalarCodegen(const CallNode *op, const std::string &op_name) {
  this->PrintIndent();
  std::string dst_name = PrintExpr(op->args[0].as<CallNode>()->args[1]);
  std::string src0_name = PrintExpr(op->args[1].as<CallNode>()->args[1]);
  std::string src1_name = PrintExpr(op->args[2]);
  std::string mode = Downcast<StringImm>(op->args[3])->value;
  this->stream << op_name << "(" << dst_name << ", " << src0_name << ", " << 
  src1_name << ", " << "CmpMode::" << mode << ");\n";
}

void CodeGenTileLangAscendPto::BinaryVecOpCodegen(const CallNode *op,
                                               const std::string &op_name) {
  std::vector<std::string> var_names;
  std::string src0_name = PrintExpr(op->args[1].as<CallNode>()->args[1]);
  std::string src1_name = PrintExpr(op->args[2].as<CallNode>()->args[1]);
  std::string dst_name = PrintExpr(op->args[0].as<CallNode>()->args[1]);

  std::string src0_offset = PrintExpr(op->args[1].as<CallNode>()->args[2]);
  std::string src1_offset = PrintExpr(op->args[2].as<CallNode>()->args[2]);
  std::string dst_offset = PrintExpr(op->args[0].as<CallNode>()->args[2]);

  std::string src0_addr = ub_data_map_[src0_name][3];
  std::string src1_addr = ub_data_map_[src1_name][3];
  std::string dst_addr = ub_data_map_[dst_name][3];

  std::string shape = PrintExpr(op->args[3]);

  std::string ub_type = ub_data_map_[dst_name][0];
  int32_t type_len = GetTypeLen(ub_type);
  for (int i = 0; i < op->args.size() - 1; i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }
  if (prefetch_n_stages_map_[var_names[1]].first > 0) {
    auto buffer_k = op->args[1].as<CallNode>()->args[2] / op->args[1].as<CallNode>()->args[3];
    tvm::arith::Analyzer analyzer;
    PrimExpr simplified_buffer_k = analyzer.Simplify(buffer_k);
    this->PrintIndent();
    this->stream << op_name << "(";
    for (int i = 0; i < var_names.size(); i++) {
      this->stream << var_names[i];
      if (i != 0) {
        this->stream << "[" << simplified_buffer_k << "]";
      }
      if (i != var_names.size() - 1) {
        this->stream << ", ";
      }
    }
    this->stream << ");\n";
  } else if (src0_offset != "0" || src1_offset != "0" || dst_offset != "0") {
    this->PrintIndent();
    this->stream << kAscendPtoScope << "binary_tile" << "<" << kAscendPtoScope << "BinaryOp::" << op_name << ", " << ub_type << ", " << shape << ">" << "("
    << dst_addr << ", " << src0_addr << ", " << src1_addr << ", " << dst_offset
    << ", " << src0_offset << ", " << src1_offset << ", " << type_len << ");\n";
  } else if (prefetch_n_stages_map_[var_names[1]].first == 0) {
    this->PrintIndent();
    this->stream << op_name << "(";
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

std::string findValueIfKeyContains(const std::map<std::string, std::string>& myMap, 
                                   const std::string& inputKey) {
    auto it = std::find_if(myMap.begin(), myMap.end(),
        [&inputKey](const auto& pair) {
            return inputKey.find(pair.first) != std::string::npos;
        });
    if (it != myMap.end()) {
        return it->second;
    }
    return "";
}

std::string getValueOrProcess(const std::map<std::string, std::string>& myMap,
                             const std::string& key) {
    auto it = myMap.find(key);
    if (it != myMap.end()) {
        return it->second;
    } else {
        return findValueIfKeyContains(myMap, key);
    }
}

void CodeGenTileLangAscendPto::BinaryVecOpsCodegen(const CallNode *op,
                                               const std::string &op_name) {
  std::vector<std::string> var_names;
  std::string operation = op_name;
  for (int i = 0; i < op->args.size() - 2; i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }
  if (op->args[2].as<CallNode>()) {
    auto var_name = PrintBufferOffset(op->args[2].as<CallNode>());
    std::string ub_name = var_names[1];
    this->PrintIndent();
    std::string index = PrintExpr(op->args[op->args.size() - 2]);
    std::string offset = PrintExpr(op->args[0].as<CallNode>()->args[2]);
    std::string scalar_name = var_name + "_scalar";
    this->stream << "pipe_barrier(PIPE_ALL);\n";
    this->stream << "auto " << scalar_name <<  "= " << var_name
                << ".GetValue(" << index
                << ");\n";
    this->stream << "pipe_barrier(PIPE_ALL);\n";
    std::vector<std::string> ub_data_vector = ub_data_map_[ub_name];
    std::string var_name_temp = ub_name + "_temp";
    std::string ub_data_type = ub_data_vector[0];
    this->PrintIndent();
    std::string loop_num = getValueOrProcess(for_num_map_, index);
    int32_t ub_data_temp_col = std::stoi(ub_data_vector[2]) * std::stoi(ub_data_vector[1]) / std::stoi(loop_num);
    this->stream << kAscendPtoScope << "TileUbDataND<" << ub_data_vector[0] << ", 1, "
    << ub_data_temp_col << ", 1, " << ub_data_temp_col << "> " << var_name_temp << ";\n";
    this->PrintIndent();
    this->stream << "TASSIGN(" << var_name_temp << ", " << ub_data_vector[3] << " + " <<
    offset << " * " << GetTypeLenString(ub_data_vector[0]) << ");\n";
    this->PrintIndent();
    this->stream << "pipe_barrier(PIPE_ALL);\n";
    this->PrintIndent();
    if (operation == "TSUBS") {
      operation = "TADDS";
      scalar_name = "-" + scalar_name;
    }
    this->stream << operation << "(";
    this->stream << var_name_temp << ", " << var_name_temp << ", " << scalar_name;
  } else {
    this->PrintIndent();
    this->stream << operation << "(";
    std::string scalar = PrintExpr(op->args[op->args.size() - 2]);
    var_names.push_back(operation == "TSUBS" ? ("-" + scalar):scalar);
    for (int i = 0; i < var_names.size(); i++) {
      this->stream << var_names[i];
      if (i != var_names.size() - 1) {
        this->stream << ", ";
      }
    }
  }
  this->stream << ");\n";
}

void CodeGenTileLangAscendPto::UnaryVecOpCodegen(const CallNode *op, const std::string& op_name) {
  std::vector<std::string> var_names;
  for (int i = 0; i < op->args.size() - 1; i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }
  this->PrintIndent();
  this->stream << op_name << "(";
  for (int i = 0; i < var_names.size(); i++) {
    this->stream << var_names[i];
    if (i != var_names.size() - 1) {
      this->stream << ", ";
    }
  }
  this->stream << ");\n";
}

void CodeGenTileLangAscendPto::ScalarOpCodegen(const CallNode *op, const std::string& op_name) {
    this->PrintIndent();
    this->stream << op_name << "(" << PrintBufferOffset(op->args[0].as<CallNode>()) << ", "
                  << PrintBufferOffset(op->args[1].as<CallNode>()) << ", "
                  << PrintExpr(op->args[2]) << ");\n";
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
    // this->stream << "pipe_barrier(PIPE_ALL);\n";
    this->stream << "auto " << scalar_name << "= " << var_name << ".GetValue("
                 << index << ");\n";
    this->stream << "pipe_barrier(PIPE_ALL);\n";
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

  // clamp_min: achieve with TMAXS
  this->stream << "TMAXS" << "(";

  for (int i = 0; i < var_names.size(); i++) {
    this->stream << var_names[i];
    if (i != var_names.size() - 1) {
      this->stream << ", ";
    }
  }
  this->stream << ", " << scalar_min << ");\n";

  // clamp_max: achieve with TMINS
  this->stream << "TMINS" << "(";

  for (int i = 0; i < var_names.size(); i++) {
    this->stream << var_names[i];
    if (i != var_names.size() - 1) {
      this->stream << ", ";
    }
  }
  this->stream << ", " << scalar_max << ");\n";
}

void CodeGenTileLangAscendPto::SigmoidCodegen(const CallNode *op, const std::string& op_name) {
  std::vector<std::string> var_names;
  for (int i = 0; i < op->args.size() - 2; i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }
  std::string dst_name = PrintExpr(op->args[0].as<CallNode>()->args[1]);
  std::vector<std::string> ub_data = ub_data_map_[dst_name];
  this->PrintIndent();
  this->stream << kAscendPtoScope << op_name << "<" << ub_data[0] << ", " << ub_data[1] << ", " << ub_data[2] << ">" << "(";
  for (int i = 0; i < var_names.size(); i++) {
    this->stream << var_names[i];
    if (i != var_names.size() - 1) {
      this->stream << ", ";
    }
  }
  this->stream << ", " << PrintExpr(op->args[op->args.size() - 1]) << ");\n";
}

void CodeGenTileLangAscendPto::CastCodegen(const CallNode *op, const std::string& op_type) {
  std::vector<std::string> var_names;
  for (int i = 0; i < op->args.size() - 2; i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }
  this->PrintIndent();
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

std::tuple<int, int, int, bool> ExtractTemplateParamsForSliceBuffer(const std::string& op_name) {
    int second_param = 0;
    int third_param = 0;
    int forth_param = 0;
    size_t left = op_name.find('<');
    size_t right = op_name.find('>');

    if (left == std::string::npos || right == std::string::npos || left >= right) {
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
        } catch (const std::exception& e) {
            return std::make_tuple(second_param, third_param, forth_param, false);
        }
    } else {
      ICHECK(false) << "reduce params less than 4.";
    }
    return std::make_tuple(second_param, third_param, forth_param, false);
}

void CodeGenTileLangAscendPto::ReduceOpCodegen(const CallNode *op) {
  std::string op_name = Downcast<StringImm>(op->args[0])->value;

  //Determine whether the reduce operation needs to be sliced.
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
    ICHECK(false) << "Only row-wise or column-wise reduce operations are supported. Row direction is denoted by -1, and column direction by 0.";
  }

  bool success = std::get<3>(template_params);
  if (!success) {
    ICHECK(false) << "ExtractTemplateParams failed";
  }

  if (op_name.find("reduce_sum") != std::string::npos) {
    op_name = (mode == "row") ? "TROWSUM" : "TCOLSUM";
  } else if (op_name.find("reduce_max") != std::string::npos) {
    op_name = (mode == "row") ? "TROWMAX" : "TCOLMAX";
  } else {
    ICHECK(false) << "not support reduce type: " << op_name;
  }

  std::vector<std::string> var_names;
  for (int i = 1; i < op->args.size(); i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }
  std::string ub_name = var_names[0];
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
  ICHECK(ub_data_vector.size() == 5) << "TileUbData needs 5 elements (type, row, col, ffts, applied DN or not), got " << ub_data_vector.size() << ".";
  if (mode == "row") {
    if (ub_data_vector[4] == "Unapplied for tileUbDataDN") { //If not applied yet, prioritize applying for it.
      this->PrintIndent();
      this->stream << kAscendPtoScope << "TileUbDataDN <" << ub_data_type << ", " << row << ", " << col << ", " << row_src << ", " << col << "> " << ub_name << "_DN;\n";
      this->PrintIndent();
      this->stream << "TASSIGN(" << ub_name << "_DN, " << ffts << ");\n";
      if (param2 == row_src && param3 == col_src) {
        this->PrintIndent();
        this->stream << op_name << "(";
        for (int i = 0; i < var_names.size(); i++) {
          this->stream << var_names[i];
          if (i == 0) {
            this->stream << "_DN";
          }
          if (i != var_names.size() - 1) {
            this->stream << ", ";
          }
        }
        this->stream << ");\n";
      } else {
        if (op_name == "TROWMAX") {
          this->PrintIndent();
          this->stream << kAscendPtoScope << "TROWMAX_with_slice_buffer <" << ub_data_type_src << ", " << ub_data_type << ", " << ub_data_type_tmp << ", " << row_src << ", " << col_src << ", " << param2 << ", " << param3 << ", " << row << ", " << row_tmp << ", " << col_tmp << "> (" << ffts_src << ", " << ffts << ", " << ub_name << "_DN, " << ub_name_tmp <<");\n";
        } else if (op_name == "TROWSUM") {
          this->PrintIndent();
          this->stream << kAscendPtoScope << "TROWSUM_with_slice_buffer <" << ub_data_type_src << ", " << ub_data_type << ", " << ub_data_type_tmp << ", " << row_src << ", " << col_src << ", " << param2 << ", " << param3 << ", " << row << ", " << row_tmp << ", " << col_tmp << "> (" << ffts_src << ", " << ffts << ", " << ub_name << "_DN, " << ub_name_tmp <<");\n";
        }
      }
      this->PrintIndent();
      this->stream << "pipe_barrier(PIPE_ALL);\n";
      this->PrintIndent();
      this->stream << "TRESHAPE(" << var_names[0] << ", " << var_names[0] << "_DN);\n";
      ub_data_vector[4] = "Applied for tileUbDataDN";
    } else if (ub_data_vector[4] == "Applied for tileUbDataDN") { //If already applied, leverage the existing application.
      this->PrintIndent();
      this->stream << op_name << "(";
      for (int i = 0; i < var_names.size(); i++) {
        this->stream << var_names[i];
        if (i == 0) {
          this->stream << "_DN";
        }
        if (i != var_names.size() - 1) {
          this->stream << ", ";
        }
      }
      this->stream << ");\n";
      this->PrintIndent();
      this->stream << "pipe_barrier(PIPE_ALL);\n";
      this->PrintIndent();
      this->stream << "TRESHAPE(" << var_names[0] << ", " << var_names[0] << "_DN);\n";
    } else {
      ICHECK(false) << "Error route in ReduceOpCodegen";
    }
  } else {
    this->PrintIndent();
    if (param2 == row_src && param3 == col_src) {
      this->stream << op_name << "(";
      for (int i = 0; i < var_names.size() - 1; i++) {
        this->stream << var_names[i];
        if (i != var_names.size() - 2) {
          this->stream << ", ";
        }
      }
      if (op_name == "TCOLSUM") {
        this->stream << ", " << ub_name_tmp << ", true";
      }
      this->stream << ");\n";
    } else {
      this->stream << kAscendPtoScope << op_name << "_with_slice_buffer <" << ub_data_type_src << ", " << ub_data_type << ", " << ub_data_type_tmp << ", " << row_src << ", " << col_src << ", " << param2 << ", " << param3 << ", " << row << ", " << row_tmp << ", " << col_tmp << "> (" << ffts_src << ", " << ffts << ", " << ub_name << ", " << ub_name_tmp <<");\n";
    }
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
      this->stream << "auto " << current_block_id
                   << " = get_block_idx();\n";
      this->PrintIndent();
      stream << "set_ffts_base_addr(ffts_Addr);\n\n";

      this->core_num_ = PrintExpr(op->value);
    } else if (iv->thread_tag == "blockIdx.y" && iv->var->name_hint != "_") {
      this->vec_id_ = AllocVarID(iv->var.get());
      this->PrintIndent();
      auto current_vec_id = this->vec_id_;
      this->stream << "auto " << current_vec_id
                   << " = get_subblockid();\n";
    }
    this->VisitStmt(op->body);
    return;
  } else if (op->attr_key == "resource_scope") { // other core
    auto resource_id = Downcast<IntImm>(op->value)->value;
    auto resource_name = resource_id == 0 ? "CUBE" : "VEC";
    std::string arch_name = (this->platform_ == "A5") ? "C310" : "C220";

    stream << "#if defined(__DAV_" << arch_name << "_" << resource_name << "__)\n";
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
  if (shape.size() > 3 || shape.size() == 0){
    ICHECK(false) << "Unsupported ubsize which is expected to be 1, 2 or 3";
  }
}

bool CodeGenTileLangAscendPto::ValidLayoutEnabled(const AllocateNode *op) {
  auto shape = buffer_shapess_[op->buffer_var];
  bool valid = false;
  std::string type = getType(op->dtype);
  int8_t typeSize = GetTypeLen(type);
  if (tvm::tir::is_zero(tvm::truncmod(shape[1] * typeSize, 32))) {
    valid = false;
  } else {
    valid = true;
  }
  return valid;
}

void CodeGenTileLangAscendPto::VisitStmt_(const AllocateNode *op) {
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get()); // var_name
  std::string scope = GetPtrStorageScope(op->buffer_var);
  std::string type = getType(op->dtype);
  const VarNode *buffer = op->buffer_var.as<VarNode>();

  if (buffer_shapess_.count(op->buffer_var)) {
    auto shape = buffer_shapess_[op->buffer_var];
  } else {
    ICHECK(false) << "Buffer_shape not found.";
  }

  /// Allocate PTO Tile Memory Address
  auto print_buffer = [&](const std::string &pos) {
    auto shape = buffer_shapess_[op->buffer_var];
    std::vector<std::string> ub_data(5);
    std::vector<std::string> l_data(3);
    ub_data[0] = type;
    l_data[0] = type;
    if (pos == kAscendPtoScope + "TileUbData") {
      UbShapeInputCheck(op);
    }
    PrimExpr target_expr;
    bool found_by_name = false;
    std::string target_var_name = op->buffer_var->name_hint;

    for (const auto& pair : address_map_) {
      Var var_key = pair.first;
      if (var_key->name_hint == target_var_name) {
        target_expr = pair.second;
        found_by_name = true;
        break;
      }
    }
    this->PrintIndent();
    // Allocate buffer
    if (found_by_name) {
      if (pos == kAscendPtoScope + "TileUbData") {
        if (shape.size() == 2) {
          ub_data[1] = PrintExpr(shape[0]);
          ub_data[2] = PrintExpr(shape[1]);
          auto valid = ValidLayoutEnabled(op);
          if (!valid) {
            stream << pos << "ND<" << type;
            for (size_t i = 0; i < shape.size(); i++) {
                stream << ", " << shape[i];
            }
          } else {
            int8_t typeSize = GetTypeLen(type);
            int8_t NDBlockSize = 32 / typeSize;
            stream << pos << "ND<" << type;
            stream << ", " << shape[0];
            stream << ", " << tvm::floordiv(shape[1] + NDBlockSize - 1, NDBlockSize) * NDBlockSize;
          }
          // stream << "> " << vid << "(";
          for (size_t i = 0; i < shape.size(); i++) {
            stream << ", " << shape[i];
          }
          stream << "> " << vid << ";\n";
          ub_data[3] = PrintExpr(target_expr);
          ub_data[4] = "Unapplied for tileUbDataDN";
          ub_data_map_[vid] = ub_data;

          this->PrintIndent();
          stream << "TASSIGN(" << vid << ", " << PrintExpr(target_expr) << ");\n";
        } else if (shape.size() == 1) {
          ub_data[1] = "1";
          ub_data[2] = PrintExpr(shape[0]);
          stream << pos << "ND<" << type << ", 1, " << shape[0] << ", 1, " << shape[0]<< "> " << vid << ";\n";
          ub_data[3] = PrintExpr(target_expr);
          ub_data[4] = "Unapplied for tileUbDataDN";
          ub_data_map_[vid] = ub_data;
          this->PrintIndent();
          stream << "TASSIGN(" << vid << ", " << PrintExpr(target_expr) << ");\n";
        } else if (shape.size() == 3) {
          ub_data[1] = PrintExpr(shape[1]);
          ub_data[2] = PrintExpr(shape[2]);
          ub_data_map_[vid] = ub_data;
          int8_t bufferNum = shape[0].as<IntImmNode>()->value;
          prefetch_n_stages_map_[vid] = std::pair<int, int> {bufferNum, 0};
          auto valid = ValidLayoutEnabled(op);
          if (!valid) {
            stream << pos << "ND<" << type;
            for (size_t i = 1; i < shape.size(); i++) {
                stream << ", " << shape[i];
            }
          } else {
            int8_t typeSize = GetTypeLen(type);
            int8_t NDBlockSize = 32 / typeSize;
            stream << pos << "ND<" << type;
            stream << ", " << shape[0];
            stream << ", " << tvm::floordiv(shape[1] + NDBlockSize - 1, NDBlockSize) * NDBlockSize;
          }
          for (size_t i = 1; i < shape.size(); i++) {
            stream << ", " << shape[i];
          }
          stream << "> " << vid << "[" << shape[0] << "];\n";
          for (size_t i = 0; i < bufferNum; i++) {
            this->PrintIndent();
            stream << "TASSIGN(" << vid << "[" << i << "], " << PrintExpr(target_expr) << ");\n";
          }
        }
      } else {
        if (!(pos == kAscendPtoScope + "TileMatL1" && shape.size() == 3)) {
        int dtype_bytes = op->dtype.bytes();
        std::vector<PrimExpr> valid_shapes;
        valid_shapes.reserve(2);
        stream << pos << "<" << type;
        int shape_value = shape[0].as<tvm::tir::IntImmNode>()->value;
        if (shape_value * dtype_bytes % 32 == 0) {
            valid_shapes.push_back(shape[0]);
        } else {
            valid_shapes.push_back(tvm::IntImm(shape[0].dtype(),
            shape_value + (32 - shape_value * dtype_bytes % 32) / dtype_bytes));
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
          stream << "TASSIGN(" << vid << ", " << PrintExpr(target_expr) << ");\n";
        } else {
          int8_t bufferNum = shape[0].as<IntImmNode>()->value;
          prefetch_n_stages_map_[vid] = std::pair<int, int> {bufferNum, 0};
          int dtype_bytes = op->dtype.bytes();
          std::vector<PrimExpr> valid_shapes;
          valid_shapes.reserve(shape.size()-1);
          stream << pos << "<" << type;
          int shape_value = shape[1].as<tvm::tir::IntImmNode>()->value;
          if (shape_value * dtype_bytes % 32 == 0) {
              valid_shapes.push_back(shape[1]);
          } else {
              valid_shapes.push_back(tvm::IntImm(shape[1].dtype(),
              shape_value + (32 - shape_value * dtype_bytes % 32) / dtype_bytes));
          }
          valid_shapes.push_back(shape[2]);
          for (size_t i = 0; i < valid_shapes.size(); i++) {
              l_data[i + 1] = PrintExpr(valid_shapes[i]);
              stream << ", " << valid_shapes[i];
          }
          for (size_t i = 1; i < shape.size(); i++) {
              stream << ", " << shape[i];
          }
          stream << "> " << vid << "[" << shape[0] << "];\n";
          for (size_t j = 0; j < bufferNum; j++) {
            this->PrintIndent();
            stream << "TASSIGN(" << vid << "[" << j << "], " << PrintExpr(target_expr) << ");\n";
          }
        }
      }
      l_data_map_[vid] = l_data;
    } else {
      if (address_offset_.find(String(pos)) == address_offset_.end()) {
        address_offset_.Set(String(pos), 0);
      }
      if (pos == kAscendPtoScope + "TileUbData") {
        if (shape.size() == 2) {

          ub_data[1] = PrintExpr(shape[0]);
          ub_data[2] = PrintExpr(shape[1]);
          auto valid = ValidLayoutEnabled(op);
          if (!valid) {
            stream << pos << "ND<" << type;
            for (size_t i = 0; i < shape.size(); i++) {
                stream << ", " << shape[i];
            }
          } else {
            int8_t typeSize = GetTypeLen(type);
            int8_t NDBlockSize = 32 / typeSize;
            stream << pos << "ND<" << type;
            stream << ", " << shape[0];
            stream << ", " << tvm::floordiv(shape[1] + NDBlockSize - 1, NDBlockSize) * NDBlockSize;
          }
          for (size_t i = 0; i < shape.size(); i++) {
            stream << ", " << shape[i];
          }
          stream << "> " << vid << ";\n";
          ub_data[3] = DEC_STR_TO_HEX_STR(PrintExpr(address_offset_[String(pos)]));
          ub_data[4] = "Unapplied for tileUbDataDN";
          ub_data_map_[vid] = ub_data;
          this->PrintIndent();
          stream << "TASSIGN(" << vid << ", " << DEC_STR_TO_HEX_STR(PrintExpr(address_offset_[String(pos)])) << ");\n";
          address_offset_.Set(
          String(pos),
          PrimExpr(int(op->ConstantAllocationSize() * op->dtype.bytes())) +
          address_offset_[String(pos)]);
        } else if (shape.size() == 1) {
          ub_data[1] = "1";
          ub_data[2] = PrintExpr(shape[0]);
          stream << pos << "ND<" << type << ", 1, " << shape[0] << ", 1, " << shape[0]<< "> " << vid << ";\n";
          ub_data[3] = DEC_STR_TO_HEX_STR(PrintExpr(address_offset_[String(pos)]));
          ub_data[4] = "Unapplied for tileUbDataDN";
          ub_data_map_[vid] = ub_data;
          this->PrintIndent();
          stream << "TASSIGN(" << vid << ", " << DEC_STR_TO_HEX_STR(PrintExpr(address_offset_[String(pos)])) << ");\n";
          address_offset_.Set(
              String(pos),
              PrimExpr(int(op->ConstantAllocationSize() * op->dtype.bytes())) +
                  address_offset_[String(pos)]);
        } else if (shape.size() == 3) {
          ub_data[1] = PrintExpr(shape[1]);
          ub_data[2] = PrintExpr(shape[2]);
          ub_data_map_[vid] = ub_data;
          int8_t bufferNum = shape[0].as<IntImmNode>()->value;
          prefetch_n_stages_map_[vid] = std::pair<int, int> {bufferNum, 0};
          // if (j !=0 ) {this->PrintIndent();}
          auto valid = ValidLayoutEnabled(op);
          if (!valid) {
            stream << pos << "ND<" << type;
            for (size_t i = 1; i < shape.size(); i++) {
                stream << ", " << shape[i];
            }
          } else {
            int8_t typeSize = GetTypeLen(type);
            int8_t NDBlockSize = 32 / typeSize;
            stream << pos << "ND<" << type;
            stream << ", " << shape[0];
            stream << ", " << tvm::floordiv(shape[1] + NDBlockSize - 1, NDBlockSize) * NDBlockSize;
          }
          for (size_t i = 1; i < shape.size(); i++) {
            stream << ", " << shape[i];
          }
          stream << "> " << vid << "[" << shape[0] << "];\n";
          for (size_t j = 0; j < bufferNum; j++) {
            this->PrintIndent();
            stream << "TASSIGN(" << vid << "[" << j << "], " << DEC_STR_TO_HEX_STR(PrintExpr(address_offset_[String(pos)])) << ");\n";
            address_offset_.Set(
                String(pos),
                PrimExpr(int(op->ConstantAllocationSize() * op->dtype.bytes())) +
                    address_offset_[String(pos)]);
          }
        }
      } else {
        if (!(pos == kAscendPtoScope + "TileMatL1" && shape.size() == 3)) {
        int dtype_bytes = op->dtype.bytes();
        std::vector<PrimExpr> valid_shapes;
        valid_shapes.reserve(2);
        stream << pos << "<" << type;
        int shape_value = shape[0].as<tvm::tir::IntImmNode>()->value;
        if (shape_value * dtype_bytes % 32 == 0) {
            valid_shapes.push_back(shape[0]);
        } else {
            valid_shapes.push_back(tvm::IntImm(shape[0].dtype(),
            shape_value + (32 - shape_value * dtype_bytes % 32) / dtype_bytes));
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
          stream << "TASSIGN(" << vid << ", " << DEC_STR_TO_HEX_STR(PrintExpr(address_offset_[String(pos)])) << ");\n";
          address_offset_.Set(
              String(pos),
              PrimExpr(int(op->ConstantAllocationSize() * op->dtype.bytes())) +
                  address_offset_[String(pos)]);
        } else {
          int8_t bufferNum = shape[0].as<IntImmNode>()->value;
          prefetch_n_stages_map_[vid] = std::pair<int, int> {bufferNum, 0};
          int dtype_bytes = op->dtype.bytes();
          std::vector<PrimExpr> valid_shapes;
          valid_shapes.reserve(shape.size()-1);
          stream << pos << "<" << type;
          int shape_value = shape[1].as<tvm::tir::IntImmNode>()->value;
          if (shape_value * dtype_bytes % 32 == 0) {
              valid_shapes.push_back(shape[1]);
          } else {
              valid_shapes.push_back(tvm::IntImm(shape[1].dtype(),
              shape_value + (32 - shape_value * dtype_bytes % 32) / dtype_bytes));
          }
          valid_shapes.push_back(shape[2]);
          for (size_t i = 0; i < valid_shapes.size(); i++) {
            l_data[i + 1] = PrintExpr(valid_shapes[i]);
              stream << ", " << valid_shapes[i];
          }
          for (size_t i = 1; i < shape.size(); i++) {
              stream << ", " << shape[i];
          }
          stream << "> " << vid << "[" << shape[0] << "];\n";
          for (size_t j = 0; j < bufferNum; j++) {
                        this->PrintIndent();
            stream << "TASSIGN(" << vid << "[" << j << "], " << DEC_STR_TO_HEX_STR(PrintExpr(address_offset_[String(pos)])) << ");\n";
            address_offset_.Set(
                String(pos),
                PrimExpr(int(op->ConstantAllocationSize() * op->dtype.bytes())) +
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

void CodeGenTileLangAscendPto::VisitExpr_(const SelectNode *op, std::ostream &os) {
  auto condition = PrintExpr(op->condition);
  auto true_value = PrintExpr(op->true_value);
  auto false_value = PrintExpr(op->false_value);

  os << "(" << condition << " ? "
     << "" << true_value << " : " << false_value << ")";
}

static void ProcessHostInput(std::ostream &os, std::vector<std::string> &arg_names,
                      std::vector<const tir::VarNode *> &shape_vars, bool add_args = true) {
  for (auto shape_var : shape_vars) {
    os << ", "
       << "int64_t " << shape_var->name_hint;
  if (add_args)
    { arg_names.push_back(shape_var->name_hint); }
  }
}

void CodeGenTileLangAscendPto::CallTilingInput(std::ostream &os, std::string func_name, std::vector<std::string> &tiling_args,
  std::vector<const tir::VarNode*> &shape_vars)
{

}

void CodeGenTileLangAscendPto::ProcessTilingInput(std::ostream &os, std::string func_name, std::vector<std::string> &tiling_args,
  std::vector<const tir::VarNode*> &shape_vars)
{

}

void CodeGenTileLangAscendPto::PrintHostFunc(const PrimFunc &f, const std::string &name,
                                          std::ostringstream &os, std::string &core,
                                          std::vector<const tir::VarNode *> &shape_vars) {
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
    os << "__gm__ uint8_t *" << v->name_hint;
  }
  ProcessHostInput(os, arg_names, shape_vars);
  int func_scope = this->BeginScope();
  os << ", uint64_t fftsAddr)\n{\n  ";
  this->PrintIndent();
  // template function
  os << name << "(";
  for (size_t i = 0; i < f->params.size(); ++i) { // params
    auto v = f->params[i];
    if (i != 0) {os << ",\n     ";}
    os << "reinterpret_cast<__gm__ " << global_tensor_template[String(v->name_hint)].dtype
    << " *>(" << v->name_hint << ")";
  }
  for (auto shape_var : shape_vars) {
    os << ", " << shape_var->name_hint;
  }
  os << ", fftsAddr);\n}\n\n";


  // call kernel
  os << "extern \"C\" void call(";
  for (size_t i = 0; i < f->params.size(); ++i) { // params
    auto v = f->params[i];
    if (i != 0) {
      os << ", ";
    }
    // arg_names.push_back(v->name_hint);
    os << "uint8_t *" << v->name_hint;
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

  address_map_ = f->GetAttr<Map<Var, PrimExpr>>("address_map").value_or(Map<Var, PrimExpr>());
  use_swizzle_ = f->GetAttr<Bool>("use_swizzle").value_or(Bool(false));
  // tiling_map_ = f->GetAttr<Map<Var, PrimExpr>>("tiling_map").value_or(Map<Var, PrimExpr>());
  buffer_shapess_ = f->GetAttr<Map<Var, Array<PrimExpr>>>("buffer_shapess").value_or(Map<Var, Array<PrimExpr>>());
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
               shape_vars.end()) && shape_var != 0) {
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
        if(shape_info[0]<'1' || shape_info[0]>'9') shape_type = "dynamic";
      }
      global_tensor gt = {shape_type, String(getType(f->buffer_map[v]->dtype)), copy_tmp_shape};
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
    }
    stream <<  "__gm__ " << getType(f->buffer_map[v]->dtype) << " *" << vid;
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
  this->stream << op_name << "(PIPE_" << src << ", " << "PIPE_" <<
      dst << ", " << "EVENT_ID" << event_id << ");\n";
}

} // namespace codegen
} // namespace tvm
