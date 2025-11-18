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

#include "../op/builtin.h"

#include "arith/pattern_match.h"

namespace tvm {
namespace codegen {

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

CodeGenTileLangAscendPto::CodeGenTileLangAscendPto() {
}

void CodeGenTileLangAscendPto::PrintFuncPrefix(std::ostream &os) {
  //os << "extern \"C\" CATLASS_GLOBAL\n";
}

std::string CodeGenTileLangAscendPto::Finish() {
  decl_stream << "#include <common/tile_tensor_impl.hpp>\"\n";
  decl_stream << "#include <common/pto_tileop.hpp>\"\n";
  decl_stream << "#include <common/constants.hpp>\n";
  decl_stream << "#include \"acl/acl.h\"\n";
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
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  std::string start = PrintExpr(op->min);
  stream << "for (";
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
  // auto var_name = var_idmap_[op->buffer->data.get()];
  // os << var_name << ".GetValue("
  //               << PrintExpr(op->indices.back()) << ")";
}

void CodeGenTileLangAscendPto::VisitExpr_(const CallNode *op, std::ostream &os) {
  
}

void CodeGenTileLangAscendPto::VisitStmt_(const AttrStmtNode *op) {
  
}

void CodeGenTileLangAscendPto::VisitStmt_(const AllocateNode *op) {
  
}

inline void PrintConst(const FloatImmNode *op, std::ostream &os,
                       CodeGenTileLangAscendPto *p) { // NOLINT(*)
}

void CodeGenTileLangAscendPto::VisitExpr_(const FloatImmNode *op,
                                       std::ostream &os) { // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenTileLangAscendPto::PreFunctionBody(const PrimFunc &f) {
  int func_scope = this->BeginScope();
  this->PrintIndent();

  // 分配GlobalTensor  i:var_name_handle  i+1:var_name  i+2:dtype  i+3:shape0  i+4:shape1
  // GlobalTensor最大支持5维，后续修改可能需要增加判断
  for (size_t i = 0; i < this->para_.size(); i += 5) {
    this->PrintIndent();
    std::string shape0 = this->para_[i + 3], shape1 = this->para_[i + 4];
    stream << "GlobalTensor<" << this->para_[i + 2] << ", Shape<1, 1, 1, " << 
    shape0 << ", " << shape1 << ">, Stride<" << shape0 << " * " << shape1 << 
    ", " << shape0 << " * " << shape1 << ", " << shape0 << " * " << shape1 << 
    ", " << shape1 << ", 1>> " << this->para_[i + 1] << "Global(" << 
    this->para_[i + 1] << ");\n";
    this->PrintIndent();
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
                      std::vector<const tir::VarNode *> &shape_vars) {
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
  
}

void CodeGenTileLangAscendPto::AddFunction(const GlobalVar &gvar,
                                        const PrimFunc &f) {
  
}

} // namespace codegen
} // namespace tvm
