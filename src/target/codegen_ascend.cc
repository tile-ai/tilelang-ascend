// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file target/codegen.cc
 */

#include "codegen_ascend.h"
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

std::string getType(const DataType &dtype) {
  if (dtype.is_float16()) {
    return "half";
  } else if (dtype.is_float()) {
    return "float";
  } else if (dtype.is_int()) {
    return "int";
  } else if (dtype.is_uint() && dtype.bits() == 8) {
    return "uint8_t";
  } else if (dtype.is_uint() && dtype.bits() == 32) {
    return "uint32_t";
  }
  LOG(FATAL) << "Unsupported data type: " << dtype;
  return "";
}

CodeGenTileLangAscend::CodeGenTileLangAscend() {
  restrict_keyword_ = "GM_ADDR";
}

void CodeGenTileLangAscend::PrintFuncPrefix(std::ostream &os) {
  os << "extern \"C\" CATLASS_GLOBAL\n";
}

std::string CodeGenTileLangAscend::Finish() {
  decl_stream << "#include \"tl_templates/ascend/common.h\"\n";
  decl_stream << "#include \"acl/acl.h\"\n";
  decl_stream << "#include <runtime/rt_ffts.h>\n";
  decl_stream << "using namespace Catlass;\n";
  decl_stream << "\n";
  std::ostringstream code;
  code << decl_stream.str();
  code << stream.str();
  return code.str();
}

void CodeGenTileLangAscend::VisitStmt_(const tir::ForNode *op) {
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

void CodeGenTileLangAscend::PrintType(DataType t,
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

void CodeGenTileLangAscend::PrintStorageScope(const std::string &scope,
                                              std::ostream &os) { // NOLINT(*)
}

void CodeGenTileLangAscend::VisitExpr_(const FloorDivNode *op,
                                       std::ostream &os) {
  os << "(";
  PrintExpr(op->a, os);
  os << " / ";
  PrintExpr(op->b, os);
  os << ")";
}

void CodeGenTileLangAscend::VisitExpr_(const FloorModNode *op,
                                       std::ostream &os) {
  os << "(";
  PrintExpr(op->a, os);
  os << " % ";
  PrintExpr(op->b, os);
  os << ")";
}

void CodeGenTileLangAscend::VisitExpr_(const BufferLoadNode *op,
                                       std::ostream &os) {
  static int index = 0;
  Array<PrimExpr> indices;
  indices.push_back(0);

  auto new_i = op->buffer->ElemOffset(indices).back();
  auto var_name = var_idmap_[op->buffer->data.get()];

  auto new_var_name = var_name + "_scalar_" + std::to_string(index);

  this->PrintIndent();
  this->stream << "auto " << new_var_name << "= " << var_name << ".GetValue("
               << PrintExpr(op->indices.back()) << ");\n";
  this->PrintIndent();
  this->stream << "AscendC::PipeBarrier<PIPE_ALL>();\n";
  os << new_var_name;
  index++;
  // os << var_name << "_scalar";
}

void CodeGenTileLangAscend::VisitExpr_(const CallNode *op, std::ostream &os) {
  auto print_buffer_offset = [&](const CallNode *op,
                                 bool has_offset = true) -> std::string {
    auto _var = op->args[1].as<VarNode>();
    auto _var_offset = PrintExpr(op->args[2]);
    auto _var_name = var_idmap_[_var];
    if (has_offset)
      return _var_name + "[" + _var_offset + "]";
    return _var_name;
  };

  if (op->op.same_as(builtin::call_extern())) {
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

      auto src_type = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;
      auto dst_type = op->args[2].as<CallNode>()->args[0].as<CallNode>()->dtype;

      static const std::unordered_map<std::string, int> kCopyOpExtraArgs = {
        {"copy_l0c_to_gm", 2},
        {"copy_gm_to_l1", 1},
        {"copy_l1_to_l0a", 0},
        {"copy_l1_to_l0b", 0},
        {"copy_gm_to_ub", 1},
        {"copy_ub_to_gm", 1},
        {"copy_ub_to_ub", 0}
      };

      bool found = false;
      int extra_args = 0;

      for (const auto& pair : kCopyOpExtraArgs) {
        if (op_name.find(pair.first) != std::string::npos) {
          found = true;
          extra_args = pair.second;
          break;
        }
      }

      if (found) {
        this->PrintIndent();
        this->stream << op_name << "(" << dst_var_id << "[" << dst_offset
                    << "], " << src_var_id << "[" << src_offset << "]";

        for (int i = 0; i < extra_args; ++i) {
          this->stream << ", " << PrintExpr(op->args[3 + i]);
        }

        this->stream << ");\n";
      } else {
        this->PrintIndent();
        this->stream << "not implemented yet\n";
      }
    } else if (op_name == "npu.fill") {
      this->PrintIndent();
    } else if (op_name.find("mma") != std::string::npos) {
      auto a_var = op->args[1].as<CallNode>()->args[1].as<VarNode>();
      auto b_var = op->args[2].as<CallNode>()->args[1].as<VarNode>();
      auto c_var = op->args[3].as<CallNode>()->args[1].as<VarNode>();

      auto a_offset = PrintExpr(op->args[1].as<CallNode>()->args[2]);
      auto b_offset = PrintExpr(op->args[2].as<CallNode>()->args[2]);
      auto c_offset = PrintExpr(op->args[3].as<CallNode>()->args[2]);

      auto a_name = var_idmap_[a_var];
      auto b_name = var_idmap_[b_var];
      auto c_name = var_idmap_[c_var];

      this->PrintIndent();
      this->stream << op_name << "(" << a_name << "[" << a_offset << "],"
                   << b_name << "[" << b_offset << "]," << c_name << "["
                   << c_offset << "], " << PrintExpr(op->args[4]) << ");\n";
    } else if (op_name.find("AscendC::CrossCoreWaitFlag") !=
               std::string::npos) {
      this->PrintIndent();
      this->stream << op_name << "(" << PrintExpr(op->args[1]) << ");\n";
    } else if (op_name.find("AscendC::CrossCoreSetFlag") != std::string::npos) {
      this->PrintIndent();
      this->stream << op_name << "(" << PrintExpr(op->args[1]) << ");\n";
    } else if (op_name.find("AscendC::WaitFlag") != std::string::npos) {
      this->PrintIndent();
      this->stream << op_name << "(" << PrintExpr(op->args[1]) << ");\n";
    } else if (op_name.find("AscendC::SetFlag") != std::string::npos) {
      this->PrintIndent();
      this->stream << op_name << "(" << PrintExpr(op->args[1]) << ");\n";
    } else if (op_name.find("Duplicate") != std::string::npos) {
      this->PrintIndent();
      auto var_name = print_buffer_offset(op->args[1].as<CallNode>());
      this->stream << op_name << "(" << var_name << ", "
                   << PrintExpr(op->args[2]) << ", " << PrintExpr(op->args[3])
                   << ");\n";
    } else if (op_name.find("PipeBarrier") != std::string::npos) {
      this->PrintIndent();
      this->stream << op_name << "();\n";
    } else if (op_name.find("reduce") != std::string::npos) {
      // this->PrintIndent();
      // this->stream << "{\n";
      // auto func_scope = this->BeginScope();
      std::vector<std::string> var_names;
      for (int i = 1; i < op->args.size(); i++) {
        auto var_name = print_buffer_offset(op->args[i].as<CallNode>());
        var_names.push_back(var_name);
        // this->PrintIndent();
        // this->stream << instr;
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
      // this->EndScope(func_scope);
      // this->PrintIndent();
      // this->stream << "}\n";
    } else if (op_name.find("MergeSort") != std::string::npos) {
      // vtm::Dump(op);
      std::vector<std::string> var_names;
      for (int i = 1; i < op->args.size() - 3; i++) {
        auto var_name = print_buffer_offset(op->args[i].as<CallNode>());
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
      this->stream << ", " << PrintExpr(op->args[op->args.size() - 3]) << ", "
                   << PrintExpr(op->args[op->args.size() - 2]) << ", "
                   << PrintExpr(op->args[op->args.size() - 1]) << ");\n";
    } else if (op_name.find("Sort") != std::string::npos) {
      // tvm::Dump(op);
      std::vector<std::string> var_names;
      for (int i = 1; i < op->args.size() - 3; i++) {
        auto var_name = print_buffer_offset(op->args[i].as<CallNode>());
        var_names.push_back(var_name);
      }

      auto var_name =
          print_buffer_offset(op->args[op->args.size() - 3].as<CallNode>(),
                              false); // tensor with offset will be
      var_names.push_back(var_name);

      this->PrintIndent();
      this->stream << op_name << "(";
      for (int i = 0; i < var_names.size(); i++) {
        this->stream << var_names[i];
        if (i != var_names.size() - 1) {
          this->stream << ", ";
        }
      }
      this->stream << ", " << PrintExpr(op->args[op->args.size() - 1])
                   << ");\n";
    } else if (op_name.find("TopK") != std::string::npos) {
      // tvm::Dump(op);
      std::vector<std::string> var_names;
      for (int i = 1; i < op->args.size() - 1; i++) {
        auto var_name = print_buffer_offset(op->args[i].as<CallNode>());
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
      this->stream << ", " << PrintExpr(op->args[op->args.size() - 1])
                   << ");\n";
    } else if (op_name.find("GatherMask") != std::string::npos) {
      // tvm::Dump(op);
      std::vector<std::string> var_names;
      for (int i = 1; i < op->args.size() - 1; i++) {
        auto var_name = print_buffer_offset(op->args[i].as<CallNode>());
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
      this->stream << ", " << PrintExpr(op->args[op->args.size() - 1])
                   << ");\n";
    } else if (op_name.find("InitSortBuf") != std::string::npos) {
      tvm::Dump(op);
      this->PrintIndent();
      auto var_name = print_buffer_offset(op->args[1].as<CallNode>());
      this->stream << op_name << "(" << var_name << ", "
                   << PrintExpr(op->args[2]) << ");\n";
    } else if (op_name.find("AscendC::ArithProgression") != std::string::npos) {
      this->PrintIndent();
      auto var_name = print_buffer_offset(op->args[1].as<CallNode>());

      this->stream << op_name << "(" << var_name << ", "
                   << PrintExpr(op->args[2]) << ", " << PrintExpr(op->args[3])
                   << ", " << PrintExpr(op->args[4]) << ");\n";
    } else if (op_name.find("AscendC::LeakyRelu") != std::string::npos ||
               op_name.find("AscendC::Axpy") != std::string::npos) {
      this->PrintIndent();
      auto var_name = print_buffer_offset(op->args[1].as<CallNode>());
      auto var_name_1 = print_buffer_offset(op->args[2].as<CallNode>());
      this->stream << op_name << "(" << var_name << ", "
                   << var_name_1 << "," << PrintExpr(op->args[3])
                   << ", " << PrintExpr(op->args[4]) << ");\n";
    } else if (op_name == "AscendC::Add" || op_name == "AscendC::Max" ||
               op_name == "AscendC::Min" || op_name == "AscendC::Sub" ||
               op_name == "AscendC::Mul" || op_name == "AscendC::Exp" ||
               op_name == "AscendC::Ln" || op_name == "AscendC::Abs" ||
               op_name == "AscendC::Reciprocal" || op_name == "AscendC::Sqrt" ||
               op_name == "AscendC::Rsqrt" || op_name == "AscendC::Not" ||
               op_name == "AscendC::Relu") {
      std::vector<std::string> var_names;
      for (int i = 1; i < op->args.size() - 1; i++) {
        auto var_name = print_buffer_offset(op->args[i].as<CallNode>());
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
      this->stream << ", " << PrintExpr(op->args[op->args.size() - 1])
                   << ");\n";
    } else if (op_name == "AscendC::Muls" || op_name == "AscendC::Adds") {
      std::vector<std::string> var_names;
      for (int i = 1; i < 3; i++) {
        auto var_name = print_buffer_offset(op->args[i].as<CallNode>());
        var_names.push_back(var_name);
      }

      if (op->args[3].as<CallNode>()) {
        auto var_name = print_buffer_offset(op->args[3].as<CallNode>(), false);
        this->PrintIndent();
        this->stream << "auto " << var_name << "_scalar = " << var_name
                     << ".GetValue(" << PrintExpr(op->args[op->args.size() - 2])
                     << ");\n";
        var_names.push_back(var_name + "_scalar");
        this->PrintIndent();
        this->stream << "AscendC::PipeBarrier<PIPE_ALL>();\n";
      } else {
        var_names.push_back(PrintExpr(op->args[op->args.size() - 2]));
      }
      this->PrintIndent();
      this->stream << op_name << "(";
      for (int i = 0; i < var_names.size(); i++) {
        this->stream << var_names[i];
        if (i != var_names.size() - 1) {
          this->stream << ", ";
        }
      }

      this->stream << ", " << PrintExpr(op->args[op->args.size() - 1])
                   << ");\n";
    } else if (op_name == "AscendC::Divs") {
      std::vector<std::string> var_names;
      for (int i = 1; i < 3; i++) {
        auto var_name = print_buffer_offset(op->args[i].as<CallNode>());
        var_names.push_back(var_name);
      }

      if (op->args[3].as<CallNode>()) {
        auto var_name = print_buffer_offset(op->args[3].as<CallNode>(), false);

        this->PrintIndent();
        this->stream << "auto " << var_name << "_scalar = 1.0f / " << var_name
                     << ".GetValue(" << PrintExpr(op->args[op->args.size() - 2])
                     << ");\n";
        var_names.push_back(var_name + "_scalar");
        this->PrintIndent();
        this->stream << "AscendC::PipeBarrier<PIPE_ALL>();\n";
      } else {
        var_names.push_back(PrintExpr(op->args[op->args.size() - 2]));
      }
      this->PrintIndent();
      this->stream << "AscendC::Muls"
                   << "(";
      for (int i = 0; i < var_names.size(); i++) {
        this->stream << var_names[i];
        if (i != var_names.size() - 1) {
          this->stream << ", ";
        }
      }

      this->stream << ", " << PrintExpr(op->args[op->args.size() - 1])
                   << ");\n";
    } else if (op_name == "AscendC::Subs") {
      std::vector<std::string> var_names;
      for (int i = 1; i < 3; i++) {
        auto var_name = print_buffer_offset(op->args[i].as<CallNode>());
        var_names.push_back(var_name);
      }

      if (op->args[3].as<CallNode>()) {
        auto var_name = print_buffer_offset(op->args[3].as<CallNode>(), false);

        this->PrintIndent();
        this->stream << "auto " << var_name << "_scalar = " << var_name
                     << ".GetValue(" << PrintExpr(op->args[op->args.size() - 2])
                     << ");\n";
        var_names.push_back("- " + var_name + "_scalar");
        this->PrintIndent();
        this->stream << "AscendC::PipeBarrier<PIPE_ALL>();\n";
      } else {
        var_names.push_back(PrintExpr(op->args[op->args.size() - 2]));
      }
      this->PrintIndent();
      this->stream << "AscendC::Adds"
                   << "(";
      for (int i = 0; i < var_names.size(); i++) {
        this->stream << var_names[i];
        if (i != var_names.size() - 1) {
          this->stream << ", ";
        }
      }

      this->stream << ", " << PrintExpr(op->args[op->args.size() - 1])
                   << ");\n";
    } else if (op_name.find("thread_block_swizzle") != std::string::npos) {
      this->PrintIndent();
      std::string expr = PrintExpr(op->args[1]);
      if (!block_id_.empty()) {
        size_t pos = 0;
        const std::string replacement = block_id_ + '_';
        const size_t block_len = block_id_.length();
        while ((pos = expr.find(block_id_, pos)) != std::string::npos) {
          expr.replace(pos, block_len, replacement);
          pos += replacement.length();
        }
      }
      this->stream << "auto " << this->block_id_ << " = " << op_name << "("
                   << expr << ");\n";
      if (op->args.size() > 2) {
        this->PrintIndent();
        this->stream << "if (" << this->block_id_
                     << " >= " << PrintExpr(op->args[2]) << ") continue;\n";
      }
    }

    else if (op_name.find("gemm_v0") != std::string::npos) {
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

      auto src_type = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;
      auto dst_type = op->args[3].as<CallNode>()->args[0].as<CallNode>()->dtype;

      this->stream << op_name << "(" << a_name << "[" << a_offset << "], "
                   << b_name << "[" << b_offset << "], " << c_name << "["
                   << c_offset << "], ascend_l0a, ascend_l0b, "
                   << PrintExpr(op->args[4]) << ");\n";
    } else if (op_name.find("gemm_v1") != std::string::npos) {
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

      auto src_type = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;
      auto dst_type = op->args[3].as<CallNode>()->args[0].as<CallNode>()->dtype;

      this->stream << op_name << "(" << a_name << "[" << a_offset << "], "
                   << b_name << "[" << b_offset << "], " << c_name << "["
                   << c_offset << "], ascend_l0a, ascend_l0b, "
                   << PrintExpr(op->args[4]) << ");\n";
    }

  } else {
    tvm::Dump(op);
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenTileLangAscend::VisitStmt_(const AttrStmtNode *op) {
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
                   << " = AscendC::GetBlockIdx();\n";
      this->PrintIndent();
      this->stream << "if ASCEND_IS_AIV {\n";
      this->PrintIndent();
      this->PrintIndent();
      this->stream << current_block_id << " = " << current_block_id
                   << " / 2;\n";
      this->PrintIndent();
      this->stream << "}\n";

      this->core_num_ = PrintExpr(op->value);
    } else if (iv->thread_tag == "blockIdx.y" && iv->var->name_hint != "_") {
      auto vec_id_ = AllocVarID(iv->var.get());
      this->PrintIndent();
      this->stream << "auto " << vec_id_ << " = AscendC::GetSubBlockIdx();\n";
    }
    this->VisitStmt(op->body);
    return;
  } else if (op->attr_key == "init_flag" || op->attr_key == "clear_flag") {
    const StringImmNode *instn = op->value.as<StringImmNode>();

    std::string inst = std::string(instn->value);
    size_t st = 0;
    for (size_t i = 0; i < inst.size(); ++i) {
      if (inst[i] == '\n') {
        this->PrintIndent();
        stream << inst.substr(st, i - st) << "\n";
        st = i + 1;
      }
    }
    this->VisitStmt(op->body);
    return;
  } else if (op->attr_key == "resource_scope") {
    auto resource_id = Downcast<IntImm>(op->value)->value;
    auto resource_name = resource_id == 0 ? "AIC" : "AIV";

    this->PrintIndent();
    stream << "if ASCEND_IS_" << resource_name << " {\n";
    int func_scope = this->BeginScope();
    this->VisitStmt(op->body);
    this->EndScope(func_scope);
    this->PrintIndent();
    stream << "}\n";
    return;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangAscend::VisitStmt_(const AllocateNode *op) {
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());
  std::string scope = GetPtrStorageScope(op->buffer_var);
  std::string type = getType(op->dtype);
  const VarNode *buffer = op->buffer_var.as<VarNode>();

  auto print_buffer = [&](const std::string &pos) {
    this->PrintIndent();
    if (address_map_.find(op->buffer_var) != address_map_.end()) {
      stream << "auto " << vid << " = " << pos << ".GetWithOffset<" << type
             << ">(" << op->ConstantAllocationSize() << ", "
             << PrintExpr(address_map_[op->buffer_var]) << ");\n";
    } else {
      if (address_offset_.find(String(pos)) == address_offset_.end()) {
        address_offset_.Set(String(pos), 0);
      }
      stream << "auto " << vid << " = " << pos << ".GetWithOffset<" << type
             << ">(" << op->ConstantAllocationSize() << ","
             << PrintExpr(address_offset_[String(pos)]) << ");\n";
      address_offset_.Set(
          String(pos),
          PrimExpr(int(op->ConstantAllocationSize() * op->dtype.bytes())) +
              address_offset_[String(pos)]);
    }
  };

  if (scope == "wmma.matrix_a") {
    print_buffer("ascend_l0a");
  } else if (scope == "wmma.matrix_b") {
    print_buffer("ascend_l0b");
  } else if (scope == "wmma.accumulator") {
    print_buffer("ascend_l0c");
  } else if (scope == "shared.dyn") {
    print_buffer("ascend_l1");
  } else if (scope == "shared") {
    print_buffer("ascend_ub");
  }
  this->PrintStmt(op->body);
}

inline void PrintConst(const FloatImmNode *op, std::ostream &os,
                       CodeGenTileLangAscend *p) { // NOLINT(*)
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

void CodeGenTileLangAscend::VisitExpr_(const FloatImmNode *op,
                                       std::ostream &os) { // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenTileLangAscend::PreFunctionBody(const PrimFunc &f) {
  int func_scope = this->BeginScope();
  this->PrintIndent();
  stream << "AscendC::SetSyncBaseAddr(fftsAddr);\n";
  this->PrintIndent();
  stream << "AscendC::TPipe pipe;\n\n";
  ICHECK(this->para_.size() % 3 == 0)
      << "CodeGenTileLangAscend: parameters should be in pairs of (var, "
         "handle, dtype)";
  for (size_t i = 0; i < this->para_.size(); i += 3) {
    this->PrintIndent();
    stream << "AscendC::GlobalTensor<" << this->para_[i + 2] << "> "
           << this->para_[i + 1] << ";\n";
    this->PrintIndent();
    stream << this->para_[i + 1] << ".SetGlobalBuffer((__gm__ "
           << this->para_[i + 2] << "*)" << this->para_[i] << ");\n";
  }
  stream << "\n";

  this->PrintIndent();
  stream << "AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;\n";
  this->PrintIndent();
  stream << "pipe.InitBuffer(ascend_l0a, 65536);\n";
  this->PrintIndent();
  stream << "AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;\n";
  this->PrintIndent();
  stream << "pipe.InitBuffer(ascend_l0b, 131072);\n";

  this->PrintIndent();
  stream << "AscendC::TBuf<AscendC::TPosition::A1> ascend_l1; "
            "pipe.InitBuffer(ascend_l1, 524032);\n";
  this->PrintIndent();
  stream << "AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c; "
            "pipe.InitBuffer(ascend_l0c, 131072);\n";
  this->PrintIndent();
  stream << "AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; "
            "pipe.InitBuffer(ascend_ub, 196352);\n";

  this->PrintIndent();
  stream << "pipe.Destroy();\n";

  this->EndScope(func_scope);
}

void CodeGenTileLangAscend::VisitExpr_(const SelectNode *op, std::ostream &os) {
  auto condition = PrintExpr(op->condition);
  auto true_value = PrintExpr(op->true_value);
  auto false_value = PrintExpr(op->false_value);

  os << "(" << condition << " ? "
     << "" << true_value << " : " << false_value << ")";
}

void ProcessHostInput(std::ostream &os, std::vector<std::string> &arg_names,
                      std::vector<const tir::VarNode *> &shape_vars) {
  for (auto shape_var : shape_vars) {
    os << ", "
       << "int64_t " << shape_var->name_hint;
    arg_names.push_back(shape_var->name_hint);
  }
}

void CodeGenTileLangAscend::CallTilingInput(std::ostream &os, std::string func_name, std::vector<std::string> &tiling_args,
  std::vector<const tir::VarNode*> &shape_vars)
{
  for (auto &tiling_arg : tiling_args) {
    this->PrintIndent();
    os << "int64_t " << tiling_arg << ";\n";
  }
  this->PrintIndent();
  os << func_name << "_tiling(";
  size_t index = 0;
  for (auto shape_var : shape_vars) {
      os << shape_var->name_hint;
      if (index != shape_vars.size() - 1) {
        os << ", ";
      }
      index++;
  }
  index = 0;
  if (tiling_args.size() != 0) {
    os << ", ";
  }
  for (auto tiling_arg : tiling_args) {
      os << tiling_arg;
      if (index != tiling_args.size() - 1) {
        os << ", ";
      }
      index++;
  }
  os << ");\n";
}

void CodeGenTileLangAscend::ProcessTilingInput(std::ostream &os, std::string func_name, std::vector<std::string> &tiling_args,
  std::vector<const tir::VarNode*> &shape_vars)
{
  std::string name = "void " + func_name + "_tiling(";
  os << name;
  for (size_t i = 0; i < shape_vars.size(); ++i) {
      os << "int64_t " << shape_vars[i]->name_hint;
      if (i != shape_vars.size() - 1) {
        os << ", ";
      }
  }
  if (tiling_map_.size() != 0 && shape_vars.size() != 0) {
    os << ", ";
  }
  size_t index = 0;
  for (auto &pair : tiling_map_) {
      os << "int64_t &" << pair.first;
      if (index != tiling_map_.size() - 1) {
        os << ", ";
      }
      tiling_args.push_back(pair.first->name_hint);
      index++;
  }
  os << ") {\n";
  int func_scope = this->BeginScope();
  for (auto &pair : tiling_map_) {
    this->PrintIndent();
    os << pair.first << " = ";
    PrintExpr(arith::Analyzer().Simplify(pair.second), os);
    os << ";\n";
  }
  this->EndScope(func_scope);
  os << "}\n\n";
}

void CodeGenTileLangAscend::PrintHostFunc(const PrimFunc &f, const std::string &name,
                                          std::ostringstream &os, std::string &core,
                                          std::vector<const tir::VarNode *> &shape_vars) {
  // TODO: implement dynamic shape version
  std::vector<std::string> tiling_args;                
  std::string tiling_func_name = name;                
  ProcessTilingInput(os, tiling_func_name, tiling_args, shape_vars);
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
  ProcessHostInput(os, arg_names, shape_vars);
  os << ", aclrtStream stream) {\n  ";

  os << "uint32_t fftsLen{0};\n  ";
  os << "uint64_t fftsAddr{0};\n  ";
  os << "rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);\n";
  int func_scope = this->BeginScope();
  CallTilingInput(os, tiling_func_name, tiling_args, shape_vars);
  this->PrintIndent();

  os << name << "<<<" << core << ", nullptr, stream>>>(";
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
  os << ", fftsAddr);\n";
  os << "}\n";
  this->EndScope(func_scope);
  std::string content = os.str();
}

void CodeGenTileLangAscend::AddFunction(const GlobalVar &gvar,
                                        const PrimFunc &f) {
  // If the function has already been forward-declared, this is a
  // no-op.
  CodeGenC::DeclareFunction(gvar, f);
  // clear previous generated state.
  this->InitFuncState(f);

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);

  address_map_ = f->GetAttr<Map<Var, PrimExpr>>("address_map").value();

  use_swizzle_ = f->GetAttr<Bool>("use_swizzle").value();

  tiling_map_ = f->GetAttr<Map<Var, PrimExpr>>("tiling_map").value();
  ICHECK(global_symbol.defined())
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";
  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);

  this->PrintFuncPrefix(stream);
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
    stream << " " << vid;
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
  index = 0;
  if (tiling_map_.size() != 0 &&
     (shape_vars.size() != 0 || f->params.size() != 0)) {
      stream << ", ";
  }
  for (const auto &pair : tiling_map_) {
    stream << "int64_t " << GetVarID(pair.first.get());
    if (index != tiling_map_.size() - 1) {
        stream << ", ";
    }
    index++;
  }
  stream << ", uint64_t fftsAddr";
  stream << ") {\n";
  this->PreFunctionBody(f);
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";

  PrintHostFunc(f, func_name, stream, this->core_num_, shape_vars);
  std::string content = stream.str();
}

} // namespace codegen
} // namespace tvm
