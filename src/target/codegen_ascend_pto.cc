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
  } else if (type == "half") {
    typeSize = 2;
  } else if (type == "int8_t" || type == "uint8_t") {
    typeSize = 1;
  } else if (type == "int16_t" || type == "uint16_t") {
    typeSize = 2;
  } else if (type == "int" || type == "uint") {
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
  } else if (type == "half") {
    typeSize = "2";
  } else if (type == "int8_t" || type == "uint8_t") {
    typeSize = "1";
  } else if (type == "int16_t" || type == "uint16_t") {
    typeSize = "2";
  } else if (type == "int" || type == "uint") {
    typeSize = "4";
  } else {
    ICHECK(false) << "Unsupported datatype";
  }
  return typeSize;
}

CodeGenTileLangAscendPto::CodeGenTileLangAscendPto() {
  // restrict_keyword_ = "__gm__ uint8_t *";
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

int GetValidShape(int shape, std::string& dtype) {
  int dtype_len = GetTypeLen(dtype);
  int shape_mod = shape  * GetTypeLen(dtype) % 32;
  if (shape_mod == 0) {
      return shape;
  }
  return shape + (32 - shape_mod) / dtype_len;
}

void CodeGenTileLangAscendPto::UnaryVecOpCodegen(const CallNode *op, const std::string& op_name) {
  auto print_tile = [&](const CallNode *op) -> std::string {
    auto _var = op->args[1].as<VarNode>();
    auto _var_name = var_idmap_[_var];
    return _var_name;
  };

  std::vector<std::string> var_names;
  for (int i = 0; i < op->args.size(); i++) {
    auto var_name = print_tile(op->args[i].as<CallNode>());
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
  auto print_tile = [&](const CallNode *op) -> std::string {
    auto _var = op->args[1].as<VarNode>();
    auto _var_name = var_idmap_[_var];
    return _var_name;
  };

    this->PrintIndent();
    this->stream << op_name << "(" << print_tile(op->args[0].as<CallNode>()) << ", "
                  << print_tile(op->args[1].as<CallNode>()) << ", "
                  << PrintExpr(op->args[2]) << ");\n";
}

void CodeGenTileLangAscendPto::ReduceOpCodegen(const CallNode *op) {
  std::string op_name = Downcast<StringImm>(op->args[0])->value;
  if (op_name.find("reduce_sum") != std::string::npos) {
    op_name = "TROWSUM";
  } else if (op_name.find("reduce_max") != std::string::npos) {
    op_name = "TROWMAX";
  } else {
    ICHECK(false) << "not support reduce type: " << op_name;
  }

  auto print_tile = [&](const CallNode *op) -> std::string {
    auto _var = op->args[1].as<VarNode>();
    auto _var_name = var_idmap_[_var];
    return _var_name;
  };

  std::vector<std::string> var_names;
  for (int i = 1; i < op->args.size(); i++) {
    auto var_name = print_tile(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }
  std::string ub_name = var_names[0];
  std::vector<std::string> ub_data_vector = ub_data_map_[ub_name];
  std::string ub_data_type = ub_data_vector[0];
  std::string row = ub_data_vector[2];
  std::string col = ub_data_vector[1];
  std::string ffts = ub_data_vector[3];
  this->PrintIndent();
  this->stream << "tl::pto::TileUbDataDN <" << ub_data_type << ", " << row << ", " << col << "> " << ub_name << "_DN(" << row << ", " << col << ");\n";
  this->PrintIndent();
  this->stream << "TASSIGN(" << ub_name << "_DN, " << ffts << ");\n";
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
}

void CodeGenTileLangAscendPto::VisitExpr_(const CallNode *op, std::ostream &os) {
  auto print_tile = [&](const CallNode *op) -> std::string {
    auto _var = op->args[1].as<VarNode>();
    auto _var_name = var_idmap_[_var];
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
        if (api_name == "TLOAD") {
          ICHECK((copy_base_addr_map_.find(String(src_var_id)) != copy_base_addr_map_.end()));
          std::string tensor_addr = copy_base_addr_map_[String(src_var_id)];
          std::string tensor_template = "GlobalTensor<" + global_tensor_template[String(tensor_addr)].dtype + ", ";
          std::string shape_template = "Shape<", stride_template = "Stride<";
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
          shape_template += ">";
          // generate stride
          for (size_t i = 0; i < 4; i++) {
            if (len > 3 - i) {
              std::string tensor_template = global_tensor_template[String(tensor_addr)].shape_list[len + i - 4];
              if (tensor_template[0] < '1' || tensor_template[0] > '9')  stride_template += "-1, ";
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
          stride_template += "1>";

          // get gm2l1 shape
          if(op_name.find("copy_gm_to_l1") != std::string::npos
          || op_name.find("copy_gm_to_ub") != std::string::npos) {
            tensor_template = tensor_template + shape_template + ", " + stride_template + ">";
          }
          this->PrintIndent();
          this->stream << tensor_template << " " << src_var_id
          << "(" << tensor_addr << " + " << src_offset;
          if(global_tensor_template[String(tensor_addr)].shape_type=="dynamic") {
            this->stream << ", " << shape_template << "(), " << stride_template << "(";
            for(size_t i = 0; i < len; i++) {
              std::string tmp_shape_info = global_tensor_template[String(tensor_addr)].shape_list[i];
              if(tmp_shape_info[0]<'1' || tmp_shape_info[0]>'9') this->stream << tmp_shape_info;
              if(i<len-3) this->stream << ", ";
            }
            this->stream << ")";
          }
          this->stream << ");\n";
        } else if (api_name == "TCVT") {
          api_name = src_type == dst_type ? "TMOV":"TCVT";
        } else if (api_name == "TSTORE") {
          ICHECK((copy_base_addr_map_.find(String(dst_var_id)) != copy_base_addr_map_.end()));
          std::string tensor_addr = copy_base_addr_map_[String(dst_var_id)];
          std::string tensor_template = "GlobalTensor<" + global_tensor_template[String(tensor_addr)].dtype+ ", ";
          std::string shape_template = "Shape<", stride_template = "Stride<";
          size_t len = global_tensor_template[String(tensor_addr)].shape_list.size();
          size_t op_arg_len = op->args.size();
          size_t shape_size = 5;
          size_t shape_len = 2;
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
          shape_template += ">";
          // generate stride
          for (size_t i = 0; i < 4; i++) {
            if (len > 3 - i) {
              std::string tensor_template = global_tensor_template[String(tensor_addr)].shape_list[len + i - 4];
              if (tensor_template[0] < '1' || tensor_template[0] > '9')  stride_template += "-1, ";
              else {
                std::string tmp_shape = "";
                for(size_t j = 0; j < 4 - i; j++) {
                  tmp_shape += global_tensor_template[String(tensor_addr)].shape_list[len - j - 1];
                  if (j < 3 - i) tmp_shape += "*";
                }
                stride_template = stride_template +  tmp_shape + ", ";
              }
            } else {
              stride_template += "1, ";
            }
          }
          stride_template += "1>";

          if(op_name.find("copy_l0c_to_gm") != std::string::npos
          || op_name.find("copy_ub_to_gm") != std::string::npos) {
            tensor_template = tensor_template + shape_template + ", " + stride_template + ">";
          }
          this->PrintIndent();
          this->stream << tensor_template << " " << dst_var_id
          << "(" << tensor_addr << " + " << dst_offset;
          if(global_tensor_template[String(tensor_addr)].shape_type=="dynamic") {
            this->stream << ", " << shape_template << "(), " << stride_template << "(";
            for(size_t i = 0; i < len; i++) {
              std::string tmp_shape_info = global_tensor_template[String(tensor_addr)].shape_list[i];
              if(tmp_shape_info[0]<'1' || tmp_shape_info[0]>'9') this->stream << tmp_shape_info;
              if(i<len-3) this->stream << ", ";
            }
            this->stream << ")";
          }
          this->stream << ");\n";
        } else if (api_name == "TEXTRACT") {
          if (op->args.size() >= 3 && op->args[5].as<IntImmNode>()->value != 0) {
              row_index = op->args[4];
              col_index = op->args[3];
          } else {
              api_name = "TCVT";
          }
        }
        this->PrintIndent();
        this->stream << api_name << "(" << dst_var_id << ", "
          << src_var_id;
        if (api_name == "TEXTRACT") {
        this->stream << ", " << row_index << ", "
          << col_index;
        } else if (api_name == "TCVT") {
            this->stream << ", pto::RoundMode::CAST_NONE";
        }
        this->stream << ");\n";
      } else {
        this->PrintIndent();
        this->stream << "not implemented yet\n";
      }
    } 
  } else if (op->op.same_as(tl::ascend_gemm_v0())) {
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
    std::string data_type_input = params["data_type_input"];
    this->stream << "tl::pto::gemm_v0" << "<" <<  params["data_type_input"] << ", " << params["data_type_output"] << ", " 
    << GetValidShape(std::stoi(params["M"]), data_type_input) << ", " 
    << GetValidShape(std::stoi(params["N"]), data_type_input) << ", "
    << GetValidShape(std::stoi(params["K"]), data_type_input) << ", "
    << params["M"] << ", " << params["N"] << ", " << params["K"] << ", "
    << params["transpose_A"] << ", " << params["transpose_B"] << ">" 
    << "(" << a_name << ", " << b_name << ", " << c_name << ", " << PrintExpr(op->args[4]) << ");\n";
  } else if (op->op.same_as(tl::ascend_pipe_barrier())) {
      std::string pipe = Downcast<StringImm>(op->args[0])->value;
      this->PrintIndent();
      this->stream << "pipe_barrier(PIPE_" << pipe << ");\n";
  } else if (op->op.same_as(tl::ascend_wait_flag())) {
      std::string src = Downcast<StringImm>(op->args[0])->value;
      std::string dst = Downcast<StringImm>(op->args[1])->value;
      std::string event_id = PrintExpr(op->args[2]);
      // wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      this->PrintIndent();
      this->stream << "wait_flag(PIPE_" << src << ", " << "PIPE_" <<
      dst << ", " << "EVENT_ID" << event_id << ");\n";
  } else if (op->op.same_as(tl::ascend_set_flag())) {
      std::string src = Downcast<StringImm>(op->args[0])->value;
      std::string dst = Downcast<StringImm>(op->args[1])->value;
      std::string event_id = PrintExpr(op->args[2]);
      // set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      this->PrintIndent();
      this->stream << "set_flag(PIPE_" << src << ", " << "PIPE_" <<
      dst << ", " << "EVENT_ID" << event_id << ");\n";
  } else if (op->op.same_as(tl::ascend_set_cross_flag())) {
    std::string pipe = Downcast<StringImm>(op->args[0])->value;
    int flag = std::stoi(PrintExpr(op->args[1]));
    int mode = 2;
    int config = 1 | (mode << 4) | (flag << 8);
    this->PrintIndent();
    this->stream << "ffts_cross_core_sync" << "(" << "PIPE_" << pipe << ", " << config << ");\n";
  } else if (op->op.same_as(tl::ascend_wait_cross_flag())) {
    std::string flag = PrintExpr(op->args[0]);
    this->PrintIndent();
    this->stream << "wait_flag_dev" << "(" << flag << ");\n";
  } else if (op->op.same_as(tl::ascend_fill())) {
    this->PrintIndent();
    this->stream << "TEXPANDS" << "(" << print_tile(op->args[1].as<CallNode>()) << ", "
               << PrintExpr(op->args[2]) << ");\n";
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
}

std::string CodeGenTileLangAscendPto::PrintBufferOffset(const CallNode *op) {
    auto _var = op->args[1].as<VarNode>();
    std::string _var_name = var_idmap_[_var];
    return _var_name;
}

void CodeGenTileLangAscendPto::BinaryVecOpCodegen(const CallNode *op,
                                               const std::string &op_name) {
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
    std::string scalar_name = var_name + "_scalar";
    this->stream << "auto " << scalar_name <<  "= " << var_name
                << ".GetValue(" << index
                << ");\n";
    std::vector<std::string> ub_data_vector = ub_data_map_[ub_name];
    std::string var_name_temp = ub_name + "_temp";
    std::string ub_data_type = ub_data_vector[0];
    this->PrintIndent();
    int32_t ub_data_temp_col = std::stoi(ub_data_vector[2]) * std::stoi(ub_data_vector[1]) / std::stoi(for_num_map_[index]);
    this->stream << "tl::pto::TileUbDataND<" << ub_data_vector[0] << ", 1, " 
    << ub_data_temp_col << "> " << var_name_temp << "(" << "1, " << ub_data_temp_col << ");\n";
    this->PrintIndent();
    this->stream << "TASSIGN(" << var_name_temp << ", " << ub_data_vector[3] << " + " <<
    index << " * " << ub_data_temp_col << " * " << GetTypeLenString(ub_data_vector[0]) << ");\n";
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
    std::string scalar = PrintExpr(op->args[op->args.size() - 1]);
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
      // this->PrintIndent();
      // this->stream << "if ASCEND_IS_AIV {\n";
      // this->PrintIndent();
      // this->PrintIndent();
      // this->stream << current_block_id << " = " << current_block_id
      //              << " / 2;\n";
      // this->PrintIndent();
      // this->stream << "}\n";

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

    stream << "#if defined(__DAV_C220_" << resource_name << "__)\n";
    if (resource_name == "VEC") {
      this->PrintIndent();
      stream << "  set_mask_norm();\n";
      this->PrintIndent();
      stream << "  set_vector_mask(-1, -1);\n";
    }
    int func_scope = this->BeginScope();
    this->VisitStmt(op->body);
    this->EndScope(func_scope);
    stream << "#endif\n";
    return;
  }
  CodeGenC::VisitStmt_(op);
}

void UbShapeInputCheck(const AllocateNode *op) {
  if (op->extents.size() > 2 || op->extents.size() == 0){
    ICHECK(false) << "Unsupported ubsize which is expected to be 1 or 2";
  }
}

bool ValidLayoutEnabled(const AllocateNode *op) {
  bool valid = false;
  std::string type = getType(op->dtype);
  int8_t typeSize = GetTypeLen(type);
  if (tvm::tir::is_zero(tvm::truncmod(op->extents[1] * typeSize, 32))) {
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
  
  /// Allocate PTO Tile Memory Address
  auto print_buffer = [&](const std::string &pos) {
    std::vector<std::string> ub_data(4);
    ub_data[0] = type;
    if (pos == "tl::pto::TileUbData") {
      UbShapeInputCheck(op);
    }
    this->PrintIndent();
    // Allocate buffer
    if (address_map_.find(op->buffer_var) != address_map_.end()) {
      if (pos == "tl::pto::TileUbData") {
        if (op->extents.size() == 2) {
          ub_data[1] = PrintExpr(op->extents[0]);
          ub_data[2] = PrintExpr(op->extents[1]);
          auto valid = ValidLayoutEnabled(op);
          if (!valid) {
            stream << pos << "ND<" << type;
            for (size_t i = 0; i < op->extents.size(); i++) {
                stream << ", " << op->extents[i];
            }
          } else {
            int8_t typeSize = GetTypeLen(type);
            int8_t NDBlockSize = 32 / typeSize;
            stream << pos << "ND<" << type;
            stream << ", " << op->extents[0];
            stream << ", " << tvm::floordiv(op->extents[1] + NDBlockSize - 1, NDBlockSize) * NDBlockSize;
          }
          stream << "> " << vid << "(";
          for (size_t i = 0; i < op->extents.size(); i++) {
              if (i < op->extents.size() - 1) {
                  stream << op->extents[i] << ", ";
              } else {
                  stream << op->extents[i] << ");\n";
              }
          }
        } else if (op->extents.size() == 1) {
          ub_data[1] = "1";
          ub_data[2] = PrintExpr(op->extents[0]);
          stream << pos << "ND<" << type << ", 1, " << op->extents[0] << "> " << vid << "(" << "1, " << op->extents[0] << ");\n";
        }
      } else {
        int dtype_bytes = op->dtype.bytes();
        std::vector<PrimExpr> valid_shapes;
        valid_shapes.reserve(op->extents.size());
        stream << pos << "<" << type;
        int shape_value = op->extents[0].as<tvm::tir::IntImmNode>()->value;
        if (shape_value * dtype_bytes % 32 == 0) {
            valid_shapes.push_back(op->extents[0]);
        } else {
            valid_shapes.push_back(tvm::IntImm(op->extents[0].dtype(), 
            shape_value + (32 - shape_value * dtype_bytes % 32) / dtype_bytes));
        }
        valid_shapes.push_back(op->extents[1]);
        for (size_t i = 0; i < valid_shapes.size(); i++) {
            stream << ", " << valid_shapes[i];
        }
        for (size_t i = 0; i < op->extents.size(); i++) {
            stream << ", " << op->extents[i];
        }
        stream << "> " << vid << ";\n";
      }
      // Allocate Start Address
      this->PrintIndent();
      ub_data[3] = DEC_STR_TO_HEX_STR(PrintExpr(address_map_[op->buffer_var]));
      ub_data_map_[vid] = ub_data;
      stream << "TASSIGN(" << vid << ", " << DEC_STR_TO_HEX_STR(PrintExpr(address_map_[op->buffer_var])) << ");\n";

    } else {
      if (address_offset_.find(String(pos)) == address_offset_.end()) {
        address_offset_.Set(String(pos), 0);
      }
      if (pos == "tl::pto::TileUbData") {
        if (op->extents.size() == 2) {
          ub_data[1] = PrintExpr(op->extents[0]);
          ub_data[2] = PrintExpr(op->extents[1]);
          auto valid = ValidLayoutEnabled(op);
          if (!valid) {
            stream << pos << "ND<" << type;
            for (size_t i = 0; i < op->extents.size(); i++) {
                stream << ", " << op->extents[i];
            }
          } else {
            int8_t typeSize = GetTypeLen(type);
            int8_t NDBlockSize = 32 / typeSize;
            stream << pos << "ND<" << type;
            stream << ", " << op->extents[0];
            stream << ", " << tvm::floordiv(op->extents[1] + NDBlockSize - 1, NDBlockSize) * NDBlockSize;
          }
          stream << "> " << vid << "(";
          for (size_t i = 0; i < op->extents.size(); i++) {
              if (i < op->extents.size() - 1) {
                  stream << op->extents[i] << ", ";
              } else {
                  stream << op->extents[i] << ");\n";
              }
          }
        } else if (op->extents.size() == 1) {
          ub_data[1] = "1";
          ub_data[2] = PrintExpr(op->extents[0]);
          stream << pos << "ND<" << type << ", 1, " << op->extents[0] << "> " << vid << "(" << "1, " << op->extents[0] << ");\n";
        }
      } else {
        int dtype_bytes = op->dtype.bytes();
        std::vector<PrimExpr> valid_shapes;
        valid_shapes.reserve(op->extents.size());
        stream << pos << "<" << type;
         int shape_value = op->extents[0].as<tvm::tir::IntImmNode>()->value;
        if (shape_value * dtype_bytes % 32 == 0) {
            valid_shapes.push_back(op->extents[0]);
        } else {
            valid_shapes.push_back(tvm::IntImm(op->extents[0].dtype(), 
            shape_value + (32 - shape_value * dtype_bytes % 32) / dtype_bytes));
        }
        valid_shapes.push_back(op->extents[1]);
        for (size_t i = 0; i < valid_shapes.size(); i++) {
            stream << ", " << valid_shapes[i];
        }
        for (size_t i = 0; i < op->extents.size(); i++) {
            stream << ", " << op->extents[i];
        }
        stream << "> " << vid << ";\n";
      }
      // Allocate Start Address
      this->PrintIndent();
      ub_data[3] = DEC_STR_TO_HEX_STR(PrintExpr(address_offset_[String(pos)]));
      ub_data_map_[vid] = ub_data;
      stream << "TASSIGN(" << vid << ", " << DEC_STR_TO_HEX_STR(PrintExpr(address_offset_[String(pos)])) << ");\n";
      address_offset_.Set(
          String(pos),
          PrimExpr(int(op->ConstantAllocationSize() * op->dtype.bytes())) +
              address_offset_[String(pos)]);
    }
  };

  if (scope == "wmma.matrix_a") {
    print_buffer("TileLeft");
  } else if (scope == "wmma.matrix_b") {
    print_buffer("TileRight");
  } else if (scope == "wmma.accumulator") {
    print_buffer("TileAcc");
  } else if (scope == "shared.dyn") {
    print_buffer("tl::pto::TileMatL1");
  } else if (scope == "shared") {
    print_buffer("tl::pto::TileUbData");
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
    // this->PrintIndent();
    // std::string shape0 = this->para_[i + 3], shape1 = this->para_[i + 4];
    // std::string copy_tmplte = "GlobalTensor<" + this->para_[i + 2] + ", Shape<1, 1, 1, " + 
    // shape0 + ", " + shape1 + ">, Stride<" +  "1, 1, " + 
    // shape0 + " * " + shape1 + ", " + shape1 + ", 1>>";
    // stream << copy_tmplte << " " << this->para_[i + 1] << "(" << this->para_[i] << ");\n";
    // this->PrintIndent();
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

} // namespace codegen
} // namespace tvm
