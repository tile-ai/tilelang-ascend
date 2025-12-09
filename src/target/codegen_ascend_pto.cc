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
  std::string scope = GetPtrStorageScope(op->buffer->data);
  std::cout << "check buf load:" << op->buffer->data.get()->name_hint << "\n";
  if (scope == "wmma.matrix_a" || scope == "wmma.matrix_b" ||
      scope == "wmma.accumulator" || scope == "shared.dyn" ||
      scope == "shared") {
    bool do_not_sup_get_from_tile = false;
    ICHECK(do_not_sup_get_from_tile); // Currently not supported
  } else {
    for (size_t i = 0; i < this->para_.size(); i += 3) {
      std::cout << "check gm var:" << this->para_[i]
      << this->para_[i + 1] << this->para_[i + 2] << "\n";
      if (this->para_[i + 1] == op->buffer->data.get()->name_hint) {
        os << "*(" << this->para_[i] << " + "
                   << PrintExpr(op->indices.back()) << ")";
        break;
      }
    }
  }
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
        {"copy_ub_to_ub", "TMOV"}
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
          size_t op_arg_len = op->args.size();
          // Dynamic Shape and Static Shape

          // generate shape
          for (size_t i = 0; i < 5; i++) {
            // if op->args.size() changes with GM shapes, add one more len.
            if (4 - i < op_arg_len - 4) shape_template += PrintExpr(op->args[i + len - 1]);
            else shape_template += "1";
            if (i < 4) shape_template += ", ";
          }
          shape_template += ">";
          // generate stride
          for (size_t i = 0; i < 4; i++) {
            if (len > 3 - i) {
              if (global_tensor_template[String(tensor_addr)].shape_type=="dynamic") stride_template += "-1, ";
              else {
                std::string tmp_shape = "";
                for(size_t j = 0; j < 4 - i; j++) {
                  tmp_shape += global_tensor_template[String(tensor_addr)].shape_list[j];
                  if (j < 3 - i) tmp_shape += "*";
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
              if(i<len-1) this->stream << ", ";
            }
            this->stream << ")";
          }
          this->stream << ");\n";
        } else if (api_name == "TSTORE") {
          ICHECK((copy_base_addr_map_.find(String(dst_var_id)) != copy_base_addr_map_.end()));
          std::string tensor_addr = copy_base_addr_map_[String(dst_var_id)];
          std::string tensor_template = "GlobalTensor<" + global_tensor_template[String(tensor_addr)].dtype+ ", ";
          std::string shape_template = "Shape<", stride_template = "Stride<";
          size_t len = global_tensor_template[String(tensor_addr)].shape_list.size();
          size_t op_arg_len = op->args.size();
          // generate shape
          for (size_t i = 0; i < 5; i++) {
            if (4 - i < op_arg_len - 4) shape_template += PrintExpr(op->args[i + len - 1]);
            else shape_template += "1";
            if (i < 4) shape_template += ", ";
          }
          shape_template += ">";
          // generate stride
          for (size_t i = 0; i < 4; i++) {
            if (len > 3 - i) {
              if (global_tensor_template[String(tensor_addr)].shape_type=="dynamic") stride_template += "-1, ";
              else {
                std::string tmp_shape = "";
                for(size_t j = 0; j < 4 - i; j++) {
                  tmp_shape += global_tensor_template[String(tensor_addr)].shape_list[j];
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
              if(i<len-1) this->stream << ", ";
            }
            this->stream << ")";
          }
          this->stream << ");\n";
        } else if (api_name == "TEXTRACT") {
          if (op->args.size() >= 3 && op->args[5].as<IntImmNode>()->value != 0) {
              row_index = op->args[4];
              col_index = op->args[3];
          } else {
              api_name = "TMOV";
          }
        }
        this->PrintIndent();
        this->stream << api_name << "(" << dst_var_id << ", "
          << src_var_id;
        if (api_name == "TEXTRACT") {
        this->stream << ", " << row_index << ", "
          << col_index;
        }
        this->stream << ");\n";
      } else {
        this->PrintIndent();
        this->stream << "not implemented yet\n";
      }
    } else if(op_name.find("pipe_barrier") != std::string::npos) {
      this->PrintIndent();
      this->stream << op_name << "\n";
    } else if(op_name.find("gemm_v0") != std::string::npos) {
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

      for(size_t pos=0;(pos=op_name.find("tl::ascend",pos))!=std::string::npos;pos+=5) {op_name.replace(pos,10,"tl::pto");}
      this->stream << op_name << "(" << a_name << ", " << b_name << ", " 
      << c_name << ", " << PrintExpr(op->args[4]) << ");\n";
    }
  } else if (op->op.same_as(tl::ascend_fill())) {
    this->PrintIndent();
    this->stream << op->args[0].as<StringImmNode>()->value << "(" << print_tile(op->args[1].as<CallNode>()) << ", "
               << PrintExpr(op->args[2]) << ");\n";
  } else if (op->op.same_as(tl::ascend_reduce())) {
    this->PrintIndent();
    this->stream << op->args[0].as<StringImmNode>()->value << "(" << print_tile(op->args[1].as<CallNode>()) << ", "
               << print_tile(op->args[2].as<CallNode>()) << ");\n";
  } else if (op->op.same_as(tl::ascend_scalar_op())) {
    this->PrintIndent();
    this->stream << op->args[0].as<StringImmNode>()->value << "(" << print_tile(op->args[1].as<CallNode>()) << ", "
                  << print_tile(op->args[2].as<CallNode>()) << ", "
                  << PrintExpr(op->args[3]) << ");\n";
  } else if (op->op.same_as(tl::ascend_unary_op())) {
    this->PrintIndent();
    this->stream << op->args[0].as<StringImmNode>()->value << "(" << print_tile(op->args[1].as<CallNode>()) << ", "
               << print_tile(op->args[2].as<CallNode>()) << ");\n";
  } else if (op->op.same_as(tl::ascend_binary_op())) {
    this->PrintIndent();
    this->stream << op->args[0].as<StringImmNode>()->value << "(" << print_tile(op->args[1].as<CallNode>()) << ", "
               << print_tile(op->args[2].as<CallNode>()) << ", "
               << print_tile(op->args[3].as<CallNode>()) << ");\n";
  } else if (op->op.same_as(tl::ascend_binary_ops())) {
    this->PrintIndent();
    this->stream << op->args[0].as<StringImmNode>()->value << "(" << print_tile(op->args[1].as<CallNode>()) << ", "
               << print_tile(op->args[2].as<CallNode>()) << ", "
               << PrintExpr(op->args[3]) << ");\n";
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
                   << " = get_block_idx();\n\n";
      // this->PrintIndent();
      // this->stream << "if ASCEND_IS_AIV {\n";
      // this->PrintIndent();
      // this->PrintIndent();
      // this->stream << current_block_id << " = " << current_block_id
      //              << " / 2;\n";
      // this->PrintIndent();
      // this->stream << "}\n";

      this->core_num_ = PrintExpr(op->value);
    }
    this->VisitStmt(op->body);
    return;
  } else if (op->attr_key == "resource_scope") { // other core
    // auto resource_id = Downcast<IntImm>(op->value)->value;
    // auto resource_name = resource_id == 0 ? "AIC" : "AIV";

    // this->PrintIndent();
    // stream << "if ASCEND_IS_" << resource_name << " {\n";
    int func_scope = this->BeginScope();
    this->VisitStmt(op->body);
    this->EndScope(func_scope);
    // this->PrintIndent();
    // stream << "}\n";
    return;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangAscendPto::VisitStmt_(const AllocateNode *op) {
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get()); // var_name
  std::string scope = GetPtrStorageScope(op->buffer_var);
  std::string type = getType(op->dtype);
  const VarNode *buffer = op->buffer_var.as<VarNode>();
  
  /// Allocate PTO Tile Memory Address
  auto print_buffer = [&](const std::string &pos) {
    this->PrintIndent();
    // Allocate buffer
    if (address_map_.find(op->buffer_var) != address_map_.end()) {
      stream << pos << "<" << type << ", " << op->extents[0] 
      << ", " << op->extents[1] << "> " << vid << ";\n";
      // Allocate Start Address
      this->PrintIndent();
      stream << "TASSIGN(" << vid << ", " << DEC_STR_TO_HEX_STR(PrintExpr(address_map_[op->buffer_var])) << ");\n";

    } else {
      if (address_offset_.find(String(pos)) == address_offset_.end()) {
        address_offset_.Set(String(pos), 0);
      }
      stream << pos << "<" << type << ", " << op->extents[0] 
      << ", " << op->extents[1] << "> " << vid << ";\n";
      this->PrintIndent();
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
  os << "extern \"C\" __global__ __aicore__ void launch_kernel(";
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
  os << ")\n{\n  ";
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
  os << ");\n}\n\n";


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
  this->PrintIndent();

  os << "launch_kernel" << "<<<" << core << ", nullptr, stream>>>(";
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
  os << ");\n}\n";
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
  this->stream << "__aicore__ ";
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
