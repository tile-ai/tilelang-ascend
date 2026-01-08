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

#include <tvm/tir/expr.h>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "../op/ascend.h"
#include "../op/builtin.h"

#include "arith/pattern_match.h"

namespace tvm {
namespace codegen {

std::string getType(const DataType &dtype) {
  if (dtype.is_float16()) {
    return "half";
  } else if (dtype.is_float()) {
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
  LOG(FATAL) << "Unsupported data type: " << dtype;
  return "";
}

CodeGenTileLangAscend::CodeGenTileLangAscend() {
  restrict_keyword_ = "GM_ADDR";
}

void CodeGenTileLangAscend::PrintFuncPrefix(std::ostream &os) {
  os << "extern \"C\" __global__ __aicore__ ";
}

std::string CodeGenTileLangAscend::Finish() {
  decl_stream << "#include \"tl_templates/ascend/common.h\"\n";
  decl_stream << "#include \"acl/acl.h\"\n";
  decl_stream << "#include <runtime/rt_ffts.h>\n";
  decl_stream << "using namespace Catlass;\n";
  decl_stream << "using uint = unsigned int;\n";
  decl_stream << "using uchar = unsigned char;\n";
  decl_stream << "using ushort = unsigned short;\n";
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
  auto var_name = var_idmap_[op->buffer->data.get()];
  os << var_name << ".GetValue(" << PrintExpr(op->indices.back()) << ")";
}

void CodeGenTileLangAscend::VisitStmt_(const BufferStoreNode *op) {
  auto var_name = var_idmap_[op->buffer->data.get()];
  this->PrintIndent();
  this->stream << var_name << ".SetValue(" << PrintExpr(op->indices.back())
               << ", " << PrintExpr(op->value) << ");\n";
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
  auto add_decl_stream = [&](std::ostringstream &ss, const std::string &str) {
    std::string content = ss.str();
    if (content.find(str) == std::string::npos) {
      ss << str;
    }
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
        {"copy_l1_to_l0a", 2},
        {"copy_l1_to_l0b", 2},
        {"copy_gm_to_ub", 1},
        {"copy_ub_to_gm", 1},
        {"copy_ub_to_ub", 0}
      };

      bool found = false;
      int extra_args = 0;

      for (const auto &pair : kCopyOpExtraArgs) {
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
                   << c_offset << "], " << PrintExpr(op->args[4]) << ", "
                   << PrintExpr(op->args[5]) << ");\n";
    } else if (op_name.find("thread_block_swizzle") != std::string::npos) {
      std::string expr = PrintExpr(op->args[1]);
      os << op_name << "(" << expr << ")";
    }

    if (op_name == "AscendC::AutoBarrier") {
      this->PrintIndent();
      std::string pipeline = "PIPE_ALL";
      if (op->args.size() > 1) {
        if (auto pipeline_imm = op->args[1].as<StringImmNode>()) {
          pipeline = pipeline_imm->value;
        }
      }
      this->stream << "AscendC::PipeBarrier<" << pipeline << ">();\n";
      return;
    } else if (op_name == "AscendC::AutoSetFlag") {
      this->PrintIndent();
      auto event_type = Downcast<StringImm>(op->args[1])->value;
      auto event_id = PrintExpr(op->args[2]);
      this->stream << "AscendC::SetFlag<AscendC::HardEvent::" << event_type
                   << ">(" << event_id << ");\n";
      return;
    } else if (op_name == "AscendC::AutoWaitFlag") {
      this->PrintIndent();
      auto event_type = Downcast<StringImm>(op->args[1])->value;
      auto event_id = PrintExpr(op->args[2]);
      this->stream << "AscendC::WaitFlag<AscendC::HardEvent::" << event_type
                   << ">(" << event_id << ");\n";
      return;
    }

    // For AutoCrossCoreSetFlag and AutoCrossCoreWaitFlag, we use the op_name
    // with adding "Auto" prefix.
    if (op_name == "AscendC::AutoCrossCoreSetFlag") {
      this->PrintIndent();
      auto model_id = op->args[1].as<IntImmNode>()->value;
      auto pipe = op->args[2].as<StringImmNode>()->value;
      auto flag_id = op->args[3].as<IntImmNode>()->value;
      this->stream << "AscendC::CrossCoreSetFlag<" << model_id << ", PIPE_"
                   << pipe << ">(" << flag_id << ");\n";
      return;
    } else if (op_name == "AscendC::AutoCrossCoreWaitFlag") {
      this->PrintIndent();
      auto flag_id = op->args[1].as<IntImmNode>()->value;
      this->stream << "AscendC::CrossCoreWaitFlag(" << flag_id << ");\n";
      return;
    }

  } else if (op->op.same_as(tl::loop_break())) {
    this->PrintIndent();
    this->stream << "break;\n";
  } else if (op->op.same_as(tl::ascend_add())) {
    BinaryVecOpCodegen(op, "AscendC::Add");
  } else if (op->op.same_as(tl::ascend_sub())) {
    BinaryVecOpCodegen(op, "AscendC::Sub");
  } else if (op->op.same_as(tl::ascend_mul())) {
    BinaryVecOpCodegen(op, "AscendC::Mul");
  } else if (op->op.same_as(tl::ascend_div())) {
    BinaryVecOpCodegen(op, "AscendC::Div");
  } else if (op->op.same_as(tl::ascend_max())) {
    BinaryVecOpCodegen(op, "AscendC::Max");
  } else if (op->op.same_as(tl::ascend_min())) {
    BinaryVecOpCodegen(op, "AscendC::Min");
  } else if (op->op.same_as(tl::ascend_bitwise_and())) {
    BinaryVecOpCodegen(op, "AscendC::And");
  } else if (op->op.same_as(tl::ascend_bitwise_or())) {
    BinaryVecOpCodegen(op, "AscendC::Or");
  } else if (op->op.same_as(tl::ascend_adds())) {
    AddsAndMulsOpCodegen(op, "AscendC::Adds");
  } else if (op->op.same_as(tl::ascend_subs())) {
    SubsOpCodegen(op);
  } else if (op->op.same_as(tl::ascend_muls())) {
    AddsAndMulsOpCodegen(op, "AscendC::Muls");
  } else if (op->op.same_as(tl::ascend_divs())) {
    DivsOpCodegen(op);
  } else if (op->op.same_as(tl::ascend_sort32())) {
    Sort32Codegen(op, "AscendC::Sort32");
  } else if (op->op.same_as(tl::ascend_compare())) {
    CompareCodegen(op, "AscendC::Compare");
  } else if (op->op.same_as(tl::ascend_compare_scalar())) {
    CompareScalarCodegen(op, "AscendC::CompareScalar");
  } else if (op->op.same_as(tl::ascend_gather())) {
    GatherCodegen(op, "AscendC::Gather");
  } else if (op->op.same_as(tl::ascend_reduce())) {
    ReduceOpCodegen(op);
  } else if (op->op.same_as(tl::ascend_block_reduce_max())) {
    BlockReduceOpCodegen(op, "AscendC::BlockReduceMax");
  } else if (op->op.same_as(tl::ascend_block_reduce_min())) {
    BlockReduceOpCodegen(op, "AscendC::BlockReduceMin");
  } else if (op->op.same_as(tl::ascend_block_reduce_sum())) {
    BlockReduceOpCodegen(op, "AscendC::BlockReduceSum");
  } else if (op->op.same_as(tl::ascend_cast())) {
    CastCodegen(op, "AscendC::Cast");
  } else if (op->op.same_as(tl::ascend_set_deq_scale())) {
    SetDeqScaleCodegen(op, "AscendC::SetDeqScale");
  } else if (op->op.same_as(tl::ascend_exp())) {
    UnaryVecOpCodegen(op, "AscendC::Exp");
  } else if (op->op.same_as(tl::ascend_ln())) {
    UnaryVecOpCodegen(op, "AscendC::Ln");
  } else if (op->op.same_as(tl::ascend_abs())) {
    UnaryVecOpCodegen(op, "AscendC::Abs");
  } else if (op->op.same_as(tl::ascend_reciprocal())) {
    UnaryVecOpCodegen(op, "AscendC::Reciprocal");
  } else if (op->op.same_as(tl::ascend_sqrt())) {
    UnaryVecOpCodegen(op, "AscendC::Sqrt");
  } else if (op->op.same_as(tl::ascend_rsqrt())) {
    UnaryVecOpCodegen(op, "AscendC::Rsqrt");
  } else if (op->op.same_as(tl::ascend_relu())) {
    UnaryVecOpCodegen(op, "AscendC::Relu");
  } else if (op->op.same_as(tl::ascend_bitwise_not())) {
    UnaryVecOpCodegen(op, "AscendC::Not");
  } else if (op->op.same_as(tl::ascend_leaky_relu())) {
    ScalarOpCodegen(op, "AscendC::LeakyRelu");
  } else if (op->op.same_as(tl::ascend_axpy())) {
    ScalarOpCodegen(op, "AscendC::Axpy");
  } else if (op->op.same_as(tl::ascend_bitwise_lshift())) {
    ShiftOpCodegen(op, "AscendC::ShiftLeft");
  } else if (op->op.same_as(tl::ascend_bitwise_rshift())) {
    ShiftOpCodegen(op, "AscendC::ShiftRight");
  } else if (op->op.same_as(tl::ascend_sin())) {
    TrigOpCodegen(op, "AscendC::Sin");
  } else if (op->op.same_as(tl::ascend_cos())) {
    TrigOpCodegen(op, "AscendC::Cos");
  } else if (op->op.same_as(tl::ascend_transpose())) {
    TransposeCodegen(op, "AscendC::Transpose");
  } else if (op->op.same_as(tl::ascend_createvecindex())) {
    CreateVecIndexCodegen(op, "AscendC::CreateVecIndex");
  } else if (op->op.same_as(tl::ascend_fill())) {
    FillCodegen(op);
  } else if (op->op.same_as(tl::ascend_arith_progression())) {
    ArithProgressionCodegen(op);
  } else if (op->op.same_as(tl::ascend_sort())) {
    SortCodegen(op);
  } else if (op->op.same_as(tl::ascend_merge_sort())) {
    MergeSortCodegen(op);
  } else if (op->op.same_as(tl::ascend_topk())) {
    TopKCodegen(op);
  } else if (op->op.same_as(tl::ascend_gather_mask())) {
    GatherMaskCodegen(op);
  } else if (op->op.same_as(tl::ascend_gatherb())) {
    GatherbCodegen(op);
  } else if (op->op.same_as(tl::ascend_select())) {
    SelectCodegen(op, "AscendC::Select");
  } else if (op->op.same_as(tl::ascend_init_sort_buf())) {
    InitSortBufCodegen(op);
  } else if (op->op.same_as(tl::ascend_pow())) {
    PowerOpCodegen(op, "AscendC::Power");
  } else if (op->op.same_as(tl::ascend_bitwise_xor())) {
    PrintOpCall(op, "AscendC::Xor", {0, op->args.size()}, {0, 0});
  } else if (op->op.same_as(tl::ascend_broadcast())) {
    BroadcastOpCodegen(op);
  } else if (op->op.same_as(tl::ascend_wait_cross_flag())) {
    PrintOpCall(op, "AscendC::CrossCoreWaitFlag", {0, 0}, {0, op->args.size()});
  } else if (op->op.same_as(tl::ascend_set_cross_flag())) {
    SetCrossFlagCodegen(op);
  } else if (op->op.same_as(tl::ascend_wait_flag())) {
    FlagOpCodegen(op, "AscendC::WaitFlag");
  } else if (op->op.same_as(tl::ascend_set_flag())) {
    FlagOpCodegen(op, "AscendC::SetFlag");
  } else if (op->op.same_as(tl::ascend_pipe_barrier())) {
    PipeBarrierCodegen(op);
  } else if (op->op.same_as(tl::ascend_sync_all())) {
    PrintOpCall(op, "AscendC::SyncAll<false>", {0, 0}, {0, 0});
  } else if (op->op.same_as(tl::ascend_gemm_v0())) {
    GemmOpCodegen(op);
  } else if (op->op.same_as(tl::ascend_gemm_v1())) {
    GemmOpCodegen(op);
  } else if (op->op.same_as(tl::ascend_printf())) {
    PrintfOpCodegen(op, "AscendC::PRINTF");
  } else if (op->op.same_as(tl::ascend_dump_tensor())) {
    DumpTensorCodegen(op);
  } else if (op->op.same_as(tl::ascend_bilinear_interpolation())) {
    BilinearInterpolationCodegen(op);
  } else if (op->op.same_as(tl::ascend_wholereducemax())) {
    WholeReduceOpCodegen(op, "AscendC::WholeReduceMax");
  } else if (op->op.same_as(tl::ascend_wholereducemin())) {
    WholeReduceOpCodegen(op, "AscendC::WholeReduceMin");
  } else if (op->op.same_as(tl::ascend_wholereducesum())) {
    PrintOpCall(op, "AscendC::WholeReduceSum", {0, 2}, {2, op->args.size()});
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

    if (found_by_name) {
      stream << "auto " << vid << " = " << pos << ".GetWithOffset<" << type
             << ">(" << op->ConstantAllocationSize() << ", "
             << PrintExpr(target_expr) << ");\n";
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
  stream << "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);\n";
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

void CodeGenTileLangAscend::CallTilingInput(
    std::ostream &os, std::string func_name,
    std::vector<std::string> &tiling_args,
    std::vector<const tir::VarNode *> &shape_vars) {
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

void CodeGenTileLangAscend::ProcessTilingInput(
    std::ostream &os, std::string func_name,
    std::vector<std::string> &tiling_args,
    std::vector<const tir::VarNode *> &shape_vars) {
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
  for (auto &key : var_sequence_) {
    if (tiling_map_.find(key) != tiling_map_.end()) {
      auto value = tiling_map_[key];
      this->PrintIndent();
      os << key << " = ";
      PrintExpr(arith::Analyzer().Simplify(value), os);
      os << ";\n";
    }
  }
  this->EndScope(func_scope);
  os << "}\n\n";
}

void CodeGenTileLangAscend::PrintHostFunc(
    const PrimFunc &f, const std::string &name, std::ostringstream &os,
    std::string &core, std::vector<const tir::VarNode *> &shape_vars) {
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
    if (v.dtype().is_handle()) {
    os << "uint8_t* " << v->name_hint;
    } else {
      os << getType(v.dtype()) << " " << v->name_hint;
  }
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

  address_map_ = f->GetAttr<Map<Var, PrimExpr>>("address_map")
                     .value_or(Map<Var, PrimExpr>());
  use_swizzle_ = f->GetAttr<Bool>("use_swizzle").value_or(Bool(false));
  tiling_map_ = f->GetAttr<Map<Var, PrimExpr>>("tiling_map")
                    .value_or(Map<Var, PrimExpr>());
  var_sequence_ = f->GetAttr<Array<Var>>("var_sequence").value_or(Array<Var>());
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
    auto tiling_varnode = pair.first.get();
    if (var_idmap_.count(tiling_varnode) == 0) {
      (void)AllocVarID(tiling_varnode);
    }
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

std::string
CodeGenTileLangAscend::PrintBufferOffset(const CallNode *call_arg_node,
                                         bool has_offset) {
  auto _var = call_arg_node->args[1].as<VarNode>();
  auto _var_offset = PrintExpr(call_arg_node->args[2]);
  auto _var_name = var_idmap_[_var];
  if (_var_name == "") {
    _var_name = _var->name_hint;
  }
  if (has_offset) {
    return _var_name + "[" + _var_offset + "]";
  }
  return _var_name;
}

void CodeGenTileLangAscend::AddDeclStream(std::ostringstream &ss,
                                          const std::string &str) {
  std::string content = ss.str();
  if (content.find(str) == std::string::npos) {
    ss << str;
  }
}

void CodeGenTileLangAscend::PrintOpCall(const CallNode *op,
                                        const std::string &op_name,
                                        std::pair<int, int> buffer_range,
                                        std::pair<int, int> expr_range) {
  std::vector<std::string> args;

  for (int i = buffer_range.first; i < buffer_range.second; ++i) {
    args.push_back(PrintBufferOffset(op->args[i].as<CallNode>(), true));
  }

  for (int i = expr_range.first; i < expr_range.second; ++i) {
    args.push_back(PrintExpr(op->args[i]));
  }

  this->PrintIndent();
  this->stream << op_name << "(";
  for (size_t i = 0; i < args.size(); ++i) {
    this->stream << args[i];
    if (i != args.size() - 1) {
      this->stream << ", ";
    }
  }
  this->stream << ");\n";
}

void CodeGenTileLangAscend::PrintConstArray(const CallNode *op, int start_idx,
                                            int len, const std::string &dtype) {
  this->stream << "(" << dtype << "[]){";
  for (int i = 0; i < len; ++i) {
    this->stream << PrintExpr(op->args[start_idx + i]);
    if (i < len - 1) {
      this->stream << ", ";
    }
  }
  this->stream << "}";
}

void CodeGenTileLangAscend::BinaryVecOpCodegen(const CallNode *op,
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
  this->stream << ", " << PrintExpr(op->args[op->args.size() - 1]) << ");\n";
}

void CodeGenTileLangAscend::UnaryVecOpCodegen(const CallNode *op,
                                              const std::string &op_name) {
  int len = op->args.size();
  PrintOpCall(op, op_name, {0, len - 1}, {len - 1, len});
}

void CodeGenTileLangAscend::SelectCodegen(const CallNode *op,
                                          const std::string &op_name) {
  std::vector<std::string> var_names;
  int para_idx = 0;
  // For para0:dst, para1:selMask, para2:src0
  for (int i = 0; i <= 2; i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }

  // For para3:src1_type
  int src1_type = std::stoi(PrintExpr(op->args[3]));
  if (src1_type == 0) {
    if (op->args[4].as<CallNode>()) {
      auto var_name4 = PrintBufferOffset(op->args[4].as<CallNode>(), false);
      this->PrintIndent();
      this->stream << "AscendC::PipeBarrier<PIPE_ALL>();\n";
      this->PrintIndent();
      this->stream << "auto " << var_name4 << "_scalar = " << var_name4
                   << ".GetValue(" << PrintExpr(op->args[4]) << ");\n";
      var_names.push_back(var_name4 + "_scalar");
    }

    auto var_name5 = Downcast<StringImm>(op->args[5])->value;
    var_names.push_back("AscendC::SELMODE::" + var_name5);

    auto var_name6 = PrintExpr(op->args[6]);
    var_names.push_back(var_name6);
  } else if (src1_type == 1) {
    auto var_name4 = PrintExpr(op->args[4]);
    var_names.push_back(var_name4);

    auto var_name5 = Downcast<StringImm>(op->args[5])->value;
    var_names.push_back("AscendC::SELMODE::" + var_name5);

    auto var_name6 = PrintExpr(op->args[6]);
    var_names.push_back(var_name6);
  } else if (src1_type == 2) {
    auto var_name4 = PrintBufferOffset(op->args[4].as<CallNode>());
    var_names.push_back(var_name4);

    auto var_name5 = Downcast<StringImm>(op->args[5])->value;
    var_names.push_back("AscendC::SELMODE::" + var_name5);

    auto var_name6 = PrintExpr(op->args[6]);
    var_names.push_back(var_name6);
  }

  this->stream << op_name << "(";
  for (int i = 0; i < var_names.size(); i++) {
    this->stream << var_names[i];
    if (i != var_names.size() - 1) {
      this->stream << ", ";
    }
  }
  this->stream << ");\n";
}

void CodeGenTileLangAscend::ScalarOpCodegen(const CallNode *op,
                                            const std::string &op_name) {
  PrintOpCall(op, op_name, {0, 2}, {2, op->args.size()});
}

void CodeGenTileLangAscend::ShiftOpCodegen(const CallNode *op,
                                           const std::string &op_name) {
  PrintOpCall(op, op_name, {0, 2}, {2, op->args.size()});
}

void CodeGenTileLangAscend::TrigOpCodegen(const CallNode *op,
                                          const std::string &op_name) {
  int len = op->args.size();
  PrintOpCall(op, op_name, {0, len - 1}, {len - 1, len});
}

void CodeGenTileLangAscend::TransposeCodegen(const CallNode *op,
                                             const std::string &op_name) {
  PrintOpCall(op, op_name, {0, op->args.size()}, {0, 0});
}

void CodeGenTileLangAscend::CreateVecIndexCodegen(const CallNode *op,
                                                  const std::string &op_name) {
  PrintOpCall(op, op_name, {0, 1}, {1, op->args.size()});
}

void CodeGenTileLangAscend::FillCodegen(const CallNode *op) {
  std::string op_name = Downcast<StringImm>(op->args[0])->value;
  PrintOpCall(op, op_name, {1, 2}, {2, 4});
}

void CodeGenTileLangAscend::ArithProgressionCodegen(const CallNode *op) {
  std::string op_name = Downcast<StringImm>(op->args[0])->value;
  PrintOpCall(op, op_name, {1, 2}, {2, 5});
}

void CodeGenTileLangAscend::SortCodegen(const CallNode *op) {
  std::string op_name = Downcast<StringImm>(op->args[0])->value;
  std::vector<std::string> var_names;
  for (int i = 1; i < op->args.size() - 2; i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }

  auto var_name =
      PrintBufferOffset(op->args[op->args.size() - 2].as<CallNode>(),
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
  this->stream << ", " << PrintExpr(op->args[op->args.size() - 1]) << ");\n";
}

void CodeGenTileLangAscend::MergeSortCodegen(const CallNode *op) {
  std::string op_name = Downcast<StringImm>(op->args[0])->value;
  int len = op->args.size();
  PrintOpCall(op, op_name, {1, len - 3}, {len - 3, len});
}

void CodeGenTileLangAscend::TopKCodegen(const CallNode *op) {
  std::string op_name = Downcast<StringImm>(op->args[0])->value;
  int len = op->args.size();
  PrintOpCall(op, op_name, {1, len - 1}, {len - 1, len});
}

void CodeGenTileLangAscend::GatherMaskCodegen(const CallNode *op) {
  std::string op_name = Downcast<StringImm>(op->args[0])->value;
  int len = op->args.size();
  PrintOpCall(op, op_name, {1, len - 1}, {len - 1, len});
}

void CodeGenTileLangAscend::GatherbCodegen(const CallNode *op) {
  std::string op_name = Downcast<StringImm>(op->args[0])->value;
  PrintOpCall(op, op_name, {1, 4}, {4, 7});
}

void CodeGenTileLangAscend::InitSortBufCodegen(const CallNode *op) {
  std::string op_name = Downcast<StringImm>(op->args[0])->value;
  PrintOpCall(op, op_name, {1, 2}, {2, 3});
}

void CodeGenTileLangAscend::AddsAndMulsOpCodegen(const CallNode *op,
                                                 const std::string &op_name) {
  std::vector<std::string> var_names;
  for (int i = 0; i < 2; i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }

  if (op->args[2].as<CallNode>()) {
    auto var_name = PrintBufferOffset(op->args[2].as<CallNode>(), false);
    this->PrintIndent();
    this->stream << "AscendC::PipeBarrier<PIPE_ALL>();\n";
    this->PrintIndent();
    this->stream << "auto " << var_name << "_scalar = " << var_name
                 << ".GetValue(" << PrintExpr(op->args[op->args.size() - 2])
                 << ");\n";
    var_names.push_back(var_name + "_scalar");
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

  this->stream << ", " << PrintExpr(op->args[op->args.size() - 1]) << ");\n";
}

void CodeGenTileLangAscend::SubsOpCodegen(const CallNode *op) {
  std::vector<std::string> var_names;
  for (int i = 0; i < 2; i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }

  if (op->args[2].as<CallNode>()) {
    auto var_name = PrintBufferOffset(op->args[2].as<CallNode>(), false);

        this->PrintIndent();
        this->stream << "AscendC::PipeBarrier<PIPE_ALL>();\n";
        this->PrintIndent();
        this->stream << "auto " << var_name << "_scalar = " << var_name
                     << ".GetValue(" << PrintExpr(op->args[op->args.size() - 2])
                     << ");\n";
        var_names.push_back("-" + var_name + "_scalar");
      } else {
        var_names.push_back("-" + PrintExpr(op->args[op->args.size() - 2]));
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

  this->stream << ", " << PrintExpr(op->args[op->args.size() - 1]) << ");\n";
}

void CodeGenTileLangAscend::DivsOpCodegen(const CallNode *op) {
  std::vector<std::string> var_names;
  for (int i = 0; i < 2; i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }

  if (op->args[2].as<CallNode>()) {
    auto var_name = PrintBufferOffset(op->args[2].as<CallNode>(), false);

    this->PrintIndent();
    this->stream << "AscendC::PipeBarrier<PIPE_ALL>();\n";
    this->PrintIndent();
    this->stream << "auto " << var_name << "_scalar = 1.0f / " << var_name
                 << ".GetValue(" << PrintExpr(op->args[op->args.size() - 2])
                 << ");\n";
    var_names.push_back(var_name + "_scalar");
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

  this->stream << ", " << PrintExpr(op->args[op->args.size() - 1]) << ");\n";
}

void CodeGenTileLangAscend::CompareCodegen(const CallNode *op,
                                           const std::string &op_name) {
  std::vector<std::string> var_names;
  for (int i = 0; i < op->args.size() - 2; i++) {
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
  this->stream << ", "
               << "AscendC::CMPMODE::" +
                      Downcast<StringImm>(op->args[op->args.size() - 2])->value
               << ", " << PrintExpr(op->args[op->args.size() - 1]) << ");\n";
}

void CodeGenTileLangAscend::CompareScalarCodegen(const CallNode *op,
                                                 const std::string &op_name) {
  std::vector<std::string> var_names;
  int para_idx = 0;
  for (int i = 0; i <= 1; i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }

  para_idx = 2;
  if (op->args[para_idx].as<CallNode>()) {
    auto var_name = PrintBufferOffset(op->args[para_idx].as<CallNode>(), false);
    this->PrintIndent();
    this->stream << "AscendC::PipeBarrier<PIPE_ALL>();\n";
    this->PrintIndent();
    this->stream << "auto " << var_name << "_scalar = " << var_name
                 << ".GetValue(" << PrintExpr(op->args[para_idx + 1]) << ");\n";
    var_names.push_back(var_name + "_scalar");
    para_idx++;
  } else {
    var_names.push_back(PrintExpr(op->args[op->args.size() - 3]));
  }
  para_idx++;

  auto var_name_mode =
      "AscendC::CMPMODE::" + Downcast<StringImm>(op->args[para_idx])->value;
  var_names.push_back(var_name_mode);
  para_idx++;

  auto var_name_size = PrintExpr(op->args[para_idx]);
  var_names.push_back(var_name_size);

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

void CodeGenTileLangAscend::Sort32Codegen(const CallNode *op,
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
  this->stream << ", " << PrintExpr(op->args[op->args.size() - 1]) << ");\n";
}

void CodeGenTileLangAscend::GatherCodegen(const CallNode *op,
                                          const std::string &op_name) {
  this->PrintIndent();
  auto var_name_1 = PrintBufferOffset(op->args[0].as<CallNode>());
  auto var_name_2 = PrintBufferOffset(op->args[1].as<CallNode>());
  auto var_name_3 = PrintBufferOffset(op->args[2].as<CallNode>());

  this->stream << op_name << "(" << var_name_1 << ", " << var_name_2 << ", "
               << var_name_3 << ", " << PrintExpr(op->args[3]) << ", "
               << PrintExpr(op->args[4]) << ");\n";
}

void CodeGenTileLangAscend::ReduceOpCodegen(const CallNode *op) {
  std::string op_name = "tl::ascend::" + Downcast<StringImm>(op->args[0])->value;

  std::vector<std::string> var_names;
  for (int i = 1; i < op->args.size(); i++) {
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

void CodeGenTileLangAscend::BlockReduceOpCodegen(const CallNode *op,
                                                 const std::string &op_name) {
  std::vector<std::string> var_names;
  int exprStartIndex = 2;
  for (int i = 0; i < exprStartIndex; i++) {
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
  for (int i = exprStartIndex; i < op->args.size(); i++) {
    this->stream << ", " << PrintExpr(op->args[i]);
  }
  this->stream << ");\n";
}

void CodeGenTileLangAscend::CastCodegen(const CallNode *op,
                                        const std::string &op_name) {
  this->PrintIndent();
  this->stream << op_name << "(";

  std::vector<std::string> var_names;
  for (int i = 0; i <= 1; i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }

  for (int i = 0; i < var_names.size(); i++) {
    this->stream << var_names[i];
    if (i != var_names.size() - 1) {
      this->stream << ", ";
    }
  }

  this->stream << ", "
               << "AscendC::RoundMode::" +
                      Downcast<StringImm>(op->args[2])->value;
  this->stream << ", " << PrintExpr(op->args[3]);
  this->stream << ");\n";
}

void CodeGenTileLangAscend::SetDeqScaleCodegen(const CallNode *op,
                                               const std::string &op_name) {
  this->PrintIndent();

  this->stream << op_name << "(";
  this->stream << PrintExpr(op->args[0]);

  this->stream << ");\n";
}

void CodeGenTileLangAscend::PowerOpCodegen(const CallNode *op,
                                           const std::string &op_name) {
  PrintOpCall(op, op_name, {0, op->args.size()}, {0, 0});
}

void CodeGenTileLangAscend::BroadcastOpCodegen(const CallNode *op) {
  std::string op_name = Downcast<StringImm>(op->args[0])->value;
  int dim = op->args[3].as<IntImmNode>()->value;

  this->PrintIndent();
  this->stream << op_name << "(";
  // 1. Dst Buffer
  this->stream << PrintBufferOffset(op->args[1].as<CallNode>()) << ",";
  // 2. Src Buffer
  this->stream << PrintBufferOffset(op->args[2].as<CallNode>()) << ",";
  // 3. Dst Shape Array
  PrintConstArray(op, 4, dim);
  this->stream << ", ";
  // 4. Src Shape Array
  PrintConstArray(op, 4 + dim, dim);
  this->stream << ");\n";
}

void CodeGenTileLangAscend::SetCrossFlagCodegen(const CallNode *op) {
  std::string pipe = Downcast<StringImm>(op->args[0])->value;
  std::string op_name = "AscendC::CrossCoreSetFlag<0x2, PIPE_" + pipe + ">";

  PrintOpCall(op, op_name, {0, 0}, {1, op->args.size()});
}

void CodeGenTileLangAscend::FlagOpCodegen(const CallNode *op,
                                          std::string op_name) {
  std::string src = Downcast<StringImm>(op->args[0])->value;
  std::string dst = Downcast<StringImm>(op->args[1])->value;

  op_name += "<AscendC::HardEvent::" + src + "_" + dst + ">";
  PrintOpCall(op, op_name, {0, 0}, {2, op->args.size()});
}

void CodeGenTileLangAscend::PipeBarrierCodegen(const CallNode *op) {
  std::string pipe = Downcast<StringImm>(op->args[0])->value;

  std::string op_name = "AscendC::PipeBarrier<PIPE_" + pipe + ">";

  PrintOpCall(op, op_name, {0, 0}, {0, 0});
}

void CodeGenTileLangAscend::GemmOpCodegen(const CallNode *op) {
  std::string op_name = "tl::ascend::" + Downcast<StringImm>(op->args[0])->value;
  
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

  this->stream << op_name << "(" << a_name << "[" << a_offset << "], " << b_name
               << "[" << b_offset << "], " << c_name << "[" << c_offset
               << "], ascend_l0a, ascend_l0b, " << PrintExpr(op->args[4])
               << ");\n";
}

void CodeGenTileLangAscend::PrintfOpCodegen(const CallNode *op,
                                            const std::string &op_name) {
  this->PrintIndent();
  this->stream << op_name << "(";
  for (size_t i = 0; i < op->args.size(); ++i) {
    if (i > 0) {
      this->stream << ", ";
    }
    if (auto *arg = op->args[i].as<CallNode>()) {
      if (arg->op.same_as(builtin::tvm_access_ptr())) {
        this->stream << PrintBufferOffset(arg, false) << ".GetPhyAddr()";
      } else {
        std::cout
            << "CallNode with builtin::tvm_access_ptr is requested, but got "
            << op->args[i] << ".\n";
      }
    } else {
      this->stream << PrintExpr(op->args[i]);
    }
  }
  this->stream << ");\n";
}

void CodeGenTileLangAscend::DumpTensorCodegen(const CallNode *op) {
  AddDeclStream(decl_stream, "#include \"tl_templates/ascend/printf.h\"\n");
  this->PrintIndent();
  this->stream << "tl::ascend::DumpTensor" << "(";

  // 0. Buffer指针
  this->stream << PrintBufferOffset(op->args[0].as<CallNode>()) << ",";
  // 1. desc
  this->stream << PrintExpr(op->args[1]) << ", ";
  // 2. dump_size
  this->stream << PrintExpr(op->args[2]) << ", ";
  // 3. dim (len(shape_info))
  this->stream << PrintExpr(op->args[3]) << ", ";

  // 4. shapeInfo数组指针
  if (op->args.size() > 4) {
    this->stream << "(uint32_t[]){";
    for (int i = 4; i < op->args.size(); ++i) {
      if (i > 4) {
        this->stream << ", ";
      }
      this->stream << PrintExpr(op->args[i]);
    }
    this->stream << "}";
  } else {
    this->stream << "nullptr";
  }

  this->stream << ");\n";
}

void CodeGenTileLangAscend::BilinearInterpolationCodegen(const CallNode *op) {
  std::string op_name = "AscendC::BilinearInterpolation";
  this->PrintIndent();
  auto var_name = PrintBufferOffset(op->args[0].as<CallNode>());
  auto var_name_1 = PrintBufferOffset(op->args[1].as<CallNode>());
  auto var_name_2 = PrintBufferOffset(op->args[2].as<CallNode>());
  auto var_name_3 = PrintBufferOffset(op->args[3].as<CallNode>());
  auto var_name_4 = PrintBufferOffset(op->args[10].as<CallNode>());
  this->stream << op_name << "(" << var_name << ", " << var_name_1 << ", "
               << var_name_2 << ", " << var_name_3 << ", "
               << PrintExpr(op->args[4]) << ", " << PrintExpr(op->args[5])
               << ", " << PrintExpr(op->args[6]) << ", "
               << PrintExpr(op->args[7]) << ", " << PrintExpr(op->args[8])
               << ", " << PrintExpr(op->args[9]) << ", " << var_name_4
               << ");\n";
}

void CodeGenTileLangAscend::WholeReduceOpCodegen(const CallNode *op,
                                                 const std::string &op_name) {
  std::vector<std::string> var_names;
  for (int i = 0; i < 2; i++) {
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
  for (int i = 2; i < op->args.size() - 1; i++) {
    this->stream << ", " << PrintExpr(op->args[i]);
  }
  this->stream << ", " << "AscendC::ReduceOrder::"
               << Downcast<StringImm>(op->args[op->args.size() - 1])->value
               << ");\n";
}

} // namespace codegen
} // namespace tvm
