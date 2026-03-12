/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file npu_loop_vectorize.cc
 * \brief vectorize the loop
 */

#include "arith/ir_mutator_with_analyzer.h"
#include "tir/analysis/var_use_def_analysis.h"

#include <tvm/arith/analyzer.h>
#include <tvm/arith/pattern.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/op.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/runtime/registry.h>

#include "arith/scalable_expression.h"
#include "tir/analysis/check_contains.h"

namespace tvm {
namespace tl {

using namespace tir;

class LoopDecompose : public StmtMutator {
private:
  // Temporary buffers of subexpression results when processing compound expressions
  std::vector<Buffer> tmp_buffers;
  // Recursion depth when processing compound expressions recursively
  int depth = 0;

  inline static std::unordered_map<std::string, std::string> TirOps2NpuirOps = {
    {"tir.exp", "tl.npuir_exp"},
    {"tir.fabs", "tl.npuir_abs"},
    {"tir.sigmoid", "tl.npuir_sigmoid"},
    {"tir.if_then_else", "tl.npuir_select"},
    {"Add", "tl.npuir_add"},
    {"Mul", "tl.npuir_mul"},
    {"Sub", "tl.npuir_sub"},
    {"Div", "tl.npuir_div"},
    {"EQ", "tl.npuir_cmp"},
    {"NE", "tl.npuir_cmp"},
    {"LT", "tl.npuir_cmp"},
    {"LE", "tl.npuir_cmp"},
    {"GE", "tl.npuir_cmp"},
    {"GT", "tl.npuir_cmp"},
    {"Copy", "tl.copy"},
    {"Broadcast", "tl.npuir_brc"},
  };

  bool IsCmpOps(const std::string& op_name) {
    return op_name == "EQ" ||
           op_name == "NE" ||
           op_name == "LT" ||
           op_name == "GT" ||
           op_name == "LE" ||
           op_name == "GE";
  }

  struct BufferAccessInfo {
    Buffer buffer;
    Array<PrimExpr> indices;
    bool is_load;  // true for BufferLoad, false for BufferStore
  };

  inline BufferAccessInfo ExtractBufferAccessInfo(const ObjectRef& node) {
    BufferAccessInfo info;

    if (auto load = node.as<BufferLoadNode>()) {
      info.buffer = load->buffer;
      info.indices = load->indices;
      info.is_load = true;
    } else if (auto store = node.as<BufferStoreNode>()) {
      info.buffer = store->buffer;
      info.indices = store->indices;
      info.is_load = false;
    } else {
      LOG(FATAL) << "Expected BufferLoad or BufferStore, but got: "
                 << node->GetTypeKey();
    }

    return info;
  }

  PrimExpr BuildRegionCall(const BufferAccessInfo& buffer, int region_id, int size) {
    PrimExpr buffer_load = BufferLoad(buffer.buffer, buffer.indices);
    std::vector<PrimExpr> args_vec = {
      buffer_load,
      make_const(DataType::Int(32), region_id),
    };
    for (int i = 0; i < buffer.indices.size(); ++i) {
      args_vec.push_back(size);
    }
    Op region_op = Op::Get("tl.region");
    return Call(DataType::Handle(), region_op, {args_vec});
  }

  BufferAccessInfo CreateTempBuffer(const BufferAccessInfo& output_buffer, std::string op_name = "npuir_add") {
    static int tmp_id = 0;
    Buffer ref = output_buffer.buffer;
    DataType dtype = ref->dtype;
    auto scope = ref.scope();
    int64_t size = 1;
    for (const auto& dim : ref->shape) {
      if (auto imm = dim.as<IntImmNode>()) {
        size *= imm->value;
      } else {
        LOG(FATAL) << "Can not create temporary buffer for dynamic shape.";
        BufferAccessInfo info {Buffer()};
        return info;
      }
    }

    std::string name = "tmp_" + std::to_string(tmp_id++) + "_buf";
    Buffer buf;
    if (!IsCmpOps(op_name)) {
      buf = Buffer(
        Var(name, PointerType(PrimType(dtype), scope)), 
        dtype, ref->shape, {}, PrimExpr(0), name, 0, 0, kDefault
      );
    } else {
      buf = Buffer(
        Var(name, PointerType(PrimType(DataType::Bool()), scope)), 
        DataType::Bool(), ref->shape, {}, PrimExpr(0), name, 0, 0, kDefault
      );
    }

    tmp_buffers.push_back(buf);
    BufferAccessInfo info {buf, output_buffer.indices};
    return info;
  }

  Stmt VisitStmt_(const BlockNode* op) override {
    auto saved_buffers = std::move(tmp_buffers);

    tmp_buffers.clear();

    Stmt new_body = VisitStmt(op->body);

    Array<Buffer> allocs = op->alloc_buffers;
    for (const Buffer& buf : tmp_buffers) {
      allocs.push_back(buf);
    }

    tmp_buffers = std::move(saved_buffers);

    if (allocs.same_as(op->alloc_buffers) && new_body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    }

    return Block(
      op->iter_vars, op->reads, op->writes,
      op->name_hint, new_body, op->init, allocs
    );
  }


  Stmt ReplaceForBody(const ForNode* op, const Array<Stmt>& statements) {
    Stmt new_body = SeqStmt::Flatten(statements);

    // replace for body with new body
    return For(op->loop_var,
               op->min,
               op->extent,
               op->kind,
               new_body,
               op->thread_binding,
               op->annotations);
  }

  static char to_lowercase(unsigned char c)
  {
      return std::tolower(c);
  }
 

  Stmt BuildNPUIRBinaryCall(std::string op_name,
                            const BufferAccessInfo& buffer_a,
                            const BufferAccessInfo& buffer_b,
                            const BufferAccessInfo& buffer_c) {
    PrimExpr region_a = BuildRegionCall(buffer_a, 1, 1);
    PrimExpr region_b = BuildRegionCall(buffer_b, 1, 1);
    PrimExpr region_c = BuildRegionCall(buffer_c, 2, 1);
    auto op_type = Op::Get(TirOps2NpuirOps[op_name]);

    Array<PrimExpr> args = {region_a, region_b, region_c};
    if (IsCmpOps(op_name)) {
      std::string str = op_name;
      std::transform(str.cbegin(), str.cend(), str.begin(), to_lowercase);
      StringImm cmp_mode(str);
      args.push_back(cmp_mode);
    }
    PrimExpr call = Call(DataType::Void(), op_type, args);
    return Evaluate(call);
  }

  Stmt BuildNPUIRBinaryCall(std::string op_name,
                            const BufferAccessInfo& buffer_a,
                            const PrimExpr& scalar_b,
                            const BufferAccessInfo& buffer_c) {
    PrimExpr region_a = BuildRegionCall(buffer_a, 1, 1);
    PrimExpr region_c = BuildRegionCall(buffer_c, 2, 1);
    auto op_type = Op::Get(TirOps2NpuirOps[op_name]);

    Array<PrimExpr> args = {region_a, scalar_b, region_c};
    if (IsCmpOps(op_name)) {
      std::string str = op_name;
      std::transform(str.cbegin(), str.cend(), str.begin(), to_lowercase);
      StringImm cmp_mode(str);
      args.push_back(cmp_mode);
    }
    PrimExpr call = Call(DataType::Void(), op_type, args);
    return Evaluate(call);
  }

  Stmt BuildNPUIRBinaryCall(std::string op_name,
                            const PrimExpr& scalar_a,
                            const BufferAccessInfo& buffer_b,
                            const BufferAccessInfo& buffer_c) {
    PrimExpr region_b = BuildRegionCall(buffer_b, 1, 1);
    PrimExpr region_c = BuildRegionCall(buffer_c, 2, 1);
    auto op_type = Op::Get(TirOps2NpuirOps[op_name]);

    Array<PrimExpr> args = {scalar_a, region_b, region_c};
    if (IsCmpOps(op_name)) {
      std::string str = op_name;
      std::transform(str.cbegin(), str.cend(), str.begin(), to_lowercase);
      StringImm cmp_mode(str);
      args.push_back(cmp_mode);
    }
    PrimExpr call = Call(DataType::Void(), op_type, args);
    return Evaluate(call);
  }

  Stmt BuildNpuirBinaryCall(std::string op_name, PrimExpr a, PrimExpr b, PrimExpr c) {
    auto op_type = Op::Get(TirOps2NpuirOps[op_name]);
    Array<PrimExpr> args = {a, b, c};
    if (IsCmpOps(op_name)) {
      std::string str = op_name;
      std::transform(str.cbegin(), str.cend(), str.begin(), to_lowercase);
      StringImm cmp_mode(str);
      args.push_back(cmp_mode);
    }
    PrimExpr call = Call(DataType::Void(), op_type, args);
    return Evaluate(call);
  }

  Stmt BuildNPUIRUnaryCall(std::string op_name, const PrimExpr& scalar, const BufferAccessInfo& buffer_out,
                           int offset) {
    PrimExpr offset_expr = {make_const(DataType::Int(32), offset)};
    PrimExpr region_out = BuildRegionCall(buffer_out, 2, 1);
    auto op_type = Op::Get(TirOps2NpuirOps[op_name]);

    Array<PrimExpr> args = {scalar, region_out};
    PrimExpr call = Call(DataType::Void(), op_type, args);
    return Evaluate(call);
  }

  Stmt BuildNPUIRUnaryCall(std::string op_name, const BufferAccessInfo& buffer_in, const BufferAccessInfo& buffer_out,
                           int offset) {
    PrimExpr offset_expr = {make_const(DataType::Int(32), offset)};
    PrimExpr region_in = BuildRegionCall(buffer_in, 1, 1);
    PrimExpr region_out = BuildRegionCall(buffer_out, 2, 1);
    auto op_type = Op::Get(TirOps2NpuirOps[op_name]);

    Array<PrimExpr> args = {region_in, region_out};
    PrimExpr call = Call(DataType::Void(), op_type, args);
    return Evaluate(call);
  }

  bool IsCopyOp(const PrimExpr &expr, const std::vector<const VarNode *> &loop_vars) {
    if (IsScalarLike(expr, loop_vars) || expr.as<BufferLoadNode>()) {
      return true;
    }
    return false;
  }

  bool IsUnaryOp(const PrimExpr& expr) {
    if (auto call = expr.as<CallNode>()) {
      std::string op_name;

      if (auto* op_ptr = call->op.as<OpNode>()) {
        op_name = op_ptr->name;
        if (op_name == "tir.exp" || op_name == "tir.fabs" || op_name == "tir.sigmoid") {
          return true;
        }
      }
    }
    return false;
  }

  bool IsBinaryOp(const PrimExpr& expr, std::string* op_type, Array<PrimExpr>* operands) {
  #define HANDLE_BINARY_OP(NodeType, OpName) \
    if (auto node = expr.as<NodeType>()) { \
      if (op_type) *op_type = OpName; \
      if (operands) { \
        operands->push_back(node->a); \
        operands->push_back(node->b); \
      } \
      return true; \
    }

    HANDLE_BINARY_OP(AddNode, "Add");
    HANDLE_BINARY_OP(MulNode, "Mul");
    HANDLE_BINARY_OP(SubNode, "Sub");
    HANDLE_BINARY_OP(DivNode, "Div");
    HANDLE_BINARY_OP(LTNode, "LT");
    HANDLE_BINARY_OP(LENode, "LE");
    HANDLE_BINARY_OP(GENode, "GE");
    HANDLE_BINARY_OP(GTNode, "GT");
    HANDLE_BINARY_OP(EQNode, "EQ");
    HANDLE_BINARY_OP(NENode, "NE");

  #undef HANDLE_BINARY_OP
    return false;
  }

  bool IsTernaryOp(const PrimExpr& expr) {
    if (auto call = expr.as<CallNode>()) {
      if (auto* op_ptr = call->op.as<OpNode>()) {
        if (op_ptr->name == "tir.if_then_else") {
          return true;
        }
      }
    }
    return false;
  }

  bool IsScalar(const PrimExpr& expr) {
    return expr.as<IntImmNode>() || expr.as<FloatImmNode>();
  }

  bool IsLoopInvariant(const Array<PrimExpr>& indices,
                       const std::vector<const VarNode*>& loop_vars) {
    for (const auto& idx : indices) {
      bool found = false;
      PostOrderVisit(idx, [&](const ObjectRef& node) {
        if (const VarNode* v = node.as<VarNode>()) {
          if (std::find(loop_vars.begin(), loop_vars.end(), v) != loop_vars.end()) {
            found = true;
          }
        }
      });
      if (found) return false;
    }
    return true;
  }

  bool IndicesSameAsLoopVars(const Array<PrimExpr>& indices,
                             const std::vector<const VarNode*>& loop_vars) {
    if (indices.size() != loop_vars.size()) {
      return false;
    }
    for (int i = 0; i < indices.size(); i++) {
      if (indices[i].as<VarNode>() != loop_vars[i]) {
        return false;
      }
    }
    return true;
  }

  bool ValidBufferIndices(const Array<PrimExpr>& indices) {
    for (const auto& index : indices) {
      // Case of constant int
      if (const auto* int_imm = index.as<tir::IntImmNode>()) {
        continue;
      }

      // Case of constant var
      if (const auto* var = index.as<tir::VarNode>()) {
        if (var->dtype.is_int()) {
          continue;
        }
      }
      return false;
    }
    return true;
  }

  bool IsScalarLike(const PrimExpr& expr,
                    const std::vector<const VarNode*>& loop_vars) {
    if (IsScalar(expr)) {
      return true;
    }

    if (auto load = expr.as<BufferLoadNode>()) {
      return IsLoopInvariant(load->indices, loop_vars);
    }

    return false;
  }

  bool HandleSimpleExpression(std::string op_name, const Array<PrimExpr>& operands,
                              const BufferAccessInfo& output_buffer, std::vector<BufferAccessInfo>* tmp_bufs,
                              const std::vector<const VarNode*>& loop_vars, Array<Stmt>* statements) {
    bool a_is_scalar = IsScalar(operands[0]);
    bool b_is_scalar = IsScalar(operands[1]);
    bool a_is_buffer = operands[0]->IsInstance<BufferLoadNode>();
    bool b_is_buffer = operands[1]->IsInstance<BufferLoadNode>();

    PrimExpr region_a;
    PrimExpr region_b;
    PrimExpr region_c;

    auto processOperand = [&](const PrimExpr& operand, bool is_scalar, bool is_buffer)
      -> std::optional<PrimExpr> {
      if (is_scalar) {
        return operand;
      } else if (is_buffer) {
        auto load = operand.as<BufferLoadNode>();
        if (ValidBufferIndices(load->indices)) {
          return BuildRegionCall(ExtractBufferAccessInfo(operand), 1, 1);
        }
        depth = 0;
        return std::nullopt;
      } else {
        depth = 0;
        return std::nullopt;
      }
    };

    if (auto result = processOperand(operands[0], a_is_scalar, a_is_buffer)) {
        region_a = *result;
    } else {
        return false;
    }
    if (auto result = processOperand(operands[1], b_is_scalar, b_is_buffer)) {
      region_b = *result;
    } else {
      return false;
    }

    BufferAccessInfo output = output_buffer;
    if(depth > 1) {
      auto load_node = a_is_buffer ? operands[0].as<BufferLoadNode>() : operands[1].as<BufferLoadNode>();
      BufferAccessInfo info {load_node->buffer, load_node->indices};
      output = CreateTempBuffer(info, op_name);
      tmp_bufs->push_back(output);
    }
    region_c = BuildRegionCall(output, 2, 1);

    Stmt stmt;
    // For hivm.vadd/hivm.vmul/hivm.vcmp, it's operand at index 0 must be vector
    if (a_is_scalar && (op_name == "Add" || op_name == "Mul" || IsCmpOps(op_name))) {
      if (op_name == "LT") op_name = "GT";
      if (op_name == "LE") op_name = "GE";
      stmt = BuildNpuirBinaryCall(op_name, region_b, region_a, region_c);
    } else {
      stmt = BuildNpuirBinaryCall(op_name, region_a, region_b, region_c);
    }

    statements->push_back(stmt);
    depth--;
    return true;
  }

  bool HandleCopyExpression(const PrimExpr &expr, const BufferAccessInfo &output_buffer,
                            const std::vector<const VarNode *> &loop_vars, Array<Stmt> *statements) {
    if (IsScalarLike(expr, loop_vars)) {
      // Copy a constant or a loop invariant var into output buffer
      Stmt statement = BuildNPUIRUnaryCall("Broadcast", expr, output_buffer, 0);
      statements->push_back(statement);
      --depth;
      return true;
    } else if (expr.as<BufferLoadNode>()) {
      // Copy from input buffer into output buffer
      Stmt statement = BuildNPUIRUnaryCall("Copy", ExtractBufferAccessInfo(expr), output_buffer, 0);
      statements->push_back(statement);
      --depth;
      return true;
    }
    return false;
  }

  bool HandleUnaryExpression(const PrimExpr& expr, const BufferAccessInfo& output_buffer, std::vector<BufferAccessInfo>* tmp_bufs,
                             const std::vector<const VarNode*>& loop_vars, Array<Stmt>* statements) {
    auto* call = expr.as<CallNode>();
    auto* op_ptr = call->op.as<OpNode>();
    std::string op_name = op_ptr->name;
    BufferAccessInfo output = output_buffer;

    if (auto load = call->args[0].as<BufferLoadNode>()) {
      if (!ValidBufferIndices(load->indices)) {
        depth = 0;
        return false;
      }
      // If indices are consistent with loop vars, it means the address offset of the buffer is 0
      int offset = 0;
      if (depth > 1) {
        output = CreateTempBuffer(output_buffer);
        tmp_bufs->push_back(output);
      }
      Stmt statement = BuildNPUIRUnaryCall(op_name, ExtractBufferAccessInfo(call->args[0]), output, offset);
      statements->push_back(statement);
      depth--;
      return true;
    } else {
      if (DecomposeExpression(call->args[0], output_buffer, tmp_bufs, loop_vars, statements)) {
        int offset = 0;
        auto last_buf = tmp_bufs->back();
        if (depth > 1) {
          output = CreateTempBuffer(last_buf);
          tmp_bufs->push_back(output);
        }
        Stmt statement = BuildNPUIRUnaryCall(op_name, last_buf, output, offset);
        statements->push_back(statement);
        depth--;
        return true;
      }
    }
    depth = 0;
    return false;
  }

  bool HandleBinaryExpression(std::string op_type, const Array<PrimExpr>& operands,
                              const BufferAccessInfo& output_buffer, std::vector<BufferAccessInfo>* tmp_bufs,
                              const std::vector<const VarNode*>& loop_vars, Array<Stmt>* statements) {
    bool left_is_simple = operands[0].as<BufferLoadNode>() || IsScalarLike(operands[0], loop_vars);
    bool right_is_simple = operands[1].as<BufferLoadNode>() || IsScalarLike(operands[1], loop_vars);
    BufferAccessInfo output = output_buffer;

    if (left_is_simple && right_is_simple) {
      return HandleSimpleExpression(op_type, operands, output_buffer, tmp_bufs, loop_vars, statements);
    } else if (!left_is_simple && right_is_simple) {
      if (DecomposeExpression(operands[0], output_buffer, tmp_bufs, loop_vars, statements)) {
        auto last_buf = tmp_bufs->back();
        if (depth > 1) {
          output = CreateTempBuffer(last_buf, op_type);
          tmp_bufs->push_back(output);
        }
        Stmt statement;
        if (IsScalar(operands[1])) {
          if (op_type == "Add" || op_type == "Mul" || IsCmpOps(op_type)) {
            statement = BuildNPUIRBinaryCall(op_type, last_buf, operands[1], output);
          } else {
            statement = BuildNPUIRBinaryCall(op_type, operands[1], last_buf, output);
          }
        } else {
          statement = BuildNPUIRBinaryCall(op_type, last_buf, ExtractBufferAccessInfo(operands[1]), output);
        }
        statements->push_back(statement);
        depth--;
        return true;
      }
    } else if (left_is_simple && !right_is_simple) {
      if (DecomposeExpression(operands[1], output_buffer, tmp_bufs, loop_vars, statements)) {
        auto last_buf = tmp_bufs->back();
        if (depth > 1) {
          output = CreateTempBuffer(last_buf, op_type);
          tmp_bufs->push_back(output);
        }
        Stmt statement;
        if (IsScalar(operands[0])) {
          if (op_type == "Add" || op_type == "Mul" || IsCmpOps(op_type)) {
            if (op_type == "LT") op_type = "GT";
            if (op_type == "LE") op_type = "GE";
            statement = BuildNPUIRBinaryCall(op_type, last_buf, operands[0], output);
          } else {
            statement = BuildNPUIRBinaryCall(op_type, operands[0], last_buf, output);
          }
        } else {
          statement = BuildNPUIRBinaryCall(op_type, ExtractBufferAccessInfo(operands[0]), last_buf, output);
        }
        statements->push_back(statement);
        depth--;
        return true;
      }
    } else if (!left_is_simple && !right_is_simple) {
      if (!DecomposeExpression(operands[0], output_buffer, tmp_bufs, loop_vars, statements)) {
        return false;
      }
      if (!DecomposeExpression(operands[1], output_buffer, tmp_bufs, loop_vars, statements)) {
        return false;
      }
      auto right = tmp_bufs->back();
      auto left = (*tmp_bufs)[tmp_bufs->size() - 2];
      if (depth > 1) {
        output = CreateTempBuffer(left, op_type);
        tmp_bufs->push_back(output);
      }

      Stmt statement = BuildNPUIRBinaryCall(op_type, left, right, output);
      statements->push_back(statement);
      depth--;
      return true;
    }
    depth = 0;
    return false;
  }

  Stmt BuildNPUIRTernaryCall(std::string op_name,
                             const BufferAccessInfo& cond_buf,
                             const BufferAccessInfo& true_buf,
                             const BufferAccessInfo& false_buf,
                             const BufferAccessInfo& out_buf,
                             int offset) {
    PrimExpr offset_expr = {make_const(DataType::Int(32), offset)};
    PrimExpr region_cond = BuildRegionCall(cond_buf, 1, 1);
    PrimExpr region_true = BuildRegionCall(true_buf, 1, 1);
    PrimExpr region_false = BuildRegionCall(false_buf, 1, 1);
    PrimExpr region_out = BuildRegionCall(out_buf, 1, 1);
    auto op_type = Op::Get(TirOps2NpuirOps[op_name]);

    Array<PrimExpr> args = {region_cond, region_true, region_false, region_out};
    PrimExpr call = Call(DataType::Void(), op_type, args);
    return Evaluate(call);
  }

  bool HandleTernaryExpression(const PrimExpr& expr, const BufferAccessInfo& output_buffer, std::vector<BufferAccessInfo>* tmp_bufs,
                             const std::vector<const VarNode*>& loop_vars, Array<Stmt>* statements) {
    auto* call = expr.as<CallNode>();
    auto* op_ptr = call->op.as<OpNode>();
    std::string op_name = op_ptr->name;
    BufferAccessInfo output = output_buffer;

    PrimExpr cond_expr = call->args[0];
    PrimExpr true_expr = call->args[1];
    PrimExpr false_expr = call->args[2];

    if (!DecomposeExpression(cond_expr, output_buffer, tmp_bufs, loop_vars, statements)) {
      depth = 0;
      return false;
    }
    BufferAccessInfo cond_buf = tmp_bufs->back();
    BufferAccessInfo true_buf;
    BufferAccessInfo false_buf;

    auto processExpr = [&](const PrimExpr& val_expr, BufferAccessInfo& buf) {
      if (val_expr.as<BufferLoadNode>()) {
        buf = ExtractBufferAccessInfo(val_expr);
      } else if (IsScalarLike(val_expr, loop_vars)) {
        buf = CreateTempBuffer(output_buffer);
        auto statement = BuildNPUIRUnaryCall("Broadcast", val_expr, buf, 0);
        statements->push_back(statement);
      } else {
        if (!DecomposeExpression(val_expr, output_buffer, tmp_bufs, loop_vars, statements)) {
          return false;
        }
        buf = tmp_bufs->back();
      }
      return true;
    };
    if (!processExpr(true_expr, true_buf)) {
      depth = 0;
      return false;
    }
    if (!processExpr(false_expr, false_buf)) {
      depth = 0;
      return false;
    }
    Stmt statement = BuildNPUIRTernaryCall(op_name, cond_buf, true_buf, false_buf, output, 0);
    statements->push_back(statement);
    depth--;
    return true;
  }

  bool DecomposeExpression(const PrimExpr& expr, const BufferAccessInfo& output_buffer, std::vector<BufferAccessInfo>* tmp_bufs,
                           const std::vector<const VarNode*>& loop_vars, Array<Stmt>* statements) {
    if (IsCopyOp(expr, loop_vars)){
      // Case of copy without computation
      ++depth;
      return HandleCopyExpression(expr, output_buffer, loop_vars, statements);
    }

    if (IsUnaryOp(expr)) {
      depth++;
      return HandleUnaryExpression(expr, output_buffer, tmp_bufs, loop_vars, statements);
    }

    std::string op_type;
    Array<PrimExpr> operands;
    if (IsBinaryOp(expr, &op_type, &operands)) {
      depth++;
      return HandleBinaryExpression(op_type, operands, output_buffer, tmp_bufs, loop_vars, statements);
    }

    if (IsTernaryOp(expr)) {
      depth++;
      return HandleTernaryExpression(expr, output_buffer, tmp_bufs, loop_vars, statements);
    }

    depth = 0;
    return false;
  }

  Stmt VectorizeSingleStatement(const ForNode *op, int loop_count) {
    const BufferStoreNode* store = op->body.as<BufferStoreNode>();
    for (const auto& index : store->indices) {
      const VarNode* var = index.as<VarNode>();
      if (!var && !IsScalar(index)) {
        return StmtMutator::VisitStmt_(op);
      }
    }

    std::vector<const VarNode*> loop_vars;
    loop_vars.push_back(op->loop_var.get());
    Array<Stmt> statements;
    std::vector<BufferAccessInfo> tmp_bufs;

    bool ret = DecomposeExpression(store->value, ExtractBufferAccessInfo(op->body), &tmp_bufs, loop_vars, &statements);
    return ret ? ReplaceForBody(op, statements) : StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    if (op->kind != ForKind::kParallel) {
      return StmtMutator::VisitStmt_(op);
    }

    const IntImmNode* loop_extent = op->extent.as<IntImmNode>();
    if (!loop_extent) {
      return StmtMutator::VisitStmt_(op);
    }

    if (!op->body.as<BufferStoreNode>()) {
      return StmtMutator::VisitStmt_(op);
    }

    auto vectorized = VectorizeSingleStatement(op, loop_extent->value);
    if (vectorized.defined()) {
      return vectorized;
    }
    return StmtMutator::VisitStmt_(op);
  }
};

class LoopVectorize : public StmtMutator {
private:
  inline static std::unordered_set<std::string> CandidateVectorizationOps = {
    "tl.copy",
    "tl.npuir_exp",
    "tl.npuir_abs",
    "tl.npuir_add",
    "tl.npuir_mul",
    "tl.npuir_sub",
    "tl.npuir_div",
    "tl.npuir_sigmoid",
    "tl.npuir_brc",
    "tl.npuir_cmp",
    "tl.npuir_select",
  };

  bool IsScalar(const PrimExpr& expr) {
    return expr.as<IntImmNode>() || expr.as<FloatImmNode>();
  }

  // return a integer if the input is able to be converted, else return null
  std::optional<int64_t> TryGetConstIntValue(const tvm::PrimExpr& expr, tvm::arith::Analyzer* analyzer = nullptr) {
    // Method 1: Direct check for IntImmNode
    if (const tvm::tir::IntImmNode* int_imm = expr.as<tvm::tir::IntImmNode>()) {
      return int_imm->value;
    }

    // Method 2: Use arith::Analyzer for simplification
    tvm::arith::Analyzer local_analyzer;
    tvm::arith::Analyzer* used_analyzer = analyzer ? analyzer : &local_analyzer;

    tvm::PrimExpr simplified = used_analyzer->Simplify(expr);
    if (const tvm::tir::IntImmNode* simplified_int = simplified.as<tvm::tir::IntImmNode>()) {
      return simplified_int->value;
    }

    // Method 3: Try to get constant bounds
    tvm::arith::ConstIntBound bound = used_analyzer->const_int_bound(expr);
    if (bound->min_value == bound->max_value) {
      return bound->min_value;
    }

    // Method 4: Handle common expression patterns
    if (const tvm::tir::SubNode* sub = expr.as<tvm::tir::SubNode>()) {
      if (sub->a.same_as(sub->b)) {
        return 0;
      }
    }

    return std::nullopt;
  }

  // return a Call to tl.region -> str: T.region(buffer[offset_expr], region_id, size)
  PrimExpr BuildRegionCall(Buffer buffer, const std::vector<PrimExpr>& offset_expr, int region_id, const std::vector<size_t>& size) {
    PrimExpr buffer_load = BufferLoad(buffer, {offset_expr});
    std::vector<PrimExpr> args_vec = {
      buffer_load,
      make_const(DataType::Int(32), region_id),
    };
    for (auto x : size) {
      args_vec.push_back(make_const(DataType::Int(32), x));
    }
    Array<PrimExpr> args(args_vec);
    Op region_op = Op::Get("tl.region");
    return Call(DataType::Handle(), region_op, args);
  }

 /**
  * Find the index of the offset containing the loop variable.
  * Returns -1 if not found, -2 if found in multiple offsets or indirect mem access.
  */
  int FindLoopVarInOffsets(const std::vector<PrimExpr>& offsets, const VarNode* loop_var) {
    int found_dim = -1;

    for (int i = 0; i < static_cast<int>(offsets.size()); ++i) {
      bool indirect_found = false;

      class LoopVarVisitor : public ExprVisitor {
      public:
        const VarNode* target_var;
        bool found = false;               // Whether appeared directly
        bool& indirect_found;              // Whether appeared indirectly
        bool indirect_scope = false;       // Status flag: whether the visitor is in a BufferLoad

        LoopVarVisitor(const VarNode* var, bool& indirect_flag)
            : target_var(var), indirect_found(indirect_flag) {}

        void VisitExpr_(const VarNode* op) override {
          if (op == target_var) {
            if (indirect_scope) {
              indirect_found = true;
            } else {
              found = true;
            }
          }
          ExprVisitor::VisitExpr_(op);
        }

        void VisitExpr_(const BufferLoadNode* op) override {
          // Visit a BufferLoad, set the status flag - indirect_scope
          bool saved_scope = indirect_scope;
          indirect_scope = true;
          for (const auto& index : op->indices) {
            VisitExpr(index);
          }
          // Reset the status flag
          indirect_scope = saved_scope;
        }
      } visitor(loop_var, indirect_found);

      // Check each offset
      visitor(offsets[i]);

      if (indirect_found) {
        // Indirect mem access detected.
        return -2;
      }

      if (visitor.found) {
        if (found_dim == -1) {
          // Record the first found
          found_dim = i;
        } else {
          // Found multiple offsets associated with the loop var
          return -2;
        }
      }
    }
    // Returns the unique offset associated with the loop var
    return found_dim;
  }

  Stmt SplitStmtToIndependentForNode(const ForNode* forNode, const Stmt& stmt){
    Var new_loop_var = Var(
      forNode->loop_var->name_hint + "_split",
      forNode->loop_var->dtype
    );

    Map<Var, PrimExpr> var_map;
    var_map.Set(forNode->loop_var, new_loop_var);
    Stmt new_body = Substitute(stmt, var_map);

    return For(new_loop_var,
               forNode->min,
               forNode->extent,
               forNode->kind,
               new_body,
               forNode->thread_binding,
               forNode->annotations,
               forNode->span);
  }

  struct RegionInfo {
    Buffer buffer;
    std::vector<PrimExpr> offsets;
    int regionId;
    std::vector<size_t> sizes;
  };

  std::optional<RegionInfo> ParseRegionCall(const PrimExpr &expr, arith::Analyzer *analyzer = nullptr) {
    // 1. must be a CallNode
    const auto *call = expr.as<CallNode>();
    if (!call) return std::nullopt;

    // 2. must be tl.region
    const auto *op_node = call->op.as<OpNode>();
    if (!op_node) return std::nullopt;
    static const auto *region_op = Op::Get("tl.region").as<OpNode>();
    if (op_node != region_op) return std::nullopt;

    // 3. check the number of args and extract attributes
    if (call->args.size() < 3) return std::nullopt;

    const auto *buffer_load = call->args[0].as<BufferLoadNode>();
    if (!buffer_load) return std::nullopt;

    Buffer buffer = buffer_load->buffer;
    Array<PrimExpr> offsets = buffer_load->indices;
    int d = buffer->shape.size();

    // 4. check the shapes
    if (offsets.size() != d) return std::nullopt;
    if (call->args.size() != d + 2) return std::nullopt;

    // 5. try to extract region id & sizes
    auto regionId_val = TryGetConstIntValue(call->args[1], analyzer);
    if (!regionId_val) return std::nullopt;
    int regionId = static_cast<int>(*regionId_val);

    std::vector<size_t> sizes;
    for (int i = 2; i < d + 2; ++i) {
      auto size_val = TryGetConstIntValue(call->args[i], analyzer);
      if (!size_val) return std::nullopt;
      sizes.push_back(static_cast<size_t>(*size_val));
    }

    // post-processing: convert offsets into vector
    std::vector<PrimExpr> offsets_vec(offsets.begin(), offsets.end());

    return RegionInfo{std::move(buffer), std::move(offsets_vec), regionId, std::move(sizes)};
  }

  bool CheckContinuity(const Buffer &buffer,
                       const std::vector<PrimExpr> &offsets,
                       const std::vector<size_t> &sizes,
                       int loop_var_dim,
                       arith::Analyzer *analyzer = nullptr) {
    arith::Analyzer local_analyzer;
    arith::Analyzer *used_analyzer = analyzer ? analyzer : &local_analyzer;

    // 1. check the sizes to ensure the continuity of mem access
    int d = buffer->shape.size();
    for (int i = 0; i < d; ++i) {
      const PrimExpr &size_expr = make_const(DataType::Int(32), sizes[i]);
      if (i <= loop_var_dim) {
        // size of current and higher dimension should be 1; otherwise, something wrong happened
        if (!used_analyzer->CanProveEqual(size_expr, 1)) return false;
      } else {
        // lower dimensions must participate in the operation completely; otherwise, the current dimension cannot be vectorized.
        // The first condition: The size is equal to the size of the corresponding dimension.
        if (!used_analyzer->CanProveEqual(size_expr, buffer->shape[i])) return false;
        // The second condition: offset equals zero
        if (!used_analyzer->CanProveEqual(offsets[i], 0)) return false;
      }
    }

    return true;
  }

  std::optional<PrimExpr> AnalyzeNewOffset(const std::vector<PrimExpr> &offsets,
                                           const VarNode* loop_var,
                                           int loop_var_dim, int start,
                                           arith::Analyzer *analyzer = nullptr) {
    arith::Analyzer local_analyzer;
    arith::Analyzer *used_analyzer = analyzer ? analyzer : &local_analyzer;

    // check the expression about loop var to ensure the continuity of mem access
    Array<PrimExpr> res = arith::DetectLinearEquation(offsets[loop_var_dim], {GetRef<Var>(loop_var)});
    if (res.empty() || !used_analyzer->CanProveEqual(res[0], 1)) {
      // Not a linear expression or the coefficient of the first term is not 1 -> disable to vectorize
      return std::nullopt;
    }

    PrimExpr new_offset = res[1] + make_const(res[1].dtype(), start);

    return used_analyzer->Simplify(new_offset);
  }

  std::optional<PrimExpr> TryVectorizeRegionInLoop(const PrimExpr& expr, const VarNode* loop_var, size_t start, size_t loop_count,
                                                   arith::Analyzer *analyzer = nullptr){
    arith::Analyzer local_analyzer;
    arith::Analyzer *used_analyzer = analyzer ? analyzer : &local_analyzer;

    // Step 1: try to extract region info
    auto info = ParseRegionCall(expr, used_analyzer);
    if (!info) return std::nullopt;

    // Step 2: check the loop var
    int loop_var_dim = FindLoopVarInOffsets(info->offsets, loop_var);
    // -2 -> multiple or indirect found -> dispersed mem access -> disable to vectorize
    if (loop_var_dim == -2) return std::nullopt;
    // -1 -> not found -> invariant -> enable to vectorize but no need change
    if (loop_var_dim == -1) {
      return BuildRegionCall(info->buffer, info->offsets, info->regionId, info->sizes);
    }

    // Step 3: check the continuity of mem access
    if (!CheckContinuity(info->buffer, info->offsets, info->sizes, loop_var_dim, used_analyzer)) {
      return std::nullopt;
    }

    // Step 4: try to analyze the offset after vectorization
    auto new_offset = AnalyzeNewOffset(info->offsets, loop_var, loop_var_dim, start, used_analyzer);
    if (!new_offset) return std::nullopt;

    // Step 5: build and return vectorized region
    info->offsets[loop_var_dim] = *new_offset;
    info->sizes[loop_var_dim] = loop_count;

    return BuildRegionCall(info->buffer, info->offsets, info->regionId, info->sizes);
  }

  Stmt VectorizeForBody(const ForNode* forNode, const Stmt& stmt) {
    arith::Analyzer analyzer;

    if (const auto* alloc=stmt.as<AllocateNode>()){
      return stmt;
    }

    // Only support evaluate node for vectorization
    const auto* evaluate = stmt.as<EvaluateNode>();
    if (!evaluate) return SplitStmtToIndependentForNode(forNode, stmt);

    // Only support call node inside evaluate node
    const auto* call = evaluate->value.as<tvm::tir::CallNode>();
    if (!call) return SplitStmtToIndependentForNode(forNode, stmt);

    // Only Ops in 'CandidateVectorizationOps' are supported
    bool flag_op_supported = false;
    if (const auto* op_node = call->op.as<tvm::OpNode>()) {
      tvm::Op op = tvm::GetRef<tvm::Op>(op_node);
      flag_op_supported = static_cast<bool>(CandidateVectorizationOps.count(op->name));
    }
    if (!flag_op_supported) return SplitStmtToIndependentForNode(forNode, stmt);

    // min_value & loop_extent_value must exist
    auto loop_min_value = TryGetConstIntValue(forNode->min, &analyzer);
    auto loop_extent_value = TryGetConstIntValue(forNode->extent, &analyzer);
    if (!loop_min_value || !loop_extent_value) {
      return SplitStmtToIndependentForNode(forNode, stmt);
    }

    // Enable vectorization only when regions satisfy specific conditions
    auto loop_var_node_ptr = forNode->loop_var.get();
    std::vector<PrimExpr> new_regions;
    for (const auto& region : call->args) {
      if (IsScalar(region) || region.as<StringImmNode>()) {
        new_regions.push_back(region);
        continue;
      }
      auto new_region = TryVectorizeRegionInLoop(region, loop_var_node_ptr, *loop_min_value, *loop_extent_value, &analyzer);
      if (!new_region) {
        return SplitStmtToIndependentForNode(forNode, stmt);
      }
      new_regions.push_back(*new_region);
    }

    Array<PrimExpr> args(new_regions);
    PrimExpr new_call = Call(DataType::Void(), call->op, args);
    return Evaluate(new_call);
  }

  Stmt VectorizeForBody(const ForNode* forNode, const SeqStmt& seqStmt) {
    std::vector<Stmt> result_vec;
    for (size_t i = 0; i < seqStmt->size(); ++i) {
      result_vec.push_back(VectorizeForBody(forNode, seqStmt[i]));
    }
    Array<Stmt> result_arr(result_vec);
    return SeqStmt::Flatten(SeqStmt(result_arr));
  }

  Stmt VisitStmt_(const ForNode* op) final {
    // try vectorize only when marked as parallel
    if (op->kind != ForKind::kParallel) {
      return StmtMutator::VisitStmt_(op);
    }

    // recursive processing
    Stmt body = op->body;
    if (const auto* forNode = body.as<ForNode>()) {
      body = VisitStmt_(forNode);
    }

    if (const auto* seqStmtNode = body.as<SeqStmtNode>()) {
      return VectorizeForBody(op, GetRef<SeqStmt>(seqStmtNode));
    } else if (const auto* stmtNode = body.as<EvaluateNode>()) {
      return VectorizeForBody(op, GetRef<Evaluate>(stmtNode));
    } else {
      return StmtMutator::VisitStmt_(op);
    }
  }
};

using namespace tir::transform;

tvm::transform::Pass NpuLoopVectorize() {
  auto pass_func = [=](PrimFunc f,IRModule m,PassContext ctx) {
    auto *new_pf = f.CopyOnWrite();
    new_pf->body = LoopDecompose()(std::move(new_pf->body));
    new_pf->body = LoopVectorize()(std::move(new_pf->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.NpuLoopVectorize", {});
}

TVM_REGISTER_GLOBAL("tl.transform.NpuLoopVectorize").set_body_typed(NpuLoopVectorize);

} // namespace tl
} // namespace tvm
