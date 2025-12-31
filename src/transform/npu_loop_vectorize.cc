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

std::unordered_map<std::string, std::string> TirOps2NpuirOps = {
    {"tir.exp", "tl.npuir_exp"},
    {"tir.fabs", "tl.npuir_abs"},
    {"Add", "tl.npuir_add"},
    {"Mul", "tl.npuir_mul"},
    {"Sub", "tl.npuir_sub"},
    {"Div", "tl.npuir_div"}
};

class LoopVectorizer : public StmtMutator {
private:
  PrimExpr BuildRegionCall(Buffer buffer, Array<PrimExpr> offset_expr, int region_id, std::vector<int> loop_ranges) {
    PrimExpr buffer_load = BufferLoad(buffer, {offset_expr});
    Array<PrimExpr> args = {
      buffer_load,
      make_const(DataType::Int(32), region_id)
    };
    for (int i = 0; i < loop_ranges.size(); i++) {
      args.push_back(make_const(DataType::Int(32), loop_ranges[i]));
    }
    Op region_op = Op::Get("tl.region");
    return Call(DataType::Handle(), region_op, args);
  }

  Stmt BuildNPUIRBinaryCall(std::string op_name, Buffer buffer_a, Buffer buffer_b, Buffer buffer_c,
                              std::vector<int> offsets, std::vector<int> loop_ranges) {
    Array<PrimExpr> offset_expr;
    for (int i = 0; i < offsets.size(); i++) {
      offset_expr.push_back(make_const(DataType::Int(32), offsets[i]));
    }
    PrimExpr region_a = BuildRegionCall(buffer_a, offset_expr, 1, loop_ranges);
    PrimExpr region_b = BuildRegionCall(buffer_b, offset_expr, 1, loop_ranges);
    PrimExpr region_c = BuildRegionCall(buffer_c, offset_expr, 2, loop_ranges);
    auto op_type = Op::Get(TirOps2NpuirOps[op_name]);

    Array<PrimExpr> args = {region_a, region_b, region_c};
    PrimExpr call = Call(DataType::Void(), op_type, args);
    return Evaluate(call);
  }

  Stmt BuildNpuirBinaryCall(std::string op_name, PrimExpr a, PrimExpr b, PrimExpr c) {
    auto op_type = Op::Get(TirOps2NpuirOps[op_name]);
    Array<PrimExpr> args = {a, b, c};
    PrimExpr call = Call(DataType::Void(), op_type, args);
    return Evaluate(call);
  }

  Stmt BuildNPUIRUnaryCall(std::string op_name, Buffer buffer_in, Buffer buffer_out, std::vector<int> offsets, std::vector<int> loop_ranges) {
    Array<PrimExpr> offset_expr;
    for (int i = 0; i < offsets.size(); i++) {
      offset_expr.push_back(make_const(DataType::Int(32), offsets[i]));
    }
    PrimExpr region_in = BuildRegionCall(buffer_in, offset_expr, 1, loop_ranges);
    PrimExpr region_out = BuildRegionCall(buffer_out, offset_expr, 2, loop_ranges);
    auto op_type = Op::Get(TirOps2NpuirOps[op_name]);

    Array<PrimExpr> args = {region_in, region_out};
    PrimExpr call =  Call(DataType::Void(), op_type, args);
    return Evaluate(call);
  }

  bool IsUnaryOp(const PrimExpr& expr) {
    if (auto call = expr.as<CallNode>()) {
      std::string op_name;

      if (auto* op_ptr = call->op.as<OpNode>()) {
        op_name = op_ptr->name;
        if (op_name == "tir.exp" || op_name == "tir.fabs") {
          return true;
        }
      }
    }
    return false;
  }

  bool IsBinaryOp(const PrimExpr& expr, std::string* op_type, Array<PrimExpr>* operands) {
    if (auto node = expr.as<AddNode>()) {
      if (op_type) *op_type = "Add";
      if (operands) {
        operands->push_back(node->a);
        operands->push_back(node->b);
      }
      return true;
    } else if (auto node = expr.as<MulNode>()) {
      if (op_type) *op_type = "Mul";
      if (operands) {
        operands->push_back(node->a);
        operands->push_back(node->b);
      }
      return true;
    } else if (auto node = expr.as<SubNode>()) {
      if (op_type) *op_type = "Sub";
      if (operands) {
        operands->push_back(node->a);
        operands->push_back(node->b);
      }
      return true;
    } else if (auto node = expr.as<DivNode>()) {
      if (op_type) *op_type = "Div";
      if (operands) {
        operands->push_back(node->a);
        operands->push_back(node->b);
      }
      return true;
    }
    return false;
  }

  bool IsScalar(const PrimExpr& expr) {
    return expr.as<IntImmNode>() || expr.as<FloatImmNode>();
  }

  bool IsLoopInvariant(const Array<PrimExpr>& indices,
                       const std::vector<const VarNode*>& loop_vars) {
    for (const auto& idx : indices) {
      if (auto var = idx.as<VarNode>()) {
        for (auto it = loop_vars.begin(); it != loop_vars.end(); ++it) {
          if (*it == var) {
            return false;
          }
        }
      }
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

  bool HandleSimpleExpression(const std::string& op_name, const Array<PrimExpr>& operands,
                              const Buffer& output_buffer, std::vector<int> loop_ranges,
                              const std::vector<const VarNode*>& loop_vars, Array<Stmt>* statements) {
    auto load_a = operands[0].as<BufferLoadNode>();
    auto load_b = operands[1].as<BufferLoadNode>();
    int size  = loop_ranges[0] * loop_ranges[1];
    int offset = 0;
    // vector OP const-scalar
    if (load_a && IsScalar(operands[1]) || load_b && IsScalar(operands[0])) {
      if (load_a && !IndicesSameAsLoopVars(load_a->indices, loop_vars)) {
        return false;
      }
      if (load_b && !IndicesSameAsLoopVars(load_b->indices, loop_vars)) {
        return false;
      }
      int buffer_idx = 0;
      int scalar_idx = 1;
      if (operands[1].as<BufferLoadNode>() && IsScalar(operands[0])) {
        scalar_idx = 0;
        buffer_idx = 1;
      }
      const BufferLoadNode* node_a = operands[buffer_idx].as<BufferLoadNode>();
      Array<PrimExpr> offset_expr;
      for (int i = 0; i < loop_ranges.size(); i++) {
        offset_expr.push_back(make_const(DataType::Int(32), offset));
      }
      PrimExpr region_a = BuildRegionCall(node_a->buffer, offset_expr, 1, loop_ranges);
      PrimExpr region_c = BuildRegionCall(output_buffer, offset_expr, 2, loop_ranges);
      auto stmt = BuildNpuirBinaryCall(op_name, region_a, operands[scalar_idx], region_c);
      statements->push_back(stmt);
      return true;
    }

    if (load_a && load_b) {
      // lis means loop invariant scalar
      bool vector_lis = !IsLoopInvariant(load_a->indices, loop_vars) && IsLoopInvariant(load_b->indices, loop_vars);
      bool lis_vector = IsLoopInvariant(load_a->indices, loop_vars) && !IsLoopInvariant(load_b->indices, loop_vars);
      bool vector_vector = !IsLoopInvariant(load_a->indices, loop_vars) && !IsLoopInvariant(load_b->indices, loop_vars);
      auto op_type = Op::Get(TirOps2NpuirOps[op_name]);
      
      Buffer buffer_a = load_a->buffer;
      Buffer buffer_b = load_b->buffer;
      PrimExpr offset_expr_b = load_b->indices[0];

      // vector OP lis
      if (vector_lis || lis_vector) {
        if (lis_vector) {
           buffer_a = load_b->buffer;
           buffer_b = load_a->buffer;
           offset_expr_b = load_a->indices[0];
        }
        Array<PrimExpr> offset_expr;
        for (int i = 0; i < loop_ranges.size(); i++) {
          offset_expr.push_back(make_const(DataType::Int(32), offset));
        }
        PrimExpr region_a = BuildRegionCall(buffer_a, offset_expr, 1, loop_ranges);
        PrimExpr region_b = BuildRegionCall(buffer_b, {offset_expr_b}, 1, {1});
        PrimExpr region_c = BuildRegionCall(output_buffer, offset_expr, 2, loop_ranges);

        Array<PrimExpr> args = {region_a, region_b, region_c};
        PrimExpr call = Call(DataType::Void(), op_type, args);
        auto stmt = Evaluate(call);
        statements->push_back(stmt);
        return true;
      }
      // vector OP vector
      if (vector_vector) {
        Array<PrimExpr> offset_expr;
        for (int i = 0; i < loop_ranges.size(); i++) {
          offset_expr.push_back(make_const(DataType::Int(32), offset));
        }
        PrimExpr region_a = BuildRegionCall(buffer_a, offset_expr, 1, loop_ranges);
        PrimExpr region_b = BuildRegionCall(buffer_b, offset_expr, 1, loop_ranges);
        PrimExpr region_c = BuildRegionCall(output_buffer, offset_expr, 2, loop_ranges);

        Array<PrimExpr> args = {region_a, region_b, region_c};
        PrimExpr call = Call(DataType::Void(), op_type, args);
        auto stmt = Evaluate(call);
        statements->push_back(stmt);
        return true;
      }
    }
    return false;
  }


  bool HandleUnaryExpression(const PrimExpr& expr, const Buffer& output_buffer, std::vector<int> loop_ranges,
                             const std::vector<const VarNode*>& loop_vars, Array<Stmt>* statements) {
    auto* call = expr.as<CallNode>();
    auto* op_ptr = call->op.as<OpNode>();
    std::string op_name = op_ptr->name;
    std::vector<int> offsets;
    int offset = 0;
    for (int i = 0; i < loop_ranges.size(); i++) {
        offsets.push_back(offset);
    }
    if (auto load = call->args[0].as<BufferLoadNode>()) {
      if (!IndicesSameAsLoopVars(load->indices, loop_vars)) {
        return false;
      }
      // If indices are consistent with loop vars, it means the address offset of the buffer is 0
      Stmt statement = BuildNPUIRUnaryCall(op_name, load->buffer, output_buffer, offsets, loop_ranges);
      statements->push_back(statement);
      return true;
    } else {
      if (DecomposeExpression(call->args[0], output_buffer, loop_ranges, loop_vars, statements)) {
        int offset = 0;
        // For an unary op, its input and output can share the same buffer.
        Stmt statement = BuildNPUIRUnaryCall(op_name, output_buffer, output_buffer, offsets, loop_ranges);
        statements->push_back(statement);
        return true;
      }
    }
    return false;
  }

  bool HandleBinaryExpression(std::string op_type, const Array<PrimExpr>& operands,
                              const Buffer& output_buffer, std::vector<int> loop_ranges,
                              const std::vector<const VarNode*>& loop_vars, Array<Stmt>* statements) {
    bool left_is_simple = operands[0].as<BufferLoadNode>() || IsScalarLike(operands[0], loop_vars);
    bool right_is_simple = operands[1].as<BufferLoadNode>() || IsScalarLike(operands[1], loop_vars);
    std::vector<int> offsets;
    int offset = 0;
    for (int i = 0; i < loop_ranges.size(); i++) {
        offsets.push_back(offset);
    }
    if (left_is_simple && right_is_simple) {
      return HandleSimpleExpression(op_type, operands, output_buffer, loop_ranges, loop_vars, statements);
    } else if (!left_is_simple && right_is_simple) {

      if (DecomposeExpression(operands[0], output_buffer, loop_ranges, loop_vars, statements)) {
        auto load_b = operands[1].as<BufferLoadNode>();
        Buffer buffer_b = load_b->buffer;
        Stmt statement = BuildNPUIRBinaryCall(op_type, output_buffer, buffer_b, output_buffer, offsets, loop_ranges);
        statements->push_back(statement);
        return true;
      }
    } else if (left_is_simple && !right_is_simple) {
      if (DecomposeExpression(operands[1], output_buffer, loop_ranges, loop_vars, statements)) {
        auto load_a = operands[0].as<BufferLoadNode>();
        Stmt statement = BuildNPUIRBinaryCall(op_type, load_a->buffer, output_buffer, output_buffer, offsets, loop_ranges);
        statements->push_back(statement);
        return true;
      }
    }
    return false;
  }

  bool DecomposeExpression(const PrimExpr& expr, const Buffer& output_buffer, std::vector<int> loop_ranges,
                           const std::vector<const VarNode*>& loop_vars, Array<Stmt>* statements) {
    if(IsUnaryOp(expr)) {
      return HandleUnaryExpression(expr, output_buffer, loop_ranges, loop_vars, statements);
    }

    std::string op_type;
    Array<PrimExpr> operands;
    if (IsBinaryOp(expr, &op_type, &operands)) {
      return HandleBinaryExpression(op_type, operands, output_buffer, loop_ranges, loop_vars, statements);
    }
    return false;
  }

  Stmt VectorizeSingleStatement(const ForNode *op, std::vector<int> loop_ranges, std::vector<const VarNode*> loop_vars) {
    const BufferStoreNode* store = op->body.as<BufferStoreNode>();
    // Now only a single-layer for loop is supported, that is, one-dimensional vectorization.
    if (store->indices.size() != loop_vars.size()) {
      return StmtMutator::VisitStmt_(op);
    }

    for(int i = 0; i < loop_vars.size(); i++) {
      PrimExpr index = store->indices[i];
      const VarNode* var = index.as<VarNode>();
      if (!var || var != loop_vars[i]) {
        return StmtMutator::VisitStmt_(op);
      }
    }

    Array<Stmt> statements;
    bool ret = DecomposeExpression(store->value, store->buffer, loop_ranges, loop_vars, &statements);
    return ret ?  SeqStmt::Flatten(statements) : StmtMutator::VisitStmt_(op);
  }


  Stmt VisitStmt_(const ForNode *op) final {
    if (op->kind != ForKind::kParallel) {
      return StmtMutator::VisitStmt_(op);
    }

    std::vector<int> loop_ranges;
    std::vector<const VarNode*> loop_vars;
    const IntImmNode* loop_extent = op->extent.as<IntImmNode>();
    if (!loop_extent) {
      return StmtMutator::VisitStmt_(op);
    }
    loop_ranges.push_back(loop_extent->value);
    loop_vars.push_back(op->loop_var.get());
    if (op->body.as<BufferStoreNode>()) {
      auto vectorized = VectorizeSingleStatement(op, loop_ranges, loop_vars);
      if (vectorized.defined()) {
        return vectorized;
      } else {
        return StmtMutator::VisitStmt_(op);
      }
    }

    const auto* inner_for = op->body.as<ForNode>();
    if (!inner_for || inner_for->kind != ForKind::kParallel) {
      return StmtMutator::VisitStmt_(op);
    }

    const IntImmNode* inner_loop_extent = inner_for->extent.as<IntImmNode>();
    if (!inner_loop_extent) {
      return StmtMutator::VisitStmt_(op);
    }

    if (!inner_for->body.as<BufferStoreNode>()) {
      return StmtMutator::VisitStmt_(op);
    }

    loop_ranges.push_back(inner_loop_extent->value);
    loop_vars.push_back(inner_for->loop_var.get());

    auto vectorized = VectorizeSingleStatement(inner_for, loop_ranges, loop_vars);
    if (vectorized.defined()) {
      return vectorized;
    }
    return StmtMutator::VisitStmt_(op);
  }
};

using namespace tir::transform;

tvm::transform::Pass NpuLoopVectorize() {
  auto pass_func = [=](PrimFunc f,IRModule m,PassContext ctx) {
    auto *new_pf = f.CopyOnWrite();
    new_pf->body = LoopVectorizer()(std::move(new_pf->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.NpuLoopVectorize", {});
}

TVM_REGISTER_GLOBAL("tl.transform.NpuLoopVectorize").set_body_typed(NpuLoopVectorize);

} // namespace tl
} // namespace tvm
