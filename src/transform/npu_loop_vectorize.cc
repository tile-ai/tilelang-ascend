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

class LoopDecompose : public StmtMutator {
private:
  // Temporary buffers of subexpression results when processing compound expressions
  std::vector<Buffer> tmp_buffers;
  // Recursion depth when processing compound expressions recursively
  int depth = 0;

  inline static std::unordered_map<std::string, std::string> TirOps2NpuirOps = {
    {"tir.exp", "tl.npuir_exp"},
    {"tir.fabs", "tl.npuir_abs"},
    {"Add", "tl.npuir_add"},
    {"Mul", "tl.npuir_mul"},
    {"Sub", "tl.npuir_sub"},
    {"Div", "tl.npuir_div"},
  };

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

  BufferAccessInfo CreateTempBuffer(const BufferAccessInfo& output_buffer) {
    static int tmp_id = 0;
    Buffer ref = output_buffer.buffer;
    DataType dtype = ref->dtype;
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
    Buffer buf = Buffer(
      Var(name, PointerType(PrimType(dtype), "shared")),
      dtype, ref->shape, {}, PrimExpr(0), name, 0, 0, kDefault
    );

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

  Stmt BuildNPUIRBinaryCall(std::string op_name,
                            const BufferAccessInfo& buffer_a,
                            const BufferAccessInfo& buffer_b,
                            const BufferAccessInfo& buffer_c) {
    PrimExpr region_a = BuildRegionCall(buffer_a, 1, 1);
    PrimExpr region_b = BuildRegionCall(buffer_b, 1, 1);
    PrimExpr region_c = BuildRegionCall(buffer_c, 2, 1);
    auto op_type = Op::Get(TirOps2NpuirOps[op_name]);

    Array<PrimExpr> args = {region_a, region_b, region_c};
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
    PrimExpr call = Call(DataType::Void(), op_type, args);
    return Evaluate(call);
  }

  Stmt BuildNpuirBinaryCall(std::string op_name, PrimExpr a, PrimExpr b, PrimExpr c) {
    auto op_type = Op::Get(TirOps2NpuirOps[op_name]);
    Array<PrimExpr> args = {a, b, c};
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

  bool HandleSimpleExpression(const std::string& op_name, const Array<PrimExpr>& operands,
                              const BufferAccessInfo& output_buffer, std::vector<BufferAccessInfo>* tmp_bufs,
                              const std::vector<const VarNode*>& loop_vars, Array<Stmt>* statements) {
    auto load_a = operands[0].as<BufferLoadNode>();
    auto load_b = operands[1].as<BufferLoadNode>();
    // vector OP const-scalar
    if ((load_a && IsScalar(operands[1])) || (load_b && IsScalar(operands[0]))) {
      if (load_a && !ValidBufferIndices(load_a->indices)) {
        depth = 0;
        return false;
      }
      if (load_b && !ValidBufferIndices(load_b->indices)) {
        depth = 0;
        return false;
      }
      int buffer_idx = 0;
      int scalar_idx = 1;
      if (load_b && IsScalar(operands[0])) {
        scalar_idx = 0;
        buffer_idx = 1;
      }

      BufferAccessInfo output = output_buffer;
      if(depth > 1) {
        auto load_node = operands[buffer_idx].as<BufferLoadNode>();
        BufferAccessInfo info {load_node->buffer, load_node->indices};
        output = CreateTempBuffer(info);
        tmp_bufs->push_back(output);
      }

      PrimExpr region_a = BuildRegionCall(ExtractBufferAccessInfo(operands[buffer_idx]), 1, 1);
      PrimExpr region_c = BuildRegionCall(output, 2, 1);
      auto stmt = BuildNpuirBinaryCall(op_name, region_a, operands[scalar_idx], region_c);
      statements->push_back(stmt);
      depth--;
      return true;
    }

    if (load_a && load_b) {
      // lis means loop invariant scalar
      bool vector_lis = !IsLoopInvariant(load_a->indices, loop_vars) && IsLoopInvariant(load_b->indices, loop_vars);
      bool lis_vector = IsLoopInvariant(load_a->indices, loop_vars) && !IsLoopInvariant(load_b->indices, loop_vars);
      bool vector_vector = !IsLoopInvariant(load_a->indices, loop_vars) && !IsLoopInvariant(load_b->indices, loop_vars);
      auto op_type = Op::Get(TirOps2NpuirOps[op_name]);
      int offset = 0;
      auto buffer_a = ExtractBufferAccessInfo(operands[0]);
      auto buffer_b = ExtractBufferAccessInfo(operands[1]);

      BufferAccessInfo output = output_buffer;
      if(depth > 1) {
        auto load_node = operands[0].as<BufferLoadNode>();
        BufferAccessInfo info {load_node->buffer, load_node->indices};
        output = CreateTempBuffer(info);
        tmp_bufs->push_back(output);
      }

      if (lis_vector) {
        buffer_a = ExtractBufferAccessInfo(operands[1]);
        buffer_b = ExtractBufferAccessInfo(operands[0]);
      }

      if (vector_lis || lis_vector || vector_vector) {
        PrimExpr offset_expr = {make_const(DataType::Int(32), offset)};
        PrimExpr region_a = BuildRegionCall(buffer_a, 1, 1);
        PrimExpr region_b = BuildRegionCall(buffer_b, 1, 1);
        PrimExpr region_c = BuildRegionCall(output, 2, 1);
        Array<PrimExpr> args = {region_a, region_b, region_c};
        PrimExpr call = Call(DataType::Void(), op_type, args);
        auto stmt = Evaluate(call);
        statements->push_back(stmt);
        depth--;
        return true;
      }
    }
    depth = 0;
    return false;
  }

  bool HandleUnaryExpression(const PrimExpr& expr, const BufferAccessInfo& output_buffer, std::vector<BufferAccessInfo>* tmp_bufs,
                             const std::vector<const VarNode*>& loop_vars, Array<Stmt>* statements) {
    auto* call = expr.as<CallNode>();
    auto* op_ptr = call->op.as<OpNode>();
    std::string op_name = op_ptr->name;
    if (auto load = call->args[0].as<BufferLoadNode>()) {
      if (!ValidBufferIndices(load->indices)) {
        depth = 0;
        return false;
      }
      // If indices are consistent with loop vars, it means the address offset of the buffer is 0
      int offset = 0;
      Stmt statement = BuildNPUIRUnaryCall(op_name, ExtractBufferAccessInfo(call->args[0]), output_buffer, offset);
      statements->push_back(statement);
      depth--;
      return true;
    } else {
      if (DecomposeExpression(call->args[0], output_buffer, tmp_bufs, loop_vars, statements)) {
        int offset = 0;
        // For an unary op, its input and output can share the same buffer.
        Stmt statement = BuildNPUIRUnaryCall(op_name, tmp_bufs->back(), output_buffer, offset);
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
          output = CreateTempBuffer(last_buf);
          tmp_bufs->push_back(output);
        }
        Stmt statement;
        if (IsScalar(operands[1])) {
          statement = BuildNPUIRBinaryCall(op_type, last_buf, operands[1], output);
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
          output = CreateTempBuffer(last_buf);
          tmp_bufs->push_back(output);
        }
        Stmt statement;
        if (IsScalar(operands[0])) {
          statement = BuildNPUIRBinaryCall(op_type, last_buf, operands[0], output);
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
        output = CreateTempBuffer(left);
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

  bool DecomposeExpression(const PrimExpr& expr, const BufferAccessInfo& output_buffer, std::vector<BufferAccessInfo>* tmp_bufs,
                           const std::vector<const VarNode*>& loop_vars, Array<Stmt>* statements) {
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
    "tl.npuir_exp",
    "tl.npuir_abs",
    "tl.npuir_add",
    "tl.npuir_mul",
    "tl.npuir_sub",
    "tl.npuir_div",
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
  * Returns -1 if not found, -2 if found in multiple offsets.
  */
  int FindLoopVarInOffsets(const Array<PrimExpr>& offsets, const VarNode* loop_var) {
    int found_dim = -1;
    arith::Analyzer analyzer;

    for (int i = 0; i < static_cast<int>(offsets.size()); ++i) {
      class LoopVarVisitor : public ExprVisitor {
      public:
        const VarNode* target_var;
        bool found = false;

        explicit LoopVarVisitor(const VarNode* var) : target_var(var) {}

        void VisitExpr_(const VarNode* op) override {
          if (op == target_var) found = true;
          ExprVisitor::VisitExpr_(op);
        }
      } visitor(loop_var);

      visitor(offsets[i]);

      if (visitor.found) {
        if (found_dim == -1) {
          found_dim = i;
        } else {
          return -2;
        }
      }
    }

    return found_dim;
  }

  bool RegionInLoopEnableVectorize(const PrimExpr& expr, const VarNode* loop_var) {
    // 1. must be a CallNode
    const auto* call = expr.as<CallNode>();
    if (!call) return false;

    // 2. must be tl.region
    const auto* op_node = call->op.as<tvm::OpNode>();
    if (!op_node) return false;
    tvm::Op op = tvm::GetRef<tvm::Op>(op_node);

    static const auto* region_op = Op::Get("tl.region").as<OpNode>();
    if (op.get() != region_op) return false;

    // 3. check the number of args and extract attributes
    if (call->args.size() < 3) return false;

    const auto* buffer_load = call->args[0].as<BufferLoadNode>();
    if (!buffer_load) return false;

    Buffer buffer = buffer_load->buffer;
    Array<PrimExpr> offsets = buffer_load->indices;

    // 4. check the shapes
    int d = buffer->shape.size();
    if (offsets.size() != d) return false;
    if (call->args.size() != d + 2) return false;

    // 5. check the loop var
    int loop_var_dim = FindLoopVarInOffsets(offsets, loop_var);
    arith::Analyzer analyzer;

    if (loop_var_dim == -1) {  // -1 -> not found -> invariant -> enable to vectorize
      return true;
    } else if (loop_var_dim == -2) {  // -2 -> multiple found -> dispersed mem access -> disable to vectorize
      return false;
    }

    // check the sizes to ensure the continuity of mem access
    for (int i = 0; i < d; ++i) {
      const PrimExpr& size = call->args[i + 2];

      if (i <= loop_var_dim) {
        if (!analyzer.CanProveEqual(size, 1)) {
          return false;
        }
      } else {
        if (!analyzer.CanProveEqual(size, buffer->shape[i])) {
          return false;
        }

        if (!analyzer.CanProveEqual(offsets[i], 0)) {
          return false;
        }
      }
    }
    return true;
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

  // Vectorize a region which belongs to a call statement
  PrimExpr VectorizeRegionInLoop(const PrimExpr& expr, const VarNode* loop_var, size_t start, size_t loop_count) {
    const auto* call = expr.as<CallNode>();
    const auto* buffer_load = call->args[0].as<BufferLoadNode>();

    Buffer buffer = buffer_load->buffer;
    Array<PrimExpr> offsets = buffer_load->indices;
    std::vector<PrimExpr> offsets_vec;
    for (const auto& offset : offsets) {
      offsets_vec.push_back(offset);
    }

    int regionId = static_cast<int>(*TryGetConstIntValue(call->args[1]));
    int dim = buffer->shape.size();
    std::vector<size_t> size_vec;
    for (int i = 2; i < dim + 2; ++i) {
      size_vec.push_back(static_cast<size_t>(*TryGetConstIntValue(call->args[i])));
    }

    int loop_var_dim = FindLoopVarInOffsets(offsets, loop_var);
    if (loop_var_dim >= 0) {
      offsets_vec[loop_var_dim] = make_const(DataType::Int(32), start);
      size_vec[loop_var_dim] = loop_count;
    }

    return BuildRegionCall(buffer, offsets_vec, regionId, size_vec);
  }

  Stmt VectorizeForBody(const ForNode* forNode, const Stmt& stmt) {
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

    // Enable vectorization only when regions satisfy specific conditions
    bool flag_region_vectorizable = true;
    auto loop_var_node_ptr = forNode->loop_var.get();
    for (const auto& region : call->args) {
      if (IsScalar(region)) continue;
      if (!RegionInLoopEnableVectorize(region, loop_var_node_ptr)) {
        flag_region_vectorizable = false;
      }
    }
    if (!flag_region_vectorizable) return SplitStmtToIndependentForNode(forNode, stmt);

    // min_value & loop_extent_value must exist
    auto loop_min_value = TryGetConstIntValue(forNode->min);
    auto loop_extent_value = TryGetConstIntValue(forNode->extent);
    if (!loop_min_value || !loop_extent_value) {
      return SplitStmtToIndependentForNode(forNode, stmt);
    }

    // All conditions satisfied, start vectorization
    std::vector<PrimExpr> new_regions;
    for (const auto& region : call->args) {
      if (IsScalar(region)) {
        new_regions.push_back(region);
        continue;
      }
      new_regions.push_back(VectorizeRegionInLoop(region, loop_var_node_ptr, *loop_min_value, *loop_extent_value));
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
