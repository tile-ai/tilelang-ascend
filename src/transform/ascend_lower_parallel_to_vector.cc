// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file ascend_lower_parallel_to_vector.cc
 * \brief Lower parallel loops to vector instructions for ascend.
 */

#include "arith/ir_mutator_with_analyzer.h"
#include "tir/analysis/var_use_def_analysis.h"

#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include "../op/builtin.h"
#include "./common/collector.h"

namespace tvm {
namespace tl {

using namespace tir;

class AscendLowerParallelToVector : public arith::IRMutatorWithAnalyzer {
 public:
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    AscendLowerParallelToVector substituter(&analyzer);
    PrimFuncNode* fptr = f.CopyOnWrite();
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

 private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  const VarNode* vector_dim_var_ = nullptr;

  class ReplaceVarExpr : public StmtExprMutator {
   public:
    const VarNode* from_;
    Var to_;
    explicit ReplaceVarExpr(const VarNode* from, Var to) : from_(from), to_(to) {}

    PrimExpr VisitExpr_(const VarNode* op) override {
      if (op == from_) {
        return to_;
      }
      return GetRef<PrimExpr>(op);
    }

    Stmt VisitStmt_(const ForNode* op) override {
      return StmtExprMutator::VisitStmt_(op);
    }
  };

  Stmt VisitStmt_(const ForNode* op) override {
    if (op->kind == ForKind::kParallel) {
      if (auto inner_for = op->body.as<ForNode>()) {
        if (inner_for->kind == ForKind::kParallel) {
          if (auto store = inner_for->body.as<BufferStoreNode>()) {
            PrimExpr total_elements = op->extent * inner_for->extent;
            std::unordered_set<const VarNode*> parallel_vars;
            parallel_vars.insert(op->loop_var.get());
            parallel_vars.insert(inner_for->loop_var.get());

            const VarNode* old_vector_dim = vector_dim_var_;
            vector_dim_var_ = inner_for->loop_var.get();

            auto vectorized =
                TryVectorizeBufferStore(store, total_elements, parallel_vars,
                                        /*has_outer_serial=*/false);

            vector_dim_var_ = old_vector_dim;

            if (vectorized.defined()) {
              return vectorized;
            }
          }
        }
      }
      else if (auto store = op->body.as<BufferStoreNode>()) {
        PrimExpr total_elements = op->extent;
        std::unordered_set<const VarNode*> parallel_vars;
        parallel_vars.insert(op->loop_var.get());

        const VarNode* old_vector_dim = vector_dim_var_;
        vector_dim_var_ = op->loop_var.get();

        auto vectorized =
            TryVectorizeBufferStore(store, total_elements, parallel_vars,
                                    /*has_outer_serial=*/false);

        vector_dim_var_ = old_vector_dim;

        if (vectorized.defined()) {
          return vectorized;
        }
      }
    }

    if (op->kind == ForKind::kSerial) {
      if (auto inner_for = op->body.as<ForNode>()) {
        if (inner_for->kind == ForKind::kParallel) {
          if (auto store = inner_for->body.as<BufferStoreNode>()) {
            PrimExpr total_elements = op->extent * inner_for->extent;
            std::unordered_set<const VarNode*> parallel_vars;
            parallel_vars.insert(inner_for->loop_var.get());

            const VarNode* old_vector_dim = vector_dim_var_;
            vector_dim_var_ = inner_for->loop_var.get();

            auto vectorized =
                TryVectorizeBufferStore(store, total_elements, parallel_vars,
                                        /*has_outer_serial=*/true);

            vector_dim_var_ = old_vector_dim;

            if (vectorized.defined()) {
              auto op_copy = make_object<ForNode>(*op);
              op_copy->body = vectorized;
              return Stmt(op_copy);
            }
          }
        }
      }
    }

    return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt TryVectorizeBufferStore(const BufferStoreNode* store,
                               PrimExpr total_elements,
                               const std::unordered_set<const VarNode*>& parallel_vars,
                               bool has_outer_serial) {
    int64_t element_count = 0;
    if (!TryGetElementCount(total_elements, &element_count)) {
      return Stmt();
    }

    Buffer output_buffer = store->buffer;

    bool is_2d_row_major = false;
    const VarNode* outer_index_var = nullptr;
    PrimExpr outer_index_expr;
    PrimExpr inner_index_expr;

    if (output_buffer->shape.size() == 2 && store->indices.size() == 2) {
      outer_index_expr = store->indices[0];
      inner_index_expr = store->indices[1];

      const VarNode* outer_var = outer_index_expr.as<VarNode>();
      const VarNode* inner_var = inner_index_expr.as<VarNode>();

      if (outer_var != nullptr && inner_var != nullptr &&
          vector_dim_var_ != nullptr && inner_var == vector_dim_var_) {
        is_2d_row_major = true;
        outer_index_var = outer_var;
      }
    }

    if (is_2d_row_major) {
      int64_t inner_vec_len = 0;
      if (auto imm = output_buffer->shape[1].as<IntImmNode>()) {
        inner_vec_len = imm->value;
      }

      if (inner_vec_len > 0 && element_count % inner_vec_len == 0) {
        int64_t outer_extent = element_count / inner_vec_len;

        PrimExpr output_offset =
            CalculateBufferOffset(store->indices, output_buffer, parallel_vars);

        Array<Stmt> row_stmts;
        bool success = DecomposeExpression(store->value, output_buffer,
                                           output_offset, inner_vec_len,
                                           parallel_vars, &row_stmts);
        if (!success || row_stmts.empty()) {
          return Stmt();
        }

        Stmt row_body;
        if (row_stmts.size() == 1) {
          row_body = row_stmts[0];
        } else {
          row_body = SeqStmt::Flatten(row_stmts);
        }

        if (has_outer_serial) {
          return row_body;
        }

        Var outer_var("outer_broadcast", DataType::Int(32));
        if (outer_index_var != nullptr) {
          ReplaceVarExpr replacer(outer_index_var, outer_var);
          row_body = replacer(row_body);
        }

        Stmt for_stmt =
            For(outer_var,
                IntImm(DataType::Int(32), 0),
                IntImm(DataType::Int(32), outer_extent),
                ForKind::kSerial,
                row_body);

        return for_stmt;
      }
    }

    Array<Stmt> statements;
    PrimExpr output_offset =
        CalculateBufferOffset(store->indices, output_buffer, parallel_vars);

    bool success = DecomposeExpression(store->value, output_buffer, output_offset,
                                       element_count, parallel_vars, &statements);

    if (success && statements.size() > 0) {
      if (statements.size() == 1) {
        return statements[0];
      } else {
        return SeqStmt::Flatten(statements);
      }
    }

    return Stmt();
  }

  bool DecomposeExpression(const PrimExpr& expr,
                           const Buffer& output_buffer,
                           const PrimExpr& output_offset,
                           int64_t element_count,
                           const std::unordered_set<const VarNode*>& parallel_vars,
                           Array<Stmt>* statements) {
    std::string unary_op_type;
    Optional<Buffer> unary_input_buffer;
    PrimExpr unary_input_offset;

    if (IsUnaryOp(expr, &unary_op_type, &unary_input_buffer, &unary_input_offset,
                        parallel_vars)) {
      auto stmt = GenerateUnaryVectorCall(unary_op_type, output_buffer, output_offset,
                                          unary_input_buffer.value(), unary_input_offset,
                                          element_count);
      statements->push_back(stmt);
      return true;
    }

    std::string op_type;
    Array<PrimExpr> operands;

    if (!IsBinaryOp(expr, &op_type, &operands)) {
      return false;
    }

    ICHECK_EQ(operands.size(), 2);

    bool left_is_simple =
        operands[0].as<BufferLoadNode>() || IsScalarLike(operands[0], parallel_vars);
    bool right_is_simple =
        operands[1].as<BufferLoadNode>() || IsScalarLike(operands[1], parallel_vars);

    bool left_is_complex =
        IsBinaryOp(operands[0], nullptr, nullptr) ||
        IsUnaryOp(operands[0], nullptr, nullptr, nullptr, parallel_vars);
    bool right_is_complex =
        IsBinaryOp(operands[1], nullptr, nullptr) ||
        IsUnaryOp(operands[1], nullptr, nullptr, nullptr, parallel_vars);

    if (left_is_simple && right_is_simple) {
      return HandleSimpleCase(op_type, operands, output_buffer, output_offset,
                              element_count, parallel_vars, statements);
    }

    if (left_is_simple && right_is_complex) {
      return HandleLeftSimpleRightComplex(op_type, operands, output_buffer, output_offset,
                                          element_count, parallel_vars, statements);
    }

    if (left_is_complex && right_is_simple) {
      return HandleLeftComplexRightSimple(op_type, operands, output_buffer, output_offset,
                                          element_count, parallel_vars, statements);
    }

    if (left_is_complex && right_is_complex) {
      throw std::runtime_error(
          "Expression with complex operations on both sides is not supported: "
          "multiple temporary buffers are required but not available");
    }

    return false;
  }

  bool IsUnaryOp(const PrimExpr& expr, std::string* op_type,
                 Optional<Buffer>* input_buffer, PrimExpr* input_offset,
                 const std::unordered_set<const VarNode*>& parallel_vars) {
    if (auto call = expr.as<CallNode>()) {
      std::string op_name;

      if (auto* op_ptr = call->op.as<OpNode>()) {
        op_name = op_ptr->name;
      } else {
        return false;
      }

      std::string ascend_op;

      if (op_name == "tir.exp") {
        ascend_op = "AscendC::Exp";
      } else if (op_name == "tir.log") {
        ascend_op = "AscendC::Ln";
      } else if (op_name == "tir.sqrt") {
        ascend_op = "AscendC::Sqrt";
      } else if (op_name == "tir.rsqrt") {
        ascend_op = "AscendC::Rsqrt";
      } else if (op_name == "tir.fabs") {
        ascend_op = "AscendC::Abs";
      } else {
        if (call->op.same_as(builtin::bitwise_not())) {
          ascend_op = "AscendC::Not";
        } else {
          return false;
        }
      }

      if (op_type) *op_type = ascend_op;

      if (call->args.size() >= 1) {
        if (auto load = call->args[0].as<BufferLoadNode>()) {
          if (input_buffer) *input_buffer = load->buffer;
          if (input_offset)
            *input_offset =
                CalculateBufferOffset(load->indices, load->buffer, parallel_vars);
          return true;
        }
      }
    }

    if (auto max_node = expr.as<MaxNode>()) {
      if (IsZero(max_node->a)) {
        if (op_type) *op_type = "AscendC::Relu";
        if (auto load = max_node->b.as<BufferLoadNode>()) {
          if (input_buffer) *input_buffer = load->buffer;
          if (input_offset)
            *input_offset =
                CalculateBufferOffset(load->indices, load->buffer, parallel_vars);
          return true;
        }
      }
      if (IsZero(max_node->b)) {
        if (op_type) *op_type = "AscendC::Relu";
        if (auto load = max_node->a.as<BufferLoadNode>()) {
          if (input_buffer) *input_buffer = load->buffer;
          if (input_offset)
            *input_offset =
                CalculateBufferOffset(load->indices, load->buffer, parallel_vars);
          return true;
        }
      }
    }
    return false;
  }

  bool HandleSimpleCase(const std::string& op_type,
                        const Array<PrimExpr>& operands,
                        const Buffer& output_buffer,
                        const PrimExpr& output_offset,
                        int64_t element_count,
                        const std::unordered_set<const VarNode*>& parallel_vars,
                        Array<Stmt>* statements) {
    Buffer left_buffer;
    PrimExpr left_offset;

    if (auto load = operands[0].as<BufferLoadNode>()) {
      left_buffer = load->buffer;
      left_offset =
          CalculateBufferOffset(load->indices, left_buffer, parallel_vars);
    } else {
      return false;
    }

    if (auto load = operands[1].as<BufferLoadNode>()) {
      if (IsScalarAccess(load->indices, parallel_vars)) {
        PrimExpr scalar_offset =
            CalculateBufferOffset(load->indices, load->buffer, parallel_vars);
        auto stmt = GenerateBufferScalarVectorCall(
            op_type, output_buffer, output_offset, left_buffer, left_offset,
            load->buffer, scalar_offset, element_count);
        statements->push_back(stmt);
        return true;
      } else {
        PrimExpr right_offset =
            CalculateBufferOffset(load->indices, load->buffer, parallel_vars);
        auto stmt = GenerateBinaryVectorCall(
            op_type, output_buffer, output_offset, left_buffer, left_offset,
            load->buffer, right_offset, element_count);
        statements->push_back(stmt);
        return true;
      }

    } else if (IsScalar(operands[1])) {
      auto stmt = GenerateScalarVectorCall(op_type, output_buffer, output_offset,
                                           left_buffer, left_offset,
                                           operands[1], element_count);
      statements->push_back(stmt);
      return true;
    }

    return false;
  }

  bool HandleLeftSimpleRightComplex(const std::string& op_type,
                                    const Array<PrimExpr>& operands,
                                    const Buffer& output_buffer,
                                    const PrimExpr& output_offset,
                                    int64_t element_count,
                                    const std::unordered_set<const VarNode*>& parallel_vars,
                                    Array<Stmt>* statements) {
    Buffer left_buffer;
    PrimExpr left_offset;

    if (auto load = operands[0].as<BufferLoadNode>()) {
      left_buffer = load->buffer;
      left_offset =
          CalculateBufferOffset(load->indices, left_buffer, parallel_vars);
    } else {
      return false;
    }

    if (!DecomposeExpression(operands[1], output_buffer, output_offset,
                             element_count, parallel_vars, statements)) {
      return false;
    }

    auto stmt = GenerateBinaryVectorCall(op_type, output_buffer, output_offset,
                                         left_buffer, left_offset, output_buffer,
                                         output_offset, element_count);
    statements->push_back(stmt);
    return true;
  }

  bool HandleLeftComplexRightSimple(const std::string& op_type,
                                    const Array<PrimExpr>& operands,
                                    const Buffer& output_buffer,
                                    const PrimExpr& output_offset,
                                    int64_t element_count,
                                    const std::unordered_set<const VarNode*>& parallel_vars,
                                    Array<Stmt>* statements) {
    if (!DecomposeExpression(operands[0], output_buffer, output_offset,
                             element_count, parallel_vars, statements)) {
      return false;
    }

    if (auto load = operands[1].as<BufferLoadNode>()) {
      if (IsScalarAccess(load->indices, parallel_vars)) {
        PrimExpr scalar_offset =
            CalculateBufferOffset(load->indices, load->buffer, parallel_vars);
        auto stmt = GenerateBufferScalarVectorCall(
            op_type, output_buffer, output_offset, output_buffer, output_offset,
            load->buffer, scalar_offset, element_count);
        statements->push_back(stmt);
        return true;
      } else {
        PrimExpr right_offset =
            CalculateBufferOffset(load->indices, load->buffer, parallel_vars);
        auto stmt = GenerateBinaryVectorCall(
            op_type, output_buffer, output_offset, output_buffer, output_offset,
            load->buffer, right_offset, element_count);
        statements->push_back(stmt);
        return true;
      }

    } else if (IsScalar(operands[1])) {
      auto stmt = GenerateScalarVectorCall(op_type, output_buffer, output_offset,
                                           output_buffer, output_offset,
                                           operands[1], element_count);
      statements->push_back(stmt);
      return true;
    }

    return false;
  }

  bool IsBinaryOp(const PrimExpr& expr, std::string* op_type,
                  Array<PrimExpr>* operands) {
    if (auto node = expr.as<AddNode>()) {
      if (op_type) *op_type = "AscendC::Add";
      if (operands) {
        operands->push_back(node->a);
        operands->push_back(node->b);
      }
      return true;
    }
    if (auto node = expr.as<SubNode>()) {
      if (op_type) *op_type = "AscendC::Sub";
      if (operands) {
        operands->push_back(node->a);
        operands->push_back(node->b);
      }
      return true;
    }
    if (auto node = expr.as<MulNode>()) {
      if (op_type) *op_type = "AscendC::Mul";
      if (operands) {
        operands->push_back(node->a);
        operands->push_back(node->b);
      }
      return true;
    }
    if (auto node = expr.as<DivNode>()) {
      if (op_type) *op_type = "AscendC::Div";
      if (operands) {
        operands->push_back(node->a);
        operands->push_back(node->b);
      }
      return true;
    }
    if (auto node = expr.as<MinNode>()) {
      if (op_type) *op_type = "AscendC::Min";
      if (operands) {
        operands->push_back(node->a);
        operands->push_back(node->b);
      }
      return true;
    }
    if (auto node = expr.as<MaxNode>()) {
      if (op_type) *op_type = "AscendC::Max";
      if (operands) {
        operands->push_back(node->a);
        operands->push_back(node->b);
      }
      return true;
    }

    if (auto call = expr.as<CallNode>()) {
      if (call->op.same_as(builtin::bitwise_and())) {
        if (op_type) *op_type = "AscendC::And";
        if (operands && call->args.size() == 2) {
          operands->push_back(call->args[0]);
          operands->push_back(call->args[1]);
        }
        return true;
      }
      if (call->op.same_as(builtin::bitwise_or())) {
        if (op_type) *op_type = "AscendC::Or";
        if (operands && call->args.size() == 2) {
          operands->push_back(call->args[0]);
          operands->push_back(call->args[1]);
        }
        return true;
      }
      if (call->op.same_as(builtin::shift_left())) {
        if (op_type) *op_type = "AscendC::ShiftLeft";
        if (operands && call->args.size() == 2) {
          operands->push_back(call->args[0]);
          operands->push_back(call->args[1]);
        }
        return true;
      }
      if (call->op.same_as(builtin::shift_right())) {
        if (op_type) *op_type = "AscendC::ShiftRight";
        if (operands && call->args.size() == 2) {
          operands->push_back(call->args[0]);
          operands->push_back(call->args[1]);
        }
        return true;
      }
    }

    return false;
  }

  Stmt GenerateUnaryVectorCall(const std::string& op_type,
                               const Buffer& output_buffer,
                               const PrimExpr& output_offset,
                               const Buffer& input_buffer,
                               const PrimExpr& input_offset,
                               int64_t element_count) {
    DataType dtype = output_buffer->dtype;
    std::string dtype_str = DTypeToString(dtype);

    Array<PrimExpr> call_args;
    call_args.push_back(StringImm(op_type));
    call_args.push_back(CreateAccessPtr(output_buffer, dtype_str, output_offset,
                                        element_count, 2));
    call_args.push_back(CreateAccessPtr(input_buffer, dtype_str, input_offset,
                                        element_count, 1));
    call_args.push_back(IntImm(DataType::Int(32), element_count));

    PrimExpr call = Call(DataType::Handle(), builtin::call_extern(), call_args);
    return Evaluate(call);
  }

  Stmt GenerateBinaryVectorCall(const std::string& op_type,
                                const Buffer& output_buffer,
                                const PrimExpr& output_offset,
                                const Buffer& input_buffer1,
                                const PrimExpr& input_offset1,
                                const Buffer& input_buffer2,
                                const PrimExpr& input_offset2,
                                int64_t element_count) {
    DataType dtype = output_buffer->dtype;
    std::string dtype_str = DTypeToString(dtype);

    Array<PrimExpr> call_args;
    call_args.push_back(StringImm(op_type));
    call_args.push_back(CreateAccessPtr(output_buffer, dtype_str, output_offset,
                                        element_count, 2));
    call_args.push_back(CreateAccessPtr(input_buffer1, dtype_str, input_offset1,
                                        element_count, 1));
    call_args.push_back(CreateAccessPtr(input_buffer2, dtype_str, input_offset2,
                                        element_count, 1));
    call_args.push_back(IntImm(DataType::Int(32), element_count));

    PrimExpr call =
        Call(DataType::Handle(), builtin::call_extern(), call_args);
    return Evaluate(call);
  }

  Stmt GenerateScalarVectorCall(const std::string& op_type,
                                const Buffer& output_buffer,
                                const PrimExpr& output_offset,
                                const Buffer& input_buffer,
                                const PrimExpr& input_offset,
                                const PrimExpr& scalar_value,
                                int64_t element_count) {
    DataType dtype = output_buffer->dtype;
    std::string dtype_str = DTypeToString(dtype);

    static const std::unordered_set<std::string> no_suffix_ops = {
        "AscendC::ShiftLeft",
        "AscendC::ShiftRight"};

    std::string scalar_op_type =
        no_suffix_ops.count(op_type) > 0 ? op_type : op_type + "s";

    Array<PrimExpr> call_args;
    call_args.push_back(StringImm(scalar_op_type));
    call_args.push_back(CreateAccessPtr(output_buffer, dtype_str, output_offset,
                                        element_count, 2));
    call_args.push_back(CreateAccessPtr(input_buffer, dtype_str, input_offset,
                                        element_count, 1));
    call_args.push_back(scalar_value);
    call_args.push_back(IntImm(DataType::Int(32), element_count));

    PrimExpr call =
        Call(DataType::Handle(), builtin::call_extern(), call_args);
    return Evaluate(call);
  }

  Stmt GenerateBufferScalarVectorCall(const std::string& op_type,
                                      const Buffer& output_buffer,
                                      const PrimExpr& output_offset,
                                      const Buffer& input_buffer,
                                      const PrimExpr& input_offset,
                                      const Buffer& scalar_buffer,
                                      const PrimExpr& scalar_offset,
                                      int64_t element_count) {
    DataType dtype = output_buffer->dtype;
    std::string dtype_str = DTypeToString(dtype);

    static const std::unordered_set<std::string> no_suffix_ops = {
        "AscendC::ShiftLeft",
        "AscendC::ShiftRight"};

    std::string scalar_op_type =
        no_suffix_ops.count(op_type) > 0 ? op_type : op_type + "s";

    int64_t scalar_extent = 1;
    if (scalar_buffer->shape.size() > 0) {
      if (auto imm = scalar_buffer->shape[0].as<IntImmNode>()) {
        scalar_extent = imm->value;
      }
    }

    Array<PrimExpr> call_args;
    call_args.push_back(StringImm(scalar_op_type));
    call_args.push_back(CreateAccessPtr(output_buffer, dtype_str, output_offset,
                                        element_count, 2));
    call_args.push_back(CreateAccessPtr(input_buffer, dtype_str, input_offset,
                                        element_count, 1));
    call_args.push_back(CreateAccessPtr(scalar_buffer, dtype_str, scalar_offset,
                                        scalar_extent, 1));
    call_args.push_back(scalar_offset);
    call_args.push_back(IntImm(DataType::Int(32), element_count));

    PrimExpr call =
        Call(DataType::Handle(), builtin::call_extern(), call_args);
    return Evaluate(call);
  }

  bool TryGetElementCount(PrimExpr total_elements, int64_t* out_count) {
    ICHECK(out_count != nullptr);
    PrimExpr simplified = analyzer_->Simplify(total_elements);

    if (auto imm = simplified.as<IntImmNode>()) {
      *out_count = imm->value;
      return true;
    }

    return false;
  }

  PrimExpr CalculateBufferOffset(const Array<PrimExpr>& indices,
                                 const Buffer& buffer,
                                 const std::unordered_set<const VarNode*>& parallel_vars) {
    if (indices.empty()) {
      return IntImm(DataType::Int(32), 0);
    }

    Array<PrimExpr> processed_indices;
    for (const auto& idx : indices) {
      if (auto var = idx.as<VarNode>()) {
        if (vector_dim_var_ != nullptr && var == vector_dim_var_) {
          processed_indices.push_back(IntImm(DataType::Int(32), 0));
          continue;
        }
      }
      processed_indices.push_back(idx);
    }

    PrimExpr offset = processed_indices[0];
    for (size_t i = 1; i < processed_indices.size(); i++) {
      offset = offset * buffer->shape[i] + processed_indices[i];
    }

    return analyzer_->Simplify(offset);
  }

  bool IsScalarAccess(const Array<PrimExpr>& indices,
                      const std::unordered_set<const VarNode*>& parallel_vars) {
    for (const auto& idx : indices) {
      if (auto var = idx.as<VarNode>()) {
        if (vector_dim_var_ != nullptr && var == vector_dim_var_) {
          return false;
        }
      }
    }
    return true;
  }

  bool IsScalarLike(const PrimExpr& expr,
                    const std::unordered_set<const VarNode*>& parallel_vars) {
    if (IsScalar(expr)) {
      return true;
    }

    if (auto load = expr.as<BufferLoadNode>()) {
      return IsScalarAccess(load->indices, parallel_vars);
    }

    return false;
  }

  PrimExpr CreateAccessPtr(const Buffer& buffer,
                           const std::string& dtype_str,
                           const PrimExpr& offset,
                           int64_t extent,
                           int access_mask) {
    return Call(DataType::Handle(), builtin::tvm_access_ptr(),
                {StringImm(dtype_str), buffer->data, offset,
                 IntImm(DataType::Int(32), extent),
                 IntImm(DataType::Int(32), access_mask)});
  }

  std::string DTypeToString(DataType dtype) {
    if (dtype.is_float()) {
      if (dtype.bits() == 16) return "float16";
      if (dtype.bits() == 32) return "float32";
      if (dtype.bits() == 64) return "float64";
    } else if (dtype.is_int()) {
      return "int" + std::to_string(dtype.bits());
    } else if (dtype.is_uint()) {
      return "uint" + std::to_string(dtype.bits());
    }
    return "";
  }

  bool IsScalar(const PrimExpr& expr) {
    return expr.as<IntImmNode>() || expr.as<FloatImmNode>() ||
           expr.as<VarNode>();
  }

  bool IsZero(const PrimExpr& expr) {
    if (auto imm = expr.as<IntImmNode>()) {
      return imm->value == 0;
    }
    if (auto imm = expr.as<FloatImmNode>()) {
      return imm->value == 0.0;
    }
    return false;
  }
};

using namespace tir::transform;

tvm::transform::Pass AscendLowerParallelToVector() {
  auto pass_func = [=](PrimFunc f,IRModule m,PassContext ctx) {
    auto new_func = AscendLowerParallelToVector::Substitute(std::move(f));
    return new_func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendLowerParallelToVector", {});
}

TVM_REGISTER_GLOBAL("tl.transform.AscendLowerParallelToVector")
    .set_body_typed(AscendLowerParallelToVector);

}  // namespace tl
}  // namespace tvm
