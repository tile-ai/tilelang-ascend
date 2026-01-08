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
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include "../op/ascend.h"
#include "../op/builtin.h"
#include "./common/collector.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <functional>

namespace tvm {
namespace tl {

using namespace tir;

namespace {


// TIR unary operation to AscendC operation mapping
const std::unordered_map<std::string, Op> kTIRUnaryOpMap = {
  {"tir.exp", tl::ascend_exp()},
  {"tir.log", tl::ascend_ln()},
  {"tir.sqrt", tl::ascend_sqrt()},
  {"tir.rsqrt", tl::ascend_rsqrt()},
  {"tir.fabs", tl::ascend_abs()}
};

// Binary operation matcher and extractor
using BinaryMatcher = std::function<bool(const PrimExpr&)>;
using BinaryExtractor = std::function<void(const PrimExpr&, PrimExpr*, PrimExpr*)>;

struct BinaryOpInfo {
  Op op_type;
  BinaryMatcher matcher;
  BinaryExtractor extractor;
};

// Create binary operation table
inline std::vector<BinaryOpInfo> CreateBinaryOpTable() {
  std::vector<BinaryOpInfo> table;
  
  // Template for arithmetic operations
  auto add_arith_op = [&table](Op name, auto matcher_fn, auto cast_fn) {
    table.push_back({
      name,
      [matcher_fn](const PrimExpr& e) { return matcher_fn(e) != nullptr; },
      [cast_fn](const PrimExpr& e, PrimExpr* a, PrimExpr* b) {
        auto node = cast_fn(e);
        *a = node->a;
        *b = node->b;
      }
    });
  };
  
  add_arith_op(tl::ascend_add(), 
    [](const PrimExpr& e) { return e.as<AddNode>(); },
    [](const PrimExpr& e) { return e.as<AddNode>(); });
  
  add_arith_op(tl::ascend_sub(),
    [](const PrimExpr& e) { return e.as<SubNode>(); },
    [](const PrimExpr& e) { return e.as<SubNode>(); });
  
  add_arith_op(tl::ascend_mul(),
    [](const PrimExpr& e) { return e.as<MulNode>(); },
    [](const PrimExpr& e) { return e.as<MulNode>(); });
  
  add_arith_op(tl::ascend_div(),
    [](const PrimExpr& e) { return e.as<DivNode>(); },
    [](const PrimExpr& e) { return e.as<DivNode>(); });
  
  add_arith_op(tl::ascend_min(),
    [](const PrimExpr& e) { return e.as<MinNode>(); },
    [](const PrimExpr& e) { return e.as<MinNode>(); });
  
  add_arith_op(tl::ascend_max(),
    [](const PrimExpr& e) { return e.as<MaxNode>(); },
    [](const PrimExpr& e) { return e.as<MaxNode>(); });
  
  // Template for builtin call operations
  auto add_builtin_op = [&table](Op name,auto builtin_fn) {
    table.push_back({
      name,
      [builtin_fn](const PrimExpr& e) {
        auto call = e.as<CallNode>();
        return call && call->op.same_as(builtin_fn());
      },
      [](const PrimExpr& e, PrimExpr* a, PrimExpr* b) {
        auto call = e.as<CallNode>();
        if (call && call->args.size() >= 2) {
          *a = call->args[0];
          *b = call->args[1];
        }
      }
    });
  };
  
  add_builtin_op(tl::ascend_bitwise_and(), builtin::bitwise_and);
  add_builtin_op(tl::ascend_bitwise_or(), builtin::bitwise_or);
  add_builtin_op(tl::ascend_bitwise_lshift(), builtin::shift_left);
  add_builtin_op(tl::ascend_bitwise_rshift(), builtin::shift_right);
  add_builtin_op(tl::ascend_bitwise_and(), tl::ascend_bitwise_and);
  add_builtin_op(tl::ascend_bitwise_or(), tl::ascend_bitwise_or);
  add_builtin_op(tl::ascend_bitwise_lshift(), tl::ascend_bitwise_lshift);
  add_builtin_op(tl::ascend_bitwise_rshift(), tl::ascend_bitwise_rshift);
  return table;
}

static const std::vector<BinaryOpInfo> kBinaryOpTable = CreateBinaryOpTable();

const std::unordered_set<std::string> kNoSuffixOps = {
  "AscendC::ShiftLeft",
  "AscendC::ShiftRight"
};

}  // namespace

class AscendLowerParallelToVector : public arith::IRMutatorWithAnalyzer {
 public:
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    AscendLowerParallelToVector substituter(&analyzer);
    PrimFuncNode* fptr = f.CopyOnWrite();
    fptr->body = substituter.VisitStmt(f->body);
    return GetRef<PrimFunc>(fptr);
  }

 private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  const VarNode* vector_dim_var_ = nullptr;
  const VarNode* outer_dim_var_ = nullptr;
  bool is_2d_vectorizing_ = false;
  int temp_buffer_id_ = 0;
  std::vector<Buffer> temp_buffers_;

  Buffer CreateTempBufferLike(const Buffer& ref) {
    DataType dtype = ref->dtype;
    int64_t num_elements = 1;
    for (const auto& dim : ref->shape) {
      if (auto imm = dim.as<IntImmNode>()) {
        num_elements *= imm->value;
      } else {
        num_elements = -1;
        break;
      }
    }
  
    if (num_elements < 0) {
      LOG(FATAL) << "Cannot create temp buffer for non-constant shape.";
      return Buffer();
    }

    Var data(
      ref->name + "_tmp_" + std::to_string(temp_buffer_id_++) + "_data",
      PointerType(PrimType(dtype), "shared")
    );

    Buffer buf = Buffer(
      data,
      dtype,
      {IntImm(DataType::Int(32), num_elements)},
      /*strides=*/{},
      /*elem_offset=*/PrimExpr(0),
      /*name=*/data->name_hint,
      /*data_alignment=*/0,
      /*offset_factor=*/0,
      /*buffer_type=*/kDefault
    );

    temp_buffers_.push_back(buf);
    return buf;
  }

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
    // Sequence of store statements (handles both single stores and sequences)
    auto TryVectorizeStoreSeq = [&](
      Stmt stmt,
      PrimExpr total_elements,
      const std::unordered_set<const VarNode*>& parallel_vars,
      bool has_outer_serial,
      const VarNode* vector_dim,
      const VarNode* outer_dim=nullptr
    ) -> Stmt {
      int64_t element_count = 0;
      if (!TryGetElementCount(total_elements, &element_count)) return Stmt();
      const VarNode* old_vector = vector_dim_var_;
      const VarNode* old_outer = outer_dim_var_;
      vector_dim_var_ = vector_dim;
      outer_dim_var_ = outer_dim;

      Stmt v = TryVectorizeBufferStoreSeq(stmt, element_count, parallel_vars, has_outer_serial);

      vector_dim_var_ = old_vector;
      outer_dim_var_ = old_outer;
      return v;
    };

    // Parallel cases
    if (op->kind == ForKind::kParallel) {
      // parallel -> (store | seq)
      {
        if (op->body.as<BufferStoreNode>() || op->body.as<SeqStmtNode>()) {
          Stmt v = TryVectorizeStoreSeq(
            op->body,
            op->extent,
            {op->loop_var.get()},
            false,
            op->loop_var.as<VarNode>()
          );
          if (v.defined()) return v;
            }
          }

      // parallel -> parallel -> (store | seq)
      const auto* inner_for = op->body.as<ForNode>();
      if (!inner_for || inner_for->kind != ForKind::kParallel) {
        return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
        }

      const auto* third_for = inner_for->body.as<ForNode>();
      if (third_for && third_for->kind == ForKind::kParallel) {
        LOG(FATAL) << "Unsupported: 3D or higher dimensional parallel loops detected. "
                  << "Only 1D and 2D parallel loops are supported for T.Parallel.";
      }

      PrimExpr total_elements = inner_for->extent * op->extent;
      std::unordered_set<const VarNode*> parallel_vars = {
        op->loop_var.get(),
        inner_for->loop_var.get()
      };

      if (inner_for->body.as<BufferStoreNode>() || inner_for->body.as<SeqStmtNode>()) {
        Stmt v = TryVectorizeStoreSeq(
          inner_for->body,
          total_elements,
          parallel_vars,
          false,
          inner_for->loop_var.get(),
          op->loop_var.get()
        );
        if (v.defined()) return v;
      }

      return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    // Serial cases
    if (op->kind == ForKind::kSerial) {
      const auto* inner_for = op->body.as<ForNode>();
      if (!inner_for || inner_for->kind != ForKind::kParallel) {
        return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
        }
      PrimExpr total_elements = inner_for->extent * op->extent;
      std::unordered_set<const VarNode*> parallel_vars = {
        inner_for->loop_var.get()
      };

      // serial -> parallel -> (store | seq)
      if (inner_for->body.as<BufferStoreNode>() || inner_for->body.as<SeqStmtNode>()) {
        Stmt v = TryVectorizeStoreSeq(
          inner_for->body,
          total_elements,
          parallel_vars,
          true,
          inner_for->loop_var.as<VarNode>()
        );
        if (!v.defined()) return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
        auto op_copy = make_object<ForNode>(*op);
        op_copy->body = v;
        return Stmt(op_copy);
      }
    }

    return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BlockNode* op) override {
    // Save outer state
    auto saved_buffers = std::move(temp_buffers_);
    int saved_temp_id = temp_buffer_id_;
    const VarNode* saved_vector_dim = vector_dim_var_;
    const VarNode* saved_outer_dim = outer_dim_var_;
    bool saved_is_2d_vectorizing = is_2d_vectorizing_;

    // Reset block-local state
    temp_buffers_.clear();
    temp_buffer_id_ = 0;
    vector_dim_var_ = nullptr;
    outer_dim_var_ = nullptr;
    is_2d_vectorizing_ = false;

    // Visit body (vectorization happens here)
    Stmt new_body = VisitStmt(op->body);

    // Attach temps to THIS block only
    Array<Buffer> allocs = op->alloc_buffers;
    for (const Buffer& buf : temp_buffers_) {
      allocs.push_back(buf);
            }

    // Restore outer state
    temp_buffers_ = std::move(saved_buffers);
    temp_buffer_id_ = saved_temp_id;
    vector_dim_var_ = saved_vector_dim;
    outer_dim_var_ = saved_outer_dim;
    is_2d_vectorizing_ = saved_is_2d_vectorizing;

    if (allocs.same_as(op->alloc_buffers) && new_body.same_as(op->body)) {
      return GetRef<Stmt>(op);
          }

    return Block(
      op->iter_vars,
      op->reads,
      op->writes,
      op->name_hint,
      new_body,
      op->init,
      allocs
    );
        }

  // Generic Plan
  struct VectorPlan {
    int64_t inner_vec_len{0};
    int64_t outer_extent{0};
    const VarNode* outer_index_var{nullptr};
    bool is_2d_vectorizable{false};
  };

  // Detect plan for vector, fill in the input plan
  bool DetectVectorPlan(
    const BufferStoreNode* store,
    int64_t element_count,
    VectorPlan* plan
  ) {
    Buffer output_buffer = store->buffer;

    /*----- 1D case -----*/
    if (output_buffer->shape.size() == 1 && store->indices.size() == 1) {
      const VarNode* idx_var = store->indices[0].as<VarNode>();
      if (idx_var == nullptr) return false;
      if (idx_var != vector_dim_var_) return false;
      if (element_count <= 0) return false;
  
      plan->inner_vec_len = element_count;
      plan->outer_extent = 1;
      plan->outer_index_var = nullptr;
      plan->is_2d_vectorizable = false;
      return true;
      }

    /*----- 2D case -----*/
    if (vector_dim_var_ == nullptr) return false;

    if (output_buffer->shape.size() == 2 && store->indices.size() == 2) {  
      const VarNode* outer_var = store->indices[0].as<VarNode>();
      const VarNode* inner_var = store->indices[1].as<VarNode>();
      if (outer_var == nullptr || inner_var == nullptr) return false;
      if (inner_var != vector_dim_var_) return false;
      const IntImmNode* inner_imm = output_buffer->shape[1].as<IntImmNode>();
      if (inner_imm == nullptr) return false;

      int64_t inner_vec_len = inner_imm->value;
      if (inner_vec_len <= 0) return false;
      if (element_count % inner_vec_len != 0) return false;

      plan->inner_vec_len = inner_vec_len;
      plan->outer_extent = element_count / inner_vec_len;
      plan->outer_index_var = outer_var;

      plan->is_2d_vectorizable = (outer_dim_var_ != nullptr && outer_var == outer_dim_var_);
      return true;
    }

    /*----- 3D db case -----*/
    if (output_buffer->shape.size() == 3 && store->indices.size() == 3) {
      const VarNode* outer_var = store->indices[1].as<VarNode>();
      const VarNode* inner_var = store->indices[2].as<VarNode>();
      if (outer_var == nullptr || inner_var == nullptr) return false;
      if (inner_var != vector_dim_var_) return false;
      const IntImmNode* inner_imm = output_buffer->shape[2].as<IntImmNode>();
      if (inner_imm == nullptr) return false;

      int64_t inner_vec_len = inner_imm->value;
      if (inner_vec_len <= 0) return false;
      if (element_count % inner_vec_len != 0) return false;

      plan->inner_vec_len = inner_vec_len;
      plan->outer_extent = element_count / inner_vec_len;
      plan->outer_index_var = outer_var;
      plan->is_2d_vectorizable = (outer_dim_var_ != nullptr && outer_var == outer_dim_var_);
      return true;
  }
    return false;
  }

  bool CheckExpressionSupports2DVectorization(
    const PrimExpr& expr,
    const std::unordered_set<const VarNode*>& parallel_vars
  ) {
    if (vector_dim_var_ == nullptr || outer_dim_var_ == nullptr) {
      return false;
    }
    class BufferLoadCollector : public StmtExprVisitor {
    public:
      std::vector<const BufferLoadNode*> loads;

      void VisitExpr_(const BufferLoadNode* op) override {
        loads.push_back(op);
        StmtExprVisitor::VisitExpr_(op);
      }
    };

    BufferLoadCollector collector;
    collector(expr);

    for (const auto* load : collector.loads) {
      bool uses_vector_dim = false;
      bool uses_outer_dim = false;
      for (const auto& idx : load->indices) {
        if (auto var = idx.as<VarNode>()) {
          if (var == vector_dim_var_) {
            uses_vector_dim = true;
          }
          if (var == outer_dim_var_) {
            uses_outer_dim = true;
          }
        }
      }

      if (!uses_vector_dim || !uses_outer_dim) {
        return false;
      }
    }

    return true;
      }

  Optional<Stmt> VectorizeStoreAsRowBody(
    const BufferStoreNode* store,
    int64_t inner_vec_len,
    int64_t outer_extent,
    bool is_2d,
    const std::unordered_set<const VarNode*>& parallel_vars
  ) {
    Buffer output_buffer = store->buffer;

    bool saved_is_2d_vectorizing = is_2d_vectorizing_;
    is_2d_vectorizing_ = is_2d;

    PrimExpr output_offset = CalculateBufferOffset(store->indices, output_buffer, parallel_vars);

        Array<Stmt> row_stmts;
    int64_t total_elements = is_2d ? (inner_vec_len * outer_extent) : inner_vec_len;
        bool success = DecomposeExpression(store->value, output_buffer,
                                        output_offset, total_elements,
                                        parallel_vars, &row_stmts, is_2d);
  
    is_2d_vectorizing_ = saved_is_2d_vectorizing;
    if (!success || row_stmts.empty()) return NullOpt;
    if (row_stmts.size() == 1) return row_stmts[0];
    return SeqStmt::Flatten(row_stmts);
        }

  Stmt TryVectorizeBufferStoreSeq(
    Stmt stmt,
    int64_t element_count,
    const std::unordered_set<const VarNode*>& parallel_vars,
    bool has_outer_serial
  ) {
    // Handle both single BufferStore and SeqStmt
    Array<Stmt> stores_to_process;
    if (const auto* store = stmt.as<BufferStoreNode>()) {
      stores_to_process = {stmt};
    } else if (const auto* seq = stmt.as<SeqStmtNode>()) {
      stores_to_process = seq->seq;
        } else {
      return Stmt(); // Not a store or sequence
        }

    // Find the first buffer store node as reference
    const BufferStoreNode* first_store = nullptr;
    for (const Stmt& s : stores_to_process) {
      if (auto st = s.as<BufferStoreNode>()) {
        first_store = st;
        break;
        }
    }
    if (first_store == nullptr) return Stmt();

    VectorPlan plan;
    if (!DetectVectorPlan(first_store, element_count, &plan)) {
      return Stmt();
        }

    if (plan.is_2d_vectorizable) {
      for (const Stmt& s : stores_to_process) {
        if (auto st = s.as<BufferStoreNode>()) {
          if (!CheckExpressionSupports2DVectorization(st->value, parallel_vars)) {
            plan.is_2d_vectorizable = false;
            break;
      }
    }
      }
    }

    Array<Stmt> bodies;
    for (const Stmt& s : stores_to_process) {
      if (auto st = s.as<BufferStoreNode>()) {
        // Must be compatible buffer store
        VectorPlan curr_plan;
        if (!DetectVectorPlan(st, element_count, &curr_plan) ||
            curr_plan.outer_extent != plan.outer_extent) {
              return Stmt();
        }

        auto body_opt = VectorizeStoreAsRowBody(
          st, 
          curr_plan.inner_vec_len, 
          curr_plan.outer_extent,
          plan.is_2d_vectorizable,
          parallel_vars
        );
        if (!body_opt.defined()) return Stmt();
        bodies.push_back(body_opt.value());
      } else {
        // Conservative: only handle pure BufferStore sequences for now.
        return Stmt();
      }
    }
    if (bodies.empty()) return Stmt();

    Stmt combined = (bodies.size() == 1) ? bodies[0] : SeqStmt::Flatten(bodies);
  
    if (plan.is_2d_vectorizable || has_outer_serial || plan.outer_extent == 1) {
      return combined;
  }

    Var outer_var("outer_broadcast_idx", DataType::Int(32));
    if (plan.outer_index_var != nullptr) {
      ReplaceVarExpr replacer(plan.outer_index_var, outer_var);
      combined = replacer(combined);
    }

    return For(
      outer_var,
      IntImm(DataType::Int(32), 0),
      IntImm(DataType::Int(32), plan.outer_extent),
      ForKind::kSerial,
      combined
    );
  }

  bool DecomposeExpression(const PrimExpr& expr,
                           const Buffer& output_buffer,
                           const PrimExpr& output_offset,
                           int64_t element_count,
                           const std::unordered_set<const VarNode*>& parallel_vars,
                           Array<Stmt>* statements,
                           bool is_2d = false) {
    Op unary_op_type;

    Optional<Buffer> unary_input_buffer;
    PrimExpr unary_input_offset;

    if (IsUnaryOp(expr, &unary_op_type, &unary_input_buffer, &unary_input_offset,
                        parallel_vars)) {
      auto stmt = GenerateUnaryVectorCall(unary_op_type, output_buffer, output_offset,
                                          unary_input_buffer.value(), unary_input_offset,
                                          element_count, is_2d);
      statements->push_back(stmt);
      return true;
    }

    Op op_type;
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
                              element_count, parallel_vars, statements, is_2d);
    }

    if (left_is_simple && right_is_complex) {
      return HandleLeftSimpleRightComplex(op_type, operands, output_buffer, output_offset,
                                          element_count, parallel_vars, statements, is_2d);
    }

    if (left_is_complex && right_is_simple) {
      return HandleLeftComplexRightSimple(op_type, operands, output_buffer, output_offset,
                                          element_count, parallel_vars, statements, is_2d);
    }

    if (left_is_complex && right_is_complex) {
      Buffer lhs_tmp = CreateTempBufferLike(output_buffer);
      Buffer rhs_tmp = CreateTempBufferLike(output_buffer);
      PrimExpr lhs_tmp_offset = output_offset;
      PrimExpr rhs_tmp_offset = output_offset;

      if (!DecomposeExpression(operands[0], lhs_tmp, lhs_tmp_offset,
                               element_count, parallel_vars, statements, is_2d)) {
        return false;
    }

      if (!DecomposeExpression(operands[1], rhs_tmp, rhs_tmp_offset,
                               element_count, parallel_vars, statements, is_2d)) {
    return false;
  }

      auto stmt = GenerateBinaryVectorCall(op_type, output_buffer, output_offset,
                                           lhs_tmp, lhs_tmp_offset, rhs_tmp,
                                           rhs_tmp_offset, element_count, is_2d);
      statements->push_back(stmt);
      return true;
    }

    return false;
  }

  bool IsUnaryOp(const PrimExpr& expr, Op* op_type,
                 Optional<Buffer>* input_buffer, PrimExpr* input_offset,
                 const std::unordered_set<const VarNode*>& parallel_vars) {
    if (auto call = expr.as<CallNode>()) {
      std::string op_name;

      if (auto* op_ptr = call->op.as<OpNode>()) {
        op_name = op_ptr->name;
      } else {
        return false;
      }

      auto it = kTIRUnaryOpMap.find(op_name);
      if (it != kTIRUnaryOpMap.end()) {
        if (op_type) *op_type = it->second;
      } else if (call->op.same_as(builtin::bitwise_not()) || call->op.same_as(tl::ascend_bitwise_not())) {
        if (op_type) *op_type = tl::ascend_bitwise_not();
      } else {
          return false;
        }

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
        if (op_type) *op_type = tl::ascend_relu();
        if (auto load = max_node->b.as<BufferLoadNode>()) {
          if (input_buffer) *input_buffer = load->buffer;
          if (input_offset)
            *input_offset =
                CalculateBufferOffset(load->indices, load->buffer, parallel_vars);
          return true;
        }
      }
      if (IsZero(max_node->b)) {
        if (op_type) *op_type = tl::ascend_relu();
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

  bool HandleSimpleCase(const Op& op_type,
                        const Array<PrimExpr>& operands,
                        const Buffer& output_buffer,
                        const PrimExpr& output_offset,
                        int64_t element_count,
                        const std::unordered_set<const VarNode*>& parallel_vars,
                        Array<Stmt>* statements,
                        bool is_2d = false) {
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
            load->buffer, scalar_offset, element_count, is_2d);
        statements->push_back(stmt);
        return true;
      } else {
        PrimExpr right_offset =
            CalculateBufferOffset(load->indices, load->buffer, parallel_vars);
        auto stmt = GenerateBinaryVectorCall(
            op_type, output_buffer, output_offset, left_buffer, left_offset,
            load->buffer, right_offset, element_count, is_2d);
        statements->push_back(stmt);
        return true;
      }

    } else if (IsScalar(operands[1])) {
      auto stmt = GenerateScalarVectorCall(op_type, output_buffer, output_offset,
                                           left_buffer, left_offset,
                                           operands[1], element_count, is_2d);
      statements->push_back(stmt);
      return true;
    }

    return false;
  }

  bool HandleLeftSimpleRightComplex(const Op& op_type,
                                    const Array<PrimExpr>& operands,
                                    const Buffer& output_buffer,
                                    const PrimExpr& output_offset,
                                    int64_t element_count,
                                    const std::unordered_set<const VarNode*>& parallel_vars,
                                    Array<Stmt>* statements,
                                    bool is_2d = false) {
    Buffer left_buffer;
    PrimExpr left_offset;

    if (auto load = operands[0].as<BufferLoadNode>()) {
      left_buffer = load->buffer;
      left_offset =
          CalculateBufferOffset(load->indices, left_buffer, parallel_vars);
    } else {
      return false;
    }

    Buffer tmp = CreateTempBufferLike(output_buffer);
    PrimExpr tmp_offset = output_offset;

    if (!DecomposeExpression(operands[1], tmp, tmp_offset,
                             element_count, parallel_vars, statements, is_2d)) {
      return false;
    }

    auto stmt = GenerateBinaryVectorCall(op_type, output_buffer, output_offset,
                                         left_buffer, left_offset, tmp,
                                         tmp_offset, element_count, is_2d);
    statements->push_back(stmt);
    return true;
  }

  bool HandleLeftComplexRightSimple(const Op& op_type,
                                    const Array<PrimExpr>& operands,
                                    const Buffer& output_buffer,
                                    const PrimExpr& output_offset,
                                    int64_t element_count,
                                    const std::unordered_set<const VarNode*>& parallel_vars,
                                    Array<Stmt>* statements,
                                    bool is_2d = false) {
    Buffer tmp = CreateTempBufferLike(output_buffer);
    PrimExpr tmp_offset = output_offset;

    if (!DecomposeExpression(operands[0], tmp, tmp_offset,
                             element_count, parallel_vars, statements, is_2d)) {
      return false;
    }

    if (auto load = operands[1].as<BufferLoadNode>()) {
      if (IsScalarAccess(load->indices, parallel_vars)) {
        PrimExpr scalar_offset =
            CalculateBufferOffset(load->indices, load->buffer, parallel_vars);
        auto stmt = GenerateBufferScalarVectorCall(
            op_type, output_buffer, output_offset, tmp, tmp_offset,
            load->buffer, scalar_offset, element_count, is_2d);
        statements->push_back(stmt);
        return true;
      } else {
        PrimExpr right_offset =
            CalculateBufferOffset(load->indices, load->buffer, parallel_vars);
        auto stmt = GenerateBinaryVectorCall(
            op_type, output_buffer, output_offset, tmp, tmp_offset,
            load->buffer, right_offset, element_count, is_2d);
        statements->push_back(stmt);
        return true;
      }

    } else if (IsScalar(operands[1])) {
      auto stmt = GenerateScalarVectorCall(op_type, output_buffer, output_offset,
                                           tmp, tmp_offset,
                                           operands[1], element_count, is_2d);
      statements->push_back(stmt);
      return true;
    }

    return false;
  }

  bool IsBinaryOp(const PrimExpr& expr, Op* op_type,
                  Array<PrimExpr>* operands) {

    for (const auto& op_info : kBinaryOpTable) {
      if (op_info.matcher(expr)) {
        if (op_type) {
          *op_type = op_info.op_type;
      }
      if (operands) {
          PrimExpr a, b;
          op_info.extractor(expr, &a, &b);
          operands->push_back(a);
          operands->push_back(b);
      }
      return true;
    }
      }
    return false;
  }

  Stmt GenerateUnaryVectorCall(const Op& op_type,
                               const Buffer& output_buffer,
                               const PrimExpr& output_offset,
                               const Buffer& input_buffer,
                               const PrimExpr& input_offset,
                               int64_t element_count,
                               bool is_2d = false) {
    DataType dtype = output_buffer->dtype;
    std::string dtype_str = DTypeToString(dtype);

    Array<PrimExpr> call_args;
    // call_args.push_back(StringImm(op_type));
    call_args.push_back(CreateAccessPtr(output_buffer, dtype_str, output_offset,
                                        element_count, 2));
    call_args.push_back(CreateAccessPtr(input_buffer, dtype_str, input_offset,
                                        element_count, 1));
    call_args.push_back(IntImm(DataType::Int(32), element_count));

    PrimExpr call = Call(DataType::Handle(), op_type, call_args);
    return Evaluate(call);
  }

  Stmt GenerateBinaryVectorCall(const Op& op_type,
                                const Buffer& output_buffer,
                                const PrimExpr& output_offset,
                                const Buffer& input_buffer1,
                                const PrimExpr& input_offset1,
                                const Buffer& input_buffer2,
                                const PrimExpr& input_offset2,
                                int64_t element_count,
                                bool is_2d = false) {
    DataType dtype = output_buffer->dtype;
    std::string dtype_str = DTypeToString(dtype);

    Array<PrimExpr> call_args;
    // call_args.push_back(StringImm(op_type));
    call_args.push_back(CreateAccessPtr(output_buffer, dtype_str, output_offset,
                                        element_count, 2));
    call_args.push_back(CreateAccessPtr(input_buffer1, dtype_str, input_offset1,
                                        element_count, 1));
    call_args.push_back(CreateAccessPtr(input_buffer2, dtype_str, input_offset2,
                                        element_count, 1));
    call_args.push_back(IntImm(DataType::Int(32), element_count));

    PrimExpr call =
        Call(DataType::Handle(), op_type, call_args);
    return Evaluate(call);
  }

  Stmt GenerateScalarVectorCall(const Op& op_type,
                                const Buffer& output_buffer,
                                const PrimExpr& output_offset,
                                const Buffer& input_buffer,
                                const PrimExpr& input_offset,
                                const PrimExpr& scalar_value,
                                int64_t element_count,
                                bool is_2d = false) {
    DataType dtype = output_buffer->dtype;
    std::string dtype_str = DTypeToString(dtype);


    // std::string scalar_op_type = kNoSuffixOps.count(op_type) > 0 ? op_type : op_type + "s";

    Op scalar_op_type = op_type;
    if (op_type.same_as(tl::ascend_add())) {
      scalar_op_type = tl::ascend_adds();
    } else if (op_type.same_as(tl::ascend_sub())) {
      scalar_op_type = tl::ascend_subs();
    } else if (op_type.same_as(tl::ascend_mul())) {
      scalar_op_type = tl::ascend_muls();
    } else if (op_type.same_as(tl::ascend_div())) {
      scalar_op_type = tl::ascend_divs();
    }

    Array<PrimExpr> call_args;
    // call_args.push_back(StringImm(scalar_op_type));
    call_args.push_back(CreateAccessPtr(output_buffer, dtype_str, output_offset,
                                        element_count, 2));
    call_args.push_back(CreateAccessPtr(input_buffer, dtype_str, input_offset,
                                        element_count, 1));
    call_args.push_back(scalar_value);
    call_args.push_back(IntImm(DataType::Int(32), element_count));

    PrimExpr call =
        Call(DataType::Handle(), scalar_op_type, call_args);
    return Evaluate(call);
  }

  Stmt GenerateBufferScalarVectorCall(const Op& op_type,
                                      const Buffer& output_buffer,
                                      const PrimExpr& output_offset,
                                      const Buffer& input_buffer,
                                      const PrimExpr& input_offset,
                                      const Buffer& scalar_buffer,
                                      const PrimExpr& scalar_offset,
                                      int64_t element_count,
                                      bool is_2d = false) {
    DataType dtype = output_buffer->dtype;
    std::string dtype_str = DTypeToString(dtype);

    // std::string scalar_op_type = kNoSuffixOps.count(op_type) > 0 ? op_type : op_type + "s";

    Op scalar_op_type = op_type;
    if (op_type.same_as(tl::ascend_add())) {
      scalar_op_type = tl::ascend_adds();
    } else if (op_type.same_as(tl::ascend_sub())) {
      scalar_op_type = tl::ascend_subs();
    } else if (op_type.same_as(tl::ascend_mul())) {
      scalar_op_type = tl::ascend_muls();
    } else if (op_type.same_as(tl::ascend_div())) {
      scalar_op_type = tl::ascend_divs();
    }

    int64_t scalar_extent = 1;
    if (scalar_buffer->shape.size() > 0) {
      if (auto imm = scalar_buffer->shape[0].as<IntImmNode>()) {
        scalar_extent = imm->value;
      }
    }

    Array<PrimExpr> call_args;
    // call_args.push_back(StringImm(scalar_op_type));
    call_args.push_back(CreateAccessPtr(output_buffer, dtype_str, output_offset,
                                        element_count, 2));
    call_args.push_back(CreateAccessPtr(input_buffer, dtype_str, input_offset,
                                        element_count, 1));
    call_args.push_back(CreateAccessPtr(scalar_buffer, dtype_str, scalar_offset,
                                        scalar_extent, 1));
    call_args.push_back(scalar_offset);
    call_args.push_back(IntImm(DataType::Int(32), element_count));

    PrimExpr call =
        Call(DataType::Handle(), scalar_op_type, call_args);
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
        // Set vectorization dim var to 0
        if (vector_dim_var_ != nullptr && var == vector_dim_var_) {
          processed_indices.push_back(IntImm(DataType::Int(32), 0));
          continue;
        }
        // Set outer dim var to 0 (2d vectorization)
        if (is_2d_vectorizing_ && outer_dim_var_ != nullptr && var == outer_dim_var_) {
          processed_indices.push_back(IntImm(DataType::Int(32), 0));
          continue;
      }
      }
      processed_indices.push_back(idx);
    }

    bool all_zero = true;
    for (const auto& idx : processed_indices) {
      if (auto imm = idx.as<IntImmNode>()) {
        if (imm->value != 0) {
          all_zero = false;
          break;
        }
      } else {
        all_zero = false;
        break;
      }
    }
  
    if (all_zero) {
      return IntImm(DataType::Int(32), 0);
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
