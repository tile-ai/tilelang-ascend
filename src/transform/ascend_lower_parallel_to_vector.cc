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

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
    {"tir.fabs", tl::ascend_abs()}};

// Binary operation matcher and extractor
using BinaryMatcher = std::function<bool(const PrimExpr &)>;
using BinaryExtractor =
    std::function<void(const PrimExpr &, PrimExpr *, PrimExpr *)>;

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
    table.push_back(
        {name,
         [matcher_fn](const PrimExpr &e) { return matcher_fn(e) != nullptr; },
         [cast_fn](const PrimExpr &e, PrimExpr *a, PrimExpr *b) {
           auto node = cast_fn(e);
           *a = node->a;
           *b = node->b;
         }});
  };

  add_arith_op(
      tl::ascend_add(), [](const PrimExpr &e) { return e.as<AddNode>(); },
      [](const PrimExpr &e) { return e.as<AddNode>(); });

  add_arith_op(
      tl::ascend_sub(), [](const PrimExpr &e) { return e.as<SubNode>(); },
      [](const PrimExpr &e) { return e.as<SubNode>(); });

  add_arith_op(
      tl::ascend_mul(), [](const PrimExpr &e) { return e.as<MulNode>(); },
      [](const PrimExpr &e) { return e.as<MulNode>(); });

  add_arith_op(
      tl::ascend_div(), [](const PrimExpr &e) { return e.as<DivNode>(); },
      [](const PrimExpr &e) { return e.as<DivNode>(); });

  add_arith_op(
      tl::ascend_min(), [](const PrimExpr &e) { return e.as<MinNode>(); },
      [](const PrimExpr &e) { return e.as<MinNode>(); });

  add_arith_op(
      tl::ascend_max(), [](const PrimExpr &e) { return e.as<MaxNode>(); },
      [](const PrimExpr &e) { return e.as<MaxNode>(); });

  // Template for builtin call operations
  auto add_builtin_op = [&table](Op name, auto builtin_fn) {
    table.push_back({name,
                     [builtin_fn](const PrimExpr &e) {
                       auto call = e.as<CallNode>();
                       return call && call->op.same_as(builtin_fn());
                     },
                     [](const PrimExpr &e, PrimExpr *a, PrimExpr *b) {
                       auto call = e.as<CallNode>();
                       if (call && call->args.size() >= 2) {
                         *a = call->args[0];
                         *b = call->args[1];
                       }
                     }});
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

// const std::unordered_set<std::string> kNoSuffixOps = {
//   "AscendC::ShiftLeft",
//   "AscendC::ShiftRight"
// };

} // namespace

class AscendLowerParallelToVector : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    AscendLowerParallelToVector substituter(&analyzer);
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = substituter.VisitStmt(f->body);
    return GetRef<PrimFunc>(fptr);
  }

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  const VarNode *vector_dim_var_ = nullptr;
  const VarNode *outer_dim_var_ = nullptr;
  bool is_2d_vectorizing_ = false;
  int temp_buffer_id_ = 0;
  std::vector<Buffer> temp_buffers_;
  int64_t vector_dim_extent_{
      0};                       // Track the extent of the vector dimension loop
  int64_t outer_dim_extent_{0}; // Track the extent of the outer dimension loop

  Buffer CreateTempBufferLike(const Buffer &ref, int64_t num_elements,
                              int64_t inner_vec_len = 0) {
    DataType dtype = ref->dtype;

    if (num_elements < 0) {
      LOG(FATAL) << "Cannot create temp buffer for non-constant shape.";
      return Buffer();
    }

    Var data(ref->name + "_tmp_" + std::to_string(temp_buffer_id_++) + "_data",
             PointerType(PrimType(dtype), "shared"));

    // Determine the shape based on the actual computation size (loop extent)
    Array<PrimExpr> shape;
    if (ref->shape.size() == 1) {
      // 1D case: just the total elements
      shape.push_back(IntImm(DataType::Int(32), num_elements));
    } else if (ref->shape.size() >= 2) {
      // 2D or higher: use the computation dimensions
      // If inner_vec_len is provided, use it to determine the shape
      if (inner_vec_len > 0) {
        int64_t outer_extent = num_elements / inner_vec_len;
        shape.push_back(IntImm(DataType::Int(32), outer_extent));
        shape.push_back(IntImm(DataType::Int(32), inner_vec_len));
      } else {
        // Fall back to 1D if we can't determine 2D shape
        shape.push_back(IntImm(DataType::Int(32), num_elements));
      }
    }

    Buffer buf = Buffer(data, dtype, shape,
                        /*strides=*/{},
                        /*elem_offset=*/PrimExpr(0),
                        /*name=*/data->name_hint,
                        /*data_alignment=*/0,
                        /*offset_factor=*/0,
                        /*buffer_type=*/kDefault);

    temp_buffers_.push_back(buf);
    return buf;
  }

  class ReplaceVarExpr : public StmtExprMutator {
  public:
    const VarNode *from_;
    Var to_;
    explicit ReplaceVarExpr(const VarNode *from, Var to)
        : from_(from), to_(to) {}

    PrimExpr VisitExpr_(const VarNode *op) override {
      if (op == from_) {
        return to_;
      }
      return GetRef<PrimExpr>(op);
    }

    Stmt VisitStmt_(const ForNode *op) override {
      return StmtExprMutator::VisitStmt_(op);
    }
  };

  // Substitute loop variables with constants in expressions
  class SubstituteLoopVars : public ExprMutator {
  public:
    const VarNode *vector_dim_var_;
    const VarNode *outer_dim_var_;
    bool is_2d_vectorizing_;

    explicit SubstituteLoopVars(const VarNode *vector_dim_var,
                                const VarNode *outer_dim_var,
                                bool is_2d_vectorizing)
        : vector_dim_var_(vector_dim_var), outer_dim_var_(outer_dim_var),
          is_2d_vectorizing_(is_2d_vectorizing) {}

    PrimExpr VisitExpr_(const VarNode *op) override {
      // Replace vector dim var with 0
      if (vector_dim_var_ != nullptr && op == vector_dim_var_) {
        return IntImm(DataType::Int(32), 0);
      }
      // Replace outer dim var with 0 if 2d vectorizing
      if (is_2d_vectorizing_ && outer_dim_var_ != nullptr &&
          op == outer_dim_var_) {
        return IntImm(DataType::Int(32), 0);
      }
      return GetRef<PrimExpr>(op);
    }
  };

  Stmt VisitStmt_(const ForNode *op) override {
    // Sequence of store statements (handles both single stores and sequences)
    auto TryVectorizeStoreSeq =
        [&](Stmt stmt, PrimExpr total_elements,
            const std::unordered_set<const VarNode *> &parallel_vars,
            bool has_outer_serial, const VarNode *vector_dim,
            const VarNode *outer_dim = nullptr) -> Stmt {
      int64_t element_count = 0;
      if (!TryGetElementCount(total_elements, &element_count))
        return Stmt();
      const VarNode *old_vector = vector_dim_var_;
      const VarNode *old_outer = outer_dim_var_;
      vector_dim_var_ = vector_dim;
      outer_dim_var_ = outer_dim;

      Stmt v = TryVectorizeBufferStoreSeq(stmt, element_count, parallel_vars,
                                          has_outer_serial);

      vector_dim_var_ = old_vector;
      outer_dim_var_ = old_outer;
      return v;
    };

    // Parallel cases
    if (op->kind == ForKind::kParallel) {
      // parallel -> (store | seq)
      {
        if (op->body.as<BufferStoreNode>() || op->body.as<SeqStmtNode>()) {
          int64_t extent = 0;
          if (TryGetElementCount(op->extent, &extent)) {
            vector_dim_extent_ = extent;
            outer_dim_extent_ = 1;
          }
          Stmt v =
              TryVectorizeStoreSeq(op->body, op->extent, {op->loop_var.get()},
                                   false, op->loop_var.as<VarNode>());
          vector_dim_extent_ = 0;
          outer_dim_extent_ = 0;
          if (v.defined())
            return v;
        }
      }

      // parallel -> parallel -> (store | seq)
      const auto *inner_for = op->body.as<ForNode>();
      if (!inner_for || inner_for->kind != ForKind::kParallel) {
        return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
      }

      const auto *third_for = inner_for->body.as<ForNode>();
      if (third_for && third_for->kind == ForKind::kParallel) {
        LOG(FATAL)
            << "Unsupported: 3D or higher dimensional parallel loops detected. "
            << "Only 1D and 2D parallel loops are supported for T.Parallel.";
      }

      PrimExpr total_elements = inner_for->extent * op->extent;
      std::unordered_set<const VarNode *> parallel_vars = {
          op->loop_var.get(), inner_for->loop_var.get()};

      if (inner_for->body.as<BufferStoreNode>() ||
          inner_for->body.as<SeqStmtNode>()) {
        int64_t inner_extent = 0;
        int64_t outer_extent = 0;
        if (TryGetElementCount(inner_for->extent, &inner_extent) &&
            TryGetElementCount(op->extent, &outer_extent)) {
          vector_dim_extent_ = inner_extent;
          outer_dim_extent_ = outer_extent;
        }
        Stmt v = TryVectorizeStoreSeq(
            inner_for->body, total_elements, parallel_vars, false,
            inner_for->loop_var.get(), op->loop_var.get());
        vector_dim_extent_ = 0;
        outer_dim_extent_ = 0;
        if (v.defined())
          return v;
      }

      return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    // Serial cases
    if (op->kind == ForKind::kSerial) {
      const auto *inner_for = op->body.as<ForNode>();
      if (!inner_for || inner_for->kind != ForKind::kParallel) {
        return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
      }
      PrimExpr total_elements = inner_for->extent * op->extent;
      std::unordered_set<const VarNode *> parallel_vars = {
          inner_for->loop_var.get()};

      // serial -> parallel -> (store | seq)
      if (inner_for->body.as<BufferStoreNode>() ||
          inner_for->body.as<SeqStmtNode>()) {
        int64_t inner_extent = 0;
        if (TryGetElementCount(inner_for->extent, &inner_extent)) {
          vector_dim_extent_ = inner_extent;
          outer_dim_extent_ = 1; // No outer parallel loop in this case
        }
        Stmt v =
            TryVectorizeStoreSeq(inner_for->body, total_elements, parallel_vars,
                                 true, inner_for->loop_var.as<VarNode>());
        vector_dim_extent_ = 0;
        outer_dim_extent_ = 0;
        if (!v.defined())
          return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
        auto op_copy = make_object<ForNode>(*op);
        op_copy->body = v;
        return Stmt(op_copy);
      }
    }

    return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BlockNode *op) override {
    // Save outer state
    auto saved_buffers = std::move(temp_buffers_);
    int saved_temp_id = temp_buffer_id_;
    const VarNode *saved_vector_dim = vector_dim_var_;
    const VarNode *saved_outer_dim = outer_dim_var_;
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
    for (const Buffer &buf : temp_buffers_) {
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

    return Block(op->iter_vars, op->reads, op->writes, op->name_hint, new_body,
                 op->init, allocs);
  }

  // Generic Plan
  struct VectorPlan {
    int64_t inner_vec_len{0};
    int64_t outer_extent{0};
    const VarNode *outer_index_var{nullptr};
    bool is_2d_vectorizable{false};
  };

  // Detect plan for vector, fill in the input plan
  bool DetectVectorPlan(const BufferStoreNode *store, int64_t element_count,
                        VectorPlan *plan) {
    Buffer output_buffer = store->buffer;

    // Helper to check if an expression contains a specific variable
    auto ContainsVar = [](const PrimExpr &expr, const VarNode *var) -> bool {
      class VarChecker : public ExprVisitor {
      public:
        const VarNode *target_var_;
        bool found_{false};

        explicit VarChecker(const VarNode *target_var)
            : target_var_(target_var) {}

        void VisitExpr_(const VarNode *op) override {
          if (op == target_var_) {
            found_ = true;
          }
          ExprVisitor::VisitExpr_(op);
        }
      };

      VarChecker checker(var);
      checker(expr);
      return checker.found_;
    };

    /*----- 1D case -----*/
    if (output_buffer->shape.size() == 1 && store->indices.size() == 1) {
      if (!ContainsVar(store->indices[0], vector_dim_var_))
        return false;
      if (element_count <= 0)
        return false;

      plan->inner_vec_len = element_count;
      plan->outer_extent = 1;
      plan->outer_index_var = nullptr;
      plan->is_2d_vectorizable = false;
      return true;
    }

    /*----- 2D case -----*/
    if (vector_dim_var_ == nullptr)
      return false;

    if (output_buffer->shape.size() == 2 && store->indices.size() == 2) {
      // Check if inner index contains the vector dimension variable
      bool is_l1_output = IsL1Buffer(output_buffer);
      if (!ContainsVar(store->indices[1], vector_dim_var_) && !is_l1_output)
        return false;

      int64_t inner_vec_len = 0;
      // Use vector_dim_extent_ if available (actual loop extent), otherwise
      // fall back to buffer shape
      if (vector_dim_extent_ > 0) {
        inner_vec_len = vector_dim_extent_;
      } else {
        const IntImmNode *inner_imm = output_buffer->shape[1].as<IntImmNode>();
        if (inner_imm == nullptr)
          return false;
        inner_vec_len = inner_imm->value;
      }

      if (inner_vec_len <= 0)
        return false;
      if (element_count % inner_vec_len != 0)
        return false;

      plan->inner_vec_len = inner_vec_len;
      plan->outer_extent = element_count / inner_vec_len;

      // Try to extract outer variable from the outer index
      plan->outer_index_var = store->indices[0].as<VarNode>();

      // Check if outer index contains the outer dimension variable (for 2D
      // vectorization)
      plan->is_2d_vectorizable =
          (outer_dim_var_ != nullptr &&
           ContainsVar(store->indices[0], outer_dim_var_));
      return true;
    }

    /*----- 3D db case -----*/
    if (output_buffer->shape.size() == 3 && store->indices.size() == 3) {
      if (!ContainsVar(store->indices[2], vector_dim_var_))
        return false;

      int64_t inner_vec_len = 0;
      // Use vector_dim_extent_ if available (actual loop extent), otherwise
      // fall back to buffer shape
      if (vector_dim_extent_ > 0) {
        inner_vec_len = vector_dim_extent_;
      } else {
        const IntImmNode *inner_imm = output_buffer->shape[2].as<IntImmNode>();
        if (inner_imm == nullptr)
          return false;
        inner_vec_len = inner_imm->value;
      }

      if (inner_vec_len <= 0)
        return false;
      if (element_count % inner_vec_len != 0)
        return false;

      plan->inner_vec_len = inner_vec_len;
      plan->outer_extent = element_count / inner_vec_len;
      plan->outer_index_var = store->indices[1].as<VarNode>();
      plan->is_2d_vectorizable =
          (outer_dim_var_ != nullptr &&
           ContainsVar(store->indices[1], outer_dim_var_));
      return true;
    }
    return false;
  }

  bool CheckExpressionSupports2DVectorization(
      const PrimExpr &expr,
      const std::unordered_set<const VarNode *> &parallel_vars) {
    if (vector_dim_var_ == nullptr || outer_dim_var_ == nullptr) {
      return false;
    }
    class BufferLoadCollector : public StmtExprVisitor {
    public:
      std::vector<const BufferLoadNode *> loads;

      void VisitExpr_(const BufferLoadNode *op) override {
        loads.push_back(op);
        StmtExprVisitor::VisitExpr_(op);
      }
    };

    BufferLoadCollector collector;
    collector(expr);
    bool container_2d_dim = false;
    for (const auto *load : collector.loads) {
      if (load->buffer->shape.size() == 2) {
        container_2d_dim = true;
      }
    }

    auto ContainsVar = [](const PrimExpr &expr, const VarNode *var) -> bool {
      class VarChecker : public ExprVisitor {
      public:
        const VarNode *target_var_;
        bool found_{false};
        explicit VarChecker(const VarNode *target_var)
            : target_var_(target_var) {}
        void VisitExpr_(const VarNode *op) override {
          if (op == target_var_) {
            found_ = true;
          }
          ExprVisitor::VisitExpr_(op);
        }
      };
      VarChecker checker(var);
      checker(expr);
      return checker.found_;
    };

    for (const auto *load : collector.loads) {
      bool uses_vector_dim = false;
      bool uses_outer_dim = false;
      for (const auto &idx : load->indices) {
        if (ContainsVar(idx, vector_dim_var_)) {
          uses_vector_dim = true;
        }
        if (ContainsVar(idx, outer_dim_var_)) {
          uses_outer_dim = true;
        }
      }

      // If the buffer uses both dimensions, it's fine
      // If the buffer uses only one dimension, check if it can be broadcast
      if (!uses_vector_dim || !uses_outer_dim) {
        if (container_2d_dim) {
          return false;
        }
        int64_t broadcast_dim = 0;
        if (!CanBroadcast(load, parallel_vars, &broadcast_dim)) {
          return false;
        }
      }
    }

    return true;
  }

  Optional<Stmt> VectorizeStoreAsRowBody(
      const BufferStoreNode *store, int64_t inner_vec_len, int64_t outer_extent,
      bool is_2d, const std::unordered_set<const VarNode *> &parallel_vars) {
    Buffer output_buffer = store->buffer;
    bool is_global_output = IsGlobalMemoryBuffer(output_buffer);
    bool is_l0c_input = false;
    if (auto cast_node = store->value.as<CastNode>()) {
      if (auto l0c_node = cast_node->value.as<BufferLoadNode>()) {
        Buffer input_buffer = l0c_node->buffer;
        is_l0c_input = IsL0CBuffer(input_buffer);
      }
    }

    bool saved_is_2d_vectorizing = is_2d_vectorizing_;
    is_2d_vectorizing_ = is_2d;

    PrimExpr output_offset =
        CalculateBufferOffset(store->indices, output_buffer, parallel_vars);

    // If output is in GM, create a temporary UB buffer sized for the
    // computation block
    Buffer actual_output_buffer = output_buffer;
    PrimExpr actual_output_offset = output_offset;
    Buffer temp_ub_buffer;

    if (is_global_output && !is_l0c_input) {
      // Create a temporary UB buffer sized for the computation block (not the
      // full GM buffer)
      int64_t total_elements = inner_vec_len * outer_extent;
      temp_ub_buffer =
          CreateTempBufferLike(output_buffer, total_elements, inner_vec_len);
      actual_output_buffer = temp_ub_buffer;
      actual_output_offset =
          IntImm(DataType::Int(32), 0); // Start from beginning of temp buffer
    }

    // Step 2: Detect all 1D buffers that can be broadcasted
    // Only collect broadcast buffers if there's no container 2d dim and output
    // is 2d
    class BroadcastableBufferCollector : public ExprVisitor {
    public:
      std::vector<BroadcastInfo> broadcast_infos;
      const std::unordered_set<const VarNode *> &parallel_vars_;
      int64_t inner_vec_len_;
      int64_t outer_extent_;
      AscendLowerParallelToVector *parent_;

      BroadcastableBufferCollector(
          const std::unordered_set<const VarNode *> &parallel_vars,
          int64_t inner_vec_len, int64_t outer_extent,
          AscendLowerParallelToVector *parent)
          : parallel_vars_(parallel_vars), inner_vec_len_(inner_vec_len),
            outer_extent_(outer_extent), parent_(parent) {}

      void VisitExpr_(const BufferLoadNode *op) override {
        // Only check 1D buffers
        if (op->buffer->shape.size() == 1) {
          int64_t broadcast_dim = 0;
          if (parent_->CanBroadcast(op, parallel_vars_, &broadcast_dim)) {
            BroadcastInfo info;
            info.load = op;
            info.broadcast_dim = broadcast_dim;
            info.outer_extent = outer_extent_;
            info.inner_vec_len = inner_vec_len_;
            broadcast_infos.push_back(info);
          }
        }
        ExprVisitor::VisitExpr_(op);
      }
    };

    class UseDimChecker : public StmtExprVisitor {
    public:
      bool container_2d_dim = false;

      void VisitExpr_(const BufferLoadNode *op) override {
        // Check if any index is not a simple variable or IntImm
        if (op->buffer->shape.size() == 2) {
          container_2d_dim = true;
        }
        StmtExprVisitor::VisitExpr_(op);
      }
    };

    UseDimChecker dim_checker;
    dim_checker(store->value);

    BroadcastableBufferCollector collector(parallel_vars, inner_vec_len,
                                           outer_extent, this);
    // Only collect broadcast buffers if there's no container 2d dim and output
    // is 2d
    if (!dim_checker.container_2d_dim && is_2d_vectorizing_) {
      collector(store->value);
    }

    // Step 3: Create temp buffers for broadcasted 1D buffers and workspace
    // buffers
    Array<Stmt> broadcast_stmts;
    std::unordered_map<const BufferLoadNode *, Buffer> broadcast_buffer_map;
    std::unordered_map<const BufferLoadNode *, Buffer> workspace_buffer_map;

    for (size_t i = 0; i < collector.broadcast_infos.size(); ++i) {
      collector.broadcast_infos[i].broadcast_buffer =
          CreateBroadcastBuffer(collector.broadcast_infos[i].load->buffer,
                                collector.broadcast_infos[i].outer_extent,
                                collector.broadcast_infos[i].inner_vec_len);
      collector.broadcast_infos[i].workspace_buffer =
          CreateBroadcastWorkspaceBuffer(
              collector.broadcast_infos[i].outer_extent,
              collector.broadcast_infos[i].inner_vec_len,
              collector.broadcast_infos[i].load->buffer->dtype);
      broadcast_buffer_map[collector.broadcast_infos[i].load] =
          collector.broadcast_infos[i].broadcast_buffer;
      workspace_buffer_map[collector.broadcast_infos[i].load] =
          collector.broadcast_infos[i].workspace_buffer;
    }

    // Step 4: Generate broadcast calls
    for (size_t i = 0; i < collector.broadcast_infos.size(); ++i) {
      const auto &info = collector.broadcast_infos[i];
      Buffer broadcast_buffer = broadcast_buffer_map[info.load];
      Buffer workspace_buffer = workspace_buffer_map[info.load];
      Stmt broadcast_stmt = GenerateBroadcastStmt(
          info.load->buffer, broadcast_buffer, workspace_buffer,
          info.broadcast_dim, info.outer_extent, info.inner_vec_len);
      broadcast_stmts.push_back(broadcast_stmt);
    }

    // Step 5: Replace 1D buffers with broadcasted ones in the expression
    class BufferLoadReplacer : public ExprMutator {
    public:
      const std::unordered_map<const BufferLoadNode *, Buffer>
          &broadcast_buffer_map_;

      BufferLoadReplacer(const std::unordered_map<const BufferLoadNode *,
                                                  Buffer> &broadcast_buffer_map)
          : broadcast_buffer_map_(broadcast_buffer_map) {}

      PrimExpr VisitExpr_(const BufferLoadNode *op) override {
        auto it = broadcast_buffer_map_.find(op);
        if (it != broadcast_buffer_map_.end()) {
          // Replace with load from broadcast buffer
          // Use [0] as index since the vectorized operation will handle
          // accessing all elements of the broadcast buffer as contiguous memory
          Array<PrimExpr> indicates;
          indicates.push_back(IntImm(DataType::Int(32), 0));
          indicates.push_back(IntImm(DataType::Int(32), 0));
          return BufferLoad(it->second, indicates);
        }
        return ExprMutator::VisitExpr_(op);
      }
    };

    BufferLoadReplacer replacer(broadcast_buffer_map);
    PrimExpr new_value = replacer(store->value);

    // Step 6: Do the normal pass with the modified expression
    Array<Stmt> row_stmts;
    row_stmts.insert(row_stmts.end(), broadcast_stmts.begin(),
                     broadcast_stmts.end());

    int64_t total_elements =
        is_2d ? (inner_vec_len * outer_extent) : inner_vec_len;
    bool success = DecomposeExpression(
        new_value, actual_output_buffer, actual_output_offset, total_elements,
        parallel_vars, &row_stmts, is_2d, inner_vec_len);

    is_2d_vectorizing_ = saved_is_2d_vectorizing;
    if (is_global_output && (!success || row_stmts.empty()))
      return NullOpt;

    // If output is in GM, add ascend_copy to copy from temp UB to GM
    if (is_global_output && !is_l0c_input) {
      Stmt copy_stmt = GenerateAscendCopy(temp_ub_buffer, output_buffer,
                                          actual_output_offset, output_offset,
                                          total_elements, is_2d);
      row_stmts.push_back(copy_stmt);
    }

    if (row_stmts.size() == 1)
      return row_stmts[0];
    return SeqStmt::Flatten(row_stmts);
  }

  Stmt TryVectorizeBufferStoreSeq(
      Stmt stmt, int64_t element_count,
      const std::unordered_set<const VarNode *> &parallel_vars,
      bool has_outer_serial) {
    // Handle both single BufferStore and SeqStmt
    Array<Stmt> stores_to_process;
    if (const auto *store = stmt.as<BufferStoreNode>()) {
      stores_to_process = {stmt};
    } else if (const auto *seq = stmt.as<SeqStmtNode>()) {
      stores_to_process = seq->seq;
    } else {
      return Stmt(); // Not a store or sequence
    }

    // Find the first buffer store node as reference
    const BufferStoreNode *first_store = nullptr;
    for (const Stmt &s : stores_to_process) {
      if (auto st = s.as<BufferStoreNode>()) {
        first_store = st;
        break;
      }
    }
    if (first_store == nullptr)
      return Stmt();

    VectorPlan plan;
    if (!DetectVectorPlan(first_store, element_count, &plan)) {
      return Stmt();
    }

    if (plan.is_2d_vectorizable) {
      for (const Stmt &s : stores_to_process) {
        if (auto st = s.as<BufferStoreNode>()) {
          if (!CheckExpressionSupports2DVectorization(st->value,
                                                      parallel_vars)) {
            plan.is_2d_vectorizable = false;
            break;
          }
        }
      }
    }

    Array<Stmt> bodies;
    for (const Stmt &s : stores_to_process) {
      if (auto st = s.as<BufferStoreNode>()) {
        // Must be compatible buffer store
        VectorPlan curr_plan;
        if (!DetectVectorPlan(st, element_count, &curr_plan) ||
            curr_plan.outer_extent != plan.outer_extent) {
          return Stmt();
        }

        auto body_opt = VectorizeStoreAsRowBody(
            st, curr_plan.inner_vec_len, curr_plan.outer_extent,
            plan.is_2d_vectorizable, parallel_vars);
        if (!body_opt.defined())
          return Stmt();
        bodies.push_back(body_opt.value());
      } else {
        // Conservative: only handle pure BufferStore sequences for now.
        return Stmt();
      }
    }
    if (bodies.empty())
      return Stmt();

    Stmt combined = (bodies.size() == 1) ? bodies[0] : SeqStmt::Flatten(bodies);

    if (plan.is_2d_vectorizable || has_outer_serial || plan.outer_extent == 1) {
      return combined;
    }

    Var outer_var("outer_broadcast_idx", DataType::Int(32));
    if (plan.outer_index_var != nullptr) {
      ReplaceVarExpr replacer(plan.outer_index_var, outer_var);
      combined = replacer(combined);
    }

    return For(outer_var, IntImm(DataType::Int(32), 0),
               IntImm(DataType::Int(32), plan.outer_extent), ForKind::kSerial,
               combined);
  }

  bool
  DecomposeExpression(const PrimExpr &expr, const Buffer &output_buffer,
                      const PrimExpr &output_offset, int64_t element_count,
                      const std::unordered_set<const VarNode *> &parallel_vars,
                      Array<Stmt> *statements, bool is_2d = false,
                      int64_t inner_vec_len = 0) {

    // L0C->GM
    if (auto cast_node = expr.as<CastNode>()) {
      if (auto l0c_node = cast_node->value.as<BufferLoadNode>()) {
        Buffer input_buffer = l0c_node->buffer;
        PrimExpr input_offset = CalculateBufferOffset(
            l0c_node->indices, input_buffer, parallel_vars);
        auto stmt =
            GenerateAscendCopy(input_buffer, output_buffer, input_offset,
                               output_offset, element_count, is_2d);
        statements->push_back(stmt);
        return true;
      }
    }

    // GM->UB GM->L1
    if (auto load = expr.as<BufferLoadNode>()) {
      Buffer input_buffer = load->buffer;
      PrimExpr input_offset =
          CalculateBufferOffset(load->indices, input_buffer, parallel_vars);

      // calculate src_indices
      Array<PrimExpr> src_indices;
      SubstituteLoopVars substitutor(vector_dim_var_, outer_dim_var_,
                                     is_2d_vectorizing_);
      for (const auto &idx : load->indices) {
        src_indices.push_back(substitutor(idx));
      }
      auto stmt =
          GenerateAscendCopy(input_buffer, output_buffer, input_offset,
                             output_offset, element_count, is_2d, src_indices);
      statements->push_back(stmt);
      return true;
    }

    Op unary_op_type;

    Optional<Buffer> unary_input_buffer;
    PrimExpr unary_input_offset;

    if (IsUnaryOp(expr, &unary_op_type, &unary_input_buffer,
                  &unary_input_offset, parallel_vars)) {
      auto stmt = GenerateUnaryVectorCall(
          unary_op_type, output_buffer, output_offset,
          unary_input_buffer.value(), unary_input_offset, element_count, is_2d);
      statements->push_back(stmt);
      return true;
    }

    Op op_type;
    Array<PrimExpr> operands;

    if (!IsBinaryOp(expr, &op_type, &operands)) {
      return false;
    }

    ICHECK_EQ(operands.size(), 2);

    bool left_is_simple = operands[0].as<BufferLoadNode>() ||
                          IsScalarLike(operands[0], parallel_vars);
    bool right_is_simple = operands[1].as<BufferLoadNode>() ||
                           IsScalarLike(operands[1], parallel_vars);

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
      return HandleLeftSimpleRightComplex(
          op_type, operands, output_buffer, output_offset, element_count,
          parallel_vars, statements, is_2d, inner_vec_len);
    }

    if (left_is_complex && right_is_simple) {
      return HandleLeftComplexRightSimple(
          op_type, operands, output_buffer, output_offset, element_count,
          parallel_vars, statements, is_2d, inner_vec_len);
    }

    if (left_is_complex && right_is_complex) {
      Buffer lhs_tmp =
          CreateTempBufferLike(output_buffer, element_count, inner_vec_len);
      Buffer rhs_tmp =
          CreateTempBufferLike(output_buffer, element_count, inner_vec_len);
      PrimExpr lhs_tmp_offset = IntImm(DataType::Int(32), 0);
      PrimExpr rhs_tmp_offset = IntImm(DataType::Int(32), 0);

      if (!DecomposeExpression(operands[0], lhs_tmp, lhs_tmp_offset,
                               element_count, parallel_vars, statements, is_2d,
                               inner_vec_len)) {
        return false;
      }

      if (!DecomposeExpression(operands[1], rhs_tmp, rhs_tmp_offset,
                               element_count, parallel_vars, statements, is_2d,
                               inner_vec_len)) {
        return false;
      }

      auto stmt = GenerateBinaryVectorCall(
          op_type, output_buffer, output_offset, lhs_tmp, lhs_tmp_offset,
          rhs_tmp, rhs_tmp_offset, element_count, is_2d);
      statements->push_back(stmt);
      return true;
    }

    return false;
  }

  bool IsUnaryOp(const PrimExpr &expr, Op *op_type,
                 Optional<Buffer> *input_buffer, PrimExpr *input_offset,
                 const std::unordered_set<const VarNode *> &parallel_vars) {
    if (auto call = expr.as<CallNode>()) {
      std::string op_name;

      if (auto *op_ptr = call->op.as<OpNode>()) {
        op_name = op_ptr->name;
      } else {
        return false;
      }

      auto it = kTIRUnaryOpMap.find(op_name);
      if (it != kTIRUnaryOpMap.end()) {
        if (op_type)
          *op_type = it->second;
      } else if (call->op.same_as(builtin::bitwise_not()) ||
                 call->op.same_as(tl::ascend_bitwise_not())) {
        if (op_type)
          *op_type = tl::ascend_bitwise_not();
      } else {
        return false;
      }

      if (call->args.size() >= 1) {
        if (auto load = call->args[0].as<BufferLoadNode>()) {
          if (input_buffer)
            *input_buffer = load->buffer;
          if (input_offset)
            *input_offset = CalculateBufferOffset(load->indices, load->buffer,
                                                  parallel_vars);
          return true;
        }
      }
    }

    if (auto max_node = expr.as<MaxNode>()) {
      if (IsZero(max_node->a)) {
        if (op_type)
          *op_type = tl::ascend_relu();
        if (auto load = max_node->b.as<BufferLoadNode>()) {
          if (input_buffer)
            *input_buffer = load->buffer;
          if (input_offset)
            *input_offset = CalculateBufferOffset(load->indices, load->buffer,
                                                  parallel_vars);
          return true;
        }
      }
      if (IsZero(max_node->b)) {
        if (op_type)
          *op_type = tl::ascend_relu();
        if (auto load = max_node->a.as<BufferLoadNode>()) {
          if (input_buffer)
            *input_buffer = load->buffer;
          if (input_offset)
            *input_offset = CalculateBufferOffset(load->indices, load->buffer,
                                                  parallel_vars);
          return true;
        }
      }
    }
    return false;
  }

  bool
  HandleSimpleCase(const Op &op_type, const Array<PrimExpr> &operands,
                   const Buffer &output_buffer, const PrimExpr &output_offset,
                   int64_t element_count,
                   const std::unordered_set<const VarNode *> &parallel_vars,
                   Array<Stmt> *statements, bool is_2d = false) {
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
      auto stmt = GenerateScalarVectorCall(
          op_type, output_buffer, output_offset, left_buffer, left_offset,
          operands[1], element_count, is_2d);
      statements->push_back(stmt);
      return true;
    }

    return false;
  }

  bool HandleLeftSimpleRightComplex(
      const Op &op_type, const Array<PrimExpr> &operands,
      const Buffer &output_buffer, const PrimExpr &output_offset,
      int64_t element_count,
      const std::unordered_set<const VarNode *> &parallel_vars,
      Array<Stmt> *statements, bool is_2d = false, int64_t inner_vec_len = 0) {
    Buffer left_buffer;
    PrimExpr left_offset;

    if (auto load = operands[0].as<BufferLoadNode>()) {
      left_buffer = load->buffer;
      left_offset =
          CalculateBufferOffset(load->indices, left_buffer, parallel_vars);
    } else {
      return false;
    }

    Buffer tmp =
        CreateTempBufferLike(output_buffer, element_count, inner_vec_len);
    PrimExpr tmp_offset = IntImm(DataType::Int(32), 0);

    if (!DecomposeExpression(operands[1], tmp, tmp_offset, element_count,
                             parallel_vars, statements, is_2d, inner_vec_len)) {
      return false;
    }

    auto stmt = GenerateBinaryVectorCall(op_type, output_buffer, output_offset,
                                         left_buffer, left_offset, tmp,
                                         tmp_offset, element_count, is_2d);
    statements->push_back(stmt);
    return true;
  }

  bool HandleLeftComplexRightSimple(
      const Op &op_type, const Array<PrimExpr> &operands,
      const Buffer &output_buffer, const PrimExpr &output_offset,
      int64_t element_count,
      const std::unordered_set<const VarNode *> &parallel_vars,
      Array<Stmt> *statements, bool is_2d = false, int64_t inner_vec_len = 0) {
    Buffer tmp =
        CreateTempBufferLike(output_buffer, element_count, inner_vec_len);
    PrimExpr tmp_offset = IntImm(DataType::Int(32), 0);

    if (!DecomposeExpression(operands[0], tmp, tmp_offset, element_count,
                             parallel_vars, statements, is_2d, inner_vec_len)) {
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
      auto stmt = GenerateScalarVectorCall(op_type, output_buffer,
                                           output_offset, tmp, tmp_offset,
                                           operands[1], element_count, is_2d);
      statements->push_back(stmt);
      return true;
    }

    return false;
  }

  bool IsBinaryOp(const PrimExpr &expr, Op *op_type,
                  Array<PrimExpr> *operands) {

    for (const auto &op_info : kBinaryOpTable) {
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

  Stmt GenerateUnaryVectorCall(const Op &op_type, const Buffer &output_buffer,
                               const PrimExpr &output_offset,
                               const Buffer &input_buffer,
                               const PrimExpr &input_offset,
                               int64_t element_count, bool is_2d = false) {
    DataType dtype = output_buffer->dtype;

    Array<PrimExpr> call_args;
    // call_args.push_back(StringImm(op_type));
    call_args.push_back(CreateAccessPtr(output_buffer, dtype, output_offset,
                                        element_count, 2));
    call_args.push_back(CreateAccessPtr(input_buffer, dtype, input_offset,
                                        element_count, 1));
    call_args.push_back(IntImm(DataType::Int(32), element_count));

    PrimExpr call = Call(DataType::Handle(), op_type, call_args);
    return Evaluate(call);
  }

  Stmt GenerateBinaryVectorCall(const Op &op_type, const Buffer &output_buffer,
                                const PrimExpr &output_offset,
                                const Buffer &input_buffer1,
                                const PrimExpr &input_offset1,
                                const Buffer &input_buffer2,
                                const PrimExpr &input_offset2,
                                int64_t element_count, bool is_2d = false) {
    DataType dtype = output_buffer->dtype;

    Array<PrimExpr> call_args;
    // call_args.push_back(StringImm(op_type));
    call_args.push_back(CreateAccessPtr(output_buffer, dtype, output_offset,
                                        element_count, 2));
    call_args.push_back(CreateAccessPtr(input_buffer1, dtype, input_offset1,
                                        element_count, 1));
    call_args.push_back(CreateAccessPtr(input_buffer2, dtype, input_offset2,
                                        element_count, 1));
    call_args.push_back(IntImm(DataType::Int(32), element_count));

    PrimExpr call = Call(DataType::Handle(), op_type, call_args);
    return Evaluate(call);
  }

  Stmt GenerateScalarVectorCall(const Op &op_type, const Buffer &output_buffer,
                                const PrimExpr &output_offset,
                                const Buffer &input_buffer,
                                const PrimExpr &input_offset,
                                const PrimExpr &scalar_value,
                                int64_t element_count, bool is_2d = false) {
    DataType dtype = output_buffer->dtype;
    std::string dtype_str = DTypeToString(dtype);

    // std::string scalar_op_type = kNoSuffixOps.count(op_type) > 0 ? op_type :
    // op_type + "s";

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
    call_args.push_back(CreateAccessPtr(output_buffer, dtype, output_offset,
                                        element_count, 2));
    call_args.push_back(CreateAccessPtr(input_buffer, dtype, input_offset,
                                        element_count, 1));
    call_args.push_back(scalar_value);
    call_args.push_back(IntImm(DataType::Int(32), element_count));

    PrimExpr call = Call(DataType::Handle(), scalar_op_type, call_args);
    return Evaluate(call);
  }

  Stmt GenerateBufferScalarVectorCall(
      const Op &op_type, const Buffer &output_buffer,
      const PrimExpr &output_offset, const Buffer &input_buffer,
      const PrimExpr &input_offset, const Buffer &scalar_buffer,
      const PrimExpr &scalar_offset, int64_t element_count,
      bool is_2d = false) {
    DataType dtype = output_buffer->dtype;

    // std::string scalar_op_type = kNoSuffixOps.count(op_type) > 0 ? op_type :
    // op_type + "s";

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
    call_args.push_back(CreateAccessPtr(output_buffer, dtype, output_offset,
                                        element_count, 2));
    call_args.push_back(CreateAccessPtr(input_buffer, dtype, input_offset,
                                        element_count, 1));
    call_args.push_back(CreateAccessPtr(scalar_buffer, dtype, scalar_offset,
                                        scalar_extent, 1));
    call_args.push_back(scalar_offset);
    call_args.push_back(IntImm(DataType::Int(32), element_count));

    PrimExpr call = Call(DataType::Handle(), scalar_op_type, call_args);
    return Evaluate(call);
  }

  bool TryGetElementCount(PrimExpr total_elements, int64_t *out_count) {
    ICHECK(out_count != nullptr);
    PrimExpr simplified = analyzer_->Simplify(total_elements);

    if (auto imm = simplified.as<IntImmNode>()) {
      *out_count = imm->value;
      return true;
    }

    return false;
  }

  PrimExpr CalculateBufferOffset(
      const Array<PrimExpr> &indices, const Buffer &buffer,
      const std::unordered_set<const VarNode *> &parallel_vars) {
    if (indices.empty()) {
      return IntImm(DataType::Int(32), 0);
    }

    // Substitute loop variables with 0 in all indices
    SubstituteLoopVars substitutor(vector_dim_var_, outer_dim_var_,
                                   is_2d_vectorizing_);
    Array<PrimExpr> processed_indices;
    for (const auto &idx : indices) {
      processed_indices.push_back(substitutor(idx));
    }

    bool all_zero = true;
    for (const auto &idx : processed_indices) {
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

  bool
  IsScalarAccess(const Array<PrimExpr> &indices,
                 const std::unordered_set<const VarNode *> &parallel_vars) {
    if (vector_dim_var_ == nullptr)
      return true;

    // Check if any index contains the vector dimension variable
    for (const auto &idx : indices) {
      class VectorDimChecker : public ExprVisitor {
      public:
        const VarNode *target_var_;
        bool found_{false};

        explicit VectorDimChecker(const VarNode *target_var)
            : target_var_(target_var) {}

        void VisitExpr_(const VarNode *op) override {
          if (op == target_var_) {
            found_ = true;
          }
          ExprVisitor::VisitExpr_(op);
        }
      };

      VectorDimChecker checker(vector_dim_var_);
      checker(idx);
      if (checker.found_) {
        return false;
      }
    }
    return true;
  }

  bool IsScalarLike(const PrimExpr &expr,
                    const std::unordered_set<const VarNode *> &parallel_vars) {
    if (IsScalar(expr)) {
      return true;
    }

    if (auto load = expr.as<BufferLoadNode>()) {
      return IsScalarAccess(load->indices, parallel_vars);
    }

    return false;
  }

  PrimExpr CreateAccessPtr(const Buffer &buffer, const DataType &dtype,
                           const PrimExpr &offset, int64_t extent,
                           int access_mask) {
    return Call(DataType::Handle(), builtin::tvm_access_ptr(),
                {TypeAnnotation(dtype), buffer->data, offset,
                 IntImm(DataType::Int(32), extent),
                 IntImm(DataType::Int(32), access_mask)});
  }

  std::string DTypeToString(DataType dtype) {
    if (dtype.is_float()) {
      if (dtype.bits() == 16)
        return "float16";
      if (dtype.bits() == 32)
        return "float32";
      if (dtype.bits() == 64)
        return "float64";
    } else if (dtype.is_int()) {
      return "int" + std::to_string(dtype.bits());
    } else if (dtype.is_uint()) {
      return "uint" + std::to_string(dtype.bits());
    }
    return "";
  }

  std::string DTypeToAscendCString(DataType dtype) {
    if (dtype.is_float()) {
      if (dtype.bits() == 16)
        return "half";
      if (dtype.bits() == 32)
        return "float";
      if (dtype.bits() == 64)
        return "float64";
    } else if (dtype.is_int()) {
      if (dtype.bits() == 4)
        return "AscendC::int4b_t";
      if (dtype.bits() == 8)
        return "int8_t";
      if (dtype.bits() == 16)
        return "int16_t";
      if (dtype.bits() == 32)
        return "int";
      if (dtype.bits() == 64)
        return "int64_t";
    } else if (dtype.is_uint()) {
      if (dtype.bits() == 8)
        return "uint8_t";
      if (dtype.bits() == 16)
        return "uint16_t";
      if (dtype.bits() == 32)
        return "uint32_t";
      if (dtype.bits() == 64)
        return "uint64_t";
    }
    return "";
  }

  // Check if a buffer load can be broadcast from 1D to 2D
  bool CanBroadcast(const BufferLoadNode *load,
                    const std::unordered_set<const VarNode *> &parallel_vars,
                    int64_t *broadcast_dim) {
    if (load->buffer->shape.size() != 1) {
      return false; // Only 1D buffers can be broadcast
    }

    if (load->indices.size() != 1) {
      return false; // Must have exactly 1 index for 1D buffer
    }

    const PrimExpr &index = load->indices[0];

    // Check if the index is a simple variable (no offset or complex
    // expressions)
    if (auto var = index.as<VarNode>()) {
      // Check if it's the vector dimension variable (broadcast along outer dim,
      // axis=0) Source should be [1, inner_vec_len] and broadcast along axis 0
      if (vector_dim_var_ != nullptr && var == vector_dim_var_) {
        *broadcast_dim = 0; // Broadcast along axis 0 (outer dimension)
        return true;
      }
      // Check if it's the outer dimension variable (broadcast along inner dim,
      // axis=1) Source should be [outer_extent, 1] and broadcast along axis 1
      if (outer_dim_var_ != nullptr && var == outer_dim_var_) {
        *broadcast_dim = 1; // Broadcast along axis 1 (inner dimension)
        return true;
      }
    }

    // If the index is not a simple variable, it might have offset or be
    // discrete access In that case, we should not use broadcasting and fall
    // back to looping
    return false;
  }

  // Create a 2D view of a 1D buffer for broadcast operation
  Buffer CreateBroadcastSourceBuffer(const Buffer &src_1d,
                                     int64_t broadcast_dim) {
    // Use the same data pointer and dtype as the source buffer
    // but create a 2D shape with one dimension = 1
    DataType dtype = src_1d->dtype;

    int64_t src_elements = 0;
    if (auto imm = src_1d->shape[0].as<IntImmNode>()) {
      src_elements = imm->value;
    } else {
      LOG(FATAL) << "Source buffer shape must be constant for broadcast";
    }

    Array<PrimExpr> shape;
    if (broadcast_dim == 0) {
      // Broadcasting along axis 0 (outer dimension): view as [1, src_elements]
      shape.push_back(IntImm(DataType::Int(32), 1));
      shape.push_back(IntImm(DataType::Int(32), src_elements));
    } else {
      // Broadcasting along axis 1 (inner dimension): view as [src_elements, 1]
      shape.push_back(IntImm(DataType::Int(32), src_elements));
      shape.push_back(IntImm(DataType::Int(32), 1));
    }

    Buffer buf = Buffer(src_1d->data, // Use the same data pointer
                        dtype, shape,
                        /*strides=*/{},
                        /*elem_offset=*/PrimExpr(0),
                        /*name=*/src_1d->name + "_broadcast_view",
                        /*data_alignment=*/0,
                        /*offset_factor=*/0,
                        /*buffer_type=*/kDefault);

    return buf;
  }

  // Create a broadcast buffer with the same dtype as the source buffer
  Buffer CreateBroadcastBuffer(const Buffer &ref, int64_t outer_extent,
                               int64_t inner_vec_len) {
    // Use the same dtype as the source buffer
    DataType dtype = ref->dtype;

    Var data(ref->name + "_broadcast_" + std::to_string(temp_buffer_id_++) +
                 "_data",
             PointerType(PrimType(dtype), "shared"));

    // Create 2D shape for broadcast
    int64_t total_elements = outer_extent * inner_vec_len;
    Array<PrimExpr> shape;
    shape.push_back(IntImm(DataType::Int(32), outer_extent));
    shape.push_back(IntImm(DataType::Int(32), inner_vec_len));
    Buffer buf = Buffer(data, dtype, shape,
                        /*strides=*/{},
                        /*elem_offset=*/PrimExpr(0),
                        /*name=*/data->name_hint,
                        /*data_alignment=*/0,
                        /*offset_factor=*/0,
                        /*buffer_type=*/kDefault);

    temp_buffers_.push_back(buf);
    return buf;
  }

  // Create a broadcast workspace buffer with uint8 type for temporary storage
  Buffer CreateBroadcastWorkspaceBuffer(int64_t outer_extent,
                                        int64_t inner_vec_len,
                                        DataType dst_dtype) {
    // Use uint8 type for the workspace buffer
    DataType dtype = DataType::UInt(8);

    Var data("broadcast_workspace_" + std::to_string(temp_buffer_id_++) +
                 "_data",
             PointerType(PrimType(dtype), "shared"));

    // Workspace buffer should be 2x the size of the dst buffer
    // Create 1D shape (vector operations expect contiguous buffers)
    int64_t total_elements = outer_extent * inner_vec_len;
    int64_t workspace_elements = 2 * total_elements;
    Array<PrimExpr> shape;
    shape.push_back(IntImm(DataType::Int(32), workspace_elements));

    Buffer buf = Buffer(data, dtype, shape,
                        /*strides=*/{},
                        /*elem_offset=*/PrimExpr(0),
                        /*name=*/data->name_hint,
                        /*data_alignment=*/0,
                        /*offset_factor=*/0,
                        /*buffer_type=*/kDefault);

    temp_buffers_.push_back(buf);
    return buf;
  }

  // Generate broadcast statement to broadcast
  Stmt GenerateBroadcastStmt(const Buffer &src_1d, const Buffer &dst_2d,
                             const Buffer &workspace, int64_t broadcast_dim,
                             int64_t outer_extent, int64_t inner_vec_len) {
    // Format: tir.call_intrin("handle", tl.ascend_broadcast(),
    // "Broadcast<{dtype}, 2, {axis}, false>",
    //                         dst.access_ptr("w"), src.access_ptr("r"),
    //                         tmp.access_ptr("r"), dim, dst_shape[0],
    //                         dst_shape[1], ..., src_shape[0], ...)
    Array<PrimExpr> broadcast_args;

    // Create a 2D view of the source buffer for broadcast operation
    Buffer src_2d_view = CreateBroadcastSourceBuffer(src_1d, broadcast_dim);

    // 0. Template argument string
    DataType dtype = src_1d->dtype;
    std::string dtype_str = DTypeToAscendCString(src_1d->dtype);
    std::string template_args =
        dtype_str + ", 2, " + std::to_string(broadcast_dim) + ", false";
    broadcast_args.push_back(StringImm("Broadcast<" + template_args + ">"));

    // 1. dst buffer access ptr (1D buffer, but broadcast op treats it as 2D
    // based on shape args)
    int64_t total_elements = outer_extent * inner_vec_len;
    broadcast_args.push_back(CreateAccessPtr(
        dst_2d, dtype, IntImm(DataType::Int(32), 0), total_elements, 2));

    // 2. src buffer access ptr (2D view)
    int64_t src_elements = 0;
    if (auto imm = src_1d->shape[0].as<IntImmNode>()) {
      src_elements = imm->value;
    } else {
      LOG(FATAL) << "Source buffer shape must be constant for broadcast";
    }
    broadcast_args.push_back(CreateAccessPtr(
        src_2d_view, dtype, IntImm(DataType::Int(32), 0), src_elements, 2));

    // 3. tmp buffer access ptr (workspace buffer, 1D shape)
    // Workspace buffer should be 2x the size of the dst buffer
    int64_t workspace_elements = 2 * total_elements;
    broadcast_args.push_back(CreateAccessPtr(workspace, DataType::UInt(8),
                                             IntImm(DataType::Int(32), 0),
                                             workspace_elements, 1));

    // 4. dim (number of dimensions)
    broadcast_args.push_back(IntImm(DataType::Int(32), 2));

    // 5. dst shape array (explicit 2D shape for broadcast operation)
    broadcast_args.push_back(IntImm(DataType::Int(32), outer_extent));
    broadcast_args.push_back(IntImm(DataType::Int(32), inner_vec_len));

    // 6. src shape array (explicit 2D shape for broadcast operation)
    // Use the shape from the 2D view of the source buffer
    broadcast_args.push_back(src_2d_view->shape[0]);
    broadcast_args.push_back(src_2d_view->shape[1]);

    PrimExpr broadcast_call =
        Call(DataType::Handle(), tl::ascend_broadcast(), broadcast_args);
    return Evaluate(broadcast_call);
  }

  // Structure to hold broadcast information
  struct BroadcastInfo {
    const BufferLoadNode *load;
    Buffer broadcast_buffer;
    Buffer workspace_buffer;
    int64_t broadcast_dim;
    int64_t outer_extent;
    int64_t inner_vec_len;
  };

  bool IsScalar(const PrimExpr &expr) {
    return expr.as<IntImmNode>() || expr.as<FloatImmNode>() ||
           expr.as<VarNode>();
  }

  bool IsZero(const PrimExpr &expr) {
    if (auto imm = expr.as<IntImmNode>()) {
      return imm->value == 0;
    }
    if (auto imm = expr.as<FloatImmNode>()) {
      return imm->value == 0.0;
    }
    return false;
  }

  // Check if a buffer is in Global Memory (GM)
  bool IsGlobalMemoryBuffer(const Buffer &buffer) {
    if (auto *ptr_type = buffer->data->type_annotation.as<PointerTypeNode>()) {
      return ptr_type->storage_scope == "global" ||
             ptr_type->storage_scope.empty();
    }
    // If there's no PointerType annotation, it's global by default
    return true;
  }

  // Check if a buffer is Unified Buffer(UB)
  bool IsUnifiedBuffer(const Buffer &buffer) {
    if (auto *ptr_type = buffer->data->type_annotation.as<PointerTypeNode>()) {
      return ptr_type->storage_scope == "shared";
    }
    return false;
  }

  // Check if a buffer is L1
  bool IsL1Buffer(const Buffer &buffer) {
    if (auto *ptr_type = buffer->data->type_annotation.as<PointerTypeNode>()) {
      return ptr_type->storage_scope == "shared.dyn";
    }
    return false;
  }

  // Check if a buffer is L0C
  bool IsL0CBuffer(const Buffer &buffer) {
    if (auto *ptr_type = buffer->data->type_annotation.as<PointerTypeNode>()) {
      return ptr_type->storage_scope == "wmma.accumulator";
    }
    return false;
  }

  // Generate ascend_copy call from UB to GM
  Stmt GenerateAscendCopy(const Buffer &src_ub, const Buffer &dst_gm,
                          const PrimExpr &src_offset,
                          const PrimExpr &dst_offset, int64_t element_count,
                          bool is_2d = false,
                          const Array<PrimExpr> &src_indices = {}) {
    // Create T.region expressions for ascend_copy
    // The format is: T.region(buffer_load_with_indices, access_mask, extent0,
    // extent1, ...) ascend_copy expects: tl.ascend_copy(src_region, dst_region,
    // enable_relu)

    // Create source region (UB) - start from [0, 0] with the temp buffer's
    // shape
    Array<PrimExpr> actual_src_indices;
    Array<PrimExpr> src_extents;

    bool is_global_output = IsGlobalMemoryBuffer(dst_gm);
    if (src_indices.empty()) {
      if (src_ub->shape.size() == 1) {
        actual_src_indices.push_back(IntImm(DataType::Int(32), 0));
        src_extents.push_back(IntImm(DataType::Int(32), element_count));
      } else if (src_ub->shape.size() == 2) {
        // Temp buffer has shape [outer_extent, inner_vec_len]
        actual_src_indices.push_back(IntImm(DataType::Int(32), 0));
        actual_src_indices.push_back(IntImm(DataType::Int(32), 0));
        src_extents.push_back(src_ub->shape[0]); // outer_extent
        src_extents.push_back(src_ub->shape[1]); // inner_vec_len
      }
    } else {
      if (src_ub->shape.size() == 1) {
        actual_src_indices = src_indices;
        src_extents.push_back(IntImm(DataType::Int(32), element_count));
      } else if (src_ub->shape.size() == 2) {
        PrimExpr row = floordiv(src_offset, src_ub->shape[1]);
        PrimExpr col = truncmod(src_offset, src_ub->shape[1]);
        actual_src_indices.push_back(row);
        actual_src_indices.push_back(col);
        src_extents.push_back(dst_gm->shape[0]); // outer_extent
        src_extents.push_back(dst_gm->shape[1]); // inner_vec_len
      }
    }

    // Create source BufferLoad
    PrimExpr src_load = BufferLoad(src_ub, actual_src_indices);

    // Create destination region (GM) - use the computed offset
    Array<PrimExpr> dst_indices;
    Array<PrimExpr> dst_extents;

    if (src_indices.empty()) {
      if (dst_gm->shape.size() == 1) {
        dst_indices.push_back(dst_offset);
        dst_extents.push_back(IntImm(DataType::Int(32), element_count));
      } else if (dst_gm->shape.size() == 2) {
        // Convert linear offset to 2D indices
        PrimExpr row = floordiv(dst_offset, dst_gm->shape[1]);
        PrimExpr col = truncmod(dst_offset, dst_gm->shape[1]);
        dst_indices.push_back(row);
        dst_indices.push_back(col);
        // Use the same extents as the source buffer (the tile size)
        dst_extents.push_back(src_ub->shape[0]);
        dst_extents.push_back(src_ub->shape[1]);
      }
    } else {
      if (dst_gm->shape.size() == 1) {
        dst_indices.push_back(IntImm(DataType::Int(32), 0));
        dst_extents.push_back(IntImm(DataType::Int(32), element_count));
      } else if (dst_gm->shape.size() == 2) {
        dst_indices.push_back(IntImm(DataType::Int(32), 0));
        dst_indices.push_back(IntImm(DataType::Int(32), 0));
        dst_extents.push_back(dst_gm->shape[0]);
        dst_extents.push_back(dst_gm->shape[1]);
      }
    }

    // Create destination BufferLoad
    PrimExpr dst_load = BufferLoad(dst_gm, dst_indices);

    // Create T.region calls
    // Format: T.region(buffer_load, access_mask, extent0, extent1, ...)
    // access_mask: 1 for read, 2 for write

    auto CreateRegionCall = [&](const PrimExpr &load, int access_mask,
                                const Array<PrimExpr> &extents) -> PrimExpr {
      Array<PrimExpr> args;
      args.push_back(load);
      args.push_back(IntImm(DataType::Int(32), access_mask));
      for (const auto &ext : extents) {
        args.push_back(ext);
      }
      return Call(DataType::Handle(), Op::Get("tl.region"), args);
    };

    PrimExpr src_region =
        CreateRegionCall(src_load, 1, src_extents); // 1 = read
    PrimExpr dst_region =
        CreateRegionCall(dst_load, 2, dst_extents); // 2 = write

    // Create the ascend_copy call
    // Format: tir.call_intrin("handle", tir.op.Op.get("tl.ascend_copy"), src,
    // dst, enable_relu)
    Array<PrimExpr> copy_args;
    copy_args.push_back(src_region);
    copy_args.push_back(dst_region);
    copy_args.push_back(Bool(false)); // enable_relu = false

    PrimExpr copy_call =
        Call(DataType::Handle(), Op::Get("tl.ascend_copy"), copy_args);

    return Evaluate(copy_call);
  }
};

using namespace tir::transform;

tvm::transform::Pass AscendLowerParallelToVector() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto new_func = AscendLowerParallelToVector::Substitute(std::move(f));
    return new_func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendLowerParallelToVector", {});
}

TVM_REGISTER_GLOBAL("tl.transform.AscendLowerParallelToVector")
    .set_body_typed(AscendLowerParallelToVector);

} // namespace tl
} // namespace tvm
