#include "arith/ir_mutator_with_analyzer.h"
#include "tir/analysis/var_use_def_analysis.h"
#include "tir/transforms/ir_utils.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include "../op/builtin.h"
#include "../op/ascend.h"
#include "./common/collector.h"

#include <iostream>
#include <string>
#include <regex>
#include <stdexcept>
#include <climits>

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

const int VECTOR_SCOPE = 1;
const int STAGE_NUM = 2;

class VectorCorePipelineCollector : public StmtVisitor {
public:
  void VisitStmt_(const AttrStmtNode* op) override {
    if (op->attr_key == "resource_scope") {
      cur_resource_scope = op->value.as<IntImmNode>()->value;
    }
    this->VisitStmt(op->body);
    cur_resource_scope = -1;
  }

  void VisitStmt_(const ForNode* op) override {
    if (cur_resource_scope == VECTOR_SCOPE && op->annotations.Get("stage_loop")) {
      target_for_nodes.push_back(op);
    }
    this->VisitStmt(op->body);
  }

  std::vector<const ForNode*> get_target_for_nodes() {
    return target_for_nodes;
  }

private:
  int cur_resource_scope = -1;
  std::vector<const ForNode*> target_for_nodes;
};


class VectorCorePipeline : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f, PassContext ctx) {
    arith::Analyzer analyzer;
    VectorCorePipeline substituer(&analyzer);
    PrimFuncNode *fptr = f.CopyOnWrite();

    VectorCorePipelineCollector collector;
    collector(f->body);

    std::vector<const ForNode*> target_for_nodes = collector.get_target_for_nodes();
    substituer.set_target_for_nodes(collector.get_target_for_nodes());

    fptr->body = substituer.VisitStmt(f->body);
    return f;
  }

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;
  std::vector<const ForNode*> target_for_nodes_;

  void set_target_for_nodes(std::vector<const ForNode*> target_for_nodes) {
    target_for_nodes_ = target_for_nodes;
  }

  Stmt VisitStmt_(const ForNode* op) override {
    for (const ForNode* for_node : target_for_nodes_) {
      if (op->body.same_as(for_node->body)) {
        Stmt inner_for = SplitBuffersInFor(op);
        return For(
          op->loop_var, 
          op->min, 
          op->extent, 
          op->kind, 
          inner_for,
          op->thread_binding, 
          op->annotations
        );
      }
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt SplitBuffersInFor(const ForNode* op) {
    Stmt new_body = op->body;
    Array<Stmt> cross_core_wait_stmts;
    Array<Stmt> cross_core_set_stmts;
    Var new_var = Var("v_i", op->loop_var.dtype());
    if (auto seq_stmt = op->body.as<SeqStmtNode>()) {
      Array<Stmt> new_seq;
      for (size_t i = 0; i < seq_stmt->size(); ++i) {
        Stmt stmt = seq_stmt->seq[i];
        if (auto eval_node = stmt.as<EvaluateNode>()) {
          auto cross_core_wait_stmt = ProcessCrossCoreWaitStmt(eval_node);
          if (cross_core_wait_stmt.defined()) {
            cross_core_wait_stmts.push_back(cross_core_wait_stmt);
            continue;
          }
          auto cross_core_set_stmt = ProcessCrossCoreSetStmt(eval_node);
          if (cross_core_set_stmt.defined()) {
            cross_core_set_stmts.push_back(cross_core_set_stmt);
            continue;
          }
          stmt = ProcessEvaluateNode(eval_node, new_var);
        }
        new_seq.push_back(stmt);
      }
      new_body = SeqStmt(new_seq);
    }

    For for_stmt = For(
          new_var, 
          0,
          STAGE_NUM,
          ForKind::kSerial,
          new_body
        );

    return SeqStmt({
      SeqStmt::Flatten(cross_core_wait_stmts),
      for_stmt,
      SeqStmt::Flatten(cross_core_set_stmts),
    });
  }

  Stmt ProcessCrossCoreWaitStmt(const EvaluateNode* eval_node) {
    if (const CallNode* call_node = eval_node->value.as<CallNode>()) {
      if (call_node->op.same_as(tl::ascend_auto_wait_cross_flag())) {
        return Evaluate(Call(DataType::Handle(), call_node->op, call_node->args, call_node->span));
      }
    }
    return Stmt();
  }

  Stmt ProcessCrossCoreSetStmt(const EvaluateNode* eval_node) {
    if (const CallNode* call_node = eval_node->value.as<CallNode>()) {
      if (call_node->op.same_as(tl::ascend_auto_set_cross_flag())) {
        return Evaluate(Call(DataType::Handle(), call_node->op, call_node->args, call_node->span));
      }
    }
    return Stmt();
  }

  Stmt ProcessEvaluateNode(const EvaluateNode* eval_node, Var new_var) {
    if (const CallNode* call_node = eval_node->value.as<CallNode>()) {
      return Evaluate(ProcessCallNode(call_node, new_var));
    }
    return Evaluate(eval_node->value);
  }

  Call ProcessCallNode(const CallNode* call_node, Var new_var) {
    if (call_node->op.same_as(tir::builtin::call_extern())) {
      return ProcessCallNodeOpCallExtern(call_node, new_var);
    }
    if (call_node->op.same_as(tl::ascend_add())
      || call_node->op.same_as(tl::ascend_sub())
      || call_node->op.same_as(tl::ascend_mul())
      || call_node->op.same_as(tl::ascend_div())
      || call_node->op.same_as(tl::ascend_max())
      || call_node->op.same_as(tl::ascend_min())
    ) {
      return ProcessCallNodeOpBinary(call_node, new_var);
    }
    if (call_node->op.same_as(tl::ascend_exp())) {
      return ProcessCallNodeOpUnary(call_node, new_var);
    }
    if (call_node->op.same_as(tl::ascend_muls())) {
      return ProcessCallNodeOpAscendMuls(call_node, new_var);
    }
    if (call_node->op.same_as(tl::ascend_reduce())) {
      return ProcessCallNodeOpAscendReduce(call_node, new_var);
    }
    if (call_node->op.same_as(tl::ascend_broadcast())) {
      return ProcessCallNodeOpAscendBroadcast(call_node, new_var);
    }
    if (call_node->op.same_as(tl::ascend_auto_set_flag())
      || call_node->op.same_as(tl::ascend_auto_wait_flag())) {
      return ProcessCallNodeOpAscendAutoSetWaitFlag(call_node, new_var);
    }
    return Call(call_node->dtype, call_node->op, call_node->args, call_node->span);
  }

  Call ProcessCallNodeOpUnary(const CallNode* call_node, Var new_var) {
    auto old_size = call_node->args[2].as<IntImmNode>()->value;
    int new_size = old_size / STAGE_NUM;

    auto arg0_arg2 = call_node->args[0].as<CallNode>()->args[2];
    auto new_arg0_arg2 = Add(arg0_arg2, Mul(new_var, new_size));

    auto arg1_arg2 = call_node->args[1].as<CallNode>()->args[2];
    auto new_arg1_arg2 = Add(arg1_arg2, Mul(new_var, new_size));

    auto new_arg0 = Call(
      DataType::Handle(), 
      tir::builtin::tvm_access_ptr(), 
      {
        call_node->args[0].as<CallNode>()->args[0], 
        call_node->args[0].as<CallNode>()->args[1], 
        new_arg0_arg2, // offset
        call_node->args[0].as<CallNode>()->args[3],
        call_node->args[0].as<CallNode>()->args[4],
      }
    );
    auto new_arg1 = Call(
      DataType::Handle(), 
      tir::builtin::tvm_access_ptr(), 
      {
        call_node->args[1].as<CallNode>()->args[0], 
        call_node->args[1].as<CallNode>()->args[1], 
        new_arg1_arg2, // offset
        call_node->args[1].as<CallNode>()->args[3],
        call_node->args[1].as<CallNode>()->args[4],
      }
    );
    return Call(call_node->dtype, call_node->op, {new_arg0, new_arg1, new_size}, call_node->span);
  }

  Call ProcessCallNodeOpBinary(const CallNode* call_node, Var new_var) {
    auto old_size = call_node->args[3].as<IntImmNode>()->value;
    int new_size = old_size / STAGE_NUM;
    // c
    auto arg0_arg2 = call_node->args[0].as<CallNode>()->args[2];
    auto new_arg0_arg2 = Add(arg0_arg2, Mul(new_var, new_size));

    // a
    auto arg1_arg2 = call_node->args[1].as<CallNode>()->args[2];
    auto new_arg1_arg2 = Add(arg1_arg2, Mul(new_var, new_size));

    // b
    auto arg2_arg2 = call_node->args[2].as<CallNode>()->args[2];
    auto new_arg2_arg2 = Add(arg2_arg2, Mul(new_var, new_size));

    auto new_arg0 = Call(
      DataType::Handle(), 
      tir::builtin::tvm_access_ptr(), 
      {
        call_node->args[0].as<CallNode>()->args[0], 
        call_node->args[0].as<CallNode>()->args[1], 
        new_arg0_arg2, // offset
        call_node->args[0].as<CallNode>()->args[3],
        call_node->args[0].as<CallNode>()->args[4],
      }
    );
    auto new_arg1 = Call(
      DataType::Handle(), 
      tir::builtin::tvm_access_ptr(), 
      {
        call_node->args[1].as<CallNode>()->args[0], 
        call_node->args[1].as<CallNode>()->args[1], 
        new_arg1_arg2, // offset
        call_node->args[1].as<CallNode>()->args[3],
        call_node->args[1].as<CallNode>()->args[4],
      }
    );
    auto new_arg2 = Call(
      DataType::Handle(), 
      tir::builtin::tvm_access_ptr(), 
      {
        call_node->args[2].as<CallNode>()->args[0], 
        call_node->args[2].as<CallNode>()->args[1], 
        new_arg2_arg2, // offset
        call_node->args[2].as<CallNode>()->args[3],
        call_node->args[2].as<CallNode>()->args[4],
      }
    );
    return Call(
      call_node->dtype, 
      call_node->op, 
      {
        new_arg0, 
        new_arg1, 
        new_arg2,
        new_size,
      }, 
      call_node->span
    );
  }

  Call ProcessCallNodeOpAscendMuls(const CallNode* call_node, Var new_var) {
    auto old_size = call_node->args[3].as<IntImmNode>()->value;
    int new_size = old_size / STAGE_NUM;

    auto arg0_arg2 = call_node->args[0].as<CallNode>()->args[2];
    auto new_arg0_arg2 = Add(arg0_arg2, Mul(new_var, new_size));

    auto arg1_arg2 = call_node->args[1].as<CallNode>()->args[2];
    auto new_arg1_arg2 = Add(arg1_arg2, Mul(new_var, new_size));

    auto new_arg0 = Call(
      DataType::Handle(), 
      tir::builtin::tvm_access_ptr(), 
      {
        call_node->args[0].as<CallNode>()->args[0], 
        call_node->args[0].as<CallNode>()->args[1], 
        new_arg0_arg2, // offset
        call_node->args[0].as<CallNode>()->args[3],
        call_node->args[0].as<CallNode>()->args[4],
      }
    );
    auto new_arg1 = Call(
      DataType::Handle(), 
      tir::builtin::tvm_access_ptr(), 
      {
        call_node->args[1].as<CallNode>()->args[0], 
        call_node->args[1].as<CallNode>()->args[1], 
        new_arg1_arg2, // offset
        call_node->args[1].as<CallNode>()->args[3],
        call_node->args[1].as<CallNode>()->args[4],
      }
    );
    return Call(
      call_node->dtype, 
      call_node->op, 
      {
        new_arg0, 
        new_arg1,
        call_node->args[2],
        new_size,
      }, 
      call_node->span
    );
  }

  Call ProcessCallNodeOpAscendReduce(const CallNode* call_node, Var new_var) {
    auto arg0 = call_node->args[0].as<StringImmNode>()->value;
    std::vector<int> old_shapes = extractNumbersFromTemplate(arg0);
    std::vector<int> new_shapes = {old_shapes[0] / STAGE_NUM, old_shapes[1], old_shapes[2]};

    // dst
    auto arg1_arg2 = call_node->args[1].as<CallNode>()->args[2];
    auto new_arg1_arg2 = Add(arg1_arg2, Mul(new_var, new_shapes[0]));

    // src
    auto arg2_arg2 = call_node->args[2].as<CallNode>()->args[2];
    auto new_arg2_arg2 = Add(arg2_arg2, Mul(new_var, new_shapes[0] * new_shapes[1]));

    auto new_arg0 = StringImm(replaceTemplateNumbers(arg0, new_shapes));
    auto new_arg1 = Call(
      DataType::Handle(), 
      tir::builtin::tvm_access_ptr(), 
      {
        call_node->args[1].as<CallNode>()->args[0], 
        call_node->args[1].as<CallNode>()->args[1], 
        new_arg1_arg2, // offset
        call_node->args[1].as<CallNode>()->args[3],
        call_node->args[1].as<CallNode>()->args[4],
      }
    );
    auto new_arg2 = Call(
      DataType::Handle(), 
      tir::builtin::tvm_access_ptr(), 
      {
        call_node->args[2].as<CallNode>()->args[0], 
        call_node->args[2].as<CallNode>()->args[1], 
        new_arg2_arg2, // offset
        call_node->args[2].as<CallNode>()->args[3],
        call_node->args[2].as<CallNode>()->args[4],
      }
    );

    return Call(
      call_node->dtype, 
      call_node->op, 
      {
        new_arg0, 
        new_arg1,
        new_arg2,
        call_node->args[3],
      }, 
      call_node->span
    );
  }

  Call ProcessCallNodeOpAscendBroadcast(const CallNode* call_node, Var new_var) {
    auto arg0 = call_node->args[0].as<StringImmNode>()->value;
    std::vector<int> template_params = extractNumbersFromTemplate(arg0);
    
    int brct_dim = template_params[0];
    int brct_axis = template_params[1];

    if (brct_dim != 2 || brct_axis != 1) {
      throw std::runtime_error(
        "Only support broadcasting with dim=2 and axis=1, but got dim=" + std::to_string(brct_dim) 
        + " and axis=" + std::to_string(brct_axis)
      );
    }

    int dst_shape_M = call_node->args[5].as<IntImmNode>()->value / STAGE_NUM;
    int dst_shape_N= call_node->args[6].as<IntImmNode>()->value;
    int src_shape_M = call_node->args[7].as<IntImmNode>()->value / STAGE_NUM;
    int src_shape_N = call_node->args[8].as<IntImmNode>()->value;

    // dstShape
    auto arg1_arg2 = call_node->args[1].as<CallNode>()->args[2];
    auto new_arg1_arg2 = Add(arg1_arg2, Mul(new_var, dst_shape_M * dst_shape_N));

    // srcShape
    auto arg2_arg2 = call_node->args[2].as<CallNode>()->args[2];
    auto new_arg2_arg2 = Add(arg2_arg2, Mul(new_var, src_shape_M * src_shape_N));

    auto new_arg1 = Call(
      DataType::Handle(), 
      tir::builtin::tvm_access_ptr(), 
      {
        call_node->args[1].as<CallNode>()->args[0], 
        call_node->args[1].as<CallNode>()->args[1], 
        new_arg1_arg2, // offset
        call_node->args[1].as<CallNode>()->args[3],
        call_node->args[1].as<CallNode>()->args[4],
      }
    );
    auto new_arg2 = Call(
      DataType::Handle(), 
      tir::builtin::tvm_access_ptr(), 
      {
        call_node->args[2].as<CallNode>()->args[0], 
        call_node->args[2].as<CallNode>()->args[1], 
        new_arg2_arg2, // offset
        call_node->args[2].as<CallNode>()->args[3],
        call_node->args[2].as<CallNode>()->args[4],
      }
    );
    return Call(
      call_node->dtype, 
      call_node->op, 
      {
        call_node->args[0], 
        new_arg1,
        new_arg2,
        call_node->args[3],
        call_node->args[4],
        dst_shape_M,
        call_node->args[6],
        src_shape_M,
        call_node->args[8],
      }, 
      call_node->span
    );
  }

  Call ProcessCallNodeOpCallExtern(const CallNode* call_node, Var new_var) {
    std::string call_name = call_node->args[0].as<StringImmNode>()->value;
    if (call_name.find("copy_ub_to_ub") != std::string::npos) {
      return ProcessCallNodeOpCallExtern4CopyUbToUb(call_node, new_var);;
    }
    if (call_name.find("copy_gm_to_ub") != std::string::npos) {
      return ProcessCallNodeOpCallExtern4CopyGmToUb(call_node, new_var);
    }
    if (call_name.find("copy_ub_to_gm") != std::string::npos) {
      return ProcessCallNodeOpCallExtern4CopyUbToGm(call_node, new_var);
    }
    return Call(call_node->dtype, call_node->op, call_node->args, call_node->span);

  }

  Call ProcessCallNodeOpCallExtern4CopyUbToUb(const CallNode* call_node, Var new_var) {
    auto arg0 = call_node->args[0].as<StringImmNode>()->value;
    std::vector<int> old_shapes = extractNumbersFromTemplate(arg0);
    std::vector<int> new_shapes = splitLastElement(old_shapes);
    auto new_size = calculateVectorProduct(new_shapes);

    // read
    auto arg1_arg2 = call_node->args[1].as<CallNode>()->args[2];
    auto new_arg1_arg2 = Add(arg1_arg2, Mul(new_var, new_size));

    // write
    auto arg2_arg2 = call_node->args[2].as<CallNode>()->args[2];
    auto new_arg2_arg2 = Add(arg2_arg2, Mul(new_var, new_size));

    auto new_arg0 = StringImm(replaceTemplateNumbers(arg0, new_shapes));
    auto new_arg1 = Call(
      DataType::Handle(), 
      tir::builtin::tvm_access_ptr(), 
      {
        call_node->args[1].as<CallNode>()->args[0], 
        call_node->args[1].as<CallNode>()->args[1], 
        new_arg1_arg2, // offset
        call_node->args[1].as<CallNode>()->args[3],
        call_node->args[1].as<CallNode>()->args[4],
      }
    );
    auto new_arg2 = Call(
      DataType::Handle(), 
      tir::builtin::tvm_access_ptr(), 
      {
        call_node->args[2].as<CallNode>()->args[0], 
        call_node->args[2].as<CallNode>()->args[1], 
        new_arg2_arg2, // offset
        call_node->args[2].as<CallNode>()->args[3],
        call_node->args[2].as<CallNode>()->args[4],
      }
    );

    return Call(
      call_node->dtype,
      call_node->op,
      {new_arg0, new_arg1, new_arg2},
      call_node->span
    );
  }

  Call ProcessCallNodeOpCallExtern4CopyGmToUb(const CallNode* call_node, Var new_var) {
    auto arg0 = call_node->args[0].as<StringImmNode>()->value;
    std::vector<int> old_shapes = extractNumbersFromTemplate(arg0);
    std::vector<int> new_shapes = splitLastElement(old_shapes);
    auto new_size = calculateVectorProduct(new_shapes);

    // read
    auto arg1_arg2 = call_node->args[1].as<CallNode>()->args[2];
    auto new_arg1_arg2 = Add(arg1_arg2, Mul(new_var, new_size));

    // write
    auto arg2_arg2 = call_node->args[2].as<CallNode>()->args[2];
    auto new_arg2_arg2 = Add(arg2_arg2, Mul(new_var, new_size));

    auto new_arg0 = StringImm(replaceTemplateNumbers(arg0, new_shapes));
    auto new_arg1 = Call(
      DataType::Handle(), 
      tir::builtin::tvm_access_ptr(), 
      {
        call_node->args[1].as<CallNode>()->args[0], 
        call_node->args[1].as<CallNode>()->args[1], 
        new_arg1_arg2, // offset
        call_node->args[1].as<CallNode>()->args[3],
        call_node->args[1].as<CallNode>()->args[4],
      }
    );
    auto new_arg2 = Call(
      DataType::Handle(), 
      tir::builtin::tvm_access_ptr(), 
      {
        call_node->args[2].as<CallNode>()->args[0], 
        call_node->args[2].as<CallNode>()->args[1], 
        new_arg2_arg2, // offset
        call_node->args[2].as<CallNode>()->args[3],
        call_node->args[2].as<CallNode>()->args[4],
      }
    );

    auto arg3 = call_node->args[3];
    auto arg4 = call_node->args[4].as<IntImmNode>()->value;
    auto new_arg4 = IntImm(DataType::Int(32), arg4 / 2);
    auto arg5 = call_node->args[5];

    return Call(
      call_node->dtype,
      call_node->op,
      {new_arg0, new_arg1, new_arg2, arg3, new_arg4, arg5},
      call_node->span
    );
  }

  Call ProcessCallNodeOpCallExtern4CopyUbToGm(const CallNode* call_node, Var new_var) {
    auto arg0 = call_node->args[0].as<StringImmNode>()->value;
    std::vector<int> old_shapes = extractNumbersFromTemplate(arg0);
    std::vector<int> new_shapes = splitLastElement(old_shapes);
    auto new_size = calculateVectorProduct(new_shapes);

    // read
    auto arg1_arg2 = call_node->args[1].as<CallNode>()->args[2];
    auto new_arg1_arg2 = Add(arg1_arg2, Mul(new_var, new_size));

    // write
    auto arg2_arg2 = call_node->args[2].as<CallNode>()->args[2];
    auto new_arg2_arg2 = Add(arg2_arg2, Mul(new_var, new_size));

    auto new_arg0 = StringImm(replaceTemplateNumbers(arg0, new_shapes));
    auto new_arg1 = Call(
      DataType::Handle(), 
      tir::builtin::tvm_access_ptr(), 
      {
        call_node->args[1].as<CallNode>()->args[0], 
        call_node->args[1].as<CallNode>()->args[1], 
        new_arg1_arg2, // offset
        call_node->args[1].as<CallNode>()->args[3],
        call_node->args[1].as<CallNode>()->args[4],
      }
    );
    auto new_arg2 = Call(
      DataType::Handle(), 
      tir::builtin::tvm_access_ptr(), 
      {
        call_node->args[2].as<CallNode>()->args[0], 
        call_node->args[2].as<CallNode>()->args[1], 
        new_arg2_arg2, // offset
        call_node->args[2].as<CallNode>()->args[3],
        call_node->args[2].as<CallNode>()->args[4],
      }
    );

    auto arg3 = call_node->args[3];
    auto arg4 = call_node->args[4].as<IntImmNode>()->value;
    auto new_arg4 = IntImm(DataType::Int(32), arg4 / 2);
    auto arg5 = call_node->args[5];

    return Call(
      DataType::Handle(),
      tir::builtin::call_extern(),
      {new_arg0, new_arg1, new_arg2, arg3, new_arg4, arg5},
      call_node->span
    );
  }

  Call ProcessCallNodeOpAscendAutoSetWaitFlag(const CallNode* call_node, Var new_var) {
    return Call(
      call_node->dtype,
      call_node->op,
      {call_node->args[0], Add(call_node->args[1], new_var)},
      call_node->span
    );
  }

  std::vector<int> splitLastElement(const std::vector<int>& input) {
    if (input.empty()) {
      throw std::invalid_argument("input vector cannot be empty");
    }

    std::vector<int> result = input;
    result.back() = result.back() / STAGE_NUM;

    return result;
  }

  int calculateVectorProduct(const std::vector<int>& nums) {
      if (nums.empty()) {
          throw std::invalid_argument("input vector cannot be empty");
      }

      int product = 1;
      for (int num : nums) {
          if (num != 0 && (product > INT_MAX / num || product < INT_MIN / num)) {
              throw std::overflow_error("integer overflow detected during multiplication");
          }
          product *= num;
      }

      return product;
  }

  std::vector<int> extractNumbersFromTemplate(const std::string& input_str) {
    std::vector<int> extracted_numbers;
    
    std::regex bracket_pattern(R"(<([^>]*)>)");
    std::smatch bracket_matches;

    // Step1: Extract the content inside <>
    if (std::regex_search(input_str, bracket_matches, bracket_pattern)) {
      if (bracket_matches.size() >= 2) {
        std::string content_inside_brackets = bracket_matches[1].str();
        
        // Step2: Extract all numbers (including negative) from the content inside <>
        std::regex number_pattern(R"(-?\d+)");
        std::sregex_iterator it(content_inside_brackets.begin(), 
                              content_inside_brackets.end(), 
                              number_pattern);
        std::sregex_iterator end;

        // Get all matched numbers and convert them to integers
        for (; it != end; ++it) {
          try {
            int num = std::stoi((*it).str());
            extracted_numbers.push_back(num);
          } catch (const std::invalid_argument& e) {
            throw std::runtime_error("cannot convert string to integer: " + std::string(e.what()));
          } catch (const std::out_of_range& e) {
            throw std::runtime_error("integer out of range: " + std::string(e.what()));
          }
        }
      }
    }

    if (extracted_numbers.empty()) {
        throw std::runtime_error("cannot find any numbers in the input string: " + input_str);
    }

    return extracted_numbers;
  }

  std::string replaceTemplateNumbers(const std::string& input_str, const int* replace_nums, size_t arr_size) {
    // Step1: Find the template parameters (the part enclosed in <>)
    std::regex bracket_pattern(R"(<([^>]*)>)");
    std::smatch bracket_matches;
    
    if (!std::regex_search(input_str, bracket_matches, bracket_pattern)) {
      throw std::runtime_error("cannot find template parameters enclosed in <> in the input string: " + input_str);
    }
    
    // The original template content (e.g., "float, 128, 64")
    std::string original_template = bracket_matches[1].str();
    std::string new_template = original_template;
    
    // Step2: Find all numbers (including negative) in the original template
    std::regex number_pattern(R"(-?\d+)");
    std::sregex_iterator num_it(original_template.begin(), original_template.end(), number_pattern);
    std::sregex_iterator num_end;
    
    // Get the count of numbers in the original template
    size_t num_count = std::distance(num_it, num_end);
    if (num_count != arr_size) {
      throw std::runtime_error(
        "replacement array length(" + std::to_string(arr_size) + 
        ") does not match number count in template(" + std::to_string(num_count) + ")"
      );
    }
    
    // Step3: Replace each number in the original template with the corresponding number from replace_nums
    std::sregex_iterator it(original_template.begin(), original_template.end(), number_pattern);
    size_t replace_idx = 0;
    std::string temp_str = original_template;
    
    for (; it != num_end && replace_idx < arr_size; ++it, ++replace_idx) {
      const std::smatch& match = *it;
      temp_str.replace(match.position(), match.length(), std::to_string(replace_nums[replace_idx]));
    }
    
    // Step4: Reconstruct the final string by replacing the original template with the new template
    std::string result = std::regex_replace(input_str, bracket_pattern, "<" + temp_str + ">");
    return result;
  }

  std::string replaceTemplateNumbers(const std::string& input_str, const std::vector<int>& replace_nums) {
    return replaceTemplateNumbers(input_str, replace_nums.data(), replace_nums.size());
  }
};

tvm::transform::Pass VectorCorePipeline() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return VectorCorePipeline::Substitute(std::move(f), ctx);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.VectorCorePipeline", {});
}

TVM_REGISTER_GLOBAL("tl.transform.VectorCorePipeline")
  .set_body_typed(VectorCorePipeline);

}
}