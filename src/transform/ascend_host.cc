// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file ascend_host.cc
 * \brief host specialized for Ascend npu
 */

#include "arith/ir_mutator_with_analyzer.h"
#include "tir/analysis/var_use_def_analysis.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include "../op/builtin.h"
#include "./common/collector.h"

namespace tvm {
namespace tl {

using namespace tir;

class HostProcesser : arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    HostProcesser substituter(&analyzer);
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = substituter.VisitStmt(f->body);
    auto fn_attr = fptr->attrs.CopyOnWrite();
    fn_attr->dict.Set("tiling_map", substituter.tiling_map_);
    return f;
  }

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  bool isNeedTiling(const Var &var, const PrimExpr &value) {
    auto check_var = [this](const tir::VarNode* v) {
      return this->cmp_set_.count(v);
    };
    if (UsesVar(value, check_var)) {
      this->cmp_set_.insert(var.get());
      return false;
    }
    if (value->IsInstance<IntImmNode>()) {
      this->cmp_set_.insert(var.get());
      return false;
    }
    return true;
  }

  Stmt VisitStmt_(const LetStmtNode *op) final {
    if (isNeedTiling(op->var, op->value) && after_thread_flag) {
      tiling_map_.Set(op->var, op->value);
      return arith::IRMutatorWithAnalyzer::VisitStmt(op->body);
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    if (isNeedTiling(op->loop_var, op->min) || isNeedTiling(op->loop_var, op->extent)) {
      return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == "thread_extent") {
      IterVar iv = Downcast<IterVar>(op->node);
      cmp_set_.insert((iv->var.get()));
      after_thread_flag = true;
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Map<Var, PrimExpr> tiling_map_;
  std::unordered_set<const VarNode*> cmp_set_;
  bool after_thread_flag = false;

};

using namespace tir::transform;

tvm::transform::Pass HostLegalize() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto new_func = HostProcesser::Substitute(std::move(f));
    return new_func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.HostLegalize", {});
}

// regist host path
TVM_REGISTER_GLOBAL("tl.transform.HostLegalize")
    .set_body_typed(HostLegalize);

}
}