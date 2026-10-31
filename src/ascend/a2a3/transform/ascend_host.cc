// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file ascend_host.cc
 * \brief host specialized for Ascend npu
 */

#include <unordered_set>
#include <utility>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "arith/ir_mutator_with_analyzer.h"
#include "tirx/analysis/var_use_def_analysis.h"

namespace tvm {
namespace tl {
namespace a2a3 {

using namespace tirx;
using namespace ffi;

class HostProcesser : arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    HostProcesser substituter(&analyzer);
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = substituter(f->body);
    f = WithAttr(std::move(f), "tiling_map", substituter.tiling_map_);
    f = WithAttr(std::move(f), "var_sequence", substituter.var_sequence_);
    return f;
  }

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  bool isNeedTiling(const Var &var, const PrimExpr &value) {
    auto check_var = [this](const tirx::VarNode *v) {
      return this->cmp_set_.count(GetRef<Var>(v));
    };
    if (UsesVar(value, check_var)) {
      this->cmp_set_.insert(var);
      return false;
    }
    if (value->IsInstance<IntImmNode>()) {
      this->cmp_set_.insert(var);
      return false;
    }
    return true;
  }

  Stmt VisitStmt_(const BindNode *op) final {
    // disable temporarily
    return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    if (isNeedTiling(op->loop_var, op->min) ||
        isNeedTiling(op->loop_var, op->extent)) {
      return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == "thread_extent") {
      IterVar iv = Downcast<IterVar>(op->node);
      cmp_set_.insert(iv->var);
      after_thread_flag = true;
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Map<Var, PrimExpr> tiling_map_;
  Array<Var> var_sequence_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> cmp_set_;
  bool after_thread_flag = false;
};

using namespace tirx::transform;

tvm::transform::Pass A2A3HostLegalize() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto new_func = HostProcesser::Substitute(std::move(f));
    return new_func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.A2A3HostLegalize", {});
}

// regist host path
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.A2A3HostLegalize", A2A3HostLegalize);
}

} // namespace a2a3
} // namespace tl
} // namespace tvm
