// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file lower_npuir_block.cc
 * \brief Strip residual TIR Block/BlockRealize shells for NPUIR codegen.
 */

#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

using namespace tir;

class LowerNpuirBlockMutator : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {
    LowerNpuirBlockMutator mutator;
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = mutator.VisitStmt(f->body);
    return f;
  }

private:
  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    Stmt body = VisitStmt(op->block);
    PrimExpr predicate = VisitExpr(op->predicate);
    if (!is_one(predicate)) {
      body = IfThenElse(predicate, std::move(body));
    }
    return body;
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    Stmt body = VisitStmt(op->body);
    if (op->init.defined()) {
      Array<Stmt> stmts{VisitStmt(op->init.value()), body};
      body = SeqStmt::Flatten(stmts);
    }
    return body;
  }
};

namespace transform {

tvm::transform::Pass LowerNpuirBlock() {
  auto pass_func = [=](PrimFunc f, IRModule m,
                       tvm::transform::PassContext ctx) {
    return LowerNpuirBlockMutator::Substitute(std::move(f));
  };
  return tir::transform::CreatePrimFuncPass(pass_func, 0, "tl.LowerNpuirBlock",
                                            {});
}

TVM_REGISTER_GLOBAL("tl.transform.LowerNpuirBlock")
    .set_body_typed(LowerNpuirBlock);

} // namespace transform
} // namespace tl
} // namespace tvm
