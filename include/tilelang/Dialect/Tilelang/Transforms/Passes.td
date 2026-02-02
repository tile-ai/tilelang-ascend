#ifndef TILELANG_PASSES
#define TILELANG_PASSES

include "mlir/Pass/PassBase.td"

def TilelangCombineOps : Pass</*cli-arg*/"tilelang-combine", /*Op*/"mlir::ModuleOp"> {
  let summary = "combine ops";
  let description = [{
    This pass aims to optimize the five following patterns:
    - `dot(a, b, 0) + c => dot(a, b, c)`

    - `addptr(addptr(ptr, idx0), idx1) => addptr(ptr, AddI(idx0, idx1))`

    - `select(cond, load(ptrs, broadcast(cond), ???), other) =>
         load(ptrs, broadcast(cond), other)`

    - `broadcast(constant) => reshaped_constant`
    - `torch.sum(x[:,:,None].expand(-1,-1,n) * y[None,:,:].expand(m,-1,-1),1)
       => dot(x,y,splat(0))`
  }];

  let constructor = "mlir::tilelang::createCombineOpsPass()";

  let dependentDialects = ["mlir::arith::ArithDialect"];
}

def TilelangReorderBroadcast : Pass</*cli-arg*/"tilelang-reorder-broadcast", /*Op*/"mlir::ModuleOp"> {
  let summary = "Moves broadcast and splat after elementwise operations";
  let description = [{
    The purpose of this pass is to transform:
      - `elementwise(broadcast(a)) => broadcast(elementwise(a))`
      - `elementwise(splat(a), splat(b), ...) => splat(elementwise(a, b, ...))`
    In the event of a match, the broadcast (or splat) operation is delayed
    and performed after the ElementWise operation.
  }];
  let constructor = "mlir::tilelang::createReorderBroadcastPass()";
  let dependentDialects = ["mlir::tilelang::TilelangDialect"];
}

def TilelangRewriteTensorPointer : Pass</*cli-arg*/"tilelang-rewrite-tensor-pointer", /*Op*/"mlir::ModuleOp"> {
  let summary = "Rewrite load/stores with tensor pointers into legacy load/stores";
  let description = [{
    This pass rewrites all load/store semantics initiated by a `tt.make_tensor_ptr` and `tt.advance` into legacy
    semantics. After this pass, `tt.make_tensor_ptr` and `tt.advance` will disappear, and it generates logics to compute
    the pointer/mask/other for each load/store.
  }];

  let constructor = "mlir::tilelang::createRewriteTensorPointerPass()";

  let dependentDialects = ["mlir::tilelang::TilelangDialect"];
}

def TilelangLoopUnroll : Pass</*cli-arg*/"tilelang-loop-unroll", /*Op*/"mlir::ModuleOp"> {
  let summary = "Loop unroller";
  let description = [{
    The pass unrolls a scf loop with tt.loop_unroll_factor attribute. The attribute specialises how many iterations
    the loop should be unrolled.
  }];
  let constructor = "mlir::tilelang::createLoopUnrollPass()";
  let dependentDialects = ["mlir::tilelang::TilelangDialect"];
}

#endif
