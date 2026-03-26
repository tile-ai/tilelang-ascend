// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/lib/Transforms/SpecializeCube.cpp
 * \brief TileLangIR Specialize Cube ops pass.
 *
 */

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "tilelangir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

namespace mlir::tilelangir {

#define GEN_PASS_DEF_TILELANGIRSPECIALIZECUBE
#include "tilelangir/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tilelangir-specialize-cube"
#define LDBG(X)                                                                \
  LLVM_DEBUG(llvm::dbgs() << "[" << DEBUG_TYPE << "] " << X << '\n')

struct TileLangIRSpecializeCube
    : impl::TileLangIRSpecializeCubeBase<TileLangIRSpecializeCube> {

  void runOnOperation() override {
    auto getAddressSpace = [](Value val) {
      return cast<hivm::AddressSpaceAttr>(
                 cast<BaseMemRefType>(val.getType()).getMemorySpace())
          .getAddressSpace();
    };

    const auto cubeSpaces = {hivm::AddressSpace::L1, hivm::AddressSpace::L0A,
                             hivm::AddressSpace::L0B, hivm::AddressSpace::L0C};

    getOperation().walk([&](memref::CopyOp op) {
      const auto srcSpace = getAddressSpace(op.getSource());
      const auto dstSpace = getAddressSpace(op.getTarget());
      IRRewriter rewriter(op);

      if (srcSpace == hivm::AddressSpace::GM &&
          llvm::is_contained(cubeSpaces, dstSpace)) {
        rewriter.replaceOpWithNewOp<hivm::ND2NZOp>(
            op, TypeRange{}, op.getSource(), op.getTarget(),
            rewriter.getUnitAttr());
      } else if (llvm::is_contained(cubeSpaces, srcSpace) &&
                 dstSpace == hivm::AddressSpace::GM) {
        rewriter.replaceOpWithNewOp<hivm::FixpipeOp>(
            op, TypeRange{}, op.getSource(), op.getTarget(),
            rewriter.getUnitAttr());
      }
    });
  }
};
#undef DEBUG_TYPE

} // namespace mlir::tilelangir
