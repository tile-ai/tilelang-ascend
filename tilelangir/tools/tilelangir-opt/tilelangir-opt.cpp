// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/tools/tilelangir-opt/tilelangir-opt.cpp
 * \brief TileLangIR modular optimizer driver (mlir-opt style).
 *
 */
 #include "tilelangir/InitAllDialects.h"
 #include "tilelangir/InitAllPasses.h"
 
 #include "bishengir/InitAllDialects.h"
 #include "bishengir/InitAllExtensions.h"
 #include "bishengir/Dialect/HFusion/Transforms/Passes.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/Support/InitLLVM.h"

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  mlir::DialectRegistry registry;

  mlir::registerAllDialects(registry);
  bishengir::registerAllDialects(registry);
  ::tilelangir::registerAllDialects(registry);

  mlir::registerAllPasses();
  mlir::hfusion::registerHFusionPasses();
  ::tilelangir::registerAllPasses();

  mlir::registerAllExtensions(registry);
  bishengir::registerAllExtensions(registry);
  // TODO: Add TileLangIR extensions

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "TileLangIR modular optimizer Tool\n",
                        registry));
}

