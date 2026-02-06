// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/include/tilelangir/InitAllPasses.h
 * \brief TileLangIR pass registration (registerAllPasses / registerTileLangIRPasses).
 *
 */

#ifndef TILELANGIR_INITALLPASSES_H
#define TILELANGIR_INITALLPASSES_H

#include "tilelangir/Transforms/Passes.h"

namespace mlir {
class OpPassManager;
}

namespace tilelangir {

/// Register all TileLangIR passes with the global registry.
inline void registerAllPasses() {
  mlir::tilelangir::registerTileLangIRPasses();
}

/// Alias for registerAllPasses().
inline void registerTileLangIRPasses() {
  registerAllPasses();
}

void buildTileLangIRCompilePipeline(mlir::OpPassManager &pm);

} // namespace tilelangir

#endif // TILELANGIR_INITALLPASSES_H
