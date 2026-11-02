// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

// Adapter boundary for the upstream synchronization engine. This layer imports
// effects and restores the original legacy TIR; it does not select sync edges.

#pragma once

#include <tvm/target/target.h>
#include <tvm/tir/function.h>

#include <memory>
#include <vector>

#include "physical_storage.h"
#include "upstream/ir_structure.h"
#include "upstream/multi_buffer.h"

namespace tvm::tl::ascend_sync {

struct LegacyExportState;
struct LegacyPipelineCase;
class LegacyPersistentPipelineImport;

struct LegacyLocalSyncGraph {
  CoreMask owner;
  std::vector<std::shared_ptr<IRStructure>> roots;
  // Analysis-only explicit/hidden reservations for this physical event pool.
  // Cross-core calls remain in the original AST, outside the local allocator.
  tir::Stmt reservations;
};

struct ImportedLegacySyncIR {
  tir::PrimFunc original_function;
  std::vector<std::shared_ptr<IRStructure>> roots;
  std::vector<LegacyLocalSyncGraph> local_graphs;
  MultiBufferPlan multibuffer;
  BufferVersionMap manual_buffer_versions;
  std::shared_ptr<PhysicalStorageResolver> physical_storage;
  std::shared_ptr<LegacyExportState> export_state;

  ConflictOracle MakeOracle() const;
};

// Import supports one execution resource and explicit scalar snapshot tasks.
// The initial physical-ring path additionally requires validated provenance,
// clocks and audited helper footprints. Unsupported effects or conditional
// protocols are diagnosed rather than omitted from dependency analysis.
ImportedLegacySyncIR
ImportLegacySync(const tir::PrimFunc &function, const Target &target,
                 const String &platform, bool ordinary_sync,
                 const LegacyPipelineCase *pipeline_case = nullptr,
                 const LegacyPersistentPipelineImport *persistent = nullptr);

// RunSyncEngine may add sync tasks, but must retain original node identities.
// Export restores the original For/If/Allocate/Let/Attr shells at exact
// anchors.
tir::PrimFunc ExportLegacySync(const ImportedLegacySyncIR &imported,
                               tir::Stmt *synchronized_scope = nullptr);

} // namespace tvm::tl::ascend_sync
