// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

#pragma once

#include <tvm/target/target.h>
#include <tvm/tir/function.h>

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "upstream/ir_structure.h"

namespace tvm::tl::ascend_sync {

struct LegacyCompositeBufferEffect {
  // A/B/C order. Keep the original pointer and its element units; the physical
  // resolver conservatively covers its complete allocation, including padding.
  tir::Call access_ptr;
  bool read{false};
  bool write{false};
  // Set only by the verified static Cube pipeline audit. Actual pointers are
  // independently checked against their clock-selected slot by the resolver.
  Optional<tir::Var> logical_ring;
  int64_t required_bytes{0};
};

struct LegacyCompositeOperation {
  tir::Call call;
  CoreMask execution_owner{kCoreCube};
  ResourcePipe acquire{ResourcePipe::kMTE1};
  ResourcePipe completion{ResourcePipe::kCube};
  std::vector<LegacyCompositeBufferEffect> buffers;
  int64_t m{0};
  int64_t n{0};
  int64_t k{0};
  int64_t k_l0{0};
  Optional<PrimExpr> pipeline_clock;
};

struct LegacyCompositePlan {
  // Device helper protocols differ even though the typed TIR ABI is shared.
  // Keep the audited profile explicit; the adapter's acquire/completion API
  // and the upstream engine remain unchanged.
  enum class HelperProfile { kAscendC, kPtoA3 };
  HelperProfile helper_profile{HelperProfile::kAscendC};
  std::unordered_map<const tir::CallNode *, LegacyCompositeOperation>
      operations;
  // Reserved for the complete function, including helper entry/exit and both
  // ping-pong slots. Existing manual events must not use these IDs either.
  std::map<std::string, std::set<int>> reserved_intra_ids;
  // Internal analysis result, never a public attribute admission switch.
  bool static_cube_pipeline{false};
  Optional<tir::Var> cube_pipeline_loop;
  std::unordered_map<const tir::CallNode *, std::pair<tir::Var, int64_t>>
      cube_ring_copies;

  const LegacyCompositeOperation *Find(const tir::CallNode *call) const;
};

// Audit typed gemm_v0 effects and the actual backend helper protocol. Static
// Cube pipelines derive role/stage topology, logical epoch coverage and each
// operand's physical slot lifecycle from injector provenance. This does not
// synthesize another schedule or assume a particular lead, period or order
// count. General outer/dynamic control and mixed C/V remain separate protocols.
// PTO/A3 uses parameterized aligned FP16-to-FP32 geometry and a conservative
// MTE2 acquire / FIX completion envelope. AscendC retains its internal N/K
// ping-pong protocol. Hidden event pools are reserved for each real backend.
// Canonicalize a pure If/else choice between identical GEMM invocations
// before snapshot normalization creates independently optional controls.
tir::PrimFunc NormalizeLegacyCompositeBranches(const tir::PrimFunc &function);

LegacyCompositePlan AnalyzeLegacyComposites(const tir::PrimFunc &function,
                                            const Target &target,
                                            const String &platform);

} // namespace tvm::tl::ascend_sync
