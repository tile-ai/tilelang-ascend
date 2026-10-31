// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

#pragma once

#include <tvm/tir/function.h>

#include <cstdint>
#include <vector>

#include "upstream/constraints.h"

namespace tvm::tl::ascend_sync {

struct LegacyPersistentLoopCertificate {
  tir::For loop;
  // Facts at the real loop header. The consumer must add the actual normalized
  // For range only while analyzing a body invocation.
  ConstrSet header_context;
  int64_t domain{0};
  int64_t wave{0};
};

struct LegacyPersistentNormalization {
  tir::PrimFunc function;
  std::vector<LegacyPersistentLoopCertificate> loops;
};

// C++-only analysis result. No public attribute grants a loop certificate.
LegacyPersistentNormalization
NormalizeLegacyPersistentWithCertificates(const tir::PrimFunc &function);

// Replace only a canonical, monotone Persistent leading-break loop with its
// exact per-worker trip count. All surviving payloads and lexical allocation
// shells stay inside the same For. Dynamic predicated Persistent loops are
// unchanged; arbitrary break/While control is not introduced into the engine.
tir::PrimFunc NormalizeLegacyPersistent(const tir::PrimFunc &function);

} // namespace tvm::tl::ascend_sync
