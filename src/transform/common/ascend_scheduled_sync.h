// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#pragma once

#include "ascend_auto_schedule.h"

namespace tvm {
namespace tl {
// Shared IR facts only. Dependency/value and protocol validation are performed
// independently of the scheduler's proposed edges and solution.
AutoRecord InspectAscendAutoFinal(tir::PrimFunc function);
AutoRecord VerifyAscendAutoMapping(tir::PrimFunc function);
void VerifyAscendAutoSyncProtocol(tir::PrimFunc function);
} // namespace tl
} // namespace tvm
