// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/op/tensorpulse.h
 * \brief Define TensorPulse-related operators.
 */

#ifndef TVM_TL_OP_TENSORPULSE_H_
#define TVM_TL_OP_TENSORPULSE_H_

#include "op.h"

namespace tvm {
namespace tl {

using namespace tir;

// V1.0 skeleton: a single elementwise op to validate the registration
// mechanism. Real op set (gemm_v0, set/wait_flag, pipe_barrier, sync_all, ...)
// is added incrementally as codegen lands.
TVM_DLL const Op &tensorpulse_add();

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_OP_TENSORPULSE_H_
