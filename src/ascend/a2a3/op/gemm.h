// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/op/gemm.h
 * \brief Define gemm operator.
 *
 */

#ifndef TVM_TL_ASCEND_A2A3_OP_GEMM_H_
#define TVM_TL_ASCEND_A2A3_OP_GEMM_H_

#include "generic.h"

namespace tvm {
namespace tl {
namespace a2a3 {

using namespace tirx;
using namespace ffi;

class Gemm : public Operator {
public:
  Gemm(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) final;
  static const Op &Get();
  enum class GemmWarpPolicy {
    kSquare = 0,
    kFullRow = 1,
    kFullCol = 2,
  } policy;

private:
  Array<PrimExpr> call_args;
  tirx::Buffer A, B, C;
  // pointer to the A, B, C
  PrimExpr Aptr, Bptr, Cptr;
  bool trans_A, trans_B;
  int M, N, K;
  bool clear_accum = false;
  // k_pack please ref to bitblas/tl/mfma_macro_generator.py::k_pack
  // only will be enabled under cdna mfma instructions
  int kPack = 1;
  int wg_wait = 0;
  bool completed_ = false;
};

} // namespace a2a3
} // namespace tl
} // namespace tvm

#endif //  TVM_TL_ASCEND_A2A3_OP_GEMM_H_
