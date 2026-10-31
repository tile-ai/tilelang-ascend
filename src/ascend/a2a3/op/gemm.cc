// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/op/gemm.cc
 *
 * Define gemm operator.
 */

#include "gemm.h"

#include "builtin.h"
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/transform.h>

#include "generic.h"

namespace tvm {
namespace tl {
namespace a2a3 {

using namespace tirx;
using namespace ffi;

Gemm::Gemm(Array<PrimExpr> args, BufferMap vmap) {
  ICHECK_GE(args.size(), 10U);
  ICHECK_LE(args.size(), 12U);
  Aptr = args[0];
  Bptr = args[1];
  Cptr = args[2];
  A = vmap[GetVarFromAccessPtr(Aptr)];
  B = vmap[GetVarFromAccessPtr(Bptr)];
  C = vmap[GetVarFromAccessPtr(Cptr)];
  trans_A = args[3].as<Bool>().value();
  trans_B = args[4].as<Bool>().value();
  M = args[5].as<IntImm>().value()->value;
  N = args[6].as<IntImm>().value()->value;
  K = args[7].as<IntImm>().value()->value;
  policy = static_cast<GemmWarpPolicy>(args[8].as<IntImm>().value()->value);
  clear_accum = args[9].as<Bool>().value();
  if (args.size() > 10) {
    kPack = args[10].as<IntImm>().value()->value;
    if (kPack != 1 && kPack != 2) {
      ICHECK(false) << "kPack must be 1 or 2";
    }
  }
  if (args.size() > 11) {
    wg_wait = args[11].as<IntImm>().value()->value;
  }
}

Stmt Gemm::Lower(const LowerArgs &, arith::Analyzer *) const {
  LOG(FATAL) << "TA generic gemm does not support A2/A3 Ascend; use the "
                "Ascend mma API";
  return {};
}

LayoutMap Gemm::InferLayout(const LayoutInferArgs &T, InferLevel) {
  ICHECK(C.scope() == "local.fragment");
  ICHECK(T.thread_bounds.defined());
  LOG(FATAL) << "Not supported " << T.target->str()
             << ": TA generic gemm has no Ascend layout implementation; "
                "use the Ascend mma API";
  return {};
}

} // namespace a2a3
} // namespace tl
} // namespace tvm
