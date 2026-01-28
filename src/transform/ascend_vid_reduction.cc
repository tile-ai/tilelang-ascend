// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file ascend_vid_reduction.cc
 * \brief host specialized for Ascend npu
 */

#include "arith/ir_mutator_with_analyzer.h"
#include "tir/analysis/var_use_def_analysis.h"
#include "tir/transforms/ir_utils.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include "../op/builtin.h"
#include "./common/collector.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

static constexpr const char *ascendVidReduction = "tl.ascend_vid_reduction";

TVM_REGISTER_PASS_CONFIG_OPTION(ascendVidReduction, Bool);

class AscendVidReduction : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f, PassContext ctx) {
    arith::Analyzer analyzer;
    AscendVidReduction substituter(&analyzer);
    return f;
  }

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;
};


tvm::transform::Pass AscendVidReduction() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto new_func = AscendVidReduction::Substitute(std::move(f), ctx);
    return new_func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendVidReduction", {});
}

// regist host path
TVM_REGISTER_GLOBAL("tl.transform.AscendVidReduction")
    .set_body_typed(AscendVidReduction);

}
}