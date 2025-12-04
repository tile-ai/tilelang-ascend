// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file ascend_combinecv.cc
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

static constexpr const char *ascendAutoCombine = "tl.ascend_auto_cv_combine";

TVM_REGISTER_PASS_CONFIG_OPTION(ascendAutoCombine, Bool);

class CVCombineEmitter : public StmtMutator {
public:
    CVCombineEmitter(bool is_aiv, Map<Var, String>& location)
        : is_aiv_(is_aiv), location_map_(location) {}

    std::string isSubstringInMap(const std::unordered_map<std::string, std::string>& m, const std::string& target) {
        if (target.empty()) {
            return std::string("");
        }
        size_t target_len = target.length();
        for (const auto& pair : m) {
            const std::string& key = pair.first;
            if (target.find(key) != std::string::npos) {
                return pair.second;
            }
        }
        return std::string("");
    }

    int32_t checkBufferScope(const Var &var) {
        int32_t check_ternaty = -1;
        if (is_aiv_) {
            if (location_map_.find(var) != location_map_.end()) {
                if (callnodeMapPos_[location_map_[var]] == "vec") {
                    check_ternaty = 1;
                } else if (callnodeMapPos_[location_map_[var]] == "cube") {
                    check_ternaty = 0;
                } else {
                    check_ternaty = -1;
                }
            }
        } else {
            if (location_map_.find(var) != location_map_.end()) {
                if (callnodeMapPos_[location_map_[var]] == "cube") {
                    check_ternaty = 1;
                } else if (callnodeMapPos_[location_map_[var]] == "vec") {
                    check_ternaty = 0;
                } else {
                    check_ternaty = -1;
                }
            }
        }
        return check_ternaty;
    }

    // some scene need
    // Stmt VisitStmt_(const ForNode *op) final {
    //     current_proccess_switch_ = false; // turn off
    //     return StmtMutator::VisitStmt_(op);
    // }

    Stmt VisitStmt_(const EvaluateNode *op) final {
        auto call_node_ = op->value.as<CallNode>();
        std::string api_name = "";
        if (call_node_ && call_node_->args[0].as<StringImmNode>()) {
            api_name = call_node_->args[0].as<StringImmNode>()->value;
        }
        auto found = isSubstringInMap(callnodeMapPos_, api_name);
        // judgement 1
        if (is_aiv_) {
            if (found == "vec") {
                current_proccess_switch_ = true; // turn on
                return StmtMutator::VisitStmt_(op);
            } else if (found == "cube") {
                current_proccess_switch_ = false; // turn off
            }
        } else {
            if (found == "cube") {
                current_proccess_switch_ = true; // turn on
                return StmtMutator::VisitStmt_(op);
            } else if (found == "vec") {
                current_proccess_switch_ = false; // turn off
            }
        }
        // judgement 2
        int32_t judge2 = -1;
        for (int i = 1; i < call_node_->args.size(); i++) {
            if (auto inter_node = call_node_->args[i].as<CallNode>()) {
                auto buf_name = Downcast<Var>(inter_node->args[1]);
                judge2 = checkBufferScope(buf_name);
                if (judge2 != -1) {
                    break;
                }
            }
        }
        if (judge2 == 1) {
            current_proccess_switch_ = true; // turn on
            return StmtMutator::VisitStmt_(op);
        } else if (judge2 == -1 && current_proccess_switch_) {
            return StmtMutator::VisitStmt_(op);
        }
        current_proccess_switch_ = false; // turn off
        return Evaluate(0);
    }



private:
    const bool is_aiv_;
    bool current_proccess_switch_ = false;
    Map<Var, String>& location_map_;
    std::unordered_map<std::string, std::string> callnodeMapPos_ = {
        {"copy_gm_to_l1", "cube"},
        {"gemm_v0", "cube"},
        {"copy_11_to_l0a", "cube"},
        {"copy_11_to_l0b", "cube"},
        {"copy_l0c_to_gm", "cube"},
        {"copy_gm_to_ub", "vec"},
        {"copy_ub_to_gm", "vec"},
        {"copy_ub_to_ub", "vec"},
        {"wmma.matrix_a", "cube"},
        {"wmma.matrix_b", "cube"},
        {"wmma.accumulator", "cube"},
        {"shared.dyn", "cube"},
        {"shared", "vec"}
    };
};

class CombineCV : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f, PassContext ctx) {
    arith::Analyzer analyzer;
    CombineCV substituter(&analyzer);
    PrimFuncNode *fptr = f.CopyOnWrite();
    tir::PostOrderVisit(f->body, [&](const ObjectRef& obj) {
        if (const auto* realize = obj.as<tir::BlockRealizeNode>()) {
            for (auto buf : realize->block->alloc_buffers) {
                String scope = GetPtrStorageScope(buf->data);
                substituter.location_map_.Set(buf->data, scope);
            }
        }
    });

    bool ascend_auto_combine = ctx->GetConfig<Bool>(ascendAutoCombine, Bool(false)).value();
    if (!ascend_auto_combine) {
      return f;
    }

    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  Stmt VisitStmt_(const BlockRealizeNode *op) override {
    if (op->block->name_hint == "tilelang_root") {
        Block block = op->block;

        CVCombineEmitter cubeStmt(false, location_map_);
        CVCombineEmitter vecStmt(true, location_map_);

        Stmt cube_code = cubeStmt(block->body);
        Stmt vec_code = vecStmt(block->body);

        Stmt cube_body = AttrStmt(make_zero(DataType::Int(32)), "resource_scope", 0, cube_code);
        Stmt vec_body = AttrStmt(make_zero(DataType::Int(32)), "resource_scope", 1, vec_code);
        Stmt combine_body = SeqStmt({cube_body, vec_body});
        block.CopyOnWrite()->body = combine_body;
        auto blockRealize = GetRef<BlockRealize>(op);
        blockRealize.CopyOnWrite()->block = block;
        return blockRealize;
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt_(op); 
  }


    Stmt VisitStmt_(const AllocateNode *op) override {
        return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    Map<Var, String> location_map_;
};


tvm::transform::Pass CombineCV() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto new_func = CombineCV::Substitute(std::move(f), ctx);
    return new_func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.CombineCV", {});
}

// regist host path
TVM_REGISTER_GLOBAL("tl.transform.CombineCV")
    .set_body_typed(CombineCV);

}
}