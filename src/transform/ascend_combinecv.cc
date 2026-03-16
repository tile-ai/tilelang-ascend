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
#include <tvm/tir/op.h>
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

static constexpr const char *ascendAutoCrossCoreSync =
    "tl.ascend_auto_cross_core_sync";

TVM_REGISTER_PASS_CONFIG_OPTION(ascendAutoCrossCoreSync, Bool);

static constexpr const int DEFAUT_MODEL_ID = 2;

struct CrossCoreSyncPoint {
  int scope;        // 0: cube, 1: vec
  int order;        // excute order
  int sync_flag_id; // cross core sync flag id
  bool is_write;    // whether write to workspace or not
  const std::string workspace_name;
  const std::string pipe; // MTE2, MTE3 or FIX
  const EvaluateNode *node;
  std::optional<const ForNode *> target_for_node = std::nullopt;
  // the ForNode to which the sync stmt will be attached. If not specified, will
  // be attached to the EvaluateNode.
  const std::vector<const ForNode *> parent_for_nodes;
  // the ForNode list (from outer to inner) before reaching the EvaluateNode

  std::string ToString() const {
    std::ostringstream oss;
    oss << "CrossCoreSyncPoint(";
    oss << "scope=" << scope;
    oss << ", order=" << order;
    oss << ", sync_flag_id=" << sync_flag_id;
    oss << ", is_write=" << is_write;
    oss << ", workspace_name=" << workspace_name;
    oss << ", pipe=" << pipe;
    if (target_for_node.has_value()) {
      oss << ", target_for_node="
          << target_for_node.value()->loop_var->name_hint;
    } else {
      oss << ", target_for_node=None";
    }
    oss << ", parent_for_nodes.size()=" << parent_for_nodes.size();
    oss << ")";
    return oss.str();
  }
};

class CrossCoreSyncCollector : public StmtVisitor {
public:
  CrossCoreSyncCollector(std::vector<CrossCoreSyncPoint> &sync_points,
                         const bool is_aiv)
      : sync_points_(sync_points), is_aiv_(is_aiv) {}

  const std::vector<CrossCoreSyncPoint> &GetSyncPoints() const {
    return sync_points_;
  }

  void VisitStmt_(const EvaluateNode *op) override {
    if (auto call_node = op->value.as<CallNode>()) {
      order_++;

      if (!call_node->op.same_as(builtin::call_extern())) {
        return;
      }

      std::string func_name = call_node->args[0].as<StringImmNode>()->value;

      if (auto cfg_info = GetGMCopyCfgInfo(func_name)) {
        bool is_write = cfg_info->first;
        std::string pipe = cfg_info->second;

        if (auto workspace_name_opt = FetchWorkspaceName(call_node)) {
          sync_points_.push_back(CrossCoreSyncPoint{
              is_aiv_ ? 1 : 0,
              order_,
              sync_flag_id_++,
              is_write,
              workspace_name_opt.value(),
              pipe,
              op,
              std::nullopt, // no target_for_node by default. Will be updated
                            // later
              current_loops_,
          });
        }
      }
    }
  }

  void VisitStmt_(const ForNode *op) override {
    current_loops_.push_back(op);
    StmtVisitor::VisitStmt_(op);
    current_loops_.pop_back();
  }

private:
  std::vector<CrossCoreSyncPoint> &sync_points_;
  const bool is_aiv_{false};
  int order_{0};
  int sync_flag_id_{0};
  std::vector<const ForNode *> current_loops_;

  /**
   * The configuration info table
   *
   * key: GM related function name
   * value: pair<isWrite, pipe>
   */
  const std::unordered_map<std::string, std::pair<bool, std::string>>
      GM_COPY_CFG_INFOS = {
          {"copy_gm_to_l1", {false, "MTE2"}},
          {"copy_l0c_to_gm", {true, "FIX"}},
          {"copy_gm_to_ub", {false, "MTE2"}},
          {"copy_ub_to_gm", {true, "MTE3"}},
      };

  std::optional<std::pair<bool, std::string>>
  GetGMCopyCfgInfo(const std::string &func_name) {
    for (const auto &item : GM_COPY_CFG_INFOS) {
      if (func_name.find(item.first) != std::string::npos) {
        return item.second;
      }
    }
    return std::nullopt;
  }

  /**
   * Fetch workspace from CallNode.
   */
  std::optional<std::string> FetchWorkspaceName(const CallNode *call_node) {
    auto args = call_node->args;
    for (int i = 1; i < args.size(); ++i) {
      if (auto inner_call_node = args[i].as<CallNode>()) {
        std::string buf_name =
            Downcast<Var>(inner_call_node->args[1])->name_hint;
        if (buf_name.find("workspace") != std::string::npos) {
          return buf_name;
        }
      }
    }
    return std::nullopt;
  }
};

class CrossCoreSyncInserter : public StmtMutator {
public:
  CrossCoreSyncInserter(std::vector<CrossCoreSyncPoint> &sync_points)
      : sync_points_(sync_points) {}

  Stmt VisitStmt_(const EvaluateNode *op) override {
    if (auto call_node = op->value.as<CallNode>()) {
      cur_order_++;
      for (const auto &sp : sync_points_) {
        // Match sync point
        if (sp.order != cur_order_) {
          continue;
        }

        // ForNode target, skip here; will be handled in ForNode visitor
        if (sp.target_for_node.has_value()) {
          continue;
        }

        // Insert wait/set flag
        return AttachSyncStmt(sp, GetRef<Stmt>(op));
      }
    }
    return GetRef<Stmt>(op);
  }

  Stmt VisitStmt_(const ForNode *op) override {
    Stmt new_body = this->VisitStmt(op->body);
    Stmt new_stmt = For(op->loop_var, op->min, op->extent, op->kind, new_body,
                        op->thread_binding, op->annotations);

    for (const auto &sp : sync_points_) {
      // Check ForNode
      if (!sp.target_for_node) {
        continue;
      }

      // Check ForNode match
      if (!op->body.same_as(sp.target_for_node.value()->body)) {
        continue;
      }

      // Insert sync stmt for the For loop
      new_stmt = AttachSyncStmt(sp, new_stmt);
    }

    return new_stmt;
  }

private:
  int cur_order_{0};
  const std::vector<CrossCoreSyncPoint> &sync_points_;

  /**
   * SetFlag After Write, WaitFlag Before Read.
   */
  Stmt AttachSyncStmt(const CrossCoreSyncPoint &sp, const Stmt &op_stmt) {
    if (sp.is_write) {
      return SeqStmt({op_stmt, GenAutoCrossCoreSetFlagStmt(sp)});
    } else {
      return SeqStmt({GenAutoCrossCoreWaitFlagStmt(sp), op_stmt});
    }
  }

  /**
   * Generate CrossCoreSetFlag
   */
  Stmt GenAutoCrossCoreSetFlagStmt(const CrossCoreSyncPoint &sp) {
    return Evaluate(Call(DataType::Handle(),
                         Op::Get("tl.ascend_auto_set_cross_flag"),
                         {
                             Integer(DEFAUT_MODEL_ID),
                             StringImm(sp.pipe),
                             Integer(sp.sync_flag_id),
                         }));
  }

  /**
   * Generate CrossCoreWaitFlag
   */
  Stmt GenAutoCrossCoreWaitFlagStmt(const CrossCoreSyncPoint &sp) {
    return Evaluate(Call(DataType::Handle(),
                         Op::Get("tl.ascend_auto_wait_cross_flag"),
                         {
                             Integer(sp.sync_flag_id),
                             StringImm(""),
                         }));
  }
};

class AutoInsertCrossCoreSync {
public:
  static void AutoInsert(Stmt &cube_code, Stmt &vec_code) {
    // Collect sync points
    std::vector<CrossCoreSyncPoint> cube_sync_points;
    std::vector<CrossCoreSyncPoint> vec_sync_points;

    CrossCoreSyncCollector cube_collector(cube_sync_points, false);
    CrossCoreSyncCollector vec_collector(vec_sync_points, true);

    cube_collector(cube_code);
    vec_collector(vec_code);

    // Check sync points consistency
    if (cube_sync_points.size() != vec_sync_points.size()) {
      LOG(FATAL) << "Mismatch in sync points between cube and vec: "
                 << "cube has " << cube_sync_points.size() << ", "
                 << "vec has " << vec_sync_points.size();
    }

    for (size_t i = 0; i < cube_sync_points.size(); ++i) {
      const auto &cube_sp = cube_sync_points[i];
      const auto &vec_sp = vec_sync_points[i];
      if (cube_sp.is_write == vec_sp.is_write) {
        LOG(FATAL) << "Inconsistent read/write operations at sync point " << i
                   << ": "
                   << "cube is_write=" << cube_sp.is_write << ", "
                   << "vec is_write=" << vec_sp.is_write;
      }
      if (cube_sp.workspace_name != vec_sp.workspace_name) {
        LOG(FATAL) << "Inconsistent workspace names at sync point " << i << ": "
                   << "cube workspace=" << cube_sp.workspace_name << ", "
                   << "vec workspace=" << vec_sp.workspace_name;
      }
    }

    // find CrossCoreSyncPoint.target_for_node at here
    PairSyncPoints(cube_sync_points, vec_sync_points);

    // Insert sync statements
    CrossCoreSyncInserter cube_sync_inserter(cube_sync_points);
    CrossCoreSyncInserter vec_sync_inserter(vec_sync_points);

    cube_code = cube_sync_inserter(cube_code);
    vec_code = vec_sync_inserter(vec_code);
  }

private:
  static int64_t GetLoopIterTimes(const ForNode *for_node) {
    const int64_t *extent_ptr = as_const_int(for_node->extent);
    ICHECK(extent_ptr) << "AutoInsertCrossCoreSync::GetLoopIterTimes only "
                          "works with constant loop sizes, but got "
                       << for_node->extent;
    return *extent_ptr;
  }

  static void PairSyncPoints(std::vector<CrossCoreSyncPoint> &cube_sync_points,
                             std::vector<CrossCoreSyncPoint> &vec_sync_points) {
    for (auto &cube_sp : cube_sync_points) {
      for (auto &vec_sp : vec_sync_points) {
        if (cube_sp.sync_flag_id != vec_sp.sync_flag_id) {
          continue;
        }
        FindTargetLoopDepth(cube_sp, vec_sp);
      }
    }
  }

  // find target ForNodes to attach sync stmts
  static void FindTargetLoopDepth(CrossCoreSyncPoint &cube_sp,
                                  CrossCoreSyncPoint &vec_sp) {
    if (cube_sp.parent_for_nodes.empty() && vec_sp.parent_for_nodes.empty()) {
      return; // sync point pairs aren't in any loop
    }

    // total loop times of sync points
    int64_t cube_loop_times = 1;
    int64_t vec_loop_times = 1;
    // current index of CrossCoreSyncPoint.parent_for_nodes
    int cube_loop_idx = 0;
    int vec_loop_idx = 0;
    // current max loop depth when cube_loop_times == vec_loop_times
    int cube_max_pair_depth = 0;
    int vec_max_pair_depth = 0;
    // handle corner case: vec has loops with 1 iter and can't catch up cube
    // loop times
    int last_pair_loop_times = 1;

    // iterate through both cube_sp.parent_for_nodes and vec_sp.parent_for_nodes
    // once
    while (cube_loop_idx < cube_sp.parent_for_nodes.size() ||
           vec_loop_idx < vec_sp.parent_for_nodes.size()) {
      while (
          cube_loop_idx < cube_sp.parent_for_nodes.size() &&
          (cube_loop_times <= vec_loop_times ||
           GetLoopIterTimes(cube_sp.parent_for_nodes.at(cube_loop_idx)) == 1)) {
        if (cube_loop_times == vec_loop_times) {
          cube_max_pair_depth = cube_loop_idx;
          vec_max_pair_depth = vec_loop_idx;
          last_pair_loop_times = cube_loop_times;
        }

        cube_loop_times *=
            GetLoopIterTimes(cube_sp.parent_for_nodes.at(cube_loop_idx));
        cube_loop_idx++;
      }

      if (cube_loop_times < vec_loop_times) {
        LOG(WARNING) << "Cube loop times (= " << cube_loop_times
                     << " ) is not enough to catch up vec loop times (= "
                     << vec_loop_times << " )" << std::endl
                     << "Cube Sync Point:" << std::endl
                     << cube_sp.ToString() << std::endl
                     << "Vec Sync Point:" << std::endl
                     << vec_sp.ToString() << std::endl;
      }

      while (
          vec_loop_idx < vec_sp.parent_for_nodes.size() &&
          (vec_loop_times <= cube_loop_times ||
           GetLoopIterTimes(vec_sp.parent_for_nodes.at(vec_loop_idx)) == 1)) {
        if (cube_loop_times == vec_loop_times) {
          cube_max_pair_depth = cube_loop_idx;
          vec_max_pair_depth = vec_loop_idx;
          last_pair_loop_times = cube_loop_times;
        }

        vec_loop_times *=
            GetLoopIterTimes(vec_sp.parent_for_nodes.at(vec_loop_idx));
        vec_loop_idx++;

        if (vec_loop_times == last_pair_loop_times) {
          // cube_loop_times steps beyond last_pair_loop_times && vec_loop_times
          // doesn't increase ( *= 1 )
          vec_max_pair_depth = vec_loop_idx;
        }
      }

      if (vec_loop_times < cube_loop_times) {
        LOG(WARNING) << "Vec loop times (= " << vec_loop_times
                     << " ) is not enough to catch up cube loop times (= "
                     << cube_loop_times << " )" << std::endl
                     << "Vec Sync Point:" << std::endl
                     << vec_sp.ToString() << std::endl
                     << "Cube Sync Point:" << std::endl
                     << cube_sp.ToString() << std::endl;
      }
    }

    if (cube_loop_times == vec_loop_times) {
      // in case the loop instantly ends after vec_loop_idx step to next loop
      cube_max_pair_depth = cube_loop_idx;
      vec_max_pair_depth = vec_loop_idx;
    }

    // target_for_node is the for loop at max_pair_depth (if it has a for loop
    // at that depth)
    if (0 <= cube_max_pair_depth &&
        cube_max_pair_depth < cube_sp.parent_for_nodes.size()) {
      cube_sp.target_for_node =
          cube_sp.parent_for_nodes.at(cube_max_pair_depth);
    }

    if (0 <= vec_max_pair_depth &&
        vec_max_pair_depth < vec_sp.parent_for_nodes.size()) {
      vec_sp.target_for_node = vec_sp.parent_for_nodes.at(vec_max_pair_depth);
    }

    // otherwise, target_for_node remains nullopt
  }
};

class CVCombineEmitter : public StmtMutator {
public:
  CVCombineEmitter(bool is_aiv, Map<Var, String> &location)
      : is_aiv_(is_aiv), location_map_(location) {}

  std::string
  isSubstringInMap(const std::unordered_map<std::string, std::string> &m,
                   const std::string &target) {
    if (target.empty()) {
      return std::string("");
    }
    size_t target_len = target.length();
    for (const auto &pair : m) {
      const std::string &key = pair.first;
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
    if (!call_node_) {
      return StmtMutator::VisitStmt_(op);
    }
    std::string api_name = "";
    if (call_node_) {
      if (const auto *str_imm = call_node_->args[0].as<StringImmNode>()) {
        api_name = str_imm->value;
      }
      if (const auto *op_node = call_node_->op.as<OpNode>();
          op_node && IsRetainedInBothScopes(op_node->name)) {
        return GetRef<Stmt>(op);
      }
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
    for (int i = 0; i < call_node_->args.size(); i++) {
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

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto buf_scope = op->buffer.scope();
    if (is_aiv_) {
      if (buf_scope == "shared") {
        return StmtMutator::VisitStmt_(op);
      } else {
        return Evaluate(0);
      }
    } else {
      if (buf_scope == "shared") {
        return Evaluate(0);
      } else {
        return StmtMutator::VisitStmt_(op);
      }
    }
  }

  bool IsRetainedInBothScopes(const std::string &api_name) {
    // APIs that do not belong to cube or vec scope,
    // and should be retained in both generated code paths (e.g. printf).
    static const std::vector<std::string> kBothScopesApis = {
        "tl.ascend_printf",
    };
    for (const auto &target_api : kBothScopesApis) {
      if (api_name.find(target_api) != std::string::npos) {
        return true;
      }
    }
    return false;
  }

private:
  const bool is_aiv_;
  bool current_proccess_switch_ = false;
  Map<Var, String> &location_map_;
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
      {"shared", "vec"}};
};

class CombineCV : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f, PassContext ctx) {
    arith::Analyzer analyzer;
    CombineCV substituter(&analyzer);
    PrimFuncNode *fptr = f.CopyOnWrite();
    tir::PostOrderVisit(f->body, [&](const ObjectRef &obj) {
      if (const auto *realize = obj.as<tir::BlockRealizeNode>()) {
        for (auto buf : realize->block->alloc_buffers) {
          String scope = GetPtrStorageScope(buf->data);
          substituter.location_map_.Set(buf->data, scope);
        }
      }
    });

    bool ascend_auto_combine =
        ctx->GetConfig<Bool>(ascendAutoCombine, Bool(false)).value();
    if (!ascend_auto_combine) {
      return f;
    }

    substituter.is_auto_cross_core_sync_ =
        ctx->GetConfig<Bool>(ascendAutoCrossCoreSync, Bool(false)).value();

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

      if (is_auto_cross_core_sync_) {
        AutoInsertCrossCoreSync::AutoInsert(cube_code, vec_code);
      }

      Stmt cube_body = AttrStmt(make_zero(DataType::Int(32)), "resource_scope",
                                0, cube_code);
      Stmt vec_body =
          AttrStmt(make_zero(DataType::Int(32)), "resource_scope", 1, vec_code);
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
  bool is_auto_cross_core_sync_{false};
};

tvm::transform::Pass CombineCV() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto new_func = CombineCV::Substitute(std::move(f), ctx);
    return new_func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.CombineCV", {});
}

// regist host path
TVM_REGISTER_GLOBAL("tl.transform.CombineCV").set_body_typed(CombineCV);

} // namespace tl
} // namespace tvm