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

#include <algorithm>
#include <deque>
#include <map>

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
  int order;        // execute order
  int sync_flag_id; // cross core sync flag id
  bool is_write;    // whether write to workspace or not
  std::string workspace_name;
  std::string pipe; // MTE2, MTE3 or FIX
  const EvaluateNode *node;
  std::optional<const ForNode *> target_for_node = std::nullopt;
  // the ForNode to which the sync stmt will be attached. If not specified, will
  // be attached to the EvaluateNode.
  std::vector<const ForNode *> parent_for_nodes;
  // the ForNode list (from outer to inner) before reaching the EvaluateNode

  // Cross interval support
  int cross_interval = 1;
  const ForNode *stage_loop = nullptr;

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
    oss << ", cross_interval=" << cross_interval;
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
          CrossCoreSyncPoint sp;
          sp.scope = is_aiv_ ? 1 : 0;
          sp.order = order_;
          sp.sync_flag_id = sync_flag_id_++;
          sp.is_write = is_write;
          sp.workspace_name = workspace_name_opt.value();
          sp.pipe = pipe;
          sp.node = op;
          sp.target_for_node = std::nullopt;
          sp.parent_for_nodes = current_loops_;
          sp.cross_interval = GetCrossInterval();
          sp.stage_loop = current_stage_loop_;
          sync_points_.push_back(sp);
        }
      }
    }
  }

  void VisitStmt_(const ForNode *op) override {
    bool is_stage_loop = op->annotations.Get("stage_loop").defined();

    if (is_stage_loop) {
      current_stage_loop_ = op;
    }

    current_loops_.push_back(op);
    StmtVisitor::VisitStmt_(op);
    current_loops_.pop_back();

    if (is_stage_loop) {
      current_stage_loop_ = nullptr;
    }
  }

  int GetCrossInterval() const {
    if (current_stage_loop_) {
      auto interval_anno =
          current_stage_loop_->annotations.Get("tl_cross_interval");
      if (interval_anno.defined()) {
        return interval_anno.as<IntImmNode>()->value;
      }
    }
    return 1;
  }

private:
  std::vector<CrossCoreSyncPoint> &sync_points_;
  const bool is_aiv_{false};
  int order_{0};
  int sync_flag_id_{0};
  std::vector<const ForNode *> current_loops_;
  const ForNode *current_stage_loop_ = nullptr;

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
          {"atomic_add_ub_to_gm", {true, "MTE3"}},
          {"atomic_add_l0c_to_gm", {true, "FIX"}},
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
   * Note: Only sync_stmt is conditional, op_stmt (data copy) always executes.
   */
  Stmt AttachSyncStmt(const CrossCoreSyncPoint &sp, const Stmt &op_stmt) {
    Stmt sync_stmt;
    if (sp.is_write) {
      sync_stmt = GenAutoCrossCoreSetFlagStmt(sp);
    } else {
      sync_stmt = GenAutoCrossCoreWaitFlagStmt(sp);
    }

    if (sp.cross_interval > 1 && sp.stage_loop != nullptr) {
      PrimExpr condition = GenSyncCondition(sp);
      // op_stmt always executes, sync_stmt is conditional
      if (sp.is_write) {
        // writer: op_stmt first, then conditional sync
        return SeqStmt(
            {op_stmt, IfThenElse(condition, sync_stmt, Evaluate(0))});
      } else {
        // reader: conditional sync first, then op_stmt
        return SeqStmt(
            {IfThenElse(condition, sync_stmt, Evaluate(0)), op_stmt});
      }
    }

    if (sp.is_write) {
      return SeqStmt({op_stmt, sync_stmt});
    } else {
      return SeqStmt({sync_stmt, op_stmt});
    }
  }

  /**
   * Generate sync condition based on cross_interval.
   * Writer (set): (stage_var % cross_interval == cross_interval - 1) ||
   * is_last_iteration Reader (wait): stage_var % cross_interval == 0
   */
  PrimExpr GenSyncCondition(const CrossCoreSyncPoint &sp) {
    const ForNode *stage_loop = sp.stage_loop;
    if (stage_loop == nullptr) {
      return make_const(DataType::Bool(), true);
    }
    PrimExpr stage_var = stage_loop->loop_var;
    PrimExpr stage_extent = stage_loop->extent;
    int cross_interval = sp.cross_interval;
    auto int32 = DataType::Int(32);

    if (sp.is_write) {
      PrimExpr mod_cond = EQ(Mod(stage_var, make_const(int32, cross_interval)),
                             make_const(int32, cross_interval - 1));
      PrimExpr last_iter_cond =
          EQ(stage_var, Sub(stage_extent, make_const(int32, 1)));
      return tir::Or(mod_cond, last_iter_cond);
    } else {
      return EQ(Mod(stage_var, make_const(int32, cross_interval)),
                make_const(int32, 0));
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
                             StringImm(sp.pipe),
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

    // Map to group sync points by workspace_name
    std::map<std::string, std::vector<CrossCoreSyncPoint *>> cube_ws_map;
    std::map<std::string, std::vector<CrossCoreSyncPoint *>> vec_ws_map;

    for (auto &sp : cube_sync_points) {
      cube_ws_map[sp.workspace_name].push_back(&sp);
    }
    for (auto &sp : vec_sync_points) {
      vec_ws_map[sp.workspace_name].push_back(&sp);
    }

    int global_sync_flag_id = 0;

    for (auto &[ws, cube_sps] : cube_ws_map) {
      auto &vec_sps = vec_ws_map[ws];

      // Check sync points consistency per workspace
      if (cube_sps.size() != vec_sps.size()) {
        LOG(FATAL) << "Mismatch in sync points between cube and vec for "
                      "workspace "
                   << ws << ": " << "cube has " << cube_sps.size() << ", "
                   << "vec has " << vec_sps.size();
      }

      for (size_t i = 0; i < cube_sps.size(); ++i) {
        auto *cube_sp = cube_sps[i];
        auto *vec_sp = vec_sps[i];

        if (cube_sp->is_write == vec_sp->is_write) {
          LOG(FATAL) << "Inconsistent read/write operations for workspace "
                     << ws << " at sync point " << i
                     << ": cube is_write=" << cube_sp->is_write << ", "
                     << "vec is_write=" << vec_sp->is_write;
        }

        // Assign a common sync_flag_id for matched pair
        cube_sp->sync_flag_id = global_sync_flag_id;
        vec_sp->sync_flag_id = global_sync_flag_id;
        global_sync_flag_id++;

        // find target_for_node using iterative depth search
        FindTargetLoopDepth(*cube_sp, *vec_sp);
      }
    }

    // Insert sync statements
    CrossCoreSyncInserter cube_sync_inserter(cube_sync_points);
    CrossCoreSyncInserter vec_sync_inserter(vec_sync_points);

    cube_code = cube_sync_inserter(cube_code);
    vec_code = vec_sync_inserter(vec_code);
  }

private:
  // return loop iter times as const int64_t* or nullptr
  static const int64_t *IterTimesAsConst(const ForNode *for_node) {
    return as_const_int(for_node->extent);
  }

  static int64_t GetLoopIterTimes(const ForNode *for_node) {
    const int64_t *extent_ptr = IterTimesAsConst(for_node);
    ICHECK(extent_ptr) << "AutoInsertCrossCoreSync::GetLoopIterTimes only "
                          "works with constant loop sizes, but got "
                       << for_node->extent;
    return *extent_ptr;
  }

  // get loop iter times but skip loop whose id in skip_loop_ids
  static int64_t GetLoopIterTimesWithSkip(
      const ForNode *for_node,
      const std::unordered_set<std::string> &skip_loop_ids) {
    if (skip_loop_ids.find(for_node->loop_var->name_hint) !=
        skip_loop_ids.end()) {
      return 1; // skip this loop by treating it as 1 iter
    }
    return GetLoopIterTimes(for_node);
  }

  // check if same depth & same name in both parent_for_nodes
  static bool
  IsSharedLoop(int loop_index,
               const std::vector<const ForNode *> &cube_parent_for_nodes,
               const std::vector<const ForNode *> &vec_parent_for_nodes) {
    if (loop_index >= cube_parent_for_nodes.size() ||
        loop_index >= vec_parent_for_nodes.size()) {
      return false;
    }
    return cube_parent_for_nodes[loop_index]->loop_var->name_hint ==
           vec_parent_for_nodes[loop_index]->loop_var->name_hint;
  }

  // collect ids of shared loops with non-constant iter times
  static std::unordered_set<std::string> CollectNonConstSharedLoopIds(
      const std::vector<const ForNode *> &cube_parent_for_nodes,
      const std::vector<const ForNode *> &vec_parent_for_nodes) {
    std::unordered_set<std::string> non_const_shared_loop_ids;
    int min_size =
        std::min(cube_parent_for_nodes.size(), vec_parent_for_nodes.size());
    for (int i = 0; i < min_size; ++i) {
      const auto *cube_loop = cube_parent_for_nodes[i];
      // is non-const loop and is shared by cube and vec
      if (IterTimesAsConst(cube_loop) == nullptr &&
          IsSharedLoop(i, cube_parent_for_nodes, vec_parent_for_nodes)) {
        non_const_shared_loop_ids.insert(cube_loop->loop_var->name_hint);
      }
    }
    return non_const_shared_loop_ids;
  }

  // find target ForNodes to attach sync stmts
  static void FindTargetLoopDepth(CrossCoreSyncPoint &cube_sp,
                                  CrossCoreSyncPoint &vec_sp) {
    if (cube_sp.parent_for_nodes.empty() && vec_sp.parent_for_nodes.empty()) {
      return; // sync point pairs aren't in any loop
    }

    auto skip_loop_ids = CollectNonConstSharedLoopIds(cube_sp.parent_for_nodes,
                                                      vec_sp.parent_for_nodes);

    if (!skip_loop_ids.empty()) {
      // log skip_loop_ids
      std::string loop_ids;
      for (const auto &_id : skip_loop_ids) {
        if (!loop_ids.empty())
          loop_ids += ", ";
        loop_ids += _id;
      }
      DLOG(DEBUG)
          << "Found " << skip_loop_ids.size()
          << " shared loop(s) with non-constant iter times: [" << loop_ids
          << "]. These loop(s) won't be counted for total loop times of \""
          << cube_sp.workspace_name << "\"'s sync points.\n";
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
    int64_t last_pair_loop_times = 1;

    // iterate through both cube_sp.parent_for_nodes and vec_sp.parent_for_nodes
    // once
    while (cube_loop_idx < cube_sp.parent_for_nodes.size() ||
           vec_loop_idx < vec_sp.parent_for_nodes.size()) {
      bool cube_idx_updated = false;
      while (
          cube_loop_idx < cube_sp.parent_for_nodes.size() &&
          (cube_loop_times <= vec_loop_times ||
           GetLoopIterTimesWithSkip(cube_sp.parent_for_nodes.at(cube_loop_idx),
                                    skip_loop_ids) == 1)) {
        if (cube_loop_times == vec_loop_times) {
          cube_max_pair_depth = cube_loop_idx;
          vec_max_pair_depth = vec_loop_idx;
          last_pair_loop_times = cube_loop_times;
        }

        const ForNode *cube_loop = cube_sp.parent_for_nodes.at(cube_loop_idx);
        cube_loop_times *= GetLoopIterTimesWithSkip(cube_loop, skip_loop_ids);
        cube_loop_idx++;
        cube_idx_updated = true;
      }

      if (cube_loop_times < vec_loop_times) {
        LOG(WARNING) << "Cube loop times (= " << cube_loop_times
                     << " ) is not enough to catch up vec loop times (= "
                     << vec_loop_times << " )\n"
                     << "Cube Sync Point:\n"
                     << cube_sp.ToString() << "\n"
                     << "Vec Sync Point:\n"
                     << vec_sp.ToString() << "\n";
      }

      bool vec_idx_updated = false;
      while (vec_loop_idx < vec_sp.parent_for_nodes.size() &&
             (vec_loop_times <= cube_loop_times ||
              GetLoopIterTimesWithSkip(vec_sp.parent_for_nodes.at(vec_loop_idx),
                                       skip_loop_ids) == 1)) {
        if (cube_loop_times == vec_loop_times) {
          cube_max_pair_depth = cube_loop_idx;
          vec_max_pair_depth = vec_loop_idx;
          last_pair_loop_times = cube_loop_times;
        }

        const ForNode *vec_loop = vec_sp.parent_for_nodes.at(vec_loop_idx);
        vec_loop_times *= GetLoopIterTimesWithSkip(vec_loop, skip_loop_ids);
        vec_loop_idx++;
        vec_idx_updated = true;

        if (vec_loop_times == last_pair_loop_times) {
          // cube_loop_times steps beyond last_pair_loop_times && vec_loop_times
          // doesn't increase ( *= 1 )
          vec_max_pair_depth = vec_loop_idx;
        }
      }

      if (vec_loop_times < cube_loop_times) {
        LOG(WARNING) << "Vec loop times (= " << vec_loop_times
                     << " ) is not enough to catch up cube loop times (= "
                     << cube_loop_times << " )\n"
                     << "Vec Sync Point:\n"
                     << vec_sp.ToString() << "\n"
                     << "Cube Sync Point:\n"
                     << cube_sp.ToString() << "\n";
      }

      if (!(cube_idx_updated || vec_idx_updated)) {
        break;
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
  Stmt VisitStmt_(const ForNode *op) final {
    Stmt new_stmt = StmtMutator::VisitStmt_(op);

    const ForNode *new_for = new_stmt.as<ForNode>();
    if (!new_for) {
      return new_stmt;
    }

    Stmt new_body = new_for->body;
    // Recursively check if the body is effectively empty
    // (e.g., BlockRealize with only alloc_buffers and Evaluate(0))
    if (IsEmptyBody(new_body)) {
      return Evaluate(0);
    }

    return new_stmt;
  }

  bool IsEmptyBody(const Stmt &stmt) {
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      if (const auto *int_imm = eval->value.as<IntImmNode>()) {
        return int_imm->value == 0;
      }
    }
    if (const auto *alloc = stmt.as<AllocateNode>()) {
      return IsEmptyBody(alloc->body);
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      // Check if block only has allocations and no actual statements
      return IsEmptyBody(realize->block->body);
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      // Block may have alloc_buffers, but we only care about the body
      return IsEmptyBody(block->body);
    }
    if (const auto *if_then_else = stmt.as<IfThenElseNode>()) {
      bool then_empty = IsEmptyBody(if_then_else->then_case);
      bool else_empty = if_then_else->else_case.defined()
                            ? IsEmptyBody(if_then_else->else_case.value())
                            : true;
      return then_empty && else_empty;
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const auto &s : seq->seq) {
        if (!IsEmptyBody(s)) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

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
        "tl.ascend_sync_all",
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
      {"copy_l1_to_l0a", "cube"},
      {"copy_l1_to_l0b", "cube"},
      {"copy_l0c_to_gm", "cube"},
      {"copy_gm_to_ub", "vec"},
      {"copy_ub_to_gm", "vec"},
      {"atomic_add_ub_to_gm", "vec"},
      {"atomic_add_l0c_to_gm", "cube"},
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
