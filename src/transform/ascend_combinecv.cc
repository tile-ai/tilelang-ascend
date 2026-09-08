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

#include "../op/ascend.h"
#include "../op/builtin.h"
#include "./common/ascend_vector_mask.h"
#include "./common/collector.h"
#include "./common/operation_config.h"

#include <algorithm>
#include <cctype>
#include <deque>
#include <map>
#include <vector>

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

namespace {

enum class AscendResource {
  kNone,
  kCommon,
  kExplicit,
  kCube,
  kVector,
};

AscendResource ResourceForStorageScope(const std::string &scope) {
  if (scope == "shared.ub") {
    return AscendResource::kVector;
  }
  if (scope == "shared.l1" || scope == "wmma.matrix_a" ||
      scope == "wmma.matrix_b" || scope == "wmma.accumulator") {
    return AscendResource::kCube;
  }
  return AscendResource::kNone;
}

std::string NormalizePipeName(std::string pipe) {
  std::transform(pipe.begin(), pipe.end(), pipe.begin(),
                 [](unsigned char ch) { return std::toupper(ch); });
  return pipe;
}

AscendResource ResourceForPipe(const std::string &pipe) {
  std::string normalized = NormalizePipeName(pipe);
  if (normalized == "V") {
    return AscendResource::kVector;
  }
  if (normalized == "M" || normalized == "MTE1" || normalized == "FIX") {
    return AscendResource::kCube;
  }
  return AscendResource::kExplicit;
}

AscendResource ResourceForPipeArgument(const CallNode *call, size_t index) {
  if (call->args.size() <= index) {
    return AscendResource::kExplicit;
  }
  const auto *pipe = call->args[index].as<StringImmNode>();
  return pipe == nullptr ? AscendResource::kExplicit
                         : ResourceForPipe(pipe->value);
}

AscendResource MergeResources(AscendResource lhs, AscendResource rhs,
                              const std::string &operation) {
  if (rhs == AscendResource::kNone || rhs == AscendResource::kCommon) {
    return lhs;
  }
  if (lhs == AscendResource::kNone || lhs == AscendResource::kCommon) {
    return rhs;
  }
  ICHECK(lhs == rhs || lhs == AscendResource::kExplicit ||
         rhs == AscendResource::kExplicit)
      << "Ascend operation mixes Cube and Vector local buffers: " << operation;
  return lhs == AscendResource::kExplicit ? rhs : lhs;
}

AscendResource ResourceForAccessPtr(const PrimExpr &expr) {
  const auto *access = expr.as<CallNode>();
  if (access == nullptr || !access->op.same_as(builtin::tvm_access_ptr()) ||
      access->args.size() < 2) {
    return AscendResource::kNone;
  }
  const auto *var = access->args[1].as<VarNode>();
  if (var == nullptr) {
    return AscendResource::kNone;
  }
  return ResourceForStorageScope(GetPtrStorageScope(GetRef<Var>(var)));
}

AscendResource ResourceForAccessPtrs(const Array<PrimExpr> &args, size_t begin,
                                     const std::string &operation) {
  AscendResource result = AscendResource::kNone;
  for (size_t i = begin; i < args.size(); ++i) {
    result = MergeResources(result, ResourceForAccessPtr(args[i]), operation);
  }
  return result;
}

std::string NormalizeFunctionName(std::string name) {
  size_t template_pos = name.find('<');
  if (template_pos != std::string::npos) {
    name.resize(template_pos);
  }
  size_t namespace_pos = name.rfind("tl::ascend::");
  if (namespace_pos != std::string::npos) {
    name = name.substr(namespace_pos + 12);
  }
  return name;
}

AscendResource ResourceForConfiguredOperation(const std::string &name,
                                              const OperationConfig &config,
                                              const Array<PrimExpr> &args,
                                              size_t begin) {
  AscendResource operands = ResourceForAccessPtrs(args, begin, name);
  if (config.default_pipeline == "PIPE_V") {
    return MergeResources(AscendResource::kVector, operands, name);
  }
  if (config.default_pipeline == "PIPE_M" ||
      config.default_pipeline == "PIPE_MTE1" ||
      config.default_pipeline == "PIPE_FIX") {
    return MergeResources(AscendResource::kCube, operands, name);
  }
  return operands == AscendResource::kNone ? AscendResource::kExplicit
                                           : operands;
}

AscendResource ResourceForPipePair(const std::string &src,
                                   const std::string &dst,
                                   const std::string &operation) {
  AscendResource src_resource = ResourceForPipe(src);
  AscendResource dst_resource = ResourceForPipe(dst);
  return MergeResources(src_resource, dst_resource, operation);
}

AscendResource ResourceForEventType(const std::string &event_type,
                                    const std::string &operation) {
  size_t separator = event_type.find('_');
  if (separator == std::string::npos) {
    return AscendResource::kExplicit;
  }
  return ResourceForPipePair(event_type.substr(0, separator),
                             event_type.substr(separator + 1), operation);
}

AscendResource ResourceForCall(const CallNode *call, std::string *operation) {
  Call call_ref = GetRef<Call>(call);
  const auto *op = call->op.as<OpNode>();
  *operation = op == nullptr ? "Ascend call" : op->name;

  if (call->op.same_as(ascend_printf()) ||
      call->op.same_as(ascend_sync_all()) ||
      call->op.same_as(ascend_use_swizzle())) {
    return AscendResource::kCommon;
  }
  if (call->op.same_as(ascend_src_code())) {
    return AscendResource::kExplicit;
  }
  if (call->op.same_as(ascend_free_pipe())) {
    if (call->args.empty()) {
      return AscendResource::kExplicit;
    }
    const auto *name = call->args[0].as<StringImmNode>();
    if (name != nullptr && name->value == "free_pipe_C") {
      return AscendResource::kCube;
    }
    if (name != nullptr && name->value == "free_pipe_V") {
      return AscendResource::kVector;
    }
    return AscendResource::kExplicit;
  }
  if (call->op.same_as(ascend_set_deq_scale()) ||
      IsVectorMaskSetter(call_ref) || IsSelectedVectorTerminal(call_ref)) {
    return AscendResource::kVector;
  }
  if (call->op.same_as(ascend_dump_tensor())) {
    AscendResource resource = call->args.empty()
                                  ? AscendResource::kNone
                                  : ResourceForAccessPtr(call->args[0]);
    return resource == AscendResource::kNone ? AscendResource::kCommon
                                             : resource;
  }
  if (call->op.same_as(ascend_set_flag()) ||
      call->op.same_as(ascend_wait_flag())) {
    if (call->args.size() < 2) {
      return AscendResource::kExplicit;
    }
    const auto *src = call->args[0].as<StringImmNode>();
    const auto *dst = call->args[1].as<StringImmNode>();
    return src == nullptr || dst == nullptr
               ? AscendResource::kExplicit
               : ResourceForPipePair(src->value, dst->value, *operation);
  }
  if (call->op.same_as(ascend_pipe_barrier()) ||
      call->op.same_as(ascend_set_cross_flag()) ||
      call->op.same_as(ascend_auto_barrier())) {
    return ResourceForPipeArgument(call, 0);
  }
  if (call->op.same_as(ascend_auto_set_flag()) ||
      call->op.same_as(ascend_auto_wait_flag())) {
    if (call->args.empty()) {
      return AscendResource::kExplicit;
    }
    const auto *event_type = call->args[0].as<StringImmNode>();
    return event_type == nullptr
               ? AscendResource::kExplicit
               : ResourceForEventType(event_type->value, *operation);
  }
  if (call->op.same_as(ascend_wait_cross_flag()) ||
      call->op.same_as(ascend_auto_set_cross_flag()) ||
      call->op.same_as(ascend_auto_wait_cross_flag())) {
    return ResourceForPipeArgument(call, 1);
  }

  if (call->op.same_as(builtin::call_extern())) {
    if (call->args.empty()) {
      return AscendResource::kExplicit;
    }
    const auto *callee = call->args[0].as<StringImmNode>();
    if (callee == nullptr) {
      return AscendResource::kExplicit;
    }
    *operation = callee->value;
    std::string name = NormalizeFunctionName(callee->value);
    if (name == "free_pipe_C") {
      return AscendResource::kCube;
    }
    if (name == "free_pipe_V") {
      return AscendResource::kVector;
    }
    auto it = GetOperationConfig().find(name);
    if (it != GetOperationConfig().end()) {
      return ResourceForConfiguredOperation(name, it->second, call->args, 1);
    }
    AscendResource resource = ResourceForAccessPtrs(call->args, 1, *operation);
    return resource == AscendResource::kNone ? AscendResource::kExplicit
                                             : resource;
  }

  if (op == nullptr) {
    return AscendResource::kNone;
  }
  auto it = GetOperationConfig().find(op->name);
  if (it != GetOperationConfig().end()) {
    return ResourceForConfiguredOperation(op->name, it->second, call->args, 0);
  }
  if (std::string(op->name).rfind("tl.ascend_", 0) == 0) {
    AscendResource resource = ResourceForAccessPtrs(call->args, 0, *operation);
    return resource == AscendResource::kNone ? AscendResource::kExplicit
                                             : resource;
  }
  return AscendResource::kNone;
}

bool IsContextDependentSyncCall(const CallNode *call) {
  if (call->op.same_as(ascend_pipe_barrier())) {
    if (call->args.empty()) {
      return false;
    }
    const auto *pipe = call->args[0].as<StringImmNode>();
    return pipe != nullptr && NormalizePipeName(pipe->value) == "ALL";
  }
  if (!call->op.same_as(ascend_set_flag()) &&
      !call->op.same_as(ascend_wait_flag())) {
    return false;
  }
  if (call->args.size() < 2 || call->args[0].as<StringImmNode>() == nullptr ||
      call->args[1].as<StringImmNode>() == nullptr) {
    return false;
  }
  std::string operation;
  return ResourceForCall(call, &operation) == AscendResource::kExplicit;
}

struct ResourceSummary {
  AscendResource resource{AscendResource::kNone};
  bool has_context_sync{false};
};

bool IsConcreteResource(AscendResource resource) {
  return resource == AscendResource::kCube ||
         resource == AscendResource::kVector;
}

class ExactResourceCollector final : public StmtExprVisitor {
public:
  ResourceSummary Collect(const Stmt &stmt) {
    VisitStmt(stmt);
    return {resource_, has_context_sync_};
  }

private:
  void Add(AscendResource resource) {
    if (resource == AscendResource::kNone ||
        resource == AscendResource::kCommon) {
      return;
    }
    if (resource_ == AscendResource::kNone) {
      resource_ = resource;
    } else if (resource_ != resource) {
      resource_ = AscendResource::kExplicit;
    }
  }

  void VisitExpr_(const CallNode *op) final {
    if (IsContextDependentSyncCall(op)) {
      has_context_sync_ = true;
      return;
    }
    std::string operation;
    Add(ResourceForCall(op, &operation));
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    Add(ResourceForStorageScope(op->buffer.scope()));
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    Add(ResourceForStorageScope(op->buffer.scope()));
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kVectorized) {
      Add(AscendResource::kVector);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == "resource_scope") {
      const auto *scope = op->value.as<IntImmNode>();
      if (scope != nullptr && scope->value == 0) {
        Add(AscendResource::kCube);
        return;
      }
      if (scope != nullptr && scope->value == 1) {
        Add(AscendResource::kVector);
        return;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  AscendResource resource_{AscendResource::kNone};
  bool has_context_sync_{false};
};

ResourceSummary SummarizeResources(const Stmt &stmt) {
  return ExactResourceCollector().Collect(stmt);
}

bool IsContextDependentSync(const Stmt &stmt) {
  const auto *evaluate = stmt.as<EvaluateNode>();
  if (evaluate == nullptr) {
    return false;
  }
  const auto *call = evaluate->value.as<CallNode>();
  if (call == nullptr) {
    return false;
  }
  return IsContextDependentSyncCall(call);
}

AscendResource RegionResource(const std::vector<ResourceSummary> &summaries) {
  AscendResource region = AscendResource::kNone;
  for (const ResourceSummary &summary : summaries) {
    if (summary.resource == AscendResource::kNone) {
      continue;
    }
    if (!IsConcreteResource(summary.resource) ||
        (region != AscendResource::kNone && region != summary.resource)) {
      return AscendResource::kExplicit;
    }
    region = summary.resource;
  }
  return region;
}

AscendResource NearestResource(const std::vector<ResourceSummary> &summaries,
                               int index, int step) {
  for (; index >= 0 && index < static_cast<int>(summaries.size());
       index += step) {
    if (summaries[index].resource != AscendResource::kNone) {
      return summaries[index].resource;
    }
  }
  return AscendResource::kNone;
}

class ContextualSyncResolver final : public StmtMutator {
  Stmt VisitStmt_(const SeqStmtNode *op) final {
    std::vector<ResourceSummary> summaries;
    summaries.reserve(op->seq.size());
    for (const Stmt &stmt : op->seq) {
      summaries.push_back(SummarizeResources(stmt));
    }

    AscendResource saved_context = context_;
    AscendResource region = RegionResource(summaries);
    AscendResource sequence_context = saved_context;
    if (IsConcreteResource(region)) {
      sequence_context = region;
    }

    Array<Stmt> seq;
    for (size_t i = 0; i < op->seq.size(); ++i) {
      AscendResource child_context = sequence_context;
      if (IsConcreteResource(summaries[i].resource)) {
        child_context = summaries[i].resource;
      } else if (!IsConcreteResource(child_context) &&
                 summaries[i].resource == AscendResource::kNone &&
                 summaries[i].has_context_sync) {
        AscendResource before =
            NearestResource(summaries, static_cast<int>(i) - 1, -1);
        AscendResource after =
            NearestResource(summaries, static_cast<int>(i) + 1, 1);
        if (before == after && IsConcreteResource(before)) {
          child_context = before;
        }
      }
      context_ = child_context;
      seq.push_back(VisitStmt(op->seq[i]));
    }
    context_ = saved_context;
    return SeqStmt(seq, op->span);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == "resource_scope") {
      return GetRef<Stmt>(op);
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    if (!IsContextDependentSync(stmt) || !IsConcreteResource(context_)) {
      return stmt;
    }
    int64_t scope = context_ == AscendResource::kVector ? 1 : 0;
    return AttrStmt(make_zero(DataType::Int(32)), "resource_scope",
                    IntImm(DataType::Int(32), scope), stmt);
  }

  AscendResource context_{AscendResource::kNone};
};

class AscendResourceScopeVerifier final : public StmtExprVisitor {
public:
  static PrimFunc Verify(PrimFunc func, bool require_explicit_scope) {
    AscendResourceScopeVerifier verifier(require_explicit_scope);
    verifier(func->body);
    return func;
  }

private:
  explicit AscendResourceScopeVerifier(bool require_explicit_scope)
      : require_explicit_scope_(require_explicit_scope) {}

  void Check(AscendResource resource, const std::string &operation) const {
    if (resource == AscendResource::kNone ||
        resource == AscendResource::kCommon) {
      return;
    }
    if (scope_ < 0) {
      ICHECK(!require_explicit_scope_ && resource != AscendResource::kExplicit)
          << "Ascend hardware operation must be inside T.Scope(\"C\") or "
             "T.Scope(\"V\"); enable tl.ascend_auto_cv_combine or scope it "
             "explicitly: "
          << operation;
      return;
    }
    if (resource == AscendResource::kCube) {
      ICHECK_EQ(scope_, 0) << "Cube operation must be inside T.Scope(\"C\"): "
                           << operation;
    } else if (resource == AscendResource::kVector) {
      ICHECK_EQ(scope_, 1) << "Vector operation must be inside T.Scope(\"V\"): "
                           << operation;
    }
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != "resource_scope") {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    const auto *scope = op->value.as<IntImmNode>();
    ICHECK(scope && (scope->value == 0 || scope->value == 1))
        << "resource_scope must be 0 (C) or 1 (V)";
    int new_scope = static_cast<int>(scope->value);
    ICHECK(scope_ < 0 || scope_ == new_scope)
        << "Conflicting nested T.Scope(\"" << (new_scope == 0 ? "C" : "V")
        << "\")";
    int saved_scope = scope_;
    scope_ = new_scope;
    VisitStmt(op->body);
    scope_ = saved_scope;
  }

  void VisitExpr_(const CallNode *op) final {
    std::string operation;
    Check(ResourceForCall(op, &operation), operation);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    AscendResource resource = ResourceForStorageScope(op->buffer.scope());
    Check(resource, "BufferStore to " + op->buffer.scope());
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    AscendResource resource = ResourceForStorageScope(op->buffer.scope());
    Check(resource, "BufferLoad from " + op->buffer.scope());
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kVectorized) {
      Check(AscendResource::kVector, "vectorized loop");
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  bool require_explicit_scope_;
  int scope_{-1};
};

class CVCombineEmitter : public StmtMutator {
public:
  explicit CVCombineEmitter(bool is_aiv) : is_aiv_(is_aiv) {}

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

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != "resource_scope") {
      return StmtMutator::VisitStmt_(op);
    }
    const auto *scope = op->value.as<IntImmNode>();
    ICHECK(scope && (scope->value == 0 || scope->value == 1));
    if (scope->value != static_cast<int>(is_aiv_)) {
      return Evaluate(0);
    }
    ++explicit_scope_depth_;
    Stmt body = VisitStmt(op->body);
    --explicit_scope_depth_;
    return body;
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
    const auto *call = op->value.as<CallNode>();
    if (call == nullptr) {
      return StmtMutator::VisitStmt_(op);
    }
    if (explicit_scope_depth_ > 0) {
      return StmtMutator::VisitStmt_(op);
    }
    std::string operation;
    AscendResource resource = ResourceForCall(call, &operation);
    if (resource == AscendResource::kCommon) {
      return GetRef<Stmt>(op);
    }
    if (resource == AscendResource::kCube ||
        resource == AscendResource::kVector) {
      bool keep = (resource == AscendResource::kVector) == is_aiv_;
      current_process_enabled_ = keep;
      return keep ? StmtMutator::VisitStmt_(op) : Evaluate(0);
    }
    ICHECK(resource != AscendResource::kExplicit)
        << "Unscoped opaque Ascend operation cannot be classified by "
           "CombineCV; place it in an explicit T.Scope: "
        << operation;
    return current_process_enabled_ ? StmtMutator::VisitStmt_(op) : Evaluate(0);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    if (explicit_scope_depth_ > 0) {
      return StmtMutator::VisitStmt_(op);
    }
    AscendResource resource = ResourceForStorageScope(op->buffer.scope());
    if (resource == AscendResource::kNone) {
      // Preserve the established CombineCV convention for scalar/global
      // stores. Their resource-specific inputs are classified separately.
      resource = AscendResource::kCube;
    }
    bool keep = (resource == AscendResource::kVector) == is_aiv_;
    current_process_enabled_ = keep;
    return keep ? StmtMutator::VisitStmt_(op) : Evaluate(0);
  }

private:
  const bool is_aiv_;
  bool current_process_enabled_{false};
  int explicit_scope_depth_{0};
};

} // namespace

class CombineCV : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f, PassContext ctx) {
    arith::Analyzer analyzer;
    CombineCV substituter(&analyzer);
    bool ascend_auto_combine =
        ctx->GetConfig<Bool>(ascendAutoCombine, Bool(false)).value();
    if (!ascend_auto_combine) {
      return f;
    }

    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = ContextualSyncResolver()(fptr->body);
    // Reject opaque outer calls and conflicting explicit scopes before the
    // split can discard their original context.
    AscendResourceScopeVerifier::Verify(f, false);
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

      CVCombineEmitter cubeStmt(false);
      CVCombineEmitter vecStmt(true);

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

tvm::transform::Pass AscendResourceScopeVerify() {
  auto pass_func = [=](PrimFunc f, IRModule, PassContext) {
    return AscendResourceScopeVerifier::Verify(std::move(f), true);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendResourceScopeVerify", {});
}

TVM_REGISTER_GLOBAL("tl.transform.AscendResourceScopeVerify")
    .set_body_typed(AscendResourceScopeVerify);

} // namespace tl
} // namespace tvm
