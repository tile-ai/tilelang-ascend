// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file cross_core_pipeline.cc
 * \brief Plan the p    ipeline between cube and vector
 */

#include "arith/ir_mutator_with_analyzer.h"
#include "tir/analysis/var_use_def_analysis.h"
#include "tir/transforms/ir_utils.h"

#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_set.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include "../op/ascend.h"
#include "../op/builtin.h"
#include "./common/collector.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

#define INVALID_SCOPE -1
#define CUBE_SCOPE 0
#define VEC_SCOPE 1

struct PipelineInfo {
  const ForNode *for_node;
  bool is_cross_core;
  int32_t scene;
  std::string loop_var_name;
};

std::unordered_map<std::string, std::string> callnodeMapPos_ = {
    {"wmma.matrix_a", "cube"},
    {"wmma.matrix_b", "cube"},
    {"wmma.accumulator", "cube"},
    {"shared.dyn", "cube"},
    {"shared", "vec"}};

int32_t checkBufferScope(Map<Var, String> location_map, const Var &var) {
  if (location_map.find(var) != location_map.end()) {
    if (callnodeMapPos_[location_map[var]] == "cube") {
      return CUBE_SCOPE;
    } else if (callnodeMapPos_[location_map[var]] == "vec") {
      return VEC_SCOPE;
    } else {
      return INVALID_SCOPE;
    }
  }
  return INVALID_SCOPE;
}

class CrossCoreDetector : public StmtVisitor {
public:
  CrossCoreDetector(Map<Var, String> location_map)
      : location_map_(location_map) {}

  std::vector<PipelineInfo> DetectCrossCorePipelines(const Stmt &stmt) {
    pipeline_infos_.clear();
    current_pipeline_info_ = nullptr;
    this->VisitStmt(stmt);
    std::vector<PipelineInfo> cross_core_pipelines;
    for (const auto &info : pipeline_infos_) {
      if (info.is_cross_core) {
        cross_core_pipelines.push_back(info);
      }
    }
    return cross_core_pipelines;
  }

  void VisitStmt_(const ForNode *loop) override {
    auto num_stages_anno = loop->annotations.Get("num_stages");
    if (num_stages_anno.defined()) {
      PipelineInfo *prev_pipeline = current_pipeline_info_;

      PipelineInfo new_info;
      new_info.for_node = loop;
      new_info.is_cross_core = false;
      new_info.scene = INVALID_SCOPE;
      new_info.loop_var_name = loop->loop_var->name_hint;

      current_pipeline_info_ = &new_info;
      pipeline_infos_.push_back(new_info);
      this->VisitStmt(loop->body);

      if (!pipeline_infos_.empty()) {
        pipeline_infos_.back() = new_info;
      }
      current_pipeline_info_ = prev_pipeline;
    } else {
      this->VisitStmt(loop->body);
    }
  }

  void VisitStmt_(const EvaluateNode *op) override {
    if (!current_pipeline_info_) {
      return;
    }

    auto call_node = op->value.as<CallNode>();
    auto scope = INVALID_SCOPE;
    for (int i = 1; i < call_node->args.size(); i++) {
      if (auto inner_node = call_node->args[i].as<CallNode>()) {
        auto buf_name = Downcast<Var>(inner_node->args[1]);
        scope = checkBufferScope(location_map_, buf_name);
        if (scope != INVALID_SCOPE) {
          break;
        }
      }
    }
    if (scope != INVALID_SCOPE) {
      if (current_pipeline_info_->scene == INVALID_SCOPE) {
        current_pipeline_info_->scene = scope;
      } else if (current_pipeline_info_->scene != scope) {
        current_pipeline_info_->is_cross_core = true;
      }
    }
  }

private:
  std::vector<PipelineInfo> pipeline_infos_;
  Map<Var, String> location_map_;
  PipelineInfo *current_pipeline_info_{nullptr};
};

class BufferMapTransformer {
public:
  BufferMapTransformer(const Map<Var, String> &location_map, int32_t num_stages)
      : location_map_(location_map), num_stages_(num_stages) {}

  Map<Var, Buffer>
  TransformBufferMap(const Map<Var, Buffer> &original_buffer_map) {
    Map<Var, Buffer> resized_buffer;
    for (const auto &kv : original_buffer_map) {
      Var var = kv.first;
      Buffer old_buffer = kv.second;
      Buffer new_buffer = old_buffer;

      if (IsWorkspaceBuffer(old_buffer)) {
        new_buffer = CreateResizedBuffer(old_buffer);
      }
      resized_buffer.Set(var, new_buffer);
    }
    return resized_buffer;
  }

private:
  bool IsWorkspaceBuffer(const Buffer &buffer) {
    std::string name = buffer->name;
    if (name.find("workspace") != 0) {
      return false;
    }
    return true;
  }

  Buffer CreateResizedBuffer(const Buffer &old_buffer) const {
    ObjectPtr<BufferNode> new_buffer =
        make_object<BufferNode>(*(old_buffer.get()));
    new_buffer->shape.insert(new_buffer->shape.begin(), PrimExpr(num_stages_));
    if (new_buffer->strides.size()) {
      ICHECK(new_buffer->strides.size() + 1 == new_buffer->shape.size());
      PrimExpr stride_0 = new_buffer->strides[0] * new_buffer->shape[1];
      new_buffer->strides.insert(new_buffer->strides.begin(), stride_0);
    }
    return Buffer(new_buffer);
  }

private:
  Map<Var, String> location_map_;
  int32_t num_stages_;
};

class LoopAnalyzer : public StmtVisitor {
public:
  /*!
   * \brief A single buffer access extracted from tvm_access_ptr(dtype, buf,
   * offset, extent, rw_mask).
   */
  struct AccessInfo {
    std::string buffer_name;
    PrimExpr offset;
    PrimExpr extent;
    bool is_read;
    bool is_write;
    int order;
  };

  struct StmtInfo {
    int idx;
    std::string type;
    std::string buffer_name;
    Stmt stmt;
    std::set<std::string> used_buffers;
    std::vector<AccessInfo> accesses;
    int32_t scope{INVALID_SCOPE};
  };

  struct WorkspaceWrite {
    int stmt_idx;
    std::string buffer_name;
    Call call;
  };

  const std::vector<std::string> IS_WRITE_GM = {
      "copy_l0c_to_gm", "copy_ub_to_gm", "atomic_add_l0c_to_gm",
      "atomic_add_ub_to_gm"};

  LoopAnalyzer(const ForNode *pipeline_loop,
               const Map<Var, String> location_map)
      : pipeline_loop_(pipeline_loop), location_map_(location_map) {}

  void Analyze() {
    all_statements_C_.clear();
    all_statements_V_.clear();
    all_statements_.clear();
    workspace_writes_C_.clear();
    workspace_writes_V_.clear();
    current_idx_C_ = 0;
    current_idx_V_ = 0;
    current_idx_ = 0;
    core_scope_ = INVALID_SCOPE;
    this->VisitStmt(pipeline_loop_->body);
  }

  const std::vector<StmtInfo> &all_statements_C() const {
    return all_statements_C_;
  }
  const std::vector<StmtInfo> &all_statements_V() const {
    return all_statements_V_;
  }
  const std::vector<StmtInfo> &all_statements() const {
    return all_statements_;
  }
  const std::vector<WorkspaceWrite> &workspace_writes_C() const {
    return workspace_writes_C_;
  }
  const std::vector<WorkspaceWrite> &workspace_writes_V() const {
    return workspace_writes_V_;
  }
  const Map<Var, String> &location_map() const { return location_map_; }

  void VisitStmt_(const SeqStmtNode *op) {
    for (const Stmt &stmt : op->seq) {
      this->VisitStmt(stmt);
    }
  }

  void VisitStmt_(const EvaluateNode *op) {
    StmtInfo info;
    info.type = "Evaluate";
    info.stmt = GetRef<Stmt>(op);
    if (auto call_node = op->value.as<CallNode>()) {
      int arg_start = call_node->op.same_as(builtin::call_extern()) ? 1 : 0;
      for (int idx = arg_start; idx < call_node->args.size(); idx++) {
        if (auto inter_node = call_node->args[idx].as<CallNode>()) {
          auto buf_name = Downcast<Var>(inter_node->args[1]);
          core_scope_ = checkBufferScope(location_map_, buf_name);
          if (core_scope_ != INVALID_SCOPE) {
            break;
          }
        }
      }
      if (core_scope_ == CUBE_SCOPE) {
        info.idx = current_idx_C_++;
      } else if (core_scope_ == VEC_SCOPE) {
        info.idx = current_idx_V_++;
      }
      if (auto workspace_name = FindWorkspaceName(call_node)) {
        info.buffer_name = workspace_name.value();
      }

      std::string normalized_name = "";
      if (call_node->op.same_as(builtin::call_extern())) {
        std::string func_name = call_node->args[0].as<StringImmNode>()->value;
        normalized_name = NormalizeFunctionName(func_name);
      } else {
        normalized_name = call_node->op.as<OpNode>()->name;
      }

      bool exists = std::find(IS_WRITE_GM.begin(), IS_WRITE_GM.end(),
                              normalized_name) != IS_WRITE_GM.end();
      if (exists) {
        WorkspaceWrite write;
        write.stmt_idx = info.idx;
        if (auto workspace_name = FindWorkspaceName(call_node)) {
          write.buffer_name = workspace_name.value();
        }
        if (core_scope_ == CUBE_SCOPE) {
          workspace_writes_C_.push_back(write);
        } else if (core_scope_ == VEC_SCOPE) {
          workspace_writes_V_.push_back(write);
        }
      }
      CollectBuffersAndAccesses(call_node, info.used_buffers, info.accesses);
    }
    if (core_scope_ == CUBE_SCOPE) {
      all_statements_C_.push_back(info);
    } else if (core_scope_ == VEC_SCOPE) {
      all_statements_V_.push_back(info);
    }
    if (core_scope_ != INVALID_SCOPE) {
      info.idx = current_idx_++;
      info.scope = core_scope_;
      all_statements_.push_back(info);
    }
  }

  void VisitStmt_(const ForNode *op) override {
    StmtInfo for_info;
    for_info.type = "For";
    for_info.stmt = GetRef<Stmt>(op);

    std::vector<StmtInfo> saved_statements_C = all_statements_C_;
    std::vector<StmtInfo> saved_statements_V = all_statements_V_;
    std::vector<StmtInfo> saved_statements = all_statements_;
    int saved_idx_C = current_idx_C_;
    int saved_idx_V = current_idx_V_;
    int saved_idx = current_idx_;

    this->VisitStmt(op->body);

    bool has_new_C = saved_statements_C.size() < all_statements_C_.size();
    bool has_new_V = saved_statements_V.size() < all_statements_V_.size();

    if (has_new_V) {
      ProcessInfo(workspace_writes_V_, all_statements_V_, saved_statements_V,
                  saved_idx_V, for_info);
      current_idx_V_ = all_statements_V_.size();
    }
    if (has_new_C) {
      ProcessInfo(workspace_writes_C_, all_statements_C_, saved_statements_C,
                  saved_idx_C, for_info);
      current_idx_C_ = all_statements_C_.size();
    }
    if (has_new_V || has_new_C) {
      StmtInfo unified_for_info = for_info;
      unified_for_info.scope = core_scope_;
      ProcessInfoUnified(all_statements_, saved_statements, saved_idx,
                         unified_for_info);
      current_idx_ = all_statements_.size();
    }
  }

  void VisitStmt_(const BlockRealizeNode *op) override {
    this->VisitStmt(op->block);
  }

private:
  void ProcessInfo(std::vector<WorkspaceWrite> &workspace_writes,
                   std::vector<StmtInfo> &all_statements,
                   std::vector<StmtInfo> &saved_statements, int saved_idx,
                   StmtInfo for_info) {
    for_info.idx = saved_idx++;

    for (auto &write_info : workspace_writes) {
      if (write_info.stmt_idx > for_info.idx) {
        write_info.stmt_idx = for_info.idx;
      }
    }

    std::set<std::string> for_node_buffers;
    std::vector<AccessInfo> for_node_accesses;
    for (size_t i = for_info.idx; i < all_statements.size(); i++) {
      auto buffers = all_statements[i].used_buffers;
      for (auto it = buffers.begin(); it != buffers.end(); ++it) {
        for_node_buffers.insert(*it);
      }
      for (const auto &access : all_statements[i].accesses) {
        for_node_accesses.push_back(access);
      }
    }
    all_statements = saved_statements;
    all_statements.push_back(for_info);
    all_statements.back().used_buffers = for_node_buffers;
    all_statements.back().accesses = for_node_accesses;
  }

  void ProcessInfoUnified(std::vector<StmtInfo> &all_statements,
                          std::vector<StmtInfo> &saved_statements,
                          int saved_idx, StmtInfo for_info) {
    for_info.idx = saved_idx++;

    std::set<std::string> for_node_buffers;
    std::vector<AccessInfo> for_node_accesses;
    for (size_t i = for_info.idx; i < all_statements.size(); i++) {
      for (const auto &buf : all_statements[i].used_buffers) {
        for_node_buffers.insert(buf);
      }
      for (const auto &access : all_statements[i].accesses) {
        for_node_accesses.push_back(access);
      }
    }
    all_statements = saved_statements;
    all_statements.push_back(for_info);
    all_statements.back().used_buffers = for_node_buffers;
    all_statements.back().accesses = for_node_accesses;
  }

  void CollectBuffersAndAccesses(const CallNode *call_node,
                                 std::set<std::string> &used_buffers,
                                 std::vector<AccessInfo> &accesses) {
    auto args = call_node->args;
    int start = call_node->op.same_as(builtin::call_extern()) ? 1 : 0;

    for (int i = start; i < args.size(); ++i) {
      if (auto inner_call_node = args[i].as<CallNode>()) {
        bool is_tvm_ap = inner_call_node->op.same_as(builtin::tvm_access_ptr());
        if (is_tvm_ap && inner_call_node->args.size() >= 5) {
          auto buf_var = Downcast<Var>(inner_call_node->args[1]);
          std::string buf_name = buf_var->name_hint;
          PrimExpr offset = inner_call_node->args[2];
          PrimExpr extent = inner_call_node->args[3];
          const IntImmNode *flag = inner_call_node->args[4].as<IntImmNode>();
          int rw_mask = flag ? static_cast<int>(flag->value) : 3;

          bool found_in_map =
              location_map_.find(buf_var) != location_map_.end();
          bool found_in_pos = false;
          if (found_in_map) {
            found_in_pos = callnodeMapPos_.find(location_map_[buf_var]) !=
                           callnodeMapPos_.end();
          }

          if (found_in_map && found_in_pos) {
            used_buffers.insert(buf_name);
          }

          AccessInfo access;
          access.buffer_name = buf_name;
          access.offset = offset;
          access.extent = extent;
          access.is_read = (rw_mask & 1) != 0;
          access.is_write = (rw_mask & 2) != 0;
          access.order = static_cast<int>(accesses.size());
          accesses.push_back(access);
        } else {
          auto buf_name = Downcast<Var>(inner_call_node->args[1]);
          bool found = location_map_.find(buf_name) != location_map_.end();
          if (found) {
            if (callnodeMapPos_.find(location_map_[buf_name]) !=
                callnodeMapPos_.end()) {
              used_buffers.insert(buf_name->name_hint);
            }
          }
        }
      }
    }
  }

  std::optional<std::string> FindWorkspaceName(const CallNode *call_node) {
    auto args = call_node->args;
    int start = call_node->op.same_as(builtin::call_extern()) ? 1 : 0;
    for (int i = start; i < args.size(); ++i) {
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

  std::string NormalizeFunctionName(const std::string &func_name) {
    std::string result = func_name;
    size_t template_pos = result.find('<');
    if (template_pos != std::string::npos) {
      result = result.substr(0, template_pos);
    }

    size_t ns_pos = result.find("tl::ascend::");
    if (ns_pos != std::string::npos) {
      result = result.substr(ns_pos + 12);
    }

    return result;
  }

private:
  const ForNode *pipeline_loop_;
  Map<Var, String> location_map_;
  std::vector<StmtInfo> all_statements_C_;
  std::vector<StmtInfo> all_statements_V_;
  std::vector<StmtInfo> all_statements_;
  std::vector<WorkspaceWrite> workspace_writes_C_;
  std::vector<WorkspaceWrite> workspace_writes_V_;
  int current_idx_C_{0};
  int current_idx_V_{0};
  int current_idx_{0};
  int32_t core_scope_{INVALID_SCOPE};
};

/*!
 * \brief Check if [a_off, a_off+a_ext) ⊇ [b_off, b_off+b_ext).
 * Conservative: returns true only when provable from constant values.
 */
static bool RangeContains(const PrimExpr &a_off, const PrimExpr &a_ext,
                          const PrimExpr &b_off, const PrimExpr &b_ext) {
  auto a0 = a_off.as<IntImmNode>();
  auto a1 = a_ext.as<IntImmNode>();
  auto b0 = b_off.as<IntImmNode>();
  auto b1 = b_ext.as<IntImmNode>();
  if (a0 && a1 && b0 && b1) {
    return a0->value <= b0->value &&
           a0->value + a1->value >= b0->value + b1->value;
  }
  return false;
}

/*!
 * \brief Check if [a_off, a_off+a_ext) ∩ [b_off, b_off+b_ext) ≠ ∅.
 * Conservative: returns true (assumes overlap) when non-constant.
 */
static bool RangeOverlaps(const PrimExpr &a_off, const PrimExpr &a_ext,
                          const PrimExpr &b_off, const PrimExpr &b_ext) {
  auto a0 = a_off.as<IntImmNode>();
  auto a1 = a_ext.as<IntImmNode>();
  auto b0 = b_off.as<IntImmNode>();
  auto b1 = b_ext.as<IntImmNode>();
  if (a0 && a1 && b0 && b1) {
    int64_t a_start = a0->value, a_end = a_start + a1->value;
    int64_t b_start = b0->value, b_end = b_start + b1->value;
    return a_start < b_end && b_start < a_end;
  }
  return true;
}

/*!
 * \brief Collect all read ranges of a specific buffer from an AST subtree.
 */
static void
CollectBufferReads(const Stmt &stmt, const std::string &target_buf,
                   std::vector<std::pair<PrimExpr, PrimExpr>> &reads) {
  if (const auto *eval = stmt.as<EvaluateNode>()) {
    if (const auto *call = eval->value.as<CallNode>()) {
      int start = call->op.same_as(builtin::call_extern()) ? 1 : 0;
      for (int i = start; i < static_cast<int>(call->args.size()); ++i) {
        if (const auto *inner = call->args[i].as<CallNode>()) {
          if (inner->op.same_as(builtin::tvm_access_ptr()) &&
              inner->args.size() >= 5) {
            auto buf_var = Downcast<Var>(inner->args[1]);
            if (buf_var->name_hint != target_buf)
              continue;
            const IntImmNode *flag = inner->args[4].as<IntImmNode>();
            int rw_mask = flag ? static_cast<int>(flag->value) : 3;
            if ((rw_mask & 1) != 0) {
              reads.push_back({inner->args[2], inner->args[3]});
            }
          }
        }
      }
    }
  } else if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const auto &s : seq->seq)
      CollectBufferReads(s, target_buf, reads);
  } else if (const auto *loop = stmt.as<ForNode>()) {
    CollectBufferReads(loop->body, target_buf, reads);
  } else if (const auto *ite = stmt.as<IfThenElseNode>()) {
    CollectBufferReads(ite->then_case, target_buf, reads);
    if (ite->else_case)
      CollectBufferReads(ite->else_case.value(), target_buf, reads);
  } else if (const auto *let = stmt.as<LetStmtNode>()) {
    CollectBufferReads(let->body, target_buf, reads);
  } else if (const auto *attr = stmt.as<AttrStmtNode>()) {
    CollectBufferReads(attr->body, target_buf, reads);
  } else if (const auto *realize = stmt.as<BlockRealizeNode>()) {
    CollectBufferReads(realize->block->body, target_buf, reads);
  }
}

/*!
 * \brief Tree-based exposed read analysis.
 *
 * For a fixed target (buffer, offset, extent), traverses the AST to determine:
 *   - has_exposed_read: a read of the target range exists that is NOT preceded
 *     by a write covering the target range.
 *   - must_kill: this subtree definitely writes (covers) the target range.
 *
 * Composition rules:
 *   Sequential(A; B):  exposed = A.exposed || (!A.kill && B.exposed)
 *                      kill = A.kill || B.kill
 *   For(body):         exposed = body.exposed  (first iteration, no prior
 * kills) kill = body.kill If(then, else):    exposed = then.exposed ||
 * else.exposed kill = then.kill && else.kill Primitive:         exposed =
 * reads_target && !covered_from_above kill = writes_covering_target
 */
static std::pair<bool, bool> CheckExposedRead(const Stmt &stmt,
                                              const std::string &target_buf,
                                              const PrimExpr &target_offset,
                                              const PrimExpr &target_extent,
                                              bool covered) {

  if (const auto *eval = stmt.as<EvaluateNode>()) {
    bool reads = false, kills = false;
    if (const auto *call = eval->value.as<CallNode>()) {
      int start = call->op.same_as(builtin::call_extern()) ? 1 : 0;
      for (int i = start; i < static_cast<int>(call->args.size()); ++i) {
        if (const auto *inner = call->args[i].as<CallNode>()) {
          if (inner->op.same_as(builtin::tvm_access_ptr()) &&
              inner->args.size() >= 5) {
            auto buf_var = Downcast<Var>(inner->args[1]);
            if (buf_var->name_hint != target_buf)
              continue;
            PrimExpr off = inner->args[2];
            PrimExpr ext = inner->args[3];
            const IntImmNode *flag = inner->args[4].as<IntImmNode>();
            int rw_mask = flag ? static_cast<int>(flag->value) : 3;
            if ((rw_mask & 1) != 0 &&
                RangeOverlaps(off, ext, target_offset, target_extent)) {
              reads = true;
            }
            if ((rw_mask & 2) != 0 &&
                RangeContains(off, ext, target_offset, target_extent)) {
              kills = true;
            }
          }
        }
      }
    }
    return {reads && !covered, kills};
  }

  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    bool any_exposed = false, any_kills = false;
    for (const auto &child : seq->seq) {
      auto [exp, kill] = CheckExposedRead(child, target_buf, target_offset,
                                          target_extent, covered || any_kills);
      any_exposed |= exp;
      any_kills |= kill;
    }
    return {any_exposed, any_kills};
  }

  if (const auto *loop = stmt.as<ForNode>()) {
    return CheckExposedRead(loop->body, target_buf, target_offset,
                            target_extent, covered);
  }

  if (const auto *ite = stmt.as<IfThenElseNode>()) {
    auto [t_exp, t_kill] = CheckExposedRead(
        ite->then_case, target_buf, target_offset, target_extent, covered);
    if (ite->else_case) {
      auto [e_exp, e_kill] =
          CheckExposedRead(ite->else_case.value(), target_buf, target_offset,
                           target_extent, covered);
      return {t_exp || e_exp, t_kill && e_kill};
    }
    return {t_exp, false};
  }

  if (const auto *let = stmt.as<LetStmtNode>()) {
    return CheckExposedRead(let->body, target_buf, target_offset, target_extent,
                            covered);
  }
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    return CheckExposedRead(attr->body, target_buf, target_offset,
                            target_extent, covered);
  }
  if (const auto *realize = stmt.as<BlockRealizeNode>()) {
    return CheckExposedRead(realize->block->body, target_buf, target_offset,
                            target_extent, covered);
  }

  return {false, false};
}

class LoopRewriter : public StmtMutator {
public:
  struct StageInfo {
    std::vector<LoopAnalyzer::StmtInfo> statements;
    std::set<std::string> used_buffers;
  };

  LoopRewriter(const LoopAnalyzer &analyzer, const ForNode *original_loop,
               int num_stages, int cross_interval = 1)
      : analyzer_(analyzer), original_loop_(original_loop),
        all_statements_(analyzer.all_statements()), num_stages_(num_stages),
        cross_interval_(cross_interval) {
    original_loop_var_ = original_loop->loop_var;
    std::string outer_var_name = original_loop_var_->name_hint + "_outer";
    outer_loop_var_ = Var(outer_var_name, original_loop_var_->dtype);
    stage_loop_var_ = Var("i", DataType::Int(32));
  }

  Stmt Rewrite() {
    std::vector<StageInfo> stages = SplitIntoStages();
    AnalyzeSharedBuffers(stages);
    return CreateStagedLoops(stages);
  }

  const std::set<std::string> &shared_buffers() const {
    return shared_buffers_;
  }

  const Var &outer_loop_var() const { return outer_loop_var_; }

  std::set<std::string> GetAllBuffersToAdjust() const {
    std::set<std::string> all_buffers = shared_buffers_;
    for (const auto &stmt_info : all_statements_) {
      if (!stmt_info.buffer_name.empty() &&
          stmt_info.buffer_name.find("workspace") != std::string::npos) {
        all_buffers.insert(stmt_info.buffer_name);
      }
    }
    return all_buffers;
  }

private:
  Stmt CreateStagedLoops(const std::vector<StageInfo> &stages) {
    Array<Stmt> stage_loops;
    for (size_t stage_idx = 0; stage_idx < stages.size(); ++stage_idx) {
      const auto &stage = stages[stage_idx];
      Stmt stage_body = CreateStageBody(stage.statements);
      Stmt stage_loop = CreateStageLoopWithBinding(stage_body);
      stage_loops.push_back(stage_loop);
    }
    if (stage_loops.size() == 1) {
      return stage_loops[0];
    } else {
      return SeqStmt(stage_loops);
    }
  }

  Stmt CreateStageBody(const std::vector<LoopAnalyzer::StmtInfo> &statements) {
    if (statements.empty()) {
      return Stmt();
    }
    if (statements.size() == 1) {
      return statements[0].stmt;
    }
    Array<Stmt> seq;
    for (const auto &stmt_info : statements) {
      seq.push_back(stmt_info.stmt);
    }
    return SeqStmt(seq);
  }

  Stmt AddLetStmtBinding(const Stmt &body) {
    PrimExpr binding_expr =
        outer_loop_var_ * make_const(outer_loop_var_.dtype(), num_stages_) +
        stage_loop_var_;
    std::string transformed_var_name =
        original_loop_var_->name_hint + "_transformed";
    Var transformed_var(transformed_var_name, original_loop_var_->dtype);

    class VarReplacer : public StmtExprMutator {
    public:
      VarReplacer(const Var &old_var, const Var &new_var)
          : old_var_(old_var), new_var_(new_var) {}

      PrimExpr VisitExpr_(const VarNode *op) override {
        if (op->name_hint == old_var_->name_hint) {
          return new_var_;
        }
        return GetRef<PrimExpr>(op);
      }

    private:
      const Var &old_var_;
      const Var &new_var_;
    };

    VarReplacer replacer(original_loop_var_, transformed_var);
    Stmt new_body = replacer(body);

    return LetStmt(transformed_var, binding_expr, new_body);
  }

  Stmt CreateStageLoopWithBinding(const Stmt &body) {
    PrimExpr min = make_const(DataType::Int(32), 0);
    PrimExpr extent = make_const(DataType::Int(32), num_stages_);
    Stmt loop_body = AddLetStmtBinding(body);

    Map<String, ObjectRef> annotations;
    annotations.Set("stage_loop", Bool(true));
    annotations.Set("tl_cross_interval", PrimExpr(cross_interval_));
    annotations.Set("tl_original_loop_var", original_loop_var_);

    return For(stage_loop_var_, min, extent, ForKind::kSerial, loop_body,
               original_loop_->thread_binding, annotations,
               original_loop_->span);
  }

  std::vector<StageInfo> SplitIntoStages() {
    std::vector<StageInfo> stages;
    if (all_statements_.empty())
      return stages;

    StageInfo current_stage;
    int32_t current_scope = all_statements_[0].scope;

    for (const auto &stmt_info : all_statements_) {
      if (stmt_info.scope != current_scope) {
        if (!current_stage.statements.empty()) {
          stages.push_back(current_stage);
          current_stage = StageInfo();
        }
        current_scope = stmt_info.scope;
      }
      current_stage.statements.push_back(stmt_info);
      for (const auto &buffer : stmt_info.used_buffers) {
        current_stage.used_buffers.insert(buffer);
      }
    }
    if (!current_stage.statements.empty()) {
      stages.push_back(current_stage);
    }

    return stages;
  }

  int32_t checkBufferScope(const std::string &buffer) {
    for (const auto &pair : analyzer_.location_map()) {
      if (pair.first.get()->name_hint == buffer) {
        std::string scope_str = pair.second;
        auto it = callnodeMapPos_.find(scope_str);
        if (it != callnodeMapPos_.end()) {
          if (it->second == "cube") {
            return CUBE_SCOPE;
          } else if (it->second == "vec") {
            return VEC_SCOPE;
          }
        }
        return INVALID_SCOPE;
      }
    }
    return INVALID_SCOPE;
  }

  Stmt ReconstructStageBody(const StageInfo &stage) const {
    if (stage.statements.size() == 1)
      return stage.statements[0].stmt;
    Array<Stmt> seq;
    for (const auto &s : stage.statements)
      seq.push_back(s.stmt);
    return SeqStmt(seq);
  }

  void AnalyzeSharedBuffers(const std::vector<StageInfo> &stages) {
    std::unordered_map<std::string, std::vector<int>> buffer_stage_map;
    for (size_t stage_idx = 0; stage_idx < stages.size(); ++stage_idx) {
      for (const auto &buffer : stages[stage_idx].used_buffers) {
        buffer_stage_map[buffer].push_back(stage_idx);
      }
    }

    for (const auto &[buffer, stage_indices] : buffer_stage_map) {
      if (stage_indices.size() <= 1) {
        continue;
      }
      bool is_shared = false;
      for (int si : stage_indices) {
        Stmt stage_body = ReconstructStageBody(stages[si]);

        std::vector<std::pair<PrimExpr, PrimExpr>> reads;
        CollectBufferReads(stage_body, buffer, reads);

        for (const auto &[offset, extent] : reads) {
          auto [exposed, kill] =
              CheckExposedRead(stage_body, buffer, offset, extent, false);
          if (exposed) {
            is_shared = true;
            break;
          }
        }
        if (is_shared)
          break;
      }

      if (is_shared) {
        shared_buffers_.insert(buffer);
      }
    }
  }

private:
  const LoopAnalyzer &analyzer_;
  const ForNode *original_loop_;
  const std::vector<LoopAnalyzer::StmtInfo> &all_statements_;
  std::set<std::string> shared_buffers_;
  int num_stages_;
  int cross_interval_;
  Var original_loop_var_;
  Var outer_loop_var_;
  Var stage_loop_var_;
};

class CrossCorePipeline : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f, PassContext ctx) {
    arith::Analyzer analyzer;
    CrossCorePipeline substituter(&analyzer);

    return substituter.Transform(std::move(f), ctx);
  }

  PrimFunc Transform(PrimFunc f, PassContext ctx) {
    PrimFuncNode *fptr = f.CopyOnWrite();
    tir::PostOrderVisit(f->body, [&](const ObjectRef &obj) {
      if (const auto *realize = obj.as<tir::BlockRealizeNode>()) {
        for (auto buf : realize->block->alloc_buffers) {
          String scope = GetPtrStorageScope(buf->data);
          location_map_.Set(buf->data, scope);
        }
      }
    });

    CrossCoreDetector detector(location_map_);
    cross_core_pipelines_ = detector.DetectCrossCorePipelines(fptr->body);
    if (cross_core_pipelines_.empty()) {
      return f;
    }

    ICHECK(cross_core_pipelines_.size() == 1)
        << "Cross_core_pipeline: only support one cross core pipeline body, "
           "but got "
        << cross_core_pipelines_.size();

    const auto info = cross_core_pipelines_[0];
    auto num_stages_anno = info.for_node->annotations.Get("num_stages");
    int num_stages = num_stages_anno.as<IntImmNode>()->value;

    BufferMapTransformer buffer_transformer(location_map_, num_stages);
    origin_map_ = fptr->buffer_map;
    auto buffer_result =
        buffer_transformer.TransformBufferMap(fptr->buffer_map);
    fptr->buffer_map = buffer_result;
    fptr->body = this->VisitStmt(fptr->body);

    auto fn_attr = fptr->attrs.CopyOnWrite();
    fn_attr->dict.Set("buffer_versions", collected_buffer_versions_);

    return f;
  }

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  Stmt VisitStmt_(const ForNode *op) override {
    if (op == cross_core_pipelines_[0].for_node) {
      return ProcessCrossCorePipeline(op);
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt ProcessCrossCorePipeline(const ForNode *pipeline) {
    LoopAnalyzer analyzer(pipeline, location_map_);
    analyzer.Analyze();
    if (analyzer.workspace_writes_C().empty() &&
        analyzer.workspace_writes_V().empty()) {
      return arith::IRMutatorWithAnalyzer::VisitStmt_(pipeline);
    }

    auto num_stages_anno = pipeline->annotations.Get("num_stages");
    int num_stages = num_stages_anno.as<IntImmNode>()->value;
    auto cross_interval_anno = pipeline->annotations.Get("tl_cross_interval");
    int cross_interval = cross_interval_anno.defined()
                             ? cross_interval_anno.as<IntImmNode>()->value
                             : 1;
    LoopRewriter rewriter(analyzer, pipeline, num_stages, cross_interval);
    Stmt staged_loops = rewriter.Rewrite();
    const Var &outer_loop_var = rewriter.outer_loop_var();
    Stmt outer_loop =
        ModifyOuterLoop(pipeline, staged_loops, num_stages, outer_loop_var);
    std::set<std::string> buffers_to_adjust = rewriter.GetAllBuffersToAdjust();
    if (!buffers_to_adjust.empty()) {
      // Build a combined buffer map that includes both function params
      // (origin_map_) and locally allocated buffers (alloc_buffers), so
      // AdjustBuffersAndAccess can look up shapes for on-chip buffers like
      // r_factors, sumexp_is, etc.
      Map<Var, Buffer> combined_map(origin_map_);
      if (const BlockRealizeNode *realize =
              pipeline->body.as<BlockRealizeNode>()) {
        for (const auto &buf : realize->block->alloc_buffers) {
          combined_map.Set(buf->data, buf);
        }
      }
      outer_loop = AdjustBuffersAndAccess(combined_map, outer_loop,
                                          buffers_to_adjust, num_stages);
    }

    if (const BlockRealizeNode *original_realize =
            pipeline->body.as<BlockRealizeNode>()) {
      const BlockNode *original_block = original_realize->block.as<BlockNode>();
      if (original_block) {
        auto new_block = make_object<BlockNode>(*original_block);
        new_block->body = outer_loop;
        const auto &shared_buffers = rewriter.shared_buffers();
        Block extended_block =
            ExtendAllBuffers(Block(new_block), num_stages, shared_buffers);
        auto extended_block_node = extended_block.CopyOnWrite();
        extended_block_node->body = outer_loop;
        Stmt new_realize =
            BlockRealize(original_realize->iter_values,
                         original_realize->predicate, extended_block);
        return this->VisitStmt(new_realize);
      }
    }
    return this->VisitStmt(outer_loop);
  }

  Block ExtendAllBuffers(const Block &block, int num_stages,
                         const std::set<std::string> &shared_buffers) {
    ObjectPtr<BlockNode> new_block = make_object<BlockNode>(*block.get());

    if (!new_block->alloc_buffers.empty()) {
      Array<Buffer> new_alloc_buffers;

      for (const auto &buffer : new_block->alloc_buffers) {
        std::string name = buffer->name;
        bool is_workspace = name.find("workspace") != std::string::npos;
        bool is_shared_buffer =
            shared_buffers.find(name) != shared_buffers.end();

        if (is_workspace || is_shared_buffer) {
          ObjectPtr<BufferNode> extended_buffer =
              make_object<BufferNode>(*buffer.get());

          if (!extended_buffer->shape.empty()) {
            Array<PrimExpr> new_shape = extended_buffer->shape;
            new_shape.insert(new_shape.begin(), PrimExpr(num_stages));
            extended_buffer->shape = new_shape;
          }

          this->collected_buffer_versions_.Set(extended_buffer->data,
                                               PrimExpr(num_stages));

          new_alloc_buffers.push_back(Buffer(extended_buffer));
        } else {
          new_alloc_buffers.push_back(buffer);
        }
      }
      new_block->alloc_buffers = new_alloc_buffers;
    }
    return Block(new_block);
  }

  Stmt AdjustBuffersAndAccess(Map<Var, Buffer> origin_map, const Stmt &stmt,
                              const std::set<std::string> &buffers_to_adjust,
                              int num_stages) {
    class BufferAccessAdjuster : public StmtMutator {
    public:
      BufferAccessAdjuster(Map<Var, Buffer> origin_map,
                           const std::set<std::string> &buffers_to_adjust,
                           int num_stages)
          : origin_map_(origin_map), buffers_to_adjust_(buffers_to_adjust),
            num_stages_(num_stages) {}

      Stmt VisitStmt_(const ForNode *op) override {
        if (op->annotations.Get("stage_loop")) {
          stage_var_ = op->loop_var;
          stage_idx_ = 0;
          Stmt new_body = this->VisitStmt(op->body);
          stage_var_ = Var();
          return For(op->loop_var, op->min, op->extent, op->kind, new_body,
                     op->thread_binding, op->annotations, op->span);
        }
        return StmtMutator::VisitStmt_(op);
      }

      Stmt VisitStmt_(const EvaluateNode *op) override {
        if (stage_var_.defined()) {
          stage_idx_++;
        }

        if (const CallNode *call = op->value.as<CallNode>()) {
          Array<PrimExpr> new_args =
              AdjustCallArgs(origin_map_, call->args, call);
          if (new_args.size() > 0) {
            if (call->op.same_as(tl::ascend_muls()) ||
                call->op.same_as(tl::ascend_adds())) {
              if (const CallNode *scalar_call = new_args[2].as<CallNode>()) {
                if (const VarNode *var = scalar_call->args[1].as<VarNode>()) {
                  std::string buffer_name = var->name_hint;
                  if (buffers_to_adjust_.find(buffer_name) !=
                      buffers_to_adjust_.end()) {
                    PrimExpr scalar_offset = scalar_call->args[2];
                    new_args.Set(3, new_args[3] + scalar_offset);
                  }
                }
              }
            }
            Call new_call(call->dtype, call->op, new_args, call->span);
            return Evaluate(new_call);
          }
        }
        return StmtMutator::VisitStmt_(op);
      }

    private:
      Array<PrimExpr> AdjustCallArgs(Map<Var, Buffer> origin_map,
                                     const Array<PrimExpr> &args,
                                     const CallNode *parent_call) {
        Array<PrimExpr> new_args;
        bool modified = false;
        PrimExpr i_value = stage_var_;

        for (const auto &arg : args) {
          if (const CallNode *inner_call = arg.as<CallNode>()) {
            if (inner_call->op.same_as(builtin::tvm_access_ptr())) {
              if (inner_call->args.size() >= 2) {
                if (const VarNode *var = inner_call->args[1].as<VarNode>()) {
                  std::string buffer_name = var->name_hint;
                  bool is_shared_buffer =
                      buffers_to_adjust_.find(buffer_name) !=
                      buffers_to_adjust_.end();
                  bool is_workspace_buffer =
                      buffer_name.find("workspace") != std::string::npos;

                  if (is_shared_buffer || is_workspace_buffer) {
                    Array<PrimExpr> new_inner_args;
                    new_inner_args.push_back(inner_call->args[0]);
                    new_inner_args.push_back(inner_call->args[1]);

                    if (inner_call->args.size() >= 3) {
                      PrimExpr original_offset = inner_call->args[2];
                      PrimExpr new_offset = original_offset;

                      if (stage_var_.defined()) {
                        if (is_shared_buffer || is_workspace_buffer) {
                          for (auto &kv : origin_map) {
                            Buffer buffer = kv.second;
                            if (buffer->name == buffer_name) {
                              PrimExpr total_size = 1;
                              for (int i = 0;
                                   i < static_cast<int>(buffer->shape.size());
                                   ++i) {
                                total_size = total_size * buffer->shape[i];
                              }
                              new_offset =
                                  original_offset + i_value * total_size;
                              break;
                            }
                          }
                        }
                      }
                      new_inner_args.push_back(new_offset);
                      for (size_t i = 3; i < inner_call->args.size(); ++i) {
                        new_inner_args.push_back(inner_call->args[i]);
                      }
                      Call new_inner_call(inner_call->dtype, inner_call->op,
                                          new_inner_args, inner_call->span);
                      new_args.push_back(new_inner_call);
                      modified = true;
                      continue;
                    }
                  }
                }
              }
            }
          }
          new_args.push_back(arg);
        }
        return modified ? new_args : Array<PrimExpr>();
      }

    private:
      const std::set<std::string> &buffers_to_adjust_;
      Map<Var, Buffer> origin_map_;
      int num_stages_;
      Var stage_var_;
      int stage_idx_;
    };
    BufferAccessAdjuster adjuster(origin_map, buffers_to_adjust, num_stages);
    return adjuster(stmt);
  }

  Stmt ModifyOuterLoop(const ForNode *pipeline, const Stmt &inner_loops,
                       int num_stages, const Var &outer_loop_var) {
    PrimExpr original_extent = pipeline->extent;
    PrimExpr new_extent;
    if (auto int_imm = original_extent.as<IntImmNode>()) {
      int loop = int_imm->value;
      new_extent = make_const(original_extent.dtype(), loop / num_stages);
    } else {
      new_extent =
          original_extent * make_const(original_extent.dtype(), num_stages);
    }

    Map<String, ObjectRef> annotations;
    annotations.Set("tl_original_extent", original_extent);

    return For(outer_loop_var, pipeline->min, new_extent, ForKind::kSerial,
               inner_loops, pipeline->thread_binding, annotations,
               pipeline->span);
  }

private:
  Map<Var, String> location_map_;
  Map<Var, Buffer> origin_map_;
  std::vector<PipelineInfo> cross_core_pipelines_;
  Map<Var, PrimExpr> collected_buffer_versions_;
};

tvm::transform::Pass CrossCorePipeline() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto new_func = CrossCorePipeline::Substitute(std::move(f), ctx);
    return new_func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.CrossCorePipeline", {});
}

// regist host path
TVM_REGISTER_GLOBAL("tl.transform.CrossCorePipeline")
    .set_body_typed(CrossCorePipeline);
} // namespace tl
} // namespace tvm
