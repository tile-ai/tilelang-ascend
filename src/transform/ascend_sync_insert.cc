// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file ascend_sync_insert.cc
 * \brief Sync insertion for Ascend NPU
 */

#include <algorithm>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

#include "arith/ir_mutator_with_analyzer.h"
#include "tir/analysis/var_use_def_analysis.h"

#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>

#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>

#include "../op/ascend.h"
#include "../op/builtin.h"
#include "./common/collector.h"
#include "./common/operation_config.h"

#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

static constexpr const char *kAscendAutoSync = "tl.ascend_auto_sync";

TVM_REGISTER_PASS_CONFIG_OPTION(kAscendAutoSync, Bool);

class AscendSyncInsert : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f, const std::string &config_path,
                             PassContext ctx, Target target,
                             std::string platform) {
    arith::Analyzer analyzer;
    AscendSyncInsert syncInserter(&analyzer, target, platform);

    auto address_map = f->GetAttr<Map<Var, PrimExpr>>("address_map")
                           .value_or(Map<Var, PrimExpr>());
    auto size_map = f->GetAttr<Map<Var, PrimExpr>>("size_map")
                        .value_or(Map<Var, PrimExpr>());
    syncInserter.InitConfig(config_path, address_map, size_map);

    PrimFuncNode *fptr = f.CopyOnWrite();
    auto fn_attr = fptr->attrs.CopyOnWrite();

    bool ascend_auto_sync =
        ctx->GetConfig<Bool>(kAscendAutoSync, Bool(false)).value();
    if (!ascend_auto_sync) {
      return f;
    }

    // Issue #1304: pre-scan for GM buffers that receive scalar stores, so
    // that every MTE3 DMA write to them (including the first one, before any
    // scalar store has been walked) gets the cache-coherence treatment.
    syncInserter.CollectScalarWrittenGmBuffers(f->body);

    auto preprocessed = syncInserter.PreprocessUnrollForLoops(f->body);

    Stmt processed_body = syncInserter(preprocessed.first);

    fptr->body = syncInserter.MergeAndRebuildForLoops(processed_body,
                                                      preprocessed.second);

    return f;
  }

  explicit AscendSyncInsert(arith::Analyzer *analyzer, Target target,
                            std::string platform)
      : arith::IRMutatorWithAnalyzer(analyzer), target_(target),
        platform_(platform) {}

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  struct LoopInfo {
    Var loop_var;
    PrimExpr min;
    PrimExpr extent;
    ForKind kind;
    Map<String, ObjectRef> annotations;
    std::string loop_id;
    int depth;

    std::string toString() const {
      std::ostringstream oss;
      oss << "LoopInfo{";
      oss << "loop_var: '" << loop_var->name_hint << "', ";
      oss << "min: " << min << ", ";
      oss << "extent: " << extent << ", ";
      oss << "kind: " << static_cast<int>(kind) << ", ";
      oss << "loop_id: '" << loop_id << "', ";
      oss << "depth: " << depth;
      oss << "}";
      return oss.str();
    }
  };

  void InitConfig(const std::string &config_path,
                  const Map<Var, PrimExpr> &address_map,
                  const Map<Var, PrimExpr> &size_map) {
    event_id_counter_ = 0;
    address_map_ = address_map;
    size_map_ = size_map;
    LoadDefaultConfig();
  }

  void LoadDefaultConfig() {
    event_mapping_ = GetEventMapping();
    operation_config_ = GetOperationConfig();
  }

  std::pair<Stmt, std::vector<LoopInfo>>
  PreprocessUnrollForLoops(const Stmt &stmt) {
    ForLoopUnroller unroller;
    auto result = unroller(stmt);
    return {result.first, result.second};
  }

  Stmt VisitStmt_(const SeqStmtNode *op) override {
    std::vector<Stmt> new_stmts;
    for (const Stmt &stmt : op->seq) {
      new_stmts.push_back(VisitStmt(stmt));
    }

    if (new_stmts.empty()) {
      return Evaluate(0);
    } else if (new_stmts.size() == 1) {
      return new_stmts[0];
    } else {
      return SeqStmt(new_stmts);
    }
  }

  Stmt VisitStmt_(const EvaluateNode *op) override {
    auto current_accesses = AnalyzeStmtAccesses(GetRef<Stmt>(op));
    auto sync_requirements = CollectSyncRequirements(current_accesses);
    auto optimized_syncs = OptimizeSyncRequirements(sync_requirements);

    std::vector<Stmt> stmts;
    for (const auto &sync_type : optimized_syncs) {
      InsertSynchronization(sync_type, stmts);
    }

    UpdateSyncStatesAfterSync(optimized_syncs);

    // Issue #1304: a GM buffer written by both scalar stores (S pipe, through
    // a write-back cache) and an MTE3 DMA needs cache-coherence treatment
    // around the DMA. The scalar store's cache-line fill (read-modify-write)
    // can snapshot GM before the DMA lands, and the dirty line's later
    // eviction stamps stale bytes over the freshly copied data. Clean the
    // destination's first cache line before the DMA and order later scalar
    // stores behind the DMA (MTE3_S). Only kernels that mix scalar and DMA
    // writes to the same GM buffer pay this cost; pure-DMA kernels are
    // unaffected.
    PrimExpr dcci_buffer;
    for (const auto &access : current_accesses) {
      if (access.is_write && access.pipeline == "PIPE_MTE3" &&
          scalar_written_gm_buffers_.count(access.buffer_name) > 0) {
        dcci_buffer = FindBufferArgExpr(op, access.buffer_name);
        break;
      }
    }
    if (dcci_buffer.defined()) {
      stmts.push_back(CreateDcci(dcci_buffer));
    }

    stmts.push_back(GetRef<Stmt>(op));

    if (dcci_buffer.defined()) {
      int event_id = AllocateEventId();
      stmts.push_back(CreateSetFlag("MTE3_S", event_id));
      stmts.push_back(CreateWaitFlag("MTE3_S", event_id));
    }

    UpdateLatestAccessHistory(current_accesses);

    if (stmts.size() == 1) {
      return stmts[0];
    } else {
      return SeqStmt(stmts);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode *op) override {
    auto value_accesses = AnalyzeExprAccesses(op->value);
    for (auto &read_access : value_accesses) {
      read_access.pipeline = "PIPE_S";
      read_access.is_write = false;
      read_access.operation = "buffer_load";
    }
    BufferAccess current_write;
    current_write.buffer_name = op->buffer->data->name_hint;
    current_write.is_write = true;
    current_write.pipeline = "PIPE_S";
    current_write.operation = "buffer_store";
    current_write.is_sliced = false;
    current_write.sync_graph = SyncGraph();
    current_write.pipe_barriers = {};
    current_write.physical_address =
        GetPhysicalAddress(current_write.buffer_name);
    value_accesses.push_back(current_write);

    auto optimized_syncs =
        OptimizeSyncRequirements(CollectSyncRequirements(value_accesses));
    std::vector<Stmt> stmts;
    for (const auto &sync_type : optimized_syncs) {
      InsertSynchronization(sync_type, stmts);
    }
    // These handoffs precede the new scalar access and cannot protect it.
    UpdateSyncStatesAfterSync(optimized_syncs);
    stmts.push_back(GetRef<Stmt>(op));
    UpdateLatestAccessHistory(value_accesses);

    if (stmts.size() == 1) {
      return stmts[0];
    } else {
      return SeqStmt(stmts);
    }
  }

  Stmt VisitStmt_(const AttrStmtNode *op) override {
    if (op->attr_key == "resource_scope") {
      const int resource_scope =
          static_cast<int>(Downcast<IntImm>(op->value)->value);
      if (resource_scope == current_resource_scope_) {
        // Explicit blocks within the same execution unit share outstanding
        // accesses. Keep their lexical scopes without resetting dependencies.
        Stmt new_body = VisitStmt(op->body);
        return AttrStmt(op->node, op->attr_key, op->value, new_body);
      }
      const int saved_resource_scope = current_resource_scope_;
      current_resource_scope_ = resource_scope;
      auto saved_access_history = current_access_history_;

      current_access_history_.clear();

      Stmt new_body = VisitStmt(op->body);

      current_access_history_ = saved_access_history;
      current_resource_scope_ = saved_resource_scope;

      return AttrStmt(op->node, op->attr_key, op->value, new_body);
    } else if (op->attr_key == "unrolled_loop") {
      const PrimExpr extent = Downcast<PrimExpr>(op->node);
      if (analyzer_->CanProve(extent <= 0)) {
        // No body access or handoff executes on this path.
        return Evaluate(0);
      }
      const bool may_be_empty = !analyzer_->CanProve(extent > 0);
      auto entry_history =
          may_be_empty ? current_access_history_ : AccessHistory{};
      const std::string loop_id = Downcast<StringImm>(op->value)->value;
      loop_exit_states_.push_back({loop_id, AccessHistory{}, false});
      Stmt new_body = VisitStmt(op->body);
      ICHECK(loop_exit_states_.back().first_iteration_seen)
          << "Missing first iteration exit for " << loop_id;
      // The second simulated iteration discovers back-edge handoffs. Its
      // additional completion facts need not hold when the real loop executes
      // only once. The first exit is the conservative invariant for all
      // positive trip counts; rebuilding must preserve both bodies' handoffs.
      current_access_history_ =
          std::move(loop_exit_states_.back().first_iteration_history);
      loop_exit_states_.pop_back();
      if (may_be_empty) {
        JoinAccessHistories(entry_history);
      }
      return AttrStmt(op->node, op->attr_key, op->value, new_body);
    } else if (op->attr_key == "iteration_end") {
      ICHECK(!loop_exit_states_.empty());
      auto &loop = loop_exit_states_.back();
      if (Downcast<StringImm>(op->value)->value == loop.loop_id + "_iter1") {
        loop.first_iteration_history = current_access_history_;
        loop.first_iteration_seen = true;
      }
      return GetRef<Stmt>(op);
    } else if (op->attr_key == "iteration_start") {
      return GetRef<Stmt>(op);
    }

    Stmt new_body = VisitStmt(op->body);
    return AttrStmt(op->node, op->attr_key, op->value, new_body);
  }

  Stmt VisitStmt_(const LetStmtNode *op) override {
    auto value_accesses = AnalyzeExprAccesses(op->value);

    bool has_sliced_access = false;
    for (const auto &access : value_accesses) {
      if (access.is_sliced) {
        has_sliced_access = true;
        break;
      }
    }

    std::vector<Stmt> stmts_before_let;
    if (has_sliced_access) {
      InsertSynchronization("PipeBarrier_ALL", stmts_before_let);
    }

    Stmt new_body = VisitStmt(op->body);

    Stmt new_let = LetStmt(op->var, op->value, new_body);

    if (!stmts_before_let.empty()) {
      stmts_before_let.push_back(new_let);
      if (stmts_before_let.size() == 1) {
        return stmts_before_let[0];
      } else {
        return SeqStmt(stmts_before_let);
      }
    } else {
      return new_let;
    }
  }

  Stmt VisitStmt_(const IfThenElseNode *op) override {
    std::vector<Stmt> stmts;
    InsertSynchronization("PipeBarrier_ALL", stmts);

    current_access_history_.clear();
    Stmt then_case = VisitStmt(op->then_case);

    Optional<Stmt> else_case;
    if (op->else_case.defined()) {
      current_access_history_.clear();
      else_case = VisitStmt(op->else_case.value());
    }

    stmts.push_back(IfThenElse(op->condition, then_case, else_case));

    InsertSynchronization("PipeBarrier_ALL", stmts);
    current_access_history_.clear();
    return SeqStmt(stmts);
  }

  Stmt MergeAndRebuildForLoops(const Stmt &processed_stmt,
                               const std::vector<LoopInfo> &loop_infos) {
    LoopRebuilder rebuilder(loop_infos);
    return rebuilder(processed_stmt);
  }

private:
  class ForLoopUnroller : public StmtMutator {
  public:
    std::pair<Stmt, std::vector<LoopInfo>> operator()(const Stmt &stmt) {
      loop_infos_.clear();
      current_depth_ = 0;
      Stmt result = VisitStmt(stmt);
      return {result, loop_infos_};
    }

    Stmt VisitStmt_(const ForNode *op) override {
      LoopInfo info;
      info.loop_var = op->loop_var;
      info.min = op->min;
      info.extent = op->extent;
      info.kind = op->kind;
      info.annotations = op->annotations;

      static int loop_counter = 0;
      info.loop_id = "loop_" + std::to_string(loop_counter++);
      info.depth = current_depth_;
      loop_infos_.push_back(info);

      current_depth_++;
      Stmt processed_body = VisitStmt(op->body);
      current_depth_--;

      std::string loop_id = info.loop_id;
      std::vector<Stmt> unrolled_stmts;

      unrolled_stmts.push_back(
          AttrStmt(make_zero(DataType::Int(32)), "iteration_start",
                   StringImm(loop_id + "_iter1"), Evaluate(0)));
      unrolled_stmts.push_back(processed_body);
      unrolled_stmts.push_back(
          AttrStmt(make_zero(DataType::Int(32)), "iteration_end",
                   StringImm(loop_id + "_iter1"), Evaluate(0)));

      unrolled_stmts.push_back(
          AttrStmt(make_zero(DataType::Int(32)), "iteration_start",
                   StringImm(loop_id + "_iter2"), Evaluate(0)));
      unrolled_stmts.push_back(processed_body);
      unrolled_stmts.push_back(
          AttrStmt(make_zero(DataType::Int(32)), "iteration_end",
                   StringImm(loop_id + "_iter2"), Evaluate(0)));

      if (unrolled_stmts.empty()) {
        return Evaluate(0);
      }

      Stmt unrolled_seq;
      if (unrolled_stmts.size() == 1) {
        unrolled_seq = unrolled_stmts[0];
      } else {
        unrolled_seq = SeqStmt(unrolled_stmts);
      }

      // Preserve the trip count while the two synthetic iterations expose
      // loop-carried dependencies to the linear access analysis.
      return AttrStmt(op->extent, "unrolled_loop", StringImm(loop_id),
                      unrolled_seq);
    }

    Stmt VisitStmt_(const SeqStmtNode *op) override {
      std::vector<Stmt> new_stmts;
      for (const Stmt &stmt : op->seq) {
        new_stmts.push_back(VisitStmt(stmt));
      }
      if (new_stmts.empty()) {
        return Evaluate(0);
      }
      return SeqStmt(new_stmts);
    }

    Stmt VisitStmt_(const AttrStmtNode *op) override {
      Stmt new_body = VisitStmt(op->body);
      return AttrStmt(op->node, op->attr_key, op->value, new_body);
    }

    Stmt VisitStmt_(const LetStmtNode *op) override {
      Stmt new_body = VisitStmt(op->body);
      return LetStmt(op->var, op->value, new_body);
    }

  private:
    std::vector<LoopInfo> loop_infos_;
    int current_depth_ = 0;
  };

  class LoopRebuilder : public StmtMutator {
  public:
    LoopRebuilder(const std::vector<LoopInfo> &loop_infos)
        : loop_infos_(loop_infos) {}

    Stmt operator()(const Stmt &stmt) { return VisitStmt(stmt); }

    Stmt VisitStmt_(const AttrStmtNode *op) override {
      if (op->attr_key == "unrolled_loop") {
        auto marker_name = op->value.as<StringImmNode>();
        if (marker_name) {
          std::string marker = marker_name->value;

          const LoopInfo *target_info = nullptr;
          for (const auto &info : loop_infos_) {
            if (info.loop_id == marker) {
              target_info = &info;
              break;
            }
          }

          if (target_info) {
            Stmt processed_body = VisitStmt(op->body);

            Stmt merged_body =
                MergeIterations(processed_body, target_info->loop_id);
            return For(target_info->loop_var, target_info->min,
                       target_info->extent, target_info->kind, merged_body,
                       NullOpt, target_info->annotations);
          }
        }
      }

      Stmt new_body = VisitStmt(op->body);
      return AttrStmt(op->node, op->attr_key, op->value, new_body);
    }

    Stmt VisitStmt_(const SeqStmtNode *op) override {
      std::vector<Stmt> new_stmts;
      for (const Stmt &stmt : op->seq) {
        new_stmts.push_back(VisitStmt(stmt));
      }
      if (new_stmts.empty()) {
        return Evaluate(0);
      }
      return SeqStmt(new_stmts);
    }

    Stmt VisitStmt_(const IfThenElseNode *op) override {
      Stmt then_case = VisitStmt(op->then_case);

      Optional<Stmt> else_case;
      if (op->else_case.defined()) {
        else_case = VisitStmt(op->else_case.value());
      }
      return IfThenElse(op->condition, then_case, else_case);
    }

  private:
    std::vector<LoopInfo> loop_infos_;

    Stmt MergeIterations(const Stmt &unrolled_body,
                         const std::string &loop_id) {
      std::vector<Stmt> all_stmts = FlattenStmts(unrolled_body);
      if (all_stmts.empty()) {
        return Evaluate(0);
      }

      std::vector<Stmt> iter1_stmts, iter2_stmts;
      bool in_iter1 = false;
      bool in_iter2 = false;
      std::string current_iter;

      for (const auto &stmt : all_stmts) {
        if (IsEmptyEvaluate(stmt)) {
          continue;
        }

        if (IsIterationStartMarker(stmt, loop_id + "_iter1")) {
          in_iter1 = true;
          in_iter2 = false;
          current_iter = "iter1";
          continue;
        } else if (IsIterationEndMarker(stmt, loop_id + "_iter1")) {
          in_iter1 = false;
          current_iter = "";
          continue;
        } else if (IsIterationStartMarker(stmt, loop_id + "_iter2")) {
          in_iter2 = true;
          in_iter1 = false;
          current_iter = "iter2";
          continue;
        } else if (IsIterationEndMarker(stmt, loop_id + "_iter2")) {
          in_iter2 = false;
          current_iter = "";
          continue;
        }

        if (IsUnrolledLoopMarker(stmt)) {
          continue;
        }

        if (in_iter1) {
          iter1_stmts.push_back(stmt);
        } else if (in_iter2) {
          iter2_stmts.push_back(stmt);
        }
      }

      if (iter1_stmts.empty() && iter2_stmts.empty()) {
        return Evaluate(0);
      }

      std::vector<Stmt> merged_stmts =
          MergeStatementSequences(iter1_stmts, iter2_stmts, loop_id);

      if (merged_stmts.empty()) {
        return Evaluate(0);
      } else if (merged_stmts.size() == 1) {
        return merged_stmts[0];
      } else {
        return SeqStmt(merged_stmts);
      }
    }

    bool IsIterationStartMarker(const Stmt &stmt, const std::string &marker) {
      if (auto attr = stmt.as<AttrStmtNode>()) {
        if (attr->attr_key == "iteration_start") {
          auto value = attr->value.as<StringImmNode>();
          if (value && value->value == marker) {
            return true;
          }
        }
      }
      return false;
    }

    bool IsEmptyEvaluate(const Stmt &stmt) {
      if (auto eval = stmt.as<EvaluateNode>()) {
        if (auto int_imm = eval->value.as<IntImmNode>()) {
          if (int_imm->value == 0) {
            return true;
          }
        }
        if (auto float_imm = eval->value.as<FloatImmNode>()) {
          if (float_imm->value == 0.0) {
            return true;
          }
        }
      }
      return false;
    }

    bool IsIterationEndMarker(const Stmt &stmt, const std::string &marker) {
      if (auto attr = stmt.as<AttrStmtNode>()) {
        if (attr->attr_key == "iteration_end") {
          auto value = attr->value.as<StringImmNode>();
          if (value && value->value == marker) {
            return true;
          }
        }
      }
      return false;
    }

    bool IsUnrolledLoopMarker(const Stmt &stmt) {
      if (auto attr = stmt.as<AttrStmtNode>()) {
        if (attr->attr_key == "unrolled_loop") {
          return true;
        }
      }
      return false;
    }

    std::vector<Stmt>
    MergeStatementSequences(const std::vector<Stmt> &iter1_stmts,
                            const std::vector<Stmt> &iter2_stmts,
                            const std::string &loop_id) {
      struct Sequence {
        std::vector<Stmt> operations;
        std::vector<std::vector<Stmt>> syncs{1};
      };
      auto split = [&](const std::vector<Stmt> &stmts) {
        Sequence result;
        for (const auto &stmt : stmts) {
          if (IsSyncStatement(stmt)) {
            result.syncs.back().push_back(stmt);
          } else if (!IsMarkerStatement(stmt) && !IsEmptyEvaluate(stmt)) {
            result.operations.push_back(stmt);
            result.syncs.emplace_back();
          }
        }
        return result;
      };
      auto first = split(iter1_stmts);
      auto second = split(iter2_stmts);
      ICHECK_EQ(first.operations.size(), second.operations.size())
          << "Loop synchronization changed the operation count in " << loop_id;

      std::vector<Stmt> merged;
      for (size_t i = 0; i <= first.operations.size(); ++i) {
        auto syncs = MergeSyncSequences(first.syncs[i], second.syncs[i]);
        merged.insert(merged.end(), syncs.begin(), syncs.end());
        if (i < first.operations.size()) {
          merged.push_back(MergeCorrespondingStatements(
              first.operations[i], second.operations[i], loop_id));
        }
      }
      return merged;
    }

    Stmt MergeBodies(const Stmt &first, const Stmt &second,
                     const std::string &loop_id) {
      auto merged = MergeStatementSequences(FlattenStmts(first),
                                            FlattenStmts(second), loop_id);
      if (merged.empty()) {
        return Evaluate(0);
      }
      return merged.size() == 1 ? merged[0] : SeqStmt(merged);
    }

    template <typename T>
    Stmt MergeBodyStatement(const Stmt &first, const Stmt &second,
                            const std::string &loop_id) {
      T left = Downcast<T>(first);
      T right = Downcast<T>(second);
      T metadata = right;
      metadata.CopyOnWrite()->body = left->body;
      ICHECK(StructuralEqual()(left, metadata));
      T result = left;
      result.CopyOnWrite()->body =
          MergeBodies(left->body, right->body, loop_id);
      return result;
    }

    Stmt MergeCorrespondingStatements(const Stmt &first, const Stmt &second,
                                      const std::string &loop_id) {
      ICHECK_EQ(first->type_index(), second->type_index());
      // Both copies originate from the same execution statement. Preserve its
      // metadata, but merge nested bodies instead of discarding the second
      // copy's loop-carried synchronization.
      if (first.as<ForNode>()) {
        return MergeBodyStatement<For>(first, second, loop_id);
      }
      if (const auto *op = first.as<IfThenElseNode>()) {
        const auto *other = second.as<IfThenElseNode>();
        IfThenElse metadata = Downcast<IfThenElse>(second);
        metadata.CopyOnWrite()->then_case = op->then_case;
        metadata.CopyOnWrite()->else_case = op->else_case;
        ICHECK(StructuralEqual()(first, metadata));
        IfThenElse result = Downcast<IfThenElse>(first);
        auto *node = result.CopyOnWrite();
        node->then_case = MergeBodies(op->then_case, other->then_case, loop_id);
        ICHECK_EQ(op->else_case.defined(), other->else_case.defined());
        if (op->else_case.defined()) {
          node->else_case = MergeBodies(op->else_case.value(),
                                        other->else_case.value(), loop_id);
        }
        return result;
      }
      if (first.as<LetStmtNode>()) {
        return MergeBodyStatement<LetStmt>(first, second, loop_id);
      }
      if (first.as<AttrStmtNode>()) {
        return MergeBodyStatement<AttrStmt>(first, second, loop_id);
      }
      if (first.as<AllocateNode>()) {
        return MergeBodyStatement<Allocate>(first, second, loop_id);
      }
      if (first.as<AllocateConstNode>()) {
        return MergeBodyStatement<AllocateConst>(first, second, loop_id);
      }
      if (first.as<DeclBufferNode>()) {
        return MergeBodyStatement<DeclBuffer>(first, second, loop_id);
      }
      if (first.as<AssertStmtNode>()) {
        return MergeBodyStatement<AssertStmt>(first, second, loop_id);
      }
      if (first.as<BufferRealizeNode>()) {
        return MergeBodyStatement<BufferRealize>(first, second, loop_id);
      }
      if (first.as<WhileNode>()) {
        return MergeBodyStatement<While>(first, second, loop_id);
      }
      if (const auto *op = first.as<BlockRealizeNode>()) {
        BlockRealize metadata = Downcast<BlockRealize>(second);
        metadata.CopyOnWrite()->block = op->block;
        ICHECK(StructuralEqual()(first, metadata));
        BlockRealize result = Downcast<BlockRealize>(first);
        result.CopyOnWrite()->block =
            Downcast<Block>(MergeCorrespondingStatements(
                op->block, second.as<BlockRealizeNode>()->block, loop_id));
        return result;
      }
      if (const auto *op = first.as<BlockNode>()) {
        const auto *other = second.as<BlockNode>();
        Block metadata = Downcast<Block>(second);
        metadata.CopyOnWrite()->body = op->body;
        metadata.CopyOnWrite()->init = op->init;
        ICHECK(StructuralEqual()(first, metadata));
        Block result = Downcast<Block>(first);
        auto *node = result.CopyOnWrite();
        node->body = MergeBodies(op->body, other->body, loop_id);
        ICHECK_EQ(op->init.defined(), other->init.defined());
        if (op->init.defined()) {
          node->init =
              MergeBodies(op->init.value(), other->init.value(), loop_id);
        }
        return result;
      }
      ICHECK(StructuralEqual()(first, second))
          << "Loop synchronization changed an execution statement in "
          << loop_id;
      return first;
    }

    bool IsMatchingEventPair(const Stmt &set, const Stmt &wait) {
      const auto *set_eval = set.as<EvaluateNode>();
      const auto *wait_eval = wait.as<EvaluateNode>();
      if (!set_eval || !wait_eval) {
        return false;
      }
      const auto *set_call = set_eval->value.as<CallNode>();
      const auto *wait_call = wait_eval->value.as<CallNode>();
      if (!set_call || !wait_call) {
        return false;
      }
      size_t offset = 0;
      if (set_call->op.same_as(builtin::call_extern()) &&
          wait_call->op.same_as(builtin::call_extern())) {
        const auto *set_name = set_call->args[0].as<StringImmNode>();
        const auto *wait_name = wait_call->args[0].as<StringImmNode>();
        if (!set_name || !wait_name ||
            std::string(set_name->value).find("AutoSetFlag") ==
                std::string::npos ||
            std::string(wait_name->value).find("AutoWaitFlag") ==
                std::string::npos) {
          return false;
        }
        offset = 1;
      } else if (!set_call->op.same_as(tl::ascend_auto_set_flag()) ||
                 !wait_call->op.same_as(tl::ascend_auto_wait_flag())) {
        return false;
      }
      return set_call->args.size() == offset + 2 &&
             wait_call->args.size() == offset + 2 &&
             StructuralEqual()(set_call->args[offset],
                               wait_call->args[offset]) &&
             StructuralEqual()(set_call->args[offset + 1],
                               wait_call->args[offset + 1]);
    }

    std::vector<Stmt> MergeSyncSequences(const std::vector<Stmt> &first,
                                         const std::vector<Stmt> &second) {
      using SyncUnit = std::vector<Stmt>;
      auto group = [&](const std::vector<Stmt> &stmts) {
        std::vector<SyncUnit> result;
        for (size_t i = 0; i < stmts.size(); ++i) {
          if (i + 1 < stmts.size() &&
              IsMatchingEventPair(stmts[i], stmts[i + 1])) {
            result.push_back({stmts[i], stmts[i + 1]});
            ++i;
          } else {
            result.push_back({stmts[i]});
          }
        }
        return result;
      };
      auto left = group(first);
      auto right = group(second);
      auto same = [&](const SyncUnit &a, const SyncUnit &b) {
        if (a.size() != b.size()) {
          return false;
        }
        // Renumbered event pairs are equivalent only as whole pairs. A lone
        // event must keep its identity, so it cannot be matched by pipe alone.
        return a.size() == 2 ? IsSameSyncOperation(a[0], b[0]) &&
                                   IsSameSyncOperation(a[1], b[1])
                             : StructuralEqual()(a[0], b[0]);
      };

      // Preserve each analysis's event order. A set union can turn [B->C,A->B]
      // and [A->B,B->C] into just [B->C,A->B], losing the second ordering.
      std::vector<std::vector<size_t>> common(
          left.size() + 1, std::vector<size_t>(right.size() + 1));
      for (size_t i = left.size(); i-- > 0;) {
        for (size_t j = right.size(); j-- > 0;) {
          common[i][j] = same(left[i], right[j])
                             ? 1 + common[i + 1][j + 1]
                             : std::max(common[i + 1][j], common[i][j + 1]);
        }
      }
      std::vector<Stmt> merged;
      size_t i = 0, j = 0;
      while (i < left.size() || j < right.size()) {
        const SyncUnit *unit;
        if (i < left.size() && j < right.size() && same(left[i], right[j])) {
          unit = &left[i++];
          ++j;
        } else if (j == right.size() ||
                   (i < left.size() && common[i + 1][j] >= common[i][j + 1])) {
          unit = &left[i++];
        } else {
          unit = &right[j++];
        }
        merged.insert(merged.end(), unit->begin(), unit->end());
      }
      return merged;
    }

    bool IsMarkerStatement(const Stmt &stmt) {
      if (auto attr = stmt.as<AttrStmtNode>()) {
        return (attr->attr_key == "iteration_start" ||
                attr->attr_key == "iteration_end" ||
                attr->attr_key == "unrolled_loop");
      }
      return false;
    }

    bool IsSyncStatement(const Stmt &stmt) {
      if (auto eval = stmt.as<EvaluateNode>()) {
        if (auto call = eval->value.as<CallNode>()) {
          if (call->op.same_as(builtin::call_extern())) {
            auto func_name_imm = call->args[0].as<StringImmNode>();
            if (func_name_imm) {
              std::string func_name = func_name_imm->value;
              return ((func_name.find("AutoBarrier") != std::string::npos ||
                       func_name.find("AutoSetFlag") != std::string::npos ||
                       func_name.find("AutoWaitFlag") != std::string::npos));
            }
          } else if (call->op.same_as(tl::ascend_auto_barrier()) ||
                     call->op.same_as(tl::ascend_auto_set_flag()) ||
                     call->op.same_as(tl::ascend_auto_wait_flag())) {
            return true;
          }
        }
      }
      return false;
    }

    bool IsSameSyncOperation(const Stmt &stmt1, const Stmt &stmt2) {
      if (!IsSyncStatement(stmt1) || !IsSyncStatement(stmt2)) {
        return false;
      }

      auto eval1 = stmt1.as<EvaluateNode>();
      auto eval2 = stmt2.as<EvaluateNode>();
      if (!eval1 || !eval2) {
        return false;
      }

      auto call1 = eval1->value.as<CallNode>();
      auto call2 = eval2->value.as<CallNode>();
      if (!call1 || !call2) {
        return false;
      }

      if (call1->op.same_as(builtin::call_extern()) &&
          call2->op.same_as(builtin::call_extern())) {
        auto func_name1 = call1->args[0].as<StringImmNode>();
        auto func_name2 = call2->args[0].as<StringImmNode>();
        if (!func_name1 || !func_name2) {
          return false;
        }

        std::string name1 = func_name1->value;
        std::string name2 = func_name2->value;

        if (name1 != name2) {
          return false;
        }

        if (name1.find("AutoBarrier") != std::string::npos) {
          if (call1->args.size() >= 2 && call2->args.size() >= 2) {
            auto pipeline1 = call1->args[1].as<StringImmNode>();
            auto pipeline2 = call2->args[1].as<StringImmNode>();
            if (pipeline1 && pipeline2) {
              return pipeline1->value == pipeline2->value;
            }
          }
          return false;
        }

        if (name1.find("AutoSetFlag") != std::string::npos ||
            name1.find("AutoWaitFlag") != std::string::npos) {
          if (call1->args.size() >= 3 && call2->args.size() >= 3) {
            auto event_type1 = call1->args[1].as<StringImmNode>();
            auto event_type2 = call2->args[1].as<StringImmNode>();
            if (event_type1 && event_type2) {
              return event_type1->value == event_type2->value;
            }
          }
          return false;
        }

        return StructuralEqual()(stmt1, stmt2);
      } else if (call1->op.same_as(call2->op)) {
        // call_intrin 判断
        // std::string op_name;
        // if (auto* op_ptr = call1->op.as<OpNode>()) {
        //     op_name = op_ptr->name;
        //     auto config_it = operation_config_.find(op_name);
        //     if (config_it == operation_config_.end()){
        //         return false;
        //     }
        // }

        if (call1->op.same_as(tl::ascend_auto_barrier())) {
          if (call1->args.size() >= 1 && call2->args.size() >= 1) {
            auto pipeline1 = call1->args[0].as<StringImmNode>();
            auto pipeline2 = call2->args[0].as<StringImmNode>();
            if (pipeline1 && pipeline2) {
              return pipeline1->value == pipeline2->value;
            }
          }
          return false;
        }

        if (call1->op.same_as(tl::ascend_auto_set_flag()) ||
            call1->op.same_as(tl::ascend_auto_wait_flag())) {
          if (call1->args.size() >= 2 && call2->args.size() >= 2) {
            auto event_type1 = call1->args[0].as<StringImmNode>();
            auto event_type2 = call2->args[0].as<StringImmNode>();
            if (event_type1 && event_type2) {
              return event_type1->value == event_type2->value;
            }
          }
          return false;
        }
        return StructuralEqual()(stmt1, stmt2);
      }

      return false;
    }

    std::vector<Stmt> FlattenStmts(const Stmt &stmt) {
      std::vector<Stmt> result;
      StmtFlattener flattener(result);
      flattener(stmt);
      return result;
    }

    class StmtFlattener : public StmtVisitor {
    public:
      StmtFlattener(std::vector<Stmt> &result) : result_(result) {}

      void VisitStmt(const Stmt &stmt) final {
        if (const auto *seq = stmt.as<SeqStmtNode>()) {
          for (const Stmt &child : seq->seq) {
            VisitStmt(child);
          }
          return;
        }
        if (const auto *attr = stmt.as<AttrStmtNode>()) {
          if (attr->attr_key == "iteration_start" ||
              attr->attr_key == "iteration_end") {
            result_.push_back(stmt);
            VisitStmt(attr->body);
            return;
          }
          if (attr->attr_key == "unrolled_loop") {
            VisitStmt(attr->body);
            return;
          }
        }
        if (stmt.as<BufferStoreNode>()) {
          Stmt barrier = Evaluate(Call(DataType::Handle(),
                                       Op::Get("tl.ascend_auto_barrier"),
                                       {StringImm("PIPE_ALL")}));
          if (result_.empty() || !StructuralEqual()(result_.back(), barrier)) {
            result_.push_back(barrier);
          }
        }
        // Compound execution statements remain intact here; their bodies are
        // merged recursively at the matching execution position.
        result_.push_back(stmt);
      }

    private:
      std::vector<Stmt> &result_;
    };
  };

  struct SyncGraph {
    std::unordered_map<std::string, std::unordered_set<std::string>> graph;

    std::string toString() const {
      std::ostringstream oss;
      oss << "SyncGraph{graph: {";
      bool first_pair = true;
      for (const auto &pair : graph) {
        if (!first_pair)
          oss << ", ";
        oss << "'" << pair.first << "': [";
        bool first_dst = true;
        for (const auto &dst : pair.second) {
          if (!first_dst)
            oss << ", ";
          oss << "'" << dst << "'";
          first_dst = false;
        }
        oss << "]";
        first_pair = false;
      }
      oss << "}}";
      return oss.str();
    }

    void AddSync(const std::string &sync_type) {
      if (sync_type.find("EventPair_") == 0) {
        std::string event = sync_type.substr(10);
        size_t pos = event.find('_');
        if (pos != std::string::npos) {
          std::string src = event.substr(0, pos);
          std::string dst = event.substr(pos + 1);
          graph[src].insert(dst);
        }
      }
    }

    bool HasPath(const std::string &src, const std::string &dst) const {
      if (src == dst)
        return true;

      std::unordered_set<std::string> visited;
      std::vector<std::string> queue = {src};
      visited.insert(src);

      while (!queue.empty()) {
        std::string current = queue.back();
        queue.pop_back();

        auto it = graph.find(current);
        if (it != graph.end()) {
          for (const auto &neighbor : it->second) {
            if (neighbor == dst)
              return true;
            if (visited.count(neighbor) == 0) {
              visited.insert(neighbor);
              queue.push_back(neighbor);
            }
          }
        }
      }

      return false;
    }

    SyncGraph IntersectCompletion(const SyncGraph &other,
                                  const std::string &producer) const {
      SyncGraph common;
      for (const auto &pair : graph) {
        for (const auto &destination : pair.second) {
          if (destination != producer && HasPath(producer, destination) &&
              other.HasPath(producer, destination)) {
            // The two paths can prove completion via different intermediate
            // pipes. Keep their common destinations, not just common edges.
            common.graph[producer].insert(destination);
          }
        }
      }
      return common;
    }

    void Merge(const SyncGraph &other) {
      for (const auto &pair : other.graph) {
        const std::string &src = pair.first;
        for (const std::string &dst : pair.second) {
          graph[src].insert(dst);
        }
      }
    }

    SyncGraph ComputeTransitiveClosure() const {
      SyncGraph closure;
      closure.graph = graph;

      std::unordered_set<std::string> nodes;
      for (const auto &pair : graph) {
        nodes.insert(pair.first);
        for (const auto &dst : pair.second) {
          nodes.insert(dst);
        }
      }

      std::vector<std::string> node_list(nodes.begin(), nodes.end());

      for (const auto &k : node_list) {
        for (const auto &i : node_list) {
          for (const auto &j : node_list) {
            if (closure.HasPath(i, k) && closure.HasPath(k, j)) {
              closure.graph[i].insert(j);
            }
          }
        }
      }

      return closure;
    }
  };

  struct BufferAccess {
    std::string buffer_name;
    bool is_write;
    std::string pipeline;
    std::string operation;
    SyncGraph sync_graph;
    std::set<std::string> pipe_barriers;
    int64_t physical_address;
    bool is_sliced; // 新增：切片操作标记

    std::string toString() const {
      std::ostringstream oss;
      oss << "BufferAccess{";
      oss << "buffer_name: '" << buffer_name << "', ";
      oss << "is_write: " << (is_write ? "true" : "false") << ", ";
      oss << "pipeline: '" << pipeline << "', ";
      oss << "operation: '" << operation << "', ";
      oss << "physical_address: " << physical_address << ", ";
      oss << "is_sliced: " << (is_sliced ? "true" : "false");
      oss << "sync_graph: " << sync_graph.toString() << ", ";
      oss << "pipe_barriers: [";
      bool first_barrier = true;
      for (const auto &barrier : pipe_barriers) {
        if (!first_barrier)
          oss << ", ";
        oss << "'" << barrier << "'";
        first_barrier = false;
      }
      oss << "]";
      oss << "}";
      return oss.str();
    }
  };

  using AccessHistory =
      std::unordered_map<std::string, std::vector<BufferAccess>>;

  struct LoopExitState {
    std::string loop_id;
    AccessHistory first_iteration_history;
    bool first_iteration_seen;
  };

  struct SyncRequirement {
    std::string sync_type;
    std::string buffer_name;

    std::string toString() const {
      std::ostringstream oss;
      oss << "SyncRequirement{";
      oss << "sync_type: '" << sync_type << "', ";
      oss << "buffer_name: '" << buffer_name << "'";
      oss << "}";
      return oss.str();
    }
  };

  struct BufferInfo {
    std::string buffer_name;
    bool is_read;
    bool is_write;
    bool is_sliced;

    std::string toString() const {
      std::ostringstream oss;
      oss << "BufferInfo{";
      oss << "buffer_name: '" << buffer_name << "', ";
      oss << "is_sliced: " << (is_sliced ? "true" : "false");
      oss << "is_read: " << (is_read ? "true" : "false") << ", ";
      oss << "is_write: " << (is_write ? "true" : "false");
      oss << "}";
      return oss.str();
    }
  };

  std::vector<BufferAccess> AnalyzeExprAccesses(const PrimExpr &expr) {
    std::vector<BufferAccess> accesses;

    ExprAccessAnalyzer analyzer;
    analyzer(expr);

    for (const auto &buffer_name : analyzer.GetAccessedBuffers()) {
      BufferAccess access;
      access.buffer_name = buffer_name;
      access.is_write = false;
      access.pipeline = "UNKNOWN";
      access.operation = "expression";
      access.sync_graph = SyncGraph();
      access.pipe_barriers = {};
      access.physical_address = GetPhysicalAddress(buffer_name);
      access.is_sliced = analyzer.IsBufferSliced(buffer_name);

      accesses.push_back(access);
    }

    return accesses;
  }

  class ExprAccessAnalyzer : public ExprVisitor {
  public:
    void VisitExpr_(const CallNode *op) override {
      if (op->op.same_as(builtin::tvm_access_ptr())) {
        if (op->args.size() >= 5) {
          if (auto var = op->args[1].as<VarNode>()) {
            std::string buffer_name = var->name_hint;
            accessed_buffers_.insert(buffer_name);

            if (auto offset = op->args[2].as<IntImmNode>()) {
              if (offset->value != 0) {
                sliced_buffers_.insert(buffer_name);
              }
            } else {
              sliced_buffers_.insert(buffer_name);
            }
          }
        }
      }
      ExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const BufferLoadNode *op) override {
      std::string buffer_name = op->buffer->data->name_hint;
      accessed_buffers_.insert(buffer_name);

      sliced_buffers_.insert(buffer_name);

      for (const auto &index : op->indices) {
        VisitExpr(index);
      }

      ExprVisitor::VisitExpr_(op);
    }

    std::unordered_set<std::string> GetAccessedBuffers() const {
      return accessed_buffers_;
    }

    bool IsBufferSliced(const std::string &buffer_name) const {
      return sliced_buffers_.count(buffer_name) > 0;
    }

  private:
    std::unordered_set<std::string> accessed_buffers_;
    std::unordered_set<std::string> sliced_buffers_;
  };

  template <typename T>
  std::string containerToString(const std::vector<T> &vec) {
    std::ostringstream oss;
    oss << "[";
    bool first = true;
    for (const auto &item : vec) {
      if (!first)
        oss << ", ";
      oss << item.toString();
      first = false;
    }
    oss << "]";
    return oss.str();
  }

  std::string containerToString(const std::vector<std::string> &vec) {
    std::ostringstream oss;
    oss << "[";
    bool first = true;
    for (const auto &item : vec) {
      if (!first)
        oss << ", ";
      oss << item;
      first = false;
    }
    oss << "]";
    return oss.str();
  }

  std::string
  containerToString(const std::unordered_map<std::string, BufferAccess> &map) {
    std::ostringstream oss;
    oss << "{";
    bool first = true;
    for (const auto &pair : map) {
      if (!first)
        oss << ", ";
      oss << "'" << pair.first << "': " << pair.second.toString();
      first = false;
    }
    oss << "}";
    return oss.str();
  }

  std::vector<BufferAccess> AnalyzeStmtAccesses(const Stmt &stmt) {
    std::vector<BufferAccess> accesses;

    if (auto eval = stmt.as<EvaluateNode>()) {
      if (auto call = eval->value.as<CallNode>()) {
        if (call->op.same_as(builtin::call_extern())) {
          std::string func_name = Downcast<StringImm>(call->args[0])->value;

          std::string normalized_name = NormalizeFunctionName(func_name);
          auto config_it = operation_config_.find(normalized_name);
          if (config_it != operation_config_.end()) {
            const auto &config = config_it->second;

            std::unordered_map<std::string, BufferAccess> buffer_access_map;

            for (const auto &buffer_config : config.buffer_accesses) {
              size_t arg_index = buffer_config.first;
              const std::string &access_type = buffer_config.second;

              if (arg_index + 1 < call->args.size()) {
                auto buffer_info =
                    ExtractBufferInfoFromAccessPtr(call->args[arg_index + 1]);
                if (!buffer_info.buffer_name.empty()) {
                  bool is_write = (access_type == "write");

                  if (buffer_access_map.find(buffer_info.buffer_name) !=
                      buffer_access_map.end()) {
                    BufferAccess &existing_access =
                        buffer_access_map[buffer_info.buffer_name];
                    if (is_write || (!existing_access.is_write && is_write)) {
                      existing_access.is_write = true;
                    }
                    existing_access.is_sliced =
                        existing_access.is_sliced || buffer_info.is_sliced;
                  } else {
                    BufferAccess access;
                    access.sync_graph = SyncGraph();
                    access.pipe_barriers = {};
                    access.physical_address =
                        GetPhysicalAddress(buffer_info.buffer_name);

                    access.buffer_name = buffer_info.buffer_name;
                    access.is_write = is_write;
                    access.pipeline = config.default_pipeline;
                    access.operation = normalized_name;
                    access.is_sliced = buffer_info.is_sliced;

                    buffer_access_map[buffer_info.buffer_name] = access;
                  }
                }
              }
            }

            for (const auto &pair : buffer_access_map) {
              accesses.push_back(pair.second);
            }
          }
        } else { //
          std::string op_name;
          if (auto *op_ptr = call->op.as<OpNode>()) {
            op_name = op_ptr->name;

            std::string normalized_name = op_name;

            auto config_it = operation_config_.find(normalized_name);
            if (config_it != operation_config_.end()) {
              const OperationConfig config = ResolveOperationConfig(call);

              std::unordered_map<std::string, BufferAccess> buffer_access_map;

              for (const auto &buffer_config : config.buffer_accesses) {
                size_t arg_index = buffer_config.first;
                const std::string &access_type = buffer_config.second;

                if (arg_index < call->args.size()) {
                  auto buffer_info =
                      ExtractBufferInfoFromAccessPtr(call->args[arg_index]);
                  if (!buffer_info.buffer_name.empty()) {
                    bool is_write = (access_type == "write");

                    if (buffer_access_map.find(buffer_info.buffer_name) !=
                        buffer_access_map.end()) {
                      BufferAccess &existing_access =
                          buffer_access_map[buffer_info.buffer_name];
                      if (is_write || (!existing_access.is_write && is_write)) {
                        existing_access.is_write = true;
                      }
                      existing_access.is_sliced =
                          existing_access.is_sliced || buffer_info.is_sliced;
                    } else {
                      BufferAccess access;
                      access.sync_graph = SyncGraph();
                      access.pipe_barriers = {};
                      access.physical_address =
                          GetPhysicalAddress(buffer_info.buffer_name);

                      access.buffer_name = buffer_info.buffer_name;
                      access.is_write = is_write;
                      access.pipeline = config.default_pipeline;
                      access.operation = normalized_name;
                      access.is_sliced = buffer_info.is_sliced;

                      buffer_access_map[buffer_info.buffer_name] = access;
                    }
                  }
                }
              }

              for (const auto &pair : buffer_access_map) {
                accesses.push_back(pair.second);
              }
            }
          }
        }
      }
    }

    return accesses;
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

  BufferInfo ExtractBufferInfoFromAccessPtr(const PrimExpr &expr) {
    BufferInfo info = {"", false, false};

    if (auto call = expr.as<CallNode>()) {
      if (call->op.same_as(builtin::tvm_access_ptr())) {
        if (call->args.size() >= 5) {
          if (auto var = call->args[1].as<VarNode>()) {
            info.buffer_name = var->name_hint;
          }

          ExprAccessAnalyzer analyzer;
          analyzer(call->args[2]);
          for (const auto &buffer_name : analyzer.GetAccessedBuffers()) {
            if (analyzer.IsBufferSliced(buffer_name)) {
              info.is_sliced = true;
            }
          }

          if (auto access_mask = call->args[4].as<IntImmNode>()) {
            int mask = access_mask->value;
            info.is_read = (mask & 1) != 0;
            info.is_write = (mask & 2) != 0;
          }
        }
      } else {
        if (call->args.size() >= 2) {
          if (auto var = call->args[1].as<VarNode>()) {
            info.buffer_name = var->name_hint;
            info.is_read = true;
            info.is_write = true;
          }
        }
      }
    }

    return info;
  }

  bool HasDataDependency(const BufferAccess &prev, const BufferAccess &curr) {
    bool shares_memory = false;

    if (prev.buffer_name == curr.buffer_name) {
      shares_memory = true;
    } else if (prev.physical_address != -1 && curr.physical_address != -1) {
      int64_t prev_size = GetBufferSize(prev.buffer_name);
      int64_t curr_size = GetBufferSize(curr.buffer_name);
      if (prev_size > 0 && curr_size > 0) {
        int64_t prev_end = prev.physical_address + prev_size;
        int64_t curr_end = curr.physical_address + curr_size;
        shares_memory = (prev.physical_address < curr_end &&
                         curr.physical_address < prev_end);
      } else {
        shares_memory = (prev.physical_address == curr.physical_address);
      }
    }

    if (shares_memory) {
      if ((prev.is_write && curr.is_write) ||  // WAW
          (prev.is_write && !curr.is_write) || // RAW
          (!prev.is_write && curr.is_write)) { // WAR
        return true;
      }
    }
    return false;
  }

  void
  UpdateLatestAccessHistory(const std::vector<BufferAccess> &current_accesses) {
    for (const auto &access : current_accesses) {
      auto &history = current_access_history_[access.buffer_name];
      if (access.is_write) {
        // The dependency checks ordered this write after the old accesses.
        // Future users must wait for this new writer instead.
        history.clear();
      } else {
        // Reads on different pipes can remain outstanding together. Retain
        // the writer as well: a later read does not supersede its RAW edges.
        history.erase(std::remove_if(history.begin(), history.end(),
                                     [&](const BufferAccess &previous) {
                                       return !previous.is_write &&
                                              previous.pipeline ==
                                                  access.pipeline;
                                     }),
                      history.end());
      }
      history.push_back(access);
    }
  }

  void JoinAccessHistories(const AccessHistory &entry_history) {
    AccessHistory joined;
    auto merge_path = [&](const AccessHistory &path) {
      for (const auto &pair : path) {
        auto &accesses = joined[pair.first];
        for (const auto &access : pair.second) {
          auto previous = std::find_if(
              accesses.begin(), accesses.end(), [&](const BufferAccess &other) {
                return other.pipeline == access.pipeline &&
                       other.is_write == access.is_write;
              });
          if (previous == accesses.end()) {
            // This access may exist only on the entry or the executed path.
            accesses.push_back(access);
            continue;
          }
          previous->sync_graph = previous->sync_graph.IntersectCompletion(
              access.sync_graph, access.pipeline.substr(5));
          for (auto barrier = previous->pipe_barriers.begin();
               barrier != previous->pipe_barriers.end();) {
            if (!access.pipe_barriers.count(*barrier)) {
              barrier = previous->pipe_barriers.erase(barrier);
            } else {
              ++barrier;
            }
          }
          previous->is_sliced |= access.is_sliced;
        }
      }
    };
    // Keep every possibly outstanding access, but only completion facts true
    // on both paths for a shared access class. Joining by pipe and read/write
    // kind bounds the state even across consecutive or nested optional loops.
    merge_path(entry_history);
    merge_path(current_access_history_);
    current_access_history_ = std::move(joined);
  }

  std::vector<SyncRequirement>
  CollectSyncRequirements(const std::vector<BufferAccess> &accesses) {
    std::vector<SyncRequirement> requirements;
    for (const auto &access : accesses) {
      if (access.is_sliced) {
        requirements.push_back({"PipeBarrier_ALL", access.buffer_name});
      }
      for (const auto &name : FindRelatedBuffers(access.buffer_name)) {
        auto it = current_access_history_.find(name);
        if (it == current_access_history_.end()) {
          continue;
        }
        for (const auto &previous : it->second) {
          if (HasDataDependency(previous, access)) {
            auto sync = GetRequiredSyncType(previous, access);
            if (!sync.empty()) {
              requirements.push_back({sync, access.buffer_name});
            }
          }
        }
      }
    }
    return requirements;
  }

  std::string GetRequiredSyncType(const BufferAccess &prev_access,
                                  const BufferAccess &curr_access) {
    if (prev_access.pipeline == curr_access.pipeline) {
      if (prev_access.pipeline != "PIPE_S" &&
          !prev_access.pipe_barriers.count("PipeBarrier_" +
                                           prev_access.pipeline)) {
        return "PipeBarrier_" + prev_access.pipeline;
      }
      return "";
    }
    std::string event_type =
        GetEventType(prev_access.pipeline, curr_access.pipeline);
    // Consult this producer's history, not the current buffer's history:
    // another alias may carry synchronization from an older generation.
    if (event_type.empty() ||
        prev_access.sync_graph.HasPath(prev_access.pipeline.substr(5),
                                       curr_access.pipeline.substr(5))) {
      return "";
    }
    return "EventPair_" + event_type;
  }

  std::string GetEventType(const std::string &src_pipeline,
                           const std::string &dst_pipeline) {
    std::string key = src_pipeline + "_" + dst_pipeline;
    auto it = event_mapping_.find(key);
    return it != event_mapping_.end() ? it->second : "";
  }

  int64_t GetPhysicalAddress(const std::string &buffer_name) {
    for (const auto &pair : address_map_) {
      if (pair.first->name_hint == buffer_name) {
        if (auto int_imm = pair.second.as<IntImmNode>()) {
          return int_imm->value;
        }
      }
    }
    return -1;
  }

  int64_t GetBufferSize(const std::string &buffer_name) {
    for (const auto &pair : size_map_) {
      if (pair.first->name_hint == buffer_name) {
        if (auto int_imm = pair.second.as<IntImmNode>()) {
          return int_imm->value;
        }
      }
    }
    return -1;
  }

  std::string GetBufferScope(const std::string &buffer_name) {
    for (const auto &pair : address_map_) {
      if (pair.first->name_hint == buffer_name) {
        return GetPtrStorageScope(pair.first);
      }
    }
    return "";
  }

  // Issue #1304: collect every GM buffer that receives a scalar store
  // (BufferStore lowers to an S-pipe write through a write-back cache). A
  // later MTE3 DMA write to one of these buffers races the cache and needs
  // the coherence treatment in VisitStmt_(EvaluateNode).
  void CollectScalarWrittenGmBuffers(const Stmt &stmt) {
    class Collector : public StmtVisitor {
    public:
      explicit Collector(AscendSyncInsert *self) : self_(self) {}
      void VisitStmt_(const BufferStoreNode *op) final {
        // The buffer var's pointer type annotation carries the storage scope
        // ("global" for GM tensors, "shared.ub" etc. for on-chip buffers);
        // address_map_ only covers planned UB buffers, so look it up directly.
        if (GetPtrStorageScope(op->buffer->data) == "global") {
          self_->scalar_written_gm_buffers_.insert(op->buffer->data->name_hint);
        }
        StmtVisitor::VisitStmt_(op);
      }

    private:
      AscendSyncInsert *self_;
    };
    Collector collector(this);
    collector(stmt);
  }

  // Find the access_ptr argument of the intrinsic call in an EvaluateNode
  // that references the given buffer.
  PrimExpr FindBufferArgExpr(const EvaluateNode *op,
                             const std::string &buffer_name) {
    if (auto call = op->value.as<CallNode>()) {
      for (const auto &arg : call->args) {
        auto info = ExtractBufferInfoFromAccessPtr(arg);
        if (info.buffer_name == buffer_name) {
          return arg;
        }
      }
    }
    return PrimExpr();
  }

  std::vector<std::string> FindRelatedBuffers(const std::string &buffer_name) {
    std::vector<std::string> related;
    int64_t target_addr = GetPhysicalAddress(buffer_name);

    if (target_addr == -1) {
      related.push_back(buffer_name);
      return related;
    }

    std::string target_scope = GetBufferScope(buffer_name);

    int64_t target_size = GetBufferSize(buffer_name);
    if (target_size <= 0) {
      // No size info, fall back to exact address match
      for (const auto &pair : address_map_) {
        if (auto int_imm = pair.second.as<IntImmNode>()) {
          if (int_imm->value == target_addr &&
              GetPtrStorageScope(pair.first) == target_scope) {
            related.push_back(pair.first->name_hint);
          }
        }
      }
      return related;
    }

    int64_t target_end = target_addr + target_size;

    for (const auto &pair : address_map_) {
      if (auto int_imm = pair.second.as<IntImmNode>()) {
        if (GetPtrStorageScope(pair.first) != target_scope) {
          continue;
        }
        int64_t other_addr = int_imm->value;
        int64_t other_size = GetBufferSize(pair.first->name_hint);
        if (other_size <= 0) {
          // No size info for other buffer, check exact match only
          if (other_addr == target_addr) {
            related.push_back(pair.first->name_hint);
          }
          continue;
        }
        int64_t other_end = other_addr + other_size;
        // Check interval overlap: [target_addr, target_end) ∩ [other_addr,
        // other_end)
        if (target_addr < other_end && other_addr < target_end) {
          related.push_back(pair.first->name_hint);
        }
      }
    }
    return related;
  }

  std::vector<std::string>
  OptimizeSyncRequirements(const std::vector<SyncRequirement> &requirements) {
    std::vector<std::string> all_required_syncs;
    for (const auto &req : requirements) {
      all_required_syncs.push_back(req.sync_type);
    }

    std::sort(all_required_syncs.begin(), all_required_syncs.end());
    all_required_syncs.erase(
        std::unique(all_required_syncs.begin(), all_required_syncs.end()),
        all_required_syncs.end());

    // Elide only handoffs already proven for each access. Treating all
    // simultaneous requirements as an unordered graph can remove a needed
    // edge using a path whose events are emitted in the opposite order.
    return all_required_syncs;
  }

  void
  UpdateSyncStatesAfterSync(const std::vector<std::string> &inserted_syncs) {
    for (const auto &sync_type : inserted_syncs) {
      for (auto &pair : current_access_history_) {
        for (BufferAccess &access : pair.second) {
          if (sync_type.find("EventPair_") == 0) {
            const auto event = sync_type.substr(10);
            const auto separator = event.find('_');
            if (separator != std::string::npos &&
                access.sync_graph.HasPath(access.pipeline.substr(5),
                                          event.substr(0, separator))) {
              // Propagate completion only along events issued after this
              // access, in emitted order. Earlier reverse-order edges cannot
              // later become a valid transitive completion path.
              access.sync_graph.AddSync(sync_type);
            }
          } else if (sync_type.find("PipeBarrier_") == 0) {
            std::string pipeline = sync_type.substr(12);
            if (access.pipeline == pipeline) {
              access.pipe_barriers.insert(sync_type);
            }
          }
        }
      }
    }
  }

  void InsertSynchronization(const std::string &sync_type,
                             std::vector<Stmt> &stmts) {
    if (sync_type == "PipeBarrier_ALL") {
      stmts.push_back(CreatePipeBarrier("PIPE_ALL"));
    } else if (sync_type.find("PipeBarrier_") == 0) {
      std::string pipeline = sync_type.substr(12);
      // A5 AIC dont need PIPE_V
      if (pipeline == "PIPE_V" && this->platform_ == "A5") {
        return;
      }
      stmts.push_back(CreatePipeBarrier(pipeline));
    } else if (sync_type.find("EventPair_") == 0) {
      std::string event_type = sync_type.substr(10);
      int event_id = AllocateEventId();
      stmts.push_back(CreateSetFlag(event_type, event_id));
      stmts.push_back(CreateWaitFlag(event_type, event_id));
    }
  }

  int AllocateEventId() {
    // Cycle 1..7 and reserve id 0 for the C++ templates (copy_gm_to_ub and
    // copy_ub_to_gm in tl_templates/ascend/common.h hard-code event id 0 for
    // their set/wait pairs). Allocating 0 here can interleave a pass-emitted
    // set/wait of the same (pipe pair, id) with a template-emitted one,
    // collapsing two sets into one and deadlocking the second wait on
    // device.
    event_id_counter_ = event_id_counter_ % 7 + 1;
    return event_id_counter_;
  }

  Stmt CreateDcci(const PrimExpr &buffer_ptr) {
    Array<PrimExpr> args = {buffer_ptr};
    return Evaluate(
        Call(DataType::Handle(), Op::Get("tl.ascend_auto_dcci"), args));
  }

  Stmt CreatePipeBarrier(const std::string &pipeline) {
    Array<PrimExpr> args = {StringImm(pipeline)};
    return Evaluate(
        Call(DataType::Handle(), Op::Get("tl.ascend_auto_barrier"), args));
  }

  Stmt CreateSetFlag(const std::string &event_type, int event_id) {
    Array<PrimExpr> args = {StringImm(event_type),
                            IntImm(DataType::Int(32), event_id)};
    return Evaluate(
        Call(DataType::Handle(), Op::Get("tl.ascend_auto_set_flag"), args));
  }

  Stmt CreateWaitFlag(const std::string &event_type, int event_id) {
    Array<PrimExpr> args = {StringImm(event_type),
                            IntImm(DataType::Int(32), event_id)};
    return Evaluate(
        Call(DataType::Handle(), Op::Get("tl.ascend_auto_wait_flag"), args));
  }

private:
  int event_id_counter_ = 0;
  int current_resource_scope_ = -1;
  // Issue #1304: GM buffers that receive scalar stores (S-pipe writes through
  // the write-back cache) anywhere in the kernel. MTE3 DMA writes to these
  // buffers get the cache-coherence treatment (dcci + MTE3_S).
  std::set<std::string> scalar_written_gm_buffers_;
  std::unordered_map<std::string, std::string> event_mapping_;
  std::unordered_map<std::string, OperationConfig> operation_config_;
  // Normally one last writer plus the latest reader on each pipe. An optional
  // loop can retain alternative writers, bounded to one per pipe as well.
  // Physical aliases are joined by FindRelatedBuffers at each access.
  AccessHistory current_access_history_;
  std::vector<LoopExitState> loop_exit_states_;
  Map<Var, PrimExpr> address_map_;
  Map<Var, PrimExpr> size_map_;
  std::string platform_;
  Target target_;
};

tvm::transform::Pass AscendSyncInsert(Target target, std::string platform) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto new_func = AscendSyncInsert::Substitute(std::move(f), "config_path",
                                                 ctx, target, platform);
    return new_func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendSyncInsert", {});
}

TVM_REGISTER_GLOBAL("tl.transform.AscendSyncInsert")
    .set_body_typed(AscendSyncInsert);

} // namespace tl
} // namespace tvm
