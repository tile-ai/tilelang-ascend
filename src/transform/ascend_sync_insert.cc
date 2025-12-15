// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file ascend_sync_insert.cc
 * \brief Sync insertion for Ascend NPU
 */

#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <stack>

#include "arith/ir_mutator_with_analyzer.h"
#include "tir/analysis/var_use_def_analysis.h"

#include <tvm/tir/analysis.h>
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

static constexpr const char *kAscendAutoSync = "tl.ascend_auto_sync";

TVM_REGISTER_PASS_CONFIG_OPTION(kAscendAutoSync, Bool);

class AscendSyncInsert : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f, const std::string& config_path, PassContext ctx) {
    arith::Analyzer analyzer;
    AscendSyncInsert syncInserter(&analyzer);
    
    auto address_map = f->GetAttr<Map<Var, PrimExpr>>("address_map").value_or(Map<Var, PrimExpr>());
    syncInserter.InitConfig(config_path, address_map);
    
    PrimFuncNode* fptr = f.CopyOnWrite();
    auto fn_attr = fptr->attrs.CopyOnWrite();

    bool ascend_auto_sync = ctx->GetConfig<Bool>(kAscendAutoSync, Bool(false)).value();
    if (!ascend_auto_sync) {
      return f;
    }
    
    auto preprocessed = syncInserter.PreprocessUnrollForLoops(f->body);
    
    Stmt processed_body = syncInserter(preprocessed.first);

    fptr->body = syncInserter.MergeAndRebuildForLoops(processed_body, preprocessed.second);
    
    return f;
  }

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

  void InitConfig(const std::string& config_path, const Map<Var, PrimExpr>& address_map) {
    event_id_counter_ = 0;
    address_map_ = address_map;
    LoadDefaultConfig();
  }

  void LoadDefaultConfig() {
    event_mapping_ = {
      {"PIPE_MTE2_PIPE_MTE1", "MTE2_MTE1"},
      {"PIPE_MTE1_PIPE_MTE2", "MTE1_MTE2"},
      {"PIPE_MTE1_PIPE_M", "MTE1_M"},
      {"PIPE_M_PIPE_MTE1", "M_MTE1"},
      {"PIPE_MTE2_PIPE_V", "MTE2_V"},
      {"PIPE_V_PIPE_MTE2", "V_MTE2"},
      {"PIPE_MTE3_PIPE_V", "MTE3_V"},
      {"PIPE_V_PIPE_MTE3", "V_MTE3"},
      {"PIPE_M_PIPE_V", "M_V"},
      {"PIPE_V_PIPE_M", "V_M"},
      {"PIPE_V_PIPE_V", "V_V"},
      {"PIPE_MTE3_PIPE_MTE1", "MTE3_MTE1"},
      {"PIPE_MTE1_PIPE_MTE3", "MTE1_MTE3"},
      {"PIPE_MTE1_PIPE_V", "MTE1_V"},
      {"PIPE_MTE2_PIPE_M", "MTE2_M"},
      {"PIPE_M_PIPE_MTE2", "M_MTE2"},
      {"PIPE_V_PIPE_MTE1", "V_MTE1"},
      {"PIPE_M_PIPE_FIX", "M_FIX"},
      {"PIPE_FIX_PIPE_M", "FIX_M"},
      {"PIPE_MTE3_PIPE_MTE2", "MTE3_MTE2"},
      {"PIPE_MTE2_PIPE_MTE3", "MTE2_MTE3"},
      {"PIPE_S_PIPE_V", "S_V"},
      {"PIPE_V_PIPE_S", "V_S"},
      {"PIPE_S_PIPE_MTE2", "S_MTE2"},
      {"PIPE_MTE2_PIPE_S", "MTE2_S"},
      {"PIPE_S_PIPE_MTE3", "S_MTE3"},
      {"PIPE_MTE3_PIPE_S", "MTE3_S"},
      {"PIPE_MTE2_PIPE_FIX", "MTE2_FIX"},
      {"PIPE_FIX_PIPE_MTE2", "FIX_MTE2"},
      {"PIPE_FIX_PIPE_S", "FIX_S"},
      {"PIPE_M_PIPE_S", "M_S"},
      {"PIPE_FIX_PIPE_MTE3", "FIX_MTE3"}
    };

    operation_config_ = {
      {"copy_gm_to_l1", {{{0, "read"}, {1, "write"}}, "PIPE_MTE2"}},
      {"copy_gm_to_l0a", {{{0, "read"}, {1, "write"}}, "PIPE_MTE2"}},
      {"copy_gm_to_l0b", {{{0, "read"}, {1, "write"}}, "PIPE_MTE2"}},
      {"copy_gm_to_ub", {{{0, "read"}, {1, "write"}}, "PIPE_MTE2"}},
      {"copy_l1_to_l0a", {{{0, "read"}, {1, "write"}}, "PIPE_MTE1"}},
      {"copy_l1_to_l0b", {{{0, "read"}, {1, "write"}}, "PIPE_MTE1"}},
      {"copy_ub_to_gm", {{{0, "read"}, {1, "write"}}, "PIPE_MTE3"}},
      {"copy_ub_to_l1", {{{0, "read"}, {1, "write"}}, "PIPE_MTE3"}},
      {"copy_l0c_to_gm", {{{0, "read"}, {1, "write"}}, "PIPE_FIX"}},
      {"copy_l0c_to_l1", {{{0, "read"}, {1, "write"}}, "PIPE_FIX"}},
      {"copy_ub_to_ub", {{{0, "read"}, {1, "write"}}, "PIPE_V"}},
      {"mma", {{{0, "read"}, {1, "read"}, {2, "write"}}, "PIPE_M"}},
      {"gemm_v0", {{{0, "read"}, {1, "read"}, {2, "write"}}, "PIPE_M"}},
      {"gemm_v1", {{{0, "read"}, {1, "read"}, {2, "write"}}, "PIPE_M"}},
      {"AscendC::Add", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Mul", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Sub", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Subs", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Div", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Divs", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Reduce", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"AscendC::Scalar", {{{0, "write"}, {1, "read"}}, "PIPE_S"}},
      {"AscendC::Exp", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"AscendC::Ln", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"AscendC::Sqrt", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"AscendC::Relu", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"AscendC::Axpy", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"AscendC::Select", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"AscendC::Abs", {{{0, "write"}, {1, "read"}}, "PIPE_M"}},
      {"Gatherb", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::CompareScalar", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Duplicate", {{{0, "write"}}, "PIPE_V"}},
      {"AscendC::Muls", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::And", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"reduce_max", {{{0, "read"}, {1, "write"}}, "PIPE_V"}},
      {"reduce_max", {{{0, "read"}, {1, "write"}}, "PIPE_V"}},
      {"reduce_sum", {{{0, "read"}, {1, "write"}}, "PIPE_V"}},
      {"AscendC::Max", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Sin", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Cos", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Cast", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
    };
  }

  std::pair<Stmt, std::vector<LoopInfo>> PreprocessUnrollForLoops(const Stmt& stmt) {
    ForLoopUnroller unroller;
    auto result = unroller(stmt);
    return {result.first, result.second};
  }

  Stmt VisitStmt_(const SeqStmtNode* op) override {
    std::vector<Stmt> new_stmts;
    for (const Stmt& stmt : op->seq) {
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

  Stmt VisitStmt_(const EvaluateNode* op) override {
    auto current_accesses = AnalyzeStmtAccesses(GetRef<Stmt>(op));

    std::vector<SyncRequirement> sync_requirements;
    for (const auto& current_access : current_accesses) {
      if (current_access.is_sliced) {
        sync_requirements.push_back({"PipeBarrier_ALL", current_access.buffer_name});
      }
      
      std::vector<std::string> related_buffers = FindRelatedBuffers(current_access.buffer_name); 
      for (const auto& buffer_name : related_buffers) {
        auto it = current_access_history_.find(buffer_name);
        if (it != current_access_history_.end()) {
          const auto& latest_access = it->second;
          if (HasDataDependency(latest_access, current_access)) {
            std::string required_sync_type = GetRequiredSyncType(latest_access, current_access);
            if (!required_sync_type.empty()) {
              sync_requirements.push_back({required_sync_type, current_access.buffer_name});
            }
          }
        }
      }
    }

    auto optimized_syncs = OptimizeSyncRequirements(sync_requirements);
    
    std::vector<Stmt> stmts;
    for (const auto& sync_type : optimized_syncs) {
        InsertSynchronization(sync_type, stmts);
    }
    
    UpdateSyncStatesAfterSync(optimized_syncs);
    
    stmts.push_back(GetRef<Stmt>(op));
    
    UpdateLatestAccessHistory(current_accesses);
    
    if (stmts.size() == 1) {
        return stmts[0];
    } else {
        return SeqStmt(stmts);
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* op) override {
    if (op->attr_key == "resource_scope") {
      auto saved_access_history = current_access_history_;
      
      current_access_history_.clear();
      
      Stmt new_body = VisitStmt(op->body);
      
      current_access_history_ = saved_access_history;
      
      return AttrStmt(op->node, op->attr_key, op->value, new_body);
    } else if (op->attr_key == "unrolled_loop") {
      Stmt new_body = VisitStmt(op->body);
      return AttrStmt(op->node, op->attr_key, op->value, new_body);
    } else if (op->attr_key == "iteration_start" || op->attr_key == "iteration_end") {
      return GetRef<Stmt>(op);
    }
    
    Stmt new_body = VisitStmt(op->body);
    return AttrStmt(op->node, op->attr_key, op->value, new_body);
  }

  Stmt VisitStmt_(const LetStmtNode* op) override {
    auto value_accesses = AnalyzeExprAccesses(op->value);
    
    bool has_sliced_access = false;
    for (const auto& access : value_accesses) {
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

  Stmt VisitStmt_(const IfThenElseNode* op) override {
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

  Stmt MergeAndRebuildForLoops(const Stmt& processed_stmt, const std::vector<LoopInfo>& loop_infos) {
    LoopRebuilder rebuilder(loop_infos);
    return rebuilder(processed_stmt);
  }

private:
  class ForLoopUnroller : public StmtMutator {
  public:
    std::pair<Stmt, std::vector<LoopInfo>> operator()(const Stmt& stmt) {
      loop_infos_.clear();
      current_depth_ = 0;
      Stmt result = VisitStmt(stmt);
      return {result, loop_infos_};
    }

    Stmt VisitStmt_(const ForNode* op) override {
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
      
      unrolled_stmts.push_back(AttrStmt(make_zero(DataType::Int(32)), "iteration_start", 
                                      StringImm(loop_id + "_iter1"), Evaluate(0)));
      unrolled_stmts.push_back(processed_body);
      unrolled_stmts.push_back(AttrStmt(make_zero(DataType::Int(32)), "iteration_end", 
                                      StringImm(loop_id + "_iter1"), Evaluate(0)));
      
      unrolled_stmts.push_back(AttrStmt(make_zero(DataType::Int(32)), "iteration_start", 
                                      StringImm(loop_id + "_iter2"), Evaluate(0)));
      unrolled_stmts.push_back(processed_body);
      unrolled_stmts.push_back(AttrStmt(make_zero(DataType::Int(32)), "iteration_end", 
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
      
      return AttrStmt(make_zero(DataType::Int(32)), "unrolled_loop", StringImm(loop_id), unrolled_seq);
    }
    
    Stmt VisitStmt_(const SeqStmtNode* op) override {
      std::vector<Stmt> new_stmts;
      for (const Stmt& stmt : op->seq) {
        new_stmts.push_back(VisitStmt(stmt));
      }
      if (new_stmts.empty()) {
        return Evaluate(0);
      }
      return SeqStmt(new_stmts);
    }
    
    Stmt VisitStmt_(const AttrStmtNode* op) override {
      Stmt new_body = VisitStmt(op->body);
      return AttrStmt(op->node, op->attr_key, op->value, new_body);
    }
    
    Stmt VisitStmt_(const LetStmtNode* op) override {
      Stmt new_body = VisitStmt(op->body);
      return LetStmt(op->var, op->value, new_body);
    }

  private:
    std::vector<LoopInfo> loop_infos_;
    int current_depth_ = 0;
  };

  class LoopRebuilder : public StmtMutator {
  public:
    LoopRebuilder(const std::vector<LoopInfo>& loop_infos) : loop_infos_(loop_infos) {}
    
    Stmt operator()(const Stmt& stmt) {
      return VisitStmt(stmt);
    }

    Stmt VisitStmt_(const AttrStmtNode* op) override {
      if (op->attr_key == "unrolled_loop") {
        auto marker_name = op->value.as<StringImmNode>();
        if (marker_name) {
          std::string marker = marker_name->value;

          const LoopInfo* target_info = nullptr;
          for (const auto& info : loop_infos_) {
            if (info.loop_id == marker) {
              target_info = &info;
              break;
            }
          }
          
          if (target_info) {
            Stmt processed_body = VisitStmt(op->body);
            
            Stmt merged_body = MergeIterations(processed_body, target_info->loop_id);
            return For(target_info->loop_var, target_info->min, target_info->extent, 
                      target_info->kind, merged_body, NullOpt, target_info->annotations);
          }
        }
      }
      
      Stmt new_body = VisitStmt(op->body);
      return AttrStmt(op->node, op->attr_key, op->value, new_body);
    }
    
    Stmt VisitStmt_(const SeqStmtNode* op) override {
      std::vector<Stmt> new_stmts;
      for (const Stmt& stmt : op->seq) {
        new_stmts.push_back(VisitStmt(stmt));
      }
      if (new_stmts.empty()) {
        return Evaluate(0);
      }
      return SeqStmt(new_stmts);
    }

    Stmt VisitStmt_(const IfThenElseNode* op) override {
      Stmt then_case = VisitStmt(op->then_case);
      
      Optional<Stmt> else_case;
      if (op->else_case.defined()) {
        else_case = VisitStmt(op->else_case.value());
      }
      return IfThenElse(op->condition, then_case, else_case);
    }

  private:
    std::vector<LoopInfo> loop_infos_;
    
    Stmt MergeIterations(const Stmt& unrolled_body, const std::string& loop_id) {
      std::vector<Stmt> all_stmts = FlattenStmts(unrolled_body);
      if (all_stmts.empty()) {
        return Evaluate(0);
      }
      
      std::vector<Stmt> iter1_stmts, iter2_stmts;
      bool in_iter1 = false;
      bool in_iter2 = false;
      std::string current_iter;
      
      for (const auto& stmt : all_stmts) {
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
      
      std::vector<Stmt> merged_stmts = MergeStatementSequences(iter1_stmts, iter2_stmts, loop_id);
      
      if (merged_stmts.empty()) {
        return Evaluate(0);
      } else if (merged_stmts.size() == 1) {
        return merged_stmts[0];
      } else {
        return SeqStmt(merged_stmts);
      }
    }

    bool IsIterationStartMarker(const Stmt& stmt, const std::string& marker) {
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
    
    bool IsEmptyEvaluate(const Stmt& stmt) {
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

    bool IsIterationEndMarker(const Stmt& stmt, const std::string& marker) {
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

    bool IsUnrolledLoopMarker(const Stmt& stmt) {
      if (auto attr = stmt.as<AttrStmtNode>()) {
        if (attr->attr_key == "unrolled_loop") {
          return true;
        }
      }
      return false;
    }

    std::vector<Stmt> MergeStatementSequences(const std::vector<Stmt>& iter1_stmts,
                                              const std::vector<Stmt>& iter2_stmts,
                                              const std::string& loop_id) {
      std::vector<Stmt> merged_stmts;
      std::vector<Stmt> exec_stmts;
      std::vector<std::vector<Stmt>> syncs_before_execs;

      std::vector<Stmt> current_syncs;
      for (const auto& stmt : iter1_stmts) {
        if (IsSyncStatement(stmt)) {
          current_syncs.push_back(stmt);
        } else if (!IsMarkerStatement(stmt)) {
          exec_stmts.push_back(stmt);
          syncs_before_execs.push_back(current_syncs);
          current_syncs.clear();
        }
      }

      if (!current_syncs.empty()) {
        syncs_before_execs.push_back(current_syncs);
        current_syncs.clear();
      }

      std::vector<std::vector<Stmt>> iter2_syncs_before_execs(exec_stmts.size());
      size_t exec_index = 0;
      for (const auto& stmt : iter2_stmts) {
        if (IsSyncStatement(stmt)) {
          if (exec_index < iter2_syncs_before_execs.size()) {
            iter2_syncs_before_execs[exec_index].push_back(stmt);
          } else {
            current_syncs.push_back(stmt);
          }
        } else if (!IsMarkerStatement(stmt)) {
          exec_index++;
        }
      }

      if (!current_syncs.empty()) {
        iter2_syncs_before_execs.push_back(current_syncs);
      }

      std::vector<std::vector<Stmt>> merged_syncs_before_execs;
      for (size_t i = 0; i < syncs_before_execs.size(); i++) {
        std::vector<Stmt> merged_syncs = syncs_before_execs[i];
        if (i < iter2_syncs_before_execs.size()) {
          for (const auto& sync : iter2_syncs_before_execs[i]) {
            if (std::find(merged_syncs.begin(), merged_syncs.end(), sync) == merged_syncs.end()) {
              merged_syncs.push_back(sync);
            }
          }
        }
        merged_syncs_before_execs.push_back(merged_syncs);
      }

      for (size_t i = 0; i < exec_stmts.size(); i++) {
        std::vector<Stmt> syncs;
        for (const auto& sync : merged_syncs_before_execs[i]) {
          if (!ContainsSync(syncs, sync)) {
            syncs.push_back(sync);
            merged_stmts.push_back(sync);
          }
        }
        merged_stmts.push_back(exec_stmts[i]);
      }

      if (merged_syncs_before_execs.size() > exec_stmts.size()) {
        for (const auto& sync : merged_syncs_before_execs[exec_stmts.size()]) {
          merged_stmts.push_back(sync);
        }
      }

      return merged_stmts;
    }
    
    bool IsMarkerStatement(const Stmt& stmt) {
      if (auto attr = stmt.as<AttrStmtNode>()) {
        return (attr->attr_key == "iteration_start" || 
                attr->attr_key == "iteration_end" ||
                attr->attr_key == "unrolled_loop");
      }
      return false;
    }
    
    bool IsSyncStatement(const Stmt& stmt) {
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
          }
        }
      }
      return false;
    }
    
    bool ContainsSync(const std::vector<Stmt>& stmts, const Stmt& sync_stmt) {
      for (const auto& stmt : stmts) {
        if (IsSyncStatement(stmt) && IsSameSyncOperation(stmt, sync_stmt)) {
          return true;
        }
      }
      return false;
    }

    bool IsSameSyncOperation(const Stmt& stmt1, const Stmt& stmt2) {
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
      if (!call1 || !call2 || !call1->op.same_as(builtin::call_extern()) || !call2->op.same_as(builtin::call_extern())) {
        return false;
      }
      
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
      
      if (name1.find("AutoSetFlag") != std::string::npos || name1.find("AutoWaitFlag") != std::string::npos) {
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
    }
    
    std::vector<Stmt> FlattenStmts(const Stmt& stmt) {
      std::vector<Stmt> result;
      StmtFlattener flattener(result);
      flattener(stmt);
      return result;
    }

    class StmtFlattener : public StmtVisitor {
    public:
      StmtFlattener(std::vector<Stmt>& result) : result_(result) {}
      
      void VisitStmt_(const SeqStmtNode* op) override {
        for (const Stmt& stmt : op->seq) {
          VisitStmt(stmt);
        }
      }
      
      void VisitStmt_(const IfThenElseNode* op) override {
        result_.push_back(GetRef<Stmt>(op));
      }

      void VisitStmt_(const EvaluateNode* op) override {
        result_.push_back(GetRef<Stmt>(op));
      }
      
      void VisitStmt_(const AttrStmtNode* op) override {
        if (op->attr_key == "iteration_start" || 
            op->attr_key == "iteration_end") {
          result_.push_back(GetRef<Stmt>(op));
          VisitStmt(op->body);
        } else if (op->attr_key == "unrolled_loop") {
          VisitStmt(op->body);
        } else if (op->attr_key == "resource_scope") {
          result_.push_back(GetRef<Stmt>(op));
        } else {
          result_.push_back(GetRef<Stmt>(op));
          VisitStmt(op->body);
        }
      }
      
      void VisitStmt_(const LetStmtNode* op) override {
        result_.push_back(GetRef<Stmt>(op));
      }
      
      void VisitStmt_(const ForNode* op) override {
        result_.push_back(GetRef<Stmt>(op));
      }
      
      void VisitStmt_(const AllocateNode* op) override {
        result_.push_back(GetRef<Stmt>(op));
        VisitStmt(op->body);
      }
      
    private:
      std::vector<Stmt>& result_;
    };
  };

  struct SyncGraph {
    std::unordered_map<std::string, std::unordered_set<std::string>> graph;

    std::string toString() const {
      std::ostringstream oss;
      oss << "SyncGraph{graph: {";
      bool first_pair = true;
      for (const auto& pair : graph) {
        if (!first_pair) oss << ", ";
        oss << "'" << pair.first << "': [";
        bool first_dst = true;
        for (const auto& dst : pair.second) {
          if (!first_dst) oss << ", ";
          oss << "'" << dst << "'";
          first_dst = false;
        }
        oss << "]";
        first_pair = false;
      }
      oss << "}}";
      return oss.str();
    }
    
    void AddSync(const std::string& sync_type) {
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
    
    bool HasPath(const std::string& src, const std::string& dst) const {
      if (src == dst) return true;
      
      std::unordered_set<std::string> visited;
      std::vector<std::string> queue = {src};
      visited.insert(src);
      
      while (!queue.empty()) {
        std::string current = queue.back();
        queue.pop_back();
        
        auto it = graph.find(current);
        if (it != graph.end()) {
          for (const auto& neighbor : it->second) {
            if (neighbor == dst) return true;
            if (visited.count(neighbor) == 0) {
              visited.insert(neighbor);
              queue.push_back(neighbor);
            }
          }
        }
      }
      
      return false;
    }
    
    void Merge(const SyncGraph& other) {
      for (const auto& pair : other.graph) {
        const std::string& src = pair.first;
        for (const std::string& dst : pair.second) {
          graph[src].insert(dst);
        }
      }
    }
    
    SyncGraph ComputeTransitiveClosure() const {
      SyncGraph closure;
      closure.graph = graph;
      
      std::unordered_set<std::string> nodes;
      for (const auto& pair : graph) {
        nodes.insert(pair.first);
        for (const auto& dst : pair.second) {
          nodes.insert(dst);
        }
      }
      
      std::vector<std::string> node_list(nodes.begin(), nodes.end());
      
      for (const auto& k : node_list) {
        for (const auto& i : node_list) {
          for (const auto& j : node_list) {
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
    bool is_sliced;  // 新增：切片操作标记

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
      for (const auto& barrier : pipe_barriers) {
        if (!first_barrier) oss << ", ";
        oss << "'" << barrier << "'";
        first_barrier = false;
      }
      oss << "]";
      oss << "}";
      return oss.str();
    }
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

  struct OperationConfig {
    std::vector<std::pair<size_t, std::string>> buffer_accesses;
    std::string default_pipeline;

    std::string toString() const {
      std::ostringstream oss;
      oss << "OperationConfig{";
      oss << "buffer_accesses: [";
      bool first_access = true;
      for (const auto& access : buffer_accesses) {
        if (!first_access) oss << ", ";
        oss << "(" << access.first << ", '" << access.second << "')";
        first_access = false;
      }
      oss << "], ";
      oss << "default_pipeline: '" << default_pipeline << "'";
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

  std::vector<BufferAccess> AnalyzeExprAccesses(const PrimExpr& expr) {
      std::vector<BufferAccess> accesses;
      
      ExprAccessAnalyzer analyzer;
      analyzer(expr);
      
      for (const auto& buffer_name : analyzer.GetAccessedBuffers()) {
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
      void VisitExpr_(const CallNode* op) override {
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
      
      void VisitExpr_(const BufferLoadNode* op) override {
          std::string buffer_name = op->buffer->data->name_hint;
          accessed_buffers_.insert(buffer_name);
          
          sliced_buffers_.insert(buffer_name);
          
          for (const auto& index : op->indices) {
              VisitExpr(index);
          }
          
          ExprVisitor::VisitExpr_(op);
      }
      
      std::unordered_set<std::string> GetAccessedBuffers() const {
          return accessed_buffers_;
      }
      
      bool IsBufferSliced(const std::string& buffer_name) const {
          return sliced_buffers_.count(buffer_name) > 0;
      }
      
  private:
      std::unordered_set<std::string> accessed_buffers_;
      std::unordered_set<std::string> sliced_buffers_;
  };

  template<typename T>
  std::string containerToString(const std::vector<T>& vec) {
    std::ostringstream oss;
    oss << "[";
    bool first = true;
    for (const auto& item : vec) {
      if (!first) oss << ", ";
      oss << item.toString();
      first = false;
    }
    oss << "]";
    return oss.str();
  }

  std::string containerToString(const std::vector<std::string>& vec) {
    std::ostringstream oss;
    oss << "[";
    bool first = true;
    for (const auto& item : vec) {
      if (!first) oss << ", ";
      oss << item;
      first = false;
    }
    oss << "]";
    return oss.str();
  }

  std::string containerToString(const std::unordered_map<std::string, BufferAccess>& map) {
    std::ostringstream oss;
    oss << "{";
    bool first = true;
    for (const auto& pair : map) {
      if (!first) oss << ", ";
      oss << "'" << pair.first << "': " << pair.second.toString();
      first = false;
    }
    oss << "}";
    return oss.str();
  }

  std::vector<BufferAccess> AnalyzeStmtAccesses(const Stmt& stmt) {
    std::vector<BufferAccess> accesses;
    
    if (auto eval = stmt.as<EvaluateNode>()) {
      if (auto call = eval->value.as<CallNode>()) {
        if (call->op.same_as(builtin::call_extern())) {
          std::string func_name = Downcast<StringImm>(call->args[0])->value;

          std::string normalized_name = NormalizeFunctionName(func_name);
          auto config_it = operation_config_.find(normalized_name);
          if (config_it != operation_config_.end()) {
            const auto& config = config_it->second;
            
            std::unordered_map<std::string, BufferAccess> buffer_access_map;
            
            for (const auto& buffer_config : config.buffer_accesses) {
              size_t arg_index = buffer_config.first;
              const std::string& access_type = buffer_config.second;
              
              if (arg_index + 1 < call->args.size()) {
                auto buffer_info = ExtractBufferInfoFromAccessPtr(call->args[arg_index + 1]);
                if (!buffer_info.buffer_name.empty()) {
                  bool is_write = (access_type == "write");
                  
                  if (buffer_access_map.find(buffer_info.buffer_name) != buffer_access_map.end()) {
                    BufferAccess& existing_access = buffer_access_map[buffer_info.buffer_name];
                    if (is_write || (!existing_access.is_write && is_write)) {
                      existing_access.is_write = true;
                    }
                    existing_access.is_sliced = existing_access.is_sliced || buffer_info.is_sliced;
                  } else {
                    BufferAccess access;
                    access.sync_graph = SyncGraph();
                    access.pipe_barriers = {};
                    access.physical_address = GetPhysicalAddress(buffer_info.buffer_name);
                    
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
            
            for (const auto& pair : buffer_access_map) {
              accesses.push_back(pair.second);
            }
          }
        }
      }
    }
    
    return accesses;
  }

  std::string NormalizeFunctionName(const std::string& func_name) {
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

  BufferInfo ExtractBufferInfoFromAccessPtr(const PrimExpr& expr) {
    BufferInfo info = {"", false, false};
    
    if (auto call = expr.as<CallNode>()) {
      if (call->op.same_as(builtin::tvm_access_ptr())) {
        if (call->args.size() >= 5) {
          if (auto var = call->args[1].as<VarNode>()) {
            info.buffer_name = var->name_hint;
          }

          ExprAccessAnalyzer analyzer;
          analyzer(call->args[2]);
          for (const auto& buffer_name : analyzer.GetAccessedBuffers()) {
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

  bool HasDataDependency(const BufferAccess& prev, const BufferAccess& curr) {
    if (prev.physical_address != -1 && curr.physical_address != -1 && 
        prev.physical_address == curr.physical_address) {
      if ((prev.is_write && curr.is_write) ||      // WAW
          (prev.is_write && !curr.is_write) ||     // RAW
          (!prev.is_write && curr.is_write)) {     // WAR
        return true;
      }
    }
    
    if (prev.buffer_name == curr.buffer_name) {
      if ((prev.is_write && curr.is_write) ||      // WAW
          (prev.is_write && !curr.is_write) ||     // RAW
          (!prev.is_write && curr.is_write)) {     // WAR
        return true;
      }
    }
    return false;
  }

  void UpdateLatestAccessHistory(const std::vector<BufferAccess>& current_accesses) {
    for (const auto& access : current_accesses) {
      current_access_history_[access.buffer_name] = access;
    }
  }

  std::string GetRequiredSyncType(const BufferAccess& prev_access, const BufferAccess& curr_access) {
    if (prev_access.pipeline == curr_access.pipeline
        && prev_access.pipe_barriers.find("PipeBarrier_" + prev_access.pipeline) == prev_access.pipe_barriers.end()) {
      return "PipeBarrier_" + prev_access.pipeline;
    } else {
      std::string event_type = GetEventType(prev_access.pipeline, curr_access.pipeline);
      if (!event_type.empty()) {
        return "EventPair_" + event_type;
      }
    }
    return "";
  }

  std::string GetEventType(const std::string& src_pipeline, const std::string& dst_pipeline) {
    std::string key = src_pipeline + "_" + dst_pipeline;
    auto it = event_mapping_.find(key);
    return it != event_mapping_.end() ? it->second : "";
  }

  int64_t GetPhysicalAddress(const std::string& buffer_name) {
    for (const auto& pair : address_map_) {
      if (pair.first->name_hint == buffer_name) {
        if (auto int_imm = pair.second.as<IntImmNode>()) {
          return int_imm->value;
        }
      }
    }
    return -1;
  }

  std::vector<std::string> FindRelatedBuffers(const std::string& buffer_name) {
    std::vector<std::string> related;
    int64_t target_addr = GetPhysicalAddress(buffer_name);
    
    if (target_addr == -1) {
      related.push_back(buffer_name);
      return related;
    }
    
    for (const auto& pair : address_map_) {
      if (auto int_imm = pair.second.as<IntImmNode>()) {
        if (int_imm->value == target_addr) {
          related.push_back(pair.first->name_hint);
        }
      }
    }
    return related;
  }

  std::vector<std::string> OptimizeSyncRequirements(const std::vector<SyncRequirement>& requirements) {
    if (requirements.empty()) {
      return {};
    }
    
    std::vector<std::string> all_required_syncs;
    for (const auto& req : requirements) {
      all_required_syncs.push_back(req.sync_type);
    }
    
    std::sort(all_required_syncs.begin(), all_required_syncs.end());
    all_required_syncs.erase(std::unique(all_required_syncs.begin(), all_required_syncs.end()), all_required_syncs.end());
    
    std::vector<std::string> final_syncs;
    
    for (const auto& sync_type : all_required_syncs) {
      bool needed = false;
      
      for (const auto& req : requirements) {
        if (req.sync_type == sync_type) {
          SyncGraph extended_graph = GetBufferSyncGraph(req.buffer_name);
          
          for (const auto& other_sync : all_required_syncs) {
            if (other_sync != sync_type) {
              extended_graph.AddSync(other_sync);
            }
          }
          
          if (!IsSyncSatisfiedByGraph(sync_type, extended_graph)) {
            needed = true;
            break;
          }
        }
      }
      
      if (needed) {
        final_syncs.push_back(sync_type);
      }
    }
    
    return final_syncs;
  }

  SyncGraph GetBufferSyncGraph(const std::string& buffer_name) {
    auto it = current_access_history_.find(buffer_name);
    if (it != current_access_history_.end()) {
      return it->second.sync_graph;
    }
    return SyncGraph();
  }

  bool IsSyncSatisfiedByGraph(const std::string& sync_type, const SyncGraph& graph) {
    if (sync_type.find("EventPair_") == 0) {
      std::string target_event = sync_type.substr(10);
      size_t pos = target_event.find('_');
      if (pos != std::string::npos) {
        std::string target_src = target_event.substr(0, pos);
        std::string target_dst = target_event.substr(pos + 1);
        return graph.HasPath(target_src, target_dst);
      }
    }
    return false;
  }

  void UpdateSyncStatesAfterSync(const std::vector<std::string>& inserted_syncs) {
    SyncGraph inserted_graph;
    for (const auto& sync_type : inserted_syncs) {
      inserted_graph.AddSync(sync_type);
    }
    
    SyncGraph transitive_closure = inserted_graph.ComputeTransitiveClosure();
    
    for (auto& pair : current_access_history_) {
      BufferAccess& access = pair.second;
      
      for (const auto& sync_type : inserted_syncs) {
        if (sync_type.find("EventPair_") == 0) {
          access.sync_graph.AddSync(sync_type);
        } else if (sync_type.find("PipeBarrier_") == 0) {
          std::string pipeline = sync_type.substr(12);
          if (access.pipeline == pipeline) {
            access.pipe_barriers.insert(sync_type);
          }
        }
      }
      
      access.sync_graph.Merge(transitive_closure);
    }
  }

  void InsertSynchronization(const std::string& sync_type, std::vector<Stmt>& stmts) {
    if (sync_type == "PipeBarrier_ALL") {
      stmts.push_back(CreatePipeBarrier("PIPE_ALL"));
    } else if (sync_type.find("PipeBarrier_") == 0) {
      std::string pipeline = sync_type.substr(12);
      stmts.push_back(CreatePipeBarrier(pipeline));
    } else if (sync_type.find("EventPair_") == 0) {
      std::string event_type = sync_type.substr(10);
      int event_id = AllocateEventId();
      stmts.push_back(CreateSetFlag(event_type, event_id));
      stmts.push_back(CreateWaitFlag(event_type, event_id));
    }
  }

  int AllocateEventId() {
    return event_id_counter_++;
  }

  Stmt CreatePipeBarrier(const std::string& pipeline) {
    Array<PrimExpr> args = {StringImm("AscendC::AutoBarrier"), StringImm(pipeline)};
    return Evaluate(Call(DataType::Handle(), builtin::call_extern(), args));
  }

  Stmt CreateSetFlag(const std::string& event_type, int event_id) {
    Array<PrimExpr> args = {StringImm("AscendC::AutoSetFlag"), StringImm(event_type), 
                           IntImm(DataType::Int(32), event_id)};
    return Evaluate(Call(DataType::Handle(), builtin::call_extern(), args));
  }

  Stmt CreateWaitFlag(const std::string& event_type, int event_id) {
    Array<PrimExpr> args = {StringImm("AscendC::AutoWaitFlag"), StringImm(event_type), 
                           IntImm(DataType::Int(32), event_id)};
    return Evaluate(Call(DataType::Handle(), builtin::call_extern(), args));
  }

private:
  int event_id_counter_ = 0;
  std::unordered_map<std::string, std::string> event_mapping_;
  std::unordered_map<std::string, OperationConfig> operation_config_;
  std::unordered_map<std::string, BufferAccess> current_access_history_;
  Map<Var, PrimExpr> address_map_;
};

tvm::transform::Pass AscendSyncInsert() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto new_func = AscendSyncInsert::Substitute(std::move(f), "config_path", ctx);
    return new_func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendSyncInsert", {});
}

TVM_REGISTER_GLOBAL("tl.transform.AscendSyncInsert")
    .set_body_typed(AscendSyncInsert);

}  // namespace tl
}  // namespace tvm