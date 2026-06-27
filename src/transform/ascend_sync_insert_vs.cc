// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file ascend_sync_insert_vs.cc
 * \brief Simplified sync insertion for Ascend NPU.
 *        Tracks PIPE_V / PIPE_S / PIPE_MTE2 / PIPE_MTE3 only.
 *        Inserts sync when at least one side is PIPE_V or PIPE_S.
 */

#include <algorithm>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "arith/ir_mutator_with_analyzer.h"

#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include "../op/ascend.h"
#include "./common/operation_config.h"

#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

static constexpr const char *kAscendAutoSyncVs = "tl.ascend_auto_sync_vs";

TVM_REGISTER_PASS_CONFIG_OPTION(kAscendAutoSyncVs, Bool);

class AscendSyncInsertVS : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f, PassContext ctx, Target target,
                             std::string platform) {
    bool enabled = ctx->GetConfig<Bool>(kAscendAutoSyncVs, Bool(false)).value();
    if (!enabled) {
      return f;
    }

    arith::Analyzer analyzer;
    AscendSyncInsertVS mutator(&analyzer, target, platform);

    auto address_map = f->GetAttr<Map<Var, PrimExpr>>("address_map")
                           .value_or(Map<Var, PrimExpr>());
    auto size_map = f->GetAttr<Map<Var, PrimExpr>>("size_map")
                        .value_or(Map<Var, PrimExpr>());
    mutator.InitConfig(address_map, size_map);

    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = mutator(f->body);
    return f;
  }

  explicit AscendSyncInsertVS(arith::Analyzer *analyzer, Target target,
                              std::string platform)
      : arith::IRMutatorWithAnalyzer(analyzer), target_(target),
        platform_(platform) {}

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  struct BufferAccess {
    std::string buffer_name;
    bool is_write;
    std::string pipeline;
    std::string operation;
    std::set<std::string> pipe_barriers;
    int64_t physical_address;
    bool is_back_edge = false;
  };

  struct SyncRequirement {
    std::string sync_type;
    std::string buffer_name;
  };

  struct BufferInfo {
    std::string buffer_name;
    bool is_read;
    bool is_write;
  };

  void InitConfig(const Map<Var, PrimExpr> &address_map,
                  const Map<Var, PrimExpr> &size_map) {
    event_id_counter_ = 0;
    address_map_ = address_map;
    size_map_ = size_map;
    event_mapping_ = GetEventMapping();
    operation_config_ = GetOperationConfig();
  }

  // ==================== VisitStmt overrides ====================

  Stmt VisitStmt_(const SeqStmtNode *op) override {
    std::vector<Stmt> new_stmts;
    for (const Stmt &stmt : op->seq) {
      new_stmts.push_back(VisitStmt(stmt));
    }
    if (new_stmts.empty()) {
      return Evaluate(0);
    }
    if (new_stmts.size() == 1) {
      return new_stmts[0];
    }
    return SeqStmt(new_stmts);
  }

  Stmt VisitStmt_(const EvaluateNode *op) override {
    if (auto call = op->value.as<CallNode>()) {
      if (call->op.same_as(tl::ascend_auto_barrier())) {
        if (call->args.size() >= 1) {
          if (auto str = call->args[0].as<StringImmNode>()) {
            std::string pipeline = str->value;
            std::string barrier = "PipeBarrier_" + pipeline;
            for (auto &pair : current_access_history_) {
              if (pair.second.pipeline == pipeline) {
                pair.second.pipe_barriers.insert(barrier);
              }
            }
          }
        }
        return GetRef<Stmt>(op);
      }
    }

    auto current_accesses = AnalyzeStmtAccesses(GetRef<Stmt>(op));

    auto scalar_reads = ScanBufferLoads(op->value);
    for (const auto &read : scalar_reads) {
      bool found = false;
      for (const auto &acc : current_accesses) {
        if (acc.buffer_name == read.buffer_name) {
          found = true;
          break;
        }
      }
      if (!found) {
        current_accesses.push_back(read);
      }
    }

    return ProcessStatement(GetRef<Stmt>(op), current_accesses);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) override {
    std::string scope = GetPtrStorageScope(op->buffer->data);
    if (scope == "local.var" || scope == "n") {
      return GetRef<Stmt>(op);
    }

    std::vector<BufferAccess> current_accesses;

    BufferAccess write_access;
    write_access.buffer_name = op->buffer->data->name_hint;
    write_access.is_write = true;
    write_access.pipeline = "PIPE_S";
    write_access.operation = "BufferStore";
    write_access.physical_address =
        GetPhysicalAddress(write_access.buffer_name);
    current_accesses.push_back(write_access);

    auto scalar_reads = ScanBufferLoads(op->value);
    for (const auto &read : scalar_reads) {
      bool found = false;
      for (const auto &acc : current_accesses) {
        if (acc.buffer_name == read.buffer_name) {
          found = true;
          break;
        }
      }
      if (!found) {
        current_accesses.push_back(read);
      }
    }

    return ProcessStatement(GetRef<Stmt>(op), current_accesses);
  }

  Stmt VisitStmt_(const LetStmtNode *op) override {
    auto scalar_reads = ScanBufferLoads(op->value);

    if (scalar_reads.empty()) {
      Stmt new_body = VisitStmt(op->body);
      return LetStmt(op->var, op->value, new_body);
    }

    return ProcessLetStatement(op, scalar_reads);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) override {
    if (op->attr_key == "resource_scope") {
      auto saved_access_history = current_access_history_;
      current_access_history_.clear();
      Stmt new_body = VisitStmt(op->body);
      current_access_history_ = saved_access_history;
      return AttrStmt(op->node, op->attr_key, op->value, new_body);
    }
    Stmt new_body = VisitStmt(op->body);
    return AttrStmt(op->node, op->attr_key, op->value, new_body);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) override {
    auto saved_history = current_access_history_;
    Stmt then_case = VisitStmt(op->then_case);

    auto then_history = current_access_history_;
    current_access_history_ = saved_history;
    Optional<Stmt> else_case;
    if (op->else_case.defined()) {
      else_case = VisitStmt(op->else_case.value());
    }

    current_access_history_ = saved_history;
    for (const auto &kv : then_history) {
      current_access_history_[kv.first] = kv.second;
    }
    return IfThenElse(op->condition, then_case, else_case);
  }

  Stmt VisitStmt_(const ForNode *op) override {
    if (is_revisit_pass_) {
      return GetRef<Stmt>(op);
    }

    auto saved_history = current_access_history_;
    Stmt first_body = VisitStmt(op->body);
    auto end_history = current_access_history_;

    current_access_history_ = saved_history;
    for (const auto &kv : end_history) {
      BufferAccess back_edge_access = kv.second;
      back_edge_access.is_back_edge = true;
      current_access_history_[kv.first] = back_edge_access;
    }

    is_revisit_pass_ = true;
    Stmt final_body = VisitStmt(first_body);
    is_revisit_pass_ = false;

    current_access_history_ = end_history;

    return For(op->loop_var, op->min, op->extent, op->kind, final_body,
               op->thread_binding, op->annotations);
  }

  Stmt VisitStmt_(const AllocateNode *op) override {
    Stmt new_body = VisitStmt(op->body);
    return Allocate(op->buffer_var, op->dtype, op->extents, op->condition,
                    new_body);
  }

  // ==================== Core processing ====================

  Stmt ProcessStatement(const Stmt &stmt,
                        const std::vector<BufferAccess> &current_accesses) {
    std::vector<BufferAccess> supported;
    for (const auto &access : current_accesses) {
      if (IsSupportedPipeline(access.pipeline)) {
        supported.push_back(access);
      }
    }

    if (supported.empty()) {
      return stmt;
    }

    std::vector<SyncRequirement> sync_requirements;
    for (const auto &current_access : supported) {
      std::vector<std::string> related =
          FindRelatedBuffers(current_access.buffer_name);
      for (const auto &buffer_name : related) {
        auto it = current_access_history_.find(buffer_name);
        if (it != current_access_history_.end()) {
          const auto &latest = it->second;
          if (HasDataDependency(latest, current_access)) {
            std::string sync_type = GetRequiredSyncType(latest, current_access);
            if (!sync_type.empty()) {
              sync_requirements.push_back(
                  {sync_type, current_access.buffer_name});
            }
          }
        }
      }
    }

    auto optimized_syncs = DedupSyncRequirements(sync_requirements);

    std::vector<Stmt> stmts;
    for (const auto &sync_type : optimized_syncs) {
      InsertSynchronization(sync_type, stmts);
    }

    UpdateSyncStatesAfterSync(optimized_syncs);
    stmts.push_back(stmt);
    UpdateLatestAccessHistory(supported);

    if (stmts.size() == 1) {
      return stmts[0];
    }
    return SeqStmt(stmts);
  }

  Stmt ProcessLetStatement(const LetStmtNode *op,
                           const std::vector<BufferAccess> &scalar_reads) {
    std::vector<SyncRequirement> sync_requirements;
    for (const auto &current_access : scalar_reads) {
      std::vector<std::string> related =
          FindRelatedBuffers(current_access.buffer_name);
      for (const auto &buffer_name : related) {
        auto it = current_access_history_.find(buffer_name);
        if (it != current_access_history_.end()) {
          const auto &latest = it->second;
          if (HasDataDependency(latest, current_access)) {
            std::string sync_type = GetRequiredSyncType(latest, current_access);
            if (!sync_type.empty()) {
              sync_requirements.push_back(
                  {sync_type, current_access.buffer_name});
            }
          }
        }
      }
    }

    auto optimized_syncs = DedupSyncRequirements(sync_requirements);

    std::vector<Stmt> stmts;
    for (const auto &sync_type : optimized_syncs) {
      InsertSynchronization(sync_type, stmts);
    }

    UpdateSyncStatesAfterSync(optimized_syncs);
    UpdateLatestAccessHistory(scalar_reads);

    Stmt new_body = VisitStmt(op->body);
    stmts.push_back(LetStmt(op->var, op->value, new_body));

    if (stmts.size() == 1) {
      return stmts[0];
    }
    return SeqStmt(stmts);
  }

  // ==================== Analysis ====================

  std::vector<BufferAccess> AnalyzeStmtAccesses(const Stmt &stmt) {
    std::vector<BufferAccess> accesses;

    auto eval = stmt.as<EvaluateNode>();
    if (!eval) {
      return accesses;
    }

    auto call = eval->value.as<CallNode>();
    if (!call) {
      return accesses;
    }

    if (call->op.same_as(builtin::call_extern())) {
      std::string func_name = Downcast<StringImm>(call->args[0])->value;
      std::string normalized = NormalizeFunctionName(func_name);
      auto config_it = operation_config_.find(normalized);
      if (config_it != operation_config_.end()) {
        CollectBufferAccesses(config_it->second, normalized, call->args, 1,
                              accesses);
      }
    } else {
      auto *op_ptr = call->op.as<OpNode>();
      if (op_ptr) {
        std::string op_name = op_ptr->name;
        auto config_it = operation_config_.find(op_name);
        if (config_it != operation_config_.end()) {
          CollectBufferAccesses(config_it->second, op_name, call->args, 0,
                                accesses);
        }
      }
    }

    return accesses;
  }

  void CollectBufferAccesses(const OperationConfig &config,
                             const std::string &op_name,
                             const Array<PrimExpr> &call_args, size_t offset,
                             std::vector<BufferAccess> &accesses) {
    std::unordered_map<std::string, BufferAccess> buffer_access_map;

    for (const auto &buffer_config : config.buffer_accesses) {
      size_t arg_index = buffer_config.first + offset;
      const std::string &access_type = buffer_config.second;

      if (arg_index >= call_args.size()) {
        continue;
      }

      auto buf_info = ExtractBufferInfoFromAccessPtr(call_args[arg_index]);
      if (buf_info.buffer_name.empty()) {
        continue;
      }

      bool is_write = (access_type == "write");
      auto it = buffer_access_map.find(buf_info.buffer_name);
      if (it != buffer_access_map.end()) {
        if (is_write) {
          it->second.is_write = true;
        }
      } else {
        BufferAccess access;
        access.buffer_name = buf_info.buffer_name;
        access.is_write = is_write;
        access.pipeline = config.default_pipeline;
        access.operation = op_name;
        access.physical_address = GetPhysicalAddress(buf_info.buffer_name);
        buffer_access_map[buf_info.buffer_name] = access;
      }
    }

    for (const auto &pair : buffer_access_map) {
      accesses.push_back(pair.second);
    }
  }

  // ExprVisitor-based collector that finds all BufferLoad nodes anywhere in
  // an expression tree (including inside Add/Mul/Cast/Select/Call, etc.).
  // The previous hand-rolled recursion only visited BufferLoadNode and
  // CallNode, silently missing loads nested in arithmetic or logical ops.
  class BufferLoadCollector : public ExprVisitor {
  public:
    BufferLoadCollector(std::vector<BufferAccess> *accesses,
                        AscendSyncInsertVS *pass)
        : accesses_(accesses), pass_(pass) {}

    void VisitExpr_(const BufferLoadNode *op) override {
      std::string scope = GetPtrStorageScope(op->buffer->data);
      if (scope != "local.var" && scope != "n") {
        BufferAccess access;
        access.buffer_name = op->buffer->data->name_hint;
        access.is_write = false;
        access.pipeline = "PIPE_S";
        access.operation = "BufferLoad";
        access.physical_address = pass_->GetPhysicalAddress(access.buffer_name);
        accesses_->push_back(access);
      }
      ExprVisitor::VisitExpr_(op);
    }

  private:
    std::vector<BufferAccess> *accesses_;
    AscendSyncInsertVS *pass_;
  };

  std::vector<BufferAccess> ScanBufferLoads(const PrimExpr &expr) {
    std::vector<BufferAccess> accesses;
    BufferLoadCollector collector(&accesses, this);
    collector(expr);
    return accesses;
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
      if (prev.pipeline != curr.pipeline) {
        return true;
      }
      if ((prev.is_write && curr.is_write) ||
          (prev.is_write && !curr.is_write) ||
          (!prev.is_write && curr.is_write)) {
        return true;
      }
    }
    return false;
  }

  std::string GetRequiredSyncType(const BufferAccess &prev_access,
                                  const BufferAccess &curr_access) {
    if (!ShouldSync(prev_access.pipeline, curr_access.pipeline)) {
      return "";
    }

    if (is_revisit_pass_ && !prev_access.is_back_edge) {
      return "";
    }

    if (prev_access.pipeline == curr_access.pipeline) {
      std::string barrier = "PipeBarrier_" + prev_access.pipeline;
      if (prev_access.pipe_barriers.find(barrier) !=
          prev_access.pipe_barriers.end()) {
        return "";
      }
      return barrier;
    }

    std::string event_type =
        GetEventType(prev_access.pipeline, curr_access.pipeline);
    if (!event_type.empty()) {
      return "EventPair_" + event_type;
    }
    return "";
  }

  std::string GetEventType(const std::string &src_pipeline,
                           const std::string &dst_pipeline) {
    std::string key = src_pipeline + "_" + dst_pipeline;
    auto it = event_mapping_.find(key);
    return it != event_mapping_.end() ? it->second : "";
  }

  // ==================== Address / size / scope ====================

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
          if (other_addr == target_addr) {
            related.push_back(pair.first->name_hint);
          }
          continue;
        }
        int64_t other_end = other_addr + other_size;
        if (target_addr < other_end && other_addr < target_end) {
          related.push_back(pair.first->name_hint);
        }
      }
    }
    return related;
  }

  // ==================== Name extraction ====================

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

    auto call = expr.as<CallNode>();
    if (!call) {
      return info;
    }

    if (call->op.same_as(builtin::tvm_access_ptr())) {
      if (call->args.size() >= 5) {
        if (auto var = call->args[1].as<VarNode>()) {
          info.buffer_name = var->name_hint;
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

    return info;
  }

  // ==================== Sync generation ====================

  void InsertSynchronization(const std::string &sync_type,
                             std::vector<Stmt> &stmts) {
    if (sync_type.find("PipeBarrier_") == 0) {
      std::string pipeline = sync_type.substr(12);
      if (pipeline == "PIPE_V" && platform_ == "A5") {
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
    event_id_counter_ = (event_id_counter_ + 1) % 8;
    return event_id_counter_;
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

  // ==================== State management ====================

  std::vector<std::string>
  DedupSyncRequirements(std::vector<SyncRequirement> &requirements) {
    if (requirements.empty()) {
      return {};
    }

    std::sort(requirements.begin(), requirements.end(),
              [](const SyncRequirement &a, const SyncRequirement &b) {
                return a.sync_type < b.sync_type;
              });

    std::vector<std::string> unique_syncs;
    for (const auto &req : requirements) {
      if (unique_syncs.empty() || unique_syncs.back() != req.sync_type) {
        unique_syncs.push_back(req.sync_type);
      }
    }
    return unique_syncs;
  }

  void
  UpdateSyncStatesAfterSync(const std::vector<std::string> &inserted_syncs) {
    for (auto &pair : current_access_history_) {
      BufferAccess &access = pair.second;
      for (const auto &sync_type : inserted_syncs) {
        if (sync_type.find("PipeBarrier_") == 0) {
          std::string pipeline = sync_type.substr(12);
          if (access.pipeline == pipeline) {
            access.pipe_barriers.insert(sync_type);
          }
        }
      }
    }
  }

  void
  UpdateLatestAccessHistory(const std::vector<BufferAccess> &current_accesses) {
    for (const auto &access : current_accesses) {
      current_access_history_[access.buffer_name] = access;
    }
  }

  // ==================== Pipeline filtering ====================

  bool IsSupportedPipeline(const std::string &p) {
    return p == "PIPE_V" || p == "PIPE_S" || p == "PIPE_MTE2" ||
           p == "PIPE_MTE3";
  }

  bool ShouldSync(const std::string &a, const std::string &b) {
    if (!IsSupportedPipeline(a) || !IsSupportedPipeline(b)) {
      return false;
    }
    // V->V (same pipeline)
    if (a == "PIPE_V" && b == "PIPE_V") {
      return true;
    }
    // S <-> other (but not S->S, scalar pipeline is in-order)
    if ((a == "PIPE_S" || b == "PIPE_S") && a != b) {
      return true;
    }
    return false;
  }

  // ==================== Members ====================

  int event_id_counter_ = 0;
  bool is_revisit_pass_ = false;
  std::unordered_map<std::string, std::string> event_mapping_;
  std::unordered_map<std::string, OperationConfig> operation_config_;
  std::unordered_map<std::string, BufferAccess> current_access_history_;
  Map<Var, PrimExpr> address_map_;
  Map<Var, PrimExpr> size_map_;
  std::string platform_;
  Target target_;
};

tvm::transform::Pass AscendSyncInsertVS(Target target, std::string platform) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return AscendSyncInsertVS::Substitute(std::move(f), ctx, target, platform);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendSyncInsertVS", {});
}

TVM_REGISTER_GLOBAL("tl.transform.AscendSyncInsertVS")
    .set_body_typed(AscendSyncInsertVS);

} // namespace tl
} // namespace tvm
