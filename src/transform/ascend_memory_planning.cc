// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file ascend_memory_planning.cc
 * \brief Memory planning for Ascend NPU
 */

#include <iostream>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "arith/ir_mutator_with_analyzer.h"
#include "tir/analysis/var_use_def_analysis.h"
#include "tir/transforms/ir_utils.h"

#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include "../op/ascend.h"
#include "../op/builtin.h"
#include "./common/attr.h"
#include "./common/collector.h"

#define ASCEND_SHARED_MEM_SIZE 196352
#define ASCEND_SHARED_DYN_MEM_SIZE 524032
#define ASCEND_WMMA_MATRIX_A_MEM_SIZE 65536
#define ASCEND_WMMA_MATRIX_B_MEM_SIZE 65536
#define ASCEND_WMMA_ACCUMULATOR_MEM_SIZE 131072

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

static constexpr const char *kAscendMemoryPlanning =
    "tl.ascend_memory_planning";

TVM_REGISTER_PASS_CONFIG_OPTION(kAscendMemoryPlanning, Bool);

static size_t AlignUp(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

class AscendMemoryPlanning : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f, PassContext ctx) {
    bool auto_ascend_memory_planning =
        ctx->GetConfig<Bool>(kAscendMemoryPlanning, Bool(false)).value();

    PrimFuncNode *fptr = f.CopyOnWrite();
    auto fn_attr = fptr->attrs.CopyOnWrite();

    Map<Var, PrimExpr> external_address_map;
    if (fn_attr->dict.count("address_map")) {
      external_address_map =
          fn_attr->dict.at("address_map").as<Map<Var, PrimExpr>>().value();
    }

    Map<Var, Array<PrimExpr>> external_shape_map;
    if (fn_attr->dict.count(kLogicBufferShapes)) {
      external_shape_map = fn_attr->dict.at(kLogicBufferShapes)
                               .as<Map<Var, Array<PrimExpr>>>()
                               .value();
    } else if (fn_attr->dict.count("buffer_shapess")) {
      external_shape_map = fn_attr->dict.at("buffer_shapess")
                               .as<Map<Var, Array<PrimExpr>>>()
                               .value();
    }

    AscendMemoryPlanner planner(f, external_address_map, external_shape_map,
                                auto_ascend_memory_planning);
    auto address_map = planner.GetAddressMap();
    auto buffer_sizes = planner.GetBufferSizes();

    Map<Var, PrimExpr> address_map_attr;
    for (const auto &kv : address_map) {
      Var buffer_var = GetRef<Var>(kv.first);
      address_map_attr.Set(buffer_var, Integer(kv.second));
    }
    fn_attr->dict.Set("address_map", address_map_attr);

    Map<Var, PrimExpr> size_map_attr;
    for (const auto &kv : buffer_sizes) {
      Var buffer_var = GetRef<Var>(kv.first);
      size_map_attr.Set(buffer_var, Integer(static_cast<int64_t>(kv.second)));
    }
    fn_attr->dict.Set("size_map", size_map_attr);
    return f;
  }

private:
  class AscendMemoryPlanner : public StmtExprVisitor {
  public:
    explicit AscendMemoryPlanner(const PrimFunc &func,
                                 Map<Var, PrimExpr> external_address_map,
                                 Map<Var, Array<PrimExpr>> external_shape_map,
                                 bool auto_plan = false) {
      memory_auto_plan = auto_plan;
      // Preserve layouts needed by CalculateBufferSize.
      // PTO 4D shapes are [physical_M, physical_N, valid_M, valid_N].
      external_shape_map_ = external_shape_map;
      memory_limits_ = {{"shared.l1", ASCEND_SHARED_DYN_MEM_SIZE},
                        {"wmma.matrix_a", ASCEND_WMMA_MATRIX_A_MEM_SIZE},
                        {"wmma.matrix_b", ASCEND_WMMA_MATRIX_B_MEM_SIZE},
                        {"wmma.accumulator", ASCEND_WMMA_ACCUMULATOR_MEM_SIZE},
                        {"shared.ub", ASCEND_SHARED_MEM_SIZE}};

      SetPreAllocBuffer(external_address_map);
      SetTmpBuffers(external_shape_map);

      operator()(func->body);
      // Most kernels have no narrow DMA. Bind ranges only when the indexed
      // copies need them, in the same preorder as the main traversal.
      if (!narrow_copies_.empty()) {
        for (const auto &stmt : copy_range_stmts_) {
          if (const auto *loop = stmt.as<ForNode>()) {
            copy_analyzer_.Bind(loop->loop_var,
                                Range::FromMinExtent(loop->min, loop->extent));
          } else if (const auto *let = stmt.as<LetStmtNode>()) {
            copy_analyzer_.Bind(let->var, let->value);
          } else if (const auto *attr = stmt.as<AttrStmtNode>()) {
            const auto *iter = attr->node.as<IterVarNode>();
            copy_analyzer_.Bind(iter->var,
                                Range::FromMinExtent(0, attr->value));
          }
        }
      }
      // Size copies after all relevant loops, including those inside Allocate.
      for (const auto *buf : origin_buffer) {
        buffer_sizes_[buf] = CalculateBufferSize(alloc_info_.at(buf).alloc);
        DLOG(DEBUG) << "Found NPU memory allocation: " << buf->name_hint
                    << " scope=" << buffer_scopes_.at(buf)
                    << " size=" << buffer_sizes_[buf] << " bytes";
      }
      PlanMemory();
    }

    const std::unordered_map<const VarNode *, int64_t> &GetAddressMap() const {
      return address_map_;
    }

    const std::unordered_map<const VarNode *, size_t> &GetBufferSizes() const {
      return buffer_sizes_;
    }

  private:
    struct StorageEntry {
      uint64_t const_nbits{0};
      std::vector<std::vector<const VarNode *>> allocs;
    };

    struct StmtEntry {
      const Object *stmt{};
      int64_t scope_pair_offset{0};
      std::vector<const VarNode *> touched;
    };

    struct EventEntry {
      std::vector<const VarNode *> gen;
      std::vector<const VarNode *> kill;
    };

    struct AllocEntry {
      size_t level{0};
      const AllocateNode *alloc{nullptr};
    };

    struct NarrowCopy {
      Call call;
      PrimExpr offset;
      int64_t columns;
      bool store;
    };

    struct StmtAttr {
      size_t level{0};
    };

    // Track a buffer touch for NPU shared memory.
    void TrackBufferTouch(const VarNode *buf) {
      auto it = alloc_info_.find(buf);
      if (it != alloc_info_.end() && it->second.alloc) {
        if (IsNPUSharedMemory(GetRef<Var>(buf))) {
          scope_.back().touched.push_back(buf);

          if (first_use_.count(buf) == 0) {
            first_use_[buf] = linear_seq_.size();
            DLOG(DEBUG) << "First use of buffer " << buf->name_hint
                        << " at statement index " << linear_seq_.size();
          }
        }
      }
    }

    void VisitStmt_(const AllocateNode *op) final {
      size_t level = scope_.size();
      const VarNode *buf = op->buffer_var.get();

      alloc_info_[buf].alloc = op;
      alloc_info_[buf].level = level;

      if (IsNPUSharedMemory(op->buffer_var)) {
        ICHECK(buffer_names_.find(buf->name_hint) == buffer_names_.end())
            << "Duplicate buffer name found: " << buf->name_hint
            << ". Please ensure all buffers have unique names.";
        buffer_names_.insert(buf->name_hint);

        std::string scope = GetPtrStorageScope(op->buffer_var);
        if (memory_limits_.count(scope)) {
          buffer_scopes_[buf] = scope;
          origin_buffer.push_back(buf);
        }
      }

      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const BufferStoreNode *op) final {
      scope_.push_back(StmtEntry());
      StmtExprVisitor::VisitStmt_(op);

      TrackBufferTouch(op->buffer->data.get());

      StmtEntry e = scope_.back();
      scope_.pop_back();
      if (!e.touched.empty()) {
        e.stmt = op;
        UpdateStmtAttr(op, scope_level_);
        linear_seq_.push_back(e);
      }
    }

    void VisitStmt_(const EvaluateNode *op) final {
      scope_.push_back(StmtEntry());
      StmtExprVisitor::VisitStmt_(op);

      StmtEntry e = scope_.back();
      scope_.pop_back();
      if (!e.touched.empty()) {
        e.stmt = op;
        UpdateStmtAttr(op, scope_level_);
        linear_seq_.push_back(e);
      }
    }

    void VisitExpr_(const BufferLoadNode *op) final {
      StmtExprVisitor::VisitExpr_(op);
      TrackBufferTouch(op->buffer->data.get());
    }

    void VisitExpr_(const VarNode *buf) final { TrackBufferTouch(buf); }

    void VisitExpr_(const CallNode *op) final {
      CollectNarrowCopy(op);
      if (op->op.same_as(builtin::tvm_access_ptr())) {
        Var buffer = Downcast<Var>(op->args[1]);
        if (IsNPUSharedMemory(buffer)) {
          TrackBufferTouch(buffer.get());
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }

    void CollectNarrowCopy(const CallNode *call) {
      if (!call->op.same_as(builtin::call_extern()) || call->args.size() < 6) {
        return;
      }
      const auto *name = call->args[0].as<StringImmNode>();
      if (name == nullptr) {
        return;
      }
      const std::string copy_name = name->value;
      const bool load = copy_name.rfind("tl::ascend::copy_gm_to_ub<", 0) == 0;
      const bool store =
          copy_name.rfind("tl::ascend::copy_ub_to_gm<", 0) == 0 ||
          copy_name.rfind("tl::ascend::atomic_add_ub_to_gm<", 0) == 0;
      if (!load && !store) {
        return;
      }
      const auto *ptr = call->args[load ? 2 : 1].as<CallNode>();
      if (ptr == nullptr || !ptr->op.same_as(builtin::tvm_access_ptr()) ||
          ptr->args.size() < 4) {
        return;
      }
      const auto *buffer = ptr->args[1].as<VarNode>();
      if (buffer == nullptr) {
        return;
      }
      const auto allocation = alloc_info_.find(buffer);
      if (allocation == alloc_info_.end() ||
          allocation->second.alloc == nullptr) {
        return;
      }
      if (!buffer_scopes_.count(buffer)) {
        return;
      }
      std::istringstream dims(copy_name.substr(copy_name.find('<') + 1));
      std::string dtype_token;
      std::getline(dims, dtype_token, ',');
      int64_t columns = 0;
      dims >> columns;
      const auto *alloc = allocation->second.alloc;
      const int64_t bits = alloc->dtype.bits() * alloc->dtype.lanes();
      if (!dims || columns <= 0 || columns >= (256 + bits - 1) / bits) {
        return;
      }
      // ExprVisitor revisits shared Call nodes; match PostOrderVisit's previous
      // once-per-node analysis, but only retain identities of narrow DMA calls.
      if (indexed_narrow_copies_.insert(call).second) {
        narrow_copies_[buffer].push_back(
            {GetRef<Call>(call), ptr->args[2], columns, store});
      }
    }

    // Cal Scope level and save linear stmt event
    template <typename T> void VisitNewScope(const T *op) {
      scope_.push_back(StmtEntry());
      StmtEntry e;
      e.stmt = op;
      UpdateStmtAttr(op, scope_level_);
      int64_t begin_index = static_cast<int64_t>(linear_seq_.size());

      linear_seq_.push_back(e);
      StmtExprVisitor::VisitStmt_(op);

      e.touched = std::move(scope_.back().touched);
      scope_.pop_back();
      int64_t end_index = static_cast<int64_t>(linear_seq_.size());

      e.scope_pair_offset = begin_index - end_index;
      linear_seq_.push_back(e);
      linear_seq_[begin_index].scope_pair_offset = end_index - begin_index;
    }

    void VisitStmt_(const AttrStmtNode *op) final {
      if (op->attr_key == tir::attr::thread_extent) {
        if (op->node.as<IterVarNode>()) {
          copy_range_stmts_.push_back(GetRef<Stmt>(op));
        }
      }
      VisitNewScope(op);
    }
    void VisitStmt_(const IfThenElseNode *op) final { VisitNewScope(op); }
    void VisitStmt_(const ForNode *op) final {
      copy_range_stmts_.push_back(GetRef<Stmt>(op));
      scope_level_++;
      VisitNewScope(op);
      scope_level_--;
    }
    void VisitStmt_(const WhileNode *op) final { VisitNewScope(op); }
    void VisitStmt_(const AssertStmtNode *op) final { VisitNewScope(op); }
    void VisitStmt_(const LetStmtNode *op) final {
      if (op->var.dtype().is_int() || op->var.dtype().is_uint()) {
        copy_range_stmts_.push_back(GetRef<Stmt>(op));
      }
      VisitNewScope(op);
    }

    void SetPreAllocBuffer(Map<Var, PrimExpr> external_address_map) {
      for (const auto &kv : external_address_map) {
        const VarNode *buf = kv.first.get();
        int64_t addr_offset = kv.second.as<IntImmNode>()->value;
        if (pre_alloc_buffer_.count(buf->name_hint)) {
          LOG(FATAL) << "Buffer " << buf->name_hint
                     << " already been allocated.";
        }
        pre_alloc_buffer_[buf->name_hint] = addr_offset;
      }
    }

    void SetTmpBuffers(Map<Var, Array<PrimExpr>> external_shape_map) {
      for (auto &kv : external_shape_map) {
        const VarNode *buf = kv.first.get();
        std::string scope = GetPtrStorageScope(kv.first);
        if (!pre_alloc_buffer_.count(buf->name_hint) && scope == "shared.ub") {
          tmp_buffers.push_back(buf);
        }
      }
    }

    void PlanMemory() {
      LivenessAnalysis();
      CollectLoopInfo();

      std::unordered_map<std::string, std::vector<const VarNode *>>
          scope_groups;
      for (const auto &kv : buffer_scopes_) {
        scope_groups[kv.second].push_back(kv.first);
      }

      DLOG(DEBUG) << "Memory planning by scope groups:";
      for (const auto &kv : scope_groups) {
        DLOG(DEBUG) << "  Scope " << kv.first << ": " << kv.second.size()
                    << " buffers";
      }

      for (const auto &scope_kv : scope_groups) {
        if (memory_auto_plan)
          PlanMemoryForScope(scope_kv.first, scope_kv.second);
        else
          PlanMemoryForScopeLinear(scope_kv.first, scope_kv.second);
      }
    }

    void LivenessAnalysis() {
      std::unordered_set<const VarNode *> touched;
      for (size_t i = linear_seq_.size(); i != 0; --i) {
        const StmtEntry &s = linear_seq_[i - 1];
        for (const VarNode *buffer : s.touched) {
          if (!touched.count(buffer)) {
            touched.insert(buffer);
            event_map_[s.stmt].kill.push_back(buffer);
          }
        }
      }

      for (size_t i = 0; i < linear_seq_.size(); ++i) {
        const StmtEntry &s = linear_seq_[i];
        for (const VarNode *buffer : s.touched) {
          if (first_use_.count(buffer) && first_use_[buffer] == i) {
            event_map_[s.stmt].gen.push_back(buffer);
          }
        }
      }

      ReorderKillPoints();

      DLOG(DEBUG) << "Liveness Analysis Results:";
      for (const auto &event_pair : event_map_) {
        const EventEntry &entry = event_pair.second;
        if (entry.gen.empty() && entry.kill.empty())
          continue;

        std::stringstream gen_ss, kill_ss;
        for (const VarNode *var : entry.gen)
          gen_ss << var->name_hint << " ";
        for (const VarNode *var : entry.kill)
          kill_ss << var->name_hint << " ";

        DLOG(DEBUG) << "  Statement: " << event_pair.first->GetTypeKey();
        if (!entry.gen.empty())
          DLOG(DEBUG) << "    GEN: " << gen_ss.str();
        if (!entry.kill.empty())
          DLOG(DEBUG) << "    KILL: " << kill_ss.str();
      }
    }

    void ReorderKillPoints() {
      std::vector<StmtEntry> gen_kill_seq;
      for (const auto &stmt_entry : linear_seq_) {
        if (!event_map_[stmt_entry.stmt].gen.empty() ||
            !event_map_[stmt_entry.stmt].kill.empty()) {
          gen_kill_seq.push_back(stmt_entry);
        }
      }

      for (auto &event_pair : event_map_) {
        const Object *stmt = event_pair.first;
        EventEntry &event = event_pair.second;

        if (event.kill.empty())
          continue;

        ICHECK(stmt_attrs_.count(stmt));
        int kill_level = stmt_attrs_.at(stmt).level;

        std::unordered_set<const VarNode *> visited_buffers;

        for (auto it = event.kill.begin(); it != event.kill.end();) {
          const VarNode *buffer = *it;
          bool found_gen = false;
          int gen_level = 0;

          for (const auto &gen_pair : event_map_) {
            const auto &gen_event = gen_pair.second;
            if (std::find(gen_event.gen.begin(), gen_event.gen.end(), buffer) !=
                gen_event.gen.end()) {
              found_gen = true;
              gen_level = stmt_attrs_.at(gen_pair.first).level;
              break;
            }
          }

          if (found_gen && kill_level > gen_level) {
            if (visited_buffers.count(buffer)) {
              ++it;
              continue;
            }

            it = event.kill.erase(it);

            const Object *last_stmt_at_level = nullptr;
            auto stmt_it = gen_kill_seq.begin();
            for (; stmt_it != gen_kill_seq.end(); ++stmt_it) {
              if (stmt_it->stmt == stmt) {
                break;
              }
            }

            for (; stmt_it != gen_kill_seq.end(); ++stmt_it) {
              auto next_it = stmt_it + 1;
              if (next_it == gen_kill_seq.end() ||
                  stmt_attrs_.at(next_it->stmt).level == gen_level - 1) {
                last_stmt_at_level = stmt_it->stmt;
                break;
              }
            }

            if (last_stmt_at_level) {
              event_map_[last_stmt_at_level].kill.push_back(buffer);
              visited_buffers.insert(buffer);
            }
          } else {
            ++it;
          }
        }
      }
    }

    int64_t FindEventIndex(const VarNode *buffer, bool use_gen) {
      for (const auto &event_pair : event_map_) {
        const EventEntry &event = event_pair.second;
        const auto &vec = use_gen ? event.gen : event.kill;
        auto it = std::find(vec.begin(), vec.end(), buffer);
        if (it != vec.end()) {
          for (size_t i = 0; i < linear_seq_.size(); ++i) {
            if (linear_seq_[i].stmt == event_pair.first) {
              return static_cast<int64_t>(i);
            }
          }
          break;
        }
      }
      return -1;
    }

    /*!
     * \brief Collect loop intervals and buffer touch indices for
     *        loop-aware liveness extension.
     *
     * After LivenessAnalysis, the KILL point of a buffer is set to its
     * last touch in the linearized IR.  However, when a buffer is
     * defined *before* a loop and touched *inside* the loop body, it
     * is actually live across every iteration — its real KILL should
     * be the loop's end, not the textual last touch inside the body.
     *
     * This method pre-computes:
     *   1. loop_intervals_  — [begin, end] index pairs for every
     *      ForNode / WhileNode in linear_seq_.
     *   2. buffer_touch_indices_ — for every buffer, a sorted set of
     *      all linear_seq_ indices where it is touched.
     */
    void CollectLoopInfo() {
      for (size_t i = 0; i < linear_seq_.size(); ++i) {
        const StmtEntry &s = linear_seq_[i];
        if (s.scope_pair_offset > 0 && s.stmt &&
            (s.stmt->IsInstance<ForNode>() ||
             s.stmt->IsInstance<WhileNode>())) {
          int64_t begin = static_cast<int64_t>(i);
          int64_t end = begin + s.scope_pair_offset;
          loop_intervals_.push_back({begin, end});
        }
      }

      for (size_t i = 0; i < linear_seq_.size(); ++i) {
        for (const VarNode *buf : linear_seq_[i].touched) {
          buffer_touch_indices_[buf].insert(static_cast<int64_t>(i));
        }
      }
    }

    /*!
     * \brief Extend a buffer's KILL index to cover loops in which it
     *        is live across iterations.
     *
     * The linearized IR represents a loop body as a single pass, so the
     * liveness analysis computes a buffer's KILL as the last touch
     * *within one iteration*.  However, when a buffer is defined
     * *before* a loop and touched *inside* the loop body, it is
     * actually live across every iteration — its real KILL should be
     * the loop's end, not the textual last touch inside the body.
     *
     * This handles the case where a buffer (e.g. index_ub) is
     * initialized once before a segment loop and read in every
     * iteration.  Without this extension, the planner may reuse the
     * buffer's memory for another buffer (e.g. merged_ub) that is
     * written inside the same loop, causing silent data corruption.
     *
     * For each loop [loop_begin, loop_end] where:
     *   - GEN < loop_begin  (buffer defined before the loop)
     *   - loop_begin <= KILL < loop_end  (buffer's last touch is inside)
     * the KILL is extended to loop_end.  This applies to all enclosing
     * loops, so we take the maximum loop_end.
     */
    int64_t ExtendKillIndex(const VarNode *buffer, int64_t gen_idx,
                            int64_t kill_idx) {
      if (kill_idx < 0) {
        return kill_idx;
      }
      auto it = buffer_touch_indices_.find(buffer);
      if (it == buffer_touch_indices_.end()) {
        return kill_idx;
      }
      const std::set<int64_t> &touches = it->second;

      int64_t extended = kill_idx;
      for (const auto &loop : loop_intervals_) {
        int64_t loop_begin = loop.first;
        int64_t loop_end = loop.second;

        if (gen_idx < loop_begin && kill_idx >= loop_begin &&
            kill_idx < loop_end) {
          auto touch_it = touches.upper_bound(loop_begin);
          if (touch_it != touches.end() && *touch_it < loop_end) {
            extended = std::max(extended, loop_end);
          }
        }
      }
      return extended;
    }

    void PlanMemoryForScope(const std::string &scope,
                            const std::vector<const VarNode *> &buffers) {
      DLOG(DEBUG) << "Planning memory for scope: " << scope;

      std::vector<LiveInterval> intervals;
      std::unordered_map<const VarNode *, int64_t> pre_alloc_scope_buffer;
      for (const VarNode *buffer : buffers) {
        if (pre_alloc_buffer_.count(buffer->name_hint) > 0) {
          pre_alloc_scope_buffer[buffer] = pre_alloc_buffer_[buffer->name_hint];
        };

        int64_t start = FindEventIndex(buffer, true);
        int64_t end = FindEventIndex(buffer, false);

        if (start != -1 && end != -1) {
          end = ExtendKillIndex(buffer, start, end);
          intervals.emplace_back(buffer, start, end, buffer_sizes_[buffer]);
          DLOG(DEBUG) << "Buffer " << buffer->name_hint << ": [" << start
                      << ", " << end << "], size=" << buffer_sizes_[buffer];
        }
      }

      std::sort(intervals.begin(), intervals.end(),
                [](const LiveInterval &a, const LiveInterval &b) {
                  return a.start < b.start;
                });

      LinearScanAllocator allocator(memory_limits_[scope],
                                    pre_alloc_scope_buffer);
      auto allocations = allocator.allocate(intervals);

      for (const auto &alloc : allocations) {
        address_map_[alloc.buffer] = alloc.offset;
        DLOG(DEBUG) << "Allocated buffer " << alloc.buffer->name_hint
                    << " at offset " << alloc.offset << " (size=" << alloc.size
                    << ")";
      }

      size_t total_used = 0;
      for (const auto &alloc : allocations) {
        total_used = std::max(total_used, alloc.offset + alloc.size);
      }

      DLOG(DEBUG) << "Scope " << scope << " memory usage: " << total_used << "/"
                  << memory_limits_[scope] << " bytes ("
                  << (total_used * 100.0 / memory_limits_[scope]) << "%)";
      if (total_used > memory_limits_[scope]) {
        DLOG(WARNING) << "Memory limit exceeded for scope " << scope << ": "
                      << total_used << " > " << memory_limits_[scope];
      }
    }

    void PlanMemoryForScopeLinear(const std::string &scope,
                                  const std::vector<const VarNode *> &buffers) {
      bool check_overflow = false; // reserve memory overflow check
      int64_t current_offset = 0;
      int64_t max_offset = 0;

      // Allocate origin buffer first, then tmp buffer in shared memory
      auto alloc_buffer = [&](const VarNode *buffer, int64_t &offset,
                              const std::string &err_prefix) -> bool {
        int64_t buf_size = buffer_sizes_[buffer];
        if (offset + buf_size > memory_limits_[scope] && check_overflow) {
          LOG(FATAL) << err_prefix << " Out of memory in scope: " << scope
                     << "\nBuffer: " << buffer->name_hint
                     << "\nRequired size: " << buf_size
                     << "\nCurrent offset: " << offset
                     << "\nMemory limit: " << memory_limits_[scope];
          return false;
        }
        address_map_[buffer] = offset;
        offset = static_cast<int64_t>(AlignUp(offset + buf_size, 32));
        return true;
      };

      for (const VarNode *buffer : origin_buffer) {
        if (std::find(buffers.begin(), buffers.end(), buffer) ==
            buffers.end()) {
          continue;
        }
        if (pre_alloc_buffer_.count(buffer->name_hint)) {
          address_map_[buffer] = pre_alloc_buffer_[buffer->name_hint];
          max_offset = std::max(
              max_offset,
              static_cast<int64_t>(pre_alloc_buffer_[buffer->name_hint] +
                                   buffer_sizes_[buffer]));
        } else if (scope != "shared.ub") {
          alloc_buffer(buffer, current_offset,
                       "Linear memory allocation failed!");
          max_offset = std::max(max_offset, current_offset);
        }
      }
      // Allocate tmp buffer/default buffer after origin buffer to avoid
      // fragmention in shared memory
      if (scope == "shared.ub") {
        for (const VarNode *buffer : origin_buffer) {
          if (std::find(tmp_buffers.begin(), tmp_buffers.end(), buffer) !=
                  tmp_buffers.end() &&
              !address_map_.count(buffer)) {
            alloc_buffer(buffer, max_offset,
                         "Linear tmp memory allocation failed!");
          }
        }
      }
    }

    struct LiveInterval {
      const VarNode *buffer;
      int64_t start;
      int64_t end;
      size_t size;

      LiveInterval(const VarNode *buf, int64_t s, int64_t e, size_t sz)
          : buffer(buf), start(s), end(e), size(sz) {}
    };

    struct Allocation {
      const VarNode *buffer;
      size_t offset;
      size_t size;
      bool is_reused;
    };

    class LinearScanAllocator {
    public:
      LinearScanAllocator(
          size_t memory_limit,
          const std::unordered_map<const VarNode *, int64_t> &pre_alloc_buffer)
          : memory_limit_(memory_limit), next_new_offset_(0),
            pre_alloc_buffer_(pre_alloc_buffer) {}

      std::vector<Allocation> allocate(std::vector<LiveInterval> &intervals) {
        std::vector<Allocation> allocations;

        for (auto &interval : intervals)
          buffer_map_[interval.buffer] = &interval;

        std::sort(intervals.begin(), intervals.end(),
                  [](const LiveInterval &a, const LiveInterval &b) {
                    return a.start < b.start;
                  });

        auto end_time_compare = [](const LiveInterval &a,
                                   const LiveInterval &b) {
          return a.end > b.end;
        };
        std::priority_queue<LiveInterval, std::vector<LiveInterval>,
                            decltype(end_time_compare)>
            active_queue(end_time_compare);

        std::vector<Allocation> active_allocations;
        std::vector<std::pair<size_t, size_t>> free_blocks;

        for (auto &interval : intervals) {
          DLOG(DEBUG) << "Processing: " << interval.buffer->name_hint << " ["
                      << interval.start << ", " << interval.end
                      << "] size=" << interval.size;

          while (!active_queue.empty() &&
                 active_queue.top().end <
                     interval.start) { // Free unactivate buffer
            const auto &expired = active_queue.top();

            auto it = std::find_if(active_allocations.begin(),
                                   active_allocations.end(),
                                   [&](const Allocation &alloc) {
                                     return alloc.buffer == expired.buffer;
                                   });
            if (it != active_allocations.end()) {
              free_blocks.emplace_back(it->offset, it->size);
              active_allocations.erase(it);

              DLOG(DEBUG) << "  Released for reuse: "
                          << expired.buffer->name_hint << " at offset "
                          << it->offset;
            }

            active_queue.pop();
          }

          mergeFreeBlocks(free_blocks);

          size_t allocated_offset;
          bool is_reused = false;

          auto pre_it = pre_alloc_buffer_.find(interval.buffer);
          if (pre_it != pre_alloc_buffer_.end()) { // Pre alloc
            allocated_offset = AlignUp(static_cast<size_t>(pre_it->second), 32);
            size_t allocated_end = allocated_offset + interval.size;

            for (const auto &active : active_allocations) {
              if (allocated_offset < active.offset + active.size &&
                  active.offset < allocated_offset + interval.size) {
                LOG(FATAL) << "Memory allocation failed for: "
                           << pre_it->first->name_hint
                           << " memory allocate conflict with: "
                           << interval.buffer->name_hint << " at "
                           << allocated_offset;
                continue;
              }
            }

            if (allocated_offset > next_new_offset_) {
              free_blocks.emplace_back(next_new_offset_,
                                       allocated_offset - next_new_offset_);
            }

            if (allocated_end > next_new_offset_) {
              next_new_offset_ = allocated_end;
            }

          } else { // Normal Alloc
            size_t new_memory_offset = AlignUp(next_new_offset_, 32);
            if (new_memory_offset + interval.size <= memory_limit_ &&
                !CheckConflict(new_memory_offset, interval)) {
              allocated_offset = new_memory_offset;
              next_new_offset_ = new_memory_offset + interval.size;
            } else {
              allocated_offset = findReusableBlock(interval, free_blocks);
              if (allocated_offset != static_cast<size_t>(-1)) {
                is_reused = true;
                DLOG(DEBUG) << "REUSED memory at offset: " << allocated_offset;
              } else {
                LOG(FATAL) << "Memory allocation failed for: "
                           << interval.buffer->name_hint
                           << " required: " << interval.size
                           << ", new memory available: "
                           << (memory_limit_ - next_new_offset_);
                continue;
              }
            }
          }

          Allocation alloc{interval.buffer, allocated_offset, interval.size,
                           is_reused};
          allocations.push_back(alloc);
          active_allocations.push_back(alloc);
          active_queue.push(interval);

          if (is_reused) {
            removeFromFreeBlocks(allocated_offset, interval.size, free_blocks);
          }
        }

        size_t total_used = next_new_offset_;
        size_t reused_count =
            std::count_if(allocations.begin(), allocations.end(),
                          [](const Allocation &a) { return a.is_reused; });
        DLOG(DEBUG) << "Memory usage: " << total_used << "/" << memory_limit_
                    << " bytes (" << (total_used * 100.0 / memory_limit_)
                    << "%)";
        DLOG(DEBUG) << "Reused buffers: " << reused_count << "/"
                    << allocations.size();

        return allocations;
      }

    private:
      bool CheckConflict(size_t offset, const LiveInterval &current) {
        for (const auto &kv : pre_alloc_buffer_) {
          const LiveInterval *pre_interval = buffer_map_.at(kv.first);
          if (pre_interval->buffer == current.buffer)
            continue;

          if (current.start < pre_interval->end &&
              pre_interval->start < current.end) {
            size_t pre_offset = static_cast<size_t>(kv.second);
            size_t pre_size = pre_interval->size;
            if (offset < pre_offset + pre_size &&
                pre_offset < offset + current.size) {
              DLOG(DEBUG) << "Buffer " << kv.first->name_hint << " conflict at "
                          << pre_offset;
              return true;
            }
          }
        }
        return false;
      }

      void
      mergeFreeBlocks(std::vector<std::pair<size_t, size_t>> &free_blocks) {
        if (free_blocks.empty())
          return;

        std::sort(free_blocks.begin(), free_blocks.end());

        std::vector<std::pair<size_t, size_t>> merged;
        merged.push_back(free_blocks[0]);

        for (size_t i = 1; i < free_blocks.size(); ++i) {
          auto &last = merged.back();
          auto &current = free_blocks[i];

          if (last.first + last.second >= current.first) {
            size_t new_end = std::max(last.first + last.second,
                                      current.first + current.second);
            last.second = new_end - last.first;
          } else {
            merged.push_back(current);
          }
        }

        free_blocks = std::move(merged);
      }

      size_t
      findReusableBlock(const LiveInterval &current,
                        std::vector<std::pair<size_t, size_t>> &free_blocks) {

        std::sort(free_blocks.begin(), free_blocks.end());

        for (const auto &block : free_blocks) {
          if (block.second >= current.size) {
            size_t aligned_offset = AlignUp(block.first, 32);
            size_t available_after_align =
                block.second - (aligned_offset - block.first);

            if (available_after_align >= current.size &&
                !CheckConflict(aligned_offset, current)) {
              return aligned_offset;
            }
          }
        }

        auto &last_block = free_blocks[free_blocks.size() - 1];
        if ((last_block.first + last_block.second) == next_new_offset_) {
          next_new_offset_ = memory_limit_;
          last_block.second = memory_limit_ - last_block.first;
          if (last_block.second >= current.size) {
            size_t aligned_offset = AlignUp(last_block.first, 32);
            size_t available_after_align =
                last_block.second - (aligned_offset - last_block.first);

            if (available_after_align >= current.size &&
                !CheckConflict(aligned_offset, current)) {
              return aligned_offset;
            }
          }
        }
        return -1;
      }

      void removeFromFreeBlocks(
          size_t offset, size_t size,
          std::vector<std::pair<size_t, size_t>> &free_blocks) {
        auto it = free_blocks.begin();
        while (it != free_blocks.end()) {
          if (offset >= it->first && offset < it->first + it->second) {
            size_t block_start = it->first;
            size_t block_size = it->second;
            size_t allocated_end = offset + size;
            size_t block_end = block_start + block_size;

            it = free_blocks.erase(it);

            if (offset > block_start) {
              free_blocks.emplace_back(block_start, offset - block_start);
            }

            if (allocated_end < block_end) {
              free_blocks.emplace_back(allocated_end,
                                       block_end - allocated_end);
            }

            break;
          } else {
            ++it;
          }
        }
      }

      size_t memory_limit_;
      size_t next_new_offset_;
      const std::unordered_map<const VarNode *, int64_t> &pre_alloc_buffer_;
      std::unordered_map<const VarNode *, const LiveInterval *> buffer_map_;
    };

    // compute_valid_extent lowers a runtime tail to nested Select nodes.
    // Refine each arm by its condition so the static tile bound (rather than
    // the unrestricted symbolic tensor dimension) determines the footprint.
    static std::pair<int64_t, int64_t>
    NarrowCopyBounds(const PrimExpr &expr, arith::Analyzer *analyzer) {
      if (const auto *select = expr.as<SelectNode>()) {
        // ConstIntBound recognizes comparisons, not an unnormalized Not.
        PrimExpr condition = analyzer->Simplify(select->condition);
        PrimExpr negated_condition = analyzer->Simplify(!select->condition);
        if (analyzer->CanProve(condition)) {
          return NarrowCopyBounds(select->true_value, analyzer);
        }
        if (analyzer->CanProve(negated_condition)) {
          return NarrowCopyBounds(select->false_value, analyzer);
        }
        std::pair<int64_t, int64_t> true_bounds, false_bounds;
        {
          With<arith::ConstraintContext> context(analyzer, condition);
          true_bounds = NarrowCopyBounds(select->true_value, analyzer);
        }
        {
          With<arith::ConstraintContext> context(analyzer, negated_condition);
          false_bounds = NarrowCopyBounds(select->false_value, analyzer);
        }
        return {std::min(true_bounds.first, false_bounds.first),
                std::max(true_bounds.second, false_bounds.second)};
      }
      auto bound = analyzer->const_int_bound(analyzer->Simplify(expr));
      return {bound->min_value, bound->max_value};
    }

    std::pair<int64_t, int64_t> GetNarrowCopyBounds(const PrimExpr &expr) {
      auto it = copy_bounds_.find(expr.get());
      if (it != copy_bounds_.end()) {
        return it->second;
      }
      // Cache only global post-Bind queries, never recursive Select contexts.
      auto bounds = NarrowCopyBounds(expr, &copy_analyzer_);
      copy_bounds_.emplace(expr.get(), bounds);
      return bounds;
    }

    size_t CalculateBufferSize(const AllocateNode *alloc) {
      size_t size_elements = 1;
      auto shape_it = external_shape_map_.find(alloc->buffer_var);
      if (shape_it != external_shape_map_.end() &&
          (*shape_it).second.size() == 4) {
        const auto &shape = (*shape_it).second;
        const IntImmNode *row = shape[0].as<IntImmNode>();
        const IntImmNode *col = shape[1].as<IntImmNode>();
        ICHECK(row && col) << "PTO physical buffer shape must be constant";
        size_elements = row->value * col->value;
      } else {
        for (const auto &extent : alloc->extents) {
          const IntImmNode *int_imm = extent.as<IntImmNode>();
          ICHECK(int_imm) << "Extent must be an integer constant";
          size_elements *= int_imm->value;
        }
      }

      // Sub-byte dtypes (e.g. int4) are nibble-packed in the AscendC memory
      // model: two elements share one byte. DataType::bytes() rounds up to 1
      // byte per element, which would over-allocate such buffers by 2x, so
      // compute the byte footprint from the exact bit width.
      size_t size_bits =
          size_elements * alloc->dtype.bits() * alloc->dtype.lanes();
      size_t size_bytes = (size_bits + 7) / 8;
      // A packed compare Buffer is logically [rows, ceil(cols/8)], but the
      // AscendC vector/MTE contracts consume one 32-byte UB data block per
      // predicate row.  Preserve the logical shape used by access_ptr and GM
      // copies while reserving the larger physical backing store.
      if (alloc->dtype == DataType::UInt(8)) {
        size_t aligned_cmp_bytes = 0;
        PostOrderVisit(alloc->body, [&](const ObjectRef &node) {
          const auto *call = node.as<CallNode>();
          if (call == nullptr ||
              (!call->op.same_as(ascend_tail_compare()) &&
               !call->op.same_as(ascend_tail_compare_scalar())) ||
              call->args.size() != 9) {
            return;
          }
          const auto *ptr = call->args[0].as<CallNode>();
          if (ptr == nullptr || !ptr->op.same_as(builtin::tvm_access_ptr()) ||
              ptr->args.size() < 2 ||
              ptr->args[1].as<VarNode>() != alloc->buffer_var.get()) {
            return;
          }
          const auto *rows = call->args[6].as<IntImmNode>();
          ICHECK(rows) << "tail compare physical rows must be static";
          aligned_cmp_bytes = std::max(aligned_cmp_bytes,
                                       static_cast<size_t>(rows->value) * 32);
        });
        size_bytes = std::max(size_bytes, aligned_cmp_bytes);
      }
      // A tail row-broadcast source such as [rows, 1] has one 32-byte UB block
      // per GM->UB burst. Preserve the logical shape used by access_ptr while
      // reserving the complete physical row-padded backing store.
      size_t padded_broadcast_bytes = 0;
      PostOrderVisit(alloc->body, [&](const ObjectRef &node) {
        const auto *call = node.as<CallNode>();
        if (call == nullptr || !call->op.same_as(ascend_tail_broadcast()) ||
            (call->args.size() != 12 && call->args.size() != 13)) {
          return;
        }
        const auto *ptr = call->args[2].as<CallNode>();
        if (ptr == nullptr || !ptr->op.same_as(builtin::tvm_access_ptr()) ||
            ptr->args.size() < 2 ||
            ptr->args[1].as<VarNode>() != alloc->buffer_var.get()) {
          return;
        }
        const bool has_tmp = call->args[3].as<CallNode>() != nullptr;
        const size_t dim_index = has_tmp ? 4 : 3;
        const size_t shape_index = dim_index + 1;
        const auto *rows = call->args[shape_index + 2].as<IntImmNode>();
        const auto *cols = call->args[shape_index + 3].as<IntImmNode>();
        ICHECK(rows && cols) << "tail broadcast source shape must be static";
        // Bit-exact row footprint (see the sub-byte note above); identical to
        // cols * dtype.bytes() for every non-sub-byte dtype.
        const size_t row_bits = static_cast<size_t>(cols->value) *
                                alloc->dtype.bits() * alloc->dtype.lanes();
        const size_t row_bytes = (row_bits + 7) / 8;
        if (rows->value > 0 && row_bytes < 32) {
          padded_broadcast_bytes = std::max(
              padded_broadcast_bytes, static_cast<size_t>(rows->value) * 32);
        }
      });
      size_bytes = std::max(size_bytes, padded_broadcast_bytes);
      // Narrow DataCopyPad rows occupy a full 32-byte UB slot. Keep logical
      // extents unchanged and reserve the actual DMA footprint, including a
      // subregion's offset. Aligned copies and packed UB->GM single bursts
      // retain their existing allocation and behavior.
      auto &copy_analyzer = copy_analyzer_;
      const auto copies = narrow_copies_.find(alloc->buffer_var.get());
      if (copies != narrow_copies_.end()) {
        for (const auto &copy : copies->second) {
          const auto *call = copy.call.get();
          const int64_t columns = copy.columns;
          const int64_t bits = alloc->dtype.bits() * alloc->dtype.lanes();
          const auto row_bounds = GetNarrowCopyBounds(call->args[4]);
          const auto col_bounds = GetNarrowCopyBounds(call->args[5]);
          PrimExpr offset = copy.offset;
          if (const auto *ramp = offset.as<RampNode>()) {
            offset = ramp->base; // CopyCodegen uses the scalar Ramp base.
          }
          const auto offset_bounds = GetNarrowCopyBounds(offset);
          // Only expand a footprint with a proven, finite bound inside the
          // logical region. Do not introduce new rejection of unrelated dynamic
          // copies whose address bounds this local analysis cannot establish.
          if (row_bounds.first < 0 || row_bounds.second <= 1 ||
              row_bounds.second > static_cast<int64_t>(size_elements) ||
              col_bounds.first < 0 || col_bounds.second > columns ||
              offset_bounds.first < 0 ||
              offset_bounds.second > static_cast<int64_t>(size_elements)) {
            continue;
          }
          const int64_t block_bytes = col_bounds.second * bits / 8;
          if (block_bytes == 0) {
            continue;
          }
          if (copy.store && col_bounds.first == columns &&
              col_bounds.second == columns &&
              copy_analyzer.CanProve(call->args[3] == call->args[5]) &&
              row_bounds.second <= 65535 / block_bytes) {
            continue; // copy_ub_to_gm uses one flat burst, not row slots.
          }
          const size_t offset_bytes =
              static_cast<size_t>(offset_bounds.second) * bits / 8;
          // blockLen < 32B and UB stride == 0: ceil(blockLen/32) + stride
          // is exactly one 32-byte slot per row, for every supported narrow
          // width.
          size_t footprint =
              offset_bytes + static_cast<size_t>(row_bounds.second) * 32;
          if (offset_bounds.first != offset_bounds.second &&
              row_bounds.first != row_bounds.second) {
            // Bound the complete end address to preserve offset/row
            // correlation. Keep the separate bounds as a fallback for
            // conditional tails.
            PrimExpr end = floordiv(cast(DataType::Int(64), offset) *
                                        make_const(DataType::Int(64), bits),
                                    8) +
                           cast(DataType::Int(64), call->args[4]) * 32;
            // Do not cache a temporary expression by its node address.
            const auto end_bounds =
                NarrowCopyBounds(copy_analyzer.Simplify(end), &copy_analyzer);
            if (end_bounds.second >= 0) {
              footprint =
                  std::min(footprint, static_cast<size_t>(end_bounds.second));
            }
          }
          ICHECK_LE(footprint, static_cast<size_t>(ASCEND_SHARED_MEM_SIZE))
              << "UB copy physical footprint exceeds the available shared.ub "
                 "memory";
          size_bytes = std::max(size_bytes, footprint);
        }
      }
      return AlignUp(size_bytes, 32);
    }

    void UpdateStmtAttr(const Object *stmt, size_t level) {
      stmt_attrs_[stmt] = StmtAttr{level};
    }

    bool IsNPUSharedMemory(Var buffer_var) {
      std::string scope = GetPtrStorageScope(buffer_var);
      return memory_limits_.count(scope) > 0;
    }

    std::unordered_map<const VarNode *, AllocEntry>
        alloc_info_; // buffer allocation and level
    std::unordered_map<const VarNode *, int64_t>
        address_map_; // buffer address map
    std::unordered_map<const VarNode *, std::string>
        buffer_scopes_; // buffer scope(UB/L1..)
    std::unordered_map<const VarNode *, size_t>
        buffer_sizes_; // buffer bytes size
    arith::Analyzer copy_analyzer_;
    std::unordered_map<const PrimExprNode *, std::pair<int64_t, int64_t>>
        copy_bounds_;
    std::vector<Stmt> copy_range_stmts_;
    std::unordered_map<const VarNode *, std::vector<NarrowCopy>> narrow_copies_;
    std::unordered_set<const CallNode *> indexed_narrow_copies_;
    // Layout map used to size PTO 4D tiles by physical footprint.
    Map<Var, Array<PrimExpr>> external_shape_map_;
    std::unordered_map<const VarNode *, size_t>
        first_use_; // buffer first use stmt scope
    std::unordered_map<std::string, int>
        memory_limits_; // buffer scope max limits
    std::unordered_map<const Object *, StmtAttr>
        stmt_attrs_; // stmt operation level
    std::unordered_map<const Object *, EventEntry>
        event_map_; // stmt gen/kill event
    std::unordered_map<std::string, int64_t>
        pre_alloc_buffer_;              // pre alloction buffer address map
    std::vector<StmtEntry> linear_seq_; // linear stmt node scopes and levels
    std::vector<StmtEntry> scope_;      // temp stmt node scopes and levels
    std::vector<const VarNode *> origin_buffer; // original buffer list
    std::unordered_set<std::string>
        buffer_names_; // buffer names for duplicate check
    std::vector<const VarNode *>
        tmp_buffers; // temp buffer list for memory planning
    std::vector<std::pair<int64_t, int64_t>>
        loop_intervals_; // [begin, end] index pairs for each loop
    std::unordered_map<const VarNode *, std::set<int64_t>>
        buffer_touch_indices_; // buffer -> sorted touch indices

    size_t scope_level_{0};
    bool memory_auto_plan{false};
  };
};

tvm::transform::Pass AscendMemoryPlanning() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto new_func = AscendMemoryPlanning::Substitute(std::move(f), ctx);
    return new_func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendMemoryPlanning", {});
}

TVM_REGISTER_GLOBAL("tl.transform.AscendMemoryPlanning")
    .set_body_typed(AscendMemoryPlanning);

} // namespace tl
} // namespace tvm
