// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file ascend_memory_planning.cc
 * \brief Memory planning for Ascend NPU
 */

#include <iostream>
#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <stack>

#include "arith/ir_mutator_with_analyzer.h"
#include "tir/analysis/var_use_def_analysis.h"
#include "tir/transforms/ir_utils.h"

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

static constexpr const char *kAscendMemoryPlanning = "tl.ascend_memory_planning";

TVM_REGISTER_PASS_CONFIG_OPTION(kAscendMemoryPlanning, Bool);

class AscendMemoryPlanning : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f, PassContext ctx) {
    bool ascend_memory_planning = ctx->GetConfig<Bool>(kAscendMemoryPlanning, Bool(false)).value();
    if (!ascend_memory_planning) {
      return f;
    }

    AscendMemoryPlanner planner(f);
    auto address_map = planner.GetAddressMap();

    PrimFuncNode *fptr = f.CopyOnWrite();
    auto fn_attr = fptr->attrs.CopyOnWrite();

    Map<Var, PrimExpr> address_map_attr;
    for (const auto& kv : address_map) {
      Var buffer_var = GetRef<Var>(kv.first);
      address_map_attr.Set(buffer_var, make_const(DataType::Int(64), kv.second));
    }
    fn_attr->dict.Set("address_map", address_map_attr);
    return f;
  }

private:
  class AscendMemoryPlanner : public StmtExprVisitor {
  public:
    explicit AscendMemoryPlanner(const PrimFunc& func) 
    {
      memory_limits_ = {
        {"shared.dyn", 524032},
        {"wmma.matrix_a", 65536},
        {"wmma.matrix_b", 131072},  
        {"wmma.accumulator", 131072},
        {"shared", 196352}
      };
      
      operator()(func->body);
      PlanMemory();
    }

    const std::unordered_map<const VarNode*, int64_t>& GetAddressMap() const {
      return address_map_;
    }

  private:
    struct StorageEntry {
      uint64_t const_nbits{0};
      std::vector<std::vector<const VarNode*>> allocs;
    };

    struct StmtEntry {
      const Object* stmt{};
      int64_t scope_pair_offset{0};
      std::vector<const VarNode*> touched;
    };

    struct EventEntry {
      std::vector<const VarNode*> gen;
      std::vector<const VarNode*> kill;
    };

    struct AllocEntry {
      size_t level{0};
      const AllocateNode* alloc{nullptr};
    };

    struct StmtAttr {
      size_t level{0};
    };

    void VisitStmt_(const AllocateNode* op) final {
      size_t level = scope_.size();
      const VarNode* buf = op->buffer_var.get();
      
      alloc_info_[buf].alloc = op;
      alloc_info_[buf].level = level;

      if (IsNPUSharedMemory(op->buffer_var)) {
        std::string scope = GetPtrStorageScope(op->buffer_var);
        if (memory_limits_.count(scope)) {
          buffer_scopes_[buf] = scope;
          buffer_sizes_[buf] = CalculateBufferSize(op);

          DLOG(DEBUG) << "Found NPU memory allocation: " << op->buffer_var->name_hint
                    << " scope=" << scope << " size=" << buffer_sizes_[buf] << " bytes";
        }
      }

      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const BufferStoreNode* op) final {
      scope_.push_back(StmtEntry());
      StmtExprVisitor::VisitStmt_(op);
      
      const VarNode* buf = op->buffer->data.get();
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

      StmtEntry e = scope_.back();
      scope_.pop_back();
      if (!e.touched.empty()) {
        e.stmt = op;
        UpdateStmtAttr(op, scope_level_);
        linear_seq_.push_back(e);
      }
    }

    void VisitStmt_(const EvaluateNode* op) final {
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

    void VisitExpr_(const BufferLoadNode* op) final {
      StmtExprVisitor::VisitExpr_(op);
      
      const VarNode* buf = op->buffer->data.get();
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

    void VisitExpr_(const VarNode* buf) final {
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

    void VisitExpr_(const CallNode* op) final {
      if (op->op.same_as(builtin::tvm_access_ptr())) {
        Var buffer = Downcast<Var>(op->args[1]);
        if (IsNPUSharedMemory(buffer)) {
          const VarNode* buf = buffer.get();
          auto it = alloc_info_.find(buf);
          if (it != alloc_info_.end() && it->second.alloc) {
            scope_.back().touched.push_back(buf);
            
            if (first_use_.count(buf) == 0) {
              first_use_[buf] = linear_seq_.size();
              DLOG(DEBUG) << "First use of buffer " << buf->name_hint 
                        << " at statement index " << linear_seq_.size();
            }
          }
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }

    template <typename T>
    void VisitNewScope(const T* op) {
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

    void VisitStmt_(const AttrStmtNode* op) final { VisitNewScope(op); }
    void VisitStmt_(const IfThenElseNode* op) final { VisitNewScope(op); }
    void VisitStmt_(const ForNode* op) final {
      scope_level_++;
      VisitNewScope(op);
      scope_level_--;
    }
    void VisitStmt_(const WhileNode* op) final { VisitNewScope(op); }
    void VisitStmt_(const AssertStmtNode* op) final { VisitNewScope(op); }

    void PlanMemory() {
      LivenessAnalysis();
      
      std::unordered_map<std::string, std::vector<const VarNode*>> scope_groups;
      for (const auto& kv : buffer_scopes_) {
        scope_groups[kv.second].push_back(kv.first);
      }

      DLOG(DEBUG) << "Memory planning by scope groups:";
      for (const auto& kv : scope_groups) {
        DLOG(DEBUG) << "  Scope " << kv.first << ": " << kv.second.size() << " buffers";
      }

      for (const auto& scope_kv : scope_groups) {
        PlanMemoryForScope(scope_kv.first, scope_kv.second);
      }
    }

    void LivenessAnalysis() {
      std::unordered_set<const VarNode*> touched;
      for (size_t i = linear_seq_.size(); i != 0; --i) {
        const StmtEntry& s = linear_seq_[i - 1];
        for (const VarNode* buffer : s.touched) {
          if (!touched.count(buffer)) {
            touched.insert(buffer);
            event_map_[s.stmt].kill.push_back(buffer);
          }
        }
      }

      for (size_t i = 0; i < linear_seq_.size(); ++i) {
        const StmtEntry& s = linear_seq_[i];
        for (const VarNode* buffer : s.touched) {
          if (first_use_.count(buffer) && first_use_[buffer] == i) {
            event_map_[s.stmt].gen.push_back(buffer);
          }
        }
      }

      ReorderKillPoints();

      DLOG(DEBUG) << "Liveness Analysis Results:";
      for (const auto& event_pair : event_map_) {
        const EventEntry& entry = event_pair.second;
        if (entry.gen.empty() && entry.kill.empty()) continue;
        
        std::stringstream gen_ss, kill_ss;
        for (const VarNode* var : entry.gen) gen_ss << var->name_hint << " ";
        for (const VarNode* var : entry.kill) kill_ss << var->name_hint << " ";
        
        DLOG(DEBUG) << "  Statement: " << event_pair.first->GetTypeKey();
        if (!entry.gen.empty()) DLOG(DEBUG) << "    GEN: " << gen_ss.str();
        if (!entry.kill.empty()) DLOG(DEBUG) << "    KILL: " << kill_ss.str();
      }
    }

    void ReorderKillPoints() {
      std::vector<StmtEntry> gen_kill_seq;
      for (const auto& stmt_entry : linear_seq_) {
        if (!event_map_[stmt_entry.stmt].gen.empty() ||
            !event_map_[stmt_entry.stmt].kill.empty()) {
          gen_kill_seq.push_back(stmt_entry);
        }
      }

      for (auto& event_pair : event_map_) {
        const Object* stmt = event_pair.first;
        EventEntry& event = event_pair.second;

        if (event.kill.empty()) continue;

        ICHECK(stmt_attrs_.count(stmt));
        int kill_level = stmt_attrs_.at(stmt).level;

        std::unordered_set<const VarNode*> visited_buffers;

        for (auto it = event.kill.begin(); it != event.kill.end();) {
          const VarNode* buffer = *it;
          bool found_gen = false;
          int gen_level = 0;

          for (const auto& gen_pair : event_map_) {
            const auto& gen_event = gen_pair.second;
            if (std::find(gen_event.gen.begin(), gen_event.gen.end(), buffer) != gen_event.gen.end()) {
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

            const Object* last_stmt_at_level = nullptr;
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

    void PlanMemoryForScope(const std::string& scope, const std::vector<const VarNode*>& buffers) {
      DLOG(DEBUG) << "Planning memory for scope: " << scope;

      std::vector<LiveInterval> intervals;
      for (const VarNode* buffer : buffers) {
        int64_t start = -1;
        int64_t end = -1;
          
        for (const auto& event_pair : event_map_) {
          const EventEntry& event = event_pair.second;
          auto it = std::find(event.gen.begin(), event.gen.end(), buffer);
          if (it != event.gen.end()) {
            for (size_t i = 0; i < linear_seq_.size(); ++i) {
              if (linear_seq_[i].stmt == event_pair.first) {
                start = static_cast<int64_t>(i);
                break;
              }
            }
            break;
          }
        }
          
        for (const auto& event_pair : event_map_) {
          const EventEntry& event = event_pair.second;
          auto it = std::find(event.kill.begin(), event.kill.end(), buffer);
          if (it != event.kill.end()) {
            for (size_t i = 0; i < linear_seq_.size(); ++i) {
              if (linear_seq_[i].stmt == event_pair.first) {
                end = static_cast<int64_t>(i);
                break;
              }
            }
            break;
          }
        }
          
        if (start != -1 && end != -1) {
          intervals.emplace_back(buffer, start, end, buffer_sizes_[buffer]);
          DLOG(DEBUG) << "Buffer " << buffer->name_hint << ": [" << start << ", " << end 
                      << "], size=" << buffer_sizes_[buffer];
        }
      }
        
      std::sort(intervals.begin(), intervals.end(), 
                [](const LiveInterval& a, const LiveInterval& b) {
                    return a.start < b.start;
                });
      
      LinearScanAllocator allocator(memory_limits_[scope]);
      auto allocations = allocator.allocate(intervals);
      
      for (const auto& alloc : allocations) {
        address_map_[alloc.buffer] = alloc.offset;
        DLOG(DEBUG) << "Allocated buffer " << alloc.buffer->name_hint 
                    << " at offset " << alloc.offset << " (size=" << alloc.size << ")";
      }
      
      size_t total_used = 0;
      for (const auto& alloc : allocations) {
        total_used = std::max(total_used, alloc.offset + alloc.size);
      }
      
      DLOG(DEBUG) << "Scope " << scope << " memory usage: " << total_used 
                  << "/" << memory_limits_[scope] << " bytes ("
                  << (total_used * 100.0 / memory_limits_[scope]) << "%)";
      if (total_used > memory_limits_[scope]) {
        DLOG(WARNING) << "Memory limit exceeded for scope " << scope 
                      << ": " << total_used << " > " << memory_limits_[scope];
      }
    }

    struct LiveInterval {
      const VarNode* buffer;
      int64_t start;
      int64_t end;
      size_t size;
      
      LiveInterval(const VarNode* buf, int64_t s, int64_t e, size_t sz)
          : buffer(buf), start(s), end(e), size(sz) {}
    };
    
    struct Allocation {
      const VarNode* buffer;
      size_t offset;
      size_t size;
      bool is_reused;
    };
    
    class LinearScanAllocator {
    public:
      LinearScanAllocator(size_t memory_limit)
          : memory_limit_(memory_limit), next_new_offset_(0) {}
      
      std::vector<Allocation> allocate(std::vector<LiveInterval>& intervals) {
        std::vector<Allocation> allocations;
        
        std::sort(intervals.begin(), intervals.end(),
                  [](const LiveInterval& a, const LiveInterval& b) {
                      return a.start < b.start;
                  });
        
        auto end_time_compare = [](const LiveInterval& a, const LiveInterval& b) {
            return a.end > b.end;
        };
        std::priority_queue<LiveInterval, std::vector<LiveInterval>, 
                            decltype(end_time_compare)> active_queue(end_time_compare);
        
        std::vector<Allocation> active_allocations;
        std::vector<std::pair<size_t, size_t>> free_blocks;
        
        for (auto& interval : intervals) {
          DLOG(DEBUG) << "Processing: " << interval.buffer->name_hint
                        << " [" << interval.start << ", " << interval.end 
                        << "] size=" << interval.size;
          
          while (!active_queue.empty() && active_queue.top().end < interval.start) {
            const auto& expired = active_queue.top();
            
            auto it = std::find_if(active_allocations.begin(), active_allocations.end(),
                                  [&](const Allocation& alloc) {
                                      return alloc.buffer == expired.buffer;
                                  });
            if (it != active_allocations.end()) {
              free_blocks.emplace_back(it->offset, it->size);
              active_allocations.erase(it);
              
              DLOG(DEBUG) << "  Released for reuse: " << expired.buffer->name_hint
                            << " at offset " << it->offset;
            }
            
            active_queue.pop();
          }
            
          mergeFreeBlocks(free_blocks);
          
          size_t allocated_offset;
          bool is_reused = false;
          
          size_t new_memory_offset = alignUp(next_new_offset_, 32);
          if (new_memory_offset + interval.size <= memory_limit_) {
            allocated_offset = new_memory_offset;
            next_new_offset_ = new_memory_offset + interval.size;
            DLOG(DEBUG) << "  Allocated NEW memory at offset: " << allocated_offset;
          } else {
            allocated_offset = findReusableBlock(interval.size, free_blocks);
            if (allocated_offset != static_cast<size_t>(-1)) {
                is_reused = true;
                DLOG(DEBUG) << "  REUSED memory at offset: " << allocated_offset;
            } else {
                DLOG(ERROR) << "Memory allocation failed for: " 
                            << interval.buffer->name_hint
                            << " required: " << interval.size
                            << ", new memory available: " << (memory_limit_ - next_new_offset_);
                continue;
            }
          }
            
            Allocation alloc{interval.buffer, allocated_offset, interval.size, is_reused};
            allocations.push_back(alloc);
            active_allocations.push_back(alloc);
            active_queue.push(interval);
            
            if (is_reused) {
                removeFromFreeBlocks(allocated_offset, interval.size, free_blocks);
            }
          }
          
          size_t total_used = next_new_offset_;
          size_t reused_count = std::count_if(allocations.begin(), allocations.end(),
                                              [](const Allocation& a) { return a.is_reused; });
          DLOG(DEBUG) << "Memory usage: " << total_used << "/" << memory_limit_
                    << " bytes (" << (total_used * 100.0 / memory_limit_) << "%)";
          DLOG(DEBUG) << "Reused buffers: " << reused_count << "/" << allocations.size();
          
          return allocations;
      }
        
    private:
      
      void mergeFreeBlocks(std::vector<std::pair<size_t, size_t>>& free_blocks) {
        if (free_blocks.empty()) return;
        
        std::sort(free_blocks.begin(), free_blocks.end());
        
        std::vector<std::pair<size_t, size_t>> merged;
        merged.push_back(free_blocks[0]);
        
        for (size_t i = 1; i < free_blocks.size(); ++i) {
          auto& last = merged.back();
          auto& current = free_blocks[i];
          
          if (last.first + last.second >= current.first) {
            size_t new_end = std::max(last.first + last.second, current.first + current.second);
            last.second = new_end - last.first;
          } else {
            merged.push_back(current);
          }
        }
        
        free_blocks = std::move(merged);
      }
      
      size_t findReusableBlock(size_t required_size, 
                                std::vector<std::pair<size_t, size_t>>& free_blocks) {
          
        std::sort(free_blocks.begin(), free_blocks.end());
          
        for (const auto& block : free_blocks) {
          if (block.second >= required_size) {
            size_t aligned_offset = alignUp(block.first, 32);
            size_t available_after_align = block.second - (aligned_offset - block.first);
            
            if (available_after_align >= required_size) {
              return aligned_offset;
            }
          }
        }
        
        auto& last_block = free_blocks[free_blocks.size() - 1];
        if ((last_block.first + last_block.second) == next_new_offset_) {
          next_new_offset_ = memory_limit_;
          last_block.second = memory_limit_ - last_block.first;
          if (last_block.second >= required_size) {
            size_t aligned_offset = alignUp(last_block.first, 32);
            size_t available_after_align = last_block.second - (aligned_offset - last_block.first);

            if (available_after_align >= required_size) {
              return aligned_offset;
            }
          }
        }
        return -1;
      }

      void removeFromFreeBlocks(size_t offset, size_t size,
                                std::vector<std::pair<size_t, size_t>>& free_blocks) {
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
                free_blocks.emplace_back(allocated_end, block_end - allocated_end);
            }
            
            break;
          } else {
              ++it;
          }
        }
      }
      
      static size_t alignUp(size_t value, size_t alignment) {
          return ((value + alignment - 1) / alignment) * alignment;
      }

      size_t memory_limit_;
      bool verbose_;
      size_t next_new_offset_;
    };

    size_t CalculateBufferSize(const AllocateNode* alloc) {
      size_t size_elements = 1;
      for (const auto& extent : alloc->extents) {
        if (const IntImmNode* int_imm = extent.as<IntImmNode>()) {
          size_elements *= int_imm->value;
        }
      }
      
      size_t size_bytes = size_elements * alloc->dtype.bytes() * alloc->dtype.lanes();
      return AlignUp(size_bytes, 32);
    }

    static size_t AlignUp(size_t value, size_t alignment) {
      return ((value + alignment - 1) / alignment) * alignment;
    }

    void UpdateStmtAttr(const Object* stmt, size_t level) {
      stmt_attrs_[stmt] = StmtAttr{level};
    }

    bool IsNPUSharedMemory(Var buffer_var) {
      std::string scope = GetPtrStorageScope(buffer_var);
      if (memory_limits_.count(scope) > 0) {
        return true;
      }
      return false;
    }

    std::unordered_map<const VarNode*, AllocEntry> alloc_info_;
    std::unordered_map<const VarNode*, int64_t> address_map_;
    std::unordered_map<const VarNode*, std::string> buffer_scopes_;
    std::unordered_map<const VarNode*, size_t> buffer_sizes_;
    std::unordered_map<const VarNode*, size_t> first_use_;
    std::unordered_map<std::string, int> memory_limits_;
    std::unordered_map<const Object*, StmtAttr> stmt_attrs_;
    std::unordered_map<const Object*, EventEntry> event_map_;
    std::vector<StmtEntry> linear_seq_;
    std::vector<StmtEntry> scope_;
    
    std::multimap<uint64_t, StorageEntry*> const_free_map_;
    std::list<StorageEntry*> sym_free_list_;
    std::unordered_map<const VarNode*, StorageEntry*> alloc_map_;
    
    size_t scope_level_{0};
    int max_layer_num_{1};
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

}  // namespace tl
}  // namespace tvm