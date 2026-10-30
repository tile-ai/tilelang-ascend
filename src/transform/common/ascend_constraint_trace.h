// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#pragma once

#include "ascend_constraint.h"
#include "ascend_lifecycle.h"
#include "ascend_resource.h"
#include "ascend_schedule_hints.h"
#include "ascend_sync_a3.h"
#include <array>
#include <limits>
#include <map>
#include <set>
#include <tvm/arith/analyzer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <vector>

namespace tvm {
namespace tl {
namespace constraint_trace {
using namespace tir;
using Record = Map<String, ObjectRef>;
using Key = std::vector<int64_t>;
using LaunchCase = std::map<std::string, std::pair<int64_t, int64_t>>;
constexpr size_t kTaskLimit = 4096;
constexpr size_t kControlLimit = 65536;

inline int PipeIndex(const std::string &pipe) {
  if (pipe == "PIPE_MTE2")
    return 0;
  if (pipe == "PIPE_V")
    return 1;
  if (pipe == "PIPE_MTE3")
    return 2;
  LOG(FATAL) << "Unsupported constraint sync PIPE: " << pipe;
  return -1;
}

struct TraceEvent {
  enum Kind { kTask, kSet, kWait, kBarrier } kind;
  int task{-1}, src{-1}, dst{-1}, slot{-1};
};

// Expand the actual IR, not a reconstructed scheduling formula. Iteration
// marker values travel through the injector's Substitute and loop unrolling.
class TraceBuilder {
public:
  TraceBuilder(PrimFunc function, Array<ObjectRef> units, bool source,
               const LaunchCase &launch_case = {}, bool strict_launch = false,
               Map<Var, PrimExpr> scalar_bindings = {},
               bool checked_arithmetic = false)
      : function_(function), units_(units), source_(source),
        launch_case_(launch_case), strict_launch_(strict_launch),
        bindings_(scalar_bindings), checked_arithmetic_(checked_arithmetic) {
    Walk(function->body);
  }

  Record Analyze() {
    Stmt body = calls.empty()       ? Stmt(Evaluate(0))
                : calls.size() == 1 ? calls[0]
                                    : Stmt(SeqStmt(calls));
    body = AttrStmt(Integer(0), "resource_scope", Integer(1), body);
    for (auto it = allocations_.rbegin(); it != allocations_.rend(); ++it) {
      Allocate allocation = *it;
      allocation.CopyOnWrite()->body = body;
      body = allocation;
    }
    if (!buffers_.empty()) {
      body = BlockRealize(
          {}, Bool(true),
          Block({}, {}, {}, "tilelang_root", body, NullOpt, buffers_));
    }
    for (auto it = launches_.rbegin(); it != launches_.rend(); ++it) {
      AttrStmt launch = *it;
      launch.CopyOnWrite()->body = body;
      body = launch;
    }
    PrimFunc probe = function_;
    probe.CopyOnWrite()->body = body;
    return AnalyzeAscendScheduleEffects(probe, "A3");
  }

  std::vector<Key> keys;
  std::vector<TraceEvent> events;
  Array<Stmt> calls;
  Array<Stmt> anchors;

private:
  PrimExpr Resolve(const PrimExpr &expr) {
    return analyzer_.Simplify(Substitute(expr, bindings_));
  }
  int64_t IntegerValue(const PrimExpr &expr) {
    if (checked_arithmetic_)
      return EvaluateAscendLifetimeInteger(expr, bindings_);
    PrimExpr value = Resolve(expr);
    const auto *integer = value.as<IntImmNode>();
    ICHECK(integer && value.dtype().lanes() == 1 &&
           (value.dtype().is_int() || value.dtype().is_uint()))
        << "Constraint sync verification requires a static integer: " << value;
    ICHECK(!value.dtype().is_uint() || value.dtype().bits() < 64)
        << "Unsupported uint64 constraint sync value";
    return integer->value;
  }
  void Tick() {
    ICHECK_LT(visits_++, kControlLimit)
        << "Constraint sync control-visit limit exceeded (65536)";
  }
  void Walk(const Stmt &statement) {
    Tick();
    if (const auto *seq = statement.as<SeqStmtNode>()) {
      for (const Stmt &child : seq->seq)
        Walk(child);
    } else if (const auto *attr = statement.as<AttrStmtNode>()) {
      if (attr->attr_key == kAscendConstraintTaskId) {
        const auto *id = attr->node.as<IntImmNode>();
        ICHECK(id && id->dtype == DataType::Int(64) && id->value >= 0 &&
               size_t(id->value) < units_.size() &&
               attr->value.dtype() == DataType::Int(32) && is_one(attr->value))
            << "Malformed constraint task marker";
        ICHECK_LT(active_task_, 0) << "Nested constraint task markers";
        active_task_ = id->value;
        size_t before = calls.size();
        Walk(attr->body);
        ICHECK_EQ(calls.size(), before + 1)
            << "Constraint task marker must contain exactly one hardware call";
        active_task_ = -1;
      } else if (attr->attr_key == kAscendConstraintIteration) {
        const auto *id = attr->node.as<IntImmNode>();
        ICHECK(id && id->dtype == DataType::Int(64) && id->value >= 0 &&
               attr->value.dtype().is_int() && attr->value.dtype().bits() <= 64)
            << "Malformed constraint iteration marker";
        ICHECK(!iterations_.count(id->value))
            << "Duplicate constraint iteration marker";
        iterations_[id->value] = IntegerValue(attr->value);
        Walk(attr->body);
        iterations_.erase(id->value);
      } else if (attr->attr_key == "resource_scope") {
        int previous = scope_;
        scope_ = IntegerValue(attr->value);
        ICHECK(scope_ == 0 || scope_ == 1)
            << "Invalid constraint resource scope";
        Walk(attr->body);
        scope_ = previous;
      } else if (attr->attr_key == "thread_extent") {
        const auto *axis = attr->node.as<IterVarNode>();
        ICHECK(axis) << "Malformed constraint launch domain";
        int64_t extent = IntegerValue(attr->value);
        ICHECK_GT(extent, 0) << "Invalid constraint launch extent";
        Map<Var, PrimExpr> saved = bindings_;
        auto coordinate = launch_case_.find(axis->thread_tag);
        ICHECK(!strict_launch_ || coordinate != launch_case_.end())
            << "Unmapped pipeline physical launch domain";
        if (coordinate != launch_case_.end()) {
          ICHECK(coordinate->second.second == extent &&
                 coordinate->second.first >= 0 &&
                 coordinate->second.first < extent)
              << "Pipeline launch domain changed after planning";
          bindings_.Set(axis->var,
                        IntImm(axis->var.dtype(), coordinate->second.first));
        } else if (extent == 1)
          bindings_.Set(axis->var, make_zero(axis->var.dtype()));
        if (launch_vars_.insert(axis->var.get()).second)
          launches_.push_back(GetRef<AttrStmt>(attr));
        Walk(attr->body);
        bindings_ = saved;
      } else {
        LOG(FATAL) << "Unsupported final constraint attribute: "
                   << attr->attr_key;
      }
    } else if (const auto *loop = statement.as<ForNode>()) {
      ICHECK(loop->kind == ForKind::kSerial || loop->kind == ForKind::kUnrolled)
          << "Unsupported constraint sync loop kind";
      int64_t minimum = IntegerValue(loop->min);
      int64_t extent = IntegerValue(loop->extent);
      ICHECK(extent >= 0 && uint64_t(extent) <= kControlLimit)
          << "Constraint sync loop exceeds static control limit";
      __int128 end = __int128(minimum) + extent;
      ICHECK(end <= std::numeric_limits<int64_t>::max())
          << "Constraint sync loop index overflow";
      DataType dtype = loop->loop_var.dtype();
      ICHECK(dtype.is_int() && dtype.bits() <= 64)
          << "Unsupported constraint loop index dtype";
      if (dtype.bits() < 64) {
        int64_t limit = int64_t(1) << (dtype.bits() - 1);
        ICHECK(minimum >= -limit && minimum < limit && end >= -limit &&
               end < limit)
            << "Constraint loop endpoint/update cannot be represented";
      }
      Map<Var, PrimExpr> saved = bindings_;
      for (int64_t index = 0; index < extent; ++index) {
        int64_t value = minimum + index;
        if (dtype.bits() < 64) {
          int64_t limit = int64_t(1) << (dtype.bits() - 1);
          ICHECK(value >= -limit && value < limit)
              << "Constraint loop index cannot be represented";
        }
        bindings_.Set(loop->loop_var, IntImm(dtype, value));
        Walk(loop->body);
      }
      bindings_ = saved;
    } else if (const auto *let = statement.as<LetStmtNode>()) {
      PrimExpr value;
      if (checked_arithmetic_ &&
          (let->var.dtype().is_int() || let->var.dtype().is_uint())) {
        int64_t integer = EvaluateAscendLifetimeInteger(
            Cast(let->var.dtype(), let->value), bindings_);
        value = IntImm(let->var.dtype(), integer);
      } else {
        value = Resolve(let->value);
      }
      PostOrderVisit(value, [](const ObjectRef &node) {
        ICHECK(!node.as<BufferLoadNode>() && !node.as<CallNode>())
            << "Unsupported memory-dependent constraint scalar binding";
      });
      Map<Var, PrimExpr> saved = bindings_;
      bindings_.Set(let->var, value);
      Walk(let->body);
      bindings_ = saved;
    } else if (const auto *branch = statement.as<IfThenElseNode>()) {
      if (IntegerValue(branch->condition))
        Walk(branch->then_case);
      else if (branch->else_case.defined())
        Walk(branch->else_case.value());
    } else if (const auto *realize = statement.as<BlockRealizeNode>()) {
      ICHECK(realize->block->iter_vars.empty() &&
             realize->block->match_buffers.empty() &&
             !realize->block->init.defined())
          << "Unsupported constraint block";
      if (!IntegerValue(realize->predicate))
        return;
      for (const Buffer &buffer : realize->block->alloc_buffers) {
        if (checked_arithmetic_) {
          int64_t bytes = buffer->dtype.bytes();
          for (const PrimExpr &extent : buffer->shape) {
            int64_t count = IntegerValue(extent);
            ICHECK(count > 0 &&
                   bytes <= std::numeric_limits<int64_t>::max() / count)
                << "Lifetime final allocation extent overflow";
            bytes *= count;
          }
        }
        if (allocation_vars_.insert(buffer->data.get()).second)
          buffers_.push_back(buffer);
      }
      Walk(realize->block->body);
    } else if (const auto *allocation = statement.as<AllocateNode>()) {
      ICHECK_EQ(IntegerValue(allocation->condition), 1)
          << "Unsupported conditional constraint allocation";
      if (allocation_vars_.insert(allocation->buffer_var.get()).second) {
        Allocate copy = GetRef<Allocate>(allocation);
        Array<PrimExpr> extents;
        int64_t bytes = allocation->dtype.bytes();
        for (const PrimExpr &extent : allocation->extents) {
          int64_t count = IntegerValue(extent);
          ICHECK_GT(count, 0) << "Invalid constraint allocation";
          if (checked_arithmetic_) {
            ICHECK(bytes <= std::numeric_limits<int64_t>::max() / count)
                << "Lifetime final allocation extent overflow";
            bytes *= count;
          }
          extents.push_back(Resolve(extent));
        }
        copy.CopyOnWrite()->extents = extents;
        allocations_.push_back(copy);
      }
      Walk(allocation->body);
    } else if (const auto *evaluate = statement.as<EvaluateNode>()) {
      if (checked_arithmetic_ && evaluate->value.as<CallNode>())
        ValidateAscendLifetimeCallIndices(Downcast<Call>(evaluate->value),
                                          bindings_);
      PrimExpr value = Resolve(evaluate->value);
      if (is_zero(value))
        return;
      const auto *call = value.as<CallNode>();
      ICHECK(call) << "Unsupported final constraint scalar statement";
      ICHECK_EQ(scope_, 1)
          << "Constraint sync requires explicit Vector ownership";
      if (Sync(call))
        return;
      ICHECK_GE(active_task_, 0)
          << "Unmarked hardware call in final constraint IR";
      ICHECK_LT(keys.size(), kTaskLimit)
          << "Constraint sync hardware-task instance limit exceeded (4096)";
      Record unit = Downcast<Record>(units_[active_task_]);
      auto loops = Downcast<Array<Integer>>(unit["loop_ids"]);
      ICHECK_EQ(loops.size(), iterations_.size())
          << "Constraint task iteration identity mismatch";
      Key key{active_task_};
      for (const Integer &id : loops) {
        ICHECK(iterations_.count(id->value))
            << "Missing constraint iteration identity";
        key.push_back(id->value);
        key.push_back(iterations_.at(id->value));
      }
      keys.push_back(key);
      events.push_back({TraceEvent::kTask, int(keys.size() - 1)});
      anchors.push_back(GetRef<Stmt>(evaluate));
      calls.push_back(Evaluate(value));
    } else {
      LOG(FATAL) << "Unsupported final constraint statement: "
                 << statement->GetTypeKey();
    }
  }

  bool Sync(const CallNode *call) {
    bool set = call->op.same_as(ascend_auto_set_flag());
    bool wait = call->op.same_as(ascend_auto_wait_flag());
    bool barrier = call->op.same_as(ascend_auto_barrier());
    if (!set && !wait && !barrier)
      return false;
    ICHECK(!source_) << "Manual synchronization in constraint source manifest";
    std::string operation;
    AscendResource resource = ResourceForCall(call, &operation);
    ICHECK(resource != AscendResource::kCube)
        << "Cube synchronization in Vector constraint kernel";
    ICHECK_EQ(call->args.size(), barrier ? 1U : 2U)
        << "Malformed automatic constraint sync ABI";
    const auto *name = call->args[0].as<StringImmNode>();
    ICHECK(name) << "Dynamic automatic constraint sync domain";
    if (barrier) {
      int pipe = name->value == "PIPE_ALL" ? -1 : PipeIndex(name->value);
      events.push_back({TraceEvent::kBarrier, -1, pipe});
    } else {
      std::string event = name->value;
      size_t separator = event.find('_');
      ICHECK(separator != std::string::npos)
          << "Malformed automatic event domain";
      std::string src = "PIPE_" + event.substr(0, separator);
      std::string dst = "PIPE_" + event.substr(separator + 1);
      auto mapping = GetEventMapping().find(src + "_" + dst);
      ICHECK(mapping != GetEventMapping().end() && mapping->second == event)
          << "Unsupported automatic event domain";
      int64_t slot = IntegerValue(call->args[1]);
      ICHECK(call->args[1].dtype() == DataType::Int(32) && slot >= 0 &&
             slot < kAscendA3LocalEventCount)
          << "Invalid compiler event slot; expected int32 ID 0..7";
      events.push_back({set ? TraceEvent::kSet : TraceEvent::kWait, -1,
                        PipeIndex(src), PipeIndex(dst), int(slot)});
    }
    return true;
  }

  PrimFunc function_;
  Array<ObjectRef> units_;
  bool source_;
  LaunchCase launch_case_;
  const bool strict_launch_;
  int active_task_{-1}, scope_{-1};
  size_t visits_{0};
  arith::Analyzer analyzer_;
  Map<Var, PrimExpr> bindings_;
  std::map<int64_t, int64_t> iterations_;
  std::set<const VarNode *> allocation_vars_, launch_vars_;
  Array<Buffer> buffers_;
  std::vector<Allocate> allocations_;
  std::vector<AttrStmt> launches_;
  const bool checked_arithmetic_;
};

struct Effect {
  Var data;
  std::string scope;
  PrimExpr begin, end;
  Optional<PrimExpr> physical_begin, physical_end;
};
struct Instance {
  std::string pipe, operation;
  Call call;
  std::vector<Effect> reads, writes;
};

inline std::pair<PrimExpr, PrimExpr> PhysicalInterval(PrimFunc function,
                                                      const Effect &effect) {
  arith::Analyzer analyzer;
  ICHECK_EQ(effect.scope, "shared.ub")
      << "Constraint sync supports only physical UB storage";
  auto addresses = function->GetAttr<Map<Var, PrimExpr>>("address_map");
  auto sizes = function->GetAttr<Map<Var, PrimExpr>>("size_map");
  ICHECK(addresses.defined() && sizes.defined() &&
         addresses.value().count(effect.data) &&
         sizes.value().count(effect.data))
      << "Missing final constraint physical allocation metadata";
  const auto *address = addresses.value()[effect.data].as<IntImmNode>();
  const auto *size = sizes.value()[effect.data].as<IntImmNode>();
  ICHECK(address && size && address->value >= 0 && size->value > 0 &&
         address->value % 32 == 0 && size->value % 32 == 0 &&
         __int128(address->value) + size->value <= kAscendA2A3UBBudget)
      << "Invalid final constraint physical allocation interval";
  ICHECK(
      analyzer.CanProve(effect.begin >= 0 && effect.end >= effect.begin &&
                        effect.end <= IntImm(DataType::Int(64), size->value)))
      << "Unproven final constraint physical access bounds";
  PrimExpr base = IntImm(DataType::Int(64), address->value);
  return {analyzer.Simplify(base + effect.begin),
          analyzer.Simplify(base + effect.end)};
}

inline std::vector<Instance>
BuildInstances(PrimFunc function, const Record &graph, bool physical) {
  auto storages = Downcast<Array<ObjectRef>>(graph["storages"]);
  std::vector<Instance> result;
  for (const ObjectRef &item : Downcast<Array<ObjectRef>>(graph["tasks"])) {
    Record task = Downcast<Record>(item);
    Instance instance;
    instance.pipe = Downcast<String>(task["pipe"]);
    instance.operation = Downcast<String>(task["operation"]);
    instance.call =
        Downcast<Call>(Downcast<Evaluate>(task["statement"])->value);
    PipeIndex(instance.pipe);
    for (bool writes : {false, true}) {
      for (const ObjectRef &entry :
           Downcast<Array<ObjectRef>>(task[writes ? "writes" : "reads"])) {
        Record access = Downcast<Record>(entry);
        int storage = Downcast<Integer>(access["storage_id"])->value;
        ICHECK(storage >= 0 && size_t(storage) < storages.size())
            << "Invalid constraint storage ID";
        Record allocation = Downcast<Record>(storages[storage]);
        Effect effect{Downcast<Var>(allocation["data"]),
                      Downcast<String>(allocation["scope"]),
                      Downcast<PrimExpr>(access["start"]),
                      Downcast<PrimExpr>(access["end"])};
        if (physical && effect.scope != "global") {
          auto interval = PhysicalInterval(function, effect);
          effect.physical_begin = interval.first;
          effect.physical_end = interval.second;
        }
        (writes ? instance.writes : instance.reads).push_back(effect);
      }
    }
    result.push_back(instance);
  }
  return result;
}

} // namespace constraint_trace
} // namespace tl
} // namespace tvm
