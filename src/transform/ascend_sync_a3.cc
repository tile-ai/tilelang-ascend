// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include <tvm/runtime/registry.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <tuple>
#include <vector>

#include "common/ascend_constraint_trace.h"
#include "common/ascend_sync_a3.h"

namespace tvm {
namespace tl {
TVM_REGISTER_PASS_CONFIG_OPTION(kAscendSyncStrategy, String);

namespace {
using namespace tir;
using Record = Map<String, ObjectRef>;
using namespace constraint_trace;

int64_t IntegerValue(const ObjectRef &object, const char *field,
                     bool int32 = false) {
  const auto *value = object.as<IntImmNode>();
  ICHECK(value && value->dtype.is_int() && value->dtype.lanes() == 1 &&
         value->dtype.bits() <= 64 && (!int32 || value->dtype.bits() == 32))
      << "A3 event allocation requires a signed integer " << field;
  return value->value;
}
String StringValue(const ObjectRef &object, const char *field) {
  ICHECK(object.as<runtime::StringObj>()) << "Malformed A3 event " << field;
  return Downcast<String>(object);
}
int EventPipe(const String &pipe) {
  if (pipe == "MTE2")
    return 0;
  if (pipe == "V")
    return 1;
  if (pipe == "MTE3")
    return 2;
  LOG(FATAL) << "Unsupported A3 Vector HardEvent PIPE: " << pipe;
  return -1;
}
String PipeName(int pipe) {
  return pipe == 0 ? String("MTE2") : pipe == 1 ? String("V") : String("MTE3");
}
struct EventDomain {
  int64_t core, instance;
  int src, dst;
  bool operator<(const EventDomain &other) const {
    return std::tie(core, instance, src, dst) <
           std::tie(other.core, other.instance, other.src, other.dst);
  }
};
struct EventInterval {
  size_t request;
  int64_t begin, end;
  int fixed{-1};
};
bool Intersects(const EventInterval &a, const EventInterval &b) {
  return a.begin < b.end && b.begin < a.end;
}

Record AllocateEvents(const Array<Record> &requests, const String &platform) {
  ICHECK_EQ(platform, "A3") << "Local event allocation supports A3 only";
  ICHECK_LE(requests.size(), 65536U) << "A3 event interval limit exceeded";
  std::map<EventDomain, std::vector<EventInterval>> groups;
  std::map<EventDomain, std::set<int>> reservations;
  for (size_t i = 0; i < requests.size(); ++i) {
    const Record &request = requests[i];
    int64_t core = IntegerValue(request.at("core"), "core"), instance = 0;
    ICHECK_EQ(core, 1) << "A3 local allocator supports Vector core kind 1 only";
    if (request.count("instance"))
      instance = IntegerValue(request.at("instance"), "instance");
    ICHECK_GE(instance, 0) << "Invalid A3 core instance";
    int src = EventPipe(StringValue(request.at("src"), "source PIPE")),
        dst = EventPipe(StringValue(request.at("dst"), "destination PIPE"));
    ICHECK_NE(src, dst)
        << "Same-PIPE dependencies require a barrier, not HardEvent";
    int64_t begin = IntegerValue(request.at("begin"), "begin"),
            end = IntegerValue(request.at("end"), "end");
    ICHECK(begin >= 0 && begin < end) << "Invalid A3 half-open event lifetime";
    EventDomain domain{core, instance, src, dst};
    EventInterval interval{i, begin, end};
    auto slot = [&](const ObjectRef &object) {
      int64_t id = IntegerValue(object, "event ID", true);
      ICHECK(id >= 0 && id < kAscendA3LocalEventCount)
          << "A3 local event ID must be int32 0..7";
      return int(id);
    };
    if (request.count("fixed_id"))
      interval.fixed = slot(request.at("fixed_id"));
    if (request.count("reserved_ids")) {
      auto ids = Downcast<Array<ObjectRef>>(request.at("reserved_ids"));
      for (const ObjectRef &id : ids)
        reservations[domain].insert(slot(id));
    }
    groups[domain].push_back(interval);
  }
  std::vector<int> assignments(requests.size(), -1);
  Array<ObjectRef> domains;
  for (auto &[domain, intervals] : groups) {
    const auto &reserved = reservations[domain];
    std::array<std::vector<EventInterval>, kAscendA3LocalEventCount> fixed;
    std::vector<std::pair<int64_t, int>> endpoints;
    for (const EventInterval &interval : intervals) {
      endpoints.emplace_back(interval.begin, 1);
      endpoints.emplace_back(interval.end, -1);
      if (interval.fixed >= 0) {
        ICHECK(!reserved.count(interval.fixed))
            << "Explicit A3 event ID conflicts with reserved ID "
            << interval.fixed;
        fixed[interval.fixed].push_back(interval);
      }
    }
    std::sort(endpoints.begin(),
              endpoints.end()); // End is released before equal begin.
    int active = 0, peak = 0;
    for (auto [time, change] : endpoints) {
      active += change;
      peak = std::max(peak, active);
    }
    ICHECK(peak <= kAscendA3LocalEventCount - int(reserved.size()))
        << "A3 local event budget exhausted for core=" << domain.core
        << " instance=" << domain.instance << " " << PipeName(domain.src) << "_"
        << PipeName(domain.dst) << ": required=" << peak
        << " available=" << kAscendA3LocalEventCount - reserved.size();
    for (auto &lane : fixed) {
      std::sort(lane.begin(), lane.end(), [](const auto &a, const auto &b) {
        return std::tie(a.begin, a.end, a.request) <
               std::tie(b.begin, b.end, b.request);
      });
      for (size_t i = 1; i < lane.size(); ++i)
        ICHECK(!Intersects(lane[i - 1], lane[i]))
            << "Explicit A3 event ID has overlapping active lifetimes";
    }
    std::sort(intervals.begin(), intervals.end(),
              [](const auto &a, const auto &b) {
                return std::make_tuple(a.begin, a.fixed < 0, a.end, a.request) <
                       std::make_tuple(b.begin, b.fixed < 0, b.end, b.request);
              });
    std::array<int64_t, kAscendA3LocalEventCount> live_until{};
    for (const EventInterval &interval : intervals) {
      int chosen = interval.fixed;
      if (chosen < 0) {
        for (int id = 0; id < kAscendA3LocalEventCount; ++id) {
          if (reserved.count(id) || live_until[id] > interval.begin)
            continue;
          const auto &lane = fixed[id];
          auto upcoming = std::lower_bound(
              lane.begin(), lane.end(), interval.begin,
              [](const EventInterval &fixed_interval, int64_t begin) {
                return fixed_interval.end <= begin;
              });
          if (upcoming != lane.end() && Intersects(*upcoming, interval))
            continue;
          chosen = id;
          break;
        }
      }
      ICHECK(chosen >= 0 && !reserved.count(chosen) &&
             live_until[chosen] <= interval.begin)
          << "No conservative A3 event coloring fits the fixed IDs and live "
             "intervals";
      live_until[chosen] = interval.end;
      assignments[interval.request] = chosen;
    }
    Array<Integer> reserved_ids, available_ids;
    for (int id = 0; id < kAscendA3LocalEventCount; ++id)
      (reserved.count(id) ? reserved_ids : available_ids)
          .push_back(Integer(id));
    domains.push_back(
        Record{{"core", Integer(domain.core)},
               {"instance", IntImm(DataType::Int(64), domain.instance)},
               {"src", PipeName(domain.src)},
               {"dst", PipeName(domain.dst)},
               {"capacity", Integer(kAscendA3LocalEventCount)},
               {"reserved_ids", reserved_ids},
               {"available_ids", available_ids},
               {"peak_required", Integer(peak)}});
  }
  Array<ObjectRef> allocated;
  for (size_t i = 0; i < assignments.size(); ++i)
    allocated.push_back(
        Record{{"request", Integer(i)},
               {"id", IntImm(DataType::Int(32), assignments[i])}});
  return {{"schema_version", Integer(1)},
          {"allocations", allocated},
          {"domains", domains}};
}
} // namespace

Map<String, ObjectRef>
AllocateAscendSyncEvents(Array<Map<String, ObjectRef>> intervals,
                         String platform) {
  return AllocateEvents(intervals, platform);
}
TVM_REGISTER_GLOBAL("tl.transform.AllocateAscendSyncEvents")
    .set_body_typed(AllocateAscendSyncEvents);

namespace {
// Reject every unowned effect before replacing the old local and VS owners.
class LocalScopeCheck final : public StmtExprVisitor {
private:
  void VisitStmt_(const AttrStmtNode *node) final {
    if (node->attr_key == "resource_scope") {
      int saved = scope_;
      const auto *value = node->value.as<IntImmNode>();
      ICHECK(value && (value->value == 0 || value->value == 1))
          << "A3 local sync requires explicit resource ownership";
      scope_ = value->value;
      StmtExprVisitor::VisitStmt_(node);
      scope_ = saved;
      return;
    }
    ICHECK(node->attr_key == "thread_extent" ||
           node->attr_key == kAscendConstraintTaskId ||
           node->attr_key == kAscendConstraintIteration)
        << "Unsupported attribute in A3 local synchronization: "
        << node->attr_key;
    StmtExprVisitor::VisitStmt_(node);
  }
  void VisitExpr_(const CallNode *node) final {
    if (node != hardware_)
      ICHECK(node->op.same_as(builtin::tvm_access_ptr()) ||
             node->op.same_as(Op::Get("tir.type_annotation")) ||
             node->op.same_as(builtin::if_then_else()))
          << "Unknown nested/opaque effect in A3 local synchronization";
    StmtExprVisitor::VisitExpr_(node);
  }
  void VisitStmt_(const EvaluateNode *node) final {
    if (is_zero(node->value))
      return;
    const auto *call = node->value.as<CallNode>();
    ICHECK(call && scope_ == 1)
        << "A3 local sync owns a wholly Vector kernel only";
    std::string operation;
    ICHECK(ResourceForCall(call, &operation) == AscendResource::kVector)
        << "Unsupported scalar/opaque/C effect in A3 local sync: " << operation;
    if (call->op.same_as(builtin::call_extern())) {
      ICHECK(!call->args.empty() && call->args[0].as<StringImmNode>())
          << "Malformed A3 local DMA";
      std::string name =
          NormalizeFunctionName(Downcast<StringImm>(call->args[0])->value);
      ICHECK(name == "copy_gm_to_ub" || name == "copy_ub_to_gm")
          << "Unsupported compound helper in A3 local sync: " << name;
    } else {
      const auto *op = call->op.as<OpNode>();
      ICHECK(op &&
             (op->name == "tl.ascend_fill" || op->name == "tl.ascend_add"))
          << "Manual/already generated synchronization or unsupported A3 local "
             "operation";
    }
    const CallNode *saved = hardware_;
    hardware_ = call;
    StmtExprVisitor::VisitStmt_(node);
    hardware_ = saved;
  }
  void VisitExpr_(const BufferLoadNode *) final {
    LOG(FATAL) << "Unmodeled scalar/VS memory effect in A3 local sync";
  }
  void VisitStmt_(const BufferStoreNode *) final {
    LOG(FATAL) << "Unmodeled scalar/VS store in A3 local sync";
  }
  int scope_{-1};
  const CallNode *hardware_{nullptr};
};

int64_t Constant(const PrimExpr &value) {
  const auto *integer = value.as<IntImmNode>();
  ICHECK(integer && integer->dtype.is_int())
      << "A3 local sync requires concrete final byte intervals";
  return integer->value;
}
bool Overlapping(const Effect &a, const Effect &b, bool noalias) {
  if (a.scope != b.scope)
    return false;
  if (a.scope == "global" && !a.data.same_as(b.data))
    return !noalias;
  const PrimExpr &a_begin =
      a.scope == "global" ? a.begin : a.physical_begin.value();
  const PrimExpr &a_end = a.scope == "global" ? a.end : a.physical_end.value();
  const PrimExpr &b_begin =
      b.scope == "global" ? b.begin : b.physical_begin.value();
  const PrimExpr &b_end = b.scope == "global" ? b.end : b.physical_end.value();
  return Constant(a_begin) < Constant(b_end) &&
         Constant(b_begin) < Constant(a_end);
}
bool PhysicalHazard(const Instance &a, const Instance &b, bool noalias) {
  auto intersect = [&](const std::vector<Effect> &left,
                       const std::vector<Effect> &right) {
    for (const Effect &x : left)
      for (const Effect &y : right)
        if (Overlapping(x, y, noalias))
          return true;
    return false;
  };
  return intersect(a.writes, b.reads) || intersect(a.reads, b.writes) ||
         intersect(a.writes, b.writes);
}
using ByteIntervals = std::vector<std::pair<int64_t, int64_t>>;
struct GlobalCase {
  LaunchCase launch;
  std::vector<int64_t> booleans;
  std::map<const VarNode *, std::array<ByteIntervals, 2>> effects;
};
void MergeIntervals(ByteIntervals *intervals) {
  std::sort(intervals->begin(), intervals->end());
  ByteIntervals merged;
  for (auto interval : *intervals) {
    if (!merged.empty() && interval.first <= merged.back().second)
      merged.back().second = std::max(merged.back().second, interval.second);
    else
      merged.push_back(interval);
  }
  *intervals = std::move(merged);
}
bool SpanOverlap(const ByteIntervals &a, const ByteIntervals &b) {
  size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i].first < b[j].second && b[j].first < a[i].second)
      return true;
    if (a[i].second <= b[j].second)
      ++i;
    else
      ++j;
  }
  return false;
}
void CheckIndependentCores(const std::vector<GlobalCase> &cases, bool noalias) {
  for (size_t i = 0; i < cases.size(); ++i)
    for (size_t j = i + 1; j < cases.size(); ++j) {
      const auto &a = cases[i], &b = cases[j];
      if (a.launch == b.launch || a.booleans != b.booleans)
        continue;
      for (const auto &left : a.effects)
        for (const auto &right : b.effects) {
          if (left.first != right.first && noalias)
            continue;
          auto overlap = [&](int x, int y) {
            if (left.second[x].empty() || right.second[y].empty())
              return false;
            return left.first != right.first ||
                   SpanOverlap(left.second[x], right.second[y]);
          };
          ICHECK(!overlap(1, 0) && !overlap(0, 1) && !overlap(1, 1))
              << "A3 local sync cannot own cross-core GM RAW/WAR/WAW; "
                 "use the audited fixed C/V handoff path";
        }
    }
}
struct LocalSite {
  Stmt statement;
  int pipe{-1};
  std::set<int> sources;
  std::set<int64_t> task_ids;
  std::map<int, size_t> requests;
};
class LocalEmitter final : public StmtMutator {
public:
  LocalEmitter(const std::vector<LocalSite> &sites, const Record &budget) {
    auto allocations = Downcast<Array<Record>>(budget.at("allocations"));
    for (const LocalSite &site : sites) {
      Array<Stmt> prefix;
      for (int source : site.sources) {
        if (source == site.pipe) {
          prefix.push_back(Evaluate(
              Call(DataType::Handle(), ascend_auto_barrier(),
                   {StringImm("PIPE_" + std::string(PipeName(site.pipe)))})));
        } else {
          size_t request = site.requests.at(source);
          int slot =
              IntegerValue(allocations[request].at("id"), "allocated ID", true);
          std::string event = std::string(PipeName(source)) + "_" +
                              std::string(PipeName(site.pipe));
          Array<PrimExpr> args{StringImm(event),
                               IntImm(DataType::Int(32), slot)};
          prefix.push_back(
              Evaluate(Call(DataType::Handle(), ascend_auto_set_flag(), args)));
          prefix.push_back(Evaluate(
              Call(DataType::Handle(), ascend_auto_wait_flag(), args)));
          // WaitFlag is queued on the destination PIPE.  The ALL fence makes
          // its consumption complete before any later reuse of this binary ID.
          prefix.push_back(
              Evaluate(Call(DataType::Handle(), ascend_auto_barrier(),
                            {StringImm("PIPE_ALL")})));
        }
      }
      if (!prefix.empty())
        prefixes_.emplace(site.statement.get(), prefix);
    }
  }

private:
  Stmt VisitStmt_(const EvaluateNode *node) final {
    auto prefix = prefixes_.find(node);
    if (prefix == prefixes_.end())
      return GetRef<Stmt>(node);
    Array<Stmt> body = prefix->second;
    body.push_back(GetRef<Stmt>(node));
    return SeqStmt(body);
  }
  std::map<const Object *, Array<Stmt>> prefixes_;
};

PrimFunc GenerateLocalSync(PrimFunc function, Target target, String platform,
                           tvm::transform::PassContext context) {
  String strategy =
      context->GetConfig<String>(kAscendSyncStrategy, String("legacy")).value();
  ICHECK(strategy == "legacy" || strategy == "a3")
      << "Unknown Ascend sync strategy: " << strategy;
  if (strategy == "legacy")
    return function;
  ICHECK_EQ(platform, "A3") << "New local synchronization supports A3 only";
  String model = target->GetAttr<String>("model", String("")).value();
  ICHECK(model == "ascendc" || model == "auto")
      << "New local sync requires AscendC/auto";
  if (auto declared = function->GetAttr<String>("npu_platform"))
    ICHECK_EQ(declared.value(), platform)
        << "A3 local sync platform metadata mismatch";
  ICHECK(context->GetConfig<String>(kAscendPipelinePlanner, String("legacy"))
             .value() == "constraint")
      << "A3 local sync requires the constraint planner; fixed C/V uses the "
         "handoff owner";
  ICHECK(
      context->GetConfig<Bool>("tl.ascend_auto_sync", Bool(false))
          .value()
          ->value &&
      context->GetConfig<Bool>("tl.ascend_memory_planning", Bool(false))
          .value()
          ->value &&
      !context->GetConfig<Bool>("tl.ascend_auto_sync_vs", Bool(false))
           .value()
           ->value &&
      !context->GetConfig<Bool>("tl.ascend_auto_cross_core_sync", Bool(false))
           .value()
           ->value)
      << "A3 local sync requires auto_sync/memory planning and a proven "
         "VS-free, non-cross-core owner";
  ICHECK(!function->GetAttr<ObjectRef>(kAscendLocalSyncPlan).defined())
      << "A3 local synchronization is already generated";
  LocalScopeCheck check;
  check(function->body);
  auto stored = function->GetAttr<Record>(kAscendConstraintPlan);
  ICHECK(stored.defined() && IntegerValue(stored.value().at("schema_version"),
                                          "manifest schema") == 3)
      << "A3 local synchronization requires the lifecycle schema 3 source "
         "contract";
  PrimFunc original = Downcast<PrimFunc>(stored.value().at("source_function"));
  Record lifetime = RebuildAscendLifetimeContract(original);
  Record source_graph = Downcast<Record>(lifetime.at("source_graph"));
  Array<ObjectRef> units = Downcast<Array<ObjectRef>>(source_graph.at("tasks"));
  bool noalias = original->HasNonzeroAttr("tir.noalias");
  std::vector<LocalSite> sites;
  std::map<const Object *, size_t> site_ids;
  PostOrderVisit(function->body, [&](const ObjectRef &object) {
    const auto *evaluate = object.as<EvaluateNode>();
    if (!evaluate || is_zero(evaluate->value))
      return;
    ICHECK_LT(sites.size(), kControlLimit)
        << "A3 local static site limit exceeded";
    if (site_ids.count(evaluate))
      return;
    site_ids.emplace(evaluate, sites.size());
    sites.push_back({GetRef<Stmt>(evaluate)});
  });
  size_t cases = 0;
  std::vector<GlobalCase> global_cases;
  for (const ObjectRef &item :
       Downcast<Array<ObjectRef>>(lifetime.at("cases"))) {
    Record selection = Downcast<Record>(item);
    LaunchCase launch;
    for (const auto &entry : Downcast<Record>(selection.at("launches"))) {
      Record domain = Downcast<Record>(entry.second);
      launch[entry.first] = {
          IntegerValue(domain.at("coordinate"), "launch coordinate"),
          IntegerValue(domain.at("extent"), "launch extent")};
    }
    TraceBuilder trace(
        function, units, false, launch, true,
        Downcast<Map<Var, PrimExpr>>(selection.at("bool_bindings")), true);
    ICHECK_EQ(trace.events.size(), trace.calls.size())
        << "A3 local input already contains generated events";
    for (const TraceEvent &event : trace.events)
      ICHECK(event.kind == TraceEvent::kTask)
          << "A3 local input contains manual/generated events";
    std::vector<Instance> actual =
        BuildInstances(function, trace.Analyze(), true);
    ICHECK_EQ(actual.size(), trace.anchors.size());
    GlobalCase global;
    global.launch = launch;
    auto booleans = Downcast<Map<Var, PrimExpr>>(selection.at("bool_bindings"));
    for (const Var &parameter : original->params)
      if (parameter.dtype().is_bool())
        global.booleans.push_back(
            EvaluateAscendLifetimeInteger(booleans.at(parameter), {}));
    for (const Instance &task : actual)
      for (int writing = 0; writing < 2; ++writing)
        for (const Effect &effect : writing ? task.writes : task.reads) {
          if (effect.scope != "global")
            continue;
          int64_t begin = Constant(effect.begin), end = Constant(effect.end);
          if (begin < end)
            global.effects[effect.data.get()][writing].push_back({begin, end});
        }
    for (auto &owner : global.effects)
      for (auto &intervals : owner.second)
        MergeIntervals(&intervals);
    global_cases.push_back(std::move(global));
    for (size_t j = 0; j < actual.size(); ++j) {
      LocalSite &site = sites.at(site_ids.at(trace.anchors[j].get()));
      int pipe = PipeIndex(actual[j].pipe);
      ICHECK(site.pipe < 0 || site.pipe == pipe)
          << "A3 local site changed its PIPE";
      site.pipe = pipe;
      site.task_ids.insert(trace.keys[j][0]);
      for (size_t i = 0; i < j; ++i)
        if (PhysicalHazard(actual[i], actual[j], noalias))
          site.sources.insert(PipeIndex(actual[i].pipe));
    }
    ++cases;
  }
  CheckIndependentCores(global_cases, noalias);
  Array<Record> intervals;
  Array<ObjectRef> physical_plan;
  for (size_t site_id = 0; site_id < sites.size(); ++site_id) {
    LocalSite &site = sites[site_id];
    if (site.pipe < 0)
      continue;
    Array<String> sources;
    Array<Integer> task_ids;
    for (int64_t id : site.task_ids)
      task_ids.push_back(Integer(id));
    for (int source : site.sources) {
      sources.push_back("PIPE_" + std::string(PipeName(source)));
      if (source == site.pipe)
        continue;
      size_t index = intervals.size();
      site.requests[source] = index;
      // Abstract issue positions cover Set, queued Wait and the ALL fence that
      // retires the lease. They are not hardware time or timing-model ticks.
      // Union across iterations may synchronize an otherwise empty pipe.
      intervals.push_back(
          Record{{"core", Integer(1)},
                 {"src", PipeName(source)},
                 {"dst", PipeName(site.pipe)},
                 {"begin", IntImm(DataType::Int(64), int64_t(index) * 3)},
                 {"end", IntImm(DataType::Int(64), int64_t(index) * 3 + 3)}});
    }
    physical_plan.push_back(
        Record{{"site", Integer(site_id)},
               {"task_ids", task_ids},
               {"pipe", String("PIPE_" + std::string(PipeName(site.pipe)))},
               {"required_sources", sources}});
  }
  Record budget = AllocateEvents(intervals, platform);
  Record plan{{"schema_version", Integer(1)},
              {"generator", String("a3")},
              {"source_phase", String("post_memory_planning")},
              {"source_function", function},
              {"physical_plan", physical_plan},
              {"intervals", intervals},
              {"event_budget", budget},
              {"case_count", Integer(cases)},
              {"vs_effects", String("absent")},
              {"lifetime_model", String("wait_all_fence_before_reuse")}};
  PrimFunc result = function;
  result.CopyOnWrite()->body = LocalEmitter(sites, budget)(function->body);
  return WithAttr(result, kAscendLocalSyncPlan, plan);
}
} // namespace

tir::transform::Pass AscendInsertSyncA3(Target target, String platform) {
  auto pass = [=](PrimFunc function, IRModule,
                  tvm::transform::PassContext context) {
    return GenerateLocalSync(function, target, platform, context);
  };
  return tir::transform::CreatePrimFuncPass(pass, 0, "tl.AscendInsertSyncA3",
                                            {});
}
TVM_REGISTER_GLOBAL("tl.transform.AscendInsertSyncA3")
    .set_body_typed(AscendInsertSyncA3);

} // namespace tl
} // namespace tvm
