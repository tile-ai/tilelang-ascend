// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/node/structural_equal.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "../op/ascend.h"
#include "common/ascend_lifecycle.h"
#include "common/ascend_resource.h"
#include "common/ascend_scheduled_sync.h"

namespace tvm {
namespace tl {
namespace {
using namespace tir;
using Record = AutoRecord;
using Key = std::vector<int64_t>;
int64_t Number(const ObjectRef &object) {
  const auto *value = object.as<IntImmNode>();
  ICHECK(value && value->dtype.is_int())
      << "Auto proof requires a signed integer";
  return value->value;
}
int64_t Field(const Record &record, const char *key) {
  ICHECK(record.count(key)) << "Missing auto proof field: " << key;
  return Number(record.at(key));
}
PrimExpr I64(int64_t value) { return IntImm(DataType::Int(64), value); }
Key Integers(const ObjectRef &object) {
  Key result;
  for (const ObjectRef &item : Downcast<Array<ObjectRef>>(object))
    result.push_back(Number(item));
  return result;
}
Array<Integer> Encode(const Key &values) {
  Array<Integer> result;
  for (int64_t value : values)
    result.push_back(Integer(IntImm(DataType::Int(64), value)));
  return result;
}
std::string Scope(const Record &record) {
  return Downcast<String>(record.at("scope"));
}
Var Data(const Record &record) { return Downcast<Var>(record.at("data")); }
bool Overlap(const Record &a, const Record &b) {
  return Data(a).same_as(Data(b)) && Field(a, "begin") < Field(b, "end") &&
         Field(b, "begin") < Field(a, "end");
}
Array<Record> Records(const Record &record, const char *key) {
  ICHECK(record.count(key)) << "Missing auto record array: " << key;
  return Downcast<Array<Record>>(record.at(key));
}
std::string Pipe(const Record &task, bool write) {
  std::string name = Downcast<String>(task.at("operation"));
  if (!write)
    return "MTE2";
  return name.find("l0c_to_gm") != std::string::npos ? "FIX" : "MTE3";
}
struct Segment {
  int64_t begin, end, producer;
};
void Define(std::vector<Segment> *segments, Segment incoming) {
  std::vector<Segment> result;
  for (const Segment &old : *segments) {
    if (old.end <= incoming.begin || incoming.end <= old.begin) {
      result.push_back(old);
      continue;
    }
    if (old.begin < incoming.begin)
      result.push_back({old.begin, incoming.begin, old.producer});
    if (incoming.end < old.end)
      result.push_back({incoming.end, old.end, old.producer});
  }
  result.push_back(incoming);
  std::sort(result.begin(), result.end(),
            [](auto a, auto b) { return a.begin < b.begin; });
  *segments = std::move(result);
}
Key Activation(const Record &storage, const Record &task) {
  Key result;
  if (Field(storage, "allocation_site_id") < 0)
    return result;
  ICHECK(storage.count("parent_loop_ids"))
      << "Missing auto allocation activation";
  Key coordinates = Integers(task.at("loop_coords"));
  ICHECK_EQ(coordinates.size() % 2, 0U);
  for (Integer loop : Downcast<Array<Integer>>(storage.at("parent_loop_ids"))) {
    bool found = false;
    for (size_t i = 0; i < coordinates.size(); i += 2)
      if (coordinates[i] == loop->value) {
        result.push_back(coordinates[i]);
        result.push_back(coordinates[i + 1]);
        found = true;
        break;
      }
    ICHECK(found) << "Missing auto allocation loop coordinate";
  }
  return result;
}
struct Channel {
  Var data{ObjectPtr<Object>()};
  bool cube_producer{false};
  Array<Record> roles, rounds;
  std::map<int64_t, std::vector<std::pair<int64_t, Record>>> producers;
};
Channel ChannelOf(const Array<Record> &tasks, bool noalias) {
  std::map<const VarNode *, std::array<bool, 4>> sides;
  std::map<const VarNode *, Var> variables;
  for (const Record &task : tasks)
    for (int write = 0; write < 2; ++write)
      for (const Record &access : Records(task, write ? "writes" : "reads")) {
        if (Scope(access) != "global")
          continue;
        Var data = Data(access);
        variables[data.get()] = data;
        int side = Field(task, "actor") == 0 ? 0 : 1;
        sides[data.get()][side + 2 * write] = true;
      }
  Channel result;
  for (const auto &[data, flags] : sides) {
    bool c2v = flags[2] && flags[1], v2c = flags[3] && flags[0];
    if (!c2v && !v2c)
      continue;
    ICHECK(!result.data.defined() && c2v != v2c && !(flags[2] && flags[3]))
        << "Auto supports one unambiguous fixed GM channel";
    result.data = variables.at(data);
    result.cube_producer = c2v;
  }
  struct Endpoint {
    Record task, access;
  };
  std::map<int64_t, std::array<std::vector<Endpoint>, 3>> groups;
  for (const Record &task : tasks) {
    int actor = Field(task, "actor");
    ICHECK(actor >= 0 && actor < 3) << "Invalid auto actor";
    for (int write = 0; write < 2; ++write) {
      std::vector<Record> spans;
      for (const Record &access : Records(task, write ? "writes" : "reads"))
        if (result.data.defined() && Data(access).same_as(result.data))
          spans.push_back(access);
      if (spans.empty())
        continue;
      bool producer = (actor == 0) == result.cube_producer;
      ICHECK(bool(write) == producer)
          << "Additional same-side GM channel effect";
      std::sort(spans.begin(), spans.end(),
                [](const Record &a, const Record &b) {
                  return Field(a, "begin") < Field(b, "begin");
                });
      Record endpoint = spans.front();
      int64_t end = Field(endpoint, "end");
      for (size_t i = 1; i < spans.size(); ++i) {
        ICHECK(Field(spans[i], "begin") == end &&
               StructuralEqual()(endpoint.at("dtype"), spans[i].at("dtype")))
            << "Auto channel rectangle must be dense and nonoverlapping";
        end = Field(spans[i], "end");
      }
      endpoint.Set("end", I64(end));
      groups[Field(task, "group")][actor].push_back({task, endpoint});
      result.roles.push_back({{"task", I64(Field(task, "id"))},
                              {"producer", Integer(producer)},
                              {"pipe", String(Pipe(task, producer))}});
    }
  }

  for (const auto &[group, endpoints] : groups) {
    ICHECK(endpoints[0].size() == endpoints[1].size() &&
           endpoints[0].size() == endpoints[2].size())
        << "Auto channel requires complete two-AIV FIFO participation";
    for (size_t round = 0; round < endpoints[0].size(); ++round) {
      const Endpoint &c = endpoints[0][round], &v0 = endpoints[1][round],
                     &v1 = endpoints[2][round];
      ICHECK(Integers(c.task.at("epoch")) == Integers(v0.task.at("epoch")) &&
             Integers(c.task.at("epoch")) == Integers(v1.task.at("epoch")))
          << "Auto GM FIFO epoch coordinates differ";
      ICHECK(StructuralEqual()(c.access.at("dtype"), v0.access.at("dtype")) &&
             StructuralEqual()(c.access.at("dtype"), v1.access.at("dtype")))
          << "Auto channel dtype mismatch";
      ICHECK(Field(c.access, "begin") == Field(v0.access, "begin") &&
             Field(v0.access, "end") == Field(v1.access, "begin") &&
             Field(v1.access, "end") == Field(c.access, "end"))
          << "Auto channel V regions must exactly partition the C footprint";
      for (int actor = 0; actor < 3; ++actor)
        ICHECK(Field(endpoints[actor][0].access, "begin") ==
                   Field(endpoints[actor][round].access, "begin") &&
               Field(endpoints[actor][0].access, "end") ==
                   Field(endpoints[actor][round].access, "end"))
            << "Auto caller workspace must remain fixed N1";
      Key ids{Field(c.task, "id"), Field(v0.task, "id"), Field(v1.task, "id")};
      result.rounds.push_back({{"group", I64(group)}, {"tasks", Encode(ids)}});
      for (int reader = 0; reader < 3; ++reader) {
        if ((reader == 0) == result.cube_producer)
          continue;
        for (int writer = 0; writer < 3; ++writer) {
          if ((writer == 0) != result.cube_producer)
            continue;
          const Endpoint &r = endpoints[reader][round],
                         &w = endpoints[writer][round];
          if (Overlap(r.access, w.access))
            result.producers[Field(r.task, "id")].push_back(
                {Field(w.task, "id"), w.access});
        }
      }
    }
  }
  for (size_t i = 0; i < tasks.size(); ++i)
    for (size_t j = i + 1; j < tasks.size(); ++j) {
      const Record &a = tasks[i], &b = tasks[j];
      bool same_group = Field(a, "group") == Field(b, "group");
      if (same_group && Field(a, "actor") == Field(b, "actor"))
        continue;
      for (int aw = 0; aw < 2; ++aw)
        for (int bw = 0; bw < 2; ++bw) {
          if (!aw && !bw)
            continue;
          for (const Record &x : Records(a, aw ? "writes" : "reads"))
            for (const Record &y : Records(b, bw ? "writes" : "reads")) {
              if (Scope(x) != "global" || Scope(y) != "global")
                continue;
              bool same = Data(x).same_as(Data(y));
              if (!same && noalias)
                continue;
              if (same && same_group && result.data.defined() &&
                  Data(x).same_as(result.data))
                continue;
              ICHECK(same && !Overlap(x, y))
                  << "Unprotected auto cross-core GM RAW/WAR/WAW";
            }
        }
    }
  return result;
}
} // namespace

AutoRecord BuildAscendAutoSourceContract(AutoRecord source) {
  Array<Record> tasks = Records(source, "tasks"),
                storages = Records(source, "storages");
  PrimFunc function = Downcast<PrimFunc>(source.at("source_function"));
  std::set<const VarNode *> global_roots;
  bool global_write = false;
  for (const Record &task : tasks)
    for (int write = 0; write < 2; ++write)
      for (const Record &e : Records(task, write ? "writes" : "reads"))
        if (Scope(e) == "global") {
          global_roots.insert(Data(e).get());
          global_write |= bool(write);
        }
  ICHECK(!global_write || global_roots.size() <= 1 ||
         function->HasNonzeroAttr("tir.noalias"))
      << "Auto requires tir.noalias for distinct writable GM parameters";
  Channel channel = ChannelOf(tasks, function->HasNonzeroAttr("tir.noalias"));
  std::map<const VarNode *, Record> catalog;
  for (const Record &storage : storages)
    ICHECK(catalog.emplace(Data(storage).get(), storage).second);
  using Domain = std::tuple<int64_t, int64_t, const VarNode *, Key>;
  std::map<Domain, std::vector<Segment>> ledger;
  std::set<const VarNode *> pinned;
  std::map<int64_t, Record> by_id;
  for (const Record &task : tasks)
    ICHECK(by_id.emplace(Field(task, "id"), task).second);
  Array<Record> reads, deps;
  std::set<std::pair<int64_t, int64_t>> seen;
  auto dependency = [&](int64_t src, int64_t dst, const char *kind) {
    if (src >= 0 && src != dst && seen.emplace(src, dst).second)
      deps.push_back({{"src", I64(src)},
                      {"dst", I64(dst)},
                      {"kind", String(kind)},
                      {"distance", I64(0)}});
  };
  for (const Record &task : tasks) {
    int64_t id = Field(task, "id");
    auto domain = [&](const Record &access) {
      Var data = Data(access);
      ICHECK(catalog.count(data.get())) << "Unknown auto source storage";
      Record storage = catalog.at(data.get());
      int64_t begin = Field(access, "begin"), end = Field(access, "end");
      ICHECK(begin >= 0 && end >= begin &&
             end <= Field(storage, "capacity_bytes"))
          << "Auto source footprint exceeds storage capacity";
      Domain result{Field(task, "group"), Field(task, "actor"), data.get(),
                    Activation(storage, task)};
      if (!ledger.count(result)) {
        std::vector<Segment> initial;
        if (Scope(access) == "global" &&
            Downcast<String>(storage.at("origin")) == "external_parameter")
          initial.push_back({0, Field(storage, "capacity_bytes"), -1});
        ledger.emplace(result, std::move(initial));
      }
      return result;
    };
    for (const Record &r : Records(task, "reads"))
      for (const Record &w : Records(task, "writes"))
        ICHECK(!Overlap(r, w) || (Field(r, "begin") == Field(w, "begin") &&
                                  Field(r, "end") == Field(w, "end")))
            << "Auto task shifted read/write overlap is unsupported";
    size_t index = 0;
    for (const Record &access : Records(task, "reads")) {
      Domain key = domain(access);
      int64_t begin = Field(access, "begin"), end = Field(access, "end");
      Array<Record> producers;
      std::vector<Segment> available = ledger[key];
      bool cross = channel.data.defined() && Data(access).same_as(channel.data);
      if (cross) {
        available.clear();
        ICHECK(channel.producers.count(id)) << "Missing auto channel producer";
        for (const auto &[writer, span] : channel.producers.at(id))
          available.push_back(
              {Field(span, "begin"), Field(span, "end"), writer});
        std::sort(available.begin(), available.end(),
                  [](auto a, auto b) { return a.begin < b.begin; });
      } else {
        Record storage = catalog.at(Data(access).get());
        if (available.empty() && Scope(access) == "global" &&
            Downcast<String>(storage.at("origin")) == "external_parameter")
          available.push_back({0, Field(storage, "capacity_bytes"), -1});
      }
      int64_t cursor = begin;
      for (const Segment &segment : available) {
        int64_t lo = std::max(begin, segment.begin),
                hi = std::min(end, segment.end);
        if (lo >= hi)
          continue;
        ICHECK(lo <= cursor) << "Auto read has no complete source definition";
        producers.push_back({{"task", I64(segment.producer)},
                             {"begin", I64(lo)},
                             {"end", I64(hi)}});
        dependency(segment.producer, id, "RAW");
        if (Scope(access) != "global" && segment.producer >= 0 &&
            Integers(by_id.at(segment.producer).at("loop_coords")) !=
                Integers(task.at("loop_coords")))
          pinned.insert(Data(access).get());
        cursor = std::max(cursor, hi);
      }
      ICHECK_EQ(cursor, end) << "Auto read has no complete source definition";
      reads.push_back({{"task", I64(id)},
                       {"read_index", I64(index++)},
                       {"data", Data(access)},
                       {"begin", I64(begin)},
                       {"end", I64(end)},
                       {"producers", producers}});
    }
    for (const Record &access : Records(task, "must_defines")) {
      Domain key = domain(access);
      Define(&ledger[key], {Field(access, "begin"), Field(access, "end"), id});
    }
    for (const Record &access : Records(task, "scratch")) {
      domain(access);
      pinned.insert(Data(access).get());
    }
  }
  for (size_t round = 0; round < channel.rounds.size(); ++round) {
    Record r = channel.rounds[round];
    Key ids = Integers(r.at("tasks"));
    if (round + 1 == channel.rounds.size() ||
        Field(channel.rounds[round + 1], "group") != Field(r, "group"))
      continue;
    Key next = Integers(channel.rounds[round + 1].at("tasks"));
    for (int reader = 0; reader < 3; ++reader)
      if ((reader == 0) != channel.cube_producer)
        for (int writer = 0; writer < 3; ++writer)
          if ((writer == 0) == channel.cube_producer)
            dependency(ids[reader], next[writer], "channel_FIFO");
  }
  Array<Var> pin;
  for (const VarNode *data : pinned)
    pin.push_back(GetRef<Var>(data));
  Record channel_record{{"roles", channel.roles}, {"rounds", channel.rounds}};
  if (channel.data.defined())
    channel_record.Set("data", channel.data);
  return {{"read_producers", reads},
          {"logical_deps", deps},
          {"pin_n1", pin},
          {"channel", channel_record}};
}

namespace {
int64_t Checked(__int128 value) {
  ICHECK(value >= std::numeric_limits<int64_t>::min() &&
         value <= std::numeric_limits<int64_t>::max())
      << "Auto byte arithmetic overflow";
  return int64_t(value);
}
class FinalInspector {
public:
  explicit FinalInspector(PrimFunc function) : function_(function) {
    int64_t groups = 1;
    int group_axes = 0, vector_axes = 0;
    PostOrderVisit(function->body, [&](const ObjectRef &object) {
      const auto *attr = object.as<AttrStmtNode>();
      if (!attr || attr->attr_key != "thread_extent")
        return;
      IterVar axis = Downcast<IterVar>(attr->node);
      int64_t n = Number(attr->value);
      ICHECK(n > 0 && n <= kAscendAutoScheduleTaskLimit)
          << "Invalid auto launch domain";
      if (axis->thread_tag == "blockIdx.x") {
        groups = n;
        ++group_axes;
      } else {
        ICHECK((axis->thread_tag == "blockIdx.y" ||
                axis->thread_tag == "threadIdx.x") &&
               n <= 2)
            << "Unsupported auto AIV launch mapping";
        vectors_ = n;
        ++vector_axes;
      }
    });
    ICHECK(group_axes <= 1 && vector_axes <= 1) << "Ambiguous auto launch axes";
    for (const auto &entry : function->buffer_map) {
      Buffer b = entry.second;
      AddAllocation(b->data, b->dtype, b->shape, b.scope());
    }
    for (group_ = 0; group_ < groups; ++group_)
      for (actor_ = 0; actor_ <= vectors_; ++actor_)
        Walk(function->body);
  }
  Record Result() {
    return {
        {"tasks", tasks_}, {"events", events_}, {"allocations", allocations_}};
  }

private:
  int64_t Value(const PrimExpr &expr) {
    return EvaluateAscendLifetimeInteger(expr, bindings_);
  }
  void AddAllocation(Var data, DataType dtype, Array<PrimExpr> shape,
                     const std::string &scope) {
    int64_t size = dtype.bytes();
    ICHECK(size > 0) << "Invalid auto allocation element type";
    for (const PrimExpr &dim : shape) {
      int64_t n = Value(dim);
      ICHECK(n > 0);
      size = Checked(__int128(size) * n);
    }
    auto found = allocation_sizes_.find(data.get());
    if (found != allocation_sizes_.end()) {
      ICHECK_EQ(found->second, size) << "Inconsistent actual auto allocation";
      return;
    }
    allocation_sizes_[data.get()] = size;
    allocations_.push_back({{"data", data},
                            {"scope", String(scope)},
                            {"capacity_bytes", I64(size)}});
  }
  void Event(const char *kind, int64_t task = -1, const std::string &pipe = "",
             int64_t id = -1) {
    ICHECK(scope_ >= 0) << "Auto hardware event outside explicit core scope";
    events_.push_back({{"group", I64(group_)},
                       {"actor", I64(actor_)},
                       {"kind", String(kind)},
                       {"task", I64(task)},
                       {"pipe", String(pipe)},
                       {"id", I64(id)}});
  }
  void EvaluateCall(const Call &original) {
    ValidateAscendLifetimeCallIndices(original, bindings_);
    arith::Analyzer analyzer;
    Call call =
        Downcast<Call>(analyzer.Simplify(Substitute(original, bindings_)));
    if (call->op.same_as(ascend_auto_barrier())) {
      ICHECK(active_ < 0 && call->args.size() == 1 &&
             call->args[0].as<StringImmNode>() &&
             Downcast<StringImm>(call->args[0])->value == "PIPE_ALL")
          << "Auto task boundaries require actual ALL fences";
      Event("ALL");
      return;
    }
    bool set = call->op.same_as(ascend_auto_set_cross_flag());
    bool wait = call->op.same_as(ascend_auto_wait_cross_flag());
    if (set || wait) {
      ICHECK(active_ < 0 && call->args.size() == (set ? 3U : 2U))
          << "Malformed auto cross event";
      if (set)
        ICHECK_EQ(Value(call->args[0]), 2)
            << "Auto cross protocol requires mode2";
      int64_t id = Value(call->args[set ? 2 : 0]);
      ICHECK(id == 0 || id == 1) << "Unowned auto cross event ID";
      const auto *pipe = call->args[1].as<StringImmNode>();
      ICHECK(pipe) << "Dynamic auto cross PIPE";
      std::string p = NormalizePipeName(pipe->value);
      ICHECK(actor_ == 0 ? (p == "MTE2" || p == "FIX")
                         : (p == "MTE2" || p == "MTE3"))
          << "Auto cross event on wrong core PIPE";
      Event(set ? "SET" : "WAIT", -1, p, id);
      return;
    }
    ICHECK(active_ >= 0) << "Unmarked/manual hardware effect in auto final IR";
    ICHECK(seen_.insert(active_).second) << "Duplicate auto task activation";
    Record task = AuditAscendAutoCall(call, actor_);
    task.Set("id", I64(active_));
    task.Set("group", I64(group_));
    task.Set("actor", I64(actor_));
    task.Set("call", call);
    tasks_.push_back(task);
    Event("TASK", active_);
  }
  void Walk(const Stmt &stmt) {
    ICHECK_LT(visits_++, size_t(kAscendAutoScheduleControlExpansionLimit))
        << "Auto final control budget exceeded";
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &s : seq->seq)
        Walk(s);
      return;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      if (attr->attr_key == "thread_extent") {
        IterVar axis = Downcast<IterVar>(attr->node);
        int64_t n = Value(attr->value);
        auto saved = bindings_;
        int64_t coordinate = axis->thread_tag == "blockIdx.x" ? group_
                             : actor_ > 0                     ? actor_ - 1
                                                              : 0;
        ICHECK(coordinate >= 0 && coordinate < n);
        bindings_.Set(axis->var, IntImm(axis->var.dtype(), coordinate));
        Walk(attr->body);
        bindings_ = saved;
        return;
      }
      if (attr->attr_key == "resource_scope") {
        int64_t scope = Value(attr->value);
        ICHECK(scope == 0 || scope == 1);
        if ((actor_ == 0) != (scope == 0))
          return;
        int saved = scope_;
        ICHECK(saved < 0 || saved == scope) << "Conflicting auto core scopes";
        scope_ = scope;
        Walk(attr->body);
        scope_ = saved;
        return;
      }
      if (attr->attr_key == kAscendAutoScheduleTaskId) {
        ICHECK(active_ < 0 && attr->node.as<IntImmNode>() &&
               Downcast<IntImm>(attr->node)->dtype == DataType::Int(64) &&
               attr->value.dtype() == DataType::Int(32) &&
               Value(attr->value) == 1)
            << "Malformed auto task marker";
        active_ = Number(attr->node);
        size_t before = tasks_.size();
        Walk(attr->body);
        ICHECK_EQ(tasks_.size(), before + 1)
            << "Auto task marker must contain one audited call";
        active_ = -1;
        return;
      }
      LOG(FATAL) << "Unsupported auto final attribute: " << attr->attr_key;
    }
    if (const auto *branch = stmt.as<IfThenElseNode>()) {
      if (Value(branch->condition))
        Walk(branch->then_case);
      else if (branch->else_case.defined())
        Walk(branch->else_case.value());
      return;
    }
    if (const auto *let = stmt.as<LetStmtNode>()) {
      ICHECK(let->var.dtype().is_int() || let->var.dtype().is_uint())
          << "Unmodeled auto scalar/pointer binding";
      auto saved = bindings_;
      bindings_.Set(
          let->var,
          IntImm(let->var.dtype(), Value(Cast(let->var.dtype(), let->value))));
      Walk(let->body);
      bindings_ = saved;
      return;
    }
    if (const auto *block = stmt.as<BlockRealizeNode>()) {
      ICHECK(block->iter_values.empty() &&
             block->block->match_buffers.empty() &&
             !block->block->init.defined())
          << "Unsupported auto final block";
      if (!Value(block->predicate))
        return;
      for (const Buffer &b : block->block->alloc_buffers)
        AddAllocation(b->data, b->dtype, b->shape, b.scope());
      Walk(block->block->body);
      return;
    }
    if (const auto *alloc = stmt.as<AllocateNode>()) {
      ICHECK_EQ(Value(alloc->condition), 1) << "Conditional auto allocation";
      AddAllocation(alloc->buffer_var, alloc->dtype, alloc->extents,
                    GetPtrStorageScope(alloc->buffer_var));
      Walk(alloc->body);
      return;
    }
    if (const auto *decl = stmt.as<DeclBufferNode>()) {
      // DeclBuffer is a view, not a second physical allocation.
      Walk(decl->body);
      return;
    }
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      if (is_zero(eval->value))
        return;
      ICHECK(eval->value.as<CallNode>()) << "Unmodeled auto scalar statement";
      EvaluateCall(Downcast<Call>(eval->value));
      return;
    }
    LOG(FATAL) << "Auto final IR retains unmaterialized/unsupported control: "
               << stmt->GetTypeKey();
  }
  PrimFunc function_;
  int64_t group_{0};
  int actor_{0}, vectors_{1}, scope_{-1};
  int64_t active_{-1};
  size_t visits_{0};
  Map<Var, PrimExpr> bindings_;
  std::map<const VarNode *, int64_t> allocation_sizes_;
  std::set<int64_t> seen_;
  Array<Record> tasks_, events_, allocations_;
};
int64_t PaddedSpan(const Record &storage) {
  if (Scope(storage) == "global")
    return Field(storage, "capacity_bytes");
  Buffer buffer = Downcast<Buffer>(storage.at("buffer"));
  ICHECK(buffer->strides.empty() && is_zero(buffer->elem_offset))
      << "Unsupported auto root allocation layout";
  int64_t outer = 1, last = 1;
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    int64_t n = EvaluateAscendLifetimeInteger(buffer->shape[i], {});
    ICHECK(n > 0);
    if (i + 1 == buffer->shape.size())
      last = n;
    else
      outer = Checked(__int128(outer) * n);
  }
  int64_t bytes = buffer->dtype.bytes();
  if (last == 1) {
    last = outer;
    outer = 1;
  }
  int64_t row = Checked(__int128(last) * bytes);
  row = Checked((__int128(row) + 31) / 32 * 32);
  return Checked(__int128(outer) * row);
}
std::vector<ObjectRef> Payload(const Call &call) {
  std::vector<ObjectRef> result;
  for (const PrimExpr &arg : call->args) {
    if (const auto *pointer = arg.as<CallNode>();
        pointer && pointer->op.same_as(builtin::tvm_access_ptr()))
      continue;
    if (arg.as<VarNode>())
      continue;
    if (arg.dtype().is_int() || arg.dtype().is_uint())
      result.push_back(I64(EvaluateAscendLifetimeInteger(arg, {})));
    else
      result.push_back(arg);
  }
  return result;
}
struct PhysicalSpan {
  int64_t begin, end;
  const VarNode *data;
  std::string scope;
};
} // namespace

AutoRecord InspectAscendAutoFinal(tir::PrimFunc function) {
  return FinalInspector(function).Result();
}

AutoRecord VerifyAscendAutoMapping(tir::PrimFunc function) {
  auto attr = function->GetAttr<Record>(kAscendAutoSchedule);
  ICHECK(attr.defined()) << "Missing auto schedule source contract";
  Record schedule = attr.value();
  ICHECK_EQ(Field(schedule, "schema_version"), 1);
  PrimFunc original = Downcast<PrimFunc>(schedule.at("source_function"));
  ICHECK(!original->GetAttr<ObjectRef>(kAscendAutoSchedule).defined())
      << "Recursive auto source contract";
  ICHECK_EQ(original->params.size(), function->params.size())
      << "Auto public parameter ABI changed";
  for (size_t i = 0; i < original->params.size(); ++i) {
    Var parameter = original->params[i];
    ICHECK(parameter.same_as(function->params[i]))
        << "Auto ordered public parameter role changed";
    ICHECK(original->buffer_map.count(parameter) ==
           function->buffer_map.count(parameter))
        << "Auto public parameter buffer ABI changed";
    if (original->buffer_map.count(parameter)) {
      Buffer expected = original->buffer_map[parameter],
             actual = function->buffer_map[parameter];
      ICHECK(expected->data.same_as(actual->data) &&
             expected->dtype == actual->dtype &&
             expected.scope() == actual.scope())
          << "Auto public parameter buffer ABI changed";
    }
  }
  ICHECK(StructuralEqual()(original->ret_type, function->ret_type))
      << "Auto return ABI changed";
  Record source = AnalyzeAscendAutoSource(original);
  Record contract = BuildAscendAutoSourceContract(source);
  Record actual = InspectAscendAutoFinal(function);
  Array<Record> original_tasks = Records(source, "tasks"),
                final_tasks = Records(actual, "tasks");
  int64_t actual_groups = 1, actual_vectors = 1;

  PostOrderVisit(function->body, [&](const ObjectRef &object) {
    const auto *a = object.as<AttrStmtNode>();
    if (!a || a->attr_key != "thread_extent")
      return;
    IterVar axis = Downcast<IterVar>(a->node);
    if (axis->thread_tag == "blockIdx.x")
      actual_groups = Number(a->value);
    else {
      actual_vectors = Number(a->value);
    }
  });
  bool empty = original_tasks.empty();
  if (empty) {
    ICHECK(final_tasks.empty() && Records(actual, "events").empty())
        << "Auto empty source acquired hardware tasks/events";
  } else {
    ICHECK(actual_groups == Field(source, "groups") &&
           actual_vectors == Field(source, "vector_extent"))
        << "Auto physical launch mapping changed after scheduling";
  }
  auto ratio = function->GetAttr<StringImm>("npu_cv_ratio");
  if (!empty && actual_vectors == 2)
    ICHECK(ratio.defined() && ratio.value()->value == "cv_1_2")
        << "Auto two-AIV launch requires the proven cv_1_2 mapping";
  if (actual_vectors == 1 &&
      std::any_of(original_tasks.begin(), original_tasks.end(),
                  [](const Record &t) { return Field(t, "actor") > 0; }))
    ICHECK(ratio.defined() && ratio.value()->value == "cv_1_1")
        << "Auto single-AIV launch requires an explicit cv_1_1 mapping";
  ICHECK_EQ(original_tasks.size(), final_tasks.size())
      << "Auto source/final task coverage changed";
  std::map<int64_t, Record> final_by_id;
  for (const Record &task : final_tasks)
    ICHECK(final_by_id.emplace(Field(task, "id"), task).second);
  std::map<const VarNode *, Record> storages, remaps, allocations;
  for (const Record &storage : Records(source, "storages"))
    storages.emplace(Data(storage).get(), storage);
  for (const Record &allocation : Records(actual, "allocations"))
    allocations.emplace(Data(allocation).get(), allocation);
  for (const auto &[data, storage] : storages)
    if (Scope(storage) == "global")
      ICHECK(allocations.count(data) &&
             Scope(allocations.at(data)) == "global" &&
             Field(allocations.at(data), "capacity_bytes") ==
                 Field(storage, "capacity_bytes"))
          << "Auto caller GM storage capacity/ABI changed";
  auto remap_attr =
      function->GetAttr<Array<Record>>(kAscendAutoScheduleBufferRemap);
  ICHECK(remap_attr.defined()) << "Missing actual auto buffer remap";
  std::set<const VarNode *> physical_owners;
  std::set<const VarNode *> pinned;
  for (const Var &data : Downcast<Array<Var>>(contract.at("pin_n1")))
    pinned.insert(data.get());
  for (const Record &remap : remap_attr.value()) {
    Var logical = Downcast<Var>(remap.at("logical_data")),
        physical = Downcast<Var>(remap.at("physical_data"));
    ICHECK(storages.count(logical.get()) &&
           remaps.emplace(logical.get(), remap).second)
        << "Unknown/duplicate auto version owner";
    physical_owners.insert(physical.get());
    Record storage = storages.at(logical.get());
    int64_t versions = Field(remap, "versions");
    ICHECK((versions == 1 || versions == 2) && Scope(storage) != "global")
        << "Unsupported auto version count/scope";
    ICHECK(!pinned.count(logical.get()) || versions == 1)
        << "Auto carry/scratch must remain N1";
    int64_t span = PaddedSpan(storage), pitch = Field(remap, "pitch"),
            total = Field(remap, "total");
    ICHECK(Field(remap, "span") == span && pitch == span &&
           total == Checked(__int128(pitch) * versions))
        << "Auto actual byte pitch/span/total mismatch";
    ICHECK(allocations.count(physical.get()) &&
           Scope(allocations.at(physical.get())) == Scope(storage))
        << "Auto actual allocation changed memory pool";
    ICHECK(allocations.count(physical.get()) &&
           Field(allocations.at(physical.get()), "capacity_bytes") >= total)
        << "Auto actual Allocate does not cover its versions";
    auto addresses = function->GetAttr<Map<Var, PrimExpr>>("address_map");
    auto sizes = function->GetAttr<Map<Var, PrimExpr>>("size_map");
    ICHECK(addresses.defined() && sizes.defined() &&
           addresses.value().count(physical) && sizes.value().count(physical))
        << "Missing auto actual physical pool mapping";
    int64_t base = Number(addresses.value()[physical]),
            size = Number(sizes.value()[physical]);
    ICHECK(Field(allocations.at(physical.get()), "capacity_bytes") <= size)
        << "Auto actual Allocate exceeds size_map physical capacity";
    ICHECK(base >= 0 && base % 32 == 0 && size >= total && size % 32 == 0 &&
           Checked(__int128(base) + size) <=
               AscendAutoMemoryBudget(String(Scope(storage))))
        << "Auto physical allocation exceeds aligned pool budget";
  }
  for (const auto &[data, allocation] : allocations)
    ICHECK(Scope(allocation) == "global" || physical_owners.count(data))
        << "Unowned auto actual local Allocate";
  auto interval = [&](const Record &task, const Record &access,
                      bool source_access) {
    Var data = Data(access);
    std::string scope = Scope(access);
    int64_t lo = Field(access, "begin"), hi = Field(access, "end");
    if (scope == "global")
      return PhysicalSpan{lo, hi, data.get(), scope};
    if (source_access) {
      ICHECK(remaps.count(data.get())) << "Missing auto local version owner";
      Record remap = remaps.at(data.get());
      int64_t n = Field(remap, "versions"), slot = -1;
      for (const Record &selection : Records(remap, "slots"))
        if (Field(selection, "task") == Field(task, "id")) {
          ICHECK(slot < 0) << "Duplicate auto task slot";
          slot = Field(selection, "slot");
        }
      ICHECK(slot >= 0 && slot < n)
          << "Missing/invalid auto finite task version slot";
      int64_t expected = 0;
      if (n > 1) {
        int64_t loop = Field(remap, "clock_loop"),
                minimum = Field(remap, "clock_min");
        bool found = false;
        for (const Record &domain : Records(source, "loops"))
          if (Field(domain, "id") == loop) {
            ICHECK_EQ(Field(domain, "min"), minimum)
                << "Auto version clock origin differs from source";
            ICHECK_GE(Field(domain, "num_stages"), n)
                << "Auto versions exceed requested source bound";
            found = true;
          }
        ICHECK(found) << "Unknown auto version clock";
        Key coords = Integers(task.at("loop_coords"));
        found = false;
        for (size_t i = 0; i < coords.size(); i += 2)
          if (coords[i] == loop) {
            int64_t delta = Checked(__int128(coords[i + 1]) - minimum);
            ICHECK(delta >= 0);
            expected = delta % n;
            found = true;
          }
        ICHECK(found) << "Auto version clock absent from task activation";
      }
      ICHECK_EQ(slot, expected) << "Wrong auto version selector";
      int64_t offset = Checked(__int128(slot) * Field(remap, "pitch"));
      lo = Checked(__int128(lo) + offset);
      hi = Checked(__int128(hi) + offset);
      data = Downcast<Var>(remap.at("physical_data"));
    }
    ICHECK(allocations.count(data.get()))
        << "Unknown actual auto local allocation";
    ICHECK(lo >= 0 && hi >= lo &&
           hi <= Field(allocations.at(data.get()), "capacity_bytes"))
        << "Auto final access exceeds actual Allocate";
    auto addresses = function->GetAttr<Map<Var, PrimExpr>>("address_map");
    auto sizes = function->GetAttr<Map<Var, PrimExpr>>("size_map");
    ICHECK(addresses.defined() && sizes.defined() &&
           addresses.value().count(data) && sizes.value().count(data))
        << "Auto final access has no physical address";
    int64_t base = Number(addresses.value()[data]),
            size = Number(sizes.value()[data]);
    ICHECK(base >= 0 && base % 32 == 0 && size % 32 == 0 && hi <= size &&
           Checked(__int128(base) + size) <=
               AscendAutoMemoryBudget(String(scope)))
        << "Invalid auto final physical footprint";
    return PhysicalSpan{Checked(__int128(base) + lo),
                        Checked(__int128(base) + hi), data.get(), scope};
  };
  std::map<int64_t, Record> source_by_id;
  for (const Record &task : original_tasks) {
    int64_t id = Field(task, "id");
    source_by_id.emplace(id, task);
    ICHECK(final_by_id.count(id)) << "Missing actual auto task";
    Record final = final_by_id.at(id);
    ICHECK(Field(task, "group") == Field(final, "group") &&
           Field(task, "actor") == Field(final, "actor"))
        << "Auto core/participant identity changed";
    Call source_call = Downcast<Call>(task.at("call")),
         final_call = Downcast<Call>(final.at("call"));
    ICHECK_EQ(source_call->args.size(), final_call->args.size())
        << "Auto call argument count changed";
    for (size_t argument = 0; argument < source_call->args.size(); ++argument) {
      const PrimExpr &a = source_call->args[argument],
                     &b = final_call->args[argument];
      if (const auto *ptr = a.as<CallNode>();
          ptr && ptr->op.same_as(builtin::tvm_access_ptr())) {
        const auto *other = b.as<CallNode>();
        ICHECK(other && other->op.same_as(builtin::tvm_access_ptr()) &&
               ptr->args.size() == 5 && other->args.size() == 5)
            << "Auto ordered pointer ABI changed";
        ICHECK(StructuralEqual()(ptr->args[0], other->args[0]) &&
               EvaluateAscendLifetimeInteger(ptr->args[3], {}) ==
                   EvaluateAscendLifetimeInteger(other->args[3], {}) &&
               EvaluateAscendLifetimeInteger(ptr->args[4], {}) ==
                   EvaluateAscendLifetimeInteger(other->args[4], {}))
            << "Auto pointer dtype/extent/mask changed";
        Var data = Downcast<Var>(ptr->args[1]);
        int64_t expected = EvaluateAscendLifetimeInteger(ptr->args[2], {});
        if (remaps.count(data.get())) {
          Record r = remaps.at(data.get());
          int64_t slot = -1;
          for (const Record &selection : Records(r, "slots"))
            if (Field(selection, "task") == id) {
              ICHECK(slot < 0);
              slot = Field(selection, "slot");
            }
          ICHECK(slot >= 0 && slot < Field(r, "versions"));
          int64_t bytes = ptr->args[0].dtype().bytes();
          ICHECK(bytes > 0 && Field(r, "pitch") % bytes == 0);
          expected = Checked(__int128(expected) +
                             __int128(slot) * Field(r, "pitch") / bytes);
          data = Downcast<Var>(r.at("physical_data"));
        }
        ICHECK(other->args[1].same_as(data) &&
               EvaluateAscendLifetimeInteger(other->args[2], {}) == expected)
            << "Auto ordered pointer role/selector changed";
      } else if (const auto *var = a.as<VarNode>()) {
        Var expected = GetRef<Var>(var);
        if (remaps.count(var))
          expected = Downcast<Var>(remaps.at(var).at("physical_data"));
        ICHECK(b.same_as(expected))
            << "Auto ordered scalar/buffer role changed";
      }
    }
    auto src_payload = Payload(Downcast<Call>(task.at("call"))),
         dst_payload = Payload(Downcast<Call>(final.at("call")));
    ICHECK_EQ(src_payload.size(), dst_payload.size())
        << "Auto call payload ABI changed";
    for (size_t i = 0; i < src_payload.size(); ++i)
      ICHECK(StructuralEqual()(src_payload[i], dst_payload[i]))
          << "Auto normalized call payload changed";
    ICHECK(Downcast<Call>(task.at("call"))
               ->op.same_as(Downcast<Call>(final.at("call"))->op))
        << "Auto call operation changed";
    for (const char *field : {"reads", "writes", "must_defines", "scratch"}) {
      auto a = Records(task, field), b = Records(final, field);
      ICHECK_EQ(a.size(), b.size()) << "Auto call effects changed";
      for (size_t i = 0; i < a.size(); ++i) {
        PhysicalSpan x = interval(task, a[i], true),
                     y = interval(final, b[i], false);
        ICHECK(x.data == y.data && x.scope == y.scope && x.begin == y.begin &&
               x.end == y.end &&
               StructuralEqual()(a[i].at("dtype"), b[i].at("dtype")))
            << "Auto actual owner/selector/physical interval differs from "
               "source";
      }
    }
    auto implicit = Records(task, "implicit_scratch"),
         actual_implicit = Records(final, "implicit_scratch");
    ICHECK_EQ(implicit.size(), actual_implicit.size());
    for (size_t i = 0; i < implicit.size(); ++i) {
      std::string pool = Scope(implicit[i]);
      ICHECK((pool == "wmma.matrix_a" || pool == "wmma.matrix_b") &&
             StructuralEqual()(implicit[i], actual_implicit[i]) &&
             Field(implicit[i], "bytes") > 0 &&
             Field(implicit[i], "bytes") <=
                 AscendAutoMemoryBudget(String(pool)))
          << "Invalid auto GEMM internal scratch";
      for (const auto &[unused, storage] : storages)
        ICHECK(Scope(storage) != pool)
            << "Auto GEMM internal scratch cannot share explicit L0 owners";
    }
    for (const Record &scratch : Records(final, "scratch")) {
      PhysicalSpan a = interval(final, scratch, false);
      for (const char *field : {"reads", "writes"})
        for (const Record &operand : Records(final, field)) {
          PhysicalSpan b = interval(final, operand, false);
          ICHECK(a.scope != b.scope || a.end <= b.begin || b.end <= a.begin)
              << "Auto reduction scratch aliases an operand";
        }
    }
    for (const Record &r : Records(final, "reads"))
      for (const Record &w : Records(final, "writes")) {
        PhysicalSpan a = interval(final, r, false),
                     b = interval(final, w, false);
        bool overlap = a.scope == b.scope && a.begin < b.end && b.begin < a.end;
        ICHECK(!overlap || (a.begin == b.begin && a.end == b.end))
            << "Auto physical shifted in-place overlap";
      }
  }
  // Independently check each original read still names the same physical bytes
  // as each of its reaching definitions, including N1 carry and partial unions.
  for (const Record &read : Records(contract, "read_producers")) {
    Record consumer = source_by_id.at(Field(read, "task"));
    Record access = Records(consumer, "reads")[Field(read, "read_index")];
    PhysicalSpan actual_read = interval(consumer, access, true);
    for (const Record &segment : Records(read, "producers")) {
      int64_t writer = Field(segment, "task");
      if (writer < 0)
        continue;
      Record producer = source_by_id.at(writer);
      bool found = false;
      for (const Record &definition : Records(producer, "must_defines")) {
        if (!Data(definition).same_as(Data(access)) ||
            Field(definition, "begin") > Field(segment, "begin") ||
            Field(definition, "end") < Field(segment, "end"))
          continue;
        Record part = definition;
        part.Set("begin", I64(Field(segment, "begin")));
        part.Set("end", I64(Field(segment, "end")));
        PhysicalSpan write = interval(producer, part, true);
        int64_t delta = Field(segment, "begin") - Field(access, "begin");
        ICHECK(write.data == actual_read.data &&
               write.begin == Checked(__int128(actual_read.begin) + delta))
            << "Auto read no longer consumes its source producer version";
        found = true;
      }
      ICHECK(found) << "Missing actual auto reaching definition";
    }
  }
  Record result{{"source", source}, {"contract", contract}, {"actual", actual}};
  return result;
}

namespace {
struct MemoryKey {
  std::string scope;
  int64_t group, actor;
  const VarNode *global;
  bool operator<(const MemoryKey &b) const {
    return std::tie(scope, group, actor, global) <
           std::tie(b.scope, b.group, b.actor, b.global);
  }
};
struct ActualSpan {
  MemoryKey key;
  int64_t begin, end;
};
struct ProofMachine {
  std::array<size_t, 3> pc{};
  std::array<std::set<int64_t>, 3> completed, issued;
  std::array<std::array<std::vector<std::set<int64_t>>, 2>, 3> tokens;
  std::array<std::array<std::vector<std::set<int64_t>>, 2>, 2> vector_sets;
};
} // namespace

void VerifyAscendAutoSyncProtocol(tir::PrimFunc function) {
  Record proof = VerifyAscendAutoMapping(function);
  Record source = Downcast<Record>(proof.at("source")),
         contract = Downcast<Record>(proof.at("contract")),
         actual = Downcast<Record>(proof.at("actual"));
  auto sync_attr = function->GetAttr<Record>(kAscendAutoScheduleSyncPlan);
  ICHECK(sync_attr.defined()) << "Missing new scheduled synchronization owner";
  Record metadata = sync_attr.value();
  ICHECK(
      Field(metadata, "schema_version") == 1 &&
      Field(metadata, "ready_id") == 0 && Field(metadata, "release_id") == 1 &&
      Downcast<String>(metadata.at("local_policy")) == "all_task_boundaries" &&
      Downcast<String>(metadata.at("cross_policy")) ==
          "mode2_fixed_n1_ready_release")
      << "Stale scheduled synchronization metadata";
  Record schedule = function->GetAttr<Record>(kAscendAutoSchedule).value();
  if (!Records(source, "tasks").empty()) {
    const auto *numeric =
        tvm::runtime::Registry::Get("tl.ascend.verify_schedule");
    ICHECK(numeric) << "Independent auto integer verifier is not registered";
    Record checked =
        (*numeric)(schedule.at("problem"), schedule.at("solution"));
    ICHECK(checked.count("valid") &&
           is_one(Downcast<PrimExpr>(checked.at("valid"))) &&
           Downcast<Array<ObjectRef>>(checked.at("errors")).empty())
        << "Invalid auto numeric schedule";
  } else {
    ICHECK(Records(actual, "tasks").empty() &&
           Records(actual, "events").empty())
        << "Zero-trip auto emits actual effects";
  }
  std::map<int64_t, Record> tasks, source_tasks;
  for (const Record &task : Records(actual, "tasks"))
    tasks.emplace(Field(task, "id"), task);
  for (const Record &task : Records(source, "tasks"))
    source_tasks.emplace(Field(task, "id"), task);
  Record channel = Downcast<Record>(contract.at("channel"));
  std::map<int64_t, Record> roles;
  for (const Record &role : Records(channel, "roles"))
    ICHECK(roles.emplace(Field(role, "task"), role).second);
  std::map<int64_t, std::array<std::vector<Record>, 3>> streams;
  for (const Record &event : Records(actual, "events"))
    streams[Field(event, "group")][Field(event, "actor")].push_back(event);
  for (auto &[group, actors] : streams)
    for (int actor = 0; actor < 3; ++actor) {
      const auto &events = actors[actor];
      std::vector<Record> stripped;
      for (size_t i = 0; i < events.size(); ++i) {
        std::string kind = Downcast<String>(events[i].at("kind"));
        if (kind == "TASK") {
          // A consumer wait is core-gating but is not the promised ALL fence.
          // The emitter places the fence before that wait for a consumer.
          ICHECK(
              i > 0 && i + 1 < events.size() &&
              Downcast<String>(events[i + 1].at("kind")) == "ALL" &&
              (Downcast<String>(events[i - 1].at("kind")) == "ALL" ||
               (i > 1 && Downcast<String>(events[i - 1].at("kind")) == "WAIT" &&
                Downcast<String>(events[i - 2].at("kind")) == "ALL")))
              << "Auto task/compound is missing its ALL completion boundary";
        }
        if (kind != "ALL")
          stripped.push_back(events[i]);
      }
      std::set<size_t> owned;
      std::vector<int64_t> issued_order;
      for (size_t i = 0; i < stripped.size(); ++i) {
        if (Downcast<String>(stripped[i].at("kind")) != "TASK")
          continue;
        int64_t id = Field(stripped[i], "task");
        issued_order.push_back(id);
        auto role = roles.find(id);
        if (role == roles.end())
          continue;
        bool producer = Field(role->second, "producer");
        std::string pipe = Downcast<String>(role->second.at("pipe"));
        if (producer) {
          ICHECK(i + 2 < stripped.size() &&
                 Downcast<String>(stripped[i + 1].at("kind")) == "SET" &&
                 Field(stripped[i + 1], "id") == 0 &&
                 Downcast<String>(stripped[i + 1].at("pipe")) == pipe &&
                 Downcast<String>(stripped[i + 2].at("kind")) == "WAIT" &&
                 Field(stripped[i + 2], "id") == 1 &&
                 Downcast<String>(stripped[i + 2].at("pipe")) == pipe)
              << "Auto producer lacks ready/immediate release drain";
          owned.insert(i + 1);
          owned.insert(i + 2);
        } else {
          ICHECK(i > 0 && i + 1 < stripped.size() &&
                 Downcast<String>(stripped[i - 1].at("kind")) == "WAIT" &&
                 Field(stripped[i - 1], "id") == 0 &&
                 Downcast<String>(stripped[i - 1].at("pipe")) == pipe &&
                 Downcast<String>(stripped[i + 1].at("kind")) == "SET" &&
                 Field(stripped[i + 1], "id") == 1 &&
                 Downcast<String>(stripped[i + 1].at("pipe")) == pipe)
              << "Auto consumer lacks ready/read/release";
          owned.insert(i - 1);
          owned.insert(i + 1);
        }
      }
      for (size_t i = 0; i < stripped.size(); ++i)
        ICHECK(Downcast<String>(stripped[i].at("kind")) == "TASK" ||
               owned.count(i))
            << "Unowned auto cross event";
      // The global numeric permutation is a dispatch specification, not a
      // hardware total order. Only its projection to each actor is observable.
      Key expected;
      Record solution = Downcast<Record>(schedule.at("solution"));
      for (int64_t id : Integers(solution.at("order"))) {
        ICHECK(source_tasks.count(id)) << "Unknown auto scheduled task";
        Record task = source_tasks.at(id);
        if (Field(task, "group") == group && Field(task, "actor") == actor)
          expected.push_back(id);
      }
      ICHECK(expected == issued_order)
          << "Actual auto actor order differs from verified solution";
    }
  auto addresses = function->GetAttr<Map<Var, PrimExpr>>("address_map");
  auto span = [&](const Record &task, const Record &effect) {
    std::string scope = Scope(effect);
    Var data = Data(effect);
    int64_t begin = Field(effect, "begin"), end = Field(effect, "end");
    if (scope == "global")
      return ActualSpan{{scope, -1, -1, data.get()}, begin, end};
    ICHECK(addresses.defined() && addresses.value().count(data));
    int64_t base = Number(addresses.value()[data]);
    return ActualSpan{
        {scope, Field(task, "group"), Field(task, "actor"), nullptr},
        Checked(__int128(base) + begin),
        Checked(__int128(base) + end)};
  };
  std::map<MemoryKey, std::vector<Segment>> memory;
  for (const Record &storage : Records(source, "storages"))
    if (Scope(storage) == "global" &&
        Downcast<String>(storage.at("origin")) == "external_parameter")
      memory[{"global", -1, -1, Data(storage).get()}] = {
          {0, Field(storage, "capacity_bytes"), -1}};
  std::map<std::pair<int64_t, int64_t>, Record> expected_reads;
  for (const Record &read : Records(contract, "read_producers"))
    expected_reads.emplace(
        std::make_pair(Field(read, "task"), Field(read, "read_index")), read);
  for (auto &[group, actors] : streams) {
    ProofMachine state;
    size_t remaining = 0;
    for (const auto &actor : actors)
      remaining += actor.size();
    while (remaining) {
      bool progress = false;
      for (int actor = 0; actor < 3; ++actor) {
        if (state.pc[actor] >= actors[actor].size())
          continue;
        Record event = actors[actor][state.pc[actor]];
        std::string kind = Downcast<String>(event.at("kind"));
        int64_t flag = Field(event, "id");
        if (kind == "WAIT" && state.tokens[actor][flag].empty())
          continue;
        if (kind == "TASK") {
          int64_t id = Field(event, "task");
          Record task = tasks.at(id);
          auto reads = Records(task, "reads");
          for (size_t index = 0; index < reads.size(); ++index) {
            ICHECK(expected_reads.count({id, int64_t(index)}));
            Record expected = expected_reads.at({id, int64_t(index)});
            ActualSpan read_span = span(task, reads[index]);
            for (const Record &producer : Records(expected, "producers")) {
              int64_t writer = Field(producer, "task");
              ICHECK(writer < 0 || state.completed[actor].count(writer))
                  << "Auto read source producer is not complete";
              int64_t lo =
                  Checked(__int128(read_span.begin) + Field(producer, "begin") -
                          Field(expected, "begin"));
              int64_t hi = Checked(__int128(read_span.begin) +
                                   Field(producer, "end") -
                                   Field(expected, "begin")),
                      cursor = lo;
              for (const Segment &last : memory[read_span.key]) {
                int64_t x = std::max(lo, last.begin),
                        y = std::min(hi, last.end);
                if (x >= y)
                  continue;
                ICHECK(x <= cursor && last.producer == writer)
                    << "Auto read sees stale/overwritten physical producer";
                cursor = std::max(cursor, y);
              }
              ICHECK_EQ(cursor, hi)
                  << "Auto read physical storage is not initialized";
            }
          }
          for (const char *field : {"writes", "scratch"})
            for (const Record &effect : Records(task, field)) {
              ActualSpan s = span(task, effect);
              Define(&memory[s.key], {s.begin, s.end, -2});
            }
          for (const Record &effect : Records(task, "must_defines")) {
            ActualSpan s = span(task, effect);
            Define(&memory[s.key], {s.begin, s.end, id});
          }
          state.issued[actor].insert(id);
        } else if (kind == "ALL") {
          state.completed[actor].insert(state.issued[actor].begin(),
                                        state.issued[actor].end());
        } else if (kind == "SET") {
          std::set<int64_t> completion = state.completed[actor];
          if (actor == 0) {
            for (int v = 1; v <= 2; ++v)
              state.tokens[v][flag].push_back(completion);
          } else {
            state.vector_sets[actor - 1][flag].push_back(completion);
            auto &a = state.vector_sets[0][flag],
                 &b = state.vector_sets[1][flag];
            while (!a.empty() && !b.empty()) {
              std::set<int64_t> joined = a.front();
              joined.insert(b.front().begin(), b.front().end());
              state.tokens[0][flag].push_back(joined);
              a.erase(a.begin());
              b.erase(b.begin());
            }
          }
          for (int core = 0; core < 3; ++core)
            ICHECK_LE(state.tokens[core][flag].size(), 1U)
                << "Auto N1 channel has multiple active epochs";
        } else if (kind == "WAIT") {
          auto &pending = state.tokens[actor][flag];
          state.completed[actor].insert(pending.front().begin(),
                                        pending.front().end());
          pending.erase(pending.begin());
        } else
          LOG(FATAL) << "Unsupported auto protocol operation";
        ++state.pc[actor];
        --remaining;
        progress = true;
      }
      ICHECK(progress) << "Auto joint C/V protocol cannot make progress";
    }
    for (int actor = 0; actor < 3; ++actor)
      for (int flag = 0; flag < 2; ++flag)
        ICHECK(state.tokens[actor][flag].empty())
            << "Auto cross protocol not drained at exit";
    for (int actor = 0; actor < 2; ++actor)
      for (int flag = 0; flag < 2; ++flag)
        ICHECK(state.vector_sets[actor][flag].empty())
            << "Auto missing second AIV release/ready participant";
  }
}

tir::transform::Pass AscendVerifyAutoSchedule(Target target, String platform) {
  auto pass = [=](PrimFunc function, IRModule,
                  tvm::transform::PassContext ctx) {
    if (ctx->GetConfig<String>("tl.ascend_pipeline_planner", String("legacy"))
            .value() != "auto")
      return function;
    ICHECK(platform == "A3" &&
           (target->GetAttr<String>("model").value_or("") == "ascendc" ||
            target->GetAttr<String>("model").value_or("") == "auto"));
    VerifyAscendAutoSyncProtocol(function);
    return function;
  };
  return tir::transform::CreatePrimFuncPass(pass, 0,
                                            "tl.AscendVerifyAutoSchedule", {});
}
TVM_REGISTER_GLOBAL("tl.transform.AscendVerifyAutoSchedule")
    .set_body_typed(AscendVerifyAutoSchedule);
TVM_REGISTER_GLOBAL("tl.transform._AscendVerifyAutoMapping")
    .set_body_typed([](PrimFunc function) {
      VerifyAscendAutoMapping(function);
      return function;
    });
} // namespace tl
} // namespace tvm
