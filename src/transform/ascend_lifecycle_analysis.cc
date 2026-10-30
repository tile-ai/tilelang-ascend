// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

#include "common/ascend_constraint.h"
#include "common/ascend_lifecycle.h"
#include "common/ascend_storage_alias.h"

namespace tvm {
namespace tl {
namespace {
using namespace tir;
using Record = Map<String, ObjectRef>;
using Key = std::vector<int64_t>;
constexpr size_t kLifetimeTaskLimit = 4096;
constexpr size_t kLifetimeControlLimit = 65536;

Integer Number(int64_t value) { return IntImm(DataType::Int(64), value); }
Array<Integer> Encode(const Key &key) {
  Array<Integer> result;
  for (int64_t value : key)
    result.push_back(Number(value));
  return result;
}
int64_t Field(const Record &record, const char *key) {
  const auto *value = record.at(key).as<IntImmNode>();
  ICHECK(value && value->dtype.is_int()) << "Invalid lifetime field: " << key;
  return value->value;
}

int64_t CheckedInteger(__int128 value) {
  ICHECK(value >= std::numeric_limits<int64_t>::min() &&
         value <= std::numeric_limits<int64_t>::max())
      << "Lifetime int64 arithmetic overflow before substitution";
  return static_cast<int64_t>(value);
}

// A singleton range can make the old simplifier fold overflow into zero.
// Evaluate the original expression before Substitute or simplification.
class CheckedIntegerEvaluator final
    : public ExprFunctor<int64_t(const PrimExpr &)> {
public:
  explicit CheckedIntegerEvaluator(Map<Var, PrimExpr> bindings)
      : bindings_(bindings) {}

private:
  int64_t VisitExpr_(const IntImmNode *op) final {
    ICHECK(!op->dtype.is_uint() || op->dtype.bits() < 64)
        << "Unsupported uint64 lifetime integer";
    return op->value;
  }
  int64_t VisitExpr_(const VarNode *op) final {
    Var variable = GetRef<Var>(op);
    ICHECK((op->dtype.is_int() || op->dtype.is_uint()) &&
           (!op->dtype.is_uint() || op->dtype.bits() < 64) &&
           bindings_.count(variable))
        << "Lifetime control/access requires a bounded static integer: "
        << variable;
    ICHECK(active_.insert(op).second) << "Cyclic lifetime integer binding";
    int64_t value = VisitExpr(bindings_[variable]);
    active_.erase(op);
    return value;
  }
  int64_t VisitExpr_(const CastNode *op) final {
    ICHECK(
        op->dtype.is_scalar() && op->dtype.bits() <= 64 &&
        (op->dtype.is_int() || (op->dtype.is_uint() && op->dtype.bits() < 64)))
        << "Unsupported lifetime integer cast";
    int64_t value = VisitExpr(op->value);
    if (op->dtype.is_bool())
      return value != 0;
    __int128 lower =
        op->dtype.is_uint() ? 0 : -(__int128(1) << (op->dtype.bits() - 1));
    __int128 upper =
        (__int128(1) << (op->dtype.bits() - (op->dtype.is_uint() ? 0 : 1))) - 1;
    ICHECK(value >= lower && value <= upper) << "Lossy lifetime integer cast";
    return value;
  }
#define LIFETIME_BINARY(Type, Expression)                                      \
  int64_t VisitExpr_(const Type##Node *op) final {                             \
    __int128 a = VisitExpr(op->a), b = VisitExpr(op->b);                       \
    return CheckedInteger(Expression);                                         \
  }
  LIFETIME_BINARY(Add, a + b)
  LIFETIME_BINARY(Sub, a - b)
  LIFETIME_BINARY(Mul, a *b)
  LIFETIME_BINARY(Min, std::min(a, b))
  LIFETIME_BINARY(Max, std::max(a, b))
  LIFETIME_BINARY(EQ, a == b)
  LIFETIME_BINARY(NE, a != b)
  LIFETIME_BINARY(LT, a < b)
  LIFETIME_BINARY(LE, a <= b)
  LIFETIME_BINARY(GT, a > b)
  LIFETIME_BINARY(GE, a >= b)
  LIFETIME_BINARY(And, a &&b)
  LIFETIME_BINARY(Or, a || b)
#undef LIFETIME_BINARY
  int64_t Divide(const PrimExpr &left, const PrimExpr &right, bool floor,
                 bool modulo) {
    __int128 a = VisitExpr(left), b = VisitExpr(right);
    ICHECK(b != 0) << "Lifetime integer division by zero";
    __int128 quotient = a / b;
    if (floor && a % b != 0 && ((a < 0) != (b < 0)))
      --quotient;
    return CheckedInteger(modulo ? a - quotient * b : quotient);
  }
  int64_t VisitExpr_(const DivNode *op) final {
    return Divide(op->a, op->b, false, false);
  }
  int64_t VisitExpr_(const ModNode *op) final {
    return Divide(op->a, op->b, false, true);
  }
  int64_t VisitExpr_(const FloorDivNode *op) final {
    return Divide(op->a, op->b, true, false);
  }
  int64_t VisitExpr_(const FloorModNode *op) final {
    return Divide(op->a, op->b, true, true);
  }
  int64_t VisitExpr_(const NotNode *op) final { return !VisitExpr(op->a); }
  int64_t VisitExpr_(const SelectNode *op) final {
    int64_t condition = VisitExpr(op->condition),
            yes = VisitExpr(op->true_value), no = VisitExpr(op->false_value);
    return condition ? yes : no;
  }
  int64_t VisitExpr_(const CallNode *op) final {
    ICHECK(op->op.same_as(builtin::if_then_else()) && op->args.size() == 3)
        << "Unsupported call in lifetime integer arithmetic";
    return VisitExpr(op->args[0]) ? VisitExpr(op->args[1])
                                  : VisitExpr(op->args[2]);
  }
  int64_t VisitExprDefault_(const Object *op) final {
    LOG(FATAL) << "Unsupported lifetime integer expression: "
               << op->GetTypeKey();
    return 0;
  }
  Map<Var, PrimExpr> bindings_;
  std::set<const VarNode *> active_;
};

class SourceMarker final : public StmtExprMutator {
public:
  explicit SourceMarker(const Record &graph) {
    for (const ObjectRef &item :
         Downcast<Array<ObjectRef>>(graph.at("tasks"))) {
      Record task = Downcast<Record>(item);
      units_.emplace(Downcast<Stmt>(task.at("statement")).get(),
                     Field(task, "id"));
    }
  }

private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == kAscendPipelineTaskHint)
      return VisitStmt(op->body);
    return StmtExprMutator::VisitStmt_(op);
  }
  Stmt VisitStmt_(const ForNode *op) final {
    int id = next_loop_++;
    loops_.push_back({id, op->loop_var});
    Stmt result = StmtExprMutator::VisitStmt_(op);
    loops_.pop_back();
    return result;
  }
  Stmt VisitStmt_(const EvaluateNode *op) final {
    auto found = units_.find(op);
    if (found == units_.end())
      return GetRef<Stmt>(op);
    Stmt body = GetRef<Stmt>(op);
    for (auto loop = loops_.rbegin(); loop != loops_.rend(); ++loop)
      body = AttrStmt(Number(loop->first), kAscendConstraintIteration,
                      cast(DataType::Int(64), loop->second), body);
    return AttrStmt(Number(found->second), kAscendConstraintTaskId,
                    IntImm(DataType::Int(32), 1), body);
  }
  std::unordered_map<const Object *, int64_t> units_;
  std::vector<std::pair<int, Var>> loops_;
  int next_loop_{0};
};

// Boolean scalar arguments select paths, never sizes, values or addresses.
class PredicateOnly final : public StmtVisitor {
  void Check(const PrimExpr &expression) {
    PostOrderVisit(expression, [](const ObjectRef &node) {
      const auto *var = node.as<VarNode>();
      ICHECK(!var || !var->dtype.is_bool())
          << "Lifetime bool scalars are supported only in predicates";
    });
  }
  void VisitStmt_(const EvaluateNode *op) final { Check(op->value); }
  void VisitStmt_(const ForNode *op) final {
    Check(op->min);
    Check(op->extent);
    StmtVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == "thread_extent")
      Check(op->value);
    StmtVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const AllocateNode *op) final {
    for (const PrimExpr &extent : op->extents)
      Check(extent);
    Check(op->condition);
    StmtVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const BlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers) {
      for (const PrimExpr &extent : buffer->shape)
        Check(extent);
      for (const PrimExpr &stride : buffer->strides)
        Check(stride);
      Check(buffer->elem_offset);
    }
    StmtVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const LetStmtNode *op) final {
    if (!op->var.dtype().is_bool())
      Check(op->value);
    StmtVisitor::VisitStmt_(op);
  }
};

struct Segment {
  int64_t begin, end;
  Key producer;
  int write_index;
  int64_t epoch;
};
struct Activation {
  Key key;
  std::vector<Segment> segments;
  int64_t next_epoch{0};
};

class SourceLifetimeTrace {
public:
  SourceLifetimeTrace(PrimFunc source, Record graph, Record catalog,
                      Record selection)
      : source_(source), graph_(graph), catalog_(catalog),
        selection_(selection) {
    tasks_ = Downcast<Array<ObjectRef>>(graph.at("tasks"));
    storages_ = Downcast<Array<ObjectRef>>(catalog.at("storages"));
    for (size_t i = 0; i < storages_.size(); ++i) {
      Record storage = Downcast<Record>(storages_[i]);
      ICHECK_EQ(Field(storage, "id"), int64_t(i));
      storage_ids_[Downcast<Var>(storage.at("data")).get()] = i;
    }
    int next_loop = 0;
    class CollectLoops final : public StmtVisitor {
    public:
      CollectLoops(std::map<const ForNode *, int> *ids, int *next)
          : ids_(ids), next_(next) {}
      void VisitStmt_(const ForNode *op) final {
        (*ids_)[op] = (*next_)++;
        StmtVisitor::VisitStmt_(op);
      }
      std::map<const ForNode *, int> *ids_;
      int *next_;
    } collect(&loop_ids_, &next_loop);
    collect(source_->body);
    bindings_ = Downcast<Map<Var, PrimExpr>>(selection.at("bool_bindings"));
    Walk(source_->body);
  }

  Record Result() {
    Record result = selection_;
    result.Set("task_keys", keys_);
    result.Set("reads", reads_);
    result.Set("writes", writes_);
    result.Set("activations", activation_records_);
    result.Set("control_visits", Number(visits_));
    return result;
  }

private:
  int64_t Value(const PrimExpr &expression) {
    return EvaluateAscendLifetimeInteger(expression, bindings_);
  }
  void AllocationBytes(DataType dtype, const Array<PrimExpr> &shape,
                       const Array<PrimExpr> &strides = {}) {
    int64_t product = dtype.bytes();
    for (const PrimExpr &extent : shape) {
      int64_t value = Value(extent);
      ICHECK_GT(value, 0) << "Invalid lifetime allocation extent";
      product = CheckedInteger(__int128(product) * value);
    }
    if (!strides.empty()) {
      ICHECK_EQ(strides.size(), shape.size());
      int64_t hull = dtype.bytes();
      for (size_t i = 0; i < strides.size(); ++i) {
        int64_t stride = Value(strides[i]);
        ICHECK_GE(stride, 0)
            << "Unsupported negative lifetime allocation stride";
        int64_t increment =
            CheckedInteger(__int128(Value(shape[i]) - 1) * stride);
        increment = CheckedInteger(__int128(increment) * dtype.bytes());
        hull = CheckedInteger(__int128(hull) + increment);
      }
      int64_t elements =
          CheckedInteger(__int128(Value(shape[0])) * Value(strides[0]));
      CheckedInteger(__int128(elements) * dtype.bytes());
    }
  }
  Record Storage(int id) {
    ICHECK(id >= 0 && size_t(id) < storages_.size());
    return Downcast<Record>(storages_[id]);
  }
  int StorageId(const Var &data) {
    auto found = storage_ids_.find(data.get());
    ICHECK(found != storage_ids_.end()) << "Unknown lifetime allocation";
    return found->second;
  }
  bool CallerEntry(int id) {
    Record storage = Storage(id);
    if (Field(storage, "allocation_site_id") >= 0)
      return false;
    if (storage.count("origin") &&
        Downcast<String>(storage.at("origin")) != "external_parameter")
      return false;
    Var data = Downcast<Var>(storage.at("data"));
    for (const auto &parameter : source_->buffer_map) {
      if (parameter.second->data.same_as(data))
        return true;
    }
    return false;
  }
  std::optional<Activation> Enter(int id) {
    Record storage = Storage(id);
    Key key{Field(storage, "allocation_site_id")};
    ICHECK_GE(key[0], 0) << "Lifetime requires a physical allocation site";
    for (Integer loop :
         Downcast<Array<Integer>>(storage.at("parent_loop_ids"))) {
      auto found = loops_.find(loop->value);
      ICHECK(found != loops_.end()) << "Missing lifetime allocation activation";
      key.push_back(loop->value);
      key.push_back(found->second);
    }
    std::optional<Activation> previous;
    auto old = active_.find(id);
    if (old != active_.end())
      previous = old->second;
    active_[id] = {key, {}, 0};
    activation_records_.push_back(
        Record{{"storage_id", Number(id)}, {"key", Encode(key)}});
    return previous;
  }
  void Exit(int id, const std::optional<Activation> &previous) {
    if (previous)
      active_[id] = *previous;
    else
      active_.erase(id);
  }
  std::pair<int64_t, int64_t> Bounds(const Record &access, int id) {
    int64_t begin = Value(Downcast<PrimExpr>(access.at("start"))),
            end = Value(Downcast<PrimExpr>(access.at("end"))),
            capacity =
                Value(Downcast<PrimExpr>(Storage(id).at("capacity_bytes")));
    ICHECK(begin >= 0 && end >= begin && end <= capacity)
        << "Lifetime access exceeds the canonical physical allocation";
    return {begin, end};
  }
  Activation &State(int id) {
    auto found = active_.find(id);
    ICHECK(found != active_.end())
        << "Storage access outside its lifetime activation";
    return found->second;
  }
  void Task(int id) {
    ICHECK(id >= 0 && size_t(id) < tasks_.size());
    ICHECK_LT(keys_.size(), kLifetimeTaskLimit)
        << "Lifetime hardware-task instance limit exceeded (4096)";
    Record task = Downcast<Record>(tasks_[id]);
    Key key{id};
    for (Integer loop : Downcast<Array<Integer>>(task.at("loop_ids"))) {
      ICHECK(loops_.count(loop->value)) << "Missing lifetime task iteration";
      key.push_back(loop->value);
      key.push_back(loops_.at(loop->value));
    }
    ICHECK(seen_.insert(key).second) << "Duplicate lifetime task instance";
    keys_.push_back(Encode(key));
    for (const ObjectRef &read_object :
         Downcast<Array<ObjectRef>>(task.at("reads"))) {
      Record read = Downcast<Record>(read_object);
      for (const ObjectRef &write_object :
           Downcast<Array<ObjectRef>>(task.at("writes"))) {
        Record write = Downcast<Record>(write_object);
        int storage = Field(read, "storage_id");
        if (storage != Field(write, "storage_id"))
          continue;
        auto r = Bounds(read, storage), w = Bounds(write, storage);
        ICHECK(r.second <= w.first || w.second <= r.first || r == w)
            << "Unsupported same-task shifted/partial read-write overlap";
      }
    }
    int read_index = 0;
    for (const ObjectRef &object :
         Downcast<Array<ObjectRef>>(task.at("reads"))) {
      Record access = Downcast<Record>(object);
      int storage = Field(access, "storage_id");
      auto [begin, end] = Bounds(access, storage);
      if (!CallerEntry(storage)) {
        Activation &state = State(storage);
        int64_t cursor = begin;
        Array<ObjectRef> pieces;
        for (const Segment &segment : state.segments) {
          if (cursor >= end)
            break;
          if (segment.end <= cursor)
            continue;
          ICHECK_LE(segment.begin, cursor)
              << "Lifetime read has an uninitialized byte/coverage hole";
          int64_t finish = std::min(end, segment.end);
          pieces.push_back(Record{{"begin", Number(cursor)},
                                  {"end", Number(finish)},
                                  {"producer_key", Encode(segment.producer)},
                                  {"write_index", Number(segment.write_index)},
                                  {"epoch", Number(segment.epoch)}});
          cursor = finish;
        }
        ICHECK_EQ(cursor, end)
            << "Lifetime read has no complete reaching definition";
        reads_.push_back(Record{{"consumer_key", Encode(key)},
                                {"read_index", Number(read_index)},
                                {"storage_id", Number(storage)},
                                {"activation", Encode(state.key)},
                                {"begin", Number(begin)},
                                {"end", Number(end)},
                                {"segments", pieces}});
      }
      ++read_index;
    }
    int write_index = 0;
    for (const ObjectRef &object :
         Downcast<Array<ObjectRef>>(task.at("writes"))) {
      Record access = Downcast<Record>(object);
      int storage = Field(access, "storage_id");
      auto [begin, end] = Bounds(access, storage);
      if (!CallerEntry(storage) && begin < end) {
        Activation &state = State(storage);
        std::vector<Segment> updated;
        for (const Segment &old : state.segments) {
          if (old.end <= begin || end <= old.begin) {
            updated.push_back(old);
          } else {
            if (old.begin < begin) {
              Segment left = old;
              left.end = begin;
              updated.push_back(left);
            }
            if (end < old.end) {
              Segment right = old;
              right.begin = end;
              updated.push_back(right);
            }
          }
        }
        int64_t epoch = state.next_epoch++;
        updated.push_back({begin, end, key, write_index, epoch});
        std::sort(updated.begin(), updated.end(),
                  [](const Segment &a, const Segment &b) {
                    return a.begin < b.begin;
                  });
        state.segments = std::move(updated);
        writes_.push_back(Record{{"writer_key", Encode(key)},
                                 {"write_index", Number(write_index)},
                                 {"storage_id", Number(storage)},
                                 {"activation", Encode(state.key)},
                                 {"begin", Number(begin)},
                                 {"end", Number(end)},
                                 {"epoch", Number(epoch)}});
      }
      ++write_index;
    }
  }
  void Walk(const Stmt &statement) {
    ICHECK_LT(visits_++, kLifetimeControlLimit)
        << "Lifetime control-visit limit exceeded (65536)";
    if (const auto *sequence = statement.as<SeqStmtNode>()) {
      for (const Stmt &child : sequence->seq)
        Walk(child);
    } else if (const auto *attribute = statement.as<AttrStmtNode>()) {
      if (attribute->attr_key == kAscendConstraintTaskId) {
        int previous = active_task_;
        ICHECK_LT(previous, 0) << "Nested lifetime task markers";
        active_task_ = Downcast<Integer>(attribute->node)->value;
        size_t before = keys_.size();
        Walk(attribute->body);
        ICHECK_EQ(keys_.size(), before + 1)
            << "Lifetime task marker must contain exactly one hardware call";
        active_task_ = previous;
      } else if (attribute->attr_key == "thread_extent") {
        IterVar axis = Downcast<IterVar>(attribute->node);
        Record launch = Downcast<Record>(
            Downcast<Map<String, ObjectRef>>(selection_.at("launches"))
                .at(axis->thread_tag));
        int64_t extent = Value(attribute->value);
        ICHECK_EQ(extent, Field(launch, "extent"));
        Map<Var, PrimExpr> saved = bindings_;
        bindings_.Set(axis->var,
                      IntImm(axis->var.dtype(), Field(launch, "coordinate")));
        Walk(attribute->body);
        bindings_ = saved;
      } else {
        // Schedule analysis has already audited all attributes and task hints.
        Walk(attribute->body);
      }
    } else if (const auto *loop = statement.as<ForNode>()) {
      ICHECK(loop->kind == ForKind::kSerial || loop->kind == ForKind::kUnrolled)
          << "Unsupported lifetime loop kind";
      int64_t minimum = Value(loop->min), extent = Value(loop->extent);
      ICHECK(extent >= 0 && extent <= int64_t(kLifetimeControlLimit))
          << "Lifetime loop exceeds static control limit";
      __int128 end = __int128(minimum) + extent;
      DataType dtype = loop->loop_var.dtype();
      ICHECK(dtype.is_int() && dtype.bits() <= 64);
      __int128 lower = -(__int128(1) << (dtype.bits() - 1)),
               upper = (__int128(1) << (dtype.bits() - 1)) - 1;
      ICHECK(minimum >= lower && end <= upper)
          << "Lifetime loop endpoint/update cannot be represented";
      int id = loop_ids_.at(loop);
      Map<Var, PrimExpr> saved = bindings_;
      for (int64_t index = 0; index < extent; ++index) {
        loops_[id] = minimum + index;
        bindings_.Set(loop->loop_var, IntImm(dtype, minimum + index));
        Walk(loop->body);
      }
      loops_.erase(id);
      bindings_ = saved;
    } else if (const auto *binding = statement.as<LetStmtNode>()) {
      Map<Var, PrimExpr> saved = bindings_;
      int64_t value = Value(Cast(binding->var.dtype(), binding->value));
      bindings_.Set(binding->var, IntImm(binding->var.dtype(), value));
      Walk(binding->body);
      bindings_ = saved;
    } else if (const auto *branch = statement.as<IfThenElseNode>()) {
      if (Value(branch->condition))
        Walk(branch->then_case);
      else if (branch->else_case.defined())
        Walk(branch->else_case.value());
    } else if (const auto *realize = statement.as<BlockRealizeNode>()) {
      if (!Value(realize->predicate))
        return;
      std::vector<std::pair<int, std::optional<Activation>>> saved;
      for (const Buffer &buffer : realize->block->alloc_buffers) {
        AllocationBytes(buffer->dtype, buffer->shape, buffer->strides);
        int storage = StorageId(buffer->data);
        saved.push_back({storage, Enter(storage)});
      }
      Walk(realize->block->body);
      for (auto previous = saved.rbegin(); previous != saved.rend(); ++previous)
        Exit(previous->first, previous->second);
    } else if (const auto *allocation = statement.as<AllocateNode>()) {
      ICHECK_EQ(Value(allocation->condition), 1)
          << "Unsupported conditional lifetime allocation";
      AllocationBytes(allocation->dtype, allocation->extents);
      int storage = StorageId(allocation->buffer_var);
      auto previous = Enter(storage);
      Walk(allocation->body);
      Exit(storage, previous);
    } else if (const auto *evaluate = statement.as<EvaluateNode>()) {
      if (is_zero(evaluate->value))
        return;
      ICHECK_GE(active_task_, 0) << "Unmarked lifetime hardware call";
      ValidateAscendLifetimeCallIndices(Downcast<Call>(evaluate->value),
                                        bindings_);
      Task(active_task_);
    } else {
      LOG(FATAL) << "Unsupported lifetime control: " << statement->GetTypeKey();
    }
  }

  PrimFunc source_;
  Record graph_, catalog_, selection_;
  Array<ObjectRef> tasks_, storages_, reads_, writes_, activation_records_;
  Array<Array<Integer>> keys_;
  std::map<const ForNode *, int> loop_ids_;
  std::unordered_map<const VarNode *, int> storage_ids_;
  Map<Var, PrimExpr> bindings_;
  std::map<int, int64_t> loops_;
  std::map<int, Activation> active_;
  std::set<Key> seen_;
  size_t visits_{0};
  int active_task_{-1};
};
} // namespace

int64_t EvaluateAscendLifetimeInteger(PrimExpr expression,
                                      Map<tir::Var, PrimExpr> bindings) {
  CheckedIntegerEvaluator evaluator(bindings);
  return evaluator(expression);
}

void ValidateAscendLifetimeCallIndices(tir::Call call,
                                       Map<tir::Var, PrimExpr> bindings) {
  PostOrderVisit(call, [&](const ObjectRef &object) {
    const auto *pointer = object.as<CallNode>();
    if (!pointer || !pointer->op.same_as(builtin::tvm_access_ptr()))
      return;
    ICHECK_EQ(pointer->args.size(), 5U) << "Malformed lifetime access pointer";
    int64_t offset = EvaluateAscendLifetimeInteger(pointer->args[2], bindings),
            extent = EvaluateAscendLifetimeInteger(pointer->args[3], bindings);
    ICHECK_GE(extent, 0) << "Negative lifetime pointer extent";
    DataType dtype = pointer->args[0].dtype();
    ICHECK(dtype.is_scalar() && dtype.bits() > 0 && dtype.bits() % 8 == 0)
        << "Unsupported lifetime pointer dtype";
    CheckedInteger(__int128(offset) * dtype.bytes());
    CheckedInteger(__int128(extent) * dtype.bytes());
  });
  const auto *operation = call->op.as<OpNode>();
  bool fill = operation && operation->name == "tl.ascend_fill";
  for (size_t i = 0; i < call->args.size(); ++i) {
    const PrimExpr &argument = call->args[i];
    // Fill's value is data, not an address/count. Its integer inputs need not
    // be statically known; all audited pointer/stride/count operands must be.
    if (fill && i == 2)
      continue;
    if (argument.dtype().is_int() || argument.dtype().is_uint())
      EvaluateAscendLifetimeInteger(argument, bindings);
  }
}

tir::PrimFunc MarkAscendScheduleSource(tir::PrimFunc source, Record graph) {
  source.CopyOnWrite()->body = SourceMarker(graph)(source->body);
  return source;
}

Record BuildAscendLifetimeContract(tir::PrimFunc source, Record graph,
                                   Record catalog) {
  PredicateOnly()(source->body);
  std::vector<Record> selections{
      Record{{"launches", Record()}, {"bool_bindings", Map<Var, PrimExpr>()}}};
  std::map<std::string, int64_t> domains;
  arith::Analyzer analyzer;
  for (const ObjectRef &item :
       Downcast<Array<ObjectRef>>(graph.at("launch_domains"))) {
    Record domain = Downcast<Record>(item);
    IterVar axis = Downcast<IterVar>(domain.at("iter_var"));
    PrimExpr extent_expr =
        analyzer.Simplify(Downcast<PrimExpr>(domain.at("extent")));
    const auto *extent = extent_expr.as<IntImmNode>();
    ICHECK(extent && extent->value > 0 && !axis->thread_tag.empty())
        << "Lifetime requires positive static launch domains";
    std::string tag = axis->thread_tag;
    if (domains.count(tag)) {
      ICHECK_EQ(domains.at(tag), extent->value);
      continue;
    }
    domains[tag] = extent->value;
    ICHECK(extent->value <= int64_t(kLifetimeTaskLimit / selections.size()))
        << "Lifetime launch/bool case limit exceeded (4096)";
    std::vector<Record> expanded;
    for (const Record &previous : selections) {
      for (int64_t coordinate = 0; coordinate < extent->value; ++coordinate) {
        Record choice = previous;
        Record launches = Downcast<Record>(choice.at("launches"));
        launches.Set(tag, Record{{"coordinate", Number(coordinate)},
                                 {"extent", Number(extent->value)}});
        choice.Set("launches", launches);
        expanded.push_back(choice);
      }
    }
    selections = std::move(expanded);
  }
  for (const Var &parameter : source->params) {
    if (!parameter.dtype().is_bool())
      continue;
    ICHECK_LE(selections.size(), kLifetimeTaskLimit / 2)
        << "Lifetime launch/bool case limit exceeded (4096)";
    std::vector<Record> expanded;
    for (const Record &previous : selections) {
      for (bool value : {false, true}) {
        Record choice = previous;
        auto bindings =
            Downcast<Map<Var, PrimExpr>>(choice.at("bool_bindings"));
        bindings.Set(parameter, Bool(value));
        choice.Set("bool_bindings", bindings);
        expanded.push_back(choice);
      }
    }
    selections = std::move(expanded);
  }
  Array<ObjectRef> cases;
  for (const Record &selection : selections)
    cases.push_back(
        SourceLifetimeTrace(source, graph, catalog, selection).Result());
  return {{"schema_version", Number(1)},
          {"source_function", source},
          {"source_graph", graph},
          {"storage_catalog", catalog},
          {"cases", cases}};
}

Record AnalyzeAscendLifetimes(tir::PrimFunc function, String platform) {
  ICHECK_EQ(platform, "A3") << "Lifetime analysis supports A3 only";
  PrimFunc canonical = CanonicalizeAscendStorageAliases(function);
  Record graph = AnalyzeAscendScheduleEffects(canonical, platform);
  Record catalog = BuildAscendStorageCatalog(canonical, graph);
  return BuildAscendLifetimeContract(MarkAscendScheduleSource(canonical, graph),
                                     graph, catalog);
}

Record RebuildAscendLifetimeContract(tir::PrimFunc marked_source) {
  class RemoveMarkers final : public StmtMutator {
    Stmt VisitStmt_(const AttrStmtNode *op) final {
      if (op->attr_key == kAscendConstraintTaskId ||
          op->attr_key == kAscendConstraintIteration)
        return VisitStmt(op->body);
      return StmtMutator::VisitStmt_(op);
    }
  } remove;
  marked_source.CopyOnWrite()->body = remove(marked_source->body);
  return AnalyzeAscendLifetimes(marked_source, "A3");
}

TVM_REGISTER_GLOBAL("tl.transform.AnalyzeAscendLifetimes")
    .set_body_typed(AnalyzeAscendLifetimes);
} // namespace tl
} // namespace tvm
