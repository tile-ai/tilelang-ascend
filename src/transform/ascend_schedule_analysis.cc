// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "common/ascend_storage_alias.h"
#include <limits>
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt_functor.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include "./common/ascend_constraint.h"
#include "./common/ascend_pipeline_structure.h"
#include "./common/ascend_resource.h"
#include "./common/ascend_schedule_hints.h"

namespace tvm {
namespace tl {
namespace {

using namespace tir;
using Record = Map<String, ObjectRef>;

// Rebuild arithmetic at its leaves. An outer int64 cast cannot repair int32
// multiplication, and retaining a narrow Cast would reintroduce that overflow.
class ScheduleIndexPromoter : public ExprMutator {
public:
  explicit ScheduleIndexPromoter(const Map<Var, Range> &domains) {
    for (const auto &entry : domains)
      proof_.Bind(entry.first, entry.second);
  }
  PrimExpr VisitExpr_(const VarNode *op) final {
    if (op->dtype.is_bool())
      return GetRef<PrimExpr>(op);
    if (op->dtype.is_int() || op->dtype.is_uint()) {
      ICHECK(op->dtype.is_int() || op->dtype.bits() < 64)
          << "Unsupported uint64 schedule index";
      return cast(DataType::Int(64), GetRef<Var>(op));
    }
    return GetRef<PrimExpr>(op);
  }
  PrimExpr VisitExpr_(const IntImmNode *op) final {
    if (op->dtype.is_bool())
      return GetRef<PrimExpr>(op);
    ICHECK(op->dtype.is_int() || (op->dtype.is_uint() && op->dtype.bits() < 64))
        << "Unsupported schedule integer literal";
    return IntImm(DataType::Int(64), op->value);
  }
  PrimExpr VisitExpr_(const CastNode *op) final {
    if (op->dtype.is_uint() && !op->dtype.is_bool() && op->dtype.bits() == 64) {
      LOG(FATAL) << "Unsupported uint64 schedule cast";
    }
    PrimExpr value = VisitExpr(op->value);
    if (op->dtype.is_int() || (op->dtype.is_uint() && !op->dtype.is_bool())) {
      const DataType source = op->value.dtype();
      bool lossless = (source.is_int() && op->dtype.is_int() &&
                       op->dtype.bits() >= source.bits()) ||
                      (source.is_uint() && op->dtype.is_int() &&
                       op->dtype.bits() > source.bits()) ||
                      (source.is_uint() && op->dtype.is_uint() &&
                       op->dtype.bits() >= source.bits());
      if (!lossless && op->dtype.bits() < 64) {
        int64_t lower =
            op->dtype.is_uint() ? 0 : -(int64_t(1) << (op->dtype.bits() - 1));
        int64_t upper =
            (int64_t(1) << (op->dtype.bits() - (op->dtype.is_uint() ? 0 : 1))) -
            1;
        if (const auto *constant = op->value.as<IntImmNode>()) {
          lossless = constant->value >= lower && constant->value <= upper;
        }
        if (!lossless &&
            (source.is_int() || (source.is_uint() && source.bits() < 64))) {
          auto bound = proof_.const_int_bound(value);
          lossless = bound->min_value >= lower && bound->max_value <= upper;
        }
      }
      ICHECK(lossless) << "Unsupported narrowing/lossy schedule index cast";
    }
    if (op->dtype.is_int() || (op->dtype.is_uint() && op->dtype.bits() < 64)) {
      return cast(DataType::Int(64), value);
    }
    return Cast(op->dtype, value);
  }
#define SCHEDULE_BINARY(Type)                                                  \
  PrimExpr VisitExpr_(const Type##Node *op) final {                            \
    return Type(VisitExpr(op->a), VisitExpr(op->b));                           \
  }
  SCHEDULE_BINARY(Add)
  SCHEDULE_BINARY(Sub)
  SCHEDULE_BINARY(Mul)
  SCHEDULE_BINARY(Div)
  SCHEDULE_BINARY(Mod)
  SCHEDULE_BINARY(FloorDiv)
  SCHEDULE_BINARY(FloorMod)
  SCHEDULE_BINARY(Min)
  SCHEDULE_BINARY(Max)
  SCHEDULE_BINARY(EQ)
  SCHEDULE_BINARY(NE)
  SCHEDULE_BINARY(LT)
  SCHEDULE_BINARY(LE)
  SCHEDULE_BINARY(GT)
  SCHEDULE_BINARY(GE)
  SCHEDULE_BINARY(And)
  SCHEDULE_BINARY(Or)
#undef SCHEDULE_BINARY
  PrimExpr VisitExpr_(const NotNode *op) final { return Not(VisitExpr(op->a)); }
  PrimExpr VisitExpr_(const SelectNode *op) final {
    return Select(VisitExpr(op->condition), VisitExpr(op->true_value),
                  VisitExpr(op->false_value));
  }
  PrimExpr VisitExpr_(const CallNode *op) final {
    ICHECK(op->op.same_as(builtin::if_then_else()))
        << "Unsupported call in schedule index/guard";
    Array<PrimExpr> args;
    for (const PrimExpr &arg : op->args)
      args.push_back(VisitExpr(arg));
    return Call(args[1].dtype(), op->op, args, op->span);
  }
  PrimExpr VisitExpr_(const BufferLoadNode *) final {
    LOG(FATAL) << "Unsupported memory-dependent schedule index/guard";
    return PrimExpr();
  }
  PrimExpr VisitExpr_(const RampNode *) final {
    LOG(FATAL) << "Unsupported Ramp schedule index";
    return PrimExpr();
  }

private:
  arith::Analyzer proof_;
};

struct Access {
  int storage;
  PrimExpr start, end;
  bool exact;
  bool bounded;
  Record AsRecord() const {
    return {{"storage_id", Integer(storage)},
            {"start", start},
            {"end", end},
            {"exact", Bool(exact)},
            {"bounded", Bool(bounded)}};
  }
};

struct Storage {
  Var data;
  std::string scope;
  PrimExpr capacity;
  Optional<PrimExpr> address;
  bool allocation{false};
};

struct Task {
  Record record;
  std::vector<Access> reads, writes;
};

struct Group {
  Stmt statement;
  Array<Integer> members, loops, path;
  bool complex{false};
  bool exact{true};
  int parent{-1};
  bool explicit_task{false};
  Optional<Integer> stage;
};

class ScheduleGraphBuilder {
public:
  explicit ScheduleGraphBuilder(PrimFunc function, String platform)
      : function_(function), platform_(platform), guard_(Bool(true)) {
    ICHECK(platform == "A3" || platform == "A2")
        << "Schedule analysis supports A2/A3 only";
    auto noalias = function->GetAttr<Bool>("tir.noalias");
    noalias_ = noalias.defined() && noalias.value()->value;
    for (const Var &parameter : function->params) {
      if (function->buffer_map.count(parameter))
        RegisterBuffer(function->buffer_map[parameter]);
    }
    auto addresses = function->GetAttr<Map<Var, PrimExpr>>("address_map");
    if (addresses.defined()) {
      for (const auto &entry : addresses.value()) {
        ICHECK(entry.second.as<IntImmNode>())
            << "Unsupported dynamic schedule address hint";
      }
      addresses_ = addresses.value();
    }
  }

  Record Build(bool include_dependencies = true) {
    ICHECK(!HasAscendScheduleHints(function_->body) || platform_ == "A3")
        << "Task/Stage analysis is supported only for A3";
    ValidateAscendScheduleHintStructure(function_);
    Walk(function_->body, -1);
    Array<ObjectRef> task_records, storage_records, group_records, dependencies;
    for (const Task &task : tasks_)
      task_records.push_back(task.record);
    for (size_t i = 0; i < storages_.size(); ++i) {
      const Storage &storage = storages_[i];
      Record record{{"id", Integer(i)},
                    {"data", storage.data},
                    {"scope", String(storage.scope)},
                    {"capacity_bytes", storage.capacity},
                    {"physical_allocation", Bool(storage.allocation)}};
      if (addresses_.count(storage.data))
        record.Set("explicit_address", addresses_[storage.data]);
      storage_records.push_back(record);
    }
    for (size_t i = 0; i < groups_.size(); ++i) {
      const Group &group = groups_[i];
      Record record{
          {"id", Integer(i)},
          {"statement", group.statement},
          {"task_ids", group.members},
          {"loop_ids", group.loops},
          {"source_path", group.path},
          {"parent_group_id", Integer(group.parent)},
          {"requires_sequential_iterations", Bool(!group.loops.empty())},
          {"schedulable",
           Bool(!group.complex && group.exact && group.members.size() == 1 &&
                group.loops.empty())}};
      if (group.explicit_task) {
        record.Set("explicit_task", Bool(true));
        if (group.stage.defined())
          record.Set("stage",
                     Integer(static_cast<int>(group.stage.value()->value)));
      }
      group_records.push_back(record);
    }
    if (include_dependencies)
      for (size_t i = 0; i < tasks_.size(); ++i) {
        for (size_t j = i + 1; j < tasks_.size(); ++j) {
          AddMemoryEdges(i, j, tasks_[i].writes, tasks_[j].reads, "RAW",
                         &dependencies);
          AddMemoryEdges(i, j, tasks_[i].reads, tasks_[j].writes, "WAR",
                         &dependencies);
          AddMemoryEdges(i, j, tasks_[i].writes, tasks_[j].writes, "WAW",
                         &dependencies);
          if (Downcast<Integer>(tasks_[i].record["group_id"])->value ==
              Downcast<Integer>(tasks_[j].record["group_id"])->value) {
            dependencies.push_back(Record{{"src", Integer(i)},
                                          {"dst", Integer(j)},
                                          {"kind", String("GROUP_ORDER")}});
          }
        }
      }
    return {{"schema_version", Integer(1)},
            {"platform", platform_},
            {"dependency_scope", String("single_iteration")},
            {"interval_unit", String("bytes")},
            {"tasks", task_records},
            {"storages", storage_records},
            {"groups", group_records},
            {"loops", loop_records_},
            {"bindings", bindings_},
            {"launch_domains", launch_domains_},
            {"dependencies", dependencies},
            {"noalias", Bool(noalias_)}};
  }

private:
  PrimExpr Wide(const PrimExpr &expr) {
    ScheduleIndexPromoter promoter(proof_domains_);
    return analyzer_.Simplify(promoter(expr));
  }
  void BindProofDomain(const Var &variable, const PrimExpr &minimum,
                       const PrimExpr &extent) {
    // A reused Var shadows the old range even when its new domain is unknown.
    proof_domains_.erase(variable);
    PrimExpr min_expr = analyzer_.Simplify(minimum),
             count_expr = analyzer_.Simplify(extent);
    const auto *min_value = min_expr.as<IntImmNode>();
    const auto *count_value = count_expr.as<IntImmNode>();
    DataType dtype = variable.dtype();
    if (!min_value || !count_value || count_value->value <= 0 ||
        !dtype.is_int() || dtype.bits() > 64)
      return;
    __int128 end = __int128(min_value->value) + count_value->value;
    int64_t lower = dtype.bits() == 64 ? std::numeric_limits<int64_t>::min()
                                       : -(int64_t(1) << (dtype.bits() - 1));
    int64_t upper = dtype.bits() == 64 ? std::numeric_limits<int64_t>::max()
                                       : (int64_t(1) << (dtype.bits() - 1)) - 1;
    if (min_value->value < lower || end > upper || count_value->value > upper)
      return;
    proof_domains_.Set(variable,
                       Range::FromMinExtent(IntImm(dtype, min_value->value),
                                            IntImm(dtype, count_value->value)));
  }
  PrimExpr Index(const PrimExpr &expr) {
    DataType dtype = expr.dtype();
    ICHECK(dtype.lanes() == 1 && !dtype.is_bool() &&
           ((dtype.is_int() && dtype.bits() <= 64) ||
            (dtype.is_uint() && dtype.bits() < 64)))
        << "Unsupported schedule integer index/count/cast";
    PrimExpr result = Wide(expr);
    ICHECK(result.dtype() == DataType::Int(64))
        << "Schedule index must remain int64";
    return result;
  }
  int Bytes(DataType dtype) {
    ICHECK(dtype.lanes() == 1 &&
           (dtype == DataType::Float(16) || dtype == DataType::Float(32)))
        << "Unsupported schedule dtype; expected scalar float16/float32";
    return dtype.bits() / 8;
  }
  int RegisterStorage(Var data, std::string scope, PrimExpr capacity,
                      bool allocation = false) {
    auto it = storage_ids_.find(data.get());
    if (it != storage_ids_.end()) {
      Storage &old = storages_[it->second];
      ICHECK_EQ(old.scope, scope) << "Inconsistent schedule storage scope";
      if (allocation) {
        ICHECK(!old.allocation || analyzer_.CanProve(old.capacity == capacity))
            << "Conflicting physical schedule allocations";
        old.capacity = capacity;
        old.allocation = true;
      } else if (capacity.defined() && !old.allocation) {
        old.capacity = old.capacity.defined()
                           ? analyzer_.Simplify(max(old.capacity, capacity))
                           : capacity;
      }
      return it->second;
    }
    int id = storages_.size();
    storage_ids_[data.get()] = id;
    storages_.push_back({data, scope,
                         capacity.defined() ? capacity : Integer(-1), NullOpt,
                         allocation});
    return id;
  }
  void RegisterBuffer(const Buffer &buffer, bool allocation = false) {
    PrimExpr size = IntImm(DataType::Int(64), 1);
    if (!buffer->strides.empty()) {
      size = Index(buffer->shape[0]) * Index(buffer->strides[0]);
    } else {
      for (const PrimExpr &extent : buffer->shape)
        size = size * Index(extent);
    }
    // Registration is independent of the operation allow-list. Unsupported
    // allocation dtypes can remain in a function without becoming a task.
    int bytes = (buffer->dtype.bits() * buffer->dtype.lanes() + 7) / 8;
    RegisterStorage(
        buffer->data, buffer.scope(),
        (allocation ? Index(size) : Index(buffer->elem_offset) + Index(size)) *
            Integer(bytes),
        allocation);
  }
  int NewGroup(const Stmt &statement, Array<Integer> path = {}) {
    int id = groups_.size();
    groups_.push_back(
        {statement, {}, loop_stack_, path, false, true, group_parent_});
    return id;
  }
  void WalkSequence(const SeqStmt &sequence, int group) {
    if (group >= 0) {
      for (const Stmt &statement : sequence->seq)
        Walk(statement, group);
      return;
    }
    for (const Stmt &statement : sequence->seq) {
      if (const auto *realize = statement.as<BlockRealizeNode>()) {
        if (is_one(realize->predicate) &&
            realize->block->body.as<SeqStmtNode>()) {
          ReadBlockMetadata(realize->block);
        }
      }
    }
    for (const auto &element : CollectAscendPipelineGroups(sequence)) {
      if (element.allocation.defined()) {
        RegisterBuffer(element.allocation, true);
      } else {
        Walk(element.statement,
             NewGroup(element.statement, element.source_path));
      }
    }
  }
  void ReadBlockMetadata(const Block &block) {
    ICHECK(block->match_buffers.empty() && block->iter_vars.empty() &&
           !block->init.defined())
        << "Unsupported unlowered schedule block";
    auto address = block->annotations.Get("address_map");
    if (address.defined()) {
      for (const auto &entry : Downcast<Map<Var, PrimExpr>>(address.value())) {
        ICHECK(entry.second.as<IntImmNode>())
            << "Unsupported dynamic schedule address hint";
        addresses_.Set(entry.first, entry.second);
      }
    }
  }
  void Walk(const Stmt &statement, int group) {
    if (const auto *sequence = statement.as<SeqStmtNode>()) {
      WalkSequence(GetRef<SeqStmt>(sequence), group);
    } else if (const auto *realize = statement.as<BlockRealizeNode>()) {
      ReadBlockMetadata(realize->block);
      for (const Buffer &buffer : realize->block->alloc_buffers)
        RegisterBuffer(buffer, true);
      PrimExpr saved = guard_;
      guard_ = analyzer_.Simplify(And(guard_, Wide(realize->predicate)));
      Walk(realize->block->body, group);
      guard_ = saved;
    } else if (const auto *attr = statement.as<AttrStmtNode>()) {
      if (attr->attr_key == kAscendStageHint) {
        int parent = group_parent_;
        if (group >= 0)
          group_parent_ = group;
        Optional<Integer> previous_stage = stage_;
        stage_ = Downcast<Integer>(attr->node);
        Walk(attr->body, -1);
        stage_ = previous_stage;
        group_parent_ = parent;
        return;
      }
      if (attr->attr_key == kAscendTaskHint) {
        int parent = group_parent_;
        if (group < 0 || !groups_[group].statement.same_as(statement)) {
          if (group >= 0)
            group_parent_ = group;
          group = NewGroup(statement);
        }
        groups_[group].explicit_task = true;
        groups_[group].stage = stage_;
        ++task_hint_depth_;
        Walk(attr->body, group);
        --task_hint_depth_;
        ICHECK_EQ(groups_[group].members.size(), 1U)
            << "T.Task requires exactly one audited hardware call";
        group_parent_ = parent;
        return;
      }
      if (attr->attr_key == kAscendPipelineTaskHint) {
        // CombineCV can leave a metadata wrapper around the empty opposite
        // core branch. It is not a task and must not acquire a phantom ID.
        if (const auto *empty = attr->body.as<EvaluateNode>()) {
          if (is_zero(empty->value))
            return;
        }
        ICHECK(attr->node.as<runtime::MapNode>() &&
               attr->value.dtype() == DataType::Int(32) && is_one(attr->value))
            << "Malformed pipeline task metadata";
        auto metadata = Downcast<Map<String, ObjectRef>>(attr->node);
        ICHECK(metadata.size() == 1 && metadata.count("stage"))
            << "Unsupported pipeline task metadata fields";
        const auto *stage = metadata["stage"].as<IntImmNode>();
        ICHECK(stage && stage->dtype == DataType::Int(64) &&
               stage->value >= 0 &&
               stage->value <= std::numeric_limits<int32_t>::max())
            << "Malformed pipeline task stage";
        int parent = group_parent_;
        if (group < 0 || !groups_[group].statement.same_as(statement)) {
          if (group >= 0)
            group_parent_ = group;
          group = NewGroup(statement);
        }
        Optional<Integer> previous = manual_stage_;
        manual_stage_ = GetRef<Integer>(stage);
        groups_[group].explicit_task = true;
        groups_[group].stage = manual_stage_;
        ++task_hint_depth_;
        Walk(attr->body, group);
        --task_hint_depth_;
        ICHECK_EQ(groups_[group].members.size(), 1U)
            << "Pipeline task metadata must wrap one audited hardware call";
        manual_stage_ = previous;
        group_parent_ = parent;
        return;
      }
      int saved = core_;
      Map<Var, Range> saved_domains = proof_domains_;
      if (attr->attr_key == "resource_scope") {
        const auto *value = attr->value.as<IntImmNode>();
        ICHECK(value && (value->value == 0 || value->value == 1))
            << "Invalid resource_scope";
        ICHECK(core_ < 0 || core_ == value->value)
            << "Conflicting schedule resource scope";
        core_ = value->value;
      } else if (attr->attr_key == "thread_extent") {
        launch_domains_.push_back(
            Record{{"iter_var", attr->node}, {"extent", Index(attr->value)}});
        if (const auto *axis = attr->node.as<IterVarNode>())
          BindProofDomain(axis->var, make_zero(axis->var.dtype()), attr->value);
      } else {
        LOG(FATAL) << "Unsupported schedule attribute: " << attr->attr_key;
      }
      // Outside a candidate pipeline loop, C/V ownership is a container, not
      // an injector statement unit. Preserve that container as a parent group
      // and expose its child statements. Inside loops, never split the unit.
      int parent = group_parent_;
      if (!task_hint_depth_ && attr->attr_key == "resource_scope" &&
          loop_stack_.empty() && group >= 0) {
        group_parent_ = group;
        Walk(attr->body, -1);
      } else {
        Walk(attr->body, group);
      }
      group_parent_ = parent;
      core_ = saved;
      proof_domains_ = saved_domains;
    } else if (const auto *allocation = statement.as<AllocateNode>()) {
      ICHECK(is_one(allocation->condition))
          << "Unsupported conditional schedule allocation";
      PrimExpr capacity = IntImm(DataType::Int(64), allocation->dtype.bytes());
      for (const PrimExpr &extent : allocation->extents)
        capacity = capacity * Index(extent);
      RegisterStorage(allocation->buffer_var,
                      tir::GetPtrStorageScope(allocation->buffer_var), capacity,
                      true);
      Walk(allocation->body, group);
    } else if (const auto *loop = statement.as<ForNode>()) {
      ICHECK(loop->kind == ForKind::kSerial || loop->kind == ForKind::kUnrolled)
          << "Unsupported schedule loop kind";
      if (group >= 0)
        groups_[group].complex = true;
      int id = loop_records_.size();
      loop_records_.push_back(Record{{"id", Integer(id)},
                                     {"var", loop->loop_var},
                                     {"min", Index(loop->min)},
                                     {"extent", Index(loop->extent)},
                                     {"parent_loop_ids", loop_stack_}});
      loop_stack_.push_back(Integer(id));
      Map<Var, Range> saved_domains = proof_domains_;
      BindProofDomain(loop->loop_var, loop->min, loop->extent);
      Walk(loop->body, -1);
      proof_domains_ = saved_domains;
      loop_stack_.pop_back();
    } else if (const auto *branch = statement.as<IfThenElseNode>()) {
      if (group < 0)
        group = NewGroup(statement);
      if (group >= 0)
        groups_[group].complex = true;
      PrimExpr saved = guard_;
      PrimExpr condition = Wide(branch->condition);
      guard_ = analyzer_.Simplify(And(saved, condition));
      Walk(branch->then_case, group);
      if (branch->else_case.defined()) {
        guard_ = analyzer_.Simplify(And(saved, Not(condition)));
        Walk(branch->else_case.value(), group);
      }
      guard_ = saved;
    } else if (const auto *let = statement.as<LetStmtNode>()) {
      if (group < 0)
        group = NewGroup(statement);
      ICHECK(let->var.dtype().is_int() || let->var.dtype().is_uint() ||
             let->var.dtype().is_bool())
          << "Unsupported pointer/scalar schedule let";
      Wide(let->value); // Reject memory-dependent or opaque expressions.
      bindings_.push_back(Record{
          {"var", let->var}, {"value", Wide(let->value)}, {"guard", guard_}});
      if (group >= 0)
        groups_[group].complex = true;
      Walk(let->body, group);
    } else if (const auto *evaluate = statement.as<EvaluateNode>()) {
      if (is_zero(evaluate->value))
        return;
      const auto *call = evaluate->value.as<CallNode>();
      ICHECK(call) << "Unsupported scalar schedule statement";
      AddTask(GetRef<Call>(call), statement,
              group < 0 ? NewGroup(statement) : group);
    } else {
      LOG(FATAL) << "Unsupported schedule statement: "
                 << statement->GetTypeKey();
    }
  }
  Access Pointer(const PrimExpr &value, int mask, PrimExpr count) {
    const auto *pointer = value.as<CallNode>();
    ICHECK(pointer && pointer->op.same_as(builtin::tvm_access_ptr()) &&
           pointer->args.size() == 5)
        << "Malformed schedule tvm_access_ptr";
    const auto *data = pointer->args[1].as<VarNode>();
    const auto *access_mask = pointer->args[4].as<IntImmNode>();
    ICHECK(data && access_mask && access_mask->value > 0 &&
           access_mask->value <= 3 && (access_mask->value & mask) == mask)
        << "Malformed schedule pointer access mask";
    DataType dtype = pointer->args[0].dtype();
    int bytes = Bytes(dtype);
    auto registered = storage_ids_.find(data);
    ICHECK(registered != storage_ids_.end())
        << "Unregistered schedule storage/raw pointer";
    int storage = registered->second;
    PrimExpr start =
        CheckedAscendStorageByteOffset(Index(pointer->args[2]), bytes);
    PrimExpr extent = CheckedAscendStorageByteOffset(Index(count), bytes);
    PrimExpr declared = Index(pointer->args[3]);
    ICHECK(!analyzer_.CanProve(start < 0))
        << "Unsupported negative schedule access";
    ICHECK(!analyzer_.CanProve(Index(count) < 0) &&
           !analyzer_.CanProve(Index(count) > declared))
        << "Schedule access exceeds declared pointer extent";
    ICHECK(analyzer_.CanProve(floormod(start, Integer(32)) == 0))
        << "Unsupported unaligned schedule access";
    PrimExpr end = Index(AddAscendStorageByteOffsets(start, extent));
    bool bounded = analyzer_.CanProve(Index(count) >= 0 &&
                                      Index(count) <= declared && start >= 0);
    const Storage &allocation = storages_[storage];
    if (allocation.scope != "global") {
      ICHECK(allocation.allocation)
          << "Unsupported local storage without physical allocation";
      ICHECK(!analyzer_.CanProve(end > allocation.capacity) &&
             !analyzer_.CanProve(end < start))
          << "Schedule access exceeds physical allocation";
      bounded = bounded &&
                analyzer_.CanProve(end <= allocation.capacity && end >= start);
    }
    return {storage, Index(start), end,
            bounded && count.as<IntImmNode>() != nullptr &&
                pointer->args[2].as<IntImmNode>() != nullptr,
            bounded};
  }
  void AddTask(const Call &call, const Stmt &statement, int group) {
    const auto *op_node = call->op.as<OpNode>();
    std::string operation =
        op_node ? std::string(op_node->name) : "opaque call";
    ICHECK(core_ == 1) << "Schedule operation requires explicit V owner: "
                       << operation;
    OperationConfig config;
    int begin = 0;
    PrimExpr count;
    if (call->op.same_as(builtin::call_extern())) {
      ICHECK(!call->args.empty()) << "Malformed schedule extern";
      const auto *callee = call->args[0].as<StringImmNode>();
      ICHECK(callee) << "Malformed schedule extern";
      operation = callee->value;
      std::string name = NormalizeFunctionName(callee->value);
      ICHECK(name == "copy_gm_to_ub" || name == "copy_ub_to_gm")
          << "Unsupported schedule extern/compound helper: " << callee->value;
      const bool load = name == "copy_gm_to_ub";
      ICHECK_EQ(call->args.size(), load ? 8U : 7U)
          << "Unsupported schedule DMA geometry/padding";
      PostOrderVisit(call->args[3], [](const ObjectRef &node) {
        ICHECK(!node.as<BufferLoadNode>() && !node.as<CallNode>())
            << "Unsupported memory-dependent/opaque DMA stride";
      });
      if (load) {
        PostOrderVisit(call->args[6], [](const ObjectRef &node) {
          ICHECK(!node.as<BufferLoadNode>() && !node.as<CallNode>())
              << "Unsupported memory-dependent/opaque DMA padding scalar";
        });
      }
      Index(call->args[3]);
      count = Index(call->args[5]);
      const auto *size = count.as<IntImmNode>();
      ICHECK(size && size->value > 0 &&
             analyzer_.CanProve(Index(call->args[4]) == 1) &&
             analyzer_.CanProve(count <= Index(call->args[load ? 7 : 6])))
          << "Unsupported dynamic DMA padding or partial tile";
      // The trailing shape is view metadata, not the emitted helper template.
      // A full aligned subview tile may be smaller; the exact template check
      // below still excludes padding/compound helpers and hidden PIPE effects.
      auto *pointer = call->args[1].as<CallNode>();
      ICHECK(pointer && pointer->args.size() == 5)
          << "Malformed schedule DMA pointer";
      DataType dtype = pointer->args[0].dtype();
      int bytes = Bytes(dtype);
      ICHECK(size->value <= 65535 / bytes && (size->value * bytes) % 32 == 0)
          << "Unsupported schedule DMA burst/alignment";
      std::string spelling = callee->value;
      spelling.erase(
          std::remove_if(spelling.begin(), spelling.end(), ::isspace),
          spelling.end());
      if (spelling.rfind("tl::ascend::", 0) == 0)
        spelling.erase(0, 12);
      ICHECK_EQ(spelling, name + "<" + (bytes == 2 ? "half" : "float") + "," +
                              std::to_string(size->value) + ">")
          << "Unsupported schedule DMA template/physical footprint";
      config = GetOperationConfig().at(name);
      begin = 1;
    } else {
      const auto *op = call->op.as<OpNode>();
      ICHECK(op &&
             (op->name == "tl.ascend_add" || op->name == "tl.ascend_fill"))
          << "Unsupported schedule operation/manual state: " << operation;
      ICHECK_EQ(call->args.size(), 4U) << "Unsupported schedule operation ABI";
      config = ResolveOperationConfig(call.get());
      count = Index(call->args[3]);
      if (op->name == "tl.ascend_fill") {
        PostOrderVisit(call->args[2], [](const ObjectRef &node) {
          ICHECK(!node.as<BufferLoadNode>() && !node.as<CallNode>())
              << "Unsupported memory-dependent/opaque fill scalar";
        });
      }
    }
    Task task;
    AscendResource resource = ResourceForCall(call.get(), &operation);
    ICHECK(resource == AscendResource::kVector)
        << "Unsupported schedule resource: " << operation;
    Array<ObjectRef> reads, writes;
    DataType operand_dtype;
    bool first = true;
    for (const auto &access : config.buffer_accesses) {
      ICHECK_LT(begin + access.first, call->args.size())
          << "Missing schedule buffer operand";
      const PrimExpr &operand = call->args[begin + access.first];
      const auto *ptr = operand.as<CallNode>();
      ICHECK(ptr && ptr->args.size() == 5)
          << "Malformed schedule buffer operand";
      DataType dtype = ptr->args[0].dtype();
      if (begin == 0) {
        ICHECK(StorageScopeForAccessPtr(operand) == "shared.ub")
            << "Managed schedule vector operands must be UB buffers";
      }
      if (first)
        operand_dtype = dtype;
      ICHECK(first || dtype == operand_dtype)
          << "Unsupported schedule mixed dtype";
      first = false;
      Access effect = Pointer(operand, access.second == "read" ? 1 : 2, count);
      groups_[group].exact = groups_[group].exact && effect.exact;
      if (access.second == "read") {
        task.reads.push_back(effect);
        reads.push_back(effect.AsRecord());
      } else {
        task.writes.push_back(effect);
        writes.push_back(effect.AsRecord());
      }
    }
    int id = tasks_.size();
    task.record = {{"id", Integer(id)},
                   {"source_order", Integer(id)},
                   {"group_id", Integer(group)},
                   {"loop_ids", loop_stack_},
                   {"operation", String(operation)},
                   {"statement", statement},
                   {"core", String("V")},
                   {"pipe", String(config.default_pipeline)},
                   {"guard", guard_},
                   {"reads", reads},
                   {"writes", writes},
                   {"state_effects", Array<ObjectRef>()}};
    tasks_.push_back(task);
    if (manual_stage_.defined())
      tasks_.back().record.Set("manual_stage", manual_stage_.value());
    groups_[group].members.push_back(Integer(id));
    if (!is_one(guard_))
      groups_[group].complex = true;
  }
  bool Overlap(const Access &a, const Access &b) {
    const Storage &left = storages_[a.storage];
    const Storage &right = storages_[b.storage];
    if (left.scope == right.scope && (!a.bounded || !b.bounded))
      return true;
    if (a.storage != b.storage) {
      if (left.scope != right.scope)
        return false;
      if (left.scope == "global")
        return !noalias_;
      if (!addresses_.count(left.data) && !addresses_.count(right.data))
        return false;
      // These are user placement hints before final MemoryPlanning; auto
      // alignment and physical reuse have not happened yet. They cannot prove
      // disjointness here, even if the raw hint values differ.
      return true;
    }
    return !analyzer_.CanProve(a.end <= b.start || b.end <= a.start);
  }
  void AddMemoryEdges(int src, int dst, const std::vector<Access> &left,
                      const std::vector<Access> &right, const char *kind,
                      Array<ObjectRef> *dependencies) {
    for (const Access &a : left) {
      for (const Access &b : right) {
        if (Overlap(a, b)) {
          dependencies->push_back(
              Record{{"src", Integer(src)},
                     {"dst", Integer(dst)},
                     {"kind", String(kind)},
                     {"storage_id", Integer(a.storage)},
                     {"other_storage_id", Integer(b.storage)}});
        }
      }
    }
  }

  PrimFunc function_;
  String platform_;
  bool noalias_{false};
  int core_{-1};
  int group_parent_{-1};
  int task_hint_depth_{0};
  Optional<Integer> stage_, manual_stage_;
  PrimExpr guard_;
  arith::Analyzer analyzer_;
  std::unordered_map<const VarNode *, int> storage_ids_;
  std::vector<Storage> storages_;
  std::vector<Task> tasks_;
  std::vector<Group> groups_;
  Array<ObjectRef> loop_records_;
  Array<ObjectRef> bindings_;
  Array<ObjectRef> launch_domains_;
  Array<Integer> loop_stack_;
  Map<Var, PrimExpr> addresses_;
  Map<Var, Range> proof_domains_;
};

} // namespace

PrimExpr PromoteAscendStorageIndex(PrimExpr expression,
                                   Map<tir::Var, Range> proof_domains) {
  DataType dtype = expression.dtype();
  ICHECK(dtype.is_scalar() && !dtype.is_bool() &&
         ((dtype.is_int() && dtype.bits() <= 64) ||
          (dtype.is_uint() && dtype.bits() < 64)))
      << "Unsupported storage pointer integer index";
  ScheduleIndexPromoter promoter(proof_domains);
  PrimExpr result = promoter(expression);
  ICHECK(result.dtype() == DataType::Int(64))
      << "Storage index must remain int64";
  return result;
}

Map<String, ObjectRef> AnalyzeAscendSchedule(tir::PrimFunc function,
                                             String platform) {
  return ScheduleGraphBuilder(function, platform).Build();
}

Map<String, ObjectRef> AnalyzeAscendScheduleEffects(tir::PrimFunc function,
                                                    String platform) {
  return ScheduleGraphBuilder(function, platform).Build(false);
}

TVM_REGISTER_GLOBAL("tl.transform.AnalyzeAscendSchedule")
    .set_body_typed(AnalyzeAscendSchedule);

} // namespace tl
} // namespace tvm
