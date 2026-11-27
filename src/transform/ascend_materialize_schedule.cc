// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <map>
#include <set>

#include "common/ascend_auto_schedule.h"
#include "common/ascend_lifecycle.h"
#include "common/ascend_storage_alias.h"

namespace tvm {
namespace tl {
namespace {
using namespace tir;
using R = AutoRecord;
Integer Int(int64_t x) { return Integer(IntImm(DataType::Int(64), x)); }
int64_t N(const ObjectRef &x) {
  return EvaluateAscendLifetimeInteger(Downcast<PrimExpr>(x), {});
}
Stmt Seq(Array<Stmt> body) {
  return body.empty()       ? Stmt(Evaluate(0))
         : body.size() == 1 ? body[0]
                            : Stmt(SeqStmt(body));
}
class AccessRewriter : public StmtExprMutator {
public:
  AccessRewriter(const std::map<const VarNode *, R> &owners, int64_t task)
      : owners_(owners), task_(task) {}

private:
  PrimExpr VisitExpr_(const CallNode *node) final {
    if (!node->op.same_as(builtin::tvm_access_ptr()))
      return StmtExprMutator::VisitExpr_(node);
    ICHECK(node->args.size() == 5 && node->args[1].as<VarNode>());
    auto it = owners_.find(node->args[1].as<VarNode>());
    if (it == owners_.end())
      return GetRef<Call>(node);
    R owner = it->second;
    int64_t slot = -1;
    for (const ObjectRef &item :
         Downcast<Array<ObjectRef>>(owner.at("slots"))) {
      R selected = Downcast<R>(item);
      if (N(selected.at("task")) == task_) {
        ICHECK(slot < 0);
        slot = N(selected.at("slot"));
      }
    }
    ICHECK(slot >= 0 && slot < N(owner.at("versions")))
        << "Missing auto emitted task slot";
    int64_t bytes = node->args[0].dtype().bytes();
    int64_t pitch = N(owner.at("pitch"));
    ICHECK(pitch % bytes == 0);
    int64_t old = EvaluateAscendLifetimeInteger(node->args[2], {});
    __int128 offset = __int128(old) + __int128(slot) * pitch / bytes;
    ICHECK(offset >= 0 && offset <= INT64_MAX)
        << "Auto typed version offset overflow";
    Array<PrimExpr> args = node->args;
    args.Set(1, Downcast<Var>(owner.at("physical_data")));
    args.Set(2, IntImm(DataType::Int(64), int64_t(offset)));
    return Call(node->dtype, node->op, args, node->span);
  }
  const std::map<const VarNode *, R> &owners_;
  int64_t task_;
};
PrimFunc Materialize(PrimFunc function, Target target, String platform) {
  CheckAscendAutoScheduleContext(target, platform);
  auto attr = function->GetAttr<R>(kAscendAutoSchedule);
  ICHECK(attr.defined()) << "Missing auto schedule";
  R schedule = attr.value();
  ICHECK(N(schedule.at("schema_version")) == 1 &&
         !N(schedule.at("materialized")))
      << "Repeated/stale auto materialization";
  PrimFunc original = Downcast<PrimFunc>(schedule.at("source_function"));
  R source = AnalyzeAscendAutoSource(original);
  auto tasks = Downcast<Array<ObjectRef>>(source.at("tasks"));
  R solution = Downcast<R>(schedule.at("solution"));
  auto execution = Downcast<Array<Integer>>(solution.at("order"));
  ICHECK_EQ(execution.size(), tasks.size());
  std::set<int64_t> covered;
  for (Integer id : execution)
    ICHECK(id.dtype().is_int() && id->value >= 0 &&
           id->value < int64_t(tasks.size()) &&
           covered.insert(id->value).second)
        << "Auto order must be a complete execution permutation";
  if (!tasks.empty()) {
    const auto *verify = runtime::Registry::Get("tl.ascend.verify_schedule");
    ICHECK(verify);
    R checked = (*verify)(schedule.at("problem"), solution);
    ICHECK(N(checked.at("valid")))
        << "Cannot materialize an invalid auto numeric plan";
  }
  auto remaps =
      function->GetAttr<Array<ObjectRef>>(kAscendAutoScheduleBufferRemap);
  ICHECK(remaps.defined());
  std::map<const VarNode *, R> owners;
  std::map<const VarNode *, Buffer> declarations;
  for (const ObjectRef &item :
       Downcast<Array<ObjectRef>>(source.at("storages"))) {
    R storage = Downcast<R>(item);
    declarations.emplace(Downcast<Var>(storage.at("data")).get(),
                         Downcast<Buffer>(storage.at("buffer")));
  }
  Array<Buffer> allocations;
  Array<ObjectRef> actual_remaps;
  Map<Var, Var> replacements;
  Map<Var, Array<PrimExpr>> shapes;
  for (const ObjectRef &item : remaps.value()) {
    R owner = Downcast<R>(item);
    Var logical = Downcast<Var>(owner.at("logical_data"));
    if (Downcast<Array<ObjectRef>>(owner.at("slots")).empty())
      continue;
    ICHECK(declarations.count(logical.get()) && !owners.count(logical.get()));
    Buffer buffer = declarations.at(logical.get());
    int64_t versions = N(owner.at("versions")), pitch = N(owner.at("pitch"));
    ICHECK(versions >= 1 && versions <= 2 && pitch > 0 && pitch % 32 == 0 &&
           N(owner.at("total")) == versions * pitch &&
           versions * pitch <= AscendAutoMemoryBudget(buffer.scope()));
    int64_t physical_elements = versions * pitch / buffer->dtype.bytes();
    ICHECK(physical_elements * buffer->dtype.bytes() == versions * pitch);
    Var physical(logical->name_hint + "_auto_schedule",
                 logical->type_annotation);
    Buffer revised = buffer;
    auto *node = revised.CopyOnWrite();
    node->data = physical;
    node->name = buffer->name + "_auto_schedule";
    if (buffer.scope() == "shared.l1" && buffer->shape.size() == 1) {
      ICHECK(buffer->strides.empty() &&
             EvaluateAscendLifetimeInteger(buffer->shape[0], {}) *
                     buffer->dtype.bytes() ==
                 pitch &&
             (pitch / buffer->dtype.bytes()) % 256 == 0)
          << "Auto flat L1 allocation must contain complete C0 fractals";
      node->shape = {Int(physical_elements)};
    } else if (buffer.scope() == "shared.l1" ||
               buffer.scope() == "wmma.accumulator") {
      ICHECK(buffer->shape.size() == 2 && buffer->strides.empty())
          << "Auto Cube allocation requires a dense two-dimensional tile";
      int64_t rows = EvaluateAscendLifetimeInteger(buffer->shape[0], {});
      int64_t cols = EvaluateAscendLifetimeInteger(buffer->shape[1], {});
      ICHECK(rows % 16 == 0 && cols % 16 == 0 &&
             rows * cols * buffer->dtype.bytes() == pitch)
          << "Auto Cube version pitch requires a complete aligned tile";
      node->shape = {Int(versions * rows), Int(cols)};
    } else
      node->shape = {Int(physical_elements)};
    node->strides = {};
    node->elem_offset = Integer(0);
    node->axis_separators = {};
    owner.Set("physical_data", physical);
    owners.emplace(logical.get(), owner);
    replacements.Set(logical, physical);
    shapes.Set(physical, node->shape);
    allocations.push_back(revised);
    actual_remaps.push_back(owner);
  }
  std::vector<AttrStmt> axes;
  Var cid{ObjectPtr<Object>()}, vid{ObjectPtr<Object>()};
  PostOrderVisit(original->body, [&](const ObjectRef &object) {
    if (const auto *n = object.as<AttrStmtNode>();
        n && n->attr_key == "thread_extent") {
      axes.push_back(GetRef<AttrStmt>(n));
      IterVar axis = Downcast<IterVar>(n->node);
      if (axis->thread_tag == "blockIdx.x")
        cid = axis->var;
      else
        vid = axis->var;
    }
  });
  Array<Stmt> sides;
  for (int64_t group = 0; group < N(source.at("groups")); ++group)
    for (int64_t actor = 0; actor < 3; ++actor) {
      Array<Stmt> body;
      for (Integer order : execution) {
        R task = Downcast<R>(tasks[order->value]);
        if (N(task.at("group")) != group || N(task.at("actor")) != actor)
          continue;
        Stmt call = AccessRewriter(owners, order->value)(
            Downcast<Stmt>(task.at("statement")));
        body.push_back(AttrStmt(IntImm(DataType::Int(64), order->value),
                                kAscendAutoScheduleTaskId, Integer(1), call));
      }
      if (body.empty())
        continue;
      Stmt side = Seq(body);
      if (actor > 0 && vid.defined())
        side = IfThenElse(vid == make_const(vid.dtype(), actor - 1), side);
      side = AttrStmt(Integer(0), "resource_scope", Integer(actor == 0 ? 0 : 1),
                      side);
      if (cid.defined())
        side = IfThenElse(cid == make_const(cid.dtype(), group), side);
      sides.push_back(side);
    }
  Stmt body = Seq(sides);
  if (!allocations.empty()) {
    Block root({}, {}, {}, "ascend_auto_schedule_root", body, NullOpt,
               allocations, {});
    body = BlockRealize({}, Bool(true), root);
  }
  // Post-order collected inner axes first, so wrapping in that same order
  // restores the original nesting without duplicating any launch domain.
  for (const AttrStmt &axis : axes)
    body = AttrStmt(axis->node, axis->attr_key, axis->value, body);
  function = RemapAscendStorageMetadata(function, replacements);
  function.CopyOnWrite()->body = body;
  for (const char *key :
       {"initial_buffer_shapes", "buffer_shapess", "logic_buffer_shapes"}) {
    Map<Var, Array<PrimExpr>> merged;
    if (auto old = function->GetAttr<Map<Var, Array<PrimExpr>>>(key))
      merged = old.value();
    for (const auto &entry : shapes)
      merged.Set(entry.first, entry.second);
    function = WithAttr(function, key, merged);
  }
  schedule.Set("materialized", Bool(true));
  function = WithAttr(function, kAscendAutoScheduleBufferRemap, actual_remaps);
  function = WithAttr(function, kAscendAutoSchedule, schedule);
  return function;
}
} // namespace
tir::transform::Pass AscendMaterializeSchedule(Target target, String platform) {
  auto pass = [=](tir::PrimFunc f, IRModule, tvm::transform::PassContext) {
    return Materialize(f, target, platform);
  };
  return tir::transform::CreatePrimFuncPass(pass, 0,
                                            "tl.AscendMaterializeSchedule", {});
}
TVM_REGISTER_GLOBAL("tl.transform.AscendMaterializeSchedule")
    .set_body_typed(AscendMaterializeSchedule);
} // namespace tl
} // namespace tvm
