// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#include <map>
#include <set>
#include <string>

#include <tvm/runtime/registry.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "../op/ascend.h"
#include "common/ascend_constraint.h"
#include "common/ascend_scheduled_sync.h"

namespace tvm {
namespace tl {
namespace {
using namespace tir;
using Record = AutoRecord;
int64_t IntegerValue(const ObjectRef &object) {
  const auto *value = object.as<IntImmNode>();
  ICHECK(value && value->dtype.is_int())
      << "Malformed scheduled synchronization integer";
  return value->value;
}
bool Enabled(Target target, String platform) {
  auto ctx = tvm::transform::PassContext::Current();
  if (ctx->GetConfig<String>(kAscendPipelinePlanner, String("legacy"))
          .value() != "auto")
    return false;
  ICHECK(platform == "A3" &&
         (target->GetAttr<String>("model").value_or("") == "ascendc" ||
          target->GetAttr<String>("model").value_or("") == "auto"))
      << "Scheduled synchronization requires A3 AscendC";
  ICHECK(
      ctx->GetConfig<String>("tl.ascend_sync_strategy", String("legacy"))
              .value() == "a3" &&
      ctx->GetConfig<Bool>("tl.ascend_auto_sync", Bool(false)).value()->value &&
      ctx->GetConfig<Bool>("tl.ascend_memory_planning", Bool(false))
          .value()
          ->value &&
      !ctx->GetConfig<Bool>("tl.ascend_auto_sync_vs", Bool(false))
           .value()
           ->value &&
      !ctx->GetConfig<Bool>("tl.ascend_auto_cross_core_sync", Bool(false))
           .value()
           ->value)
      << "Auto schedule requires sole new sync/memory ownership";
  return true;
}
Stmt All() {
  return Evaluate(
      Call(DataType::Handle(), ascend_auto_barrier(), {StringImm("PIPE_ALL")}));
}
Stmt Set(int id, const std::string &pipe) {
  return Evaluate(Call(DataType::Handle(), ascend_auto_set_cross_flag(),
                       {Integer(2), StringImm(pipe), Integer(id)}));
}
Stmt Wait(int id, const std::string &pipe) {
  return Evaluate(Call(DataType::Handle(), ascend_auto_wait_cross_flag(),
                       {Integer(id), StringImm(pipe)}));
}
struct Role {
  bool producer;
  std::string pipe;
};
class Emitter final : public StmtMutator {
public:
  explicit Emitter(Record channel) {
    for (const ObjectRef &item :
         Downcast<Array<ObjectRef>>(channel.at("roles"))) {
      Record role = Downcast<Record>(item);
      ICHECK(roles_
                 .emplace(IntegerValue(role.at("task")),
                          Role{bool(IntegerValue(role.at("producer"))),
                               Downcast<String>(role.at("pipe"))})
                 .second)
          << "Duplicate scheduled handoff role";
    }
  }

private:
  Stmt VisitStmt_(const AttrStmtNode *node) final {
    if (node->attr_key != kAscendAutoScheduleTaskId)
      return StmtMutator::VisitStmt_(node);
    int64_t id = IntegerValue(node->node);
    ICHECK(seen_.insert(id).second) << "Repeated static auto task marker";
    Array<Stmt> body{All()};
    auto role = roles_.find(id);
    if (role != roles_.end() && !role->second.producer)
      body.push_back(Wait(0, role->second.pipe));
    body.push_back(GetRef<Stmt>(node));
    body.push_back(All());
    if (role != roles_.end()) {
      if (role->second.producer) {
        body.push_back(Set(0, role->second.pipe));
        body.push_back(Wait(1, role->second.pipe));
      } else
        body.push_back(Set(1, role->second.pipe));
      body.push_back(All());
    }
    return SeqStmt(body);
  }
  std::map<int64_t, Role> roles_;
  std::set<int64_t> seen_;
};
} // namespace

tir::transform::Pass AscendInsertScheduledSync(Target target, String platform) {
  auto pass = [=](PrimFunc function, IRModule, tvm::transform::PassContext) {
    if (!Enabled(target, platform))
      return function;
    ICHECK(!function->GetAttr<ObjectRef>(kAscendAutoScheduleSyncPlan).defined())
        << "Scheduled synchronization already generated";
    Record proof = VerifyAscendAutoMapping(function);
    Record contract = Downcast<Record>(proof.at("contract"));
    PrimFunc result = function;
    result.CopyOnWrite()->body =
        Emitter(Downcast<Record>(contract.at("channel")))(function->body);
    return WithAttr(
        result, kAscendAutoScheduleSyncPlan,
        Record{{"schema_version", Integer(1)},
               {"local_policy", String("all_task_boundaries")},
               {"cross_policy", String("mode2_fixed_n1_ready_release")},
               {"ready_id", Integer(0)},
               {"release_id", Integer(1)}});
  };
  return tir::transform::CreatePrimFuncPass(pass, 0,
                                            "tl.AscendInsertScheduledSync", {});
}
TVM_REGISTER_GLOBAL("tl.transform.AscendInsertScheduledSync")
    .set_body_typed(AscendInsertScheduledSync);
} // namespace tl
} // namespace tvm
