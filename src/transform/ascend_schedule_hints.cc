// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <functional>
#include <limits>
#include <vector>

#include "common/ascend_constraint.h"
#include "common/ascend_schedule_hints.h"

namespace tvm {
namespace tl {
namespace {
using namespace tir;

bool IsHint(const String &key) {
  return key == kAscendTaskHint || key == kAscendStageHint ||
         key == kAscendPerCoreTaskHint;
}

void ValidateMarkerValue(const AttrStmtNode *hint) {
  const auto *value = hint->value.as<IntImmNode>();
  ICHECK(value && value->dtype == DataType::Int(32) && value->value == 1)
      << "Malformed Ascend schedule marker value; expected int32 1";
}

void ValidateNewScheduleControls(const PrimFunc &function) {
  bool constraint =
      tvm::transform::PassContext::Current()
          ->GetConfig<String>(kAscendPipelinePlanner, String("legacy"))
          .value() == "constraint";
  PostOrderVisit(function->body, [&](const ObjectRef &node) {
    const auto *loop = node.as<ForNode>();
    if (!loop)
      return;
    for (const char *key : {"enable_offset", kAscendPersistentGuard}) {
      auto annotation = loop->annotations.Get(key);
      if (!annotation.defined())
        continue;
      const auto *value = annotation.value().as<IntImmNode>();
      ICHECK(value && value->dtype == DataType::Bool())
          << "New scheduling controls require constant boolean values: " << key;
      ICHECK(!value->value || constraint)
          << key << " requires the constraint scheduling mode";
    }
  });
}

class HintSyntaxChecker final : public StmtExprVisitor {
public:
  HintSyntaxChecker()
      : constraint_(
            tvm::transform::PassContext::Current()
                ->GetConfig<String>(kAscendPipelinePlanner, String("legacy"))
                .value() == "constraint") {}

private:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == kAscendPerCoreTaskHint) {
      LOG(FATAL) << "PerCoreTask is unsupported";
    }
    if (op->attr_key == kAscendTaskHint) {
      ValidateAscendTaskHintPayload(op);
      ICHECK(!task_depth_ && !parallel_depth_)
          << "Nested/Parallel T.Task is unsupported";
      ++task_depth_;
      int previous = task_calls_;
      task_calls_ = 0;
      VisitStmt(op->body);
      ICHECK_EQ(task_calls_, 1) << "T.Task requires exactly one hardware call";
      task_calls_ = previous;
      --task_depth_;
      ++stage_tasks_;
      return;
    }
    if (op->attr_key == kAscendStageHint) {
      ValidateAscendStageHintPayload(op);
      ICHECK(!task_depth_ && !stage_depth_ && !parallel_depth_)
          << "T.Stage must wrap complete tasks, outside Task/Parallel/Stage";
      bool manual = constraint_ && offset_context_ && !manual_conflict_;
      ICHECK(!pipeline_conflict_ || manual)
          << "Stage(0) cannot mix with positive num_stages, "
             "legacy stage/order or enable_offset";
      const auto *stage = op->node.as<IntImmNode>();
      ICHECK(stage && (stage->value == 0 || manual))
          << "Nonzero Stage requires constraint mode and enable_offset=True";
      ++stage_depth_;
      int previous = stage_tasks_;
      stage_tasks_ = 0;
      VisitStmt(op->body);
      ICHECK_GT(stage_tasks_, 0)
          << "Stage(0) must wrap complete explicit T.Task regions";
      stage_tasks_ = previous;
      --stage_depth_;
      return;
    }
    ICHECK(!task_depth_ || op->attr_key == "resource_scope")
        << "Unsupported attribute inside T.Task: " << op->attr_key;
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    ICHECK(!task_depth_ && !stage_depth_)
        << "Loops inside Task/Stage are unsupported";
    bool previous = pipeline_conflict_;
    bool previous_offset = offset_context_;
    bool previous_manual = manual_conflict_;
    for (const char *key :
         {"tl_pipeline_order", "tl_pipeline_stage", "software_pipeline_order",
          "software_pipeline_stage"}) {
      pipeline_conflict_ = pipeline_conflict_ || op->annotations.count(key);
      manual_conflict_ = manual_conflict_ || op->annotations.count(key);
    }
    for (const char *key : {"num_stages", "enable_offset"}) {
      auto value = op->annotations.Get(key);
      if (value.defined()) {
        const auto *constant = value.value().as<IntImmNode>();
        pipeline_conflict_ =
            pipeline_conflict_ || !constant || constant->value != 0;
      }
    }
    auto offset = op->annotations.Get("enable_offset");
    if (offset.defined()) {
      const auto *enabled = offset.value().as<IntImmNode>();
      ICHECK(enabled && enabled->dtype == DataType::Bool())
          << "enable_offset requires a constant boolean";
      if (enabled->value) {
        ICHECK(constraint_) << "enable_offset=True requires constraint mode";
        offset_context_ = true;
      }
    }
    bool parallel =
        op->kind != ForKind::kSerial && op->kind != ForKind::kUnrolled;
    parallel_depth_ += parallel;
    VisitStmt(op->body);
    parallel_depth_ -= parallel;
    pipeline_conflict_ = previous;
    offset_context_ = previous_offset;
    manual_conflict_ = previous_manual;
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    ICHECK(!task_depth_ && !stage_depth_)
        << "Conditions inside Task/Stage are unsupported";
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const WhileNode *op) final {
    ICHECK(!task_depth_ && !stage_depth_)
        << "Loops inside Task/Stage are unsupported";
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const LetStmtNode *op) final {
    ICHECK(!task_depth_ && !stage_depth_)
        << "Scalar bindings inside Task/Stage are unsupported";
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    ICHECK(!task_depth_ && !stage_depth_)
        << "Scalar stores inside Task/Stage are unsupported";
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const EvaluateNode *op) final {
    if (is_zero(op->value))
      return;
    ICHECK(!stage_depth_ || task_depth_)
        << "Stage(0) must wrap complete explicit T.Task regions";
    if (task_depth_) {
      ICHECK(op->value.as<CallNode>()) << "T.Task requires one hardware call";
      ++task_calls_;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  int task_depth_{0}, stage_depth_{0}, parallel_depth_{0};
  int task_calls_{0}, stage_tasks_{0};
  bool pipeline_conflict_{false};
  bool offset_context_{false}, manual_conflict_{false};
  const bool constraint_;
};

class HintConsumer final : public StmtMutator {
public:
  explicit HintConsumer(PrimFunc function, String platform, bool constraint)
      : function_(function), platform_(platform), constraint_(constraint) {}

private:
  using Frame = std::function<Stmt(Stmt)>;
  template <typename Node, typename Ref>
  Stmt VisitFramed(const Node *op, Ref original) {
    frames_.push_back([original](Stmt body) -> Stmt {
      Ref copy = original;
      copy.CopyOnWrite()->body = body;
      return copy;
    });
    Stmt result = StmtMutator::VisitStmt_(op);
    frames_.pop_back();
    return result;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == kAscendStageHint) {
      Optional<Integer> saved = stage_;
      stage_ = Downcast<Integer>(op->node);
      Stmt body = VisitStmt(op->body);
      stage_ = saved;
      return body;
    }
    if (op->attr_key == kAscendTaskHint) {
      Stmt probe = op->body;
      for (auto frame = frames_.rbegin(); frame != frames_.rend(); ++frame)
        probe = (*frame)(probe);
      PrimFunc function = function_;
      function.CopyOnWrite()->body = probe;
      auto graph = AnalyzeAscendSchedule(function, platform_);
      auto tasks = Downcast<Array<ObjectRef>>(graph["tasks"]);
      ICHECK_EQ(tasks.size(), 1U)
          << "T.Task requires exactly one audited hardware call";
      if (constraint_ && stage_.defined()) {
        Map<String, ObjectRef> metadata{{"stage", stage_.value()}};
        return AttrStmt(metadata, kAscendPipelineTaskHint,
                        IntImm(DataType::Int(32), 1), op->body);
      }
      return op->body;
    }
    // Unrelated outer pragmas remain in the real IR, but are not task effects.
    if (op->attr_key != "resource_scope" && op->attr_key != "thread_extent")
      return StmtMutator::VisitStmt_(op);
    return VisitFramed(op, GetRef<AttrStmt>(op));
  }
  Stmt VisitStmt_(const ForNode *op) final {
    return VisitFramed(op, GetRef<For>(op));
  }
  Stmt VisitStmt_(const AllocateNode *op) final {
    return VisitFramed(op, GetRef<Allocate>(op));
  }
  Stmt VisitStmt_(const LetStmtNode *op) final {
    return VisitFramed(op, GetRef<LetStmt>(op));
  }

  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    BlockRealize original = GetRef<BlockRealize>(op);
    frames_.push_back([original](Stmt body) -> Stmt {
      Block block = original->block;
      block.CopyOnWrite()->body = body;
      // Block initialization outside the candidate is not a task effect.
      block.CopyOnWrite()->init = NullOpt;
      return BlockRealize(original->iter_values, original->predicate, block,
                          original->span);
    });
    Stmt result = StmtMutator::VisitStmt_(op);
    frames_.pop_back();
    return result;
  }
  Stmt VisitStmt_(const IfThenElseNode *op) final {
    PrimExpr condition = op->condition;
    frames_.push_back(
        [condition](Stmt body) -> Stmt { return IfThenElse(condition, body); });
    Stmt then_case = VisitStmt(op->then_case);
    frames_.pop_back();
    Optional<Stmt> else_case;
    if (op->else_case.defined()) {
      frames_.push_back([condition](Stmt body) -> Stmt {
        return IfThenElse(Not(condition), body);
      });
      else_case = VisitStmt(op->else_case.value());
      frames_.pop_back();
    }
    return IfThenElse(condition, then_case, else_case, op->span);
  }

  PrimFunc function_;
  String platform_;
  std::vector<Frame> frames_;
  Optional<Integer> stage_;
  const bool constraint_;
};

} // namespace

bool HasAscendScheduleHints(const tir::Stmt &body) {
  bool found = false;
  tir::PostOrderVisit(body, [&](const ObjectRef &object) {
    if (const auto *attribute = object.as<tir::AttrStmtNode>())
      found = found || IsHint(attribute->attr_key);
  });
  return found;
}

void ValidateAscendTaskHintPayload(const tir::AttrStmtNode *hint) {
  ValidateMarkerValue(hint);
  ICHECK(hint->node.as<runtime::MapNode>())
      << "Malformed T.Task metadata; expected an empty map";
  auto metadata = Downcast<Map<String, ObjectRef>>(hint->node);
  ICHECK(metadata.empty()) << "T.Task metadata/cost overrides are unsupported";
}

void ValidateAscendStageHintPayload(const tir::AttrStmtNode *hint) {
  ValidateMarkerValue(hint);
  const auto *stage = hint->node.as<tir::IntImmNode>();
  bool constraint =
      tvm::transform::PassContext::Current()
          ->GetConfig<String>(kAscendPipelinePlanner, String("legacy"))
          .value() == "constraint";
  ICHECK(stage && stage->dtype == DataType::Int(64) && stage->value >= 0 &&
         stage->value <= std::numeric_limits<int32_t>::max() &&
         (stage->value == 0 || constraint))
      << "Stage requires nonnegative int32-range literal; nonzero stages "
         "require constraint mode";
}

void ValidateAscendScheduleHintStructure(const tir::PrimFunc &function) {
  if (HasAscendScheduleHints(function->body))
    HintSyntaxChecker()(function->body);
}

tvm::transform::Pass AscendValidateScheduleHints() {
  auto pass = [](tir::PrimFunc function, IRModule,
                 tvm::transform::PassContext) {
    ValidateNewScheduleControls(function);
    if (!HasAscendScheduleHints(function->body))
      return function;
    ValidateAscendScheduleHintStructure(function);
    // Keep the backend gate observable if dead-code elimination drops markers.
    return WithAttr(std::move(function), kAscendScheduleHintsPending,
                    Bool(true));
  };
  return tir::transform::CreatePrimFuncPass(
      pass, 0, "tl.AscendValidateScheduleHints", {});
}

tvm::transform::Pass AscendLowerScheduleHints(Target target, String platform) {
  auto pass = [target, platform](tir::PrimFunc function, IRModule,
                                 tvm::transform::PassContext context) {
    bool has_hints = HasAscendScheduleHints(function->body);
    auto pending = function->GetAttr<Bool>(kAscendScheduleHintsPending);
    if (!has_hints && !pending.defined())
      return function;
    if (pending.defined()) {
      ICHECK(pending.value()->dtype == DataType::Bool() &&
             pending.value()->value)
          << "Malformed reserved Task/Stage pending fact";
    }
    String model = target->GetAttr<String>("model", String("")).value();
    ICHECK(platform == "A3" && (model == "ascendc" || model == "auto"))
        << "Task/Stage hints are supported only for A3 AscendC/auto";
    if (has_hints) {
      ValidateAscendScheduleHintStructure(function);
      bool constraint =
          context->GetConfig<String>(kAscendPipelinePlanner, String("legacy"))
              .value() == "constraint";
      Stmt body = HintConsumer(function, platform, constraint)(function->body);
      ICHECK(!HasAscendScheduleHints(body)) << "Unconsumed Task/Stage marker";
      function.CopyOnWrite()->body = body;
    }
    return WithoutAttr(std::move(function), kAscendScheduleHintsPending);
  };
  return tir::transform::CreatePrimFuncPass(pass, 0,
                                            "tl.AscendLowerScheduleHints", {});
}

TVM_REGISTER_GLOBAL("tl.transform.AscendValidateScheduleHints")
    .set_body_typed(AscendValidateScheduleHints);
TVM_REGISTER_GLOBAL("tl.transform.AscendLowerScheduleHints")
    .set_body_typed(AscendLowerScheduleHints);

} // namespace tl
} // namespace tvm
