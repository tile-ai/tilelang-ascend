// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

// Legacy pipeline admission and extent partitioning. No upstream scheduling
// solver or alternative dependency/flag algorithm is implemented here.

#include "legacy_pipeline_partition.h"

#include <tvm/node/structural_equal.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <limits>
#include <set>
#include <unordered_set>
#include <utility>

#include "../common/pipeline_ring_provenance.h"
#include "tir/transforms/ir_utils.h"
#include "upstream/ir_structure.h"

namespace tvm::tl::ascend_sync {

using namespace tir;

namespace {

constexpr const char *kError = "Dynamic physical pipeline partition: ";

void RequirePure(const PrimExpr &value) {
  ICHECK(value.defined() && SideEffect(value) <= CallEffectKind::kPure)
      << kError << "scope/extent expressions must be pure";
}

int Integer(const PrimExpr &value) {
  const auto *constant = value.as<IntImmNode>();
  ICHECK(constant != nullptr && constant->value >= 0 &&
         constant->value <= std::numeric_limits<int>::max())
      << kError << "expected a static nonnegative scheduling field";
  return static_cast<int>(constant->value);
}

Var Owner(const ObjectRef &node) {
  const auto ids = Downcast<Array<Var>>(node);
  ICHECK_EQ(ids.size(), 1U) << kError << "invalid owner identity";
  return ids[0];
}

// An empty executable body may legitimately have lost dead allocations/maps.
// It must still carry internally consistent producer facts. In particular,
// emptiness never repairs a bad schema or replaces the recorded true ring B.
void ValidateBasicCertificates(const PrimFunc &function) {
  const auto rings =
      function->GetAttr<PhysicalRingProvenance>(kPhysicalRingProvenance)
          .value_or(PhysicalRingProvenance{});
  for (const auto &[logical_id, fields] : rings) {
    ValidatePhysicalRingRecord(fields);
    const int versions = Integer(fields[kPhysicalRingVersions]);
    ICHECK(versions >= 2 && versions <= 8)
        << kError << "ring version count must fit the eight-slot capacity";
    ICHECK_EQ(Integer(fields[kPhysicalRingOriginalMin]), 0)
        << kError << "ring provenance requires original minimum zero";
    auto bits = [&](size_t index) {
      const auto *value = fields[index].as<IntImmNode>();
      ICHECK(value != nullptr && value->value >= 0)
          << kError << "ring provenance bit fields must be static nonnegative";
      return value->value;
    };
    const int64_t base = bits(kPhysicalRingBaseBits);
    const int64_t stride = bits(kPhysicalRingStrideBits);
    const int64_t element = bits(kPhysicalRingElementBits);
    ICHECK(stride > 0 && element > 0 && base % 256 == 0 && stride % 256 == 0 &&
           element % 8 == 0 && base % element == 0 && stride % element == 0)
        << kError << "invalid physical ring provenance bit alignment";
  }
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> matched;
  for (const char *name : {"buffer_versions", "tl.buffer_versions_map"}) {
    const auto versions = function->GetAttr<Map<Var, PrimExpr>>(name);
    if (!versions.defined())
      continue;
    for (const auto &[storage, value] : versions.value()) {
      const int count = Integer(value);
      ICHECK_GT(count, 0) << kError << "version counts must be positive";
      const auto record = rings.Get(storage);
      if (count > 1 || record.defined()) {
        ICHECK(record.defined() &&
               Integer(record.value()[kPhysicalRingVersions]) == count)
            << kError << "buffer version table disagrees with ring provenance";
        matched.insert(storage);
      }
    }
  }
  ICHECK_EQ(matched.size(), rings.size())
      << kError << "ring provenance requires a matching true version table";

  const auto owners =
      function->GetAttr<PhysicalPipelineOwners>(kPhysicalPipelineOwners)
          .value_or(PhysicalPipelineOwners{});
  ICHECK(owners.empty() || !rings.empty())
      << kError << "pipeline owner metadata requires ring provenance";
  for (const auto &[owner, fields] : owners) {
    ICHECK_EQ(fields.size(), 4U) << kError << "invalid owner record";
    ICHECK_EQ(Integer(fields[0]), 1) << kError << "unsupported owner schema";
    const int lead = Integer(fields[1]);
    ICHECK(lead > 0 && lead <= 8 && is_zero(fields[2]) &&
           Integer(fields[3]) > 0)
        << kError << "invalid pipeline owner scheduling fields";
  }
  PostOrderVisit(function->body, [&](const ObjectRef &node) {
    const auto *attribute = node.as<AttrStmtNode>();
    if (attribute == nullptr)
      return;
    const String &key = attribute->attr_key;
    if (key == kPhysicalRingClock) {
      ICHECK(rings.count(PhysicalRingClockId(attribute->node)))
          << kError << "clock identity is absent from ring provenance";
    } else if (key == kPhysicalPipelineExtent ||
               key == kPhysicalPipelineSegment) {
      ICHECK(owners.count(Owner(attribute->node)))
          << kError << "scope identity is absent from owner metadata";
      if (key == kPhysicalPipelineSegment)
        ICHECK_LE(Integer(attribute->value), kPipelineEpilogue)
            << kError << "invalid segment metadata";
    } else if (key == kPhysicalPipelineStage) {
      const auto fields = Downcast<Array<PrimExpr>>(attribute->node);
      ICHECK_EQ(fields.size(), 5U) << kError << "invalid stage metadata";
      const auto owner = owners.Get(Downcast<Var>(fields[0]));
      ICHECK(owner.defined())
          << kError << "stage identity is absent from owner metadata";
      ICHECK(Integer(fields[1]) <= kPipelineEpilogue &&
             Integer(fields[2]) <= Integer(owner.value()[1]) &&
             Integer(fields[3]) < Integer(owner.value()[3]) &&
             Integer(fields[4]) <= 1)
          << kError << "invalid stage scheduling metadata";
    }
  });
}

bool NoWork(const Stmt &statement) {
  if (const auto *evaluate = statement.as<EvaluateNode>())
    return is_zero(evaluate->value);
  if (const auto *sequence = statement.as<SeqStmtNode>())
    return std::all_of(sequence->seq.begin(), sequence->seq.end(), NoWork);
  if (const auto *binding = statement.as<LetStmtNode>())
    return SideEffect(binding->value) <= CallEffectKind::kPure &&
           NoWork(binding->body);
  if (const auto *attribute = statement.as<AttrStmtNode>()) {
    if (SideEffect(attribute->value) > CallEffectKind::kPure)
      return false;
    if (const auto *expression = attribute->node.as<PrimExprNode>())
      if (SideEffect(GetRef<PrimExpr>(expression)) > CallEffectKind::kPure)
        return false;
    return NoWork(attribute->body);
  }
  if (const auto *allocate = statement.as<AllocateNode>()) {
    if (SideEffect(allocate->condition) > CallEffectKind::kPure)
      return false;
    for (const PrimExpr &extent : allocate->extents)
      if (SideEffect(extent) > CallEffectKind::kPure)
        return false;
    return NoWork(allocate->body);
  }
  if (const auto *decl = statement.as<DeclBufferNode>()) {
    for (const PrimExpr &extent : decl->buffer->shape)
      if (SideEffect(extent) > CallEffectKind::kPure)
        return false;
    for (const PrimExpr &stride : decl->buffer->strides)
      if (SideEffect(stride) > CallEffectKind::kPure)
        return false;
    return SideEffect(decl->buffer->elem_offset) <= CallEffectKind::kPure &&
           NoWork(decl->body);
  }
  // Do not infer emptiness from an arbitrary If or dynamic/negative For.
  return false;
}

class EnvelopeFinder : public StmtExprVisitor {
public:
  explicit EnvelopeFinder(Var expected) : expected_(std::move(expected)) {}
  const AttrStmtNode *envelope{nullptr};
  ConstrSet context;

private:
  void VisitStmt(const Stmt &statement) final {
    ICHECK(statement.as<SeqStmtNode>() || statement.as<AttrStmtNode>() ||
           statement.as<LetStmtNode>() || statement.as<AllocateNode>() ||
           statement.as<DeclBufferNode>() || statement.as<EvaluateNode>() ||
           statement.as<BufferStoreNode>() || statement.as<ForNode>() ||
           statement.as<IfThenElseNode>() || statement.as<WhileNode>() ||
           statement.as<AssertStmtNode>())
        << kError << "unsupported wrapper outside the pipeline owner";
    StmtExprVisitor::VisitStmt(statement);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    RequirePure(op->value);
    if (op->node.as<PrimExprNode>())
      RequirePure(Downcast<PrimExpr>(op->node));
    if (op->attr_key == kPhysicalPipelineExtent) {
      ICHECK(envelope == nullptr && Owner(op->node).same_as(expected_))
          << kError << "requires exactly one top-level owner envelope";
      envelope = op;
      return;
    }
    ICHECK(!IsPhysicalPipelineMetadata(op->attr_key))
        << kError << "orphan scheduling metadata outside its owner";
    if (op->attr_key == "resource_scope")
      ICHECK_EQ(Integer(op->value), 1) << kError << "requires Vector-only work";
    const ConstrSet old = context;
    if (op->attr_key == "thread_extent" || op->attr_key == "virtual_thread") {
      const IterVar iv = Downcast<IterVar>(op->node);
      context.AddConstr(iv->var, Range::FromMinExtent(
                                     make_zero(op->value.dtype()), op->value));
    } else if ((op->attr_key == "tl.assume" ||
                op->attr_key == "tilelang_assume") &&
               op->node.as<PrimExprNode>()) {
      context.AddConstr(Downcast<PrimExpr>(op->node), true);
    }
    VisitStmt(op->body);
    if (envelope == nullptr)
      context = old;
  }

  void VisitStmt_(const LetStmtNode *op) final {
    RequirePure(op->value);
    const ConstrSet old = context;
    context.AddConstr(op->var, op->value);
    VisitStmt(op->body);
    if (envelope == nullptr)
      context = old;
  }

  void VisitStmt_(const AllocateNode *op) final {
    RequirePure(op->condition);
    for (const PrimExpr &extent : op->extents)
      RequirePure(extent);
    VisitStmt(op->body);
  }
  void VisitStmt_(const DeclBufferNode *op) final {
    for (const PrimExpr &extent : op->buffer->shape)
      RequirePure(extent);
    for (const PrimExpr &stride : op->buffer->strides)
      RequirePure(stride);
    RequirePure(op->buffer->elem_offset);
    VisitStmt(op->body);
  }
  void VisitStmt_(const EvaluateNode *op) final {
    ICHECK(is_zero(op->value)) << kError << "extra computation outside owner";
  }
  void VisitStmt_(const BufferStoreNode *) final {
    LOG(FATAL) << kError << "extra store outside owner";
  }
  void VisitStmt_(const ForNode *) final {
    LOG(FATAL) << kError << "an outer semantic For is unsupported";
  }
  void VisitStmt_(const IfThenElseNode *) final {
    LOG(FATAL) << kError << "an outer semantic If is unsupported";
  }
  void VisitStmt_(const WhileNode *) final {
    LOG(FATAL) << kError << "an outer semantic While is unsupported";
  }
  void VisitStmt_(const AssertStmtNode *) final {
    LOG(FATAL) << kError << "an outer semantic Assert is unsupported";
  }
  Var expected_;
};

class EnvelopeRewriter : public StmtExprMutator {
public:
  EnvelopeRewriter(Var owner, Stmt body)
      : owner_(std::move(owner)), body_(std::move(body)) {}

private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == kPhysicalPipelineExtent &&
        Owner(op->node).same_as(owner_))
      return AttrStmt(op->node, op->attr_key, op->value, body_, op->span);
    return StmtExprMutator::VisitStmt_(op);
  }
  Var owner_;
  Stmt body_;
};

class CaseRewriter {
public:
  CaseRewriter(Var owner, Var ring, PrimExpr extent, int lead, int orders,
               LegacyPipelineCase *result)
      : owner_(std::move(owner)), ring_(std::move(ring)),
        extent_(std::move(extent)), lead_(lead), orders_(orders),
        result_(result) {}

  Stmt Rewrite(const Stmt &statement, ConstrSet context, int segment = -1) {
    if (const auto *sequence = statement.as<SeqStmtNode>()) {
      Array<Stmt> children;
      for (const Stmt &child : sequence->seq)
        children.push_back(Rewrite(child, context, segment));
      return SeqStmt(children);
    }
    if (const auto *attribute = statement.as<AttrStmtNode>()) {
      RequirePure(attribute->value);
      if (attribute->attr_key == kPhysicalPipelineSegment) {
        ICHECK(Owner(attribute->node).same_as(owner_))
            << kError << "segment owner mismatch";
        const int role = Integer(attribute->value);
        ICHECK(role <= kPipelineEpilogue && segment == -1)
            << kError << "invalid or nested segment";
        ICHECK(segments_.insert(role).second) << kError << "duplicate segment";
        return AttrStmt(attribute->node, attribute->attr_key, attribute->value,
                        Rewrite(attribute->body, context, role),
                        attribute->span);
      }
      if (attribute->attr_key == kPhysicalPipelineStage)
        return RewriteStage(attribute, context, segment);
      ICHECK(attribute->attr_key != kPhysicalPipelineExtent)
          << kError << "nested pipeline owners are unsupported";
      if (attribute->attr_key == "resource_scope")
        ICHECK_EQ(Integer(attribute->value), 1)
            << kError << "requires Vector-only work";
      ICHECK(std::string(attribute->attr_key).rfind("async_", 0) != 0)
          << kError << "async stage protocols require a separate contract";
      return AttrStmt(attribute->node, attribute->attr_key, attribute->value,
                      Rewrite(attribute->body, context, segment),
                      attribute->span);
    }
    if (const auto *loop = statement.as<ForNode>()) {
      RequirePure(loop->min);
      RequirePure(loop->extent);
      ICHECK(segment == kPipelineSteady && loop->kind == ForKind::kSerial &&
             is_zero(loop->min))
          << kError << "requires one zero-based serial steady For";
      ICHECK(!saw_steady_loop_)
          << kError << "multiple steady loops are unsupported";
      saw_steady_loop_ = true;
      const PrimExpr expected = extent_ - make_const(extent_.dtype(), lead_);
      ICHECK(context.CanProve(loop->extent == expected))
          << kError << "steady extent disagrees with injector lead";
      if (context.CanProve(loop->extent <= make_zero(loop->extent.dtype())))
        return Evaluate(0);
      ICHECK(context.CanProve(loop->extent >= make_zero(loop->extent.dtype())))
          << kError << "steady extent is not provably nonnegative";
      result_->steady_loop_var = loop->loop_var;
      result_->steady_extent = expected;
      context.AddConstr(loop->loop_var,
                        Range::FromMinExtent(loop->min, loop->extent));
      context.AddConstr(loop->extent > make_zero(loop->extent.dtype()));
      For rewritten = GetRef<For>(loop);
      rewritten.CopyOnWrite()->body = Rewrite(loop->body, context, segment);
      return rewritten;
    }
    if (const auto *binding = statement.as<LetStmtNode>()) {
      RequirePure(binding->value);
      context.AddConstr(binding->var, binding->value);
      return LetStmt(binding->var, binding->value,
                     Rewrite(binding->body, context, segment), binding->span);
    }
    ICHECK(NoWork(statement))
        << kError << "unmarked computation or user control in pipeline segment";
    return statement;
  }

  void Validate(bool long_case) const {
    ICHECK_EQ(segments_.size(), 3U)
        << kError << "missing pipeline segment metadata";
    ICHECK(saw_steady_loop_) << kError << "missing materialized steady loop";
    if (long_case)
      ICHECK_EQ(steady_orders_, orders_)
          << kError << "missing or reordered steady stages";
  }

private:
  Stmt Payload(const Stmt &statement, ConstrSet context) {
    if (const auto *sequence = statement.as<SeqStmtNode>()) {
      Array<Stmt> children;
      for (const Stmt &child : sequence->seq)
        children.push_back(Payload(child, context));
      return SeqStmt(children);
    }
    if (const auto *attribute = statement.as<AttrStmtNode>()) {
      RequirePure(attribute->value);
      ICHECK(!IsPhysicalPipelineMetadata(attribute->attr_key) ||
             attribute->attr_key == kPhysicalRingClock)
          << kError << "unexpected scheduling metadata inside stage payload";
      if (attribute->attr_key == kPhysicalRingClock)
        ValidateRingStageClock(attribute, context);
      if (attribute->attr_key == "resource_scope")
        ICHECK_EQ(Integer(attribute->value), 1)
            << kError << "requires Vector-only work";
      ICHECK(std::string(attribute->attr_key).rfind("async_", 0) != 0)
          << kError << "async stage protocols require a separate contract";
      return AttrStmt(attribute->node, attribute->attr_key, attribute->value,
                      Payload(attribute->body, context), attribute->span);
    }
    if (const auto *binding = statement.as<LetStmtNode>()) {
      RequirePure(binding->value);
      context.AddConstr(binding->var, binding->value);
      return LetStmt(binding->var, binding->value,
                     Payload(binding->body, context), binding->span);
    }
    ICHECK(statement.as<EvaluateNode>() || statement.as<BufferStoreNode>())
        << kError
        << "user If/For or unsupported control inside an injector stage";
    return statement; // Actual operation/effect validation remains
                      // importer-owned.
  }

  Stmt Bounds(const Stmt &statement, const PrimExpr &canonical,
              ConstrSet context, bool checked) {
    if (const auto *branch = statement.as<IfThenElseNode>()) {
      // Mathematical equivalence cannot erase a mutable read or opaque call
      // that the original runtime predicate would evaluate.
      RequirePure(branch->condition);
      ICHECK(checked && !branch->else_case.defined() &&
             GuardsEquivalent(branch->condition, canonical, context))
          << kError
          << "injector bounds guard is not equivalent to its logical clock "
             "range";
      if (context.CanProve(!canonical))
        return Evaluate(0);
      ICHECK(context.CanProve(canonical))
          << kError << "extent case did not resolve its bounds guard";
      context.AddConstr(branch->condition);
      return Payload(branch->then_case, context);
    }
    if (const auto *attribute = statement.as<AttrStmtNode>()) {
      RequirePure(attribute->value);
      ICHECK(!IsPhysicalPipelineMetadata(attribute->attr_key) ||
             attribute->attr_key == kPhysicalRingClock)
          << kError << "unexpected metadata around a stage bounds guard";
      if (attribute->attr_key == kPhysicalRingClock)
        ValidateRingStageClock(attribute, context);
      if (attribute->attr_key == "resource_scope")
        ICHECK_EQ(Integer(attribute->value), 1)
            << kError << "requires Vector-only work";
      ICHECK(std::string(attribute->attr_key).rfind("async_", 0) != 0)
          << kError << "async stage protocols require a separate contract";
      return AttrStmt(attribute->node, attribute->attr_key, attribute->value,
                      Bounds(attribute->body, canonical, context, checked),
                      attribute->span);
    }
    if (const auto *binding = statement.as<LetStmtNode>()) {
      RequirePure(binding->value);
      context.AddConstr(binding->var, binding->value);
      return LetStmt(binding->var, binding->value,
                     Bounds(binding->body, canonical, context, checked),
                     binding->span);
    }
    ICHECK(context.CanProve(canonical))
        << kError << "missing injector guard for a possibly inactive stage";
    return Payload(statement, context);
  }

  Stmt RewriteStage(const AttrStmtNode *attribute, ConstrSet context,
                    int segment) {
    const auto fields = Downcast<Array<PrimExpr>>(attribute->node);
    ICHECK_EQ(fields.size(), 5U) << kError << "invalid stage certificate";
    ICHECK(fields[0].same_as(owner_) && Integer(fields[1]) == segment)
        << kError << "stage/segment owner mismatch";
    const int stage = Integer(fields[2]);
    const int order = Integer(fields[3]);
    ICHECK_LE(Integer(fields[4]), 1) << kError << "invalid bounds-check field";
    const bool checked = Integer(fields[4]) == 1;
    ICHECK(stage <= lead_ && order < orders_ &&
           checked == (segment != kPipelineSteady))
        << kError << "invalid stage/order/bounds-check certificate";
    const PrimExpr clock = cast(DataType::Int(64), attribute->value);
    const PrimExpr extent = cast(DataType::Int(64), extent_);
    const PrimExpr physical = clock + make_const(DataType::Int(64), stage);
    if (segment == kPipelineSteady) {
      ICHECK(result_->steady_loop_var.defined() &&
             context.CanProve(
                 physical == cast(DataType::Int(64), result_->steady_loop_var) +
                                 make_const(DataType::Int(64), lead_)))
          << kError << "steady stage clock disagrees with its real loop";
      ICHECK_EQ(order, steady_orders_++)
          << kError << "steady stage order changed";
    } else {
      ICHECK(segment == kPipelinePrologue || segment == kPipelineEpilogue)
          << kError << "stage has no segment";
      const PrimExpr offset =
          segment == kPipelinePrologue ? physical : physical - extent;
      ICHECK(context.CanProve(offset >= make_zero(DataType::Int(64)) &&
                              offset < make_const(DataType::Int(64), lead_)))
          << kError << "stage clock is outside its physical segment";
    }
    const PrimExpr canonical =
        clock >= make_zero(DataType::Int(64)) && clock < extent;
    const PrimExpr old_clock = frame_clock_;
    frame_clock_ = attribute->value;
    Stmt payload = Bounds(attribute->body, canonical, context, checked);
    frame_clock_ = old_clock;
    return AttrStmt(attribute->node, attribute->attr_key, attribute->value,
                    std::move(payload), attribute->span);
  }

  void ValidateRingStageClock(const AttrStmtNode *attribute,
                              const ConstrSet &context) const {
    ICHECK(PhysicalRingClockId(attribute->node).same_as(ring_) &&
           frame_clock_.defined() &&
           context.CanProve(cast(DataType::Int(64), attribute->value) ==
                            cast(DataType::Int(64), frame_clock_)))
        << kError << "ring clock disagrees with its injector stage clock";
  }

  Var owner_;
  Var ring_;
  PrimExpr frame_clock_;
  PrimExpr extent_;
  int lead_;
  int orders_;
  LegacyPipelineCase *result_;
  std::set<int> segments_;
  bool saw_steady_loop_{false};
  int steady_orders_{0};
};

} // namespace

bool LegacyPipelineCase::AllowsSteadyLoop(const For &loop,
                                          const ConstrSet &context) const {
  return steady_loop_var.defined() && loop->loop_var.same_as(steady_loop_var) &&
         is_zero(loop->min) &&
         context.CanProve(loop->extent == steady_extent) &&
         context.CanProve(loop->extent >= make_zero(loop->extent.dtype()));
}

bool IsLegacyPipelineNoOp(const PrimFunc &function) {
  if (!NoWork(function->body))
    return false;
  ValidateBasicCertificates(function);
  return true;
}

std::optional<LegacyPipelinePartition>
PartitionLegacyDynamicPipeline(const PrimFunc &function) {
  const auto owners =
      function->GetAttr<PhysicalPipelineOwners>(kPhysicalPipelineOwners);
  if (!owners.defined() || owners.value().empty())
    return std::nullopt;
  // Multiple static owners retain their previous admission path. A dynamic
  // family is intentionally restricted to exactly one top-level owner.
  bool dynamic = false;
  PostOrderVisit(function->body, [&](const ObjectRef &node) {
    if (const auto *attr = node.as<AttrStmtNode>())
      if (attr->attr_key == kPhysicalPipelineExtent &&
          !attr->value.as<IntImmNode>())
        dynamic = true;
  });
  if (!dynamic)
    return std::nullopt;
  ValidateBasicCertificates(function);
  ICHECK_EQ(owners.value().size(), 1U)
      << kError << "requires exactly one SIP owner";
  const auto &entry = *owners.value().begin();
  const Var owner = entry.first;
  const auto fields = entry.second;
  ICHECK_EQ(fields.size(), 4U) << kError << "invalid owner record";
  ICHECK_EQ(Integer(fields[0]), 1) << kError << "unsupported owner schema";
  const int lead = Integer(fields[1]);
  ICHECK(lead > 0 && lead <= 8 && is_zero(fields[2]) && Integer(fields[3]) > 0)
      << kError << "requires a bounded static lead and original minimum zero";
  const auto rings =
      function->GetAttr<PhysicalRingProvenance>(kPhysicalRingProvenance);
  ICHECK(rings.defined() && rings.value().size() == 1)
      << kError
      << "the first dynamic contract requires exactly one physical ring";
  const auto ring_fields = (*rings.value().begin()).second;
  ValidatePhysicalRingRecord(ring_fields);
  ICHECK_EQ(Integer(ring_fields[kPhysicalRingVersions]), lead)
      << kError
      << "the audited ring period must match the independently published lead";
  EnvelopeFinder finder(owner);
  finder(function->body);
  ICHECK(finder.envelope != nullptr)
      << kError << "missing owner extent envelope";
  const PrimExpr extent = finder.envelope->value;
  ICHECK(extent.dtype().is_scalar() && extent.dtype().is_int() &&
         extent.dtype().bits() <= 64)
      << kError << "requires a pure signed scalar extent";
  LegacyPipelinePartition plan{function, owner, extent, lead, {}};
  for (int index = 0; index <= lead; ++index) {
    LegacyPipelineCase result;
    const PrimExpr threshold = make_const(extent.dtype(), index);
    result.condition =
        index == lead ? extent >= threshold : extent == threshold;
    ConstrSet context = finder.context;
    context.AddConstr(result.condition);
    CaseRewriter rewriter(owner, (*rings.value().begin()).first, extent, lead,
                          Integer(fields[3]), &result);
    Stmt body = rewriter.Rewrite(finder.envelope->body, context);
    rewriter.Validate(index == lead);
    result.zero_work = NoWork(body);
    body = AttrStmt(result.condition, "tl.assume", Integer(1), body);
    result.function = function;
    result.function.CopyOnWrite()->body =
        EnvelopeRewriter(owner, body)(function->body);
    plan.cases.push_back(std::move(result));
  }
  return plan;
}

PrimFunc MergeLegacyPipelineCases(const LegacyPipelinePartition &partition,
                                  const std::vector<PrimFunc> &synchronized) {
  ICHECK_EQ(synchronized.size(), partition.cases.size());
  Array<Stmt> branches;
  for (size_t i = 0; i < synchronized.size(); ++i) {
    EnvelopeFinder finder(partition.owner);
    finder(synchronized[i]->body);
    ICHECK(finder.envelope != nullptr);
    branches.push_back(
        IfThenElse(partition.cases[i].condition, finder.envelope->body));
  }
  PrimFunc result = partition.original;
  Stmt body =
      EnvelopeRewriter(partition.owner, SeqStmt(branches))(result->body);
  result.CopyOnWrite()->body = ConvertSSA(std::move(body));
  return result;
}

} // namespace tvm::tl::ascend_sync
