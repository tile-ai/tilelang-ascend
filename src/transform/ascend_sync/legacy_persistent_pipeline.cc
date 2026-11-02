// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

// A private, closed-subgraph composition boundary. Both graphs use the real
// upstream engine. The six S-direction handshakes close ALL active queues,
// including initializer/drain traffic, before the next invocation can start.
// This deliberately serializes Persistent waves while retaining inner SIP.

#include "legacy_persistent_pipeline.h"

#include <tvm/runtime/logging.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt_functor.h>

#include <array>
#include <string>
#include <unordered_set>
#include <utility>

#include "../../op/ascend.h"
#include "../common/pipeline_ring_provenance.h"
#include "legacy_adapter.h"
#include "upstream/insert_sync.h"

namespace tvm::tl::ascend_sync {
using namespace tir;

namespace {

constexpr const char *kReject =
    "Persistent pipeline composition requires one certified static full64 "
    "FP32 Vector B2/lead2 SIP inside a canonical multiwave Persistent";

void RequirePure(const PrimExpr &expression, const char *where) {
  ICHECK(expression.defined() &&
         SideEffect(expression) <= CallEffectKind::kPure)
      << kReject << "; non-pure " << where;
}

int64_t Constant(const PrimExpr &expression, const char *where) {
  RequirePure(expression, where);
  const auto *value = expression.as<IntImmNode>();
  ICHECK(value != nullptr) << kReject << "; static " << where << " required";
  return value->value;
}

std::string Operation(const CallNode *call) {
  std::string name;
  if (call->op.same_as(builtin::call_extern())) {
    ICHECK(!call->args.empty() && call->args[0].as<StringImmNode>()) << kReject;
    name = Downcast<StringImm>(call->args[0])->value;
  } else if (const auto *op = call->op.as<OpNode>()) {
    name = op->name;
  }
  const size_t begin = name.find('<');
  if (begin != std::string::npos)
    name.erase(begin);
  const std::string prefix = "tl::ascend::";
  if (name.rfind(prefix, 0) == 0)
    name.erase(0, prefix.size());
  return name;
}

struct Frame {
  int64_t segment;
  int64_t stage;
  int64_t order;
  int64_t unrolled;
  PrimExpr clock;
  std::string operation;
};

class NestedOwnerFinder : public StmtExprVisitor {
public:
  bool nested{false};
  bool top_dynamic_owner{false};
  bool uncertified_clock_invocation{false};
  std::vector<Var> references;

private:
  void VisitStmt_(const ForNode *op) final {
    ++depth_;
    loops_.push_back(op->loop_var);
    StmtExprVisitor::VisitStmt_(op);
    loops_.pop_back();
    --depth_;
  }
  void VisitStmt_(const IfThenElseNode *op) final {
    ++depth_;
    StmtExprVisitor::VisitStmt_(op);
    --depth_;
  }
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == kPhysicalPipelineExtent) {
      nested |= depth_ != 0;
      top_dynamic_owner |= depth_ == 0 && !op->value.as<IntImmNode>();
      references.push_back(PhysicalRingClockId(op->node));
    } else if (op->attr_key == kPhysicalPipelineSegment) {
      references.push_back(PhysicalRingClockId(op->node));
    } else if (op->attr_key == kPhysicalPipelineStage) {
      const auto record = Downcast<Array<PrimExpr>>(op->node);
      ICHECK_EQ(record.size(), 5U) << kReject << "; malformed stage record";
      references.push_back(Downcast<Var>(record[0]));
    } else if (op->attr_key == kPhysicalRingClock) {
      RequirePure(op->value, "logical ring clock");
      uncertified_clock_invocation |= depth_ >= 2;
      if (depth_ == 1 && loops_.size() == 1) {
        arith::Analyzer analyzer;
        const auto variables = UndefinedVars(analyzer.Simplify(op->value));
        bool depends_on_iteration = false;
        for (const Var &variable : variables)
          depends_on_iteration |= variable.same_as(loops_[0]);
        uncertified_clock_invocation |= !depends_on_iteration;
      }
    }
    // Discover all owner references. An ordinary steady For INSIDE its own
    // envelope does not affect nested: only a new envelope tests depth_.
    StmtExprVisitor::VisitStmt_(op);
  }
  size_t depth_{0};
  std::vector<Var> loops_;
};

// Audit original operations before hiding any subtree behind an invocation
// task. Public provenance is verified as data; it never grants this capability.
class ClosedOwnerAudit : public StmtExprVisitor {
public:
  ClosedOwnerAudit(const PrimFunc &function,
                   const LegacyPersistentLoopCertificate &certificate)
      : function_(function), certificate_(certificate) {}

  void Run() {
    ICHECK(certificate_.wave > 0 &&
           certificate_.domain / certificate_.wave >= 2)
        << kReject
        << "; only unpredicated static multiwave domains are admitted";
    ICHECK(function_->GetAttr<Bool>("tir.noalias").value_or(Bool(false)))
        << kReject << "; independent GM parameters require tir.noalias";
    const auto owners =
        function_->GetAttr<PhysicalPipelineOwners>(kPhysicalPipelineOwners);
    ICHECK(owners.defined() && owners.value().size() == 1U) << kReject;
    for (const auto &[identity, record] : owners.value()) {
      ICHECK_EQ(record.size(), 4U) << kReject;
      ICHECK_EQ(Constant(record[0], "owner schema"), 1) << kReject;
      ICHECK_EQ(Constant(record[1], "actual lead"), 2) << kReject;
      ICHECK_EQ(Constant(record[2], "original minimum"), 0) << kReject;
      ICHECK_EQ(Constant(record[3], "stage count"), 3) << kReject;
      owner_id_ = identity;
    }
    const auto provenance =
        function_->GetAttr<PhysicalRingProvenance>(kPhysicalRingProvenance);
    ICHECK(provenance.defined() && provenance.value().size() == 1U) << kReject;
    for (const auto &[identity, record] : provenance.value()) {
      ValidatePhysicalRingRecord(record);
      ICHECK_EQ(Constant(record[kPhysicalRingVersions], "true ring B"), 2)
          << kReject;
      ICHECK_EQ(Constant(record[kPhysicalRingStrideBits], "slot stride"), 2048)
          << kReject;
      ICHECK_EQ(Constant(record[kPhysicalRingElementBits], "element bits"), 32)
          << kReject;
      ICHECK_EQ(Constant(record[kPhysicalRingOriginalMin], "ring minimum"), 0)
          << kReject;
      ring_id_ = identity;
      ring_root_ = Downcast<Var>(record[kPhysicalRingRoot]);
    }
    for (const auto &[parameter, buffer] : function_->buffer_map) {
      (void)parameter;
      ICHECK(buffer->dtype == DataType::Float(32)) << kReject;
      parameter_data_.insert(buffer->data);
    }
    VisitStmt(function_->body);
    ICHECK_EQ(outer_count_, 1U)
        << kReject << "; stale Persistent loop certificate";
    ICHECK_EQ(owner_count_, 1U) << kReject << "; missing or repeated owner";
    // Simplify removes the genuine extent-one steady For for N=3. Keep its
    // materialized stage clocks and the true storage B2; do not create an
    // artificial epoch owner. The flat graph is analyzed by the same engine.
    ICHECK(steady_count_ == 1U || (extent_ == 3 && steady_count_ == 0U))
        << kReject;
    ICHECK_EQ(frames_.size(), 9U) << kReject;
    const PrimExpr index = steady_.defined()
                               ? PrimExpr(steady_->loop_var)
                               : make_zero(frames_[2].clock.dtype());
    const int64_t n = extent_;
    const std::array<Frame, 9> expected{{
        {0, 0, 1, 1, Integer(0), "copy_gm_to_ub"},
        {0, 0, 1, 1, Integer(1), "copy_gm_to_ub"},
        {1, 2, 0, 0, index, "tl.ascend_abs"},
        {1, 0, 1, 0, index + 2, "copy_gm_to_ub"},
        {1, 2, 2, 0, index, "copy_ub_to_gm"},
        {2, 2, 0, 1, make_const(index.dtype(), n - 2), "tl.ascend_abs"},
        {2, 2, 2, 1, make_const(index.dtype(), n - 2), "copy_ub_to_gm"},
        {2, 2, 0, 1, make_const(index.dtype(), n - 1), "tl.ascend_abs"},
        {2, 2, 2, 1, make_const(index.dtype(), n - 1), "copy_ub_to_gm"},
    }};
    arith::Analyzer analyzer;
    for (size_t i = 0; i < expected.size(); ++i) {
      const Frame &actual = frames_[i], &want = expected[i];
      ICHECK(actual.segment == want.segment && actual.stage == want.stage &&
             actual.order == want.order && actual.unrolled == want.unrolled &&
             actual.operation == want.operation &&
             is_zero(analyzer.Simplify(actual.clock - want.clock)))
          << kReject << "; stage/segment/order/logical clock mismatch at " << i;
    }
    if (steady_.defined())
      ICHECK_EQ(Constant(steady_->extent, "steady extent"), n - 2) << kReject;
  }

  Stmt owner;
  size_t owner_occurrence{0};
  ConstrSet owner_context;

private:
  void VisitStmt(const Stmt &statement) final {
    ICHECK(statement.as<SeqStmtNode>() || statement.as<ForNode>() ||
           statement.as<AttrStmtNode>() || statement.as<LetStmtNode>() ||
           statement.as<AllocateNode>() || statement.as<DeclBufferNode>() ||
           statement.as<EvaluateNode>())
        << kReject << "; unsupported statement " << statement->GetTypeKey();
    ++occurrence_;
    StmtExprVisitor::VisitStmt(statement);
  }

  void VisitStmt_(const ForNode *op) final {
    RequirePure(op->min, "For minimum");
    RequirePure(op->extent, "For extent");
    ICHECK(is_zero(op->min) && op->kind == ForKind::kSerial &&
           !op->thread_binding.defined())
        << kReject;
    const ConstrSet saved = context_;
    const bool is_outer = GetRef<For>(op).same_as(certificate_.loop);
    if (is_outer) {
      ICHECK(!in_outer_ && !in_owner_) << kReject;
      ++outer_count_;
      in_outer_ = true;
      context_ = certificate_.header_context;
    } else {
      ICHECK(in_owner_ && segment_ == kPipelineSteady && !steady_.defined())
          << kReject << "; extra or nested loop";
      ICHECK_EQ(Constant(op->extent, "steady extent"), extent_ - 2) << kReject;
      steady_ = GetRef<For>(op);
      ++steady_count_;
      ICHECK_EQ(op->annotations.size(), 1U) << kReject;
      auto interval = op->annotations.Get("tl_cross_interval");
      ICHECK(interval.defined() && interval.value().as<IntImmNode>() &&
             Downcast<IntImm>(interval.value())->value == 1)
          << kReject;
    }
    context_.AddConstr(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    context_.AddConstr(op->extent > 0);
    VisitStmt(op->body);
    context_ = saved;
    if (is_outer)
      in_outer_ = false;
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    RequirePure(op->value, "attribute value");
    const ConstrSet saved_context = context_;
    const int64_t saved_resource = resource_;
    if (op->attr_key == "resource_scope") {
      resource_ = Constant(op->value, "resource scope");
      ICHECK_EQ(resource_, 1) << kReject << "; Vector only";
    }
    if (op->attr_key == "thread_extent" || op->attr_key == "virtual_thread") {
      const auto *iv = op->node.as<IterVarNode>();
      ICHECK(iv != nullptr && !in_outer_) << kReject;
      context_.AddConstr(iv->var, Range::FromMinExtent(
                                      make_zero(op->value.dtype()), op->value));
    } else if (op->attr_key == kPhysicalPipelineExtent) {
      ICHECK(in_outer_ && !in_owner_ && resource_ == 1) << kReject;
      ICHECK(PhysicalRingClockId(op->node).same_as(owner_id_)) << kReject;
      extent_ = Constant(op->value, "inner extent");
      ICHECK_GE(extent_, 3) << kReject;
      ++owner_count_;
      owner = GetRef<AttrStmt>(op);
      owner_occurrence = occurrence_;
      owner_context = context_;
      in_owner_ = true;
      VisitStmt(op->body);
      in_owner_ = false;
      context_ = saved_context;
      resource_ = saved_resource;
      return;
    } else if (op->attr_key == kPhysicalPipelineSegment) {
      ICHECK(in_owner_ && PhysicalRingClockId(op->node).same_as(owner_id_))
          << kReject;
      const int64_t previous = segment_;
      segment_ = Constant(op->value, "segment");
      ICHECK(segment_ >= 0 && segment_ <= 2) << kReject;
      VisitStmt(op->body);
      segment_ = previous;
      return;
    } else if (op->attr_key == kPhysicalPipelineStage) {
      ICHECK(in_owner_ && frame_ == nullptr && segment_ >= 0) << kReject;
      const Array<PrimExpr> record = Downcast<Array<PrimExpr>>(op->node);
      ICHECK_EQ(record.size(), 5U) << kReject;
      ICHECK(record[0].same_as(owner_id_)) << kReject;
      ICHECK_EQ(Constant(record[1], "stage segment"), segment_) << kReject;
      Frame frame{segment_,
                  Constant(record[2], "stage"),
                  Constant(record[3], "order"),
                  Constant(record[4], "unrolled marker"),
                  op->value,
                  ""};
      frame_ = &frame;
      VisitStmt(op->body);
      ICHECK(!frame.operation.empty()) << kReject << "; empty stage";
      frames_.push_back(std::move(frame));
      frame_ = nullptr;
      return;
    } else if (op->attr_key == kPhysicalRingClock) {
      ICHECK(in_owner_ && frame_ != nullptr &&
             PhysicalRingClockId(op->node).same_as(ring_id_))
          << kReject;
      arith::Analyzer analyzer;
      ICHECK(is_zero(analyzer.Simplify(op->value - frame_->clock))) << kReject;
    } else {
      ICHECK(op->attr_key == "thread_extent" ||
             op->attr_key == "virtual_thread" ||
             op->attr_key == "resource_scope")
          << kReject << "; unaudited attribute " << op->attr_key;
    }
    VisitStmt(op->body);
    context_ = saved_context;
    resource_ = saved_resource;
  }

  void VisitStmt_(const LetStmtNode *op) final {
    RequirePure(op->value, "Let definition (mutable scalar unsupported)");
    const ConstrSet saved = context_;
    context_.AddConstr(op->var, op->value);
    VisitStmt(op->body);
    context_ = saved;
  }

  void VisitStmt_(const AllocateNode *op) final {
    ICHECK(!in_outer_ && !in_owner_ && op->dtype == DataType::Float(32) &&
           is_one(op->condition))
        << kReject << "; allocation must enclose waves";
    for (const PrimExpr &extent : op->extents)
      ICHECK_GT(Constant(extent, "allocation extent"), 0) << kReject;
    VisitStmt(op->body);
  }

  void VisitStmt_(const DeclBufferNode *op) final {
    ICHECK(!in_outer_ && !in_owner_) << kReject;
    VisitStmt(op->body);
  }

  void VisitStmt_(const IfThenElseNode *) final {
    LOG(FATAL) << kReject << "; conditional owner or stage is unsupported";
  }
  void VisitStmt_(const BufferStoreNode *) final {
    LOG(FATAL) << kReject << "; mutable scalar/extra compute is unsupported";
  }
  void VisitStmt_(const WhileNode *) final { LOG(FATAL) << kReject; }

  void VisitStmt_(const EvaluateNode *op) final {
    if (op->value.as<IntImmNode>())
      return;
    const auto *call = op->value.as<CallNode>();
    ICHECK(call != nullptr) << kReject;
    const std::string operation = Operation(call);
    ICHECK(operation.find("flag") == std::string::npos &&
           operation.find("barrier") == std::string::npos)
        << kReject << "; user manual/automatic synchronization is unsupported";
    ICHECK(in_owner_ && frame_ != nullptr && frame_->operation.empty())
        << kReject << "; extra compute outside the closed owner/stage";
    ICHECK(operation == "copy_gm_to_ub" || operation == "tl.ascend_abs" ||
           operation == "copy_ub_to_gm")
        << kReject << "; unaudited helper";
    const size_t offset = call->op.same_as(builtin::call_extern()) ? 1U : 0U;
    ICHECK_GE(call->args.size(), offset + 3) << kReject;
    for (size_t i = offset; i < call->args.size(); ++i) {
      const PrimExpr &argument = call->args[i];
      const auto *pointer = argument.as<CallNode>();
      if (i < offset + 2) {
        ICHECK(pointer && pointer->op.same_as(builtin::tvm_access_ptr()) &&
               pointer->args.size() == 5U)
            << kReject;
        const auto *annotation = pointer->args[0].as<CallNode>();
        ICHECK(annotation && annotation->dtype == DataType::Float(32))
            << kReject;
        RequirePure(pointer->args[2], "access offset");
        ICHECK_EQ(Constant(pointer->args[3], "access extent"), 64) << kReject;
        const Var data = Downcast<Var>(pointer->args[1]);
        const bool ring_operand =
            (operation == "copy_gm_to_ub" && i == offset + 1) ||
            (operation == "tl.ascend_abs" && i == offset + 1);
        const bool result_operand =
            (operation == "tl.ascend_abs" && i == offset) ||
            (operation == "copy_ub_to_gm" && i == offset);
        if (ring_operand) {
          ICHECK(data.same_as(ring_root_))
              << kReject << "; ring lineage changed";
        } else if (result_operand) {
          const auto *type = data->type_annotation.as<PointerTypeNode>();
          ICHECK(type != nullptr && type->storage_scope == "shared.ub" &&
                 !data.same_as(ring_root_))
              << kReject << "; B1 result requires an independent UB root";
          ICHECK_EQ(Constant(pointer->args[2], "B1 result offset"), 0)
              << kReject << "; B1 result footprint must start at its root";
          if (!result_root_.defined())
            result_root_ = data;
          ICHECK(data.same_as(result_root_.value()))
              << kReject << "; B1 result lineage changed";
          const auto sizes = function_->GetAttr<Map<Var, PrimExpr>>("size_map");
          ICHECK(sizes.defined() && sizes.value().count(data)) << kReject;
          ICHECK_GE(Constant(sizes.value().at(data), "B1 result physical size"),
                    256)
              << kReject << "; B1 result footprint exceeds its physical span";
        } else {
          ICHECK(parameter_data_.count(data)) << kReject << "; unknown GM root";
        }
      } else {
        RequirePure(argument, "helper scalar argument");
      }
    }
    if (operation == "tl.ascend_abs") {
      ICHECK_EQ(call->args.size(), 3U) << kReject;
      ICHECK_EQ(Constant(call->args[2], "Abs count"), 64) << kReject;
    } else {
      // Template dimensions control the REAL DMA footprint, independently of
      // the access_ptr interval. Exact first-slice helpers are full 1x64 f32.
      const std::string helper = Downcast<StringImm>(call->args[0])->value;
      ICHECK(helper == "tl::ascend::" + operation + "<float, 64>") << kReject;
      ICHECK_GE(call->args.size(), 6U) << kReject;
      ICHECK_EQ(Constant(call->args[3], "DMA row stride"), 64) << kReject;
      ICHECK_EQ(Constant(call->args[4], "DMA rows"), 1) << kReject;
      ICHECK_EQ(Constant(call->args[5], "DMA columns"), 64) << kReject;
    }
    frame_->operation = operation;
  }

  PrimFunc function_;
  const LegacyPersistentLoopCertificate &certificate_;
  ConstrSet context_;
  Var owner_id_, ring_id_, ring_root_;
  Optional<Var> result_root_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> parameter_data_;
  For steady_;
  int64_t extent_{0};
  size_t occurrence_{0}, outer_count_{0}, owner_count_{0}, steady_count_{0};
  int64_t resource_{-1}, segment_{-1};
  bool in_outer_{false}, in_owner_{false};
  Frame *frame_{nullptr};
  std::vector<Frame> frames_;
};

Stmt FlagPair(const std::string &event) {
  return SeqStmt({Evaluate(Call(DataType::Handle(), tl::ascend_auto_set_flag(),
                                {StringImm(event), Integer(0)})),
                  Evaluate(Call(DataType::Handle(), tl::ascend_auto_wait_flag(),
                                {StringImm(event), Integer(0)}))});
}

Stmt Seal(const Stmt &body) {
  Array<Stmt> statements;
  for (const char *pipe : {"V", "MTE2", "MTE3"})
    statements.push_back(FlagPair(std::string("S_") + pipe));
  statements.push_back(body);
  for (const char *pipe : {"V", "MTE2", "MTE3"})
    statements.push_back(FlagPair(std::string(pipe) + "_S"));
  return AttrStmt(Integer(0), "resource_scope", Integer(1),
                  SeqStmt(statements));
}

Stmt Reservations() {
  Array<Stmt> statements;
  for (const char *pipe : {"V", "MTE2", "MTE3"}) {
    statements.push_back(FlagPair(std::string("S_") + pipe));
    statements.push_back(FlagPair(std::string(pipe) + "_S"));
  }
  return SeqStmt(statements);
}

void Synchronize(ImportedLegacySyncIR *imported, SyncTargetPolicy policy) {
  policy.physical_conflict_oracle = imported->MakeOracle();
  std::vector<std::shared_ptr<IRStructure>> synchronized;
  for (auto &graph : imported->local_graphs) {
    ICHECK_EQ(graph.owner, kCoreVector) << kReject;
    RunSyncEngine(graph.roots, imported->multibuffer,
                  imported->manual_buffer_versions, graph.reservations, policy);
    synchronized.insert(synchronized.end(), graph.roots.begin(),
                        graph.roots.end());
  }
  imported->roots = std::move(synchronized);
}

void CheckInnerPipes(const std::vector<std::shared_ptr<IRStructure>> &nodes) {
  for (const auto &node : nodes) {
    if (node->IsTask()) {
      const uint16_t pipe = node->GetPipeMask();
      ICHECK(pipe == PipeMask(ResourcePipe::kMTE2) ||
             pipe == PipeMask(ResourcePipe::kVector) ||
             pipe == PipeMask(ResourcePipe::kMTE3))
          << kReject << "; internal Scalar/other hardware task";
    } else {
      CheckInnerPipes(static_cast<ControlNode *>(node.get())->children);
    }
  }
}

} // namespace

bool LegacyPersistentPipelineImport::MatchesSource(
    const PrimFunc &function) const {
  return source_.defined() && source_.same_as(function) &&
         source_->body.same_as(function->body);
}

bool LegacyPersistentPipelineImport::AllowsOuterLoop(const For &loop) const {
  return outer_loop_.defined() && loop.same_as(outer_loop_);
}

std::optional<PrimFunc> SynchronizeLegacyPersistentPipeline(
    const LegacyPersistentNormalization &normalization, const Target &target,
    const String &platform, bool ordinary_sync,
    const SyncTargetPolicy &policy) {
  const PrimFunc &function = normalization.function;
  const auto owners =
      function->GetAttr<PhysicalPipelineOwners>(kPhysicalPipelineOwners);
  NestedOwnerFinder finder;
  finder(function->body);
  ICHECK(!finder.nested || !normalization.loops.empty())
      << kReject
      << "; outer semantic control has a nested owner with no private "
         "Persistent certificate";
  for (const Var &reference : finder.references)
    ICHECK(owners.defined() && owners.value().count(reference))
        << kReject << "; orphan pipeline owner metadata reference";
  const bool closed_candidate = !normalization.loops.empty() && finder.nested &&
                                owners.defined() && !owners.value().empty();
  // Dynamic top-level SIP is still handed to the existing real partitioner,
  // whose complete certificate/guard validation precedes engine import. Its
  // bounds If inside a steady For is not a generic nested ring invocation.
  const bool partition_candidate = finder.top_dynamic_owner && !finder.nested &&
                                   owners.defined() && !owners.value().empty();
  ICHECK(!finder.uncertified_clock_invocation || closed_candidate ||
         partition_candidate)
      << kReject << "; uncertified generic nested/reset ring invocation";
  if (!owners.defined() || owners.value().empty())
    return std::nullopt;
  if (normalization.loops.empty()) {
    return std::nullopt;
  }
  ICHECK_EQ(normalization.loops.size(), 1U) << kReject;
  ICHECK(ordinary_sync && policy.ordinary_sync && policy.vector_scalar_sync)
      << kReject
      << "; ordinary and Vector/Scalar synchronization must be enabled";
  const std::string model =
      target->GetAttr<String>("model").value_or(String(""));
  ICHECK(platform == "A3" && (model == "ascendc" || model == "auto"))
      << kReject << "; AscendC/auto A3 only";
  ClosedOwnerAudit audit(function, normalization.loops[0]);
  audit.Run();
  auto storage = std::make_shared<PhysicalStorageResolver>(function);
  ICHECK_EQ(storage->ManualRingVersions().size(), 1U) << kReject;

  LegacyPersistentPipelineImport inner;
  inner.source_ = function;
  inner.scoped_root_ = audit.owner;
  inner.root_occurrence_ = audit.owner_occurrence;
  inner.root_context_ = audit.owner_context;
  inner.storage_ = storage;
  inner.reservations_ = Reservations();
  auto imported_inner = ImportLegacySync(function, target, platform,
                                         ordinary_sync, nullptr, &inner);
  CheckInnerPipes(imported_inner.roots);

  // Whole physical spans intentionally coalesce input/output aliases, and a
  // fresh logical identity separates an invocation epoch from the REAL B2
  // storage epoch. The resolver still resolves every view to its real root.
  LegacyPersistentPipelineImport outer;
  outer.source_ = function;
  outer.outer_loop_ = normalization.loops[0].loop;
  outer.closed_owner_ = audit.owner;
  outer.closed_occurrence_ = audit.owner_occurrence;
  outer.storage_ = storage;
  outer.reservations_ = inner.reservations_;
  for (const auto &node : imported_inner.roots) {
    for (const BufferRegion &region : node->GetReadRegions())
      outer.reads_.push_back(storage->InvocationProjection(region));
    for (const BufferRegion &region : node->GetWriteRegions())
      outer.writes_.push_back(storage->InvocationProjection(region));
  }
  // Runtime parameters and outer worker/wave coordinates retain their scalar
  // dependence. Opaque metadata identities are not runtime scalar operands.
  for (const Var &variable : UndefinedVars(audit.owner, {}))
    if (!variable.dtype().is_handle())
      outer.read_vars_.push_back(variable);

  Synchronize(&imported_inner, policy);
  Stmt synchronized_owner;
  ExportLegacySync(imported_inner, &synchronized_owner);
  ICHECK(synchronized_owner.defined())
      << kReject << "; missing inner scope export";
  outer.closed_replacement_ = Seal(synchronized_owner);
  auto imported_outer = ImportLegacySync(function, target, platform,
                                         ordinary_sync, nullptr, &outer);
  Synchronize(&imported_outer, policy);
  return ExportLegacySync(imported_outer);
}

} // namespace tvm::tl::ascend_sync
