// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

#include "legacy_cross_core.h"

#include <tvm/runtime/logging.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <charconv>
#include <limits>
#include <map>
#include <string>
#include <unordered_set>
#include <utility>

#include "../common/pipeline_ring_provenance.h"
#include "legacy_runtime_contract.h"

namespace tvm::tl::ascend_sync {

using namespace tir;

namespace {

constexpr const char *kDiagnostic = "Legacy flat C/V publication";

std::string Name(const CallNode *call, size_t *offset) {
  *offset = 0;
  if (call->op.same_as(builtin::call_extern())) {
    if (call->args.empty() || !call->args[0].as<StringImmNode>())
      return {};
    *offset = 1;
    return Downcast<StringImm>(call->args[0])->value;
  }
  const auto *op = call->op.as<OpNode>();
  return op == nullptr ? "" : std::string(op->name);
}

int Literal(const PrimExpr &expression, int maximum, const char *what) {
  const auto *value = expression.as<IntImmNode>();
  ICHECK(value != nullptr && value->value >= 0 && value->value <= maximum)
      << kDiagnostic << " requires a literal " << what << " in [0, " << maximum
      << "]";
  return static_cast<int>(value->value);
}

std::string Pipe(const PrimExpr &expression) {
  const auto *literal = expression.as<StringImmNode>();
  ICHECK(literal != nullptr) << kDiagnostic << " requires a literal pipeline";
  std::string result = literal->value;
  if (result.rfind("PIPE_", 0) == 0)
    result.erase(0, 5);
  return result;
}

bool IsGM(const BufferRegion &region) {
  return region->buffer.scope().empty() || region->buffer.scope() == "global";
}

class FlatInputValidator : public StmtExprVisitor {
public:
  FlatInputValidator(const PrimFunc &function,
                     const std::vector<LegacyOwnedTask> &tasks,
                     const std::vector<LegacyOwnedControl> &controls) {
    if (auto ratio = function->GetAttr<ObjectRef>("npu_cv_ratio")) {
      const auto *literal = ratio.value().as<StringImmNode>();
      ICHECK(literal != nullptr && literal->value == "cv_1_2")
          << kDiagnostic
          << " requires npu_cv_ratio to be StringImm cv_1_2 when present";
      developer_ratio_ = true;
    }
    for (const auto &task : tasks) {
      ICHECK(task.task != nullptr && task.original_statement.defined())
          << kDiagnostic << " requires complete owned-task records";
      ICHECK(by_order_.emplace(task.order, &task).second)
          << kDiagnostic << " has duplicate statement occurrence records";
    }
    ICHECK_LE(controls.size(), 1U)
        << kDiagnostic << " supports only one static C-only prefix For";
    for (const auto &control : controls) {
      ICHECK(control.control != nullptr && control.control->task != nullptr &&
             control.owner == kCoreCube &&
             control.control->task->GetCoreMask() == kCoreCube &&
             controls_.emplace(control.order, &control).second)
          << kDiagnostic << " requires complete C-only control records";
    }
  }

  void Run(const Stmt &body) {
    VisitStmt(body);
    ICHECK_EQ(seen_, by_order_.size())
        << kDiagnostic << " task records do not match the original function";
    ICHECK_EQ(seen_controls_, controls_.size())
        << kDiagnostic << " control records do not match the original function";
    ICHECK_EQ(vector_topologies_, 1)
        << kDiagnostic << " requires one unambiguous C1/V2 subgroup topology";
  }

  const std::vector<Var> &PrivateVariables() const { return private_vars_; }
  Var Subcore() const { return subcore_.value(); }
  bool HasSingleWorkgroup() const {
    return workgroups_.size() == 1 && is_one(workgroups_[0]);
  }

  void CheckPublicationAfterPrefix(const LegacyOwnedTask &set) const {
    for (const auto &[begin, end] : control_spans_)
      ICHECK(end < set.order)
          << kDiagnostic
          << " requires the complete C-only prefix For before "
             "the unique outside-loop Set";
  }

private:
  void VisitStmt(const Stmt &statement) final {
    ++occurrence_;
    ICHECK(statement.as<SeqStmtNode>() || statement.as<AttrStmtNode>() ||
           statement.as<AllocateNode>() || statement.as<DeclBufferNode>() ||
           statement.as<LetStmtNode>() || statement.as<EvaluateNode>() ||
           statement.as<BufferStoreNode>() || statement.as<ForNode>())
        << kDiagnostic
        << " requires flat input without For/If/While or "
           "unlowered control: "
        << statement->GetTypeKey();
    StmtExprVisitor::VisitStmt(statement);
  }

  void CheckTask(const Stmt &statement, const LetStmtNode *binding = nullptr) {
    auto found = by_order_.find(occurrence_);
    ICHECK(found != by_order_.end())
        << kDiagnostic << " is missing a real effect/statement occurrence";
    const LegacyOwnedTask &record = *found->second;
    ICHECK(record.original_statement.same_as(statement))
        << kDiagnostic << " occurrence has the wrong original statement";
    ICHECK((owner_ == kCoreCube || owner_ == kCoreVector) &&
           record.owner == owner_ && record.task->GetCoreMask() == owner_)
        << kDiagnostic << " requires an explicit, unsplit C or V owner";
    ICHECK(!record.task->HasConditions())
        << kDiagnostic << " does not support subcore/conditional predicates";
    if (inside_c_loop_) {
      ICHECK(record.owner == kCoreCube &&
             !DecodeLegacyCrossFlag(record.task->stmt).has_value())
          << kDiagnostic
          << " requires all cross flags outside the C prefix For";
      for (const BufferRegion &region : record.task->GetWriteRegions())
        ICHECK(!IsGM(region)) << kDiagnostic
                              << " C-only prefix For may read GM but cannot "
                                 "publish or overwrite GM inside the loop";
    }
    if (binding != nullptr) {
      const auto *definition = record.task->stmt.as<EvaluateNode>();
      ICHECK(definition != nullptr &&
             definition->value.same_as(binding->value) &&
             record.task->GetWriteVars().size() == 1 &&
             record.task->GetWriteVars()[0].same_as(binding->var))
          << kDiagnostic << " has an invalid snapshot definition record";
    } else {
      ICHECK(record.task->stmt.same_as(statement))
          << kDiagnostic << " must retain original task statements";
    }
    ++seen_;
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (!op->value.as<IntImmNode>())
      CheckTask(GetRef<Stmt>(op));
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    CheckTask(GetRef<Stmt>(op));
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const LetStmtNode *op) final {
    private_vars_.push_back(op->var);
    if (SideEffect(op->value) > CallEffectKind::kPure)
      CheckTask(GetRef<Stmt>(op), op);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    const size_t begin = occurrence_;
    auto found = controls_.find(begin);
    ICHECK(found != controls_.end() && !inside_c_loop_ &&
           found->second->original_statement.same_as(GetRef<Stmt>(op)) &&
           found->second->control->control.same_as(GetRef<For>(op)))
        << kDiagnostic
        << " requires flat input except one certified C-only "
           "prefix For; V or nested controls are unsupported";
    const auto *minimum = op->min.as<IntImmNode>();
    const auto *extent = op->extent.as<IntImmNode>();
    ICHECK(op->kind == ForKind::kSerial && minimum && minimum->value == 0 &&
           extent && extent->value > 0 && op->annotations.empty() &&
           !op->thread_binding.defined())
        << kDiagnostic
        << " requires a static positive serial C-only prefix "
           "For with min=0";
    inside_c_loop_ = true;
    private_vars_.push_back(op->loop_var);
    StmtExprVisitor::VisitStmt_(op);
    inside_c_loop_ = false;
    control_spans_.push_back({begin, occurrence_});
    ++seen_controls_;
  }

  void VisitStmt_(const AllocateNode *op) final {
    ICHECK(is_one(op->condition))
        << kDiagnostic << " does not support predicated allocation";
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    const CoreMask previous = owner_;
    if (op->attr_key == "resource_scope") {
      const int resource = Literal(op->value, 1, "resource scope");
      owner_ = resource == 0 ? kCoreCube : kCoreVector;
    } else if (op->attr_key == "thread_extent") {
      const auto *iteration = op->node.as<IterVarNode>();
      ICHECK(iteration != nullptr)
          << kDiagnostic << " has malformed thread topology";
      const bool expert = iteration->thread_tag == "blockIdx.y";
      const bool developer = iteration->thread_tag == "threadIdx.x";
      if (expert || developer) {
        // The compiler-managed Developer path uses threadIdx.x after
        // VidReduction. A bare threadIdx.x is not sufficient evidence of the
        // C1/V2 mapping: preserve the kernel's explicit ratio contract.
        ICHECK(!developer || developer_ratio_)
            << kDiagnostic
            << " requires npu_cv_ratio=StringImm cv_1_2 for threadIdx.x";
        ICHECK_EQ(Literal(op->value, 2, "AIV subgroup extent"), 2)
            << kDiagnostic << " requires C1/V2 topology";
        ++vector_topologies_;
        ICHECK_EQ(vector_topologies_, 1)
            << kDiagnostic
            << " requires one unambiguous C1/V2 subgroup topology";
        subcore_ = iteration->var;
        private_vars_.push_back(iteration->var);
      } else {
        ICHECK(iteration->thread_tag == "blockIdx.x")
            << kDiagnostic << " has an unsupported thread topology";
        workgroups_.push_back(op->value);
      }
    } else if (op->attr_key == "virtual_thread") {
      LOG(FATAL) << kDiagnostic << " does not support virtual thread topology";
    }
    StmtExprVisitor::VisitStmt_(op);
    owner_ = previous;
  }

  void VisitExpr(const PrimExpr &expression) final {
    if (subcore_.defined() && expression.dtype().is_bool()) {
      ICHECK(!UsesVar(expression,
                      [&](const VarNode *var) {
                        return subcore_.value().same_as(GetRef<Var>(var));
                      }))
          << kDiagnostic << " does not support a subcore predicate";
    }
    StmtExprVisitor::VisitExpr(expression);
  }

  std::map<size_t, const LegacyOwnedTask *> by_order_;
  std::map<size_t, const LegacyOwnedControl *> controls_;
  std::vector<std::pair<size_t, size_t>> control_spans_;
  Optional<Var> subcore_;
  std::vector<Var> private_vars_;
  std::vector<PrimExpr> workgroups_;
  CoreMask owner_{kCoreUnassigned};
  size_t occurrence_{0}, seen_{0};
  size_t seen_controls_{0};
  int vector_topologies_{0};
  bool developer_ratio_{false};
  bool inside_c_loop_{false};
};

void CollectLocalReservation(const LegacyOwnedTask &record,
                             LegacyCrossCorePlan *plan) {
  const auto *evaluate = record.task->stmt.as<EvaluateNode>();
  const auto *call = evaluate ? evaluate->value.as<CallNode>() : nullptr;
  if (!call)
    return;
  size_t offset = 0;
  const std::string name = Name(call, &offset);
  const bool manual =
      name == "tl.ascend_set_flag" || name == "tl.ascend_wait_flag";
  const bool automatic =
      name == "tl.ascend_auto_set_flag" || name == "tl.ascend_auto_wait_flag";
  if (!manual && !automatic)
    return;
  ICHECK_EQ(call->args.size(), offset + (manual ? 3U : 2U))
      << kDiagnostic << " has a malformed local event";
  const std::string event =
      manual ? Pipe(call->args[offset]) + "_" + Pipe(call->args[offset + 1])
             : Pipe(call->args[offset]);
  const int id =
      Literal(call->args[offset + (manual ? 2 : 1)], 7, "local event ID");
  plan->reservations.push_back({record.owner, false, event, id});
}

// Without whole-function noalias, two AIV streams must not race through user
// outputs. Audit the actual DMA template, rather than access_ptr.extent (which
// may describe an entire workspace). This first slice accepts one dense,
// full-mask UB->GM transfer and proves both real AIV footprints disjoint.
void ValidateFreshWorkspaceOutput(const PrimFunc &function,
                                  const LegacyOwnedTask &record,
                                  const BufferRegion &output,
                                  const FlatInputValidator &validation) {
  const char *reject =
      "Trusted fresh-workspace C/V requires one full dense copy_ub_to_gm "
      "output with provably disjoint AIV footprints";
  const auto *evaluate = record.task->stmt.as<EvaluateNode>();
  const auto *call = evaluate ? evaluate->value.as<CallNode>() : nullptr;
  ICHECK(call != nullptr) << reject;
  size_t offset = 0;
  const std::string name = Name(call, &offset);
  const std::string prefix = "tl::ascend::copy_ub_to_gm<";
  ICHECK(name.rfind(prefix, 0) == 0 && name.back() == '>') << reject;
  std::vector<std::string> tokens;
  size_t position = prefix.size();
  while (position < name.size() - 1) {
    const size_t comma = name.find(',', position);
    const size_t end = comma == std::string::npos ? name.size() - 1 : comma;
    ICHECK(end < name.size() && end >= position) << reject;
    std::string token = name.substr(position, end - position);
    const size_t first = token.find_first_not_of(" \t\r\n");
    const size_t last = token.find_last_not_of(" \t\r\n");
    ICHECK(first != std::string::npos) << reject;
    tokens.push_back(token.substr(first, last - first + 1));
    position = end + 1;
  }
  ICHECK(tokens.size() == 2 || tokens.size() == 3) << reject;
  const DataType dtype = output->buffer->dtype;
  ICHECK(dtype.is_scalar() &&
         ((tokens[0] == "half" && dtype == DataType::Float(16)) ||
          (tokens[0] == "float" && dtype == DataType::Float(32))))
      << reject << "; unsupported DMA element type";
  auto dimension = [&](size_t index) -> int64_t {
    uint64_t value = 0;
    const std::string &token = tokens[index];
    const auto parsed =
        std::from_chars(token.data(), token.data() + token.size(), value);
    ICHECK(parsed.ec == std::errc() &&
           parsed.ptr == token.data() + token.size() && value > 0 &&
           value <= std::numeric_limits<uint32_t>::max())
        << reject;
    return static_cast<int64_t>(value);
  };
  const int64_t columns = dimension(1);
  const int64_t rows = tokens.size() == 3 ? dimension(2) : 1;
  ICHECK(columns <= std::numeric_limits<int64_t>::max() / rows) << reject;
  const int64_t volume = rows * columns;
  ICHECK(columns <= std::numeric_limits<int64_t>::max() / dtype.bytes() &&
         columns * dtype.bytes() % 32 == 0 && rows <= 4095 &&
         columns * dtype.bytes() <= 65535)
      << reject << "; unsupported DMA row alignment or hardware field";
  // AscendCopy can append two shape-metadata arguments. CopyCodegen forwards
  // only the first three scalar arguments to common.h's DMA helper.
  ICHECK(call->args.size() >= offset + 2 && call->args.size() <= offset + 7)
      << reject;
  const auto *pointer = call->args[offset + 1].as<CallNode>();
  ICHECK(pointer != nullptr && pointer->op.same_as(builtin::tvm_access_ptr()) &&
         pointer->args.size() == 5 &&
         pointer->args[1].same_as(output->buffer->data))
      << reject << "; output pointer identity mismatch";
  ProofAnalyzer mask_analyzer;
  record.task->outer_ctx.Populate(mask_analyzer);
  for (const auto &[index, expected] :
       {std::pair<size_t, int64_t>{offset + 2, columns},
        {offset + 3, rows},
        {offset + 4, columns}}) {
    if (index >= call->args.size()) {
      // C++ realdstN defaults to one; mask defaults use template dimensions.
      ICHECK(index != offset + 2 || columns == 1) << reject;
      continue;
    }
    const PrimExpr &argument = call->args[index];
    ICHECK(argument.dtype().is_scalar() &&
           ((argument.dtype().is_int() && argument.dtype().bits() <= 64) ||
            (argument.dtype().is_uint() && argument.dtype().bits() <= 32)) &&
           SideEffect(argument) <= CallEffectKind::kPure &&
           mask_analyzer.CanProve(cast(DataType::Int(64), argument) ==
                                  make_const(DataType::Int(64), expected)))
        << reject << "; unsupported stride or mask";
  }
  Optional<Buffer> parameter_buffer;
  for (const auto &[parameter, buffer] : function->buffer_map) {
    if (buffer->data.same_as(output->buffer->data)) {
      ICHECK(!parameter_buffer.defined()) << reject;
      parameter_buffer = buffer;
    }
  }
  ICHECK(parameter_buffer.defined() &&
         is_zero(parameter_buffer.value()->elem_offset) &&
         parameter_buffer.value()->strides.empty())
      << reject << "; requires a dense GM parameter";
  int64_t elements = 1;
  for (const PrimExpr &extent : parameter_buffer.value()->shape) {
    const auto *constant = extent.as<IntImmNode>();
    ICHECK(constant != nullptr && constant->value > 0 &&
           elements <= std::numeric_limits<int64_t>::max() / constant->value)
        << reject << "; requires static output capacity";
    elements *= constant->value;
  }
  const PrimExpr original = pointer->args[2];
  ICHECK(original.dtype().is_scalar() &&
         ((original.dtype().is_int() && original.dtype().bits() <= 64) ||
          (original.dtype().is_uint() && original.dtype().bits() <= 32)) &&
         SideEffect(original) <= CallEffectKind::kPure)
      << reject;
  PrimExpr starts[2];
  ProofAnalyzer footprints;
  for (int instance = 0; instance != 2; ++instance) {
    Map<Var, PrimExpr> substitution;
    for (const Var &variable : validation.PrivateVariables())
      substitution.Set(variable, variable.copy_with_suffix(
                                     instance == 0 ? "_aiv0" : "_aiv1"));
    substitution.Set(validation.Subcore(),
                     make_const(validation.Subcore().dtype(), instance));
    record.task->outer_ctx.Substitute(substitution)
        .ToConstraints()
        .Populate(footprints);
    starts[instance] =
        cast(DataType::Int(64), Substitute(original, substitution));
    ICHECK(
        footprints.CanProve(starts[instance] >= 0) &&
        footprints.CanProve(starts[instance] <=
                            make_const(DataType::Int(64), elements - volume)))
        << reject << "; output DMA exceeds its parameter allocation";
  }
  const PrimExpr span = make_const(DataType::Int(64), volume);
  ICHECK(footprints.CanProve(starts[0] <= starts[1] - span) ||
         footprints.CanProve(starts[1] <= starts[0] - span))
      << reject << "; AIV output footprints may overlap";
}

} // namespace

std::optional<LegacyCrossFlag> DecodeLegacyCrossFlag(const Stmt &statement) {
  const auto *evaluate = statement.as<EvaluateNode>();
  const auto *call = evaluate ? evaluate->value.as<CallNode>() : nullptr;
  if (!call)
    return std::nullopt;
  size_t offset = 0;
  const std::string name = Name(call, &offset);
  const bool manual_set = name == "tl.ascend_set_cross_flag";
  const bool auto_set = name == "tl.ascend_auto_set_cross_flag";
  const bool wait = name == "tl.ascend_wait_cross_flag" ||
                    name == "tl.ascend_auto_wait_cross_flag";
  if (!manual_set && !auto_set && !wait)
    return std::nullopt;
  ICHECK_EQ(call->args.size(), offset + (wait ? 2U : 3U))
      << kDiagnostic << " has a malformed existing cross-core call";
  if (wait) {
    return LegacyCrossFlag{LegacyCrossFlagKind::kWait,
                           Literal(call->args[offset], 15, "cross event ID"),
                           -1, Pipe(call->args[offset + 1])};
  }
  // Manual Set ABI: (pipe, id, mode). Auto Set ABI: (mode, pipe, id).
  return LegacyCrossFlag{
      LegacyCrossFlagKind::kSet,
      Literal(call->args[offset + (manual_set ? 1 : 2)], 15, "cross event ID"),
      Literal(call->args[offset + (manual_set ? 2 : 0)], 255, "cross mode"),
      Pipe(call->args[offset + (manual_set ? 0 : 1)])};
}

LegacyCrossCorePlan
PlanLegacyFlatCrossCore(const PrimFunc &function, const Target &target,
                        const std::string &platform,
                        const std::vector<LegacyOwnedTask> &tasks,
                        const ConflictOracle &physical_conflict_oracle,
                        const std::vector<LegacyOwnedControl> &controls) {
  const String model = target.defined()
                           ? target->GetAttr<String>("model").value_or(String())
                           : String();
  ICHECK(target.defined() && (model == "ascendc" || model == "auto") &&
         (platform == "A2" || platform == "A3"))
      << kDiagnostic << " currently supports A2/A3 AscendC only";
  const bool noalias =
      function->GetAttr<Bool>("tir.noalias").value_or(Bool(false));
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> fresh;
  for (const Var &storage : ReadTrustedFreshGMStorages(function))
    fresh.insert(storage);
  ICHECK(noalias || !fresh.empty())
      << kDiagnostic
      << " requires an explicit tir.noalias contract or a "
         "trusted Cython fresh-workspace contract";
  ICHECK(static_cast<bool>(physical_conflict_oracle))
      << kDiagnostic << " requires the physical conflict oracle";
  for (const char *attribute : {"buffer_versions", "tl.buffer_versions_map"}) {
    if (auto versions = function->GetAttr<Map<Var, PrimExpr>>(attribute)) {
      for (const auto &[storage, value] : versions.value()) {
        const auto *version = value.as<IntImmNode>();
        ICHECK(version != nullptr && version->value == 1)
            << kDiagnostic << " currently requires B1 storage: " << storage;
      }
    }
  }
  if (auto rings =
          function->GetAttr<PhysicalRingProvenance>(kPhysicalRingProvenance)) {
    ICHECK(rings.value().empty())
        << kDiagnostic << " does not support physical ring provenance";
  }
  FlatInputValidator validation(function, tasks, controls);
  validation.Run(function->body);
  ICHECK(noalias || validation.HasSingleWorkgroup())
      << kDiagnostic
      << " fresh-workspace alias support requires one static "
         "blockIdx.x workgroup";
  Map<Var, PrimExpr> cube_variables, vector_variables;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> private_variables;
  for (const Var &variable : validation.PrivateVariables()) {
    private_variables.insert(variable);
    cube_variables.Set(variable, variable.copy_with_suffix("_cross_c"));
    vector_variables.Set(variable, variable.copy_with_suffix("_cross_v"));
  }

  LegacyCrossCorePlan plan;
  const LegacyOwnedTask *set = nullptr, *wait = nullptr;
  int channel = -1;
  for (const LegacyOwnedTask &record : tasks) {
    auto flag = DecodeLegacyCrossFlag(record.task->stmt);
    if (!flag) {
      CollectLocalReservation(record, &plan);
      continue;
    }
    ICHECK(record.task->GetReadRegions().empty() &&
           record.task->GetWriteRegions().empty())
        << kDiagnostic << " requires real and shadow effects to stay separate";
    if (flag->kind == LegacyCrossFlagKind::kSet) {
      ICHECK(set == nullptr && record.owner == kCoreCube && flag->mode == 2 &&
             flag->pipe == "FIX")
          << kDiagnostic << " requires exactly one mode-2 FIX Set on C";
      set = &record;
      channel = flag->id;
    } else {
      ICHECK(wait == nullptr && record.owner == kCoreVector &&
             flag->pipe == "MTE2")
          << kDiagnostic
          << " requires exactly one unconditional Wait on V "
             "with MTE2 consumer intent";
      wait = &record;
    }
    plan.reservations.push_back({record.owner, true, "", flag->id});
  }
  ICHECK(set != nullptr && wait != nullptr &&
         DecodeLegacyCrossFlag(wait->task->stmt)->id == channel)
      << kDiagnostic << " requires one matching literal C->V channel";
  validation.CheckPublicationAfterPrefix(*set);

  std::vector<const LegacyOwnedTask *> cube, vector;
  for (const auto &record : tasks)
    (record.owner == kCoreCube ? cube : vector).push_back(&record);
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> shared;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> release_writes;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> acquire_reads;
  if (!noalias) {
    const LegacyOwnedTask *output_task = nullptr;
    Optional<BufferRegion> output_region;
    for (const auto *record : vector) {
      for (const BufferRegion &read : record->task->GetReadRegions()) {
        ICHECK(!IsGM(read) || fresh.count(read->buffer->data))
            << kDiagnostic
            << " fresh-workspace alias support permits V GM "
               "reads only from trusted workspaces";
      }
      for (const BufferRegion &write : record->task->GetWriteRegions()) {
        if (!IsGM(write))
          continue;
        ICHECK(!fresh.count(write->buffer->data) && output_task == nullptr &&
               record->order > wait->order)
            << kDiagnostic
            << " fresh-workspace alias support requires one "
               "external V GM write after Wait";
        output_task = record;
        output_region = write;
      }
    }
    ICHECK(output_task != nullptr)
        << kDiagnostic << " requires one V GM output";
    ValidateFreshWorkspaceOutput(function, *output_task, output_region.value(),
                                 validation);
  }
  auto instantiate_region = [&](const BufferRegion &region,
                                const Map<Var, PrimExpr> &substitution) {
    // Preserve Buffer identity: the physical oracle uses it to distinguish
    // opaque DMA views from exact scalar footprints. Private buffer-layout
    // expressions need an explicit oracle-side instantiation API, so reject
    // them here instead of cloning a Buffer and losing that classification.
    auto require_shared_layout = [&](const PrimExpr &value) {
      ICHECK(!UsesVar(value,
                      [&](const VarNode *var) {
                        return private_variables.count(GetRef<Var>(var));
                      }))
          << kDiagnostic << " requires core-independent buffer layout metadata";
    };
    require_shared_layout(region->buffer->elem_offset);
    for (const PrimExpr &value : region->buffer->shape)
      require_shared_layout(value);
    for (const PrimExpr &value : region->buffer->strides)
      require_shared_layout(value);
    Array<Range> ranges;
    for (const Range &range : region->region)
      ranges.push_back(
          Range::FromMinExtent(Substitute(range->min, substitution),
                               Substitute(range->extent, substitution)));
    return BufferRegion(region->buffer, ranges);
  };
  auto check = [&](const LegacyOwnedTask &c, const BufferRegion &c_region,
                   const LegacyOwnedTask &v, const BufferRegion &v_region,
                   bool publication_raw, bool publication_war = false) {
    // C/V streams execute independently. Even copied Let/snapshot variables
    // and the AIV coordinate cannot share a proof identity across executions.
    // Function parameters and the common workgroup id remain shared.
    const ConstrSet c_context = c.task->outer_ctx.Substitute(cube_variables);
    const ConstrSet v_context = v.task->outer_ctx.Substitute(vector_variables);
    const BufferRegion c_instance =
        instantiate_region(c_region, cube_variables);
    const BufferRegion v_instance =
        instantiate_region(v_region, vector_variables);
    const auto conflict =
        physical_conflict_oracle({c_context, c_instance, v_context, v_instance,
                                  nullptr, 0, 0, NullOpt, NullOpt});
    if (conflict.has_value() && !conflict.value())
      return;
    if (!noalias && publication_war && IsGM(c_region) && IsGM(v_region) &&
        !fresh.count(c_region->buffer->data) &&
        !fresh.count(v_region->buffer->data)) {
      ICHECK(c.order < set->order && v.order > wait->order)
          << kDiagnostic
          << " requires every C input read before Set and V "
             "output write after Wait";
      release_writes.insert(c_region->buffer->data);
      acquire_reads.insert(v_region->buffer->data);
      return;
    }
    ICHECK(publication_raw && IsGM(c_region) && IsGM(v_region) &&
           c_region->buffer->data.same_as(v_region->buffer->data))
        << kDiagnostic
        << " found an uncovered cross-resource RAW/WAR/WAW "
           "or unknown physical alias";
    ICHECK(c.order < set->order && v.order > wait->order)
        << kDiagnostic
        << " requires every C writer before Set and every V "
           "reader after Wait, without subsequent overwrite";
    shared.insert(c_region->buffer->data);
  };
  for (const auto *c : cube) {
    for (const auto *v : vector) {
      for (const BufferRegion &write : c->task->GetWriteRegions()) {
        for (const BufferRegion &read : v->task->GetReadRegions())
          check(*c, write, *v, read, true);
        for (const BufferRegion &other : v->task->GetWriteRegions())
          check(*c, write, *v, other, false);
      }
      for (const BufferRegion &read : c->task->GetReadRegions()) {
        for (const BufferRegion &write : v->task->GetWriteRegions())
          check(*c, read, *v, write, false, true);
      }
    }
  }
  ICHECK(!shared.empty()) << kDiagnostic
                          << " has no shared GM publication to validate";
  LegacyCrossShadowEffect release{
      set->task, PipeMask(ResourcePipe::kFixpipe), {}, {}};
  LegacyCrossShadowEffect acquire{
      wait->task, PipeMask(ResourcePipe::kScalar), {}, {}};
  // Full parameter regions are conservative fence effects. They are not real
  // reads/writes and must never be fed back into cross-conflict validation.
  // The local physical oracle remains responsible for opaque DMA footprints.
  size_t release_write_count = 0, acquire_read_count = 0;
  for (const auto &[parameter, buffer] : function->buffer_map) {
    if (!shared.count(buffer->data) && !release_writes.count(buffer->data) &&
        !acquire_reads.count(buffer->data))
      continue;
    Array<Range> ranges;
    for (const PrimExpr &extent : buffer->shape)
      ranges.push_back(Range::FromMinExtent(make_zero(extent.dtype()), extent));
    BufferRegion whole(buffer, ranges);
    if (shared.count(buffer->data)) {
      release.reads.push_back(whole);
      acquire.writes.push_back(whole);
      plan.shared_gm_storages.push_back(buffer->data);
    }
    if (release_writes.count(buffer->data)) {
      release.writes.push_back(whole);
      ++release_write_count;
    }
    if (acquire_reads.count(buffer->data)) {
      acquire.reads.push_back(whole);
      ++acquire_read_count;
    }
  }
  ICHECK_EQ(plan.shared_gm_storages.size(), shared.size())
      << kDiagnostic << " requires known parameter identities for shared GM";
  ICHECK(release_write_count == release_writes.size() &&
         acquire_read_count == acquire_reads.size())
      << kDiagnostic << " requires unique parameter identities for WAR fences";
  plan.shadows.push_back(std::move(release));
  plan.shadows.push_back(std::move(acquire));
  return plan;
}

} // namespace tvm::tl::ascend_sync
