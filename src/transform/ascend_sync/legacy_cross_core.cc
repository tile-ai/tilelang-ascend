// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

#include "legacy_cross_core.h"

#include <tvm/runtime/logging.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <tvm/ir/type.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "../common/pipeline_ring_provenance.h"
#include "legacy_integer_proof.h"
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

template <typename... Args> std::string FlatDiagnostic(Args &&...args) {
  std::ostringstream stream;
  (stream << ... << args);
  return stream.str();
}

bool RejectFlat(std::string *failure, std::string message) {
  if (failure->empty())
    *failure = std::move(message);
  return false;
}

class FlatInputValidator : public StmtExprVisitor {
public:
  FlatInputValidator(const PrimFunc &function,
                     const std::vector<LegacyOwnedTask> &tasks,
                     const std::vector<LegacyOwnedControl> &controls,
                     bool closed = false)
      : closed_(closed) {
    if (auto ratio = function->GetAttr<ObjectRef>("npu_cv_ratio")) {
      const auto *literal = ratio.value().as<StringImmNode>();
      if (!(literal != nullptr && literal->value == "cv_1_2")) {
        Reject(kDiagnostic,
               " requires npu_cv_ratio to be StringImm cv_1_2 when present");
        return;
      }
      developer_ratio_ = true;
    }
    for (const auto &task : tasks) {
      if (!(task.task != nullptr && task.original_statement.defined())) {
        Reject(kDiagnostic, " requires complete owned-task records");
        return;
      }
      if (!(by_order_.emplace(task.order, &task).second)) {
        Reject(kDiagnostic, " has duplicate statement occurrence records");
        return;
      }
    }
    if (!(closed_ || controls.size() <= 1U)) {
      Reject(kDiagnostic,
             " requires flat input except one static C-only prefix For");
      return;
    }
    for (const auto &control : controls) {
      if (!(control.control != nullptr && control.control->task != nullptr &&
            (control.owner == kCoreCube ||
             (closed_ && control.owner == kCoreVector)) &&
            control.control->task->GetCoreMask() == control.owner &&
            controls_.emplace(control.order, &control).second)) {
        Reject(
            kDiagnostic,
            " requires flat input with complete C-only prefix control records");
        return;
      }
    }
  }

  void Run(const Stmt &body) {
    if (!IsValid())
      return;
    VisitStmt(body);
    if (!IsValid())
      return;
    if (seen_ != by_order_.size()) {
      Reject(kDiagnostic, " task records do not match the original function");
      return;
    }
    if (seen_controls_ != controls_.size()) {
      Reject(kDiagnostic,
             " control records do not match the original function");
      return;
    }
    if (vector_topologies_ != 1) {
      Reject(kDiagnostic, " requires one unambiguous C1/V2 subgroup topology");
      return;
    }
  }

  bool IsValid() const { return failure_.empty(); }
  const std::string &Failure() const { return failure_; }

  const std::vector<Var> &PrivateVariables() const { return private_vars_; }
  Var Subcore() const { return subcore_.value(); }
  template <typename Diagnostic>
  Var Workgroup(const Diagnostic &diagnostic) const {
    ICHECK(workgroup_.defined() && workgroups_.size() == 1)
        << diagnostic(" requires one blockIdx.x workgroup axis");
    return workgroup_.value();
  }
  struct Allocation {
    DataType dtype;
    int64_t elements;
  };
  const auto &Allocations() const { return allocations_; }
  bool HasSingleWorkgroup() const {
    return workgroups_.size() == 1 && is_one(workgroups_[0]);
  }

  void CheckPublicationAfterPrefix(const LegacyOwnedTask &set) {
    for (const auto &[begin, end] : control_spans_)
      if (end >= set.order) {
        Reject(kDiagnostic, " requires the complete C-only prefix For before "
                            "the unique outside-loop Set");
        return;
      }
  }

private:
  template <typename... Args> void Reject(Args &&...args) {
    RejectFlat(&failure_, FlatDiagnostic(args...));
  }

  void VisitStmt(const Stmt &statement) final {
    if (!IsValid())
      return;
    ++occurrence_;
    if (!(statement.as<SeqStmtNode>() || statement.as<AttrStmtNode>() ||
          statement.as<AllocateNode>() || statement.as<DeclBufferNode>() ||
          statement.as<LetStmtNode>() || statement.as<EvaluateNode>() ||
          statement.as<BufferStoreNode>() || statement.as<ForNode>())) {
      Reject(kDiagnostic,
             " requires flat input without For/If/While or "
             "unlowered control: ",
             statement->GetTypeKey());
      return;
    }
    StmtExprVisitor::VisitStmt(statement);
  }

  void CheckTask(const Stmt &statement, const LetStmtNode *binding = nullptr) {
    auto found = by_order_.find(occurrence_);
    if (found == by_order_.end()) {
      Reject(kDiagnostic, " is missing a real effect/statement occurrence");
      return;
    }
    const LegacyOwnedTask &record = *found->second;
    if (!(record.original_statement.same_as(statement))) {
      Reject(kDiagnostic, " occurrence has the wrong original statement");
      return;
    }
    if (!((owner_ == kCoreCube || owner_ == kCoreVector) &&
          record.owner == owner_ && record.task->GetCoreMask() == owner_)) {
      Reject(kDiagnostic, " requires an explicit, unsplit C or V owner");
      return;
    }
    if (record.task->HasConditions()) {
      Reject(kDiagnostic, " does not support subcore/conditional predicates");
      return;
    }
    if (inside_c_loop_) {
      if (!((closed_ || record.owner == kCoreCube) &&
            !DecodeLegacyCrossFlag(record.task->stmt).has_value())) {
        Reject(kDiagnostic,
               " requires all cross flags outside the C prefix For");
        return;
      }
      for (const BufferRegion &region : record.task->GetWriteRegions())
        if (!(closed_ || !IsGM(region))) {
          Reject(kDiagnostic, " C-only prefix For may read GM but cannot "
                              "publish or overwrite GM inside the loop");
          return;
        }
    }
    if (binding != nullptr) {
      const auto *definition = record.task->stmt.as<EvaluateNode>();
      if (!(definition != nullptr &&
            definition->value.same_as(binding->value) &&
            record.task->GetWriteVars().size() == 1 &&
            record.task->GetWriteVars()[0].same_as(binding->var))) {
        Reject(kDiagnostic, " has an invalid snapshot definition record");
        return;
      }
    } else {
      if (!(record.task->stmt.same_as(statement))) {
        Reject(kDiagnostic, " must retain original task statements");
        return;
      }
    }
    ++seen_;
  }

  void VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    const auto *operation = call ? call->op.as<OpNode>() : nullptr;
    const bool assumption =
        closed_ && operation && operation->name == "tir.assume";
    if (!op->value.as<IntImmNode>() && !assumption)
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
    if (!(found != controls_.end() && (closed_ || !inside_c_loop_) &&
          found->second->owner == owner_ &&
          found->second->original_statement.same_as(GetRef<Stmt>(op)) &&
          found->second->control->control.same_as(GetRef<For>(op)))) {
      Reject(kDiagnostic, " requires flat input except one certified C-only "
                          "prefix For; V or nested controls are unsupported");
      return;
    }
    const auto *minimum = op->min.as<IntImmNode>();
    const auto *extent = op->extent.as<IntImmNode>();
    if (!((op->kind == ForKind::kSerial ||
           (closed_ && op->kind == ForKind::kUnrolled)) &&
          minimum && minimum->value == 0 && extent && extent->value > 0 &&
          op->annotations.empty() && !op->thread_binding.defined())) {
      Reject(kDiagnostic, " requires a static positive serial C-only prefix "
                          "For with min=0");
      return;
    }
    const bool previous_loop = inside_c_loop_;
    inside_c_loop_ = true;
    private_vars_.push_back(op->loop_var);
    StmtExprVisitor::VisitStmt_(op);
    inside_c_loop_ = previous_loop;
    control_spans_.push_back({begin, occurrence_});
    ++seen_controls_;
  }

  void VisitStmt_(const AllocateNode *op) final {
    if (!is_one(op->condition)) {
      Reject(kDiagnostic, " does not support predicated allocation");
      return;
    }
    if (closed_) {
      int64_t elements = 1;
      for (const PrimExpr &extent : op->extents) {
        const auto *literal = extent.as<IntImmNode>();
        if (!(literal && literal->value > 0 &&
              elements <=
                  std::numeric_limits<int64_t>::max() / literal->value)) {
          Reject(kDiagnostic,
                 " requires static nonoverflowing local allocations");
          return;
        }
        elements *= literal->value;
      }
      if (!(allocations_
                .emplace(op->buffer_var, Allocation{op->dtype, elements})
                .second)) {
        Reject(kDiagnostic, " has ambiguous local allocation roots");
        return;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    const CoreMask previous = owner_;
    if (op->attr_key == "resource_scope") {
      const auto *resource_value = op->value.as<IntImmNode>();
      if (!resource_value || resource_value->value < 0 ||
          resource_value->value > 1) {
        Reject(kDiagnostic, " requires a literal resource scope in [0, 1]");
        return;
      }
      const int resource = static_cast<int>(resource_value->value);
      owner_ = resource == 0 ? kCoreCube : kCoreVector;
    } else if (op->attr_key == "thread_extent") {
      const auto *iteration = op->node.as<IterVarNode>();
      if (iteration == nullptr) {
        Reject(kDiagnostic, " has malformed thread topology");
        return;
      }
      const bool expert = iteration->thread_tag == "blockIdx.y";
      const bool developer = iteration->thread_tag == "threadIdx.x";
      if (expert || developer) {
        // The compiler-managed Developer path uses threadIdx.x after
        // VidReduction. A bare threadIdx.x is not sufficient evidence of the
        // C1/V2 mapping: preserve the kernel's explicit ratio contract.
        if (!(!developer || developer_ratio_)) {
          Reject(kDiagnostic,
                 " requires npu_cv_ratio=StringImm cv_1_2 for threadIdx.x");
          return;
        }
        if (!((op->value.as<IntImmNode>() &&
               op->value.as<IntImmNode>()->value == 2))) {
          Reject(kDiagnostic, " requires C1/V2 topology");
          return;
        }
        ++vector_topologies_;
        if (vector_topologies_ != 1) {
          Reject(kDiagnostic,
                 " requires one unambiguous C1/V2 subgroup topology");
          return;
        }
        subcore_ = iteration->var;
        private_vars_.push_back(iteration->var);
      } else {
        if (!(iteration->thread_tag == "blockIdx.x")) {
          Reject(kDiagnostic, " has an unsupported thread topology");
          return;
        }
        workgroups_.push_back(op->value);
        if (closed_) {
          if (workgroup_.defined()) {
            Reject(kDiagnostic, " has duplicate workgroup axes");
            return;
          }
          workgroup_ = iteration->var;
          private_vars_.push_back(iteration->var);
        }
      }
    } else if (op->attr_key == "virtual_thread") {
      Reject(kDiagnostic, " does not support virtual thread topology");
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
    owner_ = previous;
  }

  void VisitExpr(const PrimExpr &expression) final {
    if (!IsValid())
      return;
    if (subcore_.defined() && expression.dtype().is_bool()) {
      if (!(!UsesVar(expression, [&](const VarNode *var) {
            return subcore_.value().same_as(GetRef<Var>(var));
          }))) {
        Reject(kDiagnostic, " does not support a subcore predicate");
        return;
      }
    }
    StmtExprVisitor::VisitExpr(expression);
  }

  std::map<size_t, const LegacyOwnedTask *> by_order_;
  std::map<size_t, const LegacyOwnedControl *> controls_;
  std::vector<std::pair<size_t, size_t>> control_spans_;
  Optional<Var> subcore_;
  Optional<Var> workgroup_;
  std::unordered_map<Var, Allocation, ObjectPtrHash, ObjectPtrEqual>
      allocations_;
  std::vector<Var> private_vars_;
  std::vector<PrimExpr> workgroups_;
  CoreMask owner_{kCoreUnassigned};
  size_t occurrence_{0}, seen_{0};
  size_t seen_controls_{0};
  int vector_topologies_{0};
  bool developer_ratio_{false};
  bool inside_c_loop_{false};
  bool closed_{false};
  std::string failure_;
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
bool ValidateFreshWorkspaceOutput(const PrimFunc &function,
                                  const LegacyOwnedTask &record,
                                  const BufferRegion &output,
                                  const FlatInputValidator &validation,
                                  std::string *failure) {
  auto reject_output = [&](auto &&...parts) {
    return RejectFlat(failure, FlatDiagnostic(parts...));
  };
  const char *reject =
      "Trusted fresh-workspace C/V requires one full dense copy_ub_to_gm "
      "output with provably disjoint AIV footprints";
  const auto *evaluate = record.task->stmt.as<EvaluateNode>();
  const auto *call = evaluate ? evaluate->value.as<CallNode>() : nullptr;
  if (call == nullptr) {
    return reject_output(reject);
  }
  size_t offset = 0;
  const std::string name = Name(call, &offset);
  const std::string prefix = "tl::ascend::copy_ub_to_gm<";
  if (!(name.rfind(prefix, 0) == 0 && name.back() == '>')) {
    return reject_output(reject);
  }
  std::vector<std::string> tokens;
  size_t position = prefix.size();
  while (position < name.size() - 1) {
    const size_t comma = name.find(',', position);
    const size_t end = comma == std::string::npos ? name.size() - 1 : comma;
    if (!(end < name.size() && end >= position)) {
      return reject_output(reject);
    }
    std::string token = name.substr(position, end - position);
    const size_t first = token.find_first_not_of(" \t\r\n");
    const size_t last = token.find_last_not_of(" \t\r\n");
    if (first == std::string::npos) {
      return reject_output(reject);
    }
    tokens.push_back(token.substr(first, last - first + 1));
    position = end + 1;
  }
  if (!(tokens.size() == 2 || tokens.size() == 3)) {
    return reject_output(reject);
  }
  const DataType dtype = output->buffer->dtype;
  if (!(dtype.is_scalar() &&
        ((tokens[0] == "half" && dtype == DataType::Float(16)) ||
         (tokens[0] == "float" && dtype == DataType::Float(32))))) {
    reject_output(reject, "; unsupported DMA element type");
    return false;
  }
  auto dimension = [&](size_t index) -> int64_t {
    uint64_t value = 0;
    const std::string &token = tokens[index];
    const auto parsed =
        std::from_chars(token.data(), token.data() + token.size(), value);
    if (!(parsed.ec == std::errc() &&
          parsed.ptr == token.data() + token.size() && value > 0 &&
          value <= std::numeric_limits<uint32_t>::max())) {
      reject_output(reject);
      return 0;
    }
    return static_cast<int64_t>(value);
  };
  const int64_t columns = dimension(1);
  const int64_t rows = tokens.size() == 3 ? dimension(2) : 1;
  if (!failure->empty())
    return false;
  if (!(columns <= std::numeric_limits<int64_t>::max() / rows)) {
    return reject_output(reject);
  }
  const int64_t volume = rows * columns;
  if (!(columns <= std::numeric_limits<int64_t>::max() / dtype.bytes() &&
        columns * dtype.bytes() % 32 == 0 && rows <= 4095 &&
        columns * dtype.bytes() <= 65535)) {
    reject_output(reject, "; unsupported DMA row alignment or hardware field");
    return false;
  }
  // AscendCopy can append two shape-metadata arguments. CopyCodegen forwards
  // only the first three scalar arguments to common.h's DMA helper.
  if (!(call->args.size() >= offset + 2 && call->args.size() <= offset + 7)) {
    return reject_output(reject);
  }
  const auto *pointer = call->args[offset + 1].as<CallNode>();
  if (!(pointer != nullptr && pointer->op.same_as(builtin::tvm_access_ptr()) &&
        pointer->args.size() == 5 &&
        pointer->args[1].same_as(output->buffer->data))) {
    reject_output(reject, "; output pointer identity mismatch");
    return false;
  }
  ProofAnalyzer mask_analyzer;
  record.task->outer_ctx.Populate(mask_analyzer);
  for (const auto &[index, expected] :
       {std::pair<size_t, int64_t>{offset + 2, columns},
        {offset + 3, rows},
        {offset + 4, columns}}) {
    if (index >= call->args.size()) {
      // C++ realdstN defaults to one; mask defaults use template dimensions.
      if (!(index != offset + 2 || columns == 1)) {
        return reject_output(reject);
      }
      continue;
    }
    const PrimExpr &argument = call->args[index];
    if (!(argument.dtype().is_scalar() &&
          ((argument.dtype().is_int() && argument.dtype().bits() <= 64) ||
           (argument.dtype().is_uint() && argument.dtype().bits() <= 32)) &&
          SideEffect(argument) <= CallEffectKind::kPure &&
          mask_analyzer.CanProve(cast(DataType::Int(64), argument) ==
                                 make_const(DataType::Int(64), expected)))) {
      reject_output(reject, "; unsupported stride or mask");
      return false;
    }
  }
  Optional<Buffer> parameter_buffer;
  for (const auto &[parameter, buffer] : function->buffer_map) {
    if (buffer->data.same_as(output->buffer->data)) {
      if (parameter_buffer.defined()) {
        return reject_output(reject);
      }
      parameter_buffer = buffer;
    }
  }
  if (!(parameter_buffer.defined() &&
        is_zero(parameter_buffer.value()->elem_offset) &&
        parameter_buffer.value()->strides.empty())) {
    reject_output(reject, "; requires a dense GM parameter");
    return false;
  }
  int64_t elements = 1;
  for (const PrimExpr &extent : parameter_buffer.value()->shape) {
    const auto *constant = extent.as<IntImmNode>();
    if (!(constant != nullptr && constant->value > 0 &&
          elements <= std::numeric_limits<int64_t>::max() / constant->value)) {
      reject_output(reject, "; requires static output capacity");
      return false;
    }
    elements *= constant->value;
  }
  const PrimExpr original = pointer->args[2];
  if (!(original.dtype().is_scalar() &&
        ((original.dtype().is_int() && original.dtype().bits() <= 64) ||
         (original.dtype().is_uint() && original.dtype().bits() <= 32)) &&
        SideEffect(original) <= CallEffectKind::kPure)) {
    return reject_output(reject);
  }
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
    if (!(footprints.CanProve(starts[instance] >= 0) &&
          footprints.CanProve(
              starts[instance] <=
              make_const(DataType::Int(64), elements - volume)))) {
      reject_output(reject, "; output DMA exceeds its parameter allocation");
      return false;
    }
  }
  const PrimExpr span = make_const(DataType::Int(64), volume);
  if (!(footprints.CanProve(starts[0] <= starts[1] - span) ||
        footprints.CanProve(starts[1] <= starts[0] - span))) {
    reject_output(reject, "; AIV output footprints may overlap");
    return false;
  }
  return true;
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

namespace {

bool TryPlanLegacyFlatCrossCore(const PrimFunc &function, const Target &target,
                                const std::string &platform,
                                const std::vector<LegacyOwnedTask> &tasks,
                                const ConflictOracle &physical_conflict_oracle,
                                const std::vector<LegacyOwnedControl> &controls,
                                LegacyCrossCorePlan *accepted_plan,
                                std::string *failure) {
  auto reject_plan = [&](auto &&...parts) {
    return RejectFlat(failure, FlatDiagnostic(parts...));
  };
  const String model = target.defined()
                           ? target->GetAttr<String>("model").value_or(String())
                           : String();
  if (!(target.defined() && (model == "ascendc" || model == "auto") &&
        (platform == "A2" || platform == "A3"))) {
    return reject_plan(kDiagnostic, " currently supports A2/A3 AscendC only");
  }
  const bool noalias =
      function->GetAttr<Bool>("tir.noalias").value_or(Bool(false));
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> fresh;
  for (const Var &storage : ReadTrustedAutoGMStorages(function))
    fresh.insert(storage);
  if (!(noalias || !fresh.empty())) {
    return reject_plan(kDiagnostic,
                       " requires an explicit tir.noalias contract or a "
                       "trusted Cython fresh-workspace contract");
  }
  if (!(static_cast<bool>(physical_conflict_oracle))) {
    return reject_plan(kDiagnostic, " requires the physical conflict oracle");
  }
  for (const char *attribute : {"buffer_versions", "tl.buffer_versions_map"}) {
    if (auto versions = function->GetAttr<Map<Var, PrimExpr>>(attribute)) {
      for (const auto &[storage, value] : versions.value()) {
        const auto *version = value.as<IntImmNode>();
        if (!(version != nullptr && version->value == 1)) {
          return reject_plan(kDiagnostic,
                             " currently requires B1 storage: ", storage);
        }
      }
    }
  }
  if (auto rings =
          function->GetAttr<PhysicalRingProvenance>(kPhysicalRingProvenance)) {
    if (!(rings.value().empty())) {
      return reject_plan(kDiagnostic,
                         " does not support physical ring provenance");
    }
  }
  FlatInputValidator validation(function, tasks, controls);
  validation.Run(function->body);
  if (!validation.IsValid())
    return reject_plan(validation.Failure());
  if (!(noalias || validation.HasSingleWorkgroup())) {
    return reject_plan(kDiagnostic,
                       " fresh-workspace alias support requires one static "
                       "blockIdx.x workgroup");
  }
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
    if (!(record.task->GetReadRegions().empty() &&
          record.task->GetWriteRegions().empty())) {
      return reject_plan(kDiagnostic,
                         " requires real and shadow effects to stay separate");
    }
    if (flag->kind == LegacyCrossFlagKind::kSet) {
      if (!(set == nullptr && record.owner == kCoreCube && flag->mode == 2 &&
            flag->pipe == "FIX")) {
        return reject_plan(kDiagnostic,
                           " requires exactly one mode-2 FIX Set on C");
      }
      set = &record;
      channel = flag->id;
    } else {
      if (!(wait == nullptr && record.owner == kCoreVector &&
            flag->pipe == "MTE2")) {
        return reject_plan(kDiagnostic,
                           " requires exactly one unconditional Wait on V "
                           "with MTE2 consumer intent");
      }
      wait = &record;
    }
    plan.reservations.push_back({record.owner, true, "", flag->id});
  }
  if (!(set != nullptr && wait != nullptr &&
        DecodeLegacyCrossFlag(wait->task->stmt)->id == channel)) {
    return reject_plan(kDiagnostic,
                       " requires one matching literal C->V channel");
  }
  validation.CheckPublicationAfterPrefix(*set);
  if (!validation.IsValid())
    return reject_plan(validation.Failure());

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
        if (!(!IsGM(read) || fresh.count(read->buffer->data))) {
          return reject_plan(kDiagnostic,
                             " fresh-workspace alias support permits V GM "
                             "reads only from trusted workspaces");
        }
      }
      for (const BufferRegion &write : record->task->GetWriteRegions()) {
        if (!IsGM(write))
          continue;
        if (!(!fresh.count(write->buffer->data) && output_task == nullptr &&
              record->order > wait->order)) {
          return reject_plan(kDiagnostic,
                             " fresh-workspace alias support requires one "
                             "external V GM write after Wait");
        }
        output_task = record;
        output_region = write;
      }
    }
    if (output_task == nullptr) {
      return reject_plan(kDiagnostic, " requires one V GM output");
    }
    if (!ValidateFreshWorkspaceOutput(
            function, *output_task, output_region.value(), validation, failure))
      return false;
  }
  auto instantiate_region = [&](const BufferRegion &region,
                                const Map<Var, PrimExpr> &substitution) {
    // Preserve Buffer identity: the physical oracle uses it to distinguish
    // opaque DMA views from exact scalar footprints. Private buffer-layout
    // expressions need an explicit oracle-side instantiation API, so reject
    // them here instead of cloning a Buffer and losing that classification.
    auto require_shared_layout = [&](const PrimExpr &value) {
      if (!(!UsesVar(value, [&](const VarNode *var) {
            return private_variables.count(GetRef<Var>(var));
          }))) {
        reject_plan(kDiagnostic,
                    " requires core-independent buffer layout metadata");
        return;
      }
    };
    require_shared_layout(region->buffer->elem_offset);
    for (const PrimExpr &value : region->buffer->shape)
      require_shared_layout(value);
    for (const PrimExpr &value : region->buffer->strides)
      require_shared_layout(value);
    if (!failure->empty())
      return BufferRegion();
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
    if (!failure->empty())
      return;
    // C/V streams execute independently. Even copied Let/snapshot variables
    // and the AIV coordinate cannot share a proof identity across executions.
    // Function parameters and the common workgroup id remain shared.
    const ConstrSet c_context = c.task->outer_ctx.Substitute(cube_variables);
    const ConstrSet v_context = v.task->outer_ctx.Substitute(vector_variables);
    const BufferRegion c_instance =
        instantiate_region(c_region, cube_variables);
    const BufferRegion v_instance =
        instantiate_region(v_region, vector_variables);
    if (!failure->empty())
      return;
    const auto conflict =
        physical_conflict_oracle({c_context, c_instance, v_context, v_instance,
                                  nullptr, 0, 0, NullOpt, NullOpt});
    if (conflict.has_value() && !conflict.value())
      return;
    if (!noalias && publication_war && IsGM(c_region) && IsGM(v_region) &&
        !fresh.count(c_region->buffer->data) &&
        !fresh.count(v_region->buffer->data)) {
      if (!(c.order < set->order && v.order > wait->order)) {
        reject_plan(kDiagnostic,
                    " requires every C input read before Set and V "
                    "output write after Wait");
        return;
      }
      release_writes.insert(c_region->buffer->data);
      acquire_reads.insert(v_region->buffer->data);
      return;
    }
    if (!(publication_raw && IsGM(c_region) && IsGM(v_region) &&
          c_region->buffer->data.same_as(v_region->buffer->data))) {
      reject_plan(kDiagnostic, " found an uncovered cross-resource RAW/WAR/WAW "
                               "or unknown physical alias");
      return;
    }
    if (!(c.order < set->order && v.order > wait->order)) {
      reject_plan(kDiagnostic,
                  " requires every C writer before Set and every V "
                  "reader after Wait, without subsequent overwrite");
      return;
    }
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
  if (!failure->empty())
    return false;
  if (shared.empty()) {
    return reject_plan(kDiagnostic,
                       " has no shared GM publication to validate");
  }
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
  if (plan.shared_gm_storages.size() != shared.size()) {
    return reject_plan(kDiagnostic,
                       " requires known parameter identities for shared GM");
  }
  if (!(release_write_count == release_writes.size() &&
        acquire_read_count == acquire_reads.size())) {
    return reject_plan(kDiagnostic,
                       " requires unique parameter identities for WAR fences");
  }
  plan.shadows.push_back(std::move(release));
  plan.shadows.push_back(std::move(acquire));
  *accepted_plan = std::move(plan);
  return true;
}

} // namespace

LegacyFlatEligibility
AnalyzeLegacyFlatEligibility(const PrimFunc &function, const Target &target,
                             const std::string &platform,
                             const std::vector<LegacyOwnedTask> &tasks,
                             const ConflictOracle &compiler_added_oracle,
                             const std::vector<LegacyOwnedControl> &controls,
                             bool requires_generic_composite) {
  LegacyFlatEligibility result;
  // Validate the complete allocation recipe, including explicit roles that
  // this proof deliberately does not consume, even on a structural miss.
  ReadTrustedAutoGMStorages(function);
  if (requires_generic_composite) {
    result.rejection_ = FlatDiagnostic(
        kDiagnostic, " requires a closed proof for general GEMM invocations");
    return result;
  }
  LegacyCrossCorePlan plan;
  if (TryPlanLegacyFlatCrossCore(function, target, platform, tasks,
                                 compiler_added_oracle, controls, &plan,
                                 &result.rejection_))
    result.plan_ = std::move(plan);
  return result;
}

LegacyCrossCorePlan LegacyFlatEligibility::TakePlan() {
  ICHECK(IsEligible()) << rejection_;
  LegacyCrossCorePlan result = std::move(plan_.value());
  plan_.reset();
  return result;
}

namespace {

constexpr const char *kClosedDiagnostic = "Legacy closed C/V publication";

// Diagnostic state is local to one mandatory closed proof. It cannot change
// eligibility, authorize facts, or alter a failure into a retry.
class ClosedDiagnostic {
public:
  explicit ClosedDiagnostic(std::string flat_rejection)
      : flat_rejection_(std::move(flat_rejection)) {}

  template <typename... Args> std::string operator()(Args &&...parts) const {
    std::ostringstream stream;
    stream << kClosedDiagnostic;
    (stream << ... << parts);
    if (!flat_rejection_.empty())
      stream << "\nLegacy flat profile unavailable: " << flat_rejection_;
    return stream.str();
  }

private:
  std::string flat_rejection_;
};

PrimExpr ClosedI64(int64_t value) { return IntImm(DataType::Int(64), value); }

void PopulateClosedContext(const ConstrSet &context, ProofAnalyzer *analyzer,
                           const ClosedDiagnostic &diagnostic) {
  const char *failure = "invalid integer context";
  ICHECK(TryPopulateLegacyContext(context, analyzer, &failure))
      << diagnostic(": ", failure);
}

PrimExpr ClosedInteger(const PrimExpr &value, ProofAnalyzer *analyzer,
                       const ClosedDiagnostic &diagnostic) {
  const char *failure = "invalid integer operand";
  ICHECK(TryLegacyInteger(value, analyzer, &failure))
      << diagnostic(": ", failure);
  return cast(DataType::Int(64), value);
}

std::vector<std::string> ClosedTemplate(const std::string &name,
                                        const std::string &prefix,
                                        const ClosedDiagnostic &diagnostic) {
  ICHECK(name.rfind(prefix + "<", 0) == 0 && name.back() == '>')
      << diagnostic(" requires a canonical typed DMA helper");
  std::vector<std::string> result;
  size_t position = prefix.size() + 1;
  while (position < name.size() - 1) {
    size_t end = name.find(',', position);
    if (end == std::string::npos)
      end = name.size() - 1;
    std::string token = name.substr(position, end - position);
    token.erase(std::remove_if(token.begin(), token.end(),
                               [](unsigned char c) { return std::isspace(c); }),
                token.end());
    ICHECK(!token.empty()) << diagnostic(" has an empty template token");
    result.push_back(std::move(token));
    position = end + 1;
  }
  return result;
}

int64_t ClosedDimension(const std::string &token,
                        const ClosedDiagnostic &diagnostic) {
  uint64_t result = 0;
  const auto parsed =
      std::from_chars(token.data(), token.data() + token.size(), result);
  ICHECK(parsed.ec == std::errc() &&
         parsed.ptr == token.data() + token.size() && result > 0 &&
         result <= std::numeric_limits<uint32_t>::max())
      << diagnostic(" requires positive uint32 DMA template dimensions");
  return static_cast<int64_t>(result);
}

DataType ClosedType(const std::string &name,
                    const ClosedDiagnostic &diagnostic) {
  if (name == "half")
    return DataType::Float(16);
  if (name == "float")
    return DataType::Float(32);
  ICHECK(false) << diagnostic(" has an unsupported DMA element type: ", name);
  return DataType::Void();
}

// This is the actual helper footprint, not tvm_access_ptr's conservative
// extent.
struct ClosedGMRectangle {
  const LegacyOwnedTask *record;
  Buffer buffer;
  PrimExpr offset, stride;
  int64_t rows, columns, width;
  bool write;
};

class ClosedDMAAudit {
public:
  ClosedDMAAudit(const PrimFunc &function, const FlatInputValidator &validation,
                 const ClosedDiagnostic &diagnostic)
      : function_(function), validation_(validation), diagnostic_(diagnostic) {}

  ClosedGMRectangle Run(const LegacyOwnedTask &record,
                        const BufferRegion &region, bool write) const {
    const auto &diagnostic = diagnostic_;
    const auto *evaluate = record.task->stmt.as<EvaluateNode>();
    const auto *call = evaluate ? evaluate->value.as<CallNode>() : nullptr;
    ICHECK(call && call->op.same_as(builtin::call_extern()))
        << diagnostic(" requires canonical typed DMA for each real GM effect");
    size_t argument_offset = 0;
    const std::string name = Name(call, &argument_offset);
    const bool to_l1 = name.rfind("tl::ascend::copy_gm_to_l1<", 0) == 0;
    const bool fix = name.rfind("tl::ascend::copy_l0c_to_gm<", 0) == 0;
    const bool to_ub = name.rfind("tl::ascend::copy_gm_to_ub<", 0) == 0;
    const bool from_ub = name.rfind("tl::ascend::copy_ub_to_gm<", 0) == 0;
    ICHECK(to_l1 || fix || to_ub || from_ub)
        << diagnostic(" has an unsupported GM DMA helper");
    ICHECK_EQ(write, fix || from_ub) << diagnostic(
        " helper direction disagrees with its imported GM effect");
    ICHECK((to_l1 || fix) ? record.owner == kCoreCube
                          : record.owner == kCoreVector)
        << diagnostic(" DMA helper has the wrong execution owner");
    const std::string prefix = to_l1   ? "tl::ascend::copy_gm_to_l1"
                               : fix   ? "tl::ascend::copy_l0c_to_gm"
                               : to_ub ? "tl::ascend::copy_gm_to_ub"
                                       : "tl::ascend::copy_ub_to_gm";
    const auto tokens = ClosedTemplate(name, prefix, diagnostic);
    ICHECK(fix ? tokens.size() == 6
               : (to_l1 ? tokens.size() == 3
                        : tokens.size() == 2 || tokens.size() == 3))
        << diagnostic(" has an unsupported DMA template ABI");
    const DataType dtype = ClosedType(tokens[fix ? 1 : 0], diagnostic);
    const DataType local_dtype =
        fix ? ClosedType(tokens[0], diagnostic) : dtype;
    const int64_t rows = to_l1 ? ClosedDimension(tokens[1], diagnostic)
                         : fix ? ClosedDimension(tokens[3], diagnostic)
                         : tokens.size() == 3
                             ? ClosedDimension(tokens[2], diagnostic)
                             : 1;
    const int64_t columns = to_l1 ? ClosedDimension(tokens[2], diagnostic)
                            : fix ? ClosedDimension(tokens[4], diagnostic)
                                  : ClosedDimension(tokens[1], diagnostic);
    ICHECK(!fix || (tokens[2] == "layout::RowMajor" && tokens[5] == "0" &&
                    local_dtype == DataType::Float(32)))
        << diagnostic(
               " requires standalone RowMajor FIX without conversion flags");
    ICHECK(rows <= 4095 && columns <= 65535 / dtype.bytes() &&
           columns * dtype.bytes() % 32 == 0 &&
           rows <= std::numeric_limits<int64_t>::max() / columns)
        << diagnostic(" DMA row/count exceeds the admitted hardware fields");
    // Full aligned L1/L0 tiles have precisely the dense physical footprint.
    // Partial or padded NZ tiles need a separate semantic descriptor.
    ICHECK(!(to_l1 || fix) || (rows % 16 == 0 && columns % 16 == 0))
        << diagnostic(" requires complete aligned L1/L0 DMA tiles");
    const int64_t volume = rows * columns;
    ICHECK(call->args.size() >= 6 && call->args.size() <= (to_ub ? 9U : 10U))
        << diagnostic(" has an unsupported DMA scalar ABI");
    const auto *gm = call->args[write ? 2 : 1].as<CallNode>();
    const auto *local = call->args[write ? 1 : 2].as<CallNode>();
    auto pointer = [&](const CallNode *value, DataType element, int access) {
      ICHECK(value && value->op.same_as(builtin::tvm_access_ptr()) &&
             value->args.size() == 5 && value->args[0].dtype() == element &&
             value->args[1].as<VarNode>())
          << diagnostic(" has a malformed or retyped DMA pointer");
      const auto *annotation = value->args[0].as<CallNode>();
      const auto *mask = value->args[4].as<IntImmNode>();
      ICHECK(annotation &&
             annotation->op.same_as(Op::Get("tir.type_annotation")) && mask &&
             mask->value == access)
          << diagnostic(" has an invalid DMA pointer annotation/access mask");
    };
    pointer(gm, dtype, write ? 2 : 1);
    pointer(local, local_dtype, write ? 1 : 2);
    ICHECK(gm->args[1].same_as(region->buffer->data)) << diagnostic(
        " GM effect does not retain the actual DMA root identity");
    Optional<Buffer> parameter;
    for (const auto &[key, buffer] : function_->buffer_map)
      if (buffer->data.same_as(gm->args[1])) {
        ICHECK(!parameter.defined()) << diagnostic(" has ambiguous GM roots");
        parameter = buffer;
      }
    ICHECK(parameter.defined() && parameter.value()->dtype == dtype &&
           parameter.value()->strides.empty() &&
           is_zero(parameter.value()->elem_offset) &&
           !parameter.value()->shape.empty() &&
           (parameter.value().scope().empty() ||
            parameter.value().scope() == "global"))
        << diagnostic(" requires dense, untyped-view-free GM parameter roots");
    int64_t capacity = 1, width = 0;
    for (const PrimExpr &dimension : parameter.value()->shape) {
      const auto *literal = dimension.as<IntImmNode>();
      ICHECK(literal && literal->value > 0 &&
             capacity <= std::numeric_limits<int64_t>::max() / literal->value)
          << diagnostic(" requires complete nonoverflowing static GM capacity");
      capacity *= literal->value;
      width = literal->value;
    }
    ICHECK(capacity <= std::numeric_limits<int64_t>::max() / dtype.bytes())
        << diagnostic(" GM byte capacity overflows int64");
    ProofAnalyzer analyzer;
    PopulateClosedContext(record.task->outer_ctx, &analyzer, diagnostic);
    const PrimExpr offset = ClosedInteger(gm->args[2], &analyzer, diagnostic);
    const PrimExpr extent = ClosedInteger(gm->args[3], &analyzer, diagnostic);
    const PrimExpr stride = ClosedInteger(call->args[3], &analyzer, diagnostic);
    const PrimExpr raw_rows =
        ClosedInteger(call->args[4], &analyzer, diagnostic);
    const PrimExpr raw_columns =
        ClosedInteger(call->args[5], &analyzer, diagnostic);
    const PrimExpr abi_max = ClosedI64(std::numeric_limits<uint32_t>::max());
    PrimExpr full_rows = raw_rows == ClosedI64(rows);
    PrimExpr full_columns = raw_columns == ClosedI64(columns);
    if (to_l1 || fix) {
      full_rows = full_rows || raw_rows == ClosedI64(0);
      full_columns = full_columns || raw_columns == ClosedI64(0);
    }
    ICHECK(analyzer.CanProve(
        stride >= ClosedI64(columns) && stride <= abi_max &&
        stride <= floordiv(abi_max, ClosedI64(dtype.bytes())) && full_rows &&
        full_columns && extent >= ClosedI64(0)))
        << diagnostic(
               " requires full DMA masks and nonwrapping uint32 strides");
    ICHECK(analyzer.CanProve(
        ClosedI64(rows - 1) <=
        floordiv(ClosedI64(std::numeric_limits<int64_t>::max() - columns),
                 stride)))
        << diagnostic(" strided DMA span overflows int64");
    const PrimExpr span = ClosedI64(rows - 1) * stride + ClosedI64(columns);
    ICHECK(analyzer.CanProve(
        offset >= ClosedI64(0) && span <= ClosedI64(capacity) &&
        offset <= ClosedI64(capacity) - span &&
        floormod(offset, ClosedI64(width)) <= ClosedI64(width - columns) &&
        (rows == 1 ? Bool(true)
                   : floormod(stride, ClosedI64(width)) == ClosedI64(0))))
        << diagnostic(
               " complete DMA rectangle exceeds its GM parameter capacity");
    if (fix && call->args.size() == 10)
      ICHECK(analyzer.CanProve(
          ClosedInteger(call->args[9], &analyzer, diagnostic) == ClosedI64(0)))
          << diagnostic(" requires standalone FIX unitFlag=0");
    if (to_ub && call->args.size() >= 7)
      ICHECK(SideEffect(call->args[6]) <= CallEffectKind::kPure)
          << diagnostic(" requires a pure full-mask padding operand");
    if (to_l1)
      ICHECK(analyzer.CanProve(ClosedI64(rows - 1) * stride <= abi_max))
          << diagnostic(" GM-to-L1 fallback row offset exceeds uint32");
    const Var local_root = Downcast<Var>(local->args[1]);
    auto allocation = validation_.Allocations().find(local_root);
    ICHECK(allocation != validation_.Allocations().end() &&
           allocation->second.dtype == local_dtype)
        << diagnostic(" requires an actual typed local allocation");
    const auto *local_type = local_root->type_annotation.as<PointerTypeNode>();
    const std::string scope =
        local_type ? std::string(local_type->storage_scope) : "";
    ICHECK(scope == (to_l1 ? "shared.l1"
                     : fix ? "wmma.accumulator"
                           : "shared.ub"))
        << diagnostic(" has a mismatched local DMA scope");
    const auto sizes = function_->GetAttr<Map<Var, PrimExpr>>("size_map");
    ICHECK(sizes.defined() && sizes.value().count(local_root))
        << diagnostic(" requires the final local physical size map");
    const auto *physical_size = sizes.value()[local_root].as<IntImmNode>();
    ICHECK(physical_size && physical_size->value >= 0 &&
           allocation->second.elements <=
               std::numeric_limits<int64_t>::max() / local_dtype.bytes() &&
           allocation->second.elements * local_dtype.bytes() <=
               physical_size->value)
        << diagnostic(
               " actual local allocation exceeds its final physical capacity");
    const PrimExpr local_offset =
        ClosedInteger(local->args[2], &analyzer, diagnostic);
    const PrimExpr local_extent =
        ClosedInteger(local->args[3], &analyzer, diagnostic);
    ICHECK(analyzer.CanProve(
        local_offset >= ClosedI64(0) && local_extent >= ClosedI64(volume) &&
        local_offset <= ClosedI64(allocation->second.elements - volume)))
        << diagnostic(" complete DMA footprint exceeds its local allocation");
    if (to_l1 || fix)
      ICHECK(analyzer.CanProve(local_offset == ClosedI64(0)))
          << diagnostic(" requires the complete L1/L0 tile base");
    if (record.owner == kCoreCube) {
      std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> dependent;
      dependent.insert(validation_.Subcore());
      for (const Constr &constraint : record.task->outer_ctx.constrs_)
        if (constraint.kind == Constr::kBindValue &&
            UsesVar(constraint.value, [&](const VarNode *v) {
              return dependent.count(GetRef<Var>(v));
            }))
          dependent.insert(constraint.var);
      for (const PrimExpr &geometry : {gm->args[2], call->args[3]})
        ICHECK(!UsesVar(
            geometry,
            [&](const VarNode *v) { return dependent.count(GetRef<Var>(v)); }))
            << diagnostic(
                   " cannot instantiate Cube GM geometry from the AIV id");
    }
    return {&record, parameter.value(), gm->args[2], call->args[3],
            rows,    columns,           width,       write};
  }

private:
  PrimFunc function_;
  const FlatInputValidator &validation_;
  const ClosedDiagnostic &diagnostic_;
};

enum class ClosedRelation { kDifferentWorkgroups, kDifferentAIVs };

bool ClosedRectanglesDisjoint(const ClosedGMRectangle &a,
                              const ClosedGMRectangle &b,
                              const FlatInputValidator &validation,
                              ClosedRelation relation,
                              const ClosedDiagnostic &diagnostic) {
  ICHECK(a.buffer->data.same_as(b.buffer->data) && a.width == b.width)
      << diagnostic(" requires compatible rectangle roots and widths");
  ProofAnalyzer analyzer;
  PrimExpr offsets[2], strides[2], cids[2], vids[2];
  const ClosedGMRectangle *rectangles[] = {&a, &b};
  for (int instance = 0; instance < 2; ++instance) {
    Map<Var, PrimExpr> substitution;
    for (const Var &variable : validation.PrivateVariables())
      substitution.Set(variable, variable.copy_with_suffix(
                                     instance == 0 ? "_rect0" : "_rect1"));
    // Definitions are converted to mathematical constraints only after the
    // original-width lexical context and operands have been audited above.
    rectangles[instance]
        ->record->task->outer_ctx.Substitute(substitution)
        .ToConstraints()
        .Populate(analyzer);
    offsets[instance] =
        cast(DataType::Int(64),
             Substitute(rectangles[instance]->offset, substitution));
    strides[instance] =
        cast(DataType::Int(64),
             Substitute(rectangles[instance]->stride, substitution));
    cids[instance] = substitution[validation.Workgroup(diagnostic)];
    vids[instance] = substitution[validation.Subcore()];
  }
  analyzer.EnterConstraint(relation == ClosedRelation::kDifferentWorkgroups
                               ? cids[0] != cids[1]
                               : cids[0] == cids[1] && vids[0] != vids[1]);
  const PrimExpr width = ClosedI64(a.width);
  const PrimExpr columns_a = floormod(offsets[0], width);
  const PrimExpr columns_b = floormod(offsets[1], width);
  const PrimExpr separated_columns =
      columns_a + ClosedI64(a.columns) <= columns_b ||
      columns_b + ClosedI64(b.columns) <= columns_a;
  Var row_a("publication_row_a", DataType::Int(64));
  Var row_b("publication_row_b", DataType::Int(64));
  analyzer.Bind(row_a, Range::FromMinExtent(ClosedI64(0), ClosedI64(a.rows)));
  analyzer.Bind(row_b, Range::FromMinExtent(ClosedI64(0), ClosedI64(b.rows)));
  const PrimExpr row_base_a = floordiv(offsets[0], width);
  const PrimExpr row_base_b = floordiv(offsets[1], width);
  // A single-row DMA has no observed row stride. Multirow descriptors were
  // checked to advance by complete GM rows, without crossing row boundaries.
  const PrimExpr row_step_a =
      a.rows == 1 ? ClosedI64(0) : floordiv(strides[0], width);
  const PrimExpr row_step_b =
      b.rows == 1 ? ClosedI64(0) : floordiv(strides[1], width);
  return analyzer.CanProve(separated_columns ||
                           row_base_a + row_a * row_step_a !=
                               row_base_b + row_b * row_step_b);
}

} // namespace

LegacyCrossCorePlan PlanLegacyClosedCrossCore(
    const PrimFunc &function, const Target &target, const std::string &platform,
    const std::vector<LegacyOwnedTask> &tasks,
    const ConflictOracle &physical_conflict_oracle,
    const std::vector<LegacyOwnedControl> &controls,
    Optional<Var> private_composite_pool, const std::string &flat_rejection) {
  const ClosedDiagnostic diagnostic(flat_rejection);
  const String model = target.defined()
                           ? target->GetAttr<String>("model").value_or(String())
                           : String();
  ICHECK(target.defined() && (model == "ascendc" || model == "auto") &&
         (platform == "A2" || platform == "A3"))
      << diagnostic(" supports A2/A3 AscendC only");
  const bool noalias =
      function->GetAttr<Bool>("tir.noalias").value_or(Bool(false));
  ICHECK(noalias || HasTrustedFreshGMV2Contract(function)) << diagnostic(
      " requires explicit noalias or trusted Cython fresh-v2 allocation");
  ICHECK(static_cast<bool>(physical_conflict_oracle))
      << diagnostic(" requires the physical conflict oracle");
  for (const char *attribute : {"buffer_versions", "tl.buffer_versions_map"})
    if (auto versions = function->GetAttr<Map<Var, PrimExpr>>(attribute))
      for (const auto &[storage, value] : versions.value()) {
        const auto *version = value.as<IntImmNode>();
        ICHECK(version && version->value == 1)
            << diagnostic(" requires B1 storage: ", storage);
      }
  if (auto rings =
          function->GetAttr<PhysicalRingProvenance>(kPhysicalRingProvenance))
    ICHECK(rings.value().empty()) << diagnostic(" rejects physical rings");
  FlatInputValidator validation(function, tasks, controls, true);
  validation.Run(function->body);
  ICHECK(validation.IsValid())
      << diagnostic(" structural validation failed: ", validation.Failure());
  validation.Workgroup(diagnostic);
  LegacyCrossCorePlan plan;
  const LegacyOwnedTask *set = nullptr, *wait = nullptr;
  int channel = -1;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> fresh;
  for (const Var &root : ReadTrustedFreshGMStorages(function))
    fresh.insert(root);
  std::vector<ClosedGMRectangle> rectangles;
  ClosedDMAAudit dma(function, validation, diagnostic);
  auto real_gm = [&](const BufferRegion &region) {
    return IsGM(region) &&
           (!private_composite_pool.defined() ||
            !region->buffer->data.same_as(private_composite_pool.value()));
  };
  for (const LegacyOwnedTask &record : tasks) {
    ProofAnalyzer context_audit;
    PopulateClosedContext(record.task->outer_ctx, &context_audit, diagnostic);
    if (auto flag = DecodeLegacyCrossFlag(record.task->stmt)) {
      const auto *evaluation = record.task->stmt.as<EvaluateNode>();
      const auto *call =
          evaluation ? evaluation->value.as<CallNode>() : nullptr;
      ICHECK(call && call->op.as<OpNode>())
          << diagnostic(" requires native cross-core op identity");
      ICHECK(record.task->GetReadRegions().empty() &&
             record.task->GetWriteRegions().empty())
          << diagnostic(" requires separate real and shadow effects");
      if (flag->kind == LegacyCrossFlagKind::kSet) {
        ICHECK(!set && record.owner == kCoreCube && flag->mode == 2 &&
               flag->pipe == "FIX")
            << diagnostic(" requires one loop-outside mode-2 FIX Set on C");
        set = &record;
        channel = flag->id;
      } else {
        ICHECK(!wait && record.owner == kCoreVector &&
               (flag->pipe == "MTE2" || flag->pipe.empty()))
            << diagnostic(" requires one loop-outside Wait on V with MTE2 or "
                          "empty intent");
        wait = &record;
      }
      plan.reservations.push_back({record.owner, true, "", flag->id});
      continue;
    }
    CollectLocalReservation(record, &plan);
    for (const BufferRegion &read : record.task->GetReadRegions())
      if (real_gm(read))
        rectangles.push_back(dma.Run(record, read, false));
    for (const BufferRegion &write : record.task->GetWriteRegions())
      if (real_gm(write)) {
        ICHECK(noalias || fresh.count(write->buffer->data)) << diagnostic(
            " requires fresh-v2 roots for every GM write without noalias");
        rectangles.push_back(dma.Run(record, write, true));
      }
  }
  ICHECK(set && wait && DecodeLegacyCrossFlag(wait->task->stmt)->id == channel)
      << diagnostic(" requires one matching single-use C-to-V channel");
  // Workgroup streams are independent, and two AIVs execute each V tree.
  // Root disjointness follows explicit noalias or the trusted allocator recipe;
  // every same-root write pair still needs actual rectangle disjointness.
  for (const auto &a : rectangles) {
    if (a.record->owner == kCoreCube)
      ICHECK(a.record->order < set->order)
          << diagnostic(" requires complete Cube GM effects before Set");
    if (a.record->owner == kCoreVector && a.write)
      ICHECK(a.record->order > wait->order)
          << diagnostic(" requires every Vector GM write after Wait");
    for (const auto &b : rectangles) {
      if (!a.write && !b.write)
        continue;
      if (!a.buffer->data.same_as(b.buffer->data)) {
        ICHECK(noalias || fresh.count(a.buffer->data) ||
               fresh.count(b.buffer->data))
            << diagnostic(" has unknown cross-workgroup parameter aliasing");
        continue;
      }
      ICHECK(ClosedRectanglesDisjoint(
          a, b, validation, ClosedRelation::kDifferentWorkgroups, diagnostic))
          << diagnostic(" GM rectangles may overlap across workgroups");
      if (a.record->owner == kCoreVector && b.record->owner == kCoreVector)
        ICHECK(ClosedRectanglesDisjoint(
            a, b, validation, ClosedRelation::kDifferentAIVs, diagnostic))
            << diagnostic(" GM rectangles may overlap across AIV executions");
    }
  }
  Map<Var, PrimExpr> cube_variables, vector_variables;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> private_variables;
  for (const Var &variable : validation.PrivateVariables()) {
    private_variables.insert(variable);
    cube_variables.Set(variable, variable.copy_with_suffix("_closed_c"));
    vector_variables.Set(variable, variable.copy_with_suffix("_closed_v"));
  }
  auto instantiate = [&](const BufferRegion &region,
                         const Map<Var, PrimExpr> &substitution) {
    auto require_shared_layout = [&](const PrimExpr &value) {
      ICHECK(!UsesVar(value, [&](const VarNode *v) {
        return private_variables.count(GetRef<Var>(v));
      })) << diagnostic(" requires execution-independent buffer layouts");
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
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> shared;
  bool has_publication_raw = false;
  auto check = [&](const LegacyOwnedTask &c, const BufferRegion &cr,
                   const LegacyOwnedTask &v, const BufferRegion &vr, bool raw) {
    ConstrSet cctx = c.task->outer_ctx.Substitute(cube_variables);
    ConstrSet vctx = v.task->outer_ctx.Substitute(vector_variables);
    vctx.AddConstr(cube_variables[validation.Workgroup(diagnostic)] ==
                   vector_variables[validation.Workgroup(diagnostic)]);
    const auto conflict = physical_conflict_oracle(
        {cctx, instantiate(cr, cube_variables), vctx,
         instantiate(vr, vector_variables), nullptr, 0, 0, NullOpt, NullOpt});
    if (conflict.has_value() && !conflict.value())
      return;
    ICHECK(real_gm(cr) && real_gm(vr) &&
           cr->buffer->data.same_as(vr->buffer->data))
        << diagnostic(
               " has an uncovered local or unknown cross-resource alias");
    ICHECK(c.order < set->order && v.order > wait->order) << diagnostic(
        " requires every conflicting C/V effect to be covered by Set/Wait");
    shared.insert(cr->buffer->data);
    has_publication_raw |= raw;
  };
  for (const auto &c : tasks) {
    if (c.owner != kCoreCube)
      continue;
    for (const auto &v : tasks) {
      if (v.owner != kCoreVector)
        continue;
      for (const BufferRegion &write : c.task->GetWriteRegions()) {
        for (const BufferRegion &read : v.task->GetReadRegions())
          check(c, write, v, read, true);
        for (const BufferRegion &other : v.task->GetWriteRegions())
          check(c, write, v, other, false);
      }
      for (const BufferRegion &read : c.task->GetReadRegions())
        for (const BufferRegion &write : v.task->GetWriteRegions())
          check(c, read, v, write, false);
    }
  }
  ICHECK(has_publication_raw && !shared.empty())
      << diagnostic(" has no real same-root C-to-V GM publication");
  LegacyCrossShadowEffect release{
      set->task, PipeMask(ResourcePipe::kFixpipe), {}, {}};
  LegacyCrossShadowEffect acquire{
      wait->task, PipeMask(ResourcePipe::kScalar), {}, {}};
  for (const auto &[parameter, buffer] : function->buffer_map) {
    if (!shared.count(buffer->data))
      continue;
    Array<Range> ranges;
    for (const PrimExpr &extent : buffer->shape)
      ranges.push_back(Range::FromMinExtent(make_zero(extent.dtype()), extent));
    BufferRegion whole(buffer, ranges);
    // Read/write fence effects cover RAW, WAR and WAW while the ordinary local
    // engine places the original native instructions on FIX and Scalar. In
    // particular an empty-intent Wait needs a real S_MTE2 edge before DMA.
    release.reads.push_back(whole);
    release.writes.push_back(whole);
    acquire.reads.push_back(whole);
    acquire.writes.push_back(whole);
    plan.shared_gm_storages.push_back(buffer->data);
  }
  ICHECK_EQ(plan.shared_gm_storages.size(), shared.size()) << diagnostic(
      " requires unique parameter identities for all publication roots");
  plan.shadows.push_back(std::move(release));
  plan.shadows.push_back(std::move(acquire));
  return plan;
}

} // namespace tvm::tl::ascend_sync
