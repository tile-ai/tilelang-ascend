// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

// Legacy TIR import/export only. Dependency analysis and event allocation are
// provided by the ported upstream engine, not reimplemented in this adapter.

#include "legacy_adapter.h"

#include <tvm/ir/type.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "../../op/builtin.h"
#include "../common/attr.h"
#include "../common/operation_config.h"
#include "../common/pipeline_ring_provenance.h"
#include "legacy_composite.h"
#include "legacy_cross_core.h"
#include "legacy_helper_type.h"
#include "legacy_persistent_pipeline.h"
#include "legacy_pipeline_partition.h"
#include "legacy_snapshot.h"
#include "legacy_tail_copy.h"
#include "upstream/insert_sync.h"

namespace tvm::tl::ascend_sync {

using namespace tir;

// Occurrences, rather than StmtNode addresses, identify insertion anchors. A
// legal legacy SeqStmt may contain the very same statement object twice.
struct LegacyExportState {
  std::unordered_map<IRStructure *, size_t> original_occurrences;
  std::unordered_map<IRStructure *, Stmt> original_statements;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> mutable_bindings;
  // A scalar definition task models only a LetStmt's value evaluation. Its
  // trailing sync belongs at the start of the Let body, after that evaluation,
  // rather than after the entire lexical LetStmt has finished executing.
  std::unordered_set<IRStructure *> let_value_tasks;
  // A legacy If with no else is analyzed as a zero-or-one iteration loop.
  // Its proxy index never exists in emitted TIR; all placement expressions
  // substitute its sole possible iteration value before export.
  Map<Var, PrimExpr> conditional_loop_substitutions;
  size_t root_scope_occurrence{0};
  std::unordered_map<size_t, Stmt> closed_replacements;
};

namespace {

class StatementCounter : public StmtExprVisitor {
public:
  size_t Run(const Stmt &statement) {
    VisitStmt(statement);
    return count_;
  }

private:
  void VisitStmt(const Stmt &statement) final {
    ++count_;
    StmtExprVisitor::VisitStmt(statement);
  }
  size_t count_{0};
};

void RequirePure(const PrimExpr &expression, const char *where) {
  ICHECK(expression.defined() &&
         SideEffect(expression) <= CallEffectKind::kPure)
      << "Upstream Ascend sync legacy adapter requires a pure " << where
      << "; mutable reads and opaque calls cannot be re-evaluated as guards: "
      << expression;
}

std::optional<PrimExpr> LegacyAssumption(const Stmt &statement) {
  const auto *evaluation = statement.as<EvaluateNode>();
  const auto *call = evaluation ? evaluation->value.as<CallNode>() : nullptr;
  const auto *op = call ? call->op.as<OpNode>() : nullptr;
  if (!op || op->name != "tir.assume")
    return std::nullopt;
  ICHECK_EQ(call->args.size(), 1U) << "Malformed tir.assume predicate";
  RequirePure(call->args[0], "tir.assume predicate");
  ICHECK(call->args[0].dtype().is_scalar() && call->args[0].dtype().is_bool())
      << "tir.assume requires a pure scalar boolean predicate";
  return call->args[0];
}

std::string OperationName(const CallNode *call, size_t *argument_offset) {
  *argument_offset = 0;
  std::string name;
  if (call->op.same_as(builtin::call_extern())) {
    ICHECK(!call->args.empty());
    name = Downcast<StringImm>(call->args[0])->value;
    *argument_offset = 1;
  } else if (const auto *op = call->op.as<OpNode>()) {
    name = op->name;
  }
  const size_t template_pos = name.find('<');
  if (template_pos != std::string::npos)
    name.erase(template_pos);
  const std::string prefix = "tl::ascend::";
  if (name.rfind(prefix, 0) == 0)
    name.erase(0, prefix.size());
  return name;
}

struct GmToUbShape {
  std::string element_type;
  int64_t columns;
  int64_t rows;
};

GmToUbShape ParseGmToUbShape(const CallNode *call, size_t offset,
                             const char *diagnostic) {
  ICHECK_EQ(offset, 1U) << diagnostic;
  const auto *helper = call->args[0].as<StringImmNode>();
  ICHECK(helper != nullptr) << diagnostic;
  const std::string &name = helper->value;
  const size_t begin = name.find('<');
  const size_t end = name.find('>', begin);
  ICHECK(begin != std::string::npos && end != std::string::npos &&
         name.find_first_not_of(" \t\r\n", end + 1) == std::string::npos)
      << diagnostic;
  std::vector<std::string> arguments;
  size_t position = begin + 1;
  while (position <= end) {
    const size_t comma = name.find(',', position);
    const size_t next = comma == std::string::npos || comma > end ? end : comma;
    arguments.push_back(name.substr(position, next - position));
    position = next + 1;
  }
  ICHECK(arguments.size() == 2 || arguments.size() == 3) << diagnostic;
  auto dimension = [&](size_t index) -> int64_t {
    const std::string &token = arguments[index];
    const size_t first = token.find_first_not_of(" \t\r\n");
    const size_t last = token.find_last_not_of(" \t\r\n");
    ICHECK(first != std::string::npos) << diagnostic;
    uint64_t value = 0;
    const char *limit = token.data() + last + 1;
    const auto parsed = std::from_chars(token.data() + first, limit, value);
    ICHECK(parsed.ec == std::errc() && parsed.ptr == limit && value > 0 &&
           value <= std::numeric_limits<uint32_t>::max())
        << diagnostic;
    return static_cast<int64_t>(value);
  };
  const size_t first = arguments[0].find_first_not_of(" \t\r\n");
  const size_t last = arguments[0].find_last_not_of(" \t\r\n");
  ICHECK(first != std::string::npos) << diagnostic;
  return {arguments[0].substr(first, last - first + 1), dimension(1),
          arguments.size() == 3 ? dimension(2) : 1};
}

// AscendCopy emits copy_gm_to_ub<T, dstN, dstM>(src_ptr, dst_ptr,
// strideN, validRows, validCols, padValue, ...metadata). CopyCodegen swaps
// the two pointers and forwards only those first four scalar arguments to
// common.h. The masked helper also executes V work and hard-coded event 0
// protocols, which cannot be described by its single-MTE2 operation config.
// Admit only a proof that those hidden branches cannot execute.
void RequireFullMaskGmToUb(const CallNode *call, size_t offset,
                           const ConstrSet &context) {
  ICHECK_GE(call->args.size(), offset + 2)
      << "Malformed copy_gm_to_ub buffer arguments";
  // With neither mask supplied, the C++ defaults are exactly dstM/dstN,
  // independently of the optional realSrcN argument or template dimensions.
  if (call->args.size() <= offset + 3)
    return;

  const char *reject =
      "copy_gm_to_ub requires provably full masks; tail or unknown masks "
      "execute hidden Vector work and event slots that need the later "
      "composite-effect adapter";
  const GmToUbShape shape = ParseGmToUbShape(call, offset, reject);
  ProofAnalyzer analyzer;
  context.Populate(analyzer);
  for (const auto &[argument, expected] :
       {std::pair<size_t, int64_t>{offset + 3, shape.rows},
        std::pair<size_t, int64_t>{offset + 4, shape.columns}}) {
    if (argument >= call->args.size())
      continue; // A trailing omitted mask keeps its full-tile C++ default.
    const PrimExpr &mask = call->args[argument];
    RequirePure(mask, "copy_gm_to_ub mask");
    const DataType dtype = mask.dtype();
    ICHECK(dtype.is_scalar() && ((dtype.is_int() && dtype.bits() <= 64) ||
                                 (dtype.is_uint() && dtype.bits() <= 32)))
        << reject;
    // Compare in a value-preserving signed type; constructing the expected
    // constant in a narrow mask dtype could truncate it and admit a tail.
    ICHECK(analyzer.CanProve(cast(DataType::Int(64), mask) ==
                             make_const(DataType::Int(64), expected)))
        << reject << "; mask=" << mask << ", expected=" << expected;
  }
}

// Reservation prevents ID collisions, but not cycles against a hand-written
// schedule that deliberately issues a consumer before its producer. Accept
// only forward, balanced lexical protocols until those edges are modelled.
class ManualFlagValidator : public StmtExprVisitor {
public:
  void Run(const Stmt &body) {
    VisitStmt(body);
    RequireBalanced();
  }

private:
  void RequireBalanced() const {
    ICHECK(pending_.empty())
        << "Upstream Ascend sync requires forward balanced manual flag pairs; "
           "unmatched or cross-control-flow events need a manual schedule "
           "model";
  }

  static std::string Pipe(const PrimExpr &expression) {
    const auto *value = expression.as<StringImmNode>();
    ICHECK(value != nullptr)
        << "Manual flag pipe must be a literal in the upstream sync adapter";
    std::string result = value->value;
    if (result.rfind("PIPE_", 0) == 0)
      result.erase(0, 5);
    return result;
  }

  void VisitStmt_(const ForNode *op) final {
    RequireBalanced();
    StmtExprVisitor::VisitStmt_(op);
    RequireBalanced();
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    RequireBalanced();
    VisitExpr(op->condition);
    VisitStmt(op->then_case);
    RequireBalanced();
    if (op->else_case.defined()) {
      VisitStmt(op->else_case.value());
      RequireBalanced();
    }
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    const int previous = resource_;
    if (op->attr_key == "resource_scope") {
      const auto *value = op->value.as<IntImmNode>();
      ICHECK(value != nullptr);
      resource_ = static_cast<int>(value->value);
    }
    StmtExprVisitor::VisitStmt_(op);
    resource_ = previous;
  }

  void VisitExpr_(const CallNode *op) final {
    size_t offset = 0;
    const std::string name = OperationName(op, &offset);
    const bool manual =
        name == "tl.ascend_set_flag" || name == "tl.ascend_wait_flag";
    const bool automatic =
        name == "tl.ascend_auto_set_flag" || name == "tl.ascend_auto_wait_flag";
    if (!manual && !automatic) {
      StmtExprVisitor::VisitExpr_(op);
      return;
    }
    ICHECK_EQ(op->args.size(), offset + (manual ? 3U : 2U));
    const std::string event =
        manual ? Pipe(op->args[offset]) + "_" + Pipe(op->args[offset + 1])
               : Pipe(op->args[offset]);
    const auto *id = op->args[offset + (manual ? 2 : 1)].as<IntImmNode>();
    ICHECK(id != nullptr && id->value >= 0 && id->value < 8)
        << "Upstream Ascend sync requires a literal manual event ID in [0, 8)";
    const std::string key = std::to_string(resource_) + ":" + event + ":" +
                            std::to_string(id->value);
    const bool is_set =
        name == "tl.ascend_set_flag" || name == "tl.ascend_auto_set_flag";
    if (is_set) {
      ICHECK(pending_.insert(key).second)
          << "Duplicate manual SetFlag without WaitFlag: " << key;
    } else {
      ICHECK_EQ(pending_.erase(key), 1U)
          << "Upstream Ascend sync cannot combine automatic dependencies with "
             "a manual wait before its producer: "
          << key;
    }
  }

  int resource_{-1};
  std::unordered_set<std::string> pending_;
};

CoreMask ResourceForBuffer(const Buffer &buffer) {
  const String scope = buffer.scope();
  ICHECK(scope != "shared.l0a.sf" && scope != "shared.l0b.sf")
      << "Legacy synchronization import does not support scale-factor storage";
  if (scope == "shared.ub" || scope == "shared")
    return kCoreVector;
  if (scope == "shared.l1" || scope == "wmma.matrix_a" ||
      scope == "wmma.matrix_b" || scope == "wmma.accumulator")
    return kCoreCube;
  return kCoreUnassigned;
}

BufferRegion ElementRegion(const Buffer &buffer,
                           const Array<PrimExpr> &indices) {
  Array<Range> region;
  for (const PrimExpr &index : indices)
    region.push_back(Range::FromMinExtent(index, make_const(index.dtype(), 1)));
  return BufferRegion(buffer, region);
}

bool IsPureCall(const CallNode *call) {
  const auto *node = call->op.as<OpNode>();
  if (node == nullptr)
    return false;
  const Op op = GetRef<Op>(node);
  static auto effects = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
  return effects.count(op) &&
         effects[op]->value <= static_cast<int>(CallEffectKind::kPure);
}

class LeafEffects : public ExprVisitor {
public:
  LeafEffects(TaskNode *task, const PhysicalStorageResolver *storage,
              const PhysicalRingClockMap &clocks,
              const LegacyCompositePlan &composites,
              LegacyTailCopyPlan *tail_copies,
              const Map<Var, Array<PrimExpr>> &logic_buffer_shapes,
              bool pto_transpose)
      : task_(task), storage_(storage), clocks_(clocks),
        composites_(composites), tail_copies_(tail_copies),
        logic_buffer_shapes_(logic_buffer_shapes),
        pto_transpose_(pto_transpose) {}

  void Analyze(const Stmt &statement) {
    task_->SetPipeMask(PipeMask(ResourcePipe::kScalar));
    if (const auto *store = statement.as<BufferStoreNode>()) {
      ICHECK(!storage_->IsPhysicalRingStorage(store->buffer->data))
          << "Physical ring scalar stores require a later footprint adapter";
      task_->AddWriteRegion(ElementRegion(store->buffer, store->indices));
      InferResource(ResourceForBuffer(store->buffer));
      for (const PrimExpr &index : store->indices)
        VisitExpr(index);
      VisitExpr(store->value);
    } else {
      const auto *evaluate = statement.as<EvaluateNode>();
      ICHECK(evaluate != nullptr);
      if (const auto *call = evaluate->value.as<CallNode>()) {
        if (AnalyzeOperation(call))
          return;
      }
      VisitExpr(evaluate->value);
    }
  }

  CoreMask InferredResource() const { return inferred_resource_; }

private:
  void InferResource(CoreMask resource) {
    if (resource == kCoreUnassigned)
      return;
    ICHECK(inferred_resource_ == kCoreUnassigned ||
           inferred_resource_ == resource)
        << "A scalar legacy task accesses both Cube and Vector local storage";
    inferred_resource_ = resource;
  }

  void ValidateRingHelperAccess(const CallNode *call, const std::string &name,
                                size_t offset, size_t operand,
                                const std::string &access, int64_t mask,
                                const BufferRegion &region) const {
    if (!storage_->ManualRingVersions().Get(region->buffer->data))
      return;
    const char *reject =
        "Physical ring helper footprint or slot contract is unsupported";
    ICHECK_EQ(region->region.size(), 1U) << reject;
    const DataType dtype = region->buffer->dtype;
    ICHECK(dtype.lanes() == 1 && dtype.bits() > 0 && dtype.bits() % 8 == 0)
        << reject;
    ProofAnalyzer analyzer;
    task_->outer_ctx.Populate(analyzer);
    const PrimExpr slot_elements =
        cast(DataType::Int(64), region->region[0]->extent);
    if (name == "copy_gm_to_l1" && operand == 1 && access == "write" &&
        mask == 2) {
      const auto proof = composites_.cube_ring_copies.find(call);
      ICHECK(composites_.static_cube_pipeline &&
             proof != composites_.cube_ring_copies.end() &&
             proof->second.first.same_as(region->buffer->data) &&
             analyzer.CanProve(
                 make_const(DataType::Int(64), proof->second.second) <=
                 slot_elements * Integer(dtype.bytes())))
          << reject << "; copy_gm_to_l1 has no complete certified slot";
      return;
    }
    if (name == "copy_gm_to_ub" && operand == 1 && access == "write" &&
        mask == 2) {
      const GmToUbShape shape = ParseGmToUbShape(call, offset, reject);
      const std::unordered_map<std::string, DataType> element_types = {
          {"half", DataType::Float(16)},        {"float", DataType::Float(32)},
          {"bfloat16_t", DataType::BFloat(16)}, {"int8_t", DataType::Int(8)},
          {"uint8_t", DataType::UInt(8)},       {"int16_t", DataType::Int(16)},
          {"uint16_t", DataType::UInt(16)},     {"int32_t", DataType::Int(32)},
          {"uint32_t", DataType::UInt(32)}};
      const auto element =
          element_types.find(CanonicalHelperElementType(shape.element_type));
      ICHECK(element != element_types.end() && element->second == dtype)
          << reject << "; template dtype disagrees with the physical ring";
      const int64_t bytes = dtype.bytes();
      ICHECK_LE(shape.columns,
                (std::numeric_limits<int64_t>::max() - 31) / bytes)
          << reject;
      const int64_t row_bytes = ((shape.columns * bytes + 31) / 32) * 32;
      ICHECK_LE(shape.rows, std::numeric_limits<int64_t>::max() / row_bytes)
          << reject;
      const int64_t footprint = shape.rows * row_bytes;
      // Full masks avoid the helper's hidden V/flag branch, but DMA rows can
      // still advance in whole 32-byte blocks. Use the real template shape,
      // never the access_ptr extent, to bound the complete footprint.
      ICHECK(analyzer.CanProve(make_const(DataType::Int(64), footprint) <=
                               slot_elements *
                                   make_const(DataType::Int(64), bytes)))
          << reject << "; copy_gm_to_ub exceeds its complete physical slot";
      return;
    }
    if (name == "tl.ascend_abs" && operand == 1 && access == "read" &&
        mask == 1) {
      ICHECK_LT(offset + 2, call->args.size()) << reject;
      const PrimExpr &count = call->args[offset + 2];
      RequirePure(count, "physical ring Abs count");
      const DataType count_type = count.dtype();
      ICHECK(count_type.is_scalar() &&
             ((count_type.is_int() && count_type.bits() <= 64) ||
              (count_type.is_uint() && count_type.bits() <= 32)))
          << reject;
      const PrimExpr elements = cast(DataType::Int(64), count);
      ICHECK(analyzer.CanProve(elements >= make_zero(DataType::Int(64)) &&
                               elements <= slot_elements))
          << reject << "; Abs count exceeds its complete physical slot";
      return;
    }
    LOG(FATAL) << reject << "; " << name << " operand " << operand
               << " has no audited ring effect contract";
  }

  bool AnalyzeOperation(const CallNode *call) {
    size_t offset = 0;
    const std::string name = OperationName(call, &offset);
    std::string helper = name;
    if (helper.rfind("tl.", 0) == 0)
      helper.erase(0, 3);
    if (helper.rfind("ascend_", 0) == 0)
      helper.erase(0, 7);
    if (const auto *composite = composites_.Find(call)) {
      task_->SetPipeMask(PipeMask(composite->completion));
      SetSyncBoundaryPipes(task_, composite->acquire, composite->completion);
      inferred_resource_ = composite->execution_owner;
      if (composite->pipeline_clock.defined()) {
        ProofAnalyzer analyzer;
        task_->outer_ctx.Populate(analyzer);
        ICHECK(composites_.static_cube_pipeline &&
               analyzer.CanProveEqual(call->args[4],
                                      composite->pipeline_clock.value() == 0))
            << "Cube pipeline GEMM clear must equal logical clock == 0 in "
               "its actual task context";
      }
      std::unordered_set<const CallNode *> pointers;
      for (const auto &effect : composite->buffers) {
        const auto *pointer = effect.access_ptr.get();
        BufferRegion region =
            storage_->AccessPtrRegion(pointer, clocks_, task_->outer_ctx);
        if (effect.logical_ring.defined()) {
          ProofAnalyzer analyzer;
          task_->outer_ctx.Populate(analyzer);
          ICHECK(composites_.static_cube_pipeline &&
                 region->buffer->data.same_as(effect.logical_ring.value()) &&
                 region->region.size() == 1 &&
                 analyzer.CanProve(
                     make_const(DataType::Int(64), effect.required_bytes) <=
                     cast(DataType::Int(64), region->region[0]->extent) *
                         Integer(region->buffer->dtype.bytes())))
              << "Cube pipeline GEMM physical footprint exceeds its ring slot";
        }
        if (effect.read)
          task_->AddReadRegion(region);
        if (effect.write)
          task_->AddWriteRegion(region);
        pointers.insert(pointer);
        RequirePure(pointer->args[2], "composite access offset");
        RequirePure(pointer->args[3], "composite access extent");
      }
      for (const PrimExpr &argument : call->args) {
        if (const auto *pointer = argument.as<CallNode>();
            pointer != nullptr && pointers.count(pointer))
          continue;
        RequirePure(argument, "composite scalar argument");
        VisitExpr(argument);
      }
      return true;
    }
    ICHECK(helper != "gemm_v0" && helper != "gemm_v1")
        << "Composite " << name
        << " uses MTE1, Cube and hidden L0 event slots; it needs the later "
           "composite-effect adapter";
    if (auto cross = DecodeLegacyCrossFlag(task_->stmt)) {
      task_->SetPipeMask(PipeMask(cross->kind == LegacyCrossFlagKind::kSet
                                      ? ParseAscendPipe(cross->pipe)
                                      : ResourcePipe::kScalar));
      return true;
    }
    // Existing flag calls remain in their original positions. The engine scans
    // original_body and reserves their IDs before allocating generated events.
    if (name == "tl.ascend_set_flag" || name == "tl.ascend_wait_flag" ||
        name == "tl.ascend_auto_set_flag" ||
        name == "tl.ascend_auto_wait_flag" ||
        name == "tl.ascend_pipe_barrier" || name == "tl.ascend_auto_barrier") {
      for (size_t i = offset; i < call->args.size(); ++i)
        RequirePure(call->args[i], "existing synchronization argument");
      return true;
    }

    auto configured = GetOperationConfig().find(name);
    if (configured == GetOperationConfig().end()) {
      ICHECK(IsPureCall(call))
          << "Unknown side effects in upstream Ascend sync legacy import: "
          << name;
      return false;
    }
    OperationConfig config = ResolveOperationConfigWithPipeline(
        call, name, offset, logic_buffer_shapes_, pto_transpose_);
    std::optional<LegacyTailCopyEffect> tail_copy;
    if (helper == "copy_gm_to_ub") {
      tail_copy =
          tail_copies_->AuditCall(call, offset, task_->outer_ctx, *storage_);
      if (!tail_copy)
        RequireFullMaskGmToUb(call, offset, task_->outer_ctx);
    }
    const ResourcePipe pipe = ParseAscendPipe(config.default_pipeline);
    ICHECK(pipe != ResourcePipe::kUnknown && pipe != ResourcePipe::kAll)
        << "Unsupported pipeline in operation configuration for " << name;
    task_->SetPipeMask(PipeMask(pipe));
    if (config.copy_resource_scope == 0)
      inferred_resource_ = kCoreCube;
    else if (config.copy_resource_scope == 1)
      inferred_resource_ = kCoreVector;
    else if (pipe == ResourcePipe::kCube || pipe == ResourcePipe::kMTE1 ||
             pipe == ResourcePipe::kFixpipe)
      inferred_resource_ = kCoreCube;
    else if (pipe == ResourcePipe::kVector)
      inferred_resource_ = kCoreVector;

    std::unordered_set<size_t> pointer_arguments;
    for (const auto &[index, access] : config.buffer_accesses) {
      const size_t position = offset + index;
      ICHECK_LT(position, call->args.size())
          << "Malformed configured operation " << name;
      const auto *pointer = call->args[position].as<CallNode>();
      ICHECK(pointer != nullptr &&
             pointer->op.same_as(builtin::tvm_access_ptr()))
          << "Configured buffer operand is not a legacy access_ptr in " << name;
      ICHECK_EQ(pointer->args.size(), 5U);
      const auto *mask = pointer->args[4].as<IntImmNode>();
      ICHECK(mask != nullptr && mask->value > 0 && (mask->value & ~3) == 0)
          << "Invalid legacy access mask in " << name;
      BufferRegion region =
          storage_->AccessPtrRegion(pointer, clocks_, task_->outer_ctx);
      if (tail_copy) {
        ICHECK(index == 0 || index == 1)
            << "Unexpected copy_gm_to_ub buffer argument";
        region =
            index == 0 ? tail_copy->source_read : tail_copy->destination_write;
      }
      ValidateRingHelperAccess(call, name, offset, index, access, mask->value,
                               region);
      // Keep both actual read/write bits, including in-place workspace calls.
      // Declared effects remain a lower bound if a helper uses a narrower mask.
      if ((mask->value & 1) || access == "read")
        task_->AddReadRegion(region);
      if ((mask->value & 2) || access == "write")
        task_->AddWriteRegion(region);
      pointer_arguments.insert(position);
      for (size_t argument : {2U, 3U}) {
        RequirePure(pointer->args[argument], "access_ptr offset/extent");
        // Keep captured pointer fields in the task's scalar input metadata,
        // just like explicit hardware scalar operands. Their register uses
        // remain after the original Let definitions during export.
        VisitExpr(pointer->args[argument]);
      }
    }
    for (size_t i = offset; i < call->args.size(); ++i) {
      if (pointer_arguments.count(i))
        continue;
      // A scalar load embedded in a hardware operation would need its own
      // scalar task and evaluation snapshot, which this first importer rejects.
      RequirePure(call->args[i], "hardware operation scalar argument");
      VisitExpr(call->args[i]);
    }
    return true;
  }

  void VisitExpr_(const VarNode *op) final {
    if (!op->dtype.is_handle()) {
      auto vars = task_->GetReadVars();
      const Var var = GetRef<Var>(op);
      if (std::none_of(vars.begin(), vars.end(),
                       [&](const Var &v) { return v.same_as(var); }))
        vars.push_back(var);
      task_->SetReadVars(vars);
    }
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    ICHECK(!storage_->IsPhysicalRingStorage(op->buffer->data))
        << "Physical ring scalar loads require a later footprint adapter";
    task_->AddReadRegion(ElementRegion(op->buffer, op->indices));
    InferResource(ResourceForBuffer(op->buffer));
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    ICHECK(!op->op.same_as(builtin::tvm_access_ptr()))
        << "A legacy access_ptr must belong to a configured hardware operation";
    ICHECK(IsPureCall(op))
        << "Unknown nested side effects in upstream Ascend sync legacy import: "
        << op->op;
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const ProducerLoadNode *op) final {
    LOG(FATAL) << "Legacy synchronization import requires explicit BufferLoad "
                  "storage, not ProducerLoad";
  }

  TaskNode *task_;
  const PhysicalStorageResolver *storage_;
  const PhysicalRingClockMap &clocks_;
  const LegacyCompositePlan &composites_;
  LegacyTailCopyPlan *tail_copies_;
  const Map<Var, Array<PrimExpr>> &logic_buffer_shapes_;
  bool pto_transpose_;
  CoreMask inferred_resource_{kCoreUnassigned};
};

class LegacyImporter {
public:
  LegacyImporter(ImportedLegacySyncIR *result, const Target &target,
                 const String &platform, bool ordinary_sync,
                 const LegacyCompositePlan &composites,
                 LegacyTailCopyPlan *tail_copies,
                 const LegacyPipelineCase *pipeline_case,
                 const LegacyPersistentPipelineImport *persistent)
      : result_(result), target_(target), platform_(platform),
        ordinary_sync_(ordinary_sync), composites_(composites),
        tail_copies_(tail_copies), pipeline_case_(pipeline_case),
        persistent_(persistent),
        logic_buffer_shapes_(
            result->original_function
                ->GetAttr<Map<Var, Array<PrimExpr>>>(kLogicBufferShapes)
                .value_or(Map<Var, Array<PrimExpr>>())) {}

  void Run() {
    CheckVersions();
    ConstrSet context;
    if (persistent_ != nullptr && persistent_->ScopedRoot().defined()) {
      ICHECK_GT(persistent_->RootOccurrence(), 0U);
      occurrence_ = persistent_->RootOccurrence() - 1;
      context = persistent_->RootContext();
      Import(persistent_->ScopedRoot(), &result_->roots, nullptr, context, {},
             kCoreVector, false, {});
      result_->export_state->root_scope_occurrence =
          persistent_->RootOccurrence();
    } else {
      Import(result_->original_function->body, &result_->roots, nullptr,
             context, {}, kCoreUnassigned, true, {});
    }
    if (tail_copies_->HasTailCopies()) {
      ICHECK_EQ(selected_resources_, kCoreVector)
          << "GM-to-UB tail helper synchronization currently requires "
             "Vector-only execution";
      ICHECK(result_->manual_buffer_versions.empty())
          << "GM-to-UB tail helpers require a later physical ring contract";
      tail_copies_->ValidateManualEvents();
    }
    const bool mixed = (selected_resources_ & kCoreCube) != 0 &&
                       (selected_resources_ & kCoreVector) != 0;
    if (!mixed) {
      const CoreMask selected = selected_resources_ == kCoreUnassigned
                                    ? kCoreVector
                                    : selected_resources_;
      for (const auto &task : all_tasks_)
        task->SetCoreMask(selected);
      for (const auto &task : all_tasks_)
        ICHECK(!DecodeLegacyCrossFlag(task->stmt).has_value())
            << "Existing cross-core flags require a complete mixed C/V "
               "publication protocol";
      result_->local_graphs.push_back(
          {selected, result_->roots, ReservationsFor(selected)});
      return;
    }
    ICHECK(ordinary_sync_)
        << "Mixed C/V synchronization requires tl.ascend_auto_sync=True";
    ICHECK(result_->manual_buffer_versions.empty())
        << "Mixed C/V physical rings require an epoch publication protocol";
    std::vector<LegacyOwnedTask> owned;
    std::vector<LegacyOwnedControl> controls;
    std::vector<std::pair<std::shared_ptr<IRStructure>, CoreMask>> owned_roots;
    auto add_owned_task = [&](const std::shared_ptr<IRStructure> &node) {
      ICHECK(node->IsTask())
          << "Mixed C/V requires flat input except one static C-only prefix "
             "For; nested or conditional controls are unsupported";
      auto task = std::static_pointer_cast<TaskNode>(node);
      const auto owner = task_owners_.find(task.get());
      ICHECK(owner != task_owners_.end() && owner->second != kCoreUnassigned)
          << "Mixed C/V synchronization requires a definite task owner";
      task->SetCoreMask(owner->second);
      owned.push_back(
          {task, result_->export_state->original_statements.at(task.get()),
           owner->second,
           result_->export_state->original_occurrences.at(task.get())});
      return owner->second;
    };
    for (const auto &node : result_->roots) {
      if (node->IsTask()) {
        owned_roots.push_back({node, add_owned_task(node)});
        continue;
      }
      auto control = std::static_pointer_cast<ControlNode>(node);
      const Stmt &original =
          result_->export_state->original_statements.at(node.get());
      ICHECK(node->IsControl() && original.as<ForNode>() &&
             control->control.same_as(original) && control->task != nullptr &&
             !control->children.empty())
          << "Mixed C/V requires flat input except one static C-only prefix "
             "For; conditional controls are unsupported";
      for (const auto &child : control->children)
        ICHECK_EQ(add_owned_task(child), kCoreCube)
            << "Mixed C/V requires flat input on V and C-only prefix controls";
      // Keep the real loop and its scalar header in the C graph. Its children
      // retain their parent/index and original statement anchors; a union of
      // effects is not a replacement for loop-exit/backedge analysis.
      control->task->SetCoreMask(kCoreCube);
      controls.push_back(
          {control, original, kCoreCube,
           result_->export_state->original_occurrences.at(node.get())});
      owned_roots.push_back({node, kCoreCube});
    }
    const auto plan =
        PlanLegacyFlatCrossCore(result_->original_function, target_, platform_,
                                owned, result_->MakeOracle(), controls);
    for (const auto &shadow : plan.shadows) {
      shadow.task->SetPipeMask(shadow.pipe_mask);
      for (const auto &region : shadow.reads)
        shadow.task->AddReadRegion(region);
      for (const auto &region : shadow.writes)
        shadow.task->AddWriteRegion(region);
    }
    for (CoreMask owner : {kCoreCube, kCoreVector}) {
      LegacyLocalSyncGraph local{owner, {}, ReservationsFor(owner)};
      for (const auto &[node, node_owner] : owned_roots) {
        if (node_owner != owner)
          continue;
        node->SetIndex(local.roots.size());
        local.roots.push_back(node);
      }
      result_->local_graphs.push_back(std::move(local));
    }
  }

private:
  Stmt ReservationsFor(CoreMask owner) const {
    Array<Stmt> reservations;
    for (const auto &task : all_tasks_) {
      if (task->GetCoreMask() != owner)
        continue;
      const auto *evaluate = task->stmt.as<EvaluateNode>();
      if (evaluate == nullptr)
        continue;
      const auto *call = evaluate->value.as<CallNode>();
      if (call == nullptr)
        continue;
      size_t offset = 0;
      const std::string name = OperationName(call, &offset);
      if (name == "tl.ascend_set_flag" || name == "tl.ascend_wait_flag" ||
          name == "tl.ascend_auto_set_flag" ||
          name == "tl.ascend_auto_wait_flag")
        reservations.push_back(task->stmt);
    }
    if (owner == kCoreCube) {
      for (const auto &[event, ids] : composites_.reserved_intra_ids)
        for (int id : ids)
          reservations.push_back(
              Evaluate(Call(DataType::Handle(), tl::ascend_auto_set_flag(),
                            {StringImm(event), Integer(id)})));
    }
    if (owner == kCoreVector) {
      if (persistent_ != nullptr && persistent_->Reservations().defined())
        reservations.push_back(persistent_->Reservations());
      for (const auto &[event, ids] : tail_copies_->ReservedVectorIntraIds())
        for (int id : ids)
          reservations.push_back(
              Evaluate(Call(DataType::Handle(), tl::ascend_auto_set_flag(),
                            {StringImm(event), Integer(id)})));
    }
    return reservations.empty() ? Stmt(Evaluate(0))
                                : Stmt(SeqStmt(reservations));
  }

  void CheckVersions() {
    result_->manual_buffer_versions =
        result_->physical_storage->ManualRingVersions();
    for (const auto &[storage, versions] : result_->manual_buffer_versions) {
      const String scope =
          result_->physical_storage->LookupBuffer(storage).scope();
      ICHECK((scope != "shared.l1" && scope != "shared.l1.dyn") ||
             composites_.static_cube_pipeline)
          << "Physical L1 rings require a certified static Cube pipeline; "
             "Vector/other helpers cannot borrow its storage capability";
    }
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> matched;
    for (const char *name : {"buffer_versions", "tl.buffer_versions_map"}) {
      auto versions =
          result_->original_function->GetAttr<Map<Var, PrimExpr>>(name);
      if (!versions.defined())
        continue;
      for (const auto &[storage, version] : versions.value()) {
        const auto *constant = version.as<IntImmNode>();
        ICHECK(constant != nullptr && constant->value > 0)
            << "Physical pipeline/version counts must be positive constants";
        const auto actual = result_->manual_buffer_versions.Get(storage);
        if (constant->value > 1 || actual.has_value()) {
          ICHECK(actual.has_value() && actual.value() == constant->value)
              << "Physical pipeline/version provenance is missing or disagrees "
                 "with "
              << name << " for " << storage
              << "; version metadata must not be discarded";
          matched.insert(storage);
        }
      }
    }
    ICHECK_EQ(matched.size(), result_->manual_buffer_versions.size())
        << "Physical ring provenance requires matching buffer_versions "
           "metadata for every logical ring";
  }

  CoreMask SelectResource(CoreMask explicit_resource, CoreMask inferred) {
    if (explicit_resource != kCoreUnassigned && inferred != kCoreUnassigned)
      ICHECK_EQ(explicit_resource, inferred)
          << "Configured operation conflicts with resource_scope";
    const CoreMask resource =
        explicit_resource != kCoreUnassigned ? explicit_resource : inferred;
    ICHECK(result_->manual_buffer_versions.empty() || resource != kCoreCube ||
           composites_.static_cube_pipeline)
        << "Physical Cube rings require a certified static Cube pipeline";
    if (resource == kCoreUnassigned)
      return resource;
    selected_resources_ |= resource;
    return resource;
  }

  GuardList CloneGuards(const std::vector<PrimExpr> &guards) const {
    GuardList result;
    for (const PrimExpr &guard : guards)
      result.push_back(std::make_unique<ConditionGuard>(guard));
    return result;
  }

  void Record(const std::shared_ptr<IRStructure> &node, const Stmt &statement,
              size_t occurrence,
              std::vector<std::shared_ptr<IRStructure>> *siblings,
              ControlNode *parent) {
    node->SetParent(parent);
    node->SetIndex(siblings->size());
    siblings->push_back(node);
    result_->export_state->original_occurrences.emplace(node.get(), occurrence);
    result_->export_state->original_statements.emplace(node.get(), statement);
  }

  void Import(const Stmt &statement,
              std::vector<std::shared_ptr<IRStructure>> *siblings,
              ControlNode *parent, ConstrSet context,
              std::vector<PrimExpr> guards, CoreMask explicit_resource,
              bool root_chain, PhysicalRingClockMap clocks) {
    const size_t occurrence = ++occurrence_;
    if (persistent_ != nullptr && persistent_->ClosedOwner().defined() &&
        occurrence == persistent_->ClosedOccurrence()) {
      ICHECK(statement.same_as(persistent_->ClosedOwner()))
          << "Persistent pipeline private source/occurrence mismatch";
      auto task = std::make_shared<TaskNode>(MakeSyncExtraInfo());
      task->stmt = statement;
      task->outer_ctx = std::move(context);
      task->SetGuards(CloneGuards(guards));
      task->SetPipeMask(PipeMask(ResourcePipe::kScalar));
      task->SetReadVars(persistent_->ReadVars());
      for (const BufferRegion &region : persistent_->Reads())
        task->AddReadRegion(region);
      for (const BufferRegion &region : persistent_->Writes())
        task->AddWriteRegion(region);
      task_owners_[task.get()] = SelectResource(explicit_resource, kCoreVector);
      all_tasks_.push_back(task);
      Record(task, statement, occurrence, siblings, parent);
      result_->export_state->closed_replacements.emplace(
          occurrence, persistent_->ClosedReplacement());
      // Export visits the original subtree before replacing it. Count every
      // syntactic occurrence, including repeated Stmt object references.
      occurrence_ += StatementCounter().Run(statement) - 1;
      if (root_chain)
        result_->export_state->root_scope_occurrence = occurrence;
      return;
    }
    if (const auto *allocate = statement.as<AllocateNode>()) {
      RequirePure(allocate->condition, "allocation condition");
      for (const PrimExpr &extent : allocate->extents)
        RequirePure(extent, "allocation extent");
      Import(allocate->body, siblings, parent, std::move(context),
             std::move(guards), explicit_resource, root_chain,
             std::move(clocks));
      return;
    }
    if (const auto *declaration = statement.as<DeclBufferNode>()) {
      Import(declaration->body, siblings, parent, std::move(context),
             std::move(guards), explicit_resource, root_chain,
             std::move(clocks));
      return;
    }
    if (const auto *binding = statement.as<LetStmtNode>()) {
      if (result_->export_state->mutable_bindings.count(binding->var)) {
        auto task = std::make_shared<TaskNode>(MakeSyncExtraInfo());
        task->stmt = Evaluate(binding->value);
        task->outer_ctx = context;
        task->SetGuards(CloneGuards(guards));
        LeafEffects effects(task.get(), result_->physical_storage.get(), clocks,
                            composites_, tail_copies_, logic_buffer_shapes_,
                            target_->GetAttr<String>("model").value_or("") ==
                                "pto");
        effects.Analyze(task->stmt);
        task->SetWriteVars({binding->var});
        task_owners_[task.get()] =
            SelectResource(explicit_resource, effects.InferredResource());
        all_tasks_.push_back(task);
        Record(task, statement, occurrence, siblings, parent);
        result_->export_state->let_value_tasks.insert(task.get());
        // The runtime Let defines a stable snapshot. Do not bind it to a
        // future mutable read in proof contexts: the value remains unknown.
      } else {
        RequirePure(binding->value, "LetStmt definition");
        context.AddConstr(binding->var, binding->value);
      }
      Import(binding->body, siblings, parent, std::move(context),
             std::move(guards), explicit_resource, root_chain,
             std::move(clocks));
      return;
    }
    if (const auto *attribute = statement.as<AttrStmtNode>()) {
      ICHECK(attribute->attr_key != "tl.ascend_per_core_task" &&
             attribute->attr_key != "tl.ascend_task" &&
             attribute->attr_key != "tl.schedule_unit")
          << "Scheduled task/subcore metadata requires the later scheduled-TIR "
             "adapter";
      if (attribute->attr_key == "resource_scope") {
        const auto *scope = attribute->value.as<IntImmNode>();
        ICHECK(scope != nullptr && (scope->value == 0 || scope->value == 1));
        explicit_resource = scope->value == 0 ? kCoreCube : kCoreVector;
      } else {
        RequirePure(attribute->value, "attribute value");
        if (const auto *expression = attribute->node.as<PrimExprNode>())
          RequirePure(GetRef<PrimExpr>(expression), "attribute node");
      }
      if (attribute->attr_key == "thread_extent" ||
          attribute->attr_key == "virtual_thread") {
        const IterVar iteration = Downcast<IterVar>(attribute->node);
        context.AddConstr(
            iteration->var,
            Range::FromMinExtent(make_zero(attribute->value.dtype()),
                                 attribute->value));
      } else if ((attribute->attr_key == "tilelang_assume" ||
                  attribute->attr_key == "tl.assume") &&
                 attribute->node.as<PrimExprNode>()) {
        context.AddConstr(Downcast<PrimExpr>(attribute->node), true);
      } else if (attribute->attr_key == kPhysicalRingClock) {
        const Var logical_id = PhysicalRingClockId(attribute->node);
        result_->physical_storage->ValidateRingClock(logical_id,
                                                     attribute->value, context);
        clocks.Set(logical_id, attribute->value);
      }
      Import(attribute->body, siblings, parent, std::move(context),
             std::move(guards), explicit_resource, root_chain,
             std::move(clocks));
      return;
    }
    if (root_chain)
      result_->export_state->root_scope_occurrence = occurrence;
    if (const auto *sequence = statement.as<SeqStmtNode>()) {
      for (const Stmt &child : sequence->seq) {
        Import(child, siblings, parent, context, guards, explicit_resource,
               false, clocks);
        // Import still counts the original assumption's occurrence so export
        // anchors stay aligned. Its fact applies only to later siblings in
        // this lexical sequence, not to an earlier or enclosing statement.
        if (auto predicate = LegacyAssumption(child))
          context.AddConstr(predicate.value(), true);
      }
      return;
    }
    if (const auto *branch = statement.as<IfThenElseNode>()) {
      ICHECK(result_->manual_buffer_versions.empty())
          << "Physical ring synchronization currently requires guard-free "
             "tasks";
      RequirePure(branch->condition, "IfThenElse predicate");
      ICHECK(!branch->else_case.defined())
          << "Legacy If/else must be normalized into independently anchored "
             "optional controls before synchronization";
      Var iteration("__ascend_if_proxy_" +
                        std::to_string(next_conditional_loop_++),
                    DataType::Int(32));
      PrimExpr minimum = make_zero(iteration.dtype());
      PrimExpr extent =
          Select(branch->condition, make_const(iteration.dtype(), 1),
                 make_zero(iteration.dtype()));
      auto control = std::make_shared<ControlNode>(MakeSyncExtraInfo());
      control->control =
          For(iteration, minimum, extent, ForKind::kSerial, Evaluate(0));
      SetSyncOptionalControl(control.get());
      control->SetGuards(CloneGuards(guards));
      control->task = std::make_shared<TaskNode>(MakeSyncExtraInfo());
      control->task->stmt = Evaluate(0);
      // In particular, do NOT assume this If's condition here: entering
      // the proxy loop is optional, so sync sites must remain outside it
      // unless its nonempty condition is independently established.
      control->task->outer_ctx = context;
      control->task->SetPipeMask(PipeMask(ResourcePipe::kScalar));
      control->task->SetParent(control.get());
      control->task->SetWriteVars({iteration});
      const auto reads = UndefinedVars(branch->condition);
      control->task->SetReadVars(std::vector<Var>(reads.begin(), reads.end()));
      all_tasks_.push_back(control->task);
      Record(control, statement, occurrence, siblings, parent);
      result_->export_state->conditional_loop_substitutions.Set(iteration,
                                                                minimum);
      context.AddConstr(branch->condition);
      context.AddConstr(iteration, Range::FromMinExtent(minimum, extent));
      context.AddConstr(extent > make_zero(extent.dtype()));
      // The control node represents the If itself. Adding its condition
      // as a leaf guard as well would reintroduce mismatched flat guards.
      Import(branch->then_case, &control->children, control.get(), context,
             guards, explicit_resource, false, std::move(clocks));
      control->InitializeLoopBodyContext();
      return;
    }
    if (const auto *loop = statement.as<ForNode>()) {
      ICHECK(loop->kind == ForKind::kSerial || loop->kind == ForKind::kUnrolled)
          << "Legacy sync import requires serial or unrolled loops";
      RequirePure(loop->min, "For minimum");
      RequirePure(loop->extent, "For extent");
      ICHECK(result_->manual_buffer_versions.empty() ||
             (loop->min.as<IntImmNode>() && loop->extent.as<IntImmNode>()) ||
             (persistent_ != nullptr &&
              persistent_->AllowsOuterLoop(GetRef<For>(loop))) ||
             (pipeline_case_ != nullptr &&
              pipeline_case_->AllowsSteadyLoop(GetRef<For>(loop), context)))
          << "Physical ring synchronization currently requires static For "
             "bounds";
      for (const auto &[key, value] : loop->annotations) {
        const std::string name = key;
        ICHECK(name != "num_stages" && name.rfind("tl_pipeline_", 0) != 0 &&
               name.rfind("software_pipeline_", 0) != 0 &&
               name != kMultiBufferEligible)
            << "Upstream legacy sync cannot recover pipeline annotation "
            << key;
      }
      auto control = std::make_shared<ControlNode>(MakeSyncExtraInfo());
      control->control = GetRef<For>(loop);
      if (composites_.static_cube_pipeline &&
          composites_.cube_pipeline_loop.defined() &&
          loop->loop_var.same_as(composites_.cube_pipeline_loop.value()))
        SetSyncStaticCubePipelineControl(control.get());
      control->SetGuards(CloneGuards(guards));
      control->task = std::make_shared<TaskNode>(MakeSyncExtraInfo());
      control->task->stmt = Evaluate(0);
      control->task->outer_ctx = context;
      control->task->SetPipeMask(PipeMask(ResourcePipe::kScalar));
      control->task->SetParent(control.get());
      control->task->SetWriteVars({loop->loop_var});
      auto reads = UndefinedVars(loop->min);
      for (const Var &var : UndefinedVars(loop->extent))
        reads.push_back(var);
      control->task->SetReadVars(std::vector<Var>(reads.begin(), reads.end()));
      all_tasks_.push_back(control->task);
      Record(control, statement, occurrence, siblings, parent);
      context.AddConstr(loop->loop_var,
                        Range::FromMinExtent(loop->min, loop->extent));
      context.AddConstr(loop->extent > 0);
      Import(loop->body, &control->children, control.get(), context, guards,
             explicit_resource, false, std::move(clocks));
      control->InitializeLoopBodyContext();
      return;
    }
    if (const auto *evaluate = statement.as<EvaluateNode>()) {
      if (LegacyAssumption(statement))
        return; // Compile-time information; preserve its original AST shell.
      if (evaluate->value.as<IntImmNode>())
        return;
    }
    ICHECK(statement.as<EvaluateNode>() || statement.as<BufferStoreNode>())
        << "Unsupported statement in upstream Ascend sync legacy import: "
        << statement->GetTypeKey();
    auto task = std::make_shared<TaskNode>(MakeSyncExtraInfo());
    task->stmt = statement;
    task->outer_ctx = std::move(context);
    task->SetGuards(CloneGuards(guards));
    LeafEffects effects(task.get(), result_->physical_storage.get(), clocks,
                        composites_, tail_copies_, logic_buffer_shapes_,
                        target_->GetAttr<String>("model").value_or("") ==
                            "pto");
    effects.Analyze(statement);
    task_owners_[task.get()] =
        SelectResource(explicit_resource, effects.InferredResource());
    all_tasks_.push_back(task);
    Record(task, statement, occurrence, siblings, parent);
  }

  ImportedLegacySyncIR *result_;
  std::vector<std::shared_ptr<TaskNode>> all_tasks_;
  Target target_;
  String platform_;
  bool ordinary_sync_;
  const LegacyCompositePlan &composites_;
  LegacyTailCopyPlan *tail_copies_;
  const LegacyPipelineCase *pipeline_case_;
  const LegacyPersistentPipelineImport *persistent_;
  Map<Var, Array<PrimExpr>> logic_buffer_shapes_;
  std::unordered_map<TaskNode *, CoreMask> task_owners_;
  CoreMask selected_resources_{kCoreUnassigned};
  size_t occurrence_{0};
  size_t next_conditional_loop_{0};
};

Stmt SequenceAround(const std::vector<Stmt> &before, Stmt body,
                    const std::vector<Stmt> &after) {
  if (before.empty() && after.empty())
    return body;
  Array<Stmt> sequence;
  for (const Stmt &statement : before)
    sequence.push_back(statement);
  sequence.push_back(std::move(body));
  for (const Stmt &statement : after)
    sequence.push_back(statement);
  return SeqStmt(sequence);
}

class LegacyExporter : public StmtExprMutator {
public:
  explicit LegacyExporter(const ImportedLegacySyncIR &imported)
      : imported_(imported) {}

  PrimFunc Run(Stmt *synchronized_scope) {
    synchronized_scope_ = synchronized_scope;
    Collect(imported_.roots);
    PrimFunc result = imported_.original_function;
    result.CopyOnWrite()->body = VisitStmt(result->body);
    return result;
  }

private:
  static Stmt InResourceScope(const Stmt &statement, CoreMask core) {
    ICHECK(core == kCoreCube ||
           (core != kCoreUnassigned && (core & kCoreVector) == core));
    return AttrStmt(make_zero(DataType::Int(32)), "resource_scope",
                    Integer(core == kCoreCube ? 0 : 1), statement);
  }

  void Collect(const std::vector<std::shared_ptr<IRStructure>> &nodes) {
    for (const auto &node : nodes) {
      if (auto placement = GetSyncPlacement(node.get())) {
        Stmt statement =
            Substitute(placement->statement,
                       imported_.export_state->conditional_loop_substitutions);
        statement = InResourceScope(statement, placement->core);
        if (placement->anchor == nullptr) {
          (placement->at_beginning ? prefix_ : suffix_).push_back(statement);
        } else {
          auto origin = imported_.export_state->original_occurrences.find(
              placement->anchor);
          ICHECK(origin != imported_.export_state->original_occurrences.end())
              << "Sync placement has no original legacy TIR anchor";
          if (!placement->at_beginning &&
              imported_.export_state->let_value_tasks.count(
                  placement->anchor)) {
            after_let_value_[origin->second].push_back(statement);
          } else {
            (placement->at_beginning ? before_ : after_)[origin->second]
                .push_back(statement);
          }
        }
      } else {
        auto origin =
            imported_.export_state->original_statements.find(node.get());
        ICHECK(origin != imported_.export_state->original_statements.end())
            << "Upstream synchronization created an unanchored non-sync node";
        if (node->IsTask()) {
          const auto *task = static_cast<TaskNode *>(node.get());
          if (imported_.export_state->let_value_tasks.count(node.get())) {
            const auto *binding = origin->second.as<LetStmtNode>();
            const auto *definition = task->stmt.as<EvaluateNode>();
            ICHECK(binding != nullptr && definition != nullptr &&
                   definition->value.same_as(binding->value) &&
                   task->GetWriteVars().size() == 1 &&
                   task->GetWriteVars()[0].same_as(binding->var))
                << "Synchronization must retain the scalar snapshot definition";
          } else {
            ICHECK(task->stmt.same_as(origin->second))
                << "Synchronization must not rewrite original legacy "
                   "computation";
          }
        }
      }
      if (node->IsControl())
        Collect(static_cast<ControlNode *>(node.get())->children);
    }
  }

  Stmt VisitStmt(const Stmt &statement) final {
    const size_t occurrence = ++occurrence_;
    Stmt rewritten = StmtExprMutator::VisitStmt(statement);
    auto closed = imported_.export_state->closed_replacements.find(occurrence);
    if (closed != imported_.export_state->closed_replacements.end())
      rewritten = closed->second;
    auto after_value = after_let_value_.find(occurrence);
    if (after_value != after_let_value_.end()) {
      ICHECK(rewritten.as<LetStmtNode>() != nullptr);
      LetStmt binding = Downcast<LetStmt>(rewritten);
      binding.CopyOnWrite()->body =
          SequenceAround(after_value->second, binding->body, {});
      rewritten = std::move(binding);
    }
    rewritten =
        SequenceAround(before_[occurrence], rewritten, after_[occurrence]);
    if (occurrence == imported_.export_state->root_scope_occurrence)
      rewritten = SequenceAround(prefix_, rewritten, suffix_);
    if (synchronized_scope_ != nullptr &&
        occurrence == imported_.export_state->root_scope_occurrence)
      *synchronized_scope_ = rewritten;
    return rewritten;
  }

  const ImportedLegacySyncIR &imported_;
  std::unordered_map<size_t, std::vector<Stmt>> before_, after_,
      after_let_value_;
  std::vector<Stmt> prefix_, suffix_;
  size_t occurrence_{0};
  Stmt *synchronized_scope_{nullptr};
};

} // namespace

ConflictOracle ImportedLegacySyncIR::MakeOracle() const {
  ICHECK(physical_storage != nullptr);
  return physical_storage->MakeOracle();
}

ImportedLegacySyncIR
ImportLegacySync(const PrimFunc &function, const Target &target,
                 const String &platform, bool ordinary_sync,
                 const LegacyPipelineCase *pipeline_case,
                 const LegacyPersistentPipelineImport *persistent) {
  if (persistent != nullptr)
    ICHECK(persistent->MatchesSource(function))
        << "Persistent pipeline private source/body mismatch";
  LegacySnapshotResult normalized;
  if (persistent != nullptr)
    normalized.function = function;
  else
    normalized =
        NormalizeLegacySnapshots(NormalizeLegacyCompositeBranches(function));
  ManualFlagValidator().Run(normalized.function->body);
  ImportedLegacySyncIR imported;
  imported.original_function = std::move(normalized.function);
  imported.physical_storage = persistent != nullptr
                                  ? persistent->Storage()
                                  : std::make_shared<PhysicalStorageResolver>(
                                        imported.original_function);
  imported.export_state = std::make_shared<LegacyExportState>();
  imported.export_state->mutable_bindings =
      std::move(normalized.mutable_bindings);
  const auto composites =
      AnalyzeLegacyComposites(imported.original_function, target, platform);
  LegacyTailCopyPlan tail_copies(imported.original_function, target, platform);
  LegacyImporter(&imported, target, platform, ordinary_sync, composites,
                 &tail_copies, pipeline_case, persistent)
      .Run();
  return imported;
}

PrimFunc ExportLegacySync(const ImportedLegacySyncIR &imported,
                          Stmt *synchronized_scope) {
  ICHECK(imported.export_state != nullptr);
  return LegacyExporter(imported).Run(synchronized_scope);
}

} // namespace tvm::tl::ascend_sync
