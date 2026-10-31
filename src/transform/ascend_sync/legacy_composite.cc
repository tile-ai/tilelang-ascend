// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

#include "legacy_composite.h"

#include <tvm/ir/type.h>
#include <tvm/node/structural_equal.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <limits>
#include <unordered_set>
#include <utility>

#include "../common/attr.h"
#include "../common/pipeline_ring_provenance.h"
#include "legacy_helper_type.h"
#include "upstream/constraints.h"

namespace tvm::tl::ascend_sync {

using namespace tir;

namespace {

std::string Trim(const std::string &text) {
  const size_t first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return "";
  const size_t last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

std::string OperationName(const CallNode *call) {
  std::string name;
  if (call->op.same_as(builtin::call_extern()) ||
      call->op.same_as(builtin::call_pure_extern())) {
    if (!call->args.empty()) {
      if (const auto *symbol = call->args[0].as<StringImmNode>())
        name = symbol->value;
    }
  } else if (const auto *op = call->op.as<OpNode>()) {
    name = op->name;
  }
  if (const size_t angle = name.find('<'); angle != std::string::npos)
    name.erase(angle);
  for (const std::string prefix : {"tl::ascend::", "tl.", "ascend_"}) {
    if (name.rfind(prefix, 0) == 0)
      name.erase(0, prefix.size());
  }
  return name;
}

std::string StorageScope(const Var &data) {
  const auto *pointer = data->type_annotation.as<PointerTypeNode>();
  return pointer == nullptr ? "" : std::string(pointer->storage_scope);
}

bool IsUserL0AB(const Var &data) {
  const std::string scope = StorageScope(data);
  return scope == "wmma.matrix_a" || scope == "wmma.matrix_b" ||
         scope == "shared.l0a" || scope == "shared.l0b" ||
         scope == "shared.l0a.dyn" || scope == "shared.l0b.dyn" ||
         scope == "shared.l0a.sf" || scope == "shared.l0b.sf";
}

bool IsPipelineKey(const String &key) {
  const std::string name = key;
  return name == "num_stages" || name.rfind("software_pipeline_", 0) == 0 ||
         name.rfind("tl_pipeline_", 0) == 0;
}

int64_t PositiveInteger(const std::string &text) {
  const std::string token = Trim(text);
  uint64_t value = 0;
  const auto parsed =
      std::from_chars(token.data(), token.data() + token.size(), value);
  ICHECK(!token.empty() && parsed.ec == std::errc() &&
         parsed.ptr == token.data() + token.size() && value > 0 &&
         value <= std::numeric_limits<uint32_t>::max())
      << "Composite gemm_v0 requires static positive M/N/K/kL0Size: " << text;
  return static_cast<int64_t>(value);
}

DataType ElementType(const std::string &spelling) {
  const std::string token = CanonicalHelperElementType(spelling);
  if (token == "half")
    return DataType::Float(16);
  if (token == "bfloat16_t")
    return DataType::BFloat(16);
  if (token == "float")
    return DataType::Float(32);
  if (token == "int8_t")
    return DataType::Int(8);
  if (token == "int32_t")
    return DataType::Int(32);
  LOG(FATAL) << "Composite gemm_v0 has an unsupported element type: " << token;
  return DataType::Void();
}

std::vector<std::string> TemplateArguments(const CallNode *call) {
  ICHECK_EQ(call->args.size(), 6U)
      << "Malformed typed composite gemm_v0 arguments";
  const auto *symbol = call->args[0].as<StringImmNode>();
  ICHECK(symbol != nullptr)
      << "Composite gemm_v0 requires a literal helper template";
  const std::string name = Trim(symbol->value);
  ICHECK(name.rfind("gemm_v0<", 0) == 0 && name.back() == '>')
      << "Composite gemm_v0 helper template does not match its typed op";
  const size_t begin = name.find('<') + 1;
  const size_t end = name.size() - 1;
  std::vector<std::string> arguments;
  size_t position = begin;
  while (position <= end) {
    const size_t comma = name.find(',', position);
    const size_t next = comma == std::string::npos ? end : comma;
    arguments.push_back(Trim(name.substr(position, next - position)));
    position = next + 1;
  }
  ICHECK_EQ(arguments.size(), 8U)
      << "Composite gemm_v0 requires its eight canonical template arguments";
  for (size_t index : {5U, 6U})
    ICHECK(arguments[index] == "true" || arguments[index] == "false")
        << "Composite gemm_v0 transpose flags must be static booleans";
  return arguments;
}

LegacyCompositeOperation AnalyzeOperation(const Call &call) {
  const auto arguments = TemplateArguments(call.get());
  LegacyCompositeOperation result;
  result.call = call;
  result.m = PositiveInteger(arguments[2]);
  result.n = PositiveInteger(arguments[3]);
  result.k = PositiveInteger(arguments[4]);
  result.k_l0 = PositiveInteger(arguments[7]);
  ICHECK(result.k_l0 % 16 == 0 && result.k_l0 <= 4095)
      << "Composite gemm_v0 kL0Size must be 16-aligned and at most 4095";
  const DataType input = ElementType(arguments[0]);
  const DataType output = ElementType(arguments[1]);
  ICHECK((input == DataType::Int(8) && output == DataType::Int(32)) ||
         ((input == DataType::Float(16) || input == DataType::BFloat(16) ||
           input == DataType::Float(32)) &&
          output == DataType::Float(32)))
      << "Composite gemm_v0 input/accumulator dtype combination is unsupported";
  const auto *n_actual = call->args[5].as<IntImmNode>();
  ICHECK(n_actual != nullptr && n_actual->value == result.n)
      << "Composite gemm_v0 currently requires static n_actual == N";
  ICHECK(call->args[4].dtype().is_scalar() &&
         SideEffect(call->args[4]) <= CallEffectKind::kPure)
      << "Composite gemm_v0 clear condition must be a pure scalar";
  const auto *clear = call->args[4].as<IntImmNode>();
  const bool clears_accumulator = clear != nullptr && clear->value == 1;

  for (size_t index = 1; index <= 3; ++index) {
    const auto *pointer = call->args[index].as<CallNode>();
    ICHECK(pointer != nullptr &&
           pointer->op.same_as(builtin::tvm_access_ptr()) &&
           pointer->args.size() == 5U)
        << "Composite gemm_v0 requires canonical A/B/C access_ptr operands";
    const auto *data = pointer->args[1].as<VarNode>();
    ICHECK(data != nullptr)
        << "Composite gemm_v0 requires explicit buffer Var identities";
    const std::string scope = StorageScope(GetRef<Var>(data));
    ICHECK(index <= 2 ? scope == "shared.l1" : scope == "wmma.accumulator")
        << "Composite gemm_v0 requires A/B in L1 and C in L0C";
    const auto *annotation = pointer->args[0].as<CallNode>();
    ICHECK(annotation != nullptr &&
           annotation->op.same_as(Op::Get("tir.type_annotation")) &&
           annotation->dtype == (index <= 2 ? input : output))
        << "Composite gemm_v0 pointer dtype disagrees with its helper template";
    const auto *mask = pointer->args[4].as<IntImmNode>();
    ICHECK(mask != nullptr && mask->value > 0 && mask->value <= 3)
        << "Composite gemm_v0 requires literal valid access masks";
    ICHECK(SideEffect(pointer->args[2]) <= CallEffectKind::kPure &&
           SideEffect(pointer->args[3]) <= CallEffectKind::kPure)
        << "Composite gemm_v0 pointer offset/extent must be pure";
    bool read = (mask->value & 1) != 0 || index <= 2 ||
                (index == 3 && !clears_accumulator);
    bool write = (mask->value & 2) != 0 || index == 3;
    result.buffers.push_back({GetRef<Call>(pointer), read, write});
  }
  return result;
}

void ValidateAllocationFootprints(LegacyCompositeOperation &operation,
                                  const PrimFunc &function,
                                  const PhysicalRingProvenance &rings = {}) {
  const auto sizes = function->GetAttr<Map<Var, PrimExpr>>("size_map");
  ICHECK(sizes.defined())
      << "Composite gemm_v0 requires final allocation size_map metadata";
  const int64_t rows[] = {operation.m, operation.k, operation.m};
  const int64_t columns[] = {operation.k, operation.n, operation.n};
  for (size_t index = 0; index < operation.buffers.size(); ++index) {
    const Call &pointer = operation.buffers[index].access_ptr;
    const auto *offset = pointer->args[2].as<IntImmNode>();
    const auto *extent = pointer->args[3].as<IntImmNode>();
    ICHECK_LE(rows[index],
              std::numeric_limits<int64_t>::max() / columns[index]);
    ICHECK(extent != nullptr && extent->value == rows[index] * columns[index])
        << "Composite gemm_v0 access extent disagrees with its static shape";
    const DataType dtype = pointer->args[0].dtype();
    const int64_t bytes = dtype.bits() / 8;
    // A conservative upper bound covers both zN and nZ orientation, plus
    // L0C's fractal padding. This first contract may reject tighter legal
    // layouts; it never shrinks an opaque helper to its logical pointer span.
    const int64_t quantum = std::max<int64_t>(16, 32 / bytes);
    const int64_t padded_rows = (rows[index] + quantum - 1) / quantum * quantum;
    const int64_t padded_columns =
        (columns[index] + quantum - 1) / quantum * quantum;
    ICHECK_LE(padded_rows,
              std::numeric_limits<int64_t>::max() / padded_columns / bytes);
    const int64_t required_bytes = padded_rows * padded_columns * bytes;
    const Var data = Downcast<Var>(pointer->args[1]);
    auto &effect = operation.buffers[index];
    effect.required_bytes = required_bytes;
    if (index < 2 && !rings.empty()) {
      for (const auto &[id, fields] : rings) {
        ValidatePhysicalRingRecord(fields);
        if (!fields[kPhysicalRingRoot].same_as(data))
          continue;
        ICHECK(!effect.logical_ring.defined())
            << "Cube pipeline does not support multiple rings in one root";
        const auto *stride = fields[kPhysicalRingStrideBits].as<IntImmNode>();
        ICHECK(stride != nullptr && stride->value > 0 &&
               stride->value % 8 == 0 && required_bytes <= stride->value / 8)
            << "Cube pipeline GEMM physical footprint exceeds its ring slot";
        effect.logical_ring = id;
      }
      if (effect.logical_ring.defined())
        continue; // Clock/base/offset validation uses the real task context.
      for (const char *name : {"buffer_versions", "tl.buffer_versions_map"}) {
        if (auto versions = function->GetAttr<Map<Var, PrimExpr>>(name)) {
          if (auto version = versions.value().Get(data)) {
            const auto *constant = version.value().as<IntImmNode>();
            ICHECK(constant && constant->value == 1)
                << "Cube pipeline operand requires matching ring provenance";
          }
        }
      }
    }
    ICHECK(offset != nullptr && offset->value == 0)
        << "Composite gemm_v0 currently requires base allocation pointers; "
           "subviews/rings need a separate physical footprint contract";
    const auto size = sizes.value().Get(data);
    const auto *constant_size =
        size.defined() ? size.value().as<IntImmNode>() : nullptr;
    ICHECK(constant_size != nullptr && constant_size->value >= required_bytes)
        << "Composite gemm_v0 allocation does not cover its padded physical "
           "footprint: "
        << data << " requires at least " << required_bytes << " bytes";
  }
}

bool IsKLoopBarrier(const CallNode *call) {
  const bool manual = call->op.same_as(Op::Get("tl.ascend_pipe_barrier"));
  const bool automatic = call->op.same_as(Op::Get("tl.ascend_auto_barrier"));
  if (!manual && !automatic)
    return false;
  const char *diagnostic =
      "Composite gemm_v0 external For barrier requires one literal Cube "
      "pipeline";
  ICHECK_EQ(call->args.size(), 1U) << diagnostic;
  const auto *literal = call->args[0].as<StringImmNode>();
  ICHECK(literal != nullptr) << diagnostic;
  std::string pipe = literal->value;
  if (automatic) {
    ICHECK(pipe.rfind("PIPE_", 0) == 0) << diagnostic;
    pipe.erase(0, 5);
  }
  static const std::unordered_set<std::string> cube_pipes{
      "ALL", "MTE1", "MTE2", "MTE3", "M", "FIX", "S"};
  ICHECK(cube_pipes.count(pipe)) << diagnostic;
  return true;
}

// Do not flatten or rewrite executable control. Existing, validated barriers
// remain in place, separately from the three calculation leaves we audit.
void CollectKLoopLeaves(const Stmt &statement, int resource,
                        std::vector<Call> *leaves) {
  const char *diagnostic =
      "Composite gemm_v0 external For contract requires only full A/B "
      "GM-to-L1 copies followed by one gemm; loop-internal FIX, nested control "
      "and other operations require separate validation";
  if (const auto *sequence = statement.as<SeqStmtNode>()) {
    for (const Stmt &child : sequence->seq)
      CollectKLoopLeaves(child, resource, leaves);
  } else if (const auto *binding = statement.as<LetStmtNode>()) {
    ICHECK(SideEffect(binding->value) <= CallEffectKind::kPure) << diagnostic;
    CollectKLoopLeaves(binding->body, resource, leaves);
  } else if (const auto *attribute = statement.as<AttrStmtNode>()) {
    ICHECK(attribute->attr_key == "resource_scope") << diagnostic;
    const auto *owner = attribute->value.as<IntImmNode>();
    ICHECK(owner != nullptr && owner->value == 0) << diagnostic;
    CollectKLoopLeaves(attribute->body, 0, leaves);
  } else if (const auto *evaluation = statement.as<EvaluateNode>()) {
    if (const auto *zero = evaluation->value.as<IntImmNode>()) {
      ICHECK_EQ(zero->value, 0) << diagnostic;
      return;
    }
    ICHECK(resource == -1 || resource == 0) << diagnostic;
    const auto *call = evaluation->value.as<CallNode>();
    ICHECK(call != nullptr) << diagnostic;
    if (IsKLoopBarrier(call))
      return;
    leaves->push_back(GetRef<Call>(call));
  } else {
    LOG(FATAL) << diagnostic;
  }
}

void ValidateKLoopCopy(const Call &copy, const Call &gemm_pointer,
                       const std::string &element_type, int64_t rows,
                       int64_t columns, bool physical_ring = false) {
  const char *diagnostic =
      "Composite gemm_v0 external For requires full base-allocation A/B "
      "GM-to-L1 refreshes before each gemm";
  ICHECK(copy->op.same_as(builtin::call_extern()) && copy->args.size() == 8U &&
         OperationName(copy.get()) == "copy_gm_to_l1")
      << diagnostic;
  const auto *symbol = copy->args[0].as<StringImmNode>();
  ICHECK(symbol != nullptr) << diagnostic;
  std::string compact = symbol->value;
  compact.erase(std::remove_if(compact.begin(), compact.end(),
                               [](unsigned char c) { return std::isspace(c); }),
                compact.end());
  ICHECK_EQ(compact, "tl::ascend::copy_gm_to_l1<" + element_type + "," +
                         std::to_string(rows) + "," + std::to_string(columns) +
                         ">")
      << diagnostic;
  const auto *source = copy->args[1].as<CallNode>();
  const auto *destination = copy->args[2].as<CallNode>();
  for (const CallNode *pointer : {source, destination}) {
    ICHECK(pointer != nullptr &&
           pointer->op.same_as(builtin::tvm_access_ptr()) &&
           pointer->args.size() == 5U)
        << diagnostic;
    const auto *extent = pointer->args[3].as<IntImmNode>();
    ICHECK(extent != nullptr && extent->value == rows * columns) << diagnostic;
    ICHECK(pointer->args[0].dtype() == gemm_pointer->args[0].dtype())
        << diagnostic;
  }
  const auto *source_data = source->args[1].as<VarNode>();
  ICHECK(source_data != nullptr) << diagnostic;
  const std::string source_scope = StorageScope(GetRef<Var>(source_data));
  ICHECK(source_scope.empty() || source_scope == "global") << diagnostic;
  ICHECK(destination->args[1].same_as(gemm_pointer->args[1])) << diagnostic;
  const auto *offset = destination->args[2].as<IntImmNode>();
  ICHECK(physical_ring || (offset != nullptr && offset->value == 0))
      << diagnostic;
  for (const auto &[pointer, mask] :
       {std::pair<const CallNode *, int>{source, 1}, {destination, 2}}) {
    const auto *literal = pointer->args[4].as<IntImmNode>();
    ICHECK(literal != nullptr && literal->value == mask) << diagnostic;
  }
  // These are the real/full row and column masks emitted by AscendCopy. The
  // stride may vary with the original GM tensor, but every tile is complete.
  for (size_t argument : {4U, 6U}) {
    const auto *mask = copy->args[argument].as<IntImmNode>();
    ICHECK(mask != nullptr && mask->value == rows) << diagnostic;
  }
  for (size_t argument : {5U, 7U}) {
    const auto *mask = copy->args[argument].as<IntImmNode>();
    ICHECK(mask != nullptr && mask->value == columns) << diagnostic;
  }
}

constexpr const char *kPtoProfile = "PTO flat gemm_v0 contract";

int64_t PtoProduct(int64_t a, int64_t b) {
  ICHECK(a > 0 && b > 0 && a <= std::numeric_limits<int64_t>::max() / b)
      << kPtoProfile << ": physical footprint overflows int64";
  return a * b;
}

void PtoConstant(const PrimExpr &value, int64_t expected) {
  const auto *literal = value.as<IntImmNode>();
  ICHECK(literal && literal->value == expected)
      << kPtoProfile << ": operand disagrees with its semantic descriptor";
}

void PtoPure(const PrimExpr &value) {
  ICHECK(value.defined() && SideEffect(value) <= CallEffectKind::kPure)
      << kPtoProfile << ": mutable or opaque expressions are unsupported";
}

PrimExpr PtoInteger(const PrimExpr &value, ProofAnalyzer *analyzer) {
  PtoPure(value);
  const DataType dtype = value.dtype();
  ICHECK(dtype.is_scalar() && !dtype.is_bool() &&
         ((dtype.is_int() && dtype.bits() <= 64) ||
          (dtype.is_uint() && dtype.bits() <= 32)))
      << kPtoProfile << ": requires a value-preserving scalar index";
  ICHECK(!ContainsUnsignedWrappingArithmetic(value))
      << kPtoProfile << ": wrapping unsigned index arithmetic is unsupported";
  // Prove intermediate signed arithmetic representable before using
  // mathematical address bounds. This prevents overflow/cancellation from
  // manufacturing a seemingly in-range final pointer.
  PostOrderVisit(value, [&](const ObjectRef &node) {
    const auto *expression = node.as<PrimExprNode>();
    if (!expression || expression->dtype.is_bool() ||
        (!expression->dtype.is_int() && !expression->dtype.is_uint()))
      return;
    PrimExpr divisor;
    if (const auto *operation = node.as<FloorDivNode>())
      divisor = operation->b;
    else if (const auto *operation = node.as<DivNode>())
      divisor = operation->b;
    else if (const auto *operation = node.as<FloorModNode>())
      divisor = operation->b;
    else if (const auto *operation = node.as<ModNode>())
      divisor = operation->b;
    if (divisor.defined())
      ICHECK(analyzer->CanProve(divisor != make_zero(divisor.dtype())))
          << kPtoProfile << ": index division requires a nonzero divisor";
    const int bits = expression->dtype.bits();
    const bool unsigned_type = expression->dtype.is_uint();
    ICHECK_LE(bits, unsigned_type ? 32 : 64) << kPtoProfile;
    const int64_t minimum =
        unsigned_type ? 0
                      : (bits == 64 ? std::numeric_limits<int64_t>::min()
                                    : -(int64_t(1) << (bits - 1)));
    const int64_t maximum =
        unsigned_type ? (int64_t(1) << bits) - 1
                      : (bits == 64 ? std::numeric_limits<int64_t>::max()
                                    : (int64_t(1) << (bits - 1)) - 1);
    if (const auto *conversion = node.as<CastNode>()) {
      const DataType source_type = conversion->value.dtype();
      ICHECK(source_type.is_scalar() &&
             ((source_type.is_int() && source_type.bits() <= 64) ||
              (source_type.is_uint() && source_type.bits() <= 32)))
          << kPtoProfile << ": index casts require integer source values";
      const PrimExpr source = cast(DataType::Int(64), conversion->value);
      ICHECK(analyzer->CanProve(source >= IntImm(DataType::Int(64), minimum) &&
                                source <= IntImm(DataType::Int(64), maximum)))
          << kPtoProfile << ": lossy index casts are unsupported";
    }
    const PrimExpr wide = cast(DataType::Int(64), GetRef<PrimExpr>(expression));
    ICHECK(analyzer->CanProve(wide >= IntImm(DataType::Int(64), minimum) &&
                              wide <= IntImm(DataType::Int(64), maximum)))
        << kPtoProfile << ": cannot prove nonwrapping signed index arithmetic";
  });
  return cast(DataType::Int(64), value);
}

void PtoPredicate(const PrimExpr &predicate, const ConstrSet &context) {
  PtoPure(predicate);
  ICHECK(predicate.dtype().is_scalar() && predicate.dtype().is_bool())
      << kPtoProfile << ": assumptions require a pure scalar boolean predicate";
  ProofAnalyzer analyzer;
  context.Populate(analyzer);
  PostOrderVisit(predicate, [&](const ObjectRef &node) {
    const auto *expression = node.as<PrimExprNode>();
    if (expression && !expression->dtype.is_bool() &&
        (expression->dtype.is_int() || expression->dtype.is_uint()))
      PtoInteger(GetRef<PrimExpr>(expression), &analyzer);
  });
}

// Fold only a pure choice between the same GEMM invocation. Executable
// operands are retained; the output access mask is conservative metadata.
class CompositeBranchNormalizer : public StmtExprMutator {
  // Facts are lexical: a sibling assumption applies only to later statements,
  // and a loop/Let/thread binding never escapes its body.
  Stmt VisitStmt_(const ForNode *op) final {
    const ConstrSet saved = context_;
    if (SideEffect(op->min) <= CallEffectKind::kPure &&
        SideEffect(op->extent) <= CallEffectKind::kPure) {
      ProofAnalyzer analyzer;
      context_.Populate(analyzer);
      PtoInteger(op->min, &analyzer);
      PtoInteger(op->extent, &analyzer);
      context_.AddConstr(op->loop_var,
                         Range::FromMinExtent(op->min, op->extent));
    }
    Stmt result = StmtExprMutator::VisitStmt_(op);
    context_ = saved;
    return result;
  }

  Stmt VisitStmt_(const LetStmtNode *op) final {
    const ConstrSet saved = context_;
    if (op->value.dtype().is_scalar() &&
        (op->value.dtype().is_int() || op->value.dtype().is_uint()) &&
        !op->value.dtype().is_bool() &&
        SideEffect(op->value) <= CallEffectKind::kPure) {
      ProofAnalyzer analyzer;
      context_.Populate(analyzer);
      PtoInteger(op->value, &analyzer);
      context_.AddConstr(op->var, op->value);
    }
    Stmt result = StmtExprMutator::VisitStmt_(op);
    context_ = saved;
    return result;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    const ConstrSet saved = context_;
    if (op->attr_key == "thread_extent" || op->attr_key == "virtual_thread") {
      if (const auto *iteration = op->node.as<IterVarNode>()) {
        ProofAnalyzer analyzer;
        context_.Populate(analyzer);
        PtoInteger(op->value, &analyzer);
        context_.AddConstr(
            iteration->var,
            Range::FromMinExtent(make_zero(op->value.dtype()), op->value));
        context_.AddConstr(op->value > 0);
      }
    } else if ((op->attr_key == "tl.assume" ||
                op->attr_key == "tilelang_assume") &&
               op->node.as<PrimExprNode>()) {
      const PrimExpr predicate = Downcast<PrimExpr>(op->node);
      PtoPredicate(predicate, context_);
      context_.AddConstr(predicate, true);
    }
    Stmt result = StmtExprMutator::VisitStmt_(op);
    context_ = saved;
    return result;
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    const ConstrSet saved = context_;
    Array<Stmt> children;
    for (const Stmt &child : op->seq) {
      children.push_back(VisitStmt(child));
      const auto *evaluation = child.as<EvaluateNode>();
      const auto *call =
          evaluation ? evaluation->value.as<CallNode>() : nullptr;
      if (call && call->op.same_as(builtin::assume()) &&
          call->args.size() == 1U) {
        PtoPredicate(call->args[0], context_);
        context_.AddConstr(call->args[0], true);
      }
    }
    context_ = saved;
    return SeqStmt(children, op->span);
  }

  bool EquivalentPointer(const PrimExpr &left, const PrimExpr &right,
                         const PrimExpr &guard) const {
    StructuralEqual equal;
    if (equal(left, right))
      return true;
    const auto *a = left.as<CallNode>();
    const auto *b = right.as<CallNode>();
    if (!a || !b || !a->op.same_as(builtin::tvm_access_ptr()) ||
        !b->op.same_as(a->op) || a->dtype != b->dtype || a->args.size() != 5U ||
        b->args.size() != 5U || a->args[2].dtype() != b->args[2].dtype())
      return false;
    for (size_t index : {0U, 1U, 3U, 4U})
      if (!equal(a->args[index], b->args[index]))
        return false;
    PtoPredicate(guard, context_);
    ConstrSet guarded = context_;
    guarded.AddConstr(guard, true);
    ProofAnalyzer analyzer;
    guarded.Populate(analyzer);
    // The else pointer will execute on both paths after folding. Validate its
    // offset (and the original then offset) before proving guarded equality.
    const PrimExpr a_offset = PtoInteger(a->args[2], &analyzer);
    const PrimExpr b_offset = PtoInteger(b->args[2], &analyzer);
    return analyzer.CanProve(a_offset == b_offset);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    IfThenElse branch = Downcast<IfThenElse>(StmtExprMutator::VisitStmt_(op));
    if (!branch->else_case.defined() ||
        !branch->condition.dtype().is_scalar() ||
        !branch->condition.dtype().is_bool() ||
        SideEffect(branch->condition) > CallEffectKind::kPure)
      return branch;
    const auto *left = branch->then_case.as<EvaluateNode>();
    const auto *right = branch->else_case.value().as<EvaluateNode>();
    const auto *a = left ? left->value.as<CallNode>() : nullptr;
    const auto *b = right ? right->value.as<CallNode>() : nullptr;
    if (!a || !b || !a->op.same_as(Op::Get("tl.ascend_gemm_v0")) ||
        !b->op.same_as(a->op) || a->args.size() != 6 || b->args.size() != 6)
      return branch;
    StructuralEqual equal;
    for (size_t index : {0U, 5U})
      if (!equal(a->args[index], b->args[index]))
        return branch;
    for (size_t index : {1U, 2U})
      if (!EquivalentPointer(a->args[index], b->args[index], branch->condition))
        return branch;
    for (const PrimExpr &clear : {a->args[4], b->args[4]})
      if (!clear.dtype().is_scalar() || !clear.dtype().is_bool() ||
          SideEffect(clear) > CallEffectKind::kPure)
        return branch;
    const auto *ca = a->args[3].as<CallNode>();
    const auto *cb = b->args[3].as<CallNode>();
    if (!ca || !cb || !ca->op.same_as(builtin::tvm_access_ptr()) ||
        !cb->op.same_as(ca->op) || ca->dtype != cb->dtype ||
        ca->args.size() != 5 || cb->args.size() != 5)
      return branch;
    for (size_t index = 0; index < 4; ++index)
      if (!equal(ca->args[index], cb->args[index]))
        return branch;
    const auto *ma = ca->args[4].as<IntImmNode>();
    const auto *mb = cb->args[4].as<IntImmNode>();
    if (!ma || !mb || ma->dtype != mb->dtype ||
        (ma->value != 2 && ma->value != 3) ||
        (mb->value != 2 && mb->value != 3))
      return branch;
    Call accumulator = GetRef<Call>(ca);
    Array<PrimExpr> pointer_args = accumulator->args;
    pointer_args.Set(4, make_const(ma->dtype, ma->value | mb->value));
    accumulator.CopyOnWrite()->args = std::move(pointer_args);
    // Keep the else operands: simplification may have specialized the then
    // offsets using its guard (for example q instead of q % versions at q=0).
    Call invocation = GetRef<Call>(b);
    Array<PrimExpr> args = invocation->args;
    args.Set(3, accumulator);
    args.Set(4, Select(branch->condition, a->args[4], b->args[4]));
    invocation.CopyOnWrite()->args = std::move(args);
    return Evaluate(invocation, branch->span);
  }

  ConstrSet context_;
};

// Geometry of the actual local helper, independently of any example shape.
// For aligned FP16 tiles, codegen's round16(M/N) and round32B(K) are
// identities.
struct PtoGemmDescriptor {
  int64_t m, n, k, k_l0, chunks, tail, max_chunk;
  std::array<int64_t, 3> rows, columns, bytes;
  int64_t l0a_bytes, l0b_bytes;

  explicit PtoGemmDescriptor(const LegacyCompositeOperation &operation)
      : m(operation.m), n(operation.n), k(operation.k), k_l0(operation.k_l0) {
    const auto arguments = TemplateArguments(operation.call.get());
    ICHECK(arguments[0] == "half" && arguments[1] == "float" &&
           arguments[5] == "false" && arguments[6] == "false")
        << kPtoProfile << ": requires FP16-to-FP32 and no transpose";
    ICHECK(m > 0 && n > 0 && k > 0 && k_l0 > 0 && m % 16 == 0 && n % 16 == 0 &&
           k % 16 == 0 && k_l0 % 16 == 0)
        << kPtoProfile
        << ": requires static 16-aligned physical/valid dimensions";
    chunks = 1 + (k - 1) / k_l0;
    tail = k - (chunks - 1) * k_l0;
    max_chunk = std::min(k, k_l0);
    // MMAD dimensions are 12-bit; the total K belongs to L1 and is not the
    // K dimension of an individual MMAD instruction.
    // k_l0 has its own public API limit, even when the total K is smaller.
    ICHECK_LE(k_l0, 4095) << kPtoProfile
                          << ": kL0Size exceeds the public API limit";
    ICHECK(m <= 4095 && n <= 4095 && max_chunk <= 4095 && tail > 0 &&
           tail % 16 == 0 && k <= 16384)
        << kPtoProfile << ": MMAD/TLOAD dimensions exceed native field limits";
    rows = {m, k, m};
    columns = {k, n, n};
    bytes = {PtoProduct(PtoProduct(m, k), 2), PtoProduct(PtoProduct(k, n), 2),
             PtoProduct(PtoProduct(m, n), 4)};
    // Each runtime K step uses ONE L0A/B slot at address zero. The final
    // step uses tail, not k_l0, and k_l0 may be greater than the entire K.
    l0a_bytes = PtoProduct(PtoProduct(m, max_chunk), 2);
    l0b_bytes = PtoProduct(PtoProduct(max_chunk, n), 2);
    ICHECK(bytes[2] <= 131072 && bytes[0] <= 524032 && bytes[1] <= 524032)
        << kPtoProfile << ": helper footprint exceeds L0C/L1 capacity";
  }
  void ValidatePtoResources() const {
    ICHECK(l0a_bytes <= 65536 && l0b_bytes <= 65536)
        << kPtoProfile << ": helper footprint exceeds L0A/L0B capacity";
  }
  void ValidateAscendCResources() const {
    const int64_t n_max = 32768 / PtoProduct(k_l0, 2);
    ICHECK_GT(n_max, 0) << "AscendC GEMM N tile cannot fit L0B";
    const int64_t n_tile = std::min(n, n_max);
    ICHECK(n_tile % 16 == 0 && n % n_tile == 0)
        << "AscendC GEMM requires aligned complete internal N tiles";
    const int64_t steps = PtoProduct(chunks, n / n_tile);
    const int64_t slot_budget = 65536 / (steps > 1 ? 2 : 1);
    ICHECK(PtoProduct(PtoProduct(m, k_l0), 2) <= slot_budget &&
           PtoProduct(PtoProduct(n_tile, k_l0), 2) <= slot_budget)
        << "AscendC GEMM ping-pong footprint exceeds its L0 slot budget";
  }
};

struct PtoFlatLeaf {
  Call call;
  ConstrSet context;
};

// Preserve lexical proof scopes. In particular an assumption in one sibling
// scope must never justify a different sibling's GM address.
class PtoFlatLeaves {
public:
  std::vector<PtoFlatLeaf> Run(const Stmt &body) {
    Walk(body, {}, -1);
    return std::move(leaves_);
  }

  static bool Assume(const Stmt &statement, ConstrSet *context) {
    const auto *evaluation = statement.as<EvaluateNode>();
    const auto *call = evaluation ? evaluation->value.as<CallNode>() : nullptr;
    const auto *op = call ? call->op.as<OpNode>() : nullptr;
    if (!op || op->name != "tir.assume")
      return false;
    ICHECK_EQ(call->args.size(), 1U) << kPtoProfile;
    PtoPredicate(call->args[0], *context);
    context->AddConstr(call->args[0], true);
    return true;
  }

private:
  void Walk(const Stmt &statement, ConstrSet context, int resource) {
    if (const auto *sequence = statement.as<SeqStmtNode>()) {
      for (const Stmt &child : sequence->seq)
        if (!Assume(child, &context))
          Walk(child, context, resource);
    } else if (const auto *attribute = statement.as<AttrStmtNode>()) {
      PtoPure(attribute->value);
      if (attribute->node.as<PrimExprNode>())
        PtoPure(Downcast<PrimExpr>(attribute->node));
      if (attribute->attr_key == "resource_scope") {
        PtoConstant(attribute->value, 0);
        resource = 0;
      } else if (attribute->attr_key == "thread_extent" ||
                 attribute->attr_key == "virtual_thread") {
        const auto *iteration = attribute->node.as<IterVarNode>();
        ICHECK(iteration && attribute->value.dtype().is_scalar() &&
               attribute->value.dtype().is_int())
            << kPtoProfile << ": malformed thread domain";
        context.AddConstr(
            iteration->var,
            Range::FromMinExtent(make_zero(attribute->value.dtype()),
                                 attribute->value));
        context.AddConstr(attribute->value > 0);
      } else if (attribute->attr_key == "tl.assume" ||
                 attribute->attr_key == "tilelang_assume") {
        ICHECK(attribute->node.as<PrimExprNode>()) << kPtoProfile;
        const PrimExpr predicate = Downcast<PrimExpr>(attribute->node);
        PtoPredicate(predicate, context);
        context.AddConstr(predicate, true);
      }
      Walk(attribute->body, std::move(context), resource);
    } else if (const auto *binding = statement.as<LetStmtNode>()) {
      PtoPure(binding->value);
      if ((binding->value.dtype().is_int() ||
           binding->value.dtype().is_uint()) &&
          !binding->value.dtype().is_bool()) {
        ProofAnalyzer analyzer;
        context.Populate(analyzer);
        PtoInteger(binding->value, &analyzer);
      }
      context.AddConstr(binding->var, binding->value);
      Walk(binding->body, std::move(context), resource);
    } else if (const auto *allocation = statement.as<AllocateNode>()) {
      ICHECK(is_one(allocation->condition))
          << kPtoProfile << ": predicated allocations are unsupported";
      Walk(allocation->body, std::move(context), resource);
    } else if (const auto *declaration = statement.as<DeclBufferNode>()) {
      Walk(declaration->body, std::move(context), resource);
    } else if (const auto *evaluation = statement.as<EvaluateNode>()) {
      if (evaluation->value.as<IntImmNode>() || Assume(statement, &context))
        return;
      const auto *call = evaluation->value.as<CallNode>();
      ICHECK(call && (resource == -1 || resource == 0))
          << kPtoProfile << ": requires Cube call leaves";
      const std::string name = OperationName(call);
      if (name == "set_flag" || name == "wait_flag" ||
          name == "auto_set_flag" || name == "auto_wait_flag")
        return; // Validated independently against the helper reservations.
      ICHECK(name == "copy_gm_to_l1" || name == "gemm_v0" ||
             name == "copy_l0c_to_gm")
          << kPtoProfile
          << ": extra compute or mixed C/V needs a separate protocol";
      leaves_.push_back({GetRef<Call>(call), std::move(context)});
    } else {
      LOG(FATAL) << kPtoProfile
                 << ": For/If/rings require a separate invocation protocol";
    }
  }
  std::vector<PtoFlatLeaf> leaves_;
};

class PtoGemmAudit {
public:
  PtoGemmAudit(const PrimFunc &function,
               const LegacyCompositeOperation &operation, bool pto = true)
      : function_(function), operation_(operation), geometry_(operation),
        is_pto_(pto) {
    shapes_ = function_->GetAttr<Map<Var, Array<PrimExpr>>>(kLogicBufferShapes)
                  .value_or(Map<Var, Array<PrimExpr>>{});
  }
  void AuditPipelineGM(const PtoFlatLeaf &leaf, const PrimExpr &pointer,
                       int64_t rows, int64_t columns) const {
    GM(leaf, pointer, DataType::Float(16), rows, columns, 1, true);
  }
  void AuditPipelineFix(const PtoFlatLeaf &leaf) const { Fix(leaf, true); }

  void Run() {
    PtoConstant(operation_.call->args[4], 1);
    geometry_.ValidatePtoResources();
    const auto shapes =
        function_->GetAttr<Map<Var, Array<PrimExpr>>>(kLogicBufferShapes);
    const auto addresses =
        function_->GetAttr<Map<Var, PrimExpr>>("address_map");
    const auto sizes = function_->GetAttr<Map<Var, PrimExpr>>("size_map");
    ICHECK(shapes.defined() && addresses.defined() && sizes.defined())
        << kPtoProfile
        << ": requires authoritative descriptors and allocation maps";
    shapes_ = shapes.value();
    std::array<int64_t, 3> bases, spans;
    std::array<Var, 3> roots;
    for (size_t index = 0; index < 3; ++index) {
      const Call &pointer = operation_.buffers[index].access_ptr;
      roots[index] = Downcast<Var>(pointer->args[1]);
      const DataType dtype = DataType::Float(index == 2 ? 32 : 16);
      Descriptor(roots[index], geometry_.rows[index], geometry_.columns[index],
                 true);
      // A full root view is required. Subviews need their own fractal offset
      // contract rather than relying on logical access_ptr extents.
      PtoConstant(pointer->args[2], 0);
      PtoConstant(pointer->args[3],
                  PtoProduct(geometry_.rows[index], geometry_.columns[index]));
      size_t allocations = 0;
      int64_t allocation_bytes = 0;
      PostOrderVisit(function_->body, [&](const ObjectRef &node) {
        const auto *allocation = node.as<AllocateNode>();
        if (!allocation || !allocation->buffer_var.same_as(roots[index]))
          return;
        ++allocations;
        ICHECK(allocation->dtype == dtype && is_one(allocation->condition))
            << kPtoProfile
            << ": actual allocation dtype/condition disagrees with descriptor";
        allocation_bytes = dtype.bytes();
        for (const PrimExpr &extent : allocation->extents)
          allocation_bytes = PtoProduct(allocation_bytes, Positive(extent));
      });
      ICHECK_EQ(allocations, 1U)
          << kPtoProfile << ": requires unique allocation roots";
      const auto address = addresses.value().Get(roots[index]);
      const auto size = sizes.value().Get(roots[index]);
      ICHECK(address.defined() && size.defined()) << kPtoProfile;
      bases[index] = Nonnegative(address.value());
      spans[index] = Positive(size.value());
      const int64_t capacity = index == 2 ? 131072 : 524032;
      ICHECK(allocation_bytes >= geometry_.bytes[index] &&
             spans[index] >= allocation_bytes && bases[index] % 32 == 0 &&
             spans[index] <= capacity &&
             bases[index] <= capacity - spans[index])
          << kPtoProfile
          << ": actual allocation footprint exceeds its physical pool";
    }
    ICHECK(!roots[0].same_as(roots[1]) &&
           (bases[0] + spans[0] <= bases[1] || bases[1] + spans[1] <= bases[0]))
        << kPtoProfile << ": A/B L1 allocations overlap";
    const auto leaves = PtoFlatLeaves().Run(function_->body);
    ICHECK(leaves.size() == 4U && leaves[2].call.same_as(operation_.call) &&
           OperationName(leaves[3].call.get()) == "copy_l0c_to_gm")
        << kPtoProfile
        << ": requires two full GM refreshes, one helper and one final FIX";
    for (size_t index = 0; index < 2; ++index) {
      const Call &copy = leaves[index].call;
      ValidateKLoopCopy(copy, operation_.buffers[index].access_ptr, "half",
                        geometry_.rows[index], geometry_.columns[index]);
      GM(leaves[index], copy->args[1], DataType::Float(16),
         geometry_.rows[index], geometry_.columns[index], 1, true);
    }
    Fix(leaves[3]);
  }

private:
  static int64_t Nonnegative(const PrimExpr &value) {
    const auto *literal = value.as<IntImmNode>();
    ICHECK(literal && literal->value >= 0)
        << kPtoProfile << ": requires static nonnegative metadata";
    return literal->value;
  }
  static int64_t Positive(const PrimExpr &value) {
    const int64_t result = Nonnegative(value);
    ICHECK_GT(result, 0) << kPtoProfile;
    return result;
  }
  void Descriptor(const Var &data, int64_t rows, int64_t columns,
                  bool local) const {
    const auto shape = shapes_.Get(data);
    ICHECK(shape.defined() &&
           (local ? shape.value().size() == 4U : shape.value().size() == 2U))
        << kPtoProfile << ": missing physical/valid descriptor metadata";
    for (size_t axis = 0; axis < shape.value().size(); ++axis)
      PtoConstant(shape.value()[axis], axis % 2 == 0 ? rows : columns);
  }
  void GM(const PtoFlatLeaf &leaf, const PrimExpr &argument, DataType dtype,
          int64_t rows, int64_t columns, int mask, bool input) const {
    const auto *pointer = argument.as<CallNode>();
    ICHECK(pointer && pointer->op.same_as(builtin::tvm_access_ptr()) &&
           pointer->args.size() == 5U)
        << kPtoProfile << ": malformed GM access pointer";
    const auto *annotation = pointer->args[0].as<CallNode>();
    ICHECK(annotation &&
           annotation->op.same_as(Op::Get("tir.type_annotation")) &&
           annotation->dtype == dtype)
        << kPtoProfile << ": GM pointer dtype disagrees with copy semantics";
    PtoConstant(pointer->args[3], PtoProduct(rows, columns));
    PtoConstant(pointer->args[4], mask);
    const Var data = Downcast<Var>(pointer->args[1]);
    Optional<Buffer> parameter;
    for (const auto &[key, buffer] : function_->buffer_map) {
      if (!buffer->data.same_as(data))
        continue;
      ICHECK(!parameter.defined())
          << kPtoProfile << ": ambiguous GM parameter root";
      parameter = buffer;
    }
    ICHECK(parameter.defined() && parameter.value()->dtype == dtype &&
           parameter.value()->shape.size() == 2U &&
           parameter.value()->strides.empty() &&
           is_zero(parameter.value()->elem_offset) &&
           (parameter.value().scope().empty() ||
            parameter.value().scope() == "global"))
        << kPtoProfile
        << ": requires dense GM parameters without retyped views";
    const int64_t global_rows = Positive(parameter.value()->shape[0]);
    const int64_t global_columns = Positive(parameter.value()->shape[1]);
    const int64_t capacity = PtoProduct(global_rows, global_columns);
    // PTO ComputeStrides emits int32 template arguments for the full GM
    // tensor. AscendC forwards a uint32 row pitch and has no such product ABI.
    if (is_pto_)
      ICHECK_LE(capacity, std::numeric_limits<int32_t>::max()) << kPtoProfile;
    if (is_pto_)
      Descriptor(data, global_rows, global_columns, false);
    const int64_t stride_limit =
        is_pto_ ? (input ? 65535 : std::numeric_limits<int32_t>::max())
                : std::numeric_limits<uint32_t>::max();
    ICHECK(global_columns >= columns && global_columns <= stride_limit)
        << kPtoProfile
        << ": GM stride exceeds the selected backend's native ABI";
    // Catlass TileCopyTla uses a row-by-row fallback for large AscendC
    // strides. Its relative i*srcDValue is uint32 arithmetic; prove that
    // product cannot wrap, without imposing PTO's 16-bit stride ceiling.
    if (!is_pto_ && input)
      ICHECK_LE(PtoProduct(rows - 1, global_columns),
                std::numeric_limits<uint32_t>::max())
          << "AscendC GM-to-L1 fallback row offset exceeds uint32";
    ProofAnalyzer analyzer;
    leaf.context.Populate(analyzer);
    const PrimExpr offset = PtoInteger(pointer->args[2], &analyzer);
    const PrimExpr stride = PtoInteger(leaf.call->args[3], &analyzer);
    const PrimExpr width = IntImm(DataType::Int(64), global_columns);
    // With srcN equal to the saved final GM dimension, ComputeStrides does
    // not merge dimensions, so stride[3] is exactly this dense row pitch.
    ICHECK(analyzer.CanProve(stride == width))
        << kPtoProfile << ": copy stride disagrees with the codegen descriptor";
    const int64_t span = PtoProduct(rows - 1, global_columns);
    ICHECK_LE(span, std::numeric_limits<int64_t>::max() - columns)
        << kPtoProfile;
    const int64_t actual_span = span + columns;
    ICHECK(actual_span <= capacity &&
           analyzer.CanProve(
               offset >= 0 &&
               offset <= IntImm(DataType::Int(64), capacity - actual_span) &&
               floormod(offset, width) <=
                   width - IntImm(DataType::Int(64), columns)))
        << kPtoProfile
        << ": complete strided GM rectangle exceeds its parameter";
  }
  void Fix(const PtoFlatLeaf &leaf, bool native_output = false) const {
    const Call &call = leaf.call;
    ICHECK(call->op.same_as(builtin::call_extern()) && call->args.size() == 10U)
        << kPtoProfile << ": requires canonical final FIX operands";
    const auto *destination = call->args[2].as<CallNode>();
    ICHECK(destination && destination->op.same_as(builtin::tvm_access_ptr()) &&
           destination->args.size() == 5U)
        << kPtoProfile << ": malformed FIX destination";
    const auto *destination_type = destination->args[0].as<CallNode>();
    ICHECK(destination_type &&
           destination_type->op.same_as(Op::Get("tir.type_annotation")))
        << kPtoProfile << ": untyped FIX destination";
    const DataType output_type = destination_type->dtype;
    ICHECK(output_type == DataType::Float(16) ||
           (native_output && output_type == DataType::Float(32)))
        << kPtoProfile << ": unsupported final FIX output conversion";
    const std::string output_name =
        output_type == DataType::Float(16) ? "half" : "float";
    std::string symbol = Downcast<StringImm>(call->args[0])->value;
    symbol.erase(
        std::remove_if(symbol.begin(), symbol.end(),
                       [](unsigned char c) { return std::isspace(c); }),
        symbol.end());
    ICHECK_EQ(symbol, "tl::ascend::copy_l0c_to_gm<float," + output_name +
                          ",layout::RowMajor," + std::to_string(geometry_.m) +
                          "," + std::to_string(geometry_.n) + ",0>")
        << kPtoProfile
        << ": requires a supported FP32-to-float row-major FIX without extra "
           "modes";
    const auto *source = call->args[1].as<CallNode>();
    ICHECK(source && source->op.same_as(builtin::tvm_access_ptr()) &&
           source->args.size() == 5U &&
           source->args[1].same_as(operation_.buffers[2].access_ptr->args[1]))
        << kPtoProfile << ": final FIX must consume the audited accumulator";
    const auto *annotation = source->args[0].as<CallNode>();
    ICHECK(annotation &&
           annotation->op.same_as(Op::Get("tir.type_annotation")) &&
           annotation->dtype == DataType::Float(32))
        << kPtoProfile;
    PtoConstant(source->args[2], 0);
    PtoConstant(source->args[3], PtoProduct(geometry_.m, geometry_.n));
    PtoConstant(source->args[4], 1);
    PtoConstant(call->args[4], geometry_.m);
    PtoConstant(call->args[5], geometry_.n);
    PtoConstant(call->args[6], geometry_.m);
    PtoConstant(call->args[7], geometry_.n);
    PtoConstant(call->args[8], 0);
    PtoConstant(call->args[9], 0);
    GM(leaf, call->args[2], output_type, geometry_.m, geometry_.n, 2, false);
  }

  PrimFunc function_;
  const LegacyCompositeOperation &operation_;
  PtoGemmDescriptor geometry_;
  Map<Var, Array<PrimExpr>> shapes_;
  bool is_pto_;
};

void ValidatePtoFlat(const PrimFunc &function,
                     const LegacyCompositePlan &plan) {
  ICHECK_EQ(plan.operations.size(), 1U)
      << kPtoProfile
      << ": multiple helper invocations need a separate protocol";
  PtoGemmAudit(function, plan.operations.begin()->second).Run();
}

// A certificate of the materialized schedule, not a second scheduler. Logical
// role coverage and read-before-reuse follow the actual stage/order topology;
// ring periods, lead and the number/positions of barriers are not profiles.
class StaticCubePipelineAudit {
public:
  StaticCubePipelineAudit(const PrimFunc &function, LegacyCompositePlan &plan)
      : function_(function), plan_(plan),
        first_(plan.operations.begin()->second), geometry_(first_),
        gm_audit_(function, first_,
                  plan.helper_profile ==
                      LegacyCompositePlan::HelperProfile::kPtoA3) {}

  void Run() {
    const auto owners =
        function_->GetAttr<PhysicalPipelineOwners>(kPhysicalPipelineOwners);
    const auto rings =
        function_->GetAttr<PhysicalRingProvenance>(kPhysicalRingProvenance);
    ICHECK(owners.defined() && owners.value().size() == 1 && rings.defined())
        << Error() << "requires one static pipeline owner";
    owner_ = (*owners.value().begin()).first;
    const auto fields = (*owners.value().begin()).second;
    ICHECK(fields.size() == 4 && Constant(fields[0]) == 1 &&
           Constant(fields[2]) == 0)
        << Error() << "invalid owner schema or original minimum";
    lead_ = Constant(fields[1]);
    order_count_ = Constant(fields[3]);
    ICHECK(lead_ > 0 && order_count_ > 0)
        << Error() << "empty scheduling domain";
    rings_ = rings.value();
    pto_ = plan_.helper_profile == LegacyCompositePlan::HelperProfile::kPtoA3;
    if (pto_)
      geometry_.ValidatePtoResources();
    else
      geometry_.ValidateAscendCResources();
    const auto shapes =
        function_->GetAttr<Map<Var, Array<PrimExpr>>>(kLogicBufferShapes);
    const auto addresses =
        function_->GetAttr<Map<Var, PrimExpr>>("address_map");
    const auto sizes = function_->GetAttr<Map<Var, PrimExpr>>("size_map");
    ICHECK(shapes.defined() && addresses.defined() && sizes.defined())
        << Error() << "requires final descriptors and allocation maps";
    shapes_ = shapes.value();
    addresses_ = addresses.value();
    sizes_ = sizes.value();
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> used_rings;
    for (size_t index = 0; index < 2; ++index) {
      auto &slot = slots_[index];
      const auto &effect = first_.buffers[index];
      slot.root = Downcast<Var>(effect.access_ptr->args[1]);
      slot.id = effect.logical_ring.value_or(slot.root);
      slot.stride = geometry_.bytes[index];
      if (effect.logical_ring.defined()) {
        const auto record = rings_.Get(slot.id);
        ICHECK(record.defined()) << Error() << "missing input ring provenance";
        ValidatePhysicalRingRecord(record.value());
        const auto &data = record.value();
        slot.versions = Constant(data[kPhysicalRingVersions]);
        ICHECK(slot.versions > 1 &&
               data[kPhysicalRingRoot].same_as(slot.root) &&
               Constant(data[kPhysicalRingOriginalMin]) == 0 &&
               Constant(data[kPhysicalRingElementBits]) == 16)
            << Error()
            << "input ring provenance disagrees with its actual operand";
        const int64_t base_bits = Constant(data[kPhysicalRingBaseBits]);
        const int64_t stride_bits = Constant(data[kPhysicalRingStrideBits]);
        ICHECK(base_bits % 256 == 0 && stride_bits > 0 &&
               stride_bits % 256 == 0)
            << Error() << "ring slots require complete aligned physical spans";
        slot.base = base_bits / 8;
        slot.stride = stride_bits / 8;
        used_rings.insert(slot.id);
      }
      ICHECK(slot.stride >= geometry_.bytes[index])
          << Error() << "GEMM footprint exceeds an input slot";
      const int64_t storage_bytes = PtoProduct(slot.versions, slot.stride);
      ICHECK(slot.base <= std::numeric_limits<int64_t>::max() - storage_bytes)
          << Error() << "ring span overflow";
      AuditAllocation(slot.root, DataType::Float(16), slot.base + storage_bytes,
                      524032, geometry_.columns[index], true);
    }
    ICHECK_EQ(used_rings.size(), rings_.size())
        << Error() << "unaccounted physical ring lineage";
    ICHECK(!slots_[0].root.same_as(slots_[1].root))
        << Error()
        << "independent input lineages require distinct allocation roots";
    const int64_t a_begin = Constant(addresses_.at(slots_[0].root));
    const int64_t b_begin = Constant(addresses_.at(slots_[1].root));
    const int64_t a_span = Constant(sizes_.at(slots_[0].root));
    const int64_t b_span = Constant(sizes_.at(slots_[1].root));
    ICHECK(a_begin + a_span <= b_begin || b_begin + b_span <= a_begin)
        << Error() << "A/B physical L1 allocation spans overlap";
    c_ = Downcast<Var>(first_.buffers[2].access_ptr->args[1]);
    AuditAllocation(c_, DataType::Float(32), geometry_.bytes[2], 131072,
                    geometry_.n, false);
    WalkOuter(function_->body, {});
    ICHECK(saw_owner_ && final_stores_ == 1 && !in_loop_)
        << Error() << "requires one complete owner and final FIX";
    ICHECK_EQ(seen_gemms_.size(), plan_.operations.size())
        << Error() << "unaccounted GEMM invocation";
    std::array<int, 3> data_orders{-1, -1, -1};
    for (auto &[order, record] : orders_) {
      if (record.role != Role::kBarrier) {
        const int role = static_cast<int>(record.role);
        ICHECK_EQ(data_orders[role], -1)
            << Error() << "multiple producer/consumer orders";
        data_orders[role] = order;
      }
      auto &coverage = record.coverage;
      std::sort(coverage.begin(), coverage.end());
      int64_t next = 0;
      for (const auto &[begin, end] : coverage) {
        ICHECK(begin == next && end > begin && end <= trip_count_)
            << Error() << "logical iteration coverage is missing or duplicated";
        next = end;
      }
      ICHECK_EQ(next, trip_count_)
          << Error() << "incomplete logical iteration coverage";
    }
    for (int order : data_orders)
      ICHECK_GE(order, 0) << Error()
                          << "missing A/B refresh or GEMM dataflow role";
    const auto &consumer = orders_.at(data_orders[2]);
    for (size_t index = 0; index < 2; ++index) {
      const int producer_order = data_orders[index];
      const auto &producer = orders_.at(producer_order);
      // Derive lifetime from actual issue order. Equal physical epochs need
      // strict order indices, including when producer and consumer share a
      // stage.
      ICHECK(
          producer.stage < consumer.stage ||
          (producer.stage == consumer.stage && producer_order < data_orders[2]))
          << Error() << "GEMM executes before its matching input refresh";
      const int64_t reuse_stage = slots_[index].versions + producer.stage;
      ICHECK(reuse_stage > consumer.stage ||
             (reuse_stage == consumer.stage && data_orders[2] < producer_order))
          << Error()
          << "physical slot is overwritten before its previous GEMM read";
    }
    // This private marker certifies the real helper's closed L1 release
    // protocol for the upstream engine's proved carried-edge optimization.
    plan_.cube_pipeline_loop = steady_;
  }

private:
  enum class Role : int { kCopyA = 0, kCopyB = 1, kGemm = 2, kBarrier = 3 };
  struct Slot {
    Var id, root;
    int64_t versions{1}, base{0}, stride{0};
  };
  struct Order {
    Role role;
    int64_t stage;
    std::vector<std::pair<int64_t, int64_t>> coverage;
  };
  static const char *Error() { return "Static Cube pipeline contract: "; }
  static int64_t Constant(const PrimExpr &value) {
    const auto *number = value.as<IntImmNode>();
    ICHECK(number && number->value >= 0 &&
           number->value <= std::numeric_limits<int32_t>::max())
        << Error() << "requires bounded static scheduling/allocation metadata";
    return number->value;
  }
  void Identity(const ObjectRef &node) const {
    const auto ids = Downcast<Array<Var>>(node);
    ICHECK(ids.size() == 1 && ids[0].same_as(owner_))
        << Error() << "owner identity mismatch";
  }
  void AuditAllocation(const Var &root, DataType dtype, int64_t required,
                       int64_t capacity, int64_t columns,
                       bool aggregate) const {
    ICHECK(addresses_.count(root) && sizes_.count(root) && shapes_.count(root))
        << Error() << "missing final physical allocation facts";
    const int64_t address = Constant(addresses_.at(root));
    const int64_t span = Constant(sizes_.at(root));
    ICHECK(address % 32 == 0 && required <= span && span <= capacity &&
           address <= capacity - span)
        << Error() << "allocation exceeds physical pool capacity";
    const auto shape = shapes_.at(root);
    ICHECK(shape.size() == 4 && Constant(shape[0]) > 0 &&
           Constant(shape[0]) == Constant(shape[2]) &&
           Constant(shape[1]) == columns && Constant(shape[3]) == columns)
        << Error() << "physical/valid descriptor shape disagreement";
    const int64_t descriptor_bytes =
        PtoProduct(PtoProduct(Constant(shape[0]), columns), dtype.bytes());
    ICHECK(descriptor_bytes >= required && descriptor_bytes <= span &&
           (aggregate || descriptor_bytes == required))
        << Error() << "descriptor does not cover complete physical slots";
    size_t allocations = 0;
    PostOrderVisit(function_->body, [&](const ObjectRef &node) {
      const auto *allocation = node.as<AllocateNode>();
      if (!allocation || !allocation->buffer_var.same_as(root))
        return;
      ++allocations;
      ICHECK(allocation->dtype == dtype && is_one(allocation->condition))
          << Error()
          << "requires unpredicated allocation roots without retyped views";
      int64_t bytes = dtype.bytes();
      for (const PrimExpr &extent : allocation->extents)
        bytes = PtoProduct(bytes, Constant(extent));
      ICHECK(bytes >= descriptor_bytes && bytes <= span)
          << Error() << "actual allocation disagrees with physical descriptor";
    });
    ICHECK_EQ(allocations, 1U)
        << Error() << "requires unique actual allocation roots";
  }
  void Attribute(const AttrStmtNode *op, ConstrSet *context) const {
    PtoPure(op->value);
    if (op->node.as<PrimExprNode>())
      PtoPure(Downcast<PrimExpr>(op->node));
    ICHECK(std::string(op->attr_key).rfind("async_", 0) != 0)
        << Error() << "async protocols require a separate contract";
    if (op->attr_key == "resource_scope")
      PtoConstant(op->value, 0);
    if (op->attr_key == "thread_extent" || op->attr_key == "virtual_thread") {
      const auto *iteration = op->node.as<IterVarNode>();
      ICHECK(iteration) << Error() << "malformed thread domain";
      context->AddConstr(
          iteration->var,
          Range::FromMinExtent(make_zero(op->value.dtype()), op->value));
      context->AddConstr(op->value > 0);
    } else if ((op->attr_key == "tl.assume" ||
                op->attr_key == "tilelang_assume") &&
               op->node.as<PrimExprNode>()) {
      const PrimExpr predicate = Downcast<PrimExpr>(op->node);
      PtoPredicate(predicate, *context);
      context->AddConstr(predicate, true);
    }
  }
  void Bind(const LetStmtNode *binding, ConstrSet *context) const {
    PtoPure(binding->value);
    if ((binding->value.dtype().is_int() || binding->value.dtype().is_uint()) &&
        !binding->value.dtype().is_bool()) {
      ProofAnalyzer analyzer;
      context->Populate(analyzer);
      PtoInteger(binding->value, &analyzer);
    }
    context->AddConstr(binding->var, binding->value);
  }
  void WalkOuter(const Stmt &statement, ConstrSet context) {
    if (const auto *sequence = statement.as<SeqStmtNode>()) {
      for (const Stmt &child : sequence->seq)
        if (!PtoFlatLeaves::Assume(child, &context))
          WalkOuter(child, context);
    } else if (const auto *attribute = statement.as<AttrStmtNode>()) {
      Attribute(attribute, &context);
      if (attribute->attr_key == kPhysicalPipelineExtent) {
        Identity(attribute->node);
        ICHECK(!saw_owner_) << Error() << "duplicate owner";
        saw_owner_ = true;
        trip_count_ = Constant(attribute->value);
        ICHECK(trip_count_ > 0 &&
               trip_count_ <= std::numeric_limits<int32_t>::max() - lead_)
            << Error() << "requires positive nonwrapping static trip count";
        WalkOwner(attribute->body, context, -1);
        owner_finished_ = true;
      } else {
        ICHECK(!IsPhysicalPipelineMetadata(attribute->attr_key))
            << Error() << "orphan pipeline metadata";
        WalkOuter(attribute->body, context);
      }
    } else if (const auto *allocation = statement.as<AllocateNode>()) {
      ICHECK(is_one(allocation->condition))
          << Error() << "predicated allocation";
      WalkOuter(allocation->body, context);
    } else if (const auto *declaration = statement.as<DeclBufferNode>()) {
      WalkOuter(declaration->body, context);
    } else if (const auto *binding = statement.as<LetStmtNode>()) {
      Bind(binding, &context);
      WalkOuter(binding->body, context);
    } else if (const auto *evaluation = statement.as<EvaluateNode>()) {
      if (is_zero(evaluation->value) ||
          PtoFlatLeaves::Assume(statement, &context))
        return;
      const auto *call = evaluation->value.as<CallNode>();
      ICHECK(owner_finished_ && call && OperationName(call) == "copy_l0c_to_gm")
          << Error() << "extra work outside the owner or FIX before completion";
      gm_audit_.AuditPipelineFix({GetRef<Call>(call), context});
      ++final_stores_;
    } else {
      LOG(FATAL) << Error()
                 << "outer semantic control requires a separate protocol";
    }
  }
  void AppendEvent(int64_t begin, int64_t end, int order) {
    if (in_loop_) {
      ICHECK(loop_last_order_ < order)
          << Error() << "steady frame order is not strictly increasing";
      if (loop_first_order_ < 0)
        loop_first_order_ = order;
      loop_last_order_ = order;
      return;
    }
    AppendGroup(begin, end, order, order);
  }
  void AppendGroup(int64_t begin, int64_t end, int first_order,
                   int last_order) {
    ICHECK(last_epoch_ < begin ||
           (last_epoch_ == begin && last_order_ < first_order))
        << Error() << "materialized physical epochs/orders are reordered";
    last_epoch_ = end;
    last_order_ = last_order;
  }
  void WalkOwner(const Stmt &statement, ConstrSet context, int segment) {
    if (const auto *sequence = statement.as<SeqStmtNode>()) {
      for (const Stmt &child : sequence->seq)
        if (!PtoFlatLeaves::Assume(child, &context))
          WalkOwner(child, context, segment);
    } else if (const auto *attribute = statement.as<AttrStmtNode>()) {
      Attribute(attribute, &context);
      if (attribute->attr_key == kPhysicalPipelineSegment) {
        Identity(attribute->node);
        const int role = Constant(attribute->value);
        ICHECK(segment == -1 && role >= 0 && role <= 2 && !segments_[role])
            << Error() << "duplicate or nested segment";
        segments_[role] = true;
        WalkOwner(attribute->body, context, role);
      } else if (attribute->attr_key == kPhysicalPipelineStage) {
        const auto fields = Downcast<Array<PrimExpr>>(attribute->node);
        ICHECK(fields.size() == 5 && fields[0].same_as(owner_) &&
               segment >= 0 && Constant(fields[1]) == segment)
            << Error() << "stage/segment owner mismatch";
        const int64_t stage = Constant(fields[2]);
        const int order = Constant(fields[3]);
        ICHECK(stage <= lead_ && order < order_count_ &&
               Constant(fields[4]) <= 1)
            << Error() << "stage/order/check field exceeds its domain";
        stage_work_ = 0;
        WalkStage(attribute->body, context, {}, attribute->value, stage, order,
                  segment);
        ICHECK_LE(stage_work_, 1)
            << Error() << "a stage contains multiple executable effects";
      } else {
        ICHECK(!IsPhysicalPipelineMetadata(attribute->attr_key))
            << Error() << "unexpected nested owner metadata";
        WalkOwner(attribute->body, context, segment);
      }
    } else if (const auto *loop = statement.as<ForNode>()) {
      ICHECK(segment == 1 && !in_loop_ && !steady_.defined() &&
             loop->kind == ForKind::kSerial && is_zero(loop->min) &&
             trip_count_ > lead_ &&
             Constant(loop->extent) == trip_count_ - lead_)
          << Error() << "requires one original serial steady domain";
      for (const auto &[key, value] : loop->annotations)
        ICHECK(key == "tl_cross_interval" && is_one(Downcast<PrimExpr>(value)))
            << Error() << "unsupported steady annotation";
      steady_ = loop->loop_var;
      context.AddConstr(loop->loop_var,
                        Range::FromMinExtent(loop->min, loop->extent));
      in_loop_ = true;
      loop_first_order_ = loop_last_order_ = -1;
      WalkOwner(loop->body, context, segment);
      in_loop_ = false;
      if (loop_first_order_ >= 0)
        AppendGroup(lead_, trip_count_ - 1, loop_first_order_,
                    loop_last_order_);
    } else if (const auto *binding = statement.as<LetStmtNode>()) {
      Bind(binding, &context);
      WalkOwner(binding->body, context, segment);
    } else {
      const auto *evaluation = statement.as<EvaluateNode>();
      ICHECK(evaluation && (is_zero(evaluation->value) ||
                            PtoFlatLeaves::Assume(statement, &context)))
          << Error() << "unmarked effect/control inside owner";
    }
  }
  void ValidateSlot(const Call &pointer, size_t index, const PrimExpr &clock,
                    const Map<Var, PrimExpr> &clocks,
                    const ConstrSet &context) const {
    const Slot &slot = slots_[index];
    ICHECK(pointer->op.same_as(builtin::tvm_access_ptr()) &&
           pointer->args.size() == 5 && pointer->args[1].same_as(slot.root) &&
           pointer->args[0].dtype() == DataType::Float(16))
        << Error() << "input slot lineage/type changed";
    PtoConstant(pointer->args[3],
                PtoProduct(geometry_.rows[index], geometry_.columns[index]));
    if (slot.versions > 1)
      ICHECK(clocks.count(slot.id) &&
             context.CanProve(clocks.at(slot.id) == clock))
          << Error()
          << "physical ring clock is missing or disagrees with its frame";
    ProofAnalyzer analyzer;
    context.Populate(analyzer);
    const PrimExpr actual = PtoInteger(pointer->args[2], &analyzer);
    const PrimExpr q = PtoInteger(clock, &analyzer);
    const PrimExpr expected =
        IntImm(DataType::Int(64), slot.base / 2) +
        floormod(q, IntImm(DataType::Int(64), slot.versions)) *
            IntImm(DataType::Int(64), slot.stride / 2);
    ICHECK(analyzer.CanProve(actual == expected))
        << Error()
        << "input pointer does not select its logical epoch's physical slot";
  }
  void WalkStage(const Stmt &statement, ConstrSet context,
                 Map<Var, PrimExpr> clocks, const PrimExpr &clock,
                 int64_t stage, int order, int segment) {
    if (const auto *attribute = statement.as<AttrStmtNode>()) {
      Attribute(attribute, &context);
      if (attribute->attr_key == kPhysicalRingClock) {
        const Var id = PhysicalRingClockId(attribute->node);
        ICHECK(rings_.count(id) && context.CanProve(attribute->value == clock))
            << Error() << "ring clock disagrees with its stage";
        clocks.Set(id, attribute->value);
      } else {
        ICHECK(!IsPhysicalPipelineMetadata(attribute->attr_key))
            << Error() << "nested stage metadata";
      }
      WalkStage(attribute->body, context, clocks, clock, stage, order, segment);
      return;
    }
    if (const auto *binding = statement.as<LetStmtNode>()) {
      Bind(binding, &context);
      WalkStage(binding->body, context, clocks, clock, stage, order, segment);
      return;
    }
    if (const auto *sequence = statement.as<SeqStmtNode>()) {
      for (const Stmt &child : sequence->seq)
        if (!PtoFlatLeaves::Assume(child, &context))
          WalkStage(child, context, clocks, clock, stage, order, segment);
      return;
    }
    const auto *evaluation = statement.as<EvaluateNode>();
    ICHECK(evaluation) << Error() << "conditional or unsupported stage work";
    if (is_zero(evaluation->value) ||
        PtoFlatLeaves::Assume(statement, &context))
      return;
    const auto *call = evaluation->value.as<CallNode>();
    ICHECK(call) << Error() << "stage is not a primitive invocation";
    ++stage_work_;
    ProofAnalyzer analyzer;
    context.Populate(analyzer);
    const PrimExpr q = PtoInteger(clock, &analyzer);
    const PrimExpr epoch = q + IntImm(DataType::Int(64), stage);
    int64_t begin, end, physical_begin, physical_end;
    if (in_loop_) {
      ICHECK(
          segment == 1 &&
          analyzer.CanProve(epoch == cast(DataType::Int(64), steady_.value()) +
                                         IntImm(DataType::Int(64), lead_)))
          << Error() << "steady clock disagrees with its real execution domain";
      begin = lead_ - stage;
      end = trip_count_ - stage;
      physical_begin = lead_;
      physical_end = trip_count_ - 1;
    } else {
      const auto physical = analyzer.Simplify(epoch);
      const auto *constant = physical.as<IntImmNode>();
      ICHECK(constant) << Error()
                       << "unrolled frame requires a static physical epoch";
      physical_begin = physical_end = constant->value;
      begin = physical_begin - stage;
      end = begin + 1;
    }
    ICHECK(begin >= 0 && end <= trip_count_ && end > begin &&
           (segment == 0 ? physical_begin >= 0 && physical_end < lead_
            : segment == 1
                ? physical_begin >= lead_ && physical_end < trip_count_
                : physical_begin >= trip_count_ &&
                      physical_end < trip_count_ + lead_))
        << Error() << "active frame lies outside its logical/physical segment";
    Role role;
    const std::string name = OperationName(call);
    if (name == "copy_gm_to_l1") {
      ICHECK(call->args.size() == 8) << Error() << "malformed input refresh";
      const auto *destination = call->args[2].as<CallNode>();
      ICHECK(destination && destination->args.size() == 5)
          << Error() << "malformed input pointer";
      const size_t index = destination->args[1].same_as(slots_[0].root) ? 0 : 1;
      role = index == 0 ? Role::kCopyA : Role::kCopyB;
      ValidateSlot(GetRef<Call>(destination), index, clock, clocks, context);
      ValidateKLoopCopy(GetRef<Call>(call), first_.buffers[index].access_ptr,
                        "half", geometry_.rows[index], geometry_.columns[index],
                        true);
      gm_audit_.AuditPipelineGM({GetRef<Call>(call), context}, call->args[1],
                                geometry_.rows[index],
                                geometry_.columns[index]);
      plan_.cube_ring_copies.emplace(
          call, std::make_pair(slots_[index].id, geometry_.bytes[index]));
    } else if (name == "gemm_v0") {
      role = Role::kGemm;
      const auto *operation = plan_.Find(call);
      ICHECK(operation &&
             TemplateArguments(call) == TemplateArguments(first_.call.get()))
          << Error() << "GEMM helper geometry or protocol changed";
      for (size_t index = 0; index < 2; ++index)
        ValidateSlot(operation->buffers[index].access_ptr, index, clock, clocks,
                     context);
      ICHECK(operation->buffers[2].access_ptr->args[1].same_as(c_) &&
             is_zero(operation->buffers[2].access_ptr->args[2]))
          << Error() << "GEMM accumulator lineage changed";
      ICHECK(analyzer.CanProveEqual(call->args[4], clock == 0))
          << Error() << "GEMM clear must equal its logical epoch zero";
      plan_.operations.at(call).pipeline_clock = clock;
      seen_gemms_.insert(call);
    } else {
      ICHECK(name == "pipe_barrier" || name == "auto_barrier")
          << Error() << "unmodelled stage effect";
      for (const PrimExpr &argument : call->args)
        PtoPure(argument);
      role = Role::kBarrier;
    }
    auto found = orders_.find(order);
    if (found == orders_.end())
      found = orders_.emplace(order, Order{role, stage, {}}).first;
    ICHECK(found->second.role == role && found->second.stage == stage)
        << Error() << "an order changes semantic role or stage";
    found->second.coverage.emplace_back(begin, end);
    AppendEvent(physical_begin, physical_end, order);
  }

  PrimFunc function_;
  LegacyCompositePlan &plan_;
  const LegacyCompositeOperation &first_;
  PtoGemmDescriptor geometry_;
  PtoGemmAudit gm_audit_;
  PhysicalRingProvenance rings_;
  Map<Var, Array<PrimExpr>> shapes_;
  Map<Var, PrimExpr> addresses_, sizes_;
  std::array<Slot, 2> slots_;
  std::map<int, Order> orders_;
  std::set<const CallNode *> seen_gemms_;
  Var owner_, c_;
  Optional<Var> steady_;
  int64_t trip_count_{0}, lead_{0}, order_count_{0}, last_epoch_{-1};
  int last_order_{-1}, loop_first_order_{-1}, loop_last_order_{-1},
      stage_work_{0};
  bool pto_{false}, saw_owner_{false}, owner_finished_{false}, in_loop_{false};
  bool segments_[3]{false, false, false};
  int final_stores_{0};
};

class CompositeCollector : public StmtExprVisitor {
public:
  LegacyCompositePlan Run(const PrimFunc &function, const Target &target,
                          const String &platform) {
    for (const auto &[parameter, buffer] : function->buffer_map)
      user_l0ab_ |= IsUserL0AB(buffer->data);
    for (const Var &parameter : function->params)
      user_l0ab_ |= IsUserL0AB(parameter);
    operator()(function->body);
    if (calls_.empty())
      return {};
    // The model selects device codegen. The target kind describes host-side
    // lowering and is llvm in the actual compilation path (or c in IR tests).
    const String model =
        target.defined() ? target->GetAttr<String>("model").value_or(String())
                         : String();
    const bool pto = model == "pto";
    ICHECK(platform == "A3" && (model == "ascendc" || model == "auto" || pto))
        << "Composite gemm_v0 currently requires AscendC on A3";
    const auto rings =
        function->GetAttr<PhysicalRingProvenance>(kPhysicalRingProvenance)
            .value_or(PhysicalRingProvenance{});
    const bool cube_pipeline = !rings.empty();
    if (pto && !cube_pipeline)
      ICHECK(!has_pipeline_ && loops_.empty())
          << kPtoProfile << " requires B1 flat execution without For/rings";
    ICHECK(cube_pipeline || !has_pipeline_)
        << "Composite gemm_v0 software pipelines require certified Cube rings";
    ICHECK(!has_conditional_)
        << "Composite gemm_v0 external If/While control requires separate "
           "hardware validation; the first contract is flat";
    ICHECK(!user_l0ab_ && !has_mma_)
        << "Composite gemm_v0 owns hidden L0A/L0B pools; user L0A/B and bare "
           "mma mixing is unsupported";
    for (const char *name : {"buffer_versions", "tl.buffer_versions_map"}) {
      if (auto versions = function->GetAttr<Map<Var, PrimExpr>>(name)) {
        for (const auto &[storage, version] : versions.value()) {
          const auto *constant = version.as<IntImmNode>();
          const auto record = rings.Get(storage);
          ICHECK(constant != nullptr &&
                 (constant->value == 1 ||
                  (cube_pipeline && record.defined() &&
                   ConstantRingVersion(record.value()) == constant->value)))
              << "Composite gemm_v0 physical B>1/version inputs require a "
                 "later Cube-ring contract";
        }
      }
    }

    LegacyCompositePlan plan;
    // common.h:1252,1375-1439. The helper primes/drains both slots even
    // when its internal K/N loop has only one iteration.
    if (pto) {
      plan.helper_profile = LegacyCompositePlan::HelperProfile::kPtoA3;
      // pto/common.h:145-232: EVENT_ID0==0, hence the literal war_event_id
      // is 1. Every K step reuses ONE L0A/B slot at address zero. Each pair
      // is balanced; there are no AscendC ping-pong prime/drain credits.
      plan.reserved_intra_ids = {
          {"MTE2_MTE1", {1}}, {"FIX_M", {1}},     {"M_MTE1", {1}},
          {"MTE1_M", {1}},    {"MTE1_MTE2", {1}}, {"M_FIX", {1}},
      };
    } else {
      plan.reserved_intra_ids = {
          {"MTE2_MTE1", {0}}, {"FIX_M", {0}},     {"M_MTE1", {0, 1}},
          {"MTE1_M", {0, 1}}, {"MTE1_MTE2", {0}}, {"M_FIX", {0}},
      };
    }
    for (const Call &call : manual_flags_)
      CheckManualFlag(call.get(), plan.reserved_intra_ids);
    for (const Call &call : calls_) {
      LegacyCompositeOperation operation = AnalyzeOperation(call);
      ValidateAllocationFootprints(operation, function, rings);
      if (pto) {
        operation.acquire = ResourcePipe::kMTE2;
        operation.completion = ResourcePipe::kFixpipe;
      }
      plan.operations.emplace(call.get(), std::move(operation));
    }
    if (cube_pipeline) {
      StaticCubePipelineAudit(function, plan).Run();
      plan.static_cube_pipeline = true;
    } else if (pto) {
      ValidatePtoFlat(function, plan);
    } else {
      ValidateKLoop(plan);
    }
    return plan;
  }

private:
  static int64_t ConstantRingVersion(const Array<PrimExpr> &record) {
    ValidatePhysicalRingRecord(record);
    return Downcast<IntImm>(record[kPhysicalRingVersions])->value;
  }
  void ValidateKLoop(const LegacyCompositePlan &plan) const {
    if (loops_.empty())
      return;
    const char *diagnostic =
        "Composite gemm_v0 external For contract requires one static positive "
        "serial K loop with min=0, one gemm and clear=(k == 0)";
    ICHECK(loops_.size() == 1U && calls_.size() == 1U) << diagnostic;
    const auto &[loop, resource] = loops_.front();
    const auto *minimum = loop->min.as<IntImmNode>();
    const auto *extent = loop->extent.as<IntImmNode>();
    ICHECK(loop->kind == ForKind::kSerial && minimum != nullptr &&
           minimum->value == 0 && extent != nullptr && extent->value > 0 &&
           loop->annotations.empty())
        << diagnostic;
    std::vector<Call> leaves;
    CollectKLoopLeaves(loop->body, resource, &leaves);
    ICHECK(leaves.size() == 3U && leaves.back().same_as(calls_.front()))
        << "Composite gemm_v0 external For contract requires two A/B "
           "GM-to-L1 refreshes followed by one gemm";
    // Legacy Analyzer::CanProveEqual lowers equality to lhs-rhs==0 and
    // cannot prove equality between boolean predicates, even k==0 and k==0.
    // Use the existing proof-only adapter and the actual iteration domain;
    // neither the clear expression nor emitted control is rewritten.
    ProofAnalyzer analyzer;
    analyzer.Bind(loop->loop_var,
                  Range::FromMinExtent(loop->min, loop->extent));
    ICHECK(analyzer.CanProveEqual(calls_.front()->args[4], loop->loop_var == 0))
        << diagnostic;
    const LegacyCompositeOperation &operation =
        *plan.Find(calls_.front().get());
    const auto arguments = TemplateArguments(calls_.front().get());
    const bool transpose_a = arguments[5] == "true";
    const bool transpose_b = arguments[6] == "true";
    ValidateKLoopCopy(leaves[0], operation.buffers[0].access_ptr, arguments[0],
                      transpose_a ? operation.k : operation.m,
                      transpose_a ? operation.m : operation.k);
    ValidateKLoopCopy(leaves[1], operation.buffers[1].access_ptr, arguments[0],
                      transpose_b ? operation.n : operation.k,
                      transpose_b ? operation.k : operation.n);
  }

  static void
  CheckManualFlag(const CallNode *call,
                  const std::map<std::string, std::set<int>> &reservations) {
    const std::string name = OperationName(call);
    const size_t offset = call->op.same_as(builtin::call_extern()) ||
                                  call->op.same_as(builtin::call_pure_extern())
                              ? 1
                              : 0;
    const bool automatic = name == "auto_set_flag" || name == "auto_wait_flag";
    const size_t count = automatic ? 2 : 3;
    ICHECK_EQ(call->args.size(), offset + count)
        << "Malformed manual event near composite gemm_v0";
    auto pipe = [&](size_t index) {
      const auto *literal = call->args[offset + index].as<StringImmNode>();
      ICHECK(literal != nullptr)
          << "Composite gemm_v0 requires literal manual event directions";
      std::string value = literal->value;
      if (value.rfind("PIPE_", 0) == 0)
        value.erase(0, 5);
      return value;
    };
    const std::string event = automatic ? pipe(0) : pipe(0) + "_" + pipe(1);
    const auto *id = call->args[offset + count - 1].as<IntImmNode>();
    ICHECK(id != nullptr && id->value >= 0 && id->value < 8)
        << "Composite gemm_v0 requires bounded literal manual event IDs";
    if (auto it = reservations.find(event); it != reservations.end())
      ICHECK(!it->second.count(static_cast<int>(id->value)))
          << "Manual event conflicts with composite gemm_v0 hidden event pool: "
          << event << " ID " << id->value;
  }

  void VisitStmt_(const ForNode *op) final {
    loops_.emplace_back(op, resource_);
    for (const auto &[key, value] : op->annotations)
      has_pipeline_ |= IsPipelineKey(key);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocateNode *op) final {
    user_l0ab_ |= IsUserL0AB(op->buffer_var);
    for (const auto &[key, value] : op->annotations)
      has_pipeline_ |= IsPipelineKey(key);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const DeclBufferNode *op) final {
    user_l0ab_ |= IsUserL0AB(op->buffer->data);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    has_conditional_ = true;
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileNode *op) final {
    has_conditional_ = true;
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers)
      user_l0ab_ |= IsUserL0AB(buffer->data);
    for (const auto &[key, value] : op->annotations)
      has_pipeline_ |= IsPipelineKey(key);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    user_l0ab_ |= IsUserL0AB(op->buffer->data);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    user_l0ab_ |= IsUserL0AB(op->buffer->data);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    has_pipeline_ |= IsPipelineKey(op->attr_key);
    const int previous = resource_;
    if (op->attr_key == "resource_scope") {
      const auto *resource = op->value.as<IntImmNode>();
      resource_ = resource == nullptr ? -2 : resource->value;
    }
    StmtExprVisitor::VisitStmt_(op);
    resource_ = previous;
  }

  void VisitExpr_(const CallNode *op) final {
    const std::string name = OperationName(op);
    if (name == "gemm_v1")
      LOG(FATAL) << "Composite gemm_v1 has no audited helper contract";
    if (name == "gemm_v0") {
      ICHECK(op->op.same_as(Op::Get("tl.ascend_gemm_v0")))
          << "Composite gemm_v0 only supports its canonical typed operation";
      ICHECK(resource_ == -1 || resource_ == 0)
          << "Composite gemm_v0 execution owner must be Cube";
      calls_.push_back(GetRef<Call>(op));
    }
    has_mma_ |= name == "mma";
    if (name == "set_flag" || name == "wait_flag" || name == "auto_set_flag" ||
        name == "auto_wait_flag")
      manual_flags_.push_back(GetRef<Call>(op));
    if (op->op.same_as(builtin::tvm_access_ptr()) && op->args.size() == 5U) {
      if (const auto *data = op->args[1].as<VarNode>())
        user_l0ab_ |= IsUserL0AB(GetRef<Var>(data));
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  std::vector<Call> calls_;
  std::vector<Call> manual_flags_;
  std::vector<std::pair<const ForNode *, int>> loops_;
  bool has_pipeline_{false};
  bool has_conditional_{false};
  bool user_l0ab_{false};
  bool has_mma_{false};
  int resource_{-1};
};

} // namespace

const LegacyCompositeOperation *
LegacyCompositePlan::Find(const CallNode *call) const {
  const auto it = operations.find(call);
  return it == operations.end() ? nullptr : &it->second;
}

PrimFunc NormalizeLegacyCompositeBranches(const PrimFunc &function) {
  bool has_gemm = false;
  PostOrderVisit(function->body, [&](const ObjectRef &node) {
    const auto *call = node.as<CallNode>();
    has_gemm |= call && call->op.same_as(Op::Get("tl.ascend_gemm_v0"));
  });
  if (!has_gemm)
    return function;
  const Stmt body = CompositeBranchNormalizer()(function->body);
  if (body.same_as(function->body))
    return function;
  PrimFunc result = function;
  result.CopyOnWrite()->body = body;
  return result;
}

LegacyCompositePlan AnalyzeLegacyComposites(const PrimFunc &function,
                                            const Target &target,
                                            const String &platform) {
  return CompositeCollector().Run(function, target, platform);
}

} // namespace tvm::tl::ascend_sync
