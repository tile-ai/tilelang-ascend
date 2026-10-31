// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

// Original legacy-ABI adapter, not an upstream dependency-analysis algorithm.
// Audited sources: src/op/ascend.cc (AscendCopy::Lower),
// src/target/codegen_ascend.cc (CopyCodegen), and
// src/tl_templates/ascend/common.h (copy_gm_to_ub).

#include "legacy_tail_copy.h"

#include <tvm/ir/type.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>

#include <charconv>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../common/attr.h"
#include "../common/pipeline_ring_provenance.h"
#include "legacy_helper_type.h"

namespace tvm::tl::ascend_sync {

using namespace tir;

namespace {

constexpr const char *kError = "GM-to-UB tail-copy contract: ";

PrimExpr I64(int64_t value) { return IntImm(DataType::Int(64), value); }

PrimExpr Int64(const PrimExpr &value, const char *field) {
  const DataType dtype = value.dtype();
  ICHECK(dtype.is_scalar() && ((dtype.is_int() && dtype.bits() <= 64) ||
                               (dtype.is_uint() && dtype.bits() <= 32)))
      << kError << field << " requires a scalar value-preserving integer";
  ICHECK(SideEffect(value) <= CallEffectKind::kPure)
      << kError << field << " must use its captured scalar value";
  return cast(DataType::Int(64), value);
}

std::string Trim(const std::string &text) {
  const size_t first = text.find_first_not_of(" \t\r\n");
  ICHECK(first != std::string::npos) << kError << "empty template argument";
  const size_t last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

struct Shape {
  std::string element_type;
  int64_t columns;
  int64_t rows;
};

Shape ParseShape(const CallNode *call, size_t offset) {
  ICHECK(offset == 1 && call->op.same_as(builtin::call_extern()))
      << kError << "requires the typed AscendCopy call_extern ABI";
  const auto *callee = call->args[0].as<StringImmNode>();
  ICHECK(callee != nullptr) << kError << "missing literal helper name";
  std::string name = callee->value;
  const std::string prefix = "tl::ascend::";
  if (name.rfind(prefix, 0) == 0)
    name.erase(0, prefix.size());
  const size_t begin = name.find('<');
  const size_t end = name.find('>', begin);
  ICHECK(begin != std::string::npos &&
         name.substr(0, begin) == "copy_gm_to_ub" && end != std::string::npos &&
         name.find_first_not_of(" \t\r\n", end + 1) == std::string::npos)
      << kError << "requires literal template type, columns and optional rows";
  std::vector<std::string> fields;
  for (size_t position = begin + 1; position <= end;) {
    const size_t comma = name.find(',', position);
    const size_t next = comma == std::string::npos || comma > end ? end : comma;
    fields.push_back(Trim(name.substr(position, next - position)));
    position = next + 1;
  }
  ICHECK(fields.size() == 2 || fields.size() == 3)
      << kError << "invalid template arity";
  auto dimension = [&](size_t index) -> int64_t {
    uint64_t value = 0;
    const std::string &field = fields[index];
    const auto parsed =
        std::from_chars(field.data(), field.data() + field.size(), value);
    ICHECK(parsed.ec == std::errc() &&
           parsed.ptr == field.data() + field.size() && value > 0 &&
           value <= std::numeric_limits<uint32_t>::max())
        << kError << "template dimensions must be positive uint32 literals";
    return static_cast<int64_t>(value);
  };
  return {fields[0], dimension(1), fields.size() == 3 ? dimension(2) : 1};
}

DataType DuplicateType(const std::string &type) {
  // Exact IsDuplicateSupported_v<T> from common.h, including bfloat16.
  const std::unordered_map<std::string, DataType> types = {
      {"half", DataType::Float(16)},        {"float", DataType::Float(32)},
      {"bfloat16_t", DataType::BFloat(16)}, {"int16_t", DataType::Int(16)},
      {"uint16_t", DataType::UInt(16)},     {"int32_t", DataType::Int(32)},
      {"uint32_t", DataType::UInt(32)}};
  auto found = types.find(CanonicalHelperElementType(type));
  ICHECK(found != types.end())
      << kError << "tail padding requires a type supported by Duplicate";
  return found->second;
}

const CallNode *Pointer(const PrimExpr &expression, int access,
                        DataType dtype) {
  const auto *pointer = expression.as<CallNode>();
  ICHECK(pointer != nullptr && pointer->op.same_as(builtin::tvm_access_ptr()) &&
         pointer->args.size() == 5)
      << kError << "requires an explicit access_ptr";
  const auto *annotation = pointer->args[0].as<CallNode>();
  const auto *mask = pointer->args[4].as<IntImmNode>();
  ICHECK(annotation != nullptr &&
         annotation->op.same_as(Op::Get("tir.type_annotation")) &&
         annotation->dtype == dtype)
      << kError << "template and access_ptr element types disagree";
  ICHECK(mask != nullptr && mask->value == access)
      << kError << "requires exact source-read/destination-write pointer masks";
  ICHECK(pointer->args[1].as<VarNode>() != nullptr)
      << kError << "pointer data must retain a Var identity";
  Int64(pointer->args[2], "pointer offset");
  Int64(pointer->args[3], "pointer extent");
  return pointer;
}

std::string OperationName(const CallNode *call, size_t *offset) {
  *offset = 0;
  std::string name;
  if (call->op.same_as(builtin::call_extern())) {
    *offset = 1;
    if (!call->args.empty())
      if (const auto *literal = call->args[0].as<StringImmNode>())
        name = literal->value;
  } else if (const auto *op = call->op.as<OpNode>()) {
    name = op->name;
  }
  for (const std::string prefix : {"tl::ascend::", "tl::", "tl."}) {
    if (name.rfind(prefix, 0) == 0) {
      name.erase(0, prefix.size());
      break;
    }
  }
  if (name.rfind("ascend_", 0) == 0)
    name.erase(0, 7);
  return name;
}

struct ManualEvent {
  Call call;
  int resource;
};

class OriginalEventCollector : public StmtExprVisitor {
public:
  std::vector<ManualEvent> events;
  bool has_pipeline{false};

private:
  void VisitStmt_(const AttrStmtNode *op) final {
    const int previous = resource_;
    if (op->attr_key == "resource_scope") {
      const auto *value = op->value.as<IntImmNode>();
      ICHECK(value != nullptr) << kError << "resource scope must be literal";
      resource_ = static_cast<int>(value->value);
    }
    StmtExprVisitor::VisitStmt_(op);
    resource_ = previous;
  }

  void VisitStmt_(const ForNode *op) final {
    for (const auto &[key, value] : op->annotations)
      has_pipeline |= std::string(key).find("pipeline") != std::string::npos;
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    size_t offset = 0;
    const std::string name = OperationName(op, &offset);
    if (name == "set_flag" || name == "wait_flag" || name == "auto_set_flag" ||
        name == "auto_wait_flag")
      events.push_back({GetRef<Call>(op), resource_});
    StmtExprVisitor::VisitExpr_(op);
  }

  int resource_{-1};
};

PrimExpr BufferElements(const Buffer &buffer, ProofAnalyzer *analyzer) {
  ICHECK(!buffer->shape.empty() && buffer->strides.empty())
      << kError << "GM bounds require a compact declared buffer";
  ICHECK(analyzer->CanProve(Int64(buffer->elem_offset, "GM element offset") ==
                            I64(0)))
      << kError << "GM bounds require an allocation-root view";
  PrimExpr elements = I64(1);
  const PrimExpr limit = I64(std::numeric_limits<int64_t>::max());
  for (const PrimExpr &shape : buffer->shape) {
    PrimExpr extent = Int64(shape, "GM buffer shape");
    ICHECK(analyzer->CanProve(extent >= I64(0)))
        << kError << "cannot prove nonnegative GM buffer shape";
    if (!is_one(elements))
      ICHECK(
          analyzer->CanProve(elements <= floordiv(limit, max(extent, I64(1)))))
          << kError << "cannot prove GM buffer capacity without overflow";
    elements = analyzer->Simplify(elements * extent);
  }
  return elements;
}

// PTO GMCopyCall uses saved physical descriptors, not the AscendC helper's
// template dimensions. Admit only the dense FP32/zero-pad descriptor for which
// both describe the same complete UB tile and the same GM row stride.
void AuditPtoDescriptor(const PrimFunc &function, const CallNode *call,
                        size_t offset, const Shape &shape,
                        const CallNode *source, const CallNode *destination,
                        const PrimExpr &rows, const PrimExpr &columns,
                        const PrimExpr &stride, ProofAnalyzer *analyzer) {
  const char *reject = "PTO tail-copy profile: ";
  ICHECK(shape.element_type == "float") << reject << "requires FP32";
  ICHECK(call->args.size() > offset + 5)
      << reject << "requires an explicit literal zero padding value";
  const PrimExpr &pad = call->args[offset + 5];
  const auto *integer_pad = pad.as<IntImmNode>();
  const auto *float_pad = pad.as<FloatImmNode>();
  ICHECK((integer_pad && integer_pad->value == 0) ||
         (float_pad && float_pad->value == 0.0))
      << reject << "requires an explicit literal zero padding value";
  const auto shapes =
      function->GetAttr<Map<Var, Array<PrimExpr>>>(kLogicBufferShapes);
  const Var source_data = Downcast<Var>(source->args[1]);
  const Var destination_data = Downcast<Var>(destination->args[1]);
  ICHECK(shapes.defined() && shapes.value().count(source_data) &&
         shapes.value().count(destination_data))
      << reject << "requires final GM/UB physical shape metadata";
  const auto &ub = shapes.value().at(destination_data);
  ICHECK(ub.size() == 2 || ub.size() == 4)
      << reject << "requires a dense two-dimensional UB descriptor";
  auto dimension = [&](const PrimExpr &value) {
    const auto *constant = value.as<IntImmNode>();
    ICHECK(constant && constant->value > 0 &&
           constant->value <= std::numeric_limits<int32_t>::max())
        << reject << "requires static positive int32 descriptor dimensions";
    return constant->value;
  };
  const int64_t physical_rows = dimension(ub[0]);
  const int64_t physical_columns = dimension(ub[1]);
  const int64_t valid_rows = ub.size() == 4 ? dimension(ub[2]) : physical_rows;
  const int64_t valid_columns =
      ub.size() == 4 ? dimension(ub[3]) : physical_columns;
  ICHECK(physical_rows == shape.rows && physical_columns == shape.columns &&
         valid_rows == physical_rows && valid_columns == physical_columns &&
         physical_rows <=
             std::numeric_limits<int32_t>::max() / physical_columns)
      << reject
      << "UB physical/template dimensions must match without padding "
         "or retyped views";
  ICHECK(analyzer->CanProve(Int64(destination->args[2], "PTO UB offset") ==
                                I64(0) &&
                            Int64(destination->args[3], "PTO UB extent") ==
                                I64(physical_rows * physical_columns)))
      << reject << "requires a complete nonsliced UB root view";
  size_t allocations = 0;
  PostOrderVisit(function->body, [&](const ObjectRef &node) {
    const auto *allocation = node.as<AllocateNode>();
    if (allocation && allocation->buffer_var.same_as(destination_data)) {
      ++allocations;
      // Codegen records the Allocate dtype separately from logical views and
      // disables padding on retyped pointers. Check that actual dtype too.
      ICHECK(allocation->dtype == DataType::Float(32))
          << reject << "requires an FP32 UB allocation without retyped views";
    }
  });
  ICHECK_EQ(allocations, 1U)
      << reject << "requires one explicit UB allocation root";
  size_t parameters = 0;
  for (const auto &[parameter, buffer] : function->buffer_map) {
    if (buffer->data.same_as(source_data)) {
      ++parameters;
      ICHECK(buffer->dtype == DataType::Float(32))
          << reject << "requires an FP32 GM parameter without retyped views";
    }
  }
  ICHECK_EQ(parameters, 1U)
      << reject << "requires one explicit GM parameter root";
  const auto &gm = shapes.value().at(source_data);
  ICHECK(gm.size() >= 2 && gm.size() <= 4)
      << reject << "requires static dense GM shape metadata";
  int64_t suffix = 1;
  for (size_t index = gm.size(); index != 0; --index) {
    const int64_t extent = dimension(gm[index - 1]);
    ICHECK(suffix <= std::numeric_limits<int32_t>::max() / extent)
        << reject << "GM descriptor strides exceed int32";
    suffix *= extent;
  }
  // ComputeStrides does not merge dimensions when srcN equals the final
  // dimension; its stride[3] is then exactly that dimension, stride[4] is 1.
  ICHECK(
      analyzer->CanProve(stride == I64(dimension(gm.back())) &&
                         rows <= I64(std::numeric_limits<int32_t>::max()) &&
                         columns <= I64(std::numeric_limits<int32_t>::max())))
      << reject << "GM stride/masks must match the dense int32 descriptor";
}

} // namespace

struct LegacyTailCopyPlan::State {
  PrimFunc function;
  Target target;
  String platform;
  bool has_tail{false};
  bool has_pipeline{false};
  std::vector<ManualEvent> manual_events;
  std::map<std::string, std::set<int>> reservations;

  bool IsPto() const {
    return target.defined() &&
           target->GetAttr<String>("model").value_or(String()) == "pto";
  }

  void RequireSupportedFunction() const {
    const String model =
        target.defined() ? target->GetAttr<String>("model").value_or(String())
                         : String();
    ICHECK(((platform == "A2" || platform == "A3") &&
            (model == "ascendc" || model == "auto")) ||
           (platform == "A3" && IsPto()))
        << kError << "requires AscendC on A2/A3 or the bounded PTO A3 profile";
    ICHECK(!has_pipeline)
        << kError << "software pipelines require a later tail-ring contract";
    for (const char *key : {"buffer_versions", "tl.buffer_versions_map"}) {
      if (auto versions = function->GetAttr<Map<Var, PrimExpr>>(key))
        for (const auto &[storage, version] : versions.value()) {
          const auto *count = version.as<IntImmNode>();
          ICHECK(count != nullptr && count->value == 1)
              << kError << "requires physical single-version buffers";
        }
    }
    if (auto rings =
            function->GetAttr<PhysicalRingProvenance>(kPhysicalRingProvenance))
      ICHECK(rings.value().empty())
          << kError << "physical rings require a later tail contract";
  }
};

LegacyTailCopyPlan::LegacyTailCopyPlan(const PrimFunc &function,
                                       const Target &target,
                                       const String &platform)
    : state_(std::make_shared<State>()) {
  state_->function = function;
  state_->target = target;
  state_->platform = platform;
  OriginalEventCollector collector;
  collector(function->body);
  state_->manual_events = std::move(collector.events);
  state_->has_pipeline = collector.has_pipeline;
}

std::optional<LegacyTailCopyEffect>
LegacyTailCopyPlan::AuditCall(const CallNode *call, size_t offset,
                              const ConstrSet &context,
                              const PhysicalStorageResolver &storage) {
  ICHECK(call != nullptr && call->args.size() >= offset + 2)
      << kError << "malformed GM-to-UB buffer arguments";
  // C++ defaults are full masks. Preserve their existing target/ABI admission.
  if (call->args.size() <= offset + 3)
    return std::nullopt;
  const Shape shape = ParseShape(call, offset);
  ProofAnalyzer analyzer;
  context.Populate(analyzer);
  const PrimExpr rows = Int64(call->args[offset + 3], "row mask");
  const PrimExpr columns = call->args.size() > offset + 4
                               ? Int64(call->args[offset + 4], "column mask")
                               : I64(shape.columns);
  if (analyzer.CanProve(rows == I64(shape.rows) &&
                        columns == I64(shape.columns)))
    return std::nullopt;

  state_->RequireSupportedFunction();
  ICHECK_LE(call->args.size(), offset + 8)
      << kError << "unsupported trailing copy ABI";
  const DataType dtype = DuplicateType(shape.element_type);
  const int64_t bytes = dtype.bytes();
  // Count-form Duplicate takes int32_t; DataCopyExtParams narrows row count to
  // uint16_t and length/strides to uint32_t. A small-row first contract also
  // stays within the existing helper's documented 12-bit block-count domain.
  ICHECK(shape.rows <= 4095 &&
         shape.columns <= std::numeric_limits<int32_t>::max() / shape.rows)
      << kError << "template tile exceeds Duplicate/DMA count limits";
  const int64_t elements = shape.rows * shape.columns;
  ICHECK(shape.columns <= std::numeric_limits<uint32_t>::max() / bytes)
      << kError << "DMA row byte length overflows";
  const int64_t row_bytes = shape.columns * bytes;
  ICHECK_EQ(row_bytes % 32, 0)
      << kError << "tail DMA requires a complete 32-byte-aligned row pitch";
  const PrimExpr stride = Int64(call->args[offset + 2], "GM row stride");
  ICHECK(analyzer.CanProve(
      rows >= I64(0) && rows <= I64(shape.rows) && columns >= I64(0) &&
      columns <= I64(shape.columns) && stride >= columns &&
      stride <= I64(std::numeric_limits<uint32_t>::max()) &&
      (stride - columns) * I64(bytes) <=
          I64(std::numeric_limits<uint32_t>::max())))
      << kError
      << "cannot prove nonnegative bounded masks and nonwrapping DMA stride";
  const PrimExpr nonempty = rows > I64(0) && columns > I64(0);
  if (state_->IsPto())
    ICHECK(analyzer.CanProve(nonempty))
        << "PTO tail-copy profile: requires positive masks";
  if (call->args.size() > offset + 5) {
    const PrimExpr &pad = call->args[offset + 5];
    const DataType pad_type = pad.dtype();
    ICHECK(pad_type.is_scalar() &&
           (pad_type.is_int() || pad_type.is_uint() || pad_type.is_float() ||
            pad_type.is_bfloat16()) &&
           SideEffect(pad) <= CallEffectKind::kPure)
        << kError << "padding value must be a captured arithmetic scalar";
  }
  // These physical-shape metadata operands are not forwarded by CopyCodegen.
  // Check their representation, but never mistake them for runtime masks.
  for (size_t index = offset + 6; index < call->args.size(); ++index)
    ICHECK(analyzer.CanProve(
        Int64(call->args[index], "physical-shape metadata") > I64(0)))
        << kError << "invalid physical-shape metadata";

  const CallNode *source = Pointer(call->args[offset], 1, dtype);
  const CallNode *destination = Pointer(call->args[offset + 1], 2, dtype);
  const Var source_data = Downcast<Var>(source->args[1]);
  const Var destination_data = Downcast<Var>(destination->args[1]);
  ICHECK(!storage.IsPhysicalRingStorage(source_data) &&
         !storage.IsPhysicalRingStorage(destination_data))
      << kError << "physical rings require a later tail contract";
  const Buffer source_buffer = storage.LookupBuffer(source_data);
  const Buffer destination_buffer = storage.LookupBuffer(destination_data);
  ICHECK(source_buffer.scope() == "global" &&
         destination_buffer.scope() == "shared.ub" &&
         source_buffer->dtype == dtype && destination_buffer->dtype == dtype)
      << kError << "requires matching GM and UB element types/scopes";
  const PrimExpr source_offset = Int64(source->args[2], "GM pointer offset");
  const PrimExpr destination_offset =
      Int64(destination->args[2], "UB pointer offset");
  const PrimExpr dma_span = (rows - I64(1)) * stride + columns;
  const PrimExpr source_capacity = BufferElements(source_buffer, &analyzer);
  if (state_->IsPto())
    AuditPtoDescriptor(state_->function, call, offset, shape, source,
                       destination, rows, columns, stride, &analyzer);
  ICHECK(analyzer.CanProve(
      (!nonempty || (source_offset >= I64(0) && dma_span <= source_capacity &&
                     source_offset <= source_capacity - dma_span)) &&
      Int64(source->args[3], "GM pointer extent") >= I64(0) &&
      Int64(destination->args[3], "UB pointer extent") >= I64(0)))
      << kError << "cannot prove the complete strided GM read is in bounds";
  const auto addresses =
      state_->function->GetAttr<Map<Var, PrimExpr>>("address_map");
  const auto sizes = state_->function->GetAttr<Map<Var, PrimExpr>>("size_map");
  ICHECK(addresses.defined() && sizes.defined() &&
         addresses.value().count(destination_data) &&
         sizes.value().count(destination_data))
      << kError
      << "requires the final UB allocation identity and address/size maps";
  const auto *address = addresses.value().at(destination_data).as<IntImmNode>();
  const auto *size = sizes.value().at(destination_data).as<IntImmNode>();
  ICHECK(address != nullptr && size != nullptr && address->value >= 0 &&
         size->value > 0 &&
         address->value <= std::numeric_limits<int64_t>::max() - size->value)
      << kError << "requires a static nonnegative UB allocation span";
  const int64_t tile_bytes = elements * bytes;
  if (state_->IsPto())
    ICHECK(address->value <= std::numeric_limits<int32_t>::max() - tile_bytes)
        << "PTO tail-copy profile: UB address/span exceeds int32";
  ICHECK(tile_bytes <= size->value &&
         analyzer.CanProve(
             destination_offset >= I64(0) &&
             destination_offset <= I64((size->value - tile_bytes) / bytes) &&
             floormod(I64(address->value) + destination_offset * I64(bytes),
                      I64(32)) == I64(0)))
      << kError
      << "whole padded UB tile is misaligned or exceeds its allocation";

  BufferRegion source_region = storage.AccessPtrRegion(source, {}, context);
  BufferRegion destination_region =
      storage.AccessPtrRegion(destination, {}, context);
  // Empty AscendC tiles still Duplicate the whole UB tile and complete on
  // MTE2, but the helper returns before touching GM. Normalize both bounds
  // of their empty read so an inactive slice cannot manufacture an address.
  const PrimExpr source_begin =
      analyzer.Simplify(Select(nonempty, source_offset, I64(0)));
  const PrimExpr source_span =
      analyzer.Simplify(Select(nonempty, dma_span, I64(0)));
  source_region = BufferRegion(
      source_region->buffer, {Range::FromMinExtent(source_begin, source_span)});
  destination_region =
      BufferRegion(destination_region->buffer,
                   {Range::FromMinExtent(destination_offset, I64(elements))});
  state_->has_tail = true;
  state_->reservations =
      state_->IsPto() ? std::map<std::string, std::set<int>>{{"MTE2_V", {0}},
                                                             {"V_MTE2", {0}},
                                                             {"V_MTE3", {0}}}
                      : std::map<std::string, std::set<int>>{
                            {"MTE2_V", {0}}, {"MTE3_V", {0}}, {"V_MTE2", {0}}};
  return LegacyTailCopyEffect{std::move(source_region),
                              std::move(destination_region)};
}

const std::map<std::string, std::set<int>> &
LegacyTailCopyPlan::ReservedVectorIntraIds() const {
  return state_->reservations;
}

bool LegacyTailCopyPlan::HasTailCopies() const { return state_->has_tail; }

void LegacyTailCopyPlan::ValidateManualEvents() const {
  if (!state_->has_tail)
    return;
  for (const auto &[call, resource] : state_->manual_events) {
    if (resource == 0)
      continue; // Cube and Vector own independent physical event pools.
    size_t offset = 0;
    const std::string name = OperationName(call.get(), &offset);
    const bool automatic = name == "auto_set_flag" || name == "auto_wait_flag";
    const size_t count = automatic ? 2 : 3;
    ICHECK_EQ(call->args.size(), offset + count)
        << kError << "malformed explicit event";
    auto pipe = [&](size_t index) {
      const auto *literal = call->args[offset + index].as<StringImmNode>();
      ICHECK(literal != nullptr)
          << kError << "explicit event direction must be literal";
      std::string value = literal->value;
      if (value.rfind("PIPE_", 0) == 0)
        value.erase(0, 5);
      return value;
    };
    const std::string event = automatic ? pipe(0) : pipe(0) + "_" + pipe(1);
    const auto *id = call->args[offset + count - 1].as<IntImmNode>();
    ICHECK(id != nullptr && id->value >= 0 && id->value < 8)
        << kError << "explicit event ID must be a literal in [0, 8)";
    auto reserved = state_->reservations.find(event);
    ICHECK(reserved == state_->reservations.end() ||
           !reserved->second.count(id->value))
        << kError << "explicit event conflicts with hidden " << event << "#"
        << id->value;
  }
}

} // namespace tvm::tl::ascend_sync
