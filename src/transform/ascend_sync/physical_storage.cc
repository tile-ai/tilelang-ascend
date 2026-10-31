// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

#include "physical_storage.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ir/type.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../common/pipeline_ring_provenance.h"
#include "legacy_runtime_contract.h"

namespace tvm::tl::ascend_sync {

namespace {

using namespace tir;

PrimExpr AsInt64(const PrimExpr &value, const char *description) {
  const DataType dtype = value.dtype();
  ICHECK(dtype.lanes() == 1 &&
         (dtype.is_int() || (dtype.is_uint() && dtype.bits() < 64)))
      << "Unsupported " << description
      << " dtype for physical storage: " << dtype;
  return cast(DataType::Int(64), value);
}

int64_t ElementBytes(DataType dtype) {
  ICHECK(!dtype.is_handle() && dtype.bits() > 0 && dtype.lanes() > 0)
      << "Unsupported physical storage element dtype: " << dtype;
  ICHECK(dtype.is_bool() || dtype.bits() % 8 == 0)
      << "Packed sub-byte storage needs an explicit physical footprint: "
      << dtype;
  return static_cast<int64_t>(dtype.bytes()) * dtype.lanes();
}

std::string PoolForScope(const String &scope) {
  if (scope.empty() || scope == "global")
    return "gm";
  if (scope == "shared" || scope == "shared.dyn" || scope == "shared.ub")
    return "ub";
  if (scope == "shared.l1" || scope == "shared.l1.dyn")
    return "l1";
  if (scope == "wmma.matrix_a" || scope == "shared.l0a" ||
      scope == "shared.l0a.dyn")
    return "l0a";
  if (scope == "wmma.matrix_b" || scope == "shared.l0b" ||
      scope == "shared.l0b.dyn")
    return "l0b";
  if (scope == "wmma.accumulator" || scope == "shared.l0c" ||
      scope == "shared.l0c.dyn")
    return "l0c";
  if (scope == "shared.ss" || scope == "shared.ss.dyn")
    return "ss";
  if (scope == "local" || scope == "local.var" || scope == "local.fragment")
    return "local";
  LOG(FATAL) << "Unknown or unsupported pointer storage scope: " << scope;
  return {};
}

Buffer BufferWithData(const Var &data, DataType dtype,
                      const Array<PrimExpr> &shape, const String &name) {
  return Buffer(data, dtype, shape, {}, IntImm(DataType::Int(32), 0), name, 0,
                0, kDefault);
}

void CheckNonnegativeConstant(const PrimExpr &value, const char *description) {
  if (const auto *constant = value.as<IntImmNode>()) {
    ICHECK_GE(constant->value, 0)
        << "Negative physical storage " << description;
  }
}

} // namespace

struct PhysicalStorageResolver::State {
  struct Record {
    Buffer buffer;
    Optional<Var> alias_source;
    bool parameter{false};
  };

  struct Ring {
    Var logical_id;
    Var root;
    int64_t base_bytes;
    int64_t stride_bytes;
    int64_t element_bits;
    int versions;
  };

  PrimFunc function;
  Map<Var, PrimExpr> addresses;
  Map<Var, PrimExpr> sizes;
  bool noalias{false};
  std::unordered_map<Var, Record, ObjectPtrHash, ObjectPtrEqual> records;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> parameter_data;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> fresh_gm_data;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> accessed_data;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> scalar_data;
  std::unordered_map<Var, Ring, ObjectPtrHash, ObjectPtrEqual> rings;
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> root_to_ring;
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual>
      invocation_views;
  // Access-pointer views are intentionally distinct from scalar-buffer views:
  // offset/extent alone does not specify a lowered DMA's actual footprint.
  std::vector<Buffer> access_pointer_views;

  State(PrimFunc function, Map<Var, PrimExpr> addresses,
        Map<Var, PrimExpr> sizes)
      : function(std::move(function)), addresses(std::move(addresses)),
        sizes(std::move(sizes)) {
    noalias =
        this->function->GetAttr<Bool>("tir.noalias").value_or(Bool(false));
    for (const auto &[parameter, buffer] : this->function->buffer_map) {
      RegisterBuffer(buffer);
      parameter_data.insert(buffer->data);
      records.at(buffer->data).parameter = true;
    }
    for (const Var &parameter : this->function->params) {
      if (parameter->type_annotation.as<PointerTypeNode>()) {
        EnsureRecord(parameter);
        parameter_data.insert(parameter);
        records.at(parameter).parameter = true;
      }
    }
    Collector(this)(this->function->body);
    for (const Var &storage : ReadTrustedFreshGMStorages(this->function))
      fresh_gm_data.insert(storage);
    for (const Var &data : accessed_data)
      ValidateMaps(StorageRoot(data));
    ReadRings();
  }

  void RegisterBuffer(const Buffer &buffer) {
    ICHECK(buffer.defined()) << "Undefined buffer in physical storage input";
    PoolForScope(buffer.scope());
    auto it = records.find(buffer->data);
    if (it == records.end()) {
      records.emplace(buffer->data, Record{buffer, NullOpt, false});
    } else {
      ICHECK_EQ(PoolForScope(it->second.buffer.scope()),
                PoolForScope(buffer.scope()))
          << "One pointer has incompatible physical address spaces";
      it->second.buffer = buffer;
    }
  }

  Record &EnsureRecord(const Var &data) {
    auto found = records.find(data);
    if (found != records.end())
      return found->second;
    const auto *pointer = data->type_annotation.as<PointerTypeNode>();
    ICHECK(pointer != nullptr)
        << "Unknown pointer identity or missing PointerType for " << data;
    const auto *element = pointer->element_type.as<PrimTypeNode>();
    ICHECK(element != nullptr)
        << "Unknown pointer element type in physical storage: " << data;
    PoolForScope(pointer->storage_scope);
    Buffer buffer =
        BufferWithData(data, element->dtype, {IntImm(DataType::Int(64), 1)},
                       "physical_storage_pointer");
    return records.emplace(data, Record{buffer, NullOpt, false}).first->second;
  }

  class Collector : public StmtExprVisitor {
  public:
    explicit Collector(State *state) : state_(state) {}

  private:
    void VisitStmt_(const AllocateNode *op) final {
      state_->RegisterBuffer(BufferWithData(
          op->buffer_var, op->dtype, op->extents, "physical_storage_allocate"));
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitStmt_(const DeclBufferNode *op) final {
      state_->RegisterBuffer(op->buffer);
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitStmt_(const BlockNode *op) final {
      for (const Buffer &buffer : op->alloc_buffers)
        state_->RegisterBuffer(buffer);
      for (const MatchBufferRegion &match : op->match_buffers) {
        state_->RegisterBuffer(match->source->buffer);
        state_->RegisterBuffer(match->buffer);
        auto &record = state_->records.at(match->buffer->data);
        if (!match->buffer->data.same_as(match->source->buffer->data))
          record.alias_source = match->source->buffer->data;
      }
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitStmt_(const BufferStoreNode *op) final {
      state_->RegisterBuffer(op->buffer);
      state_->accessed_data.insert(op->buffer->data);
      state_->scalar_data.insert(op->buffer->data);
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitExpr_(const BufferLoadNode *op) final {
      state_->RegisterBuffer(op->buffer);
      state_->accessed_data.insert(op->buffer->data);
      state_->scalar_data.insert(op->buffer->data);
      StmtExprVisitor::VisitExpr_(op);
    }
    void VisitExpr_(const CallNode *op) final {
      if (op->op.same_as(builtin::tvm_access_ptr()) && op->args.size() == 5) {
        if (const auto *data = op->args[1].as<VarNode>())
          state_->accessed_data.insert(GetRef<Var>(data));
      }
      StmtExprVisitor::VisitExpr_(op);
    }
    void VisitExpr_(const VarNode *op) final {
      if (op->type_annotation.as<PointerTypeNode>())
        state_->EnsureRecord(GetRef<Var>(op));
    }
    State *state_;
  };

  Var StorageRoot(const Var &data) const {
    auto ring = rings.find(data);
    if (ring != rings.end())
      return ring->second.root;
    Var current = data;
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> visited;
    while (true) {
      ICHECK(visited.insert(current).second)
          << "Cyclic match-buffer pointer alias in physical storage";
      const auto found = records.find(current);
      ICHECK(found != records.end())
          << "Unknown logical pointer identity in physical storage: " << data;
      if (!found->second.alias_source.defined())
        return current;
      // An explicitly planned view has its own physical map. Otherwise keep
      // its root allocation conservatively; no view-offset guess is made.
      if (addresses.count(current) && sizes.count(current))
        return current;
      current = found->second.alias_source.value();
    }
  }

  int64_t StaticNonnegative(const PrimExpr &expression,
                            const char *description) const {
    const auto *constant = expression.as<IntImmNode>();
    ICHECK(constant != nullptr && constant->value >= 0)
        << "Physical ring requires a static nonnegative " << description;
    return constant->value;
  }

  void ReadRings() {
    auto provenance =
        function->GetAttr<PhysicalRingProvenance>(kPhysicalRingProvenance);
    if (!provenance.defined())
      return;
    for (const auto &[logical_id, fields] : provenance.value()) {
      ValidatePhysicalRingRecord(fields);
      const int64_t versions =
          StaticNonnegative(fields[kPhysicalRingVersions], "version count");
      // One logical version needs one intra-core flag slot in the manual
      // clock protocol. This is only a necessary bound: the engine's actual
      // per-event allocation must also fit every other edge and reservation.
      ICHECK(versions >= 2 && versions <= 8)
          << "Physical ring version count exceeds the eight-slot intra-core "
             "event capacity";
      ICHECK_EQ(StaticNonnegative(fields[kPhysicalRingOriginalMin],
                                  "original loop minimum"),
                0)
          << "Physical ring adapter requires original loop minimum zero";
      const Var root = Downcast<Var>(fields[kPhysicalRingRoot]);
      EnsureRecord(root);
      ICHECK(StorageRoot(root).same_as(root))
          << "Physical ring root cannot be an unresolved match-buffer alias";
      const std::string pool = PoolForScope(records.at(root).buffer.scope());
      ICHECK(pool == "ub" || pool == "l1")
          << "Physical ring adapter requires UB or audited L1 storage";
      ICHECK(!parameter_data.count(root))
          << "Physical ring must refer to a local allocation";
      ValidateMaps(root);
      const int64_t base_bits =
          StaticNonnegative(fields[kPhysicalRingBaseBits], "base bits");
      const int64_t stride_bits =
          StaticNonnegative(fields[kPhysicalRingStrideBits], "stride bits");
      const int64_t element_bits = StaticNonnegative(
          fields[kPhysicalRingElementBits], "original element bits");
      ICHECK(stride_bits > 0 && element_bits > 0 && base_bits % 8 == 0 &&
             stride_bits % 8 == 0 && element_bits % 8 == 0 &&
             base_bits % element_bits == 0 && stride_bits % element_bits == 0)
          << "Physical ring requires aligned, nonzero element/slot bit spans";
      const int64_t address =
          StaticNonnegative(addresses.at(root), "allocation address");
      const int64_t size = StaticNonnegative(sizes.at(root), "allocation size");
      const int64_t base = base_bits / 8, stride = stride_bits / 8;
      ICHECK(address % 32 == 0 && base % 32 == 0 && stride % 32 == 0)
          << "Physical ring slot/allocation must have 32-byte alignment";
      ICHECK(base <= size && stride <= (size - base) / versions &&
             address <= std::numeric_limits<int64_t>::max() - size)
          << "Physical ring slot span exceeds its final allocation/maps";

      const Buffer &allocation = records.at(root).buffer;
      int64_t declared_bytes = ElementBytes(allocation->dtype);
      for (const PrimExpr &extent : allocation->shape) {
        const int64_t count = StaticNonnegative(extent, "allocation shape");
        ICHECK(count > 0 &&
               declared_bytes <= std::numeric_limits<int64_t>::max() / count)
            << "Physical ring allocation shape is empty or overflows";
        declared_bytes *= count;
      }
      ICHECK(base <= declared_bytes &&
             stride <= (declared_bytes - base) / versions &&
             declared_bytes <= size)
          << "Physical ring provenance disagrees with allocation shape/maps";

      Record &logical = EnsureRecord(logical_id);
      ICHECK_EQ(PoolForScope(logical.buffer.scope()), pool)
          << "Logical physical-ring identity must retain its pointer scope";
      ICHECK(logical.buffer->dtype.lanes() == 1 &&
             ElementBytes(logical.buffer->dtype) * 8 == element_bits)
          << "Physical ring original element bits disagree with logical dtype";
      ICHECK(root_to_ring.emplace(root, logical_id).second)
          << "Multiple logical physical rings sharing one allocation root are "
             "unsupported";
      rings.emplace(logical_id, Ring{logical_id, root, base, stride,
                                     element_bits, static_cast<int>(versions)});
    }

    auto physical_allocations = accessed_data;
    for (const auto &[logical_id, ring] : rings)
      physical_allocations.insert(ring.root);
    for (const auto &[logical_id, ring] : rings) {
      const std::string pool =
          PoolForScope(records.at(ring.root).buffer.scope());
      for (const Var &data : scalar_data) {
        ICHECK(!StorageRoot(data).same_as(ring.root))
            << "Physical ring scalar accesses require the later snapshot/slot "
               "adapter";
      }
      const int64_t begin =
          StaticNonnegative(addresses.at(ring.root), "allocation address");
      const int64_t size =
          StaticNonnegative(sizes.at(ring.root), "allocation size");
      for (const Var &data : physical_allocations) {
        const Var other = StorageRoot(data);
        if (other.same_as(ring.root) ||
            PoolForScope(records.at(other).buffer.scope()) != pool)
          continue;
        const int64_t other_begin =
            StaticNonnegative(addresses.at(other), "other allocation address");
        const int64_t other_size =
            StaticNonnegative(sizes.at(other), "other allocation size");
        const bool disjoint =
            size == 0 || other_size == 0 ||
            (begin <= other_begin ? other_begin - begin >= size
                                  : begin - other_begin >= other_size);
        ICHECK(disjoint)
            << "Physical ring allocation aliases another logical allocation; "
               "distinct storage epochs require a later alias protocol";
      }
    }
  }

  void ValidateClock(const Var &logical_id, const PrimExpr &clock,
                     const ConstrSet &context) const {
    ICHECK(rings.count(logical_id)) << "Unknown physical ring clock identity";
    ICHECK(clock.defined() && SideEffect(clock) <= CallEffectKind::kPure)
        << "Physical ring clock must be pure";
    const PrimExpr index = AsInt64(clock, "ring clock");
    ICHECK(context.CanProve(index >= make_zero(DataType::Int(64))))
        << "Physical ring clock must be provably nonnegative in its execution "
           "context";
  }

  void ValidateMaps(const Var &root) const {
    const auto record = records.find(root);
    ICHECK(record != records.end());
    const std::string pool = PoolForScope(record->second.buffer.scope());
    if (pool == "gm" || pool == "local")
      return;
    auto address = addresses.Get(root);
    auto size = sizes.Get(root);
    ICHECK(address.defined() && size.defined())
        << "Missing address_map/size_map entry for local storage " << root;
    AsInt64(address.value(), "address");
    AsInt64(size.value(), "size");
    CheckNonnegativeConstant(address.value(), "address");
    CheckNonnegativeConstant(size.value(), "size");
  }

  Buffer Lookup(const Var &data) {
    Record &record = EnsureRecord(data);
    ValidateMaps(StorageRoot(data));
    return record.buffer;
  }

  BufferRegion InvocationProjection(const BufferRegion &region) {
    const Var root = StorageRoot(region->buffer->data);
    ValidateMaps(root);
    auto found = invocation_views.find(root);
    if (found == invocation_views.end()) {
      const Buffer &original = records.at(root).buffer;
      const std::string pool = PoolForScope(original.scope());
      int64_t extent = 0;
      if (pool == "gm") {
        extent = ElementBytes(original->dtype);
        for (const PrimExpr &dimension : original->shape) {
          const int64_t count = StaticNonnegative(dimension, "GM shape");
          ICHECK(count > 0 &&
                 extent <= std::numeric_limits<int64_t>::max() / count)
              << "Invocation projection requires bounded nonempty GM shapes";
          extent *= count;
        }
      } else {
        ICHECK_EQ(pool, "ub")
            << "Invocation projection currently supports only GM/UB";
        extent = StaticNonnegative(sizes.at(root), "allocation size");
      }
      ICHECK_GT(extent, 0) << "Empty invocation projection";
      Var identity("invocation_physical_view",
                   PointerType(PrimType(DataType::UInt(8)), original.scope()));
      Buffer view = BufferWithData(identity, DataType::UInt(8),
                                   {IntImm(DataType::Int(64), extent)},
                                   "invocation_physical_view");
      RegisterBuffer(view);
      records.at(identity).alias_source = root;
      found = invocation_views.emplace(root, view).first;
    }
    return BufferRegion(found->second,
                        {Range::FromMinExtent(IntImm(DataType::Int(64), 0),
                                              found->second->shape[0])});
  }

  BufferRegion AccessPtr(const CallNode *call,
                         const PhysicalRingClockMap &clocks,
                         const ConstrSet &context) {
    ICHECK(call != nullptr && call->op.same_as(builtin::tvm_access_ptr()))
        << "AccessPtrRegion expects tir.tvm_access_ptr";
    ICHECK_EQ(call->args.size(), 5) << "Malformed tvm_access_ptr";
    const auto *annotation = call->args[0].as<CallNode>();
    ICHECK(annotation && annotation->op.same_as(Op::Get("tir.type_annotation")))
        << "tvm_access_ptr is missing its element type annotation";
    const auto *variable = call->args[1].as<VarNode>();
    ICHECK(variable != nullptr)
        << "tvm_access_ptr data must have a known Var identity";
    Var data = GetRef<Var>(variable);
    Lookup(data);
    const DataType dtype = annotation->dtype;
    ElementBytes(dtype);
    AsInt64(call->args[2], "access offset");
    AsInt64(call->args[3], "access extent");
    CheckNonnegativeConstant(call->args[3], "access extent");
    const auto *mask = call->args[4].as<IntImmNode>();
    ICHECK(mask && mask->value > 0 && mask->value <= 3)
        << "tvm_access_ptr requires a constant read/write mask";

    PrimExpr offset = call->args[2], extent = call->args[3];
    auto root_ring = root_to_ring.find(StorageRoot(data));
    if (root_ring != root_to_ring.end()) {
      const Ring &ring = rings.at(root_ring->second);
      ICHECK(data.same_as(ring.root))
          << "Physical ring access must use its final allocation root";
      auto clock = clocks.Get(ring.logical_id);
      ICHECK(clock.defined())
          << "Physical ring access is missing injector clock provenance";
      ValidateClock(ring.logical_id, clock.value(), context);
      ICHECK(dtype.lanes() == 1 && ElementBytes(dtype) * 8 == ring.element_bits)
          << "Physical ring access dtype disagrees with original element bits";
      ICHECK(SideEffect(offset) <= CallEffectKind::kPure &&
             SideEffect(extent) <= CallEffectKind::kPure)
          << "Physical ring pointer offset/extent must be pure";
      const int64_t bytes = ElementBytes(dtype);
      const PrimExpr slot = floormod(AsInt64(clock.value(), "ring clock"),
                                     IntImm(DataType::Int(64), ring.versions));
      const PrimExpr slot_begin =
          IntImm(DataType::Int(64), ring.base_bytes / bytes) +
          slot * IntImm(DataType::Int(64), ring.stride_bytes / bytes);
      const PrimExpr slot_extent =
          IntImm(DataType::Int(64), ring.stride_bytes / bytes);
      const PrimExpr actual_offset = AsInt64(offset, "ring access offset");
      const PrimExpr actual_extent = AsInt64(extent, "ring access extent");
      // This checks the pointer's declared interval against provenance; it
      // does NOT infer a DMA footprint from that interval. Its graph region
      // below always includes the entire padded stride published by injector.
      ICHECK(context.CanProve(actual_offset == slot_begin &&
                              actual_extent >= make_zero(DataType::Int(64)) &&
                              actual_extent <= slot_extent &&
                              actual_offset - slot_begin <=
                                  slot_extent - actual_extent))
          << "Physical ring pointer does not fit its clock-selected slot";
      data = ring.logical_id;
      offset = slot_begin;
      extent = slot_extent;
    }

    Buffer view;
    for (const Buffer &candidate : access_pointer_views) {
      if (candidate->data.same_as(data) && candidate->dtype == dtype) {
        view = candidate;
        break;
      }
    }
    if (!view.defined()) {
      // The view shape is descriptive only. Neither it nor the logical
      // pointer extent is used to shrink an opaque DMA footprint.
      view = BufferWithData(data, dtype, {IntImm(DataType::Int(64), 1)},
                            "logical_access_pointer");
      access_pointer_views.push_back(view);
    }
    return BufferRegion(view, {Range::FromMinExtent(offset, extent)});
  }

  bool IsOpaqueAccessPointer(const Buffer &buffer) const {
    for (const Buffer &candidate : access_pointer_views) {
      if (candidate.same_as(buffer))
        return true;
    }
    return false;
  }

  std::optional<Range> ByteRange(const BufferRegion &region,
                                 const Var &root) const {
    const Buffer &buffer = region->buffer;
    const std::string pool = PoolForScope(buffer.scope());
    const bool view_alias = !buffer->data.same_as(root);
    const bool opaque = IsOpaqueAccessPointer(buffer);

    if (rings.count(buffer->data)) {
      ICHECK(opaque && region->region.size() == 1)
          << "Physical ring footprint must come from a validated access_ptr";
      const int64_t bytes = ElementBytes(buffer->dtype);
      return Range::FromMinExtent(
          AsInt64(addresses.at(root), "ring allocation address") +
              AsInt64(region->region[0]->min, "ring slot offset") *
                  IntImm(DataType::Int(64), bytes),
          AsInt64(region->region[0]->extent, "ring slot extent") *
              IntImm(DataType::Int(64), bytes));
    }

    // Scalar flat UB/GM/register accesses have an exact element footprint.
    // Fractal and DMA footprints instead use final allocation spans below.
    const bool flat_scalar =
        !opaque && !view_alias && region->region.size() == 1 &&
        (pool == "ub" || pool == "gm" || pool == "local") &&
        (buffer->strides.empty() ||
         (buffer->strides.size() == 1 && is_one(buffer->strides[0])));
    if (flat_scalar) {
      const int64_t bytes = ElementBytes(buffer->dtype);
      PrimExpr begin = (AsInt64(buffer->elem_offset, "buffer element offset") +
                        AsInt64(region->region[0]->min, "region offset")) *
                       IntImm(DataType::Int(64), bytes);
      PrimExpr extent = AsInt64(region->region[0]->extent, "region extent") *
                        IntImm(DataType::Int(64), bytes);
      if (pool != "gm" && pool != "local")
        begin = begin + AsInt64(addresses.at(root), "address");
      return Range::FromMinExtent(begin, extent);
    }
    if (pool == "gm" || pool == "local")
      return std::nullopt;
    ValidateMaps(root);
    return Range::FromMinExtent(AsInt64(addresses.at(root), "address"),
                                AsInt64(sizes.at(root), "allocation size"));
  }

  std::optional<bool> MayConflict(const PhysicalConflictQuery &query) const {
    const Var &a = query.producer_region->buffer->data;
    const Var &b = query.consumer_region->buffer->data;
    const Var a_root = StorageRoot(a);
    const Var b_root = StorageRoot(b);
    ValidateMaps(a_root);
    ValidateMaps(b_root);
    const std::string a_pool = PoolForScope(records.at(a_root).buffer.scope());
    const std::string b_pool = PoolForScope(records.at(b_root).buffer.scope());
    if (a_pool != b_pool)
      return false;

    if (a_pool == "gm" && !a_root.same_as(b_root)) {
      if (parameter_data.count(a_root) && parameter_data.count(b_root) &&
          (noalias || fresh_gm_data.count(a_root) ||
           fresh_gm_data.count(b_root)))
        return false;
      return std::nullopt;
    }
    if (a_pool == "local" && !a_root.same_as(b_root))
      return false;

    // Byte overlap does not establish how unrelated logical epoch clocks
    // project onto each other. The engine must validate that protocol before
    // enabling versioned/cross-storage inputs; do not borrow either ring.
    if (query.iteration_offset != 0 && !a.same_as(b)) {
      if (a_pool != "gm") {
        auto a_address = addresses.Get(a_root), a_size = sizes.Get(a_root);
        auto b_address = addresses.Get(b_root), b_size = sizes.Get(b_root);
        if (a_address.defined() && a_size.defined() && b_address.defined() &&
            b_size.defined()) {
          const auto *a_begin = a_address.value().as<IntImmNode>();
          const auto *a_extent = a_size.value().as<IntImmNode>();
          const auto *b_begin = b_address.value().as<IntImmNode>();
          const auto *b_extent = b_size.value().as<IntImmNode>();
          if (a_begin && a_extent && b_begin && b_extent &&
              a_begin->value >= 0 && a_extent->value >= 0 &&
              b_begin->value >= 0 && b_extent->value >= 0) {
            // These complete allocation spans are independent of iteration
            // clocks. Ordered subtraction avoids overflowing begin + size.
            const bool disjoint =
                a_extent->value == 0 || b_extent->value == 0 ||
                (a_begin->value <= b_begin->value
                     ? b_begin->value - a_begin->value >= a_extent->value
                     : a_begin->value - b_begin->value >= b_extent->value);
            if (disjoint)
              return false;
          }
        }
      }
      return true;
    }

    const auto a_bytes = ByteRange(query.producer_region, a_root);
    const auto b_bytes = ByteRange(query.consumer_region, b_root);
    if (!a_bytes || !b_bytes)
      return std::nullopt;

    // This canonical buffer never enters TaskNode or emitted TIR. Its sole
    // purpose is comparing two byte ranges in one proven physical address
    // space using the original upstream context/iteration proof algorithm.
    Buffer pool = decl_buffer({IntImm(DataType::Int(64), 1)}, DataType::UInt(8),
                              "physical_byte_pool", "global");
    BufferRegion a_region(pool, {a_bytes.value()});
    BufferRegion b_region(pool, {b_bytes.value()});
    return RegionsMayConflict(query.producer_context, a_region,
                              query.consumer_context, b_region, query.loop,
                              query.iteration_offset, query.num_storage_owners,
                              {}, NullOpt, NullOpt);
  }
};

PhysicalStorageResolver::PhysicalStorageResolver(PrimFunc function)
    : PhysicalStorageResolver(
          function,
          function->GetAttr<Map<Var, PrimExpr>>("address_map")
              .value_or(Map<Var, PrimExpr>{}),
          function->GetAttr<Map<Var, PrimExpr>>("size_map")
              .value_or(Map<Var, PrimExpr>{})) {}

PhysicalStorageResolver::PhysicalStorageResolver(PrimFunc function,
                                                 Map<Var, PrimExpr> address_map,
                                                 Map<Var, PrimExpr> size_map)
    : state_(std::make_shared<State>(
          std::move(function), std::move(address_map), std::move(size_map))) {}

BufferRegion
PhysicalStorageResolver::AccessPtrRegion(const CallNode *access_ptr) const {
  return state_->AccessPtr(access_ptr, {}, {});
}

BufferRegion
PhysicalStorageResolver::AccessPtrRegion(const CallNode *access_ptr,
                                         const PhysicalRingClockMap &clocks,
                                         const ConstrSet &context) const {
  return state_->AccessPtr(access_ptr, clocks, context);
}

BufferVersionMap PhysicalStorageResolver::ManualRingVersions() const {
  BufferVersionMap result;
  for (const auto &[logical_id, ring] : state_->rings)
    result.Set(logical_id, ring.versions);
  return result;
}

bool PhysicalStorageResolver::IsPhysicalRingStorage(const Var &data) const {
  return state_->rings.count(data) ||
         state_->root_to_ring.count(state_->StorageRoot(data));
}

void PhysicalStorageResolver::ValidateRingClock(
    const Var &logical_id, const PrimExpr &clock,
    const ConstrSet &context) const {
  state_->ValidateClock(logical_id, clock, context);
}

Buffer PhysicalStorageResolver::LookupBuffer(const Var &data) const {
  return state_->Lookup(data);
}

BufferRegion
PhysicalStorageResolver::InvocationProjection(const BufferRegion &region) {
  return state_->InvocationProjection(region);
}

ConflictOracle PhysicalStorageResolver::MakeOracle() const {
  return [state = state_](const PhysicalConflictQuery &query) {
    return state->MayConflict(query);
  };
}

} // namespace tvm::tl::ascend_sync
