// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include <tvm/node/structural_equal.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/ascend_constraint.h"
#include "common/ascend_storage_alias.h"

namespace tvm {
namespace tl {
PrimExpr CheckedAscendStorageByteOffset(PrimExpr elements, int64_t bytes) {
  ICHECK(bytes > 0 && elements.dtype() == DataType::Int(64));
  if (const auto *constant = elements.as<IntImmNode>()) {
    __int128 value = __int128(constant->value) * bytes;
    ICHECK(value >= std::numeric_limits<int64_t>::min() &&
           value <= std::numeric_limits<int64_t>::max())
        << "Storage byte offset/extent multiplication exceeds signed int64";
    return IntImm(DataType::Int(64), static_cast<int64_t>(value));
  }
  return elements * IntImm(DataType::Int(64), bytes);
}
PrimExpr AddAscendStorageByteOffsets(PrimExpr left, PrimExpr right) {
  const auto *a = left.as<IntImmNode>(), *b = right.as<IntImmNode>();
  if (a && b) {
    __int128 value = __int128(a->value) + b->value;
    ICHECK(value >= std::numeric_limits<int64_t>::min() &&
           value <= std::numeric_limits<int64_t>::max())
        << "Storage byte/index offset addition exceeds signed int64";
    return IntImm(DataType::Int(64), static_cast<int64_t>(value));
  }
  return left + right;
}
namespace {
using namespace tir;
using Record = Map<String, ObjectRef>;

int64_t ElementBytes(DataType dtype) {
  ICHECK(dtype.is_scalar() && dtype.bits() > 0 && dtype.bits() % 8 == 0)
      << "Unsupported storage sub-byte/vector pointer dtype";
  return dtype.bytes();
}
PrimExpr Wide(PrimExpr value) { return PromoteAscendStorageIndex(value); }

void ValidatePhysicalExtent(DataType dtype, const Array<PrimExpr> &shape) {
  int64_t product = 1;
  int64_t bytes = (int64_t(dtype.bits()) * dtype.lanes() + 7) / 8;
  ICHECK(bytes > 0) << "Invalid storage physical allocation dtype";
  for (const PrimExpr &extent : shape) {
    const auto *constant = extent.as<IntImmNode>();
    if (!constant)
      return; // Symbolic domains remain symbolic and fail closed later.
    ICHECK(constant->dtype.is_int() && constant->value > 0 &&
           constant->value <= std::numeric_limits<int64_t>::max() / product)
        << "Storage physical allocation extent/product exceeds signed int64";
    product *= constant->value;
  }
  ICHECK(product <= std::numeric_limits<int64_t>::max() / bytes)
      << "Storage physical allocation byte extent exceeds signed int64";
}

struct Root {
  Var data;
  Buffer buffer;
  DataType dtype;
  Array<PrimExpr> shape;
};
class RootCollector : public StmtVisitor {
public:
  explicit RootCollector(PrimFunc function) {
    for (const Var &parameter : function->params)
      if (function->buffer_map.count(parameter))
        Add(function->buffer_map[parameter]);
    VisitStmt(function->body);
  }
  void Add(const Buffer &buffer, bool physical = false) {
    if (physical)
      ValidatePhysicalExtent(buffer->dtype, buffer->shape);
    if (seen.insert(buffer->data.get()).second)
      roots.push_back({buffer->data, buffer, buffer->dtype, buffer->shape});
  }
  std::vector<Root> roots;

private:
  void VisitStmt_(const BlockNode *block) final {
    for (const Buffer &buffer : block->alloc_buffers)
      Add(buffer, true);
    StmtVisitor::VisitStmt_(block);
  }
  void VisitStmt_(const AllocateNode *allocation) final {
    ValidatePhysicalExtent(allocation->dtype, allocation->extents);
    if (seen.insert(allocation->buffer_var.get()).second)
      roots.push_back({allocation->buffer_var, Buffer(), allocation->dtype,
                       allocation->extents});
    StmtVisitor::VisitStmt_(allocation);
  }
  std::unordered_set<const VarNode *> seen;
};

struct AliasBinding {
  Var root;
  PrimExpr byte_offset;
};

struct IntegerBounds {
  bool known{false};
  int64_t lower{0}, upper{0};
};
using BoundsMap = std::unordered_map<const VarNode *, IntegerBounds>;

IntegerBounds CheckedBounds(__int128 lower, __int128 upper) {
  ICHECK(lower >= std::numeric_limits<int64_t>::min() &&
         upper <= std::numeric_limits<int64_t>::max() && lower <= upper)
      << "Storage original index arithmetic exceeds signed int64 before "
         "normalization";
  return {true, static_cast<int64_t>(lower), static_cast<int64_t>(upper)};
}

// Do not ask the old Analyzer about a whole expression here: binding a large
// singleton Var can cause its simplifier to wrap a multiplication to zero.
IntegerBounds OriginalBounds(const PrimExpr &expression,
                             const BoundsMap &domains) {
  if (const auto *constant = expression.as<IntImmNode>()) {
    ICHECK(constant->dtype.is_int() || constant->dtype.is_bool() ||
           (constant->dtype.is_uint() && constant->dtype.bits() < 64))
        << "Unsupported storage integer literal";
    return {true, constant->value, constant->value};
  }
  if (const auto *variable = expression.as<VarNode>()) {
    auto found = domains.find(variable);
    return found == domains.end() ? IntegerBounds() : found->second;
  }
  if (const auto *cast_node = expression.as<CastNode>()) {
    IntegerBounds value = OriginalBounds(cast_node->value, domains);
    DataType dtype = cast_node->dtype;
    ICHECK(dtype.is_int() || dtype.is_bool() ||
           (dtype.is_uint() && dtype.bits() < 64))
        << "Unsupported storage index cast";
    if (value.known && !dtype.is_bool() && dtype.bits() < 64) {
      int64_t lower = dtype.is_uint() ? 0 : -(int64_t(1) << (dtype.bits() - 1));
      int64_t upper =
          (int64_t(1) << (dtype.bits() - (dtype.is_uint() ? 0 : 1))) - 1;
      ICHECK(value.lower >= lower && value.upper <= upper)
          << "Unsupported lossy storage index cast before normalization";
    }
    return dtype.is_bool() ? IntegerBounds{true, 0, 1} : value;
  }
  if (const auto *select = expression.as<SelectNode>()) {
    IntegerBounds condition = OriginalBounds(select->condition, domains);
    if (condition.known && condition.lower == condition.upper)
      return OriginalBounds(
          condition.lower ? select->true_value : select->false_value, domains);
    IntegerBounds a = OriginalBounds(select->true_value, domains),
                  b = OriginalBounds(select->false_value, domains);
    return a.known && b.known ? CheckedBounds(std::min(a.lower, b.lower),
                                              std::max(a.upper, b.upper))
                              : IntegerBounds();
  }
  if (const auto *logical_not = expression.as<NotNode>()) {
    OriginalBounds(logical_not->a, domains);
    return {true, 0, 1};
  }
  if (const auto *call = expression.as<CallNode>()) {
    ICHECK(call->op.same_as(builtin::if_then_else()) && call->args.size() == 3)
        << "Unsupported call in original storage index";
    return OriginalBounds(Select(call->args[0], call->args[1], call->args[2]),
                          domains);
  }
  auto binary = [&](const PrimExpr &left, const PrimExpr &right,
                    const char *operation) -> IntegerBounds {
    IntegerBounds a = OriginalBounds(left, domains),
                  b = OriginalBounds(right, domains);
    if (!a.known || !b.known)
      return {};
    __int128 al = a.lower, ah = a.upper, bl = b.lower, bh = b.upper;
    std::string op = operation;
    if (op == "+")
      return CheckedBounds(al + bl, ah + bh);
    if (op == "-")
      return CheckedBounds(al - bh, ah - bl);
    if (op == "*") {
      std::array<__int128, 4> products{al * bl, al * bh, ah * bl, ah * bh};
      return CheckedBounds(*std::min_element(products.begin(), products.end()),
                           *std::max_element(products.begin(), products.end()));
    }
    if (op == "min")
      return CheckedBounds(std::min(al, bl), std::min(ah, bh));
    if (op == "max")
      return CheckedBounds(std::max(al, bl), std::max(ah, bh));
    if (op == "/" || op == "//" || op == "%" || op == "mod") {
      ICHECK(!(bl <= 0 && bh >= 0))
          << "Possible zero divisor in original storage index";
      ICHECK(
          !(al == std::numeric_limits<int64_t>::min() && bl <= -1 && bh >= -1))
          << "Storage original index division exceeds signed int64";
      if (op == "%" || op == "mod") {
        __int128 magnitude = std::max(bl < 0 ? -bl : bl, bh < 0 ? -bh : bh);
        return CheckedBounds(op == "mod" && bl > 0 ? 0 : -magnitude + 1,
                             op == "mod" && bh < 0 ? 0 : magnitude - 1);
      }
      auto divide = [&](const __int128 x, const __int128 y) {
        __int128 quotient = x / y;
        if (op == "//" && x % y != 0 && ((x < 0) != (y < 0)))
          --quotient;
        return quotient;
      };
      std::array<__int128, 4> values{divide(al, bl), divide(al, bh),
                                     divide(ah, bl), divide(ah, bh)};
      return CheckedBounds(*std::min_element(values.begin(), values.end()),
                           *std::max_element(values.begin(), values.end()));
    }
    return {true, 0, 1};
  };
#define STORAGE_BOUNDS_NODE(Type, Op)                                          \
  if (const auto *node = expression.as<Type##Node>())                          \
  return binary(node->a, node->b, Op)
  STORAGE_BOUNDS_NODE(Add, "+");
  STORAGE_BOUNDS_NODE(Sub, "-");
  STORAGE_BOUNDS_NODE(Mul, "*");
  STORAGE_BOUNDS_NODE(Div, "/");
  STORAGE_BOUNDS_NODE(FloorDiv, "//");
  STORAGE_BOUNDS_NODE(Mod, "%");
  STORAGE_BOUNDS_NODE(FloorMod, "mod");
  STORAGE_BOUNDS_NODE(Min, "min");
  STORAGE_BOUNDS_NODE(Max, "max");
  STORAGE_BOUNDS_NODE(EQ, "bool");
  STORAGE_BOUNDS_NODE(NE, "bool");
  STORAGE_BOUNDS_NODE(LT, "bool");
  STORAGE_BOUNDS_NODE(LE, "bool");
  STORAGE_BOUNDS_NODE(GT, "bool");
  STORAGE_BOUNDS_NODE(GE, "bool");
  STORAGE_BOUNDS_NODE(And, "bool");
  STORAGE_BOUNDS_NODE(Or, "bool");
#undef STORAGE_BOUNDS_NODE
  LOG(FATAL) << "Unsupported expression in original storage index: "
             << expression;
  return {};
}
class AliasCanonicalizer : public StmtExprMutator {
public:
  explicit AliasCanonicalizer(PrimFunc function) : function_(function) {
    for (const Var &parameter : function->params)
      if (parameter.dtype().is_bool())
        domains_[parameter.get()] = {true, 0, 1};
    RootCollector collector(function);
    std::unordered_set<const VarNode *> declared_views;
    PostOrderVisit(function->body, [&](const ObjectRef &node) {
      if (const auto *declaration = node.as<DeclBufferNode>())
        declared_views.insert(declaration->buffer->data.get());
    });
    auto name = [&](const Root &root) -> std::string {
      return root.buffer.defined() && declared_views.count(root.data.get())
                 ? std::string(root.buffer->name)
                 : std::string(root.data->name_hint);
    };
    std::unordered_map<std::string, int> counts;
    for (const Root &root : collector.roots)
      ++counts[name(root)];
    int index = 0;
    for (const Root &root : collector.roots) {
      Var canonical = root.data;
      std::string desired = name(root);
      if (counts[desired] > 1)
        desired += "__tl_storage_" + std::to_string(index);
      if (desired != root.data->name_hint) {
        canonical = Var(desired, root.data->type_annotation);
        renamed_.Set(root.data, canonical);
        changed_ = true;
      }
      ++index;
      roots_.insert(root.data.get());
      roots_.insert(canonical.get());
      source_data_.Set(canonical, root.data);
    }
    if (auto aliases =
            function->GetAttr<Array<ObjectRef>>(kAscendStorageAliasProvenance))
      aliases_ = aliases.value();
    if (auto views =
            function->GetAttr<Array<Buffer>>(kAscendStorageViewProvenance))
      views_ = views.value();
    if (auto source =
            function->GetAttr<Map<Var, Var>>(kAscendStorageSourceData))
      for (const auto &entry : source.value())
        source_data_.Set(entry.first, entry.second);
  }
  PrimFunc Run() {
    Stmt body = VisitStmt(function_->body);
    Map<Var, Buffer> buffers;
    for (const auto &entry : function_->buffer_map)
      buffers.Set(entry.first, RewriteBuffer(entry.second));
    Map<String, ObjectRef> checked = function_->attrs.defined()
                                         ? function_->attrs->dict
                                         : Map<String, ObjectRef>();
    RemapMetadata(&checked);
    if (!changed_)
      return function_;
    PrimFunc result = function_;
    result.CopyOnWrite()->body = body;
    result.CopyOnWrite()->buffer_map = buffers;
    Map<String, ObjectRef> attrs = result->attrs.defined()
                                       ? result->attrs->dict
                                       : Map<String, ObjectRef>();
    attrs.Set(kAscendStorageAliasProvenance, aliases_);
    attrs.Set(kAscendStorageSourceData, source_data_);
    attrs.Set(kAscendStorageViewProvenance, views_);
    RemapMetadata(&attrs);
    return WithAttrs(result, attrs);
  }

private:
  PrimExpr IndexWide(const PrimExpr &value) {
    OriginalBounds(value, domains_);
    return PromoteAscendStorageIndex(value, proof_domains_);
  }
  void CheckBytes(const PrimExpr &elements, int64_t bytes, const PrimExpr &base,
                  bool require_known) {
    IntegerBounds index = OriginalBounds(elements, domains_),
                  origin = OriginalBounds(base, domains_);
    if (require_known)
      ICHECK(index.known && origin.known)
          << "Storage pointer alias requires a known bounded integer domain";
    if (!index.known || !origin.known)
      return;
    IntegerBounds scaled = CheckedBounds(__int128(index.lower) * bytes,
                                         __int128(index.upper) * bytes);
    CheckedBounds(__int128(scaled.lower) + origin.lower,
                  __int128(scaled.upper) + origin.upper);
  }
  Stmt WithDomain(const Var &variable, IntegerBounds range,
                  const std::function<Stmt()> &visit) {
    for (const auto &entry : bindings_)
      PostOrderVisit(entry.second.byte_offset, [&](const ObjectRef &node) {
        ICHECK(!node.same_as(variable))
            << "Storage pointer alias captures a shadowed integer variable";
      });
    auto old = domains_.find(variable.get());
    std::optional<IntegerBounds> saved;
    if (old != domains_.end())
      saved = old->second;
    Optional<Range> proof = proof_domains_.Get(variable);
    domains_[variable.get()] = range;
    proof_domains_.erase(variable);
    if (range.known && variable.dtype().is_int()) {
      __int128 extent = __int128(range.upper) - range.lower + 1;
      int bits = variable.dtype().bits();
      int64_t maximum = bits == 64 ? std::numeric_limits<int64_t>::max()
                                   : (int64_t(1) << (bits - 1)) - 1;
      int64_t minimum = bits == 64 ? std::numeric_limits<int64_t>::min()
                                   : -(int64_t(1) << (bits - 1));
      if (extent <= maximum && range.lower >= minimum && range.upper <= maximum)
        proof_domains_.Set(
            variable,
            Range::FromMinExtent(IntImm(variable.dtype(), range.lower),
                                 IntImm(variable.dtype(), int64_t(extent))));
    }
    Stmt result = visit();
    if (saved)
      domains_[variable.get()] = *saved;
    else
      domains_.erase(variable.get());
    if (proof.defined())
      proof_domains_.Set(variable, proof.value());
    else
      proof_domains_.erase(variable);
    return result;
  }
  AliasBinding Resolve(const Var &variable) {
    auto found = bindings_.find(variable.get());
    if (found != bindings_.end())
      return found->second;
    ICHECK(roots_.count(variable.get()))
        << "Unknown storage pointer root: " << variable;
    return {renamed_.Get(variable).value_or(variable),
            IntImm(DataType::Int(64), 0)};
  }
  AliasBinding Parse(const PrimExpr &value) {
    if (const auto *variable = value.as<VarNode>())
      return Resolve(GetRef<Var>(variable));
    const auto *pointer = value.as<CallNode>();
    ICHECK(pointer && pointer->op.same_as(builtin::tvm_access_ptr()) &&
           pointer->args.size() == 5)
        << "Unknown/opaque storage pointer alias expression";
    const auto *annotation = pointer->args[0].as<CallNode>();
    ICHECK(annotation &&
           annotation->op.same_as(Op::Get("tir.type_annotation")) &&
           annotation->args.empty())
        << "Malformed storage pointer type annotation";
    const auto *variable = pointer->args[1].as<VarNode>();
    ICHECK(variable) << "Malformed storage pointer root";
    // Removing a binding must not discard evaluation of hidden memory/state.
    const auto *mask = pointer->args[4].as<IntImmNode>();
    ICHECK(mask && mask->dtype.is_int() && mask->value >= 1 && mask->value <= 3)
        << "Unsupported storage pointer alias access mask";
    AliasBinding binding = Resolve(GetRef<Var>(variable));
    CheckBytes(pointer->args[2], ElementBytes(pointer->args[0].dtype()),
               binding.byte_offset, true);
    PrimExpr extent = IndexWide(pointer->args[3]);
    ICHECK(!analyzer_.CanProve(extent < 0))
        << "Invalid storage pointer alias extent";
    binding.byte_offset = analyzer_.Simplify(AddAscendStorageByteOffsets(
        binding.byte_offset, CheckedAscendStorageByteOffset(
                                 IndexWide(pointer->args[2]),
                                 ElementBytes(pointer->args[0].dtype()))));
    return binding;
  }
  Buffer RewriteBuffer(const Buffer &buffer) {
    AliasBinding binding = Resolve(buffer->data);
    for (const PrimExpr &extent : buffer->shape)
      OriginalBounds(extent, domains_);
    for (const PrimExpr &stride : buffer->strides)
      OriginalBounds(stride, domains_);
    CheckBytes(buffer->elem_offset, ElementBytes(buffer->dtype),
               binding.byte_offset, bindings_.count(buffer->data.get()) != 0);
    CheckedAscendStorageByteOffset(IndexWide(buffer->elem_offset),
                                   ElementBytes(buffer->dtype));
    auto cached = buffer_cache_.find(buffer.get());
    if (cached != buffer_cache_.end())
      for (const CachedView &view : cached->second)
        if (view.binding.root.same_as(binding.root) &&
            analyzer_.CanProve(view.binding.byte_offset == binding.byte_offset))
          return view.buffer;
    if (binding.root.same_as(buffer->data) && is_zero(binding.byte_offset))
      return buffer;
    int64_t bytes = ElementBytes(buffer->dtype);
    ICHECK(analyzer_.CanProve(
        floormod(binding.byte_offset, IntImm(DataType::Int(64), bytes)) == 0))
        << "Storage alias byte offset is not divisible by its view dtype";
    Buffer result = buffer;
    auto *node = result.CopyOnWrite();
    node->data = binding.root;
    node->elem_offset = analyzer_.Simplify(AddAscendStorageByteOffsets(
        IndexWide(buffer->elem_offset),
        analyzer_.Simplify(
            floordiv(binding.byte_offset, IntImm(DataType::Int(64), bytes)))));
    changed_ = true;
    buffer_cache_[buffer.get()].push_back({binding, result});
    return result;
  }
  PrimExpr VisitExpr_(const VarNode *variable) final {
    ICHECK(!bindings_.count(variable))
        << "Storage pointer alias escapes an audited access_ptr/view";
    Var value = GetRef<Var>(variable);
    return renamed_.Get(value).value_or(value);
  }
  PrimExpr VisitExpr_(const CallNode *call) final {
    if (!call->op.same_as(builtin::tvm_access_ptr()))
      return StmtExprMutator::VisitExpr_(call);
    ICHECK_EQ(call->args.size(), 5U) << "Malformed storage access_ptr";
    const auto *annotation = call->args[0].as<CallNode>();
    ICHECK(annotation &&
           annotation->op.same_as(Op::Get("tir.type_annotation")) &&
           annotation->args.empty())
        << "Malformed storage pointer type annotation";
    const auto *data = call->args[1].as<VarNode>();
    ICHECK(data) << "Malformed storage access_ptr root";
    AliasBinding binding = Resolve(GetRef<Var>(data));
    OriginalBounds(call->args[3], domains_);
    bool rewritten = !binding.root.same_as(GetRef<Var>(data)) ||
                     !is_zero(binding.byte_offset);
    CheckBytes(call->args[2], ElementBytes(call->args[0].dtype()),
               binding.byte_offset, rewritten);
    PrimExpr own_bytes = CheckedAscendStorageByteOffset(
        IndexWide(call->args[2]), ElementBytes(call->args[0].dtype()));
    if (binding.root.same_as(GetRef<Var>(data)) && is_zero(binding.byte_offset))
      return StmtExprMutator::VisitExpr_(call);
    int64_t bytes = ElementBytes(call->args[0].dtype());
    PrimExpr byte_offset = analyzer_.Simplify(
        AddAscendStorageByteOffsets(binding.byte_offset, own_bytes));
    ICHECK(analyzer_.CanProve(
        floormod(byte_offset, IntImm(DataType::Int(64), bytes)) == 0))
        << "Storage alias byte offset is not divisible by the consumer dtype";
    Array<PrimExpr> args = call->args;
    args.Set(1, binding.root);
    args.Set(2, analyzer_.Simplify(
                    floordiv(byte_offset, IntImm(DataType::Int(64), bytes))));
    args.Set(3, VisitExpr(call->args[3]));
    args.Set(4, VisitExpr(call->args[4]));
    changed_ = true;
    return Call(call->dtype, call->op, args, call->span);
  }
  PrimExpr VisitExpr_(const BufferLoadNode *load) final {
    for (const PrimExpr &index : load->indices)
      OriginalBounds(index, domains_);
    Array<PrimExpr> indices =
        load->indices.Map([&](PrimExpr value) { return VisitExpr(value); });
    Buffer buffer = RewriteBuffer(load->buffer);
    if (buffer.same_as(load->buffer) && indices.same_as(load->indices))
      return GetRef<PrimExpr>(load);
    return BufferLoad(buffer, indices, load->span);
  }
  Stmt VisitStmt_(const BufferStoreNode *store) final {
    for (const PrimExpr &index : store->indices)
      OriginalBounds(index, domains_);
    return BufferStore(
        RewriteBuffer(store->buffer), VisitExpr(store->value),
        store->indices.Map([&](PrimExpr value) { return VisitExpr(value); }),
        store->span);
  }
  Stmt VisitStmt_(const LetStmtNode *statement) final {
    if (!statement->var.dtype().is_handle()) {
      if (!statement->var.dtype().is_int() && !statement->var.dtype().is_uint())
        return StmtExprMutator::VisitStmt_(statement);
      IntegerBounds range = OriginalBounds(statement->value, domains_);
      return WithDomain(statement->var, range,
                        [&] { return StmtExprMutator::VisitStmt_(statement); });
    }
    ICHECK(!roots_.count(statement->var.get()))
        << "Storage pointer binding cannot shadow a physical storage Var";
    AliasBinding binding = Parse(statement->value);
    auto previous = bindings_.find(statement->var.get());
    bool shadowed = previous != bindings_.end();
    AliasBinding saved;
    if (shadowed)
      saved = previous->second;
    bindings_[statement->var.get()] = binding;
    aliases_.push_back(Record{{"data", statement->var},
                              {"root", binding.root},
                              {"byte_offset", binding.byte_offset},
                              {"kind", String("pointer_binding")},
                              {"binding_site_id", Integer(next_binding_++)},
                              {"parent_loop_vars", loop_variables_}});
    Stmt body = VisitStmt(statement->body);
    if (shadowed)
      bindings_[statement->var.get()] = saved;
    else
      bindings_.erase(statement->var.get());
    changed_ = true;
    return body;
  }
  Stmt VisitStmt_(const ForNode *loop) final {
    loop_variables_.push_back(loop->loop_var);
    IntegerBounds range;
    const auto *minimum = loop->min.as<IntImmNode>(),
               *extent = loop->extent.as<IntImmNode>();
    if (minimum && extent && extent->value > 0)
      range = CheckedBounds(minimum->value,
                            __int128(minimum->value) + extent->value - 1);
    Stmt result = WithDomain(loop->loop_var, range,
                             [&] { return StmtExprMutator::VisitStmt_(loop); });
    loop_variables_.pop_back();
    return result;
  }
  Stmt VisitStmt_(const AttrStmtNode *attribute) final {
    if (attribute->attr_key != "thread_extent")
      return StmtExprMutator::VisitStmt_(attribute);
    const auto *iterator = attribute->node.as<IterVarNode>();
    ICHECK(iterator) << "Malformed storage launch iterator";
    const auto *extent = attribute->value.as<IntImmNode>();
    IntegerBounds range;
    if (extent && extent->value > 0)
      range = {true, 0, extent->value - 1};
    return WithDomain(iterator->var, range,
                      [&] { return StmtExprMutator::VisitStmt_(attribute); });
  }
  Stmt VisitStmt_(const DeclBufferNode *declaration) final {
    Buffer view = RewriteBuffer(declaration->buffer);
    views_.push_back(view);
    changed_ = true;
    return VisitStmt(declaration->body);
  }
  Stmt VisitStmt_(const BlockNode *block) final {
    std::vector<std::pair<Var, std::optional<AliasBinding>>> saved;
    for (const MatchBufferRegion &match : block->match_buffers) {
      Buffer source = match->source->buffer, target = match->buffer;
      ICHECK(source->shape.size() == 1 && target->shape.size() == 1 &&
             source->strides.empty() && target->strides.empty() &&
             is_zero(target->elem_offset) && source->dtype == target->dtype &&
             match->source->region.size() == 1 &&
             analyzer_.CanProve(target->shape[0] ==
                                match->source->region[0]->extent))
          << "Storage match_buffer requires an exact dense one-dimensional "
             "binding";
      AliasBinding binding = Resolve(source->data);
      IntegerBounds source_offset =
                        OriginalBounds(source->elem_offset, domains_),
                    region_min =
                        OriginalBounds(match->source->region[0]->min, domains_);
      ICHECK(source_offset.known && region_min.known)
          << "Storage match_buffer requires a bounded source region";
      IntegerBounds offset =
          CheckedBounds(__int128(source_offset.lower) + region_min.lower,
                        __int128(source_offset.upper) + region_min.upper);
      CheckedBounds(__int128(offset.lower) * ElementBytes(source->dtype),
                    __int128(offset.upper) * ElementBytes(source->dtype));
      binding.byte_offset = analyzer_.Simplify(AddAscendStorageByteOffsets(
          binding.byte_offset,
          CheckedAscendStorageByteOffset(
              AddAscendStorageByteOffsets(
                  IndexWide(source->elem_offset),
                  IndexWide(match->source->region[0]->min)),
              ElementBytes(source->dtype))));
      auto old = bindings_.find(target->data.get());
      saved.push_back(
          {target->data, old == bindings_.end()
                             ? std::optional<AliasBinding>()
                             : std::optional<AliasBinding>(old->second)});
      ICHECK(!roots_.count(target->data.get()))
          << "Storage match_buffer cannot rebind a physical storage Var";
      bindings_[target->data.get()] = binding;
      views_.push_back(RewriteBuffer(target));
      aliases_.push_back(Record{{"data", target->data},
                                {"root", binding.root},
                                {"byte_offset", binding.byte_offset},
                                {"kind", String("match_buffer")},
                                {"binding_site_id", Integer(next_binding_++)},
                                {"parent_loop_vars", loop_variables_}});
      changed_ = true;
    }
    Block result = Downcast<Block>(StmtExprMutator::VisitStmt_(block));
    auto region = [&](BufferRegion value) {
      Buffer buffer = RewriteBuffer(value->buffer);
      return buffer.same_as(value->buffer)
                 ? value
                 : BufferRegion(buffer, value->region);
    };
    // An enclosing pointer binding can change a region's storage even when
    // this block has no match buffers and the root name is unchanged.
    Array<BufferRegion> reads = result->reads.Map(region);
    Array<BufferRegion> writes = result->writes.Map(region);
    if (!reads.same_as(result->reads) || !writes.same_as(result->writes) ||
        !saved.empty()) {
      auto *node = result.CopyOnWrite();
      node->reads = reads;
      node->writes = writes;
      node->match_buffers = {};
    }
    Array<Buffer> allocations;
    for (const Buffer &buffer : result->alloc_buffers)
      allocations.push_back(RewriteBuffer(buffer));
    if (!renamed_.empty())
      result.CopyOnWrite()->alloc_buffers = allocations;
    Map<String, ObjectRef> annotations = result->annotations;
    RemapMetadata(&annotations);
    if (auto layouts = annotations.Get("layout_map")) {
      Map<ObjectRef, ObjectRef> mapped;
      for (const auto &entry :
           Downcast<Map<ObjectRef, ObjectRef>>(layouts.value())) {
        if (const auto *variable = entry.first.as<VarNode>()) {
          AliasBinding binding = Resolve(GetRef<Var>(variable));
          ICHECK(is_zero(binding.byte_offset))
              << "Storage layout metadata cannot target an offset pointer "
                 "alias";
          mapped.Set(binding.root, entry.second);
        } else {
          ICHECK(entry.first.as<BufferNode>())
              << "Malformed storage layout metadata key";
          mapped.Set(RewriteBuffer(Downcast<Buffer>(entry.first)),
                     entry.second);
        }
      }
      annotations.Set("layout_map", mapped);
    }
    if (!annotations.same_as(result->annotations))
      result.CopyOnWrite()->annotations = annotations;
    for (auto entry = saved.rbegin(); entry != saved.rend(); ++entry) {
      if (entry->second)
        bindings_[entry->first.get()] = *entry->second;
      else
        bindings_.erase(entry->first.get());
    }
    return result;
  }
  Stmt VisitStmt_(const AllocateNode *allocation) final {
    for (const PrimExpr &extent : allocation->extents)
      OriginalBounds(extent, domains_);
    Allocate result =
        Downcast<Allocate>(StmtExprMutator::VisitStmt_(allocation));
    if (renamed_.count(allocation->buffer_var))
      result.CopyOnWrite()->buffer_var = renamed_[allocation->buffer_var];
    return result;
  }
  void RemapMetadata(Map<String, ObjectRef> *attrs) {
    if (auto original = attrs->Get(kAscendStorageOriginalParameters)) {
      Array<Var> values;
      for (const Var &data : Downcast<Array<Var>>(original.value()))
        values.push_back(renamed_.Get(data).value_or(data));
      attrs->Set(kAscendStorageOriginalParameters, values);
    }
    for (const char *key : {"address_map", "size_map"}) {
      auto value = attrs->Get(key);
      if (!value.defined())
        continue;
      Map<Var, PrimExpr> map = Downcast<Map<Var, PrimExpr>>(value.value()),
                         result;
      for (const auto &entry : map) {
        AliasBinding root;
        if (roots_.count(entry.first.get())) {
          root = Resolve(entry.first);
        } else {
          bool found = false;
          for (const ObjectRef &item : aliases_) {
            Record record = Downcast<Record>(item);
            if (!Downcast<Var>(record.at("data")).same_as(entry.first))
              continue;
            AliasBinding candidate{
                Downcast<Var>(record.at("root")),
                Downcast<PrimExpr>(record.at("byte_offset"))};
            ICHECK(!found || (root.root.same_as(candidate.root) &&
                              analyzer_.CanProve(root.byte_offset ==
                                                 candidate.byte_offset)))
                << "Ambiguous storage alias metadata binding";
            root = candidate;
            found = true;
          }
          ICHECK(found) << "Unknown or stale storage metadata owner";
        }
        ICHECK(is_zero(root.byte_offset))
            << "Address metadata cannot target an offset pointer alias";
        result.Set(root.root, entry.second);
      }
      attrs->Set(key, result);
    }
    for (const char *key :
         {"initial_buffer_shapes", "buffer_shapess", "logic_buffer_shapes"}) {
      auto value = attrs->Get(key);
      if (!value.defined())
        continue;
      Map<Var, Array<PrimExpr>> map = Downcast<Map<Var, Array<PrimExpr>>>(
                                    value.value()),
                                result;
      for (const auto &entry : map) {
        ICHECK(roots_.count(entry.first.get()))
            << "Storage shape metadata names unknown or stale storage";
        result.Set(renamed_.Get(entry.first).value_or(entry.first),
                   entry.second);
      }
      attrs->Set(key, result);
    }
  }
  PrimFunc function_;
  bool changed_{false};
  int next_binding_{0};
  std::unordered_set<const VarNode *> roots_;
  std::unordered_map<const VarNode *, AliasBinding> bindings_;
  struct CachedView {
    AliasBinding binding;
    Buffer buffer;
  };
  std::unordered_map<const BufferNode *, std::vector<CachedView>> buffer_cache_;
  Map<Var, Var> renamed_, source_data_;
  Array<ObjectRef> aliases_;
  Array<Buffer> views_;
  Array<Var> loop_variables_;
  BoundsMap domains_;
  Map<Var, Range> proof_domains_;
  arith::Analyzer analyzer_;
};

class AllocationSites : public StmtVisitor {
public:
  struct Info {
    int id;
    Array<Integer> loops;
    ObjectRef allocation;
  };
  std::unordered_map<const VarNode *, Info> sites;

private:
  void Register(const Var &data, ObjectRef allocation) {
    ICHECK(!sites.count(data.get()))
        << "Storage has multiple physical allocation sites";
    sites.emplace(data.get(), Info{next_site_++, loops_, allocation});
  }
  void VisitStmt_(const ForNode *loop) final {
    int id = next_loop_++;
    loops_.push_back(Integer(id));
    StmtVisitor::VisitStmt_(loop);
    loops_.pop_back();
  }
  void VisitStmt_(const BlockNode *block) final {
    for (const Buffer &buffer : block->alloc_buffers)
      Register(buffer->data, buffer);
    StmtVisitor::VisitStmt_(block);
  }
  void VisitStmt_(const AllocateNode *allocation) final {
    Register(allocation->buffer_var, GetRef<Stmt>(allocation));
    StmtVisitor::VisitStmt_(allocation);
  }
  int next_site_{0}, next_loop_{0};
  Array<Integer> loops_;
};
} // namespace

PrimFunc CanonicalizeAscendStorageAliases(PrimFunc function) {
  return AliasCanonicalizer(function).Run();
}

PrimFunc RemapAscendStorageMetadata(PrimFunc function,
                                    Map<Var, Var> replacements) {
  if (replacements.empty() || !function->attrs.defined())
    return function;
  Map<String, ObjectRef> attrs = function->attrs->dict;
  auto mapped = [&](Var data) { return replacements.Get(data).value_or(data); };
  if (auto original =
          function->GetAttr<Array<Var>>(kAscendStorageOriginalParameters)) {
    Array<Var> values;
    for (const Var &data : original.value())
      values.push_back(mapped(data));
    attrs.Set(kAscendStorageOriginalParameters, values);
  }
  if (auto source =
          function->GetAttr<Map<Var, Var>>(kAscendStorageSourceData)) {
    Map<Var, Var> result;
    for (const auto &entry : source.value())
      result.Set(mapped(entry.first), entry.second);
    attrs.Set(kAscendStorageSourceData, result);
  }
  if (auto aliases =
          function->GetAttr<Array<ObjectRef>>(kAscendStorageAliasProvenance)) {
    Array<ObjectRef> result;
    for (const ObjectRef &item : aliases.value()) {
      Record alias = Downcast<Record>(item);
      alias.Set("root", mapped(Downcast<Var>(alias.at("root"))));
      result.push_back(alias);
    }
    attrs.Set(kAscendStorageAliasProvenance, result);
  }
  if (auto views =
          function->GetAttr<Array<Buffer>>(kAscendStorageViewProvenance)) {
    Array<Buffer> result;
    for (Buffer view : views.value()) {
      if (replacements.count(view->data))
        view.CopyOnWrite()->data = mapped(view->data);
      result.push_back(view);
    }
    attrs.Set(kAscendStorageViewProvenance, result);
  }
  for (const char *key : {"address_map", "size_map"}) {
    if (auto values = function->GetAttr<Map<Var, PrimExpr>>(key)) {
      Map<Var, PrimExpr> result;
      for (const auto &entry : values.value())
        result.Set(mapped(entry.first), entry.second);
      attrs.Set(key, result);
    }
  }
  for (const char *key :
       {"initial_buffer_shapes", "buffer_shapess", "logic_buffer_shapes"}) {
    if (auto values = function->GetAttr<Map<Var, Array<PrimExpr>>>(key)) {
      Map<Var, Array<PrimExpr>> result;
      for (const auto &entry : values.value())
        result.Set(mapped(entry.first), entry.second);
      attrs.Set(key, result);
    }
  }
  return WithAttrs(function, attrs);
}

Map<String, ObjectRef> BuildAscendStorageCatalog(PrimFunc function,
                                                 Record graph) {
  AllocationSites sites;
  sites(function->body);
  auto sources = function->GetAttr<Map<Var, Var>>(kAscendStorageSourceData)
                     .value_or(Map<Var, Var>());
  auto original =
      function->GetAttr<Array<Var>>(kAscendStorageOriginalParameters);
  std::unordered_set<const VarNode *> external;
  if (original.defined())
    for (const Var &data : original.value())
      external.insert(data.get());
  else
    for (const auto &entry : function->buffer_map)
      external.insert(entry.second->data.get());
  std::unordered_map<const VarNode *, int> ids;
  Array<ObjectRef> storages, aliases, views;
  for (const ObjectRef &item : Downcast<Array<ObjectRef>>(graph["storages"])) {
    Record record = Downcast<Record>(item);
    Var data = Downcast<Var>(record["data"]);
    int id = Downcast<Integer>(record["id"])->value;
    ids[data.get()] = id;
    Record result = record;
    result.Set("source_data", sources.Get(data).value_or(data));
    auto site = sites.sites.find(data.get());
    bool allocated = site != sites.sites.end();
    result.Set("allocation_site_id", Integer(allocated ? site->second.id : -1));
    result.Set("parent_loop_ids",
               allocated ? site->second.loops : Array<Integer>());
    result.Set("origin",
               String(allocated                    ? "allocation"
                      : external.count(data.get()) ? "external_parameter"
                                                   : "internal_workspace"));
    if (allocated)
      result.Set("allocation", site->second.allocation);
    storages.push_back(result);
  }
  if (auto provenance =
          function->GetAttr<Array<ObjectRef>>(kAscendStorageAliasProvenance)) {
    for (const ObjectRef &item : provenance.value()) {
      Record alias = Downcast<Record>(item);
      Var root = Downcast<Var>(alias["root"]);
      if (!ids.count(root.get()))
        continue;
      Record result = alias;
      result.Set("storage_id", Integer(ids.at(root.get())));
      Array<Integer> parents;
      for (const Var &variable :
           Downcast<Array<Var>>(alias["parent_loop_vars"]))
        for (const ObjectRef &loop_item :
             Downcast<Array<ObjectRef>>(graph["loops"])) {
          Record loop = Downcast<Record>(loop_item);
          if (Downcast<Var>(loop["var"]).same_as(variable))
            parents.push_back(Downcast<Integer>(loop["id"]));
        }
      result.Set("parent_loop_ids", parents);
      aliases.push_back(result);
    }
  }
  std::unordered_set<const BufferNode *> seen_views;
  auto add_view = [&](const Buffer &buffer) {
    if (!ids.count(buffer->data.get()) ||
        !seen_views.insert(buffer.get()).second)
      return;
    ValidatePhysicalExtent(buffer->dtype, buffer->shape);
    PrimExpr begin = CheckedAscendStorageByteOffset(
        Wide(buffer->elem_offset), ElementBytes(buffer->dtype));
    PrimExpr elements = IntImm(DataType::Int(64), 1);
    if (buffer->strides.empty()) {
      for (const PrimExpr &extent : buffer->shape)
        elements = elements * Wide(extent);
    } else {
      elements = IntImm(DataType::Int(64), 1);
      for (size_t i = 0; i < buffer->shape.size(); ++i) {
        PrimExpr count = Wide(buffer->shape[i]) - IntImm(DataType::Int(64), 1);
        PrimExpr stride = Wide(buffer->strides[i]);
        PrimExpr term;
        if (const auto *constant = stride.as<IntImmNode>()) {
          ICHECK(constant->value >= 0)
              << "Unsupported negative storage view stride";
          term = constant->value == 0
                     ? IntImm(DataType::Int(64), 0)
                     : CheckedAscendStorageByteOffset(count, constant->value);
        } else
          term = count * stride;
        elements = AddAscendStorageByteOffsets(elements, term);
      }
    }
    arith::Analyzer analyzer;
    views.push_back(
        Record{{"storage_id", Integer(ids.at(buffer->data.get()))},
               {"buffer", buffer},
               {"data", buffer->data},
               {"dtype", String(DLDataType2String(buffer->dtype))},
               {"shape", buffer->shape},
               {"strides", buffer->strides},
               {"elem_offset", buffer->elem_offset},
               {"byte_begin", analyzer.Simplify(begin)},
               {"byte_end", analyzer.Simplify(AddAscendStorageByteOffsets(
                                begin, CheckedAscendStorageByteOffset(
                                           analyzer.Simplify(elements),
                                           ElementBytes(buffer->dtype))))}});
  };
  for (const auto &entry : function->buffer_map)
    add_view(entry.second);
  if (auto declared =
          function->GetAttr<Array<Buffer>>(kAscendStorageViewProvenance))
    for (const Buffer &buffer : declared.value())
      add_view(buffer);
  PostOrderVisit(function->body, [&](const ObjectRef &node) {
    if (const auto *block = node.as<BlockNode>()) {
      for (const Buffer &buffer : block->alloc_buffers)
        add_view(buffer);
      for (const BufferRegion &region : block->reads)
        add_view(region->buffer);
      for (const BufferRegion &region : block->writes)
        add_view(region->buffer);
    } else if (const auto *load = node.as<BufferLoadNode>())
      add_view(load->buffer);
    else if (const auto *store = node.as<BufferStoreNode>())
      add_view(store->buffer);
  });
  return {{"schema_version", Integer(1)},
          {"storages", storages},
          {"aliases", aliases},
          {"views", views}};
}

Map<String, ObjectRef> AnalyzeAscendStorageOwners(PrimFunc function,
                                                  String platform) {
  ICHECK_EQ(platform, "A3") << "Storage owner analysis supports A3 only";
  PrimFunc canonical = CanonicalizeAscendStorageAliases(function);
  Record graph = AnalyzeAscendScheduleEffects(canonical, platform);
  return BuildAscendStorageCatalog(canonical, graph);
}

tir::transform::Pass AscendCanonicalizeStorageAliases() {
  auto pass = [](PrimFunc function, IRModule,
                 tvm::transform::PassContext context) {
    if (context->GetConfig<String>(kAscendPipelinePlanner, String("legacy"))
            .value() != "constraint")
      return function;
    return CanonicalizeAscendStorageAliases(function);
  };
  return tir::transform::CreatePrimFuncPass(
      pass, 0, "tl.AscendCanonicalizeStorageAliases", {});
}
TVM_REGISTER_GLOBAL("tl.transform.AscendCanonicalizeStorageAliases")
    .set_body_typed(AscendCanonicalizeStorageAliases);
TVM_REGISTER_GLOBAL("tl.transform.AnalyzeAscendStorageOwners")
    .set_body_typed(AnalyzeAscendStorageOwners);
} // namespace tl
} // namespace tvm
