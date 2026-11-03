// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/data_type_rewriter.h>
#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm::tl {

using namespace tir;

namespace {

using Variables = std::unordered_set<const VarNode *>;

struct SignedBounds {
  int64_t minimum;
  int64_t maximum;
};

std::optional<SignedBounds> CheckedBounds(DataType dtype, __int128 minimum,
                                          __int128 maximum) {
  if (!dtype.is_scalar() || !dtype.is_int() || dtype.bits() > 64)
    return std::nullopt;
  const int64_t low = dtype.bits() == 64 ? std::numeric_limits<int64_t>::min()
                                         : -(int64_t(1) << (dtype.bits() - 1));
  const int64_t high = dtype.bits() == 64
                           ? std::numeric_limits<int64_t>::max()
                           : (int64_t(1) << (dtype.bits() - 1)) - 1;
  if (minimum < low || maximum > high || minimum > maximum)
    return std::nullopt;
  return SignedBounds{static_cast<int64_t>(minimum),
                      static_cast<int64_t>(maximum)};
}

// Never ask the legacy folder for a compound expression's bound: cancellation
// can conceal an original-width overflow. Unknown bounds preserve the cast.
std::optional<SignedBounds> OriginalBounds(const PrimExpr &value,
                                           arith::Analyzer *analyzer) {
  if (!value.dtype().is_scalar() || !value.dtype().is_int() ||
      value.dtype().bits() > 64)
    return std::nullopt;
  if (const auto *constant = value.as<IntImmNode>())
    return CheckedBounds(value.dtype(), constant->value, constant->value);
  if (value.as<VarNode>()) {
    const auto bound = analyzer->const_int_bound(value);
    if (bound->min_value == arith::ConstIntBound::kNegInf ||
        bound->max_value == arith::ConstIntBound::kPosInf)
      return std::nullopt;
    return CheckedBounds(value.dtype(), bound->min_value, bound->max_value);
  }
  if (const auto *conversion = value.as<CastNode>()) {
    auto source = OriginalBounds(conversion->value, analyzer);
    return source
               ? CheckedBounds(value.dtype(), source->minimum, source->maximum)
               : std::nullopt;
  }
  PrimExpr left, right;
  if (const auto *operation = value.as<AddNode>()) {
    left = operation->a;
    right = operation->b;
  } else if (const auto *operation = value.as<SubNode>()) {
    left = operation->a;
    right = operation->b;
  } else if (const auto *operation = value.as<MulNode>()) {
    left = operation->a;
    right = operation->b;
  } else if (const auto *operation = value.as<FloorDivNode>()) {
    left = operation->a;
    right = operation->b;
  } else if (const auto *operation = value.as<MinNode>()) {
    left = operation->a;
    right = operation->b;
  } else if (const auto *operation = value.as<MaxNode>()) {
    left = operation->a;
    right = operation->b;
  } else {
    return std::nullopt;
  }
  const auto a = OriginalBounds(left, analyzer);
  const auto b = OriginalBounds(right, analyzer);
  if (!a || !b)
    return std::nullopt;
  __int128 minimum, maximum;
  if (value.as<AddNode>()) {
    minimum = __int128(a->minimum) + b->minimum;
    maximum = __int128(a->maximum) + b->maximum;
  } else if (value.as<SubNode>()) {
    minimum = __int128(a->minimum) - b->maximum;
    maximum = __int128(a->maximum) - b->minimum;
  } else if (value.as<MulNode>()) {
    const __int128 products[] = {
        __int128(a->minimum) * b->minimum, __int128(a->minimum) * b->maximum,
        __int128(a->maximum) * b->minimum, __int128(a->maximum) * b->maximum};
    minimum = *std::min_element(std::begin(products), std::end(products));
    maximum = *std::max_element(std::begin(products), std::end(products));
  } else if (value.as<FloorDivNode>()) {
    if (b->minimum <= 0 || b->minimum != b->maximum)
      return std::nullopt;
    auto divide = [&](int64_t numerator) -> __int128 {
      const __int128 quotient = __int128(numerator) / b->minimum;
      return quotient - (numerator < 0 && numerator % b->minimum != 0);
    };
    minimum = divide(a->minimum);
    maximum = divide(a->maximum);
  } else if (value.as<MinNode>()) {
    minimum = std::min(a->minimum, b->minimum);
    maximum = std::min(a->maximum, b->maximum);
  } else {
    minimum = std::max(a->minimum, b->minimum);
    maximum = std::max(a->maximum, b->maximum);
  }
  return CheckedBounds(value.dtype(), minimum, maximum);
}

bool ValuePreservingNarrow(const CastNode *conversion) {
  if (!conversion->dtype.is_scalar() || !conversion->dtype.is_int() ||
      !conversion->value.dtype().is_scalar() ||
      !conversion->value.dtype().is_int() ||
      conversion->dtype.bits() >= conversion->value.dtype().bits() ||
      SideEffect(conversion->value) > CallEffectKind::kPure)
    return false;
  arith::Analyzer analyzer;
  const auto bounds = OriginalBounds(conversion->value, &analyzer);
  return bounds &&
         CheckedBounds(conversion->dtype, bounds->minimum, bounds->maximum)
             .has_value();
}

class IndexDependencies : public ExprVisitor {
public:
  Variables Run(const PrimExpr &expression) {
    VisitExpr(expression);
    return std::move(variables_);
  }

private:
  void VisitExpr(const PrimExpr &expression) final {
    if (expression.dtype().is_uint() && !expression.dtype().is_bool())
      return;
    ExprVisitor::VisitExpr(expression);
  }
  void VisitExpr_(const VarNode *op) final {
    if (op->dtype.is_scalar() && op->dtype.is_int())
      variables_.insert(op);
  }
  void VisitExpr_(const BufferLoadNode *) final {}
  void VisitExpr_(const CastNode *op) final {
    if (op->dtype.is_int() && op->value.dtype().is_int() &&
        (op->dtype.bits() >= op->value.dtype().bits() ||
         ValuePreservingNarrow(op)))
      VisitExpr(op->value);
  }
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::if_then_else()))
      ExprVisitor::VisitExpr_(op);
  }
  Variables variables_;
};

class ShapeIndexDemand : public StmtExprVisitor {
public:
  Variables Run(const Stmt &body, const Variables &shapes) {
    VisitStmt(body);
    bool changed = true;
    while (changed) {
      changed = false;
      for (const auto &[variable, dependencies] : definitions_)
        if (demand_.count(variable))
          for (const auto *dependency : dependencies)
            changed |= demand_.insert(dependency).second;
    }
    Variables dependent = shapes;
    changed = true;
    while (changed) {
      changed = false;
      for (const auto &[variable, dependencies] : definitions_)
        for (const auto *dependency : dependencies)
          if (dependent.count(dependency)) {
            changed |= dependent.insert(variable).second;
            break;
          }
    }
    Variables selected;
    for (const auto *variable : demand_)
      if (dependent.count(variable) && variable->dtype.is_scalar() &&
          variable->dtype.is_int() && variable->dtype.bits() == 32)
        selected.insert(variable);
    return selected;
  }

  const std::vector<Var> &Binders() const { return binders_; }

private:
  void Demand(const PrimExpr &expression) {
    for (const auto *variable : IndexDependencies().Run(expression))
      demand_.insert(variable);
  }
  void DemandBuffer(const Buffer &buffer) {
    for (const auto &extent : buffer->shape)
      Demand(extent);
    for (const auto &stride : buffer->strides)
      Demand(stride);
    Demand(buffer->elem_offset);
  }
  void DemandMetadata(const ObjectRef &value) {
    if (value.as<PrimExprNode>()) {
      Demand(Downcast<PrimExpr>(value));
    } else if (const auto *range = value.as<RangeNode>()) {
      Demand(range->min);
      Demand(range->extent);
    } else if (const auto *buffer = value.as<BufferNode>()) {
      DemandBuffer(GetRef<Buffer>(buffer));
    } else if (const auto *region = value.as<BufferRegionNode>()) {
      DemandBuffer(region->buffer);
      for (const Range &range : region->region)
        DemandMetadata(range);
    } else if (value.as<ArrayNode>()) {
      for (const ObjectRef &element : Downcast<Array<ObjectRef>>(value))
        DemandMetadata(element);
    } else if (value.as<MapNode>()) {
      for (const auto &[key, element] :
           Downcast<Map<ObjectRef, ObjectRef>>(value)) {
        DemandMetadata(key);
        DemandMetadata(element);
      }
    }
  }
  void Define(const Var &variable, const PrimExpr &value) {
    if (SideEffect(value) <= CallEffectKind::kPure) {
      Variables dependencies = IndexDependencies().Run(value);
      auto [position, inserted] = definitions_.try_emplace(variable.get());
      if (inserted)
        binders_.push_back(variable);
      auto &definition = position->second;
      definition.insert(dependencies.begin(), dependencies.end());
    }
  }
  void VisitStmt_(const ForNode *op) final {
    for (const auto &[key, value] : op->annotations)
      DemandMetadata(value);
    Define(op->loop_var, op->min);
    Define(op->loop_var, op->extent);
    Demand(op->min);
    Demand(op->extent);
    demand_.insert(op->loop_var.get());
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const BlockRealizeNode *op) final {
    for (size_t index = 0; index < op->iter_values.size(); ++index) {
      Define(op->block->iter_vars[index]->var, op->iter_values[index]);
      Demand(op->iter_values[index]);
    }
    Demand(op->predicate);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const BlockNode *op) final {
    for (const auto &[key, value] : op->annotations)
      DemandMetadata(value);
    for (const Buffer &buffer : op->alloc_buffers)
      DemandBuffer(buffer);
    for (const MatchBufferRegion &match : op->match_buffers) {
      DemandBuffer(match->buffer);
      DemandBuffer(match->source->buffer);
    }
    for (const IterVar &iteration : op->iter_vars) {
      if (iteration->dom.defined()) {
        Define(iteration->var, iteration->dom->min);
        Define(iteration->var, iteration->dom->extent);
        Demand(iteration->dom->min);
        Demand(iteration->dom->extent);
        demand_.insert(iteration->var.get());
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const DeclBufferNode *op) final {
    DemandBuffer(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const BufferRealizeNode *op) final {
    DemandBuffer(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const PrefetchNode *op) final {
    DemandBuffer(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const LetStmtNode *op) final {
    Define(op->var, op->value);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const LetNode *op) final {
    Define(op->var, op->value);
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const AttrStmtNode *op) final {
    DemandMetadata(op->node);
    if (op->attr_key == "thread_extent" || op->attr_key == "virtual_thread") {
      if (const auto *iteration = op->node.as<IterVarNode>()) {
        Define(iteration->var, op->value);
        demand_.insert(iteration->var.get());
      }
      Demand(op->value);
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const IfThenElseNode *op) final {
    Demand(op->condition);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    DemandBuffer(op->buffer);
    for (const auto &index : op->indices)
      Demand(index);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const BufferLoadNode *op) final {
    DemandBuffer(op->buffer);
    for (const auto &index : op->indices)
      Demand(index);
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(Op::Get("tl.region"))) {
      for (size_t index = 2; index < op->args.size(); ++index)
        Demand(op->args[index]);
    } else if (op->op.same_as(builtin::tvm_access_ptr()) &&
               op->args.size() == 5U) {
      Demand(op->args[2]);
      Demand(op->args[3]);
    }
    StmtExprVisitor::VisitExpr_(op);
  }
  Variables demand_;
  std::unordered_map<const VarNode *, Variables> definitions_;
  std::vector<Var> binders_;
};

class ShapeIndexLegalizer : public DataTypeLegalizer {
public:
  static PrimFunc Substitute(PrimFunc function, const Target &target,
                             tvm::transform::PassContext context) {
    const auto backend =
        context->GetConfig<String>("tl.ascend_sync_backend", String("legacy"));
    const auto platform = function->GetAttr<String>("npu_platform");
    const String model = target->GetAttr<String>("model").value_or(String());
    if (backend.value() != "upstream" ||
        (model != "ascendc" && model != "auto") || !platform.defined() ||
        (platform.value() != "A2" && platform.value() != "A3"))
      return function;
    Variables parameters;
    for (const Var &parameter : function->params)
      parameters.insert(parameter.get());
    Variables shapes;
    for (const auto &[parameter, buffer] : function->buffer_map)
      for (const PrimExpr &dimension : buffer->shape)
        if (const auto *variable = dimension.as<VarNode>())
          if (variable->dtype == DataType::Int(32) &&
              !parameters.count(variable))
            shapes.insert(variable);
    if (shapes.empty())
      return function;
    ShapeIndexDemand demand;
    Variables selected = demand.Run(function->body, shapes);
    for (const auto *parameter : parameters)
      selected.erase(parameter);
    ShapeIndexLegalizer legalizer(std::move(shapes), std::move(selected));
    for (const Var &binder : demand.Binders())
      if (!parameters.count(binder.get()))
        legalizer.InternalVariable(binder);
    for (const auto &[parameter, buffer] : function->buffer_map)
      legalizer.parameter_buffers_.insert(buffer.get());
    Stmt body = legalizer.VisitStmt(function->body);
    if (body.same_as(function->body))
      return function;
    function.CopyOnWrite()->body = std::move(body);
    return function;
  }

private:
  ShapeIndexLegalizer(Variables shapes, Variables selected)
      : shapes_(std::move(shapes)), selected_(std::move(selected)) {}

  PrimExpr WithMode(const PrimExpr &expression, bool index) {
    const bool saved = index_;
    index_ = index;
    PrimExpr result = VisitExpr(expression);
    index_ = saved;
    return result;
  }
  Buffer IndexBuffer(const Buffer &buffer) {
    if (parameter_buffers_.count(buffer.get()))
      return buffer;
    if (auto found = buffers_.find(buffer.get()); found != buffers_.end())
      return found->second;
    Array<PrimExpr> shape = buffer->shape.Map(
        [&](const PrimExpr &value) { return WithMode(value, true); });
    Array<PrimExpr> strides = buffer->strides.Map(
        [&](const PrimExpr &value) { return WithMode(value, true); });
    PrimExpr offset = WithMode(buffer->elem_offset, true);
    if (shape.same_as(buffer->shape) && strides.same_as(buffer->strides) &&
        offset.same_as(buffer->elem_offset))
      return buffer;
    Buffer result = buffer;
    auto *node = result.CopyOnWrite();
    node->shape = std::move(shape);
    node->strides = std::move(strides);
    node->elem_offset = std::move(offset);
    buffers_.emplace(buffer.get(), result);
    return result;
  }
  Var InternalVariable(const Var &variable) {
    if (selected_.count(variable.get()) && !shapes_.count(variable.get())) {
      if (auto found = var_remap_.find(variable.get());
          found != var_remap_.end())
        return found->second;
      Var replacement = variable.copy_with_dtype(DataType::Int(64));
      var_remap_.emplace(variable.get(), replacement);
      return replacement;
    }
    return variable;
  }
  PrimExpr VisitExpr(const PrimExpr &expression) final {
    const bool saved = index_;
    if (expression.dtype().is_uint() && !expression.dtype().is_bool())
      index_ = false;
    PrimExpr result = DataTypeLegalizer::VisitExpr(expression);
    index_ = saved;
    return result;
  }
  PrimExpr VisitExpr_(const VarNode *op) final {
    if (auto found = var_remap_.find(op); found != var_remap_.end())
      return index_ ? PrimExpr(found->second) : cast(op->dtype, found->second);
    if (index_ && shapes_.count(op))
      return cast(DataType::Int(64), GetRef<Var>(op));
    return GetRef<Var>(op);
  }
  bool ShapeDerived(const PrimExpr &expression) const {
    for (const auto *variable : IndexDependencies().Run(expression))
      if (shapes_.count(variable) || selected_.count(variable))
        return true;
    return false;
  }
  PrimExpr VisitExpr_(const CastNode *op) final {
    const bool transparent =
        index_ && ShapeDerived(op->value) && ValuePreservingNarrow(op);
    PrimExpr value = WithMode(op->value, false);
    if (transparent)
      return value;
    if (value.same_as(op->value))
      return GetRef<PrimExpr>(op);
    return Cast(op->dtype, value, op->span);
  }
  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    Buffer buffer = IndexBuffer(op->buffer);
    Array<PrimExpr> indices = op->indices.Map(
        [&](const PrimExpr &index) { return WithMode(index, true); });
    if (buffer.same_as(op->buffer) && indices.same_as(op->indices))
      return GetRef<PrimExpr>(op);
    auto result = GetRef<BufferLoad>(op);
    result.CopyOnWrite()->buffer = std::move(buffer);
    result.CopyOnWrite()->indices = std::move(indices);
    return result;
  }
  Stmt VisitStmt_(const BufferStoreNode *op) final {
    Buffer buffer = IndexBuffer(op->buffer);
    PrimExpr value = WithMode(op->value, false);
    Array<PrimExpr> indices = op->indices.Map(
        [&](const PrimExpr &index) { return WithMode(index, true); });
    if (buffer.same_as(op->buffer) && value.same_as(op->value) &&
        indices.same_as(op->indices))
      return GetRef<Stmt>(op);
    auto result = GetRef<BufferStore>(op);
    result.CopyOnWrite()->buffer = std::move(buffer);
    result.CopyOnWrite()->value = std::move(value);
    result.CopyOnWrite()->indices = std::move(indices);
    return result;
  }
  Stmt VisitStmt_(const LetStmtNode *op) final {
    const bool promote = selected_.count(op->var.get());
    Var variable = InternalVariable(op->var);
    PrimExpr value = WithMode(op->value, promote);
    Stmt body = VisitStmt(op->body);
    if (variable.same_as(op->var) && value.same_as(op->value) &&
        body.same_as(op->body))
      return GetRef<Stmt>(op);
    return LetStmt(variable, cast(variable.dtype(), value), body, op->span);
  }
  PrimExpr VisitExpr_(const LetNode *op) final {
    const bool promote = selected_.count(op->var.get());
    Var variable = InternalVariable(op->var);
    PrimExpr value = WithMode(op->value, promote);
    PrimExpr body = VisitExpr(op->body);
    if (variable.same_as(op->var) && value.same_as(op->value) &&
        body.same_as(op->body))
      return GetRef<PrimExpr>(op);
    return Let(variable, cast(variable.dtype(), value), body, op->span);
  }
  Stmt VisitStmt_(const ForNode *op) final {
    Var variable = InternalVariable(op->loop_var);
    PrimExpr minimum = cast(variable.dtype(), WithMode(op->min, true));
    PrimExpr extent = cast(variable.dtype(), WithMode(op->extent, true));
    Map<String, ObjectRef> annotations = IndexAnnotations(op->annotations);
    Stmt body = VisitStmt(op->body);
    if (variable.same_as(op->loop_var) && minimum.same_as(op->min) &&
        extent.same_as(op->extent) && annotations.same_as(op->annotations) &&
        body.same_as(op->body))
      return GetRef<Stmt>(op);
    auto result = GetRef<For>(op);
    auto *node = result.CopyOnWrite();
    node->loop_var = variable;
    node->min = std::move(minimum);
    node->extent = std::move(extent);
    node->annotations = std::move(annotations);
    node->body = std::move(body);
    if (op->thread_binding.defined()) {
      IterVar iteration = op->thread_binding.value();
      auto *binding = iteration.CopyOnWrite();
      binding->var = variable;
      if (binding->dom.defined())
        binding->dom = Range::FromMinExtent(
            cast(variable.dtype(), WithMode(binding->dom->min, true)),
            cast(variable.dtype(), WithMode(binding->dom->extent, true)));
      node->thread_binding = std::move(iteration);
    }
    return result;
  }
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != "thread_extent" && op->attr_key != "virtual_thread") {
      ObjectRef node = IndexMetadata(op->node);
      PrimExpr value = WithMode(op->value, false);
      Stmt body = VisitStmt(op->body);
      if (node.same_as(op->node) && value.same_as(op->value) &&
          body.same_as(op->body))
        return GetRef<Stmt>(op);
      return AttrStmt(node, op->attr_key, value, body, op->span);
    }
    const auto *old = op->node.as<IterVarNode>();
    ICHECK(old);
    Var variable = InternalVariable(old->var);
    PrimExpr extent = cast(variable.dtype(), WithMode(op->value, true));
    Range domain = old->dom;
    if (domain.defined()) {
      PrimExpr minimum = cast(variable.dtype(), WithMode(domain->min, true));
      PrimExpr span = cast(variable.dtype(), WithMode(domain->extent, true));
      if (!minimum.same_as(domain->min) || !span.same_as(domain->extent))
        domain = Range::FromMinExtent(minimum, span);
    }
    Stmt body = VisitStmt(op->body);
    if (variable.same_as(old->var) && domain.same_as(old->dom) &&
        extent.same_as(op->value) && body.same_as(op->body))
      return GetRef<Stmt>(op);
    IterVar iteration = GetRef<IterVar>(old);
    if (!variable.same_as(old->var) || !domain.same_as(old->dom))
      iteration = IterVar(domain, variable, old->iter_type, old->thread_tag);
    return AttrStmt(iteration, op->attr_key, extent, body, op->span);
  }
  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    Array<PrimExpr> values = op->iter_values.Map(
        [&](const PrimExpr &value) { return WithMode(value, true); });
    PrimExpr predicate = WithMode(op->predicate, true);
    Block block = Downcast<Block>(VisitStmt(op->block));
    for (size_t index = 0; index < values.size(); ++index)
      if (values[index].dtype() != block->iter_vars[index]->var.dtype())
        values.Set(index,
                   cast(block->iter_vars[index]->var.dtype(), values[index]));
    if (values.same_as(op->iter_values) && predicate.same_as(op->predicate) &&
        block.same_as(op->block))
      return GetRef<Stmt>(op);
    return BlockRealize(values, predicate, block, op->span);
  }
  Range IndexRange(const Range &range) {
    PrimExpr minimum = WithMode(range->min, true);
    PrimExpr extent = WithMode(range->extent, true);
    if (minimum.same_as(range->min) && extent.same_as(range->extent))
      return range;
    return Range::FromMinExtent(minimum, extent);
  }
  BufferRegion IndexRegion(const BufferRegion &region) {
    Buffer buffer = IndexBuffer(region->buffer);
    Array<Range> ranges = region->region.Map(
        [&](const Range &range) { return IndexRange(range); });
    if (buffer.same_as(region->buffer) && ranges.same_as(region->region))
      return region;
    return BufferRegion(buffer, ranges);
  }
  ObjectRef IndexMetadata(const ObjectRef &value) {
    if (value.as<PrimExprNode>())
      return WithMode(Downcast<PrimExpr>(value), true);
    if (value.as<RangeNode>())
      return IndexRange(Downcast<Range>(value));
    if (value.as<BufferNode>())
      return IndexBuffer(Downcast<Buffer>(value));
    if (value.as<BufferRegionNode>())
      return IndexRegion(Downcast<BufferRegion>(value));
    if (value.as<ArrayNode>())
      return Downcast<Array<ObjectRef>>(value).Map(
          [&](const ObjectRef &element) { return IndexMetadata(element); });
    if (value.as<MapNode>()) {
      const auto original = Downcast<Map<ObjectRef, ObjectRef>>(value);
      Map<ObjectRef, ObjectRef> result;
      bool changed = false;
      for (const auto &[key, element] : original) {
        ObjectRef new_key = key;
        if (const auto *variable = key.as<VarNode>())
          if (auto found = var_remap_.find(variable); found != var_remap_.end())
            new_key = found->second;
        ObjectRef new_element = IndexMetadata(element);
        changed |= !new_key.same_as(key) || !new_element.same_as(element);
        result.Set(new_key, new_element);
      }
      return changed ? ObjectRef(result) : value;
    }
    return value;
  }
  Map<String, ObjectRef>
  IndexAnnotations(const Map<String, ObjectRef> &annotations) {
    Map<String, ObjectRef> result = annotations;
    for (const auto &[key, value] : annotations) {
      ObjectRef rewritten = IndexMetadata(value);
      if (!rewritten.same_as(value))
        result.Set(key, rewritten);
    }
    return result;
  }
  Stmt VisitStmt_(const DeclBufferNode *op) final {
    Buffer buffer = IndexBuffer(op->buffer);
    Stmt body = VisitStmt(op->body);
    if (buffer.same_as(op->buffer) && body.same_as(op->body))
      return GetRef<Stmt>(op);
    auto result = GetRef<DeclBuffer>(op);
    result.CopyOnWrite()->buffer = std::move(buffer);
    result.CopyOnWrite()->body = std::move(body);
    return result;
  }
  Stmt VisitStmt_(const BufferRealizeNode *op) final {
    Buffer buffer = IndexBuffer(op->buffer);
    Array<Range> bounds =
        op->bounds.Map([&](const Range &range) { return IndexRange(range); });
    PrimExpr condition = WithMode(op->condition, true);
    Stmt body = VisitStmt(op->body);
    if (buffer.same_as(op->buffer) && bounds.same_as(op->bounds) &&
        condition.same_as(op->condition) && body.same_as(op->body))
      return GetRef<Stmt>(op);
    auto result = GetRef<BufferRealize>(op);
    result.CopyOnWrite()->buffer = std::move(buffer);
    result.CopyOnWrite()->bounds = std::move(bounds);
    result.CopyOnWrite()->condition = std::move(condition);
    result.CopyOnWrite()->body = std::move(body);
    return result;
  }
  Stmt VisitStmt_(const PrefetchNode *op) final {
    Buffer buffer = IndexBuffer(op->buffer);
    Array<Range> bounds =
        op->bounds.Map([&](const Range &range) { return IndexRange(range); });
    if (buffer.same_as(op->buffer) && bounds.same_as(op->bounds))
      return GetRef<Stmt>(op);
    auto result = GetRef<Prefetch>(op);
    result.CopyOnWrite()->buffer = std::move(buffer);
    result.CopyOnWrite()->bounds = std::move(bounds);
    return result;
  }
  Stmt VisitStmt_(const AllocateNode *op) final {
    Array<PrimExpr> extents = op->extents.Map(
        [&](const PrimExpr &extent) { return WithMode(extent, true); });
    PrimExpr condition = WithMode(op->condition, true);
    Stmt body = VisitStmt(op->body);
    if (extents.same_as(op->extents) && condition.same_as(op->condition) &&
        body.same_as(op->body))
      return GetRef<Stmt>(op);
    auto result = GetRef<Allocate>(op);
    result.CopyOnWrite()->extents = std::move(extents);
    result.CopyOnWrite()->condition = std::move(condition);
    result.CopyOnWrite()->body = std::move(body);
    return result;
  }
  Stmt VisitStmt_(const BlockNode *op) final {
    Array<IterVar> iterations = op->iter_vars.Map([&](const IterVar &old) {
      Var variable = InternalVariable(old->var);
      Range domain = old->dom.defined() ? IndexRange(old->dom) : old->dom;
      if (domain.defined())
        domain = Range::FromMinExtent(cast(variable.dtype(), domain->min),
                                      cast(variable.dtype(), domain->extent));
      if (variable.same_as(old->var) && domain.same_as(old->dom))
        return old;
      return IterVar(domain, variable, old->iter_type, old->thread_tag);
    });
    Array<Buffer> allocations = op->alloc_buffers.Map(
        [&](const Buffer &buffer) { return IndexBuffer(buffer); });
    Array<BufferRegion> reads = op->reads.Map(
        [&](const BufferRegion &region) { return IndexRegion(region); });
    Array<BufferRegion> writes = op->writes.Map(
        [&](const BufferRegion &region) { return IndexRegion(region); });
    Array<MatchBufferRegion> matches =
        op->match_buffers.Map([&](const MatchBufferRegion &match) {
          Buffer buffer = IndexBuffer(match->buffer);
          BufferRegion source = IndexRegion(match->source);
          if (buffer.same_as(match->buffer) && source.same_as(match->source))
            return match;
          return MatchBufferRegion(buffer, source);
        });
    Map<String, ObjectRef> annotations = IndexAnnotations(op->annotations);
    Optional<Stmt> initial;
    if (op->init.defined())
      initial = VisitStmt(op->init.value());
    Stmt body = VisitStmt(op->body);
    if (iterations.same_as(op->iter_vars) &&
        allocations.same_as(op->alloc_buffers) && reads.same_as(op->reads) &&
        writes.same_as(op->writes) && matches.same_as(op->match_buffers) &&
        annotations.same_as(op->annotations) && initial.same_as(op->init) &&
        body.same_as(op->body))
      return GetRef<Stmt>(op);
    auto result = GetRef<Block>(op);
    auto *node = result.CopyOnWrite();
    node->iter_vars = std::move(iterations);
    node->alloc_buffers = std::move(allocations);
    node->reads = std::move(reads);
    node->writes = std::move(writes);
    node->match_buffers = std::move(matches);
    node->annotations = std::move(annotations);
    node->init = std::move(initial);
    node->body = std::move(body);
    return result;
  }
  Stmt VisitStmt_(const IfThenElseNode *op) final {
    PrimExpr condition = WithMode(op->condition, true);
    Stmt then_case = VisitStmt(op->then_case);
    Optional<Stmt> else_case;
    if (op->else_case.defined())
      else_case = VisitStmt(op->else_case.value());
    if (condition.same_as(op->condition) && then_case.same_as(op->then_case) &&
        else_case.same_as(op->else_case))
      return GetRef<Stmt>(op);
    return IfThenElse(condition, then_case, else_case, op->span);
  }
  PrimExpr VisitExpr_(const CallNode *op) final {
    const bool region = op->op.same_as(Op::Get("tl.region"));
    const bool pointer = op->op.same_as(builtin::tvm_access_ptr());
    if (!region && !pointer) {
      const bool saved = index_;
      if (!op->op.same_as(builtin::if_then_else()))
        index_ = false;
      PrimExpr result = DataTypeLegalizer::VisitExpr_(op);
      index_ = saved;
      return result;
    }
    Array<PrimExpr> args;
    bool changed = false;
    for (size_t index = 0; index < op->args.size(); ++index) {
      const bool geometry = region ? index >= 2 : index == 2 || index == 3;
      PrimExpr argument = WithMode(op->args[index], geometry);
      changed |= !argument.same_as(op->args[index]);
      args.push_back(std::move(argument));
    }
    if (!changed)
      return GetRef<PrimExpr>(op);
    return Call(op->dtype, op->op, args, op->span);
  }

  std::unordered_set<const BufferNode *> parameter_buffers_;
  std::unordered_map<const BufferNode *, Buffer> buffers_;
  Variables shapes_;
  Variables selected_;
  bool index_{false};
};

} // namespace

tvm::transform::Pass AscendLegalizeShapeIndices(Target target) {
  auto pass = [target](PrimFunc function, IRModule module,
                       tvm::transform::PassContext context) {
    return ShapeIndexLegalizer::Substitute(std::move(function), target,
                                           context);
  };
  return tir::transform::CreatePrimFuncPass(
      pass, 0, "tl.AscendLegalizeShapeIndices", {});
}

TVM_REGISTER_GLOBAL("tl.transform.AscendLegalizeShapeIndices")
    .set_body_typed(AscendLegalizeShapeIndices);

} // namespace tvm::tl
