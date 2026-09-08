// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*! \file ascend_vector_mask.cc
 *  \brief Terminal schema, lookup, and structural selected-call validation.
 */

#include "common/ascend_vector_mask.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>

#include <tvm/runtime/data_type.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>

#include "../op/ascend.h"

namespace tvm {
namespace tl {

namespace {

struct SelectedTerminalRef {
  const AscendVectorSemanticOpSpec *semantic;
  const AscendVectorTerminalVariant *variant;
};

size_t PayloadArity(PayloadLayout layout) {
  switch (layout) {
  case PayloadLayout::kNone:
    return 0;
  case PayloadLayout::kCount:
    return 1;
  case PayloadLayout::kMask:
    return 2;
  case PayloadLayout::kMaskSpec:
    return 4;
  }
  LOG(FATAL) << "Unknown selected Vector payload layout";
  throw;
}

bool IsInteger(DataType dtype) { return dtype.is_int() || dtype.is_uint(); }

std::string ExternCallee(const Call &call) {
  if (!call->op.same_as(tir::builtin::call_extern()) || call->args.empty())
    return "";
  const auto *name = call->args[0].as<StringImmNode>();
  if (name == nullptr)
    return "";
  std::string value = name->value;
  size_t qualified = value.rfind("::");
  if (qualified != std::string::npos) {
    value = value.substr(qualified + 2);
  }
  size_t templated = value.find('<');
  return value.substr(0, templated);
}

std::optional<SelectedTerminalRef> SelectedTerminalOf(const Call &call) {
  for (const AscendVectorSemanticOpSpec &semantic :
       AscendVectorSemanticOpCatalog()) {
    for (const AscendVectorTerminalVariant &variant : semantic.variants) {
      if (call->op.same_as(variant.selected)) {
        return SelectedTerminalRef{&semantic, &variant};
      }
    }
  }
  return std::nullopt;
}

void ValidatePayload(const AscendVectorTerminalVariant &variant,
                     const Array<PrimExpr> &payload) {
  ICHECK_EQ(payload.size(), PayloadArity(variant.payload))
      << "Malformed " << variant.name << " payload";
  if (variant.payload == PayloadLayout::kMaskSpec) {
    const auto *mode = payload[0].as<IntImmNode>();
    ICHECK(mode && mode->dtype == DataType::Int(32) &&
           (mode->value == 0 || mode->value == 1))
        << "Selected mask mode must be an int32 constant NORMAL or COUNTER";
    ICHECK(payload[1].dtype() == DataType::Int(32));
    const auto *repeat = payload[1].as<IntImmNode>();
    ICHECK(repeat && repeat->value >= 1 && repeat->value <= 255)
        << "Selected NORMAL repeat must be an int32 constant in [1, 255]";
    ICHECK(payload[2].dtype() == DataType::UInt(64) &&
           payload[3].dtype() == DataType::UInt(64))
        << "Selected NORMAL masks must be uint64";
  } else if (variant.payload == PayloadLayout::kMask) {
    ICHECK(payload[0].dtype() == DataType::UInt(64) &&
           payload[1].dtype() == DataType::UInt(64))
        << "Selected NORMAL masks must be uint64";
  } else if (variant.payload == PayloadLayout::kCount) {
    ICHECK(IsInteger(payload[0].dtype()))
        << "Selected COUNTER count must be integer typed";
  }
}

} // namespace

DataType VectorAccessPtrDtype(const PrimExpr &expr) {
  const auto *access = expr.as<CallNode>();
  ICHECK(access && access->op.same_as(tir::builtin::tvm_access_ptr()) &&
         !access->args.empty())
      << "Expected tvm_access_ptr, got " << expr;
  if (const auto *call = access->args[0].as<CallNode>()) {
    return call->dtype;
  }
  if (const auto *str = access->args[0].as<StringImmNode>()) {
    return DataType(runtime::String2DLDataType(str->value));
  }
  LOG(FATAL) << "Unexpected access_ptr dtype " << access->args[0];
  throw;
}

DataType VectorDType(const Array<PrimExpr> &args) {
  for (const PrimExpr &arg : args) {
    const auto *call = arg.as<CallNode>();
    if (call && call->op.same_as(tir::builtin::tvm_access_ptr())) {
      return VectorAccessPtrDtype(arg);
    }
  }
  return DataType::Void();
}

std::pair<PrimExpr, PrimExpr> NormalMaskBits(int64_t lanes) {
  ICHECK_GE(lanes, 0);
  ICHECK_LE(lanes, 128);
  auto word = [](int64_t width) {
    uint64_t value = width == 64  ? std::numeric_limits<uint64_t>::max()
                     : width == 0 ? 0
                                  : (uint64_t{1} << width) - 1;
    return make_const(DataType::UInt(64), value);
  };
  return {word(std::min<int64_t>(lanes, 64)),
          word(std::max<int64_t>(lanes - 64, 0))};
}

SelectedCallView::SelectedCallView(Call selected)
    : selected_(std::move(selected)) {
  std::optional<SelectedTerminalRef> resolved = SelectedTerminalOf(selected_);
  ICHECK(resolved.has_value())
      << "Expected a selected Ascend Vector terminal, got " << selected_;
  semantic_ = resolved->semantic;
  variant_ = resolved->variant;
  size_t payload_arity = PayloadArity(variant_->payload);
  ICHECK_GE(selected_->args.size(), payload_arity)
      << "Malformed " << variant_->name << " payload";
  semantic_arity_ = selected_->args.size() - payload_arity;
  ICHECK_GE(semantic_arity_, semantic_->min_arity)
      << "Malformed " << variant_->name << " semantic ABI: expected at least "
      << static_cast<int>(semantic_->min_arity) << " arguments, got "
      << semantic_arity_;
  ICHECK(semantic_->max_arity == 255 || semantic_arity_ <= semantic_->max_arity)
      << "Malformed " << variant_->name << " semantic ABI: expected at most "
      << static_cast<int>(semantic_->max_arity) << " arguments, got "
      << semantic_arity_;
  Array<PrimExpr> payload;
  for (size_t i = 0; i < selected_->args.size(); ++i) {
    if (i < semantic_arity_) {
      semantic_args_.push_back(selected_->args[i]);
    } else {
      payload.push_back(selected_->args[i]);
    }
  }
  ValidatePayload(*variant_, payload);
}

PrimExpr SelectedCallView::repeat_time() const {
  if (variant_->payload == PayloadLayout::kMaskSpec) {
    return selected_->args[semantic_arity_ + 1];
  }
  return IntImm(DataType::Int(32), 1);
}

PrimExpr SelectedCallView::mask_mode() const {
  ICHECK(variant_->payload == PayloadLayout::kMaskSpec)
      << "Selected terminal has no mask mode payload";
  return selected_->args[semantic_arity_];
}

PrimExpr SelectedCallView::count() const {
  ICHECK(variant_->payload == PayloadLayout::kCount)
      << "Selected terminal has no count payload";
  return selected_->args[semantic_arity_];
}

PrimExpr SelectedCallView::mask_lo() const {
  ICHECK(variant_->payload == PayloadLayout::kMask ||
         variant_->payload == PayloadLayout::kMaskSpec)
      << "Selected terminal has no NORMAL mask payload";
  size_t offset = variant_->payload == PayloadLayout::kMaskSpec ? 2 : 0;
  return selected_->args[semantic_arity_ + offset];
}

PrimExpr SelectedCallView::mask_hi() const {
  ICHECK(variant_->payload == PayloadLayout::kMask ||
         variant_->payload == PayloadLayout::kMaskSpec)
      << "Selected terminal has no NORMAL mask payload";
  size_t offset = variant_->payload == PayloadLayout::kMaskSpec ? 3 : 1;
  return selected_->args[semantic_arity_ + offset];
}

bool UseCompilerManagedVectorMask(const Target &target,
                                  const std::string &platform) {
  if (!target.defined() || (platform != "A2" && platform != "A3")) {
    return false;
  }
  Optional<ObjectRef> model = target->attrs.Get("model");
  if (!model.defined()) {
    return false;
  }
  std::string value = Downcast<String>(model.value());
  return value == "ascendc" || value == "auto";
}

const std::vector<AscendVectorSemanticOpSpec> &AscendVectorSemanticOpCatalog() {
  static const std::vector<AscendVectorSemanticOpSpec> catalog = [] {
    std::vector<AscendVectorSemanticOpSpec> result;
#define TL_ASCEND_SEMANTIC_OP(base, base_callee, min_arity, max_arity, abi)    \
  result.push_back(                                                            \
      {base(), base_callee, min_arity, max_arity, AbiRecipe::abi, {}});
#define TL_ASCEND_PHYSICAL(terminal, selector, dtype_domain, operands,         \
                           payload, emitter, intrinsic)                        \
  ICHECK(!result.empty());                                                     \
  result.back().variants.push_back(                                            \
      {#terminal, terminal(), SelectorRecipe::selector,                        \
       DTypeDomain::dtype_domain, OperandRecipe::operands,                     \
       PayloadLayout::payload, ContractRecipe::kNeutral,                       \
       EmitterFamily::emitter, intrinsic});
#define TL_ASCEND_HELPER(terminal, selector, contract)                         \
  ICHECK(!result.empty());                                                     \
  result.back().variants.push_back(                                            \
      {#terminal, terminal(), SelectorRecipe::selector, DTypeDomain::kAny,     \
       OperandRecipe::kNone, PayloadLayout::kNone, ContractRecipe::contract,   \
       EmitterFamily::kHelper, ""});
#include "../op/ascend_vector_mask_ops.inc"
#undef TL_ASCEND_HELPER
#undef TL_ASCEND_PHYSICAL
#undef TL_ASCEND_SEMANTIC_OP

    std::unordered_set<std::string> semantic_keys;
    std::unordered_set<std::string> names;
    std::unordered_set<std::string> selected_ops;
    for (const AscendVectorSemanticOpSpec &semantic : result) {
      ICHECK(semantic.base.defined());
      ICHECK_LE(semantic.min_arity, semantic.max_arity);
      ICHECK(!semantic.variants.empty())
          << "Semantic Vector operation has no variants: " << semantic.base;
      bool external = semantic.base.same_as(tir::builtin::call_extern());
      bool has_callee = semantic.base_callee && semantic.base_callee[0];
      ICHECK_EQ(external, has_callee)
          << "Only call_extern semantic operations have a callee identity";
      std::string key = semantic.base->name;
      if (external) {
        key += ":" + std::string(semantic.base_callee);
      }
      ICHECK(semantic_keys.insert(key).second)
          << "Duplicate semantic Vector operation " << key;

      std::unordered_set<int> selectors;
      for (const AscendVectorTerminalVariant &variant : semantic.variants) {
        ICHECK(names.insert(variant.name).second)
            << "Duplicate Ascend Vector terminal catalog name " << variant.name;
        ICHECK(selectors.insert(static_cast<int>(variant.selector)).second)
            << "Duplicate selector candidate for " << key;
        ICHECK(variant.selected.defined() &&
               selected_ops.insert(variant.selected->name).second)
            << "Selected Op appears more than once in the terminal catalog: "
            << variant.name;
        if (variant.emitter == EmitterFamily::kHelper) {
          ICHECK(variant.dtype_domain == DTypeDomain::kAny &&
                 variant.operands == OperandRecipe::kNone &&
                 variant.payload == PayloadLayout::kNone &&
                 (variant.intrinsic == nullptr || variant.intrinsic[0] == '\0'))
              << "Helper terminal " << variant.name
              << " cannot carry raw terminal metadata";
        } else {
          ICHECK(variant.intrinsic != nullptr && variant.intrinsic[0] != '\0')
              << "Raw terminal " << variant.name
              << " has no intrinsic identity";
        }
      }
    }
    return result;
  }();
  return catalog;
}

const AscendVectorSemanticOpSpec *AscendVectorSemanticSpecOf(const Call &call) {
  std::string callee = ExternCallee(call);
  for (const AscendVectorSemanticOpSpec &semantic :
       AscendVectorSemanticOpCatalog()) {
    if (!call->op.same_as(semantic.base)) {
      continue;
    }
    if (!semantic.base.same_as(tir::builtin::call_extern()) ||
        callee == semantic.base_callee) {
      return &semantic;
    }
  }
  return nullptr;
}

bool IsSelectedVectorTerminal(const Call &call) {
  return SelectedTerminalOf(call).has_value();
}

bool RequiresSelectedVectorTerminal(const Call &call) {
  return AscendVectorSemanticSpecOf(call) != nullptr;
}

bool IsVectorMaskSetter(const Call &call) {
  return call->op.same_as(ascend_set_mask_mode()) ||
         call->op.same_as(ascend_set_mask_payload());
}

TVM_REGISTER_GLOBAL("tl.transform.AscendVectorTerminalCatalog")
    .set_body_typed([]() {
      Array<String> names;
      for (const AscendVectorSemanticOpSpec &semantic :
           AscendVectorSemanticOpCatalog()) {
        for (const AscendVectorTerminalVariant &variant : semantic.variants) {
          names.push_back(variant.selected->name);
        }
      }
      return names;
    });

} // namespace tl
} // namespace tvm
