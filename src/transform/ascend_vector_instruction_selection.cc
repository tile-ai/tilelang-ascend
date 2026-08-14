// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*! \file ascend_vector_instruction_selection.cc
 *  \brief Verify semantic Vector ABI and materialize selected terminals.
 */

#include <algorithm>
#include <cctype>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/ascend.h"
#include "common/ascend_vector_mask.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

namespace {

struct ResolvedSemanticCall {
  const AscendVectorSemanticOpSpec *semantic;
  const AscendVectorTerminalVariant *variant;
  Array<PrimExpr> payload;
};

struct ReduceCallLayout {
  size_t clear_index;
  bool has_physical_row;
};

bool IsAccessPtr(const PrimExpr &expr) {
  const auto *call = expr.as<CallNode>();
  return call && call->op.same_as(tir::builtin::tvm_access_ptr());
}

bool IsCopyUbToUb(const CallNode *call) {
  if (!call->op.same_as(builtin::call_extern()) || call->args.empty()) {
    return false;
  }
  const auto *name = call->args[0].as<StringImmNode>();
  return name &&
         std::string(name->value).find("copy_ub_to_ub<") != std::string::npos;
}

std::string Trim(std::string value) {
  auto first = std::find_if_not(value.begin(), value.end(), [](char c) {
    return std::isspace(static_cast<unsigned char>(c));
  });
  auto last = std::find_if_not(value.rbegin(), value.rend(), [](char c) {
                return std::isspace(static_cast<unsigned char>(c));
              }).base();
  return first < last ? std::string(first, last) : std::string();
}

std::vector<std::string> TemplateArguments(const std::string &tag) {
  size_t begin = tag.find('<');
  size_t end = tag.rfind('>');
  ICHECK(begin < end) << "Malformed templated operation " << tag;
  std::stringstream stream(tag.substr(begin + 1, end - begin - 1));
  std::vector<std::string> result;
  std::string item;
  while (std::getline(stream, item, ',')) {
    result.push_back(Trim(item));
  }
  return result;
}

bool DTypeAllowed(DTypeDomain domain, DataType dtype) {
  if (dtype.is_void()) {
    return domain == DTypeDomain::kAny;
  }
  bool f16 = dtype == DataType::Float(16);
  bool f32 = dtype == DataType::Float(32);
  bool bf16 = dtype.is_bfloat16();
  bool i16 = dtype == DataType::Int(16);
  bool u16 = dtype == DataType::UInt(16);
  bool i32 = dtype == DataType::Int(32);
  bool u32 = dtype == DataType::UInt(32);
  switch (domain) {
  case DTypeDomain::kAny:
    return true;
  case DTypeDomain::kFloat16:
    return f16;
  case DTypeDomain::kFloat32:
    return f32;
  case DTypeDomain::kFloat16Float32:
    return f16 || f32;
  case DTypeDomain::kFloat16Float32Int16Int32:
    return f16 || f32 || i16 || i32;
  case DTypeDomain::kFloat16Float32Int32:
    return f16 || f32 || i32;
  case DTypeDomain::kInt16UInt16:
    return i16 || u16;
  case DTypeDomain::kShiftInteger:
    return i16 || u16 || i32 || u32;
  case DTypeDomain::kDuplicate:
    return f16 || bf16 || f32 || i16 || u16 || i32 || u32;
  }
  LOG(FATAL) << "Unknown selected Vector dtype domain";
  throw;
}

void ValidateSameBufferDTypes(const ResolvedSemanticCall &resolved,
                              const Array<PrimExpr> &args,
                              std::initializer_list<size_t> indices) {
  DataType expected = DataType::Void();
  for (size_t index : indices) {
    ICHECK_LT(index, args.size());
    DataType actual = VectorAccessPtrDtype(args[index]);
    if (expected.is_void()) {
      expected = actual;
    } else {
      ICHECK_EQ(actual, expected)
          << "Unsupported AscendC Vector dtype tuple for "
          << resolved.semantic->base->name << ": buffer argument " << index
          << " has " << actual << " but the terminal requires " << expected
          << "; compiler-managed mask selection has no fallback codegen.";
    }
  }
}

bool IsMode(const std::string &mode,
            std::initializer_list<const char *> allowed) {
  return std::any_of(allowed.begin(), allowed.end(),
                     [&](const char *candidate) { return mode == candidate; });
}

bool IsRoundingMode(const std::string &mode) {
  return IsMode(mode, {"CAST_RINT", "CAST_FLOOR", "CAST_CEIL", "CAST_ROUND",
                       "CAST_TRUNC"});
}

enum class CastModeDomain : uint8_t {
  kNone,
  kRounded,
  kRoundedOrNone,
  kAll,
};

struct CastRule {
  DataType dst;
  DataType src;
  CastModeDomain modes;
};

bool IsCastModeAllowed(CastModeDomain domain, const std::string &mode) {
  switch (domain) {
  case CastModeDomain::kNone:
    return mode == "CAST_NONE";
  case CastModeDomain::kRounded:
    return IsRoundingMode(mode);
  case CastModeDomain::kRoundedOrNone:
    return mode == "CAST_NONE" || IsRoundingMode(mode);
  case CastModeDomain::kAll:
    return mode == "CAST_NONE" || mode == "CAST_ODD" || IsRoundingMode(mode);
  }
  LOG(FATAL) << "Unknown Cast mode domain";
  throw;
}

bool IsCastTupleModeSupported(DataType dst, DataType src,
                              const std::string &mode) {
  const DataType f16 = DataType::Float(16);
  const DataType f32 = DataType::Float(32);
  const DataType bf16 = DataType::BFloat(16);
  const DataType i4 = DataType::Int(4);
  const DataType i8 = DataType::Int(8);
  const DataType u8 = DataType::UInt(8);
  const DataType i16 = DataType::Int(16);
  const DataType i32 = DataType::Int(32);
  const DataType i64 = DataType::Int(64);
  static const std::vector<CastRule> rules = {
      // The dav-c220 dequant conversion ignores RoundMode after SetDeqScale.
      {f16, i32, CastModeDomain::kAll},
      {f16, i8, CastModeDomain::kNone},
      {f16, u8, CastModeDomain::kNone},
      {f16, i4, CastModeDomain::kNone},
      {f16, i16, CastModeDomain::kRoundedOrNone},
      {f16, f32, CastModeDomain::kAll},
      {f32, i32, CastModeDomain::kRoundedOrNone},
      {f32, f16, CastModeDomain::kNone},
      {f32, bf16, CastModeDomain::kNone},
      {f32, i16, CastModeDomain::kNone},
      {f32, f32, CastModeDomain::kRounded},
      {f32, i64, CastModeDomain::kRounded},
      {i32, f16, CastModeDomain::kRounded},
      {i32, f32, CastModeDomain::kRounded},
      {i32, bf16, CastModeDomain::kRounded},
      {i32, i64, CastModeDomain::kNone},
      {i16, f16, CastModeDomain::kRounded},
      {i16, f32, CastModeDomain::kRounded},
      {i16, i32, CastModeDomain::kNone},
      {i8, f16, CastModeDomain::kRoundedOrNone},
      {u8, f16, CastModeDomain::kRoundedOrNone},
      {bf16, f32, CastModeDomain::kRounded},
      {i64, f32, CastModeDomain::kRounded},
      {i64, i32, CastModeDomain::kNone},
      {i4, f16, CastModeDomain::kRoundedOrNone},
  };
  for (const CastRule &rule : rules) {
    if (rule.dst == dst && rule.src == src) {
      return IsCastModeAllowed(rule.modes, mode);
    }
  }
  return false;
}

void ValidateCast(const ResolvedSemanticCall &resolved,
                  const Array<PrimExpr> &args) {
  ICHECK_EQ(args.size(), 4U);
  DataType dst = VectorAccessPtrDtype(args[0]);
  DataType src = VectorAccessPtrDtype(args[1]);
  const auto *mode = args[2].as<StringImmNode>();
  ICHECK(mode != nullptr)
      << "Malformed AscendC Cast ABI: mode must be a string";
  ICHECK(IsCastTupleModeSupported(dst, src, mode->value))
      << "Unsupported AscendC Cast tuple for " << resolved.semantic->base->name
      << " (dst=" << dst << ", src=" << src << ", mode=" << mode->value
      << "); compiler-managed mask selection has no fallback codegen.";
}

void ValidateBroadcast(const ResolvedSemanticCall &resolved,
                       const Array<PrimExpr> &args) {
  ValidateSameBufferDTypes(resolved, args, {1, 2});
  DataType dtype = VectorAccessPtrDtype(args[1]);
  const auto *rank = args[3].as<IntImmNode>();
  ICHECK(rank != nullptr);
  PrimExpr dst_size = 1;
  PrimExpr src_size = 1;
  for (int64_t i = 0; i < rank->value; ++i) {
    dst_size *= args[4 + i];
    src_size *= args[4 + rank->value + i];
  }
  arith::Analyzer analyzer;
  bool equal = analyzer.CanProveEqual(dst_size, src_size);
  bool scalar = analyzer.CanProveEqual(src_size, 1);
  bool muls_dtype = dtype == DataType::Float(16) ||
                    dtype == DataType::Float(32) ||
                    dtype == DataType::Int(16) || dtype == DataType::Int(32);
  bool duplicate_dtype = DTypeAllowed(DTypeDomain::kDuplicate, dtype);
  ICHECK(equal || scalar)
      << "Unsupported workspace-free AscendC Broadcast shape: source and "
         "destination sizes must be provably equal or the source must be "
         "provably scalar; compiler-managed mask selection has no fallback "
         "codegen.";
  ICHECK(equal ? muls_dtype : duplicate_dtype)
      << "Unsupported workspace-free AscendC Broadcast dtype " << dtype
      << " for the " << (equal ? "equal-shape Muls" : "scalar Duplicate")
      << " path; compiler-managed mask selection has no fallback codegen.";
}

void ValidateTerminalBufferDTypes(const ResolvedSemanticCall &resolved,
                                  const Array<PrimExpr> &args) {
  switch (resolved.variant->operands) {
  case OperandRecipe::kSame012:
    return ValidateSameBufferDTypes(resolved, args, {0, 1, 2});
  case OperandRecipe::kSame01:
    return ValidateSameBufferDTypes(resolved, args, {0, 1});
  case OperandRecipe::kScalar:
    ValidateSameBufferDTypes(resolved, args, {0, 1});
    if (IsAccessPtr(args[2])) {
      ValidateSameBufferDTypes(resolved, args, {0, 2});
    }
    return;
  case OperandRecipe::kAxpy:
    return;
  case OperandRecipe::kReduce:
    return ValidateSameBufferDTypes(resolved, args, {1, 2});
  case OperandRecipe::kCast:
    return ValidateCast(resolved, args);
  case OperandRecipe::kBroadcast:
    return ValidateBroadcast(resolved, args);
  case OperandRecipe::kSame12:
    return ValidateSameBufferDTypes(resolved, args, {1, 2});
  case OperandRecipe::kRowExpand:
    ValidateSameBufferDTypes(resolved, args, {1, 2, 3});
    if (args.size() == 5) {
      ValidateSameBufferDTypes(resolved, args, {1, 4});
    }
    return;
  case OperandRecipe::kNone:
    return;
  }
  LOG(FATAL) << "Unknown selected Vector operand recipe";
}

ReduceCallLayout ParseReduceCallLayout(const Array<PrimExpr> &args,
                                       const char *name) {
  std::string malformed = "Malformed " + std::string(name) + " semantic ABI: ";
  ICHECK_GE(args.size(), 4U) << malformed << "expected at least 4 arguments";
  size_t clear_index = args.size() - 1;
  bool has_physical_row = false;
  if (!args[clear_index].dtype().is_bool()) {
    ICHECK(args[clear_index].as<IntImmNode>())
        << malformed << "physical row must be an integer constant";
    has_physical_row = true;
    ICHECK_GT(clear_index, 0U);
    --clear_index;
  }
  ICHECK(args[clear_index].dtype().is_bool())
      << malformed << "clear must precede the optional physical row";
  ICHECK_GE(clear_index, 3U);
  size_t tmp_count = clear_index - 3;
  ICHECK_LE(tmp_count, 1U) << malformed
                           << "AscendC reduce accepts at most one tmp view";
  if (tmp_count != 0) {
    ICHECK(IsAccessPtr(args[3]))
        << malformed << "tmp operands must be access_ptr views";
  }
  return {clear_index, has_physical_row};
}

void ValidateSemanticAbi(const ResolvedSemanticCall &resolved,
                         const Array<PrimExpr> &args) {
  const AscendVectorSemanticOpSpec &semantic = *resolved.semantic;
  const AscendVectorTerminalVariant &variant = *resolved.variant;
  std::string malformed =
      "Malformed " + std::string(variant.name) + " semantic ABI: ";
  size_t arity = args.size();
  ICHECK_GE(arity, semantic.min_arity)
      << malformed << "expected at least "
      << static_cast<int>(semantic.min_arity) << " arguments, got " << arity;
  ICHECK(semantic.max_arity == 255 || arity <= semantic.max_arity)
      << malformed << "expected at most "
      << static_cast<int>(semantic.max_arity) << " arguments, got " << arity;
  switch (semantic.abi) {
  case AbiRecipe::kArityOnly:
    return;
  case AbiRecipe::kCopyUb:
    ICHECK(arity == 3 || arity == 9)
        << malformed << "copy_ub_to_ub requires 3 or 9 arguments, got "
        << arity;
    return;
  case AbiRecipe::kReduce: {
    ReduceCallLayout layout = ParseReduceCallLayout(args, variant.name);
    ICHECK_EQ(variant.selector == SelectorRecipe::kReduceNarrow,
              layout.has_physical_row)
        << malformed
        << "physical-row presence does not match the selected "
           "reduce variant";
    return;
  }
  case AbiRecipe::kBroadcast: {
    size_t rank_index =
        variant.selector == SelectorRecipe::kBroadcastCounter ? 3 : 4;
    ICHECK_GT(args.size(), rank_index);
    const auto *rank = args[rank_index].as<IntImmNode>();
    ICHECK(rank && (rank->value == 1 || rank->value == 2))
        << malformed << "rank must be 1 or 2";
    size_t expected = rank_index + 1 + 2 * rank->value;
    ICHECK_EQ(args.size(), expected)
        << malformed << "rank " << rank->value << " requires " << expected
        << " arguments, got " << args.size();
    return;
  }
  case AbiRecipe::kSelect: {
    ICHECK_GE(args.size(), 4U);
    const auto *kind = args[3].as<IntImmNode>();
    ICHECK(kind && kind->value >= 0 && kind->value <= 2)
        << malformed << "select source kind must be 0, 1, or 2";
    size_t expected = kind->value == 0 ? 8 : kind->value == 1 ? 9 : 7;
    ICHECK_EQ(args.size(), expected)
        << malformed << "select source kind " << kind->value << " requires "
        << expected << " arguments, got " << args.size();
    return;
  }
  case AbiRecipe::kMergeSort: {
    ICHECK_GE(args.size(), 2U);
    const auto *num_ways = args[1].as<IntImmNode>();
    ICHECK(num_ways && num_ways->value >= 2 && num_ways->value <= 4)
        << malformed << "merge_sort input count must be 2, 3, or 4";
    size_t expected = 3 + 2 * num_ways->value;
    ICHECK_EQ(args.size(), expected)
        << malformed << num_ways->value << " inputs require " << expected
        << " arguments, got " << args.size();
    return;
  }
  case AbiRecipe::kRound: {
    size_t expected = variant.selector == SelectorRecipe::kRoundCounter ? 3 : 4;
    ICHECK_EQ(arity, expected)
        << malformed << "expected " << expected << " arguments, got " << arity;
    return;
  }
  }
  LOG(FATAL) << "Unknown semantic Vector ABI recipe";
}

void ValidateSelectedDType(const ResolvedSemanticCall &resolved,
                           const Array<PrimExpr> &args) {
  DataType dtype = VectorDType(args);
  ICHECK(!dtype.is_void()) << "Selected Vector terminal "
                           << resolved.variant->name
                           << " has no typed buffer operand";
  ICHECK(DTypeAllowed(resolved.variant->dtype_domain, dtype))
      << "Unsupported AscendC Vector dtype " << dtype << " for "
      << resolved.semantic->base->name
      << "; compiler-managed mask selection has no fallback codegen.";
  ValidateTerminalBufferDTypes(resolved, args);
  if (resolved.semantic->base.same_as(ascend_gather())) {
    ICHECK_GE(args.size(), 3U);
    ValidateSameBufferDTypes(resolved, args, {0, 1});
    ICHECK_EQ(VectorAccessPtrDtype(args[2]), DataType::UInt(32))
        << "Unsupported AscendC Gather offset dtype; expected uint32";
    bool supported = dtype == DataType::Float(16) || dtype.is_bfloat16() ||
                     dtype == DataType::UInt(16) ||
                     dtype == DataType::Int(16) ||
                     dtype == DataType::Float(32) ||
                     dtype == DataType::UInt(32) || dtype == DataType::Int(32);
    ICHECK(supported)
        << "Unsupported AscendC Gather dtype " << dtype
        << "; dav-c220 Gather supports only 16-bit and 32-bit element "
           "families.";
  }
  if (resolved.variant->operands == OperandRecipe::kAxpy) {
    ICHECK_GE(args.size(), 2U);
    DataType source_dtype = VectorAccessPtrDtype(args[1]);
    bool supported =
        (dtype == DataType::Float(16) && source_dtype == DataType::Float(16)) ||
        (dtype == DataType::Float(32) && (source_dtype == DataType::Float(16) ||
                                          source_dtype == DataType::Float(32)));
    ICHECK(supported) << "Unsupported AscendC Axpy dtype tuple (dst=" << dtype
                      << ", src=" << source_dtype
                      << "). Supported tuples are (float16, float16), "
                         "(float32, float16), and (float32, float32); "
                         "compiler-managed mask selection has no fallback "
                         "codegen.";
  }
}

const AscendVectorTerminalVariant *
FindVariant(const AscendVectorSemanticOpSpec &semantic, SelectorRecipe recipe,
            DataType dtype = DataType::Void()) {
  for (const AscendVectorTerminalVariant &variant : semantic.variants) {
    if (variant.selector == recipe &&
        (dtype.is_void() || DTypeAllowed(variant.dtype_domain, dtype))) {
      return &variant;
    }
  }
  return nullptr;
}

ResolvedSemanticCall
ResolveSemanticCall(const Call &call,
                    const AscendVectorSemanticOpSpec &semantic,
                    arith::Analyzer *analyzer) {
  const AscendVectorTerminalVariant *variant = nullptr;
  Array<PrimExpr> payload;
  switch (semantic.variants.front().selector) {
  case SelectorRecipe::kAlways:
    variant = FindVariant(semantic, SelectorRecipe::kAlways);
    break;
  case SelectorRecipe::kNaturalNormal: {
    DataType dtype = VectorDType(call->args);
    ICHECK(!dtype.is_void())
        << "Vector operation has no typed buffer operand: " << call;
    variant = FindVariant(semantic, SelectorRecipe::kNaturalNormal, dtype);
    ICHECK(variant != nullptr)
        << "Unsupported AscendC Vector dtype " << dtype << " for "
        << semantic.base->name
        << "; compiler-managed mask selection has no fallback codegen.";
    PrimExpr length = analyzer->Simplify(call->args.back());
    const auto *constant = length.as<IntImmNode>();
    int64_t repeat = 1;
    int64_t lanes = 0;
    if (constant != nullptr && constant->value > 0 &&
        constant->value <=
            std::numeric_limits<int64_t>::max() / dtype.bytes()) {
      int64_t bytes = constant->value * dtype.bytes();
      if (bytes < 256) {
        lanes = constant->value;
      } else if (bytes % 256 == 0 && bytes / 256 <= 255) {
        repeat = bytes / 256;
        lanes = 256 / dtype.bytes();
      }
    }
    if (lanes != 0) {
      auto [lo, hi] = NormalMaskBits(lanes);
      payload = {IntImm(DataType::Int(32), 0),
                 IntImm(DataType::Int(32), repeat), lo, hi};
    } else {
      payload = {IntImm(DataType::Int(32), 1), IntImm(DataType::Int(32), 1),
                 analyzer->Simplify(cast(DataType::UInt(64), length)),
                 make_zero(DataType::UInt(64))};
    }
    break;
  }
  case SelectorRecipe::kCopyUbSameType: {
    ICHECK_GE(call->args.size(), 3U);
    variant = FindVariant(semantic, SelectorRecipe::kCopyUbSameType);
    break;
  }
  case SelectorRecipe::kReduceNarrow: {
    ICHECK_GE(call->args.size(), 4U);
    std::string tag = Downcast<StringImm>(call->args[0])->value;
    std::vector<std::string> params = TemplateArguments(tag);
    ICHECK_EQ(params.size(), 4U);
    ReduceCallLayout layout =
        ParseReduceCallLayout(call->args, semantic.base->name.c_str());
    bool physical_row = layout.has_physical_row;
    bool clear = !is_zero(call->args[layout.clear_index]);
    bool half_sum = clear && tag.find("reduce_sum<half") != std::string::npos;
    SelectorRecipe recipe = physical_row ? SelectorRecipe::kReduceNarrow
                            : half_sum   ? SelectorRecipe::kReduceHalfSum
                                         : SelectorRecipe::kReduceComposite;
    variant = FindVariant(semantic, recipe);
    if (physical_row || half_sum) {
      int64_t m = std::stoll(params[1]);
      int64_t n = std::stoll(params[2]);
      int64_t dim = std::stoll(params[3]);
      int64_t lanes = dim == -1 ? n : (dim == 0 ? m : m * n);
      auto [lo, hi] = NormalMaskBits(lanes);
      payload = {lo, hi};
    }
    break;
  }
  case SelectorRecipe::kNormalMaskArg: {
    size_t mask_index =
        semantic.base.same_as(ascend_block_reduce_max()) ||
                semantic.base.same_as(ascend_block_reduce_min()) ||
                semantic.base.same_as(ascend_block_reduce_sum())
            ? 3
            : 2;
    PrimExpr mask = analyzer->Simplify(call->args[mask_index]);
    const auto *constant = mask.as<IntImmNode>();
    ICHECK(constant && constant->value >= 0 && constant->value <= 128)
        << "NORMAL mask length must be a constant in [0, 128]";
    variant = FindVariant(semantic, SelectorRecipe::kNormalMaskArg);
    auto [lo, hi] = NormalMaskBits(constant->value);
    payload = {lo, hi};
    break;
  }
  case SelectorRecipe::kBroadcastComposite:
    ICHECK_GE(call->args.size(), 4U);
    if (call->args[3].as<CallNode>()) {
      variant = FindVariant(semantic, SelectorRecipe::kBroadcastComposite);
    } else {
      variant = FindVariant(semantic, SelectorRecipe::kBroadcastCounter);
      int64_t rank = Downcast<IntImm>(call->args[3])->value;
      ICHECK_GT(rank, 0);
      PrimExpr count = 1;
      for (int64_t i = 0; i < rank; ++i) {
        count *= call->args[4 + i];
      }
      payload = {analyzer->Simplify(count)};
    }
    break;
  case SelectorRecipe::kGatherMaskFixed:
    ICHECK_GE(call->args.size(), 4U);
    variant = FindVariant(semantic, SelectorRecipe::kGatherMaskFixed);
    break;
  case SelectorRecipe::kMergeSort:
    variant = FindVariant(semantic, SelectorRecipe::kMergeSort);
    break;
  case SelectorRecipe::kRoundCounter:
    ICHECK_GE(call->args.size(), 3U);
    variant = FindVariant(semantic, call->args[2].as<CallNode>()
                                        ? SelectorRecipe::kRoundComposite
                                        : SelectorRecipe::kRoundCounter);
    break;
  default:
    LOG(FATAL) << "Unexpected semantic resolver root for " << call;
  }
  ICHECK(variant != nullptr) << "No selector recipe for " << call;
  if (variant->payload == PayloadLayout::kCount && payload.empty()) {
    payload = {analyzer->Simplify(call->args.back())};
  }
  return ResolvedSemanticCall{&semantic, variant, std::move(payload)};
}

void ValidateResolved(const ResolvedSemanticCall &resolved,
                      const Array<PrimExpr> &semantic_args) {
  ValidateSemanticAbi(resolved, semantic_args);
  ValidateSelectedDType(resolved, semantic_args);
}

void ValidateSelected(const SelectedCallView &view) {
  ResolvedSemanticCall resolved{&view.semantic_spec(), &view.variant(), {}};
  ValidateResolved(resolved, view.semantic_args());

  arith::Analyzer analyzer;
  Call semantic(view.call()->dtype, view.semantic_spec().base,
                view.semantic_args(), view.call()->span);
  ResolvedSemanticCall expected =
      ResolveSemanticCall(semantic, view.semantic_spec(), &analyzer);
  ICHECK(expected.variant->selected.same_as(view.variant().selected))
      << "Selected Ascend Vector terminal does not match its semantic call: "
      << view.call();
  size_t semantic_arity = view.semantic_args().size();
  ICHECK_EQ(view.call()->args.size(), semantic_arity + expected.payload.size())
      << "Selected Ascend Vector terminal has the wrong payload arity: "
      << view.call();
  for (size_t index = 0; index < expected.payload.size(); ++index) {
    PrimExpr actual =
        analyzer.Simplify(view.call()->args[semantic_arity + index]);
    PrimExpr wanted = analyzer.Simplify(expected.payload[index]);
    ICHECK(StructuralEqual()(actual, wanted) ||
           analyzer.CanProveEqual(actual, wanted))
        << "Selected Ascend Vector terminal payload does not match its "
           "semantic call at field "
        << index << ": expected " << wanted << ", got " << actual;
  }
}

Call SelectCall(const CallNode *call, const ResolvedSemanticCall &resolved) {
  ValidateResolved(resolved, call->args);
  Array<PrimExpr> args = call->args;
  for (const PrimExpr &arg : resolved.payload) {
    args.push_back(arg);
  }
  Call selected(call->dtype, resolved.variant->selected, std::move(args),
                call->span);
  ValidateSelected(SelectedCallView(selected));
  return selected;
}

} // namespace

class AscendVectorInstructionSelector final : public StmtExprMutator {
public:
  static PrimFunc Rewrite(PrimFunc func, Target target, std::string platform) {
    if (!UseCompilerManagedVectorMask(target, platform)) {
      return func;
    }
    AscendVectorInstructionSelector selector;
    PrimFuncNode *copy = func.CopyOnWrite();
    copy->body = selector.VisitStmt(func->body);
    return func;
  }

private:
  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call == nullptr || !in_vector_region_) {
      return StmtExprMutator::VisitStmt_(op);
    }
    Call semantic = GetRef<Call>(call);
    if (IsSelectedVectorTerminal(semantic)) {
      ValidateSelected(SelectedCallView(semantic));
      return GetRef<Stmt>(op);
    }

    const AscendVectorSemanticOpSpec *spec = nullptr;
    if (IsCopyUbToUb(call) || !call->op.same_as(builtin::call_extern())) {
      spec = AscendVectorSemanticSpecOf(semantic);
    }
    if (spec != nullptr) {
      return Evaluate(
          SelectCall(call, ResolveSemanticCall(semantic, *spec, &analyzer_)),
          op->span);
    }
    ICHECK(!IsVectorMaskSetter(semantic))
        << "Internal Vector-mask operation appeared before selection: "
        << semantic;
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != "resource_scope") {
      return StmtExprMutator::VisitStmt_(op);
    }
    const auto *scope = op->value.as<IntImmNode>();
    ICHECK(scope && (scope->value == 0 || scope->value == 1));
    bool saved_region = in_vector_region_;
    in_vector_region_ = saved_region && scope->value == 1;
    Stmt body = VisitStmt(op->body);
    in_vector_region_ = saved_region;
    return AttrStmt(op->node, op->attr_key, op->value, body, op->span);
  }

  arith::Analyzer analyzer_;
  bool in_vector_region_{true};
};

tvm::transform::Pass AscendVectorInstructionSelection(Target target,
                                                      std::string platform) {
  auto pass_func = [=](PrimFunc func, IRModule, PassContext) {
    return AscendVectorInstructionSelector::Rewrite(std::move(func), target,
                                                    platform);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendVectorInstructionSelection",
                            {});
}

TVM_REGISTER_GLOBAL("tl.transform.AscendVectorInstructionSelection")
    .set_body_typed(AscendVectorInstructionSelection);

} // namespace tl
} // namespace tvm
