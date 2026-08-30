// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#ifndef TVM_TL_TRANSFORM_COMMON_ASCEND_VECTOR_MASK_H_
#define TVM_TL_TRANSFORM_COMMON_ASCEND_VECTOR_MASK_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <tvm/target/target.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace tl {

using namespace tir;

enum class SelectorRecipe : uint8_t {
  kAlways,
  kNaturalNormal,
  kLegacyUInt8,
  kCopyUbSameType,
  kReduceNarrow,
  kReduceHalfSum,
  kReduceComposite,
  kNormalMaskArg,
  kBroadcastComposite,
  kBroadcastCounter,
  kGatherMaskFixed,
  kMergeSort,
  kRoundCounter,
  kRoundComposite,
};

enum class AbiRecipe : uint8_t {
  kArityOnly,
  kCopyUb,
  kReduce,
  kBroadcast,
  kSelect,
  kMergeSort,
  kRound,
};

enum class OperandRecipe : uint8_t {
  kNone,
  kSame012,
  kSame01,
  kScalar,
  kAxpy,
  kReduce,
  kCast,
  kBroadcast,
  kSame12,
  kRowExpand,
};

enum class ContractRecipe : uint8_t {
  kNeutral,
  kUnknown,
  kPayloadFull,
  kCompositeNormalFullUnknown,
  kCompositeNormalFullToNormalUnknownPayload,
  kSelfContainedNormalFull,
  kSelfContainedNormalExplicit,
  kCreateVecIndex,
  kGatherCount,
  kNormalExplicitLowArg,
};

enum class DTypeDomain : uint8_t {
  kAny,
  kFloat16,
  kFloat32,
  kFloat16Float32,
  kFloat16Float32Int16Int32,
  kFloat16Float32Int32,
  kInt16UInt16,
  kShiftInteger,
  kDuplicate,
};

enum class PayloadLayout : uint8_t {
  kNone,
  kCount,
  kMask,
  kMaskSpec,
};

enum class EmitterFamily : uint8_t {
  kHelper,
  kRawBinary,
  kRawUnary,
  kRawScalar,
  kRawShift,
  kRawSubs,
  kRawDivs,
  kRawAxpy,
  kRawReduce,
  kRawBlockReduce,
  kRawWholeReduce,
  kRawCast,
  kRawBroadcast,
  kRawFill,
  kRawClampMaxMin,
  kRawClamp,
  kRawRound,
  kRawRowExpand,
  kRawExpExperiment,
};

struct AscendVectorTerminalVariant {
  const char *name;
  Op selected;
  SelectorRecipe selector;
  DTypeDomain dtype_domain;
  OperandRecipe operands;
  PayloadLayout payload;
  ContractRecipe helper_contract;
  EmitterFamily emitter;
  const char *intrinsic;
};

struct AscendVectorSemanticOpSpec {
  Op base;
  const char *base_callee;
  uint8_t min_arity;
  uint8_t max_arity;
  AbiRecipe abi;
  std::vector<AscendVectorTerminalVariant> variants;
};

DataType VectorDType(const Array<PrimExpr> &args);

class SelectedCallView {
public:
  explicit SelectedCallView(Call selected);

  const AscendVectorSemanticOpSpec &semantic_spec() const { return *semantic_; }
  const AscendVectorTerminalVariant &variant() const { return *variant_; }
  const Call &call() const { return selected_; }
  const Array<PrimExpr> &semantic_args() const { return semantic_args_; }
  DataType vector_dtype() const { return VectorDType(semantic_args_); }
  PrimExpr mask_mode() const;
  PrimExpr repeat_time() const;
  PrimExpr count() const;
  PrimExpr mask_lo() const;
  PrimExpr mask_hi() const;

private:
  Call selected_;
  const AscendVectorSemanticOpSpec *semantic_{nullptr};
  const AscendVectorTerminalVariant *variant_{nullptr};
  size_t semantic_arity_{0};
  Array<PrimExpr> semantic_args_;
};

DataType VectorAccessPtrDtype(const PrimExpr &expr);
std::pair<PrimExpr, PrimExpr> NormalMaskBits(int64_t lanes);

bool UseCompilerManagedVectorMask(const Target &target,
                                  const std::string &platform);
const std::vector<AscendVectorSemanticOpSpec> &AscendVectorSemanticOpCatalog();
const AscendVectorSemanticOpSpec *AscendVectorSemanticSpecOf(const Call &call);
bool IsSelectedVectorTerminal(const Call &call);
bool RequiresSelectedVectorTerminal(const Call &call);
bool IsVectorMaskSetter(const Call &call);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_ASCEND_VECTOR_MASK_H_
