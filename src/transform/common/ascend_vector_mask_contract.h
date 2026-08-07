// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file ascend_vector_mask_contract.h
 * \brief Canonical contracts for compiler-managed Ascend Vector mask state.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_ASCEND_VECTOR_MASK_CONTRACT_H_
#define TVM_TL_TRANSFORM_COMMON_ASCEND_VECTOR_MASK_CONTRACT_H_

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/target/target.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace tl {

using namespace tir;

enum class AscendMaskMode : int32_t { kNormal = 0, kCounter = 1 };

enum class MaskRequirementKind : uint8_t { kAny, kExact };
enum class MaskEnsureKind : uint8_t { kPreserve, kExact, kUnknown };

/*!
 * \brief One canonical field of a mask contract.
 *
 * Only the five factory-produced forms are representable:
 * Any/Preserve, Any/Exact, Any/Unknown, Exact/Exact, Exact/Unknown.
 */
class MaskFieldContract {
public:
  static MaskFieldContract AnyPreserve();
  static MaskFieldContract AnyExact(PrimExpr ensured);
  static MaskFieldContract AnyUnknown();
  static MaskFieldContract ExactExact(PrimExpr required, PrimExpr ensured);
  static MaskFieldContract ExactUnknown(PrimExpr required);

  MaskRequirementKind requirement_kind() const { return requirement_kind_; }
  MaskEnsureKind ensure_kind() const { return ensure_kind_; }
  const PrimExpr &required() const { return required_; }
  const PrimExpr &ensured() const { return ensured_; }

private:
  MaskFieldContract(MaskRequirementKind requirement_kind,
                    MaskEnsureKind ensure_kind, PrimExpr required,
                    PrimExpr ensured);

  MaskRequirementKind requirement_kind_;
  MaskEnsureKind ensure_kind_;
  PrimExpr required_;
  PrimExpr ensured_;
};

struct MaskContract {
  MaskFieldContract mode;
  MaskFieldContract lo;
  MaskFieldContract hi;
};

struct MaskFacts {
  std::optional<AscendMaskMode> mode;
  std::optional<PrimExpr> lo;
  std::optional<PrimExpr> hi;

  static MaskFacts Unknown() { return {}; }
};

struct PayloadPair {
  PrimExpr lo;
  PrimExpr hi;
};

enum class NonTerminalMaskEffect : uint8_t {
  kNeutral,
  kBarrier,
  kUnclassified
};

enum class SelectedMaskContractKind : uint8_t {
  kRawCounter,
  kRawNormalDynamicPayload,
  kRawNormalFull,
  kNeutral,
  kUnknownAll,
  kCompositeNormalFull,
  kCompositeNormalFullToNormalUnknownPayload,
  kSelfContainedNormalFull,
  kSelfContainedNormalZero,
  kSelfContainedNormalDynamicPayload,
};

struct SelectedVectorTerminalSpec {
  Op selected;
  Op base;
  SelectedMaskContractKind contract_kind;
};

constexpr uint8_t kPossibleNormal = 1U << 0;
constexpr uint8_t kPossibleCounter = 1U << 1;

bool UseCompilerManagedVectorMask(const Target &target,
                                  const std::string &platform);

PrimExpr NormalizeMaskPayload(PrimExpr value, arith::Analyzer *analyzer);

bool MaskPayloadEqual(const PrimExpr &lhs, const PrimExpr &rhs,
                      arith::Analyzer *analyzer);

bool IsPureBufferFreeMaskPayload(
    const PrimExpr &value,
    const std::unordered_set<const VarNode *> &lexical_scope);

class AscendVectorMaskTargetProfile {
public:
  static const AscendVectorMaskTargetProfile &Get(const std::string &platform);

  bool LegalPayloadPair(uint8_t possible_modes, const PrimExpr &lo,
                        const PrimExpr &hi, arith::Analyzer *analyzer) const;

  PayloadPair CanonicalComplete(uint8_t possible_modes,
                                const std::optional<PrimExpr> &fixed_lo,
                                const std::optional<PrimExpr> &fixed_hi,
                                arith::Analyzer *analyzer) const;

private:
  explicit AscendVectorMaskTargetProfile(std::string platform)
      : platform_(std::move(platform)) {}

  std::string platform_;
};

bool IsSelectedVectorTerminal(const Call &call);
bool IsVectorMaskSetter(const Call &call);
const SelectedVectorTerminalSpec *
SelectedVectorTerminalSpecOf(const Call &call);
const std::vector<SelectedVectorTerminalSpec> &
SelectedVectorTerminalSpecsForBase(const Op &base);
const Op &BaseOperationOf(const Call &selected);
MaskContract MaskContractOf(const Call &selected, arith::Analyzer *analyzer);
NonTerminalMaskEffect ClassifyNonTerminalMaskEffect(const Call &call);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_ASCEND_VECTOR_MASK_CONTRACT_H_
