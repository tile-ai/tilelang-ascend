#pragma once
#include <cstdint>

namespace reduce2d_v2::vector_delay {
enum class Dependency : uint8_t {
  VcgCompleteToVectorRead,
  VcgPartialToVectorRead,
  M1CompleteBinaryToWholeReduce,
  StagedCopyToTailVcg,
  WholeReduceToMerge,
  DirectColumnToMerge,
  MergeToNextLevel
};
enum class Unit : uint8_t { BackendStartSlots, DescriptorCredits, None };
enum class WaitKind : uint8_t { None, RepeatZeroVcg, VectorBarrier };
struct DelayInput {
  Dependency dependency{};
  int32_t naturalDistance{};
};
struct DelayRule {
  Unit unit{};
  int32_t requiredDistance{};
  bool hasNumericDelay{};
};
struct Delay {
  Dependency dependency{};
  DelayRule rule{};
  int32_t naturalDistance{}, remainingDistance{};
  uint32_t repeatZeroDescriptors{};
};
constexpr uint32_t DivUp(uint32_t x, uint32_t y) { return (x + y - 1) / y; }
constexpr DelayRule RuleFor(Dependency dependency, bool sum = false) {
  switch (dependency) {
  case Dependency::VcgCompleteToVectorRead:
    return {Unit::BackendStartSlots, sum ? 21 : 14, true};
  case Dependency::VcgPartialToVectorRead:
    return {Unit::BackendStartSlots, sum ? 27 : 20, true};
  case Dependency::M1CompleteBinaryToWholeReduce:
    return {Unit::BackendStartSlots, sum ? 16 : 14, true};
  case Dependency::StagedCopyToTailVcg:
    return {Unit::DescriptorCredits, 9, true};
  default:
    return {Unit::None, 0, false};
  }
}
constexpr uint32_t RepeatZeroDescriptors(Unit unit, int32_t remaining,
                                         int32_t entryCredit = 4) {
  if (remaining <= 0)
    return 0;
  if (unit == Unit::BackendStartSlots)
    return remaining <= entryCredit
               ? 1U
               : static_cast<uint32_t>(remaining - entryCredit);
  return static_cast<uint32_t>(remaining);
}
constexpr Delay CalculateDelay(Dependency dependency, int32_t naturalDistance,
                               bool sum = false) {
  const DelayRule rule = RuleFor(dependency, sum);
  const int32_t entryCredit =
      4 + rule.requiredDistance - RuleFor(dependency).requiredDistance;
  const int32_t remaining =
      rule.hasNumericDelay && rule.requiredDistance > naturalDistance
          ? rule.requiredDistance - naturalDistance
          : 0;
  const uint32_t rep0 =
      rule.hasNumericDelay
          ? RepeatZeroDescriptors(rule.unit, remaining, entryCredit)
          : 0U;
  return {dependency, rule, naturalDistance, remaining, rep0};
}
constexpr Delay CalculateDelay(DelayInput input) {
  return CalculateDelay(input.dependency, input.naturalDistance);
}
template <Dependency D, int32_t Natural, bool Sum = false> struct StaticDelay {
  inline static constexpr Delay value = CalculateDelay(D, Natural, Sum);
};
constexpr Dependency VcgWriteDependency(bool complete) {
  return complete ? Dependency::VcgCompleteToVectorRead
                  : Dependency::VcgPartialToVectorRead;
}
constexpr DelayInput M1RecursiveInput(bool completeWrite,
                                      uint32_t nextLogical) {
  return {VcgWriteDependency(completeWrite),
          static_cast<int32_t>(DivUp(nextLogical, 64))};
}
struct M1TwoBinarySchedule {
  uint32_t betweenBinaryReductions{}, beforeWholeReduce{};
};
constexpr int32_t EmptyRunDistance(uint32_t interveningDescriptors,
                                   uint32_t emptyRuns) {
  return static_cast<int32_t>(interveningDescriptors + 1 + 4 * emptyRuns);
}
constexpr M1TwoBinarySchedule M1TwoBinary(bool firstWriteComplete) {
  constexpr uint32_t beforeWhole =
      RepeatZeroDescriptors(Unit::BackendStartSlots, 14 - 1);
  const int32_t required = firstWriteComplete ? 14 : 20;
  const uint32_t between =
      EmptyRunDistance(1 + beforeWhole, 1) >= required ? 0U : 1U;
  return {between, beforeWhole};
}
constexpr DelayInput ColumnNonFinalInput(uint32_t m, bool stagedTail,
                                         bool sum = false) {
  const int32_t groups = static_cast<int32_t>(DivUp(m, 8));
  const int32_t merges = static_cast<int32_t>(DivUp(m, 64));
  int32_t natural = groups + merges;
  if (!stagedTail && m % 8) {
    natural = 9 + 6 * static_cast<int32_t>(m / 8);
  } else if (sum) {
    // The first merge consumes at most eight group-reduction outputs. Later
    // repeats also benefit from the transition into the binary consumer.
    const int32_t first = 2 * groups - (groups < 8 ? groups - 1 : 7);
    const int32_t later = groups + merges + 7;
    natural = first < later ? first : later;
  }
  return {VcgWriteDependency(stagedTail || m % 8 == 0), natural};
}
constexpr DelayInput ColumnFinalInput(uint32_t m, int32_t extraNatural = 0) {
  const int32_t q = m / 64;
  return {VcgWriteDependency(m % 8 == 0),
          (m % 8 == 0 ? static_cast<int32_t>(DivUp(m, 64))
                      : 1 + (q == 0 ? 0 : 6 + q)) +
              extraNatural};
}
struct ColumnMergeWait {
  uint32_t leading{}, tail{};
};
constexpr ColumnMergeWait SplitColumnAccumulatorWait(uint32_t m, Delay delay,
                                                     bool binarySum) {
  const uint32_t full = m / 64;
  const uint32_t original = delay.repeatZeroDescriptors;
  if (!binarySum || m % 8 == 0 || full == 0 || full >= 13 ||
      original >= 13 - full)
    return {original, 0};
  // Match each consumer repeat to its own producer. Keep the first full read
  // safe, then let useful full repeats supply progress before the tail read.
  const uint32_t leading = full < 6 ? 6 - full : 1;
  const uint32_t tailBound = full < 8 ? 8 - full : 0;
  uint32_t total = original < tailBound ? tailBound : original;
  if (total <= leading)
    total = leading + 1;
  return {leading, total - leading};
}
constexpr int32_t ParentColumnNaturalCredit(uint32_t step, bool stagedTail,
                                            uint32_t columns, uint32_t m) {
  return step > 0 && stagedTail && columns == 2 && m < 64 && m % 8 ? 7 : 0;
}
constexpr Delay ColumnNonFinal(uint32_t m, bool stagedTail) {
  return CalculateDelay(ColumnNonFinalInput(m, stagedTail));
}
constexpr Delay ColumnFinal(uint32_t m, int32_t extraNatural = 0) {
  return CalculateDelay(ColumnFinalInput(m, extraNatural));
}
constexpr Delay StagedTail(uint32_t m) {
  return CalculateDelay(
      {Dependency::StagedCopyToTailVcg, static_cast<int32_t>(m / 8)});
}
constexpr WaitKind ChooseWait(Delay delay, bool allowRepeatZero) {
  return !delay.rule.hasNumericDelay || !allowRepeatZero
             ? WaitKind::VectorBarrier
             : (delay.remainingDistance == 0 ? WaitKind::None
                                             : WaitKind::RepeatZeroVcg);
}
} // namespace reduce2d_v2::vector_delay
