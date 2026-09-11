#pragma once
#include "reduce_2d_vector_delay.h"
#include <cstdint>
#include <type_traits>
#include <utility>
#if defined(__CCE__)
#include "kernel_operator.h"
#endif
namespace reduce2d_v2 {
// Compile-time plan.
constexpr uint32_t kDataBlockBytes = 32, kFp32PerDataBlock = 8, kMaxSteps = 16,
                   kMaxRepeats = 255, kUbElements = 196352 / sizeof(float);
constexpr uint32_t DivUp(uint32_t x, uint32_t y) { return (x + y - 1) / y; }
constexpr uint32_t AlignUp(uint32_t x, uint32_t y) { return DivUp(x, y) * y; }
constexpr uint32_t WithParity(uint32_t x, uint32_t parity) {
  return x + ((x ^ parity) & 1U);
}
enum class Tail : uint8_t { None, Full, Direct, Compact };
enum class Body : uint8_t { EvenEven, OddOdd, Mixed };
enum class Leaf : uint8_t {
  None,
  M1,
  Vcmax,
  M1TwoBinaryReductions,
  M1OneBinaryReduction,
  Columns
};
enum class Search : uint8_t { All, Compact, Raw };
enum class Reduce2DKind : uint8_t { kSum, kMax, kMin };
struct Step {
  uint32_t logical{}, pitch{}, nextLogical{}, nextPitch{};
  Tail tail{};
  Body body{};
  Leaf leaf{};
  bool merge{};
};
struct Plan {
  Step step[kMaxSteps]{};
  uint32_t count{};
  uint64_t cost4{};
  bool legal{};
};
struct Layout {
  uint32_t workElements[2]{}, work1Offset{}, auxiliaryOffset{}, elements{};
};
constexpr Layout MakeLayout(uint32_t work0, uint32_t work1, uint32_t auxiliary) {
  // Each used slot owns one extra DataBlock for its runtime parity adjustment.
  const uint32_t reserved0 = work0 ? work0 + kFp32PerDataBlock : 0;
  const uint32_t reserved1 = work1 ? work1 + kFp32PerDataBlock : 0;
  return {{work0, work1}, reserved0, reserved0 + reserved1,
          reserved0 + reserved1 + auxiliary};
}
constexpr Plan Invalid() { return {}; }
constexpr Plan Done() {
  Plan p{};
  p.legal = true;
  return p;
}
constexpr uint32_t TailRank(Tail tail) {
  return tail == Tail::Direct ? 1U : 0U;
}
constexpr Plan Best(Plan a, Plan b) {
  if (!a.legal)
    return b;
  if (!b.legal)
    return a;
  if (a.cost4 != b.cost4)
    return a.cost4 < b.cost4 ? a : b;
  if (a.step[0].nextPitch != b.step[0].nextPitch) {
    return a.step[0].nextPitch < b.step[0].nextPitch ? a : b;
  }
  return TailRank(a.step[0].tail) <= TailRank(b.step[0].tail) ? a : b;
}
constexpr Plan Prepend(Step step, Plan child, uint64_t cost4) {
  if (!child.legal || child.count >= kMaxSteps)
    return Invalid();
  for (uint32_t i = child.count; i != 0; --i) {
    child.step[i] = child.step[i - 1];
  }
  child.step[0] = step;
  ++child.count;
  child.cost4 += cost4;
  return child;
}
constexpr Body BodyFor(uint32_t pitch, uint32_t nextPitch) {
  return !(pitch & 1U) && !(nextPitch & 1U)
             ? Body::EvenEven
             : ((pitch & 1U) && (nextPitch & 1U) ? Body::OddOdd : Body::Mixed);
}
constexpr uint64_t StepCost4(uint32_t m, uint32_t logical, Tail tail,
                             Body body) {
  const uint32_t whole = logical / 128, rem = logical % 128, tailN = rem % 64;
  const uint64_t descriptors = body == Body::OddOdd
                                   ? uint64_t{2} * whole * DivUp(m, kMaxRepeats)
                                   : uint64_t{m} * whole;
  uint64_t cost = uint64_t{8} * m * whole + descriptors;
  if (rem >= 64)
    cost += uint64_t{4} * m;
  if (tail == Tail::Full)
    cost += uint64_t{4} * m;
  if (tail == Tail::Direct) {
    cost +=
        uint64_t{4} * DivUp(tailN, 8) * (7 * DivUp(m, 8) + DivUp(m, 64) + 8);
  }
  return cost;
}
constexpr bool UseColumns(uint32_t m, uint32_t logical, uint32_t pitch);
constexpr bool UseDirectColumns(uint32_t m, uint32_t logical, uint32_t pitch);
constexpr bool UseVcmax(uint32_t m, uint32_t logical, uint32_t pitch);
constexpr uint64_t ColumnCost4(uint32_t m, uint32_t logical, uint32_t pitch,
                               bool merge, bool stage);
constexpr Plan Solve(uint32_t m, uint32_t logical, uint32_t pitch, bool merge,
                     bool product, Search search);
constexpr Plan SolveM1Compact(uint32_t logical, uint32_t pitch, bool merge);
constexpr Plan SolveM1(uint32_t logical, uint32_t pitch, bool merge);
constexpr Plan Candidate(uint32_t m, uint32_t logical, uint32_t pitch,
                         bool merge, bool product, Tail tail,
                         uint32_t nextLogical, uint32_t nextPitch,
                         Search search) {
  const Body body = nextLogical ? BodyFor(pitch, nextPitch) : Body::EvenEven;
  const Step step{logical, pitch, nextLogical, nextPitch,
                  tail,    body,  Leaf::None,  merge};
  Plan child = nextLogical
                   ? Solve(m, nextLogical, nextPitch,
                           merge || tail == Tail::Direct, product, search)
                   : Done();
  if (product && child.count && child.step[0].leaf == Leaf::Columns) {
    const int32_t credit = vector_delay::ParentColumnNaturalCredit(
        1, true, DivUp(child.step[0].logical, 8), m);
    child.cost4 -= vector_delay::ColumnFinal(m).repeatZeroDescriptors -
                   vector_delay::ColumnFinal(m, credit).repeatZeroDescriptors;
  }
  uint64_t cost = StepCost4(m, logical, tail, body);
  const uint32_t tailN = logical % 64;
  if (product && tail == Tail::Direct && UseDirectColumns(m, tailN, pitch)) {
    const uint64_t columns = DivUp(tailN, 8);
    cost -= 4 * columns * (7 * DivUp(m, 8) + DivUp(m, 64) + 8);
    cost += ColumnCost4(m, tailN, pitch, merge, false);
  }
  return Prepend(step, child, cost);
}
constexpr Plan WithTail(uint32_t m, uint32_t logical, uint32_t pitch,
                        bool merge, bool product, Tail tail, Search search) {
  const uint32_t rem = logical % 128, tailN = rem % 64;
  if (tail == Tail::Full && (tailN == 0 || pitch < 8 * DivUp(logical, 64)))
    return Invalid();
  if (tail == Tail::Direct && tailN == 0)
    return Invalid();
  const uint32_t nextLogical = 16 * (logical / 128) + (rem >= 64 ? 8 : 0) +
                               (tail == Tail::Full ? DivUp(tailN, 8) : 0);
  if (nextLogical >= logical)
    return Invalid();
  if (nextLogical == 0) {
    return Candidate(m, logical, pitch, merge, product, tail, 0, 0, search);
  }
  const uint32_t compact = DivUp(nextLogical, 8);
  const uint32_t compactPitch =
      search == Search::Raw ? compact : WithParity(compact, pitch & 1U);
  Plan best = Candidate(m, logical, pitch, merge, product, tail, nextLogical,
                        compactPitch, search);
  if (product && search == Search::All && logical < 128 && tail == Tail::Full) {
    const uint32_t oppositePitch = WithParity(compact, (pitch & 1U) ^ 1U);
    if (oppositePitch != compactPitch)
      best = Best(best, Candidate(m, logical, pitch, merge, product, tail,
                                  nextLogical, oppositePitch, search));
  }
  if (search == Search::All && (m < 64 || m % 8 != 0)) {
    const uint32_t fullPitch =
        WithParity(8 * DivUp(nextLogical, 64), pitch & 1U);
    if (fullPitch != compactPitch) {
      best = Best(best, Candidate(m, logical, pitch, merge, product, tail,
                                  nextLogical, fullPitch, search));
    }
  }
  return best;
}
constexpr Plan Solve(uint32_t m, uint32_t logical, uint32_t pitch, bool merge,
                     bool product, Search search) {
  if (logical == 0)
    return Done();
  const uint32_t tailN = logical % 64;
  Plan best = tailN == 0 ? WithTail(m, logical, pitch, merge, product,
                                    Tail::None, search)
                         : Best(WithTail(m, logical, pitch, merge, product,
                                         Tail::Direct, search),
                                WithTail(m, logical, pitch, merge, product,
                                         Tail::Full, search));
  if (product) {
    if (UseVcmax(m, logical, pitch))
      best = Best(best, Prepend({logical, pitch, 0, 0, Tail::None,
                                 Body::EvenEven, Leaf::Vcmax, merge},
                                Done(), 27 * m));
    if (UseColumns(m, logical, pitch))
      best = Best(best,
                  Prepend({logical, pitch, 0, 0, Tail::None, Body::EvenEven,
                           Leaf::Columns, merge},
                          Done(), ColumnCost4(m, logical, pitch, merge, true)));
  }
  return best;
}
constexpr bool HasMixed512(const Plan &plan) {
  for (uint32_t i = 0; i < plan.count; ++i)
    if (plan.step[i].logical >= 128 && plan.step[i].nextLogical &&
        plan.step[i].body == Body::Mixed)
      return true;
  return false;
}
constexpr bool BankSafe(const Plan &plan) {
  if (HasMixed512(plan))
    return false;
  for (uint32_t i = 0; i < plan.count; ++i)
    if (plan.step[i].leaf == Leaf::Columns &&
        !(plan.step[i].pitch % 2 || plan.step[i].pitch % 4 == 2))
      return false;
  return true;
}
constexpr bool UseColumns(uint32_t m, uint32_t logical, uint32_t pitch) {
  const uint32_t blocks = DivUp(logical, 8);
  return m > 1 && blocks && blocks <= 15 && blocks <= pitch && pitch <= 31 &&
         (pitch % 2 || pitch % 4 == 2) && !(pitch % 2 && m % 8 && m > 129);
}
constexpr bool UseDirectColumns(uint32_t m, uint32_t logical, uint32_t pitch) {
  const uint32_t blocks = DivUp(logical, 8);
  return m > 1 && blocks && blocks <= 15 && blocks <= pitch && pitch <= 8191 &&
         (blocks == 1 || pitch % 4 == 2 ||
          (pitch <= 31 && pitch % 2 && !(m % 8 && m > 129)));
}
// Scratch placement and public traits.
constexpr uint32_t ColumnElements(uint32_t m, uint32_t logical, uint32_t pitch,
                                  bool merge, bool stage = true) {
  const uint32_t groups = DivUp(m, 8), columns = DivUp(logical, 8);
  if (columns == 1)
    return (merge ? AlignUp(m, 8) : 8) + (stage && m % 8 ? 8 * pitch * 8 : 0);
  uint32_t blocks =
      pitch % 2
          ? 15 + 2 * AlignUp(groups, 16)
          : (columns == 2 ? 1 + 2 * groups : 1 + 3 * AlignUp(2 * groups, 16));
  if (m % 8 && stage)
    blocks += 8 * pitch;
  return blocks * 8 + (merge ? AlignUp(m, 8) : 0);
}
constexpr Layout Place(uint32_t m, const Plan &plan) {
  if (plan.count == 1 && plan.step[0].leaf == Leaf::Columns) {
    return MakeLayout(0, 0,
                      ColumnElements(m, plan.step[0].logical,
                                     plan.step[0].pitch, plan.step[0].merge));
  }
  if (plan.count == 1 && (plan.step[0].leaf == Leaf::M1TwoBinaryReductions ||
                          plan.step[0].leaf == Leaf::M1OneBinaryReduction))
    return MakeLayout(0, 0, 64 + (plan.step[0].merge ? 8U : 0U));
  if (plan.count == 1 && plan.step[0].leaf != Leaf::None)
    return MakeLayout(0, 0, plan.step[0].merge ? AlignUp(m, 8) : 8);
  uint32_t work[2]{}, aux = AlignUp(m, 8);
  for (uint32_t i = 0; i < plan.count; ++i) {
    if (plan.step[i].leaf == Leaf::None && plan.step[i].nextLogical) {
      // Run writes even steps to work0 and odd steps to work1. Keep the full
      // physical row span: the next step can read padding beyond nextLogical.
      const uint32_t size = m * plan.step[i].nextPitch * 8;
      if (size > work[i % 2])
        work[i % 2] = size;
    }
    if (plan.step[i].leaf == Leaf::Columns) {
      const uint32_t size = ColumnElements(
          m, plan.step[i].logical, plan.step[i].pitch, plan.step[i].merge);
      if (size > aux)
        aux = size;
    }
    const uint32_t tail = plan.step[i].logical % 64;
    if (plan.step[i].tail == Tail::Direct) {
      const uint32_t size = UseDirectColumns(m, tail, plan.step[i].pitch)
                                ? ColumnElements(m, tail, plan.step[i].pitch,
                                                 plan.step[i].merge, false)
                                : DivUp(tail, 8) * AlignUp(m, 8);
      if (size > aux)
        aux = size;
    }
  }
  return MakeLayout(work[0], work[1], aux);
}
constexpr bool Fits(uint32_t m, uint32_t srcRowStride, const Plan &plan) {
  if (!plan.legal)
    return false;
  return uint64_t{m} * AlignUp(srcRowStride, 8) + AlignUp(m, 8) +
             Place(m, plan).elements <=
         kUbElements;
}
constexpr Plan Select(uint32_t m, uint32_t srcRowStride, uint32_t logical,
                      uint32_t pitch, bool merge, bool product = false) {
  Plan plan = Solve(m, logical, pitch, merge, product, Search::All);
  if (Fits(m, srcRowStride, plan))
    return plan;
  if (Fits(m, srcRowStride,
           plan = Solve(m, logical, pitch, merge, product, Search::Compact)))
    return plan;
  plan = Solve(m, logical, pitch, merge, product, Search::Raw);
  if (Fits(m, srcRowStride, plan) && !HasMixed512(plan))
    return plan;
  return Invalid();
}
constexpr bool UseVcmax(uint32_t m, uint32_t logical, uint32_t pitch) {
  const uint32_t blocks = DivUp(logical, 8);
  return logical <= 64 && pitch <= 7 &&
         ((m <= 2 && blocks >= 2) ||
          (m <= 7 && blocks >= 3 && logical % 8 != 0));
}
constexpr uint64_t ColumnCost4(uint32_t m, uint32_t logical, uint32_t pitch,
                               bool merge, bool stage) {
  const uint64_t columns = DivUp(logical, 8), groups = DivUp(m, 8),
                 reductions = DivUp(m, 64);
  const uint64_t merges = (pitch % 2 ? columns : columns - 1) + (merge ? 1 : 0);
  const uint64_t scanDesc = DivUp(m / 8, kMaxRepeats) + (m % 8 != 0);
  const uint64_t mergeDesc = DivUp(m / 64, kMaxRepeats) + (m % 64 != 0);
  if (columns == 1) {
    const auto update = vector_delay::CalculateDelay(
        vector_delay::ColumnNonFinalInput(m, stage).dependency, reductions);
    return 4 * (groups + (!stage && m % 8 ? 6 : 0) + (merge ? reductions : 0)) +
           scanDesc + (merge ? mergeDesc + update.repeatZeroDescriptors : 0);
  }
  const auto nonFinal = vector_delay::ColumnNonFinal(m, stage);
  const auto final = vector_delay::ColumnFinal(m);
  const auto update = vector_delay::CalculateDelay(
      vector_delay::ColumnFinalInput(m).dependency, reductions);
  const uint64_t nonFinalCount = pitch % 2 ? columns - 1 : columns - 2;
  return 4 * (columns * groups + merges * reductions +
              (!stage && m % 8 ? 6 * columns : 0)) +
         columns * scanDesc + merges * mergeDesc +
         nonFinalCount * nonFinal.repeatZeroDescriptors +
         final.repeatZeroDescriptors +
         (merge ? update.repeatZeroDescriptors : 0) +
         (stage && m % 8 ? 1 + vector_delay::StagedTail(m).repeatZeroDescriptors
                         : 0);
}
constexpr Plan SelectProduct(uint32_t m, uint32_t srcRowStride,
                             uint32_t logical, uint32_t pitch, bool merge) {
  if (m == 1) {
    const Plan p = SolveM1(logical, pitch, merge);
    return Fits(m, srcRowStride, p) ? p : Invalid();
  }
  Plan p = Select(m, srcRowStride, logical, pitch, merge, true);
  return p.legal ? p : Select(m, srcRowStride, logical, pitch, merge);
}
template <uint32_t M, uint32_t N, uint32_t SrcRowStride, bool Clear,
          bool Product = false>
struct GeneralTraits {
  static_assert(M > 0 && N > 0);
  static_assert(SrcRowStride >= N && (M == 1 || SrcRowStride % 8 == 0));
  inline static constexpr uint32_t kM = M;
  inline static constexpr Plan plan = [] {
    if constexpr (Product)
      return SelectProduct(M, SrcRowStride, N, DivUp(SrcRowStride, 8), !Clear);
    else
      return Select(M, SrcRowStride, N, DivUp(SrcRowStride, 8), !Clear);
  }();
  static_assert(plan.legal && plan.count < kMaxSteps && BankSafe(plan));
  inline static constexpr Layout layout = Place(M, plan);
  inline static constexpr uint32_t scratchElements = layout.elements;
};
template <typename T, Reduce2DKind Kind, bool Clear, uint32_t M, uint32_t N,
          uint32_t SrcRowStride, bool AllowRepeatZero = true>
struct Traits : GeneralTraits<M, N, SrcRowStride, Clear, true> {
  static_assert(std::is_same_v<T, float>);
  static_assert(Kind == Reduce2DKind::kMax || Kind == Reduce2DKind::kMin ||
                Kind == Reduce2DKind::kSum);
  inline static constexpr Reduce2DKind kind = Kind;
  inline static constexpr bool allowRepeatZero = AllowRepeatZero;
  inline static constexpr uint32_t kScratchElements =
      GeneralTraits<M, N, SrcRowStride, Clear, true>::scratchElements;
};
template <typename T, Reduce2DKind Kind, bool Clear, uint32_t M, uint32_t N,
          uint32_t SrcRowStride, bool AllowRepeatZero = true>
using Reduce2DTraits =
    Traits<T, Kind, Clear, M, N, SrcRowStride, AllowRepeatZero>;
template <class T, uint32_t Logical, uint32_t Pitch, bool Merge>
struct ColumnView {
  inline static constexpr uint32_t kM = T::kM;
  inline static constexpr Reduce2DKind kind = T::kind;
  inline static constexpr bool allowRepeatZero = T::allowRepeatZero;
  inline static constexpr Plan plan = Prepend(
      {Logical, Pitch, 0, 0, Tail::None, Body::EvenEven, Leaf::Columns, Merge},
      Done(), 0);
};
template <class T, uint32_t I> struct DirectTailStatic {
  inline static constexpr Step step = T::plan.step[I];
  inline static constexpr uint32_t logical = step.logical % 64;
  inline static constexpr uint32_t offset =
      step.logical / 128 * 128 + (step.logical % 128 >= 64 ? 64 : 0);
  inline static constexpr bool columns =
      UseDirectColumns(T::kM, logical, step.pitch);
  using View = ColumnView<T, logical, step.pitch, step.merge>;
};
template <class T, uint32_t I> struct StepStatic {
  inline static constexpr Step step = T::plan.step[I];
  inline static constexpr uint32_t columns = DivUp(step.logical, 8);
  inline static constexpr bool stageColumns =
      I == 0 || (T::kM % 8 && ((columns == 1 && !step.merge) ||
                               (columns == 2 && T::kM < 64)));
};
template <class T, uint32_t I> struct M1StepDelay;
template <typename T, Reduce2DKind Kind, bool Clear, uint32_t M, uint32_t N,
          uint32_t SrcRowStride>
constexpr uint32_t Reduce2DScratchElements() {
  return Traits<T, Kind, Clear, M, N, SrcRowStride>::scratchElements;
}
constexpr uint32_t Reduce2DScratchElements(uint32_t m, uint32_t n, uint32_t s,
                                           bool clear) {
  if (!m || !n || s < n || (m > 1 && s % 8) || uint64_t{m} * s >= kUbElements)
    return 0;
  const Plan p = SelectProduct(m, s, n, DivUp(s, 8), !clear);
  return p.legal && BankSafe(p) ? Place(m, p).elements : 0;
}
#if defined(__CCE__)
// Device primitives and recursive body.
template <uint32_t Rows, uint32_t Lanes> struct RowMask {
  static_assert(Rows && Rows <= 8 && Lanes && Lanes <= 8);
  inline static constexpr uint64_t value =
      ((uint64_t{1} << Lanes) - 1) *
      (uint64_t{0x0101010101010101} >> ((8 - Rows) * 8));
};
template <uint32_t Lanes> struct ContinuousMask {
  static_assert(Lanes && Lanes <= 64);
  inline static constexpr uint64_t value =
      Lanes == 64 ? ~uint64_t{0} : (uint64_t{1} << Lanes) - 1;
};
template <uint32_t Lanes> struct FullTailMask {
  static_assert(Lanes && Lanes < 64);
  inline static constexpr uint32_t blocks = DivUp(Lanes, 8);
  inline static constexpr uint64_t padding =
      blocks == 8 ? uint64_t{0} : ~uint64_t{0} << (blocks * 8);
  inline static constexpr uint64_t value =
      ((uint64_t{1} << Lanes) - 1) | padding;
};
template <uint64_t Mask> __aicore__ inline void SetMask() {
  set_vector_mask(uint64_t{0}, Mask);
}
__aicore__ inline void SetFullMask() {
  set_vector_mask(~uint64_t{0}, ~uint64_t{0});
}
template <uint32_t Count>
__aicore__ inline void Delay(__ubuf__ float *scratch) {
  if constexpr (Count) {
    vcgmax(scratch, scratch, 0, 1, 1, 8);
    Delay<Count - 1>(scratch);
  }
}
struct Context {
  __ubuf__ float *dst, *root, *work0, *work1, *aux;
};
template <class T>
__aicore__ inline void GroupReduce(__ubuf__ float *dst, __ubuf__ float *src,
                                   uint8_t repeat, uint16_t dstRepeatStride,
                                   uint16_t srcBlockStride,
                                   uint16_t srcRepeatStride) {
  if constexpr (T::kind == Reduce2DKind::kMax)
    vcgmax(dst, src, repeat, dstRepeatStride, srcBlockStride, srcRepeatStride);
  else if constexpr (T::kind == Reduce2DKind::kMin)
    vcgmin(dst, src, repeat, dstRepeatStride, srcBlockStride, srcRepeatStride);
  else
    vcgadd(dst, src, repeat, dstRepeatStride, srcBlockStride, srcRepeatStride);
}
template <class T>
__aicore__ inline void
BinaryReduce(__ubuf__ float *dst, __ubuf__ float *a, __ubuf__ float *b,
             uint8_t repeat, uint8_t dstBlockStride, uint8_t aBlockStride,
             uint8_t bBlockStride, uint8_t dstRepeatStride,
             uint8_t aRepeatStride, uint8_t bRepeatStride) {
  if constexpr (T::kind == Reduce2DKind::kMax)
    vmax(dst, a, b, repeat, dstBlockStride, aBlockStride, bBlockStride,
         dstRepeatStride, aRepeatStride, bRepeatStride);
  else if constexpr (T::kind == Reduce2DKind::kMin)
    vmin(dst, a, b, repeat, dstBlockStride, aBlockStride, bBlockStride,
         dstRepeatStride, aRepeatStride, bRepeatStride);
  else
    vadd(dst, a, b, repeat, dstBlockStride, aBlockStride, bBlockStride,
         dstRepeatStride, aRepeatStride, bRepeatStride);
}
template <class T>
__aicore__ inline void WholeReduce(__ubuf__ float *dst, __ubuf__ float *src,
                                   uint8_t repeat, uint16_t dstRepeatStride,
                                   uint16_t srcBlockStride,
                                   uint16_t srcRepeatStride) {
  if constexpr (T::kind == Reduce2DKind::kMax)
    vcmax(dst, src, repeat, dstRepeatStride, srcBlockStride, srcRepeatStride,
          ONLY_VALUE);
  else if constexpr (T::kind == Reduce2DKind::kMin)
    vcmin(dst, src, repeat, dstRepeatStride, srcBlockStride, srcRepeatStride,
          ONLY_VALUE);
  else
    vcadd(dst, src, repeat, dstRepeatStride, srcBlockStride, srcRepeatStride,
          0);
}
template <class T, class StaticDelay>
__aicore__ inline void WaitForVectorData(__ubuf__ float *scratch) {
  constexpr vector_delay::Delay delay = StaticDelay::value;
  constexpr vector_delay::WaitKind wait =
      vector_delay::ChooseWait(delay, T::allowRepeatZero);
  if constexpr (wait == vector_delay::WaitKind::RepeatZeroVcg)
    Delay<delay.repeatZeroDescriptors>(scratch);
  else if constexpr (wait == vector_delay::WaitKind::VectorBarrier)
    pipe_barrier(PIPE_V);
}
template <class T, uint32_t I, uint32_t SrcOffset, uint32_t DstOffset,
          uint32_t BlockStride, uint32_t RowStart = 0, uint32_t RowStride = 1,
          uint32_t Start = 0>
__aicore__ inline void ScanRows(__ubuf__ float *dst, __ubuf__ float *src) {
  constexpr Step s = T::plan.step[I];
  constexpr uint32_t rows = (T::kM - RowStart + RowStride - 1) / RowStride;
  if constexpr (Start < rows) {
    constexpr uint32_t count = rows - Start < kMaxRepeats ? rows - Start
                                                          : kMaxRepeats,
                       row = RowStart + RowStride * Start;
    GroupReduce<T>(dst + row * s.nextPitch * 8 + DstOffset,
                   src + row * s.pitch * 8 + SrcOffset, count,
                   RowStride * s.nextPitch, BlockStride, RowStride * s.pitch);
    ScanRows<T, I, SrcOffset, DstOffset, BlockStride, RowStart, RowStride,
             Start + count>(dst, src);
  }
}
template <class T, uint32_t I, uint32_t SrcOffset, uint32_t DstOffset,
          uint32_t BlockStride>
__aicore__ inline void Scan(__ubuf__ float *dst, __ubuf__ float *src) {
  if constexpr (T::plan.step[I].body == Body::Mixed) {
    ScanRows<T, I, SrcOffset, DstOffset, BlockStride, 0, 2>(dst, src);
    ScanRows<T, I, SrcOffset, DstOffset, BlockStride, 1, 2>(dst, src);
  } else
    ScanRows<T, I, SrcOffset, DstOffset, BlockStride>(dst, src);
}
template <class T, uint32_t I, uint32_t Row, size_t... Chunk>
__aicore__ inline void BodyRow(__ubuf__ float *dst, __ubuf__ float *src,
                               std::index_sequence<Chunk...>) {
  constexpr Step s = T::plan.step[I];
  (GroupReduce<T>(dst + Row * s.nextPitch * 8 + Chunk * 16,
                  src + Row * s.pitch * 8 + Chunk * 128, 2, 1, 2, 1),
   ...);
}
template <class T, uint32_t I, size_t... Row>
__aicore__ inline void BodyRows(__ubuf__ float *dst, __ubuf__ float *src,
                                std::index_sequence<Row...>) {
  constexpr uint32_t chunks = T::plan.step[I].logical / 128;
  (BodyRow<T, I, Row>(dst, src, std::make_index_sequence<chunks>{}), ...);
}
template <class T, uint32_t I, uint32_t Odd, size_t... Chunk>
__aicore__ inline void OddBody(__ubuf__ float *dst, __ubuf__ float *src,
                               std::index_sequence<Chunk...>) {
  (Scan<T, I, Chunk * 128 + Odd * 8, Chunk * 16 + Odd * 8, 2>(dst, src), ...);
}
template <class T, uint32_t I>
__aicore__ inline void EmitBody(__ubuf__ float *dst, __ubuf__ float *src) {
  constexpr Step s = T::plan.step[I];
  constexpr uint32_t chunks = s.logical / 128;
  if constexpr (chunks) {
    SetFullMask();
    if constexpr (s.body == Body::OddOdd) {
      OddBody<T, I, 0>(dst, src, std::make_index_sequence<chunks>{});
      OddBody<T, I, 1>(dst, src, std::make_index_sequence<chunks>{});
    } else
      BodyRows<T, I>(dst, src, std::make_index_sequence<T::kM>{});
  }
}
template <class T, uint32_t I>
__aicore__ inline void EmitRemainder(__ubuf__ float *dst, __ubuf__ float *src) {
  constexpr Step s = T::plan.step[I];
  if constexpr (s.logical % 128 >= 64) {
    SetFullMask();
    Scan<T, I, (s.logical / 128) * 128, (s.logical / 128) * 16, 1>(dst, src);
  }
  if constexpr (s.tail == Tail::Full) {
    SetMask<FullTailMask<s.logical % 64>::value>();
    Scan<T, I, (s.logical / 128) * 128 + (s.logical % 128 >= 64 ? 64 : 0),
         (s.logical / 128) * 16 + (s.logical % 128 >= 64 ? 8 : 0), 1>(dst, src);
  }
  if constexpr (s.tail == Tail::Compact) {
    SetMask<ContinuousMask<s.logical % 64>::value>();
    Scan<T, I, (s.logical / 128) * 128 + (s.logical % 128 >= 64 ? 64 : 0),
         (s.logical / 128) * 16 + (s.logical % 128 >= 64 ? 8 : 0), 1>(dst, src);
  }
}
template <class T, uint32_t I, uint32_t Block, uint32_t Valid,
          uint32_t Start = 0>
__aicore__ inline void ScanColumnGroups(__ubuf__ float *column,
                                        __ubuf__ float *src) {
  constexpr Step s = T::plan.step[I];
  constexpr uint32_t groups = T::kM / 8,
                     offset = (s.logical / 128) * 128 +
                              (s.logical % 128 >= 64 ? 64 : 0) + Block * 8;
  if constexpr (Start < groups) {
    constexpr uint32_t count =
        groups - Start < kMaxRepeats ? groups - Start : kMaxRepeats;
    GroupReduce<T>(column + Start * 8, src + Start * 8 * s.pitch * 8 + offset,
                   count, 1, s.pitch, 8 * s.pitch);
    ScanColumnGroups<T, I, Block, Valid, Start + count>(column, src);
  }
  if constexpr (Start == groups && T::kM % 8) {
    SetMask<RowMask<T::kM % 8, Valid>::value>();
    GroupReduce<T>(column + groups * 8, src + groups * 8 * s.pitch * 8 + offset,
                   1, 1, s.pitch, 8 * s.pitch);
  }
}
template <class T, bool Merge, uint32_t SrcStride>
__aicore__ inline void MergeRows(__ubuf__ float *dst, __ubuf__ float *column,
                                 uint8_t count, uint32_t offset) {
  if constexpr (!Merge && T::kind == Reduce2DKind::kSum)
    vcopy(reinterpret_cast<__ubuf__ uint32_t *>(dst + offset),
          reinterpret_cast<__ubuf__ uint32_t *>(column + offset * SrcStride),
          count, 1, SrcStride, 8, 8 * SrcStride);
  else
    BinaryReduce<T>(
        dst + offset, Merge ? dst + offset : column + offset * SrcStride,
        column + offset * SrcStride, count, 1, Merge ? 1 : SrcStride, SrcStride,
        8, Merge ? 8 : 8 * SrcStride, 8 * SrcStride);
}
template <class T, bool Merge, uint32_t SrcStride = 1, uint32_t Start = 0>
__aicore__ inline void MergeColumn(__ubuf__ float *dst,
                                   __ubuf__ float *column) {
  constexpr uint32_t full = T::kM / 64;
  if constexpr (Start < full) {
    constexpr uint32_t count =
        full - Start < kMaxRepeats ? full - Start : kMaxRepeats;
    SetMask<~uint64_t{0}>();
    MergeRows<T, Merge, SrcStride>(dst, column, count, Start * 64);
    MergeColumn<T, Merge, SrcStride, Start + count>(dst, column);
  } else if constexpr (Start == full && T::kM % 64) {
    SetMask<(uint64_t{1} << (T::kM % 64)) - 1>();
    MergeRows<T, Merge, SrcStride>(dst, column, 1, full * 64);
  }
}
template <class T, uint32_t I, uint32_t Start = 0>
__aicore__ inline void ScanVcmax(__ubuf__ float *reduced, __ubuf__ float *src) {
  constexpr Step s = T::plan.step[I];
  if constexpr (Start < T::kM) {
    constexpr uint32_t count =
        T::kM - Start < kMaxRepeats ? T::kM - Start : kMaxRepeats;
    WholeReduce<T>(reduced + Start, src + Start * s.pitch * 8, count, 1, 1,
                   s.pitch);
    ScanVcmax<T, I, Start + count>(reduced, src);
  }
}
template <class T, uint32_t I>
__aicore__ inline void EmitVcmax(Context &c, __ubuf__ float *src) {
  constexpr Step s = T::plan.step[I];
  __ubuf__ float *reduced = s.merge ? c.aux : c.dst;
  SetMask<ContinuousMask<s.logical>::value>();
  ScanVcmax<T, I>(reduced, src);
  if constexpr (s.merge) {
    WaitForVectorData<T, vector_delay::StaticDelay<
                             vector_delay::Dependency::WholeReduceToMerge, 0>>(
        c.aux);
    MergeColumn<T, true>(c.dst, reduced);
  }
}
template <class T, uint32_t I, uint32_t Column, uint32_t DstStride = 0,
          uint32_t Start = 0, bool Stage = true>
__aicore__ inline void ScanLeafColumn(__ubuf__ float *out, __ubuf__ float *src,
                                      __ubuf__ float *tail) {
  constexpr Step s = T::plan.step[I];
  constexpr uint32_t groups = T::kM / 8,
                     stride = DstStride ? DstStride : (s.pitch % 2 ? 1 : 2),
                     valid = Column + 1 == (s.logical + 7) / 8 && s.logical % 8
                                 ? s.logical % 8
                                 : 8;
  if constexpr (Start == 0)
    SetMask<RowMask<8, valid>::value>();
  if constexpr (Start < groups) {
    constexpr uint32_t count =
        groups - Start < kMaxRepeats ? groups - Start : kMaxRepeats;
    GroupReduce<T>(out + Start * stride * 8,
                   src + Start * 8 * s.pitch * 8 + Column * 8, count, stride,
                   s.pitch, 8 * s.pitch);
    ScanLeafColumn<T, I, Column, DstStride, Start + count, Stage>(out, src,
                                                                  tail);
  }
  if constexpr (Start == groups && T::kM % 8) {
    SetMask<RowMask<Stage ? 8 : T::kM % 8, valid>::value>();
    __ubuf__ float *input = Stage ? tail : src + groups * 8 * s.pitch * 8;
    GroupReduce<T>(out + groups * stride * 8, input + Column * 8, 1, stride,
                   s.pitch, 8 * s.pitch);
  }
}
template <class T, uint32_t I = 0, bool Stage = true> struct ColumnTiming {
  inline static constexpr uint32_t groups = (T::kM + 7) / 8;
  inline static constexpr uint32_t merge = (T::kM + 63) / 64;
  inline static constexpr Step step = T::plan.step[I];
  inline static constexpr uint32_t count = DivUp(step.logical, 8);
  inline static constexpr int32_t parentCredit =
      vector_delay::ParentColumnNaturalCredit(I, Stage, count, T::kM);
  inline static constexpr vector_delay::DelayInput nonFinal =
      vector_delay::ColumnNonFinalInput(T::kM, Stage);
  inline static constexpr vector_delay::DelayInput final =
      vector_delay::ColumnFinalInput(T::kM, parentCredit);
  inline static constexpr vector_delay::DelayInput update =
      vector_delay::ColumnFinalInput(T::kM);
  using NonFinal =
      vector_delay::StaticDelay<nonFinal.dependency, nonFinal.naturalDistance>;
  using Final =
      vector_delay::StaticDelay<final.dependency, final.naturalDistance>;
  using Update =
      vector_delay::StaticDelay<update.dependency, update.naturalDistance>;
};
// q/2q column leaf.
template <class T, uint32_t I, uint32_t Column>
__aicore__ inline __ubuf__ float *ColumnSlot(__ubuf__ float *base) {
  constexpr Step s = T::plan.step[I];
  constexpr uint32_t groups = ColumnTiming<T, I>::groups,
                     count = (s.logical + 7) / 8;
  if constexpr (s.pitch % 2)
    return base + (Column % 2) * ((groups + 15) / 16 * 16) * 8;
  else if constexpr (count == 2)
    return base + Column * 8;
  else
    return base +
           ((Column % 3) * ((2 * groups + 15) / 16 * 16) + Column % 2) * 8;
}
template <class T, bool Compact, uint32_t Start = 0>
__aicore__ inline void MergePair(__ubuf__ float *dst, __ubuf__ float *a,
                                 __ubuf__ float *b) {
  constexpr uint32_t repeats = T::kM / 64;
  if constexpr (Start < repeats) {
    constexpr uint32_t count = repeats - Start < kMaxRepeats ? repeats - Start
                                                             : kMaxRepeats,
                       in = Start * 128, out = Start * 64;
    SetMask<~uint64_t{0}>();
    BinaryReduce<T>(dst + out * (Compact ? 1 : 2), a + in, b + in, count,
                    Compact ? 1 : 2, 2, 2, Compact ? 8 : 16, 16, 16);
    MergePair<T, Compact, Start + count>(dst, a, b);
  }
  if constexpr (Start == repeats && T::kM % 64) {
    SetMask<(uint64_t{1} << (T::kM % 64)) - 1>();
    constexpr uint32_t in = repeats * 128, out = repeats * 64;
    BinaryReduce<T>(dst + out * (Compact ? 1 : 2), a + in, b + in, 1,
                    Compact ? 1 : 2, 2, 2, Compact ? 8 : 16, 16, 16);
  }
}
template <class T, uint32_t I, bool Stage, uint32_t Column = 1>
__aicore__ inline void
EmitOddColumns(Context &c, __ubuf__ float *src, __ubuf__ float *base,
               __ubuf__ float *tail, __ubuf__ float *result) {
  constexpr uint32_t count = (T::plan.step[I].logical + 7) / 8;
  if constexpr (Column < count) {
    ScanLeafColumn<T, I, Column, 0, 0, Stage>(ColumnSlot<T, I, Column>(base),
                                              src, tail);
    WaitForVectorData<T, typename ColumnTiming<T, I, Stage>::NonFinal>(c.aux);
    MergeColumn<T, Column != 1>(result, ColumnSlot<T, I, Column - 1>(base));
    EmitOddColumns<T, I, Stage, Column + 1>(c, src, base, tail, result);
  } else {
    WaitForVectorData<T, typename ColumnTiming<T, I, Stage>::Final>(c.aux);
    MergeColumn<T, true>(result, ColumnSlot<T, I, count - 1>(base));
  }
}
template <class T, uint32_t I, bool Stage, uint32_t Column = 1>
__aicore__ inline void
EmitEvenColumns(Context &c, __ubuf__ float *src, __ubuf__ float *base,
                __ubuf__ float *tail, __ubuf__ float *result) {
  constexpr uint32_t count = (T::plan.step[I].logical + 7) / 8;
  if constexpr (Column + 1 < count)
    ScanLeafColumn<T, I, Column + 1, 0, 0, Stage>(
        ColumnSlot<T, I, Column + 1>(base), src, tail);
  if constexpr (Column + 1 < count) {
    WaitForVectorData<T, typename ColumnTiming<T, I, Stage>::NonFinal>(c.aux);
    MergePair<T, false>(ColumnSlot<T, I, Column>(base),
                        ColumnSlot<T, I, Column - 1>(base),
                        ColumnSlot<T, I, Column>(base));
    EmitEvenColumns<T, I, Stage, Column + 1>(c, src, base, tail, result);
  } else {
    WaitForVectorData<T, typename ColumnTiming<T, I, Stage>::Final>(c.aux);
    MergePair<T, true>(result, ColumnSlot<T, I, Column - 1>(base),
                       ColumnSlot<T, I, Column>(base));
  }
}
template <class T, uint32_t I, bool Stage = true>
__aicore__ inline void EmitColumnsLeaf(Context &c, __ubuf__ float *src) {
  constexpr Step s = T::plan.step[I];
  constexpr uint32_t groups = (T::kM + 7) / 8;
  constexpr uint32_t count = (s.logical + 7) / 8;
  __ubuf__ float *result = s.merge ? c.aux : c.dst;
  __ubuf__ float *columnRaw = c.aux + (s.merge ? (T::kM + 7) / 8 * 8 : 0);
  constexpr uint32_t storage =
      s.pitch % 2 ? 15 + 2 * ((groups + 15) / 16 * 16)
                  : (count == 2 ? 1 + 2 * groups
                                : 1 + 3 * ((2 * groups + 15) / 16 * 16));
  __ubuf__ float *tail = columnRaw + (count == 1 ? 0 : storage * 8);
  if constexpr (Stage && T::kM % 8) {
    copy_ubuf_to_ubuf(tail, src + (T::kM - T::kM % 8) * s.pitch * 8, 0, 1,
                      (T::kM % 8) * s.pitch, 0, 0);
    WaitForVectorData<
        T, vector_delay::StaticDelay<
               vector_delay::Dependency::StagedCopyToTailVcg, T::kM / 8>>(
        c.aux);
  }
  if constexpr (count == 1) {
    ScanLeafColumn<T, I, 0, 1, 0, Stage>(result, src, tail);
    if constexpr (s.merge) {
      WaitForVectorData<T, typename ColumnTiming<T, I, Stage>::Final>(c.aux);
      MergeColumn<T, true>(c.dst, result);
    }
  } else {
    const uintptr_t raw = reinterpret_cast<uintptr_t>(columnRaw),
                    base =
                        reinterpret_cast<uintptr_t>(s.pitch % 2 ? result : src);
    const uint32_t pad = s.pitch % 2 ? (((base >> 5) + 8 - (raw >> 5)) & 15U)
                                     : (((base ^ raw) >> 5) & 1U) ^ 1U;
    __ubuf__ float *slots = columnRaw + pad * 8;
    if constexpr (s.pitch % 2 && count == 2) {
      ScanLeafColumn<T, I, 0, 1, 0, Stage>(result, src, tail);
      ScanLeafColumn<T, I, 1, 0, 0, Stage>(ColumnSlot<T, I, 1>(slots), src,
                                           tail);
      WaitForVectorData<T, typename ColumnTiming<T, I, Stage>::Final>(c.aux);
      MergeColumn<T, true>(result, ColumnSlot<T, I, 1>(slots));
    } else {
      ScanLeafColumn<T, I, 0, 0, 0, Stage>(ColumnSlot<T, I, 0>(slots), src,
                                           tail);
      if constexpr (s.pitch % 2)
        EmitOddColumns<T, I, Stage>(c, src, slots, tail, result);
      else {
        ScanLeafColumn<T, I, 1, 0, 0, Stage>(ColumnSlot<T, I, 1>(slots), src,
                                             tail);
        EmitEvenColumns<T, I, Stage>(c, src, slots, tail, result);
      }
    }
    if constexpr (s.merge) {
      WaitForVectorData<T, typename ColumnTiming<T, I, Stage>::Update>(c.aux);
      MergeColumn<T, true>(c.dst, result);
    }
  }
}
template <class T, uint32_t I, uint32_t Block>
__aicore__ inline void DirectBlock(Context &c, __ubuf__ float *src) {
  constexpr Step s = T::plan.step[I];
  constexpr uint32_t blocks = (s.logical % 64 + 7) / 8,
                     valid = Block + 1 == blocks && s.logical % 8
                                 ? s.logical % 8
                                 : 8;
  __ubuf__ float *column = c.aux + Block * ((T::kM + 7) / 8 * 8);
  if constexpr (T::kM >= 8)
    SetMask<RowMask<8, valid>::value>();
  ScanColumnGroups<T, I, Block, valid>(column, src);
  WaitForVectorData<T, vector_delay::StaticDelay<
                           vector_delay::Dependency::DirectColumnToMerge, 0>>(
      c.aux);
  MergeColumn<T, s.merge || Block != 0>(c.dst, column);
  if constexpr (s.nextLogical && Block + 1 == blocks)
    WaitForVectorData<T, vector_delay::StaticDelay<
                             vector_delay::Dependency::MergeToNextLevel, 0>>(
        c.aux);
}
template <class T, uint32_t I, size_t... Block>
__aicore__ inline void EmitDirect(Context &c, __ubuf__ float *src,
                                  std::index_sequence<Block...>) {
  (DirectBlock<T, I, Block>(c, src), ...);
}
// Static dispatch.
template <class T, uint32_t I>
__aicore__ inline void EmitM1(Context &, __ubuf__ float *);
template <class T, uint32_t I>
__aicore__ inline void EmitM1TwoBinary(Context &, __ubuf__ float *);
template <class T, uint32_t I>
__aicore__ inline void EmitM1OneBinary(Context &, __ubuf__ float *);
template <class T, uint32_t I = 0> __aicore__ inline void Run(Context &c) {
  if constexpr (I < T::plan.count) {
    constexpr Step s = T::plan.step[I];
    __ubuf__ float *src = I == 0 ? c.root : (I & 1U ? c.work0 : c.work1);
    if constexpr (s.leaf == Leaf::M1)
      EmitM1<T, I>(c, src);
    else if constexpr (s.leaf == Leaf::Vcmax)
      EmitVcmax<T, I>(c, src);
    else if constexpr (s.leaf == Leaf::M1TwoBinaryReductions)
      EmitM1TwoBinary<T, I>(c, src);
    else if constexpr (s.leaf == Leaf::M1OneBinaryReduction)
      EmitM1OneBinary<T, I>(c, src);
    else if constexpr (s.leaf == Leaf::Columns)
      EmitColumnsLeaf<T, I, StepStatic<T, I>::stageColumns>(c, src);
    else if constexpr (s.tail == Tail::Direct) {
      if constexpr (DirectTailStatic<T, I>::columns) {
        using C = typename DirectTailStatic<T, I>::View;
        EmitColumnsLeaf<C, 0, false>(c, src + DirectTailStatic<T, I>::offset);
      } else
        EmitDirect<T, I>(c, src,
                         std::make_index_sequence<(s.logical % 64 + 7) / 8>{});
    }
    if constexpr (s.leaf == Leaf::None && s.nextLogical) {
      __ubuf__ float *dst = I & 1U ? c.work1 : c.work0;
      EmitBody<T, I>(dst, src);
      EmitRemainder<T, I>(dst, src);
      if constexpr (T::kM == 1)
        WaitForVectorData<T, typename M1StepDelay<T, I>::Type>(c.aux);
      else
        WaitForVectorData<
            T, vector_delay::StaticDelay<
                   vector_delay::Dependency::VcgCompleteToVectorRead, 1>>(
            c.aux);
      Run<T, I + 1>(c);
    }
  }
}
template <class T>
__aicore__ inline void ReduceGeneral(__ubuf__ float *dst, __ubuf__ float *src,
                                     __ubuf__ float *tmp) {
  const uint32_t srcParity =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(src) >> 5) & 1U;
  Context c{dst, src, nullptr, nullptr, tmp + T::layout.auxiliaryOffset};
  if constexpr (T::layout.workElements[0]) {
    const uint32_t pad0 = ((reinterpret_cast<uintptr_t>(tmp) >> 5) & 1U) ^
                          srcParity ^ 1U;
    c.work0 = tmp + pad0 * 8;
  }
  if constexpr (T::layout.workElements[1]) {
    __ubuf__ float *raw1 = tmp + T::layout.work1Offset;
    const uint32_t pad1 = ((reinterpret_cast<uintptr_t>(raw1) >> 5) & 1U) ^
                          srcParity;
    c.work1 = raw1 + pad1 * 8;
  }
  Run<T>(c);
  SetFullMask();
}
template <typename T, Reduce2DKind Kind, bool Clear, uint32_t M, uint32_t N,
          uint32_t SrcRowStride, bool AllowRepeatZero = true>
__aicore__ inline void Reduce2D(const AscendC::LocalTensor<T> &dst,
                                const AscendC::LocalTensor<T> &src,
                                const AscendC::LocalTensor<T> &tmp) {
  using P = Traits<T, Kind, Clear, M, N, SrcRowStride, AllowRepeatZero>;
  ReduceGeneral<P>(reinterpret_cast<__ubuf__ float *>(dst.GetPhyAddr()),
                   reinterpret_cast<__ubuf__ float *>(src.GetPhyAddr()),
                   reinterpret_cast<__ubuf__ float *>(tmp.GetPhyAddr()));
}
#endif // __CCE__
#include "reduce_2d_m1.h"
} // namespace reduce2d_v2
