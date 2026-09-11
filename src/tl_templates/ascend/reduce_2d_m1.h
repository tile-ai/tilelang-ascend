#pragma once

// Included inside namespace reduce2d_v2 after the common planner/emitter
// definitions. M=1 records a fixed validated policy in Plan; it never enters
// the recursive General cost comparison.
constexpr Plan SolveM1Compact(uint32_t logical, uint32_t pitch, bool merge) {
  if (logical <= 64)
    return Prepend({logical, pitch, 0, 0, Tail::None, Body::EvenEven,
                    logical <= 8 ? Leaf::M1 : Leaf::Vcmax, merge},
                   Done(), logical);
  const uint32_t nextLogical = DivUp(logical, 8),
                 nextPitch = DivUp(nextLogical, 8);
  return Prepend({logical, pitch, nextLogical, nextPitch,
                  logical % 64 ? Tail::Compact : Tail::None, Body::EvenEven,
                  Leaf::None, merge},
                 SolveM1Compact(nextLogical, nextPitch, merge), logical);
}
constexpr Plan SolveM1(uint32_t logical, uint32_t pitch, bool merge) {
  if (logical <= 8)
    return Prepend(
        {logical, pitch, 0, 0, Tail::None, Body::EvenEven, Leaf::M1, merge},
        Done(), logical);
  if (logical <= 64)
    return Prepend(
        {logical, pitch, 0, 0, Tail::None, Body::EvenEven, Leaf::Vcmax, merge},
        Done(), logical);
  if (logical <= 120)
    return Prepend({logical, pitch, 0, 0, Tail::None, Body::EvenEven,
                    Leaf::M1TwoBinaryReductions, merge},
                   Done(), logical);
  if (logical == 128)
    return Prepend({logical, pitch, 0, 0, Tail::None, Body::EvenEven,
                    Leaf::M1OneBinaryReduction, merge},
                   Done(), logical);
  return SolveM1Compact(logical, pitch, merge);
}
template <class T, uint32_t I> struct M1StepDelay {
  inline static constexpr Step step = T::plan.step[I];
  inline static constexpr auto input = vector_delay::M1RecursiveInput(
      step.logical % 64 == 0 || DivUp(step.logical % 64, 8) == 8,
      step.nextLogical);
  using Type =
      vector_delay::StaticDelay<input.dependency, input.naturalDistance>;
};

#if defined(__CCE__)
template <class T, uint32_t I>
__aicore__ inline void EmitM1(Context &c, __ubuf__ float *src) {
  constexpr Step s = T::plan.step[I];
  if constexpr (s.logical == 1) {
    if constexpr (s.merge) {
      SetMask<1>();
      BinaryReduce<T>(c.dst, c.dst, src, 1, 1, 1, 1, 8, 8, 8);
    } else
      copy_ubuf_to_ubuf(c.dst, src, 0, 1, 1, 0, 0);
  } else {
    __ubuf__ float *reduced = s.merge ? c.aux : c.dst;
    SetMask<(uint64_t{1} << s.logical) - 1>();
    GroupReduce<T>(reduced, src, 1, 1, 1, 8);
    if constexpr (s.merge) {
      WaitForVectorData<
          T, vector_delay::StaticDelay<
                 vector_delay::Dependency::VcgPartialToVectorRead, 1>>(c.aux);
      MergeColumn<T, true>(c.dst, reduced);
    }
  }
}
template <class T, uint32_t I, uint32_t Count>
__aicore__ inline void FinishM1Binary(Context &c, __ubuf__ float *input) {
  constexpr Step s = T::plan.step[I];
  __ubuf__ float *reduced = s.merge ? c.aux + 64 : c.dst;
  SetMask<ContinuousMask<Count>::value>();
  WholeReduce<T>(reduced, input, 1, 1, 1, 8);
  if constexpr (s.merge) {
    WaitForVectorData<T, vector_delay::StaticDelay<
                             vector_delay::Dependency::WholeReduceToMerge, 0>>(
        c.aux);
    MergeColumn<T, true>(c.dst, reduced);
  }
}
template <class T, uint32_t I>
__aicore__ inline void EmitM1TwoBinary(Context &c, __ubuf__ float *src) {
  constexpr Step s = T::plan.step[I];
  constexpr uint32_t tail = s.logical - 64;
  __ubuf__ float *tmp = c.aux;
  if constexpr (T::kind == Reduce2DKind::kSum) {
    SetFullMask();
    GroupReduce<T>(tmp, src, 1, 1, 1, 8);
    SetMask<ContinuousMask<tail>::value>();
    GroupReduce<T>(tmp + 8, src + 64, 1, 1, 1, 8);
    WaitForVectorData<T,
                      vector_delay::StaticDelay<
                          vector_delay::Dependency::VcgPartialToVectorRead, 1>>(
        c.aux);
    FinishM1Binary<T, I, 8 + (tail + 7) / 8>(c, tmp);
  } else {
    constexpr uint32_t offset = (tail / 2 / 8) * 8, count = tail - offset;
    constexpr auto delay = vector_delay::M1TwoBinary(count % 8 == 0);
    SetMask<ContinuousMask<count>::value>();
    BinaryReduce<T>(tmp + 32, src + 64, src + 64 + offset, 1, 1, 1, 1, 8, 8, 8);
    if constexpr (T::allowRepeatZero)
      Delay<delay.betweenBinaryReductions>(c.aux);
    SetMask<ContinuousMask<32>::value>();
    BinaryReduce<T>(tmp, src, src + 32, 1, 1, 1, 1, 8, 8, 8);
    WaitForVectorData<
        T, vector_delay::StaticDelay<
               vector_delay::Dependency::M1CompleteBinaryToWholeReduce, 1>>(
        c.aux);
    FinishM1Binary<T, I, 32 + count>(c, tmp);
  }
}
template <class T, uint32_t I>
__aicore__ inline void EmitM1OneBinary(Context &c, __ubuf__ float *src) {
  __ubuf__ float *tmp = c.aux;
  SetFullMask();
  BinaryReduce<T>(tmp, src, src + 64, 1, 1, 1, 1, 8, 8, 8);
  WaitForVectorData<
      T, vector_delay::StaticDelay<
             vector_delay::Dependency::M1CompleteBinaryToWholeReduce, 1>>(
      c.aux);
  FinishM1Binary<T, I, 64>(c, tmp);
}
#endif
