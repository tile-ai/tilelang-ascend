// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#ifndef TVM_TL_ASCEND_A2A3_TL_TEMPLATES_INTEGER_MATH_H_
#define TVM_TL_ASCEND_A2A3_TL_TEMPLATES_INTEGER_MATH_H_

#include <type_traits>

namespace tl::ascend {

// Deduce the original C++ arithmetic type, including promoted small integers
// and int32 IR shape variables passed through an int64 kernel ABI. Non-scalar
// types retain their existing whole-value operator interface, not lane
// lowering.
template <typename A, typename B>
#ifdef __CCE_AICORE__
__host__ __aicore__
#endif
    inline constexpr auto floor_div(A a, B b) -> decltype(a / b) {
  using R = decltype(a / b);
  if constexpr (std::is_signed_v<R>) {
    R lhs = a;
    R rhs = b;
    R q = lhs / rhs;
    R r = lhs % rhs;
    // Keep correction branchless: helper control flow can affect CANN
    // auto-sync.
    R correction = static_cast<R>((r != 0) & ((r < 0) != (rhs < 0)));
    return q - correction;
  } else {
    return a / b;
  }
}

template <typename A, typename B>
#ifdef __CCE_AICORE__
__host__ __aicore__
#endif
    inline constexpr auto floor_mod(A a, B b) -> decltype(a % b) {
  using R = decltype(a % b);
  if constexpr (std::is_signed_v<R>) {
    R lhs = a;
    R rhs = b;
    // MIN % -1 is representable although C++ % may overflow its quotient.
    if (rhs == R(-1)) {
      return R(0);
    }
    R r = lhs % rhs;
    R correction = static_cast<R>((r != 0) & ((r < 0) != (rhs < 0)));
    return r + correction * rhs;
  } else {
    return a % b;
  }
}

} // namespace tl::ascend
#endif // TVM_TL_ASCEND_A2A3_TL_TEMPLATES_INTEGER_MATH_H_
