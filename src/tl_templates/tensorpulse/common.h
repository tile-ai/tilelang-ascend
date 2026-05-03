// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl_templates/tensorpulse/common.h
 * \brief TensorPulse intrinsic templates (V1.0 skeleton).
 *
 * Maps TensorPulse SMC microinstructions (AIACC ISA spec §5.3) to
 * compiler-emitted intrinsic calls. Currently a CPU-fallback stub —
 * real SMC instruction emission lands when the TensorPulse SDK is wired in.
 *
 * Unlike the Ascend equivalent, this header does not (yet) depend on any
 * vendor SDK (no catlass / AscendC counterpart exists for TensorPulse).
 */

#ifndef TILELANG_TL_TEMPLATES_TENSORPULSE_COMMON_H_
#define TILELANG_TL_TEMPLATES_TENSORPULSE_COMMON_H_

#include <cstddef>
#include <cstdint>

namespace tl {
namespace tensorpulse {

// Placeholder elementwise add. Real implementation dispatches to
// SMC INTADD / FPADD / FP16MUL / ... microinstructions per dtype.
template <typename T>
inline void tensorpulse_add(T* dst, const T* src0, const T* src1, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    dst[i] = src0[i] + src1[i];
  }
}

}  // namespace tensorpulse
}  // namespace tl

#endif  // TILELANG_TL_TEMPLATES_TENSORPULSE_COMMON_H_
