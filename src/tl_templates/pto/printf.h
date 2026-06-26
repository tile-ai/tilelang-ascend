#ifndef TL_TEMPLATES_PTO_PRINTF_H_
#define TL_TEMPLATES_PTO_PRINTF_H_

#include <cstdint>
#include <type_traits>

namespace tl::ascend_pto {

// Common data-type list: each entry is X(CppType, NameString).
// Add or remove types here — all consumers update automatically.
#define TL_PTO_DTYPE_LIST(X)                                                   \
  X(float, "float32")                                                          \
  X(half, "float16")                                                           \
  X(int32_t, "int32")                                                          \
  X(uint32_t, "uint32")                                                        \
  X(int16_t, "int16")                                                          \
  X(uint16_t, "uint16")                                                        \
  X(int8_t, "int8")                                                            \
  X(uint8_t, "uint8")

#if defined(_DEBUG) || defined(__CPU_SIM)

namespace detail {

template <typename T> AICORE inline void PrintScalar(T val, uint32_t col) {
  if (col > 0)
    cce::printf(" ");
  if constexpr (std::is_same_v<T, float>) {
    cce::printf("%8.4f", val);
  } else if constexpr (std::is_same_v<T, half>) {
    cce::printf("%8.4f", static_cast<float>(val));
  } else if constexpr (std::is_signed_v<T> && std::is_integral_v<T>) {
    cce::printf("%8ld", static_cast<long>(val));
  } else if constexpr (std::is_unsigned_v<T> && std::is_integral_v<T>) {
    cce::printf("%8lu", static_cast<unsigned long>(val));
  } else {
    cce::printf("%8d", static_cast<int>(val));
  }
}

template <typename T> AICORE inline const __gm__ char *TypeName() {
  return "unknown";
}
#define TL_PTO_TYPENAME_SPECIALIZE(CppType, NameStr)                           \
  template <> AICORE inline const __gm__ char *TypeName<CppType>() {       \
    return NameStr;                                                            \
  }
TL_PTO_DTYPE_LIST(TL_PTO_TYPENAME_SPECIALIZE)
#undef TL_PTO_TYPENAME_SPECIALIZE

} // namespace detail

template <typename Tile>
AICORE inline void DumpTensor(Tile &src, uint32_t desc, uint32_t dumpSize,
                                  uint8_t dim, const uint32_t shapeInfo[]) {
  pipe_barrier(PIPE_ALL);
  cce::printf("=== DumpTensor [desc=%u] UB tile, dumpSize=%u ===\n", desc,
              dumpSize);
  TPRINT(src);
}

template <typename T>
AICORE inline void DumpTensor(__gm__ T *src, uint32_t desc,
                                  uint32_t dumpSize, uint8_t dim,
                                  const uint32_t shapeInfo[]) {
  if (dumpSize == 0)
    return;
  pipe_barrier(PIPE_ALL);
  cce::printf("=== DumpTensor [desc=%u] GM tensor, dtype=%s, dumpSize=%u, "
              "dim=%u ===\n",
              desc, detail::TypeName<T>(), dumpSize, dim);

  if (dim > 0 && shapeInfo != nullptr) {
    uint32_t cols = shapeInfo[dim - 1];
    if (cols == 0)
      cols = dumpSize;
    uint32_t rows = (dumpSize + cols - 1) / cols;

    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t c = 0; c < cols; ++c) {
        uint32_t idx = r * cols + c;
        if (idx < dumpSize) {
          T val = src[idx];
          detail::PrintScalar<T>(val, c);
        }
      }
      cce::printf("\n");
    }
  } else {
    for (uint32_t i = 0; i < dumpSize; ++i) {
      T val = src[i];
      detail::PrintScalar<T>(val, i % 8);
      if ((i + 1) % 8 == 0 || i == dumpSize - 1) {
        cce::printf("\n");
      }
    }
  }
}

#else // !(_DEBUG || __CPU_SIM)

template <typename Tile>
AICORE inline void DumpTensor(Tile &, uint32_t, uint32_t, uint8_t,
                                  const uint32_t[]) {}

template <typename T>
AICORE inline void DumpTensor(__gm__ T *, uint32_t, uint32_t, uint8_t,
                                  const uint32_t[]) {}

#endif // defined(_DEBUG) || defined(__CPU_SIM)

#undef TL_PTO_DTYPE_LIST

} // namespace tl::ascend_pto

#endif // TL_TEMPLATES_PTO_PRINTF_H_
