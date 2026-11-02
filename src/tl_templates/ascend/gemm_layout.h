// Copyright (c) Tile-AI Organization.
// Licensed under the MIT License.

#ifndef TILELANG_TL_TEMPLATES_ASCEND_GEMM_LAYOUT_H_
#define TILELANG_TL_TEMPLATES_ASCEND_GEMM_LAYOUT_H_

#include <cstdint>

// Match Catlass host/aicore annotations when Bisheng parses this shared header.
// Native compiler passes need only the ordinary constexpr implementation.
#if defined(__CCE__)
#define TILELANG_GEMM_LAYOUT_HOST_DEVICE __forceinline__[host, aicore]
#else
#define TILELANG_GEMM_LAYOUT_HOST_DEVICE
#endif

namespace tl::ascend {

// Compile-time geometry shared by the AscendC helper and its host-side audit.
// zZ pads M to 16 and K to C0; nZ pads K to C0 and N to 16. The int8
// transpose-copy instructions additionally read/write whole 32x32 tiles. Slot
// bounds cover those whole instructions, rather than just logical M*K / N*K.
struct GemmV0Layout {
  uint64_t c0;
  uint64_t m_round;
  uint64_t n_round;
  uint64_t l0a_m_quantum;
  uint64_t l0b_n_quantum;
  uint64_t k_stride;
  uint64_t n_tile;
  uint64_t n_tiles;
  uint64_t k_tiles;
  uint64_t steps;
  uint64_t slot_count;
  uint64_t slot_budget;
  uint64_t l0a_elements;
  uint64_t l0b_elements;
  uint64_t l0a_bytes;
  uint64_t l0b_bytes;

  TILELANG_GEMM_LAYOUT_HOST_DEVICE constexpr GemmV0Layout(
      uint32_t m, uint32_t n, uint32_t k, uint32_t k_l0, uint32_t element_bytes,
      bool transpose_b, bool transpose_a = false)
      : c0(element_bytes == 1 || element_bytes == 2 || element_bytes == 4
               ? 32 / element_bytes
               : 0),
        m_round(RoundUp(m, 16)), n_round(RoundUp(n, 16)),
        l0a_m_quantum(element_bytes == 1 && transpose_a ? 32 : 16),
        l0b_n_quantum(element_bytes == 1 && !transpose_b ? 32 : 16),
        k_stride(RoundUp(k_l0, c0)),
        n_tile(SelectNTile(n, k_stride, element_bytes, c0, l0b_n_quantum,
                           transpose_b)),
        n_tiles(n_tile == 0 ? 0 : n / n_tile),
        k_tiles(k_l0 == 0 ? 0 : (uint64_t(k) + k_l0 - 1) / k_l0),
        steps(Product(k_tiles, n_tiles)), slot_count(steps > 1 ? 2 : 1),
        slot_budget(65536 / slot_count),
        l0a_elements(Product(RoundUp(m, l0a_m_quantum), k_stride)),
        l0b_elements(Product(RoundUp(n_tile, l0b_n_quantum), k_stride)),
        l0a_bytes(Product(l0a_elements, element_bytes)),
        l0b_bytes(Product(l0b_elements, element_bytes)) {}

  TILELANG_GEMM_LAYOUT_HOST_DEVICE constexpr bool valid() const {
    return c0 != 0 && m_round != 0 && n_round != 0 && k_stride != 0 &&
           n_tile != 0 && n_tiles != 0 && k_tiles != 0 &&
           l0a_bytes <= slot_budget && l0b_bytes <= slot_budget &&
           l0a_bytes % 512 == 0 && l0b_bytes % 512 == 0;
  }

private:
  TILELANG_GEMM_LAYOUT_HOST_DEVICE static constexpr uint64_t
  RoundUp(uint64_t value, uint64_t quantum) {
    return quantum == 0 ? 0 : (value + quantum - 1) / quantum * quantum;
  }

  // Invalid, extremely large template parameters must fail the capacity check
  // rather than wrap into a seemingly small footprint.
  TILELANG_GEMM_LAYOUT_HOST_DEVICE static constexpr uint64_t
  Product(uint64_t first, uint64_t second) {
    constexpr uint64_t maximum = uint64_t(-1);
    return second != 0 && first > maximum / second ? maximum : first * second;
  }

  TILELANG_GEMM_LAYOUT_HOST_DEVICE static constexpr uint64_t
  SelectNTile(uint64_t n, uint64_t k_stride, uint64_t bytes, uint64_t c0,
              uint64_t n_quantum, bool transpose_b) {
    if (c0 == 0 || k_stride == 0 || n == 0)
      return 0;
    if (transpose_b)
      return n; // A single step may use the whole 64 KiB L0B pool.
    const uint64_t column_bytes = Product(k_stride, bytes);
    if (Product(RoundUp(n, n_quantum), column_bytes) <= 32768)
      return n;
    // Uniform, complete N tiles preserve the helper's existing schedule. A
    // tile start must align to both the L1 C0 and the L0B 16-column fractal.
    const uint64_t alignment = c0 > 16 ? c0 : 16;
    const uint64_t limit = 32768 / column_bytes;
    for (uint64_t tile = limit / alignment * alignment; tile != 0;
         tile -= alignment)
      if (n % tile == 0)
        return tile;
    return 0;
  }
};

} // namespace tl::ascend

#undef TILELANG_GEMM_LAYOUT_HOST_DEVICE

#endif // TILELANG_TL_TEMPLATES_ASCEND_GEMM_LAYOUT_H_
