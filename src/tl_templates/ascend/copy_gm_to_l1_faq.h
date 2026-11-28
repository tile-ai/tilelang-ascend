#ifndef TL_ASCEND_COPY_GM_TO_L1_FAQ_H_
#define TL_ASCEND_COPY_GM_TO_L1_FAQ_H_

#include "catlass/gemm/tile/atlasa2/copy_gm_to_l1.hpp"

namespace tl::ascend {

// Copy a [Tokens, Heads, 128] FP16 GM slice into an aligned, full zN[128,128] L1 tile.
// The consumer indexes row head * Tokens + token. The caller owns the MTE2 synchronization.
template <class T, uint32_t TileRows, uint32_t HeadWidth, uint32_t Tokens, uint32_t Heads,
          uint32_t TokenStride>
CATLASS_DEVICE void copy_gm_to_l1_faq(AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                      bool /*need_clear*/) {
  static_assert(std::is_same_v<T, half> && TileRows == 128 && HeadWidth == 128);
  static_assert(Tokens == 32 && Heads == 4 && TokenStride == 2048);
  constexpr uint32_t c0 = Catlass::BYTE_PER_C0 / sizeof(T);

#if !defined(TILELANG_ASCEND_DISABLE_FAQ_BATCHED_COPY)
  if (reinterpret_cast<uint64_t>(src.GetPhyAddr()) % Catlass::BYTE_PER_C0 == 0 &&
      reinterpret_cast<uint64_t>(dst.GetPhyAddr()) % Catlass::BYTE_PER_C0 == 0) {
    auto src_layout = tla::MakeLayout(
        tla::MakeShape(Tokens, Heads, HeadWidth),
        tla::MakeStride(static_cast<int64_t>(TokenStride), static_cast<int64_t>(HeadWidth),
                        static_cast<int64_t>(1)));
    auto dst_layout = tla::MakeLayout<T, Catlass::layout::zN>(TileRows, HeadWidth);
    auto src_tensor = tla::MakeTensor(src, src_layout, Catlass::Arch::PositionGM{});
    auto dst_tensor = tla::MakeTensor(dst, dst_layout, Catlass::Arch::PositionL1{});
    Catlass::Gemm::Tile::TileCopyFAQTla<Catlass::Arch::AtlasA2, decltype(src_tensor),
                                      decltype(dst_tensor)>{}(dst_tensor, src_tensor);
    return;
  }
#endif

  AscendC::Nd2NzParams params{};
  params.ndNum = 1;
  params.nValue = Tokens;
  params.dValue = HeadWidth;
  params.srcDValue = TokenStride;
  params.dstNzNStride = 1;
  params.dstNzC0Stride = TileRows;
  for (uint32_t head = 0; head < Heads; ++head) {
    AscendC::DataCopy(dst[head * Tokens * c0], src[head * HeadWidth], params);
  }
}

}  // namespace tl::ascend

#endif  // TL_ASCEND_COPY_GM_TO_L1_FAQ_H_
