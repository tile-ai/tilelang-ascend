/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMacros.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>

#include "../../../../../third_party/cutlass/examples/88_hopper_fmha/collective/fmha_collective_softmax.hpp"
#include "../../../../../third_party/cutlass/examples/88_hopper_fmha/collective/fmha_common.hpp"

namespace xllm::kernel::cuda::test {
namespace {

// Research-only Hopper path for ragged segment attention.
//
// The stable implementation remains in mtgr_fused_attention_kernel.cu. This
// file is intentionally separate so that SM90A-specific TMA/WGMMA experiments
// can move quickly without destabilizing the current q64/reg-frag kernel.
//
// Initial target shape:
//   - one CTA owns one request/head/q tile
//   - one producer warp issues TMA-style K/V stage movement
//   - one consumer warpgroup owns WGMMA QK and PV
//   - online softmax state stays in registers across the K/V sweep
//   - ragged rules are still lowered to per-row visible_end predicates
//
// The first real implementation should keep this wrapper opt-in until it has
// precision parity and a direct perf comparison with the stable kernel.

constexpr int kHopperWarpSize = 32;
constexpr int kHopperProducerWarps = 1;
constexpr int kHopperConsumerWarps = 4;
constexpr int kHopperWarpsPerBlock =
    kHopperProducerWarps + kHopperConsumerWarps;
constexpr int kHopperThreadsPerBlock = kHopperWarpsPerBlock * kHopperWarpSize;
constexpr float kHopperLog2E = 1.4426950408889634f;
[[maybe_unused]] constexpr float kHopperNegInf = -INFINITY;
using MtgrHopperElement = __nv_bfloat16;
// Research toggle for isolating K/V TMA loads. The current correctness baseline
// keeps both directions enabled.
constexpr bool kHopperTmaDebugLoadKey = true;
constexpr bool kHopperTmaDebugLoadValue = true;
#ifdef MTGR_HOPPER_WGMMA_DIRECT_TMA_KEY
constexpr bool kHopperWgmmaDirectTmaKey = MTGR_HOPPER_WGMMA_DIRECT_TMA_KEY != 0;
#else
// The direct key TMA operand path is the current best-known Hopper setting for
// long odd-random ragged shapes. Keep it on by default for the research build
// so standalone compiles do not silently fall back to the much slower legacy
// staging path.
constexpr bool kHopperWgmmaDirectTmaKey = true;
#endif
#ifdef MTGR_HOPPER_WGMMA_DIRECT_TMA_VALUE
constexpr bool kHopperWgmmaDirectTmaValue =
    MTGR_HOPPER_WGMMA_DIRECT_TMA_VALUE != 0;
#else
constexpr bool kHopperWgmmaDirectTmaValue = true;
#endif
constexpr int kHopperWgmmaQueriesPerCta = 64;
constexpr int kHopperWgmmaKvTile = 64;
constexpr int kHopperWgmmaWarpsPerBlock = kHopperConsumerWarps;
constexpr int kHopperWgmmaTmaWarpsPerBlock =
    kHopperWgmmaWarpsPerBlock + kHopperProducerWarps;
constexpr int kHopperWgmmaThreadsPerBlock =
    kHopperWgmmaWarpsPerBlock * kHopperWarpSize;
constexpr int kHopperWgmmaTmaThreadsPerBlock =
    kHopperWgmmaTmaWarpsPerBlock * kHopperWarpSize;
constexpr int kHopperWgmmaMaxThreadsPerBlock = kHopperWgmmaTmaThreadsPerBlock;

struct MtgrRaggedHopperKernelPlan {
  int queries_per_cta = 64;
  int kv_tile = 128;
  int pipeline_stages = 2;
  int producer_warps = kHopperProducerWarps;
  int consumer_warps = kHopperConsumerWarps;
};

using MtgrHopperWgmmaElement = cute::bfloat16_t;
using MtgrHopperVec128 = uint4;
using MtgrHopperCuteU128 = cute::uint128_t;

template <int N>
__device__ __forceinline__ void mtgr_hopper_cp_async_wait_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

__device__ __forceinline__ void mtgr_hopper_cp_async_commit_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <bool PrefetchL2 = true>
__device__ __forceinline__ void mtgr_hopper_cp_async_load_128b(
    void* smem_ptr,
    const void* gmem_ptr,
    bool pred_guard) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  int src_size = pred_guard ? 16 : 0;
  if constexpr (PrefetchL2) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n"
                 :
                 : "r"(smem_int_ptr), "l"(gmem_ptr), "r"(src_size));
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :
                 : "r"(smem_int_ptr), "l"(gmem_ptr), "r"(src_size));
  }
#else
  if (pred_guard) {
    *reinterpret_cast<MtgrHopperVec128*>(smem_ptr) =
        *reinterpret_cast<const MtgrHopperVec128*>(gmem_ptr);
  } else {
    *reinterpret_cast<MtgrHopperVec128*>(smem_ptr) = make_uint4(0U, 0U, 0U, 0U);
  }
#endif
}

template <int HeadDim, int TmaStages>
struct MtgrHopperCutlassTmaTraits {
  static_assert(HeadDim == 64 || HeadDim == 128);
  static_assert(TmaStages >= 2);
  using Element = cutlass::bfloat16_t;
  static constexpr int Alignment = 16 / sizeof(Element);
  using LayoutQK = cute::tuple<int, cute::_1, int>;
  using LayoutV = cute::tuple<int, cute::_1, int>;
  using TileShapeQK = cute::Shape<cute::Int<kHopperWgmmaQueriesPerCta>,
                                  cute::Int<kHopperWgmmaKvTile>,
                                  cute::Int<HeadDim>>;
  using TileShapePV = decltype(cute::select<0, 2, 1>(TileShapeQK{}));
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using StageCount = cutlass::gemm::collective::StageCount<TmaStages>;

  using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      Element,
      LayoutQK,
      Alignment,
      Element,
      LayoutQK,
      Alignment,
      float,
      TileShapeQK,
      ClusterShape,
      StageCount,
      cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;

  using CollectiveMmaPV = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      Element,
      LayoutQK,
      Alignment,
      Element,
      decltype(cute::select<1, 0, 2>(LayoutV{})),
      Alignment,
      float,
      TileShapePV,
      ClusterShape,
      StageCount,
      cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;

  using SmemLayoutQ = decltype(cutlass::fmha::collective::unstageSmemLayout(
      typename CollectiveMmaQK::SmemLayoutA{},
      cute::Int<1>{}));
  using SmemLayoutK = typename CollectiveMmaQK::SmemLayoutB;
  using SmemLayoutV = typename CollectiveMmaPV::SmemLayoutB;
  using TmaKey = typename CollectiveMmaQK::Params::TMA_B;
  using TmaValue = typename CollectiveMmaPV::Params::TMA_B;
};

template <int HeadDim, int KvTile>
struct MtgrHopperCutlassWgmmaTraits;

template <>
struct MtgrHopperCutlassWgmmaTraits<128, 80> {
  using Element = cutlass::bfloat16_t;
  static constexpr int Alignment = 16 / sizeof(Element);
  using LayoutQK = cute::tuple<int, cute::_1, int>;
  using LayoutV = cute::tuple<int, cute::_1, int>;
  using TileShapeQK = cute::Shape<cute::Int<kHopperWgmmaQueriesPerCta>,
                                  cute::Int<80>,
                                  cute::Int<128>>;
  using TileShapePV = decltype(cute::select<0, 2, 1>(TileShapeQK{}));
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using StageCount = cutlass::gemm::collective::StageCount<2>;

  using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      Element,
      LayoutQK,
      Alignment,
      Element,
      LayoutQK,
      Alignment,
      float,
      TileShapeQK,
      ClusterShape,
      StageCount,
      cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;

  using CollectiveMmaPV = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      Element,
      LayoutQK,
      Alignment,
      Element,
      decltype(cute::select<1, 0, 2>(LayoutV{})),
      Alignment,
      float,
      TileShapePV,
      ClusterShape,
      StageCount,
      cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;

  using SmemLayoutQ = decltype(cutlass::fmha::collective::unstageSmemLayout(
      typename CollectiveMmaQK::SmemLayoutA{},
      cute::Int<1>{}));
  using SmemLayoutK = typename CollectiveMmaQK::SmemLayoutB;
  using SmemLayoutV = typename CollectiveMmaPV::SmemLayoutB;
  using SmemLayoutKStatic =
      decltype(cutlass::fmha::collective::unstageSmemLayout(
          typename CollectiveMmaQK::SmemLayoutB{},
          cute::Int<1>{}));
  using SmemLayoutVStatic =
      decltype(cutlass::fmha::collective::unstageSmemLayout(
          typename CollectiveMmaPV::SmemLayoutB{},
          cute::Int<1>{}));
  using TiledMmaQK = typename CollectiveMmaQK::TiledMma;
  using TiledMmaPV = decltype(cutlass::fmha::collective::convert_to_gmma_rs(
      typename CollectiveMmaPV::TiledMma{}));
};

template <>
struct MtgrHopperCutlassWgmmaTraits<128, 96> {
  using Element = cutlass::bfloat16_t;
  static constexpr int Alignment = 16 / sizeof(Element);
  using LayoutQK = cute::tuple<int, cute::_1, int>;
  using LayoutV = cute::tuple<int, cute::_1, int>;
  using TileShapeQK = cute::Shape<cute::Int<kHopperWgmmaQueriesPerCta>,
                                  cute::Int<96>,
                                  cute::Int<128>>;
  using TileShapePV = decltype(cute::select<0, 2, 1>(TileShapeQK{}));
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using StageCount = cutlass::gemm::collective::StageCount<2>;

  using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      Element,
      LayoutQK,
      Alignment,
      Element,
      LayoutQK,
      Alignment,
      float,
      TileShapeQK,
      ClusterShape,
      StageCount,
      cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;

  using CollectiveMmaPV = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      Element,
      LayoutQK,
      Alignment,
      Element,
      decltype(cute::select<1, 0, 2>(LayoutV{})),
      Alignment,
      float,
      TileShapePV,
      ClusterShape,
      StageCount,
      cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;

  using SmemLayoutQ = decltype(cutlass::fmha::collective::unstageSmemLayout(
      typename CollectiveMmaQK::SmemLayoutA{},
      cute::Int<1>{}));
  using SmemLayoutK = typename CollectiveMmaQK::SmemLayoutB;
  using SmemLayoutV = typename CollectiveMmaPV::SmemLayoutB;
  using SmemLayoutKStatic =
      decltype(cutlass::fmha::collective::unstageSmemLayout(
          typename CollectiveMmaQK::SmemLayoutB{},
          cute::Int<1>{}));
  using SmemLayoutVStatic =
      decltype(cutlass::fmha::collective::unstageSmemLayout(
          typename CollectiveMmaPV::SmemLayoutB{},
          cute::Int<1>{}));
  using TiledMmaQK = typename CollectiveMmaQK::TiledMma;
  using TiledMmaPV = decltype(cutlass::fmha::collective::convert_to_gmma_rs(
      typename CollectiveMmaPV::TiledMma{}));
};

template <int Rows>
constexpr auto mtgr_hopper_select_wgmma_mn_layout_atom() {
  if constexpr (Rows % 64 == 0) {
    return cute::GMMA::Layout_MN_SW128_Atom<MtgrHopperWgmmaElement>{};
  } else if constexpr (Rows % 32 == 0) {
    return cute::GMMA::Layout_MN_SW64_Atom<MtgrHopperWgmmaElement>{};
  } else if constexpr (Rows % 16 == 0) {
    return cute::GMMA::Layout_MN_SW32_Atom<MtgrHopperWgmmaElement>{};
  } else {
    static_assert(Rows % 8 == 0,
                  "WGMMA MN layout rows must be divisible by 8.");
    return cute::GMMA::Layout_MN_INTER_Atom<MtgrHopperWgmmaElement>{};
  }
}

template <int Rows, int Cols, int Stages = 1>
using MtgrHopperWgmmaMnLayout = decltype(cute::tile_to_shape(
    mtgr_hopper_select_wgmma_mn_layout_atom<Rows>(),
    cute::make_shape(cute::Int<Rows>{},
                     cute::Int<Cols>{},
                     cute::Int<Stages>{})));

template <int HeadDim, int KvTile>
struct MtgrHopperWgmmaStaticQueryLayoutSelector {
  using type = MtgrHopperWgmmaMnLayout<kHopperWgmmaQueriesPerCta, HeadDim>;
};

template <>
struct MtgrHopperWgmmaStaticQueryLayoutSelector<128, 96> {
  using type = typename MtgrHopperCutlassWgmmaTraits<128, 96>::SmemLayoutQ;
};

template <>
struct MtgrHopperWgmmaStaticQueryLayoutSelector<128, 80> {
  using type = typename MtgrHopperCutlassWgmmaTraits<128, 80>::SmemLayoutQ;
};

template <int HeadDim, int KvTile>
struct MtgrHopperWgmmaStaticKeyLayoutSelector {
  using type = MtgrHopperWgmmaMnLayout<KvTile, HeadDim>;
};

template <>
struct MtgrHopperWgmmaStaticKeyLayoutSelector<128, 96> {
  using type =
      typename MtgrHopperCutlassWgmmaTraits<128, 96>::SmemLayoutKStatic;
};

template <>
struct MtgrHopperWgmmaStaticKeyLayoutSelector<128, 80> {
  using type =
      typename MtgrHopperCutlassWgmmaTraits<128, 80>::SmemLayoutKStatic;
};

template <int HeadDim, int KvTile>
struct MtgrHopperWgmmaStaticValueLayoutSelector {
  using type = MtgrHopperWgmmaMnLayout<HeadDim, KvTile>;
};

template <>
struct MtgrHopperWgmmaStaticValueLayoutSelector<128, 96> {
  using type =
      typename MtgrHopperCutlassWgmmaTraits<128, 96>::SmemLayoutVStatic;
};

template <>
struct MtgrHopperWgmmaStaticValueLayoutSelector<128, 80> {
  using type =
      typename MtgrHopperCutlassWgmmaTraits<128, 80>::SmemLayoutVStatic;
};

template <int HeadDim, int KvTile, int TmaStages, bool UseTma>
struct MtgrHopperWgmmaQueryLayoutSelector {
  using type = MtgrHopperWgmmaMnLayout<kHopperWgmmaQueriesPerCta, HeadDim>;
};

template <int TmaStages, bool UseTma>
struct MtgrHopperWgmmaQueryLayoutSelector<128, 80, TmaStages, UseTma> {
  using type = typename MtgrHopperCutlassWgmmaTraits<128, 80>::SmemLayoutQ;
};

template <int TmaStages, bool UseTma>
struct MtgrHopperWgmmaQueryLayoutSelector<128, 96, TmaStages, UseTma> {
  using type = typename MtgrHopperCutlassWgmmaTraits<128, 96>::SmemLayoutQ;
};

template <int HeadDim, int KvTile, int TmaStages>
struct MtgrHopperWgmmaQueryLayoutSelector<HeadDim, KvTile, TmaStages, true> {
  using type = std::conditional_t<
      kHopperWgmmaDirectTmaKey && KvTile == 64,
      typename MtgrHopperCutlassTmaTraits<HeadDim, TmaStages>::SmemLayoutQ,
      MtgrHopperWgmmaMnLayout<kHopperWgmmaQueriesPerCta, HeadDim>>;
};

template <int HeadDim, int KvTile, int TmaStages, bool UseTma>
struct MtgrHopperWgmmaKeyLayoutSelector {
  using type = MtgrHopperWgmmaMnLayout<KvTile, HeadDim>;
};

template <int TmaStages, bool UseTma>
struct MtgrHopperWgmmaKeyLayoutSelector<128, 80, TmaStages, UseTma> {
  using type = std::conditional_t<
      UseTma,
      typename MtgrHopperCutlassWgmmaTraits<128, 80>::SmemLayoutK,
      typename MtgrHopperCutlassWgmmaTraits<128, 80>::SmemLayoutKStatic>;
};

template <int TmaStages, bool UseTma>
struct MtgrHopperWgmmaKeyLayoutSelector<128, 96, TmaStages, UseTma> {
  using type = std::conditional_t<
      UseTma,
      typename MtgrHopperCutlassWgmmaTraits<128, 96>::SmemLayoutK,
      typename MtgrHopperCutlassWgmmaTraits<128, 96>::SmemLayoutKStatic>;
};

template <int HeadDim, int KvTile, int TmaStages>
struct MtgrHopperWgmmaKeyLayoutSelector<HeadDim, KvTile, TmaStages, true> {
  using type = std::conditional_t<
      kHopperWgmmaDirectTmaKey && KvTile == 64,
      typename MtgrHopperCutlassTmaTraits<HeadDim, TmaStages>::SmemLayoutK,
      MtgrHopperWgmmaMnLayout<KvTile, HeadDim>>;
};

template <int HeadDim, int KvTile, int TmaStages, bool UseTma>
struct MtgrHopperWgmmaValueLayoutSelector {
  using type = MtgrHopperWgmmaMnLayout<HeadDim, KvTile>;
};

template <int TmaStages, bool UseTma>
struct MtgrHopperWgmmaValueLayoutSelector<128, 80, TmaStages, UseTma> {
  using type = std::conditional_t<
      UseTma,
      typename MtgrHopperCutlassWgmmaTraits<128, 80>::SmemLayoutV,
      typename MtgrHopperCutlassWgmmaTraits<128, 80>::SmemLayoutVStatic>;
};

template <int TmaStages, bool UseTma>
struct MtgrHopperWgmmaValueLayoutSelector<128, 96, TmaStages, UseTma> {
  using type = std::conditional_t<
      UseTma,
      typename MtgrHopperCutlassWgmmaTraits<128, 96>::SmemLayoutV,
      typename MtgrHopperCutlassWgmmaTraits<128, 96>::SmemLayoutVStatic>;
};

template <int HeadDim, int KvTile, int TmaStages>
struct MtgrHopperWgmmaValueLayoutSelector<HeadDim, KvTile, TmaStages, true> {
  using type = std::conditional_t<
      kHopperWgmmaDirectTmaValue && KvTile == 64,
      typename MtgrHopperCutlassTmaTraits<HeadDim, TmaStages>::SmemLayoutV,
      MtgrHopperWgmmaMnLayout<HeadDim, KvTile>>;
};

template <int KvTile>
struct MtgrHopperWgmmaQkAtomSelector;

template <>
struct MtgrHopperWgmmaQkAtomSelector<32> {
  using type = cute::SM90_64x32x16_F32BF16BF16_SS<cute::GMMA::Major::MN,
                                                  cute::GMMA::Major::MN>;
};

template <>
struct MtgrHopperWgmmaQkAtomSelector<64> {
  using type = cute::SM90_64x64x16_F32BF16BF16_SS<cute::GMMA::Major::MN,
                                                  cute::GMMA::Major::MN>;
};

template <>
struct MtgrHopperWgmmaQkAtomSelector<128> {
  using type = cute::SM90_64x128x16_F32BF16BF16_SS<cute::GMMA::Major::MN,
                                                   cute::GMMA::Major::MN>;
};

template <int HeadDim, int KvTile, int TmaStages, bool UseDirectTmaKey>
struct MtgrHopperWgmmaQkMmaSelector {
  static_assert(KvTile == 32 || KvTile == 64 || KvTile == 80 || KvTile == 128);
  using type = decltype(cute::make_tiled_mma(
      typename MtgrHopperWgmmaQkAtomSelector<KvTile>::type{}));
};

template <int TmaStages, bool UseDirectTmaKey>
struct MtgrHopperWgmmaQkMmaSelector<128, 80, TmaStages, UseDirectTmaKey> {
  using type = typename MtgrHopperCutlassWgmmaTraits<128, 80>::TiledMmaQK;
};

template <int TmaStages, bool UseDirectTmaKey>
struct MtgrHopperWgmmaQkMmaSelector<128, 96, TmaStages, UseDirectTmaKey> {
  using type = typename MtgrHopperCutlassWgmmaTraits<128, 96>::TiledMmaQK;
};

template <int HeadDim, int KvTile, int TmaStages>
struct MtgrHopperWgmmaQkMmaSelector<HeadDim, KvTile, TmaStages, true> {
  using type =
      typename MtgrHopperCutlassTmaTraits<HeadDim,
                                          TmaStages>::CollectiveMmaQK::TiledMma;
};

template <int KvTile>
struct MtgrHopperWgmmaPvAtomSelector;

template <>
struct MtgrHopperWgmmaPvAtomSelector<32> {
  using type = cute::SM90_64x32x16_F32BF16BF16_RS<cute::GMMA::Major::K,
                                                  cute::GMMA::Major::MN>;
};

template <>
struct MtgrHopperWgmmaPvAtomSelector<64> {
  using type = cute::SM90_64x64x16_F32BF16BF16_RS<cute::GMMA::Major::K,
                                                  cute::GMMA::Major::MN>;
};

template <>
struct MtgrHopperWgmmaPvAtomSelector<128> {
  using type = cute::SM90_64x128x16_F32BF16BF16_RS<cute::GMMA::Major::K,
                                                   cute::GMMA::Major::MN>;
};

template <int HeadDim, int KvTile, int TmaStages, bool UseDirectTmaValue>
struct MtgrHopperWgmmaPvMmaSelector {
  static_assert(KvTile == 32 || KvTile == 64 || KvTile == 80 || KvTile == 128);
  using type = decltype(cute::make_tiled_mma(
      typename MtgrHopperWgmmaPvAtomSelector<KvTile>::type{}));
};

template <int TmaStages, bool UseDirectTmaValue>
struct MtgrHopperWgmmaPvMmaSelector<128, 80, TmaStages, UseDirectTmaValue> {
  using type = typename MtgrHopperCutlassWgmmaTraits<128, 80>::TiledMmaPV;
};

template <int TmaStages, bool UseDirectTmaValue>
struct MtgrHopperWgmmaPvMmaSelector<128, 96, TmaStages, UseDirectTmaValue> {
  using type = typename MtgrHopperCutlassWgmmaTraits<128, 96>::TiledMmaPV;
};

template <int HeadDim, int KvTile, int TmaStages>
struct MtgrHopperWgmmaPvMmaSelector<HeadDim, KvTile, TmaStages, true> {
  using type = decltype(cutlass::fmha::collective::convert_to_gmma_rs(
      typename MtgrHopperCutlassTmaTraits<HeadDim, TmaStages>::CollectiveMmaPV::
          TiledMma{}));
};

template <int HeadDim, int KvTile>
struct MtgrHopperWgmmaSharedStorage {
  using QueryLayout =
      MtgrHopperWgmmaMnLayout<kHopperWgmmaQueriesPerCta, HeadDim>;
  using KeyLayout = MtgrHopperWgmmaMnLayout<KvTile, HeadDim>;
  using ProbLayout = MtgrHopperWgmmaMnLayout<kHopperWgmmaQueriesPerCta, KvTile>;
  using ValueLayout = MtgrHopperWgmmaMnLayout<HeadDim, KvTile>;

  alignas(128) cute::ArrayEngine<MtgrHopperWgmmaElement,
                                 cute::cosize_v<QueryLayout>> query;
  alignas(128)
      cute::ArrayEngine<MtgrHopperWgmmaElement, cute::cosize_v<KeyLayout>> key;
  alignas(128) cute::ArrayEngine<MtgrHopperWgmmaElement,
                                 cute::cosize_v<ProbLayout>> prob;
  alignas(128) cute::ArrayEngine<MtgrHopperWgmmaElement,
                                 cute::cosize_v<ValueLayout>> value;
  alignas(128) float score[kHopperWgmmaQueriesPerCta * KvTile];
  float row_m[kHopperWgmmaQueriesPerCta];
  float row_d[kHopperWgmmaQueriesPerCta];
  int visible_end[kHopperWgmmaQueriesPerCta];
  int row_valid[kHopperWgmmaQueriesPerCta];
  int has_diag[kHopperWgmmaQueriesPerCta];
  int block_max_visible_end;
};

constexpr size_t mtgr_hopper_align_up(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

template <int HeadDim, int KvTile>
constexpr size_t mtgr_hopper_wgmma_shared_bytes() {
  using QueryLayout =
      typename MtgrHopperWgmmaStaticQueryLayoutSelector<HeadDim, KvTile>::type;
  using KeyLayout =
      typename MtgrHopperWgmmaStaticKeyLayoutSelector<HeadDim, KvTile>::type;
  using ValueLayout =
      typename MtgrHopperWgmmaStaticValueLayoutSelector<HeadDim, KvTile>::type;
  size_t bytes = 0;
  bytes = mtgr_hopper_align_up(bytes, 128);
  bytes += cute::cosize_v<QueryLayout> * sizeof(MtgrHopperWgmmaElement);
  bytes = mtgr_hopper_align_up(bytes, 128);
  bytes += cute::cosize_v<KeyLayout> * sizeof(MtgrHopperWgmmaElement);
  bytes = mtgr_hopper_align_up(bytes, 128);
  bytes += cute::cosize_v<ValueLayout> * sizeof(MtgrHopperWgmmaElement);
  // Online softmax state stays in accumulator/register state, so the older
  // float scratch tile is no longer part of the live shared-memory footprint.
  bytes = mtgr_hopper_align_up(bytes, alignof(int));
  bytes += kHopperWgmmaQueriesPerCta * sizeof(int);
  bytes += kHopperWgmmaQueriesPerCta * sizeof(int);
  bytes += kHopperWgmmaQueriesPerCta * sizeof(int);
  bytes = mtgr_hopper_align_up(bytes, alignof(int64_t));
  bytes += KvTile * sizeof(int64_t);
  bytes = mtgr_hopper_align_up(bytes, alignof(int));
  bytes += sizeof(int);
  bytes = mtgr_hopper_align_up(bytes, alignof(float));
  bytes += kHopperWgmmaQueriesPerCta * sizeof(float);
  return mtgr_hopper_align_up(bytes, 16);
}

struct MtgrHopperWgmmaSoftmaxParams {
  float scale_softmax = 1.0f;
  float scale_softmax_log2 = 1.0f;
  float rp_dropout = 1.0f;
};

constexpr int kMtgrHopperWgmmaTileMaskGeneral = 0;
constexpr int kMtgrHopperWgmmaTileMaskRowColOnly = 1;
constexpr int kMtgrHopperWgmmaTileMaskAllVisible = 2;
struct MtgrHopperWgmmaProblemShape {
  const int* visible_end = nullptr;
  const int* diag_col = nullptr;
  const int* row_valid = nullptr;
  int kv_tile_start = 0;
  int valid_cols = 0;
  int tile_mask_mode = kMtgrHopperWgmmaTileMaskGeneral;
};

struct MtgrHopperWgmmaTmaTilePlan {
  bool eligible = false;
  bool use_cache = false;
  int global_kv_start = 0;
};

template <int KvTile, bool UseUnifiedSources>
__device__ __forceinline__ MtgrHopperWgmmaTmaTilePlan
mtgr_hopper_wgmma_tma_tile_plan(int kv_tile_start,
                                int valid_rows,
                                int matched_prefix,
                                int dense_kv_base,
                                const int32_t* __restrict__ block_table_row,
                                int block_size) {
  MtgrHopperWgmmaTmaTilePlan plan;
  if (valid_rows != KvTile) {
    return plan;
  }
  if constexpr (!UseUnifiedSources) {
    plan.eligible = true;
    plan.global_kv_start = dense_kv_base + kv_tile_start;
    return plan;
  }

  const int prefix_rows = min(max(matched_prefix - kv_tile_start, 0), KvTile);
  if (prefix_rows == KvTile) {
    const int logical_block = kv_tile_start / block_size;
    const int block_offset = kv_tile_start - logical_block * block_size;
    if (block_offset + KvTile <= block_size) {
      const int physical_block =
          static_cast<int>(block_table_row[logical_block]);
      plan.eligible = true;
      plan.use_cache = true;
      plan.global_kv_start = physical_block * block_size + block_offset;
    }
    return plan;
  }
  if (prefix_rows == 0) {
    plan.eligible = true;
    plan.global_kv_start = dense_kv_base + kv_tile_start;
  }
  return plan;
}

struct MtgrHopperWgmmaTileFusion {
  template <class AccQK, class IndexQK, class ProblemShape>
  CUTLASS_DEVICE void before_softmax(AccQK& acc_qk,
                                     const IndexQK& index_qk,
                                     const ProblemShape& problem_shape) {
    if (problem_shape.tile_mask_mode == kMtgrHopperWgmmaTileMaskAllVisible) {
      return;
    }
    const bool row_col_only =
        problem_shape.tile_mask_mode == kMtgrHopperWgmmaTileMaskRowColOnly;
#pragma unroll
    for (int i = 0; i < size(acc_qk); ++i) {
      const auto pos = index_qk(i);
      const int row = static_cast<int>(cute::get<0>(pos));
      const int col = static_cast<int>(cute::get<1>(pos));
      const int global_col = problem_shape.kv_tile_start + col;
      const bool row_is_valid = problem_shape.row_valid[row] != 0;
      const bool col_is_loaded = col < problem_shape.valid_cols;
      const bool prefix_visible =
          row_col_only || global_col < problem_shape.visible_end[row];
      const int diag_col = row_col_only ? -1 : problem_shape.diag_col[row];
      const bool diag_visible = diag_col >= 0 && global_col == diag_col;
      if (!(row_is_valid && col_is_loaded &&
            (prefix_visible || diag_visible))) {
        acc_qk(i) = -INFINITY;
      }
    }
  }
};

template <int HeadDim, int KvTile, int TmaStages>
constexpr size_t mtgr_hopper_wgmma_tma_shared_bytes() {
  constexpr bool kUseDirectTmaKey = kHopperWgmmaDirectTmaKey && KvTile == 64;
  constexpr bool kUseDirectTmaValue =
      kHopperWgmmaDirectTmaValue && KvTile == 64;
  using QueryLayout = typename MtgrHopperWgmmaQueryLayoutSelector<HeadDim,
                                                                  KvTile,
                                                                  TmaStages,
                                                                  true>::type;
  using KeyLayout = typename MtgrHopperWgmmaKeyLayoutSelector<HeadDim,
                                                              KvTile,
                                                              TmaStages,
                                                              true>::type;
  using ValueLayout = typename MtgrHopperWgmmaValueLayoutSelector<HeadDim,
                                                                  KvTile,
                                                                  TmaStages,
                                                                  true>::type;
  size_t bytes = 0;
  bytes = mtgr_hopper_align_up(bytes, 128);
  bytes += cute::cosize_v<QueryLayout> * sizeof(MtgrHopperWgmmaElement);
  bytes = mtgr_hopper_align_up(bytes, 128);
  bytes += cute::cosize_v<KeyLayout> * sizeof(MtgrHopperWgmmaElement);
  bytes = mtgr_hopper_align_up(bytes, 128);
  bytes += cute::cosize_v<ValueLayout> * sizeof(MtgrHopperWgmmaElement);
  bytes = mtgr_hopper_align_up(bytes, alignof(int));
  bytes += kHopperWgmmaQueriesPerCta * sizeof(int);
  bytes += kHopperWgmmaQueriesPerCta * sizeof(int);
  bytes += kHopperWgmmaQueriesPerCta * sizeof(int);
  bytes = mtgr_hopper_align_up(bytes, alignof(int64_t));
  bytes += KvTile * sizeof(int64_t);
  bytes = mtgr_hopper_align_up(bytes, alignof(int));
  bytes += sizeof(int);
  bytes = mtgr_hopper_align_up(bytes, alignof(float));
  bytes += kHopperWgmmaQueriesPerCta * sizeof(float);
  bytes = mtgr_hopper_align_up(bytes, 128);
  if constexpr (kHopperTmaDebugLoadKey && !kUseDirectTmaKey) {
    bytes += TmaStages * KvTile * HeadDim * sizeof(MtgrHopperElement);
  }
  if constexpr (kHopperTmaDebugLoadValue && !kUseDirectTmaValue) {
    bytes += TmaStages * KvTile * HeadDim * sizeof(MtgrHopperElement);
  }
  bytes = mtgr_hopper_align_up(bytes, alignof(uint64_t));
  bytes += TmaStages * sizeof(uint64_t);
  return mtgr_hopper_align_up(bytes, 16);
}

template <int HeadDim, int TmaStages>
constexpr uint32_t mtgr_hopper_direct_key_tma_transaction_bytes() {
  using Traits = MtgrHopperCutlassTmaTraits<HeadDim, TmaStages>;
  return cutlass::bits_to_bytes(
      cute::size<0>(typename Traits::SmemLayoutK{}) *
      cute::size<1>(typename Traits::SmemLayoutK{}) *
      static_cast<uint32_t>(
          cutlass::sizeof_bits<typename Traits::Element>::value));
}

template <int HeadDim, int TmaStages>
constexpr uint32_t mtgr_hopper_direct_value_tma_transaction_bytes() {
  using Traits = MtgrHopperCutlassTmaTraits<HeadDim, TmaStages>;
  return cutlass::bits_to_bytes(
      cute::size<0>(typename Traits::SmemLayoutV{}) *
      cute::size<1>(typename Traits::SmemLayoutV{}) *
      static_cast<uint32_t>(
          cutlass::sizeof_bits<typename Traits::Element>::value));
}

[[maybe_unused]] __device__ __forceinline__ float mtgr_hopper_warp_reduce_sum(
    float value) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, mask);
  }
  return value;
}

[[maybe_unused]] __device__ __forceinline__ float mtgr_hopper_warp_reduce_max(
    float value) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, mask));
  }
  return value;
}

__device__ __forceinline__ float mtgr_hopper_element_to_float(
    MtgrHopperElement value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ MtgrHopperElement
mtgr_hopper_float_to_element(float value) {
  return __float2bfloat16_rn(value);
}

__device__ __forceinline__ float mtgr_hopper_wgmma_element_to_float(
    MtgrHopperWgmmaElement value) {
  return static_cast<float>(value);
}

[[maybe_unused]] __device__ __forceinline__ int mtgr_hopper_find_batch_for_row(
    int q_idx,
    const int32_t* __restrict__ segment_offsets,
    int batch_size,
    int num_segments) {
  const int stride = num_segments + 1;
  int lo = 0;
  int hi = batch_size - 1;
  while (lo <= hi) {
    const int mid = (lo + hi) >> 1;
    const int32_t* offsets_row =
        segment_offsets + static_cast<int64_t>(mid) * stride;
    const int start = offsets_row[0];
    const int end = offsets_row[num_segments];
    if (q_idx < start) {
      hi = mid - 1;
    } else if (q_idx >= end) {
      lo = mid + 1;
    } else {
      return mid;
    }
  }
  return -1;
}

[[maybe_unused]] __device__ __forceinline__ int
mtgr_hopper_visible_end_for_segment(int q_local_idx,
                                    const int32_t* __restrict__ segment_offsets,
                                    const int32_t* __restrict__ segment_rules,
                                    int num_segments,
                                    int* seg_id_out) {
  constexpr int kRuleCausal = 0;
  constexpr int kRuleFull = 1;
  constexpr int kRuleDiagonal = 2;

  const int request_start = segment_offsets[0];
  int seg_id = 0;
#pragma unroll 1
  for (; seg_id < num_segments; ++seg_id) {
    const int seg_end_local = segment_offsets[seg_id + 1] - request_start;
    if (q_local_idx < seg_end_local) {
      break;
    }
  }
  if (seg_id >= num_segments) {
    seg_id = num_segments - 1;
  }
  if (seg_id_out != nullptr) {
    *seg_id_out = seg_id;
  }

  const int seg_start_local = segment_offsets[seg_id] - request_start;
  const int seg_end_local = segment_offsets[seg_id + 1] - request_start;
  const int rule = segment_rules[seg_id];
  if (rule == kRuleFull) {
    return seg_end_local;
  }
  if (rule == kRuleCausal) {
    return q_local_idx + 1;
  }
  if (rule == kRuleDiagonal) {
    return seg_start_local;
  }
  return seg_end_local;
}

template <int HeadDim,
          int KvTile,
          int PipelineStages,
          bool LoadKey,
          bool LoadValue,
          class TmaKey,
          class TmaValue>
__device__ __forceinline__ void mtgr_hopper_tma_issue_kv_tile(
    const TmaKey& tma_key,
    const TmaValue& tma_value,
    uint64_t* __restrict__ tma_barriers,
    MtgrHopperElement* __restrict__ k_stage,
    MtgrHopperElement* __restrict__ v_stage,
    int total_len,
    int num_heads,
    int head_idx,
    int global_kv_start,
    int pipe_stage,
    int pipe_phase,
    int warp_id,
    int lane,
    uint32_t extra_transaction_bytes = 0u) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  static_assert(HeadDim == 64 || HeadDim == 128);
  static_assert(KvTile == 32 || KvTile == 64 || KvTile == 80 || KvTile == 96 ||
                KvTile == 128);
  static_assert(PipelineStages == 1 || PipelineStages == 2 ||
                PipelineStages == 3);
  constexpr uint32_t kSingleTileBytes =
      KvTile * HeadDim * static_cast<uint32_t>(sizeof(MtgrHopperElement));
  constexpr uint32_t kTmaTransactionBytes =
      (LoadKey ? kSingleTileBytes : 0u) + (LoadValue ? kSingleTileBytes : 0u);
  using ProducerBarrier = cutlass::arch::ClusterTransactionBarrier;

  if (warp_id == 0 && lane == 0) {
    if constexpr (kTmaTransactionBytes > 0) {
      ProducerBarrier::arrive_and_expect_tx(
          &tma_barriers[pipe_stage],
          kTmaTransactionBytes + extra_transaction_bytes);
    }

    auto gmem_shape =
        cute::make_shape(cute::Int<HeadDim>{}, total_len, num_heads);
    auto cta_tiler =
        cute::make_shape(cute::Int<HeadDim>{}, cute::Int<KvTile>{});
    auto smem_layout = cute::make_layout(
        cute::make_shape(
            cute::Int<HeadDim>{}, cute::Int<KvTile>{}, cute::Int<1>{}),
        cute::make_stride(cute::Int<1>{},
                          cute::Int<HeadDim>{},
                          cute::Int<HeadDim * KvTile>{}));

    if constexpr (LoadKey) {
      MtgrHopperElement* k_tiles = k_stage;
      auto m_key = cute::domain_offset(
          cute::make_coord(0, global_kv_start),
          tma_key.get_tma_tensor(gmem_shape)(cute::_, cute::_, head_idx));
      auto g_key =
          cute::local_tile(m_key, cta_tiler, cute::make_coord(0, cute::_));
      auto s_key = cute::make_tensor(cute::make_smem_ptr(k_tiles), smem_layout);
      auto [tma_key_src, tma_key_dst] =
          cute::tma_partition(tma_key,
                              cute::Int<0>{},
                              cute::Layout<cute::_1>{},
                              cute::group_modes<0, 2>(s_key),
                              cute::group_modes<0, 2>(g_key));
      constexpr int tile_idx = 0;
      auto key_src_tile = tma_key_src(cute::_, tile_idx);
      auto key_dst_stage = tma_key_dst(cute::_, cute::Int<0>{});
      cute::copy(
          tma_key.with(tma_barriers[pipe_stage]), key_src_tile, key_dst_stage);
    }
    if constexpr (LoadValue) {
      MtgrHopperElement* v_tiles = v_stage;
      auto m_value = cute::domain_offset(
          cute::make_coord(0, global_kv_start),
          tma_value.get_tma_tensor(gmem_shape)(cute::_, cute::_, head_idx));
      auto g_value =
          cute::local_tile(m_value, cta_tiler, cute::make_coord(0, cute::_));
      auto s_value =
          cute::make_tensor(cute::make_smem_ptr(v_tiles), smem_layout);
      auto [tma_value_src, tma_value_dst] =
          cute::tma_partition(tma_value,
                              cute::Int<0>{},
                              cute::Layout<cute::_1>{},
                              cute::group_modes<0, 2>(s_value),
                              cute::group_modes<0, 2>(g_value));
      constexpr int tile_idx = 0;
      auto value_src_tile = tma_value_src(cute::_, tile_idx);
      auto value_dst_stage = tma_value_dst(cute::_, cute::Int<0>{});
      cute::copy(tma_value.with(tma_barriers[pipe_stage]),
                 value_src_tile,
                 value_dst_stage);
    }
  }
#endif

  (void)tma_key;
  (void)tma_value;
  (void)tma_barriers;
  (void)k_stage;
  (void)v_stage;
  (void)total_len;
  (void)num_heads;
  (void)head_idx;
  (void)global_kv_start;
  (void)pipe_stage;
  (void)pipe_phase;
  (void)warp_id;
  (void)lane;
  (void)extra_transaction_bytes;
}

template <int HeadDim,
          int KvTile,
          int PipelineStages,
          bool ArmBarrier,
          class TmaKey>
__device__ __forceinline__ void mtgr_hopper_tma_issue_k_tile_direct(
    const TmaKey& tma_key,
    uint64_t* __restrict__ tma_barriers,
    MtgrHopperWgmmaElement* __restrict__ key_tiles,
    int total_len,
    int num_heads,
    int head_idx,
    int global_kv_start,
    int pipe_stage,
    int pipe_phase,
    int warp_id,
    int lane,
    uint32_t extra_transaction_bytes = 0u) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  static_assert(HeadDim == 64 || HeadDim == 128);
  static_assert(KvTile == 64);
  static_assert(PipelineStages >= 2);
  using ProducerBarrier = cutlass::arch::ClusterTransactionBarrier;
  using Traits = MtgrHopperCutlassTmaTraits<HeadDim, PipelineStages>;
  constexpr uint32_t kTmaTransactionBytes = cutlass::bits_to_bytes(
      cute::size<0>(typename Traits::SmemLayoutK{}) *
      cute::size<1>(typename Traits::SmemLayoutK{}) *
      static_cast<uint32_t>(
          cutlass::sizeof_bits<typename Traits::Element>::value));

  if (warp_id == 0 && lane == 0) {
    if constexpr (ArmBarrier) {
      ProducerBarrier::arrive_and_expect_tx(
          &tma_barriers[pipe_stage],
          kTmaTransactionBytes + extra_transaction_bytes);
    }
    using X = cute::Underscore;
    using ThrMma =
        decltype(typename Traits::CollectiveMmaQK::TiledMma{}.get_slice(0));
    ThrMma mma_qk = typename Traits::CollectiveMmaQK::TiledMma{}.get_slice(0);
    auto gmem_shape =
        cute::make_shape(total_len,
                         cute::Int<HeadDim>{},
                         cute::make_shape(cute::Int<1>{}, num_heads));
    auto m_key = cute::domain_offset(
        cute::make_coord(global_kv_start,
                         cute::Int<0>{},
                         cute::make_coord(cute::Int<0>{}, cute::Int<0>{})),
        tma_key.get_tma_tensor(gmem_shape));
    auto g_key = cute::local_tile(m_key,
                                  typename Traits::TileShapeQK{},
                                  cute::make_coord(cute::_, cute::_, cute::_),
                                  cute::Step<X, cute::_1, cute::_1>{});
    auto tSgKey = mma_qk.partition_B(g_key);
    auto s_key = cute::make_tensor(cute::make_smem_ptr(key_tiles),
                                   typename Traits::SmemLayoutK{});
    auto [tKgKeyAll, tKsKey] =
        cute::tma_partition(tma_key,
                            cute::_0{},
                            cute::make_layout(cute::_1{}),
                            cute::group_modes<0, 2>(s_key),
                            cute::group_modes<0, 3>(tSgKey));
    auto tKgKey = tKgKeyAll(cute::_, cute::_, cute::_0{}, head_idx);
    // `domain_offset(global_kv_start, ...)` already rebases the global tensor
    // to the current KV tile. The partitioned TMA view must therefore always
    // read tile 0 from that rebased tensor.
    constexpr int k_index = 0;
    cute::copy(tma_key.with(tma_barriers[pipe_stage]),
               tKgKey(cute::_, k_index),
               tKsKey(cute::_, pipe_stage));
  }
#endif

  (void)tma_key;
  (void)tma_barriers;
  (void)key_tiles;
  (void)total_len;
  (void)num_heads;
  (void)head_idx;
  (void)global_kv_start;
  (void)pipe_stage;
  (void)pipe_phase;
  (void)warp_id;
  (void)lane;
  (void)extra_transaction_bytes;
}

template <int HeadDim,
          int KvTile,
          int PipelineStages,
          bool ArmBarrier,
          class TmaValue>
__device__ __forceinline__ void mtgr_hopper_tma_issue_v_tile_direct(
    const TmaValue& tma_value,
    uint64_t* __restrict__ tma_barriers,
    MtgrHopperWgmmaElement* __restrict__ value_tiles,
    int total_len,
    int num_heads,
    int head_idx,
    int global_kv_start,
    int pipe_stage,
    int pipe_phase,
    int warp_id,
    int lane,
    uint32_t extra_transaction_bytes = 0u) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  static_assert(HeadDim == 64 || HeadDim == 128);
  static_assert(KvTile == 64);
  static_assert(PipelineStages >= 2);
  using ProducerBarrier = cutlass::arch::ClusterTransactionBarrier;
  using Traits = MtgrHopperCutlassTmaTraits<HeadDim, PipelineStages>;
  constexpr uint32_t kTmaTransactionBytes = cutlass::bits_to_bytes(
      cute::size<0>(typename Traits::SmemLayoutV{}) *
      cute::size<1>(typename Traits::SmemLayoutV{}) *
      static_cast<uint32_t>(
          cutlass::sizeof_bits<typename Traits::Element>::value));

  if (warp_id == 0 && lane == 0) {
    if constexpr (ArmBarrier) {
      ProducerBarrier::arrive_and_expect_tx(
          &tma_barriers[pipe_stage],
          kTmaTransactionBytes + extra_transaction_bytes);
    }
    using X = cute::Underscore;
    using ThrMma =
        decltype(typename Traits::CollectiveMmaPV::TiledMma{}.get_slice(0));
    ThrMma mma_pv = typename Traits::CollectiveMmaPV::TiledMma{}.get_slice(0);
    auto gmem_shape =
        cute::make_shape(cute::Int<HeadDim>{},
                         total_len,
                         cute::make_shape(cute::Int<1>{}, num_heads));
    auto m_value = cute::domain_offset(
        cute::make_coord(cute::Int<0>{},
                         global_kv_start,
                         cute::make_coord(cute::Int<0>{}, cute::Int<0>{})),
        tma_value.get_tma_tensor(gmem_shape));
    auto g_value = cute::local_tile(m_value,
                                    typename Traits::TileShapePV{},
                                    cute::make_coord(cute::_, cute::_, cute::_),
                                    cute::Step<X, cute::_1, cute::_1>{});
    auto tOgValue = mma_pv.partition_B(g_value);
    auto s_value = cute::make_tensor(cute::make_smem_ptr(value_tiles),
                                     typename Traits::SmemLayoutV{});
    auto [tVgValueAll, tVsValue] =
        cute::tma_partition(tma_value,
                            cute::_0{},
                            cute::make_layout(cute::_1{}),
                            cute::group_modes<0, 2>(s_value),
                            cute::group_modes<0, 3>(tOgValue));
    auto tVgValue = tVgValueAll(cute::_, cute::_0{}, cute::_, head_idx);
    constexpr int k_index = 0;
    cute::copy(tma_value.with(tma_barriers[pipe_stage]),
               tVgValue(cute::_, k_index),
               tVsValue(cute::_, pipe_stage));
  }
#endif

  (void)tma_value;
  (void)tma_barriers;
  (void)value_tiles;
  (void)total_len;
  (void)num_heads;
  (void)head_idx;
  (void)global_kv_start;
  (void)pipe_stage;
  (void)pipe_phase;
  (void)warp_id;
  (void)lane;
  (void)extra_transaction_bytes;
}

template <int HeadDim,
          int KvTile,
          int PipelineStages,
          bool LoadKey,
          bool LoadValue>
__device__ __forceinline__ void mtgr_hopper_tma_wait_kv_tile(
    uint64_t* __restrict__ tma_barriers,
    int pipe_stage,
    int pipe_phase) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  static_assert(HeadDim == 64 || HeadDim == 128);
  static_assert(KvTile == 32 || KvTile == 64 || KvTile == 80 || KvTile == 128);
  static_assert(PipelineStages == 1 || PipelineStages == 2 ||
                PipelineStages == 3);
  constexpr uint32_t kSingleTileBytes =
      KvTile * HeadDim * static_cast<uint32_t>(sizeof(MtgrHopperElement));
  constexpr uint32_t kTmaTransactionBytes =
      (LoadKey ? kSingleTileBytes : 0u) + (LoadValue ? kSingleTileBytes : 0u);
  using ProducerBarrier = cutlass::arch::ClusterTransactionBarrier;
  if constexpr (kTmaTransactionBytes > 0) {
    ProducerBarrier::wait(&tma_barriers[pipe_stage],
                          static_cast<uint32_t>(pipe_phase));
  }
#endif

  (void)tma_barriers;
  (void)pipe_stage;
  (void)pipe_phase;
}

template <int HeadDim,
          int KvTile,
          int PipelineStages,
          bool LoadKey,
          bool LoadValue,
          class TmaKey,
          class TmaValue>
__device__ __forceinline__ void mtgr_hopper_tma_load_kv_tile(
    const TmaKey& tma_key,
    const TmaValue& tma_value,
    uint64_t* __restrict__ tma_barriers,
    MtgrHopperElement* __restrict__ k_stage,
    MtgrHopperElement* __restrict__ v_stage,
    int total_len,
    int num_heads,
    int head_idx,
    int global_kv_start,
    int pipe_stage,
    int pipe_phase,
    int warp_id,
    int lane) {
  mtgr_hopper_tma_issue_kv_tile<HeadDim,
                                KvTile,
                                PipelineStages,
                                LoadKey,
                                LoadValue>(tma_key,
                                           tma_value,
                                           tma_barriers,
                                           k_stage,
                                           v_stage,
                                           total_len,
                                           num_heads,
                                           head_idx,
                                           global_kv_start,
                                           pipe_stage,
                                           pipe_phase,
                                           warp_id,
                                           lane);
  mtgr_hopper_tma_wait_kv_tile<HeadDim,
                               KvTile,
                               PipelineStages,
                               LoadKey,
                               LoadValue>(tma_barriers, pipe_stage, pipe_phase);
}

template <int HeadDim, int LaneElems>
__device__ __forceinline__ void mtgr_hopper_accumulate_attention_token_ptr(
    const float (&q_frag)[LaneElems],
    float (&o_frag)[LaneElems],
    const MtgrHopperElement* __restrict__ key_ptr,
    const MtgrHopperElement* __restrict__ value_ptr,
    float sm_scale_log2e,
    int lane,
    float& row_m,
    float& row_d);

template <int HeadDim, int LaneElems>
__device__ __forceinline__ void mtgr_hopper_accumulate_attention_token(
    const float (&q_frag)[LaneElems],
    float (&o_frag)[LaneElems],
    const MtgrHopperElement* __restrict__ key,
    const MtgrHopperElement* __restrict__ value,
    int64_t kv_base,
    float sm_scale_log2e,
    int lane,
    float& row_m,
    float& row_d) {
  mtgr_hopper_accumulate_attention_token_ptr<HeadDim, LaneElems>(
      q_frag,
      o_frag,
      key + kv_base,
      value + kv_base,
      sm_scale_log2e,
      lane,
      row_m,
      row_d);
}

template <int HeadDim, int LaneElems>
__device__ __forceinline__ void mtgr_hopper_accumulate_attention_token_ptr(
    const float (&q_frag)[LaneElems],
    float (&o_frag)[LaneElems],
    const MtgrHopperElement* __restrict__ key_ptr,
    const MtgrHopperElement* __restrict__ value_ptr,
    float sm_scale_log2e,
    int lane,
    float& row_m,
    float& row_d) {
  float dot_lane = 0.0f;
#pragma unroll
  for (int i = 0; i < LaneElems; ++i) {
    dot_lane += q_frag[i] * mtgr_hopper_element_to_float(key_ptr[i]);
  }
  dot_lane = mtgr_hopper_warp_reduce_sum(dot_lane);

  float alpha = 0.0f;
  float beta = 0.0f;
  if (lane == 0) {
    const float score = dot_lane * sm_scale_log2e;
    const float new_m = fmaxf(row_m, score);
    alpha = row_m == kHopperNegInf ? 0.0f : exp2f(row_m - new_m);
    beta = exp2f(score - new_m);
    row_d = row_d * alpha + beta;
    row_m = new_m;
  }
  alpha = __shfl_sync(0xffffffff, alpha, 0);
  beta = __shfl_sync(0xffffffff, beta, 0);
  row_d = __shfl_sync(0xffffffff, row_d, 0);
  row_m = __shfl_sync(0xffffffff, row_m, 0);

#pragma unroll
  for (int i = 0; i < LaneElems; ++i) {
    const float v = mtgr_hopper_element_to_float(value_ptr[i]);
    o_frag[i] = o_frag[i] * alpha + beta * v;
  }
}

template <int HeadDim,
          int KvTile,
          int TmaStages,
          bool UseTma,
          bool DedicatedProducer,
          bool UseUnifiedSources,
          bool AllowMixedRequests,
          class TmaKey,
          class TmaValue>
__global__ void mtgr_ragged_segment_attention_hopper_wgmma_qk_kernel(
    const MtgrHopperElement* __restrict__ query,
    const MtgrHopperElement* __restrict__ key,
    const MtgrHopperElement* __restrict__ value,
    const int32_t* __restrict__ segment_offsets,
    const int32_t* __restrict__ segment_rules,
    const int32_t* __restrict__ q_seq_starts,
    const int32_t* __restrict__ matched_prefix_lens,
    const int32_t* __restrict__ block_table,
    const MtgrHopperElement* __restrict__ key_cache,
    const MtgrHopperElement* __restrict__ value_cache,
    MtgrHopperElement* __restrict__ output,
    CUTLASS_GRID_CONSTANT TmaKey const tma_key,
    CUTLASS_GRID_CONSTANT TmaValue const tma_value,
    CUTLASS_GRID_CONSTANT TmaKey const tma_key_cache,
    CUTLASS_GRID_CONSTANT TmaValue const tma_value_cache,
    int fixed_head_idx,
    int total_len_for_tma,
    int total_cache_len_for_tma,
    int total_live_q,
    int max_request_len,
    int num_heads,
    int num_segments,
    int block_table_stride,
    int block_size,
    float sm_scale_log2e) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  static_assert(HeadDim == 64 || HeadDim == 128);
  static_assert(KvTile == 32 || KvTile == 64 || KvTile == 80 || KvTile == 96 ||
                KvTile == 128);
  static_assert(TmaStages == 1 || TmaStages == 2 || TmaStages == 3);
  constexpr int kLaneElems = HeadDim / kHopperWarpSize;
  static_assert(HeadDim % kHopperWarpSize == 0);
  constexpr bool kUseDirectTmaKey =
      UseTma && TmaStages >= 2 && kHopperWgmmaDirectTmaKey && KvTile == 64;
  constexpr bool kUseDirectTmaValue =
      UseTma && TmaStages >= 2 && kHopperWgmmaDirectTmaValue && KvTile == 64;
  using QueryLayout = typename MtgrHopperWgmmaQueryLayoutSelector<HeadDim,
                                                                  KvTile,
                                                                  TmaStages,
                                                                  UseTma>::type;
  using KeyLayout = typename MtgrHopperWgmmaKeyLayoutSelector<HeadDim,
                                                              KvTile,
                                                              TmaStages,
                                                              UseTma>::type;
  using ValueLayout = typename MtgrHopperWgmmaValueLayoutSelector<HeadDim,
                                                                  KvTile,
                                                                  TmaStages,
                                                                  UseTma>::type;
  using TiledMmaQK =
      typename MtgrHopperWgmmaQkMmaSelector<HeadDim,
                                            KvTile,
                                            TmaStages,
                                            kUseDirectTmaKey>::type;
  using TiledMmaPV =
      typename MtgrHopperWgmmaPvMmaSelector<HeadDim,
                                            KvTile,
                                            TmaStages,
                                            kUseDirectTmaValue>::type;
  constexpr size_t kQueryStorageElems = cute::cosize_v<QueryLayout>;
  constexpr size_t kKeyStorageElems = cute::cosize_v<KeyLayout>;
  constexpr size_t kValueStorageElems = cute::cosize_v<ValueLayout>;
  constexpr bool kUseOldTmaKeyPath =
      UseTma && kHopperTmaDebugLoadKey && !kUseDirectTmaKey;
  constexpr bool kUseOldTmaValuePath =
      UseTma && kHopperTmaDebugLoadValue && !kUseDirectTmaValue;
  constexpr uint32_t kDirectKeyTransactionBytes = [] {
    if constexpr (kUseDirectTmaKey) {
      return mtgr_hopper_direct_key_tma_transaction_bytes<HeadDim, TmaStages>();
    } else {
      return 0u;
    }
  }();
  constexpr uint32_t kDirectValueTransactionBytes = [] {
    if constexpr (kUseDirectTmaValue) {
      return mtgr_hopper_direct_value_tma_transaction_bytes<HeadDim,
                                                            TmaStages>();
    } else {
      return 0u;
    }
  }();
  constexpr bool kCanSkipManualKvStage = kUseDirectTmaKey && kUseDirectTmaValue;
  extern __shared__ __align__(128) uint8_t smem_buf[];
  size_t smem_offset = 0;
  auto* query_tile_storage =
      reinterpret_cast<MtgrHopperWgmmaElement*>(smem_buf + smem_offset);
  smem_offset += kQueryStorageElems * sizeof(MtgrHopperWgmmaElement);
  smem_offset = mtgr_hopper_align_up(smem_offset, 128);
  auto* key_tile_storage =
      reinterpret_cast<MtgrHopperWgmmaElement*>(smem_buf + smem_offset);
  smem_offset += kKeyStorageElems * sizeof(MtgrHopperWgmmaElement);
  smem_offset = mtgr_hopper_align_up(smem_offset, 128);
  auto* value_tile_storage =
      reinterpret_cast<MtgrHopperWgmmaElement*>(smem_buf + smem_offset);
  smem_offset += kValueStorageElems * sizeof(MtgrHopperWgmmaElement);
  smem_offset = mtgr_hopper_align_up(smem_offset, alignof(int));
  auto* visible_end_state = reinterpret_cast<int*>(smem_buf + smem_offset);
  smem_offset += kHopperWgmmaQueriesPerCta * sizeof(int);
  auto* row_valid_state = reinterpret_cast<int*>(smem_buf + smem_offset);
  smem_offset += kHopperWgmmaQueriesPerCta * sizeof(int);
  auto* diag_col_state = reinterpret_cast<int*>(smem_buf + smem_offset);
  smem_offset += kHopperWgmmaQueriesPerCta * sizeof(int);
  smem_offset = mtgr_hopper_align_up(smem_offset, alignof(int64_t));
  auto* kv_row_base_state = reinterpret_cast<int64_t*>(smem_buf + smem_offset);
  smem_offset += KvTile * sizeof(int64_t);
  smem_offset = mtgr_hopper_align_up(smem_offset, alignof(int));
  auto* block_max_visible_end_state =
      reinterpret_cast<int*>(smem_buf + smem_offset);
  smem_offset += sizeof(int);
  smem_offset = mtgr_hopper_align_up(smem_offset, alignof(float));
  auto* row_diag_merge_state = reinterpret_cast<float*>(smem_buf + smem_offset);
  smem_offset += kHopperWgmmaQueriesPerCta * sizeof(float);
  MtgrHopperElement* tma_k_stage = nullptr;
  MtgrHopperElement* tma_v_stage = nullptr;
  uint64_t* tma_barriers = nullptr;
  if constexpr (UseTma) {
    smem_offset = mtgr_hopper_align_up(smem_offset, 128);
    if constexpr (kUseOldTmaKeyPath) {
      tma_k_stage =
          reinterpret_cast<MtgrHopperElement*>(smem_buf + smem_offset);
      smem_offset += TmaStages * KvTile * HeadDim * sizeof(MtgrHopperElement);
    }
    if constexpr (kUseOldTmaValuePath) {
      tma_v_stage =
          reinterpret_cast<MtgrHopperElement*>(smem_buf + smem_offset);
      smem_offset += TmaStages * KvTile * HeadDim * sizeof(MtgrHopperElement);
    }
    tma_barriers = reinterpret_cast<uint64_t*>(smem_buf + smem_offset);
  }

  auto s_query =
      cute::make_tensor(cute::make_smem_ptr(query_tile_storage), QueryLayout{});
  auto s_key =
      cute::make_tensor(cute::make_smem_ptr(key_tile_storage), KeyLayout{});
  auto s_value =
      cute::make_tensor(cute::make_smem_ptr(value_tile_storage), ValueLayout{});
  const int lane = threadIdx.x & (kHopperWarpSize - 1);
  const int warp_id = threadIdx.x >> 5;
  constexpr int kConsumerThreadCount =
      kHopperWgmmaWarpsPerBlock * kHopperWarpSize;
  constexpr bool kHasDedicatedProducerWarp = UseTma && DedicatedProducer;
  const bool is_consumer_thread =
      !kHasDedicatedProducerWarp || threadIdx.x < kConsumerThreadCount;
  const bool is_producer_warp =
      kHasDedicatedProducerWarp && warp_id == kHopperWgmmaWarpsPerBlock;
  const bool is_issue_warp =
      kHasDedicatedProducerWarp ? is_producer_warp : warp_id == 0;
  const int mma_thread_idx = is_consumer_thread ? threadIdx.x : 0;

  TiledMmaQK tiled_mma_qk;
  auto thr_mma_qk = tiled_mma_qk.get_thread_slice(mma_thread_idx);
  auto tCsQ = thr_mma_qk.partition_A(s_query);
  auto tCsK = thr_mma_qk.partition_B(s_key);
  auto tCrQ = thr_mma_qk.make_fragment_A(tCsQ);
  auto tCrK = thr_mma_qk.make_fragment_B(tCsK);
  TiledMmaPV tiled_mma_pv;
  auto thr_mma_pv = tiled_mma_pv.get_thread_slice(mma_thread_idx);
  auto tCsValue = thr_mma_pv.partition_B(s_value);
  auto tCrValue = thr_mma_pv.make_fragment_B(tCsValue);
  auto c_qk = cute::make_identity_tensor(cute::make_shape(
      cute::Int<kHopperWgmmaQueriesPerCta>{}, cute::Int<KvTile>{}));
  auto tPcQK = thr_mma_qk.partition_C(c_qk);
  auto c_output = cute::make_identity_tensor(cute::make_shape(
      cute::Int<kHopperWgmmaQueriesPerCta>{}, cute::Int<HeadDim>{}));
  auto tPcOutput = thr_mma_pv.partition_C(c_output);
  auto acc_pv = cute::partition_fragment_C(
      tiled_mma_pv,
      cute::make_shape(cute::Int<kHopperWgmmaQueriesPerCta>{},
                       cute::Int<HeadDim>{}));
  cute::clear(acc_pv);
  const float sm_scale = sm_scale_log2e / kHopperLog2E;
  MtgrHopperWgmmaSoftmaxParams softmax_params{
      sm_scale,
      sm_scale_log2e,
      1.0f,
  };
  cutlass::fmha::collective::CollectiveSoftmax<float,
                                               MtgrHopperWgmmaTileFusion,
                                               MtgrHopperWgmmaSoftmaxParams>
      softmax{softmax_params};
  auto softmax_state = softmax.init(acc_pv, tiled_mma_pv);

  const auto* query_wgmma =
      reinterpret_cast<const MtgrHopperWgmmaElement*>(query);
  const auto* key_wgmma = reinterpret_cast<const MtgrHopperWgmmaElement*>(key);
  const auto* value_wgmma =
      reinterpret_cast<const MtgrHopperWgmmaElement*>(value);
  const auto* key_cache_wgmma =
      reinterpret_cast<const MtgrHopperWgmmaElement*>(key_cache);
  const auto* value_cache_wgmma =
      reinterpret_cast<const MtgrHopperWgmmaElement*>(value_cache);
  const int batch_idx = static_cast<int>(blockIdx.z);
  const int head_idx = UseTma && fixed_head_idx >= 0
                           ? fixed_head_idx
                           : static_cast<int>(blockIdx.y);
  const int64_t kv_token_stride = static_cast<int64_t>(num_heads) * HeadDim;
  const int64_t head_offset = static_cast<int64_t>(head_idx) * HeadDim;
  const int q_tile_id = static_cast<int>(blockIdx.x);
  const int segment_stride = num_segments + 1;
  const int32_t* offsets_row =
      segment_offsets + static_cast<int64_t>(batch_idx) * segment_stride;
  const int seq_start = offsets_row[0];
  const int q_packed_start =
      UseUnifiedSources ? static_cast<int>(q_seq_starts[batch_idx]) : seq_start;
  const int q_packed_end =
      UseUnifiedSources ? (batch_idx + 1 < static_cast<int>(gridDim.z)
                               ? static_cast<int>(q_seq_starts[batch_idx + 1])
                               : total_live_q)
                        : offsets_row[num_segments];
  const int q_live_len = q_packed_end - q_packed_start;
  const int raw_matched_prefix =
      UseUnifiedSources ? static_cast<int>(matched_prefix_lens[batch_idx]) : 0;
  const bool request_uses_cache =
      UseUnifiedSources && (!AllowMixedRequests || raw_matched_prefix > 0);
  const int matched_prefix = request_uses_cache ? raw_matched_prefix : 0;
  const int q_block_start_live = q_tile_id * kHopperWgmmaQueriesPerCta;
  const int q_block_start_local = matched_prefix + q_block_start_live;
  const bool block_all_query_rows_valid =
      q_block_start_live + kHopperWgmmaQueriesPerCta <= q_live_len;
  const int dense_kv_base =
      UseUnifiedSources ? q_packed_start - matched_prefix : seq_start;
  if (q_block_start_live >= q_live_len) {
    return;
  }
  const int32_t* block_table_row =
      request_uses_cache
          ? block_table + static_cast<int64_t>(batch_idx) * block_table_stride
          : nullptr;

  if (threadIdx.x < kHopperWgmmaQueriesPerCta) {
    const int row_idx = threadIdx.x;
    const int q_local_idx = q_block_start_local + row_idx;
    const bool row_valid = (q_block_start_live + row_idx) < q_live_len;
    int seg_id = 0;
    const int visible_end =
        row_valid ? mtgr_hopper_visible_end_for_segment(q_local_idx,
                                                        offsets_row,
                                                        segment_rules,
                                                        num_segments,
                                                        &seg_id)
                  : 0;
    visible_end_state[row_idx] = visible_end;
    row_valid_state[row_idx] = row_valid ? 1 : 0;
    diag_col_state[row_idx] =
        row_valid && segment_rules[seg_id] == 2 ? q_local_idx : -1;
  }

  for (int flat = threadIdx.x; flat < kHopperWgmmaQueriesPerCta * HeadDim;
       flat += blockDim.x) {
    const int row_idx = flat / HeadDim;
    const int dim = flat - row_idx * HeadDim;
    const bool row_valid = (q_block_start_live + row_idx) < q_live_len;
    MtgrHopperWgmmaElement q_value = MtgrHopperWgmmaElement(0.0f);
    if (row_valid) {
      const int q_idx = q_packed_start + q_block_start_live + row_idx;
      const int64_t q_base =
          (static_cast<int64_t>(q_idx) * num_heads + head_idx) * HeadDim + dim;
      q_value = query_wgmma[q_base];
    }
    s_query(row_idx, dim, 0) = q_value;
  }
  __syncthreads();

  if constexpr (UseTma) {
    if (threadIdx.x == 0) {
      for (int stage = 0; stage < TmaStages; ++stage) {
        cutlass::arch::ClusterTransactionBarrier::init(&tma_barriers[stage], 1);
      }
    }
    cutlass::arch::fence_barrier_init();
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    int block_max_visible_end = 0;
#pragma unroll
    for (int row_idx = 0; row_idx < kHopperWgmmaQueriesPerCta; ++row_idx) {
      block_max_visible_end =
          max(block_max_visible_end, visible_end_state[row_idx]);
    }
    *block_max_visible_end_state = block_max_visible_end;
  }
  __syncthreads();

  const int block_max_visible_end = *block_max_visible_end_state;
  const int live_max_visible_end =
      UseUnifiedSources ? max(block_max_visible_end - matched_prefix, 0)
                        : block_max_visible_end;
  const int prefix_tile_count =
      request_uses_cache ? (matched_prefix + KvTile - 1) / KvTile : 0;
  const int live_tile_count = (live_max_visible_end + KvTile - 1) / KvTile;
  const int block_tile_count =
      UseUnifiedSources ? prefix_tile_count + live_tile_count : live_tile_count;
  auto get_tile_bounds =
      [&](int tile_idx, int* kv_tile_start_out, int* valid_rows_out) {
        if constexpr (UseUnifiedSources) {
          if (tile_idx < prefix_tile_count) {
            const int kv_tile_start = tile_idx * KvTile;
            *kv_tile_start_out = kv_tile_start;
            *valid_rows_out = min(KvTile, matched_prefix - kv_tile_start);
          } else {
            const int live_tile_idx = tile_idx - prefix_tile_count;
            const int live_kv_tile_start = live_tile_idx * KvTile;
            *kv_tile_start_out = matched_prefix + live_kv_tile_start;
            *valid_rows_out =
                min(KvTile, live_max_visible_end - live_kv_tile_start);
          }
        } else {
          const int kv_tile_start = tile_idx * KvTile;
          *kv_tile_start_out = kv_tile_start;
          *valid_rows_out = min(KvTile, live_max_visible_end - kv_tile_start);
        }
      };

  if constexpr (UseTma) {
    if (block_tile_count > 0) {
      int initial_kv_tile_start = 0;
      int initial_valid_rows = 0;
      get_tile_bounds(0, &initial_kv_tile_start, &initial_valid_rows);
      const auto initial_tma_plan =
          mtgr_hopper_wgmma_tma_tile_plan<KvTile, UseUnifiedSources>(
              initial_kv_tile_start,
              initial_valid_rows,
              matched_prefix,
              dense_kv_base,
              block_table_row,
              block_size);
      if (initial_tma_plan.eligible) {
        if (is_issue_warp) {
          const auto& initial_tma_key =
              initial_tma_plan.use_cache ? tma_key_cache : tma_key;
          const auto& initial_tma_value =
              initial_tma_plan.use_cache ? tma_value_cache : tma_value;
          const int initial_total_len = initial_tma_plan.use_cache
                                            ? total_cache_len_for_tma
                                            : total_len_for_tma;
          if constexpr (kUseDirectTmaKey) {
            mtgr_hopper_tma_issue_k_tile_direct<HeadDim,
                                                KvTile,
                                                TmaStages,
                                                !kUseOldTmaKeyPath &&
                                                    !kUseOldTmaValuePath>(
                initial_tma_key,
                tma_barriers,
                key_tile_storage,
                initial_total_len,
                num_heads,
                head_idx,
                initial_tma_plan.global_kv_start,
                0,
                0,
                0,
                lane,
                kDirectValueTransactionBytes);
          }
          if constexpr (kUseDirectTmaValue) {
            mtgr_hopper_tma_issue_v_tile_direct<HeadDim,
                                                KvTile,
                                                TmaStages,
                                                !kUseDirectTmaKey &&
                                                    !kUseOldTmaKeyPath &&
                                                    !kUseOldTmaValuePath>(
                initial_tma_value,
                tma_barriers,
                value_tile_storage,
                initial_total_len,
                num_heads,
                head_idx,
                initial_tma_plan.global_kv_start,
                0,
                0,
                0,
                lane);
          }
          if constexpr (kUseOldTmaKeyPath || kUseOldTmaValuePath) {
            mtgr_hopper_tma_issue_kv_tile<HeadDim,
                                          KvTile,
                                          TmaStages,
                                          kUseOldTmaKeyPath,
                                          kUseOldTmaValuePath>(
                initial_tma_key,
                initial_tma_value,
                tma_barriers,
                tma_k_stage,
                tma_v_stage,
                initial_total_len,
                num_heads,
                head_idx,
                initial_tma_plan.global_kv_start,
                0,
                0,
                0,
                lane,
                kDirectKeyTransactionBytes + kDirectValueTransactionBytes);
          }
        }
      }
    }
  }

  bool softmax_started = false;
  int prefix_fast_logical_block = -1;
  int prefix_fast_physical_block = 0;
  for (int tile_idx = 0; tile_idx < block_tile_count; ++tile_idx) {
    int kv_tile_start = 0;
    int valid_rows = 0;
    get_tile_bounds(tile_idx, &kv_tile_start, &valid_rows);
    const int pipe_stage = tile_idx % TmaStages;
    MtgrHopperElement* current_tma_k_stage = nullptr;
    MtgrHopperElement* current_tma_v_stage = nullptr;
    if constexpr (kUseOldTmaKeyPath) {
      current_tma_k_stage = tma_k_stage + pipe_stage * KvTile * HeadDim;
    }
    if constexpr (kUseOldTmaValuePath) {
      current_tma_v_stage = tma_v_stage + pipe_stage * KvTile * HeadDim;
    }
    const auto current_tma_plan =
        mtgr_hopper_wgmma_tma_tile_plan<KvTile, UseUnifiedSources>(
            kv_tile_start,
            valid_rows,
            matched_prefix,
            dense_kv_base,
            block_table_row,
            block_size);

    bool loaded_k_with_tma = false;
    bool loaded_v_with_tma = false;
    if constexpr (UseTma) {
      if constexpr (TmaStages > 1) {
        const int next_tile_idx = tile_idx + 1;
        if (next_tile_idx < block_tile_count && is_issue_warp) {
          int next_kv_tile_start = 0;
          int next_valid_rows = 0;
          get_tile_bounds(next_tile_idx, &next_kv_tile_start, &next_valid_rows);
          const auto next_tma_plan =
              mtgr_hopper_wgmma_tma_tile_plan<KvTile, UseUnifiedSources>(
                  next_kv_tile_start,
                  next_valid_rows,
                  matched_prefix,
                  dense_kv_base,
                  block_table_row,
                  block_size);
          if (next_tma_plan.eligible) {
            const int next_pipe_stage = next_tile_idx % TmaStages;
            const int next_pipe_phase = (next_tile_idx / TmaStages) & 1;
            MtgrHopperElement* next_tma_k_stage = nullptr;
            MtgrHopperElement* next_tma_v_stage = nullptr;
            if constexpr (kUseOldTmaKeyPath) {
              next_tma_k_stage =
                  tma_k_stage + next_pipe_stage * KvTile * HeadDim;
            }
            if constexpr (kUseOldTmaValuePath) {
              next_tma_v_stage =
                  tma_v_stage + next_pipe_stage * KvTile * HeadDim;
            }
            const auto& next_tma_key =
                next_tma_plan.use_cache ? tma_key_cache : tma_key;
            const auto& next_tma_value =
                next_tma_plan.use_cache ? tma_value_cache : tma_value;
            const int next_total_len = next_tma_plan.use_cache
                                           ? total_cache_len_for_tma
                                           : total_len_for_tma;
            if constexpr (kHopperTmaDebugLoadKey && kUseDirectTmaKey) {
              mtgr_hopper_tma_issue_k_tile_direct<HeadDim,
                                                  KvTile,
                                                  TmaStages,
                                                  !kUseOldTmaKeyPath &&
                                                      !kUseOldTmaValuePath>(
                  next_tma_key,
                  tma_barriers,
                  key_tile_storage,
                  next_total_len,
                  num_heads,
                  head_idx,
                  next_tma_plan.global_kv_start,
                  next_pipe_stage,
                  next_pipe_phase,
                  0,
                  lane,
                  kDirectValueTransactionBytes);
            }
            if constexpr (kHopperTmaDebugLoadValue && kUseDirectTmaValue) {
              mtgr_hopper_tma_issue_v_tile_direct<HeadDim,
                                                  KvTile,
                                                  TmaStages,
                                                  !kUseDirectTmaKey &&
                                                      !kUseOldTmaKeyPath &&
                                                      !kUseOldTmaValuePath>(
                  next_tma_value,
                  tma_barriers,
                  value_tile_storage,
                  next_total_len,
                  num_heads,
                  head_idx,
                  next_tma_plan.global_kv_start,
                  next_pipe_stage,
                  next_pipe_phase,
                  0,
                  lane);
            }
            if constexpr (kUseOldTmaKeyPath || kUseOldTmaValuePath) {
              mtgr_hopper_tma_issue_kv_tile<HeadDim,
                                            KvTile,
                                            TmaStages,
                                            kUseOldTmaKeyPath,
                                            kUseOldTmaValuePath>(
                  next_tma_key,
                  next_tma_value,
                  tma_barriers,
                  next_tma_k_stage,
                  next_tma_v_stage,
                  next_total_len,
                  num_heads,
                  head_idx,
                  next_tma_plan.global_kv_start,
                  next_pipe_stage,
                  next_pipe_phase,
                  0,
                  lane,
                  kDirectKeyTransactionBytes + kDirectValueTransactionBytes);
            }
          }
        }
      }
      if (current_tma_plan.eligible && is_consumer_thread) {
        const int pipe_phase = (tile_idx / TmaStages) & 1;
        mtgr_hopper_tma_wait_kv_tile<HeadDim,
                                     KvTile,
                                     TmaStages,
                                     kHopperTmaDebugLoadKey,
                                     kHopperTmaDebugLoadValue>(
            tma_barriers, pipe_stage, pipe_phase);
        loaded_k_with_tma = kHopperTmaDebugLoadKey;
        loaded_v_with_tma = kHopperTmaDebugLoadValue;
        if (loaded_k_with_tma || loaded_v_with_tma) {
          cutlass::arch::fence_view_async_shared();
        }
      }
    }

    const bool use_direct_tma_tile = loaded_k_with_tma && loaded_v_with_tma;
    const int prefix_rows =
        request_uses_cache
            ? min(max(matched_prefix - kv_tile_start, 0), valid_rows)
            : 0;
    const bool tile_all_prefix = request_uses_cache && prefix_rows == valid_rows;
    const bool tile_all_live = !UseUnifiedSources || prefix_rows == 0;
    const bool prefix_tile_crosses_block =
        tile_all_prefix && block_size > 0 &&
        ((kv_tile_start % block_size) + valid_rows > block_size);
    const bool pure_tile_uses_linear_base =
        tile_all_live || (tile_all_prefix && !prefix_tile_crosses_block);
    int tile_mask_mode = kMtgrHopperWgmmaTileMaskGeneral;
    const bool use_all_visible_prefix_mask =
        tile_all_prefix && valid_rows == KvTile && block_all_query_rows_valid;
    if (use_all_visible_prefix_mask) {
      tile_mask_mode = kMtgrHopperWgmmaTileMaskAllVisible;
    } else if (tile_all_prefix && (HeadDim == 64 || KvTile == 80 ||
                                   KvTile == 96 || KvTile == 128)) {
      // Prefix tiles are fully visible to every live query row in partial mode,
      // so they do not need per-row visible_end/diag checks.
      tile_mask_mode = kMtgrHopperWgmmaTileMaskRowColOnly;
    }
    int64_t pure_tile_base = 0;
    if (tile_all_prefix && !prefix_tile_crosses_block) {
      int logical_block = 0;
      int block_offset = 0;
      int physical_block = 0;
      if constexpr (KvTile == 64) {
        if (block_size == 128) {
          logical_block = tile_idx >> 1;
          block_offset = (tile_idx & 1) * KvTile;
          if (logical_block != prefix_fast_logical_block) {
            prefix_fast_logical_block = logical_block;
            prefix_fast_physical_block =
                static_cast<int>(block_table_row[logical_block]);
          }
          physical_block = prefix_fast_physical_block;
        } else {
          logical_block = kv_tile_start / block_size;
          block_offset = kv_tile_start - logical_block * block_size;
          physical_block = static_cast<int>(block_table_row[logical_block]);
        }
      } else {
        logical_block = kv_tile_start / block_size;
        block_offset = kv_tile_start - logical_block * block_size;
        physical_block = static_cast<int>(block_table_row[logical_block]);
      }
      pure_tile_base =
          (static_cast<int64_t>(physical_block) * block_size + block_offset) *
              kv_token_stride +
          head_offset;
    } else if (tile_all_live) {
      pure_tile_base = static_cast<int64_t>(q_packed_start + kv_tile_start -
                                            matched_prefix) *
                           kv_token_stride +
                       head_offset;
    }
    constexpr int kVecElems =
        sizeof(MtgrHopperVec128) / sizeof(MtgrHopperWgmmaElement);
    static_assert(HeadDim % kVecElems == 0);
    const bool need_row_base_decode =
        ((!loaded_k_with_tma) || (!loaded_v_with_tma)) &&
        !pure_tile_uses_linear_base;
    if (need_row_base_decode) {
      // Hoist block-table traversal and token-index reconstruction out of the
      // inner HeadDim loop so each row decodes its backing KV source once.
      for (int row = threadIdx.x; row < valid_rows; row += blockDim.x) {
        const int kv_local_idx = kv_tile_start + row;
        int64_t token_idx = 0;
        if constexpr (UseUnifiedSources) {
          if (row < prefix_rows) {
            const int logical_block = kv_local_idx / block_size;
            const int block_offset = kv_local_idx - logical_block * block_size;
            const int physical_block =
                static_cast<int>(block_table_row[logical_block]);
            token_idx = static_cast<int64_t>(physical_block) * block_size +
                        block_offset;
          } else {
            token_idx = static_cast<int64_t>(q_packed_start + kv_local_idx -
                                             matched_prefix);
          }
        } else {
          token_idx = static_cast<int64_t>(seq_start + kv_local_idx);
        }
        kv_row_base_state[row] = token_idx * kv_token_stride + head_offset;
      }
    }
    if (need_row_base_decode) {
      __syncthreads();
    }
    if (!is_producer_warp) {
      if constexpr (!kCanSkipManualKvStage) {
        const bool full_kv_tile = valid_rows == KvTile;
        const bool can_vectorize_manual_tile = pure_tile_uses_linear_base &&
                                               !loaded_k_with_tma &&
                                               !loaded_v_with_tma &&
                                               full_kv_tile;
        const bool can_u128_store_manual_tile =
            can_vectorize_manual_tile && KvTile != 80 && KvTile != 96;
        const bool can_vectorize_decoded_tile = !pure_tile_uses_linear_base &&
                                                !loaded_k_with_tma &&
                                                !loaded_v_with_tma &&
                                                full_kv_tile;
        if (can_u128_store_manual_tile) {
          const auto* pure_key_src =
              tile_all_prefix ? key_cache_wgmma : key_wgmma;
          const auto* pure_value_src =
              tile_all_prefix ? value_cache_wgmma : value_wgmma;
          constexpr int kU128Elems =
              sizeof(MtgrHopperCuteU128) / sizeof(MtgrHopperWgmmaElement);
          static_assert(HeadDim % kU128Elems == 0);
          constexpr int kU128Rows = HeadDim / kU128Elems;
          const int kU128Cells = kU128Rows * valid_rows;
          const int64_t kv_token_stride_u128 = kv_token_stride / kU128Elems;
          const auto* pure_key_u128 =
              reinterpret_cast<const MtgrHopperCuteU128*>(pure_key_src +
                                                          pure_tile_base);
          const auto* pure_value_u128 =
              reinterpret_cast<const MtgrHopperCuteU128*>(pure_value_src +
                                                          pure_tile_base);
          if constexpr (!kUseDirectTmaKey && !kUseDirectTmaValue) {
            auto s_key_u128 =
                cute::recast<MtgrHopperCuteU128>(s_key(cute::_, cute::_, 0));
            auto s_value_u128 =
                cute::recast<MtgrHopperCuteU128>(s_value(cute::_, cute::_, 0));
            if (tile_all_prefix || tile_all_live) {
              auto* s_key_u128_ptr = cute::raw_pointer_cast(s_key_u128.data());
              auto* s_value_u128_ptr =
                  cute::raw_pointer_cast(s_value_u128.data());
              for (int flat = threadIdx.x; flat < kU128Cells;
                   flat += kConsumerThreadCount) {
                const int tile_row = flat / kU128Rows;
                const int vec_row = flat - tile_row * kU128Rows;
                const int64_t src_idx =
                    static_cast<int64_t>(tile_row) * kv_token_stride_u128 +
                    vec_row;
                void* key_dst = static_cast<void*>(
                    s_key_u128_ptr + s_key_u128.layout()(vec_row, tile_row));
                void* value_dst = static_cast<void*>(
                    s_value_u128_ptr +
                    s_value_u128.layout()(vec_row, tile_row));
                mtgr_hopper_cp_async_load_128b(
                    key_dst, pure_key_u128 + src_idx, true);
                mtgr_hopper_cp_async_load_128b(
                    value_dst, pure_value_u128 + src_idx, true);
              }
              mtgr_hopper_cp_async_commit_group();
              mtgr_hopper_cp_async_wait_group<0>();
            } else {
              for (int flat = threadIdx.x; flat < kU128Cells;
                   flat += kConsumerThreadCount) {
                const int tile_row = flat / kU128Rows;
                const int vec_row = flat - tile_row * kU128Rows;
                const int64_t kv_base = kv_row_base_state[tile_row];
                if (tile_row < prefix_rows) {
                  const auto* key_src_u128 =
                      reinterpret_cast<const MtgrHopperCuteU128*>(
                          key_cache_wgmma + kv_base);
                  const auto* value_src_u128 =
                      reinterpret_cast<const MtgrHopperCuteU128*>(
                          value_cache_wgmma + kv_base);
                  s_key_u128(vec_row, tile_row) = key_src_u128[vec_row];
                  s_value_u128(vec_row, tile_row) = value_src_u128[vec_row];
                } else {
                  const auto* key_src_u128 =
                      reinterpret_cast<const MtgrHopperCuteU128*>(key_wgmma +
                                                                  kv_base);
                  const auto* value_src_u128 =
                      reinterpret_cast<const MtgrHopperCuteU128*>(value_wgmma +
                                                                  kv_base);
                  s_key_u128(vec_row, tile_row) = key_src_u128[vec_row];
                  s_value_u128(vec_row, tile_row) = value_src_u128[vec_row];
                }
              }
            }
          } else if constexpr (!kUseDirectTmaKey) {
            auto s_key_u128 = cute::recast<MtgrHopperCuteU128>(
                s_key(cute::_, cute::_, pipe_stage));
            auto s_value_u128 =
                cute::recast<MtgrHopperCuteU128>(s_value(cute::_, cute::_, 0));
            for (int flat = threadIdx.x; flat < kU128Cells;
                 flat += kConsumerThreadCount) {
              const int tile_row = flat / kU128Rows;
              const int vec_row = flat - tile_row * kU128Rows;
              const int64_t src_idx =
                  static_cast<int64_t>(tile_row) * kv_token_stride_u128 +
                  vec_row;
              s_key_u128(vec_row, tile_row) = pure_key_u128[src_idx];
              s_value_u128(vec_row, tile_row) = pure_value_u128[src_idx];
            }
          } else if constexpr (!kUseDirectTmaValue) {
            auto s_key_u128 =
                cute::recast<MtgrHopperCuteU128>(s_key(cute::_, cute::_, 0));
            auto s_value_u128 = cute::recast<MtgrHopperCuteU128>(
                s_value(cute::_, cute::_, pipe_stage));
            for (int flat = threadIdx.x; flat < kU128Cells;
                 flat += kConsumerThreadCount) {
              const int tile_row = flat / kU128Rows;
              const int vec_row = flat - tile_row * kU128Rows;
              const int64_t src_idx =
                  static_cast<int64_t>(tile_row) * kv_token_stride_u128 +
                  vec_row;
              s_key_u128(vec_row, tile_row) = pure_key_u128[src_idx];
              s_value_u128(vec_row, tile_row) = pure_value_u128[src_idx];
            }
          } else {
            auto s_key_u128 = cute::recast<MtgrHopperCuteU128>(
                s_key(cute::_, cute::_, pipe_stage));
            auto s_value_u128 = cute::recast<MtgrHopperCuteU128>(
                s_value(cute::_, cute::_, pipe_stage));
            for (int flat = threadIdx.x; flat < kU128Cells;
                 flat += kConsumerThreadCount) {
              const int tile_row = flat / kU128Rows;
              const int vec_row = flat - tile_row * kU128Rows;
              const int64_t src_idx =
                  static_cast<int64_t>(tile_row) * kv_token_stride_u128 +
                  vec_row;
              s_key_u128(vec_row, tile_row) = pure_key_u128[src_idx];
              s_value_u128(vec_row, tile_row) = pure_value_u128[src_idx];
            }
          }
        } else if (can_vectorize_manual_tile) {
          const auto* pure_key_src =
              tile_all_prefix ? key_cache_wgmma : key_wgmma;
          const auto* pure_value_src =
              tile_all_prefix ? value_cache_wgmma : value_wgmma;
          constexpr int kVecsPerRow = HeadDim / kVecElems;
          for (int flat = threadIdx.x; flat < valid_rows * kVecsPerRow;
               flat += kConsumerThreadCount) {
            const int tile_row = flat / kVecsPerRow;
            const int vec_idx = flat - tile_row * kVecsPerRow;
            const int dim = vec_idx * kVecElems;
            const int64_t kv_base =
                pure_tile_base +
                static_cast<int64_t>(tile_row) * kv_token_stride + dim;
            const MtgrHopperVec128 k_vec =
                *reinterpret_cast<const MtgrHopperVec128*>(pure_key_src +
                                                           kv_base);
            const MtgrHopperVec128 v_vec =
                *reinterpret_cast<const MtgrHopperVec128*>(pure_value_src +
                                                           kv_base);
            const auto* k_vals =
                reinterpret_cast<const MtgrHopperWgmmaElement*>(&k_vec);
            const auto* v_vals =
                reinterpret_cast<const MtgrHopperWgmmaElement*>(&v_vec);
#pragma unroll
            for (int i = 0; i < kVecElems; ++i) {
              if constexpr (!kUseDirectTmaKey) {
                s_key(tile_row, dim + i, 0) = k_vals[i];
              } else {
                s_key(tile_row, dim + i, pipe_stage) = k_vals[i];
              }
              if constexpr (!kUseDirectTmaValue) {
                s_value(dim + i, tile_row, 0) = v_vals[i];
              } else {
                s_value(dim + i, tile_row, pipe_stage) = v_vals[i];
              }
            }
          }
        } else if (can_vectorize_decoded_tile) {
          constexpr int kVecsPerRow = HeadDim / kVecElems;
          for (int flat = threadIdx.x; flat < valid_rows * kVecsPerRow;
               flat += kConsumerThreadCount) {
            const int tile_row = flat / kVecsPerRow;
            const int vec_idx = flat - tile_row * kVecsPerRow;
            const int dim = vec_idx * kVecElems;
            const int64_t kv_base = kv_row_base_state[tile_row] + dim;
            const auto* key_src_base =
                tile_all_prefix
                    ? key_cache_wgmma
                    : (tile_all_live ? key_wgmma
                                     : (tile_row < prefix_rows ? key_cache_wgmma
                                                               : key_wgmma));
            const auto* value_src_base =
                tile_all_prefix ? value_cache_wgmma
                                : (tile_all_live ? value_wgmma
                                                 : (tile_row < prefix_rows
                                                        ? value_cache_wgmma
                                                        : value_wgmma));
            const MtgrHopperVec128 k_vec =
                *reinterpret_cast<const MtgrHopperVec128*>(key_src_base +
                                                           kv_base);
            const MtgrHopperVec128 v_vec =
                *reinterpret_cast<const MtgrHopperVec128*>(value_src_base +
                                                           kv_base);
            const auto* k_vals =
                reinterpret_cast<const MtgrHopperWgmmaElement*>(&k_vec);
            const auto* v_vals =
                reinterpret_cast<const MtgrHopperWgmmaElement*>(&v_vec);
#pragma unroll
            for (int i = 0; i < kVecElems; ++i) {
              if constexpr (!kUseDirectTmaKey) {
                s_key(tile_row, dim + i, 0) = k_vals[i];
              } else {
                s_key(tile_row, dim + i, pipe_stage) = k_vals[i];
              }
              if constexpr (!kUseDirectTmaValue) {
                s_value(dim + i, tile_row, 0) = v_vals[i];
              } else {
                s_value(dim + i, tile_row, pipe_stage) = v_vals[i];
              }
            }
          }
        } else {
          for (int flat = threadIdx.x; flat < KvTile * HeadDim;
               flat += kConsumerThreadCount) {
            const int tile_row = flat / HeadDim;
            const int dim = flat - tile_row * HeadDim;
            MtgrHopperWgmmaElement k_value = MtgrHopperWgmmaElement(0.0f);
            MtgrHopperWgmmaElement v_value = MtgrHopperWgmmaElement(0.0f);
            if (tile_row < valid_rows) {
              const int64_t kv_base =
                  pure_tile_uses_linear_base
                      ? pure_tile_base +
                            static_cast<int64_t>(tile_row) * kv_token_stride +
                            dim
                      : kv_row_base_state[tile_row] + dim;
              if (loaded_k_with_tma) {
                if constexpr (!kUseDirectTmaKey) {
                  k_value = reinterpret_cast<MtgrHopperWgmmaElement*>(
                      current_tma_k_stage)[flat];
                }
              } else {
                if (tile_all_prefix) {
                  k_value = key_cache_wgmma[kv_base];
                } else if (tile_all_live) {
                  k_value = key_wgmma[kv_base];
                } else {
                  k_value = tile_row < prefix_rows ? key_cache_wgmma[kv_base]
                                                   : key_wgmma[kv_base];
                }
              }
              if (loaded_v_with_tma) {
                if constexpr (!kUseDirectTmaValue) {
                  v_value = reinterpret_cast<MtgrHopperWgmmaElement*>(
                      current_tma_v_stage)[flat];
                }
              } else {
                if (tile_all_prefix) {
                  v_value = value_cache_wgmma[kv_base];
                } else if (tile_all_live) {
                  v_value = value_wgmma[kv_base];
                } else {
                  v_value = tile_row < prefix_rows ? value_cache_wgmma[kv_base]
                                                   : value_wgmma[kv_base];
                }
              }
            }
            if constexpr (!kUseDirectTmaKey) {
              s_key(tile_row, dim, 0) = k_value;
            } else if (!loaded_k_with_tma) {
              s_key(tile_row, dim, pipe_stage) = k_value;
            }
            if constexpr (!kUseDirectTmaValue) {
              s_value(dim, tile_row, 0) = v_value;
            } else if (!loaded_v_with_tma) {
              s_value(dim, tile_row, pipe_stage) = v_value;
            }
          }
        }
      } else if (!use_direct_tma_tile) {
        const bool full_kv_tile = valid_rows == KvTile;
        const bool can_vectorize_manual_tile = pure_tile_uses_linear_base &&
                                               !loaded_k_with_tma &&
                                               !loaded_v_with_tma &&
                                               full_kv_tile;
        const bool can_u128_store_manual_tile = can_vectorize_manual_tile &&
                                                KvTile != 80 && KvTile != 96;
        const bool can_vectorize_decoded_tile = !pure_tile_uses_linear_base &&
                                                !loaded_k_with_tma &&
                                                !loaded_v_with_tma &&
                                                full_kv_tile;
        if (can_u128_store_manual_tile) {
          const auto* pure_key_src =
              tile_all_prefix ? key_cache_wgmma : key_wgmma;
          const auto* pure_value_src =
              tile_all_prefix ? value_cache_wgmma : value_wgmma;
          constexpr int kU128Elems =
              sizeof(MtgrHopperCuteU128) / sizeof(MtgrHopperWgmmaElement);
          static_assert(HeadDim % kU128Elems == 0);
          constexpr int kU128Rows = HeadDim / kU128Elems;
          constexpr int kU128Cells = kU128Rows * KvTile;
          const int64_t kv_token_stride_u128 = kv_token_stride / kU128Elems;
          const auto* pure_key_u128 =
              reinterpret_cast<const MtgrHopperCuteU128*>(pure_key_src +
                                                          pure_tile_base);
          const auto* pure_value_u128 =
              reinterpret_cast<const MtgrHopperCuteU128*>(pure_value_src +
                                                          pure_tile_base);
          auto s_key_u128 = cute::recast<MtgrHopperCuteU128>(
              s_key(cute::_, cute::_, pipe_stage));
          auto s_value_u128 = cute::recast<MtgrHopperCuteU128>(
              s_value(cute::_, cute::_, pipe_stage));
          for (int flat = threadIdx.x; flat < kU128Cells;
               flat += kConsumerThreadCount) {
            const int tile_row = flat / kU128Rows;
            const int vec_row = flat - tile_row * kU128Rows;
            const int64_t src_idx =
                static_cast<int64_t>(tile_row) * kv_token_stride_u128 + vec_row;
            s_key_u128(vec_row, tile_row) = pure_key_u128[src_idx];
            s_value_u128(vec_row, tile_row) = pure_value_u128[src_idx];
          }
        } else if (can_vectorize_manual_tile) {
          const auto* pure_key_src =
              tile_all_prefix ? key_cache_wgmma : key_wgmma;
          const auto* pure_value_src =
              tile_all_prefix ? value_cache_wgmma : value_wgmma;
          constexpr int kVecsPerRow = HeadDim / kVecElems;
          for (int flat = threadIdx.x; flat < valid_rows * kVecsPerRow;
               flat += kConsumerThreadCount) {
            const int tile_row = flat / kVecsPerRow;
            const int vec_idx = flat - tile_row * kVecsPerRow;
            const int dim = vec_idx * kVecElems;
            const int64_t kv_base =
                pure_tile_base +
                static_cast<int64_t>(tile_row) * kv_token_stride + dim;
            const MtgrHopperVec128 k_vec =
                *reinterpret_cast<const MtgrHopperVec128*>(pure_key_src +
                                                           kv_base);
            const MtgrHopperVec128 v_vec =
                *reinterpret_cast<const MtgrHopperVec128*>(pure_value_src +
                                                           kv_base);
            const auto* k_vals =
                reinterpret_cast<const MtgrHopperWgmmaElement*>(&k_vec);
            const auto* v_vals =
                reinterpret_cast<const MtgrHopperWgmmaElement*>(&v_vec);
#pragma unroll
            for (int i = 0; i < kVecElems; ++i) {
              s_key(tile_row, dim + i, pipe_stage) = k_vals[i];
              s_value(dim + i, tile_row, pipe_stage) = v_vals[i];
            }
          }
        } else if (can_vectorize_decoded_tile) {
          constexpr int kVecsPerRow = HeadDim / kVecElems;
          for (int flat = threadIdx.x; flat < valid_rows * kVecsPerRow;
               flat += kConsumerThreadCount) {
            const int tile_row = flat / kVecsPerRow;
            const int vec_idx = flat - tile_row * kVecsPerRow;
            const int dim = vec_idx * kVecElems;
            const int64_t kv_base = kv_row_base_state[tile_row] + dim;
            const auto* key_src_base =
                tile_all_prefix
                    ? key_cache_wgmma
                    : (tile_all_live ? key_wgmma
                                     : (tile_row < prefix_rows ? key_cache_wgmma
                                                               : key_wgmma));
            const auto* value_src_base =
                tile_all_prefix ? value_cache_wgmma
                                : (tile_all_live ? value_wgmma
                                                 : (tile_row < prefix_rows
                                                        ? value_cache_wgmma
                                                        : value_wgmma));
            const MtgrHopperVec128 k_vec =
                *reinterpret_cast<const MtgrHopperVec128*>(key_src_base +
                                                           kv_base);
            const MtgrHopperVec128 v_vec =
                *reinterpret_cast<const MtgrHopperVec128*>(value_src_base +
                                                           kv_base);
            const auto* k_vals =
                reinterpret_cast<const MtgrHopperWgmmaElement*>(&k_vec);
            const auto* v_vals =
                reinterpret_cast<const MtgrHopperWgmmaElement*>(&v_vec);
#pragma unroll
            for (int i = 0; i < kVecElems; ++i) {
              s_key(tile_row, dim + i, pipe_stage) = k_vals[i];
              s_value(dim + i, tile_row, pipe_stage) = v_vals[i];
            }
          }
        } else {
          for (int flat = threadIdx.x; flat < KvTile * HeadDim;
               flat += kConsumerThreadCount) {
            const int tile_row = flat / HeadDim;
            const int dim = flat - tile_row * HeadDim;
            MtgrHopperWgmmaElement k_value = MtgrHopperWgmmaElement(0.0f);
            MtgrHopperWgmmaElement v_value = MtgrHopperWgmmaElement(0.0f);
            if (tile_row < valid_rows) {
              const int64_t kv_base =
                  pure_tile_uses_linear_base
                      ? pure_tile_base +
                            static_cast<int64_t>(tile_row) * kv_token_stride +
                            dim
                      : kv_row_base_state[tile_row] + dim;
              if (!loaded_k_with_tma) {
                if (tile_all_prefix) {
                  k_value = key_cache_wgmma[kv_base];
                } else if (tile_all_live) {
                  k_value = key_wgmma[kv_base];
                } else {
                  k_value = tile_row < prefix_rows ? key_cache_wgmma[kv_base]
                                                   : key_wgmma[kv_base];
                }
              }
              if (!loaded_v_with_tma) {
                if (tile_all_prefix) {
                  v_value = value_cache_wgmma[kv_base];
                } else if (tile_all_live) {
                  v_value = value_wgmma[kv_base];
                } else {
                  v_value = tile_row < prefix_rows ? value_cache_wgmma[kv_base]
                                                   : value_wgmma[kv_base];
                }
              }
            }
            if (!loaded_k_with_tma) {
              s_key(tile_row, dim, pipe_stage) = k_value;
            }
            if (!loaded_v_with_tma) {
              s_value(dim, tile_row, pipe_stage) = v_value;
            }
          }
        }
      }
    }
    __syncthreads();

    if constexpr (UseTma && TmaStages == 1) {
      if (valid_rows == KvTile) {
        const int next_tile_idx = tile_idx + 1;
        if (next_tile_idx < block_tile_count) {
          int next_kv_tile_start = 0;
          int next_valid_rows = 0;
          get_tile_bounds(next_tile_idx, &next_kv_tile_start, &next_valid_rows);
          if (next_valid_rows == KvTile) {
            MtgrHopperElement* next_tma_k_stage = nullptr;
            MtgrHopperElement* next_tma_v_stage = nullptr;
            if constexpr (kUseOldTmaKeyPath) {
              next_tma_k_stage = tma_k_stage;
            }
            if constexpr (kUseOldTmaValuePath) {
              next_tma_v_stage = tma_v_stage;
            }
            if constexpr (kHopperTmaDebugLoadKey && kUseDirectTmaKey) {
              mtgr_hopper_tma_issue_k_tile_direct<HeadDim,
                                                  KvTile,
                                                  TmaStages,
                                                  !kUseOldTmaKeyPath &&
                                                      !kUseOldTmaValuePath>(
                  tma_key,
                  tma_barriers,
                  key_tile_storage,
                  total_len_for_tma,
                  num_heads,
                  head_idx,
                  seq_start + next_kv_tile_start,
                  0,
                  next_tile_idx & 1,
                  warp_id,
                  lane,
                  kDirectValueTransactionBytes);
            }
            if constexpr (kHopperTmaDebugLoadValue && kUseDirectTmaValue) {
              mtgr_hopper_tma_issue_v_tile_direct<HeadDim,
                                                  KvTile,
                                                  TmaStages,
                                                  !kUseDirectTmaKey &&
                                                      !kUseOldTmaKeyPath &&
                                                      !kUseOldTmaValuePath>(
                  tma_value,
                  tma_barriers,
                  value_tile_storage,
                  total_len_for_tma,
                  num_heads,
                  head_idx,
                  seq_start + next_kv_tile_start,
                  0,
                  next_tile_idx & 1,
                  warp_id,
                  lane);
            }
            if constexpr (kUseOldTmaKeyPath || kUseOldTmaValuePath) {
              mtgr_hopper_tma_issue_kv_tile<HeadDim,
                                            KvTile,
                                            TmaStages,
                                            kUseOldTmaKeyPath,
                                            kUseOldTmaValuePath>(
                  tma_key,
                  tma_value,
                  tma_barriers,
                  next_tma_k_stage,
                  next_tma_v_stage,
                  total_len_for_tma,
                  num_heads,
                  head_idx,
                  seq_start + next_kv_tile_start,
                  0,
                  next_tile_idx & 1,
                  warp_id,
                  lane,
                  kDirectKeyTransactionBytes + kDirectValueTransactionBytes);
            }
          }
        }
      }
    }

    if (is_consumer_thread) {
      auto acc_qk = cute::partition_fragment_C(
          tiled_mma_qk,
          cute::make_shape(cute::Int<kHopperWgmmaQueriesPerCta>{},
                           cute::Int<KvTile>{}));
      cute::clear(acc_qk);
      cute::warpgroup_fence_operand(acc_qk);
      cute::warpgroup_arrive();
      cute::gemm(
          tiled_mma_qk,
          tCrQ(cute::_, cute::_, cute::_, 0),
          tCrK(cute::_, cute::_, cute::_, kUseDirectTmaKey ? pipe_stage : 0),
          acc_qk);
      cute::warpgroup_commit_batch();
      cute::warpgroup_wait<0>();
      cute::warpgroup_fence_operand(acc_qk);
      MtgrHopperWgmmaProblemShape problem_shape{
          visible_end_state,
          diag_col_state,
          row_valid_state,
          kv_tile_start,
          valid_rows,
          tile_mask_mode,
      };
      if (softmax_started) {
        softmax.step(acc_qk,
                     tiled_mma_qk,
                     tPcQK,
                     softmax_state,
                     acc_pv,
                     tiled_mma_pv,
                     problem_shape);
      } else {
        softmax.step(acc_qk, tiled_mma_qk, tPcQK, softmax_state, problem_shape);
        softmax_started = true;
      }
      auto prob_operand =
          cutlass::fmha::collective::make_acc_into_op<MtgrHopperWgmmaElement>(
              acc_qk, typename TiledMmaPV::LayoutA_TV{});
      cute::warpgroup_fence_operand(acc_pv);
      cute::warpgroup_fence_operand(prob_operand);
      cute::warpgroup_arrive();
      cute::gemm(
          tiled_mma_pv,
          prob_operand,
          tCrValue(
              cute::_, cute::_, cute::_, kUseDirectTmaValue ? pipe_stage : 0),
          acc_pv);
      cute::warpgroup_commit_batch();
      cute::warpgroup_wait<0>();
      cute::warpgroup_fence_operand(acc_pv);
    }

    if constexpr (kHasDedicatedProducerWarp) {
      __syncthreads();
    }
  }

  if (is_consumer_thread) {
    auto acc_pv_mn = cute::make_tensor(acc_pv.data(),
                                       cutlass::fmha::collective::layout_acc_mn(
                                           tiled_mma_pv, acc_pv.layout()));
    auto tPcOutputMn =
        cute::make_tensor(tPcOutput.data(),
                          cutlass::fmha::collective::layout_acc_mn(
                              tiled_mma_pv, tPcOutput.layout()));
    if (softmax_started) {
      auto lse = softmax.tail(softmax_state, acc_pv, tiled_mma_pv);
#pragma unroll
      for (int i = 0; i < cute::size<0>(acc_pv_mn); ++i) {
#pragma unroll
        for (int j = 0; j < cute::size<1>(acc_pv_mn); ++j) {
          const int row_idx = cute::get<0>(tPcOutputMn(i, j));
          if (row_idx < kHopperWgmmaQueriesPerCta && row_valid_state[row_idx]) {
            row_diag_merge_state[row_idx] = lse(i);
          }
        }
      }
    } else {
      for (int row_idx = threadIdx.x; row_idx < kHopperWgmmaQueriesPerCta;
           row_idx += kConsumerThreadCount) {
        row_diag_merge_state[row_idx] = -INFINITY;
      }
    }
    __syncthreads();

    for (int row_idx = threadIdx.x; row_idx < kHopperWgmmaQueriesPerCta;
         row_idx += kConsumerThreadCount) {
      float self_scale = 0.0f;
      if (row_valid_state[row_idx] && diag_col_state[row_idx] >= 0) {
        const int q_idx = q_packed_start + q_block_start_live + row_idx;
        const int64_t q_base =
            (static_cast<int64_t>(q_idx) * num_heads + head_idx) * HeadDim;
        float self_dot = 0.0f;
#pragma unroll
        for (int dim = 0; dim < HeadDim; ++dim) {
          self_dot +=
              mtgr_hopper_wgmma_element_to_float(query_wgmma[q_base + dim]) *
              mtgr_hopper_wgmma_element_to_float(key_wgmma[q_base + dim]);
        }
        const float self_lse = self_dot * sm_scale;
        const float prefix_lse = row_diag_merge_state[row_idx];
        if (!::isfinite(prefix_lse)) {
          self_scale = 1.0f;
        } else {
          const float merge_max = fmaxf(prefix_lse, self_lse);
          const float prefix_w = expf(prefix_lse - merge_max);
          const float self_w = expf(self_lse - merge_max);
          self_scale = self_w / (prefix_w + self_w);
        }
      }
      row_diag_merge_state[row_idx] = self_scale;
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < cute::size<0>(acc_pv_mn); ++i) {
#pragma unroll
      for (int j = 0; j < cute::size<1>(acc_pv_mn); ++j) {
        const int row_idx = cute::get<0>(tPcOutputMn(i, j));
        const int dim = cute::get<1>(tPcOutputMn(i, j));
        if (row_idx < kHopperWgmmaQueriesPerCta && dim < HeadDim &&
            row_valid_state[row_idx]) {
          const int q_idx = q_packed_start + q_block_start_live + row_idx;
          const int64_t out_base =
              (static_cast<int64_t>(q_idx) * num_heads + head_idx) * HeadDim +
              dim;
          float out_value = softmax_started ? acc_pv_mn(i, j) : 0.0f;
          if (diag_col_state[row_idx] >= 0) {
            const float self_scale = row_diag_merge_state[row_idx];
            const int64_t diag_base =
                (static_cast<int64_t>(q_idx) * num_heads + head_idx) * HeadDim +
                dim;
            const float diag_value =
                mtgr_hopper_wgmma_element_to_float(value_wgmma[diag_base]);
            out_value =
                out_value * (1.0f - self_scale) + diag_value * self_scale;
          }
          output[out_base] = mtgr_hopper_float_to_element(out_value);
        }
      }
    }
  }
#endif

  (void)query;
  (void)key;
  (void)value;
  (void)segment_offsets;
  (void)segment_rules;
  (void)q_seq_starts;
  (void)matched_prefix_lens;
  (void)block_table;
  (void)key_cache;
  (void)value_cache;
  (void)output;
  (void)tma_key;
  (void)tma_value;
  (void)tma_key_cache;
  (void)tma_value_cache;
  (void)fixed_head_idx;
  (void)total_len_for_tma;
  (void)total_cache_len_for_tma;
  (void)total_live_q;
  (void)max_request_len;
  (void)num_heads;
  (void)num_segments;
  (void)block_table_stride;
  (void)block_size;
  (void)sm_scale_log2e;
}

template <int HeadDim, int TmaStages>
auto mtgr_hopper_make_cutlass_key_tma_descriptor(const torch::Tensor& key_snd,
                                                 int total_len,
                                                 int num_heads);

template <int HeadDim, int TmaStages>
auto mtgr_hopper_make_cutlass_value_tma_descriptor(
    const torch::Tensor& value_snd,
    int total_len,
    int num_heads);

inline int mtgr_hopper_unified_partial_hd128_kv_tile(int num_heads,
                                                     int batch_size,
                                                     int max_live_q_per_batch,
                                                     int matched_prefix_est) {
  const char* env =
      std::getenv("XLLM_MTGR_CUDA_HOPPER_UNIFIED_PARTIAL_HD128_KV_TILE");
  if (env != nullptr && env[0] != '\0') {
    const int forced = std::atoi(env);
    if (forced == 64 || forced == 80 || forced == 96 || forced == 128) {
      return forced;
    }
  }
  (void)num_heads;
  (void)batch_size;
  (void)max_live_q_per_batch;
  (void)matched_prefix_est;
  // KvTile=64 is a useful research knob, but it produced non-finite hd128
  // partial outputs on odd-length single-request cases. Keep the default
  // correctness-safe and use the env override for local experiments.
  return 128;
}

template <int HeadDim, bool AllowMixedRequests>
void launch_mtgr_ragged_segment_attention_hopper_wgmma_qk_unified_kernel(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& segment_rules_i32,
    const torch::Tensor& q_seq_starts_i32,
    const torch::Tensor& matched_prefix_lens_i32,
    const torch::Tensor& block_table_i32,
    int64_t block_size,
    int64_t max_request_len,
    double sm_scale,
    torch::Tensor output_snd) {
  const int total_live_q = static_cast<int>(query_snd.size(0));
  const int num_heads = static_cast<int>(query_snd.size(1));
  const int batch_size = static_cast<int>(segment_offsets_i32.size(0));
  const int num_segments = static_cast<int>(segment_rules_i32.size(0));
  int max_live_q_per_batch = static_cast<int>(max_request_len);
  if (batch_size == 1) {
    max_live_q_per_batch = total_live_q;
  }
  int matched_prefix_est = 0;
  if (batch_size == 1 && block_table_i32.defined() && block_size > 0) {
    const int block_count = static_cast<int>(block_table_i32.size(1));
    matched_prefix_est = max(
        block_count * static_cast<int>(block_size) - max_live_q_per_batch, 0);
  }
  const int q_tiles_per_head =
      (max_live_q_per_batch + kHopperWgmmaQueriesPerCta - 1) /
      kHopperWgmmaQueriesPerCta;
  const dim3 block(kHopperWgmmaThreadsPerBlock);
  const dim3 grid(q_tiles_per_head, num_heads, batch_size);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  auto launch_impl = [&]<int kKvTile>() {
    constexpr size_t kDynSharedBytes =
        mtgr_hopper_wgmma_shared_bytes<HeadDim, kKvTile>();
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        mtgr_ragged_segment_attention_hopper_wgmma_qk_kernel<HeadDim,
                                                             kKvTile,
                                                             1,
                                                             false,
                                                             false,
                                                             true,
                                                             AllowMixedRequests,
                                                             int,
                                                             int>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kDynSharedBytes)));
    mtgr_ragged_segment_attention_hopper_wgmma_qk_kernel<HeadDim,
                                                         kKvTile,
                                                         1,
                                                         false,
                                                         false,
                                                         true,
                                                         AllowMixedRequests,
                                                         int,
                                                         int>
        <<<grid, block, kDynSharedBytes, stream>>>(
            reinterpret_cast<const MtgrHopperElement*>(
                query_snd.data_ptr<at::BFloat16>()),
            reinterpret_cast<const MtgrHopperElement*>(
                key_snd.data_ptr<at::BFloat16>()),
            reinterpret_cast<const MtgrHopperElement*>(
                value_snd.data_ptr<at::BFloat16>()),
            segment_offsets_i32.data_ptr<int32_t>(),
            segment_rules_i32.data_ptr<int32_t>(),
            q_seq_starts_i32.data_ptr<int32_t>(),
            matched_prefix_lens_i32.data_ptr<int32_t>(),
            block_table_i32.data_ptr<int32_t>(),
            reinterpret_cast<const MtgrHopperElement*>(
                key_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<const MtgrHopperElement*>(
                value_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<MtgrHopperElement*>(
                output_snd.data_ptr<at::BFloat16>()),
            0,
            0,
            0,
            0,
            -1,
            0,
            0,
            total_live_q,
            static_cast<int>(max_request_len),
            num_heads,
            num_segments,
            static_cast<int>(block_table_i32.size(1)),
            static_cast<int>(block_size),
            static_cast<float>(sm_scale) * kHopperLog2E);
  };
  if constexpr (HeadDim == 64) {
    launch_impl.template operator()<64>();
  } else {
    const int hd128_kv_tile = mtgr_hopper_unified_partial_hd128_kv_tile(
        num_heads, batch_size, max_live_q_per_batch, matched_prefix_est);
    if (hd128_kv_tile == 128) {
      launch_impl.template operator()<128>();
    } else if (hd128_kv_tile == 80) {
      launch_impl.template operator()<80>();
    } else if (hd128_kv_tile == 96) {
      launch_impl.template operator()<96>();
    } else {
      launch_impl.template operator()<64>();
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int HeadDim, int TmaStages>
auto mtgr_hopper_make_cutlass_key_tma_descriptor(const torch::Tensor& key_snd,
                                                 int total_len,
                                                 int num_heads) {
  using Traits = MtgrHopperCutlassTmaTraits<HeadDim, TmaStages>;
  auto key_shape =
      cute::make_shape(total_len,
                       cute::Int<HeadDim>{},
                       cute::make_shape(cute::Int<1>{}, num_heads));
  auto key_stride =
      cute::make_stride(num_heads * HeadDim,
                        cute::Int<1>{},
                        cute::make_stride(cute::Int<0>{}, HeadDim));
  auto key_tensor =
      cute::make_tensor(reinterpret_cast<const typename Traits::Element*>(
                            key_snd.data_ptr<at::BFloat16>()),
                        cute::make_layout(key_shape, key_stride));
  return cute::make_tma_copy_B_sm90(
      cute::SM90_TMA_LOAD{},
      key_tensor,
      typename Traits::SmemLayoutK{}(cute::_, cute::_, cute::Int<0>{}),
      typename Traits::TileShapeQK{},
      typename Traits::ClusterShape{});
}

template <int HeadDim, int TmaStages>
auto mtgr_hopper_make_cutlass_value_tma_descriptor(
    const torch::Tensor& value_snd,
    int total_len,
    int num_heads) {
  using Traits = MtgrHopperCutlassTmaTraits<HeadDim, TmaStages>;
  auto value_shape =
      cute::make_shape(cute::Int<HeadDim>{},
                       total_len,
                       cute::make_shape(cute::Int<1>{}, num_heads));
  auto value_stride =
      cute::make_stride(cute::Int<1>{},
                        num_heads * HeadDim,
                        cute::make_stride(cute::Int<0>{}, HeadDim));
  auto value_tensor =
      cute::make_tensor(reinterpret_cast<const typename Traits::Element*>(
                            value_snd.data_ptr<at::BFloat16>()),
                        cute::make_layout(value_shape, value_stride));
  return cute::make_tma_copy_B_sm90(
      cute::SM90_TMA_LOAD{},
      value_tensor,
      typename Traits::SmemLayoutV{}(cute::_, cute::_, cute::Int<0>{}),
      typename Traits::TileShapePV{},
      typename Traits::ClusterShape{});
}

template <int HeadDim>
void launch_mtgr_ragged_segment_attention_hopper_wgmma_tma_qk_kernel(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& segment_rules_i32,
    int64_t max_request_len,
    double sm_scale,
    torch::Tensor output_snd) {
  constexpr int kKvTile = kHopperWgmmaKvTile;
  constexpr int kTmaStages = 2;
  constexpr bool kUseDirectTmaKey = kHopperWgmmaDirectTmaKey;
  constexpr bool kUseDirectTmaValue = kHopperWgmmaDirectTmaValue;
  constexpr size_t kDynSharedBytes =
      mtgr_hopper_wgmma_tma_shared_bytes<HeadDim, kKvTile, kTmaStages>();
  const int total_len = static_cast<int>(query_snd.size(0));
  const int num_heads = static_cast<int>(query_snd.size(1));
  const int batch_size = static_cast<int>(segment_offsets_i32.size(0));
  const int num_segments = static_cast<int>(segment_rules_i32.size(0));
  const int q_tiles_per_head =
      (static_cast<int>(max_request_len) + kHopperWgmmaQueriesPerCta - 1) /
      kHopperWgmmaQueriesPerCta;
  const bool use_producer_warp = true;
  const dim3 block(use_producer_warp ? kHopperWgmmaTmaThreadsPerBlock
                                     : kHopperWgmmaThreadsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  auto gmem_shape =
      cute::make_shape(cute::Int<HeadDim>{}, total_len, num_heads);
  auto gmem_stride =
      cute::make_stride(cute::Int<1>{}, num_heads * HeadDim, HeadDim);
  auto smem_layout = cute::make_layout(
      cute::make_shape(cute::Int<HeadDim>{}, cute::Int<kKvTile>{}),
      cute::make_stride(cute::Int<1>{}, cute::Int<HeadDim>{}));
  auto cta_tiler = cute::make_shape(cute::Int<HeadDim>{}, cute::Int<kKvTile>{});
  auto key_matrix =
      cute::make_tensor(reinterpret_cast<const MtgrHopperElement*>(
                            key_snd.data_ptr<at::BFloat16>()),
                        gmem_shape,
                        gmem_stride);
  auto value_matrix =
      cute::make_tensor(reinterpret_cast<const MtgrHopperElement*>(
                            value_snd.data_ptr<at::BFloat16>()),
                        gmem_shape,
                        gmem_stride);
  auto tma_key = [&]() {
    if constexpr (kUseDirectTmaKey) {
      return mtgr_hopper_make_cutlass_key_tma_descriptor<HeadDim, kTmaStages>(
          key_snd, total_len, num_heads);
    } else {
      return cute::make_tma_atom<cute::bfloat16_t>(
          cute::SM90_TMA_LOAD{}, key_matrix, smem_layout, cta_tiler);
    }
  }();
  auto tma_value = [&]() {
    if constexpr (kUseDirectTmaValue) {
      return mtgr_hopper_make_cutlass_value_tma_descriptor<HeadDim, kTmaStages>(
          value_snd, total_len, num_heads);
    } else {
      return cute::make_tma_atom<cute::bfloat16_t>(
          cute::SM90_TMA_LOAD{}, value_matrix, smem_layout, cta_tiler);
    }
  }();
  const dim3 grid(q_tiles_per_head, num_heads, batch_size);

  if (use_producer_warp) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        mtgr_ragged_segment_attention_hopper_wgmma_qk_kernel<
            HeadDim,
            kKvTile,
            kTmaStages,
            true,
            true,
            false,
            false,
            decltype(tma_key),
            decltype(tma_value)>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kDynSharedBytes)));
    mtgr_ragged_segment_attention_hopper_wgmma_qk_kernel<HeadDim,
                                                         kKvTile,
                                                         kTmaStages,
                                                         true,
                                                         true,
                                                         false,
                                                         false,
                                                         decltype(tma_key),
                                                         decltype(tma_value)>
        <<<grid, block, kDynSharedBytes, stream>>>(
            reinterpret_cast<const MtgrHopperElement*>(
                query_snd.data_ptr<at::BFloat16>()),
            reinterpret_cast<const MtgrHopperElement*>(
                key_snd.data_ptr<at::BFloat16>()),
            reinterpret_cast<const MtgrHopperElement*>(
                value_snd.data_ptr<at::BFloat16>()),
            segment_offsets_i32.data_ptr<int32_t>(),
            segment_rules_i32.data_ptr<int32_t>(),
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            reinterpret_cast<MtgrHopperElement*>(
                output_snd.data_ptr<at::BFloat16>()),
            tma_key,
            tma_value,
            tma_key,
            tma_value,
            -1,
            total_len,
            total_len,
            0,
            static_cast<int>(max_request_len),
            num_heads,
            num_segments,
            0,
            0,
            static_cast<float>(sm_scale) * kHopperLog2E);
  } else {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        mtgr_ragged_segment_attention_hopper_wgmma_qk_kernel<
            HeadDim,
            kKvTile,
            kTmaStages,
            true,
            false,
            false,
            false,
            decltype(tma_key),
            decltype(tma_value)>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kDynSharedBytes)));
    mtgr_ragged_segment_attention_hopper_wgmma_qk_kernel<HeadDim,
                                                         kKvTile,
                                                         kTmaStages,
                                                         true,
                                                         false,
                                                         false,
                                                         false,
                                                         decltype(tma_key),
                                                         decltype(tma_value)>
        <<<grid, block, kDynSharedBytes, stream>>>(
            reinterpret_cast<const MtgrHopperElement*>(
                query_snd.data_ptr<at::BFloat16>()),
            reinterpret_cast<const MtgrHopperElement*>(
                key_snd.data_ptr<at::BFloat16>()),
            reinterpret_cast<const MtgrHopperElement*>(
                value_snd.data_ptr<at::BFloat16>()),
            segment_offsets_i32.data_ptr<int32_t>(),
            segment_rules_i32.data_ptr<int32_t>(),
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            reinterpret_cast<MtgrHopperElement*>(
                output_snd.data_ptr<at::BFloat16>()),
            tma_key,
            tma_value,
            tma_key,
            tma_value,
            -1,
            total_len,
            total_len,
            0,
            static_cast<int>(max_request_len),
            num_heads,
            num_segments,
            0,
            0,
            static_cast<float>(sm_scale) * kHopperLog2E);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_mtgr_ragged_segment_attention_hopper_common_args(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& segment_rules_i32,
    torch::Tensor output_snd,
    int64_t max_request_len,
    double sm_scale) {
  CHECK(query_snd.defined());
  CHECK(key_snd.defined());
  CHECK(value_snd.defined());
  CHECK(segment_offsets_i32.defined());
  CHECK(segment_rules_i32.defined());
  CHECK(output_snd.defined());
  CHECK(query_snd.is_cuda());
  CHECK(key_snd.is_cuda());
  CHECK(value_snd.is_cuda());
  CHECK(segment_offsets_i32.is_cuda());
  CHECK(segment_rules_i32.is_cuda());
  CHECK(output_snd.is_cuda());
  CHECK_EQ(key_snd.get_device(), query_snd.get_device());
  CHECK_EQ(value_snd.get_device(), query_snd.get_device());
  CHECK_EQ(segment_offsets_i32.get_device(), query_snd.get_device());
  CHECK_EQ(segment_rules_i32.get_device(), query_snd.get_device());
  CHECK_EQ(output_snd.get_device(), query_snd.get_device());
  CHECK(query_snd.is_contiguous());
  CHECK(key_snd.is_contiguous());
  CHECK(value_snd.is_contiguous());
  CHECK(segment_offsets_i32.is_contiguous());
  CHECK(segment_rules_i32.is_contiguous());
  CHECK(output_snd.is_contiguous());
  CHECK_EQ(query_snd.scalar_type(), torch::kBFloat16);
  CHECK_EQ(key_snd.scalar_type(), torch::kBFloat16);
  CHECK_EQ(value_snd.scalar_type(), torch::kBFloat16);
  CHECK_EQ(segment_offsets_i32.scalar_type(), torch::kInt32);
  CHECK_EQ(segment_rules_i32.scalar_type(), torch::kInt32);
  CHECK_EQ(output_snd.scalar_type(), torch::kBFloat16);
  CHECK_EQ(query_snd.dim(), 3);
  CHECK_EQ(key_snd.dim(), 3);
  CHECK_EQ(value_snd.dim(), 3);
  CHECK_EQ(segment_offsets_i32.dim(), 2);
  CHECK_EQ(segment_rules_i32.dim(), 1);
  CHECK_EQ(output_snd.dim(), 3);
  CHECK_EQ(query_snd.sizes(), key_snd.sizes());
  CHECK_EQ(query_snd.sizes(), value_snd.sizes());
  CHECK_EQ(query_snd.sizes(), output_snd.sizes());
  CHECK_GT(query_snd.size(0), 0);
  CHECK_GT(segment_offsets_i32.size(0), 0);
  CHECK_EQ(segment_offsets_i32.size(1), segment_rules_i32.size(0) + 1);
  CHECK_GT(segment_rules_i32.size(0), 1);
  CHECK_GT(max_request_len, 0);
  CHECK_LE(max_request_len, query_snd.size(0));
  CHECK_GT(sm_scale, 0.0);
}

void check_mtgr_ragged_segment_attention_hopper_unified_args(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& segment_rules_i32,
    const torch::Tensor& q_seq_starts_i32,
    const torch::Tensor& matched_prefix_lens_i32,
    int64_t match_mode,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& block_table_i32,
    int64_t block_size,
    int64_t max_request_len,
    double sm_scale,
    torch::Tensor output_snd) {
  CHECK(query_snd.defined());
  CHECK(key_snd.defined());
  CHECK(value_snd.defined());
  CHECK(segment_offsets_i32.defined());
  CHECK(segment_rules_i32.defined());
  CHECK(q_seq_starts_i32.defined());
  CHECK(matched_prefix_lens_i32.defined());
  CHECK(output_snd.defined());
  CHECK(query_snd.is_cuda());
  CHECK(key_snd.is_cuda());
  CHECK(value_snd.is_cuda());
  CHECK(segment_offsets_i32.is_cuda());
  CHECK(segment_rules_i32.is_cuda());
  CHECK(q_seq_starts_i32.is_cuda());
  CHECK(matched_prefix_lens_i32.is_cuda());
  CHECK(output_snd.is_cuda());
  CHECK_EQ(key_snd.get_device(), query_snd.get_device());
  CHECK_EQ(value_snd.get_device(), query_snd.get_device());
  CHECK_EQ(segment_offsets_i32.get_device(), query_snd.get_device());
  CHECK_EQ(segment_rules_i32.get_device(), query_snd.get_device());
  CHECK_EQ(q_seq_starts_i32.get_device(), query_snd.get_device());
  CHECK_EQ(matched_prefix_lens_i32.get_device(), query_snd.get_device());
  CHECK_EQ(output_snd.get_device(), query_snd.get_device());
  CHECK(query_snd.is_contiguous());
  CHECK(key_snd.is_contiguous());
  CHECK(value_snd.is_contiguous());
  CHECK(segment_offsets_i32.is_contiguous());
  CHECK(segment_rules_i32.is_contiguous());
  CHECK(q_seq_starts_i32.is_contiguous());
  CHECK(matched_prefix_lens_i32.is_contiguous());
  CHECK(output_snd.is_contiguous());
  CHECK_EQ(query_snd.scalar_type(), torch::kBFloat16);
  CHECK_EQ(key_snd.scalar_type(), torch::kBFloat16);
  CHECK_EQ(value_snd.scalar_type(), torch::kBFloat16);
  CHECK_EQ(segment_offsets_i32.scalar_type(), torch::kInt32);
  CHECK_EQ(segment_rules_i32.scalar_type(), torch::kInt32);
  CHECK_EQ(q_seq_starts_i32.scalar_type(), torch::kInt32);
  CHECK_EQ(matched_prefix_lens_i32.scalar_type(), torch::kInt32);
  CHECK_EQ(output_snd.scalar_type(), torch::kBFloat16);
  CHECK_EQ(query_snd.dim(), 3);
  CHECK_EQ(key_snd.dim(), 3);
  CHECK_EQ(value_snd.dim(), 3);
  CHECK_EQ(segment_offsets_i32.dim(), 2);
  CHECK_EQ(segment_rules_i32.dim(), 1);
  CHECK_EQ(q_seq_starts_i32.dim(), 1);
  CHECK_EQ(matched_prefix_lens_i32.dim(), 1);
  CHECK_EQ(output_snd.dim(), 3);
  CHECK_EQ(query_snd.sizes(), key_snd.sizes());
  CHECK_EQ(query_snd.sizes(), value_snd.sizes());
  CHECK_EQ(query_snd.sizes(), output_snd.sizes());
  CHECK_GT(query_snd.size(0), 0);
  CHECK_GT(query_snd.size(1), 0);
  CHECK_GT(query_snd.size(2), 0);
  CHECK_EQ(segment_offsets_i32.size(1), segment_rules_i32.size(0) + 1);
  CHECK_EQ(q_seq_starts_i32.size(0), segment_offsets_i32.size(0));
  CHECK_EQ(matched_prefix_lens_i32.size(0), segment_offsets_i32.size(0));
  CHECK_GT(block_size, 0);
  CHECK_GT(max_request_len, 0);
  CHECK_GT(sm_scale, 0.0);

  if (match_mode == 0) {
    return;
  }

  CHECK(key_cache.is_cuda());
  CHECK(value_cache.is_cuda());
  CHECK(block_table_i32.is_cuda());
  CHECK_EQ(key_cache.get_device(), query_snd.get_device());
  CHECK_EQ(value_cache.get_device(), query_snd.get_device());
  CHECK_EQ(block_table_i32.get_device(), query_snd.get_device());
  CHECK(key_cache.is_contiguous());
  CHECK(value_cache.is_contiguous());
  CHECK(block_table_i32.is_contiguous());
  CHECK_EQ(key_cache.scalar_type(), torch::kBFloat16);
  CHECK_EQ(value_cache.scalar_type(), torch::kBFloat16);
  CHECK_EQ(block_table_i32.scalar_type(), torch::kInt32);
  CHECK_EQ(key_cache.dim(), 4);
  CHECK_EQ(value_cache.dim(), 4);
  CHECK_EQ(block_table_i32.dim(), 2);
  CHECK_EQ(key_cache.sizes(), value_cache.sizes());
  CHECK_EQ(block_table_i32.size(0), segment_offsets_i32.size(0));
  CHECK_EQ(key_cache.size(1), block_size);
  CHECK_EQ(key_cache.size(2), query_snd.size(1));
  CHECK_EQ(key_cache.size(3), query_snd.size(2));
}

torch::Tensor mtgr_hopper_build_unified_q_seq_starts(
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& matched_prefix_lens_i32) {
  const int64_t batch_size = segment_offsets_i32.size(0);
  auto q_seq_starts_i32 =
      torch::zeros({batch_size}, matched_prefix_lens_i32.options());
  if (batch_size <= 1) {
    return q_seq_starts_i32;
  }

  const int64_t last_offset_col = segment_offsets_i32.size(1) - 1;
  auto prefix_live_lens_i32 =
      segment_offsets_i32.select(1, last_offset_col)
          .slice(0, 0, batch_size - 1) -
      matched_prefix_lens_i32.slice(0, 0, batch_size - 1);
  q_seq_starts_i32.slice(0, 1).copy_(
      torch::cumsum(prefix_live_lens_i32, 0, torch::kInt32));
  return q_seq_starts_i32;
}

template <bool AllowMixedRequests>
void dispatch_mtgr_ragged_segment_attention_hopper_unified_impl(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& segment_rules_i32,
    const torch::Tensor& q_seq_starts_i32,
    const torch::Tensor& matched_prefix_lens_i32,
    const torch::Tensor& block_table_i32,
    int64_t block_size,
    int64_t max_request_len,
    double sm_scale,
    torch::Tensor output_snd) {
  c10::cuda::CUDAGuard device_guard(query_snd.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  CHECK_GE(props->major, 9)
      << "Hopper ragged segment attention unified research path requires "
         "SM90+.";

  switch (query_snd.size(2)) {
    case 64:
      launch_mtgr_ragged_segment_attention_hopper_wgmma_qk_unified_kernel<
          64,
          AllowMixedRequests>(
          query_snd,
          key_snd,
          value_snd,
          key_cache,
          value_cache,
          segment_offsets_i32,
          segment_rules_i32,
          q_seq_starts_i32,
          matched_prefix_lens_i32,
          block_table_i32,
          block_size,
          max_request_len,
          sm_scale,
          output_snd);
      return;
    case 128:
      launch_mtgr_ragged_segment_attention_hopper_wgmma_qk_unified_kernel<
          128,
          AllowMixedRequests>(
          query_snd,
          key_snd,
          value_snd,
          key_cache,
          value_cache,
          segment_offsets_i32,
          segment_rules_i32,
          q_seq_starts_i32,
          matched_prefix_lens_i32,
          block_table_i32,
          block_size,
          max_request_len,
          sm_scale,
          output_snd);
      return;
    default:
      CHECK(false) << "Unsupported head dim for Hopper unified research path: "
                   << query_snd.size(2);
  }
}

void dispatch_mtgr_ragged_segment_attention_hopper_unified_partial_only(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& segment_rules_i32,
    const torch::Tensor& q_seq_starts_i32,
    const torch::Tensor& matched_prefix_lens_i32,
    const torch::Tensor& block_table_i32,
    int64_t block_size,
    int64_t max_request_len,
    double sm_scale,
    torch::Tensor output_snd) {
  dispatch_mtgr_ragged_segment_attention_hopper_unified_impl<false>(
      query_snd,
      key_snd,
      value_snd,
      key_cache,
      value_cache,
      segment_offsets_i32,
      segment_rules_i32,
      q_seq_starts_i32,
      matched_prefix_lens_i32,
      block_table_i32,
      block_size,
      max_request_len,
      sm_scale,
      output_snd);
}

void dispatch_mtgr_ragged_segment_attention_hopper_unified_mixed(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& segment_rules_i32,
    const torch::Tensor& q_seq_starts_i32,
    const torch::Tensor& matched_prefix_lens_i32,
    const torch::Tensor& block_table_i32,
    int64_t block_size,
    int64_t max_request_len,
    double sm_scale,
    torch::Tensor output_snd) {
  dispatch_mtgr_ragged_segment_attention_hopper_unified_impl<true>(
      query_snd,
      key_snd,
      value_snd,
      key_cache,
      value_cache,
      segment_offsets_i32,
      segment_rules_i32,
      q_seq_starts_i32,
      matched_prefix_lens_i32,
      block_table_i32,
      block_size,
      max_request_len,
      sm_scale,
      output_snd);
}

void dispatch_mtgr_ragged_segment_attention_hopper_wgmma_tma_qk(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& segment_rules_i32,
    int64_t max_request_len,
    double sm_scale,
    torch::Tensor output_snd) {
  c10::cuda::CUDAGuard device_guard(query_snd.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  CHECK_GE(props->major, 9)
      << "Hopper ragged segment attention WGMMA+TMA path requires SM90+.";

  switch (query_snd.size(2)) {
    case 64:
      launch_mtgr_ragged_segment_attention_hopper_wgmma_tma_qk_kernel<64>(
          query_snd,
          key_snd,
          value_snd,
          segment_offsets_i32,
          segment_rules_i32,
          max_request_len,
          sm_scale,
          output_snd);
      return;
    case 128:
      launch_mtgr_ragged_segment_attention_hopper_wgmma_tma_qk_kernel<128>(
          query_snd,
          key_snd,
          value_snd,
          segment_offsets_i32,
          segment_rules_i32,
          max_request_len,
          sm_scale,
          output_snd);
      return;
    default:
      CHECK(false) << "Unsupported head dim for Hopper ragged segment "
                      "attention WGMMA+TMA path: "
                   << query_snd.size(2);
  }
}

}  // namespace