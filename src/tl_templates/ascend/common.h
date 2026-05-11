// clang-format off
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
// clang-format on

#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"

#if defined(__has_include)
#if __has_include("version/cann_version.h")
#include "version/cann_version.h"
#endif
#endif

#include "shmem.h"

#define CUDART_INF_F 1.0f / 0.0f

typedef AscendC::int4b_t int4b_t;

namespace tl::ascend {
using namespace Catlass;
using namespace tla;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Gemm::Block;
using namespace AscendC;

using ArchTag = Arch::AtlasA2;
using LayoutGM = layout::RowMajor;

using LayoutL0A = layout::zZ;
using LayoutL0B = layout::nZ;
using LayoutL1 = layout::zN;
using LayoutL1T = layout::nZ;

constexpr int64_t UB_HALF_SIZE = 64;

template <typename T>
constexpr bool IsDuplicateSupported_v =
    std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> ||
    std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> ||
    std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
    std::is_same_v<T, float>;

CATLASS_DEVICE void disable_dma_atomic_compat() {
#if defined(CANN_MAJOR) && CANN_MAJOR >= 9
  AscendC::DisableDmaAtomic();
#else
  AscendC::SetAtomicNone();
#endif
}

template <typename T, uint32_t dstM, uint32_t dstN>
CATLASS_DEVICE void copy_gm_to_l1(LocalTensor<T> dstTensor,
                                  GlobalTensor<T> srcTensor,
                                  uint32_t realSrcN = 1, uint32_t realTailM = 0,
                                  uint32_t realTailN = 0) {
  uint32_t tailM = realTailM == 0 ? dstM : realTailM;
  uint32_t tailN = realTailN == 0 ? dstN : realTailN;
  if (tailM != dstM || tailN != dstN) {
    AscendC::InitConstValue(
        dstTensor,
        {1, static_cast<uint16_t>(dstM * dstN * sizeof(T) / 32), 0, 0});
    AscendC::PipeBarrier<PIPE_MTE2>();
  }
  auto layout = MakeLayoutFromTag(LayoutGM{tailM, realSrcN});
  auto src_LAYOUT = MakeLayoutTile(layout, tla::MakeShape(tailM, tailN));
  auto src = tla::MakeTensor<decltype(srcTensor), decltype(src_LAYOUT),
                             AscendC::TPosition::GM>(srcTensor, src_LAYOUT);

  using LayoutL1_ = Catlass::detail::TagToLayout_t<T, LayoutL1>;
  constexpr auto layoutInL1 = tla::MakeLayout<T, LayoutL1_>(dstM, dstN);
  auto dst = tla::MakeTensor<decltype(dstTensor), decltype(layoutInL1),
                             AscendC::TPosition::A1>(dstTensor, layoutInL1);

  TileCopyTla<ArchTag, decltype(src), decltype(dst)> tileCopier;
  tileCopier(dst, src);
}

template <typename T, uint32_t srcM, uint32_t srcN, bool transpose = false>
CATLASS_DEVICE void copy_l1_to_l0a(LocalTensor<T> dstTensor,
                                   LocalTensor<T> srcTensor, uint32_t dstM,
                                   uint32_t dstN) {
  using LayoutL1_ =
      std::conditional_t<transpose,
                         Catlass::detail::TagToLayout_t<T, LayoutL1T>,
                         Catlass::detail::TagToLayout_t<T, LayoutL1>>;
  constexpr auto layout = transpose ? tla::MakeLayout<T, LayoutL1_>(srcN, srcM)
                                    : tla::MakeLayout<T, LayoutL1_>(srcM, srcN);
  auto src_LAYOUT = MakeLayoutTile(layout, tla::MakeShape(dstM, dstN));
  auto src = MakeTensor<decltype(srcTensor), decltype(src_LAYOUT),
                        AscendC::TPosition::A1>(srcTensor, src_LAYOUT);

  using LayoutL0A_ = Catlass::detail::TagToLayout_t<T, LayoutL0A>;
  auto layoutAInL0 = tla::MakeLayout<T, LayoutL0A_>(dstM, dstN);
  auto dst = tla::MakeTensor<decltype(dstTensor), decltype(layoutAInL0),
                             AscendC::TPosition::A2>(dstTensor, layoutAInL0);
  TileCopyTla<ArchTag, decltype(src), decltype(dst)> tileCopier;
  tileCopier(dst, src);
}

template <typename T, uint32_t srcM, uint32_t srcN, bool transpose = false>
CATLASS_DEVICE void copy_l1_to_l0b(LocalTensor<T> dstTensor,
                                   LocalTensor<T> srcTensor, uint32_t dstM,
                                   uint32_t dstN) {
  using LayoutL1_ =
      std::conditional_t<transpose,
                         Catlass::detail::TagToLayout_t<T, LayoutL1T>,
                         Catlass::detail::TagToLayout_t<T, LayoutL1>>;
  constexpr auto layout = transpose ? tla::MakeLayout<T, LayoutL1_>(srcN, srcM)
                                    : tla::MakeLayout<T, LayoutL1_>(srcM, srcN);
  auto src_LAYOUT = MakeLayoutTile(layout, tla::MakeShape(dstM, dstN));
  auto src = MakeTensor<decltype(srcTensor), decltype(src_LAYOUT),
                        AscendC::TPosition::A1>(srcTensor, src_LAYOUT);

  using LayoutL0B_ = Catlass::detail::TagToLayout_t<T, LayoutL0B>;
  auto layoutBInL0 = tla::MakeLayout<T, LayoutL0B_>(dstM, dstN);
  auto dst = tla::MakeTensor<decltype(dstTensor), decltype(layoutBInL0),
                             AscendC::TPosition::B2>(dstTensor, layoutBInL0);

  TileCopyTla<ArchTag, decltype(src), decltype(dst)> tileCopier;
  tileCopier(dst, src);
}

template <typename T1, typename T2, uint32_t M, uint32_t N>
CATLASS_DEVICE void mma(LocalTensor<T1> const A, LocalTensor<T1> const B,
                        LocalTensor<T2> const C, bool init, uint32_t K,
                        uint8_t unitFlag = 0) {
  MmadParams mmadParams;
  mmadParams.m = M;
  mmadParams.n = N;
  mmadParams.k = K;
  mmadParams.cmatrixInitVal = init;
  // mmadParams.unitFlag = unitFlag;

  Mmad(C, A, B, mmadParams);

  constexpr uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
  // if constexpr ((M / C0_NUM_PER_FRACTAL) * (N / C0_NUM_PER_FRACTAL) <
  //               PIPE_M_BARRIER_THRESHOLD) {
  //   PipeBarrier<PIPE_M>();
  // }
}

template <typename T1, typename T2, typename LayoutGM, uint32_t srcM,
          uint32_t srcN, bool enRelu = false>
CATLASS_DEVICE void
copy_l0c_to_gm(GlobalTensor<T2> dstTensor, LocalTensor<T1> srcTensor,
               uint32_t realDstN = 1, uint32_t realTailM = 0,
               uint32_t realTailN = 0) {
  uint32_t tailM = realTailM == 0 ? srcM : realTailM;
  uint32_t tailN = realTailN == 0 ? srcN : realTailN;
  auto layoutInL0C = tla::MakeLayoutL0C(srcM, srcN);
  auto src = tla::MakeTensor<decltype(srcTensor), decltype(layoutInL0C),
                             AscendC::TPosition::CO1>(srcTensor, layoutInL0C);
  LayoutGM gm{tailM, realDstN};
  auto layout = MakeLayoutFromTag(gm);
  auto dTensor = MakeTensor(dstTensor, layout, Arch::PositionGM{});
  auto layout_ = dTensor.layout();
  auto dst_LAYOUT = MakeLayoutTile(layout_, tla::MakeShape(tailM, tailN));
  auto dst = MakeTensor<decltype(dstTensor), decltype(dst_LAYOUT),
                        AscendC::TPosition::GM>(dstTensor, dst_LAYOUT);

  CopyL0CToGmTla<ArchTag, decltype(src), decltype(dst),
                 ScaleGranularity::NO_QUANT, enRelu>
      tileCopier;
  tileCopier(dst, src, 0);
}

template <uint32_t M, uint32_t N, uint32_t K, uint32_t block_M,
          uint32_t block_N, uint32_t SwizzleOffset = 1,
          uint32_t SwizzleDirection = 0>
CATLASS_DEVICE auto thread_block_swizzle(uint64_t pid) {
  GemmCoord problem_shape = GemmCoord(M, N, K);
  MatrixCoord tile_shape = MatrixCoord(block_M, block_N);

  GemmIdentityBlockSwizzle swizzle =
      GemmIdentityBlockSwizzle<SwizzleOffset, SwizzleDirection>(problem_shape,
                                                                tile_shape);

  auto cols = swizzle.loopsMN.column();

  auto coord = swizzle.GetBlockCoord(pid);

  // return coord;
  return coord.m() * cols + coord.n();
}

template <typename T, uint32_t dstN, uint32_t dstM = 1>
CATLASS_DEVICE void
copy_gm_to_ub(LocalTensor<T> dstTensor, GlobalTensor<T> srcTensor,
              uint32_t realSrcN = 1, uint32_t maskShapeM = dstM,
              uint32_t maskShapeN = dstN, T padValue = T(0)) {

  bool isPad = true;
  uint32_t rightPadding = 1;
  if (maskShapeN == dstN) {
    isPad = false;
    rightPadding = 0;
  }
  if (maskShapeM != dstM || maskShapeN != dstN) {
    if constexpr (IsDuplicateSupported_v<T>) {
      SetFlag<HardEvent::MTE2_V>(0);
      WaitFlag<HardEvent::MTE2_V>(0);
      SetFlag<HardEvent::MTE3_V>(0);
      WaitFlag<HardEvent::MTE3_V>(0);
      AscendC::Duplicate<T>(dstTensor, padValue, dstM * dstN);
      SetFlag<HardEvent::V_MTE2>(0);
      WaitFlag<HardEvent::V_MTE2>(0);
    }
  }
  AscendC::DataCopyExtParams dataCopyParams(
      maskShapeM, maskShapeN * sizeof(T), (realSrcN - maskShapeN) * sizeof(T),
      (dstN - maskShapeN) * sizeof(T) / 32, 0);
  AscendC::DataCopyPadExtParams<T> padParams(isPad, 0, rightPadding, padValue);
  AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
}

template <typename T, uint32_t srcN, uint32_t srcM = 1>
CATLASS_DEVICE void
copy_ub_to_gm(GlobalTensor<T> dstTensor, LocalTensor<T> srcTensor,
              uint32_t realdstN = 1, uint32_t maskShapeM = srcM,
              uint32_t maskShapeN = srcN) {
  AscendC::DataCopyExtParams dataCopyParams(
      maskShapeM, maskShapeN * sizeof(T), (srcN - maskShapeN) * sizeof(T) / 32,
      (realdstN - maskShapeN) * sizeof(T), 0);
  AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams);
}

template <typename T, uint32_t srcN, uint32_t srcM = 1>
CATLASS_DEVICE void
atomic_add_ub_to_gm(GlobalTensor<T> dstTensor, LocalTensor<T> srcTensor,
                    uint32_t realdstN = 1, uint32_t maskShapeM = srcM,
                    uint32_t maskShapeN = srcN) {
  AscendC::SetAtomicAdd<T>();
  copy_ub_to_gm<T, srcN, srcM>(dstTensor, srcTensor, realdstN, maskShapeM,
                               maskShapeN);
  disable_dma_atomic_compat();
}

template <typename T1, typename T2, typename LayoutGM, uint32_t srcM,
          uint32_t srcN, bool enRelu = false>
CATLASS_DEVICE void
atomic_add_l0c_to_gm(GlobalTensor<T2> dstTensor, LocalTensor<T1> srcTensor,
                     uint32_t realDstN = 1, uint32_t realTailM = 0,
                     uint32_t realTailN = 0) {
  AscendC::SetAtomicAdd<T2>();
  copy_l0c_to_gm<T1, T2, LayoutGM, srcM, srcN, enRelu>(
      dstTensor, srcTensor, realDstN, realTailM, realTailN);
  disable_dma_atomic_compat();
}

template <typename T1, typename T2, uint32_t len>
CATLASS_DEVICE void copy_ub_to_ub(LocalTensor<T1> dstTensor,
                                  LocalTensor<T2> srcTensor) {
  if constexpr (std::is_same_v<T1, T2>) {
    AscendC::DataCopy(dstTensor, srcTensor, len);
  } else {
    if constexpr ((std::is_same_v<T1, float> && std::is_same_v<T2, half>) ||
                  (std::is_same_v<T1, float> &&
                   std::is_same_v<T2, bfloat16_t>) ||
                  (std::is_same_v<T1, float> && std::is_same_v<T2, int16_t>) ||
                  (std::is_same_v<T1, half> && std::is_same_v<T2, int8_t>) ||
                  (std::is_same_v<T1, int16_t> &&
                   std::is_same_v<T2, int32_t>)) {
      AscendC::Cast(dstTensor, srcTensor, AscendC::RoundMode::CAST_NONE, len);
    } else {
      AscendC::Cast(dstTensor, srcTensor, AscendC::RoundMode::CAST_RINT, len);
    }
  }
}

template <typename T, uint32_t M, uint32_t N>
CATLASS_DEVICE void copy_ub_to_l1(LocalTensor<T> dstTensor,
                                  LocalTensor<T> srcTensor) {
  static_assert(std::is_same_v<T, half>, "only support half");
  static_assert(M % 16 == 0, "M must be the multiple of 16");

  AscendC::DataCopyExtParams dataCopyParams(M, N * sizeof(T), 0, 0, 0);

  AscendC::Nd2NzParams nd2nzParams;
  nd2nzParams.ndNum = 1;
  nd2nzParams.nValue = M;
  nd2nzParams.dValue = N;
  nd2nzParams.srcNdMatrixStride = 0;
  nd2nzParams.srcDValue = N;
  nd2nzParams.dstNzC0Stride = M;
  nd2nzParams.dstNzNStride = 1;
  nd2nzParams.dstNzMatrixStride = 0;

  AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, nd2nzParams);
}

template <typename T, uint32_t Len>
CATLASS_DEVICE void tile_add(LocalTensor<T> const &ubIn0,
                             LocalTensor<T> const &ubIn1,
                             LocalTensor<T> const &ubOut) {
  AscendC::Add(ubOut, ubIn0, ubIn1, Len);
}

template <typename T, uint32_t Len, uint32_t op>
CATLASS_DEVICE void elementwise_binary(LocalTensor<T> const &ubIn0,
                                       LocalTensor<T> const &ubIn1,
                                       LocalTensor<T> const &ubOut) {
  // AscendC::Elementwise(ubOut, ubIn0, ubIn1, op, Len);
  if constexpr (op == 0) {
    AscendC::Add(ubOut, ubIn0, ubIn1, Len);
  } else if constexpr (op == 1) {
    AscendC::Sub(ubOut, ubIn0, ubIn1, Len);
  } else if constexpr (op == 2) {
    AscendC::Mul(ubOut, ubIn0, ubIn1, Len);
  } else if constexpr (op == 3) {
    AscendC::Div(ubOut, ubIn0, ubIn1, Len);
  }
}

template <typename T>
CATLASS_DEVICE void shmem_put_nbi(const GlobalTensor<T> &output,
                                  const GlobalTensor<T> &input, size_t nelems,
                                  size_t newPe) {
  AscendC::TPipe pipe;
  uint32_t ub_size = UB_HALF_SIZE * 2 + 64;
  AscendC::TBuf<AscendC::TPosition::VECIN> ub_buf;
  pipe.InitBuffer(ub_buf, ub_size);
  auto ub_tensor = ub_buf.Get<T>();
  pipe.Destroy();
  __gm__ T *outputPtr = const_cast<__gm__ T *>(output.GetPhyAddr());
  __gm__ T *inputPtr = const_cast<__gm__ T *>(input.GetPhyAddr());
  __ubuf__ T *buf = reinterpret_cast<__ubuf__ T *>(ub_tensor.GetPhyAddr());
  aclshmemx_mte_put_nbi(outputPtr, inputPtr, buf, ub_size, nelems, newPe,
                        EVENT_ID0);
}

template <typename T>
CATLASS_DEVICE void shmem_ub_put_nbi(const LocalTensor<T> &ubTensor,
                                     const GlobalTensor<T> &output,
                                     size_t nelems, int newPe, int strelem) {
  aclshmemx_mte_put_nbi(const_cast<__gm__ T *>(output.GetPhyAddr() + strelem),
                        reinterpret_cast<__ubuf__ T *>(ubTensor.GetPhyAddr()),
                        nelems, newPe, EVENT_ID0);
}

template <typename T>
CATLASS_DEVICE void shmem_get_nbi(const GlobalTensor<T> &output,
                                  const GlobalTensor<T> &input, size_t nelems,
                                  size_t newPe) {
  AscendC::TPipe pipe;
  uint32_t ub_size = UB_HALF_SIZE * 2 + 64;
  AscendC::TBuf<AscendC::TPosition::VECIN> ub_buf;
  pipe.InitBuffer(ub_buf, ub_size);
  auto ub_tensor = ub_buf.Get<T>();
  pipe.Destroy();
  __gm__ T *outputPtr = const_cast<__gm__ T *>(output.GetPhyAddr());
  __gm__ T *inputPtr = const_cast<__gm__ T *>(input.GetPhyAddr());
  __ubuf__ T *buf = reinterpret_cast<__ubuf__ T *>(ub_tensor.GetPhyAddr());
  aclshmemx_mte_get_nbi(outputPtr, inputPtr, buf, ub_size, nelems, newPe,
                        EVENT_ID0);
}

template <typename T>
CATLASS_DEVICE void shmem_ub_get_nbi(const LocalTensor<T> &output,
                                     const GlobalTensor<T> &input,
                                     size_t nelems, size_t newPe) {
  aclshmemx_mte_get_nbi(reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()),
                        const_cast<__gm__ T *>(input.GetPhyAddr()), nelems,
                        newPe, EVENT_ID0);
}

template <typename T, uint32_t Len, uint32_t op>
CATLASS_DEVICE void elementwise_unary(LocalTensor<T> const &ubIn,
                                      LocalTensor<T> const &ubOut) {
  // AscendC::Elementwise(ubOut, ubIn0, ubIn1, op, Len);
  if constexpr (op == 0) {
    // TODO: Check layout, Len only has bug.
    AscendC::Exp(ubOut, ubIn, Len);
  }
}

template <typename dst, typename src, const char round_mode[], uint32_t Len>
CATLASS_DEVICE void cast(LocalTensor<dst> const &ubOut,
                         LocalTensor<src> const &ubIn) {
  AscendC::Cast(ubOut, ubIn, round_mode, Len);
}

// template <typename T, uint32_t Len>
// CATLASS_DEVICE void fill(LocalTensor<T> const &ubOut, T value) {
//   AscendC::Duplicate(ubOut, value, Len);
// }

template <typename T>
CATLASS_DEVICE void
reduce_sum_half(LocalTensor<T> const &dstTensor,
                LocalTensor<T> const &srcTensor, const int32_t mask,
                const int32_t repeatTime, const int32_t srcRepStride) {
  AscendC::WholeReduceSum<T>(dstTensor, srcTensor, mask, repeatTime, 1, 1,
                             srcRepStride);
}

template <typename T, uint32_t M, uint32_t N, int32_t dim>
CATLASS_DEVICE void
reduce_sum(LocalTensor<T> const &dstTensor, LocalTensor<T> const &srcTensor,
           LocalTensor<uint8_t> const &sharedTmpBuffer, bool clear = true) {
  uint32_t shape[] = {M, N};
  if (clear) {
    if constexpr (dim == -1) {
      AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR>(
          dstTensor, srcTensor, sharedTmpBuffer, shape, true);
    } else {
      AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA>(
          dstTensor, srcTensor, sharedTmpBuffer, shape, true);
    }
    return;
  }

  constexpr uint32_t kReduceResultLen = dim == -1 ? M : N;
  // ReduceSum appears to use scratch in a way that can interfere with a local
  // UB backup on real_shape/slice paths, so keep the old dst in scalar locals
  // before forcing clear=true and merging manually.
  T dstBackup[kReduceResultLen];
  for (uint32_t i = 0; i < kReduceResultLen; ++i) {
    dstBackup[i] = dstTensor.GetValue(i);
  }

  if constexpr (dim == -1) {
    AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR>(
        dstTensor, srcTensor, sharedTmpBuffer, shape, true);
  } else {
    AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA>(
        dstTensor, srcTensor, sharedTmpBuffer, shape, true);
  }

  for (uint32_t i = 0; i < kReduceResultLen; ++i) {
    T reducedValue = dstTensor.GetValue(i);
    dstTensor.SetValue(i, static_cast<T>(reducedValue + dstBackup[i]));
  }
}

template <typename T>
CATLASS_DEVICE T reduce_scalar_max_safe(T lhsValue, T rhsValue) {
  // Bisheng/AICore does not allow scalar half/bfloat16 comparisons inside
  // device code, so the clear=false fallback compares through float.
  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
    return static_cast<float>(lhsValue) > static_cast<float>(rhsValue)
               ? lhsValue
               : rhsValue;
  } else {
    return lhsValue > rhsValue ? lhsValue : rhsValue;
  }
}

template <typename T, uint32_t M, uint32_t N, int32_t dim>
CATLASS_DEVICE void
reduce_max(LocalTensor<T> const &dstTensor, LocalTensor<T> const &srcTensor,
           LocalTensor<uint8_t> const &sharedTmpBuffer, bool clear = true) {
  uint32_t shape[] = {M, N};
  if (clear) {
    if constexpr (dim == -1) {
      AscendC::ReduceMax<T, AscendC::Pattern::Reduce::AR>(
          dstTensor, srcTensor, sharedTmpBuffer, shape, true);
    } else {
      AscendC::ReduceMax<T, AscendC::Pattern::Reduce::RA>(
          dstTensor, srcTensor, sharedTmpBuffer, shape, true);
    }
    return;
  }

  // AscendC::ReduceMax(..., clear=false) does not reliably preserve the
  // upstream "merge old dst with reduced value" contract on real_shape/slice
  // paths, so we make the merge explicit here.
  constexpr uint32_t kReduceResultLen = dim == -1 ? M : N;
  T dstBackup[kReduceResultLen];
  for (uint32_t i = 0; i < kReduceResultLen; ++i) {
    dstBackup[i] = dstTensor.GetValue(i);
  }

  if constexpr (dim == -1) {
    AscendC::ReduceMax<T, AscendC::Pattern::Reduce::AR>(
        dstTensor, srcTensor, sharedTmpBuffer, shape, true);
  } else {
    AscendC::ReduceMax<T, AscendC::Pattern::Reduce::RA>(
        dstTensor, srcTensor, sharedTmpBuffer, shape, true);
  }

  // Keep the merge explicit instead of relying on an in-place vector max,
  // because aliasing dst with one input can produce unstable results here.
  for (uint32_t i = 0; i < kReduceResultLen; ++i) {
    T reducedValue = dstTensor.GetValue(i);
    T backupValue = dstBackup[i];
    dstTensor.SetValue(i, reduce_scalar_max_safe(reducedValue, backupValue));
  }
}

template <typename T>
CATLASS_DEVICE T reduce_scalar_min_safe(T lhsValue, T rhsValue) {
  // Bisheng/AICore does not allow scalar half/bfloat16 comparisons inside
  // device code, so the clear=false fallback compares through float.
  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
    return static_cast<float>(lhsValue) < static_cast<float>(rhsValue)
               ? lhsValue
               : rhsValue;
  } else {
    return lhsValue < rhsValue ? lhsValue : rhsValue;
  }
}

template <typename T, uint32_t M, uint32_t N, int32_t dim>
CATLASS_DEVICE void
reduce_min(LocalTensor<T> const &dstTensor, LocalTensor<T> const &srcTensor,
           LocalTensor<uint8_t> const &sharedTmpBuffer, bool clear = true) {
  uint32_t shape[] = {M, N};
  if (clear) {
    if constexpr (dim == -1) {
      AscendC::ReduceMin<T, AscendC::Pattern::Reduce::AR>(
          dstTensor, srcTensor, sharedTmpBuffer, shape, true);
    } else {
      AscendC::ReduceMin<T, AscendC::Pattern::Reduce::RA>(
          dstTensor, srcTensor, sharedTmpBuffer, shape, true);
    }
    return;
  }

  // AscendC::ReduceMin(..., clear=false) does not reliably preserve the
  // upstream "merge old dst with reduced value" contract on real_shape/slice
  // paths, so we make the merge explicit here.
  constexpr uint32_t kReduceResultLen = dim == -1 ? M : N;
  T dstBackup[kReduceResultLen];
  for (uint32_t i = 0; i < kReduceResultLen; ++i) {
    dstBackup[i] = dstTensor.GetValue(i);
  }

  if constexpr (dim == -1) {
    AscendC::ReduceMin<T, AscendC::Pattern::Reduce::AR>(
        dstTensor, srcTensor, sharedTmpBuffer, shape, true);
  } else {
    AscendC::ReduceMin<T, AscendC::Pattern::Reduce::RA>(
        dstTensor, srcTensor, sharedTmpBuffer, shape, true);
  }

  // Keep the merge explicit instead of relying on an in-place vector min,
  // because aliasing dst with one input can produce unstable results here.
  for (uint32_t i = 0; i < kReduceResultLen; ++i) {
    T reducedValue = dstTensor.GetValue(i);
    T backupValue = dstBackup[i];
    dstTensor.SetValue(i, reduce_scalar_min_safe(reducedValue, backupValue));
  }
}

static constexpr uint32_t L0AB_EVENT = 0;

template <typename T1, typename T2, uint32_t M, uint32_t N, uint32_t K,
          bool transpose_A = false, bool transpose_B = false>
CATLASS_DEVICE void
gemm_v0(LocalTensor<T1> const &A, LocalTensor<T1> const &B,
        LocalTensor<T2> const &C, // this must be located in l0c
        AscendC::TBuf<AscendC::TPosition::A2> &l0a_,
        AscendC::TBuf<AscendC::TPosition::B2> &l0b_, bool clear) {
  auto l0a = l0a_.Get<T1>();
  auto l0b = l0b_.Get<T1>();
  constexpr uint32_t kL0Size = 128;
  uint32_t kL0split = (K + kL0Size - 1) / kL0Size;
  uint32_t kL0Tail = K - (kL0split - 1) * kL0Size;
  bool initflag = false;

  SetFlag<HardEvent::MTE2_MTE1>(L0AB_EVENT);
  WaitFlag<HardEvent::MTE2_MTE1>(L0AB_EVENT);
  SetFlag<HardEvent::FIX_M>(L0AB_EVENT);
  WaitFlag<HardEvent::FIX_M>(L0AB_EVENT);

  SetFlag<HardEvent::M_MTE1>(L0AB_EVENT);
  SetFlag<HardEvent::M_MTE1>(L0AB_EVENT + 1);

  for (uint32_t kL0Idx = 0; kL0Idx < kL0split; kL0Idx++) {
    initflag = (clear && (kL0Idx == 0));
    uint32_t kSize = (kL0Idx == kL0split - 1) ? kL0Tail : kL0Size;
    uint32_t pp = (kL0Idx & 1);

    uint32_t l0a_base = pp * (M * kL0Size);
    uint32_t l0b_base = pp * (N * kL0Size);

    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT + pp);
    if constexpr (!transpose_A) {
      tl::ascend::copy_l1_to_l0a<T1, M, K>(l0a[l0a_base],
                                           A[kL0Idx * M * kL0Size], M, kSize);
    } else {
      tl::ascend::copy_l1_to_l0a<T1, K, M, true>(
          l0a[l0a_base], A[kL0Idx * 16 * kL0Size], M, kSize);
    }
    if constexpr (!transpose_B) {
      tl::ascend::copy_l1_to_l0b<T1, K, N>(l0b[l0b_base],
                                           B[kL0Idx * 16 * kL0Size], kSize, N);
    } else {
      tl::ascend::copy_l1_to_l0b<T1, N, K, true>(
          l0b[l0b_base], B[kL0Idx * N * kL0Size], kSize, N);
    }
    SetFlag<HardEvent::MTE1_M>(L0AB_EVENT + pp);
    WaitFlag<HardEvent::MTE1_M>(L0AB_EVENT + pp);
    PipeBarrier<PIPE_M>();
    tl::ascend::mma<T1, T2, M, N>(l0a[l0a_base], l0b[l0b_base], C, initflag,
                                  kSize);
    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT + pp);
  }
  WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT);
  WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT + 1);

  SetFlag<HardEvent::MTE1_MTE2>(L0AB_EVENT);
  WaitFlag<HardEvent::MTE1_MTE2>(L0AB_EVENT);
  SetFlag<HardEvent::M_FIX>(L0AB_EVENT);
  WaitFlag<HardEvent::M_FIX>(L0AB_EVENT);
}

// 2-way merge sort
template <typename T>
CATLASS_DEVICE void
MergeSort(const LocalTensor<T> &dst, const LocalTensor<uint8_t> &tmp,
          const LocalTensor<T> &src0, const LocalTensor<T> &src1,
          uint32_t blockLen0, uint32_t blockLen1) {
  // Note: tmp parameter is kept for API consistency with PTO backend but not
  // used in AscendC

  AscendC::MrgSort4Info params;
  params.elementLengths[0] = blockLen0;
  params.elementLengths[1] = blockLen1;
  params.elementLengths[2] = 0;
  params.elementLengths[3] = 0;
  params.ifExhaustedSuspension = false;
  params.validBit = 3;

  AscendC::MrgSortSrcList<T> srcList;
  srcList.src1 = src0;
  srcList.src2 = src1;
  srcList.src3 = src0;
  srcList.src4 = src0;

  AscendC::MrgSort<T>(dst, srcList, params);
  PipeBarrier<PIPE_V>();
}

// 3-way merge sort
template <typename T>
CATLASS_DEVICE void
MergeSort(const LocalTensor<T> &dst, const LocalTensor<uint8_t> &tmp,
          const LocalTensor<T> &src0, const LocalTensor<T> &src1,
          const LocalTensor<T> &src2, uint32_t blockLen0, uint32_t blockLen1,
          uint32_t blockLen2) {
  // Note: tmp parameter is kept for API consistency with PTO backend but not
  // used in AscendC

  AscendC::MrgSort4Info params;
  params.elementLengths[0] = blockLen0;
  params.elementLengths[1] = blockLen1;
  params.elementLengths[2] = blockLen2;
  params.elementLengths[3] = 0;
  params.ifExhaustedSuspension = false;
  params.validBit = 7;

  AscendC::MrgSortSrcList<T> srcList;
  srcList.src1 = src0;
  srcList.src2 = src1;
  srcList.src3 = src2;
  srcList.src4 = src0;

  AscendC::MrgSort<T>(dst, srcList, params);
  PipeBarrier<PIPE_V>();
}

// 4-way merge sort
template <typename T>
CATLASS_DEVICE void
MergeSort(const LocalTensor<T> &dst, const LocalTensor<uint8_t> &tmp,
          const LocalTensor<T> &src0, const LocalTensor<T> &src1,
          const LocalTensor<T> &src2, const LocalTensor<T> &src3,
          uint32_t blockLen0, uint32_t blockLen1, uint32_t blockLen2,
          uint32_t blockLen3) {
  // Note: tmp parameter is kept for API consistency with PTO backend but not
  // used in AscendC

  AscendC::MrgSort4Info params;
  params.elementLengths[0] = blockLen0;
  params.elementLengths[1] = blockLen1;
  params.elementLengths[2] = blockLen2;
  params.elementLengths[3] = blockLen3;
  params.ifExhaustedSuspension = false;
  params.validBit = 15;

  AscendC::MrgSortSrcList<T> srcList;
  srcList.src1 = src0;
  srcList.src2 = src1;
  srcList.src3 = src2;
  srcList.src4 = src3;

  AscendC::MrgSort<T>(dst, srcList, params);
  PipeBarrier<PIPE_V>();
}

template <typename T>
CATLASS_DEVICE void GatherMask(const LocalTensor<T> &dst,
                               const LocalTensor<T> &sortedTensor,
                               uint8_t src1Pattern) {
  uint32_t eleNum = sortedTensor.GetSize();
  GatherMaskParams gatherMaskParams;
  gatherMaskParams.repeatTimes = Ceil(eleNum * sizeof(T), 256);
  gatherMaskParams.src0BlockStride = 1;
  gatherMaskParams.src0RepeatStride = 8;
  gatherMaskParams.src1RepeatStride = 0;
  uint64_t rsvdCnt = 0; // 用于保存筛选后保留下来的元素个数
  GatherMask(dst, sortedTensor, src1Pattern, false, static_cast<uint32_t>(0),
             gatherMaskParams, rsvdCnt);
  PipeBarrier<PIPE_V>();
}

template <typename T, typename U>
CATLASS_DEVICE void GatherMask(const LocalTensor<T> &dst,
                               const LocalTensor<T> &sortedTensor,
                               const LocalTensor<U> &src1Pattern) {
  uint32_t eleNum = sortedTensor.GetSize();
  GatherMaskParams gatherMaskParams;
  gatherMaskParams.repeatTimes = Ceil(eleNum * sizeof(T), 256);
  gatherMaskParams.src0BlockStride = 1;
  gatherMaskParams.src0RepeatStride = 8;
  gatherMaskParams.src1RepeatStride = 0;
  uint64_t rsvdCnt = 0; // 用于保存筛选后保留下来的元素个数
  GatherMask(dst, sortedTensor, src1Pattern, false, static_cast<uint32_t>(0),
             gatherMaskParams, rsvdCnt);
}

template <typename T>
CATLASS_DEVICE void Gather(const LocalTensor<T> &dst,
                           const LocalTensor<T> &sortedTensor,
                           const LocalTensor<uint32_t> &src1Pattern) {

  int32_t count = src1Pattern.GetSize();
  int32_t scalarValue = sizeof(T);
  LocalTensor<int32_t> offset = const_cast<LocalTensor<uint32_t> &>(src1Pattern)
                                    .template ReinterpretCast<int32_t>();
  AscendC::Muls(offset, offset, scalarValue, count);
  AscendC::Gather(dst, sortedTensor,
                  offset.template ReinterpretCast<uint32_t>(),
                  static_cast<uint32_t>(0), static_cast<uint32_t>(count));
}

template <typename T>
CATLASS_DEVICE void
Gatherb(const LocalTensor<T> &dst, const LocalTensor<T> &src0,
        const LocalTensor<uint32_t> &offset, uint8_t repeat_time,
        uint8_t dst_blk_stride, uint8_t dst_rep_stride) {
  GatherRepeatParams gatherRepeatParams;
  gatherRepeatParams.dstBlkStride = dst_blk_stride;
  gatherRepeatParams.dstRepStride = dst_rep_stride;
  Gatherb(dst.template ReinterpretCast<uint32_t>(),
          src0.template ReinterpretCast<uint32_t>(),
          offset.template ReinterpretCast<uint32_t>(), repeat_time,
          gatherRepeatParams);
  PipeBarrier<PIPE_V>();
}

template <typename T>
CATLASS_DEVICE void InitSortBuf(const LocalTensor<T> &src, int64_t eleNum,
                                int64_t rsv = 0) {
  constexpr int32_t NEG_INF = 0xFF800000;
  constexpr uint8_t VEC_REPEAT_MAX = 255;
  constexpr uint8_t B32_VEC_ELM_NUM = 64;
  uint64_t mask1[2] = {0x5555555555555555, 0};
  uint64_t mask0[2] = {0xaaaaaaaaaaaaaaaa, 0};
  int64_t repeatNum = eleNum / B32_VEC_ELM_NUM;
  int64_t forLoop = repeatNum / VEC_REPEAT_MAX;
  int64_t forRemain = repeatNum % VEC_REPEAT_MAX;
  for (int i = 0; i < forLoop; i++) {
    Duplicate(src.template ReinterpretCast<int32_t>(), NEG_INF, mask1,
              VEC_REPEAT_MAX, 1, 8);
    Duplicate(src.template ReinterpretCast<int32_t>(), -1, mask0,
              VEC_REPEAT_MAX, 1, 8);
  }
  if (forRemain > 0) {
    Duplicate(src.template ReinterpretCast<int32_t>()[forLoop * VEC_REPEAT_MAX *
                                                      B32_VEC_ELM_NUM],
              NEG_INF, mask1, forRemain, 1, 8);
    Duplicate(src.template ReinterpretCast<int32_t>()[forLoop * VEC_REPEAT_MAX *
                                                      B32_VEC_ELM_NUM],
              -1, mask0, forRemain, 1, 8);
  }
  PipeBarrier<PIPE_V>();
}

template <typename T>
CATLASS_DEVICE void brcb(const LocalTensor<T> &dst, const LocalTensor<T> &src0,
                         const uint8_t repeatTime, const uint16_t dstBlkStride,
                         const uint16_t dstRepStride) {
  AscendC::BrcbRepeatParams repeatParams(dstBlkStride, dstRepStride);
  AscendC::Brcb<T>(dst, src0, repeatTime, repeatParams);
}

template <typename T1, typename T2, typename LayOutL1, typename LayoutGM,
          uint32_t M, uint32_t N, uint32_t K, uint32_t baseM, uint32_t baseN,
          uint32_t baseK, bool init, bool is_transpose_A = false,
          bool is_transpose_B = false, bool enable_relu = false>
CATLASS_DEVICE void gemmL1(LocalTensor<T1> A, LocalTensor<T1> B,
                           GlobalTensor<T1> C, LocalTensor<T1> A2,
                           LocalTensor<T1> B2, LocalTensor<T2> C2) {
  for (uint32_t loopM = 0; loopM < M / baseM; loopM++) {
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);

    copy_l1_to_l0a<T1, M, K, baseM, baseK>(A2, A[loopM * baseM * 16]);

    for (uint32_t loopN = 0; loopN < N / baseN; loopN++) {
      copy_l1_to_l0b<T1, K, N, baseK, baseN>(B2, B[loopN * baseN * K]);

      AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);

      mma<T1, T2, baseM, baseN, baseK, init>(A2, B2, C2);

      AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
      AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(0);
      AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
      AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);

      copy_l0c_to_gm<T1, T2, LayoutGM, baseM, baseN, M, N>(
          C[loopM * baseM * N + loopN * baseN], C2, enable_relu);

      AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
      AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(0);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
  }
}

template <typename T, int32_t dim, int32_t axis, bool isReuseSource = false>
CATLASS_DEVICE void
Broadcast(const LocalTensor<T> &dst, const LocalTensor<T> &src,
          LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t dstShape[dim],
          const uint32_t srcShape[dim]) {
  AscendC::Broadcast<T, dim, axis, isReuseSource>(dst, src, dstShape, srcShape,
                                                  sharedTmpBuffer);
}

template <typename T>
CATLASS_DEVICE void Fill(const LocalTensor<T> &dst, const T &scalarValue,
                         const int32_t &count) {
  AscendC::Duplicate<T>(dst, scalarValue, count);
}

template <typename T>
CATLASS_DEVICE void ArithProgression(const LocalTensor<T> &dst,
                                     const T firstValue, const T diffValue,
                                     const int32_t count) {
  AscendC::ArithProgression<T>(dst, firstValue, diffValue, count);
}

template <typename T>
CATLASS_DEVICE void Sort(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                         const LocalTensor<T> &tmp, const int32_t repeatTimes,
                         const int32_t actualCount) {
  if constexpr (sizeof(T) == 2) {
    // B16 (half): MrgSort requires >= 256 bytes per source, but Sort32 only
    // produces 128 bytes per block for B16. Work around by sorting in float.
    //
    // Layout in tmp (N = alignedCount, as float elements via ReinterpretCast):
    //   ftmp[0 .. N*2-1]    = Sort32 output + merge ping-pong buffer A
    //   ftmp[N*2 .. N*4-1]  = Sort<float>'s dst (merge ping-pong buffer B)
    //     - before Sort32: indices at [N*2..N*3), float_src at [N*3..N*4)
    //     - after  Sort32: entire region free for merge
    // Total: 4N float elements = 8N half elements.
    uint32_t N = repeatTimes * 32;

    auto ftmp = tmp.template ReinterpretCast<float>();
    auto float_src = ftmp[N * 3];

    // Cast half → float
    AscendC::Cast(float_src, src, AscendC::RoundMode::CAST_NONE, N);

    // Sort<float> guarantees result in dst (= ftmp[N*2])
    Sort<float>(ftmp[N * 2], float_src, ftmp, repeatTimes, actualCount);

    // Cast float result → half (2*N elements: interleaved [value, index] pairs)
    AscendC::Cast(dst, ftmp[N * 2], AscendC::RoundMode::CAST_RINT, N * 2);
    PipeBarrier<PIPE_V>();
    return;
  }

  constexpr uint32_t blockSize = 32;
  uint32_t alignedCount = repeatTimes * blockSize;
  uint32_t padCount = alignedCount - actualCount;
  uint32_t blockNum = repeatTimes;

  // Generate ascending indices as float values (0.0, 1.0, 2.0, ...) in dst
  // (temporary storage — overwritten by merge later). This allows tmp to
  // be only alignedCount*2 elements instead of alignedCount*4, because dst
  // (which is 2*alignedCount for interleaved output) doubles as the second
  // merge ping-pong buffer.
  AscendC::ArithProgression<T>(dst, T(0), T(1), alignedCount);
  PipeBarrier<PIPE_V>();
  LocalTensor<uint32_t> indices = dst.template ReinterpretCast<uint32_t>();

  // Pad src in-place with -inf for unused positions
  if (padCount > 0) {
    T negInf = -CUDART_INF_F;
    constexpr uint32_t elemPerBlock =
        32 / sizeof(T); // 16 for half, 8 for float
    uint32_t alignedActual = (actualCount / elemPerBlock) * elemPerBlock;
    uint32_t inBlockOffset = actualCount - alignedActual;

    if (inBlockOffset == 0) {
      // actualCount is already 32-byte aligned, simple Duplicate
      AscendC::Duplicate<T>(src[actualCount], negInf, padCount);
    } else {
      // Non-aligned: split into aligned bulk fill + masked partial block
      uint32_t nextAligned = alignedActual + elemPerBlock;
      // Fill full aligned blocks after the partial one
      if (nextAligned < alignedCount) {
        AscendC::Duplicate<T>(src[nextAligned], negInf,
                              alignedCount - nextAligned);
      }
      // Fill partial block using mask to preserve valid elements before
      // actualCount
      uint64_t mask0 = 0;
      for (uint32_t i = inBlockOffset; i < elemPerBlock; i++) {
        mask0 |= (1ULL << i);
      }
      uint64_t masks[2] = {mask0, 0};
      AscendC::Duplicate(src[alignedActual], negInf, masks, (uint8_t)1,
                         (uint16_t)1, (uint8_t)0);
    }
    PipeBarrier<PIPE_V>();
  }

  // Sort32: each 32-element block → tmp[0..alignedCount*2-1] (bufA)
  AscendC::Sort32(tmp, src, indices, repeatTimes);
  PipeBarrier<PIPE_V>();

  // Merge ping-pong between tmp[0..2N-1] and dst[0..2N-1].
  // tmp only needs alignedCount*2 elements (Sort32 output size).

  if (blockNum > 1) {
    uint32_t fullSegSize = blockSize;
    uint32_t lastSegSize = blockSize;
    uint32_t numSegs = blockNum;
    bool readFromTmp = true; // Sort32 output is in tmp

    while (numSegs > 1) {
      uint32_t newNumSegs = 0;
      uint32_t inOffset = 0;
      uint32_t outOffset = 0;

      for (uint32_t g = 0; g < numSegs; g += 4) {
        uint32_t groupCount = numSegs - g;
        if (groupCount > 4) {
          groupCount = 4;
        }
        uint32_t len0 = (g == numSegs - 1) ? lastSegSize : fullSegSize;
        uint32_t len1 = 0, len2 = 0, len3 = 0;
        uint32_t totalElems = len0;
        if (groupCount > 1) {
          len1 = (g + 1 == numSegs - 1) ? lastSegSize : fullSegSize;
          totalElems += len1;
        }
        if (groupCount > 2) {
          len2 = (g + 2 == numSegs - 1) ? lastSegSize : fullSegSize;
          totalElems += len2;
        }
        if (groupCount > 3) {
          len3 = (g + 3 == numSegs - 1) ? lastSegSize : fullSegSize;
          totalElems += len3;
        }

        if (groupCount == 1) {
          if (readFromTmp) {
            AscendC::DataCopy(dst[outOffset], tmp[inOffset], len0 * 2);
          } else {
            AscendC::DataCopy(tmp[outOffset], dst[inOffset], len0 * 2);
          }
        } else {
          AscendC::MrgSort4Info params;
          params.elementLengths[0] = len0;
          params.elementLengths[1] = len1;
          params.elementLengths[2] = groupCount > 2 ? len2 : 0;
          params.elementLengths[3] = groupCount > 3 ? len3 : 0;
          params.ifExhaustedSuspension = false;
          params.validBit = (1 << groupCount) - 1;

          uint32_t off0 = inOffset;
          uint32_t off1 = off0 + len0 * 2;
          uint32_t off2 = off1 + len1 * 2;
          uint32_t off3 = off2 + len2 * 2;

          AscendC::MrgSortSrcList<T> srcList;
          if (readFromTmp) {
            srcList.src1 = tmp[off0];
            srcList.src2 = tmp[off1];
            srcList.src3 = groupCount > 2 ? tmp[off2] : tmp[off0];
            srcList.src4 = groupCount > 3 ? tmp[off3] : tmp[off0];
            AscendC::MrgSort<T>(dst[outOffset], srcList, params);
          } else {
            srcList.src1 = dst[off0];
            srcList.src2 = dst[off1];
            srcList.src3 = groupCount > 2 ? dst[off2] : dst[off0];
            srcList.src4 = groupCount > 3 ? dst[off3] : dst[off0];
            AscendC::MrgSort<T>(tmp[outOffset], srcList, params);
          }
        }

        inOffset += totalElems * 2;
        outOffset += totalElems * 2;
        newNumSegs++;
      }

      PipeBarrier<PIPE_V>();

      uint32_t lastGroupStart = ((numSegs - 1) / 4) * 4;
      uint32_t lastGroupCount = numSegs - lastGroupStart;
      uint32_t newLastSegSize = 0;
      for (uint32_t i = 0; i < lastGroupCount; i++) {
        newLastSegSize +=
            (lastGroupStart + i == numSegs - 1) ? lastSegSize : fullSegSize;
      }

      fullSegSize = (newNumSegs > 1) ? 4 * fullSegSize : newLastSegSize;
      lastSegSize = newLastSegSize;
      numSegs = newNumSegs;
      readFromTmp = !readFromTmp;
    }

    // readFromTmp=true means last round wrote to tmp → result in tmp
    if (readFromTmp) {
      AscendC::DataCopy(dst, tmp, alignedCount * 2);
    }
  } else {
    // Single block: Sort32 output is in tmp, copy to dst
    AscendC::DataCopy(dst, tmp, alignedCount * 2);
  }
}

template <typename T>
CATLASS_DEVICE void ClampMax(const LocalTensor<T> &dst,
                             const LocalTensor<T> &buffer,
                             const LocalTensor<uint8_t> &tmp,
                             const T scalarValue, const int32_t count) {
  AscendC::ClampMax<T>(dst, buffer, tmp, scalarValue, count);
}

template <typename T>
CATLASS_DEVICE void TopK(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                         const LocalTensor<T> &tmp, const int32_t K,
                         const int32_t repeatTimes, const int32_t actualCount) {
  // Use tmp as the full-size sort destination (2 * alignedCount elements).
  // Sort writes its result into tmp's first region; we then copy the top-K
  // portion into dst.
  uint32_t alignedCount = repeatTimes * 32;
  // sortDst needs 2 * alignedCount elements; reuse the tail of tmp.
  // Layout of tmp: [0 .. 2*alignedCount-1] = sortDst, [2*alignedCount ..] =
  // sortTmp
  auto sortDst = tmp;
  auto sortTmp = tmp[alignedCount * 2];
  Sort<T>(sortDst, src, sortTmp, repeatTimes, actualCount);
  PipeBarrier<PIPE_V>();
  // Copy 2*K elements (interleaved value-index pairs) from sorted result to
  // dst. DataCopy requires the byte count to be a multiple of 32 bytes, so
  // round up.
  uint32_t topkElems = 2 * K;
  constexpr uint32_t elemsPerBlock = 32 / sizeof(T);
  uint32_t alignedTopk =
      ((topkElems + elemsPerBlock - 1) / elemsPerBlock) * elemsPerBlock;
  AscendC::DataCopy(dst, sortDst, alignedTopk);
}

template <typename T>
CATLASS_DEVICE void ClampMin(const LocalTensor<T> &dst,
                             const LocalTensor<T> &buffer,
                             const LocalTensor<uint8_t> &tmp,
                             const T scalarValue, const int32_t count) {
  AscendC::ClampMin<T>(dst, buffer, tmp, scalarValue, count);
}

template <typename T>
CATLASS_DEVICE void
Clamp(const LocalTensor<T> &dst, const LocalTensor<T> &buffer,
      const LocalTensor<uint8_t> &tmp, const T minScalarValue,
      const T maxScalarValue, const int32_t count) {
  AscendC::ClampMin<T>(dst, buffer, tmp, minScalarValue, count);
  AscendC::ClampMax<T>(dst, dst, tmp, maxScalarValue, count);
}

template <typename T, typename U>
CATLASS_DEVICE void
GatherMask_experiment(const LocalTensor<T> &dst, const LocalTensor<T> &src0,
                      const LocalTensor<U> &src1Pattern, const bool reduceMode,
                      const uint32_t mask, const uint32_t src0BlockStride,
                      const uint32_t repeatTimes, uint32_t src0RepeatStride,
                      const uint32_t src1RepeatStride, uint64_t rsvdCnt) {
  GatherMaskParams gatherMaskParams;
  gatherMaskParams.repeatTimes = repeatTimes;
  gatherMaskParams.src0BlockStride = src0BlockStride;
  gatherMaskParams.src0RepeatStride = src0RepeatStride;
  gatherMaskParams.src1RepeatStride = src1RepeatStride;
  GatherMask(dst, src0, src1Pattern, reduceMode, mask, gatherMaskParams,
             rsvdCnt);
}

template <typename T>
CATLASS_DEVICE void
Fill_experiment(const LocalTensor<T> &dst, const T &scalarValue, uint64_t mask0,
                const uint8_t repeatTime, const uint16_t dstBlockStride,
                const uint8_t dstRepeatStride) {
  uint64_t mask[1] = {mask0};
  AscendC::Duplicate(dst, scalarValue, mask, repeatTime, dstBlockStride,
                     dstRepeatStride);
}

template <typename T>
CATLASS_DEVICE void
Sum_experiment(const LocalTensor<T> &dst, const LocalTensor<T> &src,
               const uint32_t outter, const uint32_t inner, const uint32_t n) {
  SumParams sumParams;
  sumParams.outter = outter;
  sumParams.inner = inner;
  sumParams.n = n;
  AscendC::Sum(dst, src, sumParams);
}

template <typename T, uint32_t M, uint32_t N>
CATLASS_DEVICE void transpose_16x16(LocalTensor<T> const &dst,
                                    LocalTensor<T> const &src) {
  TransDataTo5HDParams transDataParams;
  transDataParams.dstHighHalf = false;
  transDataParams.srcHighHalf = false;
  transDataParams.repeatTimes = N;
  if (transDataParams.repeatTimes == 1) {
    transDataParams.dstRepStride = 0;
    transDataParams.srcRepStride = 0;
  } else {
    transDataParams.dstRepStride = M;
    transDataParams.srcRepStride = 1;
  }

  __ubuf__ T *dstList[16];
  __ubuf__ T *srcList[16];

  if constexpr (sizeof(T) == 4) {
    for (int32_t m = 0; m < 16; m = m + 2) {
      dstList[m] = (__ubuf__ T *)dst[16 * (m / 2)].GetPhyAddr();
      dstList[m + 1] = (__ubuf__ T *)dst[16 * (m / 2) + 16].GetPhyAddr();
    }
    for (int32_t n = 0; n < 16; n++) {
      srcList[n] = (__ubuf__ T *)src[n * 16].GetPhyAddr();
    }
  } else {
    for (int i = 0; i < 16; i++) {
      dstList[i] = (__ubuf__ T *)dst[i * N].GetPhyAddr();
      srcList[i] = (__ubuf__ T *)src[i * M].GetPhyAddr();
    }
  }

  AscendC::TransDataTo5HDImpl<T>(dstList, srcList, transDataParams);
  AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
CATLASS_DEVICE void transpose(LocalTensor<T> const &dst,
                              LocalTensor<T> const &src) {
  if constexpr (sizeof(T) == 2) {
    AscendC::Transpose(dst, src);
  } else {
    for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
        dst.SetValue(i * 16 + j, src.GetValue(j * 16 + i));
      }
    }
  }
}

} // namespace tl::ascend
