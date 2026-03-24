#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"

#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"

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

template <typename T, uint32_t dstM, uint32_t dstN>
CATLASS_DEVICE void copy_gm_to_l1(LocalTensor<T> dstTensor,
                                  GlobalTensor<T> srcTensor, uint32_t realSrcN = 1, uint32_t realTailM = 0, uint32_t realTailN = 0) {
  uint32_t tailM = realTailM == 0 ? dstM : realTailM;
  uint32_t tailN = realTailN == 0 ? dstN : realTailN;
  if (tailM != dstM || tailN != dstN) {
    AscendC::InitConstValue(dstTensor, {1, static_cast<uint16_t>(dstM * dstN * sizeof(T) / 32), 0, 0});
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
                                   LocalTensor<T> srcTensor,
                                   uint32_t dstM, uint32_t dstN) {
  using LayoutL1_ = std::conditional_t<transpose,
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
                                   LocalTensor<T> srcTensor,
                                   uint32_t dstM, uint32_t dstN) {
  using LayoutL1_ = std::conditional_t<transpose,
                                      Catlass::detail::TagToLayout_t<T, LayoutL1T>,
                                      Catlass::detail::TagToLayout_t<T, LayoutL1>>;
  constexpr auto layout = transpose ? 
  tla::MakeLayout<T, LayoutL1_>(srcN, srcM) : tla::MakeLayout<T, LayoutL1_>(srcM, srcN);
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
CATLASS_DEVICE void mma(LocalTensor<T1> const A, LocalTensor<T1> const B, LocalTensor<T2> const C,
                        bool init, uint32_t K, uint8_t unitFlag = 0) {
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

template <typename T1, typename T2, typename LayoutGM, uint32_t srcM, uint32_t srcN, bool enRelu = false>
CATLASS_DEVICE void copy_l0c_to_gm(GlobalTensor<T2> dstTensor,
                                   LocalTensor<T1> srcTensor,
                                   uint32_t realDstN = 1, uint32_t realTailM = 0, uint32_t realTailN = 0) {
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

  CopyL0CToGmTla<ArchTag, decltype(src), decltype(dst), ScaleGranularity::NO_QUANT, enRelu> tileCopier;
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
CATLASS_DEVICE void copy_gm_to_ub(LocalTensor<T> dstTensor,
                                  GlobalTensor<T> srcTensor,
                                  uint32_t realSrcN = 1) {
  AscendC::DataCopyExtParams dataCopyParams(
      dstM, dstN * sizeof(T), (realSrcN - dstN) * sizeof(T), 0, 0);
  AscendC::DataCopyPadExtParams<T> padParams(false, 0, 0, 0);
  AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
}

template <typename T, uint32_t srcN, uint32_t srcM = 1>
CATLASS_DEVICE void copy_ub_to_gm(GlobalTensor<T> dstTensor,
                                  LocalTensor<T> srcTensor,
                                  uint32_t realdstN = 1) {
  AscendC::DataCopyExtParams dataCopyParams(srcM, srcN * sizeof(T), 0,
                                            (realdstN - srcN) * sizeof(T), 0);
  AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams);
}

template <typename T1, typename T2, uint32_t len>
CATLASS_DEVICE void copy_ub_to_ub(LocalTensor<T1> dstTensor,
                                  LocalTensor<T2> srcTensor) {
  if constexpr (std::is_same_v<T1, T2>) {
    AscendC::DataCopy(dstTensor, srcTensor, len);
  } else {
    if constexpr ((std::is_same_v<T1, float> && std::is_same_v<T2, half>) ||
                  (std::is_same_v<T1, float> && std::is_same_v<T2, int16_t>) ||
                  (std::is_same_v<T1, half> && std::is_same_v<T2, int8_t>) ||
                  (std::is_same_v<T1, int16_t> && std::is_same_v<T2, int32_t>)) {
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
CATLASS_DEVICE void shmem_put_nbi(const GlobalTensor<T> &output, const GlobalTensor<T> &input,
                             size_t nelems, size_t newPe) {
    AscendC::TPipe pipe;
    uint32_t ub_size = UB_HALF_SIZE * 2 + 64;
    AscendC::TBuf<AscendC::TPosition::VECIN> ub_buf;
    pipe.InitBuffer(ub_buf, ub_size);
    auto ub_tensor = ub_buf.Get<T>();
    pipe.Destroy();
    __gm__ T* outputPtr = const_cast<__gm__ T*>(output.GetPhyAddr());
    __gm__ T* inputPtr = const_cast<__gm__ T*>(input.GetPhyAddr());
    __ubuf__ T* buf = reinterpret_cast<__ubuf__ T*>(ub_tensor.GetPhyAddr());
    aclshmemx_mte_put_nbi(outputPtr, inputPtr, buf, ub_size, nelems, newPe, EVENT_ID0);                                                                                 
}

template <typename T>
CATLASS_DEVICE void shmem_ub_put_nbi(const LocalTensor<T> &ubTensor, const GlobalTensor<T> &output, size_t nelems, int newPe, int strelem) {                                                                                  
    aclshmemx_mte_put_nbi(const_cast<__gm__ T*>(output.GetPhyAddr() + strelem),                                       
        reinterpret_cast<__ubuf__ T*>(ubTensor.GetPhyAddr()), nelems, newPe, EVENT_ID0);                                                                     
}

template <typename T>
CATLASS_DEVICE void shmem_get_nbi(const GlobalTensor<T> &output, const GlobalTensor<T> &input,
                                size_t nelems, size_t newPe) {
    AscendC::TPipe pipe;
    uint32_t ub_size = UB_HALF_SIZE * 2 + 64;
    AscendC::TBuf<AscendC::TPosition::VECIN> ub_buf;
    pipe.InitBuffer(ub_buf, ub_size);
    auto ub_tensor = ub_buf.Get<T>();
    pipe.Destroy();
    __gm__ T* outputPtr = const_cast<__gm__ T*>(output.GetPhyAddr());
    __gm__ T* inputPtr = const_cast<__gm__ T*>(input.GetPhyAddr());
    __ubuf__ T* buf = reinterpret_cast<__ubuf__ T*>(ub_tensor.GetPhyAddr());
    aclshmemx_mte_get_nbi(outputPtr, inputPtr, buf, ub_size, nelems, newPe, EVENT_ID0); 
}

template <typename T>
CATLASS_DEVICE void shmem_ub_get_nbi(const LocalTensor<T> &output, const GlobalTensor<T> &input,
                             size_t nelems, size_t newPe) {
    aclshmemx_mte_get_nbi(reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()),
        const_cast<__gm__ T*>(input.GetPhyAddr()), nelems, newPe, EVENT_ID0);
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

template <typename T, uint32_t M, uint32_t N, int32_t dim>
CATLASS_DEVICE void reduce_sum(LocalTensor<T> const &dstTensor,
                               LocalTensor<T> const &srcTensor,
                               LocalTensor<uint8_t> const &sharedTmpBuffer) {
  uint32_t shape[] = {M, N};
  if constexpr (dim == -1) {
    AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR>(
        dstTensor, srcTensor, sharedTmpBuffer, shape, true
    );
  } else {
    AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA>(
        dstTensor, srcTensor, sharedTmpBuffer, shape, true
    );
  }
}

template <typename T, uint32_t M, uint32_t N, int32_t dim>
CATLASS_DEVICE void reduce_max(LocalTensor<T> const &dstTensor,
                               LocalTensor<T> const &srcTensor,
                               LocalTensor<uint8_t> const &sharedTmpBuffer) {
  uint32_t shape[] = {M, N};
  if constexpr (dim == -1) {
    AscendC::ReduceMax<T, AscendC::Pattern::Reduce::AR>(
        dstTensor, srcTensor, sharedTmpBuffer, shape, true
    );
  } else {
    AscendC::ReduceMax<T, AscendC::Pattern::Reduce::RA>(
        dstTensor, srcTensor, sharedTmpBuffer, shape, true
    );
  }
}

template <typename T, uint32_t M, uint32_t N, int32_t dim>
CATLASS_DEVICE void reduce_min(LocalTensor<T> const &dstTensor,
                               LocalTensor<T> const &srcTensor,
                               LocalTensor<uint8_t> const &sharedTmpBuffer) {
  uint32_t shape[] = {M, N};
  // if (count > 0) {
  //   shape[1] = count / M;
  // }
  if constexpr (dim == -1) {
    AscendC::ReduceMin<T, AscendC::Pattern::Reduce::AR>(
        dstTensor, srcTensor, sharedTmpBuffer, shape, true
    );
  } else {
    AscendC::ReduceMin<T, AscendC::Pattern::Reduce::RA>(
        dstTensor, srcTensor, sharedTmpBuffer, shape, true
    );
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
      tl::ascend::copy_l1_to_l0a<T1, M, K>(
        l0a[l0a_base], A[kL0Idx * M * kL0Size], M, kSize);
    } else {
      tl::ascend::copy_l1_to_l0a<T1, K, M, true>(
        l0a[l0a_base], A[kL0Idx * 16 * kL0Size], M, kSize);
    }
    if constexpr (!transpose_B) {
      tl::ascend::copy_l1_to_l0b<T1, K, N>(
        l0b[l0b_base], B[kL0Idx * 16 * kL0Size], kSize, N);
    } else {
      tl::ascend::copy_l1_to_l0b<T1, N, K, true>(
        l0b[l0b_base], B[kL0Idx * N * kL0Size], kSize, N);
    }
    SetFlag<HardEvent::MTE1_M>(L0AB_EVENT + pp);
    WaitFlag<HardEvent::MTE1_M>(L0AB_EVENT + pp);
    PipeBarrier<PIPE_M>();
    tl::ascend::mma<T1, T2, M, N>(
      l0a[l0a_base], l0b[l0b_base], C, initflag, kSize);
    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT + pp);
  }
  WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT);
  WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT + 1);

  SetFlag<HardEvent::MTE1_MTE2>(L0AB_EVENT);
  WaitFlag<HardEvent::MTE1_MTE2>(L0AB_EVENT);
  SetFlag<HardEvent::M_FIX>(L0AB_EVENT);
  WaitFlag<HardEvent::M_FIX>(L0AB_EVENT);
}


template <typename T>
CATLASS_DEVICE void MergeSort(const LocalTensor<T> &dst,
                              const LocalTensor<T> &src, uint32_t blockSize,
                              uint32_t blockNum, uint32_t is_copy) {
  // 初始化合并排序参数
  AscendC::MrgSort4Info params;
  params.elementLengths[0] = blockSize;
  params.elementLengths[1] = blockSize;
  params.elementLengths[2] = blockSize;
  params.elementLengths[3] = blockSize;
  params.ifExhaustedSuspension = false;
  params.validBit = 0b1111;

  // 初始化源列表
  AscendC::MrgSortSrcList<T> srcList;
  srcList.src1 = src[0];
  srcList.src2 = src[blockSize * 2 * 1];
  srcList.src3 = src[blockSize * 2 * 2];
  srcList.src4 = src[blockSize * 2 * 3];
  // 执行合并排序
  AscendC::MrgSort<T>(dst, srcList, params);
  PipeBarrier<PIPE_V>();
  if (is_copy) {
    AscendC::DataCopy(src, dst, blockNum * blockSize * 2);
  }
}

template <typename T>
CATLASS_DEVICE void TopK(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                         const LocalTensor<T> &tmp, uint32_t blockSize) {
  // 初始化合并排序参数
  AscendC::MrgSort4Info params;
  params.elementLengths[0] = blockSize;
  params.elementLengths[1] = blockSize;
  params.ifExhaustedSuspension = true;
  params.validBit = 0b0011;
  // 初始化源列表
  AscendC::MrgSortSrcList<T> srcList;
  srcList.src1 = dst;
  srcList.src2 = src;
  // 执行合并排序
  AscendC::MrgSort<T>(tmp, srcList, params);
  // PipeBarrier<PIPE_V>();
  // 将结果复制到目标张量
  AscendC::DataCopy(dst, tmp, blockSize * 2);
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
  uint64_t rsvdCnt = 0;    // 用于保存筛选后保留下来的元素个数
  GatherMask(dst, sortedTensor, src1Pattern,
             false, static_cast<uint32_t>(0), gatherMaskParams, rsvdCnt);
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
  uint64_t rsvdCnt = 0;    // 用于保存筛选后保留下来的元素个数
  GatherMask(dst, sortedTensor, src1Pattern,
             false, static_cast<uint32_t>(0), gatherMaskParams, rsvdCnt);
}

template <typename T>
CATLASS_DEVICE void Gather(const LocalTensor<T> &dst,
                           const LocalTensor<T> &sortedTensor,
                           const LocalTensor<uint32_t> &src1Pattern) {
  
  int32_t count = src1Pattern.GetSize();
  int32_t scalarValue = sizeof(T);
  LocalTensor<int32_t> offset = const_cast<LocalTensor<uint32_t>&>(src1Pattern).template ReinterpretCast<int32_t>();
  AscendC::Muls(offset, offset, scalarValue, count);
  AscendC::Gather(dst, sortedTensor, offset.template ReinterpretCast<uint32_t>(),
                  static_cast<uint32_t>(0), static_cast<uint32_t>(count));
}

template <typename T>
CATLASS_DEVICE void Gatherb(const LocalTensor<T> &dst,
                            const LocalTensor<T> &src0,
                            const LocalTensor<uint32_t> &offset,
                            uint8_t repeat_time,
                            uint8_t dst_blk_stride,
                            uint8_t dst_rep_stride) {
  GatherRepeatParams gatherRepeatParams;
  gatherRepeatParams.dstBlkStride = dst_blk_stride;
  gatherRepeatParams.dstRepStride = dst_rep_stride;
  Gatherb(dst.template ReinterpretCast<uint32_t>(),
          src0.template ReinterpretCast<uint32_t>(),
          offset.template ReinterpretCast<uint32_t>(),
          repeat_time, gatherRepeatParams);
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

template <typename T1, typename T2, uint32_t L1_block_M, uint32_t L1_block_N,
          uint32_t L1_block_K, uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t BLOCK_K, bool transpose_A = false, bool transpose_B = false>
CATLASS_DEVICE void
gemm_v1(LocalTensor<T1> const &A, LocalTensor<T1> const &B,
        LocalTensor<T2> const &C, // this must be located in l0c
        AscendC::TBuf<AscendC::TPosition::A2> &l0a_,
        AscendC::TBuf<AscendC::TPosition::B2> &l0b_, bool clear) {
  auto l0a = l0a_.Get<T1>();
  auto l0b = l0b_.Get<T1>();
  AscendC::PipeBarrier<PIPE_ALL>();

  if constexpr (!transpose_A) {
    tl::ascend::copy_l1_to_l0a<half, L1_block_M, L1_block_K>
                               (l0a, A, BLOCK_M, BLOCK_K);
  } else {
    tl::ascend::copy_l1_to_l0a<half, L1_block_K, L1_block_M, true>
                               (l0a, A, BLOCK_M, BLOCK_K);
  }

  if constexpr (!transpose_B) {
    tl::ascend::copy_l1_to_l0b<half, L1_block_K, L1_block_N>
                               (l0b, B, BLOCK_K, BLOCK_N);
  } else {
    tl::ascend::copy_l1_to_l0b<half, L1_block_N, L1_block_K, true>
                               (l0b, B, BLOCK_K, BLOCK_N);
  }

  AscendC::PipeBarrier<PIPE_ALL>();
tl:
  ascend::mma<T1, T2, BLOCK_M, BLOCK_N>(l0a, l0b, C, clear, BLOCK_K);
  AscendC::PipeBarrier<PIPE_ALL>();
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
CATLASS_DEVICE void Broadcast(const LocalTensor<T> &dst, const LocalTensor<T> &src, LocalTensor<uint8_t> &sharedTmpBuffer,
                              const uint32_t dstShape[dim], const uint32_t srcShape[dim]) {
  AscendC::Broadcast<T, dim, axis, isReuseSource>(dst, src, dstShape, srcShape, sharedTmpBuffer);
}

template <typename T>
CATLASS_DEVICE void Fill(const LocalTensor<T>& dst, const T& scalarValue, const int32_t& count) {
  AscendC::Duplicate<T>(dst, scalarValue, count);
}

template <typename T>
CATLASS_DEVICE void ArithProgression(const LocalTensor<T> &dst, const T firstValue,
                                      const T diffValue, const int32_t count) {
  AscendC::ArithProgression<T>(dst, firstValue, diffValue, count);
}

template <typename T, bool isFullSort>
CATLASS_DEVICE void Sort(const LocalTensor<T> &dst, const LocalTensor<T> &concat,
                          const LocalTensor<uint32_t> &index, LocalTensor<T> &tmp,
                          const int32_t repeatTime) {
  AscendC::Sort<T, isFullSort>(dst, concat, index, tmp, repeatTime);
}


template <typename T>
CATLASS_DEVICE void ClampMax(const LocalTensor<T> &dst, const LocalTensor<T> &buffer, const LocalTensor<uint8_t> &tmp,
                             const T scalarValue, const int32_t count) {
  AscendC::ClampMax<T>(dst, buffer, tmp, scalarValue, count);
}

template <typename T>
CATLASS_DEVICE void ClampMin(const LocalTensor<T> &dst, const LocalTensor<T> &buffer, const LocalTensor<uint8_t> &tmp,
                             const T scalarValue, const int32_t count) {
  AscendC::ClampMin<T>(dst, buffer, tmp, scalarValue, count);
}

template <typename T>
CATLASS_DEVICE void Clamp(const LocalTensor<T> &dst, const LocalTensor<T> &buffer, const LocalTensor<uint8_t> &tmp,
  const T minScalarValue, const T maxScalarValue, const int32_t count) {
    AscendC::ClampMin<T>(dst, buffer, tmp, minScalarValue, count);
    AscendC::ClampMax<T>(dst, dst, tmp, maxScalarValue, count);
}

template <typename T, typename U>
CATLASS_DEVICE void GatherMask_experiment(const LocalTensor<T> &dst,
                               const LocalTensor<T> &src0,
                               const LocalTensor<U> &src1Pattern, const bool reduceMode,
                               const uint32_t mask, const uint32_t src0BlockStride,
                               const uint32_t repeatTimes, uint32_t src0RepeatStride,
                               const uint32_t src1RepeatStride, uint64_t rsvdCnt) {
  GatherMaskParams gatherMaskParams;
  gatherMaskParams.repeatTimes = repeatTimes;
  gatherMaskParams.src0BlockStride = src0BlockStride;
  gatherMaskParams.src0RepeatStride = src0RepeatStride;
  gatherMaskParams.src1RepeatStride = src1RepeatStride;
  GatherMask(dst, src0, src1Pattern,
             reduceMode, mask, gatherMaskParams, rsvdCnt);
}

template <typename T>
CATLASS_DEVICE void Fill_experiment(const LocalTensor<T> &dst,
                               const T &scalarValue, uint64_t mask0,
                               const uint8_t repeatTime, const uint16_t dstBlockStride,
                               const uint8_t dstRepeatStride) {
  uint64_t mask[1] = {mask0};
  AscendC::Duplicate(dst, scalarValue, mask, repeatTime, dstBlockStride, dstRepeatStride);
}

template <typename T>
CATLASS_DEVICE void Sum_experiment(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                               const uint32_t outter, const uint32_t inner, const uint32_t n) {
  SumParams sumParams;
  sumParams.outter = outter;
  sumParams.inner = inner;
  sumParams.n = n;
  AscendC::Sum(dst, src, sumParams);
}

template <typename T, uint32_t M, uint32_t N>
CATLASS_DEVICE void transpose_16x16(LocalTensor<T> const &dst, LocalTensor<T> const &src) {
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

  __ubuf__ T* dstList[16];
  __ubuf__ T* srcList[16];

  if constexpr (sizeof(T) == 4) {
    for (int32_t m = 0; m < 16; m = m + 2) {
      dstList[m] = (__ubuf__ T*)dst[16 * (m / 2)].GetPhyAddr();
      dstList[m + 1] = (__ubuf__ T*)dst[16 * (m / 2) + 16].GetPhyAddr();
    }
    for (int32_t n = 0; n < 16; n++) {
      srcList[n] = (__ubuf__ T*)src[n * 16].GetPhyAddr();
    }
  } else {
    for (int i = 0; i < 16; i++) {
      dstList[i] = (__ubuf__ T*)dst[i * N].GetPhyAddr();
      srcList[i] = (__ubuf__ T*)src[i * M].GetPhyAddr();
    }
  }

  AscendC::TransDataTo5HDImpl<T>(dstList, srcList, transDataParams);
  AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
CATLASS_DEVICE void transpose(LocalTensor<T> const &dst, LocalTensor<T> const &src) {
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