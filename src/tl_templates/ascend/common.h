#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"

#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"

#define CUDART_INF_F 1.0f / 0.0f


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

template <typename T, uint32_t dstM, uint32_t dstN>
CATLASS_DEVICE void copy_gm_to_l1(LocalTensor<T> dstTensor,
                                  GlobalTensor<T> srcTensor, uint32_t realSrcN = 1) {
  auto layout = MakeLayoutFromTag(LayoutGM{dstM, realSrcN});
  auto src_LAYOUT = MakeLayoutTile(layout, tla::MakeShape(dstM, dstN));
  auto src = tla::MakeTensor<decltype(srcTensor), decltype(src_LAYOUT),
                             AscendC::TPosition::GM>(srcTensor, src_LAYOUT);

  using LayoutL1_ = Catlass::detail::TagToLayout_t<T, LayoutL1>;
  constexpr auto layoutInL1 = tla::MakeLayout<T, LayoutL1_>(dstM, dstN);
  auto dst = tla::MakeTensor<decltype(dstTensor), decltype(layoutInL1),
                             AscendC::TPosition::A1>(dstTensor, layoutInL1);

  TileCopyTla<ArchTag, decltype(src), decltype(dst)> tileCopier; 
  tileCopier(dst, src);
}

template <typename T, typename LayoutL1, uint32_t srcM, uint32_t srcN>
CATLASS_DEVICE void copy_l1_to_l0a(LocalTensor<T> dstTensor,
                                   LocalTensor<T> srcTensor,
                                   uint32_t dstM, uint32_t dstN) {
  using LayoutL1_ = Catlass::detail::TagToLayout_t<T, LayoutL1>;
  constexpr auto layout = tla::MakeLayout<T, LayoutL1_>(srcM, srcN);
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

template <typename T, typename LayoutL1, uint32_t srcM, uint32_t srcN>
CATLASS_DEVICE void copy_l1_to_l0b(LocalTensor<T> dstTensor,
                                   LocalTensor<T> srcTensor,
                                   uint32_t dstM, uint32_t dstN) {
  using LayoutL1_ = Catlass::detail::TagToLayout_t<T, LayoutL1>;
  constexpr auto LAYOUT = tla::MakeLayout<T, LayoutL1_>(srcM, srcN);
  auto src_LAYOUT = MakeLayoutTile(LAYOUT, tla::MakeShape(dstM, dstN));
  ;

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
                                   uint32_t realDstN = 1) {
  auto layoutInL0C = tla::MakeLayoutL0C(srcM, srcN); 
  auto src = tla::MakeTensor<decltype(srcTensor), decltype(layoutInL0C),
                             AscendC::TPosition::CO1>(srcTensor, layoutInL0C);
  LayoutGM gm{srcM, realDstN};
  auto layout = MakeLayoutFromTag(gm);
  auto dTensor = MakeTensor(dstTensor, layout, Arch::PositionGM{});
  auto layout_ = dTensor.layout();
  auto dst_LAYOUT = MakeLayoutTile(layout_, tla::MakeShape(srcM, srcN));
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
                  (std::is_same_v<T1, half> && std::is_same_v<T2, int8_t>)) {
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

template <typename T, uint32_t Len>
CATLASS_DEVICE void fill(LocalTensor<T> const &ubOut, T value) {
  AscendC::Duplicate(ubOut, value, Len);
}

template <typename T, uint32_t M, uint32_t N, class pattern>
CATLASS_DEVICE void reduce_sum(LocalTensor<T> const &dstTensor,
                               LocalTensor<T> const &srcTensor,
                               LocalTensor<uint8_t> const &sharedTmpBuffer) {
  uint32_t shape[] = {M, N};
  AscendC::ReduceSum<T, pattern>(dstTensor, srcTensor, sharedTmpBuffer, shape,
                                 true);
}

template <typename T, uint32_t M, uint32_t N, class pattern>
CATLASS_DEVICE void reduce_max(LocalTensor<T> const &dstTensor,
                               LocalTensor<T> const &srcTensor,
                               LocalTensor<uint8_t> const &sharedTmpBuffer) {
  uint32_t shape[] = {M, N};
  AscendC::ReduceMax<T, pattern>(dstTensor, srcTensor, sharedTmpBuffer, shape,
                                 true);
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
  uint32_t kL0Size = 128;
  uint32_t kL0split = (K + kL0Size - 1) / kL0Size;
  uint32_t kL0Tail = K - (kL0split - 1) * kL0Size;
  bool initflag = false;

  // Defensive Programming: Ensure all previous operations are complete
  SetFlag<HardEvent::MTE2_MTE1>(L0AB_EVENT);
  WaitFlag<HardEvent::MTE2_MTE1>(L0AB_EVENT);
  SetFlag<HardEvent::FIX_M>(L0AB_EVENT);
  WaitFlag<HardEvent::FIX_M>(L0AB_EVENT);

  SetFlag<HardEvent::M_MTE1>(L0AB_EVENT);
  SetFlag<HardEvent::M_MTE1>(L0AB_EVENT + 1);
  for (uint32_t kL0Idx = 0; kL0Idx < kL0split; kL0Idx++) {
    initflag = (clear && (kL0Idx == 0));
    if (kL0Idx == kL0split - 1) {
        kL0Size = kL0Tail;
    }
    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT + kL0Idx % 2);
    if constexpr (!transpose_A) {
      tl::ascend::copy_l1_to_l0a<T1, layout::zN, M, K>(
        l0a[(kL0Idx % 2) * (M * kL0Size)], A[kL0Idx * M * kL0Size], M, kL0Size);
    } else {
      tl::ascend::copy_l1_to_l0a<T1, layout::nZ, M, K>(
        l0a[(kL0Idx % 2) * (M * kL0Size)], A[kL0Idx * 16 * kL0Size], M, kL0Size);
    }
    if constexpr (!transpose_B) {
      tl::ascend::copy_l1_to_l0b<T1, layout::zN, K, N>(
        l0b[(kL0Idx % 2) * (N * kL0Size)], B[kL0Idx * 16 * kL0Size], kL0Size, N);
    } else {
      tl::ascend::copy_l1_to_l0b<T1, layout::nZ, K, N>(
        l0b[(kL0Idx % 2) * (N * kL0Size)], B[kL0Idx * N * kL0Size], kL0Size, N);
    }
    SetFlag<HardEvent::MTE1_M>(L0AB_EVENT + kL0Idx % 2);
    WaitFlag<HardEvent::MTE1_M>(L0AB_EVENT + kL0Idx % 2);
    tl::ascend::mma<T1, T2, M, N>(
      l0a[(kL0Idx % 2) * (M * kL0Size)], l0b[(kL0Idx % 2) * (N * kL0Size)], C, initflag, kL0Size);
    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT + kL0Idx % 2);
  }
  WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT);
  WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT + 1);
  
  // Defensive Programming: Reverse Sync
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
                               int64_t extractNum) {
  GatherMaskParams gatherMaskParams;
  gatherMaskParams.repeatTimes = Ceil(extractNum * sizeof(float) * 2, 256);
  gatherMaskParams.src0BlockStride = 1;
  gatherMaskParams.src0RepeatStride = 8;
  gatherMaskParams.src1RepeatStride = 0;
  uint64_t rsvdCnt = 0;    // 用于保存筛选后保留下来的元素个数
  uint8_t src1Pattern = 2; // 内置固定模式
  GatherMask(dst.template ReinterpretCast<uint32_t>(),
             sortedTensor.template ReinterpretCast<uint32_t>(), src1Pattern,
             false, static_cast<uint32_t>(0), gatherMaskParams, rsvdCnt);
  PipeBarrier<PIPE_V>();
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
    tl::ascend::copy_l1_to_l0a<half, layout::zN, L1_block_M, L1_block_K>
                               (l0a, A, BLOCK_M, BLOCK_K);
  } else {
    tl::ascend::copy_l1_to_l0a<half, layout::nZ, L1_block_M, L1_block_K>
                               (l0a, A, BLOCK_M, BLOCK_K);
  }

  if constexpr (!transpose_B) {
    tl::ascend::copy_l1_to_l0b<half, layout::zN, L1_block_K, L1_block_N>
                               (l0b, B, BLOCK_K, BLOCK_N);
  } else {
    tl::ascend::copy_l1_to_l0b<half, layout::nZ, L1_block_K, L1_block_N>
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

    copy_l1_to_l0a<T1, LayOutL1, M, K, baseM, baseK>(A2, A[loopM * baseM * 16]);

    for (uint32_t loopN = 0; loopN < N / baseN; loopN++) {
      copy_l1_to_l0b<T1, LayOutL1, K, N, baseK, baseN>(B2,
                                                       B[loopN * baseN * K]);

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
} // namespace tl::ascend