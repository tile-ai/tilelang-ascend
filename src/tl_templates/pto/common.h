#include <pto/pto-inst.hpp>
#include <type_traits>

#ifdef __CCE_AICORE__
#define CUDART_INF_F 1.0f / 0.0f

#ifdef PTO_PLATFORM_A5
#define TL_PIPE_V_BARRIER() ((void)0)
#else
#define TL_PIPE_V_BARRIER() pipe_barrier(PIPE_V)
#endif

namespace tl::ascend_pto {

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols>
using TileMatL1 = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                            pto::BLayout::ColMajor, RowValid, ColValid,
                            pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols>
using TileMatL1ZN = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                              pto::BLayout::RowMajor, RowValid, ColValid,
                              pto::SLayout::ColMajor, 512, pto::PadValue::Zero>;

#ifdef PTO_PLATFORM_A5
template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols>
using TileMatL0A = pto::Tile<pto::TileType::Left, T, Rows, Cols,
                             pto::BLayout::ColMajor, RowValid, ColValid,
                             pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;
#else
template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols>
using TileMatL0A = pto::Tile<pto::TileType::Left, T, Rows, Cols,
                             pto::BLayout::RowMajor, RowValid, ColValid,
                             pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;
#endif

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols>
using TileMatL0B = pto::Tile<pto::TileType::Right, T, Rows, Cols,
                             pto::BLayout::RowMajor, RowValid, ColValid,
                             pto::SLayout::ColMajor, 512, pto::PadValue::Zero>;

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols, pto::PadValue PadVal = pto::PadValue::Null>
using TileUbDataND =
    pto::Tile<pto::TileType::Vec, T, Rows, Cols, pto::BLayout::RowMajor,
              RowValid, ColValid, pto::SLayout::NoneBox, 512, PadVal>;

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols, pto::PadValue PadVal = pto::PadValue::Null>
using TileUbDataDN =
    pto::Tile<pto::TileType::Vec, T, Rows, Cols, pto::BLayout::ColMajor,
              RowValid, ColValid, pto::SLayout::NoneBox, 512, PadVal>;

template <typename T, int32_t shape>
AICORE PTO_INLINE void mov_tile(int32_t src_addr, int32_t dst_addr,
                                int32_t src_offset, int32_t dst_offset,
                                int32_t len) {
  // TileUbDataND<float, 1, shape> src_temp_ub(1, shape);
  TileUbDataND<T, 1, shape, 1, shape> src_temp_ub;
  pto::TASSIGN(src_temp_ub, src_addr + src_offset * len);
  TileUbDataND<T, 1, shape, 1, shape> dst_temp_ub;
  pto::TASSIGN(dst_temp_ub, dst_addr + dst_offset * len);
  pto::TMOV(dst_temp_ub, src_temp_ub);
}

template <typename T1, typename T2, int32_t shape>
AICORE PTO_INLINE void cvt_tile(int32_t src_addr, int32_t dst_addr,
                                int32_t src_offset, int32_t dst_offset,
                                int32_t src_len, int32_t dst_len,
                                pto::RoundMode rmode) {
  TileUbDataND<T1, 1, shape, 1, shape> src_temp_ub;
  pto::TASSIGN(src_temp_ub, src_addr + src_offset * src_len);
  TileUbDataND<T2, 1, shape, 1, shape> dst_temp_ub;
  pto::TASSIGN(dst_temp_ub, dst_addr + dst_offset * dst_len);
  pto::TCVT(dst_temp_ub, src_temp_ub, rmode);
}

template <typename T, uint32_t M, uint32_t N, uint32_t M_L1, uint32_t N_L1,
          bool transpose = false>
AICORE PTO_INLINE void copy_l1_to_l0a(
    TileMatL0A<T, M, N, M, N> &l0a,
    std::conditional_t<transpose, TileMatL1ZN<T, M_L1, N_L1, M_L1, N_L1>,
                       TileMatL1<T, M_L1, N_L1, M_L1, N_L1>> &A,
    uint32_t indexRow, uint32_t indexCol) {
  pto::TEXTRACT(l0a, A, indexRow, indexCol);
}

template <typename T, uint32_t M, uint32_t N, uint32_t M_L1, uint32_t N_L1,
          bool transpose = false>
AICORE PTO_INLINE void copy_l1_to_l0b(
    TileMatL0B<T, M, N, M, N> &l0b,
    std::conditional_t<transpose, TileMatL1ZN<T, M_L1, N_L1, M_L1, N_L1>,
                       TileMatL1<T, M_L1, N_L1, M_L1, N_L1>> &B,
    uint32_t indexRow, uint32_t indexCol) {
  pto::TEXTRACT(l0b, B, indexRow, indexCol);
}

template <typename T1, typename T2, int M, int N, int K, int validM = M,
          int validN = N>
AICORE PTO_INLINE void mma(TileMatL0A<T1, M, K> l0a, TileMatL0B<T1, K, N> l0b,
                           pto::TileAcc<T2, M, N, validM, validN> &C,
                           bool init) {
  if (init) {
    pto::TMATMUL(C, l0a, l0b);
  } else {
    pto::TMATMUL_ACC(C, C, l0a, l0b);
  }
}

template <typename T1, typename T2, uint32_t M, uint32_t N, uint32_t K,
          uint32_t validM, uint32_t validN, uint32_t validK, uint32_t CurrentK,
          bool transpose_A, bool transpose_B>
AICORE PTO_INLINE void gemm_v0_inner(
    std::conditional_t<transpose_A, TileMatL1<T1, K, M, validK, validM>,
                       TileMatL1<T1, M, K, validM, validK>> &A,
    std::conditional_t<transpose_B, TileMatL1<T1, N, K, validN, validK>,
                       TileMatL1<T1, K, N, validK, validN>> &B,
    pto::TileAcc<T2, M, N, validM, validN> &C, uint32_t kL0Idx, bool initflag,
    event_t war_event_id, bool is_tail_block) {
  TileMatL0A<T1, M, CurrentK, M, CurrentK> l0a;
  TileMatL0B<T1, CurrentK, N, CurrentK, N> l0b;
  pto::TASSIGN(l0a, 0x0);
  pto::TASSIGN(l0b, 0x0);

  set_flag(PIPE_M, PIPE_MTE1, war_event_id);
  wait_flag(PIPE_M, PIPE_MTE1, war_event_id);

  if (!is_tail_block) {
    set_flag(PIPE_FIX, PIPE_M, war_event_id);
    wait_flag(PIPE_FIX, PIPE_M, war_event_id);
  }

  if constexpr (!transpose_A) {
    copy_l1_to_l0a<T1, M, CurrentK, M, K, false>(l0a, A, 0, kL0Idx * CurrentK);
  } else {
    TileMatL1ZN<T1, M, K, validM, validK> A_t;
    pto::TRESHAPE(A_t, A);
    copy_l1_to_l0a<T1, M, CurrentK, M, K, true>(l0a, A_t, 0, kL0Idx * CurrentK);
  }
  if constexpr (!transpose_B) {
    copy_l1_to_l0b<T1, CurrentK, N, K, N, false>(l0b, B, kL0Idx * CurrentK, 0);
  } else {
    TileMatL1ZN<T1, K, N, validK, validN> B_t;
    pto::TRESHAPE(B_t, B);
    copy_l1_to_l0b<T1, CurrentK, N, K, N, true>(l0b, B_t, kL0Idx * CurrentK, 0);
  }

  set_flag(PIPE_MTE1, PIPE_M, war_event_id);
  wait_flag(PIPE_MTE1, PIPE_M, war_event_id);

  if (initflag) {
    pto::TMATMUL(C, l0a, l0b);
  } else {
    pto::TMATMUL_ACC(C, C, l0a, l0b);
  }

  if (!is_tail_block) {
    set_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
    wait_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
  }
}

template <typename T1, typename T2, uint32_t M, uint32_t N, uint32_t K,
          uint32_t validM = M, uint32_t validN = N, uint32_t validK = K,
          uint32_t K_tail, bool transpose_A = false, bool transpose_B = false>
AICORE PTO_INLINE void
gemm_v0(std::conditional_t<transpose_A, TileMatL1<T1, K, M, validK, validM>,
                           TileMatL1<T1, M, K, validM, validK>> &A,
        std::conditional_t<transpose_B, TileMatL1<T1, N, K, validN, validK>,
                           TileMatL1<T1, K, N, validK, validN>> &B,
        pto::TileAcc<T2, M, N, validM, validN> &C, bool clear) {
  constexpr uint32_t kL0Size =
      128; // L0 slice size, adapted to 64K memory limit
  const uint32_t kL0split = (K + kL0Size - 1) / kL0Size; // Number of slices
  auto war_event_id = (event_t)(((int)EVENT_ID0 + 1) % 8);

  set_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);
  wait_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);

  for (uint32_t kL0Idx = 0; kL0Idx < kL0split; kL0Idx++) {
    const bool initflag = (clear && (kL0Idx == 0));
    const bool is_tail_block = (kL0Idx == kL0split - 1);

    if (is_tail_block) {
      gemm_v0_inner<T1, T2, M, N, K, validM, validN, validK, K_tail,
                    transpose_A, transpose_B>(A, B, C, kL0Idx, initflag,
                                              war_event_id, true);
    } else {
      gemm_v0_inner<T1, T2, M, N, K, validM, validN, validK, kL0Size,
                    transpose_A, transpose_B>(A, B, C, kL0Idx, initflag,
                                              war_event_id, false);
    }
  }

  set_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
  wait_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);

  set_flag(PIPE_M, PIPE_FIX, war_event_id);
  wait_flag(PIPE_M, PIPE_FIX, war_event_id);
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2,
          int32_t shape3, int32_t shape4, int32_t shape5, int32_t stride1,
          int32_t stride2, int32_t stride3, int32_t stride4, int32_t stride5,
          uint32_t valid1, uint32_t valid2>
AICORE PTO_INLINE void copy_gm_to_l1_dynamic(
    __gm__ T1 *handle,
    const pto::Shape<shape1, shape2, shape3, shape4, shape5> &shape,
    const pto::Stride<stride1, stride2, stride3, stride4, stride5> &stride,
    int32_t buffer_addr, int32_t offset, int32_t actualTailM = 0,
    int32_t actualTailN = 0) {
  constexpr uint8_t len = sizeof(T2);
  bool useTail = shape4 == valid1 && shape5 == valid2;
  int tailM = (useTail && actualTailM != 0) ? actualTailM : valid1;
  int tailN = (useTail && actualTailN != 0) ? actualTailN : valid2;
  TileMatL1<T2, shape4, shape5, pto::DYNAMIC, pto::DYNAMIC> L1(tailM, tailN);
  pto::TASSIGN(L1, buffer_addr + offset * len);
  pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC> dynamic_shape;
  dynamic_shape.shape[3] = useTail ? tailM : shape4;
  dynamic_shape.shape[4] = useTail ? tailN : shape5;
  pto::GlobalTensor<
      T1, pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC>,
      pto::Stride<stride1, stride2, stride3, stride4, stride5>>
      global_tensor(handle, dynamic_shape, stride);
  pto::TLOAD(L1, global_tensor);
  if (useTail && (tailM != shape4 || tailN != shape5)) {
    pto::TFILLPAD(L1, L1);
  }
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2,
          int32_t shape3, int32_t shape4, int32_t shape5, int32_t stride1,
          int32_t stride2, int32_t stride3, int32_t stride4, int32_t stride5,
          uint32_t valid1, uint32_t valid2>
AICORE PTO_INLINE void copy_l0c_to_gm_dynamic(
    __gm__ T1 *handle,
    const pto::Shape<shape1, shape2, shape3, shape4, shape5> &shape,
    const pto::Stride<stride1, stride2, stride3, stride4, stride5> &stride,
    int32_t buffer_addr, int32_t offset, int32_t actualTailM = 0,
    int32_t actualTailN = 0) {
  constexpr uint8_t len = sizeof(T2);
  bool useTail = shape4 == valid1 && shape5 == valid2;
  int tailM = (useTail && actualTailM != 0) ? actualTailM : valid1;
  int tailN = (useTail && actualTailN != 0) ? actualTailN : valid2;
  pto::TileAcc<T2, shape4, shape5, pto::DYNAMIC, pto::DYNAMIC> L0c(tailM,
                                                                   tailN);
  pto::TASSIGN(L0c, buffer_addr + offset * len);
  pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC> dynamic_shape;
  dynamic_shape.shape[3] = useTail ? tailM : shape4;
  dynamic_shape.shape[4] = useTail ? tailN : shape5;
  pto::GlobalTensor<
      T1, pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC>,
      pto::Stride<stride1, stride2, stride3, stride4, stride5>>
      global_tensor(handle, dynamic_shape, stride);
  pto::TSTORE(global_tensor, L0c);
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2,
          int32_t shape3, int32_t shape4, int32_t shape5, int32_t stride1,
          int32_t stride2, int32_t stride3, int32_t stride4, int32_t stride5,
          uint32_t ub_shape1, uint32_t ub_shape2,
          pto::PadValue PadVal = pto::PadValue::Null>
AICORE PTO_INLINE void copy_gm_to_ub_dynamic(
    __gm__ T1 *handle,
    const pto::Shape<shape1, shape2, shape3, shape4, shape5> &shape,
    const pto::Stride<stride1, stride2, stride3, stride4, stride5> &stride,
    int32_t ub_shape_addr, int32_t ub_offset, int32_t valid_row,
    int32_t valid_col) {
  constexpr uint8_t len = sizeof(T2);
  pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC> dynamic_shape;
  dynamic_shape.shape[3] = valid_row;
  dynamic_shape.shape[4] = valid_col;
  pto::GlobalTensor<
      T1, pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC>,
      pto::Stride<stride1, stride2, stride3, stride4, stride5>>
      global_tensor(handle, dynamic_shape, stride);
  if constexpr (std::is_same_v<T1, T2>) {
    // Source Tile: dynamic valid, PadVal for TLOAD 32-byte alignment
    using SrcTile = TileUbDataND<T2, ub_shape1, ub_shape2, pto::DYNAMIC,
                                 pto::DYNAMIC, PadVal>;
    SrcTile src_tile(valid_row, valid_col);
    pto::TASSIGN(src_tile, ub_shape_addr + ub_offset * len);
    pto::TLOAD(src_tile, global_tensor);

    // TFILLPAD_INPLACE: fill outside valid region with PadVal (only for tail
    // blocks with valid PadVal)
    if constexpr (PadVal != pto::PadValue::Null) {
      if (valid_row != static_cast<int32_t>(ub_shape1) ||
          valid_col != static_cast<int32_t>(ub_shape2)) {
        using DstTile = pto::Tile<pto::TileType::Vec, T2, ub_shape1, ub_shape2,
                                  pto::BLayout::RowMajor, ub_shape1, ub_shape2,
                                  pto::SLayout::NoneBox, 512, PadVal>;
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        DstTile dst_tile;
        pto::TASSIGN(dst_tile, ub_shape_addr + ub_offset * len);
        pto::TFILLPAD_INPLACE(dst_tile, src_tile);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TL_PIPE_V_BARRIER();
      }
    }
  } else {
    TileUbDataND<T1, ub_shape1, ub_shape2, pto::DYNAMIC, pto::DYNAMIC>
        temp_src_ub(valid_row, valid_col);
    pto::TASSIGN(temp_src_ub, ub_shape_addr + ub_offset * len);
    pto::TLOAD(temp_src_ub, global_tensor);
    TileUbDataND<T2, ub_shape1, ub_shape2, pto::DYNAMIC, pto::DYNAMIC>
        temp_dst_ub(valid_row, valid_col);
    pto::TASSIGN(temp_dst_ub, ub_shape_addr + ub_offset * len);
    pto::TCVT(temp_dst_ub, temp_src_ub, pto::RoundMode::CAST_NONE);
  }
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2,
          int32_t shape3, int32_t shape4, int32_t shape5, int32_t stride1,
          int32_t stride2, int32_t stride3, int32_t stride4, int32_t stride5,
          uint32_t ub_shape1, uint32_t ub_shape2>
AICORE PTO_INLINE void copy_ub_to_gm_dynamic(
    __gm__ T1 *handle,
    const pto::Shape<shape1, shape2, shape3, shape4, shape5> &shape,
    const pto::Stride<stride1, stride2, stride3, stride4, stride5> &stride,
    int32_t ub_shape_addr, int32_t ub_offset, int32_t valid_row,
    int32_t valid_col) {
  pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC> dynamic_shape;
  dynamic_shape.shape[3] = valid_row;
  dynamic_shape.shape[4] = valid_col;
  pto::GlobalTensor<
      T1, pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC>,
      pto::Stride<stride1, stride2, stride3, stride4, stride5>>
      global_tensor(handle, dynamic_shape, stride);
  constexpr uint8_t len = sizeof(T2);
  constexpr bool use_nd = (static_cast<uint64_t>(ub_shape2) * len) >= 32;
  if constexpr (std::is_same_v<T1, T2>) {
    if constexpr (use_nd) {
      TileUbDataND<T2, ub_shape1, ub_shape2, pto::DYNAMIC, pto::DYNAMIC>
          temp_ub(valid_row, valid_col);
      pto::TASSIGN(temp_ub, ub_shape_addr + ub_offset * len);
      pto::TSTORE(global_tensor, temp_ub);
    } else {
      TileUbDataDN<T2, ub_shape1, ub_shape2, pto::DYNAMIC, pto::DYNAMIC>
          temp_ub(valid_row, valid_col);
      pto::TASSIGN(temp_ub, ub_shape_addr + ub_offset * len);
      pto::TSTORE(global_tensor, temp_ub);
    }
  } else {
    if constexpr (use_nd) {
      TileUbDataND<T2, ub_shape1, ub_shape2, pto::DYNAMIC, pto::DYNAMIC>
          temp_src_ub(valid_row, valid_col);
      pto::TASSIGN(temp_src_ub, ub_shape_addr + ub_offset * len);
      TileUbDataND<T1, ub_shape1, ub_shape2, pto::DYNAMIC, pto::DYNAMIC>
          temp_dst_ub(valid_row, valid_col);
      pto::TASSIGN(temp_dst_ub, ub_shape_addr + ub_offset * sizeof(T1));
      pto::TCVT(temp_dst_ub, temp_src_ub, pto::RoundMode::CAST_NONE);
      pto::TSTORE(global_tensor, temp_dst_ub);
    } else {
      TileUbDataDN<T2, ub_shape1, ub_shape2, pto::DYNAMIC, pto::DYNAMIC>
          temp_src_ub(valid_row, valid_col);
      pto::TASSIGN(temp_src_ub, ub_shape_addr + ub_offset * len);
      TileUbDataDN<T1, ub_shape1, ub_shape2, pto::DYNAMIC, pto::DYNAMIC>
          temp_dst_ub(valid_row, valid_col);
      pto::TASSIGN(temp_dst_ub, ub_shape_addr + ub_offset * sizeof(T1));
      pto::TCVT(temp_dst_ub, temp_src_ub, pto::RoundMode::CAST_NONE);
      pto::TSTORE(global_tensor, temp_dst_ub);
    }
  }
}

enum class BinaryOp { TADD, TSUB, TMUL, TDIV, TMAX, TMIN, TAND, TOR };

template <BinaryOp Op, typename T, int32_t shape>
AICORE PTO_INLINE void binary_tile(int32_t dst_addr, int32_t src0_addr,
                                   int32_t src1_addr, int32_t dst_offset,
                                   int32_t src0_offset, int32_t src1_offset,
                                   int32_t len) {
  // TileUbDataND<T, 1, shape> src0_temp_ub(1, shape);
  TileUbDataND<T, 1, shape, 1, shape> src0_temp_ub;

  pto::TASSIGN(src0_temp_ub, src0_addr + src0_offset * len);
  // TileUbDataND<T, 1, shape> src1_temp_ub(1, shape);
  TileUbDataND<T, 1, shape, 1, shape> src1_temp_ub;

  pto::TASSIGN(src1_temp_ub, src1_addr + src1_offset * len);
  // TileUbDataND<T, 1, shape> dst_temp_ub(1, shape);
  TileUbDataND<T, 1, shape, 1, shape> dst_temp_ub;

  pto::TASSIGN(dst_temp_ub, dst_addr + dst_offset * len);
  if constexpr (Op == BinaryOp::TADD) {
    pto::TADD(dst_temp_ub, src0_temp_ub, src1_temp_ub);
  } else if constexpr (Op == BinaryOp::TSUB) {
    pto::TSUB(dst_temp_ub, src0_temp_ub, src1_temp_ub);
  } else if constexpr (Op == BinaryOp::TMUL) {
    pto::TMUL(dst_temp_ub, src0_temp_ub, src1_temp_ub);
  } else if constexpr (Op == BinaryOp::TDIV) {
    pto::TDIV(dst_temp_ub, src0_temp_ub, src1_temp_ub);
  } else if constexpr (Op == BinaryOp::TMAX) {
    pto::TMAX(dst_temp_ub, src0_temp_ub, src1_temp_ub);
  } else if constexpr (Op == BinaryOp::TMIN) {
    pto::TMIN(dst_temp_ub, src0_temp_ub, src1_temp_ub);
  } else if constexpr (Op == BinaryOp::TAND) {
    pto::TAND(dst_temp_ub, src0_temp_ub, src1_temp_ub);
  } else if constexpr (Op == BinaryOp::TOR) {
    pto::TOR(dst_temp_ub, src0_temp_ub, src1_temp_ub);
  }
}

enum class UnaryOp { TEXP, TLOG, TABS, TRECIP, TSQRT, TRSQRT, TRELU, TNOT };

template <UnaryOp Op, typename T, int32_t shape>
AICORE PTO_INLINE void unary_tile(int32_t dst_addr, int32_t src_addr,
                                  int32_t dst_offset, int32_t src_offset,
                                  int32_t len) {
  TileUbDataND<T, 1, shape, 1, shape> src_temp_ub;
  pto::TASSIGN(src_temp_ub, src_addr + src_offset * len);

  TileUbDataND<T, 1, shape, 1, shape> dst_temp_ub;
  pto::TASSIGN(dst_temp_ub, dst_addr + dst_offset * len);

  if constexpr (Op == UnaryOp::TEXP) {
    pto::TEXP(dst_temp_ub, src_temp_ub);
  } else if constexpr (Op == UnaryOp::TLOG) {
    pto::TLOG(dst_temp_ub, src_temp_ub);
  } else if constexpr (Op == UnaryOp::TABS) {
    pto::TABS(dst_temp_ub, src_temp_ub);
  } else if constexpr (Op == UnaryOp::TRECIP) {
    pto::TRECIP(dst_temp_ub, src_temp_ub);
  } else if constexpr (Op == UnaryOp::TSQRT) {
    pto::TSQRT(dst_temp_ub, src_temp_ub);
  } else if constexpr (Op == UnaryOp::TRSQRT) {
    pto::TRSQRT(dst_temp_ub, src_temp_ub);
  } else if constexpr (Op == UnaryOp::TRELU) {
    pto::TRELU(dst_temp_ub, src_temp_ub);
  } else if constexpr (Op == UnaryOp::TNOT) {
    pto::TNOT(dst_temp_ub, src_temp_ub);
  }
}

template <typename T, int32_t row, int32_t col>
AICORE PTO_INLINE void TSIGMOID(TileUbDataND<T, row, col, row, col> &dst,
                                TileUbDataND<T, row, col, row, col> &src0) {
  TMULS(src0, src0, -1);
  TL_PIPE_V_BARRIER();
  TEXP(src0, src0);
  TL_PIPE_V_BARRIER();
  TADDS(src0, src0, 1);
  TL_PIPE_V_BARRIER();
  TRECIP(dst, src0);
}

template <typename T, int32_t row, int32_t col>
AICORE PTO_INLINE void TSILU(TileUbDataND<T, row, col, row, col> &dst,
                             TileUbDataND<T, row, col, row, col> &src,
                             TileUbDataND<T, row, col, row, col> &tmp) {
  TMOV(tmp, src);
  TL_PIPE_V_BARRIER();
  TSIGMOID(dst, src);
  TL_PIPE_V_BARRIER();
  TMUL(dst, tmp, dst);
}

template <typename T, int32_t row, int32_t col>
AICORE PTO_INLINE void MulAddDst(TileUbDataND<T, row, col, row, col> &dst,
                                 TileUbDataND<T, row, col, row, col> &src0,
                                 TileUbDataND<T, row, col, row, col> &src1,
                                 TileUbDataND<T, row, col, row, col> &tmp) {
  TMUL(tmp, src0, src1);
  TL_PIPE_V_BARRIER();
  TADD(dst, dst, tmp);
}

template <typename T, int32_t row, int32_t col>
AICORE PTO_INLINE void axpy(TileUbDataND<T, row, col, row, col> &dst,
                            TileUbDataND<T, row, col, row, col> &src0,
                            float scalar_value) {
  pto::TAXPY(dst, src0, static_cast<T>(scalar_value));
}

template <typename TileType, typename DataType>
void TCI(TileType &tile, DataType firstValue);

template <typename T, int32_t row, int32_t col>
AICORE PTO_INLINE void tci(int32_t ub_addr, int32_t ub_offset, int32_t len,
                           T firstValue) {
  using TileData = TileUbDataND<T, row, col, row, col>;
  TileData temp_ub;
  TASSIGN(temp_ub, ub_addr + ub_offset * len);
  TCI<TileData, T, 0>(temp_ub, firstValue);
}

template <typename T> struct is_float_or_half : std::false_type {};

template <> struct is_float_or_half<float> : std::true_type {};

template <> struct is_float_or_half<half> : std::true_type {};

template <typename T, int32_t row, int32_t col>
AICORE PTO_INLINE typename std::enable_if<is_float_or_half<T>::value>::type
pow(TileUbDataND<T, row, col, row, col> &dst,
    TileUbDataND<T, row, col, row, col> &src0,
    TileUbDataND<T, row, col, row, col> &src1) {
  TLOG(src0, src0);
  TL_PIPE_V_BARRIER();
  TMUL(dst, src0, src1);
  TL_PIPE_V_BARRIER();
  TEXP(dst, dst);
}

enum class BinaryOps { TADDS, TSUBS, TMULS, TDIVS, TMAXS, TMINS };

template <BinaryOps Op, typename T, int32_t dst_shape, int32_t src_shape>
AICORE PTO_INLINE void binarys_tile(int32_t dst_addr, int32_t src_addr,
                                    int32_t dst_offset, int32_t src_offset,
                                    int32_t len, T scalar_value) {
  TileUbDataND<T, 1, dst_shape, 1, dst_shape> dst_temp_ub;
  pto::TASSIGN(dst_temp_ub, dst_addr + dst_offset * len);
  TileUbDataND<T, 1, src_shape, 1, src_shape> src_temp_ub;
  pto::TASSIGN(src_temp_ub, src_addr + src_offset * len);
  if constexpr (Op == BinaryOps::TADDS) {
    pto::TADDS(dst_temp_ub, src_temp_ub, scalar_value);
  } else if constexpr (Op == BinaryOps::TSUBS) {
    pto::TSUBS(dst_temp_ub, src_temp_ub, scalar_value);
  } else if constexpr (Op == BinaryOps::TMULS) {
    pto::TMULS(dst_temp_ub, src_temp_ub, scalar_value);
  } else if constexpr (Op == BinaryOps::TDIVS) {
    pto::TDIVS(dst_temp_ub, src_temp_ub, scalar_value);
  } else if constexpr (Op == BinaryOps::TMAXS) {
    pto::TMAXS(dst_temp_ub, src_temp_ub, scalar_value);
  } else if constexpr (Op == BinaryOps::TMINS) {
    pto::TMINS(dst_temp_ub, src_temp_ub, scalar_value);
  }
}

template <pipe_t pipe, pipe_t tpipe>
AICORE PTO_INLINE void set_flag_pipeline(int32_t pipeID) {
  if (pipeID >= 0 && pipeID <= 7) {
    set_flag(pipe, tpipe, static_cast<event_t>(EVENT_ID0 + pipeID));
  }
}

template <pipe_t pipe, pipe_t tpipe>
AICORE PTO_INLINE void wait_flag_pipeline(int32_t pipeID) {
  if (pipeID >= 0 && pipeID <= 7) {
    wait_flag(pipe, tpipe, static_cast<event_t>(EVENT_ID0 + pipeID));
  }
}

template <typename dstT, int32_t dstRow, int32_t dstCol, int32_t dstRowValid,
          int32_t dstColValid, typename srcT, int32_t srcRow, int32_t srcCol,
          int32_t srcRowValid, int32_t srcColValid, int32_t src_element_count>
AICORE PTO_INLINE void TROWEXPAND_with_slice_buffer(
    TileUbDataND<dstT, dstRow, dstCol, dstRow, dstCol> dst,
    TileUbDataDN<srcT, srcRow, srcCol, srcRow, srcCol> src, int32_t src_addr,
    int32_t src_offset) {
  TileUbDataDN<srcT, src_element_count, srcCol, src_element_count, srcColValid>
      src_temp_ub;
  pto::TASSIGN(src_temp_ub, src_addr + src_offset);

  pto::TROWEXPAND(dst, src_temp_ub);
}
template <pipe_t pipe>
AICORE PTO_INLINE void set_cross_flag(int32_t flag, int32_t mode) {
  int config = 1 | (mode << 4) | (flag << 8);
  ffts_cross_core_sync(pipe, config);
}

template <pipe_t pipe>
AICORE PTO_INLINE void set_intra_block_cube(int32_t flag) {
  set_intra_block(pipe, flag);
  set_intra_block(pipe, flag + 16);
}

template <pipe_t pipe>
AICORE PTO_INLINE void set_intra_block_vec(int32_t flag) {
  set_intra_block(pipe, flag);
}

AICORE PTO_INLINE void wait_cross_flag(int32_t flag) { wait_flag_dev(flag); }

template <pipe_t pipe>
AICORE PTO_INLINE void wait_intra_block_cube(int32_t flag) {
  wait_intra_block(pipe, flag);
  wait_intra_block(pipe, flag + 16);
}

template <pipe_t pipe>
AICORE PTO_INLINE void wait_intra_block_vec(int32_t flag) {
  wait_intra_block(pipe, flag);
}

// ============================================================================
// Merge Sort for PTO backend
// tmp buffer is passed from caller, MrgSortExecutedNumList is managed
// internally Each element is a value-index pair: 2 floats per element [value,
// index]
// ============================================================================

// 2-way merge sort
template <typename T, int32_t SrcCols, int32_t DstCols>
AICORE PTO_INLINE void
MergeSort(TileUbDataND<T, 1, DstCols, 1, DstCols> &dst,
          TileUbDataND<T, 1, DstCols, 1, DstCols> &tmp,
          TileUbDataND<T, 1, SrcCols, 1, SrcCols> &src0,
          TileUbDataND<T, 1, SrcCols, 1, SrcCols> &src1) {

  pto::MrgSortExecutedNumList executedNumList;
  pto::TMRGSORT<TileUbDataND<T, 1, DstCols, 1, DstCols>,
                TileUbDataND<T, 1, DstCols, 1, DstCols>,
                TileUbDataND<T, 1, SrcCols, 1, SrcCols>,
                TileUbDataND<T, 1, SrcCols, 1, SrcCols>, false>(
      dst, executedNumList, tmp, src0, src1);
  TL_PIPE_V_BARRIER();
}

// 3-way merge sort
template <typename T, int32_t SrcCols, int32_t DstCols>
AICORE PTO_INLINE void
MergeSort(TileUbDataND<T, 1, DstCols, 1, DstCols> &dst,
          TileUbDataND<T, 1, DstCols, 1, DstCols> &tmp,
          TileUbDataND<T, 1, SrcCols, 1, SrcCols> &src0,
          TileUbDataND<T, 1, SrcCols, 1, SrcCols> &src1,
          TileUbDataND<T, 1, SrcCols, 1, SrcCols> &src2) {

  pto::MrgSortExecutedNumList executedNumList;
  pto::TMRGSORT<TileUbDataND<T, 1, DstCols, 1, DstCols>,
                TileUbDataND<T, 1, DstCols, 1, DstCols>,
                TileUbDataND<T, 1, SrcCols, 1, SrcCols>,
                TileUbDataND<T, 1, SrcCols, 1, SrcCols>,
                TileUbDataND<T, 1, SrcCols, 1, SrcCols>, false>(
      dst, executedNumList, tmp, src0, src1, src2);
  TL_PIPE_V_BARRIER();
}

// 4-way merge sort
template <typename T, int32_t SrcCols, int32_t DstCols>
AICORE PTO_INLINE void
MergeSort(TileUbDataND<T, 1, DstCols, 1, DstCols> &dst,
          TileUbDataND<T, 1, DstCols, 1, DstCols> &tmp,
          TileUbDataND<T, 1, SrcCols, 1, SrcCols> &src0,
          TileUbDataND<T, 1, SrcCols, 1, SrcCols> &src1,
          TileUbDataND<T, 1, SrcCols, 1, SrcCols> &src2,
          TileUbDataND<T, 1, SrcCols, 1, SrcCols> &src3) {

  pto::MrgSortExecutedNumList executedNumList;
  pto::TMRGSORT<TileUbDataND<T, 1, DstCols, 1, DstCols>,
                TileUbDataND<T, 1, DstCols, 1, DstCols>,
                TileUbDataND<T, 1, SrcCols, 1, SrcCols>,
                TileUbDataND<T, 1, SrcCols, 1, SrcCols>,
                TileUbDataND<T, 1, SrcCols, 1, SrcCols>,
                TileUbDataND<T, 1, SrcCols, 1, SrcCols>, false>(
      dst, executedNumList, tmp, src0, src1, src2, src3);
  TL_PIPE_V_BARRIER();
}

// 2-way merge sort with asymmetric source sizes (used by Sort recursion).
template <typename T, int32_t Src0Cols, int32_t Src1Cols, int32_t DstCols>
AICORE PTO_INLINE void
MergeSortVar(TileUbDataND<T, 1, DstCols, 1, DstCols> &dst,
             TileUbDataND<T, 1, DstCols, 1, DstCols> &tmp,
             TileUbDataND<T, 1, Src0Cols, 1, Src0Cols> &src0,
             TileUbDataND<T, 1, Src1Cols, 1, Src1Cols> &src1) {
  pto::MrgSortExecutedNumList executedNumList;
  pto::TMRGSORT<TileUbDataND<T, 1, DstCols, 1, DstCols>,
                TileUbDataND<T, 1, DstCols, 1, DstCols>,
                TileUbDataND<T, 1, Src0Cols, 1, Src0Cols>,
                TileUbDataND<T, 1, Src1Cols, 1, Src1Cols>, false>(
      dst, executedNumList, tmp, src0, src1);
  TL_PIPE_V_BARRIER();
}

// 3-way merge sort with asymmetric source sizes.
template <typename T, int32_t Src0Cols, int32_t Src1Cols, int32_t Src2Cols,
          int32_t DstCols>
AICORE PTO_INLINE void
MergeSortVar(TileUbDataND<T, 1, DstCols, 1, DstCols> &dst,
             TileUbDataND<T, 1, DstCols, 1, DstCols> &tmp,
             TileUbDataND<T, 1, Src0Cols, 1, Src0Cols> &src0,
             TileUbDataND<T, 1, Src1Cols, 1, Src1Cols> &src1,
             TileUbDataND<T, 1, Src2Cols, 1, Src2Cols> &src2) {
  pto::MrgSortExecutedNumList executedNumList;
  pto::TMRGSORT<TileUbDataND<T, 1, DstCols, 1, DstCols>,
                TileUbDataND<T, 1, DstCols, 1, DstCols>,
                TileUbDataND<T, 1, Src0Cols, 1, Src0Cols>,
                TileUbDataND<T, 1, Src1Cols, 1, Src1Cols>,
                TileUbDataND<T, 1, Src2Cols, 1, Src2Cols>, false>(
      dst, executedNumList, tmp, src0, src1, src2);
  TL_PIPE_V_BARRIER();
}

// 4-way merge sort with asymmetric source sizes.
template <typename T, int32_t Src0Cols, int32_t Src1Cols, int32_t Src2Cols,
          int32_t Src3Cols, int32_t DstCols>
AICORE PTO_INLINE void
MergeSortVar(TileUbDataND<T, 1, DstCols, 1, DstCols> &dst,
             TileUbDataND<T, 1, DstCols, 1, DstCols> &tmp,
             TileUbDataND<T, 1, Src0Cols, 1, Src0Cols> &src0,
             TileUbDataND<T, 1, Src1Cols, 1, Src1Cols> &src1,
             TileUbDataND<T, 1, Src2Cols, 1, Src2Cols> &src2,
             TileUbDataND<T, 1, Src3Cols, 1, Src3Cols> &src3) {
  pto::MrgSortExecutedNumList executedNumList;
  pto::TMRGSORT<TileUbDataND<T, 1, DstCols, 1, DstCols>,
                TileUbDataND<T, 1, DstCols, 1, DstCols>,
                TileUbDataND<T, 1, Src0Cols, 1, Src0Cols>,
                TileUbDataND<T, 1, Src1Cols, 1, Src1Cols>,
                TileUbDataND<T, 1, Src2Cols, 1, Src2Cols>,
                TileUbDataND<T, 1, Src3Cols, 1, Src3Cols>, false>(
      dst, executedNumList, tmp, src0, src1, src2, src3);
  TL_PIPE_V_BARRIER();
}

// ============================================================================
// Full Sort / TopK: device-side template implementation
// ============================================================================
//
// Layout in tmp (interpreted as float):
//   bufA: tmp[0 .. 2N)         sort32 output / ping-pong A
//   bufB: dst (float full sort) OR tmp[2N .. 4N) (half OR topk)
//   bufC: tmp[2N .. 4N) when bufB == dst; tmp[4N .. 6N) when bufB lives in tmp
//
// Indices live in bufB low half before sort32 consumes them. For half input,
// the casted float source lives in bufB high half before sort32. Both regions
// are then free for use as bufB ping-pong during the merge tree.

namespace sort_detail {

// All constexpr helpers below are tagged AICORE so the CCE compiler lets
// them be called from [aicore]-attributed templates. They're still pure
// compile-time computations -- the attribute is purely a visibility hint.

// Length of the i-th segment in a level given (NumSegs, FullSize, LastSize).
// Returns 0 when i is out of range so callers can use a uniform 4-element
// length tuple.
AICORE constexpr int32_t seg_length(int32_t num_segs, int32_t full_size,
                                    int32_t last_size, int32_t i) {
  if (i >= num_segs)
    return 0;
  if (i == num_segs - 1)
    return last_size;
  return full_size;
}

// Length of the (single) last segment after one level of up-to-4-way merging.
AICORE constexpr int32_t next_last_size(int32_t num_segs, int32_t full_size,
                                        int32_t last_size) {
  int32_t last_group_start = ((num_segs - 1) / 4) * 4;
  int32_t last_group_count = num_segs - last_group_start;
  return (last_group_count - 1) * full_size + last_size;
}

// Length of a "full" segment after one level. If only one segment remains
// after this level (tree converged) "full" equals the single remaining size.
AICORE constexpr int32_t next_full_size(int32_t num_segs, int32_t full_size,
                                        int32_t last_size) {
  int32_t new_num = (num_segs + 3) / 4;
  if (new_num <= 1) {
    return next_last_size(num_segs, full_size, last_size);
  }
  return 4 * full_size;
}

// Number of merge-tree levels needed to reduce N blocks to 1.
AICORE constexpr int32_t compute_levels(int32_t blk_num) {
  int32_t n = blk_num;
  int32_t levels = 0;
  while (n > 1) {
    n = (n + 3) / 4;
    levels++;
  }
  return levels;
}

// Whether the final result lives in bufA after the merge tree finishes.
// read_from_a starts true and toggles every level, so result_in_bufA equals
// (levels % 2 == 0). For kBlockNum == 1 (zero levels) this is also true.
template <int32_t kBlockNum>
constexpr bool result_in_bufA_v = (compute_levels(kBlockNum) % 2 == 0);

// Number of float pair-elements the finalize step has to copy. For full sort
// it's 2*N; for topk it's 2*K rounded up to user_T's block alignment so the
// generated TMOV/TCVT lands on aligned bytes (matches AscendC's DataCopy).
AICORE constexpr int32_t output_pairs(int32_t n, int32_t top_k,
                                      int32_t user_t_bytes) {
  if (top_k < 0)
    return 2 * n;
  int32_t elems_per_block = 32 / user_t_bytes;
  int32_t topk_elems = 2 * top_k;
  return ((topk_elems + elems_per_block - 1) / elems_per_block) *
         elems_per_block;
}

// One sorted segment moved from a read buffer to a write buffer (no merging
// needed because the segment is alone in its 4-group).
template <typename T, int32_t Len>
AICORE PTO_INLINE void merge_group_copy(int32_t src_addr, int32_t dst_addr) {
  constexpr int32_t copy_floats = Len * 2;
  TileUbDataND<T, 1, copy_floats, 1, copy_floats> sort_cs;
  TASSIGN(sort_cs, src_addr);
  TileUbDataND<T, 1, copy_floats, 1, copy_floats> sort_cd;
  TASSIGN(sort_cd, dst_addr);
  TMOV(sort_cd, sort_cs);
}

template <typename T, int32_t Len0, int32_t Len1>
AICORE PTO_INLINE void merge_group_2way(int32_t s0, int32_t s1, int32_t md,
                                        int32_t mt) {
  constexpr int32_t dst_floats = (Len0 + Len1) * 2;
  TileUbDataND<T, 1, Len0 * 2, 1, Len0 * 2> sort_s0;
  TASSIGN(sort_s0, s0);
  TileUbDataND<T, 1, Len1 * 2, 1, Len1 * 2> sort_s1;
  TASSIGN(sort_s1, s1);
  TileUbDataND<T, 1, dst_floats, 1, dst_floats> sort_md;
  TASSIGN(sort_md, md);
  TileUbDataND<T, 1, dst_floats, 1, dst_floats> sort_mt;
  TASSIGN(sort_mt, mt);
  if constexpr (Len0 == Len1) {
    MergeSort<T, Len0 * 2, dst_floats>(sort_md, sort_mt, sort_s0, sort_s1);
  } else {
    MergeSortVar<T, Len0 * 2, Len1 * 2, dst_floats>(sort_md, sort_mt, sort_s0,
                                                    sort_s1);
  }
}

template <typename T, int32_t Len0, int32_t Len1, int32_t Len2>
AICORE PTO_INLINE void merge_group_3way(int32_t s0, int32_t s1, int32_t s2,
                                        int32_t md, int32_t mt) {
  constexpr int32_t dst_floats = (Len0 + Len1 + Len2) * 2;
  TileUbDataND<T, 1, Len0 * 2, 1, Len0 * 2> sort_s0;
  TASSIGN(sort_s0, s0);
  TileUbDataND<T, 1, Len1 * 2, 1, Len1 * 2> sort_s1;
  TASSIGN(sort_s1, s1);
  TileUbDataND<T, 1, Len2 * 2, 1, Len2 * 2> sort_s2;
  TASSIGN(sort_s2, s2);
  TileUbDataND<T, 1, dst_floats, 1, dst_floats> sort_md;
  TASSIGN(sort_md, md);
  TileUbDataND<T, 1, dst_floats, 1, dst_floats> sort_mt;
  TASSIGN(sort_mt, mt);
  if constexpr (Len0 == Len1 && Len1 == Len2) {
    MergeSort<T, Len0 * 2, dst_floats>(sort_md, sort_mt, sort_s0, sort_s1,
                                       sort_s2);
  } else {
    MergeSortVar<T, Len0 * 2, Len1 * 2, Len2 * 2, dst_floats>(
        sort_md, sort_mt, sort_s0, sort_s1, sort_s2);
  }
}

template <typename T, int32_t Len0, int32_t Len1, int32_t Len2, int32_t Len3>
AICORE PTO_INLINE void merge_group_4way(int32_t s0, int32_t s1, int32_t s2,
                                        int32_t s3, int32_t md, int32_t mt) {
  constexpr int32_t dst_floats = (Len0 + Len1 + Len2 + Len3) * 2;
  TileUbDataND<T, 1, Len0 * 2, 1, Len0 * 2> sort_s0;
  TASSIGN(sort_s0, s0);
  TileUbDataND<T, 1, Len1 * 2, 1, Len1 * 2> sort_s1;
  TASSIGN(sort_s1, s1);
  TileUbDataND<T, 1, Len2 * 2, 1, Len2 * 2> sort_s2;
  TASSIGN(sort_s2, s2);
  TileUbDataND<T, 1, Len3 * 2, 1, Len3 * 2> sort_s3;
  TASSIGN(sort_s3, s3);
  TileUbDataND<T, 1, dst_floats, 1, dst_floats> sort_md;
  TASSIGN(sort_md, md);
  TileUbDataND<T, 1, dst_floats, 1, dst_floats> sort_mt;
  TASSIGN(sort_mt, mt);
  if constexpr (Len0 == Len1 && Len1 == Len2 && Len2 == Len3) {
    MergeSort<T, Len0 * 2, dst_floats>(sort_md, sort_mt, sort_s0, sort_s1,
                                       sort_s2, sort_s3);
  } else {
    MergeSortVar<T, Len0 * 2, Len1 * 2, Len2 * 2, Len3 * 2, dst_floats>(
        sort_md, sort_mt, sort_s0, sort_s1, sort_s2, sort_s3);
  }
}

// Walk the groups within one merge-tree level. Recurses on group index G.
template <typename T, int32_t NumSegs, int32_t FullSize, int32_t LastSize,
          bool ReadFromA, int32_t G = 0, int32_t InOff = 0, int32_t OutOff = 0>
AICORE PTO_INLINE void merge_groups_loop(int32_t bufA_addr, int32_t bufB_addr,
                                         int32_t bufC_addr) {
  if constexpr (G < NumSegs) {
    constexpr int32_t len0 = seg_length(NumSegs, FullSize, LastSize, G);
    constexpr int32_t len1 = seg_length(NumSegs, FullSize, LastSize, G + 1);
    constexpr int32_t len2 = seg_length(NumSegs, FullSize, LastSize, G + 2);
    constexpr int32_t len3 = seg_length(NumSegs, FullSize, LastSize, G + 3);
    constexpr int32_t group_count =
        (len0 > 0) + (len1 > 0) + (len2 > 0) + (len3 > 0);
    constexpr int32_t total_elems = len0 + len1 + len2 + len3;
    constexpr int32_t T_BYTES = sizeof(T); // sort runs in float

    const int32_t read_base = ReadFromA ? bufA_addr : bufB_addr;
    const int32_t write_base = ReadFromA ? bufB_addr : bufA_addr;
    const int32_t in_byte_off = InOff * T_BYTES;
    const int32_t out_byte_off = OutOff * T_BYTES;

    if constexpr (group_count == 1) {
      merge_group_copy<T, len0>(read_base + in_byte_off,
                                write_base + out_byte_off);
    } else if constexpr (group_count == 2) {
      merge_group_2way<T, len0, len1>(
          read_base + in_byte_off, read_base + in_byte_off + len0 * 2 * T_BYTES,
          write_base + out_byte_off, bufC_addr);
    } else if constexpr (group_count == 3) {
      merge_group_3way<T, len0, len1, len2>(
          read_base + in_byte_off, read_base + in_byte_off + len0 * 2 * T_BYTES,
          read_base + in_byte_off + (len0 + len1) * 2 * T_BYTES,
          write_base + out_byte_off, bufC_addr);
    } else { // group_count == 4
      merge_group_4way<T, len0, len1, len2, len3>(
          read_base + in_byte_off, read_base + in_byte_off + len0 * 2 * T_BYTES,
          read_base + in_byte_off + (len0 + len1) * 2 * T_BYTES,
          read_base + in_byte_off + (len0 + len1 + len2) * 2 * T_BYTES,
          write_base + out_byte_off, bufC_addr);
    }

    merge_groups_loop<T, NumSegs, FullSize, LastSize, ReadFromA, G + 4,
                      InOff + total_elems * 2, OutOff + total_elems * 2>(
        bufA_addr, bufB_addr, bufC_addr);
  }
}

// Drive one level of the merge tree, then recurse to the next level.
template <typename T, int32_t NumSegs, int32_t FullSize, int32_t LastSize,
          bool ReadFromA>
AICORE PTO_INLINE void merge_levels(int32_t bufA_addr, int32_t bufB_addr,
                                    int32_t bufC_addr) {
  if constexpr (NumSegs > 1) {
    merge_groups_loop<T, NumSegs, FullSize, LastSize, ReadFromA>(
        bufA_addr, bufB_addr, bufC_addr);
    TL_PIPE_V_BARRIER();

    constexpr int32_t new_num_segs = (NumSegs + 3) / 4;
    constexpr int32_t new_full = next_full_size(NumSegs, FullSize, LastSize);
    constexpr int32_t new_last = next_last_size(NumSegs, FullSize, LastSize);

    merge_levels<T, new_num_segs, new_full, new_last, !ReadFromA>(
        bufA_addr, bufB_addr, bufC_addr);
  }
}

} // namespace sort_detail

// Top-level entry point. UserT is the user-facing dtype (float or half),
// internally everything sorts in float (matches AscendC's B16 workaround).
// TopK == -1 means "full sort", TopK >= 0 means "topk; emit only 2*K pairs".
template <typename UserT, int32_t N, int32_t ActualCount, int32_t TopK = -1>
AICORE PTO_INLINE void Sort(int32_t dst_addr, int32_t src_addr,
                            int32_t tmp_addr) {
  static_assert(N % 32 == 0, "Sort: N must be a multiple of 32");
  static_assert(ActualCount > 0 && ActualCount <= N,
                "Sort: 0 < ActualCount <= N");

  constexpr bool is_topk = (TopK >= 0);
  constexpr bool is_half = std::is_same_v<UserT, half>;
  constexpr bool buf_b_in_tmp = is_half || is_topk;

  constexpr int32_t T_BYTES = 4;                  // float internally
  constexpr int32_t USER_T_BYTES = sizeof(UserT); // 4 or 2
  constexpr int32_t BLOCK_NUM = N / 32;
  constexpr int32_t PAD_COUNT = N - ActualCount;

  const int32_t bufA = tmp_addr;
  const int32_t bufB = buf_b_in_tmp ? (tmp_addr + 2 * N * T_BYTES) : dst_addr;
  const int32_t bufC = tmp_addr + (buf_b_in_tmp ? 4 : 2) * N * T_BYTES;
  const int32_t indices_addr = bufB; // bufB low half before sort32
  const int32_t sort_src_addr =
      is_half ? (bufB + N * T_BYTES) : src_addr; // bufB high half for half

  // Phase 0 (half only): cast user src(half) -> float at bufB high half.
  if constexpr (is_half) {
    TileUbDataND<half, 1, N, 1, N> sort_h_src;
    TASSIGN(sort_h_src, src_addr);
    TileUbDataND<float, 1, N, 1, N> sort_f_src;
    TASSIGN(sort_f_src, sort_src_addr);
    pto::TCVT(sort_f_src, sort_h_src, pto::RoundMode::CAST_NONE);
    TL_PIPE_V_BARRIER();
  }

  // Phase 1: pad sort_src tail with -inf for [ActualCount, N).
  if constexpr (PAD_COUNT > 0) {
    TileUbDataND<float, 1, N, 1, ActualCount, pto::PadValue::Min> sort_src_v;
    TASSIGN(sort_src_v, sort_src_addr);
    TileUbDataND<float, 1, N, 1, N, pto::PadValue::Min> sort_src_f;
    TASSIGN(sort_src_f, sort_src_addr);
    pto::TFILLPAD_INPLACE(sort_src_f, sort_src_v);
    TL_PIPE_V_BARRIER();
  }

  // Phase 2: generate ascending indices in bufB low half (float values 0..N-1
  // that sort32 will reinterpret as uint32 in the value-index pair output).
  {
    TileUbDataND<float, 1, N, 1, N> sort_idx;
    TASSIGN(sort_idx, indices_addr);
    TCI<decltype(sort_idx), float, /*descending=*/0>(sort_idx, (float)0);
  }
  TL_PIPE_V_BARRIER();

  // Phase 3: sort32 (float src + uint32 indices -> bufA, 32-block sorted pairs)
  {
    TileUbDataND<float, 1, N, 1, N> sort_src;
    TASSIGN(sort_src, sort_src_addr);
    TileUbDataND<uint32_t, 1, N, 1, N> sort_idx_u;
    TASSIGN(sort_idx_u, indices_addr);
    TileUbDataND<float, 1, 2 * N, 1, 2 * N> sort_buf_a;
    TASSIGN(sort_buf_a, bufA);
    TSORT32(sort_buf_a, sort_src, sort_idx_u);
  }
  TL_PIPE_V_BARRIER();

  // Phase 4: merge tree (compile-time unrolled by sort_detail::merge_levels).
  sort_detail::merge_levels<float, BLOCK_NUM, 32, 32, true>(bufA, bufB, bufC);

  // Phase 5: finalize into dst.
  constexpr bool result_in_bufA = sort_detail::result_in_bufA_v<BLOCK_NUM>;
  constexpr int32_t OUTPUT_PAIRS =
      sort_detail::output_pairs(N, TopK, USER_T_BYTES);

  const int32_t result_addr = result_in_bufA ? bufA : bufB;

  if constexpr (is_half) {
    // Cast 2*K (or 2*N) float pairs -> halves at dst. CAST_RINT keeps the
    // integer indices exact since they were generated as 0..N-1.
    TileUbDataND<float, 1, OUTPUT_PAIRS, 1, OUTPUT_PAIRS> sort_fs;
    TASSIGN(sort_fs, result_addr);
    TileUbDataND<half, 1, OUTPUT_PAIRS, 1, OUTPUT_PAIRS> sort_fd;
    TASSIGN(sort_fd, dst_addr);
    pto::TCVT(sort_fd, sort_fs, pto::RoundMode::CAST_RINT);
    TL_PIPE_V_BARRIER();
  } else {
    // Float full sort: dst is bufB when bufB lives in dst, so the TMOV is
    // only needed when the final write landed in bufA. For topk bufB is
    // always in tmp, so we always have to copy.
    constexpr bool need_copy = is_topk || result_in_bufA;
    if constexpr (need_copy) {
      TileUbDataND<float, 1, OUTPUT_PAIRS, 1, OUTPUT_PAIRS> sort_fs;
      TASSIGN(sort_fs, result_addr);
      TileUbDataND<float, 1, OUTPUT_PAIRS, 1, OUTPUT_PAIRS> sort_fd;
      TASSIGN(sort_fd, dst_addr);
      TMOV(sort_fd, sort_fs);
      TL_PIPE_V_BARRIER();
    }
  }
}

// Dynamic-shape version: actual_count is a runtime parameter.
// N must be a compile-time constant (max_actual_num rounded to multiple of 32).
// Uses pto::DYNAMIC for valid region size, enabling runtime actual_count.
template <typename UserT, int32_t N, int32_t TopK = -1>
AICORE PTO_INLINE void SortDynamic(int32_t dst_addr, int32_t src_addr,
                                   int32_t tmp_addr, int32_t actual_count) {
  static_assert(N % 32 == 0, "SortDynamic: N must be a multiple of 32");
  // Note: actual_count is a runtime parameter; caller must ensure 0 <
  // actual_count <= N

  constexpr bool is_topk = (TopK >= 0);
  constexpr bool is_half = std::is_same_v<UserT, half>;
  constexpr bool buf_b_in_tmp = is_half || is_topk;

  constexpr int32_t T_BYTES = 4;
  constexpr int32_t USER_T_BYTES = sizeof(UserT);
  constexpr int32_t BLOCK_NUM = N / 32;
  const int32_t pad_count = N - actual_count;

  const int32_t bufA = tmp_addr;
  const int32_t bufB = buf_b_in_tmp ? (tmp_addr + 2 * N * T_BYTES) : dst_addr;
  const int32_t bufC = tmp_addr + (buf_b_in_tmp ? 4 : 2) * N * T_BYTES;
  const int32_t indices_addr = bufB;
  const int32_t sort_src_addr = is_half ? (bufB + N * T_BYTES) : src_addr;

  // Phase 0 (half only): cast user src(half) -> float at bufB high half.
  if constexpr (is_half) {
    TileUbDataND<half, 1, N, 1, N> sort_h_src;
    TASSIGN(sort_h_src, src_addr);
    TileUbDataND<float, 1, N, 1, N> sort_f_src;
    TASSIGN(sort_f_src, sort_src_addr);
    pto::TCVT(sort_f_src, sort_h_src, pto::RoundMode::CAST_NONE);
    TL_PIPE_V_BARRIER();
  }

  // Phase 1: pad sort_src tail with -inf for [actual_count, N).
  // Use DYNAMIC tile with runtime valid_col, then fill padding.
  if (pad_count > 0) {
    TileUbDataND<float, 1, N, pto::DYNAMIC, pto::DYNAMIC, pto::PadValue::Min>
        sort_src_v(1, actual_count);
    TASSIGN(sort_src_v, sort_src_addr);
    TileUbDataND<float, 1, N, 1, N, pto::PadValue::Min> sort_src_f;
    TASSIGN(sort_src_f, sort_src_addr);
    pto::TFILLPAD_INPLACE(sort_src_f, sort_src_v);
    TL_PIPE_V_BARRIER();
  }

  // Phase 2: generate ascending indices.
  {
    TileUbDataND<float, 1, N, 1, N> sort_idx;
    TASSIGN(sort_idx, indices_addr);
    TCI<decltype(sort_idx), float, /*descending=*/0>(sort_idx, (float)0);
  }
  TL_PIPE_V_BARRIER();

  // Phase 3: sort32.
  {
    TileUbDataND<float, 1, N, 1, N> sort_src;
    TASSIGN(sort_src, sort_src_addr);
    TileUbDataND<uint32_t, 1, N, 1, N> sort_idx_u;
    TASSIGN(sort_idx_u, indices_addr);
    TileUbDataND<float, 1, 2 * N, 1, 2 * N> sort_buf_a;
    TASSIGN(sort_buf_a, bufA);
    TSORT32(sort_buf_a, sort_src, sort_idx_u);
  }
  TL_PIPE_V_BARRIER();

  // Phase 4: merge tree.
  sort_detail::merge_levels<float, BLOCK_NUM, 32, 32, true>(bufA, bufB, bufC);

  // Phase 5: finalize into dst.
  constexpr bool result_in_bufA = sort_detail::result_in_bufA_v<BLOCK_NUM>;
  constexpr int32_t OUTPUT_PAIRS =
      sort_detail::output_pairs(N, TopK, USER_T_BYTES);

  const int32_t result_addr = result_in_bufA ? bufA : bufB;

  if constexpr (is_half) {
    TileUbDataND<float, 1, OUTPUT_PAIRS, 1, OUTPUT_PAIRS> sort_fs;
    TASSIGN(sort_fs, result_addr);
    TileUbDataND<half, 1, OUTPUT_PAIRS, 1, OUTPUT_PAIRS> sort_fd;
    TASSIGN(sort_fd, dst_addr);
    pto::TCVT(sort_fd, sort_fs, pto::RoundMode::CAST_RINT);
    TL_PIPE_V_BARRIER();
  } else {
    constexpr bool need_copy = is_topk || result_in_bufA;
    if constexpr (need_copy) {
      TileUbDataND<float, 1, OUTPUT_PAIRS, 1, OUTPUT_PAIRS> sort_fs;
      TASSIGN(sort_fs, result_addr);
      TileUbDataND<float, 1, OUTPUT_PAIRS, 1, OUTPUT_PAIRS> sort_fd;
      TASSIGN(sort_fd, dst_addr);
      TMOV(sort_fd, sort_fs);
      TL_PIPE_V_BARRIER();
    }
  }
}

template <typename T, int32_t Rows, int32_t Cols>
AICORE PTO_INLINE void transpose(TileUbDataND<T, Rows, Cols, Rows, Cols> &dst,
                                 TileUbDataND<T, Rows, Cols, Rows, Cols> &src,
                                 TileUbDataND<T, Rows, Cols, Rows, Cols> &tmp) {
  pto::TTRANS(dst, src, tmp);
}

template <typename DstT, typename SrcT, int32_t DstRows, int32_t DstCols,
          int32_t SrcRows, int32_t SrcCols>
AICORE PTO_INLINE void
compare(TileUbDataND<DstT, DstRows, DstCols, DstRows, DstCols> &dst,
        TileUbDataND<SrcT, SrcRows, SrcCols, SrcRows, SrcCols> &src0,
        TileUbDataND<SrcT, SrcRows, SrcCols, SrcRows, SrcCols> &src1,
        pto::CmpMode mode) {
  pto::TCMP(dst, src0, src1, mode);
}

template <typename SrcT, int32_t DstRows, int32_t DstCols, int32_t SrcRows,
          int32_t SrcCols>
AICORE PTO_INLINE void
compare(TileUbDataND<int8_t, DstRows, DstCols, DstRows, DstCols> &dst,
        TileUbDataND<SrcT, SrcRows, SrcCols, SrcRows, SrcCols> &src0,
        TileUbDataND<SrcT, SrcRows, SrcCols, SrcRows, SrcCols> &src1,
        pto::CmpMode mode) {
  auto &dst_uint8 = reinterpret_cast<
      TileUbDataND<uint8_t, DstRows, DstCols, DstRows, DstCols> &>(dst);
  pto::TCMP(dst_uint8, src0, src1, mode);
}

template <typename DstT, typename SrcT, int32_t DstRows, int32_t DstCols,
          int32_t DstRowValid, int32_t DstColValid, int32_t SrcRows,
          int32_t SrcCols, int32_t SrcRowValid, int32_t SrcColValid>
AICORE PTO_INLINE void compare_scalar(
    TileUbDataND<DstT, DstRows, DstCols, DstRowValid, DstColValid> &dst,
    TileUbDataND<SrcT, SrcRows, SrcCols, SrcRowValid, SrcColValid> &src,
    SrcT scalar, pto::CmpMode mode) {
  pto::TCMPS(dst, src, scalar, mode);
}

template <typename SrcT, int32_t DstRows, int32_t DstCols, int32_t DstRowValid,
          int32_t DstColValid, int32_t SrcRows, int32_t SrcCols,
          int32_t SrcRowValid, int32_t SrcColValid>
AICORE PTO_INLINE void compare_scalar(
    TileUbDataND<int8_t, DstRows, DstCols, DstRowValid, DstColValid> &dst,
    TileUbDataND<SrcT, SrcRows, SrcCols, SrcRowValid, SrcColValid> &src,
    SrcT scalar, pto::CmpMode mode) {
  auto &dst_uint8 = reinterpret_cast<
      TileUbDataND<uint8_t, DstRows, DstCols, DstRowValid, DstColValid> &>(dst);
  pto::TCMPS(dst_uint8, src, scalar, mode);
}

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid,
          int32_t ColValid>
AICORE PTO_INLINE void
fill_scalar(TileUbDataND<T, Rows, Cols, RowValid, ColValid> &dst, T scalar) {
  for (int i = 0; i < RowValid; i++) {
    for (int j = 0; j < ColValid; j++) {
      dst.data()[i * Cols + j] = scalar;
    }
  }
}

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid,
          int32_t ColValid>
AICORE PTO_INLINE void
tand(TileUbDataND<T, Rows, Cols, RowValid, ColValid> &dst,
     TileUbDataND<T, Rows, Cols, RowValid, ColValid> &src0,
     TileUbDataND<T, Rows, Cols, RowValid, ColValid> &src1) {
  pto::TAND(dst, src0, src1);
}

template <int32_t Rows, int32_t Cols, int32_t RowValid, int32_t ColValid>
AICORE PTO_INLINE void
tand(TileUbDataND<uint8_t, Rows, Cols, RowValid, ColValid> &dst,
     TileUbDataND<uint8_t, Rows, Cols, RowValid, ColValid> &src0,
     TileUbDataND<uint8_t, Rows, Cols, RowValid, ColValid> &src1) {
  auto &dst_u16 = reinterpret_cast<
      TileUbDataND<uint16_t, Rows, Cols / 2, RowValid, ColValid / 2> &>(dst);
  auto &src0_u16 = reinterpret_cast<
      TileUbDataND<uint16_t, Rows, Cols / 2, RowValid, ColValid / 2> &>(src0);
  auto &src1_u16 = reinterpret_cast<
      TileUbDataND<uint16_t, Rows, Cols / 2, RowValid, ColValid / 2> &>(src1);
  pto::TAND(dst_u16, src0_u16, src1_u16);
}

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid,
          int32_t ColValid>
AICORE PTO_INLINE void
tor(TileUbDataND<T, Rows, Cols, RowValid, ColValid> &dst,
    TileUbDataND<T, Rows, Cols, RowValid, ColValid> &src0,
    TileUbDataND<T, Rows, Cols, RowValid, ColValid> &src1) {
  pto::TOR(dst, src0, src1);
}

template <int32_t Rows, int32_t Cols, int32_t RowValid, int32_t ColValid>
AICORE PTO_INLINE void
tor(TileUbDataND<uint8_t, Rows, Cols, RowValid, ColValid> &dst,
    TileUbDataND<uint8_t, Rows, Cols, RowValid, ColValid> &src0,
    TileUbDataND<uint8_t, Rows, Cols, RowValid, ColValid> &src1) {
  auto &dst_u16 = reinterpret_cast<
      TileUbDataND<uint16_t, Rows, Cols / 2, RowValid, ColValid / 2> &>(dst);
  auto &src0_u16 = reinterpret_cast<
      TileUbDataND<uint16_t, Rows, Cols / 2, RowValid, ColValid / 2> &>(src0);
  auto &src1_u16 = reinterpret_cast<
      TileUbDataND<uint16_t, Rows, Cols / 2, RowValid, ColValid / 2> &>(src1);
  pto::TOR(dst_u16, src0_u16, src1_u16);
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2,
          int32_t shape3, int32_t shape4, int32_t shape5, int32_t stride1,
          int32_t stride2, int32_t stride3, int32_t stride4, int32_t stride5,
          uint32_t ub_shape1, uint32_t ub_shape2>
AICORE PTO_INLINE void atomic_add_ub_to_gm_dynamic(
    __gm__ T1 *handle,
    const pto::Shape<shape1, shape2, shape3, shape4, shape5> &shape,
    const pto::Stride<stride1, stride2, stride3, stride4, stride5> &stride,
    int32_t ub_shape_addr, int32_t ub_offset, int32_t valid_row,
    int32_t valid_col) {
  pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC> dynamic_shape;
  dynamic_shape.shape[3] = valid_row;
  dynamic_shape.shape[4] = valid_col;
  pto::GlobalTensor<
      T1, pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC>,
      pto::Stride<stride1, stride2, stride3, stride4, stride5>>
      global_tensor(handle, dynamic_shape, stride);
  constexpr uint8_t len = sizeof(T2);
  constexpr bool use_nd = (static_cast<uint64_t>(ub_shape2) * len) >= 32;

  if constexpr (use_nd) {
    TileUbDataND<T2, ub_shape1, ub_shape2, pto::DYNAMIC, pto::DYNAMIC> temp_ub(
        valid_row, valid_col);
    pto::TASSIGN(temp_ub, ub_shape_addr + ub_offset * len);
    pto::TSTORE<decltype(temp_ub), decltype(global_tensor),
                pto::AtomicType::AtomicAdd>(global_tensor, temp_ub);
  } else {
    TileUbDataDN<T2, ub_shape1, ub_shape2, pto::DYNAMIC, pto::DYNAMIC> temp_ub(
        valid_row, valid_col);
    pto::TASSIGN(temp_ub, ub_shape_addr + ub_offset * len);
    pto::TSTORE<decltype(temp_ub), decltype(global_tensor),
                pto::AtomicType::AtomicAdd>(global_tensor, temp_ub);
  }
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2,
          int32_t shape3, int32_t shape4, int32_t shape5, int32_t stride1,
          int32_t stride2, int32_t stride3, int32_t stride4, int32_t stride5,
          uint32_t l0c_shape1, uint32_t l0c_shape2>
AICORE PTO_INLINE void atomic_add_l0c_to_gm_dynamic(
    __gm__ T1 *handle,
    const pto::Shape<shape1, shape2, shape3, shape4, shape5> &shape,
    const pto::Stride<stride1, stride2, stride3, stride4, stride5> &stride,
    int32_t l0c_shape_addr, int32_t l0c_offset, int32_t valid_row,
    int32_t valid_col) {
  pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC> dynamic_shape;
  dynamic_shape.shape[3] = valid_row;
  dynamic_shape.shape[4] = valid_col;
  pto::GlobalTensor<
      T1, pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC>,
      pto::Stride<stride1, stride2, stride3, stride4, stride5>>
      global_tensor(handle, dynamic_shape, stride);
  constexpr uint8_t len = sizeof(T2);

  pto::TileAcc<T2, l0c_shape1, l0c_shape2, pto::DYNAMIC, pto::DYNAMIC> temp_l0c(
      valid_row, valid_col);
  pto::TASSIGN(temp_l0c, l0c_shape_addr + l0c_offset * len);
  pto::TSTORE<decltype(temp_l0c), decltype(global_tensor),
              pto::AtomicType::AtomicAdd>(global_tensor, temp_l0c);
}

AICORE PTO_INLINE void sync_all() { pto::SYNCALL<pto::SyncCoreType::Mix>(); }

AICORE PTO_INLINE void sync_all_aic() {
  pto::SYNCALL<pto::SyncCoreType::AICOnly>();
}

AICORE PTO_INLINE void sync_all_aiv() {
  pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
}

} // namespace tl::ascend_pto
#endif
