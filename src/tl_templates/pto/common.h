#include <pto/pto-inst.hpp>
#include <type_traits>

#ifdef __CCE_AICORE__
#define CUDART_INF_F 1.0f / 0.0f

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

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols>
using TileMatL0A = pto::Tile<pto::TileType::Left, T, Rows, Cols,
                             pto::BLayout::RowMajor, RowValid, ColValid,
                             pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

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
  bool initflag = false;

  TileMatL0A<T1, M, kL0Size, M, kL0Size> l0a;
  pto::TASSIGN(l0a, 0x0);
  TileMatL0B<T1, kL0Size, N, kL0Size, N> l0b;
  pto::TASSIGN(l0b, 0x0);

  auto war_event_id = (event_t)(((int)EVENT_ID0 + 1) % 8);

  set_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);
  wait_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);

  for (uint32_t kL0Idx = 0; kL0Idx < kL0split; kL0Idx++) {
    initflag = (clear && (kL0Idx == 0));
    const bool is_tail_block =
        (kL0Idx == kL0split - 1); // Determine whether it is a tail block

    // Dynamically define the L0 cache size based on whether the tile is an end
    // tile.
    if (is_tail_block) {
      TileMatL0A<T1, M, K_tail, M, K_tail> l0a;
      TileMatL0B<T1, K_tail, N, K_tail, N> l0b;
      pto::TASSIGN(l0a, 0x0);
      pto::TASSIGN(l0b, 0x0);

      /**
       * Added synchronization logic: Write-After-Read (WAR) protection
       * Objective: Prevent MTE1 (data transfer) from overwriting L0 before M
       * (Cube) completes processing the previous round of data
       * TODO: Support Ping-Pong buffer.
       */
      set_flag(PIPE_M, PIPE_MTE1, war_event_id);
      wait_flag(PIPE_M, PIPE_MTE1, war_event_id);

      if constexpr (!transpose_A) {
        copy_l1_to_l0a<T1, M, K_tail, M, K, false>(l0a, A, 0, kL0Idx * K_tail);
      } else {
        TileMatL1ZN<T1, M, K, validM, validK> A_t;
        pto::TRESHAPE(A_t, A);
        copy_l1_to_l0a<T1, M, K_tail, M, K, true>(l0a, A_t, 0, kL0Idx * K_tail);
      }
      if constexpr (!transpose_B) {
        copy_l1_to_l0b<T1, K_tail, N, K, N, false>(l0b, B, kL0Idx * K_tail, 0);
      } else {
        TileMatL1ZN<T1, K, N, validK, validN> B_t;
        pto::TRESHAPE(B_t, B);
        copy_l1_to_l0b<T1, K_tail, N, K, N, true>(l0b, B_t, kL0Idx * K_tail, 0);
      }

      set_flag(PIPE_MTE1, PIPE_M, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_M, war_event_id);

      if (initflag) {
        pto::TMATMUL(C, l0a, l0b);
      } else {
        pto::TMATMUL_ACC(C, C, l0a, l0b);
      }

    } else {
      // Non-tail block: The L0 cache is defined at the standard size
      // (current_kSize = kL0Size=128).
      TileMatL0A<T1, M, kL0Size, M, kL0Size> l0a;
      TileMatL0B<T1, kL0Size, N, kL0Size, N> l0b;
      pto::TASSIGN(l0a, 0x0);
      pto::TASSIGN(l0b, 0x0);

      set_flag(PIPE_M, PIPE_MTE1, war_event_id);
      wait_flag(PIPE_M, PIPE_MTE1, war_event_id);

      set_flag(PIPE_FIX, PIPE_M, war_event_id);
      wait_flag(PIPE_FIX, PIPE_M, war_event_id);

      if constexpr (!transpose_A) {
        copy_l1_to_l0a<T1, M, kL0Size, M, K, false>(l0a, A, 0,
                                                    kL0Idx * kL0Size);
      } else {
        TileMatL1ZN<T1, M, K, validM, validK> A_t;
        pto::TRESHAPE(A_t, A);
        copy_l1_to_l0a<T1, M, kL0Size, M, K, true>(l0a, A_t, 0,
                                                   kL0Idx * kL0Size);
      }
      if constexpr (!transpose_B) {
        copy_l1_to_l0b<T1, kL0Size, N, K, N, false>(l0b, B, kL0Idx * kL0Size,
                                                    0);
      } else {
        TileMatL1ZN<T1, K, N, validK, validN> B_t;
        pto::TRESHAPE(B_t, B);
        copy_l1_to_l0b<T1, kL0Size, N, K, N, true>(l0b, B_t, kL0Idx * kL0Size,
                                                   0);
      }

      set_flag(PIPE_MTE1, PIPE_M, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_M, war_event_id);

      if (initflag) {
        pto::TMATMUL(C, l0a, l0b);
      } else {
        pto::TMATMUL_ACC(C, C, l0a, l0b);
      }

      set_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
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
        DstTile dst_tile;
        pto::TASSIGN(dst_tile, ub_shape_addr + ub_offset * len);
        pto::TFILLPAD_INPLACE(dst_tile, src_tile);
      }
    }
  } else {
    TileUbDataND<T1, ub_shape1, ub_shape2, pto::DYNAMIC, pto::DYNAMIC>
        temp_src_ub(valid_row, valid_col);
    pto::TASSIGN(temp_src_ub,
                 ub_shape_addr + ub_offset * sizeof(T1) / sizeof(T1) * len);
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

template <typename T1, typename T2, int32_t shape1, int32_t shape2,
          int32_t shape3, int32_t shape4, int32_t shape5, int32_t stride1,
          int32_t stride2, int32_t stride3, int32_t stride4, int32_t stride5,
          uint32_t valid1, uint32_t valid2>
AICORE PTO_INLINE void copy_gm_to_l1(__gm__ T1 *handle, int32_t buffer_addr,
                                     int32_t offset, int32_t actualTailM = 0,
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
      global_tensor(handle, dynamic_shape);
  pto::TLOAD(L1, global_tensor);
  if (useTail && (tailM != shape4 || tailN != shape5)) {
    pto::TFILLPAD(L1, L1);
  }
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2,
          int32_t shape3, int32_t shape4, int32_t shape5, int32_t stride1,
          int32_t stride2, int32_t stride3, int32_t stride4, int32_t stride5,
          uint32_t valid1, uint32_t valid2>
AICORE PTO_INLINE void copy_l0c_to_gm(__gm__ T1 *handle, int32_t buffer_addr,
                                      int32_t offset, int32_t actualTailM = 0,
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
      global_tensor(handle, dynamic_shape);
  pto::TSTORE(global_tensor, L0c);
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2,
          int32_t shape3, int32_t shape4, int32_t shape5, int32_t stride1,
          int32_t stride2, int32_t stride3, int32_t stride4, int32_t stride5,
          uint32_t ub_shape1, uint32_t ub_shape2,
          pto::PadValue PadVal = pto::PadValue::Null>
AICORE PTO_INLINE void copy_gm_to_ub(__gm__ T1 *handle, int32_t ub_shape_addr,
                                     int32_t ub_offset, int32_t valid_row,
                                     int32_t valid_col) {
  constexpr uint8_t len = sizeof(T2);
  pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC> dynamic_shape;
  dynamic_shape.shape[3] = valid_row;
  dynamic_shape.shape[4] = valid_col;
  pto::GlobalTensor<
      T1, pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC>,
      pto::Stride<stride1, stride2, stride3, stride4, stride5>>
      global_tensor(handle, dynamic_shape);
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
        DstTile dst_tile;
        pto::TASSIGN(dst_tile, ub_shape_addr + ub_offset * len);
        pto::TFILLPAD_INPLACE(dst_tile, src_tile);
      }
    }
  } else {
    TileUbDataND<T1, ub_shape1, ub_shape2, pto::DYNAMIC, pto::DYNAMIC>
        temp_src_ub(valid_row, valid_col);
    pto::TASSIGN(temp_src_ub,
                 ub_shape_addr + ub_offset * sizeof(T1) / sizeof(T1) * len);
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
AICORE PTO_INLINE void copy_ub_to_gm(__gm__ T1 *handle, int32_t ub_shape_addr,
                                     int32_t ub_offset, int32_t valid_row,
                                     int32_t valid_col) {
  pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC> dynamic_shape;
  dynamic_shape.shape[3] = valid_row;
  dynamic_shape.shape[4] = valid_col;
  pto::GlobalTensor<
      T1, pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC>,
      pto::Stride<stride1, stride2, stride3, stride4, stride5>>
      global_tensor(handle, dynamic_shape);
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
AICORE PTO_INLINE void
TSIGMOID(TileUbDataND<T, row, col, row, col> &dst_addr,
         TileUbDataND<T, row, col, row, col> &src0_addr) {
  TMULS(src0_addr, src0_addr, -1);
  pipe_barrier(PIPE_V);
  TEXP(src0_addr, src0_addr);
  pipe_barrier(PIPE_V);
  TADDS(src0_addr, src0_addr, 1);
  pipe_barrier(PIPE_V);
  TRECIP(dst_addr, src0_addr);
}

template <typename T, int32_t row, int32_t col>
AICORE PTO_INLINE void axpy(TileUbDataND<T, row, col, row, col> &dst,
                            TileUbDataND<T, row, col, row, col> &src0,
                            float scalar_value) {
  TMULS(src0, src0, static_cast<T>(scalar_value));
  pipe_barrier(PIPE_V);
  TADD(dst, dst, src0);
  pipe_barrier(PIPE_V);
  TMULS(src0, src0, static_cast<T>(1.0f / scalar_value));
}

template <typename T1, typename T2, typename T3, int32_t rows_src,
          int32_t cols_src, int32_t validRow_src, int32_t validCol_src,
          int32_t cols_dst, int32_t row_tmp, int32_t col_tmp>
AICORE PTO_INLINE void
TROWMAX_with_slice_buffer(uint64_t handle_src, uint64_t handle_dst,
                          TileUbDataDN<T2, cols_dst, 1, validRow_src, 1> ub_DN,
                          TileUbDataND<T3, row_tmp, col_tmp> tmp_ub) {
  tl::ascend_pto::TileUbDataND<T1, rows_src, cols_src, validRow_src,
                               validCol_src>
      tileUbWithValid;
  pto::TASSIGN(tileUbWithValid, handle_src);
  pto::TROWMAX(ub_DN, tileUbWithValid, tmp_ub);
}

template <typename T1, typename T2, typename T3, int32_t rows_src,
          int32_t cols_src, int32_t validRow_src, int32_t validCol_src,
          int32_t cols_dst, int32_t row_tmp, int32_t col_tmp>
AICORE PTO_INLINE void
TROWMIN_with_slice_buffer(uint64_t handle_src, uint64_t handle_dst,
                          TileUbDataDN<T2, cols_dst, 1, validRow_src, 1> ub_DN,
                          TileUbDataND<T3, row_tmp, col_tmp> tmp_ub) {
  tl::ascend_pto::TileUbDataND<T1, rows_src, cols_src, validRow_src,
                               validCol_src>
      tileUbWithValid;
  pto::TASSIGN(tileUbWithValid, handle_src);
  pto::TROWMIN(ub_DN, tileUbWithValid, tmp_ub);
}

template <typename T1, typename T2, typename T3, int32_t rows_src,
          int32_t cols_src, int32_t validRow_src, int32_t validCol_src,
          int32_t cols_dst, int32_t row_tmp, int32_t col_tmp>
AICORE PTO_INLINE void
TROWSUM_with_slice_buffer(uint64_t handle_src, uint64_t handle_dst,
                          TileUbDataDN<T2, cols_dst, 1, validRow_src, 1> ub_DN,
                          TileUbDataND<T3, row_tmp, col_tmp> tmp_ub) {
  tl::ascend_pto::TileUbDataND<T1, rows_src, cols_src, validRow_src,
                               validCol_src>
      tileUbWithValid;
  pto::TASSIGN(tileUbWithValid, handle_src);
  pto::TROWSUM(ub_DN, tileUbWithValid, tmp_ub);
}

template <typename T1, typename T2, typename T3, int32_t rows_src,
          int32_t cols_src, int32_t validRow_src, int32_t validCol_src,
          int32_t cols_dst, int32_t row_tmp, int32_t col_tmp>
AICORE PTO_INLINE void
TCOLMAX_with_slice_buffer(uint64_t handle_src, uint64_t handle_dst,
                          TileUbDataND<T2, 1, cols_src, 1, validCol_src> ub,
                          TileUbDataND<T3, row_tmp, col_tmp> tmp_ub) {
  tl::ascend_pto::TileUbDataND<T1, rows_src, cols_src, validRow_src,
                               validCol_src>
      tileUbWithValid;
  pto::TASSIGN(tileUbWithValid, handle_src);
  pto::TCOLMAX(ub, tileUbWithValid);
}

template <typename T1, typename T2, typename T3, int32_t rows_src,
          int32_t cols_src, int32_t validRow_src, int32_t validCol_src,
          int32_t cols_dst, int32_t row_tmp, int32_t col_tmp>
AICORE PTO_INLINE void
TCOLMIN_with_slice_buffer(uint64_t handle_src, uint64_t handle_dst,
                          TileUbDataND<T2, 1, cols_src, 1, validCol_src> ub,
                          TileUbDataND<T3, row_tmp, col_tmp> tmp_ub) {
  tl::ascend_pto::TileUbDataND<T1, rows_src, cols_src, validRow_src,
                               validCol_src>
      tileUbWithValid;
  pto::TASSIGN(tileUbWithValid, handle_src);
  pto::TCOLMIN(ub, tileUbWithValid);
}

template <typename T1, typename T2, int32_t rows_src, int32_t cols_src,
          int32_t validRow_src, int32_t validCol_src, int32_t cols_dst,
          int32_t row_tmp, int32_t col_tmp>
AICORE PTO_INLINE void
TCOLSUM_with_slice_buffer(uint64_t handle_src, uint64_t handle_dst,
                          TileUbDataND<T2, 1, cols_src, 1, validCol_src> ub,
                          uint64_t tmp_addr) {
  tl::ascend_pto::TileUbDataND<T1, rows_src, cols_src, validRow_src,
                               validCol_src>
      tileUbWithValid;
  pto::TASSIGN(tileUbWithValid, handle_src);
  TileUbDataND<T1, row_tmp, col_tmp> tmp_ub;
  pto::TASSIGN(tmp_ub, tmp_addr);
  pto::TCOLSUM(ub, tileUbWithValid, tmp_ub, true);
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

template <typename T, int32_t row, int32_t col, int32_t tmp_rows>
AICORE PTO_INLINE typename std::enable_if<is_float_or_half<T>::value>::type
pow(TileUbDataND<T, row, col, row, col> &dst,
    TileUbDataND<T, row, col, row, col> &src0,
    TileUbDataND<T, row, col, row, col> &src1,
    TileUbDataND<uint8_t, tmp_rows, col, tmp_rows, col> &tmp) {
  TLOG(src0, src0);
  pipe_barrier(PIPE_V);
  TMUL(dst, src0, src1);
  pipe_barrier(PIPE_V);
  TEXP(dst, dst);
}

template <typename T, int32_t row, int32_t col, int32_t tmp_rows>
AICORE PTO_INLINE typename std::enable_if<std::is_integral<T>::value>::type
pow(TileUbDataND<T, row, col, row, col> &dst,
    TileUbDataND<T, row, col, row, col> &src0,
    TileUbDataND<T, row, col, row, col> &src1,
    TileUbDataND<uint8_t, tmp_rows, col, tmp_rows, col> &tmp) {
  using FloatT = float;
  constexpr int32_t float_buf_size = row * col * sizeof(FloatT);
  auto tmp_float0 = reinterpret_cast<__ubuf__ FloatT *>(tmp.data());
  auto tmp_float1 =
      reinterpret_cast<__ubuf__ FloatT *>(tmp.data() + float_buf_size);

  TileUbDataND<FloatT, row, col, row, col> src0_float;
  TileUbDataND<FloatT, row, col, row, col> log_src0_float;
  TileUbDataND<FloatT, row, col, row, col> src1_float;

  pto::TASSIGN(src0_float, reinterpret_cast<uint64_t>(tmp_float0));
  pto::TASSIGN(log_src0_float, reinterpret_cast<uint64_t>(tmp_float1));
  pto::TASSIGN(src1_float, reinterpret_cast<uint64_t>(tmp_float0));

  pto::TCVT(src0_float, src0, pto::RoundMode::CAST_ROUND);

  pto::TLOG(log_src0_float, src0_float);

  pto::TCVT(src1_float, src1, pto::RoundMode::CAST_ROUND);

  pto::TMUL(log_src0_float, log_src0_float, src1_float);

  pto::TEXP(log_src0_float, log_src0_float);

  pto::TCVT(dst, log_src0_float, pto::RoundMode::CAST_ROUND);
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
  switch (pipeID) {
  case 0:
    set_flag(pipe, tpipe, EVENT_ID0);
    break;
  case 1:
    set_flag(pipe, tpipe, EVENT_ID1);
    break;
  case 2:
    set_flag(pipe, tpipe, EVENT_ID2);
    break;
  case 3:
    set_flag(pipe, tpipe, EVENT_ID3);
    break;
  case 4:
    set_flag(pipe, tpipe, EVENT_ID4);
    break;
  case 5:
    set_flag(pipe, tpipe, EVENT_ID5);
    break;
  case 6:
    set_flag(pipe, tpipe, EVENT_ID6);
    break;
  case 7:
    set_flag(pipe, tpipe, EVENT_ID7);
    break;
  default:
    break;
  }
}

template <pipe_t pipe, pipe_t tpipe>
AICORE PTO_INLINE void wait_flag_pipeline(int32_t pipeID) {
  switch (pipeID) {
  case 0:
    wait_flag(pipe, tpipe, EVENT_ID0);
    break;
  case 1:
    wait_flag(pipe, tpipe, EVENT_ID1);
    break;
  case 2:
    wait_flag(pipe, tpipe, EVENT_ID2);
    break;
  case 3:
    wait_flag(pipe, tpipe, EVENT_ID3);
    break;
  case 4:
    wait_flag(pipe, tpipe, EVENT_ID4);
    break;
  case 5:
    wait_flag(pipe, tpipe, EVENT_ID5);
    break;
  case 6:
    wait_flag(pipe, tpipe, EVENT_ID6);
    break;
  case 7:
    wait_flag(pipe, tpipe, EVENT_ID7);
    break;
  default:
    break;
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

} // namespace tl::ascend_pto
#endif
