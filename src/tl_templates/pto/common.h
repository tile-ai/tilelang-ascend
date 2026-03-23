#include <pto/pto-inst.hpp>

#ifdef __CCE_AICORE__
#define CUDART_INF_F 1.0f / 0.0f

namespace tl::ascend_pto {

template <typename T, int Rows, int Cols,
          int RowValid = Rows, int ColValid = Cols>
using TileMatL1 = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                       pto::BLayout::ColMajor,
                       RowValid, ColValid,
                       pto::SLayout::RowMajor,
                       512, pto::PadValue::Zero>;


template <typename T, int Rows, int Cols,
          int RowValid = Rows, int ColValid = Cols>
using TileMatL1ZN = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                       pto::BLayout::RowMajor,
                       RowValid, ColValid,
                       pto::SLayout::ColMajor,
                       512, pto::PadValue::Zero>;


template <typename T, int Rows, int Cols, int RowValid = Rows, int ColValid = Cols>
using TileUbDataND = pto::Tile<pto::TileType::Vec, T, Rows, Cols,
                       pto::BLayout::RowMajor,
                       RowValid, ColValid>;


template <typename T, int Rows, int Cols, int RowValid = Rows, int ColValid = Cols>
using TileUbDataDN = pto::Tile<pto::TileType::Vec, T, Rows, Cols,
                       pto::BLayout::ColMajor,
                       RowValid, ColValid>;

template <typename T, int32_t shape>
AICORE PTO_INLINE void mov_tile(int32_t src_addr,
                int32_t dst_addr, int32_t src_offset, int32_t dst_offset, int32_t len) {
    // TileUbDataND<float, 1, shape> src_temp_ub(1, shape);
    TileUbDataND<T, 1, shape, 1, shape> src_temp_ub;
    pto::TASSIGN(src_temp_ub, src_addr + src_offset * len);
    TileUbDataND<T, 1, shape, 1, shape> dst_temp_ub;
    pto::TASSIGN(dst_temp_ub, dst_addr + dst_offset * len);
    pto::TMOV(dst_temp_ub, src_temp_ub);
}

template <typename T1, typename T2, int32_t shape>
AICORE PTO_INLINE void cvt_tile(int32_t src_addr,
                int32_t dst_addr, int32_t src_offset, int32_t dst_offset, int32_t src_len, int32_t dst_len, pto::RoundMode rmode) {
    TileUbDataND<T1, 1, shape, 1, shape> src_temp_ub;
    pto::TASSIGN(src_temp_ub, src_addr + src_offset * src_len);
    TileUbDataND<T2, 1, shape, 1, shape> dst_temp_ub;
    pto::TASSIGN(dst_temp_ub, dst_addr + dst_offset * dst_len);
    pto::TCVT(dst_temp_ub, src_temp_ub, rmode);
}

template <typename T1, typename T2, uint32_t M, uint32_t N, uint32_t K,
          uint32_t validM = M, uint32_t validN = N, uint32_t validK = K, uint32_t K_tail, 
          bool transpose_A = false, bool transpose_B = false>
AICORE PTO_INLINE void gemm_v0(
    std::conditional_t<transpose_A,
        TileMatL1<T1, K, M, validK, validM>,
        TileMatL1<T1, M, K, validM, validK>>& A,
    std::conditional_t<transpose_B,
        TileMatL1<T1, N, K, validN, validK>,
        TileMatL1<T1, K, N, validK, validN>>& B,
    pto::TileAcc<T2, M, N, validM, validN>& C,
    bool clear) {
    constexpr uint32_t kL0Size = 128;          // L0 slice size, adapted to 64K memory limit
    const uint32_t kL0split = (K + kL0Size - 1) / kL0Size;  // Number of slices
    bool initflag = false;

    pto::TileLeft<T1, M, kL0Size> l0a;
    pto::TASSIGN(l0a, 0x0);
    pto::TileRight<T1, kL0Size, N> l0b;
    pto::TASSIGN(l0b, 0x0);

    auto war_event_id = (event_t)(((int)EVENT_ID0 + 1) % 8);

    set_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);
    wait_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);

    for (uint32_t kL0Idx = 0; kL0Idx < kL0split; kL0Idx++) {
        initflag = (clear && (kL0Idx == 0));
        const bool is_tail_block = (kL0Idx == kL0split - 1); // Determine whether it is a tail block

        // Dynamically define the L0 cache size based on whether the tile is an end tile.
        if (is_tail_block) {
            pto::TileLeft<T1, M, K_tail> l0a;  
            pto::TileRight<T1, K_tail, N> l0b;
            pto::TASSIGN(l0a, 0x0);
            pto::TASSIGN(l0b, 0x0);

            /**
            * Added synchronization logic: Write-After-Read (WAR) protection
            * Objective: Prevent MTE1 (data transfer) from overwriting L0 before M (Cube) completes processing the previous round of data
            * TODO: Support Ping-Pong buffer.
            */
            set_flag(PIPE_M, PIPE_MTE1, war_event_id);
            wait_flag(PIPE_M, PIPE_MTE1, war_event_id);

            if constexpr (!transpose_A) {
                pto::TEXTRACT(l0a, A, 0, kL0Idx * K_tail);
            } else {
                TileMatL1ZN<T1, M, K, validM, validK> A_t;
                pto::TRESHAPE(A_t, A);
                pto::TEXTRACT(l0a, A_t, 0, kL0Idx * K_tail);
            }
            if constexpr (!transpose_B) {
                pto::TEXTRACT(l0b, B, kL0Idx * K_tail, 0);
            } else {
                TileMatL1ZN<T1, K, N, validK, validN> B_t;
                pto::TRESHAPE(B_t, B);
                pto::TEXTRACT(l0b, B_t, kL0Idx * K_tail, 0);
            }

            set_flag(PIPE_MTE1, PIPE_M, war_event_id);
            wait_flag(PIPE_MTE1, PIPE_M, war_event_id);

            if (initflag) {
                pto::TMATMUL(C, l0a, l0b);
            } else {
                pto::TMATMUL_ACC(C, C, l0a, l0b);
            }
           
        } else {
            // Non-tail block: The L0 cache is defined at the standard size (current_kSize = kL0Size=128).
            pto::TileLeft<T1, M, kL0Size> l0a; 
            pto::TileRight<T1, kL0Size, N> l0b;
            pto::TASSIGN(l0a, 0x0);
            pto::TASSIGN(l0b, 0x0);

            set_flag(PIPE_M, PIPE_MTE1, war_event_id);
            wait_flag(PIPE_M, PIPE_MTE1, war_event_id);

            set_flag(PIPE_FIX, PIPE_M, war_event_id);
            wait_flag(PIPE_FIX, PIPE_M, war_event_id);

            if constexpr (!transpose_A) {
                pto::TEXTRACT(l0a, A, 0, kL0Idx * kL0Size);
            } else {
                TileMatL1ZN<T1, M, K, validM, validK> A_t;
                pto::TRESHAPE(A_t, A);
                pto::TEXTRACT(l0a, A_t, 0, kL0Idx * kL0Size);
            }
            if constexpr (!transpose_B) {
                pto::TEXTRACT(l0b, B, kL0Idx * kL0Size, 0);
            } else {
                TileMatL1ZN<T1, K, N, validK, validN> B_t;
                pto::TRESHAPE(B_t, B);
                pto::TEXTRACT(l0b, B_t, kL0Idx * kL0Size, 0);
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

template <typename T1, typename T2, uint32_t L1_BLOCK_M, uint32_t L1_BLOCK_N, uint32_t L1_BLOCK_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t validL1_BM = L1_BLOCK_M, uint32_t validL1_BN = L1_BLOCK_N, uint32_t validL1_BK = L1_BLOCK_K,
          uint32_t validBM = BLOCK_M, uint32_t validBN = BLOCK_N, uint32_t validBK = BLOCK_K,
          bool transpose_A = false, bool transpose_B = false>
AICORE PTO_INLINE void gemm_v1(
    std::conditional_t<transpose_A,
        TileMatL1<T1, L1_BLOCK_K, L1_BLOCK_M, validL1_BK, validL1_BM>,
        TileMatL1<T1, L1_BLOCK_M, L1_BLOCK_K, validL1_BM, validL1_BK>>& A,
    std::conditional_t<transpose_B,
        TileMatL1<T1, BLOCK_N, L1_BLOCK_K, validBN, validL1_BK>,
        TileMatL1<T1, L1_BLOCK_K, BLOCK_N, validL1_BK, validBN>>& B,
    pto::TileAcc<T2, BLOCK_M, BLOCK_N, validBM, validBN>& C,
    bool clear) {

    pto::TileLeft<T1, L1_BLOCK_M, L1_BLOCK_K> l0a;
    pto::TASSIGN(l0a, 0x0);
    pto::TileRight<T1, L1_BLOCK_K, BLOCK_N> l0b;
    pto::TASSIGN(l0b, 0x0);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**
     * Added synchronization logic: Write-After-Read (WAR) protection
     * Objective: Prevent MTE1 (data transfer) from overwriting L0 before M (Cube) completes processing the previous round of data
     * TODO: Support Ping-Pong buffer.
    */
    auto war_event_id = (event_t)((int)EVENT_ID0 + 1);
    set_flag(PIPE_M, PIPE_MTE1, war_event_id);
    wait_flag(PIPE_M, PIPE_MTE1, war_event_id);

    if constexpr (!transpose_A) {
        pto::TEXTRACT(l0a, A, 0, 0);
    } else {  // transpose A
        TileMatL1ZN<T1, L1_BLOCK_M, L1_BLOCK_K, validL1_BM, validL1_BK> A_t;
        pto::TRESHAPE(A_t, A);
        pto::TEXTRACT(l0a, A_t, 0, 0);
    }
    if constexpr (!transpose_B) {
        pto::TEXTRACT(l0b, B, 0, 0);
    } else {  // transpose B
        TileMatL1ZN<T1, L1_BLOCK_K, BLOCK_N, validL1_BK, validBN> B_t;
        pto::TRESHAPE(B_t, B);
        pto::TEXTRACT(l0b, B_t, 0, 0);
    }

            set_flag(PIPE_MTE1, PIPE_M, war_event_id);
            wait_flag(PIPE_MTE1, PIPE_M, war_event_id);

    if (clear) {
        pto::TMATMUL(C, l0a, l0b);
    } else {
        pto::TMATMUL_ACC(C, C, l0a, l0b);
    }
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2, int32_t shape3,
        int32_t shape4, int32_t shape5, int32_t stride1, int32_t stride2,
        int32_t stride3, int32_t stride4, int32_t stride5, uint32_t valid1, uint32_t valid2>
AICORE PTO_INLINE void copy_gm_to_l1_dynamic(
            __gm__ T1 *handle,
            const pto::Shape<shape1, shape2, shape3, shape4, shape5>& shape,
            const pto::Stride<stride1, stride2, stride3, stride4, stride5>& stride,
            int32_t addr) {
    TileMatL1<T2, shape4, shape5, valid1, valid2> L1;
    pto::TASSIGN(L1, addr);
    pto::GlobalTensor<T1, pto::Shape<shape1, shape2, shape3, shape4, shape5>,
    pto::Stride<stride1, stride2, stride3, stride4, stride5>> global_tensor(handle, shape, stride);
    pto::TLOAD(L1, global_tensor);
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2, int32_t shape3,
        int32_t shape4, int32_t shape5, int32_t stride1, int32_t stride2,
        int32_t stride3, int32_t stride4, int32_t stride5, uint32_t valid1, uint32_t valid2>
AICORE PTO_INLINE void copy_l0c_to_gm_dynamic(
            __gm__ T1 *handle,
            const pto::Shape<shape1, shape2, shape3, shape4, shape5>& shape,
            const pto::Stride<stride1, stride2, stride3, stride4, stride5>& stride,
            int32_t addr) {
    pto::TileAcc<T2, shape4, shape5, valid1, valid2> L0c;
    pto::TASSIGN(L0c, addr);
    pto::GlobalTensor<T1, pto::Shape<shape1, shape2, shape3, shape4, shape5>,
    pto::Stride<stride1, stride2, stride3, stride4, stride5>> global_tensor(handle, shape, stride);
    pto::TSTORE(global_tensor, L0c);
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2, int32_t shape3,
        int32_t shape4, int32_t shape5, int32_t stride1, int32_t stride2,
        int32_t stride3, int32_t stride4, int32_t stride5, uint32_t ub_shape1, uint32_t ub_shape2, uint32_t valid1, uint32_t valid2>
AICORE PTO_INLINE void copy_gm_to_ub_dynamic(
            __gm__ T1 *handle,
            const pto::Shape<shape1, shape2, shape3, shape4, shape5>& shape,
            const pto::Stride<stride1, stride2, stride3, stride4, stride5>& stride,
            int32_t ub_shape_addr,
            int32_t ub_offset,
            int32_t len) {
    pto::GlobalTensor<T1, pto::Shape<shape1, shape2, shape3, shape4, shape5>,
    pto::Stride<stride1, stride2, stride3, stride4, stride5>> global_tensor(handle, shape, stride);
    TileUbDataND<T2, ub_shape1, ub_shape2, valid1, valid2> temp_ub;
    pto::TASSIGN(temp_ub, ub_shape_addr + ub_offset * len);
    pto::TLOAD(temp_ub, global_tensor);
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2, int32_t shape3,
        int32_t shape4, int32_t shape5, int32_t stride1, int32_t stride2,
        int32_t stride3, int32_t stride4, int32_t stride5, uint32_t ub_shape1, uint32_t ub_shape2 ,
            uint32_t valid1,
            uint32_t valid2>
AICORE PTO_INLINE void copy_ub_to_gm_dynamic(
            __gm__ T1 *handle,
            const pto::Shape<shape1, shape2, shape3, shape4, shape5>& shape,
            const pto::Stride<stride1, stride2, stride3, stride4, stride5>& stride,
            int32_t ub_shape_addr,
            int32_t ub_offset,
            int32_t len) {
    pto::GlobalTensor<T1, pto::Shape<shape1, shape2, shape3, shape4, shape5>,
    pto::Stride<stride1, stride2, stride3, stride4, stride5>> global_tensor(handle, shape, stride);
    // TileUbDataND<T2, ub_shape1, ub_shape2> temp_ub(valid1, valid2);
    TileUbDataND<T2, ub_shape1, ub_shape2, valid1, valid2> temp_ub;
    pto::TASSIGN(temp_ub, ub_shape_addr + ub_offset * len);
    pto::TSTORE(global_tensor, temp_ub);
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2, int32_t shape3,
        int32_t shape4, int32_t shape5, int32_t stride1, int32_t stride2,
        int32_t stride3, int32_t stride4, int32_t stride5, uint32_t valid1, uint32_t valid2>
AICORE PTO_INLINE void copy_gm_to_l1(__gm__ T1 *handle, int32_t addr, int32_t actualTailM, int32_t actualTailN) {
    TileMatL1<T2, shape4, shape5, pto::DYNAMIC, pto::DYNAMIC> L1(actualTailM, actualTailN);
    pto::TASSIGN(L1, addr);
    pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC> dynamic_shape;
    dynamic_shape.shape[3] = actualTailM;
    dynamic_shape.shape[4] = actualTailN;
    pto::GlobalTensor<T1, pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC>,
    pto::Stride<stride1, stride2, stride3, stride4, stride5>> global_tensor(handle, dynamic_shape);
    pto::TLOAD(L1, global_tensor);
    pto::TFILLPAD(L1, L1);
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2, int32_t shape3,
        int32_t shape4, int32_t shape5, int32_t stride1, int32_t stride2,
        int32_t stride3, int32_t stride4, int32_t stride5, uint32_t valid1, uint32_t valid2>
AICORE PTO_INLINE void copy_l0c_to_gm(__gm__ T1 *handle, int32_t addr, int32_t actualTailM, int32_t actualTailN) {
    pto::TileAcc<T2, shape4, shape5, valid1, valid2> L0c;
    pto::TASSIGN(L0c, addr);
    pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC> dynamic_shape;
    dynamic_shape.shape[3] = actualTailM;
    dynamic_shape.shape[4] = actualTailN;
    pto::GlobalTensor<T1, pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC>,
    pto::Stride<stride1, stride2, stride3, stride4, stride5>> global_tensor(handle, dynamic_shape);
    pto::TSTORE(global_tensor, L0c);
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2, int32_t shape3,
        int32_t shape4, int32_t shape5, int32_t stride1, int32_t stride2,
        int32_t stride3, int32_t stride4, int32_t stride5, uint32_t ub_shape1, uint32_t ub_shape2, uint32_t valid1, uint32_t valid2>
AICORE PTO_INLINE void copy_gm_to_ub(
            __gm__ T1 *handle,
             int32_t ub_shape_addr,
            int32_t ub_offset,
            int32_t len) {
    pto::GlobalTensor<T1, pto::Shape<shape1, shape2, shape3, shape4, shape5>,
    pto::Stride<stride1, stride2, stride3, stride4, stride5>> global_tensor(handle);
    TileUbDataND<T2, ub_shape1, ub_shape2, valid1, valid2> temp_ub;
    pto::TASSIGN(temp_ub, ub_shape_addr + ub_offset * len);
    pto::TLOAD(temp_ub, global_tensor);
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2, int32_t shape3,
        int32_t shape4, int32_t shape5, int32_t stride1, int32_t stride2,
        int32_t stride3, int32_t stride4, int32_t stride5, uint32_t ub_shape1, uint32_t ub_shape2 , uint32_t valid1, uint32_t valid2>
AICORE PTO_INLINE void copy_ub_to_gm(
            __gm__ T1 *handle,
            int32_t ub_shape_addr,
            int32_t ub_offset,
            int32_t len
            ) {
    pto::GlobalTensor<T1, pto::Shape<shape1, shape2, shape3, shape4, shape5>,
    pto::Stride<stride1, stride2, stride3, stride4, stride5>> global_tensor(handle);
    // TileUbDataND<T2, ub_shape1, ub_shape2> temp_ub(valid1, valid2);
    TileUbDataND<T2, ub_shape1, ub_shape2, valid1, valid2> temp_ub;
    pto::TASSIGN(temp_ub, ub_shape_addr + ub_offset * len);
    pto::TSTORE(global_tensor, temp_ub);
}

enum class BinaryOp {
    TADD,
    TSUB,
    TMUL,
    TDIV,
    TMAX,
    TMIN,
    TAND,
    TOR
};

template <BinaryOp Op, typename T, int32_t shape>
AICORE PTO_INLINE void binary_tile(int32_t dst_addr, int32_t src0_addr,
                int32_t src1_addr, int32_t dst_offset, int32_t src0_offset, int32_t src1_offset, int32_t len) {
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

enum class UnaryOp {
    TEXP,
    TLOG,
    TABS,
    TRECIP,
    TSQRT,
    TRSQRT,
    TRELU,
    TNOT
};

template <UnaryOp Op, typename T, int32_t shape>
AICORE PTO_INLINE void unary_tile(int32_t dst_addr, int32_t src_addr, int32_t dst_offset, int32_t src_offset, int32_t len) {
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
AICORE PTO_INLINE void TSIGMOID(
    TileUbDataND<T, row, col, row, col> &dst_addr,
    TileUbDataND<T, row, col, row, col> &src0_addr,
    // TileUbDataND<T, row, col, row, col> &tmp_addr,
    int32_t len
){
    TMULS(src0_addr, src0_addr, -1);
    pipe_barrier(PIPE_V);
    TEXP(src0_addr, src0_addr);
    pipe_barrier(PIPE_V);
    TADDS(src0_addr, src0_addr, 1);
    pipe_barrier(PIPE_V);
    TRECIP(dst_addr, src0_addr);
}

template <typename T, int32_t row, int32_t col>
AICORE PTO_INLINE void axpy(
    TileUbDataND<T, row, col, row, col> &dst,
    TileUbDataND<T, row, col, row, col> &src0,
    T scalar_value
){
    // tl::ascend_pto::TileUbDataND <T, row, col, row, col> axpy_ub_temp;
    TMULS(src0, src0, scalar_value);
    pipe_barrier(PIPE_V);
    TADD(dst, dst, src0);
    pipe_barrier(PIPE_V);
    TDIVS(src0, src0, scalar_value);
}

template <typename T1, typename T2, typename T3,
        int32_t rows_src, int32_t cols_src,
        int32_t validRow_src, int32_t validCol_src,
        int32_t cols_dst,
        int32_t row_tmp, int32_t col_tmp>
AICORE PTO_INLINE void TROWMAX_with_slice_buffer(
        uint64_t handle_src,
        uint64_t handle_dst,
        TileUbDataDN<T2, cols_dst, 1, validRow_src, 1> ub_DN,
        TileUbDataND<T3, row_tmp, col_tmp> tmp_ub) {
    tl::ascend_pto::TileUbDataND <T1, rows_src, cols_src, validRow_src, validCol_src> tileUbWithValid;
    pto::TASSIGN(tileUbWithValid, handle_src);
    pto::TROWMAX(ub_DN, tileUbWithValid, tmp_ub);
}

template <typename T1, typename T2, typename T3,
        int32_t rows_src, int32_t cols_src,
        int32_t validRow_src, int32_t validCol_src,
        int32_t cols_dst,
        int32_t row_tmp, int32_t col_tmp>
AICORE PTO_INLINE void TROWMIN_with_slice_buffer(
        uint64_t handle_src,
        uint64_t handle_dst,
        TileUbDataDN<T2, cols_dst, 1, validRow_src, 1> ub_DN,
        TileUbDataND<T3, row_tmp, col_tmp> tmp_ub) {
    tl::ascend_pto::TileUbDataND <T1, rows_src, cols_src, validRow_src, validCol_src> tileUbWithValid;
    pto::TASSIGN(tileUbWithValid, handle_src);
    pto::TROWMIN(ub_DN, tileUbWithValid, tmp_ub);
}

template <typename T1, typename T2, typename T3,
        int32_t rows_src, int32_t cols_src,
        int32_t validRow_src, int32_t validCol_src,
        int32_t cols_dst,
        int32_t row_tmp, int32_t col_tmp>
AICORE PTO_INLINE void TROWSUM_with_slice_buffer(
        uint64_t handle_src,
        uint64_t handle_dst,
        TileUbDataDN<T2, cols_dst, 1, validRow_src, 1> ub_DN,
        TileUbDataND<T3, row_tmp, col_tmp> tmp_ub) {
    tl::ascend_pto::TileUbDataND <T1, rows_src, cols_src, validRow_src, validCol_src> tileUbWithValid;
    pto::TASSIGN(tileUbWithValid, handle_src);
    pto::TROWSUM(ub_DN, tileUbWithValid, tmp_ub);
}

template <typename T1, typename T2, typename T3,
        int32_t rows_src, int32_t cols_src,
        int32_t validRow_src, int32_t validCol_src,
        int32_t cols_dst,
        int32_t row_tmp, int32_t col_tmp>
AICORE PTO_INLINE void TCOLMAX_with_slice_buffer(
        uint64_t handle_src,
        uint64_t handle_dst,
        TileUbDataND<T2, 1, cols_src, 1, validCol_src> ub,
        TileUbDataND<T3, row_tmp, col_tmp> tmp_ub) {
    tl::ascend_pto::TileUbDataND <T1, rows_src, cols_src, validRow_src, validCol_src> tileUbWithValid;
    pto::TASSIGN(tileUbWithValid, handle_src);
    pto::TCOLMAX(ub, tileUbWithValid);
}

template <typename T1, typename T2, typename T3,
        int32_t rows_src, int32_t cols_src,
        int32_t validRow_src, int32_t validCol_src,
        int32_t cols_dst,
        int32_t row_tmp, int32_t col_tmp>
AICORE PTO_INLINE void TCOLMIN_with_slice_buffer(
        uint64_t handle_src,
        uint64_t handle_dst,
        TileUbDataND<T2, 1, cols_src, 1, validCol_src> ub,
        TileUbDataND<T3, row_tmp, col_tmp> tmp_ub) {
    tl::ascend_pto::TileUbDataND <T1, rows_src, cols_src, validRow_src, validCol_src> tileUbWithValid;
    pto::TASSIGN(tileUbWithValid, handle_src);
    pto::TCOLMIN(ub, tileUbWithValid);
}

template <typename T1, typename T2, typename T3,
        int32_t rows_src, int32_t cols_src,
        int32_t validRow_src, int32_t validCol_src,
        int32_t cols_dst,
        int32_t row_tmp, int32_t col_tmp>
AICORE PTO_INLINE void TCOLSUM_with_slice_buffer(
        uint64_t handle_src,
        uint64_t handle_dst,
        TileUbDataND<T2, 1, cols_src, 1, validCol_src> ub,
        TileUbDataND<T3, row_tmp, col_tmp> tmp_ub) {
    tl::ascend_pto::TileUbDataND <T1, rows_src, cols_src, validRow_src, validCol_src> tileUbWithValid;
    pto::TASSIGN(tileUbWithValid, handle_src);
    pto::TCOLSUM(ub, tileUbWithValid, tmp_ub, true);
}

template<typename TileType, typename DataType>
void TCI(TileType& tile, DataType firstValue);

template <typename T, int32_t row, int32_t col>
AICORE PTO_INLINE void tci(int32_t ub_addr, int32_t ub_offset, int32_t len, T firstValue) {
    using TileData = TileUbDataND<T, row, col, row, col>;
    TileData temp_ub;
    TASSIGN(temp_ub, ub_addr + ub_offset * len);
    TCI<TileData, T, 0>(temp_ub, firstValue);
}

template <typename T, int32_t row, int32_t col>
AICORE PTO_INLINE void pow(
    TileUbDataND<T, row, col, row, col> &dst,
    TileUbDataND<T, row, col, row, col> &src0,
    TileUbDataND<T, row, col, row, col> &src1
    ) {
    TLOG(src0, src0);
    TMUL(dst, src0, src1);
    TEXP(dst, dst);
}

enum class BinaryOps {
    TADDS,
    TSUBS,
    TMULS,
    TDIVS,
    TMAXS,
    TMINS
};

template <BinaryOps Op, typename T, int32_t dst_shape, int32_t src_shape>
AICORE PTO_INLINE void binarys_tile(int32_t dst_addr, int32_t src_addr,
                int32_t dst_offset, int32_t src_offset, int32_t len, T scalar_value) {
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

template<pipe_t pipe, pipe_t tpipe> AICORE PTO_INLINE void set_flag_pipeline(int32_t pipeID) {
    switch (pipeID) {
        case 0: set_flag(pipe, tpipe, EVENT_ID0); break;
        case 1: set_flag(pipe, tpipe, EVENT_ID1); break;
        case 2: set_flag(pipe, tpipe, EVENT_ID2); break;
        case 3: set_flag(pipe, tpipe, EVENT_ID3); break;
        case 4: set_flag(pipe, tpipe, EVENT_ID4); break;
        case 5: set_flag(pipe, tpipe, EVENT_ID5); break;
        case 6: set_flag(pipe, tpipe, EVENT_ID6); break;
        case 7: set_flag(pipe, tpipe, EVENT_ID7); break;
        default:break;
    }
}

template<pipe_t pipe, pipe_t tpipe> AICORE PTO_INLINE void wait_flag_pipeline(int32_t pipeID) {
    switch (pipeID) {
        case 0: wait_flag(pipe, tpipe, EVENT_ID0); break;
        case 1: wait_flag(pipe, tpipe, EVENT_ID1); break;
        case 2: wait_flag(pipe, tpipe, EVENT_ID2); break;
        case 3: wait_flag(pipe, tpipe, EVENT_ID3); break;
        case 4: wait_flag(pipe, tpipe, EVENT_ID4); break;
        case 5: wait_flag(pipe, tpipe, EVENT_ID5); break;
        case 6: wait_flag(pipe, tpipe, EVENT_ID6); break;
        case 7: wait_flag(pipe, tpipe, EVENT_ID7); break;
        default:break;
    }
}

template <typename dstT, int32_t dstRow, int32_t dstCol, 
          int32_t dstRowValid, int32_t dstColValid, 
          typename srcT, int32_t srcRow, int32_t srcCol,
          int32_t srcRowValid, int32_t srcColValid,
          int32_t src_element_count>
AICORE PTO_INLINE void TROWEXPAND_with_slice_buffer(
    TileUbDataND<dstT, dstRow, dstCol, dstRow, dstCol> dst,
    TileUbDataDN<srcT, srcRow, srcCol, srcRow, srcCol> src,
    int32_t src_addr, int32_t src_offset) {
  TileUbDataDN<srcT, src_element_count, srcCol, src_element_count, srcColValid>
      src_temp_ub;
  pto::TASSIGN(src_temp_ub, src_addr + src_offset);

  pto::TROWEXPAND(dst, src_temp_ub);
}
template<pipe_t pipe> 
AICORE PTO_INLINE void set_cross_flag(int32_t flag, int32_t mode) {
    int config = 1 | (mode << 4) | (flag << 8);
    ffts_cross_core_sync(pipe, config);
}

template<pipe_t pipe> 
AICORE PTO_INLINE void set_intra_block_cube(int32_t flag) {
    set_intra_block(pipe, flag);
    set_intra_block(pipe, flag + 16);
}

template<pipe_t pipe> 
AICORE PTO_INLINE void set_intra_block_vec(int32_t flag) {
    set_intra_block(pipe, flag);
}

AICORE PTO_INLINE void wait_cross_flag(int32_t flag) {
    wait_flag_dev(flag);
}

template<pipe_t pipe> 
AICORE PTO_INLINE void wait_intra_block_cube(int32_t flag) {
    wait_intra_block(pipe, flag);
    wait_intra_block(pipe, flag + 16);
}

template<pipe_t pipe> 
AICORE PTO_INLINE void wait_intra_block_vec(int32_t flag) {
    wait_intra_block(pipe, flag);
}
}
#endif
