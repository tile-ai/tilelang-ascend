#include <pto/pto-inst.hpp>

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
    TileUbDataND<float, 1, shape, 1, shape> src_temp_ub;
    pto::TASSIGN(src_temp_ub, src_addr + src_offset * len);
    TileUbDataND<float, 1, shape, 1, shape> dst_temp_ub;
    pto::TASSIGN(dst_temp_ub, dst_addr + dst_offset * len);
    pto::TMOV(dst_temp_ub, src_temp_ub);
}


template <typename T1, typename T2, uint32_t M, uint32_t N, uint32_t K,
          uint32_t validM = M, uint32_t validN = N, uint32_t validK = K,
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
        
    pto::TileLeft<T1, M, K> l0a;
    pto::TASSIGN(l0a, 0x0);
    pto::TileRight<T1, K, N> l0b;
    pto::TASSIGN(l0b, 0x0);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    if constexpr (!transpose_A) {
        pto::TEXTRACT(l0a, A, 0, 0);
    } else {  // transpose A
        TileMatL1ZN<T1, M, K, validM, validK> A_t;
        pto::TRESHAPE(A_t, A);
        pto::TEXTRACT(l0a, A_t, 0, 0);
    }
    if constexpr (!transpose_B) {
        pto::TEXTRACT(l0b, B, 0, 0);
    } else {  // transpose B
        TileMatL1ZN<T1, K, N, validK, validN> B_t;
        pto::TRESHAPE(B_t, B);
        pto::TEXTRACT(l0b, B_t, 0, 0);
    }

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
     
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
            TileMatL1<T2, shape4, shape5, valid1, valid2> &L1) {
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
            pto::TileAcc<T2, shape4, shape5, valid1, valid2> &L0c) {
    pto::GlobalTensor<T1, pto::Shape<shape1, shape2, shape3, shape4, shape5>, 
    pto::Stride<stride1, stride2, stride3, stride4, stride5>> global_tensor(handle, shape, stride);
    pto::TSTORE(global_tensor, L0c);
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2, int32_t shape3,
        int32_t shape4, int32_t shape5, int32_t stride1, int32_t stride2, 
        int32_t stride3, int32_t stride4, int32_t stride5, uint32_t valid1, uint32_t valid2>
AICORE PTO_INLINE void copy_gm_to_ub_dynamic(
            __gm__ T1 *handle,
            const pto::Shape<shape1, shape2, shape3, shape4, shape5>& shape,
            const pto::Stride<stride1, stride2, stride3, stride4, stride5>& stride,
            TileUbDataND<T2, shape4, shape5> &ub) {
    pto::GlobalTensor<T1, pto::Shape<shape1, shape2, shape3, shape4, shape5>, 
    pto::Stride<stride1, stride2, stride3, stride4, stride5>> global_tensor(handle, shape, stride);
    pto::TLOAD(ub, global_tensor);
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
AICORE PTO_INLINE void copy_gm_to_l1(__gm__ T1 *handle, TileMatL1<T2, shape4, shape5, valid1, valid2> &L1) {
    pto::GlobalTensor<T1, pto::Shape<shape1, shape2, shape3, shape4, shape5>, 
    pto::Stride<stride1, stride2, stride3, stride4, stride5>> global_tensor(handle);
    pto::TLOAD(L1, global_tensor);
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2, int32_t shape3,
        int32_t shape4, int32_t shape5, int32_t stride1, int32_t stride2, 
        int32_t stride3, int32_t stride4, int32_t stride5, uint32_t valid1, uint32_t valid2>
AICORE PTO_INLINE void copy_l0c_to_gm(__gm__ T1 *handle, pto::TileAcc<T2, shape4, shape5, valid1, valid2> &L0c) {
    pto::GlobalTensor<T1, pto::Shape<shape1, shape2, shape3, shape4, shape5>, 
    pto::Stride<stride1, stride2, stride3, stride4, stride5>> global_tensor(handle);
    pto::TSTORE(global_tensor, L0c);
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2, int32_t shape3,
        int32_t shape4, int32_t shape5, int32_t stride1, int32_t stride2, 
        int32_t stride3, int32_t stride4, int32_t stride5, uint32_t valid1, uint32_t valid2>
AICORE PTO_INLINE void copy_gm_to_ub(
            __gm__ T1 *handle,
            TileUbDataND<T2, shape4, shape5, valid1, valid2> &ub) {
    pto::GlobalTensor<T1, pto::Shape<shape1, shape2, shape3, shape4, shape5>, 
    pto::Stride<stride1, stride2, stride3, stride4, stride5>> global_tensor(handle);
    pto::TLOAD(ub, global_tensor);
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
}