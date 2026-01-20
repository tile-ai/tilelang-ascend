#include <pto/pto-inst.hpp>

namespace tl::ascend_pto {
using namespace pto;

template <typename T, int Rows, int Cols,
          int RowValid = Rows, int ColValid = Cols>
using TileMatL1 = Tile<TileType::Mat, T, Rows, Cols,
                       BLayout::ColMajor,
                       RowValid, ColValid,
                       SLayout::RowMajor,
                       512, PadValue::Zero>;


template <typename T, int Rows, int Cols,
          int RowValid = Rows, int ColValid = Cols>
using TileMatL1ZN = Tile<TileType::Mat, T, Rows, Cols,
                       BLayout::RowMajor,
                       RowValid, ColValid,
                       SLayout::ColMajor,
                       512, PadValue::Zero>;


template <typename T, int Rows, int Cols, int RowValid = Rows, int ColValid = Cols>
using TileUbDataND = Tile<TileType::Vec, T, Rows, Cols,
                       BLayout::RowMajor,
                       RowValid, ColValid>;


template <typename T, int Rows, int Cols, int RowValid = Rows, int ColValid = Cols>
using TileUbDataDN = Tile<TileType::Vec, T, Rows, Cols,
                       BLayout::ColMajor,
                       RowValid, ColValid>;

template <typename T, int32_t shape>
AICORE PTO_INLINE void mov_tile(int32_t src_addr, 
                int32_t dst_addr, int32_t src_offset, int32_t dst_offset, int32_t len) {
    // TileUbDataND<float, 1, shape> src_temp_ub(1, shape);
    TileUbDataND<float, 1, shape, 1, shape> src_temp_ub;
    TASSIGN(src_temp_ub, src_addr + src_offset * len);
    TileUbDataND<float, 1, shape, 1, shape> dst_temp_ub;
    TASSIGN(dst_temp_ub, dst_addr + dst_offset * len);
    TMOV(dst_temp_ub, src_temp_ub);
}

// Valid
template <typename T1, typename T2, uint32_t M, uint32_t N, uint32_t K,
          uint32_t validM=M, uint32_t validN=N, uint32_t validK=K,
          bool transpose_A = false, bool transpose_B = false>
AICORE PTO_INLINE void gemm_v0(
            TileMatL1<T1, M, K, validM, validK> &A, // l1a
            TileMatL1<T1, K, N, validK, validN> &B, // l1b
            TileAcc<T2, M, N, validM, validN> &C,   // l0c
            bool clear) {
    // Allocate l0a/l0b
    TileLeft<T1, M, K> l0a;  // Zz
    TASSIGN(l0a, 0x0);
    TileRight<T1, K, N> l0b; // Zn
    TASSIGN(l0b, 0x0);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    TEXTRACT(l0a, A, 0, 0);
    TEXTRACT(l0b, B, 0, 0);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
     
    if(clear) {
        TMATMUL(C, l0a, l0b);
    } else {
        TMATMUL_ACC(C, C, l0a, l0b);
    }
}


// Valid Transpose
template <typename T1, typename T2, uint32_t M, uint32_t N, uint32_t K,
          uint32_t validM=M, uint32_t validN=N, uint32_t validK=K,
          bool transpose_A = false, bool transpose_B = false>
AICORE PTO_INLINE void gemm_v0(
            TileMatL1<T1, M, K, validM, validK> &A, // l1a
            TileMatL1<T1, N, K, validN, validK> &B, // l1b
            TileAcc<T2, M, N, validM, validN> &C,   // l0c
            bool clear) {
    // Allocate l0a/l0b
    TileLeft<T1, M, K> l0a;  // Zz
    TASSIGN(l0a, 0x0);
    TileRight<T1, K, N> l0b; // Zn
    TASSIGN(l0b, 0x0);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    TileMatL1ZN<T1, K, N, validK, validN> B_t;
    TRESHAPE(B_t, B);

    TEXTRACT(l0a, A, 0, 0);
    TEXTRACT(l0b, B_t, 0, 0);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
     
    if(clear) {
        TMATMUL(C, l0a, l0b);
    } else {
        TMATMUL_ACC(C, C, l0a, l0b);
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
    GlobalTensor<T1, pto::Shape<shape1, shape2, shape3, shape4, shape5>, 
    pto::Stride<stride1, stride2, stride3, stride4, stride5>> global_tensor(handle, shape, stride);
    TLOAD(L1, global_tensor);
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2, int32_t shape3,
        int32_t shape4, int32_t shape5, int32_t stride1, int32_t stride2, 
        int32_t stride3, int32_t stride4, int32_t stride5, uint32_t valid1, uint32_t valid2>
AICORE PTO_INLINE void copy_l0c_to_gm_dynamic(
            __gm__ T1 *handle,
            const pto::Shape<shape1, shape2, shape3, shape4, shape5>& shape, 
            const pto::Stride<stride1, stride2, stride3, stride4, stride5>& stride,
            TileAcc<T2, shape4, shape5, valid1, valid2> &L0c) {
    GlobalTensor<T1, pto::Shape<shape1, shape2, shape3, shape4, shape5>, 
    pto::Stride<stride1, stride2, stride3, stride4, stride5>> global_tensor(handle, shape, stride);
    TSTORE(global_tensor, L0c);
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2, int32_t shape3,
        int32_t shape4, int32_t shape5, int32_t stride1, int32_t stride2, 
        int32_t stride3, int32_t stride4, int32_t stride5, uint32_t valid1, uint32_t valid2>
AICORE PTO_INLINE void copy_gm_to_ub_dynamic(
            __gm__ T1 *handle,
            const pto::Shape<shape1, shape2, shape3, shape4, shape5>& shape,
            const pto::Stride<stride1, stride2, stride3, stride4, stride5>& stride,
            TileUbDataND<T2, shape4, shape5> &ub) {
    GlobalTensor<T1, pto::Shape<shape1, shape2, shape3, shape4, shape5>, 
    pto::Stride<stride1, stride2, stride3, stride4, stride5>> global_tensor(handle, shape, stride);
    TLOAD(ub, global_tensor);
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2, int32_t shape3,
        int32_t shape4, int32_t shape5, int32_t stride1, int32_t stride2, 
        int32_t stride3, int32_t stride4, int32_t stride5, uint32_t ub_shape1, uint32_t ub_shape2 , 
            uint32_t valid1, 
            uint32_t valid2>
AICORE PTO_INLINE void copy_ub_to_gm_dynamic(
            __gm__ T1 *handle, 
            const Shape<shape1, shape2, shape3, shape4, shape5>& shape, 
            const Stride<stride1, stride2, stride3, stride4, stride5>& stride,
            int32_t ub_shape_addr, 
            int32_t ub_offset,
            int32_t len) {
    GlobalTensor<T1, pto::Shape<shape1, shape2, shape3, shape4, shape5>, 
    pto::Stride<stride1, stride2, stride3, stride4, stride5>> global_tensor(handle, shape, stride);
    // TileUbDataND<T2, ub_shape1, ub_shape2> temp_ub(valid1, valid2);
    TileUbDataND<T2, ub_shape1, ub_shape2, valid1, valid2> temp_ub;
    TASSIGN(temp_ub, ub_shape_addr + ub_offset * len);
    TSTORE(global_tensor, temp_ub);
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2, int32_t shape3,
        int32_t shape4, int32_t shape5, int32_t stride1, int32_t stride2, 
        int32_t stride3, int32_t stride4, int32_t stride5, uint32_t valid1, uint32_t valid2>
AICORE PTO_INLINE void copy_gm_to_l1(__gm__ T1 *handle, TileMatL1<T2, shape4, shape5, valid1, valid2> &L1) {
    GlobalTensor<T1, pto::Shape<shape1, shape2, shape3, shape4, shape5>, 
    pto::Stride<stride1, stride2, stride3, stride4, stride5>> global_tensor(handle);
    TLOAD(L1, global_tensor);
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2, int32_t shape3,
        int32_t shape4, int32_t shape5, int32_t stride1, int32_t stride2, 
        int32_t stride3, int32_t stride4, int32_t stride5, uint32_t valid1, uint32_t valid2>
AICORE PTO_INLINE void copy_l0c_to_gm(__gm__ T1 *handle, TileAcc<T2, shape4, shape5, valid1, valid2> &L0c) {
    GlobalTensor<T1, pto::Shape<shape1, shape2, shape3, shape4, shape5>, 
    pto::Stride<stride1, stride2, stride3, stride4, stride5>> global_tensor(handle);
    TSTORE(global_tensor, L0c);
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2, int32_t shape3,
        int32_t shape4, int32_t shape5, int32_t stride1, int32_t stride2, 
        int32_t stride3, int32_t stride4, int32_t stride5, uint32_t valid1, uint32_t valid2>
AICORE PTO_INLINE void copy_gm_to_ub(
            __gm__ T1 *handle,
            TileUbDataND<T2, shape4, shape5, valid1, valid2> &ub) {
    GlobalTensor<T1, pto::Shape<shape1, shape2, shape3, shape4, shape5>, 
    pto::Stride<stride1, stride2, stride3, stride4, stride5>> global_tensor(handle);
    TLOAD(ub, global_tensor);
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
    GlobalTensor<T1, pto::Shape<shape1, shape2, shape3, shape4, shape5>, 
    pto::Stride<stride1, stride2, stride3, stride4, stride5>> global_tensor(handle);
    // TileUbDataND<T2, ub_shape1, ub_shape2> temp_ub(valid1, valid2);
    TileUbDataND<T2, ub_shape1, ub_shape2, valid1, valid2> temp_ub;
    TASSIGN(temp_ub, ub_shape_addr + ub_offset * len);
    TSTORE(global_tensor, temp_ub);
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

    TASSIGN(src0_temp_ub, src0_addr + src0_offset * len);
    // TileUbDataND<T, 1, shape> src1_temp_ub(1, shape);
    TileUbDataND<T, 1, shape, 1, shape> src1_temp_ub;

    TASSIGN(src1_temp_ub, src1_addr + src1_offset * len);
    // TileUbDataND<T, 1, shape> dst_temp_ub(1, shape);
    TileUbDataND<T, 1, shape, 1, shape> dst_temp_ub;

    TASSIGN(dst_temp_ub, dst_addr + dst_offset * len);
    if constexpr (Op == BinaryOp::TADD) {
        TADD(dst_temp_ub, src0_temp_ub, src1_temp_ub);
    } else if constexpr (Op == BinaryOp::TSUB) {
        TSUB(dst_temp_ub, src0_temp_ub, src1_temp_ub);
    } else if constexpr (Op == BinaryOp::TMUL) {
        TMUL(dst_temp_ub, src0_temp_ub, src1_temp_ub);
    } else if constexpr (Op == BinaryOp::TDIV) {
        TDIV(dst_temp_ub, src0_temp_ub, src1_temp_ub);
    } else if constexpr (Op == BinaryOp::TMAX) {
        TMAX(dst_temp_ub, src0_temp_ub, src1_temp_ub);
    } else if constexpr (Op == BinaryOp::TMIN) {
        TMIN(dst_temp_ub, src0_temp_ub, src1_temp_ub);
    } else if constexpr (Op == BinaryOp::TAND) {
        TAND(dst_temp_ub, src0_temp_ub, src1_temp_ub);
    } else if constexpr (Op == BinaryOp::TOR) {
        TOR(dst_temp_ub, src0_temp_ub, src1_temp_ub);
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
    TASSIGN(tileUbWithValid, handle_src);
    TROWMAX(ub_DN, tileUbWithValid, tmp_ub);
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
    TASSIGN(tileUbWithValid, handle_src);
    TROWSUM(ub_DN, tileUbWithValid, tmp_ub);
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
    TASSIGN(tileUbWithValid, handle_src);
    TCOLMAX(ub, tileUbWithValid);
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
    TASSIGN(tileUbWithValid, handle_src);
    TCOLSUM(ub, tileUbWithValid, tmp_ub, true);
}
}