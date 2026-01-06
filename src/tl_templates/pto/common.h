#include <pto/pto-inst.hpp>

namespace tl::pto {
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


template <typename T, int Rows, int Cols>
using TileUbDataND = Tile<TileType::Vec, T, Rows, Cols,
                       BLayout::RowMajor,
                       -1, -1>;


template <typename T, int Rows, int Cols>
using TileUbDataDN = Tile<TileType::Vec, T, Rows, Cols,
                       BLayout::ColMajor,
                       -1, -1>;


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
}