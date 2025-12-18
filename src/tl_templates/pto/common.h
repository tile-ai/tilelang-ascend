#include <pto/pto-inst.hpp>

namespace tl::pto {
using namespace pto;

template <typename T, int Rows, int Cols,
          int RowValid = Rows, int ColValid = Cols>
using TileMatL1 = Tile<TileType::Mat, T, Rows, Cols,
                       BLayout::ColMajor,
                       RowValid, ColValid,
                       SLayout::RowMajor,
                       512>;

template <typename T, int Rows, int Cols,
          int RowValid = Rows, int ColValid = Cols>
using TileUbDataND = Tile<TileType::Vec, T, Rows, Cols,
                       BLayout::RowMajor,
                       -1, -1>;

template <typename T, int Rows, int Cols,
          int RowValid = Rows, int ColValid = Cols>
using TileUbDataDN = Tile<TileType::Vec, T, Rows, Cols,
                       BLayout::ColMajor,
                       -1, -1>;

template <typename T1, typename T2, uint32_t M, uint32_t N, uint32_t K,
          bool transpose_A = false, bool transpose_B = false>
AICORE PTO_INLINE void gemm_v0(
            TileMatL1<T1, M, K> const &A, // l1a
            TileMatL1<T1, K, N> const &B, // l1b
            TileAcc<T2, M, N> const &C,   // l0c
            bool clear) {
    // Allocate l0a/l0b
    TileLeft<T1, M, K> l0a;
    TASSIGN(l0a, 0x0);
    TileRight<T1, K, N> l0b;
    TASSIGN(l0b, 0x0);

    // copy l1 to l0a/l0b
    TEXTRACT(l0a, A, 0, 0);
    TEXTRACT(l0b, B, 0, 0);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
     
    if(clear) {
        TMATMUL(C, l0a, l0b);
    } else { // matmul_acc
        TMATMUL_ACC(C, C, l0a, l0b);
    }
}
}