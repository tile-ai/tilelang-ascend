#include <common/tile_tensor_impl.hpp>
#include <common/pto_tile.hpp>
#include <common/constants.hpp>

namespace tl::pto {
using namespace pto;

template <typename T1, typename T2, uint32_t M, uint32_t N, uint32_t K,
          bool transpose_A = false, bool transpose_B = false>
void gemm_v0(Tile<Location::Mat, T1, M, K, M, K, 512> const &A, // l1a
             Tile<Location::Mat, T1, K, N, K, N, 512> const &B, // l1b
             Tile<Location::Acc, T2, M, N, M, N, 512> const &C, // l0c
             bool clear) {
    // Allocate l0a/l0b
    Tile<Location::Left, T1, M, K, M, K> l0a_;
    Tile<Location::Right, T1, K, N, K, N> l0b_;

    // copy l1 to l0a/l0b
    TEXTRACT(l0a_, A, 0, 0);
    TEXTRACT(l0b_, B, 0, 0);
     
    if(clear) {
        TMATMUL(C, l0a, l0b);
    } else { // matmul_acc
        TMATMUL_ACC(C, C, l0a, l0b);
    }
}
}