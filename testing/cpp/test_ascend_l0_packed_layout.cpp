#include <cstddef>
#include <cstdint>

using std::size_t;

#include "tla/layout.hpp"

using Catlass::layout::nZ;
using Catlass::layout::zN;
using Catlass::layout::zZ;

// K split: a full 128-element child tile remains packed independently of L1.
constexpr auto k_l1_parent = tla::MakeLayout<uint16_t, zZ>(64u, 256u);
constexpr auto k_l1_split = tla::GetTileLayout(
    k_l1_parent, tla::MakeShape(64u, 128u), tla::MakeCoord(0u, 0u));
constexpr auto k_l0_split = tla::MakeLayout<uint16_t, zZ>(64u, 128u);
static_assert(tla::get<0, 1>(k_l1_split.stride()) == 4096);
static_assert(tla::get<0, 1>(k_l0_split.stride()) == 2048);

// K tail: an 80-element L1 view keeps its 64x256 parent stride, while the
// independently allocated L0 destination is packed.
constexpr auto k_l1_tail = tla::GetTileLayout(
    k_l1_parent, tla::MakeShape(64u, 80u), tla::MakeCoord(0u, 0u));
constexpr auto k_l0_tail = tla::MakeLayout<uint16_t, zZ>(64u, 80u);
static_assert(tla::get<0, 1>(k_l1_tail.stride()) == 4096);
static_assert(tla::get<0, 1>(k_l0_tail.stride()) == 1280);

// N split: a 128x128 L0B tile must not inherit the 256x512 L1 stride.
constexpr auto n_l1_parent = tla::MakeLayout<uint16_t, nZ>(256u, 512u);
constexpr auto n_l1_tile = tla::GetTileLayout(
    n_l1_parent, tla::MakeShape(128u, 128u), tla::MakeCoord(0u, 0u));
constexpr auto n_l0_tile = tla::MakeLayout<uint16_t, nZ>(128u, 128u);
static_assert(tla::get<0, 1>(n_l1_tile.stride()) == 8192);
static_assert(tla::get<0, 1>(n_l0_tile.stride()) == 2048);

// Rectangular transpose-A: dstM,dstN are already post-transpose logical axes.
constexpr auto at_l1_parent = tla::MakeLayout<uint16_t, zN>(192u, 64u);
constexpr auto at_l1_tile = tla::GetTileLayout(
    at_l1_parent, tla::MakeShape(64u, 80u), tla::MakeCoord(0u, 0u));
constexpr auto at_l0_tile = tla::MakeLayout<uint16_t, zN>(64u, 80u);
static_assert(tla::get<1, 1>(at_l1_tile.stride()) == 3072);
static_assert(tla::get<1, 1>(at_l0_tile.stride()) == 1024);

// Rectangular transpose-B uses the same logical-axis contract.
constexpr auto bt_l1_parent = tla::MakeLayout<uint16_t, nZ>(192u, 64u);
constexpr auto bt_l1_tile = tla::GetTileLayout(
    bt_l1_parent, tla::MakeShape(80u, 112u), tla::MakeCoord(0u, 0u));
constexpr auto bt_l0_tile = tla::MakeLayout<uint16_t, nZ>(80u, 112u);
static_assert(tla::get<0, 1>(bt_l1_tile.stride()) == 1024);
static_assert(tla::get<0, 1>(bt_l0_tile.stride()) == 1792);
