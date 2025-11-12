// printf.h
#ifndef PRINTF_H
#define PRINTF_H

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"


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


template <typename T>
__aicore__ void DumpTensor(const LocalTensor<T> &src, uint32_t desc, uint32_t dumpSize,
                               uint8_t dim, const uint32_t shapeInfo[]) {
  AscendC::ShapeInfo shapeInfoParams(dim, shapeInfo);
  AscendC::DumpTensor(src, desc, dumpSize, shapeInfoParams);
}
template <typename T>
__aicore__ void DumpTensor(const GlobalTensor<T> &src, uint32_t desc, uint32_t dumpSize,
                               uint8_t dim, const uint32_t shapeInfo[]) {
  AscendC::ShapeInfo shapeInfoParams(dim, shapeInfo);
  AscendC::DumpTensor(src, desc, dumpSize, shapeInfoParams);
}
} // namespace tl::ascend

#endif // PRINTF_H