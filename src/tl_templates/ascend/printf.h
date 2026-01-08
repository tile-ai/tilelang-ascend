// printf.h
#ifndef PRINTF_H
#define PRINTF_H

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/layout/layout.hpp"


namespace tl::ascend {

using namespace AscendC;


template <typename T>
__aicore__ void DumpTensor(const LocalTensor<T> &src, uint32_t desc, uint32_t dumpSize,
                               uint8_t dim, const uint32_t shapeInfo[]) {
  if (dim > 0 && shapeInfo != nullptr) {
    AscendC::ShapeInfo shapeInfoParams(dim, shapeInfo);
    AscendC::DumpTensor(src, desc, dumpSize, shapeInfoParams);
  } else {
    AscendC::DumpTensor(src, desc, dumpSize);
  }
}

template <typename T>
__aicore__ void DumpTensor(const GlobalTensor<T> &src, uint32_t desc, uint32_t dumpSize,
                               uint8_t dim, const uint32_t shapeInfo[]) {
  if (dim > 0 && shapeInfo != nullptr) {
    AscendC::ShapeInfo shapeInfoParams(dim, shapeInfo);
    AscendC::DumpTensor(src, desc, dumpSize, shapeInfoParams);
  } else {
    AscendC::DumpTensor(src, desc, dumpSize);
  }
}
} // namespace tl::ascend

#endif // PRINTF_H