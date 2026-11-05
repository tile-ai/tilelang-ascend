// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#ifndef TVM_TL_ASCEND_A2A3_OP_GENERIC_H_
#define TVM_TL_ASCEND_A2A3_OP_GENERIC_H_
#include "operator.h"
namespace tvm::tl::a2a3 {
// Region parser from TA src/op/op.{h,cc}; no duplicate shared Op registration.
class RegionOp : public Operator {
public:
  RegionOp(Array<PrimExpr> args, BufferMap vmap);
  const Buffer &GetBuffer() const { return buffer_; }
  const Array<Range> &GetRanges() const { return ranges_; }
  const Array<PrimExpr> &GetExtents() const { return extents_; }
  int GetAccessMask() const { return access_mask_; }
  bool IsFullRegion() const;

private:
  Buffer buffer_;
  Array<Range> ranges_;
  int access_mask_;
  Array<PrimExpr> extents_;
};
Var GetVarFromAccessPtr(const PrimExpr &expr);
bool IsAscendCompatibilityTarget(const Target &target);
} // namespace tvm::tl::a2a3
#endif
