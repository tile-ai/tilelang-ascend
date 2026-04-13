//===----------------------------------------------------------------------===//
// TileLangIR enable-local-buffer pass
//===----------------------------------------------------------------------===//

#include "tilelangir/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Debug.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "tilelangir/Transforms/Passes.h.inc"

using namespace mlir;
using namespace mlir::tilelangir;
using namespace bishengir;

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRENABLELOCALBUFFER
#include "tilelangir/Transforms/Passes.h.inc"

static MemRefType expandMemRefType(MemRefType oldType, int32_t multiBuffer) {
  ArrayRef<int64_t> oldShape = oldType.getShape();
  SmallVector<int64_t> newShape;
  newShape.push_back(multiBuffer);
  newShape.append(oldShape.begin(), oldShape.end());

  MemRefLayoutAttrInterface newLayout = oldType.getLayout();
  if (auto stridedLayout = dyn_cast<StridedLayoutAttr>(oldType.getLayout())) {
    SmallVector<int64_t> newStrides;
    int64_t leadingStride = 1;
    bool isDynamic = false;
    for (int64_t dim : oldShape) {
      if (dim == ShapedType::kDynamic) {
        isDynamic = true;
        break;
      }
      leadingStride *= dim;
    }
    if (!isDynamic) {
      newStrides.push_back(leadingStride);
    } else {
      newStrides.push_back(ShapedType::kDynamic);
    }
    ArrayRef<int64_t> oldStrides = stridedLayout.getStrides();
    newStrides.append(oldStrides.begin(), oldStrides.end());
    newLayout = StridedLayoutAttr::get(oldType.getContext(),
                                       stridedLayout.getOffset(),
                                       newStrides);
  }
  return MemRefType::get(newShape, oldType.getElementType(), newLayout,
                         oldType.getMemorySpace());
}

static std::optional<int32_t> getMultiBufferFromMark(Value value) {
  for (Operation *user : value.getUsers()) {
    if (auto markOp = dyn_cast<annotation::MarkOp>(user)) {
      if (auto attr = markOp->getAttrOfType<IntegerAttr>("hivm.multi_buffer")) {
        return static_cast<int32_t>(attr.getInt());
      }
    }
  }
  return std::nullopt;
}

static scf::ForOp findStageLoop(Operation *op, int32_t multiBuffer) {
  Operation *curr = op;
  while (curr) {
    if (auto forOp = dyn_cast<scf::ForOp>(curr)) {
      Value upper = forOp.getUpperBound();
      APInt upperInt;
      if (!matchPattern(upper, m_ConstantInt(&upperInt)))
        break;
      if (upperInt.getZExtValue() != static_cast<uint64_t>(multiBuffer))
        break;
      Value step = forOp.getStep();
      APInt stepInt;
      if (!matchPattern(step, m_ConstantInt(&stepInt)) || stepInt != 1)
        break;
      Operation *parent = forOp->getParentOp();
      while (parent && !isa<scf::ForOp>(parent))
        parent = parent->getParentOp();
      if (auto outerFor = dyn_cast_or_null<scf::ForOp>(parent)) {
        if (outerFor->hasAttr("tilelangir.num_stages"))
          return forOp;
      }
      break;
    }
    curr = curr->getParentOp();
  }
  return nullptr;
}

static Value createSubviewAndCollapse(OpBuilder &builder, Location loc,
                                      Value source, MemRefType sourceType,
                                      Value stageIndex) {
  int64_t rank = sourceType.getRank();
  assert(rank >= 1);

  SmallVector<OpFoldResult> offsets, sizes, strides;
  offsets.push_back(stageIndex);
  sizes.push_back(builder.getIndexAttr(1));
  strides.push_back(builder.getIndexAttr(1));

  for (int i = 1; i < rank; ++i) {
    offsets.push_back(builder.getIndexAttr(0));
    if (sourceType.isDynamicDim(i)) {
      sizes.push_back(builder.createOrFold<memref::DimOp>(loc, source, i));
    } else {
      sizes.push_back(builder.getIndexAttr(sourceType.getDimSize(i)));
    }
    strides.push_back(builder.getIndexAttr(1));
  }

  auto subviewOp = builder.create<memref::SubViewOp>(loc, source, offsets, sizes, strides);
  auto subviewType = subviewOp.getResult().getType().cast<MemRefType>();

  SmallVector<ReassociationIndices> reassociation;
  reassociation.push_back({0, 1});
  for (int i = 2; i < subviewType.getRank(); ++i) {
    reassociation.push_back({i});
  }
  return builder.create<memref::CollapseShapeOp>(loc, subviewOp.getResult(), reassociation);
}

static void replaceOperandWithSubview(Operation *user, memref::AllocOp allocOp,
                                      scf::ForOp stageLoop, int32_t multiBuffer) {
  OpBuilder builder(user);
  Location loc = user->getLoc();

  Value iv = stageLoop.getInductionVar();
  Type indexType = builder.getIndexType();
  Value idx = iv;
  if (iv.getType() != indexType) {
    idx = builder.create<arith::IndexCastOp>(loc, indexType, iv);
  }

  MemRefType allocType = allocOp.getType().cast<MemRefType>();
  Value collapsed = createSubviewAndCollapse(builder, loc, allocOp, allocType, idx);

  for (OpOperand &operand : user->getOpOperands()) {
    if (operand.get() == allocOp) {
      operand.set(collapsed);
    }
  }
}

namespace {
struct TileLangIREnableLocalBuffer
    : public impl::TileLangIREnableLocalBufferBase<TileLangIREnableLocalBuffer> {
  using Base = impl::TileLangIREnableLocalBufferBase<TileLangIREnableLocalBuffer>;
  using Base::Base;
  
public:
  void runOnOperation() override;
};
}

void TileLangIREnableLocalBuffer::runOnOperation() {
  ModuleOp module = getOperation();
  if (!module) return;

  SmallVector<std::pair<memref::AllocOp, int32_t>> targets;
  module.walk([&](memref::AllocOp allocOp) {
    if (auto multi = getMultiBufferFromMark(allocOp.getResult())) {
      targets.emplace_back(allocOp, *multi);
    }
  });

  if (targets.empty()) {
    return;
  }

  for (auto &[allocOp, multiBuffer] : targets) {
    for (Operation *user : allocOp.getResult().getUsers()) {
      if (auto markOp = dyn_cast<annotation::MarkOp>(user)) {
        if (markOp->getAttr("hivm.multi_buffer")) {
          markOp->erase();
          break;
        }
      }
    }

    MemRefType oldType = allocOp.getType();
    MemRefType newType = expandMemRefType(oldType, multiBuffer);
    OpBuilder builder(allocOp);
    auto newAlloc = builder.create<memref::AllocOp>(allocOp.getLoc(), newType);
    allocOp.getResult().replaceAllUsesWith(newAlloc.getResult());
    allocOp.erase();

    SmallVector<Operation *> users(newAlloc.getResult().getUsers());
    for (Operation *user : users) {
      if (user == newAlloc) continue;

      scf::ForOp stageLoop = findStageLoop(user, multiBuffer);
      if (stageLoop) {
        replaceOperandWithSubview(user, newAlloc, stageLoop, multiBuffer);
      }
    }
  }
}

} // namespace tilelangir
} // namespace mlir