// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/lib/Transforms/InsertCVSync.cpp
 * \brief TileLangIR insert CV synchronization flags pass.
 *
 */

#include "tilelangir/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRINSERTCVSYNC
#include "tilelangir/Transforms/Passes.h.inc"

namespace {
#define DEBUG_TYPE "tilelangir-insert-cv-sync"

// 将 Value 转换为 i64 类型
static Value convertToI64(OpBuilder &builder, Location loc, Value val) {
  auto type = val.getType();
  if (type.isInteger(64))
    return val;
  if (type.isIndex())
    return builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), val);
  if (type.isInteger())
    return builder.create<arith::ExtSIOp>(loc, builder.getI64Type(), val);
  llvm_unreachable("Unsupported type for conversion to i64");
}

// 安全地获取常量整数值
static std::optional<int64_t> getConstantIntValue(Value value) {
  if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = constOp.getValue().dyn_cast_or_null<IntegerAttr>())
      return intAttr.getInt();
  }
  return std::nullopt;
}

static hivm::TCoreType anotherCoreType(const hivm::TCoreType &current) {
  return current == hivm::TCoreType::VECTOR ? hivm::TCoreType::CUBE
                                            : hivm::TCoreType::VECTOR;
}

static void buildCVSyncSet(OpBuilder &builder, Location loc,
                           hivm::TCoreType coreSrc, Value flagId) {
  auto coreSrcAttr =
      mlir::hivm::TCoreTypeAttr::get(builder.getContext(), coreSrc);
  auto sync_mode = hivm::SyncBlockInstrModeAttr::get(
      builder.getContext(),
      hivm::SyncBlockInstrMode::INTRA_BLOCK_SYNCHRONIZATION);
  auto tPipTypeAttr =
      hivm::PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_S);
  auto pipTypeAttr = hivm::PipeAttr::get(builder.getContext(),
                                         coreSrc == hivm::TCoreType::VECTOR
                                             ? hivm::PIPE::PIPE_MTE3
                                             : hivm::PIPE::PIPE_FIX);

  builder.create<hivm::SyncBlockSetOp>(loc, coreSrcAttr, pipTypeAttr,
                                       tPipTypeAttr, flagId, mlir::Value(),
                                       sync_mode);
}

static void buildCVSyncWait(OpBuilder &builder, Location loc,
                            hivm::TCoreType coreSrc, Value flagId) {
  auto coreSrcAttr =
      mlir::hivm::TCoreTypeAttr::get(builder.getContext(), coreSrc);
  auto tPipTypeAttr =
      hivm::PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_S);
  auto pipTypeAttr =
      hivm::PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_MTE2);

  builder.create<hivm::SyncBlockWaitOp>(loc, coreSrcAttr, tPipTypeAttr,
                                        pipTypeAttr, flagId);
}

struct TileLangIRInsertCVSync
    : impl::TileLangIRInsertCVSyncBase<TileLangIRInsertCVSync> {
public:
  static constexpr size_t SYNC_FLAGS_LIMIT = 16;
  void runOnOperation() override;

private:
  size_t vectorFlagCnt = 0;
  size_t cubeFlagCnt = 0;
  hivm::TCoreType beginCoreType = hivm::TCoreType::CUBE_OR_VECTOR;

  void InsertCVSyncInSinglePipeline(scf::ForOp forOp);
  void InsertInitAndClearInSinglePipeline(scf::ForOp forOp);
  std::pair<Value, Value> GetFinalFlagIds(OpBuilder &builder, Location loc,
                                          Value id, int64_t loopCnt,
                                          bool isVector, bool isLast);
};

void mlir::tilelangir::TileLangIRInsertCVSync::runOnOperation() {
  // 先收集所有外层循环，避免在遍历中修改
  SmallVector<scf::ForOp> outerLoops;
  getOperation()->walk([&](scf::ForOp forOp) {
    if (forOp->getAttr("tilelangir.num_stages"))
      outerLoops.push_back(forOp);
  });

  for (scf::ForOp outer : outerLoops) {
    // 重置计数器
    vectorFlagCnt = 0;
    cubeFlagCnt = 0;
    beginCoreType = hivm::TCoreType::CUBE_OR_VECTOR;

    InsertCVSyncInSinglePipeline(outer);

    InsertInitAndClearInSinglePipeline(outer);
  }
}

void mlir::tilelangir::TileLangIRInsertCVSync::InsertCVSyncInSinglePipeline(
    scf::ForOp outer) {
  // 收集符合条件的内部循环
  SmallVector<scf::ForOp> innerLoops;
  for (Operation &op : outer.getBody()->getOperations()) {
    if (auto inner = dyn_cast<scf::ForOp>(&op)) {
      auto coreTypeAttr = inner->getAttrOfType<hivm::TCoreTypeAttr>(
          hivm::TCoreTypeAttr::name);
      if (coreTypeAttr) {
        auto coreType = coreTypeAttr.getTcoretype();
        if (coreType == hivm::TCoreType::VECTOR ||
            coreType == hivm::TCoreType::CUBE) {
          innerLoops.push_back(inner);
        }
      }
    }
  }

  // 小于1个unroll for的情况下一定是纯C或纯V算子，无需核间同步
  if (innerLoops.size() < 2) {
    return;
  }

  // 依次处理每个内部循环
  for (int i = 0; i < innerLoops.size(); ++i) {
    scf::ForOp inner = innerLoops[i];
    auto coreTypeAttr = inner->getAttrOfType<hivm::TCoreTypeAttr>(
        hivm::TCoreTypeAttr::name);
    if (!coreTypeAttr)
      continue;
    auto coreType = coreTypeAttr.getTcoretype();

    Block *bodyBlock = inner.getBody();
    OpBuilder builder(inner->getContext());

    // 1. 转换循环变量为 i64
    Value id = inner.getInductionVar();
    builder.setInsertionPointToStart(bodyBlock);
    id = convertToI64(builder, inner.getLoc(), id);

    // 2. 验证 lowerBound == 0 且 step == 1
    auto lowerOpt = getConstantIntValue(inner.getLowerBound());
    if (!lowerOpt || *lowerOpt != 0) {
      inner.emitError("Expected lower bound to be constant 0");
      continue;
    }
    auto stepOpt = getConstantIntValue(inner.getStep());
    if (!stepOpt || *stepOpt != 1) {
      inner.emitError("Expected step to be constant 1");
      continue;
    }
    auto upperOpt = getConstantIntValue(inner.getUpperBound());
    if (!upperOpt || *upperOpt <= 0) {
      inner.emitError("Expected upper bound to be constant positive integer");
      continue;
    }

    // 5. 计算标志 ID
    auto [waitFlagId, setFlagId] = GetFinalFlagIds(
        builder, inner.getLoc(), id, *upperOpt,
        coreType == hivm::TCoreType::VECTOR, i == innerLoops.size() - 1
        );

    // 6. 插入 SyncBlockWaitOp（循环体开头）
    buildCVSyncWait(builder, inner.getLoc(), coreType, waitFlagId);

    // 7. 插入 SyncBlockSetOp（循环体末尾，Yield 之前）
    Operation *terminator = bodyBlock->getTerminator();
    builder.setInsertionPoint(terminator);
    buildCVSyncSet(builder, inner.getLoc(), coreType, setFlagId);

    if (beginCoreType == hivm::TCoreType::CUBE_OR_VECTOR) {
      beginCoreType = coreType;
    }
  }
}

std::pair<Value, Value>
mlir::tilelangir::TileLangIRInsertCVSync::GetFinalFlagIds(OpBuilder &builder,
                                                          Location loc,
                                                          Value id,
                                                          int64_t loopCnt,
                                                          bool isVector,
                                                          bool isLast=false) {
  // 计算偏移量
  size_t waitFlagOffsetInt;
  size_t setFlagOffsetInt;
  if (isVector) {
    waitFlagOffsetInt = vectorFlagCnt;
    if (isLast) {
      setFlagOffsetInt = 0;
    } else {
      setFlagOffsetInt = cubeFlagCnt;
      vectorFlagCnt += loopCnt;
    }
  } else {
    waitFlagOffsetInt = cubeFlagCnt;
    if (isLast) {
      setFlagOffsetInt = 0;
    } else {
      setFlagOffsetInt = vectorFlagCnt;
      cubeFlagCnt += loopCnt;
    }
  }

  // 计算最终标志 ID
  Value finalWaitFlagId;
  Value finalSetFlagId;

  Value syncFlagsLimit;
  if (waitFlagOffsetInt || setFlagOffsetInt) {
    syncFlagsLimit = builder.create<arith::ConstantOp>(
        loc, builder.getI64Type(),
        builder.getI64IntegerAttr(SYNC_FLAGS_LIMIT));
  }
  if(waitFlagOffsetInt) {
    Value flagOffset = builder.create<arith::ConstantOp>(
        loc, builder.getI64Type(),builder.getI64IntegerAttr(waitFlagOffsetInt));
    Value flagIdWithOffset = builder.create<arith::AddIOp>(
        loc, id, flagOffset);
    finalWaitFlagId = builder.create<arith::RemSIOp>(
        loc, flagIdWithOffset, syncFlagsLimit);
  } else {
    finalWaitFlagId = id;
  }
  if (setFlagOffsetInt) {
    Value flagOffset = builder.create<arith::ConstantOp>(
        loc, builder.getI64Type(),builder.getI64IntegerAttr(setFlagOffsetInt));
    Value flagIdWithOffset = builder.create<arith::AddIOp>(
        loc, id, flagOffset);
    finalSetFlagId = builder.create<arith::RemSIOp>(
        loc, flagIdWithOffset, syncFlagsLimit);
  } else {
    finalSetFlagId = id;
  }

  return {finalWaitFlagId, finalSetFlagId};
}

void mlir::tilelangir::TileLangIRInsertCVSync::
    InsertInitAndClearInSinglePipeline(scf::ForOp forOp) {
  if (beginCoreType == hivm::TCoreType::CUBE_OR_VECTOR) {
    return;
  }

  // 计算 endCoreType 和 endOffsetInt
  hivm::TCoreType endCoreType;
  size_t endOffsetInt;
  if (vectorFlagCnt != cubeFlagCnt) {
    endCoreType = beginCoreType;
    endOffsetInt = vectorFlagCnt;
  } else {
    endCoreType = anotherCoreType(beginCoreType);
    endOffsetInt =
        beginCoreType == hivm::TCoreType::VECTOR ? cubeFlagCnt : vectorFlagCnt;
  }

  // 获取父块和 OpBuilder
  Block *parentBlock = forOp->getBlock();
  OpBuilder builder(forOp->getContext());
  builder.setInsertionPoint(forOp);

  // 获取 tilelangir.num_stages 属性，创建边界常量（放在外层循环的父块中）
  auto intAttr = forOp->getAttr("tilelangir.num_stages").dyn_cast<IntegerAttr>();
  if (!intAttr) {
    forOp.emitError("Missing tilelangir.num_stages attribute");
    return;
  }
  Type indexType = intAttr.getType(); // 通常为 i32
  Value valueZero = builder.create<arith::ConstantOp>(
      forOp->getLoc(), indexType, builder.getIntegerAttr(indexType, 0));
  Value valueOne = builder.create<arith::ConstantOp>(
      forOp->getLoc(), indexType, builder.getIntegerAttr(indexType, 1));
  Value numStages = builder.create<arith::ConstantOp>(forOp->getLoc(), intAttr);

  // 1. 在外层循环之前插入 initForOp
  auto initForOp = builder.create<scf::ForOp>(forOp->getLoc(), valueZero,
                                              numStages, valueOne);
  auto initCoreType = anotherCoreType(beginCoreType);
  auto initCoreTypeAttr =
      mlir::hivm::TCoreTypeAttr::get(builder.getContext(), initCoreType);
  initForOp->setAttr(hivm::TCoreTypeAttr::name, initCoreTypeAttr);
  {
    // 在 initForOp 体内插入同步操作
    OpBuilder::InsertionGuard guard(builder);
    Block *initBody = initForOp.getBody();
    builder.setInsertionPointToStart(initBody);
    Value initId = initForOp.getInductionVar();
    initId = convertToI64(builder, initForOp->getLoc(), initId);
    buildCVSyncSet(builder, initForOp->getLoc(), initCoreType, initId);
  }

  // 2. 在外层循环之后插入 clearForOp
  builder.setInsertionPointAfter(forOp);
  auto clearForOp = builder.create<scf::ForOp>(forOp->getLoc(), valueZero,
                                               numStages, valueOne);
  auto clearCoreType = anotherCoreType(endCoreType);
  auto clearCoreTypeAttr =
      mlir::hivm::TCoreTypeAttr::get(builder.getContext(), clearCoreType);
  clearForOp->setAttr(hivm::TCoreTypeAttr::name, clearCoreTypeAttr);
  {
    OpBuilder::InsertionGuard guard(builder);
    Block *clearBody = clearForOp.getBody();
    builder.setInsertionPointToStart(clearBody);
    Value clearId = clearForOp.getInductionVar();
    clearId = convertToI64(builder, clearForOp->getLoc(), clearId);
    Value syncFlagsLimit = builder.create<arith::ConstantOp>(
        clearForOp->getLoc(), builder.getI64Type(),
        builder.getI64IntegerAttr(SYNC_FLAGS_LIMIT));
    Value endOffset = builder.create<arith::ConstantOp>(
        clearForOp->getLoc(), builder.getI64Type(),
        builder.getI64IntegerAttr(endOffsetInt));
    Value endIdWithOffset = builder.create<arith::AddIOp>(
        clearForOp->getLoc(), clearId, endOffset);
    Value endId = builder.create<arith::RemSIOp>(
        clearForOp->getLoc(), endIdWithOffset, syncFlagsLimit);
    buildCVSyncWait(builder, clearForOp->getLoc(), clearCoreType, endId);
  }
}

#undef DEBUG_TYPE
} // namespace

} // namespace tilelangir
} // namespace mlir
