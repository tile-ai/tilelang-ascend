/*!
 * \file tilelangir/lib/Transforms/EnableMultiBuffer.cpp
 * \brief TileLangIR enable-multi-buffer pass.
 */

#include "tilelangir/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "tilelangir/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tilelangir-enable-multi-buffer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")

using namespace mlir;
using namespace mlir::tilelangir;
using namespace bishengir;

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRENABLEMULTIBUFFER
#include "tilelangir/Transforms/Passes.h.inc"

static int g_subviewProcessedCount = 0;

static void debugPrintType(StringRef msg, Type type) {
  LLVM_DEBUG({
    std::string str;
    llvm::raw_string_ostream os(str);
    type.print(os);
    DBGS() << msg << ": " << str << "\n";
  });
}

static void debugPrintOFRArray(StringRef name, ArrayRef<OpFoldResult> arr) {
  LLVM_DEBUG({
    DBGS() << name << " (count=" << arr.size() << "): [";
    for (size_t i = 0; i < arr.size(); ++i) {
      if (i > 0) DBGS() << ", ";
      if (auto attr = dyn_cast<Attribute>(arr[i])) {
        if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
          DBGS() << intAttr.getInt();
        } else {
          attr.print(DBGS());
        }
      } else {
        DBGS() << "<dynamic>";
      }
    }
    DBGS() << "]\n";
  });
}

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
      ArrayRef<int64_t> oldStrides = stridedLayout.getStrides();
      newStrides.append(oldStrides.begin(), oldStrides.end());
      newLayout = StridedLayoutAttr::get(oldType.getContext(), stridedLayout.getOffset(), newStrides);
    } else {
      newStrides.push_back(ShapedType::kDynamic);
      ArrayRef<int64_t> oldStrides = stridedLayout.getStrides();
      newStrides.append(oldStrides.begin(), oldStrides.end());
      newLayout = StridedLayoutAttr::get(oldType.getContext(), stridedLayout.getOffset(), newStrides);
    }
  }
  
  return MemRefType::get(newShape, oldType.getElementType(), newLayout, oldType.getMemorySpace());
}

static Value createStageSubview(OpBuilder &builder, Location loc,
                                Value source, MemRefType sourceType, 
                                Value stageIndex,
                                ArrayRef<OpFoldResult> originalOffsets,
                                ArrayRef<OpFoldResult> originalSizes,
                                ArrayRef<OpFoldResult> originalStrides) {
  
  int expandedSourceRank = sourceType.getRank();
  int originalSourceRank = expandedSourceRank - 1;
  
  LLVM_DEBUG({
    DBGS() << "  [CreateStageSubview] Expanded Source Rank: " << expandedSourceRank << "\n";
    DBGS() << "  [CreateStageSubview] Original Source Rank: " << originalSourceRank << "\n";
    DBGS() << "  [CreateStageSubview] Input Offsets Count: " << originalOffsets.size() << "\n";
    debugPrintOFRArray("  [CreateStageSubview] Input Original Offsets", originalOffsets);
  });

  SmallVector<OpFoldResult> normalizedOffsets, normalizedSizes, normalizedStrides;
  
  normalizedOffsets.append(originalOffsets.begin(), originalOffsets.end());
  normalizedSizes.append(originalSizes.begin(), originalSizes.end());
  normalizedStrides.append(originalStrides.begin(), originalStrides.end());
  
  int missingDims = originalSourceRank - (int)originalOffsets.size();
  
  if (missingDims < 0) {
    llvm::errs() << "ERROR: More offsets (" << originalOffsets.size() << ") than rank (" << originalSourceRank << ")\n";
    assert(false && "Invalid subview params");
  }
  
  if (missingDims > 0) {
    LLVM_DEBUG(DBGS() << "  [CreateStageSubview] Detected " << missingDims << " implicit dims. Padding...\n");
    for (int i = 0; i < missingDims; ++i) {
      int originalDimIdx = (int)normalizedOffsets.size();
      int expandedDimIdx = originalDimIdx + 1;
      
      normalizedOffsets.push_back(builder.getIndexAttr(0));
      
      if (sourceType.isDynamicDim(expandedDimIdx)) {
        normalizedSizes.push_back(builder.createOrFold<memref::DimOp>(loc, source, expandedDimIdx));
      } else {
        normalizedSizes.push_back(builder.getIndexAttr(sourceType.getDimSize(expandedDimIdx)));
      }
      normalizedStrides.push_back(builder.getIndexAttr(1));
    }
  }

  if ((int)normalizedOffsets.size() != originalSourceRank) {
    llvm::errs() << "FATAL: Normalization failed. Size: " << normalizedOffsets.size() 
                 << ", Expected: " << originalSourceRank << "\n";
    assert(false);
  }

  SmallVector<OpFoldResult> newOffsets, newSizes, newStrides;
  newOffsets.reserve(expandedSourceRank);
  newSizes.reserve(expandedSourceRank);
  newStrides.reserve(expandedSourceRank);
  
  newOffsets.push_back(stageIndex);
  newSizes.push_back(builder.getIndexAttr(1));
  newStrides.push_back(builder.getIndexAttr(1));
  
  newOffsets.append(normalizedOffsets.begin(), normalizedOffsets.end());
  newSizes.append(normalizedSizes.begin(), normalizedSizes.end());
  newStrides.append(normalizedStrides.begin(), normalizedStrides.end());
  
  LLVM_DEBUG({
    DBGS() << "  [CreateStageSubview] Final Counts: Offsets=" << newOffsets.size() << ", Expected=" << expandedSourceRank << "\n";
    if ((int)newOffsets.size() != expandedSourceRank) {
      DBGS() << "  [CreateStageSubview] CRITICAL MISMATCH DETECTED BEFORE CREATE!\n";
    }
  });

  if ((int)newOffsets.size() != expandedSourceRank) {
    llvm::errs() << "FATAL: Mismatch before create. Got " << newOffsets.size() << " expected " << expandedSourceRank << "\n";
    assert(false && "Rank mismatch");
  }

  auto subviewOp = builder.create<memref::SubViewOp>(loc, source, newOffsets, newSizes, newStrides);
  
  SmallVector<ReassociationIndices> reassociation;
  ReassociationIndices currentGroup;
  bool mergedLeadingOnes = false;
  auto subviewType = subviewOp.getResult().getType().cast<MemRefType>();

  for (int i = 0; i < subviewType.getRank(); ++i) {
    int64_t dimSize = subviewType.getDimSize(i);
    if (!mergedLeadingOnes && dimSize == 1) {
      currentGroup.push_back(i);
    } else {
      if (!currentGroup.empty()) {
        currentGroup.push_back(i);
        reassociation.push_back(currentGroup);
        currentGroup.clear();
        mergedLeadingOnes = true;
      } else {
        reassociation.push_back({i});
      }
    }
  }
  if (!currentGroup.empty()) reassociation.push_back(currentGroup);

  LLVM_DEBUG(DBGS() << "  [CreateStageSubview] Returning collapsed value.\n");
  return builder.create<memref::CollapseShapeOp>(loc, subviewOp.getResult(), reassociation);
}

class WorkspaceExpander {
public:
  static bool expand(Operation *op) {
    bool changed = false;
    SmallVector<memref_ext::AllocWorkspaceOp> targets;
    
    op->walk([&](memref_ext::AllocWorkspaceOp allocOp) {
      Value res = allocOp.getResult();
      for (Operation *user : res.getUsers()) {
        if (auto markOp = dyn_cast<annotation::MarkOp>(user)) {
          if (markOp->getAttrOfType<IntegerAttr>("hivm.multi_buffer")) {
            targets.push_back(allocOp);
            break;
          }
        }
      }
    });
    
    for (auto allocOp : targets) {
      Value workspaceValue = allocOp.getResult();
      int32_t multiBuffer = 0;
      annotation::MarkOp markOpToRemove = nullptr;
      
      for (Operation *user : workspaceValue.getUsers()) {
        if (auto markOp = dyn_cast<annotation::MarkOp>(user)) {
          if (auto attr = markOp->getAttrOfType<IntegerAttr>("hivm.multi_buffer")) {
            multiBuffer = static_cast<int32_t>(attr.getInt());
            markOpToRemove = markOp;
            break;
          }
        }
      }
      
      if (!markOpToRemove) continue;
      
      LLVM_DEBUG(DBGS() << "Expanding workspace with factor=" << multiBuffer << "\n");
      markOpToRemove->erase();
      
      MemRefType oldType = workspaceValue.getType().cast<MemRefType>();
      MemRefType newType = expandMemRefType(oldType, multiBuffer);
      
      OpBuilder builder(allocOp);
      auto newAlloc = builder.create<memref_ext::AllocWorkspaceOp>(
          allocOp.getLoc(), newType,
          allocOp.getWorkspaceArg(),
          allocOp.getDynamicSize(),
          allocOp.getOffset());
      
      workspaceValue.replaceAllUsesWith(newAlloc.getResult());
      allocOp.erase();
      changed = true;
      
      debugPrintType("New workspace type", newType);
    }
    return changed;
  }
};

class ScopeToForConverter {
public:
  ScopeToForConverter(Operation *scopeOp, ArrayRef<Value> workspaceValues, int32_t numStage, Value outerIV)
      : scopeOp_(scopeOp), workspaceValues_(workspaceValues), numStage_(numStage), outerIV_(outerIV) {}
  
  bool convert() {
    if (!scopeOp_) return false;
    
    OpBuilder builder(scopeOp_);
    Location loc = scopeOp_->getLoc();
    
    Region *scopeRegion = nullptr;
    if (auto registeredScope = dyn_cast<scope::ScopeOp>(scopeOp_)) {
      scopeRegion = &registeredScope.getRegion();
    } else if (scopeOp_->getNumRegions() > 0) {
      scopeRegion = &scopeOp_->getRegion(0);
    }
    
    if (!scopeRegion || scopeRegion->empty()) {
      scopeOp_->erase();
      return false;
    }
    
    Value c0 = builder.create<arith::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(0));
    Value cNumStage = builder.create<arith::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(numStage_));
    Value c1 = builder.create<arith::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
    
    auto newFor = builder.create<scf::ForOp>(loc, c0, cNumStage, c1, ValueRange{});
    Value innerIV = newFor.getInductionVar();
    
    for (NamedAttribute attr : scopeOp_->getAttrs()) {
      if (attr.getName() != "operand_segment_sizes") {
        newFor->setAttr(attr.getName(), attr.getValue());
      }
    }
    
    Block *scopeBody = &scopeRegion->front();
    Block *newBody = newFor.getBody();
    Operation *terminator = newBody->getTerminator();
    
    SmallVector<Operation *> opsToMove;
    for (Operation &op : scopeBody->getOperations()) {
      if (!isa<scope::ReturnOp>(op)) {
        opsToMove.push_back(&op);
      }
    }
    
    for (Operation *op : opsToMove) {
      op->moveBefore(terminator);
    }
    
    // 在内层循环体开头生成 newOuterExpr = outerIV * numStage + innerIV
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(newBody, newBody->begin());
    Value constNumStage = builder.create<arith::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(numStage_));
    Value mul = builder.create<arith::MulIOp>(loc, outerIV_, constNumStage);
    Value addI32 = builder.create<arith::AddIOp>(loc, mul, innerIV);   // i32 类型
    Value addIdx = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), addI32); // index 类型
    
    // 调整操作，确保新操作插入在 addIdx 定义之后
    adjustOperationsInLoop(newFor, addI32, addIdx, numStage_);
    
    // 处理外层循环体中定义的、被内层循环体使用的比较操作
    handleOuterCmpOps(newFor, newBody, addI32, builder, loc);
    
    // 替换新循环体内所有对 outerIV_ 的使用为 addIdx，但跳过我们刚刚创建的操作本身
    Operation *mulOp = mul.getDefiningOp();
    Operation *addOp = addI32.getDefiningOp();
    Operation *idxCastOp = addIdx.getDefiningOp();
    for (Operation &op : llvm::make_early_inc_range(*newBody)) {
      if (&op == mulOp || &op == addOp || &op == idxCastOp) continue;
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        if (op.getOperand(i) == outerIV_) {
          op.setOperand(i, addIdx);
        }
      }
    }
    
    scopeOp_->erase();
    return true;
  }
  
private:
  // 递归查找一个值是否由 baseI32 乘以某个常数得到，返回该常数
  std::optional<int64_t> getIndexFactor(Value val, Value baseI32) {
    llvm::errs() << "  [getIndexFactor] Checking value: ";
    val.print(llvm::errs());
    llvm::errs() << "\n";
    
    if (val == baseI32) {
      llvm::errs() << "  [getIndexFactor] Found direct match (factor=1)\n";
      return 1;
    }
    
    // 处理 arith::IndexCastOp
    if (auto indexCast = val.getDefiningOp<arith::IndexCastOp>()) {
      llvm::errs() << "  [getIndexFactor] Following IndexCastOp\n";
      return getIndexFactor(indexCast.getIn(), baseI32);
    }
    
    // 处理 arith::MulIOp
    if (auto mulOp = val.getDefiningOp<arith::MulIOp>()) {
      auto lhs = mulOp.getLhs();
      auto rhs = mulOp.getRhs();
      // 尝试 lhs 是常数，rhs 可递归到 baseI32
      if (auto constOp = lhs.getDefiningOp<arith::ConstantOp>()) {
        if (auto factor = constOp.getValue().dyn_cast<IntegerAttr>()) {
          if (auto subFactor = getIndexFactor(rhs, baseI32)) {
            llvm::errs() << "  [getIndexFactor] Found mul: " << factor.getInt() << " * " << *subFactor << " = " << (factor.getInt() * *subFactor) << "\n";
            return factor.getInt() * (*subFactor);
          }
        }
      }
      // 尝试 rhs 是常数，lhs 可递归到 baseI32
      if (auto constOp = rhs.getDefiningOp<arith::ConstantOp>()) {
        if (auto factor = constOp.getValue().dyn_cast<IntegerAttr>()) {
          if (auto subFactor = getIndexFactor(lhs, baseI32)) {
            llvm::errs() << "  [getIndexFactor] Found mul: " << factor.getInt() << " * " << *subFactor << " = " << (factor.getInt() * *subFactor) << "\n";
            return factor.getInt() * (*subFactor);
          }
        }
      }
    }
    
    llvm::errs() << "  [getIndexFactor] No match\n";
    return std::nullopt;
  }
  
  void handleOuterCmpOps(scf::ForOp innerFor, Block *innerBody, Value addI32, OpBuilder &builder, Location loc) {
    Block *outerBlock = innerFor->getBlock();
    SmallVector<arith::CmpIOp> cmpsToClone;
    for (Operation &op : *outerBlock) {
      if (auto cmpOp = dyn_cast<arith::CmpIOp>(&op)) {
        bool depends = false;
        for (Value operand : cmpOp.getOperands()) {
          if (getIndexFactor(operand, outerIV_).has_value()) {
            depends = true;
            break;
          }
        }
        if (depends) {
          bool usedInInner = false;
          for (Operation *user : cmpOp.getResult().getUsers()) {
            if (user->getBlock() == innerBody) {
              usedInInner = true;
              break;
            }
          }
          if (usedInInner) {
            cmpsToClone.push_back(cmpOp);
          }
        }
      }
    }
    
    // 设置插入点为 addI32 定义之后
    builder.setInsertionPoint(innerBody, ++addI32.getDefiningOp()->getIterator());
    
    for (auto cmpOp : cmpsToClone) {
      SmallVector<Value> newOperands;
      for (Value operand : cmpOp.getOperands()) {
        auto factor = getIndexFactor(operand, outerIV_);
        if (factor.has_value()) {
          if (*factor == 1) {
            newOperands.push_back(addI32);
          } else {
            Value constFactor = builder.create<arith::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(*factor));
            Value newI32 = builder.create<arith::MulIOp>(loc, addI32, constFactor);
            if (operand.getType().isIndex()) {
              newOperands.push_back(builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), newI32));
            } else {
              newOperands.push_back(newI32);
            }
          }
        } else {
          newOperands.push_back(operand);
        }
      }
      auto newCmp = builder.create<arith::CmpIOp>(loc, cmpOp.getPredicate(), newOperands[0], newOperands[1]);
      cmpOp.getResult().replaceUsesWithIf(newCmp.getResult(), [&](OpOperand &use) {
        return use.getOwner()->getBlock() == innerBody;
      });
    }
  }
  
  void adjustOperationsInLoop(scf::ForOp forOp, Value addI32, Value addIdx, int32_t numStage) {
    Block *body = forOp.getBody();
    
    // 将 builder 的插入点设置为 addIdx 定义之后（即 addI32 和 addIdx 都已定义）
    OpBuilder builder(body, ++(addIdx.getDefiningOp()->getIterator()));
    Location loc = forOp.getLoc();
    
    Value inductionVar = forOp.getInductionVar();
    Value indexIv = inductionVar;
    if (inductionVar.getType() != builder.getIndexType()) {
      indexIv = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), inductionVar);
    }
    
    // 收集所有需要转换的 subview 和 copy
    SmallVector<memref::SubViewOp> workspaceSubviews;
    SmallVector<memref::CopyOp> copiesToAdjust;
    for (Operation &op : llvm::make_early_inc_range(*body)) {
      if (auto subview = dyn_cast<memref::SubViewOp>(&op)) {
        bool isWorkspace = false;
        for (Value ws : workspaceValues_) {
          if (subview.getSource() == ws) {
            workspaceSubviews.push_back(subview);
            isWorkspace = true;
            break;
          }
        }
        if (!isWorkspace) {
          // 非 workspace 的 subview，可能需要替换偏移量
          adjustGlobalSubviewOffset(subview, addI32, builder);
        }
      }
      else if (auto copyOp = dyn_cast<memref::CopyOp>(&op)) {
        for (Value ws : workspaceValues_) {
          if (copyOp.getSource() == ws || copyOp.getTarget() == ws) {
            copiesToAdjust.push_back(copyOp);
            break;
          }
        }
      }
    }
    
    // 处理 workspace 的 subview（需要添加 stage 维度并 collapse shape）
    for (auto subview : workspaceSubviews) {
      adjustWorkspaceSubviewOp(subview, indexIv, builder, addI32, numStage);
    }
    // 处理 copy 操作（涉及 workspace 的）
    for (auto copyOp : copiesToAdjust) {
      adjustCopyOp(copyOp, indexIv, builder, copyOp.getSource() == workspaceValues_[0] /* simplified */, copyOp.getTarget() == workspaceValues_[0], addI32, numStage);
    }
  }
  
  void adjustWorkspaceSubviewOp(memref::SubViewOp subview, Value indexIv, OpBuilder &builder,
                                Value addI32, int32_t numStage) {
    Location loc = subview.getLoc();
    Value source = subview.getSource();
    auto currentSourceType = source.getType().cast<MemRefType>();
    
    auto origOffsets = subview.getMixedOffsets();
    auto origSizes = subview.getMixedSizes();
    auto origStrides = subview.getMixedStrides();
    
    SmallVector<OpFoldResult> newOffsets;
    for (size_t i = 0; i < origOffsets.size(); ++i) {
      auto ofr = origOffsets[i];
      llvm::errs() << "  [adjustWorkspaceSubviewOp] Processing offset[" << i << "]: ";
      if (auto attr = ofr.dyn_cast<Attribute>()) {
        if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
          llvm::errs() << "constant " << intAttr.getInt();
        } else {
          llvm::errs() << "attribute ";
        }
      } else if (auto val = ofr.dyn_cast<Value>()) {
        val.print(llvm::errs());
      }
      llvm::errs() << "\n";
      
      if (auto val = ofr.dyn_cast<Value>()) {
        auto factor = getIndexFactor(val, outerIV_);
        if (factor.has_value()) {
          llvm::errs() << "  [adjustWorkspaceSubviewOp] Found factor " << *factor << " for this offset\n";
          // 在 i32 上进行乘法，然后转换为 index
          Value constFactor = builder.create<arith::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(*factor));
          Value newI32 = builder.create<arith::MulIOp>(loc, addI32, constFactor);
          Value newIdx = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), newI32);
          newOffsets.push_back(newIdx);
          llvm::errs() << "  [adjustWorkspaceSubviewOp] Replaced with new index: ";
          newIdx.print(llvm::errs());
          llvm::errs() << "\n";
          continue;
        } else {
          llvm::errs() << "  [adjustWorkspaceSubviewOp] No factor found, keeping original\n";
        }
      }
      newOffsets.push_back(ofr);
    }
    
    LLVM_DEBUG(DBGS() << "  Calling createStageSubview with Rank " << currentSourceType.getRank() << "\n");
    
    Value newResult = createStageSubview(
        builder, loc, source, currentSourceType, indexIv,
        newOffsets, origSizes, origStrides);
    
    subview.replaceAllUsesWith(newResult);
    subview.erase();
  }
  
  void adjustGlobalSubviewOffset(memref::SubViewOp subview, Value addI32, OpBuilder &builder) {
    Location loc = subview.getLoc();
    auto origOffsets = subview.getMixedOffsets();
    auto origSizes = subview.getMixedSizes();
    auto origStrides = subview.getMixedStrides();
    
    SmallVector<OpFoldResult> newOffsets;
    for (size_t i = 0; i < origOffsets.size(); ++i) {
      auto ofr = origOffsets[i];
      if (auto val = ofr.dyn_cast<Value>()) {
        auto factor = getIndexFactor(val, outerIV_);
        if (factor.has_value()) {
          if (*factor == 1) {
            newOffsets.push_back(addI32);
          } else {
            Value constFactor = builder.create<arith::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(*factor));
            Value newI32 = builder.create<arith::MulIOp>(loc, addI32, constFactor);
            Value newIdx = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), newI32);
            newOffsets.push_back(newIdx);
          }
          continue;
        }
      }
      newOffsets.push_back(ofr);
    }
    
    // 创建新的 subview，结果类型由 MLIR 自动推导
    auto newSubview = builder.create<memref::SubViewOp>(
        loc, subview.getSource(), newOffsets, origSizes, origStrides);
    subview.replaceAllUsesWith(newSubview.getResult());
    subview.erase();
  }
  
  void adjustCopyOp(memref::CopyOp copyOp, Value indexIv, OpBuilder &builder,
                    bool fixSource, bool fixTarget, Value addI32, int32_t numStage) {
    Location loc = copyOp.getLoc();
    
    Value ws = nullptr;
    for (Value candidate : workspaceValues_) {
      if (copyOp.getSource() == candidate || copyOp.getTarget() == candidate) {
        ws = candidate;
        break;
      }
    }
    
    if (!ws) return;

    auto wsType = ws.getType().cast<MemRefType>();
    
    SmallVector<OpFoldResult> offsets, sizes, strides;
    offsets.push_back(indexIv);
    sizes.push_back(builder.getIndexAttr(1));
    strides.push_back(builder.getIndexAttr(1));
    
    for (int i = 1; i < wsType.getRank(); ++i) {
        offsets.push_back(builder.getIndexAttr(0));
        if (wsType.isDynamicDim(i)) {
            sizes.push_back(builder.createOrFold<memref::DimOp>(loc, ws, i));
        } else {
            sizes.push_back(builder.getIndexAttr(wsType.getDimSize(i)));
        }
        strides.push_back(builder.getIndexAttr(1));
    }
    
    auto subviewOp = builder.create<memref::SubViewOp>(loc, ws, offsets, sizes, strides);
    
    auto subviewType = subviewOp.getResult().getType().cast<MemRefType>();
    SmallVector<ReassociationIndices> reassociation;
    ReassociationIndices currentGroup;
    bool mergedLeadingOnes = false;

    for (int i = 0; i < subviewType.getRank(); ++i) {
      int64_t dimSize = subviewType.getDimSize(i);
      if (!mergedLeadingOnes && dimSize == 1) {
        currentGroup.push_back(i);
      } else {
        if (!currentGroup.empty()) {
          currentGroup.push_back(i);
          reassociation.push_back(currentGroup);
          currentGroup.clear();
          mergedLeadingOnes = true;
        } else {
          reassociation.push_back({i});
        }
      }
    }
    if (!currentGroup.empty()) {
      reassociation.push_back(currentGroup);
    }

    Value slicedWs = builder.create<memref::CollapseShapeOp>(loc, subviewOp.getResult(), reassociation);
    
    if (fixSource) {
      builder.create<memref::CopyOp>(loc, slicedWs, copyOp.getTarget());
    } else {
      builder.create<memref::CopyOp>(loc, copyOp.getSource(), slicedWs);
    }
    copyOp.erase();
  }
  
  Operation *scopeOp_;
  SmallVector<Value> workspaceValues_;
  int32_t numStage_;
  Value outerIV_;
};

class PipelineLoopProcessor {
public:
  PipelineLoopProcessor(scf::ForOp pipelineLoop, ArrayRef<Value> workspaceValues)
      : pipelineLoop_(pipelineLoop), workspaceValues_(workspaceValues) {}
  
  bool process() {
    auto attr = pipelineLoop_->getAttrOfType<IntegerAttr>("tilelangir.num_stages");
    if (!attr) return false;
    
    int32_t numStage = static_cast<int32_t>(attr.getInt());
    LLVM_DEBUG(DBGS() << "Processing pipeline loop with tilelangir.num_stages=" << numStage 
                      << ", workspace count=" << workspaceValues_.size() << "\n");
    
    // 修改外层循环上界：new_upper = old_upper / numStage
    Value oldUpper = pipelineLoop_.getUpperBound();
    OpBuilder builder(pipelineLoop_);
    Location loc = pipelineLoop_.getLoc();
    Value constNumStage = builder.create<arith::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(numStage));
    Value newUpper = builder.create<arith::DivSIOp>(loc, oldUpper, constNumStage);
    pipelineLoop_.setUpperBound(newUpper);
    
    SmallVector<Operation *> scopesToReplace;
    for (Operation &op : pipelineLoop_.getBody()->getOperations()) {
      if (isa<scope::ScopeOp>(op) || op.getName().getStringRef() == "scope.scope") {
        scopesToReplace.push_back(&op);
      }
    }
    
    bool changed = false;
    for (Operation *scopeOp : scopesToReplace) {
      ScopeToForConverter converter(scopeOp, workspaceValues_, numStage, pipelineLoop_.getInductionVar());
      if (converter.convert()) changed = true;
    }
    return changed;
  }
  
private:
  scf::ForOp pipelineLoop_;
  SmallVector<Value> workspaceValues_;
};

namespace {
struct TileLangIREnableMultiBuffer
    : public impl::TileLangIREnableMultiBufferBase<TileLangIREnableMultiBuffer> {
  using Base = impl::TileLangIREnableMultiBufferBase<TileLangIREnableMultiBuffer>;
  using Base::Base;

public:
  void runOnOperation() override;
};
} // end anonymous namespace

void TileLangIREnableMultiBuffer::runOnOperation() {
  ModuleOp module = getOperation();
  if (!module) return;
  
  LLVM_DEBUG(DBGS() << "Starting EnableMultiBuffer pass\n");
  g_subviewProcessedCount = 0;
  
  if (!WorkspaceExpander::expand(module)) {
    LLVM_DEBUG(DBGS() << "No workspaces to expand.\n");
    return;
  }
  
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    SmallVector<Value> expandedWorkspaces;
    func.walk([&](memref_ext::AllocWorkspaceOp allocOp) {
      auto type = allocOp.getType().cast<MemRefType>();
      if (type.getRank() > 1 && type.getShape()[0] > 1) { 
         expandedWorkspaces.push_back(allocOp.getResult());
      }
    });
    
    if (expandedWorkspaces.empty()) continue;
    
    LLVM_DEBUG(DBGS() << "Found " << expandedWorkspaces.size() << " expanded workspaces in function " 
                      << func.getSymName() << "\n");

    func.walk([&](scf::ForOp forOp) {
      if (forOp->getAttr("tilelangir.num_stages")) {
          bool usesAnyWs = false;
          forOp.walk([&](Operation *op) {
              if (auto sv = dyn_cast<memref::SubViewOp>(op)) {
                  for (Value ws : expandedWorkspaces) {
                      if (sv.getSource() == ws) { 
                          usesAnyWs = true; 
                          return; 
                      }
                  }
              } else if (auto cp = dyn_cast<memref::CopyOp>(op)) {
                  for (Value ws : expandedWorkspaces) {
                      if (cp.getSource() == ws || cp.getTarget() == ws) { 
                          usesAnyWs = true; 
                          return; 
                      }
                  }
              }
          });
          
          if (usesAnyWs) {
              PipelineLoopProcessor processor(forOp, expandedWorkspaces);
              processor.process();
          }
      }
    });
  }
  
  LLVM_DEBUG(DBGS() << "Total SubViews processed: " << g_subviewProcessedCount << "\n");
  if (g_subviewProcessedCount != 6) {
    llvm::errs() << "WARNING: Expected 6 subviews to be processed, but got " << g_subviewProcessedCount << ".\n";
    llvm::errs() << "This implies some subviews were skipped or the pass crashed early.\n";
  }
}

} // namespace tilelangir
} // namespace mlir