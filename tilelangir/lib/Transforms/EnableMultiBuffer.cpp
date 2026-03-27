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
  ScopeToForConverter(Operation *scopeOp, ArrayRef<Value> workspaceValues, int32_t numStage)
      : scopeOp_(scopeOp), workspaceValues_(workspaceValues), numStage_(numStage) {}
  
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
    
    adjustOperationsInLoop(newFor);
    
    scopeOp_->erase();
    return true;
  }
  
private:
  void adjustOperationsInLoop(scf::ForOp forOp) {
    Value inductionVar = forOp.getInductionVar();
    Block *body = forOp.getBody();
    
    OpBuilder builder(body, body->begin());
    Location loc = forOp.getLoc();
    
    Value indexIv = inductionVar;
    if (inductionVar.getType() != builder.getIndexType()) {
      indexIv = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), inductionVar);
    }
    
    processBlock(body, indexIv, builder);
  }

  void processBlock(Block *block, Value indexIv, OpBuilder &builder) {
    for (Operation &op : llvm::make_early_inc_range(*block)) {
      for (auto &reg : op.getRegions()) {
        for (Block &b : reg) {
          processBlock(&b, indexIv, builder);
        }
      }

      if (auto subview = dyn_cast<memref::SubViewOp>(&op)) {
        Value source = subview.getSource();
        for (Value ws : workspaceValues_) {
          if (source == ws) {
            g_subviewProcessedCount++;
            LLVM_DEBUG({
              DBGS() << ">>> ADJUSTING SUBVIEW #" << g_subviewProcessedCount << " <<<\n";
              DBGS() << "  Source Type: "; debugPrintType("", source.getType());
              DBGS() << "  Matching Workspace: "; debugPrintType("", ws.getType());
            });
            
            auto srcType = source.getType().dyn_cast<MemRefType>();
            if (srcType && srcType.getRank() <= 3) {
               LLVM_DEBUG(DBGS() << "  WARNING: Source type rank is " << srcType.getRank() << "\n");
            }

            adjustSubviewOp(subview, indexIv, builder);
            LLVM_DEBUG(DBGS() << ">>> FINISHED SUBVIEW #" << g_subviewProcessedCount << " <<<\n");
            break;
          }
        }
      }
      else if (auto copyOp = dyn_cast<memref::CopyOp>(&op)) {
        Value source = copyOp.getSource();
        Value target = copyOp.getTarget();
        
        for (Value ws : workspaceValues_) {
          if (source == ws || target == ws) {
            LLVM_DEBUG(DBGS() << "Adjusting CopyOp for workspace.\n");
            adjustCopyOp(copyOp, indexIv, builder, source == ws, target == ws);
            break;
          }
        }
      }
    }
  }
  
  void adjustSubviewOp(memref::SubViewOp subview, Value indexIv, OpBuilder &builder) {
    Location loc = subview.getLoc();
    Value source = subview.getSource();
    auto currentSourceType = source.getType().cast<MemRefType>();
    
    auto origOffsets = subview.getMixedOffsets();
    auto origSizes = subview.getMixedSizes();
    auto origStrides = subview.getMixedStrides();

    LLVM_DEBUG(DBGS() << "  Calling createStageSubview with Rank " << currentSourceType.getRank() << "\n");
    
    Value newResult = createStageSubview(
        builder, loc, source, currentSourceType, indexIv,
        origOffsets, origSizes, origStrides);
    
    subview.replaceAllUsesWith(newResult);
    subview.erase();
  }
  
  void adjustCopyOp(memref::CopyOp copyOp, Value indexIv, OpBuilder &builder,
                    bool fixSource, bool fixTarget) {
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
    
    SmallVector<Operation *> scopesToReplace;
    for (Operation &op : pipelineLoop_.getBody()->getOperations()) {
      if (isa<scope::ScopeOp>(op) || op.getName().getStringRef() == "scope.scope") {
        scopesToReplace.push_back(&op);
      }
    }
    
    bool changed = false;
    for (Operation *scopeOp : scopesToReplace) {
      ScopeToForConverter converter(scopeOp, workspaceValues_, numStage);
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