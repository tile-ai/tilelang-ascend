// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/lib/Transforms/SplitMixKernel.cpp
 * \brief Split mixed kernel into AIC and AIV functions.
 */

#include "tilelangir/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/IR/HIVMTraits.h"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRSPLITMIXKERNEL
#include "tilelangir/Transforms/Passes.h.inc"

} // namespace tilelangir
} // namespace mlir

#define DEBUG_TYPE "tilelangir-split-mix-kernel"

using namespace mlir;
using namespace mlir::tilelangir;
using namespace mlir::hivm;

namespace {

// 获取内存空间字符串（如果存在）
static std::optional<std::string> getMemSpaceString(MemRefType type) {
  if (auto space = type.getMemorySpace()) {
    if (auto addrSpace = space.dyn_cast<AddressSpaceAttr>()) {
      AddressSpace as = addrSpace.getAddressSpace();
      switch (as) {
        case AddressSpace::GM: return "gm";
        case AddressSpace::UB: return "ub";
        case AddressSpace::L1: return "cbuf";
        case AddressSpace::L0A: return "ca";
        case AddressSpace::L0B: return "cb";
        case AddressSpace::L0C: return "cc";
        default: return "unknown";
      }
    }
  }
  return std::nullopt;
}

// 判断操作是否应该被删除（仅针对计算指令）
static bool shouldDeleteComputationOp(Operation *op, bool isAIC) {
  // 1. 处理 scf.for 循环
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    if (auto attr = forOp->getAttrOfType<TCoreTypeAttr>("hivm.tcore_type")) {
      if (isAIC) {
        // AIC: 保留 CUBE，删除 VECTOR
        return attr.getTcoretype() == TCoreType::VECTOR;
      } else {
        // AIV: 保留 VECTOR，删除 CUBE
        return attr.getTcoretype() == TCoreType::CUBE;
      }
    }
    // 没有标签的循环保留
    return false;
  }

  // 2. 处理具体的计算指令 (HIVM 指令等)
  // 原则：如果指令的操作数包含不该出现的 Memory Space，则删除该指令
  for (Value operand : op->getOperands()) {
    if (auto memRefType = operand.getType().dyn_cast<MemRefType>()) {
      if (auto space = getMemSpaceString(memRefType)) {
        if (isAIC) {
          // AIC 函数中不应该出现 UB 空间的操作
          if (*space == "ub") return true;
        } else {
          // AIV 函数中不应该出现 CUBE 相关空间的操作
          if (*space == "cbuf" || *space == "cc") return true;
        }
      }
    }
  }
  return false;
}

// 判断该 Op 是否为 "内存资源申请类" Op (Alloc, View, Subview 等)
// 这些 Op 本身不参与计算，仅用于资源管理，如果其 Result 无人使用，必须被回收
static bool isMemoryResourceOp(Operation *op) {
  // 包含 Alloc 以及各种 View 变换
  return isa<memref::AllocOp, memref::AllocaOp, memref::ViewOp, 
              memref::ReinterpretCastOp, memref::SubViewOp>(op);
}

// 核心修复：迭代删除直到不动点
static void filterOpsRobust(func::FuncOp func, bool isAIC) {
  if (func.isExternal()) return;

  bool changed;
  // 使用 do-while 循环实现迭代死代码消除 (Iterative DCE)
  // 原理：先删计算指令 -> 导致 Alloc 变成死代码 -> 删 Alloc -> 可能导致更上游的 View 变成死代码 -> ...
  do {
    changed = false;
    SmallVector<Operation *> toEraseThisRound;

    // Step 1: 遍历所有操作，标记需要删除的计算指令
    func.walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (shouldDeleteComputationOp(op, isAIC)) {
        // 只有当操作没有被使用时才能安全删除（或者确保是死代码）
        // 这里我们先标记，后续统一处理
        toEraseThisRound.push_back(op);
      }
    });

    // Step 2: 遍历所有操作，标记 "内存资源类" 的死代码
    // 即：是 Alloc/View 类型，且没有任何使用者 (use_empty)
    func.walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (isMemoryResourceOp(op) && op->use_empty()) {
        // 额外防御：不要删除函数参数相关的 View（虽然通常不会发生）
        // 如果该 Op 的 Result 没有任何 User，则标记删除
        toEraseThisRound.push_back(op);
      }
    });

    // Step 3: 执行删除
    for (auto op : toEraseThisRound) {
      // 再次检查 use_empty，防止在本轮循环中该 Op 被其他新创建的节点使用（虽然这里不会创建）
      if (op->use_empty()) {
        LLVM_DEBUG(llvm::dbgs() << "Erasing dead code: " << op << "\n");
        op->erase();
        changed = true;
      }
    }

  } while (changed); // 循环直到这一轮没有删除任何节点
}

// 移除保留循环上的 tcore_type 属性
static void stripLoopAttributes(func::FuncOp func, TCoreType targetType) {
  func.walk([&](scf::ForOp forOp) {
    if (auto attr = forOp->getAttrOfType<TCoreTypeAttr>("hivm.tcore_type")) {
      if (attr.getTcoretype() == targetType) {
        forOp->removeAttr("hivm.tcore_type");
      }
    }
  });
}

// 拆分混合内核
static LogicalResult splitMixKernel(func::FuncOp func) {
  // 检查是否为混合内核
  if (!func->hasAttr("mix_mode")) return success();
  auto mixMode = func->getAttrOfType<StringAttr>("mix_mode");
  if (!mixMode || mixMode.getValue() != "aic") return success();

  StringRef origName = func.getSymName();
  std::string aicName = (origName + "_mix_aic").str();
  std::string aivName = (origName + "_mix_aiv").str();

  // 获取模块并修改其 core_type 为 MIX
  auto module = func->getParentOfType<ModuleOp>();
  if (module) {
    // 修正：属性名改为 hivm.module_core_type
    module->setAttr("hivm.module_core_type",
                    TModuleCoreTypeAttr::get(module.getContext(), TModuleCoreType::MIX));
  }

  // 重命名原函数为 AIC 版本
  func.setSymName(aicName);
  // 修改属性
  func->removeAttr("mix_mode");
  func->setAttr("mix_mode", StringAttr::get(func.getContext(), "mix"));
  func->setAttr("hivm.func_core_type",
                TFuncCoreTypeAttr::get(func.getContext(), TFuncCoreType::AIC));
  func->setAttr("hivm.part_of_mix", UnitAttr::get(func.getContext()));

  // 克隆得到 AIV 函数
  OpBuilder builder(module);
  builder.setInsertionPointAfter(func);
  auto aivFunc = cast<func::FuncOp>(builder.clone(*func.getOperation()));
  aivFunc.setSymName(aivName);
  aivFunc->setAttr("hivm.func_core_type",
                   TFuncCoreTypeAttr::get(aivFunc.getContext(), TFuncCoreType::AIV));
  aivFunc->setAttr("hivm.part_of_mix", UnitAttr::get(aivFunc.getContext()));
  aivFunc->setAttr("mix_mode", StringAttr::get(aivFunc.getContext(), "mix"));

  // 过滤操作：使用修复后的鲁棒版本
  filterOpsRobust(func, true);   // AIC: 删除 Vector 和 UB 相关
  filterOpsRobust(aivFunc, false); // AIV: 删除 Cube 和 CBUF/CC 相关

  // 删除循环上的 tcore_type 属性
  stripLoopAttributes(func, TCoreType::CUBE);
  stripLoopAttributes(aivFunc, TCoreType::VECTOR);

  return success();
}

} // namespace

namespace mlir {
namespace tilelangir {

struct TileLangIRSplitMixKernel
    : public impl::TileLangIRSplitMixKernelBase<TileLangIRSplitMixKernel> {
  void runOnOperation() override {
    auto module = getOperation();

    SmallVector<func::FuncOp> toSplit;
    module.walk([&](func::FuncOp func) {
      if (auto mix = func->getAttrOfType<StringAttr>("mix_mode")) {
        if (mix.getValue() == "aic") {
          toSplit.push_back(func);
        }
      }
    });

    for (auto func : toSplit) {
      if (failed(splitMixKernel(func))) {
        signalPassFailure();
        return;
      }
    }
  }
};

std::unique_ptr<Pass> createTileLangIRSplitMixKernelPass() {
  return std::make_unique<TileLangIRSplitMixKernel>();
}

} // namespace tilelangir
} // namespace mlir