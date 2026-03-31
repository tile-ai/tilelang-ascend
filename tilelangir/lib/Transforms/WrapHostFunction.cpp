// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/lib/Transforms/WrapHostFunction.cpp
 * \brief Generate host-side functions encoding kernel parameter signatures.
 *
 * For each device entry kernel two host functions are injected:
 *   {baseName}_get_kernel_num_args  – returns the number of user arguments
 *   {baseName}_get_kernel_arg_type  – takes an i32 index, returns type code
 *
 * This approach supports an arbitrary number of user arguments (no packing
 * limit) and is trivially callable via ctypes from the JIT runtime.
 */

#include "tilelangir/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSet.h"

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRWRAPHOSTFUNCTION
#include "tilelangir/Transforms/Passes.h.inc"

} // namespace tilelangir
} // namespace mlir

#define DEBUG_TYPE "tilelangir-wrap-host-function"

using namespace mlir;
using namespace mlir::tilelangir;

namespace {

// ---- Type encoding (must stay in sync with jit_npu.py decoding) -----------

enum ElementTypeCode : int {
  kFP32 = 0,
  kFP16 = 1,
  kBF16 = 2,
  kI8 = 3,
  kI16 = 4,
  kI32 = 5,
  kI64 = 6,
  kU32 = 7,
  kU64 = 8,
  kFP64 = 9,
  kI1 = 10,
  kUnknown = 0x7F,
  kPointerFlag = 0x80,
};

static int encodeElementType(Type type) {
  if (type.isF32())
    return kFP32;
  if (type.isF16())
    return kFP16;
  if (type.isBF16())
    return kBF16;
  if (type.isInteger(8))
    return kI8;
  if (type.isInteger(16))
    return kI16;
  if (type.isInteger(32))
    return kI32;
  if (type.isInteger(64))
    return kI64;
  if (type.isF64())
    return kFP64;
  if (type.isInteger(1))
    return kI1;
  if (auto intTy = dyn_cast<IntegerType>(type)) {
    if (intTy.isUnsigned()) {
      if (intTy.getWidth() == 32)
        return kU32;
      if (intTy.getWidth() == 64)
        return kU64;
    }
  }
  return kUnknown;
}

static int encodeArgType(Type type) {
  if (auto memrefTy = dyn_cast<MemRefType>(type))
    return kPointerFlag | encodeElementType(memrefTy.getElementType());
  return encodeElementType(type);
}

static std::string deriveBaseName(StringRef funcName) {
  if (funcName.ends_with("_mix_aic"))
    return funcName.drop_back(8).str();
  if (funcName.ends_with("_mix_aiv"))
    return funcName.drop_back(8).str();
  return funcName.str();
}

static constexpr unsigned kCompilerPrefixArgs = 3;
static constexpr unsigned kGridSuffixArgs = 6;

/// Create `int {name}()` that returns \p numArgs.
static func::FuncOp createNumArgsHostFunc(func::FuncOp entryFunc,
                                          StringRef hostFuncName,
                                          int numArgs) {
  OpBuilder builder(entryFunc.getContext());
  builder.setInsertionPoint(entryFunc);
  Location loc = entryFunc.getLoc();
  auto i32Ty = builder.getI32Type();

  auto hostFunc = builder.create<func::FuncOp>(
      loc, hostFuncName,
      FunctionType::get(entryFunc.getContext(), /*inputs=*/{},
                        /*results=*/{i32Ty}));

  Block *block = hostFunc.addEntryBlock();
  builder.setInsertionPointToStart(block);

  auto val =
      builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(numArgs));
  builder.create<func::ReturnOp>(loc, ValueRange{val});

  hacc::utils::setHost(hostFunc);
  return hostFunc;
}

/// Create `int {name}(int index)` that maps index → type code via
/// a chain of `arith.select`, supporting an arbitrary number of entries.
static func::FuncOp createArgTypeHostFunc(func::FuncOp entryFunc,
                                          StringRef hostFuncName,
                                          ArrayRef<int> typeCodes) {
  OpBuilder builder(entryFunc.getContext());
  builder.setInsertionPoint(entryFunc);
  Location loc = entryFunc.getLoc();
  auto i32Ty = builder.getI32Type();

  auto hostFunc = builder.create<func::FuncOp>(
      loc, hostFuncName,
      FunctionType::get(entryFunc.getContext(), /*inputs=*/{i32Ty},
                        /*results=*/{i32Ty}));

  Block *block = hostFunc.addEntryBlock();
  builder.setInsertionPointToStart(block);

  Value indexArg = block->getArgument(0);

  // Default for out-of-range index
  Value result =
      builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(-1));

  // Build select chain from last to first so that index 0 is outermost
  for (int i = static_cast<int>(typeCodes.size()) - 1; i >= 0; --i) {
    Value idx =
        builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(i));
    Value typeVal = builder.create<arith::ConstantOp>(
        loc, builder.getI32IntegerAttr(typeCodes[i]));
    Value cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                              indexArg, idx);
    result = builder.create<arith::SelectOp>(loc, cmp, typeVal, result);
  }

  builder.create<func::ReturnOp>(loc, ValueRange{result});

  hacc::utils::setHost(hostFunc);
  return hostFunc;
}

} // anonymous namespace

namespace mlir {
namespace tilelangir {

struct TileLangIRWrapHostFunction
    : public impl::TileLangIRWrapHostFunctionBase<
          TileLangIRWrapHostFunction> {
  void runOnOperation() override {
    auto module = getOperation();

    llvm::StringSet<> processedBaseNames;
    SmallVector<func::FuncOp> entryFuncs;

    module.walk([&](func::FuncOp func) {
      if (func->hasAttr("hacc.entry") || func->hasAttr("hivm.entry"))
        entryFuncs.push_back(func);
    });

    for (auto entryFunc : entryFuncs) {
      std::string baseName = deriveBaseName(entryFunc.getSymName());
      if (!processedBaseNames.insert(baseName).second)
        continue;

      unsigned totalArgs = entryFunc.getNumArguments();
      unsigned numUserArgs = 0;
      SmallVector<int> typeCodes;

      if (totalArgs >= kCompilerPrefixArgs + kGridSuffixArgs) {
        numUserArgs = totalArgs - kCompilerPrefixArgs - kGridSuffixArgs;
        auto argTypes = entryFunc.getArgumentTypes();
        for (unsigned i = 0; i < numUserArgs; ++i) {
          typeCodes.push_back(encodeArgType(argTypes[kCompilerPrefixArgs + i]));
        }
      }

      createNumArgsHostFunc(entryFunc, baseName + "_get_kernel_num_args",
                            static_cast<int>(numUserArgs));
      createArgTypeHostFunc(entryFunc, baseName + "_get_kernel_arg_type",
                            typeCodes);
    }
  }
};

std::unique_ptr<Pass> createTileLangIRWrapHostFunctionPass() {
  return std::make_unique<TileLangIRWrapHostFunction>();
}

} // namespace tilelangir
} // namespace mlir
