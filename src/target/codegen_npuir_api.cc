// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file target/codegen.cc
 */

#include "codegen_npuir_api.h"
#include "../op/ascend.h"
#include "../op/builtin.h"
#include "arith/pattern_match.h"
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <elf.h>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <tvm/arith/analyzer.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/module.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/codegen.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <utility>
#include <vector>

// For adding MLIR APIs to support codegen
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/Casting.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>

// //===----------------------------------------------------------------------===//
// // HIVM Dialect
// //===----------------------------------------------------------------------===//

// #include "bishengir/Dialect/HIVM/IR/HIVM.h"

// //===----------------------------------------------------------------------===//
// // HFusion Dialect
// //===----------------------------------------------------------------------===//

// #include "bishengir/Dialect/HFusion/IR/HFusion.h"

//===----------------------------------------------------------------------===//
// HACC Dialect
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/IR/HACC.h"

using namespace mlir;

namespace tvm {
namespace codegen {

constexpr uint8_t FLAG_ID_BITS = 64;

static std::map<NPU_CORETYPE, std::string> NPU_CORETYPE_STR{
    {NPU_CORETYPE::AIC, "aic"},
    {NPU_CORETYPE::AIV, "aiv"},
    {NPU_CORETYPE::MIX, "mix"}};

static std::map<NPU_CORETYPE, mlir::hivm::TModuleCoreType>
    NPUIR_MODULECORETYPE_STR{
        {NPU_CORETYPE::AIC, mlir::hivm::TModuleCoreType::AIC},
        {NPU_CORETYPE::AIV, mlir::hivm::TModuleCoreType::AIV},
        {NPU_CORETYPE::MIX, mlir::hivm::TModuleCoreType::MIX}};

static std::map<NPU_CORETYPE, mlir::hivm::TFuncCoreType> NPUIR_FUNCCORETYPE_STR{
    {NPU_CORETYPE::AIC, mlir::hivm::TFuncCoreType::AIC},
    {NPU_CORETYPE::AIV, mlir::hivm::TFuncCoreType::AIV},
    {NPU_CORETYPE::MIX, mlir::hivm::TFuncCoreType::MIX}};

static std::map<int, std::string> coretype_syncblock_map{{0, "CUBE"},
                                                         {1, "VECTOR"}};

static std::map<int, mlir::hivm::FixpipePreReluMode> fixpipe_pre_relu_mode{
    {0, mlir::hivm::FixpipePreReluMode::NO_RELU},
    {1, mlir::hivm::FixpipePreReluMode::NORMAL_RELU},
    {2, mlir::hivm::FixpipePreReluMode::LEAKY_RELU},
    {3, mlir::hivm::FixpipePreReluMode::P_RELU}};

static std::map<std::string, mlir::hivm::PIPE> PIPE_MAP{
    {"PIPE_S", mlir::hivm::PIPE::PIPE_S},
    {"PIPE_V", mlir::hivm::PIPE::PIPE_V},
    {"PIPE_M", mlir::hivm::PIPE::PIPE_M},
    {"PIPE_MTE1", mlir::hivm::PIPE::PIPE_MTE1},
    {"PIPE_MTE2", mlir::hivm::PIPE::PIPE_MTE2},
    {"PIPE_MTE3", mlir::hivm::PIPE::PIPE_MTE3},
    {"PIPE_ALL", mlir::hivm::PIPE::PIPE_ALL},
    {"PIPE_MTE4", mlir::hivm::PIPE::PIPE_MTE4},
    {"PIPE_MTE5", mlir::hivm::PIPE::PIPE_MTE5},
    {"PIPE_V2", mlir::hivm::PIPE::PIPE_V2},
    {"PIPE_FIX", mlir::hivm::PIPE::PIPE_FIX},
    {"VIRTUAL_PIPE_MTE2_L1A", mlir::hivm::PIPE::VIRTUAL_PIPE_MTE2_L1A},
    {"VIRTUAL_PIPE_MTE2_L1B", mlir::hivm::PIPE::VIRTUAL_PIPE_MTE2_L1B},
    {"PIPE_NUM", mlir::hivm::PIPE::PIPE_NUM},
    {"PIPE_UNASSIGNED", mlir::hivm::PIPE::PIPE_UNASSIGNED},
};

static std::map<std::string, mlir::hivm::CompareMode> COMPARE_MODE{
    {"eq", mlir::hivm::CompareMode::EQ}, {"ne", mlir::hivm::CompareMode::NE},
    {"lt", mlir::hivm::CompareMode::LT}, {"gt", mlir::hivm::CompareMode::GT},
    {"ge", mlir::hivm::CompareMode::GE}, {"le", mlir::hivm::CompareMode::LE}};

static std::map<NPU_CORETYPE, mlir::hivm::TCoreType> TCORE_MAP{
    {NPU_CORETYPE::AIC, mlir::hivm::TCoreType::CUBE},
    {NPU_CORETYPE::AIV, mlir::hivm::TCoreType::VECTOR}};

static std::map<tl::SyncBlockMode, mlir::hivm::SyncBlockInstrMode>
    SYNC_BLOCK_MODE_MAP{
        {tl::SyncBlockMode::INTER_BLOCK,
         mlir::hivm::SyncBlockInstrMode::INTER_BLOCK_SYNCHRONIZATION},
        {tl::SyncBlockMode::INTER_SUBBLOCK,
         mlir::hivm::SyncBlockInstrMode::INTER_SUBBLOCK_SYNCHRONIZATION},
        {tl::SyncBlockMode::INTRA_BLOCK,
         mlir::hivm::SyncBlockInstrMode::INTRA_BLOCK_SYNCHRONIZATION},
    };

static llvm::SmallVector<int64_t>
getBroadcastDim(const Array<PrimExpr> &buffer_shape0,
                const Array<PrimExpr> &buffer_shape1) {
  llvm::SmallVector<int64_t> dims;
  if (buffer_shape0.empty() || buffer_shape1.empty()) {
    return dims;
  }
  CHECK(buffer_shape0.size() == buffer_shape1.size());
  for (int i = 0; i < buffer_shape0.size(); i++) {
    if (*as_const_int(buffer_shape0[i]) == 1 &&
        *as_const_int(buffer_shape1[i]) != 1) {
      dims.emplace_back(i);
    } else if (*as_const_int(buffer_shape0[i]) != 1 &&
               *as_const_int(buffer_shape1[i]) == 1) {
      dims.emplace_back(i);
    } else {
      CHECK(*as_const_int(buffer_shape0[i]) == *as_const_int(buffer_shape1[i]));
    }
  }
  return dims;
}

static llvm::SmallVector<int64_t>
getBroadcastDim(const llvm::ArrayRef<long int> &buffer_shape0,
                const llvm::ArrayRef<long int> &buffer_shape1) {
  llvm::SmallVector<int64_t> dims;
  if (buffer_shape0.empty() || buffer_shape1.empty()) {
    return dims;
  }
  CHECK(buffer_shape0.size() == buffer_shape1.size());
  for (int i = 0; i < buffer_shape0.size(); i++) {
    if ((buffer_shape0[i]) == 1 &&
        (buffer_shape1[i]) != 1) {
      dims.emplace_back(i);
    } else if ((buffer_shape0[i]) != 1 &&
               (buffer_shape1[i]) == 1) {
      dims.emplace_back(i);
    } else {
      CHECK((buffer_shape0[i]) == (buffer_shape1[i]));
    }
  }
  return dims;
}

static std::map<std::string, mlir::hivm::RoundMode> NPUIR_STR_ROUNDMODE{
    {"round", mlir::hivm::RoundMode::ROUND},
    {"rint", mlir::hivm::RoundMode::RINT},
    {"floor", mlir::hivm::RoundMode::FLOOR},
    {"ceil", mlir::hivm::RoundMode::CEIL},
    {"trunc", mlir::hivm::RoundMode::TRUNC},
    {"odd", mlir::hivm::RoundMode::ODD}};

static std::map<std::string, mlir::hivm::ReduceOperation> NPUIR_STR_REDUCEOP{
    {"sum", mlir::hivm::ReduceOperation::sum},
    {"prod", mlir::hivm::ReduceOperation::prod},
    {"max", mlir::hivm::ReduceOperation::max},
    {"min", mlir::hivm::ReduceOperation::min},
    {"max_with_index_left", mlir::hivm::ReduceOperation::max_with_index_left},
    {"max_with_index_right", mlir::hivm::ReduceOperation::max_with_index_right},
    {"min_with_index_left", mlir::hivm::ReduceOperation::min_with_index_left},
    {"min_with_index_right", mlir::hivm::ReduceOperation::min_with_index_right},
    {"any", mlir::hivm::ReduceOperation::any},
    {"all", mlir::hivm::ReduceOperation::all},
    {"xori", mlir::hivm::ReduceOperation::xori},
    {"ori", mlir::hivm::ReduceOperation::ori},
    {"none", mlir::hivm::ReduceOperation::none},
};

static std::map<std::string, mlir::hivm::DeinterleaveMode>
    NPUIR_STR_DEINTERLEAVEMODE{
        {"CHANNEL_0", mlir::hivm::DeinterleaveMode::CHANNEL_0},
        {"CHANNEL_1", mlir::hivm::DeinterleaveMode::CHANNEL_1},
        {"ALL_CHANNELS", mlir::hivm::DeinterleaveMode::ALL_CHANNELS},
    };

std::vector<int64_t> GetStrideFromShapeAPI(Array<tvm::PrimExpr> shape) {
  std::vector<int64_t> strides;
  int64_t total_size = 1;
  std::vector<int> shape_int;
  for (PrimExpr s : shape) {
    if (auto s_int = as_const_int(s)) {
      total_size *= *s_int;
      shape_int.push_back(*s_int);
    }
  }
  for (int i = 0; i < shape.size(); i++) {
    total_size /= shape_int[i];
    strides.push_back(total_size);
  }
  return strides;
}

namespace {
  /// Infer function core type: aic, aiv, mix
  class InferFuncCoreType : public StmtExprVisitor {
    std::map<std::string, NPU_CORETYPE> scope_coretype_map{
        {"shared", NPU_CORETYPE::AIV},
        {"shared.dyn", NPU_CORETYPE::AIC},
        {"wmma.accumulator", NPU_CORETYPE::AIC},
        {"wmma.matrix_a", NPU_CORETYPE::AIC},
        {"wmma.matrix_b", NPU_CORETYPE::AIC}};
  
  public:
    void VisitStmt(const Stmt &stmt) override {
      StmtExprVisitor::VisitStmt(stmt);
    }
    void VisitStmt_(const AttrStmtNode *op) final {
      // It is mixkernel iff there exists T.rs.
      if (op->attr_key == "resource_scope") {
        func_coretype = NPU_CORETYPE::MIX;
        return;
      }
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitExpr_(const CallNode *op) final {
      // It is cube kernel if there exists T.npuir_dot.
      if (op->op.same_as(Op::Get("tl.npuir_dot"))) {
        if (func_coretype != NPU_CORETYPE::MIX) {
          func_coretype = NPU_CORETYPE::AIC;
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }
    void VisitStmt_(const AllocateNode *op) final {
      // It is cube kernel if there exists buffer with shared.dyn/wmma.xxx address
      // space
      std::string scope = GetPtrStorageScope(op->buffer_var);
      if (func_coretype != NPU_CORETYPE::MIX) {
        if (scope_coretype_map.count(scope) != 0) {
          func_coretype = scope_coretype_map[scope];
        }
      }
      StmtExprVisitor::VisitStmt_(op);
    }
    NPU_CORETYPE func_coretype{NPU_CORETYPE::AIV};
  };
}  // namespace

/*****************************************************************************************
******************************************************************************************
Functions for CodeGenTileLangNPUIRAPI class
Todo: Remove CodeGenTileLangNPUIR class and use all functions from
CodeGenTileLangNPUIRAPI
******************************************************************************************
******************************************************************************************/

void CodeGenTileLangNPUIRAPI::SmartMemRefCopy(mlir::Value src, mlir::Value dst) {
  auto src_type = src.getType().cast<mlir::MemRefType>();
  auto dst_type = dst.getType().cast<mlir::MemRefType>();
  auto loc = builder.getUnknownLoc();

  // Copy if shape match
  if (src_type.getShape() == dst_type.getShape()) {
    builder.create<mlir::memref::CopyOp>(loc, TypeRange{}, src, dst);
    return;
  }

  // 2. Try Reinterpret Copy if shape not match but numElements match
  if (src_type.getNumElements() == dst_type.getNumElements()) {
      
    // safety check: stride hack only when elementTypes match
    if (src_type.getElementType() != dst_type.getElementType()) {
      ICHECK(false) << "HandleMemRefReshapeCopy requires same element type.";
      return;
    }

    // Get offset of src MemRef
    auto extractOp = builder.create<mlir::memref::ExtractStridedMetadataOp>(loc, src);
    mlir::Value offsetValue = extractOp.getOffset();

    llvm::SmallVector<mlir::OpFoldResult> offsets;
    offsets.push_back(offsetValue);

    // 1. get sizes for dst value.
    llvm::SmallVector<mlir::OpFoldResult> sizes;
    llvm::SmallVector<mlir::Value> dim_values;

    for (int i = 0; i < dst_type.getRank(); ++i) {
      int64_t static_dim = dst_type.getDimSize(i);
      
      if (mlir::ShapedType::isDynamic(static_dim)) {
        mlir::Value dim_val = builder.create<mlir::memref::DimOp>(loc, dst, i);
        sizes.push_back(dim_val);
        dim_values.push_back(dim_val);
      } else {
        sizes.push_back(builder.getIndexAttr(static_dim));
        dim_values.push_back(builder.create<mlir::arith::ConstantIndexOp>(loc, static_dim));
      }
    }

    // 2. Compute strides for StridedLayoutAttr
    // Rule: Calculate from the last dimension to the first. When encountering a dynamic dimension,
    // the current and all preceding strides become dynamic.
    llvm::SmallVector<int64_t> layout_strides(dst_type.getRank(), mlir::ShapedType::kDynamic);
    int64_t current_stride = 1;
    bool all_static_so_far = true;

    // Calculate from the lowest dimension (last dimension) to the highest
    for (int i = dst_type.getRank() - 1; i >= 0; --i) {
      int64_t dim_size = dst_type.getDimSize(i);
      
      // Set stride for the current dimension
      if (i == dst_type.getRank() - 1) {
        // The lowest dimension's stride is always 1
        layout_strides[i] = 1;
      } else {
        // Normal case
        layout_strides[i] = current_stride;
      }
      
      // Update current_stride for the next dimension (higher dimension)
      if (mlir::ShapedType::isDynamic(dim_size)) {
        all_static_so_far = false;
        break;
      } else {
        current_stride *= dim_size;
      }
    }

    // 3. Prepare dynamic strides parameters for reinterpret_cast
    llvm::SmallVector<mlir::OpFoldResult> strides;

    if (all_static_so_far) {
      // All static: use static stride values
      for (int64_t stride : layout_strides) {
        strides.push_back(builder.getIndexAttr(stride));
      }
    } else {
      // Has dynamic dimensions: need to compute dynamic strides
      // Create a vector to store computed strides (calculated from back to front)
      llvm::SmallVector<mlir::Value> temp_strides(dst_type.getRank());
      
      // Calculate from back to front
      mlir::Value current_dyn_stride = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
      for (int i = dst_type.getRank() - 1; i >= 0; --i) {
        // Store stride for current dimension
        temp_strides[i] = current_dyn_stride;
        
        if (i > 0) {
          // Update current_dyn_stride for the next dimension (higher dimension)
          // current_dimension_stride * current_dimension_size
          current_dyn_stride = builder.create<mlir::arith::MulIOp>(
              loc, current_dyn_stride, dim_values[i]);
        }
      }
      
      // Convert temp_strides to OpFoldResult and add to strides in order
      for (int i = 0; i < dst_type.getRank(); ++i) {
        strides.push_back(temp_strides[i]);
      }
    }

    // 4. Create StridedLayoutAttr
    auto layout = mlir::StridedLayoutAttr::get(
        &context, 
        mlir::ShapedType::kDynamic,
        layout_strides);

    // 5. Create target type
    mlir::MemRefType new_dst_type = mlir::MemRefType::get(
        dst_type.getShape(),
        dst_type.getElementType(),
        layout,
        src_type.getMemorySpace());

    // 6. Create reinterpret_cast
    mlir::Value reinterpreted_src = builder.create<mlir::memref::ReinterpretCastOp>(
        loc,
        new_dst_type,
        src,
        offsets,
        sizes,
        strides
    );
    
    // 7. Copy
    builder.create<mlir::memref::CopyOp>(loc, TypeRange{}, reinterpreted_src, dst);

    return;
  }
  ICHECK(false) << "SmartMemRefCopy: Shape mismatch and cannot interpret cast. ";
}

mlir::Value CodeGenTileLangNPUIRAPI::ScalarConvertType(const PrimExpr &imm,
                                                       DataType targetDtype) {
  auto castNode = std::make_unique<tir::Cast>(targetDtype, imm);
  return MakeValue(*castNode);
}

CodeGenTileLangNPUIRAPI::CodeGenTileLangNPUIRAPI() : builder(&context) {
  // Load MLIR dialects in the context
  this->context
      .loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                   mlir::linalg::LinalgDialect, mlir::scf::SCFDialect,
                   mlir::memref::MemRefDialect, mlir::hivm::HIVMDialect,
                   mlir::hfusion::HFusionDialect>();
  // Create MLIR module
  this->module = ModuleOp::create(UnknownLoc::get(&this->context));
}

std::string CodeGenTileLangNPUIRAPI::Finish() {
  std::string mlirCode;
  llvm::raw_string_ostream os(mlirCode);
  module->print(os);
  return mlirCode;
}

inline mlir::hivm::AddressSpace
CodeGenTileLangNPUIRAPI::GetHIVMAddressSpace(String address_space) {
  if (address_space == "global")
    return mlir::hivm::AddressSpace::GM;
  else if (address_space == "shared")
    return mlir::hivm::AddressSpace::UB;
  else if (address_space == "shared.dyn")
    return mlir::hivm::AddressSpace::L1;
  else if (address_space == "wmma.accumulator")
    return mlir::hivm::AddressSpace::L0C;
  return mlir::hivm::AddressSpace::Zero;
}

inline std::vector<long int>
CodeGenTileLangNPUIRAPI::GetShape(Array<PrimExpr> shape_in) {
  std::vector<long int> shape;
  for (PrimExpr s : shape_in) {
    if (auto s_int = as_const_int(s)) {
      // Statically known dimension
      shape.push_back(*s_int);
    } else {
      // Dynamic dimension "?x";
      shape.push_back(-1);
    }
  }
  return shape;
}

mlir::Type CodeGenTileLangNPUIRAPI::GetMLIRType(const PrimExpr &expr) {
  auto ttype = GetType(expr);
  auto DType = GetRuntimeDataType(ttype);
  return DTypetoMLIRType(DType);
}

mlir::Type CodeGenTileLangNPUIRAPI::GetMLIRType(const Buffer &buffer) {
  llvm::SmallVector<int64_t> shape, stride;
  int64_t base = 1;
  bool isDynamicShape = false;
  for (auto s : buffer->shape) {
    auto intImm = s.as<tvm::tir::IntImmNode>();
    if (intImm != nullptr) {
      shape.emplace_back(intImm->value);
      base *= intImm->value;
    } else {
      shape.emplace_back(ShapedType::kDynamic);
      isDynamicShape = true;
    }
  }
  if (buffer->strides.size()) {
    for (auto s : buffer->strides) {
      auto intImm = s.as<tvm::tir::IntImmNode>();
      if (intImm != nullptr) {
        stride.emplace_back(intImm->value);
      } else {
        stride.emplace_back(ShapedType::kDynamic);
      }
    }
  } else {
    for (auto s : buffer->shape) {
      auto intImm = s.as<tvm::tir::IntImmNode>();
      if (!isDynamicShape) {
        base /= intImm->value;
        stride.emplace_back(base);
      } else {
        stride.emplace_back(ShapedType::kDynamic);
      }
    }
  }
  auto elementType = DTypetoMLIRType(buffer->dtype);
  auto offset = 0;
  String scope = GetPtrStorageScope(buffer->data);
  auto addressSpace = GetHIVMAddressSpace(scope);
  auto addressSpaceAttr =
      mlir::hivm::AddressSpaceAttr::get(builder.getContext(), addressSpace);
  auto strideLayout =
      StridedLayoutAttr::get(builder.getContext(), offset, stride);
  return MemRefType::get(shape, elementType, strideLayout, addressSpaceAttr);
}

void CodeGenTileLangNPUIRAPI::VisitStmt_(const tir::ForNode *op) {

  CHECK(op->extent.dtype().is_int() || op->extent.dtype().is_uint());
  CHECK(op->min.dtype() == op->extent.dtype());

  auto lowerBoundId = MakeValue(op->min);
  auto upperBoundId = MakeValue(op->extent + op->min);

  // Create the loop
  auto step = builder.create<mlir::arith::ConstantOp>(
      mlir::UnknownLoc::get(&context),
      builder.getIntegerAttr(GetMLIRType(op->min), 1));
  auto forOp = builder.create<mlir::scf::ForOp>(module->getLoc(), lowerBoundId,
                                                 upperBoundId, step);
  // Set the insertion point to the body of the loop
  OpBuilder::InsertionGuard saved(builder);
  builder.setInsertionPointToStart(forOp.getBody());

  auto loop_var = op->loop_var;
  ICHECK(!var_map_.count(loop_var.get()));
  var_map_[loop_var.get()] = forOp.getInductionVar();

  this->VisitStmt(op->body);
}

void CodeGenTileLangNPUIRAPI::VisitStmt_(const tir::IfThenElseNode *op) {

  auto conditionValue = MakeValue(op->condition);

  bool elseRegionFlag = false;
  if (op->else_case) {
    elseRegionFlag = true;
  }

  mlir::Location unknown_loc = builder.getUnknownLoc();
  // Create the SCF If operation
  mlir::scf::IfOp ifOp = builder.create<mlir::scf::IfOp>(
      unknown_loc, mlir::TypeRange{}, conditionValue, true, elseRegionFlag);
  // Set the insertion point to the true region
  mlir::Block *thenBlock = &ifOp.getThenRegion().getBlocks().front();
  builder.setInsertionPointToEnd(thenBlock);
  this->VisitStmt(op->then_case);
  builder.create<mlir::scf::YieldOp>(unknown_loc);

  if (op->else_case) {
    // Set the insertion point to the false region
    mlir::Block *elseBlock = &ifOp.getElseRegion().getBlocks().front();
    builder.setInsertionPointToEnd(elseBlock);
    this->VisitStmt(op->else_case.value());
    builder.create<mlir::scf::YieldOp>(unknown_loc);
  }
  builder.setInsertionPointAfter(ifOp);
}

mlir::Type CodeGenTileLangNPUIRAPI::DTypetoMLIRType(DataType t) { // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    // ICHECK(t.is_scalar()) << "do not yet support vector types";
    return mlir::NoneType();
  }
  if (t.is_void()) {
    return builder.getNoneType();
  }
  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
    case 16:
      if (t.is_scalar()) {
        return builder.getF16Type();
      } else {
        fail = true;
      }
      break;
    case 32:
      return builder.getF32Type();
      break;
    case 64:
      return builder.getF64Type();
      break;
    default:
      fail = true;
      break;
    }
    if (!fail && (t.is_scalar() || t.bits() == 16))
      return mlir::NoneType();
  } else if (t.is_bfloat16()) {
    if (t.is_scalar()) {
      return builder.getBF16Type();
    } else {
      fail = true;
    }
    if (!fail)
      return mlir::NoneType();
  } else if (t == DataType::Bool()) {
    return builder.getI1Type();
  } else if (t.is_int() || t.is_uint()) {
    switch (t.bits()) {
    case 1: {
      if (t.is_scalar()) {
        return builder.getI1Type();
      } else {
        LOG(FATAL) << "Cannot convert type " << t;
      }
    }
    case 4: {
      if (t.is_scalar()) {
        return builder.getI4Type();
      } else {
        LOG(FATAL) << "Cannot convert type " << t;
      }
    }
    case 8: {
      if (t.is_scalar()) {
        return builder.getI8Type();
      } else {
        LOG(FATAL) << "Cannot convert type " << t;
      }
    }
    case 16: {
      if (t.is_scalar()) {
        builder.getI16Type();
      } else {
        fail = true;
      }
      if (!fail) {
        return builder.getI16Type();
      }
      break;
    }
    case 32: {
      if (t.is_scalar()) {
        builder.getI32Type();
      } else {
        fail = true;
      }
      if (!fail) {
        return builder.getI32Type();
      }
      break;
    }
    case 64: {
      if (t.is_scalar()) {
        return builder.getI64Type();
      }
      return builder.getI64Type();
    }
    default:
      fail = true;
      break;
    }
    if (!fail) {
      return mlir::NoneType();
    }
  }
  LOG(FATAL) << "Cannot convert type " << t;
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const FloorDivNode *op) {
  auto lhs = MakeValue(op->a);
  auto rhs = MakeValue(op->b);
  // FIXME: The floor div in python is not the same as arith.divsi in negative
  // scenarios.
  mlir::Value mlirVal;
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    mlirVal = BinaryOpCodegen<mlir::arith::DivSIOp, std::nullptr_t>(op, nullptr,
                                                                    lhs, rhs);
  } else if (op->dtype.is_float()) {
    mlirVal = BinaryOpCodegen<mlir::arith::DivFOp, std::nullptr_t>(op, nullptr,
                                                                   lhs, rhs);
  }
  return mlirVal;
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const FloorModNode *op) {
  auto lhs = MakeValue(op->a);
  auto rhs = MakeValue(op->b);
  mlir::Value mlirVal;
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    mlirVal = BinaryOpCodegen<mlir::arith::RemSIOp, std::nullptr_t>(op, nullptr,
                                                                    lhs, rhs);
  } else if (op->dtype.is_float()) {
    mlirVal = BinaryOpCodegen<mlir::arith::RemFOp, std::nullptr_t>(op, nullptr,
                                                                   lhs, rhs);
  }
  return mlirVal;
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const LTNode *op) {
  auto lhs = MakeValue(op->a);
  auto rhs = MakeValue(op->b);
  mlir::Value mlirVal;
  if (op->a->dtype.is_int()) {
    mlirVal = BinaryOpCodegen<mlir::arith::CmpIOp, mlir::arith::CmpIPredicate>(
        op, mlir::arith::CmpIPredicate::slt, lhs, rhs);
  } else if (op->a->dtype.is_uint()) {
    mlirVal = BinaryOpCodegen<mlir::arith::CmpIOp, mlir::arith::CmpIPredicate>(
        op, mlir::arith::CmpIPredicate::ult, lhs, rhs);
  } else {
    mlirVal = BinaryOpCodegen<mlir::arith::CmpFOp, mlir::arith::CmpFPredicate>(
        op, mlir::arith::CmpFPredicate::OLT, lhs, rhs);
  }
  return mlirVal;
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const NENode *op) {
  auto lhs = MakeValue(op->a);
  auto rhs = MakeValue(op->b);
  mlir::Value mlirVal;
  if (op->a->dtype.is_int() || op->a->dtype.is_uint()) {
    mlirVal = BinaryOpCodegen<mlir::arith::CmpIOp, mlir::arith::CmpIPredicate>(
        op, mlir::arith::CmpIPredicate::ne, lhs, rhs);
  } else {
    mlirVal = BinaryOpCodegen<mlir::arith::CmpFOp, mlir::arith::CmpFPredicate>(
        op, mlir::arith::CmpFPredicate::ONE, lhs, rhs);
  }
  return mlirVal;
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const EQNode *op) {
  auto lhs = MakeValue(op->a);
  auto rhs = MakeValue(op->b);
  mlir::Value mlirVal;
  if (op->a->dtype.is_int() || op->a->dtype.is_uint()) {
    mlirVal = BinaryOpCodegen<mlir::arith::CmpIOp, mlir::arith::CmpIPredicate>(
        op, mlir::arith::CmpIPredicate::eq, lhs, rhs);
  } else {
    mlirVal = BinaryOpCodegen<mlir::arith::CmpFOp, mlir::arith::CmpFPredicate>(
        op, mlir::arith::CmpFPredicate::OEQ, lhs, rhs);
  }
  return mlirVal;
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const LENode *op) {
  auto lhs = MakeValue(op->a);
  auto rhs = MakeValue(op->b);
  mlir::Value mlirVal;
  if (op->a->dtype.is_int()) {
    mlirVal = BinaryOpCodegen<mlir::arith::CmpIOp, mlir::arith::CmpIPredicate>(
        op, mlir::arith::CmpIPredicate::sle, lhs, rhs);
  } else if (op->a->dtype.is_uint()) {
    mlirVal = BinaryOpCodegen<mlir::arith::CmpIOp, mlir::arith::CmpIPredicate>(
        op, mlir::arith::CmpIPredicate::ule, lhs, rhs);
  } else {
    mlirVal = BinaryOpCodegen<mlir::arith::CmpFOp, mlir::arith::CmpFPredicate>(
        op, mlir::arith::CmpFPredicate::OLE, lhs, rhs);
  }
  return mlirVal;
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const GENode *op) {
  auto lhs = MakeValue(op->a);
  auto rhs = MakeValue(op->b);
  mlir::Value mlirVal;
  if (op->a->dtype.is_int()) {
    mlirVal = BinaryOpCodegen<mlir::arith::CmpIOp, mlir::arith::CmpIPredicate>(
        op, mlir::arith::CmpIPredicate::sge, lhs, rhs);
  } else if (op->a->dtype.is_uint()) {
    mlirVal = BinaryOpCodegen<mlir::arith::CmpIOp, mlir::arith::CmpIPredicate>(
        op, mlir::arith::CmpIPredicate::uge, lhs, rhs);
  } else {
    mlirVal = BinaryOpCodegen<mlir::arith::CmpFOp, mlir::arith::CmpFPredicate>(
        op, mlir::arith::CmpFPredicate::OGE, lhs, rhs);
  }
  return mlirVal;
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const GTNode *op) {
  auto lhs = MakeValue(op->a);
  auto rhs = MakeValue(op->b);
  mlir::Value mlirVal;
  if (op->a->dtype.is_int()) {
    mlirVal = BinaryOpCodegen<mlir::arith::CmpIOp, mlir::arith::CmpIPredicate>(
        op, mlir::arith::CmpIPredicate::sgt, lhs, rhs);
  } else if (op->a->dtype.is_uint()) {
    mlirVal = BinaryOpCodegen<mlir::arith::CmpIOp, mlir::arith::CmpIPredicate>(
        op, mlir::arith::CmpIPredicate::ugt, lhs, rhs);
  } else {
    mlirVal = BinaryOpCodegen<mlir::arith::CmpFOp, mlir::arith::CmpFPredicate>(
        op, mlir::arith::CmpFPredicate::OGT, lhs, rhs);
  }
  return mlirVal;
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const CastNode *op) {
  bool srcIsFloat =
      op->value->dtype.is_float() || op->value->dtype.is_bfloat16();
  bool srcIsInt = op->value->dtype.is_int();
  bool srcIsUInt = op->value->dtype.is_uint();
  bool targetIsFloat = op->dtype.is_float() || op->dtype.is_bfloat16();
  bool targetIsInt = op->dtype.is_int();
  bool targetIsUInt = op->dtype.is_uint();
  auto targetType = DTypetoMLIRType(op->dtype);

  auto val = VisitExpr(op->value);
  if (srcIsFloat && targetIsInt) {
    return builder.create<mlir::arith::FPToSIOp>(
        mlir::UnknownLoc::get(&context), targetType, val);
  } else if (srcIsFloat && targetIsUInt) {
    return builder.create<mlir::arith::FPToUIOp>(
        mlir::UnknownLoc::get(&context), targetType, val);
  } else if (srcIsInt && targetIsFloat) {
    return builder.create<mlir::arith::SIToFPOp>(
        mlir::UnknownLoc::get(&context), targetType, val);
  } else if (srcIsUInt && targetIsFloat) {
    return builder.create<mlir::arith::UIToFPOp>(
        mlir::UnknownLoc::get(&context), targetType, val);
  } else if (targetIsInt) {
    if (op->dtype.bits() > op->value->dtype.bits()) {
      return builder.create<mlir::arith::ExtSIOp>(
          mlir::UnknownLoc::get(&context), targetType, val);
    } else {
      return builder.create<mlir::arith::TruncIOp>(
          mlir::UnknownLoc::get(&context), targetType, val);
    }
  } else if (targetIsUInt) {
    if (op->dtype.bits() > op->value->dtype.bits()) {
      return builder.create<mlir::arith::ExtUIOp>(
          mlir::UnknownLoc::get(&context), targetType, val);
    } else {
      return builder.create<mlir::arith::TruncIOp>(
          mlir::UnknownLoc::get(&context), targetType, val);
    }
  } else if (targetIsFloat) {
    if (op->dtype.bits() > op->value->dtype.bits()) {
      return builder.create<mlir::arith::ExtFOp>(
          mlir::UnknownLoc::get(&context), targetType, val);
    } else {
      return builder.create<mlir::arith::TruncFOp>(
          mlir::UnknownLoc::get(&context), targetType, val);
    }
  } else {
    LOG(FATAL) << "type cast failed: " << op->value->dtype << " to "
               << op->dtype;
  }
}

mlir::Value
CodeGenTileLangNPUIRAPI::GenSubviewFromRegion(const CallNode *region_node) {
  tvm::tl::RegionOp regionop(region_node->args, this->vmap);
  return GenSubviewFromRegion(regionop.GetBuffer(), regionop.GetRanges());
}

mlir::Value CodeGenTileLangNPUIRAPI::GenSubviewFromRegion(Buffer buffer_data,
                                                          Array<Range> range) {
  /*
  range stores region details
    extent stores the shape or size of region
    min stores the offset of the region
  */
  Array<PrimExpr> region_shape, region_indeces;
  for (Range r : range) {
    region_shape.push_back(r.get()->extent);
    region_indeces.push_back(r.get()->min);
  }
  const VarNode *v = buffer_data->data.get();
  mlir::Value v_value = GetVarValue(v);
  if ((IsEqual(buffer_data->shape, region_shape) && AllZero(region_indeces))) {
    return v_value; // return original buffer and no need to create subview
  }
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> shape_val;
  SmallVector<OpFoldResult> strides_val;
  for (Range r : range) {
    // if size or offset is var, create IndexCastOp and push the mlir value into
    // the parameter of SubViewOp.
    if (auto s_int = as_const_int(r.get()->min)) {
      offsets.push_back(builder.getI64IntegerAttr(*s_int));
    } else {
      mlir::Value indexVal = CreateIndexCastOp(MakeValue(r.get()->min));
      offsets.push_back(indexVal);
    }
    if (auto s_int = as_const_int(r.get()->extent)) {
      shape_val.push_back(builder.getI64IntegerAttr(*s_int));
    } else {
      mlir::Value s_index = CreateIndexCastOp(MakeValue(r.get()->extent));
      shape_val.push_back(s_index);
    }
    strides_val.push_back(builder.getI64IntegerAttr(1));
  }

  auto subViewOp =
      builder.create<mlir::memref::SubViewOp>(builder.getUnknownLoc(),
                                               v_value,    // Original memref
                                               offsets,    // Offset
                                               shape_val,  // Sizes or shape
                                               strides_val // Strides
      );
  return subViewOp;
}

mlir::Value CodeGenTileLangNPUIRAPI::CreateIndexCastOp(mlir::Value src) {
  std::pair<bool, mlir::Value> result = CheckMLIRValueMap(src);
  if (result.first) {
    return result.second;
  }
  mlir::Value indexVal = builder.create<mlir::arith::IndexCastOp>(
    builder.getUnknownLoc(), builder.getIndexType(), src);
  UpdateMLIRValueMap(src, indexVal);
  return indexVal;
}

inline std::pair<bool, mlir::Value> CodeGenTileLangNPUIRAPI::CheckMLIRValueMap(mlir::Value val){
  mlir::Block *curr_block = builder.getInsertionBlock();
  auto it = this->mlir_value_map.find({val, curr_block});
  if (it != this->mlir_value_map.end()) {
    return std::pair(true, it->second);
  }
  return std::pair(false, mlir::Value());
}

inline void CodeGenTileLangNPUIRAPI::UpdateMLIRValueMap(const mlir::Value key, mlir::Value val){
  mlir::Block *curr_block = builder.getInsertionBlock();
  this->mlir_value_map[{key, curr_block}] = val;
}

inline std::pair<bool, mlir::Value> CodeGenTileLangNPUIRAPI::CheckPrimExprMap(const PrimExprNode * op){
  mlir::Block *curr_block = builder.getInsertionBlock();
  auto it = this->prim_expr_map.find({GetRef<PrimExpr>(op), curr_block});
  if (it != this->prim_expr_map.end()) {
    return std::pair(true, it->second);
  }
  return std::pair(false, mlir::Value());
}

inline void CodeGenTileLangNPUIRAPI::UpdatePrimExprMap(const PrimExprNode * key, mlir::Value val){
  mlir::Block *curr_block = builder.getInsertionBlock();
  this->prim_expr_map[{GetRef<PrimExpr>(key), curr_block}] = val;
}

/*
  T contains the type of binary operation
  U contains the type of comparison mode
  op contains PrimExprNode operation node
  mode contains comparison mode
  lhs contains left value
  rhs contains right value
*/
template <typename T, typename U>
mlir::Value CodeGenTileLangNPUIRAPI::BinaryOpCodegen(const PrimExprNode *op,
                                                     U mode, mlir::Value lhs,
                                                     mlir::Value rhs) {
  // check if same node already created
  // If already created return corresponding MLIR value and do not create
  // duplicated MLIR Op
  std::pair<bool, mlir::Value> result = CheckPrimExprMap(op);
  if (result.first) {
    return result.second;
  }
  mlir::Value mlirVal;
  if constexpr (std::is_same_v<U, std::nullptr_t>) {
      // create binary arithmetic operations
      mlirVal = builder.create<T>(builder.getUnknownLoc(), lhs, rhs);
  } else {
      // create binary comparison operations
      assert(mode != nullptr && "Mode must not be nullptr!");
      mlirVal = builder.create<T>(builder.getUnknownLoc(), mode, lhs, rhs);
  }
  UpdatePrimExprMap(op, mlirVal);
  return mlirVal;
}

/// Generate hivm.hir.load or hivm.hir.store for tl.ascend_copy.
/// before:
///   T.ascend_copy(T.region(A[bx, by], 1, 128, 256), T.region(A_VEC[0, 0],
///   2, 128, 256))
/// after:
///   memref.reinterpret_cast; memref.subview; memref.subview;
///   memref.copy
void CodeGenTileLangNPUIRAPI::AscendCopyCodegen(const CallNode *op) {
  tvm::tl::AscendCopy npuirop(op->args, this->vmap);
  mlir::Value src_sub_view =
      GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  mlir::Value dst_sub_view =
      GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  SmartMemRefCopy(src_sub_view, dst_sub_view);
}

template <typename T, typename U>
void CodeGenTileLangNPUIRAPI::UnaryVecOpCodegen(const CallNode *op) {
  T npuirop(op->args, this->vmap);
  auto in_data_name = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  auto out_data_name = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  auto dims = getBroadcastDim(npuirop.src->shape, npuirop.dst->shape);
  // Create HIVM Op
  builder.create<U>(builder.getUnknownLoc(), mlir::TypeRange{}, // result type
                     mlir::ValueRange{in_data_name},              // in
                     mlir::ValueRange{out_data_name},             // out
                     builder.getDenseI64ArrayAttr({}),           // transpose
                     builder.getDenseI64ArrayAttr(dims)          // broadcast
  );
}

void CodeGenTileLangNPUIRAPI::BarrierCodegen(const CallNode *op) {
  tvm::tl::NpuirPipeBarrier npuirop(op->args, this->vmap);
  mlir::hivm::PipeAttr pipAttrType = mlir::hivm::PipeAttr::get(
      builder.getContext(), PIPE_MAP[npuirop.pipe_type]);
  builder.create<mlir::hivm::PipeBarrierOp>(builder.getUnknownLoc(),
                                             pipAttrType);
}

void CodeGenTileLangNPUIRAPI::VselectCodegen(const CallNode *op) {
  /// Generate hivm.hir.vsel for tl.npuir_select.
  /// before:
  ///   T.npuir_select(Cond_VEC, A_VEC, B_VEC, C_VEC)
  /// after:
  ///   hivm.hir.vsel ins(%v__9, %A_VEC, %B_VEC : memref<32x64xi1, strided<[64,
  ///   1], offset:0>, #hivm.address_space<ub>>, memref<32x64xf16, strided<[64,
  ///   1], offset:0>, #hivm.address_space<ub>>, memref<32x64xf16, strided<[64,
  ///   1], offset:0>, #hivm.address_space<ub>>) outs(%C_VEC : memref<32x64xf16,
  ///   strided<[64, 1], offset:0>, #hivm.address_space<ub>>)
  tvm::tl::NpuirSelect npuirop(op->args, this->vmap);
  // gen memref.subview
  auto cond_data_name = GenSubviewFromRegion(npuirop.cond, npuirop.cond_range);
  auto src0_data_name = GenSubviewFromRegion(npuirop.src0, npuirop.src0_range);
  auto src1_data_name = GenSubviewFromRegion(npuirop.src1, npuirop.src1_range);
  auto dst_data_name = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  // gen mlir::hivm::VSelOp
  auto broadcastDim = getBroadcastDim(npuirop.src0->shape, npuirop.dst->shape);
  auto selOp = builder.create<mlir::hivm::VSelOp>(
      builder.getUnknownLoc(), mlir::TypeRange{},
      mlir::ValueRange{cond_data_name, src0_data_name, src1_data_name},
      mlir::ValueRange{dst_data_name}, mlir::Value());
  selOp->setAttr("broadcast", builder.getDenseI64ArrayAttr(broadcastDim));
}

void CodeGenTileLangNPUIRAPI::VbrcCodegen(const CallNode *op) {
  tvm::tl::NpuirBrc npuirop(op->args, this->vmap);
  mlir::Value src;
  llvm::ArrayRef<int64_t> inBufferShape;
  if (npuirop.in.as<IntImm>() || npuirop.in.as<FloatImm>()) {
    // Scalar case
    if (npuirop.in->dtype != npuirop.dst->dtype) {
      src = ScalarConvertType(npuirop.in, npuirop.dst->dtype);
    } else {
      src = MakeValue(npuirop.in);
    }
  } else {
    src = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
    auto srcMemref = llvm::dyn_cast<TypedValue<MemRefType>>(src);
    inBufferShape = srcMemref.getType().getShape();
  }
  Value dst = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  auto broadcastDimAttr = builder.getDenseI64ArrayAttr({});
  if (!inBufferShape.empty()) {
    auto outMemref = llvm::dyn_cast<TypedValue<MemRefType>>(dst);
    auto outBufferShape = outMemref.getType().getShape();
    auto broadcastDim = getBroadcastDim(npuirop.src->shape, npuirop.dst->shape);
    broadcastDimAttr = builder.getDenseI64ArrayAttr(broadcastDim);
  }
  builder.create<mlir::hivm::VBrcOp>(builder.getUnknownLoc(), TypeRange{},
                                      src, dst, broadcastDimAttr);
}

void CodeGenTileLangNPUIRAPI::VcastCodegen(const CallNode *op) {
  tvm::tl::NpuirCast npuirop(op->args, this->vmap);
  Value src = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  Value dst = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  auto round_mode = npuirop.round_mode;
  mlir::hivm::RoundMode mode = NPUIR_STR_ROUNDMODE[round_mode];
  auto inBufferShape =
      llvm::dyn_cast<TypedValue<MemRefType>>(src).getType().getShape();
  auto outBufferShape =
      llvm::dyn_cast<TypedValue<MemRefType>>(dst).getType().getShape();
  auto broadcastDim = getBroadcastDim(inBufferShape, outBufferShape);
  auto broadcastDimAttr = builder.getDenseI64ArrayAttr(broadcastDim);
  builder.create<mlir::hivm::VCastOp>(
      builder.getUnknownLoc(), TypeRange{}, src, dst,
      mlir::hivm::RoundModeAttr::get(&context, mode), nullptr,
      broadcastDimAttr);
}

void CodeGenTileLangNPUIRAPI::VreduceCodegen(const CallNode *op) {
  /// Generate hivm.hir.vreduce for T.npuir_reduce.
  /// before:
  ///   T.npuir_reduce(src, dst, dims, type)
  /// after:
  ///   hivm.hir.vreduce <type> ins(src) outs(dst) reduce_dims = [dims]
  tvm::tl::NpuirReduce npuirop(op->args, this->vmap);
  mlir::Location loc = builder.getUnknownLoc();
  Value src = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  Value dst = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  auto reduce_mode = npuirop.reduce_mode;
  mlir::hivm::ReduceOperation operation = NPUIR_STR_REDUCEOP[reduce_mode];
  mlir::hivm::ReduceOpAttr mode =
      mlir::hivm::ReduceOpAttr::get(&context, operation);
  builder.create<mlir::hivm::VReduceOp>(
      loc, TypeRange{}, src, dst, mode,
      builder.getDenseI64ArrayAttr(npuirop.reduce_dims));
}

void CodeGenTileLangNPUIRAPI::VcumsumCodegen(const CallNode *op) {
  /// Generate hivm.hir.cumsum for tl.npuir_cumsum.
  /// before:
  ///   T.npuir_cumsum(src, dst, dim, reverse)
  /// after:
  ///   hivm.hir.vcumsum ins(src) outs(dst) cum_dims = [0] for reverse = false
  tvm::tl::NpuirCumsum npuirop(op->args, this->vmap);
  mlir::Location loc = builder.getUnknownLoc();
  Value src = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  Value dst = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  auto reverse_mode = npuirop.reverse;
  if(reverse_mode == true){
    ICHECK(false) <<"reverse=True is not yet supported\n";
    return;
  }
  builder.create<mlir::hivm::VCumsumOp>(
      loc, TypeRange{}, src, dst,
      builder.getDenseI64ArrayAttr(npuirop.cum_dims));
}

void CodeGenTileLangNPUIRAPI::VAtomicAddCodegen(const CallNode *op) {
  /// Generate hivm.hir.store for tl.npuir_atomic_add.
  /// before:
  ///   T.npuir_atomic_add(src, dst, size)
  /// after:
  ///   hivm.hir.store ins(src) outs(dst) atomic = <add>
  tvm::tl::NpuirAtomicAdd npuirop(op->args, this->vmap);
  Value src = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  Value dst = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);

  // create StoreOp
  auto newStoreOp = builder.create<hivm::StoreOp>(
      builder.getUnknownLoc(),
      TypeRange{},
      src,
      dst
  );
  hivm::AtomicKind hvAtomicKind = hivm::AtomicKind::ADD;
  newStoreOp.setAtomicKind(hvAtomicKind);
}

void CodeGenTileLangNPUIRAPI::VgatherCodegen(const CallNode *op) {
  tvm::tl::NpuirGather npuirop(op->args, this->vmap);
  Value src = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  Value dst = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  Value indices = GenSubviewFromRegion(npuirop.indices, npuirop.indices_range);

  builder.create<mlir::hivm::VGatherOp>(builder.getUnknownLoc(), TypeRange{},
                                        src, indices, dst);
}

void CodeGenTileLangNPUIRAPI::VtransposeCodegen(const CallNode *op) {
  tvm::tl::NpuirTranspose npuirop(op->args, this->vmap);
  Value src = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  Value dst = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  auto permutation = builder.getDenseI64ArrayAttr(npuirop.permutation);
  builder.create<mlir::hivm::VTransposeOp>(builder.getUnknownLoc(), TypeRange{},
                                           src, dst, permutation);
}

void CodeGenTileLangNPUIRAPI::VinterleaveCodegen(const CallNode *op) {
  tvm::tl::NpuirInterleave npuirop(op->args, this->vmap);
  llvm::SmallVector<Value> srcs;
  size_t n_srcs = npuirop.srcs.size();
  for (size_t i = 0; i < n_srcs; i++) {
    Value src = GenSubviewFromRegion(npuirop.srcs[i], npuirop.srcs_range[i]);
    srcs.push_back(src);
  }
  mlir::ValueRange srcs_vr(srcs);
  Value dst = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  builder.create<mlir::hivm::VInterleaveOp>(
      builder.getUnknownLoc(), TypeRange{}, srcs_vr, dst,
      static_cast<int64_t>(npuirop.channel_nums));
}

void CodeGenTileLangNPUIRAPI::VdeinterleaveCodegen(const CallNode *op) {
  tvm::tl::NpuirDeinterleave npuirop(op->args, this->vmap);
  Value src = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  llvm::SmallVector<Value> dsts;
  size_t n_dsts = npuirop.dsts.size();
  for (size_t i = 0; i < n_dsts; i++) {
    Value dst = GenSubviewFromRegion(npuirop.dsts[i], npuirop.dsts_range[i]);
    dsts.push_back(dst);
  }
  mlir::ValueRange dsts_vr(dsts);
  auto channel_nums = mlir::IntegerAttr::get(
      builder.getI64Type(), static_cast<int64_t>(npuirop.channel_nums));
  mlir::hivm::DeinterleaveModeAttr index_mode =
      mlir::hivm::DeinterleaveModeAttr::get(
          &context, NPUIR_STR_DEINTERLEAVEMODE[npuirop.index_mode]);
  builder.create<mlir::hivm::VDeinterleaveOp>(builder.getUnknownLoc(),
                                              TypeRange{}, src, dsts_vr,
                                              channel_nums, index_mode);
}

void CodeGenTileLangNPUIRAPI::VarangeCodegen(const CallNode *op) {
  tvm::tl::NpuirArange npuirop(op->args, this->vmap);
  Value dst = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);

  auto offsetValue = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getI64Type(),
      builder.getI64IntegerAttr(npuirop.offset));
  mlir::Value offset = CreateIndexCastOp(offsetValue);
  llvm::SmallVector<Value> strides;
  for (auto st : npuirop.strides) {
    auto stValue = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getI64Type(),
        builder.getI64IntegerAttr(st));
    mlir::Value stride = CreateIndexCastOp(stValue);
    strides.push_back(stride);
  }

  builder.create<mlir::hivm::VArangeOp>(builder.getUnknownLoc(), TypeRange{},
                                        dst, offset, strides);
}

void CodeGenTileLangNPUIRAPI::VconcatCodegen(const CallNode *op) {
  tvm::tl::NpuirConcat npuirop(op->args, this->vmap);
  auto dim = builder.getIntegerAttr(builder.getI64Type(), npuirop.dim);
  llvm::SmallVector<Value> srcs;
  size_t n_srcs = npuirop.srcs.size();
  for (size_t i = 0; i < n_srcs; i++) {
    Value src = GenSubviewFromRegion(npuirop.srcs[i], npuirop.srcs_range[i]);
    srcs.push_back(src);
  }
  mlir::ValueRange srcs_vr(srcs);
  Value dst = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  builder.create<mlir::hivm::VConcatOp>(builder.getUnknownLoc(), TypeRange{},
                                        dim, srcs_vr, dst);
}

void CodeGenTileLangNPUIRAPI::VpadCodegen(const CallNode *op) {
  tvm::tl::NpuirPad npuirop(op->args, this->vmap);
  Value src = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  Value dst = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  Value pad_value = MakeValue(npuirop.pad_value);
  llvm::SmallVector<Value> low;
  llvm::SmallVector<Value> high;
  for (auto l : npuirop.low) {
    mlir::Value mlir_low = CreateIndexCastOp(MakeValue(l));
    low.push_back(mlir_low);
  }
  for (auto h : npuirop.high) {
    mlir::Value mlir_high = CreateIndexCastOp(MakeValue(h));
    high.push_back(mlir_high);
  }
  if (!low.empty()) {
    npuirop.s_low[npuirop.pad_dim] = ShapedType::kDynamic;
  }
  if (!high.empty()) {
    npuirop.s_high[npuirop.pad_dim] = ShapedType::kDynamic;
  }
  builder.create<mlir::hivm::VPadOp>(
      builder.getUnknownLoc(), TypeRange{}, src, dst, pad_value, low, high,
      builder.getDenseI64ArrayAttr(npuirop.s_low),
      builder.getDenseI64ArrayAttr(npuirop.s_high));
}

void CodeGenTileLangNPUIRAPI::VflipCodegen(const CallNode *op) {
  tvm::tl::NpuirFlip npuirop(op->args, this->vmap);
  Value src = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  Value dst = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  builder.create<mlir::hivm::VFlipOp>(builder.getUnknownLoc(), TypeRange{}, src,
                                      dst);
}

void CodeGenTileLangNPUIRAPI::Nd2NzCodegen(const CallNode *op) {
  // Generate hivm.hir.nd2nz for tl.npuir_load_nd2nz.
  tvm::tl::NpuirNd2nz npuirop(op->args, this->vmap);
  // gen memref.subview
  mlir::Value src = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  mlir::Value dst = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);

  // gen hivm.hir.nd2nz
  mlir::Location unknown_loc = builder.getUnknownLoc();
  mlir::TypeRange res = {};
  mlir::UnitAttr dst_continuous =
      npuirop.dst_continuous ? builder.getUnitAttr() : mlir::UnitAttr();
  builder.create<mlir::hivm::ND2NZOp>(unknown_loc, res, src, dst,
                                       dst_continuous);
}

void CodeGenTileLangNPUIRAPI::Nz2NdCodegen(const CallNode *op) {
  // Generate hivm.hir.nz2nd for tl.npuir_store_nz2nd.
  tvm::tl::NpuirNz2nd npuirop(op->args, this->vmap);
  // gen memref.subview
  mlir::Value src = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  mlir::Value dst = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);

  // gen hivm.hir.nz2nd
  builder.create<mlir::hivm::NZ2NDOp>(builder.getUnknownLoc(),
                                      mlir::TypeRange{}, src, dst);
}

void CodeGenTileLangNPUIRAPI::FixpipeCodegen(const CallNode *op) {
  // Generate hivm.hir.fixpipe for tl.npuir_store_fixpipe.
  tvm::tl::NpuirFixpipe npuirop(op->args, this->vmap);
  // gen memref.subview
  mlir::Value src = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  mlir::Value dst = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);

  // gen hivm.hir.fixpipe
  mlir::Location unknown_loc = builder.getUnknownLoc();
  mlir::TypeRange result = {};
  mlir::UnitAttr enable_nz2nd =
      npuirop.enable_nz2nd ? builder.getUnitAttr() : mlir::UnitAttr();
  mlir::hivm::FixpipePreReluMode pre_relu_mode =
      fixpipe_pre_relu_mode[npuirop.pre_relu_mode];
  auto src_dtype = npuirop.src->dtype;
  auto dst_dtype = npuirop.dst->dtype;
  mlir::hivm::FixpipePreQuantMode pre_quant_mode =
      mlir::hivm::FixpipePreQuantMode::NO_QUANT;
  if (src_dtype != dst_dtype) {
    if (src_dtype == DataType::Float(32) && dst_dtype == DataType::Float(16)) {
      pre_quant_mode = mlir::hivm::FixpipePreQuantMode::F322F16;
    } else if (src_dtype == DataType::Float(32) &&
               dst_dtype == DataType::BFloat(16)) {
      pre_quant_mode = mlir::hivm::FixpipePreQuantMode::F322BF16;
    } else if (src_dtype == DataType::Int(32) &&
               dst_dtype == DataType::Int(8)) {
      pre_quant_mode = mlir::hivm::FixpipePreQuantMode::S322I8;
    } else {
      LOG(FATAL) << "Unexpected pre-quant mode. Should not reach here.\n";
    }
  }
  mlir::hivm::FixpipePreQuantModeAttr pre_quant =
      mlir::hivm::FixpipePreQuantModeAttr::get(builder.getContext(),
                                               pre_quant_mode);
  mlir::hivm::FixpipePreReluModeAttr pre_relu =
      mlir::hivm::FixpipePreReluModeAttr::get(builder.getContext(),
                                              pre_relu_mode);
  mlir::BoolAttr channel_split = builder.getBoolAttr(npuirop.channel_split);
  builder.create<mlir::hivm::FixpipeOp>(unknown_loc, result, src, dst,
                                         enable_nz2nd, pre_quant, pre_relu,
                                         channel_split);
}

void CodeGenTileLangNPUIRAPI::DotCodegen(const CallNode *op) {
  // Generate hivm.hir.mmadL1 for tl.npuir_dot.
  // before:
  //   T.npuir_dot(T.region(A_BUF[0, 0], 1, 128, 1024),
  //               T.region(B_BUF[0, 0], 1, 1024, 256),
  //               T.region(C_BUF[0, 0], 3, 128, 256), T.bool(True))
  // after:
  // hivm.hir.mmadL1 ins(%alloc_8,  %alloc_5,  %true,  %c128,  %c64,  %c64 :
  //                     memref<128x64xf16,  #hivm.address_space<cbuf>>,
  //                     memref<64x64xf16,  #hivm.address_space<cbuf>>,
  //                     i1,  index,  index,  index)
  //                 outs(%alloc_9 : memref<128x64xf32,
  //                      #hivm.address_space<cc>>)
  tvm::tl::NpuirDot npuirop(op->args, this->vmap);
  Array<PrimExpr> a_region_shape, b_region_shape;
  for (int i = 0; i < npuirop.src0_range.size(); i++) {
    a_region_shape.push_back(npuirop.src0_range[i].get()->extent);
    b_region_shape.push_back(npuirop.src1_range[i].get()->extent);
  }

  mlir::Location unknown_loc = builder.getUnknownLoc();
  mlir::IndexType idx_ty = builder.getIndexType();
  mlir::Value a = GenSubviewFromRegion(npuirop.src0, npuirop.src0_range);
  mlir::Value b = GenSubviewFromRegion(npuirop.src1, npuirop.src1_range);
  mlir::Value c = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  mlir::TypeRange result_tensors = {};
  mlir::Value init_condition = MakeValue(npuirop.initC);
  mlir::Value real_m = CreateIndexCastOp(MakeValue(a_region_shape[0]));
  mlir::Value real_k = CreateIndexCastOp(MakeValue(b_region_shape[0]));
  mlir::Value real_n = CreateIndexCastOp(MakeValue(b_region_shape[1]));
  mlir::Value per_channel_bias = mlir::Value{};
  mlir::UnitAttr a_transpose =
      npuirop.a_transpose ? builder.getUnitAttr() : mlir::UnitAttr();
  mlir::UnitAttr b_transpose =
      npuirop.b_transpose ? builder.getUnitAttr() : mlir::UnitAttr();
  mlir::UnitAttr enable_HF32 = mlir::UnitAttr();
  builder.create<mlir::hivm::MmadL1Op>(
      unknown_loc, result_tensors, a, b, init_condition, real_m, real_k, real_n,
      c, per_channel_bias, a_transpose, b_transpose, enable_HF32);
}

void CodeGenTileLangNPUIRAPI::BitcastCodegen(const CallNode *op) {
  tvm::tl::NpuirBitcast npuirop(op->args, this->vmap);

  auto dl_dtype = tvm::runtime::String2DLDataType(npuirop.dtype);
  auto tir_dtype = DataType(dl_dtype);

  mlir::Value src = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  auto src_type = src.getType();
  if (auto memref_type = mlir::dyn_cast<MemRefType>(src_type)) {
    auto src_shape = memref_type.getShape();
    auto src_layout = memref_type.getLayout();
    auto src_memspace = memref_type.getMemorySpace();
    auto res_type = mlir::MemRefType::get(src_shape, DTypetoMLIRType(tir_dtype),
                                          src_layout, src_memspace);
    builder.create<mlir::hivm::BitcastOp>(builder.getUnknownLoc(), res_type,
                                          src);
  } else if (auto tensor_type = mlir::dyn_cast<RankedTensorType>(src_type)) {
    auto src_shape = tensor_type.getShape();
    auto res_type =
        mlir::RankedTensorType::get(src_shape, DTypetoMLIRType(tir_dtype));
    builder.create<mlir::hivm::BitcastOp>(builder.getUnknownLoc(), res_type,
                                          src);
  } else {
    llvm_unreachable("Unspported source type (expected tensor or memref)");
  }
}

template <typename T>
void CodeGenTileLangNPUIRAPI::CreateHIVMBinaryVectorOp(const CallNode *op) {
  auto processImm = [&](mlir::Value &src, int arg_id,
                        Array<PrimExpr> &buffer_shape) {
    if (op->args[arg_id].as<IntImm>() || op->args[arg_id].as<FloatImm>()) {
      // Scalar case
      const CallNode *region_node = op->args[1 - arg_id].as<CallNode>();
      const BufferLoadNode *buffer_load_node =
          region_node->args[0].as<BufferLoadNode>();
      if (op->args[arg_id]->dtype != buffer_load_node->buffer->dtype) {
        src = ScalarConvertType(op->args[arg_id],
                                buffer_load_node->buffer->dtype);
      } else {
        src = MakeValue(op->args[arg_id]);
      }
    } else {
      // Vector case
      const CallNode *region_node = op->args[arg_id].as<CallNode>();
      buffer_shape = region_node->args[0].as<BufferLoadNode>()->buffer->shape;
      src = GenSubviewFromRegion(region_node);
    }
  };
  // src0 src1
  mlir::Value src0, src1;
  Array<PrimExpr> buffer_shape0, buffer_shape1;
  processImm(src0, 0, buffer_shape0);
  processImm(src1, 1, buffer_shape1);
  // dst
  const CallNode *region_node_dst = op->args[2].as<CallNode>();
  // Result will always be a vector. No need to add scalar check.
  mlir::Value dst = GenSubviewFromRegion(region_node_dst);
  // transpose
  mlir::DenseI64ArrayAttr transpose = builder.getDenseI64ArrayAttr({});
  // broadcast
  llvm::SmallVector<int64_t> dims =
      getBroadcastDim(buffer_shape0, buffer_shape1);
  mlir::DenseI64ArrayAttr broadcast = builder.getDenseI64ArrayAttr(dims);
  // Create hivm::op
  auto loc = builder.getUnknownLoc();
  if constexpr (std::is_same_v<T, mlir::hivm::VCmpOp>) {
    mlir::hivm::CompareMode mode =
        COMPARE_MODE[op->args[3].as<StringImm>().value()->value];
    auto cmp_attr =
        mlir::hivm::CompareModeAttr::get(builder.getContext(), mode);
    builder.create<T>(loc, mlir::TypeRange{}, mlir::ValueRange{src0, src1},
                      mlir::ValueRange{dst}, cmp_attr, transpose, broadcast);
  } else if constexpr (std::is_same_v<T, mlir::hivm::VPowOp>) {
    builder.create<T>(loc, mlir::TypeRange{}, mlir::ValueRange{src0, src1},
                      mlir::ValueRange{dst}, mlir::Value(), transpose,
                      broadcast);
  } else if constexpr (std::is_same_v<T, mlir::hivm::VShROp>) {
    auto round_attr = mlir::BoolAttr::get(builder.getContext(),
                                          op->args[3].as<Bool>().value());
    builder.create<T>(loc, mlir::TypeRange{}, mlir::ValueRange{src0, src1},
                      mlir::ValueRange{dst}, round_attr, transpose, broadcast);
  } else {
    builder.create<T>(loc, mlir::TypeRange{}, mlir::ValueRange{src0, src1},
                      mlir::ValueRange{dst}, transpose, broadcast);
  }
}

template <typename T>
void CodeGenTileLangNPUIRAPI::SyncBlockCodegen(const T &sync_op) {
  // Extract values from CallNode op
  // flag can either be a constant or a SSA ID
  mlir::OpFoldResult flag_id;
  if (auto *int_imm = sync_op.flag_id.template as<tvm::tir::IntImmNode>()) {
    flag_id = builder.getI64IntegerAttr(int_imm->value);
  } else if (sync_op.flag_id.dtype().bits() < FLAG_ID_BITS) {
    auto cast_node =
        std::make_unique<tir::Cast>(DataType::Int(64), sync_op.flag_id);
    flag_id = MakeValue(*cast_node);
  } else {
    flag_id = MakeValue(sync_op.flag_id);
  }
  // Create HIVM/MLIR Attrs
  mlir::hivm::TCoreTypeAttr coreAttrType = mlir::hivm::TCoreTypeAttr::get(
      builder.getContext(), TCORE_MAP[this->current_coretype]);
  mlir::hivm::PipeAttr tPipAttrType =
      mlir::hivm::PipeAttr::get(builder.getContext(), PIPE_MAP["PIPE_S"]);
  mlir::hivm::PipeAttr pipAttrType = mlir::hivm::PipeAttr::get(
      builder.getContext(), PIPE_MAP[sync_op.pipe_type]);

  if constexpr (std::is_same_v<T, tvm::tl::NpuirSyncBlockSet> ||
                std::is_same_v<T, tvm::tl::NpuirSyncBlock>) {
    auto ffts_base_addr = mlir::Value();
    mlir::hivm::SyncBlockInstrModeAttr sync_mode =
        mlir::hivm::SyncBlockInstrModeAttr::get(
            builder.getContext(), SYNC_BLOCK_MODE_MAP[sync_op.mode]);
    // Create HIVM SyncBlockSetOp
    builder.create<mlir::hivm::SyncBlockSetOp>(
        builder.getUnknownLoc(), coreAttrType, pipAttrType, tPipAttrType,
        flag_id, ffts_base_addr, sync_mode);
  }
  if constexpr (std::is_same_v<T, tvm::tl::NpuirSyncBlockWait> ||
                std::is_same_v<T, tvm::tl::NpuirSyncBlock>) {
    // Create HIVM SyncBlockWaitOp
    builder.create<mlir::hivm::SyncBlockWaitOp>(builder.getUnknownLoc(),
                                                 coreAttrType, tPipAttrType,
                                                 pipAttrType, flag_id);
  }
}

mlir::Value CodeGenTileLangNPUIRAPI::GetEventID(PrimExpr id) {
  DataType raw_type = id.dtype();
  mlir::Value origin_id = MakeValue(id);
  mlir::Value i64_id = origin_id;
  CHECK(raw_type.is_int() || raw_type.is_uint());
  if (raw_type.bits() < FLAG_ID_BITS) {
    mlir::Location unknown_loc = builder.getUnknownLoc();
    mlir::IntegerType int64_type = builder.getI64Type();
    if (raw_type.is_int()) {
      i64_id = builder.create<mlir::arith::ExtSIOp>(unknown_loc, int64_type,
                                                     origin_id);
    } else {
      i64_id = builder.create<mlir::arith::ExtUIOp>(unknown_loc, int64_type,
                                                     origin_id);
    }
  }
  return i64_id;
}

template <typename T, typename U>
void CodeGenTileLangNPUIRAPI::PipeFlagCodegen(const CallNode *op) {
  T sync_op(op->args, this->vmap);
  mlir::Location unknown_loc = builder.getUnknownLoc();
  mlir::hivm::PipeAttr set_pipe =
      mlir::hivm::PipeAttr::get(builder.getContext(), PIPE_MAP[sync_op.pipe1]);
  mlir::hivm::PipeAttr wait_pipe =
      mlir::hivm::PipeAttr::get(builder.getContext(), PIPE_MAP[sync_op.pipe2]);
  mlir::Value event_id = GetEventID(sync_op.event_id);
  builder.create<U>(unknown_loc, set_pipe, wait_pipe, mlir::hivm::EventAttr{},
                     event_id);
}

void CodeGenTileLangNPUIRAPI::DebugPrintCodegen(const CallNode *op) {
  std::string prefix = "";
  bool hex = false;
  mlir::Value arg;
  if (op->op.same_as(Op::Get("tl.npuir_debug_print_var"))) {
    tvm::tl::NpuirDevicePrintVar npuirop(op->args, this->vmap);
    arg = MakeValue(npuirop.src);
    prefix = npuirop.prefix;
    hex = npuirop.hex;
  } else {
    tvm::tl::NpuirDevicePrintBuf npuirop(op->args, this->vmap);
    arg = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
    prefix = npuirop.prefix;
    hex = npuirop.hex;
  }

  mlir::Location unknown_loc = builder.getUnknownLoc();
  builder.create<mlir::hivm::DebugOp>(unknown_loc, "print", prefix, hex, arg,
                                       mlir::hivm::TCoreTypeAttr{});
}

void CodeGenTileLangNPUIRAPI::CallExternCodegen(const CallNode *op) {
  // Todo: Implementation pending
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const CallNode *op) {
  if (op->op.same_as(Op::Get("tl.npuir_pipe_barrier"))) {
    BarrierCodegen(op);
  } else if (op->op.same_as(builtin::call_extern())) {
    CallExternCodegen(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_set_flag"))) {
    PipeFlagCodegen<tvm::tl::NpuirSetFlag, mlir::hivm::SetFlagOp>(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_wait_flag"))) {
    PipeFlagCodegen<tvm::tl::NpuirWaitFlag, mlir::hivm::WaitFlagOp>(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_sync_block"))) {
    tvm::tl::NpuirSyncBlock sync_op(op->args, this->vmap);
    SyncBlockCodegen(sync_op);
  } else if (op->op.same_as(Op::Get("tl.npuir_sync_block_set"))) {
    tvm::tl::NpuirSyncBlockSet sync_op(op->args, this->vmap);
    SyncBlockCodegen(sync_op);
  } else if (op->op.same_as(Op::Get("tl.npuir_sync_block_wait"))) {
    tvm::tl::NpuirSyncBlockWait sync_op(op->args, this->vmap);
    SyncBlockCodegen(sync_op);
  } else if (op->op.same_as(Op::Get("tl.ascend_copy"))) {
    AscendCopyCodegen(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_add"))) {
    CreateHIVMBinaryVectorOp<mlir::hivm::VAddOp>(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_exp"))) {
    UnaryVecOpCodegen<tvm::tl::NpuirExp, mlir::hivm::VExpOp>(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_ln"))) {
    UnaryVecOpCodegen<tvm::tl::NpuirLn, mlir::hivm::VLnOp>(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_relu"))) {
    UnaryVecOpCodegen<tvm::tl::NpuirRelu, mlir::hivm::VReluOp>(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_sqrt"))) {
    UnaryVecOpCodegen<tvm::tl::NpuirSqrt, mlir::hivm::VSqrtOp>(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_rsqrt"))) {
    UnaryVecOpCodegen<tvm::tl::NpuirRsqrt, mlir::hivm::VRsqrtOp>(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_abs"))) {
    UnaryVecOpCodegen<tvm::tl::NpuirAbs, mlir::hivm::VAbsOp>(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_rec"))) {
    UnaryVecOpCodegen<tvm::tl::NpuirRec, mlir::hivm::VRecOp>(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_not"))) {
    UnaryVecOpCodegen<tvm::tl::NpuirNot, mlir::hivm::VNotOp>(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_select"))) {
    VselectCodegen(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_cmp"))) {
    CreateHIVMBinaryVectorOp<mlir::hivm::VCmpOp>(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_load_nd2nz"))) {
    Nd2NzCodegen(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_store_nz2nd"))) {
    Nz2NdCodegen(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_store_fixpipe"))) {
    FixpipeCodegen(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_dot"))) {
    DotCodegen(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_bitcast"))) {
    BitcastCodegen(op);
  }  else if (op->op.same_as(Op::Get("tl.npuir_div"))) {
    CreateHIVMBinaryVectorOp<mlir::hivm::VDivOp>(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_mul"))) {
    CreateHIVMBinaryVectorOp<mlir::hivm::VMulOp>(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_sub"))) {
    CreateHIVMBinaryVectorOp<mlir::hivm::VSubOp>(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_max"))) {
    CreateHIVMBinaryVectorOp<mlir::hivm::VMaxOp>(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_min"))) {
    CreateHIVMBinaryVectorOp<mlir::hivm::VMinOp>(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_or"))) {
    CreateHIVMBinaryVectorOp<mlir::hivm::VOrOp>(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_and"))) {
    CreateHIVMBinaryVectorOp<mlir::hivm::VAndOp>(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_xor"))) {
    CreateHIVMBinaryVectorOp<mlir::hivm::VXorOp>(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_pow"))) {
    CreateHIVMBinaryVectorOp<mlir::hivm::VPowOp>(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_shl"))) {
    CreateHIVMBinaryVectorOp<mlir::hivm::VShLOp>(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_shr"))) {
    CreateHIVMBinaryVectorOp<mlir::hivm::VShROp>(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_brc"))) {
    VbrcCodegen(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_cast"))) {
    VcastCodegen(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_reduce"))) {
    VreduceCodegen(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_cumsum"))) {
    VcumsumCodegen(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_atomic_add"))) {
    VAtomicAddCodegen(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_gather"))) {
    VgatherCodegen(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_transpose"))) {
    VtransposeCodegen(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_interleave"))) {
    VinterleaveCodegen(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_deinterleave"))) {
    VdeinterleaveCodegen(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_arange"))) {
    VarangeCodegen(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_concat"))) {
    VconcatCodegen(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_pad"))) {
    VpadCodegen(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_flip"))) {
    VflipCodegen(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_debug_print_var")) ||
             op->op.same_as(Op::Get("tl.npuir_debug_print_buffer_value"))) {
    DebugPrintCodegen(op);
  } else {
    VisitExpr_(op);
  }
  return mlir::Value();
}

void CodeGenTileLangNPUIRAPI::VisitStmt_(const LetStmtNode *op) {

  // EmitDebugLocation(op);
  const VarNode *v = op->var.get();
  ICHECK(!var_map_.count(v));
  if (v->dtype.is_handle()) {
    if (!is_restricted_) {
      alias_var_set_.insert(v);
    }
  }
  mlir::Value value = MakeValue(op->value);

  // TIR has type-annotations on variables, but not on each PrimExpr.
  // Therefore, to have the correct LLVM type for pointers, we may
  // need to introduce a pointer-cast, even though pointer-to-pointer
  // casts are not expressible with the `tir::CastNode`.
  if (v->dtype.is_handle() && v->type_annotation.defined()) {
    CHECK(op->value->dtype.is_handle())
        << "Variable " << op->var << " is a pointer with type " << op->value
        << ", but is being bound to expression with type " << op->value->dtype;
    // auto* llvm_type = GetMLIRType(v->type_annotation);
    // if (llvm_type != value.getType()) {
    //   //value->setName((v->name_hint + "_void_ptr").c_str());
    //   //value = builder_->CreatePointerCast(value, llvm_type);
    // }
  }

  // AddDebugInformation(value, op->var);
  var_map_[v] = value;
  // analyzer_->Bind(op->var, op->value);
  // if (alloc_storage_info_.count(v) && alloc_storage_info_[v].alignment > 1) {
  //   builder_->CreateAlignmentAssumption(*data_layout_, GetVarValue(v),
  //                                       alloc_storage_info_[v].alignment);
  // }
  // AddDebugInformation(value, op->var);

  VisitStmt(op->body);
}

void CodeGenTileLangNPUIRAPI::VisitStmt_(const AttrStmtNode *op) {
  if (op->attr_key == "thread_extent") {
    IterVar iv = Downcast<IterVar>(op->node);
    if (iv->thread_tag == "blockIdx.x" && iv->var->name_hint != "_") {
      mlir::Value indexOp = GetAndCastIndexOp<mlir::hivm::GetBlockIdxOp>(iv);
      var_map_[iv->var.get()] = indexOp;
    } else if (iv->thread_tag == "blockIdx.y" && iv->var->name_hint != "_") {
      mlir::Value indexOp = GetAndCastIndexOp<mlir::hivm::GetSubBlockIdxOp>(iv);
      var_map_[iv->var.get()] = indexOp;
    }
    this->VisitStmt(op->body);
    return;
  } else if (op->attr_key == "resource_scope") {
    auto resource_id = Downcast<IntImm>(op->value)->value;
    auto resource_name = resource_id == 0 ? "aic" : "aiv";
    if (NPU_CORETYPE_STR[this->current_coretype] == resource_name) {
      this->VisitStmt(op->body);
    }
    // else do nothing but return.
    return;
  }
  VisitStmt(op->body);
}

template <typename T>
mlir::Value CodeGenTileLangNPUIRAPI::GetAndCastIndexOp(const IterVar iv) {
  auto indexOp = builder.create<T>(mlir::UnknownLoc::get(&context));
  auto truncOp = builder.create<mlir::arith::TruncIOp>(
      mlir::UnknownLoc::get(&context),
      builder.getI32Type(), // The target integer type
      indexOp               // The source float value to cast
  );
  return truncOp;
}

/// Generate memref.alloc for TIR AllocateNode like T.decl_buffer.
/// before:
///      A_VEC = T.decl_buffer((128, 256), "float16", scope="shared")
/// after:
///      %A_VEC = memref.alloc() : memref<128x256xf16,
///      #hivm.address_space<ub>>

void CodeGenTileLangNPUIRAPI::VisitStmt_(const AllocateNode *op) {
  ICHECK(!is_zero(op->condition));
  std::string scope = GetPtrStorageScope(op->buffer_var);
  std::map<std::string, NPU_CORETYPE> scope_coretype_map{
      {"shared", NPU_CORETYPE::AIV},
      {"shared.dyn", NPU_CORETYPE::AIC},
      {"wmma.accumulator", NPU_CORETYPE::AIC}};
  if (scope_coretype_map[scope] == this->current_coretype) {
    std::vector<long int> shape = GetShape(op->extents);
    std::vector<int64_t> strides = GetStrideFromShapeAPI(op->extents);

    auto layoutMap =
        mlir::StridedLayoutAttr::get(builder.getContext(), 0, strides);
    mlir::hivm::AddressSpace address_space = GetHIVMAddressSpace(scope);
    auto memorySpaceHIVMAttr =
        mlir::hivm::AddressSpaceAttr::get(builder.getContext(), address_space);

    auto memrefType = mlir::MemRefType::get(shape, DTypetoMLIRType(op->dtype),
                                            layoutMap, memorySpaceHIVMAttr);

    auto allocOp = builder.create<mlir::memref::AllocOp>(
        builder.getUnknownLoc(), memrefType);

    // Update var_map_ with the new variable
    ICHECK(!var_map_.count(op->buffer_var.get()));
    var_map_[op->buffer_var.get()] = allocOp.getResult();
  }
  this->VisitStmt(op->body);
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const MinNode *op) {
  auto lhs = MakeValue(op->a);
  auto rhs = MakeValue(op->b);
  mlir::Value mlirVal;
  if (op->dtype.is_int()) {
    mlirVal =
        BinaryOpCodegen<mlir::arith::MinSIOp, std::nullptr_t>(
            op, nullptr, lhs, rhs);
  } else if (op->dtype.is_uint()) {
    mlirVal =
        BinaryOpCodegen<mlir::arith::MinUIOp, std::nullptr_t>(
            op, nullptr, lhs, rhs);
  } else if (op->dtype.is_float()) {
    mlirVal = BinaryOpCodegen<mlir::arith::MinimumFOp,
                              std::nullptr_t>(op, nullptr, lhs, rhs);
  }
  return mlirVal;
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const MaxNode *op) {
  auto lhs = MakeValue(op->a);
  auto rhs = MakeValue(op->b);
  mlir::Value mlirVal;
  if (op->dtype.is_int()) {
    mlirVal =
        BinaryOpCodegen<mlir::arith::MaxSIOp, std::nullptr_t>(
            op, nullptr, lhs, rhs);
  } else if (op->dtype.is_uint()) {
    mlirVal =
        BinaryOpCodegen<mlir::arith::MaxUIOp, std::nullptr_t>(
            op, nullptr, lhs, rhs);
  } else if (op->dtype.is_float()) {
    mlirVal = BinaryOpCodegen<mlir::arith::MaximumFOp,
                              std::nullptr_t>(op, nullptr, lhs, rhs);
  }
  return mlirVal;
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const AddNode *op) {
  auto lhs = MakeValue(op->a);
  auto rhs = MakeValue(op->b);
  mlir::Value mlirVal;
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    mlirVal =
        BinaryOpCodegen<mlir::arith::AddIOp, std::nullptr_t>(
            op, nullptr, lhs, rhs);
  } else if (op->dtype.is_float()) {
    mlirVal =
        BinaryOpCodegen<mlir::arith::AddFOp, std::nullptr_t>(
            op, nullptr, lhs, rhs);
  }
  return mlirVal;
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const SubNode *op) {
  auto lhs = MakeValue(op->a);
  auto rhs = MakeValue(op->b);
  mlir::Value mlirVal;
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    mlirVal =
        BinaryOpCodegen<mlir::arith::SubIOp, std::nullptr_t>(
            op, nullptr, lhs, rhs);
  } else if (op->dtype.is_float()) {
    mlirVal =
        BinaryOpCodegen<mlir::arith::SubFOp, std::nullptr_t>(
            op, nullptr, lhs, rhs);
  }
  return mlirVal;
}

mlir::Value
CodeGenTileLangNPUIRAPI::VisitExpr_(const FloatImmNode *op) {
  // check if same node already created
  // If already created return corresponding MLIR value and do not create duplicated MLIR Op
  std::pair<bool, mlir::Value> result = CheckPrimExprMap(op);
  if (result.first){
    return result.second;
  }
  auto type = DTypetoMLIRType(op->dtype);
  auto FloatConst = builder.create<mlir::arith::ConstantOp>(
      mlir::UnknownLoc::get(&context), builder.getFloatAttr(type, op->value));
  UpdatePrimExprMap(op, FloatConst);
  return FloatConst;
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const IntImmNode *op) {
  // check if same node already created
  // If already created return corresponding MLIR value and do not create duplicated MLIR Op
  std::pair<bool, mlir::Value> result = CheckPrimExprMap(op);
  if (result.first){
    return result.second;
  }
  auto type = DTypetoMLIRType(op->dtype);
  auto IntConst = builder.create<mlir::arith::ConstantOp>(
      mlir::UnknownLoc::get(&context),
      builder.getIntegerAttr(type, op->value));
  UpdatePrimExprMap(op, IntConst);
  return IntConst;
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const MulNode *op) {
  auto lhs = MakeValue(op->a);
  auto rhs = MakeValue(op->b);
  mlir::Value mlirVal;
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    mlirVal =
        BinaryOpCodegen<mlir::arith::MulIOp, std::nullptr_t>(
            op, nullptr, lhs, rhs);
  } else if (op->dtype.is_float()) {
    mlirVal =
        BinaryOpCodegen<mlir::arith::MulFOp, std::nullptr_t>(
            op, nullptr, lhs, rhs);
  }
  return mlirVal;
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const AndNode *op) {
  CHECK(op->a.dtype().is_int() || op->a.dtype().is_uint());
  CHECK(op->b.dtype().is_int() || op->b.dtype().is_uint());
  auto lhs = MakeValue(op->a);
  auto rhs = MakeValue(op->b);
  auto mlirVal =
      BinaryOpCodegen<mlir::arith::AndIOp, std::nullptr_t>(
          op, nullptr, lhs, rhs);
  return mlirVal;
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const OrNode *op) {
  CHECK(op->a.dtype().is_int() || op->a.dtype().is_uint());
  CHECK(op->b.dtype().is_int() || op->b.dtype().is_uint());
  auto lhs = MakeValue(op->a);
  auto rhs = MakeValue(op->b);
  auto mlirVal =
      BinaryOpCodegen<mlir::arith::OrIOp, std::nullptr_t>(
          op, nullptr, lhs, rhs);
  return mlirVal;
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const DivNode *op) {
  auto lhs = MakeValue(op->a);
  auto rhs = MakeValue(op->b);
  auto mlirVal =
      BinaryOpCodegen<mlir::arith::DivFOp, std::nullptr_t>(
          op, nullptr, lhs, rhs);
  return mlirVal;
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const SelectNode *op) {
  auto condition = MakeValue(op->condition);
  auto true_value = MakeValue(op->true_value);
  auto false_value = MakeValue(op->false_value);

  return builder.create<mlir::arith::SelectOp>(
      builder.getUnknownLoc(), condition, true_value, false_value);
}

String CodeGenTileLangNPUIRAPI::GetCurrentFunctionName(){
  return this->current_function_name;
}

void CodeGenTileLangNPUIRAPI::AddFunctionForCoreType(const GlobalVar &gvar,
                                                     const PrimFunc &f) {
  // clear previous generated state.
  this->InitFuncState();

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";
  this->current_function_name = static_cast<std::string>(global_symbol.value());
  if (this->func_coretype == NPU_CORETYPE::MIX) {
    this->current_function_name = this->current_function_name + "_mix_" + NPU_CORETYPE_STR[this->current_coretype];
  }

  // Create function type
  llvm::SmallVector<mlir::Type> funcArgs;
  llvm::DenseMap<size_t, mlir::Type> recastNeedInsert;
  // %arg0 is ffts addr
  funcArgs.emplace_back(builder.getI64Type());
  // %arg1 is SyncLockArgs
  funcArgs.emplace_back(
      MemRefType::get({ShapedType::kDynamic}, builder.getI8Type()));
  // %arg2 is workspace
  funcArgs.emplace_back(
      MemRefType::get({ShapedType::kDynamic}, builder.getI8Type()));
  int funcArgsOffset = funcArgs.size();
  this->vmap = f->buffer_map;
  for (size_t i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];

    if (v.dtype().is_handle()) {
      // add new memref obj
      auto argType = GetMLIRType(f->buffer_map[v]);
      recastNeedInsert[i] = argType;
      funcArgs.emplace_back(MemRefType::get(
          {ShapedType::kDynamic}, DTypetoMLIRType(f->buffer_map[v]->dtype),
          StridedLayoutAttr{},
          llvm::dyn_cast<MemRefType>(argType).getMemorySpace()));
    } else {
      funcArgs.emplace_back(DTypetoMLIRType(v.dtype()));
    }
  }
  // Add gridInfo for runtime
  for (int i = 0; i < 6; i++) {
    funcArgs.emplace_back(builder.getI32Type());
  }
  auto funcType = builder.getFunctionType(funcArgs, {});

  // Create function signature
  builder.setInsertionPointToEnd(module->getBody());
  auto funcOp = builder.create<func::FuncOp>(builder.getUnknownLoc(),
                                              this->current_function_name, funcType);
  mlir::Block *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  for (int i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];
    tir::Var real_v = v.dtype().is_handle() ? f->buffer_map[v]->data : v;
    var_map_[real_v.get()] = funcOp.getArgument(i + funcArgsOffset);
  }
  builder.create<hivm::SetFFTSBaseAddrOp>(builder.getUnknownLoc(),
                                           funcOp.getArgument(0));
  for (auto recastInfo : recastNeedInsert) {
    tir::Var v = f->params[recastInfo.first];
    tir::Var real_v = f->buffer_map[v]->data;
    auto memrefType = llvm::dyn_cast<MemRefType>(recastInfo.second);
    auto strideLayout =
        llvm::dyn_cast<StridedLayoutAttr>(memrefType.getLayout());
    auto recastOp = builder.create<memref::ReinterpretCastOp>(
        builder.getUnknownLoc(), memrefType, var_map_[real_v.get()],
        strideLayout.getOffset(), memrefType.getShape(),
        strideLayout.getStrides());
    var_map_[real_v.get()] = recastOp;
  }
  mlir::hacc::KernelArgTypeAttr accArgAttr = hacc::KernelArgTypeAttr::get(
      builder.getContext(), hacc::KernelArgType::kFFTSBaseAddr);
  funcOp.setArgAttr(0, "hacc.arg_type", accArgAttr);
  funcOp->setAttr("SyncBlockLockArgIdx", builder.getI64IntegerAttr(0));
  funcOp->setAttr("WorkspaceArgIdx", builder.getI64IntegerAttr(1));
  auto haccEntryAttr = hacc::stringifyHACCToLLVMIRTranslateAttr(
      hacc::HACCToLLVMIRTranslateAttr::ENTRY);
  funcOp->setAttr(haccEntryAttr, builder.getUnitAttr());
  auto haccFuncTypeAttr = hacc::HACCFuncTypeAttr::get(
      builder.getContext(), hacc::HACCFuncType::DEVICE);
  funcOp->setAttr(hacc::HACCFuncTypeAttr::name, haccFuncTypeAttr);
  auto funcCoreTypeAttr = hivm::TFuncCoreTypeAttr::get(
      builder.getContext(), NPUIR_FUNCCORETYPE_STR[this->current_coretype]);
  funcOp->setAttr(hivm::TFuncCoreTypeAttr::name, funcCoreTypeAttr);
  if (this->func_coretype == NPU_CORETYPE::MIX) {
    funcOp->setAttr(hivm::TPartOfMixAttr::name, builder.getUnitAttr());
    funcOp->setAttr("mix_mode", builder.getStringAttr(
                                    NPU_CORETYPE_STR[NPU_CORETYPE::MIX]));
  } else {
    funcOp->setAttr("mix_mode", builder.getStringAttr(
                                    NPU_CORETYPE_STR[this->current_coretype]));
  }
  // Call VisitStmt on function body
  this->VisitStmt(f->body);
  builder.create<func::ReturnOp>(builder.getUnknownLoc());
}

void CodeGenTileLangNPUIRAPI::InitFuncState() {
  var_map_.clear();
  alias_var_set_.clear();
  alloc_storage_info_.clear();
  volatile_buf_.clear();
  analyzer_.reset(new arith::Analyzer());
  prim_expr_map.clear();
  mlir_value_map.clear();
  this->current_function_name = "";
}

void CodeGenTileLangNPUIRAPI::AddFunction(const GlobalVar& gvar, const PrimFunc& f)
{
  InferFuncCoreType infer;
  infer.VisitStmt(f->body);

  this->func_coretype = infer.func_coretype; // NPU_CORETYPE::MIX;

  auto moduleCoreType = mlir::hivm::TModuleCoreTypeAttr::get(
      &this->context, NPUIR_MODULECORETYPE_STR[this->func_coretype]);
  this->module->getOperation()->setAttr(mlir::hivm::TModuleCoreTypeAttr::name,
                                        moduleCoreType);

  if (this->func_coretype == NPU_CORETYPE::MIX ||
      this->func_coretype == NPU_CORETYPE::AIC) {
    this->current_coretype = NPU_CORETYPE::AIC;
    AddFunctionForCoreType(gvar, f);
  }

  if (this->func_coretype == NPU_CORETYPE::MIX ||
      this->func_coretype == NPU_CORETYPE::AIV) {
    this->current_coretype = NPU_CORETYPE::AIV;
    AddFunctionForCoreType(gvar, f);
  }
}

// New Expr functions after removing inheritance form CodeGenC class

mlir::Value CodeGenTileLangNPUIRAPI::GetVarValue(const VarNode *v) const {
  auto it = var_map_.find(v);
  ICHECK(it != var_map_.end()) << "cannot find variable " << v->name_hint;
  return it->second;
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const VarNode *op) {
  return GetVarValue(op);
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const StringImmNode *op) {
  // Todo: Implementation pending
  LOG(FATAL) << "StringImmNode case not supported!";
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const ModNode *op) {
  auto lhs = MakeValue(op->a);
  auto rhs = MakeValue(op->b);
  mlir::Value mlirVal;
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    mlirVal =
        BinaryOpCodegen<mlir::arith::RemSIOp, std::nullptr_t>(
            op, nullptr, lhs, rhs);
  } else if (op->dtype.is_float()) {
    mlirVal =
        BinaryOpCodegen<mlir::arith::RemFOp, std::nullptr_t>(
            op, nullptr, lhs, rhs);
  }
  return mlirVal;
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const NotNode *op) {
  // check if same node already created
  // If already created return corresponding MLIR value and do not create duplicated MLIR Op
  std::pair<bool, mlir::Value> result = CheckPrimExprMap(op);
  if (result.first){
    return result.second;
  }
  // Not operator does not exist in arith
  // Need to use XOR for Not
  auto trueValue = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getI1Type(),
      builder.getBoolAttr(true));
  auto inputValue = MakeValue(op->a);
  auto xorOperation = builder.create<mlir::arith::XOrIOp>(
      builder.getUnknownLoc(), inputValue, trueValue.getResult());
 UpdatePrimExprMap(op, xorOperation);
  return xorOperation;
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const LetNode *op) {
  auto it = var_map_.find(op->var.get());
  if (it != var_map_.end()) {
    LOG(FATAL) << "Variable already exists: " << op->var.get()->name_hint;
  }
  auto var_value = MakeValue(op->value);
  var_map_[op->var.get()] = var_value;
  return MakeValue(op->body);
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const BufferLoadNode *op) {
  auto buffer = op->buffer;
  auto indices = op->indices;

  // Check pre-conditions
  if (op->dtype.lanes() != 1) {
    LOG(FATAL) << "lanes not one";
  }
  if (op->dtype != buffer->dtype) {
    LOG(FATAL) << "The load type and buffer element type do not match";
  }

  // Convert buffer from Buffer in TIR 2 memref in MLIR
  auto mem = var_map_[buffer->data.get()];

  // Convert index from PrimExpr in TIR 2 index type in MLIR
  SmallVector<mlir::Value> convert_inds;
  for (auto index : indices) {
    mlir::Value indexVal = CreateIndexCastOp(MakeValue(index));
    convert_inds.push_back(indexVal);
  }

  // Create memef.load op in MLIR
  return builder.create<mlir::memref::LoadOp>(builder.getUnknownLoc(), mem,
                                               convert_inds);
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const RampNode *op) {
  // Todo: Implementation pending
  LOG(FATAL) << "RampNode case not supported!";
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const ShuffleNode *op) {
  // Todo: Implementation pending
  LOG(FATAL) << "ShuffleNode case not supported!";
}

mlir::Value CodeGenTileLangNPUIRAPI::VisitExpr_(const BroadcastNode *op) {
  // Todo: Implementation pending
  LOG(FATAL) << "BroadcastNode case not supported!";
}

void CodeGenTileLangNPUIRAPI::VisitStmt_(const BufferStoreNode *op) {
  auto buffer = op->buffer;
  auto value = op->value;
  auto indices = op->indices;

  if (op->value.dtype().lanes() != 1) {
    LOG(FATAL) << "lanes not one";
  }
  if (op->value.dtype() != buffer->dtype) {
    LOG(FATAL) << "The store type and buffer element type do not match";
  }

  auto mem = var_map_[buffer->data.get()];

  auto mlir_value = MakeValue(value);

  SmallVector<mlir::Value> convert_inds;
  for (auto index : indices) {
    mlir::Value indexVal = CreateIndexCastOp(MakeValue(index));
    convert_inds.push_back(indexVal);
  }

  builder.create<mlir::memref::StoreOp>(builder.getUnknownLoc(), mlir_value,
                                         mem, convert_inds);
}

void CodeGenTileLangNPUIRAPI::VisitStmt_(const WhileNode *op) {
  // Todo: Implementation pending
  LOG(FATAL) << "WhileNode case not supported!";
}

void CodeGenTileLangNPUIRAPI::VisitStmt_(const AllocateConstNode *op) {
  // Todo: Implementation pending
  LOG(FATAL) << "AllocateConstNode case not supported!";
}

void CodeGenTileLangNPUIRAPI::VisitStmt_(const AssertStmtNode *op) {
  // Todo: Implementation pending
  LOG(FATAL) << "AssertStmtNode case not supported!";
}

void CodeGenTileLangNPUIRAPI::VisitStmt_(const SeqStmtNode *op) {
  // EmitDebugLocation(op);
  for (Stmt stmt : op->seq) {
    this->VisitStmt(stmt);
  }
}

void CodeGenTileLangNPUIRAPI::VisitStmt_(const EvaluateNode *op) {
  // EmitDebugLocation(op);
  MakeValue(op->value);
}

void CodeGenTileLangNPUIRAPI::VisitStmt_(const DeclBufferNode *op) {
  // EmitDebugLocation(op);
  VisitStmt(op->body);
}

} // namespace codegen
} // namespace tvm
