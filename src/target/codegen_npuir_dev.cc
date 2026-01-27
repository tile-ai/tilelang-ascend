// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file target/codegen.cc
 */

#include "codegen_npuir_dev.h"
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
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StructuredOpsUtils.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
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
#include "tvm/runtime/logging.h"

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
getBroadcastDim(const Array<PrimExpr> &buffer_shape0,
                const std::vector<int64_t> &buffer_shape1) {
  llvm::SmallVector<int64_t> dims;
  if (buffer_shape0.empty() || buffer_shape1.empty()) {
    return dims;
  }
  CHECK(buffer_shape0.size() == buffer_shape1.size());
  for (int i = 0; i < buffer_shape0.size(); i++) {
    if (*as_const_int(buffer_shape0[i]) == 1 &&
        buffer_shape1[i] != 1) {
      dims.emplace_back(i);
    } else if (*as_const_int(buffer_shape0[i]) != 1 &&
               buffer_shape1[i] == 1) {
      dims.emplace_back(i);
    } else {
      CHECK(*as_const_int(buffer_shape0[i]) == buffer_shape1[i]);
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

namespace {
  /// Infer function core type: aic, aiv, mix
  class InferFuncCoreType : public StmtExprVisitor {
    std::map<std::string, NPU_CORETYPE> scope_coretype_map{
        {"shared", NPU_CORETYPE::AIV},
        {"shared.cube", NPU_CORETYPE::AIC},
        {"wmma.accumulator", NPU_CORETYPE::AIC},
        {"wmma.matrix_a", NPU_CORETYPE::AIC},
        {"wmma.matrix_b", NPU_CORETYPE::AIC}};

  public:
    bool hasVector = false;
    bool hasCube = false;
    bool hasExpert = false;
    void VisitStmt(const Stmt &stmt) override {
      StmtExprVisitor::VisitStmt(stmt);
    }
    void VisitStmt_(const AttrStmtNode *op) final {
      // It is mixkernel iff there exists T.rs.
      if (op->attr_key == "resource_scope") {
        if (const auto* int_imm = op->value.as<IntImmNode>()) {
            if (int_imm->value == 1) {
                func_coretype = NPU_CORETYPE::MIX;
                hasExpert = true;
                return;
            }
        }
      }
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitExpr_(const CallNode *op) final {
    
      if (op->op.same_as(Op::Get("tl.npuir_dot")) 
        || op->op.same_as(Op::Get("tl.npuir_load_nd2nz")) 
        || op->op.same_as(Op::Get("tl.npuir_store_fixpipe"))) {
        hasCube = true;
      }
      else if (op->op.as<OpNode>()) {
        // Convert TVM String to std::string
        auto op_node = op->op.as<OpNode>();
        std::string op_name = op_node->name;
        // Check if it is another operation starting with tl.npuir
        if (op_name.find("tl.npuir") == 0) {
            hasVector = true;
        }
    }
      StmtExprVisitor::VisitExpr_(op);
    }
    void VisitStmt_(const AllocateNode *op) final {
      // It is cube kernel if there exists buffer with shared.dyn/wmma.xxx
      // address space
      std::string scope = GetPtrStorageScope(op->buffer_var);
      if (func_coretype != NPU_CORETYPE::MIX) {
        if (scope_coretype_map.count(scope) != 0) {
          func_coretype = scope_coretype_map[scope];
          hasExpert = true;
        }
      }
      StmtExprVisitor::VisitStmt_(op); 
    }
    NPU_CORETYPE func_coretype{NPU_CORETYPE::AIV};
  };
}  // namespace

/*****************************************************************************************
******************************************************************************************
Functions for CodeGenTileLangNPUIRDEV class
Todo: Remove CodeGenTileLangNPUIR class and use all functions from
CodeGenTileLangNPUIRDEV
******************************************************************************************
******************************************************************************************/

std::vector<int64_t>
CodeGenTileLangNPUIRDEV::GetStrideFromShapeAPI(Array<tvm::PrimExpr> shape) {
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

mlir::Value
CodeGenTileLangNPUIRDEV::ScalarConvertType(const PrimExpr &imm,
                                           DataType targetDtype) {
  auto castNode = std::make_unique<tir::Cast>(targetDtype, imm);
  return MakeValue(*castNode);
}

CodeGenTileLangNPUIRDEV::CodeGenTileLangNPUIRDEV() : builder(&context) {
  // Load MLIR dialects in the context
  this->context
      .loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                   mlir::linalg::LinalgDialect, mlir::scf::SCFDialect,
                   mlir::memref::MemRefDialect, mlir::hivm::HIVMDialect,
                   mlir::hfusion::HFusionDialect,
                   mlir::bufferization::BufferizationDialect>();
  // Create MLIR module
  this->module = ModuleOp::create(UnknownLoc::get(&this->context));
}

std::string CodeGenTileLangNPUIRDEV::Finish() {
  std::string mlirCode;
  llvm::raw_string_ostream os(mlirCode);
  module->print(os);
  return mlirCode;
}

inline mlir::hivm::AddressSpace
CodeGenTileLangNPUIRDEV::GetHIVMAddressSpace(String address_space) {
  if (address_space == "global")
    return mlir::hivm::AddressSpace::GM;
  else if (address_space == "shared")
    return mlir::hivm::AddressSpace::UB;
  else if (address_space == "shared.cube")
    return mlir::hivm::AddressSpace::L1;
  else if (address_space == "wmma.accumulator")
    return mlir::hivm::AddressSpace::L0C;
  return mlir::hivm::AddressSpace::Zero;
}

inline std::vector<long int>
CodeGenTileLangNPUIRDEV::GetShape(Array<PrimExpr> shape_in) {
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

mlir::Type CodeGenTileLangNPUIRDEV::GetMLIRType(const PrimExpr &expr) {
  auto ttype = GetType(expr);
  auto DType = GetRuntimeDataType(ttype);
  return DTypetoMLIRType(DType);
}

mlir::Type CodeGenTileLangNPUIRDEV::GetMLIRType(const Buffer &buffer) {
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

void CodeGenTileLangNPUIRDEV::VisitStmt_(const tir::ForNode *op) {

  CHECK(op->extent.dtype().is_int() || op->extent.dtype().is_uint());
  CHECK(op->min.dtype() == op->extent.dtype());

  auto lowerBoundId = MakeValue(op->min);
  auto upperBoundId = MakeValue(op->extent + op->min);

  // Collect all variables defined in the loop body,
  // which may need to be carried as loop values
  std::set<const tir::VarNode*> loop_carried_vars;
  std::vector<mlir::Value> init_values;

  // Traverse the body of the for loop body, and generate
  // region iter args
  CollectVarsUsedInBodyButDefinedOutside(op, loop_carried_vars);
  for (const auto* var_node : loop_carried_vars) {
    auto it = GetVarValue(var_node);
    ICHECK(it != mlir::Value{});
    init_values.push_back(it);
  }

  // Create the loop
  auto step = builder.create<mlir::arith::ConstantOp>(
      mlir::UnknownLoc::get(&context),
      builder.getIntegerAttr(GetMLIRType(op->min), 1));
  auto forOp = builder.create<mlir::scf::ForOp>(
      module->getLoc(),
      lowerBoundId,
      upperBoundId,
      step,
      init_values);

  // Set the insertion point to the body of the loop
  OpBuilder::InsertionGuard saved(builder);
  builder.setInsertionPointToStart(forOp.getBody());

  // Add a new layer to var_map_
  AddVarLayer();
  auto loop_var = op->loop_var;
  ICHECK(GetVarValue(loop_var.get()) == mlir::Value{});
  SetVarValue(loop_var.get(), forOp.getInductionVar());
  int iter = 0;
  for (const auto* var_node : loop_carried_vars) {
    SetVarValue(var_node, forOp.getRegionIterArg(iter++));
  }

  // Traverse the body of the for loop
  this->VisitStmt(op->body);

  // Collect the last updated value in the loop body as output yield
  std::vector<mlir::Value> yield_values;
  for (const auto* var_node : loop_carried_vars) {
    auto it = GetVarValue(var_node);
    ICHECK(it != mlir::Value{});
    yield_values.push_back(it);
  }

  if (!yield_values.empty()) {
    builder.create<mlir::scf::YieldOp>(module->getLoc(), yield_values);
  }

  // Remove the last layer of var_map_
  DeleteVarLayer();

  iter = 0;
  for (const auto* var_node : loop_carried_vars) {
    SetVarValue(var_node, forOp.getResult(iter++));
  }
}

void CodeGenTileLangNPUIRDEV::VisitStmt_(const tir::IfThenElseNode *op) {

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

void CodeGenTileLangNPUIRDEV::CollectVarsUsedInBodyButDefinedOutside(
    const tir::ForNode *op, 
    std::set<const VarNode*>& loop_carried_vars) {
  LoopCarriedVarCollector collector(this, loop_carried_vars);
  collector.VisitStmt(op->body);
}

mlir::Type CodeGenTileLangNPUIRDEV::DTypetoMLIRType(DataType t) { // NOLINT(*)
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

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const FloorDivNode *op) {
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

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const FloorModNode *op) {
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

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const LTNode *op) {
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

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const NENode *op) {
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

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const EQNode *op) {
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

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const LENode *op) {
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

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const GENode *op) {
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

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const GTNode *op) {
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

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const CastNode *op) {
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
CodeGenTileLangNPUIRDEV::GenSubviewFromRegion(const CallNode *region_node) {
  tvm::tl::RegionOp regionop(region_node->args, this->vmap);
  return GenSubviewFromRegion(regionop.GetBuffer(), regionop.GetRanges());
}

mlir::Value
CodeGenTileLangNPUIRDEV::GenExtractSliceFromRegion(const CallNode *region_node) {
  tvm::tl::RegionOp regionop(region_node->args, this->vmap);
  return GenExtractSliceFromRegion(regionop.GetBuffer(), regionop.GetRanges());
}

mlir::Value CodeGenTileLangNPUIRDEV::GenSubviewFromRegion(Buffer buffer_data,
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

mlir::Value CodeGenTileLangNPUIRDEV::GenExtractSliceFromRegion(Buffer buffer_data,
                                                               Array<Range> range) {
  Array<PrimExpr> region_shape;
  Array<PrimExpr> region_indices;
  for (Range r: range) {
    region_shape.push_back(r.get()->extent);
    region_indices.push_back(r.get()->min);
  }
  mlir::Value v_value = GetVarValue(buffer_data);
  if (IsEqual(buffer_data->shape, region_shape) && AllZero(region_indices)) {
    return v_value;
  }
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  for (Range r: range) {
    if (auto s_int = as_const_int(r.get()->min)) {
      offsets.push_back(builder.getI64IntegerAttr(*s_int));
    } else {
      mlir::Value indexVal = CreateIndexCastOp(MakeValue(r.get()->min));
      offsets.push_back(indexVal);
    }
    if (auto s_int = as_const_int(r.get()->extent)) {
      sizes.push_back(builder.getI64IntegerAttr(*s_int));
    } else {
      mlir::Value shapeVal = CreateIndexCastOp(MakeValue(r.get()->extent));
      sizes.push_back(shapeVal);
    }
    strides.push_back(builder.getI64IntegerAttr(1));
  }
  auto extractSliceOp =
      builder.create<mlir::tensor::ExtractSliceOp>(builder.getUnknownLoc(),
          v_value, offsets, sizes, strides);
  return extractSliceOp.getResult();
}


mlir::Value CodeGenTileLangNPUIRDEV::CreateIndexCastOp(mlir::Value src) {
  std::pair<bool, mlir::Value> result = CheckMLIRValueMap(src);
  if (result.first) {
    return result.second;
  }
  mlir::Value indexVal = builder.create<mlir::arith::IndexCastOp>(
    builder.getUnknownLoc(), builder.getIndexType(), src);
  UpdateMLIRValueMap(src, indexVal);
  return indexVal;
}

inline std::pair<bool, mlir::Value> CodeGenTileLangNPUIRDEV::CheckMLIRValueMap(mlir::Value val){
  mlir::Block *curr_block = builder.getInsertionBlock();
  auto it = this->mlir_value_map.find({val, curr_block});
  if (it != this->mlir_value_map.end()) {
    return std::pair(true, it->second);
  }
  return std::pair(false, mlir::Value());
}

inline void CodeGenTileLangNPUIRDEV::UpdateMLIRValueMap(const mlir::Value key, mlir::Value val){
  mlir::Block *curr_block = builder.getInsertionBlock();
  this->mlir_value_map[{key, curr_block}] = val;
}

inline std::pair<bool, mlir::Value> CodeGenTileLangNPUIRDEV::CheckPrimExprMap(const PrimExprNode * op){
  mlir::Block *curr_block = builder.getInsertionBlock();
  auto it = this->prim_expr_map.find({GetRef<PrimExpr>(op), curr_block});
  if (it != this->prim_expr_map.end()) {
    return std::pair(true, it->second);
  }
  return std::pair(false, mlir::Value());
}

inline void CodeGenTileLangNPUIRDEV::UpdatePrimExprMap(const PrimExprNode * key, mlir::Value val){
  mlir::Block *curr_block = builder.getInsertionBlock();
  this->prim_expr_map[{GetRef<PrimExpr>(key), curr_block}] = val;
}

// Type casting for mismatched element types
mlir::Value CodeGenTileLangNPUIRDEV::CreateCastIfTypeMismatch(mlir::Value src, mlir::Value dst) {
  // src is always a tensor, dst may be a tensor or a memref, the return value is always a tensor

  auto srcTensorTy = src.getType().dyn_cast<mlir::TensorType>();
  ICHECK(srcTensorTy) << "src must be a tensor";

  mlir::Type srcElemTy = mlir::getElementTypeOrSelf(src.getType());
  mlir::Type dstElemTy = mlir::getElementTypeOrSelf(dst.getType());

  if (srcElemTy == dstElemTy ) {
    return src;
  }

  auto resultTensorTy = mlir::RankedTensorType::get(
      srcTensorTy.getShape(), dstElemTy);
  
  auto loc = builder.getUnknownLoc();

  SmallVector<mlir::Value> dynamicDims;
  for (int64_t i = 0, rank = srcTensorTy.getRank(); i < rank; ++i) {
    if (srcTensorTy.isDynamicDim(i)) {
      dynamicDims.push_back(
          builder.create<mlir::tensor::DimOp>(loc, src, i));
    }
  }

  auto castDstTensor = builder.create<mlir::tensor::EmptyOp>(
      loc, resultTensorTy, dynamicDims);
  
  auto newCastOp = builder.create<mlir::hivm::VCastOp>(
      loc, resultTensorTy, src, 
      castDstTensor.getResult(), 
      mlir::hivm::RoundModeAttr::get(&context, mlir::hivm::RoundMode::RINT),
      nullptr);
      
  return newCastOp->getResult(0);
}

// Insert slice into tensor
mlir::Value CodeGenTileLangNPUIRDEV::InsertSlice(
    mlir::Value src_slice, 
    mlir::Value dst_tensor, 
    llvm::SmallVector<mlir::OpFoldResult>& dst_offsets,
    llvm::SmallVector<mlir::OpFoldResult>& dst_sizes,
    llvm::SmallVector<mlir::OpFoldResult>& dst_strides) {

  auto loc = builder.getUnknownLoc();

  auto dstTensorTy = dst_tensor.getType().dyn_cast<mlir::RankedTensorType>();
  assert(dstTensorTy && "dst_tensor must be a ranked tensor");

  auto insertOp = builder.create<mlir::tensor::InsertSliceOp>(
      loc,
      src_slice,
      dst_tensor,
      dst_offsets,
      dst_sizes,
      dst_strides
  );

  return insertOp.getResult();
}

// Reshape tensor
// Note: src type must be a tensor
mlir::Value CodeGenTileLangNPUIRDEV::MaybeReshapeTensor(mlir::Value src, mlir::Value dst) {

  ICHECK(src.getType().isa<mlir::TensorType>()) << "src must be a tensor";
  auto src_tensor_type = src.getType().cast<mlir::TensorType>();

  // Build target shape values: handle dynamic dimensions
  llvm::SmallVector<mlir::Value> target_shape_vals;
  llvm::SmallVector<int64_t> target_shape_static;
  if (dst.getType().isa<mlir::MemRefType>()) {
    auto dst_type = dst.getType().cast<mlir::MemRefType>();
    if (src_tensor_type.getShape() == dst_type.getShape()) {
      return src;
    }
    // Iterate through destination memref dimensions
    for (int i = 0; i < dst_type.getRank(); ++i) {
      int64_t dim = dst_type.getDimSize(i);
      
      if (mlir::ShapedType::isDynamic(dim)) {
        // Dynamic dimension: get actual size at runtime using memref.dim
        mlir::Value dim_val = builder.create<mlir::memref::DimOp>(builder.getUnknownLoc(), dst, i);
        target_shape_vals.push_back(dim_val);
        target_shape_static.push_back(mlir::ShapedType::kDynamic);
      } else {
        // Static dimension: create constant index value 
        mlir::Value dim_val = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), dim);
        target_shape_vals.push_back(dim_val);
        target_shape_static.push_back(dim);
      }
    }
  } else if (dst.getType().isa<mlir::RankedTensorType>()) {
    auto dst_type = dst.getType().cast<mlir::RankedTensorType>();
    if (src_tensor_type.getShape() == dst_type.getShape()) {
      return src;
    }
    // Iterate through destination memref dimensions
    for (int i = 0; i < dst_type.getRank(); ++i) {
      int64_t dim = dst_type.getDimSize(i);
      if (mlir::ShapedType::isDynamic(dim)) {
        // Dynamic dimension: get actual size at runtime using tensor.dim
        mlir::Value dim_val = builder.create<mlir::tensor::DimOp>(builder.getUnknownLoc(), dst, i);
        target_shape_vals.push_back(dim_val);
        target_shape_static.push_back(mlir::ShapedType::kDynamic);
      } else {
        // Static dimension: create constant index value 
        mlir::Value dim_val = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), dim);
        target_shape_vals.push_back(dim_val);
        target_shape_static.push_back(dim);
      }
    }    

  }

  // Create target shape type for the reshape operation
  auto target_shape_type = mlir::RankedTensorType::get(
      target_shape_static,  // Shape with dynamic dimension markers
      src_tensor_type.getElementType());
  
  // Create shape tensor (1D tensor containing all dimension sizes)
  mlir::Value shape_tensor;
  
  if (target_shape_vals.size() == 1) {
    // For 1D case, create tensor directly
    shape_tensor = builder.create<mlir::tensor::FromElementsOp>(
        builder.getUnknownLoc(), target_shape_vals);
  } else {
    // For multi-dimensional case, create 1D tensor of indices
    auto shape_tensor_type = mlir::RankedTensorType::get(
        {static_cast<int64_t>(target_shape_vals.size())},
        builder.getIndexType());
    shape_tensor = builder.create<mlir::tensor::FromElementsOp>(
        builder.getUnknownLoc(), shape_tensor_type, target_shape_vals);
  }
  
  // Create tensor.reshape operation to match destination shape
  mlir::Value reshaped_tensor = builder.create<mlir::tensor::ReshapeOp>(
      builder.getUnknownLoc(),
      target_shape_type,
      src,
      shape_tensor);
  
  return reshaped_tensor;
}

mlir::Value CodeGenTileLangNPUIRDEV::MaybeReshapeTensorByDstSize(
    mlir::Value src,
    llvm::ArrayRef<mlir::OpFoldResult> sizes) {

  ICHECK(src.getType().isa<mlir::TensorType>()) << "src must be a tensor";
  auto srcTensorTy = src.getType().cast<mlir::TensorType>();
  auto loc = builder.getUnknownLoc();

  llvm::SmallVector<mlir::Value> targetShapeVals;     // runtime values
  llvm::SmallVector<int64_t>     targetShapeStatic;   // static shape (kDynamic)

  for (mlir::OpFoldResult ofr : sizes) {
    if (auto attr = ofr.dyn_cast<mlir::Attribute>()) {
      auto intAttr = attr.cast<mlir::IntegerAttr>();
      int64_t dim = intAttr.getInt();

      targetShapeStatic.push_back(dim);
      targetShapeVals.push_back(
          builder.create<mlir::arith::ConstantIndexOp>(loc, dim));

    } else if (auto val = ofr.dyn_cast<mlir::Value>()) {
      targetShapeStatic.push_back(mlir::ShapedType::kDynamic);
      targetShapeVals.push_back(val);

    } else {
      llvm_unreachable("Invalid OpFoldResult in sizes");
    }
  }

  if (auto rankedTy = srcTensorTy.dyn_cast<mlir::RankedTensorType>()) {
    if (rankedTy.getRank() == (int64_t)targetShapeStatic.size()) {
      bool same = true;
      for (int i = 0; i < rankedTy.getRank(); ++i) {
        int64_t s0 = rankedTy.getDimSize(i);
        int64_t s1 = targetShapeStatic[i];
        if (s0 != s1) {
          if (mlir::ShapedType::isDynamic(s0) ||
              mlir::ShapedType::isDynamic(s1)) {
            same = false;
            break;
          }
          same = false;
          break;
        }
      }
      if (same) return src;
    }
  }

  auto targetTensorTy = mlir::RankedTensorType::get(
      targetShapeStatic,
      srcTensorTy.getElementType());

  mlir::Value shapeTensor;
  {
    auto shapeTensorTy = mlir::RankedTensorType::get(
        {static_cast<int64_t>(targetShapeVals.size())},
        builder.getIndexType());

    shapeTensor = builder.create<mlir::tensor::FromElementsOp>(
        loc, shapeTensorTy, targetShapeVals);
  }

  auto reshaped = builder.create<mlir::tensor::ReshapeOp>(
      loc,
      targetTensorTy,
      src,
      shapeTensor);

  return reshaped;
}

// Converts a TileLang range/region descriptor into {offs, sizes, strides}.
// NOTE: This codegen currently assumes unit-stride slicing on every dim.
// It supports ':' (contiguous) and point indexing (extent=1), e.g. [:, i, :, j].
template <typename RangeT>
CodeGenTileLangNPUIRDEV::SliceRange
CodeGenTileLangNPUIRDEV::MakeSliceRange(const RangeT& range) {
  SliceRange r;
  r.offs.reserve(range.size());
  r.sizes.reserve(range.size());
  r.strides.reserve(range.size());

  for (const auto& dim : range) {
    // offset
    if (auto off = as_const_int(dim->min)) {
      r.offs.push_back(builder.getI64IntegerAttr(*off));
    } else {
      mlir::Value offVal = CreateIndexCastOp(MakeValue(dim->min));
      r.offs.push_back(offVal);
    }

    // size (extent)
    if (auto ext = as_const_int(dim->extent)) {
      r.sizes.push_back(builder.getI64IntegerAttr(*ext));
    } else {
      mlir::Value extVal = CreateIndexCastOp(MakeValue(dim->extent));
      r.sizes.push_back(extVal);
    }

    // stride (unit stride only)
    r.strides.push_back(builder.getI64IntegerAttr(1));
  }
  return r;
}

// Allocates a statically-shaped local UB memref with a fixed alignment.
mlir::Value CodeGenTileLangNPUIRDEV::CreateStaticLocalUB(
    llvm::ArrayRef<int64_t> shape, mlir::Type elem_type, mlir::Location loc) {
  auto memrefType = mlir::MemRefType::get(shape, elem_type);
  auto allocOp = builder.create<mlir::memref::AllocOp>(loc, memrefType);
  return allocOp.getResult();
}

// Returns true if an OpFoldResult is a compile-time constant integer equal to 1.
// Used to detect static-1 dimensions for rank canonicalization.
bool CodeGenTileLangNPUIRDEV::IsStaticOneOFR(mlir::OpFoldResult ofr) const {
  if (auto attr = ofr.dyn_cast<mlir::Attribute>()) {
    if (auto ia = attr.dyn_cast<mlir::IntegerAttr>()) return ia.getInt() == 1;
  }
  return false;
}

// Holds the result of dropping static-1 dimensions from a shape expression.
// sizes: kept dims as OpFoldResult;
// projected: inferred static/dynamic ints;
// keptIdx: original indices kept.
CodeGenTileLangNPUIRDEV::CollapsedDims
CodeGenTileLangNPUIRDEV::CollapseStaticOneDims(
    llvm::ArrayRef<mlir::OpFoldResult> fullSizes) {
  CollapsedDims out;
  out.sizes.reserve(fullSizes.size());
  out.projected.reserve(fullSizes.size());
  out.keptIdx.reserve(fullSizes.size());

  for (unsigned i = 0; i < fullSizes.size(); ++i) {
    if (IsStaticOneOFR(fullSizes[i])) continue;  // drop static 1
    out.keptIdx.push_back(i);
    out.sizes.push_back(fullSizes[i]);
    if (auto attr = fullSizes[i].dyn_cast<mlir::Attribute>()) {
      out.projected.push_back(attr.cast<mlir::IntegerAttr>().getInt());
    } else {
      out.projected.push_back(mlir::ShapedType::kDynamic);
    }
  }

  // If all dims are static 1, keep a single dim=1 (same as your original code)
  if (out.sizes.empty()) {
    out.sizes.push_back(builder.getIndexAttr(1));
    out.projected.push_back(1);
    out.keptIdx.push_back(0);  // dummy
  }
  return out;
}

// Creates a rank-reduced memref.subview from a base memref using full-rank offset/size/stride arrays.
// The resulting memref rank is determined by projectedReducedShape via inferRankReducedResultType.
mlir::Value CodeGenTileLangNPUIRDEV::CreateRankReducedSubviewFromBaseRank(
    mlir::Value base,
    llvm::ArrayRef<mlir::OpFoldResult> fullOffsets,
    llvm::ArrayRef<mlir::OpFoldResult> fullSizes,
    llvm::ArrayRef<mlir::OpFoldResult> fullStrides,
    llvm::ArrayRef<int64_t> projectedReducedShape,
    mlir::Location loc) {
  auto baseTy = base.getType().cast<mlir::MemRefType>();
  ICHECK((int64_t)fullOffsets.size() == baseTy.getRank());
  ICHECK((int64_t)fullSizes.size() == baseTy.getRank());
  ICHECK((int64_t)fullStrides.size() == baseTy.getRank());

  auto reducedTy =
      mlir::memref::SubViewOp::inferRankReducedResultType(
          projectedReducedShape, baseTy, fullOffsets, fullSizes, fullStrides)
          .cast<mlir::MemRefType>();

  return builder.create<mlir::memref::SubViewOp>(
      loc, reducedTy, base, fullOffsets, fullSizes, fullStrides);
}

// Creates a same-rank memref.subview with zero offsets and unit strides, using the provided sizes.
// Used when the UB base memref already matches the canonical copy rank.
mlir::Value CodeGenTileLangNPUIRDEV::CreateSameRankDynamicSubview(
    mlir::Value base, llvm::ArrayRef<mlir::OpFoldResult> sizesSameRank,
    mlir::Location loc) {
  auto baseTy = base.getType().cast<mlir::MemRefType>();
  int64_t r = baseTy.getRank();
  ICHECK((int64_t)sizesSameRank.size() == r);

  llvm::SmallVector<mlir::OpFoldResult> offsets(r, builder.getIndexAttr(0));
  llvm::SmallVector<mlir::OpFoldResult> strides(r, builder.getIndexAttr(1));
  return builder.create<mlir::memref::SubViewOp>(loc, base, offsets, sizesSameRank, strides);
}

llvm::SmallVector<int64_t>
CodeGenTileLangNPUIRDEV::ComputeUBAllocShapeDropStaticOnes(
    mlir::RankedTensorType dst_tensor_type_ori) {
  llvm::SmallVector<int64_t> ub_alloc_shape;
  ub_alloc_shape.reserve(dst_tensor_type_ori.getRank());

  for (int64_t d : dst_tensor_type_ori.getShape()) {
    if (d == 1) continue;  // drop static 1 as in original
    ub_alloc_shape.push_back(d);
  }
  if (ub_alloc_shape.empty()) ub_alloc_shape.push_back(1);
  return ub_alloc_shape;
}

void CodeGenTileLangNPUIRDEV::EmitCopyMemrefToTensor(
    const tvm::tl::AscendCopy& npuirop,
    mlir::Value src, mlir::Value dst,
    const SliceRange& srcR, const SliceRange& dstR,
    mlir::Location loc) {
  auto dst_tensor_type_ori = dst.getType().cast<mlir::RankedTensorType>();
  auto src_memref_type_ori = src.getType().cast<mlir::MemRefType>();

  // 1) Canonicalize copy rank: drop static-1 dims using src_sizes
  CollapsedDims srcC = CollapseStaticOneDims(srcR.sizes);
  llvm::ArrayRef<mlir::OpFoldResult> copy_sizes = srcC.sizes;
  llvm::ArrayRef<int64_t> copy_projected = srcC.projected;

  // 2) Build src_view as rank-reduced subview to copy rank
  mlir::Value src_view = CreateRankReducedSubviewFromBaseRank(
    src, srcR.offs, srcR.sizes, srcR.strides, copy_projected, loc);

  // 3) Alloc UB: drop ALL static-1 dims from dst tensor type
  llvm::SmallVector<int64_t> ub_alloc_shape =
      ComputeUBAllocShapeDropStaticOnes(dst_tensor_type_ori);

  mlir::Value base_ub = CreateStaticLocalUB(
      ub_alloc_shape, dst_tensor_type_ori.getElementType(), loc);

  // 4) Create ub_view matching copy rank
  mlir::Value ub_view;
  auto ubTy = base_ub.getType().cast<mlir::MemRefType>();

  if ((int64_t)copy_sizes.size() == ubTy.getRank()) {
    ub_view = CreateSameRankDynamicSubview(base_ub, copy_sizes, loc);
  } else {
    CollapsedDims dstC = CollapseStaticOneDims(dstR.sizes);

    llvm::SmallVector<mlir::OpFoldResult> fullOffsets(ubTy.getRank(), builder.getIndexAttr(0));
    llvm::SmallVector<mlir::OpFoldResult> fullStrides(ubTy.getRank(), builder.getIndexAttr(1));
    llvm::SmallVector<mlir::OpFoldResult> fullSizes(ubTy.getRank(), builder.getIndexAttr(1));

    if ((int64_t)dstC.keptIdx.size() == (int64_t)copy_sizes.size() &&
        (int64_t)dstC.keptIdx.size() <= ubTy.getRank()) {
      for (unsigned k = 0; k < dstC.keptIdx.size(); ++k) {
        unsigned idx = dstC.keptIdx[k];
        if (idx < (unsigned)ubTy.getRank()) fullSizes[idx] = copy_sizes[k];
      }
    } else {
      for (unsigned k = 0; k < copy_sizes.size() && k < (unsigned)ubTy.getRank(); ++k) {
        fullSizes[k] = copy_sizes[k];
      }
    }

    ub_view = CreateRankReducedSubviewFromBaseRank(
        base_ub, fullOffsets, fullSizes, fullStrides, copy_projected, loc);
  }

  // 5) Copy
  builder.create<mlir::memref::CopyOp>(loc, src_view, ub_view);

  // 6) ToTensor
  mlir::Value loaded_tensor = builder.create<mlir::bufferization::ToTensorOp>(
      loc, ub_view, /*restrict=*/true, /*writable=*/false);

  // 7) Reshape to dst_slice if needed
  // mlir::Value reshaped_tensor = MaybeReshapeTensorByDstSize(loaded_tensor, dstR.sizes);

  // 8) Type Cast
  mlir::Value casted_tensor = CreateCastIfTypeMismatch(loaded_tensor, dst);

  // 9) InsertSlice (keep your existing InsertSlice signature/behavior)
  mlir::Value result = InsertSlice(
      casted_tensor, dst,
      const_cast<llvm::SmallVector<mlir::OpFoldResult>&>(dstR.offs),
      const_cast<llvm::SmallVector<mlir::OpFoldResult>&>(dstR.sizes),
      const_cast<llvm::SmallVector<mlir::OpFoldResult>&>(dstR.strides));

  // 10) SetVarValue
  SetVarValue(npuirop.dst, result);
}

void CodeGenTileLangNPUIRDEV::EmitCopyTensorToMemref(
    const tvm::tl::AscendCopy& /*npuirop*/,
    mlir::Value src, mlir::Value dst,
    const SliceRange& srcR, const SliceRange& dstR,
    mlir::Location loc) {
  mlir::Value src_slice = builder.create<mlir::tensor::ExtractSliceOp>(
      loc, src, srcR.offs, srcR.sizes, srcR.strides);

  mlir::Value dst_view = builder.create<mlir::memref::SubViewOp>(
      loc, dst, dstR.offs, dstR.sizes, dstR.strides);

  auto src_tensor_type = src_slice.getType().cast<mlir::RankedTensorType>();
  auto dst_memref_type = dst_view.getType().cast<mlir::MemRefType>();

  mlir::Value reshaped_tensor = MaybeReshapeTensor(src_slice, dst_view);

  mlir::Value casted_tensor = CreateCastIfTypeMismatch(reshaped_tensor, dst_view);

  auto matOp = builder.create<mlir::bufferization::MaterializeInDestinationOp>(
      loc, casted_tensor, dst_view);
  matOp.setWritable(true);
}

void CodeGenTileLangNPUIRDEV::EmitCopyTensorToTensor(
    const tvm::tl::AscendCopy& npuirop,
    mlir::Value src, mlir::Value dst,
    const SliceRange& srcR, const SliceRange& dstR,
    mlir::Location loc) {
  mlir::Value src_slice = builder.create<mlir::tensor::ExtractSliceOp>(
      loc, src, srcR.offs, srcR.sizes, srcR.strides);

  auto src_tensor_type = src_slice.getType().cast<mlir::RankedTensorType>();

  mlir::Value reshaped_tensor = MaybeReshapeTensorByDstSize(src_slice, dstR.sizes);

  mlir::Value casted_tensor = CreateCastIfTypeMismatch(reshaped_tensor, dst);

  mlir::Value result = InsertSlice(
      casted_tensor, dst,
      const_cast<llvm::SmallVector<mlir::OpFoldResult>&>(dstR.offs),
      const_cast<llvm::SmallVector<mlir::OpFoldResult>&>(dstR.sizes),
      const_cast<llvm::SmallVector<mlir::OpFoldResult>&>(dstR.strides));

  SetVarValue(npuirop.dst, result);
}

/*!
 * \brief Generate MLIR for `tl.ascend_copy`, covering data movement between GM (Global Memory)
 *        and local Unified Buffer (UB), with support for rank-reduction and dynamic shapes.
 *
 * \details
 * This implementation dispatches based on the source/destination IR types (MemRef vs Tensor)
 * and lowers each case into a small, explicit MLIR sequence. The key design point is that
 * the "canonical copy shape" is derived by dropping static-1 dimensions, enabling rank-reduced
 * `memref.subview` while keeping the user-facing tensor semantics intact via reshape/cast.
 *
 * Dispatch cases:
 *
 * 1) Load (MemRef -> Tensor)
 *    - Concept:
 *        Copy from GM (MemRef) into a local UB allocation, then materialize as a Tensor.
 *    - Rank/Shape Handling:
 *        a) Canonicalize the copy rank by dropping static-1 dims from the source range.
 *        b) Create a rank-reduced `memref.subview` of the source GM matching the canonical rank.
 *        c) Allocate a static local UB memref using the destination tensor shape with static-1s removed.
 *        d) Create an UB `memref.subview` matching the canonical copy rank.
 *        e) `memref.copy` from the GM view to the UB view.
 *        f) Convert UB to tensor via `bufferization.to_tensor`.
 *        g) Reshape and/or cast to match the destination slice type, then `tensor.insert_slice`
 *           into the destination tensor and update the corresponding var binding.
 *
 * 2) Store (Tensor -> MemRef)
 *    - Concept:
 *        Store a tensor slice back to GM.
 *    - Logic:
 *        a) Extract a slice from the source tensor (`tensor.extract_slice`).
 *        b) Create a destination GM subview (`memref.subview`).
 *        c) Resolve potential rank/shape mismatches via reshape, and type mismatches via cast.
 *        d) Write using `bufferization.materialize_in_destination` (writable).
 *
 * 3) Copy (Tensor -> Tensor)
 *    - Concept:
 *        Tensor-to-tensor movement / layout manipulation through slice extraction and insertion.
 *    - Logic:
 *        a) Extract slices from source and destination tensors.
 *        b) Reshape/cast the source slice to match the destination slice type.
 *        c) Insert into destination using `tensor.insert_slice` and update the var binding.
 *
 * Example lowering sketch (Load):
 *   Input:  tl.ascend_copy(src_gm[...], dst_tensor[...])
 *   Output:
 *     %src_view = memref.subview %src_gm [...]  : (rank-reduced)
 *     %ub       = memref.alloc()               : (local UB)
 *     memref.copy %src_view, %ub
 *     %val      = bufferization.to_tensor %ub
 *     %result   = tensor.insert_slice %val into %dst_tensor [...]
 */
void CodeGenTileLangNPUIRDEV::AscendCopyCodegen(const CallNode* op) {
  tvm::tl::AscendCopy npuirop(op->args, this->vmap);

  mlir::Value src = GetVarValue(npuirop.src);
  mlir::Value dst = GetVarValue(npuirop.dst);

  SliceRange srcR = MakeSliceRange(npuirop.src_range);
  SliceRange dstR = MakeSliceRange(npuirop.dst_range);

  mlir::Location loc = builder.getUnknownLoc();

  const bool src_is_memref = src.getType().isa<mlir::MemRefType>();
  const bool dst_is_memref = dst.getType().isa<mlir::MemRefType>();
  const bool src_is_tensor = src.getType().isa<mlir::TensorType>();
  const bool dst_is_tensor = dst.getType().isa<mlir::TensorType>();

  if (src_is_memref && dst_is_tensor) {
    EmitCopyMemrefToTensor(npuirop, src, dst, srcR, dstR, loc);
    return;
  }
  if (src_is_tensor && dst_is_memref) {
    EmitCopyTensorToMemref(npuirop, src, dst, srcR, dstR, loc);
    return;
  }
  if (src_is_tensor && dst_is_tensor) {
    EmitCopyTensorToTensor(npuirop, src, dst, srcR, dstR, loc);
    return;
  }
  if (src_is_memref && dst_is_memref) {
    ICHECK(false) << "Unsupported memref to memref copy yet.";
    return;
  }

  ICHECK(false) << "Unsupported copy dispatch state";
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
mlir::Value CodeGenTileLangNPUIRDEV::BinaryOpCodegen(const PrimExprNode *op,
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

/// Generate hivm.hir.vexp for tl.npuir_exp
/// before:
///     T.npuir_exp(A, B)
/// after:
///     %.* = hivm.hir.vexp ins(A) outs(B) -> tensor<>
template <typename T, typename U>
void CodeGenTileLangNPUIRDEV::UnaryVecOpCodegen(const CallNode *op) {
  T npuirop(op->args, this->vmap);
  auto in_data_name = GetVarValue(npuirop.src);
  auto out_data_name = GetVarValue(npuirop.dst);
  auto dims = getBroadcastDim(npuirop.src->shape, npuirop.dst->shape);
  mlir::Type dst_type = out_data_name.getType();
  mlir::TypeRange result_tensors(&dst_type, 1);
  // Create HIVM Op
  auto newOp = builder.create<U>(
      builder.getUnknownLoc(), 
      result_tensors,            // result type
      in_data_name,              // in
      out_data_name,             // out
      builder.getDenseI64ArrayAttr({}),           // transpose
      builder.getDenseI64ArrayAttr(dims)          // broadcast
  );
  SetVarValue(npuirop.dst, newOp->getResult(0));
}

void CodeGenTileLangNPUIRDEV::BarrierCodegen(const CallNode *op) {
  tvm::tl::NpuirPipeBarrier npuirop(op->args, this->vmap);
  mlir::hivm::PipeAttr pipAttrType = mlir::hivm::PipeAttr::get(
      builder.getContext(), PIPE_MAP[npuirop.pipe_type]);
  builder.create<mlir::hivm::PipeBarrierOp>(builder.getUnknownLoc(),
                                             pipAttrType);
}

mlir::Value CodeGenTileLangNPUIRDEV::NeedGenInsertSlice(
    Buffer buffer_data,
    Array<Range> range,
    mlir::Value src) { 

  Array<PrimExpr> region_shape, region_indices;
  for (Range r : range) {
    region_shape.push_back(r.get()->extent);
    region_indices.push_back(r.get()->min);
  }

  if (IsEqual(buffer_data->shape, region_shape) && AllZero(region_indices)) {
    return GetVarValue(buffer_data);
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

  auto srcTensorTy = src.getType().cast<mlir::TensorType>();
  auto dstTensorTy = GetVarValue(buffer_data).getType().cast<mlir::TensorType>();
  auto elemTy = dstTensorTy.getElementType();
  auto srcShape = srcTensorTy.getShape();
  
  auto emptyTensor = builder.create<mlir::tensor::EmptyOp>(
      builder.getUnknownLoc(),
      srcShape,
      elemTy);

  return emptyTensor.getResult();
}

// Convert TVM Range to MLIR OpFoldResult arrays
std::tuple<SmallVector<mlir::OpFoldResult>, 
           SmallVector<mlir::OpFoldResult>, 
           SmallVector<mlir::OpFoldResult>> 
CodeGenTileLangNPUIRDEV::CreateOpFoldResultArray(const Array<Range>& range) {
  // TODO: support dynamic shape
  SmallVector<mlir::OpFoldResult> offsets;
  SmallVector<mlir::OpFoldResult> sizes;
  SmallVector<mlir::OpFoldResult> strides;
  for (const auto& r : range) {
    // offset
    if (auto offset_int = as_const_int(r->min)) {
      offsets.push_back(builder.getI64IntegerAttr(*offset_int));
    } else {
      mlir::Value offsetVal = CreateIndexCastOp(MakeValue(r->min));
      offsets.push_back(offsetVal);
    }
    // size
    if (auto size_int = as_const_int(r->extent)) {
      sizes.push_back(builder.getI64IntegerAttr(*size_int));
    } else {
      mlir::Value sizeVal = CreateIndexCastOp(MakeValue(r->extent));
      sizes.push_back(sizeVal);
    }
    // stride (usually 1)
    strides.push_back(builder.getI64IntegerAttr(1));
  }
  return {offsets, sizes, strides};
}

mlir::Value CodeGenTileLangNPUIRDEV::ReshapeCastAndInsertSlice(
    mlir::Value tensor, 
    mlir::Value dst,   
    Array<Range> dst_range) 
{
    auto [offsets, sizes, strides] = CreateOpFoldResultArray(dst_range);

    mlir::Value reshaped = MaybeReshapeTensorByDstSize(tensor, sizes);

    mlir::Value casted = CreateCastIfTypeMismatch(reshaped, dst);

    mlir::Value result = InsertSlice(
      casted, dst,
      const_cast<llvm::SmallVector<mlir::OpFoldResult>&>(offsets),
      const_cast<llvm::SmallVector<mlir::OpFoldResult>&>(sizes),
      const_cast<llvm::SmallVector<mlir::OpFoldResult>&>(strides));

    return result;
}


void CodeGenTileLangNPUIRDEV::VselectCodegen(const CallNode *op) {
  /// Generate hivm.hir.vsel for tl.npuir_select.
  /// before:
  ///   T.npuir_select(Cond_VEC, A_VEC, B_VEC, C_VEC)
  /// after:
  ///  Partial Insertion:
  ///  %8 = tensor.empty() : tensor<32xf16>
  ///  %9 = hivm.hir.vsel ins(%Cond_VEC, %A_VEC, %B_VEC : tensor<32xi1>, tensor<32xf16>, tensor<32xf16>) outs(%8 : tensor<32xf16>) -> tensor<32xf16>
  ///  %c1 = arith.constant 1 : index
  ///  %c32 = arith.constant 32 : index
  ///  %from_elements = tensor.from_elements %c1, %c32 : tensor<2xindex>
  ///  %reshape = tensor.reshape %9(%from_elements) : (tensor<32xf16>, tensor<2xindex>) -> tensor<1x32xf16>
  ///  %inserted_slice = tensor.insert_slice %reshape into %C_VEC[%7, 0] [1, 32] [1, 1] : tensor<1x32xf16> into tensor<8x32xf16>
  ///  
  ///  Original/full shape:
  ///  %result = hivm.hir.vsel ins(%Cond_VEC, %A_VEC, %B_VEC : tensor<32xi1>, tensor<32xf16>, tensor<32xf16>) outs(%9 : tensor<32xf16>) -> tensor<32xf16>

  tvm::tl::NpuirSelect npuirop(op->args, this->vmap);

  mlir::Value cond_data_name = GetVarValue(npuirop.cond);
  mlir::Value src0_data_name = GetVarValue(npuirop.src0);
  mlir::Value src1_data_name = GetVarValue(npuirop.src1);
  mlir::Value dst_data_name = GetVarValue(npuirop.dst);

  if (!dst_data_name.getType().isa<mlir::TensorType>()) {
    return;
  }

  auto srcTensorTy = src0_data_name.getType().cast<mlir::TensorType>();
  auto srcShape = srcTensorTy.getShape();

  mlir::Value insertBase = NeedGenInsertSlice(npuirop.dst, npuirop.dst_range, src0_data_name);
  bool needInsertSlice = (insertBase != GetVarValue(npuirop.dst));

  std::vector<int64_t> srcShapeVec(srcShape.begin(), srcShape.end());
  auto broadcastDim = getBroadcastDim(npuirop.src0->shape, srcShapeVec);

  mlir::Value selOutput;

  auto selOp = builder.create<mlir::hivm::VSelOp>(
      builder.getUnknownLoc(),
      mlir::TypeRange{insertBase.getType()},
      mlir::ValueRange{cond_data_name, src0_data_name, src1_data_name},
      mlir::ValueRange{insertBase},
      mlir::Value());
  
  selOp->setAttr("broadcast", builder.getDenseI64ArrayAttr(broadcastDim));
  selOutput = selOp.getResult()[0];
  
  mlir::Value result = needInsertSlice
    ? ReshapeCastAndInsertSlice(selOutput, dst_data_name, npuirop.dst_range)
    : selOutput;

  SetVarValue(npuirop.dst, result);
}

/// Generate hivm.hir.vbrc for tl.npuir_brc.
/// before:
///    T.npuir_brc(A, B)
/// after:
///    %.* = hivm.hir.vbrc ins(A) outs(B) -> tensor<>
void CodeGenTileLangNPUIRDEV::VbrcCodegen(const CallNode *op) {
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
    src = GetVarValue(npuirop.src);
    auto srcTensor = llvm::dyn_cast<TypedValue<RankedTensorType>>(src);
    inBufferShape = srcTensor.getType().getShape();
  }
  Value dst = GetVarValue(npuirop.dst);
  auto broadcastDimAttr = builder.getDenseI64ArrayAttr({});
  if (!inBufferShape.empty()) {
    auto outTensor = llvm::dyn_cast<TypedValue<RankedTensorType>>(dst);
    auto outBufferShape = outTensor.getType().getShape();
    auto broadcastDim = getBroadcastDim(npuirop.src->shape, npuirop.dst->shape);
    broadcastDimAttr = builder.getDenseI64ArrayAttr(broadcastDim);
  }
  mlir::Type dst_type = dst.getType();
  mlir::TypeRange result_tensors(&dst_type, 1);
  auto newCastOp = builder.create<mlir::hivm::VBrcOp>(builder.getUnknownLoc(), result_tensors,
                                      src, dst, broadcastDimAttr);
  SetVarValue(npuirop.dst, newCastOp->getResult(0));
}

/// Generate hivm.hir.vcast for tl.npuir_cast.
/// before:
///    T.npuir_cast(A, B, "rint")
/// after:
///    %.* = hivm.hir.vcast ins(A) outs(B) -> tensor<>
void CodeGenTileLangNPUIRDEV::VcastCodegen(const CallNode *op) {
  tvm::tl::NpuirCast npuirop(op->args, this->vmap);
  Value src = GetVarValue(npuirop.src);
  Value dst = GetVarValue(npuirop.dst);
  auto round_mode = npuirop.round_mode;
  mlir::hivm::RoundMode mode = NPUIR_STR_ROUNDMODE[round_mode];
  auto broadcastDim = getBroadcastDim(npuirop.src->shape, npuirop.dst->shape);
  auto broadcastDimAttr = builder.getDenseI64ArrayAttr(broadcastDim);
  mlir::Type dst_type = dst.getType();
  mlir::TypeRange result_tensors(&dst_type, 1);
  auto newCastOp = builder.create<mlir::hivm::VCastOp>(
      builder.getUnknownLoc(), result_tensors, src, dst,
      mlir::hivm::RoundModeAttr::get(&context, mode), nullptr,
      broadcastDimAttr);
  SetVarValue(npuirop.dst, newCastOp->getResult(0));
}

/// Generate hivm.hir.vreduce for tl.npuir_reduce.
/// before:
///    T.npuir_reduce(A, B, "rint")
/// after:
///    %.* = hivm.hir.vreduce ins(A) outs(B) -> tensor<>
void CodeGenTileLangNPUIRDEV::VreduceCodegen(const CallNode *op) {
  tvm::tl::NpuirReduce npuirop(op->args, this->vmap);
  Value src = GenExtractSliceFromRegion(npuirop.src, npuirop.src_range);
  Value dst = GetVarValue(npuirop.dst);
  auto reduce_mode = npuirop.reduce_mode;
  mlir::hivm::ReduceOpAttr mode =
      mlir::hivm::ReduceOpAttr::get(&context, NPUIR_STR_REDUCEOP[reduce_mode]);
  mlir::Type dst_type = dst.getType();
  mlir::TypeRange result_tensors(&dst_type, 1);
  auto reduceOp = builder.create<mlir::hivm::VReduceOp>(
      builder.getUnknownLoc(), result_tensors, src, dst, mode,
      builder.getDenseI64ArrayAttr(npuirop.reduce_dims));
  SetVarValue(npuirop.dst, reduceOp->getResult(0));
}

void CodeGenTileLangNPUIRDEV::VcumsumCodegen(const CallNode *op) {
  /// Generate hivm.hir.cumsum for tl.npuir_cumsum.
  /// before:
  ///   T.npuir_cumsum(src, dst, dim, reverse)
  /// after:
  ///   %.* = hivm.hir.vcumsum ins(src) outs(dst) cum_dims = [0] -> tensor<> for reverse = false
  tvm::tl::NpuirCumsum npuirop(op->args, this->vmap);
  mlir::Location loc = builder.getUnknownLoc();
  Value src = GetVarValue(npuirop.src);
  Value dst = GetVarValue(npuirop.dst);
  mlir::Type dst_type = dst.getType();
  mlir::TypeRange result_tensors(&dst_type, 1);
  auto reverse_mode = npuirop.reverse;
  if(reverse_mode == true){
    ICHECK(false) <<"reverse=True is not yet supported\n";
    return;
  }
  auto newCumsumOp = builder.create<mlir::hivm::VCumsumOp>(
      loc, result_tensors, src, dst,
      builder.getDenseI64ArrayAttr(npuirop.cum_dims));
  SetVarValue(npuirop.dst, newCumsumOp->getResult(0));
}

void CodeGenTileLangNPUIRDEV::VAtomicAddCodegen(const CallNode *op) {
  /// Generate hivm.hir.store for tl.npuir_atomic_add.
  /// before:
  ///   T.npuir_atomic_add(src, dst, size)
  /// after:
  ///   hivm.hir.store ins(src) outs(dst) atomic = <add>
  tvm::tl::NpuirAtomicAdd npuirop(op->args, this->vmap);
  Value src = GetVarValue(npuirop.src);
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

void CodeGenTileLangNPUIRDEV::VgatherCodegen(const CallNode *op) {
  tvm::tl::NpuirGather npuirop(op->args, this->vmap);
  Value src = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  Value dst = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  Value indices = GenSubviewFromRegion(npuirop.indices, npuirop.indices_range);

  builder.create<mlir::hivm::VGatherOp>(builder.getUnknownLoc(), TypeRange{},
                                        src, indices, dst);
}

void CodeGenTileLangNPUIRDEV::VtransposeCodegen(const CallNode *op) {
  tvm::tl::NpuirTranspose npuirop(op->args, this->vmap);
  Value src = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  Value dst = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  auto permutation = builder.getDenseI64ArrayAttr(npuirop.permutation);
  builder.create<mlir::hivm::VTransposeOp>(builder.getUnknownLoc(), TypeRange{},
                                           src, dst, permutation);
}

void CodeGenTileLangNPUIRDEV::VinterleaveCodegen(const CallNode *op) {
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

void CodeGenTileLangNPUIRDEV::VdeinterleaveCodegen(const CallNode *op) {
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

void CodeGenTileLangNPUIRDEV::VarangeCodegen(const CallNode *op) {
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

void CodeGenTileLangNPUIRDEV::VconcatCodegen(const CallNode *op) {
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

void CodeGenTileLangNPUIRDEV::VpadCodegen(const CallNode *op) {
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

void CodeGenTileLangNPUIRDEV::VflipCodegen(const CallNode *op) {
  tvm::tl::NpuirFlip npuirop(op->args, this->vmap);
  Value src = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  Value dst = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);
  builder.create<mlir::hivm::VFlipOp>(builder.getUnknownLoc(), TypeRange{}, src,
                                      dst);
}

void CodeGenTileLangNPUIRDEV::Nd2NzCodegen(const CallNode *op) {
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

void CodeGenTileLangNPUIRDEV::Nz2NdCodegen(const CallNode *op) {
  // Generate hivm.hir.nz2nd for tl.npuir_store_nz2nd.
  tvm::tl::NpuirNz2nd npuirop(op->args, this->vmap);
  // gen memref.subview
  mlir::Value src = GenSubviewFromRegion(npuirop.src, npuirop.src_range);
  mlir::Value dst = GenSubviewFromRegion(npuirop.dst, npuirop.dst_range);

  // gen hivm.hir.nz2nd
  builder.create<mlir::hivm::NZ2NDOp>(builder.getUnknownLoc(),
                                      mlir::TypeRange{}, src, dst);
}

void CodeGenTileLangNPUIRDEV::FixpipeCodegen(const CallNode *op) {
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

/// Generate hivm.hir.mmadL1 for tl.npuir_dot.
/// before:
///   T.npuir_dot(T.region(A_BUF[0, 0], 1, 128, 1024),
///               T.region(B_BUF[0, 0], 1, 1024, 256),
///               T.region(C_BUF[0, 0], 3, 128, 256), T.bool(True))
/// after:
///   %.* = hivm.hir.mmadL1 ins(%.*,  %.*,  %.*,  %.*,  %.*,  %.* :
///                             tensor<128x64xf16>, tensor<64x64xf16>,
///                             i1,  index,  index,  index)
///                         outs(%.* : tensor<128x64xf32>)
///                         ->  tensor<128x64xf32>
void CodeGenTileLangNPUIRDEV::DotCodegen(const CallNode *op) {
  tvm::tl::NpuirDot npuirop(op->args, this->vmap);
  Array<PrimExpr> a_region_shape, b_region_shape;
  for (int i = 0; i < npuirop.src0_range.size(); i++) {
    a_region_shape.push_back(npuirop.src0_range[i].get()->extent);
    b_region_shape.push_back(npuirop.src1_range[i].get()->extent);
  }

  mlir::Location unknown_loc = builder.getUnknownLoc();
  mlir::IndexType idx_ty = builder.getIndexType();
  mlir::Value a = GetVarValue(npuirop.src0);
  mlir::Value b = GetVarValue(npuirop.src1);
  mlir::Value c = GetVarValue(npuirop.dst);
  mlir::Type c_type = c.getType();
  mlir::TypeRange result_tensors(&c_type, 1);
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
  auto newMmadL1Op = builder.create<mlir::hivm::MmadL1Op>(
      unknown_loc, result_tensors, a, b, init_condition, real_m, real_k, real_n,
      c, per_channel_bias, a_transpose, b_transpose, enable_HF32);
  // mmadl1 has only one output, so use getResult(0)
  mlir::Value newMmadL1OpValue = newMmadL1Op->getResult(0);
  SetVarValue(npuirop.dst, newMmadL1OpValue);
}

/// Generate hivm.hir.vadd for tl.npuir_add.
/// Generate hivm.hir.vcmp for tl.npuir_cmp.
/// Generate hivm.hir.vdiv for tl.npuir_div.
/// Generate hivm.hir.vmul for tl.npuir_vmul
/// Generate hivm.hir.vsub for tl.npuir_sub
/// Generate hivm.hir.vmax for tl.npuir_max
/// Generate hivm.hir.vmin for tl.npuir_min
/// Generate hivm.hir.vor for tl.npuir_or
/// Generate hivm.hir.vand for tl.npuir_and
/// Generate hivm.hir.vxor for tl.npuir_xor
/// Generate hivm.hir.vpow for tl.npuir_pow
/// Generate hivm.hir.vshl for tl.npuir_shl
/// Generate hivm.hir.vshr for tl.npuir_shr
template <typename T>
void CodeGenTileLangNPUIRDEV::CreateHIVMBinaryVectorOp(const CallNode *op) {
  auto processImm = [&](mlir::Value &src, int arg_id,
                        Array<PrimExpr> &buffer_shape) {
    if (op->args[arg_id].as<IntImm>() || op->args[arg_id].as<FloatImm>() || 
        op->args[arg_id].as<tir::VarNode>() || op->args[arg_id].as<tir::BufferLoadNode>()) {
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
      auto buffer_node = region_node->args[0].as<BufferLoadNode>();
      Array<PrimExpr> tmp_buffer_shape = buffer_node->buffer->shape;
      bool is_scalar_load = true;
      for (int i = 0; i < tmp_buffer_shape.size(); i++) {
        const IntImmNode* int_imm = region_node->args[2 + i].as<IntImmNode>();
        if (!int_imm || int_imm->value != 1) {
          is_scalar_load = false;
          break;
        }
      }
      const IntImmNode* int_imm = region_node->args[2].as<IntImmNode>();
      // If load only one element, do not use memref.subview, use memref.load as a scalar
      if(is_scalar_load) {
        src = VisitExpr_(buffer_node);
      } else {
        src = GetVarValue(region_node);
        buffer_shape = tmp_buffer_shape;
      }
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
  mlir::Value dst = GetVarValue(region_node_dst);

  tvm::tl::RegionOp region_dst_tmp(region_node_dst->args, vmap);
  Array<Range> dst_range = region_dst_tmp.GetRanges();


  auto srcTensorTy = src0.getType().cast<mlir::TensorType>();
  auto srcShape = srcTensorTy.getShape();

  mlir::Value insertBase = NeedGenInsertSlice(region_dst_tmp.GetBuffer(), dst_range, src0);
  bool needInsertSlice = (insertBase != GetVarValue(region_node_dst));

  // transpose
  mlir::DenseI64ArrayAttr transpose = builder.getDenseI64ArrayAttr({});
  // broadcast
  llvm::SmallVector<int64_t> dims =
      getBroadcastDim(buffer_shape0, buffer_shape1);
  mlir::DenseI64ArrayAttr broadcast = builder.getDenseI64ArrayAttr(dims);
  // typerange
  mlir::Type dst_type = dst.getType();
  mlir::TypeRange result_tensors(&dst_type, 1);
  // Create hivm::op
  auto loc = builder.getUnknownLoc();
  mlir::Value newOpValue;

  llvm::SmallVector<int64_t> elemShape(srcShape.begin(), srcShape.end());

  if constexpr (std::is_same_v<T, mlir::hivm::VCmpOp>) {
    mlir::hivm::CompareMode mode =
          COMPARE_MODE[op->args[3].as<StringImm>().value()->value];
    auto cmp_attr =
        mlir::hivm::CompareModeAttr::get(builder.getContext(), mode);
    auto newOp = builder.create<T>(loc, insertBase.getType(), mlir::ValueRange{src0, src1},
        mlir::ValueRange{insertBase}, cmp_attr, transpose, broadcast);
    newOpValue = newOp->getResult(0);
  } else if constexpr (std::is_same_v<T, mlir::hivm::VPowOp>) {
    auto newOp = builder.create<T>(loc, insertBase.getType(), mlir::ValueRange{src0, src1},
        mlir::ValueRange{insertBase}, mlir::Value(), transpose, broadcast);
    newOpValue = newOp->getResult(0);
  } else if constexpr (std::is_same_v<T, mlir::hivm::VShROp>) {
    auto round_attr = mlir::BoolAttr::get(
        builder.getContext(), op->args[3].as<Bool>().value());
    auto newOp = builder.create<T>(loc, insertBase.getType(), mlir::ValueRange{src0, src1},
        mlir::ValueRange{insertBase}, round_attr, transpose, broadcast);
    newOpValue = newOp->getResult(0);
  } else {
    auto newOp = builder.create<T>(loc, insertBase.getType(), mlir::ValueRange{src0, src1},
        mlir::ValueRange{insertBase}, transpose, broadcast);
    newOpValue = newOp->getResult(0);
  }

  mlir::Value result = needInsertSlice
    ? ReshapeCastAndInsertSlice(newOpValue, dst, dst_range)
    : newOpValue;

  SetVarValue(region_node_dst, result);
}

void CodeGenTileLangNPUIRDEV::BitcastCodegen(const CallNode *op) {
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
void CodeGenTileLangNPUIRDEV::SyncBlockCodegen(const T &sync_op) {
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

mlir::Value CodeGenTileLangNPUIRDEV::GetEventID(PrimExpr id) {
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
void CodeGenTileLangNPUIRDEV::PipeFlagCodegen(const CallNode *op) {
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

void CodeGenTileLangNPUIRDEV::DebugPrintCodegen(const CallNode *op) {
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

/// Generate hivm.hir.vreduce for tl.npuir_reshape.
/// before:
///    T.npuir_reshape(A, B)
/// after:
///    %.* = tensor.reshape %a(%b) outs(%c) -> tensor<>
void CodeGenTileLangNPUIRDEV::ReshapeCodegen(const CallNode *op) {
  tvm::tl::NpuirReshape npuirop(op->args, this->vmap);
  Value src = GetVarValue(npuirop.src);
  const auto &dstShape = npuirop.dst_shape;
  auto shapeTensorType =
      mlir::RankedTensorType::get({dstShape.size()}, builder.getIndexType());
  auto shapeAttr =
      mlir::DenseIntElementsAttr::get(shapeTensorType, dstShape);
  Value shapeTensor =
      builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), shapeAttr);

  auto srcTensorTy = src.getType().cast<mlir::RankedTensorType>();
  auto resultTensorTy =
      mlir::RankedTensorType::get(dstShape,
                                  srcTensorTy.getElementType());
  Value reshaped =
      builder.create<mlir::tensor::ReshapeOp>(
          builder.getUnknownLoc(),
          resultTensorTy,
          src,
          shapeTensor);

  SetVarValue(npuirop.dst, reshaped);
}

void CodeGenTileLangNPUIRDEV::CallExternCodegen(const CallNode *op) {
  // Todo: Implementation pending
}

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const CallNode *op) {
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
  } else if (op->op.same_as(Op::Get("tl.npuir_div"))) {
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
  } else if (op->op.same_as(Op::Get("tl.npuir_atomic_add"))) {
    VAtomicAddCodegen(op);
  } else if (op->op.same_as(Op::Get("tl.npuir_cumsum"))) {
    VcumsumCodegen(op);
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
} else if (op->op.same_as(Op::Get("tl.npuir_reshape"))) {
    ReshapeCodegen(op);
  } else {
    VisitExpr_(op);
  }
  return mlir::Value();
}

void CodeGenTileLangNPUIRDEV::VisitStmt_(const LetStmtNode *op) {

  // EmitDebugLocation(op);
  const VarNode *v = op->var.get();
  ICHECK(GetVarValue(v) == mlir::Value{});
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
  }

  SetVarValue(v, value);
  VisitStmt(op->body);
}

void CodeGenTileLangNPUIRDEV::VisitStmt_(const AttrStmtNode *op) {
  if (op->attr_key == "thread_extent") {
    IterVar iv = Downcast<IterVar>(op->node);
    if (iv->thread_tag == "blockIdx.x" && iv->var->name_hint != "_") {
      mlir::Value indexOp = GetAndCastIndexOp<mlir::hivm::GetBlockIdxOp>(iv);
      SetVarValue(iv->var.get(), indexOp);
    } else if (iv->thread_tag == "blockIdx.y" && iv->var->name_hint != "_") {
      mlir::Value indexOp = GetAndCastIndexOp<mlir::hivm::GetSubBlockIdxOp>(iv);
      SetVarValue(iv->var.get(), indexOp);
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
mlir::Value CodeGenTileLangNPUIRDEV::GetAndCastIndexOp(const IterVar iv) {
  auto indexOp = builder.create<T>(mlir::UnknownLoc::get(&context));
  auto truncOp = builder.create<mlir::arith::TruncIOp>(
      mlir::UnknownLoc::get(&context),
      builder.getI32Type(), // The target integer type
      indexOp               // The source float value to cast
  );
  return truncOp;
}

/// Generate tensor.empty() for tl.alloc_shared and tl.alloc_fragment
/// Generate tensor.empty() for TIR AllocateNode like T.decl_buffer.
/// before:
///      A_VEC = T.decl_buffer((128, 256), "float16", scope="shared")
/// after:
///      %A_VEC = tensor.empty() : tensor<128x256xf16>
void CodeGenTileLangNPUIRDEV::VisitStmt_(const AllocateNode *op) {
  ICHECK(!is_zero(op->condition));
  std::string scope = GetPtrStorageScope(op->buffer_var);
  std::map<std::string, NPU_CORETYPE> scope_coretype_map{
      {"shared", NPU_CORETYPE::AIV},
      {"shared.cube", NPU_CORETYPE::AIC},
      {"wmma.accumulator", NPU_CORETYPE::AIC}};
  if (scope_coretype_map.count(scope) == 0) {
    std::vector<long int> shape = GetShape(op->extents);

    auto tensorEmptyOp = builder.create<mlir::tensor::EmptyOp>(
        builder.getUnknownLoc(), shape, DTypetoMLIRType(op->dtype));

    // Update var_map_ with the new variable
    ICHECK(GetVarValue(op->buffer_var.get()) == mlir::Value{});
    SetVarValue(op->buffer_var.get(), tensorEmptyOp.getResult());
  }
  else if (scope_coretype_map[scope] == this->current_coretype) {
    std::vector<long int> shape = GetShape(op->extents);

    auto tensorEmptyOp = builder.create<mlir::tensor::EmptyOp>(
        builder.getUnknownLoc(), shape, DTypetoMLIRType(op->dtype));

    // Update var_map_ with the new variable
    ICHECK(GetVarValue(op->buffer_var.get()) == mlir::Value{});
    SetVarValue(op->buffer_var.get(), tensorEmptyOp.getResult());
  }
  this->VisitStmt(op->body);
}

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const MinNode *op) {
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

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const MaxNode *op) {
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

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const AddNode *op) {
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

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const SubNode *op) {
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
CodeGenTileLangNPUIRDEV::VisitExpr_(const FloatImmNode *op) {
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

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const IntImmNode *op) {
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

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const MulNode *op) {
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

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const AndNode *op) {
  CHECK(op->a.dtype().is_int() || op->a.dtype().is_uint());
  CHECK(op->b.dtype().is_int() || op->b.dtype().is_uint());
  auto lhs = MakeValue(op->a);
  auto rhs = MakeValue(op->b);
  auto mlirVal =
      BinaryOpCodegen<mlir::arith::AndIOp, std::nullptr_t>(
          op, nullptr, lhs, rhs);
  return mlirVal;
}

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const OrNode *op) {
  CHECK(op->a.dtype().is_int() || op->a.dtype().is_uint());
  CHECK(op->b.dtype().is_int() || op->b.dtype().is_uint());
  auto lhs = MakeValue(op->a);
  auto rhs = MakeValue(op->b);
  auto mlirVal =
      BinaryOpCodegen<mlir::arith::OrIOp, std::nullptr_t>(
          op, nullptr, lhs, rhs);
  return mlirVal;
}

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const DivNode *op) {
  auto lhs = MakeValue(op->a);
  auto rhs = MakeValue(op->b);
  auto mlirVal =
      BinaryOpCodegen<mlir::arith::DivFOp, std::nullptr_t>(
          op, nullptr, lhs, rhs);
  return mlirVal;
}

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const SelectNode *op) {
  auto condition = MakeValue(op->condition);
  auto true_value = MakeValue(op->true_value);
  auto false_value = MakeValue(op->false_value);

  return builder.create<mlir::arith::SelectOp>(
      builder.getUnknownLoc(), condition, true_value, false_value);
}

String CodeGenTileLangNPUIRDEV::GetCurrentFunctionName(){
  return this->current_function_name;
}

void CodeGenTileLangNPUIRDEV::AddFunctionForCoreType(const GlobalVar &gvar,
                                                     const PrimFunc &f) {
  // clear previous generated state.
  this->InitFuncState();

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";
  this->current_function_name = static_cast<std::string>(global_symbol.value());
  if (this->func_coretype == NPU_CORETYPE::MIX && this->current_coretype != NPU_CORETYPE::MIX) {
    this->current_function_name = this->current_function_name + "_mix_" + NPU_CORETYPE_STR[this->current_coretype];
  }
  else {
    this->current_function_name = this->current_function_name;
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
    SetVarValue(real_v.get(), funcOp.getArgument(i + funcArgsOffset));
  }
  builder.create<hivm::SetFFTSBaseAddrOp>(builder.getUnknownLoc(),
                                           funcOp.getArgument(0));
  for (auto recastInfo : recastNeedInsert) {
    tir::Var v = f->params[recastInfo.first];
    tir::Var real_v = f->buffer_map[v]->data;
    Array<PrimExpr> shape = f->buffer_map[v]->shape;
    auto memrefType = llvm::dyn_cast<MemRefType>(recastInfo.second);
    auto strideLayout =
        llvm::dyn_cast<StridedLayoutAttr>(memrefType.getLayout());
    SmallVector<OpFoldResult> shape_val;
    for (PrimExpr s : shape) {
      if (auto s_int = as_const_int(s)) {
        shape_val.push_back(builder.getI64IntegerAttr(*s_int));
      } else {
        mlir::Value s_index = CreateIndexCastOp(MakeValue(s));
        shape_val.push_back(s_index);
      }
    }
    size_t dim = shape.size();
    SmallVector<OpFoldResult> stride_val(dim);
    tvm::PrimExpr tmp_stride = IntImm(shape[0].dtype(), 1);
    for (int i = dim - 1; i >= 0; i--) {
      mlir::Value s_index = CreateIndexCastOp(MakeValue(tmp_stride));
      stride_val[i] = s_index;
      tmp_stride = Mul(shape[i], tmp_stride);
    }
    OpFoldResult offset = builder.getI64IntegerAttr(strideLayout.getOffset());
    auto recastOp = builder.create<memref::ReinterpretCastOp>(
        builder.getUnknownLoc(), memrefType, GetVarValue(real_v.get()),
        offset, shape_val, stride_val);
    SetVarValue(real_v.get(), recastOp);
  }
  mlir::hacc::KernelArgTypeAttr accArgAttr = hacc::KernelArgTypeAttr::get(
      builder.getContext(), hacc::KernelArgType::kFFTSBaseAddr);
  funcOp.setArgAttr(0, "hacc.arg_type", accArgAttr);
  mlir::hacc::KernelArgTypeAttr syncArgAttr = hacc::KernelArgTypeAttr::get(
      builder.getContext(), hacc::KernelArgType::kSyncBlockLock);
  funcOp.setArgAttr(1, "hacc.arg_type", syncArgAttr);
  mlir::hacc::KernelArgTypeAttr workspaceArgAttr = hacc::KernelArgTypeAttr::get(
      builder.getContext(), hacc::KernelArgType::kWorkspace);
  funcOp.setArgAttr(2, "hacc.arg_type", workspaceArgAttr);
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

void CodeGenTileLangNPUIRDEV::InitFuncState() {
  var_map_.clear();
  AddVarLayer();
  alias_var_set_.clear();
  analyzer_.reset(new arith::Analyzer());
  prim_expr_map.clear();
  mlir_value_map.clear();
  this->current_function_name = "";
}

void CodeGenTileLangNPUIRDEV::AddFunction(const GlobalVar& gvar, const PrimFunc& f)
{
    InferFuncCoreType infer;
    infer.VisitStmt(f->body);
    if (!infer.hasExpert) {
        if (infer.hasVector && infer.hasCube) {
            infer.func_coretype = NPU_CORETYPE::MIX;
        }
        if (infer.hasVector && !infer.hasCube) {
            infer.func_coretype = NPU_CORETYPE::AIV;
        }
        if (!infer.hasVector && infer.hasCube) {
            infer.func_coretype = NPU_CORETYPE::AIC;
        }
    }

    this->func_coretype = infer.func_coretype;  // NPU_CORETYPE::MIX;

    auto moduleCoreType =
        mlir::hivm::TModuleCoreTypeAttr::get(&this->context, NPUIR_MODULECORETYPE_STR[this->func_coretype]);
    this->module->getOperation()->setAttr(mlir::hivm::TModuleCoreTypeAttr::name, moduleCoreType);

    this->module->getOperation()->setAttr("memref.memref_as_ptr",
                                        UnitAttr::get(builder.getContext()));

    switch (this->func_coretype) {
        case NPU_CORETYPE::AIC:
            this->current_coretype = NPU_CORETYPE::AIC;
            AddFunctionForCoreType(gvar, f);
            break;

        case NPU_CORETYPE::AIV:
            this->current_coretype = NPU_CORETYPE::AIV;
            AddFunctionForCoreType(gvar, f);
            break;

        case NPU_CORETYPE::MIX:
            if (infer.hasExpert) {
                this->current_coretype = NPU_CORETYPE::AIV;
                AddFunctionForCoreType(gvar, f);

                this->current_coretype = NPU_CORETYPE::AIC;
                AddFunctionForCoreType(gvar, f);
            } else {
                this->current_coretype = NPU_CORETYPE::MIX;
                AddFunctionForCoreType(gvar, f);
            }
            break;

        default:
            break;
    }
}

// New Expr functions after removing inheritance form CodeGenC class

mlir::Value CodeGenTileLangNPUIRDEV::GetVarValue(const VarNode *v) const {
  for (auto it = var_map_.rbegin(); it != var_map_.rend(); ++it) {
    auto res = it->find(v);
    if (res != it->end()) {
      return res->second;
    }
  }
  return mlir::Value{};
}

mlir::Value CodeGenTileLangNPUIRDEV::GetVarValue(const CallNode *region_node) const {
  tvm::tl::RegionOp regionop(region_node->args, this->vmap);
  return GetVarValue(regionop.GetBuffer());
}

mlir::Value CodeGenTileLangNPUIRDEV::GetVarValue(const Buffer &buffer_data) const {
  auto var_ptr = buffer_data->data.get();
  return GetVarValue(var_ptr);
}

void CodeGenTileLangNPUIRDEV::SetVarValue(const VarNode *v, const mlir::Value &value) {
  ICHECK(!var_map_.empty()) << "var_map_ is empty, fail to set value";
  var_map_.back()[v] = value;
}

void CodeGenTileLangNPUIRDEV::SetVarValue(const CallNode *region_node, const mlir::Value &value) {
  tvm::tl::RegionOp regionop(region_node->args, this->vmap);
  SetVarValue(regionop.GetBuffer(), value);
}

void CodeGenTileLangNPUIRDEV::SetVarValue(const Buffer &buffer_data, const mlir::Value &value) {
  auto var_ptr = buffer_data->data.get();
  SetVarValue(var_ptr, value);
}

void CodeGenTileLangNPUIRDEV::AddVarLayer() {
  var_map_.emplace_back();
}

void CodeGenTileLangNPUIRDEV::DeleteVarLayer() {
  ICHECK(!var_map_.empty()) << "var_map_ is empty, fail to delete layer";
  var_map_.pop_back();
}

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const VarNode *op) {
  return GetVarValue(op);
}

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const StringImmNode *op) {
  // Todo: Implementation pending
  LOG(FATAL) << "StringImmNode case not supported!";
}

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const ModNode *op) {
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

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const NotNode *op) {
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

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const LetNode *op) {
  auto it = GetVarValue(op->var.get());
  if (it != mlir::Value{}) {
    LOG(FATAL) << "Variable already exists: " << op->var.get()->name_hint;
  }
  auto var_value = MakeValue(op->value);
  SetVarValue(op->var.get(), var_value);
  return MakeValue(op->body);
}

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const BufferLoadNode *op) {
  auto buffer = op->buffer;
  auto indices = op->indices;

  // Check pre-conditions
  if (op->dtype.lanes() != 1) {
    LOG(FATAL) << "lanes not one";
  }
  if (op->dtype != buffer->dtype) {
    LOG(FATAL) << "The load type and buffer element type do not match";
  }

  // Convert buffer from Buffer in TIR 2 tensor in MLIR
  auto mem = GetVarValue(buffer->data.get());

  // Convert index from PrimExpr in TIR 2 index type in MLIR
  SmallVector<mlir::Value> convert_inds;
  for (auto index : indices) {
    mlir::Value indexVal = CreateIndexCastOp(MakeValue(index));
    convert_inds.push_back(indexVal);
  }

  if (mem.getType().isa<mlir::MemRefType>()) {
    // Create a memref.load op for the memref-typed buffer.
    return builder.create<mlir::memref::LoadOp>(builder.getUnknownLoc(), mem, convert_inds);
  } else if (mem.getType().isa<mlir::TensorType>()) {
    // Create a tensor.extract op for the tensor-typed buffer.
    return builder.create<mlir::tensor::ExtractOp>(builder.getUnknownLoc(), mem, convert_inds);
  } else {
    // Throw a fatal error for illegal types
    LOG(FATAL) << "The buffer type in BufferLoadNode must be one of tensor or memref";
  }
}

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const RampNode *op) {
  // Todo: Implementation pending
  LOG(FATAL) << "RampNode case not supported!";
}

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const ShuffleNode *op) {
  // Todo: Implementation pending
  LOG(FATAL) << "ShuffleNode case not supported!";
}

mlir::Value CodeGenTileLangNPUIRDEV::VisitExpr_(const BroadcastNode *op) {
  // Todo: Implementation pending
  LOG(FATAL) << "BroadcastNode case not supported!";
}

void CodeGenTileLangNPUIRDEV::VisitStmt_(const BufferStoreNode *op) {
  auto buffer = op->buffer;
  auto value = op->value;
  auto indices = op->indices;

  if (op->value.dtype().lanes() != 1) {
    LOG(FATAL) << "lanes not one";
  }
  if (op->value.dtype() != buffer->dtype) {
    LOG(FATAL) << "The store type and buffer element type do not match";
  }

  auto mem = GetVarValue(buffer->data.get());

  auto mlir_value = MakeValue(value);

  SmallVector<mlir::Value> convert_inds;
  for (auto index : indices) {
    mlir::Value indexVal = CreateIndexCastOp(MakeValue(index));
    convert_inds.push_back(indexVal);
  }

  if (mem.getType().isa<mlir::MemRefType>()) {
    // Create a memref.store op for the memref-typed buffer.
    builder.create<mlir::memref::StoreOp>(builder.getUnknownLoc(), mlir_value,
                                          mem, convert_inds);
  } else if (mem.getType().isa<mlir::TensorType>()) {
    // Create a tensor.insert op for the tensor-typed buffer.
    mlir::Value result = builder.create<mlir::tensor::InsertOp>(builder.getUnknownLoc(), mlir_value,
                                                                mem, convert_inds);
    SetVarValue(buffer, result);
  } else {
    // Throw a fatal error for illegal types
    LOG(FATAL) << "The buffer type in BufferStoreNode must be one of tensor or memref";
  }
}

void CodeGenTileLangNPUIRDEV::VisitStmt_(const WhileNode *op) {
  // Todo: Implementation pending
  LOG(FATAL) << "WhileNode case not supported!";
}

void CodeGenTileLangNPUIRDEV::VisitStmt_(const AllocateConstNode *op) {
  // Todo: Implementation pending
  LOG(FATAL) << "AllocateConstNode case not supported!";
}

void CodeGenTileLangNPUIRDEV::VisitStmt_(const AssertStmtNode *op) {
  // Todo: Implementation pending
  LOG(FATAL) << "AssertStmtNode case not supported!";
}

void CodeGenTileLangNPUIRDEV::VisitStmt_(const SeqStmtNode *op) {
  // EmitDebugLocation(op);
  for (Stmt stmt : op->seq) {
    this->VisitStmt(stmt);
  }
}

void CodeGenTileLangNPUIRDEV::VisitStmt_(const EvaluateNode *op) {
  // EmitDebugLocation(op);
  MakeValue(op->value);
}

void CodeGenTileLangNPUIRDEV::VisitStmt_(const DeclBufferNode *op) {
  // EmitDebugLocation(op);
  VisitStmt(op->body);
}

void CodeGenTileLangNPUIRDEV::LoopCarriedVarCollector::VisitExpr_(
    const tir::CallNode *call) {
  auto check_var = [&](const tir::VarNode *var_node) {
    if (var_node && outer_->GetVarValue(var_node) != mlir::Value{}) {
      loop_carried_vars_.insert(var_node);
    }
  };
  auto process_call_arg = [&](int arg_index) {
    auto &arg = call->args[arg_index];
    if (arg.as<IntImm>() || arg.as<FloatImm>()) {
      // Immediate number type, no processing required
    } else {
      const CallNode *region_node = arg.as<CallNode>();
      if (region_node) {
        tvm::tl::RegionOp regionop(region_node->args, outer_->vmap);
        check_var(regionop.GetBuffer()->data.get());
      }
    }
  };
  if (call->op.same_as(Op::Get("tl.npuir_dot"))) {
    tvm::tl::NpuirDot npuirop(call->args, outer_->vmap);
    check_var(npuirop.src0->data.get());
    check_var(npuirop.src1->data.get());
    check_var(npuirop.dst->data.get());
  } else if (call->op.same_as(Op::Get("tl.npuir_add")) ||
             call->op.same_as(Op::Get("tl.npuir_sub")) ||
             call->op.same_as(Op::Get("tl.npuir_mul")) ||
             call->op.same_as(Op::Get("tl.npuir_div")) ||
             call->op.same_as(Op::Get("tl.npuir_pow")) ||
             call->op.same_as(Op::Get("tl.npuir_max")) ||
             call->op.same_as(Op::Get("tl.npuir_min")) ||
             call->op.same_as(Op::Get("tl.npuir_and")) ||
             call->op.same_as(Op::Get("tl.npuir_shl")) ||
             call->op.same_as(Op::Get("tl.npuir_shr")) ||
             call->op.same_as(Op::Get("tl.npuir_cmp")) ||
             call->op.same_as(Op::Get("tl.npuir_xor")) ||
             call->op.same_as(Op::Get("tl.npuir_or"))) {
    process_call_arg(0);
    process_call_arg(1);
    process_call_arg(2);
  }
  tir::StmtExprVisitor::VisitExpr_(call);
}

void CodeGenTileLangNPUIRDEV::LoopCarriedVarCollector::VisitStmt_(const tir::BufferStoreNode* op) {
  auto check_var = [&](const tir::VarNode* var_node) {
    if (var_node && outer_->GetVarValue(var_node) != mlir::Value{}) {
      loop_carried_vars_.insert(var_node);
    }
  };

  check_var(op->buffer->data.get());
}

} // namespace codegen
} // namespace tvm
