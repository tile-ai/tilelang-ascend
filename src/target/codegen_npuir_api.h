// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file target/codegen.h
 * \brief Utility to generate code
 */
// #ifndef TVM_TL_TARGET_CODEGEN_CUDA_H_
// #define TVM_TL_TARGET_CODEGEN_CUDA_H_

#include "../op/op.h"
#include "codegen_npuir.h"
#include "mlir/IR/Value.h"
#include "target/source/codegen_c.h"
#include <assert.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <tvm/arith/analyzer.h>
#include <tvm/ir/module.h>
#include <tvm/target/codegen.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <unordered_map>
#include <vector>

// For adding MLIR APIs to support codegen

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

//===----------------------------------------------------------------------===//
// HIVM Dialect
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

//===----------------------------------------------------------------------===//
// HFusion Dialect
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "tvm/ir/attrs.h"
#include "tvm/ir/expr.h"
#include "tvm/runtime/data_type.h"

using namespace mlir;

// For adding MLIR APIs to support codegen end here

namespace tvm {
namespace codegen {

// All VisitExpr inherited from ExprFunctor take PrimExpr as an argument and
// return mlir::Value All VisitStmt inherited from StmtFunctor take PrimExpr as
// an argument and return nothing
class CodeGenTileLangNPUIRAPI final
    : public ExprFunctor<mlir::Value(const PrimExpr &)>,
      public StmtFunctor<void(const Stmt &)> {
public:
  CodeGenTileLangNPUIRAPI();
  std::string Finish();
  // overload visitor
  mlir::Value VisitExpr_(const MinNode *op) final;
  mlir::Value VisitExpr_(const MaxNode *op) final;
  mlir::Value VisitExpr_(const AddNode *op) final;
  mlir::Value VisitExpr_(const AndNode *op) final;
  mlir::Value VisitExpr_(const OrNode *op) final;
  mlir::Value VisitExpr_(const SubNode *op) final;
  mlir::Value VisitExpr_(const MulNode *op) final;
  mlir::Value VisitExpr_(const DivNode *op) final;
  mlir::Value VisitExpr_(const LTNode *op) final;
  mlir::Value VisitExpr_(const LENode *op) final;
  mlir::Value VisitExpr_(const NENode *op) final;
  mlir::Value VisitExpr_(const EQNode *op) final;
  mlir::Value VisitExpr_(const GTNode *op) final;
  mlir::Value VisitExpr_(const GENode *op) final;
  mlir::Value VisitExpr_(const FloatImmNode *op) final;
  mlir::Value VisitExpr_(const IntImmNode *op) final;
  mlir::Value VisitExpr_(const CallNode *op) final;
  mlir::Value VisitExpr_(const FloorDivNode *op);
  mlir::Value VisitExpr_(const FloorModNode *op);
  mlir::Value VisitExpr_(const CastNode *op) final;
  mlir::Value VisitExpr_(const SelectNode *op) final;
  // new Expr
  mlir::Value VisitExpr_(const VarNode *op) final;
  mlir::Value VisitExpr_(const StringImmNode *op) final;
  mlir::Value VisitExpr_(const ModNode *op) final;
  mlir::Value VisitExpr_(const NotNode *op) final;
  mlir::Value VisitExpr_(const LetNode *op) final;
  mlir::Value VisitExpr_(const BufferLoadNode *op) final;
  mlir::Value VisitExpr_(const RampNode *op) final;
  mlir::Value VisitExpr_(const ShuffleNode *op) final;
  mlir::Value VisitExpr_(const BroadcastNode *op) final;

  // Stmt
  void VisitStmt_(const ForNode *op) final;
  void VisitStmt_(const tir::IfThenElseNode *op) final;
  void VisitStmt_(const AllocateNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;
  void VisitStmt_(const LetStmtNode *op) final;
  // new stmt
  void VisitStmt_(const BufferStoreNode *op) final;
  void VisitStmt_(const WhileNode *op) final;
  void VisitStmt_(const AllocateConstNode *op) final;
  void VisitStmt_(const AssertStmtNode *op) final;
  void VisitStmt_(const SeqStmtNode *op) final;
  void VisitStmt_(const EvaluateNode *op) final;
  void VisitStmt_(const DeclBufferNode *op) final;

  // Override this as a work around for __grid_constant__ parameter
  void AddFunction(const GlobalVar &gvar, const PrimFunc &f);
  void AddFunctionForCoreType(const GlobalVar &gvar, const PrimFunc &f);

  /*
  Using composite key for prim_expr_map consisting of PrimExpr and Block
  Two PrimExpr are treated equal only if they are similar and are in same scope
  Block keeps scope of PrimExpr
  */
  struct PrimExprMapKey {
    tvm::PrimExpr expr;
    const mlir::Block* block;

    bool operator==(const PrimExprMapKey &key) const {
      // Use StructuralEqual instead of pointer comparison
      static const tvm::StructuralEqual equal;
      return block == key.block && equal(expr, key.expr);
    }
  };

  struct PrimExprMapKeyHash {
    std::size_t operator()(const PrimExprMapKey &key) const {
      // Combined hash function for PrimExpr and Block
      static const tvm::StructuralHash hasher;
      return llvm::hash_combine(hasher(key.expr), key.block);
    }
  };
  // map to restrict the creation of duplicated nodes in MLIR
  std::unordered_map<PrimExprMapKey, mlir::Value, PrimExprMapKeyHash> prim_expr_map;

  /*
  Using composite key for mlir_value_map consisting of mlir::Value and Block
  Two mlir::Value are treated equal only if they are similar and are in same scope
  Block keeps scope of mlir::Value
  */
  struct MLIRValueMapKey {
    mlir::Value value;
    const mlir::Block* block;
    // check if both value and scope are same
    bool operator==(const MLIRValueMapKey &key) const {
      return value == key.value && block == key.block;
    }
  };

  struct MMLIRValueKeyHash {
    std::size_t operator()(const MLIRValueMapKey &key) const {
      // mlir::Value supports pointer identity hashing via getAsOpaquePointer()
      return llvm::hash_combine(key.value.getAsOpaquePointer(), key.block);
    }
  };
  // map to restrict the creation of duplicated nodes in MLIR (using MLIR value as key)
  std::unordered_map<MLIRValueMapKey, mlir::Value, MMLIRValueKeyHash> mlir_value_map;

  // Get current function name
  String GetCurrentFunctionName();

protected:
  // MLIR context, module, and builder
  mlir::MLIRContext context;
  mlir::OwningOpRef<ModuleOp> module;
  mlir::OpBuilder builder;
  // DType to MLIR type conversion
  mlir::Type DTypetoMLIRType(DataType t);
  // Get MLIR type of PrimExpr expression
  mlir::Type GetMLIRType(const PrimExpr &expr);
  mlir::Type GetMLIRType(const Buffer &buffer);

  // Make mlir::value from PrimExpr
  mlir::Value MakeValue(const PrimExpr &e) { return VisitExpr(e); }
  // Initialize each function state in AddFunction
  void InitFuncState();

  /*! \brief The storage information */
  struct StorageInfo {
    /*! \brief The alignment of allocation */
    int alignment{0};
  };

  // The definition of local variable.
  // Variable shadowing and scoping is not a problem in TileLang
  // Each variable assignment gets a unique name in TIR
  std::unordered_map<const VarNode *, mlir::Value> var_map_;
  // global strings
  std::unordered_map<std::string, mlir::Value> str_map_;
  // Whether current function is restricted
  bool is_restricted_{true};
  // The analyzer information
  std::unique_ptr<arith::Analyzer> analyzer_;
  // set of volatile buffer.
  std::unordered_set<const VarNode *> volatile_buf_;
  // set of var that are not restricted(can alias)
  std::unordered_set<const VarNode *> alias_var_set_;
  /*! \brief the storage scope of allocation */
  std::unordered_map<const VarNode *, StorageInfo> alloc_storage_info_;
  // Get variable value 
  mlir::Value GetVarValue(const VarNode *v) const;
  // Get the corresponding thread index
  template <typename T>
  mlir::Value GetAndCastIndexOp(const IterVar iv);

private:
  mlir::Value GetEventID(PrimExpr id);
  template <typename T, typename T1> void PipeFlagCodegen(const CallNode *op);
  mlir::Value ScalarConvertType(const PrimExpr &imm, DataType targetDtype);
  template <typename T> void SyncBlockCodegen(const T &sync_op);
  void CallExternCodegen(const CallNode *op);
  void AscendCopyCodegen(const CallNode *op);
  void Nd2NzCodegen(const CallNode *op);
  void Nz2NdCodegen(const CallNode *op);
  void VexpCodegen(const CallNode *op);
  void VbrcCodegen(const CallNode *op);
  void VcastCodegen(const CallNode *op);
  void VreduceCodegen(const CallNode *op);
  void VcumsumCodegen(const CallNode *op);
  void VgatherCodegen(const CallNode *op);
  void VtransposeCodegen(const CallNode *op);
  void VinterleaveCodegen(const CallNode *op);
  void VarangeCodegen(const CallNode *op);
  void VconcatCodegen(const CallNode *op);
  void VpadCodegen(const CallNode *op);
  void VflipCodegen(const CallNode *op);
  void VdeinterleaveCodegen(const CallNode *op);
  void FixpipeCodegen(const CallNode *op);
  void DotCodegen(const CallNode *op);
  void BitcastCodegen(const CallNode *op);
  void DebugPrintCodegen(const CallNode *op);
  template <typename T> void CreateHIVMBinaryVectorOp(const CallNode *op);
  template <typename T, typename U> void UnaryVecOpCodegen(const CallNode *op);
  void BarrierCodegen(const CallNode *op);
  void VselectCodegen(const CallNode *op);
  template <typename T, typename U>
  mlir::Value BinaryOpCodegen(const PrimExprNode *op, U mode, mlir::Value lhs, mlir::Value rhs);

  // returns HIVM address space against given address space
  mlir::hivm::AddressSpace GetHIVMAddressSpace(String address_space);
  std::vector<long int> GetShape(Array<PrimExpr> extents);

  friend void PrintConst(const FloatImmNode *op, CodeGenTileLangNPUIRAPI *p);

  mlir::Value GenMemrefLoadFromRegion(const BufferLoadNode *op);
  mlir::Value GenSubviewFromRegion(const CallNode *region_node);
  mlir::Value GenSubviewFromRegion(Buffer buffer_data, Array<Range> range);
  mlir::Value CreateIndexCastOp(mlir::Value src);
  std::pair<bool, mlir::Value> CheckMLIRValueMap(mlir::Value val);
  std::pair<bool, mlir::Value> CheckPrimExprMap(const PrimExprNode * op);
  void UpdatePrimExprMap(const PrimExprNode * key, mlir::Value val);
  void UpdateMLIRValueMap(const mlir::Value key,  mlir::Value val);

  // Whether global barrier is needed.
  bool need_global_barrier_{false};
  // Global barrier state
  std::string vid_global_barrier_state_;
  // Global barrier expected node.
  std::string vid_global_barrier_expect_;
  // whether enable fp16
  bool enable_fp16_{false};
  // whether enable bf16
  bool enable_bf16_{false};
  // whether enable fp8
  bool enable_fp8_{false};
  // whether enable int8
  bool enable_int8_{false};
  // whether enable warp shuffle intrinsics
  bool enable_warp_shuffle_{false};
  // whether need math_constants.h
  bool need_math_constants_h_{false};
  // whether need cast_smem_ptr_to_int helper function
  bool need_cast_smem_ptr_to_int_{false};

  std::vector<std::string> inst_;
  bool flush_out_{false};

  int core_num_{0};

  std::string block_id_;

  int copy_num_{0}; // Add this var

  NPU_CORETYPE func_coretype;

  // For mix kernel, generate target functions twice. One is for aic while
  // another is for aiv. current_coretype denotes which coretype that we are
  // within during visiting tir ops.
  NPU_CORETYPE current_coretype;

  tvm::tl::BufferMap vmap{tvm::tl::BufferMap()};

  // Keeps name of current function
  std::string current_function_name;
};

} // namespace codegen
} // namespace tvm
