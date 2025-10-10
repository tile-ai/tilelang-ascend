// Copyright (c) Tile-AI Corporation
//  Licensed under the MIT License

/*!
 * \file target/codegen.h
 * \brief Utility to generate code
 */
#ifndef TVM_TL_TARGET_CODEGEN_CUDA_H_
#define TVM_TL_TARGET_CODEGEN_CUDA_H_

#include <tvm/target/codegen.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <assert.h>
#include <string>
#include <unordered_map>

#include "target/source/codegen_c.h"
#include "../op/op.h"

namespace tvm {
namespace codegen {

enum class NPU_CORETYPE { AIC, AIV, MIX };

class SSAType {
public:
  String type_str = "";
  String var_id = "";

  virtual std::string printType() = 0;
};

class Scalar : public SSAType {
public:
  Scalar(String name, String type) {
    this->var_id = name;
    this->type_str = type;
  }

  std::string printType() { return type_str; }
};

class Memref : public SSAType {
  void GetIntStride();

public:
  Memref(String name, Buffer buffer, bool is_arg = false);
  Memref(String name, Array<PrimExpr> shape_in, DataType dtype_in,
         String address_space, bool var_offset_in,
         Array<PrimExpr> stride_in = Array<PrimExpr>(), int offset_in = 0,
         bool is_arg_in = false);
  std::string printType() { return type_str; }
  int dim;
  Array<PrimExpr> shape;
  Array<PrimExpr> stride;
  std::vector<unsigned long> stride_int;
  unsigned long offset = 0;
  bool var_offset = false;
  bool is_arg = false;
  DataType dtype;
  String address_space = "gm";
};

class CodeGenTileLangNPUIR final : public CodeGenC {
public:
  CodeGenTileLangNPUIR();
  std::string Finish();
  // override behavior
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintExtraAttrs(const PrimFunc &f);
  void VisitStmt_(const ForNode *op) final;
  void VisitStmt_(const tir::IfThenElseNode *op) final;
  void PrintStorageScope(const std::string &scope,
                         std::ostream &os) final;     // NOLINT(*)
  void PrintType(DataType t, std::ostream &os) final; // NOLINT(*)
  void PrintShape(Array<PrimExpr> shape, std::string delimiter,
                  std::ostream &os); // Added function
  void PrintSSAAssign(const std::string &target, const std::string &src,
                      DataType t) final;

  // overload visitor
  void VisitExpr_(const MinNode *op, std::ostream &os) final;
  void VisitExpr_(const MaxNode *op, std::ostream &os) final;
  void VisitExpr_(const AddNode *op, std::ostream &os) final;
  void VisitExpr_(const AndNode *op, std::ostream &os) final;
  void VisitExpr_(const OrNode *op, std::ostream &os) final;
  void VisitExpr_(const SubNode *op, std::ostream &os) final;
  void VisitExpr_(const MulNode *op, std::ostream &os) final;
  void VisitExpr_(const DivNode *op, std::ostream &os) final;
  void VisitExpr_(const LTNode *op, std::ostream &os) final;
  void VisitExpr_(const LENode *op, std::ostream &os) final;
  void VisitExpr_(const NENode *op, std::ostream &os) final;
  void VisitExpr_(const EQNode *op, std::ostream &os) final;
  void VisitExpr_(const GTNode *op, std::ostream &os) final;
  void VisitExpr_(const GENode *op, std::ostream &os) final;
  void VisitExpr_(const FloatImmNode *op, std::ostream &os) final;
  void VisitExpr_(const IntImmNode *op, std::ostream &os) final;
  void VisitExpr_(const CallNode *op, std::ostream &os) final;
  void VisitExpr_(const FloorDivNode *op, std::ostream &os);
  void VisitExpr_(const FloorModNode *op, std::ostream &os);
  void VisitExpr_(const CastNode *op, std::ostream &os) final;
  void VisitExpr_(const SelectNode *op, std::ostream &os) final;
  void VisitExpr_(const BufferLoadNode* op, std::ostream& os) final;
  void VisitStmt_(const AllocateNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;
  void VisitStmt_(const LetStmtNode *op) final;
  void VisitStmt_(const BufferStoreNode* op) final;

  // Override this as a work around for __grid_constant__ parameter
  void AddFunction(const GlobalVar &gvar, const PrimFunc &f);
  void AddFunctionForCoreType(const GlobalVar &gvar, const PrimFunc &f);

private:
  template <typename T>
  std::string ScalarConvertType(T *imm, DataType targetDtype);
  std::string ConstIntCodegen(const IntImmNode *op);
  std::string ConstFloatCodegen(const FloatImmNode *op);
  void CallExternCodegen(const CallNode *op, std::ostream &os);
  void AscendCopyCodegen(const CallNode *op, std::ostream &os);
  void Nd2NzCodegen(const CallNode *op, std::ostream &os);
  template <typename T>
  void UnaryVecOpCodegen(const CallNode *op, std::ostream &os);
  void VbrcCodegen(const CallNode *op, std::ostream &os);
  void VcastCodegen(const CallNode *op, std::ostream &os);
  void VreduceCodegen(const CallNode *op, std::ostream &os);
  void FixpipeCodegen(const CallNode *op, std::ostream &os);
  void DotCodegen(const CallNode *op, std::ostream &os);
  void VcmpCodegen(const CallNode *op, std::ostream &os);
  void VselectCodegen(const CallNode *op, std::ostream &os);
  void BinaryVecOpCodegen(const CallNode *op, std::string opName,
                          std::ostream &os);
  template <typename T>
  void SyncBlockSetCodegen(const T &sync_op, std::ostream &os);
  template <typename T>
  void SyncBlockWaitCodegen(const T &sync_op, std::ostream &os);

  void BarrierCodegen(const CallNode *op, std::ostream &os);
  template <typename T>
  void PipeFlagCodegen(const T &sync_op, std::ostream &os);
  std::string PrintID(PrimExpr id);
  // Whether scope such as "__shared__" or "__constant__" is part of type.
  bool IsScopePartOfType() const final { return false; }

  Array<String> GenConvertIndex(Array<PrimExpr> exprs);
  String GenSubviewFromRegion(const CallNode *region_node);
  String GenSubviewFromRegion(Buffer buffer_data, Array<Range> range);
  void GenRecastFromArg(Buffer curr_buffer, String arg_name,
                        String &recast_inst);
  String GetMemrefInfo(String name);
  String GetMemrefInfo(Memref *memrefObj);
  std::string GetBufferRef(const BufferNode* buffer, Array<PrimExpr> indices);
  // save memref name and type
  std::map<String, SSAType *> type_info;

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
};
} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_CUDA_H_
