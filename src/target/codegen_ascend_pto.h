// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file target/codegen_ascend_pto.h
 * \brief Utility to generate code
 */
#ifndef TVM_TL_TARGET_CODEGEN_ASCEND_PTO_H_
#define TVM_TL_TARGET_CODEGEN_ASCEND_PTO_H_

#include <tvm/target/codegen.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <string>
#include <unordered_map>

#include "target/source/codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenTileLangAscendPto final : public CodeGenC {
public:
  CodeGenTileLangAscendPto(std::string platform);
  std::string Finish();
  // override behavior
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintExtraAttrs(const PrimFunc &f);
  void PreFunctionBody(const PrimFunc &f) final;
  void VisitStmt_(const ForNode *op) final;
  void PrintStorageScope(const std::string &scope,
                         std::ostream &os) final;     // NOLINT(*)
  void PrintType(DataType t, std::ostream &os) final; // NOLINT(*)
  void ProcessTilingInput(std::ostream &os, std::string func_name, std::vector<std::string> &arg_names,
    std::vector<const tir::VarNode*> &shape_vars);
  void CallTilingInput(std::ostream &os, std::string func_name, std::vector<std::string> &tiling_args,
    std::vector<const tir::VarNode*> &shape_vars);
  void PrintHostFunc(const PrimFunc &f, const std::string &name, std::ostringstream &os,
                     std::string &core,
                     std::vector<const tir::VarNode*> &shape_vars);

  // overload visitor
  void VisitExpr_(const FloatImmNode *op, std::ostream &os) final;
  void VisitExpr_(const CallNode *op, std::ostream &os) final;
  void VisitExpr_(const FloorDivNode *op, std::ostream &os);
  void VisitExpr_(const FloorModNode *op, std::ostream &os);
  void VisitExpr_(const SelectNode *op, std::ostream &os) final;
  void VisitExpr_(const BufferLoadNode *op, std::ostream &os) final;
  void VisitStmt_(const BufferStoreNode *op) final;
  void VisitStmt_(const AllocateNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;

  void UnaryVecOpCodegen(const CallNode *op, const std::string& op_name);
  void ScalarOpCodegen(const CallNode *op, const std::string& op_name);
  void AxpyCodegen(const CallNode *op);
  void BinaryVecClampMaxMinOpsCodegen(const CallNode *op, const std::string& op_name);
  void BinaryVecClampOpsCodegen(const CallNode *op, const std::string& op_name);
  void SigmoidCodegen(const CallNode *op, const std::string& op_type);
  void CastCodegen(const CallNode *op, const std::string& op_type);
  void ReduceOpCodegen(const CallNode *op);

  // Override this as a work around for __grid_constant__ parameter
  void AddFunction(const GlobalVar &gvar, const PrimFunc &f);

private:
  void AutoBarrierCodegen (const CallNode *op);
  void AutoFlagOpCodegen (const CallNode *op, std::string op_name);

private:
  // Whether scope such as "__shared__" or "__constant__"  is part of type.
  bool IsScopePartOfType() const final { return false; }

  friend void PrintConst(const FloatImmNode *op, std::ostream &os,
                         CodeGenTileLangAscendPto *p);

  friend void PrintConst(const FloatImmNode *op, std::ostream &os,
                         CodeGenTileLangAscendPto *p);

  void BinaryVecOpCodegen(const CallNode* op, const std::string& op_name);

  void BinaryVecOpsCodegen(const CallNode* op, const std::string& op_name);

  void CallExternCodegen(const CallNode *op);

  void GemmV0Codegen(const CallNode *op);

  void GemmV1Codegen(const CallNode *op);
  
  void SyncAllCodegen(const CallNode *op);

  void PipeBarrierCodegen(const CallNode *op);

  void SetAndWaitFlagCodegen(const CallNode *op, const std::string &op_name);

  void HandleA5Flag(const std::string &op, const std::string &pipe, int flag);

  void SetCrossFlagCodegen(const CallNode *op);

  void AutoSetCrossFlagCodegen(const CallNode *op);

  void WaitCrossFlagCodegen(const CallNode *op);

  void FillCodegen(const CallNode *op);

  void CreateVecIndexCodegen(const CallNode *op, const std::string &op_name);

  void GatherbCodegen(const CallNode *op, const std::string &op_name);

  void GatherMaskCodegen(const CallNode *op, const std::string &op_name);

  void PowCodegen(const CallNode *op);

  void Sort32Codegen(const CallNode *op, const std::string &op_name);

  void TransposeCodegen(const CallNode *op, const std::string &op_name);

  void XorCodegen(const CallNode *op, const std::string &op_name);

  void CompareCodegen(const CallNode *op, const std::string &op_name);

  void CompareScalarCodegen(const CallNode *op, const std::string &op_name);

  void TshCodegen(const CallNode *op, const std::string &op_name);

  void ArithProgressionCodegen(const CallNode *op, const std::string &op_name);

  void PrintfOpCodegen(const CallNode *op, const std::string& op_name);

  void DumpTensorCodegen(const CallNode *op, const std::string &op_name);
  
  void BroadcastOpCodegen(const CallNode *op);

  void SelectCodegen(const CallNode *op);

  void SetDeqScaleCodegen(const CallNode *op);

  std::vector<std::string> GetGlobalTensorShapes(const CallNode *op, std::string tensor_addr);

  std::string PrintBufferOffset(const CallNode *op);
  void UbShapeInputCheck(const AllocateNode *op);
  bool ValidLayoutEnabled(const AllocateNode *op);

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

  std::string core_num_{0};

  std::vector<std::string> para_;

  std::string block_id_;
  std::string vec_id_;

  Map<Var, PrimExpr> address_map_;
  Map<Var, Array<PrimExpr>> buffer_shapess_;
  Map<Var, PrimExpr> buffer_versions_;

  Map<Var, PrimExpr> tiling_map_;
  Array<Var> var_sequence_;

  Map<String, PrimExpr> address_offset_;

  Map<String, String> copy_tmplte_map_;
  Map<String, String> copy_base_addr_map_;

  std::map<std::string, std::vector<std::string>> ub_data_map_;
  std::map<std::string, std::vector<std::string>> l_data_map_;
  std::map<std::string, std::string> for_num_map_;
  std::map<std::string, std::pair<int, int>> prefetch_n_stages_map_;

  std::unordered_map<std::string, std::string> dtype_map = {
        {"int8", "char"},
        {"int32", "int"},
        {"int8x4", "int32_t"},
        {"int32x4", "int32x4"},
        {"float16", "half"},
        {"float32", "float"},
        {"float64", "double"},
        {"float16x4", "float16x4"},
        {"bfloat16x4", "bfloat16x4"},
        {"float32x4", "float32x4"},
        {"float32x16", "float32x16"}};
  
  struct global_tensor{
    String shape_type;
    String dtype;
    Array<String> shape_list;
  };
  std::unordered_map<String, global_tensor> global_tensor_template;

  bool use_swizzle_{false};

  std::string platform_;

  std::string current_resource_scope_ = ""; // 标识是CUBE还是VEC

  int32_t select_num = 0;

  int32_t reduce_num = 0;
};

} // namespace codegen
} // namespace tvm
#endif // TVM_TL_TARGET_CODEGEN_ASCEND_PTO_H_
