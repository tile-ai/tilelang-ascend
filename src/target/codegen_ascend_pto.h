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
  void ProcessTilingInput(std::ostream &os, std::string func_name,
                          std::vector<std::string> &arg_names,
                          std::vector<const tir::VarNode *> &shape_vars);
  void CallTilingInput(std::ostream &os, std::string func_name,
                       std::vector<std::string> &tiling_args,
                       std::vector<const tir::VarNode *> &shape_vars);
  void PrintHostFunc(const PrimFunc &f, const std::string &name,
                     std::ostringstream &os, std::string &core,
                     std::vector<const tir::VarNode *> &shape_vars);

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

  void AllocateLocalVar(const AllocateNode *op, std::string &vid,
                        std::string dtype);
  void UnaryVecOpCodegen(const CallNode *op, const std::string &op_name);
  void ScalarOpCodegen(const CallNode *op, const std::string &op_name);
  void AxpyCodegen(const CallNode *op);
  void BinaryVecClampMaxMinOpsCodegen(const CallNode *op,
                                      const std::string &op_name);
  void BinaryVecClampOpsCodegen(const CallNode *op);
  void SigmoidCodegen(const CallNode *op, const std::string &op_type);
  void SiluCodegen(const CallNode *op);
  void MulAddDstCodegen(const CallNode *op);
  void CastCodegen(const CallNode *op, const std::string &op_type);
  void ReduceOpCodegen(const CallNode *op);

  enum class ReduceKind { SUM, MAX, MIN };
  enum class ReduceDirection { ROW, COL };

  struct ReduceOpInfo {
    ReduceKind kind;
    ReduceDirection direction;
    int buffer_slice_row;
    int buffer_slice_col;
  };

  // Override this as a work around for __grid_constant__ parameter
  void AddFunction(const GlobalVar &gvar, const PrimFunc &f);

  struct ShapeInfo {
    int32_t row;
    int32_t col;
    int32_t slice_row;
    int32_t slice_col;
    int32_t slice_valid_row;
    int32_t slice_valid_col;
    int32_t extent;
    PrimExpr first_addr;
    std::string offset;
    std::string type;
    std::string ub_name;
    bool is_slice;
  };

  struct BufferInfo {
    const CallNode *access_ptr;
    Var var;
    std::string id;
    PrimExpr offset;
    DataType dtype;
    Array<PrimExpr> shape;
  };

private:
  void AutoBarrierCodegen(const CallNode *op);
  void AutoFlagOpCodegen(const CallNode *op, std::string op_name);

private:
  // Whether scope such as "__shared__" or "__constant__"  is part of type.
  bool IsScopePartOfType() const final { return false; }

  friend void PrintConst(const FloatImmNode *op, std::ostream &os,
                         CodeGenTileLangAscendPto *p);

  std::string GetVarId(const Var &var) const;

  BufferInfo GetBufferInfo(const PrimExpr &arg) const;

  void BinaryVecOpCodegen(const CallNode *op, const std::string &op_name);

  void BinaryVecOpsCodegen(const CallNode *op, const std::string &op_name);

  void CallExternCodegen(const CallNode *op);

  void GemmV0Codegen(const CallNode *op);

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

  void MergeSortCodegen(const CallNode *op, const std::string &op_name);

  void SortCodegen(const CallNode *op);

  void TopKCodegen(const CallNode *op);

  // Emits a single tl::ascend_pto::Sort<UserT, N, ActualCount, TopK>(...)
  // call. The full algorithm (pad, sort32, merge tree, finalize) lives in
  // pto/common.h.
  void EmitSortAlgorithm(const CallNode *dst_call, const CallNode *src_call,
                         const CallNode *tmp_call, int32_t repeat_times,
                         int32_t actual_num, int32_t top_k);

  void TransposeCodegen(const CallNode *op, const std::string &op_name);

  void XorCodegen(const CallNode *op, const std::string &op_name);

  void CompareCodegen(const CallNode *op, const std::string &op_name);

  void CompareScalarCodegen(const CallNode *op, const std::string &op_name);

  void TshCodegen(const CallNode *op, const std::string &op_name);

  void ArithProgressionCodegen(const CallNode *op, const std::string &op_name);

  void PrintfOpCodegen(const CallNode *op, const std::string &op_name);

  void DumpTensorCodegen(const CallNode *op, const std::string &op_name);

  void BroadcastOpCodegen(const CallNode *op);

  void SelectCodegen(const CallNode *op);

  void SetDeqScaleCodegen(const CallNode *op);

  void MmaCodegen(const CallNode *op);

  std::vector<std::string> GetGlobalTensorShapes(const CallNode *op,
                                                 std::string tensor_addr);

  std::string GetPadEnum(const PrimExpr value);

  void GMCopyCall(const CallNode *call, std::string op_name);

  void CopyUBToUBCodegen(const CallNode *call);

  void CopyL1ToL0Codegen(const CallNode *call, bool is_a);

  std::string PrintBufferOffset(const CallNode *op);

  std::string GetTempVarName(const std::string &temp_name);
  void CreateUbVariableND(const std::string &temp_name,
                          const ShapeInfo &shape_info);
  void CreateUbVariableDN(const std::string &temp_name,
                          const ShapeInfo &shape_info);
  void CreateCubeVariable(const std::string &temp_name,
                          const ShapeInfo &shape_info,
                          const std::string &tile_name);
  std::string ResolveSliceName(const ShapeInfo &info);
  std::string ResolveCubeSliceName(const ShapeInfo &info,
                                   const std::string &tile_name);
  ShapeInfo GetSliceInfo(const CallNode *op);

  ReduceOpInfo ParseReduceOpInfo(const std::string &op_name);
  std::string GetReduceOpName(ReduceKind kind, ReduceDirection direction);
  void CodegenRowReduce(const ReduceOpInfo &op_info, const ShapeInfo &dst,
                        const ShapeInfo &src, const ShapeInfo &tmp);
  void CodegenColReduce(const ReduceOpInfo &op_info, const ShapeInfo &dst,
                        const ShapeInfo &src, const ShapeInfo &tmp);

  void CodegenRowBroadcast(const ShapeInfo &dst, const ShapeInfo &src);
  void CodegenColBroadcast(const ShapeInfo &dst, const ShapeInfo &src);

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

  std::string core_num_{"1"};

  std::vector<std::string> para_;

  std::string block_id_;
  std::string vec_id_;

  Map<Var, PrimExpr> address_map_;
  Map<Var, Array<PrimExpr>> buffer_shapess_;
  Map<Var, PrimExpr> buffer_versions_;

  Map<Var, PrimExpr> tiling_map_;
  Array<Var> var_sequence_;

  Map<String, PrimExpr> address_offset_;
  Map<Var, PrimExpr> buffer_address_map_;
  int64_t max_ub_addr_{0};

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

  struct global_tensor {
    String shape_type;
    String dtype;
    Array<String> shape_list;
  };
  std::unordered_map<String, global_tensor> global_tensor_template;

  std::unordered_map<std::string, int32_t> counters_;

  bool use_swizzle_{false};

  std::string platform_;

  std::string current_resource_scope_ =
      ""; // Identifies whether it's CUBE or VEC
};

} // namespace codegen
} // namespace tvm
#endif // TVM_TL_TARGET_CODEGEN_ASCEND_PTO_H_