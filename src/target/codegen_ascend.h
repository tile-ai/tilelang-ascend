// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file target/codegen.h
 * \brief Utility to generate code
 */
#ifndef TVM_TL_TARGET_CODEGEN_CUDA_H_
#define TVM_TL_TARGET_CODEGEN_CUDA_H_

#include <tvm/target/codegen.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <string>
#include <unordered_map>

#include "target/source/codegen_c.h"

namespace tvm {
namespace tl {
class SelectedCallView;
}
namespace codegen {

constexpr const char *cv_1_1 = "cv_1_1";

constexpr const char *cv_1_2 = "cv_1_2";

class CodeGenTileLangAscend final : public CodeGenC {
public:
  CodeGenTileLangAscend(std::string platform);
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
  void VisitExpr_(const MaxNode *op, std::ostream &os);
  void VisitExpr_(const MinNode *op, std::ostream &os);
  void PrintIntMinMaxTernary(const PrimExpr &a, const PrimExpr &b,
                             const char *cmp, std::ostream &os);
  void VisitExpr_(const MulNode *op, std::ostream &os) final;
  void VisitExpr_(const SelectNode *op, std::ostream &os) final;
  void VisitExpr_(const BufferLoadNode *op, std::ostream &os) final;
  void VisitStmt_(const BufferStoreNode *op) final;
  void VisitStmt_(const AllocateNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;

  // Override this as a work around for __grid_constant__ parameter
  void AddFunction(const GlobalVar &gvar, const PrimFunc &f);

private:
  std::string PrintBufferOffset(const CallNode *call_arg,
                                bool has_offset = true);

  DataType GetAccessPtrDataType(const PrimExpr &arg);

  void AddDeclStream(std::ostringstream &ss, const std::string &str);

  void PrintOpCall(const CallNode *op, const std::string &op_name,
                   std::pair<int, int> buffer_range,
                   std::pair<int, int> expr_range);

  void PrintConstArray(const CallNode *op, int start_idx, int len,
                       const std::string &dtype = "uint32_t");

  void PrintConstArray(const Array<PrimExpr> &args, int start_idx, int len,
                       const std::string &dtype = "uint32_t");

  std::vector<std::string> CollectBufferArgs(const CallNode *op, int begin,
                                             int end, bool has_offset = true);

  std::vector<std::string> CollectBufferArgs(const Array<PrimExpr> &args,
                                             int begin, int end,
                                             bool has_offset = true);

  std::string CoerceScalarArg(const PrimExpr &arg, DataType dtype);

  void EmitVectorCall(const std::string &op_name,
                      const std::vector<std::string> &args);

  bool EmitVectorHelper(const ObjectRef &semantic_op, const CallNode *op);
  void EmitSelectedVectorTerminal(const tl::SelectedCallView &view);
  void EmitSelectedRawBinary(const tl::SelectedCallView &view);
  void EmitSelectedRawUnary(const tl::SelectedCallView &view);
  void EmitSelectedRawScalar(const tl::SelectedCallView &view);
  void EmitSelectedRawReduce(const tl::SelectedCallView &view);
  void EmitSelectedRawBlockReduce(const tl::SelectedCallView &view);
  void EmitSelectedRawWholeReduce(const tl::SelectedCallView &view);
  void EmitSelectedRawCast(const tl::SelectedCallView &view);
  void EmitSelectedRawBroadcast(const tl::SelectedCallView &view);
  void EmitSelectedRawFill(const tl::SelectedCallView &view);
  void EmitSelectedRawClamp(const tl::SelectedCallView &view);
  void EmitSelectedRawRowExpand(const tl::SelectedCallView &view);
  void EmitSelectedRawExpExperiment(const tl::SelectedCallView &view);
  void MaskSetterCodegen(const CallNode *op);

  void TrigOpCodegen(const CallNode *op, const std::string &op_name);

  void TransposeCodegen(const CallNode *op, const std::string &op_name);

  void CreateVecIndexCodegen(const CallNode *op, const std::string &op_name);

  void ArithProgressionCodegen(const CallNode *op);

  void SortCodegen(const CallNode *op);

  void MergeSortCodegen(const CallNode *op);

  void TopKCodegen(const CallNode *op);

  void ShmemCodegen(const CallNode *op);

  void GatherMaskCodegen(const CallNode *op);

  void GatherbCodegen(const CallNode *op);

  void SelectCodegen(const CallNode *op, const std::string &op_name);

  void MulAddDstCodegen(const CallNode *op);

  void InitSortBufCodegen(const CallNode *op);

  void CompareCodegen(const CallNode *op, const std::string &op_name);

  void CompareScalarCodegen(const CallNode *op, const std::string &op_name);

  void Sort32Codegen(const CallNode *op, const std::string &op_name);

  void GatherCodegen(const CallNode *op, const std::string &op_name);

  void ReduceOpCodegen(const CallNode *op);

  void SetDeqScaleCodegen(const CallNode *op, const std::string &op_name);

  void PowerOpCodegen(const CallNode *op, const std::string &op_name);

  void BroadcastOpCodegen(const CallNode *op);

  void TailUnaryOpCodegen(const CallNode *op);

  void TailBinaryOpCodegen(const CallNode *op);

  void TailScalarOpCodegen(const CallNode *op);

  void TailReduceOpCodegen(const CallNode *op);

  void TailCompareOpCodegen(const CallNode *op, bool scalar);

  void TailSelectOpCodegen(const CallNode *op);

  void TailBroadcastOpCodegen(const CallNode *op);

  void RowExpandMulCodegen(const CallNode *op);

  void SetCrossFlagCodegen(const CallNode *op);

  void FlagOpCodegen(const CallNode *op, std::string op_name);

  void PipeBarrierCodegen(const CallNode *op);

  void GemmOpCodegen(const CallNode *op);

  void PrintfOpCodegen(const CallNode *op, const std::string &op_name);

  void DumpTensorCodegen(const CallNode *op);

  void SrcCodeCodegen(const CallNode *op);

  void BilinearInterpolationCodegen(const CallNode *op);

  void AutoBarrierCodegen(const CallNode *op);

  void AutoFlagOpCodegen(const CallNode *op, std::string op_name);

  void AutoSetCrossFlagCodegen(const CallNode *op);

  void AutoWaitCrossFlagCodegen(const CallNode *op);

  void UseSwizzleCodegen(const CallNode *op, std::ostream &os);

  void MmaCodegen(const CallNode *op);

  void CopyCodegen(const CallNode *op);

  void SigmoidCodegen(const CallNode *op, const std::string &op_name);

  void RoundCodegen(const CallNode *op, const std::string &op_name);

  void ReinterpretCastCodegen(const CallNode *op);

  void CreateSubExperimentCodegen(const CallNode *op,
                                  const std::string &op_name);

  void CreateAbsExperimentCodegen(const CallNode *op,
                                  const std::string &op_name);

  void CreateMinsExperimentCodegen(const CallNode *op,
                                   const std::string &op_name);

  void CreateReduceSumExperimentCodegen(const CallNode *op,
                                        const std::string &op_name);

  void GatherMaskExperimentCodegen(const CallNode *op);

  void FillExperimentCodegen(const CallNode *op);

  void SumExperimentCodegen(const CallNode *op);

  void CreateDatacacheExperimentCodegen(const CallNode *op);

  void BrcbExperimentCodegen(const CallNode *op);

private:
  // Whether scope such as "__shared__" or "__constant__"  is part of type.
  bool IsScopePartOfType() const final { return false; }

  friend void PrintConst(const FloatImmNode *op, std::ostream &os,
                         CodeGenTileLangAscend *p);

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
  // cv ratio in vid reduce mode
  std::string cv_ratio_;

  Map<Var, PrimExpr> address_map_;

  Map<Var, PrimExpr> tiling_map_;
  Array<Var> var_sequence_;

  Map<String, PrimExpr> address_offset_;

  bool use_swizzle_{false};

  bool enable_exception_dump_{false};

  std::string platform_;

  Map<Var, Array<PrimExpr>> buffer_shapes_;

  std::unordered_map<const VarNode *, DataType> buffer_dtypes_;

  // Managed Vector operations are legal in the mixed outer region and in an
  // explicit AIV scope, but never in an explicit AIC scope. Setters and
  // terminals emitted in the mixed outer region need an ASCEND_IS_AIV guard.
  bool in_managed_vector_region_{true};
  int current_resource_scope_{-1};
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_CUDA_H_
