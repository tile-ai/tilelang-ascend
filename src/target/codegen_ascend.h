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
namespace codegen {

class CodeGenTileLangAscend final : public CodeGenC {
public:
  CodeGenTileLangAscend();
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
  void VisitStmt_(const AllocateNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;

  // Override this as a work around for __grid_constant__ parameter
  void AddFunction(const GlobalVar &gvar, const PrimFunc &f);

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

  std::string core_num_{0};

  std::vector<std::string> para_;

  std::string block_id_;

  Map<Var, PrimExpr> address_map_;

  Map<Var, PrimExpr> tiling_map_;

  Map<String, PrimExpr> address_offset_;

  bool use_swizzle_{false};
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_CUDA_H_
