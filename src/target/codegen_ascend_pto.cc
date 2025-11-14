// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file target/codegen_ascend_pto.cc
 */

#include "codegen_ascend_pto.h"
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "../op/builtin.h"

#include "arith/pattern_match.h"

namespace tvm {
namespace codegen {

static std::string getType(const DataType &dtype) {
}

CodeGenTileLangAscendPto::CodeGenTileLangAscendPto() {
}

void CodeGenTileLangAscendPto::PrintFuncPrefix(std::ostream &os) {
  //os << "extern \"C\" CATLASS_GLOBAL\n";
}

std::string CodeGenTileLangAscendPto::Finish() {
  //decl_stream << "#include \"tl_templates/ascend/common.h\"\n";
  //decl_stream << "#include \"acl/acl.h\"\n";
  //decl_stream << "#include <runtime/rt_ffts.h>\n";
  // decl_stream << "using namespace Catlass;\n";
  // decl_stream << "\n";
  // std::ostringstream code;
  // code << decl_stream.str();
  // code << stream.str();
  //return code.str();
}

void CodeGenTileLangAscendPto::VisitStmt_(const tir::ForNode *op) {
  // auto flush = false;
  // if (flush_out_) {
  //   flush = true;
  //   flush_out_ = false;
  // }
  // if (op->kind == tir::ForKind::kUnrolled) {
  //   PrintIndent();
  //   stream << "#pragma unroll\n";
  // }
  // std::string extent =
  //     PrintExpr(arith::Analyzer().Simplify(op->extent + op->min));
  // PrintIndent();
  // std::string vid = AllocVarID(op->loop_var.get());
  // std::string start = PrintExpr(op->min);
  // stream << "for (";
  // PrintType(op->loop_var.dtype(), stream);
  // stream << ' ' << vid << " = " << start << "; " << vid << " < " << extent
  //        << "; ++" << vid << ") {\n";
  // int for_scope = BeginScope();
  // PrintStmt(op->body);
  // this->EndScope(for_scope);
  // PrintIndent();
  // stream << "}\n";
  // if (flush) {
  //   while (!inst_.empty()) {
  //     PrintIndent();
  //     stream << inst_.back();
  //     inst_.pop_back();
  //   }
  // }
}

void CodeGenTileLangAscendPto::PrintType(DataType t,
                                      std::ostream &os) { // NOLINT(*)

}

void CodeGenTileLangAscendPto::PrintStorageScope(const std::string &scope,
                                              std::ostream &os) { // NOLINT(*)
}

void CodeGenTileLangAscendPto::VisitExpr_(const FloorDivNode *op,
                                       std::ostream &os) {
  os << "(";
  PrintExpr(op->a, os);
  os << " / ";
  PrintExpr(op->b, os);
  os << ")";
}

void CodeGenTileLangAscendPto::VisitExpr_(const FloorModNode *op,
                                       std::ostream &os) {
  os << "(";
  PrintExpr(op->a, os);
  os << " % ";
  PrintExpr(op->b, os);
  os << ")";
}

void CodeGenTileLangAscendPto::VisitExpr_(const BufferLoadNode *op,
                                       std::ostream &os) {
  // auto var_name = var_idmap_[op->buffer->data.get()];
  // os << var_name << ".GetValue("
  //               << PrintExpr(op->indices.back()) << ")";
}

void CodeGenTileLangAscendPto::VisitExpr_(const CallNode *op, std::ostream &os) {
  
}

void CodeGenTileLangAscendPto::VisitStmt_(const AttrStmtNode *op) {
  
}

void CodeGenTileLangAscendPto::VisitStmt_(const AllocateNode *op) {
  
}

inline void PrintConst(const FloatImmNode *op, std::ostream &os,
                       CodeGenTileLangAscendPto *p) { // NOLINT(*)
}

void CodeGenTileLangAscendPto::VisitExpr_(const FloatImmNode *op,
                                       std::ostream &os) { // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenTileLangAscendPto::PreFunctionBody(const PrimFunc &f) {
  
}

void CodeGenTileLangAscendPto::VisitExpr_(const SelectNode *op, std::ostream &os) {
  auto condition = PrintExpr(op->condition);
  auto true_value = PrintExpr(op->true_value);
  auto false_value = PrintExpr(op->false_value);

  os << "(" << condition << " ? "
     << "" << true_value << " : " << false_value << ")";
}

static void ProcessHostInput(std::ostream &os, std::vector<std::string> &arg_names,
                      std::vector<const tir::VarNode *> &shape_vars) {
}

void CodeGenTileLangAscendPto::CallTilingInput(std::ostream &os, std::string func_name, std::vector<std::string> &tiling_args,
  std::vector<const tir::VarNode*> &shape_vars)
{
}

void CodeGenTileLangAscendPto::ProcessTilingInput(std::ostream &os, std::string func_name, std::vector<std::string> &tiling_args,
  std::vector<const tir::VarNode*> &shape_vars)
{
  
}

void CodeGenTileLangAscendPto::PrintHostFunc(const PrimFunc &f, const std::string &name,
                                          std::ostringstream &os, std::string &core,
                                          std::vector<const tir::VarNode *> &shape_vars) {
  
}

void CodeGenTileLangAscendPto::AddFunction(const GlobalVar &gvar,
                                        const PrimFunc &f) {
  
}

} // namespace codegen
} // namespace tvm
