/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file npu_loop_vectorize.cc
 * \brief vectorize the loop
 */

#include "arith/ir_mutator_with_analyzer.h"
#include "tir/analysis/var_use_def_analysis.h"

#include <tvm/arith/analyzer.h>
#include <tvm/arith/pattern.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "arith/scalable_expression.h"
#include "tir/analysis/check_contains.h"

namespace tvm {
namespace tl {

using namespace tir;

class LoopDecompose : public StmtMutator {
private:
  // All temporary buffers allocated within the current block,
  // registered collectively in VisitStmt_(BlockNode).
  std::vector<Buffer> tmp_buffers;

  // Records the recursion depth of the current nested expression,
  // managed via DepthGuard.
  int depth = 0;

  // Stack of parallel loop variables currently being processed (outer ->
  // inner).
  std::vector<const VarNode *> current_loop_vars_;
  std::vector<int64_t> current_loop_extents_;

  // Statements that need to be hoisted before the outermost parallel loop
  // (copy / reshape / brc).
  std::vector<Stmt> hoisted_stmts_;

  // ============================================================
  // Op mapping table
  // ============================================================

  inline static std::unordered_map<std::string, std::string> TirOps2NpuirOps = {
      {"tir.exp", "tl.npuir_exp"},
      {"tir.fabs", "tl.npuir_abs"},
      {"tir.sigmoid", "tl.npuir_sigmoid"},
      {"tir.if_then_else", "tl.npuir_select"},
      {"Add", "tl.npuir_add"},
      {"Mul", "tl.npuir_mul"},
      {"Sub", "tl.npuir_sub"},
      {"Div", "tl.npuir_div"},
      {"EQ", "tl.npuir_cmp"},
      {"NE", "tl.npuir_cmp"},
      {"LT", "tl.npuir_cmp"},
      {"LE", "tl.npuir_cmp"},
      {"GE", "tl.npuir_cmp"},
      {"GT", "tl.npuir_cmp"},
      {"Broadcast", "tl.npuir_brc"},
      {"Copy", "tl.copy"},
  };

  // ============================================================
  // RAII depth management — prevents forgetting to reset depth to 0 on return.
  // ============================================================

  struct DepthGuard {
    int &depth;
    bool committed = false;

    explicit DepthGuard(int &d) : depth(d) { depth++; }

    // Call when the operation completes successfully; decrements depth.
    void commit() {
      depth--;
      committed = true;
    }

    // If commit() was never called (i.e. an early failure occurred), reset
    // depth to 0.
    ~DepthGuard() {
      if (!committed)
        depth = 0;
    }
  };

  // ============================================================
  // Basic utility functions
  // ============================================================

  bool IsCmpOp(const std::string &op) {
    static const std::unordered_set<std::string> cmp_ops = {"EQ", "NE", "LT",
                                                            "GT", "LE", "GE"};
    return cmp_ops.count(op);
  }

  bool IsScalar(const PrimExpr &expr) {
    return expr.as<IntImmNode>() || expr.as<FloatImmNode>();
  }

  bool IsLoopInvariant(const Array<PrimExpr> &indices,
                       const std::vector<const VarNode *> &loop_vars) {
    for (const auto &idx : indices) {
      if (auto var = idx.as<VarNode>()) {
        for (auto lv : loop_vars) {
          if (lv == var)
            return false;
        }
      }
    }
    return true;
  }

  bool IsScalarLike(const PrimExpr &expr,
                    const std::vector<const VarNode *> &loop_vars) {
    if (IsScalar(expr))
      return true;
    if (auto load = expr.as<BufferLoadNode>())
      return IsLoopInvariant(load->indices, loop_vars);
    return false;
  }

  // Each component of indices must be either an integer constant or an integer
  // variable.
  bool ValidBufferIndices(const Array<PrimExpr> &indices) {
    for (const auto &idx : indices) {
      if (idx.as<IntImmNode>())
        continue;
      if (auto var = idx.as<VarNode>()) {
        if (var->dtype.is_int())
          continue;
      }
      return false;
    }
    return true;
  }

  // ============================================================
  // Op type predicates
  // ============================================================

  bool IsUnaryOp(const PrimExpr &expr) {
    if (auto call = expr.as<CallNode>()) {
      if (auto op = call->op.as<OpNode>()) {
        return op->name == "tir.exp" || op->name == "tir.fabs" ||
               op->name == "tir.sigmoid";
      }
    }
    return false;
  }

  bool IsBinaryOp(const PrimExpr &expr, std::string *op_type,
                  Array<PrimExpr> *operands) {
#define HANDLE_BINARY_OP(NodeType, OpName)                                     \
  if (auto node = expr.as<NodeType>()) {                                       \
    if (op_type)                                                               \
      *op_type = OpName;                                                       \
    if (operands) {                                                            \
      operands->push_back(node->a);                                            \
      operands->push_back(node->b);                                            \
    }                                                                          \
    return true;                                                               \
  }
    HANDLE_BINARY_OP(AddNode, "Add")
    HANDLE_BINARY_OP(MulNode, "Mul")
    HANDLE_BINARY_OP(SubNode, "Sub")
    HANDLE_BINARY_OP(DivNode, "Div")
    HANDLE_BINARY_OP(LTNode, "LT")
    HANDLE_BINARY_OP(LENode, "LE")
    HANDLE_BINARY_OP(GENode, "GE")
    HANDLE_BINARY_OP(GTNode, "GT")
    HANDLE_BINARY_OP(EQNode, "EQ")
    HANDLE_BINARY_OP(NENode, "NE")
#undef HANDLE_BINARY_OP
    return false;
  }

  bool IsTernaryOp(const PrimExpr &expr) {
    if (auto call = expr.as<CallNode>()) {
      if (auto op = call->op.as<OpNode>())
        return op->name == "tir.if_then_else";
    }
    return false;
  }

  // ============================================================
  // Buffer management
  // ============================================================

  struct BufferAccessInfo {
    Buffer buffer;
    Array<PrimExpr> indices;
    bool is_load = true;
  };

  BufferAccessInfo ExtractBufferAccessInfo(const ObjectRef &node) {
    if (auto load = node.as<BufferLoadNode>())
      return {load->buffer, load->indices, true};
    if (auto store = node.as<BufferStoreNode>())
      return {store->buffer, store->indices, false};
    LOG(FATAL) << "Expected BufferLoad or BufferStore, got: "
               << node->GetTypeKey();
    return {};
  }

  // Create a named buffer with the specified shape / scope / dtype.
  Buffer CreateTempBufferWithShape(const Array<PrimExpr> &shape, DataType dtype,
                                   const std::string &name,
                                   const std::string &scope,
                                   const std::string &op_name = "") {
    DataType actual_dtype =
        (!op_name.empty() && IsCmpOp(op_name)) ? DataType::Bool() : dtype;
    Var buf_var(name, PointerType(PrimType(actual_dtype), scope));
    return Buffer(buf_var, dtype, shape, {}, PrimExpr(0), name, 0, 0, kDefault);
  }

  // Create a temporary buffer with the same shape / scope as the reference
  // BufferAccessInfo and register it in tmp_buffers.
  BufferAccessInfo CreateTempBuffer(const BufferAccessInfo &ref,
                                    const std::string &op_name = "npuir_add") {
    static int tmp_id = 0;
    const Buffer &ref_buf = ref.buffer;
    DataType dtype = IsCmpOp(op_name) ? DataType::Bool() : ref_buf->dtype;
    std::string name = "tmp_" + std::to_string(tmp_id++) + "_buf";
    Buffer buf(Var(name, PointerType(PrimType(dtype), ref_buf.scope())), dtype,
               ref_buf->shape, {}, PrimExpr(0), name, 0, 0, kDefault);
    tmp_buffers.push_back(buf);
    return {buf, ref.indices, true};
  }

  // ============================================================
  // Region construction
  // ============================================================

  // Build a region call where every dimension uses the same integer size.
  PrimExpr BuildRegionCall(const BufferAccessInfo &info, int region_id,
                           int size) {
    PrimExpr load = BufferLoad(info.buffer, info.indices);
    Array<PrimExpr> args = {load, make_const(DataType::Int(32), region_id)};
    for (size_t i = 0; i < info.indices.size(); ++i)
      args.push_back(make_const(DataType::Int(32), size));
    return Call(DataType::Handle(), Op::Get("tl.region"), args);
  }

  // Build a region call with explicitly specified offsets and sizes.
  PrimExpr RegionND(Buffer buf, const std::vector<PrimExpr> &offsets,
                    int region_id, const std::vector<int64_t> &sizes) {
    ICHECK_EQ(buf->shape.size(), offsets.size())
        << "RegionND dimension mismatch";
    Array<PrimExpr> args = {
        BufferLoad(buf, Array<PrimExpr>(offsets.begin(), offsets.end())),
        make_const(DataType::Int(32), region_id)};
    for (auto s : sizes)
      args.push_back(make_const(DataType::Int(32), s));
    return Call(DataType::Handle(), Op::Get("tl.region"), args);
  }

  // ============================================================
  // NPUIR instruction construction
  // ============================================================

  // Append a lowercase mode string argument for comparison operators.
  void AppendCmpMode(const std::string &op_name, Array<PrimExpr> &args) {
    if (!IsCmpOp(op_name))
      return;
    std::string lower = op_name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    args.push_back(StringImm(lower));
  }

  Stmt BuildNpuirCall(const std::string &op_name, Array<PrimExpr> args) {
    AppendCmpMode(op_name, args);
    return Evaluate(
        Call(DataType::Void(), Op::Get(TirOps2NpuirOps.at(op_name)), args));
  }

  // Unary: region_in -> out
  Stmt BuildUnaryStmt(const std::string &op_name, const PrimExpr &region_in,
                      const BufferAccessInfo &out) {
    return BuildNpuirCall(op_name, {region_in, BuildRegionCall(out, 2, 1)});
  }

  // Binary: a op b -> out; a/b are already region or scalar PrimExpr values.
  Stmt BuildBinaryStmt(const std::string &op_name, const PrimExpr &region_a,
                       const PrimExpr &region_b, const BufferAccessInfo &out) {
    return BuildNpuirCall(op_name,
                          {region_a, region_b, BuildRegionCall(out, 2, 1)});
  }

  // Ternary: select(cond, true_buf, false_buf) -> out
  Stmt BuildTernaryStmt(const std::string &op_name,
                        const BufferAccessInfo &cond,
                        const BufferAccessInfo &true_buf,
                        const BufferAccessInfo &false_buf,
                        const BufferAccessInfo &out) {
    return BuildNpuirCall(op_name, {BuildRegionCall(cond, 1, 1),
                                    BuildRegionCall(true_buf, 1, 1),
                                    BuildRegionCall(false_buf, 1, 1),
                                    BuildRegionCall(out, 2, 1)});
  }

  // ============================================================
  // Reshape + Broadcast hoisting logic
  // ============================================================

  struct ReshapeBrcInfo {
    bool needed = false;
    // index_to_loop[i]: which loop level buffer dimension i maps to;
    // -1 means loop-invariant.
    std::vector<int> index_to_loop;
    // Copy/reshape source sizes aligned to buffer dimensions
    // (loop-invariant axes are filled with 1).
    std::vector<int64_t> src_sizes;
    // Reshape target shape aligned to loop dimensions
    // (unused loop axes are filled with 1).
    std::vector<int64_t> view_shape;
  };

  ReshapeBrcInfo
  CheckNeedsReshapeBrc(const BufferLoadNode *load,
                       const std::vector<const VarNode *> &loop_vars,
                       const std::vector<int64_t> &loop_extents) {
    ReshapeBrcInfo info;
    int N = loop_vars.size();
    info.index_to_loop.assign(load->indices.size(), -1);

    for (int i = 0; i < (int)load->indices.size(); i++) {
      if (auto var = load->indices[i].as<VarNode>()) {
        for (int j = 0; j < N; j++) {
          if (loop_vars[j] == var) {
            info.index_to_loop[i] = j;
            break;
          }
        }
      }
    }

    std::set<int> used(info.index_to_loop.begin(), info.index_to_loop.end());
    used.erase(-1);

    // If all loop axes are used, or the access is completely loop-invariant,
    // no special handling is needed.
    if ((int)used.size() == N || used.empty())
      return info;

    info.needed = true;

    for (int i = 0; i < (int)load->indices.size(); i++) {
      int lv = info.index_to_loop[i];
      info.src_sizes.push_back(lv >= 0 ? loop_extents[lv] : 1);
    }
    for (int j = 0; j < N; j++)
      info.view_shape.push_back(used.count(j) ? loop_extents[j] : 1);

    return info;
  }

  // Emit copy -> reshape -> brc statements into hoisted_stmts_.
  // Returns a BufferAccessInfo pointing to brc_buf whose indices are
  // aligned with output_ref.
  BufferAccessInfo
  EmitReshapeAndBroadcast(const BufferLoadNode *load, const ReshapeBrcInfo &rbi,
                          const BufferAccessInfo &output_ref,
                          const std::vector<const VarNode *> &loop_vars,
                          const std::vector<int64_t> &loop_extents,
                          const std::string &op_name = "") {
    int N = loop_extents.size();
    const std::string scope = output_ref.buffer.scope();

    // Helper: generate a vector of N zero offsets.
    auto make_zero_offsets = [](int n) {
      std::vector<PrimExpr> v;
      v.reserve(n);
      for (int i = 0; i < n; i++)
        v.push_back(make_const(DataType::Int(32), 0));
      return v;
    };

    // Helper: convert an int64 vector to Array<PrimExpr>.
    auto to_shape_expr = [](const std::vector<int64_t> &s) {
      Array<PrimExpr> arr;
      for (auto v : s)
        arr.push_back(make_const(DataType::Int(32), v));
      return arr;
    };

    // 1. local_buf: copy from the global buffer to a local-scope buffer.
    Buffer local_buf =
        CreateTempBufferWithShape(to_shape_expr(rbi.src_sizes),
                                  load->buffer->dtype, "local_src_buf", scope);
    tmp_buffers.push_back(local_buf);

    // src_offsets: loop dimensions use 0; loop-invariant dimensions keep
    // their original index (e.g. cid).
    std::vector<PrimExpr> src_offsets;
    src_offsets.reserve(load->indices.size());
    for (int i = 0; i < (int)load->indices.size(); i++)
      src_offsets.push_back(rbi.index_to_loop[i] >= 0
                                ? make_const(DataType::Int(32), 0)
                                : load->indices[i]);

    auto local_offsets = make_zero_offsets((int)rbi.src_sizes.size());

    // Hoist all reshape + brc statements.
    hoisted_stmts_.push_back(
        Evaluate(Call(DataType::Void(), Op::Get("tl.copy"),
                      {RegionND(load->buffer, src_offsets, 1, rbi.src_sizes),
                       RegionND(local_buf, local_offsets, 2, rbi.src_sizes)})));

    // Build view_shape and brc_shape aligned to the dimensionality of
    // output_ref. For each dimension in output_ref:
    //   - if it corresponds to a loop variable, brc_shape fills loop_extents[j]
    //     and view_shape fills rbi.view_shape[j] (1 for broadcast dims);
    //   - otherwise both remain 1.
    int out_dim = output_ref.buffer->shape.size();
    std::vector<int64_t> aligned_view_shape(out_dim, 1);
    std::vector<int64_t> aligned_brc_shape(out_dim, 1);
    for (int i = 0; i < out_dim; i++) {
      if (auto var = output_ref.indices[i].as<VarNode>()) {
        for (int j = 0; j < N; j++) {
          if (loop_vars[j] == var) {
            aligned_brc_shape[i] = loop_extents[j];
            aligned_view_shape[i] = rbi.view_shape[j];
            break;
          }
        }
      }
    }

    // 2. view_buf: reshape local_buf using aligned_view_shape.
    Buffer view_buf = CreateTempBufferWithShape(
        to_shape_expr(aligned_view_shape), load->buffer->dtype,
        "reshape_view_buf", scope);
    tmp_buffers.push_back(view_buf);

    auto view_offsets = make_zero_offsets(out_dim);

    hoisted_stmts_.push_back(Evaluate(
        Call(DataType::Void(), Op::Get("tl.npuir_reshape"),
             {RegionND(local_buf, local_offsets, 1, rbi.src_sizes),
              RegionND(view_buf, view_offsets, 2, aligned_view_shape)})));

    // 3. brc_buf: broadcast view_buf to the full aligned_brc_shape.
    Buffer brc_buf = CreateTempBufferWithShape(to_shape_expr(aligned_brc_shape),
                                               load->buffer->dtype, "brc_buf",
                                               scope, op_name);
    tmp_buffers.push_back(brc_buf);

    auto brc_offsets = make_zero_offsets(out_dim);
    hoisted_stmts_.push_back(
        Evaluate(Call(DataType::Void(), Op::Get("tl.npuir_brc"),
                      {RegionND(view_buf, view_offsets, 1, aligned_view_shape),
                       RegionND(brc_buf, brc_offsets, 2, aligned_brc_shape)})));

    return {brc_buf, output_ref.indices, true};
  }

  // ============================================================
  // Unified operand resolution
  // ============================================================

  // Return value semantics:
  //   nullopt           -> scalar / loop-invariant; caller uses the raw
  //   PrimExpr directly. BufferAccessInfo  -> buffer info suitable for
  //   BuildRegionCall.
  std::optional<BufferAccessInfo>
  ResolveOperand(const PrimExpr &operand, const BufferAccessInfo &output_ref,
                 std::vector<BufferAccessInfo> *tmp_bufs,
                 const std::vector<const VarNode *> &loop_vars,
                 const std::vector<int64_t> &loop_extents, Array<Stmt> *stmts,
                 const std::string &op_name = "") {

    if (IsScalarLike(operand, loop_vars))
      return std::nullopt;

    if (auto load = operand.as<BufferLoadNode>()) {
      if (!ValidBufferIndices(load->indices))
        return std::nullopt;
      auto rbi = CheckNeedsReshapeBrc(load, loop_vars, loop_extents);
      if (rbi.needed) {
        auto expanded = EmitReshapeAndBroadcast(
            load, rbi, output_ref, loop_vars, loop_extents, op_name);
        tmp_bufs->push_back(expanded);
        return expanded;
      }
      return ExtractBufferAccessInfo(operand);
    }

    // Complex sub-expression: decompose recursively; result lands in the
    // most recently added tmp_buf.
    if (!DecomposeExpression(operand, output_ref, tmp_bufs, loop_vars,
                             loop_extents, stmts))
      return std::nullopt;
    return tmp_bufs->back();
  }

  // ============================================================
  // Expression decomposition
  // ============================================================

  bool HandleUnaryExpression(const PrimExpr &expr,
                             const BufferAccessInfo &output_ref,
                             std::vector<BufferAccessInfo> *tmp_bufs,
                             const std::vector<const VarNode *> &loop_vars,
                             const std::vector<int64_t> &loop_extents,
                             Array<Stmt> *stmts) {
    DepthGuard guard(depth);

    auto *call = expr.as<CallNode>();
    std::string op_name = call->op.as<OpNode>()->name;

    auto resolved = ResolveOperand(call->args[0], output_ref, tmp_bufs,
                                   loop_vars, loop_extents, stmts, op_name);
    if (!resolved)
      return false;

    // depth > 1 means this is an intermediate sub-expression; store the
    // result in a temporary buffer.
    BufferAccessInfo output = output_ref;
    if (depth > 1) {
      output = CreateTempBuffer(*resolved);
      tmp_bufs->push_back(output);
    }

    stmts->push_back(
        BuildUnaryStmt(op_name, BuildRegionCall(*resolved, 1, 1), output));
    guard.commit();
    return true;
  }

  bool HandleBinaryExpression(std::string op_name,
                              const Array<PrimExpr> &operands,
                              const BufferAccessInfo &output_ref,
                              std::vector<BufferAccessInfo> *tmp_bufs,
                              const std::vector<const VarNode *> &loop_vars,
                              const std::vector<int64_t> &loop_extents,
                              Array<Stmt> *stmts) {
    DepthGuard guard(depth);

    auto resolved_a = ResolveOperand(operands[0], output_ref, tmp_bufs,
                                     loop_vars, loop_extents, stmts, op_name);
    auto resolved_b = ResolveOperand(operands[1], output_ref, tmp_bufs,
                                     loop_vars, loop_extents, stmts, op_name);

    // Both operands are scalar — cannot vectorize.
    if (!resolved_a && !resolved_b)
      return false;

    PrimExpr region_a =
        resolved_a ? BuildRegionCall(*resolved_a, 1, 1) : operands[0];
    PrimExpr region_b =
        resolved_b ? BuildRegionCall(*resolved_b, 1, 1) : operands[1];

    if (!resolved_a) {
      if (op_name == "Add" || op_name == "Mul" || IsCmpOp(op_name)) {
        // Commutative ops: swap operands directly.
        if (op_name == "LT")
          op_name = "GT";
        if (op_name == "LE")
          op_name = "GE";
        std::swap(region_a, region_b);
      } else {
        if (auto load = operands[0].as<BufferLoadNode>()) {
          // Loop-invariant BufferLoad on the left: wrap as a size-1 region.
          region_a =
              BuildRegionCall(ExtractBufferAccessInfo(operands[0]), 1, 1);
        } else {
          // Pure scalar Sub/Div cannot be swapped; broadcast the scalar first.
          const BufferAccessInfo &ref = *resolved_b;
          auto scalar_buf = CreateTempBuffer(ref, op_name);
          tmp_bufs->push_back(scalar_buf);
          stmts->push_back(BuildNpuirCall(
              "Broadcast", {operands[0], BuildRegionCall(scalar_buf, 2, 1)}));
          region_a = BuildRegionCall(scalar_buf, 1, 1);
        }
      }
    }

    if (!resolved_b) {
      if (auto load = operands[1].as<BufferLoadNode>()) {
        // Loop-invariant BufferLoad: wrap as a size-1 region.
        region_b = BuildRegionCall(ExtractBufferAccessInfo(operands[1]), 1, 1);
      }
      // Pure scalar (IntImm / FloatImm): use directly, no extra handling
      // needed.
    }

    const BufferAccessInfo &ref = resolved_a ? *resolved_a : *resolved_b;
    BufferAccessInfo output = output_ref;
    if (depth > 1) {
      output = CreateTempBuffer(ref, op_name);
      tmp_bufs->push_back(output);
    }

    stmts->push_back(BuildBinaryStmt(op_name, region_a, region_b, output));
    guard.commit();
    return true;
  }

  bool HandleTernaryExpression(const PrimExpr &expr,
                               const BufferAccessInfo &output_ref,
                               std::vector<BufferAccessInfo> *tmp_bufs,
                               const std::vector<const VarNode *> &loop_vars,
                               const std::vector<int64_t> &loop_extents,
                               Array<Stmt> *stmts) {
    DepthGuard guard(depth);

    auto *call = expr.as<CallNode>();
    std::string op_name = call->op.as<OpNode>()->name;

    // Scalar-like operands must be broadcast into a buffer first.
    auto ResolveTernaryOperand =
        [&](const PrimExpr &e,
            const std::string &tmp_op =
                "npuir_add") -> std::optional<BufferAccessInfo> {
      if (IsScalarLike(e, loop_vars)) {
        auto tmp = CreateTempBuffer(output_ref, tmp_op);
        tmp_bufs->push_back(tmp);
        stmts->push_back(BuildUnaryStmt("Broadcast", e, tmp));
        return tmp;
      }
      return ResolveOperand(e, output_ref, tmp_bufs, loop_vars, loop_extents,
                            stmts, op_name);
    };

    auto cond_info = ResolveTernaryOperand(call->args[0], "GT");
    auto true_info = ResolveTernaryOperand(call->args[1]);
    auto false_info = ResolveTernaryOperand(call->args[2]);

    if (!cond_info || !true_info || !false_info)
      return false;

    stmts->push_back(BuildTernaryStmt(op_name, *cond_info, *true_info,
                                      *false_info, output_ref));
    guard.commit();
    return true;
  }

  bool DecomposeExpression(const PrimExpr &expr,
                           const BufferAccessInfo &output_ref,
                           std::vector<BufferAccessInfo> *tmp_bufs,
                           const std::vector<const VarNode *> &loop_vars,
                           const std::vector<int64_t> &loop_extents,
                           Array<Stmt> *stmts) {

    if (expr.as<BufferLoadNode>()) {
      DepthGuard guard(depth);
      BufferAccessInfo src = ExtractBufferAccessInfo(expr);
      stmts->push_back(
          BuildNpuirCall("Copy", {BuildRegionCall(src, 1, 1),
                                  BuildRegionCall(output_ref, 2, 1)}));
      guard.commit();
      return true;
    }

    if (IsScalarLike(expr, loop_vars)) {
      DepthGuard guard(depth);
      stmts->push_back(BuildUnaryStmt("Broadcast", expr, output_ref));
      guard.commit();
      return true;
    }

    if (IsUnaryOp(expr))
      return HandleUnaryExpression(expr, output_ref, tmp_bufs, loop_vars,
                                   loop_extents, stmts);

    std::string op_type;
    Array<PrimExpr> operands;
    if (IsBinaryOp(expr, &op_type, &operands))
      return HandleBinaryExpression(op_type, operands, output_ref, tmp_bufs,
                                    loop_vars, loop_extents, stmts);

    if (IsTernaryOp(expr))
      return HandleTernaryExpression(expr, output_ref, tmp_bufs, loop_vars,
                                     loop_extents, stmts);

    return false;
  }

  // ============================================================
  // Loop structure collection
  // ============================================================

  struct LoopNest {
    std::vector<const ForNode *> loops; // outer -> inner
    const BufferStoreNode *store = nullptr;
  };

  // Collect consecutive parallel ForNodes starting from root until a
  // BufferStore is encountered.
  std::optional<LoopNest> CollectPerfectParallelNest(const ForNode *root) {
    LoopNest nest;
    Stmt cur = GetRef<Stmt>(root);
    while (auto f = cur.as<ForNode>()) {
      if (f->kind != ForKind::kParallel || !f->extent.as<IntImmNode>())
        return std::nullopt;
      nest.loops.push_back(f);
      cur = f->body;
    }
    nest.store = cur.as<BufferStoreNode>();
    if (!nest.store)
      return std::nullopt;
    return nest;
  }

  // ============================================================
  // Single-statement vectorization
  // ============================================================

  Stmt VectorizeSingleStatement(const ForNode *op) {
    const auto *store = op->body.as<BufferStoreNode>();

    // Store indices must be variables or constants only.
    for (const auto &idx : store->indices) {
      if (!idx.as<VarNode>() && !IsScalar(idx))
        return StmtMutator::VisitStmt_(op);
    }

    Array<Stmt> stmts;
    std::vector<BufferAccessInfo> tmp_bufs;

    // Vectorized statements are collected into stmts.
    bool ok = DecomposeExpression(
        store->value, ExtractBufferAccessInfo(op->body), &tmp_bufs,
        current_loop_vars_, current_loop_extents_, &stmts);

    if (!ok)
      return StmtMutator::VisitStmt_(op);

    return For(op->loop_var, op->min, op->extent, op->kind,
               SeqStmt::Flatten(stmts), op->thread_binding, op->annotations);
  }

  // ============================================================
  // StmtMutator overrides
  // ============================================================

  // Collect all temporary buffers allocated within the current block and
  // register them in alloc_buffers.
  Stmt VisitStmt_(const BlockNode *op) override {
    auto saved = std::move(tmp_buffers);
    tmp_buffers.clear();

    Stmt new_body = VisitStmt(op->body);

    Array<Buffer> allocs = op->alloc_buffers;
    for (const auto &buf : tmp_buffers)
      allocs.push_back(buf);
    tmp_buffers = std::move(saved);

    if (allocs.same_as(op->alloc_buffers) && new_body.same_as(op->body))
      return GetRef<Stmt>(op);

    return Block(op->iter_vars, op->reads, op->writes, op->name_hint, new_body,
                 op->init, allocs);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    if (op->kind != ForKind::kParallel)
      return StmtMutator::VisitStmt_(op);

    auto ext_imm = op->extent.as<IntImmNode>();
    if (!ext_imm)
      return StmtMutator::VisitStmt_(op);

    bool is_outermost = current_loop_vars_.empty();
    current_loop_vars_.push_back(op->loop_var.get());
    current_loop_extents_.push_back(ext_imm->value);

    Stmt result;
    if (!op->body.as<BufferStoreNode>()) {
      // Nested loop: recursively process the body.
      Stmt new_body = VisitStmt(op->body);
      result = For(op->loop_var, op->min, op->extent, op->kind, new_body,
                   op->thread_binding, op->annotations);
    } else {
      // Innermost loop: attempt vectorization.
      result = VectorizeSingleStatement(op);
    }

    current_loop_vars_.pop_back();
    current_loop_extents_.pop_back();

    // After processing the outermost parallel loop, prepend all hoisted
    // statements before it.
    if (is_outermost && !hoisted_stmts_.empty()) {
      Array<Stmt> all;
      for (auto &s : hoisted_stmts_)
        all.push_back(s);
      all.push_back(result);
      hoisted_stmts_.clear();
      return SeqStmt::Flatten(all);
    }

    return result;
  }
};

class LoopVectorize : public StmtMutator {
private:
  inline static std::unordered_set<std::string> CandidateVectorizationOps = {
      "tl.copy",      "tl.npuir_exp", "tl.npuir_abs",    "tl.npuir_add",
      "tl.npuir_mul", "tl.npuir_sub", "tl.npuir_div",    "tl.npuir_sigmoid",
      "tl.npuir_brc", "tl.npuir_cmp", "tl.npuir_select",
  };

  bool IsScalar(const PrimExpr &expr) {
    return expr.as<IntImmNode>() || expr.as<FloatImmNode>();
  }

  // return a integer if the input is able to be converted, else return null
  std::optional<int64_t>
  TryGetConstIntValue(const tvm::PrimExpr &expr,
                      tvm::arith::Analyzer *analyzer = nullptr) {
    // Method 1: Direct check for IntImmNode
    if (const tvm::tir::IntImmNode *int_imm = expr.as<tvm::tir::IntImmNode>()) {
      return int_imm->value;
    }

    // Method 2: Use arith::Analyzer for simplification
    tvm::arith::Analyzer local_analyzer;
    tvm::arith::Analyzer *used_analyzer = analyzer ? analyzer : &local_analyzer;

    tvm::PrimExpr simplified = used_analyzer->Simplify(expr);
    if (const tvm::tir::IntImmNode *simplified_int =
            simplified.as<tvm::tir::IntImmNode>()) {
      return simplified_int->value;
    }

    // Method 3: Try to get constant bounds
    tvm::arith::ConstIntBound bound = used_analyzer->const_int_bound(expr);
    if (bound->min_value == bound->max_value) {
      return bound->min_value;
    }

    // Method 4: Handle common expression patterns
    if (const tvm::tir::SubNode *sub = expr.as<tvm::tir::SubNode>()) {
      if (sub->a.same_as(sub->b)) {
        return 0;
      }
    }

    return std::nullopt;
  }

  // return a Call to tl.region -> str: T.region(buffer[offset_expr], region_id,
  // size)
  PrimExpr BuildRegionCall(Buffer buffer,
                           const std::vector<PrimExpr> &offset_expr,
                           int region_id, const std::vector<size_t> &size) {
    PrimExpr buffer_load = BufferLoad(buffer, {offset_expr});
    std::vector<PrimExpr> args_vec = {
        buffer_load,
        make_const(DataType::Int(32), region_id),
    };
    for (auto x : size) {
      args_vec.push_back(make_const(DataType::Int(32), x));
    }
    Array<PrimExpr> args(args_vec);
    Op region_op = Op::Get("tl.region");
    return Call(DataType::Handle(), region_op, args);
  }

  /**
   * Find the index of the offset containing the loop variable.
   * Returns -1 if not found, -2 if found in multiple offsets or indirect mem
   * access.
   */
  int FindLoopVarInOffsets(const std::vector<PrimExpr> &offsets,
                           const VarNode *loop_var) {
    int found_dim = -1;

    for (int i = 0; i < static_cast<int>(offsets.size()); ++i) {
      bool indirect_found = false;

      class LoopVarVisitor : public ExprVisitor {
      public:
        const VarNode *target_var;
        bool found = false;   // Whether appeared directly
        bool &indirect_found; // Whether appeared indirectly
        bool indirect_scope =
            false; // Status flag: whether the visitor is in a BufferLoad

        LoopVarVisitor(const VarNode *var, bool &indirect_flag)
            : target_var(var), indirect_found(indirect_flag) {}

        void VisitExpr_(const VarNode *op) override {
          if (op == target_var) {
            if (indirect_scope) {
              indirect_found = true;
            } else {
              found = true;
            }
          }
          ExprVisitor::VisitExpr_(op);
        }

        void VisitExpr_(const BufferLoadNode *op) override {
          // Visit a BufferLoad, set the status flag - indirect_scope
          bool saved_scope = indirect_scope;
          indirect_scope = true;
          for (const auto &index : op->indices) {
            VisitExpr(index);
          }
          // Reset the status flag
          indirect_scope = saved_scope;
        }
      } visitor(loop_var, indirect_found);

      // Check each offset
      visitor(offsets[i]);

      if (indirect_found) {
        // Indirect mem access detected.
        return -2;
      }

      if (visitor.found) {
        if (found_dim == -1) {
          // Record the first found
          found_dim = i;
        } else {
          // Found multiple offsets associated with the loop var
          return -2;
        }
      }
    }
    // Returns the unique offset associated with the loop var
    return found_dim;
  }

  Stmt SplitStmtToIndependentForNode(const ForNode *forNode, const Stmt &stmt) {
    Var new_loop_var =
        Var(forNode->loop_var->name_hint + "_split", forNode->loop_var->dtype);

    Map<Var, PrimExpr> var_map;
    var_map.Set(forNode->loop_var, new_loop_var);
    Stmt new_body = Substitute(stmt, var_map);

    return For(new_loop_var, forNode->min, forNode->extent, forNode->kind,
               new_body, forNode->thread_binding, forNode->annotations,
               forNode->span);
  }

  struct RegionInfo {
    Buffer buffer;
    std::vector<PrimExpr> offsets;
    int regionId;
    std::vector<size_t> sizes;
  };

  std::optional<RegionInfo>
  ParseRegionCall(const PrimExpr &expr, arith::Analyzer *analyzer = nullptr) {
    // 1. must be a CallNode
    const auto *call = expr.as<CallNode>();
    if (!call)
      return std::nullopt;

    // 2. must be tl.region
    const auto *op_node = call->op.as<OpNode>();
    if (!op_node)
      return std::nullopt;
    static const auto *region_op = Op::Get("tl.region").as<OpNode>();
    if (op_node != region_op)
      return std::nullopt;

    // 3. check the number of args and extract attributes
    if (call->args.size() < 3)
      return std::nullopt;

    const auto *buffer_load = call->args[0].as<BufferLoadNode>();
    if (!buffer_load)
      return std::nullopt;

    Buffer buffer = buffer_load->buffer;
    Array<PrimExpr> offsets = buffer_load->indices;
    int d = buffer->shape.size();

    // 4. check the shapes
    if (offsets.size() != d)
      return std::nullopt;
    if (call->args.size() != d + 2)
      return std::nullopt;

    // 5. try to extract region id & sizes
    auto regionId_val = TryGetConstIntValue(call->args[1], analyzer);
    if (!regionId_val)
      return std::nullopt;
    int regionId = static_cast<int>(*regionId_val);

    std::vector<size_t> sizes;
    for (int i = 2; i < d + 2; ++i) {
      auto size_val = TryGetConstIntValue(call->args[i], analyzer);
      if (!size_val)
        return std::nullopt;
      sizes.push_back(static_cast<size_t>(*size_val));
    }

    // post-processing: convert offsets into vector
    std::vector<PrimExpr> offsets_vec(offsets.begin(), offsets.end());

    return RegionInfo{std::move(buffer), std::move(offsets_vec), regionId,
                      std::move(sizes)};
  }

  bool CheckContinuity(const Buffer &buffer,
                       const std::vector<PrimExpr> &offsets,
                       const std::vector<size_t> &sizes, int loop_var_dim,
                       arith::Analyzer *analyzer = nullptr) {
    arith::Analyzer local_analyzer;
    arith::Analyzer *used_analyzer = analyzer ? analyzer : &local_analyzer;

    // 1. check the sizes to ensure the continuity of mem access
    int d = buffer->shape.size();
    for (int i = 0; i < d; ++i) {
      const PrimExpr &size_expr = make_const(DataType::Int(32), sizes[i]);
      if (i <= loop_var_dim) {
        // size of current and higher dimension should be 1; otherwise,
        // something wrong happened
        if (!used_analyzer->CanProveEqual(size_expr, 1))
          return false;
      } else {
        // lower dimensions must participate in the operation completely;
        // otherwise, the current dimension cannot be vectorized. The first
        // condition: The size is equal to the size of the corresponding
        // dimension.
        if (!used_analyzer->CanProveEqual(size_expr, buffer->shape[i]))
          return false;
        // The second condition: offset equals zero
        if (!used_analyzer->CanProveEqual(offsets[i], 0))
          return false;
      }
    }

    return true;
  }

  std::optional<PrimExpr>
  AnalyzeNewOffset(const std::vector<PrimExpr> &offsets,
                   const VarNode *loop_var, int loop_var_dim, int start,
                   arith::Analyzer *analyzer = nullptr) {
    arith::Analyzer local_analyzer;
    arith::Analyzer *used_analyzer = analyzer ? analyzer : &local_analyzer;

    // check the expression about loop var to ensure the continuity of mem
    // access
    Array<PrimExpr> res = arith::DetectLinearEquation(offsets[loop_var_dim],
                                                      {GetRef<Var>(loop_var)});
    if (res.empty() || !used_analyzer->CanProveEqual(res[0], 1)) {
      // Not a linear expression or the coefficient of the first term is not 1
      // -> disable to vectorize
      return std::nullopt;
    }

    PrimExpr new_offset = res[1] + make_const(res[1].dtype(), start);

    return used_analyzer->Simplify(new_offset);
  }

  std::optional<PrimExpr>
  TryVectorizeRegionInLoop(const PrimExpr &expr, const VarNode *loop_var,
                           size_t start, size_t loop_count,
                           arith::Analyzer *analyzer = nullptr) {
    arith::Analyzer local_analyzer;
    arith::Analyzer *used_analyzer = analyzer ? analyzer : &local_analyzer;

    // Step 1: try to extract region info
    auto info = ParseRegionCall(expr, used_analyzer);
    if (!info)
      return std::nullopt;

    // Step 2: check the loop var
    int loop_var_dim = FindLoopVarInOffsets(info->offsets, loop_var);
    // -2 -> multiple or indirect found -> dispersed mem access -> disable to
    // vectorize
    if (loop_var_dim == -2)
      return std::nullopt;
    // -1 -> not found -> invariant -> enable to vectorize but no need change
    if (loop_var_dim == -1) {
      return BuildRegionCall(info->buffer, info->offsets, info->regionId,
                             info->sizes);
    }

    // Step 3: check the continuity of mem access
    if (!CheckContinuity(info->buffer, info->offsets, info->sizes, loop_var_dim,
                         used_analyzer)) {
      return std::nullopt;
    }

    // Step 4: try to analyze the offset after vectorization
    auto new_offset = AnalyzeNewOffset(info->offsets, loop_var, loop_var_dim,
                                       start, used_analyzer);
    if (!new_offset)
      return std::nullopt;

    // Step 5: build and return vectorized region
    info->offsets[loop_var_dim] = *new_offset;
    info->sizes[loop_var_dim] = loop_count;

    return BuildRegionCall(info->buffer, info->offsets, info->regionId,
                           info->sizes);
  }

  Stmt VectorizeForBody(const ForNode *forNode, const Stmt &stmt) {
    arith::Analyzer analyzer;

    if (const auto *alloc = stmt.as<AllocateNode>())
      return stmt;

    const auto *evaluate = stmt.as<EvaluateNode>();
    if (!evaluate)
      return SplitStmtToIndependentForNode(forNode, stmt);

    const auto *call = evaluate->value.as<tvm::tir::CallNode>();
    if (!call)
      return SplitStmtToIndependentForNode(forNode, stmt);

    bool flag_op_supported = false;
    if (const auto *op_node = call->op.as<tvm::OpNode>()) {
      tvm::Op op = tvm::GetRef<tvm::Op>(op_node);
      flag_op_supported =
          static_cast<bool>(CandidateVectorizationOps.count(op->name));
    }
    if (!flag_op_supported)
      return SplitStmtToIndependentForNode(forNode, stmt);

    auto loop_min_value = TryGetConstIntValue(forNode->min, &analyzer);
    auto loop_extent_value = TryGetConstIntValue(forNode->extent, &analyzer);
    if (!loop_min_value || !loop_extent_value)
      return SplitStmtToIndependentForNode(forNode, stmt);

    auto loop_var_node_ptr = forNode->loop_var.get();
    std::vector<PrimExpr> new_regions;
    for (const auto &region : call->args) {
      if (IsScalar(region) || region.as<StringImmNode>()) {
        new_regions.push_back(region);
        continue;
      }
      if (region.as<BufferLoadNode>()) {
        new_regions.push_back(region);
        continue;
      }
      auto new_region =
          TryVectorizeRegionInLoop(region, loop_var_node_ptr, *loop_min_value,
                                   *loop_extent_value, &analyzer);
      if (!new_region)
        return SplitStmtToIndependentForNode(forNode, stmt);
      new_regions.push_back(*new_region);
    }

    Array<PrimExpr> args(new_regions);
    PrimExpr new_call = Call(DataType::Void(), call->op, args);
    return Evaluate(new_call);
  }

  Stmt VectorizeForBody(const ForNode *forNode, const SeqStmt &seqStmt) {
    std::vector<Stmt> result_vec;
    result_vec.reserve(seqStmt->size());
    for (size_t i = 0; i < seqStmt->size(); ++i) {
      result_vec.push_back(VectorizeForBody(forNode, seqStmt[i]));
    }
    Array<Stmt> result_arr(result_vec);
    return SeqStmt::Flatten(SeqStmt(result_arr));
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // try vectorize only when marked as parallel
    if (op->kind != ForKind::kParallel) {
      return StmtMutator::VisitStmt_(op);
    }

    // recursive processing
    Stmt body = op->body;
    if (const auto *forNode = body.as<ForNode>()) {
      body = VisitStmt_(forNode);
    }

    if (const auto *seqStmtNode = body.as<SeqStmtNode>()) {
      return VectorizeForBody(op, GetRef<SeqStmt>(seqStmtNode));
    } else if (const auto *stmtNode = body.as<EvaluateNode>()) {
      return VectorizeForBody(op, GetRef<Evaluate>(stmtNode));
    } else {
      return StmtMutator::VisitStmt_(op);
    }
  }
};

using namespace tir::transform;

tvm::transform::Pass NpuLoopVectorize() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto *new_pf = f.CopyOnWrite();
    new_pf->body = LoopDecompose()(std::move(new_pf->body));
    new_pf->body = LoopVectorize()(std::move(new_pf->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.NpuLoopVectorize", {});
}

TVM_REGISTER_GLOBAL("tl.transform.NpuLoopVectorize")
    .set_body_typed(NpuLoopVectorize);

} // namespace tl
} // namespace tvm
