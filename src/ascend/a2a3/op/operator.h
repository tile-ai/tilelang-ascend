// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#ifndef TVM_TL_ASCEND_A2A3_OP_OPERATOR_H_
#define TVM_TL_ASCEND_A2A3_OP_OPERATOR_H_

#include <functional>
#include <memory>

#include "op/operator.h"

namespace tvm::tl::a2a3 {

using namespace tirx;
using namespace ffi;

// TA 0deb8b93 src/op/op.h.  Keep stateful legacy inference independent of
// main's TileOperator ABI; only migrated A2/A3 TileOperators cross the adapter.
struct LowerArgs {
  Target target;
  Range thread_bounds;
  Var thread_var;
  AddWorkspaceCallback AddWorkspace;
  LayoutMap layout_map;
  Map<Buffer, Buffer> buffer_remap;
  bool disable_tma_lower;
  int resource_scope;
};

struct LayoutInferArgs {
  Target target;
  Range thread_bounds;
  LayoutMap layout_map;
  Map<Buffer, Buffer> buffer_remap;
};

struct CanonializeArgs {
  Target target;
};

class Operator {
public:
  virtual Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const;
  virtual Stmt Canonialize(const CanonializeArgs &T,
                           arith::Analyzer *analyzer) const;
  virtual LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level);
  virtual ~Operator() = default;
};

// A C++ registry preserves unique ownership without exporting raw void* via
// the new FFI. Parsers receive the full Call so generic-language annotations
// can be translated explicitly into the original TA positional arguments.
using OperatorParser =
    std::function<std::unique_ptr<Operator>(const Call &, const BufferMap &)>;
void RegisterOperatorParser(const String &op_name, OperatorParser parser);
std::unique_ptr<Operator> ParseOperator(const Call &call,
                                        const BufferMap &vmap);
std::unique_ptr<Operator> ParseOperator(const Stmt &stmt,
                                        const BufferMap &vmap);

} // namespace tvm::tl::a2a3
#endif // TVM_TL_ASCEND_A2A3_OP_OPERATOR_H_
