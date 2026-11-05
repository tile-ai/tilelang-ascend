// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "operator.h"
#include "ascend.h"

#include <unordered_map>

namespace tvm::tl::a2a3 {
namespace {
std::unordered_map<std::string, OperatorParser> &Parsers() {
  static std::unordered_map<std::string, OperatorParser> parsers;
  return parsers;
}

class TileOperatorAdapter : public Operator {
public:
  explicit TileOperatorAdapter(TileOperator op) : op_(std::move(op)) {}

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final {
    ::tvm::tl::LowerArgs args;
    args.target = T.target;
    args.thread_bounds = T.thread_bounds;
    args.thread_index = T.thread_var;
    args.layout_map = T.layout_map;
    args.buffer_remap = T.buffer_remap;
    args.add_workspace = T.AddWorkspace;
    return op_->Lower(args, analyzer);
  }

  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) final {
    arith::Analyzer analyzer;
    return op_->InferLayout(
        ::tvm::tl::LayoutInferArgs{T.target, T.thread_bounds, T.layout_map,
                                   &analyzer, T.buffer_remap},
        level);
  }

private:
  TileOperator op_;
};
} // namespace

void RegisterOperatorParser(const String &op_name, OperatorParser parser) {
  ICHECK(Parsers().emplace(std::string(op_name), std::move(parser)).second)
      << "Duplicate A2/A3 operator parser for " << op_name;
}

std::unique_ptr<Operator> ParseOperator(const Call &call,
                                        const BufferMap &vmap) {
  if (auto op = call->op.as<Op>()) {
    auto it = Parsers().find(std::string(op.value()->name));
    if (it != Parsers().end())
      return it->second(call, vmap);
    if (op.value().same_as(AscendCopy::Get()) ||
        op.value().same_as(AscendAtomicAdd::Get())) {
      return std::make_unique<TileOperatorAdapter>(
          ::tvm::tl::ParseOperator(call));
    }
  }
  return nullptr;
}

std::unique_ptr<Operator> ParseOperator(const Stmt &stmt,
                                        const BufferMap &vmap) {
  if (const auto *evaluate = stmt.as<EvaluateNode>()) {
    if (auto call = evaluate->value.as<Call>())
      return ParseOperator(call.value(), vmap);
  }
  return nullptr;
}

Stmt Operator::Lower(const LowerArgs &, arith::Analyzer *) const {
  LOG(FATAL) << "Not Implemented Lower method.";
  return Evaluate(0);
}
Stmt Operator::Canonialize(const CanonializeArgs &, arith::Analyzer *) const {
  return {};
}
LayoutMap Operator::InferLayout(const LayoutInferArgs &, InferLevel) {
  return {};
}
} // namespace tvm::tl::a2a3
