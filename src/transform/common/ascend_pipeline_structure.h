// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#pragma once

#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <vector>

namespace tvm {
namespace tl {

// Preserve the injector's allocation registration and child visitation order.
struct AscendPipelineElement {
  tir::Buffer allocation;
  tir::Stmt statement;
  Array<Integer> source_path;
};

inline std::vector<AscendPipelineElement>
CollectAscendPipelineGroups(const tir::SeqStmt &sequence) {
  std::vector<AscendPipelineElement> result;
  for (size_t i = 0; i < sequence->seq.size(); ++i) {
    const auto *realize = sequence->seq[i].as<tir::BlockRealizeNode>();
    if (realize && tir::is_one(realize->predicate) &&
        realize->block->body.as<tir::SeqStmtNode>()) {
      ICHECK(realize->block->match_buffers.empty());
      for (const tir::Buffer &buffer : realize->block->alloc_buffers) {
        result.push_back({buffer, tir::Stmt(), {Integer(i)}});
      }
      const auto *nested = realize->block->body.as<tir::SeqStmtNode>();
      for (size_t j = 0; j < nested->seq.size(); ++j) {
        result.push_back(
            {tir::Buffer(), nested->seq[j], {Integer(i), Integer(j)}});
      }
    } else {
      result.push_back({tir::Buffer(), sequence->seq[i], {Integer(i)}});
    }
  }
  return result;
}

} // namespace tl
} // namespace tvm
