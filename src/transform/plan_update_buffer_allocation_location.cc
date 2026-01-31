/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
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
 * \brief Planning where buffers to be allocated and update the AST.
 * \file plan_update_buffer_allocation_location.cc
 */

#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/var.h>

#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {
using namespace tir;

class CollectManagedAllocations : public StmtExprVisitor {
 public:
  void VisitStmt_(const BlockNode* op) final {
    for (const auto& buf : op->alloc_buffers) {
      managed_allocations.insert(buf->data.get());
    }
    for (const auto& buf : op->match_buffers) {
      managed_allocations.insert(buf->buffer->data.get());
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  /*! \brief Buffers that are allocated outside of the BlockNode, and should not be moved by
   * BufferAllocationLocator. */
  std::unordered_set<const VarNode*> managed_allocations;
};

/*! \brief Collect the allocate buffer order. */
class BufferAllocateOrderCollector : public StmtExprVisitor {
 public:
  static Array<Buffer> Collect(const PrimFunc& func) {
    BufferAllocateOrderCollector collector;
    for (const auto& kv : func->buffer_map) {
      collector.buffer_alloc_recorder_.push_back(kv.second);
    }
    collector(func->body);
    return std::move(collector.buffer_alloc_recorder_);
  }

 private:
  bool find(const Buffer& buf) {
    return std::find(buffer_alloc_recorder_.begin(), buffer_alloc_recorder_.end(), buf) !=
           buffer_alloc_recorder_.end();
  }

  void VisitStmt_(const BlockNode* op) final {
    for (const Buffer& buffer : op->alloc_buffers) {
      buffer_alloc_recorder_.push_back(buffer);
    }
    // Also visit match_buffers to collect constant buffers associated with AllocateConst nodes.
    // These buffers only appear in read and match_buffer regions.
    for (const auto& region : op->match_buffers) {
      if (!find(region->source->buffer)) {
        buffer_alloc_recorder_.push_back(region->source->buffer);
      }
    }

    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    if (!find(op->buffer)) {
      buffer_alloc_recorder_.push_back(op->buffer);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    if (!find(op->buffer)) {
      buffer_alloc_recorder_.push_back(op->buffer);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  /*! \brief The buffer allocated order recorder. */
  Array<Buffer> buffer_alloc_recorder_;
};

class BufferAllocationLocator : public StmtExprMutator {
 public:
  explicit BufferAllocationLocator(const PrimFunc& func) {
    Map<Buffer, Optional<Stmt>> buffer_lca = DetectBufferAccessLCA(func);
    // The buffer_alloc_recorder Array is used to keep the buffer allocation order
    // since the buffer_lca Map is unordered.
    Array<Buffer> buffer_alloc_recorder = BufferAllocateOrderCollector::Collect(func);
    std::unordered_set<const VarNode*> arg_buffer_vars;
    CollectManagedAllocations collector;
    collector(func->body);
    managed_allocations_ = collector.managed_allocations;

    for (const auto& kv : func->buffer_map) {
      const Buffer& buffer = kv.second;
      arg_buffer_vars.emplace(buffer->data.get());
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    // Analyze gemm calls in the entire function body to identify buffers that require special handling
    AnalyzeGemmCalls(func->body);
    // create buffers to be allocated at each stmts
    for (const auto& buffer : buffer_alloc_recorder) {
      auto it = buffer_lca.find(buffer);
      if (it != buffer_lca.end()) {
        const StmtNode* stmt = (*it).second.get();
        if (arg_buffer_vars.count(buffer->data.get())) {
          continue;
        }
        if (managed_allocations_.count(buffer->data.get())) {
          
          alloc_buffers_[stmt].push_back(buffer);
        }
        buffer_data_to_buffer_.Set(buffer->data, buffer);
      }
    }
  }

 private:

  struct GemmBufferInfo {
    bool is_gemm_buffer = false;
    bool without_init = true;
  };
  
  void AnalyzeGemmCalls(Stmt stmt) {
    class GemmCallAnalyzer : public StmtExprVisitor {
    public:
      void VisitExpr_(const CallNode* op) final {
        if (op->op.defined()) {
          std::string op_name;
          if (const OpNode* op_node = op->op.as<OpNode>()) {
            op_name = op_node->name;
            // Check if this is a dot product operation (gemm variant)
            if (op_name.find("dot") != std::string::npos) {
              // Default assumption: this gemm has initC=False or initC=(k==0)
              bool without_init = true;
              if (op->args.size() >= 4) {
                const PrimExpr& init_arg = op->args[3];
                std::ostringstream oss;
                oss << init_arg;
                std::string init_str = oss.str();

                // Check if initC is explicitly set to True
                if (init_str.find("True") != std::string::npos) {
                  without_init = false;
                }

                // Extract the buffer argument (typically the 3rd argument: C in gemm(A,B,C,initC))
                const PrimExpr& buffer_arg = op->args[2];
                const VarNode* buffer_var = nullptr;

                // Helper class to find the buffer variable from the expression
                class VarFinder : public ExprVisitor {
                public:
                  const VarNode* found_var = nullptr;

                  // Visit VarNode - when the gemm operation directly uses the buffer variable without indices
                  // Example: T.gemm(A, B, C_local, initC=(k == 0))
                  //                      ^^^^^^^ - C_local is passed as a VarNode
                  void VisitExpr_(const VarNode* op) final {
                    if (!found_var) {
                        found_var = op;
                    }
                  }

                  // Visit BufferLoadNode - when the gemm operation accesses a specific element/slice of the buffer  
                  // Example: T.gemm(A[i, k], B[k, j], C_local[i, j], initC=(k == 0))
                  //                                 ^^^^^^^^^^^^^^^ - C_local[i, j] is a BufferLoadNode
                  void VisitExpr_(const BufferLoadNode* op) final {
                    found_var = op->buffer->data.get();
                  }
                };
                
                VarFinder finder;
                finder(buffer_arg);
                buffer_var = finder.found_var;

                if (buffer_var) {
                  // Store gemm buffer information for later use
                  GemmBufferInfo info;
                  info.is_gemm_buffer = true;
                  info.without_init = without_init;
                  buffer_gemm_info[buffer_var] = info;
                } else {
                  ICHECK(false) << "Could not extract buffer var from gemm arg";
                }
              } else {
                ICHECK(false) << "Gemm call has less than 3 arguments";
              }
            }
          }
        }
        StmtExprVisitor::VisitExpr_(op);
      }
      // Map buffer variables to their gemm information
      std::unordered_map<const VarNode*, GemmBufferInfo> buffer_gemm_info;
    };
    
    GemmCallAnalyzer analyzer;
    analyzer(stmt);
    buffer_gemm_info_ = analyzer.buffer_gemm_info;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    current_for_depth++;
    // For loop num
    for_depth++;
    auto it = alloc_buffers_.find(op);
    if (it == alloc_buffers_.end()) {
      merge_depth++;
      current_for_depth--;
      return StmtMutator::VisitStmt_(op);
    }
    for (const Buffer& buf : it->second) {
      buffer_data_to_buffer_.Set(buf->data, buf);
    }

    auto node = Downcast<For>(StmtMutator::VisitStmt_(op));
    current_for_depth--;
    Array<Buffer> new_block_alloc_bufs;
    for (const Buffer& buf : it->second) {
      if (managed_allocations_.count(buf->data.get())) {
        bool is_gemm_without_init = false;
        auto gemm_it = buffer_gemm_info_.find(buf->data.get());
        if (gemm_it != buffer_gemm_info_.end() && 
            gemm_it->second.is_gemm_buffer && 
            gemm_it->second.without_init) {
          is_gemm_without_init = true;
        }
        if (is_gemm_without_init) {
          need_extract = true;
          outer_buffers.push_back(buf);
          if (!multi_for_loop) {
            alloc_buffers.push_back(buf);
          }
        } else {
          if (current_for_depth == (for_depth - 2 - merge_depth) && multi_for_loop) {
            outer_buffers.push_back(buf);
          }
          buffer_data_to_buffer_.erase(buf->data);
          new_block_alloc_bufs.push_back(buf);
        }
      }
    }
    if (need_extract) {
      // If it's the innermost loop, just rewrite the alloc buffer that we need.
      if (new_block_alloc_bufs.size() && current_for_depth == for_depth - 1 - merge_depth && for_depth != 1) {
        multi_for_loop = true;
        node.CopyOnWrite()->body = InjectOpaqueBlock(node->body, new_block_alloc_bufs);
      } else if (outer_buffers.size() && current_for_depth == (for_depth - 2 - merge_depth) && multi_for_loop) {
        // Rewrite the gemm-without-init-related alloc_buffer out one loop level
        node.CopyOnWrite()->body = InjectOpaqueBlock(node->body, outer_buffers);
      } else if (new_block_alloc_bufs.size() && current_for_depth == 0 && !multi_for_loop) {
        // If there is only one loop, just rewrite the alloc_buffer that we need.
        node.CopyOnWrite()->body = InjectOpaqueBlock(node->body, new_block_alloc_bufs);
      }
      if (multi_for_loop) {
        // When there are multi for loops, we don't need reserve alloc_buffers for the outermost layer.
        alloc_buffers.clear();
      }
    } else {
      if (new_block_alloc_bufs.size()) {
        node.CopyOnWrite()->body = InjectOpaqueBlock(node->body, new_block_alloc_bufs);
      }       
    }
    
    return std::move(node);
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    ICHECK(!op->init.defined());
    auto it = alloc_buffers_.find(op);
    if (it != alloc_buffers_.end()) {
      for (const Buffer& buf : it->second) {
        alloc_buffers.push_back(buf);
        buffer_data_to_buffer_.Set(buf->data, buf);
      }
    }
    for (const MatchBufferRegion match_buffer : op->match_buffers) {
      const Var& target_var = match_buffer->buffer->data;
      const Var& source_var = match_buffer->source->buffer->data;
      ICHECK(buffer_data_to_buffer_.count(source_var));
      buffer_data_to_buffer_.Set(target_var, match_buffer->buffer);
    }
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<BlockNode>();
    ICHECK(op != nullptr);

    // No longer consider buffers created by match_buffer inside the block when updating access
    // region.
    for (const MatchBufferRegion match_buffer : op->match_buffers) {
      const Var& target_var = match_buffer->buffer->data;
      buffer_data_to_buffer_.erase(target_var);
    }
    // No longer consider buffers allocated inside the block when updating access region.
    if (it != alloc_buffers_.end()) {
      for (const Buffer& buf : it->second) {
        buffer_data_to_buffer_.erase(buf->data);
      }
    }

    ObjectPtr<BlockNode> n = CopyOnWrite(op);
    n->alloc_buffers = std::move(alloc_buffers);
    // Erase buffer allocated inside the block from access region.
    n->reads = RemoveRedundantBufferRegion(n->reads);
    n->writes = RemoveRedundantBufferRegion(n->writes);
    return Stmt(n);
  }

  Stmt VisitStmt_(const BufferRealizeNode* op) final {
    ICHECK(false) << "Internal Error: BufferRealizeNode is not allowed in TensorIR.";
    throw;
  }

  Stmt InjectOpaqueBlock(Stmt body, const Array<Buffer>& alloc_buffers) {
    ICHECK(!alloc_buffers.empty());
    Block opaque_block(/*iter_vars=*/{},
                       /*reads=*/{},
                       /*writes=*/{},
                       /*name_hint=*/"",
                       /*body=*/std::move(body),
                       /*init=*/NullOpt,
                       /*alloc_buffers=*/alloc_buffers);
    ObjectPtr<BlockNode> n = CopyOnWrite(opaque_block.get());
    Array<Array<BufferRegion>> access =
        GetBlockReadWriteRegion(opaque_block, buffer_data_to_buffer_);
    n->reads = access[0];
    n->writes = access[1];
    BlockRealize realize({}, Bool(true), Block(n));
    return std::move(realize);
  }

  Array<BufferRegion> RemoveRedundantBufferRegion(const Array<BufferRegion>& region) const {
    Array<BufferRegion> result;
    for (const BufferRegion& buffer_region : region) {
      if (buffer_data_to_buffer_.count(buffer_region->buffer->data)) {
        result.push_back(buffer_region);
      }
    }
    return result;
  }
  std::unordered_map<const VarNode*, GemmBufferInfo> buffer_gemm_info_;
  /*! \brief The map from stmt to the buffers to be allocated under it. */
  std::unordered_map<const StmtNode*, Array<Buffer>> alloc_buffers_;
  /*! \brief The buffer already allocated during recursive visiting. */
  Map<Var, Buffer> buffer_data_to_buffer_;
  /*! \brief Buffers that are allocated within a BlockNode, and may be moved. */
  std::unordered_set<const VarNode*> managed_allocations_;
  Array<Buffer> alloc_buffers;
  int current_for_depth = 0;
  int for_depth = 0;
  int merge_depth = 0;
  Array<Buffer> outer_buffers;
  bool multi_for_loop = false;
  bool need_extract = false;
};

PrimFunc PlanAndUpdateBufferAllocationLocation(PrimFunc func) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(func)) {
    auto fptr = func.CopyOnWrite();
    BufferAllocationLocator locator(func);
    fptr->body = locator(fptr->body);
    return func;
  } else {
    return func;
  }
}

using namespace tir::transform;

namespace transform {

Pass PlanAndUpdateBufferAllocationLocation() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return tl::PlanAndUpdateBufferAllocationLocation(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.PlanAndUpdateBufferAllocationLocation", {});
}

TVM_REGISTER_GLOBAL("tl.transform.PlanAndUpdateBufferAllocationLocation")
    .set_body_typed(PlanAndUpdateBufferAllocationLocation);

}  // namespace transform

}  // namespace tl
}  // namespace tvm
