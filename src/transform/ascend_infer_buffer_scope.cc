// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include <tvm/ir/op.h>

#include <tvm/tir/transform.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/analysis.h>
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>

#include <string>
#include <unordered_map>
#include <vector>
#include <set>
#include <utility>
#include <regex>
#include <iostream>
namespace tvm {
namespace tir {

class ScopeCorrector : public StmtExprMutator {
public:
  explicit ScopeCorrector() {}

  PrimFunc Correct(PrimFunc func) {
    collector_ = std::make_unique<BufferUseCollector>();
    collector_->operator()(func->body);
    
    collector_->BuildHandleAllocMapping(func->body);
    
    InferCorrectScopes();
    
    auto new_body = this->operator()(func->body);
    
    PrimFuncNode* fptr = func.CopyOnWrite();
    fptr->body = new_body;
    
    // PrintDebugInfo();
    
    return func;
  }

private:
  struct BufferUseInfo {
    bool used_in_cube = false;
    bool used_in_vector = false;
    std::set<int> gemm_positions;
    std::set<std::string> func_names;
    std::vector<const CallNode*> call_sites;
  };

  struct BufferAllocationInfo {
    const AllocateNode* alloc_node = nullptr;
    const VarNode* handle_var = nullptr;
    std::string original_scope;
    std::string corrected_scope;
    bool is_data_bound = false;
    bool from_block_alloc = false;
  };

  class BufferUseCollector : public StmtExprVisitor {
   public:
    std::vector<std::pair<const VarNode*, const VarNode*>> ascend_copy_buffer_pairs_;
    std::unordered_map<const VarNode*, BufferUseInfo> buffer_use_info;
    std::unordered_map<const VarNode*, BufferAllocationInfo> alloc_info;
    std::unordered_map<const VarNode*, const AllocateNode*> handle_to_alloc;

    static bool IsGEMMFunction(const std::string &func_name) {
      return IsGEMMInternal(ToLower(func_name));
    }

    static bool IsVectorFunction(const std::string &func_name) {
      std::string lower_name = ToLower(func_name);

      if (IsGEMMInternal(lower_name))
        return false;

      static const std::vector<std::string> kVectorKeywords = {"copy", "memcpy",
                                                               "dma"};
      return !ContainsAny(lower_name, kVectorKeywords);
    }

    static std::string GetPtrStorageScope(Var buffer_var) {
      if (auto* ptr_type = buffer_var->type_annotation.as<PointerTypeNode>()) {
        return ptr_type->storage_scope;
      }
      return "";
    }
    
    void BuildHandleAllocMapping(const Stmt& stmt) {
      class MappingBuilder : public StmtExprVisitor {
       public:
        MappingBuilder(BufferUseCollector* parent) : parent_(parent) {}
        
        void VisitStmt_(const BlockNode* op) override {     
          for (const Buffer& buffer : op->alloc_buffers) {
            const VarNode* handle_var = buffer->data.get();
            
            BufferAllocationInfo info;
            info.alloc_node = nullptr;
            info.handle_var = handle_var;
            info.original_scope = BufferUseCollector::GetPtrStorageScope(buffer->data);
            info.is_data_bound = false;
            info.from_block_alloc = true;
            
            parent_->alloc_info[handle_var] = info;
            parent_->handle_to_alloc[handle_var] = nullptr;
          }
          
          StmtExprVisitor::VisitStmt_(op);
        }
        
        void VisitStmt_(const AllocateNode* op) override {
          
          const VarNode* handle_var = op->buffer_var.get();
          
          BufferAllocationInfo info;
          info.alloc_node = op;
          info.handle_var = handle_var;
          info.original_scope = BufferUseCollector::GetPtrStorageScope(op->buffer_var);
          info.is_data_bound = false;
          info.from_block_alloc = false;
          
          parent_->alloc_info[handle_var] = info;
          parent_->handle_to_alloc[handle_var] = op;
                    
          StmtExprVisitor::VisitStmt_(op);
        }
        
       private:
        BufferUseCollector* parent_;
      };
      
      MappingBuilder builder(this);
      builder.operator()(stmt);
    }
    
    void VisitStmt_(const AllocateNode* op) override {
      
      if (alloc_info.find(op->buffer_var.get()) == alloc_info.end()) {
        std::string scope = op->buffer_var->type_annotation.defined() ?
                           GetPtrStorageScope(op->buffer_var) : "";
        
        BufferAllocationInfo info;
        info.alloc_node = op;
        info.handle_var = op->buffer_var.get();
        info.original_scope = scope;
        info.is_data_bound = false;
        info.from_block_alloc = false;
        alloc_info[op->buffer_var.get()] = info;
        handle_to_alloc[op->buffer_var.get()] = op;
      }
      
      StmtExprVisitor::VisitStmt_(op);
    }
    
    void VisitStmt_(const BlockNode* op) override {
      
      for (const Buffer& buffer : op->alloc_buffers) {
        const VarNode* handle_var = buffer->data.get();
        
        if (alloc_info.find(handle_var) == alloc_info.end()) {
          std::string scope = buffer->data->type_annotation.defined() ?
                             GetPtrStorageScope(buffer->data) : "";
          
          BufferAllocationInfo info;
          info.alloc_node = nullptr;
          info.handle_var = handle_var;
          info.original_scope = scope;
          info.is_data_bound = false;
          info.from_block_alloc = true;
          alloc_info[handle_var] = info;
          handle_to_alloc[handle_var] = nullptr;
          
        }
      }
      
      StmtExprVisitor::VisitStmt_(op);
    }
    
    void VisitExpr_(const CallNode* op) override {
      if (const OpNode* tl_op = op->op.as<OpNode>()) {
        if (tl_op->name == "tl.ascend_copy") {

          if (op->args.size() < 2) {
            return;
          }

          const VarNode* first_buf_handle = nullptr;
          const CallNode* first_region = op->args[0].as<CallNode>();
          if (first_region && !first_region->args.empty() && first_region->args[0].as<BufferLoadNode>()) {
            const BufferLoadNode* first_buf_load = first_region->args[0].as<BufferLoadNode>();
            first_buf_handle = first_buf_load->buffer->data.get();
          }

          const VarNode* second_buf_handle = nullptr;
          const CallNode* second_region = op->args[1].as<CallNode>();
          if (second_region && !second_region->args.empty() && second_region->args[0].as<BufferLoadNode>()) {
            const BufferLoadNode* second_buf_load = second_region->args[0].as<BufferLoadNode>();
            second_buf_handle = second_buf_load->buffer->data.get();
          }

          if (first_buf_handle && second_buf_handle) {
            std::string first_name = first_buf_handle->name_hint;
            std::string second_name = second_buf_handle->name_hint;
            ascend_copy_buffer_pairs_.emplace_back(first_buf_handle, second_buf_handle);
          }
        }
      }

      if (op->op.same_as(builtin::call_extern())) {
        if (op->args.size() > 0) {
          if (auto* func_name = op->args[0].as<StringImmNode>()) {
            std::string name = func_name->value;
            
            for (size_t i = 1; i < op->args.size(); ++i) {
              AnalyzeBufferInCall(op, i, name);
            }
          }
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }
    
    void AnalyzeBufferInCall(const CallNode* call, size_t arg_idx, 
                            const std::string& func_name) {
      auto* arg = call->args[arg_idx].as<CallNode>();
      if (!arg || !arg->op.same_as(builtin::tvm_access_ptr())) {
        return;
      }
      
      if (arg->args.size() > 1) {
        if (auto* var = arg->args[1].as<VarNode>()) {
          
          BufferUseInfo& info = buffer_use_info[var];
          info.func_names.insert(func_name);
          info.call_sites.push_back(call);
          
          if (IsGEMMFunction(func_name)) {
            info.used_in_cube = true;
            
            int gemm_position = DetermineGEMMPosition(call, arg_idx, func_name);
            if (gemm_position >= 0) {
              info.gemm_positions.insert(gemm_position);
            }
          }
          
          if (IsVectorFunction(func_name)) {
            info.used_in_vector = true;
          }
        }
      }
    }
    
    int DetermineGEMMPosition(const CallNode* call, size_t arg_idx, 
                             const std::string& func_name) {
      int access_ptr_count = 0;
      for (size_t i = 1; i < arg_idx; ++i) {
        if (auto* prev_arg = call->args[i].as<CallNode>()) {
          if (prev_arg->op.same_as(builtin::tvm_access_ptr())) {
            access_ptr_count++;
          }
        }
      }
      if (func_name.find("gemm") != std::string::npos ||
          func_name.find("mma") != std::string::npos) {
        if (access_ptr_count == 0) return 0; // A -> L0A
        if (access_ptr_count == 1) return 1; // B -> L0B
        if (access_ptr_count == 2) return 2; // C -> L0C
      }
      
      return -1;
    }
    
    const AllocateNode* GetAllocForHandle(const VarNode* handle) const {
      auto it = handle_to_alloc.find(handle);
      return it != handle_to_alloc.end() ? it->second : nullptr;
    }
    
    BufferAllocationInfo* GetAllocInfo(const VarNode* handle) {
      auto it = alloc_info.find(handle);
      return it != alloc_info.end() ? &it->second : nullptr;
    }
    
    BufferUseInfo* GetUseInfo(const VarNode* handle) {
      auto it = buffer_use_info.find(handle);
      return it != buffer_use_info.end() ? &it->second : nullptr;
    }

  private:
    static std::string ToLower(std::string_view str) {
      std::string lower_str(str);
      std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      return lower_str;
    }

    static bool ContainsAny(const std::string &str,
                            const std::vector<std::string> &keywords) {
      for (const auto &keyword : keywords) {
        if (str.find(keyword) != std::string::npos) {
          return true;
        }
      }
      return false;
    }

    static bool IsGEMMInternal(const std::string &lower_name) {
      static const std::vector<std::string> kGemmKeywords = {"gemm", "mma",
                                                             "matmul"};
      return ContainsAny(lower_name, kGemmKeywords);
    }
  };

  void InferCorrectScopes() {
    for (const auto& kv : collector_->alloc_info) {
      const VarNode* handle = kv.first;
      BufferAllocationInfo* alloc_info = collector_->GetAllocInfo(handle);
      if (!alloc_info) continue;
      
      BufferUseInfo* use_info = collector_->GetUseInfo(handle);
      
      std::string original_scope = alloc_info->original_scope;
      std::string corrected_scope = original_scope;
      
      if (original_scope == "local.fragment") {
        if (use_info && !use_info->gemm_positions.empty()) {
          if (use_info->gemm_positions.count(0) > 0) {
            corrected_scope = "wmma.matrix_a";  // L0A
          } else if (use_info->gemm_positions.count(1) > 0) {
            corrected_scope = "wmma.matrix_b";  // L0B
          } else if (use_info->gemm_positions.count(2) > 0) {
            corrected_scope = "wmma.accumulator";  // L0C
          } else {
            corrected_scope = "wmma.accumulator";
          }
        } else {
          corrected_scope = "wmma.accumulator";
        }
      } else if (original_scope == "shared.dyn") {
        if (use_info) {
          if (use_info->used_in_vector && !use_info->used_in_cube) {
            corrected_scope = "shared";
          } else if (use_info->used_in_cube && !use_info->used_in_vector) {
            corrected_scope = "shared.dyn";
          } else if (use_info->used_in_cube && use_info->used_in_vector) {
            bool has_conflict = CheckSharedBufferConflict(use_info);
            if (has_conflict) {
              // todo
            }
            corrected_scope = "shared.dyn";
          } else {
            corrected_scope = original_scope;
          }
        } else {
          corrected_scope = original_scope;
        }
      }
      
      if (corrected_scope != original_scope) {
        alloc_info->corrected_scope = corrected_scope;
        
        if (alloc_info->alloc_node) {
          scope_corrections_[alloc_info->alloc_node] = corrected_scope;
        }
        
        handle_scope_corrections_[handle] = corrected_scope;
        
      }
    }
    const std::string UB_SCOPE = "shared";

    for (const auto& buf_pair : collector_->ascend_copy_buffer_pairs_) {
      const VarNode* first_buf_handle = buf_pair.first;
      const VarNode* second_buf_handle = buf_pair.second;

      BufferAllocationInfo* first_alloc_info = collector_->GetAllocInfo(first_buf_handle);
      BufferAllocationInfo* second_alloc_info = collector_->GetAllocInfo(second_buf_handle);
      if (!first_alloc_info || !second_alloc_info) {
        continue;
      }

      std::string first_buf_scope = first_alloc_info->corrected_scope.empty() 
                                    ? first_alloc_info->original_scope 
                                    : first_alloc_info->corrected_scope;
      bool first_is_ub = (first_buf_scope == UB_SCOPE);
      std::string first_name = first_buf_handle->name_hint;
      std::string second_name = second_buf_handle->name_hint;

      if (!first_is_ub) {
        continue;
      }

      BufferUseInfo* second_use_info = collector_->GetUseInfo(second_buf_handle);
      bool second_no_cube_vector = false;
      if (second_use_info == nullptr) {
        second_no_cube_vector = true;
      } else {
        if (!second_use_info->used_in_cube && !second_use_info->used_in_vector) {
          second_no_cube_vector = true;
        }
      }

      if (!second_no_cube_vector) {
        continue;
      }

      std::string second_old_scope = second_alloc_info->corrected_scope.empty() 
                                      ? second_alloc_info->original_scope 
                                      : second_alloc_info->corrected_scope;

      second_alloc_info->corrected_scope = UB_SCOPE;
      handle_scope_corrections_[second_buf_handle] = UB_SCOPE;
      if (second_alloc_info->alloc_node) {
        scope_corrections_[second_alloc_info->alloc_node] = UB_SCOPE;
      }
    }
  }
  
  bool CheckSharedBufferConflict(const BufferUseInfo* use_info) {
    return false;
  }
  
  void PrintDebugInfo() {
    LOG(INFO) << "=== Scope Correction Results ===";
    for (const auto& kv : collector_->alloc_info) {
      const VarNode* handle = kv.first;
      const BufferAllocationInfo& info = kv.second;
      
      BufferUseInfo* use_info = collector_->GetUseInfo(handle);
      
      LOG(INFO) << "Buffer: " << handle->name_hint;
      LOG(INFO) << "  Original scope: " << info.original_scope;
      LOG(INFO) << "  Corrected scope: " << info.corrected_scope;
      
      if (use_info) {
        if (!use_info->func_names.empty()) {
          LOG(INFO) << "  Used in functions:";
          for (const auto& func : use_info->func_names) {
            LOG(INFO) << "    - " << func;
          }
        }
        
        if (!use_info->gemm_positions.empty()) {
          LOG(INFO) << "  GEMM positions:";
          for (int pos : use_info->gemm_positions) {
            const char* role = "Unknown";
            if (pos == 0) role = "A (L0A)";
            else if (pos == 1) role = "B (L0B)";
            else if (pos == 2) role = "C (L0C)";
            LOG(INFO) << "    - Position " << pos << ": " << role;
          }
        }
        
        LOG(INFO) << "  Used in Cube: " << use_info->used_in_cube;
        LOG(INFO) << "  Used in Vector: " << use_info->used_in_vector;
      }
      LOG(INFO) << "---";
    }
  }

  std::unordered_map<const AllocateNode*, std::string> scope_corrections_;
  std::unordered_map<const VarNode*, std::string> handle_scope_corrections_;
  std::unordered_map<const VarNode*, Var> var_replacements_;
  std::unique_ptr<BufferUseCollector> collector_;
  
  static std::string GetPtrStorageScope(Var buffer_var) {
    if (auto* ptr_type = buffer_var->type_annotation.as<PointerTypeNode>()) {
      return ptr_type->storage_scope;
    }
    return "";
  }
  
  Var CreateVarWithCorrectScope(const Var& old_var, const std::string& new_scope) {
    auto it = var_replacements_.find(old_var.get());
    if (it != var_replacements_.end()) {
      return it->second;
    }
    
    auto ptr_type = old_var->type_annotation.as<PointerTypeNode>();
    if (ptr_type) {
      auto new_ptr_type = PointerType(ptr_type->element_type, new_scope);
      Var new_var(old_var->name_hint, new_ptr_type);
      var_replacements_[old_var.get()] = new_var;
      return new_var;
    }
    
    auto new_ptr_type = PointerType(PrimType(DataType::Void()), new_scope);
    Var new_var(old_var->name_hint, new_ptr_type);
    var_replacements_[old_var.get()] = new_var;
    return new_var;
  }
  
  Stmt VisitStmt_(const AllocateNode* op) override {
    auto it = scope_corrections_.find(op);
    if (it != scope_corrections_.end()) {
      std::string new_scope = it->second;
      
      Var new_buffer_var = op->buffer_var;
      auto handle_it = handle_scope_corrections_.find(op->buffer_var.get());
      if (handle_it != handle_scope_corrections_.end()) {
        new_buffer_var = CreateVarWithCorrectScope(op->buffer_var, handle_it->second);
      }
      
      return Allocate(new_buffer_var, op->dtype, op->extents, 
                      op->condition, VisitStmt(op->body));
    }
    
    return StmtExprMutator::VisitStmt_(op);
  }
  
  Stmt VisitStmt_(const BlockNode* op) override {    
    Array<Buffer> new_alloc_buffers;
    std::unordered_map<const VarNode*, Buffer> buffer_replacements;
    
    for (const Buffer& buffer : op->alloc_buffers) {
      const VarNode* handle = buffer->data.get();
      auto it = handle_scope_corrections_.find(handle);
      
      if (it != handle_scope_corrections_.end()) {
        std::string new_scope = it->second;
        
        Var new_data = CreateVarWithCorrectScope(buffer->data, new_scope);
        var_replacements_[handle] = new_data;
        
        auto new_buffer = Buffer(new_data, buffer->dtype, buffer->shape,
                                buffer->strides, buffer->elem_offset,
                                buffer->name, buffer->data_alignment,
                                buffer->offset_factor, buffer->buffer_type);
        
        new_alloc_buffers.push_back(new_buffer);
        buffer_replacements[handle] = new_buffer;
      } else {
        new_alloc_buffers.push_back(buffer);
      }
    }
    
    auto new_body = VisitStmt(op->body);
    auto new_block = Block(op->iter_vars, 
                          op->reads,
                          op->writes,
                          op->name_hint,
                          new_body,
                          op->init,
                          new_alloc_buffers,
                          op->match_buffers,
                          op->annotations);
    
    return new_block;
  }
  
  PrimExpr VisitExpr_(const VarNode* op) override {
    auto it = handle_scope_corrections_.find(op);
    if (it != handle_scope_corrections_.end()) {
      Var old_var = GetRef<Var>(op);
      
      auto var_it = var_replacements_.find(op);
      if (var_it != var_replacements_.end()) {
        return var_it->second;
      }
      
      Var new_var = CreateVarWithCorrectScope(old_var, it->second);
      var_replacements_[op] = new_var;
      return std::move(new_var);
    }
    
    return GetRef<PrimExpr>(op);
  }
  
  PrimExpr VisitExpr_(const BufferLoadNode* op) override {
    auto buffer = op->buffer;
    
    auto it = handle_scope_corrections_.find(buffer->data.get());
    if (it != handle_scope_corrections_.end()) {
      Var new_data = CreateVarWithCorrectScope(buffer->data, it->second);
      auto new_buffer = Buffer(new_data, buffer->dtype, buffer->shape,
                               buffer->strides, buffer->elem_offset,
                               buffer->name, buffer->data_alignment,
                               buffer->offset_factor, buffer->buffer_type);
      
      Array<PrimExpr> indices;
      indices.reserve(op->indices.size());
      for (const auto& index : op->indices) {
        indices.push_back(VisitExpr(index));
      }
      
      return BufferLoad(new_buffer, indices);
    }
    
    return StmtExprMutator::VisitExpr_(op);
  }
  
  Stmt VisitStmt_(const BufferStoreNode* op) override {
    auto buffer = op->buffer;
    
    auto it = handle_scope_corrections_.find(buffer->data.get());
    if (it != handle_scope_corrections_.end()) {
      Var new_data = CreateVarWithCorrectScope(buffer->data, it->second);
      auto new_buffer = Buffer(new_data, buffer->dtype, buffer->shape,
                               buffer->strides, buffer->elem_offset,
                               buffer->name, buffer->data_alignment,
                               buffer->offset_factor, buffer->buffer_type);
      
      auto value = VisitExpr(op->value);
      Array<PrimExpr> indices;
      indices.reserve(op->indices.size());
      for (const auto& index : op->indices) {
        indices.push_back(VisitExpr(index));
      }
      
      return BufferStore(new_buffer, value, indices);
    }
    
    return StmtExprMutator::VisitStmt_(op);
  }
  
  PrimExpr VisitExpr_(const CallNode* op) override {
    if (op->op.same_as(builtin::tvm_access_ptr())) {

      if (op->args.size() > 1) {
        if (auto* var = op->args[1].as<VarNode>()) {
          auto it = handle_scope_corrections_.find(var);
          if (it != handle_scope_corrections_.end()) {

            Array<PrimExpr> new_args;
            new_args.push_back(op->args[0]);  // type annotation
            new_args.push_back(CreateVarWithCorrectScope(GetRef<Var>(var), it->second));
            
            for (size_t i = 2; i < op->args.size(); ++i) {
              new_args.push_back(VisitExpr(op->args[i]));
            }
            
            return Call(op->dtype, op->op, new_args, op->span);
          }
        }
      }
    }
    
    return StmtExprMutator::VisitExpr_(op);
  }
};

transform::Pass InferAllocScope() {
  auto pass_func = [](PrimFunc f, IRModule m, transform::PassContext ctx) {
    ScopeCorrector corrector;
    return corrector.Correct(std::move(f));
  };
  return transform::CreatePrimFuncPass(pass_func, 0, 
                                      "tl.InferAllocScope", {});
}

TVM_REGISTER_GLOBAL("tl.transform.InferAllocScope")
    .set_body_typed(InferAllocScope);

}  // namespace tir
}  // namespace tvm