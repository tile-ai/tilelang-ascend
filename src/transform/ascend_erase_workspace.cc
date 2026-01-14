#include <tvm/tir/transform.h>
#include <tvm/tir/builtin.h>       
#include "arith/ir_mutator_with_analyzer.h"  
#include "../op/builtin.h"     

namespace tvm {
namespace tl {


enum class DstBufferScope {
  L1,
  Ub
};

enum class CopyDirection {
  None,
  UbToL1,
  L0cToUb
};

struct WorkspaceInfo {
  DataType dtype;
  std::string dtype_str;
  std::string workspace_name;
  std::string associated_buffer;
  Buffer workspace_buffer;
  Array<PrimExpr> shapes;
  PrimExpr offset;
  PrimExpr extent;
  PrimExpr dim;
};

struct ThreadMetaInfo {
  Var block_var;
  PrimExpr total_thread_nums;
  bool is_valid = false;
};

struct CopyGlobalContext{
  // Maps workspace name to the corresponding detailed workspace information
  std::unordered_map<std::string, WorkspaceInfo> workspace_map_;
  // Tracks the total number of existing workspaces
  size_t workspace_num_ = 0;
  // Maps source buffer name to the associated workspace name
  std::unordered_map<std::string, std::string> src_to_workspace_map_;
  // Maps destination buffer name to the associated workspace name
  std::unordered_map<std::string, std::string> dst_to_workspace_map_;
  // Maps destination buffer name to its corresponding access Call object
  std::unordered_map<std::string, Call> dst_to_access_map_;
  // Stores the set of all buffers that are currently being tracked
  std::unordered_set<std::string> tracked_buffers_;
  // Maps source buffer name to its corresponding destination buffer name
  std::unordered_map<std::string, std::string> src_to_dst_map_;
  // Maps destination buffer name to its corresponding scope information
  std::unordered_map<std::string, DstBufferScope> dst_to_scope_map_;
};


class CopyInfoCollector : public StmtExprVisitor {
private:
  CopyGlobalContext context_;

  static ThreadMetaInfo thread_meta_info_;

  static const std::unordered_map<std::string, DataType> type_map_;

  std::unordered_set<std::string> target_copy_stmts_ = {
    "copy_ub_to_l1",
    "copy_l0c_to_ub"
  };
  std::unordered_map<std::string, DstBufferScope> scope_table_ = {
    {"copy_ub_to_l1", DstBufferScope::L1},
    {"copy_l0c_to_ub", DstBufferScope::Ub}
  }; 
public:

  const CopyGlobalContext& GetCopyGlobalContext() const {
    return context_;
  }

  explicit CopyInfoCollector(
    const std::unordered_set<std::string>& target_copy_stmts = {},
    const std::unordered_map<std::string, DstBufferScope>& scope_table = {}
  ) {
    thread_meta_info_ = ThreadMetaInfo();
    if (!target_copy_stmts.empty()) {
      this->target_copy_stmts_ = target_copy_stmts;
    }
    if (!scope_table.empty()) {
      this->scope_table_ = scope_table;
    }
  }

  DataType ConvertStringToDataType(const std::string& type_str) {
    auto it = type_map_.find(type_str);
    
    ICHECK(it != type_map_.end()) << "[Error] Not supported type: " << type_str << ", maybe you need to update type_map_.";

    return it->second;

  }
  
  std::string IsTargetCopyExpr(const CallNode* call_node) {
    if (!call_node || call_node->op != tir::builtin::call_extern()) {
      return "";
    }
    
    if (call_node->args.empty() || !call_node->args[0].as<StringImmNode>()) {
      return "";
    }
    std::string copy_stmt = Downcast<StringImm>(call_node->args[0])->value;
    for (const auto& target_substr : target_copy_stmts_) {
      size_t pos = copy_stmt.find(target_substr);
      if (pos != std::string::npos) {
        return copy_stmt;
      }
    }
    return "";
  }
  
  void WorkspaceInfoCollector(const std::string& copy_stmt, const CallNode* src_access_ptr, const CallNode* dst_access_ptr) {
    if (src_access_ptr == nullptr || dst_access_ptr == nullptr) {
      std::cerr << "[Error]<WorkspaceInfoCollector>: src_access_ptr or dst_access_ptr is nullptr!" << std::endl;
    }
    for (auto &target_copy_stmt : target_copy_stmts_) {
      std::vector<std::string> params_vec;
      if (copy_stmt.find(target_copy_stmt) != std::string::npos) {
        CopyDirection copy_direction = CopyDirection::None;
        std::string current_param;
        std::string src_buffer_name = src_access_ptr->args[1].as<VarNode>()->name_hint;
        std::string dst_buffer_name = dst_access_ptr->args[1].as<VarNode>()->name_hint;
        if (target_copy_stmt == "copy_ub_to_l1") { 
          copy_direction = CopyDirection::UbToL1;
        } else if (target_copy_stmt == "copy_l0c_to_ub") {
          copy_direction = CopyDirection::L0cToUb;
        }
        size_t left_bracket = copy_stmt.find('<');
        size_t right_bracket = copy_stmt.rfind('>');
        if (left_bracket == std::string::npos || right_bracket == std::string::npos || left_bracket > right_bracket) {
          std::cerr << "[Warning]<WorkspaceInfoCollector>: illegal template parameters scope!" << std::endl;
        }
        std::string template_content = copy_stmt.substr(left_bracket + 1, right_bracket - left_bracket - 1);
        for (auto &c : template_content) {
          if (c == ',') {
            if (!current_param.empty()) {
              params_vec.push_back(current_param); // [type N M] or [type1 type2 LayoutGM M N enRelu]
              current_param = "";
            } else {
            std::cerr << "[Warning]<WorkspaceInfoCollector>: current_param is empty!" << std::endl;
            }
            continue;
          } 
          if (std::isspace(c)) {
            continue;
          }
          current_param.push_back(c);
        }
        params_vec.push_back(current_param);
        if (copy_direction == CopyDirection::UbToL1) {
          std::swap(params_vec[1], params_vec[2]);
        }
        ++context_.workspace_num_;
        WorkspaceInfo ws_info;
        std::stringstream ss;
        ss << "workspace_" << context_.workspace_num_;
        ws_info.workspace_name = ss.str();
        
        ws_info.associated_buffer = dst_buffer_name;
        
        ws_info.dtype_str = params_vec[0];
        
        ws_info.dtype = ConvertStringToDataType(ws_info.dtype_str);
        
        switch (copy_direction) {
          case CopyDirection::UbToL1:
          for (int i = 1; i < params_vec.size(); ++i) {
            int shapeI =  std::stoi(params_vec[i]);
            ws_info.shapes.push_back(IntImm(DataType::Int(32), shapeI));
          }
          break;
          case CopyDirection::L0cToUb:
          int shapeM = std::stoi(params_vec[3]); // [type1 type2 LayoutGM M N enRelu] 3 -> M
          int shapeN = std::stoi(params_vec[4]); // [type1 type2 LayoutGM M N enRelu] 3 -> M
          ws_info.shapes.push_back(IntImm(DataType::Int(32), shapeM)); 
          ws_info.shapes.push_back(IntImm(DataType::Int(32), shapeN)); 
          break;
        }
        
        ws_info.dim = IntImm(DataType::Int(64), ws_info.shapes.size());
        
        int64_t total_data_nums = 1;
        for (auto &i : ws_info.shapes) {
          const IntImmNode* i_node = i.as<IntImmNode>();
          ICHECK(i_node != nullptr) << "[Error]<WorkspaceInfoCollector>: Shape dimension is not a valid IntImm, cannot extract value\n";
          total_data_nums *= i_node->value;
        }
        
        ws_info.offset = thread_meta_info_.block_var * IntImm(DataType::Int(64), total_data_nums);
        
        ws_info.extent = thread_meta_info_.total_thread_nums * IntImm(DataType::Int(64), total_data_nums) - ws_info.offset;
        
        Array<PrimExpr> strides;
        BufferType buffer_type = BufferType::kDefault; 
        DataType storage_dtype = (ws_info.dtype == DataType::Bool() ? DataType::Int(8) : ws_info.dtype);

        Array<PrimExpr> real_shapes{thread_meta_info_.total_thread_nums};
        for (const auto &shape : ws_info.shapes) {
          real_shapes.push_back(shape);
        }
        ws_info.workspace_buffer = decl_buffer(
          real_shapes,
          ws_info.dtype,
          ws_info.workspace_name,
          "global"
        );

        context_.workspace_map_[ws_info.workspace_name] = ws_info;
        context_.src_to_workspace_map_[src_buffer_name] = ws_info.workspace_name;
        context_.dst_to_workspace_map_[dst_buffer_name] = ws_info.workspace_name;
      }
    }
  }

  void VisitStmt(const Stmt& stmt) final { StmtExprVisitor::VisitStmt(stmt); }

  void VisitStmt_(const EvaluateNode* op) final {
    const CallNode* call_node = op->value.as<CallNode>();
    if (!call_node) {
      return StmtExprVisitor::VisitStmt_(op);
    }

    std::string copy_stmt_name = IsTargetCopyExpr(call_node);
    if (!copy_stmt_name.empty()) {
      Array<PrimExpr> call_node_args = call_node->args;
      
      const CallNode* src_ptr = call_node_args[1].as<CallNode>();
      const CallNode* dst_ptr = call_node_args[2].as<CallNode>();
      if (!src_ptr || !dst_ptr) {
        std::cout << "[Error]<CopyInfoCollector>src_ptr or dst_ptr is nullptr" << std::endl;
        return StmtExprVisitor::VisitStmt_(op);
      }
      
      if (!src_ptr->op.same_as(tvm::tir::builtin::tvm_access_ptr()) || src_ptr->args.size() < 2) {
        std::cout << "[Error]<CopyInfoCollector> src is not access ptr or its size of srgs is too small" << std::endl;
        return StmtExprVisitor::VisitStmt_(op);
      }
      if (!dst_ptr->op.same_as(tvm::tir::builtin::tvm_access_ptr()) || dst_ptr->args.size() < 2) {
        std::cout << "[Error]<CopyInfoCollector> dst is not access ptr or its size of srgs is too small" << std::endl;
        return StmtExprVisitor::VisitStmt_(op);
      }
      const VarNode* src_name_var_node = (src_ptr->args[1].as<VarNode>());
      const VarNode* dst_name_var_node = (dst_ptr->args[1].as<VarNode>());
      if (!src_name_var_node || !dst_name_var_node) {
        std::cout << "[Error]<CopyInfoCollector> src or dst args[1] is not VarNode" << std::endl;
        return StmtExprVisitor::VisitStmt_(op);
      }

      WorkspaceInfoCollector(copy_stmt_name, src_ptr, dst_ptr);

      const std::string src_buffer_name = src_name_var_node->name_hint;
      const std::string dst_buffer_name = dst_name_var_node->name_hint;
      context_.tracked_buffers_.insert(src_buffer_name);
      context_.src_to_dst_map_[src_buffer_name] = dst_buffer_name;
      context_.dst_to_access_map_[dst_buffer_name] = GetRef<Call>(dst_ptr); 

      std::string copy_func_name = call_node_args[0].as<StringImmNode>()->value;
      for (auto &target_copy_stmt : target_copy_stmts_) {
        if (copy_func_name.find(target_copy_stmt) != std::string::npos) {
          context_.dst_to_scope_map_[dst_buffer_name] = scope_table_[target_copy_stmt];
        }
      }
    }
  }

  void VisitStmt_(const AttrStmtNode* op) override {

    if (!thread_meta_info_.is_valid) {
      if (op->attr_key == tvm::tir::attr::thread_extent) {
        if (const IterVarNode* iter_var_node = op->node.as<IterVarNode>()) {
          IterVar iter_var = GetRef<IterVar>(iter_var_node);
          if (iter_var->thread_tag == "blockIdx.x") {
            thread_meta_info_.block_var = iter_var->var;
            thread_meta_info_.total_thread_nums = op->value;
            thread_meta_info_.is_valid = true;
          }
        }
      }
    }

    StmtVisitor::VisitStmt_(op);
  }
};

const std::unordered_map<std::string, DataType> CopyInfoCollector::type_map_ = {
  {"half", tvm::runtime::DataType::Float(16)},    
  {"float16", tvm::runtime::DataType::Float(16)},
  {"float", tvm::runtime::DataType::Float(32)},
  {"float32", tvm::runtime::DataType::Float(32)},
  {"float64", tvm::runtime::DataType::Float(64)},
  {"int8", tvm::runtime::DataType::Int(8)},
  {"int16", tvm::runtime::DataType::Int(16)},
  {"int", tvm::runtime::DataType::Int(32)},
  {"int32", tvm::runtime::DataType::Int(32)},
  {"int64", tvm::runtime::DataType::Int(64)},
  {"uint8", tvm::runtime::DataType::UInt(8)},
  {"uint16", tvm::runtime::DataType::UInt(16)},
  {"uint32", tvm::runtime::DataType::UInt(32)},
  {"uint64", tvm::runtime::DataType::UInt(64)}
};

ThreadMetaInfo CopyInfoCollector::thread_meta_info_;

class AscendEraseWorkspacePass : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    CopyInfoCollector info_collector;
    info_collector.VisitStmt(f->body);
    const CopyGlobalContext& context = info_collector.GetCopyGlobalContext();
    AscendEraseWorkspacePass substituter(&analyzer, context);

    
    auto* f_mut = f.CopyOnWrite();
    f_mut->body = substituter.VisitStmt(f->body);
    
    Array<Var> new_params = f_mut->params;
    Array<IntImm> auto_gm_idx;
    for (const auto& [ws_name, ws_info] : context.workspace_map_) {
      if (ws_info.workspace_buffer.defined()) {
        Var ws_handle(ws_name + "_handle", DataType::Handle());
        if (!f_mut->buffer_map.count(ws_handle)) {
          f_mut->buffer_map.Set(ws_handle, ws_info.workspace_buffer);
          new_params.push_back(ws_handle);
          auto_gm_idx.push_back(IntImm(DataType::UInt(32), new_params.size() - 1));
        }
      }
    }
          
    f_mut->params = new_params;
    tvm::tir::PrimFunc prim_func_with_new_attr = GetRef<tvm::tir::PrimFunc>(f_mut);
    prim_func_with_new_attr = tvm::WithAttr(
      std::move(prim_func_with_new_attr),             
      "auto_gm_indices",    
      auto_gm_idx                           
    );
    
    return prim_func_with_new_attr;
  }
  
private:
        
  AscendEraseWorkspacePass(arith::Analyzer* analyzer, const CopyGlobalContext& context) 
    : arith::IRMutatorWithAnalyzer(analyzer),
      context_(context) {}
  
  const CopyGlobalContext& context_;

  std::unordered_set<std::string> inserted_buffers_;
  
  std::unordered_set<std::string> copy_stmt_table_ = {
    "copy_gm_to_l1", "copy_l1_to_l0a", "copy_l1_to_l0b",
    "copy_l0c_to_gm", "copy_gm_to_ub", "copy_ub_to_gm",
    "copy_ub_to_l1", "copy_l0c_to_ub", "copy_ub_to_ub"
  };

  std::unordered_map<std::string, std::string> copy_replace_table_ = {
    {"copy_ub_to_l1", "copy_ub_to_gm"},
    {"copy_l0c_to_ub", "copy_l0c_to_gm"}
  };

//   std::string TAB = "    ";
  
  std::string PrintDstScope(const DstBufferScope& dst_scope) {
    switch(dst_scope) {
        case DstBufferScope::L1:
          return "l1";
          break;
        case DstBufferScope::Ub:
          return "ub";
          break;
        default:
          return "UnknownScope";
      }
  }

  bool IsTargetCopyExpr(const CallNode* call_node) {

    if (!call_node || call_node->op != tir::builtin::call_extern()) {
      return false;
    }

    if (call_node->args.empty() || !call_node->args[0].as<StringImmNode>()) {
      return false;
    }
    std::string func_name = Downcast<StringImm>(call_node->args[0])->value;
    for (const auto& [target_substr, replace_substr] : copy_replace_table_) {
      size_t pos = func_name.find(target_substr);
      if (pos != std::string::npos) {
        return true;
      }
    }
    return false;
  }

  StringImm ReplaceCopyFuncName(const StringImm& orig_name) {
    std::string new_name = orig_name->value;
    std::string original_name = new_name;

    for (const auto& [target_substr, replace_substr] : copy_replace_table_) {
        size_t pos = new_name.find(target_substr);
        if (pos != std::string::npos) {
            new_name.replace(pos, target_substr.length(), replace_substr);
            break;
        }
    }

    return StringImm(new_name);
  } 

  std::vector<std::string> IsTargetAccessNode(const CallNode* call_node) {

    static const std::vector<std::string> EMPTY_BUFFER_LIST;
    std::vector<std::string> dst_buffers_vec;
    if (!call_node) {
      return EMPTY_BUFFER_LIST;
    }

    if (call_node->args.empty()) {
      return EMPTY_BUFFER_LIST;
    }

    Array<PrimExpr> call_node_args = call_node->args;
    if (call_node_args.empty()) {
      return EMPTY_BUFFER_LIST;
    }

    const StringImmNode* call_node_name = call_node_args[0].as<StringImmNode>();
    if (call_node_name != nullptr) {
      const std::string call_node_name_str = call_node_name->value;
      for (const auto& copy_stmt : copy_stmt_table_) { // copy stmts should be excluded
        if (call_node_name_str.find(copy_stmt) != std::string::npos) {
          return EMPTY_BUFFER_LIST;
        }
      }
    }

    for (int i = 0; i < call_node_args.size(); ++i) {
      const PrimExpr& arg = call_node_args[i];
      const CallNode* access_ptr = arg.as<CallNode>();

      if (!access_ptr || !access_ptr->op.same_as(builtin::tvm_access_ptr())) {
        continue;
      }

      Array<PrimExpr> access_args = access_ptr->args;
      if (access_args.empty() || access_args.size() < 2) {
        std::cout << "[Warning]<IsTargetAccessNode>: access ptr args size too small!\n";
        continue;
      }
      
      const VarNode* buffer_var = access_args[1].as<VarNode>();
      if (!buffer_var) {
        std::cout << "[Warning]<IsTargetAccessNode>: access ptr without buffer_var\n";
        continue;
      }
      
      std::string buffer_name = buffer_var->name_hint;

      for (const auto& [src_buffer_name, dst_buffer_name] : context_.src_to_dst_map_) {
        if (!inserted_buffers_.count(buffer_name) && buffer_name == dst_buffer_name) {
          dst_buffers_vec.push_back(buffer_name);
          inserted_buffers_.insert(buffer_name); 
        } else if (inserted_buffers_.count(buffer_name) && buffer_name == dst_buffer_name) {
          continue;
        }
      }
    }
    return dst_buffers_vec;
  }
  
  Call CreateAccessPtrCall(const WorkspaceInfo& ws_info, int rw_mask) {
    DataType dtype = DataType::Handle();
    const Op& op = builtin::tvm_access_ptr();

    Array<PrimExpr> args = {
      TypeAnnotation(ws_info.dtype),
      ws_info.workspace_buffer->data,
      ws_info.offset,
      ws_info.extent,
      IntImm(DataType::Int(32), rw_mask)
    };

    Call access_ptr_call(dtype, op, args);
    return access_ptr_call;
  }
  
  Stmt CreateGmToDstStmt(const Call& src_access, const Call& dst_access, 
                         const DstBufferScope& buffer_scope, PrimExpr& strideN) {
    Array<PrimExpr> args;
    const CallNode* gm_access_ptr = src_access.as<CallNode>();
    const VarNode* gm_var = gm_access_ptr->args[1].as<VarNode>();
    std::string gm_name = gm_var->name_hint;
    WorkspaceInfo ws_info = context_.workspace_map_.at(gm_name);

    std::stringstream ss;
    ss << "tl::ascend::copy_gm_to_" << PrintDstScope(buffer_scope) << "<" << ws_info.dtype_str;
    if (buffer_scope == DstBufferScope::L1) {
      for (auto& shape : ws_info.shapes) {
        // gm_to_l1 : M N
        ss << ", " << shape.as<IntImmNode>()->value;
      }
      ss << ">";
    } else if (buffer_scope == DstBufferScope::Ub) {
      // gm_to_ub : N M
      for (auto it = ws_info.shapes.rbegin(); it != ws_info.shapes.rend(); ++it) {
        ss << ", " << *it;
      }
      ss << ">";
    }

    std::string stmt_name_str = ss.str();

    StringImm stmt_name(stmt_name_str);
    args.push_back(stmt_name);
    args.push_back(src_access); 
    args.push_back(dst_access); 
    args.push_back(strideN);

    Call new_call_node(
      DataType::Handle(),
      tvm::tir::builtin::call_extern(),
      args
    );

    return Evaluate(new_call_node);
  }

  Stmt VisitStmt_(const EvaluateNode* op) final {
    Stmt orig_stmt = arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    const EvaluateNode* orig_eval_node = orig_stmt.as<EvaluateNode>();
    if (!orig_eval_node) {
      return orig_stmt;
    }


    const CallNode* orig_call = orig_eval_node->value.as<CallNode>();
    if (!orig_call) {
      return orig_stmt;
    }
    
    if (IsTargetCopyExpr(orig_call)) {
      Array<PrimExpr> new_args = orig_call->args;
      StringImm orig_func_name = Downcast<StringImm>(new_args[0]);
      const CallNode* src_access_ptr = new_args[1].as<CallNode>();
      std::string src_buffer_name = (src_access_ptr->args[1]).as<VarNode>()->name_hint;
      
      StringImm new_func_name = ReplaceCopyFuncName(orig_func_name);
      new_args.Set(0, new_func_name); 
      std::string ws_name = context_.src_to_workspace_map_.at(src_buffer_name);
      WorkspaceInfo ws_info = context_.workspace_map_.at(ws_name);
      new_args.Set(2, CreateAccessPtrCall(ws_info, 2));
      Call new_call = Call(orig_call->dtype,
                      orig_call->op,
                      new_args,
                      orig_call->span);
      return Evaluate(new_call);
    } 


    std::vector<std::string> buffer_names_vec = IsTargetAccessNode(orig_call);
    if (!buffer_names_vec.empty()) {
      Array<Stmt> seq_stmts_array;
      for (const auto& buffer_name : buffer_names_vec) {

        auto buffer_scope_it = context_.dst_to_scope_map_.find(buffer_name);

        CHECK(buffer_scope_it != context_.dst_to_scope_map_.end())
          << "[Error]<EvaluateNode>: Dst scope is not found for buffer: "
          << buffer_name << "\n";

        DstBufferScope buffer_scope = context_.dst_to_scope_map_.at(buffer_name);
        const std::string workspace_name = context_.dst_to_workspace_map_.at(buffer_name);
        WorkspaceInfo ws_info = context_.workspace_map_.at(workspace_name);

        
        Call workspace_access = CreateAccessPtrCall(ws_info, 1);
        PrimExpr strideN = ws_info.shapes[ws_info.shapes.size() - 1];
        Stmt insert_eval_stmt = this->CreateGmToDstStmt(workspace_access, context_.dst_to_access_map_.at(buffer_name), buffer_scope, strideN);
        seq_stmts_array.push_back(insert_eval_stmt);
      }
      seq_stmts_array.push_back(orig_stmt);
      return SeqStmt(seq_stmts_array);
    }
    return orig_stmt;
  }

};

namespace transform {

  using namespace tir::transform;

  tvm::transform::Pass AscendEraseWorkspace() {
    auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
      return AscendEraseWorkspacePass::Substitute(std::move(f));
    };
    return CreatePrimFuncPass(pass_func, 0, "tl.AscendEraseWorkspace", {});
  }

  TVM_REGISTER_GLOBAL("tl.transform.AscendEraseWorkspace").set_body_typed(AscendEraseWorkspace);
  } // namespace transform

}// namespace tl
}// namespace tvm