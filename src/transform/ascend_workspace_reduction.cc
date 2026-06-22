#include <algorithm>
#include "../op/builtin.h"
#include "arith/ir_mutator_with_analyzer.h"
#include <tvm/tir/builtin.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

static constexpr const char *ascendPtoUsePipeInCVCopy = "tl.ascend_pto_use_pipe_in_cv_copy";
TVM_REGISTER_PASS_CONFIG_OPTION(ascendPtoUsePipeInCVCopy, Bool);

enum class DstBufferScope { L1, Ub };

enum class CopyDirection { None, UbToL1, L0cToUb };

struct WorkspaceInfo {
  DataType dtype;
  std::string dtype_str;
  std::string workspace_name;
  std::string associated_dst_buffer_name;
  Buffer workspace_buffer;
  Array<PrimExpr> shapes;
  PrimExpr offset;
  PrimExpr extent;
  PrimExpr dim;
  int64_t per_block_ele_nums = 0;
  PrimExpr dst_l1_row_offset; // Only used for skip UB scenario to store the row
                              // offset for in-place UB to workspace copy
                              // transformation
};

struct CoreMetaInfo {
  Var cid_var;
  Var vid_var;
  PrimExpr total_core_nums;
  int vector_cnt = 1;
};

// Shared by WorkspaceInfo and PipeInfo
struct CrossCoreCopyInfo {
  std::string copy_stmt_name;
  CopyDirection direction;           // UbToL1 / L0cToUb
  std::string src_buffer_name;
  std::string dst_buffer_name;
  std::string dtype_str;             // dst dtype
  std::vector<std::string> params;   // template params (parsed, UbToL1 swapped)
  Array<PrimExpr> shapes;            // workspace shapes (C-core dimensions)

  // runtime args from ascend.cc virtual_channel
  int src_M = 0, src_N = 0;
  int dst_M = 0, dst_N = 0;

  // C-core element count
  int64_t per_block_ele_nums = 0;
};

struct CopyGlobalContext {
  // Maps workspace name to the corresponding detailed workspace information
  std::map<std::string, WorkspaceInfo> workspace_map_;
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
  // Core meta info for vid offset calculation (threads=2 scenario)
  CoreMetaInfo core_meta_info_;
  // UB buffers that skipped VidReduction (from PrimFunc attrs)
  std::unordered_set<std::string> buffers_skip_vid_reduction_;
  // Buffer shapes (from PrimFunc attrs)
  std::unordered_map<std::string, Array<PrimExpr>> buffer_shapes_;
  // PTO pipe metadata: maps dst buffer name to pipe info for deferred pop emission
  struct PipeInfo {
    int flag_id;         // pipe flag token
    int dir_type;        // 1=C2V, 2=V2C
    int slot_size;       // max(src_elems, dst_elems) * dtype_bytes
    int slot_num;        // 1
    std::string pipe_id; // "pipe_X_V2C" or "pipe_X_C2V"
    std::string op_name; // "copy_pipe_to_l1" or "copy_pipe_to_ub"
    std::string dtype_str;
    int src_M_val;
    int src_N_val;
    int dst_M_val;
    int dst_N_val;
    int split_axis = 1; // 0=TILE_NO_SPLIT, 1=TILE_UP_DOWN, 2=TILE_LEFT_RIGHT
    std::string workspace_name;
    bool has_tmp = false;
    int tmp_M_val = 0;
    int tmp_N_val = 0;
  };
  std::unordered_map<std::string, PipeInfo> pipe_info_map_;
};

class CopyInfoCollector : public StmtExprVisitor {
private:
  CopyGlobalContext context_;
  bool pto_use_pipe_;
  std::string platform_;
  bool needs_gm_workspace_;
  int pipe_flag_id_counter_ = 0;  // FlagID starts from 0

  std::unordered_set<std::string> target_copy_stmts_ = {"copy_ub_to_l1",
                                                        "copy_l0c_to_ub"};
  std::unordered_map<std::string, DstBufferScope> scope_table_ = {
      {"copy_ub_to_l1", DstBufferScope::L1},
      {"copy_l0c_to_ub", DstBufferScope::Ub}};

public:
  static const std::unordered_map<std::string, DataType> type_map_;

  const CopyGlobalContext &GetCopyGlobalContext() const { return context_; }

  explicit CopyInfoCollector(
      const std::unordered_set<std::string> &target_copy_stmts = {},
      const std::unordered_map<std::string, DstBufferScope> &scope_table = {},
      bool pto_use_pipe = false,
      const std::string &platform = "A3")
      : platform_(platform), needs_gm_workspace_(false) {
    if (!target_copy_stmts.empty()) {
      this->target_copy_stmts_ = target_copy_stmts;
    }
    if (!scope_table.empty()) {
      this->scope_table_ = scope_table;
    }
    this->pto_use_pipe_ = pto_use_pipe;
    this->needs_gm_workspace_ = (platform != "A5");
  }

  void SetSkipBuffers(const std::unordered_set<std::string> &skip_buffers) {
    context_.buffers_skip_vid_reduction_ = skip_buffers;
  }

  void SetBufferShapes(
      const std::unordered_map<std::string, Array<PrimExpr>> &shapes) {
    context_.buffer_shapes_ = shapes;
  }

  DataType ConvertStringToDataType(const std::string &type_str) {
    auto it = type_map_.find(type_str);

    ICHECK(it != type_map_.end())
        << "[Error]<WorkspaceReduction>: Not supported type: " << type_str
        << ", maybe you need to update type_map_.";

    return it->second;
  }

  std::string IsTargetCopyExpr(const CallNode *call_node) {
    if (!call_node || call_node->op != tir::builtin::call_extern()) {
      return "";
    }

    if (call_node->args.empty() || !call_node->args[0].as<StringImmNode>()) {
      return "";
    }
    std::string copy_stmt = Downcast<StringImm>(call_node->args[0])->value;
    for (const auto &target_substr : target_copy_stmts_) {
      size_t pos = copy_stmt.find(target_substr);
      if (pos != std::string::npos) {
        return copy_stmt;
      }
    }
    return "";
  }

  std::pair<const CallNode*, const CallNode*> ExtractCopyAccessPtrs(
      const CallNode *call_node) {
    Array<PrimExpr> call_node_args = call_node->args;
    const CallNode *src_ptr = call_node_args[1].as<CallNode>();
    const CallNode *dst_ptr = call_node_args[2].as<CallNode>();
    ICHECK(src_ptr != nullptr && dst_ptr != nullptr)
        << "[Error]<WorkspaceReduction>: src_ptr or dst_ptr is "
           "nullptr!";
    ICHECK(src_ptr->op.same_as(tvm::tir::builtin::tvm_access_ptr()) &&
           src_ptr->args.size() >= 2)
        << "[Error]<WorkspaceReduction>: src is not access ptr or "
           "its size of args is too small";
    ICHECK(dst_ptr->op.same_as(tvm::tir::builtin::tvm_access_ptr()) &&
           dst_ptr->args.size() >= 2)
        << "[Error]<WorkspaceReduction>: dst is not access ptr or "
           "its size of args is too small";
    return {src_ptr, dst_ptr};
  }

  std::vector<std::string> ParseTemplateParams(const std::string &func_name) {
    std::vector<std::string> params;
    size_t left = func_name.find('<');
    size_t right = func_name.rfind('>');
    ICHECK(left != std::string::npos && right != std::string::npos &&
           left < right)
        << "[Error]<WorkspaceReduction>: illegal template parameters scope!";
    std::string content = func_name.substr(left + 1, right - left - 1);
    std::string current;
    for (auto &c : content) {
      if (c == ',') {
        ICHECK(!current.empty())
            << "[Error]<WorkspaceReduction>: Empty parameter found in template!";
        params.push_back(current);  // [type N M] or 
                                    // [type1 type2 LayoutGM M N enRelu]
        current = "";
        continue;
      }
      if (!std::isspace(c)) current.push_back(c);
    }
    ICHECK(!current.empty())
        << "[Error]<WorkspaceReduction>: Empty parameter found in template!";
    params.push_back(current);
    return params;
  }

  CrossCoreCopyInfo CrossCoreCopyInfoCollector(
      const std::string &copy_stmt_name,
      const CallNode *call_node,
      const CallNode *src_access_ptr,
      const CallNode *dst_access_ptr) {
    const VarNode *src_name_var_node = (src_access_ptr->args[1].as<VarNode>());
    const VarNode *dst_name_var_node = (dst_access_ptr->args[1].as<VarNode>());
    ICHECK(src_name_var_node != nullptr && dst_name_var_node != nullptr)
        << "[Error]<WorkspaceReduction>: src or dst args[1] is not VarNode!";

    CrossCoreCopyInfo copy_info;
    copy_info.copy_stmt_name = copy_stmt_name;
    copy_info.src_buffer_name = src_name_var_node->name_hint;
    copy_info.dst_buffer_name = dst_name_var_node->name_hint;

    copy_info.params = ParseTemplateParams(copy_stmt_name);

    if (copy_stmt_name.find("copy_ub_to_l1") != std::string::npos) {
      ICHECK(copy_info.params.size() >= 2)
              << "[Error]<WorkspaceReduction> PTO: copy_ub_to_l1 template expects 2+ params, got "
              << copy_info.params.size();
      copy_info.direction = CopyDirection::UbToL1;
      copy_info.dtype_str = copy_info.params[0]; // UbToL1: [dtype, N, M] or [dtype, N]
      
      if (copy_info.params.size() >= 3) {
        // Normal UbToL1: [dtype, N, M] -> swap to [dtype, M, N]
        std::swap(copy_info.params[1], copy_info.params[2]);
        int M = std::stoi(copy_info.params[1]);
        int N = std::stoi(copy_info.params[2]);
        if (context_.core_meta_info_.vid_var.defined() &&
            context_.core_meta_info_.vector_cnt > 1) {
          M *= context_.core_meta_info_.vector_cnt;
        }
        copy_info.shapes.push_back(IntImm(DataType::Int(32), M));
        copy_info.shapes.push_back(IntImm(DataType::Int(32), N));
        copy_info.per_block_ele_nums = M * N;
      } else {
        // UbToL1 with 2 params: [dtype, N]
        int N = std::stoi(copy_info.params[1]);
        copy_info.shapes.push_back(IntImm(DataType::Int(32), N));
        copy_info.per_block_ele_nums = N;
      }
    } else {
      ICHECK(copy_info.params.size() >= 5)
              << "[Error]<WorkspaceReduction> PTO: copy_l0c_to_ub template expects 5+ params, got "
              << copy_info.params.size();
      copy_info.direction = CopyDirection::L0cToUb;
      copy_info.dtype_str = copy_info.params[1];  // L0cToUb: [src_dtype, dst_dtype,
                                        // LayoutGM, M, N, enRelu]

      int M = std::stoi(copy_info.params[3]);
      int N = std::stoi(copy_info.params[4]);
      copy_info.shapes.push_back(IntImm(DataType::Int(32), M));
      copy_info.shapes.push_back(IntImm(DataType::Int(32), N));
      copy_info.per_block_ele_nums = M * N;
    }

    if (call_node->args.size() >= 7) {
      // Read src/dst extents from runtime args (pushed by ascend.cc virtual_channel)
      auto src_N_imm = call_node->args[3].as<IntImmNode>();
      auto src_M_imm = call_node->args[4].as<IntImmNode>();
      auto dst_M_imm = call_node->args[5].as<IntImmNode>();
      auto dst_N_imm = call_node->args[6].as<IntImmNode>();
      ICHECK(src_N_imm && src_M_imm && dst_M_imm && dst_N_imm)
          << "[Error]<WorkspaceReduction>: virtual_channel args must be IntImm, "
          << "got non-IntImm PrimExpr. Dynamic shapes not yet supported.";
      copy_info.src_N = src_N_imm->value;
      copy_info.src_M = src_M_imm->value;
      copy_info.dst_M = dst_M_imm->value;
      copy_info.dst_N = dst_N_imm->value;
    }

    return copy_info;
  }

  void WorkspaceInfoCollector(const CrossCoreCopyInfo &copy_info,
                              const CallNode *dst_access_ptr) {
    ICHECK(dst_access_ptr != nullptr)
        << "[Error]<WorkspaceReduction>: dst_access_ptr is nullptr!";

    ++context_.workspace_num_;
    WorkspaceInfo ws_info;
    std::stringstream ss;
    ss << "workspace_" << context_.workspace_num_;
    ws_info.workspace_name = ss.str();
    const std::string &src_buffer_name = copy_info.src_buffer_name;
    const std::string &dst_buffer_name = copy_info.dst_buffer_name;

    ws_info.associated_dst_buffer_name = dst_buffer_name;

    // Check if src_buffer is in skip set (must be defined before switch)
    bool is_skip_ub =
        context_.buffers_skip_vid_reduction_.count(src_buffer_name) > 0;

    switch (copy_info.direction) {
    case CopyDirection::UbToL1:
      ws_info.dtype_str = copy_info.dtype_str;
      ws_info.dtype = ConvertStringToDataType(ws_info.dtype_str);

      if (is_skip_ub) {
        // Skip UB scenario: use full L1 shape from buffer_shapess
        // Template params for skip UB: [dtype, size] (no M, N)

        // dst_access_ptr args: [type_annotation, var, offset, extent, ...]
        // args.size() should be at least 4

        // Check args size before accessing
        ICHECK(dst_access_ptr->args.size() >= 4)
            << "[ERROR]<WorkspaceReduction>: dst_access_ptr args size "
               "should be at least 4, got "
            << dst_access_ptr->args.size();

        PrimExpr dst_offset = dst_access_ptr->args[2]; // row offset

        // Get dst buffer full shape from buffer_shapes_
        auto shape_it = context_.buffer_shapes_.find(dst_buffer_name);
        ICHECK(shape_it != context_.buffer_shapes_.end())
            << "[ERROR]<WorkspaceReduction>: dst_buffer " << dst_buffer_name
            << " not found in buffer_shapes_";
        Array<PrimExpr> dst_full_shape = shape_it->second;

        // Shape must be 2D [M, N] for current Skip UB UbToL1 scenario
        // Future: Extend to support multi-dimensional shapes based on
        // buffer scope
        ICHECK(dst_full_shape.size() == 2)
            << "[ERROR]<WorkspaceReduction>: Skip UB UbToL1 expects "
               "dst_full_shape to be 2D, "
               "got "
            << dst_full_shape.size();

        PrimExpr M = dst_full_shape[0]; // rows
        PrimExpr N = dst_full_shape[1]; // cols

        ws_info.shapes.push_back(M);
        ws_info.shapes.push_back(N);

        // per_block_ele_nums = M * N
        int64_t M_val = 0, N_val = 0;
        if (const IntImmNode *M_imm = M.as<IntImmNode>()) {
          M_val = M_imm->value;
        }
        if (const IntImmNode *N_imm = N.as<IntImmNode>()) {
          N_val = N_imm->value;
        }
        ws_info.per_block_ele_nums = M_val * N_val;

        // dst_extent for workspace = M * N (full L1 size)
        PrimExpr dst_full_extent = M * N;

        // base offset: cid * dst_full_extent (for workspace declaration,
        // excludes dst_offset) workspace declaration)
        ws_info.offset = context_.core_meta_info_.cid_var * dst_full_extent;

        // Store row offset separately, added during UbToL1 in-place
        // transform
        ws_info.dst_l1_row_offset = dst_offset;

        // extent: total_core_nums * dst_full_extent - base_offset
        ws_info.extent =
            context_.core_meta_info_.total_core_nums * dst_full_extent -
            ws_info.offset;

      } else {
        // Normal UB scenario: use shapes from copy_info
        ws_info.shapes = copy_info.shapes;
        ws_info.per_block_ele_nums = copy_info.per_block_ele_nums;

        ws_info.offset =
            context_.core_meta_info_.cid_var *
            IntImm(DataType::Int(64), ws_info.per_block_ele_nums);

        ws_info.extent =
            context_.core_meta_info_.total_core_nums *
                IntImm(DataType::Int(64), ws_info.per_block_ele_nums) -
            ws_info.offset;
      }
      break;
    case CopyDirection::L0cToUb:
      ws_info.dtype_str = copy_info.dtype_str; // L0cToUb: [src_dtype, dst_dtype,
                                          // LayoutGM, M, N, enRelu]
      ws_info.dtype = ConvertStringToDataType(ws_info.dtype_str);
      ws_info.shapes = copy_info.shapes;
      ws_info.per_block_ele_nums = copy_info.per_block_ele_nums;

      ws_info.offset =
          context_.core_meta_info_.cid_var *
          IntImm(DataType::Int(64), ws_info.per_block_ele_nums);

      ws_info.extent =
          context_.core_meta_info_.total_core_nums *
              IntImm(DataType::Int(64), ws_info.per_block_ele_nums) -
          ws_info.offset;
      break;
    }

    Array<PrimExpr> real_shapes;
    if (pto_use_pipe_) {
      // A2 PTO: flat 1D buffer with 2x size for FIFO double-buffering
      PrimExpr total_elems = context_.core_meta_info_.total_core_nums;
      for (const auto &shape : ws_info.shapes) {
        total_elems = total_elems * shape;
      }
      total_elems = total_elems * IntImm(DataType::Int(64), 2);
      real_shapes.push_back(total_elems);
    } else {
      real_shapes.push_back(context_.core_meta_info_.total_core_nums);
      for (const auto &shape : ws_info.shapes) {
        real_shapes.push_back(shape);
      }
    }
    ws_info.dim = IntImm(DataType::Int(64), real_shapes.size());
    ws_info.workspace_buffer = decl_buffer(
        real_shapes, ws_info.dtype, ws_info.workspace_name, "global");

    context_.workspace_map_[ws_info.workspace_name] = ws_info;
    context_.src_to_workspace_map_[copy_info.src_buffer_name] =
        ws_info.workspace_name;
    context_.dst_to_workspace_map_[copy_info.dst_buffer_name] =
        ws_info.workspace_name;
  }

  int GetDtypeBytes(const std::string &dtype_str) {
    auto it = type_map_.find(dtype_str);
    ICHECK(it != type_map_.end())
        << "[Error]<WorkspaceReduction> PTO: Unsupported dtype: " << dtype_str;
    return it->second.bytes();
  }

  // Returns: 
  //   0 = TILE_NO_SPLIT (shapes equal)
  //   1 = TILE_UP_DOWN (M differs by 2x)
  //   2 = TILE_LEFT_RIGHT (N differs by 2x)
  static int ComputeSplitAxis(int src_M, int src_N, int dst_M, int dst_N) {
    if (src_M == dst_M && src_N == dst_N) return 0;
    if (src_M * 2 == dst_M || dst_M * 2 == src_M) return 1;
    if (src_N * 2 == dst_N || dst_N * 2 == src_N) return 2;
    return 1;
  }

  void PipeInfoCollector(const CrossCoreCopyInfo &copy_info,
                         const CallNode *call_node) {
    const std::string &src_buffer_name = copy_info.src_buffer_name;
    const std::string &dst_buffer_name = copy_info.dst_buffer_name;
    bool is_ub_to_l1 = (copy_info.direction == CopyDirection::UbToL1);

    CopyGlobalContext::PipeInfo info;
    if (is_ub_to_l1) {
      info.dir_type = 2;
      info.op_name = "copy_pipe_to_l1";

      // Check if tmp buffer is provided for A5 (ND->Nz conversion)
      // After AscendCopy::Lower, args layout: [0]=func_name, [1]=src_ptr,
      // [2]=dst_ptr, [3]=srcN, [4]=srcM, [5]=dstM, [6]=dstN, [7]=tmp_ptr,
      // [8]=tmpM, [9]=tmpN
      if (call_node->args.size() > 9) {
        PrimExpr tmp_expr = call_node->args[7];
        if (auto *tmp_call = tmp_expr.as<CallNode>()) {
          info.has_tmp = true;
          info.tmp_M_val = Downcast<IntImm>(call_node->args[8])->value;
          info.tmp_N_val = Downcast<IntImm>(call_node->args[9])->value;
        }
      }
    } else {
      info.dir_type = 1;
      info.op_name = "copy_pipe_to_ub";
    }

    info.dtype_str = copy_info.dtype_str;
    info.src_N_val = copy_info.src_N;
    info.src_M_val = copy_info.src_M;
    info.dst_M_val = copy_info.dst_M;
    info.dst_N_val = copy_info.dst_N;

    info.flag_id = pipe_flag_id_counter_;
    pipe_flag_id_counter_ += 2; // one pipe needs 2 FlagID
    int dtype_bytes = GetDtypeBytes(info.dtype_str);
    info.slot_size = copy_info.per_block_ele_nums * dtype_bytes;
    info.slot_num = 1;
    info.pipe_id = "pipe_" + std::to_string(info.flag_id) + "_"
                   + (info.dir_type == 2 ? "V2C" : "C2V");
    info.split_axis = ComputeSplitAxis(info.src_M_val, info.src_N_val,
                                       info.dst_M_val, info.dst_N_val);
    info.workspace_name = needs_gm_workspace_
        ? context_.src_to_workspace_map_.at(src_buffer_name)
        : "";
    context_.pipe_info_map_[dst_buffer_name] = info;
  }

  void VisitStmt(const Stmt &stmt) final { StmtExprVisitor::VisitStmt(stmt); }

  void VisitStmt_(const EvaluateNode *op) final {
    const CallNode *call_node = op->value.as<CallNode>();
    if (!call_node) {
      return StmtExprVisitor::VisitStmt_(op);
    }

    std::string copy_stmt_name = IsTargetCopyExpr(call_node);
    if (!copy_stmt_name.empty()) {
      // Extract copy access ptrs ONCE (shared by workspace collection,
      // copy-back tracking, and PTO pipe metadata)
      auto [src_ptr, dst_ptr] = ExtractCopyAccessPtrs(call_node);

      CrossCoreCopyInfo copy_info = CrossCoreCopyInfoCollector(
          copy_stmt_name, call_node, src_ptr, dst_ptr);

      // Layer 1: Workspace collection (differs per backend)
      // Ascend C || A2 PTO: collect workspace info for GM buffer allocation
      // A5 PTO: skip workspace collection
      if (needs_gm_workspace_ || !pto_use_pipe_) {
        WorkspaceInfoCollector(copy_info, dst_ptr);
      }

      // Layer 2: Shared copy-back tracking (AscendC + PTO both need this)
      const std::string &src_buffer_name = copy_info.src_buffer_name;
      const std::string &dst_buffer_name = copy_info.dst_buffer_name;
      context_.tracked_buffers_.insert(src_buffer_name);
      context_.src_to_dst_map_[src_buffer_name] = dst_buffer_name;

      // skip UB + UbToL1: reconstruct dst_access_ptr
      bool is_skip_ub =
          context_.buffers_skip_vid_reduction_.count(src_buffer_name) > 0;
      bool is_ub_to_l1 = (copy_info.direction == CopyDirection::UbToL1);

      if (is_skip_ub && is_ub_to_l1) {
        // Get dst full shape [M, N] from buffer_shapes_
        auto shape_it = context_.buffer_shapes_.find(dst_buffer_name);
        if (shape_it != context_.buffer_shapes_.end()) {
          Array<PrimExpr> dst_full_shape = shape_it->second;
          PrimExpr M = dst_full_shape[0]; // rows
          PrimExpr N = dst_full_shape[1]; // cols
          PrimExpr full_extent = M * N;

          // Construct new dst_access_ptr: offset=0, extent=M*N
          Array<PrimExpr> new_args = {
              dst_ptr->args[0],             // type_annotation
              dst_ptr->args[1],             // var
              IntImm(DataType::Int(64), 0), // offset = 0
              full_extent,                  // extent = M * N
              dst_ptr->args[4]              // rw_mask
          };

          Call new_dst_access(dst_ptr->dtype, dst_ptr->op, new_args);
          context_.dst_to_access_map_[dst_buffer_name] = new_dst_access;
        } else {
          context_.dst_to_access_map_[dst_buffer_name] = GetRef<Call>(dst_ptr);
        }
      } else {
        context_.dst_to_access_map_[dst_buffer_name] = GetRef<Call>(dst_ptr);
      }

      std::string copy_func_name = call_node->args[0].as<StringImmNode>()->value;
      for (auto &target_copy_stmt : target_copy_stmts_) {
        if (copy_func_name.find(target_copy_stmt) != std::string::npos) {
          context_.dst_to_scope_map_[dst_buffer_name] =
              scope_table_[target_copy_stmt];
        }
      }

      // Layer 3: PTO pipe metadata collection (for deferred pop emission)
      if (pto_use_pipe_) {
        PipeInfoCollector(copy_info, call_node);
      }
    }
  }

  void VisitStmt_(const AttrStmtNode *op) override {

    if (op->attr_key == tvm::tir::attr::thread_extent) {
      if (const IterVarNode *iter_var_node = op->node.as<IterVarNode>()) {
        IterVar iter_var = GetRef<IterVar>(iter_var_node);
        if (/*!context_.core_meta_info_.is_valid &&*/
            iter_var->thread_tag == "blockIdx.x") {
          context_.core_meta_info_.cid_var = iter_var->var;
          context_.core_meta_info_.total_core_nums = op->value;
          // context_.core_meta_info_.is_valid = true;
        }
        if (iter_var->thread_tag == "threadIdx.x") {
          context_.core_meta_info_.vid_var = iter_var->var;
          context_.core_meta_info_.vector_cnt =
              Downcast<IntImm>(op->value)->value;  // 1 or 2
        }
        if (iter_var->thread_tag == "blockIdx.y") {
          context_.core_meta_info_.vid_var = iter_var->var;
          context_.core_meta_info_.vector_cnt =
              Downcast<IntImm>(op->value)->value;  // 2
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
    {"bfloat16_t", tvm::runtime::DataType::BFloat(16)},
    {"AscendC::int4b_t", tvm::runtime::DataType::Int(4)},
    {"int8_t", tvm::runtime::DataType::Int(8)},
    {"int16_t", tvm::runtime::DataType::Int(16)},
    {"int", tvm::runtime::DataType::Int(32)},
    {"int32", tvm::runtime::DataType::Int(32)},
    {"int64_t", tvm::runtime::DataType::Int(64)},
    {"uint8_t", tvm::runtime::DataType::UInt(8)},
    {"uint16_t", tvm::runtime::DataType::UInt(16)},
    {"uint32_t", tvm::runtime::DataType::UInt(32)},
    {"uint64_t", tvm::runtime::DataType::UInt(64)},
    {"float8_e4m3_t", tvm::runtime::DataType::NVFloat8E4M3()},
    {"float8_e5m2_t", tvm::runtime::DataType::NVFloat8E5M2()}};

class AscendWorkspaceReductionPass : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f, bool pto_use_pipe = false, const std::string &platform = "A3") {
    arith::Analyzer analyzer;

    // Read buffers_skip_vid_reduction from PrimFunc attrs
    std::unordered_set<std::string> skip_buffer_names;
    std::unordered_map<std::string, Array<PrimExpr>> buffer_shapes;
    if (f->attrs.defined()) {
      auto attrs_dict = f->attrs->dict;

      // Read buffers_skip_vid_reduction
      auto it_skip = attrs_dict.find("buffers_skip_vid_reduction");
      if (it_skip != attrs_dict.end()) {
        auto skip_array = Downcast<Array<String>>((*it_skip).second);
        for (const auto &name : skip_array) {
          skip_buffer_names.insert(name);
        }
      }

      // Read buffer_shapess for skip UB scenario
      auto it_shapes = attrs_dict.find("buffer_shapess");
      if (it_shapes != attrs_dict.end()) {
        auto shapes_map =
            Downcast<Map<Var, Array<PrimExpr>>>((*it_shapes).second);
        for (const auto &pair : shapes_map) {
          std::string buf_name = pair.first->name_hint;
          buffer_shapes[buf_name] = pair.second;
        }
      }
    }

    CopyInfoCollector info_collector({}, {}, pto_use_pipe, platform);
    info_collector.SetSkipBuffers(skip_buffer_names);
    info_collector.SetBufferShapes(buffer_shapes);
    info_collector.VisitStmt(f->body);
    const CopyGlobalContext &context = info_collector.GetCopyGlobalContext();
    AscendWorkspaceReductionPass substituter(&analyzer, context, pto_use_pipe, platform);

    auto *f_mut = f.CopyOnWrite();
    f_mut->body = substituter.VisitStmt(f->body);

    Array<Var> new_params = f_mut->params;
    Array<IntImm> auto_gm_idx;
    for (const auto &[ws_name, ws_info] : context.workspace_map_) {
      if (ws_info.workspace_buffer.defined()) {
        Var ws_handle(ws_name + "_handle", DataType::Handle());
        if (!f_mut->buffer_map.count(ws_handle)) {
          f_mut->buffer_map.Set(ws_handle, ws_info.workspace_buffer);
          new_params.push_back(ws_handle);
          auto_gm_idx.push_back(
              IntImm(DataType::UInt(32), new_params.size() - 1));
        }
      }
    }

    f_mut->params = new_params;
    tvm::tir::PrimFunc prim_func_with_new_attr =
        GetRef<tvm::tir::PrimFunc>(f_mut);
    prim_func_with_new_attr = tvm::WithAttr(std::move(prim_func_with_new_attr),
                                            "auto_gm_indices", auto_gm_idx);

    // Serialize pipe metadata as PrimFunc attr for codegen access
    if (pto_use_pipe && !context.pipe_info_map_.empty()) {
      Map<IntImm, Map<String, ObjectRef>> pipe_infos;
      for (const auto &[dst_buf_name, info] : context.pipe_info_map_) {
        Map<String, ObjectRef> fields;
        fields.Set("flag_id",   IntImm(DataType::Int(32), info.flag_id));
        fields.Set("dir_type",  IntImm(DataType::Int(32), info.dir_type));
        fields.Set("slot_size", IntImm(DataType::Int(32), info.slot_size));
        fields.Set("slot_num",  IntImm(DataType::Int(32), info.slot_num));
        fields.Set("pipe_id",   String(info.pipe_id));
        fields.Set("op_name",   String(info.op_name));
        fields.Set("dtype_str", String(info.dtype_str));
        fields.Set("src_M_val", IntImm(DataType::Int(32), info.src_M_val));
        fields.Set("src_N_val", IntImm(DataType::Int(32), info.src_N_val));
        fields.Set("dst_M_val", IntImm(DataType::Int(32), info.dst_M_val));
        fields.Set("dst_N_val", IntImm(DataType::Int(32), info.dst_N_val));
        fields.Set("split_axis",    IntImm(DataType::Int(32), info.split_axis));
        fields.Set("workspace_name", String(info.workspace_name));
        fields.Set("has_tmp",   IntImm(DataType::Int(32), info.has_tmp ? 1 : 0));
        fields.Set("tmp_M_val", IntImm(DataType::Int(32), info.tmp_M_val));
        fields.Set("tmp_N_val", IntImm(DataType::Int(32), info.tmp_N_val));
        pipe_infos.Set(IntImm(DataType::Int(32), info.flag_id), fields);
      }
      prim_func_with_new_attr = WithAttr(std::move(prim_func_with_new_attr),
                                         "pipe_infos", pipe_infos);
    }

    return prim_func_with_new_attr;
  }

private:
  AscendWorkspaceReductionPass(arith::Analyzer *analyzer,
                                const CopyGlobalContext &context,
                                bool pto_use_pipe = false,
                                const std::string &platform = "A3")
      : arith::IRMutatorWithAnalyzer(analyzer), context_(context),
        core_meta_info__(context.core_meta_info_), pto_use_pipe_(pto_use_pipe), platform_(platform) {}

  const CopyGlobalContext &context_;
  const CoreMetaInfo core_meta_info__; // For vid offset calculation
  bool pto_use_pipe_;
  std::string platform_;

  std::unordered_set<std::string> inserted_buffers_;

  std::unordered_set<std::string> copy_stmt_table_ = {
      "copy_gm_to_l1",  "copy_l1_to_l0a", "copy_l1_to_l0b",
      "copy_l0c_to_gm", "copy_gm_to_ub",  "copy_ub_to_gm",
      "copy_ub_to_l1",  "copy_l0c_to_ub", "copy_ub_to_ub"};

  std::unordered_map<std::string, std::string> copy_replace_table_ = {
      {"copy_ub_to_l1", "copy_ub_to_gm"}, {"copy_l0c_to_ub", "copy_l0c_to_gm"}};

  Stmt GenerateCopyToPipe(const CallNode *orig_call,
                        const std::string &func_name_str) {
    Array<PrimExpr> orig_args = orig_call->args;
    PrimExpr src_access = orig_args[1];
    PrimExpr dst_access = orig_args[2];

    // Get dst buffer name for pipe_info_map_ lookup
    const CallNode *dst_access_ptr = dst_access.as<CallNode>();
    ICHECK(dst_access_ptr && dst_access_ptr->args.size() >= 2);
    const VarNode *dst_var = dst_access_ptr->args[1].as<VarNode>();
    ICHECK(dst_var);
    std::string dst_buffer_name = dst_var->name_hint;

    auto pipe_it = context_.pipe_info_map_.find(dst_buffer_name);
    ICHECK(pipe_it != context_.pipe_info_map_.end())
        << "[Error]<WorkspaceReduction> PTO: No pipe info found for: " << dst_buffer_name;
    const auto &pipe = pipe_it->second;

    bool is_ub_to_l1 =
        (func_name_str.find("copy_ub_to_l1") != std::string::npos);
    std::string pipe_op1 =
        is_ub_to_l1 ? "copy_ub_to_pipe" : "copy_l0c_to_pipe";

    std::stringstream ss;
    ss << "tl::ascend::" << pipe_op1;
    Array<PrimExpr> args = {
        StringImm(ss.str()),
        src_access,
        src_access,
        IntImm(DataType::Int(32), pipe.flag_id)
    };
    if (is_ub_to_l1 && pipe.has_tmp && orig_args.size() > 7) {
      PrimExpr tmp_expr = orig_args[7];
      if (tmp_expr.as<CallNode>()) {
        args.push_back(tmp_expr);
      }
    }
    Call call(DataType::Handle(), tir::builtin::call_extern(), args);

    if (is_ub_to_l1 && pipe.has_tmp && platform_ == "A5" &&
        orig_args.size() > 7 && orig_args[7].as<CallNode>()) {
      Array<PrimExpr> args_nz = {StringImm("tl::ascend::copy_ub_to_ub_Nz"),
                                 src_access, orig_args[7]};
      Call call_nz(DataType::Handle(), tir::builtin::call_extern(), args_nz);
      return SeqStmt({Evaluate(call_nz), Evaluate(call)});
    }

    return Evaluate(call);
  }

  Stmt GenerateCopyFromPipe(const CopyGlobalContext::PipeInfo &pipe,
                         const Call &dst_access) {
    std::string op_name = pipe.op_name;
    if (platform_ == "A5" && op_name == "copy_pipe_to_ub") {
      op_name = "copy_pipe_to_ub_V";
    }
    std::stringstream ss;
    ss << "tl::ascend::" << op_name;

    Array<PrimExpr> args = {
        StringImm(ss.str()),
        dst_access,
        dst_access,
        IntImm(DataType::Int(32), pipe.flag_id)
    };
    Call call(DataType::Handle(), tir::builtin::call_extern(), args);
    return Evaluate(call);
  }

  std::string PrintDstScope(const DstBufferScope &dst_scope) {
    switch (dst_scope) {
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

  bool IsTargetCopyExpr(const CallNode *call_node) {

    if (!call_node || call_node->op != tir::builtin::call_extern()) {
      return false;
    }

    if (call_node->args.empty() || !call_node->args[0].as<StringImmNode>()) {
      return false;
    }
    std::string func_name = Downcast<StringImm>(call_node->args[0])->value;
    for (const auto &[target_substr, replace_substr] : copy_replace_table_) {
      size_t pos = func_name.find(target_substr);
      if (pos != std::string::npos) {
        return true;
      }
    }
    return false;
  }

  StringImm ReplaceCopyFuncName(const StringImm &orig_name) {
    std::string new_name = orig_name->value;
    std::string original_name = new_name;

    for (const auto &[target_substr, replace_substr] : copy_replace_table_) {
      size_t pos = new_name.find(target_substr);
      if (pos != std::string::npos) {
        new_name.replace(pos, target_substr.length(), replace_substr);
        break;
      }
    }

    return StringImm(new_name);
  }

  std::vector<std::string> IsTargetAccessNode(const CallNode *call_node) {

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

    for (int i = 0; i < call_node_args.size(); ++i) {
      const PrimExpr &arg = call_node_args[i];
      const CallNode *access_ptr = arg.as<CallNode>();

      if (!access_ptr || !access_ptr->op.same_as(builtin::tvm_access_ptr())) {
        continue;
      }

      Array<PrimExpr> access_args = access_ptr->args;
      ICHECK(!access_args.empty())
          << "[Error]<WorkspaceReduction>: access ptr args is empty!";
      ICHECK(access_args.size() >= 2)
          << "[Error]<WorkspaceReduction>: access ptr args size too small!";

      const VarNode *buffer_var = access_args[1].as<VarNode>();
      ICHECK(buffer_var != nullptr)
          << "[Error]<WorkspaceReduction>: access ptr args[1] is not VarNode!";

      std::string buffer_name = buffer_var->name_hint;

      // access_args layout: [type_annotation, var, offset, extent, rw_mask]
      // Only consider buffer as a "consumer" if rw_mask has read bit set.
      // Skip write-only access (rw_mask == 2) since no copy-back needed before
      // a write.
      ICHECK(access_args.size() >= 5) << "[Error]<WorkspaceReduction>: access "
                                         "ptr args size too small for rw_mask!";
      int rw_mask = Downcast<IntImm>(access_args[4])->value;
      if (!(rw_mask & 1)) {
        // write-only access → this buffer is a producer, not a consumer
        continue;
      }

      for (const auto &[src_buffer_name, dst_buffer_name] :
           context_.src_to_dst_map_) {
        if (!inserted_buffers_.count(buffer_name) &&
            buffer_name == dst_buffer_name) {
          dst_buffers_vec.push_back(buffer_name);
          inserted_buffers_.insert(buffer_name);
        } else if (inserted_buffers_.count(buffer_name) &&
                   buffer_name == dst_buffer_name) {
          continue;
        }
      }
    }
    return dst_buffers_vec;
  }

  Call CreateWorkspaceAccessPtr(const WorkspaceInfo &ws_info, int rw_mask,
                                bool other_side_is_ub) {
    DataType dtype = DataType::Handle();
    const Op &op = builtin::tvm_access_ptr();

    PrimExpr offset = ws_info.offset; // Base offset: cid * M * N

    // skip UB UbToL1 in-place: add row offset
    // Condition: dst_l1_row_offset exists and workspace as dst (rw_mask=2,
    // write) workspace restore (rw_mask=1, read) doesn't need this
    if (ws_info.dst_l1_row_offset.defined() && rw_mask == 2) {
      offset = offset + ws_info.dst_l1_row_offset;
    }

    // Check if vid offset needed: GM's other side is UB (UB halved by
    // VidReduction)
    bool need_vid_offset = other_side_is_ub &&
                           core_meta_info__.vid_var.defined() &&
                           core_meta_info__.vector_cnt > 1;

    if (need_vid_offset) {
      // vid offset: vid * per_block_ele_nums / vector_cnt
      PrimExpr vid_offset =
          core_meta_info__.vid_var *
          IntImm(DataType::Int(64), ws_info.per_block_ele_nums) /
          IntImm(DataType::Int(64), core_meta_info__.vector_cnt);
      offset = offset + vid_offset;
    }

    Array<PrimExpr> args = {TypeAnnotation(ws_info.dtype),
                            ws_info.workspace_buffer->data, offset,
                            ws_info.extent, IntImm(DataType::Int(32), rw_mask)};

    Call access_ptr_call(dtype, op, args);
    return access_ptr_call;
  }

  Stmt CreateGmToDstStmt(const Call &src_access, const Call &dst_access,
                         const DstBufferScope &buffer_scope,
                         PrimExpr &strideN) {
    Array<PrimExpr> args;
    const CallNode *gm_access_ptr = src_access.as<CallNode>();
    const VarNode *gm_var = gm_access_ptr->args[1].as<VarNode>();
    std::string gm_name = gm_var->name_hint;
    WorkspaceInfo ws_info = context_.workspace_map_.at(gm_name);

    std::stringstream ss;
    ss << "tl::ascend::copy_gm_to_" << PrintDstScope(buffer_scope) << "<"
       << ws_info.dtype_str;
    if (buffer_scope == DstBufferScope::L1) {
      for (auto &shape : ws_info.shapes) {
        // gm_to_l1 : M N
        ss << ", " << shape.as<IntImmNode>()->value;
      }
      ss << ">";
    } else if (buffer_scope == DstBufferScope::Ub) {
      // gm_to_ub: N M (reversed order)
      // threads=2: first dim (M) divided by vector_cnt, each vidcopieshalf
      int idx = 0;
      for (auto it = ws_info.shapes.rbegin(); it != ws_info.shapes.rend();
           ++it, ++idx) {
        int shape_val = (*it).as<IntImmNode>()->value;
        // Last element is original shapes[0] (M), needs division by vector_cnt
        if (idx == ws_info.shapes.size() - 1 &&
            core_meta_info__.vid_var.defined() &&
            core_meta_info__.vector_cnt > 1) {
          shape_val /= core_meta_info__.vector_cnt;
        }
        ss << ", " << shape_val;
      }
      ss << ">";
    }

    std::string stmt_name_str = ss.str();

    StringImm stmt_name(stmt_name_str);
    args.push_back(stmt_name);
    args.push_back(src_access);
    args.push_back(dst_access);
    args.push_back(strideN);

    // Add extra parameters for codegen
    if (buffer_scope == DstBufferScope::L1) {
      // copy_gm_to_l1: add realTailM=0, realTailN=0
      args.push_back(IntImm(DataType::Int(32), 0));
      args.push_back(IntImm(DataType::Int(32), 0));
    } else if (buffer_scope == DstBufferScope::Ub) {
      // copy_gm_to_ub: add maskShapeM, maskShapeN, padValue=0
      // ws_info.shapes = [M, N] (workspace shape, M includes vector_cnt)
      // Template params: dstN=N, dstM=M/vector_cnt (from reversed shapes)
      // So maskShapeM=M/vector_cnt, maskShapeN=N
      ICHECK(ws_info.shapes.size() >= 2);
      // Extract M and N from ws_info.shapes
      int M_val = ws_info.shapes[0].as<IntImmNode>()->value;
      int N_val = ws_info.shapes[1].as<IntImmNode>()->value;
      // maskShapeM considers vector_cnt (threads=2 scenario)
      if (core_meta_info__.vid_var.defined() &&
          core_meta_info__.vector_cnt > 1) {
        M_val /= core_meta_info__.vector_cnt;
      }
      args.push_back(IntImm(DataType::Int(32), M_val)); // maskShapeM
      args.push_back(IntImm(DataType::Int(32), N_val)); // maskShapeN
      args.push_back(IntImm(DataType::Int(32), 0));     // padValue
    }

    Call new_call_node(DataType::Handle(), tvm::tir::builtin::call_extern(),
                       args);

    return Evaluate(new_call_node);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    Stmt orig_stmt = arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    const EvaluateNode *orig_eval_node = orig_stmt.as<EvaluateNode>();
    if (!orig_eval_node) {
      return orig_stmt;
    }

    const CallNode *orig_call = orig_eval_node->value.as<CallNode>();
    if (!orig_call) {
      return orig_stmt;
    }

    // PTO path: replace cross-core copies with pipe pairs
    if (pto_use_pipe_ && orig_call->op == tir::builtin::call_extern() &&
        orig_call->args.size() > 0 &&
        orig_call->args[0].as<StringImmNode>()) {
      std::string func_name_str =
          Downcast<StringImm>(orig_call->args[0])->value;
      if (func_name_str.find("copy_ub_to_l1") != std::string::npos ||
          func_name_str.find("copy_l0c_to_ub") != std::string::npos) {
        return GenerateCopyToPipe(orig_call, func_name_str);
      }
    }

    // In-place transform (AscendC path)
    if (IsTargetCopyExpr(orig_call)) {
      // Build new_args from scratch keeping only the first 4 original args.
      // Extra runtime args (src_M/dst_M/dst_N at orig_call->args[4..6])
      // from ascend.cc are dropped here. AscendC codegen expects positional
      // args[3..5] = (count, maskShapeM, maskShapeN) appended via push_back below.
      Array<PrimExpr> new_args;
      new_args.push_back(orig_call->args[0]);  // function name string
      new_args.push_back(orig_call->args[1]);  // src access_ptr
      new_args.push_back(orig_call->args[2]);  // dst access_ptr (replaced below)
      new_args.push_back(orig_call->args[3]);  // src_N (element count, used as count in codegen)
      StringImm orig_func_name = Downcast<StringImm>(new_args[0]);
      const CallNode *src_access_ptr = new_args[1].as<CallNode>();
      std::string src_buffer_name =
          (src_access_ptr->args[1]).as<VarNode>()->name_hint;

      // copyA (in-place): GM's other side is source, check if source is UB
      // ub_to_l1 source is UB; l0c_to_ub source is L0C
      std::string orig_func_name_str = orig_func_name->value;
      bool other_side_is_ub =
          (orig_func_name_str.find("copy_ub_to_l1") != std::string::npos);

      StringImm new_func_name = ReplaceCopyFuncName(orig_func_name);
      new_args.Set(0, new_func_name);
      std::string ws_name = context_.src_to_workspace_map_.at(src_buffer_name);
      WorkspaceInfo ws_info = context_.workspace_map_.at(ws_name);
      new_args.Set(2, CreateWorkspaceAccessPtr(ws_info, 2, other_side_is_ub));

      // Add extra parameters for codegen
      if (orig_func_name_str.find("copy_ub_to_l1") != std::string::npos) {
        // copy_ub_to_gm: add maskShapeM, maskShapeN

        // Check if src_buffer is in skip_vid_reduction set
        bool is_skip_ub =
            context_.buffers_skip_vid_reduction_.count(src_buffer_name) > 0;

        if (is_skip_ub) {
          // Skip UB: use original UB shape from buffer_shapes_
          // These UB buffers not processed by VidReduction, vector split in
          // index array

          auto shape_it = context_.buffer_shapes_.find(src_buffer_name);
          ICHECK(shape_it != context_.buffer_shapes_.end());

          // buffer_shapes_ can be 1D [N] or 2D [M, N]
          int M_val = 1; // colsefault value
          int N_val = 1;
          if (shape_it->second.size() >= 1) {
            N_val = shape_it->second[shape_it->second.size() - 1]
                        .as<IntImmNode>()
                        ->value;
          }
          if (shape_it->second.size() >= 2) {
            M_val = shape_it->second[shape_it->second.size() - 2]
                        .as<IntImmNode>()
                        ->value;
          }

          new_args.push_back(IntImm(DataType::Int(32), M_val)); // maskShapeM
          new_args.push_back(IntImm(DataType::Int(32), N_val)); // maskShapeN
        } else {
          // Normal UB: use ws_info.shapes, considering vector_cnt
          // ws_info.shapes = [M * vector_cnt, N] (workspace stores complete
          // data) maskShapeM = M = ws_info.shapes[0] / vector_cnt (each
          // vidcopieshalf maskShapeN = N = ws_info.shapes[1]
          ICHECK(ws_info.shapes.size() >= 2);
          int M_val = ws_info.shapes[0].as<IntImmNode>()->value;
          int N_val = ws_info.shapes[1].as<IntImmNode>()->value;
          // threads=2: each vidcopies M/vector_cnt
          if (core_meta_info__.vid_var.defined() &&
              core_meta_info__.vector_cnt > 1) {
            M_val /= core_meta_info__.vector_cnt;
          }
          new_args.push_back(IntImm(DataType::Int(32), M_val)); // maskShapeM
          new_args.push_back(IntImm(DataType::Int(32), N_val)); // maskShapeN
        }
      } else if (orig_func_name_str.find("copy_l0c_to_ub") !=
                 std::string::npos) {
        // copy_l0c_to_gm: add realTailM=0, realTailN=0
        new_args.push_back(IntImm(DataType::Int(32), 0));
        new_args.push_back(IntImm(DataType::Int(32), 0));
      }

      Call new_call =
          Call(orig_call->dtype, orig_call->op, new_args, orig_call->span);
      return Evaluate(new_call);
    }

    // Insert copy-back statements (for both AscendC and PTO)
    std::vector<std::string> buffer_names_vec =
        IsTargetAccessNode(orig_call);
    if (buffer_names_vec.empty()) {
      return orig_stmt;
    }
    Array<Stmt> seq_stmts_array;
    for (const auto &buffer_name : buffer_names_vec) {
      if (pto_use_pipe_) {
        // PTO: generate copy_pipe_to_xx from PipeInfo
        auto pipe_it = context_.pipe_info_map_.find(buffer_name);
        ICHECK(pipe_it != context_.pipe_info_map_.end())
            << "[Error]<WorkspaceReduction> PTO: No pipe metadata found for buffer: "
            << buffer_name;
        seq_stmts_array.push_back(
            GenerateCopyFromPipe(pipe_it->second,
                              context_.dst_to_access_map_.at(buffer_name)));
      } else {
        // AscendC: existing copy_gm_to_dst logic
        auto buffer_scope_it =
            context_.dst_to_scope_map_.find(buffer_name);
        ICHECK(buffer_scope_it != context_.dst_to_scope_map_.end())
            << "[Error]<WorkspaceReduction>: Dst scope is not found for "
              "buffer: "
            << buffer_name << "\n";

        DstBufferScope buffer_scope =
            context_.dst_to_scope_map_.at(buffer_name);
        const std::string workspace_name =
            context_.dst_to_workspace_map_.at(buffer_name);
        WorkspaceInfo ws_info =
            context_.workspace_map_.at(workspace_name);

          // copyB (copy-back): GM's other side is destination, check if dst is UB
          // gm_to_ub destination is UB; gm_to_l1 destination is L1
        bool other_side_is_ub = (buffer_scope == DstBufferScope::Ub);
        Call workspace_access =
            CreateWorkspaceAccessPtr(ws_info, 1, other_side_is_ub);
        PrimExpr strideN = ws_info.shapes[ws_info.shapes.size() - 1];
        Stmt insert_eval_stmt = this->CreateGmToDstStmt(
            workspace_access,
            context_.dst_to_access_map_.at(buffer_name), buffer_scope,
            strideN);
        seq_stmts_array.push_back(insert_eval_stmt);
      }
    }
    seq_stmts_array.push_back(orig_stmt);
    return SeqStmt(seq_stmts_array);
  }
};


namespace transform {

using namespace tir::transform;

tvm::transform::Pass AscendWorkspaceReduction() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    bool is_pto = false;
    if (auto opt_target = f->GetAttr<Target>(tvm::attr::kTarget)) {
      Target target = opt_target.value();
      if (target->attrs.defined()) {
        auto model_attr = target->attrs.Get("model");
        if (model_attr.defined()) {
          is_pto = (Downcast<String>(model_attr) == "pto");
        }
      }
    }
    std::string platform = "A3";
    auto platform_attr = f->GetAttr<String>("npu_platform");
    if (platform_attr.defined()) {
      platform = platform_attr.value();
    }
    bool use_pipe_config = ctx->GetConfig<Bool>(ascendPtoUsePipeInCVCopy, Bool(true)).value();
    bool pto_use_pipe = is_pto && use_pipe_config;
    return AscendWorkspaceReductionPass::Substitute(std::move(f), pto_use_pipe, platform);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendWorkspaceReduction", {});
}

TVM_REGISTER_GLOBAL("tl.transform.AscendWorkspaceReduction")
    .set_body_typed(AscendWorkspaceReduction);
} // namespace transform

} // namespace tl
} // namespace tvm