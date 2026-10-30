// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/ir.cc
 * \brief Extension for the tvm script frontend.
 *
 */

#include "./transform/common/ascend_constraint.h"
#include "./transform/common/attr.h"
#include "op/builtin.h"
#include <tvm/arith/analyzer.h>
#include <tvm/script/ir_builder/tir/ir.h>

#include <limits>

namespace tvm {
namespace tl {

using namespace script::ir_builder::tir;

static Var CreateEnvThread(String name, String thread_tag, DataType dtype) {
  using namespace tvm::tir;
  using namespace tvm::script::ir_builder;
  IterVar iter_var(Range{nullptr}, Var(name, dtype),
                   tvm::tir::IterVarType::kThreadIndex, thread_tag);
  Var var = iter_var->var;
  if (Optional<PrimFuncFrame> opt_frame =
          IRBuilder::Current()->FindFrame<PrimFuncFrame>()) {
    opt_frame.value()->env_threads.Set(var, iter_var);
  } else {
    LOG(FATAL) << "EnvThread can only be used inside a PrimFunc";
  }
  return var;
}

static ForFrame MakeIterVarFrame(std::string name, PrimExpr dom) {
  using namespace tvm::tir;
  Var var = Var(name, dom->dtype);
  // Create a frame that represents a loop over the given domain.
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  n->vars.push_back(var);
  n->doms.push_back(Range(0, dom));
  n->f_make_for_loop = [](Array<Var> vars, Array<Range> doms,
                          Stmt body) -> Stmt {
    ICHECK_EQ(vars.size(), 1);
    ICHECK_EQ(doms.size(), 1);
    return For(vars[0], doms[0]->min, doms[0]->extent, ForKind::kSerial, body);
  };
  return ForFrame(n);
}

static void ValidateLoopAnnotations(const Map<String, ObjectRef> &annotations,
                                    bool allow_offset = false) {
  ICHECK(annotations.defined()) << "Loop annotations must be a non-null Map";
  for (const auto &entry : annotations) {
    std::string key = entry.first;
    if (key == "enable_offset" && allow_offset) {
      const auto *value = entry.second.as<tvm::tir::IntImmNode>();
      ICHECK(value && value->dtype.is_bool())
          << "enable_offset must be boolean";
      continue;
    }
    ICHECK(key != kAscendPersistentGuard && key != kAscendPipelineLoopId &&
           key != kAscendPipelineVersions && key != "num_stages" &&
           key != "tl_pipeline_order" && key != "tl_pipeline_stage" &&
           key != "tl_pipeline_sync" && key != "tl_pipeline_group" &&
           key != "tl_cross_interval" && key != "tl.ascend_task" &&
           key != "tl.ascend_stage" && key != "enable_offset" &&
           key != "parallel_loop_layout" &&
           key.rfind("software_pipeline_", 0) != 0 &&
           key.rfind("tl.ascend_storage_", 0) != 0 &&
           key.rfind("tl.ascend_handoff_", 0) != 0 &&
           key.rfind("tl.ascend_local_sync_", 0) != 0 &&
           key.rfind("tl.ascend_auto_schedule", 0) != 0)
        << "Loop annotation '" << key
        << "' is reserved; use its dedicated supported API";
  }
}

ForFrame ParallelFor(Array<PrimExpr> extents,
                     Map<String, ObjectRef> annotations) {
  using namespace tvm::tir;
  ValidateLoopAnnotations(annotations);
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  n->vars.reserve(extents.size());
  n->doms.reserve(extents.size());
  for (const auto &extent : extents) {
    DataType dtype = extent.dtype();
    n->vars.push_back(Var("v", extent.dtype()));
    n->doms.push_back(Range(make_const(dtype, 0), extent));
  }
  n->f_make_for_loop = [annotations](Array<Var> vars, Array<Range> doms,
                                     Stmt body) -> Stmt {
    ICHECK_EQ(vars.size(), doms.size());
    int n = vars.size();
    for (int i = n - 1; i >= 0; --i) {
      Range dom = doms[i];
      Var var = vars[i];
      // Keep the legacy coalesced-width hint on every generated dimension;
      // new user metadata belongs to the outermost loop only.
      Map<String, ObjectRef> loop_annotations;
      if (i == 0) {
        loop_annotations = annotations;
      } else if (annotations.count("coalesced_width")) {
        loop_annotations.Set("coalesced_width",
                             annotations.at("coalesced_width"));
      }
      body =
          For(var, dom->min, dom->extent, ForKind::kParallel, std::move(body),
              /*thread_binding=*/NullOpt, /*annotations=*/loop_annotations);
    }
    return body;
  };
  return ForFrame(n);
}

ForFrame PipelinedFor(PrimExpr start, PrimExpr stop, int num_stages,
                      Array<PrimExpr> order, Array<PrimExpr> stages,
                      Array<Array<PrimExpr>> sync,
                      Array<Array<PrimExpr>> groups, int cross_interval,
                      Map<String, ObjectRef> annotations) {
  using namespace tvm::tir;
  ValidateLoopAnnotations(annotations, true);
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  DataType dtype = stop.dtype();
  n->vars.push_back(Var("v", dtype));
  n->doms.push_back(Range(start, stop));
  n->f_make_for_loop = [=](Array<Var> vars, Array<Range> doms,
                           Stmt body) -> Stmt {
    ICHECK_EQ(vars.size(), doms.size());
    int n = vars.size();
    ICHECK(n == 1);
    Map<String, ObjectRef> anno = annotations;
    if (num_stages > 0)
      anno.Set("num_stages", PrimExpr(num_stages));
    if (order.size() > 0)
      anno.Set("tl_pipeline_order", order);
    if (stages.size() > 0)
      anno.Set("tl_pipeline_stage", stages);
    if (sync.size() > 0)
      anno.Set("tl_pipeline_sync", sync);
    if (groups.size() > 0)
      anno.Set("tl_pipeline_group", groups);
    if (cross_interval > 0)
      anno.Set("tl_cross_interval", PrimExpr(cross_interval));
    body = For(vars[0], doms[0]->min, doms[0]->extent, ForKind::kSerial,
               std::move(body),
               /*thread_binding=*/NullOpt, /*annotations=*/anno);
    return body;
  };
  return ForFrame(n);
}

ForFrame PersistentFor(const Array<PrimExpr> &domain, const PrimExpr &wave_size,
                       const PrimExpr &index, PrimExpr group_size,
                       int num_stages, Map<String, ObjectRef> annotations) {
  using namespace tvm::tir;
  ValidateLoopAnnotations(annotations);
  ICHECK(!domain.empty()) << "T.Persistent domain must not be empty";
  ICHECK_GE(num_stages, 0) << "T.Persistent num_stages must be nonnegative";
  auto validate_integer = [](const PrimExpr &value, const char *name) {
    DataType dtype = value.dtype();
    ICHECK((dtype.is_int() || dtype.is_uint()) && !dtype.is_bool() &&
           dtype.is_scalar() && dtype.bits() <= 64 &&
           !(dtype.is_uint() && dtype.bits() == 64))
        << "T.Persistent " << name << " must be an integer scalar, but got "
        << dtype << "; scheduling arithmetic requires lossless int64 promotion";
  };
  validate_integer(wave_size, "wave_size");
  validate_integer(index, "index");
  validate_integer(group_size, "group_size");
  arith::Analyzer validation_analyzer;
  for (const auto &extent : domain) {
    validate_integer(extent, "domain extents");
    ICHECK(!validation_analyzer.CanProveLess(extent, 1))
        << "T.Persistent domain extents must be positive, but got " << extent;
  }
  ICHECK(!validation_analyzer.CanProveLess(wave_size, 1))
      << "T.Persistent wave_size must be positive, but got " << wave_size;
  PrimExpr wide_wave_size = cast(DataType::Int(64), wave_size);
  PrimExpr wide_index = cast(DataType::Int(64), index);
  ICHECK(!validation_analyzer.CanProveLess(wide_index, 0) &&
         !validation_analyzer.CanProve(wide_index >= wide_wave_size))
      << "T.Persistent index must satisfy 0 <= index < wave_size, but got "
      << "index=" << index << " and wave_size=" << wave_size;
  ICHECK(!validation_analyzer.CanProveLess(group_size, 1))
      << "T.Persistent group_size must be positive, but got " << group_size;
  // Promote operands before multiplication or ceil division. Coordinate vars
  // retain the caller's dtype; only scheduling arithmetic is widened.
  PrimExpr last_extent = cast(DataType::Int(64), domain[domain.size() - 1]);
  PrimExpr effective_group_size =
      min(cast(DataType::Int(64), group_size), last_extent);
  bool needs_tile_guard = !validation_analyzer.CanProveEqual(
      truncmod(last_extent, effective_group_size), 0);
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  n->vars.reserve(domain.size());
  n->doms.reserve(domain.size());
  group_size = effective_group_size;
  for (int i = 0; i < domain.size(); ++i) {
    DataType dtype = domain[i].dtype();
    Var coord("v" + std::to_string(i), dtype);
    n->vars.push_back(coord);
    n->doms.push_back(Range(make_const(dtype, 0), domain[i]));
  }

  Array<PrimExpr> grouped_domain;
  auto positive_ceildiv = [](PrimExpr a, PrimExpr b) {
    // Unlike (a + b - 1) / b, this does not overflow before division.
    return truncdiv(a, b) + cast(DataType::Int(64), truncmod(a, b) != 0);
  };
  grouped_domain.push_back(positive_ceildiv(last_extent, group_size));
  for (int i = 0; i < domain.size() - 1; ++i) {
    grouped_domain.push_back(cast(DataType::Int(64), domain[i]));
  }
  grouped_domain.push_back(group_size);
  PrimExpr padded_size = grouped_domain[0];
  for (int i = 1; i < grouped_domain.size(); ++i) {
    PrimExpr lhs_value = validation_analyzer.Simplify(padded_size);
    PrimExpr rhs_value = validation_analyzer.Simplify(grouped_domain[i]);
    const int64_t *lhs = as_const_int(lhs_value);
    const int64_t *rhs = as_const_int(rhs_value);
    ICHECK(!lhs || !rhs || *lhs <= std::numeric_limits<int64_t>::max() / *rhs)
        << "T.Persistent padded domain size must fit int64";
    padded_size *= grouped_domain[i];
  }
  auto waves = positive_ceildiv(padded_size, wide_wave_size);
  PrimExpr constant_waves = validation_analyzer.Simplify(waves);
  PrimExpr constant_wave_size = validation_analyzer.Simplify(wide_wave_size);
  const int64_t *wave_count_value = as_const_int(constant_waves);
  const int64_t *wave_size_value = as_const_int(constant_wave_size);
  ICHECK(!wave_count_value || !wave_size_value ||
         *wave_count_value - 1 <=
             (std::numeric_limits<int64_t>::max() - (*wave_size_value - 1)) /
                 *wave_size_value)
      << "T.Persistent padded wave index range must fit int64";
  auto loop_var = Var("w", waves.dtype());
  bool guarded_positive =
      num_stages > 0 && (needs_tile_guard || !validation_analyzer.CanProveEqual(
                                                 padded_size, wide_wave_size));

  n->f_make_for_loop = [=](const Array<Var> &vars, const Array<Range> &doms,
                           const Stmt &body) -> Stmt {
    ICHECK_EQ(vars.size(), doms.size());
    Map<String, ObjectRef> anno = annotations;
    if (num_stages > 0) {
      anno.Set("num_stages", PrimExpr(num_stages));
      if (guarded_positive)
        anno.Set(kAscendPersistentGuard, Bool(true));
    }
    Array<PrimExpr> idxs(grouped_domain.size(), PrimExpr());
    PrimExpr global_index = loop_var * wide_wave_size + wide_index;
    PrimExpr rem = global_index;

    for (int i = grouped_domain.size() - 1; i >= 1; --i) {
      idxs.Set(i, truncmod(rem, grouped_domain[i]));
      rem = truncdiv(rem, grouped_domain[i]);
    }
    idxs.Set(0, rem);

    auto out_if = tvm::tir::IfThenElse(
        padded_size <= global_index,
        tvm::tir::Evaluate(
            tvm::tir::Call(DataType::Handle(), tvm::tl::loop_break(), {})),
        Stmt());

    arith::Analyzer analyzer;
    Stmt new_body = body;
    if (num_stages > 0) {
      PrimExpr valid = global_index < padded_size;
      if (needs_tile_guard) {
        PrimExpr last_coordinate = idxs[0] * group_size + idxs[vars.size()];
        valid = valid && last_coordinate < last_extent;
      }
      if (guarded_positive)
        new_body = tvm::tir::IfThenElse(valid, new_body, Stmt());
    } else {
      if (needs_tile_guard) {
        PrimExpr last_coordinate = idxs[0] * group_size + idxs[vars.size()];
        new_body = tvm::tir::IfThenElse(last_coordinate < last_extent, new_body,
                                        Stmt());
      }
      if (analyzer.CanProveGreaterEqual(waves, 2)) {
        new_body = SeqStmt({out_if, new_body});
      } else if (!analyzer.CanProveEqual(padded_size, wide_wave_size)) {
        new_body =
            tvm::tir::IfThenElse(global_index < padded_size, new_body, Stmt());
      }
    }
    Stmt outer = For(loop_var, make_const(loop_var.dtype(), 0), waves,
                     ForKind::kSerial, new_body, NullOpt, anno);
    for (int i = 0; i < vars.size() - 1; ++i) {
      outer =
          tvm::tir::LetStmt(vars[i], cast(vars[i].dtype(), idxs[i + 1]), outer);
    }
    outer = tvm::tir::LetStmt(vars[vars.size() - 1],
                              cast(vars[vars.size() - 1].dtype(),
                                   idxs[0] * group_size + idxs[vars.size()]),
                              outer);
    return outer;
  };

  return ForFrame(n);
}

/*!
 * \brief A frame that represents a kernel launch.
 *
 * \sa KernelLaunchFrameNode
 */
class KernelLaunchFrameNode : public TIRFrameNode {
public:
  Array<TIRFrame> frames;

  void VisitAttrs(tvm::AttrVisitor *v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("frames", &frames);
  }

  static constexpr const char *_type_key = "tl.KernelLaunchFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(KernelLaunchFrameNode, TIRFrameNode);

public:
  TVM_DLL void EnterWithScope() final {
    for (auto frame = frames.begin(); frame != frames.end(); ++frame)
      (*frame)->EnterWithScope();
  }
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  TVM_DLL void ExitWithScope() final {
    for (auto frame = frames.rbegin(); frame != frames.rend(); ++frame)
      (*frame)->ExitWithScope();
  }
};

/*!
 * \brief Managed reference to KernelLaunchFrameNode.
 *
 * \sa KernelLaunchFrameNode
 */
class KernelLaunchFrame : public TIRFrame {
public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(KernelLaunchFrame, TIRFrame,
                                                    KernelLaunchFrameNode);
};

KernelLaunchFrame KernelLaunch(Array<PrimExpr> grid_size,
                               Array<PrimExpr> block_size,
                               Map<String, ObjectRef> attrs) {
  ObjectPtr<KernelLaunchFrameNode> n = make_object<KernelLaunchFrameNode>();

  // If the kernel is a CPU kernel, we don't need to launch any threads.
  bool is_cpu_kernel_frame =
      attrs.defined() && attrs.count(tilelang_is_cpu_kernel_frame);
  bool is_npu_kernel_frame =
      attrs.defined() && attrs.count(tilelang_is_npu_kernel_frame);
  bool is_npu_kernel_frame_dev_mode =
      attrs.defined() && attrs.count(tilelang_is_npu_kernel_frame_dev_mode);

  if (is_cpu_kernel_frame) {
    // Launch CPU Kernel
    ICHECK(grid_size.size() >= 0);
    ICHECK(block_size.size() == 0) << "CPU kernel cannot have block size";
    ICHECK(attrs.defined());
    // create grid loop var
    for (int i = 0; i < grid_size.size(); i++) {
      n->frames.push_back(
          MakeIterVarFrame("block_var_" + std::to_string(i), grid_size[i]));
    }
    // Launch CPU Kernel
  } else if (is_npu_kernel_frame) { // Launch NPU Kernel
    ICHECK(grid_size.size() == 1) << "NPU kernel only supports 1D grid size";

    // Note: corresponding relationship ->
    // blockIdx.x => cube idx in ascend
    // blockIdx.y => vec idx in ascend
    n->frames.push_back(
        LaunchThread(CreateEnvThread("cid", "blockIdx.x", grid_size[0].dtype()),
                     grid_size[0]));
    n->frames.push_back(LaunchThread(
        CreateEnvThread("vid", "blockIdx.y", grid_size[0].dtype()), 2));
  } else if (is_npu_kernel_frame_dev_mode) {
    // Launch NPU Kernel In Developer Mode
    ICHECK(grid_size.size() == 1) << "NPU kernel only supports 1D grid size";
    ICHECK(block_size.size() == 1)
        << "Vid(Threads) only supports at 1D block size in developer mode";
    // develop mode without vid
    n->frames.push_back(
        LaunchThread(CreateEnvThread("cid", "blockIdx.x", grid_size[0].dtype()),
                     grid_size[0]));
    n->frames.push_back(LaunchThread(
        CreateEnvThread("vid", "threadIdx.x", block_size[0].dtype()),
        block_size[0]));

    using namespace tvm::script::ir_builder;
    Optional<PrimFuncFrame> opt_prim_func_frame =
        IRBuilder::Current()->FindFrame<PrimFuncFrame>();
    ICHECK(opt_prim_func_frame.defined())
        << "KernelLaunch must be inside PrimFunc";
    PrimFuncFrame prim_func_frame = opt_prim_func_frame.value();
    const IntImmNode *vid_value_node = block_size[0].as<IntImmNode>();
    ICHECK(vid_value_node != nullptr)
        << "block_size[0] must be compile-time constant";
    int vid_value = vid_value_node->value;
    if (vid_value == 1) {
      prim_func_frame->attrs.Set("npu_cv_ratio", StringImm(cv_1_1));
    } else {
      prim_func_frame->attrs.Set("npu_cv_ratio", StringImm(cv_1_2));
    }
  } else {
    // Launch GPU Kernel
    ICHECK(grid_size.size() <= 3);
    if (grid_size.size() > 0)
      n->frames.push_back(LaunchThread(
          CreateEnvThread("bx", "blockIdx.x", grid_size[0].dtype()),
          grid_size[0]));
    if (grid_size.size() > 1)
      n->frames.push_back(LaunchThread(
          CreateEnvThread("by", "blockIdx.y", grid_size[1].dtype()),
          grid_size[1]));
    if (grid_size.size() > 2)
      n->frames.push_back(LaunchThread(
          CreateEnvThread("bz", "blockIdx.z", grid_size[2].dtype()),
          grid_size[2]));
    if (block_size.defined()) {
      ICHECK(block_size.size() <= 3);
      if (block_size.size() > 0) {
        n->frames.push_back(LaunchThread(
            CreateEnvThread("tx", "threadIdx.x", block_size[0].dtype()),
            block_size[0]));
      }
      if (block_size.size() > 1) {
        n->frames.push_back(LaunchThread(
            CreateEnvThread("ty", "threadIdx.y", block_size[1].dtype()),
            block_size[1]));
      }
      if (block_size.size() > 2) {
        n->frames.push_back(LaunchThread(
            CreateEnvThread("tz", "threadIdx.z", block_size[2].dtype()),
            block_size[2]));
      }
    }
  }

  if (attrs.defined()) {
    auto empty_block = ::tvm::script::ir_builder::tir::Block(MainBlockName);
    empty_block->annotations = attrs;
    n->frames.push_back(empty_block);
  } else {
    n->frames.push_back(::tvm::script::ir_builder::tir::Block(MainBlockName));
  }

  return KernelLaunchFrame(n);
}

TVM_REGISTER_NODE_TYPE(KernelLaunchFrameNode);

TVM_REGISTER_GLOBAL("tl.Parallel").set_body_typed(ParallelFor);
TVM_REGISTER_GLOBAL("tl.Pipelined").set_body([](TVMArgs args, TVMRetValue *rv) {
  ICHECK(args.size() == 8 || args.size() == 9)
      << "tl.Pipelined expects 8 legacy arguments or 9 arguments with "
         "annotations";
  Map<String, ObjectRef> annotations;
  if (args.size() == 9) {
    annotations = args[8].operator Map<String, ObjectRef>();
  }
  *rv = PipelinedFor(args[0], args[1], args[2], args[3], args[4], args[5],
                     args[6], args[7], annotations);
});
TVM_REGISTER_GLOBAL("tl.Persistent")
    .set_body([](TVMArgs args, TVMRetValue *rv) {
      ICHECK(args.size() == 4 || args.size() == 6)
          << "tl.Persistent expects 4 legacy arguments or 6 arguments with "
             "stages and annotations";
      int num_stages = 0;
      Map<String, ObjectRef> annotations;
      if (args.size() == 6) {
        num_stages = args[4];
        annotations = args[5].operator Map<String, ObjectRef>();
      }
      *rv = PersistentFor(args[0], args[1], args[2], args[3], num_stages,
                          annotations);
    });
TVM_REGISTER_GLOBAL("tl.KernelLaunch").set_body_typed(KernelLaunch);

class WarpSpecializeFrameNode : public TIRFrameNode {
public:
  Array<TIRFrame> frames;

  void VisitAttrs(tvm::AttrVisitor *v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("frames", &frames);
  }

  static constexpr const char *_type_key = "tl.WarpSpecializeFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(WarpSpecializeFrameNode, TIRFrameNode);

public:
  TVM_DLL void EnterWithScope() final {
    for (auto frame = frames.begin(); frame != frames.end(); ++frame)
      (*frame)->EnterWithScope();
  }
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  TVM_DLL void ExitWithScope() final {
    for (auto frame = frames.rbegin(); frame != frames.rend(); ++frame)
      (*frame)->ExitWithScope();
  }
};

class WarpSpecializeFrame : public TIRFrame {
public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(WarpSpecializeFrame,
                                                    TIRFrame,
                                                    WarpSpecializeFrameNode);
};

WarpSpecializeFrame WarpSpecialize(Array<IntImm> warp_group_ids,
                                   PrimExpr thread_idx,
                                   int warp_group_size = 128) {
  ObjectPtr<WarpSpecializeFrameNode> n = make_object<WarpSpecializeFrameNode>();
  PrimExpr condition;
  std::vector<int> warp_groups;
  for (int i = 0; i < warp_group_ids.size(); i++) {
    warp_groups.push_back(Downcast<IntImm>(warp_group_ids[i])->value);
  }
  std::sort(warp_groups.begin(), warp_groups.end());

  // Merge consecutive groups
  std::vector<std::pair<int, int>> merged;
  for (int group : warp_groups) {
    if (merged.empty() || group != merged.back().second) {
      merged.emplace_back(group, group + 1);
    } else {
      merged.back().second = group + 1;
    }
  }

  for (const auto &[start, end] : merged) {
    PrimExpr min_bound = IntImm(thread_idx.dtype(), start) * warp_group_size;
    PrimExpr max_bound = IntImm(thread_idx.dtype(), end) * warp_group_size;
    PrimExpr range_cond = (thread_idx >= min_bound) && (thread_idx < max_bound);

    if (condition.defined()) {
      condition = tir::Or(condition, range_cond);
    } else {
      condition = range_cond;
    }
  }
  IfFrame if_frame = If(condition);
  AttrFrame attr_frame = Attr(Integer(0), "warp_specialize", Integer(1));
  n->frames.push_back(if_frame);
  n->frames.push_back(Then());
  n->frames.push_back(attr_frame);
  return WarpSpecializeFrame(n);
}

TVM_REGISTER_NODE_TYPE(WarpSpecializeFrameNode);
TVM_REGISTER_GLOBAL("tl.WarpSpecialize").set_body_typed(WarpSpecialize);

class ResourceSpecializeFrameNode : public TIRFrameNode {
public:
  Array<TIRFrame> frames;

  void VisitAttrs(tvm::AttrVisitor *v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("frames", &frames);
  }

  static constexpr const char *_type_key = "tl.ResourceSpecializeFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(ResourceSpecializeFrameNode, TIRFrameNode);

public:
  TVM_DLL void EnterWithScope() final {
    for (auto frame = frames.begin(); frame != frames.end(); ++frame)
      (*frame)->EnterWithScope();
  }
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  TVM_DLL void ExitWithScope() final {
    for (auto frame = frames.rbegin(); frame != frames.rend(); ++frame)
      (*frame)->ExitWithScope();
  }
};

class ResourceSpecializeFrame : public TIRFrame {
public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(
      ResourceSpecializeFrame, TIRFrame, ResourceSpecializeFrameNode);
};

ResourceSpecializeFrame ResourceSpecialize(String resource_name) {
  ObjectPtr<ResourceSpecializeFrameNode> n =
      make_object<ResourceSpecializeFrameNode>();
  AttrFrame attr_frame = Attr(Integer(0), resource_name, Integer(1));
  n->frames.push_back(attr_frame);
  return ResourceSpecializeFrame(n);
}

TVM_REGISTER_NODE_TYPE(ResourceSpecializeFrameNode);
TVM_REGISTER_GLOBAL("tl.ResourceSpecialize").set_body_typed(ResourceSpecialize);

class ScopeFrameNode : public TIRFrameNode {
public:
  Array<TIRFrame> frames;

  void VisitAttrs(tvm::AttrVisitor *v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("frames", &frames);
  }

  static constexpr const char *_type_key = "tl.ScopeFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScopeFrameNode, TIRFrameNode);

public:
  TVM_DLL void EnterWithScope() final {
    for (auto frame = frames.begin(); frame != frames.end(); ++frame)
      (*frame)->EnterWithScope();
  }
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  TVM_DLL void ExitWithScope() final {
    for (auto frame = frames.rbegin(); frame != frames.rend(); ++frame)
      (*frame)->ExitWithScope();
  }
};

class ScopeFrame : public TIRFrame {
public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(ScopeFrame, TIRFrame,
                                                    ScopeFrameNode);
};

ScopeFrame Scope(String scope_name) {
  ObjectPtr<ScopeFrameNode> n = make_object<ScopeFrameNode>();
  int scope_id = 0;
  if (scope_name == "V")
    scope_id = 1;
  AttrFrame attr_frame = Attr(Integer(0), "resource_scope", Integer(scope_id));
  n->frames.push_back(attr_frame);
  return ScopeFrame(n);
}

TVM_REGISTER_NODE_TYPE(ScopeFrameNode);
TVM_REGISTER_GLOBAL("tl.Scope").set_body_typed(Scope);

} // namespace tl
} // namespace tvm
