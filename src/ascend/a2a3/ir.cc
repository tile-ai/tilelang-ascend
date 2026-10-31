// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
/*!
 * \file ascend/a2a3/ir.cc
 * \brief TA's NPU launch and resource-scope frames on the neutral frontend.
 *
 * Ported from TileLang-Ascend src/ir.cc at 0deb8b93. The shared launch frame
 * owns the Python stack and binding API. Only these A2/A3 factories are new;
 * they do not replace the official A5 mixed launch or its execution regions.
 */

#include "../../ir.h"
#include "../../op/builtin.h"

#include <tvm/arith/analyzer.h>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/script/ir_builder/base.h>
#include <tvm/script/ir_builder/tir/ir.h>
#include <tvm/tirx/stmt.h>

#include "../../transform/common/attr.h"

namespace tvm {
namespace tl {
namespace a2a3 {

using namespace script::ir_builder::tirx;
using namespace ffi;

KernelLaunchFrame KernelLaunch(const Array<PrimExpr> &grid_size,
                               const Optional<Array<PrimExpr>> &block_size,
                               const Map<String, Any> &attrs) {
  ICHECK_EQ(grid_size.size(), 1)
      << "A2/A3 NPU kernel must have exactly one block dimension";
  bool developer_mode = block_size.defined();
  PrimExpr vector_extent = Integer(2);
  if (developer_mode) {
    ICHECK_EQ(block_size.value().size(), 1)
        << "A2/A3 NPU kernel must have exactly one thread dimension";
    vector_extent = block_size.value()[0];
  }
  const auto *vector_count = vector_extent.as<IntImmNode>();
  ICHECK(vector_count && (vector_count->value == 1 || vector_count->value == 2))
      << "A2/A3 NPU kernel threads must be the constant integer 1 or 2, got "
      << vector_extent;

  ObjectPtr<KernelLaunchFrameNode> n =
      tvm::ffi::make_object<KernelLaunchFrameNode>();
  ForFrame cid_frame =
      MakeThreadBindingFrame("cid", "blockIdx.x", grid_size[0]);
  n->frames.push_back(cid_frame);
  n->grid_vars.push_back(cid_frame->vars[0]);
  n->grid_extents.push_back(grid_size[0]);

  ForFrame vid_frame = MakeThreadBindingFrame("vid", "cthread", vector_extent);
  n->frames.push_back(vid_frame);
  if (!developer_mode) {
    n->grid_vars.push_back(vid_frame->vars[0]);
    n->grid_extents.push_back(vector_extent);
  }

  // Preserve the shared launch frame's accessor contract. vid is the NPU
  // thread accessor's x binding, while the actual launch remains cthread.
  // The other two axes are neutral placeholders: MaterializeKernelLaunch
  // drops them, and diagnoses accidental use as SIMT threads on this target.
  ForFrame thread_frame = MakeLaunchThreadFrame();
  n->frames.push_back(thread_frame);
  n->thread_vars = {vid_frame->vars[0], thread_frame->vars[1],
                    thread_frame->vars[2]};
  n->thread_extents = Array<PrimExpr>{vector_extent, Integer(1), Integer(1)};

  if (developer_mode) {
    script::ir_builder::IRBuilder builder =
        script::ir_builder::IRBuilder::Current();
    Optional<PrimFuncFrame> prim_func = builder->FindFrame<PrimFuncFrame>();
    ICHECK(prim_func.defined()) << "A2/A3 KernelLaunch must be inside PrimFunc";
    prim_func.value()->attrs.Set(
        "npu_cv_ratio",
        tirx::StringImm(vector_count->value == 1 ? "cv_1_1" : "cv_1_2"));
  }

  SBlockFrame main_block =
      script::ir_builder::tirx::Block(::tvm::tl::DeviceMainBlockName);
  main_block->reads = Array<tirx::BufferRegion>();
  main_block->writes = Array<tirx::BufferRegion>();
  main_block->annotations = attrs;
  n->frames.push_back(main_block);
  return KernelLaunchFrame(n);
}

// Complete TA PersistentFor from 0deb8b93, with scoped TIRX coordinate Binds.
ForFrame Persistent(const Array<PrimExpr> &domain, const PrimExpr &wave_size,
                    const PrimExpr &index, PrimExpr group_size) {
  using namespace tvm::tirx;
  ICHECK(!domain.empty());
  arith::Analyzer validation_analyzer;
  for (const auto &extent : domain) {
    ICHECK(!validation_analyzer.CanProveLess(extent, 1))
        << "T.Persistent domain extents must be positive, but got " << extent;
  }
  ICHECK(!validation_analyzer.CanProveLess(wave_size, 1))
      << "T.Persistent wave_size must be positive, but got " << wave_size;
  ICHECK(!validation_analyzer.CanProveLess(index, 0) &&
         !validation_analyzer.CanProveGreaterEqual(index - wave_size, 0))
      << "T.Persistent index must satisfy 0 <= index < wave_size, but got "
      << "index=" << index << " and wave_size=" << wave_size;
  ICHECK(!validation_analyzer.CanProveLess(group_size, 1))
      << "T.Persistent group_size must be positive, but got " << group_size;
  PrimExpr last_extent = domain[domain.size() - 1];
  PrimExpr effective_group_size = min(group_size, last_extent);
  PrimExpr group_remainder =
      validation_analyzer.Simplify(truncmod(last_extent, effective_group_size));
  if (const auto *remainder = group_remainder.as<IntImmNode>()) {
    ICHECK_EQ(remainder->value, 0)
        << "T.Persistent last domain extent must be divisible by "
        << "min(group_size, domain[-1]), but got domain[-1]=" << last_extent
        << " and group_size=" << group_size;
  }
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  n->vars.reserve(domain.size());
  n->doms.reserve(domain.size());
  PrimExpr domain_size = domain[0];
  for (int i = 1; i < domain.size(); i++) {
    domain_size *= domain[i];
  }

  auto waves = ceildiv(domain_size, wave_size);
  auto loop_var = Var("w", waves.dtype());
  group_size = min(group_size, domain[domain.size() - 1]);
  Array<Var> coord_vars;

  for (int i = 0; i < domain.size(); ++i) {
    DataType dtype = domain[i].dtype();
    Var coord("v" + std::to_string(i), dtype);
    coord_vars.push_back(coord);
    n->vars.push_back(coord);
    n->doms.push_back(Range(make_const(dtype, 0), domain[i]));
  }

  Array<PrimExpr> grouped_domain;
  grouped_domain.push_back(truncdiv(domain[domain.size() - 1], group_size));
  for (int i = 0; i < domain.size() - 1; ++i) {
    grouped_domain.push_back(domain[i]);
  }
  grouped_domain.push_back(group_size);

  n->f_make_for_loop = [=](const Array<Var> &vars, const Array<Range> &doms,
                           const Array<Optional<PrimExpr>> &,
                           Stmt body) -> Stmt {
    ICHECK_EQ(vars.size(), doms.size());
    Map<String, Any> anno;
    Array<PrimExpr> idxs(grouped_domain.size(), PrimExpr());
    PrimExpr global_index = loop_var * wave_size + index;
    PrimExpr rem = global_index;

    for (int i = grouped_domain.size() - 1; i >= 1; --i) {
      idxs.Set(i, truncmod(rem, grouped_domain[i]));
      rem = truncdiv(rem, grouped_domain[i]);
    }
    idxs.Set(0, rem);

    auto out_if = tvm::tirx::IfThenElse(
        domain_size <= global_index,
        tvm::tirx::Evaluate(
            tvm::tirx::Call(DataType::Handle(), tvm::tl::loop_break(), {})),
        std::nullopt);

    // Legacy Let values referred forward to w and were inlined by old TIR.
    // TIRX Binds must instead be dominated by the wave For. Keep the original
    // coordinate Var identities and wrapping order inside the guarded tile.
    Array<Stmt> coordinate_body;
    coordinate_body.push_back(
        tvm::tirx::Bind(vars.back(), idxs[0] * group_size + idxs[vars.size()]));
    for (int i = static_cast<int>(vars.size()) - 2; i >= 0; --i) {
      coordinate_body.push_back(tvm::tirx::Bind(vars[i], idxs[i + 1]));
    }
    coordinate_body.push_back(body);
    Stmt tile_body = SeqStmt::Flatten(coordinate_body);

    arith::Analyzer analyzer;
    Stmt new_body = tile_body;
    if (analyzer.CanProveGreaterEqual(waves, 2)) {
      new_body = SeqStmt({out_if, tile_body});
    } else if (!analyzer.CanProveEqual(domain_size, wave_size)) {
      // A unit loop can be removed during lowering, so a loop_break guard
      // would become an invalid top-level C++ break. Predicate the tile body
      // for partial first waves and symbolic wave counts instead. Under
      // Persistent's 0 <= index < wave_size contract, no guard is needed when
      // the domain is provably exactly one full wave.
      new_body = tvm::tirx::IfThenElse(global_index < domain_size, tile_body,
                                       std::nullopt);
    }
    Stmt outer =
        For(loop_var, 0, waves, ForKind::kSerial, new_body, std::nullopt, anno);
    return outer;
  };

  return ForFrame(n);
}

class ScopeFrameNode : public TIRFrameNode {
public:
  Array<TIRFrame> frames;

  static void RegisterReflection() {
    reflection::ObjectDef<ScopeFrameNode>().def_ro("frames",
                                                   &ScopeFrameNode::frames);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.A2A3ScopeFrame", ScopeFrameNode,
                                    TIRFrameNode);

  TVM_DLL void EnterWithScope() final {
    for (const TIRFrame &frame : frames) {
      frame->EnterWithScope();
    }
  }

  TVM_DLL void ExitWithScope() final {
    for (auto frame = frames.rbegin(); frame != frames.rend(); ++frame) {
      (*frame)->ExitWithScope();
    }
  }
};

class ScopeFrame : public TIRFrame {
public:
  explicit ScopeFrame(ObjectPtr<ScopeFrameNode> data) : TIRFrame(UnsafeInit{}) {
    data_ = std::move(data);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ScopeFrame, TIRFrame,
                                                ScopeFrameNode);
};

ScopeFrame Scope(const String &scope_name) {
  ObjectPtr<ScopeFrameNode> n = tvm::ffi::make_object<ScopeFrameNode>();
  // TA treats V as the vector resource and all other names as Cube.
  n->frames.push_back(
      Attr(Integer(0), "resource_scope", Integer(scope_name == "V" ? 1 : 0)));
  return ScopeFrame(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ScopeFrameNode::RegisterReflection();
  reflection::GlobalDef()
      .def("tl.A2A3KernelLaunch", KernelLaunch)
      .def("tl.A2A3Scope", Scope)
      .def("tl.A2A3Persistent", Persistent);
}

} // namespace a2a3
} // namespace tl
} // namespace tvm
