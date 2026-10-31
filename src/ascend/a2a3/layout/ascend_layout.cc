// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

// TA 0deb8b93: layout/layout.{h,cc} metadata and the allocation helper in
// transform/lower_tile_op.cc. Shared Layout owns all coordinate operations.
#include "ascend_layout.h"

#include <tvm/ffi/reflection/registry.h>

namespace tvm::tl::a2a3 {

using namespace ffi;

namespace {

class A2A3LayoutNode : public LayoutNode {
public:
  A2A3LayoutNode() = default;
  A2A3LayoutNode(const Layout &layout, AscendLayoutKind kind)
      : LayoutNode(layout->InputShape(), layout->GetForwardIndex()),
        kind_(static_cast<int>(kind)) {}

  AscendLayoutKind Kind() const { return static_cast<AscendLayoutKind>(kind_); }

  Layout Inverse() const final {
    // TA's inverse constructor omitted the tag, restoring its default 0.
    return WrapAscendLayout(LayoutNode::Inverse(), AscendLayoutKind::kRowMajor);
  }

  Layout Expand(const Array<PrimExpr> &) const final {
    LOG(FATAL) << "A2/A3 Layout.Expand is not migrated: TA had no Expand API. "
                  "Construct the complete tagged layout explicitly.";
    return {};
  }

  Layout Repeat(int, int) const final {
    LOG(FATAL) << "A2/A3 Layout.Repeat is not migrated: TA had no Repeat API. "
                  "Construct the complete tagged layout explicitly.";
    return {};
  }

  Layout Reshape(const Array<PrimExpr> &, arith::Analyzer *, const PrimExpr,
                 const PrimExpr) const final {
    LOG(FATAL)
        << "A2/A3 Layout.Reshape is not migrated: TA had no Reshape API. "
           "Construct the complete tagged layout explicitly.";
    return {};
  }

  static void RegisterReflection() {
    reflection::ObjectDef<A2A3LayoutNode>().def_ro("kind",
                                                   &A2A3LayoutNode::kind_);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.A2A3Layout", A2A3LayoutNode,
                                    LayoutNode);
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;

private:
  int kind_{0};
};

AscendLayoutKind GetKind(const Layout &layout) {
  const auto *node = layout.as<A2A3LayoutNode>();
  ICHECK(node != nullptr)
      << "A2/A3 layout metadata is missing. Construct a2a3.layout.Layout or "
         "explicitly bridge the layout with WrapAscendLayout; do not infer a "
         "TA template tag from its coordinate map.";
  return node->Kind();
}

// Keep the TA static-only override. Symbolic shapes retain OutputShape rather
// than gaining an unrelated allocation policy during the migration.
Optional<IntImm> AscendFractalExtent(const Layout &layout, DataType dtype) {
  AscendLayoutKind kind = GetKind(layout);
  bool is_zn = kind == AscendLayoutKind::kzN;
  bool is_nz = kind == AscendLayoutKind::knZ;
  bool is_zz = kind == AscendLayoutKind::kzZ;
  if ((!is_zn && !is_nz && !is_zz) || layout->InputDim() < 2 ||
      dtype.bits() == 0 || dtype.lanes() != 1) {
    return std::nullopt;
  }
  const auto *rows =
      layout->InputShape()[layout->InputDim() - 2].as<IntImmNode>();
  const auto *cols =
      layout->InputShape()[layout->InputDim() - 1].as<IntImmNode>();
  if (rows == nullptr || cols == nullptr) {
    return std::nullopt;
  }
  const int64_t ele_per_c0 = 32 * 8 / dtype.bits();
  constexpr int64_t kC0NumPerFractal = 16;
  auto round_up = [](int64_t v, int64_t m) { return (v + m - 1) / m * m; };
  auto ceil_div = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
  int64_t extent;
  if (is_zn) {
    extent = ceil_div(cols->value, ele_per_c0) *
             round_up(rows->value, kC0NumPerFractal) * ele_per_c0;
  } else if (is_nz) {
    extent = ceil_div(rows->value, ele_per_c0) *
             round_up(cols->value, kC0NumPerFractal) * ele_per_c0;
  } else {
    extent = ceil_div(rows->value, kC0NumPerFractal) *
             round_up(cols->value, ele_per_c0) * kC0NumPerFractal;
  }
  return IntImm(DataType::Int(64), extent);
}

} // namespace

Layout WrapAscendLayout(const Layout &layout, AscendLayoutKind kind) {
  ICHECK(layout.defined());
  ICHECK(layout->GetTypeKey() == std::string("tl.Layout") ||
         layout.as<A2A3LayoutNode>() != nullptr)
      << "WrapAscendLayout cannot discard additional layout state from "
      << layout->GetTypeKey()
      << ". Only a plain Layout or an existing A2A3Layout can be bridged.";
  ICHECK_GE(static_cast<int>(kind), 0);
  ICHECK_LE(static_cast<int>(kind), 4);
  return Layout(make_object<A2A3LayoutNode>(layout, kind));
}

std::string AscendLayoutStr(const Layout &layout) {
  switch (GetKind(layout)) {
  case AscendLayoutKind::kRowMajor:
    return "layout::RowMajor";
  case AscendLayoutKind::kColMajor:
    return "layout::ColumnMajor";
  case AscendLayoutKind::kzN:
    return "layout::zN";
  case AscendLayoutKind::kzZ:
    return "layout::zZ";
  case AscendLayoutKind::knZ:
    return "layout::nZ";
  }
  LOG(FATAL) << "Unknown A2/A3 layout kind";
  return {};
}

Array<PrimExpr> PhysicalAllocationShape(const Layout &layout, DataType dtype) {
  Array<PrimExpr> shape = layout->OutputShape();
  if (auto extent = AscendFractalExtent(layout, dtype)) {
    if (shape.empty()) {
      shape = {extent.value()};
    } else {
      shape.Set(shape.size() - 1, extent.value());
    }
  }
  return shape;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  A2A3LayoutNode::RegisterReflection();
  reflection::GlobalDef()
      .def("tl.ascend.a2a3.WrapLayout",
           [](const Layout &layout, int kind) {
             return WrapAscendLayout(layout,
                                     static_cast<AscendLayoutKind>(kind));
           })
      .def("tl.ascend.a2a3.LayoutStr", AscendLayoutStr)
      .def("tl.ascend.a2a3.PhysicalAllocationShape", PhysicalAllocationShape);
}

} // namespace tvm::tl::a2a3
