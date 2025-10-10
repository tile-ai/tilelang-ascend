// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/op/ascend.h
 * \brief Define ascend-related operators.
 *
 */

#ifndef TVM_TL_OP_ELEM_H_
#define TVM_TL_OP_ELEM_H_

#include "op.h"
#include "tvm/ir/expr.h"

namespace tvm {
namespace tl {

using namespace tir;

class AscendCopy : public Operator {
public:
  AscendCopy(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) final;

  static const Op &Get();

  Array<PrimExpr> args_;

  Buffer src, dst;

  Array<Range> src_range, dst_range;
};

#define NPUIR_BINARY_OP_CLASS(OPNAME)                                          \
  class Npuir##OPNAME : public Operator {                                      \
  public:                                                                      \
    Npuir##OPNAME(Array<PrimExpr> args, BufferMap vmap);                       \
    static const Op &Get();                                                    \
                                                                               \
  private:                                                                     \
    Buffer src0, src1, dst;                                                    \
    Array<Range> src0_range, src1_range, dst_range;                            \
  };

NPUIR_BINARY_OP_CLASS(Add)
NPUIR_BINARY_OP_CLASS(Sub)
NPUIR_BINARY_OP_CLASS(Mul)
NPUIR_BINARY_OP_CLASS(Div)
NPUIR_BINARY_OP_CLASS(Max)
NPUIR_BINARY_OP_CLASS(Min)

#define NPUIR_UNARY_OP_CLASS(OPNAME)                                           \
  class Npuir##OPNAME : public Operator {                                      \
  public:                                                                      \
    Npuir##OPNAME(Array<PrimExpr> args, BufferMap vmap);                       \
    static const Op &Get();                                                    \
                                                                               \
    std::string op_name;                                                       \
    Buffer src, dst;                                                           \
    Array<Range> src_range, dst_range;                                         \
  };

NPUIR_UNARY_OP_CLASS(Exp)
NPUIR_UNARY_OP_CLASS(Ln)
NPUIR_UNARY_OP_CLASS(Relu)

class NpuirDot : public Operator {
public:
  NpuirDot(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  Buffer src0, src1, dst;

  PrimExpr initC;

  bool a_transpose, b_transpose;

  Array<Range> src0_range, src1_range, dst_range;
};

/// HIVM data copy operation with on-the-fly ND to NZ layout transformation
class NpuirNd2nz : public Operator {
public:
  NpuirNd2nz(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  Buffer src, dst;

  // stored continuously in the destination buffer.
  bool dst_continuous;

  Array<Range> src_range, dst_range;
};

/// HIVM data copy operation from L0C to L1 or Global Memory via fixpipe
/// pipeline.
class NpuirFixpipe : public Operator {
public:
  NpuirFixpipe(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  // src and dst are good to share the same dtype or not.
  // If not, src to dst may enable pre-quant: S322I8, F322F16, F322BF16
  Buffer src, dst;

  bool enable_nz2nd;
  bool channel_split;

  int pre_relu_mode; // 0: no; 1: relu; 2: leaky_relu; 3: prelu.

  Array<Range> src_range, dst_range;
};

enum class SyncBlockMode : uint32_t {
  INTER_BLOCK = 0,
  INTER_SUBBLOCK = 1,
  INTRA_BLOCK = 2,
};

/// HIVM intra pipeline sync.
class NpuirPipeBarrier : public Operator {
public:
  NpuirPipeBarrier(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  std::string pipe_type;
};

/// HIVM set flag sync.
class NpuirSetFlag : public Operator {
public:
  NpuirSetFlag(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();
  static constexpr std::string_view op = "set";

  std::string pipe1;
  std::string pipe2;
  PrimExpr event_id;
};

/// HIVM wait flag sync.
class NpuirWaitFlag : public Operator {
public:
  NpuirWaitFlag(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();
  static constexpr std::string_view op = "wait";

  std::string pipe1;
  std::string pipe2;
  PrimExpr event_id;
};
/// HIVM cross block sync.
class NpuirSyncBlock : public Operator {
public:
  NpuirSyncBlock(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  SyncBlockMode mode;
  std::string pipe_type;
  PrimExpr flag_id;
};

/// HIVM cross block sync.
class NpuirSyncBlockSet : public Operator {
public:
  NpuirSyncBlockSet(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  SyncBlockMode mode;
  std::string pipe_type;
  PrimExpr flag_id;
};

/// HIVM cross block sync.
class NpuirSyncBlockWait : public Operator {
public:
  NpuirSyncBlockWait(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  std::string pipe_type;
  PrimExpr flag_id;
};

/// HIVM vector broadcast operation
/// Broadcast a vector or a scalar according to the broadcast axes array.
class NpuirBrc : public Operator {
public:
  NpuirBrc(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  PrimExpr in, out;
  Buffer src, dst;

  Array<Range> src_range, dst_range;
};

/// HIVM vector type conversion operation
/// Performs element-wise operation on N operands and produces a single result.
/// It may perform either transpose or broadcast along the way (but not both).
/// Currently transpose is not supported.
class NpuirCast : public Operator {
public:
  NpuirCast(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  Buffer src, dst;
  std::string round_mode;

  Array<Range> src_range, dst_range;
};

/// HIVM vector reduction operation
/// Reduce one or more axes of the source vector according to the reduction axes
/// array, starting from an init value.
class NpuirReduce : public Operator {
public:
  NpuirReduce(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  Buffer src, dst;
  std::string reduce_mode;
  std::vector<int> reduce_dims;

  Array<Range> src_range, dst_range;
};

class NpuirSelect : public Operator {
public:
  NpuirSelect(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();

  Buffer cond, src0, src1, dst;
  Array<Range> cond_range, src0_range, src1_range, dst_range;
};

class NpuirCmp : public Operator {
public:
  NpuirCmp(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();

  Buffer src0, src1, dst;
  Array<Range> src0_range, src1_range, dst_range;
  std::string cmp_mod;
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_ELEM_H_