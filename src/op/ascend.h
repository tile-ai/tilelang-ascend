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
NPUIR_BINARY_OP_CLASS(Or)
NPUIR_BINARY_OP_CLASS(And)
NPUIR_BINARY_OP_CLASS(Xor)
NPUIR_BINARY_OP_CLASS(Pow)
NPUIR_BINARY_OP_CLASS(Shl)

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
NPUIR_UNARY_OP_CLASS(Sqrt)
NPUIR_UNARY_OP_CLASS(Rsqrt)
NPUIR_UNARY_OP_CLASS(Abs)
NPUIR_UNARY_OP_CLASS(Rec)
NPUIR_UNARY_OP_CLASS(Not)

class NpuirDot : public Operator {
public:
  NpuirDot(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  Buffer src0, src1, dst;

  PrimExpr initC;

  bool a_transpose, b_transpose;

  Array<Range> src0_range, src1_range, dst_range;
};

/// HIVM data copy operation with on-the-fly ND to NZ layout transformation.
class NpuirNd2nz : public Operator {
public:
  NpuirNd2nz(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  Buffer src, dst;

  // stored continuously in the destination buffer.
  bool dst_continuous;

  Array<Range> src_range, dst_range;
};

/// HIVM data copy operation from L1 to Global Memory with NZ2ND conversion.
class NpuirNz2nd : public Operator {
public:
  NpuirNz2nd(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  Buffer src, dst;

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
class NpuirSetFlag: public Operator {
public:
  NpuirSetFlag(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();
  static constexpr std::string_view op = "set";

  std::string pipe1;
  std::string pipe2;
  PrimExpr event_id;
};

/// HIVM wait flag sync.
class NpuirWaitFlag: public Operator {
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
  std::vector<int64_t> reduce_dims;

  Array<Range> src_range, dst_range;
  bool clear;
};

class NpuirCumsum : public Operator {
public:
  NpuirCumsum(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  Buffer src, dst;
  bool reverse;
  std::vector<int64_t> cum_dims;

  Array<Range> src_range, dst_range;
};

class NpuirAtomicAdd : public Operator {
public:
  NpuirAtomicAdd(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  Buffer src, dst;
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

class NpuirShr : public Operator {
public:
  NpuirShr(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();

  Buffer src0, src1, dst;
  Array<Range> src0_range, src1_range, dst_range;
  bool round;
};

/// HIVM device print operation (print var info)
/// Device-side print for debugging
class NpuirDevicePrintVar : public Operator {
public:
  NpuirDevicePrintVar(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  PrimExpr src;
  std::string prefix;
  bool hex;
};

/// HIVM device print operation (print buffer value)
/// Device-side print for debugging
class NpuirDevicePrintBuf : public Operator {
public:
  NpuirDevicePrintBuf(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  Buffer src;
  std::string prefix;
  bool hex;

  Array<Range> src_range;
};

/// HIVM vector gather operation
/// Retrieve elements from a tensor/memref according to given indices
class NpuirGather : public Operator {
public:
  NpuirGather(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  Buffer src, dst, indices;

  Array<Range> src_range, dst_range, indices_range;
};

/// HIVM vector transpose operation
/// Permutes the dimensions of src according to the given permutation.
class NpuirTranspose : public Operator {
public:
  NpuirTranspose(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  Buffer src, dst;
  std::vector<int64_t> permutation;

  Array<Range> src_range, dst_range;
};

/// HIVM vector interleave operation
/// Interleaves the values of N tensors along their last dimension.
class NpuirInterleave : public Operator {
public:
  NpuirInterleave(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  std::vector<Buffer> srcs;
  Buffer dst;
  int64_t channel_nums;

  std::vector<Array<Range>> srcs_range;
  Array<Range> dst_range;
};

/// HIVM vector deinterleave operation
/// Deinterleave one tensor along the last dimension.
class NpuirDeinterleave : public Operator {
public:
  NpuirDeinterleave(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  Buffer src;
  std::vector<Buffer> dsts;
  int64_t channel_nums;
  std::string index_mode;

  Array<Range> src_range;
  std::vector<Array<Range>> dsts_range;
};

/// HIVM vector arange operation
/// Fill a vector with range 0,1,2... based on strides and offset.
class NpuirArange : public Operator {
public:
  NpuirArange(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  Buffer dst;
  std::vector<int64_t> strides;
  int64_t offset;

  Array<Range> dst_range;
};

/// HIVM vector concat operation
/// Constructs a tensor out of a variadic list of input tensors, concatenated
/// along a static dimension number.
class NpuirConcat : public Operator {
public:
  NpuirConcat(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  std::vector<Buffer> srcs;
  Buffer dst;
  int64_t dim;

  std::vector<Array<Range>> srcs_range;
  Array<Range> dst_range;
};

/// HIVM vector pad operation
/// Pads the input operand.
class NpuirPad : public Operator {
public:
  NpuirPad(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  Buffer src;
  Buffer dst;
  PrimExpr pad_value;
  int pad_dim;
  std::vector<PrimExpr> low;
  std::vector<PrimExpr> high;
  std::vector<int64_t> s_low;
  std::vector<int64_t> s_high;

  Array<Range> src_range;
  Array<Range> dst_range;
};

/// HIVM vector flip operation
/// Flips a tensor along the last dimension.
class NpuirFlip : public Operator {
public:
  NpuirFlip(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  Buffer src;
  Buffer dst;

  Array<Range> src_range;
  Array<Range> dst_range;
};

/// HIVM bitcast operation
/// Converts a tensor/memref from one element type to another while preserving
/// the underlying bit representation.
class NpuirBitcast : public Operator {
public:
  NpuirBitcast(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();

  Buffer src;
  std::string dtype;

  Array<Range> src_range;
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_ELEM_H_


