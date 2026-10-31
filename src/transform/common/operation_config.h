// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*
 * \file operation_config.h
 * \brief Operation configuration
 */
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>

#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../op/ascend.h"

namespace tvm {
namespace tl {

struct OperationConfig {
  std::vector<std::pair<size_t, std::string>> buffer_accesses;
  std::string default_pipeline;
  // Copy ABI: source, destination, then any required scratch buffers.
  // Pipe helpers carry the same local endpoint in both leading positions.
  std::vector<std::string> copy_scopes{};
  // Execution side of the lowered call: 0 = C, 1 = V; independent of pipe.
  int copy_resource_scope{-1};

  std::string toString() const {
    std::ostringstream oss;
    oss << "OperationConfig{";
    oss << "buffer_accesses: [";
    bool first_access = true;
    for (const auto &access : buffer_accesses) {
      if (!first_access)
        oss << ", ";
      oss << "(" << access.first << ", '" << access.second << "')";
      first_access = false;
    }
    oss << "], ";
    oss << "default_pipeline: '" << default_pipeline << "'";
    oss << "}";
    return oss.str();
  }
};

inline const std::unordered_map<std::string, OperationConfig> &
GetOperationConfig() {
  static std::unordered_map<std::string, OperationConfig> operation_config_ = {
      {"copy_gm_to_l1",
       {{{0, "read"}, {1, "write"}}, "PIPE_MTE2", {"global", "shared.l1"}, 0}},
      {"copy_gm_to_l0a",
       {{{0, "read"}, {1, "write"}},
        "PIPE_MTE2",
        {"global", "wmma.matrix_a"},
        0}},
      {"copy_gm_to_l0b",
       {{{0, "read"}, {1, "write"}},
        "PIPE_MTE2",
        {"global", "wmma.matrix_b"},
        0}},
      {"copy_gm_to_ub",
       {{{0, "read"}, {1, "write"}}, "PIPE_MTE2", {"global", "shared.ub"}, 1}},
      {"copy_l1_to_l0a",
       {{{0, "read"}, {1, "write"}},
        "PIPE_MTE1",
        {"shared.l1", "wmma.matrix_a"},
        0}},
      {"copy_l1_to_l0b",
       {{{0, "read"}, {1, "write"}},
        "PIPE_MTE1",
        {"shared.l1", "wmma.matrix_b"},
        0}},
      {"copy_ub_to_gm",
       {{{0, "read"}, {1, "write"}}, "PIPE_MTE3", {"shared.ub", "global"}, 1}},
      {"atomic_add_ub_to_gm",
       {{{0, "read"}, {1, "write"}}, "PIPE_MTE3", {"shared.ub", "global"}, 1}},
      {"atomic_add_l0c_to_gm",
       {{{0, "read"}, {1, "write"}},
        "PIPE_FIX",
        {"wmma.accumulator", "global"},
        0}},
      {"copy_ub_to_l1",
       {{{0, "read"}, {1, "write"}},
        "PIPE_MTE3",
        {"shared.ub", "shared.l1"},
        1}},
      {"copy_l0c_to_gm",
       {{{0, "read"}, {1, "write"}},
        "PIPE_FIX",
        {"wmma.accumulator", "global"},
        0}},
      {"copy_l0c_to_l1",
       {{{0, "read"}, {1, "write"}},
        "PIPE_FIX",
        {"wmma.accumulator", "shared.l1"},
        0}},
      {"copy_ub_to_ub",
       {{{0, "read"}, {1, "write"}}, "PIPE_V", {"shared.ub", "shared.ub"}, 1}},
      {"copy_ub_to_ub_Nz",
       {{{0, "read"}, {1, "write"}}, "PIPE_V", {"shared.ub", "shared.ub"}, 1}},
      {"copy_ub_to_pipe",
       {{{0, "read"}, {1, "read"}, {3, "write"}},
        "PIPE_MTE3",
        {"shared.ub", "shared.ub"},
        1}},
      {"copy_pipe_to_l1",
       {{{0, "write"}, {1, "write"}},
        "PIPE_MTE3",
        {"shared.l1", "shared.l1"},
        0}},
      {"copy_l0c_to_pipe",
       {{{0, "read"}, {1, "read"}},
        "PIPE_FIX",
        {"wmma.accumulator", "wmma.accumulator"},
        0}},
      {"copy_pipe_to_ub",
       {{{0, "write"}, {1, "write"}},
        "PIPE_MTE2",
        {"shared.ub", "shared.ub"},
        1}},
      {"copy_pipe_to_ub_V",
       {{{0, "write"}, {1, "write"}}, "PIPE_V", {"shared.ub", "shared.ub"}, 1}},
      {"mma", {{{0, "read"}, {1, "read"}, {2, "write"}}, "PIPE_M"}},
      {"gemm_v0", {{{0, "read"}, {1, "read"}, {2, "write"}}, "PIPE_M"}},
      {"gemm_v1", {{{0, "read"}, {1, "read"}, {2, "write"}}, "PIPE_M"}},
      {"AscendC::Add", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Adds", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Mul", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Sub", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Subs", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Div", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Divs", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Reduce", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"AscendC::Scalar", {{{0, "write"}, {1, "read"}}, "PIPE_S"}},
      {"AscendC::Exp", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"AscendC::Ln", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"AscendC::Sqrt", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"AscendC::Rsqrt", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"AscendC::Relu", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"AscendC::Axpy", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"AscendC::Select", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"AscendC::Abs", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"Gatherb", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::CompareScalar",
       {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Duplicate", {{{0, "write"}}, "PIPE_V"}},
      {"AscendC::Muls", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::And", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Or", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Not", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"reduce_max", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"reduce_min", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"AscendC::ClampMax", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"AscendC::ClampMin", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"AscendC::Clamp", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"AscendC::Round", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"reduce_sum", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"AscendC::Max", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Min", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Sin", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Cos", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Cast", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"AscendC::Sigmoid",
       {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Silu", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"tl.ascend_silu", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"AscendC::MulAddDst",
       {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_mul_add_dst",
       {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::ShiftLeft",
       {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::ShiftRight",
       {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"AscendC::Sort", {{{1, "write"}, {2, "read"}, {3, "write"}}, "PIPE_V"}},
      {"AscendC::ArithProgression", {{{0, "write"}}, "PIPE_V"}},
      {"GatherMask", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"AscendC::BilinearInterpolation",
       {{{0, "write"},
         {1, "read"},
         {2, "read"},
         {3, "read"},
         {4, "read"},
         {5, "read"},
         {6, "read"},
         {7, "read"},
         {8, "read"},
         {9, "read"},
         {10, "read"}},
        "PIPE_V"}},
      {"AscendC::WholeReduceMax",
       {{{0, "write"},
         {1, "read"},
         {2, "read"},
         {3, "read"},
         {4, "read"},
         {5, "read"},
         {6, "read"}},
        "PIPE_V"}},
      {"AscendC::WholeReduceMin",
       {{{0, "write"},
         {1, "read"},
         {2, "read"},
         {3, "read"},
         {4, "read"},
         {5, "read"},
         {6, "read"}},
        "PIPE_V"}},
      {"AscendC::WholeReduceSum",
       {{{0, "write"},
         {1, "read"},
         {2, "read"},
         {3, "read"},
         {4, "read"},
         {5, "read"},
         {6, "read"}},
        "PIPE_V"}},

      {"tl.ascend_mma", {{{1, "read"}, {2, "read"}, {3, "write"}}, "PIPE_M"}},
      {"tl.ascend_gemm_v0",
       {{{1, "read"}, {2, "read"}, {3, "write"}}, "PIPE_M"}},
      {"tl.ascend_gemm_v1",
       {{{1, "read"}, {2, "read"}, {3, "write"}}, "PIPE_M"}},
      {"tl.ascend_add", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_adds", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_mul", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_muls", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_sub", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_subs", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_div", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_divs", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_max", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_maxs", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_min", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_mins", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_bitwise_and",
       {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_bitwise_or",
       {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_compare",
       {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_compare_scalar",
       {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_exp", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"tl.ascend_ln", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"tl.ascend_abs", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"tl.ascend_reciprocal", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"tl.ascend_sqrt", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"tl.ascend_rsqrt", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"tl.ascend_relu", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"tl.ascend_bitwise_not", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"tl.ascend_select",
       {{{0, "write"}, {1, "read"}, {2, "read"}, {4, "read"}}, "PIPE_V"}},
      {"tl.ascend_leaky_relu", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"tl.ascend_axpy", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"tl.ascend_bitwise_lshift",
       {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_bitwise_rshift",
       {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_sort32",
       {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_createvecindex", {{{1, "write"}}, "PIPE_V"}},
      {"tl.ascend_arith_progression", {{{1, "write"}}, "PIPE_V"}},
      {"tl.ascend_sin", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_cos", {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_transpose", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"tl.ascend_gather",
       {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_reduce",
       {{{1, "write"}, {2, "read"}, {3, "read"}}, "PIPE_V"}},
      {"tl.ascend_block_reduce_max", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"tl.ascend_block_reduce_min", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"tl.ascend_block_reduce_sum", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"tl.ascend_shmem_ub_get_nbi",
       {{{1, "write"}, {2, "read"}}, "PIPE_MTE2"}},
      {"tl.ascend_shmem_get_nbi", {{{1, "write"}, {2, "read"}}, "PIPE_MTE3"}},
      {"tl.ascend_shmem_put_nbi", {{{1, "write"}, {2, "read"}}, "PIPE_MTE3"}},
      {"tl.ascend_shmem_ub_put_nbi",
       {{{1, "read"}, {2, "write"}}, "PIPE_MTE3"}},

      {"tl.ascend_scalar", {{{0, "write"}, {1, "read"}}, "PIPE_S"}},
      {"tl.ascend_gatherb",
       {{{1, "write"}, {2, "read"}, {3, "read"}}, "PIPE_V"}},
      {"tl.ascend_duplicate", {{{0, "write"}}, "PIPE_V"}},
      {"tl.ascend_cast", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},

      {"tl.ascend_pow",
       {{{0, "write"}, {1, "read"}, {2, "read"}, {3, "read"}}, "PIPE_V"}},
      {"tl.ascend_bitwise_xor",
       {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_broadcast", {{{1, "write"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_fill", {{{1, "write"}}, "PIPE_V"}},
      {"tl.arith_progression", {{{1, "write"}}, "PIPE_V"}},
      {"tl.ascend_sort",
       {{{1, "write"}, {2, "read"}, {3, "read"}, {4, "read"}}, "PIPE_V"}},
      {"tl.ascend_merge_sort",
       {{{2, "write"},
         {3, "write"},
         {4, "read"},
         {5, "read"},
         {6, "read"},
         {7, "read"}},
        "PIPE_V"}},
      {"tl.ascend_topk",
       {{{1, "write"}, {2, "read"}, {3, "read"}, {4, "read"}, {5, "read"}},
        "PIPE_V"}},
      {"tl.ascend_gather_mask", {{{1, "write"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_init_sort_buf", {{{1, "write"}}, "PIPE_V"}},

      {"tl.ascend_bilinear_interpolation",
       {{{0, "write"}, {1, "read"}, {2, "read"}, {3, "read"}, {10, "read"}},
        "PIPE_V"}},
      {"tl.ascend_wholereducemax", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"tl.ascend_wholereducemin", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"tl.ascend_wholereducesum", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"tl.ascend_sigmoid",
       {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_clamp_max", {{{1, "write"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_clamp_min", {{{1, "write"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_clamp", {{{1, "write"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_round", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},

      {"tl.ascend_sub_experiment",
       {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_abs_experiment", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"tl.ascend_mins_experiment", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"tl.ascend_reducesum_experiment",
       {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_reducesum_mask_experiment",
       {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_gather_mask_experiment",
       {{{1, "write"}, {2, "read"}, {3, "read"}}, "PIPE_V"}},
      {"tl.ascend_fill_experiment", {{{1, "write"}}, "PIPE_V"}},
      {"tl.ascend_sum_experiment", {{{1, "write"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_datacachecleanandinvalid_experiment",
       {{{1, "write"}}, "PIPE_S"}},
      {"tl.ascend_brcb_experiment", {{{1, "write"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_row_expand_mul_experiment",
       {{{1, "write"}, {2, "read"}, {3, "read"}}, "PIPE_V"}},
      {"tl.ascend_row_expand_sub_experiment",
       {{{1, "write"}, {2, "read"}, {3, "read"}}, "PIPE_V"}},
      {"tl.ascend_row_expand_div_experiment",
       {{{1, "write"}, {2, "read"}, {3, "read"}}, "PIPE_V"}},
      {"tl.ascend_exp_experiment", {{{1, "write"}, {2, "read"}}, "PIPE_V"}},

      // Internal tail-aware ops produced by AscendTailMaskPropagation.
      {"tl.ascend_tail_unary", {{{1, "write"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_tail_binary",
       {{{1, "write"}, {2, "read"}, {3, "read"}}, "PIPE_V"}},
      {"tl.ascend_tail_scalar", {{{1, "write"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_tail_reduce", {{{1, "write"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_tail_compare",
       {{{0, "write"}, {1, "read"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_tail_compare_scalar",
       {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
      {"tl.ascend_tail_select",
       {{{1, "write"}, {2, "read"}, {3, "read"}, {4, "read"}, {6, "read"}},
        "PIPE_V"}},
      {"tl.ascend_tail_broadcast", {{{1, "write"}, {2, "read"}}, "PIPE_V"}},
      {"tl.ascend_copy_cv_experiment",
       {{{0, "read"}, {1, "write"}},
        "PIPE_FIX",
        {"wmma.accumulator", "shared.ub"},
        0}},
      {"tl.ascend_copy_vc_experiment",
       {{{0, "read"}, {1, "write"}, {2, "write"}},
        "PIPE_V",
        {"shared.ub", "shared.l1", "shared.ub"},
        1}},
  };

  return operation_config_;
}

inline const std::unordered_map<std::string, std::string> &GetEventMapping() {
  static std::unordered_map<std::string, std::string> event_mapping_ = {
      {"PIPE_MTE2_PIPE_MTE1", "MTE2_MTE1"},
      {"PIPE_MTE1_PIPE_MTE2", "MTE1_MTE2"},
      {"PIPE_MTE1_PIPE_M", "MTE1_M"},
      {"PIPE_M_PIPE_MTE1", "M_MTE1"},
      {"PIPE_MTE2_PIPE_V", "MTE2_V"},
      {"PIPE_V_PIPE_MTE2", "V_MTE2"},
      {"PIPE_MTE3_PIPE_V", "MTE3_V"},
      {"PIPE_V_PIPE_MTE3", "V_MTE3"},
      {"PIPE_M_PIPE_V", "M_V"},
      {"PIPE_V_PIPE_M", "V_M"},
      {"PIPE_V_PIPE_V", "V_V"},
      {"PIPE_MTE3_PIPE_MTE1", "MTE3_MTE1"},
      {"PIPE_MTE1_PIPE_MTE3", "MTE1_MTE3"},
      {"PIPE_MTE1_PIPE_V", "MTE1_V"},
      {"PIPE_MTE2_PIPE_M", "MTE2_M"},
      {"PIPE_M_PIPE_MTE2", "M_MTE2"},
      {"PIPE_V_PIPE_MTE1", "V_MTE1"},
      {"PIPE_M_PIPE_FIX", "M_FIX"},
      {"PIPE_FIX_PIPE_M", "FIX_M"},
      {"PIPE_MTE3_PIPE_MTE2", "MTE3_MTE2"},
      {"PIPE_MTE2_PIPE_MTE3", "MTE2_MTE3"},
      {"PIPE_S_PIPE_V", "S_V"},
      {"PIPE_V_PIPE_S", "V_S"},
      {"PIPE_S_PIPE_MTE2", "S_MTE2"},
      {"PIPE_MTE2_PIPE_S", "MTE2_S"},
      {"PIPE_S_PIPE_MTE3", "S_MTE3"},
      {"PIPE_MTE3_PIPE_S", "MTE3_S"},
      {"PIPE_MTE2_PIPE_FIX", "MTE2_FIX"},
      {"PIPE_FIX_PIPE_MTE2", "FIX_MTE2"},
      {"PIPE_FIX_PIPE_S", "FIX_S"},
      {"PIPE_M_PIPE_S", "M_S"},
      {"PIPE_FIX_PIPE_MTE3", "FIX_MTE3"}};

  return event_mapping_;
}

/*! \brief A set of memory scopes that require their layout to be flattened.
 */
const std::unordered_set<std::string> kScopesToFlatten = {
    "shared",        "shared.ub",     "shared.l1",
    "wmma.matrix_a", "wmma.matrix_b", "wmma.accumulator"};

/*! \brief The memory scope that requires alignment for its inner
 * dimension. */
const std::unordered_map<std::string, int> kScopeForAlignment = {
    {"shared.ub", 32 * 8}};

struct WorkspaceOpConfig {
  int64_t tmp_arg_index;
  bool ascendc_supported;
  bool pto_supported;
};

inline const std::unordered_map<const tvm::OpNode *, WorkspaceOpConfig> &
GetWorkspaceOpConfigs() {
  // Public calls use one optional workspace slot regardless of whether a
  // target consumes it. InjectTmpBuffer owns target-specific insertion and
  // removal; downstream analyses must inspect the resulting call layout.
  static const std::unordered_map<const tvm::OpNode *, WorkspaceOpConfig> ops =
      {
          {tl::ascend_clamp().get(), {3, true, true}},
          {tl::ascend_clamp_max().get(), {3, true, true}},
          {tl::ascend_clamp_min().get(), {3, true, true}},
          {tl::ascend_reduce().get(), {3, true, true}},
          {tl::ascend_sort().get(), {3, true, true}},
          {tl::ascend_topk().get(), {3, true, true}},
          {tl::ascend_sigmoid().get(), {2, true, true}},
          {tl::ascend_bilinear_interpolation().get(), {10, true, false}},
          {tl::ascend_sin().get(), {2, true, false}},
          {tl::ascend_cos().get(), {2, true, false}},
          {tl::ascend_pow().get(), {3, true, true}},
          {tl::ascend_bitwise_xor().get(), {3, true, true}},
          {tl::ascend_round().get(), {2, true, true}},
          {tl::ascend_broadcast().get(), {3, true, true}},
          {tl::ascend_reducesum_experiment().get(), {2, true, false}},
          {tl::ascend_reducesum_mask_experiment().get(), {2, true, false}},
          {tl::ascend_merge_sort().get(), {3, true, true}},
          {tl::ascend_select().get(), {3, true, true}},
          {tl::ascend_gather_mask().get(), {4, true, true}},
          {tl::ascend_gather().get(), {5, true, true}},
      };
  return ops;
}

inline OperationConfig ResolveOperationConfig(const tvm::tir::CallNode *call,
                                              const std::string &name,
                                              size_t argument_offset) {
  const bool scalar_binary =
      name == "tl.ascend_adds" || name == "tl.ascend_muls" ||
      name == "tl.ascend_subs" || name == "tl.ascend_divs" ||
      name == "tl.ascend_maxs" || name == "tl.ascend_mins";
  const bool scalar_compare = name == "tl.ascend_compare_scalar";
  const bool external_scalar =
      name == "AscendC::Adds" || name == "AscendC::Muls" ||
      name == "AscendC::Subs" || name == "AscendC::Divs" ||
      name == "AscendC::Maxs" || name == "AscendC::Mins" ||
      name == "AscendC::CompareScalar";
  ICHECK(!external_scalar)
      << "Scalar call_extern operand ABI is not audited for " << name;
  const auto config_it = GetOperationConfig().find(name);
  ICHECK(config_it != GetOperationConfig().end())
      << "Missing operation configuration for " << name;

  OperationConfig config = config_it->second;
  if (scalar_binary || scalar_compare) {
    ICHECK_EQ(argument_offset, 0U)
        << "Scalar call_extern operand ABI is not audited for " << name;
    // The public scalar intrinsics have two ABIs: a scalar value, or a
    // buffer pointer followed by its scalar index. Do not discard that read
    // when fixing the immediate-value form's overly broad catalog entry.
    const size_t scalar_arity = scalar_compare ? 5U : 4U;
    ICHECK_GE(call->args.size(), argument_offset);
    const size_t arity = call->args.size() - argument_offset;
    ICHECK(arity == scalar_arity || arity == scalar_arity + 1)
        << "Malformed scalar operand arity in " << name;
    const bool buffer_scalar = arity == scalar_arity + 1;
    auto require_pointer = [&](size_t index) {
      const auto *pointer =
          call->args[argument_offset + index].as<tvm::tir::CallNode>();
      ICHECK(pointer != nullptr &&
             pointer->op.same_as(tvm::tir::builtin::tvm_access_ptr()) &&
             pointer->args.size() == 5U)
          << "Malformed buffer operand " << index << " in " << name;
    };
    auto require_integer = [&](size_t index) {
      const DataType dtype = call->args[argument_offset + index].dtype();
      ICHECK(dtype.is_scalar() && !dtype.is_bool() &&
             (dtype.is_int() || dtype.is_uint()))
          << "Malformed scalar index/count in " << name;
    };
    require_pointer(0);
    require_pointer(1);
    config.buffer_accesses = {{0, "write"}, {1, "read"}};
    if (buffer_scalar) {
      require_pointer(2);
      require_integer(3);
      config.buffer_accesses.emplace_back(2, "read");
    } else {
      const DataType dtype = call->args[argument_offset + 2].dtype();
      ICHECK(dtype.is_scalar() && (dtype.is_int() || dtype.is_uint() ||
                                   dtype.is_float() || dtype.is_bfloat16()))
          << "Malformed scalar value in " << name;
    }
    require_integer(arity - 1);
    if (scalar_compare) {
      const auto *mode =
          call->args[argument_offset + arity - 2].as<tvm::tir::StringImmNode>();
      ICHECK(mode != nullptr && (mode->value == "EQ" || mode->value == "NE" ||
                                 mode->value == "GT" || mode->value == "GE" ||
                                 mode->value == "LT" || mode->value == "LE"))
          << "Malformed scalar comparison mode in " << name;
    }
    return config;
  }
  const auto *op_node = call->op.as<tvm::OpNode>();
  if (argument_offset != 0 || op_node == nullptr ||
      !GetWorkspaceOpConfigs().count(op_node)) {
    return config;
  }

  // Optional workspace operands shift later arguments. Derive the declared
  // buffer accesses from the transformed call instead of maintaining a second
  // positional ABI for these public intrinsics.
  config.buffer_accesses.clear();
  for (size_t i = 0; i < call->args.size(); ++i) {
    const auto *arg = call->args[i].as<tvm::tir::CallNode>();
    if (!arg || !arg->op.same_as(tvm::tir::builtin::tvm_access_ptr())) {
      continue;
    }
    ICHECK_GE(arg->args.size(), 5U)
        << "Malformed tvm_access_ptr in " << op_node->name;
    const auto *mask = arg->args[4].as<tvm::IntImmNode>();
    ICHECK(mask && mask->value > 0 && (mask->value & ~3) == 0)
        << "Invalid access mask in " << op_node->name;
    if (mask->value & 1) {
      config.buffer_accesses.emplace_back(i, "read");
    }
    if (mask->value & 2) {
      config.buffer_accesses.emplace_back(i, "write");
    }
  }
  return config;
}

inline OperationConfig ResolveOperationConfig(const tvm::tir::CallNode *call) {
  const auto *op_node = call->op.as<tvm::OpNode>();
  ICHECK(op_node);
  return ResolveOperationConfig(call, op_node->name, 0);
}

// Shape-independent clients above only inspect operands and memory effects.
// Synchronization clients must use this context-aware resolver because a
// semantic intrinsic's default pipeline need not match its selected helper.
inline OperationConfig ResolveOperationConfigWithPipeline(
    const tvm::tir::CallNode *call, const std::string &name,
    size_t argument_offset,
    const Map<tir::Var, Array<PrimExpr>> &logic_buffer_shapes,
    bool pto_transpose) {
  OperationConfig config = ResolveOperationConfig(call, name, argument_offset);
  if (name == "tl.ascend_transpose") {
    // The execution resource is AIV even when the helper falls back to scalar
    // GetValue/SetValue. PIPE_S alone does not identify an execution resource.
    config.copy_resource_scope = 1;
    if (pto_transpose) {
      // PTO lowers this intrinsic to TTRANS, not the AscendC helper below.
      return config;
    }
    ICHECK_EQ(argument_offset, 0U)
        << "Transpose call_extern operand ABI is not audited";
    ICHECK_EQ(call->args.size(), 2U) << "Malformed transpose operand arity";
    const auto *source = call->args[1].as<tir::CallNode>();
    ICHECK(source != nullptr &&
           source->op.same_as(tir::builtin::tvm_access_ptr()) &&
           source->args.size() == 5U)
        << "Malformed transpose source access_ptr";
    const auto *data = source->args[1].as<tir::VarNode>();
    ICHECK(data != nullptr) << "Malformed transpose source storage";
    const auto shape = logic_buffer_shapes.Get(GetRef<tir::Var>(data));
    ICHECK(shape.defined() && shape.value().size() >= 2U)
        << "Transpose synchronization requires its logical source shape";
    const auto *rows = shape.value()[0].as<IntImmNode>();
    const auto *columns = shape.value()[1].as<IntImmNode>();
    const DataType dtype = source->args[0].dtype();
    const bool integer = (dtype.is_int() || dtype.is_uint()) &&
                         (dtype.bits() == 8 || dtype.bits() == 16 ||
                          dtype.bits() == 32 || dtype.bits() == 64);
    // The emitter spells other floating types as float, losing their width.
    // Do not infer the helper branch from a dtype it cannot preserve.
    ICHECK(dtype.is_scalar() &&
           (integer || dtype == DataType::Float(16) ||
            dtype == DataType::Float(32) || dtype == DataType::BFloat(16)))
        << "Transpose synchronization requires an audited element type";
    ICHECK(rows != nullptr && columns != nullptr && rows->value > 0 &&
           columns->value > 0)
        << "Transpose synchronization requires a static positive helper shape";
    const int bytes = dtype.bytes();
    // Keep this predicate identical to tl::ascend::transpose in common.h.
    // B32 dimensions such as 8 and 24 are DMA-aligned but use the scalar path.
    const bool vector_helper = (bytes == 2 || bytes == 4) &&
                               !dtype.is_bfloat16() && rows->value % 16 == 0 &&
                               columns->value % 16 == 0;
    config.default_pipeline = vector_helper ? "PIPE_V" : "PIPE_S";
    return config;
  }
  return config;
}

inline OperationConfig ResolveOperationConfigWithPipeline(
    const tvm::tir::CallNode *call,
    const Map<tir::Var, Array<PrimExpr>> &logic_buffer_shapes,
    bool pto_transpose) {
  const auto *op_node = call->op.as<tvm::OpNode>();
  ICHECK(op_node);
  return ResolveOperationConfigWithPipeline(call, op_node->name, 0,
                                            logic_buffer_shapes, pto_transpose);
}

} // namespace tl
} // namespace tvm
