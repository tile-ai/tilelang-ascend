// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*
 * \file operation_config.h
 * \brief Operation configuration
 */
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

  std::string toString() const {
    std::ostringstream oss;
    oss << "OperationConfig{";
    oss << "buffer_accesses: [";
    bool first_access = true;
    for (const auto& access : buffer_accesses) {
      if (!first_access) oss << ", ";
      oss << "(" << access.first << ", '" << access.second << "')";
      first_access = false;
    }
    oss << "], ";
    oss << "default_pipeline: '" << default_pipeline << "'";
    oss << "}";
    return oss.str();
  }
};

inline const std::unordered_map<std::string, OperationConfig>&
GetOperationConfig() {
  static std::unordered_map<std::string, OperationConfig> operation_config_ = {
      {"copy_gm_to_l1", {{{0, "read"}, {1, "write"}}, "PIPE_MTE2"}},
      {"copy_gm_to_l0a", {{{0, "read"}, {1, "write"}}, "PIPE_MTE2"}},
      {"copy_gm_to_l0b", {{{0, "read"}, {1, "write"}}, "PIPE_MTE2"}},
      {"copy_gm_to_ub", {{{0, "read"}, {1, "write"}}, "PIPE_MTE2"}},
      {"copy_l1_to_l0a", {{{0, "read"}, {1, "write"}}, "PIPE_MTE1"}},
      {"copy_l1_to_l0b", {{{0, "read"}, {1, "write"}}, "PIPE_MTE1"}},
      {"copy_ub_to_gm", {{{0, "read"}, {1, "write"}}, "PIPE_MTE3"}},
      {"copy_ub_to_l1", {{{0, "read"}, {1, "write"}}, "PIPE_MTE3"}},
      {"copy_l0c_to_gm", {{{0, "read"}, {1, "write"}}, "PIPE_FIX"}},
      {"copy_l0c_to_l1", {{{0, "read"}, {1, "write"}}, "PIPE_FIX"}},
      {"copy_ub_to_ub", {{{0, "read"}, {1, "write"}}, "PIPE_V"}},
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
      {"tl.ascend_createvecindex", {{{0, "write"}, {1, "read"}}, "PIPE_V"}},
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
       {{{1, "write"}}, "PIPE_V"}},
  };

  return operation_config_;
}

inline const std::unordered_map<std::string, std::string>& GetEventMapping() {
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
    "shared", "shared.dyn", "wmma.matrix_a", "wmma.matrix_b",
    "wmma.accumulator"};

/*! \brief The memory scope that requires alignment for its inner
 * dimension. */
const std::unordered_map<std::string, int> kScopeForAlignment = {
    {"shared", 32 * 8}};

const std::unordered_map<const tvm::OpNode*, int64_t> ascendc_tmp_arg_ops = {
    {tl::ascend_clamp().get(), 3},
    {tl::ascend_clamp_max().get(), 3},
    {tl::ascend_clamp_min().get(), 3},
    {tl::ascend_reduce().get(), 3},
    {tl::ascend_sort().get(), 3},
    {tl::ascend_topk().get(), 3},
    {tl::ascend_sigmoid().get(), 2},
    {tl::ascend_bilinear_interpolation().get(), 10},
    {tl::ascend_sin().get(), 2},
    {tl::ascend_cos().get(), 2},
    {tl::ascend_pow().get(), 3},
    {tl::ascend_bitwise_xor().get(), 3},
    {tl::ascend_round().get(), 2},
    {tl::ascend_broadcast().get(), 3},
    {tl::ascend_reducesum_experiment().get(), 2},
    {tl::ascend_reducesum_mask_experiment().get(), 2},
    {tl::ascend_merge_sort().get(), 3},
};

// The PTO currently supports the following vector APIs with tmp parameters.
// However, among these, only the reduce and bitwise_xor operators actually
// require tmp. For other APIs, tmp is retained to keep the codegen logic for
// obtaining API arguments unchanged.
const std::unordered_map<const tvm::OpNode*, int64_t> pto_tmp_arg_ops = {
    {tl::ascend_clamp().get(), 3},       {tl::ascend_clamp_max().get(), 3},
    {tl::ascend_clamp_min().get(), 3},   {tl::ascend_reduce().get(), 3},
    {tl::ascend_sigmoid().get(), 2},     {tl::ascend_pow().get(), 3},
    {tl::ascend_bitwise_xor().get(), 3}, {tl::ascend_round().get(), 2},
    {tl::ascend_broadcast().get(), 3},   {tl::ascend_merge_sort().get(), 3},
    {tl::ascend_select().get(), 3},      {tl::ascend_gather_mask().get(), 4},
};

}  // namespace tl
}  // namespace tvm