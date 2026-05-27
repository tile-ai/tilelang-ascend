#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR Q_handle,  GM_ADDR K_handle,  GM_ADDR V_handle,  GM_ADDR Output_handle,  GM_ADDR q_seq_starts_handle,  GM_ADDR kv_seq_starts_handle,  GM_ADDR actual_q_len_handle,  GM_ADDR actual_kv_len_handle,  GM_ADDR cum_seq_tiles_per_request_handle,  GM_ADDR visible_end_handle,  GM_ADDR diag_col_handle,  GM_ADDR matched_prefix_lens_handle,  GM_ADDR block_table_handle,  GM_ADDR key_cache_handle,  GM_ADDR value_cache_handle,  GM_ADDR task_meta_handle,  GM_ADDR workspace_kv_handle,  GM_ADDR workspace_kv_v_handle,  GM_ADDR workspace_s_handle,  GM_ADDR workspace_p_handle,  GM_ADDR workspace_o_handle, int64_t total_live_q, int64_t total_live_kv, int64_t batch, int64_t max_request_len, int64_t max_blocks, int64_t num_blocks, int64_t total_tasks, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<bfloat16_t> Q;
  Q.SetGlobalBuffer((__gm__ bfloat16_t*)Q_handle);
  AscendC::GlobalTensor<bfloat16_t> K;
  K.SetGlobalBuffer((__gm__ bfloat16_t*)K_handle);
  AscendC::GlobalTensor<bfloat16_t> V;
  V.SetGlobalBuffer((__gm__ bfloat16_t*)V_handle);
  AscendC::GlobalTensor<bfloat16_t> Output;
  Output.SetGlobalBuffer((__gm__ bfloat16_t*)Output_handle);
  AscendC::GlobalTensor<int> q_seq_starts;
  q_seq_starts.SetGlobalBuffer((__gm__ int*)q_seq_starts_handle);
  AscendC::GlobalTensor<int> kv_seq_starts;
  kv_seq_starts.SetGlobalBuffer((__gm__ int*)kv_seq_starts_handle);
  AscendC::GlobalTensor<int> actual_q_len;
  actual_q_len.SetGlobalBuffer((__gm__ int*)actual_q_len_handle);
  AscendC::GlobalTensor<int> actual_kv_len;
  actual_kv_len.SetGlobalBuffer((__gm__ int*)actual_kv_len_handle);
  AscendC::GlobalTensor<int> cum_seq_tiles_per_request;
  cum_seq_tiles_per_request.SetGlobalBuffer((__gm__ int*)cum_seq_tiles_per_request_handle);
  AscendC::GlobalTensor<float> visible_end;
  visible_end.SetGlobalBuffer((__gm__ float*)visible_end_handle);
  AscendC::GlobalTensor<float> diag_col;
  diag_col.SetGlobalBuffer((__gm__ float*)diag_col_handle);
  AscendC::GlobalTensor<int> matched_prefix_lens;
  matched_prefix_lens.SetGlobalBuffer((__gm__ int*)matched_prefix_lens_handle);
  AscendC::GlobalTensor<int> block_table;
  block_table.SetGlobalBuffer((__gm__ int*)block_table_handle);
  AscendC::GlobalTensor<bfloat16_t> key_cache;
  key_cache.SetGlobalBuffer((__gm__ bfloat16_t*)key_cache_handle);
  AscendC::GlobalTensor<bfloat16_t> value_cache;
  value_cache.SetGlobalBuffer((__gm__ bfloat16_t*)value_cache_handle);
  AscendC::GlobalTensor<int> task_meta;
  task_meta.SetGlobalBuffer((__gm__ int*)task_meta_handle);
  AscendC::GlobalTensor<bfloat16_t> workspace_kv;
  workspace_kv.SetGlobalBuffer((__gm__ bfloat16_t*)workspace_kv_handle);
  AscendC::GlobalTensor<bfloat16_t> workspace_kv_v;
  workspace_kv_v.SetGlobalBuffer((__gm__ bfloat16_t*)workspace_kv_v_handle);
  AscendC::GlobalTensor<float> workspace_s;
  workspace_s.SetGlobalBuffer((__gm__ float*)workspace_s_handle);
  AscendC::GlobalTensor<bfloat16_t> workspace_p;
  workspace_p.SetGlobalBuffer((__gm__ bfloat16_t*)workspace_p_handle);
  AscendC::GlobalTensor<float> workspace_o;
  workspace_o.SetGlobalBuffer((__gm__ float*)workspace_o_handle);

  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
  pipe.InitBuffer(ascend_l0a, 65536);
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
  pipe.InitBuffer(ascend_l0b, 65536);
  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1; pipe.InitBuffer(ascend_l1, 524032);
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c; pipe.InitBuffer(ascend_l0c, 131072);
  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  pipe.Destroy();
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    cid = cid / 2;
  }
  auto q_l1 = ascend_l1.GetWithOffset<bfloat16_t>(4096, 0);
  auto k_l1 = ascend_l1.GetWithOffset<bfloat16_t>(4096, 8192);
  auto acc_s_l0c = ascend_l0c.GetWithOffset<float>(4096, 0);
  auto acc_s_l1 = ascend_l1.GetWithOffset<bfloat16_t>(4096, 16384);
  auto v_l1 = ascend_l1.GetWithOffset<bfloat16_t>(4096, 24576);
  auto acc_o_l0c = ascend_l0c.GetWithOffset<float>(4096, 16384);
  auto acc_o = ascend_ub.GetWithOffset<float>(2048, 0);
  auto sumexp = ascend_ub.GetWithOffset<float>(32, 8192);
  auto m_i = ascend_ub.GetWithOffset<float>(32, 8320);
  auto visible_end_ub = ascend_ub.GetWithOffset<float>(64, 8448);
  auto diag_col_ub = ascend_ub.GetWithOffset<float>(64, 8704);
  auto acc_o_half = ascend_ub.GetWithOffset<bfloat16_t>(2048, 46848);
  auto kv_ub = ascend_ub.GetWithOffset<bfloat16_t>(64, 8960);
  auto acc_s_ub = ascend_ub.GetWithOffset<float>(2048, 9088);
  auto acc_s_ub_ = ascend_ub.GetWithOffset<float>(2048, 17280);
  auto kv_col_base_ub = ascend_ub.GetWithOffset<float>(64, 25472);
  auto kv_col_float_ub = ascend_ub.GetWithOffset<float>(64, 25728);
  auto mask_valid_ub = ascend_ub.GetWithOffset<uint8_t>(8, 25984);
  auto mask_vis_ub = ascend_ub.GetWithOffset<uint8_t>(8, 26016);
  auto mask_diag_ub = ascend_ub.GetWithOffset<uint8_t>(8, 26048);
  auto mask_combined_ub = ascend_ub.GetWithOffset<uint8_t>(8, 26080);
  auto m_i_prev = ascend_ub.GetWithOffset<float>(32, 26112);
  auto tmp_ub = ascend_ub.GetWithOffset<uint8_t>(8192, 26240);
  auto sumexp_i_ub = ascend_ub.GetWithOffset<float>(32, 34432);
  auto acc_s_half = ascend_ub.GetWithOffset<bfloat16_t>(2048, 34560);
  auto acc_o_ub = ascend_ub.GetWithOffset<float>(2048, 38656);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    for (int32_t core_index = 0; core_index < ((total_tasks + 23) / 24); ++core_index) {
      AscendC::PipeBarrier<PIPE_ALL>();
      if (((core_index * 24) + cid) < total_tasks) {
        AscendC::PipeBarrier<PIPE_ALL>();
        int32_t condval;
        if ((batch == 1)) {
          condval = 0;
        } else {
          condval = cum_seq_tiles_per_request.GetValue((((int64_t)batch) - (int64_t)2));
        }
        int32_t cum_before = condval;
        AscendC::PipeBarrier<PIPE_ALL>();
        int32_t q_packed_start = ((((core_index * 384) + ((cid / 4) * 64)) + q_seq_starts.GetValue((((int64_t)batch) - (int64_t)1))) - (cum_before * 64));
        tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(q_l1[0], Q[((q_packed_start * 256) + ((cid % 4) * 64))], 256, ((64 <= (total_live_q - q_packed_start)) ? 64 : ((0 < (total_live_q - q_packed_start)) ? (total_live_q - q_packed_start) : 0)), 64);
        AscendC::PipeBarrier<PIPE_ALL>();
        int32_t prefix_len_b = matched_prefix_lens.GetValue((((int64_t)batch) - (int64_t)1));
        AscendC::PipeBarrier<PIPE_ALL>();
        int32_t kv_len_b = (prefix_len_b + actual_kv_len.GetValue((((int64_t)batch) - (int64_t)1)));
        for (int32_t k_i = 0; k_i < ((kv_len_b + 63) / 64); ++k_i) {
          AscendC::CrossCoreWaitFlag(0);
          AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(5);
          AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(5);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(k_l1[0], workspace_kv[(cid * 4096)], 64, 64, 64);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(1);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(1);
          AscendC::SetFlag<AscendC::HardEvent::FIX_M>(6);
          AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(6);
          tl::ascend::gemm_v0<bfloat16_t, float, 64, 64, 64, false, true>(q_l1[0], k_l1[0], acc_s_l0c[0], ascend_l0a, ascend_l0b, (bool)1);
          AscendC::SetFlag<AscendC::HardEvent::M_FIX>(2);
          AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(2);
          AscendC::PipeBarrier<PIPE_FIX>();
          tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 64, 0>(workspace_s[(cid * 4096)], acc_s_l0c[0], 64, 64, 64);
          AscendC::CrossCoreSetFlag<2, PIPE_FIX>(4);
          AscendC::CrossCoreWaitFlag(3);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(acc_s_l1[0], workspace_p[(cid * 4096)], 64, 64, 64);
          AscendC::CrossCoreWaitFlag(1);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(v_l1[0], workspace_kv_v[(cid * 4096)], 64, 64, 64);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(3);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(3);
          tl::ascend::gemm_v0<bfloat16_t, float, 64, 64, 64, false, false>(acc_s_l1[0], v_l1[0], acc_o_l0c[0], ascend_l0a, ascend_l0b, (bool)1);
          AscendC::SetFlag<AscendC::HardEvent::M_FIX>(4);
          AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(4);
          tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 64, 0>(workspace_o[(cid * 4096)], acc_o_l0c[0], 64, 64, 64);
          AscendC::CrossCoreSetFlag<2, PIPE_FIX>(2);
        }
      }
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::PipeBarrier<PIPE_ALL>();
    }
  }
  if ASCEND_IS_AIV {
    for (int32_t core_index_1 = 0; core_index_1 < ((total_tasks + 23) / 24); ++core_index_1) {
      AscendC::PipeBarrier<PIPE_ALL>();
      if (((core_index_1 * 24) + cid) < total_tasks) {
        AscendC::PipeBarrier<PIPE_ALL>();
        int32_t condval_1;
        if ((batch == 1)) {
          condval_1 = 0;
        } else {
          condval_1 = cum_seq_tiles_per_request.GetValue((((int64_t)batch) - (int64_t)2));
        }
        int32_t cum_before_1 = condval_1;
        AscendC::PipeBarrier<PIPE_ALL>();
        int32_t q_packed_start_1 = ((((core_index_1 * 384) + ((cid / 4) * 64)) + q_seq_starts.GetValue((((int64_t)batch) - (int64_t)1))) - (cum_before_1 * 64));
        tl::ascend::Fill<float>(acc_o[0], 0.000000e+00f, 2048);
        tl::ascend::Fill<float>(sumexp[0], 0.000000e+00f, 32);
        tl::ascend::Fill<float>(m_i[0], -1.073742e+09f, 32);
        AscendC::PipeBarrier<PIPE_ALL>();
        int32_t prefix_len_b_1 = matched_prefix_lens.GetValue((((int64_t)batch) - (int64_t)1));
        AscendC::PipeBarrier<PIPE_ALL>();
        int32_t kv_len_b_1 = (prefix_len_b_1 + actual_kv_len.GetValue((((int64_t)batch) - (int64_t)1)));
        tl::ascend::copy_gm_to_ub<float, 64>(visible_end_ub[0], visible_end[((((core_index_1 * 384) + ((cid / 4) * 64)) + ((batch - 1) * max_request_len)) - (cum_before_1 * 64))], (max_request_len * batch), 1, ((1 <= ((((max_request_len / 64) + cum_before_1) - (cid / 4)) - (core_index_1 * 6))) ? 64 : ((0 < ((((cum_before_1 * 64) + max_request_len) - ((cid / 4) * 64)) - (core_index_1 * 384))) ? ((((cum_before_1 * 64) + max_request_len) - ((cid / 4) * 64)) - (core_index_1 * 384)) : 0)), 0.000000e+00f);
        tl::ascend::copy_gm_to_ub<float, 64>(diag_col_ub[0], diag_col[((((core_index_1 * 384) + ((cid / 4) * 64)) + ((batch - 1) * max_request_len)) - (cum_before_1 * 64))], (max_request_len * batch), 1, ((1 <= ((((max_request_len / 64) + cum_before_1) - (cid / 4)) - (core_index_1 * 6))) ? 64 : ((0 < ((((cum_before_1 * 64) + max_request_len) - ((cid / 4) * 64)) - (core_index_1 * 384))) ? ((((cum_before_1 * 64) + max_request_len) - ((cid / 4) * 64)) - (core_index_1 * 384)) : 0)), 0.000000e+00f);
        for (int32_t k_i_1 = 0; k_i_1 < ((kv_len_b_1 + 63) / 64); ++k_i_1) {
          for (int32_t row = 0; row < 32; ++row) {
            AscendC::PipeBarrier<PIPE_ALL>();
            if ((((k_i_1 * 64) + (vid * 32)) + row) < prefix_len_b_1) {
              AscendC::PipeBarrier<PIPE_ALL>();
              int32_t physical_block = block_table.GetValue((((((int64_t)batch) - (int64_t)1) * ((int64_t)max_blocks)) + ((int64_t)k_i_1)));
              tl::ascend::copy_gm_to_ub<bfloat16_t, 64>(kv_ub[0], key_cache[((((physical_block * 16384) + (vid * 8192)) + (row * 256)) + ((cid % 4) * 64))], (num_blocks * 16384), 1, 64, bfloat16_t(0.000000e+00f));
            } else {
              AscendC::PipeBarrier<PIPE_ALL>();
              if ((((k_i_1 * 64) + (vid * 32)) + row) < kv_len_b_1) {
                AscendC::PipeBarrier<PIPE_ALL>();
                int32_t kv_packed_pos = (((((k_i_1 * 64) + (vid * 32)) + kv_seq_starts.GetValue((((int64_t)batch) - (int64_t)1))) + row) - prefix_len_b_1);
                tl::ascend::copy_gm_to_ub<bfloat16_t, 64>(kv_ub[0], K[((kv_packed_pos * 256) + ((cid % 4) * 64))], (total_live_kv * 256), 1, 64, bfloat16_t(0.000000e+00f));
              } else {
                tl::ascend::Fill<bfloat16_t>(kv_ub[0], 0.000000e+00f, 64);
              }
              AscendC::PipeBarrier<PIPE_ALL>();
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            tl::ascend::copy_ub_to_gm<bfloat16_t, 64>(workspace_kv[(((cid * 4096) + (vid * 2048)) + (row * 64))], kv_ub[0], 98304, 1, 64);
          }
          AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(0);
          tl::ascend::Fill<float>(acc_s_ub[0], 0.000000e+00f, 2048);
          AscendC::CrossCoreWaitFlag(4);
          tl::ascend::copy_gm_to_ub<float, 64, 32>(acc_s_ub_[0], workspace_s[((cid * 4096) + (vid * 2048))], 64, 32, 64, 0.000000e+00f);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(5);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(5);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Add(acc_s_ub[0], acc_s_ub[0], acc_s_ub_[0], 2048);
          AscendC::PipeBarrier<PIPE_V>();
          {
          AscendC::Muls(acc_s_ub[0], acc_s_ub[0], 1.250000e-01f, 2048);
          }
          tl::ascend::ArithProgression<float>(kv_col_base_ub[0], 0.000000e+00f, 1.000000e+00f, 64);
          AscendC::PipeBarrier<PIPE_V>();
          {
          AscendC::Adds(kv_col_float_ub[0], kv_col_base_ub[0], ((float)(k_i_1 * 64)), 64);
          }
          int32_t condval_2;
          if ((kv_len_b_1 < ((k_i_1 * 64) + 64))) {
            condval_2 = (kv_len_b_1 - (k_i_1 * 64));
          } else {
            condval_2 = 64;
          }
          AscendC::CompareScalar(mask_valid_ub[0], kv_col_base_ub[0], ((float)condval_2), AscendC::CMPMODE::LT, 64);
          for (int32_t row_1 = 0; row_1 < 32; ++row_1) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::PipeBarrier<PIPE_ALL>();
            auto visible_end_ub_scalar = visible_end_ub.GetValue(((vid * 32) + row_1));
            AscendC::CompareScalar(mask_vis_ub[0], kv_col_float_ub[0], visible_end_ub_scalar, AscendC::CMPMODE::LT, 64);
            AscendC::PipeBarrier<PIPE_ALL>();
            auto diag_col_ub_scalar = diag_col_ub.GetValue(((vid * 32) + row_1));
            AscendC::CompareScalar(mask_diag_ub[0], kv_col_float_ub[0], diag_col_ub_scalar, AscendC::CMPMODE::EQ, 64);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Or(mask_combined_ub[0], mask_vis_ub[0], mask_diag_ub[0], 8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::And(mask_combined_ub[0], mask_combined_ub[0], mask_valid_ub[0], 8);
            AscendC::PipeBarrier<PIPE_V>();
AscendC::Select<float, uint8_t>(acc_s_ub[(row_1 * 64)], mask_combined_ub[0], acc_s_ub[(row_1 * 64)], static_cast<float>(-CUDART_INF_F), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 64);
          }
          tl::ascend::copy_ub_to_ub<float, float, 32>(m_i_prev[0], m_i[0]);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::reduce_max<float, 32, 64, -1>(m_i[0], acc_s_ub[0], tmp_ub[0], true);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Max(m_i[0], m_i[0], m_i_prev[0], 32);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Sub(m_i_prev[0], m_i_prev[0], m_i[0], 32);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Exp(m_i_prev[0], m_i_prev[0], 32);
          for (int32_t row_2 = 0; row_2 < 32; ++row_2) {
            AscendC::PipeBarrier<PIPE_V>();
            {
            AscendC::PipeBarrier<PIPE_ALL>();
            auto m_i_scalar = -(float)m_i.GetValue(row_2);
            AscendC::Adds(acc_s_ub[(row_2 * 64)], acc_s_ub[(row_2 * 64)], m_i_scalar, 64);
            }
          }
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Exp(acc_s_ub[0], acc_s_ub[0], 2048);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::reduce_sum<float,  32,  64,  -1>(sumexp_i_ub[0], acc_s_ub[0], tmp_ub[0], true);
          AscendC::Mul(sumexp[0], sumexp[0], m_i_prev[0], 32);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Add(sumexp[0], sumexp[0], sumexp_i_ub[0], 32);
          for (int32_t row_3 = 0; row_3 < 32; ++row_3) {
            AscendC::PipeBarrier<PIPE_V>();
            {
            AscendC::PipeBarrier<PIPE_ALL>();
            auto m_i_prev_scalar = m_i_prev.GetValue(row_3);
            AscendC::Muls(acc_o[(row_3 * 64)], acc_o[(row_3 * 64)], m_i_prev_scalar, 64);
            }
          }
          tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(acc_s_half[0], acc_s_ub[0]);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(6);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(6);
          tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_p[((cid * 4096) + (vid * 2048))], acc_s_half[0], 64, 32, 64);
          AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(3);
          for (int32_t row_4 = 0; row_4 < 32; ++row_4) {
            AscendC::PipeBarrier<PIPE_ALL>();
            if ((((k_i_1 * 64) + (vid * 32)) + row_4) < prefix_len_b_1) {
              AscendC::PipeBarrier<PIPE_ALL>();
              int32_t physical_block_1 = block_table.GetValue((((((int64_t)batch) - (int64_t)1) * ((int64_t)max_blocks)) + ((int64_t)k_i_1)));
              tl::ascend::copy_gm_to_ub<bfloat16_t, 64>(kv_ub[0], value_cache[((((physical_block_1 * 16384) + (vid * 8192)) + (row_4 * 256)) + ((cid % 4) * 64))], (num_blocks * 16384), 1, 64, bfloat16_t(0.000000e+00f));
            } else {
              AscendC::PipeBarrier<PIPE_ALL>();
              if ((((k_i_1 * 64) + (vid * 32)) + row_4) < kv_len_b_1) {
                AscendC::PipeBarrier<PIPE_ALL>();
                int32_t kv_packed_pos_1 = (((((k_i_1 * 64) + (vid * 32)) + kv_seq_starts.GetValue((((int64_t)batch) - (int64_t)1))) + row_4) - prefix_len_b_1);
                tl::ascend::copy_gm_to_ub<bfloat16_t, 64>(kv_ub[0], V[((kv_packed_pos_1 * 256) + ((cid % 4) * 64))], (total_live_kv * 256), 1, 64, bfloat16_t(0.000000e+00f));
              } else {
                tl::ascend::Fill<bfloat16_t>(kv_ub[0], 0.000000e+00f, 64);
              }
              AscendC::PipeBarrier<PIPE_ALL>();
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            tl::ascend::copy_ub_to_gm<bfloat16_t, 64>(workspace_kv_v[(((cid * 4096) + (vid * 2048)) + (row_4 * 64))], kv_ub[0], 98304, 1, 64);
          }
          AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(1);
          AscendC::CrossCoreWaitFlag(2);
          tl::ascend::copy_gm_to_ub<float, 64, 32>(acc_o_ub[0], workspace_o[((cid * 4096) + (vid * 2048))], 64, 32, 64, 0.000000e+00f);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(7);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(7);
          AscendC::Add(acc_o[0], acc_o[0], acc_o_ub[0], 2048);
        }
        for (int32_t row_5 = 0; row_5 < 32; ++row_5) {
          AscendC::PipeBarrier<PIPE_ALL>();
          if (0.000000e+00f <= visible_end_ub.GetValue(((vid * 32) + row_5))) {
            AscendC::PipeBarrier<PIPE_ALL>();
            auto sumexp_scalar = 1.0f / (float)sumexp.GetValue(((vid * 32) + row_5));
            AscendC::Muls(acc_o[(row_5 * 64)], acc_o[(row_5 * 64)], sumexp_scalar, 64);
          }
          AscendC::PipeBarrier<PIPE_ALL>();
          AscendC::PipeBarrier<PIPE_ALL>();
        }
        tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(acc_o_half[0], acc_o[0]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
        tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(Output[(((vid * 8192) + (q_packed_start_1 * 256)) + ((cid % 4) * 64))], acc_o_half[0], 256, ((1 <= (((total_live_q - q_packed_start_1) / 32) - vid)) ? 32 : ((0 < ((total_live_q - q_packed_start_1) - (vid * 32))) ? ((total_live_q - q_packed_start_1) - (vid * 32)) : 0)), 64);
      }
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::PipeBarrier<PIPE_ALL>();
    }
  }
}

void main_kernel_tiling(int64_t total_live_q, int64_t total_live_kv, int64_t batch, int64_t max_request_len, int64_t max_blocks, int64_t num_blocks, int64_t total_tasks) {
}

extern "C" void call(uint8_t* Q_handle, uint8_t* K_handle, uint8_t* V_handle, uint8_t* Output_handle, uint8_t* q_seq_starts_handle, uint8_t* kv_seq_starts_handle, uint8_t* actual_q_len_handle, uint8_t* actual_kv_len_handle, uint8_t* cum_seq_tiles_per_request_handle, uint8_t* visible_end_handle, uint8_t* diag_col_handle, uint8_t* matched_prefix_lens_handle, uint8_t* block_table_handle, uint8_t* key_cache_handle, uint8_t* value_cache_handle, uint8_t* task_meta_handle, uint8_t* workspace_kv_handle, uint8_t* workspace_kv_v_handle, uint8_t* workspace_s_handle, uint8_t* workspace_p_handle, uint8_t* workspace_o_handle, int64_t total_live_q, int64_t total_live_kv, int64_t batch, int64_t max_request_len, int64_t max_blocks, int64_t num_blocks, int64_t total_tasks, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling(total_live_q, total_live_kv, batch, max_request_len, max_blocks, num_blocks, total_tasks);
  main_kernel<<<24, nullptr, stream>>>(Q_handle, K_handle, V_handle, Output_handle, q_seq_starts_handle, kv_seq_starts_handle, actual_q_len_handle, actual_kv_len_handle, cum_seq_tiles_per_request_handle, visible_end_handle, diag_col_handle, matched_prefix_lens_handle, block_table_handle, key_cache_handle, value_cache_handle, task_meta_handle, workspace_kv_handle, workspace_kv_v_handle, workspace_s_handle, workspace_p_handle, workspace_o_handle, total_live_q, total_live_kv, batch, max_request_len, max_blocks, num_blocks, total_tasks, fftsAddr);
}