#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;

extern "C" CATLASS_GLOBAL
void main_kernel( GM_ADDR Q_handle,  GM_ADDR KV_handle,  GM_ADDR Indices_handle,  GM_ADDR Output_handle,  GM_ADDR actual_q_len_handle,  GM_ADDR actual_kv_len_handle,  GM_ADDR block_table_handle,  GM_ADDR workspace_1_handle,  GM_ADDR workspace_2_handle,  GM_ADDR workspace_3_handle,  GM_ADDR workspace_4_handle,  GM_ADDR workspace_5_handle, int64_t seq_len, int64_t block_table_len, uint64_t fftsAddr) {
  int64_t batch = 1;
  AscendC::SetSyncBaseAddr(fftsAddr);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<bfloat16_t> Q;
  Q.SetGlobalBuffer((__gm__ bfloat16_t*)Q_handle);
  AscendC::GlobalTensor<bfloat16_t> KV;
  KV.SetGlobalBuffer((__gm__ bfloat16_t*)KV_handle);
  AscendC::GlobalTensor<int> Indices;
  Indices.SetGlobalBuffer((__gm__ int*)Indices_handle);
  AscendC::GlobalTensor<bfloat16_t> Output;
  Output.SetGlobalBuffer((__gm__ bfloat16_t*)Output_handle);
  AscendC::GlobalTensor<int> actual_q_len;
  actual_q_len.SetGlobalBuffer((__gm__ int*)actual_q_len_handle);
  AscendC::GlobalTensor<int> actual_kv_len;
  actual_kv_len.SetGlobalBuffer((__gm__ int*)actual_kv_len_handle);
  AscendC::GlobalTensor<int> block_table;
  block_table.SetGlobalBuffer((__gm__ int*)block_table_handle);
  AscendC::GlobalTensor<bfloat16_t> workspace_1;
  workspace_1.SetGlobalBuffer((__gm__ bfloat16_t*)workspace_1_handle);
  AscendC::GlobalTensor<bfloat16_t> workspace_2;
  workspace_2.SetGlobalBuffer((__gm__ bfloat16_t*)workspace_2_handle);
  AscendC::GlobalTensor<float> workspace_3;
  workspace_3.SetGlobalBuffer((__gm__ float*)workspace_3_handle);
  AscendC::GlobalTensor<bfloat16_t> workspace_4;
  workspace_4.SetGlobalBuffer((__gm__ bfloat16_t*)workspace_4_handle);
  AscendC::GlobalTensor<float> workspace_5;
  workspace_5.SetGlobalBuffer((__gm__ float*)workspace_5_handle);

  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
  pipe.InitBuffer(ascend_l0a, 65536);
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
  pipe.InitBuffer(ascend_l0b, 131072);
  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1; pipe.InitBuffer(ascend_l1, 524032);
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c; pipe.InitBuffer(ascend_l0c, 131072);
  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  pipe.Destroy();
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    cid = cid / 2;
  }
  auto q_l1 = ascend_l1.GetWithOffset<bfloat16_t>(32768, 0);
  auto q_tail_l1 = ascend_l1.GetWithOffset<bfloat16_t>(4096, 65536);
  auto kv_l1 = ascend_l1.GetWithOffset<bfloat16_t>(32768, 73728);
  auto kv_tail_l1 = ascend_l1.GetWithOffset<bfloat16_t>(4096, 139264);
  auto acc_s_l0c = ascend_l0c.GetWithOffset<float>(4096, 0);
  auto acc_s_l1 = ascend_l1.GetWithOffset<bfloat16_t>(4096, 139264);
  auto acc_o_l0c = ascend_l0c.GetWithOffset<float>(32768, 0);
  auto acc_o = ascend_ub.GetWithOffset<float>(16384, 0);
  auto sumexp = ascend_ub.GetWithOffset<float>(32, 65536);
  auto m_i = ascend_ub.GetWithOffset<float>(32, 65664);
  auto indices_ub_ = ascend_ub.GetWithOffset<int>(64, 65792);
  auto indices_ub_float = ascend_ub.GetWithOffset<float>(64, 66048);
  auto mask_ub = ascend_ub.GetWithOffset<uint8_t>(8, 164480);
  auto kv_ub = ascend_ub.GetWithOffset<bfloat16_t>(512, 66048);
  auto kv_tail_ub = ascend_ub.GetWithOffset<bfloat16_t>(64, 67072);
  auto acc_s_ub_ = ascend_ub.GetWithOffset<float>(2048, 74368);
  auto acc_s_ub = ascend_ub.GetWithOffset<float>(2048, 66048);
  auto m_i_prev = ascend_ub.GetWithOffset<float>(32, 74240);
  auto tmp_ub = ascend_ub.GetWithOffset<uint8_t>(24576, 74368);
  auto sumexp_i_ub = ascend_ub.GetWithOffset<float>(32, 98944);
  auto acc_s_half = ascend_ub.GetWithOffset<bfloat16_t>(2048, 98944);
  auto acc_o_ub = ascend_ub.GetWithOffset<float>(16384, 98944);
  auto acc_o_half = ascend_ub.GetWithOffset<bfloat16_t>(16384, 98944);
  auto vid = AscendC::GetSubBlockIdx();
  for (int32_t core_index = 0; core_index < (((seq_len * batch) + 9) / 10); ++core_index) {
    if (((core_index * 20) + cid) < ((seq_len * batch) * 2)) {
      int32_t act_q_len = actual_q_len.GetValue(((((((int64_t)core_index) * (int64_t)20) + ((int64_t)cid)) / (((int64_t)seq_len) * (int64_t)2)) % ((int64_t)batch)));
      if (((((core_index * 20) + cid) % (seq_len * 2)) / 2) < act_q_len) {
        if ASCEND_IS_AIC {
          tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 512>(q_l1[0], Q[(((((((core_index * 20) + cid) / (seq_len * 2)) % batch) * seq_len) * 73728) + ((((core_index * 20) + cid) % (seq_len * 2)) * 36864))], 576);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(q_tail_l1[0], Q[((((((((core_index * 20) + cid) / (seq_len * 2)) % batch) * seq_len) * 73728) + ((((core_index * 20) + cid) % (seq_len * 2)) * 36864)) + 512)], 576);
          AscendC::PipeBarrier<PIPE_ALL>();
          for (int32_t __1 = 0; __1 < 32; ++__1) {
            AscendC::CrossCoreWaitFlag(0);
            AscendC::PipeBarrier<PIPE_ALL>();
            tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 512>(kv_l1[0], workspace_1[(cid * 32768)], 512);
            tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(kv_tail_l1[0], workspace_2[(cid * 4096)], 64);
            AscendC::PipeBarrier<PIPE_ALL>();
            tl::ascend::gemm_v0<bfloat16_t, float, 64, 64, 512, false, true>(q_l1[0], kv_l1[0], acc_s_l0c[0], ascend_l0a, ascend_l0b, (bool)1);
            AscendC::PipeBarrier<PIPE_ALL>();
            tl::ascend::gemm_v0<bfloat16_t, float, 64, 64, 64, false, true>(q_tail_l1[0], kv_tail_l1[0], acc_s_l0c[0], ascend_l0a, ascend_l0b, (bool)0);
            AscendC::PipeBarrier<PIPE_ALL>();
            tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 64>(workspace_3[(cid * 4096)], acc_s_l0c[0], 64, 0);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(1);
            AscendC::CrossCoreWaitFlag(2);
            AscendC::PipeBarrier<PIPE_ALL>();
            tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(acc_s_l1[0], workspace_4[(cid * 4096)], 64);
            AscendC::PipeBarrier<PIPE_ALL>();
            tl::ascend::gemm_v0<bfloat16_t, float, 64, 512, 64, false, false>(acc_s_l1[0], kv_l1[0], acc_o_l0c[0], ascend_l0a, ascend_l0b, (bool)1);
            AscendC::PipeBarrier<PIPE_ALL>();
            tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 512>(workspace_5[(cid * 32768)], acc_o_l0c[0], 512, 0);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(3);
            AscendC::CrossCoreWaitFlag(4);
          }
        }
        if ASCEND_IS_AIV {
          AscendC::Duplicate<float>(acc_o[0], 0.000000e+00f, 16384);
          AscendC::Duplicate<float>(sumexp[0], 0.000000e+00f, 32);
          AscendC::Duplicate<float>(m_i[0], -1.073742e+09f, 32);
          AscendC::PipeBarrier<PIPE_ALL>();
          for (int32_t i_i = 0; i_i < 32; ++i_i) {
            tl::ascend::copy_gm_to_ub<int, 64>(indices_ub_[0], Indices[((((((core_index * 20) + cid) % (seq_len * 2)) / 2) * 2048) + (i_i * 64))], (seq_len * 2048));
            AscendC::PipeBarrier<PIPE_ALL>();
            tl::ascend::copy_ub_to_ub<float, int, 64>(indices_ub_float[0], indices_ub_[0]);
            AscendC::PipeBarrier<PIPE_ALL>();
            int32_t actual_len = actual_kv_len.GetValue(((((((int64_t)core_index) * (int64_t)20) + ((int64_t)cid)) / (((int64_t)seq_len) * (int64_t)2)) % ((int64_t)batch)));
            AscendC::PipeBarrier<PIPE_ALL>();
            float valid_kv_len = min(((float)((((core_index * 20) + cid) % (seq_len * 2)) / 2)), ((float)actual_len));
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::CompareScalar(mask_ub[0], indices_ub_float[0], valid_kv_len, AscendC::CMPMODE::LE, 64);
            AscendC::PipeBarrier<PIPE_ALL>();
            for (int32_t bi_i = 0; bi_i < 32; ++bi_i) {
              int32_t index_i = indices_ub_.GetValue(((vid * 32) + bi_i));
              // AscendC::PipeBarrier<PIPE_S>();
              AscendC::PipeBarrier<PIPE_ALL>();
              AscendC::DataCacheCleanAndInvalid<int, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(block_table);
              int32_t block_i = 0; // block_table.GetValue(((((int64_t)index_i) / (int64_t)128) + (((((((int64_t)core_index) * (int64_t)20) + ((int64_t)cid)) / (((int64_t)seq_len) * (int64_t)2)) % ((int64_t)batch)) * ((int64_t)block_table_len))));
              AscendC::PipeBarrier<PIPE_ALL>();
              if (-1 < index_i) {
                tl::ascend::copy_gm_to_ub<bfloat16_t, 512>(kv_ub[0], KV[((block_i * 73728) + ((index_i % 128) * 576))], 38043648);
                tl::ascend::copy_gm_to_ub<bfloat16_t, 64>(kv_tail_ub[0], KV[(((block_i * 73728) + ((index_i % 128) * 576)) + 512)], 38043648);
              } else {
                AscendC::Duplicate<bfloat16_t>(kv_ub[0], 0.000000e+00f, 512);
                AscendC::Duplicate<bfloat16_t>(kv_tail_ub[0], 0.000000e+00f, 64);
              }
              AscendC::PipeBarrier<PIPE_ALL>();
              tl::ascend::copy_ub_to_gm<bfloat16_t, 512>(workspace_1[(((cid * 32768) + (vid * 16384)) + (bi_i * 512))], kv_ub[0], 655360);
              tl::ascend::copy_ub_to_gm<bfloat16_t, 64>(workspace_2[(((cid * 4096) + (vid * 2048)) + (bi_i * 64))], kv_tail_ub[0], 81920);
              AscendC::PipeBarrier<PIPE_ALL>();
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
            AscendC::Duplicate<float>(acc_s_ub_[0], 0.000000e+00f, 2048);
            AscendC::PipeBarrier<PIPE_ALL>();
            for (int32_t i = 0; i < 32; ++i) {
AscendC::Select(acc_s_ub[(i * 64)], mask_ub[0], acc_s_ub_[(i * 64)], -CUDART_INF_F, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 64);
              AscendC::PipeBarrier<PIPE_ALL>();
            }
            tl::ascend::copy_ub_to_ub<float, float, 32>(m_i_prev[0], m_i[0]);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::CrossCoreWaitFlag(1);
            tl::ascend::copy_gm_to_ub<float, 64, 32>(acc_s_ub_[0], workspace_3[((cid * 4096) + (vid * 2048))], 64);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Add(acc_s_ub[0], acc_s_ub[0], acc_s_ub_[0], 2048);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Muls(acc_s_ub[0], acc_s_ub[0], 4.166667e-02f, 2048);
            AscendC::PipeBarrier<PIPE_ALL>();
            tl::ascend::reduce_max<float, 32, 64, AscendC::Pattern::Reduce::AR>(m_i[0], acc_s_ub[0], tmp_ub[0]);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Max(m_i[0], m_i[0], m_i_prev[0], 32);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Sub(m_i_prev[0], m_i_prev[0], m_i[0], 32);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Exp(m_i_prev[0], m_i_prev[0], 32);
            AscendC::PipeBarrier<PIPE_ALL>();
            for (int32_t h_i = 0; h_i < 32; ++h_i) {
              AscendC::PipeBarrier<PIPE_ALL>();
              auto m_i_scalar = m_i.GetValue(h_i);
              AscendC::PipeBarrier<PIPE_ALL>();
              AscendC::Adds(acc_s_ub[(h_i * 64)], acc_s_ub[(h_i * 64)], - m_i_scalar, 64);
              AscendC::PipeBarrier<PIPE_ALL>();
            }
            AscendC::Exp(acc_s_ub[0], acc_s_ub[0], 2048);
            AscendC::PipeBarrier<PIPE_ALL>();
            tl::ascend::reduce_sum<float, 32, 64, AscendC::Pattern::Reduce::AR>(sumexp_i_ub[0], acc_s_ub[0], tmp_ub[0]);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Mul(sumexp[0], sumexp[0], m_i_prev[0], 32);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Add(sumexp[0], sumexp[0], sumexp_i_ub[0], 32);
            AscendC::PipeBarrier<PIPE_ALL>();
            for (int32_t h_i_1 = 0; h_i_1 < 32; ++h_i_1) {
              AscendC::PipeBarrier<PIPE_ALL>();
              auto m_i_prev_scalar = m_i_prev.GetValue(h_i_1);
              AscendC::PipeBarrier<PIPE_ALL>();
              AscendC::Muls(acc_o[(h_i_1 * 512)], acc_o[(h_i_1 * 512)], m_i_prev_scalar, 512);
              AscendC::PipeBarrier<PIPE_ALL>();
            }
            tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(acc_s_half[0], acc_s_ub[0]);
            AscendC::PipeBarrier<PIPE_ALL>();
            tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_4[((cid * 4096) + (vid * 2048))], acc_s_half[0], 64);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(2);
            AscendC::CrossCoreWaitFlag(3);
            AscendC::PipeBarrier<PIPE_ALL>();
            tl::ascend::copy_gm_to_ub<float, 512, 32>(acc_o_ub[0], workspace_5[((cid * 32768) + (vid * 16384))], 512);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Add(acc_o[0], acc_o[0], acc_o_ub[0], 16384);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::CrossCoreSetFlag<0x2, PIPE_V>(4);
            AscendC::PipeBarrier<PIPE_ALL>();
          }
          for (int32_t h_i_2 = 0; h_i_2 < 32; ++h_i_2) {
            AscendC::PipeBarrier<PIPE_ALL>();
            auto sumexp_scalar = 1.0f / sumexp.GetValue(h_i_2);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Muls(acc_o[(h_i_2 * 512)], acc_o[(h_i_2 * 512)], sumexp_scalar, 512);
            AscendC::PipeBarrier<PIPE_ALL>();
          }
          tl::ascend::copy_ub_to_ub<bfloat16_t, float, 16384>(acc_o_half[0], acc_o[0]);
          AscendC::PipeBarrier<PIPE_ALL>();
          tl::ascend::copy_ub_to_gm<bfloat16_t, 512, 32>(Output[((((((((core_index * 20) + cid) / (seq_len * 2)) % batch) * seq_len) * 65536) + ((((core_index * 20) + cid) % (seq_len * 2)) * 32768)) + (vid * 16384))], acc_o_half[0], 512);
        }
      }
    }
  }
}

void main_kernel_tiling(int64_t batch, int64_t seq_len, int64_t block_table_len) {
}

extern "C" void call(uint8_t* Q_handle, uint8_t* KV_handle, uint8_t* Indices_handle, uint8_t* Output_handle, uint8_t* actual_q_len_handle, uint8_t* actual_kv_len_handle, uint8_t* block_table_handle, uint8_t* workspace_1_handle, uint8_t* workspace_2_handle, uint8_t* workspace_3_handle, uint8_t* workspace_4_handle, uint8_t* workspace_5_handle, int64_t seq_len, int64_t block_table_len, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  // main_kernel_tiling(batch, seq_len, block_table_len);
  main_kernel<<<20, nullptr, stream>>>(Q_handle, KV_handle, Indices_handle, Output_handle, actual_q_len_handle, actual_kv_len_handle, block_table_handle, workspace_1_handle, workspace_2_handle, workspace_3_handle, workspace_4_handle, workspace_5_handle, seq_len, block_table_len, fftsAddr);
}