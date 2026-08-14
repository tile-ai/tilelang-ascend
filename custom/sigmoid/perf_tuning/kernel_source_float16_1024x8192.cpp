#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR A_handle,  GM_ADDR B_handle, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<half> A;
  A.SetGlobalBuffer((__gm__ half*)A_handle);
  AscendC::GlobalTensor<half> B;
  B.SetGlobalBuffer((__gm__ half*)B_handle);

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
  auto a_ub = ascend_ub.GetWithOffset<half>(8192, 0);
  auto b_ub = ascend_ub.GetWithOffset<half>(8192, 32768);
  auto tmp_ub = ascend_ub.GetWithOffset<uint8_t>(16384, 16384);
  auto vid = AscendC::GetSubBlockIdx();
  for (int32_t block_idx = 0; block_idx < 22; ++block_idx) {
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
    tl::ascend::copy_gm_to_ub<half, 128, 64>(a_ub[0], A[((((((block_idx * 3) + (cid / 8)) / 8) * 1048576) + (vid * 524288)) + ((((block_idx * 24) + cid) % 64) * 128))], 8192, ((-15 <= ((0 - vid) - ((((block_idx * 3) + (cid / 8)) / 8) * 2))) ? 64 : ((((block_idx * 3) + (cid / 8)) < 64) ? ((1024 - (vid * 64)) - ((((block_idx * 3) + (cid / 8)) / 8) * 128)) : 0)), 128, half(0.000000e+00f));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(5);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(5);
    AscendC::Sigmoid(b_ub[0], a_ub[0], tmp_ub[0], 8192);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(2);
    AscendC::PipeBarrier<PIPE_MTE3>();
    tl::ascend::copy_ub_to_gm<half, 128, 64>(B[((((((block_idx * 3) + (cid / 8)) / 8) * 1048576) + (vid * 524288)) + ((((block_idx * 24) + cid) % 64) * 128))], b_ub[0], 8192, ((-15 <= ((0 - vid) - ((((block_idx * 3) + (cid / 8)) / 8) * 2))) ? 64 : ((((block_idx * 3) + (cid / 8)) < 64) ? ((1024 - (vid * 64)) - ((((block_idx * 3) + (cid / 8)) / 8) * 128)) : 0)), 128);
  }
}

void main_kernel_tiling() {
}

extern "C" void call(uint8_t* A_handle, uint8_t* B_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<24, nullptr, stream>>>(A_handle, B_handle, fftsAddr);
}
