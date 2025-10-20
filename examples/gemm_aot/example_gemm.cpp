#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;

extern "C" CATLASS_GLOBAL
void main_kernel( GM_ADDR A_handle,  GM_ADDR B_handle,  GM_ADDR C_handle, uint64_t fftsAddr) {
  AscendC::SetSyncBaseAddr(fftsAddr);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<half> A;
  A.SetGlobalBuffer((__gm__ half*)A_handle);
  AscendC::GlobalTensor<half> B;
  B.SetGlobalBuffer((__gm__ half*)B_handle);
  AscendC::GlobalTensor<half> C;
  C.SetGlobalBuffer((__gm__ half*)C_handle);

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
  auto A_L1 = ascend_l1.GetWithOffset<half>(8192,0);
  auto B_L1 = ascend_l1.GetWithOffset<half>(16384,16384);
  auto C_L0 = ascend_l0c.GetWithOffset<float>(32768,0);
  if ASCEND_IS_AIC {
    for (int32_t k = 0; k < 128; ++k) {
      tl::ascend::copy_gm_to_l1<half, 128, 64>(A_L1[0], A[(((cid / 4) * 1048576) + (k * 64))], 8192);
      tl::ascend::copy_gm_to_l1<half, 64, 256>(B_L1[0], B[((k * 65536) + ((cid % 4) * 256))], 1024);
      AscendC::PipeBarrier<PIPE_ALL>();
      tl::ascend::gemm_v0<half, float, 128, 256, 64, false, false>(A_L1[0], B_L1[0], C_L0[0], ascend_l0a, ascend_l0b, (k == 0));
      AscendC::PipeBarrier<PIPE_ALL>();
    }
    tl::ascend::copy_l0c_to_gm<float, half, layout::RowMajor, 128, 256>(C[(((cid / 4) * 131072) + ((cid % 4) * 256))], C_L0[0], 1024, 0);
  }
}

void main_kernel_tiling() {
}

extern "C" void call(uint8_t* A_handle, uint8_t* B_handle, uint8_t* C_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<256, nullptr, stream>>>(A_handle, B_handle, C_handle, fftsAddr);
}
