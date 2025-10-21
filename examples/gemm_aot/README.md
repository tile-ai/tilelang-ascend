# TileLang-Ascend AOT Operator Example

This directory provides an example of implementing an **Ahead-of-Time (AOT)** compiled operator, using **GEMM (General Matrix Multiplication)** as a demonstration.  
It shows how an operator can be generated, compiled, and tested through a simple automated workflow.


## Overview

The example demonstrates the full workflow of building and testing an AOT-compiled operator:
1. **Operator generation** (from Python to C++)  
2. **Compilation** of the generated operator code  
3. **Validation** of results through test scripts  

---

## Directory Structure

```
gemm_aot/
├── example_gemm.py           # Python script that generates GEMM operator code
├── build.sh                  # Build script to compile generated C++ code
├── example_gemm.cpp          # Generated C++ operator code
├── test_example_gemm.py      # Script to test the compiled AOT operator
├── run_example_gemm_aot.sh   # One-click script to run the full example
└── README.md
```

---

## Running the Example

We assume that you have already [installed the tilelang-ascend](../../README.md#tilelang-ascend-installation).

To run the complete AOT GEMM example:

```bash
bash run_example_gemm_aot.sh
```

## How to debug
This example demonstrates how to use the AOT (Ahead-of-Time) compilation workflow for GEMM while adding debugging functionalities through **PRINTF** and **DumpTensor**.

In this updated version of the AOT GEMM example:
- The operator library is loaded directly from `./kernel_lib.so`.
- The kernel code includes additional debugging tools such as:
  - `AscendC::InitDump()` to initialize dump buffers.
  - `AscendC::DumpTensor()` to inspect specific intermediate tensors.
  - `AscendC::PRINTF();` for textual runtime diagnostics.

These additions enable developers to track intermediate computations, validate memory layout correctness, and analyze numerical precision issues.

It illustrates both the Python-side integration and the kernel-side instrumentation for runtime data inspection and verification.


## Kernel Implementation(example_gemm_debug.cpp)

The kernel has been enhanced with debugging utilities.

```cpp
#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;

extern "C" CATLASS_GLOBAL
void main_kernel( GM_ADDR A_handle,  GM_ADDR B_handle,  GM_ADDR C_handle, GM_ADDR dumpHandle, uint64_t fftsAddr) {
  AscendC::InitDump(true, dumpHandle, 1024 * 1024); // Initialize dump for debugging
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
      AscendC::DumpTensor(A_L1, 96, 128 * 64);  // DumpTensor
      AscendC::PRINTF("Add debug info.");  // PRINTF
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

extern "C" void call(uint8_t* A_handle, uint8_t* B_handle, uint8_t* C_handle, GM_ADDR dumpHandle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<256, nullptr, stream>>>(A_handle, B_handle, C_handle, dumpHandle, fftsAddr);
}
```

## Python Script(test_example_gemm_debug.py)
Add dump_workspace on global memory for dump and print.
```
import torch
import argparse
import ctypes
from functools import partial
torch.manual_seed(42)

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=8192, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=8192, help="Matrix K dimension")
args = parser.parse_args()

M = args.m
N = args.n
K = args.k


a = torch.randn(M, K).half().npu()
b = torch.randn(K, N).half().npu()
c = torch.empty(M, N).half().npu()
dump_workspace = torch.empty(75 * 1024 * 1024 // 4).float().npu()
print("init successful!")

lib_path = "./kernel_lib.so"
lib = ctypes.CDLL(lib_path)
stream = torch.npu.current_stream()._as_parameter_

def tl_gemm():
    return lib.call(
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(c.data_ptr()),
        ctypes.c_void_p(dump_workspace.data_ptr()),
        stream)

tl_gemm()
dump_workspace.detach().cpu().numpy().tofile("dump_workspace.bin")
torch.npu.synchronize()

ref_c = a @ b
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")

```


## Output Files

After execution, you should have:
- `dump_workspace.bin` — binary dump of intermediate tensor states.

Then use show_kernel_debug_data to obtain and parse the debugging information (parse the bin file into a readable format.

```
show_kernel_debug_data dump_workspace.bin ./
```

