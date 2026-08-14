"""单 case 运行脚本，用于 msprof op simulator 生成流水图。

使用小 shape 避免 simulator 太慢（[128, 128] = 16K elements）。
从 CPU 生成数据再 .npu() 搬到 NPU（避免 simulator 模式下 aclnnInplaceNormal 错误）。
选择 p=1（L1 范数，最简单的 abs + reduce_sum 组合）。
"""
import sys
sys.path.insert(0, "/mnt/workspace/gitCode/cann/tilelang-ascend/examples/cann-bench/foreach_norm")

import torch
import tilelang
tilelang.disable_cache()

from foreach_norm import l1_norm_kernel

# 参数
batch = 1
N = 16384  # 128 * 128
block_N = 8192
launch_cores = 1  # 单核，避免多核解析崩溃
dtype = "float16"

# 从 CPU 生成数据
torch.manual_seed(0)
a_cpu = torch.randn(batch, N, dtype=torch.float16, device="cpu")
a = a_cpu.npu()

# 编译 kernel（带 -g 调试信息需要环境变量 TL_CCE_DEBUG_INFO=1）
kernel = l1_norm_kernel(batch, N, block_N, launch_cores, dtype=dtype)

# 运行
b = kernel(a)
torch.npu.synchronize()
print("done", b.shape, b.dtype)
print("result:", b.cpu())
