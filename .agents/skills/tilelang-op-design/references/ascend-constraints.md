# 技术约束清单（必须遵守）

本项目为 TileLang-Ascend（华为昇腾 NPU），与 GPU 版 TileLang 有显著差异。
**外部参考实现不可直接使用，必须转换为 Ascend 兼容方案。**

## 目录

- [1. 本项目已知限制](#1-本项目已知限制)
- [2. 强制检测规则](#2-强制检测规则)
- [3. 警告输出格式](#3-警告输出格式)

---

## 1. 本项目已知限制

| 约束 | 说明 | 影响 | 替代方案 |
|------|------|------|----------|
| **不支持三维 Kernel** | `T.Kernel` 只接受一维 block 数 | 三维并行设计无法实现 | 使用 `block_metadata` 预计算机制（参考 `examples/grouped_gemm/`） |
| **threads 参数限制** | 只支持 1 或 2，不支持大值 | `threads=128` 等设计报错 | 默认不指定 threads 或设为 2 |
| **动态循环边界不支持** | 循环次数不能依赖 tensor 值（如 `batch_sizes[bz]`） | `T.Pipelined(batch_sizes[bz])` 报错 | 预计算最大循环次数，用 `T.serial(max_iters)` + 条件判断 |
| **流水线不支持动态边界** | `T.Pipelined` 的循环次数必须静态 | 动态批次无法流水线 | 改用 `T.serial` 或预计算固定迭代次数 |
| **部分 GPU API 不可用** | CUDA 专用 API 在 Ascend 不存在 | 直接移植 GPU 代码失败 | 查阅本项目 `examples/` 确认 Ascend API |
| **GEMM 要求 M,N 为 block 整数倍** | `M // block_M` 整除依赖；`M < block_M` 时零 block 启动 | 输出全零或除零编译崩溃 | 设计文档 §4/§5 必须明确处理策略：host 侧 padding+crop 或 Kernel 动态 block |
| **L0C 容量上限** | A2/A3 设备 L0C = 128KB | `block_M × block_N × sizeof(accum) > 128KB` 导致 segfault | 设计 block 时满足 `block_M × block_N ≤ 16384`（float32 accum） |

## 2. 强制检测规则

在设计文档生成前，**必须**执行以下检测：

| 检测项 | 触发条件 | 处理方式 |
|--------|----------|----------|
| 三维 Kernel | 参考实现包含 `T.Kernel(..., batch_count)` 或 3 个维度参数 | **立即警告**，提出 `block_metadata` 方案 |
| threads 参数 | 参考实现 threads > 2 | **立即警告**，建议 threads=2 或移除 |
| 动态循环边界 | 循环边界依赖 tensor 值 | **立即警告**，提出静态边界 + 条件判断方案 |
| GPU 专用 API | CUDA 相关 API（如 `T.gemm` 通用版） | **立即警告**，查阅本项目确认 Ascend API |
| GEMM 非整除风险 | `M` 或 `N` 不被 block size 整除（即 `M % block_M ≠ 0` 或 `N % block_N ≠ 0`） | **立即警告**，要求 design 中明确 padding 策略 |
| L0C 溢出风险 | block_M × block_N × sizeof(accum_dtype) > 131072 (128KB) | **立即警告**，建议减小 block 或拆分 |

## 3. 警告输出格式

```
⚠️ 技术限制检测警告

检测到参考实现包含本项目不支持的功能：

1. 三维 Kernel（本项目只支持一维 Kernel）
   - 参考实现：T.Kernel(m_num, n_num, batch_count)
   - 本项目方案：T.Kernel(total_blocks) + block_metadata 预计算表
   - 参考：examples/grouped_gemm/example_grouped_gemm_fwd.py

2. 动态循环边界（本项目不支持 tensor 值作为循环边界）
   - 参考实现：T.Pipelined(batch_sizes[bz])
   - 本项目方案：T.serial(max_k_iters) + if k < k_iters 条件判断
   - 参考：examples/grouped_gemm/example_grouped_gemm_fwd.py

建议：
- 先查阅本项目 examples/ 中的同类实现
- 确认 Ascend API 用法后再生成设计文档

是否继续生成设计文档？
```
