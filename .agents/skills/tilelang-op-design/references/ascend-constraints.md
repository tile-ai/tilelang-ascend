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
| **L0C 容量上限** | A2/A3 设备 L0C = 128KB | `block_M × block_N × sizeof(accum) > 128KB` 导致 segfault | 设计 block 时满足 `block_M × block_N ≤ 16384`（float32 accum） |
| **T.copy 不支持列方向 strided 切片** | AscendC `DataCopyPad` 硬件指令只支持行间 gap、不支持行内（列方向）strided access。当切片的列维（最后一个 extent≠1 的维度）不是 buffer 最内维且之后有 shape>1 的维度时，产生**静默数据错位** | 4D+ 切片如 `x[b, m:m+M, n:n+N, k:k+1]`（K>1）无法正确搬运 | 用 3D 切片 + host 侧 `permute`+`reshape` 降维到 3D `(batch, M, N)` 使列维成为最内维（仅 contiguous 张量可用）；详见 `tilelang-api-best-practices/references/api-kernel-memory.md` §T.copy 多维切片的硬件限制 |

## 2. 强制检测规则

在设计文档生成前，**必须**执行以下检测：

| 检测项 | 触发条件 | 处理方式 |
|--------|----------|----------|
| 三维 Kernel | 参考实现包含 `T.Kernel(..., batch_count)` 或 3 个维度参数 | **立即警告**，提出 `block_metadata` 方案 |
| threads 参数 | 参考实现 threads > 2 | **立即警告**，建议 threads=2 或移除 |
| 动态循环边界 | 循环边界依赖 tensor 值 | **立即警告**，提出静态边界 + 条件判断方案 |
| GPU 专用 API | CUDA 相关 API（如 `T.gemm` 通用版） | **立即警告**，查阅本项目确认 Ascend API |
| GEMM 非整除风险 | `M` 或 `N` 不被 block size 整除（即 `M % block_M ≠ 0` 或 `N % block_N ≠ 0`） | **立即警告**，要求 design 中明确使用 `T.ceildiv(M, block_M)` 或 `T.ceildiv(N, block_N)` |
| L0C 溢出风险 | block_M × block_N × sizeof(accum_dtype) > 131072 (128KB) | **立即警告**，建议减小 block 或拆分 |
| T.copy 列方向 strided 切片 | design.md 中出现 4D+ 切片搬运，且切片的列维（最后一个 extent≠1 的维度）不是 buffer 最内维 | **立即警告**，要求改为 3D 切片 + host 侧 reshape 降维方案；详见 `tilelang-api-best-practices/references/api-kernel-memory.md` §T.copy 多维切片的硬件限制 |
| Host 侧 `reshape` 隐式拷贝 | design.md 中 host 侧对 `permute`/`transpose`/`movedim` 后（非 contiguous）的张量做 `reshape`（尤其是 `reshape(-1)`） | **立即警告**，指出 `reshape` 对非 contiguous 张量等价于 `.contiguous()`，属 §3.2 禁止行为；要求改用 stride buffer 方案（见下方「非连续数据归一化设计」） |
| Host 侧 `torch.nn.functional.*` / `torch.cat` 等隐式 aclnn 调用 | design.md 中 host 侧出现 `torch.nn.functional.pad`/`cat`/`interpolate`、`torch.cat`/`stack`、`.to(dtype)` dtype 转换、`.clone()` 等操作 | **立即警告**，指出这些操作在 NPU tensor 上会触发 aclnn 调用，cann-bench 评测环境可能裁剪 aclnn 导致运行时失败；属 §3.2 禁止行为 #5；要求改为在 kernel 内用 `T.tile.cast`/`T.copy`+`pad_value` 等完成 |
| 输出侧切片 + reshape 隐式 contiguous | design.md 中 kernel 调用后对输出张量做切片（如 `y[:,:,:,:S]`）再 `reshape` | **立即警告**，指出切片后 tensor 非 contiguous，reshape 会隐式 `.contiguous()` → `aclnnCopy`；属 §3.2 禁止行为 #5；要求改为让 kernel 直接输出到与原始 shape 一致的 buffer（`T.copy`+`pad_value` 处理尾块），host 侧仅做纯 view reshape |

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

---

## 4. 非连续数据归一化设计

当算子输入数据在 GM 中非连续（如 `permute`/`transpose`/`movedim` 后），需要做维度归一化才能交给 kernel 处理。有以下方案，选错会导致严重性能问题或 aclnn 调用。

### 方案 A：host 侧 reshape 降维（仅 contiguous 张量可用）

host 侧用 `reshape` 降维到连续形态。

**约束**：`reshape` 仅在张量 contiguous 时是零拷贝 view。对非 contiguous 张量，`reshape`（尤其是 `reshape(-1)`）会触发物理拷贝，等价于 `.contiguous()`，属 §3.2 禁止行为。**仅当输入已 contiguous 时可用此方案。**

### 方案 B：stride 作为 JIT 编译期常量传入 kernel（推荐）

host 侧只算 stride 参数（纯 Python 整数运算），作为 `@tilelang.jit` 函数的 Python 参数传入（JIT 编译期常量）。kernel 内用 `T.alloc_var` 累加偏移 + 逐行 `T.copy` 搬运，完全消除 host 侧物理拷贝。

**适用条件**：不限于 transpose——任何算子需要对非 contiguous 张量做维度归一化或数据重排时均可使用。典型场景包括 N-D permute、gather/scatter 后的重排、非连续切片的批量处理等。

**设计要点**：
- stride/shape 参数作为 JIT 编译期常量传入 kernel（**不打包成 GM tensor**，避免额外 GM→UB 搬运）
- kernel 接受 1D flat GM tensor + stride 参数（Python int）
- `T.alloc_var` 声明累加变量，用 stride 常量累加计算 batch_offset
- 逐行 `T.copy` 搬运（每次只搬一段物理连续的数据）
- 纯 Vector 算子启用 `AUTO_CV_COMBINE` 时检查生成代码中 `alloc_var` 的定义/使用核归属；
  只有确认发生误分核时才关闭该配置

### 设计自检

design.md 中如果对非 contiguous 张量做 `reshape` 降维，必须回答：
- 输入张量在 `reshape` 之前是否 contiguous？（`x.is_contiguous()`）
- 如果非 contiguous，`reshape` 会触发物理拷贝——是否已改用方案 B（stride 作为 JIT 编译期常量传入 kernel）？
- 如果答案仍是 `reshape` → **设计违规**，必须改用方案 B
