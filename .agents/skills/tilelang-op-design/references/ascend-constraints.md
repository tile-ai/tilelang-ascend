# 技术约束清单（必须遵守）

本项目为 TileLang-Ascend（华为昇腾 NPU），与 GPU 版 TileLang 有显著差异。
**外部参考实现不可直接使用，必须转换为 Ascend 兼容方案。**

## 目录

- [1. 本项目已知限制](#1-本项目已知限制)
- [2. 强制检测规则](#2-强制检测规则)
- [3. 警告输出格式](#3-警告输出格式)
- [4. 算子 kernel 划分原则（强制规则）](#4-算子-kernel-划分原则强制规则)
- [5. Host 侧 Buffer 操作约束（设计阶段必须遵守）](#5-host-侧-buffer-操作约束设计阶段必须遵守)
- [6. 非连续数据归一化设计](#6-非连续数据归一化设计)

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
| Host 侧 `reshape` 隐式拷贝 | design.md 中 host 侧对 `permute`/`transpose`/`movedim` 后（非 contiguous）的张量做 `reshape`（尤其是 `reshape(-1)`） | **立即警告**，指出 `reshape` 对非 contiguous 张量等价于 `.contiguous()`，属 §5 禁止行为；要求改用 stride buffer 方案（见 §6「非连续数据归一化设计」） |
| Host 侧 `torch.nn.functional.*` / `torch.cat` 等隐式 aclnn 调用 | design.md 中 host 侧出现 `torch.nn.functional.pad`/`cat`/`interpolate`、`torch.cat`/`stack`、`.to(dtype)` dtype 转换、`.clone()` 等操作 | **立即警告**，指出这些操作在 NPU tensor 上会触发 aclnn 调用，cann-bench 评测环境可能裁剪 aclnn 导致运行时失败；属 §5 禁止行为 #5；要求改为在 kernel 内用 `T.tile.cast`/`T.copy`+`pad_value` 等完成 |
| 输出侧切片 + reshape 隐式 contiguous | design.md 中 kernel 调用后对输出张量做切片（如 `y[:,:,:,:S]`）再 `reshape` | **立即警告**，指出切片后 tensor 非 contiguous，reshape 会隐式 `.contiguous()` → `aclnnCopy`；属 §5 禁止行为 #5；要求改为让 kernel 直接输出到与原始 shape 一致的 buffer（`T.copy`+`pad_value` 处理尾块），host 侧仅做纯 view reshape |

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

## 4. 算子 kernel 划分原则（强制规则）⭐

多 kernel 方案必须保证支持域完整：先提供覆盖全部声明输入域的通用路径，再按 dtype、
shape、对齐性或输入拓扑增加有限快路径。具体取值可以用于性能特化；只要未命中的输入
可靠回落到通用路径，就不要求特化条件本身"封闭可穷举"。

设计时记录每条路径的适用谓词、fallback、语义等价依据和验证用例。禁止仅枚举若干
ndim、perm 或 shape 而没有 fallback，也禁止新增一个取值就让算子变成"不支持"。

生成 design.md 前必须写出以下审计；任一项无法回答时，不得进入实现阶段：

```text
[DISPATCH-COVERAGE]
supported_domain: <design 声明的 shape/dtype/attr 范围>
generic_fallback: <kernel/path；没有则写 none>
specializations:
  - predicate: <纯 metadata 可判定条件>
    fallback_on_miss: <path>
    equivalence_evidence: <索引映射或参考实现>
unsupported_inputs: <必须与 supported_domain 不冲突>
result: pass/fail
```

判定规则：

1. `generic_fallback == none` 时，所有分支谓词的并集必须覆盖 `supported_domain`，否则 fail。
2. 有 fallback 时允许具体 dtype/shape/perm 快路径，但每条未命中输入必须落入 fallback。
3. 分派只能读取 shape、stride、dtype、attr 等 metadata，不能读取或改动 tensor 数据。
4. 每条 specialization 和 fallback 都必须在验证计划中至少有一个命中 case。

### 4.1 基于输入结构特征的多路径分派（性能设计指导）

> 当算子输入具有多种结构特征（如 perm 拓扑类型、数据连续性、对齐性等）时，可设计多条快路径按结构特征分派。这不同于 §4 的"通用+特化"——特化是按 dtype 等封闭条件划分 kernel，多路径分派是按**输入的结构拓扑**选择不同的搬运/计算策略。

**约束**：
- 分派谓词必须可判定，且不能读取或改动 tensor 数据
- 必须保留**通用 fallback**路径：无法归入任何快路径的输入走通用实现
- 快路径的判断逻辑在 host 侧完成（纯 Python 整数/元数据运算），不触碰数据

**思路**：
1. 分析算子输入可能的结构形态，找出可利用的结构特征（如"哪些轴连续""最内轴是否移动"等）
2. 每种结构形态设计一条快路径——核心是利用结构特征减少搬运次数或避免非连续访问
3. 保留通用 fallback 处理无法归类的输入

> **⚠️ 避免示例覆盖**：上述思路适用于所有输入具有结构差异的算子，**不限于 transpose 的 perm 拓扑**。判断准则是有没有可利用的结构特征让某些输入走更快的路径，有就分派，没有就统一处理。

参考思路：host 侧入口按输入结构特征（如 perm 拓扑）分派多条快路径，通用 fallback 兜底

### 4.2 数据重排的性能可行性（强制）

涉及物理布局变化时，必须读取
[coding-conventions.md §6.1](../../tilelang-op-develop/references/coding-conventions.md#61-数据重排的正向实现配方)，
为每条结构路径和最大/关键用例完成 GM/DMA/地址解码/并行度成本验收。具体 record、
UB-local reorder 和 dtype fallback 配方以该 reference 为准，不在主流程重复。

design.md 必须为每条路径输出：

```text
[REORDER-COST]
path: <name>
gm_passes: <完整读写次数>
dma_transactions: <估算值>
average_dma_bytes: <估算值>
gm_scalar_accesses: <估算值>
address_div_mod_per_element: <估算值>
active_cores / serial_tasks_per_core: <估算值>
largest_case_timeout_gate: <case + timeout>
result: pass/fail
```

若大张量主路径的 GM 标量访问接近 numel、短 DMA 达到数十万次，或没有最大 case
timeout 门禁，结果必须为 fail，并按 reference 重新选择候选实现。

---

## 5. Host 侧 Buffer 操作约束（设计阶段必须遵守）⭐

> **⚠️ 核心原则：host 侧禁止改动 NPU 张量 buffer 内的真实内容，禁止触发任何 aclnn 调用**
>
> 算子的所有核心计算逻辑（数据搬运、数学运算、归约、维度重排、padding 等）必须在 `@tilelang.jit` 装饰的 kernel 函数内部完成。**host 侧（kernel 外的 Python 代码）对 NPU 侧张量 buffer 内的真实内容（数据值、物理排布、数据指针）一律不得改动**——只允许做只改 stride/shape 元数据的视图操作。**约束范围覆盖 kernel 调用前（输入预处理）和 kernel 调用后（输出后处理）的完整 host 代码路径。**
>
> **违规示例**（都属于"改动 buffer 真实内容"或"触发 aclnn"，一律禁止）：
> - `.contiguous()` / `.reshape(...).contiguous()` / `.permute(...).contiguous()` / `.transpose(...).contiguous()` —— 触发真实数据拷贝/重排
> - host 侧 padding：`x_padded = torch.zeros(...); x_padded[:, :M] = x; x = x_padded` —— 创建新 buffer + 写入数据 + 顶替原输入
> - **`torch.nn.functional.pad(x, ...)` / `torch.cat` / `torch.stack`** —— 隐蔽违规：表面是函数调用，实质是创建新 buffer + 数据拷贝，在 NPU 上会调用 `aclnnPad`/`aclnnCat` 等 aclnn 算子。等同禁止行为"用新 buffer 作弊"
> - 直接改写 buffer 内容：`x[:] = ...`、`x.add_(1)`、`torch.mul(x, 2, out=x)`
> - 用另一个经过 host 计算或物理化的 tensor 替代原输入后传入 kernel
> - `reshape` 无法保持原 storage/stride 时发生的隐式物理化；不能仅凭
>   `is_contiguous()` 判断，需证明目标 shape 与当前 stride 兼容，或比较操作前后的
>   storage/data pointer
> - **输出侧切片 + reshape**（隐蔽违规，group_norm 案例）：kernel 输出后对输出张量做切片（如 `y[:, :, :, :S]`）使 tensor 变为非 contiguous，随后 `reshape` 会隐式调用 `.contiguous()` → `aclnnCopy`。**约束范围不仅限于输入侧，kernel 调用后的输出后处理同样适用**。解法：让 kernel 直接输出到与原始 shape 一致的 buffer（通过 `T.copy` 的 `pad_value` 处理尾块），host 侧无需切片+reshape
>
> **允许**的 host 侧操作：经证明只改 stride/shape 元数据、不触碰 storage 的
> `reshape`/`view`/`transpose`/`permute`/`expand`，以及数据准备、kernel 调用和结果验证。
>
> **判定准则**：host 侧任何会改变 NPU 张量「数据指针」或「物理存储内容/排布」的操作均禁止；只改 metadata（stride/shape）的允许。拿不准时，一律放入 kernel。
>
> **`reshape` 语义判定**：`is_contiguous()` 为 True 是常见充分条件，不是必要条件。
> 对非 contiguous 输入先证明该目标 reshape 可返回共享 storage 的 view；不能证明时，
> 使用 stride-aware kernel 路径。
>
> **非整除处理**：输入、输出 GM 两侧必须显式使用 valid extent/BufferRegion；前端按
> 动态切片裁剪搬运，但不会替设计补齐错误的完整 tile 区域。无需 host padding，
> design.md 中不得出现 host padding + crop。
>
> **aclnn 依赖约束**（评测环境兼容性）：cann-bench 评测环境中 aclnn 编译产物可能被裁剪，host 侧任何会隐式触发 aclnn 调用的操作都会导致运行时失败。以下操作在 NPU tensor 上会触发 aclnn，一律禁止：
> - `torch.nn.functional.pad` / `torch.nn.functional.interpolate` / `torch.nn.functional.cat` 等 `torch.nn.functional.*` 计算 API
> - `torch.cat` / `torch.stack` 等会创建并填充新 buffer 的操作；`split`/切片本身可能
>   只是 view，需审计其后是否发生物理化
> - 对非 contiguous 张量的 `reshape`（隐式 `.contiguous()` → `aclnnCopy`），**包括输出侧切片后的 reshape**
> - `.to(another_dtype)` dtype 转换（触发 `aclnnCast`）；如需 dtype 转换应在 kernel 内用 `T.tile.cast` 完成
> - `.clone()` / `.copy_()` 等显式拷贝
>
> **判定方法**：逐项证明 host tensor 操作只改变 metadata；不能从 API 名称或
> `is_contiguous()` 单一布尔值直接下结论。
>
> **设计自检**（Phase 4 质量自检时核对）：审计 kernel 前后完整 host 路径。下游
> [tilelang-op-develop SKILL.md §3](../../tilelang-op-develop/SKILL.md) 会再次校验，
> 违规设计在 Stage 2 返回 `[DESIGN_ERROR]`。

审计必须记录：

```text
[HOST-METADATA-AUDIT]
operation: <host tensor operation>
input_stride -> output_stride: <...>
shares_storage / same_data_ptr: true/false/unknown
aclnn_or_physical_copy: true/false/unknown
result: allow/reject
```

任何 `unknown` 按 reject 处理；改为 kernel 内实现或用可证明的 metadata-only 路径。

---

## 6. 非连续数据归一化设计

当算子输入数据在 GM 中非连续（如 `permute`/`transpose`/`movedim` 后），需要做维度归一化才能交给 kernel 处理。有以下方案，选错会导致严重性能问题或 aclnn 调用。

### 方案 A：host 侧 reshape 降维（仅 contiguous 张量可用）

host 侧用 `reshape` 降维到连续形态。

**约束**：`reshape` 仅在张量 contiguous 时是零拷贝 view。对非 contiguous 张量，`reshape`（尤其是 `reshape(-1)`）会触发物理拷贝，等价于 `.contiguous()`，属 §5 禁止行为。**仅当输入已 contiguous 时可用此方案。**

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
