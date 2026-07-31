# 关键编码规范

## 目录

- [1. Buffer 分配](#1-buffer-分配)
- [2. 数据搬运索引](#2-数据搬运索引)
  - [2.1 T.copy 多维切片的硬件限制](#21-tcopy-多维切片的硬件限制)
- [3. 同步](#3-同步)
- [4. 广播](#4-广播)
- [5. 测试模板](#5-测试模板)
- [6. 非连续数据维度归一化](#6-非连续数据维度归一化消除-host-侧-reshape-物理拷贝)
- [7. Dtype 性能特化](#7-dtype-性能特化)

---

## 1. Buffer 分配

```python
# VEC_NUM = 2，每个 vector 核处理 block_M // VEC_NUM 行
a_ub = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)
```

Developer 模式下：
```python
# Vector 核 buffer（编译器映射到 UB）
packed_ub = T.alloc_shared([block_M // VEC_NUM, block_N], dtype)

# Cube 核 buffer（编译器映射到 L1/L0）
A_L1 = T.alloc_shared([block_M, block_K], dtype)
B_L1 = T.alloc_shared([block_N, block_K], dtype)
C_L0 = T.alloc_fragment([block_M, block_N], accum_dtype)
```

## 2. 数据搬运索引

```python
# 标准索引模式（纯 Vector 算子）
row_start = bx * block_M + vid * block_M // VEC_NUM
T.copy(A[row_start, by * block_N], a_ub)
T.copy(a_ub, B[row_start, by * block_N])
```

**✅ CV 融合场景 — Developer 模式（推荐，默认消除 workspace/vid）**：
```python
# T.Kernel(block_num, threads=2, is_npu=True) as (cid)  —— threads=2，无 vid 轴
for bi_i in range(BI):                       # 整程，无 vid 偏移
    T.copy(KV[..., idx[bi_i], ...], kv_ub)
    T.copy(kv_ub, kv_l1[bi_i, :])            # gather 直连片上 L1，无 workspace
...
T.copy(acc_s_l0c, acc_s_ub_)                 # L0C → shared 直连，无 GM 往返
```
前提链：`threads=2` → 消 vid → 消 workspace；完整映射表见 [mode-examples.md §6](../../tilelang-custom-skill/tilelang-programming-model-guide/references/mode-examples.md#6-cv-融合--推荐写法消除-workspace--vidthreads2)。

**⚠️ 回退写法 — workspace 索引一致性（Expert/混合或复杂场景）**：
```python
VEC_NUM = 2
block_N_2 = block_N // VEC_NUM

for row in T.serial(block_N_2):
    actual_row = bn * block_N + vid * block_N_2 + row  # 关键索引
    
    # 读数据和写 workspace 都必须用 actual_row
    T.copy(B_packed[actual_row, chunk_offset], packed_ub)  # ✓
    # ... 处理 ...
    T.copy(output_ub, workspace[actual_row, chunk_offset * 2])  # ✓（必须一致）

# Cube 核读取完整 block_N（不涉及 vid）
T.copy(workspace[bn * block_N, k_offset], B_L1)  # 完整 block_N
```

**易错点（仅回退写法）**：workspace 写入时忘记使用 `actual_row`，导致数据错乱。

### 2.1 T.copy 多维切片的硬件限制

`T.copy` 的 GM↔UB / GM↔L1 / L0C→GM 搬运底层使用 AscendC 的 `DataCopyPad` 硬件指令。该指令的搬运模型是**每行内数据连续、行间可有 gap**（通过 `srcGap`/`dstGap` 参数控制），**不支持列方向（行内）strided access**——即每行内相邻元素之间不能有间隔。

**受限场景**：当切片的列维度（最后一个 `extent != 1` 的维度）**不是 buffer 的最内维**，且列维之后还有 `shape > 1` 的维度时，列方向数据不连续，会产生**静默数据错位**（不报错，结果错误）。

| 切片形式 | extents | 列维 | 列维是否最内维 | 是否支持 | 说明 |
|----------|---------|------|---------------|---------|------|
| `x[b, m:m+M, n:n+N]`（3D，buffer `[B,M,N]`）| `[1, M, N]` | N (dim 2) | 是 | ✅ | 列数据连续 |
| `x[b, m:m+M, n:n+N, :]`（4D 全取 K，buffer `[B,M,N,K]`）| `[1, M, N, K]` | K (dim 3) | 是 | ✅ | 列数据连续（K 维全取） |
| `x[b, 1, m:m+M, n:n+N]`（4D，buffer `[B,H,M,N]`，H=1）| `[1, 1, M, N]` | N (dim 3) | 是 | ✅ | 列数据连续 |
| `x[b, m:m+M, n:n+N, k:k+1]`（4D 单取 K，buffer `[B,M,N,K]`，K>1）| `[1, M, N, 1]` | N (dim 2) | **否**（dim 3 的 K>1） | ❌ | 列方向每个元素间隔 K 个位置，DataCopyPad 无法搬运 |

**判定方法**：检查 `T.copy` 的 GM 切片表达式，找出最后一个 `extent != 1` 的维度（即列维），确认它是 buffer 的最后一维（`shape.size() - 1`）。如果不是，且列维之后的维度有 `shape > 1`，则受限。

**规避方案**：

1. **host 侧 reshape 降维到 3D**：host 侧用 `permute` + `reshape` 把 4D+ 降维到 3D `(batch, M, N)`，使列维成为最内维。**⚠️ `reshape(-1)` 对非 contiguous 张量（如 `permute` 后）会触发物理拷贝，等价于 `.contiguous()`，属 [SKILL.md §3.2](../SKILL.md) 禁止行为 #2，不得使用。** 替代方案：用 stride 作为 JIT 编译期常量传入 kernel（详见 §6 方案 A）。
2. **调整 buffer 布局**：将 GM buffer 的维度排列调整为让列维（需要切片的维度）成为最内维。

> **根因**：`src/op/ascend.cc` 的 `find_active_dim_indices` 和 `compute_strideN` 对 4D+ 切片的维度识别有缺陷，且即使修复后，`DataCopyPad` 硬件指令本身也不支持列方向 strided access。这是 AscendC 硬件的固有限制，非 codegen bug。详见 `tilelang-api-best-practices/references/api-kernel-memory.md` §T.copy 多维切片的硬件限制。

## 3. 同步

```python
# Expert 模式：手动同步
with T.Scope("V"):
    T.copy(A[...], a_ub)
    T.barrier_all()
    T.tile.exp(a_ub, a_ub)
    T.barrier_all()
    T.copy(a_ub, B[...])

# Developer 模式 + 自动同步：无需手动 barrier
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

## 4. 广播

```python
# 归约结果 [M, 1] 广播到 [M, N]
max_ub = T.alloc_ub([block_M // VEC_NUM, 1], dtype)
max_2d_ub = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)
T.tile.broadcast(max_2d_ub, max_ub)
```

## 5. 测试模板

分层结构、main 分发器、`--level` 与分层标记见 [examples/code-skeleton.md](../examples/code-skeleton.md) 的 `test_{op}.py` 段（kernel 与测试分文件：kernel 在 `{op}.py`，测试在 `test_{op}.py` 并 `from {op} import {op}`）。单个用例的 golden 对比写法：

```python
# golden 对比（混合容差双门限，放在 try/except 内，按层打标记）
# get_precision/check_precision 见 examples/code-skeleton.md，阈值定义见
# tilelang-op-test-design/references/precision-standard.md（按 dtype，整型 0 误差精确匹配）
ref_output = torch.nn.functional.softmax(input_data, dim=-1)  # 或手写 golden
passed, ratio, max_abs = check_precision(output, ref_output, dtype)
# 通过条件：matched_ratio ≥ required_matched_ratio 且 max_abs_error ≤ max_abs_error_limit
# L0/L1（阻塞）：通过 → print("[PRECISION_PASS] ...")；失败 → print("[PRECISION_FAIL] ...") 且 ok=False
# Boundary（合法特殊值，非阻塞）：同上比精度，精度过 → "[BOUNDARY_PASS]"；不过或抛异常 → "[BOUNDARY_WARN]"
# L2（负向，非阻塞）：不比精度——非法输入正确抛异常 → "[BOUNDARY_PASS]"；静默接受 → "[BOUNDARY_WARN]"
```

### 5.1 NaN/Inf 位置敏感输入与比较

对支持 NaN/Inf 的浮点算子，特殊值用例必须能发现数据重排、索引和尾块覆盖错误。
全 NaN 输入在元素被搬到错误位置后仍然处处为 NaN，因此不能作为唯一门禁。

```python
def make_mixed_nan_input(shape, dtype, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(shape, dtype=torch.float32, generator=gen) * 2.0 - 1.0
    nan_mask = torch.rand(shape, dtype=torch.float32, generator=gen) < 0.5
    flat_mask = nan_mask.view(-1)
    if flat_mask.numel() > 0:
        flat_mask[0] = True   # 至少一个 NaN
    if flat_mask.numel() > 1:
        flat_mask[1] = False  # 至少一个有限值
    x[nan_mask] = float("nan")
    return x.to(dtype)        # CPU 上完成，随后只做一次 H2D


def check_special_masks(actual, golden):
    actual = actual.float()
    golden = golden.float()
    masks_match = (
        torch.equal(torch.isnan(actual), torch.isnan(golden))
        and torch.equal(torch.isposinf(actual), torch.isposinf(golden))
        and torch.equal(torch.isneginf(actual), torch.isneginf(golden))
    )
    finite = torch.isfinite(actual) & torch.isfinite(golden)
    return masks_match, finite
```

`check_precision()` 先调用 `check_special_masks()`；mask 不一致立即失败，mask 一致后
只在 `finite` 区域计算混合容差。普通有限值用例不需要注入特殊值。全 NaN/全 Inf
可以作为补充语义用例，但不能替代混合特殊值门禁。不要仅依赖
`torch.allclose(equal_nan=True)`，显式 mask 能分别验证 NaN、正 Inf 和负 Inf 的位置。

## 6. 非连续数据维度归一化（消除 host 侧 reshape 物理拷贝）

当算子需要对非 contiguous 张量做维度归一化时，`reshape(-1)` 会触发物理拷贝（属禁止行为）。以下两种替代方案按优先级选择。

### 方案 A：stride 作为 JIT 编译期常量传入 kernel（推荐，无额外 GM 搬运）

**约束**：
1. stride/shape 参数作为 `@tilelang.jit` 函数的 Python 参数传入（JIT 编译期常量），**不打包成 GM tensor**
2. kernel 内用 `T.alloc_var` 累加偏移，stride 直接用于地址计算（静态展开，最多约 6 个 batch 轴）
3. `T.copy` 不暴露 stride 参数：每次只搬一段物理连续的数据，非连续方向用逐行循环搬运
4. 若启用 `AUTO_CV_COMBINE`，检查生成代码中 `alloc_var` 的定义和使用是否在正确核；
   只有确认发生误分核时才关闭该配置

**使用方法**：

```
host 侧：
  1. 计算 row-major strides（纯 Python 整数运算）
  2. 把 stride 值作为 Python int 传入 @tilelang.jit 函数参数
  3. 仅当输入 storage 本来连续且目标 flatten 与 stride 兼容时，传入共享 storage 的
     1D view；否则保留原 rank，让 kernel 按原 shape/stride 计算合法连续 BufferRegion

kernel 内：
  1. T.alloc_var 声明累加变量（boff_s, bidx_s 等）
  2. 用 stride（JIT 编译期常量）+ var 累加计算 batch_offset
  3. 逐行 T.copy(x[src_offset], ub) — 每次搬一段物理连续的行
  4. T.tile 计算或转置
  5. 逐行 T.copy(ub, y[dst_offset]) — 写回
```

> **为什么不用 GM tensor 传 stride**：打包成 int32 GM tensor 传入 kernel 会增加一次 GM→UB 搬运（`T.copy(stride_buf, ub_s)`），且需要额外分配 GM 空间。作为 JIT 编译期常量传入则无此开销。

### 方案 B：连续轴组 reshape（仅 contiguous 张量可用）

**约束**：仅当输入张量 contiguous 时可用。contiguous 张量的 `reshape` 是零拷贝 view，不触发物理拷贝。

**使用方法**：当 perm 可以分解为"连续轴组交换"（`prefix + A + B → prefix + B + A`）时，host 侧直接 `reshape(batch, M, N)`（零拷贝），调用 3D kernel 完成 `(batch, M, N) → (batch, N, M)`。

> **判定**：`x.is_contiguous() == True` 是常见充分条件；为 False 时仍需检查目标
> shape 与 stride 的兼容性及是否共享 storage，不能直接断言一定复制。

> **适用条件**：不限于 transpose——任何算子需要对非 contiguous 张量做维度归一化时均可使用方案 A；输入本身 contiguous 且可分解为连续轴组交换时可用方案 B。

### 6.1 数据重排的正向实现配方

本节适用于 transpose、layout transform、blocked/unblocked layout、通用 gather/scatter
以及任何“输入输出元素不变、物理位置改变”的算子。按以下步骤生成实现，不要从禁止项
反推代码。

#### 步骤 1：识别最大连续 record

分别分析输入和输出 stride，找到两侧都保持物理连续的最大元素组，将它视为一个
`record`。record 可以是一条尾轴、多个相邻轴的乘积，也可以是 blocked layout 中
的一个内块。

```text
logical layout: prefix + group_A + group_B + record
target layout:  prefix + group_B + group_A + record
```

这里的 `prefix/group_A/group_B/record` 是结构角色，不代表固定 ndim、perm 或 shape。
host 只用 Python 整数计算各组乘积与 stride，不读取 NPU 数据。

#### 步骤 2：按成本从低到高选择路径

| 顺序 | 可利用的结构 | 正向实现 |
|------|--------------|----------|
| 1 | 输入输出都有同一连续 record | 用二维/成组 `T.copy` 一次聚合多条 record |
| 2 | 可零拷贝折叠成二维 tile | 块 DMA 到 UB → UB 内 tile 运算/重排 → 块 DMA 写回 |
| 3 | 一次变换无法覆盖，但可拆为相邻连续组变换 | 执行有限次块状 stage，每 stage 完整顺序搬运 |
| 4 | 无可利用连续结构 | JIT stride 通用 fallback，保证完整覆盖 |

选择路径时先计算 transaction 数量，而不是先调 core 数。增加 core 只能分摊已有任务，
不能消除数百万次短 DMA 或逐元素 GM 访问。

#### 步骤 3：实现 record-aware 聚合搬运

若一个 task 固定外层坐标和 `record_group`，让它一次处理
`valid_rows × record_len`，不要每条 record 单独发命令：

```python
record_ub = T.alloc_ub((block_rows, record_len), dtype)

for local_task in T.serial(single_core_load):
    task = cid * single_core_load + local_task
    if task < task_count:
        # 由 task 解码外层坐标、row_start 和 valid_rows
        T.copy(
            x[outer, row_start:row_start + valid_rows, record_group, 0:record_len],
            record_ub[0:valid_rows, 0:record_len],
        )
        T.copy(
            record_ub[0:valid_rows, 0:record_len],
            y[outer, record_group, row_start:row_start + valid_rows, 0:record_len],
        )
```

`block_rows` 由 UB 容量机械计算：

```python
block_rows = max(1, min(rows, ub_budget_bytes // (record_len * dtype_bytes)))
task_count = outer_count * record_group_count * ceildiv(rows, block_rows)
core_num = max(1, min(task_count, vector_core_count))
single_core_load = ceildiv(task_count, core_num)
```

这一骨架利用 DataCopy 的行间 gap 收集连续 record。应用到其他算子时，替换结构角色
和坐标解码即可，不要照抄固定四维签名。可从已验证实现
`examples/transpose/transpose.py::_kernel_4d_record_swap` 学习 `T.copy` 切片形式。

#### 步骤 4：没有直接 record 路径时，在 UB 内重排

GM 两侧仍然使用块 DMA，把不连续/标量工作限制在 UB：

```python
T.copy(x[input_tile_slice], input_ub)
local_reorder(output_ub, input_ub)  # tile primitive、cast 或 UB-local scalar fallback
T.copy(output_ub, y[output_tile_slice])
```

`local_reorder` 可以是 transpose、pack/unpack、lane 交换、索引选择或其他片上操作。
关键不变量是：GM→UB 和 UB→GM 为块搬运，标量循环不直接遍历 strided GM。

#### 步骤 5：复杂排列采用分阶段连续组变换

若目标布局无法一次转换，把它分解为若干相邻连续组的交换/旋转。每个 stage 都套用
步骤 3 或步骤 4，并更新当前 shape/轴顺序，直到得到目标布局。这会增加完整 GM pass，
但通常比海量 4B/8B 短 DMA 更稳定。

设计时记录 `stage_count`，估算完整 GM 流量约为：

```text
read_write_bytes = 2 * stage_count * numel * dtype_bytes
```

#### 步骤 6：用数字决定是否接受

对每条路径至少输出：

```text
task_count
estimated_dma_transactions
average_dma_bytes
gm_scalar_access_count
per_core_serial_tasks
```

经验性的诊断顺序：

- `estimated_dma_transactions` 很大：扩大每个 task 聚合的 record/row 数。
- `average_dma_bytes` 只有几个或几十字节：改用块状 stage。
- `gm_scalar_access_count` 接近 `numel`：改成块 DMA + UB-local reorder。
- transaction 已合理但单 core 任务多：最后再调整 tile 与 core_num。

生成后用 `get_kernel_source()` 确认 DataCopy 位于 GM↔UB，`GetValue/SetValue`
等标量操作只访问 UB，并用最大任务数/最差 dtype case 做 timeout 验证。

## 7. Dtype 性能特化

某些 dtype 在硬件指令中回退标量路径（无向量加速），可通过 dtype 变换使用硬件加速。**不限于下列具体 dtype**——判断准则是"目标 dtype 是否有硬件加速指令"，适用于任何符合此条件的 dtype。

### 约束

1. **零拷贝 reinterpret 仅限同宽 dtype**：`view` 不拷贝数据，但要求两个 dtype 字节数相同（如 bfloat16 ↔ int16 均 2 字节）
2. **kernel 内 cast 须用 `T.tile.cast`**：low2high / high2low 方向，元素数须 16 对齐
3. **禁止 host 侧 dtype 转换/拆分**：`.to(dtype)` 触发 `aclnnCast`、`.contiguous()` 触发 `aclnnCopy`、`torch.stack` 触发 `aclnnCat`——dtype 适配必须在 kernel 内完成

### 思路

**场景 A：目标 dtype 有同宽的硬件加速 dtype** → host 侧零拷贝 `view` reinterpret，用硬件加速 dtype 处理，输出再 `view` 回来

**场景 B：目标 dtype 在 kernel 内可 cast 到硬件支持 dtype** → kernel 内 `T.tile.cast` 到支持 dtype 处理，再 cast 回来（UB 需额外分配 cal_dtype buffer）

**场景 C：目标 dtype 超出硬件指令范围且算子是纯数据搬运** → 优先把数据当作
opaque record，用块/二维 `T.copy` 完成结构重排；无法直接搬运时，使用块 DMA
GM→UB、UB-local 标量重排、块 DMA UB→GM。不得退化成逐元素 strided GM 访问。

> **通用判断方法**：查看 `get_kernel_source()` 输出或 `src/tl_templates/ascend/common.h`，确认目标 dtype 是否走标量路径。如果走标量路径，寻找同宽的有硬件加速的 dtype 做 reinterpret，或 cast 到更宽的 dtype 做 kernel 内转换。

参考实现：bfloat16→int16 host 侧 `view` reinterpret、int8 kernel 内 `T.tile.cast`
到 float16；int64 优先 record-aware DMA，通用 fallback 可接受 `T.tile.transpose`
lowering 成 UB-local scalar，但必须以块 DMA 包住并通过最大 case 超时门禁。详见
[tilelang-api-best-practices/references/api-compute.md §4.10](../../tilelang-custom-skill/tilelang-api-best-practices/references/api-compute.md#ttiletranspose-的-dtype-支持与标量回退)。

### int64 数据重排补充

单纯“逐行 `T.copy`”只能搬连续行，不能完成任意二维转置。候选顺序是：

1. 输入/输出共享连续 suffix record 时，用二维/成组 DMA 直接完成 record 重排。
2. 其他情况用块 DMA 搬入 UB，在 UB 内做局部标量重排，再块 DMA 写回。
3. `T.tile.transpose` 对 int64 即使 lowering 为 `SetValue/GetValue`，只要标量访问
   局限于 UB，仍可能比逐元素 strided GM 更好；须检查生成源码并实测最大 case。

拆成两个 int32 lane 仅是可选实验；lane 提取、交织写回和 UB planner 未经端到端
精度验证时不得采用，也不得在 host 侧拆分/拼回。
