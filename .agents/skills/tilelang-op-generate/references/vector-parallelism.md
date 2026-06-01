# V 核并行化编码规范

Ascend NPU C:V = 1:2，默认两个 V 核执行相同工作。

> **Developer 模式（推荐）默认用 `threads=2` 让编译器自动并行两个 V 核，不手动写 `vid`**——`T.Kernel(block_num, threads=2, is_npu=True) as (cid)`，循环整程、无 `vid` 偏移、CV 交互片上直连。详见 [mode-examples.md §6](../../tilelang-custom-skill/tilelang-expert-to-developer/references/mode-examples.md#6-cv-融合--推荐写法消除-workspace--vidthreads2)。
> 本文以下手动 `vid` 切分写法用于 **Expert/混合或复杂场景回退**；正确使用 `vid` 可让两个 V 核分担任务。

## 目录

- [1. 按行切分](#1-按行切分)
- [2. 中间 buffer 索引一致性](#2-中间-buffer-索引一致性)
- [3. 模式三：CV 融合中的 V 核并行化](#3-模式三cv-融合中的-v-核并行化)

---

## 1. 按行切分

```python
VEC_NUM = 2
block_M_2 = block_M // VEC_NUM

with T.Kernel(grid_size, is_npu=True) as (cid, vid):
    row_start = cid * block_M + vid * block_M_2
    
    # Buffer 分配：只需分配 V 核负责的行数
    data_ub = T.alloc_shared((block_M_2, block_N), dtype)
    
    # 读入数据
    T.copy(A[row_start, by * block_N], data_ub)
    
    # 计算
    ...
    
    # 写出数据（索引必须与读一致）
    T.copy(data_ub, B[row_start, by * block_N])
```

## 2. 中间 buffer 索引一致性

当 V 核读写中间 buffer（workspace、临时 buffer）时，必须保持索引一致：

```python
# 错误：读写索引不一致
for row in T.serial(block_N_2):
    actual_row = bn * block_N + vid * block_N_2 + row
    T.copy(src[actual_row, ...], temp_ub)
    T.copy(temp_ub, dst[bn * block_N + row, ...])  # ❌ 索引不一致

# 正确：读写索引一致
for row in T.serial(block_N_2):
    actual_row = bn * block_N + vid * block_N_2 + row
    T.copy(src[actual_row, ...], temp_ub)
    T.copy(temp_ub, dst[actual_row, ...])  # ✓ 索引一致
```

## 3. 模式三：CV 融合中的 V 核并行化

> **Developer 模式（推荐）默认消除 workspace/vid**：用 `T.Kernel(block_num, threads=2, is_npu=True) as (cid)` 让编译器自动并行两个 V 核，循环用整程 `range(BI)`、无 `vid` 偏移，Cube↔Vector 片上直连、无 workspace。前提链与映射表见 [mode-examples.md §6](../../tilelang-custom-skill/tilelang-expert-to-developer/references/mode-examples.md#6-cv-融合--推荐写法消除-workspace--vidthreads2)。
> 下面是**回退写法**（手动 vid + workspace），仅用于复杂同步/多版本流水场景。

CV 融合算子中（回退写法），V 核负责预处理，Cube 核负责 GEMM：

```python
VEC_NUM = 2
block_N_2 = block_N // VEC_NUM

# Vector 核部分：使用 vid 分配任务
for row in T.serial(block_N_2):
    actual_row = bn * block_N + vid * block_N_2 + row
    T.copy(B_packed[actual_row, ...], ...)
    T.copy(..., workspace[actual_row, ...])

# Cube 核部分：读取完整 block_N（不涉及 vid）
T.copy(workspace[bn * block_N, ...], B_L1)
T.gemm_v0(A_L1, B_L1, C_L0, ...)
```
