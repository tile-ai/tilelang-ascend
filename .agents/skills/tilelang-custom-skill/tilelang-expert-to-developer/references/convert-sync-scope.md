# 同步与作用域转换规则

## 概述

Developer 模式和 Expert 模式在同步控制和执行作用域上的差异是最显著的。Developer 模式依赖编译器自动化，Expert 模式需要手动精确控制。

## 作用域（T.Scope）转换

### Expert 模式：显式 T.Scope

必须用 `T.Scope("C")` 和 `T.Scope("V")` 显式标记 Cube 和 Vector 的执行区域。

```python
# Expert 模式
with T.Scope("C"):
    for k in T.serial(loop_k):
        T.copy(A[bx * block_M, k * K_L1], A_L1)
        T.copy(B[k * K_L1, by * block_N], B_L1)
        T.barrier_all()
        T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
        T.barrier_all()
    T.copy(C_L0, C[bx * block_M, by * block_N])
    T.set_cross_flag("FIX", 0)     # 通知 Vector 核

with T.Scope("V"):
    T.wait_cross_flag(0)             # 等待 Cube 核完成
    T.copy(C[...], c_ub)
    T.barrier_all()
    T.tile.add(c_ub, c_ub, d_ub)
    T.barrier_all()
    T.copy(c_ub, C[...])
```

### Developer 模式：无需 T.Scope

编译器通过 `AUTO_CV_COMBINE` 自动分析代码，将 Cube 操作和 Vector 操作分离到不同核上执行。

```python
# Developer 模式 — 无 T.Scope，无手动同步
for k in T.serial(loop_k):
    T.copy(A[bx * block_M, k * K_L1], A_L1)
    T.copy(B[k * K_L1, by * block_N], B_L1)
    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
T.copy(C_L0, workspace[bx * block_M, by * block_N])

# Vector 部分
T.copy(workspace[...], c_ub)
for i, j in T.Parallel(block_M, block_N):
    c_ub[i, j] = c_ub[i, j] + d_ub[i, j]
T.copy(c_ub, C[...])
```

## 同步转换规则

### Expert → Developer：移除同步

| Expert 模式 | Developer 模式 | 说明 |
|-------------|---------------|------|
| `T.barrier_all()` | （自动插入） | 由 AUTO_SYNC 处理 |
| `T.set_cross_flag("FIX", 0)` | （自动插入） | 由 AUTO_CV_SYNC 处理 |
| `T.wait_cross_flag(0)` | （自动插入） | 由 AUTO_CV_SYNC 处理 |
| `T.set_flag(...)` | （自动插入） | 由 AUTO_SYNC 处理 |
| `T.wait_flag(...)` | （自动插入） | 由 AUTO_SYNC 处理 |

**转换步骤：**
1. 删除所有 `T.barrier_all()`
2. 删除所有 `T.set_cross_flag` / `T.wait_cross_flag`
3. 删除所有 `T.set_flag` / `T.wait_flag`
4. 删除 `T.Scope("C")` 和 `T.Scope("V")`

## 纯 Vector 场景（无 Cube）

### Expert 模式
```python
with T.Scope("V"):
    T.copy(A[...], a_ub)
    T.copy(B[...], b_ub)
    T.barrier_all()
    T.tile.add(c_ub, a_ub, b_ub)
    T.barrier_all()
    T.copy(c_ub, C[...])
```

### Developer 模式
```python
# 无需 T.Scope，无需 barrier
T.copy(A[...], a_ub)
T.copy(B[...], b_ub)
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = a_ub[i, j] + b_ub[i, j]
T.copy(c_ub, C[...])
```

## 纯 Cube 场景（GEMM）

### Expert 模式
```python
with T.Scope("C"):
    for k in T.serial(loop_k):
        T.copy(A[...], A_L1)
        T.copy(B[...], B_L1)
        T.barrier_all()
        T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
        T.barrier_all()
    T.copy(C_L0, C[...])
```

### Developer 模式
```python
# 无需 T.Scope，无需 barrier
for k in T.serial(loop_k):
    T.copy(A[...], A_L1)
    T.copy(B[...], B_L1)
    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
T.copy(C_L0, C[...])
```

## 核间数据交换

Cube 核和 Vector 核只能通过 **Global Memory / L2 Cache** 交换数据。

### Expert 模式

需要显式通过 GM 中转 + 手动核间同步：

```python
with T.Scope("C"):
    T.copy(C_L0, C[bx * block_M, by * block_N])  # L0C → GM
    T.set_cross_flag("FIX", 0)

with T.Scope("V"):
    T.wait_cross_flag(0)
    T.copy(C[bx * block_M + vid * ..., by * block_N], c_ub)  # GM → UB
```

### Developer 模式

编译器自动处理同步，但仍需通过 workspace tensor 中转：

```python
T.copy(C_L0, workspace[bx * block_M, by * block_N])  # 编译器自动处理同步
T.copy(workspace[...], c_ub)
```

## pass_configs 要求

Developer 模式的自动同步需要开启以下开关：

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 Cube/Vector 分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,          # 自动核内同步
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,       # 自动核间同步
}
```

## 注意事项

1. **流水线场景特殊**：`T.Pipelined` 中的同步由流水线调度器管理，转换时需特别注意
2. **workspace 必要性**：Cube→Vector 数据交换始终需要 GM/workspace 中转，两种模式皆然
3. **MEMORY_PLANNING**：开启后编译器自动管理 buffer 地址，无需 `T.annotate_address`