# T.copy

## 1. 功能说明

在 Ascend 平台的不同内存层级之间搬运数据。根据 src 和 dst 的 scope 自动选择对应的搬运路径，支持 GM → L1、GM ↔ UB、L1 → L0A/L0B、L0C → GM/UB、UB → L1、UB → UB 共 9 条路径。

## 2. 函数原型

### 2.1 函数定义

```python
def copy(
    src: Buffer | BufferLoad | BufferRegion,
    dst: Buffer | BufferLoad,
    enable_relu: bool = False,
    transpose: bool | None = False,
    pad_value: float | int | PrimExpr | None = None,
    tmp: Buffer | BufferLoad | None = None,
    unit_flag: int | None = None,
    real_k: int | PrimExpr | None = None,
    real_n: int | PrimExpr | None = None,
):
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| src | 输入 | 源 buffer 或其切片 | 张量（tensor） | 必填 |
| dst | 输出 | 目标 buffer 或其切片 | 张量（tensor） | 必填 |
| enable_relu | 输入 | 是否在搬运时执行 ReLU | 布尔 | 可选（默认 `False`） |
| transpose | 输入 | 是否转置 | 布尔 | 可选（默认 `False`） |
| pad_value | 输入 | UB 未使用区域的填充值 | 浮点数 / 整数 / PrimExpr | 可选（默认 `None`，填 0） |
| tmp | 输入 | UB→L1 的 ND→NZ 格式转换临时缓冲区 | 张量（tensor） | 可选（默认 `None`） |
| unit_flag | 输入 | L0C→GM fixpipe 的 unitFlag（`0b10` 累加 / `0b11` flush） | 整数 | 可选（默认 `None`） |
| real_k | 输入 | L1→L0A/L0B 运行时 K 轴收缩长度 | 整数 / PrimExpr | 可选（默认 `None`） |
| real_n | 输入 | L1→L0B 运行时 N 轴输出宽度 | 整数 / PrimExpr | 可选（默认 `None`） |

> **类型说明**：
> - **tensor**：通过 `T.alloc_shared`、`T.alloc_L1`、`T.alloc_ub` 等分配的缓冲区（Buffer），或其切片（BufferRegion / BufferLoad）

> **内存层级说明**：
> - src 和 dst 的 scope 决定搬运路径，编译器自动选择对应的硬件指令
> - 数据流：`GM → L1`、`GM ↔ UB`、`L1 → L0A/L0B`、`L0C → GM/UB`、`UB → L1`、`UB → UB`

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | src | dst |
|------|:---:|:---:|
| Ascend A2 / A3 | float16, float32, bfloat16, int8, int16, int32 | float16, float32, bfloat16, int8, int16, int32 |

#### 2.3.2 Shape 支持

- 支持 1D 和 2D
- L1 → L0A/L0B 须为 2D 矩阵分形

#### 2.3.3 Copy 路径说明

**Cube（矩阵乘）流程**：

| 路径 | src scope | dst scope | dtype | 特殊参数 |
|------|-----------|-----------|-------|---------|
| GM → L1 | `global` | `shared.l1` | float16, float32, bfloat16, int8, int16, int32 | — |
| L1 → L0A | `shared.l1` | `wmma.matrix_a` | float16, bfloat16 | transpose, real_k |
| L1 → L0B | `shared.l1` | `wmma.matrix_b` | float16, bfloat16 | transpose, real_k, real_n |
| L0C → GM | `wmma.accumulator` | `global` | float32 | enable_relu, unit_flag |
| L0C → UB | `wmma.accumulator` | `shared.ub` | float16, bfloat16 | enable_relu |

**Vector 流程**：

| 路径 | src scope | dst scope | dtype | 特殊参数 |
|------|-----------|-----------|-------|---------|
| GM → UB | `global` | `shared.ub` | float16, float32, bfloat16, int8, int16, int32 | pad_value |
| UB → GM | `shared.ub` | `global` | float16, float32, bfloat16, int8, int16, int32 | — |
| UB → UB | `shared.ub` | `shared.ub` | float16, float32, bfloat16, int8, int16, int32 | — |

**跨 CV 搬运**：

| 路径 | src scope | dst scope | dtype | 特殊参数 |
|------|-----------|-----------|-------|---------|
| UB → L1 | `shared.ub` | `shared.l1` | float16, bfloat16 | tmp |

> 不支持跨级搬运：GM → L0A/L0B/L0C 须通过 GM → L1 → L0 两步完成（硬件约束）

#### 2.3.4 分形尺寸

L1 → L0A/L0B 以分形矩阵为最小搬运单元，各 dtype 对应的分形尺寸如下：

| dtype | L0A 分形 | L0B 分形 | K 最小值 |
|-------|---------|---------|:-------:|
| float16 / bfloat16（2B） | 16×16 | 16×16 | 16 |

> M、N 须 ≥ 16（硬件约束）

### 2.4 约束条件

1. 必须在 `T.Kernel` 作用域内调用
2. `transpose` 仅对 L1 → L0A/L0B 路径生效
3. `enable_relu` 仅对 L0C → GM/L0C → UB 路径生效
4. `unit_flag` 仅对 L0C → GM 路径生效（`0b10` 累加 / `0b11` flush）
5. `real_k` / `real_n` 仅对 L1 → L0A/L0B 路径生效
6. `pad_value` 仅对 GM → UB 路径生效
7. UB → UB 跨类型 cast 不支持的组合：bfloat16 ↔ float16/int8/int16、float32 → int8、int8 → float32/bfloat16/int16/int32、int16 → bfloat16/int8/int32、int32 → bfloat16/int8。仅 ascendc 支持：float16/float32 → int8/int16/int32。仅 pto 支持：bfloat16 → int32
8. 不支持跨级搬运：GM → L0A/L0B/L0C 须通过 GM → L1 → L0 两步完成（硬件约束）
9. GM ↔ UB 的列维度须为编译期常量（硬件约束）
10. 操作数地址需 32 字节对齐（硬件约束）
11. L1 → L0A/L0B 的 M、N 须 ≥ 16，K 须满足分形尺寸要求（硬件约束）
12. GM → L1 多段 copy 时，仅首段自动清零整个 L1 tile，后续段不清零（硬件约束）
13. L1 → L0A/L0B 使用 sliced buffer 时，按切片的有效行数搬运，非物理 buffer 行数
14. 跨 CV 搬运（UB → L1、L0C → UB）的 src 和 dst shape 须 rank 相等，最多一个维度不同且须为 2 倍关系，其余维度须完全一致
15. GM → UB 多段（strided）copy 中，每段有效长度和段间距均须为 32 字节整倍数；有效长度非整倍数时自动补 pad_value，段间距非整倍数会静默截断导致数据错位（硬件约束）

## 3. 示例代码

**示例 1：GM → L1 搬运（GEMM 输入）**

```python
block_M, block_K = 128, 128
dtype = "float16"

A_L1 = T.alloc_L1((block_M, block_K), dtype)
T.copy(A[bx * block_M, 0], A_L1)
```

**示例 2：GM → UB 搬运（带 pad_value）**

```python
block_M, block_N = 128, 128
dtype = "float16"

a_ub = T.alloc_ub((block_M, block_N), dtype)
T.copy(A[bx * block_M, by * block_N], a_ub, pad_value=0.0)
```

**示例 3：L1 → L0A（带 transpose）**

```python
block_M, block_K = 128, 128
dtype = "float16"

A_L1 = T.alloc_L1((block_M, block_K), dtype)
A_L0 = T.alloc_L0A((block_M, block_K), dtype)
T.copy(A_L1, A_L0, transpose=True)
```

**示例 4：L0C → GM（与 T.mma 配对的 unit_flag）**

```python
block_M, block_N = 128, 128
accum_dtype = "float32"

C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)
T.mma(A_L0, B_L0, C_L0, unit_flag=0b11)
T.copy(C_L0, C[bx * block_M, by * block_N], unit_flag=0b11)
```

> `unit_flag=0b11` 须与前序 `T.mma(unit_flag=0b11)` 配对使用，不可单独使用。

**示例 5：UB → UB 跨类型 cast**

```python
block_M, block_N = 128, 128

a_ub = T.alloc_ub((block_M, block_N), "float32")
b_ub = T.alloc_ub((block_M, block_N), "float16")
T.copy(a_ub, b_ub)
```

**示例 6：UB → L1（跨 CV 搬运）**

```python
block_M, block_N = 128, 128
dtype = "float16"

a_ub = T.alloc_ub((block_M // 2, block_N), dtype)
a_l1 = T.alloc_L1((block_M, block_N), dtype)
T.copy(a_ub, a_l1)
```
