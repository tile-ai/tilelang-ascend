# T.gemm_v0

## 1. 功能说明

对操作数矩阵 A 和 B 执行分形矩阵乘加运算：`C += op(A) × op(B)`，其中 `op` 为可选转置操作。当 `init=True` 时先清零累加器 C 再计算，等价于 `C = op(A) × op(B)`。

## 2. 函数原型

### 2.1 函数定义

```python
def gemm_v0(
    A: Buffer | BufferRegion,
    B: Buffer | BufferRegion,
    C: Buffer | BufferRegion,
    transpose_A: bool = False,
    transpose_B: bool = False,
    init: bool = False,
    kL0Size: int = 128,
    n_actual: int = None,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| A | 输入 | 左矩阵，最后两维为矩阵维度，须位于 L1 | 张量（tensor） | 必填 |
| B | 输入 | 右矩阵，最后两维为矩阵维度，须位于 L1 | 张量（tensor） | 必填 |
| C | 输入/输出 | 输出矩阵（累加器），最后两维为矩阵维度，须位于 L0C | 张量（tensor） | 必填 |
| transpose_A | 输入 | 是否转置矩阵 A | 布尔 | 可选（默认 `False`） |
| transpose_B | 输入 | 是否转置矩阵 B | 布尔 | 可选（默认 `False`） |
| init | 输入 | 是否在计算前将累加器 C 清零（`True`：清零后计算 C=A×B；`False`：在现有 C 上累加 C+=A×B） | 布尔 | 可选（默认 `False`） |
| kL0Size | 输入 | K 轴分块大小，控制 L1→L0A/L0B 的数据搬运粒度，须为 16 的倍数且 ≤ 4095 | 整数 | 可选（默认 `128`） |
| n_actual | 输入 | 运行时输出列数（≤ N 且为 16 的倍数），仅 transpose_B 路径生效。当前仅 ascendc 后端支持 | 整数 | 可选（默认 `None`） |

> **类型说明**：
> - **tensor**：通过 `T.alloc_L1`、`T.alloc_L0C` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
>
> **内存层级说明**：
> - A、B 须位于 L1（通过 `T.alloc_L1` 分配），gemm_v0 内部自动将数据从 L1 搬运到 L0A/L0B 进行矩阵乘计算
> - C 须位于 L0C（通过 `T.alloc_L0C` 分配），Mmad 硬件指令的输出固定写入 L0C
> - 数据流：`GM → L1（用户搬运）→ L0A/L0B（gemm_v0 自动搬运）→ Mmad 计算 → L0C（结果）→ GM（用户搬运）`

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | A | B | C |
|------|:---:|:---:|:---:|
| Ascend A2 / A3 | float16, bfloat16, int8 | float16, bfloat16, int8 | float32（A/B 为浮点时）, int32（A/B 为 int8 时） |

#### 2.3.2 Shape 支持

- A、B、C：≥2D，最后两维为矩阵维度（M×K、K×N、M×N）

#### 2.3.3 A/B/C dtype 组合约束

A、B 的 dtype 必须相同，C 的 dtype 由 A/B 的 dtype 决定：

| A/B dtype | C dtype |
|-----------|---------|
| float16 | float32 |
| bfloat16 | float32 |
| int8 | int32 |

#### 2.3.4 分形尺寸

硬件以分形矩阵为最小计算单元，各 dtype 对应的分形尺寸如下：

| A/B dtype | A 分形（L0A） | B 分形（L0B） | C 分形（L0C） |
|-----------|--------------|--------------|--------------|
| float16 / bfloat16（2B） | 16×16 | 16×16 | 16×16 |
| int8（1B） | 16×32 | 32×16 | 16×16 |

### 2.4 约束条件

1. A、B、C 必须为 ≥2D 张量，>2D 时所有前导维度必须为 1
2. A 与 B 的 K 维度必须一致：若 `transpose_A=False`，取 `A.shape[-1]` 作为 K；否则取 `A.shape[-2]`；B 同理。两边 K 值必须相等
3. A 与 B 的 dtype 必须相同
4. A、B 须位于 L1，C 须位于 L0C（硬件约束）。gemm_v0 内部自动将 A/B 从 L1 搬运到 L0A/L0B，用户无需手动搬运到 L0A/L0B
5. A 的起始地址需 512 字节对齐（硬件约束）
6. B 的起始地址需 512 字节对齐（硬件约束）
7. C 的起始地址需 256 个元素对齐（硬件约束）
8. Mmad 单次调用的 m/n/k 取值范围为 `[0, 4095]`，当 m/n/k 中任意一个为 0 时指令不执行（硬件约束）。gemm_v0 通过 tiling 自动拆分，整体矩阵的 M/N/K 不受此限制
9. 当 `M=1` 时硬件默认开启 GEMV（General Matrix-Vector Multiplication），此时 A 须以 ND 格式排布而非 ZZ 格式（硬件约束）
10. `kL0Size` 须为 16 的倍数且 ≤ 4095

## 3. 示例代码

**示例 1：基本用法（fp16 输入，fp32 累加，init=True 清零后计算）**

```python
block_M, block_K, block_N = 128, 128, 128
A_L1 = T.alloc_L1((block_M, block_K), "float16")
B_L1 = T.alloc_L1((block_K, block_N), "float16")
C_L0 = T.alloc_L0C((block_M, block_N), "float")
T.gemm_v0(A_L1, B_L1, C_L0, init=True)
```

**示例 2：累加模式（init=False，在现有 C 值上累加）**

```python
A_L1 = T.alloc_L1((block_M, block_K), "float16")
B_L1 = T.alloc_L1((block_K, block_N), "float16")
C_L0 = T.alloc_L0C((block_M, block_N), "float")
T.gemm_v0(A_L1, B_L1, C_L0, init=False)  # C_L0 += A_L1 × B_L1
```

**示例 3：转置矩阵 B**

```python
A_L1 = T.alloc_L1((block_M, block_K), "float16")
B_L1 = T.alloc_L1((block_N, block_K), "float16")  # B 的 shape 为 (N, K)
C_L0 = T.alloc_L0C((block_M, block_N), "float")
T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=True)
```
