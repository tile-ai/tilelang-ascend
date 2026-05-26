# 计算原语：GEMM、归约与 Tile 扩展操作

---

## 1. 矩阵计算（GEMM）

### T.gemm_v0(A, B, C, transpose_A=False, transpose_B=False, init=False)

块级矩阵乘操作，计算 C += op(A) × op(B)。A、B 位于 shared 层级，C 位于 fragment 层级。

**参数**：

- `A`：左输入矩阵（shared 层级）
- `B`：右输入矩阵（shared 层级）
- `C`：结果累加输出矩阵（fragment 层级）
- `transpose_A`：是否转置 A（默认 False）
- `transpose_B`：是否转置 B（默认 False）
- `init`：是否在计算前将 C 清零（默认 False）。第一次迭代需要清零，后续累加。

**示例**（来自 `examples/gemm/example_gemm.py`）：

```python
A_L1 = T.alloc_L1([block_M, block_K], dtype)
B_L1 = T.alloc_L1([block_K, block_N], dtype)
C_L0 = T.alloc_L0C([block_M, block_N], accum_dtype)

for k in T.serial(loop_k):
    T.copy(A[bx * block_M, k * block_K], A_L1)
    T.copy(B[k * block_K, by * block_N], B_L1)
    T.barrier_all()
    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
    T.barrier_all()
T.copy(C_L0, C[bx * block_M, by * block_N])
```

**带转置的用法**：

```python
T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
```

**⚠️ 重要：矩阵 Buffer 分形限制**

使用 `T.gemm_v0` 时，矩阵 Buffer 必须满足最小分形限制。分形大小固定为 512 Byte（L0A/L0B）或 256 元素（L0C），shape 与 dtype 相关：

**分形公式**：
- **L0A**：`16 × (32B / sizeof(AType))`，固定 512 Byte
- **L0B**：`(32B / sizeof(BType)) × 16`，固定 512 Byte
- **L0C**：`16 × 16`，固定 256 元素（不随 dtype 变化）

**不同 dtype 的最小维度限制**：

| dtype | sizeof | L0A 分形 | L0B 分形 | 最小限制 |
|-------|--------|----------|----------|---------|
| int8 / uint8 | 1 Byte | 16 × 32 | 32 × 16 | M ≥ 16, K ≥ 32, N ≥ 16 |
| float16 / bfloat16 | 2 Byte | 16 × 16 | 16 × 16 | M ≥ 16, K ≥ 16, N ≥ 16 |
| int32 / float32 | 4 Byte | 16 × 8 | 8 × 16 | M ≥ 16, K ≥ 8, N ≥ 16 |

**L0C 分形固定为 16 × 16，不随 dtype 变化**，因此 M 和 N 的最小值始终为 16。

**常见错误**：`block_N = 8` 不满足 L0C 分形限制（N ≥ 16），会导致计算结果错误。

**示例**：int8 GEMM 的正确 block size 选择
```python
block_M = 64   # ≥ 16 ✓
block_N = 16   # ≥ 16 ✓（满足 L0B/L0C 分形限制）
block_K = 256  # ≥ 32 ✓（int8 的 L0A/L0B K 维度限制）
```

### T.mma(A, B, C, init=False)

NPU 级别的矩阵乘累加指令，比 `gemm_v0` 更底层。不支持 `transpose_A`/`transpose_B`。通常配合 `T.alloc_L0A`/`T.alloc_L0B` 和 `T.annotate_layout` 使用。

```python
A_L0 = T.alloc_L0A([block_M, block_K], dtype)
B_L0 = T.alloc_L0B([block_K, block_N], dtype)
C_L0 = T.alloc_L0C([block_M, block_N], accum_dtype)
T.annotate_layout({A_L1: make_zn_layout(A_L1), B_L1: make_zn_layout(B_L1)})
T.mma(A_L0, B_L0, C_L0, init=True)
```

---

## 2. 归约操作

### T.reduce_sum(buffer, out, dim=-1, clear=True, real_shape=None)

### T.reduce_max(buffer, out, dim=-1, clear=True, real_shape=None)

### T.reduce_min(buffer, out, dim=-1, clear=True, real_shape=None)

Ascend fast-path reduce 原语，主要服务于 UB tile / slice buffer 场景。

**参数**：

- `buffer`：输入 buffer 或 buffer slice
- `out`：目的输出 buffer 或 buffer slice
- `dim`：reduce 轴
- `clear`：是否在计算前初始化输出
- `real_shape`：2D slice buffer 的逻辑有效范围；未设置时默认使用物理 buffer 形状

**当前支持范围**：

- 1D buffer：`0 / -1`
- 2D buffer：`0 / 1 / -1 / -2`
- 3D buffer：仅支持 trailing-tile 轴 `0 / 1 / -1 / -2`

**`clear` 语义**：

- `clear=True`：先初始化输出，再写入 reduce 结果
- `clear=False`：将 reduce 结果 merge 到已有输出
  - `reduce_sum`：`new_out = old_out + reduced_result`
  - `reduce_max`：`new_out = max(old_out, reduced_result)`
  - `reduce_min`：`new_out = min(old_out, reduced_result)`

**输出 shape 约束**（以 2D 输入 `[M, N]` 为例）：

- `dim=-1`：输出可为 `[M]` 或 `[M, 1]`
- `dim=0`：输出可为 `[N]` 或 `[1, N]`
- 对设置了 `real_shape` 的 2D slice buffer，当前前端还兼容部分 physical-layout 输出形式，例如 `[physical_cols]` 或 `[1, physical_cols]`

**使用建议**：

- `clear` 和 `real_shape` 同时支持关键字传参和兼容的 positional 传参形式
- 推荐优先使用关键字形式，以获得更清晰的可读性
- 非法 `dim`、非法 `real_shape`、非法输出 shape 会在前端直接报错，而不是静默进入后端

**典型用法**：

```python
# Softmax / attention 场景
T.reduce_max(acc_s_ub, m_i, dim=-1)
T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)

# clear=False merge 语义
T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1, clear=False)

# slice buffer + real_shape
T.reduce_max(in_shared, out_shared, dim=-1, real_shape=[4, 4])
```

---

## 3. Element-wise 运算（Developer 模式 T.Parallel）

在 `T.Parallel` 循环内使用符号 API，跨平台兼容。

```python
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = a_ub[i, j] + b_ub[i, j]
```

**浮点单目运算**：

| 运算 | 算符表达 |
|------|---------|
| 绝对值 | `T.abs(x)` |
| 指数 | `T.exp(x)` |
| 对数 | `T.log(x)` |
| 开平方 | `T.sqrt(x)` |
| 平方根倒数 | `T.rsqrt(x)` |
| ReLU | `T.max(a, 0)` |

**浮点双目运算**：`+`, `-`, `*`, `/`, `T.min(a, b)`, `T.max(a, b)`

**整形运算**：`~`(位非), `<<`, `>>`, `&`(位与), `|`(位或)

**向量-标量运算与广播**：

```python
# 向量-标量
for j in T.Parallel(block_N):
    c_ub[j] = a_ub[j] + 1

# 行广播
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = a_ub[i, j] * b_ub[i]  # b_ub.shape = (block_M // VEC_NUM,)

# 维度不匹配广播
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = b_ub[j] + 5  # b_ub 是 1D，c_ub 是 2D
```

**列切分模式**：

```python
for i in range(block_M // VEC_NUM):  # 行顺序
    for j in T.Parallel(block_N):    # 列并行
        c_ub[i, j] = a_ub[i, j] * b_ub[i, j]
```

### 3.1 T.Parallel 在 TileLang-Ascend 上的限制

> **核心原理**：`T.Parallel` 在 TileLang-Ascend 上会被编译器 lowering 为 `T.tile.xxx` Buffer 级 SIMD 指令。因此，T.Parallel 的能力边界受限于 AscendC Vector 指令的能力。

#### 支持的循环维度

- ✅ **1D 并行**：`for j in T.Parallel(N)`
- ✅ **2D 并行**：`for i, j in T.Parallel(M, N)`
- ✅ **serial + parallel 组合**：`for i in range(M): for j in T.Parallel(N)`
- ❌ **3D 或更高维并行**：不支持，会触发编译错误

#### 支持的表达式类型

`T.Parallel` 内的表达式会被自动分解并翻译为 Vector 指令。**仅支持以下模式**：

| 类型 | 支持的表达式 | 备注 |
|------|-------------|------|
| 简单赋值 | `a[i] = b[i]` | 等价于 `T.copy` |
| 简单运算 | `c[i] = a[i] + b[i]` | 等价于 `T.tile.add` |
| 标量运算 | `c[i] = a[i] + scalar` | 等价于 `T.tile.add` |
| 广播运算 | `c[i,j] = a[i,j] * b[j]` | 自动广播处理（仅支持 1D→2D，索引必须是简单变量） |
| 复合表达式 | `c[i] = a[i] * b[i] + d[i]` | 自动分解为多步操作 |
| 离散索引  | 非简单变量索引，如 `a[idx[i]]` | 编译器退回到 `T.serial` 循环 |

#### 不支持的表达式

以下表达式**无法在 T.Parallel 中使用**，需要改用其他方案：

| 不支持的表达式 | 错误类型 | 替代方案 |
|---------------|---------|---------|
| `if-else` 条件分支 | 编译错误（SIMD 架构不支持元素级条件判断） | 使用 `T.tile.compare` + `T.tile.select` |
| `T.if_then_else(...)` | 编译错误 ("undefined Variable v_thread") | 使用 `T.tile.compare` + `T.tile.select` |
| `tir.reinterpret("int8", ...)` | 运行时错误 | 使用 `T.reinterpretcast`（整个 buffer） |
| `T.int8(expr)` 或 `.astype("int8")` | 编译错误或数据异常 | 使用 `T.tile.cast`（整个 buffer） |
| 非线性索引 `a[i*i]` | 未实现 | 使用 `T.tile.xxx` + 手动索引计算 |
| 动态 shift `a[i] >> shift[i]` | 不支持（shift 必须是 scalar） | 使用固定 scalar shift |

#### 循环范围要求

`T.Parallel` 的循环范围必须是编译期可确定的常量值（IntImm），不支持动态变量作为循环边界。

#### 从 CUDA TileLang 迁移注意事项

TileLang-Ascend 的 T.Parallel 语法与 CUDA 版本对齐，但底层执行模型不同：

- **CUDA (SIMT)**：每个元素独立执行，支持复杂控制流
- **Ascend (SIMD)**：所有元素并行执行相同指令，不支持条件分支

CUDA 代码中的以下模式在 Ascend 上需要改写：

```python
# CUDA 版本（SIMT，逐元素条件判断）
for i in T.Parallel(N):
    if a[i] > threshold:      # ❌ Ascend 不支持
        b[i] = a[i] * scale
    else:
        b[i] = a[i]

# Ascend 版本（SIMD，用 compare + select 替代）
T.tile.compare(mask_ub, a_ub, threshold, "GT")
T.tile.select(b_ub, mask_ub, a_scaled_ub, a_ub, "VSEL_CMPMASK_SPR")
```

详细用法参考 `docs/tutorials/t_parallel.md`。

Pass 设计详见 `.agents/skills/tilelang-pass-analyzer/references/pass-designs/ascend_lower_parallel_to_vector_design.md`。

---

## 4. Tile 扩展原语（T.tile.xxx Buffer 级 SIMD 操作）

`T.tile.xxx` 系列接口直接触发 Tile 级的 Ascend 操作。它们既可用于全手动 Expert 模式，也可在 Developer pass_configs 下作为混合模式原语使用。

### 4.1 基础算术

| API | 功能 | src1 类型 |
|-----|------|----------|
| `T.tile.add(dst, src0, src1)` | dst = src0 + src1 | buffer 或 scalar |
| `T.tile.sub(dst, src0, src1)` | dst = src0 - src1 | buffer 或 scalar |
| `T.tile.mul(dst, src0, src1)` | dst = src0 * src1 | buffer 或 scalar |
| `T.tile.div(dst, src0, src1)` | dst = src0 / src1 | buffer 或 scalar |
| `T.tile.max(dst, src0, src1)` | dst = max(src0, src1) | buffer 或 scalar |
| `T.tile.min(dst, src0, src1)` | dst = min(src0, src1) | buffer 或 scalar |

### 4.2 单目运算

| API | 功能 |
|-----|------|
| `T.tile.exp(dst, src0)` | dst = exp(src0) |
| `T.tile.ln(dst, src0)` | dst = ln(src0) |
| `T.tile.abs(dst, src0)` | dst = abs(src0) |
| `T.tile.reciprocal(dst, src0)` | dst = 1/src0 |
| `T.tile.sqrt(dst, src0)` | dst = √src0 |
| `T.tile.rsqrt(dst, src0)` | dst = 1/√src0 |
| `T.tile.relu(dst, src0)` | dst = max(0, src0) |

### 4.3 需要额外参数的运算

| API | 功能 |
|-----|------|
| `T.tile.leaky_relu(dst, src0, scalar)` | Leaky ReLU，scalar 为负斜率系数 |
| `T.tile.axpy(dst, src0, scalar)` | dst = scalar * src0 + dst |
| `T.tile.sin(dst, src0)` | dst = sin(src0) |
| `T.tile.cos(dst, src0)` | dst = cos(src0) |

### 4.4 复合运算

| API | 功能 |
|-----|------|
| `T.tile.mul_add_dst(dst, src0, src1)` | dst = src0 * src1 + dst（融合乘加） |
| `T.tile.silu(dst, src0)` | dst = src0 * sigmoid(src0)（SiLU/Swish 激活） |

**说明**：
- `mul_add_dst` 执行融合乘加操作，将 src0 和 src1 相乘后加到 dst 上
- dst 既作为输入（累加器）也作为输出
- 支持 half、float 类型（Atlas A2/A3）
- 也支持 int16_t、uint16_t、int32_t、uint32_t（Atlas 200I/500 A2）

- `silu` 执行 SiLU (Swish) 激活函数: x * sigmoid(x)
- 支持 half、float 类型（Atlas A2/A3）

### 4.5 逻辑运算

| API | 功能 |
|-----|------|
| `T.tile.bitwise_and(dst, src0, src1)` | dst = src0 & src1 |
| `T.tile.bitwise_or(dst, src0, src1)` | dst = src0 \| src1 |
| `T.tile.bitwise_not(dst, src0)` | dst = ~src0 |
| `T.tile.bitwise_xor(dst, src0, src1)` | dst = src0 ^ src1 |
| `T.tile.bitwise_lshift(dst, src0, scalar)` | 左移操作 |
| `T.tile.bitwise_rshift(dst, src0, scalar)` | 右移操作 |


### 4.6 比较操作

#### T.tile.compare(dst, src0, src1, mode)

逐元素比较，结果为 bit mask（1=true，0=false）。src1 可以是 buffer 或 scalar。

**mode 取值**：`"EQ"`, `"NE"`, `"GT"`, `"GE"`, `"LT"`, `"LE"`

```python
T.tile.compare(c_ub, a_ub, b_ub, "EQ")   # tensor vs tensor
T.tile.compare(c_ub, a_ub, 1.0, "GT")     # tensor vs scalar
```

### 4.7 选择操作

#### T.tile.select(dst, selMask, src0, src1, selMode)

根据 selMask 的比特位选取元素。bit=1 选 src0，bit=0 选 src1。

**selMode 取值**：

- `"VSEL_CMPMASK_SPR"`：根据 compare mask 选择
- `"VSEL_TENSOR_SCALAR_MODE"`：tensor 和 scalar 之间选择
- `"VSEL_TENSOR_TENSOR_MODE"`：两个 tensor 之间选择

```python
T.tile.select(c_ub, selmask_ub, a_ub, b_ub, "VSEL_CMPMASK_SPR")
T.tile.select(c_ub, selmask_ub, a_ub, 1.0, "VSEL_TENSOR_SCALAR_MODE")
T.tile.select(c_ub, mask_ub, a_ub, b_ub, "VSEL_TENSOR_TENSOR_MODE")
```

### 4.8 gather_mask

#### T.tile.gather_mask(dst, src, src1Pattern)

根据 mask 模式收集元素。

**固定模式**（src1Pattern 为字符串）：

- `"P0101"`：按偶数索引  `"P1010"`：按奇数索引
- `"P0001"/"P0010"/"P0100"/"P1000"`：每四个取一个
- `"P1111"`：取全部

**自定义模式**（src1Pattern 为 buffer）：按索引选取。

```python
T.tile.gather_mask(b_ub, a_ub, "P0101")
```

### 4.9 精度转换

#### T.tile.cast(dst, src, mode, count)

**mode 取值**：`"CAST_NONE"`, `"CAST_RINT"`, `"CAST_FLOOR"`, `"CAST_CEIL"`, `"CAST_ROUND"`, `"CAST_TRUNC"`, `"CAST_ODD"`

```python
T.tile.cast(b_ub, a_ub, "CAST_RINT", 4096)
```

### 4.10 数据操作

| API | 功能 |
|-----|------|
| `T.tile.fill(buffer, value)` | 用 value 填充 buffer |
| `T.tile.createvecindex(dst, first_value)` | 创建从 first_value 开始的向量索引序列 |
| `T.tile.transpose(dst, src)` | 16×16 二维矩阵数据块转置 |
| `T.tile.gather(dst, src, src_offset, src_base_addr)` | 按偏移收集数据 |
| `T.tile.arith_progression(buffer, first_value, diff_value, count)` | 生成等差数列 |

### 4.10 原子操作

#### T.tile.atomic_add(dst, src)

将本地 tensor tile 原子累加到 GM 目标区域。该 API 是 Ascend 专属的 `T.tile` 原语，不等价于主仓 GPU 风格的全局 `T.atomic_add`。

**V1 支持范围**：

- `dst` 必须是 GM/global buffer、buffer load 或 region
- `src` 必须是本地 tensor，当前主要面向 UB/shared buffer 和 L0C/fragment buffer
- `src` 与 `dst` dtype 必须一致
- 支持 1D 和 2D tile region 的 local -> GM 原子累加
- 不支持 `return_prev`、`memory_order`、`use_tma`、常量 src 或任意表达式 src

**支持的数据类型**：

int8, int16, float16, bfloat16, int32, float32

**使用建议**：

- 如果业务语义是从 0 开始累加，调用 kernel 前或 kernel 内需要显式清零 GM 输出。
- 在混合模式下可配合自动同步和内存规划使用，不要求手写 `T.Scope("V")` 或 `T.barrier_all()`。

**UB -> GM 示例**：

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

src_ub = T.alloc_ub((tile_n,), "float32")
T.tile.fill(src_ub, 1.0)
T.tile.atomic_add(C[0], src_ub)
```
示例中的pass_config只是最小用法。在混合模式或需要自动 C/V 分离时，可以同时开启 `TL_ASCEND_AUTO_CV_COMBINE`；如果存在 C/V 核间依赖，再配合 `TL_ASCEND_AUTO_CV_SYNC`。

**L0C -> GM 示例**：

适用于矩阵计算结果需要原子累加到 GM 的场景，如多 block/core 的 GEMM 累加。

```python
src_l0c = T.alloc_L0C((block_M, block_N), dtype)
T.gemm_v0(..., ..., src_l0c, init=True)
T.tile.atomic_add(C[..., ...], src_l0c)
```

**底层实现**：

底层会生成 Ascend C 的 DMA atomic add 语义：开启 `SetAtomicAdd<T>()`，执行 local -> GM 的 `DataCopyPad`，再通过兼容 helper 关闭 atomic 状态。
### 4.11 排序操作

#### T.tile.sort(dst, src, actual_num)

**参数**：

  - dst：存储排序后结果的目标缓冲区(val0, index0, val1, index1 ,...)
  - src：源操作数，待排序数据(val0, val1, val2, ...)
  - actual_num：src 中实际参与排序的元素数量

**功能**：排序函数，将任意长度数据按照数值大小进行一次性降序排序

**举例**：

```
# 对131个数进行排序
# 131向上对齐到160，src.shape = (1, 160), actual_num = 131
T.tile.sort(dst, src, actual_num)
```

**注意事项**：
  - `dst`与 `src` 数据类型相同，仅支持float32和float16数据类型
  - `src` 的大小需要满足32或32的整数倍

#### T.tile.merge_sort(dst, src0, src1, src2=None, src3=None)

将多个已排序数据块合并，支持 2/3/4-way 归并。输入/输出均为 value-index pair 格式。

```python
T.tile.merge_sort(merge_dst, src0, src1)            # 2-way
T.tile.merge_sort(merge_dst, src0, src1, src2)       # 3-way
T.tile.merge_sort(merge_dst, src0, src1, src2, src3) # 4-way
```

#### T.tile.topk(dst, src, K, actual_num)

**参数**：

  - dst：存储TopK结果的目标缓冲区(val0, index0, val1, index1 ,...)
  - src：包含输入数据的源缓冲区(val0, val1, val2, ...)
  - K：前K个排序结果
  - actual_num：实际参与排序的元素个数

**功能**：执行 TopK 操作，实现对源数据的一次性从大到小排序，选择前K个元素，以（数、索引）的方式输出

**举例**:

```
# 对41个数进行排序，选择前10个数
# 需要使41向上对齐至32 * 2 = 64，K = 10, actual_num = 41
# topk_global.shape = (1, 20)sort_result.shape = (1, 64)
T.tile.topk(topk_global, sort_result, K, actual_num)
```

**注意事项**：
  - `src` 的大小需要满足32或32的整数倍

### 4.12 两种编程范式对比

```python
# 方式一：T.Parallel + 符号 API（Developer 模式，跨平台兼容）
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    b_ub[i, j] = T.exp(a_ub[i, j])

# 方式二：T.tile 扩展原语（Expert / 混合模式，直接触发硬件指令）
T.tile.exp(b_ub, a_ub)
```
