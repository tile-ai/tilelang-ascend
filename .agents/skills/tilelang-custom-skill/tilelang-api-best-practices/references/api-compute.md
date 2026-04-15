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

### T.reduce_sum(buffer, out, dim)

### T.reduce_max(buffer, out, dim)

### T.reduce_min(buffer, out, dim)

对输入 buffer 按指定维度进行归约。

**参数**：

- `buffer`：输入 buffer（2D）
- `out`：目的输出 buffer
- `dim`：reduce 轴（-1 表示最后一维）

**归约轴说明**（shape 为 (M, N) 的 2D 矩阵）：

- `dim=0`：沿第一维归约，输出 shape 为 (N,)
- `dim=-1`：沿最后一维归约，输出 shape 为 (M,)

**Softmax 中的典型用法**（来自 `examples/softmax/`）：

```python
T.reduce_max(acc_s_ub, m_i, dim=-1)
T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
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

---

## 4. Tile 扩展原语（Expert 模式 T.tile.xxx）

`T.tile.xxx` 系列接口直接触发 Tile 级的 Vector 操作指令。

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

### 4.4 逻辑运算

| API | 功能 |
|-----|------|
| `T.tile.bitwise_and(dst, src0, src1)` | dst = src0 & src1 |
| `T.tile.bitwise_or(dst, src0, src1)` | dst = src0 \| src1 |
| `T.tile.bitwise_not(dst, src0)` | dst = ~src0 |
| `T.tile.bitwise_xor(dst, src0, src1)` | dst = src0 ^ src1 |
| `T.tile.bitwise_lshift(dst, src0, scalar)` | 左移操作 |
| `T.tile.bitwise_rshift(dst, src0, scalar)` | 右移操作 |


### 4.5 比较操作

#### T.tile.compare(dst, src0, src1, mode)

逐元素比较，结果为 bit mask（1=true，0=false）。src1 可以是 buffer 或 scalar。

**mode 取值**：`"EQ"`, `"NE"`, `"GT"`, `"GE"`, `"LT"`, `"LE"`

```python
T.tile.compare(c_ub, a_ub, b_ub, "EQ")   # tensor vs tensor
T.tile.compare(c_ub, a_ub, 1.0, "GT")     # tensor vs scalar
```

### 4.6 选择操作

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

### 4.7 gather_mask

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

### 4.8 精度转换

#### T.tile.cast(dst, src, mode, count)

**mode 取值**：`"CAST_NONE"`, `"CAST_RINT"`, `"CAST_FLOOR"`, `"CAST_CEIL"`, `"CAST_ROUND"`, `"CAST_TRUNC"`, `"CAST_ODD"`

```python
T.tile.cast(b_ub, a_ub, "CAST_RINT", 4096)
```

### 4.9 数据操作

| API | 功能 |
|-----|------|
| `T.tile.fill(buffer, value)` | 用 value 填充 buffer |
| `T.tile.createvecindex(dst, first_value)` | 创建从 first_value 开始的向量索引序列 |
| `T.tile.transpose(dst, src)` | 16×16 二维矩阵数据块转置 |
| `T.tile.gather(dst, src, src_offset, src_base_addr)` | 按偏移收集数据 |
| `T.tile.arith_progression(buffer, first_value, diff_value, count)` | 生成等差数列 |

### 4.10 排序操作

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

### 4.11 两种编程范式对比

```python
# 方式一：T.Parallel + 符号 API（推荐，跨平台兼容）
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    b_ub[i, j] = T.exp(a_ub[i, j])

# 方式二：T.tile 扩展原语（Expert 模式，直接触发硬件指令）
T.tile.exp(b_ub, a_ub)
```
