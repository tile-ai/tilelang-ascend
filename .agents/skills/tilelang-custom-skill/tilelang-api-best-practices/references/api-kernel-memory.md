# Kernel 定义、内存分配与数据搬运

## 目录

- [1. Kernel 定义与启动](#1-kernel-定义与启动)
- [2. 内存分配原语](#2-内存分配原语)
- [3. 数据搬运原语](#3-数据搬运原语)
- [4. V 核并行化](#4-v-核并行化)
- [5. 完整示例](#5-完整示例)

---

## 1. Kernel 定义与启动

### @T.prim_func

定义一个 TileLang kernel 函数。参数类型为 `T.Tensor`。

```python
@T.prim_func
def add_kernel(
    A: T.Tensor((M, N), dtype),
    B: T.Tensor((M, N), dtype),
    C: T.Tensor((M, N), dtype),
):
    ...
```

**支持的 dtype**：
- `float16`, `float32`, `bfloat16`
- `int8`, `int16`, `int32`, `int64`
- `uint8`, `uint16`, `uint32`, `uint64`

### 动态 shape 符号

- **T.dyn[...]**：通过 buffer 的 shape 属性获取动态维度
  ```python
  K = T.dyn['K']
  @T.prim_func
  def foo(A: T.Tensor((K,), 'float32')):
      N = A.shape[0]
      for i in T.serial(N):
          ...
  ```

- **T.dynamic(name, dtype)**：创建可直接使用的 tir.Var
  ```python
  K = T.dynamic('K', 'int32')
  @T.prim_func
  def bar(A: T.Tensor((K,), 'float32')):
      for i in T.serial(K):
          ...
  ```


### T.Kernel

定义 kernel 运行上下文，创建 tile block 与逻辑核的绑定。

```python
with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
    bx = cid // n_num
    by = cid % n_num
    ...
```

- **cid**：计算任务 ID，范围 [0, block_num)
- **vid**：Vector 单元索引（0 或 1），C、V 核配比为 1:2

### @jit 装饰器

触发即时编译，将 kernel 编译为 NPU 可执行代码。

```python
@jit(out_idx=[-1], pass_configs=pass_configs)
def tile_add(M, N, block_M, block_N, dtype='float'):
    @T.prim_func
    def main(...):
        ...
    return main
```

**参数**：
- `out_idx`：指定输出参数索引，如 `[-1]` 表示最后一个参数为输出
- `workspace_idx`：工作空间参数索引（详见下方 workspace 机制）
- `pass_configs`：编译配置选项

**常用 pass_configs**：
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 自动同步插入（核内）
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # 自动内存规划
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动CV分离（核间流水线需要）
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,      # 自动同步插入（CV核间）
    # TL_ASCEND_INJECT_TMP_BUFFER 默认 True，无需显式设置
    # 关闭后需手动分配 tmp_ub，详见下方 tmp_buffer 机制
}
```

#### tmp_buffer 机制

`reduce`/`broadcast`/`sigmoid`/`sort`/`topk` 等操作需要框架内部临时缓冲区。`InjectTmpBuffer` pass 负责自动注入，可通过 `TL_ASCEND_INJECT_TMP_BUFFER` 开关控制。

**自动注入模式（默认）**：框架自动创建 `tmp_ub`（uint8, UB scope），用户无需干预。

**手动分配模式**（`TL_ASCEND_INJECT_TMP_BUFFER: False`）：

```python
from tilelang.transform import get_tmp_buffer_size

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_INJECT_TMP_BUFFER: False,
    # ... 其他配置
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def my_kernel(M, N, block_M, dtype="float32"):
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M,), dtype)):
        with T.Kernel(M // block_M, is_npu=True) as (cid, vid):
            # 手动分配 tmp_ub，名称必须为 "tmp_ub"，dtype 必须为 uint8
            tmp_size = get_tmp_buffer_size((block_M, N), dtype, "reduce")
            tmp_ub = T.alloc_ub((tmp_size,), "uint8")

            a_ub = T.alloc_ub((block_M, N), dtype)
            b_ub = T.alloc_ub((block_M,), dtype)
            T.copy(A[cid * block_M, :], a_ub)
            T.reduce_max(a_ub, b_ub, dim=-1)
            T.copy(b_ub, B[cid * block_M])
    return main
```

手动模式下，框架按名称匹配并校验 tmp buffer：

| tmp buffer 名称 | dtype | 用途 |
|:--------------:|:-----:|:----:|
| `tmp_ub` | uint8 | 通用 tmp（reduce/broadcast/sigmoid 等） |
| `tmp_ub_1`, `tmp_ub_2`, ... | 按 op dtype | sort/topk/xor/merge_sort 等需 dtype 匹配 |
| `tmp_ub_reduce_out` | uint8 | pto target 下 reduce clear=False 输出 |

`get_tmp_buffer_size(shape, dtype, op_type)` 辅助函数计算所需最小字节数：

```python
from tilelang.transform import get_tmp_buffer_size

# reduce: elements × dtype_bytes
get_tmp_buffer_size((64, 256), "float32", "reduce")   # → 65536
# broadcast: elements × dtype_bytes / 4
get_tmp_buffer_size((64, 256), "float32", "broadcast") # → 16384
# sort: elements × dtype_bytes × 8
get_tmp_buffer_size((64, 256), "float32", "sort")      # → 524288
```

#### workspace 机制

**作用**：workspace buffer 用于 Cube 核（L1）和 Vector 核（UB）之间的数据中转。

由于 Ascend 硬件限制，UB 和 L1 不能直接互通，必须通过 Global Memory 中转：

```
L0C → workspace(GM) → UB   # Cube 输出到 Vector 处理
UB → workspace(GM) → L1    # Vector 输出到 Cube
```

**使用方式**：

1. 在 `@jit` 中指定 `workspace_idx`：
```python
@jit(out_idx=[-1], workspace_idx=[3])  # workspace 是第 3 个参数
def kernel(M, N, K, ...):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
        workspace: T.Tensor((M, N), accum_dtype),  # workspace_idx=[3]
    ):
        ...
```
> 注意：定义 workspace buffer 时，名称也应包含 "workspace"

2. 数据流示例（来自 `examples/quant_batch_matmul`）：
```python
# GEMM 输出 (L0C) → workspace → Vector 核处理
T.copy(C_L0, workspace[bm * block_M, bn * block_N])
T.copy(workspace[bm * block_M + vid * block_M_2, bn * block_N], c_ub)

# Vector 核处理后输出
T.copy(c_out, C[bm * block_M + vid * block_M_2, bn * block_N])
```
### 查看生成的 AscendC 代码

```python
kernel = tile_add(M, N, block_M, block_N)
print(kernel.get_kernel_source())
```

---

## 2. 内存分配原语

### Developer 模式

TileLang 对存储层级进行了抽象，分为 global、shared 和 fragment 三个级别。在 Ascend 平台中，shared 层级对应 L1 Buffer 和 Unified Buffer (UB)，fragment 层级对应 L0A/L0B/L0C Buffer。用户无需指定具体硬件存储，TileLang 编译器会根据程序上下文自动识别。

#### T.alloc_shared(shape, dtype)

分配 shared 层级的存储空间。

```python
A_L1 = T.alloc_shared((block_M, block_K), dtype)
```

#### T.alloc_fragment(shape, dtype)

分配 fragment 层级的存储空间。

```python
C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)
```

#### T.alloc_var(dtype, init, scope='local.var')

分配标量变量，支持初始化。适用于标志位、计数器、临时标量。

```python
flag = T.alloc_var("bool", init=False)
counter = T.alloc_var("int32", init=1)
b = T.alloc_var("int32", init=a)  # 用另一个变量的值初始化
```

### Expert 模式

显式指定存储位置，适用于需要精确控制内存分配的场景。

| API | 存储层级 | 抽象层级 | 说明 |
|-----|---------|---------|-----|
| `T.alloc_ub(shape, dtype)` | Unified Buffer | shared | Vector 存储单元 |
| `T.alloc_L1(shape, dtype)` | L1 Buffer | shared | Cube 存储单元 |
| `T.alloc_L0A(shape, dtype)` | L0A Buffer | fragment | Cube 左矩阵 |
| `T.alloc_L0B(shape, dtype)` | L0B Buffer | fragment | Cube 右矩阵 |
| `T.alloc_L0C(shape, dtype)` | L0C Buffer | fragment | Cube 输出/累加 |

**实际使用示例**（来自 `examples/gemm/example_gemm.py`）：

```python
A_L1 = T.alloc_L1([block_M, block_K], dtype)
B_L1 = T.alloc_L1([block_K, block_N], dtype)
C_L0 = T.alloc_L0C([block_M, block_N], accum_dtype)
```

**⚠️ 重要：存储单元对齐要求**

Ascend NPU 不同存储单元有不同的对齐要求：

| 存储单元 | 对齐要求 |
|---------|---------|
| Global Memory (GM) | 无对齐要求 |
| Unified Buffer (UB) | 32 Byte |
| L1 Buffer | 32 Byte |
| L0A Buffer | 512 Byte |
| L0B Buffer | 512 Byte |
| L0C Buffer | 64 Byte |

**⚠️ 重要：存储单元大小限制**

根据 Ascend910B3 平台配置：

| 存储单元 | 大小上限 |
|---------|---------|
| L0A | 65536 Byte |
| L0B | 65536 Byte |
| L0C | 131072 Byte |
| L1 | 524288 Byte |
| UB | 196608 Byte |

> 更多参数参见：`$ASCEND_HOME_PATH/$(uname -m)-linux/data/platform_config/Ascend910B3.ini`

分配 Buffer 时需确保不超出上限，并满足对齐要求。

---

## 3. 数据搬运原语

### T.copy(src, dst)

在不同内存层级之间搬运 tile 数据块。支持 tir.Buffer、BufferLoad、BufferRegion 类型。

**支持的搬运路径**：

| src | dst | 说明 |
|-----|-----|------|
| GM | L1 | Global Memory → L1 Buffer |
| L1 | L0A | L1 Buffer → L0A Buffer（Cube 左矩阵）|
| L1 | L0B | L1 Buffer → L0B Buffer（Cube 右矩阵）|
| L0C | GM | L0C Buffer → Global Memory |
| GM | UB | Global Memory → Unified Buffer |
| UB | GM | Unified Buffer → Global Memory |
| UB | UB | Unified Buffer → Unified Buffer |

> **注意**：UB 和 L1 之间**不能直接搬运**。Cube 核（L1）和 Vector 核（UB）的数据传递需要通过 workspace buffer（GM）中转。

**使用示例**：

```python
# GM → L1
T.copy(A[bx * block_M, k * block_K], A_L1)

# GM → UB（vid 切分）
T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

# UB → GM
T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

# L0C → GM
T.copy(C_L0, C[bx * block_M, by * block_N])

# BufferRegion 切片搬运
T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], k_l1)
```

---

## 4. V 核并行化

### 4.1 基本原理

Ascend NPU 每个 AI Core 有 1 个 Cube 核和 2 个 Vector 核（C:V = 1:2）：

- `cid`：计算任务 ID，范围 `[0, block_num)`
- `vid`：Vector 核索引，取值 0 或 1
- `VEC_NUM`：常量，通常设为 2

**默认行为**：两个 V 核（vid=0 和 vid=1）执行完全相同的代码，算力浪费。

**正确做法**：利用 `vid` 让两个 V 核分担任务。

### 4.2 V 核分担任务的模式

#### 模式一：按行切分（最常见）

每个 V 核处理 `block_dim // VEC_NUM` 行：

```python
VEC_NUM = 2
block_M_2 = block_M // VEC_NUM  # 每个 V 核处理一半行数

with T.Kernel(grid_size, is_npu=True) as (cid, vid):
    # 计算 V 核负责的起始行
    row_start = cid * block_M + vid * block_M_2
    
    # 分配 buffer（只需分配 V 核负责的行数）
    data_ub = T.alloc_shared((block_M_2, block_N), dtype)
    
    # 读入数据
    T.copy(A[row_start, by * block_N], data_ub)
    
    # 计算
    ...
    
    # 写出数据
    T.copy(data_ub, B[row_start, by * block_N])
```

**关键点**：读写索引必须一致，都使用 `row_start` 或基于 `vid` 计算的索引。

#### 模式二：按任务切分

每个 V 核处理不同的计算任务：

```python
VEC_NUM = 2

with T.Kernel(num_tasks, is_npu=True) as (cid, vid):
    # 每个 V 核处理不同的任务
    task_id = cid * VEC_NUM + vid
    
    if task_id < total_tasks:
        # 处理 task_id
        ...
```

### 4.3 workspace 索引一致性（易错点）

当 V 核读写 workspace（或任何中间 buffer）时，**必须保持索引逻辑一致**：

```python
# 错误：读写索引不一致
for row in T.serial(block_N_2):
    actual_row = bn * block_N + vid * block_N_2 + row
    T.copy(src[actual_row, ...], temp_ub)   # 读用 actual_row ✓
    # ... 处理 ...
    T.copy(temp_ub, dst[bn * block_N + row, ...])  # ❌ 写没用 actual_row

# 正确：读写索引一致
for row in T.serial(block_N_2):
    actual_row = bn * block_N + vid * block_N_2 + row
    T.copy(src[actual_row, ...], temp_ub)   # 读用 actual_row ✓
    # ... 处理 ...
    T.copy(temp_ub, dst[actual_row, ...])   # 写也用 actual_row ✓
```

**原则**：同一数据在不同阶段的索引必须基于相同的计算逻辑。

### 4.4 Cube 核不涉及 vid

Cube 核做 GEMM 时，不使用 vid 切分，读取完整的 block：

```python
# Cube 核部分（不涉及 vid）
A_L1 = T.alloc_shared((block_M, block_K), dtype)  # 完整 block_M
B_L1 = T.alloc_shared((block_N, block_K), dtype)  # 完整 block_N

T.copy(A[bm * block_M, k_offset], A_L1)   # 完整 block_M
T.copy(B[bn * block_N, k_offset], B_L1)   # 完整 block_N
T.gemm_v0(A_L1, B_L1, C_L0, ...)
```

---

## 5. 完整示例

来自 `docs/TileLang-Ascend Programming Guide.md` §2.2：

```python
import tilelang
import tilelang.language as T
from tilelang import jit
import torch

M, N = 1024, 1024
block_M, block_N = 128, 128
VEC_NUM = 2

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@jit(out_idx=[-1], pass_configs=pass_configs)
def tile_add(M: int, N: int, block_M: int, block_N: int, dtype: str = 'float'):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def add_kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            for i, j in T.Parallel(block_M // VEC_NUM, block_N):
                c_ub[i, j] = a_ub[i, j] + b_ub[i, j]

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return add_kernel

func = tile_add(M, N, block_M, block_N)
a = torch.randn(M, N).npu()
b = torch.randn(M, N).npu()
c = func(a, b)
```
