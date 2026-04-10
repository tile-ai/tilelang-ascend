# Kernel 定义、内存分配与数据搬运

---

## 1. Kernel 定义与启动

### @T.prim_func

定义一个 TileLang kernel 函数。参数类型为 `T.Tensor` 或 `T.Buffer`。

```python
@T.prim_func
def add_kernel(
    A: T.Tensor((M, N), dtype),
    B: T.Tensor((M, N), dtype),
    C: T.Tensor((M, N), dtype),
):
    ...
```

**支持的 dtype**：`float16, float32, bfloat16, int8, int16, int32, int64, uint8, uint16, uint32, uint64`

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
- **vid**：Vector 单元索引（0 或 1），A2/A3 架构 CV 核配比可为 1:2 或 1:1
- **VEC_NUM**：通常设为 2，表示每个 AI Core 有 2 个 Vector 计算单元

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
- `workspace_idx`：工作空间参数索引（如 Flash Attention 中 `workspace_idx=[4,5,6]`）
- `pass_configs`：编译配置选项

**常用 pass_configs**：
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 自动同步插入
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # 自动内存规划
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动CV分离（核间流水线需要）
}
```

### 查看生成的 AscendC 代码

```python
func = tile_add(M, N, block_M, block_N)
print(f"{func.get_kernel_source()}")
```

---

## 2. 内存分配原语

### Developer 模式

TileLang 对存储层级进行了抽象，分为 Global、shared 和 fragment 三个级别。在 Ascend 平台中，shared 层级对应 L1 Buffer 和 Unified Buffer，fragment 层级对应 L0A/L0B/L0C Buffer。用户无需指定具体硬件存储，TileLang 编译器会根据程序上下文自动识别。

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

| API | 存储层级 | 说明 |
|-----|---------|------|
| `T.alloc_ub(shape, dtype)` | Unified Buffer (UB) | Vector 计算 |
| `T.alloc_L1(shape, dtype)` | L1 Buffer | 片上缓存 |
| `T.alloc_L0A(shape, dtype)` | L0A Buffer | Cube 左矩阵 |
| `T.alloc_L0B(shape, dtype)` | L0B Buffer | Cube 右矩阵 |
| `T.alloc_L0C(shape, dtype)` | L0C Buffer | Cube 输出/累加 |

**实际使用示例**（来自 `examples/gemm/example_gemm.py`）：

```python
A_L1 = T.alloc_L1([block_M, block_K], dtype)
B_L1 = T.alloc_L1([block_K, block_N], dtype)
C_L0 = T.alloc_L0C([block_M, block_N], accum_dtype)
```

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
| UB | L1 | Unified Buffer → L1 Buffer |

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

## 4. 完整示例

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
