# Kernel 定义与启动

## 概述

TileLang kernel 是基于 TIR（TVM IR）的函数，通过 `@T.prim_func` 装饰器定义，使用 `@jit` 装饰器触发即时编译。

## 核心 API

### @T.prim_func

定义一个 TileLang kernel 函数。

```python
@T.prim_func
def add_kernel(
    A: T.Tensor((M, N), dtype),
    B: T.Tensor((M, N), dtype),
    C: T.Tensor((M, N), dtype),
):
    ...
```

### T.Tensor / T.Buffer

声明 kernel 参数的张量类型，包含 shape 和 dtype 信息。

```python
A: T.Tensor((M, N), "float16")
```

**支持的 dtype**：`float16, float32, bfloat16, int8, int16, int32, int64, uint8, uint16, uint32, uint64`

### 动态 shape 符号

- **T.dyn[...]**：通过 buffer 的 shape 属性获取动态维度
  ```python
  K = T.dyn['K']
  @T.prim_func
  def foo(A: T.Tensor((K,), 'float32')):
      N = A.shape[0]
  ```

- **T.dynamic(name, dtype)**：创建可直接使用的 tir.Var
  ```python
  K = T.dynamic('K', 'int32')
  @T.prim_func
  def bar(A: T.Tensor((K,), 'float32')):
      for i in T.serial(K):
          ...
  ```

> 注意：`T.symbolic` 是 `T.dynamic` 的已弃用别名。

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
- `pass_configs`：编译配置选项

**常用 pass_configs**：
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 自动同步插入
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # 自动内存规划
}
```

## 完整示例

```python
import tilelang
import tilelang.language as T
from tilelang import jit
import torch

M, N = 1024, 1024
block_M, block_N = 128, 128
VEC_NUM = 2

@jit(out_idx=[-1], pass_configs={
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
})
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

## 查看生成的 AscendC 代码

```python
func = tile_add(M, N, block_M, block_N)
print(f"{func.get_kernel_source()}")
```
