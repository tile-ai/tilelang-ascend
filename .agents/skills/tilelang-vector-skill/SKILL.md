---
name: tilelang-vector-skill
description: TileLang Ascend Vector算子生成指南。用于开发昇腾NPU上的Vector算子，包括激活函数、逐元素运算、归一化、归约等。当用户要求编写vector算子、逐元素操作、激活函数、归一化算子、reduce操作，或询问如何用TileLang实现具体的数学运算时触发此skill。
---

# TileLang Vector算子开发指南

## 概述

Vector算子在昇腾NPU的Vector核上执行，适用于逐元素运算、归约、归一化等操作。本skill帮助你快速生成正确、高效的vector算子实现。

## 核心概念

### 硬件架构
- **Vector核**: 执行逐元素运算（exp, add, mul等）和归约操作
- **UB (Unified Buffer)**: Vector核的本地缓存，数据必须先拷贝到UB才能计算
- **VEC_NUM**: 通常为2，表示每个block由2个vector核并行处理

### 数据流
```
GM (全局内存)
  ↕ T.copy
UB (Vector核缓存)
  → T.tile.xxx (计算)
UB
  ↕ T.copy
GM
```

## 开发流程

### 1. 需求分析
明确算子的：
- 数学公式（如 `y = sigmoid(x) = 1/(1+exp(-x))`）
- 输入输出shape
- 数据类型（float, float16等）
- 是否需要归约操作

### 2. 选择模式
根据复杂度选择：
- **简单elementwise**: 单层循环，无需中间buffer
- **归约类**: 需要reduce_sum/max/min，需要tmp buffer
- **多步骤**: 需要多个中间buffer，注意内存规划

### 3. 确定分块参数
```python
block_M = 64   # 每个block处理的行数，影响并行度
block_N = 128  # 每个block处理的列数，影响内存占用
VEC_NUM = 2    # 固定为2
```

**内存限制**: UB空间有限，所有buffer总大小需在限制内
- 单个buffer: `block_M // VEC_NUM * block_N * sizeof(dtype)`
- 需要验证不会内存溢出

### 4. 动态Shape设置（如需要）

如果算子的某个维度是动态的（如batch、seq_len等），可以使用 `T.symbolic` 定义：

```python
@tilelang.jit(out_idx=[...], pass_configs=pass_configs)
def my_dynamic_op(N, block_M, block_N, dtype="float16"):
    # 在 @T.prim_func 之前定义动态维度
    M = T.symbolic("M")
    shape = [M, N]  # 定义shape变量
    
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    
    @T.prim_func
    def main(
        A: T.Tensor(shape, dtype),  # 使用shape变量
        B: T.Tensor(shape, dtype),
        Out: T.Tensor(shape, dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            # Kernel实现

    return main
```

**注意**：
- `M = T.symbolic("M")` 必须放在 `@tilelang.jit` 装饰的函数内部、`@T.prim_func` 之前
- shape 需要先定义为变量（如 `shape = [M, N]`），然后在 `T.Tensor(shape, dtype)` 中使用
- 动态维度会从输入tensor的shape自动推断

### 5. 编写Kernel结构

#### 基础模板
```python
@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def my_vector_op(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            # Kernel实现
            
    return main
```

#### 分块处理模式
```python
# 模式1: 按行分块（适用于行独立的操作）
with T.Kernel(m_num, is_npu=True) as (cid, vid):
    bx = cid
    for by in T.serial(n_num):
        # 处理每个列块

# 模式2: 按块分块（适用于elementwise操作）
with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
    bx = cid // n_num
    by = cid % n_num
    # 处理单个块
```

### 6. 内存分配

#### UB分配
```python
# 工作buffer
a_ub = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)

# 中间计算buffer（根据需要）
temp_ub = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)

# 归约tmp buffer（仅用于reduce操作）
tmp_ub = T.alloc_ub([3 * DataType(dtype).bits // 8 * block_M // VEC_NUM * block_N], "uint8")
```

### 7. 数据搬运

```python
# GM → UB
T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

# UB → GM
T.copy(a_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])
```

### 8. 计算实现

#### Elementwise操作
```python
with T.Scope("V"):
    T.barrier_all()
    T.tile.add(c_ub, a_ub, b_ub)  # c = a + b
    T.barrier_all()
```

#### 激活函数
```python
with T.Scope("V"):
    T.barrier_all()
    T.tile.exp(exp_ub, a_ub)           # exp(x)
    T.tile.add(exp_ub, exp_ub, 1.0)    # exp(x) + 1
    T.tile.reciprocal(b_ub, exp_ub)    # 1/(exp(x)+1)
    T.barrier_all()
```

#### 归约操作
```python
with T.Scope("V"):
    T.reduce_max(a_ub, max_ub, tmp_ub, dim=-1)  # 行最大值
    T.reduce_sum(a_ub, sum_ub, tmp_ub, dim=-1)  # 行求和
```

### 9. 广播操作
当需要将1D向量广播到2D矩阵时：
```python
max_ub = T.alloc_ub([block_M // VEC_NUM, 1], dtype)     # [M, 1]
max_2d_ub = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)  # [M, N]

T.tile.broadcast(max_2d_ub, max_ub, tmp_ub)  # 广播 [M,1] → [M,N]
```

### 10. 同步
在Vector核内，不同操作间需要同步：
```python
T.barrier_all()  # Vector核内同步
```

## 常见模式

### Pattern 1: 简单Elementwise
适用：逐元素四则运算、激活函数
- 单层循环遍历所有块
- 每块独立计算
- 示例：sigmoid, relu, add, mul

### Pattern 2: 行归约
适用：需要沿某维度归约的操作
- 先累积所有列块的数据
- 执行归约操作
- 广播结果用于后续计算
- 示例：softmax, layer_norm, reduce_sum

### Pattern 3: 多步骤计算
适用：复杂公式，需要中间结果
- 合理分配中间buffer
- 注意buffer复用（开启内存规划）
- 示例：swish, gelu

## 性能优化

### 分块大小选择
- **小block**: 更好的并行度，但可能增加调度开销
- **大block**: 更好的数据局部性，但占用更多UB空间
- **经验值**: 
  - block_M: 32-128
  - block_N: 64-256（需确保 `block_M // VEC_NUM * block_N` 不会太大）

### 内存优化
开启自动内存规划以复用buffer：
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

### 减少同步
合并连续的同类型操作：
```python
# 差：每次操作都同步
T.tile.exp(a_ub, a_ub)
T.barrier_all()
T.tile.add(a_ub, a_ub, 1.0)
T.barrier_all()

# 好：类型相同的操作间同步
T.tile.exp(a_ub, a_ub)
T.tile.add(a_ub, a_ub, 1.0)
T.barrier_all()
```

## 调试技巧

### 1. 开启调试模式
```python
export TL_DEBUG=1
```

### 2. 使用printf
```python
T.printf("max_ub[0] = %f\n", max_ub[0])
```

### 3. 检查生成的代码
```python
print(func.get_kernel_source())
```

### 4. 小规模验证
先用小shape测试正确性，再扩大规模：
```python
test_configs = [
    (64, 64, 32, 64),      # 小规模
    (256, 256, 64, 128),   # 中等规模
    (1024, 1024, 64, 128), # 大规模
]
```

## 常见错误

### 1. 内存溢出
**错误**: `Memory allocation failed`
**原因**: UB空间不足
**解决**: 减小block_M或block_N，或优化buffer使用

### 2. 维度不匹配
**错误**: `Source and Dest dimension must match`
**原因**: broadcast操作的源和目标shape不符合要求
**解决**: 确保源是 `[M, 1]` 或 `[1, N]`，目标是 `[M, N]`

### 3. 缺少同步
**现象**: 结果不稳定或错误
**原因**: 不同操作间未同步
**解决**: 在T.tile操作间添加 `T.barrier_all()`

## API快速参考

详见 [api-quickref.md](references/api-quickref.md)

## 示例代码

详见 [examples.md](references/examples.md)

## 疑难解答

详见 [troubleshooting.md](references/troubleshooting.md)