# Developer 模式完整指南

## 核心理念

Developer 模式使用抽象化的 Tile Library 接口，编译器自动处理：
- **存储映射**：`alloc_shared` 自动映射到 L1 或 UB
- **同步插入**：自动在搬运和计算之间插入 barrier
- **CV 分离**：自动将 Cube 和 Vector 操作分离到不同核
- **核间同步**：自动在 Cube→Vector 数据交换点插入同步

---

## 必选 pass_configs

### 含 Cube 的算子（GEMM、融合算子）— 4 个全开

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}
```

### 纯 Vector 算子（无 GEMM）— 只需 2 个

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

---

## 内存分配

| Developer API | 编译器自动映射 | 使用场景 |
|---------------|-------------|---------|
| `T.alloc_shared(shape, dtype)` | L1（被 GEMM 使用）或 UB（被 Vector 使用） | 所有中间 buffer |
| `T.alloc_fragment(shape, dtype)` | L0C（GEMM 累加器） | GEMM 输出累加 |

**编译器判断规则：**
- buffer 被用作 `T.gemm_v0/v1` 的输入 → 映射到 **L1**
- buffer 被用于 `T.Parallel` 或 `T.tile.*` → 映射到 **UB**

---

## 计算表达

### Element-wise 运算（T.Parallel）

```python
for i, j in T.Parallel(block_M, block_N):
    c[i, j] = a[i, j] + b[i, j]          # 加法
    c[i, j] = a[i, j] * scalar            # 标量乘
    c[i, j] = T.exp(a[i, j])             # 指数
    c[i, j] = T.max(a[i, j], b[i, j])    # 最大值
    c[i, j] = T.log(a[i, j])             # 对数
    c[i, j] = T.sqrt(a[i, j])            # 平方根
    c[i, j] = T.abs(a[i, j])             # 绝对值
    c[i, j] = T.rsqrt(a[i, j])           # 平方根倒数
```

### 自动广播（2D-1D）

```python
# 2D buffer 与 1D buffer 运算，自动按行广播
for h_i, j in T.Parallel(block_M, block_N):
    a[h_i, j] = a[h_i, j] - tile_max[h_i]     # 行广播减
    a[h_i, j] = a[h_i, j] / prev_sum[h_i]      # 行广播除
    a[h_i, j] = a[h_i, j] * scale[h_i]         # 行广播乘
```

### 复合表达式

```python
for i in T.Parallel(block_M):
    m_i[i] = T.max(m_i[i], m_i_prev[i])
    m_i_prev[i] = T.exp(m_i_prev[i] - m_i[i])

for h_i, j in T.Parallel(block_M, block_N):
    acc_s[h_i, j] = T.exp(acc_s[h_i, j] - m_i[h_i])
```

### 矩阵计算

```python
# A_shared, B_shared: alloc_shared（自动映射到 L1）
# C_fragment: alloc_fragment（自动映射到 L0C）
T.gemm_v0(A_shared, B_shared, C_fragment, transpose_A=False, transpose_B=False, init=True)
```

### 归约操作

```python
# 归约 API 在两种模式下通用
T.reduce_max(acc_s_ub, m_i, tmp_ub, dim=-1)
T.reduce_sum(acc_s_ub, sumexp_ub, tmp_ub, dim=-1)
T.reduce_min(acc_s_ub, min_ub, tmp_ub, dim=-1)

# tmp_ub 大小：[3 * DataType(accum_dtype).bits // 8 * rows * cols]，dtype 为 "uint8"
```

---

## 数据搬运

```python
# GM → shared buffer
T.copy(A[bx * block_M, k * K_L1], A_L1)

# fragment → GM（通过 workspace 中转到 Vector）
T.copy(C_L0, workspace[bx * block_M, by * block_N])

# workspace → shared buffer（Vector 侧读取）
T.copy(workspace[bx * block_M + vid * half_M, by * block_N], c_ub)

# shared buffer → GM（最终输出）
T.copy(c_ub, Output[bx * block_M + vid * half_M, by * block_N])
```

---

## 循环调度

```python
# 普通循环
for k in T.serial(loop_k):
    ...

# 流水线调度（搬运与计算重叠）
for k in T.Pipelined(T.ceildiv(seq_len, block_N), num_stages=2):
    ...

# 循环展开
for k in T.unroll(4):
    ...
```

---

## Cube→Vector 数据中转

Developer 模式下，Cube 输出（`alloc_fragment` / L0C）无法直接被 Vector 读取，必须通过 **workspace tensor** 中转：

```python
@tilelang.jit(out_idx=[2], workspace_idx=4, pass_configs=pass_configs)
def my_op(...):
    @T.prim_func
    def main(A, B, C, D,
             workspace: T.Tensor((M, N), dtype)):  # workspace 参数
        with T.Kernel(...) as (cid, vid):
            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)
            c_ub = T.alloc_shared((block_M // 2, block_N), dtype)

            # Cube 部分：结果写入 workspace
            T.gemm_v0(...)
            T.copy(C_L0, workspace[bx * block_M, by * block_N])

            # Vector 部分：从 workspace 读取
            T.copy(workspace[bx * block_M + vid * half_M, by * block_N], c_ub)
            for i, j in T.Parallel(half_M, block_N):
                c_ub[i, j] = c_ub[i, j] + d_ub[i, j]
            T.copy(c_ub, C[...])
    return main
```

**关键：**
- `workspace_idx` 在 `@tilelang.jit` 中声明（可以是整数或列表）
- workspace 作为函数参数出现，JIT 自动分配
- 编译器自动在 workspace 写入和读取之间插入核间同步

---

## Developer 模式禁止事项

1. **不使用** `T.Scope("C")` 或 `T.Scope("V")` — 编译器自动分离
2. **不使用** `T.barrier_all()` / `T.set_flag` / `T.wait_flag` — 编译器自动同步
3. **不使用** `T.set_cross_flag` / `T.wait_cross_flag` — 编译器自动核间同步
4. **不使用** `T.alloc_L1` / `T.alloc_ub` / `T.alloc_L0A` / `T.alloc_L0B` / `T.alloc_L0C` — 使用 `alloc_shared` / `alloc_fragment`
5. **不使用** `T.annotate_address` — 编译器自动内存规划

---

## 代码骨架

```python
import tilelang
import tilelang.language as T
import torch

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}

@tilelang.jit(out_idx=[<输出索引>], workspace_idx=[<workspace索引>], pass_configs=pass_configs)
def my_operator(<形状参数>, dtype="float16", accum_dtype="float"):
    block_num = ...

    @T.prim_func
    def main(
        Input1: T.Tensor((<shape>), dtype),
        Output: T.Tensor((<shape>), dtype),
        workspace: T.Tensor((<shape>), dtype),  # 如需 Cube→Vector 中转
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            # 1. 计算 block 坐标
            bx = cid // n_num
            by = cid % n_num

            # 2. 分配 buffer
            buf = T.alloc_shared((<shape>), dtype)
            acc = T.alloc_fragment((<shape>), accum_dtype)

            # 3. 数据搬运 + 计算
            T.copy(Input1[...], buf)
            T.gemm_v0(...)  # 或 T.Parallel 计算

            # 4. 写回
            T.copy(acc, Output[...])

    return main

# 实例化 & 调用
func = my_operator(M, N, K, ...)
result = func(input_tensor)
```
