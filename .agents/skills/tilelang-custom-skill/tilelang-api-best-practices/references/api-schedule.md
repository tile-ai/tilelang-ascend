# 调度原语

## 概述

TileLang 提供多种循环调度原语，用于控制循环执行方式、实现流水线并行和多核负载均衡。

## 循环原语

### T.serial(N) / T.serial(start, end, step)

普通 for 循环。

```python
for i in T.serial(N):        # 0..N-1
    ...

for i in T.serial(0, N, 2):  # 0, 2, 4, ...
    ...
```

### T.unroll(N)

循环展开，适用于小循环次数。

```python
for k in T.unroll(K_TILE):
    acc += a[k] * b[k]
```

### While 循环

```python
i = 0
while i < N:
    ...
    if done:
        break
    i += 1
```

> 支持 break 和 continue。

## T.Pipelined

### 功能

实现计算/搬运的流水线并行，通过预取来掩盖内存访问延迟。

### 语法

```python
for var in T.Pipelined(range, num_stages=N):
    ...
```

- `range`：迭代次数
- `num_stages`：预取阶段数（小于 range-1 的正整数）

### 核内流水线（Intra-core）

```python
for k in T.Pipelined(loop_k, num_stages=2):
    T.copy(A[bx * block_M, k * block_K], A_L1)
    T.copy(B[k * block_K, by * block_N], B_L1)
    T.barrier_all()
    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
    T.barrier_all()
```

`num_stages=2` 时执行顺序：

| Time | Copy A/B | Compute |
|------|----------|---------|
| t₀ | copy_A_0, copy_B_0 | |
| t₁ | copy_A_1, copy_B_1 | |
| t₂ | copy_A_2, copy_B_2 | gemm_0 |
| t₃ | copy_A_3, copy_B_3 | gemm_1 |
| t₄ | | gemm_2 |
| t₅ | | gemm_3 |

### 核间流水线（Inter-core）

Cube 和 Vector 核之间的流水并行：

```python
for k in T.Pipelined(T.ceildiv(seq_len, block_N), num_stages=2):
    T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], k_l1)
    T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
    T.copy(acc_s_l0c, workspace_1[cid, :, :])

    T.tile.fill(acc_s_ub, 0.0)
    T.copy(workspace_1[cid, vid * block_M // 2:...], acc_s_ub_)
    T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
    ...
```

**注意**：
- 核间流水线与核内流水线不能同时开启
- 使用核间流水线必须开启：`tl.ascend_auto_cv_combine: True`, `tl.ascend_auto_cross_core_sync: True`

## T.Persistent

### 功能

优化数据块在 AI Core 间的调度，使相邻数据块交由同一 AI Core 处理，提高缓存命中率。

### 语法

```python
for bx, by in T.Persistent(domain, wave_size, index, group_size=...):
    ...
```

### 示例

```python
with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
    A_L1 = T.alloc_shared((block_M, K_L1), dtype)
    B_L1 = T.alloc_shared((K_L1, block_N), dtype)
    C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

    for bx, by in T.Persistent([T.ceildiv(M, block_M), T.ceildiv(N, block_N)],
                                core_num, cid):
        loop_k = T.ceildiv(K, K_L1)
        for k in T.serial(loop_k):
            T.copy(A[bx * block_M, k * K_L1], A_L1)
            T.copy(B[k * K_L1, by * block_N], B_L1)
            T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
        T.copy(C_L0, C[bx * block_M, by * block_N])
```

## 性能调优工具

### msProf

```bash
# 上板性能分析
msprof op --kernel-name="your_kernel_func_name" python your_kernel_script.py

# 仿真性能分析
msprof op simulator --soc-version=<ascend_version> --kernel-name="your_kernel_func_name" python your_kernel_script.py
```

## 最佳实践

1. **合理选择 num_stages**：过大会增加预取阶段的内存压力，通常 2-3 即可
2. **T.Persistent 适用于大规模矩阵计算**：提高 L1 缓存命中率
3. **核间流水线需要额外配置**：必须开启 auto_cv_combine 和 auto_cross_core_sync
4. **T.unroll 适用于小常量循环**：编译器会展开循环体
