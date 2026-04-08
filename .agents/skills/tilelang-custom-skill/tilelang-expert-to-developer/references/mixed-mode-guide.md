# 混合模式指南

## 定义

混合模式是实践中最常见的编程方式：**Developer 模式处理主体逻辑**，Expert 模式的扩展接口补充 Developer 模式暂不支持的操作。

---

## 使用规则

1. 使用 **Developer 模式的 pass_configs**（4 个开关全开）
2. 使用 `T.alloc_shared` / `T.alloc_fragment` 分配内存
3. 主体计算用 `T.Parallel` + 符号运算
4. 特殊操作用 Expert 扩展 API 补充
5. **不使用** `T.Scope` / 手动同步

---

## Developer 模式中可直接使用的 Expert API

### 常用扩展

| Expert API | 功能 | Developer 模式中可用 |
|-----------|------|---------------------|
| `T.tile.fill(buffer, value)` | 常量填充 | ✅ 推荐 |
| `T.tile.cast(dst, src, mode, count)` | 精度转换 | ✅ 推荐 |
| `T.tile.broadcast(dst, src, tmp)` | 1D→2D 广播 | ✅ |
| `T.tile.axpy(dst, src, scalar)` | dst += scalar*src | ✅ |
| `T.tile.compare(dst, src0, src1, mode)` | 比较 | ✅ |
| `T.tile.select(dst, mask, src0, src1, mode)` | 条件选择 | ✅ |
| `T.tile.sin(dst, src, tmp)` / `T.tile.cos(dst, src, tmp)` | 三角函数 | ✅ |
| `T.tile.sigmoid(dst, src, tmp)` | Sigmoid | ✅ |
| `T.tile.gather(dst, src, offset, base_addr)` | 数据收集 | ✅ |
| `T.tile.sort(dst, src, indices, tmp, repeat)` | 排序 | ✅ |
| `T.tile.topk(dst, src, tmp, block_size)` | Top-K | ✅ |

### 归约操作（两种模式通用）

```python
T.reduce_max(buf, out, tmp, dim=-1)
T.reduce_sum(buf, out, tmp, dim=-1)
T.reduce_min(buf, out, tmp, dim=-1)
```

---

## 典型混合模式场景

### 场景 1：初始化 + Developer 计算

```python
# Expert: T.tile.fill 初始化
T.tile.fill(acc_o, 0.0)
T.tile.fill(sumexp, 0.0)
T.tile.fill(m_i, -2**30)

# Developer: T.Parallel 计算
for i, j in T.Parallel(block_M, block_N):
    acc_s[i, j] = acc_s[i, j] * sm_scale
```

### 场景 2：归约 + Developer 后处理

```python
# Expert: reduce（两种模式通用 API）
T.reduce_max(acc_s_ub, m_i, tmp_ub, dim=-1)

# Developer: T.Parallel 后处理
for i in T.Parallel(block_M):
    m_i[i] = T.max(m_i[i], m_i_prev[i])
    m_i_prev[i] = T.exp(m_i_prev[i] - m_i[i])
```

### 场景 3：精度转换中转

```python
# Developer: float32 累加计算
for h_i, j in T.Parallel(block_M, block_N):
    acc_o[h_i, j] = acc_o[h_i, j] / sumexp[h_i]

# Expert: 精度转换（T.Parallel 不支持跨类型赋值）
T.copy(acc_o, acc_o_half)  # float32 → float16
T.copy(acc_o_half, Output[...])
```

### 场景 4：Expert 广播 + Developer 计算

```python
# Expert: 显式广播
T.tile.broadcast(buf_2d, sumexp, tmp_ub)

# Expert: 向量乘加
T.tile.axpy(buf_2d, work_ub, sm_scale)
```

---

## 完整混合模式示例：Flash Attention 核心循环

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}

# Expert 初始化
T.tile.fill(acc_o, 0.0)
T.tile.fill(sumexp, 0.0)
T.tile.fill(m_i, -2**30)

for k in T.Pipelined(T.ceildiv(seq_len, block_N), num_stages=2):
    # Cube: GEMM（Developer API）
    T.copy(K[...], k_l1)
    T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
    T.copy(acc_s_l0c, workspace_1[cid, :, :])

    # Vector: Online Softmax（混合）
    T.copy(workspace_1[cid, vid * half_M:...], acc_s_ub)
    T.copy(m_i, m_i_prev)

    # Developer: T.Parallel 符号计算
    for i, j in T.Parallel(half_M, block_N):
        acc_s_ub[i, j] = acc_s_ub[i, j] * sm_scale

    # Expert: 归约
    T.reduce_max(acc_s_ub, m_i, tmp_ub, dim=-1)

    # Developer: T.Parallel 更新
    for i in T.Parallel(half_M):
        m_i[i] = T.max(m_i[i], m_i_prev[i])
        m_i_prev[i] = T.exp(m_i_prev[i] - m_i[i])

    for h_i, j in T.Parallel(half_M, block_N):
        acc_s_ub[h_i, j] = T.exp(acc_s_ub[h_i, j] - m_i[h_i])

    T.reduce_sum(acc_s_ub, sumexp_i_ub, tmp_ub, dim=-1)

    for i in T.Parallel(half_M):
        sumexp[i] *= m_i_prev[i]
        sumexp[i] += sumexp_i_ub[i]

    # Developer: 更新历史 acc_o
    for h_i, j in T.Parallel(half_M, dim):
        acc_o[h_i, j] = acc_o[h_i, j] * m_i_prev[h_i]

    # ... softmax @ V 后续步骤 ...
```

---

## 注意事项

1. 混合模式下 **pass_configs 必须使用 Developer 配置**（4 个全开）
2. `T.tile.fill` 和 `T.reduce_*` 是最常见的混合使用 API
3. 精度转换（float32→float16）通常通过 `T.copy` 实现，不需要 `T.tile.cast`
4. 混合模式代码的调试方法与 Developer 模式一致（`T.printf`, `T.dump_tensor`）
