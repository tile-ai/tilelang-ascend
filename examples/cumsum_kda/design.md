# cumsum_kda 设计文档

## 1. 概述

### 1.1 算子名称

`chunk_local_cumsum_*` / `chunk_global_cumsum_*`
（`examples/cumsum_kda/example_cumsum_kda.py`）

### 1.2 功能描述

该 example 迁移自主仓 KDA/FLA cumsum 示例，包含四个 kernel：

- `chunk_local_cumsum_scalar`
- `chunk_global_cumsum_scalar`
- `chunk_local_cumsum_vector`
- `chunk_global_cumsum_vector`

覆盖两类语义：

- **local cumsum**：每个 chunk 内独立前缀和
- **global cumsum**：跨 chunk 累加，显式维护 `carry`

支持输入布局：

- `head_first=True`
- `head_first=False`

并保留 wrapper 层：

- `scale`
- `cu_seqlens`
- canonical `chunk_indices`
- 统一 dispatcher

---

## 2. 实现结论

### 2.1 编程模式

当前实现采用：

- **Developer 模式 fast path**
- **host wrapper fallback / slicing**

这里同样需要强调：Developer 模式允许手写 `for` 循环。当前实现虽然没有直接使用 `T.cumsum`，但它已经具备 Developer 模式的核心特征：

- 使用 `T.alloc_shared`
- 开启自动同步
- 开启自动内存规划
- 不显式写 `T.Scope("V")`
- 不手写同步原语

### 2.2 fast path 与 fallback 的边界

fast path 仅覆盖对齐 dense 场景。

scalar fast path 条件：

- `head_first=True`
- `H % 2 == 0`
- `SEQ_LEN % BT == 0`

vector fast path 额外要求：

- `S_DIM % BS == 0`

其余情况通过 wrapper 保证语义一致：

- `head_first=False`
- odd-H
- T 尾块
- S 尾块
- varlen (`cu_seqlens`)

---

## 3. API 状态说明

### 3.1 `T.cumsum` 的真实状态

和 GDN 一样，`T.cumsum` 当前**不是未导出状态**。

公开入口仍来自：

- `tilelang/language/__init__.py`
- `tilelang/language/reduce.py`

并且 `src/op/reduce.cc` 中已存在 `shared/shared.dyn` lowering。

当前 KDA example 没有直接采用 `T.cumsum`，主要原因是：

1. global 模式需要显式维护 `carry`
2. local/global 两类 kernel 的控制流不同
3. wrapper 还要兼顾 fallback、varlen 和 dispatcher

因此显式循环在当前版本里更直接，也更容易和主仓语义逐项对齐。

### 3.2 为什么部分 shared buffer 写成二维退化 tile

当前 Ascend shared layout 推导在部分路径上会给 shared buffer 注入默认 zN layout，而该 layout 构造按二维 `(i, j)` 处理。

因此逻辑上一维的 shared buffer 不直接写成：

- `[BT]`
- `[BS]`
- `[1]`

而是写成二维退化形式：

- scalar 输入/输出缓冲：`[1, BT]`
- scalar `carry` / `total`：`[1, 1]`
- vector `carry` / `total`：`[1, BS]`

这属于当前 Ascend lowering 稳定性约束，不改变算子语义。

---

## 4. Kernel 设计

## 4.1 local scalar

- shape: `(B, H, T)`
- grid: `chunk_num * B * (H // 2)`
- 每个 block 处理一个 batch-head 的一个 chunk

缓冲：

| Buffer | Shape |
|---|---:|
| `b_s` | `[1, BT]` |
| `b_o` | `[1, BT]` |
| `total_buf` | `[1, 1]` |

## 4.2 global scalar

与 local scalar 的区别在于：

- block 负责一个 batch-head 的所有 chunk
- 通过 `carry[0, 0]` 显式维护跨 chunk 累加值

缓冲：

| Buffer | Shape |
|---|---:|
| `b_s` | `[1, BT]` |
| `b_o` | `[1, BT]` |
| `carry` | `[1, 1]` |
| `b_ss_buf` | `[1, 1]` |

## 4.3 local vector

- shape: `(B, H, T, S)`
- `BS = min(32, 2 ** floor(log2(S_DIM)))`
- grid: `s_block_num * chunk_num * B * (H // 2)`

缓冲：

| Buffer | Shape |
|---|---:|
| `b_s` | `[BT, BS]` |
| `b_o` | `[BT, BS]` |
| `total_buf` | `[1, BS]` |

## 4.4 global vector

在 local vector 的基础上增加跨 chunk 的 `carry`：

| Buffer | Shape |
|---|---:|
| `b_s` | `[BT, BS]` |
| `b_o` | `[BT, BS]` |
| `carry` | `[1, BS]` |
| `b_ss_buf` | `[1, BS]` |

---

## 5. 数据流与计算

### 5.1 local 路径

```text
GM[s] -> shared[b_s]
shared[b_s] -> shared[b_o]     (for-loop prefix sum)
shared[b_o] -> GM[o]
```

若 `reverse=True`，则先累加出 chunk 总和，再做：

```text
suffix = total - prefix + input
```

### 5.2 global 路径

```text
for each chunk:
    GM[s] -> shared[b_s]
    shared[b_s] -> shared[b_o]         (chunk local prefix)
    shared[b_o] += carry
    write back
    carry += chunk_sum
```

这也是当前版本没有直接用 `T.cumsum` 替代 global kernel 的最主要原因。

---

## 6. 片上内存预算

A2/A3 / 910B 的 UB/shared 预算按 `192KB` 评估。

### 6.1 scalar（`BT=64`）

- `b_s`: 256B
- `b_o`: 256B
- `carry/total`: 4B
- `b_ss_buf`: 4B

总量约 `520B`，远小于 `192KB`。

### 6.2 vector（`BT=64, BS=32`）

- `b_s`: 8KB
- `b_o`: 8KB
- `carry/total`: 128B
- `b_ss_buf`: 128B

总量约 `16.5KB`，远小于 `192KB`。

---

## 7. pass_configs

当前 fast path 使用：

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

说明：

- `AUTO_SYNC`：由编译器处理同步
- `MEMORY_PLANNING`：由编译器处理 shared 侧内存规划

因此当前 fast path 仍然可以归类为 Developer 模式。

---

## 8. 与主仓语义的对齐方式

当前版本与主仓语义对齐方式如下：

- `reverse`：kernel 内支持
- `scale`：wrapper 后处理
- `head_first=False`：wrapper fallback
- odd-H：wrapper fallback
- T/S 尾块：wrapper fallback
- `cu_seqlens`：wrapper 按序列切片复用 dense 路径
- `chunk_indices`：仅接受 canonical 形式

因此当前版本的准确表述应为：

- **dense 对齐子集由 Ascend kernel fast path 处理**
- **其余语义由 wrapper/reference 保底**

---

## 9. 验证结论

当前 example 自测覆盖了：

- local/global
- scalar/vector
- 正向/反向
- `scale`
- `head_first=True/False`
- odd-H
- T/S 尾块
- varlen wrapper

当前通过结论可总结为：

1. fast path 对齐 dense 场景已跑通
2. wrapper fallback 场景与 reference 对齐
3. 当前代码已经达到“语义正确、边界清晰”的上库要求

---

## 10. 当前实现边界

仍需在 PR 描述中说明以下事实：

1. 当前不是所有场景都由 Ascend kernel 直接覆盖
2. 当前是 “Developer fast path + wrapper fallback”
3. `T.cumsum` 已导出且存在 shared 路径 lowering，但本 example 仍采用显式循环实现
4. global 路径的 `carry` 逻辑目前是手写的，不是 `T.cumsum` 一步替代

以上都是实现边界说明，不属于文档错误。
