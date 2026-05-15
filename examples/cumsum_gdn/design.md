# cumsum_gdn 设计文档

## 1. 概述

### 1.1 算子名称

`chunk_cumsum`（`examples/cumsum_gdn/example_cumsum.py`）

### 1.2 功能描述

将输入序列按 `chunk_size=C` 分块，在每个 chunk 内独立计算前缀和。

- `reverse=False`：普通前缀和
- `reverse=True`：反向前缀和
- `use_fragment=True`：保留一条额外中间缓冲路径，行为与主路径等价

输入支持两种布局：

- `head_first=True`：`(B, H, L)`
- `head_first=False`：`(B, L, H)`

### 1.3 数学公式

对任意 chunk 内位置 `i`：

正向：

$$
y_i = \sum_{j=0}^{i} x_j
$$

反向：

$$
y_i = \sum_{j=i}^{C-1} x_j
$$

实现上反向模式通过

$$
\text{suffix}_i = \text{total} - \text{prefix}_i + x_i
$$

完成转换。

---

## 2. 实现结论

### 2.1 编程模式

当前实现采用：

- **Developer 模式 fast path**
- **host wrapper fallback**

这里的 Developer 模式并不要求必须使用 `T.cumsum` 或 `T.Parallel`。当前 fast path 仍然使用手写 `for` 循环，但它已经符合 Developer 模式的核心特征：

- 使用 `T.alloc_shared`
- 使用自动同步
- 使用自动内存规划
- 不显式写 `T.Scope("V")`
- 不手写 `barrier/set_flag/wait_flag`

### 2.2 fast path 与 fallback 的边界

只有以下条件同时满足时，才进入 TileLang kernel fast path：

- `head_first=True`
- `H % 2 == 0`
- `L % C == 0`

其他场景统一回退到 PyTorch reference：

- `head_first=False`
- odd-H
- 尾块（`L % C != 0`）

这样做的目的不是功能缺失，而是优先保证当前 Ascend 路径的稳定性与 correctness。

---

## 3. API 状态说明

### 3.1 `T.cumsum` 的真实状态

`T.cumsum` 当前**不是未导出状态**。

公开 API 来自：

- `tilelang/language/__init__.py`
- `tilelang/language/reduce.py`

同时，`src/op/reduce.cc` 中已经存在针对 `shared/shared.dyn` 的 lowering。

因此，这个 example 当前**没有**使用 `T.cumsum`，并不是因为“API 不存在”或“完全没有后端”，而是因为：

1. 当前算子需要显式处理 `reverse`
2. `use_fragment` 路径也要维持现有语义
3. 当前 Ascend shared layout 推导对这类临时 buffer 更稳妥的写法，仍然是显式循环

### 3.2 为什么 shared buffer 写成二维

当前 Ascend shared-layout inference 在部分路径上会对 shared buffer 注入默认 zN layout。
这一布局生成逻辑按二维 `(i, j)` 处理，因此逻辑上一维的 shared buffer 若直接写成 `[C]`，会在 lowering 阶段触发异常。

因此当前实现将逻辑上一维缓冲显式 materialize 成二维退化 tile：

- `g_ub`: `[1, C]`
- `s_ub`: `[1, C]`
- `total_ub`: `[1, 1]`
- `fragment_ub`: `[1, C]`

这只是实现层面的稳定性处理，不改变算子语义。

---

## 4. Kernel 设计

### 4.1 核心映射

- `chunk_num = ceildiv(L, C)`
- `VEC_NUM = 2`
- `h_block_num = H // VEC_NUM`
- grid 大小：`B * h_block_num * chunk_num`

每个 kernel block 负责：

- 一个 batch
- 一个 head 分片
- 一个 chunk

### 4.2 片上缓冲

| Buffer | Shape | 作用 |
|---|---:|---|
| `g_ub` | `[1, C]` | 当前 chunk 输入 |
| `s_ub` | `[1, C]` | 当前 chunk 输出 |
| `total_ub` | `[1, 1]` | reverse 模式总和 |
| `fragment_ub` | `[1, C]` | `use_fragment=True` 中间缓冲 |

以 `float32` 为例：

- `C=32` 时总占用约 `388B`
- `C=64` 时总占用约 `772B`

均远小于 A2/A3 / 910B 的 `192KB` UB/shared 预算。

### 4.3 数据流

fast path 数据流为：

```text
GM[G] -> shared[g_ub]
shared[g_ub] -> shared[s_ub]/shared[fragment_ub]  (for-loop cumsum)
shared[s_ub] -> GM[S]
```

反向模式会额外先规约得到 `total_ub[0, 0]`，再做 `total - prefix + input` 转换。

---

## 5. 同步与 pass_configs

当前 fast path 使用：

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

含义：

- `TL_ASCEND_AUTO_SYNC=True`：由编译器插入同步
- `TL_ASCEND_MEMORY_PLANNING=True`：由编译器负责 shared 侧内存规划

因此本算子不需要手动写同步原语。

---

## 6. 对齐主仓语义的方式

与主仓 example 的语义对齐方式如下：

- `reverse`：kernel 内显式支持
- `use_fragment`：kernel 内显式支持
- `head_first=False`：wrapper 级 fallback 保证语义一致
- odd-H：wrapper 级 fallback 保证语义一致
- 尾块：wrapper 级 fallback 保证语义一致

该实现可概括为：

- **语义完整**
- **fast path 只覆盖对齐子集**

---

## 7. 验证结论

当前 example 自测覆盖了：

- 正向 / 反向
- `use_fragment=True/False`
- `head_first=True/False`
- odd-H
- 非整除尾块

验证结果表明：

- 对齐场景走 Developer-mode fast path
- 非对齐场景走 PyTorch reference fallback
- 两条路径在 example 自测中与 reference 对齐

---

## 8. 当前实现边界

当前实现具有以下边界：

1. 当前不是“所有场景都由 Ascend kernel 直接处理”
2. 当前是“Developer fast path + wrapper fallback”
3. `T.cumsum` 已存在公开 API 与 shared 路径 lowering，但本算子暂未直接采用

以上三点用于说明当前实现的代码路径覆盖范围，不影响本文档中的算子语义定义。
