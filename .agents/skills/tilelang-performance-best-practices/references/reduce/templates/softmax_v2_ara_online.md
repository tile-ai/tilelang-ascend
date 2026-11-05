# ARA Online Softmax

## 适用条件

输入逻辑布局为 `[A1, R, A0]`，沿中间轴 `R` 做 softmax；`R × tile_A0` 无法整体进入 UB，并且少一次完整输入遍历能够抵消 online 状态更新开销。

## 数据流

每个 task 唯一持有一个 `(a1, a0_tile)`，`running_max[A0]` 与 `running_sum[A0]` 使用 fp32 并常驻 UB。第一遍沿 R chunk 流式读取：

```text
chunk_max = max(x_chunk, axis=R)
m_new     = max(m_old, chunk_max)
l_new     = l_old * exp(m_old - m_new)
          + sum(exp(x_chunk - m_new), axis=R)
```

等价的分块合并形式是：

```text
chunk_sum = sum(exp(x_chunk - chunk_max), axis=R)
l_new = l_old * exp(m_old - m_new) + chunk_sum * exp(chunk_max - m_new)
```

第二遍重新读取输入并输出 `exp(x - running_max) / running_sum`。尾 R chunk 的 max lane 填负无穷、sum lane 填零；A0 尾部使用 mask，不能让 padding 改变状态。

实现从 [当前 online 示例](../../../../../../examples/softmax/example_online_softmax.py) 的 T.reduce_max/sum 与 T.tile.exp 组合派生并适配 R 轴；若 strided R chunk 不能形成有效 burst，应优先融合 layout conversion，而不是逐元素 GM 访问。

## 精度与性能门禁

- 与 fp32 `torch.softmax` 比较，覆盖 `R=tile±1`、极长 R、重复最大值、极大正负值和接口要求的 NaN/Inf 语义。
- 比较 online 两遍与 recompute 三遍的完整 GM 字节和 latency。
- 报告 UB 中两个 fp32 状态数组、输入/输出 tile、buffer versions 和安全余量。
- 只有目标后端的定向精度测试全通过且端到端更快时才选择 online。

## 实现状态

状态为 `DESIGN_ONLY`；链接示例只提供局部结构依据，不证明本策略已有端到端 kernel。实现后须完成上述精度与性能门禁。
