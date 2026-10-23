# Scale / Bias 等随路小块合并搬运

## 适用条件

K 循环中重复发起很短的 scale、bias 或查表数据 DMA，目标数据在 GM 中可聚合为
连续且合法的区间，已有证据显示搬运发射/地址计算成本值得处理。若分片不连续、
聚合后的格式与计算不兼容、L1 预算不足，先排除此候选或缩小合并范围。

## 方案

将 g 轮随路数据一次搬入，在接下来的 g 轮按偏移使用；下一组合并前等待上一组最后
消费者归还。需要与主数据并发时单独核对其 stage 数量和 ownership。

```text
group_begin = floor(k/g)*g
load metadata[group_begin:min(group_begin+g, count)] once
use metadata_slot[k-group_begin] with data_tile[k]
```

旧文档的“2 KiB×8=16 KiB”是组合示例，不支持它同时提出的“至少20 KiB”硬阈值。
将搬运长度作为平台/shape 相关候选；不承诺带宽必然从 50%–70% 到 80% 以上。

## 兼容性审计

- 合并布局须保留原 scale 分组、量化轴与 dtype，不能把任意 scale 当成 GEMM B。
- 检查最后不足 g 轮的合法范围，避免完整组读取越过 GM。
- 偏移切片要满足实际 DMA、L1 和消费指令的地址/长度对齐。
- 合并后的 buffer 计入 [L1 总预算](tiling_design.md)；更大不一定能保持原 pipeline。
- 如果数据在多输出 tile 间也复用，可与 [Full-load](fullload_design.md) 联合考虑，
  但把两项变化及各自假设分清楚。

后续执行阶段比较减少的 DMA 次数、增大的驻留量与完整算子耗时。本文件保留旧 guide
§2.5 的通用优化方向，没有当前仓库端到端 scale-coalescing 新测量。
