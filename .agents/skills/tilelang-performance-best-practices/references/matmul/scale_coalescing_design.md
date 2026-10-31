# Block-scale 合并载入

## 适用条件

scale/bias/LUT 沿 K 连续且单次片段过小，copy 固定开销明显。


K 循环中重复发起很短的 scale、bias 或查表数据 DMA，目标数据在 GM 中可聚合为
连续且合法的区间，已有证据显示搬运发射/地址计算成本值得处理。若分片不连续、
聚合后的格式与计算不兼容、L1 预算不足，先排除此候选或缩小合并范围。

## TileLang 实现流程

扩大 scale 的 L1 tile，一次搬入多个 K iteration 对应的数据；内层按静态 offset 选择当前 scale 子片。先定位目标算子已有的 scale 消费路径与量化接口，再确定两侧 scale 缓冲和合并块长度。

基础实现参考 [当前 GEMM 示例](../../../../../examples/gemm/example_gemm.py) 与 [显式 MMA 示例](../../../../../examples/gemm/example_gemm_intrinsic.py)：按当前 B 的物理布局设置 transpose_B，使用 T.gemm_v0 或 T.mma；浮点 GEMM 的 L0C 使用 fp32，并且只在第一个有效 K tile 设置 init=True。输出 tile 的核数不超过独立任务数；M/N/K 尾块必须走已验证的 padded-copy 或专用 fallback。

### 方案

将 g 轮随路数据一次搬入，在接下来的 g 轮按偏移使用；下一组合并前等待上一组最后
消费者归还。需要与主数据并发时单独核对其 stage 数量和 ownership。

```text
group_begin = floor(k/g)*g
load metadata[group_begin:min(group_begin+g, count)] once
use metadata_slot[k-group_begin] with data_tile[k]
```

“2 KiB×8=16 KiB”是组合示例，不能据此设定“至少20 KiB”的硬阈值。
将搬运长度作为平台/shape 相关候选；不承诺带宽必然从 50%–70% 到 80% 以上。

## 精度门禁

每个 K block 的 scale 必须与 operand block 精确对齐，覆盖 K 尾部和不同 scale dtype。

所有输入 dtype 与输出 dtype 分开测试。覆盖 M/N/K 的 tile±1、长 K 消除误差、正负抵消、大小量混合、0、NaN/Inf 契约。浮点 GEMM 的任何 K 分片的 partial 和最终归约保持 fp32；不得以放宽容差掩盖累加顺序或输出转换错误。

### 兼容性审计

- 合并布局须保留原 scale 分组、量化轴与 dtype，不能把任意 scale 当成 GEMM B。
- 检查最后不足 g 轮的合法范围，避免完整组读取越过 GM。
- 偏移切片要满足实际 DMA、L1 和消费指令的地址/长度对齐。
- 合并后的 buffer 计入 [L1 总预算](tiling_design.md)；更大不一定能保持原 pipeline。
- 如果数据在多输出 tile 间也复用，可与 [Full-load](fullload_design.md) 联合考虑，
  但把两项变化及各自假设分清楚。

后续执行阶段比较减少的 DMA 次数、增大的驻留量与完整算子耗时。端到端 scale-coalescing 的具体收益待验证。

## 性能门禁

扫描合并倍率，记录 MTE2 指令数、有效带宽和 L1 占用；不使用固定 20KB 阈值代替实测。

对每个候选运行编译、目标后端的定向精度测试和统一 benchmark。报告 latency、TFLOPS、Cube/MTE2 时间、GM 字节、L1/L0/UB 占用、核利用率和生成代码体积。只有完整算子端到端更快且没有精度回退时进入 dispatch。
