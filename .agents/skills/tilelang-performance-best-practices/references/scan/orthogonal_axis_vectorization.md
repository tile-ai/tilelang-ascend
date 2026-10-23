# 正交轴向量化与 compare/select

## 识别结构

适用于 `for independent_row: for scan_pos: scalar_update`：行间无依赖、扫描轴有真实前缀依赖、内层由标量 if/GetValue/SetValue 实现。把正交行并成连续 lane，每个扫描步用向量 compare/select/min 更新所有 lane；保留扫描步的顺序。不同于已有 tile 操作的逐行循环，它需要联合布局和条件状态设计。

先尝试原 `[M,R,N]` 布局中连续 N 轴；必要时物化 `[scan,parallel]`。比较一次输入重排、扫描、输出恢复的完整时间，避免两次互相抵消的 transpose+contiguous。N 太窄时转置可能有收益，不能把某经验 N=64 当所有 dtype 的启用阈值。

## 当前 API 能力

[compare 定义](../../../../../tilelang/language/ascend_tile.py) 明确支持 fp16/fp32/int32，mask 为 bit-packed int8/uint8，输入总字节需 256B 对齐，int32 compare 仅支持 EQ。选择模式与具体限制参见 [compare/select API](../../../tilelang-custom-skill/tilelang-api-best-practices/references/api-compute.md)。先确定计算 dtype 和对齐，再选择 N 与 UB 预算。

| 模式 | 何时评估 | 必查事项 |
|---|---|---|
| `VSEL_CMPMASK_SPR` | 已验证可用 dtype、单次完整 mask 可覆盖全部有效 lane 的小 tile | 不能跨多次 compare 后只使用最后一块 SPR；compare/select 间不得破坏 mask |
| `VSEL_TENSOR_TENSOR_MODE` | 大 tile、需保存多个 mask 或整数选择路径 | mask buffer 大小、bit 布局、生命周期及 scratch |

旧 fp32 路径用 64 lane 划分小 tile，N>64 时不能照搬其 SPR 快路；int32 选择使用已验证 tensor 模式。fp16 的输入对齐粒度与 fp32 不同，不能将“64 lane”泛化到全部类型。mask 调试按一位一元素解释，额外分配更大的 buffer 不会改变其 bit 语义。

## int32 的精确路径

默认必须保持公开 int32 全值域，不能因示例数据较小而默认 cast 到 fp32。fp32 能精确表示连续整数范围 `[-2^24,2^24]`；超过范围可能把不同值合并，导致 value、tie-break 和 index 错误。只有调用方已有明确值域保证，才准入转换候选；临时全量 min/max 扫描所得保证也要计入端到端成本。

可考虑精确 int32 `min(old,curr)` 加 EQ 构造条件，例如 `EQ(curr,min(old,curr))` 对应 `curr<=old`；先证明当前 int32 min 路径、选择模式、tie-break 及前后状态保留。需要“相等保留旧 index”时不能直接使用 <= 更新。若当前 API 不能精确表达，保留正确 fallback，不强制向量化所有 dtype。

index 若借 fp32 承载，同样必须证明所有索引 <=2^24；否则用精确整数表示。最终输出 int64 的转换留在 NPU 合法路径，并检查 int32 中间索引本身的范围。整数 padding 用 `iinfo` 对应幺元，不能写 ±inf。

## NaN、tie 与更新顺序

浮点 `EQ(curr,curr)` 可作为 NaN 检测候选，但需在目标 backend 验证 NaN/Inf 行为；不把旧“所有比较（含 NE）对 NaN 返回 False”当通用定律，也不默认任意 `min` 都满足算子要求的 NaN 传播。

先依据公开 reference 写出状态表：正常 curr/正常 old、curr NaN、old NaN、两者 NaN、相等值、±0、±inf。明确 first/last occurrence，value 与 index 同步更新：

```text
normal_update = 对旧value与curr的合法比较（< 或 <= 由tie约定决定）
nan_update = 按reference对curr_nan/old_nan定义
update = 对normal与NaN条件的合成
new_value = select(update, curr, old_value)
new_index = select(update, curr_index, old_index)
```

真实实现可用多次 compare/select，也可用已验证 min+index 更新，但须保证比较读取的是所需旧状态。SPR 只能保留当前 compare 结果；先生成两个 mask 再以 SPR 消费第一个会误用最后 mask，需紧邻 compare/select 或用 tensor mask。

默认 kernel 内处理合法 NaN。`bool(torch.isnan(x).any())` 是设备扫描+host 同步，不是“零运行时开销”的 JIT 参数；只有有限值合约或已有元数据才可免费选择无 NaN 版本，其他情况 A/B 比较完整链。整数无 NaN 可以由 dtype 静态确定。

## 回归与 fallback

检查 value 与 index、所有支持 dtype、±2^24 邻域、int32 极值、大索引、mask 边界、重复最小值、首/中/末多个 NaN、±inf 和跨 chunk。正确标量/UB-local fallback 可以保留；不能为消除 fallback 而降低值域、放宽精度或额外偷跑 host reference。
