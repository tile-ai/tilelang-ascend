# Broadcast、参数加载与驻留

## 先确认 shape 与实际地址

按 [当前 broadcast DSL](../../../../../tilelang/language/ascend_tile.py) 和 [API 计算参考](../../../tilelang-custom-skill/tilelang-api-best-practices/references/api-compute.md) 核对支持的扩展轴、rank、dtype、scratch 与 slice。旧归一化经验中的末轴扩展是一个稳定候选结构，不据此否定当前 API 的其他合法用法。

连续小输入可在任务循环外加载一次；多轴广播先合并逻辑连续维，区分 outer / broadcast / inner。内层连续时可让一个 task 复用源 slice 写多个输出片段；不连续维需显式地址映射，不能假定任意 N 维广播都可一条 DMA。

## 统计量与 GM 参数采用不同 staging

| 来源 | 候选数据流 | 注意 |
|---|---|---|
| 所有行共享的统计标量 | `[1] → fill [rows,1] → broadcast [rows,block_S]` | rows 为真实有效行数 |
| 每行独立统计量 | `[rows] / [rows,1] → 合法形状整理 → broadcast` | 按当前 keepdim/shape 规则，不默认 copy 跨 rank |
| GM per-row gamma/beta | 有效 1D slice → aligned staging → cast → padded broadcast → 有效行 | rows_padded 与 rows 区分 |

旧 GroupNorm 的五步参数路径应保留为结构参考：

1. host/签名将参数表达为 `[G, rows]`，对当前组搬 `param[g,0:rows]`；避免误用二维 slice 改变 DMA 解释。
2. staging 为 `[rows_padded,1]`，按实际 dtype/cast 对齐分配。需要预填充时 fill 在 copy 前；copy 的 `pad_value` 与填充值一致或具有明确覆盖范围。
3. cast 使用完整、已初始化且满足指令约束的 `rows_padded` 范围，低精度统计/计算通常提升 fp32。
4. 广播为 `[rows_padded,block_S]`，预算包含完整临时 buffer。
5. 仅保留/消费有效 `[rows,block_S]` 区域，循环外常驻。`gamma_bc_full` 只有在最后 consumer 完成后才可复用。

`rows_padded=ceildiv(rows,16)*16` 是旧 fp16/bf16 参数路径的示例，不能代替所有 dtype 的约束。`T.tile.fill` 从 scalar buffer 传值与 `T.copy` 重新解释 shape 不等价；出现 `StructuralEqual` 或 broadcast mismatch 时查 API，不机械换成跨 rank copy。

## 优化选择

同时比较逐行指令、整 tile broadcast、多行批处理与参数驻留。较大 broadcast 倍率可减少重复 GM 读，较小时 staging 固定成本可能更高；每核各自加载参数的成本也要计入。显式 GM padding 与 kernel 内 fill+valid copy 按总流量比较。

常驻参数保持单版本；流式输入/输出按 [生命周期](../common/ub_resident_design.md) 决定版本。参数广播与后续输入 copy 的 storage 若发生重叠，先修规划/依赖，不以每轮全局 barrier 掩盖。

## 验证

覆盖 size=1 轴、多轴、标量、rows 对齐边界、最后一组、不同 broadcast 倍率、所有 dtype 和允许 stride。唯一编号参数可检查广播方向，非均匀 gamma/beta 可暴露错位；全 1 参数不足以验证映射。记录生成 DMA、重复读字节、指令数、UB 及完整 latency。
