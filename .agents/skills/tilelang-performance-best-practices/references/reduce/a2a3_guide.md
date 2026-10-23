# Reduction、Norm 与多行 tile

## 按结构选策略

先确认归约轴、输出 owner、累加 dtype、数值语义和真实 source stride。整轴可驻留时比较 full-load；超过容量时比较分块状态、recompute 与 [online 遍数融合](vector_reduce_pass_fusion.md)；输出任务太少且归约极长时可评估 partial+final 的两阶段拆分，计入 workspace、所有 launch 和加法顺序变化。

前端与后端限制优先查 [T.reduce_sum](../../../../../docs/api_docs/T.reduce_sum.md)、[row reduce 规范](../../../../../docs/ascend/row_reduce.md)、[dtype 覆盖测试](../../../../../testing/python/language/test_tilelang_ascend_language_reduce_dtype_coverage.py) 和 [2D reduce 测试](../../../../../testing/python/language/test_tilelang_ascend_language_reduce2d_v2.py)。旧局部失败不构成全平台 API 禁令：当前文档支持 2D dim=0/-2 与 dim=1/-1，也支持匹配 reduced/keepdim 的输出；其中 dtype、BufferRegion、clear=False 等有后端差异。

## 多行 tile 的三层决策

### 1. Layout

把外层逐行循环改为 `[rows,block_S]` 前，写出真实 GM stride、二维 copy 的 src/dst pitch 与有效区。`[N,C,S]` reshape 为 `[N,G,rows,S]` 只是暴露组语义，**不会将 row stride 从 S 改为 block_S**。有 gap 的矩形是否支持取决于当前搬运 lowering；不能声称加一维就自动变连续。

用 `arange`/唯一行列编码验证多行 copy、跨组、跨 batch、双尾块；小样本全相同值无法证明布局正确。所有输出写回同样检查 pitch。MIX 下按实际 vid 分片或唯一 owner 处理，不一律 `vid==0`。

### 2. 维度与资源

分别记录 `rows`、`rows_padded`、`block_S`、真实 `S_orig`、必要的 `S_padded`、input/cal/output dtype。rows 扩大使所有 tile 成本乘 rows，旧固定 block_S=256 会在大 cpg 下超限；按 [预算](../common/hardware_resource_discovery.md) 反推合法候选，不保留旧 `max_bs<=512` 的任意 cap，也不使用固定六份计算/输入 buffer 替代实际枚举。

rows_per_tile 与 stage 为独立轴：先比较单行/多行数据流，再比较流水。归约轴可放 UB 不等于一次只能处理一行；短记录充足时须评估多行连续 DMA，若任务不足、尾部或容量反而恶化则保留单行。

### 3. 计算模式

针对跨块的总 sum/sumsq，可比较以下两种：每块先归约再合并小状态；或逐块累积到 `[rows,block_S]`、循环外归约。后者减少归约指令但增加常驻量并改变求和顺序，只在统计语义等价且误差满足原容差时采用，不禁止所有循环内 reduce。

```text
跨行共享统计量：
  每块 valid 数据 → 累加 [rows,block_S]
  → 末轴归约 [rows] → 再归约 [1]
逐行独立统计量：
  每块 valid 数据 → 累加 [rows,block_S]
  → 末轴归约 [rows]（不能再把各行混合）
```

每块无效 lane 填幺元，任务开始时清零累加；平均值分母为真实 `rows*S_orig`，不能包含 padding。行输出还需满足当前 fp32 row-reduce 输出对齐/填充规范，不能只凭逻辑 `[rows]` 分配就假定合法。

旧文观察到某种 `[rows,1]`/dim=0 组合结果异常，保留为“复现具体 shape/target/IR 后诊断”的经验，不能替代当前合法接口。二步末轴归约是候选，非唯一正确路线。

## 逐元素 epilogue 与宽输出轴

每行归约后再写回整行时，广播参数循环外常驻；归约轴已满足实际对齐即可按精确长度预算，避免无理由 next-power-of-two 放大。可第二遍从 UB 重读原行，以少量重算换较小 live state。把每行不变的因子合并后再进入写回。

非归约连续轴宽、归约计算轻且小 DMA 很多时，合并连续输出 tile；记录 `outer*ceil(output_inner/tile_inner)`、DMA 次数、padding、waves 和版本 footprint。输出轴扩张不应改变归约轴或输出 owner，tile 与 stage 分别 A/B。

## 小状态与精度

固定小矩阵可跨多轮迭代驻留 UB，寄存器局部驻留需当前 lowering 证据；backward 只保存真正需要的快照。每组倒数复用属于候选，不能无验证改变累加和舍入。统计量、eps、softmax max/sum 与长链状态以 fp32 为起点，尾部 max/sum 用正确幺元。

测试短/长归约、rows=1/核数边界、非整块、抵消、大小量、NaN/Inf、公开 forward/backward；比较完整归约及 epilogue，不只比较被融合掉的一段。
