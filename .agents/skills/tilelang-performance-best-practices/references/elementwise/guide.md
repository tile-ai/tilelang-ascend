# Elementwise、向量化与融合

## 选择实现

以 [单缓冲 Add](../../../../../examples/elementwise/elementwise_add.py) 和目标同类 kernel 为起点。保留既有模式：Developer 可用 `T.Parallel`，Expert/混合模式按 [Tile API](../../../../../tilelang/language/ascend_tile.py) 的 `T.tile.*`。示例存在仅证明参考结构，不代表本轮设备已测试或比当前 baseline 快。

逐 case 先记录 [任务树与数据流](tiling_task_vector_search.md)，再按连续搬运、任务粒度、向量化、融合和 [流水](double_buffer_design.md) 建候选。不用“elementwise 必然 memory-bound”代替 profiling。

## 从逐行循环到整 tile

当循环各行独立而每行反复发射 sub/mul/div 等 tile 指令，可先将每行参数广播，再一次处理整块。例如逐行 `row -= row_max` 可改为 `[rows,1] → [rows,cols]` broadcast 加整 tile sub。

收益是减少 scalar 地址、循环和指令下发；代价是广播本身及完整参数 UB。若 UB 不足，可跨若干行批处理而非全量 broadcast。真正有前缀依赖的 scan 不能直接去掉循环，改用 [正交轴向量化](../scan/orthogonal_axis_vectorization.md)。

## 融合先证明数学和类型语义

当前 `T.tile.axpy(dst, src, alpha)` 的契约是 `dst = alpha*src + dst`。正确候选是将 `tmp=alpha*x; y=y+tmp` 改为 `axpy(y,x,alpha)`。旧文把 `x=x*scale; x=x-m` 改成 `axpy(x,m,scale)` **不等价**，新表达式会是 `x+scale*m`；不得沿用。

| 原表达式 | 候选 | 核验 |
|---|---|---|
| `acc += x*w` | `T.tile.mul_add_dst(acc,x,w)` | dst 是输入累加值，alias、dtype、舍入 |
| `dst += alpha*src` | `T.tile.axpy(dst,src,alpha)` | scalar 参数、原 dst 内容 |
| `max(x,0)` | `T.tile.relu` | NaN、±0、dtype 和边界语义 |
| 分段负半轴缩放 | `T.tile.leaky_relu` | 阈值、斜率及端点 |
| `1/sqrt(x)` | `T.tile.rsqrt` | 负数/0/非有限、近似误差 |
| `y/sqrt(x)` | `rsqrt(x)` 后乘 y | 运算顺序及原容差 |

相似名称不是等价证明；不假定 fast-math 自动融合。对 select/clamp/compare 等组合先查当前 API 和生成代码。低精度长链与超越函数按算子需求用 fp32，最终边界转换；降低内部精度只能作为保持原 reference/容差的实验。

旧“归一化只能 div、不允许倒数乘”属于局部精度经验：两种路径可能舍入不同，当前测试决定；同样，自动同步不保证任意 in-place alias 合法，手动路径交替 dst 也不自动保证依赖正确。

## 短记录与共享字段

多个字段共享同一短记录、有效区域和索引时，比较一次遍历联合处理与多轮独立处理：记录重复索引、load/store、临时结果与 live state。外提不变量，复用已物化且可恢复的中间值；同时计算增加的读取与减少的重算。

参数梯度等 partial 若消费方、归约轴和生命周期相同，可打包连续输出以减少 GM buffer/copy/host reduction 次数。是否值得取决于端到端路径；不能只减少 Python tensor 数量就判定优化有效。

## 验证

覆盖原所有 dtype、允许 shape、尾块、极值/NaN 与输出顺序。固定 tile/task/stage 比较指令变化，再评估交互组合；生成代码、实际 latency、内存压力和编译体积共同裁决。原已授权的优化无需因普通融合额外增加确认流程。
