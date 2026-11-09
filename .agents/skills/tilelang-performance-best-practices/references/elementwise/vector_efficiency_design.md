# Elementwise SIMD 与融合

## 目标与适用条件

用于连续逐元素、量化转换、激活和简单复合表达式。

## TileLang Ascend 实现

连续规则路径使用当前模式的 T.Parallel/T.tile 原语；bf16/fp16 超越函数先转 fp32。把只使用一次的中间结果保存在寄存器或同一 UB tile，融合相邻表达式以消除 GM round-trip。复杂索引按当前 DSL 已验证的访问路径处理。

仿射链 `a*x+b` 将当前支持且数学等价的融合原语与 mul+add 作为不同 lowering 候选；不假设 fast-math 会自动融合，从当前生成代码确认指令，并分别验证精度和同口径性能。

短记录可把多个 token 打包到一个向量，按当前已验证的索引/Vector 重排路径处理，并将常量与索引模式移出热循环。常见整 tile 可由 Python wrapper 固化 shape 和展开数，其他 shape 保留动态尾块。完整结构见 [索引短记录变换](indexed_short_record.md)。

实现沿用当前仓库 T.Kernel 的网格和 `(cid, vid)` 映射。纯向量任务从已确认的可用 AIV 核数及 launch 映射取核数，并以 min(核数, 独立任务数) 限制空核。GM 与 UB/L1 之间使用 T.copy；连续规则计算使用当前模式的 T.Parallel 或 T.tile 原语；跨 tile 任务使用当前 T.Persistent、Fixed Core 核内循环或已验证的 T.Pipelined，多版本方式沿用当前编译器支持的流水或显式槽位所有权。

### 从逐行循环到整 tile

当循环各行独立而每行反复发射 sub/mul/div 等 tile 指令，可先将每行参数广播，再一次处理整块。例如逐行 `row -= row_max` 可改为 `[rows,1] → [rows,cols]` broadcast 加整 tile sub。

收益是减少 scalar 地址、循环和指令下发；代价是广播本身及完整参数 UB。若 UB 不足，可跨若干行批处理而非全量 broadcast。真正有前缀依赖的 scan 不能直接去掉循环，改用 [正交轴向量化](../scan/orthogonal_axis_vectorization.md)。

### 融合的数学与类型语义

当前 `T.tile.axpy(dst, src, alpha)` 的契约是 `dst = alpha*src + dst`。正确候选是将 `tmp=alpha*x; y=y+tmp` 改为 `axpy(y,x,alpha)`。将 `x=x*scale; x=x-m` 改成 `axpy(x,m,scale)` **不等价**，后者计算 `x+scale*m`，不能替换。

| 原表达式 | 候选 | 核验 |
|---|---|---|
| `acc += x*w` | `T.tile.mul_add_dst(acc,x,w)` | dst 是输入累加值，alias、dtype、舍入 |
| `dst += alpha*src` | `T.tile.axpy(dst,src,alpha)` | scalar 参数、原 dst 内容 |
| `max(x,0)` | `T.tile.relu` | NaN、±0、dtype 和边界语义 |
| 分段负半轴缩放 | `T.tile.leaky_relu` | 阈值、斜率及端点 |
| `1/sqrt(x)` | `T.tile.rsqrt` | 负数/0/非有限、近似误差 |
| `y/sqrt(x)` | `rsqrt(x)` 后乘 y | 运算顺序及原容差 |

相似名称不是等价证明；不假定 fast-math 自动融合。对 select/clamp/compare 等组合先查当前 API 和生成代码。低精度长链与超越函数按算子需求用 fp32，最终边界转换；降低内部精度只能作为保持原 reference/容差的实验。

归一化使用 div 或倒数乘可能产生不同舍入，需按当前精度测试选择；同样，自动同步不保证任意 in-place alias 合法，手动路径交替 dst 也不自动保证依赖正确。

## 精度门禁

严格保持 reference 的运算顺序、饱和/舍入模式和输出 cast 边界。量化必须覆盖零 scale、极值与非有限输入契约。

必须覆盖最小 shape、常见 shape、最大 shape、tile-1/tile/tile+1、非 32B 尾块、0、正负极值以及接口规定的 NaN/Inf 语义。不得通过扩大容差、降低 reference 精度或跳过 case 获得通过。

## 性能门禁

测向量 lane 利用率、指令数量、GM 字节和 spill。融合导致寄存器压力或代码膨胀时拆分。

先用目标仓库既有后端与命令运行定向精度测试，再运行相关完整测试套；全部通过后，在相同输入、dtype、warmup、repeat、设备和并发条件下测量性能。报告 kernel latency、有效 GM 带宽、UB 占用、stage 数和基线差异。

覆盖原所有 dtype、允许 shape、尾块、极值/NaN 与输出顺序。固定 tile/task/stage 比较指令变化，再评估交互组合；生成代码、实际 latency、内存压力和编译体积共同裁决。

## 可执行代码与证据

可执行结构见 [Vector Add](../../../../../examples/elementwise/elementwise_add.py) 和 [流水版本](../../../../../examples/elementwise/elementwise_add_pipeline.py)。
