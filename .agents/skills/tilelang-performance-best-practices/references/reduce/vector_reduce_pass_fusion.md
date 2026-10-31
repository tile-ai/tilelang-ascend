# 归约遍数融合：Online Softmax

## 适用条件

归约轴无法整体驻留、现实现至少两次完整输入扫描，且存在可证明的统计量合并公式。先从 [online softmax 示例](../../../../../examples/softmax/example_online_softmax.py) 派生；本文未重新验证该示例的当前设备性能。已经一遍完成或整行驻留时，不凭“online”名称追加复杂状态。

| 结构 | 候选变化 | 限制 |
|---|---|---|
| Softmax | max/sum/output 三遍 → online max+sum 与 output 两遍 | 修正成本、尾部、非有限语义 |
| LogSumExp | max/sum 两遍 → 一遍可合并状态 | 最终 max+log(sum)，溢出语义 |
| 稳定 L2 norm | scale/sumsq 两遍 → 一遍可合并 scale/sumsq | 使用独立推导的缩放平方公式，不能直接复用 exp |
| 带统计量归一化 | 合并可结合统计遍次 | 方差等须验证数值稳定算法，不能盲用 sum/sumsq |

## 数据流与数学

对每个输出行保持 fp32 `m,s`：

```text
初始 m=-inf, s=0
读入 chunk x
m_new = max(m, max(x))
s_new = s*exp(m-m_new) + sum(exp(x-m_new))
m,s = m_new,s_new
第二遍输出 exp(x-m)/s
```

也可计算 chunk 的 `m_tile/s_tile` 后用两侧缩放因子合并。公式对常规有限值稳定；空 chunk、全 `-inf`、`+inf` 与 NaN 不能仅靠初始化自动正确（如 `-inf - -inf`）；按原 reference 明确处理并验证。mask 在 max 前应用，padding 不参与统计。

## TileLang 实现顺序

第一遍每 chunk：合法 GM→UB → 必要 fp32 cast → row max → 更新 m → 计算旧 s 的修正 → broadcast m_new → sub/exp → row sum → 合并 s。常驻 `prev_max/prev_sum` 不随 tile 版本化；输入/临时量按 producer/consumer 依赖选择版本。

第二遍广播最终 m/s，重新搬入并 cast 输入，计算 exp/div，再**显式将最终计算 buffer cast/copy 到输出 buffer 后写 GM**。旧示意中计算 `a_cal` 后直接写未更新的 `a` 是不完整数据流，不能照抄。输出只在最终边界转换 dtype。

广播与 reduced/keepdim shape 以当前 API 为准。示例可能使用 `[rows,1]` 状态，不能将其机械换成不同 rank；参数广播和 scratch 见 [广播指南](../broadcast/broadcast_design.md)。

## 资源模型

第一遍包括输入 `rows*block_N*input_bytes`、fp32 工作 tile、broadcast max、tile max/sum、prev max/sum、修正临时量；第二遍包括 broadcast max/sum 和输出 buffer。编译器规划能否复用两个阶段的临时量需实际地址证据。

从当前 UB 有效容量扣除常驻/padding/scratch/版本再搜索 block_N；不能使用硬编码 192 KiB、固定份数或统一 16 元素对齐。多行批处理、流水与在线状态耦合，联合变化后重算。

## 验证与收益口径

固定原 reference 和容差，覆盖 N≫block_N、N=block_N±1、M=1、多行尾部、相同值、极值/抵消、mask 及 NaN/Inf 合约。检查各 chunk 后 m/s 或小例子的逐步结果。

三遍读变两遍读时，**输入读取字节**理论下降三分之一；若同 dtype 输出一遍，读写总流量从 4 份变 3 份，理论下降四分之一。修正 exp/mul 增加计算，cache、固定开销和流水也会影响时间；两者都不是实际 latency 收益承诺。

测完整 wrapper/kernel chain 的 latency、GM bytes、Vector/MTE 时间与 UB。Online 与 double buffer、向量化、Fixed Core 可组合，但每个组合都需验证。AXPY 仅在当前表达式真正满足 `dst+=scalar*src` 且精度通过时适用，不能把任意修正公式生硬替换为 axpy。
