**中文** | [English](benchmark_gated_mla_prefill.md)

# Gated MLA Prefill 基准

## 1. 算子

```
ql[b,t,h,:] = q_content[b,t,h,:] @ w_k[h]                              # [128] @ [128,512] -> [512]
s[b,t,h,k]  = scale * ( ql[b,t,h,:] · c[b,k,:] + qa[b,t,h,:] · e[b,k,:] )
p[b,t,h,k]  = softmax_k( s[b,t,h,k] )                                   # 仅在可见 k 上
z[b,t,h,:]  = Σ_k p[b,t,h,k] · c[b,k,:]                                 # [512], FP32
y[b,t,h,:]  = ( z[b,t,h,:] @ w_v[h]ᵀ ) * sigmoid( gate_logits[b,t,h,:] ) # [128]
```

- Kimi K3 Gated MLA 层的注意力块,从七个已投影输入到 W_O 之前。该层为 NoPE,
  64 宽的辅助 cache 携带普通 key 通道而非旋转位置编码;输出门在发布配置中启用。
- `scale = (D + E)^-0.5 = 192^-0.5`,与官方 `KimiMLAAttention.scaling` 一致。
- 可见性:key `k` 对 query 行 `t` 可见当且仅当 `k < min(query_start + t + 1, kv_len)`;
  超出有效 query 长度的行写出精确零。
- dtype:score、online softmax 状态与 `z` 为 FP32;`ql`、cache、权重、门与输出同输入 dtype。
- 参考模型:[Kimi-K3 `modeling_kimi_linear.py`](https://huggingface.co/moonshotai/Kimi-K3/blob/c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721/modeling_kimi_linear.py)
  (revision `c5d1dd4c`),生产配置 `mla_use_nope: true`、`mla_use_output_gate: true`、
  `num_attention_heads: 96`、`kv_lora_rank: 512`、`qk_nope_head_dim: 128`、
  `qk_rope_head_dim: 64`、`v_head_dim: 128`。模型源码仅作溯源链接,不随包分发。

## 2. 硬件与软件

| 项 | 取值 |
|---|---|
| NPU | Ascend 910B3 |
| CANN | 9.2.0 |
| 框架 | torch 2.7.1 + torch_npu 2.7.1 |
| 编程模式 | Expert(显式 C/V 所有权、L1/L0/UB 与核间同步) |

## 3. 架构(三次 kernel 发射)

与 kpool 一类的单 kernel 算子不同,本算子的三个阶段形状与计算特性差异很大,
不强行融成一次发射。

| 组件 | 选择 | 理由 |
|---|---|---|
| Q 吸收 | Vector kernel,FP32 累加后显式舍入回输入 dtype | `[128] @ [128,512]` 逐头独立,Cube 分形收益不抵搬运 |
| 注意力核心 | 持久 Cube,20 个物理块;online softmax 状态常驻 UB | 块数固定为物理核数,摊薄每块序幕 |
| 值恢复 + 门 | Vector kernel,`(Z @ W_Vᵀ) * sigmoid(G)` 融合 | 门是逐通道逐元素,与值恢复同一次遍历 |
| `block_n` 选择 | host 侧 `public_prefill_block_n`,**不暴露为调参旋钮** | 只有实测过的那一个 profile 用 128,其余 64 |
| 掩码 | head 对齐时整块 packed mask,否则逐行 | 整块路径省去逐行比较 |
| 双入口 | 满足 `iterations == 1 且 heads % block_m == 0 且 batch == 1` 时编译 `main_pipelined`,否则 `main` | 见下节 |

### 跨 wave Cube/Vector 软件流水

基线是单缓冲四 flag 握手,Cube 与 Vector 严格交替。同形 PipeUtilization(各核中位)显示:

| | 基线 | 流水后 |
|---|---:|---:|
| Cube 阻塞在 flag | 57.63 µs / 129.66 µs(44%) | **1.19 µs(1%)** |
| Vector 阻塞在 flag | 66.95 µs / 130.55 µs(51%) | **23.28 µs(25%)** |
| `cube_time` | 14.66 µs | **14.66 µs** |
| `mte1_time` | 29.07 µs | **29.07 µs** |

每个核的等待量接近另一个核的忙碌量,即两者从不重叠。同一采样里 GM→L1 带宽占用 2.5%、
vector bank 冲突 1%,**既不是带宽也不是冲突受限**。

流水把 Cube 循环错开一个 wave:**wave w 的 QK 在 wave w−1 的 PV 之前发射**,
于是 Cube 做 GEMM 时 Vector 正在处理夹在中间的 softmax。三个 workspace 增加 2 槽维度,
四 flag 改为按 workspace 配对的六个 ready/consumed 信号量,消费者在循环前预置一个槽位;
两个 scope 在循环后各自排空自己预置的信号量,不让任何 cross flag 跨越 launch 存活。

`cube_time` 与 `mte1_time` 一字未变,**只有调度变了,计算量没变**。

**适用域是正确性边界,不是调优选择。** C 与 E 按 batch tag 常驻 L1,
错开的循环会让 wave w 的 QK 把新 batch 的 C/E 载入待执行 PV 仍在读的缓冲。
放开 `batch == 1` 需要第二个 C/E 槽位,约多占 144 KiB L1,必须在硬件上确认预算。
测试文件里有一条负判据钉住这个边界。

## 4. 性能

口径:`msprof op` 单 kernel Task Duration,每格 3 次独立采集 × 8 次发射、丢弃首次,
报中位数与半极差。FP16 / PTO。`waves = ceil(Tq × H / block_m / blocks)`,`blocks = min(20, 任务数)`。

| shape | waves | 优化前 | 本 PR | 加速 |
|---|---:|---:|---:|---:|
| H32 / Tq128 / Tk128 | 7 | 56.04 ±0.43 | 41.28 ±0.53 | 1.358× |
| H96 / Tq64 / Tk128 | 10 | 76.12 ±0.52 | 55.29 ±0.25 | 1.377× |
| H64 / Tq128 / Tk128 | 13 | 96.73 ±0.20 | 69.70 ±0.19 | 1.388× |
| H96 / Tq128 / Tk64 | 20 | 73.11 ±0.65 | 51.87 ±0.22 | 1.409× |
| H96 / Tq128 / Tk128 | 20 | 140.87 ±0.86 | 99.35 ±0.99 | 1.418× |
| H96 / Tq256 / Tk128 | 39 | 263.36 ±0.48 | 182.68 ±0.83 | 1.442× |
| B2 / H96 / Tq128 / Tk128 | — | 267.22 ±0.61 | 265.16 ±1.74 | 1.008× |
| H96 / Tq128 / **Tk256** | — | 180.85 ±1.32 | 180.44 ±0.32 | 1.002× |
| **H48** / Tq128 / Tk128 | — | 91.45 ±0.62 | 93.09 ±0.39 | 0.982× |

**趋势比单点有用。** 加速随 wave 数单调上升(7→1.358、10→1.377、13→1.388、20→1.418、39→1.442);
而在 wave 数相同的两格(Tk128 与 Tk64,均 20 wave)上加速基本持平(1.418 与 1.409),
尽管绝对耗时差了近一倍。**决定收益的是并行度,不是工作量** —— 这正是错开一拍的流水应有的形状:
填充与排空各占一个 wave,开销为 `1/waves`。

后三格是守卫之外的形状,全部落在 1.00× 附近(0.982–1.008×,与各自半极差同量级)。
**域外零收益是设计结果,不是遗漏。**

### 逐步实测(FP16 / PTO,B1/H96/Tq=Tk128)

| 步 | 优化 | Task |
|---|---|---:|
| 0 | 优化前 | 208.16 µs |
| 1 | 统一 query tile 整块因果 mask | 193.96 µs |
| 2 | 单 KV 块在同 batch 跨 wave 驻留 | 176.98 µs |
| 3 | 精确单块状态恒等式消除 | 165.86 µs |
| 4 | QK L0 K 粒度 128 | 159.58 µs |
| 5 | 已结束生命周期的 query L1 区预取 | 151.66 µs |
| 6 | PTO PV64 / AscendC PV32 | 143.50 µs |
| 7 | **跨 wave Cube/Vector 软件流水** | **98.55 µs** |
| 8 | **Cube 阶段判定单步进位** | **97.80 µs** |

第 7 步之前各步为单进程 pilot 快照,第 7/8 步为 3 次采集的中位数,两者不可直接相减作为严格增量。

### 对官方参考的比例

共同低精度核心范围,对 absorbed `npu_fused_infer_attention_score_v2`,Graph 摊销计时,
B1/H96/Tq=Tk128、FP16,24 次独立 trial:

| 后端 | 本 PR | 对官方 FIA v2 |
|---|---:|---:|
| PTO | 97.23 µs | **75.07%** |
| AscendC | 88.67 µs | **82.86%** |

**这些比例取自同一份 kernel 在一套包内对比装置下的测量**,本 example 不随包携带那套装置;
本节给出口径、形状与环境以便复现,但不宣称本 example 自带该对比。

**不能把 Graph 比例表述为所有调用域达标。** 同口径的普通(非 Graph)调用由主机侧开销主导,
对 FIA 约 66%,本优化按设计不改善它。

### 剩余方向

Cube 侧忙碌 89.28 µs 中,`cube_time` 仅 14.66 µs、`mte1_time` 29.07 µs,
**约 45 µs 既不是算也不是搬** —— 指令发射与核间同步下发是其候选来源。
仓库内 `examples/flash_attention/fa_opt` 记录的阶梯里,"优化核间同步下发次数"与
"用 axpy 减少指令数"分别带来 60%→72% 与 72%→80%,这两步本 PR 尚未实施;
按本算子 A9 之后 Cube 为关键路径(Cube busy 89.28 µs 对 Vector 68.53 µs)判断,
前者优先级更高。

## 5. 精度

`test_example_gated_mla_prefill.py`,22 项。三类判据不同标准:

**预算判据** —— 整链对模块内 FP32 参考,位置敏感:先要求超出有效 query 长度的行**精确为零**,
再对其余用 `rtol=2^-9`、`atol=2^-10`。覆盖 FP16/BF16 × PTO/AscendC。

**解析判据** —— 令 `QL = QA = 0` 则可见 score 全为 0、可见 key 概率为 `exp(0)`;
再令 `latent_kv[k,:] = k`,则 PV 累加为 `Σ(0..n-1)`、除以归一化因子后为 `(n-1)/2`。
**所有期望值都是 0.5 的整数倍,而 mask 多收或少收一个 key 恰好移动 0.5**,
因此主判据是「四舍五入到最近的 0.5 后精确相等」,没有容差可争论;
另有一条 8 ULP of FP32 的上界防止舍入掩盖真实漂移。
覆盖整块 mask 与逐行 mask 两条路径、`query_start` 平移、`kv_len` 钳位。

**逐位相同判据** —— 同批内各 batch 与单独计算必须逐位相同;空 batch 不得扰动相邻活跃 batch。
这两条用容差会比不测更糟:它们要抓的是 tile 越界写入相邻 batch,
产生的是有界且看起来合理的小差值。

**负判据** —— 非法形状、非连续(strided)操作数、错误 dtype 必须抛出具体异常并匹配消息片段,
不接受笼统的异常;以及流水入口在 `batch ≥ 2` 与多 KV 块时**必须不被编译**,
如果哪天它开始被编译,说明文档里写的域边界已经变了。

## 6. 已知限制

- **流水入口仅在 `iterations == 1`、`heads % block_m == 0` 且 `batch == 1` 时编译。**
  前两条是守卫性质,第三条是正确性边界(见第 3 节)。域外形状走原实现,实测零收益。
- **BF16 无取得同等精度资格的外部性能参考**,因此不提供 BF16 对标比例。
- **普通(非 Graph)调用无显著提速**,该路径由主机侧开销主导。
- 对官方 FIA 的比例只在一个工作点取得(B1/H96/Tq=Tk128、FP16、核心范围、Graph 计时);
  第 4 节的多 shape 表报的是**优化前后加速比**,不是对合格参考的比例。
- 不包含 decode 路径、cache 追加、packed 布局适配与完整模型集成。
