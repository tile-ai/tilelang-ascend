# Gated MLA prefill 跨 wave Cube/Vector 流水（A9+A10）

**Historical exploration report.** Current production a9a96f9e versus exact a172 parent is in [the source-pinned current report](PREFILL_A9_CURRENT_20261113.md). This file preserves prototype173/old-control7c data under the identities stated below.

本轮仍只修改一个生产文件 `gmla/kernels/prefill.py`，另新增一个测试文件与其 CI 入口，并增补 `gmla/kernels/design.md`。未优化 decode，未修改框架模板。保留原精度门槛与全部通用回退路径。

父提交 `a1723d4b96f0547ca2b0e25d8b6bc6fc1ec4baae`（即 A2+A3v2+A4+A5+A6+A8 选中版）。

## 环境与计算边界

device=Ascend910B3；torch=2.7.1+cpu，torch_npu=2.7.1.post4，CANN 9.2.0，Python 3.12.9。Torch 包名中的 `+cpu` 不表示在 CPU 计时，实际执行设备为 Ascend910B3。

- 核心范围：B1/H96/Tq=Tk=128，QL/QA/C/E→latent512，双方使用同一独立 CPU64 投影后 round 到 LP 的 QL。
- 本报告的 Graph/eager 数来自 `prefill_experiment.py --scope core --mode benchmark`，与既有 `PREFILL_OPTIMIZATION_20261112.md` 同口径；单 kernel Task 数来自 `msprof op`，**两者不可混比**。

## 2026-11-13 接手核验：历史数据身份

当前生产prefill SHA为a9a96f9e6fd5e4598be4b028aebfea00f5d0f20b832c465a2a46d4177c1ec6db。
以下核心77.58%/83.11%来自prototype v3（SHA17387529316b6037...），不能直接归给最终生产源码。
核心base组的被计时candidate确为父版a0e1c997，但每个trial内部名为previous的安装control仍为更早7c13f473。
原完整链12trial的候选确为当前a9a生产源；FP16外部参考比例85.54%/98.20%可按其口径引用。
完整链previous同样是7c13旧control，因此1.339/1.368倍不能称为相对a172父版的增量收益。
全部24个旧核心/完整链trial实际记录Torch-NPU2.7.1.post4，已更正原post11描述。
不得跨节点直接比较旧绝对值，也不能凭base/previous文件名推断版本。

最终生产a9a与正常安装a172父版的24trial（两范围、两精度、两后端、3trial）正在补证；
这里不提前写入未完成的新结果。BF16共同LP核心不是生产HiLo完整链。

## 诊断依据

对父提交做同形 `msprof op` + PipeUtilization 采样（所有 Cube 核与 Vector 核的各字段中位数，单位 µs）：

|字段|父提交|A9+A10|
|---|---:|---:|
|aic_time|129.66|90.47|
|aic_cube_time|14.66|**14.66**|
|aic_cube_ratio|0.10|0.15|
|aic_mte1_time|29.07|**29.07**|
|aic_mte2_time|17.40|16.63|
|aic_scalar_wait_id1|38.28|0.28|
|aic_scalar_wait_id3|19.35|0.91|
|aiv_time|130.55|91.81|
|aiv_vec_time|20.54|16.02|
|aiv_scalar_wait_id0|22.86|16.77|
|aiv_scalar_wait_id2|44.09|6.51|

字段可能重叠，不是可相加的端到端分解。

**Cube 计算单元只活跃 10.3%、Vector 14.4%**；Cube 阻塞 57.63/129.66（44%），Vector 阻塞 66.95/130.55（51%）。每个核的等待量接近另一个核的忙碌量——这正是单缓冲握手的特征：两者从不重叠。同一采样的 GM→L1 带宽占用率为 2.5%、vector bank 冲突 1%，因此瓶颈既不是带宽也不是冲突。

优化后 **Cube 等待降至 1.19 µs（1%）、Vector 降至 23.28 µs（25%）**，而 `aic_cube_time` 与 `aic_mte1_time` **一字未变**——这是"只改调度、未改计算量"的直接证据。

## 实现

Cube 循环错开一个 wave：wave w 的 QK 在 wave w−1 的 PV 之前发射，于是 Cube 做 GEMM 时 Vector 正在处理夹在中间的 softmax。score/probability/partial 三个 workspace 增加 2 槽维度（GM workspace 1.80MB→3.60MB），四 flag 改为按 workspace 配对的六个 ready/consumed 信号量，消费者在循环前预置一个槽位；协议形式取自仓库内 `examples/mha_sink_fwd_bhsd/mha_sink_fwd_bhsd.py`。两个 scope 在循环后排空各自预置的信号量，不让 cross flag 跨越 launch 存活。

A10 把 PV 阶段的判定改为从 QK 阶段单步进位的一个标志（该阶段寻址只用 slot），省掉每 wave 三次 Metadata 读、一次整除与一次取模。

UB 高水位 96032→96160 字节（上限 196608），L1 与 L0C 不变；所有非流水形状的 buffer plan 与上一版完全相同。

### 适用域

`pipeline_domain = iterations == 1 and heads % block_m == 0 and batch == 1`：

- `iterations == 1` → **Tk ≤ block_n（最大 128）**
- `heads % block_m == 0` → 头数须为行 tile 的整数倍
- `batch == 1` → **正确性边界**，非容量选择：C/E 按 batch tag 常驻 L1，而错开的循环在 wave w−1 的 PV 之前发射 wave w 的 QK；两者属于不同 batch 时 QK 会覆写待用的 C/E。放开需要第二个 C/E 槽位（约多 144 KiB L1），须在硬件上确认预算。

域内 `rows = Tq × heads` 必为 block_m 的整数倍，故 head-tail 路径在该域内静态不可达。**域外形状编译原 `main`，其生成 CPP 与上一版逐字节相同**（两后端 × 两 dtype × block_n=64 回退 × batch=2 均已核对）。

## 性能

正式核心基准，每格 3 次独立 trial（独立进程，先精度资格后计时），报告中位数与半极差：

|后端|版本|候选 Graph|官方 FIA|FIA/候选|
|---|---|---:|---:|---:|
|PTO|父提交|145.87 ±1.44|71.34 ±2.82|48.91%|
|PTO|**探索prototype v3**|**98.49 ±1.91**|76.41 ±3.77|**77.58%**|
|AscendC|父提交|135.17 ±2.22|74.63 ±2.39|55.21%|
|AscendC|**探索prototype v3**|**89.25 ±1.83**|74.17 ±2.52|**83.11%**|

父提交两行复现既有报告的 49.64% / 54.87%（差异落在 FIA 参考自身的采集波动内），可据此认定本表与原报告同口径。该探索prototype在已测核心Graph工作点超过70%；最终生产字节的门槛由当前a9a的新对照确认，不据prototype表直接勾选。

**普通（非 Graph）调用未改善**：eager_wall 对 FIA 为 66.5%（PTO）/ 66.2%（AscendC），对照父提交的 66.8% / 66.4%。该路径由主机侧开销主导（候选 eager_wall 约 578 µs 对 Graph 约 98 µs），A9 按设计不改善它。**不能把 Graph 比例表述为所有调用域达标。**

`msprof op` 单 kernel Task，3 次采集 × 8 发射、丢弃首次：

|后端|父提交|A9+A10|旧/新|
|---|---:|---:|---:|
|PTO|141.00 ±0.59|**97.80 ±0.45**|1.442×|
|AscendC|131.21 ±1.08|**84.13 ±0.90**|1.560×|

### 多 shape 矩阵（FP16/PTO，`msprof op` 单 kernel Task）

每格 3 次独立采集 × 8 发射、丢弃首次，报告中位数与半极差。`waves = ceil(Tq × H / block_m / blocks)`，`blocks = min(20, 任务数)`。

|shape|waves|基线|A9+A10|旧/新|域|
|---|---:|---:|---:|---:|---|
|H32 / Tq128 / Tk128|7|56.04 ±0.43|41.28 ±0.53|1.358×|流水|
|H96 / Tq64 / Tk128|10|76.12 ±0.52|55.29 ±0.25|1.377×|流水|
|H64 / Tq128 / Tk128|13|96.73 ±0.20|69.70 ±0.19|1.388×|流水|
|H96 / Tq128 / Tk64|20|73.11 ±0.65|51.87 ±0.22|1.409×|流水|
|H96 / Tq128 / Tk128|20|140.87 ±0.86|99.35 ±0.99|1.418×|流水|
|H96 / Tq256 / Tk128|39|263.36 ±0.48|182.68 ±0.83|1.442×|流水|
|B2 / H96 / Tq128 / Tk128|—|267.22 ±0.61|265.16 ±1.74|1.008×|回退|
|H96 / Tq128 / **Tk256**（iterations=2）|—|180.85 ±1.32|180.44 ±0.32|1.002×|回退|
|**H48** / Tq128 / Tk128（头数未对齐）|—|91.45 ±0.62|93.09 ±0.39|0.982×|回退|

**趋势比单点更有用。** 加速随 wave 数单调上升：7→1.358、10→1.377、13→1.388、20→1.418、39→1.442；而在 wave 数相同的两格（H96/Tq128 的 Tk128 与 Tk64，均为 20 个 wave）上，加速基本持平（1.418 与 1.409），尽管绝对耗时差了近一倍。**决定收益的是 wave 数，不是工作量**——这正是错开一拍的流水应有的形状：填充与排空各占一个 wave，开销为 1/waves，wave 越多摊得越薄。

三个域外形状全部落在 1.00× 附近（0.982–1.008×，与各自半极差同量级），确认守卫确实让它们走原实现、既不受益也不受损。**域外收益为零是设计结果，不是遗漏。**

逐步实测（FP16/PTO

|步|优化|Task|
|---|---|---:|
|0|优化前|208.16|
|A2|整块因果 mask|193.96|
|A3v2|单 KV 块跨 wave 驻留|176.98|
|A4|单块状态恒等式消除|165.86|
|A5|QK L0 K 粒度 128|159.58|
|A6|dead query L1 区预取|151.66|
|A8|PTO PV64 / AscendC PV32|143.50|
|**A9**|**跨 wave CV 软件流水**|**98.55**|
|**A10**|**Cube 阶段判定单步进位**|**97.80**|

A1（214.80）与 A7（149.96）未选。A9 之前各步为单进程 pilot 快照，A9/A10 为 3 次采集的中位数，两者不可直接相减作为严格增量。A10 相对 A9 的增量约 1%，**低于其 scalar 占比所提示的 5–7%**：该 setup 开销与其它流水重叠，在边际上并不制约。

## 正确性

与父提交成对对拍，**32 组全部逐位相同**（`norm_max = 0.000e+00`）：2 后端 × 2 dtype × 流水域与回退域 × Tq 4/128 × seed 0/17。

新增 `tests/test_prefill_mask_exact_npu.py`，不使用算子的 rtol/atol 预算：

- **解析式因果 mask 判据**。令 `QL = QA = 0` 则可见 score 全为 0，可见 key 概率为 `exp(0)`；令 `C[k,:] = k` 则 PV 累加为 `sum(0..n-1)`，除以归一化因子后为 `(n-1)/2`。所有期望值均为 0.5 的整数倍，而 mask 多收或少收一个 key 恰好移动 0.5，故主判据为「四舍五入到最近 0.5 后精确相等」。实测最坏偏差 **1.02 ULP of FP32（3.814697e-06）**，另有一条 8 ULP 上界防止舍入掩盖漂移。覆盖 packed 整块 mask 与原逐行 mask、query_start 平移、kv_len 钳位、inactive batch 归零。
- **逐位相同的自洽判据**：同批内各 batch 与单独计算必须逐位相同；空 batch 不得扰动相邻活跃 batch。

该解析判据不能声称 `0.0e+00`：它经过硬件向量指数与一次 FP32 除法。KDA PR1605 的三条不变量能报 `0.0e+00`，是因为它们比较的是**同一套算术的两条路径**，而非解析 oracle。

**batch 隔离判据在开发中抓到一个真实缺陷**，并据此把流水域收窄到 `batch == 1`，详见上文"适用域"与优化日志 `[A9-DEBUG-2]`。

早期19项失败已经定位为真实的预编译登记缺陷，不能归为“不支持的单进程配置”。
编译登记按npu marker或npu_device fixture筛选，而运行期按包路径拦截临时编译；新增mask文件缺少marker，
导致真实pytest --forked下出现GMLA_UNPLANNED_NATIVE_COMPILE。同时修正了测试强制使用的生产不可达block_n。

修复后真实串行pytest --forked：包内core213 passed/40 deselected、385.89s；full253 passed/0 failed、764.53s。
新文件29例（core7、low_priority22），包内既有224例。上述不含目录路由保留的原L0，
不能与历史含L0的233核心/255全量直接比较；逐文件或串行通过不能代替并行CI与加载/编译审计。

run_core_ci.py实际使用--forked -n8并强制保留L0，--workers只控制CPU预编译worker。
它审计pool、loaded factory数、published cache和native_compile_violations，新节点完整执行待补。
120秒是该包装器额外本地目标。托管workflow按路径有不同并行配置，例如目标pytest车道--forked -n4，
未调用这个120秒包装器；本地、并行完整目录与托管CI应独立记录。

## 限制

1. 探索prototype核心FIA比例只覆盖B1/H96/Tq=Tk128、FP16、Graph；多shape表是旧/新单kernel Task，不是全shape FIA比例。当前生产正式新对照待完成。
2. 原完整链12trial普通Event基本持平，不把Graph收益外推到所有调用或模型吞吐；其previous为7c13旧control。
3. BF16外部参考在原完整链两后端均未通过同等精度资格，比例无效。不得仅因调用FIA-v2就推断一定是最快参考档。
4. packed/model保留N64且不在流水域，不把七输入N128 profile收益外推到模型路径。
5. 历史217.205s/120s属于旧节点全目录包装器；新节点包内串行385.89s/764.53s是另一边界。120秒不等于托管CI判据，也不自动豁免PR模板测试成本要求。
6. 托管NPU CI需新提交后核对，不能以本地结果替代。
7. 首次真实H96/hidden7168/qrank1536随机权重单层smoke，FP16两后端通过，BF16两后端失败且归属待对照；无checkpoint、完整模型服务、多卡/异步调度或通用cache集成证据。
