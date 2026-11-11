# Gated MLA 平衡性能测试报告（2026-11-12）

本轮完成 **3 个场景 × 2 种 dtype × 2 个后端 × 3 次独立进程 = 36 次正式试验**。候选在全部试验中通过原始 FP64 精度门槛；参考实现按场景分别判定资格。**本轮没有修改 kernel，这是一份补全测试与统计口径的报告，不是优化前后对比。**

正式结果只使用 `full_matrix_v2`，不合并 pilot、独立 host 诊断或硬件 profile。正式试验启动时间为 2026-11-11 15:25–15:46 UTC，即日本时间 2026-11-12 00:25–00:46；每个 trial 的完整时间戳保留在公开数据。

## 范围与资格

- 公共七输入 forward，输出位于 W_O 之前；不包含外部 Q/KV/gate 投影、norm、cache append、W_O 或通信。B=1、H=96，D/E/R/V=128/64/512/128。
- `decode4096`：Tq=1、物理 Tk=4096、可见长度4096；`decode17`：同一4096物理容量、可见长度17；`prefill128`：Tq=Tk=128、start=0、因果注意力。
- dtype 为 FP16/BF16，候选后端为 PTO/AscendC。两个参考使用 torch_npu API；表中 target 指候选编译后端。
- 每个 trial 先以 seed 0/1/17 验证资格，**只有 seed 0 参与计时**。输出同时比较原始七输入的 expanded 与 absorbed FP64 参考；候选还检查 query/latent 中间量。
- 门槛保持 rtol=0.01、atol=0.002、全局最大归一化误差≤0.005，不因 dtype 或参考来源放宽。候选108次 seed 资格检查、432次计时后边界检查均通过。只读检查覆盖七输入、metadata、源参考常量和历史 FIA 的独立 table/mask。
- 每个参考、每个计时边界都要求三个 trial 的初始资格与该边界计时后检查通过，才给出比例。保留不合格参考的诊断耗时，但不赋予比例；尤其不能因为计时 seed 0 通过，就忽略 seed 17 的失败。

## 环境与两个参考的区别

Ascend 910B3，20 Cube / 40 Vector 核，192 MiB L2；Python 3.12.9，torch 2.7.1+cpu 配合 torch_npu 2.7.1.post11，TileLang 0.1.4。关联 host 记录的 CANN 路径为 cann-9.2.0。`+cpu` 是 torch 包标识，实际测试通过 torch_npu 在 NPU 执行。FP32 matmul precision 为 highest，allow_hf32=false；源码清单、kernel SHA 和各 trial 环境保留在 JSON。

1. **源参考**：固定 vLLM-Ascend commit `e4d533d33211f0696bcdf65eda47245d52e428c5`，直接调用隔离的原 gate kernel。decode 为低精度 Q 投影、head padding、FIA v2、真实 transpose batchmatmul 和 gate；prefill 为低精度 KV linear、原 FIA 路径及 gate。源 prefill 在 FP16 输入下仍将 FIA 操作数转换为 BF16，这次没有通过原预算，不能标成合格 FP16 比较。静态权重/表/掩码准备排除在每次调用之外，动态投影、必要布局处理、FIA 与 gate 保留在调用内。
2. **历史 FP32 组合参考**：官方 FIA v2 加 PyTorch FP32 投影/gate 的组合。它不是源参考的同一个实现，且 FIA 内部仍含低精度计算；“FP32”不表示全链 FP32。两条参考单独列示，不能互换或挑选比例混成一个结论。

来源：[固定 MLA 源码](https://github.com/vllm-project/vllm-ascend/blob/e4d533d33211f0696bcdf65eda47245d52e428c5/vllm_ascend/attention/mla_v1.py)、[固定 gate 源码](https://github.com/vllm-project/vllm-ascend/blob/e4d533d33211f0696bcdf65eda47245d52e428c5/vllm_ascend/ops/triton/flash_attention_output.py)；本报告使用已归档的 source_adapter/源码哈希记录。

## 计时与聚合定义

- 每个 trial 预热10轮，随后6种三方法执行排列各出现一次；每轮 eager 测10次，因此每方法有60个 Event和60个 wall 样本。Graph 每轮测5次，共30个样本，正式采样前预热回放5轮。输入与权重复用，不刷新 L2。
- **E / Event**：eager 公共调用的设备事件区间；不能称为纯 kernel Task 时间。**W / wall**：主机计时，包含结束事件同步。**G / Graph**：回放 Event时间除以捕获调用数，decode每图50次、prefill每图5次。Graph回放不会重新执行每次Python校验，不能当成带完整主机调用开销的单次部署时延。
- 主表耗时为 `median(三个 trial 的 median)`；P95列为 `median(三个 trial 的 P95)`，trial内用 nearest-rank `ceil(0.95*n)`。这不是把三个进程的样本混合后重算P95；原始trial P95及中位数范围保留在JSON。**Graph P95是批量摊销样本的P95，不是单请求尾延迟。**
- 比例定义为 `median_trial(reference_trial_median / candidate_trial_median)`，不是两个跨trial汇总耗时直接相除。它是固定工作量下按耗时倒数表示的相对执行速率，**不是模型吞吐**；100%表示同速，比例不能跨 E/W/G 比较。

## 主表：12个组合

耗时单位均为 **µs**；耗时格写作 **median / P95**。比例格按 **E / W / G** 顺序。`—`表示参考未取得资格，不表示未测量或零耗时。

| 场景 | dtype | 后端 | 候选 E median/P95 | 候选 W median/P95 | 候选 G median/P95 | 源参考/候选 E/W/G | 历史FP32/候选 E/W/G |
|---|---|---|---:|---:|---:|---:|---:|
| decode4096 | FP16 | pto | 2209.51 / 2387.30 | 2282.45 / 2581.84 | 157.85 / 158.64 | 45.92% / 48.15% / 69.57% | 47.89% / 50.08% / 81.38% |
| decode4096 | FP16 | ascendc | 2214.16 / 2337.28 | 2278.36 / 2424.08 | 149.73 / 150.32 | 46.29% / 48.81% / 74.76% | 48.46% / 50.76% / 86.01% |
| decode4096 | BF16 | pto | 2306.39 / 2424.98 | 2370.20 / 2538.39 | 187.26 / 187.69 | — / — / — | 45.46% / 47.39% / 68.90% |
| decode4096 | BF16 | ascendc | 2258.36 / 2361.14 | 2322.47 / 2429.46 | 170.29 / 170.56 | — / — / — | 45.14% / 47.17% / 75.63% |
| decode17 | FP16 | pto | 2214.21 / 2320.56 | 2274.13 / 2397.83 | 93.53 / 94.04 | 45.72% / 48.10% / 81.31% | 48.42% / 50.39% / 106.85% |
| decode17 | FP16 | ascendc | 2213.79 / 2291.34 | 2279.03 / 2412.79 | 84.43 / 85.06 | 45.33% / 47.93% / 88.90% | 47.13% / 49.24% / 117.22% |
| decode17 | BF16 | pto | 2314.11 / 2395.92 | 2376.40 / 2575.04 | 96.11 / 96.53 | — / — / — | — / — / — |
| decode17 | BF16 | ascendc | 2363.17 / 2493.06 | 2427.04 / 2674.18 | 87.27 / 87.90 | — / — / — | — / — / — |
| prefill128 | FP16 | pto | 2089.46 / 2175.00 | 2152.56 / 2261.73 | 454.03 / 458.18 | — / — / — | 61.01% / 63.23% / 70.50% |
| prefill128 | FP16 | ascendc | 2134.50 / 2200.12 | 2197.58 / 2305.74 | 405.82 / 408.76 | — / — / — | 61.28% / 63.45% / 78.71% |
| prefill128 | BF16 | pto | 2228.89 / 2366.22 | 2295.89 / 2584.49 | 544.21 / 548.82 | — / — / — | — / — / — |
| prefill128 | BF16 | ascendc | 2215.00 / 2305.30 | 2282.26 / 2412.88 | 466.65 / 469.39 | — / — / — | — / — / — |

**资格边界：**源 decode 的 FP16 两个场景合格；BF16 decode4096 在 seed17 失败，decode17 在0/1/17均失败，因此BF16源比例全部留空。源 prefill 的两种dtype均未通过。历史FP32组合的 FP16全部场景及 BF16 decode4096 合格；BF16 decode17/prefill128不合格。失败指标及索引保留在公开JSON的精度记录中，未用修改阈值获得资格。

在合格的**源 decode Graph**比较中，FP16长decode为 PTO 69.57%、AscendC 74.76%；短前缀为81.31%、88.90%。在另一条**历史FP32组合 Graph**比较中，FP16 prefill为70.50%、78.71%。这些数字对应不同参考，不合成一个总分。**FP16 decode 的 eager Event/wall 源比例只有约45%–49%**，因此不能把约70%的Graph观测概括为完整公开调用也达到70%；本报告不以Graph数字代替它。

## 独立主机诊断：host_diagnostic_a2

这是 decode17/FP16/PTO 的独立诊断进程，未并入主表三个trial。**未插桩**的60次测量：Event中位数 **2100.32 µs**，含同步wall中位数 **2173.59 µs**。重新预热后单独运行30次cProfile：插桩调用中位数2956.77 µs，插桩外的调用后同步中位数106.58 µs。后二者不当作正式性能结果，也不相加为另一种端到端定义。

| 函数 | 每次forward调用次数 | self µs/forward | inclusive µs/forward |
|---|---:|---:|---:|
| `_forward_validated` | 1 | 60.05 | 2224.61 |
| `lambda_forward` | 4 | 670.66 | 1340.12 |
| `require_dense_base` | 15 | 296.38 | 766.30 |
| `validate_forward_inputs` | 1 | 87.14 | 589.62 |
| `_view` | 3 | 49.84 | 503.57 |
| `guarded_view` | 3 | 39.55 | 427.71 |

**inclusive包含子调用，行与行相互重叠，不能相加。**这是被插桩扰动的主机时间归因，包含Python、驱动/发射等待，Cython内部可能不可见；不能由 `Event−Graph` 得到validator耗时，也不能把某行inclusive解释为删除校验后必然可节省的时间。入口、无插桩计时后及profile后输出和只读检查均通过。该结果提供后续排查线索，不构成已经优化变快的证据。

## 独立硬件快照：msprof op

decode17/FP16/PTO，`--kernel-name=main_kernel --launch-count=4 --warm-up=5 --replay-mode=kernel`。四份OpBasicInfo CSV已逐份读出并与记录SHA核对；以下是4条kernel replay Task快照。

| 记录序号 | Task µs | Block Dim | Mix Block Dim | 原始CSV |
|---:|---:|---:|---:|---|
| 0 | 28.88 | 20 | 40 | [CSV 0](hardware_20261112/OpBasicInfo_0.csv) |
| 1 | 27.22 | 18 | 36 | [CSV 1](hardware_20261112/OpBasicInfo_1.csv) |
| 2 | 6.58 | 3 | 6 | [CSV 2](hardware_20261112/OpBasicInfo_2.csv) |
| 3 | 36.66 | 20 | 40 | [CSV 3](hardware_20261112/OpBasicInfo_3.csv) |

这些数值不包含完整主机调用/排队边界，不能称为端到端耗时，也没有同边界参考Task测量，**不计算性能比例**。不把四个数相加后与Event或Graph相减归因。

源码关联证据为：profile入口的identity与其validation入口一致；validation导出的4份CPP哈希又与正式benchmark一致。**profile模式本身没有单独导出CPP，因此不是“直接比较profile CPP”。**

## 复现与文件

正式suite为串行独立进程，3trial、6rounds、10eager repeats、5graph repeats、warmup10，保留未合格参考诊断耗时。命令骨架如下；应使用已固定的安装包、原始隔离gate及相同环境，禁止用新的source inventory冒充本轮来源。

```bash
python run_paired_suite.py \
  --installed-root <pinned-installed-site> \
  --inventory <pinned_inventory.json> \
  --benchmark-root <package/benchmark> \
  --output <new-output-directory> \
  --trials 3 --rounds 6 --repeats 10 --graph-repeats 5 --warmup 10 \
  --time-unqualified
```

[精简公开数据](PERFORMANCE_BALANCED_20261112.public.json)保留每个trial/method/metric的原始样本一次、执行顺序、精度资格、输入哈希和来源。原始round样本与paired样本的重复副本被去除；相同索引仍可还原配对，各类来源/精度记录通过records引用去重。私有设备UUID与绝对安装路径不用于公开展示。

完整原始依据保留在下列归档中：`full_matrix_v2/summary.json`（正式汇总）、`full_matrix_v2/suite.json`（串行执行记录）、`host_diagnostic_a2/host_decode_diagnostic.json`（主机诊断）、`VERIFIED.json`（硬件关联校验）。这些是归档内相对路径，并非公开仓库链接。原始失败的首次host环境检查也保留在归档，但不并入成功的a2结果。

- 主归档 `perf_evidence_verified.tar.gz` SHA256：`e805e8a2e00b1170157f84ffda4b7476b87e28b29fb5145f06431354d86188c6`。
- 四CSV补充包 `hardware_csvs.tar.gz` SHA256：`7961bf1a423a495de2ecf2cf04b2a5077a8d95c9e36589d45eccb11d1c0f5887`。
- 精简JSON删除重复存储，不改变任何样本数值；导出时已逐项回读核对全部16200个正式计时样本，独立重算108条method/metric汇总及合格比例。

本报告可用于说明当前已测范围的性能与限制；不代表新增优化、不表示所有参考都合格，也不扩大到未测shape或完整模型部署。
