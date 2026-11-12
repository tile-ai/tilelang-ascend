# Gated MLA 对齐性能报告（2026-11-12）

本轮新增预校验完整链和共同低精度输入注意力核心两组独立实验，每组36次新进程试验，共72次。生产算子没有修改；25个Python生产文件及完整链生成kernel均与上一轮逐项核对一致。计时边界变化不是算子提速，原公开接口报告继续保留。

18个FP16核心trial还逐项核对了attention/merge生成源码，与对应完整链使用的kernel完全一致。BF16共同LP核心的补偿策略边界另行说明，不作这个等同声明。

正式结果仅使用tools_v5和两个matrix_v5批次；pilot、失败或排除的先前批次不参与统计。实际计时模块通过精确路径加载，并在采样前核对实际文件哈希与报告声明一致。

环境：device=Ascend910B3; torch=2.7.1+cpu; torch_npu=2.7.1.post11; tilelang=0.1.4; matmul_precision=highest; matmul_allow_hf32=False。Torch包名中的+cpu不表示本次在CPU运行，实际执行设备为torch_npu注册的NPU。

Python 3.12.9；硬件 profiler 来自 `/home/developer/Ascend/cann-9.2.0/bin/msprof`（CANN9.2.0）。

## 范围与实验方法

- B1/H96；decode4096、同物理容量的prefix17，以及因果prefill128；FP16/BF16、PTO/AscendC。
- 每个组合3个独立进程，每进程以seed0/1/17取得资格，只有seed0计时；预热10轮、6个平衡顺序轮次。每方法60个Event及60个同步wall样本，30个Graph摊销样本。decode每图50次，prefill每图5次。
- 完整链：candidate表示预校验后执行原生产路径，public_forward是同一候选的公开接口控制，source是固定vLLM-Ascend源路径组合。动态Q投影、attention、值恢复、gate、内部guards和输出/workspace分配仍在每次调用内。
- 核心：双方使用独立CPU64投影后round到LP的同一QL和QA/C/E，输出latent512；投影、值恢复、gate均不计入。decode FIA按源布局预补H96至128，候选为H96。prefill参考是独立absorbed FIA-v2组件，与输出V128的源prefill全链内部attention不同。
- **BF16共同LP核心不是生产HiLo路径。**生产完整链保留补偿；共同LP只测已有无query residual specialization。候选核心输出FP32、FIA保留LP输出，双方均对实际共同输入的同一CPU64 golden检查。
- 原预算rtol0.01、atol0.002、最大幅度归一化误差0.005不变。参考需在所有seed及每个计时边界复核通过才给比例。表中不合格参考仍显示诊断耗时，比例留空。
- 单位µs，耗时为三个trial中位数的中位数 / 三个trial P95的中位数。比例为median_trial(参考中位数/候选中位数)，不是汇总耗时直接相除，也不是模型吞吐。Graph P95是批量摊销样本尾部，不是单请求P95。

## 预校验完整链

public是接口开销控制，source才是外部参考。每个seed还检查public/prepared字节一致、输入Q/cache/gate原位变化后重新计算一致，并恢复及检查输入只读。

|场景 / dtype / 后端|公开wall median/P95|预校验wall median/P95|源wall median/P95|源/预校验 wall|预校验Graph median/P95|源Graph median/P95|源/预校验 Graph|
|---|---:|---:|---:|---:|---:|---:|---:|
|decode4096 / float16 / pto|2149.93 / 2303.81|1782.90 / 1910.36|1079.20 / 1228.20|60.53%|159.09 / 159.57|113.70 / 113.89|71.24%|
|decode4096 / float16 / ascendc|2147.97 / 2293.43|1776.44 / 1871.73|1059.34 / 1202.89|59.63%|150.23 / 151.02|113.66 / 114.06|75.44%|
|decode4096 / bfloat16 / pto|2248.50 / 2318.60|1874.63 / 1923.10|1053.66 / 1194.05|—|187.69 / 188.39|109.14 / 109.80|—|
|decode4096 / bfloat16 / ascendc|2265.85 / 2371.90|1895.25 / 1962.00|1077.18 / 1230.57|—|170.84 / 171.40|112.67 / 113.28|—|
|decode17 / float16 / pto|2187.87 / 2384.45|1819.47 / 1917.61|1080.86 / 1221.59|58.98%|92.19 / 92.95|75.13 / 75.78|82.42%|
|decode17 / float16 / ascendc|2178.79 / 2302.31|1815.47 / 1960.06|1082.92 / 1226.65|59.75%|84.59 / 85.51|78.11 / 78.73|92.34%|
|decode17 / bfloat16 / pto|2249.45 / 2426.71|1890.24 / 1999.23|1068.23 / 1224.15|—|96.91 / 97.39|75.88 / 76.58|—|
|decode17 / bfloat16 / ascendc|2249.13 / 2329.88|1882.82 / 1955.12|1054.75 / 1189.31|—|87.50 / 88.03|77.59 / 78.32|—|
|prefill128 / float16 / pto|2051.98 / 2189.82|1687.91 / 1757.42|1300.00 / 1464.48|—|457.96 / 462.60|158.61 / 161.40|—|
|prefill128 / float16 / ascendc|2075.91 / 2235.56|1699.86 / 1821.64|1305.68 / 1487.28|—|409.51 / 415.10|164.17 / 169.14|—|
|prefill128 / bfloat16 / pto|2169.02 / 2282.43|1803.09 / 1873.73|1071.61 / 1227.20|—|548.99 / 553.96|135.84 / 141.56|—|
|prefill128 / bfloat16 / ascendc|2177.48 / 2318.90|1800.66 / 1963.31|1048.66 / 1202.99|—|471.05 / 474.04|138.27 / 143.07|—|

## 共同LP注意力核心

以下全部是组件结果；尤其BF16不得替代生产完整链性能。核心计时包括候选decode的attention与merge全部launch。

|场景 / dtype / 后端|候选wall median/P95|FIA wall median/P95|FIA/候选 wall|候选Graph median/P95|FIA Graph median/P95|FIA/候选 Graph|
|---|---:|---:|---:|---:|---:|---:|
|decode4096 / float16 / pto|827.72 / 935.69|442.72 / 550.63|52.94%|100.10 / 101.11|44.70 / 45.04|44.46%|
|decode4096 / float16 / ascendc|834.71 / 934.52|436.20 / 530.45|52.66%|97.99 / 98.58|43.80 / 44.39|44.80%|
|decode4096 / bfloat16 / pto|820.80 / 916.66|437.31 / 518.83|52.89%|115.40 / 115.79|45.28 / 45.87|39.08%|
|decode4096 / bfloat16 / ascendc|862.49 / 994.17|449.31 / 590.22|52.56%|110.92 / 111.43|44.28 / 44.74|39.92%|
|decode17 / float16 / pto|825.52 / 947.55|441.41 / 544.07|52.67%|33.12 / 33.85|18.87 / 19.36|57.08%|
|decode17 / float16 / ascendc|836.13 / 951.38|433.92 / 610.35|52.04%|32.22 / 33.24|18.71 / 19.48|58.04%|
|decode17 / bfloat16 / pto|823.88 / 954.44|435.21 / 529.15|53.16%|34.68 / 35.43|18.84 / 19.49|54.34%|
|decode17 / bfloat16 / ascendc|840.73 / 930.82|438.85 / 523.24|52.20%|34.16 / 34.45|18.95 / 19.45|55.45%|
|prefill128 / float16 / pto|644.05 / 767.70|409.80 / 514.93|63.88%|214.21 / 218.14|71.98 / 79.19|33.49%|
|prefill128 / float16 / ascendc|659.56 / 740.32|420.89 / 516.54|64.12%|197.54 / 202.98|70.13 / 77.87|35.50%|
|prefill128 / bfloat16 / pto|681.56 / 797.56|414.26 / 491.14|—|252.99 / 258.37|70.98 / 78.70|—|
|prefill128 / bfloat16 / ascendc|667.90 / 844.59|431.32 / 575.45|—|227.38 / 230.78|72.53 / 77.82|—|

## Event补充

Event区间含发射与排队间隙，不是硬件kernel Task。完整原始样本、资格和每个进程的P95见JSON。

|范围 / 场景 / dtype / 后端|候选Event median/P95|参考Event median/P95|参考/候选 Event|
|---|---:|---:|---:|
|prepared-full-chain / decode4096 / float16 / pto|1716.36 / 1799.42|999.49 / 1135.02|58.23%|
|prepared-full-chain / decode4096 / float16 / ascendc|1708.85 / 1789.26|978.15 / 1116.70|57.24%|
|prepared-full-chain / decode4096 / bfloat16 / pto|1807.93 / 1861.16|971.40 / 1096.12|—|
|prepared-full-chain / decode4096 / bfloat16 / ascendc|1825.34 / 1895.78|995.84 / 1112.14|—|
|prepared-full-chain / decode17 / float16 / pto|1756.32 / 1839.72|994.31 / 1129.66|56.51%|
|prepared-full-chain / decode17 / float16 / ascendc|1739.43 / 1827.04|996.40 / 1131.84|57.28%|
|prepared-full-chain / decode17 / bfloat16 / pto|1822.11 / 1904.16|987.08 / 1123.78|—|
|prepared-full-chain / decode17 / bfloat16 / ascendc|1808.11 / 1879.32|972.80 / 1092.94|—|
|prepared-full-chain / prefill128 / float16 / pto|1621.98 / 1668.78|1221.17 / 1355.50|—|
|prepared-full-chain / prefill128 / float16 / ascendc|1627.25 / 1738.50|1215.87 / 1385.26|—|
|prepared-full-chain / prefill128 / bfloat16 / pto|1729.27 / 1812.12|985.50 / 1128.60|—|
|prepared-full-chain / prefill128 / bfloat16 / ascendc|1729.88 / 1796.26|967.96 / 1110.12|—|
|shared-lp-core / decode4096 / float16 / pto|747.24 / 809.86|363.58 / 416.80|48.66%|
|shared-lp-core / decode4096 / float16 / ascendc|758.66 / 817.74|367.74 / 431.60|48.59%|
|shared-lp-core / decode4096 / bfloat16 / pto|744.42 / 808.74|368.10 / 418.88|48.52%|
|shared-lp-core / decode4096 / bfloat16 / ascendc|787.85 / 849.30|378.02 / 453.38|48.17%|
|shared-lp-core / decode17 / float16 / pto|755.54 / 801.48|371.21 / 426.40|49.13%|
|shared-lp-core / decode17 / float16 / ascendc|766.14 / 825.84|362.98 / 439.68|48.16%|
|shared-lp-core / decode17 / bfloat16 / pto|751.63 / 827.48|364.80 / 418.76|48.38%|
|shared-lp-core / decode17 / bfloat16 / ascendc|771.55 / 821.60|368.47 / 429.00|47.97%|
|shared-lp-core / prefill128 / float16 / pto|591.44 / 626.28|345.17 / 393.62|58.36%|
|shared-lp-core / prefill128 / float16 / ascendc|593.27 / 629.24|353.07 / 380.96|59.42%|
|shared-lp-core / prefill128 / bfloat16 / pto|625.45 / 698.16|340.70 / 406.22|—|
|shared-lp-core / prefill128 / bfloat16 / ascendc|609.04 / 669.82|357.62 / 415.60|—|

## 结果解读

FP16/PTO下，预校验完整链对源decode组合的Graph相对速率为：长decode 71.24%、prefix17 82.42%；同两点普通同步wall为60.53%/58.98%。源prefill未取得精度资格，没有完整链源比例。

共同LP核心对FIA的Graph比例分别为：长decode 44.46%、prefix17 57.08%、prefill128 33.49%。核心仍有明显差距，后续应优先分析prefill和长decode的注意力计算/搬运/同步。预校验边界没有修改生产kernel，也不能将完整链比例用作注意力核心已达标的证明。

## 证据和复现

已从原始样本独立重算全部统计，共核对27000个正式时间样本；每条跨进程汇总也与原始summary.json逐项一致。不同生成kernel、共同输入或边界不能合并；graph中途失败的部分样本仅保留诊断，不进入正式耗时。

- [原始样本与精度资格](PERFORMANCE_ALIGNED_20261112.public.json)
- [边界、约束、社区PR依据与完整复现命令](ALIGNMENT.md)
- [原公开接口平衡性能报告](PERFORMANCE_BALANCED_20261112.md)

本报告不表示完整模型部署、所有dtype/参考均合格，或所有合入门槛已满足。

## 硬件Task快照

在正式72次试验结束后，另行对decode17/FP16/PTO执行msprof op：kernel replay、warm-up5，核心选2条launch，完整链选4条。两个应用入口的三seed精度/只读检查均通过；入口生成CPP与对应正式trial逐项相同。以下为独立硬件Task，不是完整调用时延，不相加后与Event/Graph相减，也不计算对官方的Task比例。

|范围|序号 / 阶段|Task µs|Block / Mix Block|当前 / 额定频率|原始CSV|
|---|---|---:|---:|---:|---|
|shared-lp-core|0 / attention|25.68|18 / 36|1800 / 1800|[CSV](hardware_alignment_20261112/core/OpBasicInfo_0.csv)|
|shared-lp-core|1 / merge|6.60|3 / 6|1800 / 1800|[CSV](hardware_alignment_20261112/core/OpBasicInfo_1.csv)|
|prepared-full-chain|0 / q_absorb|27.50|20 / 40|1800 / 1800|[CSV](hardware_alignment_20261112/full/OpBasicInfo_0.csv)|
|prepared-full-chain|1 / attention|26.14|18 / 36|1800 / 1800|[CSV](hardware_alignment_20261112/full/OpBasicInfo_1.csv)|
|prepared-full-chain|2 / merge|6.78|3 / 6|1800 / 1800|[CSV](hardware_alignment_20261112/full/OpBasicInfo_2.csv)|
|prepared-full-chain|3 / value_gate|36.24|20 / 40|1800 / 1800|[CSV](hardware_alignment_20261112/full/OpBasicInfo_3.csv)|

硬件记录关联到相同入口与生成源码；没有直接比对profile机器码。全部正式原始记录和profile归档为 `aligned_final_v5.tar.gz`，SHA256 `212bf3e9b2d021d3d188a8b3770277121462ae93598b92d99b8075acac684bd7`，不包含可重建kernel cache。

公开JSON以 `records` 和 `$record` 引用去重重复的环境、来源与精度记录；计时数组不变。导出后已递归还原并逐项核对，原始27000个正式时间样本未改变；计时后检查的第二份副本已与保留记录逐项比对后移除。
