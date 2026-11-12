# Gated MLA prefill 内核优化报告（2026-11-12）

本轮只修改一个生产文件 `gmla/kernels/prefill.py`，未优化 decode，也未修改框架模板。保留原精度门槛和通用回退路径。
基线提交 `de16598e67bce8168775fce23f9b207dd6259401`；prefill 源码由 `7c13f47328dfefb3bb680d643bdeb3f5da8c1a77cf33fe4bebc1b61623f4660f` 更新为 `a0e1c997ea1204ea12f508311fe4d584f2274585198503c322259c5de0abc2c0`。

## 环境与计算边界

device=Ascend910B3; torch=2.7.1+cpu; torch_npu=2.7.1.post11; tilelang=0.1.4; matmul_precision=highest; allow_hf32=False。Python 3.12.9。实际执行设备为 Ascend910B3；Torch 包名中的 +cpu 不表示本轮在 CPU 计时。

- 核心：B1/H96/Tq=Tk128，QL/QA/C/E→latent512。双方使用同一独立CPU64投影后round到LP的QL，均对同一CPU64核心参考检查。候选/旧版输出FP32、FIA保留LP输出。
- **BF16共享LP核心不等于生产HiLo路径。**完整链保留真实BF16 query和P的残差补偿；不以关闭补偿取得生产性能收益。
- 完整链：原七输入→Q吸收投影、attention、值恢复和gate，位于W_O之前。新旧两侧都预校验并绑定固定storage；动态计算/分配仍计入。外部参考为**历史FP32投影/gate＋官方FIA组合**，FIA内部仍有低精度运算，不能称为全链FP32或vLLM源prefill。
- norm、外部输入/gate投影、cache append、W_O、通信和完整模型服务不计入上述范围。

## 精度与回归

原门槛固定为 rtol0.01、atol0.002、最大幅度归一化误差≤0.005；没有因dtype或参考来源放宽。
104个本轮数值回归（13形状×seed0/17×2dtype×2后端）的新旧两方法全部通过，覆盖跨batch/inactive、通用head、物理尾块、单/多KV块、续接、M16/32与N64/128，以及BF16有/无query residual。
另有4组后端/dtype Graph绑定检查，每组均覆盖C/E联合更新和E-only更新，确认storage不变、数学参考和输出确实变化，并恢复输入字节和结果。只读检查由固定哈希driver逐次调用；原报告没有独立的每case readonly布尔字段，本报告不伪造该字段。
这些是已列出的回归域与代表性性能实验，不代表所有可声明shape、完整模型部署或托管CI均已通过。正常安装与核心/扩展测试证据见文末单独补充。

## 计时与聚合

24次独立进程＝2范围×2dtype×2后端×3trial。每trial先以seed0/1/17验证资格，只有seed0计时。预热10轮，6种方法顺序各一次；每方法60个Event、60个同步wall、30个Graph摊销样本（每图5次、5轮回放预热）。输入/权重复用，不刷新L2。
Event包含发射/排队间隙，不是纯硬件Task；wall包含结束同步；Graph P95是摊销样本尾部，**不是单请求P95**。三种指标分别报告。
耗时格为三个trial中位数的中位数 / 三个trial P95的中位数。旧/新和外部/新均为median_trial(参考中位数/新候选中位数)，不是汇总耗时直接相除，也不是模型吞吐。第三方参考须全部初始seed与该边界复核合格才给比例；失格耗时仅作诊断。

## 共同LP注意力核心

|dtype / 后端|旧Graph median/P95 µs|新Graph median/P95 µs|旧/新|外部Graph median/P95 µs|合格外部/新|
|---|---:|---:|---:|---:|---:|
|float16 / pto|212.24 / 216.64|144.18 / 150.16|147.33%|71.33 / 79.17|49.64%|
|float16 / ascendc|196.60 / 199.92|134.21 / 135.25|147.93%|73.64 / 76.84|54.87%|
|bfloat16 / pto|252.23 / 255.82|173.08 / 177.38|145.53%|71.14 / 75.90|—|
|bfloat16 / ascendc|222.10 / 227.08|155.21 / 160.02|143.01%|71.27 / 75.21|—|

|dtype / 后端 / 边界|旧median/P95 µs|新median/P95 µs|旧/新|外部median/P95 µs|合格外部/新|
|---|---:|---:|---:|---:|---:|
|float16 / pto / eager_event_us|584.32 / 628.46|550.19 / 603.42|106.20%|351.65 / 411.28|63.93%|
|float16 / pto / eager_wall_us|638.29 / 733.09|618.78 / 686.96|103.15%|418.15 / 503.31|68.03%|
|float16 / ascendc / eager_event_us|578.81 / 624.54|564.08 / 611.34|102.61%|354.94 / 403.96|61.65%|
|float16 / ascendc / eager_wall_us|641.41 / 717.03|629.69 / 685.66|100.91%|421.91 / 499.46|65.95%|
|bfloat16 / pto / eager_event_us|627.90 / 703.46|557.44 / 629.12|112.82%|352.50 / 408.12|—|
|bfloat16 / pto / eager_wall_us|685.07 / 812.07|626.56 / 743.09|109.34%|420.43 / 532.28|—|
|bfloat16 / ascendc / eager_event_us|591.09 / 617.48|548.23 / 596.42|107.82%|338.52 / 387.76|—|
|bfloat16 / ascendc / eager_wall_us|644.26 / 726.76|615.66 / 670.17|104.71%|403.97 / 495.57|—|

## 预校验完整链

|dtype / 后端|旧Graph median/P95 µs|新Graph median/P95 µs|旧/新|外部Graph median/P95 µs|合格外部/新|
|---|---:|---:|---:|---:|---:|
|float16 / pto|463.66 / 467.27|393.87 / 398.70|117.26%|312.41 / 322.00|79.12%|
|float16 / ascendc|412.29 / 416.10|348.44 / 351.05|118.58%|322.47 / 326.81|92.44%|
|bfloat16 / pto|551.72 / 554.86|465.88 / 469.39|118.67%|325.49 / 329.22|—|
|bfloat16 / ascendc|472.30 / 478.13|402.36 / 403.87|117.61%|320.03 / 322.98|—|

|dtype / 后端 / 边界|旧median/P95 µs|新median/P95 µs|旧/新|外部median/P95 µs|合格外部/新|
|---|---:|---:|---:|---:|---:|
|float16 / pto / eager_event_us|1624.15 / 1692.92|1615.58 / 1694.90|99.60%|1208.54 / 1327.28|75.77%|
|float16 / pto / eager_wall_us|1696.52 / 1776.63|1689.61 / 1779.36|99.59%|1287.30 / 1428.79|77.30%|
|float16 / ascendc / eager_event_us|1624.50 / 1682.90|1621.35 / 1675.86|100.09%|1225.83 / 1348.64|76.24%|
|float16 / ascendc / eager_wall_us|1693.25 / 1806.90|1690.96 / 1749.12|99.67%|1302.11 / 1450.06|77.80%|
|bfloat16 / pto / eager_event_us|1717.89 / 1796.56|1726.54 / 1794.12|99.74%|1189.29 / 1310.02|—|
|bfloat16 / pto / eager_wall_us|1791.69 / 1875.61|1792.63 / 1919.13|99.80%|1268.02 / 1421.77|—|
|bfloat16 / ascendc / eager_event_us|1729.15 / 1804.58|1733.40 / 1794.66|100.05%|1200.07 / 1327.70|—|
|bfloat16 / ascendc / eager_wall_us|1799.19 / 1912.41|1804.36 / 1882.00|99.84%|1279.64 / 1429.77|—|

## 单进程探索与硬件快照

A1未选，A2+A3v2+A4+A5+A6+A8组成最终选中候选；A7 metadata缓存未选。各pilot/ablation属于独立单进程探索，**不能将相邻pilot的差值当作严格控制的增量加速**。最终收益以以上24次成对实验为准。

下面是各自应用入口精度通过、CPP哈希核对后的msprof单kernel Task快照；不与Event/wall/Graph混比，不把Task称为端到端时延。

|版本|状态/内容|Task µs|Block / Mix|频率|CSV|
|---|---|---:|---:|---:|---|
|baseline|保留基线|208.16|20 / 40|1800 / 1800|[CSV](hardware_prefill_20261112/baseline/OpBasicInfo_0.csv)|
|A1|未选：行广播收益不明确，额外UB不值得|214.80|20 / 40|1800 / 1800|[CSV](hardware_prefill_20261112/A1/OpBasicInfo_0.csv)|
|A2|选入：统一query tile的整块因果mask|193.96|20 / 40|1800 / 1800|[CSV](hardware_prefill_20261112/A2/OpBasicInfo_0.csv)|
|A3v2|选入：单KV块在同batch跨wave驻留|176.98|20 / 40|1800 / 1800|[CSV](hardware_prefill_20261112/A3v2/OpBasicInfo_0.csv)|
|A4|选入：精确单块状态恒等式消除|165.86|20 / 40|1800 / 1800|[CSV](hardware_prefill_20261112/A4/OpBasicInfo_0.csv)|
|A5|选入：QK L0 K粒度128|159.58|20 / 40|1800 / 1800|[CSV](hardware_prefill_20261112/A5/OpBasicInfo_0.csv)|
|A6|选入：已结束生命周期的query L1区预取|151.66|20 / 40|1800 / 1800|[CSV](hardware_prefill_20261112/A6/OpBasicInfo_0.csv)|
|A7|未选：metadata缓存无明确附加收益|149.96|20 / 40|1800 / 1800|[CSV](hardware_prefill_20261112/A7/OpBasicInfo_0.csv)|
|A8|选入：PTO PV64 / AscendC PV32|143.50|20 / 40|1800 / 1800|[CSV](hardware_prefill_20261112/A8/OpBasicInfo_0.csv)|

## 可核验数据与限制

已独立重算10800个正式时间样本及全部跨进程汇总；计时后每方法六轮Event/Graph复核均检查。实际加载的计时helper文件SHA为 `13f6b117b6df7d63063111b551cd0d70fc7c95211eff6212acab7c0a39ad4e1d`，与原始报告及归档文件一致。
不同driver/variant/原25源文件、协议、输入/metadata、环境或生成CPP的trial不能聚合。原始CSV与CPP逐文件检查声明哈希。采样数组只保留一份；公开JSON的records/$record去重经递归还原核对，没有改变样本和资格。
[机器可读样本与资格](PREFILL_OPTIMIZATION_20261112.public.json)。公开数据删除命令、私有安装路径和device UUID/properties，原始完整归档仍应由贡献者保留。

不以本报告替代最终提交的CI、完整核心/扩展套件、simulator或部署验证。最终收录范围和未达标项仍需按仓库要求处理。

## 定位依据与实现要点

下表是同形FP16/PTO硬件采样中所有Cube核／Vector核的各字段中位数，单位µs。字段可能重叠，不是可相加的端到端分解；等待下降只作为与代码修改一致的证据。

|字段|原版|选中版|
|---|---:|---:|
|aic_cube_time(us)|19.20|14.66|
|aic_mte1_time(us)|35.32|29.07|
|aic_mte2_time(us)|31.68|17.54|
|aic_scalar_wait_id1_time(us)|68.30|38.68|
|aic_scalar_wait_id3_time(us)|19.30|20.56|
|aiv_vec_time(us)|33.79|20.54|
|aiv_scalar_time(us)|37.05|23.87|

[基线详细计数](hardware_prefill_20261112/baseline/PipeUtilization.csv) · [选中版详细计数](hardware_prefill_20261112/selected/PipeUtilization.csv)。同目录另保留Memory/MemoryL0/ArithmeticUtilization/ResourceConflictRatio原始CSV。

改动包括：同query tile整块因果mask；单KV块按batch在L1驻留；一次KV处理时消除初始online状态的恒等操作；在Vector softmax窗口预取下一任务的query；QK/PV按真实后端L0预算分块。C/E与query标记都在每次kernel入口显式reset，跨调用不保留输入；非整齐head与多KV保留通用路径，BF16 query/P残差保持。

一次实验暴露了alloc_var非零初始化annotation未到达实际PTO Allocate的问题：生成代码默认0使batch0首载入被跳过。失败版本未计时／未选入，修订在Cube scope内显式赋-1；同指针输入、图回放和metadata更新回归用于防止此类stale状态。没有修改框架生成器。

## 正常安装与完整测试补充

优化wheel的25个生产Python文件中仅prefill.py改变，所装源码SHA与本报告选中变体一致。
8组正常安装验证（核心/完整链×两dtype×两后端）全部通过，实际生成CPP同时与独立变体及正式性能trial逐项一致。
完整目录233项核心测试通过（pytest209.88s，0failed/0skipped）；冷启动完整进程217.21s，包含CPU worker并行编译、加载、执行和审计。
原复现工具保留120秒目标，因此wrapper返回1，pytest返回0、artifact检查通过；**120秒目标未达标**，没有修改阈值或把状态写为通过。
另以全新编译/审计目录执行22项扩展，全部通过，完整进程158.52s。核心与扩展分别预编译99/73个唯一factory，无NPU参与编译；所有worker/parent ABI和加载源码一致、published cache不变、未发生意外native recompile。

新增自动回归仅3个核心NPU节点，复用既有代表点并增强输入/Graph/metadata更新；另补1个CPU缓冲容量门禁。未隐藏关键回归，22项扩展标记保持。
CPU包内选择178 passed、46项device deselected。上述完整目录包括原legacy L0。
[正常安装、代码/测试指纹与测试状态](PREFILL_VALIDATION_20261112.json)。代码及测试与已测树逐字节对应，报告/文档资产后补，不把文档更新误当数值代码再验证。

原始数值/性能/profiler归档SHA256：`60af5c41e2a9a966cd388646dfe1764bbb7a62ca30e4e6abf9590ed31541d6ed`。
正常安装/完整测试审计归档SHA256：`d3da838a940297feecef7124042e7332bda6e1a2c8d5603a75787a7ec7435371`。
二者均不包含可重建kernel cache；完整源码和原始记录在本地及远端备份。

## 更新源码的代表性仿真

本轮新源在PTO/AscendC、FP16/BF16四组B1/H96/Tq=Tk128 prefill上全部完成prepare、direct、simulator和audit。原预算精度、输入只读和每组3次真实physical launch均通过；25个生产源与已测树一致，三阶段source/native/CPP一致，CPP与同dtype/后端正式完整链trial0一致。正式性能JSON未保存逐库native哈希，因此不声称formal trial逐库native匹配。

[精简仿真报告与实际日志文字渲染图](PREFILL_SIMULATOR_20261112.md) · [身份与审计摘要](PREFILL_SIMULATOR_20261112.json)。图是实际日志及已审计字段的文字渲染，非MindStudio GUI截图。原生source-line mapping不可用，simulator时延不是硬件性能；完整trace留存在私有仿真证据中。未改工具仍输出历史r1.7标签，实际版本由包含本轮prefill a0e的源指纹确定。
