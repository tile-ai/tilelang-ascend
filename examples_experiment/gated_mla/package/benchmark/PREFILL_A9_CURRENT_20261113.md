# Gated MLA A9/A10 当前生产版本性能报告（2026-11-13）

本报告由完整24个正式trial重新核对，不沿用旧prototype或错误control的数字。

父提交：a1723d4b96f0547ca2b0e25d8b6bc6fc1ec4baae；父prefill：a0e1c997ea1204ea12f508311fe4d584f2274585198503c322259c5de0abc2c0。
候选prefill：a9a96f9e6fd5e4598be4b028aebfea00f5d0f20b832c465a2a46d4177c1ec6db。正常安装control的25源与父整包一致，仅prefill与候选不同。

## 环境、范围与计时

实际记录：Python 3.12.9，Torch 2.7.1+cpu，Torch-NPU 2.7.1.post4，TileLang 0.1.4，设备 Ascend910B3。
代表工作点 B1/H96/Tq=Tk128，D/E/R/V=128/64/512/128；highest matmul，allow_hf32=false。

- 完整链：七输入至W_O之前，包括Query吸收投影、attention、Value恢复及gate；预校验并绑定storage，计算/分配仍计入。
- 核心：共同LP的QL/QA/C/E至latent512；BF16共享LP核心不代表生产HiLo完整链。
- 核心参考为官方FIA-v2；完整链参考为历史FP32投影/gate加FIA-v2组合，并非官方同名融合算子。实现/中间精度差异保留。FP32投影/gate组合可能慢于充分优化的低精度组合，本表不证明官方最佳完整链；其他MLA/prolog接口尚未在本七输入边界适配并取得同门资格，不据接口版本或RoPE参数将其排除。
- 两scope×两dtype×两后端×3trial，共24个独立进程。每trial先验证seed0/1/17，只计时seed0。
- 六种方法顺序平衡；warmup10，每方法60个Event、60个同步wall、30个Graph摊销样本，Graph每图5次、回放预热5次。
- 表格为三trial中位数的中位数与半极差；P95为各trial nearest-rank P95的中位数。不是单请求P95。
- 比例为同trial参考/候选中位数后再跨trial汇总；不是汇总耗时直接相除，也不是模型吞吐。
- Event、wall、Graph、硬件Task分开；本报告不把历史prototype Task快照标为当前生产采样。

## 预校验完整链

### graph_per_call_us

|dtype/backend|父版中位±半极差 / P95，µs|候选中位±半极差 / P95，µs|参考中位±半极差 / P95，µs|父/候选|合格参考/候选|
|---|---:|---:|---:|---:|---:|
|bfloat16/ascendc|395.03±3.15 / 400.59|343.25±1.15 / 348.93|293.04±5.13 / 297.97|1.151×±0.005|失格，不给比例|
|bfloat16/pto|455.32±2.35 / 459.61|407.88±2.41 / 413.49|292.43±2.22 / 296.14|1.113×±0.002|失格，不给比例|
|float16/ascendc|339.58±0.11 / 343.63|294.68±0.68 / 299.09|295.93±2.90 / 300.31|1.152×±0.003|100.54%±0.81|
|float16/pto|387.37±1.94 / 392.21|339.15±1.31 / 344.36|296.95±3.50 / 300.18|1.139×±0.004|87.92%±1.19|

### eager_event_us

|dtype/backend|父版中位±半极差 / P95，µs|候选中位±半极差 / P95，µs|参考中位±半极差 / P95，µs|父/候选|合格参考/候选|
|---|---:|---:|---:|---:|---:|
|bfloat16/ascendc|1669.23±19.37 / 1749.56|1669.23±19.45 / 1751.92|1118.57±12.53 / 1262.36|1.000×±0.000|失格，不给比例|
|bfloat16/pto|1660.93±21.93 / 1732.94|1666.39±27.29 / 1750.46|1126.75±16.18 / 1261.98|0.997×±0.003|失格，不给比例|
|float16/ascendc|1555.38±12.85 / 1635.50|1560.93±10.47 / 1619.26|1119.45±8.49 / 1247.40|1.000×±0.006|71.72%±1.03|
|float16/pto|1525.56±18.44 / 1614.50|1523.59±15.16 / 1612.88|1089.07±16.91 / 1204.92|1.000×±0.003|71.26%±1.70|

### eager_wall_us

|dtype/backend|父版中位±半极差 / P95，µs|候选中位±半极差 / P95，µs|参考中位±半极差 / P95，µs|父/候选|合格参考/候选|
|---|---:|---:|---:|---:|---:|
|bfloat16/ascendc|1742.40±20.69 / 1855.43|1750.73±21.52 / 1831.17|1215.12±7.82 / 1375.58|0.999×±0.003|失格，不给比例|
|bfloat16/pto|1739.78±17.28 / 1827.74|1742.92±25.10 / 1825.68|1217.75±12.39 / 1397.45|0.998×±0.005|失格，不给比例|
|float16/ascendc|1634.67±8.19 / 1707.97|1637.25±13.31 / 1747.06|1212.00±7.44 / 1353.03|1.001×±0.004|74.03%±1.06|
|float16/pto|1608.01±20.82 / 1752.26|1599.73±18.25 / 1703.89|1174.87±16.07 / 1310.36|0.996×±0.006|73.48%±1.62|

## 共同LP注意力核心

### graph_per_call_us

|dtype/backend|父版中位±半极差 / P95，µs|候选中位±半极差 / P95，µs|参考中位±半极差 / P95，µs|父/候选|合格参考/候选|
|---|---:|---:|---:|---:|---:|
|bfloat16/ascendc|156.98±0.56 / 160.81|101.18±2.49 / 103.74|73.50±1.23 / 77.12|1.562×±0.038|失格，不给比例|
|bfloat16/pto|174.83±1.91 / 178.53|122.43±0.87 / 124.68|75.21±1.81 / 77.07|1.433×±0.012|失格，不给比例|
|float16/ascendc|132.63±1.11 / 134.34|88.67±0.54 / 92.24|73.47±2.71 / 75.90|1.481×±0.012|82.86%±2.55|
|float16/pto|144.55±0.95 / 146.62|97.23±0.66 / 100.35|73.24±1.20 / 75.38|1.484×±0.017|75.07%±0.85|

### eager_event_us

|dtype/backend|父版中位±半极差 / P95，µs|候选中位±半极差 / P95，µs|参考中位±半极差 / P95，µs|父/候选|合格参考/候选|
|---|---:|---:|---:|---:|---:|
|bfloat16/ascendc|542.82±30.24 / 602.50|543.64±32.75 / 598.46|335.89±25.01 / 414.46|0.998×±0.004|失格，不给比例|
|bfloat16/pto|527.54±2.11 / 571.12|527.90±3.13 / 559.02|327.52±0.07 / 369.22|1.005×±0.005|失格，不给比例|
|float16/ascendc|515.35±5.83 / 556.70|510.72±7.76 / 565.24|318.05±6.00 / 380.56|1.005×±0.006|61.28%±0.74|
|float16/pto|521.49±6.55 / 552.90|524.45±5.69 / 572.04|327.21±4.88 / 389.34|0.997×±0.003|62.33%±0.29|

### eager_wall_us

|dtype/backend|父版中位±半极差 / P95，µs|候选中位±半极差 / P95，µs|参考中位±半极差 / P95，µs|父/候选|合格参考/候选|
|---|---:|---:|---:|---:|---:|
|bfloat16/ascendc|623.70±41.16 / 723.36|622.95±43.01 / 681.19|410.00±36.07 / 524.27|0.996×±0.005|失格，不给比例|
|bfloat16/pto|597.82±3.65 / 701.20|597.92±5.40 / 686.03|399.60±3.87 / 482.27|0.997×±0.004|失格，不给比例|
|float16/ascendc|586.28±8.37 / 680.29|582.79±12.36 / 675.88|392.01±8.20 / 496.32|1.006×±0.007|65.97%±0.66|
|float16/pto|595.94±5.06 / 686.09|599.76±5.11 / 678.60|400.74±6.14 / 502.09|0.998×±0.003|66.76%±0.48|

## 精度、源码与历史归属

原预算rtol0.01、atol0.002、max_normalized<=0.005未放宽。候选与父版全部三seed及计时后检查通过；
BF16外部参考若未同时通过相同资格，保留诊断时间但比例为null，不能据其诊断耗时宣称达标。
逐trial原JSON、25源、driver/helper/protocol、输入/metadata及实际CPP均与独立审计匹配；全部10800个原始时间样本及表格已复算。

旧核心77.58%/83.11%来自prototype17387529；旧完整链previous为7c13旧control，非a172父版。
它们保留作历史，不拿来填写本报告。不同节点的旧绝对值不直接相减。
A9只在单KV块、head对齐、batch1启用；GM双槽与UB+128带来资源成本，其他形状保留回退。
本工作点比例不外推全部shape、普通调用、packed/model调用或完整模型吞吐。

## CI与模型证据的独立状态

目录路由含原L0：core 243项、扩展44项；两车道pytest返回0、artifact审计通过。完整进程分别248.367132s/201.838996s。
本性能导出器不启动NPU、不重跑测试、不把CPU/历史包内serial或旧SHA托管CI结果当当前完整目录验收。

core包装器返回1，pytest返回0，120秒目标=False；功能/精度及artifact通过不把该耗时目标改写为通过。
120秒是包内额外本地目标，不是托管CI判据；PR模板测试成本要求和托管CI/review仍需独立处理。

已附BF16父/候选只读审计：四run原状态仍diagnostic_only_failed；三phase对FP64的W_O前后门全部通过。
在已测B2 packed/fallback随机权重真实维度fixture中，父/候选保存位模式与实际CPP一致，未见A9新增失败；原跨官方门仍失败。
这不是新B1流水入口的模型集成通过，也不是checkpoint/完整模型质量验证。完整原始参数/hidden没有独立跨run字节快照。

## 复现入口

先加载兼容的CANN和已编译TileLang环境。将父a172整包正常安装到独立SITE，候选source在SITE之外；
从源码树外使用verify_installed.py生成新inventory。不要把优化前更早7c13 site当父control。

~~~bash
python -m pip install --no-deps --no-build-isolation --target "$SITE" "$PARENT_PKG"
PYTHONPATH="$SITE:$PYTHONPATH" python "$PKG/benchmark/verify_installed.py" --source-root "$PARENT_PKG" --output "$OUT/parent_inventory.json"
PYTHONPATH="$SITE:$PYTHONPATH" python "$PKG/benchmark/prefill_optimization_benchmark.py" --variant-file "$PKG/gmla/kernels/prefill.py" --scope full-chain --dtype float16 --target pto --mode benchmark --installed-root "$SITE" --inventory "$OUT/parent_inventory.json" --alignment-tools "$PKG/benchmark" --protocol-root "$PKG/benchmark/protocol" --seeds 0 1 17 --rounds 6 --repeats 10 --graph-repeats 5 --graph-batch 5 --warmup 10 --order-seed 1700 --output "$OUT/trial_fp16_pto_t1"
~~~

分别对core/full-chain、float16/bfloat16、pto/ascendc执行，各用1700/1701/1702三个独立新进程与新输出目录。
可核验的去隐私原始样本见同名.public.json；解码器expand_public恢复所有保留字段，去重展开后与sanitized原结构完全相等。
私有绝对命令/路径和设备UUID被显式替换，原JSON SHA保留。因此公开文件是声明去隐私后的无损编码，不是假称完整原始JSON零删改。


## 正常安装与当前硬件实测

正常wheel安装8组合全部通过原3seed与实际CPP检查，每格匹配全部3个正式候选trial。25生产源相同；wheel冻结元数据/README与后补文档不混称。详见[验证摘要](PREFILL_A9_VALIDATION_20261113.json)。

当前PTO/FP16核心各3次独立msprof：父Task140.64±0.90µs，候选98.98±0.47µs，全部20/40 blocks、1800MHz；精度及formal CPP匹配。它不是完整链、Graph或FIA比例。

|Pipe字段，µs|父版|候选|
|---|---:|---:|
|aic_cube_time(us)|14.66|14.66|
|aic_mte1_time(us)|29.07|29.07|
|aic_mte2_time(us)|17.25|16.50|
|aic_scalar_wait_id1_time(us)|37.19|0.32|
|aic_scalar_wait_id3_time(us)|19.55|0.93|
|aiv_vec_time(us)|20.54|16.02|
|aiv_scalar_wait_id0_time(us)|23.20|17.79|
|aiv_scalar_wait_id2_time(us)|43.81|6.14|

先取各核中位、再取3次中位；字段可能重叠，不能相加还原耗时。变化与减少互等的调度设计一致，不能凭计数证明全部算术逐位相同；位模式对照仅在已明确的B2模型回退fixture获得。[原12CSV及摘要](PREFILL_A9_HARDWARE_20261113.json)。

## 最终目录验收与仿真边界

新增3个CPU符号协议回归后，最终core243通过（pytest236.02s、完整248.367132s、pytest0/artifacts通过、wrapper1/120s失败）；44扩展在此前独立车道201.838996s通过，其生产/NPU tests及low_priority选择未改。两lane分次执行，不伪称同一运行。

当前PTO/FP16新源仿真prepare/direct通过，simulator阶段900秒超时，audit未执行，整体status=failed、原因未定；不记为新A9仿真通过，其他3组合仍是旧源历史。真实NPU实测通过不将该补充项改为绿色。[实际终态](PREFILL_A9_SIMULATOR_20261113.json)。
