# BF16真实维度单层诊断（2026-11-13）

父a172整包与当前a9候选、PTO/AscendC、prefill17/continuation3/decode1均已实际运行并经CPU只读审计。
四个原状态仍为diagnostic_only_failed。候选/父版对同输入FP64的W_O前后门全部通过；原跨官方门未通过。
同后端各phase实际保存的操作数/输出位模式、生成CPP和失败门列表在父/候选间一致，
说明在这个确定性B2 packed/fallback fixture中未见A9新增回归。不能把这条结论写成原模型验收通过。

|phase|比较|max_abs|max_normalized|逐元素超预算|原门|
|---|---|---:|---:|---:|---|
|prefill|before_wo_vs_actual_official_layer|0.0078125|0.00505050505051|10|FAIL|
|prefill|after_wo_vs_actual_official_layer|0.00390625|0.00574712643678|1|FAIL|
|prefill|before_wo_vs_fp64_seven_input|0.00390427858991|0.00253470885263|0|PASS|
|prefill|after_wo_vs_fp64_core_plus_wo|0.00236146019797|0.00348860299923|0|PASS|
|continuation|before_wo_vs_actual_official_layer|0.00390625|0.00896860986547|0|FAIL|
|continuation|after_wo_vs_actual_official_layer|0.0009765625|0.00512820512821|0|FAIL|
|continuation|before_wo_vs_fp64_seven_input|0.000975821363229|0.00224819755442|0|PASS|
|continuation|after_wo_vs_fp64_core_plus_wo|0.000586389012612|0.00308153270564|0|PASS|
|decode|before_wo_vs_actual_official_layer|0.001953125|0.00431034482759|0|PASS|
|decode|after_wo_vs_actual_official_layer|0.0009765625|0.00537634408602|0|FAIL|
|decode|before_wo_vs_fp64_seven_input|0.000975696546521|0.0021502630027|0|PASS|
|decode|after_wo_vs_fp64_core_plus_wo|0.000510382832627|0.00281507805397|0|PASS|

原门同时要求逐元素0.002+0.01*abs(reference)和max_normalized<=0.005，因此逐元素0超预算不一定通过。
不放宽容差、不静默fallback，也不把官方dense路径与absorbed路径的差异直接定性为某方实现错误。

范围是H96/hidden7168/qrank1536随机权重单层，checkpoint_loaded=false；B2/N64走流水域外回退。
没有新B1流水模型集成、真实checkpoint、完整模型质量或服务吞吐结论。
每次运行断言权重/hidden只读，已保存外部输入和KV权重跨run逐位核对；
未独立保存全部模型权重与原始hidden，不宣称完整参数集做过跨运行字节快照。
NPU安装site未pull，未做逐库native运行时哈希证明。

[精简验证及原审计SHA](PREFILL_A9_VALIDATION_20261113.json)。完整tensor/CPP/结果归档由贡献者保留，不分发模型源码或checkpoint。
