# TileLang A2/A3 精度与性能门禁

## 实施顺序

1. 优先从当前仓库同类算子实现派生；使用 bundled code 前先查 [模板成熟度](template_status.md)。
2. 通过实际导入路径定位 TileLang，核对每个 API 在 当前后端 lowering 中存在；不使用 CUDA-only schedule、target 或 pass config。
3. 沿用原 reference 和容差；首次建立 reference 时使用 fp32 或更高精度 PyTorch reference，先实现公开接口承诺的完整 shape/dtype fallback。
4. 运行定向精度测试；修复全部数值、越界、编译和未实现路径。
5. 一次只改变 tile、核数、stage、resident、执行域、融合中的一个变量。
6. 每次变更重新跑定向精度测试，再在固定环境 benchmark。
7. 代表 shape 无回退后运行相关完整测试套；接口要求时继续项目要求的扩展测试。

## 精度矩阵

先读取公开接口合约，再覆盖最小/常见/最大 shape、允许余数类、动态维、非连续 stride、所有 dtype、forward/backward、0、正负极值、抵消、NaN/Inf 与空任务。接口承诺通用尾块时增加 tile-1/tile/tile+1；接口只允许特定对齐时不得把未实现 fallback 当成既有能力。Reduction、Norm、Softmax、Scan carry 和 GEMM L0C 默认 fp32；输出只在最终边界转换。

记录测试收集数、通过数、首个失败、shape、dtype、容差、异常类别和根因。`NotImplementedError` 是未实现，不是通过。禁止扩大容差、降低 reference 精度、跳过 case 或只报告此前通过的子集。

## 性能矩阵

保持输入分布、shape、dtype、输出语义、warmup、repeat、设备、频率和并发一致。报告 kernel latency、端到端 latency、有效 GM bytes/带宽、Cube/Vector/MTE 时间、核利用率、UB/L1/L0/工作区字节、stage 和生成代码体积。

多 kernel 与 host pipeline 必须计入全部 launch、event、workspace 与同步。没有当前设备实测数据时只报告设计，不报告收益。

## 命令

```bash
pytest <test_file> -x
pytest <test_file>
```

当前技能未提供上述其他平台的 bundled Python 模板和回归脚本；使用目标已有测试入口。单文件内嵌测试沿用其实际命令，不强制改成独立 pytest。执行者先按目标工程要求加载环境，保持原后端与编程模式。

正确性测试仅在确认设备绑定和隔离机制后按可用设备数并发，否则串行；OOM 时降低并发，不减少用例。benchmark 保持设备独占。

## A2/A3 依赖、正确性与职责补充

一次只改变一个变量用于隔离归因；若合法数据流必须同时调整布局、buffer、同步或 dispatch，作为同一方案说明依赖，不按 Edit 次数拆断必要耦合。

检查生成代码中的 C/V 归属、MTE2/MTE1/Vector/MMA/MTE3/FIX 依赖、buffer alias、初始化 token、稳态复用和末尾 drain。使用 [硬件资源与预算](common/hardware_resource_discovery.md) 重算 padding、临时量与版本。复用多阶段 layout kernel 前同时证明输入输出有效 region、多 batch、双尾块和 dtype 静态变体。

本文件是后续执行阶段的验收要求；分析技能只设计验证。实施者负责代码与正确性，调用方组织采集、最终采纳及回滚。精度失败不得成为最终基线，缺少本轮设备数据不得宣称已获性能收益。
