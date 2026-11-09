# 核内流水、CV overlap 与配置

## 先固定依赖和现有模式

代码中的 `IS_ASCEND_AIC/IS_ASCEND_AIV/KERNEL_TYPE_MIX` 可帮助定位生成执行侧，不能用 compute 占比阈值推断 kernel 类型。按用户已有 Developer/Expert/混合模式实施，配置细节遵循 [编程模型指南](../../../tilelang-custom-skill/tilelang-programming-model-guide/SKILL.md) 和 [T.Pipelined 教程](../../../../../docs/tutorials/t_pipelined.md)。

| 路径 | 起点与核验 |
|---|---|
| Developer/Hybrid 自动流水 | 保留自动同步/规划，检查实际版本、通知等待和 overlap |
| `T.gemm_v0` 同类模板 | 核对模板内部的 L1/L0 流水和参数，不能用不存在的 L0 入参组合代替其契约 |
| Expert `T.mma` / 手工 Vector 双缓冲 | 新 kernel 用可编辑 LOCK 源及相应预处理器表达所有权；当前仓已有 low-level/legacy flag 可作为生成层参考 |
| 需要 MTE2↔MTE1 或 MTE2/V/MTE3 精确排布 | 若生成自动同步阻塞已证明可重叠的依赖，再针对已完成所有权证明的手动路径调整 `AUTO_SYNC` |
| CV workspace 交互 | 自动识别成立时启用 `TL_ASCEND_AUTO_CV_SYNC`；否则按已验证的跨核所有权方案实施 |

真实 LOCK 参考为 [kernel.tl](../../../../../examples/flash_attention/fa_opt/expert_v3/kernel.tl) 和 [预处理器](../../../../../examples/flash_attention/fa_opt/expert_v3/tools/preprocess.py)，不规定不存在的统一源文件名。

纯 Vector 是否调整 AUTO_SYNC 由依赖和生成代码决定，不能根据固定收益比例推断同步状态。`AUTO_SYNC=False` 不会自动建立正确流水；仅删除 barrier 或交替 dst 也不能代替完整依赖。检查 [同步插入](../../../../../src/transform/ascend_sync_insert.cc) 与生成代码。纯 Vector 的 CombineCV 不必默认关闭；若出现 `alloc_var` 定义与使用跨核的具体错误，先定位 classifier，诊断回退关闭时补齐显式 V scope。

## 核内生命周期

每个槽位经历 `free → copy-in ready → compute complete → copy-out complete → free`。版本数量由异步生产/消费重叠决定，输入/输出/临时 buffer 逐项判断。V-only 临时量可共用的前提是上一消费者已完成，跨 pipe 的临时量同样可能需要版本。

Vector 低层事件常见 `MTE3→MTE2`（可覆写）、`MTE2→V`（输入就绪）、`V→MTE3`（输出就绪）。Cube 分开处理 GM→L1 的 MTE2、L1→L0 的 MTE1、MMA 的 M、输出 FIX；事件方向与真实生产/消费匹配，初始化及末尾 drain 必须成对。具体语法由当前 DSL/生成器决定。

正确 slot 推导见 [双缓冲专题](../elementwise/double_buffer_design.md)。预取第 0 块以后，第一轮必须消费 slot 0；从 `si=1` 配 `cur=si%2` 会先等待未填充 slot，这是不合法的轮转。

## CV 核间流水

只在存在跨 C/V 数据依赖时配置跨核同步；单纯含 GEMM 不代表需要。当前教程用 GM workspace 表达 C/V handoff，可能受益于 L2，但不能仅因 workspace 小于某历史 L2 容量就保证命中。连续布局、每 core 槽位复用和工作集大小都要计入。

候选从 stage 1 与 stage 2 配对开始，按循环次数、资源预算和实际气泡增加 stage；少于 stage 的有效迭代常难摊薄启停成本，保留浅流水。`num_stages` 增加时重新核算所有 UB/L1/L0/workspace。避免未经支持验证的嵌套 `T.Pipelined`；当前教程推荐外层 CV pipeline 配合 flat 的核内 ownership。

`cross_interval` 仅对已启用且正确识别的核间 pipeline 生效。先比较 1 与 2，检查 writer 在组末/最终尾组发布、reader 在组首消费、workspace 存活足够长。不能把每两次同步机械应用于所有生产消费关系。更大间隔是待验证候选，不承诺收益；同步频率降低也可能减少 overlap。

隔离同步开销可作为诊断实验，但删掉计算/搬运会改变依赖与优化结果，不能用空壳比例独立认定瓶颈。最终判断完整路径 latency 和 timeline。

## 生效与停止条件

检查生成 buffer 地址、版本切换、同步方向与首尾调度，然后在同输入/同语义下测 stage 1/2 latency、MTE/Vector/Cube overlap 和噪声。编译成功、正确性通过或两个 buffer 均不能单独证明生效。

自动分析失败只否决该组合。流水候选已经由每核独立迭代、容量和 MTE/Compute 可重叠性准入时，自动失败或部分生效后必须按 [多版本流水门禁](../elementwise/double_buffer_design.md) 完成手动显式多版本代表，或提供 API、lowering、容量或依赖的确定不可实施证据；不能只将它留为可选建议。自动组合完整生效时可记录 `NOT_NEEDED_AUTO_COMPLETE`，不再重复测试手动等价实现。所有候选记录 tile、stage、访问形状、控制流和同步配置；不能将单个 tile 的失败泛化为整个流水方向失败。零/单次循环、尾块、改变 case 顺序和连续多次调用必须纳入回归。
