# 非对齐尾块实现

## 目标与适用条件

用于元素数、行宽或矩阵维度不能整除 DMA、SIMD 或 tile 粒度的路径。

## TileLang Ascend 实现

GM 侧只访问 valid 元素；UB 按可能触达的完整 Vector 指令 footprint 分配。Reduction 的无效 lane 写入幺元；transpose/gather 的输出只回写有效矩形。能在 Python factory 中按余数类生成专用 kernel 时，优先消除热循环内的动态分支。

实现沿用当前仓库 T.Kernel 的网格和 `(cid, vid)` 映射。纯向量任务从已确认的可用 AIV 核数及 launch 映射取核数，并以 min(核数, 独立任务数) 限制空核。GM 与 UB/L1 之间使用 T.copy；连续规则计算使用当前模式的 T.Parallel 或 T.tile 原语；跨 tile 任务使用当前 T.Persistent、Fixed Core 核内循环或已验证的 T.Pipelined，多版本方式沿用当前编译器支持的流水或显式槽位所有权。

## 精度门禁

对每种 dtype 分别验证 DMA 对齐、SIMD lane 数和转换后的 fp32 footprint。用 canary 检查尾部前后没有越界写。

必须覆盖最小 shape、常见 shape、最大 shape、tile-1/tile/tile+1、非 32B 尾块、0、正负极值以及接口规定的 NaN/Inf 语义。不得通过扩大容差、降低 reference 精度或跳过 case 获得通过。

## 性能门禁

分别 benchmark 整 tile 与 tile±1，防止为尾块增加的分支拖慢主路径。若专用 kernel 数量过多导致编译或缓存压力，则合并低频余数类。

先用目标仓库既有后端与命令运行定向精度测试，再运行相关完整测试套；全部通过后，在相同输入、dtype、warmup、repeat、设备和并发条件下测量性能。报告 kernel latency、有效 GM 带宽、UB 占用、stage 数和基线差异。

## 可执行代码与证据

参考 [transpose 测试](../../../../../testing/python/language/test_tilelang_ascend_language_transpose.py) 和 [尾块测试](../../../../../testing/python/language/test_tilelang_ascend_language_tail_block.py)。

---

## A2/A3 实现补充与约束

## 尾块、padding 与边界所有权

### 有效范围与物理 footprint

GM 读写必须处于接口允许范围，UB 按指令会触达的完整 footprint 分配。`valid` 限制 GM 不等于 UB padding 已初始化；sum 填 0、max 填负无穷、min 填正无穷，整数使用可表示的对应幺元。NaN、全遮蔽行和无有效元素的语义单独按接口处理。

主路径与尾部可分离：静态 JIT 维度使用 Python `//`、`%` 得到 `n_full/partial`；真正动态维度用当前 DSL 已支持的 device 控制流，不要求把所有动态接口改成静态。`partial==0` 不分配零长度 buffer、不访问尾部。

在合法的 GM→UB pad 路径上，先 fill 再 copy；后 fill 会覆盖有效数据。`pad_value` 的支持范围以当前 [copy 实现](../../../../../tilelang/language/copy_op.py) 和 [尾块测试](../../../../../testing/python/language/test_tilelang_ascend_language_tail_block.py) 为准，不能推广到所有层级。编译器只保证的 padding 范围之外仍需显式初始化。

### 两种输出契约

1. **精确输出**：输出 tensor 为原逻辑 shape，GM/UB 两侧均使用 `valid` slice。
2. **内部 padded 输出**：wrapper 确实分配 padded storage，kernel 签名与写入物理范围一致，最后裁剪。返回 view 的 stride、连续性和后续 reshape 是否复制也要检查。

旧 GroupNorm 示例使用第二种，不意味着所有算子都必须扩大输出。只修改签名不会为用户提供的 tensor 增加存储；不能用 host 裁剪掩盖越界写。

### 多行与多 batch

输入始终声明真实维度；tail tile 可分配完整 UB，但 `T.copy(x[b, m, n], full_ub)` 仅给标量 GM 起点，不能证明后续完整 tile 留在该 batch。输入和输出同时写成有界二维 BufferRegion，并核对当前 lowering 是否裁剪 block 端点；若未证明，显式使用 `valid_m/valid_n`。

`M/N < tile`、两个轴同时非整除、多 batch、最后 batch 是必测组合。详见 [多阶段转换](../conversion/multi_stage_contiguous.md)。fixed-core 只能改变任务分配，不能修复单任务跨 batch。

### 首尾与流水

每个新 task 的状态独立初始化；每个复用 stage 的无效 lane 都必须安全。测试 0/1/2/3 次迭代、stage±1 次、tile±1 元素、odd rows、双尾块及 case 顺序变化。可用 canary、唯一索引和稀疏 NaN mask 检查边界；全零或全 NaN 不能充分暴露错位。

接口只承诺特定对齐时覆盖允许余数类，不凭优化技能扩大产品合约。任何 fallback、尾块或空任务变化都需全量 dispatch 覆盖。
