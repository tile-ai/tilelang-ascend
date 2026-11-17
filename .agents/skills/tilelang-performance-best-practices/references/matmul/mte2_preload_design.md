# GEMM 提前发射与深流水

## 适用条件

基础 2-stage 已正确，但生成代码中下一 K tile 的 copy 启动晚于可用时机。

### 适用与排除

当 GM→L1 的等待暴露在每轮计算前，K 方向至少有两次搬运，且下一轮 L1 目的槽
可在当前轮计算期间取得所有权，可考虑 MTE2 预取。只有一轮、数据已常驻、容量无法
容纳另一槽，或瓶颈完全在其他串行依赖时，不应预期预取必然获益。

GM→L1 属于 MTE2，L1→L0 属于 MTE1，两种预取必须分开分析；后者属于 [L0 Ping-Pong](pingpong_design.md)。

## TileLang 实现流程

按当前模式通过可支持的 T.Pipelined、物理 buffer 槽数和循环嵌套表达预取。只有生成源码和 timeline 证明 copy 前移，才保留配置；Expert 手动路径按当前 LOCK DSL/生成事件的所有权合同实现。

基础实现参考 [当前 GEMM 示例](../../../../../examples/gemm/example_gemm.py) 与 [显式 MMA 示例](../../../../../examples/gemm/example_gemm_intrinsic.py)：按当前 B 的物理布局设置 transpose_B，使用 T.gemm_v0 或 T.mma；浮点 GEMM 的 L0C 使用 fp32，并且只在第一个有效 K tile 设置 init=True。输出 tile 的核数不超过独立任务数；M/N/K 尾块必须走已验证的 padded-copy 或专用 fallback。

### 三阶段方案

1. Prefetch：获得 slot 0，搬入第 0 个 GM tile，发布 MTE2→MTE1 就绪。
2. Steady state：在另一槽可用时搬入第 k+1 个 GM tile；消费第 k 个 L1 tile 的
   全部 L0 子块；最后一次 MTE1 读取后才归还该 L1 槽。
3. Epilogue：消费最后一个已预取 tile，完成 L0C 导出和所有 token 的 drain。

```text
first:      load GM[k=0] → L1[0]
iteration:  load GM[k+1] → L1[(k+1)%S1]   (存在下一块且目标槽已归还)
            consume L1[k%S1] → L0 → MMA
last:       consume final tile; export accumulated C; drain
```

这是调度说明，完整同步和范围检查由所选 kernel 合同决定。参考
[intrinsic GEMM](../../../../../examples/gemm/example_gemm_intrinsic.py)
的 GM 首轮预取结构时，须同时处理其固定四次 L0 读取和整除形状假设。

## 精度门禁

预取版本不得覆盖仍被 T.gemm_v0/T.mma 路径消费的 L1 tile。

所有输入 dtype 与输出 dtype 分开测试。覆盖 M/N/K 的 tile±1、长 K 消除误差、正负抵消、大小量混合、0、NaN/Inf 契约。浮点 GEMM 的任何 K 分片的 partial 和最终归约保持 fp32；不得以放宽容差掩盖累加顺序或输出转换错误。

### 必须保持的条件

- 目的槽未归还前不发起 MTE2；数据未就绪前不让 MTE1 消费。
- A/B 同属于一次 MMA 的数据必须匹配同一个 K 范围，不能分别轮转错位。
- K 尾块的有效范围、padding 和累加初始化与原算子语义一致。
- 持久任务循环切换输出 tile 时，区分“上一任务的末块”和“下一任务的首块”。
- 当前 kernel 的自动同步若阻塞重叠，先指出具体依赖与生成位置，不凭开关名推断。

后续执行阶段比较完整 kernel 的耗时、等待和占用；旧“busy≤70%”只是历史启发，
不是预取的硬件门槛。具体候选的性能待上板验证。

## 性能门禁

以生成代码和 timeline 验证实际前移；若仅增加内存没有缩短 gap，则回退。

对每个候选运行编译、目标后端的定向精度测试和统一 benchmark。报告 latency、TFLOPS、Cube/MTE2 时间、GM 字节、L1/L0/UB 占用、核利用率和生成代码体积。只有完整算子端到端更快且没有精度回退时进入 dispatch。
