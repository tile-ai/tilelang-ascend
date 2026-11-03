# GEMM 一侧驻留

## 适用条件

A 或 B 的完整可复用 panel 连同 scale 可放入 L1，并被多个对侧输出 tile 复用。

### 适用证据

同一核负责多个对侧输出 tile，A 或 B 的相同 GM 区间被反复搬入，且该数据在整个
复用区间保持不变。若 `baseM×K×dtype_bytes` 加其他活跃 L1 buffer 可容纳，考虑 A
全 K 常驻；B 侧对称。`L1/2` 是为另一侧双缓冲留空间的估算方式，不是固定比例要求。

## TileLang 实现流程

把 resident panel 的 T.copy 移到输出 tile 遍历外；流式侧仍按 K tile 搬运。scheduler 的遍历方向应让复用侧在连续任务间保持不变。L1 预算包含 resident panel、scale、流式 buffer versions 和安全余量。

基础实现参考 [当前 GEMM 示例](../../../../../examples/gemm/example_gemm.py) 与 [显式 MMA 示例](../../../../../examples/gemm/example_gemm_intrinsic.py)：按当前 B 的物理布局设置 transpose_B，使用 T.gemm_v0 或 T.mma；浮点 GEMM 的 L0C 使用 fp32，并且只在第一个有效 K tile 设置 init=True。输出 tile 的核数不超过独立任务数；M/N/K 尾块必须走已验证的 padded-copy 或专用 fallback。

### 候选结构

```text
before: for output_tile: for K_tile: load A; load B; compute
after:  load reusable A once
        for output_tile: for K_tile: load B; read corresponding A slice; compute
```

若小侧本来会重复搬入 T 次，理论可消除其 `(T-1)/T` 重复读取；这是该数据的读取量，
不是整个算子的流量或耗时下降。矩阵过大时可考虑更小范围的常驻复用，避免为了
“全量”破坏原有双缓冲。

## 精度门禁

涉及量化 scale 时，其索引、block size 和应用顺序必须与目标算子的 blockscaled reference 一致；普通 GEMM 不因本设计自动获得量化接口。

所有输入 dtype 与输出 dtype 分开测试。覆盖 M/N/K 的 tile±1、长 K 消除误差、正负抵消、大小量混合、0、NaN/Inf 契约。浮点 GEMM 的任何 K 分片的 partial 和最终归约保持 fp32；不得以放宽容差掩盖累加顺序或输出转换错误。

### 容量与生命周期

预算包含常驻小侧、流式大侧全部槽、scale/bias、布局转换及其他活跃 L1 区域。
常驻区域不能被内存复用、地址别名或下一任务预取提前覆盖。MTE2 生产后，在所有
相关 MTE1 读取结束之前保留 ownership；在第一次消费后释放常驻区域是错误的。

采用 Fixed Core 改变任务顺序时重新检查复用成立：跨 batch、不同转置或不同 A/B
指针不能沿用旧常驻值。初始 tile/tail padding 也要与被复用 shape 一致。

方案给出“同一核重复引用的地址范围、常驻生命周期、减少字节数、增加 L1 字节数”，
执行阶段再验证实际收益。该设计的具体 kernel 与收益待验证。
接口与存储约束见 [Matmul 索引](guide.md) 和 [tiling](tiling_design.md)。

## 性能门禁

比较节省的 GM bytes 与 resident 初始化成本；重复次数不足两次不启用。

对每个候选运行编译、目标后端的定向精度测试和统一 benchmark。报告 latency、TFLOPS、Cube/MTE2 时间、GM 字节、L1/L0/UB 占用、核利用率和生成代码体积。只有完整算子端到端更快且没有精度回退时进入 dispatch。
