# Cube MTE2 预取

## 适用与排除

当 GM→L1 的等待暴露在每轮计算前，K 方向至少有两次搬运，且下一轮 L1 目的槽
可在当前轮计算期间取得所有权，可考虑 MTE2 预取。只有一轮、数据已常驻、容量无法
容纳另一槽，或瓶颈完全在其他串行依赖时，不应预期预取必然获益。

旧文档用 `T.copy(L1,L0)` 解释 MTE2，混淆了 MTE1。这里将 GM→L1 预取与
L1→L0 预取分开；后者属于 [L0 Ping-Pong](pingpong_design.md)。

## 三阶段方案

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

## 必须保持的条件

- 目的槽未归还前不发起 MTE2；数据未就绪前不让 MTE1 消费。
- A/B 同属于一次 MMA 的数据必须匹配同一个 K 范围，不能分别轮转错位。
- K 尾块的有效范围、padding 和累加初始化与原算子语义一致。
- 持久任务循环切换输出 tile 时，区分“上一任务的末块”和“下一任务的首块”。
- 当前 kernel 的自动同步若阻塞重叠，先指出具体依赖与生成位置，不凭开关名推断。

后续执行阶段比较完整 kernel 的耗时、等待和占用；旧“busy≤70%”只是历史启发，
不是预取的硬件门槛。本轮未采集新数据或上板验证。
