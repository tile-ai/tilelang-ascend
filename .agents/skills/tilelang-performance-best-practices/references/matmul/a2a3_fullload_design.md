# Cube 小侧 Full-load

## 适用证据

同一核负责多个对侧输出 tile，A 或 B 的相同 GM 区间被反复搬入，且该数据在整个
复用区间保持不变。若 `baseM×K×dtype_bytes` 加其他活跃 L1 buffer 可容纳，考虑 A
全 K 常驻；B 侧对称。`L1/2` 是为另一侧双缓冲留空间的估算方式，不是固定比例要求。

## 候选结构

```text
before: for output_tile: for K_tile: load A; load B; compute
after:  load reusable A once
        for output_tile: for K_tile: load B; read corresponding A slice; compute
```

若小侧本来会重复搬入 T 次，理论可消除其 `(T-1)/T` 重复读取；这是该数据的读取量，
不是整个算子的流量或耗时下降。矩阵过大时可考虑更小范围的常驻复用，避免为了
“全量”破坏原有双缓冲。

## 容量与生命周期

预算包含常驻小侧、流式大侧全部槽、scale/bias、布局转换及其他活跃 L1 区域。
常驻区域不能被内存复用、地址别名或下一任务预取提前覆盖。MTE2 生产后，在所有
相关 MTE1 读取结束之前保留 ownership；在第一次消费后释放常驻区域是错误的。

采用 Fixed Core 改变任务顺序时重新检查复用成立：跨 batch、不同转置或不同 A/B
指针不能沿用旧常驻值。初始 tile/tail padding 也要与被复用 shape 一致。

方案给出“同一核重复引用的地址范围、常驻生命周期、减少字节数、增加 L1 字节数”，
执行阶段再验证实际收益。该方法继承旧 guide §2.4 的原则，本次未构造或实测新 kernel。
接口与存储约束见 [Matmul 索引](guide.md) 和 [tiling](tiling_design.md)。
