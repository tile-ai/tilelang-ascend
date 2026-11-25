# Sklansky Prefix Network

## 适用条件

扫描轴静态、完整 tile 可驻留 UB，并且当前向量路径的 lane shift/gather/broadcast 能高效表达树形 combine。带 argmin/argmax 索引传播的 scan 不直接使用该模板。

## 算法

对长度 `R` 的 resident tile 展开 `ceil(log2(R))` 层。第 `k` 层的半组宽为 `2^k`，组宽为 `2^(k+1)`；每组使用下半组最后一个元素作为 anchor，与上半组全部 target 做结合：

```text
half  = 1 << k
group = half << 1
anchor = group_start + half - 1
target = [group_start + half, min(group_start + group, R))
```

Cumsum 的 combine 是加法，Cumprod 是乘法，无索引 Cummin/Cummax 分别是 min/max。尾组只更新真实 target，padding 不参与结果。上面的公式给出每层准确的 anchor/target 关系；层间依赖还须落实到可编译、可验证的 kernel。

跨 tile 时，当前 tile 的最后一个 prefix 作为 carry（浮点使用 fp32，整数保持精确 dtype），由同一个 row owner 串联到下一 tile。没有经过当前向量路径 microtest 前，使用与公开算子一致的高精度/精确 reference 作为 oracle，而不是把串行 fallback 称为 Sklansky kernel。

## 精度与性能门禁

覆盖每个 `2^k-1/2^k/2^k+1` 边界、非整尾部、多 tile carry、正负抵消及接口规定的溢出和 NaN/Inf 语义。性能上与 Kogge-Stone 比较 fanout、指令数、寄存器压力和生成代码长度。

## 实现状态

状态为 `DESIGN_ONLY`；链接示例只提供局部结构依据，不证明本策略已有端到端 kernel。实现后须完成上述精度与性能门禁。
