# xAttention 示例

这里按版本组织已有的 **Expert v1** 和新增的 **Expert v2**。它们都计算共享前缀与每个 beam 私有
序列上的完整 attention，但计算组织和输入接口不同；v2 不是原接口的直接替换，
也不表示已经证明它在当前主线上普遍更快。

```text
xattention/
  README.md
  expert_v1/
    kernel.py
    paged.py
  expert_v2/
    kernel.py
    README.md
    history_measurements.csv
```

## 入口

- [`expert_v1/kernel.py`](expert_v1/kernel.py)：已有的非分页示例。
- [`expert_v1/paged.py`](expert_v1/paged.py)：已有的分页示例，通过块表寻址 K/V。
- [`expert_v2/kernel.py`](expert_v2/kernel.py)：新增的连续 K/V 输入版本；中文优化历程和
  历史对照数据与代码放在同一目录，见 [Expert v2 说明](expert_v2/README.md)。

v1 的两个脚本只移动路径，内容保持不变。旧路径不再保留一份重复入口。

## v2 与 v1 的区别

| 项目 | Expert v1 | Expert v2 |
|---|---|---|
| 共享/私有计算 | 分配不同的 Cube 核组，分别计算两部分 | 参与核处理共享前缀，两个私有 token 用 Vector 计算 |
| 最终合并 | 两部分输出与统计量写到全局内存，全局同步后再合并 | 在最后一个共享 key 块的输出更新中直接合并，复用片上数据 |
| 输入接口 | 调用端准备工作区及长度、块表等辅助输入 | 直接传入五个连续 BF16 张量；输出和工作区由 JIT 管理 |
| 私有序列长度 | 默认四个 token，另传运行时长度 | 固定两个 token |
| K/V 寻址 | 分别提供非分页、分页入口 | 只支持连续布局，不支持分页块表 |
| 参数设置 | 修改脚本中的配置 | 请求数、每请求 beam 数、共享长度由命令行指定 |

v2 固定 32 个 query 头、8 个 K/V 头和 128 的头维度；共享长度须为 1024 的整数倍，
每请求 beam 数须为 64 的整数倍。它沿用完整 attention 的数学目标，但只覆盖这里
明确列出的输入范围，不承担原有分页示例的功能。

默认运行 v2：

```bash
source ./set_env.sh
python examples/xattention/expert_v2/kernel.py
```

`bench_test.sh` 会发现并执行已有两个脚本和 `expert_v2/kernel.py`。
它检查运行与正确性，不会自动比较这些实现的性能；Markdown 和 CSV 不会执行。
