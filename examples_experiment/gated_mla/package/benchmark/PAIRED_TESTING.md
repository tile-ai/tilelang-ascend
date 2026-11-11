# Gated MLA 平衡顺序性能测试

本轮完善计时与证据，不改变生产算子或旧版冻结协议。正式结果要求真实 NPU；CPU/fake-backend 验证只证明参考计算、调度与报告逻辑。

## 测什么

矩阵为三个场景 × 两种 dtype × 两种后端，每个组合使用三个独立进程，共36个 trial：

- `decode4096`：B1/H96/Tq1/Tk4096，全部可见。
- `decode17`：相同物理尺寸，可见前缀17。
- `prefill128`：B1/H96/Tq=Tk128，causal、全部有效 query。
- dtype：FP16/BF16；候选后端：默认 PTO、显式 AscendC。

每个 trial 用0/1/17三个 seed 检验候选和两条参考，计时固定首个 seed 0。其他多 batch/head、长 cache、续接等由已有功能测试覆盖，不把本矩阵外推为全部 shape 的性能保证。

## 三种计时口径

| 字段 | 边界 | 解释 |
|---|---|---|
| `eager_event_us` | NPU start/end Event 包围一次完整公开 callable | 可能包括等待主机下发的流内空隙，不是各 kernel Task 耗时之和 |
| `eager_wall_us` | Python 调用前到 end Event 同步结束 | 包括公开入口、分配/调度与同步，不含 start.record 的主机调用 |
| `graph_per_call_us` | graph replay 的 Event 区间除以 capture 调用数 | decode每图50次、prefill每图5次；是批量摊销值，P95也不是独立普通调用的尾延迟 |

输入生成、H2D、CPU参考、JIT与静态权重准备在计时外；原始 callable 中的运行期运算仍在计时内。输入和权重复用，不清空L2。测量范围到 gate 后、W_O 前，外部投影/归一化、cache append、通信和整模型服务不在本对照范围。

## 怎样控制顺序与波动

每轮按候选、历史 FP32 FIA 组合、vLLM 源路径组合的六种全排列之一运行。六轮恰好覆盖全部排列，默认每轮每方法测10次普通调用。图为每方法一次 capture，五次预热 replay，随后六轮各五次交替 replay。

每轮结束在计时外复核输出、输入与静态常量。候选任何精度/只读失败都终止后续工作。

每个 trial 保存原始样本、顺序、median、p90、p95、标准差。跨进程给出 trial 中位数的中位数与范围；不混合原始样本，也不把某一次最小值写成代表结果。

## 对标资格

预算保持 `rtol=0.01, atol=0.002, max_normalized<=0.005`，展开/吸收两份 FP64 参考独立检查。每种计时边界分别要求初始三 seed 和后验检查全部通过；graph 不可用时仍可保留合格的普通调用结果。

`--time-unqualified` 仅允许诊断耗时，不能得到性能比例。不合格参考的比例永远为空。源码、输入/metadata 字节、运行时或计时配置不同的 trial 拒绝合并。gate 默认采用 SHA/AST 固定的原 Triton 实现，直接 launcher 不等于完整 vLLM 服务调用。

## 复现

先按原 benchmark README 准备兼容的 CANN/TileLang/Torch-NPU/Triton 环境，并正常安装候选包。需要完整 source package 的 FILELIST、旧 protocol 和参考 adapter，不能只拷一个脚本。工具不会安装或全局升级依赖。在 package 目录执行：

```bash
python benchmark/run_paired_suite.py \
  --benchmark-root /path/to/package/benchmark \
  --installed-root /path/to/installed/site \
  --inventory /path/to/pinned_inventory.json \
  --output /new/full_matrix --trials 3 --time-unqualified
```

加 `--dry-run` 只写计划、不初始化设备。先试单个组合可增加 `--cases decode17 --dtypes float16 --targets pto`。必须串行使用同一 NPU，不能与另一个 benchmark/profiler 并发。

`suite.json` 保存每个子进程状态和超时；单 trial 出错后不启动后续试验。`summary.json/SUMMARY.md` 汇总成功 trial，但未完成计划必须标为 partial。工具源码、候选源文件、参考代码和精度记录均保留哈希。

另用 `host_decode_diagnostic.py` 对同一公开候选入口做主机归因。cProfile 与未插桩计时分开，inclusive 行会重叠，不能相加，也不能直接称为删掉校验后的收益。具体接口见 `HOST_DIAGNOSTIC.md`。
