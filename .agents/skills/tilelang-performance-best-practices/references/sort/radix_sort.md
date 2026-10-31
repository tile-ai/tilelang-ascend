# TopK 与排序

## 实现

参考 `examples/topk_selector/example_topk_selector.py`、`examples/sort/example_merge_sort.py` 和 `tilelang/language/ascend_tile.py` 的 `topk`/`sort`/`merge_sort` 核对 TopK 接口、排序规则与当前实现。按目标接口明确越界 lane 的填充值和重复值的决胜规则；涉及尾块搬运时核对 copy pad value。不能将示例的固定 shape 或行为泛化为全部输入的保证。

大排序采用分层结构：Kernel A 对每个 tile 生成已排序候选，Kernel B 按行合并候选；需要全局 radix/histogram 时使用独立 workspace 和顺序 kernel launch，不依赖未验证的 grid barrier。sortable key transform 必须与 dtype、符号位、NaN 排序契约一致。

## 精度与性能

覆盖 K=1/边界/全长、重复值 tie-break、±0、NaN、±Inf、index dtype、尾块和稳定性。比较完整端到端 latency、workspace、GM bytes、Vector/MTE 时间与基线；多 kernel 必须计入全部 launch。

当前 TopK 示例的检查方式和输入范围不构成稳定排序、NaN/tie 或任意尾块的保证。分层排序与全局 radix/histogram 段落是设计候选，需实现并验证；源码参考不构成这些候选的设备验证结论。
