# RoPE 优化索引

## 实现

以 [当前 rope.py](../../../../../examples/pos_embedding/rope.py) 和 [两种布局示例](../../../../../examples/pos_embedding/RotaryPositionEmbedding/rope_half_interleaved.py) 为源码参考。Python factory 固化 layout、dim、dtype 与 position offset；按当前 T.Kernel 的 cid/vid 合同以 grid-stride 分配 token×head task。输入、sin、cos 通过 T.copy 进入 UB，使用当前已支持的 T.tile/T.Parallel 路径以 fp32 完成成对旋转，最后转换输出 dtype。

half-split 与 interleaved 必须生成不同 kernel，不能在热循环中动态分支。任意 position tensor 需要先验证当前后端的标量/离散索引 lowering；编译期 position offset 的连续位置是另一独立候选，不能扩大既有示例的语义。生产路径使用可 lowering 的 Tile API、跨 head 复用或算子融合，并附带硬件、版本、命令和原始性能结果。

### 适用范围

适合 RoPE / interleave / gather 中只由 shape 和固定排列决定的小索引或符号张量。
若每次调用在 CPU 生成并传入 NPU，或 kernel 为固定 mask 增加 GM→UB 搬运，可以
比较片上生成与原方案。若 mask 已长期缓存且生成成本更高，也可能没有收益。

参考 [rope_mask.py](../../../../../examples/pos_embedding/rope_mask.py) 与
[rope.py](../../../../../examples/pos_embedding/rope.py)。以下代码用于说明实现技术，支持范围以该示例的 shape/dtype 条件为准；新的候选需上板验证。

### 相邻元素交换的索引

对于 interleaved RoPE，局部旋转为 `[-x1,x0,-x3,x2,...]`。索引序列通过 `i^1`
交换相邻元素，乘以被 gather 数据的字节宽度得到偏移。源码先把 x 转成 fp32，
因此其偏移乘数是 4；更换计算 dtype 时不能照搬常量。

```text
createvecindex: 0,1,2,3,...
XOR 1:         1,0,3,2,...
byte offset:   swapped_index * sizeof(gathered_dtype)
```

示例用 `createvecindex(int32)`、转 int16、`bitwise_xor`、转 fp32、转 int32，
然后 `reinterpretcast` 为 uint32 mask。这条组合体现了索引类型与指令约束，
但“int16 一定更快”“中转 float 一定最优”没有独立测量支持。先查
[ascend_tile.py](../../../../../tilelang/language/ascend_tile.py) 的当前 dtype 支持，
再决定是否有等价的短路径。

int16 中间索引必须在可表示范围内；多行 flatten 后的最大索引也要算入，不能只看
单行 `rope_dim`。`rope_dim` 应与相邻成对语义一致，尾行、padding 和 gather 的
byte offset 均需合法。reinterpret 只改位解释，不能替代数值转换。

### sin 符号与广播

示例在 UB 生成符号 mask：先 `fill(-1)`，再将奇数位置写成 `+1`；乘到 sin 后
与交换后的 x 相乘。该写法避免读取额外 sin_mask，但仍有 `rope_dim/2` 次标量赋值，
因此只可当成可比较的实现，不应称完全向量化。

sin/cos 从 `[1,rope_dim]` broadcast 到 `[row_per_vec,rope_dim]`，随后：

```text
rotated_x = gather(x, swapped_indices)
output = x*cos + rotated_x*(sin*sign_mask)
```

若准备融合乘加，先对应真实数学表达及 dtype 舍入路径。`axpy(dst,src,alpha)` 是
`dst += alpha*src`，不能任意替代两个 tensor 的乘法；依据当前 API 选择等价复合指令。

### 布局、分工与容量审计

当前示例每个 Vector 子核处理 `block_M/2` 行，局部 sin/cos 通过
`row_sin_cos=row_x//head_num` 选取，再共享给这些行。改变 block_M/head_num 时必须
确认子块没有跨越需要不同 sin/cos 的 head 组。`M//block_M` 的 launch 还假设
整除，新增尾行前要修复范围覆盖。

原位写回仅更新末尾 `rope_dim`，应验证未旋转前缀保持不变。多种 RoPE 布局
（half-split 与 interleaved）不可共用同一个交换 mask；可对照
[half/interleaved 示例](../../../../../examples/pos_embedding/RotaryPositionEmbedding/rope_half_interleaved.py)。

预算包含索引转换链、uint32 mask、sin/cos broadcast、x/rotated_x/output、输入输出
dtype 转换的同时活跃存储。优先辨明生命周期再考虑内存规划，不因消除了 GM mask
就忽略新增 UB 和 Vector 工作量。

## 精度

覆盖 position=0/最大值、offset、奇偶 token/head、dim 边界、两种 layout、原位/非原位、bf16/fp16/fp32。reference 的 sin/cos、乘加和拼接使用 fp32；输出只在最终边界转换。

## 性能

比较每 head 重载 sin/cos 与 token-grouped resident 版本，报告 GM bytes、Vector/MTE 时间、UB 占用和 latency。定向精度测试全通过后运行相关完整测试套。

### 分析结果与后续验证

候选应明确减少的是两份辅助 mask 的 host 生成/搬运或 GM 读取；x、sin、cos 与输出
仍需访问，不能将整个 RoPE 总 GM 访问写成“3次降到1次”。当前 wrapper 的 `.to(device)`
属于调用边界，分析端到端成本时单独识别输入是否已经在目标设备。

后续验证覆盖原位输出、未旋转前缀、旋转维度、head 分组、行尾和支持 dtype；比较
片上生成成本与已缓存 mask 的对照。已有
[test_rope.py](../../../../../examples/pos_embedding/test_rope.py) 和
[test_rope_mask.py](../../../../../examples/pos_embedding/test_rope_mask.py)
提供现有验证入口，实施时按目标接口运行相应验证。

## 专题入口

- [RoPE 融合](vector_fused_design.md)
