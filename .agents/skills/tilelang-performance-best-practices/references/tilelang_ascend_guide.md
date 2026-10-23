# TileLang A2/A3 实现指南

## 目录

- [事实来源](#事实来源)
- [执行域](#执行域)
- [内存与流水](#内存与流水)
- [精度规则](#精度规则)
- [性能规则](#性能规则)
- [算子模式](#算子模式)
- [验证矩阵](#验证矩阵)

## 事实来源

按以下优先级确认 API：

1. 当前目标算子与 `examples/**/*.py`：实现结构与 API 参考，性能状态按具体示例和当前验证结果判断。
2. 用 `python -c 'import tilelang; print(tilelang.__file__)'` 定位实际导入源码，不按兄弟目录名推断版本。
3. 实际 TileLang 源码的 `examples/` 与 `testing/`：可运行示例、API 边界和 lowering 回归。
4. 实际 TileLang 源码的 `tilelang/language/` 与 当前后端 lowering：API 定义和后端约束。
5. 当前后端 codegen/lowering：确认目标工程已选后端下确实支持。

## 执行域

```python
import tilelang
from tilelang import language as T

@tilelang.jit(out_idx=[-1])
def make_kernel(n: int, num_cores: int):
    # num_cores 由已确认的目标硬件资源传入
    @T.prim_func
    def kernel(x: T.Tensor((n,), "float32"), y: T.Tensor((n,), "float32")):
        with T.Kernel(num_cores, is_npu=True) as (core_id, vid):
            ...
    return kernel
```

- Developer / 混合模式使用 `T.Parallel`、符号表达式和同类实现中已验证的 `T.tile.*`，按实际依赖保留自动同步及内存规划。
- Expert 模式使用显式 `T.Scope("C"/"V")`、`T.alloc_L1/ub/L0A/L0B/L0C` 和当前 LOCK DSL 表达生产消费所有权；历史 low-level flag 只作核验参考。
- 同类仓库实现优先于抽象偏好：例如 Norm/Reduction 先复用当前 `examples/normalization/`、`examples/softmax/` 的归约结构并完成当前 case 验证，再把其他 Vector 写法作为需要独立验证的优化候选。
- L1/L0A/L0B/L0C 与 `T.gemm_v0` / `T.mma`：Cube 矩阵计算。
- `cid/vid`、`threads` 与 workspace 的划分按 [公共语言参考](../../../../docs/language_ref/primitives.md) 和 [编程模式指南](../../tilelang-custom-skill/tilelang-programming-model-guide/SKILL.md) 核对。执行侧变更必须通过精度与性能实测。

不能将其他后端的 Kernel / threads / 执行域规则直接套入 A2/A3；`T.tile.*`、GEMM 或手写 scope 的存在也不能单独决定编程模式或要求关闭全部 pass。

## 内存与流水

- UB 使用当前模式的 `T.alloc_shared` 或显式 `T.alloc_ub`；GM/UB/L1/L0 搬运使用仓库对应路径中已验证的 `T.copy`。
- 动态尾块只搬运 `valid` 范围，但 UB 行宽按 DMA/SIMD 可能触达的完整对齐 footprint 分配。
- 权重、scale、小表等复用数据尽量在 Persistent 循环外搬入 UB并使用单版本。
- tile 输入输出按 stage 多版本：

```python
# 显式槽位的分配示意；完整生产/消费、同步和首尾调度见多版本流水专题。
x_ub = T.alloc_ub((num_stages, block), "float32")
y_ub = T.alloc_ub((num_stages, block), "float32")
```

当前 `T.Persistent(domain, wave_size, index, group_size)` 仅表达任务映射，不接收 `num_stages`；流水使用当前 `T.Pipelined` 或已验证的手动多版本。输入、输出和临时 buffer 按跨迭代活跃区间决定版本，动态尾块不能扩大 GM 有效范围。完整首轮/稳态/尾轮和自动/手动准入要求见 [多版本流水](elementwise/double_buffer_design.md)。

UB 预算公式：`Σ(buffer_elems × dtype_bytes × versions) + padding + resident + safety_margin`。按 [硬件资源与有效容量](common/hardware_resource_discovery.md) 区分平台用户可用量、当前编译器预算与隐式临时量，不重复计入已扣除的预留。

## 精度规则

- fp16/bf16 输入转 fp32 后再执行 reduction、variance、rsqrt、softmax max/sum 和长链乘加。
- GEMM 默认使用 fp32 L0C accumulator，完成后再转换输出 dtype。
- softmax 使用稳定形式 `exp(x - max(x)) / sum(exp(x - max(x)))`。
- RMSNorm/LayerNorm 的 `eps` 加在 fp32 统计量上。
- 尾块不得读取未初始化 UB lane；完整 Vector 访问需要清零、填充或正确 mask。
- 容差沿用同类仓库测试并结合数值尺度；不得为掩盖 dtype、尾块或累加错误而扩大容差。

## 性能规则

1. 先减少 GM traffic：常驻复用数据、融合中间结果、避免重复回写。
2. 再建立 MTE/Vector/Cube overlap：测试当前流水的 2/3 stages，并以 profiling 选择。
3. SIMD 内层优先静态向量数，避免不必要的动态循环、标量分支和过度 unroll。
4. 核数不超过 `min(硬件核数, 独立任务数)`。每核存在常量搬入、索引构造等固定段时，同时 A/B 较少核数 + Persistent 多 task/core，按固定段复制成本、wave、负载均衡与实测选择。
5. UB、寄存器、指令体积共同约束 tile；性能下降时检查 spill、代码膨胀和占用。
6. 性能比较保持输入分布、shape、dtype、输出语义、warmup/repeat 和并发设置一致。

## 算子模式

### Elementwise

GM→UB，当前 `T.Parallel` / `T.tile.*` 计算，UB→GM。标量权重按当前 fill/broadcast API 复用。bf16/fp16 超越函数先按当前 API 转换到 fp32，计算后再转换输出。

### Reduction / Norm

- 尽量让单个输出只由一个核负责，避免原子累加。
- 跨 tile 状态使用单版本 resident UB；每 tile 使用 fp32 vector accumulator。
- 大 reduction 可评估 split-K partials + final reduction，但必须计入额外 GM traffic。
- 若公开接口包含 backward，则单独验证其 fp32 累加和尾块；forward-only primitive 明确记录 backward 不在范围。

### Transpose

- shape/dtype 支持时使用当前 Vector gather/scatter；UB 两维都按向量访问 footprint padding。
- 索引向量无法表达的 dtype 使用仓库已验证的 UB-local 或合法 DMA 路径；通用 int64 不经有损浮点中转。
- `T.assume` 只表达调用方真实保证的约束，测试覆盖每个允许的余数类。

### GEMM

```python
# 当前 A2/A3 示例的结构示意；完整调用从 examples/gemm/example_gemm.py 派生。
with T.Kernel(num_cube_cores, is_npu=True) as (block_id, _):
    a_l1 = T.alloc_L1((tile_m, tile_k), dtype)
    b_l1 = T.alloc_L1((tile_k, tile_n), dtype)
    c_l0 = T.alloc_L0C((tile_m, tile_n), "float32")
    with T.Scope("C"):
        for kt in T.serial(T.ceildiv(k, tile_k)):
            T.copy(a_gm_slice, a_l1)
            T.copy(b_gm_slice, b_l1)
            T.barrier_all()
            T.gemm_v0(a_l1, b_l1, c_l0, init=(kt == 0))
            T.barrier_all()
        T.copy(c_l0, c_gm_slice)
```

这是当前 A2/A3 样例的结构示意，不是 `T.gemm_v0` / `T.mma` 的唯一合法布局。流水是单独验证的优化候选，不能将示意中的保守 barrier 作为性能模板。切片、padding 和输出搬运必须从同类已运行样例适配；嵌套 K 子循环时只在首个有效子 tile（例如 `kt == 0 and sk == 0`）清零。不要假设任意 M/N/K 尾块会自动正确处理。

## 验证矩阵

| 维度 | 必测点 |
|---|---|
| shape | 最小、常见、最大、公开接口允许余数类、动态维；接口承诺通用尾块时再加 tile±1 |
| dtype | 每个支持的输入/输出 dtype；低精度输入与 fp32 reference |
| 数值 | 0、正负、极值、小量、重复值；需要时覆盖 NaN/Inf |
| 路径 | 接口要求的 forward/backward、尾块、resident/fallback、单核/多核 |
| 性能 | 定向精度测试通过后，对代表 shape 比较 latency 并记录回归阈值 |

```bash
pytest <test_file> -x
pytest <test_file>
```

仅在确认 worker 设备绑定和隔离机制后增加 `-n`，并发不超过可用设备数，否则串行；OOM 时降低并发，不减少用例。benchmark 保持设备独占。

失败时分别记录：收集数、通过数、首个失败 case、异常类型和根因。`NotImplementedError` 表示覆盖到了尚未实现的仓库路径，不是容差问题；实现该路径后必须从定向精度测试重新验证。

## A2/A3 接口与参考补充

API 入口见 [API 最佳实践](../../tilelang-custom-skill/tilelang-api-best-practices/SKILL.md)，模式和 pass 配置见 [编程模式指南](../../tilelang-custom-skill/tilelang-programming-model-guide/SKILL.md)。接口分别定位到 [ascend.py](../../../../tilelang/language/ascend.py)、[ascend_tile.py](../../../../tilelang/language/ascend_tile.py) 与 [Persistent](../../../../tilelang/language/persistent.py)。GEMM 结构见 [基础 GEMM](../../../../examples/gemm/example_gemm.py)，所有代码块仅解释结构，不代表本轮编译或性能验证。

- 自动划分 C/V 不等于数据依赖已同步；核内、跨核和 workspace 的生产消费分别验证。手动同步遵循当前 LOCK DSL，不能未经核验复制历史 raw flag。
- `get_kernel_source()` 用于检查实际执行侧、地址、同步和指令，是否重叠与更快仍需上板证据。
- 不把 reshape 当作改变物理 stride 的操作；Host 预处理、输出后处理和额外 launch 都计入完整路径。测试 golden 的 CPU 工作不等于产品路径可消除开销。
- 融合前先验证代数等价，再核对舍入、dtype 和饱和模式。整数转浮点前证明值域，NaN/tie/signed zero 与索引语义按原接口验证。
- 改布局、dtype、batch 或 stage 时先验证专题定义的 kernel 合同，再接通 planner；多阶段路径包含所有中间空间与有效范围。
