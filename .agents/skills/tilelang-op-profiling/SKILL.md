---
name: tilelang-op-profiling
description: TileLang 算子 NPU 性能采集与分析，用于采集 TileLang kernel 性能数据、定位性能瓶颈并给出优化建议。当用户在 TileLang 算子开发过程中提到"上板性能"、"算子性能测试"、"硬件性能验证"、"NPU性能采集"、"NPU profiling"等场景时触发。
---

# TileLang 算子上板性能采集与调优

在真实 NPU 上采集 TileLang kernel 性能数据，系统化解读性能指标文件，判定性能是否达标，定位瓶颈类型，并给出可操作的优化建议。

## 后端

上游传入 target 时原样使用；否则优先复用现有入口已经显式指定的 target。两者都未指定时默认使用 AscendC：现有 JIT 入口不传 target 时会使用 `target="auto"`，当前仓库将 `auto` 解析为 AscendC；新建临时 profiling 入口时，在 `@tilelang.jit` 或 `tilelang.compile` / `tilelang.engine.lower` 中显式传入 `target="ascendc"`。AOT 入口的 `lower` 与 `LibraryGenerator` 使用同一 target，预运行和 `msprof` 采集也保持该 target。

## 职责边界

- 本 Skill 负责：运行入口识别、正确性门禁、`msprof op` 采集、kernel/case/launch 映射校验、原始数据归档、性能判定、瓶颈诊断、修改代码、正确性复验和优化前后对比。
- 优先复用算子已有 target、运行入口和参数。
- `do_bench` 可用于建立稳定延迟基线，但最终硬件流水、缓存、冲突等结论以 `msprof op` 数据为准。报告必须标明数据来源和单位。

---

## 适用场景

| 场景 | 说明 |
|------|------|
| 算子开发完成后的性能验收 | 确认算子达到预期性能水平 |
| 性能问题定位 | 通过 CSV 指标精确定位瓶颈 |
| 优化效果验证 | 对比优化前后的 CSV 数据 |
| Agent team 测试阶段 | tester/developer 调用，自动化性能分析 |

---

## 工作流

多 launch 算子、混合 AIC/AIV 计时、AOT/AclGraph 入口或用户要求 pytest 最终验收时，先读 [整链计时与采集](references/kernel-chain-measurement.md)。下面一个 case 一次 kernel 的命令用于单 kernel 入口；多 kernel 链路按实际名称集合与 launch 次数调整。

```
Step 1: 准备并校验 TileLang profiling 入口
    ↓
Step 2: msprof op 采集指标
    ├── 成功 → 继续
    └── 仅 DBI/插桩/导出故障且直调正常 → 普通 msprof fallback
    ↓
Step 3: 归档数据 + 生成统计摘要
    ↓
Step 4: 读摘要 → 对照性能标准判定（需要时读原始 CSV）
    ↓
Step 5: 如不达标，查阅瓶颈优化速查表
    ↓
Step 6: 修改代码 → 回到 Step 2 重新采集（数据自动归档为新一轮）
```

### Step 1: 准备并校验 TileLang profiling 入口

先检查目标算子目录，按以下优先级选择入口：

1. 已有 `perf_*.py`、`--perf`、`--profiler msprof` 或 `bench.sh` 入口；
2. 已有单文件 example 的 `__main__` 入口；
3. AOT 入口或 AclGraph 的公开 replay 入口；
4. 只有上述入口无法隔离目标 kernel 时，才在临时目录生成最小 profiling wrapper。

按以下顺序确定算子输出根目录 `<variant_output_dir>`：

1. 上游已传入 `variant_output_dir` 时原样使用；
2. 未传入时，从目标 kernel 的生产源码路径确定 `<operator_name>`：
   - `examples/operators/<name>/...` 使用 `<name>`；
   - 其他 `examples/.../<name>/<source_file>` 使用生产源码所在的实际算子目录 `<name>`；
   - 源码直接位于 `examples/` 时，从文件名去掉 `example_` 前缀和扩展名；
3. 默认令 `<variant_output_dir>` 为仓库根目录下的
   `examples/operators/<operator_name>`，所有算子的 profiling 输出统一放在
   `examples/operators/`；
4. 临时 profiling wrapper 只作为运行入口，不能用临时目录名推断算子名；
5. 无法唯一确定生产源码或 `<operator_name>` 时停止并询问，不能猜测另一个算子目录。

例如 `examples/gemm/example_gemm_intrinsic.py` 默认解析为
`examples/operators/gemm`，实际输出根目录为 `examples/operators/gemm/perf_tuning/`。
首次采集的 `<variant>` 为 `baseline`；每次修改代码后的复测选择下一个未使用的
`iter_NNN`。

从脚本所在目录运行，或显式设置可靠的仓库根目录/PYTHONPATH。不要把临时 wrapper 写入 `examples/`，避免被批量 CI 自动发现。每个 case 先完成现有正确性测试；精度失败、编译失败或 NPU 执行失败时停止，不能输出“有效性能结论”。

profiling 入口应把 JIT/AOT 构建与正式采集分开，正式采集只发射已确认的目标 kernel。若使用现有入口，记录其完整命令和工作目录；若生成 wrapper，只接受该 case 必需参数，并只调用本地目标 kernel。禁止用整个 Python 进程或 host 计时代替设备 kernel 时间。

采集前先触发 TileLang JIT 并确认目标 kernel 在 NPU 上实际运行：

```bash
cd <entry_workdir>
python <entry_file> <correctness_or_single_launch_args>
```

每轮 baseline 和每个优化方案都必须使用对应版本的最新源码重新编译/预运行。若入口直接导入生产 wrapper，重新确认导入路径、源码 SHA256 和调用链。记录 case、规范化 case 参数或 case 清单 SHA256、target、缓存状态、完整命令、工作目录、源码 SHA256、预热/正式 launch 参数和实际解析出的 kernel 名。

### Step 2: msprof op 采集

```bash
CASE_COUNT=<matched_launch_count>
OP_OUTPUT_DIR=<upstream_variant_output_dir_or_examples/operators/operator_name>
VARIANT=<baseline_or_iter_NNN>
CASE_ID=<case_id>
CASE_OUTPUT="$OP_OUTPUT_DIR/perf_tuning/msprof_raw/$VARIANT/$CASE_ID"

# 可使用单 case 入口，也可复用能严格校验顺序的多 case 入口
env -u ASCEND_RT_VISIBLE_DEVICES \
  msprof op --warm-up=10 --launch-count="$CASE_COUNT" --output="$CASE_OUTPUT" \
  --kernel-name=<expected_kernel> \
  python <entry_file> <entry_args>
```

**关键参数**:

| 参数 | 说明 | 何时使用 |
|------|------|---------|
| `--warm-up=N` | 工具内预热 | 使用 10，避免 DVFS 影响首次运行 |
| `--launch-count=N` | 最多采集 N 次匹配的 kernel launch，不会替应用重复执行 | 单 case 单 kernel 通常为 1；多 case 或整链按实际匹配调用总数 |
| `--output=<dir>` | 指定输出目录 | 避免结果散落 |
| `--kernel-name=<name>` | 只采集已校验的目标 TileLang kernel | 单核入口必须使用；整链按上述参考核对过滤能力、核集合和完整次数 |
| 不传 `--soc-version` | 上板模式由 runtime 自动识别设备 | 所有真实 NPU 采集 |

执行上板采集前优先只运行一次 `npu-smi info`，依据健康状态、显存和进程信息选择候选物理设备，不逐设备启动探测 kernel。若未安装 `npu-smi`，改用一次 `asys info -r=status -d=0` 检查设备状态；若两个命令都不可用，则跳过状态预检查并自行选择物理设备，不得仅因此阻塞采集。msprof 命令必须移除继承的 `ASCEND_RT_VISIBLE_DEVICES`，由 host adapter 直接选择物理设备；禁止用 `ASCEND_RT_VISIBLE_DEVICES=<非零物理 ID>` 驱动 `msprof op`。正确性 pytest 仍可按其既有规则使用可见设备列表。

不得因为系统中存在任意 `msprof` 进程就全局等待；其他物理设备上的采集不是本次任务的阻塞条件。若日志出现 `GetUserDevIdByDeviceId`、`ErrCode=107001` 或 `Failed to convert the driver device ID`，判定为设备 ID 映射错误，使用新输出目录并在移除可见设备映射、显式选择物理设备后重试，不得归因为并发占用。

**输出**：通常在指定目录或当前目录下生成 `OPPROF_{timestamp}_XXX/`。不同工具版本可能输出不同指标集合，必须先探测实际目录和 CSV schema。`OpBasicInfo.csv` 与 `PipeUtilization.csv` 是本 Skill 完整诊断的最低要求；其他指标缺失时如实标记，不得伪造。多 launch 时必须校验目录数、索引连续性和入口调用顺序。

### Step 2.5: 普通 msprof fallback

`msprof op` 是首选。只有同时满足以下条件时才允许 fallback：

1. 同一份最新入口直调编译和运行正常；
2. `msprof op` 失败点明确位于 DBI 初始化、插桩或结果导出，而不是 kernel 编译、设备执行、精度、目标名或 case 映射；
3. 使用全新的输出目录，保持设备、输入、case 顺序和正式采集次数不变。

普通 `msprof` 没有采集前的精确 kernel-name 过滤，因此必须使用支持内部重复执行的现有入口，或在临时目录生成 wrapper，让每个 case 在一个进程中先 warm-up、再正式 launch。下面的参数占位符表示入口自身的重复执行参数，不能假定所有 example 都支持同名 CLI：

```bash
FALLBACK_OUTPUT="$OP_OUTPUT_DIR/perf_tuning/msprof_raw/$VARIANT/$CASE_ID/fallback_<attempt_id>"
env -u ASCEND_RT_VISIBLE_DEVICES msprof --output="$FALLBACK_OUTPUT" \
  --application="python <entry_file> <single_case_args_with_internal_warmup_and_repeats>" \
  --task-time=on --aic-mode=task-based --aic-metrics=PipeUtilization
```

导出后在 `mindstudio_profiler_output/task_time*.csv`（或当前版本等价的 task/op summary）中按**完整且实际解析出的目标 kernel 名**后过滤。要求匹配行数恰好等于：

```text
case_count * (warm_up + launch_count)
```

匹配行按入口的 case 顺序分组，每组前 `warm_up` 行丢弃，后 `launch_count` 行作为该 case 正式结果。行数、顺序、kernel 名或源码/case SHA 任一不一致时停止，禁止猜测映射。fallback 结果必须标明来源为普通 `msprof`；它仍是设备侧 task/kernel 时间，不是 pytest 总耗时或 host 计时。

普通 `msprof` 若能导出 task-based pipe 指标，可用于同口径趋势与流水判定；不能导出某项 `msprof op` 专属 CSV 时如实标为缺失，不能伪造或把不等价字段混算。基线和候选的最终比较优先使用同一种采集方式；若一方不得不 fallback，另一方也用普通 `msprof` 重采。

错误分类按失败位置而非只看末尾错误码。例如设备初始化阶段的 `507033` 应先检查设备健康与映射，不能归为 kernel 失败；`msprof op` 的 `507035` 只有在直调正常且日志证明位于 DBI/插桩阶段时，才能触发上述 fallback。相同故障重试时始终使用新输出目录。

### Step 3: 归档数据 + 生成统计摘要

```bash
# 找到最新 OPPROF 目录
OPPROF_DIR=$(ls -td "$CASE_OUTPUT"/OPPROF_* | head -1)
ARCHIVE_DIR="$OP_OUTPUT_DIR/perf_tuning/profiling/$VARIANT/$CASE_ID"

# 平铺结果或仅含一个 launch 的分层结果可以直接传 OPPROF 根目录
python3 {skill_path}/scripts/perf_summary.py \
  "$OPPROF_DIR" \
  --output-dir "$ARCHIVE_DIR" \
  --kernel-name <expected_kernel> \
  --entry <entry_file> --workdir <entry_workdir> --command "$PROFILE_CMD" \
  --case-id <case_id> --target <entry_target> --device <device_id> \
  --warm-up 10 --launch-count <launch_count> \
  --source-sha256 <source_sha256> --case-sha256 <case_sha256>
```

同一轮的所有 case 使用同一个 `<variant>`；脚本拒绝覆盖已存在的 case 归档目录，禁止
删除或复用旧轮次。`OP_OUTPUT_DIR` 必须沿用 Step 1 已解析的 `<variant_output_dir>`，不能
在 Step 3 临时改到仓库根目录或当前工作目录。

脚本先检查传入目录本身，再向下查找同时包含 `OpBasicInfo.csv` 和
`PipeUtilization.csv` 的目录。只找到一个候选时自动解析；找到多个候选时停止并列出
候选，必须按 launch index 与入口调用顺序的已校验映射，显式传入其中一个 launch
目录，禁止按时间或目录名猜测。

`msprof op` 结果由脚本自动：
1. 创建明确指定的 `--output-dir`，目录已存在时拒绝覆盖；
2. 兼容平铺 OPPROF 与单个分层 launch，复制实际存在的指标 CSV，并将时间戳后缀文件名恢复为标准名称；
3. 校验 `<expected_kernel>` 在 `OpBasicInfo.csv` 中唯一匹配；零匹配或多匹配立即失败，避免分析到其他 kernel
4. 生成 `summary.txt` 统计摘要（仅聚合目标 kernel；AIC/AIV 按有计时的 profiler 行分别统计，**不做判定**）
5. 在同目录保存 `metadata.json`，至少记录传入根目录、实际 CSV 目录、入口、命令、case、规范化 case 参数或 case 清单 SHA256、target、device、kernel、warm-up、launch-count、采集方式、源码 SHA256 和时间单位

**Agent 分析流程**：
1. **先读 `summary.txt`** — 获取全局概览（约 30 行紧凑文本）
2. **结合 `references/csv_fields_reference.md`** — 理解各指标含义和阈值
3. **发现异常时读原始 CSV** — 如核间不均衡，Read `PipeUtilization.csv` 查看逐核数据
4. **按 4.5 章节核算实际搬运数据量与运算数据量** — 从目标 TileLang 源码枚举 GM 读写张量、Cube/Vector 计算量（区分两者），用于算力与带宽核算
5. **结合 `references/optimization_quickref.md` 和目标 TileLang 源码** — 将 profiling 瓶颈映射为 TileLang Tiling、数据布局、流水和计算实现的可操作优化建议
6. **用中文输出分析文件** — 交付必须使用中文

> **重要**：`summary.txt` 只是统计聚合，不包含分析判定。所有瓶颈判定、优化建议由 Agent 结合 Step 4 和 Step 5 的标准自主完成。原始 CSV 文件完整保留在同目录下，随时可以 Read。

`summary.txt` 中的 `timed subcores` 只表示 `aic_time(us)` 或 `aiv_time(us)` 大于 0
的 profiler 行，不等于承担有效算子任务的核。kernel 启动、flag 初始化/清理、同步和
编译器生成的 mix-kernel 框架都可能让零任务子核产生非零时间；计算或 GM 搬运计数非零
也只能作为工作证据，不能单独证明它对应用户源码中的有效 Tile。

分析多核结果时，必须结合目标源码检查 `T.Kernel`、总 Tile/任务数、`my_count` 或
grid-stride 分配、边界 guard、swizzle 以及用户代码与编译器生成同步的边界，并分别报告：

- `Block Dim` 表示的启动 block 数；
- AIC/AIV 的 `timed subcores` 数；
- 按源码任务分配确认的 task-bearing block 数；无法可靠确认时明确写“未确认”，禁止用
  `ai*_time > 0` 或单个计数阈值代替。

摘要中的 ratio 平均值按 timed rows 计算。比较核间均衡性或据此判定瓶颈前，应先排除
没有承担可比业务任务的行；关键路径仍取有计时行的最大耗时。

### Step 4: 性能标准判定

用户给定逐 case 耗时、带宽或 pytest 验收口径时，以该目标判定是否达标。下表与理论耗时用于定位优化空间，不能替代用户验收；表内经验阈值不是单项充分诊断条件。

#### 4.1 总体判定流程

```
读取 OpBasicInfo.csv → 获取 Task Duration 和 Block Dim
    ↓
读取 PipeUtilization.csv → 找到各流水占比最高的单元
    ↓
按 4.5 核算实际搬运数据量 + 实际运算数据量（区分 Cube/Vector）
    ↓
计算理论耗时（搬运量/带宽 或 计算量/算力）
    ↓
比较实际耗时 vs 理论耗时
    ├── 差距 <20% → 接近所用模型的估算下界，另核对用户目标
    ├── 差距 20-50% → 评估优化空间，查阅瓶颈优化表
    └── 差距 >50% → 优先排查模型、计时口径和瓶颈
```

#### 4.2 各指标经验诊断信号

先核对实际数据量、有效 Tile 数、Block Dim、task-bearing block 数和各流水绝对耗时，
再使用下表。小数据量、单 block、启动核多于有效任务或固定开销占比较高时，低 ratio、
低带宽利用率和低 L2 命中率不能单独证明 kernel 实现存在问题；ratio 表示对应流水的
cycle 占比，不等于硬件峰值利用率。

| 指标 | 通常参考 | 观察区间 | 重点排查信号 |
|------|---------|---------|---------|
| **核间负载均衡** | 承担可比任务的核 `ai*_time(us)` 差异 <10% | 差异 10-30% | 差异 >30% |
| **Block Dim** | 与有效 Tile 数和运行设备可用核数匹配 | 远小于有效并行度 | Block Dim = 1 且存在足够工作量 |
| **VEC ratio** | 与算子类型匹配（见 4.3） | VEC ratio >80% | VEC ratio >90% 且无优化空间 |
| **MTE2 ratio** | <30%（计算型算子） | 30-50% | >50%（搬运成为瓶颈） |
| **fixpipe_ratio** | <5% | 5-15% | >15%（检查输出转换、原子写回及对齐，不能直接断言未对齐） |
| **icache_miss_rate** | <5% | 5-15% | >15%（代码量过大） |
| **bank conflict 总占比** | `aiv_vec_total_cflt_ratio` <5% | 5-15% | >15% |
| **L2 Cache 总命中率** | 有复用预期时 >80% | 有复用预期且为 50-80% 时结合 GM/MTE 耗时观察 | 有复用预期、GM/MTE 位于关键路径且 <50% |
| **头开销** | <总耗时的 10% | 10-30% | >30% |
| **DoubleBuffer 效果** | MTE2/VEC 重叠 >30% | 重叠 10-30% | 重叠 <5% |
| **带宽利用率** | 搬运量足够且 GM 位于关键路径时 `bw_usage_rate` >60% | 同一前提下为 30-60% | 同一前提下 <30%，并且绝对 MTE 耗时显著 |

#### 4.3 不同算子类型的预期 ratio 分布

下表是工作量足够时的典型分布，不是验收阈值。小 workload、尾块、动态边界或有效
并行度不足都可能显著改变比例，必须结合绝对耗时和源码工作量解释。

| 算子类型 | 主导流水 | 预期 ratio | 异常信号 |
|---------|---------|-----------|---------|
| **Elementwise**（Add/Mul/Relu） | VEC | vec_ratio 50-80% | MTE2 ratio > VEC ratio |
| **Reduction**（ReduceSum/Max） | VEC | vec_ratio 40-70% | scalar_ratio >20% |
| **Activation**（Softmax/Gelu） | VEC | vec_ratio 60-85% | 大量 cast 指令 |
| **MatMul** | CUBE | cube_ratio 40-70% | vec_ratio > cube_ratio |
| **纯搬运**（Transpose/Concat） | MTE2/MTE3 | mte2+mte3 合计 >50% | VEC ratio >30% |

#### 4.4 理论耗时计算

> **前置**：计算理论耗时前，必须先按 4.5 章节核算该算子**实际搬运数据量**与**实际运算数据量**，并由用户或可信 metadata 提供当前设备的完整峰值规格。没有可信规格时跳过精确理论利用率，只报告实测指标与同口径对比。

**搬运理论耗时**:
```
理论耗时(us) = 实际搬运数据量(Byte) / GM 峰值带宽
```

只有用户或采集产物明确提供与本次设备一致的峰值带宽时才可代入；不能从 910B/910C 产品族名称、`npu-smi` 的泛化名称或其他机器数据推断。

**计算理论耗时**:
```
理论耗时(us) = 实际计算量(FLOP) / 对应单元理论算力
```

理论算力必须同时匹配完整型号、实际核数、频率、数据类型和执行单元。Cube、Vector 与整芯片总算力不得混用；缺少可信规格时不计算精确算力利用率。

#### 4.5 搬运量与计算量核算（算力/带宽分析前提）

> 上板数据只有真实耗时，**搬运量、计算量必须从算子源码自行核算**，据此才能算出有效带宽、实际算力和理论耗时。运算数据量必须区分 Cube 与 Vector。

##### 4.5.1 实际搬运数据量核算（GM↔UB）

从算子源码枚举所有 GM 张量的读写，逐个核算字节数：

| 方向 | 张量 | 字节计算 |
|------|------|---------|
| 读 | 每个输入 | 元素总数 × 单位元素字节数 |
| 写 | 每个输出 | 元素总数 × 单位元素字节数 |

- 元素总数 = 张量全部维度之积（`prod(shape)`）；单位元素字节数由 dtype 决定（如 fp32=4、bf16=2）。
- 累加得「读总量 / 写总量 / 总搬运量」。
- **多输入/多份数据**：若某张量在 GM 中按多份存在（如按维度切分的多片、多份中间结果），搬运量须计入全部份，不能只算一份。
- 与 msprof `Memory.csv` 的 `read_main_memory_datas` / `GM→UB` 等字段交叉验证，确认核算口径。

##### 4.5.2 实际运算数据量核算（区分 Cube / Vector）

从源码判定算子类型与计算量归属：

- **Cube 计算量**：`T.gemm_v0` / `T.mma` 路径 → 每 tile `2 × M × N × K` FLOP；无矩阵乘则为 0。
- **Vector 计算量**：`T.Parallel` 或 `T.tile.*` 内逐元素运算，按「每元素运算次数 × 元素数」核算逻辑 FLOP。

判定依据（与 `ArithmeticUtilization.csv` 交叉验证）：
- `aic_cube_fops > 0` / `aic_mac_ratio > 0` → 有 Cube 计算
- `aic_cube_fops = 0` → 纯 Vector 算子，Cube 计算量 = 0

##### 4.5.3 算力与带宽核算

```
有效带宽(GB/s)   = 总搬运字节 / 实测稳态耗时
带宽利用率       = 有效带宽 / GM 峰值带宽
实际算力(FLOP/s) = 实际计算量 / 实测稳态耗时
```

上式按源码逻辑字节计算的比例仅称“有效带宽比”，不是物理 HBM 利用率；后者需要 profiler 的实际 HBM 流量与精确设备 SKU 峰值。共享缓存、原子读改写和多次消费会使两种字节口径不同，不能混用。

- 只读 / 只写按对应方向核算；读写同时发生时注意 MTE2/MTE3 带宽共享。
- 计算密集型算子用实际算力 vs 理论算力判断是否接近计算上限。
- 小数据量算子优先看搬运理论耗时与头开销占比，不要单看带宽利用率。

**判定逻辑**:
- 实际 MTE2 耗时 ≈ 搬运理论耗时 → MTE2 已达上限，优化方向是流水编排
- 实际 MTE2 耗时 >> 搬运理论耗时 → MTE2 未达上限，检查对齐和搬运粒度
- 实际 VEC 耗时 ≈ 理论计算耗时 → VEC 已达上限，优化空间有限
- Task Duration ≈ 最长流水耗时 → 流水编排良好，其他流水已被掩盖

### Step 5: 瓶颈定位与优化

确认瓶颈类型后，查阅 `references/optimization_quickref.md`，结合目标 TileLang 源码制定具体优化方法。

下面的 ratio 条件只用于快速导航。确认瓶颈时必须同时检查该流水的绝对耗时是否显著，
并由 Memory/Arithmetic 数据或目标源码提供一致证据；不能仅因某个 ratio 最大就直接判定瓶颈。

**快速查找**:

| 瓶颈类型 | 判定条件 | 首选优化 |
|---------|---------|---------|
| **VEC Bound** | `aiv_vec_ratio` 最高 | UB 融合、减少 Cast、融合指令 |
| **MTE2 Bound** | `ai*_mte2_ratio` 最高 | 增大连续搬运粒度、检查对齐、Tile 复用和 L2 命中 |
| **CUBE Bound** | `aic_cube_ratio` 最高 | 调整 Tile shape，提升 L0/L1 数据复用 |
| **SCALAR Bound** | `ai*_scalar_ratio` >30% | 减少动态分支、标量循环和运行时索引计算 |
| **核间不均衡** | 承担可比任务的核耗时差异 >10% | 调整 Tiling 切分策略 |
| **Bank Conflict** | `vec_bank_cflt_ratio` >5% | 调整 TileLang 布局、stride 或 padding |
| **头开销大** | 头开销占比 >30% | 减少核数和动态分支，简化 kernel 启动路径 |
| **DoubleBuffer 未生效** | MTE2/VEC 无重叠 | 调整 pipeline stage，检查搬运与计算重叠 |
| **流水线气泡** | 多单元均 30-50%，无主导 | 调整 pipeline stage、Tile 切分和异步迭代 |

### Step 6: 修改代码并验证优化效果

根据 Step 5 的诊断修改目标 TileLang kernel。每轮只改变一个主要变量，并记录修改内容；
不得通过缩减输入、放宽精度标准、跳过边界 case 或改变计时口径制造性能收益。

修改后先重新运行 Step 1 的正确性门禁。精度、编译或 NPU 执行失败时停止，不得进入性能
对比；正确性通过后回到 Step 2 和 Step 3，使用相同 case、target、设备、kernel 过滤、
warm-up 和 launch-count 重新采集，并归档到下一个 `iter_NNN`。

对比 `baseline/<case_id>/summary.txt` 与 `iter_NNN/<case_id>/summary.txt`，同时检查：

1. Task Duration 是否下降；
2. 关键路径及已确认瓶颈的绝对耗时是否改善；
3. 承担可比任务的核间差异是否改善；
4. 实际搬运量、计算量和 kernel/case 映射是否仍为同一口径；
5. 正确性是否保持通过，且没有新增异常等待、冲突或缓存退化。

若优化无收益或出现回退，如实保留该轮证据并说明结果；下一轮基于已确认的新证据选择一个
新的主要变量，重复 Step 6，不覆盖既有归档。

## 数据目录结构

以下 `<variant_output_dir>` 在上游未传入时固定解析为仓库根目录下的
`examples/operators/<operator_name>`。

### msprof 输出（临时）

不同工具版本可能输出以下任一种结构。

平铺结构：

```
<variant_output_dir>/perf_tuning/msprof_raw/<variant>/<case_id>/
└── OPPROF_{timestamp}_XXX/
    ├── OpBasicInfo.csv
    ├── PipeUtilization.csv
    └── ...                     # 当前工具实际输出的其他指标
```

分层结构：

```
<variant_output_dir>/perf_tuning/msprof_raw/<variant>/<case_id>/
└── OPPROF_{timestamp}_XXX/<resolved_kernel>/
    ├── 0/                      # 第 1 次匹配 launch
    │   ├── OpBasicInfo_<timestamp>.csv
    │   ├── PipeUtilization_<timestamp>.csv
    │   └── ...                 # 当前工具实际输出的其他指标
    ├── 1/                      # 第 2 次匹配 launch
    └── ...
```

### 归档目录（持久）

```
<variant_output_dir>/perf_tuning/profiling/
├── baseline/
│   └── <case_id>/
│       ├── metadata.json
│       ├── OpBasicInfo.csv
│       ├── PipeUtilization.csv
│       ├── *.csv               # 当前工具实际输出的其他指标
│       └── summary.txt
├── iter_001/
│   └── <case_id>/
│       ├── metadata.json
│       ├── *.csv
│       └── summary.txt
└── iter_002/                  # 后续修改轮次继续递增
    └── <case_id>/
        └── ...
```

各 CSV 文件的完整字段说明 → `references/csv_fields_reference.md`

---

## 注意事项

1. **必须 warm-up**: 首次运行受 DVFS 影响，耗时偏高。始终使用 `--warm-up=10`
2. **频率检查**: 读取 `OpBasicInfo.csv` 的 `Current Freq` 和 `Rated Freq`，若 Current < Rated，说明芯片未满频运行
3. **MTE2/MTE3 带宽共享**: 同时读写 GM 时，总带宽被共享，理论耗时应按 `(MTE2搬运量 + MTE3搬运量) / GM带宽` 计算
4. **小数据量场景**: 数据量很小时头开销占比会很高，这不一定是算子问题，而是数据量不足
5. **多核同地址访问**: 多核同时读同一 512B 地址范围会被串行化，导致 MTE2 耗时异常

---

## 参考资源

| 文件 | 内容 | 何时查阅 |
|------|------|---------|
| `references/csv_fields_reference.md` | 支持指标 CSV 的完整字段定义、可选性和阈值 | Step 4 分析时，需要理解具体字段含义 |
| `references/optimization_quickref.md` | profiling 瓶颈到 TileLang Tiling、buffer、流水、布局和计算实现的优化映射 | Step 5 定位瓶颈并在 Step 6 实施修改时 |
| `scripts/perf_summary.py` | schema 探测、统计摘要生成和 CSV 归档 | Step 3 自动归档和生成摘要 |

> 同一核心、同一时间分母下的流水 ratio 之和可作重叠线索，不能仅凭约 100% 断言完全串行，也不能把 130% 当作硬门槛。结合生成代码的事件/版本轮转和可用 timeline 验证；AIC/AIV 的比例不直接相加。理想重叠估算只是候选收益上界，仍受依赖、共享带宽和同步约束。
