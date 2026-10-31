# 自动同步迁移：实现与验证

本页记录实际实现与验收，实施契约和组件边界见[迁移设计](pass-design.md)，
代码出处见[来源清单](upstream-sources.md)，历史路线论证见
[RFC #1971](https://github.com/tile-ai/tilelang-ascend/issues/1971)。

## 当前结论

已落地显式可选的 `upstream` 后端：实际运行移植的主仓依赖分析、
同步规划、覆盖验证和事件分配；不是旧 SyncInsert/VS 的包装。
独立 Z3ProofService 使用主仓证明核心，并修复移植测试发现的状态和位运算问题。
没有整体升级 TVM，也没有引入 AutoSchedule 调度求解器。

当前实现覆盖核内 ordinary/VS、控制流与快照，以及受限的 UB/L1 流水、
typed GEMM、既有 C→V 发布及静态 C-prefix K-loop、Persistent、受限 Persistent+SIP
和 GM→UB tail helper，也包含 A3 PTO GEMM 的 K128/256 窄 profile。
各模块均有明确的输入契约，不能推导为通用流水、任意混合 C/V 或全部复合 helper
已受支持。历史验收结果及其适用版本见[验收摘要](validation/summary.json)。
**不是完整替换完成**：S3/S4 仍是部分验收，S5 未执行。默认仍为 legacy；
旧算法、过渡 backend 配置和原来的同步开关有效默认值均未移除。

## 构建和使用

安装与构建方式见 [Z3 默认启用与 PyPI 依赖](z3-pypi-install.md)。
`USE_ASCEND_SYNC_Z3` 当前默认 ON，pip 构建及运行均依赖 z3-solver，Python 包从
运行环境加载 Z3。直接 CMake 构建需先安装构建依赖，或显式指定包含头文件与原生库的
`Z3_ROOT`；显式根目录保持最高优先级。同步算法仍默认 `legacy`。

对已有 PrimFunc `program` 显式启用：

```python
kernel = tilelang.compile(
    program,
    target="ascendc",
    platform="A3",
    pass_configs={
        "tl.ascend_sync_backend": "upstream",
        "tl.ascend_auto_sync": True,
        "tl.ascend_auto_sync_vs": True,
    },
    compile_flags=["--cce-auto-sync=off"],
)
```

新增公共入口为 `tilelang.transform.AscendInsertSync(target, platform)`，backend 配置
也可使用 `PassConfigKey.TL_ASCEND_SYNC_BACKEND`。phase 中新旧整条同步链互斥。
缺少证明服务、未知副作用或无法证明的协议会诊断，不静默调用旧算法。

ordinary/VS 配置含义保持：VS-only 覆盖 V→V 和 S↔(V/MTE2/MTE3)，不会顺带启用所有
Vector/DMA 同步。一般 Developer 自动同步应同时开启 ordinary 与 VS。

### v18 适配修正

- 同一 helper 类型解析现在将 emitter 的 `int` 归一化为 `int32_t`，保留原 dtype
  白名单和 access_ptr 类型校验；覆盖真实 DSL 的 int32 tail-copy 和 int8 GEMM。
- 普通 GEMM K-loop 将已校验的 Cube 管线屏障与计算叶分开统计，保留原屏障和顺序，
  继续要求两次完整 A/B 刷新、一个 GEMM 及正确的 clear 条件。
- AscendC transpose 根据 helper 实际 dtype/shape 选择 S 或 V 管线；标量 fallback
  仍在 AIV 执行，PTO 的 TTRANS 保持 V。快照和流水分析只解析操作效果，同步消费者
  另传逻辑形状，不能以默认 V 管线替代缺失的形状信息。
- 共享同步语义变化也影响 legacy，因此实际 Ascend target 的三种编译缓存 key 和
  schema2 manifest 均绑定原生同步 ABI。旧 Ascend schema1 或缺失 manifest 触发
  cache miss，显式数据库恢复会在加载动态库前拒绝；CUDA/HIP/普通 LLVM 不查询该 ABI。

针对性回归见 `test_ascend_upstream_sync_transpose.py`、现有 GEMM K-loop、tail-copy
和缓存套件，以及语言 transpose 测试中的 `test_transpose_sync_matches_helper_pipeline`。
这些修正不扩大下面尚未审计的流水、跨核或动态手动事件契约。

### v19 空尾块与间接地址

- A2/A3 AscendC 的 typed GM→UB tail helper 接受零行或零列。空块仍填充整个 UB
  tile，并在隐藏的 V→MTE2 完成边界之后返回，不发出 GM DMA；适配层将其 GM 读范围
  规范为零长度，仍无条件验证整个 UB allocation、对齐及隐藏事件预留。
- 非空路径仍须证明完整的 strided GM 访问范围，负 mask、超界 mask 和非空越界访问
  继续拒绝。PTO 的独立 helper 尚未审计空块，仍要求正 mask。
- 已配置硬件操作的 `access_ptr` offset/extent 中，显式标量 BufferLoad 在原语句
  边界捕获为 Let。对索引内存的读取参与依赖分析，导出保留先定义标量寄存器、再由 DMA
  使用的顺序；后续覆盖索引不会改变已捕获的地址。不将快照等同于未来的内存重读。
  opaque/read-state intrinsic、ring provenance 和 tail/composite 边界证明保留原有约束。
- shared helper 及同步语义变化由 native ABI v19 隔离旧缓存。针对性回归位于现有
  tail-copy、snapshot 与 device 套件。

## 当前支持与拒绝边界

已验证范围：

- 非流水的单 C 或单 V 图，基本 copy/fill、读写冲突及普通 For。
- 可变 If 条件、已有 Let 和硬件标量参数中的 BufferLoad 在原求值位置形成真实 Let
  快照；S definition task 参与读写依赖，值不会被后续内存覆盖重新解释。
- 非 ring 的 If 使用零或一次的控制节点分析；If/else 先真实捕获一次条件，再分别保留
  两个互斥分支及独立插入锚点。条件不成立时仍有正确的外部完成链，不借可选 producer
  消除直接依赖。
- 不同 Var 共用最终地址的 RAW/WAR/WAW；DMA footprint 使用保守 allocation 范围。
- 独立逻辑 storage 与物理冲突身份、原始语句顺序和词法作用域保持。
- A2/A3 V barrier、A5 V_V event 的主机策略矩阵；S 与 Vector/DMA 的主机配置矩阵。
- 普通顺向、平衡且 literal ID 的已有手动事件预留；容量不足明确失败。
- 默认 AutoCV 下的小尺寸 abs、Parallel add 和原地 ReLU，内存规划开关两种配置。
- scalar-index copy 的 S→DMA 同步，ordinary 和 VS-only 两种配置。
- 同一 PrimFunc 对象重复调用幂等；maps、配置或其他同步输入改变时拒绝旧 proof
  marker。当前 structural hash 不承诺跨序列化重建的 free Var 身份稳定；不能把该
  marker 当作可跨进程携带的证明证书。
- 单 V 的已物化 UB ring：injector 发布真实 B、包含 padding 的
  slot stride 和每个访问的时钟；StorageRewrite 传播 root/base bits，MemoryPlanning
  不把元数据壳误作生命周期 scope。当前审计访问限完整块 GM→UB、abs 与 B1 GM store。
  实际 B2/B3 静态和动态 SIP 已有独立 token/完成关系检查及设备验证；动态输入限单个
  顶层、injector 认证的 owner/segment/stage family，短 trip、空 steady 和长循环分别
  保持完整 seed/drain 协议。使用真实 manual clock 和版本表，不构造自动 owner plan。
- A3 AscendC/`auto` 下受限 typed `gemm_v0`：审计 MTE1 acquire、Cube completion、
  实际 A/B/C footprint 和隐藏 L0 事件；支持已审核的静态 full-tile 外部 K-loop。
  另有 A3 AscendC/`auto` 的受限静态 Cube SIP：真实 lead=1/min=0、K trip≥3，独立
  L1 A/B 双 ring B2、对齐 fp16→fp32、无 transpose 或内部 K splitting，只有一个
  最终 FIX。使用保守的 strong-0 完成 fence 保持 helper 隐藏事件语义，可能减少重叠，
  不承诺提速。这不包含 `gemm_v1` 或任意 scratch/流水组合。
- v8 增加 A3 PTO 的 flat/B1 `gemm_v0`：仅 M=N16、K128/256、kL0=128、fp16→fp32、
  clear=True、无 transpose、n_actual=N。它使用一个隐藏 L0A/L0B slot 和 ID1，
  已分别审计 acquire/completion、内部反馈及事件预留；修正原 helper 重复置位问题。
  两种 K 各重复三次设备调用通过，CCE 自动同步关闭。For/ring/CV 组合仍拒绝；
  来源、原协议反例与共享 helper/cache 边界见[来源记录](upstream-sources.md)。
- flat/B1/C1V2 的既有单次 mode-2 FIX C→V 发布：保留原 Set/Wait，验证真实跨资源
  RAW/WAR/WAW 后，C/V 分别运行核内引擎。支持 Expert `blockIdx.y=2` 与具有
  `cv_1_2` 契约的 Developer `threadIdx.x=2`，不生成新的跨核协议。
- v5 增加同一发布前的单个静态正 trip C-only K-loop：保留完整 Control 树和原计算
  锚点，发布必须在整个循环及其最终 FIX 之后，V epilogue 保持 flat。跨核验证使用
  真实 leaf 与控制区间；全 GM 参数 shadow 仅用于释放/获得证明，防止循环变量逃出
  词法域。无整函数 noalias 的输出复用 A/B 场景也已设备验收。
- 上述发布可使用显式 `tir.noalias`；Cython JIT 也可通过版本化 fresh workspace 契约
  运行无整函数 noalias 的首片：单工作组、V GM 只读可信 workspace 集合、单一外部 GM
  输出且真实 DMA/vid footprint 不交叠；WAR shadow 保证 C 输入读完后才覆盖输出。
  原始 lower/AOT/其他 adapter 不从裸 `auto_gm_indices` 推断 fresh。
- Cython 契约进入 cache key；manifest 核对原函数源码 hash、lowered 参数数目及真实
  自动分配索引。原函数 3 参数、lowered artifact 4 参数且 workspace 索引为 `[3]`
  是合法情况；缺失或不匹配的表不变成 `[]`。旧独立 AutoTunerCache 若未在 key 前准备
  契约，保守跳过该 fresh-sensitive artifact 的磁盘发布，保留当前内存 adapter。
  初始 engine cache ABI 加固已在 v4 原生构建验收：当时仅 upstream key
  查询 `tl.ascend_sync.EngineCacheABI`，将权威 native 版本纳入 KernelCache、Tuner 和
  CompileArgs key；upstream manifest 升为 schema2 并绑定同一 ABI，直接 database
  重建也核对。v18 将该保护扩至实际 Ascend target 的 legacy key 和 schema2 manifest；
  仅非 Ascend target 不查询该 FFI。省略 target 的底层 helper 保留原 schema1 兼容入口，
  实际编译缓存和 database 路径必须传递 target。
  每次依赖分析、importer、policy 或物化语义改变都必须 bump native engine 版本；
  包版本或 git HEAD 不覆盖同 HEAD 的工作树重建，原函数源码 hash 也不能替代该版本。
  helper 协议变更同样须 bump。另已审计活动 Cython/ctypes 链：新 adapter 每次实际
  运行 Bisheng 并产生唯一临时库，没有按 kernel source 另读一个低层缓存；通用
  `.cycache` 扩展不 include PTO helper。因此 ABI miss 后会重新编译修正后的头文件。
- 单 C/单 V 的 Persistent：规范领先 break 收紧成等价的每 worker trip count；识别
  Simplify 的 `cid//d` 及经原始范围证明与 LE 等价的 EQ，保留计算、顺序及数组词法
  作用域。部分首 wave 与动态 predicated loop 沿用原控制模型，支持内部普通嵌套循环。
- v6 增加 A3 AscendC/`auto` 的受限 Persistent+SIP：FP32、完整 64 元素块、单个
  规范静态多 wave Persistent、显式 noalias，内部 UB B2/lead2 且静态 trip≥3，
  ordinary 与 VS 同时启用。内部和外部均运行真实依赖分析、验证及分配；每个 wave
  六对事件封口，额外 12 次 flag 调用，外层 wave 串行而内层 SIP 保留。原计算、分配、
  true B 与 clock 均保留；没有清空版本表或使用虚构 epoch。这不代表通用多 owner。
- A2/A3 AscendC/`auto` 下已审计的 GM→UB tail-copy envelope：保持一次原 helper 调用，
  以真实内部握手验证统一 MTE2 acquire/completion，覆盖完整 Duplicate footprint，
  预留隐藏 event-0 并拒绝显式碰撞；动态 mask 不重复求值。设备验收在 A3。

尚不支持，明确拒绝：

- 缺失或不一致的流水 provenance、非 UB ring、任意用户分支 ring、多 owner。
  ring 与其他逻辑 allocation 的别名、缺失 clock、跨槽 helper footprint 均明确拒绝。
- 除上述已认证 C-prefix 外的混合 C/V 循环、分支或多次发布，新跨核协议和
  PerCoreTask/新 scheduled-TIR 输入。
- 未审计的 GEMM 类型/形状/footprint、`gemm_v1`、动态或其他形态的 Cube ring/软件流水；不能仅用
  单 PIPE_M 元数据推导复合完成语义。通用 GEMM 验收仍未完成。
- 直接读取可变内存的 For bounds、未知状态读取和 opaque 表达式；真实快照仅覆盖显式
  BufferLoad，纯符号动态 extent 不因此被禁止。
- 非规范 break、While，以及上述 v6 契约以外的 Persistent 与 SIP 组合；没有把普通
  嵌套 For 的结果当作嵌套 ring/owner 生命周期的证明。
- 反向 manual wait、跨控制流借用 token 或未平衡事件；仅预留 ID 不能证明无死锁。
- 未建模的操作和动态 manual ID；不能靠扩大 barrier 或取模掩盖未知协议。
- 未证明合法范围、未支持类型或 footprint 的 tail copy、PTO 零 mask，以及 tail 与
  ring/mixed C/V 的组合。full-copy 既有路径与 tail 首片的 target 支持范围分别处理。

原 Z3 翻译保留数学整数索引模型，不宣称完整机器整数回绕语义。
非保值 cast、非法或无法证明合法范围的 shift 显式诊断；超时/unknown 不作为消除依赖的证据。

## 验证记录与复验

`validation/` 仅保留三份文件：

- [测试清单](validation/test-suites.json)：主机、设备和独立 legacy VS suite 的精确 pytest 选择集。
- [性能原始数据](validation/performance.json)：最新保留的 v8 四用例测量，包含环境、
  样本、同步调用数量和内存信息。
- [验收摘要](validation/summary.json)：汇总最近记录的 Z3 默认 ON 检查点和独立 legacy VS
  结果，保留环境、引擎 ABI、测试范围、安装检查及已知限制。

摘要中的完整验收对应 v8；清理基线 `aa219a3f` 已为 v10，新增了 unsigned 回绕和
截断除法修复，尚无与该版本匹配的完整验收记录。摘要另列闲置代码清理的针对性检查：
打包测试、18 个同步翻译单元的 Linux 目标对象编译，以及测试迁移和活跃路径不变性检查。
这些检查不包含完整原生库链接、NPU 执行或安装验收，不能将历史结果视为当前源码通过。

历史分项记录和重复审计文件已移除，原内容可从摘要注明的 Git revision 查阅。
原始日志、验证脚本、完整 source manifest 和产物只记录了本机临时路径，未随仓库保存；
受测快照还保留了两份未提交的 TVM 修改，未保存其补丁。因此这些记录供历史追溯，
不能单凭提交 SHA 或文件 hash 重建精确受测输入。后续验收应保存可访问的日志与产物，
并绑定确切源码版本及必要补丁。

### 按测试清单复验

在已构建的 CANN/Bisheng 环境中，从仓库根目录运行以下命令。将 `host` 替换为
`device` 或 `legacy_vs` 可分别执行设备组合或完整旧 VS 编译回归：

```bash
python - host <<'PY'
import json
import subprocess
import sys
from pathlib import Path

suites = json.loads(
    Path("docs/ascend/sync_migration/validation/test-suites.json").read_text()
)
subprocess.run(
    [sys.executable, "-m", "pytest", "-q", *suites[sys.argv[1]], "-p", "no:cacheprovider"],
    check=True,
)
PY
```

`host` 包含原主机范围及 PyPI Z3 运行时测试；codegen 矩阵和真实缓存重建需要
Bisheng/CANN，但不启动 NPU。`device` 需要 A3 NPU，迁移用例显式关闭 Bisheng 自动补同步。
`legacy_vs` 不启动 NPU，不使用 marker 过滤，也不包含在主机计数内。
测试文件合并并增加 transpose 管线回归后，`host` 清单为 31 个文件；helper admission 用例并入 tail-copy 套件，
有效的旧打包校验并入 PyPI Z3 套件。摘要保留原验收计数，不将文件组织调整当作重新验收。

设备选择集包含 12 个原设备测试文件和一个 transpose 管线回归函数选择器，后者覆盖
legacy/upstream 两后端的实际 helper 路径。

ON 验证应先确认 `tl.ascend_sync.Z3ProofServiceAvailable` 返回真，避免将大量 skip
当作迁移覆盖通过。OFF 构建复用同一清单，验证不可用诊断及能力门禁；设备组合中的独立
wave-seal 原语和 legacy transpose 不依赖证明服务，其余 upstream 迁移用例应跳过。
主机、设备和版本矩阵的计数不能叠加。

真实缓存重建可单独执行：

```bash
python -m pytest -q \
  testing/python/language/test_ascend_upstream_sync_runtime_contract_cache_compile.py
```

### 性能与安装验收边界

保留的性能数据比较 vector abs、静态 B3 pipeline、K-loop GEMM 和 Cube B2 SIP。
计时包含 host launch gaps，不能推导纯设备执行时间收益，也未覆盖 Persistent+SIP
每 wave 额外封口的开销。重新采样可运行：

```bash
python testing/python/language/benchmark_ascend_upstream_sync.py \
  --output /tmp/ascend-sync-performance.json
```

安装验收须分别覆盖 wheel、PEP660 editable 和直接 CMake 安装，并核对原生库来源、
Z3 正反查询、引擎 ABI 及许可证。历史 compiler 安装通过不代表 standalone runtime-only
加载通过；后者曾因 `IRBuilderFrameNode` RTTI 缺失失败。安装的静态库没有新增 CMake
导出包，外部使用归档时仍需显式链接 Z3。更多安装检查见
[Z3 安装文档](z3-pypi-install.md#验证方式)。

## 下一阶段门槛

1. 按旧公开前端实际生成的输入逐项补齐剩余 Cube 流水、跨资源循环和已启用 target/codegen
   组合；受限模块通过不等于它们任意组合也通过，不能删除门禁充数。
2. 扩大受限 Persistent+SIP 之外的真实嵌套 owner/epoch 生命周期；automatic counter/
   multi-owner 与 manual 模式分开建模，不把 version 表清空来规避协议。
3. 扩大 ring helper 白名单前，逐个验证真实 DMA/Vector footprint、隐藏事件池与完成语义。
4. 后续每次扩大适配范围后，重新执行完整 ON/OFF、设备、真实缓存重建和发布产物检查，
   更新测试清单与验收摘要，绑定实际受测版本；不用历史或分项计数替代当前验收。
5. 按 S4 验收声明支持的 target/codegen 与性能回归后，才进入 S5：构建依赖与默认切换、
   两个旧 API alias 的统一语义/幂等测试、删除旧算法及过渡分派。默认目前不变。
   保持 v18 已覆盖实际 Ascend target 的 native ABI 保护；默认行为改变也必须更新 ABI，
   避免省略配置的新默认路径复用旧缓存，不能影响无关 target 的 key。
