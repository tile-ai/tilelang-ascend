# A2/A3 Ascend 调度能力迁移计划与进展

将主仓 Ascend 的任务依赖分析、自动同步、AutoSchedule 和循环调度实践，
逐步迁入 tilelang-ascend 的 A2/A3（910B/910C）执行路径。每个独立大步骤
完成后更新本文件、提交并推送，等待用户审核与明确的继续指令。

当前有效验收口径：用户已明确本环境为 A3，后续跳过 A2 专属测试，A2 不再作为各步
验收门槛。保留既有 A2 兼容代码和历史测试事实；共享源码数据不表示 A2 设备已验收。

## 1. 仓库与基线

| 项目 | 固定信息 |
| --- | --- |
| 推送 remote | erhsh |
| remote 地址 | https://github.com/erhsh/tilelang-ascend.git |
| 工作分支 | codex/ascend-a2a3-scheduler-migration |
| tilelang-ascend 固定源码基线 | 43b0b69720f479894a13c02746599eccee5c5769（来源：origin/ascendc_pto） |
| 主仓参考 | tile-ai/tilelang，1c0f402e166a7d0a055c991d2f82d2175a7917b7 |
| 技术参考文件 | [pass-design.md](pass-design.md) |

该完整提交号是迁移与回归比较的固定依据，不随分支 HEAD、提交合并或文档修订而改变。
变更源码基线须单独说明影响并经用户确认。

实施基线的自动 C/V 分域默认开启，跨核同步、核内同步和自动内存规划策略默认关闭。
源码已有资源归属验证、
Vector 指令选择、mask 合法化、Persistent 参数/partial-wave 保护及 L1 多版本修复。
这些设施的具体合同与验证范围以 S1 审计和后续记录为准，优先复用而非重复迁入。

## 2. 提交与人工审核规则

1. 一次只开展用户明确允许的一个大步骤。技术上可以并行的任务，也不因此获得
   跨步骤自动执行的授权。
2. 每步包含该步实现、必要回归、文档和本计划进展更新。一个大步骤可以包含
   多个紧密相关的文件或局部修改，不按每个小编辑单独打断用户。
   每步末尾还须设计验证脚本；S1–S8 实施交付时，实际提交该步可独立运行的
   2–3 个验证入口，覆盖正常用例、边界或负例。涉及设备执行的步骤须提供
   A3 NPU 验证入口。脚本可封装已有 pytest 用例，不重复实现同一套断言。
3. 完成后检查提交范围，单独 commit 并 push 到本分支；同时报告提交号、
   变更摘要、真实验证结果及剩余问题。每步至少有一个可独立审核的提交。
4. 推送后停止。没有反馈、经过一段时间或仅收到一般性确认，都不代表可以开始
   下一步；收到用户明确要求继续后，才启动下一步。
5. 用户指出问题时，先在当前步骤修复、重跑受影响检查、更新计划并推送修复提交，
   再等待复核。没有必要时不重复无关测试。
6. 缺少硬件/native 验证时，可以推送供审阅的候选实现，但必须标成“待环境验证”，
   不标为该步验收完成。当前只要求 A3 结果，A2 历史结果保留；测试范围调整不删除
   A2 兼容代码，也不构成 A2 新能力的验证结论。
7. 不自动开启新策略为默认，不自动合并到默认分支。未经用户明确授权，不重写
   已审核提交或强制推送。基线升级、默认行为改变或新增目标平台需等待用户指令。

计划只保留当前范围、进展、验收证据及有效决策；提交详情查看 Git 历史。

## 3. 当前进展

| 步骤 | 独立交付 | 实现状态 | 验证状态 | 人工审核 |
| --- | --- | --- | --- | --- |
| S0 | 迁移计划与技术参考入仓 | 文档已完成 | 文档静态检查通过 | 用户已确认，并明确授权 S1 |
| S1 | fork 基线审计与前置合同 | 审计、局部修复及三个入口已交付 | 主机通过；历史 A2/A3 native 各 105 项通过；A3 14 项各 100 轮通过 | 用户已明确授权 S2；A2 不再阻塞验收 |
| S2 | 能力、任务图与 solver 合同（M0） | 显式分析、能力模型、solver/verifier 和三个入口已交付 | solver 105、capability 42、native 171 及 runner 31 项通过；无新增设备执行路径 | 用户已明确授权 S3 |
| S3 | Parallel/Pipelined/Persistent 接口与安全性 | 循环接口、域保护、受限 Task/Stage 和三个入口已交付 | native 117 + 29、兼容回归 170、runner 44 项通过；A3 9 项、1000 次 launch/golden 通过 | 用户已明确授权 S4 |
| S4 | 单核约束规划与同步协议验证（M1） | opt-in constraint、最终协议验证和三个入口已交付 | 原生 45 + 50、兼容 170 + 117 + 29、runner 77 项通过；A3 8 项、800 次 launch/golden 通过 | 用户已明确授权 S5 |
| S5 | 跨迭代流水与多缓冲（M2） | A3有效成本、实际版本/配置证明及guarded波流水已交付 | 校准32、原生234、兼容411项通过；A3 26用例/2600次launch与golden、4组性能已记录 | 用户已明确授权S6 |
| S6 | owner/epoch、别名与生命周期（M3） | canonical storage与source/final epoch bridge已交付 | 正式原生61+49、全部兼容/CLI通过；A3 20用例、2000次launch/golden通过 | 用户已明确授权S7 |
| S7 | A2/A3 新同步生成器（M4） | A3核内生成器与固定C/V workspace握手已交付 | 正式native70+60、兼容与CLI214通过；A3 20用例、3200次launch/golden通过 | 用户已明确授权S8 |
| S8 | 完整自动调度与混合 C/V 接管（M5） | 新后端、版本、同步与三个入口已完成 | 原生74、CLI283和旧阶段兼容通过；A3正确性14用例/1400次、性能6用例/480次通过，回归已记录 | 已交付；用户已选择整体集成验收 |
| 整体验收 | S1–S8 最终构建跨阶段验证 | 已完成，生产源码不变 | 新增主机18/105/42；A3数值97worker/11000golden及8性能worker通过，同源S8证据已复核 | 待用户审核 |
| 正式代码整理 | 内部协议、标识符与诊断按功能命名 | 已完成，保留调度、同步及 schema 行为 | 静态与主机检查通过；native/NPU 待重建验证 | 待用户审核 |
| 关键源码修复 | Fill 操作数及 Block 别名访问元数据 | 两处修复已实现，临时 native 回归已准备 | 格式、静态及缓存检查通过；native/NPU 待环境验证 | 待用户审核 |
| 正式 Python 格式 | 5 个正式 Python 文件与 CI 格式对齐 | 已完成，无逻辑修改 | Ruff 0.17.3 格式及 lint、AST 一致性、diff 检查通过 | 待用户审核 |
| 完整 PR 格式门禁 | CI 覆盖的全部 Python 文件 | 剩余 25 个脚本及 53 个测试文件已机械格式化 | 全部 111 个 Python 文件通过 Ruff 格式及 lint；语义 AST 和 diff 检查通过 | 待用户审核 |
| CI 调度依赖与验证适配 | 容器内 Z3 依赖、正式 IR key 和 cache ABI 断言 | CI 安装及求解预检、4 个脚本和 15 个测试文件已更新 | Z3 实际求解、solver/oracle 105 项及入口 11 项通过；native/NPU 待 CI 复跑 | 待用户审核 |

当前工作补齐 CI 的调度依赖与验证适配。测试容器加载环境后，使用同一 Python
安装 `z3-solver>=4.13.0,<4.15.5`，先检查导入、版本与最小 SAT 求解，再运行测试。
Z3 仍为调度器可选依赖，不加入通用运行时依赖。迁移测试与脚本使用正式的
constraint、pipeline、storage、handoff、local sync 和 auto 元数据名称；cache
ABI 明确断言 constraint 5、sync 2、auto 3。同步更新负例诊断及 legacy 元数据
隔离检查，保留原有用例、物理存储与同步协议 oracle，不增加 skip 或放宽断言。
完整 PR 的格式 CI 已通过；本轮全部 111 个 Python 文件再次通过 Ruff 0.17.3
格式及 lint 检查，工作区 diff 检查通过。

本地主机使用 Z3 4.15.4，solver/oracle 105 项、独立 `s2_check_solver.py` 入口及
S2 入口回归 11 项通过。CI YAML、内外两层 Bash 语法与 Python 语法检查通过。
当前 Windows 环境缺少匹配的 TVM/native 构建和 NPU，IR、缓存及设备 oracle 的
执行结果须由更新后的 Linux/A3 CI 确认；不将修复前的通过结果作为本轮验收。
本步只修复 CI 依赖与现存验证材料，不改变其最终合入范围；推送后等待审核。

本步独立验证入口：

| 入口 | 环境 | 验证内容与通过条件 |
| --- | --- | --- |
| `s2_check_solver.py --output <报告路径>` | Python、pytest、受支持的 Z3，无需 TVM/NPU | 真实 solver 探测及全部 105 项数值协议/oracle 通过 |
| `s4_check_sync_coverage.py`、`s8_check_integration_ir.py` | Linux、匹配的 native 构建与 build manifest；传入 `--platform a3 --output <报告路径>` | 新元数据贯通最终 IR，损坏协议被拒绝；auto 的物理存储、缓存及 source oracle 通过 |
| `s7_run_sync_stress_npu.py`、`s8_run_workloads_npu.py` | A3 NPU 及匹配构建；按入口参数运行 | 正式 local sync/handoff/auto 元数据被设备验证脚本识别，原有数值及协议检查通过 |

关键源码修复已实现：Auto 调用审计按公开 ABI 区分 Fill 与 Add 的目标
操作数，并验证 Fill helper、目标类型及无访存副作用的算术标量。存储别名规范化
同步改写 Block 的 reads/writes，覆盖外层指针 Let 绑定且没有 match_buffer 或
根名称变化的情况；未改变的 BufferRegion 保留身份。受影响的 constraint/auto
缓存 ABI 递增，legacy 及独立同步策略的缓存行为保持不变。
临时回归使用真实 TVM TIR/FFI，覆盖合法与非法 Fill、Add、外层 Let 别名、
match_buffer、源 IR 不变及重复规范化；本环境缺少匹配的 TVM/native 构建，
尚未运行其 native 断言，上述临时回归仍未纳入仓库。本步仅同步已在 PR 中的
迁移测试与验收脚本，不改变其最终合入范围。
constraint 配合 legacy 同步时，排队 Wait 与事件 ID 复用的验证风险仍待 native
最小用例及设备压力测试确认；本轮不改变同步策略或插入额外屏障。
固定源码基线、主仓参考、默认策略及 schema 兼容逻辑保持不变。历史设备结果见
第 15 节，不沿用历史 PASS 作为本次构建结果；推送后等待用户审核。
实施状态与验证状态分列：“已实现”不等于“native/NPU 验收通过”。

## 4. 分步交付与验收

### S0 计划和技术参考

- 将本计划与技术设计提交到指定工作分支。
- 明确实施基线、参考提交、审核流程、验证口径和停止点；基线改变时同步校正。
- 验收：提交只包含上述文档；远端分支存在且提交号与本地一致；报告链接。
- 停止：等待用户对计划的反馈以及是否继续 S1 的明确指令。

验证脚本设计（文档步骤，无需 NPU）：`s0_check_docs.py` 检查计划状态、链接、
代码围栏和步骤完整性；`s0_check_publication.py` 检查 remote/分支、提交范围与
本地/远端提交一致性。

### S1 fork 基线审计与前置合同

- 核对 43b0b697 的实际 Pass、默认配置、FFI、代码生成与现有回归，更新技术参考。
- 审计资源分类/操作效果、可选 workspace 参数、未知调用和手写状态的处理；
  决定后续规划器所需的最小前置保护及可共享接口。
- 核对 DMA、切片与 Buffer view 的 elem_offset，以及旧 annotate_address 的地址合同。
  记录实际 Bisheng 构建身份、有效编译参数和缓存层次，区分真实编译与缓存恢复。
- 回归已有 Persistent、L1 分形多版本、zero/one-trip 和物理地址复用合同。
  只修复确认的缺口，不重复移植已有设施。
- 验收：形成逐项源文件/配置/测试映射，明确需补入的保护及其实际位置；
  如果本步补生产修复，须有相应 native 回归，涉及设备行为则另列 A3 结果。
- 后续依赖：S2、S3 均以本步确认后的合同为准。

验证脚本设计：

| 入口 | 环境 | 独立验证内容与通过条件 |
| --- | --- | --- |
| s1_check_baseline.py | 主机 Python + Git | 核对实际基线、Pass/默认配置及测试映射，区分已有能力与真实缺口 |
| s1_check_legacy_lowering.py | 已构建 TVM/专仓，无需 NPU | 基础 Vector/GEMM/CV 的 legacy lowering 与确认后的基线回归；按支持的合同验证生成代码 |
| s1_run_baseline_npu.py | A3 NPU | 运行本步确认的基准算子，数值与重复 launch 正确；保留 S1 历史记录 |

### S2 A3 能力、任务图与 solver 合同

- 明确 UB/L1/L0 各池可用预算、合法 core/PIPE/搬运路由和事件域；
  不复用 950 容量、延迟或 flag spill 参数。
- 在专仓 lowered call 上构造受审计的任务、读写区间、别名和副作用模型；
  共享旧规划器可用的抽取逻辑，不复制一套矛盾分类。
- 区间端点形成前先提升内部索引算术位宽，保留 elem_offset 和切片偏移；
  配置寄存器 effects 与普通 buffer 依赖分开建模，不假定其状态可以多版本化。
- 迁移纯数字约束求解、旧 FFI 接入和独立结果 verifier；新依赖按需加载，
  无 solver、无解和超时给出明确诊断，legacy 不增加拒绝条件。
- 验收：RAW/WAR/WAW、资源冲突、未知调用、别名及无解/超时测试通过；
  静态 TIR 和 native 注册测试通过；默认/显式 legacy 行为保持一致。

验证脚本设计：

| 入口 | 环境 | 独立验证内容与通过条件 |
| --- | --- | --- |
| s2_check_solver.py | 主机 Python + solver 依赖，无需 TVM/NPU | DAG、RAW/WAR/WAW、资源冲突、配置寄存器 phase 的纯协议/约束、无解和受控超时；结果通过独立 verifier |
| s2_check_task_ir.py | 已构建 TVM/专仓，无需 NPU | 任务 effects、int32 边界、动态尾界、非零 elem_offset/连续切片、别名及 legacy 隔离；错误结果被拒绝 |
| s2_check_capabilities.py | 主机 Python，无需 NPU | 验证 A3 能力模型的预算、对齐、PIPE/搬运与事件域一致性；不把模型检查当作设备实测 |

### S3 循环接口与 Persistent 安全性

- 保留 Parallel 的 A2/A3 Vector lowering；追加外层调度注解时避免逐维重复。
- Pipelined 保留现有 num_stages/order/stage/sync/group/cross_interval，兼容补 annotations。
- Persistent 回归已有参数合法性与 partial-single-wave 保护，再补 num_stages/annotations
  和非整除 group 的有效 tile 谓词；只修复确认的安全缺口，静态非法域保留明确诊断。
- Task/Stage 的标记只与验证和消费器一起开放；不静默接受尚未实现的调度语义。
  Task 内部条件与候选任务选择分别分析，保持完整任务同步边界及单 PIPE/core 合同；
  未支持的控制流明确拒绝，PerCoreTask 仍后置。
- 验收：FFI/IR 注解与任务域覆盖测试通过，旧调用兼容；Persistent 的已有安全合同
  与新增尾块在 A3 做数值验证。注解透传不代表 wave 流水已通过，后者归 S5。

验证脚本设计：

| 入口 | 环境 | 独立验证内容与通过条件 |
| --- | --- | --- |
| s3_check_loop_ir.py | 已构建 TVM/专仓，无需 NPU | Parallel/Pipelined 参数兼容、注解位置、Task/Stage 合法与非法组合 |
| s3_check_persistent_domain.py | 已构建 TVM/专仓，无需 NPU | 从实际生成循环核对 tile 恰好覆盖一次；检查少于 core、非整除 group、单/多 wave 和非法域诊断 |
| s3_run_loop_npu.py | A3 NPU | 验证 Persistent 安全合同、尾块、有效 tile 输出及重复调用；不据此宣称 wave 流水已支持 |

### S4 单核约束规划与最终同步协议验证

- 在现有 PipelinePlanning 内增加显式选择的 constraint 策略，默认 legacy。
- 当前首片限 A3 AscendC、单 Vector、静态循环、单 stage/单 version；
  使用审计过的 GM→UB→Vector→GM，拒绝 mixed C/V、opaque/raw 和手写 flag 歧义。
- 输出现有 stage/order ABI：任务展开必须与 injector 一致；只运行一个 planner、
  一次旧 injector、一套旧同步，不引入额外异步队列语义。
- 在最终旧同步输出后验证事件 set/wait 配对和活跃区间；不把模分配 ID 视作安全证明。
  回归当前基线的同步行为，不为验证方便插入已确认冗余的等待。
- 将受支持任务图的 RAW/WAR/WAW 及最终地址复用依赖映射到最终代码，逐条确认
  由合法 PIPE 隐式顺序、barrier 或 set/wait 完成关系覆盖；配对正确不能代替依赖覆盖。
- 验收：策略配置和缓存隔离、ABI、拒绝条件、最终事件协议及 legacy 回归通过；
  A3 完成数值、尾块和重复 launch 验证后，才能标记本步验收完成。

验证脚本设计：

| 入口 | 环境 | 独立验证内容与通过条件 |
| --- | --- | --- |
| s4_check_pipeline_ir.py | 已构建 TVM/专仓，无需 NPU | strategy/cache 隔离、stage/order 的展开 ABI、单次规划/注入及不支持输入的诊断 |
| s4_check_sync_coverage.py | 已构建 TVM/专仓，无需 NPU | 最终依赖覆盖、事件配对和活跃槽复用；注入缺 wait/重复 set 等负例必须失败 |
| s4_run_vector_npu.py | A3 NPU | GM→UB→Vector→GM，覆盖数值、尾块和重复 launch；legacy 与 constraint 分别执行并比较 |

### S5 跨迭代流水和多缓冲

- 引入经 A3 校准的 latency/II、多个 stage 和旧 injector 的多版本展开。
- 回归已有 L1 分形 tail/stride 修复；仅补入 S1 确认仍缺失的必要修复。
- 计算真实 padding、对齐、临时区及物理版本占用；验证跨迭代事件活跃期。
- 开放手动 Stage/enable_offset 前，验证适用配置寄存器的 reaching setter、入口值
  和下一次物理覆盖顺序；条件 setter 或未知 effects 保持保守依赖。
- 单独验收带 guard 的 Persistent wave 流水，不由 S3 的接口测试推导其正确性。
- 验收：0/1/N trip、trip 少于 stage 数、非整除尾块、1/2/3 版本及容量边界通过；
  A3 数值、稳定性与同条件性能记录完整，性能收益不预设。

验证脚本设计：

| 入口 | 环境 | 独立验证内容与通过条件 |
| --- | --- | --- |
| s5_check_versions_ir.py | 已构建 TVM/专仓，无需 NPU | 实际版本/容量、L1 tail/stride、事件活跃期及配置寄存器跨 Stage/迭代覆盖；错误状态被拒绝 |
| s5_run_pipeline_npu.py | A3 NPU | 0/1/N trip、短循环、1/2/3 版本及尾块；数值正确且无复用污染/挂起 |
| s5_run_persistent_pipeline_npu.py | A3 NPU | 独立验证带 guard 的单/多 wave 流水，全部有效 tile 恰好执行一次且尾 wave 同步正确 |

### S6 生命周期与 owner/epoch

- 迁移 alias-aware owner、完整 write-before-read epoch 和版本索引合同。
- 校验初始化、生产完成、消费完成、下一次写入和退出路径；处理条件、嵌套、
  连续循环及共享存储的多视图，不将 buffer 名称当作独立性证明。
- 显式地址参与版本、布局和指针改写时，保持 storage/view 的基址与偏移一致，
  同步更新或清理分配元数据。精确 placement 是可选扩展，不作为 S4 的前置条件；
  如采用主仓新合同，须显式正版本数并验证完整物理区间，不能改变 legacy 地址语义。
- 本步先在 bridge 上分析和校验，不自动从旧 injector/同步器夺取生成所有权。
- 验收：readiness/reuse、guard、zero-trip 和跨 tile 状态测试通过；
  新旧逻辑与最终物理别名关系一致，A3 重复运行和复用压力测试通过。

验证脚本设计：

| 入口 | 环境 | 独立验证内容与通过条件 |
| --- | --- | --- |
| s6_check_owner_ir.py | 已构建 TVM/专仓，无需 NPU | owner/epoch、地址/版本/别名、raw pointer 与分配元数据；错误 owner 被拒绝，可选精确 placement 启用时拒绝未知项 |
| s6_check_lifecycle_guards.py | 已构建 TVM/专仓，无需 NPU | 条件、可空、嵌套/连续循环及退出路径的 readiness/reuse；不存在无来源 wait |
| s6_run_reuse_npu.py | A3 NPU | 随机 scratch、跨 tile、多次 launch 复用压力；数值正确且无跨轮污染 |

### S7 新同步生成器

- 先在固定保守调度下引入新生成器，再逐步开放重排；按 A3 事件域分配并诊断。
- 消费最终内存规划的地址/大小，重建物理别名和复用边；不能沿用早期逻辑图代替。
- 覆盖 GM workspace 的生产通知和消费释放；同一依赖只由一个自动同步器负责。
  VS/标量副作用未覆盖前，不移除其旧保护。
- 验收：核内/跨核 set/wait、事件预算、显式 ID 冲突、初始化与 drain、workspace
  双向握手通过；A3 长流水、多 core、重复 launch 无死锁或串扰。

验证脚本设计：

| 入口 | 环境 | 独立验证内容与通过条件 |
| --- | --- | --- |
| s7_check_event_budget.py | 已构建 TVM/专仓，无需 NPU | 事件域上限前/恰好上限/超限、显式 ID 冲突和物理地址别名；正确分配或明确拒绝 |
| s7_check_handoff_ir.py | 已构建 TVM/专仓，无需 NPU | GM workspace 生产通知/消费释放、初始化/drain、VS/标量保护及同步单一所有权 |
| s7_run_sync_stress_npu.py | A3 NPU | 长流水、多 core、重复 launch 与并发 stream；进程级超时检测，无挂起/串扰 |

### S8 完整调度与混合 C/V 接管

- 在已证明的任务、版本与同步合同上接入完整约束调度，扩展混合 C/V。
- 明确接管边界：同一区域退出旧 injector/旧自动同步，不叠加新旧版本管理。
- 保留 A2/A3 GM workspace 路径，不把 950 直连、VF 或 flag spill 当作可移植能力。
- 验收：GEMM、归约、GEMM+Vector/Attention 等代表性混合算子与旧 Expert 回归通过；
  A3 记录编译、host 与设备耗时及回归情况。PTO 若纳入，另列独立验收。
- 完成后仍保持 opt-in；是否改变默认策略另由用户决定。

验证脚本设计：

| 入口 | 环境 | 独立验证内容与通过条件 |
| --- | --- | --- |
| s8_check_integration_ir.py | 已构建 TVM/专仓，无需 NPU | 接管区域只保留一套规划/版本/同步；保留 A2/A3 搬运与 legacy 合同，不出现 950 专属调用 |
| s8_run_workloads_npu.py | A3 NPU | GEMM、归约、混合 C/V 和旧 Expert 回归；所有定义输出满足 golden，重复调用稳定 |
| s8_benchmark_npu.py | A3 NPU | 正确性前置通过后，在同输入/设备/计时法下分别记录编译、host、设备耗时与回归；不预设提速结论 |

## 5. 验证脚本交付规范

第 4 节的脚本名均为计划入口，拟位于 `scripts/ascend_migration/`。
每步实施时交付对应脚本和最短运行命令，并更新实际路径、用例与结果；
入口重命名也须同步更新。

- 每个脚本可从仓库根目录单独运行，不依赖先跑其他验证脚本或推进下一步骤。
  允许依赖本步已构建代码及明确列出的 Python/TVM/CANN 环境，不自动安装、
  重编译或修改用户的全局配置；验证用例可显式选择 legacy/constraint 等配置，
  并在结果中记录。环境问题应清楚报告。
- 纯主机、native 无 NPU、NPU 三类依赖分别说明。`--help` 不应初始化 NPU；
  复用已有 pytest 时精确选择本步用例，保留原断言而非再造一套相同测试。
  各入口可共享薄 runner 的环境检查、参数和报告逻辑，不重复建设测试框架；
  纯主机入口不导入 tilelang/TVM，native/NPU 入口要求原生构建与当前源码匹配。
- NPU 入口接受显式 `--platform a3`、`--device`、`--seed`、`--repeat`、
  `--timeout` 和 `--output`。本环境使用 a3；保留 a2 参数兼容的旧入口也不得仅改标签
  就将同一设备结果记成两代通过。主机入口只要求与其验证有关的参数。
- 退出码约定：0 为 PASS，1 为验证失败，2 为环境/参数错误，3 为整体 SKIP，
  124 为超时。包装 pytest 时需明确转换；未收集用例或全部跳过不能报告 PASS。
- PASS 仅表示该入口全部必跑用例实际通过。部分必跑用例未执行、仅 xfail、
  无设备或脚本未实现均不能记成 PASS；报告区分 FAIL、BLOCKED、SKIP 和
  NOT_IMPLEMENTED，整步验收还要核对所有要求的入口及 A3 结果。
- `--output` 写入机器可读 JSON，记录步骤、实际提交号、命令、平台/设备、
  源码脏状态、调度配置、Python/TVM/CANN/torch_npu 等适用环境版本、随机种子、
  用例通过/失败/跳过数和错误摘要。
  涉及设备编译时记录所选 Bisheng 路径与完整构建身份、解析后的有序 flags 及缓存模式；
  编译验证使用隔离缓存或显式绕过缓存，冷热缓存/恢复测试另列结果。
  不假定某一层缓存加入工具链身份就会使全部缓存自动失效，也不因此迁入主仓运行链。
  纯 IR 测试不要求不适用的设备工具链；基于实际编译的证据与缓存命中结果分别记录。
  大日志/生成源码可保存到旁边的用户指定目录，不提交二进制或运行产物。
- 脚本既要覆盖正常结果，也要有缺依赖、非法输入或错误调度等可判定的失败路径。
  脚本只检查自身文件存在、或能 import，而不验证本步合同，不算完成验收入口。
- 每步最终报告列出脚本、完整命令、真实结果及未执行原因。缺少 NPU 时仍交付
  可在目标环境运行的入口，但状态保持“未运行/待环境验证”。
- 可能挂起的 NPU 用例由独立进程隔离；超时后停止该组验证，不能继续复用尚未
  确认健康的设备上下文，也不把线程超时当成 NPU 任务已经终止。

以下命令仅说明拟议接口；须在相应步骤实际提交脚本后才能运行：

```bash
# 已构建 TVM/专仓，无需 NPU
python scripts/ascend_migration/s4_check_pipeline_ir.py --platform a3 --output /tmp/s4-ir.json

# 在实际 A3 环境运行；平台参数必须与设备匹配
python scripts/ascend_migration/s4_run_vector_npu.py --platform a3 --device 0 --seed 0 --repeat 100 --timeout 120 --output /tmp/s4-a3.json
```

## 6. 统一验证与进度记录

每次进展更新都填写以下信息；仅写实际执行结果：

| 记录项 | 必填内容 |
| --- | --- |
| 当前步骤 | S 编号、范围、实现状态和人工审核状态 |
| 变更 | 实际文件与行为，必要的技术方案调整 |
| 检查 | 真实命令或工具、环境、结果、失败原因 |
| 验证脚本 | 本步实际交付的入口、最短运行命令、适用环境及 JSON/日志路径；设计、已交付、已执行分别记录 |
| native | 构建/注册/lowering/codegen 是否验证，不能由 Python 语法检查代替 |
| A3 | 实际 CANN、设备、用例和结果；没有运行则标明未运行；A2 历史记录不作为门槛 |
| 发布 | 远端分支、本步骤交付链接与推送结果；提交详情查看 Git 历史 |
| 审核 | 用户反馈、修复内容、是否明确允许下一步 |

验证逐层推进：数据/solver → 静态 TIR → native 构建/lowering → 代码生成 →
A3 数值与同步压力 → 性能。优先验证真实依赖、协议和覆盖范围，不固定具体
flag 编号或整份生成源码。可能挂起的设备用例使用独立进程控制超时。

各步骤在对应的主机、native 或 Linux/CANN A3 环境中验收；缺少相应运行结果时
保持“待环境验证”。进展随交付更新第 3 节，并按本节记录实际验证证据。

## 7. S1 实际交付与验证记录

### 7.1 范围与有效决策

- 用户已确认 S0 计划并授权 S1。工作分支和固定源码基线保持第 1 节所列值；
  S1 交付时未开始 S2，也未新增调度策略、solver 或 Task/Stage 接口，默认配置不变。
- 技术参考第 11 节补齐实际 Pass 顺序、配置、旧 FFI、资源归属、workspace 参数、
  Persistent、L1 版本及地址复用的源码/回归映射。已有资源归属和 workspace 保护复用，
  不重复迁入；未知调用、复合 PIPE、寄存器状态及完整 effects 合同留给后续步骤。
- 确认并局部修复非零 `elem_offset` 被重复累加：DMA 源/目标/tmp、atomic 源/目标，
  以及三个 Python BufferRegion 指针 helper，在传给 `access_ptr` 前去掉已经计入的
  view 基址。回归覆盖零偏移、两端非零偏移、连续 view、1D/2D stride 和公开 fill 路径。
- 旧 `annotate_address` 仍是字节地址/复用合同；不引入新 placement API，不统一原有
  auto 对齐与 linear 保留地址的差异。非对齐地址的 native 数学检查不表示设备支持；
  设备偏移用例使用满足相应搬运/向量约束的对齐切片。

### 7.2 环境、构建和缓存证据

| 项目 | 实际记录 |
| --- | --- |
| 主机 | Python 3.12.9；torch 2.12.0+cpu；torch_npu 2.12.0；CANN 9.1.0 |
| TVM | 固定子模块 `c2921fdaf795b1103d21abc962e83a209c7258d7` 的临时副本，应用仓库自带两份补丁；包含 patched Python parser；原仓子模块保持干净 |
| 构建 | 显式 CMake/Ninja，Ascend 开启，CUDA/ROCM/LLVM/GTest/libbacktrace 关闭；构建产物在 `build/`，不覆盖包内旧库、不安装 Python 包、不修改全局配置 |
| 身份记录 | `/tmp/ascend-migration-s1/build/native-build.json`；核验源码、TVM Python/依赖、补丁、CMakeCache、构建日志及三个 `.so` 哈希；同时核对实际加载路径 |
| 真实设备 | A3 `Ascend910_9362`，逻辑 device 0；本机未提供 A2，不能据此宣称 A2 设备验收通过 |
| Bisheng | PATH 选择 `/home/developer/Ascend/cann-9.1.0/bin/bisheng`，realpath 为同一 CANN 的 `tools/bisheng_compiler/bin/bisheng`；clang 15.0.5，构建时间 2026-04-30T17:11:05+08:00 |
| 编译器哈希 | SHA256 `7ffc97205a0582ce00a9ec163530696bf3971b24669c43c1c02cc9c7b56f541f`；完整 `--version`、resource dir 和 CANN version.info 写入设备 JSON |
| 编译与缓存 | 每个设备用例新进程、新 KernelCache 目录并 disable_cache，直接编译新 kernel；记录实际 Bisheng argv、返回码、源码/产物哈希、有效有序 flags 和实际 pass_configs；Cython host wrapper 缓存另记，不当作设备编译证据 |

预装库虽可导入，但没有可信的当前源码构建记录，未作为最终 native 验收依据。
可信未修复新库的新增回归为 21 failed / 15 passed，证实偏移问题；修复后新增回归
A2/A3 各 36 项通过，并纳入下列各 105 项的完整 native 入口。
构建准备和 manifest 生成命令见技术参考第 11.5 节；实际配置、构建及复现日志保存在
`/tmp/ascend-migration-s1/build/`。首次 8 并发构建以 130 中断后用 16 并发续建成功；
一次与链接重叠的检查出现 `file too short`，等待构建成功结束后重新验证通过，未计作通过证据。

### 7.3 已交付入口与运行结果

以下保留 S1 当时执行的命令，含 A2 历史记录；后续不要求重跑 A2 专属检查。
运行前在仓库根目录执行 `source set_env.sh`。native/NPU 入口使用该次显式构建的记录：

```bash
export TL_ASCEND_MIGRATION_BUILD_MANIFEST=/tmp/ascend-migration-s1/build/native-build.json

python scripts/ascend_migration/s1_check_baseline.py --output /tmp/ascend-migration-s1/host.json
python scripts/ascend_migration/s1_check_legacy_lowering.py --platform a3 --output /tmp/ascend-migration-s1/native-a3.json
python scripts/ascend_migration/s1_check_legacy_lowering.py --platform a2 --output /tmp/ascend-migration-s1/native-a2.json
python scripts/ascend_migration/s1_run_baseline_npu.py --platform a3 --device 0 --seed 0 --repeat 100 --timeout 120 --output /tmp/ascend-migration-s1/s1-a3.json

# 负例：本机实际为 A3，必须拒绝误标成 A2，不是 A2 设备验收
python scripts/ascend_migration/s1_run_baseline_npu.py --platform a2 --device 0 --seed 0 --repeat 100 --timeout 120 --output /tmp/ascend-migration-s1/platform-a2-negative.json
```

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 纯主机基线审计 | PASS，18 项；不导入 TileLang/TVM/NPU | `host.json` |
| native，参数 a3 | PASS，105 passed / 0 failed / 0 skipped；必跑节点完整 | `native-a3.json` 及其 artifacts/JUnit |
| native，参数 a2 | PASS，105 passed / 0 failed / 0 skipped；必跑节点完整 | `native-a2.json` 及其 artifacts/JUnit |
| runner 主机回归 | PASS，20 项；含 skip/xfail、零收集、遗漏必跑节点、超时杀进程组、缺构建记录、旧源码/库身份及优化模式负例 | `runner-tests.xml` |
| 偏移用例定向设备复验 | PASS，DMA/atomic/fill 各实编一次、同一 kernel 100 轮 launch/golden | `offset-repeat100.json` |
| A3 NPU | PASS，单次完整入口 14/14、0 failed/skip；14 次真实冷编译，同一 kernel 各 100 次 launch/golden，共 1400 次 | `s1-a3.json`，artifacts 路径见该报告 |
| 实际 A3 误标为 A2 的负例 | 按预期 BLOCKED、退出 2；0 编译/launch，全部 14 项未执行 | `platform-a2-negative.json` |
| A2 NPU | 历史未运行；用户已取消本环境的 A2 验收门槛 | 不作为后续执行要求 |

上表证据文件均相对 `/tmp/ascend-migration-s1/`。

native 两个参数分别作用于新增 S1 fixtures；既有 fixtures 保留各自的 A2/A3/PTO
目标与配置，报告逐类注明。它们是主机 TIR/lowering/codegen 验证，没有调用 Bisheng
或执行 kernel，不能替代两代设备结果。

设备预检最初尝试将已有 IR-only 标量 C Persistent fixture 用于数值测试，发现部分
输出未写，随即停止该组；只确认该载体不适合本次数值验收，未据此认定生产 Persistent
算法有缺陷，也未把缓存行竞争推断当成已证实根因。设备入口改为复用现有
`test_tilelang_ascend_language_l1_to_l0.py` 的 GEMM Persistent kernel，显式匹配
launch core 数和 wave_size，分别检查 8 tile/8 core、4 tile/8 core 和 9 tile/4 core。
原有 IR fixture 和生产 Persistent 实现均保持原样；初次失败的 `s1-a3-smoke.json`
及修正后剩余用例预检的 `s1-a3-remainder-smoke.json` 均保留，不充作正式整组通过证据。

首次 100 轮设备运行在 DMA 偏移用例发现数值失败，前 11 项已通过，后续用例未执行。
该地址 factory 原为 native 生成代码检查设计，未提供 GM→UB→GM 的运行时同步；
首次 smoke 偶然通过不构成安全证明。仅在设备入口对 DMA/atomic/fill 偏移用例
显式开启已有 `tl.ascend_auto_sync` 和 `tl.ascend_memory_planning`，生产默认值和
native 地址断言不变；失败批次另存为诊断，修正后重新验证完整正式入口。
对照生成代码的 GM 地址仍为 `a[40]` / `o[24]`，两次 copy 之间新增既有
`MTE2_MTE3` set/wait 完成关系；该配置下三个偏移用例各 100 轮定向检查通过。
该失败批次保存在 `s1-a3-copy-unsynchronized-failure.json`，原 artifacts 目录保留；
后续每次入口调用创建独立 artifacts 目录，避免复跑覆盖旧日志。

runner 回归命令：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q testing/python/ascend_migration/test_s1_runners.py --junitxml=/tmp/ascend-migration-s1/runner-tests.xml
```

格式检查限定本步文件。Python lint 通过；C++ 全文件 clang-format 检查只有固定基线
已有第 370 行的三条相同诊断，本步修改行没有新增诊断，未为此改写无关代码。
日志分别为 `build/clang-format-check.log` 和 `build/clang-format-baseline.log`。
完整格式检查记录为 `format-report.json` 和 `format-report.md`。
JSON/JUnit、生成源码、二进制和大日志均不提交。

### 7.4 审核与剩余验证

S1 实现、入口和审计记录已交付；A3 已完成上述验收用例。用户已明确后续只要求 A3
验证并授权开始 S2；历史 A2 设备未运行，不再作为 S1 的待验收条件。
本步提交只推送到 `erhsh/codex/ascend-a2a3-scheduler-migration`，提交详情查看该分支
Git 历史，推送结果在最终交付报告列出。不创建 PR、不合并、不强推；推送后停止，
等待用户审核和明确的下一步指令。

## 8. S2 实际交付与验证记录

### 8.1 范围和接口

- 用户已明确授权 S2，并取消后续 A2 专属测试与验收门槛；实现起点为 S1 提交
  `685bce5e`，第 1 节固定源码基线和主仓参考提交均不改变。
- 新增显式只读 `AnalyzeAscendSchedule(PrimFunc, platform="A3") -> Map` 和旧注册
  `tl.transform.AnalyzeAscendSchedule`。图保留 task/group/loop/guard/storage 与依赖，
  不修改原 IR，不接入 `phase.py`，不新增 PassContext 策略或默认行为。
- 共用原有资源归属分类、workspace 解析和 injector 分组规则；共享抽取保持旧逻辑。
  首片严格区分受审计的可调度组、保守分析结果与未知 effects 的明确拒绝。
- 新增独立 A3 capability profile，以及纯数字 straight/modulo solver 与不依赖 Z3
  的结果 verifier。整数协议不等于已生成正确的多版本、流水或同步代码；S3/S4 及
  后续执行接管不属于 S2。具体边界见技术参考第 12 节及实现骨架。
- Z3 只作为 `ascend-scheduler` 可选 extra；legacy 导入不探测 Z3。当前测试依赖安装
  在 `/tmp/ascend-migration-s2/venv`，未修改全局 Python 环境。

### 8.2 审计限制与修正记录

能力数据来自当前 AscendC 编译器预算和 CANN 9.1/dav-2201 声明，不是设备测量。
跨核事件的四位编码不能证明可自由分配的槽位，因此本步仍不支持跨核调度预算。
旧 auto allocator 的 UB 超量探针曾退出 139，根因未定位；正式容量负例验证新增
capability verifier 的结构化拒绝，不把该崩溃当成通过，不修改 default linear 行为。

实现评审复现过一个纯数字假 UNSAT：带 int64 完成时间上界和固定 manual stage 时，
II=1 可行，但二分搜索 `[1,16]` 会错误跳过该解。现改为全局 deadline 内升序搜索，
只有完整指定域均不可满足才返回 UNSAT，并永久保留该回归。还区分了 Z3 真正缺失、
导入损坏、UNKNOWN 和 TIMEOUT，独立 verifier 对故意漏掉资源对的候选仍能拒绝。

原生分析评审针对 managed GM operand、未声明 local pointer、view 扩张 allocation、
显式窄化 cast、DMA 参数隐藏内存/opaque effects、未知语义 attr 和条件 allocation
补入拒绝或保守处理。循环图仅声明单迭代分析，要求顺序迭代，不冒称已覆盖跨迭代
readiness/reuse；这些情况不影响 legacy 接受范围。

### 8.3 验证入口与结果

以下为本步交付的三个入口，均可从仓库根目录独立运行；native 入口要求显式构建后
的源码身份 manifest，脚本不自动重编译、安装依赖或修改全局配置。

```bash
source set_env.sh
export TL_ASCEND_MIGRATION_BUILD_MANIFEST=/tmp/ascend-migration-s2/native-build.json

/tmp/ascend-migration-s2/venv/bin/python scripts/ascend_migration/s2_check_solver.py --output /tmp/ascend-migration-s2/solver.json
/tmp/ascend-migration-s2/venv/bin/python scripts/ascend_migration/s2_check_capabilities.py --platform a3 --output /tmp/ascend-migration-s2/capabilities-a3.json
/tmp/ascend-migration-s2/venv/bin/python scripts/ascend_migration/s2_check_task_ir.py --platform a3 --output /tmp/ascend-migration-s2/task-ir-a3.json
```

| 检查 | 当前实际结果 | 证据 |
| --- | --- | --- |
| 纯 solver / verifier | PASS，105 项，含 13 项独立小规模穷举 oracle | `solver.json` |
| A3 capability | PASS，42 项，包含模型、预算、对齐、路由和事件域负例 | `capabilities-a3.json` |
| runner 隔离回归 | PASS，S1 20 项及 S2 11 项；覆盖必跑空文件不能躲在其他通过项后 | `runner-tests.xml` |
| native Task IR / FFI / legacy | PASS，171 项：Task IR 41、capability IR 16、solver FFI 9、S1 回归 105；无失败/跳过/漏必跑项 | `task-ir-a3.json` |

上表证据位于 `/tmp/ascend-migration-s2/`。原生构建沿用固定 TVM 提交的已补丁临时
副本，manifest 核验 C++/TVM 源码、patched Python、CMakeCache、日志和实际库路径。
新 API 不产生设备执行代码，本步未运行新的 NPU 调度验收，也不要求 A2 验证。
原生 source SHA256 为
`46aea1e2d37fee143fe4e3eb77b7c1575d7be6ee952075965b628334c9236ca3`，
153 个 C++/header 等构建输入；最终构建日志为
`build/native-final-boundary-build.log`，最终 JUnit/日志目录为
`task-ir-a3-artifacts-xxbb1dwu`。验证期间源码处于 S1 提交上的 dirty 工作区，
报告保留当时的 commit/dirty 状态与源码哈希；不能把报告中的旧提交号误当成
最终提交前没有修改源码。

初次 native 运行的测试 IR 构造错误（单元素 SeqStmt、缺少 IfThenElse 的 else 参数）
诊断保留在 `task-ir-a3-first-failure.json`；修正后 166 项通过的中间结果保存在
`task-ir-a3-before-final-boundary-checks.json`。最终 171 项另包含 nested Block 初始化/地址提示
守恒、Float/Bool count 拒绝和 cold legacy/Z3 隔离，不用较早报告替代最终验收。

runtime 依赖为原环境 Python 3.12.9，隔离 venv 中实际 Z3 为 4.15.4.0；真实缺 Z3
的原环境入口另验证为 BLOCKED/2，见 `solver-missing.json`。runner 回归命令为：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/ascend-migration-s2/venv/bin/python -m pytest -q testing/python/ascend_migration/test_s1_runners.py testing/python/ascend_migration/test_s2_entrypoints.py --junitxml=/tmp/ascend-migration-s2/runner-tests.xml
```

修改的 Python 文件 lint、新 Python 的 100 列格式检查，以及三个新 C++ 文件的
clang-format 检查均通过；`git diff --check` 通过。格式记录为
`format-report.json` / `format-report.md`，产物不入库。
本步只提交推送 S2，提交详情查看分支 Git 历史，推送结果由最终报告列出；
S2 交付后已收到用户明确的 S3 执行指令，按本计划继续单步实施。

## 9. S3 实际交付与验证记录

### 9.1 范围与有效决策

- 实施起点为 S2 提交 `a56a4d13`，固定源码基线和主仓参考不变；按用户要求只做
  A3 验收，不新增 A2 专属测试。保留旧循环调用和 native FFI 参数兼容。
- Parallel/Pipelined/Persistent 增加 annotations，新增 metadata 只挂外层，保留
  原有 coalesced_width 行为；Python/native 双层拒绝保留调度字段注入。
- Persistent 支持末维非整除 group 的 padding 与有效 tile guard，内部索引算术
  提升至 int64；静态溢出明确诊断，符号正值的运行时范围仍是调用者合同。
- Persistent 的 num_stages 默认 0；正值只允许可证明无 guard 的单个完整 wave。
  多 wave、partial、非整除或无法证明上述条件的符号输入拒绝正值；wave 流水仍归 S5。
- Task 只允许一个 S2 已审计的 Vector call，Stage 只允许 0。早期检查结构、后期
  复用 S2 分析审计并消费；不开放 cost override、多调用/内部条件/Cube/手写同步、
  非零 Stage 或旧 stage/order 混用。不含标记的函数维持 legacy 路径。
- 实现位置、精确边界和未来扩展见技术参考第 13 节及公开语言说明。

### 9.2 独立入口与原生结果

```bash
source set_env.sh
export TL_ASCEND_MIGRATION_BUILD_MANIFEST=/tmp/ascend-migration-s3/native-build.json

/tmp/ascend-migration-s2/venv/bin/python scripts/ascend_migration/s3_check_loop_ir.py --platform a3 --output /tmp/ascend-migration-s3/loop-ir.json
/tmp/ascend-migration-s2/venv/bin/python scripts/ascend_migration/s3_check_persistent_domain.py --platform a3 --output /tmp/ascend-migration-s3/domain.json
python scripts/ascend_migration/s3_run_loop_npu.py --platform a3 --device 0 --seed 0 --repeat 100 --timeout 120 --output /tmp/ascend-migration-s3/s3-a3.json
```

manifest 必须由当前源码显式构建生成，不能复用 S2 manifest；创建方法沿用技术参考
第 11.5 节。入口不自动安装依赖或编译库。当前 native 使用既有隔离 venv，设备使用
原 Python 3.12.9、torch/torch_npu 2.12.0、CANN 9.1.0。

| 检查 | 实际结果 | 证据（相对 `/tmp/ascend-migration-s3/`） |
| --- | --- | --- |
| 循环与 Task/Stage IR | PASS，117 项：循环 74、标记 43；0 fail/skip | `loop-ir.json` |
| 实际 Persistent 域 | PASS，29 项：静态/符号正值、解释器故障注入及非法参数 | `domain.json` |
| S1/S2 native 兼容回归 | PASS，170 项，0 fail/skip/遗漏必跑项 | `s2-regression-a3.json` |
| runner 主机回归 | PASS，44 项：S1 20、S2 11、S3 13 | `runner-tests.xml` |
| A3 NPU | PASS，9/9 项；9 次真实冷编译、1000 次 launch/golden，0 fail/skip | `s3-a3.json` |

兼容回归比 S2 的 171 项少一项，是旧非整除 group 拒绝用例现已合法，改由 S3 的
真实域 exact-once 与数值验证覆盖；没有通过 skip 隐藏旧失败。大域溢出测试只检查
实际 IR 的位宽和首尾坐标，不枚举全部 tile。

最终 native source SHA256：
`daa2899ad727495a13162ad3083f7f6f37e9319204ad80d226e002acb430b940`，
155 个构建输入；manifest 为 `native-build.json`，构建日志为
`build-pending-final.log`。继续使用固定 TVM 提交及已补丁临时副本；验证报告记录
S2 HEAD 上的 dirty 工作区、源码哈希和实际库路径。报告中的提交号不是最终 S3
提交号，匹配依据是已核验的源码/库身份。

早期测试输出与中间构建仅作诊断；最终两个 native 入口及兼容回归均在上述冻结
manifest 下运行。另有针对 pending 后端门禁的回归：显式简化删除 dead marker 后
仍拒绝 PTO/A5；这证明跨改写的守恒，不把它表述为已复现标准 pipeline 绕过。

设备为实际 A3 `Ascend910_9362`、逻辑 device 0。六个静态 Persistent 域覆盖完整/少于
core/多 wave、内部 padding 和尾 wave，以及单 core：`[2,4]/8/g2`、`[2,2]/8/g2`、
`[3,3]/4/g8`、`[1,5]/8/g2`、`[3,10]/8/g4`、`[1,5]/1/g2`，各 100 轮。
符号 M 的同一已编译 kernel 以 M=128/384 两个正值绑定各运行 100 轮。
另有无 guard 单满 wave 的 num_stages=1 与消费 Task/Stage(0) 的 Vector add，
各 100 轮；全部逐次重置输出 sentinel 并比较 golden。前置 smoke 9/9 也通过，
但以完整 `s3-a3.json` 的 1000 次为本步设备证据，不把 smoke 次数叠加。

最终设备 artifacts 为 `s3-a3-artifacts-0k0wmop1`；每项保存真实编译 argv、有序
flags、resolved pass_configs、生成源码/产物 hash、工具链和实际加载身份。
所选 Bisheng 为 `/home/developer/Ascend/cann-9.1.0/bin/bisheng`，实际工具链
clang 15.0.5，compiler SHA256 为
`7ffc97205a0582ce00a9ec163530696bf3971b24669c43c1c02cc9c7b56f541f`。
各 worker 独立目录且 kernel 内存/磁盘缓存禁用；独立 host wrapper 缓存不充当
设备编译证据。所有设备 case 均匹配上述最终 source/库 manifest。

本步 Python lint、新 Python 的 100 列格式检查、四个修改/新增 C++/header 的
clang-format 检查，以及 `git diff --check` 均通过；文档相对链接和围栏检查通过。
记录为 `format-report.json` / `format-report.md`，日志、JSON、JUnit 和二进制不提交。

### 9.3 审核边界

S3 提交包含循环、验证/消费、测试、三个入口与文档；不接入 S4 constraint 策略，
不宣称跨 wave 流水、owner/epoch 或新同步器已实现。S3 已完成上述验证并按约推送；
用户随后明确授权 S4，按本计划继续单步实施。

## 10. S4 实际交付与验证记录

### 10.1 范围和配置

- 用户明确授权 S4，实施起点为 S3 提交 `8aa8b494`；固定源码基线、主仓参考和推送
  分支不变。当前 constraint 首片支持 A3 AscendC/auto，A2 保留 legacy，不做 A2 专属验收。
- `TL_ASCEND_PIPELINE_PLANNER` 选择 legacy/constraint，默认 legacy。原有 planner
  类保持，constraint 在同一入口分发，Python 仅该分支懒注册可选 solver。
- constraint 要求 auto_sync/memory_planning 开启、cross-core/VS 同步关闭；全内核
  受审计，至少一个静态 serial、num_stages=1 候选。只输出 stage 0/version 1 和
  旧 order ABI，不引入第二个 planner/injector/同步器。
- 最终只读验证位于最后 Simplify 与 ResourceScopeVerify 之间，通过稳定 task/iter
  身份检查原 logical 依赖和最终 UB 物理复用；按 set 时刻的完成快照审计事件配对、
  活跃槽、传递关系及 drain。相关 V→V 依赖需要显式完成证明，不加冗余等待来过检查。
- 静态 UB f16/f32 allocation 先逐维检查真实声明的乘积，再计算行 padding 和保守总预算，
  复用旧 allocator 的 196352 字节常量；最终地址/大小和访问范围再核验。手工 placement、未知 effects、
  多 stage/version、动态 padding DMA、跨 dataVar remap 和无法证明的控制流拒绝。
- 有限展开限额为每 trace 4096 硬件实例、65536 控制访问，单 loop extent≤65536。
  solver 的单位 latency/II 只用于抽象顺序，不是 A3 时延测量或性能收益承诺。
  完整合同见技术参考第 14 节与公开语言说明。

### 10.2 别名、尾块与验证载体

普通未声明 noalias 的 GM 参数保持保守；已确认 copy/add/copy 循环可能缺少上一轮
GM store 到下一轮未知别名 GM load 的完成路径，这类输入明确拒绝。
设备 fixture 显式声明 tir.noalias=True，A/B/O 三个实际独立 torch 分配满足该合同；
native 同时保留 noalias=False 的拒绝测试，不把独立分配事实推广给普通调用者。

尾块采用完整对齐短 tile：full192=64×3、tail160=64×2+32，后者独立尾 allocation
不自动代表物理 reuse。native 另用最终全/部分重叠地址的正负例证明 reuse 覆盖，
并验证物理顺序与原 logical 方向不能混淆。设备按 f16/f32、两种形状和两种策略
分别编译，同 CPU seed 生成输入，逐次重置输出 sentinel、比较 golden，并在入口
末端比较跨策略实际输入/输出 hash；仅相同 seed 或缓存禁用不能充当一致性证明。

### 10.3 独立入口和验证结果

```bash
source set_env.sh
export TL_ASCEND_MIGRATION_BUILD_MANIFEST=/tmp/ascend-migration-s4/native-build.json

/tmp/ascend-migration-s2/venv/bin/python scripts/ascend_migration/s4_check_pipeline_ir.py --platform a3 --output /tmp/ascend-migration-s4/pipeline-ir.json
/tmp/ascend-migration-s2/venv/bin/python scripts/ascend_migration/s4_check_sync_coverage.py --platform a3 --output /tmp/ascend-migration-s4/sync-coverage.json
/tmp/ascend-migration-s2/venv/bin/python scripts/ascend_migration/s4_run_vector_npu.py --platform a3 --device 0 --seed 0 --repeat 100 --timeout 120 --output /tmp/ascend-migration-s4/s4-a3.json
```

复用 S2 的隔离 venv，实际 Z3 为 4.15.4.0，preflight记录torch 2.7.1+cpu、
torch_npu 2.7.1.post4及CANN 9.1.0；
未安装全局依赖。三个入口的缺失/损坏 Z3 预检在 runtime 导入前给出 BLOCKED/2。
manifest 要由当前源码显式构建，不能复用 S3；方法沿用技术参考第 11.5 节。

| 检查 | 当前实际结果 | 证据（相对 `/tmp/ascend-migration-s4/`） |
| --- | --- | --- |
| 规划 ABI、拒绝边界和真实 cache key/路由 | PASS，45 项：planner 42、cache 3；0 fail/skip | `pipeline-ir.json` |
| 最终协议与 logical/physical 覆盖 | PASS，50 项，0 fail/skip | `sync-coverage.json` |
| S1/S2/S3 native 兼容 | PASS，170 + 117 + 29 项，0 fail/skip/漏必跑项 | `s2-regression-a3.json`、`s3-loop-regression-a3.json`、`s3-domain-regression-a3.json` |
| runner 主机回归 | PASS，77 项：S1 20、S2 11、S3 13、S4 33 | `runner-tests.xml` |
| A3 NPU | PASS，8/8 项；8 次真实冷编译、800 次 launch/golden，4 组跨策略输入/输出 hash 一致，0 fail/skip | `s4-a3.json` |

预算溢出修复后的最终 native source SHA256：
`a129ba375514d31697cddcad83ba67d681f38a6cbcf41abb29b1e6f1a55be5fb`，
158 个构建输入，`build-final.log` exit 0，`native-build.json` 已写入并核验。
旧 manifest 与报告以 `*-before-budget-overflow-fix.json` 保留为诊断。
固定 TVM 及已补丁临时副本沿用 S1，报告保存 S3 HEAD 上 dirty 工作区、源码/库身份与
真实加载路径；提交号不是最终 S4 提交号，匹配依据是已验证的 source/库 manifest。

首轮诊断发现迭代 marker 的 value 被旧 NarrowDataType 合法缩位后遭误拒，现允许合法
整数缩位并逐实例匹配数值；原始 ID node 保持 int64。synthetic fixtures 的旧 TVM
AttrStmt/call 构造错误也已修正，诊断日志保留，不用中间分项通过替代正式整组结果。

最终复核另确认巨大多维 UB shape 的容量可在旧 TVM 常量乘法中回绕为 16384 字节，
绕过分析后的预算检查。修复把逐维除法前置检查放在 Analyze 之前；Buffer 与 Allocate
声明各保留一个精确负例，均在进入旧 allocator 前拒绝。修复前已通过的原生和设备
报告以 `*-before-budget-overflow-fix.json` 保存为诊断，不计入修复后的最终验收。

最终设备 artifacts 为 `s4-a3-artifacts-a95k7t67`，实际 SoC `Ascend910_9362`、逻辑
device 0；Python 3.12.9、torch 2.7.1+cpu、torch_npu 2.7.1.post4、CANN 9.1.0，
Z3 API 版本 4.15.4。
PATH 所选 Bisheng 为 `/home/developer/Ascend/cann-9.1.0/bin/bisheng`，实际 clang
15.0.5，compiler SHA256 为
`7ffc97205a0582ce00a9ec163530696bf3971b24669c43c1c02cc9c7b56f541f`。
每项各 100 次 attempted/completed/golden，保存真实编译 argv、有序 flags、resolved
配置和生成源码/产物 hash；全部匹配最终 source/库 manifest。独立 worker 与缓存目录、
禁用 kernel 内存/磁盘缓存的控制沿用 S1/S3，host wrapper 缓存不计为设备编译。
上述 800 次仅统计预算修复后的正式入口，不叠加修复前或 smoke 的次数。

修改文件的 Python lint、8 个新 Python 的 100 列格式检查、6 个 C++/header 的
clang-format、`git diff --check` 及文档相对链接/围栏检查通过。记录为
`format-report.json` / `format-report.md`，JSON/JUnit、日志和二进制均不提交。

### 10.4 提交和审核边界

只提交 S4 的实现、永久测试、三个入口与文档；不推进 S5 的跨迭代流水、多版本、
硬件时延校准或新同步器。上述正式原生、A3 与格式检查已完成，按约提交推送到既定分支，
然后停止等待用户审核与明确的 S5 指令。

## 11. S5 实施记录

S5 实现、CPU 与 A3 设备验收已完成。本节记录实际证据与性能限制；未执行 A2 专项
测试。S5交付时停止等待审核；后续S6授权和结果见第12节。

### 11.1 实现范围与边界

- 保持 legacy 默认与原规划/注入/同步流水。constraint 多 stage 接收完整、带摘要的
  `tl.ascend_pipeline_cost_model` Map；内部 PassConfig 使用该版 TVM 支持的 DictAttrs。
  Python loader 校验模型内容，模型内容进入既有缓存键。legacy 不导入 Z3。
- `num_stages=1/2/3` 是版本搜索上限；规划器与旧 injector 共用实际版本、shape、stride
  推导，并对修订后的实际 N 再次验证。最终 remap 与 Allocate 记录真实占用，不从请求数
  推断版本。版本基址统一使用 `(logical_iteration-min) % N`。
- 旧 injector 的 trip1、stage[0,1,2] 中间 unit 重复执行已复现并修复；分段不重叠，
  steady 不产生负 extent。0-trip 是显式 no-op，不宣称执行了多版本分配。
- 非零手动 Stage 需要 constraint 正候选及显式 `enable_offset=True`。Stage 保持源程序
  imperative 值语义；冲突排程必须 UNSAT。任一显式 Stage 使候选未标注 unit 取 Stage0；
  候选外非零 Stage 拒绝，Stage0 可按原顺序消费。Task 成本 override 仍拒绝。
- 仅支持 loop-private、单行连续、每版 32B 对齐的 f16/f32 UB。每迭代须先 full-write
  后读；full-write 后的完整 in-place 更新仍需 producer/slot 证明。外部只读 UB、
  共享写 UB、跨迭代 carry 不在本片范围，不扩展为 S6 通用 owner/epoch。
- 最终证明分别检查静态 core/vid 的源 producer 与实际同 slot 最近 writer、物理覆盖及
  事件完成关系。managed mask 使用每个 Vector 域的一份状态，Legalize 末端只读检查
  reaching setter/入口未知/下一次覆盖；不复制 mask 到 buffer version，不额外插同步。
  未审计配置寄存器、opaque 或条件 stateful 操作仍拒绝。
- guarded Persistent 使用纯 If-noelse 保留每 logical tile 的 padding/末 wave guard，
  不把 loop_break 搬入 stage。原 `num_stages=0` 语义保留。
- staging horizon 为 64，窗口×unit 上限 65536、任务实例上限 4096；整数域与容量在
  展开前检查。既有 L1 分形 tail/stride 仅做 legacy 回归，不开放 constraint Cube。

### 11.2 A3 有效成本的测量政策

校准在独立 S4 源码/三库快照中运行 legacy 微基准，SoC 为 Ascend910_9362，CANN 9.1.0。
manifest schema 2 明确标记 `relocated_verified_build`，保留原构建命令和
CMAKE_HOME_DIRECTORY，逐字节验证原 manifest/cache/log、相对源码及三库；不冒称快照
重新构建。普通 schema 1 仍严格匹配当前源码和库。

矩阵为四类 GM→UB、UB→GM、fill、add × f16/f32 × tile32/64 × dependent/independent，
共 32 个微基准，对应 16 条成本记录。采用同一 `msprof op` BasicInfo-only 政策：
N=0/32/128/256/512，每 N 每 round 5 次预热、20 次测量，3 rounds；每个微基准先逐 N
做 3 次数值校验。独立模式使用固定 128-tile arena，在有界批次复用处 drain。
每次应用调用有 journal，并与 main_kernel 的唯一 BasicInfo 行一一匹配；PMU replay
不计为独立样本。频率、SoC、CANN、bisheng SHA 和有序 flags 均入证据。

多 N 斜率包含 issue、同步、批次 drain 的有效成本；它不是原始指令延迟。
hierarchical bootstrap 纳入 round 漂移，要求 95% CI 下界正、相对半宽≤20%、
相对残差≤10%。最多两批，重试必须合并全部原始样本与 rounds 再判定，保留首批失败；
不丢弃失败批、不 clamp 为 1。取 CI 上界向上取整至 ns，`quantum_ns=1`。
小 N 的早期噪声试验与 PMU 诊断不混入正式成本表。

完整校准已实际通过：32/32，16 records；12750 次 profiling 应用调用、10200 个
measured samples。GM→UB 的 f32/tile32 与 f32/tile64 independent 各两批，均以合并
全部样本后的结果通过；其余单批。两项 add/f32/tile64 复用已接受的同政策 pilot，
已核对生成源码、工具链及身份，不重新计为额外样本。
模型为 `/tmp/ascend-migration-s5/a3-effective-cost-model.json`，canonical SHA256
`f2aa7ecfa4e592199c81801140903352f6e34fc1cee7be220987d6c5753a13fd`，
生产 loader 独立复核通过。实际 ordered flags 为 `[-O2]`，current/rated 频率均1650MHz。
bisheng SHA256 为
`7ffc97205a0582ce00a9ec163530696bf3971b24669c43c1c02cc9c7b56f541f`。
`calibration-acceptance.json` 逐项核对32份 accuracy、34份 profile worker 的引用 SHA、
编译及调用计数：480 次 accuracy launch/golden，profiling 的510次 post-group
golden，接受证据内66次真实 bisheng 编译。profiling 不冒称每次调用均做独立 golden。
模型文件 SHA256 为
`5549a92dc630e629252945d31dbefb73145570f7e6b7487a41083282dae3b569`。
本次矩阵由 `/tmp/ascend-migration-s5/run-calibration-matrix.py` 分阶段驱动；永久入口的
`--calibrate-profile` 封装同一 worker、BasicInfo parser、合并拟合及导出政策，用于后续
重现。公开控制器的失败路径与导出合同通过 host 测试。

### 11.3 对照基线

普通非零 trip 用同一逻辑 kernel 比较两种 planner。Persistent 的 legacy 基线使用
`num_stages=0` 的串行 wave；constraint 使用 2/3，因为 legacy 的既定合同拒绝 guarded
positive stages。两侧逻辑域、输入、golden 和编译 flags 相同，报告保留各自 directive，
不宣称两侧 stage 参数相同。

零次 legacy 原夹具留下死 UB allocation，旧 codegen 因缺 address_map 拒绝；该失败
作为诊断保留。专用 serial zero baseline 在前缀用 fill(3)、fill(4)、add、copy 产生7；
constraint 仍保留独立 anchor 和原零次 candidate。两者有效输出/未写 canary 相同，
元数据明确源程序差异；不通过共享候选 UB 弱化新模式的 loop-private 门禁。

### 11.4 最终 CPU 验证

当前构建证据为 `/tmp/ascend-migration-s5/native-build.json`，project source SHA256
`e125218f60dc09de220c5a3041c0854575ef6aa520d8bfc702cc9eb529f127c1`，
module SHA256 `118b8e2f4c4dcc17c3e71901ea611c8d7cd221a8a6407f5e8ae92ac37364347f`。
这是 guarded 仿射边界精确归一化后的构建：旧 TVM 未能从联合 guard 证明展开的字节
上界，修复利用精确整数余数归一化等价比较，仍要求 CanProve。9×1 合法域完整 lowering
及去 guard/真实越界拒绝3项已通过。

| 已执行检查 | 结果 | 证据 |
| --- | --- | --- |
| S5 原生 IR/最终同步/mask/成本协议/快照/cache/最终 trace | 234 PASS，0 skip | versions-ir.json |
| S1/S2 兼容 | 170 PASS，0 skip | legacy-s1-s2-ir.json |
| S3 loop 兼容 | 117 PASS，0 skip | legacy-s3-loop-ir.json |
| S1–S5 入口 host 回归 | 全量132 PASS；末次受影响18项复验PASS，去重133项 | cpu-validation-summary.json；两次实际命令/XML与节点均保留 |
| S3 domain 兼容 | 29 PASS，0 skip | legacy-s3-domain.json |
| S4 pipeline/sync 兼容 | 45 + 50 PASS，0 skip | legacy-s4-pipeline-ir.json、legacy-s4-sync.json |
| 完整真实成本模型预检 | 24 PASS | real-profile-ir.json；实际版本、完整lowering与最终GM写域 |

以上 JSON 位于 `/tmp/ascend-migration-s5/`。原生项含明确标记的 synthetic 成本协议
fixture，不构成硬件校准结果。真实成本矩阵与设备验收的独立证据见11.2、11.6。
旧523构建的12份正式报告/日志原字节保存在
`diagnostic-523/`，archive-index记录哈希；旧 manifest 身份也保留在旧报告内。
本地临时日志/JSON/二进制不提交。

`cpu-validation-summary.json` 汇总全部实际公开入口命令和节点。clang-format15检查7个
整文件及 ir.cc/inject_pipeline.cc 的改动范围通过；全部变更 Python 的 ruff检查通过。

真实模型预检中，steady1/2 分别物化1/2版本；steady3 的 manual(0,0,3,3) 加显式 offset
物化 A/B/C 各3份、每个缓冲区768B。trip1/2 的 constraint 夹具也实际物化3份；zero
为无 remap 的 no-op。这些是注入/分配证据，设备结果另行统计。

### 11.5 复核入口

使用既有 venv（含 Z3 4.15.4.0）与 CANN 环境；实际库加载路径由 manifest 预检核对。
以下入口均只接受 A3：

```bash
source set_env.sh
migration_python=/tmp/ascend-migration-s2/venv/bin/python
"$migration_python" scripts/ascend_migration/s5_check_versions_ir.py \
  --platform a3 --build-manifest /tmp/ascend-migration-s5/native-build.json \
  --output /tmp/ascend-migration-s5/review-versions-ir.json
"$migration_python" scripts/ascend_migration/s5_run_pipeline_npu.py \
  --platform a3 --device 0 --repeat 100 --timeout 180 \
  --build-manifest /tmp/ascend-migration-s5/native-build.json \
  --cost-profile /tmp/ascend-migration-s5/a3-effective-cost-model.json \
  --output /tmp/ascend-migration-s5/review-pipeline-npu.json
"$migration_python" scripts/ascend_migration/s5_run_persistent_pipeline_npu.py \
  --platform a3 --device 0 --repeat 100 --timeout 180 \
  --build-manifest /tmp/ascend-migration-s5/native-build.json \
  --cost-profile /tmp/ascend-migration-s5/a3-effective-cost-model.json \
  --output /tmp/ascend-migration-s5/review-persistent-npu.json
```

若需重新采集完整成本矩阵，可使用当前匹配构建或显式 `--calibration-source-root`
指定独立且经 manifest 验证的快照；该模式无需已有模型：

```bash
"$migration_python" scripts/ascend_migration/s5_run_pipeline_npu.py \
  --platform a3 --device 0 --timeout 180 --calibration-timeout 180 \
  --build-manifest /tmp/ascend-migration-s5/native-build.json \
  --calibrate-profile /tmp/ascend-migration-s5/review-cost-model.json \
  --output /tmp/ascend-migration-s5/review-calibration.json
```

### 11.6 A3 设备与性能结果

普通流水正式父入口 `pipeline-npu.json` 已 PASS（进程退出0）：18/18 用例、
18次真实 bisheng 编译、1800次同 kernel 调用及1800次逐轮 golden。8组实际输出
pair 的输入/golden/输出摘要一致；两项既有 L1 tail/parallel 各100轮也通过。
真实配置、kernel源码、实际 remap/Allocate 与加载库身份保存在逐用例证据。

Persistent 正式父入口 `persistent-npu.json` 已 PASS（进程退出0）：8/8 用例、8次真实
bisheng 编译、800次同 kernel 调用及800次逐轮 golden。single、partial f16、3×10
group tail 和修复后的9×1多 wave 两侧均通过；4组输入/golden/输出摘要一致，实际最终
GM store trace 的 valid tile exact-once 与 NaN canary 同时验证。

两入口总计26个数值用例、26次真实编译、2600次调用/逐轮 golden。另8个性能 worker
各独立编译，3 rounds ×（5预热+20测量）=75次应用调用，合计600次调用、480个
measured samples、24次 post-group golden；不把性能调用当作逐轮数值断言。
正式执行使用 device0、seed0、repeat100、timeout120 与本节 manifest/model，两个
父入口严格串行。`device-validation-summary.json` 汇总两父报告与逐 worker 证据。
两份正式preflight均记录torch 2.7.1+cpu、torch_npu 2.7.1.post4、TVM 0.17.dev0，
按实际报告记录环境版本。

| 夹具 | legacy median ns（95% CI） | constraint median ns（95% CI） | 观测 legacy/constraint |
| --- | --- | --- | --- |
| 普通 steady2 | 5901（5281–6781） | 5651（5401–5971） | 1.04423996 |
| 普通 tail_f16 | 4671（4141–5661） | 4431（4081–4821） | 1.05416385 |
| Persistent group_tail | 5781（5481–5911） | 6681（6561–6901） | 0.86528963 |
| Persistent partial_f16 | 4771（4691–4831） | 4761（4701–4971） | 1.00210040 |

性能来自 `msprof op` BasicInfo-only Task Duration；两侧输入/输出摘要、SoC、flags与
1650MHz频率一致。每策略独立冷编译，输入以同政策预热，不做 timed cache flush。
CI 为各策略 median 的 hierarchical bootstrap 区间；表中比值是观测 median 比值。
group_tail 的观测耗时增加约15.6%，作为本阶段已知性能退化保留；其余三组的区间有
重叠，不据此宣称稳健提速。性能收益不预设的验收约定保持，数值/覆盖/同步正确性与
性能观测分别记录。性能 helper 源SHA记录在父/子证据并核对一致。

### 11.7 交付停止点

本次交付仅包含 S5 实现、永久测试、三个公开入口及文档；legacy 仍为默认。
完整校准模型、原始采样、manifest、JSON和二进制保留于 `/tmp/ascend-migration-s5/`，
通过本节命令可复核。提交推送至 `erhsh/codex/ascend-a2a3-scheduler-migration` 后
停止等待用户审核；该S5交付停止点之后，用户已授权S6，结果见第12节。

## 12. S6 实施与验收记录

S6实现、正式原生/兼容及A3 reuse验收已完成。本节区分静态证明、数值容差与字节
hash证据。范围仅A3，未运行A2专项测试，交付后等待审核，不启动S7。

### 12.1 实现合同

- constraint 在 early/late 两处幂等规范化 storage/view/known pointer alias，再运行
  既有 planner、一次旧 injector 和一套旧同步器。legacy 默认与生成算法保持。
- owner 按 dataVar/allocation site 及已证明 alias 归并，同名异 Var 不混淆；dtype/
  offset view 共用 actual N 和 byte pitch。scope 推断与改写同步维护 Buffer 缓存、
  Block reads/writes/match/init、address/layout keys 及函数元数据。
- schema3 在 Solve/Inject 前逐 byte segment 证明 source 完整定义。task 先读后写；
  N1 支持 preinit readonly/carry、partial-write union、guard/zero、nested/连续serial
  activation。fresh allocation 不借旧 epoch，内部 workspace 不借 caller-entry。
- 最终从 source_function 重建 graph/catalog/readiness，忽略 stored PASS/cases/
  producer 的证明效力，并校验 stored catalog 自洽。实际最近 physical writer 必须匹配
  source producer，再复用旧事件 closure 校验生产、全部消费者、下一写完成关系及
  未消费 event token 的退出 drain；不额外声称证明所有无后继末尾 call 完成。
- 同 task 的 R/W 仅允许完全同区间或不相交；shifted partial overlap 无已审计的
  multi-repeat 快照保证，source/final 均拒绝。exact in-place 和跨 task union 保留。
- 原始 byte/index 算术在 Analyzer 简化、常量替换前逐步 checked int64；已实证的
  singleton `i=2**62, offset=i*64` 与静态 alias byte-scale 回绕均被拒绝。物理声明
  乘积也先 checked，显式只读 API 不报告回绕后的假容量。
- 已审计单行 DMA 仍要求准确 template/count、32B 对齐、无动态 padding 与完整界限；
  未发出的末尾 view metadata 可大于实际 count，不将其误作 helper 执行长度。
- annotate_address 保留 legacy 字节 hint/向上32B语义，允许已证明的生命周期复用；
  地址预算按 actual N×pitch/padding 核对。本步不新增 exact-placement API。
- 内部 constraint compiler ABI3 进入真实 memory/disk key，隔离同 HEAD dirty 升级的
  schema2 旧产物；legacy key 格式不变。无新 PassConfig 开关或 ambient context 行为。

公共只读 API 为 AnalyzeAscendStorageOwners、AnalyzeAscendLifetimes，输入 IR 不变；
三入口为 s6_check_owner_ir.py、s6_check_lifecycle_guards.py、s6_run_reuse_npu.py。
private N>1 仍限连续1D/1×L、32B版本对齐、每个 read footprint 在该迭代完整定义且无
carry。nested positive pipeline、一般动态整数/while、opaque effects仍拒绝；有限bool
case仅证明编译器控制路径，不推导设备 scalar ABI。完整设计见技术参考第16节。

### 12.2 构建身份与 CPU 验证

`/tmp/ascend-migration-s6/native-build.json` 已 write/verify：163个构建输入，source
SHA256 `180efb03b144476aeecf0fcca2680fff3fef8e0aabd310a3b3c833b48f9e808e`，
module SHA256 `8c8101c48ee81bd01cb7444eda6b2fc616b8d5129cb71dd78a353af76160c570`。
正式入口使用此冻结源码/库，不把早期开发库通过当作本步最终证据。

| 正式入口/检查 | 已实际执行结果 | 证据（相对 /tmp/ascend-migration-s6/） |
| --- | --- | --- |
| s6_check_owner_ir | PASS 61，0skip，含实际owner与cache隔离 | owner-ir.json |
| s6_check_lifecycle_guards | PASS 49，0skip：source31 + final18 | lifecycle-guards.json |
| S1 native | PASS 104，0skip | cpu-validation-summary.json关联实际公开入口报告 |
| S2 native | PASS 170，0skip | 同上，保留原负例及明确诊断 |
| S3 loop/domain | PASS 129 + 29，0skip | 含新增S6私有metadata保留键 |
| S4 pipeline/sync | PASS 45 + 50，0skip | 合法旧address hint及错误物理覆盖等正反例 |
| S5 versions | PASS 235，0skip | 保留实际多版本/guard/mask/legacy合同回归 |
| 六阶段 CLI host | PASS 154，0skip | 完整节点/命令/XML记在CPU汇总 |

`cpu-validation-summary.json` 汇总全部实际命令、必跑节点、source/库身份与格式检查，
不把有交集的入口数目相加冒充唯一测试数。新增/owned8个C++/header整文件clang-format
通过；ir/infer/inject按改动hunks检查，旧3个测试ruff/format通过。实际诊断中旧negative
的拒绝点可能因
新边界检查前移，仅更新明确错误类型的 regex，原输入与 require-raises 保留；原失败
报告/日志作为诊断保留，不冒充正式通过。

### 12.3 A3 对照与压力方案

基础为 readonly f16/f32、carry0/1/5、offset views、连续与nested循环八类，逐类配对
legacy/constraint。每个 case 仅真实编译一次，同 kernel 重复运行，每轮确定性刷新
finite Scratch/A、tile/launch 标签并记录 input/output/golden SHA、逐轮数值与canary。

legacy CPU预检确认 readonly f16/f32、alias_views 的旧 positive pipeline 因同owner
连续写拒绝；dtype_views/private dtype也受此门禁。仅这些 legacy case 改 serial0
baseline，其余已通过旧规划的 carry/连续/nested保留stage1。constraint基础stage1、
private dtype stage3不变；报告逐case记录双方directive/role，不宣称指令参数相同。
旧模式拒绝发生于0次bisheng/0次launch，日志保留；不修改旧算法以支持新增dtype流水。

两个 dtype 候选以 f32[32]/f16[64] 的同128B storage构造有限half bit pairs与零容差
golden。N1与独立anchor的private actual≥2分别检查；launch前按source task dataVar/
source_data关联actual remap/pitch/Allocate，不依赖u/v名称或requested版本。候选先做
内部1轮真实编译/数值pilot，再决定纳入公开必跑集合。四项pilot已实际PASS：每项1次
bisheng、1次launch/golden，输出SHA均等于golden；private constraint实际owner id4、
f32/f16同一128B storage、actualN3、pitch128、Allocate384B。owner id仅是本次报告值，
验证依据为Var/provenance和字节关系，不把id作为新API常量。

两种dtype已纳入公开必跑矩阵，10类fixture×legacy/constraint为20case；private
tiles1，其余tiles3，每项100轮，正式父入口已PASS。pilot不叠加为正式压力次数。

### 12.4 正式设备结果与复核

`reuse-npu.json` 正式PASS，进程退出0；`device-validation-summary.json` 独立核对
全部worker、编译命令、逐轮摘要、canary和owner证据：

| 项目 | 实际结果 |
| --- | --- |
| 数值用例/真实bisheng编译 | 20/20 PASS，各1次编译 |
| 同kernel attempted/completed/golden | 2000 / 2000 / 2000 |
| 每轮input/output/golden摘要记录 | 2000组；各输入每case均100个不同hash |
| 两策略逐轮input/output/golden字段比较 | 1000/1000组PASS |
| 未触及区canary检查 | 313600个元素通过 |
| private mixed dtype | 实际N3、两view同128B owner/pitch、Allocate384B |
| 独立pilot | 4次编译/4次launch/golden，单独统计 |

golden使用 `assert_close(equal_nan=True)`：普通f32的rtol/atol为1e-5，f16为1e-2，
dtype reinterpret变体为0。整输出与golden字节hash相等为1800/2000；f16两项共200轮
不相等，字节hash包含NaN表示，不能把数值容差通过写成全字节相等。两策略的每轮
input/output/golden字段仍严格一致1000/1000。fixed-storage路径以实际空remap及
source contract/Allocate记录，不将requested1伪造为显式物化版本证据。

实际device0为Ascend910_9362，torch 2.7.1+cpu、torch_npu 2.7.1.post4、
TVM 0.17.dev0/c2921fd；CANN9.1.0与bisheng SHA沿S5。本步复用S5完整真实模型，
canonical SHA `f2aa7ecfa4e592199c81801140903352f6e34fc1cee7be220987d6c5753a13fd`，
不重校准、不采性能。本任务worker/pilot/profiler均已退出；其他仓库的无关任务不处理，
不将本任务清理报告扩称为全机持续空闲。

实际设备命令使用seed0、repeat100、timeout120；原生与设备复核入口如下：

```bash
source set_env.sh
migration_python=/tmp/ascend-migration-s2/venv/bin/python
"$migration_python" scripts/ascend_migration/s6_check_owner_ir.py \
  --platform a3 --build-manifest /tmp/ascend-migration-s6/native-build.json \
  --output /tmp/ascend-migration-s6/review-owner-ir.json
"$migration_python" scripts/ascend_migration/s6_check_lifecycle_guards.py \
  --platform a3 --build-manifest /tmp/ascend-migration-s6/native-build.json \
  --output /tmp/ascend-migration-s6/review-lifecycle-guards.json
"$migration_python" scripts/ascend_migration/s6_run_reuse_npu.py \
  --platform a3 --device 0 --seed 0 --repeat 100 --timeout 120 \
  --build-manifest /tmp/ascend-migration-s6/native-build.json \
  --cost-profile /tmp/ascend-migration-s5/a3-effective-cost-model.json \
  --output /tmp/ascend-migration-s6/review-reuse-npu.json
```

### 12.5 交付停止点

本次仅提交S6实现、永久测试、三个入口及相关文档，保留legacy默认；临时模型、
manifest、原始JSON/日志与二进制保留在 `/tmp/ascend-migration-s6/` 或既有S5目录。
推送至 `erhsh/codex/ascend-a2a3-scheduler-migration` 后停止，等待用户审核。
S6交付时S7新同步生成器尚未获得执行授权；当前授权与进展见第3、13节。
S5已记录的性能退化不在S6调优范围。

## 13. S7 实施记录

### 13.1 本步合同

用户已明确授权S7，仅在A3验收，交付后提交推送并停下等待审核。设计与实施骨架
分别见技术参考第17节、pass-impl-skeleton的S7节。固定源码/主仓参考与之前步骤一致。

新增同步选择 `tl.ascend_sync_strategy="a3"`，默认仍legacy。constraint+a3负责已审计
Vector核内依赖；legacy planner+a3负责固定explicit-C/V的GM workspace ready/release，
保留旧核内/VS保护，不接管S8混合流水与排程。新缓存同步ABI只进入a3策略产物。
核内池为每个合法Vector实例/有向PIPE域8个ID；跨核mode2用户池0–10，系统/工具链
保留11–15，硬件计数器深度15。事件预算与能力来源见技术参考，不按位宽推可用池。

核内生成器以最终MP地址/大小重建物理RAW/WAR/WAW，保持现有任务顺序。每个跨PIPE
Set→Wait后加真实PIPE_ALL fence；独立final verifier到fence才归还binary flag租约，
不把IR中相邻Wait当作硬件已清零。未匹配Wait的Set不能借ALL清除，单PIPE barrier
不能代替ALL；缺ALL、迟ALL、错误barrier和复用活跃ID均有真实因果负例。
这是保守首片，不承诺吞吐提升；S5成本不重校准，既有性能退化不在本步调优。

跨核首片为真实cv_1_2、单GM channel、N1固定物理slot、静态serial FIFO。Producer
写后在MTE3/FIX通知READY0，立即等待RELEASE1；consumer等READY、在最后GM读的MTE2
完成后释放，两个V必须完整参与。无初始虚假token，零轮不发协议事件，末轮自然drain。
源码和最终IR分别重建actor/group/epoch/字节区间与实际通知；不信plan/PASS字段。
只读共享与不相交输出允许，其余跨group/非channel跨actor写冲突明确拒绝。
Expert默认映射只在精确two-AIV轴证实时规范化显式cv_1_2，最终仍严格核映射；
不同迭代坐标/参与guard、positive mixed pipeline、手写cross协议和未知隐式flag使用拒绝。
非空handoff要求`tir.noalias=True`，caller须保证不同GM参数实际不别名。

四种GM copy helper的完整template/dtype与runtime几何均已核对。Vector GM↔UB搬运
保守要求rows<=4095，符合本机DataCopyPad blockCount域；UBGM某些flat分支虽可用单
block，首片也不额外放宽。Cube GM2L1/FIX按各自审计条件，不将此数冒充全部引擎上限。
4095/4096正负例均来自正常T.copy lowering，UB声明预算均约139KB合法，正例核对
actual size_map/Allocate>=131040B，明确区分内存容量与descriptor合法性。
内部auto_gm_indices的ready与未知entry输入另做native
正负例；正式设备矩阵使用显式caller workspace，不冒称隐藏workspace自动生成路径已验收。

### 13.2 最终构建身份

`/tmp/ascend-migration-s7/native-build.json` 记录并校验168个native构建输入：
source SHA256 `c1d82dcb61aa370dd6ac3e4dabdb2c42147d42249199cdfc9b19924599f50d09`；
module SHA256 `a01644f4182f94496efb7e2b5c892bd1cf90b7e720662ff4893013c06fac279a`。
TVM/runtime SHA仍为`79912ad7…`/`a98401d8…`，TVM固定基线与两项patch保持。
225个Python输入冻结SHA256为
`215c850a3e8aed86257024390f897a275a38dcebd964bd70f1c9aaff73db80a2`。
正式native/设备均匹配上述构建和Python输入。
报告保留实施起点`e34c8891`上的dirty工作区身份；最终提交与验收的关联以已核验的
源码、Python输入及实际加载库为准，不把报告中的旧HEAD当作最终S7提交号。

硬件原文、SDK/helper实际文件路径和SHA留在`hardware-evidence/`；前期失败日志和
`diagnostic-89c94b6d/`、`diagnostic-fc5eb220/`保留，不当作最终验收。
fc5/2dde构建的三项正常形状A3 pilot已实际3compile/3launch/3golden，输出摘要均等于
golden、147968个canary元素检查通过；均single stream，且不叠加最终压力次数。

### 13.3 正式设备矩阵

20个worker使用同kernel逐轮刷新输入/标签和workspace scratch：Vector 0/1/3短循环，
257轮的实际N1/2/3与1/8 workers；GM v2c/c2v为17轮×4组单slot；双stream分别使用
513轮Vector×8 workers和17轮GM×1组。16普通case各100次，4双stream case每轮
两stream×两独立buffer batch、各400次。正式20/20项全部PASS，20次真实冷编译、
3200次attempted/completed/golden，0fail/skip。

最终报告`sync-stress-npu-aggregate.json`是两批正式证据的只读汇总，不代表额外设备
执行：前17项来自`sync-stress-before-trace-timeout.json`的真实PASS行（120秒预算），
后三项来自`formal-last-three.json`（300秒预算）。两批源码/库/225个Python输入一致，
20项case JSON保持原字节，原TIMEOUT报告不改写；公开`compare_pairs`完整复核通过。
最终汇总SHA256为`3d3438d115c456fda35f60b474100b82be254ddcb8efb4054a1fa5a603953c36`。

`root-device-audit.json`另从原始汇总逐项核对：1600条严格跨策略记录、400个双门
pending窗口及其真实先后顺序均通过；3200次完整输出摘要和1200次workspace摘要均
等于各自golden，44902400个未写canary元素通过检查。该只读审计不增加设备次数。
14个Vector worker实际flags为`[-O2]`，与S5成本模型匹配；6个GM worker沿用legacy
planner，实际flags为`[-O3, --cce-auto-sync=off]`，不应用S5 constraint成本。各策略
配对的实际flags一致，不将GM握手验证表述为成本校准或性能测量。
设备为逻辑0、实际SoC`Ascend910_9362`，torch2.7.1+cpu、torch_npu2.7.1.post4、
CANN9.1.0；真实Bisheng SHA仍为`7ffc97205a0582ce00a9ec163530696bf3971b24669c43c1c02cc9c7b56f541f`。

Vector配对旧/新自动核内生成器；GM配对已证明正确的手写legacy mode2双向握手与
新生成器。两者GM的旧auto-cross均关闭，避免将缺release的旧单向协议发设备。
native另以小写workspace名先证旧auto-cross确会插通知，再证新owner启用时不会叠加。
实际A/B/O/workspace地址独立。双stream各使用一个独立`ExternalEvent`启动门，先wait，
再提交原目标batch与各自done event；两侧host调用均返回、done/stream查询均未完成后，
第三普通stream分别record释放两门，再同步目标并检查golden。每门每轮只一次wait，
前轮目标完成后复用；本机ValueWait后的ValueWrite0自动复位已按安装版源码核对，
不把noop reset当成关闭门，也不预先record gate。只有lazy done event预record。
此证据范围是start-gated、两目标队列同时已提交且未完成，不主张自然窗口或硬件时间重叠。
两个环境开关`PER_STREAM_QUEUE=0`、`ASCEND_LAUNCH_BLOCKING=0`只在独立双stream worker
导入runtime前设置，不修改共享配置或`TASK_QUEUE_ENABLE`。单stream报告不作并发结论。

本机torch_npu为2.7.1.post4，安装版本声明git提交
`d99d6cda7b9ab4370e7c03f697302760b231538a`。
精确源码、安装文件、CANN真实符号与队列排序证据留在`external-event-audit.json`和
`stream-gate-audit.json`。Cython获取`.npu_stream`先flush普通stream的CPU提交队列，
使Torch的device wait先于direct RT launch入队；flush不等同设备同步。
只使用公开的[ExternalEvent wait-before-record API](https://www.hiascend.com/document/detail/en/Pytorch/2600/apiref/torchnpuCustomapi/docs/en/custom_APIs/torch_npu-npu/torch_npu-npu-ExternalEvent%28%29.record%28%29.md)，不新增设备kernel或mode0核间barrier。

### 13.4 正式 CPU 结果

最终构建与Python冻结身份下，两个S7公开入口和受影响的旧阶段回归均实际通过，0skip。
`cpu-validation-summary.json`保留全部真实命令、各入口JSON及JUnit/日志路径；
入口之间可能有交集，不将所有行相加冒充唯一测试数。

| 正式入口/检查 | 实际结果 |
| --- | --- |
| s7_check_event_budget | PASS70：allocator33 + local23 + cache14 |
| s7_check_handoff_ir | PASS60：handoff50 + ownership10 |
| S1 legacy native | PASS104 |
| S2 task IR | PASS170 |
| S3 loop / domain | PASS129 / 29 |
| S4 pipeline / sync | PASS45 / 50 |
| S5 versions | PASS235 |
| S6 owner / lifecycle | PASS61 / 49 |
| S1–S7 CLI host | PASS214 |

16个变更Python文件ruff lint、11个新增Python文件format100、6个新增/owned C++文件
clang-format及git diff--check均通过；其它共享C++按改动hunk检查，不扩面重排旧文件。
独立只读末审核对了默认legacy、Python/C++路由、FFI注册、缓存ABI和保留旧VS保护，
未发现待修源码问题；设备结果单列，不由CPU通过推导。

并发驱动补充后重验两个S7公开入口与全部CLI；214个编译器/旧阶段Python输入及
native构建均字节未变，旧阶段回归保留同一c1d82dcb/a01644f4的实际通过记录。

### 13.5 并发观察的诊断记录

首轮未加启动门的正式执行前16个single-stream worker全部PASS，共1600条已保存
launch/golden记录；`streams_vector_legacy`完成400次调用，但100个查询窗口没有
同时pending，父入口FAIL并停止，后三项未执行。该失败回调未return，原始报告没有
golden_checks/rounds，不能把推断的断言经过补写成400条已保存golden证据。
原始JSON/日志保留为`sync-stress-before-pending-depth.*`。

驱动改为同一个live metadata对象交给既有run_worker catch，异常时保留已采集的
handles、窗口与逐轮记录，不改共享runner。随后原fixture repeat3实际诊断保存12条
golden/hash，三次窗口均为stream complete=[true,false]。诊断沿用原core/batch/trip，
pending门禁保留。普通query会先flush hostqueue，短kernel的自然待完成窗口不能仅靠
顺序提交保证，因此用标准双门控制提交窗口。

最终双门方案的single-round设备验证：Vector legacy、GM legacy/a3各1次真实编译，
各4次目标调用/golden，两个done/stream在释放前均false；释放与post-sync顺序、
逐call摘要、canary均已独立核对。共3compile/12launch不叠正式；上述早期三项正常
形状pilot也另列。

加门后的正式执行前17项PASS并保存2000条launch/golden，第18项
`streams_vector_a3`在120.415秒总进程预算处TIMEOUT/124，后两项未执行。
18项都已确认真实Bisheng编译；日志已越过最终IR证据采集，进入输入buffer分配。
worker只在callback返回或异常时落盘，超时报告中的0次快照不能证明实际零调用，
也不能据此判为设备死锁。该批原始证据保留为
`sync-stress-before-trace-timeout.*`及`worker-timeout-summary.json`。

保持原用例与120秒限制的临时包装诊断另保留为`phase-diagnostic.json`：runtime
准备10.24秒结束，compile到104.07秒才返回，sync证据采集到106.09秒，最终输出
trace到116.43秒，首次目标提交在116.58秒。随后submit与golden检查持续前进。
30/60/90秒堆栈及`phase-observer.json`的CPU状态支持编译/检查耗时判断；
诊断不计正式launch/golden，也不把host时间当作kernel性能。正式补跑总进程预算
使用300秒，涵盖冷编译、证据采集与100轮验证；源码、输入规模、事件协议、
pending/数值门禁不变。该Vector用例原入口实际134.519秒PASS，另外两项GM用例
分别40.376/38.325秒PASS。前17项在完整Python/native冻结身份一致下复用，保留
原始批次、命令和报告SHA；最终公开`compare_pairs`已检查完整20项，结果见13.3节。

### 13.6 复核入口与交付边界

```bash
source set_env.sh
export TL_ASCEND_MIGRATION_BUILD_MANIFEST=/tmp/ascend-migration-s7/native-build.json

/tmp/ascend-migration-s2/venv/bin/python scripts/ascend_migration/s7_check_event_budget.py --platform a3 --output /tmp/ascend-migration-s7/review-event-budget.json
/tmp/ascend-migration-s2/venv/bin/python scripts/ascend_migration/s7_check_handoff_ir.py --platform a3 --output /tmp/ascend-migration-s7/review-handoff-ir.json
/tmp/ascend-migration-s2/venv/bin/python scripts/ascend_migration/s7_run_sync_stress_npu.py --platform a3 --device 0 --seed 0 --repeat 100 --timeout 300 --cost-profile /tmp/ascend-migration-s5/a3-effective-cost-model.json --output /tmp/ascend-migration-s7/review-sync-stress.json
```

manifest须由当前源码显式构建并核验，方法沿用技术参考第11.5节；上述本机路径不是
仓库自带产物，入口不会自动编译或安装依赖。压力入口每case独立进程与缓存目录，
真实Bisheng记录、实际配置/库、最终IR、逐次输入/输出/workspace摘要、NaN canary、
提交/查询/释放/同步顺序均保留在输出及artifacts中。

本步只交付S7实现、永久测试、三个入口和四份文档；不提交JSON、日志、模型和二进制。
S7交付时停止等待审核；用户随后已明确授权S8，当前进展见第3、14节。

## 14. S8 实施记录

用户已明确授权S8。设计与实施骨架已补入技术参考第18节及对应S8骨架；新后端已
构建，原生74、CLI283及旧阶段兼容均已实际通过。A3正式正确性14用例/1400次、
benchmark六用例/480次均已通过；实际性能及编译回归见下文。
本步只做A3专项验收，保留legacy默认与A2兼容代码。

### 14.1 范围与构建身份

新增`planner="auto"`与`sync="a3"`的整核接管分支，任务规划、新版本物化和同步
均由新后端负责，旧planner/injector/local/VS/cross链不在该分支运行。legacy与
constraint bridge保留。首片为有限静态域，最多256任务/65536控制访问/62个数字
资源域；UB/L1可用N1/N2，单GM channel保持N1与caller可见ABI/终值。
GEMM准入为f16输入/f32累加、clear=true完整对齐复合调用；归约为已审计的
clear=true f32行归约。scratch必须具有正确UB/F32指针ABI、足够几何容量且不与
本call payload重叠。未知调用、未审计状态、动态while及多channel等明确拒绝。
`abstract_issue_step_v1`只决定合法有限次序，不是硬件ns；不读取S5成本profile。

最终候选构建记录为`/tmp/ascend-migration-s8/native-build.json`：174个native输入，
source SHA256 `257659998dc6656fe5456e2c4207e4eb4da00465284f7879fa723d49032a2dec`，
module SHA256 `4771eeefbf6ff07edc1cd17d94f6ea834e3e553f94d6162455f56a4d1de84d98`。
TVM/runtime仍为`79912ad7…`/`a98401d8…`，固定TVM源码与两项patch不变。
237个Python输入的最终冻结SHA256为
`8657121883e9f25bf2398df5e41a31e821a6a2fefadf9adab949a9fcfe603f76`。
主机、旧阶段回归与正式正确性运行使用此前冻结`9c5ca67d…`；设备进程全退出后仅
`test_s8_sync_contract.py`调整排版，AST相同，其余236个Python文件逐字节未变。
`python-format-provenance-bridge.json`保留前后完整清单和SHA，原报告不改写。
报告记录S7 HEAD `59f3b220`上的dirty工作区；最终提交与验收通过已核验的源码/
Python/实际加载库关联，不把旧HEAD当作最终S8提交。

设备逻辑0、实际SoC为`Ascend910_9362`；Python3.12.9、torch2.7.1+cpu、
torch_npu2.7.1.post4、CANN9.1.0。实际Bisheng仍为
`/home/developer/Ascend/cann-9.1.0/tools/bisheng_compiler/bin/bisheng`，SHA256
`7ffc97205a0582ce00a9ec163530696bf3971b24669c43c1c02cc9c7b56f541f`。
普通配对显式flags为`[--cce-auto-sync=off, -O3]`，实际解析为
`[-O2, --cce-auto-sync=off, -O3]`；真实有序argv逐对相同，仅排除唯一cpp输入与
`-o`输出临时路径。两独立Expert沿用原flags，不拿它们作auto性能配对。

### 14.2 正式 CPU 验证

| 检查 | 实际结果 | 证据（相对`/tmp/ascend-migration-s8/`） |
| --- | --- | --- |
| s8_check_integration_ir | PASS74：integration33、cache7、source/final/protocol/mask fault34；0skip | `integration-ir.json` |
| S1–S8主机CLI与路由/计时政策 | PASS283，0skip；执行前后237个Python输入相同 | `host-cli.json`、`formal-host-cli/pytest.xml` |
| S1/S2 legacy/task native | PASS104/170 | `compat/s1_check_legacy_lowering.json`、`compat/s2_check_task_ir.json` |
| S3 loop/domain | PASS129/29 | `compat/s3_check_loop_ir.json`、`compat/s3_check_persistent_domain.json` |
| S4 pipeline/sync | PASS45/50 | `compat/s4_check_pipeline_ir.json`、`compat/s4_check_sync_coverage.json` |
| S5 versions | PASS235 | `compat/s5_check_versions_ir.json` |
| S6 owner/lifecycle | PASS61/49 | `compat/s6_check_owner_ir.json`、`compat/s6_check_lifecycle_guards.json` |
| S7 event/handoff | PASS70/60 | `compat/s7_check_event_budget.json`、`compat/s7_check_handoff_ir.json` |

兼容入口完整命令与结果保留在`compatibility-summary.json`，两进程只读并行。
上述入口之间有交集，不把每行数字相加当作唯一测试总数。
测试排版调整后，公开S8入口按最终`86571218…`冻结身份再次实际74/74 PASS，
0skip，运行前后输入核验相同；最终报告SHA256为
`86d80bd3c06e2f109497c6314a857e281c17c3af3cff9d05b02eb37e31159a26`。
旧报告保存在`integration-ir-before-format.json`；主机和旧阶段源码均未变，
保留其原始通过记录，`cpu-validation-summary.json`明确列出两次冻结范围。

### 14.3 A3 矩阵与前置 pilot

正确性必跑14个worker：GEMM single(64×64×32,1core,1trip)与batch3
(128×64×32,4cores,3trips)、reduce single/batch3、mixed_add(1core,3trips)、
mixed_reduce(4cores,3trips)，各legacy/auto配对；另有独立原Expert C128精度与
L1 K=200/stages2回归。正式repeat100已完成14次真实编译、1400次target调用/golden，
六组600条跨策略记录。普通12case逐次换输入；Expert沿原helper固定输入并逐轮
重置输出/检查，不把它们表述为逐轮生成新随机输入。

修复后同snapshot的3pair/6pilot已实际全部PASS：每case各1次真实Bisheng编译、
1次target调用/golden，3pair输入/输出/golden/workspace及真实有序编译选项一致，
83072个canary元素检查通过。auto实际N2/actor内重排均按IR门禁验证：GEMM有L1N2，
reduce有UBN2，mixed_reduce的UB均N1，不从请求数推断所有buffer都是N2。
证据为`pilots-int64-fixed.json`及独立摘要；6compile/6call不叠加正式次数。

正式`workloads-npu.json`为14/14 PASS、0fail/skip，SHA256为
`d8a6bfc7ee61422f25cde79cc5e0d47aab101a0048e2255956170444c80737f5`。
`root-workloads-audit.json`另从原始报告复核14次编译、1400次完整调用/golden及
600轮严格配对；普通用例保存1200条逐轮记录，全部输出摘要等于golden，400条
workspace摘要等于golden，12816000个未写canary元素通过检查。实际typed访问、
allocation/size-map边界与要求的N2/重排均复核通过。Expert200次沿用原helper断言，
不补造逐轮摘要。该审计不代表额外设备执行。

性能六worker：gemm_batch3、reduce_batch3、mixed_reduce各legacy/auto；每case
20个timed rounds，每round3次warmup加1次timed，全部reset/golden在计时区外。
实际6compile、480call/golden，其中360warmup、120timed samples；三组240条完整
逐调用配对。编译wall/CPU及Bisheng、host dispatch wall、NPU Event span分别记录；
不相减为“纯指令延迟”，不预设提速，所有回归如实列出。

`benchmark-npu.json`为6/6 PASS、0fail/skip，SHA256为
`92511902feacc69f8043f4d0fd686ffbfacfa787c2a33e637ba2ecdfd61165f6`。
`root-benchmark-audit.json`独立复算raw samples的中位数与bootstrap区间，复核
全部warmup/测量先后、同输入/输出/workspace、真实编译flags、保留cpp/kernel库
SHA及6645760个canary元素；不代表额外设备执行。mixed_reduce两策略计时pilot
另为2compile/8call/golden/2samples，不叠加正式次数。

下表单位为微秒，括号为中位数95% bootstrap区间（1000次resample，seed0）；
变化为auto相对legacy，正数为回归。每worker20个实测样本，Event包含主机提交
间隙，不能由这些数值推导纯kernel指令延迟或普遍提速。mixed的legacy为完整
READY/FREE手写握手，auto为新生成器；两者实际编译选项与全部计算数据相同。

| 用例 | 指标 | legacy 中位数（95%区间） | auto 中位数（95%区间） | 变化 |
| --- | --- | --- | --- | --- |
| gemm_batch3 | host dispatch | 248.2860 (237.9820–261.3220) | 242.7770 (236.8315–254.3220) | -2.22% |
| gemm_batch3 | NPU Event span | 251.1405 (235.4800–272.1810) | 246.2605 (241.0100–253.5410) | -1.94% |
| reduce_batch3 | host dispatch | 206.8460 (206.1310–207.4670) | 199.9915 (199.4660–200.8170) | -3.31% |
| reduce_batch3 | NPU Event span | 215.5600 (215.1305–216.2400) | 206.5605 (205.7905–207.1000) | -4.17% |
| mixed_reduce | host dispatch | 246.5120 (243.3820–259.2820) | 255.6915 (249.1465–273.4670) | +3.72% |
| mixed_reduce | NPU Event span | 250.3205 (246.3200–260.1110) | 256.5210 (248.4405–274.1205) | +2.48% |

每case仅一次真实冷编译，下表单位为秒；wall包含前端/TVM/Bisheng/adapter，
不含后置证据序列化或数据准备；parent process与回收child CPU分别列出。

| 用例 | 策略 | 编译wall | parent CPU | child CPU | Bisheng wall |
| --- | --- | --- | --- | --- | --- |
| gemm_batch3 | legacy | 4.741 | 0.119 | 4.742 | 4.712 |
| gemm_batch3 | auto | 19.247 | 14.688 | 4.867 | 4.836 |
| reduce_batch3 | legacy | 4.566 | 0.114 | 4.566 | 4.540 |
| reduce_batch3 | auto | 5.351 | 0.841 | 4.644 | 4.611 |
| mixed_reduce | legacy | 5.036 | 0.140 | 5.008 | 4.992 |
| mixed_reduce | auto | 53.770 | 49.156 | 5.615 | 5.593 |

auto编译wall分别为legacy的约4.06/1.17/10.68倍，当前有限DAG求解/证明路径的
编译成本明显增加；这是已知交付限制。本步采用每任务前后ALL保护审计helper
内部借用资源，不将abstract issue次序表述为真实硬件重叠证明，也不将三组观测
当作成本校准。计时期间未并行运行本任务的CPU检查或其它NPU任务。

### 14.4 实施中发现的合同缺口

早期构建与native诊断依次修复了当前TVM的Optional/API适配、实际已flatten的L1
形状、归约U8 backing上的F32 scratch指针及真实几何容量/UB归属/不重叠要求、
caller参数顺序/dtype/容量ABI，以及空任务与非空任务的launch axis区别。
对应负例进入永久native测试；前期失败报告留在`diagnostic-37b2f3c0/`、
`diagnostic-b9a19f12/`与`.diagnostics/`，不计入最终通过次数。

首次设备pilot前五项PASS，第六项四core mixed_reduce_auto在Bisheng之前因
`resource_flags[92]`失败，故该worker为零次编译/调用。根因是当前TVM的
`Integer(int)`将64位resource mask隐式收窄至32位，并非dense域编号溢出。
调度记录、物化形状字段和verifier Key改为显式Int64 IntImm；保留原四core/
三trip/120task用例及dense映射，新增bit39高位mask和逐任务资源冲突回归。
`int64-record-repair.json`为当时的修复/构建记录，后续通过另见正式native、pilot
及workload报告；旧`pilots.json`的FAIL保留，不改写为PASS。

17个变更Python的ruff lint、12个新增Python的format100、7个新增/完整维护C++的
clang-format及git diff--check均已实际通过。共享旧C++仅检查变更hunk，不扩面
重排。`format-report.json`、`format-report.md`及修复前报告均在临时证据目录。
`final-routing-review.json`对最终冻结源码的默认策略、旧链隔离、配置/FFI、
缓存ABI和35文件范围进行独立只读复核，未发现待修问题；它不代替测试或设备证据。
`final-acceptance-summary.json`关联原始结果、库哈希与两次Python冻结的有效范围。

### 14.5 复核入口与交付边界

```bash
source set_env.sh
export TL_ASCEND_MIGRATION_BUILD_MANIFEST=/tmp/ascend-migration-s8/native-build.json

/tmp/ascend-migration-s2/venv/bin/python scripts/ascend_migration/s8_check_integration_ir.py --platform a3 --timeout 120 --output /tmp/ascend-migration-s8/review-integration-ir.json
/tmp/ascend-migration-s2/venv/bin/python scripts/ascend_migration/s8_run_workloads_npu.py --platform a3 --device 0 --seed 0 --repeat 100 --timeout 300 --output /tmp/ascend-migration-s8/review-workloads-npu.json
/tmp/ascend-migration-s2/venv/bin/python scripts/ascend_migration/s8_benchmark_npu.py --platform a3 --device 0 --seed 0 --repeat 20 --timeout 300 --output /tmp/ascend-migration-s8/review-benchmark-npu.json
```

manifest须按技术参考第11.5节由当前源码构建并核验；上述临时环境和产物不是仓库
自带依赖，入口不自动安装或重编译。每个NPU case独立进程/缓存，保存真实编译
命令、实际配置/库、最终IR/合同和逐次golden记录；性能入口另保存真实编译输入
及kernel库，计时区内不序列化证据。各入口失败/超时立即停止该组并明确未运行项。

本步只提交S8实现、永久测试、三个公开入口及四份文档，不提交JSON、日志、模型
或二进制。保持legacy默认，不新增A2测试或S5成本校准；推送后停止等待审核。

## 15. S1–S8 整体集成验收

用户已选择整体集成验收。本轮保留生产源码与默认行为，核验 S8 同源原生/CLI/
auto 设备证据，补跑基线、solver、capability 和 S1–S7 全部旧 A3 设备公开入口。
不将历史不同构建设备结果直接提升为当前构建通过，也不重复执行已核验的同源
S8 1880 次设备调用。三项补充主机入口实际 18/105/42 PASS，七个旧阶段 A3 设备
公开入口均完整 PASS、零 fail/skip/未运行：97 个数值 worker、97 次真实编译、
11000 次调用及原容差 golden。S5 自动性能部分另有 8 次编译/600 次调用、480 个
实测样本与 24 次轮末 golden，不冒充逐调用检查。

S8 同源原始证据、各入口独立审计、600 个 BasicInfo CSV、S7 的 1600 轮配对/
400 个双 stream pending 窗口，以及前后源码/库/537 输入身份均已实际复核通过。
数值仍沿用原 dtype 容差；S6 readonly_f16 不宣称逐位等于 golden。性能回归如实
保留：本轮四组 constraint 中位耗时增加 0.64%–15.44%，S8 原有 auto 冷编译与
Event 观测单列。没有改生产源码、默认策略或重新校准模型；最终设备无存活测试
进程。整体验收结论为已交付 A3 合同的集成正确性 PASS，不代表普遍提速或扩大
支持域。

完整身份、复用条件、必跑矩阵、实际结果、性能限制和复核命令见
[acceptance-report.md](acceptance-report.md)。本轮完成后仅提交两份文档并推送，
随后停止等待用户审核；日志、JSON、profile 和二进制不入仓。
