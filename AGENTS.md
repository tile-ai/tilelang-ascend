# AGENTS.md

## 项目概述

本项目是 TileLang-Ascend 算子开发项目，基于 TVM 编译器基础设施，提供 Python DSL 用于开发华为昇腾 NPU 上的高性能 AI 计算 kernel。

### 核心功能

- 使用 Python DSL + `@tilelang.jit` 编写昇腾 NPU 自定义 kernel
- 支持 Developer 模式（自动化）和 Expert 模式（手动控制）两种编程范式
- 提供完整的编译、测试、调试及性能调优工作流

---

## Skills 索引

#### 算子开发与编排
- `tilelang-op-design`：生成算子设计方案（design.md），含三维 Kernel / threads / 动态边界 / L0C 容量 / GEMM 非整除等技术约束检测
- `tilelang-op-generate`：基于 design.md 生成算子实现代码、内嵌 golden 与测试用例
- `tilelang-ascend-tile-api`：新增或封装 `T.tile.xxx` 小 API 时端到端打通前端、lowering / codegen、helper、测试与文档
- `tilelang-expert-to-developer`：Developer / Expert / 混合模式选择、pass_configs 配置与转换指南
- `tilelang-api-best-practices`：TileLang API 速查与最佳实践（Kernel 定义、内存分配、计算原语、调度同步）

#### Pass 分析与设计
- `tilelang-pass-analyzer`：Pass 功能分析、对比、分类查询
- `tilelang-pass-workflow-analyzer`：Pass 工作流分析、执行顺序、依赖关系、新 Pass 定位
- `tilelang-pass-design`：Pass 设计方案与实现模式

#### 调试与错误处理
- `tilelang-debug-helper`：为算子添加 GDB 调试代码，配置 CMakeLists.txt 与 VSCode 联合调试
- `tilelang-error-fixer`：编译 / 运行时错误诊断与修复

#### 性能调优
- `tilelang-perf-optimization`：性能优化方案、最佳实践（Flash Attention / GEMM intrinsic / RoPE）与反模式排查

#### 环境与工具
- `tilelang-env-check`：环境检查与配置验证（CANN、torch_npu、子模块、编译产物、环境变量）
- `tilelang-submodule-pull`：自动拉取代码和子模块
- `tilelang-github-operations`：GitHub PR 创建与操作

#### Skill 管理
- `skill-creator`：创建新 skill
- `skill-journal`：算子开发反馈记录 schema
- `tilelang-skill-review`：聚合 skill-journal 反馈，按命令式 apply / reject 落到对应 SKILL.md
- `tilelang-review-skill`：通用 skill 质量评审

### 算子开发编排体系（OpenCode 多代理）

由 [`@tilelang-op-orchestrator`](.opencode/agents/tilelang-op-orchestrator.md) 作为 Primary 驱动 3 阶段状态机，调度 3 个 Subagent：

- [`@tilelang-op-analyst`](.opencode/agents/tilelang-op-analyst.md) (Stage 1)：调用 `tilelang-op-design` 完成需求理解与设计
- [`@tilelang-op-developer`](.opencode/agents/tilelang-op-developer.md) (Stage 2)：调用 `tilelang-op-generate` 完成代码实现、测试、精度调试（一站式，attempt 上限 5 次）
- [`@tilelang-op-perf-tuner`](.opencode/agents/tilelang-op-perf-tuner.md) (Stage 3，**可选**)：调用 `tilelang-perf-optimization` 完成性能调优

`DESIGN.md` 非硬性约束——Subagent 在实施中发现设计错误时返回 `[DESIGN_ERROR]`，Orchestrator 回退到 Stage 1 重做（不设次数上限）。新建算子直接对 `@tilelang-op-orchestrator` 描述需求；单独使用某个 skill 可走 `/tilelang-op-design`、`/tilelang-op-generate`、`/tilelang-perf-optimization`、`/tilelang-env-check` 跳过编排层。

---

## 通用原则

> **严格遵循以下原则**

1. **如实报告，禁止伪完成**
   - 未验证的结果，不得表述为"已完成"或"已通过"
   - 未实际执行的命令、测试、构建、提交或发布，不得声称已执行
   - 遇到失败、阻塞、权限不足或信息缺失时，必须明确说明，不得伪造过程或结果
2. **先验证，再下结论**
   - 能通过代码、文件、日志、测试或工具直接确认的事项，优先基于证据判断，不以猜测代替验证
   - 若当前环境无法完成验证，必须明确说明验证缺口、已知范围与剩余风险
3. **区分事实、推断与建议**
   - 结论应明确区分"已确认事实"、"基于上下文的推断"、"建议采取的动作"
   - 禁止编造不存在的文件、输出、报错、性能收益、验证状态或用户意图
4. **遵循最小必要改动原则**
   - 优先复用现有实现、既有模式和项目约定，避免无依据的重写、扩面或过度设计
   - 只解决当前任务要求的问题，不擅自引入额外功能、依赖或流程复杂度

---

## 核心算子开发原则

> **严格遵循以下原则**

1. **不要凭记忆猜 API**
   - 第一步查 `tilelang-api-best-practices`（API 速查表和详细文档）
   - 第二步查 `examples/` 中的同类实现
   - 第三步若文档未覆盖，查源码 `tilelang/language/ascend_tile.py` · `tilelang/language/ascend.py` · `testing/python/language/`
   - 禁止凭直觉编造 API 调用、猜测参数签名；`tilelang/language/pto.py` 已废弃禁止使用
2. **从示例入手，禁止从空白文件开始**
   - 写新 kernel 前先在 `examples/` 中找到最相似的实现作为参考
   - 在已验证示例的结构和 API 用法基础上修改
3. **遵循硬件内存层级，不可跨级访问**
   - 昇腾 NPU 内存层级：GM ↔ L1（Cube 缓存）/ UB（Vector 缓冲）↔ L0A/L0B → L0C
   - 跨级搬运必须通过 `T.copy`，禁止 GM → L0 直接搬运等违规路径
4. **优先复用，定位问题而非重写**
   - 优先使用 `tilelang/language/` 中已有原语，不重新造轮子
   - 遇到错误时定位具体问题点并修复，禁止推翻重写或下意识简化代码
5. **新算子必须创建独立目录**
   - 每个新算子在 `examples/{op}/` 下创建独立文件夹（如 `examples/softmax/`），文件夹命名与算子名一致
   - 禁止将新算子放入 `normalization/`、`activation/` 等已有分类目录，禁止直接在 `examples/` 根目录创建 `.py`
6. **编程模式必须由用户明确指定**
   - 设计算子时必须先询问 Developer / Expert / 混合，**禁止用默认值绕过**
   - Developer：`alloc_shared/fragment` + 自动同步 + 全部 pass_configs 开启
   - Expert：显式 `alloc_L1/ub/L0A/L0B/L0C` + 手动 `T.Scope("C"/"V")` + 手动 `T.barrier_all/set_flag/wait_flag`
   - 详细对照见 `tilelang-expert-to-developer`
7. **遇到错误先分析原因，不绕过门禁**
   - 编译错误：定位行号 → 对比 API 文档 → 参考 examples/ 同类实现
   - 运行时错误：`T.dump_tensor` + `T.printf` 渐进式定位
   - 精度错误：从最小用例开始分段验证中间结果，检查 dtype
   - 环境问题：`source set_env.sh`，调 `tilelang-env-check`
   - 禁止一遇到错误就全部重写、不分析原因就尝试其他方案
8. **代码仓探索必须使用 subagent**
   - 禁止在 primary agent 中大规模探索代码仓
   - 使用 Explore Agent 查找资料，使用 Plan Agent 进行方案设计

---

## 开发规范

- Python: PEP 8, 类型注解, 行宽 100
- C++: Google Style, clang-format, 行宽 100
- 命名: `snake_case.py` / `snake_case.cc` / `test_<模块>_<功能>.py`

---

## 附录

- 架构详情：[architecture.md](.agents/skills/tilelang-custom-skill/architecture.md)
- TileLang 编程指南：[TileLang-Ascend Programming Guide](docs/TileLang-Ascend%20Programming%20Guide.md)
