# AGENTS.md

本文件为 AI Agent 在本代码仓库中进行 TileLang-Ascend 算子开发提供指导。

## 项目概述

本项目是 TileLang-Ascend 算子开发项目，基于 TVM 编译器基础设施，提供 Python DSL 用于开发华为昇腾 NPU 上的高性能 AI 计算 kernel。

### 核心功能

- 使用 Python DSL + `@tilelang.jit` 编写昇腾 NPU 自定义 kernel
- 支持 Developer 模式（自动化）和 Expert 模式（手动控制）两种编程范式
- 提供完整的编译、测试、调试及性能调优工作流

### 编译流程

```
@tilelang.jit (Python DSL)
  → tilelang/engine/lower.py (lowering)
  → src/transform/ (IR 变换 pass)
  → src/target/codegen_ascend_pto.cc (代码生成)
  → CANN 工具链 → NPU 执行
```

## Skills 目录说明

本项目的 skills 位于 `.agents/skills/` 目录下，每个 skill 包含一个 SKILL.md 文件。

可用 skills ：


| 技能                           | 触发时机            | 说明                                                                                                  |
| ---------------------------- | --------------- | --------------------------------------------------------------------------------------------------- |
| tilelang-custom-skill  | 需要时触发     | 通用skill |
| tilelang-ascend-tile-api | 新增或封装 `T.tile.xxx` 小 API 时 | 端到端打通 `ascend_tile.py` 前端、C++ lowering/codegen、Ascend C helper、测试与文档 |
| tilelang-op-design | 设计算子时 | 算子方案设计，生成 design.md |
| tilelang-op-generate | 实现算子时 | 基于 design.md 生成算子代码 |
| tilelang-pass-analyzer       | **查询 Pass 功能时立即触发**  | Pass 功能分析、对比、分类查询。关键词："XX pass 是干什么的"、"分析 XX pass"、"XX 和 YY pass 的区别" |
| tilelang-pass-workflow-analyzer | **查询 Pass 工作流时立即触发** | Pass 工作流分析、执行顺序、依赖关系、新 Pass 定位。关键词："pass 的工作流程"、"Pass 执行顺序"、"Pass 依赖关系"、"如何添加新 Pass" |
| skill-creator      | 创建skill时          | 用于创建skill  |

## Skill 自动调用规则 ⭐⭐⭐

**必须在第一时间识别用户意图，查阅 Skills 表格，匹配可用 skill以及触发时机，再决定是否调用，禁止先用 grep/read 等工具。**

### 自动调用流程

```
用户提问
  ↓
1. 分析用户意图（要做什么任务）
  ↓
2. 查阅 Skills 表格，检查哪些 skill 可用
  ↓
3. 根据触发时机匹配对应的 skill
  ↓
4. 匹配成功 → 立即调用对应 skill
  ↓
5. 不匹配 → 使用其他工具（grep/read/glob）
```

### 识别示例

| 用户问题 | 意图分析 | 匹配 Skill | 调用动作 |
|---------|---------|-----------|---------|
| "lower tile op 是干什么的" | Pass 功能查询 | tilelang-pass-analyzer | 立即调用 |
| "写一个 softmax 算子" | Vector 算子开发 | tilelang-vector-skill | 立即调用 |
| "VectorizeLoop pass 的作用" | Pass 功能查询 | tilelang-pass-analyzer | 立即调用 |
| "同步相关的 pass 有哪些" | Pass 分类查询 | tilelang-pass-analyzer | 立即调用 |
| "pass 的工作流程" | Pass 工作流查询 | tilelang-pass-workflow-analyzer | 立即调用 |
| "创建一个新的 skill" | Skill 创建 | skill-creator | 立即调用 |

## 核心原则 ⭐⭐⭐

严格遵循以下核心原则。

### 原则 1：不要凭记忆猜 API

- ✅ 第一步：查阅 [tilelang-api-best-practices](.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/SKILL.md)（API 速查表和详细文档）
- ✅ 第二步：查阅 `examples/` 中的同类实现
- ✅ 第三步：若文档未覆盖，查阅源码 `tilelang/language/ascend_tile.py` · `tilelang/language/ascend.py` · `testing/python/language/`
- ❌ 禁止：凭直觉编造 API 调用、猜测参数签名
- ❌ 禁止：`tilelang/language/pto.py` 此文件已废弃，禁止使用

### 原则 2：从示例入手

- 写新 kernel 前，先在 `examples/` 中找到最相似的实现
- 参考其结构和 API 用法，在此基础上修改
- ❌ 禁止：从空白文件开始写 kernel

### 原则 3：注意双层修改

- 改动常需同时修改 Python（`tilelang/`）和 C++（`src/`）
- 新增语言原语时，需要同步更新前端定义和后端 lowering/codegen
- 修改 IR 变换 pass 后，检查对已有算子的影响

### 原则 4：遵循硬件内存层级

昇腾 NPU 内存层级严格，**不可跨级访问**：

```
GM（全局内存）
  ↕ T.copy
L1（Cube 核缓存）/ UB（Vector 核缓冲）
  ↕ T.copy
L0A / L0B（矩阵输入寄存器）→ L0C（矩阵输出寄存器）
```

### 原则 5：优先复用，定位问题而非重写

- 优先使用 `tilelang/language/` 中的已有原语，不要重新造轮子
- 遇到错误时，定位具体问题点并修复
- ❌ 禁止：遇到错误就推翻重写、下意识简化代码

### 原则 6：新增算子必须创建独立目录

- 每个**新算子**必须在 `examples/` 下创建独立的文件夹，如 `examples/softmax/`
- 文件夹命名与算子名一致，代码文件放在其中
- ❌ 禁止：将新算子放入已有的 `normalization/`、`activation/` 等分类目录
- ❌ 禁止：直接在 `examples/` 根目录创建 `.py` 文件

## Developer 模式 vs Expert 模式

| 维度 | Developer（自动化） | Expert（手动控制） |
| --- | --- | --- |
| 内存分配 | `T.alloc_shared/fragment` 编译器自动映射 | `T.alloc_L1/ub/L0A/L0B/L0C` 显式指定 |
| 计算 | `T.Parallel` + 符号运算 | `T.tile.add/exp/max` 等 |
| 作用域 | 编译器自动分离 Cube/Vector | 显式 `with T.Scope("C"/"V")` |
| 同步 | 自动 | 手动 `T.barrier_all/set_flag/wait_flag` |
| pass_configs | 全部开启 | 全部关闭或不设 |

详细对照、pass_configs 配置与转换指南：[tilelang-expert-to-developer](.agents/skills/tilelang-custom-skill/tilelang-expert-to-developer/SKILL.md)

## API 参考

编写 kernel 时的 API 查阅入口：

- **API 速查与最佳实践**：[tilelang-api-best-practices](.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/SKILL.md)
  - [Kernel 定义、内存分配、数据搬运](.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-kernel-memory.md)
  - [计算原语：GEMM、归约、Tile 扩展操作](.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-compute.md)
  - [调度、同步与调试](.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-schedule-sync.md)

若上述文档未覆盖，查阅源码：`tilelang/language/ascend_tile.py` · `tilelang/language/ascend.py` · `testing/python/language/`

## 分阶段开发指南

严格按照以下阶段开发算子

### 阶段一：需求分析与方案设计

1. 理解算子的数学公式和计算逻辑
2. 在 `examples/` 中找到最相似的实现作为参考
3. 必须询问使用 Developer 模式、Expert 模式或混合编程模式，否则不进入下一步
4. 将公式拆解为 TileLang API 的组合，查阅 API 文档验证可行性
5. 按需调用 skill

### 阶段二：算子实现

1. **必须在 `examples/` 下创建以算子命名的独立文件夹**（如 `examples/softmax/`）
2. 参考同类示例的代码结构
3. 一次只实现一个核心函数，立即验证
4. ❌ 禁止：将新算子放入已有的分类目录（如 `normalization/`、`activation/`）

核心实现要点：

- 使用 `@tilelang.jit` 装饰器定义 kernel
- 按硬件内存层级正确分配和搬运数据
- Developer 模式下利用编译器自动化特性
- Expert 模式下显式管理作用域和同步

### 阶段三：构建和测试

```bash
# 设置环境
source set_env.sh

# 运行测试
python examples/<算子>.py
```

验证步骤：

- Level 0：小规模数据 → 基础功能验证
- Level 1：典型规模 → 正确性验证
- Level 2：边界值/极值 → 鲁棒性验证
- Level 3：大规模数据 → 性能验证

### 阶段四：调试

遇到问题时：

1. 在 kernel 中使用 `T.printf` 打印中间值
2. 使用 `T.dump_tensor` 检查张量内容
3. 查看 build 目录中生成的 Ascend C 代码
4. 参考 [debug-helper](.agents/skills/tilelang-custom-skill/tilelang-debug-helper/SKILL.md) 配置 GDB
5. 遇到错误时，优先使用 [tilelang-error-fixer](.agents/skills/tilelang-custom-skill/tilelang-error-fixer/SKILL.md) 进行调试

### 阶段五：编写测试

在 `examples/` 中添加测试文件，命名规范：`test_<模块>_<功能>.py`

## 新增算子流程

1. `tilelang/language/` 定义 Python API
2. `src/transform/` 或 `tilelang/transform/` 实现 lowering
3. `src/target/` 添加代码生成（如需）
4. **`examples/` 创建以算子命名的独立文件夹**（如 `examples/softmax/softmax.py`）
5. `testing/python/` 编写测试

## 错误处理 ⭐

| 错误类型 | 处理方式 |
| --- | --- |
| 编译错误 | 定位错误行号，对比 API 文档检查用法，参考 `examples/` 同类实现 |
| 运行时错误 | `T.dump_tensor` + `T.printf` 定位问题，采用渐进式调试 |
| 精度错误 | 从最小用例开始，分段验证中间结果，检查数据类型 |
| 环境问题 | 运行 `source set_env.sh`，参考 [env-check](.agents/skills/tilelang-custom-skill/ascendc-env-check/SKILL.md) |

**禁止：一遇到错误就全部重写、不分析原因就尝试其他方案。**

## 开发规范

- Python: PEP 8, 类型注解, 行宽 100
- C++: Google Style, clang-format, 行宽 100
- 命名: `snake_case.py` / `snake_case.cc` / `test_<模块>_<功能>.py`

## 附录

### 架构详情

详见 [architecture.md](.agents/skills/tilelang-custom-skill/architecture.md)

### TileLang 编程指南

详见 [TileLang-Ascend Programming Guide](docs/TileLang-Ascend%20Programming%20Guide.md)

### 文件查找规则

- 优先使用 Glob 工具通过文件名模式搜索
- 文档中的路径可能是相对路径，需用 Glob 在整个项目中搜索
- 使用 Explore Agent 查找资料，使用 Plan Agent 进行方案设计
