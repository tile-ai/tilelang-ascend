---
name: tilelang-pass-generate
description: "根据 pass-design.md 与 workflow 分析结果生成 TileLang-Ascend Pass 的最终实现代码（不含 UT/ST）。先输出实现骨架文档（pass-impl-skeleton.md）确认框架设计，再生成 C++ 实现、Python 封装、Pipeline 接入，并完成最小冒烟验证。测试生成由后续独立 skill 负责。触发关键词：实现 Pass、生成 Pass 代码、Pass 编码、根据设计文档实现 Pass、写 Pass 代码、落地 Pass、新增 Pass 实现。"
---

# TileLang-Ascend Pass 代码生成 Skill

---

## 1. 目标与边界

### 1.1 本 skill 负责（In Scope）

把已经定型的 Pass 设计文档（`pass-design.md`）落到代码上，覆盖：

- **C++ 实现**：`src/transform/<pass_name>.cc`
- **Python 封装**：`tilelang/transform/__init__.py`
- **配置键（可选）**：`tilelang/transform/pass_config.py`
- **Pipeline 接入**：`tilelang/engine/phase.py`
- **最小冒烟验证**：导入是否成功、Pipeline 是否仍然能跑通最小 example、跨文件命名是否一致

### 1.2 本 skill 不做（Out of Scope）

- ❌ **不生成 UT / ST 测试代码** —— 由后续单独的测试生成 skill 负责
- ❌ 不重做 Pass 定位与方案决策（属于 `tilelang-pass-design`）
- ❌ 不修改 TVM 原生 Pass

> 所有重大决策必须沿用 `pass-design.md` 已经写定的内容；如果设计文档里某项是「待确认」或缺失，则停下来回到 `tilelang-pass-design` 补齐，不在本 skill 里临时拍板。

> 完成代码生成后，提示用户调用「Pass 测试生成 skill（待创建）」补 UT/ST，不在本 skill 里偷跑。

---

## 2. 必需输入

| 字段 | 说明 | 缺失处理 |
|------|------|----------|
| `pass-design.md` | 已通过 `tilelang-pass-design` 自检的 Pass 设计文档 | 询问用户位置；若没有则建议先跑 `tilelang-pass-design` |
| Pass 名称 | C++ 类名、Python 函数名、注册名 | 从 `pass-design.md` §1.1 读取 |
| 阶段归属与位置 | `Phase 1 / Phase 2`，以及插入位点 | 从 `pass-design.md` §2 读取 |
| 父类与核心方法 | `IRMutatorWithAnalyzer` / `StmtExprVisitor` / `StmtExprMutator` 等 | 从 `pass-design.md` §4.1 读取 |
| 输入/输出 attrs | 上下游数据传递 | 从 `pass-design.md` §2.4、§3 读取 |

> 设计文档 §5「测试方案」本 skill 不消费，仅作为下游测试 skill 的输入保留。

**输入校验规则**：

1. 若 `pass-design.md` 不存在 → 立即停止，引导用户使用 `tilelang-pass-design`
2. 若设计文档中 §2 / §3 / §4 任一关键章节出现「待确认」「待补充」「TODO」 → 立即停止，列出缺口并要求补齐
3. 若 Pass 名称、阶段归属、Pipeline 位置任何一项不明确 → 立即停止，要求用户先回到 design skill

---

## 3. 工作流程

「**先骨架、再代码、最后冒烟**」三段式流程，骨架阶段不可跳过。

```
[输入校验]
   ↓
[Phase A: 信息收集]
   ↓
[Phase B: 生成实现骨架文档 pass-impl-skeleton.md]   ← 框架设计文档
   ↓
[用户确认骨架]
   ↓
[Phase C: 落代码（C++ / Python / pass_config / phase.py）]
   ↓
[Phase D: 最小冒烟验证（不跑 UT/ST）]
   ↓
[Phase E: 收尾报告 + 引导补测试]
```

### Phase A：信息收集

按以下顺序读取，禁止一上来就 grep 整个 `src/transform/`：

1. **设计文档**：用户指定路径，或默认在当前目录 `pass-design.md`
2. **总体约束**：`.agents/skills/tilelang-pass-agents-guide.md`
3. **实现模式**：`.agents/skills/tilelang-pass-design/references/pass-impl-patterns.md`
4. **Pipeline 现状**：`tilelang/engine/phase.py`（确认插入位点的上下文与函数签名）
5. **Python 封装现状**：`tilelang/transform/__init__.py`（确认现有命名风格、`_ffi_api` 调用方式）
6. **相似 Pass 源码**：`pass-design.md` §4.1 指定的参考 Pass，仅读 1–2 个最接近的实现，不得扫整目录

> 信息源冲突时，优先级：`pass-design.md` > `pass-impl-patterns.md` > 相似 Pass 源码 > 自身经验。

### Phase B：生成实现骨架文档（pass-impl-skeleton.md）

骨架文档是「**代码层面的最后一次结构化对齐**」。它比 `pass-design.md` 更落地，但又比真实代码更轻量，目的是在写代码前一次性把以下决策列清楚：

1. **改动文件清单**（含状态：新建 / 修改），**仅限实现侧 4 个文件**：`.cc` / `__init__.py` / `pass_config.py` / `phase.py`
2. **C++ 类骨架**：类名、父类、成员变量、构造函数签名、所有要重写的 `VisitStmt_` / `VisitExpr_` 方法签名（不写实现体）
3. **Substitute 入口流程**：步骤化伪代码（读取 attrs → 构造 mutator → MutateFunc → 设置 attrs）
4. **辅助类/辅助函数清单**（如 `Detector`、`Analyzer`、`Rewriter` 模式拆分）
5. **Attr 读写表**：键名 / 类型 / 来源 Pass / 缺失策略
6. **注册与配置键**：`TVM_REGISTER_GLOBAL` 完整字符串、可选的 `TVM_REGISTER_PASS_CONFIG_OPTION`
7. **Python 封装函数签名**：参数、docstring 要点
8. **Pipeline 接入点**：在 `phase.py` 哪一行（具体到上下游 Pass 名称）
9. **最小冒烟验证步骤**：先跑哪一条命令（导入冒烟 / 编译冒烟 / 最小 example 能跑）

> ⚠️ 骨架文档不写测试用例。测试相关内容由后续独立 skill 负责，本骨架里仅在末尾留一行「测试待补由 Pass 测试生成 skill 处理」作为交棒标记。

骨架文档基于 `templates/pass-impl-skeleton-template.md` 填写，输出到 `pass-impl-skeleton.md`。完成后**必须停下来等用户确认**，再进入 Phase C。

> 骨架阶段禁止做的事：写完整 `VisitStmt_` 函数体、动除骨架文档外的任何文件。

### Phase C：落代码

在用户确认骨架后，按以下顺序逐文件落地。**每完成一个文件，立即做最小检查再走下一个**（参见 §6 增量验证策略）。

| 顺序 | 文件 | 主要内容 | 完成后立即检查 |
|------|------|----------|----------------|
| 1 | `src/transform/<pass_name>.cc` | C++ 实现，按骨架填充 Visit 方法体、辅助函数、注册宏 | 语法/include 完整性、注册宏字符串、namespace 闭合 |
| 2 | `tilelang/transform/__init__.py` | Python 封装函数 + 必要的 import | `python -c "from tilelang.transform import <Pass>"` |
| 3 | `tilelang/transform/pass_config.py` | 仅当骨架声明需要新增配置键时改动 | 读一次该文件确认没有重名 |
| 4 | `tilelang/engine/phase.py` | 在指定 Pass 前/后插入一行调用 | 视觉对齐上下文，避免插错阶段 |

> 本 skill 不写、不改 `testing/python/` 下任何文件。

#### 落代码硬约束

- **禁止脱离 `pass-impl-patterns.md` 模板**：父类继承、`Substitute` 静态入口、`CreatePrimFuncPass` 调用方式必须与模板一致
- **禁止改动 `tir.transform.*` 等 TVM 原生 Pass**（来自 `tilelang-pass-agents-guide.md` 约束 3）
- **禁止只改 `.cc` 不改 Python 封装**（约束 4）
- **禁止把多种职责塞进一个 Pass**（约束 4：功能正交）
- **配置键缺省值默认 `False`**：新增 Pass 默认不开启，需要走 `pass_configs` 显式启用，除非 `pass-design.md` §4.5 明确写了「默认开启」并给出理由
- **Attr 读取必须做 defined() 检查**：缺失时按设计文档的策略处理（报错 / 跳过 / 默认值），不得静默崩溃
- **C++ 注释保持最小**：仅在 WHY 不直观时写一行（来自仓库总规约）

### Phase D：最小冒烟验证

本 skill 只做不依赖 UT/ST 的冒烟验证，按优先级顺序执行（完成第一项即可继续，但必须至少做一项）：

1. **导入冒烟**：
   ```bash
   python -c "import tilelang; from tilelang.transform import <Pass>; print(<Pass>())"
   ```
   验证 Python 封装、`_ffi_api` 注册、C++ 端 `TVM_REGISTER_GLOBAL` 字符串四方一致。
2. **构建冒烟**（如本地能跑构建）：项目根目录的最小重新构建，验证 C++ 文件能编过。
3. **Pipeline 冒烟**：跑一条**已存在**的最小 example（如 `examples/elementwise/...`）确认 pipeline 没有因为本 Pass 的接入而崩。
4. **跨文件一致性 grep**：
   ```bash
   grep -n "<PassName>" src/transform/<pass>.cc tilelang/transform/__init__.py tilelang/engine/phase.py
   grep -n "tl.<pass_lower>" src/transform/<pass>.cc tilelang/transform/pass_config.py
   ```

> ⚠️ **本阶段不写、不跑 UT/ST 单测**。如果配套的 `testing/python/` 下已有更早的相关测试，可以顺便跑一下作为额外冒烟，但**禁止为本 Pass 新建任何测试文件**。

若验证失败，按以下顺序定位：

- 编译错误 → 头文件缺失、TVM API 签名变化、命名空间问题
- 注册错误 → `TVM_REGISTER_GLOBAL` 字符串与 Python 调用名不一致
- Attr 读取错误 → 类型签名（`Map<Var, ...>` vs `Map<Buffer, ...>`）不匹配
- Pipeline 顺序错误 → 上游 attrs 未产生时本 Pass 已被调用

**禁止失败后无脑加 try/except 把异常吞掉**（仓库总规约：不要遮住问题）。

### Phase E：收尾报告 + 引导补测试

参考 §7 模板输出报告，必须包含：
- 已生成 / 修改的实现侧文件
- 已跑过的冒烟项 与 未即时验证项
- **明确提示「UT/ST 待补，建议下一步使用 Pass 测试生成 skill」**

---

## 4. 落地阶段决策速查

| 场景 | 默认做法 |
|------|----------|
| 父类是 `IRMutatorWithAnalyzer` | 在 `Substitute` 中执行 `mutator.MutateFunc(f)`，构造函数传 `f->body` |
| 父类是 `StmtExprVisitor` | 在 `Substitute` 中执行 `collector.VisitStmt(f->body)`，最后 `f.WithAttrs({...})` |
| 父类是 `StmtExprMutator` | 不需要 analyzer，直接重写 `VisitStmt_/VisitExpr_` |
| Pass 有输入 attrs | `f->GetAttr<...>(key)` + `defined()` 检查；缺失策略走设计文档 |
| Pass 有输出 attrs | 在 `Substitute` 末尾 `f.WithAttrs({{key, value}})` |
| Pass 是 Ascend 特定 | 加 `is_npu` 判断或仅在 `OptimizeForTarget` 中调用 |
| Pass 与现有 Pass 功能重合 | **回退**：在现有 Pass 内做增量改动，不新增 Pass |
| 设计文档与相似 Pass 实现冲突 | 以设计文档为准；如设计文档不合理，停下来回到 design skill |

---

## 5. 修改 / 重构 Pass 的差异化流程

本 skill 也处理「修改已有 Pass」「重构已有 Pass」，差异如下：

### 5.1 修改已有 Pass

- 跳过 Phase B 中的「类骨架」「注册键」部分，仅写出**「目标行为差异点 + 关键修改方法清单」**
- 必须先在骨架文档里列出：
  - 当前行为 vs 目标行为
  - 真正控制该行为的函数（精确到方法名 / 关键 if 分支）
  - 最小修改范围（行数 / 受影响 Visit 方法）
- Phase C 强制做最小修改，不允许顺手重排其他无关代码
- Phase D 冒烟之外，**不在本 skill 内补回归测试**；在收尾报告里明确写「需补 X 类回归 case」，交棒给测试 skill

### 5.2 重构已有 Pass

- 默认语义保持不变，注册名、Python 封装签名、phase.py 调用位置都不动
- 骨架文档里要写出：
  - 重构前后的类划分对照（旧类 → 新类）
  - 哪些是「纯结构整理」，哪些是「为了整理而不得不动的语义点」
- 禁止在同一个重构 PR 里混入功能性改动
- Phase D 冒烟跑一条 example 确认 pipeline 不退化即可；广覆盖回归留给测试 skill

---

## 6. 增量验证策略

每完成一个**有实质行为的修改**就停一次，禁止连续大改后再一次性验证。

| 修改点 | 立即验证手段 |
|--------|--------------|
| 改了 C++ 文件 | 至少 `clang-format` / 本地 build；如时间不允许，至少 grep 注册宏字符串 |
| 改了 Python `__init__.py` | `python -c "from tilelang.transform import <Pass>"` |
| 改了 `phase.py` | 跑一条最小 example 编译（如 `examples/elementwise/...`）确认 pipeline 不炸 |
| 改了 `pass_config.py` | grep 一次配置键名，确认与 C++ 字符串完全一致 |

如果某一步无法立即验证（环境问题、build 缓慢），**必须明确告诉用户**「这一步未即时验证，待用户在本机确认」，不得伪装成已验证。

---

## 7. 完成报告模板

```
## Pass 代码生成报告

- Pass 名称: {Pass 名称}
- 任务类型: 新增 / 修改 / 重构
- 阶段归属: {Phase 1 / Phase 2}
- Pipeline 位置: {具体位置}

### 骨架文档
- 路径: {pass-impl-skeleton.md 路径}
- 用户确认状态: ✅ 已确认 / ⚠️ 未确认即落代码（不应出现）

### 已生成 / 修改文件（仅实现侧，不含测试）
| 文件 | 状态 | 行数变化 |
| `src/transform/<pass_name>.cc` | 新建 | +XXX |
| `tilelang/transform/__init__.py` | 修改 | +X |
| `tilelang/transform/pass_config.py` | 修改 / 未改 | +X |
| `tilelang/engine/phase.py` | 修改 | +1 |

### 已执行的冒烟验证
1. 导入冒烟: ✅ / ❌（失败原因）
2. 跨文件命名 grep 一致: ✅ / ❌
3. 最小 example 跑通: ✅ / ❌ / 未即时验证
4. 构建冒烟: ✅ / ❌ / 未即时验证

### 未即时验证项（需用户在本机确认）
- {项目}

### 剩余风险
- {风险 1}
- {风险 2}

### ⚠️ 测试待补（交棒）
- 本 skill 仅做实现侧代码生成，不生成 UT/ST。
- 建议下一步使用「Pass 测试生成 skill」（待创建）补充以下测试：
  - 功能测试：{设计文档 §5.1 已列}
  - 依赖测试：{设计文档 §5.2 已列}
  - 边界测试：{设计文档 §5.3 已列}
  - （修改任务）回归测试：{在本次修改中需要覆盖的目标行为差异}
```

---

## 8. 与其他 Skill 的关系

| Skill | 关系 | 衔接点 |
|-------|------|--------|
| `tilelang-pass-agents-guide` | 上层指导 | 总体执行流程、约束、文件清单 |
| `tilelang-pass-analyzer` | 依赖 | 相似 Pass 实现、IR 示例 |
| `tilelang-pass-workflow-analyzer` | 依赖 | Pipeline 位置、依赖图 |
| `tilelang-pass-design` | **强依赖（输入）** | 提供 `pass-design.md` |
| `tilelang-pass-generate`（本 skill） | 实现代码生成终点 | 输出 `pass-impl-skeleton.md` + 落实现侧代码 + 冒烟验证 |
| Pass 测试生成 skill（**待创建**） | **下游交棒** | 接收本 skill 输出 + 设计文档 §5，生成 UT/ST |

---

## 9. 错误处理

| 场景 | 处理方式 |
|------|----------|
| `pass-design.md` 缺失 | 停止，引导用户先跑 `tilelang-pass-design` |
| 设计文档中阶段归属、Pipeline 位置不明确 | 停止，列出缺口，要求补齐 |
| 设计文档与现有 Pass 注册名冲突 | 停止，询问是否复用现有 Pass 而不是新增 |
| 用户跳过骨架直接要求落代码 | 仍然先输出骨架（可压缩），不允许跳过 |
| 用户要求顺手把 UT/ST 也写了 | 拒绝，引用本 skill §1.2 边界，引导去用测试生成 skill |
| 落代码阶段编译失败 | 不要加规避分支；定位真实原因后告诉用户 |
| 用户要求把 Pass 放到 TVM 原生 Pass 里 | 拒绝，引用 `tilelang-pass-agents-guide` 约束 3 |

---

## 10. 参考资料

| 文件 | 路径 | 用途 |
|------|------|------|
| Pass 总体约束 | `.agents/skills/tilelang-pass-agents-guide.md` | 工作流、约束、文件清单 |
| Pass 设计文档模板 | `.agents/skills/tilelang-pass-design/templates/pass-design-template.md` | 反查设计文档结构 |
| Pass 实现模式 | `.agents/skills/tilelang-pass-design/references/pass-impl-patterns.md` | C++ 类模板、Visit 模式、注册方式 |
| 骨架文档模板 | `templates/pass-impl-skeleton-template.md`（本 skill） | 实现骨架格式 |
| 落地清单 | `references/code-generation-checklist.md`（本 skill） | 实现侧文件检查项 |
| 接入位点参考 | `references/integration-points.md`（本 skill） | `__init__.py` / `phase.py` / `pass_config.py` 接入示例 |

---

## 11. 注意事项

1. **骨架阶段不可跳过**：哪怕用户着急，也要先输出 `pass-impl-skeleton.md`。骨架可以压缩，但不能省。
2. **设计文档是源真相**：所有重大决策都按设计文档来；设计文档不合理就回到 design skill，不要在生成阶段拍板。
3. **每改一处立刻冒烟**：禁止连续多文件改动后再一次性测试。
4. **不修 TVM 原生 Pass**：除非设计文档明确写了无法绕开的理由。
5. **Python 封装、phase.py、pass_config.py 三处一致**：Pass 名称、配置键名称、参数顺序必须三边对齐，grep 一次确认。
6. **不写测试**：本 skill 只生成实现侧代码，UT/ST 由独立 skill 处理；收尾报告里要把测试缺口列清楚以便交棒。
7. **报告要诚实**：未跑过的冒烟就标「未即时验证」，不要冒充已验证。
