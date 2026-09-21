# {Pass 名称} 实现骨架文档（pass-impl-skeleton.md）

> 本文件是「**代码层面的最后一次结构化对齐**」。比 `pass-design.md` 更落地，但比真实代码更轻量。
> 在用户确认本骨架前，**禁止生成任何 .cc / .py 代码**。
>
> ⚠️ 本骨架**只覆盖实现侧**（C++ / Python wrapper / pass_config / phase.py）。
> UT/ST 测试不在本骨架范围内，由后续独立 skill 处理。

---

## 0. 任务摘要

- **任务类型**：{新增 / 修改 / 重构}
- **设计文档**：{pass-design.md 路径}
- **Pass 名称**：{PassName}
- **阶段归属**：{Phase 1: LowerAndLegalize / Phase 2: OptimizeForTarget}
- **Pipeline 位置**：{在 PassA 后、PassB 前}
- **平台范围**：{平台无关 / Ascend 特定}

---

## 1. 改动文件清单（实现侧，仅 4 类）

| 序号 | 文件 | 状态 | 说明 |
|------|------|------|------|
| 1 | `src/transform/{pass_name}.cc` | {新建 / 修改} | C++ 主实现 |
| 2 | `tilelang/transform/__init__.py` | 修改 | Python 封装 |
| 3 | `tilelang/transform/pass_config.py` | {修改 / 不动} | 配置键定义（可选） |
| 4 | `tilelang/engine/phase.py` | 修改 | Pipeline 接入 |
| 5 | （可选）`src/transform/common/{helper}.h` | 新建 | 公共辅助类 |

> 测试文件（`testing/python/...`）**不在本骨架范围内**，由 Pass 测试生成 skill 处理。

---

## 2. C++ 类骨架（仅签名，不写实现体）

### 2.1 主类

```cpp
namespace tvm {
namespace tl {

class {PassName} : public {父类: arith::IRMutatorWithAnalyzer / StmtExprVisitor / StmtExprMutator} {
public:
  // Pass 入口
  static PrimFunc Substitute(PrimFunc f, PassContext ctx);

private:
  // 构造函数
  {PassName}(PrimFunc f, PassContext ctx, {输入 attrs 参数});

  // 成员变量
  {Map<Var, Array<PrimExpr>>} input_attr_;       // 来自 {上游 Pass}
  {Map<Buffer, PrimExpr>}     output_attr_;      // 供给 {下游 Pass}
  // ... 其他状态

  // 重写的 Visit 方法（仅列签名）
  Stmt VisitStmt_(const ForNode* op) final;
  Stmt VisitStmt_(const BufferStoreNode* op) final;
  Expr VisitExpr_(const BufferLoadNode* op) final;
  // ... 仅列将要重写的节点类型

  // 辅助方法
  bool IsTargetPattern(const {NodeType}* op) const;
  Stmt RewriteStmt(const {NodeType}* op);
};

}  // namespace tl
}  // namespace tvm
```

### 2.2 辅助类（如有，按 CrossCorePipeline 模式拆分）

```cpp
class {PassName}Detector : public StmtExprVisitor { ... };
class {PassName}Analyzer { ... };
class {PassName}Rewriter : public IRMutatorWithAnalyzer { ... };
```

> 仅当设计文档 §4.1 明确表示需要多类协作时才拆分；否则保持单类。

---

## 3. Substitute 入口流程（伪代码，逐行）

```
PrimFunc Substitute(PrimFunc f, PassContext ctx):
  1. 读取配置（若有配置键）
     bool enabled = ctx->GetConfig<Bool>(k{PassName}Config, Bool({默认值})).value();
     if (!enabled) return f;

  2. 读取上游 attrs
     auto attr = f->GetAttr<{Type}>("{key}");
     if (!attr.defined()) {
       // 缺失策略：{报错 / 跳过 / 默认值}
     }

  3. 构造变换器实例
     {PassName} mutator(f, ctx, attr.value());

  4. 执行变换
     PrimFunc new_f = mutator.MutateFunc(f);          # IRMutatorWithAnalyzer
     // 或 mutator.VisitStmt(f->body);                # StmtExprVisitor

  5. 设置输出 attrs（如有）
     new_f = new_f.WithAttrs({{"{out_key}", mutator.GetOutput()}});

  6. 返回
     return new_f;
```

---

## 4. Attr 读写表

### 4.1 输入 attrs

| 键名 | 类型 | 来源 Pass | 缺失策略 |
|------|------|-----------|----------|
| `{key1}` | `Map<Var, Array<PrimExpr>>` | `{产生 Pass}` | {报错 / 跳过 / 默认值} |
| `{key2}` | `Map<Buffer, PrimExpr>`     | `{产生 Pass}` | {报错 / 跳过 / 默认值} |

### 4.2 输出 attrs

| 键名 | 类型 | 下游消费 Pass |
|------|------|----------------|
| `{out_key1}` | `{Type}` | `{消费 Pass}` |

---

## 5. 注册与配置键

### 5.1 Pass 注册

```cpp
tvm::transform::Pass {PassName}() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return {PassName}::Substitute(std::move(f), ctx);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.{PassName}", {});
}

TVM_REGISTER_GLOBAL("tl.transform.{PassName}").set_body_typed({PassName});
```

### 5.2 配置键（可选）

```cpp
static constexpr const char *k{PassName}Config = "tl.{pass_name_lower}";
TVM_REGISTER_PASS_CONFIG_OPTION(k{PassName}Config, Bool);
```

> 默认值：`Bool(false)`。Ascend 特定 Pass 默认不开启，由 `pass_configs` 显式启用。

---

## 6. Python 封装函数签名

```python
# tilelang/transform/__init__.py

def {PassName}({可选参数: target=None / is_npu=False}):
    """{一句话说明}。

    {补充说明：上下游依赖、典型使用场景}

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass.
    """
    return _ffi_api.{PassName}({可选参数})
```

---

## 7. Pipeline 接入点（具体到上下文）

### 7.1 接入位置

**目标函数**：`{LowerAndLegalize / OptimizeForTarget}`，文件 `tilelang/engine/phase.py`

**插入位置**：在 `{上游 Pass 行}` 之后、`{下游 Pass 行}` 之前

### 7.2 接入 diff（伪 diff）

```python
    mod = tilelang.transform.{上游 Pass}()(mod)
+   # ===== 新增 Pass =====
+   mod = tilelang.transform.{PassName}()(mod)
+   # =====
    mod = tilelang.transform.{下游 Pass}()(mod)
```

### 7.3 配置键索引（如有）

```python
# tilelang/transform/pass_config.py

class PassConfigKey(str, Enum):
    TL_{PASS_NAME_UPPER} = "tl.{pass_name_lower}"
    """Enable/disable {PassName}. Default: False."""
```

---

## 8. 最小冒烟验证步骤（生成代码后立即跑，不依赖 UT/ST）

按顺序执行，遇到第一个失败就停下来定位：

1. **导入冒烟**：
   ```bash
   python -c "import tilelang; from tilelang.transform import {PassName}; print({PassName}())"
   ```
2. **跨文件命名一致性 grep**：
   ```bash
   grep -n "{PassName}" src/transform/{pass_name}.cc tilelang/transform/__init__.py tilelang/engine/phase.py
   grep -n "tl.{pass_name_lower}" src/transform/{pass_name}.cc tilelang/transform/pass_config.py
   ```
3. **最小 example 跑通**（已有 example，不新建）：
   ```bash
   python {仓库已有的最小 example，例如 examples/elementwise/...}
   ```
4. **构建冒烟**（如果本机能跑）：{按仓库脚本}

---

## 9. 风险与待确认项

### 9.1 已知风险

- {风险 1：例如 attr 类型与上游 Pass 是否完全一致}
- {风险 2：例如 Pipeline 顺序是否与现有 Pass 冲突}

### 9.2 待用户确认

- {项 1：是否需要新增配置键，默认值是否合理}
- {项 2：Pipeline 位置是否需要进一步收敛}

---

## 10. 测试待补（交棒标记）

> 本骨架文档**不展开测试细节**。
>
> 设计文档 §5 已经列出测试方案（功能/依赖/边界），由后续独立的 Pass 测试生成 skill 负责落地。
> 实现代码生成完成后，请使用 Pass 测试生成 skill 接续这部分工作。

---

## 11. 用户确认提示

请用户确认以下三件事后，才进入代码生成阶段：

1. ☐ 上述 C++ 类骨架（父类、Visit 方法、辅助类）符合预期
2. ☐ Pipeline 接入位置（具体到上下游 Pass）符合预期
3. ☐ Python 封装函数签名（参数、命名）符合预期

确认后回复「确认骨架」或指出需要调整的具体条目。
