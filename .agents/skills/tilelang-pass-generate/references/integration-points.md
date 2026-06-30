# Pass 接入点参考

本文件汇总落代码阶段三个核心接入位置的现成模板与对齐原则。代码生成时按这里的样式照搬，不要自创风格。

> ⚠️ 本文件**只覆盖实现侧**：`tilelang/transform/__init__.py` / `tilelang/engine/phase.py` / `tilelang/transform/pass_config.py` / C++ 注册块。
> 测试文件接入由 Pass 测试生成 skill（待创建）单独处理，本文件不涉及。

---

## 1. `tilelang/transform/__init__.py`

### 1.1 现有风格

打开 `tilelang/transform/__init__.py`，所有 Pass 封装统一是「**函数 + docstring + 一行 `_ffi_api` 调用**」。

### 1.2 无参 Pass 模板

```python
def {PassName}():
    """{一句话功能描述}。

    {可选：补充上下文，如依赖的 attrs、典型使用场景}

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass.
    """
    return _ffi_api.{PassName}()  # type: ignore
```

### 1.3 带参 Pass 模板

```python
def {PassName}(target=None, is_npu: bool = False):
    """{一句话功能描述}。

    Parameters
    ----------
    target : tvm.target.Target, optional
        The compilation target.
    is_npu : bool
        Whether the current target is Ascend NPU.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass.
    """
    return _ffi_api.{PassName}(target, is_npu)  # type: ignore
```

### 1.4 对齐要点

- 函数名：与 C++ `TVM_REGISTER_GLOBAL("tl.transform.{PassName}")` 中的尾段一致
- `_ffi_api.{PassName}` 中的方法名：和 `TVM_REGISTER_GLOBAL` 第二段一致（去掉 `tl.transform.`）
- 不要写复杂逻辑：Python 封装就是一层透传

---

## 2. `tilelang/engine/phase.py`

### 2.1 Phase 1 当前结构（节选）

```python
def LowerAndLegalize(mod: IRModule, target: Target) -> IRModule:
    # InjectTmpBuffer 受 TL_ASCEND_INJECT_TMP_BUFFER 控制（默认 True）
    # 关闭后用户需手动分配 tmp_ub
    mod = tilelang.transform.InjectTmpBuffer(target)(mod)
    mod = tilelang.transform.AscendInferBufferScope()(mod)
    mod = tilelang.transform.AscendVidReduction()(mod)
    mod = tilelang.transform.BufferShapeCollector()(mod)
    mod = tir.transform.BindTarget(target)(mod)
    mod = tilelang.transform.HostProcesser()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tilelang.transform.AscendLowerParallelToVector()(mod)
    mod = tilelang.transform.LayoutInference()(mod)
    mod = tilelang.transform.CollectBufferShapes()(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    mod = tir.transform.Simplify()(mod)
    return mod
```

### 2.2 Phase 2 当前结构（节选）

```python
def OptimizeForTarget(mod: IRModule, target: Target, platform: str) -> IRModule:
    mod = tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.CrossCorePipeline()(mod)
    mod = tilelang.transform.CombineCV()(mod)
    mod = tilelang.transform.PipelinePlanning()(mod)
    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
    mod = tilelang.transform.AscendLowerOpaqueBlock()(mod)
    mod = tir.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tilelang.transform.Flatten2DBuffer()(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tilelang.transform.VectorizeLoop(...)(mod)
    mod = tilelang.transform.AscendStorageRewrite(is_npu=...)(mod)
    # ... 后续 Pass
    return mod
```

### 2.3 接入风格

- 一行一个 `mod = tilelang.transform.{Pass}()(mod)`
- 缩进与上下文一致（4 空格）
- 每行不写注释，除非需要解释「为什么必须放在这里」
- 配置驱动：若 Pass 受 `pass_configs` 控制，仍然写在 phase.py 中调用，由 Pass 自己读 config 决定是否生效（避免 phase.py 出现复杂 if）

### 2.4 接入位置 diff 模板

```python
    mod = tilelang.transform.{上游 Pass}()(mod)
+   mod = tilelang.transform.{PassName}()(mod)
    mod = tilelang.transform.{下游 Pass}()(mod)
```

> 不要插到错误阶段：Phase 1 是「Lowering + 合法化」，Phase 2 是「目标后端优化」，看 `pass-design.md` §2.1 决定。

---

## 3. `tilelang/transform/pass_config.py`

### 3.1 现有风格

`PassConfigKey(str, Enum)` 类内每个键一行，附 docstring 简述用途和默认值。

### 3.2 新增键模板

```python
class PassConfigKey(str, Enum):
    # ... 已有键 ...

    TL_{PASS_NAME_UPPER} = "tl.{pass_name_lower}"
    """Enable/disable {PassName}. Default: False.

    {补充：何时启用、对哪些算子有效}
    """
```

### 3.3 对齐要点

- 键名 `tl.xxx` 全小写，下划线分词，与 C++ `static constexpr const char *` 字符串完全一致
- 默认值通过 C++ 端 `ctx->GetConfig<Bool>(key, Bool(false))` 决定，Python 这边不重复声明默认值
- Ascend 特定 Pass 默认 `False`，由用户显式开启

---

## 4. C++ 注册块完整模板

放在每个 `src/transform/<pass_name>.cc` 文件末尾：

```cpp
namespace tvm {
namespace tl {

// ========== Pass 实现 ==========
// class {PassName} : public ... { ... };

// ========== Pass 注册 ==========
tvm::transform::Pass {PassName}() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return {PassName}::Substitute(std::move(f), ctx);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.{PassName}", {});
}

TVM_REGISTER_GLOBAL("tl.transform.{PassName}")
    .set_body_typed({PassName});

// ========== 配置键（可选） ==========
static constexpr const char *k{PassName}Config = "tl.{pass_name_lower}";
TVM_REGISTER_PASS_CONFIG_OPTION(k{PassName}Config, Bool);

}  // namespace tl
}  // namespace tvm
```

### 4.1 命名约定

| 元素 | 形态 |
|------|------|
| C++ 类名 | `PascalCase`，如 `AscendMemoryPlanning` |
| 注册函数名 | 与类名一致 |
| `CreatePrimFuncPass` 第三参数 | `"tl.{PascalCase}"` |
| `TVM_REGISTER_GLOBAL` | `"tl.transform.{PascalCase}"` |
| 配置键字符串 | `"tl.{lower_snake_case}"` |
| Python 函数名 | 与 C++ 类名一致（`PascalCase`） |
| 文件名 | `lower_snake_case.cc`（与 `src/transform/` 现有文件一致） |
| Python 配置 enum | `TL_{UPPER_SNAKE_CASE}` |

---

## 5. 跨文件命名对齐速查

落代码完成后跑一次：

```bash
grep -n "{PassName}" \
  src/transform/{pass_name}.cc \
  tilelang/transform/__init__.py \
  tilelang/engine/phase.py

grep -n "tl.{pass_name_lower}" \
  src/transform/{pass_name}.cc \
  tilelang/transform/pass_config.py
```

三处 + 配置键两处都能命中且一致 → 命名通过。

---

## 6. 与测试 skill 的交棒

实现侧代码生成完毕后，本 skill 的产出可以作为测试 skill 的输入：

- **本 skill 的产出**：实现完成的 `.cc` / `__init__.py` / `pass_config.py` / `phase.py`
- **测试 skill 的输入**：上述实现文件 + `pass-design.md` §5 测试方案
- **测试 skill 的产出**：`testing/python/...` 下的 UT/ST

> 本 skill 完成报告里要把「测试待补清单」明确列出，便于测试 skill 直接接续。
