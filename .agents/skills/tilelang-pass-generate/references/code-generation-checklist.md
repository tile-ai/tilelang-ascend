# Pass 代码生成 Checklist

每完成一个文件落地后逐项过一遍，全部通过才算该文件完成。

> ⚠️ 本 checklist **只覆盖实现侧 4 类文件**：C++ / Python wrapper / pass_config / phase.py。
> UT/ST 测试不在本 skill 范围内，由后续独立的 Pass 测试生成 skill 负责。

---

## 1. C++ 实现文件 `src/transform/<pass_name>.cc`

| # | 检查项 | 失败处理 |
|---|--------|----------|
| 1 | `#include` 完整：`<tvm/tir/transform.h>`、`<tvm/arith/analyzer.h>`（如继承 `IRMutatorWithAnalyzer`）等 | 缺什么补什么 |
| 2 | `namespace tvm { namespace tl { ... } }` 闭合正确 | 修正 |
| 3 | 主类继承的父类与骨架文档一致 | 退回骨架，不要在代码层临时换父类 |
| 4 | `static PrimFunc Substitute(PrimFunc f, PassContext ctx)` 入口存在 | 补 |
| 5 | 上游 attr 读取做了 `defined()` 检查 | 补 |
| 6 | 缺失策略与设计文档一致（报错 / 跳过 / 默认值） | 修正 |
| 7 | 输出 attr 通过 `f.WithAttr(s)` 设置（如有） | 补 |
| 8 | `CreatePrimFuncPass(pass_func, 0, "tl.{PassName}", {})` 第三个字符串与类名一致 | 修正 |
| 9 | `TVM_REGISTER_GLOBAL("tl.transform.{PassName}").set_body_typed(...)` 字符串与 Python 调用对齐 | 修正 |
| 10 | 配置键（如有）：`TVM_REGISTER_PASS_CONFIG_OPTION(...)` 一次注册，键名与 `pass_config.py` 完全一致 | 修正 |
| 11 | 注释保持最少：仅在 WHY 不直观时一行注释 | 删除冗余注释 |
| 12 | 没有 `TODO` / `FIXME` / 占位符 | 删除或落实 |
| 13 | 没有引入对 `tir::transform::*` 原生 Pass 的修改 | 退回设计 |

---

## 2. Python 封装 `tilelang/transform/__init__.py`

| # | 检查项 | 失败处理 |
|---|--------|----------|
| 1 | 函数名与 C++ `TVM_REGISTER_GLOBAL` 字符串对齐（去掉 `tl.transform.` 前缀） | 修正 |
| 2 | 调用 `_ffi_api.{PassName}(...)` 而不是 `tvm.tl.transform.*` | 修正 |
| 3 | docstring 至少一句说明 + Returns 段 | 补充 |
| 4 | 参数顺序与 C++ Pass 注册函数一致 | 修正 |
| 5 | 没有破坏现有 import 顺序 | 调整 |

> 验证：`python -c "from tilelang.transform import {PassName}; print({PassName}())"`

---

## 3. 配置键 `tilelang/transform/pass_config.py`

仅在新增配置键时改动。

| # | 检查项 | 失败处理 |
|---|--------|----------|
| 1 | 新键放在合适的分组下（参考现有键的分类） | 调整位置 |
| 2 | 键名 `tl.xxx` 与 C++ 中的字符串完全一致 | 修正 |
| 3 | 默认值与 C++ `GetConfig<T>(key, default)` 中的默认值一致 | 修正 |
| 4 | 注释一句话说明用途与默认值 | 补充 |

---

## 4. Pipeline 接入 `tilelang/engine/phase.py`

| # | 检查项 | 失败处理 |
|---|--------|----------|
| 1 | 插入位置与设计文档 §2.3 一致 | 修正 |
| 2 | 上游 Pass 在本 Pass 之前调用 | 修正 |
| 3 | 下游 Pass 在本 Pass 之后调用 | 修正 |
| 4 | 没有插到错误的阶段（Phase 1 vs Phase 2） | 修正 |
| 5 | 一行调用风格与上下文 Pass 保持一致（缩进、注释） | 调整 |
| 6 | （Ascend 特定 Pass）仅在 NPU 路径触发，必要时配 `is_npu` 检查 | 修正 |

---

## 5. 跨文件一致性

最后做一次跨文件对齐：

| # | 检查项 | 命令 |
|---|--------|------|
| 1 | Pass 名称在 C++ 类、注册宏、Python 函数三处一致 | `grep -n "{PassName}" src/transform/{pass_name}.cc tilelang/transform/__init__.py` |
| 2 | 配置键在 C++ 字符串、`pass_config.py` 两处一致 | `grep -n "tl.{pass_name_lower}" src/transform/{pass_name}.cc tilelang/transform/pass_config.py` |
| 3 | `phase.py` 中的调用与 `__init__.py` 中的封装函数同名 | `grep -n "{PassName}" tilelang/engine/phase.py tilelang/transform/__init__.py` |
| 4 | 无残留的 TODO / FIXME / 占位符 | `grep -rn "TODO\|FIXME\|XXX" src/transform/{pass_name}.cc` |

---

## 6. 冒烟验证（不依赖 UT/ST）

| # | 项目 | 通过判定 |
|---|------|----------|
| 1 | 导入冒烟 | `python -c "from tilelang.transform import {PassName}; print({PassName}())"` 无异常 |
| 2 | 跨文件命名 grep 一致 | §5 中 4 条命令的输出都符合预期 |
| 3 | 最小 example 跑通 | 已有 example（如 `examples/elementwise/...`）能正常编译/运行 |
| 4 | 构建冒烟 | C++ 文件能通过项目构建脚本编译（如本机能跑） |

> 至少跑通 1、2 两项。3、4 视环境而定，未跑过的必须在报告里如实标注「未即时验证」。

---

## 7. 验证报告

最后确认报告里如实写明：

- 已跑通的命令
- 未即时验证的项
- 已知剩余风险
- 是否覆盖所有设计文档承诺的**实现行为**（测试覆盖由下游 skill 处理）
- **测试待补清单**（从设计文档 §5 抽取，作为给测试 skill 的交棒）

> 不要把「未跑过」写成「已通过」。
> 不要在本 skill 内顺手写测试，那是下游 skill 的职责。
