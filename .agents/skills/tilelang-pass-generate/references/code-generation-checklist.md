# Pass 代码生成 Checklist

每完成一个文件落地后逐项过一遍，全部通过才算该文件完成。

> 本 checklist 覆盖四个基础接入点，以及设计证明必需的 catalog / public header /
> `OperationConfig` supporting files。
> UT/ST 测试不在本 skill 范围内，由后续独立的 Pass 测试生成 skill 负责。

---

## 1. C++ 实现文件 `src/transform/<pass_name>.cc`

| # | 检查项 | 失败处理 |
|---|--------|----------|
| 1 | `#include` 完整：`<tvm/tir/transform.h>`、`<tvm/arith/analyzer.h>`（如继承 `IRMutatorWithAnalyzer`）等 | 缺什么补什么 |
| 2 | `namespace tvm { namespace tl { ... } }` 闭合正确 | 修正 |
| 3 | 主类继承的父类与骨架文档一致 | 退回骨架，不要在代码层临时换父类 |
| 4 | `static PrimFunc Substitute(PrimFunc f, PassContext ctx)` 入口存在 | 补 |
| 5 | 上游 attr 读取做了 `defined()` 检查；consumed/produced/invalidated structural facts 与骨架一致 | 补 |
| 6 | Attr 缺失和结构事实失效策略与设计一致 | 修正 |
| 7 | 输出 attr 与结构性 IR contract 均按骨架发布 | 补 |
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
| 4 | 没有跨越 MemoryPlanning、SyncInsert、ResourceScopeVerify、Selection/Legalize 的 contract 边界 | 修正 |
| 5 | 一行调用风格与上下文 Pass 保持一致（缩进、注释） | 调整 |
| 6 | （Ascend 特定 Pass）仅在 NPU 路径触发，必要时配 `is_npu` 检查 | 修正 |

---

## 5. Supporting files 与跨文件一致性

- 新 call 的 memory/pipeline effect 已按需更新 `OperationConfig`；resource owner 可由共用
  classifier 证明。
- 新 managed Vector terminal 的 catalog ABI、Selection、contract 与 emitter 保持一致。

最后做一次跨文件对齐：

| # | 检查项 | 命令 |
|---|--------|------|
| 1 | Pass 名称在 C++ 类、注册宏、Python 函数三处一致 | `grep -n "{PassName}" src/transform/{pass_name}.cc tilelang/transform/__init__.py` |
| 2 | 配置键在 C++ 字符串、`pass_config.py` 两处一致 | `grep -n "tl.{pass_name_lower}" src/transform/{pass_name}.cc tilelang/transform/pass_config.py` |
| 3 | `phase.py` 中的调用与 `__init__.py` 中的封装函数同名 | `grep -n "{PassName}" tilelang/engine/phase.py tilelang/transform/__init__.py` |
| 4 | 无残留的 TODO / FIXME / 占位符 | `grep -rn "TODO\|FIXME\|XXX" src/transform/{pass_name}.cc` |

---

## 6. 冒烟验证（不依赖 UT/ST）

| # | 项目 | 通过判定 | 必跑条件 |
|---|------|----------|----------|
| 1 | 导入冒烟 | `python -c "from tilelang.transform import {PassName}; print({PassName}())"` 无异常 | **始终必跑** |
| 2 | 跨文件命名 grep 一致 | §5 中 4 条命令的输出都符合预期 | **始终必跑** |
| 3 | 构建冒烟 | C++ 文件能通过项目构建脚本编译 | **修改 shared-library C++ 时必跑** |
| 4 | 现有聚焦回归 | 既有编译/行为用例经过新路径并通过 | **始终必跑；无覆盖则报告测试缺口** |

> **执行规则（不可跳步）：**
> - 第 1、2 项与环境无关，**任何情况下都必须跑**。
> - 第 3 项不能用 grep 替代；构建环境缺失时明确报告阻塞。
> - 第 4 项必须经过新行为；现有测试无覆盖时明确报告永久测试缺口。

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
