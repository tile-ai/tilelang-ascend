---
name: tilelang-mode-guide
description: TileLang Ascend Developer 模式与 Expert 模式编程指南。当需要编写 Developer/Expert/混合模式的算子代码、了解两种模式的编程规范和 API 用法、进行模式转换时触发此 skill。提供各模式的完整编写规则、代码模板、硬件约束和检查清单。
---

# TileLang Ascend Developer 与 Expert 模式编程指南

本 skill 提供 Developer 模式和 Expert 模式的完整编程参考。选定模式后，按照对应指南编写算子代码。

---

## 模式概览

| 维度 | Developer 模式 | Expert 模式 |
|------|---------------|-------------|
| **内存分配** | `T.alloc_shared` / `T.alloc_fragment`（编译器自动映射） | `T.alloc_L1` / `T.alloc_ub` / `T.alloc_L0A/L0B/L0C`（显式指定） |
| **计算表达** | `T.Parallel` + 符号运算（`+`, `T.exp` 等） | `T.tile.add` / `T.tile.exp` 等 |
| **执行作用域** | 编译器自动分离 Cube/Vector | 显式 `with T.Scope("C"/"V")` |
| **同步控制** | 自动（pass_configs 开关） | 手动 `T.barrier_all` / `T.set_flag` / `T.wait_flag` |
| **pass_configs** | 需开启自动化开关 | 通常关闭或不设置 |

详细对比与选择建议：[模式对比与选择策略](references/mode-overview.md)

---

## 编写指南

根据选定的模式，查阅对应指南：

### Developer 模式

完整编写规则、API 用法、代码骨架、禁止事项：

→ [Developer 模式编写指南](references/developer-mode-guide.md)

**关键要点速记：**
- pass_configs 4 个开关全开（纯 Vector 只需 2 个）
- 内存用 `alloc_shared` / `alloc_fragment`
- 计算用 `T.Parallel` + 符号运算
- 不写 `T.Scope`、不写同步、不写显式内存层级
- Cube→Vector 数据中转用 workspace tensor

### Expert 模式

完整编写规则、同步原语、双缓冲、高级优化：

→ [Expert 模式编写指南](references/expert-mode-guide.md)

**关键要点速记：**
- 无 pass_configs 或全部 False
- 内存用 `alloc_L1` / `alloc_ub` / `alloc_L0A/L0B/L0C`
- 计算用 `T.tile.*` 操作
- 必须写 `with T.Scope("C")` / `with T.Scope("V")`
- 必须手动管理 `set_flag` / `wait_flag` 同步
- Flag 初始化和清理必须成对

### 混合模式

Developer 主体 + Expert 扩展 API 的编写规则：

→ [混合模式编写指南](references/mixed-mode-guide.md)

**关键要点速记：**
- 使用 Developer 的 pass_configs
- 主体用 `T.Parallel`，特殊操作用 `T.tile.fill` / `T.reduce_*` 等 Expert API 补充

---

## 硬件约束

编写任何模式的算子都需要了解的硬件基础：

→ [硬件架构基础](references/hardware-architecture.md)

**核心约束：**
- 内存层级：GM → L1/UB → L0A/L0B → L0C，**不可跨级访问**
- 每核 2 个 Vector 线程（`vid` = 0 或 1）
- GEMM block 需 16 对齐
- 累加器必须用 `float`（float32）

---

## 代码模板

选定模式后，从对应模板出发编写代码：

| 算子类型 | 适用模式 | 模板 |
|---------|---------|------|
| 纯 Vector（elementwise, activation） | Developer | [template-vector-op](templates/template-vector-op.md) |
| GEMM 矩阵乘法 | Developer | [template-gemm-developer](templates/template-gemm-developer.md) |
| Cube + Vector 融合（matmul_add 等） | Developer | [template-fused-op](templates/template-fused-op.md) |
| Flash Attention | Developer + 混合 | [template-flash-attention](templates/template-flash-attention.md) |
| 高性能 GEMM（双缓冲流水线） | Expert | [template-expert-gemm](templates/template-expert-gemm.md) |

---

## 检查清单

编写完成后，按对应模式的清单逐项检查：

→ [检查清单与常见错误](references/generation-checklist.md)

---

## 模式转换

需要将 Expert 模式代码转换为 Developer 模式（或反向）时：

- [内存分配转换](references/convert-memory.md)
- [计算原语转换](references/convert-compute.md)
- [同步与作用域转换](references/convert-sync-scope.md)
- [pass_configs 配置](references/convert-passconfigs.md)
- [完整转换示例](references/convert-examples.md)

---

## 示例代码位置

- **Developer 模式**：`examples/developer_mode/`
- **Expert 模式**：`examples/gemm/example_gemm_intrinsic.py`、`examples/flash_attention/fa_opt/`
