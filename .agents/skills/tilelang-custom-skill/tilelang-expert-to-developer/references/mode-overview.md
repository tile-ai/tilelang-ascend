# Developer 模式与 Expert 模式概览

## 两种模式的定位

TileLang Ascend 提供两种编程模式，位于编译降级流程的不同层级，可以在同一 kernel 中混合使用。

### Developer 模式（Hardware-Aware with Tile Library）

- **目标用户**：对 AI 芯片内存层次结构有基本了解的开发人员
- **核心理念**：使用抽象化的 Tile Library 接口，编译器自动处理存储映射、同步插入、CV 分离
- **优势**：代码简洁、易于维护、理论上可跨架构平台兼容
- **限制**：无法进行细粒度的硬件控制

### Expert 模式（Hardware-Aware with Thread Primitives）

- **目标用户**：对底层硬件特性（Cube、Vector、MTE、UB 等）有深入理解的专家
- **核心理念**：显式控制存储层级、执行作用域、同步时机
- **优势**：最大灵活性，可针对特定架构进行极致优化
- **限制**：代码复杂度高、平台绑定

## 关键差异对照表

| 维度 | Developer 模式 | Expert 模式 |
|------|---------------|-------------|
| **内存分配** | `T.alloc_shared(shape, dtype)` — 编译器自动映射到 L1 或 UB | `T.alloc_L1()` / `T.alloc_ub()` — 显式指定存储位置 |
| | `T.alloc_fragment(shape, dtype)` — 编译器自动映射到 L0C | `T.alloc_L0A()` / `T.alloc_L0B()` / `T.alloc_L0C()` — 显式指定 |
| **计算表达** | `T.Parallel` + 符号 API（`+`, `T.exp`, `T.max` 等） | `T.tile.add()`, `T.tile.exp()`, `T.tile.max()` 等 |
| **执行作用域** | 无需指定，编译器通过 `AUTO_CV_COMBINE` 自动分离 Cube/Vector | 需要显式 `with T.Scope("C"):` 和 `with T.Scope("V"):` |
| **同步控制** | 自动（`AUTO_SYNC` + `AUTO_CV_SYNC`） | 手动 `T.barrier_all()`, `T.set_flag/wait_flag`, `T.set_cross_flag/wait_cross_flag` |
| **pass_configs** | 需要开启多项自动化开关 | 通常不需要（手动控制一切） |

## 何时选择哪种模式

### 推荐：使用 Developer 模式

- 快速原型开发
- 算法验证阶段
- 追求代码可读性和可维护性
- 不熟悉 Ascend 硬件细节
- 大多数算子开发场景

### 选择 Expert 模式

- 性能关键路径的极致优化
- 需要精确控制流水线同步时机
- 需要使用 `T.annotate_address` 手动规划内存布局
- 需要使用 `T.use_swizzle` 等高级特性
- 需要精确控制 Cube/Vector 核的执行分工

### 混合使用

实践中最常见的是混合编程方式：Developer 模式处理主体逻辑，Expert 模式的扩展接口（如 `T.tile.fill`、`T.tile.cast` 等）用于补充 Developer 模式暂不支持的操作。