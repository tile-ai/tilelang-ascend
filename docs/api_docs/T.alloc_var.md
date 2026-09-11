# T.alloc_var

## 1. 功能说明

分配单元素标量变量（内部 shape 固定为 `[1]`），用于条件标志位、循环计数器、临时标量等场景。

## 2. 函数原型

### 2.1 函数定义

`alloc_var` 支持多种调用形式，统一签名如下：

```python
def alloc_var(
    dtype: str,
    *args,
    scope: str = "local.var",
    init: PrimExpr | int | float | None = None,
) -> Buffer
```

> **多签名说明**：`alloc_var` 的第二个位置参数根据类型自动区分 `init` 和 `scope`，详见 [2.3.3 alloc_var 调用方式](#233-alloc_var-调用方式)。

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dtype | 输入 | 变量的数据类型（如 `"int32"`、`"bool"`、`"float32"`） | 字符串 | 必填 |
| init | 输入 | 初始值，支持常量、表达式或其他 `alloc_var` 变量；为 `None` 时不初始化 | 整数 / 浮点数 / PrimExpr | 可选（默认 `None`） |
| scope | 输入 | 内存作用域 | 字符串 | 可选（默认 `"local.var"`） |

> **返回值说明**：
> - 返回单元素 `T.Buffer` 对象（shape 为 `[1]`），通过索引 `[0]` 或直接赋值访问

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dtype |
|------|:-----:|
| Ascend A2 / A3 | int32, bool, float32, float16, bfloat16, int8, int16, int64, uint32, uint64 |

> 仅 ascendc 支持：uint8, uint16
> pto 后端所有 dtype 均精度验证失败。

#### 2.3.2 Shape 支持

- 固定单元素（内部 shape 为 `[1]`，用户无需指定 shape）

#### 2.3.3 alloc_var 调用方式

`alloc_var` 的第二个位置参数根据类型自动区分 `init` 和 `scope`：

| 调用方式 | 含义 |
|---------|------|
| `T.alloc_var("int32")` | 无初始值 |
| `T.alloc_var("int32", 1)` | init=1 |
| `T.alloc_var("int32", "local.var")` | scope="local.var" |
| `T.alloc_var("int32", 1, "local.var")` | init=1, scope="local.var" |
| `T.alloc_var("int32", init=1)` | init=1（关键字参数） |
| `T.alloc_var("int32", "local.var", init=1)` | init=1, scope="local.var" |

### 2.4 约束条件

1. 必须在 `T.Kernel` 作用域内调用
2. `init` 不可重复指定（位置参数和关键字参数不能同时提供 init）
3. `scope` 必须为字符串类型
4. 最多接受 3 个位置参数（dtype、init、scope）

## 3. 示例代码

**示例 1：分配标量变量**

```python
flag = T.alloc_var("bool", init=False)
counter = T.alloc_var("int32", init=0)
value = T.alloc_var("float32", init=0.0)
```

**示例 2：变量间初始化**

```python
a = T.alloc_var("int32", init=1)
b = T.alloc_var("int32", init=a)
```
