# Bug 分析：Buffer.__getitem__ 符号化 slice 边界导致 int() 转换失败

## 1. 问题概述

在 TVMScript 解析阶段，当函数调用参数中包含带**符号化边界**的 slice（如 `buf[row, fill_len:kv_size]`，其中 `fill_len` 是 `T.if_then_else` 返回的 `tir.Sub` 表达式）时，`Buffer.__getitem__` 错误地走了 `BufferLoad + Ramp` 路径，对符号表达式调用 `int()` 导致崩溃。

**错误信息**：

```
error: int() argument must be a string, a bytes-like object or a real number, not 'Sub'
 --> examples/sparse-fa/sparse_fa_v4.py:196:49
     |
 196 |   T.tile.fill(acc_s_ub[row, fill_len:kv_size], NEG_INF)
     |               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

## 2. 触发场景

### 2.1 触发条件

同时满足以下条件时触发：

1. **在 `@T.prim_func` 内部**（TVMScript 解析上下文）
2. **函数调用的参数**中包含 buffer 的 slice 索引（如 `some_func(buf[a:b], ...)`）
3. **slice 的边界是符号表达式**（如 `tir.Var`、`tir.Sub`、`tir.Add` 等 `PrimExpr` 子类）

### 2.2 触发代码示例

```python
@T.prim_func
def main(Q: T.Tensor([total_q, heads, dim], "bfloat16"), ...):
    fill_len = T.if_then_else(raw_len < kv_size, raw_len, kv_size)
    # fill_len 的类型是 tir.Call（PrimExpr 子类），不是 int

    # 触发点：在函数调用参数中使用符号化 slice
    T.tile.fill(acc_s_ub[row, fill_len:kv_size], NEG_INF)
    #           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #           这个表达式作为 T.tile.fill 的参数被解析
```

### 2.3 不触发的场景

```python
# 场景 A：slice 边界是常量 → 不触发
T.tile.fill(acc_s_ub[row, 0:128], NEG_INF)

# 场景 B：不在函数调用参数中（直接赋值）→ 不触发
acc_s_ub[row, fill_len:kv_size] = NEG_INF

# 场景 C：slice 边界是 tir.Var（非复合表达式）→ 不触发
# （因为 tir.Var 可能恰好通过 int() 转换）
```

## 3. 错误原因

### 3.1 根因：两个组件对 `step=1` 的语义理解不一致

这是 **TVM 内部实现的 bug**，涉及两个组件之间的契约不一致。

#### 组件 A：TVMScript Evaluator（AST 预处理器）

**文件**：`3rdparty/tvm/python/tvm/script/parser/core/evaluator.py:193-213`

```python
for arg in args:
    if isinstance(arg, doc.Subscript) and isinstance(arg.slice, (doc.Slice, doc.Tuple)):
        # ...
        for s in check_slices:
            if not s.step and s.upper and s.lower:
                s.step = doc.Constant(1, ...)  # ← 自动注入 step=1
```

当解析函数调用的参数时，evaluator 会对所有**没有显式 step 的 slice** 自动注入 `step=1`。

**设计意图**：`step=1` 等同于"没有 step"，只是补一个默认值。

#### 组件 B：Buffer.__getitem__（索引解析器）

**文件**：`3rdparty/tvm/python/tvm/tir/buffer.py:187-190`（修复前）

```python
has_slice = any(isinstance(i, slice) for i in indices)
has_step = any(isinstance(i, slice) and i.step is not None for i in indices)
#                                        ^^^^^^^^^^^^^^^^^
#                                        step=1 时 is not None → True

if has_slice and not has_step:
    # 路径 1：返回 BufferRegion（正确路径）
    ...
else:
    # 路径 2：返回 BufferLoad + Ramp（错误路径）
    lanes = analyzer.simplify((stop - start + step - 1) // step)
    expr_indices.append(Ramp(start, step, int(lanes)))  # 💥 int(Sub) 失败
```

`__getitem__` 用 `i.step is not None` 判断"是否有 step"。当 evaluator 注入了 `step=1` 后，`step` 不是 `None`，所以 `has_step = True`，导致走了 `BufferLoad + Ramp` 路径。

### 3.2 完整调用链

```
用户代码: T.tile.fill(acc_s_ub[row, fill_len:kv_size], NEG_INF)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    函数调用参数中的 subscript

  ↓ TVMScript Parser 解析函数调用

evaluator.py:_visit()
  ↓ 检测到 doc.Call 的参数中有 doc.Subscript + doc.Slice
  ↓ 发现 slice 没有 step → 自动注入 step=1
  ↓ slice 变为: fill_len:kv_size:1

  ↓ eval 执行 acc_s_ub[row, fill_len:kv_size:1]

buffer.py:__getitem__((row, slice(fill_len, kv_size, 1)))
  ↓ has_slice = True  (有 slice)
  ↓ has_step  = True  (step=1 is not None)  ← 错误判断
  ↓ 进入 else 分支（BufferLoad + Ramp 路径）
  ↓ lanes = (kv_size - fill_len + 1 - 1) // 1 = kv_size - fill_len
  ↓ lanes 的类型是 tir.Sub（符号表达式）
  ↓ int(lanes) → TypeError: int() argument must be ... not 'Sub'
```

### 3.3 两条路径的语义差异

| 维度 | BufferRegion 路径 | BufferLoad + Ramp 路径 |
|------|-------------------|----------------------|
| 条件 | `has_slice and not has_step` | 其他情况 |
| 返回类型 | `BufferRegion`（区域引用） | `BufferLoad`（元素加载） |
| 用途 | 传给 `T.copy`、`T.tile.fill` 等 | 标量/向量化加载 |
| 对符号边界的处理 | `Range(min, extent)` 支持符号 | `Ramp(start, step, int(lanes))` 要求常量 lanes |

## 4. 修复方案

### 4.1 修改内容

**文件**：`3rdparty/tvm/python/tvm/tir/buffer.py:188-190`

```python
# 修复前
has_step = any(isinstance(i, slice) and i.step is not None for i in indices)

# 修复后
has_step = any(
    isinstance(i, slice) and i.step is not None and i.step != 1 for i in indices
)
```

### 4.2 修复逻辑

增加 `i.step != 1` 判断，使 `step=1` 等同于"没有 step"：

- `step=None` → 无 step → `has_step = False`
- `step=1`（evaluator 注入）→ 等同于无 step → `has_step = False`
- `step=2`（用户显式指定）→ 有 step → `has_step = True`

这样 `fill_len:kv_size:1` 会正确走 BufferRegion 路径，`Range` 构造支持符号表达式，不会触发 `int()` 转换。

### 4.3 影响范围

- **不影响**：常量 slice（如 `buf[0:128]`），无论走哪条路径都能正确工作
- **不影响**：用户显式指定 `step != 1` 的情况（如 `buf[0:128:2]`），仍走 Ramp 路径
- **修复**：符号化 slice 边界在函数调用参数中的使用

## 5. 关联问题

修复此 bug 后，`fill` 函数中 `math.prod(buffer_extent)` 也可能遇到符号表达式问题（当 extent 包含 `tir.Sub` 等），需同步处理：

**文件**：`tilelang/language/ascend_tile.py`

```python
# _handle_buffer_region 中
extent = [x.extent for x in br.region]
if any(isinstance(e, PrimExpr) for e in extent):
    size_extent = reduce(operator.mul, extent)   # 支持符号表达式
else:
    size_extent = math.prod(extent)              # 纯数值

# fill 函数中同理
if any(isinstance(e, PrimExpr) for e in buffer_extent):
    size = reduce(operator.mul, buffer_extent)
else:
    size = math.prod(buffer_extent)
```

## 6. 总结

| 项目 | 内容 |
|------|------|
| Bug 类型 | TVM 内部实现 bug（组件间契约不一致） |
| 根因 | evaluator 注入 `step=1`，`__getitem__` 将其误判为"有 step" |
| 影响 | 函数调用参数中使用符号化 slice 边界的 buffer 索引 |
| 修复 | `buffer.py:188` 增加 `i.step != 1` 判断 |
| 修复文件 | `3rdparty/tvm/python/tvm/tir/buffer.py` |
| 关联修复 | `tilelang/language/ascend_tile.py` 中 `math.prod` → `reduce(operator.mul)` |
