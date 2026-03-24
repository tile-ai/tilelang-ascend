# pass_configs 配置说明

## Developer 模式标准配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 Cube/Vector 分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,          # 自动核内同步插入
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,    # 自动内存规划/复用
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,       # 自动核间同步插入
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def my_kernel(...):
    ...
```

## Expert 模式

Expert 模式通常不设 pass_configs，或只保留 `MEMORY_PLANNING`：

```python
@tilelang.jit(out_idx=[-1])  # 无 pass_configs
def my_kernel(...):
    ...
```

## 各开关详解

### TL_ASCEND_AUTO_CV_COMBINE

**功能**：自动将 kernel 中的 Cube 操作和 Vector 操作分离到不同的执行核。

| 场景 | 设置 | 效果 |
|------|------|------|
| Developer 模式 | `True` | 无需手写 `T.Scope("C")` / `T.Scope("V")` |
| Expert 模式 | `False` 或不设 | 需要手写 `T.Scope` |

### TL_ASCEND_AUTO_SYNC

**功能**：自动在数据搬运和计算之间插入 `T.barrier_all()` 等同步指令。

| 场景 | 设置 | 效果 |
|------|------|------|
| Developer 模式 | `True` | 无需手写 `T.barrier_all()` |
| Expert 模式 | `False` 或不设 | 需要手写同步指令 |

### TL_ASCEND_MEMORY_PLANNING

**功能**：自动分析 buffer 生命周期，实现内存复用。

| 场景 | 设置 | 效果 |
|------|------|------|
| Developer 模式 | `True` | 自动复用 buffer 空间，无需 `T.annotate_address` |
| Expert 模式 | 可选 `True` | Expert 模式也可受益；或使用 `T.annotate_address` 手动规划 |

### TL_ASCEND_AUTO_CV_SYNC

**功能**：自动在 Cube Scope 和 Vector Scope 之间插入 `T.set_cross_flag` / `T.wait_cross_flag`。

| 场景 | 设置 | 效果 |
|------|------|------|
| Developer 模式 | `True` | 无需手写核间同步 |
| Expert 模式 | `False` 或不设 | 需要手写 `T.set_cross_flag` / `T.wait_cross_flag` |

## Expert → Developer 转换步骤

在 `@tilelang.jit` 装饰器中添加完整 pass_configs：

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def my_kernel(...):
    ...
```

同时从代码中移除：
- `T.Scope("C")` / `T.Scope("V")`
- `T.barrier_all()`
- `T.set_cross_flag` / `T.wait_cross_flag`
- `T.set_flag` / `T.wait_flag`
- `T.annotate_address`

## 混合模式

当 Developer 模式中混用少量 Expert API（如 `T.tile.fill`）时，仍然使用 Developer 的完整 pass_configs：

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}
```