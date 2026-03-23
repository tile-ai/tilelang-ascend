# 同步原语

## 概述

TileLang 提供核内流水线同步和核间同步两类原语，用于协调不同执行管线和 AI Core 之间的数据依赖。

## 核内同步

### T.set_flag(src, dst, eventId)

设置核内流水线同步标志，表示源管线（producer）已完成任务。

**参数**：
- `src`：源管线（"fix", "mte1", "mte2", "mte3", "m", "v"）
- `dst`：目的管线
- `eventId`：事件 ID

### T.wait_flag(src, dst, eventId)

等待核内流水线同步标志，阻塞目的管线直到源管线完成。

**参数**：与 `T.set_flag` 相同

**示例**：
```python
# MTE2 完成数据搬运后通知 Vector
T.set_flag("mte2", "v", 0)
T.wait_flag("mte2", "v", 0)
```

### T.barrier_all()

所有管线的全局屏障。确保所有管线（Scalar, Vector, Cube, MTE 等）在此之前的指令全部完成。

```python
T.barrier_all()
```

### T.pipe_barrier(pipe)

特定管线的屏障。

```python
T.pipe_barrier("v")     # 等待 Vector 管线完成
T.pipe_barrier("mte3")  # 等待 MTE3 管线完成
```

### T.sync_all()

全局同步，确保计算单元（块/核心）内的内存一致性和执行同步。

```python
T.sync_all()
```

## 核间同步

### T.set_cross_flag(pipe, flag, mode=2)

设置核间同步标志。

**参数**：
- `pipe`：发出 set 操作的管线（如 "MTE3", "V"）
- `flag`：事件 ID 索引
- `mode`：同步模式
  - 0：所有 AIC 或所有 AIV 之间
  - 1：同组内所有 AIV 之间
  - 2：同组内 AIC 和 AIV 之间（默认）

### T.wait_cross_flag(flag, pipe="")

等待核间同步标志。

**参数**：
- `flag`：事件 ID 索引
- `pipe`：等待的管线（仅 A5 平台支持，其他架构必须为空字符串）

**示例**：
```python
# Cube 核完成计算后通知 Vector 核
T.set_cross_flag("MTE3", 0)
T.wait_cross_flag(0)
```

## 最佳实践

1. **优先使用自动同步**：开启 `TL_ASCEND_AUTO_SYNC: True`，让编译器自动插入同步
2. **手动同步用于 Expert 模式**：需要精确控制流水线时才手动插入同步原语
3. **set_flag/wait_flag 成对使用**：确保每个 set 都有对应的 wait
4. **barrier_all 开销较大**：尽量使用更细粒度的 `pipe_barrier` 或 `set_flag/wait_flag`
5. **核间流水线需要配合同步**：使用 `T.Pipelined` 的核间模式时，需开启 `tl.ascend_auto_cross_core_sync: True`
