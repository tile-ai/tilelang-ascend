# T.reduce_min

## 1. 功能说明

对输入 buffer 沿指定维度求最小值：`out = min(buffer, axis=dim)`

## 2. 函数原型

### 2.1 函数定义

```python
def reduce_min(
    buffer: Buffer | BufferRegion,
    out: Buffer | BufferRegion,
    dim: int = -1,
    *,
    clear: bool = True,
    real_shape: list[int] | None = None,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| buffer | 输入 | 被归约的源数据 | 张量（tensor） | 必填 |
| out | 输出 | 存放归约结果 | 张量（tensor） | 必填 |
| dim | 输入 | 归约轴，取值范围与 buffer 维度相关，详见 [2.3.3 dim 说明](#233-dim-说明) | 整数 | 可选（默认 `-1`） |
| clear | 输入 | 是否在归约前初始化 out。True：归约前将 out 初始化；False：在 out 已有值上比较 | 布尔 | 可选（默认 `True`） |
| real_shape | 输入 | 切片 UB tile 的逻辑 2D shape，用于处理 buffer 物理 shape 与逻辑 shape 不一致的场景 | 整数列表（长度 2）/ None | 可选（默认 `None`） |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | buffer | out |
|------|:------:|:---:|
| Ascend A2 / A3 | float16, float32 | float16, float32 |

#### 2.3.2 Shape 支持

- 支持 1D、2D（3D 暂不支持，精度异常）

#### 2.3.3 dim 说明

dim 的合法取值取决于 buffer 的维度（rank）：

| buffer rank | 合法 dim 值 | 说明 |
|:-----------:|:----------:|------|
| 1D | `0` / `-1` | 等价，对全部元素求最小值 |
| 2D | `0` / `1` / `-1` / `-2` | `0`/`-2`：沿行方向归约；`1`/`-1`：沿列方向归约 |

### 2.4 约束条件

1. buffer 与 out 的 dtype 必须相同
2. out 的 shape 必须匹配归约结果（reduced shape 或 keepdim shape）
3. 仅支持单轴归约（dim 为单个整数），不支持多维同时归约
4. real_shape 若提供，长度必须为 2，且每个元素 ≤ 对应 buffer extent
5. buffer 地址需 32 字节对齐（硬件约束）
6. out 起始地址需 4 字节对齐（float16）/ 8 字节对齐（float32）（硬件约束）
7. 不支持返回最小值索引
8. BufferRegion 切片输入（`buffer[:, a:b]`）仅在 ascendc 后端支持，pto 后端不支持

## 3. 示例代码

**示例 1：1D buffer 全量归约**

```python
src = T.alloc_ub((128,), "float16")
out = T.alloc_ub((1,), "float16")
T.reduce_min(src, out, dim=-1)
```

**示例 2：2D buffer 沿行方向归约**

```python
src = T.alloc_ub((16, 128), "float16")
out = T.alloc_ub((16, 1), "float16")
T.reduce_min(src, out, dim=-1)
```

**示例 3：2D buffer 沿列方向归约**

```python
src = T.alloc_ub((16, 128), "float16")
out = T.alloc_ub((1, 128), "float16")
T.reduce_min(src, out, dim=0)
```

**示例 4：clear=False 累加模式**

```python
src = T.alloc_ub((128,), "float16")
out = T.alloc_ub((1,), "float16")
T.reduce_min(src, out, clear=False)
```
