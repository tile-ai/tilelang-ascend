# T.tile.arith_progression

## 1. 功能说明

生成等差数列：`buffer[i] = first_value + i * diff_value`（i = 0, 1, ..., count-1）

## 2. 函数原型

### 2.1 函数定义

```python
def arith_progression(
    buffer: Buffer,
    first_value: PrimExpr,
    diff_value: PrimExpr,
    count: PrimExpr,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| buffer | 输出 | 存放等差数列 | 张量（tensor） | 必填 |
| first_value | 输入 | 等差数列的首个元素值，须与 buffer 的 dtype 一致 | 标量（scalar） | 必填 |
| diff_value | 输入 | 等差数列元素之间的差值（须 ≥ 0），须与 buffer 的 dtype 一致 | 标量（scalar） | 必填 |
| count | 输入 | 等差数列的长度（元素个数），须 > 0 | 整数 | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
> - **scalar**：单个元素值，可以是 buffer 元素访问（BufferLoad）或 Python 标量/表达式（PrimExpr）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | buffer | first_value | diff_value |
|------|:------:|:-----------:|:----------:|
| Ascend A2 / A3 | float16, float32, int16, int32 | 同 buffer | 同 buffer |

> uint16、uint32 仅 pto 后端支持
>
> float16 仅 ascendc 后端支持

#### 2.3.2 Shape 支持

- 支持 1D（count 决定写入元素数，不受 buffer shape 约束；buffer 大小须 ≥ `count * sizeof(dtype)`）

### 2.4 约束条件

1. first_value、diff_value 的数据类型須与 buffer 的元素类型保持一致
2. diff_value 须 ≥ 0（硬件约束，运行时检查；编译期不触发）
3. count 须 > 0（硬件约束，运行时检查；编译期不触发）
4. buffer 的容量必须 ≥ `count * sizeof(dtype)`
5. 仅支持 ND 格式（硬件约束）

#### 2.4.1 pto 后端已知限制

| 限制 | 原因 | 影响 |
|------|------|------|
| diff_value 固定为 1 | pto codegen 的 TCI 调用未传递 diff_value | diff_value ≠ 1 时结果错误（步长始终为 1） |
| count 被忽略 | pto codegen 的 TCI 调用未传递 count | 部分写入（count < buffer size）时写满整个 buffer |

## 3. 示例代码

**示例 1：float16 等差数列，步长 0.5**

```python
buf = T.alloc_ub((256,), "float16")
T.tile.arith_progression(buf, 0.0, 0.5, 256)  # [0.0, 0.5, 1.0, ..., 127.5]
```

**示例 2：int32 等差数列，步长 2**

```python
buf = T.alloc_ub((128,), "int32")
T.tile.arith_progression(buf, 1, 2, 128)  # [1, 3, 5, ..., 255]
```

**示例 3：部分写入 buffer**

```python
buf = T.alloc_ub((512,), "float32")
T.tile.arith_progression(buf, 0.0, 1.0, 300)  # buffer 有 512 空间，只写入前 300 个元素
```
