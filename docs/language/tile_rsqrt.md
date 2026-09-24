# T.tile.rsqrt

## 1. 功能说明

对源操作数逐元素求平方根的倒数：`dst[i] = 1 / sqrt(src0[i])`

## 2. 函数原型

### 2.1 函数定义

```python
def rsqrt(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放平方根倒数结果 | 张量（tensor） | 必填 |
| src0 | 输入 | 源操作数 | 张量（tensor） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src0 |
|------|:---:|:----:|
| Ascend A2 / A3 | float16, float32 | float16, float32 |

- 整数 dtype（int16/int32 等）会在编译期报错，不支持

#### 2.3.2 Shape 支持

- 支持 1D 和 2D
- 支持整行切片（如 `buf[0:32, :]`）；仅计算切片区域内元素，区域外内容未定义
- 不支持 2D 列偏移切片（如 `buf[:, 8:40]`）：真机实测两端均触发 aicore 异常（507015）

### 2.4 约束条件

1. dst 与 src0 的元素总数必须相同（Python 断言，报错信息 "size must be same"）
2. dst 可与 src0 为同一 buffer（原地运算，如 `T.tile.rsqrt(a_ub, a_ub)`）
3. 操作数地址需 32 字节对齐（硬件约束）
4. 精度（真机实测）：Ascend C 与 PTO 后端均为硬件近似实现，最大相对误差约 3e-3（0.3% 量级）
5. 若对精度要求较高，推荐组合 `T.tile.sqrt + T.tile.div`（`1/sqrt(x) = sqrt(x)/x`）：实测最大相对误差降至 float32 约 1.8e-7、float16 约 6.7e-4（参考 issue #1225 相关讨论）
6. 特殊值遵循 IEEE 语义：`rsqrt(0)=inf`、`rsqrt(负数)=nan`、`rsqrt(inf)=0`、`rsqrt(nan)=nan`

## 3. 示例代码

**示例 1：逐元素平方根倒数**

```python
src0 = T.alloc_ub((256,), "float16")
dst = T.alloc_ub((256,), "float16")
T.tile.rsqrt(dst, src0)  # dst[i] = 1 / sqrt(src0[i])
```
