# T.tile.exp

## 1. 功能说明

对源操作数逐元素执行自然指数运算：`dst[i] = e ^ src0[i]`

## 2. 函数原型

### 2.1 函数定义

```python
def exp(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放指数运算结果 | 张量（tensor） | 必填 |
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
2. dst 可与 src0 为同一 buffer（原地运算，如 `T.tile.exp(a_ub, a_ub)`）
3. 操作数地址需 32 字节对齐（硬件约束）
4. 精度：真机实测 float32 最大相对误差 ≤ 1e-5，float16 ≤ 1e-3（rtol/atol 口径，与测试容差一致）
5. 特殊值遵循 IEEE 语义：`exp(0)=1`、`exp(-inf)=0`、`exp(inf)=inf`、`exp(nan)=nan`、上溢（如 `exp(100)`）结果为 inf

## 3. 示例代码

**示例 1：逐元素指数运算**

```python
src0 = T.alloc_ub((256,), "float16")
dst = T.alloc_ub((256,), "float16")
T.tile.exp(dst, src0)  # dst[i] = e ^ src0[i]
```