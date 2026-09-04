# T.tile.reciprocal

## 1. 功能说明

对源操作数逐元素取倒数：`dst[i] = 1 / src0[i]`

## 2. 函数原型

### 2.1 函数定义

```python
def reciprocal(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放倒数运算结果 | 张量（tensor） | 必填 |
| src0 | 输入 | 源操作数 | 张量（tensor） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src0 |
|------|:---:|:----:|
| Ascend A2 / A3 | float16, float32 | float16, float32 |

- 整数 dtype（int16/int32 等）在 Ascend C 后端编译期报错；PTO 后端能编译整数 dtype，但输出会被错误地改写为 float32，结果不可用，均不支持

#### 2.3.2 Shape 支持

- 支持 1D 和 2D
- 支持整行切片（如 `buf[0:32, :]`）；仅计算切片区域内元素，区域外内容未定义
- 不支持 2D 列偏移切片（如 `buf[:, 8:40]`）：真机实测两端均触发 aicore 异常（507015）

### 2.4 约束条件

1. dst 与 src0 的元素总数必须相同（Python 断言，报错信息 "size must be same"）
2. 原地运算（dst 与 src0 为同一 buffer）仅 Ascend C 后端支持；PTO 后端不支持（结果错误），如需原地请在 PTO 下先拷贝一份
3. 操作数地址需 32 字节对齐（硬件约束）
4. 精度（真机实测）：Ascend C 后端为硬件近似实现，最大相对误差约 3e-3（float16/float32 均为 0.3% 量级）；PTO 后端内部用除法实现，结果精确
5. 特殊值遵循 IEEE 语义：`1/0=inf`、`1/±inf=±0`、`1/nan=nan`、`1/-1=-1`

## 3. 示例代码

**示例 1：逐元素取倒数**

```python
src0 = T.alloc_ub((256,), "float16")
dst = T.alloc_ub((256,), "float16")
T.tile.reciprocal(dst, src0)  # dst[i] = 1 / src0[i]
```