# T.tile.round

## 1. 功能说明

将 buffer 中的元素四舍五入到最接近的整数：`dst[i] = round(src[i])`

当小数部分恰好为 0.5 时，采用"向偶数舍入"（banker's rounding）策略。

## 2. 函数原型

### 2.1 函数定义

```python
def round(
    dst: Buffer | BufferRegion,
    src: Buffer | BufferRegion,
    count: PrimExpr,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放舍入运算结果 | 张量（tensor） | 必填 |
| src | 输入 | 源操作数 | 张量（tensor） | 必填 |
| count | 输入 | 参与计算的元素个数 | 整数（integer） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
> - **integer**：正整数表达式（PrimExpr），指定参与计算的元素个数

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src |
|------|:---:|:----:|
| Ascend A2 / A3 | float16, float32 | float16, float32 |
| Ascend A5 | float16, float32 | float16, float32 |

#### 2.3.2 Shape 支持

- 支持 1D 和 2D
- dst 与 src 的元素总数必须相同，count 不超过 dst 和 src 的元素总数

### 2.4 约束条件

1. count 为必填参数，需显式指定参与计算的元素个数（不同于多数 tile API 自动推断 size）
2. dst 与 src 的 dtype 必须相同
3. 仅支持 float16 和 float32 两种 dtype
4. 只支持四舍五入到整数，不支持四舍五入到小数位
5. 小数部分为 0.5 时向偶数舍入（banker's rounding）：3.5 舍入到 4，2.5 舍入到 2
6. 操作数地址需 32 字节对齐（硬件约束）
7. Ascend C 后端的 Round 为高阶 API，需要临时空间（tmpBuffer），TileLang 未暴露 sharedTmpBuffer 参数
8. 接口内部使用框架自动申请的临时缓冲区（大小为 `max(256, N × sizeof(dtype))` 字节，N 为元素个数），无需用户手动分配

## 3. 示例代码

**示例 1：四舍五入**

```python
src = T.alloc_ub((1024,), "float16")
dst = T.alloc_ub((1024,), "float16")
T.tile.round(dst, src, 1024)  # banker's rounding: 3.5→4, 2.5→2
```

**示例 2：量化前的舍入**

```python
values = T.alloc_ub((256,), "float32")
rounded = T.alloc_ub((256,), "float32")
T.tile.round(rounded, values, 256)  # 舍入到整数后再 cast 到 int32
```
