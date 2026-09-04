# T.tile.cos

## 1. 功能说明

对操作数逐元素执行余弦运算：`dst[i] = cos(src[i])`

> cos 为独立实现，直接映射到 `tl.ascend_cos` intrinsic。Ascend C 后端使用高阶 API，内部需要临时空间（tmpBuffer）存储中间变量。PTO 后端不支持 sin/cos（无原生 PTO-ISA 指令）。

## 2. 函数原型

### 2.1 函数定义

```python
def cos(
    dst: Buffer | BufferRegion,
    src: Buffer | BufferRegion,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放余弦运算结果 | 张量（tensor） | 必填 |
| src | 输入 | 源操作数 | 张量（tensor） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src |
|------|:---:|:---:|
| Ascend A2 / A3 | float16, float32 | float16, float32 |
| Ascend A5 | float16, float32 | float16, float32 |

#### 2.3.2 Shape 支持

- 支持 1D 和 2D

### 2.4 约束条件

1. dst 与 src 的元素总数必须相同，否则 assert 失败
2. dst 与 src 的 dtype 必须一致（Ascend C 约束）
3. 不支持源操作数与目的操作数地址重叠（Ascend C 约束）
4. 操作数地址需 32 字节对齐（硬件约束）
5. A2/A3 平台输入值域必须在 [-65504.0, 65504.0] 范围内（POLYNOMIAL_APPROXIMATION 算法限制），超出此范围结果不可预测
6. PTO 后端不支持 sin/cos，使用 PTO 后端编译含 cos 的 kernel 会产生无效代码
7. 接口内部使用框架自动申请的临时缓冲区（大小为 `2 × N × sizeof(dtype)` 字节，N 为元素个数），无需用户手动分配

## 3. 示例代码

**示例 1：1D 余弦运算**

```python
src = T.alloc_ub((1024,), "float16")
dst = T.alloc_ub((1024,), "float16")
T.tile.cos(dst, src)
```

**示例 2：2D 余弦运算**

```python
src = T.alloc_ub((128, 64), "float16")
dst = T.alloc_ub((128, 64), "float16")
T.tile.cos(dst, src)
```
