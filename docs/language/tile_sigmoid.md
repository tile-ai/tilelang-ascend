# T.tile.sigmoid

## 1. 功能说明

对操作数逐元素执行 Sigmoid 激活运算：`dst[i] = 1 / (1 + e^(-src[i]))`

> sigmoid 为独立实现，直接映射到 `tl.ascend_sigmoid` intrinsic，非 `unary_op` 通用分发器路径。

## 2. 函数原型

### 2.1 函数定义

```python
def sigmoid(
    dst: Buffer | BufferRegion,
    src: Buffer | BufferRegion,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放 Sigmoid 运算结果 | 张量（tensor） | 必填 |
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

1. dst 与 src 的元素总数必须相同
2. dst 与 src 的 dtype 必须一致（Ascend C 硬件约束）
3. src 与 dst 的地址不能重叠
4. 操作数地址需 32 字节对齐（硬件约束）
5. 接口内部使用框架自动申请的临时缓冲区（大小为 `N × sizeof(dtype)` 字节，N 为元素个数），无需用户手动分配

## 3. 示例代码

**示例 1：1D Sigmoid**

```python
src = T.alloc_ub((256,), "float16")
dst = T.alloc_ub((256,), "float16")
T.tile.sigmoid(dst, src)
```

**示例 2：2D Sigmoid**

```python
src = T.alloc_ub((128, 256), "float16")
dst = T.alloc_ub((128, 256), "float16")
T.tile.sigmoid(dst, src)
```
