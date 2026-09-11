# T.tile.sigmoid

## 1. 功能说明

对源操作数逐元素执行 Sigmoid 激活运算：`dst[i] = 1 / (1 + e^(-src[i]))`

> sigmoid 为独立实现，直接映射到 `tl.ascend_sigmoid` intrinsic，非 `unary_op` 通用分发器路径。

## 2. 函数原型

### 2.1 函数定义

```python
def sigmoid(
    dst: Buffer | BufferRegion,
    src: Buffer | BufferRegion,
    *,
    tmp: Buffer | BufferRegion | None = None,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放 Sigmoid 运算结果 | 张量（tensor） | 必填 |
| src | 输入 | 源操作数 | 张量（tensor） | 必填 |
| tmp | 输入/输出 | 可选显式 UB 临时存储；未提供时由框架自动申请 | 张量（tensor） | 可选（默认 `None`） |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
> - **tmp**：可使用任意固定宽度标量 dtype 声明，lowering 阶段会按所选后端重新解释其存储

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src |
|------|:---:|:---:|
| Ascend A2 / A3 | float16, float32 | float16, float32 |

- 整数 dtype（int16/int32 等）会在编译期报错，不支持

#### 2.3.2 Shape 支持

- 支持 1D 和 2D
- 支持整行切片（如 `buf[0:32, :]`）；仅计算切片区域内元素，区域外内容未定义
- 不支持 2D 列偏移切片（如 `buf[:, 8:40]`）：真机实测两端均触发 aicore 异常（507015）

### 2.4 约束条件

1. dst 与 src 的元素总数应相同（无运行时校验，不匹配时产生未定义结果）
2. dst 与 src 的 dtype 必须一致（Ascend C 约束）
3. 操作数地址需 32 字节对齐（硬件约束）
4. 未提供 `tmp` 时，接口内部使用框架自动申请的临时缓冲区（大小为 `N × sizeof(dtype)` 字节，N 为元素个数），无需用户手动分配
5. 原地运算（dst 与 src 为同一 buffer）仅 ascendc 支持；pto 结果错误（实测最大误差 ≈0.98）
6. 特殊值遵循 IEEE 语义：`sigmoid(0)=0.5`、`sigmoid(-inf)=0`、`sigmoid(inf)=1`、`sigmoid(nan)=nan`

## 3. 示例代码

**示例 1：1D Sigmoid**

```python
src = T.alloc_ub((256,), "float16")
dst = T.alloc_ub((256,), "float16")
T.tile.sigmoid(dst, src)
```

**示例 2：显式指定临时缓冲区**

```python
src = T.alloc_ub((256,), "float16")
dst = T.alloc_ub((256,), "float16")
tmp = T.alloc_ub((256,), "float16")
T.tile.sigmoid(dst, src, tmp=tmp)
```