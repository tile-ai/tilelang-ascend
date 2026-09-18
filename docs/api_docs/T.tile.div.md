# T.tile.div

## 1. 功能说明

对两个操作数逐元素执行除法：`dst[i] = src0[i] / src1[i]`

## 2. 函数原型

### 2.1 函数定义

```python
def div(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    src1: Buffer | BufferRegion | BufferLoad | PrimExpr,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放运算结果 | 张量（tensor） | 必填 |
| src0 | 输入 | 第一个源操作数（被除数） | 张量（tensor） | 必填 |
| src1 | 输入 | 第二个源操作数（除数），支持 tensor 或 scalar；除数为 0 时行为未定义 | 张量（tensor）/ 标量（scalar） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
> - **scalar**：单个元素值，可以是 buffer 元素访问（BufferLoad，仅支持 1D 单索引）或 Python 标量/表达式（PrimExpr）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src0 | src1 |
|------|:---:|:----:|:----:|
| Ascend A2 / A3 | float16, float32 | float16, float32 | 同 dst |

- 仅支持浮点 dtype；整数 dtype（int16/int32）全部形式均不支持：ascendc 编译失败，pto 的 scalar/BufferLoad 形式可编译但结果错误
- src1 为 tensor 时，dtype 必须与 dst 一致；src1 为 scalar 时自动转换为 buffer 的 dtype

#### 2.3.2 Shape 支持

- 支持 1D 和 2D
- 支持整行切片（如 `buf[2, :]`）及覆盖完整最后一维的连续多行区域（如 `buf[0:2, :]`）
- 更高维 buffer 需通过切片降维为 1D/2D 的切片传入

### 2.4 约束条件

1. dst 与 src0 的大小必须一致
2. src1 为切片（BufferRegion）时，其大小必须与 dst 一致
3. src1 为 Buffer 时大小不做校验：小于 dst 时产生越界读取，大于 dst 时仅前 dst 大小个元素参与运算（不报错）
4. dst、src0、src1（tensor 形式）的 dtype 必须一致；dtype 不一致会在编译期报错
5. 操作数地址需 32 字节对齐（硬件约束）
6. 仅支持整行/整 buffer 的连续区域；2D 列偏移切片（如 `buf[0, 8:40]`）会产生错误结果或触发 aicore 异常（507015），不支持
7. src1 为 buffer 元素访问时仅支持 1D 单索引（如 `buf[i]`）；多维元素访问只取第一个索引，结果错误（实测行为，两后端一致）
8. 整数 dtype 不支持（硬件约束，`AscendC::Div` 的 static_assert 仅 half/float）
9. scalar/BufferLoad 除法通过乘以倒数实现（`Muls(dst, src0, 1.0f / src1)`）：2 的幂次除数精确，其余除数引入舍入误差
10. 标量仅支持作为 src1（右操作数）：除法不可交换，`标量 / Buffer` 形式（如 `2.0 / buf`）当前无法表达；Ascend C `Divs` 接口原生支持（flexible scalar），TileLang 前端暂未暴露该形式（缺口记录）

## 3. 示例代码

**示例 1：tensor-tensor 除法**

```python
src0 = T.alloc_ub((256,), "float16")
src1 = T.alloc_ub((256,), "float16")
dst  = T.alloc_ub((256,), "float16")
T.tile.div(dst, src0, src1)
```

**示例 2：tensor-scalar 除法**

```python
src0 = T.alloc_ub((256,), "float16")
dst  = T.alloc_ub((256,), "float16")
T.tile.div(dst, src0, 2.0)  # dst[i] = src0[i] / 2.0
```

**示例 3：tensor-BufferLoad 除法（Flash Attention 归一化模式）**

```python
acc_o = T.alloc_ub((128, 64), "float16")
sumexp = T.alloc_ub((128,), "float16")
for h_i in range(128):
    T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])  # acc_o[h_i, :] /= sumexp[h_i]
```