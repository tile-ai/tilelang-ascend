# T.tile.mul_add_dst

## 1. 功能说明

将 src0 与 src1 逐元素相乘，再与 dst 中的现有值相加，结果写回 dst：`dst[i] = src0[i] * src1[i] + dst[i]`

dst 同时作为输入（累加器）和输出，这是该 API 与普通 `mul` + `add` 组合的关键区别。

## 2. 函数原型

### 2.1 函数定义

```python
def mul_add_dst(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    src1: Buffer | BufferRegion,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输入/输出 | 目标 buffer，同时作为累加器输入和输出，必须位于 UB scope | 张量（tensor） | 必填 |
| src0 | 输入 | 乘法第一个源操作数 | 张量（tensor） | 必填 |
| src1 | 输入 | 乘法第二个源操作数 | 张量（tensor） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src0 | src1 |
|------|:---:|:----:|:----:|
| Ascend A2 / A3 | float16, float32 | float16, float32 | float16, float32 |
| Ascend A5 | float16, float32, int64, uint64 | float16, float32, int64, uint64 | float16, float32, int64, uint64 |

> **混合精度说明**：支持 dst=float32 + src0/src1=float16 的组合（half 乘法结果累加到 float 累加器）。

> **后端差异说明**：A5 平台上 int64/uint64 仅 Ascend C 后端支持。

#### 2.3.2 Shape 支持

- 支持 1D 和 2D
- dst、src0、src1 的元素总数必须相同

### 2.4 约束条件

1. dst 为 read-write 语义：调用前 dst 必须包含有效数据，调用后 dst 内容被原地修改
2. dst、src0、src1 的元素总数必须相同，否则 assert 失败
3. dst 必须位于 UB（Unified Buffer）scope
4. 当 src dtype 为 float16、dst dtype 为 float32 时，不支持地址重叠
5. MulAddDst 受 bank 冲突影响：地址不重叠时只能达到一半理论并行度
6. 操作数地址需 32 字节对齐（硬件约束）
7. 接口内部使用框架自动申请的临时缓冲区（大小与 dst 相同），无需用户手动分配

## 3. 示例代码

**示例 1：融合乘加累加**

```python
a_ub = T.alloc_ub((64, 128), "float16")
b_ub = T.alloc_ub((64, 128), "float16")
c_ub = T.alloc_ub((64, 128), "float32")  # dst 必须预先包含有效数据
T.tile.mul_add_dst(c_ub, a_ub, b_ub)  # c_ub = a_ub * b_ub + c_ub
```

**示例 2：Softmax 中的乘加**

```python
acc_s_ub = T.alloc_ub((block_M, block_N), "float16")
acc_s_ub_ = T.alloc_ub((block_M, block_N), "float16")
acc_o_ub = T.alloc_ub((block_M, block_N), "float16")
T.tile.mul_add_dst(acc_o_ub, acc_s_ub, acc_s_ub_)  # acc_o = acc_s * acc_s_ + acc_o
```
