# T.tile.clear

## 1. 功能说明

将 buffer 中的所有元素清零：`buffer[i] = 0`

## 2. 函数原型

### 2.1 函数定义

```python
def clear(
    buffer: Buffer | BufferRegion,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|---------|
| buffer | 输入/输出 | 待清零的 buffer | 张量（tensor） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）

### 2.3 参数规格

#### 2.3.1 DataType 支持

以下 dtype 基于 Ascend A2 / A3（910B）真机验证：

| dtype | Ascend C | PTO |
|-------|----------|-----|
| float16 | 支持 | 支持 |
| float32 | 支持 | 支持 |
| bfloat16 | 支持 | 支持 |
| int16 | 支持 | 支持 |
| uint16 | 支持 | 支持 |
| int32 | 支持 | 支持 |
| uint32 | 支持 | 支持 |
| int8 | 不支持 | 支持 |
| uint8 | 不支持 | 支持 |

#### 2.3.2 Shape 支持

- 支持 1D 和 2D

### 2.4 约束条件

1. buffer 必须位于 UB 内存（通过 `T.alloc_ub` 分配）
2. buffer 地址需 32 字节对齐（硬件约束）
3. 行宽对齐（当前 Ascend C 后端实现的支持范围，非硬件限制）：对完整的二维
   tile，Ascend C 后端当前要求**每一行**的逻辑数据字节数为 32 的倍数。对 float16、bfloat16、int16、uint16（每元素 2 字节），即 tile 内维元素数需为 16 的倍数；
   对 float32、int32、uint32（每元素 4 字节），需为 8 的倍数。
   整个 tile 总字节数为 32 的倍数并不足够：例如 `(64, 100)` 的 int16 tile 总大小为
   `64 × 100 × 2 = 12800` 字节，但每一行为 `100 × 2 = 200` 字节，不满足行宽
   要求。当前实现对此类输入可能编译并运行成功但返回错误结果，且暂无编译期
   拒绝检查。起始地址对齐（第 2 条）与行宽约束是相互独立的两个条件，需分别满足。
4. size 由 buffer shape 自动推断，无需显式传入 count 参数

## 3. 示例代码

**示例 1：清零 UB 缓冲区**

```python
a_ub = T.alloc_ub((block_M, block_N), "float16")
T.tile.fill(a_ub, 10.0)
T.tile.clear(a_ub)  # 将 a_ub 所有元素清零
```

**示例 2：清零 BufferRegion 切片**

```python
a_ub = T.alloc_ub((block_M, block_N), "float32")
T.tile.fill(a_ub, 10.0)
T.tile.clear(a_ub[0:block_M, 0:block_N])  # 清零指定区域
```
