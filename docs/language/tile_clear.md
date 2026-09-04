# T.tile.clear

## 1. 功能说明

将 buffer 中的所有元素清零：`buffer[i] = 0`

内部委托给 `fill(buffer, 0)` 实现，不生成独立的底层指令。

## 2. 函数原型

### 2.1 函数定义

```python
def clear(
    buffer: Buffer | BufferRegion | tir.Var,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| buffer | 输入/输出 | 待清零的 buffer | 张量（tensor） / 变量（var） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
> - **var**：tir.Var 变量，当传入 tir.Var 时，clear 会通过 `T.has_let_value` / `T.get_let_value` 解析为 BufferRegion 再调用 fill

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | buffer |
|------|:------:|
| Ascend A2 / A3 | 所有已支持的 dtype（float16, float32, bfloat16, int8, uint8, int16, uint16, int32, uint32） |
| Ascend A5 | 所有已支持的 dtype（float16, float32, bfloat16, int8, uint8, int16, uint16, int32, uint32, int64, uint64） |

> **说明**：clear 委托给 fill 实现，dtype 支持范围与 fill 相同。fill 的底层指令（Duplicate / TEXPANDS）支持所有常见 dtype。

> **后端差异说明**：A2/A3 平台上 bfloat16 仅 Ascend C 后端支持；int8/uint8 仅 PTO 后端支持。

#### 2.3.2 Shape 支持

- 支持 1D 和 2D
- size 由 buffer shape 自动推断（BufferRegion 时取 region extent 的乘积，Buffer 时取 shape 的乘积）

### 2.4 约束条件

1. clear 委托给 fill(buffer, 0) 实现，不生成独立的底层指令
2. buffer 地址需 32 字节对齐（硬件约束）
3. size 由 buffer shape 自动推断，无需显式传入 count 参数
4. 当传入 tir.Var 时，需确保该变量已通过 `T.has_let_value` 绑定到有效的 BufferRegion

## 3. 示例代码

**示例 1：清零累加器**

```python
acc_s_ub = T.alloc_ub((block_M, block_N), "float16")
T.tile.clear(acc_s_ub)  # 将 acc_s_ub 所有元素清零
```

**示例 2：清零 L0C 累加器**

```python
C_L0 = T.alloc_L0C((block_M, block_N), "float32")
T.tile.clear(C_L0)  # 清零 L0C 累加器
```
