# T.tile.transpose

## 1. 功能说明

对二维矩阵数据块进行转置：`dst[i][j] = src[j][i]`（i ∈ [0, W)，j ∈ [0, H)，src shape 为 [H, W]，dst shape 为 [W, H]）

## 2. 函数原型

### 2.1 函数定义

```python
def transpose(
    dst: Buffer,
    src: Buffer,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放转置结果，shape 为 `[W, H]` | 张量（tensor） | 必填 |
| src | 输入 | 待转置的源矩阵，shape 为 `[H, W]` | 张量（tensor） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），本 API 仅接受整块 Buffer，不支持切片（BufferRegion）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src |
|------|:---:|:---:|
| Ascend A2 / A3 | float16, int16, uint16, float32, int32, uint32 | float16, int16, uint16, float32, int32, uint32 |

- 其他 dtype（int8、bfloat16、int64 等）通过标量逐元素回退实现，功能正确但性能较低；其中 int64 仅 ascendc 后端支持（pto 后端不支持 8 字节 dtype）

#### 2.3.2 Shape 支持

- 仅支持 2D，`src` shape 为 `[H, W]`，`dst` shape 为 `[W, H]`
- `src` 的 H 和 W 必须为编译期静态值（动态维度不支持）

#### 2.3.3 实现路径

根据 dtype 与 shape 自动分派到不同实现路径：

| 条件 | 实现路径 | 说明 |
|------|---------|------|
| H=16, W=16，B16 dtype（非 bfloat16） | `AscendC::Transpose` 硬件指令 | 单条指令完成，最快路径 |
| H、W 均为 16 的倍数，B16/B32 dtype（非 bfloat16） | `TransDataTo5HD` 分块转置 | 按 16×16 子块逐块转置 |
| 其他（int8、bfloat16，或 H/W 非 16 倍数但满足 32 字节对齐） | 标量逐元素循环 | 功能正确，性能较低 |

### 2.4 约束条件

1. dst 与 src 的 dtype 必须一致
2. dst 的 shape 必须为 `[W, H]`，即 src shape `[H, W]` 的转置
3. src 的 H 和 W 必须满足 32 字节对齐：`H * sizeof(dtype)` 和 `W * sizeof(dtype)` 均为 32 的倍数（硬件约束）。具体地：B16 dtype（float16/int16/uint16/bfloat16）要求 H、W 为 16 的倍数；B32 dtype（float32/int32/uint32）要求 H、W 为 8 的倍数；int8 要求 H、W 为 32 的倍数
4. src 的 H 和 W 必须为编译期静态值
5. src 和 dst 的地址不得重叠（不支持 in-place 转置）
6. 操作数地址需 32 字节对齐（硬件约束）

## 3. 示例代码

**示例 1：16×16 转置（硬件指令路径）**

```python
src = T.alloc_ub((16, 16), "float16")
dst = T.alloc_ub((16, 16), "float16")
T.tile.transpose(dst, src)
```

**示例 2：非方阵转置（分块路径）**

```python
src = T.alloc_ub((64, 32), "float32")
dst = T.alloc_ub((32, 64), "float32")
T.tile.transpose(dst, src)
```

**示例 3：int8 标量回退路径**

```python
src = T.alloc_ub((32, 32), "int8")
dst = T.alloc_ub((32, 32), "int8")
T.tile.transpose(dst, src)
```