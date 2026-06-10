# RoPE 算子性能优化最佳实践

本文档总结了 RoPE (Rotary Position Embedding) 算子在 TileLang-Ascend 上的性能优化手段，对比原始实现与优化版本的关键差异。

## 目录

- [优化概览](#优化概览)
- [优化手段详解](#优化手段详解)
- [性能优化总结](#性能优化总结)
- [最佳实践建议](#最佳实践建议)
- [适用场景](#适用场景)
- [参考资料](#参考资料)

---

## 优化概览

| 优化项 | 原始实现 (`rope_mask.py`) | 优化实现 (`rope.py`) | 性能收益 |
|--------|---------------------------|---------------------|---------|
| Mask 生成方式 | CPU 预计算 + GM 搬运 | NPU 动态生成 | 减少 GM 访问 |
| 数据布局 | 外部生成索引 | 内联生成索引 | 消除外部依赖 |
| 内存访问 | 需额外搬运 mask/sin_mask | 零额外搬运 | 降低内存带宽压力 |

---

## 优化手段详解

### 1. NPU 内动态生成 Mask（核心优化）

#### 原始实现（rope_mask.py）

**问题**：在 CPU 端预计算 mask，再搬运到 NPU，增加 GM 访问开销。

```python
# CPU 端预计算（rope_mask.py）
idx = torch.arange(rope_dim * (block_M // 2), dtype=torch.int64, device="cpu")
mask = torch.empty(rope_dim * (block_M // 2), dtype=torch.uint32, device="cpu")
mask[0::2] = idx[1::2].to(torch.uint32)  # 偶数位放奇数索引
mask[1::2] = idx[0::2].to(torch.uint32)  # 奇数位放偶数索引
mask = mask * 4  # 字节偏移

sin_mask = torch.ones(rope_dim, dtype=torch.float32, device=device)
sin_mask[0::2] = -1
```

```python
# Kernel 内搬运（rope_mask.py）
mask_ub = T.alloc_shared([row_per_vec, rope_dim], MASK_DTYPE)
T.copy(mask, mask_ub)  # GM -> UB 搬运开销

sin_mask_ub = T.alloc_shared(rope_dim, ACC_DTYPE)
T.copy(sin_mask, sin_mask_ub)  # GM -> UB 搬运开销
```

#### 优化实现（rope.py）

**收益**：直接在 NPU 上动态生成 mask，消除 GM 访问。

```python
# 1. 生成索引序列 [0, 1, 2, 3, ..., rope_dim-1]
idx_ub = T.alloc_shared([row_per_vec, rope_dim], "int32")
T.tile.createvecindex(idx_ub, 0)  # 向量化索引生成

# 2. 通过 XOR 实现索引交错 [0,1,2,3,...] → [1,0,3,2,...]
tmp_ub_i16 = T.alloc_shared([row_per_vec, rope_dim], "int16")
ones_mask_ub = T.alloc_shared([row_per_vec, rope_dim], "int16")
T.copy(idx_ub, tmp_ub_i16)
T.tile.fill(ones_mask_ub, 1)  # 填充全 1
T.tile.bitwise_xor(mask_ub_i16, tmp_ub_i16, ones_mask_ub)  # idx ^ 1

# 3. 数据类型转换链：int16 → float32 → int32 → uint32
T.copy(mask_ub_i16, mask_ub_f32)
T.copy(mask_ub_f32, mask_ub_i32)
T.tile.mul(mask_ub_i32, mask_ub_i32, 4)  # 乘以 4（字节偏移）
T.reinterpretcast(mask_ub, mask_ub_i32, "uint32_t")  # 位重解释
```

**关键 API**：
- `T.tile.createvecindex`：向量化生成索引序列
- `T.tile.bitwise_xor`：位异或操作，实现索引交错（`idx ^ 1`）
- `T.tile.fill`：填充常量值
- `T.reinterpretcast`：数据类型位重解释，避免转换开销

---

### 2. NPU 内动态生成 sin_mask

#### 原始实现

```python
# CPU 端生成（rope_mask.py）
sin_mask = torch.ones(rope_dim, dtype=torch.float32, device=device)
sin_mask[0::2] = -1
```

```python
# Kernel 内搬运（rope_mask.py）
sin_mask_ub = T.alloc_shared(rope_dim, ACC_DTYPE)
T.copy(sin_mask, sin_mask_ub)
```

#### 优化实现

```python
# NPU 内动态生成（rope.py）
sin_mask_ub = T.alloc_ub(rope_dim, ACC_DTYPE)
T.tile.fill(sin_mask_ub, -1.0)  # 先填充 -1
for i in T.serial(0, rope_dim // 2):
    sin_mask_ub[2 * i + 1] = 1.0  # 奇数位设为 1
```

**优化点**：
- 使用 `T.tile.fill` 向量化填充初始值
- 仅对奇数位进行标量赋值（数量为 `rope_dim // 2`）

---

### 3. 数据类型转换优化

#### 原始实现

```python
# 直接使用 uint32 mask（rope_mask.py）
mask_ub = T.alloc_shared([row_per_vec, rope_dim], MASK_DTYPE)
T.copy(mask, mask_ub)
```

#### 优化实现

```python
# 利用硬件特性的转换链（rope.py）
mask_ub_i16 = T.alloc_shared([row_per_vec, rope_dim], "int16")
mask_ub_f32 = T.alloc_shared([row_per_vec, rope_dim], "float32")
mask_ub_i32 = T.alloc_shared([row_per_vec, rope_dim], "int32")
mask_ub = T.alloc_shared([row_per_vec, rope_dim], MASK_DTYPE)

# int16 → float32 → int32 → uint32
T.copy(mask_ub_i16, mask_ub_f32)
T.copy(mask_ub_f32, mask_ub_i32)
T.tile.mul(mask_ub_i32, mask_ub_i32, 4)
T.reinterpretcast(mask_ub, mask_ub_i32, "uint32_t")
```

**优化点**：
- `int16` XOR 操作效率更高（16 位整数运算）
- `float32` 中间转换用于后续乘法操作（硬件优化路径）
- `T.reinterpretcast` 避免数据拷贝，仅改变类型视图

---

## 性能优化总结

| 维度 | 原始实现 | 优化实现 |
|------|---------|---------|
| **GM 访问次数** | 3 次（x + mask + sin_mask） | 1 次（仅 x） |
| **外部依赖** | 需要 CPU 预计算 | 无外部依赖 |
| **内存带宽** | 需搬运额外 mask | 零额外搬运 |
| **代码复杂度** | CPU+NPU 混合逻辑 | 纯 NPU 逻辑 |

---

## 最佳实践建议

### ✅ 推荐做法

1. **在 NPU 上动态生成小型常量张量**（如 mask、索引）
   - 使用 `T.tile.createvecindex`、`T.tile.fill`、`T.tile.bitwise_xor` 等向量化操作
   - 避免不必要的 GM 访问
2. **优先使用 Tile API 实现向量化操作**
   - `T.tile.createvecindex` 生成索引序列
   - `T.tile.fill` 批量填充常量
   - `T.tile.bitwise_xor` 位运算
3. **合理设计数据类型转换链**
   - 利用 `T.reinterpretcast` 避免拷贝
   - 遵循硬件友好的转换路径

### ❌ 避免做法

1. **避免小型张量的 CPU → NPU 搬运**
   - mask、索引等小型常量应在 NPU 内生成
   - 减少 GM 访问开销

2. **避免外部生成依赖**
   - Kernel 应自包含，减少外部状态
   - 提高可移植性和可维护性

3. **避免非必要的参数传递**
   - 简化 kernel 接口
   - 降低调用开销

---

## 适用场景

本优化方案适用于以下算子：

- ✅ RoPE（Rotary Position Embedding）
- ✅ 需要 mask 或索引操作的算子
- ✅ 数据重排（permute、transpose）
- ✅ 元素交错操作（interleave、gather）

**核心思想**：将小型常量张量的生成从 CPU 转移到 NPU，利用向量化指令消除 GM 访问开销。

---

## 参考资料

- 原始实现：`examples/pos_embedding/rope_mask.py`
- 优化实现：`examples/pos_embedding/rope.py`
- API 参考：`.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/SKILL.md`