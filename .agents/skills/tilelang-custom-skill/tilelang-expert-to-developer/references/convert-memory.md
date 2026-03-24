# 内存分配转换规则

## 转换映射表

### Expert → Developer

| Expert 模式 | Developer 模式 | 说明 |
|-------------|---------------|------|
| `T.alloc_L1(shape, dtype)` | `T.alloc_shared(shape, dtype)` | 编译器自动识别为 L1 |
| `T.alloc_ub(shape, dtype)` | `T.alloc_shared(shape, dtype)` | 编译器自动识别为 UB |
| `T.alloc_L0C(shape, dtype)` | `T.alloc_fragment(shape, dtype)` | 编译器自动识别为 L0C |
| `T.alloc_L0A(shape, dtype)` | 通常不需要显式分配 | L1→L0A 搬运由编译器处理 |
| `T.alloc_L0B(shape, dtype)` | 通常不需要显式分配 | L1→L0B 搬运由编译器处理 |

## 编译器如何判断 alloc_shared 映射到 L1 还是 UB？

编译器根据 **buffer 的使用上下文** 自动判断：

- 如果 buffer 被用作 `T.gemm_v0/v1` 的输入 → 映射到 **L1 Buffer**
- 如果 buffer 被用于 `T.Parallel` 循环体的计算 → 映射到 **Unified Buffer**
- 如果 buffer 被用于 `T.tile.xxx` 操作 → 映射到 **Unified Buffer**

## 示例：GEMM 内存分配转换

### Expert 模式
```python
A_L1 = T.alloc_L1((block_M, K_L1), dtype)           # 显式 L1
B_L1 = T.alloc_L1((K_L1, block_N), dtype)           # 显式 L1
C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype) # 显式 L0C
```

### Developer 模式
```python
A_L1 = T.alloc_shared((block_M, K_L1), dtype)       # 自动 → L1
B_L1 = T.alloc_shared((K_L1, block_N), dtype)       # 自动 → L1
C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)  # 自动 → L0C
```

## 示例：Vector 计算内存分配转换

### Expert 模式
```python
a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)  # 显式 UB
b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
```

### Developer 模式
```python
a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)  # 自动 → UB
b_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
c_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
```

## 注意事项

1. **Developer 模式不需要 L0A/L0B**：`T.gemm_v0` 会自动处理 L1→L0A/L0B 的搬运
2. **混合模式**：Developer 模式代码中可以混用 `T.alloc_ub` 等 Expert API，编译器能正确处理
3. **MEMORY_PLANNING**：开启后编译器自动复用 buffer 空间，无需 `T.annotate_address`