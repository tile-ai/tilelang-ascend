# 内存分配原语

## 概述

TileLang 对存储层级进行了抽象，分为 Global、shared 和 fragment 三个级别。片上存储分配是 kernel 开发的关键环节。

## 存储层级对应关系（Ascend 平台）

| TileLang 层级 | Ascend 硬件存储 | 用途 |
|--------------|----------------|------|
| Global | HBM (Global Memory) | 主存，容量大但带宽相对较低 |
| shared | L1 Buffer / Unified Buffer (UB) | 片上高速缓存，编译器自动区分 |
| fragment | L0A / L0B / L0C Buffer | 寄存器级存储，Cube 计算单元的输入/输出 |

## Developer 模式 API

### T.alloc_shared(shape, dtype)

分配 shared 层级的存储空间。编译器根据上下文自动判断分配到 L1 Buffer 还是 Unified Buffer。

```python
A_L1 = T.alloc_shared((block_M, block_K), dtype)      # 用于 Cube 计算时分配到 L1
a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)  # 用于 Vector 计算时分配到 UB
```

### T.alloc_fragment(shape, dtype)

分配 fragment 层级的存储空间。编译器根据上下文自动判断分配到 L0A、L0B 还是 L0C。

```python
C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)
```

## Expert 模式 API

显式指定存储位置，适用于需要精确控制内存分配的场景。

### T.alloc_ub(shape, dtype)

在 Unified Buffer 中分配存储。

```python
a_ub = T.alloc_ub((block_M, block_N), 'int16')
```

### T.alloc_L1(shape, dtype)

在 L1 Buffer 中分配存储。

```python
A_L1 = T.alloc_L1((block_M, K_L1), dtype)
```

### T.alloc_L0A(shape, dtype)

在 L0A Buffer 中分配存储（Cube 计算左矩阵）。

```python
A_L0 = T.alloc_L0A((block_M, K_L1), dtype)
```

### T.alloc_L0B(shape, dtype)

在 L0B Buffer 中分配存储（Cube 计算右矩阵）。

```python
B_L0 = T.alloc_L0B((block_M, K_L1), dtype)
```

### T.alloc_L0C(shape, dtype)

在 L0C Buffer 中分配存储（Cube 计算输出）。

```python
C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)
```

## 最佳实践

1. **优先使用 Developer 模式 API**（`T.alloc_shared` / `T.alloc_fragment`），让编译器自动选择存储位置
2. **VEC_NUM 切分**：Vector 计算时，每个 AI Core 有 2 个 Vector 单元，通常使用 `block_M // VEC_NUM` 作为 UB 分配的第一维
3. **开启自动内存规划**：`TL_ASCEND_MEMORY_PLANNING: True`，避免手动管理复杂的内存复用
4. **注意容量限制**：片上存储容量有限，合理设计 block 大小以避免内存溢出
