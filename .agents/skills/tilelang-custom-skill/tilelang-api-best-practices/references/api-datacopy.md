# 数据搬运原语

## 概述

`T.copy` 是 TileLang 中统一的数据搬运接口，支持在不同内存层级之间拷贝 tile 数据块。

## API

### T.copy(src, dst)

将数据从 src 搬运到 dst 所在的存储空间。

**参数**：
- `src`：源数据 buffer（支持 tir.Buffer、BufferLoad、BufferRegion）
- `dst`：目的数据 buffer

**支持的搬运路径**：

| src | dst | 说明 |
|-----|-----|------|
| GM | L1 | Global Memory → L1 Buffer |
| L1 | L0A | L1 Buffer → L0A Buffer（Cube 左矩阵）|
| L1 | L0B | L1 Buffer → L0B Buffer（Cube 右矩阵）|
| L0C | GM | L0C Buffer → Global Memory |
| GM | UB | Global Memory → Unified Buffer |
| UB | GM | Unified Buffer → Global Memory |
| UB | UB | Unified Buffer → Unified Buffer |
| UB | L1 | Unified Buffer → L1 Buffer |

## 使用示例

### GM → shared (L1/UB)

```python
# 从 Global Memory 搬运到 L1（用于 Cube 计算）
T.copy(A[bx * block_M, k * block_K], A_L1)

# 从 Global Memory 搬运到 UB（用于 Vector 计算）
T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
```

### shared → GM

```python
# 从 UB 搬运回 Global Memory
T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])
```

### fragment → GM

```python
# 从 L0C 搬运到 Global Memory
T.copy(C_L0, C[bx * block_M, by * block_N])
```

### BufferRegion 切片搬运

```python
# 使用切片表示搬运范围
T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], k_l1)
T.copy(workspace_1[cid, vid * block_M // 2:vid * block_M // 2 + block_M // 2, :], acc_s_ub_)
```

## 最佳实践

1. **数据对齐**：搬运数据时注意硬件对齐要求
2. **搬运与计算重叠**：配合 `T.Pipelined` 实现数据预取，掩盖搬运延迟
3. **最小化搬运量**：合理设计 tile 大小，减少不必要的数据搬运
4. **利用 VEC_NUM 切分**：每个 Vector 单元处理一部分数据，减小单次搬运量
