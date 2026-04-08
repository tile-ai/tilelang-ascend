# Vector算子API快速参考

## 内存分配

| API | 说明 | 示例 |
|-----|------|------|
| `T.alloc_ub(shape, dtype)` | 分配UB缓存 | `T.alloc_ub([32, 128], "float")` |

**注意**: UB总空间有限，需合理规划buffer大小

## 数据搬运

| API | 说明 | 示例 |
|-----|------|------|
| `T.copy(src, dst)` | GM ↔ UB数据拷贝 | `T.copy(A[...], a_ub)` |

## Tile操作（逐元素）

### 数学运算

| API | 功能 | 数学表示 |
|-----|------|---------|
| `T.tile.add(dst, src0, src1)` | 加法 | dst = src0 + src1 |
| `T.tile.sub(dst, src0, src1)` | 减法 | dst = src0 - src1 |
| `T.tile.mul(dst, src0, src1)` | 乘法 | dst = src0 * src1 |
| `T.tile.div(dst, src0, src1)` | 除法 | dst = src0 / src1 |
| `T.tile.max(dst, src0, src1)` | 最大值 | dst = max(src0, src1) |
| `T.tile.min(dst, src0, src1)` | 最小值 | dst = min(src0, src1) |

**注意**: src1可以是buffer或scalar标量

### 激活函数

| API | 功能 | 数学表示 |
|-----|------|---------|
| `T.tile.exp(dst, src)` | 指数 | dst = exp(src) |
| `T.tile.ln(dst, src)` | 自然对数 | dst = ln(src) |
| `T.tile.sqrt(dst, src)` | 平方根 | dst = √src |
| `T.tile.rsqrt(dst, src)` | 平方根倒数 | dst = 1/√src |
| `T.tile.reciprocal(dst, src)` | 倒数 | dst = 1/src |
| `T.tile.relu(dst, src)` | ReLU | dst = max(0, src) |
| `T.tile.abs(dst, src)` | 绝对值 | dst = \|src\| |

### 其他操作

| API | 功能 | 示例 |
|-----|------|------|
| `T.tile.fill(buffer, value)` | 填充 | `T.tile.fill(a_ub, 0.0)` |
| `T.tile.neg(dst, src)` | 取负 | dst = -src |

## 归约操作

| API | 功能 | 输出shape |
|-----|------|-----------|
| `T.reduce_sum(buf, out, tmp, dim)` | 求和 | (M,) when dim=-1 on (M, N) |
| `T.reduce_max(buf, out, tmp, dim)` | 最大值 | (M,) when dim=-1 on (M, N) |
| `T.reduce_min(buf, out, tmp, dim)` | 最小值 | (M,) when dim=-1 on (M, N) |

**tmp buffer大小**:
```python
tmp_size = 3 * DataType(dtype).bits // 8 * block_M // VEC_NUM * block_N
tmp_ub = T.alloc_ub([tmp_size], "uint8")
```

## 广播操作

| API | 功能 | shape变化 |
|-----|------|-----------|
| `T.tile.broadcast(dst, src, tmp)` | 广播 | [M, 1] → [M, N] 或 [1, N] → [M, N] |

**示例**:
```python
max_ub = T.alloc_ub([M, 1], dtype)      # [M, 1]
max_2d_ub = T.alloc_ub([M, N], dtype)   # [M, N]
T.tile.broadcast(max_2d_ub, max_ub, tmp_ub)
```

## 同步

| API | 说明 |
|-----|------|
| `T.barrier_all()` | Vector核内同步 |
| `T.pipe_barrier("V")` | Vector管线屏障 |

## 调度原语

| API | 说明 |
|-----|------|
| `T.serial(n)` | 顺序循环 |
| `T.ceildiv(a, b)` | 向上取整除法 |

## Kernel定义

| API | 说明 |
|-----|------|
| `@tilelang.jit(out_idx=[...])` | JIT编译装饰器 |
| `@T.prim_func` | 定义prim_func |
| `T.Kernel(block_num, is_npu=True)` | Kernel启动 |
| `T.Scope("V")` | Vector核作用域 |

## 常用Pass配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```