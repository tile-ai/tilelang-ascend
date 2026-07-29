# Hadamard 变换算子设计文档

## 1. 概述

### 1.1 算子名称

hadamard_transform（快速 Hadamard 变换，基于共享内存实现）

### 1.2 功能描述

对输入向量执行快速 Hadamard 变换（Fast Walsh-Hadamard Transform），使用蝶形网络结构实现高效计算，支持 2 到 32768 维度的向量变换。

### 1.3 数学公式

Hadamard 矩阵递归定义为：

$$
H_2 = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
$$

$$
H_n = H_2 \otimes H_{n/2} = \begin{bmatrix} H_{n/2} & H_{n/2} \\ H_{n/2} & -H_{n/2} \end{bmatrix}
$$

其中 $\otimes$ 表示 Kronecker 积。对于输入向量 $x$，Hadamard 变换为：

$$
y = H_n \cdot x
$$

使用快速蝶形网络算法，计算复杂度从 $O(n^2)$ 降低到 $O(n \log n)$。

### 1.4 算法描述

快速 Hadamard 变换采用蝶形网络结构。Ascend NPU 无 CUDA 线程/warp 模型，采用 **Block 级并行 + 串行蝶形** 的实现方式，分为两个阶段：

1. **块内蝶形（log2(block_size) 级）**：每个 Block（cid）处理 block_size 个元素，数据加载到 UB 后串行完成块内蝶形操作
2. **跨块蝶形（log2(n) - log2(block_size) 级）**：当 n > block_size 时，由 Host 端协调，每级跨块蝶形调用一次 kernel，配对两个 block 的数据进行蝶形

**蝶形操作公式**：
```
for each stage i from 0 to log2(n)-1:
    for each pair (k, k + 2^i):
        a = x[k]
        b = x[k + 2^i]
        x[k] = a + b
        x[k + 2^i] = a - b
```

### 1.5 数据流图

```
n ≤ block_size:
  GM[A] → UB[data_ub] → butterfly(intra-block serial) → GM[B]

n > block_size:
  GM[A] → UB[data_ub] → butterfly(intra-block) → GM[B]
  GM → UB[data_ub/data2_ub] → butterfly(cross-block pair) → GM[B]  (one kernel call per stage)
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer 模式 + 共享内存数据交换

### 2.2 选型理由

1. **计算类型判定**：纯 Vector 算子（element-wise 加减操作），无 matmul
2. **复杂度级别**：多步迭代计算（log2(n) 级蝶形操作），需要中间缓冲
3. **平台特性约束**：
   - Ascend NPU **不支持 CUDA warp shuffle 指令和 threads 线程模型**
   - Ascend 使用 Block 级并行模型，每个 Block 有 cid（计算任务 ID）和 vid（Vector 单元索引，固定 2 个）
   - 数据计算在 UB 中完成，Block 间数据交换通过 GM 中转（Host 协调多 kernel 调用）
4. **同步需求**：块内蝶形串行执行无需额外同步；跨块蝶形由 Host 端串行调用 kernel，天然有序

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | T.alloc_ub（UB 内存，Vector 缓冲） |
| 计算方式 | T.serial 串行蝶形 + 符号运算（加减） |
| 作用域 | 编译器自动分离，无需手动 T.Scope |
| 同步方式 | Developer 模式自动同步（TL_ASCEND_AUTO_SYNC） |
| 数据交换 | 块内串行蝶形；跨块通过 Host 协调多 kernel 调用经 GM 中转 |

---

## 3. API 映射设计

### 3.1 公式拆解

**块内蝶形（n ≤ block_size 或块内阶段）**：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | 加载：data_ub[i] = A[batch, offset + i] | 从 GM 加载 block_size 个元素到 UB |
| 2 | 块内蝶形：log2(block_size) 级迭代 | 在 UB 内串行完成蝶形操作 |
| 3 | 写回：B[batch, offset + i] = data_ub[i] | 从 UB 写回 GM |

**跨块蝶形（n > block_size 的跨块阶段，每级一次 kernel）**：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | 加载：data_ub[k] = A[batch, src_offset + k]; data2_ub[k] = A[batch, dst_offset + k] | 加载配对两个 half 到 UB |
| 2 | 蝶形：tmp_ub[k] = data_ub[k] + data2_ub[k] | 蝶形加法，结果写 tmp_ub |
| 3 | 写回上半：B[batch, src_offset + k] = tmp_ub[k] | 写回 GM |
| 4 | 蝶形：tmp_ub[k] = data_ub[k] - data2_ub[k] | 蝶形减法，结果写 tmp_ub |
| 5 | 写回下半：B[batch, dst_offset + k] = tmp_ub[k] | 写回 GM |

### 3.2 TileLang API 映射

**块内蝶形 kernel（hadamard_block_intra）**：

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | 加载 | T.copy | A[batch, offset:offset+block_size] → data_ub | Developer |
| 2 | 块内蝶形 | T.serial(log_block) + T.serial 内层 | 蝶形加减操作 | Developer |
| 3 | 写回 | T.copy | data_ub → B[batch, offset:offset+block_size] | Developer |

**跨块蝶形 kernel（hadamard_cross_block_pair）**：

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | 加载配对 | T.copy ×2 | A[src] → data_ub; A[dst] → data2_ub | Developer |
| 2 | 蝶形加 | T.serial(half) | tmp_ub[k] = data_ub[k] + data2_ub[k] | Developer |
| 3 | 写回上半 | T.copy | tmp_ub → B[src] | Developer |
| 4 | 蝶形减 | T.serial(half) | tmp_ub[k] = data_ub[k] - data2_ub[k] | Developer |
| 5 | 写回下半 | T.copy | tmp_ub → B[dst] | Developer |

### 3.3 计算伪代码

**块内蝶形 kernel（hadamard_block_intra）**：

```python
@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def hadamard_block_intra(b, n, block_size, dtype):
    log_block = int(math.log2(block_size))
    num_blocks_per_batch = n // block_size
    total_blocks = b * num_blocks_per_batch

    @T.prim_func
    def main(A: T.Tensor((b, n), dtype), B: T.Tensor((b, n), dtype)):
        with T.Kernel(total_blocks, is_npu=True) as (cid, vid):
            if vid == 0:
                batch_id = cid // num_blocks_per_batch
                block_id_in_batch = cid % num_blocks_per_batch
                offset = block_id_in_batch * block_size

                data_ub = T.alloc_ub((block_size,), dtype)

                # 1. Load
                T.copy(A[batch_id, offset : offset + block_size], data_ub)

                # 2. Intra-block serial butterfly
                for stage in T.serial(log_block):
                    chunk_size = 1 << (stage + 1)
                    chunk_num = block_size // chunk_size
                    for chunk_idx in T.serial(chunk_num):
                        base = chunk_idx * chunk_size
                        half = chunk_size // 2
                        for k in T.serial(half):
                            a_val = data_ub[base + k]
                            b_val = data_ub[base + k + half]
                            data_ub[base + k] = a_val + b_val
                            data_ub[base + k + half] = a_val - b_val

                # 3. Write back
                T.copy(data_ub, B[batch_id, offset : offset + block_size])

    return main
```

**跨块蝶形 kernel（hadamard_cross_block_pair）**：

```python
@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def hadamard_cross_block_pair(b, n, block_size, cross_stage, dtype):
    chunk_size = 1 << (cross_stage + 1)
    half = chunk_size // 2
    num_chunks_per_batch = n // chunk_size
    total_chunks = b * num_chunks_per_batch

    @T.prim_func
    def main(A: T.Tensor((b, n), dtype), B: T.Tensor((b, n), dtype)):
        with T.Kernel(total_chunks, is_npu=True) as (cid, vid):
            batch_id = cid // num_chunks_per_batch
            chunk_id_in_batch = cid % num_chunks_per_batch

            data_ub = T.alloc_ub((half,), dtype)
            data2_ub = T.alloc_ub((half,), dtype)
            tmp_ub = T.alloc_ub((half,), dtype)

            src_offset = chunk_id_in_batch * chunk_size
            dst_offset = src_offset + half

            # 1. Load paired halves
            T.copy(A[batch_id, src_offset : src_offset + half], data_ub)
            T.copy(A[batch_id, dst_offset : dst_offset + half], data2_ub)

            # 2. Butterfly add + write back upper half
            for k in T.serial(half):
                tmp_ub[k] = data_ub[k] + data2_ub[k]
            T.copy(tmp_ub, B[batch_id, src_offset : src_offset + half])

            # 3. Butterfly subtract + write back lower half
            for k in T.serial(half):
                tmp_ub[k] = data_ub[k] - data2_ub[k]
            T.copy(tmp_ub, B[batch_id, dst_offset : dst_offset + half])

    return main
```

**Host 端协调完整变换（n > block_size 时多级跨块调用）**：

```python
def hadamard_transform_complete(b, n, dtype="float", block_size=1024):
    if n <= block_size:
        kernel = hadamard_block_intra(b, n, n, dtype)
        return lambda x: kernel(x)

    log_n = int(math.log2(n))
    log_block = int(math.log2(block_size))

    kernel_intra = hadamard_block_intra(b, n, block_size, dtype)
    cross_kernels = [
        hadamard_cross_block_pair(b, n, block_size, stage, dtype)
        for stage in range(log_block, log_n)
    ]

    def full_transform(x):
        y = kernel_intra(x)
        for kernel_cross in cross_kernels:
            y = kernel_cross(y)
        return y

    return full_transform
```

### 3.4 API 可行性确认

| API | 来源 | 是否可用 | 说明 |
|-----|------|---------|------|
| T.alloc_ub | tilelang/language/allocate.py | ✅ | 用于分配 UB 内存（Vector 缓冲），蝶形计算数据缓冲 |
| T.copy | tilelang/language/copy.py | ✅ | GM↔UB 数据搬运 |
| T.serial | tilelang/language | ✅ | 串行循环，用于蝶形迭代 |
| T.Kernel | tilelang/language/kernel.py | ✅ | Kernel 定义，支持 threads 参数, 用法参考`with T.Kernel(..., threads=2, is_npu=True) as (cid):` |
| T.ceildiv | tilelang/language | ✅ | 整除向上取整 |
| T.barrier_all | tilelang/language/ascend.py | ✅ | 同步屏障（自动同步开启时由编译器插入） |

> **注意**：`T.alloc_local` 已从 TileLang-Ascend 移除，不再支持。Ascend 上使用 `T.alloc_ub` 分配 Vector 缓冲，使用 `T.alloc_var` 分配标量变量。`T.alloc_shared` 在 Ascend 上由编译器自动映射到 L1/UB，本算子纯 Vector 计算显式使用 `T.alloc_ub` 更清晰。

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| A | (batch, n) | float32 / float16 / bfloat16 | 输入向量，n 必须是 2 的幂次，范围 [2, 32768] |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| B | (batch, n) | float32 / float16 / bfloat16 | 输出向量，shape 与输入相同 |

### 4.3 中间缓冲区

**块内蝶形 kernel（hadamard_block_intra）**：

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| data_ub | (block_size,) | float32 | UB | 块内蝶形数据缓冲 |

**跨块蝶形 kernel（hadamard_cross_block_pair）**：

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| data_ub | (half,) | float32 | UB | 蝶形上半数据（src） |
| data2_ub | (half,) | float32 | UB | 蝶形下半数据（dst） |
| tmp_ub | (half,) | float32 | UB | 蝶形结果临时缓冲 |

### 4.4 内存搬运路径

```
intra-block butterfly:
  GM[A] --T.copy--> UB[data_ub] --serial butterfly--> UB[data_ub] --T.copy--> GM[B]

cross-block butterfly (one kernel per stage):
  GM[A_src] --T.copy--> UB[data_ub]
  GM[A_dst] --T.copy--> UB[data2_ub]
  UB[data_ub/data2_ub] --butterfly add/sub--> UB[tmp_ub] --T.copy--> GM[B_src/B_dst]
```

### 4.5 UB 内存预算

**块内蝶形 kernel**（假设 block_size=1024, dtype=float32）：

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| data_ub | (1024,) | float32 | 4096 |
| **总计** | | | 4096 / 192KB UB |

**跨块蝶形 kernel**（假设 half=block_size/2=512, dtype=float32）：

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| data_ub | (512,) | float32 | 2048 |
| data2_ub | (512,) | float32 | 2048 |
| tmp_ub | (512,) | float32 | 2048 |
| **总计** | | | 6144 / 192KB UB |

**约束**：UB 容量 192KB，block_size 受 UB 限制。当 block_size * elem_size 接近 UB 上限时需减小 block_size。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| n | T.dyn['n'] 或直接作为 JIT 参数 | 2 ~ 32768（2 的幂次） |

### 4.7 JIT 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def hadamard_block_intra(b, n, block_size, dtype="float"):
    ...
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 算子仅包含 element-wise 加减运算，无 matmul，判定为纯 Vector。

### 5.2 Block 划分

```python
# block_size is the number of elements for intra-block butterfly, must be a power of 2
# n ≤ block_size: single kernel completes all butterfly stages
# n > block_size: intra-block butterfly + multi-stage cross-block butterfly
block_size = 1024  # default, constrained by UB capacity
num_blocks_per_batch = n // block_size
total_blocks = b * num_blocks_per_batch  # each Block processes block_size elements
```

**选择理由**：
- block_size 取 2 的幂次，便于蝶形操作
- block_size 受 UB 容量约束：block_size * elem_size ≤ 192KB
- 跨块阶段由 Host 端串行调用 kernel，每级跨块一次 kernel 调用

### 5.3 约束分析

- **对齐约束**: n、block_size 必须是 2 的幂次，n % block_size == 0 ✓
- **UB 容量**: 块内 kernel data_ub = block_size * elem_size
  - block_size=1024, float32: 1024 * 4 = 4KB，远小于 192KB UB
  - 跨块 kernel: 3 * half * elem_size，half = chunk_size / 2
- **L0 容量**: 无 Cube 计算，不适用
- **Ascend Block 模型**: 每个 Block 由 cid 标识，vid 取 0/1，用 `vid == 0` 守卫执行

### 5.4 注意事项

1. **跨块蝶形 Host 协调**：n > block_size 时，跨块每级需单独调用一次 kernel，由 Host 串行调度
2. **无线程模型**：Ascend 无 CUDA warp/threads 概念，块内蝶形全部串行执行
3. **vid 守卫**：纯 Vector 计算，用 `if vid == 0:` 确保只在 vid=0 的 Vector 核上执行

---

## 6. 循环与调度结构

### 6.1 循环结构总结

**块内蝶形 kernel**：

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| Block 并行 | Block 级并行 | T.Kernel(total_blocks, is_npu=True) | 每个 Block 处理一个 block_size 数据块 |
| 蝶形 stage | 串行迭代 | T.serial(log_block) | log2(block_size) 级蝶形串行执行 |
| chunk 遍历 | 串行 | T.serial(chunk_num) | 每 stage 内 chunk 串行 |
| 蝶形对 | 串行 | T.serial(half) | chunk 内蝶形对串行 |

**跨块蝶形 kernel**：

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| Chunk 并行 | Block 级并行 | T.Kernel(total_chunks, is_npu=True) | 每个 Block 处理一对 half |
| 蝶形加减 | 串行 | T.serial(half) | half 个元素串行蝶形 |

### 6.2 循环伪代码

**Intra-block butterfly core logic**:
```python
for stage in T.serial(log_block):       # log_block = log2(block_size)
    chunk_size = 1 << (stage + 1)
    chunk_num = block_size // chunk_size
    for chunk_idx in T.serial(chunk_num):
        base = chunk_idx * chunk_size
        half = chunk_size // 2
        for k in T.serial(half):
            a_val = data_ub[base + k]
            b_val = data_ub[base + k + half]
            data_ub[base + k] = a_val + b_val
            data_ub[base + k + half] = a_val - b_val
```

**Cross-block butterfly core logic**:
```python
# Load paired halves
T.copy(A[batch_id, src_offset : src_offset + half], data_ub)
T.copy(A[batch_id, dst_offset : dst_offset + half], data2_ub)

# Butterfly add, write back upper half
for k in T.serial(half):
    tmp_ub[k] = data_ub[k] + data2_ub[k]
T.copy(tmp_ub, B[batch_id, src_offset : src_offset + half])

# Butterfly subtract, write back lower half
for k in T.serial(half):
    tmp_ub[k] = data_ub[k] - data2_ub[k]
T.copy(tmp_ub, B[batch_id, dst_offset : dst_offset + half])
```

### 6.3 流水线优化

暂不使用 T.Pipelined，因为：
1. 蝶形操作有严格的数据依赖，每一级必须等待前一级完成
2. 数据交换需要同步，无法并行化
3. 可后续尝试使用 T.Pipelined(num_stages=1) 进行轻微优化

### 6.4 尾块处理

- 输入 n 必须是 2 的幂次，无尾块问题
- 若 n 不是 2 的幂次，需在前端进行 zero-padding 或报错

---

## 7. 同步策略

### 7.1 同步模式

**模式**: Developer 模式自动同步 + 关键点手动 T.barrier_all()

### 7.2 同步点说明

| 位置 | 同步 API | 理由 |
|------|----------|------|
| 数据写入 shared 后 | T.barrier_all() | 确保所有线程写入完成 |
| 数据从 shared 读取前 | T.barrier_all() | 确保数据可读 |
| 蝶形操作前后（可选） | 自动同步 | Developer 模式自动处理 |

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

---

## 8. 验证方案

### 8.1 Golden 函数

```python
def ref_hadamard(x: torch.Tensor):
    """Reference implementation based on scipy"""
    import scipy.linalg
    assert x.ndim == 2
    dim = x.shape[-1]
    assert is_pow_of_2(dim)
    H = torch.tensor(scipy.linalg.hadamard(dim, dtype=float), dtype=x.dtype, device=x.device)
    return torch.nn.functional.linear(x, H)
```

### 8.2 测试用例

| 用例名 | 级别 | Shape | dtype | 说明 |
|--------|------|-------|-------|------|
| basic_small | Level 0 | (4, 8) | float32 | 最小功能验证 |
| typical_512 | Level 1 | (64, 512) | float32 | 典型配置 |
| typical_1024 | Level 1 | (64, 1024) | float32 | Warp 级测试 |
| typical_8192 | Level 1 | (64, 8192) | float32 | Block 级测试 |
| large_scale | Level 3 | (64, 32768) | float32 | 最大规模性能测试 |
| dtype_fp16 | Level 2 | (64, 1024) | float16 | float16 精度测试 |

### 8.3 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float16 | 1e-2 | 1e-2 |
| float32 | 1e-3 | 1e-3 |
| bfloat16 | 1e-2 | 1e-2 |

---

## 9. 风险点与注意事项

### 9.1 已知约束

1. **Ascend 不支持 warp shuffle / threads 线程模型**：块内蝶形全部串行执行，跨块由 Host 协调多 kernel 调用
2. **n 必须是 2 的幂次**：算法依赖蝶形网络结构
3. **UB 容量限制**：block_size * elem_size 受 192KB UB 约束
4. **跨块 kernel 调用开销**：n > block_size 时每级跨块需一次 kernel 调用，级数多时 Host 调度开销增大

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB 溢出 | block_size 过大，data_ub 超限 | 编译/运行失败 | 减小 block_size |
| 跨块配对错位 | src_offset/dst_offset 计算错误 | 数据错乱 | 按 chunk_size 对齐计算 offset |
| 非幂次输入 | n 不是 2 的幂次 | 算法错误 | 前端验证或 zero-padding |
| vid 守卫缺失 | 未用 `if vid == 0` 守卫 | 重复执行/数据错乱 | 纯 Vector 计算加 `if vid == 0:` |

### 9.3 特殊场景处理

1. **大规模输入（n > block_size）**：块内蝶形 + 多级跨块蝶形，Host 串行调度
2. **混合精度**：float16 节省 UB 空间，可增大 block_size
3. **动态 shape**：JIT 支持动态 n，但需保证是 2 的幂次

---

## 10. 交付清单

### 10.1 目录结构

```
examples/hadamard_transform/
├── example_hadamard_transform.py  # operator implementation + basic tests
├── design.md                      # this design document
└── README.md                      # usage instructions (optional)
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| design.md | ✅ 已完成 | 设计文档 |
| example_hadamard_transform.py | ✅ 已完成 | 算子实现（块内 + 跨块完整方案） |
| test_hadamard.py | ⬜ 待实现 | 测试文件（可选，放入 testing/） |

### 10.3 命名规范

- 目录名: hadamard_transform（snake_case）
- 实现文件: example_hadamard_transform.py
- 测试文件: test_hadamard.py

### 10.4 实现顺序

1. ✅ 设计文档（design.md）
2. ✅ Golden 函数（验证基准，见 example_hadamard_transform.py 中 ref_hadamard）
3. ✅ 算子实现（example_hadamard_transform.py）
4. ⬜ 基础测试（Level 0 + Level 1）
5. ⬜ 边界测试（Level 2）
6. ⬜ 性能测试（Level 3，可选）

---

## 附录：关键差异分析（CUDA vs Ascend）

| 特性 | CUDA | Ascend | 影响 |
|------|------|--------|------|
| Warp shuffle | 支持（__shfl_*） | **不支持** | 需用共享内存替代或串行蝶形 |
| Warp 大小 | 32 线程/warp | **无此概念** | 无法使用 warp 级并行 |
| 线程索引 | threadIdx.x | **不支持 threads 参数** | Ascend 使用 cid/vid 模型 |
| 共享内存 | shared memory | UB | 内存层级对应 |
| 同步 | __syncthreads() | T.barrier_all() | API 差异 |
| Block 级并行 | CUDA Block | T.Kernel Block | 类似，但 Ascend Block 内无线程级并行 |
| threads 参数 | T.Kernel(..., threads=N) | **不支持**（src/ir.cc:247-257） | Ascend kernel 只支持 1D grid (cid)，vid 固定为 2 |

**关键适配点**：
1. Ascend NPU **不支持 CUDA threads 参数和 warp shuffle**
2. Ascend 使用 Block 级并行模型，每个 Block 有 cid 和 vid（固定 2 个 Vector 单元）
3. 无法使用线程级并行，必须采用串行蝶形网络或 Block 级协作
4. 使用共享内存（UB）进行数据交换

---

## 附录 B：当前实现状态

### 实现文件

| 文件 | 说明 | 支持范围 |
|------|------|---------|
| example_hadamard_transform.py | **完整版本（块内 + Host 协调跨块蝶形）** | **任意 n（2 的幂次）** |

### 完整实现方案（n > 1024）

当 n > block_size 时，Hadamard 变换分为两个阶段：

**阶段 1：块内蝶形（log2(block_size) 级）**
- Kernel: `hadamard_block_intra`
- 每个 Block 处理 block_size 个元素
- 并行执行 block 内的蝶形操作

**阶段 2：跨块蝶形（log2(n) - log2(block_size) 级）**
- Kernel: `hadamard_cross_block_pair`
- Host 端协调，每级跨块蝶形调用一次 kernel
- 处理 chunk_size = 2^(cross_stage+1) 的蝶形配对

**示例：n=2048, block_size=1024**
```
Stage 1 (Kernel 1): first 10 butterfly stages (intra-block)
  Block 0: processes elements 0-1023
  Block 1: processes elements 1024-2047
  
Stage 2 (Kernel 2): 11th butterfly stage (cross-block)
  elements 0-1023 paired with elements 1024-2047 for butterfly
  a[i] + b[i], a[i] - b[i]
```

### 跨块蝶形 Kernel 设计

```python
@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def hadamard_cross_block_pair(b, n, block_size, cross_stage, dtype):
    chunk_size = 1 << (cross_stage + 1)
    half = chunk_size // 2

    # Each chunk maps to one Block
    # src_offset = chunk_id * chunk_size
    # dst_offset = src_offset + half
    # Butterfly: src paired with dst
```

### 性能特性

| 范围 | Kernel 调用次数 | 性能特点 |
|------|----------------|---------|
| n ≤ block_size | 1 次 | 高效，单 kernel 完成 |
| n = 2 * block_size | 2 次 | 块内 + 1 级跨块 |
| n = 4 * block_size | 3 次 | 块内 + 2 级跨块 |
| n = 8 * block_size | 4 次 | 块内 + 3 级跨块 |

### 后续优化方向

1. 合并跨块蝶形 kernel（多级合并为单次调用）
2. 使用 workspace 减少 GM 访问
3. 使用 T.Pipelined 进行流水线优化