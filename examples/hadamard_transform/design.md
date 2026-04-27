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

快速 Hadamard 变换采用蝶形网络结构，分为三个层次：

1. **线程内计算（n ≤ 8）**：每个线程处理 thread_elem 个元素，在寄存器内完成前 log2(thread_elem) 级蝶形操作
2. **Block 内 Warp 级交换（n ≤ 512）**：通过共享内存实现 Warp 间数据交换，完成 log2(threads/warps) 级蝶形操作
3. **Block 级交换（n ≤ 32768）**：通过共享内存实现跨 Warp 数据交换，完成 log2(warps) 级蝶形操作

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
GM[A] → UB[local] → 蝶形计算(线程内) → 共享内存交换(Warp级) → 蝶形计算(Warp级) → 共享内存交换(Block级) → 蝶形计算(Block级) → UB[local] → GM[B]
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer 模式 + 共享内存数据交换

### 2.2 选型理由

1. **计算类型判定**：纯 Vector 算子（element-wise 加减操作），无 matmul
2. **复杂度级别**：多步迭代计算（log2(n) 级蝶形操作），需要中间缓冲
3. **平台特性约束**：
   - Ascend NPU **不支持 CUDA warp shuffle 指令**
   - Ascend 使用 Block 级并行模型，每个 Block 有 cid（计算任务 ID）和 vid（Vector 单元索引）
   - 线程间数据交换必须通过共享内存（shared memory）实现
4. **同步需求**：多级蝶形计算需要同步，Developer 模式自动同步更方便

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | T.alloc_shared（共享内存）+ T.alloc_local（寄存器内存） |
| 计算方式 | T.Parallel + T.serial + 符号运算（加减） |
| 作用域 | 编译器自动分离，无需手动 T.Scope |
| 同步方式 | Developer 模式自动同步（T.barrier_all 在关键点） |
| 数据交换 | 使用共享内存替代 warp shuffle |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | 加载：local[i] = A[bx, offset + i] | 从 GM 加载到寄存器 |
| 2 | 线程内蝶形：log2(thread_elem) 级迭代 | 在寄存器内完成蝶形操作 |
| 3 | Warp 级数据交换：shared[tx, :] = local; local = shared[another_tx, :] | 通过共享内存交换数据 |
| 4 | Warp 级蝶形：log2(warp_size) 级迭代 | 完成 Warp 内蝶形操作 |
| 5 | Block 级数据交换：shared[src_tx, :] = local; local = shared[tgt_tx, :] | 跨 Warp 数据交换 |
| 6 | Block 级蝶形：log2(warps) 级迭代 | 完成 Block 内蝶形操作 |
| 7 | 写回：B[bx, offset + i] = local[i] | 从寄存器写回 GM |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | 加载 | T.copy 或 T.vectorized 循环 | A[bx, offset:offset+thread_elem] → local | Developer |
| 2 | 线程内蝶形 | T.serial(thread_round) + T.serial 循环 | 蝶形加减操作 | Developer |
| 3 | Warp 级数据交换 | T.alloc_shared + T.copy | shared[threads, thread_elem] 存储交换数据 | Developer |
| 4 | Warp 级蝶形 | T.serial(warp_round) + T.serial 循环 | 蝶形加减操作 | Developer |
| 5 | Block 级数据交换 | T.alloc_shared + T.copy | 跨 Warp 数据交换 | Developer |
| 6 | Block 级蝶形 | T.serial(block_round) + T.serial 循环 | 蝶形加减操作 | Developer |
| 7 | 写回 | T.copy 或 T.vectorized 循环 | local → B[bx, offset:offset+thread_elem] | Developer |

### 3.3 计算伪代码

```python
@tilelang.jit(out_idx=[1])
def hadamard(b, n, dtype):
    logN = int(math.log2(n))
    threads = [...]  # 根据 n 确定 threads 数量
    thread_elem = n // threads
    
    @T.prim_func
    def main(A: T.Tensor((b, n), dtype), B: T.Tensor((b, n), dtype)):
        with T.Kernel(b, threads=threads) as bx:
            local = T.alloc_local((thread_elem,), dtype)
            shared = T.alloc_shared((threads, thread_elem), dtype)
            tx = T.get_thread_binding(0)
            
            # 1. 加载
            for i in T.vectorized(thread_elem):
                local[i] = A[bx, tx * thread_elem + i]
            
            # 2. 线程内蝶形
            for i in T.serial(thread_round):
                # 蝶形操作逻辑
                ...
            
            # 3. Warp 级数据交换 + 蝶形
            for i in T.serial(warp_round):
                # 通过 shared 交换数据
                T.barrier_all()
                ...
            
            # 4. Block 级数据交换 + 蝶形
            if block_round > 0:
                ...
            
            # 5. 写回
            for i in T.vectorized(thread_elem):
                B[bx, tx * thread_elem + i] = local[i]
    
    return main
```

### 3.4 API 可行性确认

| API | 来源 | 是否可用 | 说明 |
|-----|------|---------|------|
| T.alloc_local | tilelang/language/allocate.py | ✅ | 用于分配线程私有寄存器内存 |
| T.alloc_shared | tilelang/language/allocate.py | ✅ | 用于分配共享内存（对应 Ascend UB） |
| T.get_thread_binding | tilelang/language | ✅ | 获取线程绑定（获取 tx） |
| T.serial | tilelang/language | ✅ | 普通循环 |
| T.vectorized | tilelang/language | ✅ | 向量化循环（加速数据搬运） |
| T.barrier_all | tilelang/language | ✅ | 同步屏障 |
| T.Kernel | tilelang/language | ✅ | Kernel 定义，支持 threads 参数 |
| T.macro | tilelang/language | ✅ | 宏定义（可选，用于封装蝶形操作） |

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

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| local | (thread_elem,) | float32 | UB/寄存器 | 线程私有数据缓冲 |
| shared | (threads, thread_elem) | float32 | UB | 蝶形操作数据交换缓冲 |
| another_val | (thread_elem,) | float32 | UB/寄存器 | 蝶形操作临时缓冲 |

### 4.4 内存搬运路径

```
GM[A] --T.vectorized--> UB[local] --线程内蝶形--> UB[local]
UB[local] --T.copy--> UB[shared] --交换--> UB[local] --蝶形--> UB[local]
UB[local] --T.copy--> UB[shared] --跨Warp交换--> UB[local] --蝶形--> UB[local]
UB[local] --T.vectorized--> GM[B]
```

### 4.5 UB 内存预算

假设 n=32768, threads=256, thread_elem=128, dtype=float32：

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| local | (128,) | float32 | 512 |
| shared | (256, 128) | float32 | 131072 |
| another_val | (128,) | float32 | 512 |
| **总计** | | | 132096 / ~128KB UB |

**约束**：shared buffer 最大 128KB，当 n * elem_size > 128KB 时需要分批交换。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| n | T.dyn['n'] 或直接作为 JIT 参数 | 2 ~ 32768（2 的幂次） |

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[1],
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    },
)
def hadamard(b, n, dtype):
    ...
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 算子仅包含 element-wise 加减运算，无 matmul，判定为纯 Vector。

### 5.2 Block 划分

```python
# 线程数量根据 n 动态确定
threads_table = {
    2: 1, 4: 1, 8: 1, 16: 2, 32: 4, 64: 8, 128: 16,
    256: 32, 512: 32, 1024: 128, 2048: 256,
    4096: 256, 8192: 256, 16384: 256, 32768: 256
}
threads = threads_table[logN]
thread_elem = n // threads  # 每个线程处理的元素数
```

**选择理由**：
- n 较小时使用少量线程，避免资源浪费
- n 较大时使用足够线程以并行化计算
- 约束：thread_elem 应为 2 的幂次，便于蝶形操作

### 5.3 约束分析

- **对齐约束**: n 必须是 2 的幂次 ✓
- **UB 容量**: shared buffer = threads * thread_elem * elem_size
  - 当 n=32768, float32: 256 * 128 * 4 = 128KB，刚好满 UB
  - 需要分批交换策略（exchange_round）
- **L0 容量**: 无 Cube 计算，不适用
- **Ascend Block 模型**: threads 对应 Block 内的并行单元

### 5.4 注意事项

1. **分批交换**：当 n * elem_size > 128KB 时，需要分多批进行数据交换
2. **Warp 概念映射**：Ascend 没有 CUDA warp 概念，但可以类似分组（假设 warp_size=32）
3. **线程索引**：Ascend 使用 T.get_thread_binding 获取线程索引，而非 CUDA threadIdx.x

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| Batch 维度 | Block 级并行 | T.Kernel(b, threads=threads) | 每个 Block 处理一个 batch |
| 线程级 | 隐式并行 | T.Kernel threads 参数 | Block 内 threads 个并行单元 |
| 蝶形迭代 | 串行迭代 | T.serial(logN) | logN 级蝶形操作串行执行 |
| 元素访问 | 向量化 | T.vectorized(thread_elem) | 加速数据搬运 |
| 蝶形内循环 | 串行 | T.serial(chunknum/half) | 蝶形操作内部循环 |

### 6.2 循环伪代码

```python
# 蝶形操作核心逻辑
for i in T.serial(thread_round):  # thread_round = log2(thread_elem)
    chunksize = 1 << (i + 1)
    chunknum = thread_elem // chunksize
    for j in T.serial(chunknum):
        chunkbase = j * chunksize
        for k in T.serial(chunksize // 2):
            # 蝶形操作
            a = local[chunkbase + k]
            b = local[chunkbase + k + chunksize // 2]
            local[chunkbase + k] = a + b
            local[chunkbase + k + chunksize // 2] = a - b

# 共享内存数据交换 + 蝶形
for i in T.serial(warp_round):
    # 数据交换（通过 shared）
    T.barrier_all()
    for j in T.vectorized(thread_elem):
        shared[tx, j] = local[j]
    T.barrier_all()
    another_tx = tx ^ (1 << i)  # 异或计算配对线程
    for j in T.vectorized(thread_elem):
        local[j] = shared[another_tx, j]
    T.barrier_all()
    # 蝶形操作
    ...
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
    """基于 scipy 的参考实现"""
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

1. **Ascend 不支持 warp shuffle**：必须使用共享内存替代，性能可能受影响
2. **n 必须是 2 的幂次**：算法依赖蝶形网络结构
3. **UB 容量限制**：当 n * elem_size > 128KB 时需要分批交换
4. **线程模型差异**：Ascend 的并行模型与 CUDA warp 不同

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB 溢出 | n 过大，shared buffer 超限 | 编译/运行失败 | 分批交换，减小 thread_elem |
| 同步缺失 | 数据交换未同步 | 数据错乱 | 添加 T.barrier_all() |
| 非幂次输入 | n 不是 2 的幂次 | 算法错误 | 前端验证或 zero-padding |
| 线程索引错误 | Ascend 线程索引获取方式错误 | 数据错位 | 使用 T.get_thread_binding(0) |

### 9.3 特殊场景处理

1. **大规模输入（n > 8192）**：需要分批交换策略
2. **混合精度**：需要注意 UB 容量变化（float16 节省空间）
3. **动态 shape**：JIT 支持动态 n，但需保证是 2 的幂次

---

## 10. 交付清单

### 10.1 目录结构

```
examples/hadamard_transform/
├── example_hadamard.py     # 算子实现 + 简单测试
├── design.md               # 本设计文档
└── README.md               # 使用说明（可选）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| design.md | ✅ 已完成 | 设计文档 |
| example_hadamard.py | ⬜ 待实现 | 算子实现 |
| test_hadamard.py | ⬜ 待实现 | 测试文件（可选，放入 testing/） |

### 10.3 命名规范

- 目录名: hadamard_transform（snake_case）
- 实现文件: example_hadamard.py
- 测试文件: test_hadamard.py

### 10.4 实现顺序

1. ✅ 设计文档（design.md）
2. ⬜ Golden 函数（验证基准）
3. ⬜ 算子实现（example_hadamard.py）
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
| example_hadamard.py | 基础版本（单 Block） | n ≤ 1024 |
| example_hadamard_optimized.py | 多 Block 框架（n > 1024 部分实现） | n ≤ 1024 完整，n > 1024 仅块内 |
| example_hadamard_complete.py | **完整版本（Host 协调跨块蝶形）** | **任意 n（2 的幂次）** |

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
阶段 1 (Kernel 1): 前 10 级蝶形（块内）
  Block 0: 处理元素 0-1023
  Block 1: 处理元素 1024-2047
  
阶段 2 (Kernel 2): 第 11 级蝶形（跨块）
  元素 0-1023 与元素 1024-2047 进行蝶形
  a[i] + b[i], a[i] - b[i]
```

### 跨块蝶形 Kernel 设计

```python
@tilelang.jit
def hadamard_cross_block_pair(b, n, block_size, cross_stage, dtype):
    chunk_size = 2^(cross_stage + 1)
    half = chunk_size // 2
    
    # 每个 chunk 对应一个 Block
    # src_offset = chunk_id * chunk_size
    # dst_offset = src_offset + half
    # 蝶形: src 与 dst 配对
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