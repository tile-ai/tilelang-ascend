# Expert 模式完整指南

## 核心理念

Expert 模式提供对 Ascend NPU 硬件的完全控制：
- **显式内存分配**：指定每个 buffer 的存储层级
- **显式作用域**：手动划分 Cube 和 Vector 执行区域
- **手动同步**：精确控制管线间的 flag 同步
- **手动流水线**：双缓冲、多级流水线优化

---

## pass_configs

Expert 模式通常**全部关闭**或不设置：

```python
# 方式 1：全部显式关闭
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}

# 方式 2：不传 pass_configs
@tilelang.jit(out_idx=[-1])
```

---

## 内存分配

| Expert API | 存储层级 | 用途 |
|------------|---------|------|
| `T.alloc_L1(shape, dtype)` | L1 Buffer | Cube 核数据缓存 |
| `T.alloc_ub(shape, dtype)` | Unified Buffer | Vector 核工作缓冲 |
| `T.alloc_L0A(shape, dtype)` | L0A Register | GEMM 矩阵 A 输入 |
| `T.alloc_L0B(shape, dtype)` | L0B Register | GEMM 矩阵 B 输入 |
| `T.alloc_L0C(shape, dtype)` | L0C Register | GEMM 累加输出 |
| `T.alloc_var(dtype, init=val)` | 标量变量 | 循环计数器、标志 |

### 双缓冲分配

```python
# 第一维为 2，表示两个 slot 交替使用
A_L1 = T.alloc_L1((2, block_M, K_L1), dtype)
A_L0 = T.alloc_L0A((2, block_M, block_K), dtype)
B_L0 = T.alloc_L0B((2, block_K, block_N), dtype)
```

---

## 执行作用域

Expert 模式**必须**显式声明 Cube 和 Vector 作用域：

```python
with T.Scope("C"):     # Cube 核作用域
    # 矩阵计算、L1/L0 数据搬运
    T.copy(...)
    T.mma(...)

with T.Scope("V"):     # Vector 核作用域
    # 元素级运算、UB 数据搬运
    T.tile.add(...)
    T.reduce_max(...)
```

---

## 同步原语

### 核内同步（Intra-core）

```python
T.barrier_all()                          # 全管线屏障
T.pipe_barrier("V")                      # 指定管线屏障
T.set_flag(src_pipe, dst_pipe, eventId)  # 生产者发信号
T.wait_flag(src_pipe, dst_pipe, eventId) # 消费者等信号
```

**管线名称：** `"fix"/"FIX"`, `"mte1"/"MTE1"`, `"mte2"/"MTE2"`, `"mte3"/"MTE3"`, `"m"/"M"`, `"v"/"V"`

### 典型 Flag 同步流

```
MTE2（GM→L1 搬入）→ MTE1（L1→L0 搬运）→ M（矩阵计算）→ FIX（L0C→GM 搬出）
```

```python
# MTE2 完成 → 通知 MTE1
T.set_flag("mte2", "mte1", event_id)
T.wait_flag("mte2", "mte1", event_id)

# MTE1 完成 → 通知 M
T.set_flag("mte1", "m", event_id)
T.wait_flag("mte1", "m", event_id)

# M 完成 → 通知 FIX
T.set_flag("m", "fix", event_id)
T.wait_flag("m", "fix", event_id)
```

### 核间同步（Cross-core，Cube ↔ Vector）

```python
T.set_cross_flag(pipe, flag_id, mode=2)  # 发送核间信号
T.wait_cross_flag(flag_id, pipe="")      # 等待核间信号
```

**典型核间流水线：**

```python
with T.Scope("C"):
    T.mma(...)
    T.copy(l0c, workspace[cid, ...])
    T.set_cross_flag("FIX", SEM_C2V)     # Cube → Vector

with T.Scope("V"):
    T.wait_cross_flag(SEM_C2V)            # 等待 Cube
    T.copy(workspace[cid, ...], ub_buf)
    # ... Vector 计算 ...
    T.set_cross_flag("MTE3", SEM_V2C)    # Vector → Cube（释放 workspace）
```

---

## Flag 初始化/清理模式

Expert 模式下必须在计算前后正确初始化和清理 flag：

```python
@T.macro
def init_flag():
    """初始化所有管线 flag — 预设为"可用"状态"""
    T.set_flag("mte1", "mte2", 0)
    T.set_flag("mte1", "mte2", 1)
    T.set_flag("m", "mte1", 0)
    T.set_flag("m", "mte1", 1)
    T.set_flag("fix", "m", 0)

@T.macro
def clear_flag():
    """清理所有管线 flag — 消费残余信号"""
    T.wait_flag("mte1", "mte2", 0)
    T.wait_flag("mte1", "mte2", 1)
    T.wait_flag("m", "mte1", 0)
    T.wait_flag("m", "mte1", 1)
    T.wait_flag("fix", "m", 0)

with T.Scope("C"):
    init_flag()
    # ... 计算 ...
    clear_flag()
```

---

## Tile 操作（Vector 侧计算）

### 双目运算

```python
T.tile.add(dst, src0, src1)       # dst = src0 + src1
T.tile.sub(dst, src0, src1)       # dst = src0 - src1
T.tile.mul(dst, src0, src1)       # dst = src0 * src1
T.tile.div(dst, src0, src1)       # dst = src0 / src1
T.tile.max(dst, src0, src1)       # 逐元素最大值
T.tile.min(dst, src0, src1)       # 逐元素最小值
```

### 单目运算

```python
T.tile.exp(dst, src)              # 指数
T.tile.ln(dst, src)               # 对数
T.tile.abs(dst, src)              # 绝对值
T.tile.sqrt(dst, src)             # 平方根
T.tile.rsqrt(dst, src)            # 平方根倒数
T.tile.reciprocal(dst, src)       # 取倒数
T.tile.relu(dst, src)             # ReLU
T.tile.sigmoid(dst, src, tmp)     # Sigmoid
```

### 特殊操作

```python
T.tile.fill(buffer, value)                     # 常量填充
T.tile.cast(dst, src, mode, count)             # 精度转换
T.tile.broadcast(dst, src, tmp)                # 1D → 2D 广播
T.tile.axpy(dst, src, scalar)                  # dst += scalar * src
T.tile.transpose(dst, src)                     # 16×16 转置
T.tile.compare(dst, src0, src1, mode)          # 比较
T.tile.select(dst, mask, src0, src1, mode)     # 条件选择
T.tile.gather(dst, src, offset, base_addr)     # 数据收集
T.tile.sort(dst, src, indices, tmp, repeat)    # 排序
T.tile.topk(dst, src, tmp, block_size)         # Top-K
```

---

## 高级优化模式

### 双缓冲（Double Buffering）

```python
A_L1 = T.alloc_L1((2, block_M, K_L1), dtype)  # 2 个 slot

for k in T.serial(loop_k):
    slot = k % 2
    # 预取下一迭代到另一个 slot
    if k < loop_k - 1:
        T.wait_flag("mte1", "mte2", (k + 1) % 2)
        T.copy(A[..., (k + 1) * K_L1], A_L1[(k + 1) % 2, :, :])
        T.set_flag("mte2", "mte1", (k + 1) % 2)
    # 当前 slot 正在被计算
    T.wait_flag("mte2", "mte1", slot)
    T.mma(A_L1[slot, :, :], B_L1[slot, :, :], C_L0, ...)
```

### 持久化调度（Persistent Kernel）

```python
for bx, by in T.Persistent(
    [T.ceildiv(M, block_M), T.ceildiv(N, block_N)],
    core_num, cid):
    # 每个核动态领取多个 tile 任务，减少 kernel 重启开销
    ...
```

### 静态任务分配

```python
q_tasks = block_num // NUM_CORES
r_tasks = block_num % NUM_CORES

def task_range(cid_val):
    start = cid_val * q_tasks + T.if_then_else(cid_val < r_tasks, cid_val, r_tasks)
    count = q_tasks + T.if_then_else(cid_val < r_tasks, 1, 0)
    return start, count

my_start, my_count = task_range(cid)
for t in T.serial(my_count):
    task_id = my_start + t
    ...
```

### Swizzle 优化

```python
cid = T.use_swizzle(i * core_num + cid, M, N, K, block_M, block_N, off=3)
```

### Layout 注解

```python
from tilelang.intrinsics import make_zn_layout, make_nz_layout

T.annotate_layout({
    q_l1: make_zn_layout(q_l1),      # Z-Normal 布局
    k_l1: make_nz_layout(k_l1),      # Normal-Z 布局
})
```

---

## 代码骨架

```python
import tilelang
import tilelang.language as T
import torch

@tilelang.jit(out_idx=[-1])
def my_operator(<形状参数>, dtype="float16", accum_dtype="float"):
    core_num = 20

    @T.macro
    def init_flag():
        T.set_flag("mte1", "mte2", 0)
        T.set_flag("mte1", "mte2", 1)
        T.set_flag("m", "mte1", 0)
        T.set_flag("m", "mte1", 1)
        T.set_flag("fix", "m", 0)

    @T.macro
    def clear_flag():
        T.wait_flag("mte1", "mte2", 0)
        T.wait_flag("mte1", "mte2", 1)
        T.wait_flag("m", "mte1", 0)
        T.wait_flag("m", "mte1", 1)
        T.wait_flag("fix", "m", 0)

    @T.prim_func
    def main(
        Input1: T.Tensor((<shape>), dtype),
        Output: T.Tensor((<shape>), dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            # 显式分配各级存储
            A_L1 = T.alloc_L1((2, block_M, K_L1), dtype)
            A_L0 = T.alloc_L0A((2, block_M, block_K), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)
            work_ub = T.alloc_ub((block_M // 2, block_N), accum_dtype)

            with T.Scope("C"):
                init_flag()
                # Cube 计算 + 手动流水线
                ...
                clear_flag()

            with T.Scope("V"):
                # Vector 计算 + T.tile.* 操作
                ...

    return main
```
