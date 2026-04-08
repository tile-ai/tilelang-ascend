# 模板 E：高性能 GEMM（Expert 模式）

**适用于：** 极致优化的矩阵乘法，手动双缓冲、多级流水线

## 核心优化

- **双缓冲**：L1 和 L0 各 2 个 slot，搬运与计算重叠
- **手动 Flag 同步**：精确控制管线间依赖
- **Swizzle**：优化核间负载均衡
- **多核任务分配**：每个核处理多个 tile

## pass_configs

Expert 模式不设 pass_configs：

```python
@tilelang.jit(out_idx=[-1])  # 无 pass_configs
```

## 完整模板

```python
import tilelang
import tilelang.language as T
import torch

@tilelang.jit(out_idx=[-1])
def matmul_expert(M, N, K, block_M, block_N, block_K, K_L1, S1, S2,
                  dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    core_num = 20  # 按硬件配置调整

    # =====================
    # Flag 初始化/清理宏
    # =====================
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
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            # =====================
            # 双缓冲 Buffer 分配
            # =====================
            A_L1 = T.alloc_L1((S1, block_M, K_L1), dtype)       # L1 双缓冲
            B_L1 = T.alloc_L1((S1, K_L1, block_N), dtype)
            A_L0 = T.alloc_L0A((S2, block_M, block_K), dtype)   # L0 双缓冲
            B_L0 = T.alloc_L0B((S2, block_K, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                init_flag()

                # =====================
                # 多核任务循环
                # =====================
                for i in T.serial(T.ceildiv(m_num * n_num, core_num)):
                    # Swizzle 优化负载均衡
                    cid = T.use_swizzle(i * core_num + cid, M, N, K, block_M, block_N, off=3)
                    if cid < m_num * n_num:
                        bx = cid // n_num
                        by = cid % n_num

                        loop_k = T.ceildiv(K, K_L1)

                        # =====================
                        # 首次加载到 L1[0]
                        # =====================
                        T.wait_flag("mte1", "mte2", 0)
                        T.copy(A[bx * block_M, 0], A_L1[0, :, :])
                        T.copy(B[0, by * block_N], B_L1[0, :, :])
                        T.set_flag("mte2", "mte1", 0)

                        T.wait_flag("fix", "m", 0)

                        # =====================
                        # K 方向主循环（L1 双缓冲）
                        # =====================
                        for k in T.serial(loop_k):
                            # 预取下一个 K_L1 块到 L1 另一个 slot
                            if k < loop_k - 1:
                                T.wait_flag("mte1", "mte2", (k + 1) % S1)
                                T.copy(A[bx * block_M, (k + 1) * K_L1],
                                       A_L1[(k + 1) % S1, :, :])
                                T.copy(B[(k + 1) * K_L1, by * block_N],
                                       B_L1[(k + 1) % S1, :, :])
                                T.set_flag("mte2", "mte1", (k + 1) % S1)

                            # =====================
                            # L0 子循环（L0 双缓冲）
                            # =====================
                            loop_kk = T.ceildiv(K_L1, block_K)
                            for kk in T.serial(loop_kk):
                                if kk == 0:
                                    T.wait_flag("mte2", "mte1", k % S1)

                                # L1 → L0 搬运
                                T.wait_flag("m", "mte1", kk % S2)
                                T.copy(A_L1[k % S1, 0, kk * block_K],
                                       A_L0[kk % S2, :, :])
                                T.copy(B_L1[k % S1, kk * block_K, 0],
                                       B_L0[kk % S2, :, :])

                                if kk == 3:  # 提前释放 L1 slot
                                    T.set_flag("mte1", "mte2", k % S1)

                                T.set_flag("mte1", "m", kk % S2)
                                T.wait_flag("mte1", "m", kk % S2)

                                # MMA 计算
                                T.mma(A_L0[kk % S2, :, :], B_L0[kk % S2, :, :], C_L0,
                                      init=T.And(k == 0, kk == 0))

                                T.set_flag("m", "mte1", kk % S2)

                        # =====================
                        # 写回结果
                        # =====================
                        T.set_flag("m", "fix", 0)
                        T.wait_flag("m", "fix", 0)
                        T.copy(C_L0, C[bx * block_M, by * block_N])
                        T.set_flag("fix", "m", 0)

                clear_flag()

    return main


# 实例化
func = matmul_expert(8192, 1024, 8192, 128, 256, 64, 256, 2, 2)

print(func.get_kernel_source())  # 查看生成的 AscendC 代码

# 测试
torch.manual_seed(0)
a = torch.randn(8192, 8192).half().npu()
b = torch.randn(8192, 1024).half().npu()
c = func(a, b)
ref_c = a @ b
torch.npu.synchronize()

torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")
```

## Flag 同步详解

### 管线依赖关系

```
MTE2（GM→L1）→ MTE1（L1→L0）→ M（MMA 计算）→ FIX（L0C→GM）
```

### Flag 含义

| Flag | src → dst | 含义 |
|------|-----------|------|
| `("mte1", "mte2", slot)` | MTE1 释放 → MTE2 可搬入 | L1 slot 已被消费，可覆写 |
| `("mte2", "mte1", slot)` | MTE2 完成 → MTE1 可使用 | L1 数据已就绪 |
| `("m", "mte1", slot)` | M 释放 → MTE1 可搬入 | L0 slot 已被消费 |
| `("mte1", "m", slot)` | MTE1 完成 → M 可计算 | L0 数据已就绪 |
| `("fix", "m", 0)` | FIX 释放 → M 可写入 | L0C 已搬出，可覆写 |
| `("m", "fix", 0)` | M 完成 → FIX 可搬出 | L0C 计算完成 |

### 双缓冲节奏

```
时间 →
L1[0]: Load -------- | 使用中 --------- | 可覆写 Load -------
L1[1]:               | Load ---------- | 使用中 ---------
L0[0]: ...  Copy  MMA | ... Copy MMA   |
L0[1]:       Copy MMA | ...  Copy MMA  |
```

## 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `S1` | L1 双缓冲 stage 数 | 2 |
| `S2` | L0 双缓冲 stage 数 | 2 |
| `block_M` | M 方向分块 | 128 |
| `block_N` | N 方向分块 | 256 |
| `block_K` | L0 级 K 分块 | 64 |
| `K_L1` | L1 级 K 分块 | 256 |
| `core_num` | 使用的 AI Core 数 | 20（910B） |

## 变体：Persistent Kernel

```python
# 替换任务分配方式，使用 T.Persistent 动态领取 tile
for bx, by in T.Persistent(
    [T.ceildiv(M, block_M), T.ceildiv(N, block_N)],
    core_num, cid):
    # 与上面相同的双缓冲流水线
    ...
```
