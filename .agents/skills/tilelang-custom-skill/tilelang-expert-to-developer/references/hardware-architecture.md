# 硬件架构基础

## 昇腾 NPU 内存层级

```
GM（全局内存）—— 所有核共享
  ↕ T.copy（MTE2 搬入 / MTE3 搬出）
L1（Cube 核缓存）/ UB（Vector 核统一缓冲区）—— 每核私有
  ↕ T.copy（MTE1 搬运引擎）
L0A / L0B（矩阵输入寄存器）→ L0C（矩阵输出累加寄存器）
```

**关键约束：不可跨级访问。** 数据必须按层级逐级搬运，不可 GM→L0 直接访问。

---

## 存储层级对照

| 存储层级 | 容量（典型） | 用途 | Developer API | Expert API |
|---------|------------|------|--------------|-----------|
| GM | 大 | 全局输入/输出 | `T.Tensor` 参数 | `T.Tensor` 参数 |
| L1 | 64-256KB/核 | Cube 核数据缓存 | `T.alloc_shared` | `T.alloc_L1` |
| UB | 16-32KB/核 | Vector 核工作缓冲 | `T.alloc_shared` | `T.alloc_ub` |
| L0A | 256B/MMA | GEMM 矩阵 A 输入 | 自动（gemm_v0 内部） | `T.alloc_L0A` |
| L0B | 256B/MMA | GEMM 矩阵 B 输入 | 自动（gemm_v0 内部） | `T.alloc_L0B` |
| L0C | 512B/MMA | GEMM 累加输出 | `T.alloc_fragment` | `T.alloc_L0C` |

---

## 计算引擎

| 引擎 | 类型 | 功能 | 管线标识 |
|------|------|------|---------|
| Cube 核 | 矩阵计算 | GEMM / MMA 矩阵乘加 | `"m"` / `"M"` |
| Vector 核 | 向量计算 | Element-wise、Reduce、Softmax 等 | `"v"` / `"V"` |
| MTE1 | 搬运引擎 | L1 ↔ L0 数据搬运 | `"mte1"` / `"MTE1"` |
| MTE2 | 搬运引擎 | GM → L1/UB 数据搬入 | `"mte2"` / `"MTE2"` |
| MTE3 | 搬运引擎 | L1/UB → GM 数据搬出 | `"mte3"` / `"MTE3"` |
| FIX | 辅助引擎 | L0C → GM 搬出 | `"fix"` / `"FIX"` |

---

## 核间协作模型

每个 AI Core 包含一个 Cube 核和一个 Vector 核。

```python
with T.Kernel(block_num, is_npu=True) as (cid, vid):
    # cid: 核编号（Core ID），范围 [0, block_num)
    # vid: 向量线程编号（Vector ID），取值 0 或 1（每核 2 个 Vector 线程）
```

- Cube 核和 Vector 核共享同一核的 L1/UB 存储
- Cube 核和 Vector 核只能通过 **GM / workspace** 交换数据
- 核间同步使用 `T.set_cross_flag` / `T.wait_cross_flag`（Expert）或自动插入（Developer）

---

## 数据流向示例

### GEMM 数据流
```
GM → (MTE2) → L1 → (MTE1) → L0A/L0B → (Cube MMA) → L0C → (FIX) → GM
```

### Vector 计算数据流
```
GM → (MTE2) → UB → (Vector) → UB → (MTE3) → GM
```

### Cube + Vector 融合数据流
```
GM → L1 → L0A/L0B → L0C → (FIX) → GM(workspace)
                                         ↓
                              GM(workspace) → (MTE2) → UB → (Vector) → UB → (MTE3) → GM
```

---

## 硬件常量

| 参数 | 910B 典型值 | 说明 |
|------|-----------|------|
| AI Core 数 | 20-24 | 每个核含 1 Cube + 1 Vector |
| L0 总容量 | 64KB | L0A + L0B + L0C 共享 |
| GEMM block 对齐 | 16 | block_M/block_N 通常为 16 的倍数 |
| VEC_NUM | 2 | 每核 2 个 Vector 线程 |
