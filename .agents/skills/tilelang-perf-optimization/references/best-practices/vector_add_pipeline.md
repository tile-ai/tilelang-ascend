# Vector 核内 Pipeline 优化参考

本文档给出一个 Vector 算子核内 MTE2/V/MTE3 流水的参考写法。目标不是覆盖所有 elementwise 算子，而是提供一个清晰的三阶段结构，便于后续把 `T.tile.add` 替换成目标算子的 Vector 计算。

基线来源：`examples/elementwise/elementwise_add.py`

## 适用场景

- Vector 算子主体是 GM -> UB -> Vector compute -> GM。
- 单个逻辑 block 内存在多轮 tile 处理，可以用双 UB stage 交替复用。
- 希望把下一轮 GM -> UB 搬运与当前轮 Vector 计算、UB -> GM 写回重叠。

## 从单缓冲 Vector Add 优化为 Pipeline

`elementwise_add.py` 是一个标准的单缓冲 Vector 算子：每个逻辑 block 只搬入一块 A/B，在 UB 中做一次 `T.tile.add`，再写回 C。

```python
a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

T.barrier_all()
T.tile.add(c_ub, a_ub, b_ub)
T.barrier_all()

T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])
```

这个版本结构简单，但 GM -> UB、Vector compute、UB -> GM 基本串行，`T.barrier_all()` 也会扩大等待范围。优化到 pipeline 版本时，核心思路是把“一个大块一次做完”改成“一个逻辑 block 内再切成多个 tile”，并用两个 UB stage 交替承载当前 tile 和下一 tile。

### 优化步骤

1. **拆分 block 内工作量**

原始版本中每个 Vector sub-core 一次处理 `block_M // VEC_NUM` 行。pipeline 版本新增 `sub_M`，把 `block_M` 拆成多轮：

```python
rows_per_vec = sub_M // VEC_NUM
tiles_per_vec = block_M // sub_M
```

这样每轮只处理 `rows_per_vec * block_N`，主循环可以在 tile 维度上做流水。

2. **把 UB 从单份扩展成双 stage**

原始版本：

```python
a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
```

pipeline 版本：

```python
stages = 2
a_ub = T.alloc_ub((stages, rows_per_vec, block_N), dtype)
b_ub = T.alloc_ub((stages, rows_per_vec, block_N), dtype)
c_ub = T.alloc_ub((stages, rows_per_vec, block_N), dtype)
```

第一维 stage 用于 ping-pong 复用：`cur = tile % stages` 消费当前块，`nxt = (tile + 1) % stages` 预取下一块。

3. **把地址计算改成 tile-relative**

原始版本只有一个起始行：

```python
row = bx * block_M + vid * block_M // VEC_NUM
```

pipeline 版本中每个 tile 都要叠加 `tile * sub_M`：

```python
cur_row = bx * block_M + vid * rows_per_vec + tile * sub_M
next_row = bx * block_M + vid * rows_per_vec + (tile + 1) * sub_M
```

注意 `vid * rows_per_vec` 是当前 Vector sub-core 在本轮 `sub_M` 内的行偏移，`tile * sub_M` 是逻辑 block 内的 tile 偏移。

4. **用事件替换全局 barrier**

单缓冲版本用 `T.barrier_all()` 简单保证搬运、计算、写回顺序。pipeline 版本需要表达更细的 producer-consumer 关系：

```python
T.set_flag("mte3", "mte2", stage)  # stage 可重新搬入
T.set_flag("mte2", "v", stage)     # stage 输入可被计算
T.set_flag("v", "mte3", stage)     # stage 输出可被写回
```

调优时不要机械保留 `T.barrier_all()`；优先把依赖写成 `set_flag/wait_flag`，让 MTE2、Vector、MTE3 三条 pipeline 有机会重叠。

5. **把串行流程改成三阶段**

最终结构固定为：

- prefetch：先搬入第 0 个 tile。
- main body：每轮预取 `tile + 1`，同时消费并写回 `tile`。
- epilogue：消费并写回最后一个已经预取、但主循环还没处理的 tile。

这个拆分能避免在循环里写复杂的 `if tile < tiles_per_vec - 1`，也能让首尾边界更容易检查。

## 三阶段结构

### 1. Prefetch

Prefetch 阶段只做第一块数据的 GM -> UB 搬运，并设置 `mte2 -> v` 事件，给主循环提供第一个可消费的 tile。

```python
T.wait_flag("mte3", "mte2", 0)
T.copy(A[first_row, by * block_N], a_ub[0, :, :])
T.copy(B[first_row, by * block_N], b_ub[0, :, :])
T.set_flag("mte2", "v", 0)
```

### 2. Main Body

Main body 使用 ping-pong stage。每轮先预取 `tile + 1` 到 `nxt`，再消费当前 `cur`，最后把 `cur` 写回 GM 并释放给下一次 MTE2 使用。

```python
for tile in T.serial(0, tiles_per_vec - 1):
    cur = tile % stages
    nxt = (tile + 1) % stages

    T.wait_flag("mte3", "mte2", nxt)
    T.copy(A[next_row, by * block_N], a_ub[nxt, :, :])
    T.copy(B[next_row, by * block_N], b_ub[nxt, :, :])
    T.set_flag("mte2", "v", nxt)

    T.wait_flag("mte2", "v", cur)
    T.tile.add(c_ub[cur, :, :], a_ub[cur, :, :], b_ub[cur, :, :])
    T.set_flag("v", "mte3", cur)

    T.wait_flag("v", "mte3", cur)
    T.copy(c_ub[cur, :, :], C[cur_row, by * block_N])
    T.set_flag("mte3", "mte2", cur)
```

这里的关键点是事件粒度要对齐具体 pipeline，而不是使用 `T.barrier_all()` 进行全局等待：

- `mte3 -> mte2`：对应 UB stage 已经写回完成，可以被下一次 GM -> UB 复用。
- `mte2 -> v`：当前 UB stage 的输入数据已经搬入，可以被 Vector 消费。
- `v -> mte3`：当前 UB stage 的计算结果已经产生，可以写回 GM。

### 3. Epilogue

主循环最后一次只预取了尾 tile，还没有消费它。Epilogue 负责消费并写回最后一个已预取 tile，避免尾块遗漏。

```python
last_tile = tiles_per_vec - 1
last_stage = last_tile % stages

T.wait_flag("mte2", "v", last_stage)
T.tile.add(c_ub[last_stage, :, :], a_ub[last_stage, :, :], b_ub[last_stage, :, :])
T.set_flag("v", "mte3", last_stage)

T.wait_flag("v", "mte3", last_stage)
T.copy(c_ub[last_stage, :, :], C[last_row, by * block_N])
T.set_flag("mte3", "mte2", last_stage)
```

## 参数建议

- `stages = 2`：优先使用双缓冲，结构简单且 UB 占用可控。
- `sub_M`：每个 pipeline step 处理的行数，需能被两个 Vector core 平分。
- `rows_per_vec = sub_M // 2`：每个 Vector sub-core 处理的行数。
- `tiles_per_vec = block_M // sub_M`：主循环次数应至少为 2，否则 pipeline 起停开销可能大于收益。

UB 占用估算：

```text
3 buffers * stages * rows_per_vec * block_N * sizeof(dtype)
```

其中 3 buffers 分别是 `a_ub`、`b_ub`、`c_ub`。需要给临时 buffer 和编译器插入的中间变量留余量。

## 检查点

- `block_M % sub_M == 0`
- `sub_M % 2 == 0`
- `block_N` 对齐当前搬运和 Vector 指令的要求
- 生成的 Ascend C 中应能看到 `MTE3_MTE2`、`MTE2_V`、`V_MTE3` 三类事件

## 完整 Pipeline Kernel 参考

下面代码保留完整的 kernel 结构，便于其他 Vector 算子 pipeline 优化参考。

```python
@tilelang.jit(out_idx=[-1])
def vec_add_pipeline(M, N, block_M, block_N, sub_M, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2
    stages = 2
    rows_per_vec = sub_M // VEC_NUM
    tiles_per_vec = block_M // sub_M

    @T.macro
    def init_flags():
        T.set_flag("mte3", "mte2", 0)
        T.set_flag("mte3", "mte2", 1)

    @T.macro
    def drain_flags():
        T.wait_flag("mte3", "mte2", 0)
        T.wait_flag("mte3", "mte2", 1)

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((stages, rows_per_vec, block_N), dtype)
            b_ub = T.alloc_ub((stages, rows_per_vec, block_N), dtype)
            c_ub = T.alloc_ub((stages, rows_per_vec, block_N), dtype)

            with T.Scope("V"):
                init_flags()

                # Prefetch: fill stage 0 before entering the steady-state loop.
                first_row = bx * block_M + vid * rows_per_vec
                T.wait_flag("mte3", "mte2", 0)
                T.copy(A[first_row, by * block_N], a_ub[0, :, :])
                T.copy(B[first_row, by * block_N], b_ub[0, :, :])
                T.set_flag("mte2", "v", 0)

                # Main body: prefetch tile k + 1 while Vector consumes tile k.
                for tile in T.serial(0, tiles_per_vec - 1):
                    cur = tile % stages
                    nxt = (tile + 1) % stages
                    cur_row = bx * block_M + vid * rows_per_vec + tile * sub_M
                    next_row = bx * block_M + vid * rows_per_vec + (tile + 1) * sub_M

                    T.wait_flag("mte3", "mte2", nxt)
                    T.copy(A[next_row, by * block_N], a_ub[nxt, :, :])
                    T.copy(B[next_row, by * block_N], b_ub[nxt, :, :])
                    T.set_flag("mte2", "v", nxt)

                    T.wait_flag("mte2", "v", cur)
                    T.tile.add(c_ub[cur, :, :], a_ub[cur, :, :], b_ub[cur, :, :])
                    T.set_flag("v", "mte3", cur)

                    T.wait_flag("v", "mte3", cur)
                    T.copy(c_ub[cur, :, :], C[cur_row, by * block_N])
                    T.set_flag("mte3", "mte2", cur)

                # Epilogue: consume and store the final prefetched tile.
                last_tile = tiles_per_vec - 1
                last_stage = last_tile % stages
                last_row = bx * block_M + vid * rows_per_vec + last_tile * sub_M

                T.wait_flag("mte2", "v", last_stage)
                T.tile.add(c_ub[last_stage, :, :], a_ub[last_stage, :, :], b_ub[last_stage, :, :])
                T.set_flag("v", "mte3", last_stage)

                T.wait_flag("v", "mte3", last_stage)
                T.copy(c_ub[last_stage, :, :], C[last_row, by * block_N])
                T.set_flag("mte3", "mte2", last_stage)

                drain_flags()

    return main
```
