# Gated MLA Prefill 算子设计文档

## 1. 概述

### 1.1 算子名称

`gated_mla_prefill` —— Kimi K3 Gated MLA 层的 prefill 前向。在 latent KV cache 上做
吸收式（absorbed）多头潜在注意力，再做值恢复与逐通道 sigmoid 输出门，返回 `W_O` 之前的结果。

**编程模式**：Expert（显式内存分配 + 手动跨核同步 + `T.tile.xxx` 原语）
**目标硬件**：昇腾 910B3，20 个 AI 核

### 1.2 功能描述

七个输入，一个输出：

| 张量 | 形状 | dtype |
|---|---|---|
| `q_content` | `[B, Tq, H, 128]` | fp16 / bf16 |
| `q_aux` | `[B, Tq, H, 64]` | fp16 / bf16 |
| `latent_kv` | `[B, Tk, 512]` | fp16 / bf16 |
| `key_aux` | `[B, Tk, 64]` | fp16 / bf16 |
| `w_k` | `[H, 128, 512]` | fp16 / bf16 |
| `w_v` | `[H, 128, 512]` | fp16 / bf16 |
| `gate_logits` | `[B, Tq, H, 128]` | fp16 / bf16 |
| `metadata` | `[B, 8]` | int32 |
| **输出** | `[B, Tq, H, 128]` | fp16 / bf16 |

`metadata` 前三列是运行期的有效 query 长度、有效 cache 长度和 `query_start`，**是 kernel 的
运行期输入，不是编译期参数**——同一份编译产物要服务不同的有效长度。

计算本身：

```
ql[b,t,h,:] = q_content[b,t,h,:] @ w_k[h]            # 把 W_K 吸收进 query
s[b,t,h,k]  = scale * ( ql[b,t,h,:]·c[b,k,:] + q_aux[b,t,h,:]·e[b,k,:] )
p[b,t,h,k]  = softmax_k( s[b,t,h,k] )                 # 仅对可见 k
z[b,t,h,:]  = Σ_k p[b,t,h,k] · c[b,k,:]
y[b,t,h,:]  = ( z[b,t,h,:] @ w_v[h]ᵀ ) * sigmoid( gate_logits[b,t,h,:] )
```

其中 `c = latent_kv`、`e = key_aux`，`scale` 默认 `1/sqrt(192)`。

**可见性**：`k < min(query_start + t + 1, kv_len)`，即左上对齐的 causal 掩码加上
`kv_len` 钳位。`query_start` 让一段 query 可以落在更长 cache 的任意偏移上。

**吸收式是关键**。常规写法要把 512 宽的 latent 展开成每头 128 宽的 K/V 再算注意力；
吸收式把 `W_K` 先折进 query，直接拿 512 宽的 latent 当 K，也直接拿它当 V。
两者代数等价（对潜在秩的求和可以换序），但吸收式不产生 `[B,Tk,H,128]` 的中间张量。

---

## 2. 实现结论

### 2.1 编程模式

**Expert 模式**，四个自动 pass 全部关闭（见 §7）。具体表现：

- 显式 `T.Scope("C")` / `T.Scope("V")` 划分 Cube 核与 Vector 核的代码区
- 显式 `T.alloc_L1` / `T.alloc_L0A` / `T.alloc_L0B` / `T.alloc_L0C` / `T.alloc_ub`
- 显式 `T.annotate_address({...})` 给出每个 scope 内互不重叠、32 字节对齐的地址
- 手写 `T.set_cross_flag` / `T.wait_cross_flag` 做 Cube↔Vector 同步

选 Expert 而不是 Developer，是因为本算子的核心优化（§5 的跨 wave 错拍流水）要求
**在一个 wave 内让 Cube 的 QK 先于 Vector 的 PV 发射**。这个顺序是手工编排的结果，
自动同步 pass 看到的是一条串行依赖链，无法产生这种错拍。

### 2.2 流水入口与标量回退的边界

两个入口，由 host 侧 `pipeline_domain` 静态判定，**编译期二选一**：

```python
iterations = ceil(cache_capacity / block_n)
single_block_uniform = (iterations == 1) and (heads % block_m == 0)
pipeline_domain = single_block_uniform and (batch == 1)
```

三个条件的性质不同：

- `iterations == 1`：**结构条件**。错拍流水把 wave `w` 的 QK 提到 wave `w-1` 的 PV 之前，
  多轮 cache 迭代会让这个提前量跨越迭代边界，编排失效。
- `heads % block_m == 0`：**结构条件**。保证每个行块只属于一个 batch，不跨 batch 边界。
- `batch == 1`：**正确性边界，不是性能取舍**。`c_l1` / `e_l1` 按 batch tag 常驻 L1；
  错拍之后 wave `w` 的 QK 会为新 batch 重新加载 C/E，覆写 wave `w-1` 的 PV 仍在读的内容。
  单 batch 时 cache tag 永不改变，冲突不可能发生。

放开第三条需要第二组 C/E 槽位，代价是 **+144 KiB L1**（`c_l1` 128 KiB + `e_l1` 16 KiB）。
按 §6 的预算，`188 + 144 = 332 KiB` 仍在 512 KiB 的 L1 之内，**所以这是一个尚未在硬件上
验证的缺口，而不是容量墙**。`test_example_gated_mla_prefill.py` 里有一条负判据钉住这个
边界：流水入口在 `batch ≥ 2` 和多 KV 块时**必须不被编译**；哪天它开始被编译，说明本文档
描述的域边界已经变了。

### 2.3 block_n 的选择

`public_prefill_block_n(batch, query_capacity, heads, cache_capacity, scale)` 是 host 侧
选择器，**只对实测过的那一个 profile** `(1, 128, 96, 128)` 且默认 scale 返回 `128`，
其余一律 `64`。

这个契约刻意窄。它的后果是：只有实测 profile 能进流水入口，其余形状都走标量回退。
§9 给出了这个取舍的量化代价。

---

## 3. API 状态说明

### 3.1 为什么关掉自动内存规划

`TL_ASCEND_MEMORY_PLANNING` 关闭后，每个 scope 的地址由 host 侧 `buffer_plan()` 算出，
再用 `T.annotate_address` 下发。这样做有三个理由：

1. **地址是可审计的纯元数据**。`buffer_plan()` 是纯函数，不碰设备，单元测试可以直接
   断言每个 buffer 的 offset 和总量（§6 的表就是它的输出）。
2. **流水的 2 槽 workspace 需要确定的槽位布局**。自动规划不知道两个槽要轮换。
3. **bf16 的 `prob_round_ub` 是条件分配的**，开启时才占 UB；由 host 决定更直观。

### 3.2 跨核同步原语的用法

`T.set_cross_flag(unit, id)` / `T.wait_cross_flag(id)` 是 Cube 与 Vector 之间的信号量。
`unit` 是发信方最后一条写操作所在的流水单元（`"FIX"` / `"MTE3"` / `"V"`），用于让
硬件知道该等哪条流水排空。

基线实现用 4 个 flag 做严格乒乓；错拍流水改用 **6 个信号量，按 workspace 配对**（§5.2）。

### 3.3 掩码的块级跳过

`c_max_visible = min(query_start + 行内最大 query 下标 + 1, kv_len)` 在 Cube 侧逐行块算出，
然后 `if k * block_n < c_max_visible` 决定第 `k` 个 cache 块是否参与计算。

**被掩码完全遮住的 cache 块不会被搬运，也不会参与矩阵乘。** 这是一个真实的性能性质：
同一个 `Tq=64, Tk=128` 形状，`query_start=0` 时只有一块活、`query_start=64` 时两块都活，
实测耗时差一倍（79.29 µs 对 139.80 µs）。引用本算子的耗时时必须说明 `query_start`。

---

## 4. Kernel 设计

三次 kernel 发射，由 `gated_mla_prefill()` 串联：

| # | kernel | 计算 | 引擎 |
|:-:|---|---|:-:|
| 1 | `q_absorb_kernel` | `ql = q_content @ w_k`，把 `W_K` 吸收进 query | Vector |
| 2 | `build_cube_attention` | latent 注意力：QK、在线 softmax、PV | Cube + Vector |
| 3 | `value_gate_kernel` | `y = (z @ w_vᵀ) * sigmoid(gate_logits)` | Vector |

第 2 次发射是本算子的主体，也是唯一被优化的部分；第 1、3 次是直白实现。

### 4.1 第 2 次发射的任务划分

```
rows        = query_capacity * heads          # 行 = (query, head) 对
row_tiles   = ceil(rows / block_m)
total_tasks = batch * row_tiles
blocks      = min(max_blocks=20, total_tasks) # 物理核数上限
waves       = ceil(total_tasks / blocks)
```

每个任务处理 `block_m = 32` 行。实测 profile 下 `rows = 128 × 96 = 12288`，
`total_tasks = 384`，`blocks = 20`，`waves = 20`。

### 4.2 在线 softmax

Vector 侧按行维护 `maximum` / `normalizer` / `previous`，Cube 侧算出的 score 块
经 workspace 交给 Vector 做指数与归一化，概率再送回 Cube 做 PV。
`partial` 累加器是 FP32，最后一轮才归一化并落回 fp16/bf16。

### 4.3 bf16 的概率修复

`probability_precision_mode(dtype)` 对 bf16 返回带残差的模式：bf16 的 8 位尾数不足以
表示概率，`prob_round_ub` 额外保存一份 fp32 概率用于补偿（`use_p_residual`）。
这是 bf16 比 fp16 慢的原因（实测 profile 上 PTO 贵 26%、AscendC 贵 15%），
**而且只在流水入口上贵**——因为它给被错拍遮住的那个阶段又加了向量活。

---

## 5. 数据流与计算

### 5.1 基线：严格乒乓

Cube 算完 wave `w` 的 QK → 置 flag → Vector 做 softmax → 置 flag → Cube 做 PV → …
两个核严格交替、**从不重叠**。实测 Cube 只活跃 10.3%，等待占 44%。

### 5.2 错拍：跨 wave Cube/Vector 软件流水

把 Cube 的循环**整体错开一个 wave**：wave `w` 的 QK 在 wave `w-1` 的 PV **之前**发射。
于是 Vector 做 wave `w-1` 的 softmax 时，Cube 已经在算 wave `w` 的 QK。

实现上三处改动：

1. **三个 workspace 各加一个 2 槽维**（`score` / `probability` / `partial`），
   让相邻两个 wave 的数据不互相覆写。
2. **4 个 flag 改成 6 个信号量，按 workspace 配对**——每个 workspace 一对
   「就绪 / 已消费」：

   | 信号量 | 值 | 含义 | 方向 |
   |---|:-:|---|---|
   | `SEM_SCORE_C2V` | 0 | score 槽就绪 | Cube → Vector |
   | `SEM_PROB_V2C` | 1 | probability 槽就绪 | Vector → Cube |
   | `SEM_PART_C2V` | 2 | partial 槽就绪 | Cube → Vector |
   | `SEM_PART_V2C` | 3 | partial 槽已消费 | Vector → Cube |
   | `SEM_SCORE_V2C` | 4 | score 槽已消费 | Vector → Cube |
   | `SEM_PROB_C2V` | 5 | probability 槽已消费 | Cube → Vector |

   「已消费」这一半是错拍引入的：不等对方用完就复用槽位会直接写坏数据。

3. **消费方在进入循环前先预置一个槽**，循环结束后两侧各自排空自己预置的那一个。

### 5.3 为什么收益跟着并行度走

填充与排空各占一个 wave，所以开销是 `1/waves`。实测加速随 wave 数单调上升
（7 wave → 1.358×，39 wave → 1.442×），而 wave 数相同、绝对耗时差近一倍的两格加速持平。
**决定收益的是并行度，不是工作量。**

---

## 6. 片上内存预算

910B 上限：**L1 512 KiB、L0C 128 KiB、UB 192 KiB**。
下表由 `buffer_plan()` 直接算出（`block_m = 32`、`max_blocks = 20`、`B1/H96/Tq128`）：

| 配置 | L1 | L0C | UB | workspace |
|---|---:|---:|---:|---:|
| `block_n=128` fp16 流水（实测 profile） | 188.0 KiB | 80.0 KiB | 93.9 KiB | 3.44 MiB |
| `block_n=128` bf16 流水 | 188.0 KiB | 80.0 KiB | 101.9 KiB | 3.75 MiB |
| `block_n=64` fp16 标量（回退） | 112.0 KiB | 72.0 KiB | 79.3 KiB | 1.48 MiB |
| `block_n=64` bf16 标量 | 112.0 KiB | 72.0 KiB | 83.3 KiB | 1.56 MiB |

主要占用项：

- **L1**：`c_l1`（`block_n × 512 × 2`）是大头，`block_n=128` 时 128 KiB；
  其次 `q_l1` 32 KiB、`e_l1` 16 KiB、`p_l1` 8 KiB、`qa_l1` 4 KiB。
- **L0C**：`partial_l0c`（`block_m × 512 × 4` = 64 KiB）加 `score_l0c`（16 KiB）。
- **UB**：`z` 和 `partial` 各 32 KiB（`hm × 512 × 4`，`hm = block_m/2 = 16`），
  其余是 score / 概率 / 掩码相关的小块。bf16 多一块 `prob_round_ub` 8 KiB，
  流水模式多两个 `normalizer_slot` 各 64 B。
- **workspace**：`slots × blocks × (score + probability + partial)`，
  流水把 `slots` 从 1 提到 2，因此翻倍。

三个 scope 都留有余量，**最紧的是 L1 的 188/512 KiB**。§2.2 讨论的放开 `batch == 1`
需要再要 144 KiB，332 KiB 仍然放得下。

**`block_n ≥ 256` 不可用**：实测被编译器拒绝，报 `unsupported attention tile`。
这不是预算问题，是可用 tile 的限制，后果见 §10。

---

## 7. pass_configs

```python
EXPERT_PASSES = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
}
```

三次发射共用同一份配置。逐条说明：

- `AUTO_SYNC: False` —— 同步点由 §5.2 的六信号量协议手工给出。
- `MEMORY_PLANNING: False` —— 地址由 `buffer_plan()` 给出，理由见 §3.1。
- `AUTO_CV_COMBINE: False` —— Cube/Vector 的代码区由 `T.Scope` 显式划分，
  不交给编译器合并。
- `AUTO_CV_SYNC: False` —— 这条是错拍流水的前提。自动 CV 同步会在 Cube 与 Vector
  之间插入保守的屏障，恰好消灭掉我们要的那一拍提前量。

---

## 8. 与 Kimi K3 语义的对齐方式

对照 Kimi K3 released config（`config.json`）：

| 配置项 | 值 | 本算子 |
|---|---|---|
| `hidden_size` | 7168 | — |
| `num_attention_heads` | 96 | `H = 96` |
| `kv_lora_rank` | 512 | `latent_kv` 宽度 |
| `qk_nope_head_dim` | 128 | `q_content` 末维 |
| `qk_rope_head_dim` | 64 | `q_aux` / `key_aux` 末维 |
| `v_head_dim` | 128 | 输出末维 |
| `mla_use_nope` | `True` | 吸收式走 nope 通道 |
| `mla_use_output_gate` | `True` | **第 3 次发射的 sigmoid 门** |
| `torch_dtype` | `bfloat16` | bf16 路径（§4.3） |

`scale = 1/sqrt(192) = 1/sqrt(128 + 64)`，即 nope 与 rope 两段维度之和。

**输出门是 released config 开启的**，所以它是算子语义的一部分，不是可选项；
它在 `value_gate_kernel` 里实现为 Vector 侧的数值稳定 sigmoid
（按符号分支选 `1/(1+e^{-|x|})` 或 `e^{-|x|}/(1+e^{-|x|})`），而不是只存在于参考实现里。

**范围**：只有 prefill 前向。decode、cache 追加、packed 布局适配和完整模型集成不在本算子内。

---

## 9. 验证结论

详细数据见 `benchmark_gated_mla_prefill.md`（中文版 `_zh.md`），此处只给结论。

### 9.1 精度

模块带**两个不共享任何收缩顺序**的参考实现：吸收式（kernel 实现的那种）与展开式
（先把 latent 展成逐头 K/V 再走常规注意力）。6 个 CPU 用例实测两者最差相差
**0.37 ULP**，这是误差下限——kernel 容差若定得比两个正确参考之间的差距还紧，
量的就是参考而不是 kernel。

据此选定 `rtol = 2⁻⁹`、`atol = 2⁻¹⁰`，明显宽于该下限。

### 9.2 判据分四类

- **预算判据**：整链对 FP32 参考，位置敏感——先要求超出有效 query 长度的行**精确为零**，
  再对其余用上述容差。覆盖 fp16/bf16 × PTO/AscendC。
- **解析判据**：令 `QL = QA = 0`、`latent_kv[k,:] = k`，则期望值全是 0.5 的整数倍，
  而掩码多收或少收一个 key 恰好移动 0.5，故判据是「四舍五入到最近 0.5 后精确相等」。
- **逐位相同判据**：同批内各 batch 与单独计算必须逐位相同；空 batch 不得扰动相邻活跃 batch。
- **负判据**：非法形状、非连续操作数、错误 dtype 必须抛出具体异常；以及 §2.2 的域边界。

### 9.3 性能

实测 profile（`B1/H96/Tq128/Tk128`，即 K3 工作点）对官方
`npu_fused_infer_attention_score_v2` 达 **74.9%**（PTO，两臂均先过合格判据）。

固定形状只动 `block_n`，可把流水的收益与形状的影响分开：

| 形状 | 标量 `bn=64` | 流水 `bn=128` | 倍数 |
|---|---:|---:|---:|
| `B1/H96/Tq128/Tk128` | 204.54 µs | 92.98 µs | **2.20×** |
| `B1/H96/Tq64/Tk128` | 141.94 µs | 54.31 µs | **2.61×** |

两个入口的归一化误差在所示位数上完全相同，说明它们算的是同一件事。

### 9.4 CI 成本

| 入口 | 内容 | 墙钟 |
|---|---|---:|
| `test_example_gated_mla_prefill.py` | 28 项（含 6 项纯 CPU 金标互证） | 200 s |
| `example_gated_mla_prefill.py` `__main__` | 5 个配置 × 2 后端 | 233 s |

两者可并行，墙钟约等于较慢的一个（约 4 分钟）。测试一栏是**冷 JIT 缓存**的实测值，
即 CI 的起点；缓存命中时为 173 s。

---

## 10. 当前实现边界

- **流水入口仅在 `iterations == 1`、`heads % block_m == 0` 且 `batch == 1` 时编译。**
  前两条是结构条件，第三条是正确性边界（§2.2）。域外形状走标量入口，实测零收益。
- **`block_n ≥ 256` 被拒（`unsupported attention tile`）**，因此 `Tk > 128` 时
  任何 `block_n` 都到不了流水入口。这是可用 tile 决定的上限，不是调优缺口。
- **`public_prefill_block_n` 比「能合格的范围」更窄。** 在 `B1/H96/Tq64/Tk128` 上，
  流水的 `block_n = 128` 精度合格且快 2.61×，但选择器返回 64。未放宽的理由：
  其契约刻意为「仅实测 profile」，且新打开的这条路径目前没有测试覆盖。
- **引用耗时必须说明 `query_start`**，原因见 §3.3 的块级跳过。
- **普通（非 Graph）调用无显著提速**，该路径由主机侧开销主导，本优化按设计不改善它。
- **bf16 有计时、无对标比例**，因为该 dtype 下没有能通过同等精度资格的外部参考。
- 不含 decode 路径、cache 追加、packed 布局适配与完整模型集成。
