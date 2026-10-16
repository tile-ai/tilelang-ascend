# situ_glu 算子设计文档

## 1. 概述

### 1.1 算子名称

situ_glu（SiTU Gated Linear Unit，SiTU 门控线性单元）

### 1.2 功能描述

对输入张量 `x` 沿指定维度 `dim` 切分为门控（gate）与上路径（up）两半，按 SiTU 公式计算门控激活并相乘输出。对标 CANN ops-nn 仓 `activation/situ_glu`（aclnnSituGlu / cann_ops_nn.situ_glu），作为其在 TileLang-Ascend 上的等价实现。

### 1.3 数学公式

对输入 `x`（dim 维大小为偶数，令 `h = x.shape[dim] // 2`）：

$$
\begin{aligned}
\text{activate\_left}=\text{True（默认）}: \quad & gate = x_{[\dim, :h]}, \quad up = x_{[\dim, h:]} \\
\text{activate\_left}=\text{False}: \quad & gate = x_{[\dim, h:]}, \quad up = x_{[\dim, :h]} \\[6pt]
situ\_a &= \beta \cdot \tanh\!\left(\frac{gate}{\beta}\right) \cdot \text{sigmoid}(gate) \\[6pt]
up' &= \begin{cases} linear\_beta \cdot \tanh\!\left(\dfrac{up}{linear\_beta}\right) & linear\_beta > 0 \\ up & linear\_beta \le 0 \end{cases} \\[6pt]
y &= situ\_a \cdot up'
\end{aligned}
$$

输出 `y` 的 shape 与 `x` 维度数一致，dim 维大小减半，其余维度不变。

### 1.4 算法描述

1. **合轴（host 侧，纯 metadata view）**：将 `x` 按 `dim` 合并为二维 `(M, N)`，其中 `M = prod(shape[:dim])`，`N = prod(shape[dim:])`（即 cut），`N` 必须为偶数。
2. **分块搬运**：输出空间按 `(block_M, block_N)` 分块（每 block 内 `vid ∈ {0,1}` 子块各处理 `sub_M = block_M//2` 行），从 GM 分别搬入 gate 半行 tile 与 up 半行 tile（列偏移相差 `N//2`，由 `activate_left` 决定先后）。
3. **计算（统一 fp32）**：`tanh` 无 TileLang 原语，用数值稳定恒等式 `tanh(t) = 2·sigmoid(2t) − 1` 组合实现；gate 路径 2 次 sigmoid + 标量乘/减 + 逐元素乘；`linear_beta > 0` 时 up 路径再做同构 tanh 有界化。
4. **回写**：将结果 tile 写回输出 `B (M, N//2)`。

> **⚠️ Host 侧 Buffer 操作约束**：host 侧仅允许经证明共享 storage 的 metadata view（contiguous 张量的 reshape，见 §1.4-审计）与 kernel 调用；禁止任何真实数据搬运与 aclnn 调用。

**[HOST-METADATA-AUDIT]**

| operation | input_stride → output_stride | shares_storage | aclnn_or_physical_copy | result |
|-----------|------------------------------|----------------|------------------------|--------|
| `x.reshape(M, N)`（合轴） | contiguous 输入，reshape 仅改 shape/stride，storage 不变 | true（前置校验 `x.is_contiguous()`） | false | allow |
| `y_2d.view(orig_shape)`（输出还原） | kernel 输出 buffer 本身 contiguous，view 仅改 metadata | true | false | allow |

非 contiguous 输入在 host 侧 metadata 校验阶段显式报错（v1 不支持，见 §10.3），不触发任何隐式拷贝。

### 1.5 数据流图

```
GM[A (M, N)] --T.copy--> UB[gate_ub] ─┐
GM[A (M, N)] --T.copy--> UB[up_ub] ──┤
                                      ├─ fp32 计算（sigmoid/mul/sub）─> UB[y_ub] --T.copy--> GM[B (M, N//2)]
（fp16/bf16 时 GM→UB 后先 cast 到 fp32，算完 cast 回原 dtype）
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: 混合（Developer 语义主体 + Stage-3 性能热路径 Expert 化）

Stage 3 后内核为 flag 流水版（`elementwise_add_pipeline.py` 模式）：显式
`T.set_flag/wait_flag` 双缓冲流水（MTE2 预取 ∥ V 计算 ∥ MTE3 回写），pass_configs
保持全关（`AUTO_SYNC` 会插入 `PIPE_ALL` 循环屏障杀死流水，已实测确认）。host 侧
（校验/合轴 view/参数分派）仍为 Developer 语义。

### 2.2 选型理由

- 算子为**纯 Vector 逐元素融合**（tanh·sigmoid·mul），无 matmul、无归约、无 L0 参与判定，不需要手动 C/V 划分与核间流水线
- 计算步骤全部为 UB 内 tile 级原语，`AUTO_SYNC` / `AUTO_CV_COMBINE` 两个 pass 即可覆盖同步与线程合并，无编译器无法覆盖的复杂同步场景
- 与已验证参考实现 `examples/developer_mode/gelu_mul_developer.py`（同为 GLU 族 split+激活+乘）结构完全同构，开发与维护成本最低

**实现期修正（已验证）**：`threads=2` 形式与 `T.tile.sigmoid` 不兼容——`AscendVidReduction` pass 会将二元原语按子块减半（2048）但不改写 sigmoid 的 count（保持整 tile 4096），生成 `Sigmoid(t_ub, t_ub, tmp, 4096)` 违反 AscendC 断言 `calCount <= srcTensor.GetSize()`（buffer 仅 2048）。故改用 `examples/activation/tanh.py` 已验证的等价 Developer 形式：`T.Kernel(m_num * n_num, is_npu=True) as (cid, vid)` + `VEC_NUM=2` 手动子块切分（仍是 `alloc_shared` + 自动同步，无手动 Scope/barrier）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared((sub_M, block_N), dtype)`（Developer 规范，自动映射 UB；sub_M = block_M // VEC_NUM） |
| 计算方式 | `T.Kernel(m_num * n_num, is_npu=True) as (cid, vid)`，`VEC_NUM=2` 子块各处理半 tile 行（tanh.py 验证模式） |
| 作用域 | 编译器自动分离（无显式 `T.Scope`） |
| 同步方式 | 自动同步（`TL_ASCEND_AUTO_SYNC`），无手动 barrier |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `t = (2/β) · gate` | 标量乘，折叠 tanh 的内层缩放与 2 倍系数 |
| 2 | `s1 = sigmoid(t)` | tanh 组合第 1 步 |
| 3 | `tanh_t = 2·s1 − 1` | 恒等式 `tanh(t) = 2σ(2t) − 1`（数值稳定，大 \|t\| 饱和不溢出） |
| 4 | `s2 = sigmoid(gate)` | gate 的 sigmoid |
| 5 | `situ_a = β · tanh_t · s2` | 逐元素乘 + 标量乘（β=1.0 时编译期折叠省去标量乘） |
| 6 | `up' = λ·tanh(up/λ)`（仅 λ>0） | 与步骤 1-3 同构：`mul(2/λ) → sigmoid → mul 2 → sub 1 → mul λ` |
| 7 | `y = situ_a · up'`（或 `· up`） | 最终逐元素乘 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 搬入 | gate/up tile | `T.copy(A[m0:m0+vm, goff+n0:goff+n0+vn], gate_ub[:vm, :vn])` | 动态切片（valid extent） | Developer |
| 1 | `t = (2/β)·gate` | `T.tile.mul(t_ub, gate_ub, 2.0/beta)` | 标量作 src1 | Developer |
| 2 | `s1 = σ(t)` | `T.tile.sigmoid(t_ub, t_ub)` | 原地 unary | Developer |
| 3 | `2s1−1` | `T.tile.mul(t_ub, t_ub, 2.0)` + `T.tile.sub(t_ub, t_ub, 1.0)` | 标量二元 | Developer |
| 4 | `s2 = σ(gate)` | `T.tile.sigmoid(s_ub, gate_ub)` | — | Developer |
| 5 | `situ_a` | `T.tile.mul(t_ub, t_ub, s_ub)`（+ `T.tile.mul(t_ub, t_ub, beta)` 当 β≠1） | — | Developer |
| 6 | `up'` | 同 1-3 结构，作用于 `up_ub`/`s_ub` | 仅 λ>0 编译期分支 | Developer |
| 7 | `y` | `T.tile.mul(y_ub, t_ub, up_ub)`（或 `s_ub`） | — | Developer |
| cast | fp16/bf16 ↔ fp32 | `T.tile.cast(dst, src, mode, count)` | 入向 `CAST_NONE`；出向 bf16 `CAST_RINT` / fp16 `CAST_NONE` | Developer |
| 搬出 | y tile | `T.copy(y_ub[:vm, :vn], B[m0:m0+vm, n0:n0+vn])` | 动态切片 | Developer |

### 3.3 计算伪代码

```python
# JIT 签名：block_M/block_N 切块；beta、linear_beta、activate_left 为编译期常量参数
@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def situ_glu(block_M, block_N, beta, linear_beta, activate_left, dtype="float"):
    M = T.symbolic("M")            # pre（合轴后行数）
    N = T.symbolic("N")            # cut（合轴后列数，必须为偶数）
    VEC_NUM = 2
    sub_M = block_M // VEC_NUM     # 每个子块处理的行数
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N // 2, block_N)
    ACC = "float32"
    need_cast = dtype not in ("float", "float32")

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N // 2), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx, by = cid // n_num, cid % n_num
            row0 = bx * block_M + vid * sub_M
            n0 = by * block_N
            gate_off = 0 if activate_left else N // 2   # 编译期分支
            up_off   = N // 2 if activate_left else 0

            # 每子块 (sub_M, block_N)；fp32 5 个 fp32 buffer + cast 路径 3 个
            # dtype buffer（无条件分配，保证 if 分支共享作用域，swi_glu_v2 模式）
            gate_ub = T.alloc_shared((sub_M, block_N), ACC)
            up_ub   = T.alloc_shared((sub_M, block_N), ACC)
            t_ub    = T.alloc_shared((sub_M, block_N), ACC)
            s_ub    = T.alloc_shared((sub_M, block_N), ACC)
            y_ub    = T.alloc_shared((sub_M, block_N), ACC)
            gate_raw = T.alloc_shared((sub_M, block_N), dtype)
            up_raw   = T.alloc_shared((sub_M, block_N), dtype)
            y_raw    = T.alloc_shared((sub_M, block_N), dtype)

            # 1. 搬入：全 tile + 标量起点；前端切片分析自动 clamp 到张量边界并
            #    对读入零填充（valid extent 的自动化实现，见 §5.4）
            if need_cast:
                T.copy(A[row0, gate_off + n0], gate_raw)
                T.copy(A[row0, up_off + n0], up_raw)
                T.tile.cast(gate_ub, gate_raw, "CAST_NONE", elem_num)
                T.tile.cast(up_ub, up_raw, "CAST_NONE", elem_num)
            else:
                T.copy(A[row0, gate_off + n0], gate_ub)
                T.copy(A[row0, up_off + n0], up_ub)

            # 2. gate 路径：situ_a = beta * tanh(gate/beta) * sigmoid(gate)
            T.tile.mul(t_ub, gate_ub, 2.0 / beta)   # t = 2g/β
            T.tile.sigmoid(t_ub, t_ub)              # s1 = σ(t)
            T.tile.mul(t_ub, t_ub, 2.0)             # 2s1
            T.tile.sub(t_ub, t_ub, 1.0)             # tanh(g/β)
            T.tile.sigmoid(s_ub, gate_ub)           # s2 = σ(g)
            T.tile.mul(t_ub, t_ub, s_ub)            # tanh·σ
            if beta != 1.0:                         # 编译期折叠
                T.tile.mul(t_ub, t_ub, beta)        # situ_a

            # 3. up 路径（linear_beta > 0 编译期分支）
            if linear_beta > 0:
                T.tile.mul(s_ub, up_ub, 2.0 / linear_beta)
                T.tile.sigmoid(s_ub, s_ub)
                T.tile.mul(s_ub, s_ub, 2.0)
                T.tile.sub(s_ub, s_ub, 1.0)         # tanh(u/λ)
                T.tile.mul(s_ub, s_ub, linear_beta) # u'
                T.tile.mul(y_ub, t_ub, s_ub)        # y = situ_a · u'
            else:
                T.tile.mul(y_ub, t_ub, up_ub)       # y = situ_a · u

            # 4. 搬出
            if need_cast:
                T.tile.cast(y_raw, y_ub, out_cast_mode, elem_num)
                T.copy(y_raw, B[row0, n0])
            else:
                T.copy(y_ub, B[row0, n0])
    return main
```

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|----------|----------|
| `T.Kernel(grid, is_npu=True) as (cid, vid)` | `examples/activation/tanh.py:23`；`kernel.py:215`（blocks 接受 PrimExpr） | ✅ 示例已验证 + 本算子编译验证 |
| `T.alloc_shared` | `tanh.py:27-31`（(cid, vid) + alloc_shared 组合） | ✅ 示例已验证 |
| `T.tile.mul/sub(dst, src, scalar)` | `gelu_mul.py:43`（`T.tile.mul(temp_ub, temp_ub, 0.044715)`）、`swi_glu.py:57` | ✅ 示例已验证 |
| `T.tile.sigmoid(dst, src)` 原地 | `ascend_tile.py:1265`；lowering `AscendC::Sigmoid`（`codegen_ascend.cc:704`）；fp32 隐式 workspace 测试 `testing/python/language/test_tilelang_ascend_language_elementwise.py:5380-5399` | ✅ 测试已验证（fp32, rtol 1e-4）；原地安全性经 CANN `sigmoid_impl.h` 确认（Muls→Exp→Adds→Div 全程 dst==src 带 barrier） |
| `T.tile.cast(dst, src, mode, count)` | `swi_glu_v2.py:84/101`（CAST_NONE / CAST_RINT） | ✅ 示例已验证 + 本算子生成代码验证（bf16 出向 CAST_RINT） |
| 原地 unary（`exp(temp_ub, temp_ub)` 同构） | `gelu_mul.py:49`、`tanh.py:36-37` | ✅ 示例已验证 |
| `T.ceildiv` + symbolic `M/N` | `swi_glu_v2.py:33-34`、`act_quant.py:34,40,50`（symbolic 进 grid） | ✅ 示例已验证 + 本算子编译验证（grid 表达式 `((M+63)/64) * ((N+126)/128)`） |
| `T.tile.tanh` | **不存在**（`ascend_tile.py` 无定义，`src/` 无 lowering） | ❌ 需组合实现（本设计 §3.1 步骤 1-3，`examples/activation/tanh.py` 为 exp 组合先例，本设计改用更稳定的 sigmoid 组合） |
| `threads=2` + `T.tile.sigmoid` | **不兼容**：`AscendVidReduction` 不改写 sigmoid count（保持整 tile 4096 > 子块 buffer 2048），触发 AscendC 断言 `calCount <= srcTensor.GetSize()` | ❌ 已弃用 threads=2，改 `(cid, vid)` + `VEC_NUM=2`（见 §2.2 实现期修正） |

---

## 3.5 技术约束确认

### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维 grid `m_num * n_num`，kernel 内 `bx = cid // n_num` 分解 |
| threads 参数限制（仅 1 或 2） | No（未使用 threads） | `(cid, vid)` + `VEC_NUM=2` 手动子块切分（tanh.py 模式） |
| 动态循环边界不支持 | No | 无 tensor 值依赖的循环边界；grid 为 symbolic 表达式（`act_quant.py` 先例） |
| 流水线不支持动态边界 | No | v1 无 T.Pipelined（双缓冲列入 Stage 3 性能优化） |
| T.copy 列方向 strided 切片 | No | 所有切片均为二维行主序（行 gap 合法，列维为最内维） |
| L0C 容量上限 | No | 纯 Vector，无 L0 |

### 3.5.2 参考实现差异说明

**参考实现**：CANN ops-nn `activation/situ_glu`（AscendC 实现，https://gitcode.com/cann/ops-nn/tree/master/activation/situ_glu ）

| 差异项 | 参考实现（AscendC） | 本项目（TileLang） | 转换方案 |
|--------|---------------------|--------------------|----------|
| tanh 实现 | `AscendC::Tanh` 高阶 API | 无 `T.tile.tanh` 原语 | `tanh(t) = 2σ(2t) − 1` 组合（数值稳定，与参考实现的 `Tanh` 数学等价；对照 golden 验证） |
| 数据搬运 | `DataCopyPad` + Short-H/Long-H 双路径 tiling | `T.copy` 动态切片（ lowering 生成 DataCopyPad） | 单路径：valid-extent 切片自动处理对齐与尾块；Short/Long-H 分路列为 Stage 3 性能优化选项 |
| 持久化多核循环 | persistent `blockIdx < usedCoreNum_` + 核内 serial 循环 | 每 block 一调度（grid = m_num×n_num, threads=2） | Developer 规范写法；persistent + 双缓冲列入 Stage 3 |
| 双缓冲 | xQueue DB=2 | v1 单缓冲 | Stage 3 性能优化项 |
| 中间同步 | 显式 `PipeBarrier<PIPE_V>()` | `TL_ASCEND_AUTO_SYNC` 自动插入 | pass 自动化 |
| 混合精度 | fp16/bf16 cast→fp32 计算→cast 回（bf16 出向 CAST_RINT） | 同款策略（对齐参考实现 `situ_glu.hpp:324-328` 的平台差异注释） | 完全一致 |

### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/developer_mode/gelu_mul_developer.py` | 高度相似 | GLU 族 split+激活+乘 的 Developer 模式完整骨架（threads=2 + alloc_shared + 同款 pass_configs） |
| `examples/activation/swi_glu_v2.py` | 高度相似 | symbolic M/N、`N//2` 切片偏移、cast 路径（CAST_NONE/CAST_RINT）、ceildiv 网格 |
| `examples/activation/tanh.py` | 中度相似 | tanh 组合实现先例 + 非整除 shape（300×300, block 64×64）的 ceildiv 尾块处理 |
| `examples/deepseek_v4/act_quant.py` | 中度相似 | symbolic M 进 `T.Kernel` grid 表达式、bf16→fp32 cast 计算模式 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `A`（合轴后） | `(M, N)` | float16 / bfloat16 / float32 | M = prod(shape[:dim])，N = prod(shape[dim:])，N 必须为偶数；原始输入为 1-8 维 ND 连续张量 |

（代表性 dtype 以 float32 列出；dtype 全集见 §9.3 精度表。）

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `B`（合轴后） | `(M, N // 2)` | 与 A 一致 | kernel 输出后由 host 侧 view 还原为原始维度（dim 减半） |

### 4.3 中间缓冲区

**fp32 路径（5 buffer，每子块）**：

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `gate_ub` | `(sub_M, block_N)` | float32 | UB | gate tile（sub_M = block_M // VEC_NUM） |
| `up_ub` | `(sub_M, block_N)` | float32 | UB | up tile |
| `t_ub` | `(sub_M, block_N)` | float32 | UB | scratch：t → s1 → 2s1−1 → situ_a |
| `s_ub` | `(sub_M, block_N)` | float32 | UB | scratch：s2（λ>0 时复用为 u'） |
| `y_ub` | `(sub_M, block_N)` | float32 | UB | 输出 tile |

**fp16/bf16 路径（追加 3 buffer，fp32 5 buffer 复用；均无条件分配以共享 if 分支作用域）**：

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `gate_raw` / `up_raw` | `(sub_M, block_N)` | 原 dtype | UB | GM 搬入原 dtype tile |
| `y_raw` | `(sub_M, block_N)` | 原 dtype | UB | cast 后输出 tile |

### 4.4 内存搬运路径

```
fp32:   GM[A] --T.copy--> UB[gate_ub/up_ub] --tile 计算--> UB[y_ub] --T.copy--> GM[B]
fp16/bf16: GM[A] --T.copy--> UB[gate_raw/up_raw] --cast--> UB[gate_ub/up_ub](fp32)
           --tile 计算--> UB[y_ub](fp32) --cast--> UB[y_raw] --T.copy--> GM[B]
```

无 L1/L0 参与判定（纯 Vector）。

### 4.5 UB 内存预算

默认 `block_M = 64, block_N = 64`，`sub_M = block_M // VEC_NUM = 32`（每子块/每核独立 UB，2048 元素/tile）：

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| gate_ub / up_ub / t_ub / s_ub / y_ub | (32, 64) | float32 | 5 × 8192 = 40960 |
| gate_raw / up_raw / y_raw | (32, 64) | fp16/bf16 | 3 × 4096 = 12288 |
| sigmoid 隐式 tmp（框架分配） | — | uint8 | 8192 |
| **fp32 总计** | | | **49 KB** / 196608 (192KB, A2/A3) ✓ |
| **fp16/bf16 总计** | | | **61 KB** / 196608 ✓ |

（实测生成代码：fp32 路径经 `MEMORY_PLANNING` 复用后仅分配 5 buffer + tmp；cast 路径 y_raw 被复用为 gate_raw。fp32 若取 `block_N = 128` 翻倍仍在预算内；fp16/bf16 取 128 也仅 122KB，但默认统一 64×64。）

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| `M`（pre） | `T.symbolic("M")` | 1 ~ 2^31（含 0：M=0 时 grid=0 空跑） |
| `N`（cut） | `T.symbolic("N")` | ≥2 且为偶数（host 侧校验） |

一次编译按 `(dtype, block_M, block_N, beta, linear_beta, activate_left)` 缓存，任意 M/N 复用。

### 4.7 JIT 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def situ_glu(block_M, block_N, beta, linear_beta, activate_left, dtype="float"):
    ...
```

（`tanh.py:7-11` 同款三 pass；纯 Vector 无 CV 交互，不启用 `AUTO_CV_SYNC`。）

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 全部计算为 UB 内逐元素原语（sigmoid/mul/sub/cast），无 matmul（不涉 L1/L0），无跨行归约。

### 5.2 Block 划分

```python
block_M = 64   # 行分块：UB 预算内（§4.5），与 swi_glu_v2 的 min(block,64) 约定一致
block_N = 64   # 列分块：作用于输出空间 N//2；64 元素 × 2B = 128B 满足 32B 对齐
m_num = T.ceildiv(M, block_M)
n_num = T.ceildiv(N // 2, block_N)
block_num = m_num * n_num   # 一维 grid（symbolic 表达式）
```

每个 block 处理输出的一个 `(block_M, block_N)` tile，对应读入 gate（列偏移 `gate_off + n0`）与 up（列偏移 `up_off + n0`）两个同形 tile。

### 5.3 约束分析

- **对齐约束**: block_N=64；fp32 行 256B、fp16/bf16 行 128B，均满足 32B 对齐 ✓。`N//2` 非 16（bf16）/8（fp32）元素倍数时，up 半区起始地址非 32B 对齐——`T.copy` lowering 走 DataCopyPad 自动处理，**正确性不受影响，性能略降**（与官方实现的 halfRowAligned→Long-H 回退同一取舍，`situ_glu_tiling.cpp:238-241`）
- **UB 容量**: fp32 80KB / 混合精度 104KB < 192KB ✓（§4.5）
- **L0 容量**: 不适用（无 Cube 计算）

### 5.4 注意事项

- 非整除 M、N//2：grid 用 `T.ceildiv`，**搬运采用全 tile + 标量起点，由前端切片分析自动生成 clamp 后的 valid extent 与读入零填充**（实现期验证：生成代码中行/列 extent 均为 `min(block, remaining)` 三元表达式、up 半区按 `N − N/2` clamp、pad 值 0.0）。显式 `T.min` 动态 region 语法会触发 `AscendVidReduction` 的 `Extent is not IntImm` 崩溃，不可用
- host 侧不做 padding + crop（约束 §5 禁止）
- M=0（空 Tensor）：grid 表达式为 0，kernel 不启动，host 侧直接返回空输出（metadata 判定）

> **⚠️ 非整除已显式设计**：valid extent 由前端 `T.copy` 切片分析自动裁剪（生成代码已验证）；不使用标量 GM 起点配超出张量边界的完整 tile 写入，host 侧无 padding + crop。

### 5.5 数据搬运性能可行性

本算子为逐元素计算（无物理布局变化），非数据重排算子；带宽分析如下（bf16、(4096, 18432)、block 64×64）：

| 路径 | 代表性最大 case | GM pass | DMA 数/平均字节 | GM 标量访问 | 地址 div/mod | AIV 并行度 | 结论 |
|------|-----------------|---------|------------------|-------------|--------------|------------|------|
| 通用单路径 | (4096, 18432) bf16 | 读 1 次 + 写 0.5 次/元素对 | 每 block 3 次（2 入 1 出）/ ~8-16KB | 0 | 每 block 1 次 //、1 次 % | 9216 blocks × 2 threads | 可行 |

无逐元素 GM 标量访问、无短 DMA 风暴；`N//2` 不对齐时 DMA 效率下降已记录（§5.3）。

**[DISPATCH-COVERAGE]**

```text
supported_domain:
  shape: 任意 1-8 维 ND 连续张量（host 合轴为 (M, N)）；M ≥ 0；N ≥ 2 且为偶数
  dtype: float32 / float16 / bfloat16
  attr: dim ∈ [-ndim, ndim-1]；beta ≠ 0；linear_beta 任意 float；activate_left 任意 bool
generic_fallback: 统一 2D 合轴 kernel（全部输入经 metadata view 归一到 (M, N) 后走同一 kernel 结构）
specializations:
  - predicate: dtype ∈ {float16, bfloat16}（编译期 dtype 参数）
    fallback_on_miss: fp32 原生路径（同一 kernel，无 cast）
    equivalence_evidence: cast 到 fp32 计算后按参考实现同款 round mode 回转（bf16 CAST_RINT / fp16 CAST_NONE，situ_glu.hpp:314-328）
  - predicate: linear_beta > 0（host 侧纯 attr 比较，编译期分支）
    fallback_on_miss: linear_beta ≤ 0 的 up 直通路径
    equivalence_evidence: aclnnSituGlu 文档定义 linear_beta ≤ 0 时 up 透传
  - predicate: beta == 1.0（编译期标量折叠）
    fallback_on_miss: 一般 beta 路径（多一次标量 mul）
    equivalence_evidence: 2/β = 2、β = 1 时乘法可折叠，数学恒等
  - predicate: activate_left == False（编译期偏移交换）
    fallback_on_miss: activate_left == True 路径
    equivalence_evidence: gate_off/up_off 交换，索引映射对称
unsupported_inputs: 非 contiguous x（v1 显式报错）、beta == 0、N 为奇数、dim 越界、ndim > 8
result: pass（generic kernel 覆盖全部 supported_domain；所有 specialization 均有命中用例见 §9.2/§9.4）
```

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| block 级（M × N//2 空间） | 调度并行 | `T.Kernel(m_num * n_num)` | 每 block 一个输出 tile |
| 子块级（行二分） | 调度并行 | `(cid, vid)`，`vid ∈ {0,1}` | 每子块处理 `sub_M = block_M // 2` 行（tanh.py 模式） |
| 核内 | 无循环 | — | 单 tile 单趟，无迭代累积 |

### 6.2 循环伪代码

```python
with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
    bx, by = cid // n_num, cid % n_num        # 二维分解为一维 grid
    row0 = bx * block_M + vid * sub_M         # 子块行偏移
    # 单趟：搬入 -> 计算 -> 搬出，无核内循环
```

### 6.3 流水线优化（Stage 3 已实施）

**采用方案**：每 grid block 内 flag 流水（`elementwise_add_pipeline.py` 模式）——
block 持有 `(block_M, block_N)` 输出 tile，按 `vec_proc = block_M/sub_M` 个行块
（sub_M 行 × 2 vid 子块）流水处理：预取 mm+1 的 MTE2 与 mm 的 V 计算、mm-1 的
MTE3 回写重叠。关键实现事实：

1. `swi_glu_v2.py` 式的 `stages` 维数组 **不产生真实流水**（生成代码含循环边界
   `PipeBarrier<PIPE_ALL>`，实测确认迭代间完全串行）——必须显式 flag 协议
2. `AUTO_SYNC` 开启会在循环顶插 `PIPE_ALL` 并注入干扰标志（V_MTE3(1)），与手动
   flag 流水**不兼容**——pass_configs 须全关（V 链无屏障的正确性经真机 3×33 用例
   重复验证）
3. `T.Pipelined` 需静态循环边界（本仓已知约束），symbolic 路线不可用——这正是
   flag 手动流水的价值
4. tile 配置实测标定（A3/40 AIV，bf16 (4096,18432)）：`(512,256,sub_M=32)` 最优
   643 GB/s；wrapper 按 shape 自适应分派（`_pick_blocks`：M≥4096→(512,256)，
   M≥1024→(256,256)，否则 (128,128)）
5. `M/N` 参数化双入口：None=symbolic（一次编译任意 shape），int=静态特化
   （dispatch 0.57→0.18ms）

### 6.4 尾块处理

- grid 用 `T.ceildiv` 覆盖尾块；搬运用全 tile + 标量起点，**前端切片分析自动将 copy extent clamp 到张量边界并对读入零填充**（生成代码已验证：`min(block, remaining)` 三元表达式）
- 输出侧 copy 同样被 clamp，不会写出 B 的边界外，无跨 block 写入竞态（每 block 输出区域互不重叠）

> **⚠️ 尾块已显式设计**：valid extent 由前端 `T.copy` 切片分析自动裁剪；无 host 侧 padding + crop。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式规范）

### 7.2 同步点说明

由 `TL_ASCEND_AUTO_SYNC` pass 自动插入（搬入后 MTE2→V、计算链间 V→V、搬出前 V→MTE3），无需手动 `T.barrier_all` / `set_flag` / `wait_flag`。参考实现中每个向量指令后的 `PipeBarrier<PIPE_V>()`（`situ_glu.hpp:267-289`）由该 pass 等价覆盖。

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,        # 自动同步
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # 子块合并
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,  # UB buffer 复用规划
}
```

（实现期验证：`MEMORY_PLANNING` 使 y_ub/y_raw 与 gate buffer 复用，fp32 实际 UB 占用从 5 buffer 降至 4 + tmp。）

---

## 8. 融合算子设计

### 8.1 融合算子判定

**判定结果**: 否

**判定依据**: 无 GEMM/Cube 计算，纯 Vector 逐元素融合（split + 激活 + 乘），不涉及 Cube↔Vector 交互与 workspace 中转。

Developer 模式规范记录：`T.Kernel(block_num, is_npu=True) as (cid, vid)`（`(cid, vid)` 子块切分变体，见 §2.2 实现期修正），装饰器无 `workspace_idx`，签名无 `workspace_*` 参数，无 `alloc_L1/alloc_ub/alloc_fragment` 片上直连需求（§8.2 Developer 分支，不产出 workspace 规格）。

---

## 9. 验证方案

### 9.1 Golden 函数

```python
def golden_situ_glu(x, dim=-1, beta=1.0, linear_beta=0.0, activate_left=True):
    """CPU PyTorch 参考实现（数据准备与 golden 均在 CPU 完成）"""
    assert x.shape[dim] % 2 == 0, "x.shape[dim] must be even"
    h = x.shape[dim] // 2
    gate, up = torch.split(x, h, dim=dim)
    if not activate_left:
        gate, up = up, gate
    gate = gate.float()
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up = up.float()
    if linear_beta > 0:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (situ_a * up).to(x.dtype)
```

### 9.2 L0 门槛测试计划

> 设计阶段只给出 L0 门槛用例（规则 shape，block 整除）；L1（功能，含不规则/尾块 shape）/ L2（异常输入）/ Boundary（特殊值）完整分层套件由 `tilelang-op-test-design` 生成。

| 用例名 | 级别 | Shape（原始输入） | dtype | attrs | 说明 |
|--------|------|-------------------|-------|-------|------|
| l0_basic_fp32 | L0 | (256, 256) | float32 | 默认 | 最小功能验证，block 整除 |
| l0_basic_fp16 | L0 | (256, 256) | float16 | 默认 | cast 路径（出向 CAST_NONE） |
| l0_basic_bf16 | L0 | (256, 256) | bfloat16 | 默认 | cast 路径（出向 CAST_RINT） |
| l0_typical_bf16 | L0 | (1024, 2048) | bfloat16 | 默认 | LLM 典型 FFN 形状 |
| l0_activate_right | L0 | (256, 256) | float32 | activate_left=False | gate/up 交换分支 |
| l0_beta | L0 | (256, 256) | float32 | beta=0.7 | 一般 β 路径（多一次标量乘） |
| l0_linear_beta | L0 | (256, 256) | float32 | linear_beta=1.5 | up 有界化分支 |
| l0_dim_mid_3d | L0 | (2, 128, 512) | float32 | dim=1 | 非 last 维合轴（merged (2, 65536)） |

### 9.3 精度标准

> 混合容差：逐元素 `|actual-golden| ≤ atol + rtol·|golden|`，整体判定 `matched_ratio ≥ required_matched_ratio` 且 `max_abs_error ≤ max_abs_error_limit`；阈值仅按 dtype。

| dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|-------|------|------|---------------------|------------------------|
| float16 | 2⁻¹⁴ (6.10e-5) | 2⁻⁹ (1.95e-3) | 1e-1 | 0.99 |
| bfloat16 | 2⁻¹⁰ (9.77e-4) | 2⁻⁶ (1.56e-2) | 1e0 | 0.99 |
| float32 | 2⁻¹⁶ (1.53e-5) | 2⁻¹⁰ (9.77e-4) | 1e-2 | 0.99 |

特殊浮点值：有限输入下 `tanh`/`sigmoid` 均有界饱和，无 NaN 产生；Inf 输入经恒等式 `2σ(2x)−1` 正确饱和到 ±1（优于 exp 组合的 `inf/inf=NaN` 问题）。NaN 传播性行为与 golden 一致。Boundary 用例（含稀疏 NaN/Inf 混合输入 + mask 严格比较）交由 `tilelang-op-test-design` 的完整套件覆盖。

### 9.3.1 性能实测（A3 真机，bench_situ_glu.py，2026-10-12）

基线说明：官方 aclnnSituGlu 属 CANN 9.2 分支，本环境 CANN 9.0.0 未收录（libopapi
无该符号），以 `tanh+sigmoid+mul` torch 未融合组合为基线（SituGlu 融合所替代的形态）。

| shape | dtype | TileLang(ms) | 未融合(ms) | 加速比 | TileLang 带宽 |
|---|---|---|---|---|---|
| (1024, 2048) | bf16 | 0.6023 | 0.0608 | 0.10x | 11 GB/s* |
| (4096, 18432) | bf16 | 0.9161 | 1.3950 | **1.52x** | 247 GB/s |
| (8192, 18432) | bf16 | 1.8666 | 2.7799 | **1.49x** | 243 GB/s |
| (4096, 18432) | fp16 | 0.9104 | 1.3738 | **1.51x** | 249 GB/s |
| (1024, 2048) | fp32 | 0.6180 | 0.0622 | 0.11x | 21 GB/s* |

*小 shape 受 **JIT 每调用 dispatch 开销**支配（与 kernel 无关）：symbolic shape 下
`cython_wrapper.forward` 每次调用对输出 shape 做 TVM `undefined_vars/substitute/
simplify`（FFI）+ 新建 Analyzer，实测固定 ~0.57ms（(1,2) 输入亦 0.568ms）；改静态
shape 编译后降至 ~0.18ms（仍有 torch.empty 输出分配 + ctypes marshaling 基础开销）。
kernel 本身 (1024,2048) 约 26us。

**与官方 aclnnSituGlu kernel 对比（Stage 3 后最终数据）**（本环境 CANN 9.0.0 无该
算子，从 ops-nn 源码独立编译官方 AscendC kernel 作基线——复用 op_kernel/situ_glu.hpp
原码 + learning-hub 直调 harness + 复刻 op_host tiling，A3 真机混合容差全 PASS；
两侧同源 device event 计时）：

| shape | dtype | TileLang v2(us) | 官方 kernel(us) | **v2/官方** | v2 带宽 | 未融合 torch(us) |
|---|---|---|---|---|---|---|
| (4096, 18432) | bf16 | **352.5** | 727.6 | **2.06x** | 643 GB/s | 1421 |
| (8192, 18432) | bf16 | **678.5** | 1444.0 | **2.13x** | 668 GB/s | 2746 |
| (4096, 18432) | fp16 | **348.1** | 723.1 | **2.08x** | 651 GB/s | 1402 |
| (4096, 18432) | fp32 | **520.1** | 816.6 | **1.57x** | 871 GB/s | 2158 |

Stage 1 → Stage 3 演进：v1 单趟 927.8µs/244GB/s → v2 flag 流水 352.5µs/643GB/s
（**2.6x**）。v2 已显著超越官方 kernel（官方 311 GB/s 为 DB2 TQue 双缓冲，v2 的
更深流水 + 更大 chunk DMA 效率更高）。

小 shape：TileLang 受框架 launcher 每调用开销支配（symbolic ~0.57ms、静态特化
~0.18ms，kernel 本体 (1024,2048) 约 50us 以内），官方 harness 为 C++ 直调零开销——
属框架 launcher 层差距（symbolic 调用路径对输出 shape 逐次做 TVM FFI 解析，
可按 shape 元组缓存，已整理为上游 issue 反馈）

### 9.4 性能可行性哨兵（强制执行）

| 用例名 | Shape | dtype/属性 | 覆盖路径 | 单 case timeout | 选择理由 |
|--------|-------|------------|----------|-----------------|----------|
| perf_worst_path | (4096, 18432) | bfloat16 / 默认 attrs | 通用单路径 + cast 路径最大任务量 | 300 s（含编译） | LLM FFN 最大典型形状：75.5M 元素对、9216 blocks，DMA 与 cast 负载最大；bf16 为最坏 dtype（cast 开销 + 最松容差下混叠风险） |

测试数据、随机数、golden 均在 CPU 完成；运行阶段只做 H2D → TileLang kernel → D2H，无 aclnn 依赖。

---

## 10. 风险点与注意事项

### 10.1 已知约束

1. `T.tile.tanh` 不存在——本设计用 `2σ(2t)−1` 组合（API 缺口已在 §3.4 标注；组合精度 max_err 1.79e-07，CPU 验证）
2. ~~`T.Kernel` grid 为 symbolic 表达式~~ **已验证可用**：生成 launch 代码 `<<<((M+63)/64) * ((N+126)/128), ...>>>`（act_quant 先例成立）
3. bf16 出向 cast 必须 `CAST_RINT`（参考实现平台差异警告：CAST_NONE 在非 A2 平台产出错误数据，`situ_glu.hpp:38-39`）——生成代码已验证
4. `threads=2` + `T.tile.sigmoid` 不兼容（`AscendVidReduction` 不改写 sigmoid count）——已改用 `(cid, vid)` 模式规避（§2.2）；该 pass 缺陷建议反馈上游

### 10.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB 溢出 | block_N 过大（如 512） | 编译失败 | 保持默认 64×64（§4.5 预算，混合路径 128 亦可行） |
| ~~sigmoid 原地写不支持~~ | — | — | **已验证支持**：CANN `sigmoid_impl.h` 为 Muls→Exp→Adds→Div 全原地链，生成代码正确 |
| `Extent is not IntImm` 编译崩溃 | 显式 `T.min` 动态 region 切片 + 子块 pass | 编译失败 | 使用全 tile + 标量起点，前端自动 clamp（§5.4） |
| `if` 分支内分配的 buffer 跨分支引用 | `if need_cast` 两处分支共享 raw buffer | 解析错误 `Undefined variable` | raw buffer 无条件外层分配（swi_glu_v2 模式，已实现） |
| 半区地址非 32B 对齐 | `N//2` 非 16（bf16）元素倍数 | 性能下降（非正确性） | DataCopyPad 自动处理；性能敏感场景 Stage 3 引入 Long-H 式分路 |
| β=0 | host 校验缺失 | kernel 内除零 → NaN | host 侧 metadata 校验显式报错（已实现） |
| 空 Tensor | M=0 或 N=0 | grid=0 空跑 | host 侧 metadata 判定直接返回空输出（已实现） |

### 10.3 特殊场景处理

- **非连续输入**：v1 不支持（host 侧 `is_contiguous()` 校验后显式报错）；后续可按约束 §6 方案 B 增加 stride 作为 JIT 编译期常量的路径
- **极小 shape**（如 N=2）：n_num=1、单 block，valid extent 退化仍正确
- **β 极小**：`g/β` 放大后 tanh 饱和 ±1（sigmoid 组合数值稳定，无 exp 溢出）
- **ndim > 8**：合轴后 kernel 与 ndim 无关，但为对齐 aclnn 接口约束（1-8 维），host 侧校验报错

---

## 11. 交付清单

### 11.1 目录结构

```
examples/situ_glu/
├── situ_glu.py            # 纯 kernel（@tilelang.jit，可 import，无 golden/测试/__main__）
├── test_situ_glu.py       # from situ_glu import situ_glu + golden + 分层测试 + main
├── proto.yaml             # 算子接口规格（dtype/attr），供覆盖门禁派生应覆盖维度
├── design.md              # 本设计文档
└── README.md              # 使用说明（可选）
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `design.md` | 已完成 | 设计文档 |
| `proto.yaml` | 已完成 | 算子接口规格（dtype 全集取自 §9.3 精度表、attr 取自 §1/§4，机械派生），覆盖门禁 `coverage_check.py --proto` 用 |
| `situ_glu.py` | 已实现（L0 真机通过） | 纯 kernel（@tilelang.jit） |
| `test_situ_glu.py` | 已实现（L0 真机 9/9 通过，exit 0） | golden + 分层测试 + main（`from situ_glu import situ_glu`） |

### 11.3 命名规范

- 目录名: `situ_glu`（snake_case）
- kernel 文件: `situ_glu.py`
- 测试文件: `test_situ_glu.py`（顶部 `from situ_glu import situ_glu`）

### 11.4 实现顺序

1. ✅ 设计文档（design.md）+ proto.yaml + L0 门槛测试计划（§9.2）
2. ✅ kernel 实现（`situ_glu.py`，纯 @tilelang.jit）
3. ✅ 测试文件（`test_situ_glu.py`）：import kernel + Golden 函数 + L0 用例 + main
4. ✅ L0 门槛测试通过（A3/Ascend910_9362 真机 9/9 PASS，含性能哨兵，matched_ratio 均 1.0000，exit 0）
5. ✅ 扩展分层套件（`tilelang-op-test-design` 场景 B）：L1×24（尾块/质数/退化/rank1-8/值域/attr 全覆盖，带 D-* 覆盖标签）+ L2×9（负向拒绝）+ Boundary×8（稀疏 ±inf/nan、全零、dtype 边界、空 tensor，位置敏感比较）；覆盖门禁 coverage_check.py 19/19 PASS
6. ✅ 全量套件运行（A3 真机 `--level all`：L0 9/9 + L1 24/24 PRECISION_PASS，L2 9/9 + Boundary 8/8 BOUNDARY_PASS，0 FAIL / 0 WARN，exit 0；v2 流水版 3 轮重复无 flaky）
7. ✅ Stage 3 性能调优：flag 双缓冲流水 + tile 标定（512,256）+ M/N 双入口，大 shape **2.0-2.1x 超越官方 kernel**（bf16 643-668 GB/s vs 官方 311-314）

### 11.5 算子 proto.yaml（覆盖门禁用，Stage 1 产出）

见同目录 `proto.yaml`（dtype 全集 = §9.3 精度表；attr 覆盖全部影响计算路径的属性：dim / beta / linear_beta / activate_left）。
