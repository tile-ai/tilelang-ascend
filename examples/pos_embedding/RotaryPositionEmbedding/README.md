# RotaryPositionEmbedding (RoPE)

基于 TileLang DSL 实现的昇腾 NPU 旋转位置编码（Rotary Position Embedding）算子，
配套 `msprof` 性能对比脚本与精度测试，对标 CANN 官方算子
`aclnnRotaryPositionEmbedding`（通过 `torch_npu.npu_rotary_mul` 调用）。

支持三种编译后端：**AscendC JIT**（默认）、**AOT 预编译**、**PTO 后端**。

## 算子说明

RoPE 将位置信息以旋转矩阵的方式注入 Query/Key，核心公式：

```
out = x * cos + rotate(x) * sin_signed
```

其中 `rotate` 与 `sin_signed` 的符号模式由 layout 决定：


| Layout        | 别名             | rotate 规则                                                         | sin 符号                 | 典型模型     |
| ------------- | ---------------- | ------------------------------------------------------------------- | ------------------------ | ------------ |
| `half`        | GPT-NeoX / LLaMA | `rotate([x1, x2]) = [-x2, x1]`（前后半交换取负）                    | `[-1,...,-1, +1,...,+1]` | LLaMA / Qwen |
| `interleaved` | GPT-J            | `rotate([x0, x1, x2, x3]) = [-x1, x0, -x3, x2]`（相邻配对交换取负） | `[-1,+1,-1,+1,...]`      | GPT-J / NeMo |

- **部分 RoPE**：仅对 `x` 最后 `rope_dim` 维做旋转，前 `hidden_size - rope_dim` 维不变（`dim_start = hidden_size - rope_dim`，`dim_start=0` 即全旋转）。
- **原地更新**：kernel 直接写回 `x` 的对应 slice，不额外分配输出 tensor。
- **NPU 内部生成 mask**：interleaved 路径的 gather 索引和 sin 符号掩码均在 UB 上用 `createvecindex` / `bitwise_xor` / 标量赋值生成，无需 host 下发。

## 支持范围


| 维度     | 支持                                             |
| -------- | ------------------------------------------------ |
| 输入布局 | TND`[BS, N, D]`、BSND `[B, S, N, D]`             |
| dtype    | float16 / bfloat16（内部 fp32 累加）             |
| layout   | half / interleaved（编译期常量，走不同代码分支） |
| rope_dim | 偶数，且`<= hidden_size`；支持全旋转与部分旋转   |
| 多核     | 最多 48 核，按`block_M` 分块，含 tail 行兜底     |
| 编译后端 | AscendC（JIT）、AscendC（AOT）、PTO              |

## 文件清单


| 文件                       | 说明                                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------ |
| `rope_half_interleaved.py` | TileLang RoPE kernel + Python wrapper + 精度测试（L0/L1/L2/boundary）+ perf 模式     |
| `bench.sh`                 | 性能对比编排：msprof op 算子级采集，两侧分别跑，计算加速比                           |
| `aot_rope.py`              | AOT 编译 + 测试：lower kernel → LibraryGenerator →`rope_lib.so`，`--test` 验证精度 |
| `pto_rope.py`              | PTO 后端测试：`target="pto"` Expert 模式 + 精度验证                                  |
| `README.md`                | 本文件                                                                               |

## 快速开始

### 1. 环境

```bash
source set_env.sh                 # CANN 环境变量
# 确认依赖：python -c "import tilelang, torch_npu"
```

### 2. 精度测试

```bash
cd examples/pos_embedding/RotaryPositionEmbedding

# Golden = CANN 官方算子 (torch_npu.npu_rotary_mul)
python rope_half_interleaved.py --level l0       # L0 门槛（6 用例）
python rope_half_interleaved.py --level all      # L0 + L1 + L2 负向 + boundary

# 单用例
python rope_half_interleaved.py --shape 4 64 128 128 --layout half --dtype float16
```

### 3. 性能对比

```bash
bash bench.sh                                    # 跑默认 shape 组
bash bench.sh --list                             # 仅列出 shape
bash bench.sh --shape "4 64 128 128" --layout half   # 单 shape
bash bench.sh --warmup 10 --launch-count 50      # 自定义预热/采集次数
```

### 4. AOT 编译与测试

```bash
# 编译为 .so（脱离 Python/tilelang 运行时）
python aot_rope.py --shape 16 64 512 256 --layout half --dtype float16 -o rope_lib.so

# 编译 + 立即测试
python aot_rope.py --shape 16 64 512 256 --layout half --dtype float16 --test
```

### 5. PTO 后端测试

```bash
# PTO 后端（Expert 模式：T.Scope + alloc_ub）
python pto_rope.py --shape 16 64 512 256 --layout half --dtype float16
python pto_rope.py --shape 16 64 512 256 --layout interleaved --dtype bfloat16
```

## 编程模式

### AscendC 后端（默认，Developer 模式）

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,        # 自动 barrier
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,  # UB 内存复用
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # CV cast 合并（reinterpretcast 需要）
}
```

### PTO 后端（Expert 模式）

PTO 后端使用 CCE 编译器（`-xcce`），要求 `T.Scope("V")` 生成 `[aicore]` attribute：

```python
@tilelang.jit(target="pto")
def kernel(...):
    @T.prim_func
    def main(x, sin, cos):
        with T.Kernel(...) as (cid, vid):
            with T.Scope("V"):
                x_ub = T.alloc_ub(...)       # Expert: alloc_ub（非 alloc_shared）
                T.copy(x, x_ub)
                T.barrier_all()              # Expert: 手动同步
                T.tile.add(...)
```

## 内存层级与 Tiling

- 数据流：`GM → UB(x/sin/cos) → L0C(无) → UB(计算) → GM`，全程 Vector 路径，不涉及 Cube。
- `block_M` 由 `select_block_M()` 按约束自动选取：`head_num % (block_M//2) == 0` 且 UB 总量 ≤ 192KB。
- 多核：`total_chunks = ceil(M / block_M)`，`num_blocks = min(total_chunks, 48)`，每个 block 串行处理若干 chunk。

## 性能测量方法

- **工具**：`msprof op` 算子级采集 `Task Duration(us)`（`--warm-up` 预热 + `--launch-count` 采集次数）。
- **两侧入口**：都通过 `python rope_half_interleaved.py --perf --side {tl|cann}` 启动单次 kernel launch。
- **加速比**：`cann_us / tl_us`（>1.0 表示 TileLang 更快）。
- **Op Type 映射**：TileLang 侧 = `kernel_kernel`；CANN 侧 = `RotaryPositionEmbedding`。

详见 `bench_mark.md`。
