# RotaryPositionEmbedding (RoPE)

基于 TileLang DSL 实现的昇腾 NPU 旋转位置编码（Rotary Position Embedding）算子，
配套 `msprof` 性能对比脚本与精度测试，对标 CANN 官方算子
`aclnnRotaryPositionEmbedding`（`ops-transformer/posembedding/rotary_position_embedding`）。

## 算子说明

RoPE 将位置信息以旋转矩阵的方式注入 Query/Key，核心公式：

```
out = x * cos + rotate(x) * sin_signed
```

其中 `rotate` 与 `sin_signed` 的符号模式由 layout 决定：

| Layout | 别名 | rotate 规则 | sin 符号 | 典型模型 |
|--------|------|------------|---------|---------|
| `half` | GPT-NeoX / LLaMA | `rotate([x1, x2]) = [-x2, x1]`（前后半交换取负） | `[-1,...,-1, +1,...,+1]` | LLaMA / Qwen |
| `interleaved` | GPT-J | `rotate([x0, x1, x2, x3]) = [-x1, x0, -x3, x2]`（相邻配对交换取负） | `[-1,+1,-1,+1,...]` | GPT-J / NeMo |

- **部分 RoPE**：仅对 `x` 最后 `rope_dim` 维做旋转，前 `hidden_size - rope_dim` 维不变（`dim_start = hidden_size - rope_dim`，`dim_start=0` 即全旋转）。
- **原地更新**：kernel 直接写回 `x` 的对应 slice，不额外分配输出 tensor。
- **NPU 内部生成 mask**：interleaved 路径的 gather 索引和 sin 符号掩码均在 UB 上用 `createvecindex` / `bitwise_xor` / 标量赋值生成，无需 host 下发。

## 支持范围

| 维度 | 支持 |
|------|------|
| 输入布局 | TND `[BS, N, D]`、BSND `[B, S, N, D]` |
| dtype | float16 / bfloat16（内部 fp32 累加） |
| layout | half / interleaved（编译期常量，走不同代码分支） |
| rope_dim | 偶数，且 `<= hidden_size`；支持全旋转与部分旋转 |
| 多核 | 最多 48 核，按 `block_M` 分块，含 tail 行兜底 |

## 文件清单

| 文件 | 说明 |
|------|------|
| `rope_half_interleaved.py` | TileLang RoPE kernel + Python wrapper + 精度测试（L0/L1/L2/boundary） |
| `perf_rope.py` | TileLang 性能驱动，按 `--repeats` 多次启动 kernel，不计时（由 msprof 外套采集） |
| `perf_rope_ascendc.cpp` | AscendC 性能驱动，通过 ACL 调 `aclnnRotaryPositionEmbedding` |
| `CMakeLists.txt` | `perf_rope_ascendc` 的 CMake 构建（自动探测 CANN 库路径） |
| `build_perf_rope.sh` | 构建脚本：`cmake -S . -B build && make` |
| `bench.sh` | 性能对比编排：两侧分别跑 msprof，解析 CSV，计算加速比 |
| `README.md` | 本文件 |
| `bench_mark.md` | 性能与精度测试结果 |

## 快速开始

### 1. 环境

```bash
source set_env.sh                 # CANN 环境变量
# 确认 tilelang 已安装：python -c "import tilelang"
```

### 2. 构建 AscendC 性能驱动

```bash
bash build_perf_rope.sh           # 产物：./build/perf_rope_ascendc
```

### 3. 精度测试

```bash
python rope_half_interleaved.py --level l0       # L0 门槛（6 用例）
python rope_half_interleaved.py --level l1       # L1 功能（12 用例）
python rope_half_interleaved.py --level all      # L0 + L1 + L2 负向 + boundary
python rope_half_interleaved.py --shape 4 64 128 128 --layout half --dtype float16  # 单用例
```

### 4. 性能对比

```bash
bash bench.sh                                    # 跑默认 shape 组（half/interleaved × fp16/bf16 + BSND）
bash bench.sh --list                             # 仅列出 shape，不执行
bash bench.sh --shape "4 64 128 128" --layout half   # 单 shape
bash bench.sh --repeats 10                       # 每个算子启动 10 次
```

## 编程模式

采用 **Developer 模式**（`alloc_shared` + 自动同步），`pass_configs`：

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,        # 自动 barrier
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,  # UB 内存复用
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,     # Vector/Cube 自动同步
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # CV cast 合并
}
```

## 内存层级与 Tiling

- 数据流：`GM → UB(x/sin/cos) → L0C(无) → UB(计算) → GM`，全程 Vector 路径，不涉及 Cube。
- `block_M` 由 `select_block_M()` 按约束自动选取：`head_num % (block_M//2) == 0` 且 UB 总量 ≤ 192KB；interleaved 路径多 7 个 mask buffer，factor 更高，故大 rope_dim 时自动降到 `block_M=32`。
- 多核：`total_chunks = ceil(M / block_M)`，`num_blocks = min(total_chunks, 48)`，每个 block 串行处理若干 chunk。

## 性能测量方法

- **工具**：`msprof` 采集 device 侧 `Task Duration(us)`（来自 `op_summary_*.csv`，逐次 launch）。
- **协议**：`--repeats 6`，丢首次 launch（冷启动 / DVFS），取其余平均。
- **加速比**：`ac_us / tl_us`（>1.0 表示 TileLang 更快）。
- **Op Type 映射**：TileLang 侧 = `main_kernel`；AscendC 侧 = `RotaryPositionEmbedding`。
- **layout → aclnn mode**：`half → 0`，`interleaved → 2`（见 `aclnn_rotary_position_embedding.h`）。

详见 `bench_mark.md`。
