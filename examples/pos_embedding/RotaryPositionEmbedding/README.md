# RotaryPositionEmbedding (RoPE)

基于 TileLang DSL 实现的昇腾 NPU 旋转位置编码（Rotary Position Embedding）算子，
配套 `msprof` 性能对比脚本与精度测试，对标 CANN 官方算子
`aclnnRotaryPositionEmbedding`（通过 `torch_npu.npu_rotary_mul` 调用）。

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
| `rope_half_interleaved.py` | TileLang RoPE kernel + Python wrapper + 精度测试（L0/L1/L2/boundary）+ perf 模式 |
| `bench.sh` | 性能对比编排：两侧分别跑 msprof，解析 CSV，计算加速比 |
| `README.md` | 本文件 |
| `bench_mark.md` | 性能与精度测试结果 |

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
- `block_M` 由 `select_block_M()` 按约束自动选取：`head_num % (block_M//2) == 0` 且 UB 总量 ≤ 192KB。
- 多核：`total_chunks = ceil(M / block_M)`，`num_blocks = min(total_chunks, 48)`，每个 block 串行处理若干 chunk。

## 性能测量方法

- **工具**：`msprof` 采集 device 侧 `Task Duration(us)`。
- **两侧入口**：都通过 `python rope_half_interleaved.py --perf --side {tl|cann}` 启动单次 kernel launch。
- **加速比**：`cann_us / tl_us`（>1.0 表示 TileLang 更快）。
- **Op Type 映射**：TileLang 侧 = `kernel_kernel`；CANN 侧 = `RotaryPositionEmbedding`。

详见 `bench_mark.md`。
