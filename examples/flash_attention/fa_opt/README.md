# 使用说明

### 参数列表

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--B` | `[4]` | Batch Size 列表 (如 `--B 1 2 4`) |
| `--S` | `[4096]` | Sequence Length 列表 |
| `--H` | `[16]` | Number of Heads 列表 |
| `--D` | `[128]` | Head Dimension 列表 |
| `--iter-mode` | `zip` | 遍历模式：`zip` (按索引一一对应) 或 `product` (全排列组合) |
| `--tl` | `./flash...pipeline...py` | TileLang 脚本路径 |
| `--ascendc` | `./flash...ascendc.py` | AscendC 脚本路径 |
| `--log` | `./log` | 日志输出目录 |

### 运行示例

**1. 列表模式 (Zip, 默认)**
按索引对应运行 4 组用例：`(2, 8192, 32, 512)`, `(4, 4096, 32, 512)`, `(8, 2048, 32, 512)`, `(16, 1024, 32, 512)`。

```bash
python run.py \
    --iter-mode zip \
    --B 2 4 8 16 \
    --S 8192 4096 2048 1024 \
    --H 32 32 32 32 \
    --D 512 512 512 512 \
    --log ./log \
    --tl ./flash_attn_bhsd_auto_pipeline_h32_d512.py \
    --ascendc ./flash_attn_bhsd_ascendc.py
```

**2. 全排列模式 (Product)**
运行 B(1, 8) 与 S(2048, 4096) 的所有组合（共 2x2=4 组），保持 H=16, D=128。

```bash
python run.py \
    --iter-mode product \
    --B 1 8 \
    --S 2048 4096 \
    --H 16 \
    --D 128 \
    --log ./log \
    --tl ./flash_attn_bhsd_auto_pipeline_h16_d128.py \
    --ascendc ./flash_attn_bhsd_ascendc.py
```