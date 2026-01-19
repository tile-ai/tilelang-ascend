# 使用说明

### 参数列表

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--B` | `[4]` | Batch Size 列表 (如 `--B 1 2 4`) |
| `--S` | `[4096]` | Sequence Length 列表 |
| `--H` | `[16]` | Number of Heads 列表 |
| `--D` | `[128]` | Head Dimension 列表 |
| `--iter-mode` | `zip` | 遍历模式：`zip` (按索引一一对应) 或 `product` (全排列组合) |
| `--tl` | `./flash...pipeline.py` | TileLang 脚本路径 |
| `--ascendc` | `./flash...ascendc.py` | AscendC 脚本路径 |
| `--log` | `./log` | 日志输出目录 |

### 运行示例

**1. 列表模式 (Zip, 默认)**
按索引对应运行 3 组用例：`(1, 2048, 16, 128)`, `(2, 4096, 32, 128)`, `(4, 8192, 32, 256)`。

```bash
python run.py \
    --iter-mode zip \
    --B 1 2 4 \
    --S 2048 4096 8192 \
    --H 16 32 32 \
    --D 128 128 256 \
    --log ./log \
    --tl ./flash_attn_bhsd_cc_sync_auto_pipeline.py \
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
    --tl ./flash_attn_bhsd_cc_sync_auto_pipeline.py \
    --ascendc ./flash_attn_bhsd_ascendc.py
```