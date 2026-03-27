# TileLang Ascend Operators

将 TileLang Ascend 算子注册到 PyTorch，支持 **离线安装即用**。

## 特点

- ✅ **离线安装即用** - 无需编译器，安装后直接使用
- ✅ **独立运行时** - 不依赖 tilelang 源码
- ✅ **PyTorch 集成** - 支持 `torch.ops.tl_ascend_ops.xxx` 调用
- ✅ **torch_npu 注入** - 支持 `torch_npu.xxx` 调用

## 安装

### 环境要求

- Python >= 3.8
- PyTorch > 2.6.0
- torch_npu

### 方式一：从 wheel 包安装（推荐）

```bash
# 安装预编译的 wheel 包
pip install tl_ascend_ops-0.1.0-py3-none-any.whl
```

### 方式二：从源码安装

```bash
cd examples/torch_tl_ops
pip install -e .
```

## 使用方法

### 方式一：通过包调用

```python
import torch
import tl_ascend_ops

q = torch.randn(512, 128, dtype=torch.float16, device="npu")
k = torch.randn(512, 128, dtype=torch.float16, device="npu")
v = torch.randn(512, 128, dtype=torch.float16, device="npu")

output = tl_ascend_ops.flash_attention(q, k, v)
```

### 方式二：通过 torch_npu 调用

```python
import torch
import torch_npu
import tl_ascend_ops  # 需要先导入以触发注入

q = torch.randn(512, 128, dtype=torch.float16, device="npu")
k = torch.randn(512, 128, dtype=torch.float16, device="npu")
v = torch.randn(512, 128, dtype=torch.float16, device="npu")

output = torch_npu.flash_attention(q, k, v)
```

### 方式三：通过 torch.ops 调用

```python
import torch
import tl_ascend_ops

q = torch.randn(512, 128, dtype=torch.float16, device="npu")
k = torch.randn(512, 128, dtype=torch.float16, device="npu")
v = torch.randn(512, 128, dtype=torch.float16, device="npu")

output = torch.ops.tl_ascend_ops.flash_attention(q, k, v)
```

## 开发指南

### 预编译内核

在开发机上运行预编译脚本：

```bash
cd examples/torch_tl_ops
python compile/precompile.py
```

预编译产物会保存到 `src/kernels/` 目录。

### 打包发布

```bash
cd examples/torch_tl_ops

# 方式一：使用 pip wheel
pip wheel . --no-deps -w dist/

# 方式二：使用 build
pip install build
python -m build --wheel
```

打包后的 wheel 文件在 `dist/` 目录下。

### 目录结构

```
examples/torch_tl_ops/
├── src/                    # 包源码 (tl_ascend_ops 包)
│   ├── __init__.py         # 包入口，算子注入
│   ├── loader.py           # 独立内核加载器
│   ├── registry.py         # 算子注册中心
│   ├── ops/                # 算子定义
│   │   ├── base.py         # 基类
│   │   ├── flash_attention.py
│   │   └── gemm.py
│   ├── utils/              # 共享工具库
│   │   └── npu_utils.so    # NPU 工具库 (所有内核共享)
│   └── kernels/            # 预编译内核
│       └── flash_attention/
│           ├── metadata.pkl    # 内核元数据
│           └── main.so         # 启动器
├── compile/                # 编译脚本
│   ├── precompile.py       # 预编译脚本
│   └── kernels/            # 内核定义
│       ├── flash_attention.py
│       └── gemm.py
├── tests/                  # 测试文件
│   ├── test_flash_attention.py
│   └── test_gemm.py
├── setup.py                # 安装配置
└── README.md
```

## 技术实现

### 独立加载器

`loader.py` 实现了独立的内核加载器，不依赖 tilelang 源码：

```python
from tl_ascend_ops.loader import KernelRegistry

# 加载预编译内核
kernel = KernelRegistry.get_kernel("flash_attention")

# 执行内核
output = kernel(q, k, v)
```

### 内核文件

| 文件 | 说明 |
|------|------|
| `metadata.pkl` | 内核元数据（参数信息、shape、grid 等） |
| `main.so` | 启动器，包含 `launch` 函数 |
| `npu_utils.so` | 工具库，包含内核加载函数 |

### 依赖关系

| 依赖 | 编译时 | 运行时 |
|------|--------|--------|
| tilelang | ✅ | ❌ |
| Bisheng 编译器 | ✅ | ❌ |
| g++ | ✅ | ❌ |
| torch | ✅ | ✅ |
| torch_npu | ✅ | ✅ |

## 可用算子

| 算子 | Shape | 说明 |
|------|-------|------|
| `flash_attention` | 固定 (512, 128) | Flash Attention |

## License

MIT License
