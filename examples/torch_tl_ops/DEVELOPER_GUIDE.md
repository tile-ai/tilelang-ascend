# TileLang Ascend Operators 开发者教程

本文档面向开发者，介绍如何使用和扩展 `tl_ascend_ops` 包。

## 目录

1. [快速开始](#快速开始)
2. [使用方法](#使用方法)
3. [新增算子开发流程](#新增算子开发流程)
4. [开发原理](#开发原理)
5. [常见问题](#常见问题)

---

## 快速开始

### 安装

```bash
# 从 wheel 包安装（推荐）
pip install tl_ascend_ops-0.1.0-py3-none-any.whl

# 或从源码安装
cd examples/torch_tl_ops
pip install -e .
```

### 基本使用

```python
import torch
import tl_ascend_ops

# 创建输入张量
q = torch.randn(512, 128, dtype=torch.float16, device="npu")
k = torch.randn(512, 128, dtype=torch.float16, device="npu")
v = torch.randn(512, 128, dtype=torch.float16, device="npu")

# 调用算子
output = tl_ascend_ops.flash_attention(q, k, v)
```

---

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

output = torch_npu.flash_attention(q, k, v)
```

### 方式三：通过 torch.ops 调用

```python
import torch
import tl_ascend_ops

output = torch.ops.tl_ascend_ops.flash_attention(q, k, v)
```

---

## 新增算子开发流程

新增一个算子需要完成以下步骤：

### Step 1: 编写内核定义

在 `compile/kernels/` 目录下创建内核定义文件，例如 `compile/kernels/new_op.py`：

```python
"""
New Op 内核定义

描述算子的功能和 shape 特性
"""

import os
import torch
import tilelang
import tilelang.language as T


def compile_new_op_kernel():
    """编译 New Op 内核"""
    print("=" * 60)
    print("编译 New Op 内核")
    print("=" * 60)
    
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    
    # 定义内核参数（固定 shape 或动态 shape）
    # 固定 shape 示例
    M, N, K = 1024, 1024, 1024
    
    # 动态 shape 示例
    # M = T.symbolic("M")
    # N = T.symbolic("N")
    # K = T.symbolic("K")
    
    @tilelang.jit(out_idx=[2], target="npuir")  # out_idx 指定输出参数索引
    def new_op_kernel(block_M=128, block_N=128, dtype="float16"):
        
        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),  # 输出
        ):
            # 编写内核逻辑
            with T.Kernel(T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True) as (cid, _):
                # ... 内核实现 ...
                pass
        
        return main
    
    print("正在编译...")
    kernel = new_op_kernel()
    
    # 测试验证
    print("\n测试运行...")
    a = torch.randn(M, K, dtype=torch.float16, device="npu")
    b = torch.randn(K, N, dtype=torch.float16, device="npu")
    
    c = kernel(a, b)
    
    # 验证结果
    ref = a @ b  # 替换为正确的参考实现
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
    print("✓ 验证通过")
    
    return kernel
```

### Step 2: 注册内核到编译脚本

在 `compile/kernels/__init__.py` 中注册新内核：

```python
from .flash_attention import compile_flash_attention_kernel
from .gemm import compile_gemm_kernel
from .new_op import compile_new_op_kernel  # 新增

KERNEL_REGISTRY = {
    "flash_attention": compile_flash_attention_kernel,
    "gemm": compile_gemm_kernel,
    "new_op": compile_new_op_kernel,  # 新增
}
```

### Step 3: 编写算子定义

在 `src/ops/` 目录下创建算子定义文件，例如 `src/ops/new_op.py`：

```python
"""
New Op 算子定义
"""

from typing import Optional
import torch
from .base import BaseOp


class NewOpOp(BaseOp):
    """New Op 算子"""
    
    _kernel = None
    
    @property
    def name(self) -> str:
        return "new_op"
    
    @property
    def signature(self) -> str:
        # 定义 PyTorch 算子签名
        return "new_op(Tensor A, Tensor B) -> Tensor"
    
    def get_kernel(self, registry):
        if NewOpOp._kernel is None:
            NewOpOp._kernel = registry.get_kernel(self.name)
        return NewOpOp._kernel
    
    def impl(self, A: torch.Tensor, B: torch.Tensor, registry=None) -> torch.Tensor:
        """算子实现"""
        kernel = self.get_kernel(registry)
        output = kernel(A, B)
        return output
    
    def python_api(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        New Op 算子
        
        Args:
            A: 输入张量 [M, K], NPU tensor, float16
            B: 输入张量 [K, N], NPU tensor, float16
        
        Returns:
            输出张量 [M, N]
        """
        if A.device.type != "npu":
            raise ValueError("A must be an NPU tensor")
        if B.device.type != "npu":
            raise ValueError("B must be an NPU tensor")
        
        return torch.ops.tl_ascend_ops.new_op(A, B)


new_op_op = NewOpOp()
```

### Step 4: 注册算子

在 `src/registry.py` 中注册新算子：

```python
def register_all_ops() -> Dict[str, BaseOp]:
    """注册所有算子"""
    from .ops.flash_attention import flash_attention_op
    from .ops.gemm import gemm_op
    from .ops.new_op import new_op_op  # 新增
    
    ops = {
        "flash_attention": flash_attention_op,
        "gemm": gemm_op,
        "new_op": new_op_op,  # 新增
    }
    
    for name, op in ops.items():
        _register_op(op)
    
    print(f"✓ 已注册 {len(ops)} 个算子: {', '.join(ops.keys())}")
    return ops
```

### Step 5: 导出 Python API

在 `src/__init__.py` 中导出新算子：

```python
# 导入算子 Python API
from .ops.flash_attention import flash_attention_op
from .ops.gemm import gemm_op
from .ops.new_op import new_op_op  # 新增

flash_attention = flash_attention_op.python_api
gemm = gemm_op.python_api
new_op = new_op_op.python_api  # 新增

__all__ = [
    "flash_attention",
    "gemm",
    "new_op",  # 新增
    ...
]
```

### Step 6: 编写测试

在 `tests/` 目录下创建测试文件，例如 `tests/test_new_op.py`：

```python
"""
New Op 算子测试
"""

import torch
import torch_npu
import tl_ascend_ops


def test_new_op():
    """测试 New Op 算子"""
    print("=" * 60)
    print("测试 New Op 算子")
    print("=" * 60)
    
    M, N, K = 1024, 1024, 1024
    
    a = torch.randn(M, K, dtype=torch.float16, device="npu:0")
    b = torch.randn(K, N, dtype=torch.float16, device="npu:0")
    ref = a @ b
    
    # 方式 1: 通过包调用
    c = tl_ascend_ops.new_op(a, b)
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
    print("  ✓ tl_ascend_ops.new_op 验证通过")
    
    # 方式 2: 通过 torch_npu 调用
    c2 = torch_npu.new_op(a, b)
    torch.testing.assert_close(c2, ref, rtol=1e-2, atol=1e-2)
    print("  ✓ torch_npu.new_op 验证通过")
    
    # 方式 3: 通过 torch.ops 调用
    c3 = torch.ops.tl_ascend_ops.new_op(a, b)
    torch.testing.assert_close(c3, ref, rtol=1e-2, atol=1e-2)
    print("  ✓ torch.ops.tl_ascend_ops.new_op 验证通过")


if __name__ == "__main__":
    test_new_op()
```

### Step 7: 预编译和测试

```bash
# 预编译内核
cd examples/torch_tl_ops
python compile/precompile.py

# 安装
pip install -e .

# 测试
python tests/test_new_op.py
```

---

## 开发原理

### 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                      用户代码                                 │
│  tl_ascend_ops.flash_attention(q, k, v)                     │
│  torch_npu.flash_attention(q, k, v)                         │
│  torch.ops.tl_ascend_ops.flash_attention(q, k, v, scale)    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PyTorch 算子层                            │
│  registry.py: 注册算子到 torch.library                       │
│  ops/*.py: 算子定义和 Python API                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    内核加载层                                │
│  loader.py: NPUKernelLoader                                 │
│  - 加载 metadata.pkl (内核元数据)                            │
│  - 加载 main.so (启动器)                                     │
│  - 加载 npu_utils.so (工具库)                                │
│  - 计算动态 shape 和 grid                                    │
│  - 执行内核                                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    预编译内核                                │
│  kernels/flash_attention/                                   │
│  ├── metadata.pkl    # 内核元数据                            │
│  ├── main.so         # 启动器                               │
│  └── npu_utils.so    # 工具库                               │
└─────────────────────────────────────────────────────────────┘
```

### 为什么需要预编译？

#### 1. 消除编译时依赖

TileLang 编译内核需要：
- tilelang 源码
- Bisheng 编译器（华为 Ascend 编译器）
- g++ 编译器
- TVM 框架

预编译后，运行时只需要：
- PyTorch
- torch_npu

#### 2. 提高部署效率

| 阶段 | 编译时 | 运行时 |
|------|--------|--------|
| 编译内核 | 分钟级 | - |
| 加载内核 | - | 毫秒级 |

#### 3. 保护知识产权

预编译后的 `.so` 文件是二进制格式，无法直接查看源码。

### 预编译产物说明

#### metadata.pkl

包含内核运行所需的所有元数据：

```python
{
    "symbolic": {"M": (0, 0), "N": (0, 1), "K": (0, 0)},  # 动态 shape 变量
    "out_idx": [2],              # 输出参数索引
    "param_info": [              # 参数信息
        {"dtype": torch.float16, "shape": ["M", "K"], "is_output": False},
        {"dtype": torch.float16, "shape": ["K", "N"], "is_output": False},
        {"dtype": torch.float16, "shape": ["M", "N"], "is_output": True},
    ],
    "signature": {0: "input", 1: "input", 2: "output"},  # 参数签名
    "gridfunc": "ceildiv(M, 128) * ceildiv(N, 128)",     # Grid 计算表达式
    "kernel_src": b"...",         # 内核二进制代码
    "kernel_name": "matmul",      # 内核名称
    "tensor_kinds": [...],        # 张量类型
    "shared": 0,                  # 共享内存大小
    "mix_mode": "",               # 混合模式
}
```

#### main.so

启动器，包含 `launch` 函数，负责：
- 接收参数
- 调用内核执行

#### npu_utils.so

工具库，包含：
- `load_kernel_binary`: 加载内核二进制到设备
- 其他 NPU 运行时工具函数

### 内核加载流程

```python
# 1. 加载 metadata.pkl
with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# 2. 加载 main.so
spec = importlib.util.spec_from_file_location("main", "main.so")
main_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_module)
launch = main_module.launch

# 3. 加载 npu_utils.so
spec = importlib.util.spec_from_file_location("npu_utils", "npu_utils.so")
npu_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(npu_utils)

# 4. 加载内核二进制到设备
npu_utils.load_kernel_binary(
    kernel_name,
    kernel_src,
    shared,
    device,
    kernel_mode,
)

# 5. 执行内核
launch(grid_x, grid_y, grid_z, stream, t_function, ...)
```

### 动态 Shape 处理

#### 原理

TileLang 支持 `T.symbolic()` 定义动态 shape：

```python
M = T.symbolic("M")
N = T.symbolic("N")
K = T.symbolic("K")

@T.prim_func
def matmul(
    A: T.Tensor((M, K), dtype),
    B: T.Tensor((K, N), dtype),
    C: T.Tensor((M, N), dtype),
):
    ...
```

#### 运行时计算

`loader.py` 在执行时根据输入张量的实际 shape 计算动态值：

```python
def _calc_grid(self, orig_to_input, *args):
    """计算 grid 维度和动态值"""
    dynamic_val = {}
    
    for name, (arg_idx, dim_idx) in self.symbolic.items():
        # 从输入张量获取实际值
        arg = args[arg_idx]
        value = arg.shape[dim_idx]
        dynamic_val[name] = value
    
    # 计算 grid
    grid_value = eval(self.gridfunc, {"math": math, **dynamic_val})
    
    return dynamic_val
```

### PyTorch 算子注册原理

使用 `torch.library.Library` API 注册算子：

```python
# 定义算子签名
lib_def = torch.library.Library("tl_ascend_ops", "DEF")
lib_def.define("flash_attention(Tensor Q, Tensor K, Tensor V) -> Tensor")

# 注册实现
lib_impl = torch.library.Library("tl_ascend_ops", "IMPL")
lib_impl.impl("flash_attention", _flash_attention_impl, "PrivateUse1")
```

`PrivateUse1` 是 PyTorch 的扩展 dispatch key，用于自定义后端实现。

---

## 常见问题

### Q1: 如何调试内核加载问题？

添加调试打印：

```python
# loader.py
print(f"Loading kernel from: {self.kernel_dir}")
print(f"metadata: {self.metadata}")
print(f"grid: {self.grid}")
print(f"dynamic_val: {dynamic_val}")
```

### Q2: 如何处理 TVM 依赖问题？

确保 `precompile.py` 中的 `_to_pure_python` 函数正确转换所有 TVM 类型：

```python
def _to_pure_python(obj):
    # TVM IntImm -> int
    if hasattr(obj, 'value'):
        return int(obj.value)
    
    # TVM tir.Var -> str
    if hasattr(obj, 'name'):
        return str(obj.name)
    
    # 保留 torch.dtype 等不涉及 TVM 的类型
    if isinstance(obj, torch.dtype):
        return obj
    
    ...
```

### Q3: 如何支持新的数据类型？

在内核定义中使用 `dtype` 参数：

```python
@tilelang.jit(out_idx=[2], target="npuir")
def my_kernel(dtype="float16"):
    ...
```

### Q4: 如何优化内核性能？

1. 调整 block 大小
2. 使用流水线（`T.Pipelined`）
3. 使用共享内存（`T.alloc_shared`）
4. 参考 TileLang 性能优化文档

---

## 参考资料

- [TileLang 文档](https://github.com/tilelang/tilelang)
- [PyTorch 自定义算子](https://pytorch.org/tutorials/advanced/custom_ops.html)
- [华为 Ascend 开发文档](https://www.hiascend.com/)

---

*文档版本: 0.1.0*
*更新时间: 2026-03-16*
