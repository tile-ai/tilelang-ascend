# TileLang-Ascend AI Core Exception Dump 技术白皮书

## 1. 概述

在昇腾 NPU 上执行自定义算子时，AI Core 硬件异常（如非法内存访问、MTE 错误、Cube/Vector 计算溢出等）会导致 kernel 执行中断。传统调试手段难以在异常发生瞬间捕获 kernel 的输入张量状态，开发者往往只能看到一条硬件错误码，无法获知异常时各输入 tensor 的实际数据。

TileLang-Ascend AI Core Exception Dump 特性通过 CANN 的异常回调机制，在 AI Core 硬件异常触发时自动捕获并保存 kernel 的输入张量数据，使开发者能够在事后恢复和分析异常现场。

### 核心能力

- **自动捕获**：AI Core 异常发生时，由 CANN 运行时异步回调，无需手动介入
- **张量级精度**：保存完整的输入张量数据（非截断的寄存器快照）
- **Kernel 级隔离**：每个 kernel 的 dump 文件以 kernel name 为前缀，互不干扰
- **零侵入开关**：通过 `TL_ASCEND_EXCEPTION_DUMP` 编译配置控制，默认关闭，不影响未启用时的编译产物
- **双后端支持**：同时覆盖 AscendC (`ascendc`) 和 PTO (`pto`) 两个 codegen 后端
- **端到端工具链**：提供 Python 函数封装 dump 文件查找、CANN msaicerr 工具调用和 tensor 数据读取

### 适用场景

| 场景 | 说明 |
|------|------|
| Kernel 调试 | 硬件异常后恢复输入张量，用于本地复现问题 |
| CI 自动化测试 | 测试用例中触发异常并验证 dump 数据与预期输入一致 |
| 生产环境排障 | 开启配置后部署，异常发生时自动落盘，无需复现 |

---

## 2. 技术原理

### 2.1 工作流程

```
 kernel 执行 ──→ AI Core 硬件异常
                      │
                      ▼
          CANN 调用已注册的异常回调
                      │
          ┌───────────┴───────────┐
          │                       │
    从异常信息中获取        回调函数在 Host 侧
    kernel args buffer       搜索 MAGIC 标记
    (device 内存)            定位 ParamSizeInfo
          │                       │
          └───────┬───────────────┘
                  ▼
     构建 acldumpTensorInfo 数组
     (tensor 地址、大小、数据类型)
                  │
                  ▼
     acldumpSaveExceptionInfo(kernel_name, ...)
                  │
                  ▼
     <ASCEND_DUMP_PATH>/extra-info/data-dump/<devId>/
       <kernel_name>.custom.<timestamp>
                  │
                  ▼ 事后解析
     msaicerr.py 解析 → per-tensor .bin 文件
                  │
                  ▼
     read_msaicerr_bin() → np.ndarray
```

### 2.2 异常回调机制

TileLang 在编译时将一个异常回调函数注册到 CANN 运行时。当 AI Core 发生硬件异常时，CANN 会自动调用该回调，并传入异常信息。回调通过 CANN API 从异常信息中提取 kernel 的参数缓冲区（device 内存），拷贝到 Host 后进行解析。

### 2.3 ParamSizeInfo 与 MAGIC 搜索

回调收到的参数缓冲区包含 kernel 的所有原始参数（tensor 指针、tiling 参数等），但回调本身不知道哪些位置是 tensor、各 tensor 的大小和数据类型。

TileLang 的解决方案是在编译时将一个 `ParamSizeInfo` 结构体附加到 kernel 参数末尾，包含：

- **MAGIC 值**：唯一的 8 字节标记 (`0x474e414c454c4954`，即 ASCII "TILELANG")
- **kernel name**：算子名称，用作 dump 文件名前缀
- **tensor 元信息**：每个 tensor 的字节数、设备地址、ACL 数据类型编码

回调在参数缓冲区中按 8 字节步长搜索 MAGIC 值，定位到 `ParamSizeInfo` 后即可读取所有 tensor 的描述信息。这种方式使回调代码与 kernel 的具体参数布局完全解耦。

### 2.4 Dump 文件生成

回调根据 `ParamSizeInfo` 构建 CANN 的 `acldumpTensorInfo` 数组，调用 `acldumpSaveExceptionInfo` 将 tensor 数据保存到文件。dump 文件路径由 CANN 环境变量控制：

```
<ASCEND_DUMP_PATH>/extra-info/data-dump/<devId>/<kernel_name>.custom.<timestamp>
```

每个 kernel 的 dump 文件以 kernel name 为前缀，支持多 kernel 场景下的文件隔离。

### 2.5 配置控制

特性通过 `TL_ASCEND_EXCEPTION_DUMP` 编译配置控制，**默认关闭**。默认关闭的原因：

1. 该功能依赖 CANN 提供的 `aclrtSetExceptionInfoCallback`、`acldumpSaveExceptionInfo` 等 API，旧版 CANN 可能不支持
2. 开启后 kernel 签名会附加额外参数，改变编译产物 ABI
3. 异常回调注册会改变 CANN 运行时行为

配置通过 TVM 标准的 `PassContext` 机制传递：Python `pass_configs` → `PassContext.config` → Codegen 读取 → 条件化生成 C++ 代码。关闭时不生成任何相关代码，编译产物与未添加该特性时完全一致。

---

## 3. 环境依赖

### 3.1 CANN 版本要求

**CANN >= 9.3.0**

需要 CANN 提供以下 API（见 `acl/acl_rt.h` 和 `acl/acl_dump.h`）：

| API | 作用 |
|-----|------|
| `aclrtSetExceptionInfoCallback` | 注册异常回调 |
| `aclrtGetArgsFromExceptionInfo` | 从异常信息获取 kernel args |
| `acldumpSaveExceptionInfo` | 保存 tensor 数据到 dump 文件 |

### 3.2 环境变量

| 环境变量 | 必需 | 说明 |
|----------|------|------|
| `ASCEND_DUMP_PATH` | 是 | dump 文件根路径 |
| `ASCEND_DUMP_SCENE` | 是 | 设为 `aic_err_brief_dump` 启用异常 dump |
| `ASCEND_HOME_PATH` | 解析时 | 定位 CANN 的 `msaicerr.py` 工具 |

> **关键**：`ASCEND_DUMP_PATH` 和 `ASCEND_DUMP_SCENE` 必须在 `import torch` / ACL 初始化**之前**设置，否则 CANN 不会读取到这些配置。

---

## 4. 使用指南

### 4.1 开启配置

在 `tilelang.jit` 装饰器或 `tilelang.compile` 中通过 `pass_configs` 传入：

```python
import tilelang

@tilelang.jit(
    out_idx=[2],
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_EXCEPTION_DUMP: True,
    },
)
def my_kernel(M, N, dtype="float16"):
    ...
```

### 4.2 设置环境变量

```python
import os

# 必须在 import torch 之前设置
os.environ["ASCEND_DUMP_PATH"] = "/tmp/exc_dump"
os.environ["ASCEND_DUMP_SCENE"] = "aic_err_brief_dump"

import torch
import tilelang
```
ASCEND_DUMP_SCENE环境变量用来使能该功能，如果不设置即使发生异常也不会进行tensor的dump功能。
ASCEND_DUMP_PATH环境变量用来设置异常dump文件的存储路径，不设置默认则为当前工作目录。
### 4.3 捕获异常并解析 dump

在 `except` 块中调用 `parse_exception_dump()`，自动完成 dump 文件查找、msaicerr 解析和 tensor 数据读取：

```python
from tilelang.tools.ascend_exception_dump_bin import parse_exception_dump

try:
    # 执行可能触发 AI Core 异常的 kernel 调用
    result = kernel(a, b)
except Exception as e:
    print(f"AI Core exception: {e}")

    # 解析 dump 文件，返回 tensor 数据列表
    tensors = parse_exception_dump(
        dump_path="/tmp/exc_dump",
        kernel_name="main_kernel",
    )

    for t in tensors:
        data = t["data"]                # np.ndarray
        print(f"  {t['type']}[{t['index']}] dtype={t['dtype']}, "
              f"shape={data.shape}, min={data.min():.4f}, max={data.max():.4f}")
```

### 4.4 返回值说明

`parse_exception_dump()` 返回 `list[dict]`，每个 dict 包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `data` | `np.ndarray` | tensor 数据（1-D，需按实际 shape reshape） |
| `type` | `str` | `"input"` / `"output"` / `"workspace"` |
| `index` | `int` | 同类型 tensor 的序号 |
| `dtype` | `str` | 数据类型字符串，如 `"float16"` |
| `file` | `str` | `.bin` 文件路径 |

### 4.5 手动解析（可选）

如果不使用 `parse_exception_dump()` 封装，可以分步操作：

```bash
# 1. 用 CANN 的 msaicerr.py 解析 dump 文件
python $ASCEND_HOME_PATH/tools/msaicerr/msaicerr.py \
    -d /tmp/exc_dump/extra-info/data-dump/0/main_kernel.custom.<timestamp> \
    -out /tmp/parsed
```

```python
# 2. 读取 .bin 文件
from tilelang.tools.ascend_exception_dump_bin import read_msaicerr_bin
import numpy as np

data = read_msaicerr_bin(
    "/tmp/parsed/main_kernel.custom.xxx.input.0.float16.bin",
    dtype=np.float16,
    shape=(128, 128),
)
```

### 4.6 完整示例

参见 `examples/exception_dump_test/example_exception_dump.py`。

---

## 5. Dump 文件路径规则

```
<ASCEND_DUMP_PATH>/
└── extra-info/
    └── data-dump/
        └── <devId>/                              # 设备 ID
            └── <kernel_name>.custom.<timestamp>   # dump 文件
```

msaicerr.py 解析后生成 per-tensor `.bin` 文件，命名格式：

```
<kernel_name>.custom.<timestamp>.<tensor_type>.<tensor_index>.<dtype>.bin
```

例如：`main_kernel.custom.20260731191027177.input.0.float16.bin`

---

## 6. API 速查

### Python 配置

| API | 说明 |
|-----|------|
| `tilelang.PassConfigKey.TL_ASCEND_EXCEPTION_DUMP` | 配置键，值为 `"tl.ascend_exception_dump"`，默认 `False` |

### Python 工具

| 函数 | 说明 |
|------|------|
| `parse_exception_dump(dump_path, kernel_name, output_dir, wait_seconds)` | 一站式解析：查找 dump 文件 → msaicerr 解析 → 读取 numpy 数组 |
| `read_msaicerr_bin(file_path, dtype, shape, header_size)` | 读取单个 `.bin` 文件为 numpy 数组 |

### CANN 环境变量

| 环境变量 | 值 | 说明 |
|----------|-----|------|
| `ASCEND_DUMP_PATH` | 路径 | dump 文件根路径 |
| `ASCEND_DUMP_SCENE` | `aic_err_brief_dump` | 启用 AI Core 异常 dump |
| `ASCEND_HOME_PATH` | CANN 安装路径 | 定位 msaicerr.py 工具 |
