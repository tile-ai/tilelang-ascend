---
name: npu-arch
description: Ascend NPU 架构知识查询技能。通过芯片型号映射、架构代际划分和 archXX 特性说明，帮助判断目标平台能力、特性支持与条件编译策略。当需要确认芯片型号、NpuArch/SocVersion、架构差异、特性支持或编译分支条件时使用。
---

# Ascend NPU 架构知识

## 架构代际概述

| 概念 | 说明 |
|-----|------|
| **NpuArch** | 芯片架构号，定义指令集和微架构，运行时通过 `GetCurNpuArch()` 获取 |
| **SocVersion** | 片上系统版本，软件命名标识，运行时通过 `GetSocVersion()` 获取 |
| **__NPU_ARCH__** | Device 侧编译宏，四位数值，用于条件编译 |
| **archXX** | 算子仓目录简写，取 DAV 编号前两位（如 DAV_3510 → arch35） |
| **__DAV_C310__** | 构建系统内部宏，等价于 `NpuArch::DAV_3510` / `arch35` / `__NPU_ARCH__=3510`，不可按数值推断 |

## 完整映射表

完整产品系列 / SocVersion / NpuArch / 芯片型号映射见 [`npu-hardware-params.md` §0 产品映射表](references/npu-hardware-params.md#0-产品映射表)。

## 获取当前架构

```cpp
#include "utils/tiling/platform/platform_ascendc.h"

auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
NpuArch npuArch = ascendcPlatform.GetCurNpuArch();         // DAV_2201 / DAV_3510 / ...
platform_ascendc::SocVersion socVer = ascendcPlatform.GetSocVersion();
```

`GetCurNpuArch()` 失败返回 `NpuArch::DAV_RESV`，`GetSocVersion()` 失败返回 `SocVersion::RESERVED_VERSION`。

## DAV_3510 相对 DAV_2201 的关键变化

> 详细硬件参数真值见 `references/npu-hardware-params.md`。运行时必须以 `PlatformAscendC` 接口获取实际值，禁止硬编码。

### Buffer（同架构内通常一致）

| Buffer | DAV_2201 | DAV_3510 |
|--------|----------|----------|
| L0C | 128 KB | 256 KB |
| UB | 192 KB | 248 KB |
| BT | 1 KB | 4 KB |

### 频率/核数/L2/Memory（因子型号/形态而异）

| 参数 | Ascend910B2 (DAV_2201) | Ascend950PR PCIE | Ascend950PR Server |
|------|------------------------|------------------|-------------------|
| Cube 核数 | 24 | 28 | 32 |
| 频率 | 1.8 GHz | 1.65 GHz | 1.65 GHz |
| L2 | 192 MB | 112 MB | 128 MB |
| Memory | 64 GB | 112 GB | 128 GB |

> 详见 [典型SKU示例](references/npu-hardware-params.md#子型号变化参数)。
>
> Memory 带宽：950PR 1.6/1.4 TB/s 已由 950 白皮书表3-1 证实，910B2 仍为公开资料与经验值，详见 `npu-hardware-params.md` §4。

### 指令集与微架构

| 类别 | 变化 |
|------|------|
| 数据格式 | 新增 FP8 / MXFP8 / MXFP4 / HiF8 Cube MMAD |
| CV 直通通路 | 新增 L0C→UB、UB→L1、SSBuffer 消息 |
| 同步机制 | BufferID 替代 set/wait 强配对 |
| 编程模型 | 新增 SIMT、SIMD-Regbase、NDDMA、CCU 通算融合 |

## 详细文档索引

`references/` 按需加载：

- **`npu-hardware-params.md`** — 硬件参数参考：各架构子型号一致参数、典型SKU示例、基于公开资料与经验值的规格
- **`simt-arch-guide.md`** — SIMT 架构概念参考：线程层次结构、Warp调度机制、内存空间、UB分区、数据类型、函数调用层级、LAUNCH_BOUND与寄存器数量、SIMT与SIMD核心差异
- **`npu-arch-guide.md`**：
  - **典型硬件参数与获取方式**：核数 / Buffer 容量典型值，`GetCoreNumA*` / `GetCoreMemSize` / `aclrtGetDeviceInfo` 接口用法（含硬编码反例）
  - **DAV_3510 微架构**：AI Core Buffer 层级、MTE 引擎、CV 数据通路改动、BufferID 同步
  - **SIMT vs SIMD**：硬件能力差异与适用场景
  - **SIMD-Regbase**：寄存器组、核心优势、代码形态变化
  - **数据格式扩展**：完整数据类型表、C++ 类型名映射、MXFP Tiling 约束
  - **NDDMA / CCU**：高维 DMA 用法、CCU 三种通信范式、KFC 调度模型
  - **架构兼容性检查清单 / 参考信息来源**

## 官方资料来源

- 《[昇腾950 NPU架构白皮书](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf)》—— Ascend950PR/950DT（DAV_3510）官方规格与微架构真源，章节内容地图见 `references/npu-arch-guide.md` §参考信息来源

> **范围边界**：本技能聚焦架构判断与硬件能力识别。算子目录结构、CMake 配置、文件命名约定等工程模板内容不在本技能范围内。
