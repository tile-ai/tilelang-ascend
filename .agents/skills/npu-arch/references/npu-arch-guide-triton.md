# NPU 架构代际说明

本文档说明 Ascend NPU 的架构代际划分及其对算子开发的影响。

---

## 目录

1. [架构代际概述](#架构代际概述)
2. [完整映射表](#完整映射表)
3. [典型硬件参数与获取方式](#典型硬件参数与获取方式)
4. [DAV_3510 微架构与编程影响](#dav_3510-微架构与编程影响)
5. [SIMT vs SIMD 硬件能力差异](#simt-vs-simd-硬件能力差异)
6. [DAV_3510 数据格式扩展](#dav_3510-数据格式扩展)
7. [架构兼容性检查清单](#架构兼容性检查清单)
8. [参考信息来源](npu-arch-guide.md#参考信息来源)

---

## 架构代际概述

### 核心概念

| 概念 | 说明 |
|-----|------|
| **__NPU_ARCH__** | Device 侧编译宏，四位数值，用于条件编译 |
| **archXX** | 算子仓架构目录简写，取 DAV 编号前两位，如 arch22、arch35 |

### 架构目录简写（archXX）

| 目录 | 对应 NpuArch | 芯片 |
|------|-------------|------|
| **arch22** | DAV_2201 | Ascend910B 系列、Ascend910_93 |
| **arch35** | DAV_3510 | Ascend950DT / Ascend950PR |

> 命名规则：`archXX` = `arch` + DAV 编号前两位。如 DAV_**22**01 → arch22，DAV_**35**10 → arch35。

### 架构代号别名

| 代号 | 对应 SocVersion | 对应 NpuArch | 说明 |
|-----|----------------|-------------|------|
| **A2** | ASCEND910B | DAV_2201 | Ascend910B1~B4, Ascend910B2C |
| **A3** | ASCEND910B (含 Ascend910_93) | DAV_2201 | 训练/推理芯片 |
| **A5** | ASCEND950 | DAV_3510 | Ascend950DT (Decode) / Ascend950PR (Prefill) |

**核心关系：一对多**

一个 NpuArch 可以对应多个 SocVersion。例如 `DAV_2201` 对应 Ascend910B1~B4、Ascend910B2C、Ascend910_93。

> **注意**：对 NPU 核内算子开发来说，通常不需要感知具体 SocVersion，使用 NpuArch 来区分芯片有利于易用性和可维护性。

### 关键细节

- `Ascend910_93` 的运行时映射关系详见 [`npu-hardware-params.md` §0](npu-hardware-params.md#0-产品映射表)

---

## 完整映射表

完整产品系列 / SocVersion / NpuArch / 芯片型号映射见 [`npu-hardware-params.md` §0 产品映射表](npu-hardware-params.md#0-产品映射表)。

---

## 典型硬件参数与获取方式

> ⚠️ **核心原则**：本节作为 NPU 架构知识在 skills 仓中的上游真源，下游 skill 应消费本节数据，**反向自引用即视为风险**。
>
> 表内为**规格典型值**（族系规格），具体 SKU 可能裁剪（如 Ascend910B3 / Ascend910B4 均为 20 核），vNPU 切分后单实例可见值更低。**算子运行时必须通过下方接口获取实际值**，硬编码典型值会越界或浪费。
>
> 完整参数参考（架构常量/典型SKU/基于公开资料与经验值）见 `npu-hardware-params.md`。

### 算力与系统规格

| 规格项 | Ascend910B2 (DAV_2201) | Ascend950PR PCIE (DAV_3510) | Ascend950PR Server (DAV_3510) |
|--------|:---:|:---:|:---:|
| CubeCore 核数 | **24** | **28** | **32** |
| 频率 (GHz) | 1.8 | 1.65 | 1.65 |
| Cube 算力 BF16/FP16 | 353T | 378T | 432T |
| Cube 算力 FP8/HiF8/MXFP8 | — | 757T | 865T |
| Cube 算力 MXFP4 | — | 1514T | 1730T |
| Vector 算力 FP16 | 22T | 47T [²](#whitepaper) | 54T [²](#whitepaper) |
| Memory 容量 (GB) | 64 | 112 | 128 |
| Memory 带宽 | 1.6 TB/s [¹](#unverified) | 1.4 TB/s [²](#whitepaper) | 1.6 TB/s [²](#whitepaper) |

> **注意**：以上为选定子型号的典型值（真源：`platform_config/*.ini`）。**同架构其他子型号的核数、频率、L2、Memory 等存在差异**（如 910B4 核数 20@1.5GHz / L2 96MB / Memory 32GB），详见 `npu-hardware-params.md` 典型SKU示例。

#### Cube 算力公式推导

Cube BF16/FP16 理论峰值算力（TFLOPS）由以下公式计算：

```
TFLOPS = M × K × N × 核数 × cube_freq(MHz) × 2 ÷ 10⁶
```

**参数含义**：

| 参数 | 值 | 来源 |
|------|:--:|------|
| M × K × N (Cube MAC 阵列) | 16×16×16 = **4096** | INI `cube_m_size=cube_k_size=cube_n_size=16` |
| AICore 核数 | 24(910B2) / 28(PCIE) / 32(Server) | INI `[SoCInfo] ai_core_cnt` |
| cube_freq | 1800(910B2) / 1650(950PR) MHz | INI `[AICoreSpec] cube_freq` |
| ×2 | FMA 计为 2 次浮点运算 | 1 次 MAC = 1 乘 + 1 加 |
| ÷10⁶ | MAC/s → TFLOPS | TFLOPS = 10¹² FLOPS = 10⁶ × 10⁶ MAC×2 |

**推导示例**：

```
950PR Server:  4096 × 32 × 1650 × 2 ÷ 10⁶ = 432.54 TFLOPS
950PR PCIE:    4096 × 28 × 1650 × 2 ÷ 10⁶ = 378.47 TFLOPS
Ascend910B2:   4096 × 24 × 1800 × 2 ÷ 10⁶ = 353.89 TFLOPS
```

> **Cube 算力与 Vector 算力**：上表中 Cube 算力为纯 Cube 单元算力，不含 Vector Core 贡献。Vector 算力单独列出，总芯片算力 = Cube + Vector。950 白皮书表3-1 给出的 950PR BF16/FP16 总算力 486/425T 即 Cube(432/378) + Vector(54/47) 之和——DAV_3510 的 Vector Core 已**原生支持 BF16**（白皮书 §4.1.2），不存在 BF16 只能用 Cube 的限制。

#### FP8_E4M3FN 算力推导

不同于 FP16/BF16 使用 `cube_m/n/k_size=16×16×16=4096`，FP8_E4M3FN 的 Cube MAC 阵列更大（INI `[DtypeMKN]` 段）：

```
DT_FLOAT8_E4M3FN = 16,32,16  →  M×K×N = 16×32×16 = 8192
```

代入公式：

```
950PR Server:  8192 × 32核 × 1650 MHz × 2(FMA) ÷ 10⁶ = 865 TFLOPS
950PR PCIE:    8192 × 28核 × 1650 MHz × 2(FMA) ÷ 10⁶ = 757 TFLOPS
```

FP8 族系（含 HiF8、MXFP8）均视为 8192 MAC/周期。MXFP4 按 INT4 的 MKN（`DT_INT4=16,64,16`）推导：16384 MAC/周期，2 倍于 FP8。

```
950PR Server:  16384 × 32核 × 1650 MHz × 2(FMA) ÷ 10⁶ = 1730 TFLOPS
950PR PCIE:    16384 × 28核 × 1650 MHz × 2(FMA) ÷ 10⁶ = 1514 TFLOPS
```

> 注：HiF8、MXFP8 在 INI 中无独立的 DtypeMKN 条目，但算力等效于 E4M3FN。950 白皮书 §4.1.1 确认：相同频率下 HiF8/MXFP8/FP8 提供 2 倍 FP16 张量算力，MXFP4 提供 4 倍；表3-1 对 PCIE 档 Cube 算力按向下取整标注为 FP8 756 / MXFP4 1513 TFLOPS（公式值为 756.9/1513.9）。

#### Vector 算力推导

Vector 算力公式（FP16）：

```
TFLOPS = vec_calc_size × vector_core_cnt × vec_freq(MHz) × 2(FMA) ÷ 10⁶
```

| 参数 | 910B2 | 950PR PCIE | 950PR Server | 来源 |
|------|:---:|:---:|:---:|------|
| vec_calc_size | 128 | 128 | 128 | INI `[AICoreSpec] vec_calc_size` |
| vector_core_cnt | 48 | 56 | 64 | INI `[SoCInfo] vector_core_cnt` |
| 频率 (MHz) | 1800 | 1650 | 1650 | 910B2: `cube_freq`（INI 无独立 `vec_freq`）；950PR: `[VectorCoreSpec] vec_freq` |

**910B2**：128 × 48 × 1800 × 2 ÷ 10⁶ = **22 TFLOPS**\
**950PR PCIE**：128 × 56 × 1650 × 2 ÷ 10⁶ = 23.7T，含 Regbase OOO 双发再 ×2 = **47 TFLOPS**[²](#whitepaper)\
**950PR Server**：128 × 64 × 1650 × 2 ÷ 10⁶ = 27T，含 Regbase OOO 双发再 ×2 = **54 TFLOPS**[²](#whitepaper)

> 950 白皮书确认：Vector Core 升级为**双发射 Register-Based SIMD 架构**（§3）并支持乱序执行（OOO，§4.1.3），单核 FP16/FP32 算力较上一代提升 100%（§4.1.2）；表3-1 官方 Vector 算力 FP16/BF16 = 54/47T、FP32 = 27/23T（均为 Server/PCIE 两档值，每档 FP16 与 BF16 相同），与上式一致。注意并非所有 Vector 指令均支持双发。

### AIV (Vector) 核数

**适用于 `CubeCore,VectorCore` 型架构**：每个 Cube Core 配 2 个 Vector Core（DAV_2201 / DAV_3510 已验证）。

**不适用于**：`AICore,VectorCore` 型（DAV_2002，AICore 与 VectorCore 非 1:2）及单核集成型（DAV_3002 三合一、DAV_1001 无独立 Vec）。详见 `npu-hardware-params.md` §3。

**实际值以 `GetCoreNumAiv()` 为准**，部分 SKU 或 vNPU 实例可能裁剪。

### Buffer 容量（每 AI Core）

| Buffer | Ascend910B2 | Ascend950PR | 用途 |
|--------|:---:|:---:|------|
| **L1** | 512 KB | 512 KB | Cube 输入缓存 |
| **L0A** | 64 KB | 64 KB | Cube 左矩阵操作数 |
| **L0B** | 64 KB | 64 KB | Cube 右矩阵操作数 |
| **L0C** | 128 KB | **256 KB** | Cube 输出（Ascend950PR 翻倍） |
| **UB** | 192 KB | **248 KB** | Vector 工作区，每 AIV 独立一份 |
| **L2** | 192 MB | **128 MB** (Server) / **112 MB** (PCIE) | 跨核共享缓存 |
| **BT** (biasSize) | 1 KB | **4 KB** | FixPipe Bias 表 |
| **SSBuffer** | — | 256 KB [¹](#unverified) | **DAV_3510 新增** AIC↔AIV 核间消息通路 |

> **UB 口径说明（"按组计"为推断，证据链见 `npu-arch-guide.md`）**：950 白皮书表4-2 原文标注 UB = **512 KB per AI Core**；与 INI 对齐的解读是按 **AI Core 组（1 AIC + 2 AIV）计物理容量**（每 AIV 物理 256 KB × 2）；INI `[VectorCoreSpec] ub_size = 253952` = **248 KB** 是**每 AIV 用户可用值**（物理 256 KB − 8 KB 预留，精确吻合）。旁证：同表 "L1 512KB per AI Core" 也只有按组理解才自洽（L1 在 AIC 侧、每组一份）。三者不矛盾，Tiling 以 `GetCoreMemSize` 返回值为准。


## DAV_3510 微架构与编程影响

### AI Core Buffer 层级

决定 LocalTensor 分配位置、DataCopy 路径选择、流水编排。

| Buffer | 用途 | 编程影响 |
|--------|-----|---------|
| **L1 Buffer** | Cube 输入缓存 | 矩阵乘左/右矩阵驻留 |
| **L0A / L0B** | Cube 操作数 | MTE1 搬入目标 |
| **L0C Buffer** | Cube 输出（**Ascend950PR 扩容**） | 影响基本块 Tiling 上限 |
| **UB / Unified Buffer** | Vector 工作区 | Vector 计算主战场，SIMT/SIMD 共享 |
| **SSBuffer** | **DAV_3510 新增**：CV 间消息通路 | 替代 GM workspace 的细粒度同步 |
| **BT / FP Buffer** | FixPipe 配置 | 量化/重排参数 |
| **ND-DMA Cache** | NDDMA 缓存 | 离散搬运优化 |

### MTE 数据搬运引擎

| 引擎 | 路径 |
|-----|------|
| MTE1 | L1 ↔ L0A/B/C, L1 ↔ UB |
| MTE2 | GM → L1 / UB |
| MTE3 | UB → GM / L1 |

> DAV_3510 对多核同时访问 Global Memory 同地址场景进行了性能优化，矩阵乘相关算子的分核模板可据此简化（不再需要为规避同地址冲突而设计复杂的错位策略）。 [¹](#unverified)

### DAV_3510 相比 DAV_2201 的关键数据通路改动

DAV_3510 新增三条通路，使 Cube 与 Vector 可直接交换数据，避免 GM workspace 中转（白皮书 §4.1.4 证实 CV 直通通路；§4.1.1 确认 L0C→UB 支持随路量化 FP32/INT32→BF16/FP16/FP8/INT8）：

1. **L0C → UBuffer**：Cube 结果直达 Vector，FixPipe 输出到 UB
2. **UBuffer ↔ L1**：Vector 与 Cube 双向直连（UB→L1 为新增方向），避免 GM 中转
3. **SSBuffer 消息通路**：CV 间细粒度同步信号 [¹](#unverified)

**典型受益场景**：FA / FIA 类融合算子可彻底消除 workspace A/B/C 的 GM 读写。

> 注：DAV_3510 上 L1→GM 和 GM→L0A/L0B 的数据通路已被删除，依赖这些路径的 kernel 需要改造为替代路径（如 L1→UB→GM、GM→L1→L0A/L0B）。 [¹](#unverified)

### 指令序列与 BufferID 同步

DAV_3510 上各执行单元拥有独立指令队列：Scalar / Cube / FixPipe / MTE1 / MTE2 / MTE3 / SIMD VF or SIMT VF。

**BufferID 同步机制**：消除原 set/wait 强制配对需求，简化多流水算子的同步代码。

> **CCU 通算融合**：DAV_3510 在 IO-Die 上新增集合通信引擎，本文不展开，详见 `npu-arch-guide.md` §CCU 通算融合开发模型（三种通信范式、KFC 调度模型）。

---

## SIMT vs SIMD 硬件能力差异

| 维度 | SIMT | SIMD |
|-----|------|------|
| 编程范式 | 标量编程（线程视角） | 矢量编程（VF 内连续计算） |
| 控制逻辑 | Warp Scheduler 硬件分支调度 | 软件展开循环 |
| GM 离散访问 | **支持直接访问**（核内 DCache） | 不支持，需先搬入 UB |
| 寄存器类型 | 标量寄存器（per-Thread） | 矢量寄存器（per-VF） |
| 寄存器组织 | 128 KB 寄存器堆，按线程并发数切分 [¹](#unverified) | 多条 VL=256 B 矢量寄存器 |

**SIMT 独有硬件**（DAV_3510 新增）：Warp Scheduler（每 AIV 4 个）[¹](#unverified) / SIMT Register File / SIMT DCache（最大 128KB，复用 UB 作 Cacheline，128B 粒度）[¹](#unverified)
**SIMD/SIMT 共享**：ALU / ICache / Unified Buffer

**SIMT 适用场景**：Gather/Scatter、Hash 插入、随机数、排序（含原子操作）。

**SIMT 不适用场景**：
- 大块 dense BF16/FP16 矩阵乘 / 卷积 — SIMD + Cube 流水更高效
- 长向量顺序计算 — SIMD 矢量化指令吞吐更高
- DAV_3510 之前架构 — SIMT 硬件不存在，无法使用

---

## DAV_3510 数据格式扩展

### Cube MMAD 支持的数据类型

DAV_3510 上 Cube MMAD 计算单元支持的数据类型（黑体为 DAV_3510 新增）：

| 格式 | 备注 |
|-----|------|
| FP16 / BF16 / HF32 / FP32 / S8 | 通用类型 |
| **FP8 E5M2 / E4M3** | 静态/动态量化 |
| **MXFP8 E5M2/E4M3** | 32 个 Data 共享 1 个 Scale |
| **MXFP4 E2M1/E1M2** | 4-bit 浮点 |
| **HiF8** | 华为自定义格式 |

> 注：DAV_3510 不再支持 4:2 稀疏矩阵计算，原依赖该特性提速的 kernel 需要改为稠密或其他支持的稀疏策略。

### 类型名映射

| 格式 | C++ 类型名 |
|-----|----------|
| FP8 E5M2 | `fp8_e5m2_t` |
| FP8 E4M3FN | `fp8_e4m3fn_t` |
| HiFloat8 | `hifloat8_t` |
| INT4 | `int4b_t`（注：用于 Vector / 权重存储，**非** Cube MMAD 支持） |

### MXFP 类型对 Tiling 的特殊要求

32 个 Data 共享 1 个 Scale，若 Scale 采用与 Data 相同的 StepK 会导致 Scale 的 TileSize 过小、带宽利用率低，需采用独立的 `scaleFactor` 缓存。

---

## 架构兼容性检查清单

开发算子时，请确认：

- [ ] 通用实现在所有目标架构上测试通过
- [ ] 如有 arch35 特殊实现，已单独测试
- [ ] Tiling 逻辑正确识别架构并选择实现
- [ ] 性能在目标架构上达到基线要求

---

## 参考信息来源

### 官方白皮书

- 《[昇腾950 NPU架构白皮书](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf)》（华为，Ascend950PR/950DT）：DAV_3510 官方规格与微架构真源。章节内容地图见 `npu-arch-guide.md` §参考信息来源。

### CANN 安装包可见（`${ASCEND_HOME_PATH}/<arch>/`）

| 文件 | 内容 |
|------|------|
| `asc/include/utils/tiling/platform/platform_ascendc.h` | `SocVersion` 枚举、`PlatformAscendC` 接口 |
| `include/platform/soc_spec.h` | `NpuArch` 枚举完整定义 |

### 标注

- <a id="unverified">¹</a> 基于公开资料与经验值，暂未在当前安装的 INI 及 950 白皮书中找到直接对应的可核验字段。详见 `npu-hardware-params.md` §4。
- <a id="whitepaper">²</a> 已由《昇腾950 NPU架构白皮书》证实（原 ¹ 项升级）。
