# NPU 硬件参数真值参考

> **数据源**：`${ASCEND_HOME_PATH}/<arch>/data/platform_config/*.ini`（arch 如 `aarch64-linux`、`x86_64-linux`、`arm64-linux`），参数值以运行时 `PlatformAscendC` 接口返回为准。
>
> **DAV_3510 官方规格真源**：《[昇腾950 NPU架构白皮书](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf)》——表3-1（950PR/950DT 各 SKU 核数、算力、Memory、L2）、表4-2（Memory 层次各 Buffer 物理容量）。
>
> **数据可信度约定**：本文档中「INI 字段」/「实测」均指从 ini 读取到的静态配置值。**ini 文件是发现硬件参数的必要不充分条件** — 并非每个 ini 都对应真实量产芯片，部分配置可能来自工程样片、仿真平台或规划中的 SKU。文档聚焦架构级常量（§1、§2）和关键差异速查（§3），变化参数仅以典型量产 SKU 举例。

---

## 0. 产品映射表

按 NpuArch 列出 Ascend NPU 全代际的产品系列、SocVersion 与具体芯片型号。**本表为 SKILL.md 与 `npu-arch-guide.md` 共用的真源，需要修改芯片清单时仅在此处更新。**

| 产品系列 | SocVersion | NpuArch | __NPU_ARCH__ | 芯片型号 |
|---------|-----------|---------|:---:|---------|
| Atlas 训练系列 | ASCEND910 | DAV_1001 | 1001 | Ascend910 |
| Atlas 推理系列 | ASCEND310P | DAV_2002 | 2002 | Ascend310P1, Ascend310P3, Ascend310P5, Ascend310P7 |
| Atlas A2 训练/推理 | ASCEND910B | DAV_2201 | 2201 | Ascend910B1~B4, Ascend910B2C |
| Atlas A3 训练/推理 | ASCEND910B | DAV_2201 | 2201 | Ascend910_93 |
| Atlas 200I/500 A2 推理 | ASCEND310B | DAV_3002 | 3002 | Ascend310B1~B4 |
| Atlas A5 训练 | ASCEND950 | DAV_3510 | 3510 | Ascend950DT (Decode) |
| Atlas A5 推理 | ASCEND950 | DAV_3510 | 3510 | Ascend950PR (Prefill) |
| Mate80、Pura90等系列 | Kirin9030 | DAV_3113 | 3113 | Kirin9030 |
| MatePad Edge、MateBookPro等系列 | KirinX90 | DAV_3003 | 3003 | KirinX90 |

> **一对多关系**：一个 NpuArch 可对应多个 SocVersion / 芯片型号。例如 `DAV_2201` 对应 Ascend910B1~B4、Ascend910B2C、Ascend910_93。
>
> **950 系列定位**（950 白皮书 §2/§3）：**Ascend950PR** 面向高性能推荐、大模型 **Prefill** 及多模态推理；**Ascend950DT** 覆盖预训练、后训练及推理（含 **Decode** 与 Prefill）全流程（Memory 规格见典型 SKU 示例）。上表中的 (Decode)/(Prefill) 为场景侧重标注，并非唯一用途。
>
> **运行时映射**：`Ascend910_93` 的 SocVersion 字符串在运行时映射到 `SocVersion::ASCEND910B`（非独立枚举值）。源码中虽存在 `SocVersion::ASCEND910_93` 枚举值（platform_ascendc.h），但 convertMap 中 `"Ascend910_93"` 映射到的是 `ASCEND910B`，该枚举仅在少数内部模块使用。NpuArch 同为 `DAV_2201`。

---

## 1. 服务器跨架构一致参数

以下参数在所有已验证的**服务器** Ascend NPU 架构上保持一致（Kirin 端侧平台除外）：

| 参数 | INI 字段 | 值 | 说明 |
|------|---------|:---:|------|
| L0A | `[AICoreSpec] l0_a_size` | 64 KB (65536) | Cube 左矩阵操作数 |
| L0B | `[AICoreSpec] l0_b_size` | 64 KB (65536) | Cube 右矩阵操作数 |
| Cube MAC 阵列 | `cube_m_size / cube_k_size / cube_n_size` | 16×16×16 | 一个周期完成 4096 次 MAC |

> **注意**：即便以上参数通常一致，代码中仍应通过 `GetCoreMemSize` 获取，避免硬编码。Kirin 端侧平台（DAV_3003/DAV_3113）Cube MAC 阵列不同，DAV_3113 L0A/L0B 更小，详见 §2.6/§2.7。

---

## 2. 各架构参数（子型号间通常一致）

通常同 NpuArch 的子型号中，以下参数值一致：

### 2.1 DAV_1001 — Ascend910 系列

| 参数 | INI 字段 | 值 |
|------|---------|:---:|
| NpuArch | `NpuArch` | 1001 |
| L1 | `l1_size` | 1 MB (1048576) |
| L0C | `l0_c_size` | 256 KB (262144) |
| UB | `ub_size` | 256 KB (262144) |
| L2 | `l2_size` | 32 MB (33554432) |
| BT | `bt_size` | — (不存在) |
| 稀疏 | `sparsity` | — (不存在) |
| 核心类型 | `core_type_list` | `AICore`（无独立 VectorCore） |
| 核间关系 | — | VectorCore 不存在，ai_core 内聚 Cube+Vector 功能 |

### 2.2 DAV_2002 — Ascend310P 系列

| 参数 | INI 字段 | 值 |
|------|---------|:---:|
| NpuArch | `NpuArch` | 2002 |
| L1 | `l1_size` | 1 MB (1048576) |
| L0C | `l0_c_size` | 256 KB (262144) |
| UB | `ub_size` | 256 KB (262144) |
| L2 | `l2_size` | 16 MB (16777216) |
| Memory | `memory_size` | 24 GB (24000000000) |
| BT | `bt_size` | — (不存在) |
| 稀疏 | `sparsity` | — (不存在) |
| 核心类型 | `core_type_list` | `AICore,VectorCore`（无独立 CubeCore） |
| 核间关系 | — | Cube 功能集成在 AICore 内，AICore 与 VectorCore 非 1:2 关系 |

### 2.3 DAV_2201 — Ascend910B / Ascend910_93 系列

| 参数 | INI 字段 | 值 |
|------|---------|:---:|
| NpuArch | `NpuArch` | 2201 |
| L1 | `l1_size` | 512 KB (524288) |
| L0C | `l0_c_size` | 128 KB (131072) |
| UB | `ub_size` | 192 KB (196608) |
| BT | `bt_size` | 1 KB (1024) |
| 稀疏 | `sparsity` | 1（支持 4:2） |
| 核心类型 | `core_type_list` | `CubeCore,VectorCore` |
| 核间关系 | — | CubeCore : VectorCore = 1 : 2 |

### 2.4 DAV_3002 — Ascend310B 系列

| 参数 | INI 字段 | 值 |
|------|---------|:---:|
| NpuArch | `NpuArch` | 3002 |
| L1 | `l1_size` | 1 MB (1048576) |
| L0C | `l0_c_size` | 128 KB (131072) |
| UB | `ub_size` | 248 KB (253952) |
| BT | `bt_size` | 1 KB (1024) |
| L2 | `l2_size` | 4 MB (4194304) |
| 稀疏 | `sparsity` | 1（支持 4:2） |
| 核心类型 | `core_type_list` | `AICore,VectorCore,CubeCore`（三合一） |
| 核心数 | `ai_core_cnt / cube_core_cnt / vector_core_cnt` | 均为 1（单核集成） |

### 2.5 DAV_3510 — Ascend950DT / Ascend950PR 系列

| 参数 | INI 字段 | 值 |
|------|---------|:---:|
| NpuArch | `NpuArch` | 3510 |
| L1 | `l1_size` | 512 KB (524288) |
| L0C | `l0_c_size` | 256 KB (262144) |
| UB | `ub_size` | 248 KB (253952) |
| BT | `bt_size` | 4 KB (4096) |
| 稀疏 | `sparsity` | 0（不再支持 4:2） |
| 核心类型 | `core_type_list` | `CubeCore,VectorCore` |
| 核间关系 | — | CubeCore : VectorCore = 1 : 2 |

> **与 950 白皮书对照**：白皮书表4-2 的物理容量标注为 L1 512KB / L0A/L0B 64KB / L0C 256KB per AI Core，与 INI 一致；**UB 标注为 512KB per AI Core**，与 INI 对齐的解读是按 AI Core 组（1 AIC + 2 AIV）计的物理容量（每 AIV 物理 256KB × 2，"按组计"口径为推断）。INI `ub_size` 248KB 为每 AIV 用户可用值——物理 256KB − 8KB 预留 = 248KB = 253952B，与 INI 精确吻合（本地 CANN 全部 950PR/950DT SKU 已核实一致）。完整证据链见 `npu-arch-guide.md` §Buffer 容量。

### 2.6 DAV_3003 — KirinX90 端侧系列

> **重要说明**：KirinX90 端侧平台使用，架构代号 `dav-l300`。

| 参数 | INI 字段 | 值 | 与服务器版差异 |
|------|---------|:---:|------|
| NpuArch | `NpuArch` | 3003 | 新架构，不在旧版映射表中 |
| AIC_version | `AIC_version` | AIC-L-300 | 端侧专用版本标识 |
| 核数 | `ai_core_cnt` | 1 | **单核**（服务器版多核） |
| VectorCore | `vector_core_cnt` | 1 | 单 VectorCore |
| L1 | `l1_size` | 1 MB (1048576) | 与 DAV_1001/2002/3002 相同 |
| L0A | `l0_a_size` | 64 KB (65536) | **与服务器版相同**（DAV_3113 为 32 KB） |
| L0B | `l0_b_size` | 64 KB (65536) | **与服务器版相同**（DAV_3113 为 32 KB） |
| L0C | `l0_c_size` | 128 KB (131072) | 与 DAV_2201/3002 相同 |
| UB | `ub_size` | 128 KB (131072) | **比服务器版小**（DAV_2201=192KB, DAV_3510=248KB） |
| L2 | `l2_size` | 0 | **无 L2 Cache** |
| BT | `bt_size` | 1 KB (1024) | 与 DAV_2201/3002/3113 相同 |
| Cube MAC 阵列 | `cube_m/n/k_size` | 16×8×16 | **N 维减半，服务器版为 16×16×16** |
| 稀疏 | `sparsity` | 1（支持 4:2） | 与 DAV_2201/3002/3113 相同 |
| 核心类型 | `core_type_list` | `AICore,VectorCore` | Cube 功能集成在 AICore 内 |
| vec_calc_size | `vec_calc_size` | 128 | 向量计算单元大小 |

> **开发注意事项**：
> - UB 容量仅 128 KB，Tiling 设计时需特别注意 UB 切分大小
> - 无 L2 Cache，数据搬运策略需考虑 GM 直通
> - Cube 阵列 N=8（而非 16），矩阵乘法输出维度不同
> - 单核设计，无多核切分需求
> - L0A/L0B 与服务器版相同（64KB），但 UB/L0C 较小，需注意 Cube 输出到 UB 的搬运策略



### 2.7 DAV_3113 — Kirin9030 端侧系列

> **重要说明**：Kirin 端侧平台使用 `mobile-station` 版 CANN，开发依赖 simulator 而非实际硬件。架构代号 `dav-l311`，`__NPU_ARCH__=3113`。

| 参数 | INI 字段 | 值 | 与服务器版差异 |
|------|---------|:---:|------|
| NpuArch | `NpuArch` | 3113 | 新架构，不在旧版映射表中 |
| AIC_version | `AIC_version` | AIC-L-311 | 端侧专用版本标识 |
| 核数 | `ai_core_cnt` | 1 | **单核**（服务器版多核） |
| VectorCore | `vector_core_cnt` | 1 | 单 VectorCore |
| L1 | `l1_size` | 512 KB (524288) | 与 DAV_2201/3510 相同 |
| L0A | `l0_a_size` | 32 KB (32768) | **服务器版的一半** |
| L0B | `l0_b_size` | 32 KB (32768) | **服务器版的一半** |
| L0C | `l0_c_size` | 64 KB (65536) | **服务器版的一半** |
| UB | `ub_size` | 128 KB (131072) | **比 DAV_2201(192KB) 和 DAV_3510(248KB) 都小** |
| L2 | `l2_size` | 0 | **无 L2 Cache** |
| BT | `bt_size` | 1 KB (1024) | 与 DAV_2201/3002 相同 |
| Cube MAC 阵列 | `cube_m/n/k_size` | 16×8×16 | **服务器版 16×16×16 的变体，N 维减半** |
| 稀疏 | `sparsity` | 1（支持 4:2） | 与 DAV_2201/3002 相同 |
| 核心类型 | `core_type_list` | `AICore,VectorCore` | Cube 功能集成在 AICore 内 |
| vec_calc_size | `vec_calc_size` | 128 | 向量计算单元大小 |

> **开发注意事项**：
> - UB 容量仅 128 KB，Tiling 设计时需特别注意 UB 切分大小
> - 无 L2 Cache，数据搬运策略需考虑 GM 直通
> - Cube 阵列 N=8（而非 16），矩阵乘法输出维度不同
> - 单核设计，无多核切分需求

> **关于 UB 容量**：表内值为 INI `ub_size` 字段，即 `GetCoreMemSize(CoreMemType::UB, ...)` 返回的用户可用容量。**运行时始终以该接口返回值分块**，禁止硬编码。
>
> **FB（Fix Buffer）**：FixPipe 量化 scale 存储区。INI 字段 `fb0_size` / `fb1_size` / `fb2_size` / `fb3_size`；`GetCoreMemSize(FB)` 返回 `fb0_size`。

---

### 子型号变化参数

> **同一 NpuArch 内，核数、频率、L2、Memory 等参数可能随子型号不同而变化。**
>
> `platform_config/*.ini` 涵盖多种配置（含工程样片、仿真配置），ini 文件是发现硬件参数的必要不充分条件，因此**不在此处枚举"参数范围"**。算子代码**必须通过 `PlatformAscendC` 接口运行时获取**。
>
> 作为参考，以下为有 archXX 特化开发路径的架构典型量产 SKU 示例（数据来源：对应 ini 文件）：
>
> | 架构 | 示例型号 | CubeCore | VectorCore | 频率 | L2 | Memory |
> |------|---------|:---:|:---:|:---:|:---:|:---:|
> | DAV_2201 | Ascend910B2 | 24 | 48 | 1.8 GHz | 192 MB | 64 GB |
> | DAV_3510 (PCIE) | Ascend950PR_957b | 28 | 56 | 1.65 GHz | 112 MB | 112 GB |
> | DAV_3510 (Server) | Ascend950PR_9589 | 32 | 64 | 1.65 GHz | 128 MB | 128 GB |
>
> > 注：DAV_2201 / DAV_3510 中 CubeCore : VectorCore = 1 : 2。
> >
>
> **Ascend950DT**（950 白皮书表3-1）：Cube 36/32/28 核、Vector 72/64/56 核三档；Memory 144/96 GB @ 4TB/s；L2 128MB；Cube BF16 486/432/378T、FP8 族 973/865/756T、MXFP4 1946/1730/1513T；Vector FP16/BF16 60/54/47T（三档，每档 FP16=BF16）。950PR 为 32/28 核两档（128/112GB @ 1.6/1.4TB/s，L2 128/112MB），与上表 INI 数据一致。


---

## 3. 架构关键差异速查

| 特征 | DAV_1001 | DAV_2002 | DAV_2201 | DAV_3002 | DAV_3510 | DAV_3003 (Kirin) | DAV_3113 (Kirin) |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 核心类型 | AICore | AICore+VectorCore | CubeCore+VectorCore | AICore+VectorCore+CubeCore | CubeCore+VectorCore | AICore+VectorCore | AICore+VectorCore |
| Cube:Vec 比例 | N/A (无Vec) | 非 1:2 | 1:2 | N/A (单核) | 1:2 | N/A (单核) | N/A (单核) |
| L1 | 1 MB | 1 MB | 512 KB | 1 MB | 512 KB | 1 MB | 512 KB |
| L0A | 64 KB | 64 KB | 64 KB | 64 KB | 64 KB | 64 KB | **32 KB** |
| L0B | 64 KB | 64 KB | 64 KB | 64 KB | 64 KB | 64 KB | **32 KB** |
| L0C | 256 KB | 256 KB | 128 KB | 128 KB | 256 KB | 128 KB | 64 KB |
| UB | 256 KB | 256 KB | 192 KB | 248 KB | 248 KB | 128 KB | 128 KB |
| BT | — | — | 1 KB | 1 KB | 4 KB | 1 KB | 1 KB |
| 稀疏 4:2 | — | — | 支持 | 支持 | 不支持 | 支持 | 支持 |
| Cube 阵列 | 16³ | 16³ | 16³ | 16³ | 16³ | **16×8×16** | **16×8×16** |

---

## 4. 基于公开资料与经验值的规格

以下信息无法从当前安装的 INI 直接推导，数据来源于公开资料或工程经验积累。**自《[昇腾950 NPU架构白皮书](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf)》（下称"950白皮书"）发布后，其中已被白皮书证实的条目在"信息来源"列标注对应章节**：

> **位置标注约定**：本表"文档位置"列使用章节锚点（如 `Guide §5`），不使用行号，以避免文档行号漂移导致引用失效。

| # | 内容 | 文档位置 | 信息来源 |
|---|------|---------|---------|
| 1 | SIMT Register File 128KB | Guide §5 SIMT vs SIMD | 公开资料 / 经验值 |
| 2 | SIMT DCache 最大 128KB | Guide §5 SIMT vs SIMD | 同上 |
| 3 | SSBuffer 256KB | Guide §3 Buffer 容量 / SKILL.md §DAV_3510 关键变化 | 同上 |
| 4 | CV 直通通路：L0C→UB、UB→L1、SSBuffer 消息 | Guide §4 关键数据通路改动 / SKILL.md §DAV_3510 关键变化 | L0C→UB（含随路量化）与 L1↔UB 直连：950白皮书 §4.1.1/§4.1.4；SSBuffer 消息通路：公开资料 / 经验值 |
| 5 | L1→GM 和 GM→L0A/L0B 通路已删除 | Guide §4 关键数据通路改动 | 公开资料 / 经验值 |
| 6 | BufferID 取代 set/wait 同步 | Guide §4 指令序列与 BufferID 同步 / SKILL.md §DAV_3510 关键变化 | 950白皮书 §4.1.6 |
| 7 | 多核同时访问 GM 同地址性能优化 | Guide §4 MTE 数据搬运引擎 | 公开资料 / 经验值 |
| 8 | SIMD-Regbase：OOO 指令双发 | Guide §6 SIMD-Regbase | 950白皮书 §3/§4.1.2/§4.1.3 |
| 9 | Warp Scheduler 每 AIV 4 个 | Guide §5 SIMT vs SIMD | 公开资料 / 经验值 |
| 10 | NDDMA + ND-DMA Cache 规格 | Guide §8 NDDMA 高维 DMA | 950白皮书 §4.1.5 |
| 11 | CCU 三种通信范式及 KFC 调度变化 | Guide §9 CCU 通算融合 | CCU 硬件架构与算法支持：950白皮书 §4.6.4；三种通信范式与 KFC 调度细节：公开资料 / 经验值 |
| 12 | Vector 算力 950PR Server=54T / PCIE=47T：基准 27T/23.7T × 双发 2，但并非所有 Vector 指令均支持双发 | Guide §3 算力与系统规格 → Vector 算力推导 | 950白皮书 表3-1 + §3/§4.1.2/§4.1.3 |
| 13 | Memory 带宽 Server 1.6 TB/s / PCIE 1.4 TB/s | Guide §3 算力与系统规格 | 950白皮书 表3-1/§4.3.1；910B2 带宽仍为公开资料 / 经验值 |
