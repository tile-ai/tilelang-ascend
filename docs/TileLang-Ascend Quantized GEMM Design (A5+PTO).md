# TileLang-Ascend 量化 GEMM 设计方案（A5 + PTO）

## 一、背景与目标

华为昇腾 **A5（910_95 / Davinci C310）** 的 Cube 单元原生支持三类量化矩阵乘：

| 硬件指令 | 数据格式 | Scale | 累加器 |
|---|---|---|---|
| `TMATMUL` (FP8 路径) | `float8_e4m3_t` × `float8_e5m2_t` | 无 | fp32 |
| `TMATMUL_MX` (MXFP8) | `float8_e4m3_t/e5m2_t` + `float8_e8m0_t` 块 scale | 每 32-K 一个 exponent | fp32 |
| `TMATMUL_MX` (MXFP4) | `float4_e2m1x2_t/e1m2x2_t` + `float8_e8m0_t` | 每 32-K 一个 exponent | fp32 |

PTO-ISA (`3rdparty/pto-isa/`) 已暴露 `TMATMUL` / `TMATMUL_MX` 的 C++ intrinsic，但 TileLang 编译器前端**完全未打通**。本方案目标是在 TileLang Python DSL 上，以 `T.gemm_v0`（Phase A 复用）+ `T.gemm_mx`（Phase B/C 新增）两个 API 封装上述三类指令，走 **PTO backend + A5 平台** 路线。

## 二、总体架构（全栈分层）

```
┌──────────────────────────────────────────────────────────────┐
│  用户代码  T.gemm_v0(A[e4m3_float8], B[e5m2_float8], C[fp32]) │
│           T.gemm_mx(A, B, C, sA, sB, format="e2m1x2")       │
│              tilelang/language/ascend.py :: _dtype()         │
└──────────────────────────┬───────────────────────────────────┘
                           │ TIR intrinsic (StringImm 模板串)
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Op 注册  tl.ascend_gemm_v0 (5 inputs)                       │
│           tl.ascend_gemm_mx (7 inputs: name,A,B,C,sA,sB,init)│
│              src/op/ascend.{h,cc}                            │
└──────────────────────────┬───────────────────────────────────┘
                           │ Lower + Pass (PIPE_M / cube)
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  PTO Codegen   getType() / GetTypeLen() / PrintType()        │
│                GemmV0Codegen / GemmMxCodegen                  │
│              src/target/codegen_ascend_pto.cc                │
└──────────────────────────┬───────────────────────────────────┘
                           │ 生成: tl::ascend_pto::gemm_mx<T1,T2,...>(...)
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Runtime 模板  gemm_v0<T1,T2,...>  — 已有                    │
│                gemm_mx<T1,T2,...>  — 新增 (A5 only guard)    │
│                TileScaleL1<uint8_t, ...>  — 新增             │
│              src/tl_templates/pto/common.h                   │
└──────────────────────────┬───────────────────────────────────┘
                           │ pto::TMATMUL / pto::TMATMUL_MX
                           ▼
                   A5 CCE + PTO-ISA (TMATMUL_MX 硬件指令)
```

## 三、三阶段实施规划

### Phase A — FP8 GEMM（复用 `T.gemm_v0`）

| 项 | 内容 |
|---|---|
| API | `T.gemm_v0(A[e4m3_float8], B[e4m3_float8], C[fp32])` |
| 后端硬件 | A5 原生 `TMATMUL` 的 fp8×fp8→fp32 路径，**无需 Scale tile** |
| 打通范围 | 仅 dtype 映射层 —— Python `_dtype()` + C++ `getType`/`GetTypeLen`/`PrintType`/`AscendCopy::Lower::get_dtype` |
| 示例 | `examples/gemm/example_gemm_fp8_pto.py` |
| 工作量 | 1–2 天 |

### Phase B — MXFP8 GEMM + OCP MX Block Scale（新增 `T.gemm_mx`）

| 项 | 内容 |
|---|---|
| API | `T.gemm_mx(A, B, C, scale_a, scale_b, init=..., format="e4m3"\|"e5m2")` |
| 后端硬件 | A5 的 `TMATMUL_MX` 指令，**每 32 个 K 元素绑一个 e8m0 块 scale** |
| 新增组件 | `ascend_gemm_mx` Op 注册 · C++ `GemmMxCodegen()` · PTO 模板 `gemm_mx` + `TileScaleL1` uint8 别名 · Pass 配置 (`PIPE_M` + `cube`) |
| 关键决策 | **复用 `uint8` 表示 e8m0 scale**（不改 TVM 类型系统）；codegen 在 `TileScaleL1` 上 `uint8 → float8_e8m0_t` 重解释 |
| 示例 | `examples/gemm/example_gemm_mxfp8_pto.py` |
| 工作量 | 4–5 天 |

### Phase C — MXFP4 GEMM（扩展 `T.gemm_mx`）

| 项 | 内容 |
|---|---|
| API | `T.gemm_mx(A[uint8], B[uint8], C, sA, sB, format="e2m1x2"\|"e1m2x2")` |
| 后端硬件 | 同 B 的 `TMATMUL_MX`，但 L0A/L0B 模板用 `TileLeft<T=float4_e2m1x2_t,...>`，PTO-ISA 的 `TEXTRACT` 自动按 `x2` semantic 取前 K/2 字节 |
| 新增组件 | Python `_MX_FORMAT_TO_CTYPE` 添加 MXFP4 映射 · `common.h` 加 `float4_e2m1x2_t`/`float4_e1m2x2_t` uint8 别名 · C++ `GetTypeLen` 识别 packed types |
| 关键决策 | **L1 buffer 形状 `(M, K_logical)` uint8**，仅前 `K_logical/2` 字节有效；K 模板常量取 logical K，使 API/Scale 语义统一 |
| 示例 | `examples/gemm/example_gemm_mxfp4_pto.py` |
| 工作量 | 1–2 天 |

## 四、关键技术决策

### 决策 1：e8m0 scale 类型复用 `uint8`

| 方案 | 工作量 | 优势 | 劣势 |
|---|---|---|---|
| A. **复用 `uint8`（采用）** | 0 TypeCode 改动 | 零侵入 TVM；host 用 `torch.uint8`；codegen 层映射到 `float8_e8m0_t` | 类型语义丢失；IDE 提示不直观 |
| B. 新增 `kFloat8_e8m0 = 9U` | 改 3rdparty/tvm 5+ 文件 | 语义干净 | 跨仓库改动；CI/构建风险 |
| C. 用 TVM custom type (≥129) | 需 custom dtype 框架 | 隔离 | 现有路径未充分支持 |

**采纳 A**：scale buffer 是 host 生成的 uint8 e8m0 字节数组；codegen 时由 `TileScaleL1<uint8_t, ...>` 的 C++ 模板直接重解释为 A5 的 `float8_e8m0_t`。

### 决策 2：`gemm_mx` 引入 `format` 参数

```python
def gemm_mx(A, B, C, scale_a, scale_b, init=False, format=None):
    """
    format : "e4m3" | "e5m2" | "e2m1x2" | "e1m2x2"
      None → 从 A 的 buffer.dtype 推断（e4m3_float8 → e4m3，否则默认 e5m2）
    """
```

- **集中映射表** `_MX_FORMAT_TO_CTYPE` 把 format → C++ 类型名（如 `float4_e2m1x2_t`）
- **模板串统一格式**：`gemm_mx<{data_type_input}, {C_type}, M, N, K>`，在 codegen 层直接透传
- **Shape 校验增强**：`K % 64 == 0`（硬件必需）；MXFP4 额外 `K % 2 == 0`（打包字节）

### 决策 3：MXFP4 的 L1 buffer 形状约定

```
声明: A_L1: T.alloc_L1((M, K_logical), "uint8")
物理: 每行前 K_logical/2 字节是真实 packed FP4，后 K_logical/2 字节是零填充
```

**原因**：保持 `M, N, K_logical` 在所有 API 入口 (data buffer、scale buffer `(M, K/32)`、codegen 模板常量) 语义统一。代价是声明多占 2× storage（仅声明，不读写无用字节）。

### 决策 4：A5-only 守卫 + 非 A5 平台兜底

| 平台 | 行为 |
|---|---|
| **A5 (C310)** | CCE 编译暴露原生 `float8_e4m3_t`/`float8_e5m2_t`；`gemm_mx` 模板实例化 → `TMATMUL_MX` 真实指令 |
| **非 A5 (C220)** | `common.h` 顶部 typedef (`using float8_e4m3_t = int8_t;` 等) 让 codegen 字符串可解析；`#else` 分支提供 `gemm_mx` stub + `static_assert(sizeof(T1)==0, "only on A5")`, 实例化即报错 |
| **Python 示例** | 启动时 `determine_platform() != "A5"` → 打印 `[SKIP] ...` 并 `sys.exit(0)`，不让编译链走到 CCE |

### 决策 5：Host 端量化

`TQUANT` 封装不在本方案范围内。host 端用 PyTorch 完成：
- **FP8（Phase A）**：`torch.float16.to(torch.float8_e4m3fn)`
- **MXFP8（Phase B）**：`quantize_mxfp8_host(x, block=32)` — float16 → `float8_eXmY_t` 数据 + `uint8` e8m0 scale
- **MXFP4（Phase C）**：`quantize_mxfp4_host(x, block=32, fmt="e2m1x2")` — float16 → packed uint8 数据 + `uint8` e8m0 scale

## 五、完整调用链图示

### 5.1 Phase A — FP8 GEMM

```
用户: T.gemm_v0(A[e4m3_float8], B[e4m3_float8], C[fp32])
   │
   ├─ ascend.py::_dtype(A) → "float8_e4m3_t"       ★ 新增映射
   │    生成模板串: "gemm_v0<float8_e4m3_t, float, M, N, K, false, false>"
   │
   ▼
   TIR Intrinsic: tl.ascend_gemm_v0(name, Aptr, Bptr, Cptr, init)
   │
   ├─ Op 注册: src/op/ascend.cc:1132 (已有)
   │
   ├─ Copy 降级 (T.copy):
   │    src/op/ascend.cc:90 AscendCopy::Lower::get_dtype()
   │    → "float8_e4m3_t"                         ★ 新增分支
   │
   ├─ PTO Codegen:
   │    getType(fp8) → "float8_e4m3_t"             ★ 新增分支
   │    PrintType(fp8) → "float8_e4m3_t" + enable_fp8_
   │    GetValidShape(..., "float8_e4m3_t") → 1 byte
   │    GemmV0Codegen → pto::gemm_v0<float8_e4m3_t, float, ...>
   │
   ▼
   PTO 模板 (tl_templates/pto/common.h):
      gemm_v0<T1=fp8, T2=fp32, M, N, K, validM, validN, validK, K_tail>(...)
        └─ pto::TMATMUL / TMATMUL_ACC  ← A5 CheckMadValid 显式支持 fp8×fp8→fp32
   ▼ CCE bisheng (--cce-aicore-arch=dav-c310) → A5 硬件
```

### 5.2 Phase B/C — MX GEMM

```
用户: T.gemm_mx(A[u8], B[u8], C[fp32], sA[u8], sB[u8], format="e2m1x2")
   │
   ├─ ascend.py::gemm_mx()
   │    校验 K%64==0, sA.shape==(M,K/32), sB.shape==(K/32,N)
   │    查 _MX_FORMAT_TO_CTYPE["e2m1x2"] → "float4_e2m1x2_t"
   │    生成串: "gemm_mx<float4_e2m1x2_t, float, M, N, K>"
   │
   ▼
   TIR Op: tl.ascend_gemm_mx(name, Aptr, Bptr, Cptr, sAptr, sBptr, init)
   │
   ├─ Op 注册: src/op/ascend.cc:1142  ★ 新增 (7 inputs)
   ├─ Pass 配置:
   │    operation_config.h: "tl.ascend_gemm_mx" → PIPE_M, 读/读/写/读/读
   │    ascend_combinecv.cc: "gemm_mx" → "cube"
   │
   ├─ PTO Codegen: GemmMxCodegen()  ★ 新增
   │    解析 sA / sB shape_info，调 ResolveCubeSliceName(... TileScaleL1)
   │    生成: pto::gemm_mx<float4_e2m1x2_t, float, M,N,K,... >(A,B,C,sA,sB,clear)
   │
   ▼
   PTO 模板 (tl_templates/pto/common.h):
      #ifdef PTO_PLATFORM_A5
      gemm_mx<T1, T2, M, N, K, validM, validN, validK, K_tail>(A, B, C, sA, sB, clear)
        ├─ K 切片循环 (kL0Size=128)
        │   - TEXTRACT(l0a, A) / TEXTRACT(l0b, B)
        │   - TEXTRACT(l0sa, sA) → ScaleLeft<uint8, M, CurrentK/32>
        │   - TEXTRACT(l0sb, sB) → ScaleRight<uint8, CurrentK/32, N>
        │   - pto::TMATMUL_MX(C, l0a, l0sa, l0b, l0sb) [initflag=true: 清零写]
        │   - pto::TMATMUL_MX(C, C, l0a, l0sa, l0b, l0sb) [累加]
        │   - flag 同步 (MTE2↔MTE1↔M↔FIX)
      #else
      static_assert(sizeof(T1)==0, "A5 only");
      #endif
   ▼ bisheng (--cce-aicore-arch=dav-c310 -DREGISTER_BASE -DPTO_PLATFORM_A5) → A5
```

## 六、文件改动清单

| 文件 | 改动要点 |
|---|---|
| `tilelang/language/ascend.py` | `_dtype()` 加 FP8；新增 `gemm_mx()` + `_MX_FORMAT_TO_CTYPE` |
| `tilelang/language/pto.py` | `_dtype()` 同步 FP8 映射 |
| `src/op/ascend.h` | `TVM_DLL const Op &ascend_gemm_mx();` |
| `src/op/ascend.cc` | `ascend_gemm_mx` Op 注册；`AscendCopy::Lower::get_dtype` + `AscendAtomicAdd::Lower::get_dtype` 加 FP8 分支 |
| `src/target/codegen_ascend_pto.cc` | `getType`/`GetTypeLen`/`PrintType` 支持 fp8/fp4；新增 `GemmMxCodegen()` |
| `src/target/codegen_ascend_pto.h` | `GemmMxCodegen()` 声明 |
| `src/tl_templates/pto/common.h` | 非 A5 FP8/FP4 uint8 别名；`TileScaleL1`；`gemm_mx` 模板（A5-ifdef + stub） |
| `src/transform/common/operation_config.h` | `gemm_mx` 与 `tl.ascend_gemm_mx` → PIPE_M 配置 |
| `src/transform/ascend_combinecv.cc` | `"gemm_mx" → "cube"` |
| `examples/gemm/example_gemm_fp8_pto.py` | Phase A 示例 + A5 平台检测 |
| `examples/gemm/example_gemm_mxfp8_pto.py` | Phase B 示例（含 host 量化） |
| `examples/gemm/example_gemm_mxfp4_pto.py` | Phase C 示例（含 host 量化 + 精度参考） |

## 七、验证状态

| Phase | 语法验证 | Python 导入 | A3 平台跳过 | A5 硬件实测 |
|---|---|---|---|---|
| A FP8 | ✅ | ✅ | ✅ (`[SKIP]`) | ⏳ 待 A5 机器 |
| B MXFP8 | ✅ | ✅ | ✅ (`[SKIP]`) | ⏳ 待 A5 机器 |
| C MXFP4 | ✅ | ✅ | ✅ (`[SKIP]`) | ⏳ 待 A5 机器 |

## 八、已知风险与后续工作

### 风险 1：MXFP4 物理打包与硬件对齐
- 当前 L1 声明 `(M, K_logical)` uint8，仅前 K/2 字节有效
- 若 A5 `TMATMUL_MX` 期望严格的 `(M, K/2)` shape + 模板 K 解耦，需重构 gemm_mx 模板
- **验证方法**：在 A5 跑 `example_gemm_mxfp4_pto.py`，若 `TEXTRACT` scale 索引错位则调整

### 风险 2：Scale 绑定语义
- 当前实现用 `TEXTRACT` 从 `TileScaleL1` 取子块到局部 `ScaleLeft`/`ScaleRight`
- A5 硬件可能期望 `TGET_SCALE_ADDR`（基于 data tile 地址绑定的 scale）语义
- **备选**：在 `gemm_mx` 模板内改为 `TGET_SCALE_ADDR(l0a)` 获取 scale tile

### 风险 3：MXFP4 量化精度
- host 端 `quantize_mxfp4_host` 用 round-to-nearest + bit 截断
- A5 硬件 MXFP4 rounding mode 可能不同（默认 RNE / round-to-odd）
- **后续**：在 host 端提供 `rounding_mode` 参数，或引入 PTO `TQUANT` 在 NPU 内量化（Phase C+）

### 后续工作
1. **引入 `TQUANT`** 激活量化（host 量化是 LLM 推理的瓶颈）
2. **INT8 + post-GEMM dequant** 路径在 A3 上的 FP8 替代（`examples/deepseek_v4/int8_gemm.py`）
3. **Cost model 支持**：`3rdparty/pto-isa/include/pto/costmodel/` 当前未覆盖 MX 变体，会阻塞 autotuner
4. **CI 集成**：添加 A5-only 的 e2e 量化 GEMM test（`pytest.mark.a5`）

## 九、调用示例速查

```python
# Phase A — FP8 GEMM (A5 only)
@tilelang.jit(out_idx=[-1], target="pto", pass_configs=...)
def fp8_matmul(M, N, K, block_M, block_N, K_L1):
    @T.prim_func
    def main(A: T.Tensor((M, K), "e4m3_float8"),
             B: T.Tensor((K, N), "e4m3_float8"),
             C: T.Tensor((M, N), "float32")):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            A_L1 = T.alloc_L1((block_M, K_L1), "e4m3_float8")
            B_L1 = T.alloc_L1((K_L1, block_N), "e4m3_float8")
            C_L0 = T.alloc_L0C((block_M, block_N), "float32")
            with T.Scope("C"):
                for k in T.serial(loop_k):
                    T.copy(A[bx*block_M, k*K_L1], A_L1)
                    T.copy(B[k*K_L1, by*block_N], B_L1)
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k==0))   # ← A5 原生 FP8 TMATMUL
                T.copy(C_L0, C[bx*block_M, by*block_N])

# Phase B — MXFP8 GEMM
T.gemm_mx(A_L1, B_L1, C_L0, sA_L1, sB_L1, init=(k==0), format="e5m2")

# Phase C — MXFP4 GEMM
T.gemm_mx(A_L1, B_L1, C_L0, sA_L1, sB_L1, init=(k==0), format="e2m1x2")
```

## 十、总结

本方案以 **最小侵入** 为核心（不改 TVM 类型系统、`uint8` 承载 e8m0 scale、`gemm_mx` 复用现成的 PTO codegen 框架），在 3 个阶段内打通 **FP8 → MXFP8 → MXFP4** 三类量化矩阵乘的全栈：Python DSL · TIR Op · PTO Codegen · C++ 模板 · Pass 配置 · 平台兜底。在 A5 硬件实测后若 scale 绑定 / FP4 打包符合预期，即可投产 LLM 推理量化 kernel。
