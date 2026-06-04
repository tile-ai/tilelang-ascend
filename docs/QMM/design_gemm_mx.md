# GEMM MX (量化矩阵乘) 设计文档

## 1. 概述

### 1.1 算子名称

GEMM MX — MXFP 量化矩阵乘（基于 pto-isa TMATMUL_MX 指令）

### 1.2 功能描述

在 tilelang-ascend 编译器框架中封装 pto-isa 的 `TMATMUL_MX`、`TQUANT`、`TDEQUANT` 指令，使 GEMM 支持 MXFP（Microscaling Floating Point）量化矩阵乘。提供两层 API：高层自动路由 + 底层直接调用，Scale tile 由编译器自动管理。

### 1.3 数学公式

$$
C_{i,j} = \sum_{k} \text{dequant}(A_{i,k}, S^A_{i,\lfloor k/32 \rfloor}) \times \text{dequant}(B_{k,j}, S^B_{\lfloor k/32 \rfloor,j})
$$

其中 $\text{dequant}(x, s) = x \times 2^{s - \text{bias}}$，$S^A$、$S^B$ 为 E8M0 block scale，每 32 个 K 元素对应一个 scale。

### 1.4 数据流图

```
                    ┌─ GM[A: FP8] ──T.copy──→ L1[a_l1] ──TEXTRACT──→ L0A[a_l0a] ─┐
                    │                                                              │
                    ├─ GM[Sa: E8M0] ──────────────────────────→ GetScale ─────────┤
                    │                                                              ├──→ TMATMUL_MX → L0C → GM[C: float]
                    ├─ GM[B: FP8] ──T.copy──→ L1[b_l1] ──TEXTRACT──→ L0B[b_l0b] ─┤
                    │                                                              │
                    └─ GM[Sb: E8M0] ──────────────────────────→ GetScale ─────────┘
```

### 1.5 支持的量化格式

| 格式 | A dtype | B dtype | Scale dtype | Block Size | K 约束 |
|------|---------|---------|------------|------------|--------|
| MXFP8 | `float8_e4m3` / `float8_e5m2` | `float8_e4m3` / `float8_e5m2` | `float8_e8m0` | 32 | K % 64 == 0 |
| MXFP4 | `float4_e2m1x2` / `float4_e1m2x2` | `float4_e2m1x2` / `float4_e1m2x2` | `float8_e8m0` | 32 | K % 64 == 0, K even |

---

## 2. 现有架构分析

### 2.1 当前 GEMM 调用链路（PTO 后端）

```
Python API                IR Op                     PTO Codegen              Runtime Template            pto-isa
─────────                 ──────                    ───────────              ────────────────            ───────
T.gemm_v0(A,B,C)    →   tl.ascend_gemm_v0   →     GemmV0Codegen()    →    gemm_v0<...>()      →    TMATMUL/TMATMUL_ACC
  ascend.py:341           ascend.cc:1132            codegen_ascend_pto:1250  common.h:115              pto_instr.hpp
  模板串编码7个参数       5个输入(opaque)           extractTemplateParams     K-splitting+L1→L0+sync    硬件指令
```

**关键特征**：
- 模板串 `"gemm_v0<dtype_in, dtype_out, M, N, K, transA, transB>"` 编码所有属性
- PTO codegen 解析模板串 → `GetValidShape()` 对齐 → 生成 `tl::ascend_pto::gemm_v0<...>`
- Runtime template 内部管理 K 维 L0 切割（kL0Size=128）、L1→L0A/L0B 搬运、同步
- 仅支持 `TMATMUL` / `TMATMUL_ACC`（标准矩阵乘），不支持 `TMATMUL_MX`

### 2.2 当前量化支持现状

| 能力 | 状态 | 说明 |
|------|------|------|
| FP8 dtype 类型系统 | ✅ 已定义 | `e4m3_float8`, `e5m2_float8` 在 `ast/ir.py` 定义，但 codegen 注释掉 |
| FP8 PTO codegen | ❌ 注释 | `codegen_ascend_pto.cc:460` FP8 support commented out |
| INT8 GEMM | ✅ 软件方案 | INT8×INT8→INT32 + Vector 核 scaling，`examples/deepseek_v4/int8_gemm.py` |
| INT4 反量化 | ✅ 软件方案 | V 核 unpack + INT8 GEMM，`examples/dequantize_gemm/` |
| MXFP (TMATMUL_MX) | ❌ 未实现 | pto-isa 已支持，tilelang-ascend 无封装 |
| TQUANT 在线量化 | ❌ 未实现 | pto-isa 已支持，tilelang-ascend 无封装 |
| TDEQUANT 反量化 | ❌ 未实现 | pto-isa 已支持（仅 A2/A3），tilelang-ascend 无封装 |

### 2.3 PTO-ISA TMATMUL_MX API 签名

```cpp
// 基本模式（初始化累加器）
pto::TMATMUL_MX(
    TileAcc cMatrix,        // 输出累加器
    TileLeft aMatrix,       // A 矩阵 L0 tile
    TileLeftScale aScale,   // A 的 E8M0 scale tile
    TileRight bMatrix,      // B 矩阵 L0 tile
    TileRightScale bScale   // B 的 E8M0 scale tile
);

// 累加模式（cOut = cIn + A*B）
pto::TMATMUL_MX(
    TileAcc cOutMatrix,     // 输出累加器
    TileAcc cInMatrix,      // 输入累加器（与输出相同）
    TileLeft aMatrix,
    TileLeftScale aScale,
    TileRight bMatrix,
    TileRightScale bScale
);
```

**硬件约束**（A5 架构）：
- K 必须是 64 的倍数
- Scale block = 32，scale tile 维度：`[M, ceil(K/32)]`（A）和 `[ceil(K/32), N]`（B）
- 累加器类型固定为 `float`
- Scale tile 通过 `GetScaleAddr()` 获取（硬件相对地址）或 `TGET_SCALE_ADDR`
- **不支持 Kirin9030**（static_assert 失败），**不支持 A2/A3**

---

## 3. 设计方案

### 3.1 编程模式选型

**选定模式**: Developer + 混合

**理由**：用户选择编译器自动管理 Scale tile，匹配 Developer 模式理念。Scale tile 的分配和搬运对 DSL 用户透明，编译器在 `gemm_mx` runtime template 中自动处理。

### 3.2 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Python DSL 层                                │
│                                                                     │
│  T.gemm_v0(A,B,C, quant_mode="mxfp8", scale_A=Sa, scale_B=Sb)     │
│       │  (高层 API，自动路由)                                        │
│       ↓                                                             │
│  T.gemm_mx(A,B,C, scale_A, scale_B, init=False)                   │
│       │  (底层 API，直接 MXFP GEMM)                                 │
│       ↓                                                             │
│  T.tile.quant(dst, src, exp_out, mode="mxfp8")  ← 在线量化         │
│  T.tile.dequant(dst, src, scale, offset)        ← 反量化           │
│       ↓                                                             │
│  tir.call_intrin("tl.ascend_gemm_mx", template_str, ...)           │
│  tir.call_intrin("tl.ascend_tquant", ...)                          │
│  tir.call_intrin("tl.ascend_tdequant", ...)                        │
└──────────────────────────────┬──────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│                      C++ IR 层                                       │
│                                                                      │
│  src/op/ascend.cc:                                                   │
│    TIR_DEFINE_TL_BUILTIN(ascend_gemm_mx)  → 7 inputs                │
│    TIR_DEFINE_TL_BUILTIN(ascend_tquant)   → 6 inputs                │
│    TIR_DEFINE_TL_BUILTIN(ascend_tdequant) → 5 inputs                │
│                                                                      │
│  src/transform/common/operation_config.h:                            │
│    ascend_gemm_mx  → PIPE_M  (Cube pipeline)                        │
│    ascend_tquant   → PIPE_V  (Vector pipeline)                      │
│    ascend_tdequant → PIPE_V  (Vector pipeline)                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│                      PTO Codegen 层                                  │
│                                                                      │
│  src/target/codegen_ascend_pto.cc:                                   │
│    GemmMxCodegen(op)   → 解析模板串 → gemm_mx<...>()                │
│    TQuantCodegen(op)   → tquant<...>()                              │
│    TDequantCodegen(op) → tdequant<...>()                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│                   PTO Runtime Template 层                             │
│                                                                      │
│  src/tl_templates/pto/common.h:                                     │
│    gemm_mx<T1, T2, TS, M, N, K, ...>(A, B, C, Sa, Sb, clear)      │
│      → K splitting (kL0Size)                                         │
│      → L1→L0A/L0B copy (TEXTRACT)                                   │
│      → GetScaleAddr / TGET_SCALE_ADDR                                │
│      → pto::TMATMUL_MX(C, l0a, sa, l0b, sb)                        │
│    tquant_mxfp8(dst, src, exp, max_buf, scaling_buf)                │
│      → pto::TQUANT<QuantType::MXFP8>(...)                          │
│    tdequant(dst, src, scale, offset)                                │
│      → pto::TDEQUANT(...)                                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│                      pto-isa 硬件层                                   │
│                                                                      │
│  pto::TMATMUL_MX      — MXFP 量化矩阵乘硬件指令 (A5 only)          │
│  pto::TQUANT           — 在线量化 FP32/BF16/FP16 → MXFP8 (A5 only) │
│  pto::TDEQUANT         — 反量化 INT8/INT16 → FP32 (A2/A3 only)     │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.3 API 设计

#### 3.3.1 高层 API：扩展 `T.gemm_v0`

```python
# tilelang/language/ascend.py

def gemm_v0(A, B, C, transpose_A=False, transpose_B=False, init=False,
            quant_mode=None, scale_A=None, scale_B=None):
    """
    标准 GEMM 扩展。当 quant_mode 非 None 时，自动路由到 MXFP GEMM。

    Args:
        quant_mode: None (标准GEMM) | "mxfp8" | "mxfp4"
        scale_A: E8M0 scale tile for A, shape=(M, ceil(K/32)), dtype=float8_e8m0
        scale_B: E8M0 scale tile for B, shape=(ceil(K/32), N), dtype=float8_e8m0
    """
    if quant_mode is not None:
        assert scale_A is not None and scale_B is not None
        return gemm_mx(A, B, C, scale_A, scale_B,
                       transpose_A=transpose_A, transpose_B=transpose_B,
                       init=init, quant_mode=quant_mode)
    # ... 原有标准 GEMM 逻辑不变 ...
```

#### 3.3.2 底层 API：`T.gemm_mx`

```python
# tilelang/language/ascend.py

def gemm_mx(A, B, C, scale_A, scale_B,
            transpose_A=False, transpose_B=False, init=False,
            quant_mode="mxfp8"):
    """
    MXFP 量化矩阵乘。直接映射到 pto::TMATMUL_MX。

    Args:
        A: FP8/FP4 矩阵 tile (L1 级别)
        B: FP8/FP4 矩阵 tile (L1 级别)
        C: 累加器 (L0C 级别, dtype=float)
        scale_A: E8M0 scale tile for A
        scale_B: E8M0 scale tile for B
        quant_mode: "mxfp8" | "mxfp4"
    """
    A = _legalize_arguments(A)
    B = _legalize_arguments(B)
    C = _legalize_arguments(C)
    scale_A = _legalize_arguments(scale_A)
    scale_B = _legalize_arguments(scale_B)

    M, N = C.shape[-2], C.shape[-1]
    K = A.shape[-2] if transpose_A else A.shape[-1]

    Aptr = _retrieve_ptr(A, "r")
    Bptr = _retrieve_ptr(B, "r")
    Cptr = _retrieve_ptr(C, "w" if init is True else "rw")
    SaPtr = _retrieve_ptr(scale_A, "r")
    SbPtr = _retrieve_ptr(scale_B, "r")

    dtype_A = _dtype(A)
    dtype_C = _dtype(C)
    dtype_S = _dtype(scale_A)  # float8_e8m0

    return T.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_gemm_mx"),
        f"gemm_mx<{dtype_A}, {dtype_C}, {dtype_S}, {M}, {N}, {K}, "
        f"{str(transpose_A).lower()}, {str(transpose_B).lower()}, {quant_mode}>",
        Aptr, Bptr, Cptr, SaPtr, SbPtr, init,
    )
```

**IR 调用签名**：
```
tl.ascend_gemm_mx(
    "gemm_mx<dtype_A, dtype_C, dtype_S, M, N, K, transA, transB, quant_mode>",
    Aptr, Bptr, Cptr, SaPtr, SbPtr, init
)
```
- args[0]: 模板串
- args[1]: A buffer pointer
- args[2]: B buffer pointer
- args[3]: C buffer pointer
- args[4]: scale_A pointer
- args[5]: scale_B pointer
- args[6]: init (bool)

#### 3.3.3 底层 MMA API：`T.mma_mx`

```python
# tilelang/language/customize.py

def npu_gemm_mx(A, B, C, scale_A, scale_B, init=False):
    """
    低层 MXFP MMA 指令。A,B 为 L0A/L0B 级别。
    映射到 pto::TMATMUL_MX，无 K 切割。
    """
    M, N = C.shape[-2], C.shape[-1]
    K = A.shape[-1]

    Aptr = retrieve_ptr(A, "r")
    Bptr = retrieve_ptr(B, "r")
    Cptr = retrieve_ptr(C, "w" if init is True else "rw")
    SaPtr = retrieve_ptr(scale_A, "r")
    SbPtr = retrieve_ptr(scale_B, "r")

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_mma_mx"),
        f"mma_mx<{_dtype(A)}, {_dtype(C)}, {_dtype(scale_A)}, {M}, {N}, {K}>",
        Aptr, Bptr, Cptr, SaPtr, SbPtr, init,
    )
```

#### 3.3.4 量化 API：`T.tile.quant`

```python
# tilelang/language/ascend_tile.py

def quant(dst, src, exp_out, max_buf=None, scaling_buf=None, mode="mxfp8"):
    """
    TQUANT 在线量化指令。将 FP32/BF16/FP16 数据量化为 MXFP8。

    Args:
        dst: 输出 FP8 tile (uint8 存储)
        src: 输入 FP32/BF16/FP16 tile
        exp_out: 输出 E8M0 exponent tile
        max_buf: 临时 max 缓冲 (与 src 同 dtype)
        scaling_buf: 临时 scaling 缓冲 (与 src 同 dtype)
        mode: "mxfp8" | "int8_sym" | "int8_asym"
    """
    assert mode == "mxfp8", "Currently only mxfp8 quantization is supported"
    dst_ptr = _retrieve_ptr(dst, "w")
    src_ptr = _retrieve_ptr(src, "r")
    exp_ptr = _retrieve_ptr(exp_out, "w")
    max_ptr = _retrieve_ptr(max_buf, "rw") if max_buf else None
    scaling_ptr = _retrieve_ptr(scaling_buf, "rw") if scaling_buf else None

    return T.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_tquant"),
        f"tquant<{_dtype(src)}, {_dtype(dst)}, {mode}>",
        dst_ptr, src_ptr, exp_ptr, max_ptr, scaling_ptr,
    )
```

#### 3.3.5 反量化 API：`T.tile.dequant`

```python
def dequant(dst, src, scale, offset=None):
    """
    TDEQUANT 反量化指令 (A2/A3 only)。

    Args:
        dst: 输出 float tile
        src: 输入 int8/int16 tile
        scale: float scale tile
        offset: float offset tile (仅 INT8_ASYM 模式)
    """
    dst_ptr = _retrieve_ptr(dst, "w")
    src_ptr = _retrieve_ptr(src, "r")
    scale_ptr = _retrieve_ptr(scale, "r")
    offset_ptr = _retrieve_ptr(offset, "r") if offset else None

    return T.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_tdequant"),
        f"tdequant<{_dtype(src)}, {_dtype(dst)}>",
        dst_ptr, src_ptr, scale_ptr, offset_ptr,
    )
```

### 3.4 C++ IR 注册

```cpp
// src/op/ascend.h — 新增声明
TVM_DLL const Op &ascend_gemm_mx();       // 7 inputs
TVM_DLL const Op &ascend_mma_mx();        // 7 inputs
TVM_DLL const Op &ascend_tquant();         // 6 inputs
TVM_DLL const Op &ascend_tdequant();       // 5 inputs

// src/op/ascend.cc — 新增注册
TIR_DEFINE_TL_BUILTIN(ascend_gemm_mx)
    .set_num_inputs(7)    // template_str, Aptr, Bptr, Cptr, SaPtr, SbPtr, init
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_mma_mx)
    .set_num_inputs(7)    // template_str, Aptr, Bptr, Cptr, SaPtr, SbPtr, init
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_tquant)
    .set_num_inputs(6)    // template_str, DstPtr, SrcPtr, ExpPtr, MaxPtr, ScalingPtr
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_tdequant)
    .set_num_inputs(5)    // template_str, DstPtr, SrcPtr, ScalePtr, OffsetPtr
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));
```

### 3.5 Pipeline 配置

```cpp
// src/transform/common/operation_config.h — 新增条目

// GEMM MX: 5 个读 (A, B, Sa, Sb) + 1 个写 (C), Cube pipeline
{"tl.ascend_gemm_mx",  {{{1, "read"}, {2, "read"}, {3, "write"},
                          {4, "read"}, {5, "read"}}, "PIPE_M"}},

// MMA MX: 同上，L0 级别
{"tl.ascend_mma_mx",   {{{1, "read"}, {2, "read"}, {3, "write"},
                          {4, "read"}, {5, "read"}}, "PIPE_M"}},

// TQUANT: Vector pipeline
{"tl.ascend_tquant",   {{{1, "write"}, {2, "read"}, {3, "write"},
                          {4, "write"}, {5, "write"}}, "PIPE_V"}},

// TDEQUANT: Vector pipeline
{"tl.ascend_tdequant", {{{1, "write"}, {2, "read"},
                          {3, "read"}, {4, "read"}}, "PIPE_V"}},
```

### 3.6 PTO Codegen

```cpp
// src/target/codegen_ascend_pto.h — 新增声明
void GemmMxCodegen(const CallNode *op);
void TQuantCodegen(const CallNode *op);
void TDequantCodegen(const CallNode *op);

// src/target/codegen_ascend_pto.cc — VisitExpr_ 分发
} else if (op->op.same_as(tl::ascend_gemm_mx())) {
    GemmMxCodegen(op);
} else if (op->op.same_as(tl::ascend_mma_mx())) {
    MmaMxCodegen(op);
} else if (op->op.same_as(tl::ascend_tquant())) {
    TQuantCodegen(op);
} else if (op->op.same_as(tl::ascend_tdequant())) {
    TDequantCodegen(op);
}
```

#### GemmMxCodegen 实现

```cpp
void CodeGenTileLangAscendPto::GemmMxCodegen(const CallNode *op) {
  std::string template_args = Downcast<StringImm>(op->args[0])->value;

  // 解析模板串参数
  // "gemm_mx<dtype_A, dtype_C, dtype_S, M, N, K, transA, transB, quant_mode>"
  auto params = extractTemplateParamsEx(template_args, {
      "data_type_input", "data_type_output", "data_type_scale",
      "M", "N", "K", "transpose_A", "transpose_B", "quant_mode"
  });

  uint32_t K = std::stoi(params["M"]) ...;  // 同 GemmV0Codegen 逻辑
  uint32_t kL0split = (K + kL0SliceSize - 1) / kL0SliceSize;
  uint32_t kL0Tail = K - (kL0split - 1) * kL0SliceSize;

  // 数据 tile 解析
  ShapeInfo a_info = GetSliceInfo(op->args[1].as<CallNode>());
  ShapeInfo b_info = GetSliceInfo(op->args[2].as<CallNode>());
  ShapeInfo c_info = GetSliceInfo(op->args[3].as<CallNode>());
  ShapeInfo sa_info = GetSliceInfo(op->args[4].as<CallNode>());
  ShapeInfo sb_info = GetSliceInfo(op->args[5].as<CallNode>());

  std::string a_name = ResolveCubeSliceName(a_info, kAscendPtoScope + "TileMatL1");
  std::string b_name = ResolveCubeSliceName(b_info, kAscendPtoScope + "TileMatL1");
  std::string c_name = ResolveCubeSliceName(c_info, "pto::TileAcc");
  std::string sa_name = ResolveSliceName(sa_info, kAscendPtoScope + "TileScaleA");
  std::string sb_name = ResolveSliceName(sb_info, kAscendPtoScope + "TileScaleB");

  // 生成 PTO 调用
  this->PrintIndent();
  this->stream << kAscendPtoScope << "gemm_mx<"
               << params["data_type_input"] << ", "
               << params["data_type_output"] << ", "
               << params["data_type_scale"] << ", "
               << GetValidShape(M) << ", " << GetValidShape(N) << ", " << GetValidShape(K) << ", "
               << params["M"] << ", " << params["N"] << ", " << params["K"]
               << ", " << kL0Tail << ", "
               << params["transpose_A"] << ", " << params["transpose_B"]
               << ", " << params["quant_mode"]
               << ">(" << a_name << ", " << b_name << ", " << c_name
               << ", " << sa_name << ", " << sb_name
               << ", " << PrintExpr(op->args[6]) << ");\n";
}
```

#### TQuantCodegen 实现

```cpp
void CodeGenTileLangAscendPto::TQuantCodegen(const CallNode *op) {
  std::string template_args = Downcast<StringImm>(op->args[0])->value;
  auto params = extractTemplateParamsEx(template_args, {
      "data_type_src", "data_type_dst", "mode"
  });

  this->PrintIndent();
  this->stream << kAscendPtoScope << "tquant_mxfp8<"
               << params["data_type_src"] << ", "
               << params["data_type_dst"] << ", "
               << params["mode"]
               << ">(";
  // dst, src, exp, max_buf, scaling_buf
  for (int i = 1; i <= 5; i++) {
    if (i > 1) this->stream << ", ";
    if (op->args[i].as<IntImm>() && op->args[i].as<IntImm>()->value == 0) {
      this->stream << "nullptr";  // null pointer for optional args
    } else {
      this->stream << PrintExpr(op->args[i]);
    }
  }
  this->stream << ");\n";
}
```

#### TDequantCodegen 实现

```cpp
void CodeGenTileLangAscendPto::TDequantCodegen(const CallNode *op) {
  std::string template_args = Downcast<StringImm>(op->args[0])->value;
  auto params = extractTemplateParamsEx(template_args, {
      "data_type_src", "data_type_dst"
  });

  this->PrintIndent();
  this->stream << kAscendPtoScope << "tdequant<"
               << params["data_type_src"] << ", "
               << params["data_type_dst"]
               << ">(";
  for (int i = 1; i <= 4; i++) {
    if (i > 1) this->stream << ", ";
    this->stream << PrintExpr(op->args[i]);
  }
  this->stream << ");\n";
}
```

### 3.7 PTO Runtime Template（新增）

```cpp
// src/tl_templates/pto/common.h — 新增

// ============================================================================
// MXFP Scale Tile 类型
// ============================================================================

template <typename T, int Rows, int Cols, int RowValid = Rows, int ColValid = Cols>
using TileScaleA = pto::Tile<pto::TileType::ScaleLeft, T, Rows, Cols,
                              pto::BLayout::RowMajor, RowValid, ColValid,
                              pto::SLayout::RowMajor, 32, pto::PadValue::Zero>;

template <typename T, int Rows, int Cols, int RowValid = Rows, int ColValid = Cols>
using TileScaleB = pto::Tile<pto::TileType::ScaleRight, T, Rows, Cols,
                              pto::BLayout::ColMajor, RowValid, ColValid,
                              pto::SLayout::ColMajor, 32, pto::PadValue::Zero>;

// ============================================================================
// gemm_mx: MXFP 量化矩阵乘 runtime wrapper
// ============================================================================

template <typename T1, typename T2, typename TS,
          uint32_t M, uint32_t N, uint32_t K,
          uint32_t validM = M, uint32_t validN = N, uint32_t validK = K,
          uint32_t K_tail,
          bool transpose_A = false, bool transpose_B = false,
          const char* quant_mode = "mxfp8">
AICORE PTO_INLINE void
gemm_mx(std::conditional_t<transpose_A, TileMatL1<T1, K, M, validK, validM>,
                             TileMatL1<T1, M, K, validM, validK>> &A,
        std::conditional_t<transpose_B, TileMatL1<T1, N, K, validN, validK>,
                             TileMatL1<T1, K, N, validK, validN>> &B,
        pto::TileAcc<T2, M, N, validM, validN> &C,
        TileScaleA<TS, validM, validK / 32> &Sa,
        TileScaleB<TS, validK / 32, validN> &Sb,
        bool clear) {

  constexpr uint32_t kL0Size = 128;
  constexpr uint32_t kMXScaleFactor = 32;
  constexpr uint32_t kMXScalePerL0 = kL0Size / kMXScaleFactor;  // 4 scale values per L0 slice

  const uint32_t kL0split = (K + kL0Size - 1) / kL0Size;
  bool initflag = false;

  auto war_event_id = (event_t)(((int)EVENT_ID0 + 1) % 8);
  set_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);
  wait_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);

  for (uint32_t kL0Idx = 0; kL0Idx < kL0split; kL0Idx++) {
    initflag = (clear && (kL0Idx == 0));
    const bool is_tail_block = (kL0Idx == kL0split - 1);

    if (is_tail_block) {
      // Tail block: use K_tail-sized L0 tiles + K_tail/32 scale elements
      constexpr uint32_t kMXScaleTail = K_tail / kMXScaleFactor;

      TileMatL0A<T1, M, K_tail, M, K_tail> l0a;
      TileMatL0B<T1, K_tail, N, K_tail, N> l0b;
      pto::TASSIGN(l0a, 0x0);
      pto::TASSIGN(l0b, 0x0);

      // Scale tiles for tail block
      TileScaleA<TS, validM, kMXScaleTail> sa_tile;
      TileScaleB<TS, kMXScaleTail, validN> sb_tile;

      set_flag(PIPE_M, PIPE_MTE1, war_event_id);
      wait_flag(PIPE_M, PIPE_MTE1, war_event_id);

      // Copy L1 → L0A (data)
      if constexpr (!transpose_A) {
        copy_l1_to_l0a<T1, M, K_tail, M, K, false>(l0a, A, 0, kL0Idx * K_tail);
      }  // transpose path similar...

      // Copy L1 → L0B (data)
      if constexpr (!transpose_B) {
        copy_l1_to_l0b<T1, K_tail, N, K, N, false>(l0b, B, kL0Idx * K_tail, 0);
      }

      // Get scale addresses
      // Scale A tile is at hardware-relative address based on L0A position
      uint64_t sa_addr = GetScaleAddr(A, kL0Idx * kMXScalePerL0);
      pto::TASSIGN(sa_tile, sa_addr);
      uint64_t sb_addr = GetScaleAddr(B, kL0Idx * kMXScalePerL0);
      pto::TASSIGN(sb_tile, sb_addr);

      set_flag(PIPE_MTE1, PIPE_M, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_M, war_event_id);

      if (initflag) {
        pto::TMATMUL_MX(C, l0a, sa_tile, l0b, sb_tile);
      } else {
        pto::TMATMUL_MX(C, C, l0a, sa_tile, l0b, sb_tile);
      }
    } else {
      // Standard block: kL0Size=128 slices + kMXScalePerL0=4 scale elements
      TileMatL0A<T1, M, kL0Size, M, kL0Size> l0a;
      TileMatL0B<T1, kL0Size, N, kL0Size, N> l0b;
      pto::TASSIGN(l0a, 0x0);
      pto::TASSIGN(l0b, 0x0);

      TileScaleA<TS, validM, kMXScalePerL0> sa_tile;
      TileScaleB<TS, kMXScalePerL0, validN> sb_tile;

      set_flag(PIPE_M, PIPE_MTE1, war_event_id);
      wait_flag(PIPE_M, PIPE_MTE1, war_event_id);

      // Copy L1 → L0A
      if constexpr (!transpose_A) {
        copy_l1_to_l0a<T1, M, kL0Size, M, K, false>(l0a, A, 0, kL0Idx * kL0Size);
      } else {
        // transpose path...
      }

      // Copy L1 → L0B
      if constexpr (!transpose_B) {
        copy_l1_to_l0b<T1, kL0Size, N, K, N, false>(l0b, B, kL0Idx * kL0Size, 0);
      } else {
        // transpose path...
      }

      // Get scale addresses
      uint64_t sa_addr = GetScaleAddr(A, kL0Idx * kMXScalePerL0);
      pto::TASSIGN(sa_tile, sa_addr);
      uint64_t sb_addr = GetScaleAddr(B, kL0Idx * kMXScalePerL0);
      pto::TASSIGN(sb_tile, sb_addr);

      set_flag(PIPE_MTE1, PIPE_M, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_M, war_event_id);

      if (initflag) {
        pto::TMATMUL_MX(C, l0a, sa_tile, l0b, sb_tile);
      } else {
        pto::TMATMUL_MX(C, C, l0a, sa_tile, l0b, sb_tile);
      }

      set_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
    }
  }

  set_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
  wait_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
}

// ============================================================================
// mma_mx: 低层 MXFP MMA (无 K 切割)
// ============================================================================

template <typename T1, typename T2, typename TS, int M, int N, int K>
AICORE PTO_INLINE void
mma_mx(TileMatL0A<T1, M, K> l0a,    // FP8/FP4 L0A tile
       TileMatL0B<T1, K, N> l0b,    // FP8/FP4 L0B tile
       pto::TileAcc<T2, M, N> &C,   // Float accumulator
       TileScaleA<TS, M, K/32> sa,  // Scale A tile
       TileScaleB<TS, K/32, N> sb,  // Scale B tile
       bool init) {
  if (init) {
    pto::TMATMUL_MX(C, l0a, sa, l0b, sb);
  } else {
    pto::TMATMUL_MX(C, C, l0a, sa, l0b, sb);
  }
}

// ============================================================================
// tquant_mxfp8: 在线量化
// ============================================================================

template <typename TSrc, typename TDst, const char* Mode>
AICORE PTO_INLINE void
tquant_mxfp8(Tile<TDst> &dst,
             Tile<TSrc> &src,
             Tile<uint8_t> *exp,       // E8M0 exponents output
             Tile<TSrc> *max_buf,       // scratch: max values
             Tile<TSrc> *scaling_buf) { // scratch: scaling factors
  if constexpr (strcmp(Mode, "mxfp8") == 0) {
    pto::TQUANT<pto::QuantType::MXFP8>(dst, src, exp, max_buf, scaling_buf);
  }
}

// ============================================================================
// tdequant: INT8/INT16 → FP32 反量化 (A2/A3 only)
// ============================================================================

template <typename TSrc, typename TDst>
AICORE PTO_INLINE void
tdequant(Tile<TDst> &dst,
         Tile<TSrc> &src,
         Tile<TDst> &scale,
         Tile<TDst> &offset) {
  pto::TDEQUANT(dst, src, scale, offset);
}
```

### 3.8 FP8/FP4 类型支持

#### 3.8.1 启用 FP8 codegen

需要取消注释并扩展 `codegen_ascend_pto.cc` 中的 FP8 类型处理：

```cpp
// src/target/codegen_ascend_pto.cc — FP8 类型映射
// 取消注释并添加:
// float8_e4m3 → float8_e4m3_t (pto-isa)
// float8_e5m2 → float8_e5m2_t (pto-isa)
// float8_e8m0 → float8_e8m0_t (pto-isa)
// float4_e2m1x2 → float4_e2m1x2_t (pto-isa)
// float4_e1m2x2 → float4_e1m2x2_t (pto-isa)
```

#### 3.8.2 Python dtype 注册

```python
# tilelang/language/ascend.py — 新增 dtype 字符串映射
_MX_DTYPE_MAP = {
    "float8_e4m3": "float8_e4m3_t",
    "float8_e5m2": "float8_e5m2_t",
    "float8_e8m0": "float8_e8m0_t",
    "float4_e2m1x2": "float4_e2m1x2_t",
    "float4_e1m2x2": "float4_e1m2x2_t",
}
```

### 3.9 Scale Tile 管理策略（编译器自动）

Scale tile 管理对 DSL 用户**完全透明**，在两个层面实现：

1. **Python DSL 层**：用户传入 `scale_A` 和 `scale_B` 参数（已在 L1 级别分配好的 E8M0 tile），API 内部编码到 IR。

2. **Runtime Template 层**：`gemm_mx` 模板函数在 K 切片循环中，通过 `GetScaleAddr()` 从数据 tile 地址计算 scale tile 的硬件相对地址，自动绑定到当前 L0 切片对应的 scale 区间。

**Scale tile 生命周期**：
```
GM[Scale_A: E8M0, (M, ceil(K/32))]
    ↓ T.copy (用户负责 GM→L1 搬运，与 data tile 同层级)
L1[Scale_A_tile]
    ↓ GetScaleAddr (runtime template 自动处理)
L0[Sa_tile: (validM, kMXPerL0)]  ← 每个 K 切片 4 个 scale values (FP8) 或 8 个 (FP4)
```

### 3.10 CPU 仿真器支持

在 `src/tl_templates/pto/common.h` 中通过条件编译区分 NPU 和 CPU 路径：

```cpp
#ifdef PTO_PLATFORM_A5
  // NPU: 使用 pto::TMATMUL_MX 硬件指令
  pto::TMATMUL_MX(C, l0a, sa_tile, l0b, sb_tile);
#elif defined(PTO_PLATFORM_CPU)
  // CPU: 使用 pto::TMATMUL_MX CPU 仿真（cpu/TMatmul.hpp 已支持）
  pto::TMATMUL_MX(C, l0a, sa_tile, l0b, sb_tile);
#else
  static_assert(false, "TMATMUL_MX requires A5 or CPU platform");
#endif
```

---

## 4. 修改文件清单

### 4.1 Python 前端

| 文件 | 修改内容 |
|------|---------|
| `tilelang/language/ascend.py` | 扩展 `gemm_v0()` 增加 `quant_mode/scale_A/scale_B` 参数；新增 `gemm_mx()` 函数 |
| `tilelang/language/customize.py` | 新增 `npu_gemm_mx()` 低层 MMA API |
| `tilelang/language/ascend_tile.py` | 新增 `quant()` 和 `dequant()` tile 原语 |
| `tilelang/language/__init__.py` | 导出 `gemm_mx`、`quant`、`dequant` 新符号 |
| `tilelang/language/ast/ir.py` | 补充 `float8_e8m0`、`float4` 相关 dtype 定义 |

### 4.2 C++ IR

| 文件 | 修改内容 |
|------|---------|
| `src/op/ascend.h` | 声明 `ascend_gemm_mx()`、`ascend_mma_mx()`、`ascend_tquant()`、`ascend_tdequant()` |
| `src/op/ascend.cc` | 注册 4 个新 TIR builtin op |

### 4.3 Transform Pass

| 文件 | 修改内容 |
|------|---------|
| `src/transform/common/operation_config.h` | 新增 4 个 op 的 pipeline 配置 |

### 4.4 PTO Codegen

| 文件 | 修改内容 |
|------|---------|
| `src/target/codegen_ascend_pto.h` | 声明 `GemmMxCodegen()`、`MmaMxCodegen()`、`TQuantCodegen()`、`TDequantCodegen()` |
| `src/target/codegen_ascend_pto.cc` | 实现 4 个新 codegen 函数；启用 FP8/FP4 类型映射；`extractTemplateParamsEx` 扩展（支持 9+ 参数） |
| `src/tl_templates/pto/common.h` | 新增 `TileScaleA/B` 类型、`gemm_mx`、`mma_mx`、`tquant_mxfp8`、`tdequant` template |

### 4.5 示例与测试

| 文件 | 修改内容 |
|------|---------|
| `examples/gemm_mx/example_gemm_mxfp8.py` | MXFP8 GEMM 示例 |
| `examples/gemm_mx/example_gemm_mxfp4.py` | MXFP4 GEMM 示例 |
| `examples/gemm_mx/example_quant_matmul.py` | TQUANT + TMATMUL_MX 完整流水线示例 |
| `testing/python/language/test_tilelang_ascend_gemm_mx.py` | 单元测试 |

---

## 5. Tiling 策略约束

### 5.1 MXFP 特殊约束

| 约束 | 值 | 说明 |
|------|-----|------|
| K % 64 | == 0 | TMATMUL_MX 硬件要求 |
| FP4: K even | == true | FP4 pack 2 elements/byte |
| Scale block | 32 | 每 32 个 K 元素对应 1 个 E8M0 scale |
| L0 切片内的 scale 数 | kL0Size/32 = 4 (FP8) | 每个 L0 切片含 4 个 scale values |
| L0A 最小 | M ≥ 16, K ≥ 32 | 硬件分形限制 |
| L0B 最小 | K ≥ 32, N ≥ 16 | 硬件分形限制 |
| L0C 容量 | M×N×4 ≤ 128KB | float 累加器 |
| 累加器类型 | float 固定 | TMATMUL_MX 输出固定 float |

### 5.2 推荐 Block 配置

| 场景 | block_M | block_N | block_K | 说明 |
|------|---------|---------|---------|------|
| 小规模验证 | 64 | 64 | 64 | 最小配置，CPU 仿真可运行 |
| 典型生产 | 128 | 256 | 128 | 参考 pto-isa MXFP8 performance kernel |
| MXFP4 场景 | 256 | 256 | 256 | FP4 可增大 K 切片 |

---

## 6. 验证方案

### 6.1 Golden 函数

```python
def golden_mxfp8_gemm(A_fp8, B_fp8, scale_A, scale_B, M, N, K):
    """CPU 参考实现：手动 dequant + float matmul"""
    import numpy as np

    # Dequant A: FP8 × 2^(scale - bias) → float
    block_size = 32
    k_blocks = K // block_size
    A_dequant = np.zeros((M, K), dtype=np.float32)
    B_dequant = np.zeros((K, N), dtype=np.float32)

    for b in range(k_blocks):
        k_start = b * block_size
        k_end = k_start + block_size
        for i in range(M):
            s_a = 2.0 ** (scale_A[i, b].astype(np.int32) - 127)
            A_dequant[i, k_start:k_end] = A_fp8[i, k_start:k_end].astype(np.float32) * s_a
        for j in range(N):
            s_b = 2.0 ** (scale_B[b, j].astype(np.int32) - 127)
            B_dequant[k_start:k_end, j] = B_fp8[k_start:k_end, j].astype(np.float32) * s_b

    return A_dequant @ B_dequant
```

### 6.2 测试用例分级

| 级别 | 用例 | Shape (M,K,N) | dtype | 说明 |
|------|------|---------------|-------|------|
| L0 | 最小功能 | (64, 64, 64) | mxfp8 | CPU 仿真验证 |
| L1 | 典型配置 | (128, 128, 256) | mxfp8 | NPU 正确性 |
| L1 | MXFP4 | (128, 128, 128) | mxfp4 | FP4 格式验证 |
| L2 | 边界 K 切片 | (128, 192, 256) | mxfp8 | K 非 128 倍数 |
| L2 | 转置 A | (128, 128, 256) | mxfp8 | transpose_A=True |
| L3 | 大规模 | (6144, 6144, 6144) | mxfp8 | 性能验证 (32 核) |

### 6.3 测试命令

```bash
# CPU 仿真（无需 NPU 硬件）
python examples/gemm_mx/example_gemm_mxfp8.py --target cpu_sim

# NPU A5 硬件
source set_env.sh
python examples/gemm_mx/example_gemm_mxfp8.py --target pto

# 单元测试
pytest testing/python/language/test_tilelang_ascend_gemm_mx.py -v
```

---

## 7. 风险点与注意事项

### 7.1 已知约束

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| TMATMUL_MX 仅 A5 | A2/A3 无法使用 | 编译期检查目标架构，A2/A3 fallback 到 INT8 GEMM + 软件 scaling |
| FP8 codegen 注释 | FP8 data tile 无法编译 | 需先启用 FP8 类型映射（取消注释 + 扩展） |
| Scale 地址绑定 | Scale tile 必须与 data tile 在同 L1 相邻 | `GetScaleAddr` 硬件保证，runtime template 封装 |
| K % 64 约束 | 非法 K 导致硬件异常 | Python API 层 assert K % 64 == 0 |
| FP4 pack 2/byte | 偏移计算需除以 2 | runtime template 中按 `sizeof(T1)` 处理 |
| `extractTemplateParamsEx` | 需支持 9+ 参数（含 quant_mode 字符串） | 扩展现有解析函数 |

### 7.2 实现顺序

1. ✅ 设计文档（本文档）
2. ⬜ **Step 1**: FP8/E8M0/FP4 dtype 类型启用（codegen 取消注释 + Python dtype 注册）
3. ⬜ **Step 2**: C++ IR op 注册（`ascend_gemm_mx`, `ascend_mma_mx`, `ascend_tquant`, `ascend_tdequant`）
4. ⬜ **Step 3**: PTO runtime template（`gemm_mx`, `mma_mx`, `tquant_mxfp8`, `tdequant`）
5. ⬜ **Step 4**: PTO codegen（`GemmMxCodegen`, `MmaMxCodegen`, `TQuantCodegen`, `TDequantCodegen`）
6. ⬜ **Step 5**: Python frontend（`gemm_v0` 扩展 + `gemm_mx` + `quant/dequant`）
7. ⬜ **Step 6**: Pipeline config 注册
8. ⬜ **Step 7**: 示例 + 测试

### 7.3 与现有代码的兼容性

- `T.gemm_v0()` 不传 `quant_mode` 时行为**完全不变**，走原有 `gemm_v0` 路径
- `T.mma()` 不传 scale 参数时行为**完全不变**
- PTO template `gemm_v0<...>` 和 `mma<...>` 签名不变，新增 `gemm_mx<...>` / `mma_mx<...>` 不冲突
- `extractTemplateParams` 现有 7 参数解析不变，新增 `extractTemplateParamsEx` 支持更多参数
