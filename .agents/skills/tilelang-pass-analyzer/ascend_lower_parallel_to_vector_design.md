# ascend_lower_parallel_to_vector.cc 设计文档

## 1. 背景与目标

### 1.1 需求来源

在 TileLang 的编程模型中，T.Parallel 是用于表达 tile 内元素向量化计算 的核心原语。

在 Ascend 场景下，典型的 Vector 计算模式通常是：
- 将大张量划分为若干 tile
- 每个 tile 在片上缓冲区（UB）中完成 Load → Compute → Store
- 在 Compute 阶段，利用 AscendC 的 vector 能力对 tile 内元素进行计算

T.Parallel 就是用于描述这一 Compute 阶段的 tile内元素向量化计算 的结构化语义。而 ascend_lower_parallel_to_vector.cc 则是实现 T.Parallel 功能的 pass。

### 1.2 业务价值

在 IR 层以"并行循环"的形式描述数据并行，而不直接暴露底层硬件指令细节，极大的提升用户的算子编程体验。

### 1.3 技术目标

在 TileLang 中，用一种统一的 IR 结构（for ... in T.Parallel(...)）表达一元/二元操作的元素向量化运算，使前端编写的 kernel 更简洁、更易理解。

#### 1.3.1 复用 TileLang 主仓计算原语

鼓励在 T.Parallel 中使用符号算子（如 T.exp、T.log、T.sqrt、T.max 等），而不是直接写 vector 指令，从而：
- 保持算子表达简洁，与主仓 IR 对齐
- 实现不同后端之间的最大可移植性

#### 1.3.2 与 AscendC vector 能力协同并兼容

对于 AscendC 特有的 vector 能力：
- 新增的 T.tile.xxx 接口，内容为原 ascend.py 中的 vector 操作（集中放在 ascend_tile.py 中），保留 TileLang-Ascend 中 vector 原语能力
- 高层算子既可以纯用 T.Parallel + T.exp 等符号 API，也可以兼容原 vector 类型 tile 原语

---

## 2. 整体设计

### 2.1 框架架构图

```
                          ┌─────────────────────────────────────┐
                          │         前端 DSL (T.Parallel)        │
                          │   for i, j in T.Parallel(M, N):     │
                          │       C[i, j] = A[i, j] + B[i, j]   │
                          └─────────────────────────────────────┘
                                        │
                                        ▼
                          ┌─────────────────────────────────────┐
                          │        IR 层 (TIR ForNode)           │
                          │   ForNode(kind=kParallel, extent=N) │
                          │       BufferStore(C, A + B)         │
                          └─────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   ▼                   │
                    │    ┌─────────────────────────────┐    │
                    │    │  AscendLowerParallelToVector │    │
                    │    │       (本 Pass 核心)          │    │
                    │    └─────────────────────────────┘    │
                    │                   │                   │
                    ▼                   ▼                   ▼
        ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
        │  表达式分解模块  │ │  广播处理模块   │ │  GM写入处理模块 │
        │ DecomposeExpr() │ │ CanBroadcast()  │ │ AutoCopy GM→UB │
        └─────────────────┘ └─────────────────┘ └─────────────────┘
                    │                   │                   │
                    └───────────────────┼───────────────────┘
                                        ▼
                          ┌─────────────────────────────────────┐
                          │      后端 IR (AscendC Vector Call)   │
                          │   Call(op=ascend_add,               │
                          │        args=[out_ptr, in1_ptr,      │
                          │                 in2_ptr, count])    │
                          └─────────────────────────────────────┘
                                        │
                                        ▼
                          ┌─────────────────────────────────────┐
                          │         AscendC 代码生成             │
                          │   AscendC::Add(out, in1, in2, count)│
                          └─────────────────────────────────────┘
```

### 2.2 Pass 定位与触发

| 维度 | 说明 |
|------|------|
| 所属阶段 | Phase 1: LowerAndLegalize（IR Lowering 与合法化） |
| 执行时机 | 在 `Simplify()` 之后、`LayoutInference()` 之前 |
| 平台特性 | Ascend 专用 Pass |
| 启用方式 | 默认启用，无配置开关 |

### 2.3 核心功能模块

```
┌──────────────────────────────────────────────────────────────────┐
│                    AscendLowerParallelToVector                    │
├──────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐  │
│  │ 循环结构识别   │  │ 向量化计划检测 │  │ 表达式分解       │  │
│  │ VisitStmt_()  │→│ DetectVectorPlan│→│ DecomposeExpression│  │
│  └────────────────┘  └────────────────┘  └────────────────────┘  │
│          │                    │                    │              │
│          ▼                    ▼                    ▼              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐  │
│  │ 1D/2D/3D 支持 │  │ 广播判断       │  │ 临时Buffer分配   │  │
│  │ Serial嵌套    │  │ CanBroadcast() │  │ CreateTempBuffer │  │
│  └────────────────┘  └────────────────┘  └────────────────────┘  │
│          │                    │                    │              │
│          └────────────────────┼────────────────────┘              │
│                               ▼                                   │
│                    ┌────────────────────┐                         │
│                    │ AscendC Call 生成  │                         │
│                    │ GenerateVectorCall │                         │
│                    └────────────────────┘                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. 详细设计

### 3.1 数据结构设计

#### 3.1.1 主类结构

```cpp
class AscendLowerParallelToVector : public arith::IRMutatorWithAnalyzer {
private:
    // 循环变量追踪
    const VarNode *vector_dim_var_ = nullptr;    // 内层向量化维度变量
    const VarNode *outer_dim_var_ = nullptr;     // 外层维度变量（2D向量化）
    bool is_2d_vectorizing_ = false;             // 是否启用2D向量化
    
    // 临时Buffer管理
    int temp_buffer_id_ = 0;                     // Buffer命名计数器
    std::vector<Buffer> temp_buffers_;           // 当前Block的临时Buffer列表
    
    // 维度信息
    int64_t vector_dim_extent_{0};               // 向量化维度范围
    int64_t outer_dim_extent_{0};                // 外层维度范围
};
```

#### 3.1.2 操作映射表

**一元操作映射**（TIR → AscendC）：

| TIR Op | AscendC Op | 说明 |
|--------|-----------|------|
| `tir.exp` | `tl::ascend_exp()` | 指数函数 |
| `tir.log` | `tl::ascend_ln()` | 自然对数 |
| `tir.sqrt` | `tl::ascend_sqrt()` | 平方根 |
| `tir.rsqrt` | `tl::ascend_rsqrt()` | 平方根倒数 |
| `tir.fabs` | `tl::ascend_abs()` | 绝对值 |
| `max(x, 0)` | `tl::ascend_relu()` | ReLU |
| `tir.bitwise_not` | `tl::ascend_bitwise_not()` | 按位取反 |

**二元操作映射**：

| TIR Op | AscendC Op | Scalar版本 | 说明 |
|--------|-----------|-----------|------|
| `Add` | `tl::ascend_add()` | `ascend_adds()` | 加法 |
| `Sub` | `tl::ascend_sub()` | `ascend_subs()` | 减法 |
| `Mul` | `tl::ascend_mul()` | `ascend_muls()` | 乘法 |
| `Div` | `tl::ascend_div()` | `ascend_divs()` | 除法 |
| `Min` | `tl::ascend_min()` | - | 最小值 |
| `Max` | `tl::ascend_max()` | - | 最大值 |
| `bitwise_and` | `tl::ascend_bitwise_and()` | - | 按位与 |
| `bitwise_or` | `tl::ascend_bitwise_or()` | - | 按位或 |
| `shift_left` | `tl::ascend_bitwise_lshift()` | - | 左移 |
| `shift_right` | `tl::ascend_bitwise_rshift()` | - | 右移 |

#### 3.1.3 VectorPlan 结构

```cpp
struct VectorPlan {
    int64_t inner_vec_len{0};        // 内层向量化长度（元素数）
    int64_t outer_extent{0};         // 外层维度范围
    const VarNode *outer_index_var{nullptr};  // 外层索引变量
    bool is_2d_vectorizable{false};  // 是否可进行2D向量化
};
```

#### 3.1.4 BroadcastInfo 结构

```cpp
struct BroadcastInfo {
    const BufferLoadNode *load;      // 原始1D Buffer加载
    Buffer broadcast_buffer;         // 广播后的2D Buffer
    Buffer workspace_buffer;         // 广播工作空间Buffer
    int64_t broadcast_dim;           // 广播维度（0或1）
    int64_t outer_extent;            // 外层范围
    int64_t inner_vec_len;           // 内层向量化长度
};
```

### 3.2 核心逻辑

#### 3.2.1 Pass 入口

```python
# 伪代码：Pass入口流程

输入: PrimFunc f
处理流程:
    1. 创建 analyzer → arith::Analyzer()
    2. 创建 substituter → AscendLowerParallelToVector(analyzer)
    3. 遍历并变换 body → substituter.VisitStmt(f->body)
    4. 更新 func body → fptr->body = new_body
输出: 变换后的 PrimFunc
```

#### 3.2.2 循环结构识别（VisitStmt_）

```python
# 伪代码：循环结构识别

输入: ForNode op
处理流程:
    if op->kind == kParallel:
        # Case 1: parallel → (store | seq)
        if body is BufferStore or SeqStmt:
            尝试向量化 → TryVectorizeStoreSeq()
        
        # Case 2: parallel → parallel → (store | seq) [2D向量化]
        if inner_for->kind == kParallel:
            if third_for exists: ERROR "不支持3D及以上"
            尝试2D向量化 → TryVectorizeStoreSeq(is_2d=True)
    
    if op->kind == kSerial:
        # Case 3: serial → parallel → (store | seq)
        if inner_for->kind == kParallel:
            尝试向量化 → TryVectorizeStoreSeq(has_outer_serial=True)
            若成功，保留外层Serial循环
输出: 向量化后的Stmt 或 原循环
```

#### 3.2.3 向量化计划检测（DetectVectorPlan）

```python
# 伪代码：向量化计划检测

输入: BufferStoreNode *store, element_count
处理流程:
    # 1D Case
    if output_buffer->shape.size() == 1:
        if indices[0] contains vector_dim_var:
            plan.inner_vec_len = element_count
            plan.outer_extent = 1
            plan.is_2d_vectorizable = false
    
    # 2D Case
    if output_buffer->shape.size() == 2:
        if indices[1] contains vector_dim_var:
            inner_vec_len = vector_dim_extent_ or buffer->shape[1]
            plan.inner_vec_len = inner_vec_len
            plan.outer_extent = element_count / inner_vec_len
            plan.is_2d_vectorizable = (outer_dim_var in indices[0])
    
    # 3D Case (double buffer)
    if output_buffer->shape.size() == 3:
        类似2D处理，使用indices[1]和indices[2]
输出: VectorPlan (inner_vec_len, outer_extent, is_2d_vectorizable)
```

#### 3.2.4 表达式分解（DecomposeExpression）

```python
# 伪代码：表达式分解

输入: PrimExpr expr, output_buffer, element_count
处理流程:
    # Case 1: L0C→GM (Cast + BufferLoad from L0C)
    if expr is Cast of L0C BufferLoad:
        生成 ascend_copy(L0C → GM)
    
    # Case 2: 简单数据搬运 (BufferLoad)
    if expr is BufferLoad:
        生成 ascend_copy(input → output)
    
    # Case 3: 一元操作
    if IsUnaryOp(expr):
        生成 GenerateUnaryVectorCall(op, out, in, count)
    
    # Case 4: 二元操作
    if IsBinaryOp(expr):
        # Simple-Simple: 两边都是BufferLoad或Scalar
        if left_simple and right_simple:
            HandleSimpleCase() → 直接生成vector call
        
        # Simple-Complex: 一边简单，一边复杂
        if left_simple and right_complex:
            CreateTempBuffer(rhs_tmp)
            DecomposeExpression(rhs → rhs_tmp)
            GenerateBinaryVectorCall(lhs, rhs_tmp → out)
        
        # Complex-Complex: 两边都复杂
        if left_complex and right_complex:
            CreateTempBuffer(lhs_tmp, rhs_tmp)
            DecomposeExpression(lhs → lhs_tmp)
            DecomposeExpression(rhs → rhs_tmp)
            GenerateBinaryVectorCall(lhs_tmp, rhs_tmp → out)
输出: Array<Stmt> statements (vector calls序列)
```

#### 3.2.5 广播处理

```python
# 伪代码：广播处理流程

输入: 1D BufferLoad in 2D context
处理流程:
    1. 检测可广播性 → CanBroadcast()
       - 仅支持1D buffer
       - index必须是简单变量（无offset）
       - 判断广播维度
    
    2. 创建广播Buffer → CreateBroadcastBuffer()
       - 形状: [outer_extent, inner_vec_len]
    
    3. 创建工作空间Buffer → CreateBroadcastWorkspaceBuffer()
       - 类型: uint8
       - 大小: 2 * total_elements
    
    4. 生成广播调用 → GenerateBroadcastStmt()
       - 调用 tl.ascend_broadcast()
       - 参数: dst, src_2d_view, workspace, dims, shapes
输出: 广播后的2D Buffer及相关Stmt
```

#### 3.2.6 拷贝处理

```python
# 伪代码：拷贝处理（含GM写入、L0C输出、简单搬运）

输入: BufferStore 或 表达式中的BufferLoad
处理流程:
    # Case 1: GM直接写入（输出目标为GM）
    if IsGlobalMemoryBuffer(output_buffer) and not is_l0c_input:
        # 创建临时UB Buffer存放计算结果
        temp_ub = CreateTempBufferLike(output, total_elements, inner_vec_len)
        
        # 将计算写入临时UB
        DecomposeExpression(expr → temp_ub)
        
        # 生成 UB→GM 复制
        GenerateAscendCopy(temp_ub → output_gm)
    
    # Case 2: L0C→GM（矩阵计算输出）
    if expr is Cast of L0C BufferLoad:
        生成 ascend_copy(L0C → GM)
    
    # Case 3: 简单数据搬运（纯BufferLoad表达式）
    if expr is BufferLoad:
        生成 ascend_copy(input_buffer → output_buffer)
    
    # Case 4: UB/L1直接写入
    if output in UB/L1:
        直接写入目标Buffer（无需额外copy）
输出: 计算Stmt + Copy Stmt (若需要)
```

### 3.3 IR 变换示例

**输入伪 IR：**
```tir
// 前端 DSL: for i, j in T.Parallel(64, 128):
//               C[i, j] = A[i, j] + B[i, j]

for i in parallel(64):
    for j in parallel(128):
        C_ub[i, j] = A_ub[i, j] + B_ub[i, j]  ← 变换点
```

**输出伪 IR：**
```tir
// 后端 IR: 直接生成 AscendC Add 调用
evaluate(call tl.ascend_add(
    access_ptr(C_ub, dtype, offset=0, extent=8192, mask=2),
    access_ptr(A_ub, dtype, offset=0, extent=8192, mask=1),
    access_ptr(B_ub, dtype, offset=0, extent=8192, mask=1),
    8192  // element_count = 64 * 128
))
```

**变换要点：**
- 双层 parallel 循环合并为单个 vector call
- 循环变量 `i, j` 替换为 offset=0（向量化起始点）
- extent = 循环范围乘积（总元素数）
- mask: 1=read, 2=write

### 3.4 代码位置

| 组件 | 路径 | 核心元素 |
|------|------|---------|
| C++ 实现 | `src/transform/ascend_lower_parallel_to_vector.cc` | `AscendLowerParallelToVector::Substitute()` |
| Python API | `tilelang/transform/__init__.py` | `tl.transform.AscendLowerParallelToVector` |
| AscendC Op定义 | `src/op/ascend.h` | `tl::ascend_add()`, `tl::ascend_exp()` 等 |

---

## 4. 验证章节

### 4.1 测试文件对应关系

| 测试文件 | 测试类型 | 关键测试场景 |
|---------|---------|-------------|
| `test_tilelang_ascend_language_parallel.py` | 基础功能 | 二元操作、一元操作、复合操作、1D操作、广播操作 |
| `test_tilelang_ascend_language_parallel_auto_copy.py` | GM写入 | 自动UB→GM复制、不同dtype、复杂表达式、L0C输出 |
| `test_tilelang_ascend_language_parallel_complex.py` | 复杂场景 | 多buffer赋值、链式操作、嵌套计算、多临时buffer |
| `test_tilelang_ascend_language_parallel_discrete.py` | 离散索引 | gather操作、离散索引访问 |

### 4.2 测试用例设计

#### 4.2.1 基础二元操作测试

| 测试用例 | IR 特征 | 验证点 |
|---------|---------|--------|
| `test_add_operation` | `a + b` | Add→ascend_add映射 |
| `test_sub_operation` | `a - b` | Sub→ascend_sub映射 |
| `test_mul_operation` | `a * b` | Mul→ascend_mul映射 |
| `test_div_operation` | `a / b` | Div→ascend_div映射 |
| `test_min_operation` | `T.min(a, b)` | Min→ascend_min映射 |
| `test_max_operation` | `T.max(a, b)` | Max→ascend_max映射 |
| `test_and_operation` | `a & b` | BitwiseAnd→ascend_bitwise_and |
| `test_or_operation` | `a | b` | BitwiseOr→ascend_bitwise_or |

#### 4.2.2 一元操作测试

| 测试用例 | IR 特征 | 验证点 |
|---------|---------|--------|
| `test_abs_operation` | `T.abs(a)` | fabs→ascend_abs |
| `test_exp_operation` | `T.exp(a)` | tir.exp→ascend_exp |
| `test_log_operation` | `T.log(a)` | tir.log→ascend_ln |
| `test_sqrt_operation` | `T.sqrt(a)` | tir.sqrt→ascend_sqrt |
| `test_rsqrt_operation` | `T.rsqrt(a)` | tir.rsqrt→ascend_rsqrt |
| `test_relu_operation` | `T.max(a, 0)` | max(x,0)→ascend_relu |
| `test_not_operation` | `~a` | bitwise_not→ascend_bitwise_not |
| `test_shiftleft_operation` | `a << scalar` | shift_left→ascend_bitwise_lshift |
| `test_shiftright_operation` | `a >> scalar` | shift_right→ascend_bitwise_rshift |

#### 4.2.3 复合表达式测试

| 测试用例 | 表达式 | 验证点 |
|---------|--------|--------|
| `test_fused_mul_add` | `a * b + a` | 复合表达式分解、临时buffer |
| `test_fused_add_mul` | `a * (b + a)` | 右侧复杂→临时buffer |
| `test_scalar_operation` | `a + 1.0` | Scalar→ascend_adds |

#### 4.2.4 广播语义测试

| 测试用例 | 广播模式 | 实现方式 | 验证点 |
|---------|---------|---------|--------|
| `test_column_parallel_buffer_scalar_mul` | `b[i]` broadcast to `[M, N]` | for循环索引语义 | 行向量广播效果 |
| `test_row_parallel_buffer_scalar_mul` | `b[j]` broadcast to `[M, N]` | for循环索引语义 | 列向量广播效果 |
| `test_column_parallel_buffer_unmatch` | `b[j] + 5` broadcast | for循环索引语义 | 列向量广播 + scalar |
| `test_row_parallel_buffer_unmatch` | `b[i] + 5` broadcast | for循环索引语义 | 行向量广播 + scalar |

#### 4.2.5 GM 直接写入测试

| 测试用例 | 场景 | 验证点 |
|---------|------|--------|
| `test_parallel_auto_copy` | UB计算→GM输出 | 自动创建temp_ub + ascend_copy |
| `test_parallel_auto_copy_complex` | 复杂表达式→GM | 多步计算后自动复制 |
| `test_matmul` | L0C→GM | Cast(L0C)直接输出 |

#### 4.2.6 复杂场景测试

| 测试用例 | 场景 | 验证点 |
|---------|------|--------|
| `test_complex_dual_assignment` | 两个Buffer并行赋值 | SeqStmt处理 |
| `test_complex_chained_operations` | k1=a+b; k2=k1*c | 中间结果复用 |
| `test_complex_triple_assignments` | k1=a+b; k2=c*d; k3=k1+k2 | 多路径依赖 |
| `test_complex_deep_expression` | 4层嵌套表达式 | 多临时buffer管理 |
| `test_complex_nested_temp_buffer` | `a*a+a-b` | Complex-Complex分解 |

#### 4.2.7 离散索引测试

| 测试用例 | 索引模式 | 验证点 |
|---------|---------|--------|
| `test_parallel_discrete_mat_mat` | `a[idx[i], j]` | gather离散访问 |
| `test_parallel_discrete_mat_row` | `a[idx[i], j] + b[i]` | gather + row广播 |
| `test_parallel_discrete_mat_col` | `a[idx[i], j] + b[j]` | gather + col广播 |

### 4.3 IR 校验单元测试

针对 Pass 特点，可构造 IR 校验测试验证：
- 二元操作 IR 变换正确性
- 复杂表达式分解（Complex-Complex 场景）
- 广播 IR 生成（`tl.ascend_broadcast` 调用）
- GM 自动复制（temp_ub allocation + `tl.ascend_copy`）

---

## 5. 附录

### 5.1 支持的循环模式

| 模式 | IR 结构 | 向量化方式 |
|------|---------|-----------|
| 1D Parallel | `parallel → store` | 单次 vector call |
| 2D Parallel | `parallel → parallel → store` | 单次 2D vector call |
| Serial嵌套Parallel | `serial → parallel → store` | 保留外层serial，内层向量化 |

### 5.2 Buffer 类型判断

| Storage Scope | 判断函数 | 说明 |
|--------------|---------|------|
| `global` / 空 | `IsGlobalMemoryBuffer()` | GM（全局内存） |
| `shared` | `IsUnifiedBuffer()` | UB（片上缓冲） |
| `shared.dyn` | `IsL1Buffer()` | L1（Cube核缓存） |
| `wmma.accumulator` | `IsL0CBuffer()` | L0C（矩阵输出寄存器） |

### 5.3 关键限制

1. **不支持3D及以上Parallel嵌套**：最多支持2层 parallel 循环
2. **不支持 if-else 分支**：SIMD架构不支持离散处理，T.Parallel 内不能包含条件分支语句
3. **离散索引 fallback**：非简单变量索引（如 `a[idx[i]]`）退回到 serial 循环
4. **广播限制**：仅支持1D→2D广播，且索引必须是简单变量
5. **常量要求**：element_count 必须可简化为 IntImm（常量）

### 5.4 与其他 Pass 的关系

```
# Phase 1: LowerAndLegalize
Simplify → AscendLowerParallelToVector → LayoutInference → LowerTileOp → LegalizeVectorizedLoop
      │              │                      │                  │                │
      │              │                      │                  │                │
   IR简化      Parallel向量化        Layout推断       Tile操作Lowering   Loop合法化

# Phase 2: OptimizeForTarget
AscendLowerOpaqueBlock → AscendStorageRewrite → AscendMemoryPlanning → AscendSyncInsert
          │                    │                     │                    │
          │                    │                     │                    │
     Opaque Block Lowering   存储重写           内存地址规划          同步点插入
```

> 注：`AscendLowerParallelToVector` 属于 Phase 1，在 IR Lowering 阶段执行