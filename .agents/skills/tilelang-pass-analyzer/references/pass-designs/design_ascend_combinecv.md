# 1. 背景与目标

## 需求来源

Ascend NPU（华为昇腾处理器）采用独特的异构计算架构，包含两种不同类型的计算单元：
- **Cube核心**：专注于矩阵计算（如GEMM、矩阵乘法等），对应硬件上的AI Core
- **Vector核心**：专注于向量计算（如逐元素操作、广播、激活函数等），对应硬件上的AI Vector

在TileLang编译框架中，需要将TIR（Tensor IR）表示的计算图转换为可在Ascend NPU上高效执行的代码。

本需求未开发前，用户需要手动编写Cube和Vector核心的代码，并且需要深入理解硬件架构，增加了开发者的负担，编码如下所示：

``` python
with scope("C"):
    process on cube
with scope("V"):
    process on vector
```

本需求开发后，前端不显示规定scope，后端自动识别并分离Cube和Vector类型的计算操作，最后达到的前端书写效果为：

``` python
process on cube
process on vector
process on cube
```

## 业务价值

该功能解决了以下核心痛点：
1. **编程复杂性**：手动编写cube和vector分离的代码需要深入理解硬件架构，增加了开发者负担
2. **同步难题**：cube和vector核心之间的数据依赖和同步关系复杂，容易出错

通过自动化的代码分离和同步机制，可以：
- 降低开发者编写Ascend算子的门槛
- 确保数据同步的正确性和高效性

## 技术目标

1. **功能指标**：
   - 自动识别并分离cube和vector类型的计算操作
   - 自动生成跨核心同步代码，确保数据依赖正确

2. **性能指标**：
   - 跨核心同步开销最小化，仅在必要时插入同步
   - 同步点选择最优化，减少等待时间

---

# 2. 整体设计

## 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    CombineCV Pass                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  输入: TIR PrimFunc                                          │
│         ↓                                                    │
│  ┌─────────────────────────────────────────────┐            │
│  │  1. Location Map 构建                       │            │
│  │     - 分析buffer的storage scope             │            │
│  │     - 建立 Var -> Scope 映射                │            │
│  └─────────────────────────────────────────────┘            │
│         ↓                                                    │
│  ┌─────────────────────────────────────────────┐            │
│  │  2. CVCombineEmitter (双路径生成)           │            │
│  │     ┌─────────────┐  ┌─────────────┐       │            │
│  │     │ CubeEmitter │  │ VecEmitter  │       │            │
│  │     │ (is_aiv=F)  │  │ (is_aiv=T)  │       │            │
│  │     └─────────────┘  └─────────────┘       │            │
│  │         ↓                ↓                  │            │
│  │     cube_code       vec_code                │            │
│  └─────────────────────────────────────────────┘            │
│         ↓                                                    │
│  ┌─────────────────────────────────────────────┐            │
│  │  3. AutoInsertCrossCoreSync (可选)          │            │
│  │     - CrossCoreSyncCollector                │            │
│  │     - 同步点匹配与分配                       │            │
│  │     - CrossCoreSyncInserter                 │            │
│  └─────────────────────────────────────────────┘            │
│         ↓                                                    │
│  ┌─────────────────────────────────────────────┐            │
│  │  4. 最终组合                                │            │
│  │     cube_body: AttrStmt(resource_scope=0)   │            │
│  │     vec_body:  AttrStmt(resource_scope=1)   │            │
│  │     combine_body: SeqStmt{cube, vec}        │            │
│  └─────────────────────────────────────────────┘            │
│         ↓                                                    │
│  输出: 分离后的 PrimFunc                                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 跨核心同步流程

```
Cube Core                          Vector Core
    │                                   │
    │  copy_gm_to_l1 (MTE2)             │
    │         ↓                         │
    │  GEMM computation                 │
    │         ↓                         │
    │  copy_l0c_to_gm (FIX)             │
    │         ↓                         │
    │  SetFlag(sync_id) ─────────────────┼──→ WaitFlag(sync_id)
    │         │                         │         ↓
    │         │                         │  copy_gm_to_ub (MTE2)
    │         │                         │         ↓
    │         │                         │  Vector computation
    │         │                         │         ↓
    │         │                         │  copy_ub_to_gm (MTE3)
    │         ↓                         │         ↓
    │  (继续下一个迭代)                  │  SetFlag(sync_id+1)
    │                                   │         ↓
    │←── WaitFlag(sync_id+1) ───────────┼─────────┘
    │                                   │
```

### 同步点匹配逻辑

```
workspace_1:
  cube: [write_order_0, read_order_1, write_order_2]
  vec:  [read_order_0, write_order_1, read_order_2]
  
  匹配规则:
    cube[write] ↔ vec[read]  (write → read 依赖)
    cube[read]  ↔ vec[write] (read ← write 依赖)
    
  sync_flag_id 分配:
    pair_0: sync_flag_id = 0
    pair_1: sync_flag_id = 1
    pair_2: sync_flag_id = 2
```

---

# 3. 详细设计

## 数据结构设计

### CrossCoreSyncPoint

```cpp
struct CrossCoreSyncPoint {
  int scope;        // 0: cube, 1: vec - 核心类型标识
  int order;        // 执行顺序编号（用于匹配同步点）
  int sync_flag_id; // 跨核心同步标志ID
  bool is_write;    // 是否为写操作（true: write到workspace，false: read从workspace）
  std::string workspace_name; // workspace缓冲区名称
  std::string pipe; // 管道类型：MTE2(搬运入), MTE3(搬出), FIX(输出)
  
  const EvaluateNode *node; // 原始IR节点
  
  // 同步语句挂载位置
  std::optional<const ForNode *> target_for_node;
  
  // 父循环节点列表（从外到内）
  std::vector<const ForNode *> parent_for_nodes;
  
  // Cross interval支持（减少同步频率）
  int cross_interval = 1;
  const ForNode *stage_loop = nullptr;
};
```

**字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| scope | int | 核心类型：0=cube, 1=vector |
| order | int | 在该核心内的执行顺序 |
| sync_flag_id | int | 同步标志ID，匹配的cube/vec对使用相同ID |
| is_write | bool | true=写操作(需要SetFlag), false=读操作(需要WaitFlag) |
| workspace_name | string | workspace缓冲区标识，用于匹配同步对 |
| pipe | string | 管道类型，影响SetFlag的具体行为 |
| target_for_node | optional<ForNode*> | 同步语句应挂载的循环层 |
| parent_for_nodes | vector<ForNode*> | IR节点所在的所有父循环 |
| cross_interval | int | 跨度间隔，用于减少同步频率 |

### Stmt与CV分类映射关系表

```cpp
std::unordered_map<std::string, std::string> callnodeMapPos_ = {
    // Cube核心操作
    {"copy_gm_to_l1", "cube"},      // GM → L1 搬运
    {"gemm_v0", "cube"},            // 矩阵乘法
    {"copy_l1_to_l0a", "cube"},     // L1 → L0A
    {"copy_l1_to_l0b", "cube"},     // L1 → L0B
    {"copy_l0c_to_gm", "cube"},     // L0C → GM 输出
    
    // Vector核心操作
    {"copy_gm_to_ub", "vec"},       // GM → UB 搬运
    {"copy_ub_to_gm", "vec"},       // UB → GM 输出
    {"copy_ub_to_ub", "vec"},       // UB内部搬运
    
    // Buffer scope映射
    {"wmma.matrix_a", "cube"},      // WMMA矩阵A缓冲区
    {"wmma.matrix_b", "cube"},      // WMMA矩阵B缓冲区
    {"wmma.accumulator", "cube"},   // WMMA累加器
    {"shared.dyn", "cube"},         // 动态共享内存(cube)
    {"shared", "vec"}               // 共享内存(vector)
};
```

### Stmt与Pipeline的读写映射关系表

```cpp
std::unordered_map<std::string, std::pair<bool, std::string>> GM_COPY_CFG_INFOS = {
    {"copy_gm_to_l1", {false, "MTE2"}},    // 读操作，MTE2管道
    {"copy_l0c_to_gm", {true, "FIX"}},     // 写操作，FIX管道
    {"copy_gm_to_ub", {false, "MTE2"}},    // 读操作，MTE2管道
    {"copy_ub_to_gm", {true, "MTE3"}}      // 写操作，MTE3管道
};
```

## 核心逻辑

### 1. Location Map构建

后序遍历IR，收集所有buffer定义及其storage scope，建立Var → Scope映射表。

### 2. CVCombineEmitter - 双路径代码生成

对同一个IR生成两个代码路径：
- **CubeEmitter**：保留Cube核心操作（矩阵计算、L1/L0相关），过滤Vector操作
- **VecEmitter**：保留Vector核心操作（UB相关、shared scope），过滤Cube操作

判断逻辑：
1. **API名称匹配**：根据预定义的API分类表判断属于cube还是vec
2. **Buffer scope检查**：检查buffer的storage scope决定是否保留
3. **状态开关控制**：通过`current_proccess_switch`控制是否保留后续语句

### 3. CrossCoreSyncCollector - 同步点收集

遍历IR，收集所有涉及workspace的GM拷贝操作，记录：
- 核心类型（cube/vec）
- 执行顺序
- 读写方向和管道类型
- workspace名称
- 父循环链

### 4. AutoInsertCrossCoreSync - 同步点匹配与分配

核心流程：
1. **收集同步点**：分别收集cube和vec的同步点
2. **按workspace分组**：将同步点按workspace名称分组
3. **匹配同步对**：cube的write与vec的read匹配，分配统一的sync_flag_id
4. **寻找最优挂载位置**：调用FindTargetLoopDepth确定同步语句挂载的循环层
5. **插入同步语句**：调用CrossCoreSyncInserter插入SetFlag/WaitFlag

### 5. FindTargetLoopDepth - 最优同步位置算法

**目标**：找到两个核心循环次数相等的最大深度，减少同步频率。

**算法**：
- 双指针遍历cube和vec的循环链
- 计算累计循环次数
- 找到次数相等时的最大深度作为挂载点
- 跳过迭代次数为1的循环和非const的共享循环

### 6. CrossCoreSyncInserter - 同步语句插入

插入逻辑：
- **写者**：操作后SetFlag（通知读者数据已准备好）
- **读者**：操作前WaitFlag（等待写者通知）
- **条件同步**：当cross_interval > 1时，根据循环变量条件判断是否需要同步

### 7. 最终组合

对tilelang_root block：
1. 双路径生成：分别生成cube_code和vec_code
2. 自动同步：插入跨核心同步（可选）
3. 添加标记：resource_scope=0（cube）和resource_scope=1（vec）
4. 组合输出：SeqStmt(cube_body, vec_body)

---

# 4. 验证章节

## 接口和算子测试例

### Pass配置测试

```cpp
// UT测试例：验证Pass配置选项
TEST(CombineCV, ConfigOptions) {
    // 测试默认配置
    PassContext ctx = PassContext::Create();
    EXPECT_FALSE(ctx->GetConfig<Bool>("tl.ascend_auto_cv_combine", Bool(false)).value());
    
    // 测试开启配置
    ctx = PassContext::Create(
        ConfigContext::Create()
        ->AddConfig("tl.ascend_auto_cv_combine", Bool(true))
    );
    EXPECT_TRUE(ctx->GetConfig<Bool>("tl.ascend_auto_cv_combine", Bool(false)).value());
    
    // 测试跨核心同步配置
    ctx = PassContext::Create(
        ConfigContext::Create()
        ->AddConfig("tl.ascend_auto_cv_combine", Bool(true))
        ->AddConfig("tl.ascend_auto_cross_core_sync", Bool(true))
    );
    EXPECT_TRUE(ctx->GetConfig<Bool>("tl.ascend_auto_cross_core_sync", Bool(false)).value());
}
```

### CV分离测试

```cpp
// UT测试例：验证Cube/Vector分离
TEST(CombineCV, BasicSeparation) {
    // 构造简单GEMM + Elementwise的IR
    PrimFunc func = ConstructGEMMPlusElementwiseIR();
    
    // 执行CombineCV Pass
    auto ctx = PassContext::Create(
        ConfigContext::Create()
        ->AddConfig("tl.ascend_auto_cv_combine", Bool(true))
    );
    PrimFunc transformed = CombineCV::Substitute(func, ctx);
    
    // 验证输出结构
    auto root_block = GetRootBlock(transformed);
    auto seq_stmt = root_block->body.as<SeqStmtNode>();
    
    EXPECT_TRUE(seq_stmt != nullptr);
    EXPECT_EQ(seq_stmt->seq.size(), 2);
    
    // 验证resource_scope标记
    auto cube_attr = seq_stmt->seq[0].as<AttrStmtNode>();
    auto vec_attr = seq_stmt->seq[1].as<AttrStmtNode>();
    
    EXPECT_TRUE(cube_attr != nullptr);
    EXPECT_TRUE(vec_attr != nullptr);
    EXPECT_EQ(cube_attr->attr_key, "resource_scope");
    EXPECT_EQ(vec_attr->attr_key, "resource_scope");
    EXPECT_EQ(cube_attr->value.as<IntImmNode>()->value, 0);  // cube=0
    EXPECT_EQ(vec_attr->value.as<IntImmNode>()->value, 1);  // vec=1
}
```

### API分类测试

```cpp
// UT测试例：验证API正确分类
TEST(CVCombineEmitter, APIClassification) {
    Map<Var, String> location_map;
    CVCombineEmitter cube_emitter(false, location_map);
    CVCombineEmitter vec_emitter(true, location_map);
    
    // 测试Cube API
    EXPECT_EQ(cube_emitter.isSubstringInMap(cube_emitter.callnodeMapPos_, "gemm_v0"), "cube");
    EXPECT_EQ(vec_emitter.isSubstringInMap(vec_emitter.callnodeMapPos_, "gemm_v0"), "cube");
    
    // 测试Vector API
    EXPECT_EQ(cube_emitter.isSubstringInMap(cube_emitter.callnodeMapPos_, "copy_ub_to_gm"), "vec");
    EXPECT_EQ(vec_emitter.isSubstringInMap(vec_emitter.callnodeMapPos_, "copy_ub_to_gm"), "vec");
}
```

## Pass测试例设计

### 同步点收集测试

```cpp
// UT测试例：验证同步点收集
TEST(CrossCoreSyncCollector, SyncPointCollection) {
    // 构造包含GM拷贝操作的IR
    Stmt cube_code = ConstructCubeCodeWithWorkspace();
    Stmt vec_code = ConstructVecCodeWithWorkspace();
    
    std::vector<CrossCoreSyncPoint> cube_sync_points;
    std::vector<CrossCoreSyncPoint> vec_sync_points;
    
    CrossCoreSyncCollector cube_collector(cube_sync_points, false);
    CrossCoreSyncCollector vec_collector(vec_sync_points, true);
    
    cube_collector(cube_code);
    vec_collector(vec_code);
    
    // 验证同步点数量
    EXPECT_EQ(cube_sync_points.size(), 2);  // copy_gm_to_l1, copy_l0c_to_gm
    EXPECT_EQ(vec_sync_points.size(), 2);   // copy_gm_to_ub, copy_ub_to_gm
    
    // 验证同步点属性
    EXPECT_EQ(cube_sync_points[0].scope, 0);     // cube
    EXPECT_EQ(cube_sync_points[0].is_write, false);  // copy_gm_to_l1是读操作
    EXPECT_EQ(cube_sync_points[0].pipe, "MTE2");
    
    EXPECT_EQ(cube_sync_points[1].is_write, true);    // copy_l0c_to_gm是写操作
    EXPECT_EQ(cube_sync_points[1].pipe, "FIX");
}
```

### 同步点匹配测试

```cpp
// UT测试例：验证同步点匹配与sync_flag_id分配
TEST(AutoInsertCrossCoreSync, SyncPointMatching) {
    Stmt cube_code = ConstructCubeCodeWithWorkspace();
    Stmt vec_code = ConstructVecCodeWithWorkspace();
    
    AutoInsertCrossCoreSync::AutoInsert(cube_code, vec_code);
    
    // 验证匹配后的sync_flag_id一致性
    // cube的write应该与vec的read匹配
    // cube的read应该与vec的write匹配
}
```

### 循环深度匹配测试

```cpp
// UT测试例：验证FindTargetLoopDepth算法
TEST(FindTargetLoopDepth, Algorithm) {
    // 场景1：相同循环结构
    CrossCoreSyncPoint cube_sp = CreateSyncPointWithLoops({2, 3, 4});
    CrossCoreSyncPoint vec_sp = CreateSyncPointWithLoops({2, 3, 4});
    
    FindTargetLoopDepth(cube_sp, vec_sp);
    
    // 验证：应该在最大深度匹配
    EXPECT_TRUE(cube_sp.target_for_node != nullptr);
    EXPECT_TRUE(vec_sp.target_for_node != nullptr);
    
    // 场景2：不同循环结构
    // cube: 2*3*4=24次迭代
    // vec: 4*6=24次迭代
    cube_sp = CreateSyncPointWithLoops({2, 3, 4});
    vec_sp = CreateSyncPointWithLoops({4, 6});
    
    FindTargetLoopDepth(cube_sp, vec_sp);
    
    // 验证：在总次数相等处匹配
    // cube在迭代24次后，vec也在迭代24次后
}
```

### IR结构验证测试

```cpp
// UT测试例：验证最终IR结构
TEST(CombineCV, IRStructureValidation) {
    PrimFunc func = ConstructComplexGEMMIR();
    
    auto ctx = PassContext::Create(
        ConfigContext::Create()
        ->AddConfig("tl.ascend_auto_cv_combine", Bool(true))
        ->AddConfig("tl.ascend_auto_cross_core_sync", Bool(true))
    );
    
    PrimFunc transformed = CombineCV::Substitute(func, ctx);
    
    // 验证IR结构完整性
    VerifyIRWellFormed(transformed);
    
    // 验证同步语句插入正确
    VerifySyncStmts(transformed);
    
    // 验证workspace使用一致
    VerifyWorkspaceConsistency(transformed);
}
```

## 算子测试例（example目录）

### 示例1：简单GEMM + Elementwise

```python
# example/simple_gemm_elementwise.py
import tilelang
from tilelang.transform import CombineCV

@tilelang.jit(pass_configs={
    "tl.ascend_auto_cv_combine": True,
    "tl.ascend_auto_cross_core_sync": True
})
def gemm_elementwise(M, N, K):
    # GEMM计算（Cube核心）
    # Elementwise操作（Vector核心）
    ...
```
