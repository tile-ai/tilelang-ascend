---
name: tilelang-error-fixer
description: TileLang-Ascend 错误诊断、调试与修复技能。融合了 GDB 故障定位、IR Dump 分析、目标代码倒推（AOT）的诊断能力。提供从环境检查、错误定位、AOT验证到 C++ Pass 修复的端到端工作流。
---

# TileLang-Ascend Pass 错误诊断与修复技能

本技能提供 TileLang-Ascend 框架中 Pass 层面错误的完整诊断和修复能力，支持从前端报错、Core Dump 崩溃追溯到后端 C++ Pass 的修复，并集成 AOT 目标代码验证机制。

## 使用场景
- **遇到 Core Dump 或 Segmentation Fault 等严重崩溃，且无其他明确错误信息**
- TileLang 编译报错或生成的 Ascend 目标代码结果不符合预期（精度/性能问题）
- 编写或修改 `src/transform/*.cc` 下的新 Pass 时出现逻辑异常
- 需要使用 AOT 提前编译模式进行算子调试与倒推

## 触发机制
当用户输入包含以下关键字时自动触发：
- **"Core Dump"**、**"Segment Fault"**、**"段错误"**
- **"TileLang 编译报错"**、**"Pass 生成代码错误"**
- **"排查精度问题，分析 IR"**、**"使用 AOT 调试这个算子"**
- **"修复 TileLang Pass"**

## 工作流程

### 步骤 1：环境与状态检查
在进行任何 Pass 调试前，必须确保底层环境正常。
1. 执行环境检查脚本验证 `TL_ROOT`, `PYTHONPATH`, `ASCEND_HOME_PATH`, `LD_LIBRARY_PATH` 等变量。
2. 确认 C++ 工程是否处于最新编译状态。

### 步骤 2：多维度错误信息获取 (Dump & Catch)
根据报错类型，采取不同的信息获取策略：

1. **严重崩溃 (Core Dump / Segment Fault)**:
   - 现象：脚本运行直接崩溃，没有任何 Python 异常堆栈。
   - 动作：使用 GDB 挂载 Python 进程进行排查。
     ```bash
     gdb --args python xxx.py
     # 在 GDB 命令行输入：
     (gdb) run
     # 崩溃后输入 bt 查看堆栈，精确定位是 C++ 源码中哪个函数/Pass 引发了内存违规
     ```

2. **C++ 编译报错 (Build Error)**:
   - 读取 `/tmp/tmp*.cpp` (获取最新的生成的失败代码) 定位语法或底层 API 错误。

3. **逻辑/精度异常 (IR Dump)**:
   - 追踪 Pass 阶段性变换。在 `tilelang/engine/lower.py` 中的 `lower()` 函数内，添加 `print(mod)` 以查看最终 IR 图：
     ```python
     def lower(...):
         ...
         mod = OptimizeForTarget(mod, target)
         print(mod) # <--- 在此处打印 IR，排查 Pass 优化后的结果
         codegen_mod = device_codegen(mod, target)
     ```

4. **目标代码分析 (Target Dump)**:
   - 通过 `@tilelang.jit(debug_root_path=...)` 或 `.get_kernel_source()` 抓取最终生成的 C++ 核函数。

### 步骤 3：核心错误原因分析 (TileLang 专属模式)
分析获取的 IR、目标代码或 GDB 堆栈，重点排查 TileLang-Ascend 特有的雷区：

1. **循环嵌套与次数匹配**：对比 Cube 和 Vec 部分的循环 `extent`，诊断类似 "Vec loop times is not enough" 的报错。
2. **跨核同步点 (Sync Points)**：追踪 `ascend_auto_set_cross_flag` / `wait_flag`，排查死锁或数据竞争。
3. **工作空间 (Workspace) 访问**：检查张量切片范围，确认访问是否越界。
4. **数据依赖与流水线 (Pipeline)**：分析 `T.Pipelined` 的 `num_stages`，检查读写重叠冲突。

### 步骤 4：修复方案制定 (支持 AOT 倒推)
LLM 根据步骤 3 提出修复方案。对于代码生成或精度相关的复杂问题，**强烈建议采用 AOT (Ahead-Of-Time) 模式**进行验证：

**方案 A：直接修复 C++ Pass（适用于 GDB 已明确指出问题代码行的情况）**
- 直接定位 `src/transform/` 或 `src/target/` 下的 `.cc` 文件，提供代码修改建议。

**方案 B：AOT 逆向工程（可参考 `examples/gemm_aot` 目录）**
AOT 模式将算子执行拆解为三个清晰的阶段：
1. **算子生成**：TileLang 前端语句 -> AscendC 目标代码（Python -> C++）。
2. **算子编译**：手动或使用脚本编译生成的算子代码（AscendC -> `.so` 动态库）。
3. **算子调用**：在 Python 中使用 Ctypes 直接加载并调用该 `.so` 库进行测试。

*工作流说明*：先修改导出的 `.c/.cpp` 目标代码，使用 AOT 流程编译为 `.so` 并通过 Ctypes 验证。**如果 AOT 测试通过**，再逆向推导，生成修改底层 C++ Pass 的具体代码方案。

### 步骤 5：执行修复与 C++ 重新编译
1. 展示修复方案（C++ 代码 Diff），等待用户确认。
2. 备份涉及的原始 C++ 文件。
3. **强制执行编译命令**：修改 C++ Pass 后，必须在 `build` 目录下执行 `make -j$(nproc)`。

### 步骤 6：修复验证与回滚
1. 重新执行用户的重现脚本。
2. 对比修复前后的 IR 图（`diff before.tir after.tir`）或目标代码，确认修改符合预期。
3. **失败处理**：如果编译失败或运行不通过，恢复备份代码，清理 `/tmp` 缓存，重新进入步骤 3。

---

## 输出格式

### TileLang Pass 诊断与修复报告

```markdown
## TileLang Pass 分析报告

### 基本信息
- **触发报错模块**：[如：Segmentation Fault / CombineCV Pass]
- **涉及算子/脚本**：`example_flash_attention.py`
- **获取的调试信息**：[如：GDB Backtrace 指向 `src/transform/combine_cv.cc:145` / `lower.py` 打印的 IR 图]

### 错误原因分析
- **错误分类**：[如：空指针解引用 / 循环迭代不匹配]
- **详细分析**：通过 GDB 堆栈发现，在处理 `T.Pipelined` 节点时，试图访问不存在的 `extent` 属性，导致 Core Dump。
- **定位代码**：`src/transform/cross_core_pipeline.cc: 89行`

### 修复方案建议 (需用户确认)
**建议直接修复 Pass C++ 源码**
1. 修改 `src/transform/cross_core_pipeline.cc`，在访问节点前增加类型检查...
2. 代码 Diff:
   ```cpp
   // Before
   auto extent = op->extent.as<IntImmNode>()->value;
   // After
   if (const auto* imm = op->extent.as<IntImmNode>()) {
       auto extent = imm->value;
       // ...
   }
   ```
3. 修复后将自动执行 `make -j8`。

*(如果是生成逻辑错误，将建议先参考 `examples/gemm_aot` 使用 AOT 流程修改目标代码进行验证)*

### 执行与验证结果
- 编译状态：✅ 成功
- 测试脚本验证：✅ 通过
- Core Dump 状态：已解决
```