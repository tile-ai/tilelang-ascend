# XLLM 客户算子迁移

本目录包含从 `/mnt/workspace/temp/xllm/xllm/compiler/tilelang/targets/ascend/kernels/` 迁移的三个客户算子实现。

## 背景

这些算子来自 XLLM 项目的 TileLang Ascend kernel 实现，用于验证我们的编译器对真实客户代码的支持情况。

## 算子列表

### 1. `fused_gdn_gating.py` - 融合 GDN Gating
- **功能**: 实现 Gated Deep Network 的 gating 机制，包含 softplus、sigmoid、exp、abs 等操作的融合
- **特点**: 
  - 多 pipeline 协作 (MTE2/MTE3/V)
  - 动态 chunk 处理
  - 支持多种 head 数量 (4, 6, 8, 12, 16, 24, 32, 48, 64, 128)
- **测试**: 运行 `python fused_gdn_gating.py` 会验证 JIT 编译并与 PyTorch 参考实现对比

### 2. `rope.py` - 旋转位置编码 (RoPE)
- **功能**: 实现旋转位置编码的 in-place 计算
- **特点**:
  - 多 pipeline 流水线处理
  - 使用 `T.tile.gather` 进行元素交换
  - 支持 bf16 输入和输出
- **测试**: 运行 `python rope.py` 会验证 kernel 正确性

### 3. `split_qkv_rmsnorm_mrope.py` - 融合 QKV 分离 + RMSNorm + M-RoPE
- **功能**: 实现 Transformer 中的 QKV 分离、RMS 归一化和多头旋转位置编码的融合操作
- **特点**:
  - 复杂的 Tensor Parallelism 支持
  - 多个 head 配置 (支持 Qwen3.5/3.6 等多种模型)
  - 融合 RMSNorm 和 M-RoPE 计算
- **测试**: 运行 `python split_qkv_rmsnorm_mrope.py` 会验证 kernel 正确性

### 4. `utils.py` - 工具函数
包含以下辅助功能：
- **Pipeline 同步宏**: `mte2_notify_v`, `v_wait_mte2` 等用于多 pipeline 间同步
- **硬件信息检测**: `detect_vec_core_num` 检测向量核心数量
- **Pass 配置**: `DEFAULT_ASCEND_PASS_CONFIGS` 默认编译器配置

## 使用方法

### 前置要求
1. 确保已正确安装 tilelang 和 torch_npu
2. 需要在 Ascend NPU 硬件上运行
3. 设置好 `PYTHONPATH` 环境变量

### 运行测试

```bash
# 测试 fused_gdn_gating
python fused_gdn_gating.py

# 测试 rope
python rope.py

# 测试 split_qkv_rmsnorm_mrope
python split_qkv_rmsnorm_mrope.py
```

## 与客户实现的一致性

迁移时保持了以下一致性：
1. **Kernel 实现**: 所有 `T.prim_func` kernel 代码完全保留，包括 buffer 分配、pipeline 同步、计算逻辑
2. **Pass 配置**: 使用客户相同的编译器 pass 配置
3. **硬件适配**: 保留 `detect_vec_core_num` 和 UB 内存预算等硬件相关的启发式
4. **参考验证**: 保留 PyTorch 参考实现用于验证正确性

## 修改说明

移除的内容（与测试无关）：
- AOT 代码生成相关代码 (`generate_source`, `register_kernel` 等)
- argparse 命令行参数解析
- logger 输出改为 print

## 许可证

这些算子代码来自 XLLM 项目，遵循 Apache License 2.0。

详细信息请参阅 `/mnt/workspace/temp/xllm/LICENSE`。
