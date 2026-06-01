# 代码骨架模板

生成 `example_{op}.py` 时的文件结构：

```python
import tilelang
from tilelang import DataType, language as T
import torch

# ========== 算子实现 ==========
@tilelang.jit(out_idx=[...], pass_configs={...})
def op_name(M, N, block_M, block_N, dtype="float"):
    # 分块计算
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2

    @T.prim_func
    def main(Input: T.Tensor((M, N), dtype), Output: T.Tensor((M, N), dtype)):
        with T.Kernel(..., is_npu=True) as (cid, vid):
            # buffer 分配
            # 数据搬入
            # 计算
            # 数据搬出
            pass

    return main

# ========== 测试 ==========
if __name__ == "__main__":
    tilelang.disable_cache()  # 在 __main__ 中禁用编译缓存
    torch.manual_seed(...)
    test_configs = [...]  # 来自 design.md §8

    for config in test_configs:
        # 1. 创建 kernel
        # 2. 生成输入数据
        # 3. 执行 kernel
        # 4. golden 对比
        # 5. 精度检查
        pass

    print("Test Passed!")
```

**融合算子注意事项**：

**Developer 模式（推荐，默认消除 workspace/vid）**：
- 装饰器无 `workspace_idx`，函数签名无 workspace 参数
- `T.Kernel(block_num, threads=2, is_npu=True) as (cid)`（单轴 + `threads=2`）
- Cube↔Vector 用 `alloc_shared/alloc_fragment` 片上 `T.copy` 直连，无 GM 往返、无 `vid` 偏移
- 完整骨架/映射表见 [tilelang-expert-to-developer mode-examples.md §6](../../tilelang-custom-skill/tilelang-expert-to-developer/references/mode-examples.md#6-cv-融合--推荐写法消除-workspace--vidthreads2)

**回退（Expert/混合或复杂同步场景）**：
- 函数签名包含 workspace 参数，`workspace_idx` 指定索引位置
- Cube 核输出通过 `T.copy` 写入 workspace，Vector 核从 workspace 读取（见 mode-examples.md §7）
