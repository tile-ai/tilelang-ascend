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
- 函数签名需包含 workspace 参数，`workspace_idx` 指定索引位置
- Cube 核输出通过 `T.copy` 写入 workspace，Vector 核从 workspace 读取
