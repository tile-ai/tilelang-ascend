# Z3 默认启用与 PyPI 依赖

本次调整仅涉及 Z3 构建、加载和打包；同步后端仍默认 `legacy`，使用新同步算法仍需
显式选择 `tl.ascend_sync_backend="upstream"`。历史验收摘要见文末。

## 安装与编译

`USE_ASCEND_SYNC_Z3` 的新默认值为 ON。构建依赖和运行依赖均声明
`z3-solver>=4.13.0,<4.15.5`，Linux 构建还要求 `patchelf>=0.17.2`。
标准 pip 隔离构建会安装所声明的构建依赖，无需手动提供 Z3 路径。例如，在已有
CANN 配置的开发环境中：

```bash
USE_ASCEND=true python -m pip install -e .
```

这里 `USE_ASCEND` 是已有的后端构建选项。显式禁用证明服务仍可使用
`USE_ASCEND_SYNC_Z3=OFF`；pip 仍会按依赖声明安装 z3-solver，但 OFF 原生库不会
预加载它。已有 CMake cache 中的 OFF 会被保留，不会强制改成 ON。

直接 CMake 构建不会执行 pip 安装，需要先安装构建依赖，或指定具有头文件和原生库的
`Z3_ROOT`。显式根目录保持最高优先级，失效时直接诊断，不混用其他环境的库。
通过 pip 构建时，setup 将两个 Python 解释器变量都指向当前解释器，避免隔离环境错配。
下面以 Ascend 后端、关闭 LLVM 的原生构建为例：

```bash
python -m pip install -r requirements-build.txt
cmake -S . -B build -DUSE_ASCEND=ON -DUSE_LLVM=OFF \
  -DPython3_EXECUTABLE="$(command -v python)"
cmake --build build --target tilelang tilelang_module tilelang_static --parallel 12
```

## 原生库加载

wheel 与 editable 包不再捆绑 `libz3.so*`，而是通过当前运行环境的 z3-solver 分发元数据
定位原生库。包内私有文件 `tilelang/lib/ascend_sync_z3_runtime.json` 记录本次构建的
实际启用状态，安装校验不依赖环境变量是否恰好设置。

ON 包先校验运行依赖及原生版本，再以 RTLD_GLOBAL 加载所选 Z3，并保留句柄。
Python 包的 TileLang 共享库副本移除 Z3 对应的版本化 DT_NEEDED 项，避免构建端
`libz3.so.4.15` 阻止运行端合法的 `libz3.so.4.13` 提供 C API。其他动态依赖保留；
`$ORIGIN` 继续用于 TileLang/TVM 自身。这个过程不导入 Python z3 模块。

只对包目录中的副本执行 ELF 修补；原始 CMake 输出保留正常链接依赖和旁库，独立
`cmake --install` 的原生安装闭包保持不变。旧包目录中的 Z3 旁库会清理，许可证和
NOTICE 继续分发。缺少 provider、版本越界、状态文件损坏或缺少 patchelf 均明确报错。

## 验证方式

```bash
python -m pytest -q testing/python/test_pypi_z3_runtime.py -p no:cacheprovider
```

新增测试包含真实 ELF 的拷贝、依赖修补和跨 SONAME 加载；其中的小型 C provider 仅用于
验证加载机制，不替代真实 Z3 求解器测试。真实构建/安装还应覆盖：

- 无 ON/Z3_ROOT 参数的隔离 wheel、PEP660 editable 安装。
- 原生 Z3 4.13.0、4.15.4 的构建端与运行端四种组合，以及真实证明与同步语义测试。
- Python 包不含 Z3 旁库，实际加载路径来自运行环境，原始 CMake 输出没有被修补或删除。
- 默认 ON、显式 OFF、已有 OFF cache、有效/无效显式根目录，以及原生安装。
- 完整主机回归和 A3 设备测试；结果另存新的验收记录，不借用历史 v8 的计数。

## 历史验收记录

Linux aarch64、Python 3.11 环境下的 Z3 默认启用、wheel/editable 安装和同步回归结果
汇总在[验收摘要](validation/summary.json)。该记录对应原生引擎 v8；当前代码为 v10，
摘要中的结果不构成当前 v10 的完整验收。

主机、设备及独立 legacy VS 回归的文件集合保存在
[测试清单](validation/test-suites.json)，便于重新执行各自范围。历史 A3 性能样本见
[性能原始数据](validation/performance.json)；这些样本对应 v8 的四个小 kernel，
计时包含 host launch gaps，不能据此推导 v10 的性能或设备执行时间收益。

原始验证脚本、日志和完整源码清单未随仓库保存。原记录中的工作区存在未提交的 TVM
修改，仅保存文件及差异的哈希，没有相应补丁；精确受测输入与干净 checkout 的一致性
尚不能复核。摘要保留了这些证据边界，后续验收应基于明确的源码版本重新执行并保存
可获取的脚本、日志和产物。
