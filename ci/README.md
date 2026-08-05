# 算子测试提交指南

我们在把 `examples/` 下的算子从老的 `bench_test.sh` 迁到 Pytest。

老流程判断算子对不对,是去 stdout 里 grep `Kernel Output Match!` 这句话 —— 算子把这句话打出来就算过,哪怕结果是错的。新流程用真的数值断言。

**这件事需要大家按照任务分配,写测试文件。**

---

## 大致流程

```
① 从 ascendc_pto 切分支
② 查登记表，拿到你的测试该叫什么名字
③ 照样板写测试
④ 本地跑通 + 格式检查
⑤ 提 PR，base 选 ascendc_pto
⑥ 等 CI（约 11 分钟）
```

**你只需要交一个文件:你写的那个测试。**登记表已经把 140 个算子的测试文件名全预留好了,**不用改它**。

---

## 一、你可以提交什么

打开 `ci/operator_test_manifest.yaml`,里面按目录分组列了 140 个算子:

```yaml
  examples/xattention/xattention.py: examples/xattention/test_xattention.py
```

- **冒号左边**:算子文件
- **冒号右边**:你要新建的测试文件,完整路径,照抄,一个字都别改

一条登记只有在测试文件真实存在时才生效。文件没建之前,那个算子照常由 `bench_test.sh` 跑,不受影响。

---

## 二、照样板写

样板:**`examples/batch_gemm/test_batch_gemm.py`**

打开它照着写。**算子原文件一个字都别改。**

样板里那 20 行同时示范了三件事,下面逐条说。

### 红线:加载算子的那句,必须写在测试函数里面

```python
# ✗ 写在模块顶层 —— pytest 收集阶段就会执行算子
EXAMPLE = _load_example()

def test_xxx():
    ...
```

```python
# ✓ 写在测试函数里 —— 真正跑这个测试时才执行
def test_xxx():
    example = _load_example()
```

pytest 分两步:先「收集」(把每个 `test_*.py` import 一遍,看看里面有哪些测试),再「运行」。

写模块顶层的话,算子在**收集阶段**就跑了。这时候它一旦失败,pytest 认为是「这个文件读不进来」,后果比测试失败严重得多:

```
顶层加载：  Interrupted: 1 error during collection   → 同一批里其它文件一个都没跑
函数内加载：1 failed, 3 passed                        → 只死自己那一个
```

写在函数里,收集阶段只读测试文件本身那几行 import,完全不碰 NPU。

**这一条比下面所有内容都重要。**

### 算子顶层有 `parse_args()` 的,要把 `sys.argv` 换掉

不少算子在顶层就解析命令行:

```python
args = parser.parse_args()      # 它去读 sys.argv
```

而 CI 跑的是 `pytest --forked -n 8 -m "..."`,`sys.argv` 里全是 pytest 的参数。算子的 parser 不认识 `--forked`,直接 `SystemExit`。

所以加载前把 `sys.argv` 临时换成算子自己的样子,加载完换回来 —— 样板里就是这么写的:

```python
original_argv = sys.argv
try:
    # batch_gemm.py parses arguments at import time. Hide Pytest arguments
    # while loading it without changing the original Example.
    sys.argv = [str(source)]
    spec.loader.exec_module(module)
finally:
    sys.argv = original_argv
```

`finally` 里那句一定要有:算子中途抛异常时也得把 `sys.argv` 还回去,否则后面的测试全遭殃。

要指定 shape 就把参数拼进去:

```python
sys.argv = [str(source), "--m", str(m), "--n", str(n), "--k", str(k)]
```

### 三种写法,按你的算子是什么结构挑一种

**A. 算子有现成函数可以调**(`main()` / `check_case()` / `batch_matmul()` 之类)

加载完直接调它,最省事,样板就是这一种:

```python
def test_batch_gemm_accuracy() -> None:
    example = _load_example()
    kernel = example.batch_matmul(batch, m, n, k, block_M=128, block_N=256, K_L1=64)
    ...
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
```

**B. 算子没有 `if __name__ == "__main__"`,顶层直接跑完并断言**

`exec_module` 执行完就等于算子跑完、精度也比完了,异常自然抛给 pytest。测试体就一行:

```python
def test_silu_run() -> None:
    _load_example()
```

**C. 算子的活儿写在 `if __name__ == "__main__":` 里**

这种文件被 import 时那段**根本不执行**(因为 `__name__` 不是 `__main__`),要用 `runpy` 指定名字来跑:

```python
import runpy
import sys
from pathlib import Path


def _run_example(*argv) -> None:
    source = Path(__file__).with_name("xattention.py")
    original_argv = sys.argv
    try:
        sys.argv = [str(source), *argv]
        runpy.run_path(str(source), run_name="__main__")
    finally:
        sys.argv = original_argv
```

### 不要用 `subprocess` 包装原脚本

`subprocess.run([sys.executable, "silu.py"])` 这种写法测试也能过,精度也真能抓错,但算子的代码跑在**子进程**里,而 `--cov` 挂在 pytest 进程上,追不进去。

算子一旦登记进本表,`bench_test.sh` 就不再用 `coverage run` 跑它的原脚本了(改走 `pytest --cov`),所以子进程写法会让这个算子的覆盖率变成 0。实测同一个算子:

```
subprocess      1 passed   31s   tilelang 覆盖 0 行
importlib 直调  1 passed   27s   tilelang 覆盖 3629 行 / 144 文件
```

顺带,失败时 subprocess 只能给你一段 stdout,进程内直调会直接指到 `assert_close` 那一行,告诉你哪个元素差多少。

### shape 别照抄原文件里最大的那组

CI 是 `pytest --forked -n 8`,八个用例同时在设备上。原文件为了跑性能常用很大的 shape,单个张量几百 MB 甚至几 GB,几个撞一起就 `NPU out of memory`,而且撞不撞看调度,时绿时红。

挑够用的小 shape,把大的留给原文件自己。注意有些 kernel 内部有 `assert dim == 128` 这类约束,改之前先看一眼。

如果算子的参考实现本身就要几十 GB(比如先把完整的 score 张量算出来再 mask),那这个算子暂时不适合迁,让它继续留在 `bench_test.sh` 里跑。

---

## 三、提交前检查

### 1. 本地跑通

运行前请参考仓库的doc文件自行完成基本环境配置；

完成环境配置后

```bash
python3 -m pytest examples/<目录>/test_<算子>.py -v
```

**看到 `passed` 才算过:**

```
============================= test session starts ==============================
collected 1 item

examples/batch_gemm/test_batch_gemm.py::test_batch_gemm_accuracy PASSED  [100%]

============================== 1 passed in 12.36s ==============================
```

写了几个 case 就是 `N passed`,数字对上你写的个数。


### 2. 确认文件名和登记表一致

再核对一遍路径和文件名,和登记表冒号右边**完全一样**。

名字对不上不会当场报错,但那个测试**不会被 Pytest 接管** —— 它会被当成普通脚本由 `bench_test.sh` 收走执行,而 pytest 风格的文件直接 `python` 跑是不干活的,CI 那一格就会红。日志里会有一句 `operator test(s) matching no manifest entry`,把没认领的文件名列出来,照着改就行。

### 3. 本地进行格式检查

CI 的格式检查只跑 **ruff**。本地装一个就能完全复现,不用起昇腾环境,几秒钟的事。

#### 安装检查工具

```bash
pip install -U ruff
python3 -m ruff --version          # 能打出版本号就装好了
```

#### 检查

**先 `cd` 到仓库根目录**,再跑。规则(行宽 140、双引号等)写在根目录的 `pyproject.toml` 里,ruff 会自己往上找 —— 在别的目录下跑读不到配置,结果和 CI 对不上。

```bash
cd /path/to/tilelang-ascend        # ← 必须先切到仓库根目录

python3 -m ruff format --check --diff examples/<目录>/test_<算子>.py
python3 -m ruff check examples/<目录>/test_<算子>.py
```

**两条都是这个输出、退出码都是 0,才算过:**

```
1 file already formatted
All checks passed!
```

#### 修改

没过不用手动改,让 ruff 自己来 —— 把 `--check` 去掉、给 `check` 加 `--fix`,它会**直接改你的文件**:

```bash
python3 -m ruff check --fix examples/<目录>/test_<算子>.py
python3 -m ruff format examples/<目录>/test_<算子>.py
```

```
Found 1 error (1 fixed, 0 remaining).
1 file reformatted
```

改完再跑一遍上面的「查」确认全绿,然后**把 ruff 改过的版本提交上去**。

### 4. 只提交 test 文件

**只允许新增 `test_*.py`,不允许修改仓库里的任何其他文件。**

```
你的算子文件本身                     ← 一个字都别改
ci/operator_test_manifest.yaml       ← 名字已经预留好了，不用改
examples/bench_test.sh
.github/workflows/ci_cd.yml
.gitignore
src/ 或 tilelang/ 下的任何文件
```

提之前 `git status` 看一眼,或者在 PR 的 **Files changed** 页面确认:**列表里应该只有你新建的 test 文件。**

碰了上面那些,CI 会从 11 分钟变 50 分钟,还会被打回。

**一次可以提交多个测试。**几个算子写完放一个 PR 里没问题,CI 也只跑你新加的那几个。

---

## 四、提到哪个分支

**base 选 `ascendc_pto`**,也就是仓库的默认分支,开 PR 时一般不用改。

四个下拉框核对成这样:

```
base repository:  tile-ai/tilelang-ascend        base:     ascendc_pto
head repository:  <你的用户名>/tilelang-ascend    compare:  <你自己的分支>
```

唯一容易错的是 **`base repository` 默认可能是你自己的 fork** —— 选错就变成给自己提 PR 了。

提错了不用重开,PR 页面标题下面点 **Edit** 可以改。

标题写 `[Test] Add <算子> Pytest test`。

---

## 五、提完之后关注什么

### CI 跑三项

| 检查项 | 干什么 | 多久 |
|---|---|---|
| `Check PR Status and Changes` | 看你改了哪些文件、决定跑多少 | 几秒 |
| `AscendC-PTO CI` | 代码格式(ruff) | 半分钟 |
| `Benchmark Tests (Ascend NPU)` | 编译 + 跑测试 | **约 11 分钟** |

最后那项内部是:

```
增量编译    约 9.5 分钟    任何 PR 都要，省不掉
你的测试    约 1 分钟      只跑你新加的这一个
```

一次加好几个测试也一样,只跑你加的那几个,不跑全量。

### 一直 queued 不动

在排队等昇腾机器,别的 PR 占着。**这是正常的,等就行**,见过排 70 多分钟的。

### 红了怎么办

点 **Details** 进去,展开红色那一步看报错。常见两种:

**① 日志里有 `operator test(s) matching no manifest entry`,列出了你的文件**

文件名或位置和登记表对不上。回「一、你可以提交什么」查表,照冒号右边那个路径改。

**② 跟你的代码没关系的**

比如 `Checkout code` 报 `curl 28: Connection timed out`、编译中途莫名断了、排队超时被取消。

这种在 PR 下面发一条评论,内容包含 `/re-test`,机器人会只把失败的那几项重跑一遍:

```
/re-test
```

---

## 总结

```
从 ascendc_pto 切分支 → 查登记表拿文件名 → 照 test_batch_gemm.py 写
→ 本地跑通 + ruff 检查 → PR 提到 ascendc_pto
```

写测试时守住这四条:

1. **加载算子那句写在测试函数里**,不写模块顶层 —— 写顶层的话一个算子失败会让整批测试 `Interrupted`
2. 算子顶层有 `parse_args()` 的,**加载前后换掉 `sys.argv`**
3. **别用 `subprocess` 包装原脚本** —— 覆盖率会记成 0
4. **shape 挑小的** —— CI 是 8 路并行,照抄原文件的大 shape 容易 OOM

**只交测试文件,不改 manifest。**
