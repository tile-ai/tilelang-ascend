# IR Pass Trace — 编译 Pass 可视化工具

<div style="text-align: left;">
<em>相关文档:</em> <a href="debug_tools_for_tilelang.md">Debugging Tile Language Programs</a>
</div>

零侵入的编译 pass 可视化工具，生成自包含 HTML 页面，展示所有 pass 的 IR diff。

## 快速开始

```bash
# 1. 设置环境变量并运行 kernel
TILELANG_DUMP_PASSES=1 python my_kernel.py

# 2. 打开生成的 HTML 页面
open ./tmp/ir_dump/my_kernel_20260614120000/ir_trace.html
```

无需修改 kernel 代码，`tilelang/__init__.py` 自动检测环境变量并激活 patch。

## 环境变量

| 变量 | 值 | 说明 |
|---|---|---|
| `TILELANG_DUMP_PASSES` | `0` / 不设置 / `off` / `false` | 关闭（默认） |
| | `1` / `all` / `on` / `true` | dump 所有 phase |
| | `phase1` | 只 dump Phase 1（LowerAndLegalize） |
| | `phase2` | 只 dump Phase 2（OptimizeForTarget） |
| `TILELANG_DUMP_DIR` | 路径 | 自定义输出目录（默认 `./tmp/ir_dump/{kernel}_{timestamp}/`） |

## 页面功能

### Diff 可视化

- **Phase tabs**：切换 LowerAndLegalize / OptimizeForTarget
- **左侧栏**：列出所有 pass，绿点 = 有变更，灰点 = 无变更，附带 +N −M 统计
- **Diff 表格**：GitHub 风格 side-by-side，支持行内字符级高亮
- **智能配对**：仅空格差异的行自动对齐，淡蓝色显示（区别于红/绿的内容差异）
- **上下文折叠**：未变更的行默认折叠，点击展开

### 三级视觉层次

| 类型 | 符号 | 背景色 | 含义 |
|-----|------|--------|------|
| 无差异 | （空） | 白/灰 | 行完全相同 |
| 空格差异 | `~` | 淡蓝 `#f0f0ff` | 仅缩进/空格不同 |
| 内容差异 | `−` / `+` | 红 / 绿 | 代码实质变化 |

## 键盘快捷键

| 快捷键 | 功能 |
|--------|------|
| `j` / `k` | 跳转到下一个/上一个 pass |
| `Shift+E` | 展开所有 pass 中隐藏的上下文行 |
| `Escape` | 取消对齐模式 / 清除选中 |

## 手动对齐（Beyond Compare 风格）

当自动 diff 配对不正确时使用（例如左侧某行应与右侧不同行对齐）。

### 操作流程

```
F7  →  点击左侧行号  →  F7  →  点击右侧行号  →  对齐完成
                                        Escape → 取消
```

1. **按 F7**：顶部出现黄色横幅 "Click a left (before) line number"
2. **点击左侧行号**：该行号变橙色，横幅显示 "Left line N selected"
3. **按 F7**：橙色变蓝色锁定，横幅变蓝 "Click a right (after) line number"
4. **点击右侧行号**：两行合并为一行，自动计算字符级 diff，横幅消失
5. **Esc**：任意步骤均可取消

### 效果

- 对齐后的行显示左侧 3px 橙色边线
- 左右内容并排，变化的字符用红/绿高亮
- 原配对中被"挤掉"的内容保留为独立行（孤儿行）

### 典型场景

```
自动配对:                         手动对齐后:
  15 │−│ A = buf[0]                15 │−│ A = buf[0]  │ 20 │+│ A = buf[0] * 2
  16 │−│ B = buf[1]                16 │−│ B = buf[1]
     │+│ A = buf[0] * 2    →      17 │+│ C = buf[2]
  17 │+│ C = buf[2]
```

## 工具栏按钮

| 按钮 | 功能 |
|------|------|
| ⊞ Show all context | 展开当前 pass 所有隐藏行 |
| ⊟ Collapse | 重新折叠上下文行 |
| 📋 Copy Before | 复制完整的 pass 前 IR 到剪贴板 |
| 📋 Copy After | 复制完整的 pass 后 IR 到剪贴板 |

## Pass 折叠/展开

- 点击 changed pass 的标题栏可折叠/展开 diff 内容
- 标题栏 sticky 固定，长 diff 滚动时始终可见

## 行号点击高亮

- 点击任意行号，该行高亮为黄色（再次点击取消）

## 输出文件

```
./tmp/ir_dump/{kernel_name}_{timestamp}/
  ir_trace.html              ← 主输出，浏览器直接打开
  phase1_LowerAndLegalize/   ← 原始 .tir 文件（可用 BeyondCompare 比较）
    00_InjectTmpBuffer_before.tir
    00_InjectTmpBuffer_after.tir
    ...
  phase2_OptimizeForTarget/
    ...
```
