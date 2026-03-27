# GitHub CLI 安装与配置指南

> 参考文档：
> - Linux: https://github.com/cli/cli/blob/trunk/docs/install_linux.md
> - macOS: https://github.com/cli/cli/blob/trunk/docs/install_macos.md
> - Windows: https://github.com/cli/cli/blob/trunk/docs/install_windows.md

## 安装

### Ubuntu/Debian（apt 方式）

> **向用户展示的提示**：GitHub packages 源下载可能较为缓慢，请耐心等待。如用户希望加速，可尝试手动下载：https://github.com/cli/cli/releases

**步骤一：添加 GitHub CLI 源**

```bash
sudo mkdir -p -m 755 /etc/apt/keyrings
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null
sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
```

**步骤二：后台启动安装命令**

由于国内网络下载可能需要 5-10 分钟，应将安装命令放入后台运行，避免因超时被中断：

```bash
# 后台运行安装，输出重定向到日志文件
nohup sh -c 'DEBIAN_FRONTEND=noninteractive sudo apt-get install gh -y > /tmp/gh_install.log 2>&1' &
```

**步骤三：启动后台任务监控安装并验证结果**

使用 task 工具启动后台 agent，它会自动监控安装进度、实时输出日志、并在完成后验证结果：

```
task(
  description="Monitor gh installation progress",
  prompt="""
gh 安装命令正在后台运行。请循环执行以下命令直到进程结束：

sleep 30 && echo "=== 日志内容 ===" && tail /tmp/gh_install.log && echo "" && echo "=== 进程状态 ===" && (pgrep -a apt || pgrep -a dpkg || pgrep -a https || echo "无相关进程")

当进程状态显示"无相关进程"时，执行 `gh --version` 验证安装并返回结果。
""",
  subagent_type="general"
)
```

task 工具会阻塞等待直到安装完成。task 返回后即可继续后续流程（如认证 gh）。

**常见问题处理**：

| 问题 | 解决方案 |
|------|----------|
| 进程被意外中断 | 重新执行步骤二 |
| 锁文件被占用 | `sudo pkill -9 -f apt; sudo pkill -9 -f dpkg` 后重试 |
| 安装失败 | 查看 `cat /tmp/gh_install.log` 日志定位问题 |

### CentOS/RHEL/Fedora

```bash
sudo dnf install gh
# 或
sudo yum install gh
```

### macOS

```bash
brew install gh
```

### Windows

```powershell
winget install GitHub.cli
# 或
choco install gh
```

---

## 认证 GitHub

### 方式一：设备码登录（推荐，服务器环境）

适用于无浏览器环境（如 SSH 远程服务器）：

```bash
gh auth login
```

> **重要**：国内网络连接 GitHub 不稳定，`gh auth login` 可能因网络超时失败。**遇到网络错误必须自动重试至少 3 次**，不要向用户询问其他认证方式。

按提示选择：
1. **GitHub.com** 或 GitHub Enterprise
2. **HTTPS** 协议
3. **Login with a web browser**
4. 终端会显示一次性设备码（如 `529D-6CA9`）
5. 访问 https://github.com/login/device 输入设备码完成认证

### 方式二：环境变量（无需 gh auth login）

设置环境变量后无需执行 `gh auth login`：

```bash
export GH_TOKEN="your_personal_access_token"
```

**创建 Token**：https://github.com/settings/tokens/new
- 勾选 `repo`、`workflow` 等所需权限

### 方式三：Token 文件

```bash
echo "your_token" | gh auth login --with-token
```

### 验证认证状态

```bash
gh auth status
# 期望输出：✓ Logged in as <username>
```

---

## Token 管理

| 操作 | 方法 |
|------|------|
| 查看授权 | https://github.com/settings/applications |
| 创建 Token | https://github.com/settings/tokens/new |
| 登出 | `gh auth logout` |
| 刷新 Token | `gh auth refresh` |

---

## 注意事项

### 安装相关

1. **国内网络下载慢**：apt 方式从 GitHub packages 源下载可能需要 5-10 分钟
2. **使用后台安装 + task 监控**：必须使用 `nohup ... &` 后台运行安装命令，然后用 task 工具启动后台 agent 监控进度。禁止直接在前台运行 `apt-get install`，即使设置很长的超时，也可能因下载慢而有超时中断风险
3. **监控日志用 tail 而不是 cat**：监控进度时使用 `tail -f /tmp/gh_install.log` 实时输出日志变化，而不是用 cat 每次打印全部内容
4. **不要 kill 正在下载的进程**：如果下载进程还在运行，即使下载很慢，也不要主动 kill
5. **锁文件冲突**：若提示 lock 文件被占用，先 kill 卡住的进程再重试

### 认证相关

6. **设备码有效期**：设备码有效期约 15 分钟，需及时在浏览器完成认证
7. **Token 权限**：确保 Token 有 `repo`、`workflow` 等必要权限

### 网络相关

8. **国内网络不稳定，必须自动重试**：GitHub 操作（如 `gh auth login`、`git push`、`gh pr create`）因网络超时或连接失败时，**必须自动重试至少 3 次**，禁止因网络问题向用户询问其他方案。只有连续失败 3 次后才考虑其他认证方式（如 Token）
9. **代理设置**：如需代理，设置 `export https_proxy=http://proxy:port`