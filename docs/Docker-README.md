# TileLang Ascend Docker 使用指南

本指南介绍如何使用 TileLang Ascend 项目的 Docker 环境进行开发和部署。

## 概述

此 Dockerfile 基于 Ubuntu 22.04，预装了 TileLang 项目及其依赖，包括 Ascend NPU 支持。容器已预编译 AscendNPU-IR 组件，并优化了镜像大小以便打包 ISO。

## 构建镜像

### 默认构建（中国大陆用户）

```bash
sudo docker build -f docker/Dockerfile -t tilelang-ascend .
```

此构建会：
- 使用清华镜像加速 Git 子模块下载
- 预编译 AscendNPU-IR 组件
- 清理源码以减小镜像体积

### 其他地区构建

如果您在中国大陆以外地区，可以指定地区参数跳过 URL 替换：

```bash
docker build --build-arg REGION=other -f docker/Dockerfile -t tilelang-ascend .
```

## 运行容器

### 交互式运行

```bash
docker run -it --rm tilelang-ascend
```

进入容器后，您会看到欢迎消息：

```
Welcome to TileLang AscendNPU IR Docker container!

TileLang URL: https://github.com/tile-ai/tilelang-ascend/tree/npuir

Pre-compiled AscendNPU-IR binaries are available at /build/AscendNPU-IR/build/
To reinstall or update NPU IR with pre-compiled binaries (faster), use:
  bash install_npuir.sh --bishengir-path=/build/AscendNPU-IR/build/install
```

如需进入已运行容器，也可执行：

```bash
docker exec -it <container_name_or_id> bash
```

### 挂载工作目录

```bash
docker run -it --rm -v $(pwd):/workspace tilelang-ascend
```

## 容器特性

### 预编译组件

- **AscendNPU-IR**: 已预编译并存储在 `/build/AscendNPU-IR/build/`
- **TileLang**: 已安装到 Python 环境中（Python 3.9/3.10/3.11）

### 环境配置

- **用户**: `tilelang`
- **工作目录**: `/home/tilelang`
- **时区**: Asia/Shanghai
- **Python**: 默认 Python 3.10，多版本支持

### 加速优化

- **Git 镜像**: 中国大陆默认使用清华镜像加速子模块下载
- **编译缓存**: 预编译二进制文件可复用，避免重复编译

## 使用示例

### 1. 验证环境

```bash
# 进入容器后
python -c "import tilelang; print('TileLang imported successfully')"
```

### 2. 重新安装 NPU IR（使用预编译）

```bash
cd tilelang-ascend
bash install_npuir.sh --bishengir-path=/build/AscendNPU-IR/build/install
```

### 3. 开发工作流

```bash
# 挂载代码目录
docker run -it --rm -v /path/to/your/code:/workspace tilelang-ascend

# 在容器内
cd /workspace
# 开始开发...
```

## 构建参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CHIP_TYPE` | `A3` | Ascend 芯片类型 |
| `CANN_VERSION` | `8.5.0` | CANN 版本 |
| `REGION` | `ChinaMainland` | 地区设置（影响 Git 镜像） |

## 故障排除

### 构建失败

- 检查网络连接（特别是 Git 子模块下载）
- 确认系统支持 Docker 和 Ascend 环境

### 运行时问题

- 确保主机有 Ascend 设备访问权限
- 检查 Python 版本兼容性

### 性能优化

- 使用预编译二进制文件可显著减少安装时间
- 考虑使用 Docker layer 缓存加速重建

## 贡献

如果您发现问题或有改进建议，请提交 Issue 或 Pull Request。
