#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Probe the SoC identity of the current environment.

获取链（npu-smi 的 Chip Name 作为 short-soc-version 不可信，禁止用于型号识别——issue #587）：
    1. full-soc-version（完整型号，如 Ascend950PR_9579）：
       首选 asys `info -r=hardware` 的 Chip Info；失败时回退 DSMI（dsmi_get_chip_info）。
    2. NpuArch（如 3510）：首选 asys 的 Arch Info；失败时回退 ini 文件（full-soc-version 精确匹配）。
    3. short-soc-version（如 Ascend950）/ CCE_AIV_version / variant_dir：来自 ini 文件。
       其中 short-soc-version 用于算子原型定义文件 xxx算子_def.cpp 的 AddConfig() 第一个参数
       （如 this->AICore().AddConfig("ascend950", aicConfig)）。

用法:
    python3 get_npu_arch.py           # 人读报告（含证据链注释）
    python3 get_npu_arch.py --raw     # 仅输出裸 NpuArch 数值，如 3510
    python3 get_npu_arch.py --json    # 机器可读 JSON

前置条件：source CANN 安装目录 set_env.sh（asys 与 libdrvdsmi_host.so 依赖其 PATH / LD_LIBRARY_PATH）。
"""

import ctypes
import json
import logging
import os
import platform
import re
import subprocess
import sys

_LOGGER = logging.getLogger(__name__)

MAX_CHIP_NAME = 32


class HalChipInfo(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_char * MAX_CHIP_NAME),
        ("name", ctypes.c_char * MAX_CHIP_NAME),
        ("version", ctypes.c_char * MAX_CHIP_NAME),
    ]


# ---------------------------------------------------------------------------
# CANN home 定位（沿用原逻辑）
# ---------------------------------------------------------------------------

def _derive_from_opp_path():
    opp = os.environ.get("ASCEND_OPP_PATH", "")
    if opp and opp.endswith("/opp"):
        toolkit = opp[:-4]
        if os.path.isdir(toolkit) and os.path.isdir(os.path.join(toolkit, "compiler")):
            return toolkit
    return None


def _resolve_toolkit_path(base_path):
    if os.path.isdir(os.path.join(base_path, "compiler")):
        return base_path

    toolkit_dir = os.path.join(base_path, "ascend-toolkit")
    if os.path.isdir(toolkit_dir):
        candidates = []
        for d in os.listdir(toolkit_dir):
            if d == "latest":
                continue
            dpath = os.path.join(toolkit_dir, d)
            if os.path.isdir(dpath) and os.path.isdir(os.path.join(dpath, "compiler")):
                candidates.append(d)
        candidates.sort(reverse=True)
        if candidates:
            return os.path.join(toolkit_dir, candidates[0])

        latest = os.path.join(toolkit_dir, "latest")
        if os.path.islink(latest):
            real = os.path.realpath(latest)
            if os.path.isdir(real) and os.path.isdir(os.path.join(real, "compiler")):
                return real

    cann_candidates = []
    for d in os.listdir(base_path):
        if not d.startswith("cann-"):
            continue
        dpath = os.path.join(base_path, d)
        if os.path.isdir(dpath) and os.path.isdir(os.path.join(dpath, "compiler")):
            cann_candidates.append(d)
    cann_candidates.sort(reverse=True)
    if cann_candidates:
        return os.path.join(base_path, cann_candidates[0])

    return None


def get_cann_home():
    derived = _derive_from_opp_path()
    for var in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME"):
        path = os.environ.get(var, "")
        if path and os.path.isdir(path):
            resolved = _resolve_toolkit_path(path)
            if resolved:
                return resolved

    if derived:
        return derived

    for var in ("ASCEND_HOME_PATH", "ASCEND_CANN_HOME"):
        path = os.environ.get(var, "")
        if path and os.path.isdir(path):
            resolved = _resolve_toolkit_path(path)
            if resolved:
                return resolved
    raise RuntimeError(
        "Cannot locate CANN toolkit installation. Set one of: "
        "ASCEND_TOOLKIT_HOME, ASCEND_HOME, ASCEND_HOME_PATH, ASCEND_CANN_HOME"
    )


def get_arch_dir():
    return f"{platform.machine()}-linux"


# ---------------------------------------------------------------------------
# 层 1：full-soc-version 探测
# ---------------------------------------------------------------------------

def _find_asys():
    """定位 asys 可执行文件：优先 ASCEND_HOME_PATH/tools 下，其次 PATH。"""
    if os.environ.get("ASCEND_HOME_PATH"):
        cand = os.path.join(
            os.environ["ASCEND_HOME_PATH"], "tools", "ascend_system_advisor", "asys", "asys"
        )
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    for cand in (
        os.path.join(os.path.expanduser("~"), "Ascend", "tools",
                     "ascend_system_advisor", "asys", "asys"),
        "/usr/local/Ascend/tools/ascend_system_advisor/asys/asys",
    ):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    # PATH 中查找（仅返回命令名，交由 subprocess 解析）
    from shutil import which
    return which("asys")


_ASYS_HW_CACHE = {}


def _run_asys_hardware(asys_cmd):
    """执行 asys info -r=hardware，返回原始 stdout；失败返回 None。进程内缓存一次。"""
    if asys_cmd in _ASYS_HW_CACHE:
        return _ASYS_HW_CACHE[asys_cmd]
    try:
        result = subprocess.run(
            [asys_cmd, "info", "-r=hardware"],
            capture_output=True, text=True, timeout=60,
        )
        output = result.stdout if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        output = None
    _ASYS_HW_CACHE[asys_cmd] = output
    return output


def _parse_asys_field(output, label):
    """从 asys 表格输出提取单值字段，返回 (value, evidence_line)。首列精确匹配 label。"""
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if cells and cells[0] == label and len(cells) >= 2:
                return cells[1], stripped
    return None, None


def _parse_full_soc_from_chip_info(chip_info_value):
    """Chip Info 值形如 'Ascend 950PR_9579 V100'，去掉版本段得到 'Ascend950PR_9579'。

    版本段识别：最后一个空格分隔且以 V 开头纯字母数字的 token，直接丢弃。
    """
    tokens = chip_info_value.split()
    if len(tokens) < 2:
        return chip_info_value.replace(" ", "")
    if len(tokens) >= 3 and re.fullmatch(r"V\w+", tokens[-1]):
        tokens = tokens[:-1]
    return "".join(tokens)  # 'Ascend' + '950PR_9579'


def probe_full_soc_via_asys():
    """首选：asys info -r=hardware 的 Chip Info。返回 full_soc 或 None。

    注意：部分 asys 版本无 Arch Info 字段，NpuArch 层另行探测。
    """
    asys_cmd = _find_asys()
    if not asys_cmd:
        _LOGGER.debug("asys not found")
        return None
    output = _run_asys_hardware(asys_cmd)
    if not output:
        _LOGGER.debug("asys info -r=hardware returned nothing")
        return None
    chip_info, evidence = _parse_asys_field(output, "Chip Info")
    if not chip_info:
        return None
    full_soc = _parse_full_soc_from_chip_info(chip_info)
    if not full_soc or not full_soc.startswith(("Ascend", "Kirin")):
        return None
    return full_soc


def probe_full_soc_via_dsmi():
    """备选：DSMI dsmi_get_chip_info（TTK dsmi_interface 模式，单设备查询）。返回 full_soc 或 None。
    """
    try:
        dll = ctypes.CDLL("libdrvdsmi_host.so")
        device_count = (ctypes.c_int * 1)()
        dll.dsmi_get_device_count.restype = ctypes.c_int
        if dll.dsmi_get_device_count(device_count) != 0 or device_count[0] <= 0:
            _LOGGER.debug("dsmi_get_device_count failed or no device")
            return None

        device_id = ctypes.c_int(0)
        info = HalChipInfo()
        dll.dsmi_get_chip_info.restype = ctypes.c_int
        if dll.dsmi_get_chip_info(device_id, ctypes.byref(info)) != 0:
            _LOGGER.debug("dsmi_get_chip_info failed")
            return None
    except (OSError, AttributeError) as e:
        # OSError: 库不可加载；AttributeError: 驱动版本不匹配导致 dsmi_* 符号缺失
        _LOGGER.debug("DSMI probe unavailable: %s", e)
        return None

    chip_type = info.type.decode().strip()
    chip_name = info.name.decode().strip()
    full_soc = chip_type + chip_name
    if not full_soc.startswith(("Ascend", "Kirin")):
        return None
    return full_soc


def probe_full_soc():
    """full-soc-version：asys Chip Info 首选，DSMI 兜底。返回 (full_soc, source) 或 None。"""
    result = probe_full_soc_via_asys()
    if result:
        return result, "asys"
    result = probe_full_soc_via_dsmi()
    if result:
        return result, "dsmi"
    return None


# ---------------------------------------------------------------------------
# 层 2：NpuArch 探测
# ---------------------------------------------------------------------------

def probe_npu_arch_via_asys():
    """首选：asys info -r=hardware 的 Arch Info（部分版本无此字段）。

    返回 (npu_arch_str, evidence_line) 或 None。
    """
    asys_cmd = _find_asys()
    if not asys_cmd:
        return None
    output = _run_asys_hardware(asys_cmd)
    if not output:
        return None
    arch, evidence = _parse_asys_field(output, "Arch Info")
    if arch is None:
        return None
    arch = arch.strip()
    if not re.fullmatch(r"\d{4}", arch):
        return None
    return arch, evidence


# ---------------------------------------------------------------------------
# 层 3：ini 查询（NpuArch 兜底 + short-soc-version / CCE_AIV_version / variant_dir）
# ---------------------------------------------------------------------------

def _ini_has_soc_version(ini_path, full_soc):
    """检查单个 ini 是否含 SoC_version=full_soc 精确行（全文件逐行匹配）。"""
    try:
        with open(ini_path, "r", errors="ignore") as f:
            for line in f:
                if line.strip() == f"SoC_version={full_soc}":
                    return True
    except OSError:
        pass
    return False


def find_ini_for_soc(cann_home, full_soc):
    """按 SoC_version 字段精确匹配 ini（lookup_arch_variant 模式，禁止前缀模糊）。"""
    config_dir = os.path.join(cann_home, get_arch_dir(), "data", "platform_config")
    if not os.path.isdir(config_dir):
        return None
    for name in sorted(os.listdir(config_dir)):
        if not name.endswith(".ini"):
            continue
        ini_path = os.path.join(config_dir, name)
        if _ini_has_soc_version(ini_path, full_soc):
            return ini_path
    return None


def read_ini_fields(ini_path):
    """读取 [version] 段关键字段。"""
    fields = {}
    in_version_section = False
    with open(ini_path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line == "[version]":
                in_version_section = True
                continue
            if in_version_section and line.startswith("["):
                break
            if in_version_section and "=" in line:
                key, val = line.split("=", 1)
                fields[key.strip()] = val.strip()
    return fields


def variant_dir_from_aiv(ccec_aiv_version):
    """dav-c310-vec -> dav_c310；非该格式返回原值。"""
    m = re.match(r"^dav-([a-z0-9]+)-vec$", ccec_aiv_version or "")
    if m:
        return f"dav_{m.group(1)}"
    return ccec_aiv_version


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def _apply_ini_fields(result, ini_path):
    """将 ini 字段写入 result：short-soc-version / CCE_AIV_version / variant_dir；NpuArch 兜底。"""
    fields = read_ini_fields(ini_path)
    result["ini_path"] = ini_path
    result["short_soc"] = fields.get("Short_SoC_version")
    result["ccec_aiv_version"] = fields.get("CCEC_AIV_version")
    result["variant_dir"] = variant_dir_from_aiv(fields.get("CCEC_AIV_version", ""))
    ini_arch = fields.get("NpuArch")
    if not result["npu_arch"]:
        if ini_arch:
            result["npu_arch"], result["npu_arch_source"] = ini_arch, "ini"
    elif ini_arch and ini_arch != result["npu_arch"]:
        result["warnings"].append(
            f"NpuArch 不一致：asys={result['npu_arch']}，ini={ini_arch}（以 asys 为准，请核对）"
        )


def _probe_npu_count():
    """设备数（asys NPU Count），失败返回 None。"""
    asys_cmd = _find_asys()
    if not asys_cmd:
        return None
    output = _run_asys_hardware(asys_cmd)
    if not output:
        return None
    count, _ = _parse_asys_field(output, "NPU Count")
    if count is None:
        return None
    digits = re.sub(r"\D", "", count)
    return int(digits) if digits else None


def probe_all():
    """按层级探测，返回 dict（含每项来源与证据）。"""
    result = {
        "full_soc": None,
        "full_soc_source": None,
        "npu_arch": None,
        "npu_arch_source": None,
        "short_soc": None,
        "ccec_aiv_version": None,
        "variant_dir": None,
        "ini_path": None,
        "npu_count": None,
        "warnings": [],
    }

    # 层 1：full-soc-version
    probed = probe_full_soc()
    if not probed:
        result["warnings"].append(
            "full-soc-version 探测失败：asys 与 DSMI 均不可用或无设备。"
            "请确认已 source CANN 安装目录下的 set_env.sh。"
        )
        result["npu_count"] = _probe_npu_count()
        return result
    result["full_soc"], result["full_soc_source"] = probed

    # 层 2：NpuArch（asys 首选）
    arch = probe_npu_arch_via_asys()
    if arch:
        result["npu_arch"], result["npu_arch_source"] = arch[0], "asys"

    # 层 3：ini（short-soc-version / CCE_AIV_version / variant_dir；NpuArch 兜底）
    try:
        cann_home = get_cann_home()
    except RuntimeError as e:
        result["warnings"].append(f"{e}；short-soc-version/CCE_AIV_version/variant_dir 无法从 ini 获取")
        cann_home = None

    if cann_home:
        ini_path = find_ini_for_soc(cann_home, result["full_soc"])
        if ini_path:
            _apply_ini_fields(result, ini_path)
        else:
            result["warnings"].append(
                f"platform_config 下无 SoC_version={result['full_soc']} 的 ini，"
                "short-soc-version/CCE_AIV_version/variant_dir 未知"
            )

    if not result["npu_arch"]:
        result["warnings"].append("NpuArch 探测失败：asys Arch Info 与 ini 均未提供")

    result["npu_count"] = _probe_npu_count()
    return result


# ---------------------------------------------------------------------------
# 输出（打印风格参考 lookup_arch_variant.sh：值 + 行尾用途注释，强化记忆）
# ---------------------------------------------------------------------------

def _format_report(r):
    lines = []
    lines.append(f"full-soc-version={r['full_soc']} (source={r['full_soc_source']})")
    if r["npu_arch"] is not None:
        lines.append(f"NpuArch={r['npu_arch']} (source={r['npu_arch_source']})   "
                     "# __NPU_ARCH__ 宏分支 / --npu-arch 编译值；读头文件只走该宏分支")
    if r["short_soc"]:
        lines.append(f"short-soc-version={r['short_soc']}   "
                     "# 算子原型定义 xxx算子_def.cpp 中 AddConfig() 第一个参数（如 "
                     "this->AICore().AddConfig(\"ascend950\", aicConfig)）")
    if r["ccec_aiv_version"]:
        lines.append(f"CCE_AIV_version={r['ccec_aiv_version']}")
    if r["variant_dir"]:
        lines.append(f"variant_dir={r['variant_dir']}   "
                     f"# 读CANN安装目录下源码时只看 **/{r['variant_dir']}/ 下的文件")
    if r["ini_path"]:
        lines.append(f"ini={r['ini_path']}")
    if r["npu_count"] is not None:
        lines.append(f"npu_count={r['npu_count']}")
    lines.append("dav-%s" % r["npu_arch"] if r["npu_arch"] else "dav-UNKNOWN")
    for w in r["warnings"]:
        lines.append(f"WARNING: {w}")
    return "\n".join(lines)


def main() -> int:
    args = sys.argv[1:]
    raw_mode = "--raw" in args
    json_mode = "--json" in args

    r = probe_all()

    if raw_mode:
        if r["npu_arch"]:
            print(r["npu_arch"])
            return 0
        for w in r["warnings"]:
            print(f"WARNING: {w}", file=sys.stderr)
        return 1

    if json_mode:
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0 if (r["full_soc"] and r["npu_arch"]) else 1

    if not r["full_soc"]:
        print(_format_report(r))
        return 1

    print(_format_report(r))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
