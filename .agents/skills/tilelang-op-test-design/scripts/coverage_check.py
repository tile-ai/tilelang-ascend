# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""覆盖自检 checker —— tilelang-op-test-design 测试覆盖矩阵门禁。

判定 test_{op}.py 是否覆盖了 references/coverage-matrix.md 规定的【强制维度】。

用法:
    python coverage_check.py path/to/test_{op}.py [--proto path/to/proto.yaml]

判定来源（与 coverage-matrix.md 一致）:
    应覆盖集 = 类别强制维度（COVERAGE_CATEGORY 或启发式）∪ proto 的 dtype/attr 派生维度
    实际覆盖集 = L1_CASES 的 tags 汇总 ∪ COVERAGE_MANIFEST 显式计数
    豁免 = COVERAGE_NA（仅对"可豁免"维度生效；强制维度写豁免仍判 MISS）

退出码: 全 PASS/N/A → 0；任一强制维度 MISS → 1。
本脚本仅依赖标准库（proto.yaml 解析为可选；缺 PyYAML 时降级为纯文本扫描）。
"""

import argparse
import ast
import logging
import os
import re
import sys

logger = logging.getLogger(__name__)

# ---- 各类别的【强制维度】与【可豁免维度】（对应 coverage-matrix.md §二）----
BASE_REQUIRED = {
    "D-SHAPE-ALIGNED",
    "D-SHAPE-EDGE",
    "D-SPECIAL-ZERO",
    "D-EXC-DTYPE",
    "D-EXC-SHAPE",
}
VALRANGE = {"D-VALRANGE-S", "D-VALRANGE-M", "D-VALRANGE-L", "D-VALRANGE-ASYM"}
TAIL = {"D-SHAPE-TAIL-1", "D-SHAPE-TAIL-MID", "D-SHAPE-PRIME"}
FP_SPECIAL = {"D-SPECIAL-INF", "D-SPECIAL-NAN", "D-SPECIAL-DBOUND"}

# 每类别: (额外强制维度集, 可豁免维度集)
CATEGORY_RULES = {
    "Activation": (VALRANGE | FP_SPECIAL, TAIL),
    "Reduction": (VALRANGE | FP_SPECIAL | TAIL, set()),
    "Softmax": (VALRANGE | FP_SPECIAL | TAIL, set()),
    "Normalization": (VALRANGE | FP_SPECIAL | TAIL, set()),
    "GEMM": (TAIL, FP_SPECIAL | {"D-VALRANGE-L"}),
    "Fusion": (VALRANGE | FP_SPECIAL | TAIL, set()),
    "Quant": (TAIL, FP_SPECIAL),
    "PureInteger": (set(), FP_SPECIAL | VALRANGE),
    "Vector": (VALRANGE | FP_SPECIAL, TAIL),  # 通用兜底
}


def _safe_literal(value):
    """ast.literal_eval，失败返回 None。"""
    try:
        return ast.literal_eval(value)
    except Exception:
        return None


def _module_assign(tree, name):
    """提取模块级 `name = <literal>` 的字面量值，失败返回 None。"""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if name in target_names:
            return _safe_literal(node.value)
    return None


def _str_const(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _list_tags(node):
    """一个 list 字面量里以 D- 开头的字符串（即用例的 tags）。"""
    for elt in node.elts:
        v = _str_const(elt)
        if v and v.startswith("D-"):
            yield v


def _collect_tags(tree):
    """收集所有 list 字面量里以 D- 开头的字符串（用例 tags）的计数。"""
    counts = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.List):
            continue
        for v in _list_tags(node):
            counts[v] = counts.get(v, 0) + 1
    return counts


def parse_example(path):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    manifest = _module_assign(tree, "COVERAGE_MANIFEST") or {}
    na = _module_assign(tree, "COVERAGE_NA") or {}
    category = _module_assign(tree, "COVERAGE_CATEGORY")
    tag_counts = _collect_tags(tree)
    # 合并 tags 汇总与显式 manifest（取较大值，避免重复计数低估）
    actual = dict(tag_counts)
    for k, v in manifest.items() if isinstance(manifest, dict) else []:
        if isinstance(v, int):
            actual[k] = max(actual.get(k, 0), v)
    return src, actual, (na if isinstance(na, dict) else {}), category


def detect_category(src, declared):
    if declared in CATEGORY_RULES:
        return declared, True
    s = src.lower()
    has_mm = bool(re.search(r"@\s|matmul|t\.gemm|\bmma\b", s))
    has_softmax = "softmax" in s or ("exp" in s and "sum" in s)
    has_norm = any(k in s for k in ("layernorm", "rms_norm", "rmsnorm", "group_norm", "var(", "mean("))
    has_quant = any(k in s for k in ("quant", "int8", "scale_out"))
    if has_mm and (has_softmax or has_norm):
        return "Fusion", False
    if has_mm:
        return "GEMM", False
    if has_softmax:
        return "Softmax", False
    if has_norm:
        return "Normalization", False
    if has_quant:
        return "Quant", False
    return "Vector", False  # 保守兜底


def load_proto_dtypes_attrs(proto_path):
    dtypes, attrs = set(), set()
    if not proto_path:
        return dtypes, attrs
    try:
        text = open(proto_path, encoding="utf-8").read()
    except OSError:
        return dtypes, attrs
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        op = data.get("operator", data) if isinstance(data, dict) else {}
        for inp in op.get("inputs", []) or []:
            for dt in inp.get("dtype", []) or []:
                dtypes.add(str(dt))
        for at in op.get("attrs", []) or []:
            if at.get("name"):
                attrs.add(str(at["name"]))
    except Exception:
        # 降级：纯文本扫描。dtype 关键词全局扫；attrs name 仅限 attrs: 区块，避免误抓 inputs/outputs 名。
        for dt in ("float16", "float32", "bfloat16", "int8", "int32", "int64", "uint8"):
            if dt in text:
                dtypes.add(dt)
        attrs.update(re.findall(r"-\s*name:\s*([A-Za-z_]\w*)", _attrs_block(text)))
    return dtypes, attrs


def _attrs_block(text):
    """从 proto 文本中切出 `attrs:` 到下一个同级键（inputs/outputs/schema/note...）之间的片段。"""
    lines = text.splitlines()
    start = None
    indent = 0
    for i, ln in enumerate(lines):
        m = re.match(r"^(\s*)attrs:\s*$", ln)
        if m:
            start, indent = i + 1, len(m.group(1))
            break
    if start is None:
        return ""
    out = []
    for ln in lines[start:]:
        # 同级或更外层的新键 → attrs 区块结束
        if re.match(rf"^\s{{0,{indent}}}[A-Za-z_]+:", ln) and not ln.strip().startswith("-"):
            break
        out.append(ln)
    return "\n".join(out)


_DTYPE_SHORT = {"float16": "fp16", "float32": "fp32", "bfloat16": "bf16"}


def build_required(category, dtypes, attrs):
    extra, exempt = CATEGORY_RULES[category]
    required = set(BASE_REQUIRED) | set(extra)
    for dt in dtypes:
        required.add(f"D-DTYPE-{_DTYPE_SHORT.get(dt, dt)}")
    for at in attrs:
        required.add(f"D-PARAM-{at}")
    return required, exempt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("example", help="examples/{op}/test_{op}.py")
    ap.add_argument("--proto", default=None, help="examples/{op}/proto.yaml（可选，用于派生 dtype/attr 维度）")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    src, actual, na, declared = parse_example(args.example)
    op = re.sub(r"^test_|\.py$", "", args.example.replace("\\", "/").split("/")[-1])
    # 类别启发式：test_{op}.py 不含 kernel 代码，补读同目录 kernel {op}.py 增强判定
    # （COVERAGE_CATEGORY 已声明时优先用声明，不依赖此启发式）
    kernel_src = ""
    kernel_path = os.path.join(os.path.dirname(args.example), op + ".py")
    if os.path.isfile(kernel_path):
        try:
            kernel_src = open(kernel_path, encoding="utf-8").read()
        except OSError:
            kernel_src = ""
    category, explicit = detect_category(src + "\n" + kernel_src, declared)
    dtypes, attrs = load_proto_dtypes_attrs(args.proto)
    required, exempt = build_required(category, dtypes, attrs)

    logger.info(f"== Coverage Matrix: {op} ==")
    logger.info(f"   category={category}" + ("" if explicit else " (启发式推断, 建议在文件中声明 COVERAGE_CATEGORY)"))
    if not dtypes:
        logger.info("   NOTE: 未提供/解析到 proto dtype，D-DTYPE-* 仅按文件内 tags 校验，可能漏检缺失 dtype。")

    n_pass = n_na = n_miss = 0
    for dim in sorted(required):
        got = actual.get(dim, 0)
        if got >= 1:
            logger.info(f"[PASS] {dim:22s} need>=1 got {got}")
            n_pass += 1
        elif dim in na and dim in exempt:
            logger.info(f"[N/A ] {dim:22s} reason: {na[dim]}")
            n_na += 1
        else:
            extra = "  <-- 强制维度写了豁免也无效" if (dim in na and dim not in exempt) else ""
            logger.info(f"[MISS] {dim:22s} need>=1 got 0{extra}")
            n_miss += 1

    verdict = "PASS" if n_miss == 0 else "FAIL"
    logger.info(f"COVERAGE: {n_pass} PASS / {n_miss} MISS / {n_na} N/A  -> {verdict}" + (" (exit 1)" if n_miss else ""))
    sys.exit(1 if n_miss else 0)


if __name__ == "__main__":
    main()
