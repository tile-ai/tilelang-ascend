#!/usr/bin/env python3
"""
Generate simple HTML coverage reports for C++ files from coverage.info

Usage:
    python scripts/generate_cpp_coverage_html.py <file_name>
    python scripts/generate_cpp_coverage_html.py --list

Examples:
    python scripts/generate_cpp_coverage_html.py ascend_vid_reduction.cc
    python scripts/generate_cpp_coverage_html.py ascend_lower_parallel_to_vector
"""

import re
import sys
import os
from pathlib import Path
from collections import defaultdict


def parse_single_file(info_file: Path, target_file: str):
    """解析单个文件的覆盖率（使用精确的正则表达式）"""
    content = info_file.read_text()

    # 精确匹配目标文件，避免匹配到其他文件
    pattern = r"SF:([^\n]+" + re.escape(target_file) + r"\.cc)\n((?:(?!SF:)(?!end_of_record).)*?)end_of_record"
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        return None, {}

    source_file = match.group(1)
    file_content = match.group(2)

    # 只解析 DA 行（行覆盖率数据）
    line_exec = defaultdict(int)
    da_matches = re.findall(r"DA:(\d+),(\d+)", file_content)

    for ln, cnt in da_matches:
        line_exec[int(ln)] = max(line_exec[int(ln)], int(cnt))

    return source_file, line_exec


def generate_cpp_html_report(source_file: str, line_exec: dict, output_file: Path):
    """生成 C++ coverage HTML 报告（正确的格式化内容）"""

    # 找到源文件路径
    paths = [
        source_file,
        source_file.split("tilelang-ascend/")[-1] if "tilelang-ascend/" in source_file else source_file,
    ]

    source_path = None
    for p in paths:
        if os.path.exists(p):
            source_path = p
            break

    if not source_path:
        return None

    with open(source_path) as f:
        source_lines = f.readlines()

    # 统计
    covered = sum(1 for c in line_exec.values() if c > 0)
    total = len(line_exec)
    percent = covered / total * 100 if total > 0 else 0

    # 生成正确的 HTML（格式化内容，不是原始文本）
    html = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        f"<title>C++ Coverage: {Path(source_file).name}</title>",
        "<style>",
        "body{font-family:Consolas,monospace;margin:20px;background:#f5f5f5}",
        ".summary{background:#fff;padding:15px;margin:20px;border-radius:5px}",
        ".run{background:#c8e6c9;padding:2px 5px}",
        ".mis{background:#ffcdd2;padding:2px 5px}",
        ".pln{background:#fff;padding:2px 5px}",
        ".line{display:block;font-size:13px}",
        ".lineno{display:inline-block;width:50px;color:#666;text-align:right;margin-right:10px}",
        ".count{display:inline-block;width:80px;color:#006;font-weight:bold;text-align:right;margin-right:10px}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>C++ Coverage Report</h1>",
        f"<h2>{source_file}</h2>",
        '<div class="summary">',
        f'<b style="font-size:18px">Coverage: {percent:.2f}%</b><br>',
        f"Executed: {covered} / {total} lines<br>",
        f"Missing: {total - covered} lines<br><br>",
        '<span style="color:green;font-weight:bold">■ Green = Executed</span><br>',
        '<span style="color:red;font-weight:bold">■ Red = Not Executed</span><br>',
        '<span style="color:gray">■ White = Non-executable</span><br><br>',
        "<b>Number after line is execution count</b>",
        "</div>",
        '<pre style="font-family:Consolas;background:#fff;padding:15px">',
    ]

    # 处理每一行代码（格式化，不是原始文本）
    for i, line in enumerate(source_lines, 1):
        if i in line_exec:
            cnt = line_exec[i]
            css = "run" if cnt > 0 else "mis"
            count = str(cnt)
        else:
            css = "pln"
            count = ""

        # 转义 HTML（避免显示原始 coverage.info 文本）
        escaped = line.rstrip().replace("<", "&lt;").replace(">", "&gt;")
        html.append(f'<span class="line {css}"><span class="lineno">{i}</span><span class="count">{count:>8}</span>{escaped}</span>')

    html.extend(["</pre>", "</body>", "</html>"])

    # 保存
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(html))

    return output_file


def list_all_files(info_file: Path):
    """列出所有文件"""
    content = info_file.read_text()

    # 提取所有 tilelang-ascend/src 的文件
    pattern = r"SF:([^\n]+tilelang-ascend/src/[^\n]+\.cc)\n"
    matches = re.findall(pattern, content)

    # 解析每个文件
    all_files = {}
    for source_file in matches:
        # 找到对应的 DA 数据
        file_pattern = r"SF:" + re.escape(source_file) + r"\n((?:(?!SF:)(?!end_of_record).)*?)end_of_record"
        file_match = re.search(file_pattern, content, re.DOTALL)

        if file_match:
            file_content = file_match.group(1)
            line_exec = defaultdict(int)
            da_matches = re.findall(r"DA:(\d+),(\d+)", file_content)

            for ln, cnt in da_matches:
                line_exec[int(ln)] = max(line_exec[int(ln)], int(cnt))

            covered = sum(1 for c in line_exec.values() if c > 0)
            total = len(line_exec)
            percent = covered / total * 100 if total > 0 else 0

            rel_path = source_file.split("tilelang-ascend/")[-1]
            all_files[rel_path] = {"covered": covered, "total": total, "percent": percent}

    return all_files


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    info_file = project_root / "coverage_data" / "coverage.info"
    output_dir = project_root / "coverage_reports" / "cpp_html"

    if not info_file.exists():
        print(f"❌ coverage.info not found: {info_file}")
        sys.exit(1)

    if len(sys.argv) == 1 or sys.argv[1] == "--list":
        # 列出所有文件
        all_files = list_all_files(info_file)

        print("C++ Coverage Summary:")
        print("=" * 80)

        sorted_files = sorted(all_files.items(), key=lambda x: x[1]["percent"], reverse=True)

        for rel_path, data in sorted_files:
            status = "✓" if data["percent"] >= 80 else "❌"
            print(f"{status} {rel_path}: {data['percent']:.2f}% ({data['covered']}/{data['total']})")

        print()
        print(f"Total: {len(all_files)} files")
        print()
        print("Usage:")
        print("  python scripts/generate_cpp_coverage_html.py <file_name>")

    else:
        # 生成特定文件
        target_name = sys.argv[1]

        print(f"Generating HTML for: {target_name}")

        source_file, line_exec = parse_single_file(info_file, target_name)

        if not source_file:
            print(f"❌ File not found: {target_name}")
            sys.exit(1)

        # 生成文件名
        rel_path = source_file.split("tilelang-ascend/")[-1]
        safe_name = rel_path.replace("/", "_").replace(".cc", "")
        output_file = output_dir / f"{safe_name}.html"

        result = generate_cpp_html_report(source_file, line_exec, output_file)

        if result:
            covered = sum(1 for c in line_exec.values() if c > 0)
            total = len(line_exec)
            print(f"✓ Generated: {result}")
            print(f"  Coverage: {covered}/{total} = {covered / total * 100:.2f}%")
            print(f"  Missing: {total - covered} lines")
            print(f"  File size: {result.stat().st_size / 1024:.1f} KB")
        else:
            print("❌ Failed to generate HTML")
