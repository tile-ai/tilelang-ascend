#!/usr/bin/env python3
"""查看 Python 文件的详细覆盖率信息"""

import json
import sys
from pathlib import Path
from typing import List, Tuple


def group_lines(lines: List[int]) -> List[Tuple[int, int]]:
    """将连续行号分组"""
    if not lines:
        return []
    
    groups = []
    start = lines[0]
    prev = lines[0]
    
    for line in lines[1:]:
        if line == prev + 1:
            prev = line
        else:
            groups.append((start, prev))
            start = line
            prev = line
    groups.append((start, prev))
    
    return groups


def show_file_coverage(coverage_json: Path, file_path: str):
    """显示文件的详细覆盖信息"""
    data = json.load(open(coverage_json))
    
    if file_path not in data['files']:
        print(f"❌ 文件不存在: {file_path}")
        return
    
    file_data = data['files'][file_path]
    summary = file_data['summary']
    
    print(f"\n{'='*60}")
    print(f"文件: {file_path}")
    print(f"{'='*60}")
    print(f"覆盖率: {summary['percent_covered_display']}%")
    print(f"已覆盖行: {summary['covered_lines']}/{summary['num_statements']}")
    print(f"分支覆盖: {summary['covered_branches']}/{summary['num_branches']} ({summary['percent_branches_covered_display']}%)")
    
    # 未覆盖行分组
    print(f"\n--- 未覆盖行 ({len(file_data['missing_lines'])} 行) ---")
    groups = group_lines(file_data['missing_lines'])
    for g in groups[:20]:  # 只显示前20组
        if g[0] == g[1]:
            print(f"  第 {g[0]} 行")
        else:
            print(f"  第 {g[0]}-{g[1]} 行 ({g[1]-g[0]+1} 行)")
    
    if len(groups) > 20:
        print(f"  ... 还有 {len(groups)-20} 组")
    
    # 未覆盖分支
    print(f"\n--- 未覆盖分支 ({len(file_data['missing_branches'])} 个) ---")
    for branch in file_data['missing_branches'][:10]:
        print(f"  {branch[0]} -> {branch[1]}")
    
    if len(file_data['missing_branches']) > 10:
        print(f"  ... 还有 {len(file_data['missing_branches'])-10} 个分支")


def check_line(coverage_json: Path, file_path: str, line_num: int):
    """检查特定行是否覆盖"""
    data = json.load(open(coverage_json))
    
    if file_path not in data['files']:
        print(f"❌ 文件不存在: {file_path}")
        return
    
    file_data = data['files'][file_path]
    
    if line_num in file_data['executed_lines']:
        print(f"✅ 第 {line_num} 行: 已覆盖")
    elif line_num in file_data['missing_lines']:
        print(f"❌ 第 {line_num} 行: 未覆盖")
    else:
        print(f"⚪ 第 {line_num} 行: 非可执行行")


def list_low_coverage_files(coverage_json: Path, threshold: float = 50.0):
    """列出覆盖率低于阈值的文件"""
    data = json.load(open(coverage_json))
    
    low_files = []
    for file_path, file_data in data['files'].items():
        percent = file_data['summary']['percent_covered']
        if percent < threshold:
            low_files.append((file_path, percent))
    
    low_files.sort(key=lambda x: x[1])
    
    print(f"\n=== 覆盖率低于 {threshold}% 的文件 ({len(low_files)} 个) ===")
    for file_path, percent in low_files[:20]:
        print(f"  {file_path}: {percent:.2f}%")


if __name__ == "__main__":
    coverage_json = Path("coverage_data/coverage.json")
    
    if len(sys.argv) == 1:
        # 默认：列出低覆盖率文件
        list_low_coverage_files(coverage_json)
    
    elif sys.argv[1] == "file":
        # 查看特定文件
        if len(sys.argv) < 3:
            print("用法: python view_file_coverage.py file <文件路径>")
            sys.exit(1)
        show_file_coverage(coverage_json, sys.argv[2])
    
    elif sys.argv[1] == "line":
        # 检查特定行
        if len(sys.argv) < 4:
            print("用法: python view_file_coverage.py line <文件路径> <行号>")
            sys.exit(1)
        check_line(coverage_json, sys.argv[2], int(sys.argv[3]))
    
    else:
        print("用法:")
        print("  python view_file_coverage.py                    # 列出低覆盖率文件")
        print("  python view_file_coverage.py file <文件路径>    # 查看文件详情")
        print("  python view_file_coverage.py line <文件路径> <行号>  # 检查特定行")