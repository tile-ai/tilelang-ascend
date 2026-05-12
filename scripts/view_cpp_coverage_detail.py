#!/usr/bin/env python3
"""查看 C++ 文件的具体行覆盖率"""

import sys
import re
from pathlib import Path

def parse_coverage_info(coverage_file, target_file):
    """解析 coverage.info 文件"""
    covered_lines = {}
    uncovered_lines = {}
    
    with open(coverage_file) as f:
        content = f.read()
    
    # 找到目标文件的 section
    pattern = f"SF:.*/{target_file}"
    match = re.search(pattern, content)
    if not match:
        print(f"File {target_file} not found in coverage.info")
        return
    
    # 找到这个文件的范围
    start = match.start()
    # 找到下一个 SF: 或文件结束
    next_sf = re.search(r"\nSF:", content[start+10:])
    if next_sf:
        end = start + 10 + next_sf.start()
    else:
        end = len(content)
    
    file_section = content[start:end]
    
    # 解析 DA 行
    for line in file_section.split("\n"):
        if line.startswith("DA:"):
            parts = line.split(":")[1].split(",")
            line_num = int(parts[0])
            count = int(parts[1])
            if count > 0:
                covered_lines[line_num] = count
            else:
                uncovered_lines[line_num] = 0
    
    return covered_lines, uncovered_lines

def print_summary(target_file, covered_lines, uncovered_lines):
    """打印摘要"""
    total = len(covered_lines) + len(uncovered_lines)
    covered = len(covered_lines)
    
    print(f"\n=== {target_file} Coverage Summary ===")
    print(f"Total measurable lines: {total}")
    print(f"Covered lines: {covered}")
    print(f"Uncovered lines: {len(uncovered_lines)}")
    print(f"Coverage rate: {covered/total*100:.2f}%")
    
    # 按行号排序，分组显示未覆盖的代码块
    print(f"\n=== Uncovered Code Blocks ===")
    if uncovered_lines:
        sorted_uncovered = sorted(uncovered_lines.keys())
        blocks = []
        start_line = sorted_uncovered[0]
        prev_line = start_line
        
        for line in sorted_uncovered[1:]:
            if line == prev_line + 1:
                prev_line = line
            else:
                blocks.append((start_line, prev_line))
                start_line = line
                prev_line = line
        blocks.append((start_line, prev_line))
        
        for start, end in blocks:
            if start == end:
                print(f"  Line {start}")
            else:
                print(f"  Lines {start}-{end} ({end-start+1} lines)")
    else:
        print("  All lines covered!")
    
    # 显示执行次数最多的行
    print(f"\n=== Most Executed Lines (Top 10) ===")
    sorted_covered = sorted(covered_lines.items(), key=lambda x: -x[1])
    for line, count in sorted_covered[:10]:
        print(f"  Line {line}: {count} executions")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/view_cpp_coverage_detail.py <filename>")
        print("Example: python scripts/view_cpp_coverage_detail.py ascend_lower_parallel_to_vector.cc")
        sys.exit(1)
    
    project_root = Path(__file__).resolve().parent.parent
    coverage_file = project_root / "coverage_data/coverage.info"
    target_file = sys.argv[1]
    
    if not coverage_file.exists():
        print(f"Error: {coverage_file} not found")
        sys.exit(1)
    
    result = parse_coverage_info(coverage_file, target_file)
    if result:
        print_summary(target_file, *result)
