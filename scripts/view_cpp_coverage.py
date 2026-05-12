#!/usr/bin/env python3
"""View specific C++ file coverage from .info file"""
import sys
from pathlib import Path

def parse_lcov_info(info_file, target_file):
    with open(info_file, 'r') as f:
        lines = f.readlines()
    
    in_target = False
    covered_lines = []
    uncovered_lines = []
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('SF:'):
            source_file = line.split(':', 1)[1]
            if target_file in source_file:
                in_target = True
                print(f"Found: {source_file}")
            else:
                in_target = False
        
        elif in_target and line.startswith('DA:'):
            parts = line[3:].split(',')
            line_num = int(parts[0])
            hit_count = int(parts[1])
            
            if hit_count > 0:
                covered_lines.append(line_num)
            else:
                uncovered_lines.append(line_num)
    
    return covered_lines, uncovered_lines

if len(sys.argv) < 3:
    print("Usage: python view_cpp_coverage.py <coverage.info> <target_file>")
    sys.exit(1)

info_file = sys.argv[1]
target_file = sys.argv[2]

covered, uncovered = parse_lcov_info(info_file, target_file)

print(f"\nCovered: {len(covered)} lines")
print(f"Uncovered: {len(uncovered)} lines")
print(f"Total: {len(covered) + len(uncovered)} lines")
print(f"Coverage: {len(covered) / (len(covered) + len(uncovered)) * 100:.2f}%")

if uncovered:
    print(f"\nUncovered lines: {uncovered[:50]}...")
