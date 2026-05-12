#!/usr/bin/env python3
"""Check coverage.json for specific files"""
import json
import sys
from pathlib import Path

coverage_file = sys.argv[1] if len(sys.argv) > 1 else 'coverage_data/coverage.json'

try:
    data = json.load(open(coverage_file))
    files = data.get('files', {})
    
    print(f"\n=== Coverage Analysis: {Path(coverage_file).name} ===")
    print(f"Total files: {len(files)}")
    
    # Check key files
    key_files = [
        'tilelang/language/ascend_tile.py',
        'tilelang/language/reduce_ascend.py',
        'tilelang/language/reduce.py',
    ]
    
    print("\n=== Key Files Coverage ===")
    for key in key_files:
        if key in files:
            v = files[key]
            covered = v['summary']['covered_lines']
            total = v['summary']['num_statements']
            percent = v['summary']['percent_covered']
            print(f"{key}: {covered}/{total} lines, {percent:.2f}%")
        else:
            print(f"{key}: NOT FOUND")
    
    # Overall
    total_lines = sum(v['summary']['num_statements'] for v in files.values())
    covered_lines = sum(v['summary']['covered_lines'] for v in files.values())
    overall_percent = (covered_lines / total_lines * 100) if total_lines > 0 else 0
    
    print(f"\n=== Overall ===")
    print(f"Total: {covered_lines}/{total_lines} = {overall_percent:.2f}%")

except Exception as e:
    print(f"Error: {e}")
