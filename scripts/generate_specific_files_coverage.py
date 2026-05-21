#!/usr/bin/env python3
"""
Generate coverage report for specific files

Usage:
    python3 scripts/generate_specific_files_coverage.py
"""

import json
import re
from pathlib import Path
from datetime import datetime

# 需要统计的文件列表
FILES_TO_ANALYZE = [
    # C++ files
    "src/transform/ascend_collect_buffer_shape.cc",
    "src/transform/ascend_host.cc",
    "src/transform/cross_core_pipeline.cc",
    "src/transform/ascend_vid_reduction.cc",
    "src/transform/ascend_lower_parallel_to_vector.cc",
    "src/op/ascend.cc",
    "src/transform/ascend_memory_planning.cc",
    "src/transform/ascend_sync_insert.cc",
    "src/transform/ascend_combinecv.cc",
    "src/transform/allocate_tmp_buffer.cc",
    "src/transform/ascend_infer_buffer_scope.cc",
    "src/target/codegen_ascend.cc",
    "src/transform/pipeline_planning.cc",
    "src/transform/inject_pipeline.cc",
    "src/target/codegen_ascend_pto.cc",
    "src/transform/ascend_storage_rewrite.cc",
    "src/transform/flatten_buffer.cc",
    "src/ir.cc",
    "src/transform/lower_tile_op.cc",
    "src/transform/legalize_safe_memory_access.cc",
    
    # Python files
    "tilelang/transform/pass_config.py",
    "tilelang/language/tir/ir.py",
    "tilelang/intrinsics/ascend_layout.py",
    "tilelang/language/ascend.py",
    "tilelang/carver/template/matmul.py",
    "tilelang/language/pipeline.py",
    "tilelang/language/customize.py",
    "tilelang/language/parallel.py",
    "tilelang/autotuner/tuner.py",
    "tilelang/language/reduce_ascend.py",
    "tilelang/language/ascend_tile.py",
    "tilelang/jit/kernel.py",
    "tilelang/cache/kernel_cache.py",
    "tilelang/jit/adapter/libgen.py",
    "tilelang/engine/phase.py",
]


def parse_python_coverage(coverage_json_path: Path, files_to_analyze: list):
    """Parse Python coverage from coverage.json"""
    if not coverage_json_path.exists():
        print(f"Warning: {coverage_json_path} not found")
        return {}
    
    data = json.load(open(coverage_json_path))
    files_data = data.get("files", {})
    
    result = {}
    for file_path in files_to_analyze:
        if not file_path.endswith('.py'):
            continue
        
        # 尝试多种路径匹配方式
        matched_key = None
        for key in files_data.keys():
            if file_path in key or key.endswith(file_path):
                matched_key = key
                break
        
        if matched_key:
            file_data = files_data[matched_key]
            summary = file_data["summary"]
            covered = summary["covered_lines"]
            total = summary["num_statements"]
            percent = summary["percent_covered"]
            
            result[file_path] = {
                "covered": covered,
                "total": total,
                "percent": percent,
                "missing": total - covered
            }
        else:
            # 文件未在覆盖率数据中
            result[file_path] = {
                "covered": 0,
                "total": 0,
                "percent": 0.0,
                "missing": 0,
                "note": "未找到覆盖率数据"
            }
    
    return result


def parse_cpp_coverage(coverage_info_path: Path, files_to_analyze: list):
    """Parse C++ coverage from coverage.info"""
    if not coverage_info_path.exists():
        print(f"Warning: {coverage_info_path} not found")
        return {}
    
    content = coverage_info_path.read_text()
    
    # 解析所有文件
    all_files_data = {}
    pattern = r'SF:(.+?\.cc)\n(.*?)end_of_record'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for source_file, file_content in matches:
        # 只处理 tilelang-ascend/src 下的文件
        if 'tilelang-ascend/src' not in source_file:
            continue
        
        # 提取文件名
        rel_path = source_file.split('tilelang-ascend/')[-1]
        
        # 解析 DA 行（行覆盖率）
        line_exec = {}
        da_matches = re.findall(r'DA:(\d+),(\d+)', file_content)
        
        for line_num_str, exec_count_str in da_matches:
            line_num = int(line_num_str)
            exec_count = int(exec_count_str)
            # 对于重复行，取最大执行次数
            if line_num in line_exec:
                line_exec[line_num] = max(line_exec[line_num], exec_count)
            else:
                line_exec[line_num] = exec_count
        
        if len(line_exec) > 0:
            covered_lines = sum(1 for count in line_exec.values() if count > 0)
            total_lines = len(line_exec)
            percent = covered_lines / total_lines * 100 if total_lines > 0 else 0
            
            all_files_data[rel_path] = {
                "covered": covered_lines,
                "total": total_lines,
                "percent": percent,
                "missing": total_lines - covered_lines
            }
    
    # 提取需要的文件
    result = {}
    for file_path in files_to_analyze:
        if not file_path.endswith('.cc'):
            continue
        
        if file_path in all_files_data:
            result[file_path] = all_files_data[file_path]
        else:
            # 文件未在覆盖率数据中
            result[file_path] = {
                "covered": 0,
                "total": 0,
                "percent": 0.0,
                "missing": 0,
                "note": "未找到覆盖率数据"
            }
    
    return result


def generate_markdown_report(python_data: dict, cpp_data: dict, output_path: Path):
    """Generate Markdown coverage report"""
    
    lines = []
    lines.append("# 核心文件覆盖率统计报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 总体统计
    python_total = sum(d["total"] for d in python_data.values())
    python_covered = sum(d["covered"] for d in python_data.values())
    python_percent = python_covered / python_total * 100 if python_total > 0 else 0
    
    cpp_total = sum(d["total"] for d in cpp_data.values())
    cpp_covered = sum(d["covered"] for d in cpp_data.values())
    cpp_percent = cpp_covered / cpp_total * 100 if cpp_total > 0 else 0
    
    total_lines = python_total + cpp_total
    total_covered = python_covered + cpp_covered
    total_percent = total_covered / total_lines * 100 if total_lines > 0 else 0
    
    lines.append("## 总体覆盖率")
    lines.append("")
    lines.append("| 语言 | 文件数 | 总行数 | 已覆盖 | 未覆盖 | 覆盖率 |")
    lines.append("|------|--------|--------|--------|--------|--------|")
    lines.append(f"| Python | {len(python_data)} | {python_total} | {python_covered} | {python_total - python_covered} | {python_percent:.2f}% |")
    lines.append(f"| C++ | {len(cpp_data)} | {cpp_total} | {cpp_covered} | {cpp_total - cpp_covered} | {cpp_percent:.2f}% |")
    lines.append(f"| **总计** | **{len(python_data) + len(cpp_data)}** | **{total_lines}** | **{total_covered}** | **{total_lines - total_covered}** | **{total_percent:.2f}%** |")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # C++ 文件详情（按覆盖率排序）
    lines.append("## C++ 文件覆盖率详情")
    lines.append("")
    
    cpp_sorted = sorted(cpp_data.items(), key=lambda x: -x[1]["percent"])
    
    # 分组：高覆盖率、中等覆盖率、低覆盖率
    cpp_high = [(f, d) for f, d in cpp_sorted if d["percent"] >= 80]
    cpp_medium = [(f, d) for f, d in cpp_sorted if 50 <= d["percent"] < 80]
    cpp_low = [(f, d) for f, d in cpp_sorted if d["percent"] < 50]
    
    if cpp_high:
        lines.append("### ✅ 高覆盖率（≥80%）")
        lines.append("")
        lines.append("| 文件 | 覆盖率 | 已覆盖 | 总行数 | 未覆盖 |")
        lines.append("|------|--------|--------|--------|--------|")
        for file_path, data in cpp_high:
            fname = file_path.split('/')[-1]
            lines.append(f"| {fname} | {data['percent']:.2f}% | {data['covered']} | {data['total']} | {data['missing']} |")
        lines.append("")
    
    if cpp_medium:
        lines.append("### ⚠️ 中等覆盖率（50%-80%）")
        lines.append("")
        lines.append("| 文件 | 覆盖率 | 已覆盖 | 总行数 | 未覆盖 |")
        lines.append("|------|--------|--------|--------|--------|")
        for file_path, data in cpp_medium:
            fname = file_path.split('/')[-1]
            lines.append(f"| {fname} | {data['percent']:.2f}% | {data['covered']} | {data['total']} | {data['missing']} |")
        lines.append("")
    
    if cpp_low:
        lines.append("### ❌ 低覆盖率（<50%）")
        lines.append("")
        lines.append("| 文件 | 覆盖率 | 已覆盖 | 总行数 | 未覆盖 |")
        lines.append("|------|--------|--------|--------|--------|")
        for file_path, data in cpp_low:
            fname = file_path.split('/')[-1]
            note = data.get('note', '')
            lines.append(f"| {fname} | {data['percent']:.2f}% | {data['covered']} | {data['total']} | {data['missing']} | {note} |")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Python 文件详情（按覆盖率排序）
    lines.append("## Python 文件覆盖率详情")
    lines.append("")
    
    py_sorted = sorted(python_data.items(), key=lambda x: -x[1]["percent"])
    
    py_high = [(f, d) for f, d in py_sorted if d["percent"] >= 80]
    py_medium = [(f, d) for f, d in py_sorted if 50 <= d["percent"] < 80]
    py_low = [(f, d) for f, d in py_sorted if d["percent"] < 50]
    
    if py_high:
        lines.append("### ✅ 高覆盖率（≥80%）")
        lines.append("")
        lines.append("| 文件 | 覆盖率 | 已覆盖 | 总行数 | 未覆盖 |")
        lines.append("|------|--------|--------|--------|--------|")
        for file_path, data in py_high:
            fname = file_path.split('/')[-1]
            lines.append(f"| {fname} | {data['percent']:.2f}% | {data['covered']} | {data['total']} | {data['missing']} |")
        lines.append("")
    
    if py_medium:
        lines.append("### ⚠️ 中等覆盖率（50%-80%）")
        lines.append("")
        lines.append("| 文件 | 覆盖率 | 已覆盖 | 总行数 | 未覆盖 |")
        lines.append("|------|--------|--------|--------|--------|")
        for file_path, data in py_medium:
            fname = file_path.split('/')[-1]
            lines.append(f"| {fname} | {data['percent']:.2f}% | {data['covered']} | {data['total']} | {data['missing']} |")
        lines.append("")
    
    if py_low:
        lines.append("### ❌ 低覆盖率（<50%）")
        lines.append("")
        lines.append("| 文件 | 覆盖率 | 已覆盖 | 总行数 | 未覆盖 | 备注 |")
        lines.append("|------|--------|--------|--------|--------|------|")
        for file_path, data in py_low:
            fname = file_path.split('/')[-1]
            note = data.get('note', '')
            lines.append(f"| {fname} | {data['percent']:.2f}% | {data['covered']} | {data['total']} | {data['missing']} | {note} |")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # 全部文件列表
    lines.append("## 全部文件覆盖率列表")
    lines.append("")
    lines.append("| 文件 | 语言 | 覆盖率 | 已覆盖 | 总行数 | 未覆盖 |")
    lines.append("|------|------|--------|--------|--------|--------|")
    
    # 合并并按覆盖率排序
    all_files_sorted = []
    for f, d in cpp_data.items():
        all_files_sorted.append((f, "C++", d))
    for f, d in python_data.items():
        all_files_sorted.append((f, "Python", d))
    
    all_files_sorted.sort(key=lambda x: -x[2]["percent"])
    
    for file_path, lang, data in all_files_sorted:
        fname = file_path.split('/')[-1]
        note = data.get('note', '')
        if note:
            lines.append(f"| {fname} | {lang} | {data['percent']:.2f}% | {data['covered']} | {data['total']} | {data['missing']} | {note} |")
        else:
            lines.append(f"| {fname} | {lang} | {data['percent']:.2f}% | {data['covered']} | {data['total']} | {data['missing']} |")
    
    # 写入文件
    output_path.write_text('\n'.join(lines), encoding='utf-8')
    
    print(f"✓ 报告已生成: {output_path}")
    print()
    print("=== 统计摘要 ===")
    print(f"总覆盖率: {total_percent:.2f}% ({total_covered}/{total_lines})")
    print(f"  - Python: {python_percent:.2f}% ({python_covered}/{python_total})")
    print(f"  - C++: {cpp_percent:.2f}% ({cpp_covered}/{cpp_total})")


def main():
    project_root = Path(__file__).resolve().parent.parent
    coverage_json = project_root / "coverage_data" / "coverage.json"
    coverage_info = project_root / "coverage_data" / "coverage.info"
    output_path = project_root / "核心文件覆盖率统计.md"
    
    print("解析覆盖率数据...")
    python_data = parse_python_coverage(coverage_json, FILES_TO_ANALYZE)
    cpp_data = parse_cpp_coverage(coverage_info, FILES_TO_ANALYZE)
    
    generate_markdown_report(python_data, cpp_data, output_path)


if __name__ == "__main__":
    main()