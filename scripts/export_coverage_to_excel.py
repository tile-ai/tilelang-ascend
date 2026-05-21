#!/usr/bin/env python3
"""
Export coverage data to Excel spreadsheet

Usage:
    python scripts/export_coverage_to_excel.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path
import pandas as pd


class CoverageExcelExporter:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.coverage_json = self.project_root / "coverage_data" / "coverage.json"
        self.coverage_info = self.project_root / "coverage_data" / "coverage.info"
        
    def collect_python_coverage(self):
        """Collect Python coverage from coverage.json"""
        if not self.coverage_json.exists():
            print("Warning: coverage.json not found")
            return []
        
        data = json.load(open(self.coverage_json))
        files = data.get("files", {})
        
        rows = []
        for file_path, file_data in files.items():
            covered = file_data["summary"]["covered_lines"]
            total = file_data["summary"]["num_statements"]
            percent = file_data["summary"]["percent_covered"]
            
            # Remove project root prefix
            rel_path = file_path.replace(str(self.project_root) + '/', '')
            
            rows.append({
                "语言": "Python",
                "文件路径": rel_path,
                "覆盖率": f"{percent:.2f}%",
                "已覆盖行数": covered,
                "总行数": total,
                "未覆盖行数": total - covered
            })
        
        return rows
    
    def collect_cpp_coverage(self):
        """Collect C++ coverage from coverage.info"""
        if not self.coverage_info.exists():
            print("Warning: coverage.info not found")
            return []
        
        content = self.coverage_info.read_text()
        
        # Parse all .cc files
        pattern = r'SF:(.+?\.cc)\n(.*?)end_of_record'
        matches = re.findall(pattern, content, re.DOTALL)
        
        rows = []
        for source_file, file_content in matches:
            # Filter only tilelang-ascend/src files
            if 'tilelang-ascend/src' not in source_file:
                continue
            
            # Parse DA lines (handle duplicates by taking max)
            line_exec = {}
            da_matches = re.findall(r'DA:(\d+),(\d+)', file_content)
            
            for line_num_str, exec_count_str in da_matches:
                line_num = int(line_num_str)
                exec_count = int(exec_count_str)
                
                if line_num in line_exec:
                    line_exec[line_num] = max(line_exec[line_num], exec_count)
                else:
                    line_exec[line_num] = exec_count
            
            # Calculate coverage
            covered_lines = sum(1 for count in line_exec.values() if count > 0)
            total_lines = len(line_exec)
            
            if total_lines > 0:
                percent = covered_lines / total_lines * 100
                rel_path = source_file.split('tilelang-ascend/')[-1]
                
                rows.append({
                    "语言": "C++",
                    "文件路径": rel_path,
                    "覆盖率": f"{percent:.2f}%",
                    "已覆盖行数": covered_lines,
                    "总行数": total_lines,
                    "未覆盖行数": total_lines - covered_lines
                })
        
        return rows
    
    def export_to_excel(self):
        """Export coverage data to Excel"""
        print("Collecting Python coverage...")
        python_rows = self.collect_python_coverage()
        print(f"  ✓ Found {len(python_rows)} Python files")
        
        print("Collecting C++ coverage...")
        cpp_rows = self.collect_cpp_coverage()
        print(f"  ✓ Found {len(cpp_rows)} C++ files")
        
        # Combine all data
        all_rows = python_rows + cpp_rows
        
        # Create DataFrame
        df = pd.DataFrame(all_rows)
        
        # Sort by language and coverage percentage
        df['覆盖率数值'] = df['覆盖率'].str.rstrip('%').astype(float)
        df = df.sort_values(['语言', '覆盖率数值'], ascending=[True, False])
        df = df.drop('覆盖率数值', axis=1)
        
        # Calculate overall statistics
        python_total_lines = sum(r['总行数'] for r in python_rows)
        python_covered_lines = sum(r['已覆盖行数'] for r in python_rows)
        python_percent = python_covered_lines / python_total_lines * 100 if python_total_lines > 0 else 0
        
        cpp_total_lines = sum(r['总行数'] for r in cpp_rows)
        cpp_covered_lines = sum(r['已覆盖行数'] for r in cpp_rows)
        cpp_percent = cpp_covered_lines / cpp_total_lines * 100 if cpp_total_lines > 0 else 0
        
        # Create summary DataFrame
        summary_data = [
            {"语言": "Python", "文件路径": "总体统计", "覆盖率": f"{python_percent:.2f}%",
             "已覆盖行数": python_covered_lines, "总行数": python_total_lines, "未覆盖行数": python_total_lines - python_covered_lines},
            {"语言": "C++", "文件路径": "总体统计", "覆盖率": f"{cpp_percent:.2f}%",
             "已覆盖行数": cpp_covered_lines, "总行数": cpp_total_lines, "未覆盖行数": cpp_total_lines - cpp_covered_lines},
        ]
        summary_df = pd.DataFrame(summary_data)
        
        # Write to Excel
        output_path = self.project_root / "coverage_report.xlsx"
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Write summary sheet
            summary_df.to_excel(writer, sheet_name='总体统计', index=False)
            
            # Write Python coverage sheet
            python_df = df[df['语言'] == 'Python']
            python_df.to_excel(writer, sheet_name='Python覆盖率', index=False)
            
            # Write C++ coverage sheet
            cpp_df = df[df['语言'] == 'C++']
            cpp_df.to_excel(writer, sheet_name='C++覆盖率', index=False)
            
            # Write all files sheet
            df.to_excel(writer, sheet_name='全部文件', index=False)
        
        print(f"\n✓ Excel report saved to: {output_path}")
        print(f"\n统计摘要:")
        print(f"  Python: {python_percent:.2f}% ({python_covered_lines}/{python_total_lines} 行)")
        print(f"  C++: {cpp_percent:.2f}% ({cpp_covered_lines}/{cpp_total_lines} 行)")
        print(f"  总计: {len(python_rows)} Python 文件, {len(cpp_rows)} C++ 文件")
        
        return output_path


if __name__ == "__main__":
    exporter = CoverageExcelExporter()
    exporter.export_to_excel()