#!/usr/bin/env python3
"""
Coverage Report Generator for TileLang-Ascend
Generates comprehensive coverage reports including Python and C++ coverage

Key features:
- Correctly handles duplicate DA lines in coverage.info (takes max execution count)
- Filters only tilelang-ascend/src for accurate C++ coverage
- Provides detailed per-file coverage breakdown
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple

class CoverageReportGenerator:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.coverage_data_dir = self.project_root / "coverage_data"
        self.report_dir = self.project_root / "coverage_reports"
        self.coverage_json = self.coverage_data_dir / "coverage.json"
        self.coverage_info = self.project_root / "coverage_data" / "coverage.info"
        
        self.coverage_data_dir.mkdir(exist_ok=True)
        self.report_dir.mkdir(exist_ok=True)
    
    def generate_report(self):
        """Generate complete coverage report"""
        print("Generating coverage report...")
        
        # Parse Python coverage
        python_data = self._collect_python_coverage()
        
        # Parse C++ coverage (only tilelang-ascend/src)
        cpp_data = self._parse_cpp_coverage()
        
        # Generate Markdown report
        self._generate_markdown_report(python_data, cpp_data)
        
        print("✓ Markdown report saved to coverage_report.md")
    
    def _collect_python_coverage(self) -> Dict:
        """Collect Python coverage from coverage.json"""
        if not self.coverage_json.exists():
            print("Warning: coverage.json not found")
            return {}
        
        try:
            data = json.load(open(self.coverage_json))
            files = data.get("files", {})
            
            summary = {
                "total_lines": 0,
                "covered_lines": 0,
                "files": {}
            }
            
            for file_path, file_data in files.items():
                covered = file_data["summary"]["covered_lines"]
                total = file_data["summary"]["num_statements"]
                percent = file_data["summary"]["percent_covered"]
                
                summary["total_lines"] += total
                summary["covered_lines"] += covered
                summary["files"][file_path] = {
                    "covered": covered,
                    "total": total,
                    "percent": percent
                }
            
            summary["percent"] = (summary["covered_lines"] / summary["total_lines"] * 100) if summary["total_lines"] > 0 else 0
            
            return summary
        except Exception as e:
            print(f"Error parsing coverage.json: {e}")
            return {}
    
    def _parse_cpp_coverage(self) -> Dict:
        """Parse C++ coverage from coverage.info, filtering only tilelang-ascend/src
        
        Key fix: Correctly handle duplicate DA lines by taking max execution count
        """
        if not self.coverage_info.exists():
            print("Warning: coverage.info not found")
            return {}
        
        try:
            content = self.coverage_info.read_text()
            
            # Parse all source files
            files_data = {}
            pattern = r'SF:(.+?\.cc)\n(.*?)end_of_record'
            matches = re.findall(pattern, content, re.DOTALL)
            
            for source_file, file_content in matches:
                # Filter only tilelang-ascend/src files
                if 'tilelang-ascend/src' not in source_file:
                    continue
                
                # Parse line coverage data
                # Key fix: Handle duplicate DA lines by taking max execution count
                line_exec = {}
                
                da_matches = re.findall(r'DA:(\d+),(\d+)', file_content)
                for line_num_str, exec_count_str in da_matches:
                    line_num = int(line_num_str)
                    exec_count = int(exec_count_str)
                    
                    # Take max execution count for duplicate lines
                    if line_num in line_exec:
                        line_exec[line_num] = max(line_exec[line_num], exec_count)
                    else:
                        line_exec[line_num] = exec_count
                
                # Calculate coverage
                covered_lines = sum(1 for count in line_exec.values() if count > 0)
                total_lines = len(line_exec)
                
                if total_lines > 0:
                    percent = covered_lines / total_lines * 100
                    rel_path = source_file.split('tilelang-ascend/')[-1] if 'tilelang-ascend/' in source_file else source_file
                    files_data[rel_path] = {
                        "covered": covered_lines,
                        "total": total_lines,
                        "percent": percent,
                        "line_exec": line_exec
                    }
            
            # Calculate overall summary
            total_covered = sum(f["covered"] for f in files_data.values())
            total_lines = sum(f["total"] for f in files_data.values())
            
            overall_percent = (total_covered / total_lines * 100) if total_lines > 0 else 0
            
            return {
                "total_lines": total_lines,
                "covered_lines": total_covered,
                "percent": overall_percent,
                "files": files_data
            }
        except Exception as e:
            print(f"Error parsing coverage.info: {e}")
            return {}
    
    def _generate_markdown_report(self, python_data: Dict, cpp_data: Dict):
        """Generate Markdown coverage report"""
        report_path = self.project_root / "coverage_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# TileLang Coverage Report\n\n")
            
            # Python coverage
            if python_data and python_data.get('total_lines', 0) > 0:
                f.write("## Python Coverage\n\n")
                f.write(f"**Overall Coverage**: {python_data['percent']:.2f}%\n\n")
                f.write(f"- Total Lines: {python_data['total_lines']}\n")
                f.write(f"- Covered Lines: {python_data['covered_lines']}\n\n")
                
                # Files with low coverage
                f.write("### Files with Low Coverage (< 50%)\n\n")
                low_cov = [(k, v) for k, v in python_data['files'].items() if v['percent'] < 50]
                low_cov.sort(key=lambda x: x[1]['percent'], reverse=True)
                
                for file_path, data in low_cov[:20]:
                    rel_path = file_path.replace(str(self.project_root) + '/', '')
                    f.write(f"- {rel_path}: {data['percent']:.2f}% ({data['covered']}/{data['total']})\n")
                f.write("\n")
            
            # C++ coverage
            if cpp_data and cpp_data.get('total_lines', 0) > 0:
                f.write("## C++ Coverage\n\n")
                f.write(f"**Overall Coverage**: {cpp_data['percent']:.2f}%\n\n")
                f.write(f"- Total Lines: {cpp_data['total_lines']}\n")
                f.write(f"- Covered Lines: {cpp_data['covered_lines']}\n\n")
                
                # Files with low coverage
                f.write("### Files with Low Coverage (< 50%)\n\n")
                low_cov = [(k, v) for k, v in cpp_data['files'].items() if v['percent'] < 50]
                low_cov.sort(key=lambda x: x[1]['percent'], reverse=True)
                
                for file_path, data in low_cov[:20]:
                    f.write(f"- {file_path}: {data['percent']:.2f}% ({data['covered']}/{data['total']})\n")
                f.write("\n")
                
                # Files with high coverage
                f.write("### Files with High Coverage (> 80%)\n\n")
                high_cov = [(k, v) for k, v in cpp_data['files'].items() if v['percent'] > 80]
                high_cov.sort(key=lambda x: x[1]['percent'], reverse=True)
                
                for file_path, data in high_cov[:10]:
                    f.write(f"- {file_path}: {data['percent']:.2f}% ({data['covered']}/{data['total']})\n")
                f.write("\n")
            
            f.write("---\nGenerated by `scripts/generate_coverage_report.py`\n")


if __name__ == "__main__":
    generator = CoverageReportGenerator()
    generator.generate_report()