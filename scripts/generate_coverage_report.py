#!/usr/bin/env python3
"""
Coverage Report Generator for TileLang-Ascend
Generates comprehensive coverage reports including Python and C++ coverage

Key fixes:
- Uses build_dir (not tilelang_objs_dir) to collect all .gcda files
- Timeout=300 to avoid timeout issues
- Dynamic project root detection (no hardcoded paths)
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any

class CoverageReportGenerator:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.coverage_data_dir = self.project_root / "coverage_data"
        self.report_dir = self.project_root / "coverage_reports"
        self.coverage_json = self.coverage_data_dir / "coverage.json"
        self.coverage_info = self.coverage_data_dir / "coverage.info"
        
        self.coverage_data_dir.mkdir(exist_ok=True)
        self.report_dir.mkdir(exist_ok=True)
    
    def generate_report(self):
        """Generate complete coverage report"""
        print("Generating coverage report...")
        
        # Collect C++ coverage first (if not exists)
        if not self.coverage_info.exists():
            self._collect_cpp_coverage()
        
        # Collect Python coverage
        python_data = self._collect_python_coverage()
        
        # Parse C++ coverage
        cpp_data = self._parse_cpp_coverage()
        
        # Generate Markdown report
        self._generate_markdown_report(python_data, cpp_data)
        
        print("✓ Coverage report generated")
    
    def _collect_cpp_coverage(self):
        """Collect C++ coverage from build directory"""
        print("Collecting C++ coverage data...")
        
        build_dir = self.project_root / "build"
        if not build_dir.exists():
            print("Warning: Build directory not found")
            return
        
        # Key fix: Use build_dir (not tilelang_objs_dir) to collect ALL .gcda files
        cmd_capture = [
            "lcov",
            "--capture",
            "--directory", str(build_dir),  # Key fix: entire build directory
            "--output-file", str(self.coverage_info),
            "--rc", "lcov_branch_coverage=0",
            "--ignore-errors", "source,graph",
            "--no-checksum"
        ]
        
        print(f"Running: lcov --capture --directory {build_dir}")
        result = subprocess.run(
            cmd_capture,
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=300  # Key fix: 300 seconds timeout
        )
        
        if result.returncode != 0:
            print(f"Warning: lcov capture failed: {result.stderr}")
            return
        
        print(f"✓ Captured C++ coverage: {self.coverage_info.stat().st_size} bytes")
    
    def _collect_python_coverage(self) -> Dict:
        """Collect Python coverage from coverage.json"""
        if not self.coverage_json.exists():
            print("Warning: coverage.json not found, generating from .coverage file")
            
            # Try to generate from .coverage file
            coverage_file = self.coverage_data_dir / ".coverage"
            if coverage_file.exists():
                cmd = [
                    "coverage", "json",
                    "-o", str(self.coverage_json),
                    "--include=tilelang/*,examples/*"
                ]
                subprocess.run(cmd, cwd=self.project_root)
            
            if not self.coverage_json.exists():
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
        """Parse C++ coverage from coverage.info"""
        if not self.coverage_info.exists():
            return {}
        
        try:
            # Parse lcov .info file
            result = subprocess.run(
                ["lcov", "--summary", str(self.coverage_info)],
                capture_output=True, text=True
            )
            
            # Parse summary
            lines_info = {}
            for line in result.stdout.split('\n'):
                if 'lines' in line and '%' in line:
                    # Parse: "lines......: 75.2% (419 of 557 lines)"
                    import re
                    match = re.search(r'(\d+\.\d+)% \((\d+) of (\d+) lines\)', line)
                    if match:
                        lines_info = {
                            'percent': float(match.group(1)),
                            'covered': int(match.group(2)),
                            'total': int(match.group(3))
                        }
            
            return {'lines': lines_info}
        except Exception as e:
            print(f"Error parsing coverage.info: {e}")
            return {}
    
    def _generate_markdown_report(self, python_data: Dict, cpp_data: Dict):
        """Generate Markdown coverage report"""
        report_path = self.project_root / "coverage_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# TileLang Coverage Report\n\n")
            
            # Python coverage
            if python_data:
                f.write("## Python Coverage\n\n")
                f.write(f"**Overall Coverage**: {python_data['percent']:.2f}%\n\n")
                f.write(f"- Total Lines: {python_data['total_lines']}\n")
                f.write(f"- Covered Lines: {python_data['covered_lines']}\n\n")
                
                # Top files by coverage
                f.write("### Files with Low Coverage (< 50%)\n\n")
                low_cov = [(k, v) for k, v in python_data['files'].items() if v['percent'] < 50]
                low_cov.sort(key=lambda x: x[1]['percent'], reverse=True)
                
                for file_path, data in low_cov[:20]:
                    f.write(f"- {file_path}: {data['percent']:.2f}% ({data['covered']}/{data['total']})\n")
            
            # C++ coverage
            if cpp_data and cpp_data.get('lines'):
                f.write("\n## C++ Coverage\n\n")
                lines = cpp_data['lines']
                f.write(f"**Overall Coverage**: {lines['percent']:.2f}%\n\n")
                f.write(f"- Total Lines: {lines['total']}\n")
                f.write(f"- Covered Lines: {lines['covered']}\n\n")
            
            f.write("\n---\nGenerated by `scripts/generate_coverage_report.py`\n")
        
        print(f"✓ Markdown report saved to {report_path}")

if __name__ == "__main__":
    generator = CoverageReportGenerator()
    generator.generate_report()
