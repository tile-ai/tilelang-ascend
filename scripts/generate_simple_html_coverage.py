#!/usr/bin/env python3
"""
Generate simple HTML coverage reports (single file, no external dependencies)

Usage:
    python scripts/generate_simple_html_coverage.py [file_path]

    # Generate for specific file
    python scripts/generate_simple_html_coverage.py tilelang/language/ascend_tile.py

    # Generate for all files below threshold
    python scripts/generate_simple_html_coverage.py --all --threshold 80
"""

import json
import sys
from pathlib import Path


def generate_simple_html(coverage_json: Path, file_path: str, output_dir: Path):
    """Generate a simple HTML coverage report for a single file"""

    # 1. Read coverage data from coverage.json
    data = json.load(open(coverage_json))

    if file_path not in data["files"]:
        print(f"❌ File not found in coverage data: {file_path}")
        return None

    file_data = data["files"][file_path]

    # 2. Read source code
    try:
        source_lines = open(file_path).readlines()
    except FileNotFoundError:
        print(f"❌ Source file not found: {file_path}")
        return None

    # 3. Build HTML content
    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        f"<title>Coverage: {file_path}</title>",
        "<style>",
        # CSS styles (embedded in HTML, no external files)
        "body { font-family: Consolas, monospace; margin: 20px; background: #f5f5f5; }",
        "h1 { color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }",
        ".summary { background: #fff; padding: 15px; margin: 20px; border-radius: 5px; }",
        ".run { background: #c8e6c9 !important; }",  # Green - executed
        ".mis { background: #ffcdd2 !important; }",  # Red - not executed
        ".pln { background: #fff; }",  # White - non-executable
        ".line { padding: 2px; display: block; font-size: 13px; }",
        ".lineno { display: inline-block; width: 50px; color: #999; text-align: right; margin-right: 10px; }",
        "</style>",
        "</head>",
        "<body>",
        # Title
        f"<h1>Coverage Report: {file_path}</h1>",
        # Summary
        '<div class="summary">',
        "<p><b>Coverage: {:.2f}%</b></p>".format(file_data["summary"]["percent_covered"]),
        "<p>Executed: {} / {} lines</p>".format(file_data["summary"]["covered_lines"], file_data["summary"]["num_statements"]),
        "<p>Missing: {} lines</p>".format(file_data["summary"]["missing_lines"]),
        '<p><span style="color: green;">■ Green = Executed</span> | <span style="color: red;">■ Red = Not Executed</span></p>',
        "</div>",
        # Source code
        '<pre style="font-family: Consolas, monospace; background: #fff; padding: 10px;">',
    ]

    # 4. Process each line
    for i, line in enumerate(source_lines, 1):
        # Determine coverage status
        if i in file_data["executed_lines"]:
            css_class = "run"  # Green
        elif i in file_data["missing_lines"]:
            css_class = "mis"  # Red
        else:
            css_class = "pln"  # White (comments, blank lines)

        # Escape HTML characters
        escaped_line = line.rstrip().replace("<", "&lt;").replace(">", "&gt;")

        # Add line with styling
        html_lines.append(f'<span class="line {css_class}"><span class="lineno">{i}</span>{escaped_line}</span>')

    html_lines.extend(["</pre>", "</body>", "</html>"])

    # 5. Save HTML file
    safe_name = file_path.replace("/", "_").replace(".py", "")
    output_file = output_dir / f"{safe_name}.html"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(html_lines))

    return output_file


def generate_all_below_threshold(coverage_json: Path, output_dir: Path, threshold: float = 80.0):
    """Generate HTML for all files below coverage threshold"""

    data = json.load(open(coverage_json))
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for file_path, file_data in data["files"].items():
        # Skip high coverage files
        if file_data["summary"]["percent_covered"] > threshold:
            continue

        result = generate_simple_html(coverage_json, file_path, output_dir)
        if result:
            count += 1
            print(f"  ✓ {file_path}: {file_data['summary']['percent_covered']:.1f}%")

    return count


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    coverage_json = project_root / "coverage_data" / "coverage.json"
    output_dir = project_root / "coverage_reports" / "simple_html"

    if not coverage_json.exists():
        print(f"❌ coverage.json not found: {coverage_json}")
        sys.exit(1)

    # Parse arguments
    if len(sys.argv) == 1:
        # Default: generate all files below 80%
        print("Generating HTML reports for files below 80% coverage...")
        count = generate_all_below_threshold(coverage_json, output_dir, threshold=80.0)
        print(f"\n✓ Generated {count} reports in {output_dir}")

    elif sys.argv[1] == "--all":
        # Generate for all files
        threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
        print("Generating HTML reports for all files...")
        count = generate_all_below_threshold(coverage_json, output_dir, threshold=threshold)
        print(f"\n✓ Generated {count} reports in {output_dir}")

    else:
        # Generate for specific file
        file_path = sys.argv[1]
        print(f"Generating HTML report for: {file_path}")
        result = generate_simple_html(coverage_json, file_path, output_dir)
        if result:
            print(f"✓ Generated: {result}")
