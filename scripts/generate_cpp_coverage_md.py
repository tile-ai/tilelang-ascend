#!/usr/bin/env python3
"""Generate C++ coverage markdown table from lcov output."""

import subprocess
import re
import sys


def parse_lcov_list(info_file):
    result = subprocess.run(["lcov", "--list", info_file], capture_output=True, text=True)

    lines = result.stdout.split("\n")

    print("| File | Lines | Functions |")
    print("|------|-------|----------|")

    for line in lines:
        if "Filename" in line or "===" in line or "Reading" in line or line.strip().startswith("[/"):
            continue
        if not line.strip():
            continue

        match = re.match(r"^(\S+)\s+\|(\S+)\s+(\d+)\|(\S+)\s+(\d+)", line.strip())
        if match:
            filename = match.group(1)
            lines_rate = match.group(2)
            funcs_rate = match.group(4)
            print(f"| {filename} | {lines_rate} | {funcs_rate} |")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 generate_cpp_coverage_md.py <coverage.info>")
        sys.exit(1)
    parse_lcov_list(sys.argv[1])
