"""Audit precision checks in every examples Python file."""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys


EXCLUDED = {"__init__.py", "utils.py", "golden.py", "setup.py"}
PRECISION_MARKERS = ("matched_ratio", "max_abs", "PRECISION_FAIL", "check_precision", "_check_precision")
SPECIAL_MARKERS = ("isfinite", "isnan", "isinf")


def read_files(root: pathlib.Path):
    files = [p for p in (root / "examples").rglob("*.py") if p.name not in EXCLUDED]
    precision = []
    entries = []
    syntax_errors = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(marker in text for marker in PRECISION_MARKERS):
            precision.append((path, text))
        if re.search(r"__main__|def\s+test_|pytest|PRECISION_", text):
            entries.append(path)
        try:
            ast.parse(text, filename=str(path))
        except SyntaxError as error:
            syntax_errors.append((path, error))
    return files, precision, entries, syntax_errors


def audit_precision(path: pathlib.Path, text: str):
    failures = []
    floating = any(token in text for token in ("rtol", "atol", "isfinite", "isnan"))
    integer = bool(re.search(r"int8|int16|int32|int64|uint8", text))
    if floating:
        if "0.99" not in text:
            failures.append("missing required matched ratio 0.99")
        if not re.search(r"max_abs(?:_error)?|max_diff|cap", text):
            failures.append("missing maximum absolute error hard cap")
        if "isfinite" not in text or "isnan" not in text or "isinf" not in text:
            failures.append("missing complete NaN/Inf structure checks")
    if integer and "torch.equal" not in text and "==" not in text:
        failures.append("missing integer exact comparison")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    parser.add_argument("--allow-uncovered", action="store_true")
    parser.add_argument("--strict", action="store_true", help="fail on precision or coverage findings")
    args = parser.parse_args()

    files, precision, entries, syntax_errors = read_files(args.root)
    entry_set = set(entries)
    uncovered = [path for path, _ in precision if path not in entry_set]
    auditable = [(path, text) for path, text in precision if re.search(r"def\s+(?:_?check_precision|get_precision)\s*\(", text)]
    manual_review = [(path, text) for path, text in precision if path not in {item[0] for item in auditable}]
    failures = [(path, audit_precision(path, text)) for path, text in auditable]
    failures = [(path, problems) for path, problems in failures if problems]

    print(f"all_python_files={len(files)}")
    print(f"precision_files={len(precision)}")
    print(f"auditable_precision_files={len(auditable)}")
    print(f"manual_review_precision_files={len(manual_review)}")
    print(f"runnable_candidates={len(entries)}")
    print(f"covered_precision_files={len(precision) - len(uncovered)}")
    print(f"uncovered_precision_files={len(uncovered)}")
    print(f"syntax_errors={len(syntax_errors)}")
    print(f"precision_audit_failures={len(failures)}")
    for path, problems in failures:
        print(f"FAIL {path.relative_to(args.root)}: {'; '.join(problems)}")
    for path, error in syntax_errors:
        print(f"SYNTAX_FAIL {path.relative_to(args.root)}: {error}")
    if uncovered:
        print("UNCOVERED_PRECISION")
        for path in uncovered:
            print(path.relative_to(args.root))

    failed = bool(syntax_errors or (args.strict and (failures or (uncovered and not args.allow_uncovered))))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
