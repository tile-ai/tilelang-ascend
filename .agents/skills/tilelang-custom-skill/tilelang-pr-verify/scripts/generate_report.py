#!/usr/bin/env python3
"""Generate a Markdown PR verification report by comparing before/after logs.

Usage:
  python generate_report.py --before-log <before.log> --after-log <after.log>
      --pr-url <url> --pr-title <title> --backend <backend>
      --before-sha <sha> --after-sha <sha> --needs-rebuild <true|false>
      --output <path>

Parses run_examples.sh output logs, compares per-script results, and produces
a Markdown report highlighting:
  - NEW FAIL (regressions introduced by the PR)
  - FIXED (issues fixed by the PR)
  - Still failing (pre-existing issues)
  - Added/Removed scripts (PR changed the example set)
"""

import argparse
import os
import re
import sys
from collections import OrderedDict


def classify_failure(last_lines):
    """Classify the failure type from the last few lines of output.

    NOTE: 本函数与 tilelang-run-examples/scripts/export_to_excel.py 中 classify_failure 保持同步；
    修改失败分类规则时两处必须同步更新。
    """
    text = "\n".join(last_lines)
    if "Compilation Failed!" in text:
        return "编译失败", "bisheng 编译报错"
    if "Unsupport SyncAll in pto backend" in text:
        return "pto不支持", "Unsupport SyncAll in pto backend"
    if "Unresolved call Op(tl.ascend_reinterpretcast)" in text:
        return "pto不支持", "Unresolved call Op(tl.ascend_reinterpretcast)"
    if "Downcast" in text and "failed" in text:
        return "内部错误", "Downcast type mismatch"
    if "Mismatched elements" in text:
        match = re.search(r"Mismatched elements: (.+?)$", text, re.MULTILINE)
        detail = match.group(1).strip() if match else "精度不匹配"
        return "精度不匹配", detail
    if "accuracy:" in text or "The precision is not correct" in text:
        match = re.search(r"accuracy: ([\d.]+)", text)
        detail = f"accuracy {match.group(1)}" if match else "精度不匹配"
        return "精度不匹配", detail
    if "vector::reserve" in text or "length_error" in text:
        return "NPU设备错误", "std::length_error: vector::reserve"
    if "aicore exception" in text or "rtDeviceSynchronizeWithTimeout" in text:
        return "NPU设备错误", "aicore exception / npuSynchronizeDevice failed"
    if "open device" in text and "failed" in text:
        return "NPU设备错误", "open device failed"
    # timeout: exit code 124 from coreutils `timeout`, or explicit TIMEOUT marker
    if "Exit: 124" in text or "Timeout after" in text or "TIMEOUT" in text.upper():
        match = re.search(r"Timeout after (\d+)s", text)
        detail = f"Timeout after {match.group(1)}s" if match else "task-timeout exceeded"
        return "超时(TIMEOUT)", detail
    if "Exit code 139" in text or "Exit: 139" in text:
        return "段错误(Segfault)", "Exit code 139"
    if "Exit code" in text:
        return "运行时错误", text.split("Exit")[-1].strip()[:60]
    return "未知", text[:80]


def parse_log(log_path):
    """Parse run_examples.sh output log and extract per-script results.

    Returns OrderedDict of {script: (status, fail_type, fail_detail)}.

    NOTE: 本函数与 tilelang-run-examples/scripts/export_to_excel.py 中 parse_log 保持同步；
    解析逻辑修改时两处必须同步更新。
    """
    results = OrderedDict()
    if not os.path.exists(log_path):
        print(f"Error: Log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    with open(log_path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    pattern = re.compile(
        r"\[(PASSED|FAILED|TIMEOUT)\]\s+\[([^\]]+)\]\s+\[([\d.]+)s\]\s+(.+?)(?:\s+\(Exit:\s*\d+\))?\n"
        r"(?:\s+Last line: (.+?)\n)?",
        re.MULTILINE,
    )

    for match in pattern.finditer(content):
        status = match.group(1)
        time_block = match.group(2) or ""
        elapsed = match.group(3) or ""
        script = match.group(4).strip()
        if " ~ " in time_block:
            start_time, end_time = time_block.split(" ~ ", 1)
        else:
            start_time = end_time = time_block
        fail_type = ""
        fail_detail = ""
        if status in ("FAILED", "TIMEOUT"):
            last_line = match.group(5) or ""
            subsequent_lines = []
            start = match.end()
            remaining = content[start : start + 500]
            lines = remaining.split("\n")
            for line in lines[:5]:
                stripped = line.strip()
                # Stop at the next progress line: format is "(<n>/<total>) [PASSED|FAILED|TIMEOUT] ...".
                # Use `in` (not startswith) because the line begins with "(n/total)".
                if not stripped or any(tag in stripped for tag in ("[PASSED]", "[FAILED]", "[TIMEOUT]")):
                    break
                subsequent_lines.append(stripped)
            all_last = [last_line] + subsequent_lines if last_line else subsequent_lines
            fail_type, fail_detail = classify_failure(all_last)
        results[script] = (status, fail_type, fail_detail, start_time, end_time, elapsed)

    return results


def extract_summary(log_path):
    """Extract the Total/Passed/Failed summary line from a log."""
    with open(log_path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Use findall + last match: when --skip-pytest=false, run_examples.sh
    # emits two "Total:" lines (bench-only at :409, bench+pytest at :487).
    # The final line is the authoritative summary; re.search would pick the
    # first (bench-only) and diverge from verify_pr.sh's `grep | tail -1`.
    matches = re.findall(r"^Total:\s+(\d+)\s+\|\s+Passed:\s+(\d+)\s+\|\s+Failed:\s+(\d+)", content, re.MULTILINE)
    if matches:
        total, passed, failed = matches[-1]
        total, passed, failed = int(total), int(passed), int(failed)
        rate = (passed * 100 // total) if total > 0 else 0
        return {"total": total, "passed": passed, "failed": failed, "rate": rate}

    return {"total": 0, "passed": 0, "failed": 0, "rate": 0}


def extract_timing(log_path):
    """Extract Run Start/End/Elapsed from run_examples.sh log summary."""
    with open(log_path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    start = re.search(r"^Run Start:\s+(.+)$", content, re.MULTILINE)
    end = re.search(r"^Run End:\s+(.+)$", content, re.MULTILINE)
    elapsed = re.search(r"^Elapsed:\s+(\d+)s", content, re.MULTILINE)
    return {
        "start": start.group(1).strip() if start else "N/A",
        "end": end.group(1).strip() if end else "N/A",
        "elapsed_secs": int(elapsed.group(1)) if elapsed else 0,
    }


def format_elapsed(secs):
    """Format seconds into a human-readable string like '5m04s' or '33s'."""
    if secs >= 60:
        return f"{secs // 60}m{secs % 60:02d}s"
    return f"{secs}s"


def fmt_delta(curr, prev):
    """Format a numeric delta as a string with sign."""
    delta = curr - prev
    if delta > 0:
        return f"+{delta}"
    elif delta < 0:
        return str(delta)
    return "0"


def compare_results(before_results, after_results):
    """Compare before/after results and categorize scripts.

    Returns dict with keys: new_fails, fixed, still_failing, still_passing,
    added_scripts, removed_scripts (all lists).

    NOTE: TIMEOUT counts as a failure state, equivalent to FAILED, so
    that a TIMEOUT<->FAILED transition is "无变化" and TIMEOUT->PASSED is FIXED.
    """
    before_scripts = set(before_results.keys())
    after_scripts = set(after_results.keys())

    new_fails = []
    fixed = []
    still_failing = []
    still_passing = []
    added_scripts = sorted(after_scripts - before_scripts)
    removed_scripts = sorted(before_scripts - after_scripts)

    def _norm(s):
        return "FAILED" if s in ("FAILED", "TIMEOUT") else s

    for script in sorted(before_scripts & after_scripts):
        b_status = _norm(before_results[script][0])
        a_status = _norm(after_results[script][0])

        if b_status == "FAILED" and a_status == "PASSED":
            fixed.append(script)
        elif b_status == "PASSED" and a_status == "FAILED":
            new_fails.append(script)
        elif b_status == "FAILED" and a_status == "FAILED":
            still_failing.append(script)
        elif b_status == "PASSED" and a_status == "PASSED":
            still_passing.append(script)

    return {
        "new_fails": new_fails,
        "fixed": fixed,
        "still_failing": still_failing,
        "still_passing": still_passing,
        "added_scripts": added_scripts,
        "removed_scripts": removed_scripts,
    }


def generate_report(args):
    before_results = parse_log(args.before_log)
    after_results = parse_log(args.after_log)
    before_summary = extract_summary(args.before_log)
    after_summary = extract_summary(args.after_log)

    cmp = compare_results(before_results, after_results)
    new_fails = cmp["new_fails"]
    fixed = cmp["fixed"]
    still_failing = cmp["still_failing"]
    still_passing = cmp["still_passing"]
    added_scripts = cmp["added_scripts"]
    removed_scripts = cmp["removed_scripts"]

    lines = []
    # Escape pipe chars so a title/url containing '|' won't break Markdown table layout.
    pr_title_esc = args.pr_title.replace("|", "\\|")
    pr_url_esc = args.pr_url.replace("|", "\\|")
    lines.append("# PR 验证报告")
    lines.append("")
    lines.append("## PR 信息")
    lines.append("")
    lines.append("| 项目 | 内容 |")
    lines.append("|------|------|")
    lines.append(f"| **标题** | {pr_title_esc} |")
    lines.append(f"| **链接** | {pr_url_esc} |")
    lines.append(f"| **后端** | {args.backend} |")
    lines.append(f"| **Before (merge-base)** | `{args.before_sha[:12]}` |")
    lines.append(f"| **After (head)** | `{args.after_sha[:12]}` |")
    lines.append(f"| **重编译** | {'是' if args.needs_rebuild.lower() == 'true' else '否'} |")
    lines.append("")

    lines.append("## 对比摘要")
    lines.append("")
    lines.append("| 指标 | Before | After | 变化 |")
    lines.append("|------|--------|-------|------|")

    lines.append(
        f"| 总数 | {before_summary['total']} | {after_summary['total']} | {fmt_delta(after_summary['total'], before_summary['total'])} |"
    )
    lines.append(
        f"| 通过 | {before_summary['passed']} | {after_summary['passed']} | {fmt_delta(after_summary['passed'], before_summary['passed'])} |"
    )
    lines.append(
        f"| 失败 | {before_summary['failed']} | {after_summary['failed']} | {fmt_delta(after_summary['failed'], before_summary['failed'])} |"
    )
    lines.append(
        f"| 通过率 | {before_summary['rate']}% | {after_summary['rate']}% | {fmt_delta(after_summary['rate'], before_summary['rate'])}% |"
    )
    lines.append("")

    lines.append("## 结果分类")
    lines.append("")
    lines.append("| 分类 | 数量 |")
    lines.append("|------|------|")
    lines.append(f"| :green_circle: PR 修复 (FIXED) | {len(fixed)} |")
    lines.append(f"| :red_circle: 新增回归 (NEW FAIL) | {len(new_fails)} |")
    lines.append(f"| :yellow_circle: 持续失败 | {len(still_failing)} |")
    lines.append(f"| :white_circle: 持续通过 | {len(still_passing)} |")
    lines.append(f"| :new: 新增脚本 | {len(added_scripts)} |")
    lines.append(f"| :wastebasket: 移除脚本 | {len(removed_scripts)} |")
    lines.append("")

    before_timing = extract_timing(args.before_log)
    after_timing = extract_timing(args.after_log)
    total_secs = before_timing["elapsed_secs"] + after_timing["elapsed_secs"]

    lines.append("## 耗时统计")
    lines.append("")
    lines.append("| 阶段 | 开始时间 | 结束时间 | 耗时 |")
    lines.append("|------|---------|---------|------|")
    lines.append(
        f"| Before (merge-base) | {before_timing['start']} | {before_timing['end']} | {format_elapsed(before_timing['elapsed_secs'])} |"
    )
    lines.append(f"| After (head) | {after_timing['start']} | {after_timing['end']} | {format_elapsed(after_timing['elapsed_secs'])} |")
    lines.append(f"| **合计** | — | — | **{format_elapsed(total_secs)}** |")
    lines.append("")

    if new_fails:
        lines.append("## :red_circle: 新增回归（NEW FAIL）")
        lines.append("")
        lines.append("**PR 引入的回归，需重点关注：**")
        lines.append("")
        lines.append("| 脚本 | 失败类型 | 失败详情 |")
        lines.append("|------|---------|---------|")
        for script in new_fails:
            _, fail_type, fail_detail, *_ = after_results[script]
            lines.append(f"| `{script}` | {fail_type} | {fail_detail} |")
        lines.append("")

    if fixed:
        lines.append("## :green_circle: PR 修复（FIXED）")
        lines.append("")
        lines.append("| 脚本 |")
        lines.append("|------|")
        for script in fixed:
            lines.append(f"| `{script}` |")
        lines.append("")

    if still_failing:
        lines.append("## :yellow_circle: 持续失败（与本 PR 无关）")
        lines.append("")
        lines.append("| 脚本 | 失败类型 | Before | After |")
        lines.append("|------|---------|--------|-------|")
        for script in still_failing:
            b_type = before_results[script][1] or "-"
            a_type = after_results[script][1] or "-"
            lines.append(f"| `{script}` | {a_type} | {b_type} | {a_type} |")
        lines.append("")

    if added_scripts:
        lines.append("## :new: 新增脚本（PR 添加）")
        lines.append("")
        lines.append("| 脚本 | 结果 |")
        lines.append("|------|------|")
        for script in added_scripts:
            status = after_results[script][0]
            icon = ":green_circle:" if status == "PASSED" else ":red_circle:"
            lines.append(f"| `{script}` | {icon} {status} |")
        lines.append("")

    if removed_scripts:
        lines.append("## :wastebasket: 移除脚本（PR 删除）")
        lines.append("")
        lines.append("| 脚本 | 原结果 |")
        lines.append("|------|--------|")
        for script in removed_scripts:
            status = before_results[script][0]
            icon = ":green_circle:" if status == "PASSED" else ":red_circle:"
            lines.append(f"| `{script}` | {icon} {status} |")
        lines.append("")

    lines.append("## 详细报告")
    lines.append("")
    lines.append(f"- Excel: `{os.path.join(os.path.dirname(args.output), 'run_examples_results.xlsx')}`")
    lines.append(f"- Before 日志: `{args.before_log}`")
    lines.append(f"- After 日志: `{args.after_log}`")
    lines.append("")

    report = "\n".join(lines)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report saved to: {args.output}")
    print(f"  FIXED: {len(fixed)} | NEW FAIL: {len(new_fails)} | Still failing: {len(still_failing)}")
    print(
        f"  Pass rate: {before_summary['rate']}% -> {after_summary['rate']}% ({fmt_delta(after_summary['rate'], before_summary['rate'])}%)"
    )


def discover_backends(output_dir):
    """Discover backend subdirectories containing before.log and after.log.

    Returns a sorted list of backend names (subdirectory names).
    """
    backends = []
    if not os.path.isdir(output_dir):
        return backends
    for entry in sorted(os.listdir(output_dir)):
        full = os.path.join(output_dir, entry)
        if os.path.isdir(full) and os.path.isfile(os.path.join(full, "before.log")) and os.path.isfile(os.path.join(full, "after.log")):
            backends.append(entry)
    return backends


def generate_multi_backend_report(args):
    """Generate a consolidated Markdown report across multiple backends.

    Discovers backend subdirectories under --output-dir, parses each backend's
    before/after logs, and produces a summary report at --output.
    """
    output_dir = args.output_dir
    backends = discover_backends(output_dir)

    if not backends:
        print(
            f"Error: No backend subdirectories with before.log/after.log found in {output_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Gather data for each backend
    backend_data = OrderedDict()
    total_fixed = 0
    total_new_fails = 0
    total_still_failing = 0
    total_elapsed_secs = 0

    for b in backends:
        b_dir = os.path.join(output_dir, b)
        before_log = os.path.join(b_dir, "before.log")
        after_log = os.path.join(b_dir, "after.log")

        before_results = parse_log(before_log)
        after_results = parse_log(after_log)
        before_summary = extract_summary(before_log)
        after_summary = extract_summary(after_log)
        before_timing = extract_timing(before_log)
        after_timing = extract_timing(after_log)
        cmp = compare_results(before_results, after_results)

        elapsed_secs = before_timing["elapsed_secs"] + after_timing["elapsed_secs"]
        total_elapsed_secs += elapsed_secs
        total_fixed += len(cmp["fixed"])
        total_new_fails += len(cmp["new_fails"])
        total_still_failing += len(cmp["still_failing"])

        backend_data[b] = {
            "before_results": before_results,
            "after_results": after_results,
            "before_summary": before_summary,
            "after_summary": after_summary,
            "before_timing": before_timing,
            "after_timing": after_timing,
            "elapsed_secs": elapsed_secs,
            "cmp": cmp,
        }

    # Build report
    lines = []
    pr_title_esc = args.pr_title.replace("|", "\\|")
    pr_url_esc = args.pr_url.replace("|", "\\|")
    backends_str = ", ".join(backends)

    lines.append("# PR 验证报告（汇总）")
    lines.append("")
    lines.append("## PR 信息")
    lines.append("")
    lines.append("| 项目 | 内容 |")
    lines.append("|------|------|")
    lines.append(f"| **标题** | {pr_title_esc} |")
    lines.append(f"| **链接** | {pr_url_esc} |")
    lines.append(f"| **后端** | {backends_str} |")
    lines.append(f"| **Before (merge-base)** | `{args.before_sha[:12]}` |")
    lines.append(f"| **After (head)** | `{args.after_sha[:12]}` |")
    lines.append(f"| **重编译** | {'是' if args.needs_rebuild.lower() == 'true' else '否'} |")
    lines.append("")

    # Per-backend summary table
    lines.append("## 各后端对比摘要")
    lines.append("")
    lines.append("| 后端 | Before 通过率 | After 通过率 | 变化 | FIXED | NEW FAIL | 持续失败 | 耗时 |")
    lines.append("|------|-------------|------------|------|-------|----------|---------|------|")

    for b in backends:
        d = backend_data[b]
        bs = d["before_summary"]
        as_ = d["after_summary"]
        rate_delta = fmt_delta(as_["rate"], bs["rate"])
        lines.append(
            f"| {b} | {bs['rate']}% ({bs['passed']}/{bs['total']}) "
            f"| {as_['rate']}% ({as_['passed']}/{as_['total']}) "
            f"| {rate_delta}% | {len(d['cmp']['fixed'])} | {len(d['cmp']['new_fails'])} "
            f"| {len(d['cmp']['still_failing'])} | {format_elapsed(d['elapsed_secs'])} |"
        )
    lines.append(
        f"| **合计** | — | — | — | {total_fixed} | {total_new_fails} | {total_still_failing} | **{format_elapsed(total_elapsed_secs)}** |"
    )
    lines.append("")

    # Overall conclusion
    lines.append("## 总体结论")
    lines.append("")
    if total_new_fails == 0 and total_fixed == 0:
        lines.append(f"本 PR 在所有后端（{backends_str}）均**未引入回归，也未修复已有问题**。")
        lines.append(f"共 {total_still_failing} 个持续失败（与本 PR 无关）。")
    elif total_new_fails > 0:
        lines.append(f":red_circle: 本 PR 引入 **{total_new_fails}** 个新增回归，需重点关注！")
        if total_fixed > 0:
            lines.append(f":green_circle: 同时修复了 **{total_fixed}** 个已有问题。")
    else:
        lines.append(f":green_circle: 本 PR 修复了 **{total_fixed}** 个已有问题，未引入回归。")
    lines.append("")

    # NEW FAIL section (grouped by backend)
    any_new_fails = False
    for b in backends:
        cmp = backend_data[b]["cmp"]
        if cmp["new_fails"]:
            if not any_new_fails:
                lines.append("## :red_circle: 新增回归（NEW FAIL）")
                lines.append("")
                any_new_fails = True
            lines.append(f"### {b}")
            lines.append("")
            lines.append("| 脚本 | 失败类型 | 失败详情 |")
            lines.append("|------|---------|---------|")
            after_results = backend_data[b]["after_results"]
            for script in cmp["new_fails"]:
                _, fail_type, fail_detail, *_ = after_results[script]
                lines.append(f"| `{script}` | {fail_type} | {fail_detail} |")
            lines.append("")

    # FIXED section (grouped by backend)
    any_fixed = False
    for b in backends:
        cmp = backend_data[b]["cmp"]
        if cmp["fixed"]:
            if not any_fixed:
                lines.append("## :green_circle: PR 修复（FIXED）")
                lines.append("")
                any_fixed = True
            lines.append(f"### {b}")
            lines.append("")
            lines.append("| 脚本 |")
            lines.append("|------|")
            for script in cmp["fixed"]:
                lines.append(f"| `{script}` |")
            lines.append("")

    # Still failing section (grouped by backend)
    any_still_failing = False
    for b in backends:
        cmp = backend_data[b]["cmp"]
        if cmp["still_failing"]:
            if not any_still_failing:
                lines.append("## :yellow_circle: 持续失败（与本 PR 无关）")
                lines.append("")
                any_still_failing = True
            lines.append(f"### {b}")
            lines.append("")
            lines.append("| 脚本 | 失败类型 | Before | After |")
            lines.append("|------|---------|--------|-------|")
            before_results = backend_data[b]["before_results"]
            after_results = backend_data[b]["after_results"]
            for script in cmp["still_failing"]:
                b_type = before_results[script][1] or "-"
                a_type = after_results[script][1] or "-"
                lines.append(f"| `{script}` | {a_type} | {b_type} | {a_type} |")
            lines.append("")

    # Detailed reports links
    lines.append("## 各后端详细报告")
    lines.append("")
    for b in backends:
        lines.append(f"### {b}")
        lines.append("")
        lines.append(f"- Markdown: `{b}/pr_verify_report.md`")
        lines.append(f"- Excel: `{b}/run_examples_results.xlsx`")
        lines.append(f"- Before 日志: `{b}/before.log`")
        lines.append(f"- After 日志: `{b}/after.log`")
        lines.append("")

    report = "\n".join(lines)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Multi-backend summary report saved to: {args.output}")
    print(f"  Backends: {backends_str}")
    print(f"  FIXED: {total_fixed} | NEW FAIL: {total_new_fails} | Still failing: {total_still_failing}")


def main():
    parser = argparse.ArgumentParser(description="Generate PR verification Markdown report")
    parser.add_argument("--before-log", help="Path to before (merge-base) log file")
    parser.add_argument("--after-log", help="Path to after (head) log file")
    parser.add_argument("--pr-url", required=True, help="PR URL")
    parser.add_argument("--pr-title", required=True, help="PR title")
    parser.add_argument("--backend", help="Backend type used (single-backend mode)")
    parser.add_argument("--before-sha", required=True, help="Before (merge-base) commit SHA")
    parser.add_argument("--after-sha", required=True, help="After (head) commit SHA")
    parser.add_argument("--needs-rebuild", required=True, help="Whether rebuild was needed (true/false)")
    parser.add_argument("--output", required=True, help="Output Markdown file path")
    parser.add_argument("--multi", action="store_true", help="Multi-backend summary mode")
    parser.add_argument("--output-dir", help="Output directory containing backend subdirs (multi mode)")
    args = parser.parse_args()

    if args.multi:
        if not args.output_dir:
            print("Error: --output-dir is required in --multi mode", file=sys.stderr)
            sys.exit(1)
        generate_multi_backend_report(args)
    else:
        if not args.before_log or not args.after_log or not args.backend:
            print(
                "Error: --before-log, --after-log, and --backend are required in single-backend mode",
                file=sys.stderr,
            )
            sys.exit(1)
        generate_report(args)


if __name__ == "__main__":
    main()
