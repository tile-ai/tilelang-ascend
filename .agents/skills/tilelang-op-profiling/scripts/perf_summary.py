# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Archive msprof op CSV output and summarize one TileLang kernel.

Usage:
    python3 perf_summary.py <OPPROF_dir> --output-dir <archive_dir> \
        --kernel-name <expected_kernel> [metadata options]

The script copies every supported CSV that is present to an explicit output
directory, then generates a summary from only the requested kernel. AIC and
AIV data are summarized independently when a TileLang kernel uses both core
types. OpBasicInfo.csv and PipeUtilization.csv are required; other metrics are
optional because their availability depends on the installed profiler, device,
and requested metric set.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import os
import shutil
import statistics
from typing import Any
from collections.abc import Iterable, Sequence


KERNEL_NAME_FIELDS = (
    "Op Name",
    "OpName",
    "Kernel Name",
    "KernelName",
    "op_name",
    "kernel_name",
    "Task Name",
    "task_name",
)

REQUIRED_CSV_NAMES = (
    "OpBasicInfo.csv",
    "PipeUtilization.csv",
)

OPTIONAL_CSV_NAMES = (
    "ArithmeticUtilization.csv",
    "Memory.csv",
    "MemoryL0.csv",
    "MemoryUB.csv",
    "L2Cache.csv",
    "ResourceConflictRatio.csv",
)

CSV_NAMES = REQUIRED_CSV_NAMES + OPTIONAL_CSV_NAMES


def optional_float(val: Any) -> float | None:
    """Parse one numeric field while preserving missing values as missing."""
    if val is None or str(val).strip() in ("", "N/A", "NA"):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def safe_float(val: Any, default: float = 0.0) -> float:
    value = optional_float(val)
    return default if value is None else value


def numeric_values(rows: Sequence[dict[str, str]], field: str, multiply: float = 1.0) -> list[float]:
    """Return numeric values for a field without turning NA rows into zeros."""
    values = []
    for row in rows:
        value = optional_float(row.get(field))
        if value is not None:
            values.append(value * multiply)
    return values


def read_csv(path: str) -> list[dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as file:
        rows = []
        for row in csv.DictReader(file):
            rows.append(
                {str(key).strip(): value.strip() if isinstance(value, str) else value for key, value in row.items() if key is not None}
            )
        return rows


def metric_csv_path(opprof_dir: str, csv_name: str, required: bool = True) -> str | None:
    """Resolve a fixed CSV name or one timestamp-suffixed CSV."""
    exact = os.path.join(opprof_dir, csv_name)
    if os.path.isfile(exact):
        return exact
    stem, extension = os.path.splitext(csv_name)
    matches = glob.glob(os.path.join(opprof_dir, f"{stem}_*{extension}"))
    if not matches and not required:
        return None
    if len(matches) != 1:
        raise ValueError(f"expected one {csv_name} in {opprof_dir}, got {len(matches)}")
    return matches[0]


def has_metric_csv(opprof_dir: str, csv_name: str) -> bool:
    """Check whether a directory contains a fixed or timestamp-suffixed metric."""
    exact = os.path.join(opprof_dir, csv_name)
    if os.path.isfile(exact):
        return True
    stem, extension = os.path.splitext(csv_name)
    return bool(glob.glob(os.path.join(opprof_dir, f"{stem}_*{extension}")))


def resolve_profile_dir(profile_input_root: str) -> str:
    """Resolve a flat OPPROF or exactly one nested launch directory."""
    if all(has_metric_csv(profile_input_root, name) for name in REQUIRED_CSV_NAMES):
        return profile_input_root

    candidates = []
    for current_dir, dir_names, _ in os.walk(profile_input_root):
        dir_names.sort()
        if current_dir == profile_input_root:
            continue
        if all(has_metric_csv(current_dir, name) for name in REQUIRED_CSV_NAMES):
            candidates.append(os.path.abspath(current_dir))

    if not candidates:
        required = ", ".join(REQUIRED_CSV_NAMES)
        raise ValueError(f"no profile directory containing {required} was found under {profile_input_root}")
    if len(candidates) > 1:
        formatted = "\n  ".join(candidates)
        raise ValueError(
            f"multiple profile directories were found; pass one explicit launch directory instead of guessing the mapping:\n  {formatted}"
        )
    return candidates[0]


def stat_line(
    name: str,
    values: Sequence[float],
    fmt: str = ".1f",
    unit: str = "",
    multiply: float = 1.0,
) -> str | None:
    """Generate a min/avg/max line, omitting an all-zero metric."""
    scaled = [value * multiply for value in values]
    if not scaled or all(abs(value) < 0.001 for value in scaled):
        return None
    suffix = f"    ({unit})" if unit else ""
    return f"{'  ' + name:<24s} {min(scaled):>10{fmt}} {statistics.mean(scaled):>10{fmt}} {max(scaled):>10{fmt}}{suffix}"


def row_kernel_name(row: dict[str, str]) -> str | None:
    for field in KERNEL_NAME_FIELDS:
        raw_value = row.get(field)
        value = raw_value.strip() if isinstance(raw_value, str) else ""
        if value:
            return value
    return None


def unique_kernel_names(rows: Iterable[dict[str, str]]) -> list[str]:
    return sorted({name for row in rows if (name := row_kernel_name(row))})


def unique_match(expected: str, names: Sequence[str], source: str) -> str:
    """Resolve expected to exactly one runtime kernel name."""
    exact = [name for name in names if name == expected]
    if exact:
        return exact[0]

    folded_expected = expected.casefold()
    casefold_exact = [name for name in names if name.casefold() == folded_expected]
    if len(casefold_exact) == 1:
        return casefold_exact[0]

    contains = [name for name in names if folded_expected in name.casefold()]
    if len(contains) == 1:
        return contains[0]
    if not contains:
        raise ValueError(f"{source}: kernel '{expected}' not found; available kernels: {', '.join(names) if names else '(none)'}")
    raise ValueError(f"{source}: kernel '{expected}' is ambiguous; matches: {', '.join(contains)}")


def filter_rows_for_kernel(
    rows: list[dict[str, str]],
    expected: str,
    resolved: str,
    source: str,
    raw_kernel_count: int,
) -> list[dict[str, str]]:
    """Filter a CSV by kernel name when it exposes a supported name column."""
    if not rows:
        return []
    names = unique_kernel_names(rows)
    if not names:
        if raw_kernel_count > 1:
            raise ValueError(
                f"{source}: no kernel-name column, but OpBasicInfo.csv contains {raw_kernel_count} kernels; refusing to mix their metrics"
            )
        return rows

    selected_name = resolved if resolved in names else unique_match(expected, names, source)
    filtered = [row for row in rows if row_kernel_name(row) == selected_name]
    if not filtered:
        raise ValueError(f"{source}: no rows remain after selecting kernel '{selected_name}'")
    return filtered


def load_kernel_tables(opprof_dir: str, expected: str) -> tuple[dict[str, list[dict[str, str]]], str]:
    basic_name = "OpBasicInfo.csv"
    basic_path = metric_csv_path(opprof_dir, basic_name)
    if basic_path is None:
        raise ValueError(f"{basic_name} is missing")
    basic_rows = read_csv(basic_path)
    if not basic_rows:
        raise ValueError(f"{basic_name} is missing or empty")
    basic_kernels = unique_kernel_names(basic_rows)
    if not basic_kernels:
        raise ValueError(f"{basic_name} has no supported kernel-name column; expected one of: {', '.join(KERNEL_NAME_FIELDS)}")

    resolved = unique_match(expected, basic_kernels, basic_name)
    raw_kernel_count = len(basic_kernels)
    tables: dict[str, list[dict[str, str]]] = {}
    for csv_name in CSV_NAMES:
        csv_path = metric_csv_path(
            opprof_dir,
            csv_name,
            required=csv_name in REQUIRED_CSV_NAMES,
        )
        rows = read_csv(csv_path) if csv_path else []
        tables[csv_name] = filter_rows_for_kernel(rows, expected, resolved, csv_name, raw_kernel_count)
    return tables, resolved


def timed_core_prefixes(rows: Sequence[dict[str, str]]) -> list[str]:
    """Return core types that have non-zero profiler timing or counters."""
    timed = []
    for prefix in ("aiv", "aic"):
        if any(safe_float(value) > 0 for row in rows for field, value in row.items() if field.startswith(f"{prefix}_")):
            timed.append(prefix)
    return timed


def rows_for_core_type(rows: Sequence[dict[str, str]], prefix: str) -> list[dict[str, str]]:
    """Select positive-time rows when a table exposes per-core timing."""
    time_field = f"{prefix}_time(us)"
    if not any(optional_float(row.get(time_field)) is not None for row in rows):
        return list(rows)
    return [row for row in rows if safe_float(row.get(time_field)) > 0]


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Return one linearly interpolated percentile for a non-empty sequence."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def positive_evidence_count(
    rows: Sequence[dict[str, str]],
    prefix: str,
    fields: Sequence[str],
) -> tuple[int, int] | None:
    """Count rows with positive counters without calling them effective cores."""
    selected_rows = rows_for_core_type(rows, prefix)
    available_fields = [field for field in fields if any(optional_float(row.get(field)) is not None for row in selected_rows)]
    if not selected_rows or not available_fields:
        return None
    positive_rows = sum(any(safe_float(row.get(field)) > 0 for field in available_fields) for row in selected_rows)
    return positive_rows, len(selected_rows)


def append_work_evidence(
    lines: list[str],
    tables: dict[str, list[dict[str, str]]],
    prefix: str,
) -> None:
    """Report observable work counters, not inferred task-bearing cores."""
    compute_fields = ("aiv_vec_fops",) if prefix == "aiv" else ("aic_cube_fops",)
    transfer_fields = (
        ("GM_to_UB_datas(KB)", "UB_to_GM_datas(KB)")
        if prefix == "aiv"
        else (
            "GM_to_L1_datas(KB)",
            "L0C_to_GM_datas(KB)",
        )
    )
    evidence = []
    compute_count = positive_evidence_count(tables["ArithmeticUtilization.csv"], prefix, compute_fields)
    if compute_count is not None:
        evidence.append(f"FLOP 计数非零={compute_count[0]}/{compute_count[1]}")
    transfer_count = positive_evidence_count(tables["Memory.csv"], prefix, transfer_fields)
    if transfer_count is not None:
        evidence.append(f"GM 搬运非零={transfer_count[0]}/{transfer_count[1]}")
    if evidence:
        lines.append("  工作计数证据（不等同于有效任务核）: " + " | ".join(evidence))


def archive_csvs(opprof_dir: str, output_dir: str) -> list[str]:
    os.makedirs(output_dir, exist_ok=False)
    copied = []
    for csv_name in CSV_NAMES:
        csv_path = metric_csv_path(
            opprof_dir,
            csv_name,
            required=csv_name in REQUIRED_CSV_NAMES,
        )
        if csv_path is None:
            continue
        shutil.copy2(csv_path, os.path.join(output_dir, csv_name))
        copied.append(csv_name)
    return copied


def append_ratio_and_bandwidth(lines: list[str], rows: Sequence[dict[str, str]], prefix: str) -> None:
    if prefix == "aiv":
        ratio_fields = [
            ("vec_ratio%", "aiv_vec_ratio"),
            ("scalar_ratio%", "aiv_scalar_ratio"),
            ("mte2_ratio%", "aiv_mte2_ratio"),
            ("mte3_ratio%", "aiv_mte3_ratio"),
            ("icache_miss%", "aiv_icache_miss_rate"),
        ]
        bandwidth_fields = [
            ("mte2_active_bw", "aiv_mte2_active_bw(GB/s)"),
            ("mte3_active_bw", "aiv_mte3_active_bw(GB/s)"),
        ]
    else:
        ratio_fields = [
            ("cube_ratio%", "aic_cube_ratio"),
            ("scalar_ratio%", "aic_scalar_ratio"),
            ("mte1_ratio%", "aic_mte1_ratio"),
            ("mte2_ratio%", "aic_mte2_ratio"),
            ("mte3_ratio%", "aic_mte3_ratio"),
            ("fixpipe_ratio%", "aic_fixpipe_ratio"),
            ("icache_miss%", "aic_icache_miss_rate"),
        ]
        bandwidth_fields = [
            ("mte1_active_bw", "aic_mte1_active_bw(GB/s)"),
            ("mte2_active_bw", "aic_mte2_active_bw(GB/s)"),
            ("mte3_active_bw", "aic_mte3_active_bw(GB/s)"),
            ("fixpipe_active_bw", "aic_fixpipe_active_bw(GB/s)"),
        ]

    for display_name, field in ratio_fields:
        line = stat_line(display_name, numeric_values(rows, field), ".2f", multiply=100)
        if line:
            lines.append(line)
    for display_name, field in bandwidth_fields:
        line = stat_line(
            display_name,
            numeric_values(rows, field),
            ".1f",
            unit="GB/s",
        )
        if line:
            lines.append(line)


def append_scalar_breakdown(lines: list[str], rows: Sequence[dict[str, str]], prefix: str) -> None:
    fields = [
        ("single", f"{prefix}_scalar_single_time(us)"),
        ("dual", f"{prefix}_scalar_dual_time(us)"),
        ("wait", f"{prefix}_scalar_wait_time(us)"),
        ("mte2_stall", f"{prefix}_scalar_mte2_stall_time(us)"),
        ("mte3_stall", f"{prefix}_scalar_mte3_stall_time(us)"),
    ]
    if prefix == "aiv":
        fields.extend(
            [
                ("vec_stall", "aiv_scalar_vector_stall_time(us)"),
                ("ub_stall", "aiv_scalar_stall_by_ub_time(us)"),
            ]
        )
    else:
        fields.extend(
            [
                ("cube_stall", "aic_scalar_cube_stall_time(us)"),
                ("mte1_stall", "aic_scalar_mte1_stall_time(us)"),
            ]
        )
    fields.append(("wait_ib", f"{prefix}_scalar_wait_ib_time(us)"))

    parts = []
    for display_name, field in fields:
        values = numeric_values(rows, field)
        if not values:
            continue
        average = statistics.mean(values)
        if average > 0.001:
            parts.append(f"{display_name}: {average:.2f}us")
    if parts:
        lines.append(f"  SCALAR 子类 ({prefix}, avg): " + " | ".join(parts))


def generate_summary(
    tables: dict[str, list[dict[str, str]]],
    output_dir: str,
    expected_kernel: str,
    resolved_kernel: str,
) -> str:
    lines = ["=== 上板性能统计摘要 ==="]

    basic_rows = tables["OpBasicInfo.csv"]
    first = basic_rows[0]
    op_type = first.get("Op Type", "unknown")
    durations = numeric_values(basic_rows, "Task Duration(us)")
    duration = statistics.mean(durations) if durations else 0.0
    block_dims = [int(value) for value in numeric_values(basic_rows, "Block Dim")]
    current_freqs = numeric_values(basic_rows, "Current Freq")
    rated_freqs = numeric_values(basic_rows, "Rated Freq")
    lines.append(f"Requested Kernel: {expected_kernel}")
    lines.append(f"Resolved Kernel:  {resolved_kernel} | Type: {op_type}")
    if durations:
        lines.append(
            f"Task Duration(us): min={min(durations):.2f} | avg={duration:.2f} | max={max(durations):.2f} | samples={len(durations)}"
        )
    if block_dims:
        lines.append(f"BlockDim: {min(block_dims)}..{max(block_dims)}")
    if current_freqs or rated_freqs:
        current_freq = f"{statistics.mean(current_freqs):.0f}" if current_freqs else "N/A"
        rated_freq = f"{statistics.mean(rated_freqs):.0f}" if rated_freqs else "N/A"
        lines.append(f"Freq(avg): {current_freq}/{rated_freq}")

    pipe_rows = tables["PipeUtilization.csv"]
    prefixes = timed_core_prefixes(pipe_rows)
    critical_times: list[float] = []
    for prefix in prefixes:
        selected_rows = rows_for_core_type(pipe_rows, prefix)
        core_times = numeric_values(selected_rows, f"{prefix}_time(us)")
        critical_times.extend(core_times)
        lines.append("")
        lines.append(f"--- PipeUtilization ({prefix.upper()}, {len(selected_rows)} timed subcores) ---")
        lines.append(f"  {'':24s} {'min':>10s} {'avg':>10s} {'max':>10s}")
        line = stat_line(f"{prefix}_time(us)", core_times, ".2f")
        if line:
            lines.append(line)
        p50 = percentile(core_times, 0.50)
        p90 = percentile(core_times, 0.90)
        if p50 is not None and p90 is not None:
            lines.append(f"  计时行耗时分布: p50={p50:.2f}us | p90={p90:.2f}us | max={max(core_times):.2f}us")
        append_work_evidence(lines, tables, prefix)
        append_ratio_and_bandwidth(lines, selected_rows, prefix)
        append_scalar_breakdown(lines, selected_rows, prefix)

    if pipe_rows and not prefixes:
        lines.extend(["", "--- PipeUtilization ---", "  未检测到非零 AIV/AIC 指标"])
    if critical_times and duration > 0:
        critical_path = max(critical_times)
        overhead = max(0.0, duration - critical_path)
        mode = "max(AIV, AIC)" if len(prefixes) == 2 else prefixes[0].upper()
        lines.extend(
            [
                "",
                "--- 关键路径与头开销 ---",
                f"  计时子核耗时口径: {mode}",
                f"  Task Duration(avg): {duration:.2f}us | 关键路径: {critical_path:.2f}us | "
                f"头开销: {overhead:.2f}us ({overhead / duration * 100:.1f}%)",
            ]
        )

    mem_rows = tables["Memory.csv"]
    if mem_rows:
        lines.extend(["", "--- Memory ---"])
        transfers = [
            ("GM→UB", "GM_to_UB_datas(KB)", "GM_to_UB_bw_usage_rate(%)"),
            ("UB→GM", "UB_to_GM_datas(KB)", "UB_to_GM_bw_usage_rate(%)"),
            ("GM→L1", "GM_to_L1_datas(KB)", "GM_to_L1_bw_usage_rate(%)"),
        ]
        for display, data_field, usage_field in transfers:
            data = numeric_values(mem_rows, data_field)
            if sum(data) > 0:
                usage = numeric_values(mem_rows, usage_field)
                usage_text = f"{statistics.mean(usage):.2f}%" if usage else "N/A"
                lines.append(f"  {display}: {sum(data):.1f}KB total ({statistics.mean(data):.1f}KB/profiler row), BW usage: {usage_text}")
        for display, field in [
            ("主存读取", "read_main_memory_datas(KB)"),
            ("主存写入", "write_main_memory_datas(KB)"),
        ]:
            data = numeric_values(mem_rows, field)
            if sum(data) > 0:
                lines.append(f"  {display}: {sum(data):.1f}KB total")

        gm_ub_total = sum(numeric_values(mem_rows, "GM_to_UB_datas(KB)"))
        mte2_instructions = sum(numeric_values(mem_rows, "aiv_mte2_instructions"))
        mte2_instructions += sum(numeric_values(mem_rows, "aic_mte2_instructions"))
        if gm_ub_total > 0 and mte2_instructions > 0:
            average_transfer = gm_ub_total / mte2_instructions
            lines.append(f"  Avg MTE2 transfer: {average_transfer:.2f}KB ({int(mte2_instructions)} instructions total)")

        for display, field in [
            ("GM→UB avg BW", "aiv_gm_to_ub_bw(GB/s)"),
            ("UB→GM avg BW", "aiv_ub_to_gm_bw(GB/s)"),
            ("GM→L1 avg BW", "aic_gm_to_l1_bw(GB/s)"),
        ]:
            values = numeric_values(mem_rows, field)
            if sum(values) > 0:
                lines.append(f"  {display}: {statistics.mean(values):.2f} GB/s")

    ub_rows = tables["MemoryUB.csv"]
    if ub_rows:
        ub_rows = rows_for_core_type(ub_rows, "aiv")
        parts = []
        for display, field in [
            ("UB read BW (vector)", "aiv_ub_read_bw_vector(GB/s)"),
            ("UB write BW (vector)", "aiv_ub_write_bw_vector(GB/s)"),
            ("UB read BW (scalar)", "aiv_ub_read_bw_scalar(GB/s)"),
            ("UB write BW (scalar)", "aiv_ub_write_bw_scalar(GB/s)"),
        ]:
            values = numeric_values(ub_rows, field)
            if not values:
                continue
            average = statistics.mean(values)
            if average > 0.001:
                parts.append(f"  {display}: avg={average:.1f} GB/s")
        if parts:
            lines.extend(["", "--- MemoryUB ---", *parts])

    l0_rows = tables["MemoryL0.csv"]
    if l0_rows:
        l0_rows = rows_for_core_type(l0_rows, "aic")
        parts = []
        for display, field in [
            ("L0A read BW", "aic_l0a_read_bw(GB/s)"),
            ("L0A write BW", "aic_l0a_write_bw(GB/s)"),
            ("L0B read BW", "aic_l0b_read_bw(GB/s)"),
            ("L0B write BW", "aic_l0b_write_bw(GB/s)"),
            ("L0C read BW (cube)", "aic_l0c_read_bw_cube(GB/s)"),
            ("L0C write BW (cube)", "aic_l0c_write_bw_cube(GB/s)"),
        ]:
            values = numeric_values(l0_rows, field)
            if not values:
                continue
            average = statistics.mean(values)
            if average > 0.001:
                parts.append(f"  {display}: {average:.1f} GB/s")
        if parts:
            lines.extend(["", "--- MemoryL0 ---", *parts])

    l2_rows = tables["L2Cache.csv"]
    if l2_rows:
        l2_prefixes = timed_core_prefixes(l2_rows) or prefixes
        lines.extend(["", "--- L2Cache ---"])
        for prefix in l2_prefixes:
            selected_rows = rows_for_core_type(l2_rows, prefix)
            parts = []
            for display, field in [
                ("total_hit", f"{prefix}_total_hit_rate(%)"),
                ("read_hit", f"{prefix}_read_hit_rate(%)"),
                ("write_hit", f"{prefix}_write_hit_rate(%)"),
            ]:
                values = numeric_values(selected_rows, field)
                if any(value > 0 for value in values):
                    parts.append(f"{display}: avg={statistics.mean(values):.1f}% (min={min(values):.1f}%, max={max(values):.1f}%)")
            if parts:
                lines.append(f"  {prefix.upper()}: " + " | ".join(parts))
            hits = sum(numeric_values(selected_rows, f"{prefix}_write_cache_hit"))
            misses = sum(numeric_values(selected_rows, f"{prefix}_write_cache_miss_allocate"))
            if hits + misses > 0:
                lines.append(f"  {prefix.upper()} write cache: hit={int(hits)} miss={int(misses)}")

    conflict_rows = tables["ResourceConflictRatio.csv"]
    if conflict_rows:
        lines.extend(["", "--- ResourceConflict ---"])
        if "aiv" in prefixes or timed_core_prefixes(conflict_rows) == ["aiv"]:
            aiv_rows = rows_for_core_type(conflict_rows, "aiv")
            parts = []
            for display, field in [
                ("vec_total_cflt", "aiv_vec_total_cflt_ratio"),
                ("bankgroup_cflt", "aiv_vec_bankgroup_cflt_ratio"),
                ("bank_cflt", "aiv_vec_bank_cflt_ratio"),
                ("resc_cflt", "aiv_vec_resc_cflt_ratio"),
                ("mte_cflt", "aiv_vec_mte_cflt_ratio"),
            ]:
                values = numeric_values(aiv_rows, field, multiply=100)
                if any(value > 0 for value in values):
                    parts.append(f"{display}: {statistics.mean(values):.2f}%")
            if parts:
                lines.append("  AIV conflicts: " + " | ".join(parts))

        for prefix in timed_core_prefixes(conflict_rows) or prefixes:
            selected_rows = rows_for_core_type(conflict_rows, prefix)
            wait_fields = (
                [("vec_wait", "aiv_vec_wait_ratio"), ("mte2_wait", "aiv_mte2_wait_ratio"), ("mte3_wait", "aiv_mte3_wait_ratio")]
                if prefix == "aiv"
                else [
                    ("cube_wait", "aic_cube_wait_ratio"),
                    ("mte1_wait", "aic_mte1_wait_ratio"),
                    ("mte2_wait", "aic_mte2_wait_ratio"),
                    ("mte3_wait", "aic_mte3_wait_ratio"),
                ]
            )
            parts = []
            for display, field in wait_fields:
                values = numeric_values(selected_rows, field, multiply=100)
                if any(value > 0 for value in values):
                    parts.append(f"{display}: {statistics.mean(values):.2f}%")
            if parts:
                lines.append(f"  {prefix.upper()} waits: " + " | ".join(parts))

    arithmetic_rows = tables["ArithmeticUtilization.csv"]
    if arithmetic_rows:
        lines.extend(["", "--- ArithmeticUtilization ---"])
        vector_rows = rows_for_core_type(arithmetic_rows, "aiv")
        vector_parts = []
        for display, field in [
            ("vec_fp32", "aiv_vec_fp32_ratio"),
            ("vec_fp16", "aiv_vec_fp16_ratio"),
            ("vec_int32", "aiv_vec_int32_ratio"),
            ("vec_int16", "aiv_vec_int16_ratio"),
            ("vec_misc", "aiv_vec_misc_ratio"),
        ]:
            values = numeric_values(vector_rows, field, multiply=100)
            if not values:
                continue
            average = statistics.mean(values)
            if average > 0.01:
                vector_parts.append(f"{display}: {average:.1f}%")
        vector_fops_values = numeric_values(vector_rows, "aiv_vec_fops")
        vector_fops = statistics.mean(vector_fops_values) if vector_fops_values else 0.0
        if vector_fops > 0:
            vector_parts.append(f"vec_fops: {vector_fops:.0f}/timed row")
        if vector_parts:
            lines.append("  AIV: " + " | ".join(vector_parts))

        cube_rows = rows_for_core_type(arithmetic_rows, "aic")
        cube_parts = []
        for display, field in [
            ("cube_fp16", "aic_cube_fp16_ratio"),
            ("cube_int8", "aic_cube_int8_ratio"),
        ]:
            values = numeric_values(cube_rows, field, multiply=100)
            if not values:
                continue
            average = statistics.mean(values)
            if average > 0.01:
                cube_parts.append(f"{display}: {average:.1f}%")
        cube_fops_values = numeric_values(cube_rows, "aic_cube_fops")
        cube_fops = statistics.mean(cube_fops_values) if cube_fops_values else 0.0
        if cube_fops > 0:
            cube_parts.append(f"cube_fops: {cube_fops:.0f}/timed row")
        if cube_parts:
            lines.append("  AIC: " + " | ".join(cube_parts))

    lines.extend(
        [
            "",
            "--- 原始数据位置 ---",
            f"  CSV 文件: {output_dir}/",
            "  归档文件保持 msprof 原始内容；summary.txt 仅统计 Resolved Kernel。",
            "  如需逐核详情，请读取对应 CSV 文件并继续使用同一 kernel 过滤条件。",
        ]
    )
    return "\n".join(lines)


def write_metadata(
    output_dir: str,
    args: argparse.Namespace,
    resolved_kernel: str,
    copied_csvs: Sequence[str],
    profile_input_root: str,
    profile_input_dir: str,
) -> str:
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "collection_method": "msprof op",
        "profile_input_root": profile_input_root,
        "profile_input_dir": profile_input_dir,
        "entry": args.entry,
        "workdir": args.workdir,
        "command": args.command,
        "case_id": args.case_id,
        "case_sha256": args.case_sha256,
        "target": args.target,
        "device": args.device,
        "requested_kernel": args.kernel_name,
        "resolved_kernel": resolved_kernel,
        "warm_up": args.warm_up,
        "launch_count": args.launch_count,
        "source_sha256": args.source_sha256,
        "time_unit": "us",
        "archived_csvs": list(copied_csvs),
        "missing_optional_csvs": [name for name in OPTIONAL_CSV_NAMES if name not in copied_csvs],
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive msprof CSVs and summarize exactly one TileLang kernel.")
    parser.add_argument(
        "opprof_dir",
        help="Path to a flat OPPROF root or one explicit nested launch directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New archive directory; existing paths are never overwritten",
    )
    parser.add_argument(
        "--kernel-name",
        required=True,
        help="Expected runtime kernel name; zero or ambiguous matches are rejected",
    )
    parser.add_argument("--entry", help="Profiled Python, shell, AOT, or graph entry")
    parser.add_argument("--workdir", help="Working directory used to launch the entry")
    parser.add_argument("--command", help="Exact profiling command")
    parser.add_argument("--case-id", help="Stable identifier for the profiled case")
    parser.add_argument("--target", help="Target already selected by the operator entry")
    parser.add_argument("--device", help="Logical or physical device identifier")
    parser.add_argument("--warm-up", type=int, default=10)
    parser.add_argument("--launch-count", type=int, default=1)
    parser.add_argument("--source-sha256", help="SHA256 of the profiled operator source")
    parser.add_argument(
        "--case-sha256",
        help="SHA256 of normalized case parameters or the ordered case list",
    )
    args = parser.parse_args()

    profile_input_root = os.path.abspath(args.opprof_dir)
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(profile_input_root):
        parser.error(f"'{profile_input_root}' is not an OPPROF directory")
    if os.path.exists(output_dir):
        parser.error(f"archive directory already exists: {output_dir}")
    try:
        opprof_dir = resolve_profile_dir(profile_input_root)
        metric_csv_path(opprof_dir, "PipeUtilization.csv")
        tables, resolved_kernel = load_kernel_tables(opprof_dir, args.kernel_name)
        summary = generate_summary(tables, output_dir, args.kernel_name, resolved_kernel)
        copied = archive_csvs(opprof_dir, output_dir)
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as file:
            file.write(summary)
            file.write("\n")
        metadata_path = write_metadata(
            output_dir,
            args,
            resolved_kernel,
            copied,
            profile_input_root,
            opprof_dir,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Kernel verified: requested='{args.kernel_name}', resolved='{resolved_kernel}'")
    print(f"Profile directory resolved to: {opprof_dir}")
    print(f"Archived {len(copied)} CSV files to: {output_dir}")
    print(f"Summary written to: {summary_path}")
    print(f"Metadata written to: {metadata_path}")
    print()
    print(summary)


if __name__ == "__main__":
    main()
