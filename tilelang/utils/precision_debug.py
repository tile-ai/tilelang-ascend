# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
Precision debug tool: drop-in replacement for torch.testing.assert_close.
On mismatch, writes a text report, saves tensors, and prints an ASCII diff map.
"""
from __future__ import annotations

import inspect
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Optional, Union

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Character-level diff mapping constants
# ---------------------------------------------------------------------------
_LEVEL_CHARS = ["·", "-", "~", "*", "#", "@"]
_LEVEL_CHARS_NP = np.array(_LEVEL_CHARS)
_LEVEL_THRESHOLDS = np.array([1.0, 3.0, 10.0, 50.0, 200.0])
_PASS_CHARS = frozenset({"·", "|", " "})
_PARALLEL_THRESHOLD = 512  # use thread pool when row count exceeds this

# ---------------------------------------------------------------------------
# Default tolerances based on dtype.
# ---------------------------------------------------------------------------
DEFAULT_TOLERANCE = {
    "float16": (1e-3, 1e-3),
    "bfloat16": (2e-2, 2e-2),
    "float32": (1e-4, 1e-4),
    "int8": (0.0, 0.0),
    "int16": (0.0, 0.0),
    "int32": (0.0, 0.0),
    "int64": (0.0, 0.0),
    "uint8": (0.0, 0.0),
    "bool": (0.0, 0.0),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prod(shape: tuple[int, ...]) -> int:
    p = 1
    for s in shape:
        p *= int(s)
    return p


def _get_caller_info() -> tuple[str, str]:
    """Return (caller_filename_no_ext, caller_func_name) for the assert_close caller."""
    frame = inspect.currentframe()
    if frame is None:
        return "unknown", "unknown"
    for _ in range(4):
        frame = frame.f_back
        if frame is None:
            return "unknown", "unknown"
    filename = os.path.splitext(os.path.basename(frame.f_code.co_filename))[0]
    func = frame.f_code.co_name or "unknown"
    return filename, func


def _equalize_for_compare(
    actual: torch.Tensor, expected: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move to CPU and promote to common dtype for comparison."""
    a, b = actual.detach(), expected.detach()
    if a.device.type != "cpu":
        a = a.cpu()
    if b.device.type != "cpu":
        b = b.cpu()
    if a.dtype != b.dtype:
        dtype = torch.promote_types(
            a.dtype if a.dtype.is_floating_point else torch.float32,
            b.dtype if b.dtype.is_floating_point else torch.float32,
        )
        a, b = a.to(dtype), b.to(dtype)
    return a, b


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _compute_stats(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float,
    atol: float,
    *,
    equal_nan: bool,
) -> dict[str, Any]:
    """Compute mismatch statistics between two tensors."""
    a, b = _equalize_for_compare(actual, expected)
    diff_abs = (a - b).abs()
    safe_b = b.abs().clamp(min=1e-12)
    diff_rel = (diff_abs / safe_b).nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    mismatched = ~torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    total = a.numel()
    num_mismatched = int(mismatched.sum().item())

    flat_abs = diff_abs.reshape(-1)
    flat_rel = diff_rel.reshape(-1)
    flat_mm = mismatched.reshape(-1)

    max_abs = float(flat_abs.max().item())
    max_rel = float(flat_rel.max().item())
    mean_abs = float(flat_abs.mean().item())
    median_abs = float(flat_abs.median().item())

    max_abs_idx = list(np.unravel_index(int(flat_abs.argmax().item()), a.shape))
    max_rel_idx = list(np.unravel_index(int(flat_rel.argmax().item()), a.shape))

    # Top-10 by absolute diff
    if num_mismatched > 0:
        mm_idx = flat_mm.nonzero(as_tuple=True)[0]
        vals = flat_abs[mm_idx]
        top10_flat = mm_idx[vals.argsort(descending=True)[:10]]
    else:
        top10_flat = flat_abs.argsort(descending=True)[:10]

    top10_list = []
    for fi in top10_flat.tolist():
        idx = list(np.unravel_index(fi, a.shape))
        av, bv = float(a.reshape(-1)[fi].item()), float(b.reshape(-1)[fi].item())
        top10_list.append({
            "index": idx,
            "actual": av,
            "expected": bv,
            "abs_diff": abs(av - bv),
            "rel_diff": abs(av - bv) / (abs(bv) + 1e-12),
        })

    hist_edges = np.linspace(0, max_abs if max_abs > 0 else 1, 11)
    hist_vals, _ = np.histogram(flat_abs.numpy(), bins=hist_edges)
    hist_max = int(max(hist_vals.max(), 1))

    return {
        "shape": tuple(a.shape), "dtype": str(a.dtype), "device": str(actual.device),
        "rtol": rtol, "atol": atol, "total": total,
        "num_mismatched": num_mismatched,
        "max_abs": max_abs, "max_abs_idx": max_abs_idx,
        "max_rel": max_rel, "max_rel_idx": max_rel_idx,
        "mean_abs": mean_abs, "median_abs": median_abs,
        "top10": top10_list,
        "hist_edges": hist_edges, "hist_vals": hist_vals, "hist_max": hist_max,
        "diff_abs": diff_abs, "actual_cpu": a, "expected_cpu": b,
    }


# ---------------------------------------------------------------------------
# Text report (saved to file)
# ---------------------------------------------------------------------------

def _write_report(out_dir: str, stats: dict[str, Any], msg: str) -> None:
    lines = [
        "Precision Debug Report",
        "=" * 60,
        f"shape: {stats['shape']}",
        f"dtype: {stats['dtype']}",
        f"device: {stats['device']}",
        f"rtol: {stats['rtol']}, atol: {stats['atol']}",
        "",
        "Summary",
        "-" * 40,
        f"Total elements: {stats['total']}",
        f"Mismatched: {stats['num_mismatched']}"
        f" ({100.0 * stats['num_mismatched'] / max(1, stats['total']):.4f}%)",
        "",
        "Difference statistics",
        "-" * 40,
        f"Max abs diff: {stats['max_abs']} at index {stats['max_abs_idx']}",
        f"Max rel diff: {stats['max_rel']} at index {stats['max_rel_idx']}",
        f"Mean abs diff: {stats['mean_abs']}",
        f"Median abs diff: {stats['median_abs']}",
        "",
        "Top-10 largest absolute differences",
        "-" * 40,
    ]
    for i, row in enumerate(stats["top10"], 1):
        lines.append(
            f"  {i}. idx={row['index']}  actual={row['actual']}  expected={row['expected']}"
            f"  abs={row['abs_diff']}  rel={row['rel_diff']}"
        )
    lines.extend(["", "Difference distribution (abs diff bins)", "-" * 40])
    edges, vals, hmax = stats["hist_edges"], stats["hist_vals"], stats["hist_max"]
    for i in range(len(edges) - 1):
        bar = "*" * (int(40 * vals[i] / hmax) if hmax > 0 else 0)
        lines.append(f"  [{edges[i]:.2e}, {edges[i + 1]:.2e}): {bar} ({vals[i]})")
    if msg:
        lines.extend(["", "Message", "-" * 40, msg])
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Vectorised diff-level computation & string building
# ---------------------------------------------------------------------------

def _arr_to_levels(arr: np.ndarray, atol: float) -> np.ndarray:
    """Map abs-diff values to integer levels 0-5 (vectorised)."""
    tol = max(atol, 1e-12)
    ratios = arr / tol
    levels = np.zeros(arr.shape, dtype=np.intp)
    for i in range(len(_LEVEL_THRESHOLDS)):
        levels[ratios >= _LEVEL_THRESHOLDS[i]] = i + 1
    return levels


def _levels_rows_to_strings(
    levels: np.ndarray,
    has_sep: bool,
    outer: int,
    group_size: int,
) -> list[str]:
    """Convert rows of integer levels (2-D) to character strings.

    If *has_sep*, inserts ``|`` between every *group_size* characters.
    """
    results: list[str] = []
    for row in levels:
        if has_sep and outer > 1:
            parts: list[str] = []
            for g in range(outer):
                s = g * group_size
                parts.append("".join(_LEVEL_CHARS_NP[row[s:s + group_size]]))
            results.append("|".join(parts))
        else:
            results.append("".join(_LEVEL_CHARS_NP[row]))
    return results


def _parallel_build_strings(
    levels: np.ndarray,
    has_sep: bool,
    outer: int,
    group_size: int,
) -> list[str]:
    """Thread-parallel version of :func:`_levels_rows_to_strings`."""
    total = len(levels)
    n_workers = min(4, os.cpu_count() or 2)
    chunk = max(1, (total + n_workers - 1) // n_workers)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [
            pool.submit(
                _levels_rows_to_strings,
                levels[s:min(s + chunk, total)],
                has_sep, outer, group_size,
            )
            for s in range(0, total, chunk)
        ]
        out: list[str] = []
        for f in futs:
            out.extend(f.result())
    return out


# ---------------------------------------------------------------------------
# Smart line-layout helpers
# ---------------------------------------------------------------------------

def _format_row_label(
    idx: tuple[int, ...],
    leading_shape: tuple[int, ...],
    merged_shape: tuple[int, ...],
) -> str:
    """Build a slice-notation label for one row.

    Example: shape=(2,64,8,128), split=3, idx=(1,32,3)  →  ``[1,32,3,:128]``
    """
    parts: list[str] = []
    for dim_i, val in enumerate(idx):
        w = max(1, len(str(leading_shape[dim_i] - 1)))
        parts.append(str(val).rjust(w))
    for d in merged_shape:
        parts.append(f":{d}")
    return "[" + ",".join(parts) + "]"


def _compute_label_width(
    leading_shape: tuple[int, ...],
    merged_shape: tuple[int, ...],
) -> int:
    """Fixed character width of the widest possible row label."""
    parts: list[str] = []
    for d in leading_shape:
        parts.append(str(max(0, d - 1)))
    for d in merged_shape:
        parts.append(f":{d}")
    return len("[" + ",".join(parts) + "]")


def _choose_line_split(shape: tuple[int, ...], term_width: int = 200) -> int:
    """Decide how to split *shape* into row-index dims and merged-per-row dims.

    Returns *split* such that ``shape[:split]`` iterates rows and
    ``shape[split:]`` is flattened into one line.
    """
    if not shape:
        return 0
    ndim = len(shape)
    min_row_chars = min(40, term_width // 4)

    # Pass 1: merge as many tail dims as fit without downsampling
    best = ndim - 1
    for s in range(ndim - 1, -1, -1):
        merged = shape[s:]
        elems = _prod(merged)
        lw = _compute_label_width(shape[:s], merged) + 3  # "  " prefix + space
        outer = merged[0] if len(merged) > 1 else 1
        n_seps = (outer - 1) if len(merged) > 1 else 0
        if elems + n_seps + lw <= term_width:
            best = s
        else:
            break

    # Pass 2: if rows are still too short, allow moderate downsampling (≤20×)
    if _prod(shape[best:]) < min_row_chars and best > 0:
        for s in range(best - 1, -1, -1):
            merged = shape[s:]
            elems = _prod(merged)
            lw = _compute_label_width(shape[:s], merged) + 3
            avail = max(1, term_width - lw)
            if elems / avail <= 20:
                best = s
            else:
                break

    return best


# ---------------------------------------------------------------------------
# Batch row-content builder (vectorised + optional threading)
# ---------------------------------------------------------------------------

def _build_all_row_contents(
    diff_abs: torch.Tensor,
    total_rows: int,
    merged_shape: tuple[int, ...],
    atol: float,
    max_content: int,
) -> list[str]:
    """Build the ASCII diff string for every row at once."""
    inner = _prod(merged_shape) if merged_shape else 1
    if inner == 0 or total_rows == 0:
        return [""] * total_rows

    # --- reshape to 2-D and move to numpy once ---
    flat = diff_abs.reshape(total_rows, inner).numpy()

    has_sep = len(merged_shape) > 1
    outer = int(merged_shape[0]) if has_sep else 1
    group_size = inner // outer if outer > 0 else inner

    # --- determine whether block-max downsampling is needed ---
    if has_sep:
        sep_count = outer - 1
        usable = max(1, max_content - sep_count)
        per_group = max(1, usable // max(1, outer))
        need_ds = group_size > per_group
    else:
        per_group = max_content
        need_ds = inner > max_content

    # --- downsample if necessary ---
    if need_ds:
        if has_sep:
            grouped = flat.reshape(total_rows, outer, group_size)
            blk = max(1, -(-group_size // per_group))
            cols = -(-group_size // blk)
            ds = np.zeros((total_rows, outer, cols), dtype=flat.dtype)
            for b in range(cols):
                s, e = b * blk, min((b + 1) * blk, group_size)
                ds[:, :, b] = np.nanmax(grouped[:, :, s:e], axis=2)
            levels = _arr_to_levels(ds.reshape(total_rows, outer * cols), atol)
            final_gs = cols
        else:
            blk = max(1, -(-inner // max_content))
            cols = -(-inner // blk)
            ds = np.zeros((total_rows, cols), dtype=flat.dtype)
            for b in range(cols):
                s, e = b * blk, min((b + 1) * blk, inner)
                ds[:, b] = np.nanmax(flat[:, s:e], axis=1)
            levels = _arr_to_levels(ds, atol)
            final_gs = cols
            outer = 1
            has_sep = False
    else:
        levels = _arr_to_levels(flat, atol)
        final_gs = group_size

    # --- build strings (parallel for large row counts) ---
    if total_rows > _PARALLEL_THRESHOLD:
        return _parallel_build_strings(levels, has_sep, outer, final_gs)
    return _levels_rows_to_strings(levels, has_sep, outer, final_gs)


# ---------------------------------------------------------------------------
# ASCII diff map (console + file)
# ---------------------------------------------------------------------------

def _print_diff_map(
    stats: dict[str, Any],
    atol: float,
    run_dir: str,
    term_width: int = 200,
) -> None:
    """Print an ASCII diff map to console and save to *diff_map.txt*."""
    diff_abs: torch.Tensor = stats["diff_abs"]
    shape = tuple(diff_abs.shape)
    ndim = len(shape)

    out: list[str] = []
    out.append(
        f"  Legend: · pass  - 1~3x  ~ 3~10x  * 10~50x  # 50~200x  @ >200x"
        f"  (atol={atol})"
    )

    if ndim == 0:
        tol = max(atol, 1e-12)
        lv = int(sum(1 for t in _LEVEL_THRESHOLDS if float(diff_abs.item()) / tol >= t))
        out.append(f"  [] {_LEVEL_CHARS[min(lv, 5)]}")
    else:
        split = _choose_line_split(shape, term_width=term_width)
        leading_shape = shape[:split]
        merged_shape = shape[split:]

        # --- layout description ---
        dim_lead = ", ".join(f"d{i}={shape[i]}" for i in range(split))
        dim_tail = ", ".join(f"d{i}={shape[i]}" for i in range(split, ndim))
        if split == 0:
            out.append(f"  Shape: {shape}  (all dims merged into one row)")
        else:
            out.append(f"  Shape: {shape}  Layout: [{dim_lead}] x ({dim_tail})")

        total_rows = _prod(leading_shape) if leading_shape else 1
        label_w = _compute_label_width(leading_shape, merged_shape)
        max_content = max(10, term_width - label_w - 3)

        # --- vectorised row content ---
        contents = _build_all_row_contents(
            diff_abs, total_rows, merged_shape, atol, max_content,
        )

        # --- build labels ---
        labels: list[str] = []
        for fi in range(total_rows):
            if leading_shape:
                idx = tuple(int(i) for i in np.unravel_index(fi, leading_shape))
            else:
                idx = ()
            labels.append(_format_row_label(idx, leading_shape, merged_shape))

        # --- run-length encode by content (compress identical rows) ---
        #   group = (first_flat, count, content, first_label, last_label, has_error)
        groups: list[tuple[int, int, str, str, str, bool]] = []
        for i in range(total_rows):
            c = contents[i]
            err = any(ch not in _PASS_CHARS for ch in c)
            if groups and groups[-1][2] == c:
                g = groups[-1]
                groups[-1] = (g[0], g[1] + 1, g[2], g[3], labels[i], g[5] or err)
            else:
                groups.append((i, 1, c, labels[i], labels[i], err))

        # --- emit lines ---
        for _, count, content, first_lbl, last_lbl, has_err in groups:
            fl = first_lbl.ljust(label_w)
            if count == 1:
                out.append(f"  {fl} {content}")
            elif count == 2:
                ll = last_lbl.ljust(label_w)
                out.append(f"  {fl} {content}")
                out.append(f"  {ll} {content}")
            else:
                out.append(f"  {fl} {content}")
                tag = "same pattern" if has_err else "all pass"
                out.append(
                    f"    ... ({count - 1} more rows, {tag},"
                    f" to {last_lbl}) ..."
                )

    # --- write to file ---
    with open(os.path.join(run_dir, "diff_map.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    # --- print to console ---
    print("\n".join(out))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prec_assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    dtype: Optional[Union[str, torch.dtype]] = None,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    output_dir: str = ".precision_debug",
    save_tensors: bool = True,
    print_map: bool = True,
    term_width: int = 200,
    msg: str = "",
    equal_nan: bool = True,
) -> None:
    """Drop-in replacement for ``torch.testing.assert_close``.

    On mismatch the function:
    1. Writes a text report (``report.txt``).
    2. Saves actual / expected tensors (``actual.pt``, ``expected.pt``).
    3. Prints an ASCII diff map to the console (and ``diff_map.txt``).
    4. Raises ``AssertionError``.
    """
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Shape mismatch: actual {actual.shape} vs expected {expected.shape}. " + msg
        )

    # Handle dtype-based default tolerances (Compatibility with testing/npuir/testcommon.py)
    if rtol is None or atol is None:
        target_dtype = dtype if dtype is not None else actual.dtype
        # Convert torch.dtype to string name
        if isinstance(target_dtype, torch.dtype):
            dname = str(target_dtype).split(".")[-1]
        else:
            dname = str(target_dtype)

        def_rtol, def_atol = DEFAULT_TOLERANCE.get(dname, (1e-5, 1e-5))
        if rtol is None:
            rtol = def_rtol
        if atol is None:
            atol = def_atol

    a, b = _equalize_for_compare(actual, expected)
    if torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan).all():
        return

    stats = _compute_stats(actual, expected, rtol, atol, equal_nan=equal_nan)
    filename, func = _get_caller_info()
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in f"{filename}_{func}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"{safe}_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    _write_report(run_dir, stats, msg)

    if save_tensors:
        torch.save(actual.cpu(), os.path.join(run_dir, "actual.pt"))
        torch.save(expected.cpu(), os.path.join(run_dir, "expected.pt"))

    if print_map:
        try:
            _print_diff_map(stats, atol, run_dir, term_width=term_width)
        except Exception as e:
            with open(os.path.join(run_dir, "report.txt"), "a", encoding="utf-8") as f:
                f.write(f"\n\nASCII diff map failed: {e}\n")

    print(
        f"[PrecisionDebug] FAILED: shape={stats['shape']} dtype={stats['dtype']}\n"
        f"  Mismatched: {stats['num_mismatched']}/{stats['total']}"
        f" ({100.0 * stats['num_mismatched'] / max(1, stats['total']):.2f}%)\n"
        f"  Max abs diff: {stats['max_abs']} at index {stats['max_abs_idx']}\n"
        f"  Max rel diff: {stats['max_rel']} at index {stats['max_rel_idx']}\n"
        f"  Mean abs diff: {stats['mean_abs']}\n"
        f"  Output saved to: {run_dir}"
    )

    raise AssertionError(
        f"Tensors not close (rtol={rtol}, atol={atol}). "
        f"Mismatched: {stats['num_mismatched']}/{stats['total']}. "
        f"See {run_dir}. " + msg
    )
