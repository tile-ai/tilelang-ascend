# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""
IR Pass Trace - Zero-intrusion debug tool for visualizing compilation passes.

Generates a self-contained HTML page showing all passes in the compilation
pipeline with side-by-side IR diff for each pass.

Usage:
    import tilelang
    import tilelang.engine.pass_trace; tilelang.engine.pass_trace.patch()

    # Then run your kernel as normal:
    TILELANG_DUMP_PASSES=1 python my_kernel.py

    # Open the generated HTML:
    open ir_dump/ir_trace.html

Environment variables:
    TILELANG_DUMP_PASSES: 0/unset=off, 1/all=all phases, phase1, phase2
    TILELANG_DUMP_DIR:    output directory (default: ./tmp/ir_dump/{kernel_name}_{timestamp}/)
"""

from __future__ import annotations
import os
import difflib
from dataclasses import dataclass, field
from typing import List


@dataclass
class PassRecord:
    """Result of running a single pass."""
    phase: str
    name: str
    index: int
    before_text: str
    after_text: str
    changed: bool
    add_lines: int = 0
    del_lines: int = 0


# ---------------------------------------------------------------------------
# Global state: records collected during compilation
# ---------------------------------------------------------------------------
_records: List[PassRecord] = []


# ---------------------------------------------------------------------------
# Dump control
# ---------------------------------------------------------------------------
def _is_dump_enabled_for_phase(phase: str) -> bool:
    """Check env var to decide if dumping is enabled for this phase."""
    mode = os.environ.get("TILELANG_DUMP_PASSES", "0")
    if mode in ("0", "", "off", "false"):
        return False
    if mode in ("1", "all", "on", "true"):
        return True
    if mode == "phase1":
        return "phase1" in phase
    if mode == "phase2":
        return "phase2" in phase
    return False


# ---------------------------------------------------------------------------
# Dump directory initialization (lazy, once per compilation)
# ---------------------------------------------------------------------------
_dump_dir: str | None = None


def _ensure_dump_dir() -> str:
    """Initialize and return the dump directory path (created on first call).

    Default path: ./tmp/ir_dump/{kernel_name}_YYYYMMDDHHmmSS/
    Override with TILELANG_DUMP_DIR env var.
    """
    global _dump_dir

    if _dump_dir is not None:
        return _dump_dir

    # User override takes precedence
    env_dir = os.environ.get("TILELANG_DUMP_DIR", "")
    if env_dir:
        _dump_dir = env_dir
    else:
        from datetime import datetime
        import sys
        script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0] or "kernel"
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        _dump_dir = os.path.join(".", "tmp", "ir_dump", f"{script_name}_{timestamp}")

    os.makedirs(_dump_dir, exist_ok=True)
    return _dump_dir


# ---------------------------------------------------------------------------
# Core: run_pass
# ---------------------------------------------------------------------------
def run_pass(pass_obj, mod, pass_name: str, phase_name: str, pass_index: int):
    """Execute a single pass, capturing before/after IR.

    Args:
        pass_obj:   A TVM Pass object (result of pass_factory())
        mod:        Current IRModule
        pass_name:  Human-readable pass name for display
        phase_name: Phase identifier (e.g. "phase1_LowerAndLegalize")
        pass_index: Sequential index within the phase

    Returns:
        The transformed IRModule (result of pass_obj(mod))
    """
    global _records

    should_dump = _is_dump_enabled_for_phase(phase_name)

    if should_dump:
        _ensure_dump_dir()
        before_text = str(mod)
    else:
        before_text = ""

    # Execute the actual pass
    mod = pass_obj(mod)

    if should_dump:
        after_text = str(mod)
        changed = before_text != after_text

        add_count = 0
        del_count = 0
        if changed:
            # Compute add/del counts via SequenceMatcher
            sm = difflib.SequenceMatcher(None, before_text.splitlines(), after_text.splitlines())
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "insert":
                    add_count += j2 - j1
                elif tag == "delete":
                    del_count += i2 - i1
                elif tag == "replace":
                    add_count += j2 - j1
                    del_count += i2 - i1

        record = PassRecord(
            phase=phase_name,
            name=pass_name,
            index=pass_index,
            before_text=before_text,
            after_text=after_text,
            changed=changed,
            add_lines=add_count,
            del_lines=del_count,
        )
        _records.append(record)

        # Also dump raw .tir files (useful for BeyondCompare)
        _save_raw_files(record)

        # Console progress
        tag = "CHANGED" if changed else "NO-OP"
        print(f"  [pass_trace] {phase_name}/{pass_index:02d}_{pass_name}: {tag}")

    return mod


def _save_raw_files(record: PassRecord):
    """Write before/after .tir files to disk (phase subdirectory layout)."""
    dump_dir = _dump_dir
    if not dump_dir:
        return

    phase_dir = os.path.join(dump_dir, record.phase)
    os.makedirs(phase_dir, exist_ok=True)

    prefix = f"{record.index:02d}_{record.name}"
    with open(os.path.join(phase_dir, f"{prefix}_before.tir"), "w") as f:
        f.write(record.before_text)
    with open(os.path.join(phase_dir, f"{prefix}_after.tir"), "w") as f:
        f.write(record.after_text)


# ---------------------------------------------------------------------------
# Debug phase functions (re-implementations of phase.py with dump wrappers)
# ---------------------------------------------------------------------------
def debug_LowerAndLegalize(mod, target):
    """Debug version of LowerAndLegalize with IR dump per pass.

    Pass sequence is identical to tilelang.engine.phase.LowerAndLegalize.
    """
    # Reset state for a fresh compilation
    reset()

    import tilelang.transform
    from tilelang import tvm as tvm
    from tvm import tir

    phase = "phase1_LowerAndLegalize"

    # allocate the tmp buffer for vector api
    mod = run_pass(tilelang.transform.InjectTmpBuffer(target), mod, "InjectTmpBuffer", phase, 0)
    mod = run_pass(tilelang.transform.AscendInferBufferScope(), mod, "AscendInferBufferScope", phase, 1)
    # Vid reduction
    mod = run_pass(tilelang.transform.AscendVidReduction(), mod, "AscendVidReduction", phase, 2)
    # Collect buffer shape
    mod = run_pass(tilelang.transform.BufferShapeCollector(), mod, "BufferShapeCollector", phase, 3)
    # Bind the target device information to the module
    mod = run_pass(tir.transform.BindTarget(target), mod, "BindTarget", phase, 4)
    # Identify and filter host tiling data for npu
    mod = run_pass(tilelang.transform.HostProcesser(), mod, "HostProcesser", phase, 5)
    # Simplify the IR expressions
    mod = run_pass(tir.transform.Simplify(), mod, "Simplify", phase, 6)
    # Lower parallel loops to vector instructions for Ascend.
    mod = run_pass(tilelang.transform.AscendLowerParallelToVector(), mod, "AscendLowerParallelToVector", phase, 7)
    # Infer memory layouts for fragments and shared memory
    mod = run_pass(tilelang.transform.LayoutInference(), mod, "LayoutInference", phase, 8)
    mod = run_pass(tilelang.transform.CollectBufferShapes(), mod, "CollectBufferShapes", phase, 9)
    # Lower high-level tile operations to low-level operations
    mod = run_pass(tilelang.transform.LowerTileOp(), mod, "LowerTileOp", phase, 10)
    # Erase manual workspace allocations for virtual CV copy in Ascend
    mod = run_pass(tilelang.transform.AscendWorkspaceReduction(), mod, "AscendWorkspaceReduction", phase, 11)
    # Legalize vectorized loops to ensure they are valid
    mod = run_pass(tilelang.transform.LegalizeVectorizedLoop(), mod, "LegalizeVectorizedLoop", phase, 12)
    # Add safety checks for memory accesses
    mod = run_pass(tilelang.transform.LegalizeSafeMemoryAccess(), mod, "LegalizeSafeMemoryAccess", phase, 13)
    # Simplify again to clean up any duplicated conditions
    mod = run_pass(tir.transform.Simplify(), mod, "Simplify", phase, 14)

    return mod


def debug_OptimizeForTarget(mod, target, platform):
    """Debug version of OptimizeForTarget with IR dump per pass.

    Pass sequence is identical to tilelang.engine.phase.OptimizeForTarget.
    """
    import tilelang.transform
    from tilelang import tvm as tvm
    from tvm import tir

    # Lazy imports to avoid circular dependency
    from tilelang.engine.phase import allow_vectorize
    from tilelang.utils.target import check_npu_availability

    phase = "phase2_OptimizeForTarget"
    pass_ctx = tilelang.transform.get_pass_context()

    mod = run_pass(tir.transform.PlanAndUpdateBufferAllocationLocation(), mod, "PlanAndUpdateBufferAllocationLocation", phase, 0)
    mod = run_pass(tilelang.transform.CrossCorePipeline(), mod, "CrossCorePipeline", phase, 1)
    mod = run_pass(tilelang.transform.CombineCV(), mod, "CombineCV", phase, 2)
    mod = run_pass(tilelang.transform.PipelinePlanning(), mod, "PipelinePlanning", phase, 3)
    mod = run_pass(tilelang.transform.InjectSoftwarePipeline(), mod, "InjectSoftwarePipeline", phase, 4)
    mod = run_pass(tilelang.transform.AscendLowerOpaqueBlock(), mod, "AscendLowerOpaqueBlock", phase, 5)
    mod = run_pass(tir.transform.NarrowDataType(32), mod, "NarrowDataType", phase, 6)
    mod = run_pass(tilelang.transform.ConfigIndexBitwidth(), mod, "ConfigIndexBitwidth", phase, 7)
    # Collect buffer shape and flatten buffer shape to 2D
    mod = run_pass(tilelang.transform.Flatten2DBuffer(), mod, "Flatten2DBuffer", phase, 8)
    mod = run_pass(tilelang.transform.FlattenBuffer(), mod, "FlattenBuffer", phase, 9)
    mod = run_pass(tir.transform.Simplify(), mod, "Simplify", phase, 10)
    mod = run_pass(
        tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx)),
        mod, "VectorizeLoop", phase, 11,
    )
    mod = run_pass(
        tilelang.transform.AscendStorageRewrite(is_npu=check_npu_availability()),
        mod, "AscendStorageRewrite", phase, 12,
    )
    mod = run_pass(tir.transform.UnrollLoop(), mod, "UnrollLoop", phase, 13)
    mod = run_pass(tir.transform.RenormalizeSplitPattern(), mod, "RenormalizeSplitPattern", phase, 14)
    mod = run_pass(tir.transform.Simplify(), mod, "Simplify", phase, 15)
    mod = run_pass(tir.transform.RemoveNoOp(), mod, "RemoveNoOp", phase, 16)
    mod = run_pass(tir.transform.RewriteUnsafeSelect(), mod, "RewriteUnsafeSelect", phase, 17)
    mod = run_pass(tir.transform.HoistIfThenElse(), mod, "HoistIfThenElse", phase, 18)
    mod = run_pass(tilelang.transform.AscendMemoryPlanning(), mod, "AscendMemoryPlanning", phase, 19)
    mod = run_pass(tilelang.transform.AscendSyncInsert(target, platform), mod, "AscendSyncInsert", phase, 20)

    # After all passes complete, generate the HTML report
    if _records and _dump_dir:
        html_path = os.path.join(_dump_dir, "ir_trace.html")
        generate_html(_records, html_path)
        print(f"  [pass_trace] HTML report written to: {html_path}")

    return mod


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------
_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #f5f7fa;
    color: #1e293b;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

/* ---- Header ---- */
.header {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
    padding: 12px 20px;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.header h1 { font-size: 17px; font-weight: 600; }
.header .sub { font-size: 12px; opacity: 0.6; margin-top: 2px; }

/* ---- Phase tabs ---- */
.phase-tabs {
    display: flex;
    background: #e2e8f0;
    border-bottom: 1px solid #cbd5e1;
    flex-shrink: 0;
}
.phase-tab {
    padding: 8px 20px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    color: #64748b;
    border-bottom: 3px solid transparent;
    transition: all 0.15s;
    user-select: none;
}
.phase-tab:hover { background: #f1f5f9; color: #1e293b; }
.phase-tab.active {
    color: #2563eb;
    border-bottom-color: #2563eb;
    background: #f5f7fa;
}

/* ---- Summary bar ---- */
.summary-bar {
    background: white;
    padding: 8px 20px;
    border-bottom: 1px solid #e2e8f0;
    font-size: 13px;
    flex-shrink: 0;
}
.summary-bar .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 10px;
    margin-right: 8px;
    font-weight: 600;
    font-size: 12px;
}
.badge-total   { background: #e2e8f0; color: #475569; }
.badge-changed { background: #dcfce7; color: #166534; }
.badge-noop    { background: #f1f5f9; color: #94a3b8; }

/* ---- Main layout ---- */
.main {
    display: flex;
    flex: 1;
    overflow: hidden;
}

/* ---- Sidebar ---- */
.sidebar {
    width: 270px;
    min-width: 270px;
    background: white;
    border-right: 1px solid #e2e8f0;
    overflow-y: auto;
    padding: 8px 0;
    flex-shrink: 0;
}
.sidebar .section-title {
    padding: 8px 14px 4px;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    font-weight: 700;
}

.pass-link {
    display: flex;
    align-items: center;
    padding: 5px 14px;
    font-size: 12.5px;
    cursor: pointer;
    color: #334155;
    text-decoration: none;
    transition: background 0.1s;
    gap: 7px;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
}
.pass-link:hover { background: #f1f5f9; }
.pass-link.active { background: #eff6ff; color: #2563eb; }

.pass-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.pass-dot.changed { background: #22c55e; }
.pass-dot.noop    { background: #d1d5db; }

.pass-label {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex: 1;
}

.pass-idx {
    font-size: 10px;
    color: #94a3b8;
    flex-shrink: 0;
    width: 18px;
    text-align: right;
}

.pass-stats {
    font-size: 10px;
    flex-shrink: 0;
    white-space: nowrap;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
}
.pass-stats .st-add { color: #1a7f37; }
.pass-stats .st-del { color: #cf222e; }

/* ---- Content area ---- */
.content {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
}

.pass-section {
    display: none;
    background: white;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.05);
    padding: 20px;
    margin-bottom: 16px;
}
.pass-section.active { display: block; }

.pass-section.collapsed > *:not(.pass-header) { display: none; }

.pass-header {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #f1f5f9;
    border-radius: 4px;
    transition: background 0.1s;
    position: sticky;
    top: 0;
    z-index: 10;
    background: white;
}
.pass-header.collapsible { cursor: pointer; user-select: none; }
.pass-header.collapsible:hover { background: #f8fafc; }

.pass-toggle {
    width: 16px;
    flex-shrink: 0;
    font-size: 10px;
    color: #94a3b8;
    text-align: center;
}
.pass-section:not(.collapsed) > .pass-header .pass-toggle::before { content: '\\25BC'; }
.pass-section.collapsed > .pass-header .pass-toggle::before { content: '\\25B6'; }

.pass-header h2 { font-size: 15px; font-weight: 600; }
.pass-header .status {
    font-size: 11px;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 10px;
    letter-spacing: 0.03em;
    margin-left: auto;
}
.status-changed { background: #dcfce7; color: #166534; }
.status-noop    { background: #f1f5f9; color: #94a3b8; }

.noop-msg {
    color: #94a3b8;
    font-size: 13px;
    text-align: center;
    padding: 16px;
}

.ir-toggle {
    display: block;
    margin: 10px auto 0;
    padding: 5px 18px;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 5px;
    cursor: pointer;
    font-size: 12px;
    color: #64748b;
    transition: background 0.1s;
}
.ir-toggle:hover { background: #e2e8f0; }

.ir-block {
    display: none;
    margin-top: 10px;
    max-height: 500px;
    overflow: auto;
}
.ir-block.show { display: block; }

.ir-block pre {
    background: #0f172a;
    color: #e2e8f0;
    padding: 14px;
    border-radius: 6px;
    font-size: 11.5px;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    line-height: 1.55;
    white-space: pre;
    tab-size: 2;
}

/* ---- GitHub-style diff table ---- */
.diff-table-wrap {
    overflow-x: auto;
    border-radius: 6px;
    border: 1px solid #d0d7de;
    background: #ffffff;
}
.diff-table-wrap table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 12px;
    line-height: 20px;
    table-layout: fixed;
}
.diff-table-wrap td {
    padding: 0 10px;
    white-space: pre-wrap;
    word-break: break-all;
    vertical-align: top;
}

/* Line number gutter */
.diff-table-wrap .ln {
    width: 50px;
    min-width: 50px;
    max-width: 50px;
    text-align: right;
    padding: 0 8px;
    color: #8c959f;
    background: #f6f8fa;
    border-right: 1px solid #d0d7de;
    user-select: none;
    font-size: 11px;
}

/* Sign column (+/-) */
.diff-table-wrap .sg {
    width: 20px;
    min-width: 20px;
    max-width: 20px;
    text-align: center;
    padding: 0;
    user-select: none;
    font-weight: 700;
}

/* Equal (unchanged) lines */
.diff-table-wrap .ln-eq { background: #f6f8fa; }
.diff-table-wrap .eq { background: #ffffff; }

/* Deleted lines */
.diff-table-wrap .ln-del { background: #ffd7d5; color: #82071e; }
.diff-table-wrap .sg-del { background: #ffd7d5; color: #cf222e; }
.diff-table-wrap .del { background: #ffebe9; color: #24292f; }
.diff-table-wrap .del-word { background: #ffcecb; border-radius: 2px; }

/* Added lines */
.diff-table-wrap .ln-add { background: #abf2ca; color: #116329; }
.diff-table-wrap .sg-add { background: #abf2ca; color: #1a7f37; }
.diff-table-wrap .add { background: #dafbe1; color: #24292f; }
.diff-table-wrap .add-word { background: #acf2bd; border-radius: 2px; }

/* Whitespace-only changes (subtle styling, closer to equal) */
.diff-table-wrap .ln-ws { background: #e8e8f8; color: #4040a0; }
.diff-table-wrap .sg-ws { background: #e8e8f8; color: #6060b0; }
.diff-table-wrap .ws { background: #f0f0ff; color: #24292f; }
.diff-table-wrap .ws .del-word { background: #d8d8f0; border-radius: 2px; }
.diff-table-wrap .ws .add-word { background: #d8d8f0; border-radius: 2px; }

/* Hidden (collapsible) rows */
.diff-table-wrap tr.row-hidden { display: none; }

/* Row highlight (click line number to highlight) */
.diff-table-wrap tr.row-hl td { background: #fff8c5 !important; }
.diff-table-wrap td.ln:not(:empty) { cursor: pointer; }
.diff-table-wrap td.ln:not(:empty):hover { text-decoration: underline; }

/* ---- Manual alignment mode (Beyond Compare style) ---- */
.align-status {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 999;
    padding: 8px 20px;
    font-size: 13px;
    font-family: 'Segoe UI', system-ui, sans-serif;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.align-status.active { display: block; }
.align-status.mode-left { background: #fff3cd; color: #664d03; }
.align-status.mode-right { background: #cfe2ff; color: #084298; }
.align-status kbd {
    display: inline-block;
    padding: 1px 6px;
    font-size: 11px;
    font-family: inherit;
    background: rgba(0,0,0,0.08);
    border: 1px solid rgba(0,0,0,0.15);
    border-radius: 3px;
    margin: 0 2px;
}

/* Left line selected (step 1) */
.diff-table-wrap td.align-pending {
    outline: 2px solid #f59e0b;
    outline-offset: -2px;
    background: #fef3c7 !important;
}
.diff-table-wrap td.align-pending-sibling {
    background: #fef3c7 !important;
}

/* Waiting for right click (step 2, after F7) */
.diff-table-wrap td.align-locked {
    outline: 2px solid #3b82f6;
    outline-offset: -2px;
    background: #dbeafe !important;
}
.diff-table-wrap td.align-locked-sibling {
    background: #dbeafe !important;
}

/* Right line candidate hover */
.diff-table-wrap.waiting-right td[data-side="r"]:not(:empty) {
    cursor: crosshair;
}
.diff-table-wrap.waiting-right td[data-side="r"]:not(:empty):hover {
    outline: 2px dashed #3b82f6;
    outline-offset: -2px;
}

/* Successfully aligned row */
.diff-table-wrap tr.row-aligned {
    border-left: 3px solid #f59e0b;
}

/* Context expand toolbar */
.diff-toolbar {
    display: flex;
    gap: 6px;
    margin-bottom: 6px;
    padding: 6px 8px;
    background: #f6f8fa;
    border: 1px solid #d0d7de;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
}
.diff-toolbar + .diff-table-wrap { border-radius: 0 0 6px 6px; }
.diff-toolbar button {
    padding: 3px 10px;
    font-size: 12px;
    cursor: pointer;
    background: #fff;
    border: 1px solid #d0d7de;
    border-radius: 4px;
    color: #24292f;
    line-height: 20px;
    transition: background 0.1s;
}
.diff-toolbar button:hover:not(:disabled) { background: #f3f4f6; }
.diff-toolbar button:disabled { opacity: 0.35; cursor: default; }

/* Inline expand button rows (GitHub-style full-row) */
.diff-table-wrap .btn-row td {
    text-align: center;
    padding: 4px 8px;
    background: #ddf4ff;
    color: #0969da;
    cursor: pointer;
    font-size: 12px;
    border-top: 1px solid #54aeff;
    border-bottom: 1px solid #54aeff;
    user-select: none;
    line-height: 20px;
}
.diff-table-wrap .btn-row td:hover { background: #c8e9ff; }
.diff-table-wrap .btn-row.all-expanded td {
    display: none;
}
.diff-table-wrap .btn-row .exp-arrow {
    font-weight: 700;
    font-size: 14px;
    padding: 0 2px;
}
.diff-table-wrap .btn-row .exp-label {
    padding: 0 4px;
}

/* ---- Empty state ---- */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #94a3b8;
    font-size: 14px;
}

/* Copy buttons */
.btn-copy { float: right; }
.copy-spacer { flex: 1; }
.copy-toast {
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: #1f2937;
    color: #fff;
    padding: 8px 18px;
    border-radius: 6px;
    font-size: 13px;
    z-index: 9999;
    animation: toastFade 1.5s ease forwards;
    pointer-events: none;
}
@keyframes toastFade {
    0%,60% { opacity: 1; }
    100% { opacity: 0; }
}
"""

_JS = """
/* ---- P4: Alignment mode global state ---- */
var _alignMode = null;      // null | 'left' | 'right'
var _pendingLeft = null;    // left td element selected
var _alignStatus = null;    // status bar element (initialized on DOMContentLoaded)

function cancelAlign() {
    if (_pendingLeft) {
        var row = _pendingLeft.closest('tr');
        if (row) row.querySelectorAll('.align-pending,.align-pending-sibling,.align-locked,.align-locked-sibling')
            .forEach(function(c){ c.classList.remove('align-pending','align-pending-sibling','align-locked','align-locked-sibling'); });
    }
    document.querySelectorAll('.align-locked,.align-locked-sibling').forEach(function(c){ c.classList.remove('align-locked','align-locked-sibling'); });
    document.querySelectorAll('.waiting-right').forEach(function(c){ c.classList.remove('waiting-right'); });
    _pendingLeft = null;
    _alignMode = null;
    if (_alignStatus) { _alignStatus.className = 'align-status'; _alignStatus.innerHTML = ''; }
}

function showAlignStatus(mode, msg) {
    if (!_alignStatus) return;
    _alignStatus.className = 'align-status active mode-' + mode;
    _alignStatus.innerHTML = msg;
}

function showPass(el, id) {
    document.querySelectorAll('.pass-section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.pass-link').forEach(l => l.classList.remove('active'));
    var sec = document.getElementById(id);
    if (sec) sec.classList.add('active');
    if (el) el.classList.add('active');
    if (typeof cancelAlign === 'function') cancelAlign();
}

function showPhase(el, phase) {
    document.querySelectorAll('.phase-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.sidebar').forEach(s => s.style.display = 'none');
    document.querySelectorAll('.summary-bar').forEach(s => s.style.display = 'none');
    if (el) el.classList.add('active');
    var sb = document.getElementById('sb-' + phase);
    if (sb) sb.style.display = '';
    var sm = document.getElementById('sm-' + phase);
    if (sm) sm.style.display = '';
    // hide all sections, show first of this phase
    document.querySelectorAll('.pass-section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.pass-link').forEach(l => l.classList.remove('active'));
    var first = document.querySelector('[data-phase="' + phase + '"]');
    if (first) {
        var sid = first.getAttribute('data-target');
        showPass(first, sid);
    }
}

function toggleIr(btn) {
    var block = btn.nextElementSibling;
    if (block) {
        block.classList.toggle('show');
        btn.textContent = block.classList.contains('show')
            ? '\\u25BC Collapse IR' : '\\u25B6 Show full IR';
    }
}

function toggleCollapse(el) {
    var sec = el.closest('.pass-section');
    sec.classList.toggle('collapsed');
}

function copyIr(btn, side) {
    var sec = btn.closest('.pass-section');
    var el = sec.querySelector('.ir-data-' + side);
    if (!el) return;
    var text = el.textContent;
    var label = side === 'before' ? 'Before' : side === 'after' ? 'After' : 'IR';
    function showToast() {
        var toast = document.createElement('div');
        toast.className = 'copy-toast';
        toast.textContent = 'Copied ' + label + ' IR';
        document.body.appendChild(toast);
        setTimeout(function() { toast.remove(); }, 1600);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(showToast).catch(function() {
            fallbackCopy(text, showToast);
        });
    } else {
        fallbackCopy(text, showToast);
    }
}
function fallbackCopy(text, cb) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.style.cssText = 'position:fixed;left:-9999px';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); if (cb) cb(); } catch(e) {}
    ta.remove();
}

function expand(el, n, evt) {
    var run = el.dataset.run;
    var tr = el.closest('tr');
    var table = tr.closest('table');
    var limit = (evt && evt.altKey) ? 999999 : n;
    var shown = 0, r;
    r = tr.previousElementSibling;
    while (r && shown < limit) {
        if (!r.classList.contains('row-hidden') || r.dataset.run !== run) break;
        r.classList.remove('row-hidden'); shown++;
        r = r.previousElementSibling;
    }
    shown = 0;
    r = tr.nextElementSibling;
    while (r && shown < limit) {
        if (!r.classList.contains('row-hidden') || r.dataset.run !== run) break;
        r.classList.remove('row-hidden'); shown++;
        r = r.nextElementSibling;
    }
    updBtns(table);
}

function expandAll(btn) {
    var table = btn.closest('.pass-section').querySelector('table');
    table.querySelectorAll('tr.row-hidden').forEach(function(r) { r.classList.remove('row-hidden'); });
    updBtns(table);
}

function collapseCtx(btn) {
    var table = btn.closest('.pass-section').querySelector('table');
    table.querySelectorAll('tr[data-collapse="1"]').forEach(function(r) {
        if (!r.classList.contains('btn-row')) r.classList.add('row-hidden');
    });
    updBtns(table);
}

function updBtns(table) {
    var sec = table.closest('.pass-section');
    var hid = table.querySelectorAll('tr.row-hidden');
    var ba = sec.querySelector('.btn-expand-all');
    if (hid.length === 0) {
        table.querySelectorAll('.btn-row').forEach(function(r) { r.classList.add('all-expanded'); });
        if (ba) ba.disabled = true;
        return;
    }
    table.querySelectorAll('.btn-row td[data-run]').forEach(function(td) {
        var run = td.dataset.run;
        var tr = td.closest('tr');
        var hasHidden = false;
        var r = tr.previousElementSibling;
        while (r) {
            if (r.classList.contains('row-hidden') && r.dataset.run === run) { hasHidden = true; break; }
            if (!r.classList.contains('row-hidden') || r.dataset.run !== run) break;
            r = r.previousElementSibling;
        }
        if (!hasHidden) {
            r = tr.nextElementSibling;
            while (r) {
                if (r.classList.contains('row-hidden') && r.dataset.run === run) { hasHidden = true; break; }
                if (!r.classList.contains('row-hidden') || r.dataset.run !== run) break;
                r = r.nextElementSibling;
            }
        }
        tr.style.display = hasHidden ? '' : 'none';
    });
    if (ba) ba.disabled = false;
}

/* ---- P4: Manual alignment (Beyond Compare style) ---- */

function escHtml(text) {
    return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function jsInlineDiff(before, after) {
    var m = before.length, n = after.length;
    var left = [], right = [], isWsOnly = true;
    var sm = [];
    for (var i = 0; i <= m; i++) {
        sm[i] = [];
        for (var j = 0; j <= n; j++) sm[i][j] = 0;
    }
    for (var i = m - 1; i >= 0; i--) {
        for (var j = n - 1; j >= 0; j--) {
            if (before[i] === after[j]) sm[i][j] = sm[i+1][j+1] + 1;
            else sm[i][j] = Math.max(sm[i+1][j], sm[i][j+1]);
        }
    }
    var i = 0, j = 0;
    while (i < m && j < n) {
        if (before[i] === after[j]) {
            left.push(escHtml(before[i]));
            right.push(escHtml(after[j]));
            i++; j++;
        } else if (sm[i+1][j] >= sm[i][j+1]) {
            var chunk = before[i];
            var start = i;
            i++;
            while (i < m && (j >= n || sm[i+1] && sm[i+1][j] >= sm[i][j+1]) && before[i] !== after[j]) {
                chunk += before[i]; i++;
            }
            if (chunk.trim() !== '') isWsOnly = false;
            left.push('<span class="del-word">' + escHtml(chunk) + '</span>');
        } else {
            var chunk = after[j];
            var start = j;
            j++;
            while (j < n && (i >= m || sm[i][j+1] > sm[i+1][j]) && before[i] !== after[j]) {
                chunk += after[j]; j++;
            }
            if (chunk.trim() !== '') isWsOnly = false;
            right.push('<span class="add-word">' + escHtml(chunk) + '</span>');
        }
    }
    if (i < m) {
        var rest = before.slice(i);
        if (rest.trim() !== '') isWsOnly = false;
        left.push('<span class="del-word">' + escHtml(rest) + '</span>');
    }
    if (j < n) {
        var rest = after.slice(j);
        if (rest.trim() !== '') isWsOnly = false;
        right.push('<span class="add-word">' + escHtml(rest) + '</span>');
    }
    return { left: left.join(''), right: right.join(''), isWsOnly: isWsOnly };
}

function getBeforeLines(section) {
    var el = section.querySelector('.ir-data-before');
    return el ? el.textContent.split('\\n') : [];
}
function getAfterLines(section) {
    var el = section.querySelector('.ir-data-after');
    return el ? el.textContent.split('\\n') : [];
}

function alignRows(leftTd, rightTd) {
    var leftRow = leftTd.closest('tr');
    var rightRow = rightTd.closest('tr');
    if (leftRow === rightRow) return;

    var lIdx = parseInt(leftTd.getAttribute('data-idx'), 10);
    var rIdx = parseInt(rightTd.getAttribute('data-idx'), 10);
    var section = leftRow.closest('.pass-section');
    var beforeLines = getBeforeLines(section);
    var afterLines = getAfterLines(section);

    var leftBeforeLine = beforeLines[lIdx] || '';
    var rightAfterLine = afterLines[rIdx] || '';

    // Collect cell references from both rows (3 cells each side, 6 total per row)
    var lCells = Array.from(leftRow.children);   // [ln, sg, code, ln, sg, code]
    var rCells = Array.from(rightRow.children);

    // Helper: read a cell's text content
    function cellText(cells, side, pos) {
        // side: 0=left(0-2), 1=right(3-5); pos: 0=ln, 1=sg, 2=code
        var idx = side * 3 + pos;
        return cells[idx] ? cells[idx].textContent : '';
    }

    // Save content from both rows before any modification
    var leftContent = {
        ln: cellText(lCells, 0, 0), sg: cellText(lCells, 0, 1), code: lCells[2] ? lCells[2].innerHTML : '',
        lnCls: lCells[0] ? lCells[0].className : '', sgCls: lCells[1] ? lCells[1].className : '', codeCls: lCells[2] ? lCells[2].className : ''
    };
    var rightContent = {
        ln: cellText(rCells, 1, 0), sg: cellText(rCells, 1, 1), code: rCells[5] ? rCells[5].innerHTML : '',
        lnCls: rCells[3] ? rCells[3].className : '', sgCls: rCells[4] ? rCells[4].className : '', codeCls: rCells[5] ? rCells[5].className : ''
    };

    // Check if leftRow's right side has content that will be displaced
    var leftRowHasRight = cellText(lCells, 1, 0) !== '' || cellText(lCells, 1, 1) !== '' || cellText(lCells, 1, 2) !== '';
    // Check if rightRow's left side has content that will be displaced
    var rightRowHasLeft = cellText(rCells, 0, 0) !== '' || cellText(rCells, 0, 1) !== '' || cellText(rCells, 0, 2) !== '';

    // Create orphan rows for displaced content (by cloning, before modifying originals)
    if (leftRowHasRight) {
        var orphanRow = document.createElement('tr');
        // Empty left side
        orphanRow.innerHTML = '<td class="ln"></td><td class="sg"></td><td></td>';
        // Clone right side from leftRow
        for (var k = 3; k < 6; k++) {
            if (lCells[k]) orphanRow.appendChild(lCells[k].cloneNode(true));
        }
        leftRow.parentNode.insertBefore(orphanRow, leftRow.nextSibling);
    }
    if (rightRowHasLeft) {
        var orphanRow = document.createElement('tr');
        // Clone left side from rightRow
        for (var k = 0; k < 3; k++) {
            if (rCells[k]) orphanRow.appendChild(rCells[k].cloneNode(true));
        }
        // Empty right side
        var emptyR = document.createElement('td'); emptyR.className = 'ln'; orphanRow.appendChild(emptyR);
        var emptyS = document.createElement('td'); emptyS.className = 'sg'; orphanRow.appendChild(emptyS);
        orphanRow.appendChild(document.createElement('td'));
        rightRow.parentNode.insertBefore(orphanRow, rightRow);
    }

    // Now update leftRow: set left side from leftContent, right side from rightContent
    // Left side of leftRow stays as the aligned left
    if (lCells[0]) { lCells[0].className = leftContent.lnCls; lCells[0].textContent = leftContent.ln; }
    if (lCells[1]) { lCells[1].className = leftContent.sgCls; lCells[1].textContent = leftContent.sg; }
    if (lCells[2]) { lCells[2].className = leftContent.codeCls; lCells[2].innerHTML = leftContent.code; }
    // Right side of leftRow gets the right content
    if (lCells[3]) { lCells[3].className = rightContent.lnCls; lCells[3].textContent = rightContent.ln; }
    if (lCells[4]) { lCells[4].className = rightContent.sgCls; lCells[4].textContent = rightContent.sg; }
    if (lCells[5]) { lCells[5].className = rightContent.codeCls; lCells[5].innerHTML = rightContent.code; }

    // Remove rightRow (its content is now in leftRow)
    if (rightRow.parentNode) rightRow.parentNode.removeChild(rightRow);

    // Compute inline diff and update cell styling on leftRow
    var diff = jsInlineDiff(leftBeforeLine, rightAfterLine);

    // leftRow cells: [0]=leftLn, [1]=leftSg, [2]=leftCode, [3]=rightLn, [4]=rightSg, [5]=rightCode
    if (diff.isWsOnly) {
        lCells[0].className = 'ln ln-ws';
        lCells[1].className = 'sg sg-ws'; lCells[1].textContent = '~';
        lCells[2].className = 'ws'; lCells[2].innerHTML = diff.left;
        lCells[3].className = 'ln ln-ws';
        lCells[4].className = 'sg sg-ws'; lCells[4].textContent = '~';
        lCells[5].className = 'ws'; lCells[5].innerHTML = diff.right;
    } else {
        lCells[0].className = 'ln ln-del';
        lCells[1].className = 'sg sg-del'; lCells[1].textContent = '−';
        lCells[2].className = 'del'; lCells[2].innerHTML = diff.left;
        lCells[3].className = 'ln ln-add';
        lCells[4].className = 'sg sg-add'; lCells[4].textContent = '+';
        lCells[5].className = 'add'; lCells[5].innerHTML = diff.right;
    }

    leftRow.classList.add('row-aligned');
}
"""


def _merge_whitespace_diffs(opcodes: list, before_lines: list, after_lines: list) -> list:
    """Post-process opcodes to merge adjacent delete+insert into replace.

    When lines differ only by leading/trailing whitespace, SequenceMatcher
    treats them as separate delete and insert operations. This function
    merges such pairs into replace operations so they get inline diff
    highlighting (showing the whitespace difference).

    Matching criteria: lines are equal after stripping whitespace.
    """
    result = []
    i = 0
    while i < len(opcodes):
        tag, i1, i2, j1, j2 = opcodes[i]

        # Check if this is a delete followed by insert (or vice versa)
        if tag == "delete" and i + 1 < len(opcodes) and opcodes[i + 1][0] == "insert":
            # delete + insert pattern
            _, _, _, j1_next, j2_next = opcodes[i + 1]
            del_lines = list(range(i1, i2))
            ins_lines = list(range(j1_next, j2_next))

            # Match lines that are equal after stripping
            matched_del = set()
            matched_ins = set()
            pairs = []

            for di, d_idx in enumerate(del_lines):
                d_stripped = before_lines[d_idx].strip()
                for ii, i_idx in enumerate(ins_lines):
                    if ii in matched_ins:
                        continue
                    if before_lines[d_idx].strip() == after_lines[i_idx].strip():
                        pairs.append((d_idx, i_idx))
                        matched_del.add(di)
                        matched_ins.add(ii)
                        break

            if pairs:
                # Sort pairs by original order
                pairs.sort()
                # Emit unmatched deletes before first pair
                prev_d = i1
                prev_i = j1_next
                for d_idx, i_idx in pairs:
                    # Unmatched deletes before this pair
                    while prev_d < d_idx:
                        result.append(("delete", prev_d, prev_d + 1, prev_i, prev_i))
                        prev_d += 1
                    # Unmatched inserts before this pair
                    while prev_i < i_idx:
                        result.append(("insert", d_idx, d_idx, prev_i, prev_i + 1))
                        prev_i += 1
                    # The replace pair
                    result.append(("replace", d_idx, d_idx + 1, i_idx, i_idx + 1))
                    prev_d = d_idx + 1
                    prev_i = i_idx + 1
                # Remaining unmatched deletes
                for d_idx in range(prev_d, i2):
                    result.append(("delete", d_idx, d_idx + 1, prev_i, prev_i))
                # Remaining unmatched inserts
                for i_idx in range(prev_i, j2_next):
                    result.append(("insert", i2, i2, i_idx, i_idx + 1))

                i += 2  # Skip both delete and insert
                continue

        elif tag == "insert" and i + 1 < len(opcodes) and opcodes[i + 1][0] == "delete":
            # insert + delete pattern (same logic, reversed)
            _, i1_next, i2_next, _, _ = opcodes[i + 1]
            ins_lines = list(range(j1, j2))
            del_lines = list(range(i1_next, i2_next))

            matched_ins = set()
            matched_del = set()
            pairs = []

            for ii, i_idx in enumerate(ins_lines):
                i_stripped = after_lines[i_idx].strip()
                for di, d_idx in enumerate(del_lines):
                    if di in matched_del:
                        continue
                    if after_lines[i_idx].strip() == before_lines[d_idx].strip():
                        pairs.append((d_idx, i_idx))
                        matched_ins.add(ii)
                        matched_del.add(di)
                        break

            if pairs:
                pairs.sort()
                prev_d = i1_next
                prev_i = j1
                for d_idx, i_idx in pairs:
                    while prev_i < i_idx:
                        result.append(("insert", prev_d, prev_d, prev_i, prev_i + 1))
                        prev_i += 1
                    while prev_d < d_idx:
                        result.append(("delete", prev_d, prev_d + 1, prev_i, prev_i))
                        prev_d += 1
                    result.append(("replace", d_idx, d_idx + 1, i_idx, i_idx + 1))
                    prev_d = d_idx + 1
                    prev_i = i_idx + 1
                for i_idx in range(prev_i, j2):
                    result.append(("insert", prev_d, prev_d, i_idx, i_idx + 1))
                for d_idx in range(prev_d, i2_next):
                    result.append(("delete", d_idx, d_idx + 1, j2, j2))

                i += 2
                continue

        # No merge needed, keep original opcode
        result.append((tag, i1, i2, j1, j2))
        i += 1

    return result


def _make_diff_html(before_text: str, after_text: str, context: int = 3) -> str:
    """Generate a GitHub-style side-by-side diff HTML table.

    Uses difflib.SequenceMatcher.get_opcodes() and renders all lines.
    Lines beyond `context` from any change are marked collapsible (hidden
    by default) so the user can expand them interactively via JS.
    """
    before_lines = before_text.splitlines()
    after_lines = after_text.splitlines()

    sm = difflib.SequenceMatcher(None, before_lines, after_lines)
    opcodes = sm.get_opcodes()

    # Post-process: merge adjacent delete+insert into replace when lines
    # differ only by whitespace (so they get inline diff highlighting)
    opcodes = _merge_whitespace_diffs(opcodes, before_lines, after_lines)

    if all(tag == "equal" for tag, *_ in opcodes):
        return '<p class="noop-msg">No differences.</p>'

    # Mark every line as collapsed or visible.
    before_collapse = [True] * len(before_lines)
    after_collapse = [True] * len(after_lines)
    for tag, i1, i2, j1, j2 in opcodes:
        if tag != "equal":
            for i in range(i1, i2):
                before_collapse[i] = False
            for j in range(j1, j2):
                after_collapse[j] = False
    # Keep `context` visible equal-lines on each side of every change.
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for k in range(min(context, i2 - i1)):
                before_collapse[i1 + k] = False
                after_collapse[j1 + k] = False
            for k in range(min(context, i2 - i1)):
                before_collapse[i2 - 1 - k] = False
                after_collapse[j2 - 1 - k] = False

    rows = []

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                collapsed = before_collapse[i]
                hidden_attr = ' class="row-hidden" data-collapse="1"' if collapsed else ""
                ln_l = f'<td class="ln ln-eq" data-side="l" data-idx="{i}">{i + 1}</td>'
                ln_r = f'<td class="ln ln-eq" data-side="r" data-idx="{j}">{j + 1}</td>'
                rows.append(
                    f"<tr{hidden_attr}>{ln_l}"
                    f'<td class="sg"></td><td class="eq">{_esc(before_lines[i])}</td>'
                    f"{ln_r}"
                    f'<td class="sg"></td><td class="eq">{_esc(after_lines[j])}</td></tr>'
                )
        elif tag == "replace":
            # Smart pairing: match lines by stripped content first, then by position
            left_indices = list(range(i1, i2))
            right_indices = list(range(j1, j2))

            # Phase 1: Match lines that are identical after stripping whitespace
            pairs = []
            used_left = set()
            used_right = set()

            for li in left_indices:
                left_stripped = before_lines[li].strip()
                for ri in right_indices:
                    if ri in used_right:
                        continue
                    if before_lines[li].strip() == after_lines[ri].strip():
                        pairs.append((li, ri))
                        used_left.add(li)
                        used_right.add(ri)
                        break

            # Phase 2: For remaining lines, use positional pairing
            remaining_left = [i for i in left_indices if i not in used_left]
            remaining_right = [j for j in right_indices if j not in used_right]

            # Merge pairs with positional pairing of remaining lines
            # Create a unified list of (left_idx, right_idx, is_paired) tuples
            # sorted by the order they should appear
            all_rows = []

            # Add matched pairs
            for li, ri in pairs:
                all_rows.append((li, ri, True))

            # Add positional pairs for remaining
            for k in range(max(len(remaining_left), len(remaining_right))):
                if k < len(remaining_left) and k < len(remaining_right):
                    all_rows.append((remaining_left[k], remaining_right[k], False))
                elif k < len(remaining_left):
                    all_rows.append((remaining_left[k], None, False))
                else:
                    all_rows.append((None, remaining_right[k], False))

            # Sort by the minimum of left/right indices to maintain visual order
            def sort_key(row):
                li, ri, _ = row
                if li is not None and ri is not None:
                    return min(li * 1000, ri * 1000)
                elif li is not None:
                    return li * 1000
                else:
                    return ri * 1000

            all_rows.sort(key=sort_key)

            # Render rows
            for li, ri, is_matched in all_rows:
                if li is not None and ri is not None:
                    # Paired line: use inline diff
                    left_html, right_html, is_ws_only = _inline_diff(before_lines[li], after_lines[ri])
                    if is_ws_only and is_matched:
                        # Whitespace-only change: use subtle styling
                        ln_l = f'<td class="ln ln-ws" data-side="l" data-idx="{li}">{li + 1}</td>'
                        ln_r = f'<td class="ln ln-ws" data-side="r" data-idx="{ri}">{ri + 1}</td>'
                        rows.append(
                            f"<tr>{ln_l}"
                            f'<td class="sg sg-ws">~</td><td class="ws">{left_html}</td>'
                            f"{ln_r}"
                            f'<td class="sg sg-ws">~</td><td class="ws">{right_html}</td></tr>'
                        )
                    else:
                        # Content change: use red/green styling
                        ln_l = f'<td class="ln ln-del" data-side="l" data-idx="{li}">{li + 1}</td>'
                        ln_r = f'<td class="ln ln-add" data-side="r" data-idx="{ri}">{ri + 1}</td>'
                        rows.append(
                            f"<tr>{ln_l}"
                            f'<td class="sg sg-del">−</td><td class="del">{left_html}</td>'
                            f"{ln_r}"
                            f'<td class="sg sg-add">+</td><td class="add">{right_html}</td></tr>'
                        )
                elif li is not None:
                    # Unpaired left line (deletion only)
                    ln_l = f'<td class="ln ln-del" data-side="l" data-idx="{li}">{li + 1}</td>'
                    rows.append(
                        f"<tr>{ln_l}"
                        f'<td class="sg sg-del">−</td><td class="del">{_esc(before_lines[li])}</td>'
                        f'<td class="ln"></td><td class="sg"></td><td></td></tr>'
                    )
                else:
                    # Unpaired right line (insertion only)
                    ln_r = f'<td class="ln ln-add" data-side="r" data-idx="{ri}">{ri + 1}</td>'
                    rows.append(
                        f'<tr><td class="ln"></td><td class="sg"></td><td></td>'
                        f"{ln_r}"
                        f'<td class="sg sg-add">+</td><td class="add">{_esc(after_lines[ri])}</td></tr>'
                    )
        elif tag == "delete":
            for i in range(i1, i2):
                ln_l = f'<td class="ln ln-del" data-side="l" data-idx="{i}">{i + 1}</td>'
                rows.append(
                    f"<tr>{ln_l}"
                    f'<td class="sg sg-del">−</td><td class="del">{_esc(before_lines[i])}</td>'
                    f'<td class="ln"></td><td class="sg"></td><td></td></tr>'
                )
        elif tag == "insert":
            for j in range(j1, j2):
                ln_r = f'<td class="ln ln-add" data-side="r" data-idx="{j}">{j + 1}</td>'
                rows.append(
                    f'<tr><td class="ln"></td><td class="sg"></td><td></td>'
                    f"{ln_r}"
                    f'<td class="sg sg-add">+</td><td class="add">{_esc(after_lines[j])}</td></tr>'
                )

    # --- Identify contiguous runs of collapsed (hidden) rows ---
    collapse_flags = []
    for row_html in rows:
        collapse_flags.append('data-collapse="1"' in row_html and 'class="row-hidden"' in row_html)

    run_list = []  # list of (start_idx, end_idx) — end is exclusive
    in_run = False
    for idx, is_collapsed in enumerate(collapse_flags):
        if is_collapsed and not in_run:
            run_start = idx
            in_run = True
        elif not is_collapsed and in_run:
            run_list.append((run_start, idx))
            in_run = False
    if in_run:
        run_list.append((run_start, len(rows)))

    # Tag hidden rows with their run index (used by JS to scope expansion)
    for run_idx, (r1, r2) in enumerate(run_list):
        for i in range(r1, r2):
            rows[i] = rows[i].replace('class="row-hidden"', f'class="row-hidden" data-run="{run_idx}"')

    # --- Insert expand button rows ---
    # One "↑↓ Expand" button after each run, expanding hidden rows in both directions.
    # Insertions are applied from bottom to top so indices stay valid.
    insertions = []
    for ri, (r1, r2) in enumerate(run_list):
        insertions.append((r2, (
            f'<tr class="btn-row">'
            f'<td colspan="6" data-run="{ri}" '
            f'onclick="expand(this,20,event)">'
            f'<span class="exp-arrow">↑↓</span>'
            f'<span class="exp-label">Expand</span>'
            f'</td></tr>'
        )))

    for pos, html in sorted(insertions, key=lambda x: x[0], reverse=True):
        rows.insert(pos, html)

    return (
        '<div class="diff-table-wrap">'
        '<table><colgroup>'
        '<col style="width:50px"><col style="width:20px"><col>'
        '<col style="width:50px"><col style="width:20px"><col>'
        '</colgroup>'
        + "\n".join(rows)
        + "</table></div>"
    )


def generate_html(records: List[PassRecord], output_path: str):
    """Generate a self-contained HTML file with pass trace visualization.

    Args:
        records:     List of PassRecord from all passes
        output_path: Path to write the HTML file
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Group records by phase
    phases: dict = {}
    for rec in records:
        phases.setdefault(rec.phase, []).append(rec)

    # Build phase tabs + sidebars + summary bars
    phase_tabs_html = []
    sidebars_html = []
    summaries_html = []
    sections_html = []

    for pi, (phase_name, phase_records) in enumerate(phases.items()):
        is_active = pi == 0
        active_cls = " active" if is_active else ""
        active_style = "" if is_active else " style=\"display:none\""

        pretty_phase = phase_name.replace("_", " ").replace("phase1", "Phase 1:").replace("phase2", "Phase 2:")

        n_total = len(phase_records)
        n_changed = sum(1 for r in phase_records if r.changed)
        n_noop = n_total - n_changed

        # Tab
        phase_tabs_html.append(
            f'<div class="phase-tab{active_cls}" '
            f'onclick="showPhase(this, \'{phase_name}\')">{pretty_phase}</div>'
        )

        # Summary bar
        summaries_html.append(
            f'<div class="summary-bar" id="sm-{phase_name}"{active_style}>'
            f'<span class="badge badge-total">{n_total} passes</span>'
            f'<span class="badge badge-changed">{n_changed} changed</span>'
            f'<span class="badge badge-noop">{n_noop} no-op</span>'
            f'</div>'
        )

        # Sidebar
        links = []
        for rec in phase_records:
            dot_cls = "changed" if rec.changed else "noop"
            sid = f"sec-{rec.phase}-{rec.index}"
            stats_html = ""
            if rec.changed:
                stats_html = (
                    f'<span class="pass-stats">'
                    f'<span class="st-add">+{rec.add_lines}</span> '
                    f'<span class="st-del">−{rec.del_lines}</span>'
                    f'</span>'
                )
            links.append(
                f'<a class="pass-link" data-phase="{rec.phase}" data-target="{sid}" '
                f'onclick="showPass(this, \'{sid}\')">'
                f'<span class="pass-idx">{rec.index:02d}</span>'
                f'<span class="pass-dot {dot_cls}"></span>'
                f'<span class="pass-label">{rec.name}</span>'
                f'{stats_html}'
                f'</a>'
            )
        sidebars_html.append(
            f'<div class="sidebar" id="sb-{phase_name}"{active_style}>'
            f'<div class="section-title">Passes</div>'
            + "\n".join(links)
            + "</div>"
        )

        # Main sections
        for rec in phase_records:
            sid = f"sec-{rec.phase}-{rec.index}"

            if rec.changed:
                # Generate GitHub-style side-by-side diff table with
                # collapsible context lines + expand toolbar
                diff_html = _make_diff_html(rec.before_text, rec.after_text, context=3)
                diff_content = (
                    f'<div class="diff-toolbar">'
                    f'<button class="btn-expand-all" '
                    f'onclick="expandAll(this)">⊞ Show all context</button>'
                    f'<button onclick="collapseCtx(this)">⊟ Collapse</button>'
                    f'<span class="copy-spacer"></span>'
                    f'<button class="btn-copy" onclick="copyIr(this,\'before\')">📋 Copy Before</button>'
                    f'<button class="btn-copy" onclick="copyIr(this,\'after\')">📋 Copy After</button>'
                    f'</div>'
                    f'{diff_html}'
                )
                status_html = '<span class="status status-changed">CHANGED</span>'
                _safe_before = rec.before_text.replace("</script>", r"<\/script>")
                _safe_after = rec.after_text.replace("</script>", r"<\/script>")
                sections_html.append(
                    f'<div class="pass-section" id="{sid}">'
                    f'<div class="pass-header collapsible" onclick="toggleCollapse(this)">'
                    f'<span class="pass-toggle"></span>'
                    f'<h2>{rec.index:02d}. {rec.name}</h2>'
                    f'{status_html}'
                    f'</div>'
                    f'{diff_content}'
                    f'<script type="text/plain" class="ir-data-before">{_safe_before}</script>'
                    f'<script type="text/plain" class="ir-data-after">{_safe_after}</script>'
                    f'</div>'
                )
            else:
                diff_content = (
                    '<p class="noop-msg">This pass did not modify the IR.</p>'
                    '<button class="ir-toggle" onclick="toggleIr(this)">&#9654; Show full IR</button>'
                    '<button class="btn-copy" onclick="copyIr(this,\'after\')">📋 Copy IR</button>'
                    f'<div class="ir-block"><pre>{_esc(rec.after_text)}</pre></div>'
                )
                status_html = '<span class="status status-noop">NO-OP</span>'
                _safe_after = rec.after_text.replace("</script>", r"<\/script>")
                sections_html.append(
                    f'<div class="pass-section" id="{sid}">'
                    f'<div class="pass-header">'
                    f'<h2>{rec.index:02d}. {rec.name}</h2>'
                    f'{status_html}'
                    f'</div>'
                    f'{diff_content}'
                    f'<script type="text/plain" class="ir-data-after">{_safe_after}</script>'
                    f'</div>'
                )

    # Auto-show first pass + keyboard shortcuts
    auto_show_js = ""
    if records:
        first_sid = f"sec-{records[0].phase}-{records[0].index}"
        auto_show_js = (
            "document.addEventListener('DOMContentLoaded', function() {\n"
            f"  var first_sid = '{first_sid}';\n"
            "  var el = document.querySelector('[data-target=\"' + first_sid + '\"]');\n"
            "  if (el) showPass(el, first_sid);\n"
            "  _alignStatus = document.getElementById('align-status');\n"
            "\n"
            "  // --- P1: j/k keyboard navigation + Shift+E global expand ---\n"
            "  var passLinks = document.getElementsByClassName('pass-link');\n"
            "  var activeLink = el || null;\n"
            "\n"
            "  document.addEventListener('keydown', function(e) {\n"
            "    var tag = (e.target.tagName || '').toLowerCase();\n"
            "    if (tag === 'input' || tag === 'textarea' || tag === 'select') return;\n"
            "\n"
            "    // --- P4: F7 alignment flow ---\n"
            "    if (e.key === 'F7') {\n"
            "      e.preventDefault();\n"
            "      if (_alignMode === null) {\n"
            "        _alignMode = 'left';\n"
            "        showAlignStatus('left', 'Align: Click a <b>left</b> (before) line number &nbsp; <kbd>F7</kbd> to confirm &nbsp; <kbd>Esc</kbd> to cancel');\n"
            "      } else if (_alignMode === 'left' && _pendingLeft) {\n"
            "        _alignMode = 'right';\n"
            "        _pendingLeft.classList.remove('align-pending');\n"
            "        _pendingLeft.classList.add('align-locked');\n"
            "        var sib = _pendingLeft.nextElementSibling;\n"
            "        while (sib && sib.getAttribute('data-side') !== 'r') {\n"
            "          if (sib.getAttribute('data-side') === 'l') sib.classList.add('align-locked-sibling');\n"
            "          sib = sib.nextElementSibling;\n"
            "        }\n"
            "        var wrap = _pendingLeft.closest('.diff-table-wrap');\n"
            "        if (wrap) wrap.classList.add('waiting-right');\n"
            "        showAlignStatus('right', 'Align: Click a <b>right</b> (after) line number to align &nbsp; <kbd>Esc</kbd> to cancel');\n"
            "      } else if (_alignMode === 'right') {\n"
            "        cancelAlign();\n"
            "      }\n"
            "      return;\n"
            "    }\n"
            "\n"
            "    if (e.key === 'Escape' && _alignMode) {\n"
            "      e.preventDefault();\n"
            "      cancelAlign();\n"
            "      return;\n"
            "    }\n"
            "\n"
            "    if (e.key === 'j' || e.key === 'k') {\n"
            "      e.preventDefault();\n"
            "      var idx = -1;\n"
            "      for (var i = 0; i < passLinks.length; i++) {\n"
            "        if (passLinks[i] === activeLink) { idx = i; break; }\n"
            "      }\n"
            "      if (e.key === 'j' && idx < passLinks.length - 1) idx++;\n"
            "      else if (e.key === 'k' && idx > 0) idx--;\n"
            "      else return;\n"
            "      var next = passLinks[idx];\n"
            "      var phase = next.getAttribute('data-phase');\n"
            "      var curTab = document.querySelector('.phase-tab.active');\n"
            "      if (!curTab || curTab.textContent.indexOf(phase.replace('phase1','Phase 1').replace('phase2','Phase 2')) < 0) {\n"
            "        var tabs = document.querySelectorAll('.phase-tab');\n"
            "        for (var t = 0; t < tabs.length; t++) {\n"
            "          if (tabs[t].getAttribute('onclick').indexOf(phase) > -1) {\n"
            "            showPhase(tabs[t], phase); break;\n"
            "          }\n"
            "        }\n"
            "      }\n"
            "      activeLink = next;\n"
            "      var sid = next.getAttribute('data-target');\n"
            "      showPass(next, sid);\n"
            "      next.scrollIntoView({block: 'nearest'});\n"
            "      var sec = document.getElementById(sid);\n"
            "      if (sec) sec.scrollIntoView({block: 'start'});\n"
            "    }\n"
            "\n"
            "    // Shift+E: expand all hidden context in all passes\n"
            "    if (e.key === 'E' && e.shiftKey && !e.ctrlKey && !e.metaKey) {\n"
            "      e.preventDefault();\n"
            "      document.querySelectorAll('.pass-section table tr.row-hidden').forEach(function(r) {\n"
            "        r.classList.remove('row-hidden');\n"
            "      });\n"
            "      document.querySelectorAll('.pass-section table').forEach(function(tbl) { updBtns(tbl); });\n"
            "    }\n"
            "  });\n"
            "\n"
            "  // --- P4: Alignment click handler (capture phase, before P2 highlight) ---\n"
            "  document.addEventListener('click', function(e) {\n"
            "    if (!_alignMode) return;\n"
            "    var td = e.target.closest('td[data-side]');\n"
            "    if (!td) return;\n"
            "    var section = td.closest('.pass-section.active');\n"
            "    if (!section) return;\n"
            "\n"
            "    if (_alignMode === 'left' && td.getAttribute('data-side') === 'l' && td.textContent.trim() !== '') {\n"
            "      e.stopPropagation();\n"
            "      if (_pendingLeft) {\n"
            "        var oldRow = _pendingLeft.closest('tr');\n"
            "        if (oldRow) oldRow.querySelectorAll('.align-pending,.align-pending-sibling')\n"
            "          .forEach(function(c){ c.classList.remove('align-pending','align-pending-sibling'); });\n"
            "      }\n"
            "      _pendingLeft = td;\n"
            "      td.classList.add('align-pending');\n"
            "      var sib = td.nextElementSibling;\n"
            "      while (sib && sib.getAttribute('data-side') !== 'r') {\n"
            "        if (sib.getAttribute('data-side') === 'l') sib.classList.add('align-pending-sibling');\n"
            "        sib = sib.nextElementSibling;\n"
            "      }\n"
            "      showAlignStatus('left', 'Left line <b>' + td.textContent.trim() + '</b> selected &nbsp; Press <kbd>F7</kbd> to confirm &nbsp; <kbd>Esc</kbd> to cancel');\n"
            "      return;\n"
            "    }\n"
            "\n"
            "    if (_alignMode === 'right' && td.getAttribute('data-side') === 'r' && td.textContent.trim() !== '' && _pendingLeft) {\n"
            "      e.stopPropagation();\n"
            "      alignRows(_pendingLeft, td);\n"
            "      cancelAlign();\n"
            "      return;\n"
            "    }\n"
            "  }, true);\n"
            "\n"
            "  // --- P2: Click line number to highlight row ---\n"
            "  document.addEventListener('click', function(e) {\n"
            "    var td = e.target.closest('td[data-side]');\n"
            "    if (!td) return;\n"
            "    var tr = td.closest('tr');\n"
            "    if (!tr || tr.closest('.pass-section.active') === null) return;\n"
            "    var wasHl = tr.classList.contains('row-hl');\n"
            "    document.querySelectorAll('tr.row-hl').forEach(function(r) { r.classList.remove('row-hl'); });\n"
            "    if (!wasHl) tr.classList.add('row-hl');\n"
            "  });\n"
            "});\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TileLang IR Pass Trace</title>
<style>{_CSS}</style>
</head>
<body>
<div class="align-status" id="align-status"></div>
<div class="header">
  <h1>TileLang IR Pass Trace</h1>
  <div class="sub">Compilation pipeline visualization &middot; {len(records)} passes recorded</div>
</div>
<div class="phase-tabs">
  {"".join(phase_tabs_html)}
</div>
{"".join(summaries_html)}
<div class="main">
  {"".join(sidebars_html)}
  <div class="content">
    {"".join(sections_html)}
  </div>
</div>
<script>
{_JS}
{auto_show_js}
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def _esc(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _inline_diff(line_before: str, line_after: str) -> tuple[str, str, bool]:
    """Compute character-level inline diff between two lines.

    Returns:
        - left_html: HTML string for the before line with highlights
        - right_html: HTML string for the after line with highlights
        - is_ws_only: True if the only differences are whitespace

    Used in `replace` blocks to show exactly which characters changed,
    similar to GitHub's inline word diff.
    """
    sm = difflib.SequenceMatcher(None, line_before, line_after)
    left_parts = []
    right_parts = []
    is_ws_only = True

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            eq = _esc(line_before[i1:i2])
            left_parts.append(eq)
            right_parts.append(eq)
        else:
            # Check if this non-equal part is only whitespace
            left_chunk = line_before[i1:i2] if i2 > i1 else ""
            right_chunk = line_after[j1:j2] if j2 > j1 else ""
            if left_chunk.strip() != "" or right_chunk.strip() != "":
                is_ws_only = False

            if tag == "replace":
                left_parts.append(f'<span class="del-word">{_esc(left_chunk)}</span>')
                right_parts.append(f'<span class="add-word">{_esc(right_chunk)}</span>')
            elif tag == "delete":
                left_parts.append(f'<span class="del-word">{_esc(left_chunk)}</span>')
            elif tag == "insert":
                right_parts.append(f'<span class="add-word">{_esc(right_chunk)}</span>')

    return "".join(left_parts), "".join(right_parts), is_ws_only


# ---------------------------------------------------------------------------
# Monkey-patch entry point
# ---------------------------------------------------------------------------
def patch():
    """Activate IR pass tracing via monkey-patching.

    Replaces LowerAndLegalize and OptimizeForTarget in the
    tilelang.engine.lower module namespace.  Because the lower() function
    resolves these names through its __globals__ (the module dict), it
    will transparently use the debug versions.

    Call this ONCE at the top of your kernel script:
        import tilelang.engine.pass_trace; tilelang.engine.pass_trace.patch()
    """
    import sys

    # tilelang.engine.__init__.py does `from .lower import lower` which
    # shadows the module with the function.  Use sys.modules to get the
    # actual module object.
    _lower_mod = sys.modules["tilelang.engine.lower"]

    _lower_mod.LowerAndLegalize = debug_LowerAndLegalize
    _lower_mod.OptimizeForTarget = debug_OptimizeForTarget
    print("[pass_trace] IR pass tracing patched. Set TILELANG_DUMP_PASSES=1 to enable.")


def reset():
    """Clear collected records and cached dump dir (useful between multiple compilations)."""
    global _records, _dump_dir
    _records = []
    _dump_dir = None
