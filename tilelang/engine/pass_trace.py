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
    diff_line_count: int = 0
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

        diff_lines = []
        add_count = 0
        del_count = 0
        if changed:
            diff_lines = list(difflib.unified_diff(
                before_text.splitlines(keepends=True),
                after_text.splitlines(keepends=True),
                fromfile=f"before {pass_name}",
                tofile=f"after {pass_name}",
            ))
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
            diff_line_count=len(diff_lines),
            add_lines=add_count,
            del_lines=del_count,
        )
        _records.append(record)

        # Also dump raw .tir files (useful for BeyondCompare)
        _save_raw_files(record)

        # Console progress
        tag = "CHANGED" if changed else "NO-OP"
        print(f"  [pass_trace] {phase_name}/{pass_index:02d}_{pass_name}: {tag}")
    else:
        # No dumping, just run the pass silently
        pass

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

/* Hidden (collapsible) rows */
.diff-table-wrap tr.row-hidden { display: none; }

/* Row highlight (click line number to highlight) */
.diff-table-wrap tr.row-hl td { background: #fff8c5 !important; }
.diff-table-wrap td.ln:not(:empty) { cursor: pointer; }
.diff-table-wrap td.ln:not(:empty):hover { text-decoration: underline; }

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
function showPass(el, id) {
    document.querySelectorAll('.pass-section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.pass-link').forEach(l => l.classList.remove('active'));
    var sec = document.getElementById(id);
    if (sec) sec.classList.add('active');
    if (el) el.classList.add('active');
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
    navigator.clipboard.writeText(el.textContent).then(function() {
        var toast = document.createElement('div');
        toast.className = 'copy-toast';
        toast.textContent = 'Copied ' + (side === 'before' ? 'Before' : side === 'after' ? 'After' : 'IR') + ' IR';
        document.body.appendChild(toast);
        setTimeout(function() { toast.remove(); }, 1600);
    });
}

function expand(el, n) {
    var run = el.dataset.run;
    var tr = el.closest('tr');
    var table = tr.closest('table');
    var limit = (event && event.altKey) ? 999999 : n;
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
"""


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
                ln_l = f'<td class="ln ln-eq" data-line="l">{i + 1}</td>'
                ln_r = f'<td class="ln ln-eq" data-line="r">{j + 1}</td>'
                rows.append(
                    f"<tr{hidden_attr}>{ln_l}"
                    f'<td class="sg"></td><td class="eq">{_esc(before_lines[i])}</td>'
                    f"{ln_r}"
                    f'<td class="sg"></td><td class="eq">{_esc(after_lines[j])}</td></tr>'
                )
        elif tag == "replace":
            left = list(range(i1, i2))
            right = list(range(j1, j2))
            for k in range(max(len(left), len(right))):
                if k < len(left) and k < len(right):
                    i, j = left[k], right[k]
                    ln_l = f'<td class="ln ln-del" data-line="l">{i + 1}</td>'
                    ln_r = f'<td class="ln ln-add" data-line="r">{j + 1}</td>'
                    rows.append(
                        f"<tr>{ln_l}"
                        f'<td class="sg sg-del">−</td><td class="del">{_esc(before_lines[i])}</td>'
                        f"{ln_r}"
                        f'<td class="sg sg-add">+</td><td class="add">{_esc(after_lines[j])}</td></tr>'
                    )
                elif k < len(left):
                    i = left[k]
                    ln_l = f'<td class="ln ln-del" data-line="l">{i + 1}</td>'
                    rows.append(
                        f"<tr>{ln_l}"
                        f'<td class="sg sg-del">−</td><td class="del">{_esc(before_lines[i])}</td>'
                        f'<td class="ln"></td><td class="sg"></td><td></td></tr>'
                    )
                else:
                    j = right[k]
                    ln_r = f'<td class="ln ln-add" data-line="r">{j + 1}</td>'
                    rows.append(
                        f'<tr><td class="ln"></td><td class="sg"></td><td></td>'
                        f"{ln_r}"
                        f'<td class="sg sg-add">+</td><td class="add">{_esc(after_lines[j])}</td></tr>'
                    )
        elif tag == "delete":
            for i in range(i1, i2):
                ln_l = f'<td class="ln ln-del" data-line="l">{i + 1}</td>'
                rows.append(
                    f"<tr>{ln_l}"
                    f'<td class="sg sg-del">−</td><td class="del">{_esc(before_lines[i])}</td>'
                    f'<td class="ln"></td><td class="sg"></td><td></td></tr>'
                )
        elif tag == "insert":
            for j in range(j1, j2):
                ln_r = f'<td class="ln ln-add" data-line="r">{j + 1}</td>'
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
            f'onclick="expand(this,20)">'
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
                sections_html.append(
                    f'<div class="pass-section" id="{sid}">'
                    f'<div class="pass-header collapsible" onclick="toggleCollapse(this)">'
                    f'<span class="pass-toggle"></span>'
                    f'<h2>{rec.index:02d}. {rec.name}</h2>'
                    f'{status_html}'
                    f'</div>'
                    f'{diff_content}'
                    f'<script type="text/plain" class="ir-data-before">{rec.before_text}</script>'
                    f'<script type="text/plain" class="ir-data-after">{rec.after_text}</script>'
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
                sections_html.append(
                    f'<div class="pass-section" id="{sid}">'
                    f'<div class="pass-header">'
                    f'<h2>{rec.index:02d}. {rec.name}</h2>'
                    f'{status_html}'
                    f'</div>'
                    f'{diff_content}'
                    f'<script type="text/plain" class="ir-data-after">{rec.after_text}</script>'
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
            "\n"
            "  // --- P1: j/k keyboard navigation + Shift+E global expand ---\n"
            "  var passLinks = document.getElementsByClassName('pass-link');\n"
            "  var activeLink = el || null;\n"
            "\n"
            "  document.addEventListener('keydown', function(e) {\n"
            "    var tag = (e.target.tagName || '').toLowerCase();\n"
            "    if (tag === 'input' || tag === 'textarea' || tag === 'select') return;\n"
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
            "  // --- P2: Click line number to highlight row ---\n"
            "  document.addEventListener('click', function(e) {\n"
            "    var td = e.target.closest('td[data-line]');\n"
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
<div class="header">
  <h1>TileLang IR Pass Trace</h1>
  <div class="sub">Compilation pipeline visualization &middet; {len(records)} passes recorded</div>
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
