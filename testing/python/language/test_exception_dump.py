"""Tests for AI Core exception dump functionality.

Design doc: dump.md — when AI Core hits a hardware exception, the
callback searches kernel args for MAGIC, locates ParamSizeInfo, and
saves input tensor data to a log file (or via acldumpSaveExceptionInfo
when available).  The dump is silent — no stdout/stderr output.
"""

import ctypes
import glob
import json
import os
import re
import subprocess
import sys

import pytest
import torch

import tilelang
import tilelang.language as T

PASS_AUTO = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.disable_cache()
    yield


def _make_kernel():
    """3 tensor params (x, y, z) — matches design doc spec."""

    @T.prim_func
    def main(
        x: T.Tensor((128, 128), "float16"),
        y: T.Tensor((128, 128), "float16"),
        z: T.Tensor((128, 128), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            x_ub = T.alloc_ub((128, 128), "float16")
            y_ub = T.alloc_ub((128, 128), "float16")
            z_ub = T.alloc_ub((128, 128), "float16")
            T.copy(x[:, :], x_ub)
            T.copy(y[:, :], y_ub)
            for i, j in T.Parallel(128, 128):
                z_ub[i, j] = x_ub[i, j] + y_ub[i, j]
            T.copy(z_ub[:, :], z[:, :])

    return main


def _parse_tensor_data(output, idx):
    """Extract hex bytes from 'tensor[idx] data (first N bytes): XX XX ...'"""
    pattern = rf"tensor\[{idx}\] data \(first \d+ bytes\): ([0-9a-f ]+)"
    m = re.search(pattern, output)
    if m is None:
        return None
    return [int(b, 16) for b in m.group(1).strip().split()]


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_exception_dump_npu_correctness(target):
    """Kernel with exception dump still produces correct results on NPU."""
    prim_func = _make_kernel()
    kernel = tilelang.compile(
        prim_func,
        out_idx=[2],
        pass_configs=PASS_AUTO,
        target=target,
    )
    x = torch.randn(128, 128, dtype=torch.float16).npu()
    y = torch.randn(128, 128, dtype=torch.float16).npu()
    z = kernel(x, y)
    z_ref = (x + y).cpu()
    torch.testing.assert_close(z.cpu(), z_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_exception_dump_callback_magic_not_found(target):
    """Verify callback returns -2 when MAGIC is not found in args (silently)."""
    prim_func = _make_kernel()
    kernel = tilelang.compile(
        prim_func,
        out_idx=[2],
        pass_configs=PASS_AUTO,
        target=target,
    )

    so_path = kernel.adapter.libpath
    lib = ctypes.CDLL(so_path)
    func = lib.tilelang_dump_from_host_args
    func.restype = ctypes.c_int
    func.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

    junk = b"\x00" * 64
    buf_c = (ctypes.c_ubyte * len(junk)).from_buffer_copy(junk)
    ret = func(ctypes.cast(buf_c, ctypes.c_void_p), ctypes.c_uint32(len(junk)))

    assert ret == -2, f"Expected -2 for MAGIC not found, got {ret}"


_HW_EXCEPTION_SCRIPT = r"""
import ctypes, sys, json, torch, os, glob

dump_dir = sys.argv[2]
os.environ["TILELANG_EXCEPTION_DUMP_DIR"] = dump_dir
os.makedirs(dump_dir, exist_ok=True)

# Clear any old dump files
for f in glob.glob(os.path.join(dump_dir, "tilelang_exception_dump_*.log")):
    os.remove(f)

so_path = sys.argv[1]
lib = ctypes.CDLL(so_path)
call_fn = lib.call
call_fn.restype = None
call_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p]

# Step 1: Normal execution — registers exception callback inside call()
x = torch.randn(128, 128, dtype=torch.float16).npu()
y = torch.randn(128, 128, dtype=torch.float16).npu()
stream = torch.npu.current_stream().npu_stream
call_fn(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr()),
        ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(stream))
torch.npu.synchronize()

# Step 2: Prepare tensors with known data for exception-triggering launch
x_exc = torch.arange(128 * 128, dtype=torch.float16).reshape(128, 128).npu()
y_exc = torch.full((128, 128), 3.14, dtype=torch.float16).npu()

x_hex = list(x_exc.cpu().contiguous().numpy().tobytes()[:128])
y_hex = list(y_exc.cpu().contiguous().numpy().tobytes()[:128])

result = {
    "x_addr": x_exc.data_ptr(),
    "y_addr": y_exc.data_ptr(),
    "expected_size": 128 * 128 * 2,
    "x_hex": x_hex,
    "y_hex": y_hex,
}
print("RESULT_JSON=" + json.dumps(result), flush=True)

# Step 3: Launch with valid x, y but null z -> write to null triggers exception
exc_stream = torch.npu.Stream()
call_fn(ctypes.c_void_p(x_exc.data_ptr()),
        ctypes.c_void_p(y_exc.data_ptr()),
        ctypes.c_void_p(0),
        ctypes.c_void_p(exc_stream.npu_stream))
try:
    exc_stream.synchronize()
except Exception:
    pass

# Step 4: Wait a moment for callback to finish writing, then read dump file
import time
time.sleep(1)
dump_files = glob.glob(os.path.join(dump_dir, "tilelang_exception_dump_*.log"))
if dump_files:
    with open(sorted(dump_files)[-1], "r") as f:
        dump_content = f.read()
    print("DUMP_BEGIN", flush=True)
    print(dump_content, flush=True)
    print("DUMP_END", flush=True)
else:
    print("DUMP_BEGIN", flush=True)
    print("DUMP_END", flush=True)
"""


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_exception_dump_callback_via_hw_exception(target, tmp_path):
    """Trigger a real NPU hardware exception and verify dump file is generated.

    The callback writes tensor info silently to a log file.  No stdout/stderr
    output is expected from the callback itself.
    """
    prim_func = _make_kernel()
    kernel = tilelang.compile(
        prim_func,
        out_idx=[2],
        pass_configs=PASS_AUTO,
        target=target,
    )

    so_path = kernel.adapter.libpath
    dump_dir = str(tmp_path)

    proc = subprocess.run(
        [sys.executable, "-c", _HW_EXCEPTION_SCRIPT, so_path, dump_dir],
        capture_output=True,
        text=True,
        timeout=120,
    )

    stdout = proc.stdout
    stderr = proc.stderr

    assert "RESULT_JSON=" in stdout, (
        f"Subprocess did not produce RESULT_JSON.\n"
        f"exit code: {proc.returncode}\n"
        f"stdout:\n{stdout}\nstderr:\n{stderr}"
    )

    # Parse the RESULT_JSON line and DUMP_BEGIN..DUMP_END block from subprocess
    json_line = None
    dump_content = ""
    in_dump = False
    for line in stdout.splitlines():
        if line.startswith("RESULT_JSON="):
            json_line = json.loads(line[len("RESULT_JSON="):])
        elif line.strip() == "DUMP_BEGIN":
            in_dump = True
        elif line.strip() == "DUMP_END":
            in_dump = False
        elif in_dump:
            dump_content += line + "\n"

    assert json_line is not None, (
        f"RESULT_JSON not found in subprocess output:\n{stdout}\n{stderr}"
    )

    x_addr = json_line["x_addr"]
    y_addr = json_line["y_addr"]
    expected_size = json_line["expected_size"]
    x_expected_hex = json_line["x_hex"]
    y_expected_hex = json_line["y_hex"]

    # Verify dump file was generated with correct content
    assert "TileLang Exception Dump" in dump_content, (
        f"Dump file not generated or missing header:\n{dump_content}"
    )
    assert "Tensor count: 3" in dump_content, (
        f"Expected tensor count 3:\n{dump_content}"
    )

    assert f"tensor[0]: addr=0x{x_addr:x}" in dump_content, (
        f"x address 0x{x_addr:x} not in dump:\n{dump_content}"
    )
    assert f"size={expected_size} bytes" in dump_content, (
        f"Expected size {expected_size} not found:\n{dump_content}"
    )
    assert "dataType=1" in dump_content, (
        f"Expected float16 dataType=1:\n{dump_content}"
    )

    assert f"tensor[1]: addr=0x{y_addr:x}" in dump_content, (
        f"y address 0x{y_addr:x} not in dump:\n{dump_content}"
    )

    assert "tensor[2]: addr=0x0" in dump_content, (
        f"Expected null z address (0x0):\n{dump_content}"
    )

    x_dumped = _parse_tensor_data(dump_content, 0)
    assert x_dumped is not None, (
        f"tensor[0] data hex dump not found:\n{dump_content}"
    )
    assert x_dumped == x_expected_hex, (
        f"tensor[0] data mismatch:\n"
        f"  expected: {x_expected_hex[:16]}...\n"
        f"  dumped:   {x_dumped[:16]}..."
    )

    y_dumped = _parse_tensor_data(dump_content, 1)
    assert y_dumped is not None, (
        f"tensor[1] data hex dump not found:\n{dump_content}"
    )
    assert y_dumped == y_expected_hex, (
        f"tensor[1] data mismatch:\n"
        f"  expected: {y_expected_hex[:16]}...\n"
        f"  dumped:   {y_dumped[:16]}..."
    )

    assert "tensor[2] data" not in dump_content, (
        f"tensor[2] should have no data dump (null addr):\n{dump_content}"
    )
