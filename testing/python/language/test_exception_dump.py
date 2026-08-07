"""Tests for AI Core exception dump functionality.

When AI Core hits a hardware exception, the callback searches kernel args
for MAGIC, locates ParamSizeInfo (including kernel name), and saves input
tensor data via CANN's acldumpSaveExceptionInfo.  The dump file is then
parsed by msaicerr.py into per-tensor .bin files.
"""

import ctypes
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

import tilelang
import tilelang.language as T
from tilelang.tools.ascend_exception_dump_bin import parse_exception_dump

PASS_AUTO = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_EXCEPTION_DUMP: True,
}

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()

_MSAICERR_PATH = os.path.join(os.environ.get("ASCEND_HOME_PATH", ""), "tools", "msaicerr", "msaicerr.py")


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


@pytest.mark.ci_skip
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


@pytest.mark.ci_skip
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

dump_path = os.environ["ASCEND_DUMP_PATH"]

# Clear any old dump files
for f in glob.glob(os.path.join(dump_path, "extra-info", "data-dump", "*")):
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

x_np = x_exc.cpu().numpy()
y_np = y_exc.cpu().numpy()

result = {
    "x_bytes": x_np.tobytes().hex(),
    "y_bytes": y_np.tobytes().hex(),
    "shape": [128, 128],
    "dtype": "float16",
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

# Step 4: Wait for callback to finish writing
import time
time.sleep(2)
"""


@pytest.mark.ci_skip
@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
@pytest.mark.skipif(not os.path.isfile(_MSAICERR_PATH), reason=f"msaicerr.py not found at {_MSAICERR_PATH}")
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_exception_dump_callback_via_hw_exception(target, tmp_path):
    """Trigger a real NPU hardware exception and verify dump file is generated.

    The callback calls CANN's acldumpSaveExceptionInfo with the kernel name,
    producing a dump file under <ASCEND_DUMP_PATH>/extra-info/data-dump/<dev>/.
    msaicerr.py then parses it into per-tensor .bin files, which we compare
    against the known input data.
    """
    prim_func = _make_kernel()
    kernel = tilelang.compile(
        prim_func,
        out_idx=[2],
        pass_configs=PASS_AUTO,
        target=target,
    )

    so_path = kernel.adapter.libpath
    dump_path = str(tmp_path)

    env = os.environ.copy()
    env["ASCEND_DUMP_PATH"] = dump_path
    env["ASCEND_DUMP_SCENE"] = "aic_err_brief_dump"

    proc = subprocess.run(
        [sys.executable, "-c", _HW_EXCEPTION_SCRIPT, so_path],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )

    stdout = proc.stdout
    stderr = proc.stderr

    assert "RESULT_JSON=" in stdout, (
        f"Subprocess did not produce RESULT_JSON.\nexit code: {proc.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
    )

    json_line = None
    for line in stdout.splitlines():
        if line.startswith("RESULT_JSON="):
            json_line = json.loads(line[len("RESULT_JSON=") :])
            break

    assert json_line is not None, f"RESULT_JSON not found in subprocess output:\n{stdout}\n{stderr}"

    x_expected = np.frombuffer(bytes.fromhex(json_line["x_bytes"]), dtype=np.float16).reshape(json_line["shape"])
    y_expected = np.frombuffer(bytes.fromhex(json_line["y_bytes"]), dtype=np.float16).reshape(json_line["shape"])

    # Parse the CANN-generated dump file via parse_exception_dump
    tensors = parse_exception_dump(dump_path, kernel_name="main_kernel", wait_seconds=0)

    assert len(tensors) >= 2, f"Expected at least 2 tensors, got {len(tensors)}"

    # bin files are sorted by name; input.0 < input.1
    x_dumped = tensors[0]["data"].reshape(tuple(json_line["shape"]))
    y_dumped = tensors[1]["data"].reshape(tuple(json_line["shape"]))

    np.testing.assert_array_equal(x_dumped, x_expected)
    np.testing.assert_array_equal(y_dumped, y_expected)
