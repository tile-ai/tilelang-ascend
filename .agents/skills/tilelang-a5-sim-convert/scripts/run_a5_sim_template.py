#!/usr/bin/env python3
"""Run tilelang DSL kernel on A5 camodel simulator — one command, no manual env setup.

Usage:
    python testing/camodel/run_a5_sim.py                  # default: gemm
    python testing/camodel/run_a5_sim.py -t gemm           # explicit

All environment (CANN paths, camodel lib, torch_npu disable) is set up
internally, before any import.
"""

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import numpy as np


# =============================================================================
# Phase 0 — Environment auto-setup (runs BEFORE import tilelang / torch)
# =============================================================================

def _find_ascend_home():
    for d in [os.environ.get("ASCEND_HOME_PATH", ""),
              os.environ.get("ASCEND_HOME", ""),
              "/usr/local/Ascend/ascend-toolkit/latest"]:
        if d and os.path.isdir(d):
            return d
    for base in ["/usr/local/CANN", "/usr/local/Ascend"]:
        if not os.path.isdir(base):
            continue
        for e in sorted(os.listdir(base), reverse=True):
            p = os.path.join(base, e)
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "bin", "setenv.bash")):
                return p
    raise RuntimeError("CANN not found. Set ASCEND_HOME_PATH.")


def _source_cann(ascend_home):
    """Source CANN setenv.bash and capture resulting environment."""
    setenv = os.path.join(ascend_home, "bin", "setenv.bash")
    if not os.path.exists(setenv):
        return
    r = subprocess.run(
        f"source {setenv} && env",
        shell=True, executable=shutil.which("bash") or "bash",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    for line in r.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ[k] = v


def _find_sim_lib(ascend_home):
    """Find Ascend950PR_9599 simulator lib/ directory."""
    sim_base = os.path.join(ascend_home, "tools", "simulator")
    for soc in ["Ascend950PR_9599", "Ascend910_9599"]:
        lib_dir = os.path.join(sim_base, soc, "lib")
        if os.path.isdir(lib_dir):
            return lib_dir
    raise RuntimeError(f"Simulator not found under {sim_base}")


def setup(log_dir=None):
    ascend_home = _find_ascend_home()
    os.environ["ASCEND_HOME_PATH"] = ascend_home
    _source_cann(ascend_home)

    sim_lib = _find_sim_lib(ascend_home)
    lib64 = os.path.join(ascend_home, "x86_64-linux", "lib64")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    paths = [sim_lib, lib64] + [p for p in ld.split(":") if p and p not in (sim_lib, lib64)]
    os.environ["LD_LIBRARY_PATH"] = ":".join(paths)

    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["TL_RUN_MODE"] = "sim"
    os.environ["TL_PLATFORM"] = "A5"

    # Camodel log output directory (default: ./camodel_log)
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), "camodel_log")
    os.makedirs(log_dir, exist_ok=True)
    os.environ["CAMODEL_LOG_PATH"] = log_dir

    # Re-exec self if LD_LIBRARY_PATH wasn't set at shell level before launch.
    if "_A5_SIM_REEXEC" not in os.environ:
        os.environ["_A5_SIM_REEXEC"] = "1"
        os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)

    print(f"[INFO] ascend:  {ascend_home}")
    print(f"[INFO] sim lib: {sim_lib}")
    print(f"[INFO] log dir: {log_dir}")
    return sim_lib


# =============================================================================
# Phase 1 — Load camodel runtime (rt APIs)
# =============================================================================

def load_runtime(sim_lib):
    sys.setdlopenflags(os.RTLD_LAZY | os.RTLD_GLOBAL)
    # Load by full path — LD_LIBRARY_PATH changes from within Python
    # don't reliably propagate to dlopen.
    rt_path = os.path.join(sim_lib, "libruntime_camodel.so")
    rt = ctypes.CDLL(rt_path)
    rt.rtSetDevice.argtypes = [ctypes.c_int32]
    rt.rtSetDevice.restype = ctypes.c_int
    rt.rtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64, ctypes.c_int, ctypes.c_uint16]
    rt.rtMalloc.restype = ctypes.c_int
    rt.rtFree.argtypes = [ctypes.c_void_p]
    rt.rtFree.restype = ctypes.c_int
    rt.rtMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
    rt.rtMemcpy.restype = ctypes.c_int
    rt.rtStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int32]
    rt.rtStreamCreate.restype = ctypes.c_int
    rt.rtStreamSynchronize.argtypes = [ctypes.c_void_p]
    rt.rtStreamSynchronize.restype = ctypes.c_int
    rt.rtStreamDestroy.argtypes = [ctypes.c_void_p]
    rt.rtStreamDestroy.restype = ctypes.c_int
    rt.rtDeviceReset.argtypes = [ctypes.c_int32]
    rt.rtDeviceReset.restype = ctypes.c_int
    if rt.rtSetDevice(0) != 0:
        raise RuntimeError("rtSetDevice(0) failed")
    return rt


def dev_malloc(rt, size):
    p = ctypes.c_void_p()
    assert rt.rtMalloc(ctypes.byref(p), size, 2, 0) == 0
    return p


# =============================================================================
# Phase 2 — Kernel definition (add your own here)
# =============================================================================

def make_gemm(M=1024, N=512, K=256, block_M=128, block_N=256, K_L1=64):
    import tilelang.language as T
    m_num, n_num = M // block_M, N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx, by = cid // n_num, cid % n_num
            A_L1 = T.alloc_L1((block_M, K_L1), "float16")
            B_L1 = T.alloc_L1((K_L1, block_N), "float16")
            C_L0 = T.alloc_L0C((block_M, block_N), "float")
            with T.Scope("C"):
                for k in T.serial(T.ceildiv(K, K_L1)):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)
                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                    T.barrier_all()
                T.copy(C_L0, C[bx * block_M, by * block_N])
    return main


KERNELS = {"gemm": make_gemm}


# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="A5 camodel simulator — one-command runner")
    parser.add_argument("-t", "--kernel", default="gemm", choices=list(KERNELS.keys()),
                        help="Built-in kernel name")
    parser.add_argument("--log-dir", default=None,
                        help="Camodel log output directory (default: ./camodel_log)")
    args = parser.parse_args()

    # ---- 0. auto env setup ----
    sim_lib = setup(log_dir=args.log_dir)

    # ---- 1. load camodel runtime ----
    print("=== Load camodel runtime ===")
    rt = load_runtime(sim_lib)

    # ---- 2. import tilelang (now env is ready) ----
    import tilelang
    from tilelang.jit.adapter.libgen import LibraryGenerator
    tilelang.cache.clear_cache()

    # ---- 3. generate kernel source ----
    print(f"=== Generate kernel: {args.kernel} ===")
    prim_func = KERNELS[args.kernel]()
    artifact = tilelang.lower(prim_func, target="pto", platform="A5")
    print(f"  Source: {len(artifact.kernel_source.splitlines())} lines")

    # ---- 4. compile .so ----
    print("=== Compile kernel ===")
    libgen = LibraryGenerator(target="pto", platform="A5")
    libgen.update_lib_code(artifact.kernel_source)
    libgen.compile_lib()
    so = libgen.get_lib_path()
    print(f"  {so}")

    # ---- 5. load kernel .so ----
    print("=== Load kernel ===")
    kl = ctypes.CDLL(so)
    kl.call.argtypes = [ctypes.c_void_p] * 4
    kl.call.restype = None

    # ---- 6. prepare data ----
    print("=== Prepare data ===")
    M, N, K = 1024, 512, 256
    h_A = np.zeros((M, K), dtype=np.float16)
    h_B = np.zeros((K, N), dtype=np.float16)
    h_C = np.zeros((M, N), dtype=np.float16)
    for i in range(M):
        for kk in range(K):
            h_A[i, kk] = np.float16((i % 100 + 1) * (kk % 100 + 1) * 0.0001)
    for kk in range(K):
        for j in range(N):
            h_B[kk, j] = np.float16((kk % 100 + 1) * (j % 100 + 1) * 0.0001)
    h_Ref = h_A.astype(np.float32) @ h_B.astype(np.float32)

    d_A = dev_malloc(rt, M * K * 2)
    d_B = dev_malloc(rt, K * N * 2)
    d_C = dev_malloc(rt, M * N * 2)
    rt.rtMemcpy(d_A, M * K * 2, h_A.ctypes.data, M * K * 2, 1)
    rt.rtMemcpy(d_B, K * N * 2, h_B.ctypes.data, K * N * 2, 1)
    stream = ctypes.c_void_p()
    rt.rtStreamCreate(ctypes.byref(stream), 0)

    # ---- 7. launch ----
    print("=== Launch kernel ===")
    kl.call(d_A, d_B, d_C, stream)
    rt.rtStreamSynchronize(stream)
    rt.rtMemcpy(h_C.ctypes.data, M * N * 2, d_C, M * N * 2, 2)

    # ---- 8. verify ----
    print("=== Verify ===")
    diff = np.abs(h_C.astype(np.float32) - h_Ref)
    rel = diff / np.maximum(np.abs(h_Ref), 1e-6)
    print(f"  Max abs error: {diff.max():.6f}")
    print(f"  Max rel error: {rel.max():.6f}")
    if rel.max() < 0.1:
        print("KERNEL OUTPUT MATCH!")
    else:
        print("FAILED")
        sys.exit(1)

    rt.rtFree(d_A); rt.rtFree(d_B); rt.rtFree(d_C)
    rt.rtStreamDestroy(stream); rt.rtDeviceReset(0)
    libgen.remove_lib()
    print("Done!")


if __name__ == "__main__":
    main()
