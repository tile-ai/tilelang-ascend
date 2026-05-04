"""Minimal smoke test: libtilelang loads and the tensorpulse target is wired
end-to-end through TVM and tilelang.utils.target.determine_target.

Run from repo root:

    export PYTHONPATH=$PWD/3rdparty/tvm/python:$PYTHONPATH
    export TVM_LIBRARY_PATH=$PWD/build/tvm
    python3 testing/python/test_tensorpulse_target_smoke.py

We stub the tilelang package instead of running its __init__.py to avoid
pulling in torch / Cython adapter compilation that aren't needed for a
target-registration smoke test.
"""

import ctypes
import os
import sys
import types

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LIB_NAMES = ("libtilelang_module.so", "libtilelang_module.dylib",
              "libtilelang.so", "libtilelang.dylib")
LIB = next((os.path.join(REPO, "build", n)
            for n in _LIB_NAMES
            if os.path.exists(os.path.join(REPO, "build", n))), None)


def _stub_tilelang_package(tvm_module):
    """Register a minimal tilelang package so submodules import cleanly."""
    pkg = types.ModuleType("tilelang")
    pkg.__path__ = [os.path.join(REPO, "tilelang")]
    pkg.tvm = tvm_module  # satisfies `from tilelang import tvm as tvm`
    sys.modules["tilelang"] = pkg

    contrib = types.ModuleType("tilelang.contrib")
    contrib.__path__ = [os.path.join(REPO, "tilelang", "contrib")]
    sys.modules["tilelang.contrib"] = contrib


def main():
    assert LIB is not None and os.path.exists(LIB), (
        f"no tilelang shared lib under {REPO}/build (tried {_LIB_NAMES}); "
        "build the project with USE_TENSORPULSE=ON first")

    # 1. ctypes-load libtilelang so its TVM_REGISTER_GLOBAL hooks fire.
    import tvm
    from tvm.target import Target

    handle = ctypes.CDLL(LIB)
    print(f"[ok] loaded {handle._name}")

    # 2. target.build.tilelang_tensorpulse registered by rt_mod_tensorpulse.cc.
    fn = tvm._ffi.get_global_func("target.build.tilelang_tensorpulse",
                                  allow_missing=True)
    assert fn is not None, "target.build.tilelang_tensorpulse not registered"
    print("[ok] global func target.build.tilelang_tensorpulse registered")

    # 3. determine_target maps the string alias to the llvm target with the key.
    _stub_tilelang_package(tvm)
    from tilelang.utils.target import determine_target

    s = determine_target("tensorpulse")
    assert s == "llvm --keys=tensorpulse", f"unexpected target string: {s!r}"
    print(f"[ok] determine_target('tensorpulse') -> {s!r}")

    # 4. The Target object actually carries the tensorpulse key.
    tgt = Target(s)
    keys = list(tgt.keys)
    assert "tensorpulse" in keys, f"missing tensorpulse key: {keys}"
    print(f"[ok] Target(...).keys = {keys}")

    # 5. Lowering pass registered (transform/tensorpulse_lower_opaque_block.cc).
    pass_fn = tvm._ffi.get_global_func("tl.transform.TensorPulseLowerOpaqueBlock",
                                       allow_missing=True)
    print(f"[{'ok' if pass_fn else 'warn'}] TensorPulseLowerOpaqueBlock "
          f"{'registered' if pass_fn else 'NOT FOUND under tl.transform.*'}")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    sys.exit(main() or 0)
