"""Import/call diagnostics remain usable with a real OFF native library."""

import pytest

import tilelang.ascend.a2a3.language as T
from tilelang.ascend.a2a3 import _native


def test_private_persistent_missing_constructor_is_a_clear_native_diagnostic(monkeypatch):
    actual = _native.ffi.get_global_func

    def missing(name, *args, **kwargs):
        return None if name == "tl.A2A3Persistent" else actual(name, *args, **kwargs)

    monkeypatch.setattr(_native.ffi, "get_global_func", missing)
    with pytest.raises(RuntimeError, match="Persistent.*USE_ASCEND=ON.*tl.A2A3Persistent"):
        T.Persistent([4, 2], 20, 0)
