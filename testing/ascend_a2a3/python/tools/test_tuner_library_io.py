"""Own-library IO classification needs no Ascend target or native operators."""

import ctypes
import errno
import importlib
import os
import sys

import pytest

cache_module = importlib.import_module("tilelang.ascend.a2a3.kernel_cache")


def _missing_loader(path):
    return OSError(f"{path}: cannot open shared object file: No such file or directory")


@pytest.mark.parametrize("native_error", [False, True])
def test_only_confirmed_missing_requested_library_is_recoverable(tmp_path, native_error):
    library = tmp_path / "kernel_lib.so"
    error = _missing_loader(library) if native_error else FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(library))
    assert cache_module._is_missing_own_library(error, library)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="CANN native loader diagnostic uses the Linux dlopen format")
def test_real_loader_missing_object_error_is_recoverable(tmp_path):
    library = tmp_path / "kernel_lib.so"
    with pytest.raises(OSError) as failure:
        ctypes.CDLL(str(library))
    assert cache_module._is_missing_own_library(failure.value, library)


@pytest.mark.parametrize(
    "kind",
    ["permission", "wrong_elf", "truncated", "dependency", "foreign_directory", "basename", "metadata", "external_file"],
)
def test_nearby_errors_do_not_become_missing_library(tmp_path, kind):
    library = tmp_path / "kernel_lib.so"
    errors = {
        "permission": PermissionError(errno.EACCES, "Permission denied", str(library)),
        "wrong_elf": OSError(f"{library}: wrong ELF class: ELFCLASS32"),
        "truncated": OSError(f"{library}: file too short"),
        "dependency": _missing_loader("libkernel_lib.so"),
        "foreign_directory": _missing_loader(tmp_path / "other" / library.name),
        "basename": _missing_loader(library.name),
        "metadata": FileNotFoundError(errno.ENOENT, "No such file or directory", str(tmp_path / "a2a3_runtime.json")),
        "external_file": FileNotFoundError(errno.ENOENT, "No such file or directory", str(tmp_path / "user-input.pkl")),
    }
    assert not cache_module._is_missing_own_library(errors[kind], library)


def test_recreated_library_is_not_a_cache_miss(tmp_path):
    library = tmp_path / "kernel_lib.so"
    error = _missing_loader(library)
    library.write_bytes(b"recreated before classification")
    assert not cache_module._is_missing_own_library(error, library)


@pytest.mark.parametrize("stat_errno", [errno.EACCES, errno.ENOTDIR])
def test_stat_failure_is_not_proof_of_removal(tmp_path, monkeypatch, stat_errno):
    library = tmp_path / "kernel_lib.so"
    original = os.stat

    def stat(path, *args, **kwargs):
        if os.fspath(path) == str(library):
            raise OSError(stat_errno, os.strerror(stat_errno), path)
        return original(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(cache_module.os, "stat", stat)
        assert not cache_module._is_missing_own_library(_missing_loader(library), library)
