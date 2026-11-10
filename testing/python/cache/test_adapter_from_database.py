# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Cache-load regression test for ``adapter.libpath`` restoration.

When a compiled kernel is loaded from the disk cache, ``from_database`` must
restore ``lib_generator.libpath`` so that ``adapter.libpath`` points at the
cached ``kernel_lib.so`` instead of staying ``None``. Tests that re-open the
shared library in a subprocess (e.g. the trap test in
``test_tilelang_ascend_language_copy_ub_to_gm_alignment.py``) depend on it.

This test exercises the real ``CythonKernelAdapter.from_database`` path and only
stubs the external dependencies: the actual dynamic-library load
(``LibraryGenerator.load_lib``) and NPU device access (``torch.npu``). It does
not compile a kernel, run on a device, or trigger a trap.

``CtypesKernelAdapter.from_database`` carries the same libpath bug and fix, but
its ``from_database`` is currently not exercisable in isolation: it references
an undefined ``_legalize_result_idx`` and constructs ``LibraryGenerator``
without the required ``platform`` argument (pre-existing bit-rot since
PR #1496, unrelated to the libpath fix). It is covered by the same source fix
but not by this test until that bit-rot is addressed separately.
"""

from unittest import mock

import torch

import tilelang.language as T
from tilelang.engine.param import KernelParam
from tilelang.jit.adapter.cython.adapter import CythonKernelAdapter
from tilelang.jit.adapter.libgen import LibraryGenerator


@T.prim_func
def _simple(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):  # type: ignore
    with T.Kernel(1, is_npu=True) as (cid, vid):
        a_ub = T.alloc_ub((16,), "float32")
        T.copy(A, a_ub)
        T.copy(a_ub, B)


def test_from_database_restores_libpath():
    func = _simple
    params = [KernelParam.from_buffer(func.buffer_map[p]) for p in func.params]
    kernel_lib_path = "/fake/cache/key/kernel_lib.so"
    fake_lib = mock.MagicMock()

    with (
        mock.patch.object(LibraryGenerator, "load_lib", return_value=fake_lib) as load_lib,
        mock.patch.object(torch, "npu", mock.MagicMock()),
    ):
        adapter = CythonKernelAdapter.from_database(
            params=params,
            result_idx=[1],
            workspace_idx=[],
            auto_gm_idx=[],
            target="cuda",
            platform="A3",
            func_or_mod=func,
            kernel_global_source="// fake kernel source",
            kernel_lib_path=kernel_lib_path,
        )

    assert adapter.libpath == kernel_lib_path
    load_lib.assert_called_once_with(lib_path=kernel_lib_path)
