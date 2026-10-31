# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Original publication scenarios on the current two real atomic writers.

The retired stdlib atomic_copy entry is not recreated. A test-local operation
reads source bytes, as both current library publishers do, while the actual
_safe_write_file owns the temporary file, fsync, replace and cleanup.
"""

import mmap
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tilelang.autotuner.param import AutotuneResult
from tilelang.cache.kernel_cache import KernelCache


# A fresh child does not inherit ctypes-loaded libraries from an actual OFF
# parent runner. Reproduce that runner's explicit OFF load, never an ON fallback.
_CHILD_BOOTSTRAP = """
import ctypes
import os
import pathlib
if os.environ.get('SKIP_LOADING_TILELANG_SO') == '1':
    import tvm
    ctypes.CDLL(str(pathlib.Path(os.environ['TVM_LIBRARY_PATH']) / 'libtilelang.so'), mode=ctypes.RTLD_GLOBAL)
"""


class TestAtomicCopy(unittest.TestCase):
    publisher = KernelCache
    publisher_name = "kernel"

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.src = self.root / "compiled.so"
        self.dst = self.root / "kernel_lib.so"
        self.src.write_bytes(b"new library" * 1024)
        self.dst.write_bytes(b"old library" * 1024)

    def publish(self, src, dst, operation=None):
        def write(file):
            if operation is None:
                file.write(Path(src).read_bytes())
            else:
                operation(src, file)

        self.publisher._safe_write_file(str(dst), "wb", write)

    def assert_no_temporary_files(self):
        self.assertEqual(list(self.root.glob(".kernel_lib.so.*.tmp")), [])

    def test_first_publication(self):
        self.dst.unlink()
        self.publish(self.src, self.dst)
        self.assertEqual(self.dst.read_bytes(), self.src.read_bytes())
        self.assert_no_temporary_files()

    def test_preserves_existing_mapping(self):
        original = self.dst.read_bytes()
        with self.dst.open("rb") as old_file:
            old_inode = os.fstat(old_file.fileno()).st_ino
            with mmap.mmap(old_file.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                self.publish(self.src, self.dst)
                self.assertEqual(mapped[:], original)
                self.assertEqual(old_file.read(), original)
                self.assertNotEqual(self.dst.stat().st_ino, old_inode)
        self.assertEqual(self.dst.read_bytes(), self.src.read_bytes())

    def test_destination_unchanged_during_copy(self):
        original = self.dst.read_bytes()

        def interrupted_copy(src, file):
            file.write(b"partial")
            self.assertEqual(self.dst.read_bytes(), original)
            file.seek(0)
            file.write(Path(src).read_bytes())

        self.publish(self.src, self.dst, interrupted_copy)
        self.assertEqual(self.dst.read_bytes(), self.src.read_bytes())
        self.assert_no_temporary_files()

    def test_copy_failure_keeps_old_file_and_cleans_temporary(self):
        original = self.dst.read_bytes()

        def fail_copy(src, file):
            file.write(b"partial")
            raise OSError("simulated copy failure")

        with self.assertRaisesRegex(OSError, "simulated copy failure"):
            self.publish(self.src, self.dst, fail_copy)
        self.assertEqual(self.dst.read_bytes(), original)
        self.assert_no_temporary_files()

    def test_replace_failure_keeps_old_file_and_cleans_temporary(self):
        original = self.dst.read_bytes()
        with patch("os.replace", side_effect=OSError("replace failure")), self.assertRaisesRegex(OSError, "replace failure"):
            self.publish(self.src, self.dst)
        self.assertEqual(self.dst.read_bytes(), original)
        self.assert_no_temporary_files()

    def test_same_source_and_destination(self):
        original = self.dst.read_bytes()
        self.publish(self.dst, self.dst)
        self.assertEqual(self.dst.read_bytes(), original)
        self.assert_no_temporary_files()

    @unittest.skipUnless(os.name == "posix", "POSIX permissions")
    def test_preserves_source_permissions(self):
        # Retain the source scenario/input, but assert the actual modern writer
        # contract: default creation mode, not retired shutil.copy metadata.
        self.src.chmod(0o750)
        control = self.root / "default-created-file"
        control.write_bytes(b"control")
        self.publish(self.src, self.dst)
        self.assertEqual(self.src.stat().st_mode & 0o777, 0o750)
        self.assertEqual(self.dst.stat().st_mode & 0o777, control.stat().st_mode & 0o777)
        self.assertEqual(self.dst.read_bytes(), self.src.read_bytes())
        self.assert_no_temporary_files()

    def test_concurrent_processes_publish_complete_files(self):
        script = (
            _CHILD_BOOTSTRAP
            + """
import pathlib
import sys
from tilelang.cache.kernel_cache import KernelCache
from tilelang.autotuner.param import AutotuneResult
writer = KernelCache if sys.argv[3] == 'kernel' else AutotuneResult
for _ in range(40):
    writer._safe_write_file(sys.argv[2], 'wb', lambda file: file.write(pathlib.Path(sys.argv[1]).read_bytes()))
    assert pathlib.Path(sys.argv[2]).read_bytes() in (b'A' * 65536, b'B' * 131072)
"""
        )
        sources = [self.root / "a.so", self.root / "b.so"]
        sources[0].write_bytes(b"A" * 65536)
        sources[1].write_bytes(b"B" * 131072)
        self.publish(sources[0], self.dst)
        processes = []
        try:
            for i in range(4):
                processes.append(
                    subprocess.Popen(
                        [sys.executable, "-c", script, str(sources[i % 2]), str(self.dst), self.publisher_name],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                )
            for process in processes:
                _, stderr = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 0, stderr.decode())
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.communicate()
        self.assert_no_temporary_files()

    @unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("cc"), "Requires Linux and a host C compiler")
    def test_loaded_shared_library_survives_republication(self):
        for filename, value in ((self.dst, 17), (self.src, 29)):
            subprocess.run(
                ["cc", "-shared", "-fPIC", "-x", "c", "-o", str(filename), "-"],
                input=f"int cached_value(void) {{ return {value}; }}".encode(),
                check=True,
                capture_output=True,
                timeout=30,
            )
        script = (
            _CHILD_BOOTSTRAP
            + """
import ctypes
import pathlib
import resource
import sys
from tilelang.cache.kernel_cache import KernelCache
from tilelang.autotuner.param import AutotuneResult
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
writer = KernelCache if sys.argv[3] == 'kernel' else AutotuneResult
loaded = ctypes.CDLL(sys.argv[2])
assert loaded.cached_value() == 17
writer._safe_write_file(sys.argv[2], 'wb', lambda file: file.write(pathlib.Path(sys.argv[1]).read_bytes()))
assert loaded['cached_value']() == 17
"""
        )
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.src), str(self.dst), self.publisher_name],
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        result = subprocess.run(
            [sys.executable, "-c", "import ctypes, sys; assert ctypes.CDLL(sys.argv[1]).cached_value() == 29", str(self.dst)],
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode())


class TestAutotuneAtomicCopy(TestAtomicCopy):
    """The independently implemented autotune writer has the same safety needs."""

    publisher = AutotuneResult
    publisher_name = "autotune"


if __name__ == "__main__":
    unittest.main()
