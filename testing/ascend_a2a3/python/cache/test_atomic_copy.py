# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Host-only cache publication tests for the current atomic writer."""

import mmap
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tilelang.cache.kernel_cache import KernelCache


class TestAtomicCopy(unittest.TestCase):
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
                file.write(KernelCache._load_binary(str(src)))
            else:
                operation(src, file)

        KernelCache._safe_write_file(str(dst), "wb", write)

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
            file.write(KernelCache._load_binary(str(src)))

        self.publish(self.src, self.dst, interrupted_copy)
        self.assertEqual(self.dst.read_bytes(), self.src.read_bytes())

    def test_copy_failure_keeps_old_file_and_cleans_temporary(self):
        original = self.dst.read_bytes()

        def fail_copy(src, file):
            file.write(b"partial")
            raise OSError("simulated copy failure")

        with self.assertRaises(OSError):
            self.publish(self.src, self.dst, fail_copy)
        self.assertEqual(self.dst.read_bytes(), original)
        self.assert_no_temporary_files()

    def test_replace_failure_keeps_old_file_and_cleans_temporary(self):
        original = self.dst.read_bytes()
        with patch.object(os, "replace", side_effect=OSError("replace failure")), self.assertRaises(OSError):
            self.publish(self.src, self.dst)
        self.assertEqual(self.dst.read_bytes(), original)
        self.assert_no_temporary_files()

    def test_same_source_and_destination(self):
        original = self.dst.read_bytes()
        self.publish(self.dst, self.dst)
        self.assertEqual(self.dst.read_bytes(), original)
        self.assert_no_temporary_files()

    @unittest.skipUnless(os.name == "posix", "POSIX permissions")
    def test_uses_default_creation_permissions(self):
        self.src.chmod(0o750)
        control = self.root / "default-created-file"
        control.write_bytes(b"control")
        self.publish(self.src, self.dst)
        self.assertEqual(self.dst.stat().st_mode & 0o777, control.stat().st_mode & 0o777)

    def test_concurrent_processes_publish_complete_files(self):
        script = """
import pathlib
import sys
from tilelang.cache.kernel_cache import KernelCache
for _ in range(40):
    KernelCache._safe_write_file(sys.argv[2], "wb", lambda file: file.write(KernelCache._load_binary(sys.argv[1])))
    assert pathlib.Path(sys.argv[2]).read_bytes() in (b'A' * 65536, b'B' * 131072)
"""
        sources = [self.root / "a.so", self.root / "b.so"]
        sources[0].write_bytes(b"A" * 65536)
        sources[1].write_bytes(b"B" * 131072)
        self.publish(sources[0], self.dst)
        processes = []
        try:
            for i in range(4):
                processes.append(
                    subprocess.Popen(
                        [sys.executable, "-c", script, str(sources[i % 2]), str(self.dst)],
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
        # Run the dlopen scenario in a subprocess: the old in-place publication
        # can crash the dynamic loader. Never overwrite a real kernel cache.
        for filename, value in ((self.dst, 17), (self.src, 29)):
            subprocess.run(
                ["cc", "-shared", "-fPIC", "-x", "c", "-o", str(filename), "-"],
                input=f"int cached_value(void) {{ return {value}; }}".encode(),
                check=True,
                capture_output=True,
                timeout=30,
            )
        script = """
import ctypes
import resource
import sys
from tilelang.cache.kernel_cache import KernelCache
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
loaded = ctypes.CDLL(sys.argv[2])
assert loaded.cached_value() == 17
KernelCache._safe_write_file(sys.argv[2], "wb", lambda file: file.write(KernelCache._load_binary(sys.argv[1])))
assert loaded['cached_value']() == 17
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.src), str(self.dst)],
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


if __name__ == "__main__":
    unittest.main()
