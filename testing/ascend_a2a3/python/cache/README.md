# Cache library publication regression tests

Run these tests with TileLang and TVM installed; no NPU kernel is executed:

```bash
python -m unittest discover -s testing/ascend_a2a3/python/cache -p 'test_*.py' -v
```

Alternatively, with pytest installed:

```bash
python -m pytest testing/ascend_a2a3/python/cache/test_atomic_copy.py -q
```

The tests use the current cache interfaces. A2/A3 loader recovery checks require
an Ascend-enabled build. The supported platform is Linux; the shared-library
test additionally requires a host C compiler named `cc`. That test is skipped
when Linux or `cc` is unavailable. It compiles two tiny host libraries and runs
the loader checks in subprocesses with core dumps disabled.

## Why replacement must be atomic

A cached `kernel_lib.so` may already be mapped into another process. Copying
directly onto that pathname truncates and rewrites the mapped inode, which can
crash the loader or invalidate executable pages. A writer lock alone does not
protect existing readers from this operation.

The publication checks use `KernelCache._safe_write_file`: write into a unique
temporary file in the destination directory, then publish with `os.replace`.
Existing mappings retain the old inode; new readers see a complete library.
Temporary files are removed on failure. Files use normal creation permissions
from the current umask instead of inheriting source permission bits.

Coverage includes first publication, old file descriptors and mappings, destination
visibility during copying, copy/replace failures, permissions, identical source
and destination paths, multiple publishing processes, and real shared-library
loading before and after replacement.

## Scope

This is single-file atomic publication, not a transaction for a complete cache
entry. It does not coordinate concurrent cache deletion, prevent readers from
observing independently updated metadata, or guarantee persistence after a system
crash. No cache-key format, generated kernel code, or kernel execution path changes.
