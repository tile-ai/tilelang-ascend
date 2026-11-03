# Cache library publication regression tests

Run these tests without installing TileLang or initializing a device:

```bash
python -m unittest discover -s testing/python/cache -p 'test_*.py' -v
```

Alternatively, with pytest installed:

```bash
python -m pytest testing/python/cache/test_atomic_copy.py -q
```

The tests load the standard-library-only file helper directly, avoiding TVM and
device-runtime initialization. The supported platform is Linux; the shared-library
test additionally requires a host C compiler named `cc`. That test is skipped
when Linux or `cc` is unavailable. It compiles two tiny host libraries and runs
the loader checks in subprocesses with core dumps disabled.

## Why replacement must be atomic

A cached `kernel_lib.so` may already be mapped into another process. Copying
directly onto that pathname truncates and rewrites the mapped inode, which can
crash the loader or invalidate executable pages. A writer lock alone does not
protect existing readers from this operation.

The three library publishers use `atomic_copy`: copy into a unique temporary file
in the destination directory, then publish with `os.replace`. Existing mappings
retain the old inode; new readers see a complete library. Temporary files are
removed on failure. Source permission bits are preserved, as with `shutil.copy`.

Coverage includes first publication, old file descriptors and mappings, destination
visibility during copying, copy/replace failures, permissions, identical source
and destination paths, multiple publishing processes, and real shared-library
loading before and after replacement.

## Scope

This is single-file atomic publication, not a transaction for a complete cache
entry. It does not coordinate concurrent cache deletion, prevent readers from
observing independently updated metadata, or guarantee persistence after a system
crash. No cache-key format, generated kernel code, or kernel execution path changes.
