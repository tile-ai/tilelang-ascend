# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Filesystem helpers for publishing compiled artifacts."""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import suppress


def is_file_missing(path: str | os.PathLike) -> bool:
    """Return whether ``path`` positively does not exist on the filesystem.

    A cached ``kernel_lib.so`` may be deleted by a concurrent ``clear_cache``
    between the metadata read and the shared-library load. ``os.path.exists``
    cannot be used for that classification: it also returns ``False`` for
    permission errors and broken symlinks because it swallows ``OSError``, so a
    failing load plus ``os.path.exists(path) is False`` does not prove the file
    was deleted.

    ``os.stat`` distinguishes the cases. ``FileNotFoundError`` means the entry
    is gone and the caller may safely fall back to a cache miss. Any other
    ``OSError`` (permission, not-a-directory, ...) means the path is present but
    unusable, so the original error must be preserved.
    """
    try:
        os.stat(path)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def error_reports_missing_path(error: OSError, path: str | os.PathLike) -> bool:
    """Return whether ``error`` names ``path`` itself as the missing object.

    ``ctypes.CDLL(path)`` raises :class:`OSError` (``errno`` is ``None``) whose
    message is ``"<name>: cannot open shared object file: <reason>"`` where
    ``<name>`` is the object the dynamic loader failed to load. Only the exact
    ``"cannot open shared object file: No such file or directory"`` reason means
    an object is missing; other reasons (``"file too short"``, ``"wrong ELF
    class"``, permission failures) are not missing-library errors. The named
    object must also resolve to ``path`` itself, so a same-basename library in
    another directory or a dependency whose name merely contains the target's
    basename is never mistaken for the target.
    """
    message = str(error)
    suffix = ": cannot open shared object file: No such file or directory"
    if not message.endswith(suffix):
        return False
    named = message[: -len(suffix)]
    return os.path.abspath(named) == os.path.abspath(os.fspath(path))


def atomic_copy(src: str | os.PathLike, dst: str | os.PathLike) -> None:
    """Publish a complete file without modifying an existing destination inode.

    Cached shared libraries may already be mapped into another process. Copying
    directly onto them truncates that process's backing file. A temporary file
    in the destination directory followed by replace preserves existing mappings
    and ensures readers see either the old or the new complete library.

    This does not make a multi-file cache entry transactional or synchronize
    publication with concurrent cache deletion.
    """
    dst = os.fspath(dst)
    fd, temporary_path = tempfile.mkstemp(prefix=f".{os.path.basename(dst)}.", suffix=".tmp", dir=os.path.dirname(dst) or ".")
    try:
        os.close(fd)
        shutil.copy(src, temporary_path)
        os.replace(temporary_path, dst)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary_path)
