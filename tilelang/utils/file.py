# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Filesystem helpers for publishing compiled artifacts."""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import suppress


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
