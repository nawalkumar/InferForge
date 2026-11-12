"""Atomic report writing (architecture §4.10, IMP-P05).

`write_atomic(text, path)` writes UTF-8 to a temporary file in the target's directory and
renames it over `path`, so a reader never sees a partial file and an existing file is either
fully replaced (documented `--json` behaviour) or untouched. The file gets the mode a normal
`open()` would (`0o666 & ~umask`). Any failure raises `OutputError` (CLI exit 4) and removes
the temporary file.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path


class OutputError(Exception):
    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"cannot write {path}: {reason}")
        self.path = path
        self.reason = reason


def _reason(exc: OSError) -> str:
    return exc.strerror or type(exc).__name__


def write_atomic(text: str, path: Path) -> None:
    path = Path(path)
    if path.is_dir():
        raise OutputError(path, "Is a directory")
    try:
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    except OSError as exc:
        raise OutputError(path, _reason(exc)) from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp_name, 0o666 & ~umask)
        os.replace(tmp_name, path)
    except BaseException as exc:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        if isinstance(exc, OSError):
            raise OutputError(path, _reason(exc)) from exc
        raise
