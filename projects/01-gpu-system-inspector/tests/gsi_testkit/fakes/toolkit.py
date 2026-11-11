"""Fake `ToolkitFiles`: an in-memory read-only tree with symlinks and injected failures.

`from_snapshot` builds the tree a fixture's CUDA environment implies: each candidate root
(with its `version.json` when that was the version source, and `bin/nvcc` when present) and
the active `nvcc`. Paths are POSIX-style, Windows fixtures included (`C:/Program Files/…`).
"""

from __future__ import annotations

import json
import posixpath
from dataclasses import dataclass, field

from gpu_system_inspector.models import DataOrigin, InspectionSnapshot

DIRECTORY, FILE = "directory", "file"


def nvcc_version_output(version: str) -> str:
    """What `nvcc --version` prints for `version` (`X.Y.Z`)."""
    release = ".".join(version.split(".")[:2])
    return (
        "nvcc: NVIDIA (R) Cuda compiler driver\n"
        "Copyright (c) 2005-2024 NVIDIA Corporation\n"
        f"Cuda compilation tools, release {release}, V{version}\n"
    )


def version_json(version: str) -> str:
    return json.dumps({"cuda": {"name": "CUDA SDK", "version": version}})


@dataclass
class FakeToolkitFiles:
    directories: set[str] = field(default_factory=set)
    files: dict[str, str] = field(default_factory=dict)
    links: dict[str, str] = field(default_factory=dict)
    failures: dict[tuple[str, str], OSError] = field(default_factory=dict)
    calls: list[tuple[str, str]] = field(default_factory=list)

    provenance = DataOrigin.FIXTURE

    @classmethod
    def from_snapshot(cls, snapshot: InspectionSnapshot) -> FakeToolkitFiles:
        tree = cls()
        cuda = snapshot.cuda_environment
        if cuda.toolkits.is_collected:
            for candidate in cuda.toolkits.value or ():
                tree.add_directory(candidate.root_path)
                version = candidate.version
                if version.is_collected and (version.method or "").endswith(
                    "version.json:cuda.version"
                ):
                    tree.add_file(
                        posixpath.join(candidate.root_path, "version.json"),
                        version_json(version.value or ""),
                    )
                if candidate.nvcc_path.is_collected:
                    tree.add_file(candidate.nvcc_path.value or "")
        if cuda.active_nvcc_path.is_collected:
            tree.add_file(cuda.active_nvcc_path.value or "")
        return tree

    # --- building -------------------------------------------------------------------------

    def add_directory(self, path: str) -> FakeToolkitFiles:
        while path not in ("", "/") and posixpath.dirname(path) != path:
            self.directories.add(path)
            path = posixpath.dirname(path)
        return self

    def add_file(self, path: str, content: str = "") -> FakeToolkitFiles:
        self.files[path] = content
        self.add_directory(posixpath.dirname(path))
        return self

    def add_link(self, path: str, target: str) -> FakeToolkitFiles:
        self.links[path] = target
        self.add_directory(posixpath.dirname(path))
        return self

    def fail(self, operation: str, path: str, error: OSError) -> FakeToolkitFiles:
        self.failures[(operation, path)] = error
        return self

    # --- ToolkitFiles ---------------------------------------------------------------------

    def kind(self, path: str) -> str | None:
        self._call("kind", path)
        resolved = self.realpath(path)
        if resolved in self.directories:
            return DIRECTORY
        if resolved in self.files:
            return FILE
        return None

    def list_dir(self, path: str) -> list[str]:
        self._call("list_dir", path)
        resolved = self.realpath(path)
        if resolved not in self.directories:
            raise FileNotFoundError(2, "No such file or directory", path)
        entries = self.directories | set(self.files) | set(self.links)
        return sorted(
            posixpath.basename(e)
            for e in entries
            if e != resolved and posixpath.dirname(e) == resolved
        )

    def read_text(self, path: str, limit: int) -> str:
        self._call("read_text", path)
        resolved = self.realpath(path)
        if resolved in self.files:
            return self.files[resolved][: limit + 1]
        if resolved in self.directories:
            raise IsADirectoryError(21, "Is a directory", path)
        raise FileNotFoundError(2, "No such file or directory", path)

    def realpath(self, path: str) -> str:
        parts = path.split("/")
        resolved = parts[0]
        for part in parts[1:]:
            if part in ("", "."):  # as os.path.realpath: no trailing or doubled separators
                continue
            resolved = f"{resolved}/{part}" if resolved != "/" else f"/{part}"
            hops = 0
            while resolved in self.links:
                resolved = self.links[resolved]
                hops += 1
                assert hops < 40, f"symlink loop at {path}"
        return resolved or path

    def join(self, *parts: str) -> str:
        return posixpath.join(*parts)

    def dirname(self, path: str) -> str:
        return posixpath.dirname(path)

    def basename(self, path: str) -> str:
        return posixpath.basename(path)

    def _call(self, operation: str, path: str) -> None:
        self.calls.append((operation, path))
        failure = self.failures.get((operation, path))
        if failure is not None:
            raise failure
