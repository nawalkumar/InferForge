"""Shared helpers for discovering and describing InferForge workspace members."""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
MEMBER_KINDS = {"packages": "package", "projects": "project"}
IGNORED_DIRS = {
    ".venv",
    "venv",
    "build",
    "dist",
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "node_modules",
}

_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def normalize_name(name: str) -> str:
    """PEP 503 normalisation, so `Foo_Bar` and `foo-bar` compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(requirement: str) -> str | None:
    match = _REQ_NAME.match(requirement)
    return normalize_name(match.group(1)) if match else None


def load_toml(path: Path) -> dict:
    with path.open("rb") as fh:
        return tomllib.load(fh)


@dataclass
class Member:
    kind: str  # "package" or "project"
    path: Path
    pyproject: dict
    name: str = ""
    runtime_requirements: list[str] = field(default_factory=list)
    optional_requirements: list[str] = field(default_factory=list)
    group_requirements: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        project = self.pyproject.get("project", {})
        self.name = normalize_name(project.get("name", self.path.name))
        self.runtime_requirements = list(project.get("dependencies", []))
        for reqs in project.get("optional-dependencies", {}).values():
            self.optional_requirements.extend(reqs)
        for group, reqs in self.pyproject.get("dependency-groups", {}).items():
            self.group_requirements[group] = [r for r in reqs if isinstance(r, str)]

    @property
    def version(self) -> str | None:
        return self.pyproject.get("project", {}).get("version")

    def all_requirements(self) -> list[str]:
        reqs = self.runtime_requirements + self.optional_requirements
        for group in self.group_requirements.values():
            reqs.extend(group)
        return reqs

    def requirement_names(self, runtime_only: bool = False) -> set[str]:
        reqs = self.runtime_requirements if runtime_only else self.all_requirements()
        return {n for n in (requirement_name(r) for r in reqs) if n}

    def import_roots(self) -> set[str]:
        """Top-level importable names this member provides."""
        base = self.path / "src"
        if not base.is_dir():
            base = self.path
        roots = set()
        for entry in base.iterdir():
            if entry.name.startswith((".", "_")) or entry.name in IGNORED_DIRS:
                continue
            if entry.name in {"tests", "test", "docs", "scripts"}:
                continue
            if entry.is_dir() and (entry / "__init__.py").exists():
                roots.add(entry.name)
            elif entry.is_file() and entry.suffix == ".py" and base.name == "src":
                roots.add(entry.stem)
        return roots

    def python_files(self) -> list[Path]:
        files = []
        for path in self.path.rglob("*.py"):
            rel_parts = path.relative_to(self.path).parts
            if any(part in IGNORED_DIRS for part in rel_parts):
                continue
            files.append(path)
        return sorted(files)

    def display(self) -> str:
        return f"{self.kind} '{self.name}' ({self.path.relative_to(self.path.parents[1])})"


def discover_members(root: Path = REPO_ROOT) -> list[Member]:
    members = []
    for directory, kind in MEMBER_KINDS.items():
        base = root / directory
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            pyproject = child / "pyproject.toml"
            if child.is_dir() and pyproject.is_file():
                members.append(Member(kind=kind, path=child, pyproject=load_toml(pyproject)))
    return members


def find_member(identifier: str, root: Path = REPO_ROOT) -> Member:
    """Look up a member by directory name or distribution name."""
    wanted = normalize_name(identifier)
    for member in discover_members(root):
        if member.name == wanted or normalize_name(member.path.name) == wanted:
            return member
    raise SystemExit(f"error: no workspace member named '{identifier}' under {root}")


def imported_roots(path: Path) -> set[str]:
    """Absolute top-level module names imported by a Python file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return set()
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots
