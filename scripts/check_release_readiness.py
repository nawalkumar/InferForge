#!/usr/bin/env python3
"""Run the automatable part of docs/release-checklist.md for one package.

This script NEVER uploads anything. Passing it is necessary but not sufficient for
publication: the manual review items it prints must still be signed off.

Usage:
    uv run python scripts/check_release_readiness.py <package> [--build] [--check-pypi]
"""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from _workspace import (
    IGNORED_DIRS,
    REPO_ROOT,
    Member,
    discover_members,
    find_member,
    requirement_name,
)

SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)
# PEP 440 spellings of SemVer pre-releases (1.0.0a1, 1.0.0rc2) are also accepted.
PEP440_PRE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:(a|b|rc)\d+)?$")

SECRET_PATTERNS = {
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    "OpenAI-style key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{30,}"),
    "hard-coded credential": re.compile(
        r"""(?i)\b(?:api[_-]?key|secret|token|password|passwd)\b\s*[:=]\s*["'][^"'\s]{8,}["']"""
    ),
}
FORBIDDEN_FILES = re.compile(r"(^|/)(\.env(\..+)?|id_rsa|id_ed25519|.*\.pem|.*\.key|\.pypirc)$")
TEXT_SUFFIXES = {
    ".py",
    ".pyi",
    ".toml",
    ".md",
    ".txt",
    ".cfg",
    ".ini",
    ".json",
    ".yaml",
    ".yml",
    ".rst",
    ".sh",
    "",
}

MANUAL_REVIEW = [
    "Package naming availability on the target registry (use --check-pypi as a first pass)",
    "License compatibility of every runtime dependency with Apache-2.0",
    "Git history of the package path reviewed for secrets",
    "Internal dependency versions are already published",
    "Tests pass on lowest and highest supported Python versions",
    "Version has not been published before and the bump matches the changes",
    "Reviewer sign-off recorded in the release pull request",
]


@dataclass
class Report:
    failures: list[str] = field(default_factory=list)
    passes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def check(self, ok: bool, message: str) -> None:
        (self.passes if ok else self.failures).append(message)


def is_semver(version: str) -> bool:
    return bool(SEMVER.match(version) or PEP440_PRE.match(version))


def module_constants(init_file: Path) -> dict[str, object]:
    tree = ast.parse(init_file.read_text(encoding="utf-8"))
    constants: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        constants[target.id] = ast.literal_eval(node.value)
                    except ValueError:
                        constants[target.id] = ...
    return constants


def scan_text(label: str, text: str) -> list[str]:
    return [
        f"{label}: possible {kind}"
        for kind, pattern in SECRET_PATTERNS.items()
        if pattern.search(text)
    ]


def scan_tree(member: Member) -> list[str]:
    findings = []
    for path in sorted(member.path.rglob("*")):
        rel = path.relative_to(member.path)
        if any(part in IGNORED_DIRS for part in rel.parts) or not path.is_file():
            continue
        if FORBIDDEN_FILES.search(rel.as_posix()):
            findings.append(f"{rel}: sensitive file must not be in the package")
            continue
        if path.suffix in TEXT_SUFFIXES and path.stat().st_size < 2_000_000:
            findings.extend(scan_text(str(rel), path.read_text(encoding="utf-8", errors="ignore")))
    return findings


def check_structure(member: Member, root: Path, report: Report) -> None:
    project = member.pyproject.get("project", {})
    expected_dist = f"inferforge-{member.path.name}"
    import_name = expected_dist.replace("-", "_")

    for rel in (
        "README.md",
        "CHANGELOG.md",
        "LICENSE",
        "docs/api.md",
        "tests/unit",
        "tests/integration",
        f"src/{import_name}/py.typed",
    ):
        report.check((member.path / rel).exists(), f"has {rel}")

    report.check("build-system" in member.pyproject, "declares [build-system]")
    report.check(member.name == expected_dist, f"distribution name is '{expected_dist}'")
    for key in ("requires-python", "readme", "license", "authors", "urls", "classifiers"):
        report.check(key in project, f"[project] declares '{key}'")

    version = project.get("version", "")
    report.check(bool(version) and is_semver(version), f"version '{version}' is SemVer")

    license_file = member.path / "LICENSE"
    if license_file.exists():
        report.check(
            license_file.read_bytes() == (root / "LICENSE").read_bytes(),
            "LICENSE matches the repository license",
        )

    init_file = member.path / "src" / import_name / "__init__.py"
    if not init_file.exists():
        report.check(False, f"has src/{import_name}/__init__.py")
        return
    constants = module_constants(init_file)
    report.check(
        isinstance(constants.get("__all__"), (list, tuple)),
        "__init__.py defines a literal __all__ (the public API)",
    )
    declared_version = constants.get("__version__")
    if isinstance(declared_version, str):
        report.check(declared_version == version, "__version__ matches pyproject version")
    else:
        report.notes.append("__version__ is computed dynamically; verify it manually")

    changelog = member.path / "CHANGELOG.md"
    if changelog.exists() and version:
        report.check(
            f"[{version}]" in changelog.read_text(encoding="utf-8"),
            f"CHANGELOG.md has an entry for [{version}]",
        )

    api_doc = member.path / "docs" / "api.md"
    if api_doc.exists() and isinstance(constants.get("__all__"), (list, tuple)):
        text = api_doc.read_text(encoding="utf-8")
        missing = [name for name in constants["__all__"] if name not in text]
        report.check(
            not missing,
            "docs/api.md documents every public name"
            + (f" (missing: {', '.join(missing)})" if missing else ""),
        )


def check_dependencies(member: Member, members: list[Member], report: Report) -> None:
    internal = {m.name: m for m in members}
    for req in member.runtime_requirements:
        name = requirement_name(req)
        if name in internal:
            report.check(
                internal[name].kind == "package",
                f"runtime dependency '{name}' is a package, not a project",
            )
            report.check(
                any(op in req for op in (">", "<", "=", "~")),
                f"internal dependency '{req}' has a version constraint",
            )


def check_artifacts(member: Member, import_name: str, report: Report) -> None:
    uv = shutil.which("uv")
    if uv is None:
        report.notes.append("uv not found; skipped build artifact inspection")
        return
    with tempfile.TemporaryDirectory(prefix="inferforge-release-") as tmp:
        result = subprocess.run(
            [uv, "build", "--out-dir", tmp, str(member.path)], capture_output=True, text=True
        )
        report.check(result.returncode == 0, "package builds (sdist + wheel)")
        if result.returncode != 0:
            report.notes.append(result.stderr.strip())
            return
        wheels = list(Path(tmp).glob("*.whl"))
        sdists = list(Path(tmp).glob("*.tar.gz"))
        report.check(len(wheels) == 1 and len(sdists) == 1, "exactly one wheel and one sdist")

        for wheel in wheels:
            with zipfile.ZipFile(wheel) as zf:
                names = zf.namelist()
                stray = [
                    n
                    for n in names
                    if not n.startswith((f"{import_name}/", f"{import_name}-"))
                    and ".dist-info/" not in n
                ]
                report.check(
                    not stray,
                    "wheel contains only the package and metadata"
                    + (f" (unexpected: {', '.join(stray[:5])})" if stray else ""),
                )
                report.check(f"{import_name}/py.typed" in names, "wheel ships py.typed")
                for name in names:
                    if FORBIDDEN_FILES.search(name) or "__pycache__" in name:
                        report.check(False, f"wheel must not contain {name}")
                    elif Path(name).suffix in TEXT_SUFFIXES:
                        text = zf.read(name).decode("utf-8", errors="ignore")
                        for finding in scan_text(f"wheel:{name}", text):
                            report.check(False, finding)

        for sdist in sdists:
            with tarfile.open(sdist) as tf:
                for name in tf.getnames():
                    if FORBIDDEN_FILES.search(name) or any(
                        f"/{d}/" in f"/{name}/" for d in IGNORED_DIRS
                    ):
                        report.check(False, f"sdist must not contain {name}")


def check_pypi_name(dist_name: str, report: Report) -> None:
    url = f"https://pypi.org/pypi/{dist_name}/json"
    try:
        with urllib.request.urlopen(url, timeout=10):
            report.notes.append(f"'{dist_name}' already exists on PyPI; confirm you own it")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            report.notes.append(f"'{dist_name}' is not registered on PyPI (still verify manually)")
        else:
            report.notes.append(f"PyPI lookup for '{dist_name}' failed: HTTP {exc.code}")
    except urllib.error.URLError as exc:
        report.notes.append(f"PyPI lookup for '{dist_name}' failed: {exc.reason}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("package", help="package directory name or distribution name")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    parser.add_argument("--build", action="store_true", help="build and inspect artifacts")
    parser.add_argument(
        "--check-pypi",
        action="store_true",
        help="query PyPI (read-only) for distribution name availability",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    member = find_member(args.package, root)
    if member.kind != "package":
        raise SystemExit(f"error: {member.display()} is not a reusable package")

    report = Report()
    check_structure(member, root, report)
    check_dependencies(member, discover_members(root), report)
    secrets = scan_tree(member)
    report.check(not secrets, "no secrets or sensitive files found in the source tree")
    report.failures.extend(secrets)
    if args.build:
        check_artifacts(member, member.name.replace("-", "_"), report)
    else:
        report.notes.append("build artifacts not inspected (re-run with --build)")
    if args.check_pypi:
        check_pypi_name(member.name, report)

    print(f"Release readiness for {member.name} {member.version or '(no version)'}\n")
    for line in report.passes:
        print(f"  PASS  {line}")
    for line in report.failures:
        print(f"  FAIL  {line}")
    for line in report.notes:
        print(f"  NOTE  {line}")
    print("\nManual review still required (docs/release-checklist.md):")
    for item in MANUAL_REVIEW:
        print(f"  [ ] {item}")

    if report.failures:
        print(f"\nNOT READY: {len(report.failures)} automated check(s) failed.")
        return 1
    print("\nAutomated checks passed. Complete the manual review before publishing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
