#!/usr/bin/env python3
"""Prove that a package works when installed outside the monorepo.

Steps:
  1. Build wheels for the package and its internal runtime dependencies.
  2. Create a fresh virtual environment in a temporary directory outside the repository.
  3. Install the wheels by explicit path (third-party dependencies come from the index).
  4. Check that the package is imported from that environment, not from the repo.
  5. Copy the package's tests to the temporary directory and run them there.

Nothing is uploaded anywhere.

Usage:
    uv run python scripts/verify_isolated_install.py <package> [--python 3.12] [--keep]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from _workspace import REPO_ROOT, Member, discover_members, find_member, requirement_name


def run(cmd: list[str], **kwargs) -> None:
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def internal_closure(target: Member, members: list[Member]) -> list[Member]:
    """The target plus every internal member it needs at runtime, dependencies first."""
    by_name = {m.name: m for m in members}
    ordered: list[Member] = []
    done: set[str] = set()
    visiting: set[str] = set()

    def visit(member: Member) -> None:
        if member.name in done:
            return
        if member.name in visiting:
            raise SystemExit(f"error: dependency cycle through '{member.name}'")
        visiting.add(member.name)
        for dep_name in sorted(member.requirement_names(runtime_only=True)):
            if dep_name in by_name:
                visit(by_name[dep_name])
        visiting.discard(member.name)
        done.add(member.name)
        ordered.append(member)

    visit(target)
    return ordered


def test_requirements(member: Member, internal_names: set[str]) -> list[str]:
    reqs: list[str] = []
    for group in ("test", "dev"):
        reqs.extend(member.group_requirements.get(group, []))
    external = [r for r in reqs if requirement_name(r) not in internal_names]
    if not any(requirement_name(r) == "pytest" for r in external):
        external.append("pytest")
    return external


def clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for var in ("PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "PYTHONHOME"):
        env.pop(var, None)
    return env


def verify(target_id: str, root: Path, python: str | None, keep: bool) -> int:
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("error: 'uv' is required (https://docs.astral.sh/uv/)")

    members = discover_members(root)
    target = find_member(target_id, root)
    if target.kind != "package":
        raise SystemExit(f"error: {target.display()} is not a reusable package")

    tests_dir = target.path / "tests"
    if not tests_dir.is_dir():
        raise SystemExit(f"error: {target.display()} has no tests/ directory")

    import_roots = sorted(target.import_roots())
    if not import_roots:
        raise SystemExit(f"error: could not determine the import name of {target.display()}")

    closure = internal_closure(target, members)
    internal_names = {m.name for m in members}
    workdir = Path(tempfile.mkdtemp(prefix=f"inferforge-isolated-{target.path.name}-"))
    if workdir.resolve().is_relative_to(root.resolve()):
        raise SystemExit("error: temporary directory must be outside the repository")

    env = clean_env()
    try:
        wheels_dir = workdir / "wheels"
        for member in closure:
            run([uv, "build", "--wheel", "--out-dir", str(wheels_dir), str(member.path)], env=env)
        wheels = sorted(wheels_dir.glob("*.whl"))

        venv = workdir / "venv"
        venv_cmd = [uv, "venv", str(venv)]
        if python:
            venv_cmd += ["--python", python]
        run(venv_cmd, env=env, cwd=workdir)
        venv_python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

        install = [uv, "pip", "install", "--python", str(venv_python)]
        run(install + [str(w) for w in wheels], env=env, cwd=workdir)
        extra = test_requirements(target, internal_names)
        run(install + extra, env=env, cwd=workdir)

        probe = (
            "import importlib, pathlib, sys\n"
            f"repo = pathlib.Path({str(root.resolve())!r})\n"
            f"for name in {import_roots!r}:\n"
            "    mod = importlib.import_module(name)\n"
            "    origin = pathlib.Path(mod.__file__).resolve()\n"
            "    if origin.is_relative_to(repo):\n"
            "        sys.exit(f'{name} was imported from the repository: {origin}')\n"
            "    print(f'{name} -> {origin}')\n"
        )
        run([str(venv_python), "-c", probe], env=env, cwd=workdir)

        shutil.copytree(tests_dir, workdir / "tests")
        shutil.copy2(target.path / "pyproject.toml", workdir / "pyproject.toml")
        run(
            [
                str(venv_python),
                "-m",
                "pytest",
                "tests",
                "-p",
                "no:cacheprovider",
                "--rootdir",
                str(workdir),
            ],
            env=env,
            cwd=workdir,
        )
    except subprocess.CalledProcessError as exc:
        print(f"\nIsolated install verification FAILED for {target.name}: {exc}")
        return 1
    finally:
        if keep:
            print(f"Kept working directory: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    print(
        f"\nIsolated install verification passed for {target.name} "
        f"(with {len(closure) - 1} internal dependency/dependencies)."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("package", help="package directory name or distribution name")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    parser.add_argument("--python", help="Python version or interpreter for the venv")
    parser.add_argument("--keep", action="store_true", help="keep the temporary directory")
    args = parser.parse_args(argv)
    return verify(args.package, args.root.resolve(), args.python, args.keep)


if __name__ == "__main__":
    sys.exit(main())
