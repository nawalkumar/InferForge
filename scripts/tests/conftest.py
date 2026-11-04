from __future__ import annotations

import shutil
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))


class FakeRepo:
    """Builds a throwaway InferForge-shaped repository for script tests."""

    def __init__(self, root: Path) -> None:
        self.root = root
        shutil.copy2(REPO_ROOT / "LICENSE", root / "LICENSE")
        (root / "packages").mkdir()
        (root / "projects").mkdir()

    def add_member(
        self,
        kind: str,
        name: str,
        deps: list[str] | None = None,
        sources: dict[str, str] | None = None,
        version: str = "0.1.0",
        extra_pyproject: str = "",
    ) -> Path:
        base = self.root / ("packages" if kind == "package" else "projects") / name
        dist = f"inferforge-{name}" if kind == "package" else name
        import_name = dist.replace("-", "_")
        pkg_dir = base / "src" / import_name
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "__init__.py").write_text(
            f'__all__ = ["hello"]\n__version__ = "{version}"\n\n\n'
            f'def hello():\n    return "{name}"\n'
        )
        for rel, content in (sources or {}).items():
            path = base / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(content))
        dep_list = ", ".join(f'"{d}"' for d in deps or [])
        (base / "pyproject.toml").write_text(
            textwrap.dedent(
                f"""\
                [build-system]
                requires = ["hatchling"]
                build-backend = "hatchling.build"

                [project]
                name = "{dist}"
                version = "{version}"
                requires-python = ">=3.10"
                dependencies = [{dep_list}]
                """
            )
            + textwrap.dedent(extra_pyproject)
        )
        return base


@pytest.fixture
def fake_repo(tmp_path: Path) -> FakeRepo:
    return FakeRepo(tmp_path)
