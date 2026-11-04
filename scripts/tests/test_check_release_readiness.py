from __future__ import annotations

import shutil

import pytest

import check_release_readiness as crr


def make_complete_package(repo, name="core", version="1.2.0"):
    base = repo.add_member(
        "package",
        name,
        version=version,
        extra_pyproject="""\
            readme = "README.md"
            license = "Apache-2.0"
            authors = [{ name = "InferForge" }]
            classifiers = ["Programming Language :: Python :: 3"]

            [project.urls]
            Homepage = "https://example.invalid/inferforge"
            """,
    )
    (base / "README.md").write_text(f"# inferforge-{name}\n")
    (base / "CHANGELOG.md").write_text(f"# Changelog\n\n## [{version}] - 2026-11-04\n")
    shutil.copy2(repo.root / "LICENSE", base / "LICENSE")
    (base / "docs").mkdir()
    (base / "docs" / "api.md").write_text("# API\n\n## `hello()`\n")
    (base / "tests" / "unit").mkdir(parents=True)
    (base / "tests" / "integration").mkdir()
    (base / "src" / f"inferforge_{name}" / "py.typed").write_text("")
    return base


@pytest.mark.parametrize(
    "version, ok",
    [
        ("0.1.0", True),
        ("1.2.3", True),
        ("1.0.0-rc.1", True),
        ("1.0.0rc1", True),
        ("1.0.0+build.5", True),
        ("1.0", False),
        ("01.0.0", False),
        ("v1.0.0", False),
    ],
)
def test_semver_validation(version, ok):
    assert crr.is_semver(version) is ok


def test_complete_package_passes(fake_repo, capsys):
    make_complete_package(fake_repo)
    assert crr.main(["core", "--root", str(fake_repo.root)]) == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out
    assert "Manual review still required" in out


def test_missing_files_fail(fake_repo, capsys):
    fake_repo.add_member("package", "core")
    assert crr.main(["core", "--root", str(fake_repo.root)]) == 1
    out = capsys.readouterr().out
    for expected in ("has CHANGELOG.md", "has LICENSE", "has docs/api.md", "has tests/unit"):
        assert f"FAIL  {expected}" in out


def test_version_mismatch_and_changelog_fail(fake_repo, capsys):
    base = make_complete_package(fake_repo)
    init = base / "src" / "inferforge_core" / "__init__.py"
    init.write_text(init.read_text().replace('"1.2.0"', '"1.1.0"'))
    (base / "CHANGELOG.md").write_text("# Changelog\n")
    assert crr.main(["core", "--root", str(fake_repo.root)]) == 1
    out = capsys.readouterr().out
    assert "FAIL  __version__ matches pyproject version" in out
    assert "FAIL  CHANGELOG.md has an entry for [1.2.0]" in out


def test_undocumented_public_name_fails(fake_repo, capsys):
    base = make_complete_package(fake_repo)
    (base / "docs" / "api.md").write_text("# API\n")
    assert crr.main(["core", "--root", str(fake_repo.root)]) == 1
    assert "missing: hello" in capsys.readouterr().out


def test_modified_license_fails(fake_repo, capsys):
    base = make_complete_package(fake_repo)
    (base / "LICENSE").write_text("MIT\n")
    assert crr.main(["core", "--root", str(fake_repo.root)]) == 1
    assert "FAIL  LICENSE matches the repository license" in capsys.readouterr().out


@pytest.mark.parametrize(
    "rel, content",
    [
        (".env", "X=1\n"),
        ("src/inferforge_core/settings.py", 'API_KEY = "abcdef1234567890"\n'),
        ("src/inferforge_core/aws.py", 'KEY = "AKIAABCDEFGHIJKLMNOP"\n'),
    ],
)
def test_secrets_fail(fake_repo, capsys, rel, content):
    base = make_complete_package(fake_repo)
    (base / rel).write_text(content)
    assert crr.main(["core", "--root", str(fake_repo.root)]) == 1
    assert "FAIL  no secrets or sensitive files" in capsys.readouterr().out


def test_unconstrained_internal_dependency_fails(fake_repo, capsys):
    fake_repo.add_member("package", "base")
    base = make_complete_package(fake_repo)
    pyproject = base / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text().replace("dependencies = []", 'dependencies = ["inferforge-base"]')
    )
    assert crr.main(["core", "--root", str(fake_repo.root)]) == 1
    assert "has a version constraint" in capsys.readouterr().out


def test_projects_are_not_release_candidates(fake_repo):
    fake_repo.add_member("project", "gateway")
    with pytest.raises(SystemExit, match="not a reusable package"):
        crr.main(["gateway", "--root", str(fake_repo.root)])
