"""The end-to-end tests build real wheels and a real venv, so they need uv and index access.

Set INFERFORGE_SKIP_SLOW=1 to skip them.
"""

from __future__ import annotations

import os
import shutil

import pytest

import verify_isolated_install as vii
from _workspace import discover_members, find_member

slow = pytest.mark.skipif(
    shutil.which("uv") is None or os.environ.get("INFERFORGE_SKIP_SLOW") == "1",
    reason="requires uv and package index access",
)


def add_tested_package(repo, name, deps=None, test_body="assert True"):
    import_name = f"inferforge_{name}"
    return repo.add_member(
        "package",
        name,
        deps=deps,
        sources={
            "tests/unit/test_basic.py": (
                f"import {import_name}\n\n\ndef test_it():\n    {test_body}\n"
            )
        },
    )


@slow
def test_package_with_internal_dependency_installs_and_passes(fake_repo):
    add_tested_package(fake_repo, "core")
    add_tested_package(
        fake_repo,
        "tracing",
        deps=["inferforge-core>=0.1,<0.2"],
        test_body="import inferforge_core; assert inferforge_core.hello() == 'core'",
    )
    assert vii.main(["tracing", "--root", str(fake_repo.root)]) == 0


@slow
def test_failing_tests_are_reported(fake_repo):
    add_tested_package(fake_repo, "core", test_body="assert False")
    assert vii.main(["core", "--root", str(fake_repo.root)]) == 1


@slow
def test_undeclared_internal_dependency_fails_outside_monorepo(fake_repo):
    add_tested_package(fake_repo, "core")
    add_tested_package(fake_repo, "tracing", test_body="import inferforge_core")
    assert vii.main(["tracing", "--root", str(fake_repo.root)]) == 1


def test_internal_closure_orders_dependencies_first(fake_repo):
    fake_repo.add_member("package", "core")
    fake_repo.add_member("package", "metrics", deps=["inferforge-core>=0.1"])
    fake_repo.add_member(
        "package", "tracing", deps=["inferforge-metrics>=0.1", "inferforge-core>=0.1"]
    )
    members = discover_members(fake_repo.root)
    closure = vii.internal_closure(find_member("tracing", fake_repo.root), members)
    assert [m.name for m in closure] == [
        "inferforge-core",
        "inferforge-metrics",
        "inferforge-tracing",
    ]


def test_projects_are_rejected(fake_repo):
    fake_repo.add_member("project", "gateway")
    with pytest.raises(SystemExit, match="not a reusable package"):
        vii.main(["gateway", "--root", str(fake_repo.root)])
