from __future__ import annotations

import check_dependencies


def errors_for(repo) -> list[str]:
    errors, _, _ = check_dependencies.check(repo.root)
    return errors


def test_empty_repository_passes(fake_repo):
    assert errors_for(fake_repo) == []


def test_project_consuming_declared_package_passes(fake_repo):
    fake_repo.add_member("package", "core")
    fake_repo.add_member(
        "project",
        "gateway",
        deps=["inferforge-core>=0.1,<0.2"],
        sources={"src/gateway/app.py": "from inferforge_core import hello\n"},
    )
    assert errors_for(fake_repo) == []


def test_package_depending_on_project_is_rejected(fake_repo):
    fake_repo.add_member("project", "gateway")
    fake_repo.add_member("package", "core", deps=["gateway"])
    errors = errors_for(fake_repo)
    assert any("packages must never depend on applications" in e for e in errors)


def test_project_depending_on_project_is_rejected(fake_repo):
    fake_repo.add_member("project", "gateway")
    fake_repo.add_member("project", "dashboard", deps=["gateway"])
    errors = errors_for(fake_repo)
    assert any("extract the shared code into a package" in e for e in errors)


def test_cycle_between_packages_is_rejected(fake_repo):
    fake_repo.add_member("package", "a", deps=["inferforge-b"])
    fake_repo.add_member("package", "b", deps=["inferforge-c"])
    fake_repo.add_member("package", "c", deps=["inferforge-a"])
    cycles = [e for e in errors_for(fake_repo) if e.startswith("dependency cycle")]
    assert len(cycles) == 1
    assert "inferforge-a" in cycles[0] and "inferforge-c" in cycles[0]


def test_cycle_through_dependency_group_is_rejected(fake_repo):
    fake_repo.add_member("package", "a", deps=["inferforge-b"])
    fake_repo.add_member(
        "package", "b", extra_pyproject='\n[dependency-groups]\ntest = ["inferforge-a"]\n'
    )
    assert any(e.startswith("dependency cycle") for e in errors_for(fake_repo))


def test_undeclared_import_of_project_source_is_rejected(fake_repo):
    fake_repo.add_member("project", "gateway")
    fake_repo.add_member(
        "project",
        "dashboard",
        sources={"src/dashboard/views.py": "from gateway import hello\n"},
    )
    errors = errors_for(fake_repo)
    assert any("imports 'gateway'" in e and "without declaring" in e for e in errors)


def test_undeclared_import_of_package_is_rejected(fake_repo):
    fake_repo.add_member("package", "core")
    fake_repo.add_member(
        "project", "gateway", sources={"tests/test_app.py": "import inferforge_core\n"}
    )
    assert any("imports 'inferforge_core'" in e for e in errors_for(fake_repo))


def test_requirement_names_are_normalised(fake_repo):
    fake_repo.add_member("package", "core")
    fake_repo.add_member(
        "project",
        "gateway",
        deps=["InferForge_Core[extra]>=0.1 ; python_version >= '3.10'"],
        sources={"src/gateway/app.py": "import inferforge_core\n"},
    )
    assert errors_for(fake_repo) == []


def test_find_cycles_reports_each_cycle_once():
    graph = {"a": {"b"}, "b": {"a"}, "c": {"c"}}
    cycles = check_dependencies.find_cycles(graph)
    assert sorted(len(c) for c in cycles) == [2, 3]


def test_main_exit_codes(fake_repo, capsys):
    assert check_dependencies.main(["--root", str(fake_repo.root)]) == 0
    fake_repo.add_member("project", "gateway")
    fake_repo.add_member("package", "core", deps=["gateway"])
    assert check_dependencies.main(["--root", str(fake_repo.root), "--graph"]) == 1
    assert "FAILED" in capsys.readouterr().out
