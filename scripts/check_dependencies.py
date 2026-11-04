#!/usr/bin/env python3
"""Enforce InferForge dependency rules across all workspace members.

Rules (see docs/architecture.md):
  * packages must not depend on projects;
  * projects must not depend on other projects;
  * the internal dependency graph must be acyclic;
  * every import of another member's code must be backed by a declared dependency.

Usage:
    uv run python scripts/check_dependencies.py [--graph] [--root PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _workspace import REPO_ROOT, Member, discover_members, imported_roots


def internal_edges(members: list[Member]) -> dict[str, set[str]]:
    names = {m.name for m in members}
    return {m.name: m.requirement_names() & names - {m.name} for m in members}


def find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    cycles: list[list[str]] = []
    seen_cycles: set[frozenset[str]] = set()
    state: dict[str, int] = {}  # 0/absent = unvisited, 1 = on stack, 2 = done
    stack: list[str] = []

    def visit(node: str) -> None:
        state[node] = 1
        stack.append(node)
        for dep in sorted(graph.get(node, ())):
            if state.get(dep) == 1:
                cycle = stack[stack.index(dep) :] + [dep]
                key = frozenset(cycle)
                if key not in seen_cycles:
                    seen_cycles.add(key)
                    cycles.append(cycle)
            elif state.get(dep) is None:
                visit(dep)
        stack.pop()
        state[node] = 2

    for node in sorted(graph):
        if state.get(node) is None:
            visit(node)
    return cycles


def check(root: Path) -> tuple[list[str], dict[str, set[str]], list[Member]]:
    members = discover_members(root)
    by_name = {m.name: m for m in members}
    graph = internal_edges(members)
    errors: list[str] = []

    for member in members:
        for dep_name in sorted(graph[member.name]):
            dep = by_name[dep_name]
            if member.kind == "package" and dep.kind == "project":
                errors.append(
                    f"{member.display()} depends on {dep.display()}: "
                    "packages must never depend on applications"
                )
            elif member.kind == "project" and dep.kind == "project":
                errors.append(
                    f"{member.display()} depends on {dep.display()}: "
                    "extract the shared code into a package instead"
                )

    for cycle in find_cycles(graph):
        errors.append("dependency cycle: " + " -> ".join(cycle))

    owner_of_import: dict[str, Member] = {}
    for member in members:
        for import_root in member.import_roots():
            other = owner_of_import.get(import_root)
            if other is not None:
                errors.append(
                    f"import name '{import_root}' is provided by both "
                    f"{other.display()} and {member.display()}"
                )
            owner_of_import[import_root] = member

    for member in members:
        declared = graph[member.name]
        for py_file in member.python_files():
            for import_root in sorted(imported_roots(py_file)):
                owner = owner_of_import.get(import_root)
                if owner is None or owner.name == member.name or owner.name in declared:
                    continue
                rel = py_file.relative_to(root)
                errors.append(
                    f"{rel} imports '{import_root}' from {owner.display()} "
                    "without declaring it as a dependency"
                )

    return errors, graph, members


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    parser.add_argument("--graph", action="store_true", help="print the internal graph")
    args = parser.parse_args(argv)

    errors, graph, members = check(args.root.resolve())

    if args.graph:
        kinds = {m.name: m.kind for m in members}
        print("Internal dependency graph:")
        for name in sorted(graph):
            deps = ", ".join(sorted(graph[name])) or "(none)"
            print(f"  [{kinds[name]}] {name} -> {deps}")
        print()

    if errors:
        print(f"Dependency check FAILED with {len(errors)} problem(s):")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"Dependency check passed ({len(members)} workspace member(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
