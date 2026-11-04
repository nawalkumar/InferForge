# InferForge

InferForge is a modular multi-project repository. It holds independent applications and
the reusable Python packages extracted from them. Each package is built to be installed
and used on its own, outside this repository.

## Repository layout

```
projects/   Independent applications (uv workspace members)
packages/   Reusable, installable libraries (uv workspace members)
docs/       Architecture, policies, roadmap and integration documentation
scripts/    Repository-level developer utilities (and their tests)
```

Dependencies only flow `projects → packages → third-party`. Packages never depend on
applications, applications never depend on each other, and the package graph is
acyclic. See [docs/architecture.md](docs/architecture.md).

## Getting started

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10+.

```bash
uv sync                                   # install every workspace member (editable) + dev tools
uv run pytest                             # repository tooling tests
uv run python scripts/check_dependencies.py --graph
```

Each workspace member runs its own tests from its directory, for example
`cd packages/<name> && uv run pytest`.

## Developer scripts

| Script | Purpose |
| ------ | ------- |
| `scripts/check_dependencies.py` | Enforces dependency direction, acyclicity, and that every cross-member import is declared. |
| `scripts/verify_isolated_install.py <pkg>` | Builds wheels, installs them into a fresh venv outside the repo, and runs the package tests there. |
| `scripts/check_release_readiness.py <pkg> [--build] [--check-pypi]` | Runs the automated part of the release checklist. Never uploads. |

## Reusable packages

Packages are **extracted** from applications once a second application needs the same
capability. They are never created speculatively. Read these before adding or changing
a package:

- [docs/package-policy.md](docs/package-policy.md): required contents, SemVer,
  compatibility and the extraction process.
- [docs/release-checklist.md](docs/release-checklist.md): mandatory manual review before
  any public release. Nothing is published automatically.
- [docs/component-reuse-map.md](docs/component-reuse-map.md): inventory of packages,
  their origins, consumers, versions and publication status.

## License

[Apache-2.0](LICENSE). Each package ships a copy of the same license.
