# InferForge Repository Architecture

InferForge is a multi-project monorepo. It contains independent applications and the
reusable libraries extracted from them. Each library is built so it can be installed and
used without the monorepo.

## Layout

| Directory   | Contents                                                              |
| ----------- | --------------------------------------------------------------------- |
| `projects/` | Independent applications. Each one is a uv workspace member with its own `pyproject.toml`. |
| `packages/` | Reusable libraries that can be installed on their own. Each one is a uv workspace member. |
| `docs/`     | Architecture, policies, roadmap and integration documentation.        |
| `scripts/`  | Repository-level developer utilities and their tests.                 |

The repository root is a *virtual* uv workspace. It has no `[project]` table and is
never built or published. Its only jobs are to tie the members together for local
development and to hold the shared developer tooling (pytest, ruff).

## Dependency direction

Dependencies flow in one direction only:

```
projects/*  ──►  packages/*  ──►  third-party libraries
```

| From      | To        | Allowed | Notes                                                                 |
| --------- | --------- | ------- | --------------------------------------------------------------------- |
| project   | package   | Yes     | Must be declared in the project's `pyproject.toml`.                   |
| package   | package   | Yes     | Must be declared, and the package graph must be acyclic.              |
| package   | project   | **No**  | Packages must never know about applications.                         |
| project   | project   | **No**  | Shared code is extracted into a package (see the extraction policy). |

`scripts/check_dependencies.py` enforces these rules on both declared dependencies and
actual `import` statements, so a member can't depend on another member's source tree
without declaring it.

## Consuming packages

Applications use packages through declared dependencies. They never copy source files
or add paths to `sys.path`.

```toml
# projects/<app>/pyproject.toml
[project]
dependencies = ["inferforge-example>=1.2,<2"]

[tool.uv.sources]
inferforge-example = { workspace = true }
```

- **Inside the monorepo**, `{ workspace = true }` makes uv install the package from
  `packages/` as an editable install. Changes are picked up immediately.
- **Outside the monorepo**, the same requirement (`inferforge-example>=1.2,<2`) resolves
  from a package index or a built wheel. `[tool.uv.sources]` is ignored when the
  package is installed by other tools, so the published metadata stays clean.

The version constraint in `dependencies` is the real contract, and it must be
satisfiable by a released version of the package. Workspace sourcing doesn't change it.

## Naming conventions

| Thing                     | Convention              | Example               |
| ------------------------- | ----------------------- | --------------------- |
| Package directory         | `packages/<name>/`      | `packages/tracing/`   |
| Distribution name (PyPI)  | `inferforge-<name>`     | `inferforge-tracing`  |
| Import name               | `inferforge_<name>`     | `inferforge_tracing`  |
| Project directory         | `projects/<name>/`      | `projects/gateway/`   |

Package source uses the `src/` layout (`packages/<name>/src/inferforge_<name>/`). This
way the tests run against the installed package and not against the working tree.

## Validation layers

1. **Static dependency check** (`scripts/check_dependencies.py`). Declared graph is
   acyclic, direction rules hold, and no undeclared cross-member imports exist.
2. **Monorepo tests.** `uv run pytest` inside each member, with workspace packages
   installed as editable installs.
3. **Isolated install check** (`scripts/verify_isolated_install.py`). Builds the package
   and its internal dependencies as wheels, installs them into a fresh virtual
   environment outside the repository, and runs the package's tests from a temporary
   directory.
4. **Release readiness check** (`scripts/check_release_readiness.py`). Runs the automated
   part of the [release checklist](release-checklist.md). It never uploads anything.

See [package-policy.md](package-policy.md) for package requirements and the extraction
process, and [component-reuse-map.md](component-reuse-map.md) for the current inventory.
