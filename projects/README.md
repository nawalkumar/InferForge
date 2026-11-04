# Projects

Each subdirectory is an independent application and a uv workspace member.

## Adding a project

```
projects/<name>/
├── pyproject.toml
├── README.md
├── src/<name>/
└── tests/
```

Minimal `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "<name>"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = []

[dependency-groups]
test = ["pytest>=8"]
```

## Rules

- A project may depend on packages in `packages/`, and only through a declared
  dependency plus a workspace source:

  ```toml
  [project]
  dependencies = ["inferforge-<pkg>>=0.1,<0.2"]

  [tool.uv.sources]
  inferforge-<pkg> = { workspace = true }
  ```

- A project must **not** depend on or import from another project. If two projects
  need the same code, follow the
  [extraction policy](../docs/package-policy.md#extraction-policy).
- Never copy package source into a project, and never manipulate `sys.path` to reach
  other members.
- Run `uv run python scripts/check_dependencies.py` from the repository root before
  opening a pull request.
