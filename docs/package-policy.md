# Reusable Package Policy

This document defines what a package in `packages/` must provide, how packages are
versioned, and when and how code is extracted into a package.

## Extraction policy

Packages are **extracted, not speculated**. Do not create empty or placeholder packages.

1. Implement and validate the capability inside the application that first needs it.
2. Extract it only when a *second* application genuinely needs the same capability.
3. Run the extraction as follows:
   1. **Identify generic functionality.** Find the parts that don't depend on the
      originating application's domain, configuration or data.
   2. **Separate application-specific logic.** Anything app-specific stays in the
      application, usually as configuration or as an adapter built on the package API.
   3. **Extract a stable public API.** Move the generic code into
      `packages/<name>/src/inferforge_<name>/` and export the public surface explicitly
      (see "Public API" below).
   4. **Add package tests and documentation.** Port the relevant tests from the
      originating application, add unit and integration tests for the public API, and
      write the README and API docs.
   5. **Integrate into the consumer application(s).** Replace the original code in the
      originating application *and* wire up the new consumer, both through a declared
      dependency.
   6. **Verify that no consumer imports from the original application's source.** Run
      `scripts/check_dependencies.py` and `scripts/verify_isolated_install.py <name>`.
4. Record the package in [component-reuse-map.md](component-reuse-map.md).

## Required package contents

A package is not considered *stable* until every item below is present.
`scripts/check_release_readiness.py` checks the items that can be automated.

```
packages/<name>/
├── pyproject.toml            # independent build config ([build-system] + [project])
├── README.md                 # purpose, installation, quick start, compatibility
├── CHANGELOG.md              # Keep a Changelog format
├── LICENSE                   # copy of the repository license (Apache-2.0)
├── docs/api.md               # public API reference
├── src/inferforge_<name>/
│   ├── __init__.py           # defines __all__ and __version__
│   └── py.typed              # ships type information
└── tests/
    ├── unit/
    └── integration/
```

### `pyproject.toml` requirements

- `[build-system]` declared (default: `hatchling`), so the package builds with any
  PEP 517 frontend.
- `[project]` with `name = "inferforge-<name>"`, a SemVer `version`,
  `requires-python`, `license = "Apache-2.0"`, `readme`, `authors`, `classifiers` and
  `[project.urls]`.
- **Explicit dependencies.** Every third-party and internal import has a matching entry
  in `dependencies` (or an optional-dependency extra). Do not rely on transitive
  dependencies.
- Internal dependencies use a bounded range (for example `inferforge-core>=1.2,<2`) and
  `[tool.uv.sources] ... = { workspace = true }` for local development.
- Test-only dependencies go in `[dependency-groups]`, not in `dependencies`.

### Public API

- The public API is exactly what `inferforge_<name>/__init__.py` lists in `__all__`.
- Modules and names that start with `_` are private and may change in any release.
- Each public name is documented in `docs/api.md`.

## Semantic versioning

Packages follow [SemVer 2.0.0](https://semver.org/). Each package is versioned
independently of the others and of the applications.

| Change                                                    | Bump  |
| --------------------------------------------------------- | ----- |
| Removing or incompatibly changing a public name            | MAJOR |
| Raising the minimum supported Python or a dependency floor so existing users break | MAJOR |
| Adding a public name, parameter with a default, or extra   | MINOR |
| Bug fixes, docs, internal refactors                        | PATCH |

- `0.y.z` means initial development. The API may change in any MINOR release, and
  consumers should pin `>=0.y,<0.(y+1)`.
- A package reaches `1.0.0` once it has at least two consumers, a reviewed public API
  and complete documentation.
- Every version bump comes with a `CHANGELOG.md` entry.

## Compatibility policy

Each package README states its compatibility policy. Default:

- **Python.** Supports every CPython version that is not end-of-life at release time,
  down to the declared `requires-python` floor. Dropping a Python version is a MINOR
  bump during `0.x` and a MAJOR bump from `1.0` onwards.
- **Deprecation.** From `1.0` onwards, public names are deprecated with a
  `DeprecationWarning` for at least one MINOR release before they are removed in the
  next MAJOR release.
- **Dependencies.** Third-party dependency ranges are as wide as the tests justify. An
  upper bound is added only when there is a known incompatibility.
- **Internal consumers.** Projects in this repository pin package ranges the same way
  external users would. No consumer may depend on unreleased behaviour of a package
  without bumping the package version.

## Testing

- `tests/unit/` covers the public API in isolation.
- `tests/integration/` covers realistic use, including interaction with internal
  dependencies.
- Tests import only from `inferforge_<name>` and never from `projects/` or another
  package's private modules.
- The tests must pass in both monorepo mode (`uv run pytest`) and isolated mode
  (`scripts/verify_isolated_install.py <name>`).

## Publication

Nothing is published automatically. There is no CI job or script that uploads to PyPI.
Publication is a deliberate, manual step that is taken only after the
[release checklist](release-checklist.md) has been completed and reviewed.
