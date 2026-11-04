# Packages

Each subdirectory is a reusable library that can be installed on its own and is a uv
workspace member.

**This directory is intentionally empty until a capability is extracted.** A package is
created only when a second application genuinely needs functionality that has already
been implemented and validated in its originating application. See the
[extraction policy](../docs/package-policy.md#extraction-policy).

## Conventions

| Item              | Convention                                         |
| ----------------- | -------------------------------------------------- |
| Directory         | `packages/<name>/`                                 |
| Distribution name | `inferforge-<name>`                                |
| Import name       | `inferforge_<name>` (in `src/` layout)             |
| Version           | SemVer, independent per package                    |
| License           | Apache-2.0 (copy of the repository `LICENSE`)      |

The full list of required contents is in
[docs/package-policy.md](../docs/package-policy.md#required-package-contents).

## Checks for every package

```bash
uv run python scripts/check_dependencies.py
uv run python scripts/verify_isolated_install.py <name>
uv run python scripts/check_release_readiness.py <name> --build
```

When you add a package, record it in
[docs/component-reuse-map.md](../docs/component-reuse-map.md).
