# Package Release Checklist

Complete this checklist for **every** release of a package to PyPI or another public
registry. Copy it into the release pull request description and tick each item.
Publication is always manual. No tooling in this repository uploads packages.

Automated checks help but don't replace the review:

```bash
uv run python scripts/check_dependencies.py
uv run python scripts/check_release_readiness.py <name>
uv run python scripts/verify_isolated_install.py <name>
```

## Package: `inferforge-<name>` — version `X.Y.Z`

### 1. Package naming availability
- [ ] The distribution name is not already taken on the target registry
      (check `https://pypi.org/project/<dist-name>/` and the TestPyPI equivalent).
- [ ] The name doesn't infringe on or confuse with an existing project or trademark.
- [ ] The import name (`inferforge_<name>`) doesn't clash with a popular existing module.

### 2. Licensing
- [ ] `LICENSE` in the package matches the repository license (Apache-2.0).
- [ ] `license` in `pyproject.toml` is set and matches `LICENSE`.
- [ ] All runtime dependencies have licenses compatible with Apache-2.0.
- [ ] Any vendored or adapted third-party code is attributed, and its license is included.

### 3. Secrets and sensitive information
- [ ] No credentials, tokens, private keys or `.env` files in the source tree or in the
      built artifacts (`check_release_readiness.py` scans for common patterns).
- [ ] No internal hostnames, personal data, or proprietary model weights or datasets.
- [ ] Git history for the package path has been reviewed for accidentally committed secrets.

### 4. Build artifacts
- [ ] `uv build packages/<name>` produces exactly one sdist and one wheel.
- [ ] The wheel contains only `inferforge_<name>/` (plus `py.typed`) and its metadata.
      No tests, caches or files from outside the package.
- [ ] The sdist contains the source, tests, README, CHANGELOG and LICENSE, and nothing else.
- [ ] Artifacts are built from a clean checkout at the tagged commit.

### 5. Dependency metadata
- [ ] Every import in the package is covered by a declared dependency.
- [ ] Internal dependencies point at versions that are **already published**.
- [ ] Version ranges are justified (no unnecessary upper bounds or exact pins).
- [ ] `requires-python` is correct and covered by tests.

### 6. Tests
- [ ] Unit and integration tests pass in monorepo mode.
- [ ] `verify_isolated_install.py <name>` passes, which shows the package works
      outside the monorepo.
- [ ] Tests pass on the lowest and highest supported Python versions.

### 7. Documentation
- [ ] README covers purpose, installation, quick start and compatibility policy.
- [ ] `docs/api.md` documents every name in `__all__`.
- [ ] `CHANGELOG.md` has an entry for this version with the release date.

### 8. Release version
- [ ] The version follows SemVer and the bump matches the changes (see
      [package-policy.md](package-policy.md#semantic-versioning)).
- [ ] `pyproject.toml` `version` and `__version__` agree.
- [ ] The version has not been published before (registries reject reuse).
- [ ] The git tag `inferforge-<name>-vX.Y.Z` will be created on the release commit.

### Sign-off
- [ ] Reviewer: ____________________  Date: ____________
- [ ] Optional: published to TestPyPI first, then installed and smoke-tested from there.
- [ ] Manual publication command run by a maintainer (for example `uv publish`).
- [ ] [component-reuse-map.md](component-reuse-map.md) updated with the publication status.
