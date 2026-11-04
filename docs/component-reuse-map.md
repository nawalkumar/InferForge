# Component Reuse Map

This is the single source of truth for every reusable package in `packages/`: where it
came from, who uses it, and how mature it is. Update it as part of any pull request that
extracts, versions, re-wires or publishes a package.

## Status vocabulary

**Implementation status**

| Status        | Meaning                                                                 |
| ------------- | ----------------------------------------------------------------------- |
| `candidate`   | Code in an application has been flagged as reusable but not extracted yet. |
| `extracting`  | Extraction is in progress (steps 1–5 of the extraction policy).         |
| `extracted`   | Lives in `packages/`, has at least one consumer, and is at version `0.x`. |
| `stable`      | `>=1.0.0`, two or more consumers, all required package contents present. |
| `deprecated`  | Scheduled for removal or replacement. Note the replacement.             |

**Publication status**

| Status        | Meaning                                                                 |
| ------------- | ----------------------------------------------------------------------- |
| `internal`    | Used only inside the monorepo and not intended for publication yet.     |
| `review`      | The [release checklist](release-checklist.md) is in progress.           |
| `testpypi`    | Published to TestPyPI for verification.                                 |
| `published`   | Published to a public registry. Link the release.                       |

## Packages

| Package | Origin (project) | Consumers | Version | Implementation status | Test evidence | Publication status |
| ------- | ---------------- | --------- | ------- | --------------------- | ------------- | ------------------ |
| _None yet._ Packages are added here when they are extracted. See [package-policy.md](package-policy.md#extraction-policy). | | | | | | |

## Extraction candidates

Reusable capabilities found in applications that don't have a second consumer yet.
Listing them here makes the eventual extraction deliberate rather than accidental.

| Capability | Lives in | Potential consumers | Notes |
| ---------- | -------- | ------------------- | ----- |
| _None yet._ | | | |

## How to fill in "Test evidence"

Link to concrete, reproducible evidence, for example:

- the CI run or commit where `uv run pytest` passed for the package,
- the output of `scripts/verify_isolated_install.py <name>` (date and commit),
- the consumer project tests that exercise the package.
