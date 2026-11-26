# ADR-0011: Production Documentation And CI

- Status: Accepted
- Date: 2026-11-26
- Supersedes: None; extends [ADR-0010](0010-foxes-documentation-alignment.md)

## Context

The user requested production-ready FOXES Optimization documentation and
GitHub/GitLab pipelines as close to FOXES as possible, explicitly allowing
Sphinx warnings. The documentation executes both maintained notebooks, but an
audit found obsolete FOXES URLs and a comparison against an unpublished tag.
GitHub Actions referenced older action versions than FOXES; GitLab CI already
matched exactly.

## Decision

Align the three GitHub workflows with FOXES's published action versions and
explicit documentation notebook execution. Retain FOXES Optimization's own
PyPI/TestPyPI URLs and the existing branch/tag triggers, Python 3.10--3.14 test
and notebook matrices, job dependencies, version/tag check, artifact names,
trusted-publishing permissions, and signed GitHub release flow.

Keep GitLab CI identical to FOXES: Python 3.14, the uv container, the existing
runner tag, pre-commit/test/notebook stages, and cache policy. Do not introduce
additional release jobs or a different CI architecture.

Documentation deploys from `main` after pre-commit, installs the `doc` extra,
and builds with `nb_execution_mode=auto`. Missing notebook outputs trigger
execution; execution errors fail the build under `nb_execution_raise_on_error`.
Sphinx warnings remain visible and non-fatal; no warning suppression or
warnings-as-errors deployment gate is added.

Use the live FOXES documentation URLs. Until the current release tag exists,
label its changelog comparison as development and compare the prior version
with `dev`. Once published, update it to the version-specific comparison.
Validate external links and generated local links separately from runtime
pytest, following the established test-input boundary.

## Compatibility And Migration

No Python, YAML, optimization, output, or dependency contract changes.
The existing GitHub environments and trusted-publisher settings remain required;
no credentials, publication, tags, or external settings are changed locally.
FOXES remains read-only. No compatibility branch or legacy CI path is added.

## Consequences

- Both projects share CI maintenance patterns and published action versions.
- Documentation includes executed notebook output without blocking on accepted
  API/docstring warnings.
- External link checks continue to detect actual HTTP errors rather than
  suppressing them to accommodate a not-yet-published version.
- Hosted execution, protected environments, runner availability, and trusted
  publishers still require maintainer-controlled configuration.

## Verification

- Compare parsed GitHub workflows with FOXES after translating only the package
  project URLs, and compare GitLab CI byte-for-byte.
- Check action references, YAML structure, shell syntax, and the release
  version/tag guard's matching and mismatching cases.
- Build HTML with notebook execution enabled and verify both notebooks'
  executed cells and output assets.
- Run external link checking and validate local files and fragments in HTML.
- Run package tests and repository-wide pre-commit; confirm FOXES is unchanged.

## Alternatives Considered

- Treat Sphinx warnings as errors: rejected by explicit user instruction.
- Disable notebook execution for deployment: rejected because published
  examples must include their results and execution failures must be visible.
- Add different CI gates or GitLab publication jobs: rejected in favour of
  following FOXES closely without expanding the pipeline architecture.
- Publish a release tag to repair the comparison URL: rejected because release
  publication was not requested.
