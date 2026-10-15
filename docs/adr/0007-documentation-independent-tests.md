# ADR-0007: Documentation-Independent Tests

- Status: Accepted
- Date: 2026-10-13

## Context

The user requested complete removal of tests that depend on documentation files
across FOXES, FOXES Optimization, and iwopy. FOXES Optimization has no such tests.

## Decision

Pytest tests must not read or depend on files under `docs/`. Keep documentation
validation in separate builds and review. Preserve the existing runtime suite
and corporate design.

## Consequences

No tests need deletion here. Future runtime tests must not require documentation
fixtures in isolated package test environments.

## Alternatives

Conditionally skipping docs-dependent tests or copying documentation into test
sandboxes retains the dependency and was rejected.

## Verification

Run the full pytest suite with documentation-file reads blocked, plus pre-commit.
