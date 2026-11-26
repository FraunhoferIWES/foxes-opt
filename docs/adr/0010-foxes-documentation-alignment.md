# ADR-0010: FOXES Documentation Alignment

- Status: Accepted
- Date: 2026-11-26
- Supersedes: None; extends [ADR-0002](0002-corporate-design.md)

## Context

FOXES and FOXES Optimization are Fraunhofer projects. The user confirmed
corporate design and requested that FOXES Optimization documentation match
FOXES as closely as possible, especially its appearance, without modifying
FOXES.

FOXES uses Sphinx Immaterial with an IWES token adapter and CSS overrides.
FOXES Optimization previously used the theme's stock teal palette and numpydoc,
with no local styling or template adapter.

## Decision

Retain the corporate policy in ADR-0002. Reuse the FOXES documentation token
adapter, CSS overrides, and SVG-backed domain-index template as maintained
FOXES Optimization sources under `docs/source/_static/` and
`docs/source/_templates/`. Load `iwes-tokens.css` before `iwes.css`.

Use the same light IWES palette, typography with existing font-size floors,
spacing, flat controls, tables, notices, focus indicators, responsive layout,
and reduced-motion handling. Preserve FOXES Optimization's project identity,
URLs, package API, and optimization-specific content.
Wrap long API type names locally to prevent horizontal page overflow at 320px;
this is the sole addition to the reused FOXES stylesheet.

Align API rendering with FOXES through Sphinx Napoleon, AutoAPI imported and
inherited members, and separated constructor sections. Replace numpydoc rather
than supporting both renderers. Use current intersphinx inventories and fail
explicitly when executed documentation notebooks fail.

The authoritative token source remains
`docs/fraunhofer-design/design-tokens.json`. Review adapter parity outside
pytest under [ADR-0008](0008-test-file-dependency-boundary.md).
Tokens marked `needs-review`, including the monospaced stack, some font/layout
dimensions, hover/active colours, and feedback mappings, retain that status;
copying their existing mappings from FOXES does not approve them.

## Compatibility And Migration

This changes documentation presentation and rendering only. Python, YAML,
optimization variables, objective/constraint components, results, callbacks,
pipelines, and runtime FOXES/iwopy integrations are unchanged. No legacy theme,
dual renderer, new runtime dependency, or cross-checkout build dependency is
introduced. FOXES remains read-only.

## Consequences

- Both documentation sites share their corporate visual language.
- FOXES Optimization owns its local assets; future visual updates must be
  checked against FOXES and the authoritative token document.
- Sphinx's bundled Napoleon replaces the separate numpydoc dependency.
- Documentation assets and templates are tracked; generated AutoAPI and build
  output remain ignored.
- The developer continues to own environment setup; rendering checks use the
  prepared `doc` extra through `uv run --no-sync`.

## Verification

- Compare the reused CSS, SVG, and template with FOXES and resolve every token
  reference when checking adapter parity.
- Build Sphinx HTML with saved notebook output for appearance checks.
- Inspect landing, installation, generated API, and notebook pages at 320,
  768, and 1280px, including keyboard focus, search, and reduced motion.
- Run repository-wide pre-commit after the final edit.
- Confirm that FOXES has no worktree changes.

## Alternatives Considered

- Free design: rejected because the user confirmed both projects follow the
  Fraunhofer corporate design.
- Keep the stock teal theme or numpydoc: rejected because it retains avoidable
  visual and API-rendering differences.
- Load assets from the adjacent FOXES checkout: rejected because independent
  documentation builds must not require that checkout.
- Introduce a browser application or shared plotting adapter: rejected as
  unrelated to Sphinx documentation alignment.

## References

- [Architecture](../architecture.md#cross-cutting-decisions)
- [Development documentation workflow](../development.md#documentation)
- [Fraunhofer IWES UI guidelines](../fraunhofer-design/ui-guidelines.md)
- [Fraunhofer IWES design tokens](../fraunhofer-design/design-tokens.json)
