# ADR-0005: Current iwopy Integration Contracts

- Status: Accepted
- Date: 2026-10-13
- Supersedes: [ADR-0004](0004-bound-certified-spacing-components.md), its automatic
  iwopy SLSQP movement-box and reduced-Jacobian integration assumption only.

## Context

iwopy 1.0.0 returns CSR Jacobians and constructs optimizers by backend family
and algorithm, with explicit controls and flat native options. FOXES
Optimization still used removed matrix APIs, optimizer factories, and grouped
options in YAML, pipelines, examples, notebooks, tests, and documentation.
Current iwopy SLSQP does not invoke the spacing selection hook.

## Decision

Require iwopy >=1.0.0. iwopy owns `get_jacobian()`/`calc_jacobian()` CSR assembly;
FOXES Optimization supplies component dependency masks and analytical derivative
vectors. Reserve gradient terminology for scalar functions. Native IPOPT and
`trust-constr` preserve sparse constraints; SLSQP requires dense native rows.

`foxes_opt.input.yaml` calls `iwopy.create_optimizer()` with YAML `family` and
`optimizer` selectors and flat options. Layout stages expose `optimizer_family`
and `optimizer_pars`, whose `optimizer` entry selects the algorithm. Boundary
repair can override the family, selector, and options. Stage-owned `family` and
`problem` cannot be duplicated in pass-through mappings. FOXES `algo_pars`
remain simulation settings. Construction leaves lifecycle and callbacks intact.
All maintained examples and notebooks use `iwopy.create_optimizer()` with
explicit backend, algorithm, and problem selectors; they do not directly import
or construct concrete optimizer adapters.

Keep `MinDistConstraint.active_components()` and its bound certificates. A caller
using its result must enforce the supplied coordinate bounds throughout the
solve. Current iwopy SLSQP evaluates every registered spacing component and
performs full feasibility checks.

## Compatibility And Migration

Update all maintained callers, examples, notebooks, tests, and public records
directly, without legacy matrix names, grouped options, or factory aliases.
Public pipeline backend selectors become `optimizer_family` and
`boundary_repair_optimizer_family`. Variables, component identities, result
schemas, population ordering, and FOXES engine ownership are unchanged.
The developer owns environment changes; routine commands remain `uv run --no-sync`.

## Consequences

- Sparse Jacobian storage reaches downstream workflows without a dense buffer.
- Native SLSQP rows, dependency masks, simulation populations, finite-difference
  buffers, and dense Hessian approximations remain separate memory costs.
- Full spacing rows remain part of native solves; this change does not claim
  automatic pair reduction.
- Existing layout construction/lifecycle methods and cohesive pipeline/callback
  test modules exceed the normal size guidelines. Preserve their ownership
  instead of adding unrelated lifecycle refactors or redistributing tests.

## Verification

- Derivative tests compare CSR values, selections, structural zeros, numerical
  fallback, and non-differentiable errors.
- [Optimizer configuration tests](../../tests/test_optimizer_configuration.py)
  run real YAML-selected optimizers and validate configuration boundaries.
- [Spacing tests](../../tests/test_min_dist_constraint.py) exercise full native
  SLSQP and sparse `trust-constr` constraint Jacobians.
- Pipeline and integrated pymoo layout tests preserve lifecycle and callbacks.
- Execute changed examples and notebooks; run full pytest and pre-commit gates.

## Alternatives Considered

- Compatibility shims were rejected under the forward-only policy.
- Dense downstream copies were rejected except for small test comparisons and
  backend-required native boundaries.
- Reintroducing removed iwopy solver controls downstream was rejected: optimizer
  internals belong to iwopy, and current full-component evaluation is correct.

## References

- [Architecture](../architecture.md)
- [Naming conventions](../naming-conventions.md)
- [ADR-0001](0001-forward-only-development.md)
