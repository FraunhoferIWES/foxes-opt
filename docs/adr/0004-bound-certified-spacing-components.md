# ADR-0004: Bound-Certified Spacing Components

- Status: Accepted
- Date: 2026-10-12
- Supersedes: None

## Context

All-pair minimum-spacing components can dominate dense solver Jacobians. A pair
that is distant at the current layout can become relevant after movement, so a
current-distance cutoff is not an acceptance guarantee.

## Decision

Implement iwopy's bound-certificate hook in `MinDistConstraint`. Compute a lower
bound on separation between each pair's enforced coordinate boxes. Omit a pair
only when this exceeds nominal required spacing with a numerical reserve.
Preserve fixed turbines and the larger-diameter D-unit rule. Retain components
conservatively for unbounded movement, unsupported parameterizations, variable diameters, or state-dependent
coordinates that do not provide a single fixed layout.

Keep full pair indices, counts, names, values, derivatives, and checks unchanged.
iwopy SLSQP owns enforced movement boxes, selection refresh between solves,
fixed reduced Jacobians within a solve, and full best-feasible/final validation.

## Consequences

- Distant pairs disappear from solver rows only with a geometric certificate.
- Useful reduction requires sufficiently local enforced movement bounds.
- Full value/check cost remains; selection is not a simulation approximation.
- Newly relevant pairs are included when a caller starts a solve in a new region.
- Existing full-spacing callers keep the same component contracts.
- No dependencies or production case inputs are added.

## Alternatives Considered

- Prune by current distance: unsafe for future movements.
- Mutate component count during optimization: breaks fixed solver structure and
  public component identities.
- Use only reduced feasibility checks: could hide a certificate or mapping bug.

## References

- [Architecture](../architecture.md)
- [Naming conventions](../naming-conventions.md)
