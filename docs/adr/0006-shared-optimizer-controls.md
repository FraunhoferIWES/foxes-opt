# ADR-0006: Shared iwopy Optimizer Controls

## Status

Accepted. Supplements ADR-0005's current iwopy factory and sparse contracts.

## Context

iwopy standardizes meaningful optimizer controls without identifying search
populations, pending batches, iterations, and evaluation counters. FOXES
Optimization forwards these settings through YAML and pipeline mappings and
publishes runnable Python and notebook callers.

## Decision

- Use canonical `max_iterations` and `max_evaluations` only where supported;
  do not restore native aliases or approximate unsupported counters.
- Use pymoo `iterations` / `evaluations` for pure limits. Preserve native
  convergence settings and document generation-boundary evaluation overshoot.
- Keep absolute `pop_size`, pending-work `batch_size`, and SciPy DE's
  `population_multiplier` distinct. Use iwopy's public explanation/support table
  rather than owning a second optimizer policy.
- Migrate all maintained Python, YAML, and notebook calls in the same change.
  Existing example CLI flags retain their names and map to canonical controls.
- Preserve developer-owned environments; routine validation uses
  `uv run --no-sync` without dependency synchronization.

## Consequences

Settings transfer by meaning, not by equal numeric values. SLSQP and IPOPT have
iteration limits but no portable evaluation cap. FOXES evaluations, derivatives,
and finalization are not universally intercepted or counted as native objective
calls. Configuration regressions cover current controls and reject old names.
