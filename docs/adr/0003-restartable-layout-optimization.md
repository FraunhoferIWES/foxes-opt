# ADR-0003: Restartable Layout Optimization

- Status: Accepted
- Date: 2026-10-05
- Supersedes: None

## Context

Long-running layout optimizations persist intermediate layout CSV files, but
`LayoutPipeline` could only begin from results produced in the current process.
Restarting an optimizer stage manually risked rerunning earlier stages,
overwriting numbered layout snapshots, truncating optimization history, and
losing the pipeline table. Layout feasibility also combines distinct constraint
families, such as farm boundaries and pairwise minimum distances, whose useful
physical tolerances differ.

The generic iwopy pipeline now supports an initial result for the first selected
stage, and iwopy solvers derive tolerances from each registered constraint.
FOXES Optimization needs one public workflow that maps persisted layouts and
family-specific tolerances onto those contracts.

## Decision

`LayoutPipeline` owns persisted-layout restart loading. `read_layout_index()`
selects `layout_<index>.csv` by numeric suffix independent of zero padding,
requires exactly one match, and validates turbine indices, finite x/y
coordinates, and `(n_turbines, 2)` shape. Relative restart directories resolve
below the pipeline base directory. `LayoutPipeline.run()` accepts a restart
layout index and directory and passes the loaded layout to iwopy as the initial
result for the first selected stage.

`LayoutOptimizerStage` accepts `min_dist_constraint_pars` for the automatically
created minimum-distance constraint. Explicit entries in `constraints` retain
their own constructor parameters, allowing boundary and minimum-distance
families to own different tolerances. Stage-owned construction parameters are
rejected from the pass-through mapping.

`WriteLayoutCallback.step_offset` continues snapshot numbers.
`WriteOptimizationHistoryCallback.iteration_offset` continues CSV iteration
numbers, and its append mode preserves only a non-empty file with the expected
header. Existing output paths are otherwise overwritten by the callback that
owns them. A restart caller preserves the pipeline table by constructing the
pipeline with `reset_table=False`.

## Compatibility And Migration

All new restart inputs are optional, so ordinary full-pipeline runs retain their
existing call shape. Restart callers move directly to the indexed layout and
offset APIs; no alternate filename format or legacy callback mode is retained.
Constraint-family parameters are supplied on the relevant constraint factory
mapping rather than as one optimizer-wide tolerance.

Maintained callers that restart an optimizer stage must select that later stage,
use the same persisted index for callback offsets, enable history append, and
avoid resetting the pipeline table.

## Consequences

- A layout optimization can resume from a validated persisted coordinate set
  without executing earlier pipeline stages.
- Snapshot and history output remains monotonic and preserves prior evidence
  when offsets match the selected restart index.
- Boundary and spacing feasibility can use physically meaningful independent
  tolerances.
- Restart correctness depends on callers keeping the selected layout index,
  callback offsets, stage selection, and history file consistent.
- Ambiguous numeric suffixes and malformed history schemas fail explicitly
  rather than selecting or appending silently.

## Verification

- `tests/pipelines/test_layout_pipeline.py` covers padded index loading, missing
  and ambiguous snapshots, and propagation to a selected stage.
- `tests/pipelines/test_random_subset.py` covers per-family constraint parameters
  and reserved-key rejection.
- `tests/test_layout_wind_rose_callback.py` covers restart snapshot offsets and
  invalid offsets.
- `tests/test_optimization_history_callback.py` covers append continuation,
  invalid offsets, and malformed headers.

## Alternatives Considered

- Parse and load restart CSV files in each application script. Rejected because
  validation and path semantics belong to the layout pipeline.
- Use one tolerance for every constraint family. Rejected because boundary and
  pairwise-distance residuals have different numerical and physical scales.
- Restart callback numbering at zero in a new directory. Rejected because it
  fragments one optimization history and makes selected-layout provenance less
  direct.

## References

- [Architecture](../architecture.md#pipelines-and-callbacks)
- [Naming conventions](../naming-conventions.md#pipelines-callbacks-and-outputs)
- `foxes_opt/pipelines/layout_pipelines.py`
- `foxes_opt/pipelines/stages/layout_optimizer.py`
- `foxes_opt/callbacks/write_layout.py`
- `foxes_opt/callbacks/write_optimization_history.py`
- iwopy ADR-0003, Constraint-Owned Solver Tolerances
- iwopy ADR-0004, Pipeline Restart Results
