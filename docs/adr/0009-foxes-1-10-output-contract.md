# ADR-0009: FOXES 1.10 Output Contract

- Status: Accepted
- Date: 2026-11-20
- Supersedes: None

## Context

FOXES 1.10 removes `foxes.output`, including dataset-bound plotting and
`FarmResultsEval`. Output models now live under `foxes.models.outputs` and run
through algorithm calculation entry points. The ordinary farm-results preset
selects fewer fields than optimization callers require, and output-selected
capacity, efficiency and annual yields must be requested during calculation.
FOXES YAML models no longer accept imperative `functions` or signature injection.

## Decision

Require FOXES >=1.10.0 and consume its current APIs directly, following
[ADR-0001](0001-forward-only-development.md).

- `FarmOptProblem` requests `farm_results_all` unless the caller explicitly
  overrides `calc_farm_args.outputs`, preserving optimized fields and
  population-major results.
- State replacement uses current finalization, loaded-data and chunk-store
  APIs, removing the superseded state-data compatibility shim.
- Layout callbacks use a copied farm and a separate wake-free algorithm for
  metadata-only output execution. Pipeline and example plots use constructor
  options and `calc_outputs`, preserving established figures and styles.
- Pipeline calculations collect annual yields, efficiency and capacity
  explicitly. FOXES owns normalized state and whole-farm metric aggregation.
- Optimization result writers are parent-only FOXES output models, retaining
  their direct dataset/NetCDF APIs. Single-objective success remains an
  attribute. As confirmed by the user, multi-objective success is a variable
  on `pop`, preserving each candidate's flag instead of the invalid scalar
  conversion in the superseded writer. YAML injects iwopy results only
  at this owned writer boundary and returns `(outputs, results_storage)`.
- YAML output models use constructor options; only Matplotlib entries use
  `functions`. Figure setup can run in the engine phase before dependent models.
- Wake-growth names use `k1`/`ambk1` instead of removed `ka` multipliers.

## Compatibility And Migration

All maintained examples, notebooks and YAML configurations move to this contract
in the same change. There are no import aliases, legacy method dispatchers or
arbitrary signature-injection paths. Python runtime support, optimization
variable names, component order, population/state order, single-objective datasets,
snapshot paths and corporate plot styles remain unchanged.

Multi-objective datasets now store `success` on `pop`, not as a scalar attribute.
This intentional schema correction is included with the output migration.

Optimization writer YAML entries use `to_file` and `write_pars`; one simple
`$name` result label stores their dataset. Ordinary FOXES output entries retain
FOXES's own constructor-reference handling.

## Consequences

FOXES engines own output execution and chunking. Optimization writers and
metadata-only snapshots do not dispatch scientific worker calculations.
Explicit pipeline metric collection avoids rerunning simulation solely to
populate tables. Retaining all calculated optimization fields can cost more
memory than FOXES's narrower ordinary preset, but preserves the established
optimization result contract. Existing cohesive pipeline and callback output
methods remain in their owning modules despite exceeding size guidelines.

## Verification

- [Population regression tests](../../tests/00_layout_single_state/test_farm_opt_problem_reset_states.py)
  cover repeated mapped-variable calculations and state restoration.
- [Callback tests](../../tests/test_layout_wind_rose_callback.py) cover snapshot
  schemas, isolation, styles and explicit output failures.
- [Pipeline tests](../../tests/pipelines/test_layout_pipeline.py) cover real
  metric aggregation and CSV/layout/flow exports.
- [YAML tests](../../tests/test_optimizer_configuration.py) execute the public
  constructor configuration in figure and no-figure modes and reject invalid
  writer definitions.
- Execute both maintained notebooks, the full runtime suite and all pre-commit
  hooks. Check documentation links internally only.

## Alternatives Considered

- Import aliases cannot restore the removed imperative output behavior and
  violate forward-only development.
- Reimplementing the old result evaluator duplicates FOXES metric and
  aggregation ownership.
- Requesting FOXES's ordinary default collector loses optimized quantities such
  as `FV.YAWM`.

## References

- [Architecture](../architecture.md)
- [Naming conventions](../naming-conventions.md)
- [Development workflow](../development.md)
- [YAML contract](../source/parameter_files.md)
