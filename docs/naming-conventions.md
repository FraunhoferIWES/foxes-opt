# FOXES Optimization Naming And Typing Conventions

## Purpose

This document defines stable FOXES Optimization vocabulary and naming rules for
Python code, optimization arrays, FOXES variables, iwopy functions, YAML,
pipelines, tests, documentation, and public contracts. Use the same term at
every layer unless an external format requires a mapping. See
[architecture](architecture.md) for ownership and runtime contracts and
[development](development.md) for contributor commands.

Record a broadly consequential change to these rules in an
[ADR](adr/README.md). Do not rename an established public identifier solely to
match a newer style preference; Python exports, variable names, function
components, YAML keys, pipeline data keys, and result fields are
compatibility-sensitive.

## General Rules

- Prefer a precise domain term over a generic noun such as `item`, `manager`,
	`handler`, or `data_object`.
- Use singular names for one object and plural names for collections. Preserve
	established external names from FOXES and iwopy.
- Keep the same noun in code, docstrings, tests, YAML, examples, and user
	documentation.
- Use an established abbreviation only when it appears below or in the FOXES or
	iwopy public contract. Do not create near-synonyms for shortness.
- Name booleans as predicates or flags, counts with `n_`, index arrays with
	`_inds`, individual indices with `_i`, and starting offsets with `_i0` where
	that established convention applies.
- Include physical units in docstrings and metadata, not in public variable
	names, unless the existing external contract includes the unit.
- Preserve external names at integration boundaries and translate them once
	into the owning package vocabulary.

## Domain Vocabulary

| Preferred term | Meaning | Avoid or distinguish from |
|---|---|---|
| FOXES Optimization | The project and library described here | `FOXES` when only the simulation package is meant |
| `foxes-opt` | The Python distribution and repository name | The import package name |
| `foxes_opt` | The import package | The distribution spelling in package metadata |
| FOXES | The wind-farm simulation package used by a problem | The optimizer or optimization problem |
| iwopy | The generic optimization framework and optimizer interface | FOXES simulation execution |
| algorithm / `algo` | A FOXES `Algorithm` that calculates farm or point results | iwopy optimizer or optimization algorithm |
| problem | An iwopy `Problem` defining variables and registered functions | FOXES simulation algorithm |
| farm optimization problem | A `FarmOptProblem` that evaluates candidates through FOXES | Geometry-only problem |
| geometry-only problem | A problem evaluated from layout geometry without FOXES results | Farm simulation problem |
| optimizer | An iwopy optimizer that solves a problem | Problem, objective, or FOXES engine |
| objective | An iwopy function whose components are optimized | Constraint or complete problem |
| constraint | An iwopy function whose component bounds define feasibility | Objective or area boundary object |
| component | One scalar output of an objective or constraint | Optimization variable |
| individual / candidate | One complete assignment of integer and floating variables | One turbine or state |
| population | An ordered batch of optimization individuals | FOXES states or turbines |
| optimization variable | One integer or floating degree of freedom exposed to iwopy | FOXES result variable |
| farm variable | A FOXES `FV` quantity changed by an optimization problem | Generic optimization variable when no FOXES mapping exists |
| states | A FOXES `States` provider for ambient conditions | Population or optimization iteration |
| wind farm / `farm` | The ordered FOXES `WindFarm` and its turbines | Layout array or farm results |
| layout / `layout_xy` | Turbine horizontal coordinates shaped `(n_turbines, 2)` | Complete `WindFarm` object |
| farm results | The xarray dataset calculated by the FOXES algorithm | iwopy optimization result |
| problem results / `problem_results` | Problem-specific payload returned after applying one individual or population and passed to functions | Final optimizer result |
| optimizer result / `opt_results` | Backend-specific result returned by `optimizer.solve()`; it can contain `problem_results` | FOXES farm results or direct problem payload |
| valid / feasible | A candidate whose constraint components satisfy their bounds | A successfully evaluated but infeasible candidate |
| callback | An iwopy optimizer-event observer with explicit side effects | Pipeline stage or objective |
| pipeline | An ordered `LayoutPipeline` workflow that propagates stage results | Optimizer lifecycle |
| stage | One transformation from `prev_results` to `(success, results)` | Optimizer iteration or callback event |
| FOXES engine | The backend that chunks and executes simulation calculations | iwopy optimizer backend |

## Python Symbols And Files

- Modules, functions, methods, parameters, and local variables use
	`snake_case`; classes use `PascalCase`; module constants use `UPPER_CASE`.
- Established public spellings such as `FarmOptProblem`, `OptFarmVars`,
	`LayoutPipeline`, `LocalFD`, `FC`, and `FV` remain canonical.
- Prefix non-public implementation details with one underscore. Do not export a
	private helper through a package `__init__.py`.
- Keep the package root small. Re-export public concepts from their owning
	subpackage; import `foxes_opt.pipelines` explicitly rather than silently
	widening the root API.
- Package and source directories use lowercase names. Example directories use
	descriptive `snake_case`. Tests use `test_<subject>.py` and
	`test_<behavior>()` or an equally explicit observable behavior name.
- Put code in the module that owns the concept described in
	[architecture](architecture.md#module-boundaries). Do not add a generic
	`helpers.py` when an existing focused module owns the behavior.
- Use `from __future__ import annotations` consistently with the surrounding
	module. Production annotations must remain valid for Python 3.10.

## Common Type Annotations

Infer types from the owning base class and call site rather than from a name
alone, but use these established meanings unless the local contract says
otherwise.

| Name | Usual type | Notes |
|---|---|---|
| `algo` | `foxes.core.Algorithm` | FOXES simulation orchestration, not an optimizer |
| `farm` | `foxes.core.WindFarm` | Not a coordinate array or xarray dataset |
| `states` | `foxes.core.States` | May be wrapped by `PopulationStates` during vectorized evaluation |
| `mbook` | `foxes.models.ModelBook` | FOXES model registry |
| `problem` | `iwopy.Problem` or a derived `foxes_opt` problem | Use the narrowest derived type required by the API |
| `optimizer` | `iwopy.Optimizer` | Initialized against a problem before solving |
| `objective` | `iwopy.Objective` or `FarmObjective` | May expose one or several components |
| `constraint` | `iwopy.Constraint` or `FarmConstraint` | Bounds and component order are part of the contract |
| `callback` | `iwopy.OptimizerCallback` | Runs at optimizer-defined events |
| `boundary` | `foxes.utils.geom2d.AreaGeometry` | Geometry used by area/layout constraints |
| `farm_results` | `xarray.Dataset` | FOXES simulation result |
| `problem_results` | Problem-specific payload, often `Any` at generic function boundaries | For `FarmOptProblem`, farm results or farm/point results |
| `opt_results` | Backend-specific iwopy optimization result | Returned by `optimizer.solve()` and may contain `problem_results` |
| `layout_xy` | `numpy.ndarray` | Shape `(n_turbines, 2)` |
| `vars_int` | `numpy.ndarray` | Integer variables for one individual or population |
| `vars_float` | `numpy.ndarray` | Floating variables for one individual or population |
| `fig` | `matplotlib.figure.Figure` | Add `None` only when allowed by the contract |
| `ax` | `matplotlib.axes.Axes` | Preserve caller-supplied axes |

Before using `Any` or `object`, inspect the iwopy base class, the nearest FOXES
contract, and call sites. A problem-specific `problem_results` payload or
backend-specific `opt_results` value may remain `Any` at a generic boundary;
that does not justify broadening arrays, problems, functions, or callbacks that
have a concrete contract. Runtime imports needed during FOXES engine or
optimizer execution must not be hidden behind `TYPE_CHECKING`.

## Optimization Arrays And Counts

Use separate `vars_int` and `vars_float` arrays as required by iwopy. Names and
shapes are observable contracts:

| Name | Shape or meaning |
|---|---|
| `n_vars_int` | Number of integer optimization variables |
| `n_vars_float` | Number of floating optimization variables |
| `n_components` | Number of scalar objective or constraint outputs |
| `n_pop` | Number of individuals in a vectorized population |
| `n_states` | Number of original FOXES states per individual |
| `n_turbines` | Number of turbines in the farm or layout |
| `vars_int` | `(n_vars_int,)` or `(n_pop, n_vars_int)` |
| `vars_float` | `(n_vars_float,)` or `(n_pop, n_vars_float)` |
| `layout_xy` | `(n_turbines, 2)` |
| mapped farm variable | `(n_states, n_selected_turbines)` or `(n_pop, n_states, n_selected_turbines)` |

Use `var_names_int` and `var_names_float` for ordered public variable names and
`cmpnts` or `components` only where the owning iwopy API establishes that
spelling. Preserve population-major ordering when a population is flattened
into FOXES states: candidate first, original state second.

Use `_min` and `_max` for variable or component bounds, `_tol` for feasibility
tolerance, and `_gradients` or `_derivatives` according to the owning iwopy
method. Do not call a constraint Jacobian a layout gradient when its first axis
is actually the component axis.

## FOXES Dimensions And Variables

Import `foxes.constants as FC` for structural dimensions and
`foxes.variables as FV` for physical fields. Use constants such as `FV.X`,
`FV.Y`, `FV.H`, `FV.D`, `FV.WS`, `FV.REWS`, `FV.P`, `FV.CT`, and their ambient
counterparts instead of local string copies.

Use `tvar(variable, turbine)` for the established per-turbine optimization name
`<variable>_<turbine:04d>`. Do not duplicate its zero-padding or separator logic
at a call site.

`OptFarmVars` uses these variable types exactly:

- `uniform`: one value shared across states and turbines;
- `state`: one value per state, using five-digit state indices in generated
	names;
- `turbine`: one value per turbine; and
- `state-turbine`: one value for each state/turbine pair.

Do not substitute `state_turbine` for the public `state-turbine` selector or
rename generated variables without updating YAML, tests, examples, and result
consumers.

## Objectives, Constraints, And Geometry

- Objective and constraint class names describe the quantity or rule, for
	example `MaxFarmPower`, `MinimalMaxTI`, `MinDistConstraint`, and
	`NearestGroupConstraint`.
- Use `min`, `max`, `sum`, `mean_no_weights`, and state-only `weights` for the
	established farm-result contractions. Do not document `mean` when the accepted
	selector is `mean_no_weights`.
- Use `m` and `D` for the established distance-unit selectors. Explain whether
	`D` means one turbine's diameter, a pairwise diameter rule, or another owning
	API convention.
- A constraint component has a name, lower/upper bound, tolerance, and stable
	position in the returned vector. Preserve the same order in values,
	derivatives, diagnostics, and result output.
- Constraint objects own tolerance. Use `min_dist_constraint_pars` for the
	automatic minimum-distance family and each explicit `constraints` entry for
	that family's settings; do not use one optimizer-wide tolerance name.
- Geometry names distinguish an `AreaGeometry` boundary from `layout_xy`, a
	regular-grid parameterization, and a FOXES `WindFarm`.
- `valid` means all relevant component bounds pass at the declared tolerance;
	it does not mean only that evaluation returned finite values.

## Factories, YAML, And CLI

- YAML keys use the owning Python constructor parameter name unless the input
	adapter explicitly documents a mapping.
- Problem, objective, constraint, optimizer, and output class names resolved
	from YAML are public identifiers. Check configuration examples and tests before
	renaming one.
- Keep class resolution and validation in `foxes_opt.input`. Reusable solving or
	output logic belongs below the CLI parser.
- The public console script is `foxes_opt_yaml`; Python modules and callable
	names retain underscores even though the distribution uses a hyphen.
- Filesystem parameters end in `_file`, `_path`, or `_dir` according to what the
	caller may supply. Use `out_dir` where that established CLI/YAML contract
	already exists.
- Error messages identify the problem, function, component, variable, stage,
	backend, path, expected shape, and actual value needed to diagnose failure.

## Pipelines, Callbacks, And Outputs

- Pipeline classes end in `Pipeline`; composable stage classes end in `Stage`.
- Use `prev_results` for a stage's incoming result and `results` for its returned
	payload. Keep the `(success, results)` pair distinct from optimizer
	`opt_results`.
- Use `restart_layout_index` for a persisted layout's numeric file suffix and
	`restart_layout_dir` for its containing directory. Zero padding is a file
	format detail, not part of the index value.
- Callback classes describe their side effect, for example writing a layout or
	optimization history. Parameters distinguish iteration/step indices from
	objective evaluation counts. Use `step_offset` for layout snapshots and
	`iteration_offset` for optimization-history rows.
- Result datasets and tables use objective and constraint component names from
	the owning iwopy functions. Do not regenerate friendlier labels in a writer.
- Plotting parameters use `color` for one colour and `colors` or role-specific
	names for collections. Preserve caller-supplied figures, axes, labels, and
	paths.

## Documentation Names

Use [docstring conventions](docstrings.md) for required coverage, NumPy-style
structure, optimization/scientific contracts, examples, and review. This
document remains authoritative for names used inside those docstrings.

- Section entries use the exact signature parameter or semantic return name.
- State array shapes with the names in this document and use established `FC`
	dimensions and `FV` variables where the array enters FOXES.
- Distinguish optimizer results, FOXES farm results, and written xarray results.
- Link to the owning public API rather than inventing a prose-only synonym that
	users cannot search for.

## Test Names

Name focused tests after observable behavior. Use parametrization identifiers
that expose the problem, objective, constraint, optimizer, population mode, or
pipeline stage under test.

- Root `tests/test_*.py` files cover focused objectives, constraints,
	derivatives, and callbacks.
- `tests/00_layout_single_state/` covers integrated layout optimization and
	reset behavior.
- `tests/pipelines/` covers layout-pipeline and stage contracts.

Do not call a single-individual unit test a population test or an analytical
derivative check an optimizer integration test.

## ADR Naming

- File format: `NNNN-short-kebab-case-title.md`
- Start at `0001` and increment monotonically.
- Keep an ADR title stable after merge unless a later ADR supersedes the
	decision.
