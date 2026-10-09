# FOXES Optimization Development Guide

## Purpose

This guide collects repeatable repository workflows for human contributors and
coding agents. [AGENTS.md](../AGENTS.md) remains the policy source;
[architecture](architecture.md) explains runtime ownership and contracts; and
[naming conventions](naming-conventions.md) defines FOXES Optimization, FOXES,
and iwopy vocabulary, types, arrays, and identifiers.
[Docstring conventions](docstrings.md) define the public Python documentation
contract.

Prefer a focused check that can disprove the current change before running a
broad suite. When code files change, finish with the full test suite. Every
development change finishes with repository-wide pre-commit, current docstrings
and FOXES Optimization documentation, and an entry in the final current-version
changelog section.

## Forward-Only Development

FOXES Optimization development is strictly forward-looking by default under
[ADR-0001](adr/0001-forward-only-development.md). Implement the target contract
directly, update all maintained callers, and remove superseded code, tests,
examples, aliases, formats, fallback behavior, and documentation in the same
change.

Do not add a compatibility shim, deprecation branch, dual input format, or other
legacy path for convenience. A legacy exception requires explicit user
authorization, a bounded scope, and an ADR with an objective removal condition.
Identifying downstream impact remains mandatory; preserving the old contract
does not.

This policy governs changes from ADR-0001 onward. It does not require unrelated
historical compatibility code to be removed during an otherwise focused task.

## Environment And Dependencies

FOXES Optimization supports Python 3.10 through 3.14. Keep production syntax
compatible with 3.10 even when developing on a newer interpreter.

### Developer-Owned Environment Setup

The developer creates and maintains the default uv environment. During routine
development, always execute project commands with `uv run --no-sync`. Never use
bare `uv run`, run `uv sync`, or install or uninstall packages. If dependencies
are missing or stale, ask the developer to re-sync the environment.

The developer-owned default setup is based on these commands:

```console
uv sync --extra dev --extra test --upgrade
uv pip uninstall foxes iwopy
uv pip install -e <path to foxes>[test,dev,mpi,shp] --upgrade
uv pip install -e <path to iwopy>[opt] --upgrade
```

These setup commands are for the developer, not coding agents or routine
development. The editable FOXES and iwopy installs are intentional; mandatory
`--no-sync` prevents an ordinary command from replacing them through implicit
dependency resolution. CI owns its isolated environment separately and may
synchronize it in workflow jobs.

The dependency sets in `pyproject.toml` are optional extras, not uv dependency
groups. Install them with `--extra`; `--dev` does not select the `dev` extra.

| Dependency set | Use |
|---|---|
| Main dependencies | FOXES, iwopy, pymoo, pygmo, xarray, and NetCDF runtime support |
| `test` | pytest, notebook execution, pre-commit, and mypy |
| `dev` | contributor hooks, mypy, object-size inspection, and Jupyter |
| `doc` | Sphinx, AutoAPI, numpydoc, Immaterial, MyST-NB, and documentation notebooks |

Put a new dependency in the narrowest justified set. An import required by base
`foxes_opt` belongs in main dependencies. A development, test, or documentation
tool belongs in its corresponding extra. `uv.lock` is currently ignored and is
not a repository source of dependency versions.

## Finding The Owning Code

Start from the smallest behavior owner, then inspect one abstraction boundary
and the nearest tests.

| Change | Start in | Then inspect |
|---|---|---|
| Problem lifecycle or FOXES evaluation | `foxes_opt/core/` | iwopy base contract, concrete problems, and integrated layout tests |
| Farm-variable mapping | `foxes_opt/core/farm_vars_problem.py` | `OptFarmVars`, `tvar()` users, and objective/constraint consumers |
| Concrete problem formulation | `foxes_opt/problems/` | Core problem base, public exports, and corresponding examples/tests |
| Objective behavior or contraction | `foxes_opt/objectives/` | iwopy function contract, FOXES result dimensions, and focused objective tests |
| Boundary, spacing, or grouping rule | `foxes_opt/constraints/` | Geometry APIs, variable dependencies, derivatives, and focused constraint tests |
| Pipeline workflow | `foxes_opt/pipelines/` | Stage base contract, propagated results, examples, and `tests/pipelines/` |
| Optimizer callback | `foxes_opt/callbacks/` | iwopy callback lifecycle and callback output tests |
| YAML or CLI behavior | `foxes_opt/input/yaml/` | Constructor contracts, parameter-file docs, and YAML examples |
| Result conversion or writing | `foxes_opt/output/` | iwopy result shape, xarray schema, and output consumers |
| Public import | Owning package `__init__.py` | Root exports, AutoAPI output, examples, and import callers |

Useful searches include the class or function name, iwopy method name, `FC`
dimension, `FV` variable, objective/constraint component name, YAML key, and
pipeline data key. Search examples before changing constructors or class names;
they are often the clearest public call sites.

Do not treat `build/lib/foxes_opt/` as source. The maintained package is
`foxes_opt/`.

## Focused Validation

Run the narrowest applicable command first:

```console
# One behavior
uv run --no-sync pytest tests/path/test_module.py::test_behavior -q

# One test module or package area
uv run --no-sync pytest tests/path/test_module.py -q

# Production type checking, as configured in pyproject.toml
uv run --no-sync mypy foxes_opt

# Repository hooks on touched files
uv run --no-sync pre-commit run --files foxes_opt/path.py tests/test_module.py
```

Before closure, broaden to the repository gates:

```console
# Full Python test suite, required when code files changed
uv run --no-sync pytest tests

# Notebook execution used by CI
uv run --no-sync pytest --nbmake notebooks

# All formatting, linting, type, and file-hygiene hooks
uv run --no-sync pre-commit run --all-files
```

Pre-commit owns the configured Ruff and mypy invocations. A hook can modify a
file and still exit non-zero; inspect the diff and rerun it.

## Development Closure

A change is not complete until all closure evidence describes the final working
tree:

1. When code files changed, add or update tests for changed logic. Run the most
	discriminating focused tests first, then run
	`uv run --no-sync pytest tests`
	successfully. Documentation-only changes run their applicable link, render,
	version, or structure checks instead.
2. Run `uv run --no-sync pre-commit run --all-files` after the last edit. If a
	hook changes a file, inspect it and rerun the entire command.
3. Review every affected public Python API against
	[docstring conventions](docstrings.md). Update variables, component names,
	shapes, units, bounds, lifecycle, side effects, returns, errors, derivatives,
	and examples to match the final implementation.
4. Update `docs/architecture.md`, `docs/naming-conventions.md`, this guide,
	`docs/source/`, examples, ADRs, and any other FOXES Optimization information
	affected by the change. Do not knowingly leave stale prose, diagrams,
	commands, or contracts.
5. Read `project.version` from `pyproject.toml`. Confirm the final version
	section in `CHANGELOG.md` is exactly `## v<version>` and add a
	style-consistent entry for the change to that section.

Run notebook tests when notebooks change. A Sphinx build is not part of default
closure; use it as an explicit targeted check when requested or when rendered
Sphinx/AutoAPI output itself is under investigation. A documentation-only
change does not run the full runtime test suite, but it still runs applicable
documentation checks and all other closure gates.

## Test Structure

The test paths encode ownership:

| Area | Purpose |
|---|---|
| `tests/test_*.py` | Focused objectives, constraints, derivatives, callbacks, and shared behavior |
| `tests/00_layout_single_state/` | Integrated single-state layout optimization and problem reset behavior |
| `tests/pipelines/` | Layout pipelines, stage contracts, and deterministic subset behavior |
| `notebooks/` | Executable public workflows checked separately with nbmake |

Logic tests cover a successful path, a meaningful error path, and a relevant
edge case. For numerical code, assert values, shapes, component order,
population order, bounds, finite or NaN behavior, and tolerances as applicable.
Use deterministic synthetic arrays or explicitly public example data. Do not
use private site, customer, measurement, or research data as a fixture.

### Problems And Population Evaluation

Test one individual before testing a vectorized population. Population tests
compare `calc_population()` with stacked `calc_individual()` results and use at
least two candidates whose values expose ordering mistakes. For FOXES-backed
problems, include more than one state when the behavior reshapes population
states.

When problem application changes, test initialization requirements, repeated
evaluation, restoration of original states after population use, explicit and
default FOXES engine paths where affected, and both farm-only and point-result
behavior when supported.

### Objectives, Constraints, And Derivatives

Assert the number, names, order, bounds, and values of function components.
Selection tests must prove that a requested component subset remains in the
requested order.

For analytical derivatives:

- compare with centered finite differences at differentiable points;
- cover fixed or unrelated variables through the dependency mask;
- verify selected components and variable order;
- cover a non-differentiable or zero-distance error when the mathematics has
	one; and
- verify the documented numerical fallback signal, including `NaN`, where the
	iwopy contract uses it.

A derivative unit test does not replace an optimizer integration test when the
change affects iwopy's gradient assembly or backend behavior.

### Pipelines, Callbacks, And Files

Pipeline tests assert accepted `prev_results` forms, returned
`(success, results)`, stage order, determinism, and failure behavior. A failed
layout stage must preserve the documented incoming layout where required.

Callback and writer tests use temporary directories. Assert file naming,
step/evaluation semantics, schema, values, and overwrite behavior; avoid brittle
whole-image comparisons unless exact rendering is the behavior under test. For
plots, assert meaningful artists, coordinates, labels, limits, colors, or
returned objects.

`tests/test_optimization_history_callback.py` covers opt-in initial rows,
integer and float initial variables, selected objectives, undefined-value
errors, and unchanged defaults and restart append behavior.

## Examples And Notebooks

Examples and notebooks are public workflows, not scratch space. Keep their
commands and package imports current when an API changes. Run a changed example
directly with `uv run --no-sync python ...` and run a changed notebook through
nbmake:

```console
uv run --no-sync pytest --nbmake notebooks/path.ipynb
```

Use non-interactive plotting in automated checks and write generated files to a
temporary or documented result path. Examples must not depend on confidential
data, network access, or an undeclared local package checkout.

## Documentation

Public Python APIs follow [docstring conventions](docstrings.md). Types remain
in annotations; NumPy-style docstrings describe the optimization, scientific,
array, and runtime contract. Sphinx uses numpydoc and AutoAPI for package
references and MyST-NB for Markdown/notebook content.

When an explicit Sphinx rendering check is useful, use the prepared environment:

```console
uv run --no-sync sphinx-build -E -b html docs/source docs/build/html
```

If the documentation tools are unavailable, ask the developer to re-sync the
environment with the required documentation dependencies. Do not synchronize or
install them as part of the development task.

Open `docs/build/html/index.html` for local inspection. `docs/build/` and
generated AutoAPI pages under `docs/source/_*` are build output; do not edit or
commit them as source.

Update the relevant `docs/source/` page when public behavior, setup, a YAML
contract, output schema, or example changes. Keep parameter-file examples
runnable. The root `CHANGELOG.md` is the changelog source;
`docs/source/CHANGELOG.md` is a symlink to it.

Every development change adds a concise bullet to the final version section,
whose heading must match `project.version` from `pyproject.toml`. Preserve the
section's current category style and keep the full-changelog link current when
release metadata changes.

## CI Parity

GitHub Actions runs tests and notebooks on every supported Python version,
currently 3.10 through 3.14. A local change that passes only on the newest
interpreter is not sufficient when syntax, typing, dependencies, or numerical
behavior can differ by version.

CI owns and synchronizes its isolated environment. Its workflow commands are:

```console
uv sync --extra test
uv run pre-commit run --all-files
uv run pytest tests
uv run pytest --nbmake notebooks
```

These are CI commands, not local development instructions. CI's environment
ownership and distribution-build jobs are exceptions to the local
`uv run --no-sync` rule and are not a model for ordinary development commands.

## Release-Sensitive Changes

`pyproject.toml` is the package-version source and Sphinx reads it directly.
Release tags use `v<version>`; publish CI requires the tag and project version to
agree.

Before a release-sensitive change is complete, check:

1. Public imports and the `foxes_opt_yaml` console entry point.
2. Problem, objective, constraint, callback, pipeline, and YAML class names.
3. Optimization variable names, component names/order, population ordering,
	and output schemas.
4. FOXES and iwopy dependency floors and integration behavior.
5. The matching changelog section, API documentation, examples, and notebooks.
6. Tests across the supported Python floor and ceiling where the change is
	version-sensitive.

## Generated And Transient Paths

Do not edit these as implementation sources:

- `build/`, `dist/`, and `foxes_opt.egg-info/`
- `docs/build/` and generated AutoAPI output
- Python, pytest, mypy, Ruff, notebook, uv, and pre-commit caches
- local `results/`, `scratch/`, logs, and optimizer/backend output

Some result-like files are intentional public example or reference assets.
Confirm that a tracked test, example, or documentation page owns the file before
changing it, and regenerate it only when the task explicitly changes that
expected contract.

## Change Checklists

### Problem, Objective, Or Constraint

- Identify the iwopy base contract and FOXES integration boundary.
- Preserve variable, component, population, state, and turbine ordering.
- Declare bounds, dependencies, derivatives, and failure conditions.
- Add focused value/error/edge coverage and realistic integration coverage.
- Update exports, YAML construction, API docs, examples, and the current
	changelog section where affected.

### Pipeline, Callback, Or Output

- Identify stage input/output result forms and lifecycle/event ownership.
- Keep file side effects explicit and deterministic.
- Test temporary-path output, schema, naming, and relevant failure behavior.
- Preserve caller-supplied plotting objects and follow the applicable visual
	policy under `docs/fraunhofer-design/`.
- Update pipeline/YAML docs, examples, and the current changelog section where
	affected.

### External Integration

- Validate external values and class names at the boundary.
- Preserve the original exception as the cause when adding context.
- Test the minimum supported dependency contract and an error path.
- Update the dependency floor only with evidence and record consequential
	compatibility decisions in an ADR.
- Keep CLI parsing thin and reusable behavior importable for in-process tests.
