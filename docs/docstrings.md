# FOXES Optimization Docstring Conventions

## Purpose And Authority

This document is the canonical docstring standard for FOXES Optimization
production Python. It expands the repository policy in
[AGENTS.md](../AGENTS.md) and uses the terms and types from
[naming conventions](naming-conventions.md). Architecture and runtime ownership
remain defined in [architecture](architecture.md).

FOXES Optimization uses NumPy-style docstrings rendered by numpydoc and Sphinx
AutoAPI. Type information belongs in annotations and is not repeated in
docstring section labels. Docstrings describe the current contract only. Under
the forward-only policy, do not document removed behavior, deprecated aliases,
or a legacy path unless an explicitly authorized exception still exists.

Existing source contains older variations. They are not precedents for new
work. When a public API is touched, migrate its affected docstrings directly to
this standard without preserving obsolete wording or formatting.

## Required Coverage

Document every public:

- module that owns a distinct user-facing or extension concern;
- class and constructor;
- function and method;
- property;
- class method and static method; and
- abstract or protected extension hook that downstream problems, functions,
	callbacks, or pipeline stages are expected to implement or call.

A symbol is public when it is exported by a package, documented as an extension
contract, used by examples or YAML, or intentionally available to downstream
callers without a private underscore prefix. Public class names resolved from
YAML and the `foxes_opt_yaml` implementation require the same care even when
most callers reach them indirectly.

Private helpers need a docstring when their contract, component ordering,
dimensions, mutation, algorithm, derivative behavior, or failure mode is not
obvious from the signature and code. Do not add empty narration to a trivial
private helper.

### Overrides And Inheritance

- An override that changes behavior, accepted values, side effects, errors,
	return semantics, lifecycle, components, derivatives, or shapes has its own
	complete docstring.
- A pure pass-through override with exactly the inherited contract may inherit
	the base docstring. Do not copy identical prose into both places.
- An abstract method documents implementation obligations, including required
	variables, shapes, component order, mutation, and returns. A concrete override
	documents additional or changed behavior while remaining understandable in
	the generated inheritance view.
- An overload set has one authoritative docstring on the implementation.
	Explain behavior selected by each overload condition without duplicating type
	annotations.

## Structure And Style

Use triple double quotes. Start with a concise one-line summary:

- Use an imperative verb for an operation: "Calculate objective components."
- Use a noun phrase for a class, property, or exposed value: "A FOXES-backed
	farm optimization problem."
- State the domain result, not the implementation technique.
- End the summary with a period.

Add a blank line and extended prose only when it clarifies optimization or
scientific meaning, lifecycle, mutation, ordering, side effects, or constraints.
Use the exact vocabulary from `docs/naming-conventions.md`.

Use these sections in this order, omitting sections that do not apply:

1. `Parameters`
2. `Attributes`
3. `Returns` or `Yields`
4. `Raises`
5. `Warns`
6. `See Also`
7. `Notes`
8. `References`
9. `Examples`

Section headings use NumPy style exactly:

```text
Parameters
----------
```

Use reStructuredText inside Python docstrings:

- Write literals, parameter names, class names, `FC`/`FV` constants, and short
	code expressions with double backticks, for example ``vars_float`` and
	``(n_pop, n_vars_float)``.
- Use resolvable Sphinx roles such as `:class:` or `:meth:` only when a real
	cross-reference helps the reader.
- Use `:math:` for inline mathematics and a `.. math::` block for displayed
	equations.
- Do not use Markdown links or headings inside a Python docstring.

## Parameters

List parameters in signature order. Use the exact parameter name without a type
suffix; omit `self` and `cls`. Write variadic parameters as `args` and `kwargs`,
matching the name without leading asterisks.

```text
Parameters
----------
vars_int
    Integer variables with shape ``(n_vars_int,)``.
vars_float
    Floating-point variables with shape ``(n_vars_float,)``.
problem_results
    Results returned by applying the variables to the problem.
components
    Selected component indices in requested output order, or ``None`` for all
    components.
```

Parameter prose describes:

- domain meaning and role;
- accepted semantics that annotations cannot express;
- physical units;
- shapes, dimensions, component order, population order, and required
	`FC`/`FV` fields;
- whether the value is read, mutated, retained, consumed, or written;
- relationships or exclusivity with other parameters; and
- behavior of meaningful sentinel values such as ``None``.

Do not repeat the annotation, write `optional`, or restate a literal default
that is already clear in the signature. Explain a default only when its
behavior or scientific meaning is not obvious. For a boolean, describe what
enabling it does instead of writing "Boolean flag."

Distinguish a FOXES algorithm from an iwopy optimizer, a farm result from an
optimizer result, and a class name resolved from YAML from an instance supplied
directly. Do not imply that an instance-only API accepts `str` merely because a
configuration adapter resolves strings elsewhere.

## Attributes

Use `Attributes` for public state that callers inspect or set directly and that
is not already fully represented by documented properties. Describe mutation,
units, shapes, and lifecycle validity. Do not list private caches or
implementation state.

Class prose explains the concept and invariants. Constructor arguments belong
in the `__init__` docstring, following the AutoAPI configuration that combines
class and constructor content. Do not duplicate constructor parameters in the
class docstring.

## Returns And Yields

Omit `Returns` for a function that always returns `None`. Otherwise, use a
semantic result name, not a type name:

```text
Returns
-------
values
    Objective values with shape ``(n_pop, n_selected_components)``.
```

- Never use `int`, `dict`, `ndarray`, `Dataset`, or another annotation as the
	return entry name. Types are rendered from the signature.
- For multiple positional returns, document one semantic entry per returned
	component in order.
- For a mapping, describe key meaning, value meaning, shapes, and ownership or
	copying behavior.
- For xarray output, name significant dimensions, coordinates, variables,
	component ordering, and relevant attributes.
- For an iterator or generator, use `Yields` instead of `Returns` and describe
	one yielded element.
- If a method mutates an input or object and also returns it, state both facts.
- If a return can be `None`, explain the condition that produces `None`.

A property uses a noun-phrase summary and a semantic `Returns` entry matching
the exposed concept. A context manager documents the value returned by
`__enter__` and cleanup guaranteed by `__exit__` in the class or factory
docstring.

## Raises And Warnings

Document exceptions that are intentional parts of the public contract and that
a caller can reasonably prevent or handle. Use the exception class as the entry
and describe the exact condition:

```text
Raises
------
RuntimeError
    If the problem has not been initialized.
ValueError
    If the layout shape or selected component indices are invalid.
```

Do not list incidental implementation exceptions, assertions, or impossible
states. If the implementation deliberately translates a lower-level exception,
document the public exception and preserve the original as its cause.

Use `Warns` only for emitted Python warnings, naming the warning class and its
condition. Do not use it for verbosity output or general caveats; put those in
`Notes`.

## See Also, Notes, And References

Use `See Also` for a small set of directly related public APIs, with one short
reason for each relationship. Do not turn it into a package index.

Use `Notes` for details that matter after the basic contract is understood:

- objective or constraint equations and sign conventions;
- variable, component, population, state, or turbine ordering;
- initialization and finalization requirements;
- mutation of the FOXES algorithm, states, farm, or pipeline data;
- finite-difference and analytical derivative behavior;
- NaN, clipping, tolerance, infeasibility, or optimizer failure policy;
- caching, copying, ownership, or thread/process considerations; and
- backend or dependency behavior visible to callers.

Use `References` for scientific publications or external specifications that
define the implemented method. Give enough bibliographic information to locate
the source and use a stable DOI or URL where available. Do not cite a source
that merely mentions the method.

## Examples

Add `Examples` when correct use is not obvious from the signature, especially
for problem construction, objective/constraint registration, YAML construction,
engine contexts, callbacks, result writers, and multi-stage pipelines. A trivial
getter does not need an example.

Examples must:

- use the public import surface;
- be minimal, deterministic, and runnable;
- use synthetic or explicitly public package/example data;
- avoid network access, interactive input, and undeclared local dependencies;
- show the result or assertion that proves the behavior; and
- remain compatible with Python 3.10.

Use `>>>` prompts for short doctest-style examples. Longer optimization
workflows belong in `examples/`, notebooks, or `docs/source/`; link them instead
of embedding an unmaintainable script.

## Optimization Contracts

A FOXES Optimization docstring makes hidden iwopy and array contracts explicit.

### Individuals And Populations

- `vars_int` for one individual has shape ``(n_vars_int,)`` and for a
	population has shape ``(n_pop, n_vars_int)``.
- `vars_float` for one individual has shape ``(n_vars_float,)`` and for a
	population has shape ``(n_pop, n_vars_float)``.
- `calc_individual()` values have shape ``(n_selected_components,)``.
- `calc_population()` values have shape
	``(n_pop, n_selected_components)``.
- `layout_xy` has shape ``(n_turbines, 2)`` and uses meters unless the owning
	API declares another unit.
- Mapped farm variables state whether their shape is
	``(n_states, n_selected_turbines)`` or
	``(n_pop, n_states, n_selected_turbines)``.

State whether a method supports vectorized population evaluation or loops
internally through individuals. For FOXES-backed population results, describe
population-major ordering when it is exposed or reshaped.

### Components, Bounds, And Dependencies

An objective or constraint docstring states:

- the number and stable names/order of scalar components;
- whether each objective component is minimized or maximized;
- lower and upper bounds and feasibility tolerance for constraints;
- the meaning and shape of component values;
- behavior of the `components` selector, including output order; and
- variable dependencies when they are specialized.

`vardeps_float()` returns a boolean mask shaped
``(n_components, n_vars_float)``. Explain why a dependency is true or false
when the implementation narrows the default iwopy mask.

Do not describe a vector of components as one scalar result. Do not call a
candidate valid merely because calculation succeeded; distinguish finite
evaluation from constraint feasibility.

### Derivatives

For `ana_deriv()`, document:

- that `var` is one floating-variable index;
- whether `components=None` means every component;
- that the return contains one derivative per selected component in selected
	order;
- physical units of the derivative when meaningful;
- conditions producing exact zero, `NaN` as an iwopy numerical-derivative
	fallback signal, or an exception at a non-differentiable point; and
- any cache or mutation visible across repeated calls.

For a complete gradient/Jacobian API, state the component and variable axes
explicitly. Never rely on the word "gradient" to communicate array orientation.

Analytical derivative claims require tests against finite differences at
differentiable points. A piecewise derivative docstring identifies the branch
or tie where differentiability fails.

### Problem Lifecycle And Results

Problem APIs state whether initialization is required, which FOXES objects they
retain or mutate, and when original states or farm values are restored.

`apply_individual()` and `apply_population()` document what `problem_results`
contains for farm-only and farm-plus-point configurations. This problem-specific
payload is passed to objective and constraint calculations. The separate result
returned by `optimizer.solve()` is `opt_results`; it may have a backend-specific
type and can contain `problem_results` as one attribute.

Optimizer-facing APIs document initialization, solve, and finalization
requirements. FOXES engine behavior belongs in the problem or pipeline API that
opens or consumes the engine context, not in every objective.

## FOXES Scientific Contracts

When an API consumes or returns FOXES data:

- name structural dimensions with `FC` constants;
- name physical fields with `FV` constants;
- state units, state/turbine selections, and contraction order;
- distinguish ambient and calculated values;
- distinguish farm results from point results; and
- explain weighting, normalization, NaN handling, and scaling where relevant.

For population evaluation, describe how the FOXES state axis maps to
``(FC.POP, FC.STATE)`` before contraction. For weighted objectives, state
accepted weight dimensions and whether weights are normalized over the selected
state axis.

Use the established contraction selectors ``min``, ``max``, ``sum``,
``mean_no_weights``, and state-only ``weights`` exactly. Do not shorten or
normalize those public values in prose.

## Special API Forms

### Abstract Methods And Callbacks

An abstract method defines what an implementation receives, may mutate, and
must return. Describe required variable arrays, problem results, component
order, and lifecycle rather than one current subclass algorithm.

A callback documents the iwopy event in which it runs, the optimizer state it
reads, files it writes, step or evaluation numbering, initialization/finalizing
behavior, and whether failure propagates. Its annotation remains the type
authority.

### Pipelines And Stages

A pipeline documents its stage order, accepted initial result, propagated result
forms, and final `(success, results)` pair. A stage documents the accepted
`prev_results` forms, the meaning of its returned `(success, results)`, mutation,
deterministic behavior, side effects, and fallback on failure. Distinguish stage
`results` from FOXES `farm_results`, function `problem_results`, and optimizer
`opt_results`.

### YAML And CLI Boundaries

Document accepted paths, YAML sections and keys, public class-name resolution,
defaults with non-obvious behavior, and output locations. Identify validation
failures at the input boundary. Use the same name and meaning as the Python
contract or describe the adapter mapping explicitly.

Keep command-line docstrings focused on argument-to-library orchestration. The
reusable function below the parser owns the calculation contract.

### Plotting And File Output

Document whether an API creates or consumes a figure/axes, preserves caller
styles, writes one file or a sequence, creates directories, overwrites files,
and returns the written path or plotting object. State image step numbering and
CSV/NetCDF schema where callers consume it.

## Templates

### Function Or Method

```python
def calc_individual(
    vars_int: np.ndarray,
    vars_float: np.ndarray,
    problem_results: Any,
    components: Sequence[int] | None = None,
) -> np.ndarray:
    """Calculate selected constraint components for one individual.

    Parameters
    ----------
    vars_int
        Integer variables with shape ``(n_vars_int,)``.
    vars_float
        Floating-point variables with shape ``(n_vars_float,)``.
    problem_results
        Results returned by applying the individual to the problem.
    components
        Selected component indices in requested output order, or ``None`` for
        all components.

    Returns
    -------
    values
        Constraint values with shape ``(n_selected_components,)``.

    Raises
    ------
    ValueError
        If a selected component index is invalid.
    """
```

### Class And Constructor

```python
class ExampleObjective(FarmObjective):
    """An objective calculated from one FOXES farm-result variable."""

    def __init__(self, problem: FarmOptProblem, variable: str) -> None:
        """Initialize the objective.

        Parameters
        ----------
        problem
            The FOXES-backed optimization problem.
        variable
            The ``FV`` farm-result variable to optimize.
        """
```

### Property

```python
@property
def layout_xy(self) -> np.ndarray:
    """The current turbine layout.

    Returns
    -------
    layout_xy
        Turbine x/y coordinates in meters with shape ``(n_turbines, 2)``.
    """
```

Templates illustrate structure, not required wording. Replace every generic
phrase with the concrete optimization or scientific contract.

## Review Checklist

Before closing a change that touches a public API, confirm:

- [ ] The summary describes the current public contract.
- [ ] Signature names and docstring parameter names match exactly.
- [ ] Annotations contain types; prose does not duplicate them.
- [ ] Variable, component, population, state, and turbine shapes/order are
	explicit.
- [ ] Bounds, tolerances, units, `FC` dimensions, and `FV` variables are
	explicit where applicable.
- [ ] Mutation, initialization/finalization, caching, engine behavior, and file
	side effects are described where observable.
- [ ] Derivative axes, fallback signals, non-differentiable cases, and
	dependencies are complete.
- [ ] Semantic returns, intentional exceptions, and warning conditions are
	complete.
- [ ] Examples use public APIs and runnable public or synthetic data.
- [ ] Removed behavior and legacy names are absent.
- [ ] Related base classes, overrides, YAML docs, API docs, examples, and
	architecture records remain consistent.

Run the closure commands in [development](development.md#development-closure).
A Sphinx build is not part of default closure. To inspect rendered docstrings or
AutoAPI pages explicitly, run:

```console
uv run --no-sync sphinx-build -E -b html docs/source docs/build/html
```

If Sphinx is unavailable, ask the developer to re-sync the environment with the
documentation dependencies. Do not install them during the task.

Generated AutoAPI pages under `docs/source/_*` and HTML under `docs/build/` are
outputs, not sources. Inspect them, but do not edit or commit them.
