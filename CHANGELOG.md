# Changelog

## v0.1

- Copying code from `foxes.opt`

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.1](https://github.com/FraunhoferIWES/foxes/commits/v0.1)

## v0.1.3

- Adding documentation

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.1.3](https://github.com/FraunhoferIWES/foxes/commits/v0.1.3)

## v0.1.4

- Adding example notebook: `layout_optimization.ipynb`

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.1.4](https://github.com/FraunhoferIWES/foxes/commits/v0.1.4)

## v0.1.5

- Adding first test `00_layout_single_state`

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.1.5](https://github.com/FraunhoferIWES/foxes/commits/v0.1.5)

## v0.1.6

- Fixing bug in test

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.1.6](https://github.com/FraunhoferIWES/foxes/commits/v0.1.6)

## v0.1.7

- Updated documention

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.1.7](https://github.com/FraunhoferIWES/foxes/commits/v0.1.7)

## v0.2

- Adding github workflow for automatic testing
- Adding support for foxes engines

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.2](https://github.com/FraunhoferIWES/foxes/commits/v0.2)

## v0.2.1

- Bug fixed with `MaxFarmPower` objective, now using states weights in states contraction

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.2.1](https://github.com/FraunhoferIWES/foxes/commits/v0.2.1)

## v0.2.2

- New notebook: `wake_steering.ipynb`, demonstrating how to optimize yaw angles

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.2.2](https://github.com/FraunhoferIWES/foxes/commits/v0.2.2)

## v0.3

- Introducing command line application `foxes_opt_yaml`: Runs optimization from `yaml` parameter input file, no Python script needed.
- Changes in `FarmVarsObjective`: Renaming contraction rule `mean` into `mean_no_weights`.
- Adding run-time factories to problems, objectives, and constraints.
- Introducing `output` sub package, and first outputs `SingleOptResultsWriter` and `MultiOptResultsWriter`
- New example `yaml_input`, demonstrating how to run *foxes_out through a yaml parameter file

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.3](https://github.com/FraunhoferIWES/foxes/commits/v0.3)

## v0.4

- Support for *foxes* v1.3

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.4](https://github.com/FraunhoferIWES/foxes/commits/v0.4)

## v0.5

- Support for *foxes* v1.4

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.5](https://github.com/FraunhoferIWES/foxes/commits/v0.5)

## v0.6

- Compatibility with *foxes* v1.5
- Dropping support for Python 3.8
- Introducing minimal package versions of dependencies

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.6](https://github.com/FraunhoferIWES/foxes/commits/v0.6)

## v0.7.0

- Compatibility with *foxes* v1.7.0
- Moving currently not running or deprecated opt problems to new `scratch` folder: `GeomLayoutGridded`, `GeomLayout`, `RegGridsLayoutOptProblem`
- Bug fixes:
  - Bugs fixed that caused error with `RegularLayoutOptProblem`
  - Bug fixed with `GG` and layout problems

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.7.0](https://github.com/FraunhoferIWES/foxes/commits/v0.7.0)

## v0.7.1

- Compatibility with *foxes* v1.7.4
- Compatibility with *iwopy* v0.5.0
- New optimization pipeline: `LayoutPipeline`, combining different optimizations for generating an optimal wind farm layout

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.7.1](https://github.com/FraunhoferIWES/foxes/commits/v0.7.1)

## v0.8.0

- General:
  - Compatibility with *foxes* v1.9.2
  - Fixed vectorized wake-steering optimization with gridded rotor weights
  - Dropping support for Python 3.9
  - Introducing type annotations
  - Raising overlapping dependency minimum versions in `pyproject.toml` to be no lower than the corresponding minimum versions in *foxes*
- Documentation:
  - Migrating Sphinx API docs from `sphinx_immaterial.apidoc.python.apigen` to AutoAPI
- Examples:
  - New example `layout_field_data`, demonstrating pymoo-based layout optimization with `FieldData` states loaded from multiple NetCDF files

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes/commits/v0.8.0](https://github.com/FraunhoferIWES/foxes/commits/v0.8.0)

## v0.9.0

- General:
  - Expanded type annotations and tightened code checks across the package
- Dependencies:
  - Raised the minimum supported versions to *foxes* v1.9.6 and *iwopy* v0.5.0,
    and aligned overlapping dependency floors with both projects
- Callbacks:
  - Added an optimizer-independent layout callback with optional intermediate
    CSV output, configurable image snapshots and objective titles, and red
    highlighting of constraint violations in the legend
- Pipelines:
  - `LayoutPipeline` now retains stage results and supports layout plot and CSV
    output
  - Added `LayoutOptimizerStage` for optimizer-driven refinement of layouts from
    previous stages, with user-selectable optimization problem types
  - Added `RandomSubsetStage` for successive layout optimizations over random
    turbine and state subsets, forwarding layout optimizer configuration through
    its base stage
- Examples:
  - Added pymoo and SLSQP wind-rose examples using the layout callback
  - Added a pygmo-IPOPT wind-rose example without callbacks, since pygmo exposes
    neither exact live IPOPT iterations nor their decision vectors
  - Added a pygmo-IPOPT single-state layout example with configurable,
    layout-scale finite differences; progress output reports native IPOPT
    iterations separately from objective function evaluations
- Bug fixes:
  - Weighted state contractions now normalize by the selected weight sum
  - Fixed multi-state population optimization for population-major *foxes* state
    ordering

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes-opt/commits/v0.9.0](https://github.com/FraunhoferIWES/foxes-opt/commits/v0.9.0)

## v0.9.1

- General:
  - Added support for Python 3.14
- Callbacks:
  - Layout image snapshots now omit turbine annotations, use true turbine radii
    with color-matched outlines, use configurable validity colors that default
    to high-contrast blue and red, and live in a file-type subdirectory
  - Added a CSV optimization-history callback that records objective values and
    violated constraint-component counts for each iteration
- Pipelines:
  - `LayoutOptimizerStage` now forwards callbacks to the optimizer
- Constraints:
  - Added `NearestGroupConstraint`, enforcing nearest-distance and minimum
    connected-group-size limits with piecewise analytical position derivatives
  - `MinDistConstraint` now provides analytical position derivatives to *iwopy*
  - Vectorized `MinDistConstraint` variable-dependency mask construction for
    faster analytical Jacobians on large farms
  - Added analytical derivatives for area boundaries and geometrical layout
    constraints, including composed and internally excluded geometries
  - Cached bulk signed-distance gradients when assembling area-geometry
    Jacobians
- Objectives:
  - Added analytical derivatives for turbine-count and geometrical layout
    objectives; FOXES-result objectives continue to require `iwopy.LocalFD`
- Bug fixes:
  - Kept validity colors visible for small true-radius turbines in layout
    snapshots by matching marker edges to their fill colors

**Full Changelog**: [https://github.com/FraunhoferIWES/foxes-opt/compare/v0.9.0...v0.9.1](https://github.com/FraunhoferIWES/foxes-opt/compare/v0.9.0...v0.9.1)
