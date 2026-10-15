# Welcome to FOXES Optimization

## Overview

*FOXES Optimization* is a modular Python package for wind-farm optimization by
Fraunhofer IWES. Typical applications include:

- Wind-farm layout optimization,
- Wake steering through turbine yaw optimization,
- Constrained optimization of turbine positions and farm geometries,
- Multi-stage layout optimization pipelines.

FOXES Optimization combines wind-farm and wake modelling from
[FOXES](https://github.com/FraunhoferIWES/foxes) with the shared optimizer
interface provided by [iwopy](https://github.com/FraunhoferIWES/iwopy). It
supports vectorized population evaluation and uses FOXES execution engines for
local or distributed calculations.

| Resource | Link |
| :--- | :--- |
| Documentation | [https://fraunhoferiwes.github.io/foxes-opt](https://fraunhoferiwes.github.io/foxes-opt) |
| Source code | [https://github.com/FraunhoferIWES/foxes-opt](https://github.com/FraunhoferIWES/foxes-opt) |
| PyPI | [https://pypi.org/project/foxes-opt/](https://pypi.org/project/foxes-opt/) |
| conda-forge | [https://anaconda.org/conda-forge/foxes-opt](https://anaconda.org/conda-forge/foxes-opt) |
| License | [MIT](https://github.com/FraunhoferIWES/foxes-opt/blob/main/LICENSE) |

## Requirements

Requires Python 3.10 through 3.14.

## Installation

There are multiple ways to install *foxes-opt*.

### Installation as standard user

```console
pip install foxes[opt]
```
or
```console
pip install foxes-opt
```
or
```console
conda install foxes-opt -c conda-forge
```

### Installation as developer

The developer creates and maintains the uv environment from the `foxes-opt`
repository root. The default setup uses editable FOXES and iwopy checkouts:

```console
uv sync --extra dev --extra test --upgrade
uv pip uninstall foxes iwopy
uv pip install -e <path to foxes>[test,dev,mpi,shp] --upgrade
uv pip install -e <path to iwopy>[opt] --upgrade
```

During development, run project commands with `uv run --no-sync` so uv does not
replace those editable installs. Coding agents and other contributors do not
change the environment; if it needs repair, they ask the developer to re-sync
it. See the [development guide](docs/development.md) for the complete workflow.

## Citation

Please cite the JOSS paper [FOXES: Farm Optimization and eXtended yield
Evaluation Software](https://doi.org/10.21105/joss.05464).

BibTeX:
```bibtex
@article{
    Schmidt2023,
    author = {Jonas Schmidt and Lukas Vollmer and Martin Dörenkämper and Bernhard Stoevesandt},
    title = {FOXES: Farm Optimization and eXtended yield Evaluation Software},
    doi = {10.21105/joss.05464},
    url = {https://doi.org/10.21105/joss.05464},
    year = {2023},
    publisher = {The Open Journal},
    volume = {8},
    number = {86},
    pages = {5464},
    journal = {Journal of Open Source Software}
}
```

## Acknowledgements

Development of *FOXES Optimization* was supported by:

- BMWK projects *Smart Wind Farms* (0325851B), *GW-Wakes* (0325397B), and *X-Wakes* (03EE3008A)
- BMBF project *H2Digital* (03SF0635)
- Horizon Europe projects *FLOW* (101084205) and *AIRE* (101083716)
