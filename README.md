# foxes-opt

The package *foxes-opt* provides optimization functionality for the
*Farm Optimization and eXtended yield Evaluation Software* [foxes](https://github.com/FraunhoferIWES/foxes)
and is based on the optimization interface [iwopy](https://github.com/FraunhoferIWES/iwopy).

All three open-source Python packages *foxes*, *foxes-opt* and *iwopy* are provided and maintained by Fraunhofer IWES.

The calculation is fully vectorized and its fast performance is owed to [dask](https://www.dask.org/). Also the parallelization on local or remote clusters is enabled via `dask`. The wind farm
optimization capabilities invoke the [iwopy](https://github.com/FraunhoferIWES/iwopy) package which
as well supports vectorization.

`foxes` is build upon many years of experience with wake model code development at IWES, starting with the C++ based in-house code _flapFOAM_ (2011-2019) and the Python based direct predecessor _flappy_ (2019-2022).

Documentation: [https://fraunhoferiwes.github.io/foxes-opt/](https://fraunhoferiwes.github.io/foxes-opt/)

Source code: [https://github.com/FraunhoferIWES/foxes-opt](https://github.com/FraunhoferIWES/foxes-opt)

PyPi reference: [https://pypi.org/project/foxes-opt/](https://pypi.org/project/foxes-opt/)

Anaconda reference: [https://anaconda.org/conda-forge/foxes-opt](https://anaconda.org/conda-forge/foxes-opt)

## Requirements

The supported Python versions are `Python 3.10`...`3.14`.

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
it. See `docs/development.md` for the complete workflow.

## Wind Rose Layout Snapshots

Pass `--write_layouts` to `examples/layout_wind_rose/run_pymoo.py` or
`examples/layout_wind_rose/run_slsqp.py` to enable `WriteLayoutCallback`;
intermediate output is disabled by default. Images are written under
`examples/layout_wind_rose/results/`; `--layout_image_type` selects the format
(default: `jpg`). See the
[callback guide](https://fraunhoferiwes.github.io/foxes-opt/api_callbacks.html)
for output options, initial snapshots, history, and restart behavior.

## Citation

Please cite the JOSS paper [FOXES: Farm Optimization and eXtended yield
Evaluation Software](https://doi.org/10.21105/joss.05464).

Bibtex:
```
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
