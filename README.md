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

As a developer, first clone both repositories,
and then install via pip using the `-e` flag:

```console
git clone https://github.com/FraunhoferIWES/foxes.git
pip install -e foxes

git clone https://github.com/FraunhoferIWES/foxes-opt.git
pip install -e foxes-opt
```

If you want to contribute your developments, please replace
the above repository locations by your personal forks.

## Wind Rose Layout Snapshots

Pass `--write_layouts` to `examples/layout_wind_rose/run_pymoo.py` or
`examples/layout_wind_rose/run_slsqp.py` to enable `WriteLayoutCallback`;
intermediate output is disabled by default. The
callback writes the best first-objective individual of each completed optimizer
step to `examples/layout_wind_rose/results/`, regardless of the working directory.
The examples disable CSV output and write one image per generation or accepted
iteration.
Use `--layout_image_type` to select its file type; the default is `jpg`.
Each plot title shows the selected layout's first objective name and value.
Turbines associated with violated constraints are red; valid turbines are orange,
and each image includes an upper-left legend for these colors.
The examples write these snapshots without per-file log messages.
Early layouts may be infeasible; these are intermediate population snapshots,
not final results.
The callback options `write_csv` and `write_image` independently disable either
format; both default to `True`. The callback accepts both iteration and
evaluation events, so it can be used with all callback-enabled iwopy optimizers.
The SLSQP and IPOPT examples use `LocalFD` for finite-difference gradients.
The IPOPT example runs through pygmo and does not offer callbacks because pygmo
exposes neither exact live IPOPT iterations nor their decision vectors.
Running the example again overwrites matching generation numbers.
The existing `results/` rule in `.gitignore` excludes these generated files.

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
