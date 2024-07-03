# foxes-opt

The package *foxes-opt* provides optimization functionality for the
*Farm Optimization and eXtended yield Evaluation Software* [foxes](https://github.com/FraunhoferIWES/foxes)
and is based on the optimization interface [iwopy](https://github.com/FraunhoferIWES/iwopy).

All three open-source Python packages *foxes*, *foxes-opt* and *iwopy* are provided and maintained by Fraunhofer IWES.

The calculation is fully vectorized and its fast performance is owed to [dask](https://www.dask.org/). Also the parallelization on local or remote clusters is enabled via `dask`. The wind farm
optimization capabilities invoke the [iwopy](https://github.com/FraunhoferIWES/iwopy) package which
as well supports vectorization.

`foxes` is build upon many years of experience with wake model code development at IWES, starting with the C++ based in-house code _flapFOAM_ (2011-2019) and the Python based direct predecessor _flappy_ (2019-2022).

Documentation: [https://fraunhoferiwes.github.io/foxes-opt/index.html](https://fraunhoferiwes.github.io/foxes-opt/index.html)

Source code: [https://github.com/FraunhoferIWES/foxes-opt](https://github.com/FraunhoferIWES/foxes-opt)

PyPi reference: [https://pypi.org/project/foxes-opt/](https://pypi.org/project/foxes-opt/)

Anaconda reference: [https://anaconda.org/conda-forge/foxes-opt](https://anaconda.org/conda-forge/foxes-opt)

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
