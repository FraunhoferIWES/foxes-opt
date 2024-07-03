# Installation

There are multiple ways to install *foxes-opt*.

## Installation as standard user 

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

## Installation as developer

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
