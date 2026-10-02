# Installation

There are multiple ways to install *foxes-opt*.
The supported Python versions are 3.10 through 3.14.

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
it. See the [development guide](../development.md) for the complete workflow.
