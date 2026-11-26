# Installation

FOXES Optimization supports Python 3.10 through 3.14.
It requires FOXES >=1.10.0 and iwopy >=1.0.0.

## pip

Install the latest release from [PyPI](https://pypi.org/project/foxes-opt/):

```console
pip install foxes-opt
```

Alternatively, install it through the FOXES optimization extra:

```console
pip install "foxes[opt]"
```

## conda

Install the latest release from
[conda-forge](https://anaconda.org/conda-forge/foxes-opt):

```console
conda install foxes-opt -c conda-forge
```

## Optional dependencies

| Extra | Use |
|---|---|
| `doc` | Documentation tools and notebooks |
| `dev` | Development tools |
| `test` | Test tools and notebook execution |

Install one or more extras with bracket syntax:

```console
pip install "foxes-opt[doc,test]"
```

## Development setup

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
it. See the
[development guide](https://github.com/FraunhoferIWES/foxes-opt/blob/main/docs/development.md)
for the complete workflow, including documentation environment setup.
