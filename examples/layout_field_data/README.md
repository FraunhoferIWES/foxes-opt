# foxes-opt example: _layout_field_data_

This example runs a pymoo-based layout optimization with heterogeneous ambient states that are read from multiple NetCDF files through `foxes.input.states.FieldData`.

## Generate the example data

The repository already contains five deterministic NetCDF input files in `data/`. To regenerate them, run:

```console
uv run python generate_data.py
```

## Check options

Inspect the command line options with:

```console
uv run python run_pymoo.py -h
```

## Run the optimization

Run the example with the default five-file pattern:

```console
uv run python run_pymoo.py
```

Run it without figures, for quicker smoke testing:

```console
uv run python run_pymoo.py --nofig -P 12 -G 3 -nt 6
```

Population vectorization is disabled by default in this example because `FieldData` states are evaluated reliably through pymoo's individual-evaluation path. You can still opt in explicitly:

```console
uv run python run_pymoo.py --vectorize
```
