import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def build_dataset(
    file_index,
    n_states_per_file=10,
    state_start=np.datetime64("2024-01-01T00:00:00"),
    state_step=np.timedelta64(10, "m"),
):
    """Create one deterministic heterogeneous field dataset."""
    x = np.array([-200.0, 150.0, 500.0, 850.0, 1200.0], dtype=np.float64)
    y = np.array([-300.0, 50.0, 400.0, 750.0, 1100.0, 1450.0], dtype=np.float64)
    state_offsets = np.arange(n_states_per_file, dtype=np.float64)
    state0 = file_index * n_states_per_file
    state = state_start + (state0 + state_offsets.astype(np.int64)) * state_step

    x_grid = x[None, None, :]
    y_grid = y[None, :, None]
    state_grid = state_offsets[:, None, None]

    ws = 8.0 + 0.35 * file_index + 0.25 * state_grid
    ws = ws + 0.0012 * (x_grid - 400.0) + 0.0008 * (y_grid - 500.0)

    wd = 245.0 + 6.0 * file_index + 4.0 * state_grid
    wd = wd + 0.004 * (x_grid - 400.0) - 0.003 * (y_grid - 500.0)

    return xr.Dataset(
        data_vars={
            "ws": (("state", "y", "x"), ws),
            "wd": (("state", "y", "x"), wd),
        },
        coords={"state": state, "y": y, "x": x},
        attrs={"description": "Synthetic FieldData states for layout optimization."},
    )


def write_datasets(
    output_dir,
    n_files=5,
    n_states_per_file=10,
    state_start=np.datetime64("2024-01-01T00:00:00"),
    state_step=np.timedelta64(10, "m"),
):
    """Write the requested number of datasets to NetCDF files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_index in range(n_files):
        dataset = build_dataset(
            file_index,
            n_states_per_file=n_states_per_file,
            state_start=state_start,
            state_step=state_step,
        )
        file_path = output_dir / f"states_{file_index}.nc"
        dataset.to_netcdf(file_path)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-dir",
        default=Path(__file__).resolve().parent / "data",
        type=Path,
        help="The output directory for the generated NetCDF files.",
    )
    parser.add_argument(
        "-n",
        "--n-files",
        default=5,
        type=int,
        help="The number of NetCDF files to generate.",
    )
    parser.add_argument(
        "-ns",
        "--n-states-per-file",
        default=10,
        type=int,
        help="The number of states to generate per NetCDF file.",
    )
    parser.add_argument(
        "--state-start",
        default="2024-01-01T00:00:00",
        help="The timestamp of the first generated state.",
    )
    parser.add_argument(
        "--state-step-minutes",
        default=10,
        type=int,
        help="The time step between generated states in minutes.",
    )
    return parser.parse_args()


def main():
    """Generate deterministic NetCDF files for the example."""
    args = parse_args()
    state_start = np.datetime64(args.state_start)
    state_step = np.timedelta64(args.state_step_minutes, "m")
    write_datasets(
        args.output_dir,
        n_files=args.n_files,
        n_states_per_file=args.n_states_per_file,
        state_start=state_start,
        state_step=state_step,
    )


if __name__ == "__main__":
    main()