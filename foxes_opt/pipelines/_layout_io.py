from pathlib import Path

import numpy as np
import pandas as pd


def read_layout_index(
    base_dir: Path,
    pipeline_name: str,
    n_turbines: int,
    layout_index: int,
    layout_dir: str | Path,
) -> np.ndarray:
    """Read and validate a persisted layout by numeric file suffix."""
    if isinstance(layout_index, bool) or not isinstance(
        layout_index, (int, np.integer)
    ):
        raise TypeError("layout_index must be an integer")
    if layout_index < 0:
        raise ValueError("layout_index must be non-negative")

    directory = Path(layout_dir)
    if not directory.is_absolute():
        directory = base_dir / directory
    matches = []
    for path in directory.glob("layout_*.csv"):
        try:
            index = int(path.stem.removeprefix("layout_"))
        except ValueError:
            continue
        if index == layout_index:
            matches.append(path)
    if not matches:
        raise FileNotFoundError(
            f"{pipeline_name}: No layout with index {layout_index} in {directory}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"{pipeline_name}: Multiple layouts with index {layout_index} in "
            f"{directory}: {sorted(matches)}"
        )

    data = pd.read_csv(matches[0])
    columns = {str(column).lower(): column for column in data.columns}
    if "index" in columns:
        indices = data[columns["index"]].to_numpy()
        if not np.array_equal(indices, np.arange(n_turbines)):
            raise ValueError(
                f"{pipeline_name}: Invalid turbine indices in {matches[0]}"
            )
    try:
        layout_xy = data[[columns["x"], columns["y"]]].to_numpy(dtype=float)
    except KeyError as error:
        raise ValueError(
            f"{pipeline_name}: Layout {matches[0]} requires x and y columns"
        ) from error
    if not np.all(np.isfinite(layout_xy)):
        raise ValueError(f"{pipeline_name}: Non-finite coordinates in {matches[0]}")
    expected_shape = (n_turbines, 2)
    if layout_xy.shape != expected_shape:
        raise ValueError(
            f"Invalid layout coordinates shape, got {layout_xy.shape}, "
            f"expected {expected_shape}."
        )
    return layout_xy
