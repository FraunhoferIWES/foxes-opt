from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from foxes import Turbine, WindFarm
from foxes.output import FarmLayoutOutput

from foxes_opt.pipelines import LayoutPipeline


def _write_plot(tmp_path, monkeypatch, **kwargs):
    farm = WindFarm()
    farm.add_turbine(Turbine([0.0, 0.0], D=100.0, H=90.0))
    farm.add_turbine(Turbine([500.0, 0.0], D=100.0, H=90.0))
    pipeline = LayoutPipeline(tmp_path, {}, 2, [], None)
    plot_args = {}
    algo = SimpleNamespace(farm=farm)

    def capture_plot(output, **plot_kwargs):
        plot_args.update(plot_kwargs)
        plot_args["uses_algorithm"] = output.algo is algo

    monkeypatch.setattr(FarmLayoutOutput, "get_figure", capture_plot)
    monkeypatch.setattr(plt, "savefig", lambda *args, **savefig_kwargs: None)
    try:
        pipeline.write_layout_plot(
            algo,
            SimpleNamespace(),
            title="Layout",
            table_index=0,
            show_flow=False,
            **kwargs,
        )
    finally:
        plt.close("all")
    return plot_args


def test_layout_plot_matches_callback_style(tmp_path, monkeypatch):
    plot_args = _write_plot(tmp_path, monkeypatch)

    assert plot_args["annotate"] == 0
    assert plot_args["uses_algorithm"] is True
    assert plot_args["true_turbine_radii"] is True
    assert list(plot_args["c"]) == ["tab:blue", "tab:blue"]
    np.testing.assert_array_equal(plot_args["edgecolors"], plot_args["c"])
    assert plot_args["linewidths"] == 0.4
    assert plot_args["zorder"] == 5
    assert plot_args["legend_labels"] == {
        "tab:blue": "Valid turbine",
        "red": "Constraint violation",
    }


def test_layout_plot_style_can_be_overridden(tmp_path, monkeypatch):
    plot_args = _write_plot(
        tmp_path,
        monkeypatch,
        true_turbine_radii=False,
        c="green",
        edgecolors="black",
        linewidths=2.0,
        zorder=7,
        legend_labels=None,
    )

    assert plot_args["true_turbine_radii"] is False
    assert plot_args["c"] == "green"
    assert plot_args["edgecolors"] == "black"
    assert plot_args["linewidths"] == 2.0
    assert plot_args["zorder"] == 7
    assert plot_args["legend_labels"] is None


def test_read_layout_index_supports_optimizer_snapshots(tmp_path):
    layout = np.array([[1.0, 2.0], [3.0, 4.0]])
    layout_dir = tmp_path / "iterations"
    layout_dir.mkdir()
    pd.DataFrame(
        {
            "index": [0, 1],
            "x": layout[:, 0],
            "y": layout[:, 1],
        }
    ).to_csv(layout_dir / "layout_00188.csv", index=False)
    pipeline = LayoutPipeline(tmp_path, {}, 2, [], None)

    result = pipeline.read_layout_index(188, "iterations")

    np.testing.assert_allclose(result, layout)


def test_read_layout_index_rejects_missing_snapshot(tmp_path):
    pipeline = LayoutPipeline(tmp_path, {}, 2, [], None)

    with pytest.raises(FileNotFoundError, match="No layout with index 188"):
        pipeline.read_layout_index(188, "iterations")


def test_read_layout_index_rejects_ambiguous_numeric_suffix(tmp_path):
    layout_dir = tmp_path / "iterations"
    layout_dir.mkdir()
    layout = pd.DataFrame({"index": [0, 1], "x": [1.0, 2.0], "y": [3.0, 4.0]})
    layout.to_csv(layout_dir / "layout_188.csv", index=False)
    layout.to_csv(layout_dir / "layout_00188.csv", index=False)
    pipeline = LayoutPipeline(tmp_path, {}, 2, [], None)

    with pytest.raises(ValueError, match="Multiple layouts with index 188"):
        pipeline.read_layout_index(188, "iterations")


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (
            pd.DataFrame({"index": [1, 0], "x": [1.0, 2.0], "y": [3.0, 4.0]}),
            "Invalid turbine indices",
        ),
        (pd.DataFrame({"index": [0, 1], "x": [1.0, 2.0]}), "requires x and y columns"),
        (
            pd.DataFrame({"index": [0, 1], "x": [1.0, np.inf], "y": [3.0, 4.0]}),
            "Non-finite coordinates",
        ),
        (pd.DataFrame({"x": [1.0], "y": [3.0]}), "Invalid layout coordinates shape"),
    ],
)
def test_read_layout_index_rejects_malformed_snapshot(tmp_path, data, message):
    layout_dir = tmp_path / "iterations"
    layout_dir.mkdir()
    data.to_csv(layout_dir / "layout_00188.csv", index=False)
    pipeline = LayoutPipeline(tmp_path, {}, 2, [], None)

    with pytest.raises(ValueError, match=message):
        pipeline.read_layout_index(188, "iterations")


def test_run_passes_restart_layout_to_selected_stage(tmp_path, monkeypatch):
    layout = np.array([[1.0, 2.0], [3.0, 4.0]])
    layout_dir = tmp_path / "iterations"
    layout_dir.mkdir()
    pd.DataFrame(
        {
            "index": [0, 1],
            "x": layout[:, 0],
            "y": layout[:, 1],
        }
    ).to_csv(layout_dir / "layout_00188.csv", index=False)
    pipeline = LayoutPipeline(tmp_path, {}, 2, [], None)
    calls = []

    def run_pipeline(self, **kwargs):
        calls.append(kwargs)
        return True, kwargs["initial_results"]

    monkeypatch.setattr(
        "foxes_opt.pipelines.layout_pipelines.Pipeline.run",
        run_pipeline,
    )

    success, results = pipeline.run(
        start_stage=1,
        restart_layout_index=188,
        restart_layout_dir="iterations",
        verbosity=0,
    )

    assert success
    np.testing.assert_allclose(calls[0]["initial_results"], layout)
    np.testing.assert_allclose(results[0], layout)
    assert results[1] is None


def test_run_passes_initial_layout_to_first_stage(tmp_path, monkeypatch):
    layout = np.array([[1.0, 2.0], [3.0, 4.0]])
    pipeline = LayoutPipeline(tmp_path, {}, 2, [], None)
    calls = []

    def run_pipeline(self, **kwargs):
        calls.append(kwargs)
        return True, kwargs["initial_results"]

    monkeypatch.setattr(
        "foxes_opt.pipelines.layout_pipelines.Pipeline.run",
        run_pipeline,
    )

    success, results = pipeline.run(initial_layout=layout, verbosity=0)

    assert success
    np.testing.assert_allclose(calls[0]["initial_results"], layout)
    np.testing.assert_allclose(results[0], layout)
    assert results[1] is None


def test_run_rejects_conflicting_initial_layouts(tmp_path):
    pipeline = LayoutPipeline(tmp_path, {}, 2, [], None)
    layout = np.array([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(ValueError, match="mutually exclusive"):
        pipeline.run(
            initial_layout=layout,
            restart_layout_index=0,
            verbosity=0,
        )


def test_run_rejects_non_finite_initial_layout(tmp_path):
    pipeline = LayoutPipeline(tmp_path, {}, 2, [], None)
    layout = np.array([[1.0, 2.0], [3.0, np.nan]])

    with pytest.raises(ValueError, match="non-finite"):
        pipeline.run(initial_layout=layout, verbosity=0)
