from types import SimpleNamespace

import matplotlib
import numpy as np

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