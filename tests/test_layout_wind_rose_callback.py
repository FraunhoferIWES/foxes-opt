import ast
import argparse
from pathlib import Path
import runpy
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import foxes
from iwopy.core import OptimizerCallbackData

from foxes_opt.callbacks import WriteLayoutCallback
from foxes_opt.problems.layout import FarmLayoutOptProblem


@pytest.mark.parametrize("script", ["run_pymoo.py", "run_slsqp.py"])
def test_example_callbacks_disable_csv(script):
    path = Path(__file__).parents[1] / "examples/layout_wind_rose" / script
    tree = ast.parse(path.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "WriteLayoutCallback"
    ]

    assert len(calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert ast.literal_eval(keywords["write_csv"]) is False
    assert ast.literal_eval(keywords["verbosity"]) == 0
    image_format = keywords["image_format"]
    assert isinstance(image_format, ast.Attribute)
    assert image_format.attr == "layout_image_type"


def test_layout_snapshots_are_opt_in(monkeypatch):
    class ParsingComplete(Exception):
        pass

    parse_args = argparse.ArgumentParser.parse_args

    def inspect_arguments(parser, *args, **kwargs):
        defaults = parse_args(parser, [])
        assert defaults.write_layouts is False
        assert defaults.layout_image_type == "jpg"
        assert parse_args(parser, ["--write_layouts"]).write_layouts is True
        assert (
            parse_args(parser, ["--layout_image_type", "png"]).layout_image_type
            == "png"
        )
        raise ParsingComplete

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", inspect_arguments)
    with pytest.raises(ParsingComplete):
        runpy.run_path(
            str(Path(__file__).parents[1] / "examples/layout_wind_rose/run_pymoo.py"),
            run_name="__main__",
        )


def test_slsqp_layout_snapshots_are_opt_in(monkeypatch):
    class ParsingComplete(Exception):
        pass

    parse_args = argparse.ArgumentParser.parse_args

    def inspect_arguments(parser, *args, **kwargs):
        defaults = parse_args(parser, [])
        enabled = parse_args(parser, ["--write_layouts"])
        assert defaults.write_layouts is False
        assert defaults.layout_image_type == "jpg"
        assert defaults.maxiter == 100
        assert enabled.write_layouts is True
        assert (
            parse_args(parser, ["--layout_image_type", "png"]).layout_image_type
            == "png"
        )
        raise ParsingComplete

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", inspect_arguments)
    with pytest.raises(ParsingComplete):
        runpy.run_path(
            str(Path(__file__).parents[1] / "examples/layout_wind_rose/run_slsqp.py"),
            run_name="__main__",
        )


@pytest.mark.parametrize("example", ["layout_wind_rose", "layout_single_state"])
def test_ipopt_example_uses_pygmo_ipopt(example):
    path = Path(__file__).parents[1] / "examples" / example / "run_ipopt.py"
    tree = ast.parse(path.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Optimizer_pygmo"
    ]

    assert len(calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    algo_pars = {
        ast.literal_eval(key): value
        for key, value in zip(
            keywords["algo_pars"].keys,
            keywords["algo_pars"].values,
        )
    }
    assert ast.literal_eval(algo_pars["type"]) == "ipopt"
    assert isinstance(algo_pars["max_iter"], ast.Attribute)
    assert algo_pars["max_iter"].attr == "maxiter"
    assert ast.literal_eval(algo_pars["print_level"]) == 5
    assert isinstance(algo_pars["tol"], ast.Attribute)
    assert algo_pars["tol"].attr == "tol"
    assert "setup_pars" not in keywords


def test_single_state_ipopt_finite_difference_parameters(monkeypatch):
    class ParsingComplete(Exception):
        pass

    parse_args = argparse.ArgumentParser.parse_args

    def inspect_arguments(parser, *args, **kwargs):
        defaults = parse_args(parser, [])
        custom = parse_args(parser, ["-O", "2", "--fd-delta", "25"])
        assert defaults.fd_order == 1
        assert defaults.fd_delta == 10.0
        assert custom.fd_order == 2
        assert custom.fd_delta == 25.0
        raise ParsingComplete

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", inspect_arguments)
    path = Path(__file__).parents[1] / "examples/layout_single_state/run_ipopt.py"
    with pytest.raises(ParsingComplete):
        runpy.run_path(str(path), run_name="__main__")

    tree = ast.parse(path.read_text())
    local_fd = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "LocalFD"
    )
    keywords = {keyword.arg: keyword.value for keyword in local_fd.keywords}
    assert isinstance(keywords["deltas"], ast.Attribute)
    assert keywords["deltas"].attr == "fd_delta"
    assert isinstance(keywords["fd_order"], ast.Attribute)
    assert keywords["fd_order"].attr == "fd_order"


@pytest.mark.parametrize("example", ["layout_wind_rose", "layout_single_state"])
def test_ipopt_example_has_no_callbacks(example):
    path = Path(__file__).parents[1] / "examples" / example / "run_ipopt.py"
    tree = ast.parse(path.read_text())
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    solve_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "solve"
    )

    assert "WriteLayoutCallback" not in names
    assert all(keyword.arg != "callbacks" for keyword in solve_call.keywords)
    solve_keywords = {keyword.arg: keyword.value for keyword in solve_call.keywords}
    assert ast.literal_eval(solve_keywords["verbosity"]) == 0


def test_ipopt_releases_figures_before_process_engine():
    path = Path(__file__).parents[1] / "examples/layout_wind_rose/run_ipopt.py"
    tree = ast.parse(path.read_text())
    engine_assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "engine"
            for target in node.targets
        )
    )
    collect_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "gc"
        and node.func.attr == "collect"
    )
    deleted_axes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Delete)
        and any(
            isinstance(target, ast.Name) and target.id == "ax"
            for target in node.targets
        )
    ]

    assert len(deleted_axes) == 2
    assert max(node.lineno for node in deleted_axes) < collect_call.lineno
    assert collect_call.lineno < engine_assignment.lineno


def _snapshot(
    tmp_path,
    wrapped=False,
    initial_variables_int=0,
    initial_variables_float=(10.0, 20.0, 30.0, 40.0),
    initial_objectives=None,
    initial_constraints=None,
    constraints=None,
    **kwargs,
):
    farm = foxes.WindFarm()
    farm.add_turbine(foxes.Turbine([10.0, 20.0], turbine_models=[], D=100, H=90))
    farm.add_turbine(foxes.Turbine([30.0, 40.0], turbine_models=[], D=100, H=90))
    problem = FarmLayoutOptProblem.__new__(FarmLayoutOptProblem)
    problem._sel_turbines = None
    problem._maximize = np.array([True])
    problem.objs = SimpleNamespace(component_names=["power"])
    problem.cons = SimpleNamespace(functions=constraints or [])
    problem.initial_values_int = lambda: initial_variables_int
    problem.initial_values_float = lambda: (
        None
        if initial_variables_float is None
        else np.asarray(initial_variables_float, dtype=float)
    )

    def evaluate_individual(vars_int, vars_float):
        assert vars_int.shape == (0,)
        return (
            np.asarray(initial_objectives or [1.0], dtype=float),
            np.asarray(initial_constraints or [], dtype=float),
        )

    problem.evaluate_individual = evaluate_individual
    problem.check_constraints_individual = lambda values: values <= 0.0
    turbine_types = [SimpleNamespace(name="test_type") for _ in farm.turbines]
    problem.algo = SimpleNamespace(
        farm=farm,
        farm_controller=SimpleNamespace(turbine_types=turbine_types),
    )
    callback = WriteLayoutCallback(tmp_path / "results", "layout", **kwargs)
    optimizer_problem = problem
    if wrapped:
        optimizer_problem = SimpleNamespace(
            base_problem=problem,
            n_vars_int=problem.n_vars_int,
            initial_values_int=problem.initial_values_int,
            initial_values_float=problem.initial_values_float,
            evaluate_individual=problem.evaluate_individual,
        )
    callback.initialize(SimpleNamespace(problem=optimizer_problem))
    return callback, farm


@pytest.fixture
def snapshot(tmp_path):
    return _snapshot(tmp_path)


def test_initial_snapshot_written_by_default(tmp_path):
    callback, _ = _snapshot(tmp_path, write_image=False)

    layout = pd.read_csv(callback.out_dir / "layout_00000.csv")
    np.testing.assert_allclose(layout[["x", "y"]], [[10.0, 20.0], [30.0, 40.0]])
    assert set(layout["turbine_type"]) == {"test_type"}


def test_initial_snapshot_evaluates_constraint_validity(tmp_path, monkeypatch):
    constraint = SimpleNamespace(
        n_components=lambda: 1,
        var_names_float=["X_0000", "Y_0000", "X_0001", "Y_0001"],
        vardeps_float=lambda: np.array([[False, False, True, True]]),
    )
    plot_args = {}

    def capture_plot(output, file_name, **kwargs):
        plot_args.update(kwargs)

    monkeypatch.setattr(foxes.output.FarmLayoutOutput, "write_plot", capture_plot)
    _snapshot(
        tmp_path,
        write_csv=False,
        initial_objectives=[1.5],
        initial_constraints=[1.0],
        constraints=[constraint],
    )

    assert plot_args["title"] == "power: 1.5"
    assert list(plot_args["c"]) == ["tab:blue", "red"]


def test_initial_snapshot_can_be_disabled(tmp_path):
    callback, _ = _snapshot(
        tmp_path,
        write_initial=False,
        initial_variables_float=None,
    )

    assert not list(callback.out_dir.iterdir())


def test_initial_snapshot_requires_initial_float_variables(tmp_path):
    with pytest.raises(ValueError, match="require initial float variables"):
        _snapshot(tmp_path, initial_variables_float=None)


def test_snapshot_step_offset_avoids_restart_collisions(tmp_path):
    callback, _ = _snapshot(
        tmp_path,
        write_initial=False,
        step_offset=188,
    )

    callback.notify(_data([[100.0, 200.0, 300.0, 400.0]], iteration=1))

    assert not (callback.out_dir / "layout_00001.csv").exists()
    assert (callback.out_dir / "layout_00189.csv").exists()


@pytest.mark.parametrize("step_offset", [True, 1.5])
def test_snapshot_rejects_non_integer_step_offset(tmp_path, step_offset):
    with pytest.raises(TypeError, match="step_offset must be an integer"):
        WriteLayoutCallback(tmp_path, "layout", step_offset=step_offset)


def test_snapshot_rejects_negative_step_offset(tmp_path):
    with pytest.raises(ValueError, match="step_offset must be non-negative"):
        WriteLayoutCallback(tmp_path, "layout", step_offset=-1)


def _data(
    values,
    iteration=1,
    event="iteration",
    objectives=None,
    n_evaluations=None,
    constraints=None,
):
    variables = np.asarray(values, dtype=float).reshape(len(values), 4)
    if objectives is None:
        objectives = np.arange(len(values), dtype=float)
    constraint_values = (
        np.empty((len(values), 0))
        if constraints is None
        else np.asarray(constraints, dtype=float).reshape(len(values), -1)
    )
    return OptimizerCallbackData(
        event=event,
        iteration=iteration,
        n_evaluations=n_evaluations,
        vars_int=np.empty((len(values), 0), dtype=int),
        vars_float=variables,
        objs=np.asarray(objectives, dtype=float).reshape(len(values), 1),
        cons=constraint_values,
    )


def test_snapshot_files_and_isolation(snapshot):
    callback, farm = snapshot
    initial_figures = plt.get_fignums()
    values = [[100.0, 200.0, 300.0, 400.0], [500.0, 600.0, 700.0, 800.0]]
    for generation in (1, 2):
        callback.notify(_data(values, iteration=generation, objectives=[1.0, 2.0]))
        basename = callback.out_dir / f"layout_{generation:05d}"
        layout = pd.read_csv(basename.with_suffix(".csv"))
        np.testing.assert_allclose(
            layout[["x", "y"]],
            np.asarray(values[1]).reshape(2, 2),
        )
        assert set(layout["turbine_type"]) == {"test_type"}
        image = plt.imread(callback.out_dir / "jpg" / basename.with_suffix(".jpg").name)
        assert image.size > 0 and np.ptp(image) > 0
    np.testing.assert_allclose(
        [turbine.xy for turbine in farm.turbines], [[10, 20], [30, 40]]
    )
    assert plt.get_fignums() == initial_figures


def test_snapshot_plot_title_contains_selected_objective(tmp_path, monkeypatch):
    callback, _ = _snapshot(tmp_path, write_csv=False)
    plot_args = {}

    def capture_title(output, file_name, **kwargs):
        plot_args.update(kwargs)

    monkeypatch.setattr(foxes.output.FarmLayoutOutput, "write_plot", capture_title)
    callback.notify(
        _data(
            [[100.0, 200.0, 300.0, 400.0], [500.0, 600.0, 700.0, 800.0]],
            objectives=[1.0, 2.5],
        )
    )

    assert plot_args["title"] == "power: 2.5"
    assert plot_args["annotate"] == 0
    assert plot_args["true_turbine_radii"] is True
    np.testing.assert_array_equal(plot_args["edgecolors"], plot_args["c"])
    assert plot_args["linewidths"] == 0.4
    assert plot_args["zorder"] == 5


def test_snapshot_marks_constraint_turbines_red(tmp_path, monkeypatch):
    callback, _ = _snapshot(tmp_path, write_csv=False)
    constraint = SimpleNamespace(
        n_components=lambda: 1,
        check_individual=lambda values, verbosity=0: values <= 0.0,
        var_names_float=["X_0000", "Y_0000", "X_0001", "Y_0001"],
        vardeps_float=lambda: np.array([[False, False, True, True]]),
    )
    callback._problem.cons = SimpleNamespace(
        functions=[constraint], n_components=lambda: 1
    )
    plot_args = {}

    def capture_colors(output, file_name, **kwargs):
        plot_args.update(kwargs)

    monkeypatch.setattr(foxes.output.FarmLayoutOutput, "write_plot", capture_colors)
    callback.notify(_data([[100.0, 200.0, 300.0, 400.0]], constraints=[[1.0]]))

    assert list(plot_args["c"]) == ["tab:blue", "red"]
    assert list(plot_args["edgecolors"]) == ["tab:blue", "red"]
    assert plot_args["legend_labels"] == {
        "tab:blue": "Valid turbine",
        "red": "Constraint violation",
    }


def test_snapshot_accepts_custom_validity_colors(tmp_path, monkeypatch):
    callback, _ = _snapshot(
        tmp_path,
        write_csv=False,
        valid_color="#00aa44",
        invalid_color="#cc00cc",
    )
    monkeypatch.setattr(callback, "_invalid_turbines", lambda data, selected: {1})
    plot_args = {}

    def capture_colors(output, file_name, **kwargs):
        plot_args.update(kwargs)

    monkeypatch.setattr(foxes.output.FarmLayoutOutput, "write_plot", capture_colors)
    callback.notify(_data([[100.0, 200.0, 300.0, 400.0]]))

    assert list(plot_args["c"]) == ["#00aa44", "#cc00cc"]
    assert list(plot_args["edgecolors"]) == ["#00aa44", "#cc00cc"]
    assert plot_args["legend_labels"] == {
        "#00aa44": "Valid turbine",
        "#cc00cc": "Constraint violation",
    }


def test_snapshot_supports_wrapped_problem(tmp_path):
    callback, _ = _snapshot(tmp_path, wrapped=True)
    callback.notify(_data([[100.0, 200.0, 300.0, 400.0]]))

    assert (callback.out_dir / "layout_00001.csv").exists()
    assert (callback.out_dir / "jpg" / "layout_00001.jpg").exists()


@pytest.mark.parametrize(
    ("write_csv", "write_image"),
    [(True, False), (False, True), (False, False)],
)
def test_snapshot_output_formats_can_be_disabled(tmp_path, write_csv, write_image):
    callback, _ = _snapshot(
        tmp_path,
        write_csv=write_csv,
        write_image=write_image,
    )
    callback.notify(_data([[100.0, 200.0, 300.0, 400.0]]))
    basename = callback.out_dir / "layout_00001"

    assert basename.with_suffix(".csv").exists() is write_csv
    assert (
        callback.out_dir / "jpg" / basename.with_suffix(".jpg").name
    ).exists() is write_image


def test_snapshot_accepts_custom_image_format(tmp_path):
    callback, _ = _snapshot(tmp_path, write_csv=False, image_format=".png")
    callback.notify(_data([[100.0, 200.0, 300.0, 400.0]]))

    assert (callback.out_dir / "png" / "layout_00001.png").exists()


def test_snapshot_uses_evaluation_count(snapshot):
    callback, _ = snapshot
    callback.notify(
        _data(
            [[100.0, 200.0, 300.0, 400.0]],
            event="evaluation",
            iteration=None,
            n_evaluations=7,
        )
    )
    assert (callback.out_dir / "layout_00007.csv").exists()
    assert (callback.out_dir / "jpg" / "layout_00007.jpg").exists()


def test_snapshot_skips_empty_population(snapshot):
    callback, _ = snapshot
    existing = set(callback.out_dir.rglob("*"))
    callback.notify(_data([]))
    assert set(callback.out_dir.rglob("*")) == existing


def test_snapshot_requires_generation(snapshot):
    callback, _ = snapshot
    existing = set(callback.out_dir.rglob("*"))
    with pytest.raises(ValueError, match="iteration or evaluation count"):
        callback.notify(_data([[1, 2, 3, 4]], iteration=None))
    assert set(callback.out_dir.rglob("*")) == existing


def test_snapshot_propagates_output_errors(tmp_path):
    callback, _ = _snapshot(tmp_path, write_initial=False)
    callback.out_dir.rmdir()
    callback.out_dir.write_text("not a directory")
    with pytest.raises(OSError):
        callback.notify(_data([[1, 2, 3, 4]]))


def test_snapshot_validates_generation_step(tmp_path):
    with pytest.raises(ValueError, match="at least 1"):
        WriteLayoutCallback(tmp_path, "layout", n_step=0)


def test_snapshot_validates_image_format(tmp_path):
    with pytest.raises(ValueError, match="file extension"):
        WriteLayoutCallback(tmp_path, "layout", image_format="../jpg")
