from pathlib import Path

import foxes
import foxes.variables as FV
import numpy as np
import foxes_opt.pipelines.stages.layout_optimizer as layout_optimizer
from foxes_opt.constraints import MinDistConstraint

from foxes_opt.pipelines import LayoutPipeline
from foxes_opt.pipelines.stages import LayoutOptimizerStage, RandomSubsetStage


class _States:
    def size(self):
        return 7


class _Pipeline:
    def __init__(self, stage, base_dir):
        self.stage = stage
        self.base_dir = Path(base_dir)
        self.n_turbines = 5
        self.states = _States()
        self.farm_boundary = object()

    def find_stage(self, name):
        return 0 if name == self.stage.name else -1

    def read_layout(self, results):
        return results


class _TestStage(RandomSubsetStage):
    def __init__(self, accepted, **kwargs):
        super().__init__(**kwargs)
        self.accepted = accepted
        self.calls = []
        self.writes = []

    def _optimize_step(self, layout_xy, turbine_indices, state_indices, verbosity):
        call_index = len(self.calls)
        step = call_index % len(self.accepted)
        self.calls.append(
            (
                layout_xy.copy(),
                turbine_indices.copy(),
                state_indices.copy(),
            )
        )
        candidate = layout_xy.copy()
        candidate[turbine_indices] += step + 1
        return self.accepted[step], candidate

    def _write_step(self, layout_xy, step, layout_plot_pars, verbosity):
        self.writes.append((step, layout_xy.copy()))


class _OptimizerStage(LayoutOptimizerStage):
    def __init__(self, result_success, **kwargs):
        super().__init__(optimizer_type="test", **kwargs)
        self.result_success = result_success
        self.start_layout = None

    def _run_layout_optimizer(
        self, layout_xy, states=None, sel_turbines=None, verbosity=1
    ):
        self.start_layout = layout_xy.copy()
        candidate = layout_xy + 1.0
        results = type("Results", (), {"success": self.result_success})()
        return results, candidate


def _stage(tmp_path, accepted=(True, False, True), **kwargs):
    pars = dict(
        n_subset_turbines=2,
        n_subset_states=3,
        n_steps=3,
        optimizer_type="test",
        seed=42,
    )
    pars.update(kwargs)
    stage = _TestStage(accepted=accepted, **pars)
    stage.initialize(_Pipeline(stage, tmp_path))
    return stage


def test_stage_stops_on_failed_step_without_mutation(tmp_path):
    stage = _stage(tmp_path)
    initial = np.zeros((5, 2))

    success, first = stage.run(prev_results=initial, verbosity=0)
    first_calls = stage.calls.copy()
    stage_again = _stage(tmp_path)
    success_again, second = stage_again.run(prev_results=initial, verbosity=0)
    second_calls = stage_again.calls

    assert not success and not success_again
    np.testing.assert_allclose(first, second)
    for (_, first_turbines, first_states), (_, second_turbines, second_states) in zip(
        first_calls, second_calls
    ):
        np.testing.assert_array_equal(first_turbines, second_turbines)
        np.testing.assert_array_equal(first_states, second_states)
        assert len(np.unique(first_turbines)) == 2
        assert len(np.unique(first_states)) == 3

    expected_before_rejected = initial.copy()
    expected_before_rejected[first_calls[0][1]] += 1
    np.testing.assert_allclose(first_calls[1][0], expected_before_rejected)
    assert len(first_calls) == 2


def test_stage_disables_turbine_and_state_subsets_with_none(tmp_path):
    stage = _stage(
        tmp_path,
        n_subset_turbines=None,
        n_subset_states=None,
    )

    turbine_indices, state_indices = stage._sample_subsets(np.random.default_rng(42))

    assert turbine_indices is None
    assert state_indices is None


def test_random_subset_stage_accepts_selected_problem_type(tmp_path):
    stage = RandomSubsetStage(
        n_subset_turbines=2,
        n_subset_states=3,
        n_steps=1,
        optimizer_type="test",
        problem_type="CustomProblem",
    )
    stage.initialize(_Pipeline(stage, tmp_path))

    assert stage.problem_type == "CustomProblem"


def test_stage_writes_only_accepted_steps_when_enabled(tmp_path):
    stage = _stage(tmp_path, write_step_results=True)

    success, _ = stage.run(prev_results=np.zeros((5, 2)), verbosity=0)

    assert not success
    assert [step for step, _ in stage.writes] == [0]


def test_stage_skips_intermediate_outputs_by_default(tmp_path):
    stage = _stage(tmp_path)

    success, _ = stage.run(prev_results=np.zeros((5, 2)), verbosity=0)

    assert not success
    assert stage.writes == []


def test_layout_optimizer_stage_uses_previous_layout(tmp_path):
    stage = _OptimizerStage(result_success=True)
    pipeline = _Pipeline(stage, tmp_path)
    stage.initialize(pipeline)
    layout = np.zeros((pipeline.n_turbines, 2))

    success, result = stage.run(prev_results=layout, verbosity=0)

    assert success
    np.testing.assert_allclose(stage.start_layout, layout)
    np.testing.assert_allclose(result, layout + 1.0)


def test_layout_optimizer_stage_keeps_previous_layout_on_failure(tmp_path):
    stage = _OptimizerStage(result_success=False)
    pipeline = _Pipeline(stage, tmp_path)
    stage.initialize(pipeline)
    layout = np.zeros((pipeline.n_turbines, 2))

    success, result = stage.run(prev_results=layout, verbosity=0)

    assert not success
    np.testing.assert_allclose(result, layout)


def test_layout_optimizer_stage_installs_default_functions(monkeypatch, tmp_path):
    stage = LayoutOptimizerStage(optimizer_type="test")
    stage.initialize(_Pipeline(stage, tmp_path))
    calls = []

    class _Problem:
        farm = type("Farm", (), {"n_turbines": 5})()

        def add_objective(self, objective):
            calls.append(("objective", objective))

        def add_constraint(self, constraint):
            calls.append(("constraint", constraint))

    monkeypatch.setattr(
        layout_optimizer,
        "MaxFarmREWS",
        lambda problem, sel_turbines: ("max_rews", problem, sel_turbines),
    )
    monkeypatch.setattr(
        layout_optimizer,
        "FarmBoundaryConstraint",
        lambda problem: ("boundary", problem),
    )

    def fake_new(constraint_type, *args, **kwargs):
        assert constraint_type == "MinDistConstraint"
        return (
            "min_dist",
            kwargs["problem"],
            kwargs["min_dist"],
            kwargs["min_dist_unit"],
        )

    monkeypatch.setattr(layout_optimizer.FarmConstraint, "new", fake_new)

    problem = _Problem()
    stage._add_functions(problem)

    assert calls == [
        ("objective", ("max_rews", problem, [0, 1, 2, 3, 4])),
        ("constraint", ("boundary", problem)),
        ("constraint", ("min_dist", problem, 2.5, "D")),
    ]


def test_layout_optimizer_stage_skips_min_dist_when_none(monkeypatch, tmp_path):
    stage = LayoutOptimizerStage(optimizer_type="test", min_dist=None)
    stage.initialize(_Pipeline(stage, tmp_path))
    calls = []

    class _Problem:
        farm = type("Farm", (), {"n_turbines": 5})()

        def add_objective(self, objective):
            calls.append(("objective", objective))

        def add_constraint(self, constraint):
            calls.append(("constraint", constraint))

    monkeypatch.setattr(layout_optimizer, "MaxFarmREWS", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        layout_optimizer, "FarmBoundaryConstraint", lambda problem: None
    )
    monkeypatch.setattr(
        layout_optimizer.FarmConstraint,
        "new",
        lambda *args, **kwargs: calls.append(("unexpected", args, kwargs)),
    )

    stage._add_functions(_Problem())

    assert [call[0] for call in calls] == ["objective", "constraint"]


def test_min_dist_constraint_can_check_only_selected_pairs():
    class _Problem:
        sel_turbines = [0, 2]
        farm = type("Farm", (), {"n_turbines": 4})()

        def tvar(self, var, ti):
            return f"{var}_{ti}"

        def var_names_int(self):
            return []

        def var_names_float(self):
            return [
                self.tvar(FV.X, 0),
                self.tvar(FV.Y, 0),
                self.tvar(FV.X, 2),
                self.tvar(FV.Y, 2),
            ]

    default_constraint = MinDistConstraint(_Problem(), min_dist=1.0)
    default_constraint.initialize(verbosity=0)
    selected_only_constraint = MinDistConstraint(
        _Problem(),
        min_dist=1.0,
        check_only_selected=True,
    )
    selected_only_constraint.initialize(verbosity=0)

    assert default_constraint._i2t.tolist() == [
        [0, 1],
        [0, 2],
        [0, 3],
        [2, 1],
        [2, 3],
    ]
    assert selected_only_constraint._i2t.tolist() == [[0, 2]]


def test_layout_optimizer_stage_uses_selected_problem_type(monkeypatch, tmp_path):
    stage = LayoutOptimizerStage(
        optimizer_type="test",
        problem_type="CustomProblem",
        problem_pars={"custom": 3},
    )
    pipeline = _Pipeline(stage, tmp_path)
    stage.initialize(pipeline)
    factory_calls = []

    class _Algo:
        initialized = False
        running = False

    class _Problem:
        sel_turbines = [0]

        def initialize(self, verbosity):
            pass

    class _Results:
        success = True
        vars_float = np.array([3.0, 4.0])

    class _Optimizer:
        def initialize(self, verbosity):
            pass

        def solve(self, verbosity):
            return _Results()

        def finalize(self, results, verbosity):
            pass

    def new_problem(cls, problem_type, **kwargs):
        factory_calls.append((problem_type, kwargs))
        return _Problem()

    algo = _Algo()
    monkeypatch.setattr(
        layout_optimizer.FarmOptProblem,
        "new",
        classmethod(new_problem),
    )
    monkeypatch.setattr(pipeline, "get_algo", lambda **kwargs: algo, raising=False)
    monkeypatch.setattr(stage, "_add_functions", lambda problem: None)
    monkeypatch.setattr(stage, "_prepare_problem", lambda problem: problem)
    monkeypatch.setattr(
        stage,
        "_create_optimizer",
        lambda problem, optimizer_type=None, optimizer_pars=None: _Optimizer(),
    )

    _, candidate = stage._run_layout_optimizer(np.zeros((5, 2)), verbosity=0)

    assert factory_calls == [
        (
            "CustomProblem",
            {
                "name": "layout_optimizer_problem",
                "algo": algo,
                "sel_turbines": None,
                "custom": 3,
            },
        )
    ]
    np.testing.assert_allclose(candidate[0], [3.0, 4.0])


def test_layout_optimizer_stage_boundary_repair_reduces_main_selection(
    monkeypatch, tmp_path
):
    stage = LayoutOptimizerStage(
        optimizer_type="test",
        optimizer_pars={"main": True},
        boundary_repair=True,
        boundary_repair_optimizer_type="repair_test",
        boundary_repair_optimizer_pars={"repair": True},
        min_dist=0.25,
        min_dist_unit="m",
    )

    class _Boundary:
        def points_distance(self, xy):
            return np.abs(np.abs(xy[:, 0]) - 1.0)

        def points_inside(self, xy):
            return np.abs(xy[:, 0]) < 1.0

    class _PipelineWithBoundary(_Pipeline):
        def __init__(self, stage, base_dir):
            super().__init__(stage, base_dir)
            self.farm_boundary = _Boundary()

    class _Algo:
        initialized = False
        running = False

    class _Problem:
        n_vars_int = 0
        farm = type("Farm", (), {"n_turbines": 3})()

        def __init__(self, sel_turbines):
            self.sel_turbines = sel_turbines

        def add_objective(self, objective):
            pass

        def add_constraint(self, constraint):
            pass

        def initialize(self, verbosity):
            pass

    class _Optimizer:
        def __init__(self, problem):
            self.problem = problem

        def initialize(self, verbosity):
            pass

        def solve(self, verbosity):
            values = {
                (1, 2): np.array([0.5, 0.0, 0.6, 0.0]),
                (0,): np.array([10.0, 0.0]),
            }[tuple(self.problem.sel_turbines)]
            return type(
                "Results",
                (),
                {"success": True, "vars_int": None, "vars_float": values},
            )()

        def finalize(self, results, verbosity):
            pass

    pipeline = _PipelineWithBoundary(stage, tmp_path)
    stage.initialize(pipeline)
    factory_calls = []
    constraint_calls = []
    optimizer_calls = []

    def new_problem(cls, problem_type, **kwargs):
        factory_calls.append((problem_type, kwargs.copy()))
        return _Problem(kwargs["sel_turbines"])

    def fake_new(constraint_type, *args, **kwargs):
        constraint_calls.append((constraint_type, kwargs.copy()))
        return (constraint_type, kwargs)

    monkeypatch.setattr(
        layout_optimizer.FarmOptProblem,
        "new",
        classmethod(new_problem),
    )
    monkeypatch.setattr(
        layout_optimizer, "FarmBoundaryConstraint", lambda problem: None
    )
    monkeypatch.setattr(layout_optimizer, "MaxFarmREWS", lambda *args, **kwargs: None)
    monkeypatch.setattr(layout_optimizer.FarmConstraint, "new", fake_new)
    monkeypatch.setattr(pipeline, "get_algo", lambda **kwargs: _Algo(), raising=False)
    monkeypatch.setattr(
        stage,
        "_create_optimizer",
        lambda problem, optimizer_type=None, optimizer_pars=None: (
            optimizer_calls.append((optimizer_type, optimizer_pars)),
            _Optimizer(problem),
        )[1],
    )

    layout = np.array([[0.0, 0.0], [2.0, 0.0], [0.9, 0.0]])
    results, candidate = stage._run_layout_optimizer(layout, verbosity=0)

    assert results.success
    assert [call[1]["sel_turbines"] for call in factory_calls] == [[1, 2], [0]]
    assert factory_calls[0][1]["name"] == "layout_optimizer_boundary_repair_problem"
    assert factory_calls[1][1]["name"] == "layout_optimizer_problem"
    assert constraint_calls[0][1]["check_only_selected"]
    assert optimizer_calls == [("repair_test", {"repair": True}), (None, None)]
    np.testing.assert_allclose(candidate, [[10.0, 0.0], [0.5, 0.0], [0.6, 0.0]])


def test_layout_optimizer_stage_merges_boundary_repair_optimizer_settings(monkeypatch):
    stage = LayoutOptimizerStage(
        optimizer_type="test",
        optimizer_pars={"common": "main", "main": True},
        boundary_repair_optimizer_type="repair_test",
        boundary_repair_optimizer_pars={"common": "repair", "repair": True},
    )
    problem = object()
    calls = []

    def fake_new(*args, **kwargs):
        calls.append((args, kwargs))
        return "optimizer"

    monkeypatch.setattr(layout_optimizer.Optimizer, "new", fake_new)

    optimizer = stage._create_optimizer(
        problem,
        stage.boundary_repair_optimizer_type,
        stage.boundary_repair_optimizer_pars,
    )

    assert optimizer == "optimizer"
    assert calls == [
        (
            (),
            {
                "problem": problem,
                "optimizer_type": "repair_test",
                "common": "repair",
                "main": True,
                "repair": True,
            },
        )
    ]


def test_layout_optimizer_stage_boundary_repair_uses_one_d_without_min_dist(
    monkeypatch, tmp_path
):
    stage = LayoutOptimizerStage(
        optimizer_type="test",
        boundary_repair=True,
        min_dist=None,
    )

    class _Boundary:
        def points_distance(self, xy):
            return np.array([99.0, 199.0, 301.0])

        def points_inside(self, xy):
            return np.ones(len(xy), dtype=bool)

    class _PipelineWithBoundary(_Pipeline):
        def __init__(self, stage, base_dir):
            super().__init__(stage, base_dir)
            self.farm_boundary = _Boundary()

    class _Farm:
        def get_rotor_diameters(self, algo):
            return np.array([100.0, 200.0, 300.0])

    class _Algo:
        initialized = True
        running = False
        farm = _Farm()

        def finalize(self):
            pass

    pipeline = _PipelineWithBoundary(stage, tmp_path)
    stage.initialize(pipeline)
    monkeypatch.setattr(pipeline, "get_algo", lambda **kwargs: _Algo(), raising=False)

    turbines = stage._boundary_repair_turbines(
        np.zeros((3, 2)),
        states=None,
        sel_turbines=[0, 1, 2],
        verbosity=0,
    )

    assert turbines == [0, 1]


def test_layout_optimizer_stage_boundary_repair_failure_returns_original(
    monkeypatch, tmp_path
):
    stage = LayoutOptimizerStage(
        optimizer_type="test",
        boundary_repair=True,
        min_dist=0.25,
        min_dist_unit="m",
    )

    class _Boundary:
        def points_distance(self, xy):
            return np.maximum(np.abs(xy[:, 0]) - 1.0, 0.0)

        def points_inside(self, xy):
            return np.abs(xy[:, 0]) < 1.0

    class _PipelineWithBoundary(_Pipeline):
        def __init__(self, stage, base_dir):
            super().__init__(stage, base_dir)
            self.farm_boundary = _Boundary()

    class _Algo:
        initialized = False
        running = False

    class _Problem:
        n_vars_int = 0
        farm = type("Farm", (), {"n_turbines": 3})()

        def __init__(self, sel_turbines):
            self.sel_turbines = sel_turbines

        def add_objective(self, objective):
            pass

        def add_constraint(self, constraint):
            pass

        def initialize(self, verbosity):
            pass

    class _Optimizer:
        def __init__(self, problem):
            self.problem = problem

        def initialize(self, verbosity):
            pass

        def solve(self, verbosity):
            return type(
                "Results",
                (),
                {
                    "success": False,
                    "vars_int": None,
                    "vars_float": np.array([0.5, 0.0]),
                },
            )()

        def finalize(self, results, verbosity):
            pass

    pipeline = _PipelineWithBoundary(stage, tmp_path)
    stage.initialize(pipeline)

    def new_problem(cls, problem_type, **kwargs):
        return _Problem(kwargs["sel_turbines"])

    monkeypatch.setattr(
        layout_optimizer.FarmOptProblem,
        "new",
        classmethod(new_problem),
    )
    monkeypatch.setattr(
        layout_optimizer, "FarmBoundaryConstraint", lambda problem: None
    )
    monkeypatch.setattr(layout_optimizer, "MaxFarmREWS", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        layout_optimizer.FarmConstraint, "new", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(pipeline, "get_algo", lambda **kwargs: _Algo(), raising=False)
    monkeypatch.setattr(
        stage,
        "_create_optimizer",
        lambda problem, optimizer_type=None, optimizer_pars=None: _Optimizer(problem),
    )

    layout = np.array([[0.0, 0.0], [2.0, 0.0], [0.2, 0.0]])
    results, candidate = stage._run_layout_optimizer(layout, verbosity=0)

    assert not results.success
    np.testing.assert_allclose(candidate, layout)


def test_stage_runs_vectorized_gg_with_lazy_state_subset(tmp_path):
    states = foxes.input.states.StatesTable(
        data_source="wind_rose_bremen.csv",
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "ws", FV.WD: "wd", FV.WEIGHT: "weight"},
        fixed_vars={FV.RHO: 1.225, FV.TI: 0.04},
    )
    stage = RandomSubsetStage(
        n_subset_turbines=1,
        n_subset_states=2,
        n_steps=1,
        optimizer_type="GG",
        optimizer_pars={
            "step_max": 20.0,
            "step_min": 10.0,
            "n_max_steps": 2,
            "max_iterations": 0,
            "vectorized": True,
        },
        problem_wrapper_type="LocalFD",
        problem_wrapper_pars={"deltas": 0.1, "fd_order": 1},
        min_dist=2.0,
        seed=7,
    )
    pipeline = LayoutPipeline(
        base_dir=tmp_path,
        algo_pars={
            "algo_type": "Downwind",
            "rotor_model": "centre",
            "wake_models": ["Bastankhah2014_linear_lim_k004"],
        },
        n_turbines=3,
        turbine_models=["NREL5MW"],
        farm_boundary=foxes.utils.geom2d.Circle([0.0, 0.0], 1000.0),
        states=states,
    )
    pipeline.add_stage(stage)
    stage.initialize(pipeline)
    layout = np.array([[-400.0, 0.0], [0.0, 0.0], [400.0, 0.0]])
    stage._ensure_state_count(layout)
    selected, _ = stage._sample_subsets(np.random.default_rng(stage.seed))

    with foxes.Engine.new("single"):
        success, result = stage.run(prev_results=layout, verbosity=0)

    fixed = np.setdiff1d(np.arange(pipeline.n_turbines), selected)
    assert success
    assert result.shape == layout.shape
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result[fixed], layout[fixed])
    assert np.all(pipeline.farm_boundary.points_inside(result))
