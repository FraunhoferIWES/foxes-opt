from pathlib import Path

import foxes
import foxes.variables as FV
import numpy as np
import foxes_opt.pipelines.stages.layout_optimizer as layout_optimizer

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


def test_stage_repeats_seeded_sampling_and_rejects_without_mutation(tmp_path):
    stage = _stage(tmp_path)
    initial = np.zeros((5, 2))

    success, first = stage.run(prev_results=initial, verbosity=0)
    first_calls = stage.calls.copy()
    success_again, second = stage.run(prev_results=initial, verbosity=0)
    second_calls = stage.calls[3:]

    assert success and success_again
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
    np.testing.assert_allclose(first_calls[2][0], expected_before_rejected)


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

    assert success
    assert [step for step, _ in stage.writes] == [0, 2]


def test_stage_skips_intermediate_outputs_by_default(tmp_path):
    stage = _stage(tmp_path)

    success, _ = stage.run(prev_results=np.zeros((5, 2)), verbosity=0)

    assert success
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
    monkeypatch.setattr(
        layout_optimizer,
        "MinDistConstraint",
        lambda problem, min_dist, min_dist_unit: (
            "min_dist",
            problem,
            min_dist,
            min_dist_unit,
        ),
    )

    problem = _Problem()
    stage._add_functions(problem)

    assert calls == [
        ("objective", ("max_rews", problem, [0, 1, 2, 3, 4])),
        ("constraint", ("boundary", problem)),
        ("constraint", ("min_dist", problem, 2.5, "D")),
    ]


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
    monkeypatch.setattr(stage, "_create_optimizer", lambda problem: _Optimizer())

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
