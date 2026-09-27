from types import SimpleNamespace

import foxes
import foxes.variables as FV
import numpy as np
from foxes.algorithms.downwind.models import PopulationStates

from foxes_opt.constraints import FarmBoundaryConstraint, MinDistConstraint
from foxes_opt.core.farm_opt_problem import FarmOptProblem
from foxes_opt.core.farm_vars_problem import FarmVarsProblem
from foxes_opt.objectives import MaxFarmPower
from foxes_opt.problems import OptFarmVars
from foxes_opt.problems.layout import FarmLayoutOptProblem


def test_reset_states_keeps_population_loaded_data():
    farm = foxes.WindFarm(boundary=foxes.utils.geom2d.Circle([0.0, 0.0], 1000.0))
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=np.array([0.0, 0.0]),
        xy_step=np.array([100.0, 0.0]),
        n_turbines=2,
        turbine_models=["NREL5MW"],
    )

    states = foxes.input.states.StatesTable(
        data_source="wind_rose_bremen.csv",
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "ws", FV.WD: "wd", FV.WEIGHT: "weight"},
        fixed_vars={FV.RHO: 1.225, FV.TI: 0.04},
    )

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        rotor_model="centre",
        wake_models=["Bastankhah2014_linear_lim_k004"],
        verbosity=0,
    )

    problem = FarmLayoutOptProblem("layout_opt_reset", algo)
    problem.add_objective(MaxFarmPower(problem))
    problem.add_constraint(FarmBoundaryConstraint(problem))
    problem.initialize()

    n_pop = 3
    vars_float = np.zeros((n_pop, 2 * farm.n_turbines), dtype=float)
    vars_int = np.zeros((n_pop, 0), dtype=int)
    problem.update_problem_population(vars_int, vars_float)

    assert isinstance(algo.states, PopulationStates)
    assert algo.states.initialized

    ld = algo.loaded_data
    assert "PopulationStates_smap" in ld["data_vars"]
    assert "StatesTable_data" in ld["data_vars"]
    assert "StatesTable_weight" in ld["data_vars"]


def test_farm_layout_population_uses_population_major_order():
    farm = foxes.WindFarm(boundary=foxes.utils.geom2d.Circle([0.0, 0.0], 1000.0))
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=np.array([0.0, 0.0]),
        xy_step=np.array([100.0, 0.0]),
        n_turbines=2,
        turbine_models=["NREL5MW"],
    )

    states = foxes.input.states.StatesTable(
        data_source="wind_rose_bremen.csv",
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "ws", FV.WD: "wd", FV.WEIGHT: "weight"},
        fixed_vars={FV.RHO: 1.225, FV.TI: 0.04},
    )

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        rotor_model="centre",
        wake_models=["Bastankhah2014_linear_lim_k004"],
        verbosity=0,
    )

    problem = FarmLayoutOptProblem("layout_pop_order", algo)
    problem.add_objective(MaxFarmPower(problem))
    problem.add_constraint(FarmBoundaryConstraint(problem))
    problem.initialize()

    n_pop = 2
    vars_float = np.array(
        [
            [10.0, 1.0, 20.0, 2.0],
            [100.0, 11.0, 200.0, 22.0],
        ]
    )
    vars_int = np.zeros((n_pop, 0), dtype=int)
    problem.update_problem_population(vars_int, vars_float)

    n_states0 = problem._org_n_states
    assert n_states0 > 1

    xy0 = algo.farm.turbines[0].xy
    xy1 = algo.farm.turbines[1].xy

    assert np.allclose(xy0[:n_states0], [10.0, 1.0])
    assert np.allclose(xy1[:n_states0], [20.0, 2.0])
    assert np.allclose(xy0[n_states0 : 2 * n_states0], [100.0, 11.0])
    assert np.allclose(xy1[n_states0 : 2 * n_states0], [200.0, 22.0])


class _DummyModel:
    def __init__(self):
        self.data = {}

    def reset(self):
        self.data = {}

    def add_var(self, name, values):
        self.data[name] = np.asarray(values)


class _DummyFarmVarsProblem(FarmVarsProblem):
    def opt2farm_vars_individual(self, vars_int, vars_float):
        return {}

    def opt2farm_vars_population(self, vars_int, vars_float, n_states):
        n_pop = len(vars_float)
        data = np.zeros((n_pop, n_states, 1), dtype=float)
        data[:, 0, 0] = [11.0, 22.0, 33.0]
        data[:, 1, 0] = [101.0, 202.0, 303.0]
        return {"dummy_var": data}


def test_farm_vars_population_flattening_is_population_major(monkeypatch):
    monkeypatch.setattr(FarmOptProblem, "update_problem_population", lambda *args: None)

    algo = SimpleNamespace(
        mbook=SimpleNamespace(turbine_models={"dummy": _DummyModel()}),
        farm=SimpleNamespace(n_turbines=1, turbines=[SimpleNamespace(xy=np.zeros(2))]),
        n_turbines=1,
    )

    problem = _DummyFarmVarsProblem("dummy_problem", algo)
    problem._model_vars = {"dummy": ["dummy_var"]}
    problem._org_n_states = 2

    vars_float = np.zeros((3, 0), dtype=float)
    vars_int = np.zeros((3, 0), dtype=int)
    problem.update_problem_population(vars_int, vars_float)

    got = algo.mbook.turbine_models["dummy"].data["dummy_var"][:, 0]
    expected = np.array([11.0, 101.0, 22.0, 202.0, 33.0, 303.0])
    assert np.allclose(got, expected)


def test_population_matches_individual_for_multiple_states():
    farm = foxes.WindFarm(boundary=foxes.utils.geom2d.Circle([0.0, 0.0], 1000.0))
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=np.array([0.0, 0.0]),
        xy_step=np.array([250.0, 0.0]),
        n_turbines=2,
        turbine_models=["NREL5MW"],
    )
    states = foxes.input.states.ScanStates(
        scans={
            FV.WS: [8.0, 10.0],
            FV.WD: [270.0],
            FV.TI: [0.08],
            FV.RHO: [1.225],
        }
    )
    algo = foxes.algorithms.Downwind(
        farm,
        states,
        rotor_model="centre",
        wake_models=["Bastankhah2014_linear_lim_k004"],
        verbosity=0,
    )

    points = np.array(
        [
            [[100.0, 50.0, 80.0]],
            [[150.0, -50.0, 80.0]],
        ]
    )
    problem = FarmLayoutOptProblem("layout_pop_multi_state", algo, points=points)
    objective = MaxFarmPower(problem)
    boundary = FarmBoundaryConstraint(problem, disc_inside=True)
    min_dist = MinDistConstraint(problem, min_dist=2.0, min_dist_unit="D")
    problem.add_objective(objective)
    problem.add_constraint(boundary)
    problem.add_constraint(min_dist)
    problem.initialize()

    vars_float = np.array(
        [
            [0.0, 0.0, 250.0, 0.0],
            [50.0, 25.0, 400.0, -20.0],
        ]
    )
    vars_int = np.zeros((len(vars_float), 0), dtype=int)
    population_results, population_point_results = problem.apply_population(
        vars_int, vars_float
    )
    population_objective = objective.calc_population(
        vars_int, vars_float, population_results
    )
    population_boundary = boundary.calc_population(
        vars_int, vars_float, population_results
    )
    population_min_dist = min_dist.calc_population(
        vars_int, vars_float, population_results
    )

    n_states = problem._org_n_states
    assert n_states == 2
    for pop_index in range(len(vars_float)):
        individual_results, individual_point_results = problem.apply_individual(
            vars_int[pop_index], vars_float[pop_index]
        )
        pop_slice = slice(pop_index * n_states, (pop_index + 1) * n_states)
        for variable in [FV.X, FV.Y, FV.P]:
            np.testing.assert_allclose(
                population_results[variable].isel(state=pop_slice),
                individual_results[variable],
            )
        for variable in [FV.WS, FV.WD, FV.TI, FV.RHO, FV.WEIGHT]:
            np.testing.assert_allclose(
                population_point_results[variable].isel(state=pop_slice),
                individual_point_results[variable],
            )
        np.testing.assert_allclose(
            population_objective[pop_index],
            objective.calc_individual(
                vars_int[pop_index], vars_float[pop_index], individual_results
            ),
        )
        np.testing.assert_allclose(
            population_boundary[pop_index],
            boundary.calc_individual(
                vars_int[pop_index], vars_float[pop_index], individual_results
            ),
        )
        np.testing.assert_allclose(
            population_min_dist[pop_index],
            min_dist.calc_individual(
                vars_int[pop_index], vars_float[pop_index], individual_results
            ),
        )


def test_opt_farm_vars_population_repeated_apply_is_consistent():
    farm = foxes.WindFarm(boundary=foxes.utils.geom2d.Circle([0.0, 0.0], 1000.0))
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=np.array([0.0, 0.0]),
        xy_step=np.array([200.0, 0.0]),
        n_turbines=2,
        turbine_models=["opt_yawm", "yawm2yaw", "NREL5MW"],
    )

    states = foxes.input.states.ScanStates(
        scans={
            FV.WS: [8.0, 10.0],
            FV.WD: [270.0],
            FV.TI: [0.08],
            FV.RHO: [1.225],
        }
    )

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        rotor_model="centre",
        wake_models=["Bastankhah2016_linear_ka02"],
        wake_deflection="JimenezProj",
        verbosity=0,
    )

    problem = OptFarmVars("opt_yawm", algo)
    problem.add_var(FV.YAWM, float, 0.0, -30.0, 30.0, level="state-turbine")
    problem.add_objective(MaxFarmPower(problem))
    problem.initialize(verbosity=0)

    n_pop = 4
    vars_int = np.zeros((n_pop, problem.n_vars_int), dtype=int)
    vars_float = np.arange(n_pop * problem.n_vars_float, dtype=float).reshape(
        n_pop, problem.n_vars_float
    )
    population_vars = problem.opt2farm_vars_population(
        vars_int, vars_float, problem._org_n_states
    )[FV.YAWM]
    assert population_vars.shape == (n_pop, problem._org_n_states, farm.n_turbines)
    for pop_index in range(n_pop):
        individual_vars = problem.opt2farm_vars_individual(
            vars_int[pop_index], vars_float[pop_index]
        )[FV.YAWM]
        np.testing.assert_array_equal(population_vars[pop_index], individual_vars)

    res0 = problem.apply_population(vars_int, vars_float)
    res1 = problem.apply_population(vars_int, vars_float)

    assert isinstance(algo.states, PopulationStates)
    assert algo.states.n_pop == n_pop
    assert res0.sizes["state"] == n_pop * problem._org_n_states
    assert res1.sizes["state"] == n_pop * problem._org_n_states
    np.testing.assert_array_equal(
        res0[FV.YAWM], population_vars.reshape(res0[FV.YAWM].shape)
    )
    np.testing.assert_array_equal(res1[FV.YAWM], res0[FV.YAWM])
