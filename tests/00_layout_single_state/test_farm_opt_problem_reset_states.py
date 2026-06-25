import numpy as np
from types import SimpleNamespace

import foxes
import foxes.variables as FV
from foxes.algorithms.downwind.models import PopulationStates
from foxes_opt.core.farm_opt_problem import FarmOptProblem
from foxes_opt.core.farm_vars_problem import FarmVarsProblem
from foxes_opt.constraints import FarmBoundaryConstraint
from foxes_opt.objectives import MaxFarmPower
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


def test_farm_layout_population_uses_state_major_order():
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

    assert np.allclose(xy0[:2], [[10.0, 1.0], [100.0, 11.0]])
    assert np.allclose(xy1[:2], [[20.0, 2.0], [200.0, 22.0]])
    assert np.allclose(xy0[2:4], [[10.0, 1.0], [100.0, 11.0]])
    assert np.allclose(xy1[2:4], [[20.0, 2.0], [200.0, 22.0]])


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
        data = np.zeros((n_states, n_pop, 1), dtype=float)
        data[0, :, 0] = [11.0, 22.0, 33.0]
        data[1, :, 0] = [101.0, 202.0, 303.0]
        return {"dummy_var": data}


def test_farm_vars_population_flattening_is_state_major(monkeypatch):
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
    expected = np.array([11.0, 22.0, 33.0, 101.0, 202.0, 303.0])
    assert np.allclose(got, expected)
