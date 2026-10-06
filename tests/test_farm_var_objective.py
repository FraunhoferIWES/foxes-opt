from types import SimpleNamespace

import numpy as np
import xarray as xr

import foxes.constants as FC
import foxes.variables as FV
from foxes_opt.objectives.farm_vars import FarmVarObjective


def _objective():
    objective = FarmVarObjective.__new__(FarmVarObjective)
    objective.name = "test"
    objective.rules = {FC.STATE: "weights", FC.TURBINE: "sum"}
    return objective


def test_weight_normalization_individual():
    data = xr.DataArray([[1.0, 2.0], [3.0, 6.0]], dims=(FC.STATE, FC.TURBINE))
    weights = xr.DataArray([1.0, 3.0], dims=(FC.STATE,))

    normalized = _objective()._contract(data, weights)

    np.testing.assert_allclose(normalized, 7.5)


def test_weight_normalization_population_and_turbines():
    data = xr.DataArray(
        [
            [[1.0, 4.0], [3.0, 8.0]],
            [[2.0, 5.0], [6.0, 9.0]],
        ],
        dims=(FC.POP, FC.STATE, FC.TURBINE),
    )
    weights = xr.DataArray(
        [
            [[1.0, 3.0], [3.0, 1.0]],
            [[2.0, 1.0], [2.0, 3.0]],
        ],
        dims=(FC.POP, FC.STATE, FC.TURBINE),
    )

    normalized = _objective()._contract(data, weights)

    np.testing.assert_allclose(normalized, [7.5, 12.0])


def test_weight_normalization_population():
    data = xr.DataArray(
        [
            [[1.0, 4.0], [3.0, 8.0]],
            [[2.0, 5.0], [6.0, 9.0]],
        ],
        dims=(FC.POP, FC.STATE, FC.TURBINE),
    )
    weights = xr.DataArray(
        [[1.0, 3.0], [2.0, 2.0]],
        dims=(FC.POP, FC.STATE),
    )

    normalized = _objective()._contract(data, weights)

    np.testing.assert_allclose(normalized, [9.5, 11.0])


def test_selected_turbines_use_matching_population_data_and_weights():
    problem = SimpleNamespace(
        farm=SimpleNamespace(n_turbines=3),
        sel_turbines=[0, 2],
    )
    objective = FarmVarObjective(
        problem,
        name="selected_power",
        variable=FV.P,
        contract_states="weights",
        contract_turbines="sum",
        minimize=False,
    )
    data = np.array(
        [
            [[1.0, 100.0, 3.0], [5.0, 100.0, 7.0]],
            [[2.0, 100.0, 4.0], [6.0, 100.0, 8.0]],
        ]
    )
    weights = np.array(
        [
            [[1.0, 9.0, 3.0], [3.0, 9.0, 1.0]],
            [[2.0, 9.0, 1.0], [2.0, 9.0, 3.0]],
        ]
    )
    results = xr.Dataset(
        {
            FV.P: ((FC.STATE, FC.TURBINE), data.reshape(4, 3)),
            FV.WEIGHT: ((FC.STATE, FC.TURBINE), weights.reshape(4, 3)),
            "n_pop": 2,
            "n_org_states": 2,
        }
    )

    actual = objective.calc_population(
        np.zeros((2, 0), dtype=int),
        np.zeros((2, 0)),
        results,
    )

    np.testing.assert_allclose(actual[:, 0], [8.0, 11.0])
