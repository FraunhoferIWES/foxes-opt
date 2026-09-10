import numpy as np
import xarray as xr

import foxes.constants as FC
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
