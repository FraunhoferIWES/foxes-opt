import numpy as np
import pytest
from foxes.utils.geom2d import Circle
from iwopy import Objective

from foxes_opt.problems.layout.geom_layouts import GeomRegGrids, OMaxN
from foxes_opt.problems.layout.geom_layouts.objectives import (
    MaxDensity,
    MaxGridSpacing,
    MeMiMaDist,
)


def _finite_difference(
    problem: GeomRegGrids,
    objective: Objective,
    variables_int: np.ndarray,
    variables_float: np.ndarray,
) -> np.ndarray:
    epsilon = 1e-6
    out = np.zeros((objective.n_components(), len(variables_float)))
    for var in range(len(variables_float)):
        plus = variables_float.copy()
        minus = variables_float.copy()
        plus[var] += epsilon
        minus[var] -= epsilon
        plus_results = problem.apply_individual(variables_int, plus)
        minus_results = problem.apply_individual(variables_int, minus)
        plus_values = objective.calc_individual(variables_int, plus, plus_results)
        minus_values = objective.calc_individual(variables_int, minus, minus_results)
        out[:, var] = (plus_values - minus_values) / (2 * epsilon)
    return out


def _setup() -> tuple[GeomRegGrids, np.ndarray, np.ndarray]:
    problem = GeomRegGrids(
        boundary=Circle(np.zeros(2), 20.0),
        min_dist=2.0,
        n_grids=2,
        n_max=10,
        n_row_max=4,
    )
    problem.add_objective(OMaxN(problem))
    problem.initialize(verbosity=0)
    variables_int = np.array([2, 3, 2, 2])
    variables_float = np.array([-8.0, -4.0, 3.0, 4.0, 12.0, 3.0, 1.0, 2.5, 3.5, 31.0])
    return problem, variables_int, variables_float


def test_geometry_objective_derivatives_via_iwopy() -> None:
    problem, variables_int, variables_float = _setup()
    objectives = [
        MaxGridSpacing(problem),
        MaxDensity(problem, dfactor=2),
        MeMiMaDist(problem),
    ]
    for objective in objectives:
        objective.initialize(verbosity=0)
        analytical = problem.get_gradients(
            variables_int, variables_float, func=objective
        )
        numerical = _finite_difference(
            problem, objective, variables_int, variables_float
        )
        np.testing.assert_allclose(analytical, numerical, rtol=2e-5, atol=2e-6)


def test_max_grid_spacing_derivative_fails_at_tie() -> None:
    problem, variables_int, variables_float = _setup()
    objective = MaxGridSpacing(problem)
    objective.initialize(verbosity=0)
    variables_float[2] = variables_float[7]

    with pytest.raises(ValueError, match="Failed to calculate derivatives"):
        problem.get_gradients(variables_int, variables_float, func=objective)
