import numpy as np
from foxes.utils.geom2d import Circle

from foxes_opt.problems.layout.geom_layouts import GeomRegGrid, OMaxN
from foxes_opt.problems.layout.geom_layouts.constraints import (
    Boundary,
    CMinDensity,
    MinDist,
)


def _finite_difference(
    problem: GeomRegGrid,
    function: Boundary | CMinDensity | MinDist,
    variables_int: np.ndarray,
    variables_float: np.ndarray,
) -> np.ndarray:
    epsilon = 1e-6
    out = np.zeros((function.n_components(), len(variables_float)))
    for var in range(len(variables_float)):
        plus = variables_float.copy()
        minus = variables_float.copy()
        plus[var] += epsilon
        minus[var] -= epsilon
        plus_results = problem.apply_individual(variables_int, plus)
        minus_results = problem.apply_individual(variables_int, minus)
        plus_values = function.calc_individual(variables_int, plus, plus_results)
        minus_values = function.calc_individual(variables_int, minus, minus_results)
        out[:, var] = (plus_values - minus_values) / (2 * epsilon)
    return out


def test_geometry_constraint_derivatives_via_iwopy() -> None:
    problem = GeomRegGrid(
        boundary=Circle(np.zeros(2), 10.0), n_turbines=4, min_dist=4.0
    )
    problem.add_objective(OMaxN(problem))
    problem.initialize(verbosity=0)
    variables_int = np.array([], dtype=int)
    variables_float = np.array([0.12, -0.18, 4.2, 4.8, 17.0])

    functions = [
        Boundary(problem),
        MinDist(problem),
        CMinDensity(problem, min_value=6.0, dfactor=2),
    ]
    for function in functions:
        function.initialize(verbosity=0)
        analytical = problem.get_gradients(
            variables_int, variables_float, func=function
        )
        numerical = _finite_difference(
            problem, function, variables_int, variables_float
        )
        np.testing.assert_allclose(analytical, numerical, rtol=2e-5, atol=2e-6)
