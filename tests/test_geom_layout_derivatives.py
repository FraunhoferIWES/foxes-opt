import numpy as np
from foxes.utils.geom2d import Circle

from foxes_opt.problems.layout.geom_layouts import GeomRegGrid, GeomRegGrids, OMaxN


def _check_layout_derivatives(
    problem: GeomRegGrid | GeomRegGrids,
    variables_int: np.ndarray,
    variables_float: np.ndarray,
) -> None:
    epsilon = 1e-6
    base_points, base_valid = problem.apply_individual(variables_int, variables_float)
    for var in range(len(variables_float)):
        plus = variables_float.copy()
        minus = variables_float.copy()
        plus[var] += epsilon
        minus[var] -= epsilon
        plus_points, plus_valid = problem.apply_individual(variables_int, plus)
        minus_points, minus_valid = problem.apply_individual(variables_int, minus)
        np.testing.assert_array_equal(plus_valid, base_valid)
        np.testing.assert_array_equal(minus_valid, base_valid)
        finite_difference = (plus_points - minus_points) / (2 * epsilon)
        np.testing.assert_allclose(
            problem.layout_derivative(variables_int, variables_float, var),
            finite_difference,
            rtol=1e-6,
            atol=1e-7,
        )
    np.testing.assert_allclose(
        problem.apply_individual(variables_int, variables_float)[0], base_points
    )


def test_geom_reg_grid_layout_derivatives() -> None:
    problem = GeomRegGrid(
        boundary=Circle(np.zeros(2), 10.0), n_turbines=4, min_dist=4.0
    )
    problem.add_objective(OMaxN(problem))
    problem.initialize(verbosity=0)
    variables_float = np.array([0.12, -0.18, 4.2, 4.8, 17.0])

    _check_layout_derivatives(problem, problem.initial_values_int(), variables_float)


def test_geom_reg_grids_layout_derivatives() -> None:
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

    _check_layout_derivatives(problem, variables_int, variables_float)
