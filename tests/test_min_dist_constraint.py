import numpy as np
import pytest
from iwopy import SimpleProblem

from foxes_opt.constraints import MinDistConstraint


class _Turbine:
    def __init__(self, xy: np.ndarray) -> None:
        self.xy = xy


class _Farm:
    def __init__(self, layout: np.ndarray) -> None:
        self.turbines = [_Turbine(xy.copy()) for xy in layout]
        self.n_turbines = len(self.turbines)


class _LayoutProblem(SimpleProblem):
    def __init__(self, layout: np.ndarray, sel_turbines: list[int]) -> None:
        self.farm = _Farm(layout)
        self.sel_turbines = sel_turbines
        names = [self.tvar(coord, ti) for ti in sel_turbines for coord in ("X", "Y")]
        values = layout[sel_turbines].reshape(-1)
        super().__init__("layout", float_vars=names, init_values_float=values)

    @classmethod
    def tvar(cls, var: str, turbine_i: int) -> str:
        return f"{var}_{turbine_i:04d}"

    @classmethod
    def parse_tvar(cls, var: str) -> tuple[str, int]:
        name, turbine = var.split("_")
        return name, int(turbine)


def _setup(
    layout: list[list[float]], sel_turbines: list[int]
) -> tuple[_LayoutProblem, MinDistConstraint, np.ndarray]:
    values = np.asarray(layout, dtype=float)
    problem = _LayoutProblem(values, sel_turbines)
    constraint = MinDistConstraint(problem, min_dist=2.0, min_dist_unit="D")
    constraint.initialize()
    return problem, constraint, values[sel_turbines].reshape(-1)


def test_analytical_derivatives_via_iwopy() -> None:
    problem, constraint, variables = _setup(
        [[0.0, 0.0], [3.0, 4.0], [0.0, 8.0]], [0, 1, 2]
    )

    gradients = problem.get_gradients(
        np.array([], dtype=int), variables, func=constraint
    )
    expected = np.array(
        [
            [0.6, 0.8, -0.6, -0.8, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, -0.6, 0.8, 0.6, -0.8],
        ]
    )
    np.testing.assert_allclose(gradients, expected)

    selected = problem.get_gradients(
        np.array([], dtype=int),
        variables,
        func=constraint,
        components=[2, 0],
        vars=["Y_0002", "X_0000"],
    )
    np.testing.assert_allclose(selected, [[-0.8, 0.0], [0.0, 0.6]])


def test_components_map_to_dependent_turbines() -> None:
    _, constraint, _ = _setup([[0.0, 0.0], [3.0, 4.0], [0.0, 8.0]], [0, 1, 2])

    assert constraint.component_turbines([2]) == {1, 2}
    assert constraint.component_turbines([]) == set()


def test_analytical_derivatives_with_fixed_turbines() -> None:
    problem, constraint, variables = _setup([[0.0, 0.0], [3.0, 4.0], [0.0, 8.0]], [0])

    gradients = problem.get_gradients(
        np.array([], dtype=int), variables, func=constraint
    )

    np.testing.assert_allclose(gradients, [[0.6, 0.8], [0.0, 1.0]])


def test_analytical_derivatives_with_integer_farm_coordinates() -> None:
    layout = np.array([[0, 0], [3, 4]], dtype=int)
    problem = _LayoutProblem(layout, [0])
    constraint = MinDistConstraint(problem, min_dist=2.0)
    constraint.initialize()
    variables = np.array([0.5, 0.0])

    gradients = problem.get_gradients(
        np.array([], dtype=int), variables, func=constraint
    )

    distance = np.hypot(2.5, 4.0)
    np.testing.assert_allclose(gradients, [[2.5 / distance, 4.0 / distance]])


def test_analytical_derivatives_fail_at_zero_distance() -> None:
    problem, constraint, variables = _setup([[0.0, 0.0], [0.0, 0.0]], [0, 1])

    with pytest.raises(ValueError, match="Failed to calculate derivatives"):
        problem.get_gradients(np.array([], dtype=int), variables, func=constraint)
