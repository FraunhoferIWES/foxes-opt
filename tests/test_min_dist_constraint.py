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

    def get_rotor_diameters(self, algo):
        return np.ones(self.n_turbines)


class _LayoutProblem(SimpleProblem):
    def __init__(self, layout: np.ndarray, sel_turbines: list[int]) -> None:
        self.farm = _Farm(layout)
        self.algo = None
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


def test_active_pairs_use_guarded_boxes_without_changing_components():
    _, constraint, variables = _setup([[0.0, 0.0], [3.0, 0.0], [20.0, 0.0]], [0, 1, 2])

    selected = constraint.active_components(
        variables, variables - 0.75, variables + 0.75
    )

    np.testing.assert_array_equal(selected, [0])
    assert constraint.n_components() == 3
    np.testing.assert_array_equal(constraint._i2t, [[0, 1], [0, 2], [1, 2]])


def test_active_pairs_preserve_unbounded_and_fixed_turbines():
    _, constraint, variables = _setup([[0.0, 0.0], [3.0, 0.0], [20.0, 0.0]], [0])
    np.testing.assert_array_equal(
        constraint.active_components(
            variables, np.array([-2.0, 0.0]), np.array([2.0, 0.0])
        ),
        [0],
    )
    np.testing.assert_array_equal(
        constraint.active_components(
            variables, np.full(2, -np.inf), np.full(2, np.inf)
        ),
        [0, 1],
    )


def test_active_pairs_reject_unknown_distance_units():
    _, constraint, variables = _setup([[0.0, 0.0], [3.0, 0.0]], [0, 1])
    constraint.min_dist_unit = "invalid"
    with pytest.raises(ValueError, match="Unknown min_dist_unit"):
        constraint.active_components(variables, variables, variables)


def test_active_pairs_use_larger_pair_diameter(monkeypatch):
    problem, constraint, variables = _setup([[0.0, 0.0], [3.0, 0.0]], [0, 1])
    monkeypatch.setattr(
        problem.farm, "get_rotor_diameters", lambda algo: np.array([1.0, 2.0])
    )
    np.testing.assert_array_equal(
        constraint.active_components(variables, variables, variables), [0]
    )


def test_active_pairs_retain_state_dependent_fixed_coordinates():
    problem, constraint, variables = _setup([[0.0, 0.0], [20.0, 0.0]], [0])
    problem.farm.turbines[1].xy = np.array([[20.0, 0.0], [1.0, 0.0]])
    np.testing.assert_array_equal(
        constraint.active_components(variables, variables, variables), [0]
    )


@pytest.mark.parametrize("variable", ["DX", "D"])
def test_active_pairs_retain_uncertified_parameterizations(variable):
    _, constraint, variables = _setup([[0.0, 0.0], [20.0, 0.0]], [0])
    constraint.var_names_float[0] = f"{variable}_0000"
    assert constraint.var_names_float[0] == f"{variable}_0000"
    np.testing.assert_array_equal(
        constraint.active_components(variables, variables - 30.0, variables + 30.0),
        [0],
    )


def test_active_pairs_refresh_for_changed_guarded_region():
    _, constraint, variables = _setup([[0.0, 0.0], [20.0, 0.0]], [0])
    np.testing.assert_array_equal(
        constraint.active_components(variables, variables - 1.0, variables + 1.0), []
    )
    moved = variables + np.array([18.0, 0.0])
    np.testing.assert_array_equal(
        constraint.active_components(moved, moved - 1.0, moved + 1.0), [0]
    )
