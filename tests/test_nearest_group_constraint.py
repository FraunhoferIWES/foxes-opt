import numpy as np
import pytest
import xarray as xr
from foxes import constants as FC
from foxes import variables as FV
from iwopy import SimpleProblem

from foxes_opt.constraints import NearestGroupConstraint
from foxes_opt.core import FarmConstraint


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
        names = [self.tvar(coord, ti) for ti in sel_turbines for coord in (FV.X, FV.Y)]
        values = layout[sel_turbines].reshape(-1)
        super().__init__("layout", float_vars=names, init_values_float=values)

    @classmethod
    def tvar(cls, var: str, turbine_i: int) -> str:
        return f"{var}_{turbine_i:04d}"

    @classmethod
    def parse_tvar(cls, var: str) -> tuple[str, int]:
        name, turbine = var.split("_")
        return name, int(turbine)


class _PolarLayoutProblem(_LayoutProblem):
    def __init__(self, layout: np.ndarray) -> None:
        self.farm = _Farm(layout)
        self.sel_turbines = list(range(len(layout)))
        names = [
            self.tvar(variable, turbine)
            for turbine in self.sel_turbines
            for variable in ("r", "theta")
        ]
        SimpleProblem.__init__(
            self,
            "polar_layout",
            float_vars=names,
            init_values_float=np.zeros(len(names)),
        )


def _setup(
    layout: list[list[float]],
    max_nearest_dist: float,
    min_nearest_group_size: int,
    sel_turbines: list[int] | None = None,
    **kwargs: object,
) -> tuple[_LayoutProblem, NearestGroupConstraint, np.ndarray]:
    values = np.asarray(layout, dtype=float)
    selected = list(range(len(values))) if sel_turbines is None else sel_turbines
    problem = _LayoutProblem(values, selected)
    constraint = NearestGroupConstraint(
        problem,
        max_nearest_dist=max_nearest_dist,
        min_nearest_group_size=min_nearest_group_size,
        **kwargs,
    )
    constraint.initialize()
    return problem, constraint, values[selected].reshape(-1)


def _individual_results(layout: np.ndarray) -> xr.Dataset:
    return xr.Dataset(
        {
            FV.X: ((FC.STATE, FC.TURBINE), layout[None, :, 0]),
            FV.Y: ((FC.STATE, FC.TURBINE), layout[None, :, 1]),
        }
    )


def _population_results(layouts: np.ndarray, n_states: int = 2) -> xr.Dataset:
    n_pop, n_turbines, _ = layouts.shape
    repeated = np.repeat(layouts[:, None], n_states, axis=1).reshape(
        n_pop * n_states, n_turbines, 2
    )
    return xr.Dataset(
        {
            FV.X: ((FC.STATE, FC.TURBINE), repeated[..., 0]),
            FV.Y: ((FC.STATE, FC.TURBINE), repeated[..., 1]),
            "n_pop": n_pop,
            "n_org_states": n_states,
        }
    )


def test_chain_group_uses_connected_component_size() -> None:
    layout = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
    _, constraint, variables = _setup(
        layout.tolist(), max_nearest_dist=1.1, min_nearest_group_size=3
    )

    values = constraint.calc_individual(
        np.array([], dtype=int), variables, _individual_results(layout)
    )
    selected = constraint.calc_individual(
        np.array([], dtype=int),
        variables,
        _individual_results(layout),
        components=[3, 1],
    )

    np.testing.assert_allclose(values, [-0.1, -0.1, -0.1, 6.9])
    np.testing.assert_allclose(selected, [6.9, -0.1])


def test_group_can_include_fixed_turbines() -> None:
    layout = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    _, constraint, variables = _setup(
        layout.tolist(),
        max_nearest_dist=1.1,
        min_nearest_group_size=3,
        sel_turbines=[0],
    )

    values = constraint.calc_individual(
        np.array([], dtype=int), variables, _individual_results(layout)
    )

    np.testing.assert_allclose(values, [-0.1])


def test_group_size_one_still_enforces_nearest_distance() -> None:
    layout = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
    _, constraint, variables = _setup(
        layout.tolist(), max_nearest_dist=1.5, min_nearest_group_size=1
    )

    values = constraint.calc_individual(
        np.array([], dtype=int), variables, _individual_results(layout)
    )

    np.testing.assert_allclose(values, [-0.5, -0.5, 2.5])


def test_population_matches_individual() -> None:
    layouts = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]],
            [[0.0, 0.0], [3.0, 0.0], [6.0, 0.0], [9.0, 0.0]],
        ]
    )
    _, constraint, _ = _setup(
        layouts[0].tolist(), max_nearest_dist=1.1, min_nearest_group_size=3
    )
    variables = layouts.reshape(2, -1)
    variables_int = np.zeros((2, 0), dtype=int)

    population = constraint.calc_population(
        variables_int, variables, _population_results(layouts)
    )
    individuals = np.stack(
        [
            constraint.calc_individual(
                variables_int[i], variables[i], _individual_results(layout)
            )
            for i, layout in enumerate(layouts)
        ]
    )

    np.testing.assert_allclose(population, individuals)
    np.testing.assert_allclose(population[1], np.full(4, 1.9))


def test_explicit_population_keeps_unselected_turbines_fixed() -> None:
    problem, constraint, _ = _setup(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        max_nearest_dist=1.1,
        min_nearest_group_size=3,
        sel_turbines=[0],
        infer_vars=False,
    )
    problem.farm.turbines[1].xy = np.array([[100.0, 0.0], [200.0, 0.0]])
    variables = np.array([[0.0, 0.0], [0.2, 0.0]])

    values = constraint.calc_population(
        np.zeros((2, 0), dtype=int), variables, problem_results=None
    )

    np.testing.assert_allclose(values[:, 0], [-0.1, -0.1])


def test_analytical_derivatives_via_iwopy() -> None:
    layout = [[0.0, 0.0], [3.0, 4.0], [0.0, 10.0]]
    problem, constraint, variables = _setup(
        layout, max_nearest_dist=7.0, min_nearest_group_size=3
    )

    gradients = problem.get_gradients(
        np.array([], dtype=int), variables, func=constraint
    )

    distance = np.sqrt(45.0)
    expected_row = [
        0.0,
        0.0,
        3.0 / distance,
        -6.0 / distance,
        -3.0 / distance,
        6.0 / distance,
    ]
    np.testing.assert_allclose(gradients, np.tile(expected_row, (3, 1)))


def test_analytical_derivatives_fail_at_zero_distance() -> None:
    problem, constraint, variables = _setup(
        [[0.0, 0.0], [0.0, 0.0]],
        max_nearest_dist=1.0,
        min_nearest_group_size=1,
    )

    with pytest.raises(ValueError, match="Failed to calculate derivatives"):
        problem.get_gradients(np.array([], dtype=int), variables, func=constraint)


def test_non_xy_layout_variables_fall_back_to_numerical_derivatives() -> None:
    problem = _PolarLayoutProblem(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
    constraint = NearestGroupConstraint(
        problem,
        max_nearest_dist=1.1,
        min_nearest_group_size=3,
    )
    constraint.initialize()
    variables = problem.initial_values_float()

    assert np.all(constraint.vardeps_float())
    for variable in range(len(variables)):
        assert np.all(
            np.isnan(constraint.ana_deriv(np.array([], dtype=int), variables, variable))
        )


@pytest.mark.parametrize(
    ("max_nearest_dist", "min_nearest_group_size", "message"),
    [
        (0.0, 2, "max_nearest_dist must be positive"),
        (1.0, 0, "min_nearest_group_size must be at least 1"),
    ],
)
def test_rejects_invalid_parameters(
    max_nearest_dist: float, min_nearest_group_size: int, message: str
) -> None:
    problem = _LayoutProblem(np.array([[0.0, 0.0], [1.0, 0.0]]), [0, 1])

    with pytest.raises(ValueError, match=message):
        NearestGroupConstraint(
            problem,
            max_nearest_dist=max_nearest_dist,
            min_nearest_group_size=min_nearest_group_size,
        )


def test_rejects_impossible_group_size() -> None:
    problem = _LayoutProblem(np.array([[0.0, 0.0], [1.0, 0.0]]), [0, 1])
    constraint = NearestGroupConstraint(
        problem,
        max_nearest_dist=1.0,
        min_nearest_group_size=3,
    )

    with pytest.raises(ValueError, match="exceeds the number of checked turbines"):
        constraint.initialize()


def test_runtime_factory_finds_constraint() -> None:
    problem = _LayoutProblem(np.array([[0.0, 0.0], [1.0, 0.0]]), [0, 1])

    constraint = FarmConstraint.new(
        "NearestGroupConstraint",
        problem=problem,
        max_nearest_dist=2.0,
        min_nearest_group_size=2,
    )

    assert isinstance(constraint, NearestGroupConstraint)
