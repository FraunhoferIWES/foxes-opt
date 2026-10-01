import numpy as np
import pytest
from foxes.utils.geom2d import AreaGeometry, Circle, ClosedPolygon, HalfPlane
from iwopy import SimpleProblem

from foxes_opt.constraints import AreaGeometryConstraint


class _Turbine:
    def __init__(self, xy: np.ndarray) -> None:
        self.xy = xy


class _Farm:
    def __init__(self, layout: np.ndarray) -> None:
        self.turbines = [_Turbine(xy.copy()) for xy in layout]
        self.n_turbines = len(self.turbines)


class _LayoutProblem(SimpleProblem):
    def __init__(self, layout: np.ndarray) -> None:
        self.farm = _Farm(layout)
        self.sel_turbines = list(range(len(layout)))
        names = [
            self.tvar(coord, ti) for ti in self.sel_turbines for coord in ("X", "Y")
        ]
        super().__init__("layout", float_vars=names, init_values_float=layout.flat)

    @classmethod
    def tvar(cls, var: str, turbine_i: int) -> str:
        return f"{var}_{turbine_i:04d}"

    @classmethod
    def parse_tvar(cls, var: str) -> tuple[str, int]:
        name, turbine = var.split("_")
        return name, int(turbine)


def _rectangle(x_min: float, y_min: float, x_max: float, y_max: float) -> ClosedPolygon:
    return ClosedPolygon(
        np.array(
            [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
            ],
            dtype=float,
        )
    )


def _setup(
    layout: list[list[float]],
    geometry: AreaGeometry | None = None,
    **kwargs: object,
) -> tuple[_LayoutProblem, AreaGeometryConstraint]:
    values = np.asarray(layout, dtype=float)
    problem = _LayoutProblem(values)
    constraint = AreaGeometryConstraint(
        problem,
        "boundary",
        Circle(np.zeros(2), 10.0) if geometry is None else geometry,
        **kwargs,
    )
    constraint.initialize()
    return problem, constraint


def _assert_matches_finite_difference(
    geometry: AreaGeometry,
    points: list[list[float]],
    **kwargs: object,
) -> None:
    problem, constraint = _setup(points, geometry=geometry, infer_vars=False, **kwargs)
    variables = problem.initial_values_float()
    variables_int = np.array([], dtype=int)
    analytical = problem.get_gradients(variables_int, variables, func=constraint)
    epsilon = 1e-6
    numerical = np.zeros_like(analytical)
    for var in range(len(variables)):
        plus = variables.copy()
        minus = variables.copy()
        plus[var] += epsilon
        minus[var] -= epsilon
        plus_values = constraint.calc_individual(variables_int, plus, None)
        minus_values = constraint.calc_individual(variables_int, minus, None)
        numerical[:, var] = (plus_values - minus_values) / (2 * epsilon)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-6, atol=1e-7)


def test_analytical_derivatives_via_iwopy() -> None:
    problem, constraint = _setup([[3.0, 4.0], [12.0, 5.0]])
    variables = problem.initial_values_float()

    gradients = problem.get_gradients(
        np.array([], dtype=int), variables, func=constraint
    )
    np.testing.assert_allclose(
        gradients,
        [[0.6, 0.8, 0.0, 0.0], [0.0, 0.0, 12.0 / 13.0, 5.0 / 13.0]],
    )

    selected = problem.get_gradients(
        np.array([], dtype=int),
        variables,
        func=constraint,
        components=[1, 0],
        vars=["Y_0001", "X_0000"],
    )
    np.testing.assert_allclose(selected, [[5.0 / 13.0, 0.0], [0.0, 0.6]])


@pytest.mark.parametrize(
    ("geometry", "points"),
    [
        pytest.param(
            _rectangle(-10.0, -8.0, 9.0, 7.0),
            [[-7.0, 1.0], [8.0, 3.0], [12.0, -2.0], [11.0, 9.0]],
            id="polygon",
        ),
        pytest.param(
            _rectangle(-9.0, -4.0, -3.0, 4.0) + Circle([5.0, 0.0], 3.0),
            [[-5.0, 1.0], [5.0, 1.0], [0.0, 1.0], [9.0, 1.0]],
            id="binary-union",
        ),
        pytest.param(
            _rectangle(-9.0, -4.0, -5.0, 4.0)
            + [Circle([0.0, 0.0], 2.0), Circle([7.0, 0.0], 2.5)],
            [[-6.5, 1.0], [0.5, 0.0], [7.0, 1.0], [3.5, 1.0]],
            id="list-union",
        ),
        pytest.param(
            (_rectangle(-9.0, -4.0, -5.0, 4.0) + Circle([0.0, 0.0], 2.0))
            + Circle([7.0, 0.0], 2.5),
            [[-6.5, 1.0], [0.5, 0.0], [7.0, 1.0], [3.5, 1.0]],
            id="chained-union",
        ),
        pytest.param(
            (_rectangle(-9.0, -4.0, -5.0, 4.0) + Circle([0.0, 0.0], 2.0))
            + [Circle([7.0, 0.0], 2.5)],
            [[-6.5, 1.0], [0.5, 0.0], [7.0, 1.0], [3.5, 1.0]],
            id="union-plus-list",
        ),
        pytest.param(
            _rectangle(-9.0, -4.0, -5.0, 4.0)
            + (Circle([0.0, 0.0], 2.0) + Circle([7.0, 0.0], 2.5)),
            [[-6.5, 1.0], [0.5, 0.0], [7.0, 1.0], [3.5, 1.0]],
            id="geometry-plus-union",
        ),
        pytest.param(
            (_rectangle(-9.0, -4.0, -5.0, 4.0) + Circle([0.0, 0.0], 2.0))
            + (Circle([7.0, 0.0], 2.5) + Circle([13.0, 0.0], 1.5)),
            [[-6.5, 1.0], [0.5, 0.0], [7.0, 1.0], [13.0, 0.5]],
            id="union-plus-union",
        ),
        pytest.param(
            _rectangle(-10.0, -9.0, 10.0, 9.0) - Circle([0.0, 0.0], 3.0),
            [[1.0, 0.0], [4.5, 1.0], [9.0, 0.0], [11.0, 2.0]],
            id="single-exclusion",
        ),
        pytest.param(
            _rectangle(-10.0, -9.0, 10.0, 9.0)
            - [Circle([-4.0, 0.0], 1.5), Circle([4.0, 0.0], 2.0)],
            [[-4.0, 0.5], [4.5, 0.0], [0.0, 2.0], [11.0, -2.0]],
            id="list-exclusion",
        ),
        pytest.param(
            (_rectangle(-10.0, -9.0, 10.0, 9.0) - Circle([-4.0, 0.0], 1.5))
            - Circle([4.0, 0.0], 2.0),
            [[-4.0, 0.5], [4.5, 0.0], [0.0, 2.0], [11.0, -2.0]],
            id="chained-exclusion",
        ),
        pytest.param(
            _rectangle(-10.0, -9.0, 10.0, 9.0)
            - (Circle([-4.0, 0.0], 1.5) + Circle([4.0, 0.0], 2.0)),
            [[-4.0, 0.5], [4.5, 0.0], [0.0, 2.0], [11.0, -2.0]],
            id="exclude-union",
        ),
        pytest.param(
            (_rectangle(-12.0, -7.0, -2.0, 7.0) + _rectangle(2.0, -7.0, 12.0, 7.0))
            - Circle([-6.0, 0.0], 2.0),
            [[-6.0, 0.5], [-10.0, 1.0], [5.0, 2.0], [0.5, 1.0]],
            id="exclude-from-union",
        ),
        pytest.param(
            _rectangle(-10.0, -9.0, 10.0, 9.0)
            - (Circle([0.0, 0.0], 4.0) - Circle([0.0, 0.0], 1.0)),
            [[0.5, 0.0], [2.0, 0.0], [5.5, 1.0], [11.0, 0.0]],
            id="nested-island",
        ),
        pytest.param(
            (_rectangle(-10.0, -9.0, 10.0, 9.0) - Circle([0.0, 0.0], 4.0))
            + Circle([0.0, 0.0], 1.0),
            [[0.5, 0.0], [2.0, 0.0], [5.5, 1.0], [11.0, 0.0]],
            id="union-after-exclusion",
        ),
        pytest.param(
            Circle([-2.0, 0.0], 5.0) - Circle([2.0, 0.0], 5.0).inverse(),
            [[0.8, 1.0], [-5.0, 0.5], [5.0, -0.5], [1.0, 6.0]],
            id="intersection-via-subtraction",
        ),
        pytest.param(
            HalfPlane([0.0, 0.0], [1.0, 0.0]),
            [[2.0, 3.0], [-2.0, 3.0], [4.0, -5.0]],
            id="half-plane",
        ),
    ],
)
def test_composed_geometry_derivatives_match_finite_difference(
    geometry: AreaGeometry, points: list[list[float]]
) -> None:
    _assert_matches_finite_difference(geometry, points)


def test_disc_inside_derivatives_match_finite_difference() -> None:
    _assert_matches_finite_difference(
        _rectangle(-10.0, -8.0, 9.0, 7.0),
        [[-7.0, 1.0], [12.0, -2.0]],
        disc_inside=True,
        D=2.0,
    )


@pytest.mark.parametrize("point", [[10.0, 0.0], [0.0, 0.0]])
def test_analytical_derivatives_fail_at_nondifferentiable_points(
    point: list[float],
) -> None:
    problem, constraint = _setup([point])

    with pytest.raises(ValueError, match="Failed to calculate derivatives"):
        problem.get_gradients(
            np.array([], dtype=int), problem.initial_values_float(), func=constraint
        )


@pytest.mark.parametrize(
    ("geometry", "point"),
    [
        (_rectangle(-5.0, -5.0, 5.0, 5.0), [5.0, 1.0]),
        (_rectangle(-5.0, -5.0, 5.0, 5.0), [5.0, 5.0]),
        (_rectangle(-5.0, -5.0, 5.0, 5.0), [0.0, 0.0]),
        (
            Circle([-3.0, 0.0], 1.0) + Circle([3.0, 0.0], 1.0),
            [0.0, 0.0],
        ),
        (
            _rectangle(-10.0, -9.0, 10.0, 9.0) - Circle([0.0, 0.0], 3.0),
            [3.0, 0.0],
        ),
    ],
    ids=[
        "polygon-edge",
        "polygon-vertex",
        "polygon-eqidistant-edges",
        "union-eqidistant-boundaries",
        "excluded-area-boundary",
    ],
)
def test_composed_geometry_derivatives_fail_at_nondifferentiable_points(
    geometry: AreaGeometry, point: list[float]
) -> None:
    problem, constraint = _setup([point], geometry=geometry)

    with pytest.raises(ValueError, match="Failed to calculate derivatives"):
        problem.get_gradients(
            np.array([], dtype=int), problem.initial_values_float(), func=constraint
        )
