from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
from iwopy import SimpleProblem
from iwopy.core import OptFunction

from foxes_opt.objectives import MaxNTurbines
from foxes_opt.problems.layout.geom_layouts.constraints import (
    CFixN,
    CMaxN,
    CMinN,
    Valid,
)
from foxes_opt.problems.layout.geom_layouts.objectives import OFixN, OMaxN, OMinN


class _Problem(SimpleProblem):
    def __init__(self) -> None:
        super().__init__(
            "discrete", float_vars=["x", "y"], init_values_float=[1.0, 2.0]
        )
        self.farm = SimpleNamespace(n_turbines=3)


@pytest.mark.parametrize(
    ("factory", "n_components"),
    [
        (lambda problem: MaxNTurbines(problem), 1),
        (lambda problem: Valid(problem), 1),
        (lambda problem: CMinN(problem, N=2), 1),
        (lambda problem: CMaxN(problem, N=2), 1),
        (lambda problem: CFixN(problem, N=2), 2),
        (lambda problem: OMaxN(problem), 1),
        (lambda problem: OMinN(problem), 1),
        (lambda problem: OFixN(problem, N=2), 1),
    ],
)
def test_piecewise_constant_derivatives_via_iwopy(
    factory: Callable[[_Problem], OptFunction], n_components: int
) -> None:
    problem = _Problem()
    function = factory(problem)
    function.initialize()
    variables = problem.initial_values_float()

    gradients = problem.get_gradients(np.array([], dtype=int), variables, func=function)

    np.testing.assert_array_equal(gradients, np.zeros((n_components, 2)))
    selected = problem.get_gradients(
        np.array([], dtype=int),
        variables,
        func=function,
        components=[n_components - 1],
        vars=["y"],
    )
    np.testing.assert_array_equal(selected, [[0.0]])
