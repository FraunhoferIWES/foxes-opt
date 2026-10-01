from collections.abc import Sequence
from typing import Any

import numpy as np
from foxes.config import config
from iwopy import Constraint, Problem
from scipy.spatial.distance import cdist

from foxes_opt.core.area_geometry_derivatives import signed_distance_derivatives
from foxes_opt.core.zero_derivatives import ZeroFloatDerivatives
from foxes_opt.problems.layout.geom_layouts.derivatives import (
    layout_data,
    maximin_distance_derivative,
)


class Valid(ZeroFloatDerivatives, Constraint):
    """
    Validity constraint for purely geometrical layouts problems.

    """

    def __init__(self, problem: Problem, name: str = "valid", **kwargs: Any) -> None:
        """
        Parameters
        ----------
        problem
            The underlying geometrical layout
            optimization problem
        name
            The constraint name
        kwargs
            Additioal parameters for the base class

        """
        super().__init__(
            problem,
            name,
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
            **kwargs,
        )

    def n_components(self) -> int:
        """
        Returns the number of components of the
        function.

        Returns
        -------
        value
            The number of components.

        """
        return 1

    def calc_individual(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        problem_results: tuple[np.ndarray, np.ndarray],
        cmpnts: list[int] | None = None,
    ) -> np.ndarray:
        """
        Calculate values for a single individual of the
        underlying problem.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_vars_int,)
        vars_float
            The float variable values, shape: (n_vars_float,)
        problem_results
            The results of the variable application
            to the problem
        components
            The selected components or None for all

        Returns
        -------
        values
            The component values, shape: (n_sel_components,)

        """
        __, valid = problem_results
        return np.atleast_1d(np.sum(~valid))

    def calc_population(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        problem_results: tuple[np.ndarray, np.ndarray],
        cmpnts: list[int] | None = None,
    ) -> np.ndarray:
        """
        Calculate values for all individuals of a population.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float
            The float variable values, shape: (n_pop, n_vars_float)
        problem_results
            The results of the variable application
            to the problem
        components
            The selected components or None for all

        Returns
        -------
        values
            The component values, shape: (n_pop, n_sel_components)

        """
        __, valid = problem_results
        return np.sum(~valid, axis=1)[:, None]


class Boundary(Constraint):
    """
    Boundary constraint for purely geometrical layouts problems.

    """

    def __init__(
        self,
        problem: Problem,
        n_turbines: int | None = None,
        D: float | None = None,
        name: str = "boundary",
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        problem
            The underlying geometrical layout
            optimization problem
        n_turbines
            The number of turbines
        D
            The rotor diameter
        name
            The constraint name
        kwargs
            Additioal parameters for the base class

        """
        super().__init__(
            problem,
            name,
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
            **kwargs,
        )
        self.n_turbines = problem.n_turbines if n_turbines is None else n_turbines
        self.D = problem.D if D is None else D

    def n_components(self) -> int:
        """
        Returns the number of components of the
        function.

        Returns
        -------
        value
            The number of components.

        """
        return self.n_turbines

    def ana_deriv(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        var: int,
        components: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Calculate the analytical derivative for one float variable.

        Parameters
        ----------
        vars_int
            The integer variable values
        vars_float
            The float variable values
        var
            The float variable index
        components
            The selected components, or None for all

        Returns
        -------
        deriv
            The derivative values, shape: (n_sel_components,)

        """
        cidx = (
            np.arange(self.n_components(), dtype=int)
            if components is None
            else np.asarray(components, dtype=int)
        )
        data = layout_data(self.problem, vars_int, vars_float, var)
        if data is None:
            return np.full(len(cidx), np.nan, dtype=np.float64)
        points, __, derivatives = data
        return signed_distance_derivatives(
            self.problem.boundary, points[cidx], derivatives[cidx]
        )

    def calc_individual(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        problem_results: tuple[np.ndarray, np.ndarray],
        cmpnts: list[int] | None = None,
    ) -> np.ndarray:
        """
        Calculate values for a single individual of the
        underlying problem.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_vars_int,)
        vars_float
            The float variable values, shape: (n_vars_float,)
        problem_results
            The results of the variable application
            to the problem
        components
            The selected components or None for all

        Returns
        -------
        values
            The component values, shape: (n_sel_components,)

        """
        xy, __ = problem_results

        dists = self.problem.boundary.points_distance(xy)
        dists[self.problem.boundary.points_inside(xy)] *= -1

        if self.D is not None:
            dists += self.D / 2

        return dists

    def calc_population(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        problem_results: tuple[np.ndarray, np.ndarray],
        cmpnts: list[int] | None = None,
    ) -> np.ndarray:
        """
        Calculate values for all individuals of a population.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float
            The float variable values, shape: (n_pop, n_vars_float)
        problem_results
            The results of the variable application
            to the problem
        components
            The selected components or None for all

        Returns
        -------
        values
            The component values, shape: (n_pop, n_sel_components)

        """
        xy, __ = problem_results
        n_pop, n_xy = xy.shape[:2]

        xy = xy.reshape(n_pop * n_xy, 2)
        dists = self.problem.boundary.points_distance(xy)
        dists[self.problem.boundary.points_inside(xy)] *= -1
        dists = dists.reshape(n_pop, n_xy)

        if self.D is not None:
            dists += self.D / 2

        return dists


class MinDist(Constraint):
    """
    Minimal distance constraint for purely geometrical layouts problems.

    """

    def __init__(
        self,
        problem: Problem,
        min_dist: float | None = None,
        n_turbines: int | None = None,
        name: str = "min_dist",
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        problem
            The underlying geometrical layout
            optimization problem
        min_dist
            The minimal distance between turbines
        n_turbines
            The number of turbines
        name
            The constraint name
        kwargs
            Additioal parameters for the base class

        """
        super().__init__(
            problem,
            name,
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
            **kwargs,
        )
        self.min_dist = problem.min_dist if min_dist is None else min_dist
        self.n_turbines = problem.n_turbines if n_turbines is None else n_turbines

    def initialize(self, verbosity: int = 0) -> None:
        """
        Initialize the constaint.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent

        """
        N = self.n_turbines
        i2t: list[list[int]] = []  # i --> (ti, tj)
        self._t2i: np.ndarray[tuple[int, ...], np.dtype[Any]] = np.full(
            [N, N], -1
        )  # (ti, tj) --> i
        i = 0
        for ti in range(N):
            for tj in range(N):
                if ti != tj and self._t2i[ti, tj] < 0:
                    i2t.append([ti, tj])
                    self._t2i[ti, tj] = i
                    self._t2i[tj, ti] = i
                    i += 1
        self._i2t: np.ndarray[tuple[int, int], np.dtype[np.int_]] = np.asarray(
            i2t, dtype=int
        )
        self._cnames: list[str] = [f"{self.name}_{ti}_{tj}" for ti, tj in self._i2t]
        super().initialize(verbosity)

    def n_components(self) -> int:
        """
        Returns the number of components of the
        function.

        Returns
        -------
        value
            The number of components.

        """
        return len(self._i2t)

    def ana_deriv(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        var: int,
        components: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Calculate the analytical derivative for one float variable.

        Parameters
        ----------
        vars_int
            The integer variable values
        vars_float
            The float variable values
        var
            The float variable index
        components
            The selected components, or None for all

        Returns
        -------
        deriv
            The derivative values, shape: (n_sel_components,)

        """
        cidx = (
            np.arange(self.n_components(), dtype=int)
            if components is None
            else np.asarray(components, dtype=int)
        )
        data = layout_data(self.problem, vars_int, vars_float, var)
        if data is None:
            return np.full(len(cidx), np.nan, dtype=np.float64)
        points, __, derivatives = data
        pairs = self._i2t[cidx]
        delta = points[pairs[:, 0]] - points[pairs[:, 1]]
        delta_derivative = derivatives[pairs[:, 0]] - derivatives[pairs[:, 1]]
        distance = np.linalg.norm(delta, axis=1)
        out = np.zeros(len(cidx), dtype=np.float64)
        moving = np.any(delta_derivative != 0.0, axis=1)
        differentiable = distance > 0.0
        active = moving & differentiable
        out[active] = (
            -np.einsum("pd,pd->p", delta[active], delta_derivative[active])
            / distance[active]
        )
        out[moving & ~differentiable] = np.nan
        return out

    def calc_individual(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        problem_results: tuple[np.ndarray, np.ndarray],
        cmpnts: list[int] | None = None,
    ) -> np.ndarray:
        """
        Calculate values for a single individual of the
        underlying problem.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_vars_int,)
        vars_float
            The float variable values, shape: (n_vars_float,)
        problem_results
            The results of the variable application
            to the problem
        components
            The selected components or None for all

        Returns
        -------
        values
            The component values, shape: (n_sel_components,)

        """
        xy, __ = problem_results

        a = np.take_along_axis(xy, self._i2t[:, 0, None], axis=0)
        b = np.take_along_axis(xy, self._i2t[:, 1, None], axis=0)
        d = np.linalg.norm(a - b, axis=-1)

        return self.min_dist - d

    def calc_population(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        problem_results: tuple[np.ndarray, np.ndarray],
        cmpnts: list[int] | None = None,
    ) -> np.ndarray:
        """
        Calculate values for all individuals of a population.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float
            The float variable values, shape: (n_pop, n_vars_float)
        problem_results
            The results of the variable application
            to the problem
        components
            The selected components or None for all

        Returns
        -------
        values
            The component values, shape: (n_pop, n_sel_components)

        """
        xy, __ = problem_results

        a = np.take_along_axis(xy, self._i2t[None, :, 0, None], axis=1)
        b = np.take_along_axis(xy, self._i2t[None, :, 1, None], axis=1)
        d = np.linalg.norm(a - b, axis=-1)

        return self.min_dist - d


class CMinN(ZeroFloatDerivatives, Constraint):
    """
    Minimal number of turbines constraint for purely geometrical layouts problems.

    """

    def __init__(
        self, problem: Problem, N: int, name: str = "cminN", **kwargs: Any
    ) -> None:
        super().__init__(
            problem,
            name,
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
            **kwargs,
        )
        """
        Constructor.

        Parameters
        ----------
        problem
            The underlying geometrical layout
            optimization problem
        N
            The minimal number of turbines
        name
            The constraint name
        kwargs
            Additioal parameters for the base class

        """
        self.N = N

    def n_components(self) -> int:
        """
        Returns the number of components of the
        function.

        Returns
        -------
        value
            The number of components.

        """
        return 1

    def calc_individual(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        problem_results: tuple[np.ndarray, np.ndarray],
        cmpnts: list[int] | None = None,
    ) -> np.ndarray:
        """
        Calculate values for a single individual of the
        underlying problem.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_vars_int,)
        vars_float
            The float variable values, shape: (n_vars_float,)
        problem_results
            The results of the variable application
            to the problem
        components
            The selected components or None for all

        Returns
        -------
        values
            The component values, shape: (n_sel_components,)

        """
        __, valid = problem_results
        return np.atleast_1d(self.N - np.sum(valid))

    def calc_population(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        problem_results: tuple[np.ndarray, np.ndarray],
        cmpnts: list[int] | None = None,
    ) -> np.ndarray:
        """
        Calculate values for all individuals of a population.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float
            The float variable values, shape: (n_pop, n_vars_float)
        problem_results
            The results of the variable application
            to the problem
        components
            The selected components or None for all

        Returns
        -------
        values
            The component values, shape: (n_pop, n_sel_components)

        """
        __, valid = problem_results
        return self.N - np.sum(valid, axis=1)[:, None]


class CMaxN(ZeroFloatDerivatives, Constraint):
    """
    Maximal number of turbines constraint for purely geometrical layouts problems.

    """

    def __init__(
        self, problem: Problem, N: int, name: str = "cmaxN", **kwargs: Any
    ) -> None:
        """
        Parameters
        ----------
        problem
            The underlying geometrical layout
            optimization problem
        N
            The maximal number of turbines
        name
            The constraint name
        kwargs
            Additioal parameters for the base class

        """
        super().__init__(
            problem,
            name,
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
            **kwargs,
        )
        self.N = N

    def n_components(self) -> int:
        """
        Returns the number of components of the
        function.

        Returns
        -------
        value
            The number of components.

        """
        return 1

    def calc_individual(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        problem_results: tuple[np.ndarray, np.ndarray],
        cmpnts: list[int] | None = None,
    ) -> np.ndarray:
        """
        Calculate values for a single individual of the
        underlying problem.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_vars_int,)
        vars_float
            The float variable values, shape: (n_vars_float,)
        problem_results
            The results of the variable application
            to the problem
        components
            The selected components or None for all

        Returns
        -------
        values
            The component values, shape: (n_sel_components,)

        """
        __, valid = problem_results
        return np.atleast_1d(np.sum(valid) - self.N)

    def calc_population(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        problem_results: tuple[np.ndarray, np.ndarray],
        cmpnts: list[int] | None = None,
    ) -> np.ndarray:
        """
        Calculate values for all individuals of a population.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float
            The float variable values, shape: (n_pop, n_vars_float)
        problem_results
            The results of the variable application
            to the problem
        components
            The selected components or None for all

        Returns
        -------
        values
            The component values, shape: (n_pop, n_sel_components)

        """
        __, valid = problem_results
        return np.sum(valid, axis=1)[:, None] - self.N


class CFixN(ZeroFloatDerivatives, Constraint):
    """
    Fixed number of turbines constraint for purely geometrical layouts problems.

    """

    def __init__(
        self, problem: Problem, N: int, name: str = "cfixN", **kwargs: Any
    ) -> None:
        """
        Parameters
        ----------
        problem
            The underlying geometrical layout
            optimization problem
        N
            The number of turbines
        name
            The constraint name
        kwargs
            Additioal parameters for the base class

        """
        super().__init__(
            problem,
            name,
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
            tol=0.1,
            **kwargs,
        )
        self.N = N

    def n_components(self) -> int:
        """
        Returns the number of components of the
        function.

        Returns
        -------
        value
            The number of components.

        """
        return 2

    def calc_individual(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        problem_results: tuple[np.ndarray, np.ndarray],
        cmpnts: list[int] | None = None,
    ) -> np.ndarray:
        """
        Calculate values for a single individual of the
        underlying problem.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_vars_int,)
        vars_float
            The float variable values, shape: (n_vars_float,)
        problem_results
            The results of the variable application
            to the problem
        components
            The selected components or None for all

        Returns
        -------
        values
            The component values, shape: (n_sel_components,)

        """
        __, valid = problem_results
        vld = np.sum(valid)
        return np.array([self.N - vld, vld - self.N])

    def calc_population(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        problem_results: tuple[np.ndarray, np.ndarray],
        cmpnts: list[int] | None = None,
    ) -> np.ndarray:
        """
        Calculate values for all individuals of a population.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float
            The float variable values, shape: (n_pop, n_vars_float)
        problem_results
            The results of the variable application
            to the problem
        components
            The selected components or None for all

        Returns
        -------
        values
            The component values, shape: (n_pop, n_sel_components)

        """
        __, valid = problem_results
        vld = np.sum(valid, axis=1)
        return np.stack([self.N - vld, vld - self.N], axis=-1)


class CMinDensity(Constraint):
    """
    Minimal turbine density constraint for purely geometrical layouts problems.

    """

    def __init__(
        self,
        problem: Problem,
        min_value: float,
        dfactor: int = 1,
        name: str = "min_density",
    ) -> None:
        """
        Parameters
        ----------
        problem
            The underlying geometrical layout
            optimization problem
        min_value
            The minimal turbine density
        dfactor
            Delta factor for grid spacing
        name
            The constraint name
        kwargs
            Additioal parameters for the base class

        """
        super().__init__(
            problem,
            name,
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
        )
        self.min_value = min_value
        self.dfactor = dfactor

    def n_components(self) -> int:
        """
        Returns the number of components of the
        function.

        Returns
        -------
        value
            The number of components.

        """
        return 1

    def initialize(self, verbosity: int) -> None:
        """
        Initialize the object.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent

        """
        super().initialize(verbosity)

        # define regular grid of probe points:
        geom = self.problem.boundary
        pmin = geom.p_min()
        pmax = geom.p_max()
        detlta = self.problem.min_dist / self.dfactor
        self._probes = np.stack(
            np.meshgrid(
                np.arange(pmin[0] - detlta, pmax[0] + 2 * detlta, detlta),
                np.arange(pmin[1] - detlta, pmax[1] + 2 * detlta, detlta),
                indexing="ij",
            ),
            axis=-1,
        )
        nx, ny = self._probes.shape[:2]
        n: int = nx * ny
        self._probes = self._probes.reshape(n, 2)

        # reduce to points within geometry:
        valid = geom.points_inside(self._probes)
        self._probes = self._probes[valid]

    def ana_deriv(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        var: int,
        components: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Calculate the analytical derivative for one float variable.

        Parameters
        ----------
        vars_int
            The integer variable values
        vars_float
            The float variable values
        var
            The float variable index
        components
            The selected components, or None for all

        Returns
        -------
        deriv
            The derivative values, shape: (n_sel_components,)

        """
        n_components = self.n_components() if components is None else len(components)
        if not n_components:
            return np.empty(0, dtype=np.float64)
        data = layout_data(self.problem, vars_int, vars_float, var)
        if data is None:
            return np.full(n_components, np.nan, dtype=np.float64)
        points, valid, derivatives = data
        value = maximin_distance_derivative(self._probes, points, valid, derivatives)
        return np.full(n_components, value, dtype=np.float64)

    def calc_individual(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        problem_results: tuple[np.ndarray, np.ndarray],
        cmpnts: list[int] | None = None,
    ) -> np.ndarray:
        """
        Calculate values for a single individual of the
        underlying problem.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_vars_int,)
        vars_float
            The float variable values, shape: (n_vars_float,)
        problem_results
            The results of the variable application
            to the problem
        components
            The selected components or None for all

        Returns
        -------
        values
            The component values, shape: (n_sel_components,)

        """
        xy, valid = problem_results
        xy = xy[valid]
        dists: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = cdist(
            self._probes, xy
        )
        return np.atleast_1d(np.nanmax(np.nanmin(dists, axis=1)) - self.min_value)

    def calc_population(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        problem_results: tuple[np.ndarray, np.ndarray],
        cmpnts: list[int] | None = None,
    ) -> np.ndarray:
        """
        Calculate values for all individuals of a population.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float
            The float variable values, shape: (n_pop, n_vars_float)
        problem_results
            The results of the variable application
            to the problem
        components
            The selected components or None for all

        Returns
        -------
        values
            The component values, shape: (n_pop, n_sel_components)

        """
        n_pop = vars_float.shape[0]
        xy, valid = problem_results
        out = np.full(n_pop, 1e20, dtype=config.dtype_double)
        for pi in range(n_pop):
            if np.any(valid[pi]):
                hxy = xy[pi][valid[pi]]
                dists: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = cdist(
                    self._probes, hxy
                )
                out[pi] = np.nanmax(np.nanmin(dists, axis=1)) - self.min_value
        return out[:, None]
