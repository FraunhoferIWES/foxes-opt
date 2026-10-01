from collections.abc import Sequence

import numpy as np
from foxes.config import config
from iwopy import Objective, Problem
from scipy.spatial.distance import cdist

from foxes_opt.core.zero_derivatives import ZeroFloatDerivatives
from foxes_opt.problems.layout.geom_layouts.derivatives import (
    extreme_derivative,
    layout_data,
    maximin_distance_derivative,
    nearest_distance_derivatives,
)


class OMaxN(ZeroFloatDerivatives, Objective):
    """
    Maximal number of turbines objective
    for purely geometrical layouts problems.

    """

    def __init__(self, problem: Problem, name: str = "maxN") -> None:
        """
        Parameters
        ----------
        problem
            The underlying geometrical layout
            optimization problem
        name
            The constraint name

        """
        super().__init__(
            problem,
            name,
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
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

    def maximize(self) -> list[bool]:
        """
        Returns flag for maximization of each component.

        Returns
        -------
        flags
            Bool array for component maximization,
            shape

        """
        return [True]

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
        return np.atleast_1d(np.sum(valid))

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
        return np.sum(valid, axis=1)[:, None]


class OMinN(OMaxN):
    """
    Minimal number of turbines objective
    for purely geometrical layouts problems.

    """

    def __init__(self, problem: Problem, name: str = "ominN") -> None:
        """
        Parameters
        ----------
        problem
            The underlying geometrical layout
            optimization problem
        name
            The constraint name

        """
        super().__init__(problem, name)

    def maximize(self) -> list[bool]:
        return [False]


class OFixN(ZeroFloatDerivatives, Objective):
    """
    Fixed number of turbines objective
    for purely geometrical layouts problems.

    """

    def __init__(self, problem: Problem, N: int, name: str = "ofixN") -> None:
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

        """
        super().__init__(
            problem,
            name,
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
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

    def maximize(self) -> list[bool]:
        """
        Returns flag for maximization of each component.

        Returns
        -------
        flags
            Bool array for component maximization,
            shape

        """
        return [False]

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
        N: np.float64 = np.sum(valid, dtype=np.float64)
        return np.atleast_1d(np.maximum(N - self.N, self.N - N))

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
        N = np.sum(valid, axis=1, dtype=np.float64)[:, None]
        return np.maximum(N - self.N, self.N - N)


class MaxGridSpacing(Objective):
    """
    Maximal grid spacing objective
    for purely geometrical layouts problems.

    """

    def __init__(self, problem: Problem, name: str = "max_dxdy") -> None:
        """
        Parameters
        ----------
        problem
            The underlying geometrical layout
            optimization problem
        name
            The constraint name

        """
        super().__init__(
            problem,
            name,
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
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

    def maximize(self) -> list[bool]:
        """
        Returns flag for maximization of each component.

        Returns
        -------
        flags
            Bool array for component maximization,
            shape

        """
        return [True]

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
        grid, grid_var = divmod(var, 5)
        if grid_var not in (2, 3):
            return np.zeros(n_components, dtype=np.float64)
        spacing = vars_float.reshape(self.problem.n_grids, 5)[:, 2:4]
        minimum = np.nanmin(spacing)
        active = np.argwhere(np.isclose(spacing, minimum))
        target = np.array([grid, grid_var - 2])
        if not np.any(np.all(active == target, axis=1)):
            return np.zeros(n_components, dtype=np.float64)
        value = 1.0 if len(active) == 1 else np.nan
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
        vflt = vars_float.reshape(self.problem.n_grids, 5)
        delta = np.minimum(vflt[:, 2], vflt[:, 3])
        return np.atleast_1d(np.nanmin(delta))

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
        vflt = vars_float.reshape(n_pop, self.problem.n_grids, 5)
        delta = np.minimum(vflt[:, :, 2], vflt[:, :, 3])
        return np.nanmin(delta, axis=1)[:, None]


class MaxDensity(Objective):
    """
    Maximal turbine density objective
    for purely geometrical layouts problems.

    """

    def __init__(
        self,
        problem: Problem,
        dfactor: int = 1,
        min_dist: float | None = None,
        name: str = "max_density",
    ) -> None:
        """
        Parameters
        ----------
        problem
            The underlying geometrical layout
            optimization problem
        dfactor
            Delta factor for grid spacing
        min_dist
            The minimal distance
        name
            The constraint name

        """
        super().__init__(
            problem,
            name,
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
        )
        self.dfactor: int = dfactor
        self.min_dist = problem.min_dist if min_dist is None else min_dist

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

    def maximize(self) -> list[bool]:
        """
        Returns flag for maximization of each component.

        Returns
        -------
        flags
            Bool array for component maximization,
            shape

        """
        return [False]

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
        detlta = self.min_dist / self.dfactor
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
        return np.atleast_1d(np.nanmax(np.nanmin(dists, axis=1)))

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
                out[pi] = np.nanmax(np.nanmin(dists, axis=1))
        return out[:, None]


class MeMiMaDist(Objective):
    """
    Mean-min-max distance objective
    for purely geometrical layouts problems.

    """

    def __init__(
        self,
        problem: Problem,
        scale: float = 500.0,
        c1: int = 1,
        c2: int = 1,
        c3: int = 1,
        name: str = "MiMaMean",
    ) -> None:
        """
        Parameters
        ----------
        problem
            The underlying geometrical layout
            optimization problem
        scale
            The distance scale
        c1
            Parameter for mean weighting
        c2
            Parameter for max diff weighting
        c3
            Parameter for min diff weighting
        name
            The constraint name

        """
        super().__init__(
            problem,
            name,
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
        )
        self.scale: float = scale
        self.c1: int = c1
        self.c2: int = c2
        self.c3: int = c3

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

    def maximize(self) -> list[bool]:
        """
        Returns flag for maximization of each component.

        Returns
        -------
        flags
            Bool array for component maximization,
            shape

        """
        return [True]

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
        points, __, point_derivatives = data
        distances, derivatives = nearest_distance_derivatives(
            points, point_derivatives, self.scale * len(points)
        )
        mean = np.mean(distances)
        minimum = np.min(distances)
        maximum = np.max(distances)
        dmean = np.mean(derivatives)
        dminimum = extreme_derivative(distances, derivatives, maximize=False)
        dmaximum = extreme_derivative(distances, derivatives, maximize=True)
        value = (
            2 * self.c1 * mean * dmean
            - 2 * self.c2 * (mean - minimum) * (dmean - dminimum)
            - 2 * self.c3 * (mean - maximum) * (dmean - dmaximum)
        )
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
        xy, _valid = problem_results
        # xy = xy[valid]

        dists: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = cdist(xy, xy)
        np.fill_diagonal(dists, np.inf)
        dists = np.min(dists, axis=1) / self.scale / len(xy)

        mean: np.floating = np.average(dists)
        mi: np.floating = np.min(dists)
        ma: np.floating = np.max(dists)
        return np.atleast_1d(
            self.c1 * mean**2 - self.c2 * (mean - mi) ** 2 - self.c3 * (mean - ma) ** 2
        )

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
        xy, _valid = problem_results
        n_pop, n_xy = xy.shape[:2]

        out = np.zeros((n_pop, 1), dtype=config.dtype_double)
        for pi in range(n_pop):
            hxy = xy[pi]  # , valid[pi]]

            dists: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = cdist(hxy, hxy)
            np.fill_diagonal(dists, np.inf)
            dists = np.min(dists, axis=1) / self.scale / n_xy

            mean: np.floating = np.average(dists)
            mi: np.floating = np.min(dists)
            ma: np.floating = np.max(dists)
            out[pi, 0] = (
                self.c1 * mean**2
                - self.c2 * (mean - mi) ** 2
                - self.c3 * (mean - ma) ** 2
            )

        return out
