from typing import Any

import numpy as np

from foxes_opt.core.farm_constraint import FarmConstraint
from foxes_opt.core.farm_opt_problem import FarmOptProblem
import foxes.variables as FV
import foxes.constants as FC


class MinDistConstraint(FarmConstraint):
    """
    Turbines must keep at least a minimal
    spatial distance.

    Attributes
    ----------
    farm
        The wind farm
    sel_turbines
        The selected turbines
    min_dist
        The minimal distance
    min_dist_unit
        The minimal distance unit, either m or D


    """

    def __init__(
        self,
        problem: FarmOptProblem,
        min_dist: float,
        min_dist_unit: str = "m",
        name: str = "dist",
        sel_turbines: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        problem
            The underlying optimization problem
        min_dist
            The minimal distance
        min_dist_unit
            The minimal distance unit, either m or D
        name
            The name of the constraint
        sel_turbines
            The selected turbines
        kwargs
            Additional parameters for `iwopy.Constraint`

        """
        self.min_dist: float = min_dist
        self.min_dist_unit: str = min_dist_unit

        selt = problem.sel_turbines if sel_turbines is None else sel_turbines
        vrs = []
        for ti in selt:
            vrs += [problem.tvar(FV.X, ti), problem.tvar(FV.Y, ti)]

        super().__init__(problem, name, sel_turbines, vnames_float=vrs, **kwargs)

    def initialize(self, verbosity: int = 0) -> None:
        """
        Initialize the constaint.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent

        """
        N = self.farm.n_turbines
        i2t: list[list[int]] = []  # i --> (ti, tj)
        self._t2i: np.ndarray[tuple[int, int], np.dtype[np.int_]] = np.full(
            [N, N], -1
        )  # (ti, tj) --> i
        i = 0
        for ti in self.sel_turbines:
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

    def vardeps_float(self) -> np.ndarray[tuple[int, int], np.dtype[Any]]:
        """
        Gets the dependencies of all components
        on the function float variables

        Returns
        -------
        deps
            The dependencies of components on function
            variables, shape

        """
        turbs = list(self.problem.sel_turbines)
        deps: np.ndarray[tuple[int, int, int], np.dtype[Any]] = np.zeros(
            (self.n_components(), len(turbs), 2), dtype=bool
        )
        for i, titj in enumerate(self._i2t):
            for t in titj:
                if t in turbs:
                    j: int = turbs.index(t)
                    deps[i, j] = True
        return deps.reshape(self.n_components(), 2 * len(turbs))

    def calc_individual(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        problem_results: Any,
        components: list[int] | None = None,
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
        xy = np.stack(
            [problem_results[FV.X].to_numpy(), problem_results[FV.Y].to_numpy()],
            axis=-1,
        )
        if not np.all(np.abs(np.min(xy, axis=0) - np.max(xy, axis=0)) < 1e-13):
            raise ValueError(f"Constraint '{self.name}': Require state independet XY")
        xy = xy[0]

        cidx = np.arange(self.n_components(), dtype=int)
        if components is not None and len(components) < self.n_components():
            cidx = np.asarray(components, dtype=int)

        a: np.ndarray[tuple[int, ...], np.dtype[Any]] = np.take_along_axis(
            xy, self._i2t[cidx][:, 0, None], axis=0
        )
        b: np.ndarray[tuple[int, ...], np.dtype[Any]] = np.take_along_axis(
            xy, self._i2t[cidx][:, 1, None], axis=0
        )
        d = np.linalg.norm(a - b, axis=-1)

        if self.min_dist_unit == "m":
            mind: Any = self.min_dist

        elif self.min_dist_unit == "D":
            D = problem_results[FV.D].to_numpy()
            if not np.all(np.abs(np.min(D, axis=0) - np.max(D, axis=0)) < 1e-13):
                raise ValueError(
                    f"Constraint '{self.name}': Require state independet D"
                )
            D = D[0]

            Da = np.take_along_axis(D, self._i2t[cidx][:, 0], axis=0)
            Db = np.take_along_axis(D, self._i2t[cidx][:, 1], axis=0)
            mind = self.min_dist * np.maximum(Da, Db)

        return mind - d

    def calc_population(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        problem_results: Any,
        components: list[int] | None = None,
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
        n_pop = problem_results["n_pop"].values
        n_states = problem_results["n_org_states"].values
        n_turbines = problem_results.sizes[FC.TURBINE]

        xy = np.stack(
            [problem_results[FV.X].to_numpy(), problem_results[FV.Y].to_numpy()],
            axis=-1,
        )

        xy = xy.reshape(n_states, n_pop, n_turbines, 2)
        if not np.all(np.abs(np.min(xy, axis=0) - np.max(xy, axis=0)) < 1e-13):
            raise ValueError(f"Constraint '{self.name}': Require state independet XY")
        xy = xy[0]

        cidx = np.arange(self.n_components(), dtype=int)
        if components is not None and len(components) < self.n_components():
            cidx = np.asarray(components, dtype=int)

        a: np.ndarray[tuple[int, ...], np.dtype[Any]] = np.take_along_axis(
            xy, self._i2t[cidx][None, :, 0, None], axis=1
        )
        b: np.ndarray[tuple[int, ...], np.dtype[Any]] = np.take_along_axis(
            xy, self._i2t[cidx][None, :, 1, None], axis=1
        )
        d = np.linalg.norm(a - b, axis=-1)

        if self.min_dist_unit == "m":
            mind: Any = self.min_dist

        elif self.min_dist_unit == "D":
            D = problem_results[FV.D].to_numpy().reshape(n_states, n_pop, n_turbines)
            if not np.all(np.abs(np.min(D, axis=0) - np.max(D, axis=0)) < 1e-13):
                raise ValueError(
                    f"Constraint '{self.name}': Require state independet D"
                )
            D = D[0]

            Da = np.take_along_axis(D, self._i2t[cidx][None, :, 0], axis=1)
            Db = np.take_along_axis(D, self._i2t[cidx][None, :, 1], axis=1)
            mind = self.min_dist * np.maximum(Da, Db)

        return mind - d
