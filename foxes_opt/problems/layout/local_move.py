from typing import Any

import numpy as np
from foxes.config import config

from foxes_opt.core.farm_opt_problem import FarmOptProblem


class DiscreteLocalMoveOptProblem(FarmOptProblem):
    """
    Move turbines on a local discrete grid, cut to a given radius

    """

    def __init__(
        self,
        *args: Any,
        radius: float,
        spacing: float,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        args
            Additional positional arguments passed to the base class initializer.
        radius
            The radius within which turbines can be locally moved.
        spacing
            The spacing of points in meters.
        kwargs
            Additional keyword arguments passed to the base class initializer.
        
        """
        super().__init__(*args, **kwargs)
        self.radius = radius
        self.spacing = spacing

    def initialize(self, verbosity: int = 1) -> None:
        """
        Initialize the problem.

        Parameters
        ----------
        verbosity: int
            The verbosity level, 0 = silent

        """
        if not self.initialized:
            self._layout0 = np.zeros((self.n_sel_turbines, 2), dtype=config.dtype_double)
            for i, ti in enumerate(self.sel_turbines):
                self._layout0[i] = self.algo.farm.turbines[ti].xy

            assert self.spacing > 0, f"{self.name}: spacing must be positive"
            assert self.radius >= 2 * self.spacing, f"{self.name}: radius must be at least twice the spacing"

            dx = np.arange(0.0, self.radius + self.spacing, self.spacing)
            dx = np.concatenate([np.flip(-dx[1:]), dx], axis=0)
            self._delta_pts = np.zeros((dx.size, dx.size, 2), dtype=config.dtype_double)
            self._delta_pts[:, :, 0] = dx[:, None]
            self._delta_pts[:, :, 1] = dx[None, :]
            self._delta_pts = self._delta_pts.reshape(-1, 2)
            self._delta_pts = self._delta_pts[np.linalg.norm(self._delta_pts, axis=1) <= self.radius]
            self._N = len(self._delta_pts)
            self._i0 = np.argwhere(np.linalg.norm(self._delta_pts, axis=1) == 0.0).flatten()
            assert self._i0.size == 1, f"{self.name}: There should be exactly one zero delta point"
            self._i0 = self._i0[0]
            if verbosity > 0:
                print(f"{self.name}: Generated {self._N} local move points within radius {self.radius}. Min {self._delta_pts.min(axis=0)}, max {self._delta_pts.max(axis=0)}")
            
            super().initialize(verbosity=verbosity)

    def var_names_int(self) -> list[str]:
        """
        The names of integer variables.

        Returns
        -------
        names
            The names of the integer variables

        """
        vrs = []
        for ti in self.sel_turbines:
            vrs += [self.tvar("i", ti)]
        return vrs

    def initial_values_int(self) -> np.ndarray:
        """
        The initial values of the integer variables.

        Returns
        -------
        values
            Initial integer values, shape: (n_vars_int,)

        """
        return np.full(self.n_sel_turbines, self._i0, dtype=config.dtype_int)

    def min_values_int(self) -> np.ndarray:
        """
        The minimal values of the integer variables.

        Use -numpy.inf for unbounded.

        Returns
        -------
        values
            Minimal integer values, shape: (n_vars_int,)

        """
        return np.zeros(self.n_sel_turbines, dtype=config.dtype_int)

    def max_values_int(self) -> np.ndarray:
        """
        The maximal values of the integer variables.

        Use numpy.inf for unbounded.

        Returns
        -------
        values
            Maximal integer values, shape: (n_vars_int,)

        """
        return np.full(self.n_sel_turbines, self._N - 1, dtype=config.dtype_int)
        
    def update_problem_individual(
        self, vars_int: np.ndarray, vars_float: np.ndarray
    ) -> None:
        """
        Update the algo and other data using
        the latest optimization variables.

        This function is called before running the farm
        calculation.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_vars_int,)
        vars_float
            The float variable values, shape: (n_vars_float,)

        """
        super().update_problem_individual(vars_int, vars_float)
        for i, ti in enumerate(self.sel_turbines):
            t = self.algo.farm.turbines[ti]
            t.xy = self._layout0[i] + self._delta_pts[vars_int[i]]

    def update_problem_population(
        self, vars_int: np.ndarray, vars_float: np.ndarray
    ) -> None:
        """
        Update the algo and other data using
        the latest optimization variables.

        This function is called before running the farm
        calculation.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_pop, n_vars_int,)
        vars_float
            The float variable values, shape: (n_pop, n_vars_float,)

        """
        super().update_problem_population(vars_int, vars_float)

        n_pop: int = len(vars_int)
        n_ostates = self._org_n_states
        assert n_ostates is not None
        n_states = n_pop * n_ostates

        inds = np.broadcast_to(vars_int, (n_ostates, n_pop, self.n_sel_turbines))
        inds = inds.reshape(n_states, self.n_sel_turbines)

        for i, ti in enumerate(self.sel_turbines):
            t = self.algo.farm.turbines[ti]
            t.xy = self._layout0[None, i] + self._delta_pts[inds[:, i]]
