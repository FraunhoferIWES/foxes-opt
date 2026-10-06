from typing import Any

import numpy as np
from foxes.config import config

from foxes_opt.core.farm_opt_problem import FarmOptProblem
import foxes.variables as FV


class LocalMove(FarmOptProblem):
    """
    Base class for local move optimization problems.

    """

    def __init__(
        self,
        *args: Any,
        radius: float,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        args
            Additional positional arguments passed to the base class initializer.
        radius
            The radius within which turbines can be locally moved.
        kwargs
            Additional keyword arguments passed to the base class initializer.
        """
        super().__init__(*args, **kwargs)
        self.radius = radius

    def initialize(self, verbosity: int = 1) -> None:
        """
        Initialize the problem.

        Parameters
        ----------
        verbosity: int
            The verbosity level, 0 = silent

        """
        if not self.initialized:
            self._layout0 = np.zeros(
                (self.n_sel_turbines, 2), dtype=config.dtype_double
            )
            for i, ti in enumerate(self.sel_turbines):
                self._layout0[i] = self.algo.farm.turbines[ti].xy

            assert self.radius > 0, f"{self.name}: radius must be positive"

            super().initialize(verbosity=verbosity)

    def var_names_float(self) -> list[str]:
        """
        The names of float variables.

        Returns
        -------
        names
            The names of the float variables

        """
        vrs = []
        for ti in self.sel_turbines:
            vrs += [self.tvar("r", ti), self.tvar("theta", ti)]
        return vrs

    def initial_values_float(self) -> np.ndarray:
        """
        The initial values of the float variables.

        Returns
        -------
        values
            Initial float values, shape: (n_vars_float,)

        """
        return np.zeros(self.n_sel_turbines * 2, dtype=config.dtype_double)

    def min_values_float(self) -> np.ndarray:
        """
        The minimal values of the float variables.

        Use -numpy.inf for unbounded.

        Returns
        -------
        values
            Minimal float values, shape: (n_vars_float,)

        """
        return np.zeros(self.n_sel_turbines * 2, dtype=config.dtype_double)

    def max_values_float(self) -> np.ndarray:
        """
        The maximal values of the float variables.

        Use numpy.inf for unbounded.

        Returns
        -------
        values
            Maximal float values, shape: (n_vars_float,)

        """
        out = np.zeros((self.n_sel_turbines, 2), dtype=config.dtype_double)
        out[:, 0] = self.radius
        out[:, 1] = 2 * np.pi
        return out.reshape(self.n_sel_turbines * 2)

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
        r_theta = vars_float.reshape(self.n_sel_turbines, 2)
        for i, ti in enumerate(self.sel_turbines):
            t = self.algo.farm.turbines[ti]
            t.xy = self._layout0[i] + r_theta[i, 0] * np.array(
                [np.cos(r_theta[i, 1]), np.sin(r_theta[i, 1])]
            )

        super().update_problem_individual(vars_int, vars_float)

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
        n_pop: int = len(vars_float)
        n_ostates = self._org_n_states
        assert n_ostates is not None
        n_states = n_pop * n_ostates

        r_theta = vars_float.reshape(n_pop, self.n_sel_turbines, 2)
        r_theta = np.broadcast_to(
            r_theta[:, None, ...], (n_pop, n_ostates, self.n_sel_turbines, 2)
        )
        r_theta = r_theta.reshape(n_states, self.n_sel_turbines, 2)
        for i, ti in enumerate(self.sel_turbines):
            t = self.algo.farm.turbines[ti]
            t.xy = self._layout0[None, i, :] + r_theta[:, i, 0, None] * np.stack(
                [np.cos(r_theta[:, i, 1]), np.sin(r_theta[:, i, 1])], axis=-1
            )

        super().update_problem_population(vars_int, vars_float)


class LocalSquareMove(FarmOptProblem):
    """
    Base class for local square move optimization problems.

    """

    def __init__(
        self,
        *args: Any,
        square_length: float,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        args
            Additional positional arguments passed to the base class initializer.
        square_length
            The side length of the square within which turbines can be locally moved.
        kwargs
            Additional keyword arguments passed to the base class initializer.
        """
        super().__init__(*args, **kwargs)
        self.square_length = square_length

    def initialize(self, verbosity: int = 1) -> None:
        """
        Initialize the problem.

        Parameters
        ----------
        verbosity: int
            The verbosity level, 0 = silent

        """
        if not self.initialized:
            self._layout0 = np.zeros(
                (self.n_sel_turbines, 2), dtype=config.dtype_double
            )
            for i, ti in enumerate(self.sel_turbines):
                self._layout0[i] = self.algo.farm.turbines[ti].xy

            assert self.square_length > 0, (
                f"{self.name}: square_length must be positive"
            )

            super().initialize(verbosity=verbosity)

    def var_names_float(self) -> list[str]:
        """
        The names of float variables.

        Returns
        -------
        names
            The names of the float variables

        """
        vrs = []
        for ti in self.sel_turbines:
            vrs += [self.tvar(FV.X, ti), self.tvar(FV.Y, ti)]
        return vrs

    def initial_values_float(self) -> np.ndarray:
        """
        The initial values of the float variables.

        Returns
        -------
        values
            Initial float values, shape: (n_vars_float,)

        """
        return self._layout0.reshape(self.n_sel_turbines * 2).copy()

    def min_values_float(self) -> np.ndarray:
        """
        The minimal values of the float variables.

        Use -numpy.inf for unbounded.

        Returns
        -------
        values
            Minimal float values, shape: (n_vars_float,)

        """
        xy = np.zeros((self.n_sel_turbines, 2), dtype=config.dtype_double)
        for i in range(self.n_sel_turbines):
            xy[i] = self._layout0[i] - self.square_length / 2
        return xy.reshape(self.n_sel_turbines * 2)

    def max_values_float(self) -> np.ndarray:
        """
        The maximal values of the float variables.

        Use numpy.inf for unbounded.

        Returns
        -------
        values
            Maximal float values, shape: (n_vars_float,)

        """
        xy = np.zeros((self.n_sel_turbines, 2), dtype=config.dtype_double)
        for i in range(self.n_sel_turbines):
            xy[i] = self._layout0[i] + self.square_length / 2
        return xy.reshape(self.n_sel_turbines * 2)

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
        xy = vars_float.reshape(self.n_sel_turbines, 2)
        for i, ti in enumerate(self.sel_turbines):
            t = self.algo.farm.turbines[ti]
            t.xy = xy[i]

        super().update_problem_individual(vars_int, vars_float)

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
        n_pop: int = len(vars_float)
        n_ostates = self._org_n_states
        assert n_ostates is not None
        n_states = n_pop * n_ostates

        xy = vars_float.reshape(n_pop, self.n_sel_turbines, 2)
        xy = np.broadcast_to(
            xy[:, None, ...], (n_pop, n_ostates, self.n_sel_turbines, 2)
        )
        xy = xy.reshape(n_states, self.n_sel_turbines, 2)
        for i, ti in enumerate(self.sel_turbines):
            t = self.algo.farm.turbines[ti]
            t.xy = xy[:, i, :]

        super().update_problem_population(vars_int, vars_float)


class DiscreteLocalMove(FarmOptProblem):
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
            self._layout0 = np.zeros(
                (self.n_sel_turbines, 2), dtype=config.dtype_double
            )
            for i, ti in enumerate(self.sel_turbines):
                self._layout0[i] = self.algo.farm.turbines[ti].xy

            assert self.spacing > 0, f"{self.name}: spacing must be positive"
            assert self.radius >= 2 * self.spacing, (
                f"{self.name}: radius must be at least twice the spacing"
            )

            dx = np.arange(0.0, self.radius + self.spacing, self.spacing)
            dx = np.concatenate([np.flip(-dx[1:]), dx], axis=0)
            self._delta_pts = np.zeros((dx.size, dx.size, 2), dtype=config.dtype_double)
            self._delta_pts[:, :, 0] = dx[:, None]
            self._delta_pts[:, :, 1] = dx[None, :]
            self._delta_pts = self._delta_pts.reshape(-1, 2)
            self._delta_pts = self._delta_pts[
                np.linalg.norm(self._delta_pts, axis=1) <= self.radius
            ]
            self._N = len(self._delta_pts)
            self._i0 = np.argwhere(
                np.linalg.norm(self._delta_pts, axis=1) == 0.0
            ).flatten()
            assert self._i0.size == 1, (
                f"{self.name}: There should be exactly one zero delta point"
            )
            self._i0 = self._i0[0]
            if verbosity > 0:
                print(
                    f"{self.name}: Generated {self._N} local move points within radius {self.radius}. Min {self._delta_pts.min(axis=0)}, max {self._delta_pts.max(axis=0)}"
                )

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
        for i, ti in enumerate(self.sel_turbines):
            t = self.algo.farm.turbines[ti]
            t.xy = self._layout0[i, :] + self._delta_pts[vars_int[i], :]

        super().update_problem_individual(vars_int, vars_float)

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
        n_pop: int = len(vars_int)
        n_ostates = self._org_n_states
        assert n_ostates is not None
        n_states = n_pop * n_ostates

        inds = np.broadcast_to(
            vars_int[:, None, :], (n_pop, n_ostates, self.n_sel_turbines)
        )
        inds = inds.reshape(n_states, self.n_sel_turbines)

        """
        # save debug image, showing all candidate positions for this turbine and the selected one:
        import matplotlib.pyplot as plt
        D = 282.0
        min_dist = 2 * D
        plt.figure(figsize=(10, 10))
        plt.scatter(self._layout0[:,0],self._layout0[:,1], s=2,c='gray', label='Original')
        """

        for i, ti in enumerate(self.sel_turbines):
            t = self.algo.farm.turbines[ti]
            t.xy = self._layout0[None, i, :] + self._delta_pts[inds[:, i]]

            """
            plt.scatter(self._layout0[i,0,None] + self._delta_pts[inds[:, i], 0], self._layout0[i,1,None] + self._delta_pts[inds[:, i], 1], s=2, c='blue', alpha=0.3,label='Candidates')
            plt.scatter(self._layout0[i,0] + self._delta_pts[inds[0, i], 0], self._layout0[i,1] + self._delta_pts[inds[0, i], 1], s=2, c='red', label='Selected')
            plt.scatter(self._layout0[:,0], self._layout0[:,1], s=2, c='black', label='Original')
            plt.gca().add_patch(plt.Circle((self._layout0[i,0] + self._delta_pts[inds[0, i], 0], self._layout0[i,1] + self._delta_pts[inds[0, i], 1]), min_dist, color='black', fill=False, alpha=0.3))
            """

        """
        plt.title(f'Candidate positions')
        plt.xlabel('X offset')
        plt.ylabel('Y offset')
        plt.axis('equal')
        #plt.legend()
        plt.savefig(f'candidate_positions.png')
        plt.close('all')
        """

        super().update_problem_population(vars_int, vars_float)
