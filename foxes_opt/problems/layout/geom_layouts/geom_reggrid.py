from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from foxes.config import config
from foxes.utils.geom2d import AreaGeometry
from iwopy import Problem
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class GeomRegGrid(Problem):
    """
    A regular grid within a boundary geometry.

    This optimization problem does not involve
    wind farms.

    """

    def __init__(
        self,
        boundary: AreaGeometry,
        n_turbines: int,
        min_dist: float,
        max_dist: float | None = None,
        D: float | None = None,
    ) -> None:
        """
        Parameters
        ----------
        boundary
            The boundary geometry
        n_turbines
            The number of turbines in the layout
        min_dist
            The minimal distance between points
        max_dist
            The maximal distance between points
        D
            The diameter of circle fully within boundary

        """
        super().__init__(name="geom_reg_grid")

        self.boundary = boundary
        self.n_turbines = n_turbines
        self.min_dist = float(min_dist)
        self.max_dist: float | None = (
            float(max_dist) if max_dist is not None else max_dist
        )
        self.D = D

        self._SX = "sx"
        self._SY = "sy"
        self._DX = "dx"
        self._DY = "dy"
        self._ALPHA = "alpha"

    def initialize(self, verbosity: int = 1) -> None:
        """
        Initialize the object.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent

        """
        super().initialize(verbosity)

        pmin = self.boundary.p_min()
        pmax = self.boundary.p_max()
        self._pc = 0.5 * (pmin + pmax)
        self._diag: float = float(np.linalg.norm(pmax - pmin))
        self.max_dist = self._diag if self.max_dist is None else self.max_dist
        self._nrow: int = (
            int(np.maximum(self._diag / self.min_dist, np.sqrt(self.n_turbines) + 0.5))
            + 3
        )

        if verbosity > 0:
            print("Grid data:")
            print(f"  pmin        = {pmin}")
            print(f"  pmax        = {pmax}")
            print(f"  min dist    = {self.min_dist}")
            print(f"  max dist    = {self.max_dist}")
            print(f"  n row max   = {self._nrow}")
            print(f"  n max       = {self._nrow**2}")

        self.apply_individual(self.initial_values_int(), self.initial_values_float())

    def var_names_float(self) -> list[str]:
        """
        The names of float variables.

        Returns
        -------
        names
            The names of the float variables

        """
        return [self._SX, self._SY, self._DX, self._DY, self._ALPHA]

    def initial_values_float(self) -> np.ndarray:
        """
        The initial values of the float variables.

        Returns
        -------
        values
            Initial float values, shape: (n_vars_float,)

        """
        vals = np.zeros(5, dtype=config.dtype_double)
        vals[2:4] = self.min_dist
        return vals

    def min_values_float(self) -> np.ndarray:
        """
        The minimal values of the float variables.

        Use -numpy.inf for unbounded.

        Returns
        -------
        values
            Minimal float values, shape: (n_vars_float,)

        """
        vals = np.zeros(5, dtype=config.dtype_double)
        vals[:2] = -0.5
        vals[2:4] = self.min_dist
        return vals

    def max_values_float(self) -> np.ndarray:
        """
        The maximal values of the float variables.

        Use numpy.inf for unbounded.

        Returns
        -------
        values
            Maximal float values, shape: (n_vars_float,)

        """
        vals = np.zeros(5, dtype=config.dtype_double)
        vals[:2] = 0.5
        vals[2:4] = self.max_dist
        vals[4] = 90.0
        return vals

    def layout_derivative(
        self, vars_int: np.ndarray, vars_float: np.ndarray, var: int
    ) -> np.ndarray:
        """
        Calculate the layout derivative for one float variable.

        Parameters
        ----------
        vars_int
            The integer variable values
        vars_float
            The float variable values
        var
            The float variable index

        Returns
        -------
        derivative
            Point-coordinate derivatives, shape: (n_turbines, 2)

        """
        if var < 0 or var >= len(vars_float):
            raise IndexError(f"Float variable index {var} out of range")

        sx, sy, dx, dy, alpha = vars_float
        angle = np.deg2rad(alpha)
        axis_x = np.array([np.cos(angle), np.sin(angle)])
        axis_y = np.array([-np.sin(angle), np.cos(angle)])
        indices = np.arange(self._nrow) - (self._nrow - 1) / 2
        row = indices[:, None] + sx
        column = indices[None, :] + sy
        shape = (self._nrow, self._nrow, 2)
        if var == 0:
            derivative = np.broadcast_to(dx * axis_x, shape)
        elif var == 1:
            derivative = np.broadcast_to(dy * axis_y, shape)
        elif var == 2:
            derivative = row[:, :, None] * axis_x
        elif var == 3:
            derivative = column[:, :, None] * axis_y
        else:
            derivative = np.deg2rad(1.0) * (
                row[:, :, None] * dx * axis_y - column[:, :, None] * dy * axis_x
            )
        derivative = np.broadcast_to(derivative, shape).reshape(-1, 2)

        points, valid = self._all_grid_points(vars_float)
        selected = np.flatnonzero(valid)
        if len(selected) < self.n_turbines:
            selected = np.append(
                selected, np.flatnonzero(~valid)[: self.n_turbines - len(selected)]
            )
        return derivative[selected[: self.n_turbines]]

    def _all_grid_points(self, vars_float: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create all grid points and their boundary-validity flags."""
        sx, sy, dx, dy, alpha = vars_float
        angle = np.deg2rad(alpha)
        axis_x = np.array([np.cos(angle), np.sin(angle)])
        axis_y = np.array([-np.sin(angle), np.cos(angle)])
        indices = np.arange(self._nrow) - (self._nrow - 1) / 2
        points = (
            self._pc[None, None, :]
            + (indices[:, None, None] + sx) * dx * axis_x
            + (indices[None, :, None] + sy) * dy * axis_y
        ).reshape(-1, 2)
        valid = self.boundary.points_inside(points)
        if self.D is not None:
            valid &= self.boundary.points_distance(points) >= self.D / 2
        return points, valid

    def apply_individual(
        self, vars_int: np.ndarray, vars_float: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply new variables to the problem.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_vars_int,)
        vars_float
            The float variable values, shape: (n_vars_float,)

        Returns
        -------
        problem_results
            The results of the variable application
            to the problem

        """
        pts, valid = self._all_grid_points(vars_float)

        nvl = np.sum(valid)
        if nvl >= self.n_turbines:
            return pts[valid][: self.n_turbines], np.ones(self.n_turbines, dtype=bool)
        else:
            qts: np.ndarray[tuple[int, ...], np.dtype[Any]] = np.append(
                pts[valid], pts[~valid][: (self.n_turbines - nvl)], axis=0
            )
            vld: np.ndarray[tuple[int], np.dtype[Any]] = np.zeros(
                self.n_turbines, dtype=bool
            )
            vld[:nvl] = True
            return qts, vld

    def apply_population(
        self, vars_int: np.ndarray, vars_float: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply new variables to the problem,
        for a whole population.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float
            The float variable values, shape: (n_pop, n_vars_float)

        Returns
        -------
        problem_results
            The results of the variable application
            to the problem

        """
        n_pop = vars_float.shape[0]
        sx = vars_float[:, 0]
        sy = vars_float[:, 1]
        dx = vars_float[:, 2]
        dy = vars_float[:, 3]
        alpha = vars_float[:, 4]

        a = np.deg2rad(alpha)
        nax: np.ndarray[tuple[int, ...], np.dtype[Any]] = np.stack(
            [np.cos(a), np.sin(a)], axis=-1
        )
        nay: np.ndarray[tuple[int, ...], np.dtype[Any]] = np.stack(
            [-np.sin(a), np.cos(a)], axis=-1
        )

        pts = (
            self._pc[None, None, None, :]
            + (
                np.arange(self._nrow)[None, :, None, None]
                - (self._nrow - 1) / 2
                + sx[:, None, None, None]
            )
            * dx[:, None, None, None]
            * nax[:, None, None, :]
            + (
                np.arange(self._nrow)[None, None, :, None]
                - (self._nrow - 1) / 2
                + sy[:, None, None, None]
            )
            * dy[:, None, None, None]
            * nay[:, None, None, :]
        )
        pts = pts.reshape(n_pop * self._nrow**2, 2)

        if self.D is None:
            valid = self.boundary.points_inside(pts)
        else:
            valid = self.boundary.points_inside(pts) & (
                self.boundary.points_distance(pts) >= self.D / 2
            )
        valid = valid.reshape(n_pop, self._nrow**2)
        pts = pts.reshape(n_pop, self._nrow**2, 2)

        nvl = np.sum(valid, axis=1)
        qts = np.zeros((n_pop, self.n_turbines, 2), dtype=config.dtype_double)
        vld: np.ndarray[tuple[int, int], np.dtype[Any]] = np.zeros(
            (n_pop, self.n_turbines), dtype=bool
        )
        for pi in range(n_pop):
            if nvl[pi] >= self.n_turbines:
                qts[pi] = pts[pi, valid[pi]][: self.n_turbines]
                vld[pi] = np.ones(self.n_turbines, dtype=bool)
            else:
                qts[pi] = np.append(
                    pts[pi, valid[pi]],
                    pts[pi, ~valid[pi]][: (self.n_turbines - nvl[pi])],
                    axis=0,
                )
                vld[pi, : nvl[pi]] = True

        return qts, vld

    def get_fig(
        self,
        xy: np.ndarray | None = None,
        valid: np.ndarray | None = None,
        ax: "Axes | None" = None,
        title: str | None = None,
        true_circle: bool = True,
        **bargs: Any,
    ) -> "Axes":
        """
        Return plotly figure axis.

        Parameters
        ----------
        xy
            The xy coordinate array, shape: (n_points, 2)
        valid
            Boolean array of validity, shape: (n_points,)
        ax
            The figure axis
        title
            The figure title
        true_circle
            Draw points as circles with diameter self.D
        bars
            The boundary plot arguments

        Returns
        -------
        ax
            The figure axis

        """
        if ax is None:
            __, ax = plt.subplots()

        hbargs: dict[str, str] = {"fill_mode": "inside_lightgray"}
        hbargs.update(bargs)
        self.boundary.add_to_figure(ax, **hbargs)

        if xy is not None:
            if valid is not None:
                xy = xy[valid]
            if not true_circle or self.D is None:
                ax.scatter(xy[:, 0], xy[:, 1], color="orange")
            else:
                for x, y in xy:
                    ax.add_patch(
                        plt.Circle((x, y), self.D / 2, color="blue", fill=True)
                    )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        if title is None:
            if xy is None:
                title = "Optimization area"
            else:
                lxy: int = len(xy) if xy is not None else 0
                dists: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = cdist(
                    xy, xy
                )
                np.fill_diagonal(dists, 1e20)
                title = f"N = {lxy}, min_dist = {np.min(dists):.1f} m"
        ax.set_title(title)

        return ax
